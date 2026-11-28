"""LoRA fine-tuning on the execution-verified data, and execution-based evaluation of the result.

Meant for one GPU (a Kaggle T4). Trains on Spider-DuckDB plus the DuckDB-specific examples, then
evaluates the base model and the fine-tuned model under identical conditions (same prompt, greedy
decoding, same precision) on three sets:
  baseline    the 50 hand-written questions in evals/baseline_v0.jsonl
  synth_val   DuckDB-specific validation examples on the held-out repos database
  spider_val  a fixed sample of Spider-DuckDB validation (databases unseen in training)

Run:
    python -m text2sql.sft --smoke     # a few steps and a few eval questions, to check the setup
    python -m text2sql.sft             # full run
    python -m text2sql.sft --push      # also upload the adapter and results to the Hub (needs HF_TOKEN)

torch, transformers and peft are imported inside functions, so the rest of the package does not
need them (install with `pip install -e ".[train]"`).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb

from .evaluate import DEFAULT_EVAL, load_eval, score_sql, summarize
from .prompt import SYSTEM, extract_sql, user_prompt
from .schemas import build_all as build_eval_dbs, schema_ddl

ROOT = Path(__file__).resolve().parents[2]
SPIDER_REPO = "bhargavis/spider-duckdb"
SYNTH_DIR = ROOT / "data" / "processed" / "duckdb_synth"
WORK = ROOT / "data" / "sft"
RESULTS = ROOT / "results"

# LoRA goes on the language model only: attention, linear-attention (Gated DeltaNet) and MLP
# projections. The vision encoder is not loaded at all (AutoModelForCausalLM gives the text model).
LORA_TARGETS = (r".*layers\.\d+\.(self_attn\.[qkvo]_proj|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)"
                r"|mlp\.(gate|up|down)_proj)")


@dataclass
class Config:
    model_id: str = "Qwen/Qwen3.5-0.8B"
    synth_repeat: int = 4        # the DuckDB-specific set is small; repeat it in the training mix
    epochs: float = 2.0
    lr: float = 2e-4
    warmup_ratio: float = 0.03
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    max_len: int = 2048          # prompt + answer tokens; longer examples are skipped
    batch_size: int = 4
    grad_accum: int = 4
    fp16: bool = True            # T4 has no bf16; fp32 master weights with fp16 autocast
    seed: int = 42
    spider_val_sample: int = 300
    eval_batch_size: int = 8
    max_new_tokens: int = 256
    smoke: bool = False
    hub_repo: str = "bhargavis/duckdb-text2sql-qwen3.5-0.8b-lora"


# --------------------------------------------------------------------------- prompt and tokens
def messages(schema: str, question: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_prompt(schema, question)}]


def target_text(sql: str) -> str:
    return f"```sql\n{sql}\n```"


def prompt_text(tok, schema: str, question: str) -> str:
    # same chat template as at inference; enable_thinking=False adds the empty <think></think> block
    return tok.apply_chat_template(messages(schema, question), tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)


def encode(tok, schema: str, question: str, sql: str, max_len: int) -> dict | None:
    """Token ids with labels only on the answer. Prompt and answer are tokenized separately,
    exactly as at inference, where the prompt is tokenized alone and the model continues it."""
    p = tok(prompt_text(tok, schema, question), add_special_tokens=False)["input_ids"]
    t = tok(target_text(sql) + tok.eos_token, add_special_tokens=False)["input_ids"]
    if len(p) + len(t) > max_len:
        return None
    return {"input_ids": p + t, "labels": [-100] * len(p) + t}


# --------------------------------------------------------------------------- data
def _read_parquet(path: Path) -> list[dict]:
    with duckdb.connect() as con:
        rel = con.execute(f"SELECT * FROM read_parquet('{path}')")
        cols = [d[0] for d in rel.description]
        return [dict(zip(cols, r)) for r in rel.fetchall()]


def spider_files(work: Path = WORK) -> dict[str, Path]:
    """Download Spider-DuckDB from the Hub (parquet splits and databases)."""
    from huggingface_hub import hf_hub_download

    files = {s: Path(hf_hub_download(SPIDER_REPO, f"data/{s}.parquet", repo_type="dataset"))
             for s in ("train", "validation")}
    db_dir = work / "spider_databases"
    if not db_dir.exists():
        z = hf_hub_download(SPIDER_REPO, "databases.zip", repo_type="dataset")
        with zipfile.ZipFile(z) as f:
            f.extractall(work)
        (work / "databases").rename(db_dir)
    files["databases"] = db_dir
    return files


def synth_files() -> Path:
    if not (SYNTH_DIR / "train.parquet").exists():
        from .synth import build

        build()
    return SYNTH_DIR


def training_rows(cfg: Config, spider_train: Path, synth_dir: Path) -> list[dict]:
    spider = [{"schema": r["schema"], "question": r["question"], "sql": r["sql"], "source": "spider"}
              for r in _read_parquet(spider_train)]
    synth = [{"schema": r["schema"], "question": r["question"], "sql": r["sql"], "source": "duckdb_synth"}
             for r in _read_parquet(synth_dir / "train.parquet")]
    rows = spider + synth * cfg.synth_repeat
    random.Random(cfg.seed).shuffle(rows)
    return rows


def eval_sets(cfg: Config, spider: dict[str, Path], synth_dir: Path, work: Path = WORK) -> dict[str, list[dict]]:
    """Each example: id, db (name), db_path, schema, question, gold_sql, ordered, difficulty."""
    eval_db_dir = work / "eval_dbs"
    paths = build_eval_dbs(eval_db_dir)
    schemas = {db: schema_ddl(p) for db, p in paths.items()}
    baseline = [{**ex, "db_path": str(paths[ex["db"]]), "schema": schemas[ex["db"]]}
                for ex in load_eval(DEFAULT_EVAL)]
    synth = [{"id": r["id"], "db": r["db_id"], "db_path": str(synth_dir / "databases" / f"{r['db_id']}.duckdb"),
              "schema": r["schema"], "question": r["question"], "gold_sql": r["sql"], "ordered": r["ordered"],
              "difficulty": r["family"]} for r in _read_parquet(synth_dir / "validation.parquet")]
    spider_val = _read_parquet(spider["validation"])
    spider_val = random.Random(cfg.seed).sample(spider_val, min(cfg.spider_val_sample, len(spider_val)))
    spider_val = [{"id": r["id"], "db": r["db_id"], "db_path": str(spider["databases"] / f"{r['db_id']}.duckdb"),
                   "schema": r["schema"], "question": r["question"], "gold_sql": r["sql"],
                   "ordered": r["ordered"], "difficulty": "spider"} for r in spider_val]
    sets = {"baseline": baseline, "synth_val": synth, "spider_val": spider_val}
    if cfg.smoke:
        sets = {k: v[:8] for k, v in sets.items()}
    return sets


# --------------------------------------------------------------------------- model
def load_model(cfg: Config):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    model, info = AutoModelForCausalLM.from_pretrained(cfg.model_id, dtype=torch.float32,
                                                       output_loading_info=True)
    # a silent random init of missing weights would make every result meaningless
    missing = [k for k in info["missing_keys"] if "visual" not in k]
    if missing:
        raise RuntimeError(f"weights not loaded for {len(missing)} tensors, e.g. {missing[:3]}")
    if torch.cuda.is_available():
        model = model.cuda()
    return model, tok


def add_lora(model, cfg: Config):
    from peft import LoraConfig, get_peft_model

    lcfg = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                      target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if n_train == 0:
        raise RuntimeError("LoRA matched no modules; check LORA_TARGETS against the model's layer names")
    return model


def answer_loss(model, batch):
    """Cross-entropy on answer tokens only. The vocabulary has 248k entries, so instead of building
    logits for every position the LM head runs only where there is a label (a few dozen tokens per
    example rather than the whole schema). This is the main memory saving."""
    import torch.nn.functional as F

    core = model.get_base_model() if hasattr(model, "get_base_model") else model
    hidden = core.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
    labels = batch["labels"][:, 1:]
    mask = labels != -100
    logits = core.lm_head(hidden[:, :-1][mask]).float()
    return F.cross_entropy(logits, labels[mask])


def collate(examples: list[dict], pad_id: int):
    import torch

    n = max(len(e["input_ids"]) for e in examples)
    ids = torch.full((len(examples), n), pad_id, dtype=torch.long)
    labels = torch.full((len(examples), n), -100, dtype=torch.long)
    att = torch.zeros((len(examples), n), dtype=torch.long)
    for i, e in enumerate(examples):  # right padding: padded positions come after the real tokens
        k = len(e["input_ids"])
        ids[i, :k] = torch.tensor(e["input_ids"])
        labels[i, :k] = torch.tensor(e["labels"])
        att[i, :k] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": att}


def length_grouped_batches(encoded: list[dict], batch_size: int, seed: int) -> list[list[dict]]:
    """Shuffle, then sort within windows of 32 batches so each batch has similar lengths
    (less padding) while the order across windows stays random."""
    rng = random.Random(seed)
    items = encoded[:]
    rng.shuffle(items)
    window = batch_size * 32
    batches = []
    for i in range(0, len(items), window):
        chunk = sorted(items[i:i + window], key=lambda e: len(e["input_ids"]))
        batches += [chunk[j:j + batch_size] for j in range(0, len(chunk), batch_size)]
    rng.shuffle(batches)
    return batches


def train(model, tok, encoded: list[dict], cfg: Config, log=print) -> list[dict]:
    import torch

    device = next(model.parameters()).device
    use_amp = cfg.fp16 and device.type == "cuda"
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    epochs = math.ceil(cfg.epochs)
    schedule = []
    for ep in range(epochs):
        batches = length_grouped_batches(encoded, cfg.batch_size, cfg.seed + ep)
        if ep == epochs - 1 and cfg.epochs < epochs:  # fractional last epoch
            batches = batches[: int(len(batches) * (cfg.epochs - ep))]
        schedule += batches
    if cfg.smoke:
        schedule = schedule[: 10 * cfg.grad_accum]
    total_steps = max(1, len(schedule) // cfg.grad_accum)
    warmup = max(1, int(total_steps * cfg.warmup_ratio))

    def lr_at(step: int) -> float:  # linear warmup, then cosine decay to 10%
        if step < warmup:
            return cfg.lr * (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return cfg.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))

    history, running, since_log, t0, step = [], 0.0, 0, time.time(), 0
    opt.zero_grad(set_to_none=True)
    for i, batch in enumerate(schedule):
        b = {k: v.to(device) for k, v in collate(batch, tok.pad_token_id).items()}
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            loss = answer_loss(model, b) / cfg.grad_accum
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at batch {i}; try --no-fp16")
        scaler.scale(loss).backward()
        running += loss.item()
        if (i + 1) % cfg.grad_accum == 0:
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            step += 1
            since_log += 1
            if step % 10 == 0 or step == total_steps:
                # running holds the sum of per-step mean losses since the last log line
                history.append({"step": step, "loss": round(running / since_log, 4),
                                "lr": lr_at(step - 1), "elapsed_s": round(time.time() - t0)})
                log(f"step {step}/{total_steps}  loss {history[-1]['loss']:.4f}  "
                    f"{history[-1]['elapsed_s']}s")
                running, since_log = 0.0, 0
            if step >= total_steps:
                break
    model.eval()
    return history


# --------------------------------------------------------------------------- generation and scoring
def generate(model, tok, prompts: list[str], batch_size: int, max_new_tokens: int) -> list[str]:
    """Greedy decoding with left padding, longest prompts first so memory peaks early."""
    import torch

    tok.padding_side = "left"
    device = next(model.parameters()).device
    order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
    out = [""] * len(prompts)
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16,
                                             enabled=device.type == "cuda"):
            gen = model.generate(**enc, do_sample=False, max_new_tokens=max_new_tokens,
                                 pad_token_id=tok.pad_token_id)
        for j, i in enumerate(idx):
            out[i] = tok.decode(gen[j, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return out


def batching_is_safe(model, tok, prompts: list[str], max_new_tokens: int) -> bool:
    """Left padding must not change outputs (the model mixes linear and full attention).
    Compare batched and one-at-a-time generation on a few prompts."""
    sample = prompts[:6]
    return generate(model, tok, sample, len(sample), max_new_tokens) == generate(model, tok, sample, 1,
                                                                                 max_new_tokens)


def evaluate_model(model, tok, sets: dict[str, list[dict]], cfg: Config, label: str, log=print) -> dict:
    all_prompts = [prompt_text(tok, ex["schema"], ex["question"]) for exs in sets.values() for ex in exs]
    bs = cfg.eval_batch_size if batching_is_safe(model, tok, all_prompts, 64) else 1
    log(f"[{label}] eval batch size {bs}")
    summary, records = {}, []
    for name, exs in sets.items():
        t0 = time.time()
        raw = generate(model, tok, [prompt_text(tok, ex["schema"], ex["question"]) for ex in exs], bs,
                       cfg.max_new_tokens)
        recs = []
        for ex, text in zip(exs, raw):
            sql = extract_sql(text)
            rec = {"set": name, "id": ex["id"], "db": ex["db"], "difficulty": ex["difficulty"], "pred_sql": sql}
            rec.update(score_sql(ex, Path(ex["db_path"]), sql))
            recs.append(rec)
        summary[name] = summarize(recs)
        summary[name]["seconds"] = round(time.time() - t0)
        log(f"[{label}] {name}: {summary[name]['execution_accuracy']}% of {len(recs)}")
        records += recs
    return {"summary": summary, "records": records}


# --------------------------------------------------------------------------- main
def run(cfg: Config, push: bool = False) -> dict:
    import torch

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    WORK.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    spider = spider_files()
    synth_dir = synth_files()
    sets = eval_sets(cfg, spider, synth_dir)

    model, tok = load_model(cfg)
    rows = training_rows(cfg, spider["train"], synth_dir)
    encoded = [e for r in rows if (e := encode(tok, r["schema"], r["question"], r["sql"], cfg.max_len))]
    print(f"training examples: {len(encoded)} kept, {len(rows) - len(encoded)} over {cfg.max_len} tokens")

    base = evaluate_model(model, tok, sets, cfg, "base")
    model = add_lora(model, cfg)
    model.print_trainable_parameters()
    history = train(model, tok, encoded, cfg)
    tuned = evaluate_model(model, tok, sets, cfg, "fine-tuned")

    run_id = time.strftime("%Y%m%d-%H%M%S") + ("-smoke" if cfg.smoke else "")
    out_dir = WORK / f"run-{run_id}"
    model.save_pretrained(out_dir / "adapter")
    tok.save_pretrained(out_dir / "adapter")
    result = {"run_id": run_id, "config": asdict(cfg), "train_examples": len(encoded),
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
              "history": history, "base": base["summary"], "fine_tuned": tuned["summary"]}
    (RESULTS / f"sft_{run_id}.json").write_text(json.dumps(result, indent=2))
    with open(RESULTS / f"sft_{run_id}_predictions.jsonl", "w") as f:
        for which, res in (("base", base), ("fine_tuned", tuned)):
            for rec in res["records"]:
                f.write(json.dumps({"model": which, **rec}) + "\n")
    print_comparison(result)
    if push and not cfg.smoke:
        model.push_to_hub(cfg.hub_repo, private=True)
        tok.push_to_hub(cfg.hub_repo, private=True)
        from huggingface_hub import upload_file

        for p in (RESULTS / f"sft_{run_id}.json", RESULTS / f"sft_{run_id}_predictions.jsonl"):
            upload_file(path_or_fileobj=str(p), path_in_repo=f"results/{p.name}", repo_id=cfg.hub_repo)
        print(f"pushed to https://huggingface.co/{cfg.hub_repo} (private)")
    return result


def print_comparison(result: dict) -> None:
    print(f"\n{'set':<12}{'base':>8}{'fine-tuned':>12}")
    for name in result["base"]:
        b, t = result["base"][name]["execution_accuracy"], result["fine_tuned"][name]["execution_accuracy"]
        print(f"{name:<12}{b:>7}%{t:>11}%")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="10 optimizer steps and 8 eval questions per set")
    ap.add_argument("--push", action="store_true", help="upload adapter and results to the Hub (private)")
    ap.add_argument("--model-id", default=Config.model_id)
    ap.add_argument("--epochs", type=float, default=Config.epochs)
    ap.add_argument("--lr", type=float, default=Config.lr)
    ap.add_argument("--synth-repeat", type=int, default=Config.synth_repeat)
    ap.add_argument("--no-fp16", action="store_true", help="train in fp32 (slower; use if loss is not finite)")
    ap.add_argument("--hub-repo", default=Config.hub_repo)
    args = ap.parse_args()
    cfg = Config(model_id=args.model_id, epochs=args.epochs, lr=args.lr, synth_repeat=args.synth_repeat,
                 fp16=not args.no_fp16, smoke=args.smoke, hub_repo=args.hub_repo)
    run(cfg, push=args.push)


if __name__ == "__main__":
    main()
