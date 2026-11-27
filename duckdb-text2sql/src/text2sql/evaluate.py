"""Execution-accuracy evaluation.

Examples:
  python -m text2sql.evaluate --backend gold
  python -m text2sql.evaluate --backend openai --model qwen2.5-coder-1.5b \
      --base-url http://localhost:8080/v1
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from .compare import results_match, run_query
from .models import Backend, GoldBackend, OpenAICompatBackend
from .prompt import extract_sql
from .schemas import DEFAULT_DIR, build_all, schema_ddl

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL = ROOT / "evals" / "baseline_v0.jsonl"


def load_eval(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def evaluate(backend: Backend, examples: list[dict], db_dir: Path = DEFAULT_DIR) -> list[dict]:
    schemas: dict[str, str] = {}
    records = []
    for ex in examples:
        db = db_dir / f"{ex['db']}.duckdb"
        schema = schemas.setdefault(ex["db"], schema_ddl(db))
        rec = {"id": ex["id"], "db": ex["db"], "difficulty": ex["difficulty"]}
        t0 = time.perf_counter()
        try:
            raw = backend.generate(schema, ex["question"], ex["gold_sql"])
        except Exception as e:  # network / server errors
            rec.update(status="gen_error", error=str(e)[:300], correct=False)
            records.append(rec)
            continue
        rec["latency_s"] = round(time.perf_counter() - t0, 2)
        rec["pred_sql"] = extract_sql(raw)
        rec.update(score_sql(ex, db, rec["pred_sql"]))
        records.append(rec)
    return records


def score_sql(ex: dict, db: Path, sql: str) -> dict:
    """Execute predicted SQL and compare against the gold result."""
    gold_rows = run_query(db, ex["gold_sql"])
    try:
        pred_rows = run_query(db, sql)
    except Exception as e:
        return {"status": "exec_error", "error": str(e).splitlines()[0][:300], "correct": False}
    ok = results_match(gold_rows, pred_rows, ex.get("ordered", False))
    return {"status": "correct" if ok else "wrong_result", "error": "", "correct": ok}


def rescore(records: list[dict], examples: list[dict], db_dir: Path = DEFAULT_DIR) -> list[dict]:
    """Re-score saved predictions (e.g. after changing comparison rules) without calling a model."""
    by_id = {ex["id"]: ex for ex in examples}
    out = []
    for rec in records:
        rec = dict(rec)
        if "pred_sql" in rec:
            ex = by_id[rec["id"]]
            rec.update(score_sql(ex, db_dir / f"{ex['db']}.duckdb", rec["pred_sql"]))
        out.append(rec)
    return out


def summarize(records: list[dict]) -> dict:
    def acc(rs):
        return round(100 * sum(r["correct"] for r in rs) / len(rs), 1) if rs else 0.0

    groups: dict[str, dict[str, list]] = {"difficulty": defaultdict(list), "db": defaultdict(list),
                                          "status": defaultdict(list)}
    for r in records:
        for key in groups:
            groups[key][r[key]].append(r)
    return {
        "n": len(records),
        "execution_accuracy": acc(records),
        "by_difficulty": {k: acc(v) for k, v in sorted(groups["difficulty"].items())},
        "by_db": {k: acc(v) for k, v in sorted(groups["db"].items())},
        "status_counts": {k: len(v) for k, v in sorted(groups["status"].items())},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["gold", "openai"], default="gold")
    ap.add_argument("--model", help="model name for the openai backend")
    ap.add_argument("--base-url", default="http://localhost:8080/v1")
    ap.add_argument("--api-key-env", help="env var holding the API key, e.g. HF_TOKEN")
    ap.add_argument("--eval-file", type=Path, default=DEFAULT_EVAL)
    ap.add_argument("--limit", type=int, help="only run the first N examples")
    ap.add_argument("--rebuild-dbs", action="store_true")
    ap.add_argument("--rescore", type=Path, metavar="RESULTS_JSONL",
                    help="re-score saved predictions instead of calling a model")
    args = ap.parse_args()

    if args.rebuild_dbs or not (DEFAULT_DIR / "web.duckdb").exists():
        build_all()
    examples = load_eval(args.eval_file)
    if args.rescore:
        records = rescore(load_eval(args.rescore), examples)
        stem = args.rescore.name.removesuffix(".jsonl")
    else:
        backend: Backend = (GoldBackend() if args.backend == "gold"
                            else OpenAICompatBackend(args.model, args.base_url, args.api_key_env))
        records = evaluate(backend, examples[: args.limit])
        stem = f"{backend.name.replace('/', '_')}_{time.strftime('%Y%m%d-%H%M%S')}"
    summary = summarize(records)

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"{stem}.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    (out_dir / f"{stem}.summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"details: results/{stem}.jsonl")


if __name__ == "__main__":
    main()
