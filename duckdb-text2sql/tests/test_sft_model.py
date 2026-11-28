"""Model tests for the fine-tuning code.

Need torch, transformers and peft (pip install -r requirements-train.txt); skipped otherwise.
They use a tiny randomly initialised Qwen3.5 text model with the real tokenizer, so they run on
CPU in seconds without downloading model weights:

    uv run --with torch==2.11.0 --with transformers==5.23.0 --with peft==0.21.3 \\
        --with accelerate pytest tests/test_sft_model.py
"""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")

from text2sql import sft  # noqa: E402
from text2sql.sft import Config  # noqa: E402

SCHEMA = "CREATE TABLE singer(singer_id INTEGER, name VARCHAR, genres VARCHAR[]);"


@pytest.fixture(scope="module")
def tok():
    try:
        return transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
    except Exception as e:  # offline
        pytest.skip(f"tokenizer not available: {e}")


@pytest.fixture
def tiny(tok):
    torch.manual_seed(0)
    cfg = transformers.AutoConfig.for_model(
        "qwen3_5_text", vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"], num_attention_heads=2,
        num_key_value_heads=1, head_dim=16, linear_num_key_heads=2, linear_num_value_heads=2,
        linear_key_head_dim=8, linear_value_head_dim=8, tie_word_embeddings=True,
        # large init: with the default 0.02 the frozen tied output layer caps logits below 1,
        # so a random tiny model could not fit anything and the test would prove nothing
        initializer_range=0.5,
        eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    model = transformers.AutoModelForCausalLM.from_config(cfg)
    model.eval()
    return model


def test_encode_puts_labels_on_answer_only(tok):
    e = sft.encode(tok, SCHEMA, "How many singers?", "SELECT count(*) FROM singer", 4096)
    n_prompt = sum(1 for x in e["labels"] if x == -100)
    prompt = tok.decode(e["input_ids"][:n_prompt])
    assert prompt == sft.prompt_text(tok, SCHEMA, "How many singers?")
    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    answer = tok.decode(e["input_ids"][n_prompt:])
    assert answer == "```sql\nSELECT count(*) FROM singer\n```<|im_end|>"
    assert e["labels"][n_prompt:] == e["input_ids"][n_prompt:]
    assert sft.encode(tok, SCHEMA, "q", "SELECT 1", 10) is None


def test_answer_loss_equals_full_loss(tok, tiny):
    batch = sft.collate([sft.encode(tok, SCHEMA, "How many singers?", "SELECT count(*) FROM singer", 4096),
                         sft.encode(tok, SCHEMA, "List names.", "SELECT name FROM singer", 4096)],
                        tok.pad_token_id)
    full = tiny(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"]).loss
    assert torch.allclose(sft.answer_loss(tiny, batch), full, atol=1e-5)


def test_lora_attaches_and_training_reduces_loss(tok, tiny):
    cfg = Config(lr=1e-2, epochs=40, batch_size=2, grad_accum=1, lora_dropout=0.0, warmup_ratio=0.0)
    model = sft.add_lora(tiny, cfg)
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    assert any("linear_attn.in_proj_qkv" in n for n in names) and any("self_attn.q_proj" in n for n in names)
    enc = [sft.encode(tok, SCHEMA, q, s, 4096) for q, s in
           [("How many singers?", "SELECT count(*) FROM singer"),
            ("Singers with genre 'jazz'?", "SELECT name FROM singer WHERE list_contains(genres, 'jazz')")]]
    history = sft.train(model, tok, enc, cfg, log=lambda *_: None)
    assert history[-1]["loss"] < history[0]["loss"] * 0.5
    # generation goes through the LoRA-wrapped model
    outs = sft.generate(model, tok, [sft.prompt_text(tok, SCHEMA, "How many singers?")], 1, 8)
    assert isinstance(outs[0], str)


def test_batched_generation_matches_single(tok, tiny):
    prompts = [sft.prompt_text(tok, SCHEMA, q) for q in
               ["How many singers?", "What are the names of all singers in the table, sorted?", "Count genres."]]
    assert sft.generate(tiny, tok, prompts, 3, 12) == sft.generate(tiny, tok, prompts, 1, 12)
