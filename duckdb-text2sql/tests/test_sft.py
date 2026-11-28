"""Tests for the fine-tuning code that need no ML libraries. Model tests: test_sft_model.py."""

import re

from text2sql.sft import LORA_TARGETS, length_grouped_batches, target_text

# Linear layer names of Qwen/Qwen3.5-0.8B (layer index replaced by N), from the model config.
REAL_LINEAR_NAMES = [
    "lm_head",
    *[f"model.language_model.layers.N.linear_attn.{n}"
      for n in ("in_proj_a", "in_proj_b", "in_proj_qkv", "in_proj_z", "out_proj")],
    *[f"model.language_model.layers.N.mlp.{n}" for n in ("down_proj", "gate_proj", "up_proj")],
    *[f"model.language_model.layers.N.self_attn.{n}" for n in ("k_proj", "o_proj", "q_proj", "v_proj")],
    "model.visual.blocks.N.attn.proj", "model.visual.blocks.N.attn.qkv",
    "model.visual.blocks.N.mlp.linear_fc1", "model.visual.merger.linear_fc2",
]


def test_lora_targets_language_model_only():
    hit = {n for n in REAL_LINEAR_NAMES if re.fullmatch(LORA_TARGETS, n.replace(".N.", ".3."))}
    assert not {n for n in hit if "visual" in n or n == "lm_head"}
    assert {n.rsplit(".", 1)[1] for n in hit} == {
        "in_proj_qkv", "in_proj_z", "out_proj", "down_proj", "gate_proj", "up_proj",
        "q_proj", "k_proj", "v_proj", "o_proj"}
    # the text-only model has no "language_model" prefix; it must match too
    assert re.fullmatch(LORA_TARGETS, "model.layers.0.self_attn.q_proj")


def test_length_grouped_batches_use_every_example_once():
    enc = [{"input_ids": [0] * n, "id": i} for i, n in enumerate(range(1, 300))]
    batches = length_grouped_batches(enc, 4, seed=1)
    assert sorted(e["id"] for b in batches for e in b) == list(range(299))
    assert batches == length_grouped_batches(enc, 4, seed=1)


def test_target_text():
    assert target_text("SELECT 1") == "```sql\nSELECT 1\n```"
