# duckdb-text2sql — project notes for Claude

Goal: small, execution-verified text-to-SQL model for DuckDB, published on Hugging Face (dataset + model + Space). Phases: 0 baseline eval → 1 synthetic dataset pipeline → 2 LoRA SFT → 3 RL w/ execution rewards → 4 Space + CI/CD.

## Layout
- `src/text2sql/schemas.py` — deterministic eval DBs (seeded) → `data/dbs/*.duckdb`
- `src/text2sql/compare.py` — execution + result comparison rules (see module docstring)
- `src/text2sql/prompt.py`, `models.py` — prompt, SQL extraction, backends (gold, OpenAI-compatible)
- `src/text2sql/evaluate.py` — CLI; writes `results/` (git-ignored)
- `evals/baseline_v0.jsonl` — 50 hand-written questions; don't edit gold SQL without re-running tests

## Commands
- `uv run pytest -q 2>&1 | tail -20`
- `uv run python -m text2sql.evaluate --backend gold`

## Rules
- See ../CLAUDE.md for repo-wide rules (no commits by Claude, branch naming, machine limits).
