# duckdb-text2sql

Small, **execution-verified** text-to-SQL models for [DuckDB](https://duckdb.org): ask a question in English, get DuckDB SQL that actually runs and returns the right answer, locally.

## Status

Currently working on phase 0 (baseline evaluation).

| Phase | What | Status |
|---|---|---|
| 0 | Eval harness + 50 hand-written questions, baseline off-the-shelf models | In progress |
| 1 | Synthetic dataset pipeline, verified by executing every query in DuckDB | Planned |
| 2 | Fine-tune a small model (LoRA / SFT) | Planned |
| 3 | RL with execution-based rewards | Planned |
| 4 | Hugging Face Space demo + CI/CD | Planned |

## How evaluation works

- `src/text2sql/schemas.py` builds four deterministic DuckDB databases (ecommerce, flights, hr, web analytics) covering joins, dates, self-joins, recursive hierarchies, and DuckDB `LIST` columns.
- `evals/baseline_v0.jsonl` holds 50 questions (easy / medium / hard) with reference SQL.
- A model's SQL is **executed** and its result compared to the reference result (**execution accuracy**). Column names and column order are ignored, numbers are compared to 2 decimals, and row order only matters when the question asks for an ordering.

## Quickstart

```bash
uv sync
uv run pytest                                   # eval set sanity checks
uv run python -m text2sql.evaluate --backend gold   # must be 100%

# any OpenAI-compatible server (llama.cpp llama-server, Ollama, LM Studio, vLLM, HF Inference Providers)
uv run python -m text2sql.evaluate --backend openai \
    --model <model-name> --base-url http://localhost:8080/v1
```

Results are written to `results/` (git-ignored).

## Baseline results

_Coming soon._
