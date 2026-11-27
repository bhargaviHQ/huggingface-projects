# duckdb-text2sql

Small, **execution-verified** text-to-SQL models for [DuckDB](https://duckdb.org): ask a question in English, get DuckDB SQL that actually runs and returns the right answer, locally.

## Status

Phase 0 (baseline evaluation) is done. Phase 1a (Spider converted to DuckDB) is done; 1b (DuckDB-specific examples) is next.

| Phase | What | Status |
|---|---|---|
| 0 | Eval harness + 50 hand-written questions, baseline off-the-shelf models | Done |
| 1a | Spider converted to DuckDB, every query verified by execution | Done |
| 1b | DuckDB-specific examples (`LIST`/`STRUCT` columns, `QUALIFY`, dates) | Next |
| 2 | Fine-tune a small model (LoRA / SFT) | Planned |
| 3 | RL with execution-based rewards | Planned |
| 4 | Hugging Face Space demo + CI/CD | Planned |

## Training data: Spider-DuckDB

`src/text2sql/spider.py` converts [Spider](https://yale-lily.github.io/spider) (SQLite) into a DuckDB dataset and keeps an example only if the DuckDB query returns the same result as the original SQLite query on the same data. The dataset card in [`dataset_cards/spider-duckdb.md`](dataset_cards/spider-duckdb.md) describes every step.

| | train | validation |
|---|---|---|
| Spider examples | 7,000 | 1,034 |
| Kept after execution check | 6,339 (90.6%) | 919 (88.9%) |

A plain sqlglot transpile kept only 4,515 train examples (64.5%). Most of the gap came from SQLite behaviour that DuckDB rejects or treats differently, fixed in the query AST before transpiling:

- double-quoted strings (`WHERE status != "Village"`), which SQLite reads as string literals
- selecting columns that are neither grouped nor aggregated
- `LIKE`, which ignores case in SQLite but not in DuckDB (sqlglot chose not to translate this: [tobymao/sqlglot#9226](https://github.com/tobymao/sqlglot/issues/9226))
- integer division, which needs column types to transpile correctly

Each fix is still checked by execution, so a wrong rewrite cannot slip in. A bug in an early version of the `GROUP BY` fix was caught this way: the rewritten queries returned different results and were rejected.

Spider's databases are not in this repo. To rebuild:

```bash
mkdir -p data/raw && cd data/raw
curl -LO https://huggingface.co/datasets/HAL-9001/spider-databases/resolve/main/spider_data.zip
unzip -q spider_data.zip 'spider_data/database/*' spider_data/train_spider.json spider_data/dev.json
cd ../..
uv run python -m text2sql.spider   # about 6 minutes; writes data/processed/spider_duckdb/
uv run python -m text2sql.hub      # assembles data/hf/spider-duckdb/ for the Hub
```

The mirrored `spider_data.zip` was checked against the official Spider questions ([xlangai/spider](https://huggingface.co/datasets/xlangai/spider)): all 7,000 train and 1,034 dev rows are identical.

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

Results are written to `results/` (git-ignored). After changing comparison rules, re-score saved predictions without calling the model again:

```bash
uv run python -m text2sql.evaluate --rescore results/<run>.jsonl
```

## Baseline results

Off-the-shelf [Qwen3.5](https://huggingface.co/Qwen) instruct models, no fine-tuning. Setup: 4-bit GGUF (`Q4_K_M` from [unsloth](https://huggingface.co/unsloth)), llama.cpp `llama-server` on CPU (Apple M-series, 8 GB RAM), thinking mode off, temperature 0.

| Model | Overall | Easy (12) | Medium (25) | Hard (13) | Median latency |
|---|---|---|---|---|---|
| Qwen3.5-0.8B | 30.0% | 66.7% | 28.0% | 0.0% | 4.2 s |
| Qwen3.5-2B | 50.0% | 75.0% | 48.0% | 30.8% | 6.1 s |
| Qwen3.5-4B | 88.0% | 100.0% | 96.0% | 61.5% | 14.3 s |

What goes wrong for the small models:

- **Hallucinated columns** are the most common failure (e.g. `orders.product_id`, which lives in `order_items`). 22 of the 0.8B model's 35 failures are binder errors like this.
- **DuckDB `LIST` columns** are handled badly: `LIKE` on a list, `avg()` over a list, indexing `pages[0]` (DuckDB lists are 1-based). The web analytics schema, which uses lists, is the weakest area for both small models (8.3% and 16.7%).
- **Hard questions** (window functions with `QUALIFY`, recursive CTEs, per-group top-N) are near zero below 4B.

The fine-tuning target is to bring the 0.8B or 2B model close to the 4B model, while keeping their speed advantage (2.3x and 3.4x lower median latency than the 4B on this machine).

### Reproduce

llama.cpp's Homebrew formula needs macOS 14+, so on older macOS build it from source (takes a few minutes):

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build build --target llama-server -j 8
```

Download the models (e.g. `Qwen3.5-2B-Q4_K_M.gguf` from [unsloth/Qwen3.5-2B-GGUF](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF)) and run:

```bash
LLAMA_SERVER=~/llama.cpp/build/bin/llama-server \
  scripts/run_baseline.sh ~/models/Qwen3.5-0.8B-Q4_K_M.gguf ~/models/Qwen3.5-2B-Q4_K_M.gguf
```

The script runs on CPU by default (`LLAMA_ARGS="-ngl 0 --device none"`). On macOS 13 the Metal backend crashed while loading these models, so GPU offload is opt-in (`LLAMA_ARGS="-ngl 99"`).
