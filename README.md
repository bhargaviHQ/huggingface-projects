# Hugging Face Projects

Hands-on ML projects built with the Hugging Face ecosystem (🤗 Transformers, TRL, PEFT, Datasets, Spaces). Each project lives in its own folder with its own environment, evaluation, and README.

| Project | What it does | Status |
|---|---|---|
| [duckdb-text2sql](duckdb-text2sql/) | Small, execution-verified text-to-SQL model for DuckDB: dataset pipeline → fine-tuning → RL with execution rewards → Space demo | 🚧 Phase 0: baseline eval |

## Conventions

- One folder per project, each a self-contained [uv](https://docs.astral.sh/uv/) project (`cd <project> && uv sync`).
- Every project has an evaluation harness before any training.
- Artifacts (datasets, models, demos) are published on the Hugging Face Hub at [huggingface.co/bhargavis](https://huggingface.co/bhargavis).
