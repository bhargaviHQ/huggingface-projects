# Hugging Face Projects

ML projects built with the Hugging Face ecosystem (Transformers, TRL, PEFT, Datasets, Spaces). Each project lives in its own folder with its own environment, evaluation and README.

| Project | Description | Status |
|---|---|---|
| [duckdb-text2sql](duckdb-text2sql/) | Small text-to-SQL model for DuckDB, trained on execution-verified data. Covers the dataset pipeline, fine-tuning, RL with execution rewards and a Space demo. | In progress (phase 2: fine-tuning) |

## Conventions

- One folder per project. Each is a self-contained [uv](https://docs.astral.sh/uv/) project (`cd <project> && uv sync`).
- Every project gets an evaluation harness before any training.
- Datasets, models and demos are published at [huggingface.co/bhargavis](https://huggingface.co/bhargavis).
