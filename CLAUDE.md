# huggingface-projects — monorepo notes for Claude

One folder per project, each a self-contained uv project with its own CLAUDE.md. Read only the CLAUDE.md of the project being worked on.

## Rules
- Do NOT run `git commit`/push. The user commits, pushes, opens PRs on GitHub (web UI) and merges. When a chunk is done, give branch name, commit commands, commit message, and PR title/description + web-UI steps.
- Work happens on feature branches named `<project>/<topic>` (e.g. `duckdb-text2sql/baseline`).
- Machine: Apple Silicon, 8 GB RAM, macOS 13.1 — keep local models ≤1.5B (4-bit); training goes to Colab/Kaggle GPUs.
- Ask before large downloads (models) or long runs.
