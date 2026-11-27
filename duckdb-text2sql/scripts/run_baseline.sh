#!/usr/bin/env bash
# Run the baseline eval for one or more GGUF models with llama.cpp's llama-server.
#
# Usage:
#   scripts/run_baseline.sh ~/models/Qwen3.5-0.8B-Q4_K_M.gguf [more.gguf ...]
#
# Env vars:
#   LLAMA_SERVER  path to llama-server (default: llama-server on PATH)
#   LLAMA_ARGS    extra server args (default runs on CPU; set to "-ngl 99" to use the GPU)
#   PORT          server port (default 8080)
set -euo pipefail

LLAMA_SERVER=${LLAMA_SERVER:-llama-server}
LLAMA_ARGS=${LLAMA_ARGS:-"-ngl 0 --device none"}
PORT=${PORT:-8080}
cd "$(dirname "$0")/.."

for model in "$@"; do
  name=$(basename "$model" .gguf)
  echo "=== $name"
  # Thinking mode is turned off so we measure direct SQL generation.
  # shellcheck disable=SC2086
  "$LLAMA_SERVER" -m "$model" --port "$PORT" -c 8192 --jinja --reasoning off $LLAMA_ARGS \
    > "results/server-$name.log" 2>&1 &
  pid=$!
  trap 'kill $pid 2>/dev/null' EXIT
  for _ in $(seq 1 120); do
    curl -sf "localhost:$PORT/health" >/dev/null && break
    sleep 1
  done
  uv run python -m text2sql.evaluate --backend openai --model "$name" \
    --base-url "http://localhost:$PORT/v1"
  kill "$pid"; wait "$pid" 2>/dev/null || true
done
