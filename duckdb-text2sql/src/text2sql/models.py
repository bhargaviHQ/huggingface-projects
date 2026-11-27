"""Model backends that turn (schema, question) into a raw text response."""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Protocol

from .prompt import SYSTEM, user_prompt


class Backend(Protocol):
    name: str

    def generate(self, schema: str, question: str, gold_sql: str) -> str: ...


class GoldBackend:
    """Returns the reference SQL. Sanity check: must score 100%."""

    name = "gold"

    def generate(self, schema: str, question: str, gold_sql: str) -> str:
        return f"```sql\n{gold_sql}\n```"


class OpenAICompatBackend:
    """Any OpenAI-compatible chat endpoint: llama.cpp `llama-server`, Ollama, LM Studio,
    vLLM, or Hugging Face Inference Providers (https://router.huggingface.co/v1)."""

    def __init__(self, model: str, base_url: str, api_key_env: str | None = None,
                 max_tokens: int = 512, timeout_s: float = 120.0):
        self.model = model
        self.name = model
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = os.environ.get(api_key_env, "") if api_key_env else ""
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s

    def generate(self, schema: str, question: str, gold_sql: str) -> str:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": user_prompt(schema, question)}],
            "temperature": 0,
            "max_tokens": self.max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.url, json.dumps(body).encode(), headers)
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return json.load(resp)["choices"][0]["message"]["content"]
