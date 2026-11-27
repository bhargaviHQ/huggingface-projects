"""Prompt construction and SQL extraction."""

from __future__ import annotations

import re

SYSTEM = (
    "You are an expert data engineer who writes correct DuckDB SQL. "
    "Given a database schema and a question, reply with a single DuckDB SQL query "
    "in a ```sql code block and nothing else."
)


def user_prompt(schema: str, question: str) -> str:
    return f"Schema:\n{schema}\n\nQuestion: {question}"


_FENCE = re.compile(r"```(?:sql|duckdb)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)
_START = re.compile(r"\b(WITH|SELECT|FROM)\b", re.IGNORECASE)


def extract_sql(text: str) -> str:
    """Pull the SQL query out of a model response."""
    m = _FENCE.search(text)
    if m:
        sql = m.group(1)
    else:
        start = _START.search(text)
        sql = text[start.start():] if start else text
    return sql.strip().rstrip(";").strip()
