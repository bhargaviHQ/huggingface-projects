"""Execution-based comparison of query results.

Two results match when they contain the same rows. Rules:
- numbers compare after rounding to 2 decimals (int 3 == float 3.0 == Decimal('3.00'))
- dates/timestamps compare by ISO string; a midnight timestamp equals the same date
- column order and column names are ignored (values within a row are compared as a multiset),
  but the number of columns must match
- row order matters only when the question asks for an ordering (``ordered=True``)
"""

from __future__ import annotations

import datetime as dt
import threading
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Any

import duckdb

Row = tuple[Any, ...]


def _norm_value(v: Any) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float, Decimal)):
        f = round(float(v), 2)
        return repr(0.0 if f == 0 else f)  # avoid "-0.0"
    if isinstance(v, dt.datetime) and v.time() == dt.time(0) and v.tzinfo is None:
        return v.date().isoformat()  # midnight timestamp == date (e.g. date_trunc output)
    if isinstance(v, (dt.datetime, dt.date, dt.time)):
        return v.isoformat()
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(_norm_value(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ",".join(f"{k}:{_norm_value(x)}" for k, x in sorted(v.items())) + "}"
    return str(v)


def _norm_row(row: Row) -> tuple[str, ...]:
    return tuple(sorted(_norm_value(v) for v in row))


def results_match(gold: list[Row], pred: list[Row], ordered: bool) -> bool:
    if len(gold) != len(pred):
        return False
    if gold and len(gold[0]) != len(pred[0]):
        return False
    g, p = [_norm_row(r) for r in gold], [_norm_row(r) for r in pred]
    return g == p if ordered else Counter(g) == Counter(p)


def run_query(db_path: Path, sql: str, timeout_s: float = 10.0) -> list[Row]:
    """Run SQL read-only against a database; raises on error or timeout."""
    with duckdb.connect(str(db_path), read_only=True) as con:
        timer = threading.Timer(timeout_s, con.interrupt)
        timer.start()
        try:
            return con.execute(sql).fetchall()
        finally:
            timer.cancel()
