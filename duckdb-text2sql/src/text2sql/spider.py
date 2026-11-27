"""Convert Spider (SQLite) into an execution-verified DuckDB text-to-SQL dataset.

For every Spider example:
  1. the SQLite database is copied into DuckDB (column types inferred from the stored values),
  2. the SQLite gold query is transpiled to DuckDB with sqlglot,
  3. both queries are executed and kept only if they return the same result.

Run: python -m text2sql.spider            (expects data/raw/spider_data, see README)
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import duckdb
import pyarrow as pa
import sqlglot
from sqlglot import exp
from sqlglot.optimizer.annotate_types import annotate_types
from sqlglot.optimizer.qualify import qualify

from .compare import results_match, run_query
from .schemas import schema_ddl

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "spider_data"
OUT = ROOT / "data" / "processed" / "spider_duckdb"
SPLITS = {"train": "train_spider.json", "validation": "dev.json"}


# --------------------------------------------------------------------------- database copy
_NUMERIC_DECL = ("INT", "REAL", "FLOA", "DOUB", "NUMERIC", "DEC")


def _is_blank(v) -> bool:
    return isinstance(v, str) and not v.strip()


def _infer_type(declared: str, values: list) -> str:
    """Pick a DuckDB type that preserves SQLite's stored values.

    Blank strings in a column declared numeric are treated as NULL (they are missing values in
    Spider, e.g. bike_1.weather.max_temperature_f). Anything else non-numeric makes the column VARCHAR.
    """
    numeric_decl = any(t in declared.upper() for t in _NUMERIC_DECL)
    present = [v for v in values if v is not None and not (numeric_decl and _is_blank(v))]
    if present:
        kinds = {type(v) for v in present}
        if kinds <= {int}:
            return "BIGINT"
        if kinds <= {int, float}:
            return "DOUBLE"
        if kinds == {bytes}:
            return "BLOB"
        return "VARCHAR"
    d = declared.upper()
    if "INT" in d:
        return "BIGINT"
    if any(t in d for t in ("REAL", "FLOA", "DOUB", "NUMERIC", "DEC")):
        return "DOUBLE"
    return "VARCHAR"


def _sqlite_connect(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    con.text_factory = lambda b: b.decode("utf-8", errors="replace")
    return con


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def copy_sqlite_to_duckdb(sqlite_path: Path, duck_path: Path) -> None:
    src = _sqlite_connect(sqlite_path)
    duck_path.unlink(missing_ok=True)
    with duckdb.connect(str(duck_path)) as dst:
        tables = [r[0] for r in src.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
        for table in tables:
            cols = src.execute(f"PRAGMA table_info({_q(table)})").fetchall()  # cid,name,type,...
            rows = src.execute(f"SELECT * FROM {_q(table)}").fetchall()
            types = [_infer_type(c[2] or "", [r[i] for r in rows]) for i, c in enumerate(cols)]
            col_defs = ", ".join(f"{_q(c[1])} {t}" for c, t in zip(cols, types))
            dst.execute(f"CREATE TABLE {_q(table)} ({col_defs})")
            if rows:
                arrow = pa.table([_arrow_column([r[i] for r in rows], t) for i, t in enumerate(types)],
                                 names=[f"c{i}" for i in range(len(cols))])
                dst.register("src_rows", arrow)
                dst.execute(f"INSERT INTO {_q(table)} SELECT * FROM src_rows")
                dst.unregister("src_rows")
    src.close()


_ARROW_TYPES = {"BIGINT": pa.int64(), "DOUBLE": pa.float64(), "BLOB": pa.binary(), "VARCHAR": pa.string()}


def _arrow_column(values: list, duck_type: str) -> pa.Array:
    if duck_type in ("BIGINT", "DOUBLE"):
        values = [None if v is None or _is_blank(v) else v for v in values]
    if duck_type == "DOUBLE":
        values = [None if v is None else float(v) for v in values]
    elif duck_type == "VARCHAR":  # mixed-type columns keep SQLite's text form of each value
        values = [v if v is None or isinstance(v, str) else str(v) for v in values]
    return pa.array(values, type=_ARROW_TYPES[duck_type])


# --------------------------------------------------------------------------- per-example verification
@dataclass
class Result:
    id: str
    db_id: str
    question: str
    sqlite_sql: str
    sql: str | None
    ordered: bool
    status: str  # verified | empty_result | duplicate | transpile_error | sqlite_error | duckdb_error | mismatch
    error: str = ""
    num_rows: int = 0
    method: str = ""  # transpile method that produced the verified query
    rewrites: list[str] = field(default_factory=list)  # SQLite-quirk rewrites applied


def is_ordered(sql: str) -> bool:
    """True if the outermost query has an ORDER BY (row order is then part of the answer)."""
    try:
        return sqlglot.parse_one(sql, read="sqlite").args.get("order") is not None
    except Exception:
        return False


TRANSPILE_METHODS = ("plain", "typed", "qualified")


def _columns_outside_aggregates(node: exp.Expression):
    """Column refs in this SELECT's scope that are not inside an aggregate or a subquery."""
    for child in node.iter_expressions():
        if isinstance(child, (exp.AggFunc, exp.Select, exp.Subquery)):
            continue
        if isinstance(child, exp.Column):
            yield child
        else:
            yield from _columns_outside_aggregates(child)


def fix_sqlite_quirks(expr: exp.Expression, schema: dict) -> tuple[exp.Expression, list[str]]:
    """Rewrite SQLite-only behaviour into standard SQL that DuckDB accepts.

    - double_quoted_string: SQLite treats "Village" as a string literal when no column has that
      name. Standard SQL (and DuckDB) read it as an identifier.
    - like_to_ilike: SQLite LIKE ignores case for ASCII letters; DuckDB LIKE does not.
    - bare_group_by_column: SQLite allows selecting columns that are neither grouped nor
      aggregated. They are added to GROUP BY; execution verification later rejects the rewrite
      if it changes the result.
    """
    applied: list[str] = []
    # SQLite LIKE is case-insensitive (for ASCII); DuckDB LIKE is case-sensitive.
    for like in list(expr.find_all(exp.Like)):
        like.replace(exp.ILike(this=like.this, expression=like.expression))
        if "like_to_ilike" not in applied:
            applied.append("like_to_ilike")
    known = {c.lower() for cols in schema.values() for c in cols}
    for col in list(expr.find_all(exp.Column)):
        ident = col.this
        if (isinstance(ident, exp.Identifier) and ident.quoted and not col.table
                and ident.name.lower() not in known):
            col.replace(exp.Literal.string(ident.name))
            if "double_quoted_string" not in applied:
                applied.append("double_quoted_string")
    for select in expr.find_all(exp.Select):
        group = select.args.get("group")
        if not group:
            continue
        grouped = {g.sql().lower() for g in group.expressions}
        aliases = {e.alias.lower() for e in select.expressions if isinstance(e, exp.Alias)}
        scope = [*select.expressions, *([select.args["order"]] if select.args.get("order") else [])]
        for item in scope:
            if isinstance(item, exp.Alias):
                item = item.this
            if isinstance(item, (exp.AggFunc, exp.Select, exp.Subquery)):
                continue
            for col in ([item] if isinstance(item, exp.Column) else _columns_outside_aggregates(item)):
                key = col.sql().lower()
                if key not in grouped and col.name.lower() not in aliases:
                    group.append("expressions", col.copy())
                    grouped.add(key)
                    if "bare_group_by_column" not in applied:
                        applied.append("bare_group_by_column")
    return expr, applied


def transpile(sql: str, method: str = "plain", schema: dict | None = None) -> str:
    """SQLite -> DuckDB.

    plain:     syntax-only transpile; keeps the original query shape.
    typed:     annotate column types first, so e.g. SQLite integer division stays integer
               division. Only resolves columns that carry a table qualifier.
    qualified: qualify every column, then annotate types. Most robust, but rewrites the
               query (adds table prefixes and aliases), so it is the last resort.
    """
    return transpile_with_fixes(sql, method, schema)[0]


def transpile_with_fixes(sql: str, method: str = "plain",
                         schema: dict | None = None) -> tuple[str, list[str]]:
    """Like transpile(), also returning which SQLite-quirk rewrites were applied."""
    expr = sqlglot.parse_one(sql, read="sqlite")
    applied: list[str] = []
    if schema:
        expr, applied = fix_sqlite_quirks(expr, schema)
    if method == "qualified":
        expr = qualify(expr, schema=schema, dialect="sqlite", quote_identifiers=False, identify=False)
    if method in ("typed", "qualified"):
        expr = annotate_types(expr, schema=schema, dialect="sqlite")
    return expr.sql("duckdb"), applied


def duckdb_schema(duck_path: Path) -> dict[str, dict[str, str]]:
    with duckdb.connect(str(duck_path), read_only=True) as con:
        rows = con.execute("SELECT table_name, column_name, data_type FROM information_schema.columns "
                           "ORDER BY table_name, ordinal_position").fetchall()
    schema: dict[str, dict[str, str]] = {}
    for table, col, typ in rows:
        schema.setdefault(table, {})[col] = typ
    return schema


def run_sqlite(path: Path, sql: str, timeout_s: float = 10.0) -> list[tuple]:
    con = _sqlite_connect(path)
    deadline = time.monotonic() + timeout_s
    con.set_progress_handler(lambda: time.monotonic() > deadline, 10_000)
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


_NULLS_FIRST = re.compile(r"\s+NULLS FIRST\b")


def _simplify(sql: str, gold: list, duck_path: Path, ordered: bool) -> str:
    """Drop the NULLS FIRST that sqlglot adds to mimic SQLite's NULL ordering, but only if the
    result is unchanged (i.e. the sorted columns have no NULLs that would move)."""
    simpler = _NULLS_FIRST.sub("", sql)
    if simpler == sql:
        return sql
    try:
        return simpler if results_match(gold, run_query(duck_path, simpler), ordered) else sql
    except Exception:
        return sql


def verify_example(ex_id: str, db_id: str, question: str, sqlite_sql: str,
                   sqlite_path: Path, duck_path: Path, schema: dict | None = None) -> Result:
    """Try each transpile method in order; keep the first DuckDB query whose result matches SQLite."""
    r = Result(ex_id, db_id, question.strip(), sqlite_sql.strip(), None, is_ordered(sqlite_sql), "")
    try:
        gold = run_sqlite(sqlite_path, sqlite_sql)
    except Exception as e:
        r.status, r.error = "sqlite_error", str(e)[:300]
        return r
    r.num_rows = len(gold)
    if schema is None:
        schema = duckdb_schema(duck_path)
    for method in TRANSPILE_METHODS:
        try:
            sql, applied = transpile_with_fixes(sqlite_sql, method, schema)
        except Exception as e:
            r.status, r.error = "transpile_error", str(e).splitlines()[0][:300]
            continue
        try:
            pred = run_query(duck_path, sql)
        except Exception as e:
            r.status, r.error, r.sql = "duckdb_error", str(e).splitlines()[0][:300], sql
            continue
        if results_match(gold, pred, r.ordered):
            sql = _simplify(sql, gold, duck_path, r.ordered)
            r.sql, r.method, r.error, r.rewrites = sql, method, "", applied
            # an empty result "matches" trivially, so it doesn't count as verified
            r.status = "verified" if gold else "empty_result"
            return r
        r.status, r.sql = "mismatch", sql
        r.error = f"sqlite={gold[:3]!r} duckdb={pred[:3]!r}"[:300]
    return r


# --------------------------------------------------------------------------- pipeline
def build(raw: Path = RAW, out: Path = OUT) -> dict:
    db_out = out / "databases"
    db_out.mkdir(parents=True, exist_ok=True)
    report: dict = {}
    for split, fname in SPLITS.items():
        examples = json.loads((raw / fname).read_text())
        results, seen, schemas = [], set(), {}
        for i, ex in enumerate(examples):
            db_id = ex["db_id"]
            sqlite_path = raw / "database" / db_id / f"{db_id}.sqlite"
            duck_path = db_out / f"{db_id}.duckdb"
            if db_id not in schemas:
                if not duck_path.exists():
                    copy_sqlite_to_duckdb(sqlite_path, duck_path)
                schemas[db_id] = duckdb_schema(duck_path)
            r = verify_example(f"spider_{split}_{i:05d}", db_id, ex["question"], ex["query"],
                               sqlite_path, duck_path, schemas[db_id])
            key = (db_id, " ".join(r.question.lower().split()))
            if r.status == "verified" and key in seen:
                r.status = "duplicate"
            seen.add(key)
            results.append(r)
        _write_split(out, split, results)
        report[split] = {
            "total": len(results),
            "status": dict(Counter(r.status for r in results).most_common()),
            "verified_by_method": dict(Counter(r.method for r in results if r.status == "verified")),
            "verified_with_rewrite": dict(Counter(w for r in results if r.status == "verified"
                                                  for w in r.rewrites)),
        }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    return report


def _write_split(out: Path, split: str, results: list[Result]) -> None:
    schemas: dict[str, str] = {}
    db_out = out / "databases"
    rows = []
    for r in results:
        if r.status != "verified":
            continue
        schema = schemas.setdefault(r.db_id, schema_ddl(db_out / f"{r.db_id}.duckdb"))
        rows.append({"id": r.id, "db_id": r.db_id, "question": r.question, "sql": r.sql,
                     "schema": schema, "ordered": r.ordered, "source": "spider",
                     "transpile_method": r.method, "rewrites": r.rewrites,
                     "original_sqlite_sql": r.sqlite_sql, "num_rows": r.num_rows})
    # every non-verified example is kept for analysis (e.g. sqlglot issues to report upstream)
    with open(out / f"{split}_rejected.jsonl", "w") as f:
        for r in results:
            if r.status != "verified":
                f.write(json.dumps(asdict(r)) + "\n")
    jsonl = out / f"{split}.jsonl"
    jsonl.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with duckdb.connect() as con:
        con.execute(f"COPY (SELECT * FROM read_json('{jsonl}', format='newline_delimited')) "
                    f"TO '{out / f'{split}.parquet'}' (FORMAT parquet)")


if __name__ == "__main__":
    t0 = time.time()
    print(json.dumps(build(), indent=2))
    print(f"done in {time.time() - t0:.0f}s, output: {OUT}")
