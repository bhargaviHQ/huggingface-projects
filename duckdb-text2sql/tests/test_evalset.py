"""Guards the quality of the hand-written eval set itself."""

import pytest

from text2sql.compare import run_query
from text2sql.evaluate import DEFAULT_EVAL, evaluate, load_eval
from text2sql.models import GoldBackend
from text2sql.schemas import build_all

EXAMPLES = load_eval(DEFAULT_EVAL)


@pytest.fixture(scope="session")
def db_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("dbs")
    build_all(d)
    return d


def test_ids_unique_and_fields_present():
    ids = [e["id"] for e in EXAMPLES]
    assert len(ids) == len(set(ids))
    for e in EXAMPLES:
        assert {"id", "db", "difficulty", "question", "gold_sql", "ordered"} <= e.keys()
        assert e["difficulty"] in {"easy", "medium", "hard"}


@pytest.mark.parametrize("ex", EXAMPLES, ids=[e["id"] for e in EXAMPLES])
def test_gold_runs_and_is_nonempty(ex, db_dir):
    rows = run_query(db_dir / f"{ex['db']}.duckdb", ex["gold_sql"])
    assert rows, "gold query returned no rows"
    assert all(v is not None for v in rows[0]) or len(rows) > 1, "gold returned only NULLs"


def test_databases_are_deterministic(tmp_path, db_dir):
    build_all(tmp_path)
    for ex in EXAMPLES:
        a = run_query(db_dir / f"{ex['db']}.duckdb", ex["gold_sql"])
        b = run_query(tmp_path / f"{ex['db']}.duckdb", ex["gold_sql"])
        assert sorted(map(repr, a)) == sorted(map(repr, b)), ex["id"]


def test_gold_backend_scores_100(db_dir):
    records = evaluate(GoldBackend(), EXAMPLES, db_dir)
    failed = [r["id"] for r in records if not r["correct"]]
    assert not failed
