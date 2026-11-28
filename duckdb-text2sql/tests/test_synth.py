import pytest

from text2sql.schemas import BUILDERS as EVAL_DBS
from text2sql.synth import VALIDATION_DBS, Candidate, generate, lit, verify
from text2sql.synth_dbs import BUILDERS as SYNTH_DBS, build_all

# families whose target query picks one row where several can tie; only these may mismatch
TIE_FAMILIES = {"unnest_top", "hour_top", "busiest_date", "arg_max", "top_n_per_group",
                "qualify_latest", "struct_list_top"}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    paths = build_all(tmp_path_factory.mktemp("synth"))
    cands = generate(paths)
    for c in cands:
        verify(c, paths[c.db_id])
    return paths, cands


def test_synth_databases_are_separate_from_eval_databases():
    assert not set(SYNTH_DBS) & set(EVAL_DBS)
    assert VALIDATION_DBS < set(SYNTH_DBS)


def test_lit_escapes_quotes():
    assert lit("O'Brien") == "'O''Brien'"
    assert lit(3) == "3"


def test_every_query_runs(built):
    _, cands = built
    errors = [(c.family, c.error) for c in cands if c.status.endswith("_error")]
    assert errors == []


def test_mismatches_only_come_from_ties(built):
    _, cands = built
    assert {c.family for c in cands if c.status == "mismatch"} <= TIE_FAMILIES


def test_most_candidates_verify_and_cover_every_database(built):
    _, cands = built
    ok = [c for c in cands if c.status == "verified"]
    assert len(ok) / len(cands) > 0.9
    assert {c.db_id for c in ok} == set(SYNTH_DBS)


def test_generation_is_deterministic(built):
    paths, cands = built
    again = generate(paths)
    assert [(c.question, c.sql) for c in again] == [(c.question, c.sql) for c in cands]


def test_queries_use_current_lambda_syntax(built):
    _, cands = built
    # DuckDB 2.0 rejects the old "x -> expr" lambda syntax by default
    assert not [c.sql for c in cands if "->" in c.sql]
    assert any("lambda" in c.sql for c in cands)


def test_tie_is_rejected(built):
    paths, _ = built
    c = Candidate("music", "t", [], "q", "SELECT 1 AS x FROM range(2) LIMIT 1",
                  "SELECT 1 AS x FROM range(2)")
    assert verify(c, paths["music"]).status == "mismatch"


def test_trivial_and_empty_results_are_rejected(built):
    paths, _ = built
    zero = "SELECT count(*) FROM artists WHERE false"
    assert verify(Candidate("music", "t", [], "q", zero, zero), paths["music"]).status == "trivial_result"
    empty = "SELECT name FROM artists WHERE false"
    assert verify(Candidate("music", "t", [], "q", empty, empty), paths["music"]).status == "empty_result"
