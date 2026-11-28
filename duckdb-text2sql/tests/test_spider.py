import sqlite3

import duckdb
import pytest

from text2sql.spider import _infer_type, copy_sqlite_to_duckdb, is_ordered, verify_example


@pytest.fixture
def dbs(tmp_path):
    sqlite_path, duck_path = tmp_path / "t.sqlite", tmp_path / "t.duckdb"
    con = sqlite3.connect(sqlite_path)
    con.executescript("""
        CREATE TABLE singer (singer_id INT, name TEXT, age INT, net_worth REAL, note TEXT);
        INSERT INTO singer VALUES (1, 'Ada', 30, 1.5, 'x'), (2, 'Bo', 41, 2, 7), (3, 'Cy', 25, NULL, NULL);
        CREATE TABLE "song list" (song_id INT, singer_id INT, title TEXT);
        INSERT INTO "song list" VALUES (10, 1, 'One'), (11, 1, 'Two'), (12, 2, 'Three');
        CREATE TABLE empty_t (a INTEGER, b VARCHAR(20));
    """)
    con.commit()
    con.close()
    copy_sqlite_to_duckdb(sqlite_path, duck_path)
    return sqlite_path, duck_path


def test_infer_type():
    assert _infer_type("INT", [1, 2, None]) == "BIGINT"
    assert _infer_type("REAL", [1, 2.5]) == "DOUBLE"
    assert _infer_type("INT", [1, "n/a"]) == "VARCHAR"
    assert _infer_type("INTEGER", []) == "BIGINT"
    assert _infer_type("VARCHAR(20)", [None]) == "VARCHAR"


def test_copy_preserves_tables_types_and_values(dbs):
    _, duck_path = dbs
    with duckdb.connect(str(duck_path), read_only=True) as con:
        types = dict(con.execute("SELECT column_name, data_type FROM information_schema.columns "
                                 "WHERE table_name = 'singer'").fetchall())
        assert types == {"singer_id": "BIGINT", "name": "VARCHAR", "age": "BIGINT",
                         "net_worth": "DOUBLE", "note": "VARCHAR"}
        assert con.execute("SELECT note FROM singer ORDER BY singer_id").fetchall() == [("x",), ("7",), (None,)]
        assert con.execute('SELECT count(*) FROM "song list"').fetchone() == (3,)
        assert con.execute("SELECT count(*) FROM empty_t").fetchone() == (0,)


def test_is_ordered():
    assert is_ordered("SELECT name FROM singer ORDER BY age")
    assert not is_ordered("SELECT name FROM singer WHERE singer_id IN (SELECT 1 ORDER BY 1)")


def _verify(dbs, sql):
    return verify_example("x", "t", "q?", sql, *dbs)


def test_verified_join(dbs):
    r = _verify(dbs, 'SELECT T1.name, count(*) FROM singer AS T1 JOIN "song list" AS T2 '
                     "ON T1.singer_id = T2.singer_id GROUP BY T1.name")
    assert r.status == "verified", r.error
    assert r.num_rows == 2 and r.sql


def test_integer_division_semantics_survive_transpile(dbs):
    # SQLite integer division (41 / 2 = 20) must not become float division in DuckDB
    r = _verify(dbs, "SELECT age / 2 FROM singer WHERE singer_id = 2")
    assert r.status == "verified", (r.sql, r.error)
    assert r.method != "plain"  # plain transpile gives 20.5; a type-aware fallback is needed


def test_plain_method_preferred_when_it_works(dbs):
    assert _verify(dbs, "SELECT name FROM singer WHERE age > 26").method == "plain"


def test_double_quoted_string_literal(dbs):
    r = _verify(dbs, 'SELECT name FROM singer WHERE name != "Ada"')
    assert r.status == "verified", (r.sql, r.error)
    assert "'Ada'" in r.sql and "double_quoted_string" in r.rewrites


def test_double_quoted_real_column_stays_identifier(dbs):
    r = _verify(dbs, 'SELECT "name" FROM singer WHERE age > 26')
    assert r.status == "verified" and r.rewrites == []


def test_bare_group_by_column_added_to_group_by(dbs):
    r = _verify(dbs, 'SELECT T1.singer_id, T1.name, count(*) FROM singer AS T1 JOIN "song list" AS T2 '
                     "ON T1.singer_id = T2.singer_id GROUP BY T1.singer_id")
    assert r.status == "verified", (r.sql, r.error)
    assert "bare_group_by_column" in r.rewrites
    assert "GROUP BY T1.singer_id, T1.name" in r.sql


def test_aggregated_columns_not_added_to_group_by(dbs):
    r = _verify(dbs, "SELECT avg(age), name FROM singer GROUP BY name")
    assert r.status == "verified", (r.sql, r.error)
    assert r.rewrites == [] and "GROUP BY name" in r.sql and "age" not in r.sql.split("GROUP BY")[1]


def test_like_is_case_insensitive_like_sqlite(dbs):
    r = _verify(dbs, "SELECT name FROM singer WHERE name LIKE 'a%'")
    assert r.status == "verified", (r.sql, r.error)
    assert "ILIKE" in r.sql and "like_to_ilike" in r.rewrites


def test_nulls_first_kept_only_when_it_matters(dbs):
    # net_worth has a NULL: SQLite sorts it first, DuckDB last, so NULLS FIRST must stay
    r = _verify(dbs, "SELECT name FROM singer ORDER BY net_worth")
    assert r.status == "verified" and "NULLS FIRST" in r.sql
    # age has no NULLs: the clause is noise and gets dropped
    r = _verify(dbs, "SELECT name FROM singer ORDER BY age")
    assert r.status == "verified" and "NULLS" not in r.sql


def test_blank_strings_in_numeric_column_become_null(tmp_path):
    sqlite_path, duck_path = tmp_path / "w.sqlite", tmp_path / "w.duckdb"
    con = sqlite3.connect(sqlite_path)
    con.executescript("CREATE TABLE w (temp INTEGER); INSERT INTO w VALUES (80), (90), ('');")
    con.commit()
    con.close()
    copy_sqlite_to_duckdb(sqlite_path, duck_path)
    with duckdb.connect(str(duck_path), read_only=True) as d:
        assert d.execute("SELECT data_type FROM information_schema.columns").fetchone() == ("BIGINT",)
        assert d.execute("SELECT count(temp), count(*) FROM w").fetchone() == (2, 3)


def test_empty_result_is_not_verified(dbs):
    assert _verify(dbs, "SELECT name FROM singer WHERE age > 100").status == "empty_result"


def test_sqlite_error_is_reported(dbs):
    assert _verify(dbs, "SELECT missing_col FROM singer").status == "sqlite_error"
