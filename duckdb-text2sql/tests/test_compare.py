from datetime import date
from decimal import Decimal

from text2sql.compare import results_match
from text2sql.prompt import extract_sql


def test_numeric_types_and_rounding_match():
    assert results_match([(3,)], [(3.0,)], ordered=False)
    assert results_match([(Decimal("12.345"),)], [(12.35,)], ordered=False)
    assert not results_match([(12.3,)], [(12.4,)], ordered=False)


def test_column_order_ignored_but_count_enforced():
    assert results_match([("a", 1)], [(1, "a")], ordered=False)
    assert not results_match([("a",)], [("a", 1)], ordered=False)


def test_row_order_only_when_ordered():
    gold, pred = [(1,), (2,)], [(2,), (1,)]
    assert results_match(gold, pred, ordered=False)
    assert not results_match(gold, pred, ordered=True)


def test_duplicates_matter():
    assert not results_match([(1,), (1,)], [(1,), (2,)], ordered=False)


def test_dates():
    assert results_match([(date(2024, 1, 1),)], [(date(2024, 1, 1),)], ordered=True)


def test_extract_sql():
    assert extract_sql("Here:\n```sql\nSELECT 1;\n```\nDone") == "SELECT 1"
    assert extract_sql("SELECT a FROM t;") == "SELECT a FROM t"
    assert extract_sql("Sure! SELECT 2") == "SELECT 2"
