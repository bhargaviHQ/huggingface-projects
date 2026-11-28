"""Generate DuckDB-specific text-to-SQL examples from templates, verified by execution.

Each template produces a question, a target query that uses a DuckDB feature (LIST and STRUCT
columns, lambdas, QUALIFY, arg_max, FILTER, GROUP BY ALL, date functions), and a second query
that answers the same question a different way (often plain SQL). An example is kept only if
both queries run, return a non-trivial result, and agree. Ties are caught this way: a "top 1"
target query returns one row, while its rank()-based check returns every tied row.

Templates are applied to the databases in synth_dbs.py. One database (repos) is held out as the
validation split, so validation measures transfer to an unseen schema.

Run: python -m text2sql.synth
"""

from __future__ import annotations

import calendar
import json
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import duckdb

from .compare import results_match, run_query
from .schemas import schema_ddl
from .synth_dbs import build_all

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "processed" / "duckdb_synth"
VALIDATION_DBS = {"repos"}
SEED = 7
PER_TEMPLATE = 8  # max slot values sampled per template and table


def lit(v) -> str:
    """SQL literal for a slot value."""
    if isinstance(v, str):
        return "'" + v.replace("'", "''") + "'"
    return str(v)


@dataclass
class Candidate:
    db_id: str
    family: str
    features: list[str]
    question: str
    sql: str
    check_sql: str
    ordered: bool = False
    status: str = ""
    error: str = ""
    num_rows: int = 0
    id: str = ""


@dataclass
class Gen:
    db_id: str
    con: duckdb.DuckDBPyConnection
    rng: random.Random
    out: list[Candidate] = field(default_factory=list)

    def values(self, sql: str) -> list:
        # sorted so sampling does not depend on the order DuckDB happens to return rows in
        return sorted((r[0] for r in self.con.execute(sql).fetchall() if r[0] is not None), key=repr)

    def sample(self, sql: str, k: int = PER_TEMPLATE) -> list:
        vals = self.values(sql)
        return self.rng.sample(vals, min(k, len(vals)))

    def pairs(self, sql: str, k: int = PER_TEMPLATE) -> list[tuple]:
        vals = self.values(sql)
        combos = [(a, b) for i, a in enumerate(vals) for b in vals[i + 1:]]
        return self.rng.sample(combos, min(k, len(combos)))

    def add(self, family: str, features: list[str], phrasings: list[str], sql: str, check: str,
            ordered: bool = False, **slots) -> None:
        q = self.rng.choice(phrasings).format(**slots)
        self.out.append(Candidate(self.db_id, family, features, q, " ".join(sql.split()),
                                  " ".join(check.split()), ordered))


# --------------------------------------------------------------------------- table metadata
@dataclass(frozen=True)
class ListCol:
    table: str
    pk: str
    col: str
    ent: str        # plural noun for a row: "artists"
    ent1: str       # singular: "artist"
    item: str       # singular noun for a list element: "genre"
    items: str      # plural: "genres"
    label: str      # column that names a row
    label_desc: str  # "names"
    can_be_empty: bool = False


@dataclass(frozen=True)
class StructField:
    table: str
    col: str
    field: str
    desc: str       # "site"
    ent: str
    ent1: str
    numeric: bool
    label: str
    label_desc: str


@dataclass(frozen=True)
class TimeCol:
    table: str
    col: str
    ent: str        # "plays", "sign-ups"
    is_timestamp: bool


@dataclass(frozen=True)
class Latest:  # one row per group, chosen by time
    table: str
    group: str
    group_desc: str
    ts: str
    ret: str
    ret_desc: str
    ent1: str


@dataclass(frozen=True)
class ArgMax:  # per group, the row with the highest/lowest value
    table: str
    group: str
    group_desc: str
    ret: str
    ret_desc: str
    val: str
    high_desc: str  # "longest"
    low_desc: str   # "shortest"
    ent: str


@dataclass(frozen=True)
class Filtered:  # count(*) FILTER (WHERE ...)
    table: str
    group: str
    group_desc: str
    cond: str
    cond_desc: str  # "were skipped"
    ent: str


@dataclass(frozen=True)
class Duration:
    table: str
    start: str
    end: str
    ent: str
    verb: str       # "resolve"
    start_word: str  # "raised"
    end_word: str    # "resolved"
    group: str
    group_desc: str


META: dict[str, dict[str, list]] = {
    "music": {
        "lists": [ListCol("artists", "artist_id", "genres", "artists", "artist", "genre", "genres", "name", "names"),
                  ListCol("tracks", "track_id", "tags", "tracks", "track", "tag", "tags", "title", "titles", True)],
        "structs": [],
        "times": [TimeCol("plays", "played_at", "plays", True),
                  TimeCol("listeners", "signup_date", "listener sign-ups", False),
                  TimeCol("tracks", "release_date", "track releases", False)],
        "latest": [Latest("plays", "listener_id", "listener", "played_at", "track_id", "track id", "play"),
                   Latest("plays", "track_id", "track", "played_at", "device", "device", "play")],
        "argmax": [ArgMax("tracks", "artist_id", "artist", "title", "title", "duration_s",
                          "longest", "shortest", "tracks"),
                   ArgMax("artists", "country", "country", "name", "name", "formed_year",
                          "most recently formed", "oldest", "artists")],
        "filtered": [Filtered("plays", "device", "device", "seconds_played < 30",
                              "lasted under 30 seconds", "plays"),
                     Filtered("tracks", "artist_id", "artist", "explicit", "are explicit", "tracks")],
        "durations": [],
    },
    "recipes": {
        "lists": [ListCol("recipes", "recipe_id", "tags", "recipes", "recipe", "tag", "tags", "name", "names", True)],
        "structs": [StructField("recipes", "nutrition", "calories", "calories", "recipes", "recipe", True, "name", "names"),
                    StructField("recipes", "nutrition", "protein_g", "grams of protein", "recipes", "recipe", True,
                                "name", "names")],
        "times": [TimeCol("reviews", "reviewed_at", "reviews", False)],
        "latest": [Latest("reviews", "recipe_id", "recipe", "reviewed_at", "rating", "rating", "review")],
        "argmax": [ArgMax("recipes", "cuisine", "cuisine", "name", "name", "prep_minutes",
                          "slowest to prepare", "quickest to prepare", "recipes"),
                   ArgMax("recipes", "chef", "chef", "name", "name", "servings",
                          "largest (by servings)", "smallest (by servings)", "recipes")],
        "filtered": [Filtered("reviews", "recipe_id", "recipe", "rating = 5", "gave 5 stars", "reviews"),
                     Filtered("recipes", "cuisine", "cuisine", "prep_minutes <= 30",
                              "take 30 minutes or less", "recipes")],
        "durations": [],
    },
    "iot": {
        "lists": [ListCol("devices", "device_id", "sensors", "devices", "device", "sensor type", "sensor types",
                          "name", "names"),
                  ListCol("alerts", "alert_id", "codes", "alerts", "alert", "alert code", "alert codes",
                          "alert_id", "ids")],
        "structs": [StructField("devices", "location", "site", "site", "devices", "device", False, "name", "names"),
                    StructField("devices", "location", "floor", "floor", "devices", "device", True, "name", "names")],
        "times": [TimeCol("alerts", "raised_at", "alerts", True),
                  TimeCol("devices", "installed_on", "device installations", False)],
        "latest": [Latest("readings", "device_id", "device", "ts", "temperature", "temperature", "reading"),
                   Latest("alerts", "device_id", "device", "raised_at", "severity", "severity", "alert")],
        "argmax": [ArgMax("readings", "device_id", "device", "ts", "time", "temperature",
                          "hottest", "coldest", "readings")],
        "filtered": [Filtered("alerts", "severity", "severity", "resolved_at IS NULL",
                              "are still unresolved", "alerts"),
                     Filtered("readings", "device_id", "device", "temperature > 25",
                              "were above 25 degrees", "readings")],
        "durations": [Duration("alerts", "raised_at", "resolved_at", "alerts", "resolve", "raised", "resolved", "severity",
                               "severity")],
    },
    "repos": {
        "lists": [ListCol("repos", "repo_id", "topics", "repositories", "repository", "topic", "topics", "name", "names", True),
                  ListCol("issues", "issue_id", "labels", "issues", "issue", "label", "labels", "issue_id", "ids", True),
                  ListCol("commits", "sha", "files_changed", "commits", "commit", "changed file", "changed files",
                          "sha", "hashes")],
        "structs": [StructField("repos", "owner", "kind", "owner type", "repositories", "repository", False, "name", "names"),
                    StructField("repos", "owner", "login", "owner", "repositories", "repository", False, "name", "names")],
        "times": [TimeCol("commits", "committed_at", "commits", True),
                  TimeCol("issues", "opened_at", "new issues", True),
                  TimeCol("repos", "created_at", "new repositories", False)],
        "latest": [Latest("commits", "repo_id", "repository", "committed_at", "author", "author", "commit"),
                   Latest("issues", "author", "author", "opened_at", "title", "title", "issue")],
        "argmax": [ArgMax("repos", "language", "language", "name", "name", "stars",
                          "most starred", "least starred", "repositories"),
                   ArgMax("commits", "repo_id", "repository", "sha", "hash", "additions",
                          "largest (by lines added)", "smallest (by lines added)", "commits")],
        "filtered": [Filtered("issues", "repo_id", "repository", "closed_at IS NULL", "are still open", "issues"),
                     Filtered("commits", "author", "author", "deletions > additions",
                              "removed more lines than they added", "commits")],
        "durations": [Duration("issues", "opened_at", "closed_at", "issues", "close", "opened", "closed", "repo_id",
                               "repository")],
    },
}


# --------------------------------------------------------------------------- list families
def list_families(g: Gen, m: ListCol) -> None:
    t, c, pk = m.table, m.col, m.pk
    vals_sql = f"SELECT DISTINCT x FROM (SELECT unnest({c}) AS x FROM {t}) ORDER BY x"
    for v in g.sample(vals_sql):
        g.add("list_contains_count", ["list_contains"],
              ["How many {ent} have {v} among their {items}?",
               "Count the {ent} whose {items} include {v}.",
               "How many {ent} have the {item} {v}?"],
              f"SELECT count(*) FROM {t} WHERE list_contains({c}, {lit(v)})",
              f"SELECT count(DISTINCT {pk}) FROM (SELECT {pk}, unnest({c}) AS x FROM {t}) WHERE x = {lit(v)}",
              ent=m.ent, items=m.items, item=m.item, v=lit(v))
    for v in g.sample(vals_sql):
        g.add("list_contains_rows", ["list_contains"],
              ["Which {ent} have {v} among their {items}? Return their {label_desc}.",
               "List the {label_desc} of the {ent} with the {item} {v}.",
               "Give the {label_desc} of all {ent} whose {items} include {v}."],
              f"SELECT {m.label} FROM {t} WHERE list_contains({c}, {lit(v)})",
              f"SELECT {m.label} FROM {t} WHERE EXISTS (SELECT 1 FROM unnest({t}.{c}) AS u(x) WHERE u.x = {lit(v)})",
              ent=m.ent, items=m.items, item=m.item, v=lit(v), label_desc=m.label_desc)
    for a, b in g.pairs(vals_sql):
        g.add("list_has_all", ["list_has_all"],
              ["How many {ent} have both {a} and {b} among their {items}?",
               "Count the {ent} whose {items} include both {a} and {b}."],
              f"SELECT count(*) FROM {t} WHERE list_has_all({c}, [{lit(a)}, {lit(b)}])",
              f"SELECT count(*) FROM {t} WHERE list_contains({c}, {lit(a)}) AND list_contains({c}, {lit(b)})",
              ent=m.ent, items=m.items, a=lit(a), b=lit(b))
    for a, b in g.pairs(vals_sql):
        g.add("list_has_any", ["list_has_any"],
              ["Which {ent} have {a} or {b} among their {items}? Return their {label_desc}.",
               "List the {label_desc} of {ent} with at least one of the {items} {a} and {b}."],
              f"SELECT {m.label} FROM {t} WHERE list_has_any({c}, [{lit(a)}, {lit(b)}])",
              f"SELECT {m.label} FROM {t} WHERE list_contains({c}, {lit(a)}) OR list_contains({c}, {lit(b)})",
              ent=m.ent, items=m.items, a=lit(a), b=lit(b), label_desc=m.label_desc)
    for n in (1, 2, 3):
        g.add("list_length_filter", ["len"],
              ["How many {ent} have more than {n} {items}?",
               "Count the {ent} with more than {n} {items}."],
              f"SELECT count(*) FROM {t} WHERE len({c}) > {n}",
              f"SELECT count(*) FROM {t} WHERE (SELECT count(*) FROM unnest({t}.{c})) > {n}",
              ent=m.ent, items=m.items, n=n)
    g.add("list_length_avg", ["len"],
          ["On average, how many {items} does each {ent1} have?",
           "What is the average number of {items} per {ent1}?"],
          f"SELECT avg(len({c})) FROM {t}",
          f"SELECT (SELECT count(*) FROM (SELECT unnest({c}) FROM {t})) / (SELECT count(*) FROM {t})",
          ent1=m.ent1, items=m.items)
    g.add("list_length_rows", ["len"],
          ["For each {ent1}, how many {items} does it have? Return the {label} and the number of {items}.",
           "Show every {ent1} with its number of {items}. Return the {label} and the count."],
          f"SELECT {m.label}, len({c}) FROM {t}",
          f"SELECT t.{m.label}, (SELECT count(*) FROM unnest(t.{c})) FROM {t} AS t",
          ent1=m.ent1, items=m.items, label=m.label)
    g.add("unnest_group_count", ["unnest"],
          ["For each {item}, how many {ent} have it? Return the {item} and the count.",
           "How many {ent} are there per {item}? Return each {item} with its count.",
           "Count the {ent} for every {item}."],
          f"SELECT x, count(*) FROM (SELECT unnest({c}) AS x FROM {t}) GROUP BY x",
          f"SELECT v.x, count(*) FROM (SELECT DISTINCT unnest({c}) AS x FROM {t}) AS v "
          f"JOIN {t} ON list_contains({t}.{c}, v.x) GROUP BY v.x",
          ent=m.ent, item=m.item)
    g.add("unnest_group_count_ordered", ["unnest"],
          ["List each {item} with the number of {ent} that have it, most common first. "
           "Break ties alphabetically.",
           "Rank the {items} by how many {ent} have them, highest count first and ties in alphabetical "
           "order. Return the {item} and the count."],
          f"SELECT x, count(*) AS n FROM (SELECT unnest({c}) AS x FROM {t}) GROUP BY x ORDER BY n DESC, x",
          f"SELECT v.x, count(*) AS n FROM (SELECT DISTINCT unnest({c}) AS x FROM {t}) AS v "
          f"JOIN {t} ON list_contains({t}.{c}, v.x) GROUP BY v.x ORDER BY n DESC, v.x",
          ordered=True, ent=m.ent, item=m.item, items=m.items)
    for direction, word in (("DESC", "most"), ("ASC", "least")):
        g.add("unnest_top", ["unnest"],
              ["Which {item} appears in the {word} {ent}?",
               "What is the {word} common {item} among {ent}?"],
              f"SELECT x FROM (SELECT unnest({c}) AS x FROM {t}) GROUP BY x ORDER BY count(*) {direction} LIMIT 1",
              f"SELECT x FROM (SELECT unnest({c}) AS x FROM {t}) GROUP BY x "
              f"QUALIFY rank() OVER (ORDER BY count(*) {direction}) = 1",
              ent=m.ent, item=m.item, word=word)
    g.add("list_distinct_count", ["unnest"],
          ["How many distinct {items} appear across all {ent}?",
           "How many different {items} are used in {t}?"],
          f"SELECT count(DISTINCT x) FROM (SELECT unnest({c}) AS x FROM {t})",
          f"SELECT len(list_distinct(flatten(list({c})))) FROM {t}",
          ent=m.ent, items=m.items, t=t)
    if m.can_be_empty:
        g.add("list_empty", ["len"],
              ["How many {ent} have no {items}?",
               "Count the {ent} whose list of {items} is empty."],
              f"SELECT count(*) FROM {t} WHERE len({c}) = 0",
              f"SELECT count(*) FROM {t} AS t WHERE NOT EXISTS (SELECT 1 FROM unnest(t.{c}))",
              ent=m.ent, items=m.items)
    for v in g.sample(vals_sql, 3):
        g.add("list_filter_lambda", ["list_filter", "lambda"],
              ["For each {ent1}, list its {items} other than {v}. Return the {label} and the list.",
               "Show the {items} of every {ent1} with {v} removed. Return the {label} and the remaining "
               "{items} as a list."],
              f"SELECT {m.label}, list_filter({c}, lambda x: x != {lit(v)}) FROM {t}",
              # check: rebuild the list with unnest, keeping element order
              f"SELECT t.{m.label}, coalesce((SELECT list(x ORDER BY i) FROM (SELECT unnest(t.{c}) AS x, "
              f"generate_subscripts(t.{c}, 1) AS i) WHERE x <> {lit(v)}), []) FROM {t} AS t",
              ent1=m.ent1, items=m.items, label=m.label, v=lit(v))


# --------------------------------------------------------------------------- struct families
def struct_families(g: Gen, m: StructField) -> None:
    t, ref = m.table, f"{m.col}.{m.field}"
    ext = f"struct_extract({m.col}, '{m.field}')"
    if not m.numeric:
        for v in g.sample(f"SELECT DISTINCT {ref} FROM {t}", 4):
            g.add("struct_field_filter", ["struct"],
                  ["Which {ent} have {v} as their {desc}? Return their {label_desc}.",
                   "List the {label_desc} of {ent} whose {desc} is {v}."],
                  f"SELECT {m.label} FROM {t} WHERE {ref} = {lit(v)}",
                  f"SELECT {m.label} FROM {t} WHERE {ext} = {lit(v)}",
                  ent=m.ent, desc=m.desc, v=lit(v), label_desc=m.label_desc)
        g.add("struct_field_group", ["struct", "group_by_all"],
              ["How many {ent} are there for each {desc}? Return the {desc} and the count.",
               "Count the {ent} per {desc}."],
              f"SELECT {ref}, count(*) FROM {t} GROUP BY ALL",
              f"SELECT {ext} AS f, count(*) FROM {t} GROUP BY f",
              ent=m.ent, desc=m.desc)
    else:
        g.add("struct_field_avg", ["struct"],
              ["What is the average {desc} per {ent1}?",
               "Across all {ent}, what is the mean {desc}?"],
              f"SELECT avg({ref}) FROM {t}",
              f"SELECT sum({ext}) / count({ext}) FROM {t}",
              ent=m.ent, ent1=m.ent1, desc=m.desc)
        qs = g.values(f"SELECT quantile_disc({ref}, [0.25, 0.5, 0.75]) FROM {t}")[0]
        for n in qs:
            g.add("struct_field_threshold", ["struct"],
                  ["Which {ent} have more than {n} {desc}? Return their {label_desc}.",
                   "List the {label_desc} of {ent} with over {n} {desc}."]
                  if m.desc.startswith(("calories", "grams")) else
                  ["Which {ent} have a {desc} above {n}? Return their {label_desc}.",
                   "List the {label_desc} of {ent} whose {desc} is greater than {n}."],
                  f"SELECT {m.label} FROM {t} WHERE {ref} > {n}",
                  f"SELECT {m.label} FROM {t} WHERE {ext} > {n}",
                  ent=m.ent, desc=m.desc, n=n, label_desc=m.label_desc)


# --------------------------------------------------------------------------- time families
def time_families(g: Gen, m: TimeCol) -> None:
    t, c = m.table, m.col
    g.add("date_trunc_month", ["date_trunc"],
          ["How many {ent} were there in each month? Return the first day of the month and the count, "
           "ordered by month.",
           "Show the number of {ent} per month, in date order. Return the month start date and the count."],
          f"SELECT date_trunc('month', {c}) AS month, count(*) FROM {t} GROUP BY month ORDER BY month",
          f"SELECT make_date(year({c}), month({c}), 1) AS m, count(*) FROM {t} GROUP BY m ORDER BY m",
          ordered=True, ent=m.ent)
    g.add("running_total", ["window", "date_trunc"],
          ["Show the running total of {ent} by month: for each month, the first day of the month and the "
           "cumulative count up to and including that month, ordered by month.",
           "What is the cumulative number of {ent} at the end of each month? Return the month start date "
           "and the running total in date order."],
          f"SELECT date_trunc('month', {c}) AS month, sum(count(*)) OVER (ORDER BY date_trunc('month', {c})) "
          f"FROM {t} GROUP BY month ORDER BY month",
          f"SELECT m.month, (SELECT count(*) FROM {t} WHERE {c} < m.month + INTERVAL 1 MONTH) FROM "
          f"(SELECT DISTINCT make_date(year({c}), month({c}), 1) AS month FROM {t}) AS m ORDER BY m.month",
          ordered=True, ent=m.ent)
    months = g.values(f"SELECT DISTINCT year({c}) * 100 + month({c}) FROM {t} ORDER BY 1")
    for ym in g.rng.sample(months, min(4, len(months))):
        y, mo = divmod(ym, 100)
        start = f"{y}-{mo:02d}-01"
        g.add("date_range_count", ["dates"],
              ["How many {ent} were there in {period}?", "Count the {ent} in {period}."],
              f"SELECT count(*) FROM {t} WHERE {c} >= DATE '{start}' AND {c} < DATE '{start}' + INTERVAL 1 MONTH",
              f"SELECT count(*) FROM {t} WHERE year({c}) = {y} AND month({c}) = {mo}",
              ent=m.ent, period=f"{calendar.month_name[mo]} {y}")
    g.add("dayname_group", ["dates"],
          ["How many {ent} were there on each day of the week? Return the day name and the count.",
           "Break down the {ent} by weekday. Return the name of the day and the number of {ent}."],
          f"SELECT dayname({c}) AS day, count(*) FROM {t} GROUP BY day",
          f"SELECT strftime({c}, '%A') AS d, count(*) FROM {t} GROUP BY d",
          ent=m.ent)
    years = g.values(f"SELECT DISTINCT year({c}) FROM {t}")
    if len(years) > 1:
        g.add("year_group", ["dates"],
              ["How many {ent} were there each year? Return the year and the count.",
               "Count the {ent} per year."],
              f"SELECT year({c}) AS y, count(*) FROM {t} GROUP BY y",
              f"SELECT date_part('year', {c}) AS y, count(*) FROM {t} GROUP BY y",
              ent=m.ent)
    if m.is_timestamp:
        g.add("hour_top", ["dates"],
              ["During which hour of the day (0-23) were there the most {ent}?",
               "Which hour of the day has the highest number of {ent}? Return the hour as a number."],
              f"SELECT hour({c}) AS h FROM {t} GROUP BY h ORDER BY count(*) DESC LIMIT 1",
              f"SELECT CAST(strftime({c}, '%H') AS INTEGER) AS h FROM {t} GROUP BY h "
              f"QUALIFY rank() OVER (ORDER BY count(*) DESC) = 1",
              ent=m.ent)
        g.add("busiest_date", ["dates"],
              ["On which date were there the most {ent}?",
               "Which single day had the highest number of {ent}? Return the date."],
              f"SELECT CAST({c} AS DATE) AS d FROM {t} GROUP BY d ORDER BY count(*) DESC LIMIT 1",
              f"SELECT date_trunc('day', {c}) AS d FROM {t} GROUP BY d "
              f"QUALIFY rank() OVER (ORDER BY count(*) DESC) = 1",
              ent=m.ent)


# --------------------------------------------------------------------------- window and aggregate families
def _ret(col: str, desc: str) -> str:
    """How to ask for a grouping column in the answer: "artist id" when it is an id column."""
    return f"{desc} id" if col.endswith("_id") else desc



def latest_families(g: Gen, m: Latest) -> None:
    t = m.table
    for direction, word, agg in (("DESC", "most recent", "max"), ("ASC", "first", "min")):
        g.add("qualify_latest", ["qualify", "window"],
              ["For each {group}, what was the {ret} of the {word} {ent1}? Return the {group_ret} and the {ret}.",
               "Show the {ret} of the {word} {ent1} for every {group}. Return the {group_ret} and the {ret}."],
              f"SELECT {m.group}, {m.ret} FROM {t} "
              f"QUALIFY row_number() OVER (PARTITION BY {m.group} ORDER BY {m.ts} {direction}) = 1",
              f"SELECT t.{m.group}, t.{m.ret} FROM {t} AS t JOIN (SELECT {m.group}, {agg}({m.ts}) AS ts "
              f"FROM {t} GROUP BY {m.group}) AS x ON t.{m.group} = x.{m.group} AND t.{m.ts} = x.ts",
              group=m.group_desc, group_ret=_ret(m.group, m.group_desc), ret=m.ret_desc, word=word,
              ent1=m.ent1)


def argmax_families(g: Gen, m: ArgMax) -> None:
    t = m.table
    for fn, direction, word in (("arg_max", "DESC", m.high_desc), ("arg_min", "ASC", m.low_desc)):
        g.add("arg_max", ["arg_max"],
              ["For each {group}, what is the {ret} of the {word} of its {ent}? Return the {group_ret} and the {ret}.",
               "Which of the {ent} is the {word} for each {group}? Return the {group_ret} and its {ret}."],
              f"SELECT {m.group}, {fn}({m.ret}, {m.val}) FROM {t} GROUP BY {m.group}",
              f"SELECT {m.group}, {m.ret} FROM {t} WHERE {m.val} IS NOT NULL "
              f"QUALIFY rank() OVER (PARTITION BY {m.group} ORDER BY {m.val} {direction}) = 1",
              group=m.group_desc, group_ret=_ret(m.group, m.group_desc), ret=m.ret_desc, word=word, ent=m.ent)
    for n in (2, 3):
        g.add("top_n_per_group", ["qualify", "window"],
              ["For each {group}, list the {n} {word} {ent}. Return the {group_ret} and the {ret}.",
               "Show the top {n} {ent} for each {group}, ranked {word} first. Return the {group_ret} and the {ret}."],
              f"SELECT {m.group}, {m.ret} FROM {t} WHERE {m.val} IS NOT NULL "
              f"QUALIFY row_number() OVER (PARTITION BY {m.group} ORDER BY {m.val} DESC) <= {n}",
              f"SELECT a.{m.group}, a.{m.ret} FROM {t} AS a WHERE a.{m.val} IS NOT NULL AND "
              f"(SELECT count(*) FROM {t} AS b WHERE b.{m.group} = a.{m.group} AND b.{m.val} > a.{m.val}) < {n}",
              group=m.group_desc, group_ret=_ret(m.group, m.group_desc), ret=m.ret_desc, word=m.high_desc,
              ent=m.ent, n=n)


def filtered_families(g: Gen, m: Filtered) -> None:
    g.add("filter_clause", ["filter_clause", "group_by_all"],
          ["For each {group}, how many {ent} are there in total and how many of them {cond}? "
           "Return the {group_ret}, the total and the filtered count.",
           "Per {group}, give the number of {ent} and the number that {cond}."],
          f"SELECT {m.group}, count(*), count(*) FILTER (WHERE {m.cond}) FROM {m.table} GROUP BY ALL",
          f"SELECT {m.group}, count(*), sum(CASE WHEN {m.cond} THEN 1 ELSE 0 END) FROM {m.table} "
          f"GROUP BY {m.group}",
          group=m.group_desc, group_ret=_ret(m.group, m.group_desc), ent=m.ent, cond=m.cond_desc)


def duration_families(g: Gen, m: Duration) -> None:
    t = m.table
    g.add("duration_avg", ["dates"],
          ["On average, how many hours does it take to {verb} {ent}? Only count {ent} that were {end}.",
           "What is the average time in hours from when {ent} are {start} to when they are {end}, "
           "for those that have been {end}?"],
          f"SELECT avg(epoch({m.end} - {m.start})) / 3600 FROM {t} WHERE {m.end} IS NOT NULL",
          f"SELECT avg(date_diff('second', {m.start}, {m.end})) / 3600.0 FROM {t}",
          ent=m.ent, verb=m.verb, start=m.start_word, end=m.end_word)
    g.add("duration_group", ["dates"],
          ["For each {group}, what is the average number of hours taken to {verb} its {ent}? "
           "Ignore {ent} that have not been {end}. Return the {group_ret} and the average hours.",
           "Per {group}, how long does it take on average, in hours, to {verb} {ent}?"],
          f"SELECT {m.group}, avg(epoch({m.end} - {m.start})) / 3600 FROM {t} "
          f"WHERE {m.end} IS NOT NULL GROUP BY {m.group}",
          f"SELECT {m.group}, sum(date_diff('second', {m.start}, {m.end})) / count({m.end}) / 3600.0 "
          f"FROM {t} GROUP BY {m.group} HAVING count({m.end}) > 0",
          group=m.group_desc, group_ret=_ret(m.group, m.group_desc), ent=m.ent, verb=m.verb, end=m.end_word)


# --------------------------------------------------------------------------- schema-specific templates
def custom_music(g: Gen) -> None:
    for v in g.sample("SELECT DISTINCT unnest(tags) FROM tracks", 4):
        g.add("join_list_contains", ["list_contains", "join"],
              ["Which listeners have played at least one track tagged {v}? Return their usernames.",
               "List the usernames of listeners who played a track with the tag {v}."],
              f"SELECT DISTINCT l.username FROM listeners AS l JOIN plays AS p ON p.listener_id = l.listener_id "
              f"JOIN tracks AS t ON t.track_id = p.track_id WHERE list_contains(t.tags, {lit(v)})",
              f"SELECT username FROM listeners WHERE listener_id IN (SELECT p.listener_id FROM plays AS p, "
              f"tracks AS t, unnest(t.tags) AS u(tag) WHERE p.track_id = t.track_id AND u.tag = {lit(v)})",
              v=lit(v))
    g.add("join_unnest_group", ["unnest", "join"],
          ["How many plays did each genre get? A play counts for every genre of the track's artist. "
           "Return the genre and the number of plays.",
           "Count plays per artist genre. Return the genre and the play count."],
          "SELECT g.genre, count(*) FROM plays AS p JOIN tracks AS t USING (track_id) "
          "JOIN artists AS a USING (artist_id), unnest(a.genres) AS g(genre) GROUP BY g.genre",
          "SELECT x.genre, count(*) FROM plays AS p JOIN tracks AS t ON p.track_id = t.track_id JOIN "
          "(SELECT artist_id, unnest(genres) AS genre FROM artists) AS x ON x.artist_id = t.artist_id "
          "GROUP BY x.genre")
    for plan in ("free", "premium", "family"):
        g.add("date_diff", ["dates", "join"],
              ["For listeners on the {plan} plan, what is the average number of days between signing up "
               "and their first play?",
               "On average, how many days after signing up did {plan}-plan listeners first play a track?"],
              f"SELECT avg(date_diff('day', l.signup_date, f.first_play)) FROM listeners AS l JOIN "
              f"(SELECT listener_id, min(played_at) AS first_play FROM plays GROUP BY listener_id) AS f "
              f"USING (listener_id) WHERE l.plan = {lit(plan)}",
              f"SELECT avg(CAST(f.first_play AS DATE) - l.signup_date) FROM listeners AS l JOIN "
              f"(SELECT listener_id, CAST(min(played_at) AS DATE) AS first_play FROM plays GROUP BY listener_id) "
              f"AS f ON f.listener_id = l.listener_id WHERE l.plan = {lit(plan)}",
              plan=plan)
    g.add("lag", ["window"],
          ["For each month, how many plays were there and how did that change from the previous month? "
           "Return the month start date, the count and the difference from the month before "
           "(NULL for the first month), ordered by month.",
           "Show the month-over-month change in the number of plays: month start date, plays, and the "
           "difference from the previous month, in date order."],
          "SELECT date_trunc('month', played_at) AS m, count(*) AS n, n - lag(n) OVER (ORDER BY m) "
          "FROM plays GROUP BY m ORDER BY m",
          "SELECT a.m, a.n, a.n - b.n FROM (SELECT make_date(year(played_at), month(played_at), 1) AS m, "
          "count(*) AS n FROM plays GROUP BY m) AS a LEFT JOIN (SELECT make_date(year(played_at), "
          "month(played_at), 1) AS m, count(*) AS n FROM plays GROUP BY m) AS b "
          "ON b.m = a.m - INTERVAL 1 MONTH ORDER BY a.m",
          ordered=True)
    g.add("string_agg", ["list_aggregate"],
          ["For each artist, list the titles of their tracks in alphabetical order as a single list. "
           "Return the artist name and the list.",
           "Give every artist with tracks together with an alphabetically sorted list of their track titles."],
          "SELECT a.name, list(t.title ORDER BY t.title) FROM artists AS a JOIN tracks AS t USING (artist_id) "
          "GROUP BY a.name",
          "SELECT a.name, list_sort(list(t.title)) FROM artists AS a JOIN tracks AS t "
          "ON t.artist_id = a.artist_id GROUP BY a.name")


def custom_recipes(g: Gen) -> None:
    names = "SELECT DISTINCT i.name FROM recipes, unnest(ingredients) AS u(i)"
    for v in g.sample(names, 6):
        g.add("struct_list_contains", ["list_of_struct", "lambda"],
              ["Which recipes use {v}? Return the recipe names.",
               "List the names of recipes that have {v} as an ingredient."],
              f"SELECT name FROM recipes WHERE list_contains(list_transform(ingredients, lambda i: i.name), {lit(v)})",
              f"SELECT r.name FROM recipes AS r WHERE EXISTS (SELECT 1 FROM unnest(r.ingredients) AS u(i) "
              f"WHERE u.i.name = {lit(v)})",
              v=lit(v))
    for v in g.sample(names, 6):
        g.add("struct_list_sum", ["list_of_struct", "unnest"],
              ["How many grams of {v} are used across all recipes in total?",
               "What is the total weight in grams of {v} over every recipe?"],
              f"SELECT sum(u.i.grams) FROM recipes, unnest(ingredients) AS u(i) WHERE u.i.name = {lit(v)}",
              f"SELECT sum(list_sum(list_transform(list_filter(ingredients, lambda i: i.name = {lit(v)}), "
              f"lambda i: i.grams))) FROM recipes",
              v=lit(v))
    for cuisine in g.sample("SELECT DISTINCT cuisine FROM recipes", 4):
        g.add("struct_list_total", ["list_of_struct", "lambda"],
              ["For each {c} recipe, what is the total weight of its ingredients in grams? "
               "Return the recipe name and the total.",
               "Show the {c} recipes with the combined grams of all their ingredients."],
              f"SELECT name, list_sum(list_transform(ingredients, lambda i: i.grams)) FROM recipes "
              f"WHERE cuisine = {lit(cuisine)}",
              f"SELECT r.name, (SELECT sum(u.i.grams) FROM unnest(r.ingredients) AS u(i)) FROM recipes AS r "
              f"WHERE r.cuisine = {lit(cuisine)}",
              c=cuisine)
    g.add("struct_list_top", ["list_of_struct", "unnest"],
          ["Which ingredient is used in the most recipes?",
           "What is the most common ingredient across all recipes?"],
          "SELECT u.i.name FROM recipes, unnest(ingredients) AS u(i) GROUP BY u.i.name ORDER BY count(*) DESC LIMIT 1",
          "SELECT x FROM (SELECT unnest(list_transform(ingredients, lambda i: i.name)) AS x FROM recipes) "
          "GROUP BY x QUALIFY rank() OVER (ORDER BY count(*) DESC) = 1")
    for v, grams in [(v, g.rng.choice([100, 200, 300])) for v in g.sample(names, 4)]:
        g.add("struct_list_filter", ["list_of_struct", "lambda"],
              ["Which recipes use more than {n} grams of {v}? Return the recipe names.",
               "List the recipes that call for over {n} g of {v}."],
              f"SELECT name FROM recipes WHERE len(list_filter(ingredients, lambda i: i.name = {lit(v)} "
              f"AND i.grams > {grams})) > 0",
              f"SELECT DISTINCT r.name FROM recipes AS r, unnest(r.ingredients) AS u(i) "
              f"WHERE u.i.name = {lit(v)} AND u.i.grams > {grams}",
              v=lit(v), n=grams)
    g.add("struct_group_join", ["struct", "join"],
          ["Across all reviews, what is the average rating and the average recipe calories for each cuisine? "
           "Return the cuisine, the average rating and the average calories.",
           "Per cuisine, give the average review rating and the average calories of the reviewed recipes, "
           "counting each review once."],
          "SELECT r.cuisine, avg(v.rating), avg(r.nutrition.calories) FROM reviews AS v "
          "JOIN recipes AS r USING (recipe_id) GROUP BY ALL",
          "SELECT r.cuisine, avg(v.rating), avg(struct_extract(r.nutrition, 'calories')) FROM reviews AS v "
          "JOIN recipes AS r ON r.recipe_id = v.recipe_id GROUP BY r.cuisine")
    g.add("exclude", ["star_exclude"],
          ["Show every column of the quick-tagged recipes except ingredients and nutrition.",
           "Return all recipe columns other than ingredients and nutrition, for recipes tagged 'quick'."],
          "SELECT * EXCLUDE (ingredients, nutrition) FROM recipes WHERE list_contains(tags, 'quick')",
          "SELECT recipe_id, name, cuisine, chef, prep_minutes, servings, tags FROM recipes "
          "WHERE 'quick' IN (SELECT unnest(tags))")


def custom_iot(g: Gen) -> None:
    g.add("struct_group_join", ["struct", "join"],
          ["What is the average temperature reading at each site? Return the site and the average temperature.",
           "Per site, give the mean temperature across all readings from its devices."],
          "SELECT d.location.site, avg(r.temperature) FROM readings AS r JOIN devices AS d USING (device_id) "
          "GROUP BY ALL",
          "SELECT struct_extract(d.location, 'site') AS s, sum(r.temperature) / count(r.temperature) "
          "FROM readings AS r JOIN devices AS d ON d.device_id = r.device_id GROUP BY s")
    for did in g.sample("SELECT device_id FROM devices", 4):
        g.add("daily_avg", ["date_trunc"],
              ["What was the daily average temperature for device {d}? Return the date and the average, "
               "ordered by date.",
               "For device {d}, show each day with its average temperature, in date order."],
              f"SELECT CAST(ts AS DATE) AS day, avg(temperature) FROM readings WHERE device_id = {did} "
              f"GROUP BY day ORDER BY day",
              f"SELECT date_trunc('day', ts) AS day, avg(temperature) FROM readings WHERE device_id = {did} "
              f"GROUP BY day ORDER BY day",
              ordered=True, d=did)
    for sensor in ("humidity", "co2", "motion"):
        g.add("join_list_contains", ["list_contains", "struct", "join"],
              ["How many devices at each site have a {s} sensor? Return the site and the count.",
               "Per site, count the devices that include a {s} sensor."],
              f"SELECT location.site, count(*) FROM devices WHERE list_contains(sensors, '{sensor}') GROUP BY ALL",
              f"SELECT struct_extract(location, 'site') AS s, count(*) FROM devices AS d WHERE EXISTS "
              f"(SELECT 1 FROM unnest(d.sensors) AS u(x) WHERE u.x = '{sensor}') GROUP BY s",
              s=sensor)
    g.add("lag", ["window"],
          ["For device 1, show each reading's time, temperature, and the change from the previous reading "
           "(NULL for the first), ordered by time.",
           "List the readings of device 1 in time order with the temperature difference from the reading "
           "before it."],
          "SELECT ts, temperature, temperature - lag(temperature) OVER (ORDER BY ts) FROM readings "
          "WHERE device_id = 1 ORDER BY ts",
          "SELECT a.ts, a.temperature, a.temperature - (SELECT b.temperature FROM readings AS b "
          "WHERE b.device_id = 1 AND b.ts < a.ts ORDER BY b.ts DESC LIMIT 1) FROM readings AS a "
          "WHERE a.device_id = 1 ORDER BY a.ts",
          ordered=True)


def custom_repos(g: Gen) -> None:
    for v in g.sample("SELECT DISTINCT unnest(labels) FROM issues", 4):
        g.add("join_list_contains", ["list_contains", "join"],
              ["How many open issues labelled {v} does each repository have? Return the repository name "
               "and the count, for repositories with at least one.",
               "Per repository, count the issues with the label {v} that are still open."],
              f"SELECT r.name, count(*) FROM issues AS i JOIN repos AS r USING (repo_id) "
              f"WHERE list_contains(i.labels, {lit(v)}) AND i.closed_at IS NULL GROUP BY r.name",
              f"SELECT r.name, count(*) FROM repos AS r JOIN issues AS i ON i.repo_id = r.repo_id "
              f"WHERE {lit(v)} IN (SELECT unnest(i.labels)) AND i.closed_at IS NULL GROUP BY r.name",
              v=lit(v))
    g.add("struct_group_join", ["struct", "group_by_all"],
          ["What is the total number of stars for repositories owned by users versus organizations? "
           "Return the owner type and the total stars.",
           "Sum the stars of repositories per owner type."],
          "SELECT owner.kind, sum(stars) FROM repos GROUP BY ALL",
          "SELECT struct_extract(owner, 'kind') AS k, sum(stars) FROM repos GROUP BY k")
    g.add("join_unnest_group", ["unnest", "join"],
          ["How many commits were made to repositories with each topic? Return the topic and the number "
           "of commits.",
           "Count commits per repository topic. A commit counts once for every topic of its repository."],
          "SELECT t.topic, count(*) FROM commits AS c JOIN repos AS r USING (repo_id), "
          "unnest(r.topics) AS t(topic) GROUP BY t.topic",
          "SELECT x.topic, count(*) FROM commits AS c JOIN (SELECT repo_id, unnest(topics) AS topic "
          "FROM repos) AS x ON x.repo_id = c.repo_id GROUP BY x.topic")
    for f in g.sample("SELECT DISTINCT unnest(files_changed) FROM commits", 4):
        g.add("join_list_contains", ["list_contains", "join"],
              ["Which authors have changed {f}? Return the distinct author logins.",
               "List every author who made a commit that touched {f}."],
              f"SELECT DISTINCT author FROM commits WHERE list_contains(files_changed, {lit(f)})",
              f"SELECT author FROM commits, unnest(files_changed) AS u(f) WHERE u.f = {lit(f)} GROUP BY author",
              f=lit(f))
    g.add("string_agg", ["list_aggregate", "struct"],
          ["For each organization, list the names of its repositories in alphabetical order. "
           "Return the organization login and the list.",
           "Give every organization owner with a sorted list of its repository names."],
          "SELECT owner.login, list(name ORDER BY name) FROM repos WHERE owner.kind = 'organization' GROUP BY ALL",
          "SELECT struct_extract(owner, 'login') AS o, list_sort(list(name)) FROM repos "
          "WHERE struct_extract(owner, 'kind') = 'organization' GROUP BY o")


CUSTOM = {"music": custom_music, "recipes": custom_recipes, "iot": custom_iot, "repos": custom_repos}


# --------------------------------------------------------------------------- generation and verification
def generate(db_paths: dict[str, Path], seed: int = SEED) -> list[Candidate]:
    cands: list[Candidate] = []
    for i, (db_id, path) in enumerate(db_paths.items()):
        with duckdb.connect(str(path), read_only=True) as con:
            g = Gen(db_id, con, random.Random(seed + i))
            meta = META[db_id]
            for m in meta["lists"]:
                list_families(g, m)
            for m in meta["structs"]:
                struct_families(g, m)
            for m in meta["times"]:
                time_families(g, m)
            for m in meta["latest"]:
                latest_families(g, m)
            for m in meta["argmax"]:
                argmax_families(g, m)
            for m in meta["filtered"]:
                filtered_families(g, m)
            for m in meta["durations"]:
                duration_families(g, m)
            CUSTOM[db_id](g)
            cands.extend(g.out)
    return cands


def _trivial(rows: list) -> bool:
    """A single row of zeros or NULLs (e.g. a count of 0) says little about the query."""
    return len(rows) == 1 and all(v is None or v == 0 for v in rows[0])


def verify(c: Candidate, db_path: Path) -> Candidate:
    try:
        target = run_query(db_path, c.sql)
    except Exception as e:  # noqa: BLE001
        c.status, c.error = "target_error", str(e).splitlines()[0][:300]
        return c
    try:
        check = run_query(db_path, c.check_sql)
    except Exception as e:  # noqa: BLE001
        c.status, c.error = "check_error", str(e).splitlines()[0][:300]
        return c
    c.num_rows = len(target)
    if not target:
        c.status = "empty_result"
    elif _trivial(target):
        c.status = "trivial_result"
    elif not results_match(target, check, c.ordered):
        c.status = "mismatch"
    else:
        c.status = "verified"
    return c


def build(out: Path = OUT) -> dict:
    db_paths = build_all(out / "databases")
    cands = generate(db_paths)
    seen_sql, seen_q = set(), set()
    for c in cands:
        verify(c, db_paths[c.db_id])
        key_sql = (c.db_id, c.sql)
        key_q = (c.db_id, " ".join(c.question.lower().split()))
        if c.status == "verified" and (key_sql in seen_sql or key_q in seen_q):
            c.status = "duplicate"
        seen_sql.add(key_sql)
        seen_q.add(key_q)
    report: dict = {}
    for split in ("train", "validation"):
        rows = [c for c in cands if (c.db_id in VALIDATION_DBS) == (split == "validation")]
        for n, c in enumerate(rows):
            c.id = f"duckdb_{split}_{n:05d}"
        _write_split(out, split, rows, db_paths)
        ok = [c for c in rows if c.status == "verified"]
        report[split] = {
            "total": len(rows),
            "databases": sorted({c.db_id for c in rows}),
            "status": dict(Counter(c.status for c in rows).most_common()),
            "verified_by_family": dict(Counter(c.family for c in ok).most_common()),
            "verified_by_feature": dict(Counter(f for c in ok for f in c.features).most_common()),
        }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    return report


def _write_split(out: Path, split: str, rows: list[Candidate], db_paths: dict[str, Path]) -> None:
    schemas = {db: schema_ddl(p) for db, p in db_paths.items()}
    records = [{"id": c.id, "db_id": c.db_id, "question": c.question, "sql": c.sql,
                "schema": schemas[c.db_id], "ordered": c.ordered, "source": "duckdb_synth",
                "family": c.family, "features": c.features, "check_sql": c.check_sql,
                "num_rows": c.num_rows}
               for c in rows if c.status == "verified"]
    with open(out / f"{split}_rejected.jsonl", "w") as f:
        for c in rows:
            if c.status != "verified":
                f.write(json.dumps(asdict(c)) + "\n")
    jsonl = out / f"{split}.jsonl"
    jsonl.write_text("".join(json.dumps(r) + "\n" for r in records))
    with duckdb.connect() as con:
        con.execute(f"COPY (SELECT * FROM read_json('{jsonl}', format='newline_delimited')) "
                    f"TO '{out / f'{split}.parquet'}' (FORMAT parquet)")


if __name__ == "__main__":
    t0 = time.time()
    print(json.dumps(build(), indent=2))
    print(f"done in {time.time() - t0:.0f}s, output: {OUT}")
