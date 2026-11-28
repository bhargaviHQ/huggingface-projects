"""Deterministic DuckDB databases with nested types (LIST, STRUCT, LIST of STRUCT) and timestamps.

These are used to generate DuckDB-specific training examples. They are deliberately different
from the evaluation databases in schemas.py (ecommerce, flights, hr, web) so the evaluation
stays held out.

Run: python -m text2sql.synth_dbs [out_dir]
"""

from __future__ import annotations

import random
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import duckdb

from .schemas import COUNTRIES, _rand_date, _unique_names

SEED = 1234
DEFAULT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "duckdb_synth" / "databases"


def _unique_combos(rng: random.Random, a: list[str], b: list[str], n: int, sep: str = " ") -> list[str]:
    combos = [f"{x}{sep}{y}" for x in a for y in b]
    rng.shuffle(combos)
    return combos[:n]


def _ts(rng: random.Random, start: datetime, days: int) -> datetime:
    return start + timedelta(days=rng.randrange(days), hours=rng.randrange(24),
                             minutes=rng.randrange(60), seconds=rng.randrange(60))


# --------------------------------------------------------------------------- music streaming
GENRES = ["rock", "pop", "jazz", "hip hop", "electronic", "folk", "classical", "metal", "soul", "indie"]
TRACK_TAGS = ["chill", "upbeat", "acoustic", "live", "instrumental", "remix", "workout", "sad",
              "party", "focus"]
ADJ = ["Velvet", "Silent", "Electric", "Golden", "Broken", "Midnight", "Crimson", "Wild", "Lucky",
       "Hollow", "Neon", "Paper", "Stone", "Glass", "Northern"]
NOUN = ["Owls", "Rivers", "Engines", "Lanterns", "Wolves", "Harbors", "Echoes", "Satellites",
        "Gardens", "Foxes", "Tides", "Machines"]
WORD = ["Light", "Rain", "Summer", "Fire", "Home", "Ocean", "Dream", "City", "Heart", "Road",
        "Night", "Morning", "Shadow", "Signal", "Static", "Window", "Thunder", "Orbit"]


def build_music(con: duckdb.DuckDBPyConnection, rng: random.Random) -> None:
    con.execute("""
        CREATE TABLE artists (artist_id INTEGER PRIMARY KEY, name VARCHAR, country VARCHAR,
                              formed_year INTEGER, genres VARCHAR[]);
        CREATE TABLE tracks (track_id INTEGER PRIMARY KEY, artist_id INTEGER, title VARCHAR,
                             duration_s INTEGER, release_date DATE, explicit BOOLEAN, tags VARCHAR[]);
        CREATE TABLE listeners (listener_id INTEGER PRIMARY KEY, username VARCHAR, country VARCHAR,
                                plan VARCHAR, signup_date DATE);
        CREATE TABLE plays (play_id INTEGER PRIMARY KEY, listener_id INTEGER, track_id INTEGER,
                            played_at TIMESTAMP, seconds_played INTEGER, device VARCHAR);
    """)
    names = _unique_combos(rng, ["The " + a for a in ADJ], NOUN, 60)
    artists = [(i, n, rng.choice(COUNTRIES), rng.randint(1975, 2020), rng.sample(GENRES, rng.randint(1, 3)))
               for i, n in enumerate(names, start=1)]
    con.executemany("INSERT INTO artists VALUES (?, ?, ?, ?, ?)", artists)
    titles = _unique_combos(rng, ADJ + WORD, WORD, 420)
    tracks = []
    for tid, title in enumerate(titles, start=1):
        tracks.append((tid, rng.randint(1, 60), title, rng.randint(95, 420),
                       _rand_date(rng, date(2015, 1, 1), date(2024, 12, 31)), rng.random() < 0.2,
                       rng.sample(TRACK_TAGS, rng.randint(0, 3))))
    con.executemany("INSERT INTO tracks VALUES (?, ?, ?, ?, ?, ?, ?)", tracks)
    listeners = []
    for lid, name in enumerate(_unique_names(rng, 200), start=1):
        listeners.append((lid, name.replace(" ", ".").lower() + str(lid), rng.choice(COUNTRIES),
                          rng.choices(["free", "premium", "family"], [5, 4, 2])[0],
                          _rand_date(rng, date(2022, 1, 1), date(2024, 12, 31))))
    con.executemany("INSERT INTO listeners VALUES (?, ?, ?, ?, ?)", listeners)
    durations = {t[0]: t[3] for t in tracks}
    plays = []
    for pid in range(1, 6001):
        tid = rng.randint(1, 420)
        full = rng.random() < 0.7
        plays.append((pid, rng.randint(1, 200), tid, _ts(rng, datetime(2025, 1, 1), 181),
                      durations[tid] if full else rng.randint(5, durations[tid]),
                      rng.choices(["phone", "desktop", "smart speaker", "car"], [6, 3, 1, 1])[0]))
    con.executemany("INSERT INTO plays VALUES (?, ?, ?, ?, ?, ?)", plays)


# --------------------------------------------------------------------------- recipes
CUISINES = ["Italian", "Mexican", "Indian", "Japanese", "Thai", "French", "Greek", "American"]
DISHES = ["Curry", "Salad", "Soup", "Pasta", "Tacos", "Stew", "Bowl", "Pie", "Risotto", "Stir Fry",
          "Flatbread", "Noodles"]
STYLE = ["Spicy", "Smoky", "Creamy", "Lemon", "Garlic", "Herb", "Roasted", "Crispy", "Sweet",
         "Tangy", "Rustic", "Green", "Golden"]
INGREDIENTS = ["onion", "garlic", "tomato", "rice", "chicken", "tofu", "potato", "carrot", "spinach",
               "lentils", "chickpeas", "cheese", "butter", "olive oil", "flour", "egg", "milk",
               "ginger", "chili", "coconut milk", "mushroom", "bell pepper", "beef", "basil"]
RECIPE_TAGS = ["vegetarian", "vegan", "gluten free", "quick", "spicy", "comfort food",
               "high protein", "budget"]


def build_recipes(con: duckdb.DuckDBPyConnection, rng: random.Random) -> None:
    con.execute("""
        CREATE TABLE recipes (recipe_id INTEGER PRIMARY KEY, name VARCHAR, cuisine VARCHAR,
                              chef VARCHAR, prep_minutes INTEGER, servings INTEGER, tags VARCHAR[],
                              ingredients STRUCT(name VARCHAR, grams INTEGER)[],
                              nutrition STRUCT(calories INTEGER, protein_g DOUBLE, fat_g DOUBLE));
        CREATE TABLE reviews (review_id INTEGER PRIMARY KEY, recipe_id INTEGER, reviewer VARCHAR,
                              rating INTEGER, reviewed_at DATE);
    """)
    chefs = _unique_names(rng, 25)
    recipes = []
    for rid, name in enumerate(_unique_combos(rng, STYLE, DISHES, 150), start=1):
        ingredients = [{"name": n, "grams": rng.randrange(10, 600, 5)}
                       for n in rng.sample(INGREDIENTS, rng.randint(3, 8))]
        nutrition = {"calories": rng.randrange(150, 1100, 10), "protein_g": round(rng.uniform(2, 60), 1),
                     "fat_g": round(rng.uniform(1, 55), 1)}
        recipes.append((rid, name, rng.choice(CUISINES), rng.choice(chefs), rng.randrange(10, 125, 5),
                        rng.randint(1, 8), rng.sample(RECIPE_TAGS, rng.randint(0, 3)), ingredients,
                        nutrition))
    con.executemany("INSERT INTO recipes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", recipes)
    reviewers = _unique_names(rng, 120)
    reviews = [(i, rng.randint(1, 150), rng.choice(reviewers), rng.choices([1, 2, 3, 4, 5], [1, 1, 3, 5, 4])[0],
                _rand_date(rng, date(2023, 1, 1), date(2025, 6, 30))) for i in range(1, 951)]
    con.executemany("INSERT INTO reviews VALUES (?, ?, ?, ?, ?)", reviews)


# --------------------------------------------------------------------------- IoT sensors
SITES = ["Austin Plant", "Berlin Office", "Chennai Lab", "Denver Warehouse", "Oslo Office"]
SENSORS = ["temperature", "humidity", "co2", "motion", "light", "noise"]
ALERT_CODES = ["OVERHEAT", "LOW_BATTERY", "OFFLINE", "HUMIDITY_HIGH", "SENSOR_FAULT", "TAMPER"]


def build_iot(con: duckdb.DuckDBPyConnection, rng: random.Random) -> None:
    con.execute("""
        CREATE TABLE devices (device_id INTEGER PRIMARY KEY, name VARCHAR, model VARCHAR,
                              installed_on DATE, location STRUCT(site VARCHAR, floor INTEGER),
                              sensors VARCHAR[]);
        CREATE TABLE readings (device_id INTEGER, ts TIMESTAMP, temperature DOUBLE, humidity DOUBLE);
        CREATE TABLE alerts (alert_id INTEGER PRIMARY KEY, device_id INTEGER, raised_at TIMESTAMP,
                             resolved_at TIMESTAMP, severity VARCHAR, codes VARCHAR[]);
    """)
    devices = []
    for did in range(1, 41):
        devices.append((did, f"sensor-{did:03d}", rng.choice(["TX-100", "TX-200", "AQ-5", "EnviroPro"]),
                        _rand_date(rng, date(2021, 1, 1), date(2024, 12, 31)),
                        {"site": rng.choice(SITES), "floor": rng.randint(1, 6)},
                        ["temperature"] + rng.sample(SENSORS[1:], rng.randint(0, 3))))
    con.executemany("INSERT INTO devices VALUES (?, ?, ?, ?, ?, ?)", devices)
    readings = []
    for did in range(1, 41):
        base = rng.uniform(18, 26)
        offset = timedelta(minutes=rng.randrange(0, 120))
        for k in range(21 * 12):  # every 2 hours for 21 days
            ts = datetime(2025, 3, 1) + offset + timedelta(hours=2 * k, seconds=rng.randrange(60))
            temp = None if rng.random() < 0.02 else round(base + rng.gauss(0, 2.5), 2)
            hum = None if rng.random() < 0.02 else round(rng.uniform(25, 75), 1)
            readings.append((did, ts, temp, hum))
    con.executemany("INSERT INTO readings VALUES (?, ?, ?, ?)", readings)
    alerts = []
    for aid in range(1, 321):
        raised = _ts(rng, datetime(2025, 1, 1), 120)
        resolved = None if rng.random() < 0.15 else raised + timedelta(minutes=rng.randint(5, 2880))
        alerts.append((aid, rng.randint(1, 40), raised, resolved,
                       rng.choices(["low", "medium", "high", "critical"], [5, 4, 2, 1])[0],
                       rng.sample(ALERT_CODES, rng.randint(1, 2))))
    con.executemany("INSERT INTO alerts VALUES (?, ?, ?, ?, ?, ?)", alerts)


# --------------------------------------------------------------------------- code hosting
LANGS = ["Python", "Rust", "Go", "TypeScript", "Java", "C++", "Scala"]
TOPICS = ["machine-learning", "cli", "database", "web", "devops", "data-engineering", "testing",
          "security", "visualization", "api"]
LABELS = ["bug", "enhancement", "documentation", "good first issue", "question", "performance",
          "duplicate", "help wanted"]
REPO_A = ["fast", "tiny", "open", "py", "deep", "auto", "micro", "hyper", "quick", "smart", "go", "data"]
REPO_B = ["parser", "cache", "queue", "lint", "graph", "sync", "server", "store", "flow", "router",
          "bench", "vault"]
FILES = ["README.md", "setup.py", "pyproject.toml", "src/main.py", "src/utils.py", "src/cli.py",
         "tests/test_main.py", "tests/test_utils.py", "docs/index.md", ".github/workflows/ci.yml",
         "Cargo.toml", "src/lib.rs", "go.mod", "main.go", "package.json", "src/index.ts", "Makefile"]
ISSUE_WORDS = ["Crash when", "Support for", "Docs for", "Slow", "Wrong output from", "Add option for",
               "Typo in", "Flaky test in"]


def build_repos(con: duckdb.DuckDBPyConnection, rng: random.Random) -> None:
    con.execute("""
        CREATE TABLE repos (repo_id INTEGER PRIMARY KEY, name VARCHAR,
                            owner STRUCT(login VARCHAR, kind VARCHAR), language VARCHAR,
                            topics VARCHAR[], created_at DATE, stars INTEGER);
        CREATE TABLE commits (sha VARCHAR PRIMARY KEY, repo_id INTEGER, author VARCHAR,
                              committed_at TIMESTAMP, additions INTEGER, deletions INTEGER,
                              files_changed VARCHAR[]);
        CREATE TABLE issues (issue_id INTEGER PRIMARY KEY, repo_id INTEGER, title VARCHAR,
                             author VARCHAR, opened_at TIMESTAMP, closed_at TIMESTAMP,
                             labels VARCHAR[]);
    """)
    logins = [n.split()[0].lower() + n.split()[1][:3].lower() for n in _unique_names(rng, 60)]
    orgs = ["acme-labs", "openfield", "bytecraft", "datanauts", "northwind-dev"]
    repos = []
    for rid, name in enumerate(_unique_combos(rng, REPO_A, REPO_B, 80, sep=""), start=1):
        owner = ({"login": rng.choice(orgs), "kind": "organization"} if rng.random() < 0.4
                 else {"login": rng.choice(logins), "kind": "user"})
        repos.append((rid, name, owner, rng.choice(LANGS), rng.sample(TOPICS, rng.randint(0, 4)),
                      _rand_date(rng, date(2016, 1, 1), date(2024, 6, 30)),
                      int(rng.paretovariate(1.2) * 20)))
    con.executemany("INSERT INTO repos VALUES (?, ?, ?, ?, ?, ?, ?)", repos)
    commits, seen = [], set()
    while len(commits) < 3000:
        sha = "%012x" % rng.getrandbits(48)
        if sha in seen:
            continue
        seen.add(sha)
        commits.append((sha, rng.randint(1, 80), rng.choice(logins), _ts(rng, datetime(2024, 7, 1), 365),
                        rng.randint(0, 400), rng.randint(0, 250), rng.sample(FILES, rng.randint(1, 5))))
    con.executemany("INSERT INTO commits VALUES (?, ?, ?, ?, ?, ?, ?)", commits)
    issues = []
    for iid in range(1, 1201):
        opened = _ts(rng, datetime(2024, 7, 1), 365)
        closed = None if rng.random() < 0.3 else opened + timedelta(hours=rng.randint(1, 24 * 60),
                                                                    minutes=rng.randrange(60))
        issues.append((iid, rng.randint(1, 80), f"{rng.choice(ISSUE_WORDS)} {rng.choice(REPO_B)}",
                       rng.choice(logins), opened, closed, rng.sample(LABELS, rng.randint(0, 3))))
    con.executemany("INSERT INTO issues VALUES (?, ?, ?, ?, ?, ?, ?)", issues)


BUILDERS = {"music": build_music, "recipes": build_recipes, "iot": build_iot, "repos": build_repos}


def build_all(out_dir: Path = DEFAULT_DIR) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for i, (name, builder) in enumerate(BUILDERS.items()):
        path = out_dir / f"{name}.duckdb"
        path.unlink(missing_ok=True)
        with duckdb.connect(str(path)) as con:
            builder(con, random.Random(SEED + i))
        paths[name] = path
    return paths


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DIR
    for name, p in build_all(out).items():
        print(f"built {name}: {p}")
