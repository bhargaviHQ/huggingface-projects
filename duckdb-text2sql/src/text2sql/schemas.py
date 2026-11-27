"""Build small, deterministic DuckDB databases used for evaluation.

Run: python -m text2sql.schemas [out_dir]
"""

from __future__ import annotations

import random
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import duckdb

SEED = 42
DEFAULT_DIR = Path(__file__).resolve().parents[2] / "data" / "dbs"

FIRST = ["Ava", "Ben", "Chloe", "Diego", "Emma", "Farah", "Gabe", "Hana", "Ivan", "Jade",
         "Kai", "Lena", "Mateo", "Nora", "Omar", "Priya", "Quinn", "Rosa", "Sam", "Tara",
         "Umar", "Vera", "Wes", "Ximena", "Yusuf", "Zoe", "Arjun", "Bea", "Cole", "Dana"]
LAST = ["Adams", "Brown", "Chen", "Diaz", "Evans", "Fischer", "Garcia", "Hughes", "Ito", "Jones",
        "Khan", "Lopez", "Miller", "Nguyen", "Okafor", "Patel", "Quist", "Rossi", "Smith", "Tanaka",
        "Usman", "Vargas", "Wright", "Xu", "Yang", "Zimmer", "Alvarez", "Baker", "Costa", "Dubois"]
COUNTRIES = ["USA", "Canada", "UK", "Germany", "India", "Brazil", "Japan", "Australia"]


def _unique_names(rng: random.Random, n: int) -> list[str]:
    combos = [f"{f} {l}" for f in FIRST for l in LAST]
    rng.shuffle(combos)
    return combos[:n]


def _rand_date(rng: random.Random, start: date, end: date) -> date:
    return start + timedelta(days=rng.randrange((end - start).days + 1))


# --------------------------------------------------------------------------- ecommerce
PRODUCTS = {
    "Electronics": ["Wireless Mouse", "Mechanical Keyboard", "USB-C Hub", "Noise Cancelling Headphones",
                    "4K Monitor", "Webcam", "Portable SSD", "Bluetooth Speaker"],
    "Books": ["Data Pipelines Handbook", "SQL Cookbook", "Deep Learning Basics", "Clean Code",
              "Statistics Primer", "Designing Data Systems"],
    "Home": ["Coffee Grinder", "Desk Lamp", "Ceramic Mug Set", "Air Purifier", "Standing Desk Mat",
             "French Press", "Throw Blanket"],
    "Sports": ["Yoga Mat", "Running Shoes", "Water Bottle", "Resistance Bands", "Jump Rope",
               "Cycling Gloves"],
    "Clothing": ["Hoodie", "Rain Jacket", "Wool Socks", "Baseball Cap", "Denim Jeans",
                 "Linen Shirt", "Winter Scarf"],
}
PRICE_RANGE = {"Electronics": (19, 450), "Books": (12, 60), "Home": (15, 220),
               "Sports": (8, 140), "Clothing": (10, 160)}


def build_ecommerce(con: duckdb.DuckDBPyConnection, rng: random.Random) -> None:
    con.execute("""
        CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name VARCHAR, email VARCHAR,
                                country VARCHAR, signup_date DATE);
        CREATE TABLE products (product_id INTEGER PRIMARY KEY, name VARCHAR, category VARCHAR,
                               price DECIMAL(10, 2));
        CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_id INTEGER, order_date DATE,
                             status VARCHAR);  -- completed | cancelled | returned | pending
        CREATE TABLE order_items (order_id INTEGER, product_id INTEGER, quantity INTEGER,
                                  unit_price DECIMAL(10, 2));
    """)
    customers = []
    for cid, name in enumerate(_unique_names(rng, 200), start=1):
        email = name.lower().replace(" ", ".") + "@example.com"
        customers.append((cid, name, email, rng.choice(COUNTRIES),
                          _rand_date(rng, date(2023, 1, 1), date(2024, 12, 31))))
    con.executemany("INSERT INTO customers VALUES (?, ?, ?, ?, ?)", customers)

    products, pid = [], 1
    for category, names in PRODUCTS.items():
        lo, hi = PRICE_RANGE[category]
        for pname in names:
            products.append((pid, pname, category, round(rng.uniform(lo, hi), 2)))
            pid += 1
    con.executemany("INSERT INTO products VALUES (?, ?, ?, ?)", products)

    orders, items = [], []
    statuses, weights = ["completed", "cancelled", "returned", "pending"], [75, 10, 7, 8]
    for oid in range(1, 1501):
        cid = rng.randint(1, 180)  # customers 181-200 never order
        signup = customers[cid - 1][4]
        odate = max(signup, _rand_date(rng, date(2024, 1, 1), date(2025, 6, 30)))
        orders.append((oid, cid, odate, rng.choices(statuses, weights)[0]))
        for p in rng.sample(products, rng.randint(1, 4)):
            price = p[3] if rng.random() > 0.15 else round(p[3] * 0.9, 2)
            items.append((oid, p[0], rng.randint(1, 5), price))
    con.executemany("INSERT INTO orders VALUES (?, ?, ?, ?)", orders)
    con.executemany("INSERT INTO order_items VALUES (?, ?, ?, ?)", items)


# --------------------------------------------------------------------------- flights
AIRLINES = [("AA", "American Airlines"), ("DL", "Delta Air Lines"), ("UA", "United Airlines"),
            ("WN", "Southwest Airlines"), ("AS", "Alaska Airlines"), ("B6", "JetBlue Airways")]
AIRPORTS = [("SEA", "Seattle", "WA"), ("SFO", "San Francisco", "CA"), ("LAX", "Los Angeles", "CA"),
            ("JFK", "New York", "NY"), ("ORD", "Chicago", "IL"), ("ATL", "Atlanta", "GA"),
            ("DFW", "Dallas", "TX"), ("DEN", "Denver", "CO"), ("BOS", "Boston", "MA"),
            ("MIA", "Miami", "FL"), ("PHX", "Phoenix", "AZ"), ("AUS", "Austin", "TX")]


def build_flights(con: duckdb.DuckDBPyConnection, rng: random.Random) -> None:
    con.execute("""
        CREATE TABLE airlines (airline_code VARCHAR PRIMARY KEY, airline_name VARCHAR);
        CREATE TABLE airports (airport_code VARCHAR PRIMARY KEY, city VARCHAR, state VARCHAR);
        CREATE TABLE flights (flight_id INTEGER PRIMARY KEY, airline_code VARCHAR, origin VARCHAR,
                              dest VARCHAR, scheduled_departure TIMESTAMP,
                              dep_delay_minutes INTEGER,  -- NULL when cancelled; negative = early
                              cancelled BOOLEAN, distance_miles INTEGER);
    """)
    con.executemany("INSERT INTO airlines VALUES (?, ?)", AIRLINES)
    con.executemany("INSERT INTO airports VALUES (?, ?, ?)", AIRPORTS)
    codes = [a[0] for a in AIRPORTS]
    distance = {}
    for i, a in enumerate(codes):
        for b in codes[i + 1:]:
            distance[(a, b)] = distance[(b, a)] = rng.randrange(200, 2800, 10)
    # skew airline volume so per-airline aggregates are distinct
    airline_w = [30, 26, 22, 18, 12, 8]
    origin_w = [16, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3]
    rows = []
    for fid in range(1, 3001):
        origin = rng.choices(codes, origin_w)[0]
        dest = rng.choice([c for c in codes if c != origin])
        day = _rand_date(rng, date(2025, 1, 1), date(2025, 3, 31))
        ts = datetime(day.year, day.month, day.day, rng.randint(5, 22), rng.randrange(0, 60, 5))
        cancelled = rng.random() < 0.03
        delay = None if cancelled else max(-15, int(rng.gauss(8, 28)))
        rows.append((fid, rng.choices([a[0] for a in AIRLINES], airline_w)[0], origin, dest, ts,
                     delay, cancelled, distance[(origin, dest)]))
    con.executemany("INSERT INTO flights VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)


# --------------------------------------------------------------------------- hr
DEPARTMENTS = [(1, "Executive", "New York"), (2, "Engineering", "Seattle"), (3, "Sales", "Chicago"),
               (4, "Marketing", "New York"), (5, "Finance", "New York"), (6, "Data", "Seattle"),
               (7, "Support", "Austin"), (8, "People", "Chicago")]
DEPT_SIZE = {1: 1, 2: 34, 3: 22, 4: 13, 5: 10, 6: 17, 7: 15, 8: 8}  # sums to 120
TITLES = ["Associate", "Specialist", "Senior Specialist", "Lead"]


def build_hr(con: duckdb.DuckDBPyConnection, rng: random.Random) -> None:
    con.execute("""
        CREATE TABLE departments (department_id INTEGER PRIMARY KEY, department_name VARCHAR,
                                  location VARCHAR);
        CREATE TABLE employees (employee_id INTEGER PRIMARY KEY, name VARCHAR, department_id INTEGER,
                                manager_id INTEGER,  -- NULL for the CEO
                                title VARCHAR, hire_date DATE, salary DECIMAL(10, 2));
    """)
    con.executemany("INSERT INTO departments VALUES (?, ?, ?)", DEPARTMENTS)
    names = iter(_unique_names(rng, 120))
    rows = [(1, next(names), 1, None, "CEO", date(2015, 3, 2), 350000)]
    eid = 2
    for dept_id in range(2, 9):
        head = eid
        rows.append((head, next(names), dept_id, 1, "Director",
                     _rand_date(rng, date(2015, 6, 1), date(2019, 12, 31)),
                     rng.randrange(200000, 260000, 500)))
        eid += 1
        for _ in range(DEPT_SIZE[dept_id] - 1):
            rows.append((eid, next(names), dept_id, head, rng.choice(TITLES),
                         _rand_date(rng, date(2016, 1, 1), date(2025, 10, 31)),
                         rng.randrange(50000, 180000, 500)))
            eid += 1
    # a few employees out-earn their director (useful self-join case)
    rows = [r if r[0] not in (5, 40, 77) else (*r[:6], 265000) for r in rows]
    con.executemany("INSERT INTO employees VALUES (?, ?, ?, ?, ?, ?, ?)", rows)


# --------------------------------------------------------------------------- web analytics
INTERESTS = ["ai", "data", "cloud", "security", "design", "gaming", "finance"]
PAGES = ["/home", "/pricing", "/docs", "/blog", "/signup", "/login", "/checkout", "/about"]
PAGE_W = [10, 6, 8, 7, 3, 5, 2, 4]


def build_web(con: duckdb.DuckDBPyConnection, rng: random.Random) -> None:
    con.execute("""
        CREATE TABLE users (user_id INTEGER PRIMARY KEY, username VARCHAR, country VARCHAR,
                            interests VARCHAR[]);
        CREATE TABLE sessions (session_id INTEGER PRIMARY KEY, user_id INTEGER, device VARCHAR,
                               started_at TIMESTAMP, ended_at TIMESTAMP,
                               pages VARCHAR[]);  -- ordered list of page paths visited
    """)
    users = []
    for uid, name in enumerate(_unique_names(rng, 150), start=1):
        username = name.split()[0].lower() + str(rng.randint(10, 99)) + f"_{uid}"
        users.append((uid, username, rng.choice(COUNTRIES), rng.sample(INTERESTS, rng.randint(1, 3))))
    con.executemany("INSERT INTO users VALUES (?, ?, ?, ?)", users)
    sessions = []
    for sid in range(1, 1201):
        start = datetime(2025, 5, 1) + timedelta(days=rng.randrange(31), hours=rng.randrange(24),
                                                 minutes=rng.randrange(60))
        end = start + timedelta(minutes=rng.randint(1, 90))
        n = rng.randint(1, 8)
        pages = (["/home"] if rng.random() < 0.7 else []) + rng.choices(PAGES, PAGE_W, k=n)
        sessions.append((sid, rng.randint(1, 150), rng.choices(["desktop", "mobile", "tablet"], [5, 4, 1])[0],
                         start, end, pages[:8]))
    con.executemany("INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?)", sessions)


BUILDERS = {"ecommerce": build_ecommerce, "flights": build_flights, "hr": build_hr, "web": build_web}


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


def schema_ddl(db_path: Path) -> str:
    """CREATE TABLE statements for every table in the database (used in prompts)."""
    with duckdb.connect(str(db_path), read_only=True) as con:
        rows = con.execute("SELECT sql FROM duckdb_tables() ORDER BY table_name").fetchall()
    return "\n".join(r[0] + ";" for r in rows)


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DIR
    for name, p in build_all(out).items():
        print(f"built {name}: {p}")
