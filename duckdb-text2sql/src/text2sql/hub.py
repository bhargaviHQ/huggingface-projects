"""Assemble the Spider-DuckDB dataset folder for upload to the Hugging Face Hub.

Run after `python -m text2sql.spider`:
    python -m text2sql.hub
Then upload (requires `hf auth login`):
    hf upload <user>/spider-duckdb data/hf/spider-duckdb --repo-type dataset
"""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

from .spider import OUT

ROOT = Path(__file__).resolve().parents[2]
HF_DIR = ROOT / "data" / "hf" / "spider-duckdb"
CARD_TEMPLATE = ROOT / "dataset_cards" / "spider-duckdb.md"


def build(src: Path = OUT, dst: Path = HF_DIR) -> Path:
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "data").mkdir(parents=True)
    for split in ("train", "validation"):
        shutil.copy(src / f"{split}.parquet", dst / "data" / f"{split}.parquet")
    with zipfile.ZipFile(dst / "databases.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for db in sorted((src / "databases").glob("*.duckdb")):
            z.write(db, f"databases/{db.name}")
    report = json.loads((src / "report.json").read_text())
    card = CARD_TEMPLATE.read_text()
    for split in ("train", "validation"):
        r = report[split]
        card = card.replace(f"{{{split}_total}}", f"{r['total']:,}")
        card = card.replace(f"{{{split}_verified}}", f"{r['status'].get('verified', 0):,}")
    card = card.replace("{stats_table}", stats_table(report))
    (dst / "README.md").write_text(card)
    return dst


def stats_table(report: dict) -> str:
    """Markdown table of pipeline outcomes per split, generated from report.json."""
    rows = [
        ("Spider examples", lambda r: r["total"]),
        ("Kept (verified)", lambda r: r["status"].get("verified", 0)),
        ("  via plain transpile", lambda r: r["verified_by_method"].get("plain", 0)),
        ("  via type-aware transpile", lambda r: r["verified_by_method"].get("typed", 0)
                                                 + r["verified_by_method"].get("qualified", 0)),
        ("Rejected: result mismatch", lambda r: r["status"].get("mismatch", 0)),
        ("Rejected: empty result", lambda r: r["status"].get("empty_result", 0)),
        ("Rejected: DuckDB error", lambda r: r["status"].get("duckdb_error", 0)),
        ("Rejected: SQLite error", lambda r: r["status"].get("sqlite_error", 0)),
        ("Rejected: duplicate question", lambda r: r["status"].get("duplicate", 0)),
    ]
    lines = ["| | train | validation |", "|---|---|---|"]
    for label, get in rows:
        lines.append(f"| {label} | {get(report['train']):,} | {get(report['validation']):,} |")
    lines.append("")
    lines.append("| Rewrite applied (kept examples) | train | validation |")
    lines.append("|---|---|---|")
    for key in ("double_quoted_string", "bare_group_by_column", "like_to_ilike"):
        lines.append(f"| `{key}` | {report['train']['verified_with_rewrite'].get(key, 0):,} "
                     f"| {report['validation']['verified_with_rewrite'].get(key, 0):,} |")
    return "\n".join(lines)


if __name__ == "__main__":
    out = build()
    for p in sorted(out.rglob("*")):
        if p.is_file():
            print(f"{p.stat().st_size / 1e6:8.1f} MB  {p.relative_to(out)}")
