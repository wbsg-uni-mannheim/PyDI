#!/usr/bin/env python3
from __future__ import annotations

import json
import sqlite3
from pathlib import Path


def count_jsonl_lines(paths: list[Path]) -> int:
    total = 0
    for path in paths:
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            for _ in fh:
                total += 1
    return total


def count_sqlite_records(path: Path) -> tuple[int, int]:
    con = sqlite3.connect(path)
    cur = con.cursor()
    records = cur.execute("SELECT COUNT(*) FROM records").fetchone()[0]
    clusters = cur.execute(
        "SELECT COUNT(DISTINCT cluster_id) FROM records WHERE cluster_id IS NOT NULL"
    ).fetchone()[0]
    con.close()
    return records, clusters


def count_clusters_json(path: Path) -> tuple[int, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    clusters = len(data)
    offers = sum(len(c.get("offers", [])) for c in data)
    return clusters, offers


def main() -> int:
    base = Path("./")
    data_dir = base / "data"

    jsonl_files = sorted(data_dir.glob("*.jsonl"))
    jsonl_total = count_jsonl_lines(jsonl_files) if jsonl_files else 0

    sqlite_paths = [
        data_dir / "dedup.sqlite",
        data_dir / "dedup_filtered.sqlite",
    ]

    clusters_paths = [
        base / "clusters_filtered_1.json",
        base / "clusters_filtered_2.json",
        base / "clusters_filtered_3.json",
    ]

    print("JSONL inputs")
    print(f"- files: {len(jsonl_files)}")
    print(f"- total lines: {jsonl_total}")
    print("")

    print("SQLite")
    for path in sqlite_paths:
        if not path.exists():
            print(f"- {path.name}: missing")
            continue
        records, clusters = count_sqlite_records(path)
        print(f"- {path.name}: records={records} clusters={clusters}")
    print("")

    print("Cluster JSONs")
    for path in clusters_paths:
        if not path.exists():
            print(f"- {path.name}: missing")
            continue
        clusters, offers = count_clusters_json(path)
        print(f"- {path.name}: clusters={clusters} offers={offers}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
