#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from pathlib import Path


category_keyword_mapping = {
    "gpu": ["gpu", "graphics card", "video card", "nvidia", "amd radeon"],
    "ssd": ["ssd", "solid state drive", "nvme", "m.2"],
    "hdd": ["hdd", "hard drive", "hard disk drive"],
    "sticks": ["stick", "flash drive", "usb drive", "thumb drive"],
}


def build_fts_query(keywords: list[str]) -> str:
    parts = []
    for kw in keywords:
        kw = kw.strip().replace('"', '""')
        parts.append(f'"{kw}"')
    return " OR ".join(parts)


def fetch_clusters_fts(cur: sqlite3.Cursor, query: str) -> set[str]:
    rows = cur.execute(
        """
        SELECT DISTINCT cluster_id
        FROM records
        WHERE cluster_id IS NOT NULL
          AND rowid IN (
              SELECT rowid FROM records_fts WHERE records_fts MATCH ?
          )
        """,
        (query,),
    )
    return {row[0] for row in rows}


def fetch_clusters_like(cur: sqlite3.Cursor, keywords: list[str]) -> set[str]:
    clauses = []
    params: list[str] = []
    for kw in keywords:
        kw = kw.lower()
        like_param = f"%{kw}%"
        clauses.append(
            "(LOWER(COALESCE(title,'')) LIKE ? OR LOWER(COALESCE(description,'')) LIKE ?)"
        )
        params.extend([like_param, like_param])
    where = " OR ".join(clauses)
    rows = cur.execute(
        f"""
        SELECT DISTINCT cluster_id
        FROM records
        WHERE cluster_id IS NOT NULL
          AND ({where})
        """,
        params,
    )
    return {row[0] for row in rows}


def init_output_schema(cur: sqlite3.Cursor, schema: str) -> None:
    if not schema.isidentifier():
        raise ValueError("Invalid schema name")
    cur.execute(f"""
    CREATE TABLE IF NOT EXISTS {schema}.records (
      id TEXT PRIMARY KEY,
      payload TEXT NOT NULL,
      cluster_id TEXT,
      title TEXT,
      description TEXT
    )
    """)
    cur.execute(f"""
    CREATE VIRTUAL TABLE IF NOT EXISTS {schema}.records_fts USING fts5(
      title, description, content='records', content_rowid='rowid'
    )
    """)
    cur.execute(f"""
    CREATE TRIGGER IF NOT EXISTS {schema}.records_ai AFTER INSERT ON {schema}.records BEGIN
      INSERT INTO records_fts(rowid, title, description)
      VALUES (new.rowid, new.title, new.description);
    END;
    """)
    cur.execute(f"""
    CREATE TRIGGER IF NOT EXISTS {schema}.records_ad AFTER DELETE ON {schema}.records BEGIN
      INSERT INTO records_fts(records_fts, rowid, title, description)
      VALUES('delete', old.rowid, old.title, old.description);
    END;
    """)
    cur.execute(f"""
    CREATE TRIGGER IF NOT EXISTS {schema}.records_au AFTER UPDATE ON {schema}.records BEGIN
      INSERT INTO records_fts(records_fts, rowid, title, description)
      VALUES('delete', old.rowid, old.title, old.description);
      INSERT INTO records_fts(rowid, title, description)
      VALUES (new.rowid, new.title, new.description);
    END;
    """)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db",
        default=Path("./data/dedup.sqlite"),
        type=Path,
        help="Input SQLite DB from clean_data.py",
    )
    parser.add_argument(
        "--out",
        default=Path("./data/dedup_filtered.sqlite"),
        type=Path,
        help="Output SQLite DB with filtered clusters",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output DB if it already exists",
    )
    parser.add_argument(
        "--reset-out",
        action="store_true",
        help="Drop and recreate output schema if it already exists",
    )
    parser.add_argument(
        "--sample-clusters",
        type=int,
        default=0,
        help="Sample N clusters from the filtered set and write JSON output",
    )
    parser.add_argument(
        "--sample-out",
        type=Path,
        default=None,
        help="Output JSON file for sampled clusters",
    )
    args = parser.parse_args()

    if args.out.resolve() == args.db.resolve():
        print("Refusing to overwrite the input DB. Choose a different --out path.", file=sys.stderr)
        return 2
    if (args.sample_clusters and args.sample_out is None) or (
        args.sample_out is not None and args.sample_clusters <= 0
    ):
        print("Use --sample-clusters N together with --sample-out PATH.", file=sys.stderr)
        return 2

    if args.out.exists():
        if not args.overwrite:
            print(f"Output DB already exists: {args.out}. Use --overwrite to replace it.", file=sys.stderr)
            return 2
        args.out.unlink()

    con = sqlite3.connect(args.db)
    cur = con.cursor()

    fts_exists = cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='records_fts'"
    ).fetchone()

    if fts_exists:
        clusters_by_category = {
            name: fetch_clusters_fts(cur, build_fts_query(keywords))
            for name, keywords in category_keyword_mapping.items()
        }
    else:
        clusters_by_category = {
            name: fetch_clusters_like(cur, keywords)
            for name, keywords in category_keyword_mapping.items()
        }

    keep_clusters = set()
    for cluster_id in set().union(*clusters_by_category.values()):
        matches = sum(
            1 for clusters in clusters_by_category.values() if cluster_id in clusters
        )
        if matches == 1:
            keep_clusters.add(cluster_id)
    keep_clusters = sorted(keep_clusters)
    if not keep_clusters:
        print("No clusters matched both GPU and memory keywords.", file=sys.stderr)
        return 1

    cur.execute("ATTACH DATABASE ? AS out", (str(args.out),))
    init_output_schema(cur, "out")
    if args.reset_out:
        cur.execute("DELETE FROM out.records")
        cur.execute("DELETE FROM out.records_fts")
        cur.execute("INSERT INTO out.records_fts(records_fts) VALUES('rebuild')")

    cur.execute("CREATE TEMP TABLE keep_clusters (cluster_id TEXT PRIMARY KEY)")
    cur.executemany(
        "INSERT INTO keep_clusters(cluster_id) VALUES (?)",
        [(cid,) for cid in keep_clusters],
    )
    cur.execute("""
    INSERT INTO out.records(id, payload, cluster_id, title, description)
    SELECT id, payload, cluster_id, title, description
    FROM records
    WHERE cluster_id IN (SELECT cluster_id FROM keep_clusters)
    """)
    cur.execute("INSERT INTO out.records_fts(records_fts) VALUES('rebuild')")
    con.commit()

    kept_records = cur.execute("SELECT COUNT(*) FROM out.records").fetchone()[0]

    if args.sample_clusters:
        sample_size = min(args.sample_clusters, len(keep_clusters))
        sampled = random.sample(keep_clusters, sample_size)
        rows = cur.execute(
            """
            SELECT cluster_id, payload
            FROM records
            WHERE cluster_id IN (SELECT cluster_id FROM keep_clusters)
              AND cluster_id IN ({})
            ORDER BY cluster_id
            """.format(",".join("?" for _ in sampled)),
            sampled,
        )
        clustered: dict[str, list[dict]] = {}
        for cluster_id, payload in rows:
            clustered.setdefault(cluster_id, []).append(json.loads(payload))
        output = [
            {"cluster_id": cid, "offers": clustered.get(cid, [])} for cid in sampled
        ]
        args.sample_out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    con.close()

    print(f"Clusters kept: {len(keep_clusters)}")
    print(f"Records kept: {kept_records}")
    print(f"Output DB: {args.out.resolve()}")
    if args.sample_clusters:
        print(f"Sample output: {args.sample_out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
