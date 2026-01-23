#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Iterator, Optional


TABLE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?$")


def iter_clustered_rows(
    cur: sqlite3.Cursor,
    table: str,
    include_null: bool,
) -> Iterator[tuple[Optional[str], list[dict[str, Any]]]]:
    if not TABLE_RE.match(table):
        raise ValueError(f"Invalid table name: {table}")
    where_clause = "" if include_null else "WHERE cluster_id IS NOT NULL"
    query = f"""
    SELECT cluster_id, payload
    FROM {table}
    {where_clause}
    ORDER BY cluster_id
    """

    sentinel = object()
    current_id: object = sentinel
    offers: list[dict[str, Any]] = []
    for cluster_id, payload in cur.execute(query):
        if current_id is sentinel:
            current_id = cluster_id
        if cluster_id != current_id:
            yield current_id, offers
            current_id = cluster_id
            offers = []
        offers.append(json.loads(payload))
    if current_id is not sentinel:
        yield current_id, offers


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("./data/dedup_filtered.sqlite"),
        help="Input SQLite DB (records table with payload JSON).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("./clusters_filtered_1.json"),
        help="Output JSON file.",
    )
    parser.add_argument(
        "--table",
        default="records",
        help="Table to read from (use schema.table if needed).",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indent level for JSON output (use 0 for compact).",
    )
    parser.add_argument(
        "--include-null-cluster",
        action="store_true",
        help="Include rows where cluster_id is NULL.",
    )
    args = parser.parse_args()

    if args.out.exists():
        print(f"Output already exists: {args.out}. Remove it or choose another path.", file=sys.stderr)
        return 2

    con = sqlite3.connect(args.db)
    cur = con.cursor()

    indent = None if args.indent == 0 else args.indent
    with args.out.open("w", encoding="utf-8") as out_fh:
        out_fh.write("[\n")
        first = True
        for cluster_id, offers in iter_clustered_rows(
            cur,
            args.table,
            args.include_null_cluster,
        ):
            if not first:
                out_fh.write(",\n")
            obj = {"cluster_id": cluster_id, "offers": offers}
            out_fh.write(json.dumps(obj, ensure_ascii=False, indent=indent))
            first = False
        out_fh.write("\n]\n")

    con.close()
    print(f"JSON output: {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
