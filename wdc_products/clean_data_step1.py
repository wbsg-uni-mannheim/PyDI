#!/usr/bin/env python3
from pathlib import Path
import json
import os
import sqlite3
import sys
try:
    import orjson
except ModuleNotFoundError:
    orjson = None
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = None

def main() -> int:
    use_fts = True
    use_tqdm = False
    data_dir = Path("./data")
    files = sorted(data_dir.glob("*.jsonl"))
    if not files:
        print(f"No .jsonl files found in: {data_dir.resolve()}", file=sys.stderr)
        return 2

    db_path = data_dir / "dedup.sqlite"

    # Choose dedup behavior:
    # - "INSERT OR IGNORE": first record for an id wins
    # - "INSERT OR REPLACE": last record for an id wins
    UPSERT = "INSERT OR IGNORE"

    con = sqlite3.connect(db_path)
    cur = con.cursor()

    # Pragmas: better performance while remaining safe enough for this use case
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS records (
      id TEXT PRIMARY KEY,
      payload TEXT NOT NULL,
      cluster_id TEXT,
      title TEXT,
      description TEXT
    )
    """)
    columns = [row[1] for row in cur.execute("PRAGMA table_info(records)")]
    if "cluster_id" not in columns:
        cur.execute("ALTER TABLE records ADD COLUMN cluster_id TEXT")
    if "title" not in columns:
        cur.execute("ALTER TABLE records ADD COLUMN title TEXT")
    if "description" not in columns:
        cur.execute("ALTER TABLE records ADD COLUMN description TEXT")

    if use_fts:
        fts_exists = cur.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='records_fts'"
        ).fetchone()
        if not fts_exists:
            cur.execute("""
            CREATE VIRTUAL TABLE records_fts USING fts5(
              title, description, content='records', content_rowid='rowid'
            )
            """)
        # Disable FTS triggers during bulk load for speed.
        cur.execute("DROP TRIGGER IF EXISTS records_ai")
        cur.execute("DROP TRIGGER IF EXISTS records_ad")
        cur.execute("DROP TRIGGER IF EXISTS records_au")
        con.commit()

    # Single transaction for faster bulk inserts.
    con.execute("BEGIN")
    BATCH_SIZE = 1000_000
    batch = []
    total_lines = 0
    skipped_no_id = 0
    skipped_bad_json = 0
    loads = orjson.loads if orjson else json.loads
    tqdm_iter = (lambda it, desc: tqdm(it, desc=desc, unit="lines")) if (tqdm and use_tqdm) else (lambda it, desc: it)
    last_log_at = 0
    log_every = 500_000

    for f in files:
        with f.open("r", encoding="utf-8") as fh:
            for line in tqdm_iter(fh, f.name):
                total_lines += 1
                if total_lines - last_log_at >= log_every:
                    print(f"Read {total_lines} lines...", file=sys.stderr)
                    last_log_at = total_lines
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = loads(line)
                except (json.JSONDecodeError, ValueError):
                    skipped_bad_json += 1
                    continue

                _id = obj.get("id")
                if _id is None:
                    skipped_no_id += 1
                    continue
                if obj.get("description") is None:
                    continue

                cluster_id = obj.get("cluster_id")
                cluster_id_value = None if cluster_id is None else str(cluster_id)
                title_value = obj.get("title")
                description_value = obj.get("description")
                batch.append((
                    str(_id),
                    json.dumps(obj, ensure_ascii=False),
                    cluster_id_value,
                    title_value,
                    description_value,
                ))

                if len(batch) >= BATCH_SIZE:
                    print(f"Flushing batch of {len(batch)} rows...", file=sys.stderr)
                    cur.executemany(
                        f"{UPSERT} INTO records(id,payload,cluster_id,title,description) "
                        "VALUES (?,?,?,?,?)",
                        batch,
                    )
                    batch.clear()

    if batch:
        cur.executemany(
            f"{UPSERT} INTO records(id,payload,cluster_id,title,description) "
            "VALUES (?,?,?,?,?)",
            batch,
        )
        batch.clear()

    # Drop clusters with fewer than 4 entries (including NULL cluster_id).
    cur.execute("""
    DELETE FROM records
    WHERE cluster_id IS NULL
       OR cluster_id IN (
           SELECT cluster_id
           FROM records
           WHERE cluster_id IS NOT NULL
           GROUP BY cluster_id
           HAVING COUNT(*) < 4
       )
    """)
    con.commit()

    if use_fts:
        # Rebuild FTS and restore triggers after bulk load.
        cur.execute("INSERT INTO records_fts(records_fts) VALUES('rebuild')")
        cur.execute("""
        CREATE TRIGGER IF NOT EXISTS records_ai AFTER INSERT ON records BEGIN
          INSERT INTO records_fts(rowid, title, description)
          VALUES (new.rowid, new.title, new.description);
        END;
        """)
        cur.execute("""
        CREATE TRIGGER IF NOT EXISTS records_ad AFTER DELETE ON records BEGIN
          INSERT INTO records_fts(records_fts, rowid, title, description)
          VALUES('delete', old.rowid, old.title, old.description);
        END;
        """)
        cur.execute("""
        CREATE TRIGGER IF NOT EXISTS records_au AFTER UPDATE ON records BEGIN
          INSERT INTO records_fts(records_fts, rowid, title, description)
          VALUES('delete', old.rowid, old.title, old.description);
          INSERT INTO records_fts(rowid, title, description)
          VALUES (new.rowid, new.title, new.description);
        END;
        """)
        con.commit()

    # Basic stats
    unique_ids = cur.execute("SELECT COUNT(*) FROM records").fetchone()[0]
    con.close()

    print(f"Input files: {len(files)}")
    print(f"Lines read: {total_lines}")
    print(f"Unique ids written: {unique_ids}")
    if skipped_no_id:
        print(f"Skipped (missing id): {skipped_no_id}")
    if skipped_bad_json:
        print(f"Skipped (bad json): {skipped_bad_json}")
    print(f"SQLite DB (cache): {db_path.resolve()}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
