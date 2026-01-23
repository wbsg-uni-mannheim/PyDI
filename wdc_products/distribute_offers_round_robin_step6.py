#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in",
        dest="input_path",
        type=Path,
        default=Path("clusters_filtered_3.json"),
        help="Input clusters JSON file.",
    )
    parser.add_argument(
        "--out-dir",
        dest="out_dir",
        type=Path,
        default=Path("./"),
        help="Output directory for offers_1..offers_4 JSON files.",
    )
    args = parser.parse_args()

    clusters = json.loads(args.input_path.read_text(encoding="utf-8"))
    buckets: list[list[dict[str, Any]]] = [[], [], [], []]

    for cluster in clusters:
        offers = cluster.get("offers", [])
        for idx, offer in enumerate(offers):
            bucket_index = idx % 4
            buckets[bucket_index].append(offer)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for i, bucket in enumerate(buckets, 1):
        out_path = args.out_dir / f"offers_{i}.json"
        out_path.write_text(
            json.dumps(bucket, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(f"Input clusters: {len(clusters)}")
    for i, bucket in enumerate(buckets, 1):
        print(f"offers_{i}: {len(bucket)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
