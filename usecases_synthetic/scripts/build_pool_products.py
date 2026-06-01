"""Build the products pool directly from ``cluster_id``.

Products is structurally different from companies / games / music: it
ships with native cluster membership in every source record
(``cluster_id`` field). Records sharing the same ``cluster_id`` across
sources are matches by construction, which makes the standard
``build_pool.py`` two-pipeline merger (PLM + rule-based) unnecessary —
the pool is derivable directly from cluster membership.

Outputs the same schema as ``build_pool.py`` so downstream consumers
(K2 niche-density dispatcher, EM pool-agreement diagnostic) need no
special case for products:

* ``pools/products/pooled_positives.csv`` with columns
  ``id1, id2, source_1, source_2, pool_agreement`` (canonical-pair
  ordering by lexicographic ``(id1, id2)`` sort within each row).
* ``pools/products/pool_stats.json`` with cluster-size distribution,
  egregious-cluster filter telemetry, and per-pair counts.

``pool_agreement`` is the number of distinct sources participating in
the cluster (max 4). This mirrors the semantics of the music / games /
companies pool where ``pool_agreement >= 2`` means "two independent
evidence pipelines agreed"; on products, cluster membership replaces
the cross-pipeline agreement, so any cluster spanning >= 2 sources
maps to ``pool_agreement >= 2`` by construction.

Egregious-cluster cap reuses the build_pool.py rule
``max(ceil(P99 of cluster sizes), 3 * n_sources)``; clusters larger
than the cap are dropped to suppress transitive-closure artefacts.

Run::

    source pydi-dev/bin/activate
    python usecases_synthetic/scripts/build_pool_products.py
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from usecases_synthetic.lib.loaders import load_domain_sources  # noqa: E402

DOMAIN = "products"
POOL_DIR = REPO_ROOT / "usecases_synthetic" / "pools" / DOMAIN

logger = logging.getLogger("build_pool_products")


def _canonical_pair(id_a: str, id_b: str) -> tuple[str, str]:
    """Return ``(id1, id2)`` lex-sorted."""
    return (id_a, id_b) if id_a <= id_b else (id_b, id_a)


def _compute_egregious_cap(sizes: list[int], n_sources: int) -> int:
    """Return ``max(ceil(P99), 3 * n_sources)`` floor, mirroring build_pool.py."""
    if not sizes:
        return 3 * n_sources
    p99 = float(np.percentile(sizes, 99))
    return max(math.ceil(p99), 3 * n_sources)


def build_pool_from_cluster_id(
    sources: dict[str, pd.DataFrame],
    *,
    apply_cap: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Build the pool by grouping records on ``cluster_id`` cross-source.

    Parameters
    ----------
    sources : dict
        ``{source_name: DataFrame}`` — each DataFrame must carry
        ``id`` (string) and ``cluster_id`` columns.
    apply_cap : bool
        When True (default) apply the egregious-cluster size cap.

    Returns
    -------
    pool_df, stats
        ``pool_df`` has columns ``id1, id2, source_1, source_2,
        pool_agreement``. ``stats`` is a JSON-serialisable dict.
    """
    n_sources = len(sources)
    if n_sources < 2:
        raise ValueError(
            f"Need at least 2 sources to build a cross-source pool; got {n_sources}"
        )

    rows = []
    for source_name, df in sources.items():
        if "id" not in df.columns or "cluster_id" not in df.columns:
            raise KeyError(
                f"{source_name}: expected 'id' + 'cluster_id' columns, "
                f"got {list(df.columns)}"
            )
        sub = df[["id", "cluster_id"]].copy()
        sub["source"] = source_name
        rows.append(sub)
    records = pd.concat(rows, ignore_index=True)
    records["id"] = records["id"].astype(str)
    records["cluster_id"] = records["cluster_id"].astype(str)

    cluster_sizes = records.groupby("cluster_id").size().tolist()
    cap = _compute_egregious_cap(cluster_sizes, n_sources) if apply_cap else None

    dropped_clusters: list[tuple[str, int]] = []
    pool_rows: list[dict] = []
    per_pair_counts: Counter = Counter()

    for cluster_id, group in records.groupby("cluster_id"):
        size = len(group)
        if cap is not None and size > cap:
            dropped_clusters.append((cluster_id, size))
            continue

        sources_in_cluster = group["source"].unique().tolist()
        if len(sources_in_cluster) < 2:
            # Singleton cluster (only one source); skip — no cross-source
            # match to emit. Products has no singleton clusters today but
            # the guard is cheap.
            continue

        pool_agreement = len(sources_in_cluster)
        items = list(zip(group["id"].tolist(), group["source"].tolist(), strict=True))
        for (id_a, src_a), (id_b, src_b) in combinations(items, 2):
            if src_a == src_b:
                # Same-source pair (cluster has 2 rows from same source)
                # — drop to mirror build_pool.py's same-source filter.
                continue
            id1, id2 = _canonical_pair(id_a, id_b)
            if id1 == id_a:
                source_1, source_2 = src_a, src_b
            else:
                source_1, source_2 = src_b, src_a
            pool_rows.append(
                {
                    "id1": id1,
                    "id2": id2,
                    "source_1": source_1,
                    "source_2": source_2,
                    "pool_agreement": pool_agreement,
                }
            )
            pair_key = tuple(sorted((source_1, source_2)))
            per_pair_counts[pair_key] += 1

    pool_df = pd.DataFrame(
        pool_rows,
        columns=["id1", "id2", "source_1", "source_2", "pool_agreement"],
    )
    # Dedupe — clusters spanning all 4 sources can in principle emit
    # the same (id1, id2) twice if a source has duplicate ids within a
    # cluster. None today, but guard cheaply.
    if not pool_df.empty:
        pool_df = pool_df.drop_duplicates(subset=["id1", "id2"], keep="first")

    cluster_sizes_sorted = sorted(cluster_sizes, reverse=True)
    cluster_size_dist = {
        "available": True,
        "n_components": len(cluster_sizes),
        "max_size": max(cluster_sizes) if cluster_sizes else 0,
        "p95_size": int(np.percentile(cluster_sizes, 95)) if cluster_sizes else 0,
        "p99_size": int(np.percentile(cluster_sizes, 99)) if cluster_sizes else 0,
        "top_sizes": cluster_sizes_sorted[:10],
    }

    agreement_hist = (
        pool_df["pool_agreement"].value_counts().sort_index().to_dict()
        if not pool_df.empty
        else {}
    )

    stats = {
        "domain": DOMAIN,
        "source": "cluster_id (native, no PLM/rule-based pipelines)",
        "n_sources": n_sources,
        "sources": sorted(sources.keys()),
        "pool_size": int(len(pool_df)),
        "cluster_size_distribution": cluster_size_dist,
        "egregious_cluster_cap": cap,
        "egregious_clusters_dropped": [
            {"cluster_id": cid, "size": size} for cid, size in dropped_clusters
        ],
        "pool_agreement_histogram": {int(k): int(v) for k, v in agreement_hist.items()},
        "per_pair_pool_size": {
            f"{a}__{b}": int(v) for (a, b), v in sorted(per_pair_counts.items())
        },
    }
    return pool_df, stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the products pool from cluster_id (replaces the "
            "PLM + rule-based pipelines used for companies/games/music)."
        )
    )
    parser.add_argument(
        "--no-cap",
        action="store_true",
        help="Skip the egregious-cluster size cap (debug only).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    logger.info("Loading products sources via load_domain_sources('%s')", DOMAIN)
    sources = load_domain_sources(DOMAIN)
    for name, df in sources.items():
        logger.info("  %s: %d rows", name, len(df))

    pool_df, stats = build_pool_from_cluster_id(sources, apply_cap=not args.no_cap)

    POOL_DIR.mkdir(parents=True, exist_ok=True)
    pool_path = POOL_DIR / "pooled_positives.csv"
    stats_path = POOL_DIR / "pool_stats.json"
    pool_df.to_csv(pool_path, index=False)
    with stats_path.open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, sort_keys=True)

    logger.info("Wrote %d pool rows -> %s", len(pool_df), pool_path)
    logger.info(
        "Egregious cap=%s; dropped %d clusters",
        stats["egregious_cluster_cap"],
        len(stats["egregious_clusters_dropped"]),
    )
    logger.info("Per-pair pool sizes:")
    for pair, n in stats["per_pair_pool_size"].items():
        logger.info("  %s: %d", pair, n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
