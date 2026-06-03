#!/usr/bin/env python3
"""Build the papers pool by matching all potential pairs on DOI.

Like ``build_pool.py`` (companies / games / music) the papers pool is a
"probably-positive, do not perturb" protection set covering *all* likely
cross-source matches over the full source data -- not just the labelled
EM-gold subset. ``build_pool.py`` discovers those matches with a blocker
sweep plus a trained Ditto PLM (+ LLM adjudication in the margin band);
papers instead has a strong deterministic identity key -- the **DOI** --
so no Ditto is needed. Every record in all three sources carries a
unique, 100%-populated DOI, so two cross-source records are a match iff
they share a normalized DOI. DOI is therefore both the blocking key and
the (exact) matcher.

Coverage vs the labelled gold: DOI matching finds ~156k cross-source
pairs (dblp<->crossref ~50k, dblp<->open_alex ~55k, crossref<->open_alex
~50k) against only ~6.6k labelled gold positives, and crucially surfaces
the crossref<->open_alex matches that the EM gold never enumerates (it
only ships dblp-anchored pairs).

The matched pairs are unioned with the EM-gold positives (kept
unconditionally, mirroring ``build_pool.py`` bucket A): a gold positive
that is NOT a DOI match (a handful of same-paper / different-DOI cases)
is still retained.

Output (same schema as ``build_pool.py`` so downstream consumers --
the K2 niche dispatcher, K4 coverage pool pairs, the EM pool-agreement
diagnostic, and the fusion silver-standard partner graph -- need no
special case):

* ``usecases_synthetic/pools/papers/pooled_positives.csv`` with columns
  ``id1, id2, source_1, source_2, score, in_gold, in_human, in_ditto,
  decision_path``. ``decision_path`` is ``doi_exact`` / ``doi+gold`` /
  ``gold_only``; ``score`` is 1.0 (deterministic); ``in_ditto`` /
  ``in_human`` are False (no PLM / human-baseline pipeline for papers).
* ``usecases_synthetic/pools/papers/pool_stats.json`` with per-pair
  match counts, gold overlap, and the cluster-size distribution.

The egregious-cluster cap mirrors ``build_pool.py``:
``max(ceil(P99 of cluster sizes), 3 * n_sources)``. DOI is ~1:1 across
sources so clusters are tiny (<= one record per source); the cap only
guards against the rare within-source duplicate DOI.

Run::

    source pydi-dev/bin/activate
    python usecases_synthetic/scripts/build_pool_papers.py
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from collections import defaultdict
from itertools import combinations, product
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from usecases_synthetic.lib.domain_config import POOLS_DIR  # noqa: E402
from usecases_synthetic.lib.loaders import load_domain_sources  # noqa: E402
from usecases_synthetic.lib.variant_loader import load_variant  # noqa: E402

logger = logging.getLogger(__name__)

DOMAIN = "papers"
# Canonical source ordering -> id1 belongs to the earlier source.
SOURCE_ORDER = ["dblp", "crossref", "open_alex"]
POOL_COLUMNS = [
    "id1",
    "id2",
    "source_1",
    "source_2",
    "score",
    "in_gold",
    "in_human",
    "in_ditto",
    "decision_path",
]

_DOI_PREFIX_RE = re.compile(r"^(https?://)?(dx\.)?doi\.org/", re.IGNORECASE)


def _normalize_doi(value: object) -> str | None:
    """Return a normalized DOI string (lowercased, prefix-stripped) or None.

    DOIs are case-insensitive; the target schema stores them without the
    ``https://doi.org/`` prefix, but this strips one defensively so a
    stray prefix never blocks a match.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "none" or text.lower() == "nan":
        return None
    text = _DOI_PREFIX_RE.sub("", text)
    return text.lower() or None


def _order_pair(
    src_a: str, id_a: str, src_b: str, id_b: str
) -> tuple[str, str, str, str]:
    """Order a pair so (source_1, id1) is the earlier source in SOURCE_ORDER."""
    if SOURCE_ORDER.index(src_a) <= SOURCE_ORDER.index(src_b):
        return src_a, id_a, src_b, id_b
    return src_b, id_b, src_a, id_a


def _doi_index(sources: dict[str, pd.DataFrame]) -> dict[str, dict[str, list[str]]]:
    """Return ``{source: {normalized_doi: [record_ids]}}``."""
    out: dict[str, dict[str, list[str]]] = {}
    for name, df in sources.items():
        by_doi: dict[str, list[str]] = defaultdict(list)
        for rid, doi in zip(df["id"].astype(str), df["doi"], strict=True):
            ndoi = _normalize_doi(doi)
            if ndoi is not None:
                by_doi[ndoi].append(rid)
        out[name] = by_doi
        logger.info(
            "%s: %d records, %d with a usable DOI, %d distinct DOIs",
            name,
            len(df),
            sum(len(v) for v in by_doi.values()),
            len(by_doi),
        )
    return out


def _gold_positive_keys(bundle: object) -> set[tuple[str, str]]:
    """Return the set of (id1, id2) gold positives across all splits/pairs."""
    keys: set[tuple[str, str]] = set()
    for (src1, src2), splits in bundle.em_splits.items():  # type: ignore[attr-defined]
        for df in splits.values():
            pos = df[df["label"].astype(bool)]
            for id1, id2 in zip(
                pos["id1"].astype(str), pos["id2"].astype(str), strict=True
            ):
                s1, oid1, s2, oid2 = _order_pair(src1, id1, src2, id2)
                keys.add((oid1, oid2))
    return keys


def build_papers_pool(apply_cap: bool = True) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build the papers pool from DOI matches unioned with gold positives.

    Parameters
    ----------
    apply_cap : bool, default True
        Drop pairs in clusters larger than
        ``max(ceil(P99 of cluster sizes), 3 * n_sources)``.

    Returns
    -------
    tuple
        ``(pool_df, stats)``.
    """
    sources = load_domain_sources(DOMAIN)
    n_sources = len(sources)
    doi_idx = _doi_index(sources)
    gold_keys = _gold_positive_keys(load_variant(DOMAIN, "baseline"))

    # pair_key -> record dict. Union of DOI matches (all 3 source pairs)
    # and gold positives.
    pairs: dict[tuple[str, str], dict[str, object]] = {}
    per_pair_doi: dict[str, int] = {}

    for src_a, src_b in combinations(SOURCE_ORDER, 2):
        idx_a, idx_b = doi_idx[src_a], doi_idx[src_b]
        shared = set(idx_a) & set(idx_b)
        count = 0
        for doi in shared:
            for id_a, id_b in product(idx_a[doi], idx_b[doi]):
                s1, oid1, s2, oid2 = _order_pair(src_a, id_a, src_b, id_b)
                key = (oid1, oid2)
                in_gold = key in gold_keys
                pairs[key] = {
                    "id1": oid1,
                    "id2": oid2,
                    "source_1": s1,
                    "source_2": s2,
                    "score": 1.0,
                    "in_gold": in_gold,
                    "in_human": False,
                    "in_ditto": False,
                    "decision_path": "doi+gold" if in_gold else "doi_exact",
                }
                count += 1
        per_pair_doi[f"{src_a}_{src_b}"] = count
        logger.info("Pair %s<->%s: %d DOI matches", src_a, src_b, count)

    # Add gold positives that are NOT DOI matches (kept unconditionally).
    id_to_source = {
        rid: name for name, df in sources.items() for rid in df["id"].astype(str)
    }
    gold_only = 0
    for oid1, oid2 in gold_keys:
        if (oid1, oid2) in pairs:
            continue
        s1 = id_to_source.get(oid1, "")
        s2 = id_to_source.get(oid2, "")
        pairs[(oid1, oid2)] = {
            "id1": oid1,
            "id2": oid2,
            "source_1": s1,
            "source_2": s2,
            "score": 1.0,
            "in_gold": True,
            "in_human": False,
            "in_ditto": False,
            "decision_path": "gold_only",
        }
        gold_only += 1
    logger.info("Gold-only (non-DOI) positives retained: %d", gold_only)

    pool = pd.DataFrame(list(pairs.values()), columns=POOL_COLUMNS)

    clusters, sizes = _cluster_sizes(pool)
    p99 = int(math.ceil(pd.Series(sizes).quantile(0.99))) if sizes else 0
    cap = max(p99, 3 * n_sources)

    dropped_pairs = 0
    dropped_clusters: list[dict[str, object]] = []
    if apply_cap and sizes:
        member_to_size = {m: len(c) for c in clusters.values() for m in c}
        keep = pool.apply(
            lambda r: member_to_size.get(str(r["id1"]), 0) <= cap
            and member_to_size.get(str(r["id2"]), 0) <= cap,
            axis=1,
        )
        dropped_pairs = int((~keep).sum())
        for rep, members in clusters.items():
            if len(members) > cap:
                dropped_clusters.append({"representative": rep, "size": len(members)})
        pool = pool[keep].reset_index(drop=True)
        if dropped_pairs:
            clusters, sizes = _cluster_sizes(pool)

    stats: dict[str, object] = {
        "domain": DOMAIN,
        "source": "doi_exact_match (deterministic identity key) U em_gold_positives",
        "n_sources": n_sources,
        "pool_size": int(len(pool)),
        "per_pair_doi_matches": per_pair_doi,
        "n_gold_positive_keys": len(gold_keys),
        "n_gold_only_retained": gold_only,
        "n_clusters": len(clusters),
        "cluster_size_distribution": {
            "max": int(sizes[0]) if sizes else 0,
            "p99": p99,
            "cap": cap,
            "n_size_2": int(sum(1 for s in sizes if s == 2)),
            "n_size_3": int(sum(1 for s in sizes if s == 3)),
            "n_size_gt_3": int(sum(1 for s in sizes if s > 3)),
        },
        "egregious_filter": {
            "applied": apply_cap,
            "dropped_pairs": dropped_pairs,
            "dropped_clusters": dropped_clusters,
        },
    }
    return pool, stats


def _cluster_sizes(pool: pd.DataFrame) -> tuple[dict[str, set[str]], list[int]]:
    """Build transitive-closure clusters from the pool's partner graph.

    Mirrors ``fusion_perfect_clusters._partner_graph`` + closure so the
    reported sizes match what the silver builder will see.
    """
    partners: dict[str, set[str]] = defaultdict(set)
    for a, b in zip(pool["id1"].astype(str), pool["id2"].astype(str), strict=True):
        partners[a].add(b)
        partners[b].add(a)

    seen: set[str] = set()
    clusters: dict[str, set[str]] = {}
    for seed in sorted(partners):
        if seed in seen:
            continue
        stack = [seed]
        comp: set[str] = set()
        while stack:
            node = stack.pop()
            if node in comp:
                continue
            comp.add(node)
            stack.extend(partners[node] - comp)
        seen |= comp
        clusters[min(comp)] = comp
    sizes = sorted((len(c) for c in clusters.values()), reverse=True)
    return clusters, sizes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-cap",
        action="store_true",
        help="Skip the egregious-cluster cap (keep every matched pair).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)-5s] %(name)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    pool, stats = build_papers_pool(apply_cap=not args.no_cap)

    out_dir = POOLS_DIR / DOMAIN
    out_dir.mkdir(parents=True, exist_ok=True)
    pool_path = out_dir / "pooled_positives.csv"
    stats_path = out_dir / "pool_stats.json"
    pool.to_csv(pool_path, index=False)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    logger.info("Wrote %d pool pairs to %s", len(pool), pool_path)
    logger.info("Cluster stats: %s", stats["cluster_size_distribution"])
    logger.info(
        "Gold overlap: %d gold keys, %d gold-only retained",
        stats["n_gold_positive_keys"],
        stats["n_gold_only_retained"],
    )
    logger.info("Wrote stats to %s", stats_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
