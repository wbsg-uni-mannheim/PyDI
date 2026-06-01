"""
Source-composition metrics for end-to-end pipeline evaluation.

In a multi-source data-integration pipeline, each cluster has a
*source signature* — the multiset of source datasets its records
come from. Silver clusters typically have ≤ 1 record per source
(each source contributes distinct entities globally). Pipeline
mistakes produce characteristic signature deviations that the
record-level metrics in :mod:`PyDI.evaluation.clustering` aggregate
away. This module exposes them directly.

Metrics
-------

* :func:`same_source_collision_rate` — fraction of clusters that
  contain ≥ 2 records from the same source. **Strong indicator of
  EM false positives**: silver basically never produces these (a
  source's record IDs are globally unique entity-by-entity).
* :func:`source_mix_distribution` — histogram over each cluster's
  ``frozenset(sources)``. Compare silver vs pipe with JS divergence
  to catch global "wrong source composition" patterns.
* :func:`per_source_coverage_rate` — per source, fraction of
  clusters with ≥ 1 record from that source. Catches "pipeline
  dropped most discogs records into singletons" patterns.
* :func:`source_composition_summary` — runs all three and packages
  the silver/pipe comparison in a single dict for panel emission.

Inputs are the long-form ``(record_id, source, cluster_id)``
DataFrames produced by the silver loaders and by
:func:`PyDI.evaluation.clustering.membership_from_correspondences`.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, FrozenSet, List, Mapping

import pandas as pd

from .distributional import jensen_shannon_divergence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Same-source collision
# ---------------------------------------------------------------------------


def same_source_collision_rate(
    membership: pd.DataFrame,
) -> Dict[str, Any]:
    """Fraction of clusters with ≥ 2 records from the same source.

    Plain-language question: "How often does the pipeline lump two
    records from the same source dataset into one cluster?". Silver
    basically never does this — each source's records are globally
    unique entities — so a non-zero rate on the pipeline side is a
    strong indicator of EM over-merge.

    Returns
    -------
    dict
        Keys:

        * ``overall`` — fraction of clusters with at least one
          same-source collision.
        * ``by_source`` — per source, fraction of clusters where
          that specific source contributed ≥ 2 records.
        * ``n_clusters`` — total cluster count in the input.
    """
    if membership.empty:
        return {"overall": 0.0, "by_source": {}, "n_clusters": 0}

    grouped = membership.groupby("cluster_id")
    n_clusters = len(grouped)
    overall_collisions = 0
    per_source_collisions: Dict[str, int] = {}

    for _, group in grouped:
        source_counts = group["source"].value_counts()
        offenders = source_counts[source_counts >= 2]
        if not offenders.empty:
            overall_collisions += 1
            for source in offenders.index:
                per_source_collisions[source] = per_source_collisions.get(source, 0) + 1

    by_source = {
        source: count / n_clusters for source, count in per_source_collisions.items()
    }
    # Include sources with 0 collisions so the dict is comprehensive.
    for source in membership["source"].unique():
        by_source.setdefault(str(source), 0.0)

    return {
        "overall": overall_collisions / n_clusters,
        "by_source": by_source,
        "n_clusters": n_clusters,
    }


# ---------------------------------------------------------------------------
# Source-mix distribution
# ---------------------------------------------------------------------------


def source_mix_distribution(
    membership: pd.DataFrame,
) -> Dict[FrozenSet[str], float]:
    """Per-cluster source signature histogram, normalised to a distribution.

    Plain-language question: "For each cluster, *which* set of sources
    does it touch (ignoring multiplicities)? What's the overall
    distribution of these source-sets across clusters?".

    A cluster with members from ``{musicbrainz, discogs}`` and one
    with members from ``{musicbrainz, discogs, lastfm}`` are treated
    as different keys. Multiplicities are intentionally ignored here
    — the multiplicity signal is captured by
    :func:`same_source_collision_rate`.

    Returns
    -------
    dict[frozenset[str], float]
        Probability mass per source-set. Sums to 1.0 (unless empty).
    """
    if membership.empty:
        return {}
    counts: Counter = Counter()
    for _, group in membership.groupby("cluster_id"):
        sig = frozenset(str(s) for s in group["source"].unique())
        counts[sig] += 1
    total = sum(counts.values())
    if total == 0:
        return {}
    return {sig: count / total for sig, count in counts.items()}


# ---------------------------------------------------------------------------
# Per-source coverage
# ---------------------------------------------------------------------------


def per_source_coverage_rate(
    membership: pd.DataFrame,
) -> Dict[str, float]:
    """Per source, fraction of clusters that contain ≥ 1 record from it.

    Plain-language question: "How frequently does each source
    contribute to a cluster?". A source whose coverage drops
    significantly between silver and pipe means that source's records
    are landing in singletons (no cross-source matches) or being
    absorbed into wrong clusters.

    Returns
    -------
    dict[str, float]
        ``source_name -> coverage_rate`` over all clusters in
        ``membership``.
    """
    if membership.empty:
        return {}
    by_cluster_sources = membership.groupby("cluster_id")["source"].apply(set)
    n_clusters = len(by_cluster_sources)
    if n_clusters == 0:
        return {}
    sources_seen = sorted(set().union(*by_cluster_sources))
    return {
        source: sum(1 for s in by_cluster_sources if source in s) / n_clusters
        for source in sources_seen
    }


# ---------------------------------------------------------------------------
# Combined silver-vs-pipe summary
# ---------------------------------------------------------------------------


def _delta(
    silver_dict: Mapping[str, float], pipe_dict: Mapping[str, float]
) -> Dict[str, Dict[str, float]]:
    keys = sorted(set(silver_dict) | set(pipe_dict))
    return {
        key: {
            "reference": float(silver_dict.get(key, 0.0)),
            "pipe": float(pipe_dict.get(key, 0.0)),
            "delta": float(pipe_dict.get(key, 0.0)) - float(silver_dict.get(key, 0.0)),
        }
        for key in keys
    }


def _mix_to_string_keys(
    mix: Mapping[FrozenSet[str], float],
) -> Dict[str, float]:
    """Stringify frozenset keys so the histogram is JSON-serialisable."""
    return {
        "|".join(sorted(sig)) if sig else "<empty>": prob for sig, prob in mix.items()
    }


def source_composition_summary(
    pipe_membership: pd.DataFrame,
    silver_membership: pd.DataFrame,
) -> Dict[str, Any]:
    """Compare silver and pipeline source-composition signals.

    Returns
    -------
    dict
        Keys:

        * ``same_source_collision_rate`` — ``{reference, pipe, delta,
          by_source: {<source>: {reference, pipe, delta}}}``.
        * ``per_source_coverage_rate`` — ``{<source>: {reference, pipe,
          delta}}``.
        * ``source_mix_distribution_js`` — JS divergence between the
          two source-mix distributions (bounded ``[0, 1]``).
        * ``source_mix_distribution_reference`` /
          ``source_mix_distribution_pipe`` — string-keyed histograms
          for inspection.
    """
    silver_coll = same_source_collision_rate(silver_membership)
    pipe_coll = same_source_collision_rate(pipe_membership)

    by_source_collision = _delta(silver_coll["by_source"], pipe_coll["by_source"])

    coverage = _delta(
        per_source_coverage_rate(silver_membership),
        per_source_coverage_rate(pipe_membership),
    )

    silver_mix = source_mix_distribution(silver_membership)
    pipe_mix = source_mix_distribution(pipe_membership)
    silver_mix_str = _mix_to_string_keys(silver_mix)
    pipe_mix_str = _mix_to_string_keys(pipe_mix)
    js = jensen_shannon_divergence(pipe_mix_str, silver_mix_str)

    return {
        "same_source_collision_rate": {
            "reference": float(silver_coll["overall"]),
            "pipe": float(pipe_coll["overall"]),
            "delta": float(pipe_coll["overall"] - silver_coll["overall"]),
            "by_source": by_source_collision,
        },
        "per_source_coverage_rate": coverage,
        "source_mix_distribution_js": js,
        "source_mix_distribution_reference": silver_mix_str,
        "source_mix_distribution_pipe": pipe_mix_str,
    }
