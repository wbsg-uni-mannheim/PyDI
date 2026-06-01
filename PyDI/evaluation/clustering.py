"""
Alignment-based clustering metrics for end-to-end pipeline evaluation.

The pipeline produces a clustering of source records (one cluster per
fused row); the silver standard produces another clustering. Two
pipelines can produce identical aggregate distributions yet assign
records to completely wrong clusters — this module exploits the
record-level alignment to catch that.

Metrics implemented:

* :func:`bcubed_scores` — BCubed precision/recall/F1 (primary).
* :func:`adjusted_rand_index` and :func:`normalized_mutual_information`
  thin wrappers around :mod:`sklearn.metrics`.
* :func:`pairwise_scores` — pairwise precision/recall/F1 (secondary).
* :func:`cluster_alignment` — greedy max-overlap alignment from each
  silver cluster to its best pipeline counterpart, plus per-pair
  Jaccard. Returns the triage table that ``cluster_alignment.csv`` is
  built from, alongside mean Jaccard, matched-cluster rate, cluster
  purity and inverse purity.

Membership inputs are long-form ``(record_id, source, cluster_id)``
DataFrames as produced by :mod:`PyDI.evaluation.silver_standard` and by
:func:`PyDI.evaluation.clustering.membership_from_correspondences`.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Membership building from a correspondence DataFrame
# ---------------------------------------------------------------------------


def membership_from_correspondences(
    datasets: Sequence[pd.DataFrame],
    correspondences: pd.DataFrame,
    *,
    id_column: Optional[str] = None,
) -> pd.DataFrame:
    """Reconstruct cluster membership from a correspondences DataFrame.

    Delegates to :func:`PyDI.fusion.engine.build_record_groups_from_correspondences`
    so the panel's grouping matches what the fusion engine produced
    bit-for-bit.

    Parameters
    ----------
    datasets : sequence of DataFrame
        Source datasets. Each must have ``dataset_name`` in ``df.attrs``.
    correspondences : DataFrame
        Post-clusterer correspondences (``id1``, ``id2``, ``score``).
    id_column : str, optional
        Column carrying the record id. Passed through to the engine
        helper.

    Returns
    -------
    DataFrame
        Long-form ``(record_id, source, cluster_id)`` matching the
        silver loaders' membership shape.
    """
    from PyDI.fusion.engine import build_record_groups_from_correspondences

    groups = build_record_groups_from_correspondences(
        list(datasets), correspondences, id_column=id_column
    )

    rows: List[Dict[str, str]] = []
    for group in groups:
        cluster_id = group.group_id
        for record_id, source in group.source_datasets.items():
            rows.append(
                {
                    "record_id": str(record_id),
                    "source": str(source),
                    "cluster_id": str(cluster_id),
                }
            )
    return pd.DataFrame(rows, columns=["record_id", "source", "cluster_id"])


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _membership_to_dict(membership: pd.DataFrame) -> Dict[str, str]:
    """Build a ``record_id -> cluster_id`` mapping. Last write wins on duplicates."""
    return {
        str(row["record_id"]): str(row["cluster_id"])
        for _, row in membership.iterrows()
    }


def _shared_records(pipe: Dict[str, str], silver: Dict[str, str]) -> List[str]:
    return sorted(set(pipe.keys()) & set(silver.keys()))


def _labels_aligned(
    pipe: Dict[str, str], silver: Dict[str, str]
) -> Tuple[List[str], List[str], List[str]]:
    records = _shared_records(pipe, silver)
    pipe_labels = [pipe[r] for r in records]
    silver_labels = [silver[r] for r in records]
    return records, pipe_labels, silver_labels


# ---------------------------------------------------------------------------
# BCubed P/R/F1
# ---------------------------------------------------------------------------


def bcubed_scores(
    pipe_membership: pd.DataFrame, silver_membership: pd.DataFrame
) -> Dict[str, float]:
    """BCubed precision / recall / F1 — primary clustering metric.

    Plain-language question: "For each source record, did the pipeline
    lump me with the right other records?". Each record contributes one
    vote regardless of its cluster's size, which makes BCubed robust to
    cluster-size skew.

    Computed only on the intersection of records present in both
    memberships. Records in one but not the other are dropped (with a
    debug-log notice).

    Returns
    -------
    dict
        Keys: ``precision``, ``recall``, ``f1``.
    """
    pipe = _membership_to_dict(pipe_membership)
    silver = _membership_to_dict(silver_membership)
    records, pipe_labels, silver_labels = _labels_aligned(pipe, silver)
    if not records:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    pipe_clusters: Dict[str, List[str]] = defaultdict(list)
    silver_clusters: Dict[str, List[str]] = defaultdict(list)
    for record, p_label, s_label in zip(records, pipe_labels, silver_labels):
        pipe_clusters[p_label].append(record)
        silver_clusters[s_label].append(record)

    record_to_pipe = dict(zip(records, pipe_labels))
    record_to_silver = dict(zip(records, silver_labels))

    precisions: List[float] = []
    recalls: List[float] = []
    for record in records:
        pipe_cluster = set(pipe_clusters[record_to_pipe[record]])
        silver_cluster = set(silver_clusters[record_to_silver[record]])
        overlap = pipe_cluster & silver_cluster
        if pipe_cluster:
            precisions.append(len(overlap) / len(pipe_cluster))
        if silver_cluster:
            recalls.append(len(overlap) / len(silver_cluster))

    precision = float(np.mean(precisions)) if precisions else 0.0
    recall = float(np.mean(recalls)) if recalls else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------------
# ARI / NMI
# ---------------------------------------------------------------------------


def adjusted_rand_index(
    pipe_membership: pd.DataFrame, silver_membership: pd.DataFrame
) -> float:
    """Adjusted Rand Index between pipeline and silver clusterings.

    Plain-language question: "Across all pairs of records, what
    fraction does the pipeline agree with silver on (whether same
    cluster or different cluster), corrected for what a random
    clustering would score?". Random ≈ 0; perfect = 1; worse than
    random < 0.
    """
    from sklearn.metrics import adjusted_rand_score

    pipe = _membership_to_dict(pipe_membership)
    silver = _membership_to_dict(silver_membership)
    _, pipe_labels, silver_labels = _labels_aligned(pipe, silver)
    if not pipe_labels:
        return 0.0
    return float(adjusted_rand_score(silver_labels, pipe_labels))


def normalized_mutual_information(
    pipe_membership: pd.DataFrame, silver_membership: pd.DataFrame
) -> float:
    """Normalized Mutual Information between pipeline and silver clusterings.

    Plain-language question: "If I tell you which silver cluster a
    record is in, how much does that reduce uncertainty about which
    pipeline cluster it is in (and vice versa)?". Bounded ``[0, 1]``;
    0 = independent, 1 = agree up to relabeling.
    """
    from sklearn.metrics import normalized_mutual_info_score

    pipe = _membership_to_dict(pipe_membership)
    silver = _membership_to_dict(silver_membership)
    _, pipe_labels, silver_labels = _labels_aligned(pipe, silver)
    if not pipe_labels:
        return 0.0
    return float(normalized_mutual_info_score(silver_labels, pipe_labels))


# ---------------------------------------------------------------------------
# Pairwise P/R/F1
# ---------------------------------------------------------------------------


def pairwise_scores(
    pipe_membership: pd.DataFrame, silver_membership: pd.DataFrame
) -> Dict[str, float]:
    """Pairwise precision / recall / F1 — secondary clustering metric.

    Plain-language question: "Of all pairs of records, treat silver
    as labels and the pipeline as predictions of same-vs-different
    cluster. What's the F1?". Big clusters dominate pair counts; that
    skew is why BCubed is the load-bearing F1 for the panel.
    """
    pipe = _membership_to_dict(pipe_membership)
    silver = _membership_to_dict(silver_membership)
    records, _, _ = _labels_aligned(pipe, silver)
    n = len(records)
    if n < 2:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    pipe_clusters: Dict[str, List[str]] = defaultdict(list)
    silver_clusters: Dict[str, List[str]] = defaultdict(list)
    for record in records:
        pipe_clusters[pipe[record]].append(record)
        silver_clusters[silver[record]].append(record)

    def _within_cluster_pairs(clusters: Dict[str, List[str]]) -> set[Tuple[str, str]]:
        pairs: set[Tuple[str, str]] = set()
        for members in clusters.values():
            sorted_members = sorted(members)
            for i in range(len(sorted_members)):
                for j in range(i + 1, len(sorted_members)):
                    pairs.add((sorted_members[i], sorted_members[j]))
        return pairs

    pipe_pairs = _within_cluster_pairs(pipe_clusters)
    silver_pairs = _within_cluster_pairs(silver_clusters)

    true_positive = len(pipe_pairs & silver_pairs)
    precision = true_positive / len(pipe_pairs) if pipe_pairs else 0.0
    recall = true_positive / len(silver_pairs) if silver_pairs else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------------
# Greedy cluster alignment
# ---------------------------------------------------------------------------


def cluster_alignment(
    pipe_membership: pd.DataFrame,
    silver_membership: pd.DataFrame,
    *,
    matched_threshold: float = 0.5,
) -> Dict[str, object]:
    """Greedy max-overlap alignment from each silver cluster to the best pipeline cluster.

    Plain-language question per row: "For this silver cluster, which
    pipeline cluster did the most of its records end up in, and how
    well do they overlap?". The row-level table is the most useful
    artifact when the user needs to point at *which* clusters went
    wrong; the scalar summaries are derived from it.

    Returns
    -------
    dict
        Keys:

        * ``table`` (DataFrame): one row per silver cluster with
          ``silver_cluster_id``, ``best_pipe_cluster_id``,
          ``overlap_count``, ``silver_size``, ``pipe_size``, ``jaccard``.
        * ``mean_jaccard`` (float): mean Jaccard across silver clusters.
        * ``matched_cluster_rate_at_threshold`` (float): fraction of
          silver clusters with Jaccard ≥ ``matched_threshold``.
        * ``cluster_purity_pipe`` (float): macro purity of pipeline
          clusters (largest silver share / pipeline cluster size,
          averaged).
        * ``inverse_purity_silver`` (float): symmetric variant —
          largest pipeline share / silver cluster size, averaged.
    """
    pipe = _membership_to_dict(pipe_membership)
    silver = _membership_to_dict(silver_membership)
    records = _shared_records(pipe, silver)

    pipe_clusters: Dict[str, set[str]] = defaultdict(set)
    silver_clusters: Dict[str, set[str]] = defaultdict(set)
    for record in records:
        pipe_clusters[pipe[record]].add(record)
        silver_clusters[silver[record]].add(record)

    table_rows: List[Dict[str, object]] = []
    for silver_id, silver_members in silver_clusters.items():
        best_pipe_id: Optional[str] = None
        best_overlap = 0
        best_pipe_size = 0
        best_jaccard = 0.0
        for pipe_id, pipe_members in pipe_clusters.items():
            overlap = len(silver_members & pipe_members)
            if overlap == 0:
                continue
            union = len(silver_members | pipe_members)
            jaccard = overlap / union if union > 0 else 0.0
            if overlap > best_overlap or (
                overlap == best_overlap and jaccard > best_jaccard
            ):
                best_pipe_id = pipe_id
                best_overlap = overlap
                best_pipe_size = len(pipe_members)
                best_jaccard = jaccard

        table_rows.append(
            {
                "silver_cluster_id": silver_id,
                "best_pipe_cluster_id": best_pipe_id,
                "overlap_count": best_overlap,
                "silver_size": len(silver_members),
                "pipe_size": best_pipe_size,
                "jaccard": best_jaccard,
            }
        )

    table = pd.DataFrame(table_rows)
    if not table.empty:
        mean_jaccard = float(table["jaccard"].mean())
        matched_rate = float((table["jaccard"] >= matched_threshold).mean())
    else:
        mean_jaccard = 0.0
        matched_rate = 0.0

    cluster_purity_pipe = _purity(pipe_clusters, silver)
    inverse_purity_silver = _purity(silver_clusters, pipe)

    if not table.empty:
        size_delta = table["pipe_size"].astype(float) - table["silver_size"].astype(
            float
        )
        size_match_rate = float((size_delta == 0).mean())
        mean_size_delta = float(size_delta.mean())
        max_pipe_size = int(table["pipe_size"].max())
        max_silver_size = int(table["silver_size"].max())
    else:
        size_match_rate = 0.0
        mean_size_delta = 0.0
        max_pipe_size = 0
        max_silver_size = 0

    return {
        "table": table,
        "mean_jaccard": mean_jaccard,
        "matched_cluster_rate_at_threshold": matched_rate,
        "matched_threshold": matched_threshold,
        "cluster_purity_pipe": cluster_purity_pipe,
        "inverse_purity_silver": inverse_purity_silver,
        "size_match_rate": size_match_rate,
        "mean_size_delta": mean_size_delta,
        "max_size_overshoot": max_pipe_size - max_silver_size,
    }


def _purity(clusters: Dict[str, set[str]], opposite_labels: Dict[str, str]) -> float:
    if not clusters:
        return 0.0
    per_cluster: List[float] = []
    for members in clusters.values():
        if not members:
            continue
        counts: Dict[str, int] = defaultdict(int)
        for record in members:
            counts[opposite_labels.get(record, "<unassigned>")] += 1
        per_cluster.append(max(counts.values()) / len(members))
    return float(np.mean(per_cluster)) if per_cluster else 0.0


# ---------------------------------------------------------------------------
# Cluster-size extraction
# ---------------------------------------------------------------------------


def cluster_sizes(membership: pd.DataFrame) -> List[int]:
    """Return one cluster size per cluster present in *membership*."""
    if membership.empty:
        return []
    return [
        int(s)
        for s in membership.groupby("cluster_id")["record_id"]
        .nunique()
        .sort_index()
        .tolist()
    ]
