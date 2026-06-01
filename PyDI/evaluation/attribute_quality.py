"""
Fused-attribute quality metrics for end-to-end pipeline evaluation.

This module covers the ``correctness.fact`` subdimension of the panel:

* §3.6 fused-attribute quality on aligned clusters — exact accuracy,
  normalized Levenshtein similarity, MAE / MedAE, datetime delta.
* §3.7.1 conflict-only fused-attribute accuracy — restricted to cells
  where input sources disagreed.
* §3.7.2 source-attribution distribution per attribute — JS divergence
  on the "which source won" histogram (silver vs pipeline). Requires
  per-cell provenance on both sides; gracefully skipped otherwise.
* §3.7.3 multi-truth / list-valued set agreement — set precision /
  recall / F1 / Jaccard per cluster × set-valued attribute.
* §3.7.4 per-attribute density / coverage delta.
* §3.7.5 conflict rate (diagnostic context).
* §3.7.6 per-cluster fully-correct rate.
* §3.7.7 synthesis rate per attribute (silver / pipe / delta).
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import pandas as pd

from .distributional import jensen_shannon_divergence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type helpers
# ---------------------------------------------------------------------------


def _is_missing(value: Any) -> bool:
    """Match the missing-value convention used by ``PyDI.fusion.evaluation``."""
    if isinstance(value, (list, tuple, set)):
        return len(value) == 0
    if isinstance(value, np.ndarray):
        return value.size == 0
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _as_set(value: Any) -> Optional[set[str]]:
    if _is_missing(value):
        return None
    if isinstance(value, (list, tuple, set)):
        return {str(v).strip() for v in value if not _is_missing(v)}
    return {str(value).strip()}


def _levenshtein_similarity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    if not a and not b:
        return 1.0
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    distance = _levenshtein_distance(a, b)
    return 1.0 - distance / max_len


def _levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


# ---------------------------------------------------------------------------
# §3.6 fused-attribute quality (on aligned clusters)
# ---------------------------------------------------------------------------


def _compare_scalar(
    pipe_value: Any,
    silver_value: Any,
    col_type: str,
    *,
    numerical_tolerance: float,
) -> Dict[str, Any]:
    """Compare a single (pipe_value, silver_value) cell. Returns per-cell stats."""
    pipe_missing = _is_missing(pipe_value)
    silver_missing = _is_missing(silver_value)
    if pipe_missing and silver_missing:
        return {"evaluable": False}
    if silver_missing:
        return {"evaluable": False}
    if pipe_missing:
        return {
            "evaluable": True,
            "correct": False,
            "similarity": 0.0,
            "abs_error": None,
            "within_tolerance": False,
            "day_delta": None,
        }

    if col_type == "numerical":
        try:
            pipe_num = float(pipe_value)
            silver_num = float(silver_value)
        except (TypeError, ValueError):
            return _compare_text(pipe_value, silver_value)
        abs_error = abs(pipe_num - silver_num)
        denominator = abs(silver_num) if silver_num != 0 else 1.0
        within = abs_error <= numerical_tolerance * denominator
        return {
            "evaluable": True,
            "correct": pipe_num == silver_num,
            "similarity": 1.0 if pipe_num == silver_num else 0.0,
            "abs_error": abs_error,
            "within_tolerance": within,
            "day_delta": None,
        }

    if col_type == "datetime":
        pipe_dt = pd.to_datetime(pipe_value, errors="coerce")
        silver_dt = pd.to_datetime(silver_value, errors="coerce")
        if pd.isna(pipe_dt) or pd.isna(silver_dt):
            return _compare_text(pipe_value, silver_value)
        delta_days = abs((pipe_dt - silver_dt).total_seconds()) / 86400.0
        return {
            "evaluable": True,
            "correct": pipe_dt == silver_dt,
            "similarity": 1.0 if pipe_dt == silver_dt else 0.0,
            "abs_error": None,
            "within_tolerance": None,
            "day_delta": delta_days,
        }

    return _compare_text(pipe_value, silver_value)


def _compare_text(pipe_value: Any, silver_value: Any) -> Dict[str, Any]:
    pipe_str = str(pipe_value)
    silver_str = str(silver_value)
    similarity = _levenshtein_similarity(pipe_str, silver_str)
    return {
        "evaluable": True,
        "correct": pipe_str == silver_str,
        "similarity": similarity,
        "abs_error": None,
        "within_tolerance": None,
        "day_delta": None,
    }


def _compare_list(pipe_value: Any, silver_value: Any) -> Dict[str, Any]:
    pipe_set = _as_set(pipe_value)
    silver_set = _as_set(silver_value)
    if pipe_set is None and silver_set is None:
        return {"evaluable": False}
    if silver_set is None:
        return {"evaluable": False}
    if pipe_set is None:
        return {
            "evaluable": True,
            "correct": False,
            "set_precision": 0.0,
            "set_recall": 0.0,
            "set_f1": 0.0,
            "set_jaccard": 0.0,
        }
    intersection = pipe_set & silver_set
    union = pipe_set | silver_set
    precision = len(intersection) / len(pipe_set) if pipe_set else 0.0
    recall = len(intersection) / len(silver_set) if silver_set else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    jaccard = len(intersection) / len(union) if union else 1.0
    return {
        "evaluable": True,
        "correct": pipe_set == silver_set,
        "set_precision": precision,
        "set_recall": recall,
        "set_f1": f1,
        "set_jaccard": jaccard,
    }


# ---------------------------------------------------------------------------
# Aligned-row iteration
# ---------------------------------------------------------------------------


def _align_clusters(
    pipe_df: pd.DataFrame,
    silver_df: pd.DataFrame,
    alignment_table: pd.DataFrame,
    *,
    pipe_id_column: str,
    silver_id_column: str,
) -> List[Tuple[Any, Any, pd.Series, pd.Series]]:
    """Iterate aligned (silver_cluster_id, pipe_cluster_id, silver_row, pipe_row)."""
    pipe_indexed = pipe_df.set_index(pipe_df[pipe_id_column].astype(str), drop=False)
    silver_indexed = silver_df.set_index(
        silver_df[silver_id_column].astype(str), drop=False
    )

    aligned: List[Tuple[Any, Any, pd.Series, pd.Series]] = []
    for _, row in alignment_table.iterrows():
        silver_id = row["silver_cluster_id"]
        pipe_id = row["best_pipe_cluster_id"]
        if pipe_id is None or (isinstance(pipe_id, float) and math.isnan(pipe_id)):
            continue
        silver_key = str(silver_id)
        pipe_key = str(pipe_id)
        if silver_key not in silver_indexed.index or pipe_key not in pipe_indexed.index:
            continue
        silver_row = silver_indexed.loc[silver_key]
        pipe_row = pipe_indexed.loc[pipe_key]
        if isinstance(silver_row, pd.DataFrame):
            silver_row = silver_row.iloc[0]
        if isinstance(pipe_row, pd.DataFrame):
            pipe_row = pipe_row.iloc[0]
        aligned.append((silver_id, pipe_id, silver_row, pipe_row))
    return aligned


# ---------------------------------------------------------------------------
# §3.6 main entry point
# ---------------------------------------------------------------------------


def fused_attribute_quality(
    pipe_df: pd.DataFrame,
    silver_df: pd.DataFrame,
    alignment_table: pd.DataFrame,
    column_types: Mapping[str, str],
    *,
    pipe_id_column: str = "cluster_id",
    silver_id_column: str = "cluster_id",
    numerical_tolerance: float = 0.04,
    numerical_tolerance_overrides: Optional[Mapping[str, float]] = None,
    semantic_value_similarity: Optional[Callable[[str, str], float]] = None,
    semantic_value_threshold: float = 0.85,
) -> Dict[str, Any]:
    """§3.6 fused-attribute quality on aligned clusters.

    Plain-language question per attribute: "Once silver and pipeline
    clusters have been aligned, do the pipeline's fused values for this
    attribute match silver's fused values?".

    Returns
    -------
    dict
        Keys:

        * ``per_attribute`` — ``{attribute: {accuracy, similarity_mean,
          mae, medae, pct_within_tolerance, day_delta_mean, count}}``.
        * ``macro_accuracy`` (float) — mean of per-attribute accuracy.
        * ``micro_accuracy`` (float) — total correct / total evaluable
          across all (cluster, attribute) cells.
        * ``per_cluster_correctness`` — DataFrame with one row per
          aligned cluster carrying per-attribute correctness flags.
          Used by :func:`per_cluster_fully_correct_rate`.

    Notes
    -----
    Skips:

    * Attributes tagged ``identifier`` — not evaluable.
    * Attributes tagged ``list`` — handled by :func:`list_attribute_set_metrics`.
    * Attributes absent from either side.
    """
    overrides = dict(numerical_tolerance_overrides or {})

    per_attribute: Dict[str, Dict[str, Any]] = {}
    per_attribute_correct: Dict[str, List[bool]] = {}
    per_attribute_similarities: Dict[str, List[float]] = {}
    per_attribute_abs_errors: Dict[str, List[float]] = {}
    per_attribute_within_tolerance: Dict[str, List[bool]] = {}
    per_attribute_day_deltas: Dict[str, List[float]] = {}
    per_attribute_semantic_correct: Dict[str, List[bool]] = {}
    per_attribute_semantic_similarity: Dict[str, List[float]] = {}

    aligned = _align_clusters(
        pipe_df,
        silver_df,
        alignment_table,
        pipe_id_column=pipe_id_column,
        silver_id_column=silver_id_column,
    )

    cluster_correctness_rows: List[Dict[str, Any]] = []

    evaluable_attributes = [
        attr
        for attr, col_type in column_types.items()
        if col_type not in {"identifier", "list"}
        and attr in pipe_df.columns
        and attr in silver_df.columns
    ]

    for silver_id, pipe_id, silver_row, pipe_row in aligned:
        per_attr_correct: Dict[str, Optional[bool]] = {}
        for attribute in evaluable_attributes:
            col_type = column_types[attribute]
            tolerance = overrides.get(attribute, numerical_tolerance)
            result = _compare_scalar(
                pipe_row.get(attribute),
                silver_row.get(attribute),
                col_type,
                numerical_tolerance=tolerance,
            )
            if not result.get("evaluable", False):
                per_attr_correct[attribute] = None
                continue

            correct = bool(result["correct"])
            per_attr_correct[attribute] = correct
            per_attribute_correct.setdefault(attribute, []).append(correct)
            similarity = result.get("similarity")
            if similarity is not None:
                per_attribute_similarities.setdefault(attribute, []).append(similarity)
            abs_error = result.get("abs_error")
            if abs_error is not None:
                per_attribute_abs_errors.setdefault(attribute, []).append(abs_error)
            within = result.get("within_tolerance")
            if within is not None:
                per_attribute_within_tolerance.setdefault(attribute, []).append(
                    bool(within)
                )
            day_delta = result.get("day_delta")
            if day_delta is not None:
                per_attribute_day_deltas.setdefault(attribute, []).append(day_delta)

            if semantic_value_similarity is not None and col_type in {
                "text",
                "categorical",
            }:
                if correct:
                    # exact match → semantically correct by definition
                    sim = 1.0
                    sem_correct = True
                else:
                    pipe_value = pipe_row.get(attribute)
                    silver_value = silver_row.get(attribute)
                    try:
                        sim = float(
                            semantic_value_similarity(
                                str(pipe_value), str(silver_value)
                            )
                        )
                    except Exception as exc:  # noqa: BLE001 — caller-supplied callable
                        logger.debug(
                            "semantic_value_similarity raised on %s: %s",
                            attribute,
                            exc,
                        )
                        sim = 0.0
                    sem_correct = sim >= semantic_value_threshold
                per_attribute_semantic_similarity.setdefault(attribute, []).append(sim)
                per_attribute_semantic_correct.setdefault(attribute, []).append(
                    sem_correct
                )

        row = {"silver_cluster_id": silver_id, "pipe_cluster_id": pipe_id}
        row.update(per_attr_correct)
        cluster_correctness_rows.append(row)

    total_correct = 0
    total_evaluable = 0
    for attribute in evaluable_attributes:
        flags = per_attribute_correct.get(attribute, [])
        if not flags:
            continue
        correct_count = sum(1 for f in flags if f)
        accuracy = correct_count / len(flags)
        total_correct += correct_count
        total_evaluable += len(flags)

        similarities = per_attribute_similarities.get(attribute, [])
        abs_errors = per_attribute_abs_errors.get(attribute, [])
        within = per_attribute_within_tolerance.get(attribute, [])
        day_deltas = per_attribute_day_deltas.get(attribute, [])

        per_attribute[attribute] = {
            "accuracy": accuracy,
            "count": len(flags),
        }
        if similarities:
            per_attribute[attribute]["similarity_mean"] = float(np.mean(similarities))
        if abs_errors:
            per_attribute[attribute]["mae"] = float(np.mean(abs_errors))
            per_attribute[attribute]["medae"] = float(np.median(abs_errors))
        if within:
            per_attribute[attribute]["pct_within_tolerance"] = float(np.mean(within))
        if day_deltas:
            per_attribute[attribute]["day_delta_mean"] = float(np.mean(day_deltas))
        sem_flags = per_attribute_semantic_correct.get(attribute, [])
        sem_sims = per_attribute_semantic_similarity.get(attribute, [])
        if sem_flags:
            per_attribute[attribute]["semantic_accuracy"] = float(np.mean(sem_flags))
        if sem_sims:
            per_attribute[attribute]["semantic_similarity_mean"] = float(
                np.mean(sem_sims)
            )

    macro_accuracy = (
        float(np.mean([m["accuracy"] for m in per_attribute.values()]))
        if per_attribute
        else 0.0
    )
    micro_accuracy = total_correct / total_evaluable if total_evaluable > 0 else 0.0

    correctness_df = pd.DataFrame(cluster_correctness_rows)
    return {
        "per_attribute": per_attribute,
        "macro_accuracy": macro_accuracy,
        "micro_accuracy": micro_accuracy,
        "per_cluster_correctness": correctness_df,
        "evaluable_attributes": evaluable_attributes,
    }


# ---------------------------------------------------------------------------
# §3.7.1 conflict-only accuracy
# §3.7.5 conflict rate
# ---------------------------------------------------------------------------


def _has_conflict(values: Sequence[Any]) -> bool:
    seen: set[str] = set()
    for v in values:
        if _is_missing(v):
            continue
        seen.add(_canonical_str(v))
        if len(seen) >= 2:
            return True
    return False


def _canonical_str(value: Any) -> str:
    if isinstance(value, (list, tuple, set)):
        return "|".join(sorted(str(v) for v in value))
    return str(value)


def _build_cluster_source_values(
    sources_by_record: Mapping[str, Mapping[str, Any]],
    membership: pd.DataFrame,
    attributes: Sequence[str],
) -> Dict[str, Dict[str, List[Any]]]:
    """Build ``{cluster_id: {attribute: [values from each source record]}}``.

    Used by both conflict-only accuracy (§3.7.1) and conflict rate
    (§3.7.5). ``sources_by_record`` maps record id → ``{attribute:
    value}`` row.
    """
    if membership.empty:
        return {}
    grouped: Dict[str, Dict[str, List[Any]]] = {}
    for _, row in membership.iterrows():
        cluster_id = str(row["cluster_id"])
        record_id = str(row["record_id"])
        record = sources_by_record.get(record_id)
        if record is None:
            continue
        bucket = grouped.setdefault(cluster_id, {a: [] for a in attributes})
        for attribute in attributes:
            bucket.setdefault(attribute, []).append(record.get(attribute))
    return grouped


def conflict_metrics(
    pipe_df: pd.DataFrame,
    silver_df: pd.DataFrame,
    alignment_table: pd.DataFrame,
    column_types: Mapping[str, str],
    *,
    pipe_id_column: str,
    silver_id_column: str,
    pipe_membership: pd.DataFrame,
    silver_membership: pd.DataFrame,
    pipe_source_records: Mapping[str, Mapping[str, Any]],
    silver_source_records: Mapping[str, Mapping[str, Any]],
    numerical_tolerance: float,
    numerical_tolerance_overrides: Optional[Mapping[str, float]],
) -> Dict[str, Any]:
    """§3.7.1 conflict-only accuracy + §3.7.5 conflict rate.

    Plain-language question for §3.7.1: "Restricted to cluster ×
    attribute cells where the source records disagreed, did the
    pipeline pick the same fused value as silver?". This is the real
    test of fusion strategy — every reasonable policy gets the
    agreement-cells right.

    Plain-language question for §3.7.5: "What fraction of clusters
    had at least one attribute where source records disagreed?". Pipe
    and silver reported separately as context.
    """
    overrides = dict(numerical_tolerance_overrides or {})

    evaluable_attributes = [
        attr
        for attr, col_type in column_types.items()
        if col_type not in {"identifier", "list"}
        and attr in pipe_df.columns
        and attr in silver_df.columns
    ]

    pipe_cluster_values = _build_cluster_source_values(
        pipe_source_records, pipe_membership, evaluable_attributes
    )
    silver_cluster_values = _build_cluster_source_values(
        silver_source_records, silver_membership, evaluable_attributes
    )

    aligned = _align_clusters(
        pipe_df,
        silver_df,
        alignment_table,
        pipe_id_column=pipe_id_column,
        silver_id_column=silver_id_column,
    )

    per_attribute_correct: Dict[str, List[bool]] = {}
    conflict_clusters_pipe: set[str] = set()
    conflict_clusters_silver: set[str] = set()

    for cluster_id, sources in pipe_cluster_values.items():
        if any(_has_conflict(vals) for vals in sources.values()):
            conflict_clusters_pipe.add(cluster_id)
    for cluster_id, sources in silver_cluster_values.items():
        if any(_has_conflict(vals) for vals in sources.values()):
            conflict_clusters_silver.add(cluster_id)

    for silver_id, pipe_id, silver_row, pipe_row in aligned:
        silver_key = str(silver_id)
        pipe_key = str(pipe_id)
        silver_sources = silver_cluster_values.get(silver_key, {})
        pipe_sources = pipe_cluster_values.get(pipe_key, {})
        for attribute in evaluable_attributes:
            silver_vals = silver_sources.get(attribute, [])
            pipe_vals = pipe_sources.get(attribute, [])
            if not _has_conflict(silver_vals) and not _has_conflict(pipe_vals):
                continue
            col_type = column_types[attribute]
            tolerance = overrides.get(attribute, numerical_tolerance)
            result = _compare_scalar(
                pipe_row.get(attribute),
                silver_row.get(attribute),
                col_type,
                numerical_tolerance=tolerance,
            )
            if not result.get("evaluable", False):
                continue
            per_attribute_correct.setdefault(attribute, []).append(
                bool(result["correct"])
            )

    per_attribute: Dict[str, Dict[str, Any]] = {}
    total_correct = 0
    total_count = 0
    for attribute, flags in per_attribute_correct.items():
        correct = sum(1 for f in flags if f)
        total_correct += correct
        total_count += len(flags)
        per_attribute[attribute] = {
            "accuracy": correct / len(flags),
            "count": len(flags),
        }
    micro = total_correct / total_count if total_count > 0 else 0.0
    macro = (
        float(np.mean([v["accuracy"] for v in per_attribute.values()]))
        if per_attribute
        else 0.0
    )

    n_pipe_clusters = pipe_cluster_values.keys()
    n_silver_clusters = silver_cluster_values.keys()
    return {
        "per_attribute": per_attribute,
        "micro_accuracy": micro,
        "macro_accuracy": macro,
        "conflict_rate_pipe": (
            len(conflict_clusters_pipe) / len(n_pipe_clusters)
            if n_pipe_clusters
            else 0.0
        ),
        "conflict_rate_silver": (
            len(conflict_clusters_silver) / len(n_silver_clusters)
            if n_silver_clusters
            else 0.0
        ),
    }


# ---------------------------------------------------------------------------
# §3.7.2 source-attribution distribution
# §3.7.7 synthesis rate
# ---------------------------------------------------------------------------


def source_attribution_metrics(
    silver_provenance: pd.DataFrame,
    pipe_provenance: pd.DataFrame,
    alignment_table: pd.DataFrame,
    *,
    source_inference: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """§3.7.2 source-attribution JS divergence + §3.7.7 synthesis rate.

    Plain-language question for §3.7.2: "Per attribute, which source
    won each fused cell, and does the pipeline's winning-source
    distribution match silver's?". Mass for a composite cell carrying
    ``source_ids = [A, B]`` is split equally per §6.5 Q9.

    Plain-language question for §3.7.7: "Per attribute, how often
    does each side produce a fused value drawn from multiple sources
    (composite provenance) rather than picking one?".

    Parameters
    ----------
    silver_provenance, pipe_provenance : DataFrame
        Long-form ``(cluster_id, attribute, source_ids)`` where
        ``source_ids`` is a list. May be ``None``-typed (the panel
        guards against that earlier).
    alignment_table : DataFrame
        From :func:`PyDI.evaluation.clustering.cluster_alignment`.
    source_inference : mapping, optional
        Maps source-id prefix → source name. Used to bucket
        individual record ids under their dataset name. When ``None``,
        the raw record id is used as the bucket label (less aggregated
        but still well-defined).

    Returns
    -------
    dict
        Keys: ``source_attribution_js_per_attribute`` (dict),
        ``synthesis_rate_per_attribute`` (``{attr: {silver, pipe, delta}}``).
    """

    def _bucket(record_id: str) -> str:
        if source_inference is None:
            return record_id
        for prefix, source in source_inference.items():
            if record_id.startswith(prefix):
                return source
        return record_id

    silver_lookup = _provenance_lookup(silver_provenance)
    pipe_lookup = _provenance_lookup(pipe_provenance)

    silver_mass: Dict[str, Dict[str, float]] = {}
    pipe_mass: Dict[str, Dict[str, float]] = {}
    silver_synth: Dict[str, List[int]] = {}
    pipe_synth: Dict[str, List[int]] = {}

    for _, row in alignment_table.iterrows():
        silver_id = str(row["silver_cluster_id"])
        pipe_id = row["best_pipe_cluster_id"]
        if pipe_id is None or (isinstance(pipe_id, float) and math.isnan(pipe_id)):
            continue
        pipe_id = str(pipe_id)

        silver_cells = silver_lookup.get(silver_id, {})
        pipe_cells = pipe_lookup.get(pipe_id, {})

        for attribute, silver_ids in silver_cells.items():
            mass = silver_mass.setdefault(attribute, {})
            _add_mass(mass, silver_ids, _bucket)
            silver_synth.setdefault(attribute, []).append(
                1 if len(silver_ids) > 1 else 0
            )

        for attribute, pipe_ids in pipe_cells.items():
            mass = pipe_mass.setdefault(attribute, {})
            _add_mass(mass, pipe_ids, _bucket)
            pipe_synth.setdefault(attribute, []).append(1 if len(pipe_ids) > 1 else 0)

    attributes = sorted(set(silver_mass.keys()) | set(pipe_mass.keys()))
    js_per_attribute: Dict[str, float] = {}
    for attribute in attributes:
        silver_counts = silver_mass.get(attribute, {})
        pipe_counts = pipe_mass.get(attribute, {})
        silver_total = sum(silver_counts.values())
        pipe_total = sum(pipe_counts.values())
        silver_dist = (
            {k: v / silver_total for k, v in silver_counts.items()}
            if silver_total
            else {}
        )
        pipe_dist = (
            {k: v / pipe_total for k, v in pipe_counts.items()} if pipe_total else {}
        )
        js_per_attribute[attribute] = jensen_shannon_divergence(pipe_dist, silver_dist)

    synthesis_per_attribute: Dict[str, Dict[str, float]] = {}
    synth_attrs = sorted(set(silver_synth.keys()) | set(pipe_synth.keys()))
    for attribute in synth_attrs:
        silver_flags = silver_synth.get(attribute, [])
        pipe_flags = pipe_synth.get(attribute, [])
        silver_rate = float(np.mean(silver_flags)) if silver_flags else 0.0
        pipe_rate = float(np.mean(pipe_flags)) if pipe_flags else 0.0
        synthesis_per_attribute[attribute] = {
            "silver": silver_rate,
            "pipe": pipe_rate,
            "delta": pipe_rate - silver_rate,
        }

    return {
        "source_attribution_js_per_attribute": js_per_attribute,
        "synthesis_rate_per_attribute": synthesis_per_attribute,
    }


def _provenance_lookup(
    provenance: pd.DataFrame,
) -> Dict[str, Dict[str, List[str]]]:
    out: Dict[str, Dict[str, List[str]]] = {}
    for _, row in provenance.iterrows():
        cluster_id = str(row["cluster_id"])
        attribute = str(row["attribute"])
        source_ids = row["source_ids"]
        if isinstance(source_ids, str):
            source_ids = [s.strip() for s in source_ids.split("+") if s.strip()]
        elif not isinstance(source_ids, (list, tuple)):
            continue
        out.setdefault(cluster_id, {})[attribute] = [str(s) for s in source_ids]
    return out


def _add_mass(
    counter: Dict[str, float],
    source_ids: Sequence[str],
    bucket_fn: Callable[[str], str],
) -> None:
    if not source_ids:
        return
    weight = 1.0 / len(source_ids)
    for source_id in source_ids:
        bucket = bucket_fn(source_id)
        counter[bucket] = counter.get(bucket, 0.0) + weight


# ---------------------------------------------------------------------------
# §3.7.3 list-valued set agreement
# ---------------------------------------------------------------------------


def list_attribute_set_metrics(
    pipe_df: pd.DataFrame,
    silver_df: pd.DataFrame,
    alignment_table: pd.DataFrame,
    column_types: Mapping[str, str],
    *,
    pipe_id_column: str = "cluster_id",
    silver_id_column: str = "cluster_id",
) -> Dict[str, Dict[str, float]]:
    """§3.7.3 set precision / recall / F1 / Jaccard per list-valued attribute.

    Plain-language question: "For attributes whose value is a set
    (genres, platforms, authors, languages), do the pipeline's sets
    match silver's per aligned cluster?".
    """
    list_attributes = [
        attr
        for attr, col_type in column_types.items()
        if col_type == "list" and attr in pipe_df.columns and attr in silver_df.columns
    ]
    if not list_attributes:
        return {}

    aligned = _align_clusters(
        pipe_df,
        silver_df,
        alignment_table,
        pipe_id_column=pipe_id_column,
        silver_id_column=silver_id_column,
    )

    per_attribute: Dict[str, Dict[str, float]] = {}
    for attribute in list_attributes:
        precisions: List[float] = []
        recalls: List[float] = []
        f1s: List[float] = []
        jaccards: List[float] = []
        for _, _, silver_row, pipe_row in aligned:
            result = _compare_list(pipe_row.get(attribute), silver_row.get(attribute))
            if not result.get("evaluable", False):
                continue
            precisions.append(result["set_precision"])
            recalls.append(result["set_recall"])
            f1s.append(result["set_f1"])
            jaccards.append(result["set_jaccard"])
        if not precisions:
            continue
        per_attribute[attribute] = {
            "set_precision": float(np.mean(precisions)),
            "set_recall": float(np.mean(recalls)),
            "set_f1": float(np.mean(f1s)),
            "set_jaccard": float(np.mean(jaccards)),
            "count": len(precisions),
        }
    return per_attribute


# ---------------------------------------------------------------------------
# §3.7.4 per-attribute density / coverage delta
# ---------------------------------------------------------------------------


def per_attribute_density_delta(
    pipe_df: pd.DataFrame,
    silver_df: pd.DataFrame,
    column_types: Mapping[str, str],
) -> Dict[str, Dict[str, float]]:
    """§3.7.4 per-attribute density delta.

    Plain-language question: "Per attribute, what fraction of fused
    rows have a non-null value, and how does that compare to silver?".
    """
    out: Dict[str, Dict[str, float]] = {}
    for attribute, col_type in column_types.items():
        if col_type == "identifier":
            continue
        if attribute not in pipe_df.columns or attribute not in silver_df.columns:
            continue
        reference_density = 1.0 - float(silver_df[attribute].isna().mean())
        pipe_density = 1.0 - float(pipe_df[attribute].isna().mean())
        out[attribute] = {
            "reference_density": reference_density,
            "pipe_density": pipe_density,
            "delta": pipe_density - reference_density,
        }
    return out


# ---------------------------------------------------------------------------
# §3.7.6 per-cluster fully-correct rate
# ---------------------------------------------------------------------------


def per_cluster_fully_correct_rate(
    per_cluster_correctness: pd.DataFrame,
    evaluable_attributes: Sequence[str],
) -> float:
    """§3.7.6 per-cluster fully-correct rate.

    Plain-language question: "Of all aligned clusters, in what fraction
    were *all* attribute values simultaneously correct?". A
    downstream consumer that treats a fused entity as a single record
    cares more about fully-correct entities than about partial
    correctness.
    """
    if per_cluster_correctness.empty or not evaluable_attributes:
        return 0.0
    rows_with_data = per_cluster_correctness.dropna(
        subset=list(evaluable_attributes), how="all"
    )
    if rows_with_data.empty:
        return 0.0

    def _all_correct(row: pd.Series) -> bool:
        flags = [row.get(attr) for attr in evaluable_attributes]
        flags = [f for f in flags if f is not None and not _is_missing(f)]
        if not flags:
            return False
        return all(bool(f) for f in flags)

    return float(rows_with_data.apply(_all_correct, axis=1).mean())
