"""Stage-agnostic metric helpers for committee validation.

This module contains pure metric computations used by every committee
runner (schema-matching, entity-matching, fusion). Keeping them flat and
dict-based means :func:`delta` is a trivial key-wise subtraction with no
stage coupling.

The committee-validated augmentation loop described in
``knobs/cross_cutting.md`` compares metrics measured on a variant
against metrics measured on the original baseline. A *collapse* is
defined as "the metric has dropped so far that the variant is no longer
informative", and is detected by :func:`collapse_flag`.

Notes
-----
This file is named ``validation_metrics`` (not ``metrics``) to avoid
colliding with any future top-level metric module in ``usecases_synthetic.lib``.
"""

from __future__ import annotations

from typing import Hashable, Iterable, Mapping

import pandas as pd

DEFAULT_COLLAPSE_DROP: float = 0.5
DEFAULT_COLLAPSE_FLOOR: float = 0.15


def f1(tp: int, fp: int, fn: int) -> float:
    """Compute F1 from true/false positive/negative counts.

    Parameters
    ----------
    tp : int
        Number of true positives.
    fp : int
        Number of false positives.
    fn : int
        Number of false negatives.

    Returns
    -------
    float
        F1 score in ``[0.0, 1.0]``. Returns ``0.0`` when
        ``tp + fp + fn == 0``.
    """
    if tp + fp + fn == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def precision_recall_f1(
    pred: Iterable[Hashable],
    gold: Iterable[Hashable],
) -> dict[str, float]:
    """Compute set-based precision / recall / F1.

    Parameters
    ----------
    pred : iterable of hashable
        Predicted items. Duplicates are collapsed to a set.
    gold : iterable of hashable
        Ground-truth items. Duplicates are collapsed to a set.

    Returns
    -------
    dict[str, float]
        Keys ``"precision"``, ``"recall"``, ``"f1"``, ``"tp"``,
        ``"fp"``, ``"fn"``.

    Notes
    -----
    Empty predictions and empty gold both yield zero metrics — there is
    no signal to report. Callers that want ``precision = 1.0`` for empty
    predictions must handle that case themselves.
    """
    pred_set = set(pred)
    gold_set = set(gold)
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall > 0.0:
        f1_score = 2 * precision * recall / (precision + recall)
    else:
        f1_score = 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1_score,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }


def macro_f1(per_partition: Mapping[str, Mapping[str, float]]) -> float:
    """Macro-average an ``"f1"`` metric across partitions.

    Parameters
    ----------
    per_partition : mapping
        Mapping from partition label (e.g. a source pair) to a flat
        metric dict containing an ``"f1"`` key. Partitions whose metric
        dict lacks ``"f1"`` are ignored.

    Returns
    -------
    float
        Arithmetic mean of the partitions' ``"f1"`` values, or ``0.0``
        when no partition contributes.
    """
    values = [
        partition["f1"]
        for partition in per_partition.values()
        if "f1" in partition
    ]
    if not values:
        return 0.0
    return sum(values) / len(values)


def per_attribute_accuracy(
    fused: pd.DataFrame,
    gold: pd.DataFrame,
    columns: Iterable[str],
    *,
    id_column: str = "id",
) -> dict[str, float]:
    """Compute per-attribute accuracy of a fused DataFrame against gold.

    Rows are joined on ``id_column``. For each requested column, the
    fraction of joined rows whose fused value matches the gold value is
    reported. NaN == NaN is counted as a match so that missingness in
    gold doesn't penalise a correct "no value" prediction.

    Parameters
    ----------
    fused : pandas.DataFrame
        Fused records produced by a fusion strategy.
    gold : pandas.DataFrame
        Gold-standard fused records. Must contain ``id_column``.
    columns : iterable of str
        Columns to score. Columns missing from either side contribute
        ``0.0``.
    id_column : str, optional
        Primary key used to align rows. Default ``"id"``.

    Returns
    -------
    dict[str, float]
        Mapping from attribute name to accuracy in ``[0.0, 1.0]``.
    """
    if id_column not in fused.columns or id_column not in gold.columns:
        return {col: 0.0 for col in columns}

    merged = fused.merge(
        gold,
        on=id_column,
        how="inner",
        suffixes=("_pred", "_gold"),
    )
    if len(merged) == 0:
        return {col: 0.0 for col in columns}

    out: dict[str, float] = {}
    for col in columns:
        pred_col = f"{col}_pred"
        gold_col = f"{col}_gold"
        if pred_col not in merged.columns or gold_col not in merged.columns:
            out[col] = 0.0
            continue
        pred_series = merged[pred_col]
        gold_series = merged[gold_col]
        both_nan = pred_series.isna() & gold_series.isna()
        equal = pred_series == gold_series
        matches = (equal | both_nan).sum()
        out[col] = float(matches) / len(merged)
    return out


def delta(
    baseline: Mapping[str, float],
    measured: Mapping[str, float],
) -> dict[str, float]:
    """Key-wise difference ``measured - baseline``.

    Keys present in only one mapping default to ``0.0`` on the missing
    side. This is what the committee loop calls "delta from baseline"
    for monotonicity checks.

    Parameters
    ----------
    baseline : mapping
        Baseline metric dict.
    measured : mapping
        Variant metric dict.

    Returns
    -------
    dict[str, float]
        ``{key: measured[key] - baseline[key]}`` for every key in either
        input.
    """
    keys = set(baseline) | set(measured)
    return {
        key: float(measured.get(key, 0.0)) - float(baseline.get(key, 0.0))
        for key in keys
    }


def collapse_flag(
    measured: Mapping[str, float],
    baseline: Mapping[str, float],
    *,
    metric: str = "f1",
    max_drop: float = DEFAULT_COLLAPSE_DROP,
    floor: float = DEFAULT_COLLAPSE_FLOOR,
) -> bool:
    """Return ``True`` iff a metric collapsed below usable signal.

    The committee loop treats either condition as a collapse:

    1. The metric dropped by more than ``max_drop`` from the baseline.
    2. The absolute metric value fell below ``floor``.

    Parameters
    ----------
    measured : mapping
        Variant metric dict.
    baseline : mapping
        Baseline metric dict.
    metric : str, optional
        Which metric key to check. Default ``"f1"``.
    max_drop : float, optional
        Maximum tolerated drop. Default ``0.5``.
    floor : float, optional
        Minimum tolerated absolute value. Default ``0.15``.

    Returns
    -------
    bool
        ``True`` if the metric collapsed, ``False`` otherwise.
    """
    measured_value = float(measured.get(metric, 0.0))
    baseline_value = float(baseline.get(metric, 0.0))
    if measured_value < floor:
        return True
    if (baseline_value - measured_value) > max_drop:
        return True
    return False
