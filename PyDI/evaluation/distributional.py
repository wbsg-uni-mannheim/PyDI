"""
Distributional metrics for end-to-end pipeline evaluation.

This module implements:

* **Schema diff** — which columns are in pipeline-only, silver-only,
  shared; which shared columns have dtype mismatches. The per-column
  per-column drift metrics are skipped on mismatched columns.
* **Cluster-size distribution comparison** — Wasserstein-1 (primary)
  on the empirical size distribution, JS divergence as a smoothed
  symmetric alternative, plus summary statistics (singleton rate,
  max/mean size, gini) for both pipeline and silver.
* **Type-routed per-column metrics** — dispatches per ``column_types``
  on a column name → ``categorical | numerical | text | datetime |
  list | identifier`` mapping.
* **Universal column drift** — JS divergence on string-cast +
  binned-numerical histograms per column, plus an unweighted mean
  across columns ("column_drift").

Every metric is intended to answer a plain-language question about
how the pipeline compares to silver. See ``plans/plan_e2e_metrics.md``
§2.3 for the framing.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Histogram + divergence helpers
# ---------------------------------------------------------------------------


def _hashable(value: Any) -> Any:
    """Coerce list / set / dict cell values into a hashable form so they
    can be counted in :class:`collections.Counter`. Tuples and frozensets
    are already hashable and pass through; unhashable nested structures
    fall back to their ``repr`` so categorical histograms still produce a
    valid distribution (per-cell stability matters more than structural
    equality here).
    """
    if isinstance(value, list):
        return tuple(_hashable(v) for v in value)
    if isinstance(value, set):
        return tuple(sorted((_hashable(v) for v in value), key=repr))
    if isinstance(value, dict):
        return tuple(
            sorted(
                ((k, _hashable(v)) for k, v in value.items()),
                key=lambda kv: repr(kv[0]),
            )
        )
    try:
        hash(value)
        return value
    except TypeError:
        return repr(value)


def _values_to_probability(values: Iterable[Any]) -> Dict[Any, float]:
    counts = Counter(_hashable(v) for v in values if not _is_nan(v))
    total = sum(counts.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in counts.items()}


def _is_nan(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _aligned_probability_vectors(
    p: Mapping[Any, float], q: Mapping[Any, float]
) -> Tuple[np.ndarray, np.ndarray]:
    keys = sorted(set(p.keys()) | set(q.keys()), key=lambda x: str(x))
    pv = np.array([p.get(k, 0.0) for k in keys], dtype=float)
    qv = np.array([q.get(k, 0.0) for k in keys], dtype=float)
    return pv, qv


def jensen_shannon_divergence(
    p: Mapping[Any, float], q: Mapping[Any, float], *, base: float = 2.0
) -> float:
    """Jensen-Shannon divergence between two discrete distributions.

    Bounded in ``[0, log(base) base = 1]`` for ``base = 2``. Symmetric;
    smoothed via the average ``m = (p + q) / 2`` so zero-support
    categories don't blow up.

    Parameters
    ----------
    p, q : mapping
        Probability mass functions keyed by category.
    base : float, default 2.0
        Logarithm base. ``2.0`` makes JS bounded in ``[0, 1]``.

    Returns
    -------
    float
        Jensen-Shannon divergence; ``0.0`` when both empty.
    """
    pv, qv = _aligned_probability_vectors(p, q)
    if pv.sum() == 0 and qv.sum() == 0:
        return 0.0
    if pv.sum() == 0 or qv.sum() == 0:
        return 1.0 if base == 2.0 else float(np.log(base))

    m = 0.5 * (pv + qv)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(a > 0, np.log(a / b) / np.log(base), 0.0)
        return float(np.sum(np.where((a > 0) & (b > 0), a * ratio, 0.0)))

    return 0.5 * (_kl(pv, m) + _kl(qv, m))


def total_variation_distance(p: Mapping[Any, float], q: Mapping[Any, float]) -> float:
    """Half the L1 distance between two probability distributions. Bounded ``[0, 1]``."""
    pv, qv = _aligned_probability_vectors(p, q)
    return 0.5 * float(np.sum(np.abs(pv - qv)))


def wasserstein_1d(values_a: Iterable[float], values_b: Iterable[float]) -> float:
    """Wasserstein-1 (earth mover's) distance between two 1D samples.

    Equivalent to the L1 distance between sorted-sample CDFs. Symmetric
    and unbounded (units of the input axis). Returns ``0.0`` when
    either input is empty (with a debug-log notice).
    """
    a = np.asarray([float(v) for v in values_a if not _is_nan(v)], dtype=float)
    b = np.asarray([float(v) for v in values_b if not _is_nan(v)], dtype=float)
    if a.size == 0 or b.size == 0:
        logger.debug("wasserstein_1d: empty input on one side; returning 0")
        return 0.0
    try:
        from scipy.stats import wasserstein_distance

        return float(wasserstein_distance(a, b))
    except ImportError:
        a_sorted = np.sort(a)
        b_sorted = np.sort(b)
        n = max(len(a_sorted), len(b_sorted))
        quantiles = np.linspace(0.0, 1.0, n)
        a_q = np.quantile(a_sorted, quantiles)
        b_q = np.quantile(b_sorted, quantiles)
        return float(np.mean(np.abs(a_q - b_q)))


# ---------------------------------------------------------------------------
# Schema diff
# ---------------------------------------------------------------------------


def schema_diff(pipe_df: pd.DataFrame, silver_df: pd.DataFrame) -> Dict[str, Any]:
    """Compare schemas of pipeline-fused and silver dataframes.

    Returns
    -------
    dict
        Keys: ``columns_shared``, ``columns_pipe_only``,
        ``columns_silver_only``, ``dtype_mismatches`` (list of
        ``{column, pipe_dtype, silver_dtype}``). The runner uses this
        to decide which per-column metrics to skip.
    """
    pipe_cols = list(pipe_df.columns)
    silver_cols = list(silver_df.columns)
    pipe_set = set(pipe_cols)
    silver_set = set(silver_cols)

    shared = sorted(pipe_set & silver_set)
    pipe_only = sorted(pipe_set - silver_set)
    silver_only = sorted(silver_set - pipe_set)

    dtype_mismatches: List[Dict[str, str]] = []
    for column in shared:
        pipe_dtype = str(pipe_df[column].dtype)
        silver_dtype = str(silver_df[column].dtype)
        if pipe_dtype != silver_dtype:
            dtype_mismatches.append(
                {
                    "column": column,
                    "pipe_dtype": pipe_dtype,
                    "silver_dtype": silver_dtype,
                }
            )

    return {
        "columns_shared": shared,
        "columns_pipe_only": pipe_only,
        "columns_silver_only": silver_only,
        "dtype_mismatches": dtype_mismatches,
    }


# ---------------------------------------------------------------------------
# Cluster-size distribution comparison
# ---------------------------------------------------------------------------


def _gini(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    sorted_vals = np.sort(values.astype(float))
    n = sorted_vals.size
    if sorted_vals.sum() == 0:
        return 0.0
    cumulative = np.cumsum(sorted_vals)
    return float((n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n)


def cluster_size_summary(
    sizes_pipe: Iterable[int], sizes_silver: Iterable[int]
) -> Dict[str, float]:
    """Compare two empirical cluster-size distributions.

    Plain-language question: "Are the pipeline's clusters shaped the
    same way the silver's clusters are shaped?". The headline number
    is Wasserstein-1 — "average number of records you'd have to move
    to convert one distribution into the other". JS divergence is
    reported as a smoothed symmetric secondary; singleton/max/mean/gini
    on each side support diagnostic reading.

    Returns
    -------
    dict
        Keys: ``wasserstein_1``, ``js_divergence``,
        ``singleton_rate_pipe``, ``singleton_rate_silver``,
        ``singleton_rate_delta``, ``max_size_pipe``, ``max_size_silver``,
        ``mean_size_pipe``, ``mean_size_silver``, ``gini_pipe``,
        ``gini_silver``.
    """
    sp = np.asarray([s for s in sizes_pipe if s is not None], dtype=int)
    ss = np.asarray([s for s in sizes_silver if s is not None], dtype=int)

    p = _values_to_probability(sp.tolist())
    q = _values_to_probability(ss.tolist())

    return {
        "wasserstein_1": wasserstein_1d(sp.astype(float), ss.astype(float)),
        "js_divergence": jensen_shannon_divergence(p, q),
        "singleton_rate_pipe": (float(np.mean(sp == 1)) if sp.size else 0.0),
        "singleton_rate_silver": (float(np.mean(ss == 1)) if ss.size else 0.0),
        "singleton_rate_delta": (
            (float(np.mean(sp == 1)) if sp.size else 0.0)
            - (float(np.mean(ss == 1)) if ss.size else 0.0)
        ),
        "max_size_pipe": int(sp.max()) if sp.size else 0,
        "max_size_silver": int(ss.max()) if ss.size else 0,
        "mean_size_pipe": float(sp.mean()) if sp.size else 0.0,
        "mean_size_silver": float(ss.mean()) if ss.size else 0.0,
        "gini_pipe": _gini(sp) if sp.size else 0.0,
        "gini_silver": _gini(ss) if ss.size else 0.0,
    }


# ---------------------------------------------------------------------------
# Universal column drift (§2.3 universal metric)
# ---------------------------------------------------------------------------


def _string_cast_histogram(
    values: Iterable[Any],
    *,
    is_numeric: bool = False,
    bins: int = 50,
    shared_edges: Optional[np.ndarray] = None,
) -> Dict[Any, float]:
    """Build a probability histogram with uniform representation.

    Strategy per §2.3: string-cast everything (so categorical / text
    columns hash uniformly) and bin numerical columns to ``bins``
    fixed-width buckets first. Missing values are excluded.

    When comparing two numerical series the caller must pass shared
    ``shared_edges`` so both histograms use the same bucket labels;
    otherwise each series gets its own min/max-derived edges and the
    resulting histograms are spuriously disjoint.
    """
    clean = [v for v in values if not _is_nan(v)]
    if is_numeric:
        numeric: List[float] = []
        for v in clean:
            try:
                numeric.append(float(v))
            except (TypeError, ValueError):
                continue
        if not numeric:
            return {}
        arr = np.asarray(numeric, dtype=float)
        if arr.size == 0:
            return {}
        if shared_edges is not None:
            edges = shared_edges
        else:
            lo, hi = float(arr.min()), float(arr.max())
            if hi == lo:
                return {f"[{lo:.6g}]": 1.0}
            edges = np.linspace(lo, hi, bins + 1)
        if edges[0] == edges[-1]:
            return {f"[{edges[0]:.6g}]": 1.0}
        n_bins = len(edges) - 1
        idx = np.clip(np.digitize(arr, edges[1:-1]), 0, n_bins - 1)
        bucket_labels = [f"[{edges[i]:.6g},{edges[i + 1]:.6g})" for i in range(n_bins)]
        counts: Dict[Any, int] = {}
        for i in idx:
            label = bucket_labels[int(i)]
            counts[label] = counts.get(label, 0) + 1
        total = sum(counts.values())
        return {k: v / total for k, v in counts.items()}

    return _values_to_probability([str(v) for v in clean])


def column_drift(
    pipe_series: pd.Series,
    silver_series: pd.Series,
    *,
    is_numeric: bool = False,
    bins: int = 50,
) -> float:
    """Universal per-column drift metric (JS divergence; bounded ``[0, 1]``).

    Plain-language question: "Did this column's overall value
    distribution drift between pipeline and silver, on a uniform
    string-cast (or numerical-binned) representation?". This is the
    headline cross-column metric — comparable across column types so
    it can be averaged. The diagnostic story is *only* readable from
    the type-routed metrics; ``column_drift`` is for ranking, not
    diagnosis.

    For numerical columns the bin edges are computed from the union
    of both inputs so the histograms share bucket labels — otherwise
    a small shift would spuriously max out JS divergence at 1.0.
    """
    shared_edges: Optional[np.ndarray] = None
    if is_numeric:
        pipe_nums = pd.to_numeric(pipe_series, errors="coerce").dropna().to_numpy()
        silver_nums = pd.to_numeric(silver_series, errors="coerce").dropna().to_numpy()
        if pipe_nums.size and silver_nums.size:
            lo = float(min(pipe_nums.min(), silver_nums.min()))
            hi = float(max(pipe_nums.max(), silver_nums.max()))
            if hi > lo:
                shared_edges = np.linspace(lo, hi, bins + 1)
    p = _string_cast_histogram(
        pipe_series, is_numeric=is_numeric, bins=bins, shared_edges=shared_edges
    )
    q = _string_cast_histogram(
        silver_series, is_numeric=is_numeric, bins=bins, shared_edges=shared_edges
    )
    return jensen_shannon_divergence(p, q)


# ---------------------------------------------------------------------------
# Per-column type-routed metrics (§2.3 table)
# ---------------------------------------------------------------------------


COLUMN_TYPES = {
    "categorical",
    "numerical",
    "text",
    "datetime",
    "list",
    "identifier",
}


def _nan_rate(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    return float(series.isna().mean())


def _cardinality(series: pd.Series) -> int:
    return int(series.dropna().astype(str).nunique())


def categorical_metrics(
    pipe_series: pd.Series, silver_series: pd.Series
) -> Dict[str, float]:
    """JS divergence on the value-frequency histogram.

    TV distance was dropped in the v2 rework — JS and TV gave near-
    identical signal on every realistic scenario, so we keep only JS
    (consistent with the universal ``column_drift`` choice). Call
    :func:`total_variation_distance` directly if you need TV.
    """
    p = _values_to_probability(pipe_series.tolist())
    q = _values_to_probability(silver_series.tolist())
    return {"js_divergence": jensen_shannon_divergence(p, q)}


def numerical_metrics(
    pipe_series: pd.Series, silver_series: pd.Series
) -> Dict[str, float]:
    """Wasserstein-1 + KS statistic on the empirical distribution."""
    pipe_vals = pd.to_numeric(pipe_series, errors="coerce").dropna().to_numpy()
    silver_vals = pd.to_numeric(silver_series, errors="coerce").dropna().to_numpy()
    out: Dict[str, float] = {
        "wasserstein_1": wasserstein_1d(pipe_vals, silver_vals),
    }
    try:
        from scipy.stats import ks_2samp

        if pipe_vals.size and silver_vals.size:
            out["ks_statistic"] = float(ks_2samp(pipe_vals, silver_vals).statistic)
        else:
            out["ks_statistic"] = 0.0
    except ImportError:
        logger.debug("scipy not available; KS statistic skipped")
    return out


def text_metrics(pipe_series: pd.Series, silver_series: pd.Series) -> Dict[str, float]:
    """Wasserstein-1 on string-length distribution + JS on token frequencies."""
    pipe_str = pipe_series.dropna().astype(str)
    silver_str = silver_series.dropna().astype(str)

    pipe_lengths = pipe_str.map(len).to_numpy()
    silver_lengths = silver_str.map(len).to_numpy()

    pipe_tokens: List[str] = []
    silver_tokens: List[str] = []
    for value in pipe_str:
        pipe_tokens.extend(_tokenize(value))
    for value in silver_str:
        silver_tokens.extend(_tokenize(value))

    return {
        "length_wasserstein_1": wasserstein_1d(pipe_lengths, silver_lengths),
        "token_js_divergence": jensen_shannon_divergence(
            _values_to_probability(pipe_tokens),
            _values_to_probability(silver_tokens),
        ),
    }


def _tokenize(value: str) -> List[str]:
    import string

    cleaned = value.lower().translate(
        str.maketrans(string.punctuation, " " * len(string.punctuation))
    )
    return [tok for tok in cleaned.split() if tok]


def datetime_metrics(
    pipe_series: pd.Series, silver_series: pd.Series
) -> Dict[str, float]:
    """Wasserstein-1 on epoch seconds (reported in days)."""
    # utc=True forces a tz-aware datetime64[ns, UTC] result even when the
    # inputs carry mixed time zones; without it pandas returns an object
    # Series of Timestamps and the epoch-second cast below raises
    # ("int() argument must be ... not 'Timestamp'"). tz_convert(None)
    # then drops the tz (keeping the instant) so .astype('int64') — which
    # is only defined on tz-naive datetime64 — is well-formed.
    pipe_dt = pd.to_datetime(pipe_series, errors="coerce", utc=True).dropna()
    silver_dt = pd.to_datetime(silver_series, errors="coerce", utc=True).dropna()
    pipe_secs = pipe_dt.dt.tz_convert(None).astype("int64").to_numpy() / 1e9
    silver_secs = silver_dt.dt.tz_convert(None).astype("int64").to_numpy() / 1e9
    w_seconds = wasserstein_1d(pipe_secs, silver_secs)
    return {"wasserstein_1_days": w_seconds / 86400.0}


def compute_type_routed_metrics(
    pipe_df: pd.DataFrame,
    silver_df: pd.DataFrame,
    column_types: Mapping[str, str],
    *,
    skipped_columns: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    """Dispatch per-column type-routed metrics per ``column_types`` map.

    Plain-language question per column: "Did the values for this
    column drift in the way that matters for its data type? (e.g. for
    a date column, by how many days on average?)".

    Returns a long-form list of ``{column, type, metric, value,
    silver_value, pipe_value}`` rows ready to be concatenated into
    ``column_metrics.csv``. ``silver_value`` / ``pipe_value`` are only
    populated for the always-also-report metrics (NaN rate, cardinality)
    where each side's raw number is informative.

    Skipped on:

    * Columns tagged ``identifier`` — no distributional comparison.
    * Columns tagged ``list`` — handled at cluster level by
      :mod:`PyDI.evaluation.attribute_quality` (§3.7.3).
    * Columns listed in ``skipped_columns`` (schema-diff mismatches).
    """
    skipped = set(skipped_columns or [])
    rows: List[Dict[str, Any]] = []

    shared_columns = [c for c in pipe_df.columns if c in silver_df.columns]

    for column in shared_columns:
        if column in skipped:
            continue
        col_type = column_types.get(column)
        if col_type is None:
            logger.debug(
                "Column %r has no entry in column_types; skipping per-column metrics",
                column,
            )
            continue
        if col_type not in COLUMN_TYPES:
            logger.warning(
                "Unknown column type %r for column %r; skipping", col_type, column
            )
            continue
        if col_type in {"identifier", "list"}:
            continue

        pipe_series = pipe_df[column]
        silver_series = silver_df[column]

        type_routed: Dict[str, float]
        if col_type == "categorical":
            type_routed = categorical_metrics(pipe_series, silver_series)
        elif col_type == "numerical":
            type_routed = numerical_metrics(pipe_series, silver_series)
        elif col_type == "text":
            type_routed = text_metrics(pipe_series, silver_series)
        elif col_type == "datetime":
            type_routed = datetime_metrics(pipe_series, silver_series)
        else:
            type_routed = {}

        is_numeric = col_type == "numerical"
        rows.append(
            {
                "column": column,
                "type": col_type,
                "metric": "column_drift",
                "value": column_drift(
                    pipe_series, silver_series, is_numeric=is_numeric
                ),
                "silver_value": None,
                "pipe_value": None,
            }
        )

        for metric_name, value in type_routed.items():
            rows.append(
                {
                    "column": column,
                    "type": col_type,
                    "metric": metric_name,
                    "value": value,
                    "silver_value": None,
                    "pipe_value": None,
                }
            )

        silver_nan = _nan_rate(silver_series)
        pipe_nan = _nan_rate(pipe_series)
        rows.append(
            {
                "column": column,
                "type": col_type,
                "metric": "nan_rate_delta",
                "value": pipe_nan - silver_nan,
                "silver_value": silver_nan,
                "pipe_value": pipe_nan,
            }
        )

        silver_card = _cardinality(silver_series)
        pipe_card = _cardinality(pipe_series)
        rows.append(
            {
                "column": column,
                "type": col_type,
                "metric": "cardinality_delta",
                "value": pipe_card - silver_card,
                "silver_value": silver_card,
                "pipe_value": pipe_card,
            }
        )

    return rows


def column_drift_panel(
    pipe_df: pd.DataFrame,
    silver_df: pd.DataFrame,
    column_types: Mapping[str, str],
    *,
    skipped_columns: Optional[Iterable[str]] = None,
) -> Dict[str, float]:
    """Compute the universal ``column_drift`` per column plus the mean.

    Excludes ``identifier`` / ``list`` columns (per §2.3 table) and any
    column in ``skipped_columns`` (e.g. schema-diff mismatches).
    """
    skipped = set(skipped_columns or [])
    shared_columns = [c for c in pipe_df.columns if c in silver_df.columns]
    per_column: Dict[str, float] = {}
    for column in shared_columns:
        if column in skipped:
            continue
        col_type = column_types.get(column)
        if col_type is None or col_type in {"identifier", "list"}:
            continue
        per_column[column] = column_drift(
            pipe_df[column],
            silver_df[column],
            is_numeric=(col_type == "numerical"),
        )
    if per_column:
        per_column_with_mean = dict(per_column)
        per_column_with_mean["mean"] = float(np.mean(list(per_column.values())))
        return per_column_with_mean
    return {"mean": 0.0}
