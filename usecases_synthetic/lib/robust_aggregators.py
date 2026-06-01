"""Robust numeric aggregators for the synthetic fusion committee.

These strategies complement the simple aggregators in
``PyDI.fusion.conflict_resolution.numeric`` (``mean``, ``median``,
``maximum``, ``minimum``) with robust-statistics alternatives that
handle heavy-tailed distributions, symmetric outliers, or heavy
contamination without requiring a truth-discovery pass.

Cell-local and fully deterministic; no sampling, no tuning from labels.

Three estimators are provided:

- ``trimmed_mean`` — discards the top / bottom ``trim`` fraction before
  averaging. Robust to symmetric outliers.
- ``huber_m_estimator`` — iterative Huber M-estimate solver. Downweights
  extreme residuals beyond ``k * sigma`` without discarding them.
- ``median_of_means`` — partitions values into ``n_blocks`` groups,
  computes the mean within each group, then returns the median of the
  block means. Heavy-tail optimal with provable sub-Gaussian error bounds.

Placement rationale
-------------------
This module lives under ``usecases_synthetic/lib/`` rather than
``PyDI/fusion/conflict_resolution/`` because the synthetic fusion
committee is the only caller and
``plans/plan_committee_finalization.md`` §Process-requirement item 4
directs adapters to the synthetic tree to keep ``PyDI/`` read-only from
the synthetic pipeline's perspective. The three functions still
implement the ``PyDI.fusion.base.ConflictResolutionFunction`` protocol
(``(values, **kwargs) -> (value, confidence, metadata)``) so they plug
straight into ``fusion_committee.yaml`` strategy specs.

References
----------
- Huber, P. J. (1964). "Robust Estimation of a Location Parameter".
  Annals of Mathematical Statistics, 35(1), 73-101.
- Nemirovskij, A. S., Yudin, D. B. (1983). "Problem complexity and method
  efficiency in optimization" (median-of-means construction).
- Lerasle, M., Oliveira, R. I. (2011). "Robust empirical mean estimators".
  arXiv:1112.3914.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

FusionResult = Tuple[Any, float, Dict[str, Any]]


# ---------------------------------------------------------------------------
# Validity helpers (inlined rather than imported from PyDI internals).
#
# Mirrors ``PyDI.fusion.base._is_valid_value`` / ``PyDI.fusion.conflict_resolution.utils._filter_valid_values``
# so the synthetic module stays fully decoupled from PyDI's private surface.
# If either helper in PyDI evolves, this copy should follow — reference the
# canonical implementation during review.
# ---------------------------------------------------------------------------


def _is_valid_value(value: Any) -> bool:
    """Check whether ``value`` is fusable (not None, not NA, not empty list)."""
    if value is None:
        return False
    if isinstance(value, (list, tuple)):
        return len(value) > 0
    try:
        if isinstance(value, np.ndarray):
            return value.size > 0
    except Exception:
        pass
    try:
        return not pd.isna(value)
    except Exception:
        return True


def _filter_valid_values(values: List[Any]) -> List[Any]:
    """Drop ``None`` / NaN / empty-list entries from ``values``."""
    return [v for v in values if _is_valid_value(v)]


def _coerce_numeric(values: List[Any]) -> List[float]:
    """Convert values to floats, silently dropping non-numeric entries."""
    numeric: List[float] = []
    for v in values:
        try:
            numeric.append(float(v))
        except (TypeError, ValueError):
            continue
    return numeric


def trimmed_mean(values: List[Any], trim: float = 0.1, **kwargs: Any) -> FusionResult:
    """Symmetric trimmed mean.

    Parameters
    ----------
    values : List[Any]
        Values to aggregate. Non-numeric entries are dropped.
    trim : float, default 0.1
        Fraction of observations to remove from each tail before averaging.
        Must satisfy ``0.0 <= trim < 0.5``. The plan's [§C3 rationale]
        suggests ``0.1`` as the default for companies-scale numerics.

    Returns
    -------
    FusionResult
        ``(resolved_value, confidence, metadata)``. Confidence is 0.0 when
        no numeric values survive, 1.0 when only one numeric value is
        provided (no trimming possible), and ``max(0.1, 1 - cv_trimmed)``
        otherwise where ``cv_trimmed`` is the coefficient of variation of
        the trimmed sample.
    """
    if not 0.0 <= trim < 0.5:
        raise ValueError(f"trim must be in [0.0, 0.5), got {trim!r}")

    valid = _filter_valid_values(values)
    if not valid:
        return None, 0.0, {"reason": "no_valid_values", "rule": "trimmed_mean"}

    numeric = _coerce_numeric(valid)
    if not numeric:
        return None, 0.0, {"reason": "no_numeric_values", "rule": "trimmed_mean"}

    if len(numeric) == 1:
        return (
            float(numeric[0]),
            1.0,
            {
                "rule": "trimmed_mean",
                "trim": trim,
                "num_values": 1,
                "num_trimmed": 0,
            },
        )

    arr = np.asarray(numeric, dtype=float)
    result = float(stats.trim_mean(arr, proportiontocut=trim))

    n_trim_each = int(np.floor(trim * len(arr)))
    n_trimmed = min(2 * n_trim_each, len(arr))

    sorted_arr = np.sort(arr)
    lo = n_trim_each
    hi = len(arr) - n_trim_each
    kept = sorted_arr[lo:hi] if hi > lo else sorted_arr

    if len(kept) > 1 and result != 0:
        cv = float(np.std(kept) / abs(result))
        confidence = max(0.1, min(1.0, 1.0 - cv))
    else:
        confidence = 1.0

    return (
        result,
        confidence,
        {
            "rule": "trimmed_mean",
            "trim": trim,
            "num_values": len(numeric),
            "num_trimmed": n_trimmed,
            "range": [float(min(numeric)), float(max(numeric))],
        },
    )


def huber_m_estimator(
    values: List[Any],
    k: float = 1.345,
    max_iter: int = 50,
    tol: float = 1e-6,
    **kwargs: Any,
) -> FusionResult:
    """Iteratively reweighted Huber M-estimator of location.

    Solves ``argmin_mu sum_i rho_k((x_i - mu) / s)`` using the Huber loss
    ``rho_k(u) = u^2 / 2`` for ``|u| <= k`` and ``rho_k(u) = k|u| - k^2/2``
    otherwise. Scale ``s`` is re-estimated each iteration via the MAD.

    Parameters
    ----------
    values : List[Any]
        Values to aggregate. Non-numeric entries are dropped.
    k : float, default 1.345
        Huber tuning constant. ``k = 1.345`` yields 95% efficiency under
        Gaussian noise (Huber 1964, commonly-cited default).
    max_iter : int, default 50
        Maximum IRLS iterations.
    tol : float, default 1e-6
        Convergence tolerance on location update (absolute).

    Returns
    -------
    FusionResult
        ``(resolved_value, confidence, metadata)``. Confidence reflects
        fraction of values that fell inside the Huber quadratic region
        ``|x - mu| / s <= k`` at convergence.
    """
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k!r}")

    valid = _filter_valid_values(values)
    if not valid:
        return None, 0.0, {"reason": "no_valid_values", "rule": "huber_m_estimator"}

    numeric = _coerce_numeric(valid)
    if not numeric:
        return None, 0.0, {"reason": "no_numeric_values", "rule": "huber_m_estimator"}

    if len(numeric) == 1:
        return (
            float(numeric[0]),
            1.0,
            {
                "rule": "huber_m_estimator",
                "k": k,
                "num_values": 1,
                "n_iter": 0,
            },
        )

    arr = np.asarray(numeric, dtype=float)
    mu = float(np.median(arr))
    mad = float(np.median(np.abs(arr - mu)))
    # 1.4826: MAD-to-sigma conversion constant for Gaussian tails.
    sigma = mad * 1.4826 if mad > 0 else float(np.std(arr))
    if sigma == 0:
        return (
            float(mu),
            1.0,
            {
                "rule": "huber_m_estimator",
                "k": k,
                "num_values": len(numeric),
                "n_iter": 0,
                "note": "zero_scale_identical_values",
            },
        )

    n_iter = 0
    for step in range(max_iter):
        n_iter = step + 1
        residuals = (arr - mu) / sigma
        abs_r = np.abs(residuals)
        # Guard against residual == 0 to avoid divide warnings; weight is 1
        # anywhere inside the quadratic region, so those cells never consult
        # the k/|r| branch.
        safe_denom = np.where(abs_r > 0, abs_r, 1.0)
        weights = np.where(abs_r <= k, 1.0, k / safe_denom)
        new_mu = float(np.sum(weights * arr) / np.sum(weights))
        if abs(new_mu - mu) < tol:
            mu = new_mu
            break
        mu = new_mu
        mad = float(np.median(np.abs(arr - mu)))
        sigma = mad * 1.4826 if mad > 0 else sigma

    residuals = (arr - mu) / sigma
    inside = int(np.sum(np.abs(residuals) <= k))
    confidence = max(0.1, min(1.0, inside / len(arr)))

    return (
        float(mu),
        confidence,
        {
            "rule": "huber_m_estimator",
            "k": k,
            "num_values": len(numeric),
            "n_iter": n_iter,
            "scale": float(sigma),
            "num_inliers": inside,
            "range": [float(arr.min()), float(arr.max())],
        },
    )


def median_of_means(
    values: List[Any],
    n_blocks: int | None = None,
    **kwargs: Any,
) -> FusionResult:
    """Median-of-means heavy-tail-optimal location estimator.

    Partitions ``values`` into ``n_blocks`` equal-size blocks, computes the
    mean within each block, then returns the median of the block means.
    For ``n_blocks = 1`` this reduces to the arithmetic mean; for
    ``n_blocks = len(values)`` it reduces to the median. The estimator
    enjoys sub-Gaussian concentration bounds even under heavy-tailed
    noise (Lerasle-Oliveira 2011).

    Parameters
    ----------
    values : List[Any]
        Values to aggregate. Non-numeric entries are dropped.
    n_blocks : int, optional
        Number of blocks. When ``None``, defaults to
        ``max(1, floor(log(n) / log(2)))`` which scales gracefully with
        sample size. When explicitly passed, must satisfy
        ``1 <= n_blocks <= len(numeric_values)``.

    Returns
    -------
    FusionResult
        ``(resolved_value, confidence, metadata)``. Confidence reflects
        block-mean agreement (``1 - cv_of_block_means``, clipped).
    """
    valid = _filter_valid_values(values)
    if not valid:
        return None, 0.0, {"reason": "no_valid_values", "rule": "median_of_means"}

    numeric = _coerce_numeric(valid)
    if not numeric:
        return None, 0.0, {"reason": "no_numeric_values", "rule": "median_of_means"}

    n = len(numeric)
    if n_blocks is None:
        n_blocks = max(1, int(np.floor(np.log(n) / np.log(2)))) if n > 1 else 1

    if not 1 <= n_blocks <= n:
        raise ValueError(
            f"n_blocks must satisfy 1 <= n_blocks <= {n}, got {n_blocks!r}"
        )

    arr = np.asarray(numeric, dtype=float)
    # Split as evenly as possible; leftover values go to the last block.
    block_size = n // n_blocks
    block_means: List[float] = []
    for i in range(n_blocks):
        start = i * block_size
        end = (i + 1) * block_size if i < n_blocks - 1 else n
        block_means.append(float(np.mean(arr[start:end])))

    result = float(np.median(block_means))

    if len(block_means) > 1 and result != 0:
        cv = float(np.std(block_means) / abs(result))
        confidence = max(0.1, min(1.0, 1.0 - cv))
    else:
        confidence = 1.0

    return (
        result,
        confidence,
        {
            "rule": "median_of_means",
            "n_blocks": n_blocks,
            "num_values": n,
            "block_size": block_size,
            "block_means": block_means,
            "range": [float(arr.min()), float(arr.max())],
        },
    )


__all__ = [
    "trimmed_mean",
    "huber_m_estimator",
    "median_of_means",
]
