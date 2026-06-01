"""Scoring helpers for the EM committee runner.

Three kinds of metric:

1. **Reported F1** — closed-set precision / recall / F1 of predicted
   correspondences against a test gold with explicit negatives. Under
   plan_revision.md C10/C11 the committee runs the closed-set scorer
   against two parallel test versions per source pair —
   ``em_test_baseline_pruned.csv`` (Set 1) and
   ``em_test_corner_filled.csv`` (Set 2) — surfacing
   ``f1_baseline_test`` (per-level reference) and ``f1_regen_test``
   (monotonicity surface). The open-set scorer that previously
   evaluated against the original human gold is retired.

2. **Pool agreement** (diagnostic) — how much the predictions overlap with
   the pooled positives from ``pools/<domain>/pooled_positives.csv``.
   This is *asymmetric*: pool-precision (fraction of predictions in the
   pool) and pool-recall (fraction of pool pairs recovered).  Never
   reported as "the F1"; used by M8 to disambiguate collapse from
   hidden-positive noise.

3. **Blocking metrics** — pair recall and reduction ratio for a candidate
   set produced by a blocker, pre-matching.  Introduced in C2.4b to drive
   the blocking committee's "select-best-blocker" step: a candidate set
   with pair_recall ≥ 0.97 is eligible for the matching committee, and
   ties are broken by reduction_ratio (higher = more pruning).

See ``knobs/cross_cutting.md`` Protection-set semantics for rationale.
"""

from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# Reported metric: closed-set F1
# ---------------------------------------------------------------------------


def _correspondences_to_pairs(
    df: pd.DataFrame,
) -> set[tuple[str, str]]:
    """Extract ``{(id1, id2)}`` from a correspondence DataFrame.

    Parameters
    ----------
    df : DataFrame
        Must have columns ``id1`` and ``id2``.

    Returns
    -------
    set of (str, str)
        Unique predicted/gold pairs.
    """
    if df.empty:
        return set()
    return set(zip(df["id1"].astype(str), df["id2"].astype(str), strict=True))


def _gold_positive_pairs(gold: pd.DataFrame) -> set[tuple[str, str]]:
    """Extract the positive pairs from an EM gold DataFrame.

    EM gold files have a ``label`` column (string ``"true"`` / ``"false"``
    or boolean ``True`` / ``False``).  Only positive pairs contribute to
    the P/R/F1 computation.

    Parameters
    ----------
    gold : DataFrame
        Must have columns ``id1``, ``id2``, ``label``.

    Returns
    -------
    set of (str, str)
        Positive gold pairs.
    """
    if gold.empty:
        return set()
    label_col = gold["label"]
    # Handle both string ("true"/"false") and boolean labels.
    if label_col.dtype == object:
        mask = label_col.str.lower() == "true"
    else:
        mask = label_col.astype(bool)
    positives = gold.loc[mask]
    return set(
        zip(
            positives["id1"].astype(str),
            positives["id2"].astype(str),
            strict=True,
        )
    )


def _gold_all_pairs(gold: pd.DataFrame) -> set[tuple[str, str]]:
    """Extract the full set of gold pairs (positive + negative).

    Used by :func:`score_em_correspondences_closed_set` to restrict
    scoring to the gold's universe of judged pairs.

    Parameters
    ----------
    gold : DataFrame
        Must have columns ``id1``, ``id2``.

    Returns
    -------
    set of (str, str)
        Every pair in the gold table, canonicalised ``(a, b)`` with
        ``a < b`` so orientation does not matter.
    """
    if gold.empty:
        return set()
    raw = set(zip(gold["id1"].astype(str), gold["id2"].astype(str), strict=True))
    return {tuple(sorted(p)) for p in raw}


def score_em_correspondences_closed_set(
    pred: pd.DataFrame,
    gold: pd.DataFrame,
) -> dict[str, float]:
    """Closed-set P/R/F1 of predictions vs a test gold with explicit negatives.

    Restricts predictions to the gold's judged pair set before computing
    precision / recall / F1. That is the appropriate semantics for a
    closed-set benchmark (e.g. K2 regenerated gold, which contains both
    positive and negative judgments) and replaced the retired open-set
    scorer (plan_revision.md C10):

    - TP = predicted positives that are also gold positives.
    - FP = predicted positives that fall on gold *negatives*.
    - FN = gold positives the matcher did not predict.

    Predicted pairs outside the gold's universe are **out of scope** —
    they are simply ignored rather than counted as FP. This prevents
    artificial precision collapse when the matcher scans a larger pair
    space than the closed-set benchmark covers.

    Parameters
    ----------
    pred : DataFrame
        Predicted correspondences (columns ``id1``, ``id2``).
    gold : DataFrame
        Closed-set gold with columns ``id1``, ``id2``, ``label``
        (positives *and* negatives).

    Returns
    -------
    dict[str, float]
        Keys ``"precision"``, ``"recall"``, ``"f1"``, ``"tp"``,
        ``"fp"``, ``"fn"``, plus ``"pred_scoped"`` (number of predicted
        pairs that fell within the gold universe — useful diagnostic
        for coverage).
    """
    pred_canon = {tuple(sorted(p)) for p in _correspondences_to_pairs(pred)}
    gold_positive_canon = {tuple(sorted(p)) for p in _gold_positive_pairs(gold)}
    gold_universe = _gold_all_pairs(gold)

    pred_scoped = pred_canon & gold_universe

    tp = len(pred_scoped & gold_positive_canon)
    fp = len(pred_scoped - gold_positive_canon)
    fn = len(gold_positive_canon - pred_scoped)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "pred_scoped": float(len(pred_scoped)),
    }


# ---------------------------------------------------------------------------
# Pool agreement (diagnostic)
# ---------------------------------------------------------------------------


def _pool_to_pairs(pool: pd.DataFrame) -> set[tuple[str, str]]:
    """Extract ``{(id1, id2)}`` from a pooled-positives DataFrame.

    Parameters
    ----------
    pool : DataFrame
        Columns ``id1``, ``id2`` (plus optional ``source_1``,
        ``source_2``, ``pool_agreement``).

    Returns
    -------
    set of (str, str)
        Canonicalised pool pairs.
    """
    if pool.empty:
        return set()
    raw = set(zip(pool["id1"].astype(str), pool["id2"].astype(str), strict=True))
    return {tuple(sorted(p)) for p in raw}


def pool_agreement(
    pred: pd.DataFrame,
    pool: pd.DataFrame | None,
) -> dict[str, float]:
    """Compute the asymmetric pool-agreement diagnostic.

    Parameters
    ----------
    pred : DataFrame
        Predicted correspondences (columns ``id1``, ``id2``).
    pool : DataFrame or None
        Pooled positives.  ``None`` → all-zero result.

    Returns
    -------
    dict[str, float]
        ``"pool_precision"`` — fraction of pred pairs in the pool.
        ``"pool_recall"`` — fraction of pool pairs recovered by pred.
        ``"pool_overlap"`` — number of pairs in both pred and pool.
    """
    if pool is None or pool.empty:
        return {
            "pool_precision": 0.0,
            "pool_recall": 0.0,
            "pool_overlap": 0.0,
        }

    pred_canon = {tuple(sorted(p)) for p in _correspondences_to_pairs(pred)}
    pool_canon = _pool_to_pairs(pool)

    overlap = len(pred_canon & pool_canon)
    pool_prec = overlap / len(pred_canon) if pred_canon else 0.0
    pool_rec = overlap / len(pool_canon) if pool_canon else 0.0

    return {
        "pool_precision": pool_prec,
        "pool_recall": pool_rec,
        "pool_overlap": float(overlap),
    }


def _filter_pool_for_pair(
    pool: pd.DataFrame | None,
    pair: tuple[str, str],
) -> pd.DataFrame:
    """Filter the pooled-positives to a specific source pair.

    Handles both orderings: ``(A, B)`` matches pool rows where
    ``source_1/source_2`` is ``(A, B)`` or ``(B, A)``.

    Parameters
    ----------
    pool : DataFrame or None
        Full pooled positives with ``source_1``, ``source_2`` columns.
    pair : tuple of str
        ``(src1, src2)`` source pair to filter for.

    Returns
    -------
    DataFrame
        Filtered pool rows for this pair (may be empty).
    """
    if pool is None or pool.empty:
        return pd.DataFrame(columns=["id1", "id2"])
    src1, src2 = pair
    mask_fwd = (pool["source_1"] == src1) & (pool["source_2"] == src2)
    mask_rev = (pool["source_1"] == src2) & (pool["source_2"] == src1)
    return pool.loc[mask_fwd | mask_rev]


def score_em_vs_pool(
    pred: pd.DataFrame,
    pool: pd.DataFrame | None,
    pair: tuple[str, str],
) -> dict[str, float]:
    """Precision / recall / F1 of predicted correspondences vs the pool.

    The pool (constructed from independent sources — human baseline and
    PLM matcher) serves as the gold standard.  Predicted pairs not in
    the pool are FP; pool pairs not predicted are FN.

    Parameters
    ----------
    pred : DataFrame
        Predicted correspondences (columns ``id1``, ``id2``).
    pool : DataFrame or None
        Full pooled positives.
    pair : tuple of str
        Source pair to filter the pool for.

    Returns
    -------
    dict[str, float]
        Keys ``"precision"``, ``"recall"``, ``"f1"``, ``"tp"``,
        ``"fp"``, ``"fn"``.
    """
    pair_pool = _filter_pool_for_pair(pool, pair)
    if pair_pool.empty:
        pred_n = len(_correspondences_to_pairs(pred)) if not pred.empty else 0
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "tp": 0.0,
            "fp": float(pred_n),
            "fn": 0.0,
        }

    pred_canon = {tuple(sorted(p)) for p in _correspondences_to_pairs(pred)}
    pool_canon = _pool_to_pairs(pair_pool)

    tp = len(pred_canon & pool_canon)
    fp = len(pred_canon - pool_canon)
    fn = len(pool_canon - pred_canon)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }


# ---------------------------------------------------------------------------
# Blocking metrics: pair recall + reduction ratio
# ---------------------------------------------------------------------------


def blocking_pair_recall(
    candidates: pd.DataFrame,
    gold: pd.DataFrame,
) -> dict[str, float]:
    """Fraction of positive gold pairs retained in a candidate set.

    Pair recall = ``|gold+ ∩ candidates| / |gold+|``, where ``gold+`` is
    the set of positively-labelled pairs in the EM gold.  Used to score a
    blocker: a blocker whose candidate set drops true matches has no
    chance of producing good downstream F1, regardless of which matcher
    is applied.

    The comparison is order-insensitive — ``(a, b)`` matches ``(b, a)``.

    Parameters
    ----------
    candidates : DataFrame
        Candidate pairs produced by a blocker.  Must have columns
        ``id1`` and ``id2``.
    gold : DataFrame
        EM gold DataFrame with columns ``id1``, ``id2``, ``label``.

    Returns
    -------
    dict[str, float]
        Keys ``"pair_recall"``, ``"gold_positives"`` (count of positive
        gold pairs), ``"covered"`` (count retained in candidates),
        ``"missed"`` (count not retained).
    """
    gold_canon = {tuple(sorted(p)) for p in _gold_positive_pairs(gold)}
    if not gold_canon:
        return {
            "pair_recall": 0.0,
            "gold_positives": 0.0,
            "covered": 0.0,
            "missed": 0.0,
        }

    cand_canon = {tuple(sorted(p)) for p in _correspondences_to_pairs(candidates)}
    covered = len(gold_canon & cand_canon)
    missed = len(gold_canon) - covered
    pair_recall = covered / len(gold_canon)

    return {
        "pair_recall": pair_recall,
        "gold_positives": float(len(gold_canon)),
        "covered": float(covered),
        "missed": float(missed),
    }


def reduction_ratio(
    candidates: pd.DataFrame,
    n_left: int,
    n_right: int,
) -> dict[str, float]:
    """Fraction of the full ``|L| × |R|`` pair-space pruned by a blocker.

    Reduction ratio = ``1 − |candidates| / (|L| × |R|)``.  Higher is
    better (more aggressive pruning); 1.0 means the candidate set is
    empty, 0.0 means no pruning.  Deduplicates candidate pairs so a
    blocker emitting the same pair twice does not penalise itself.

    Parameters
    ----------
    candidates : DataFrame
        Candidate pairs produced by a blocker.  Must have columns
        ``id1`` and ``id2``.
    n_left : int
        Row count of the left source.
    n_right : int
        Row count of the right source.

    Returns
    -------
    dict[str, float]
        Keys ``"reduction_ratio"``, ``"candidate_count"`` (deduplicated
        unordered pair count), ``"full_space"`` (``n_left * n_right``).

    Raises
    ------
    ValueError
        If ``n_left`` or ``n_right`` is negative, or ``n_left * n_right``
        is zero (undefined ratio).
    """
    if n_left < 0 or n_right < 0:
        raise ValueError(
            f"n_left and n_right must be non-negative; got {n_left}, {n_right}"
        )
    full_space = n_left * n_right
    if full_space == 0:
        raise ValueError(
            "reduction_ratio is undefined when the pair-space is empty "
            f"(n_left={n_left}, n_right={n_right})"
        )

    cand_canon = {tuple(sorted(p)) for p in _correspondences_to_pairs(candidates)}
    count = len(cand_canon)
    rr = 1.0 - count / full_space

    return {
        "reduction_ratio": rr,
        "candidate_count": float(count),
        "full_space": float(full_space),
    }
