"""Corner-case pair miner for Knob 02 — Entity Niche Density.

.. note::

    ``regenerate_em_splits`` historically ran at K2 time using a snapshot
    of ``ids_present`` captured *before* K4 demotes any records. The
    helper :class:`RegenPools` and :func:`regenerate_em_splits_from_pools`
    let the orchestrator defer the regen emission until **after** K4
    runs, so the regen never references IDs that K4 later removed at
    hard. ``regenerate_em_splits`` itself is unchanged.

Implements ``knobs/knob_02_niche_density.md`` §"Corner-case pair miner
(test regeneration + hard-negative selection)":

- **Recall-biased** union across metrics: if *any* metric flags a pair
  as hard, the pair is classified as a corner case. Contrast with the
  niche-density scorer which is consensus-biased.
- ``hard_match``: same-cluster pair with some metric ``sim < t_match[m]``
  (matchers will find it hard).
- ``hard_non_match``: cross-cluster pair with some metric
  ``sim > t_nonmatch[m]`` (deceptively similar).
- ``label_collision`` adds a hard deterministic rule: any cross-cluster
  pair with identical normalised labels is a ``hard_non_match``.
- ``attribute_overlap`` is intentionally omitted from the miner per the
  knob card (franchise-mates are not automatically corner cases).

The miner is used for (a) measuring the current corner-case ratio and
(b) regenerating the EM test set stratified by corner-case status.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Literal, Sequence

import numpy as np

from .niche_metrics import (
    attribute_overlap,
    lexical_extended_jaccard,
    normalize_label,
)

logger = logging.getLogger(__name__)


CornerCaseKind = Literal["hard_match", "hard_non_match"]

RecordPair = tuple[str, str]
PlmScorer = Callable[[Sequence[RecordPair]], dict[RecordPair, float]]
LlmAdjudicator = Callable[[RecordPair], bool]


HardNegativeVerdict = Literal[
    "keep_strong",
    "keep_adjudicated",
    "drop_above_theta",
    "drop_adjudicated",
    "no_score",
]


@dataclass
class HardNegativeAudit:
    """Per-record-pair audit entry emitted by the hard-negative gate."""

    rid_a: str
    rid_b: str
    plm_score: float | None
    verdict: HardNegativeVerdict
    theta: float
    delta: float
    llm_says_match: bool | None = None


@dataclass
class CornerCasePair:
    """A pair classified as a corner case by one or more metrics.

    Parameters
    ----------
    i, j : int
        Row indices of the pair in the canonical frame (i < j).
    kind : {"hard_match", "hard_non_match"}
        Corner-case classification.
    triggered_by : list of str
        Metric names that flagged this pair as hard.
    """

    i: int
    j: int
    kind: CornerCaseKind
    triggered_by: list[str]


# ---------------------------------------------------------------------------
# Metric thresholds
# ---------------------------------------------------------------------------


@dataclass
class MetricThresholds:
    """Per-metric pair-miner thresholds.

    ``t_match[m]`` — a same-cluster pair with similarity *below*
    ``t_match[m]`` under metric m is a ``hard_match``.
    ``t_nonmatch[m]`` — a cross-cluster pair with similarity *above*
    ``t_nonmatch[m]`` is a ``hard_non_match``.
    """

    t_match: dict[str, float]
    t_nonmatch: dict[str, float]


# ---------------------------------------------------------------------------
# Hard-negative policy (PLM score-margin gate + LLM adjudicator)
# ---------------------------------------------------------------------------


@dataclass
class HardNegativePolicy:
    """Score-margin gate applied to candidate hard-negative record pairs.

    The similarity-based corner-case miner flags pairs as
    ``hard_non_match`` based on decoupled hardness signals (lexical /
    embedding cosine / token overlap). Before promoting such a pair to
    the regenerated EM test set as a hard negative, we additionally
    require an honest matcher (``plm_scorer``) to agree the pair is
    **not** a match. Pairs the matcher thinks are matches would be
    hidden positives, not hard negatives.

    Let ``s = plm_scorer(pair)`` and ``θ = plm_threshold_theta``,
    ``δ = plm_margin_delta``. The gate decides:

    - ``s < θ − δ``            → keep (matcher strongly says no).
    - ``θ − δ ≤ s < θ``        → margin band: ask ``llm_adjudicator``.
      If the LLM says the pair is a match, drop; else keep.
    - ``s ≥ θ``                → drop (matcher says match — likely a
      hidden positive, not a hard negative).

    When ``plm_scorer`` is ``None`` the gate is a no-op and every pair
    is kept (records ``verdict="no_score"`` in the audit). Pairs without
    a recorded score (e.g. record-id missing from one of the source
    frames) are treated the same way.

    Parameters
    ----------
    plm_scorer : callable or None
        Maps a sequence of record-id pairs to a
        ``{(rid_a, rid_b): score}`` dict. Scores are softmax class-1
        probabilities in ``[0, 1]``. Callers route record-id order to
        match their source tables; the gate looks up by the exact key
        it passes in.
    plm_threshold_theta : float
        Decision threshold θ of the PLM. Pinned from the baseline
        measurement pass (companies: 0.36 from D8 calibration).
    plm_margin_delta : float
        Safety margin δ around θ. The margin band is ``[θ−δ, θ)``.
    llm_adjudicator : callable or None
        Maps a single record-id pair to ``True`` if the LLM judges the
        pair a match (i.e. drop it), ``False`` otherwise. When absent
        and the margin band is non-empty, margin-band pairs are
        conservatively **dropped** (no LLM = no way to rescue).
    """

    plm_scorer: PlmScorer | None
    plm_threshold_theta: float
    plm_margin_delta: float
    llm_adjudicator: LlmAdjudicator | None = None
    gate_mode: str = "margin_only"
    """One of ``margin_only`` (default, legacy 3-band logic above) or
    ``full_llm`` (every pair routed to ``llm_adjudicator`` regardless
    of score; PLM score recorded in audit but not used for routing
    decisions). ``full_llm`` mirrors the pool-builder bucket-C policy
    (2026-05-26 follow-up) — defensible when Ditto calibration is in
    transitional state (e.g. step 5/6 reruns where music / companies /
    products use pre-R7-padding-fix checkpoints). Added 2026-05-27
    (step 4h option a)."""


def apply_hard_negative_policy(
    candidate_pairs: Sequence[RecordPair],
    *,
    policy: HardNegativePolicy | None,
) -> tuple[list[RecordPair], list[HardNegativeAudit]]:
    """Filter *candidate_pairs* by the hard-negative score-margin gate.

    Parameters
    ----------
    candidate_pairs : sequence of (str, str)
        Record-id pairs already classified as ``hard_non_match`` by the
        similarity-based miner. Order is the caller's responsibility
        (the gate looks up scores by the exact tuple passed in).
    policy : HardNegativePolicy or None
        When ``None``, the gate is a no-op: every pair is kept and the
        audit is empty.

    Returns
    -------
    kept_pairs : list of (str, str)
        Subset of *candidate_pairs* surviving the gate, preserving
        input order.
    audit : list of HardNegativeAudit
        One entry per input pair, including dropped pairs. The caller
        can persist this to provenance.
    """
    if policy is None or policy.plm_scorer is None:
        no_score_audit: list[HardNegativeAudit] = [
            HardNegativeAudit(
                rid_a=a,
                rid_b=b,
                plm_score=None,
                verdict="no_score",
                theta=(policy.plm_threshold_theta if policy else float("nan")),
                delta=(policy.plm_margin_delta if policy else float("nan")),
            )
            for (a, b) in candidate_pairs
        ]
        return list(candidate_pairs), no_score_audit

    theta = float(policy.plm_threshold_theta)
    delta = float(policy.plm_margin_delta)
    if delta < 0 or theta < 0 or theta > 1:
        raise ValueError(f"invalid PLM threshold/margin: theta={theta}, delta={delta}")

    if policy.gate_mode not in ("margin_only", "full_llm"):
        raise ValueError(
            f"invalid gate_mode {policy.gate_mode!r}; "
            "expected 'margin_only' or 'full_llm'"
        )

    scores = policy.plm_scorer(list(candidate_pairs))

    # full_llm mode: every pair routed to LLM adjudicator regardless of PLM
    # score. PLM score recorded in audit but not used for routing decisions.
    # Mirrors bucket-C policy (pool_builder.build_buckets post-2026-05-26).
    if policy.gate_mode == "full_llm":
        kept_full: list[RecordPair] = []
        audit_full: list[HardNegativeAudit] = []
        for pair in candidate_pairs:
            raw = scores.get(pair)
            score: float | None = float(raw) if raw is not None else None
            if policy.llm_adjudicator is None:
                # No LLM available — conservative drop (full_llm without an
                # adjudicator is an invariant violation; we should never get
                # here in production, but fall through safely).
                audit_full.append(
                    HardNegativeAudit(
                        rid_a=pair[0],
                        rid_b=pair[1],
                        plm_score=score,
                        verdict="drop_adjudicated",
                        theta=theta,
                        delta=delta,
                        llm_says_match=None,
                    )
                )
                continue
            says_match = bool(policy.llm_adjudicator(pair))
            if says_match:
                audit_full.append(
                    HardNegativeAudit(
                        rid_a=pair[0],
                        rid_b=pair[1],
                        plm_score=score,
                        verdict="drop_adjudicated",
                        theta=theta,
                        delta=delta,
                        llm_says_match=True,
                    )
                )
            else:
                audit_full.append(
                    HardNegativeAudit(
                        rid_a=pair[0],
                        rid_b=pair[1],
                        plm_score=score,
                        verdict="keep_adjudicated",
                        theta=theta,
                        delta=delta,
                        llm_says_match=False,
                    )
                )
                kept_full.append(pair)
        return kept_full, audit_full

    # margin_only (legacy 3-band) dispatch follows.
    kept: list[RecordPair] = []
    audit: list[HardNegativeAudit] = []
    for pair in candidate_pairs:
        raw = scores.get(pair)
        if raw is None:
            # Missing score: conservative keep (no evidence the pair is
            # a match). Explicit "no_score" audit row for debugging.
            audit.append(
                HardNegativeAudit(
                    rid_a=pair[0],
                    rid_b=pair[1],
                    plm_score=None,
                    verdict="no_score",
                    theta=theta,
                    delta=delta,
                )
            )
            kept.append(pair)
            continue

        score = float(raw)
        if score >= theta:
            audit.append(
                HardNegativeAudit(
                    rid_a=pair[0],
                    rid_b=pair[1],
                    plm_score=score,
                    verdict="drop_above_theta",
                    theta=theta,
                    delta=delta,
                )
            )
            continue

        if score < theta - delta:
            audit.append(
                HardNegativeAudit(
                    rid_a=pair[0],
                    rid_b=pair[1],
                    plm_score=score,
                    verdict="keep_strong",
                    theta=theta,
                    delta=delta,
                )
            )
            kept.append(pair)
            continue

        # Margin band [theta - delta, theta).
        if policy.llm_adjudicator is None:
            audit.append(
                HardNegativeAudit(
                    rid_a=pair[0],
                    rid_b=pair[1],
                    plm_score=score,
                    verdict="drop_adjudicated",
                    theta=theta,
                    delta=delta,
                    llm_says_match=None,
                )
            )
            continue

        says_match = bool(policy.llm_adjudicator(pair))
        if says_match:
            audit.append(
                HardNegativeAudit(
                    rid_a=pair[0],
                    rid_b=pair[1],
                    plm_score=score,
                    verdict="drop_adjudicated",
                    theta=theta,
                    delta=delta,
                    llm_says_match=True,
                )
            )
        else:
            audit.append(
                HardNegativeAudit(
                    rid_a=pair[0],
                    rid_b=pair[1],
                    plm_score=score,
                    verdict="keep_adjudicated",
                    theta=theta,
                    delta=delta,
                    llm_says_match=False,
                )
            )
            kept.append(pair)

    return kept, audit


# ---------------------------------------------------------------------------
# Dense similarity computation
# ---------------------------------------------------------------------------


def pair_similarities(
    i: int,
    j: int,
    *,
    labels: Sequence[str],
    tfidf_matrix,
    embeddings: np.ndarray | None,
    inner_token_threshold: float,
    stopwords: set[str] | None,
) -> dict[str, float]:
    """Compute per-metric raw similarities for a single ``(i, j)`` pair.

    Returns a dict keyed by metric name with the metrics that can be
    evaluated given the inputs (TF-IDF and embeddings are optional).
    """
    sims: dict[str, float] = {}
    sims["ext_jaccard"] = lexical_extended_jaccard(
        labels[i],
        labels[j],
        inner_token_threshold=inner_token_threshold,
        stopwords=stopwords,
    )
    if tfidf_matrix is not None:
        ai = tfidf_matrix[i]
        aj = tfidf_matrix[j]
        num = float((ai @ aj.T).toarray().ravel()[0])
        da = float(np.sqrt((ai @ ai.T).toarray().ravel()[0]))
        dbv = float(np.sqrt((aj @ aj.T).toarray().ravel()[0]))
        sims["tfidf"] = num / (da * dbv) if da > 0 and dbv > 0 else 0.0
    if embeddings is not None:
        sims["embedding"] = float(
            np.clip(np.dot(embeddings[i], embeddings[j]), -1.0, 1.0)
        )
    return sims


# ---------------------------------------------------------------------------
# Mining
# ---------------------------------------------------------------------------


def classify_pair(
    i: int,
    j: int,
    *,
    sims: dict[str, float],
    same_cluster: bool,
    thresholds: MetricThresholds,
    labels: Sequence[str],
) -> CornerCasePair | None:
    """Classify a single pair as a corner case (or return ``None``).

    Applies the recall-biased union rule: if any metric crosses its
    threshold, the pair is classified. Label collision is a hard rule
    for the cross-cluster case.
    """
    triggered: list[str] = []

    if same_cluster:
        kind: CornerCaseKind = "hard_match"
        # Label-collision same-cluster pairs are always hard_match.
        if normalize_label(labels[i]) and normalize_label(labels[i]) == normalize_label(
            labels[j]
        ):
            triggered.append("label_collision")
        for metric_name, sim in sims.items():
            t = thresholds.t_match.get(metric_name)
            if t is None:
                continue
            if sim < t:
                triggered.append(metric_name)
    else:
        kind = "hard_non_match"
        # Label-collision cross-cluster is a hard deterministic rule.
        if normalize_label(labels[i]) and normalize_label(labels[i]) == normalize_label(
            labels[j]
        ):
            triggered.append("label_collision")
        for metric_name, sim in sims.items():
            t = thresholds.t_nonmatch.get(metric_name)
            if t is None:
                continue
            if sim > t:
                triggered.append(metric_name)

    if not triggered:
        return None
    return CornerCasePair(
        i=i, j=j, kind=kind, triggered_by=list(dict.fromkeys(triggered))
    )


def mine_corner_cases(
    *,
    candidate_pairs: Sequence[tuple[int, int]],
    cluster_of: Sequence[int],
    labels: Sequence[str],
    tfidf_matrix,
    embeddings: np.ndarray | None,
    thresholds: MetricThresholds,
    inner_token_threshold: float = 0.8,
    stopwords: set[str] | None = None,
) -> list[CornerCasePair]:
    """Mine corner-case pairs from a candidate list.

    Parameters
    ----------
    candidate_pairs : sequence of (int, int)
        Row-index pairs to evaluate. Typically the union of
        same-cluster and cross-cluster pairs sampled from the
        post-Knob-2 canonical set.
    cluster_of : sequence of int
        Ground-truth cluster id per entity index. Pairs with equal
        values are same-cluster.
    labels : sequence of str
        Primary labels for label-collision detection.
    tfidf_matrix
        Pre-computed TF-IDF sparse matrix (or ``None`` to skip).
    embeddings : numpy.ndarray or None
        Pre-computed row-normalised embeddings (or ``None`` to skip).
    thresholds : MetricThresholds
        Per-metric thresholds.
    inner_token_threshold : float, default 0.8
        Forwarded to ``lexical_extended_jaccard``.
    stopwords : set of str or None
        Forwarded to ``lexical_extended_jaccard``.

    Returns
    -------
    list of CornerCasePair
        One entry per pair classified as hard (any metric triggered).
    """
    out: list[CornerCasePair] = []
    for i, j in candidate_pairs:
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        same = cluster_of[a] == cluster_of[b] and cluster_of[a] >= 0
        sims = pair_similarities(
            a,
            b,
            labels=labels,
            tfidf_matrix=tfidf_matrix,
            embeddings=embeddings,
            inner_token_threshold=inner_token_threshold,
            stopwords=stopwords,
        )
        pair = classify_pair(
            a,
            b,
            sims=sims,
            same_cluster=same,
            thresholds=thresholds,
            labels=labels,
        )
        if pair is not None:
            out.append(pair)
    return out


# ---------------------------------------------------------------------------
# Ratio + regeneration
# ---------------------------------------------------------------------------


def measure_corner_case_ratio(
    pairs: Sequence[tuple[int, int]],
    corner_cases: Sequence[CornerCasePair],
) -> float:
    """Return ``len(corner_cases) / len(pairs)`` (or 0 if empty)."""
    if not pairs:
        return 0.0
    return len(corner_cases) / len(pairs)


def regenerate_em_test_set(
    *,
    positive_record_pairs: Sequence[tuple[str, str]],
    negative_record_pairs: Sequence[tuple[str, str]],
    corner_case_negatives: set[tuple[str, str]],
    target_ratio: float,
    target_size: int,
    rng: np.random.Generator,
    corner_case_positives: set[tuple[str, str]] | None = None,
) -> list[tuple[str, str, bool]]:
    """Sample an EM test set stratified by corner-case ratio.

    Pairs are **source-record IDs** (matching the original EM gold
    convention), not canonical-entity indices. Positives are drawn from
    within-cluster source-record pairs (different sources, same
    canonical entity); negatives are drawn from cross-cluster
    source-record pairs (one representative record from each of two
    distinct canonical entities). The caller is responsible for the
    canonical→record mapping.

    Returns a list of ``(rid_a, rid_b, is_match)`` triples of length at
    most ``target_size``. The corner-case budget is split evenly across
    positives (``corner_case_positives``) and negatives
    (``corner_case_negatives``); each half is capped by pool size so the
    overall ratio may undershoot the target when one pool runs dry.

    Parameters
    ----------
    positive_record_pairs : sequence of (str, str)
        Candidate positive source-record pairs (same canonical entity).
    negative_record_pairs : sequence of (str, str)
        Candidate negative source-record pairs (different canonical
        entities).
    corner_case_negatives : set of (str, str)
        Subset of ``negative_record_pairs`` flagged as hard by the
        corner-case miner.
    target_ratio : float
        Desired fraction of corner cases in the output.
    target_size : int
        Desired output size.
    rng : numpy.random.Generator
        Seeded RNG for sampling.
    corner_case_positives : set of (str, str) or None
        Subset of ``positive_record_pairs`` flagged as hard. Defaults to
        the empty set — the canonical-level miner cannot mine
        record-level positive corner cases, so positives are treated as
        non-corner unless the caller passes explicit hard positives.

    Returns
    -------
    list of (str, str, bool)
        ``(rid_a, rid_b, is_match)`` triples.
    """
    if target_ratio < 0.0 or target_ratio > 1.0:
        raise ValueError(f"target_ratio must be in [0, 1], got {target_ratio}")

    corner_pos = corner_case_positives or set()

    def _split(
        pool: Sequence[tuple[str, str]],
        corner_set: set[tuple[str, str]],
    ) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        hard: list[tuple[str, str]] = []
        easy: list[tuple[str, str]] = []
        for a, b in pool:
            key = (a, b) if a < b else (b, a)
            if key in corner_set:
                hard.append((a, b))
            else:
                easy.append((a, b))
        return hard, easy

    same_hard, same_easy = _split(positive_record_pairs, corner_pos)
    cross_hard, cross_easy = _split(negative_record_pairs, corner_case_negatives)

    n_corner = int(round(target_size * target_ratio))

    # Split evenly across match / non-match for both corner and easy.
    n_match_total = target_size // 2
    n_nonmatch_total = target_size - n_match_total
    n_match_corner = min(n_corner // 2, len(same_hard))
    n_nonmatch_corner = min(n_corner - n_match_corner, len(cross_hard))
    n_match_easy = min(n_match_total - n_match_corner, len(same_easy))
    n_nonmatch_easy = min(n_nonmatch_total - n_nonmatch_corner, len(cross_easy))

    def _sample(pool: list[tuple[str, str]], k: int) -> list[tuple[str, str]]:
        if k <= 0 or not pool:
            return []
        if k >= len(pool):
            chosen = list(pool)
        else:
            idxs = rng.choice(len(pool), size=k, replace=False)
            chosen = [pool[int(x)] for x in idxs]
        return chosen

    out: list[tuple[str, str, bool]] = []
    for a, b in _sample(same_hard, n_match_corner):
        out.append((a, b, True))
    for a, b in _sample(same_easy, n_match_easy):
        out.append((a, b, True))
    for a, b in _sample(cross_hard, n_nonmatch_corner):
        out.append((a, b, False))
    for a, b in _sample(cross_easy, n_nonmatch_easy):
        out.append((a, b, False))

    # Deterministic order independent of sampling history.
    out.sort(key=lambda t: (int(t[2]), t[0], t[1]))
    return out


@dataclass(frozen=True)
class SplitSpec:
    """Target size and positive ratio for one regenerated EM split.

    Attributes
    ----------
    name : str
        Split name (``"train"``, ``"val"``, ``"test"``).
    size : int
        Total number of rows to emit.
    positive_ratio : float
        Fraction of rows that should carry ``label == "true"``. The
        builder respects the ratio exactly unless the per-class pool
        runs dry, in which case the split undershoots in size rather
        than violating the ratio.
    """

    name: str
    size: int
    positive_ratio: float


VERSION_BASELINE_PRUNED = "baseline_pruned"
VERSION_CORNER_FILLED = "corner_filled"
REGEN_VERSIONS: tuple[str, ...] = (VERSION_BASELINE_PRUNED, VERSION_CORNER_FILLED)


def regenerate_em_splits(
    *,
    original_pairs_by_split: dict[
        tuple[str, str], dict[str, list[tuple[str, str, bool]]]
    ],
    ids_present: set[str],
    pool_positives_by_pair: dict[tuple[str, str], list[RecordPair]],
    interpolated_positives_by_pair: dict[tuple[str, str], list[RecordPair]],
    cluster_positives_by_pair: dict[tuple[str, str], list[RecordPair]],
    negatives_by_pair: dict[tuple[str, str], list[RecordPair]],
    corner_case_negatives_by_pair: dict[tuple[str, str], set[RecordPair]],
    split_specs_by_pair: dict[tuple[str, str], list[SplitSpec]],
    target_ratio: float,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    """Edit original EM splits into augmented-variant splits, emitting
    two parallel versions per (pair, split) — ``baseline_pruned`` and
    ``corner_filled`` (plan_revision.md C11, 2026-05-22).

    Rather than rebuild splits from scratch, this regenerator treats
    the *original* train / val / test files as the starting point. For
    each split we:

    1. **Carry over surviving originals.** For every original
       ``(id1, id2, label)`` whose ids both still exist in
       ``ids_present`` (i.e. neither was K2-removed), the row is kept
       verbatim. K2 is the only knob that can invalidate an original
       pair: it removes entities outright; no other knob renames,
       merges, or splits clusters, so labels never flip for surviving
       ids.
    2. **Backfill (corner_filled only).** The remaining
       (positive / negative) budget vs the original split's
       ``(size, positive_ratio)`` is filled exclusively from the
       corner-mined pools:

       - **Corner-case positives**: K2-interpolated cross-source pairs
         (``interpolated_positives_by_pair``) — the deliberately-niche
         entities K2 hard injects.
       - **Corner-case negatives**: ``corner_case_negatives_by_pair``
         — the hard-negative-gated cross-cluster pairs.

       Per C11 option (i), the backfill is 100% corner-mined; the
       previous easy-positive / easy-negative spillover is intentionally
       **removed**. When the corner pool is exhausted the realised size
       is allowed to fall below ``spec.size`` (logged as a warning) —
       this is the design contract.

    The two emitted versions are:

    * **baseline_pruned** (Set 1) — survivors only, no backfill. A
      level-dependent, corner-biased subsample of the original gold;
      its row count shrinks with K2 intensity.
    * **corner_filled** (Set 2) — survivors plus the corner-mined
      backfill above. By construction Set 1 ⊂ Set 2 (every survivor
      appears in both versions; only Set 2 carries the corner backfill
      rows on top).

    Each emitted row carries a ``version`` column so the caller can
    group by (pair, split, version) and write one CSV per group.

    Disjointness across the three splits is enforced via the same
    consumed-pair tracking as before (canonical-pair keys, gold-canon
    filtered out of backfill pools up front).

    Pairs are **source-record IDs**, same convention as the original
    EM gold files. All input pair lists are expected to be already
    oriented so the first id belongs to ``src1`` and the second to
    ``src2``.

    Parameters
    ----------
    original_pairs_by_split : dict
        ``{(src1, src2): {split_name: [(id1, id2, is_positive), ...]}}``
        — the original (pre-augmentation) splits, used as the starting
        point for both versions.
    ids_present : set of str
        Record ids that still exist in the augmented source frames
        (post-K2 removal / interpolation projection). An original pair
        is kept iff both of its ids are in this set.
    pool_positives_by_pair : dict
        ``(src1, src2) -> [pool record pairs]``. Retained in the
        signature for API stability + audit-trail bookkeeping (counted
        when classifying surviving originals as corner vs not); no
        longer used as a backfill source under C11.
    interpolated_positives_by_pair : dict
        ``(src1, src2) -> [K2-interpolated cross-source pairs]``.
        Corner-positive backfill pool (used only by ``corner_filled``).
    cluster_positives_by_pair : dict
        ``(src1, src2) -> [cross-source pairs from surviving K2
        clusters, excluding interpolated]``. Retained for the same
        audit-trail / API stability reason as ``pool_positives_by_pair``;
        no longer used as a backfill source under C11.
    negatives_by_pair : dict
        ``(src1, src2) -> [cross-cluster record pairs]`` (post
        hard-negative-gate). Source of the corner-negative pool
        (filtered through ``corner_case_negatives_by_pair``). Easy
        negatives are intentionally not consumed.
    corner_case_negatives_by_pair : dict
        Subset of ``negatives_by_pair`` flagged as corner-case by the
        K2 miner. Corner-negative backfill pool (used only by
        ``corner_filled``).
    split_specs_by_pair : dict
        ``(src1, src2) -> [SplitSpec]``. Target size + positive ratio
        per split; read from the original split files. ``corner_filled``
        aims for these targets; ``baseline_pruned`` is whatever survives.
    target_ratio : float
        Retained in the signature for backwards compatibility — under
        C11 the corner backfill consumes the *entire* remaining
        positive / negative budget (not just ``round(size *
        target_ratio)``), so this parameter is now only validated for
        range. Pass K2's per-(domain, level)
        ``target_corner_case_ratio`` if you want the value to flow
        through audit logs.
    rng : numpy.random.Generator
        Seeded RNG; deterministic sub-streams are spawned per pair per
        split for run-to-run reproducibility.

    Returns
    -------
    list of dict
        One dict per emitted row with keys ``id1``, ``id2``,
        ``source_1``, ``source_2``, ``label`` (``"true"`` /
        ``"false"``), ``split``, ``pair_name`` (formatted
        ``src1_2_src2``), and ``version`` (one of
        ``baseline_pruned`` / ``corner_filled``). Each surviving pair
        appears in both versions; each corner backfill pair appears
        only in ``corner_filled``.
    """
    if not (0.0 <= target_ratio <= 1.0):
        raise ValueError(f"target_ratio must be in [0, 1], got {target_ratio}")

    def _canon(rid_a: str, rid_b: str) -> tuple[str, str]:
        return (rid_a, rid_b) if rid_a < rid_b else (rid_b, rid_a)

    def _take(
        pool: Sequence[RecordPair],
        k: int,
        consumed: set[RecordPair],
        sub_rng: np.random.Generator,
    ) -> list[RecordPair]:
        if k <= 0:
            return []
        available = [p for p in pool if _canon(*p) not in consumed]
        if not available:
            return []
        if k >= len(available):
            chosen = list(available)
        else:
            idxs = sub_rng.choice(len(available), size=k, replace=False)
            chosen = [available[int(i)] for i in idxs]
        for p in chosen:
            consumed.add(_canon(*p))
        return chosen

    rows: list[dict[str, object]] = []

    for pair in sorted(split_specs_by_pair):
        src1, src2 = pair
        pair_name = f"{src1}_2_{src2}"

        # Build per-pair candidate pools, partitioned into corner / easy.
        # Every backfill candidate is filtered to ``ids_present`` so a
        # stale pool entry (e.g. the pooled_positives CSV referencing a
        # record that K2 just removed) can never sneak back into the
        # regen output. Production K2 already passes filtered lists, but
        # the filter is cheap insurance against callers that don't.
        def _both_present(pair_tuple: RecordPair) -> bool:
            return pair_tuple[0] in ids_present and pair_tuple[1] in ids_present

        # Reserve every canonical gold pair across all splits so a
        # higher-priority split's backfill cannot ``consume`` a pair
        # that a lower-priority split's gold owns. Pre-fix, this caused
        # ~13 hard-level music-small survivors to silently relocate
        # (e.g. train gold pair X carried over in regen as val instead
        # of train). The survival pass still iterates per-split, but
        # backfill pools are filtered up-front to skip every gold canon.
        # See plan_s1_final.md F10.
        all_gold_canon: set[RecordPair] = set()
        for split_pairs in original_pairs_by_split.get(pair, {}).values():
            for id1, id2, _ in split_pairs:
                all_gold_canon.add(_canon(id1, id2))

        def _eligible_backfill(pair_tuple: RecordPair) -> bool:
            return (
                _both_present(pair_tuple) and _canon(*pair_tuple) not in all_gold_canon
            )

        interp_pool: list[RecordPair] = [
            p
            for p in interpolated_positives_by_pair.get(pair, [])
            if _eligible_backfill(p)
        ]
        # ``interp_canon`` remains the full interpolated set (incl. any
        # K2 pairs that happen to overlap with gold) so survivor-pass
        # corner-bookkeeping is accurate.
        interp_canon: set[RecordPair] = {
            _canon(a, b)
            for a, b in interpolated_positives_by_pair.get(pair, [])
            if _both_present((a, b))
        }
        # ``cluster_pool`` / ``pool_pos`` were the legacy easy-positive
        # backfill source. Under C11 they are no longer consumed for
        # backfill, but we still touch them via ``_eligible_backfill``
        # so the audit-trail signature is consistent and so callers that
        # forgot to filter out invalid ids surface the same warning.
        for _p in cluster_positives_by_pair.get(pair, []):
            _eligible_backfill(_p)
        for _p in pool_positives_by_pair.get(pair, []):
            _eligible_backfill(_p)

        neg_corner_set: set[RecordPair] = {
            _canon(a, b)
            for a, b in corner_case_negatives_by_pair.get(pair, set())
            if _both_present((a, b))
        }
        # Build the corner-negative pool from ``negatives_by_pair``
        # filtered through ``corner_case_negatives_by_pair``. Easy
        # negatives are intentionally not consumed under C11.
        corner_neg_pool: list[RecordPair] = [
            (a, b)
            for a, b in negatives_by_pair.get(pair, [])
            if _eligible_backfill((a, b)) and _canon(a, b) in neg_corner_set
        ]

        # Consumed record pairs for this source pair — enforces
        # disjointness across splits AND across the corner_filled
        # backfill (a single canonical pair only ever appears once
        # under one (split, version)). Canonicalised so (a,b) and (b,a)
        # count as the same pair.
        consumed: set[RecordPair] = set()
        pair_rng = np.random.default_rng(
            int(rng.integers(0, 2**31 - 1)) ^ (hash(pair_name) & 0xFFFFFFFF)
        )

        # Drain the pool in priority order test → val → train so test
        # gets first dibs on the cleanest backfill; val drives
        # monotonicity; train absorbs the remainder.
        _PRIORITY = {"test": 0, "val": 1, "train": 2}
        ordered_specs = sorted(
            split_specs_by_pair[pair],
            key=lambda s: (_PRIORITY.get(s.name, 99), s.name),
        )

        orig_for_pair = original_pairs_by_split.get(pair, {})

        for spec in ordered_specs:
            split_rng = np.random.default_rng(
                int(pair_rng.integers(0, 2**31 - 1)) ^ (hash(spec.name) & 0xFFFFFFFF)
            )

            target_pos = int(round(spec.size * spec.positive_ratio))
            target_neg = spec.size - target_pos

            # ---- Survival pass: carry over original pairs still valid.
            # Both versions share these survivors — emission below
            # writes them under both ``baseline_pruned`` and
            # ``corner_filled``.
            kept_pos: list[RecordPair] = []
            kept_neg: list[RecordPair] = []
            kept_pos_corner_count = 0
            kept_neg_corner_count = 0
            for id1, id2, is_pos in orig_for_pair.get(spec.name, []):
                if id1 not in ids_present or id2 not in ids_present:
                    continue
                key = _canon(id1, id2)
                if key in consumed:
                    # Same pair seen earlier — keep determinism + dedup.
                    continue
                consumed.add(key)
                if is_pos:
                    kept_pos.append((id1, id2))
                    if key in interp_canon:
                        kept_pos_corner_count += 1
                else:
                    kept_neg.append((id1, id2))
                    if key in neg_corner_set:
                        kept_neg_corner_count += 1

            # ---- Corner backfill (corner_filled only) -----------------
            # C11 contract: fill the *entire* remaining positive /
            # negative budget from the corner-mined pools, with no easy
            # spillover. If the corner pool runs dry the realised size
            # is allowed to fall below ``spec.size`` (logged as a
            # warning). Mirrors the corresponding K10 "rate vs count"
            # decision from C3 — we accept the shortfall instead of
            # quietly injecting non-corner pairs to hit the target.
            pos_slots_remaining = max(0, target_pos - len(kept_pos))
            picked_pos_corner = _take(
                interp_pool, pos_slots_remaining, consumed, split_rng
            )

            neg_slots_remaining = max(0, target_neg - len(kept_neg))
            picked_neg_corner = _take(
                corner_neg_pool, neg_slots_remaining, consumed, split_rng
            )

            # ---- Emit baseline_pruned (Set 1 — survivors only) -------
            # Deterministic order: positives first (lex sorted), then
            # negatives (lex sorted).
            for rid_a, rid_b in sorted(kept_pos):
                rows.append(
                    {
                        "id1": rid_a,
                        "id2": rid_b,
                        "source_1": src1,
                        "source_2": src2,
                        "label": "true",
                        "split": spec.name,
                        "pair_name": pair_name,
                        "version": VERSION_BASELINE_PRUNED,
                    }
                )
            for rid_a, rid_b in sorted(kept_neg):
                rows.append(
                    {
                        "id1": rid_a,
                        "id2": rid_b,
                        "source_1": src1,
                        "source_2": src2,
                        "label": "false",
                        "split": spec.name,
                        "pair_name": pair_name,
                        "version": VERSION_BASELINE_PRUNED,
                    }
                )

            # ---- Emit corner_filled (Set 2 — survivors + corners) ----
            for rid_a, rid_b in sorted(kept_pos + list(picked_pos_corner)):
                rows.append(
                    {
                        "id1": rid_a,
                        "id2": rid_b,
                        "source_1": src1,
                        "source_2": src2,
                        "label": "true",
                        "split": spec.name,
                        "pair_name": pair_name,
                        "version": VERSION_CORNER_FILLED,
                    }
                )
            for rid_a, rid_b in sorted(kept_neg + list(picked_neg_corner)):
                rows.append(
                    {
                        "id1": rid_a,
                        "id2": rid_b,
                        "source_1": src1,
                        "source_2": src2,
                        "label": "false",
                        "split": spec.name,
                        "pair_name": pair_name,
                        "version": VERSION_CORNER_FILLED,
                    }
                )

            # ---- Shortfall + drift audit (corner_filled) -------------
            realised_pos_cf = len(kept_pos) + len(picked_pos_corner)
            realised_neg_cf = len(kept_neg) + len(picked_neg_corner)
            realised_size_cf = realised_pos_cf + realised_neg_cf
            if realised_size_cf < spec.size:
                logger.warning(
                    "%s/%s corner_filled undersized: target=%d realised=%d "
                    "(pos %d/%d, neg %d/%d) — corner pool exhausted "
                    "(C11 accepts the shortfall instead of injecting "
                    "non-corner pairs)",
                    pair_name,
                    spec.name,
                    spec.size,
                    realised_size_cf,
                    realised_pos_cf,
                    target_pos,
                    realised_neg_cf,
                    target_neg,
                )
            else:
                # Size on-target but per-class drift still possible: when
                # one class's corner pool drains while the other is
                # exactly on target, the realised pos_ratio drifts even
                # though the total size matches. Surface for the S.5a
                # contract check.
                realised_ratio = realised_pos_cf / max(realised_size_cf, 1)
                drift_pp = abs(realised_ratio - spec.positive_ratio) * 100.0
                if drift_pp > 2.0:
                    logger.warning(
                        "%s/%s corner_filled pos_ratio drift: "
                        "realised=%.4f target=%.4f (|delta|=%.2fpp > 2pp) "
                        "— per-class corner pool ran dry with size on-target",
                        pair_name,
                        spec.name,
                        realised_ratio,
                        spec.positive_ratio,
                        drift_pp,
                    )

    return rows


@dataclass(frozen=True)
class RegenPools:
    """Pool snapshot needed to (re)run :func:`regenerate_em_splits`.

    Captured at K2 time so the orchestrator can call regen again after
    K4 has demoted records, with a refreshed ``ids_present`` over the
    post-K4 source frames. This avoids the hard-level orphan-ID issue
    where K2-emitted regen references IDs that K4 then removes.
    """

    original_pairs_by_split: dict[
        tuple[str, str], dict[str, list[tuple[str, str, bool]]]
    ]
    pool_positives_by_pair: dict[tuple[str, str], list[RecordPair]]
    interpolated_positives_by_pair: dict[tuple[str, str], list[RecordPair]]
    cluster_positives_by_pair: dict[tuple[str, str], list[RecordPair]]
    negatives_by_pair: dict[tuple[str, str], list[RecordPair]]
    corner_case_negatives_by_pair: dict[tuple[str, str], set[RecordPair]]
    split_specs_by_pair: dict[tuple[str, str], list[SplitSpec]]
    target_ratio: float


def regenerate_em_splits_from_pools(
    pools: RegenPools,
    ids_present: set[str],
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    """Re-run :func:`regenerate_em_splits` using a stashed K2 pool snapshot.

    Use this from the orchestrator after K4 runs to refresh the
    ``ids_present`` filter against the post-K4 source frames so the
    regen never references IDs that K4 demoted away. The pools
    themselves are level-invariant (they are derived from K2's
    cluster + interpolation results, which K4 does not mutate).
    """
    return regenerate_em_splits(
        original_pairs_by_split=pools.original_pairs_by_split,
        ids_present=ids_present,
        pool_positives_by_pair=pools.pool_positives_by_pair,
        interpolated_positives_by_pair=pools.interpolated_positives_by_pair,
        cluster_positives_by_pair=pools.cluster_positives_by_pair,
        negatives_by_pair=pools.negatives_by_pair,
        corner_case_negatives_by_pair=pools.corner_case_negatives_by_pair,
        split_specs_by_pair=pools.split_specs_by_pair,
        target_ratio=pools.target_ratio,
        rng=rng,
    )
