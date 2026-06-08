"""Batch-mode truth-discovery fusion factories.

Replaces the per-cell adapters that previously lived at
``usecases_synthetic/lib/{truthfinder,ltm,casefusion,fusionquery,accusim}_fusion.py``.

Why batch?
----------
The upstream methods (TruthFinder / LTM / CASE / FusionQuery's EMFusioner) are
designed to **learn one source-trust vector across many (entity, source, value)
claims** and then re-score every entity's candidate values with that vector.
PyDI's per-cell ``ConflictResolutionFunction`` API only ever gives the resolver
one record group's claims at a time — at companies-small scale that is K=3
sources × 1-3 distinct values per call, which collapses every TD method to a
similarity-aware vote and throws away the entire reason for choosing a TD
method over plain voting.

This module fits each upstream method **once per (attribute, strategy) pair**
on the full attribute corpus, looks up each entity's winning value by group id,
and exposes the result through a per-cell closure that the existing
``DataFusionEngine`` can consume without modification. AccuSim is paper-
reimplemented as a batch routine in the same file (no upstream code).

Factory protocol
----------------
Every factory has the signature::

    factory(
        datasets: list[pd.DataFrame],
        correspondences: pd.DataFrame,
        target_attr: str,
        *,
        id_column: str | dict | None = None,
        ...method-specific params,
    ) -> ConflictResolutionFunction

The committee runner detects strategy specs with ``factory: true`` and calls
the factory once per (attribute, strategy) before constructing the
``DataFusionStrategy``. The returned callable still respects the standard
``(values, **kwargs) -> (value, confidence, metadata)`` contract; it consults
``kwargs['group_id']`` to look up the precomputed batch winner, and falls back
to "first valid value" for groups that were not seen during the batch fit
(typically singleton groups that the engine excluded).
"""

from __future__ import annotations

import logging
import random as _stdlib_random
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from PyDI.fusion.engine import build_record_groups_from_correspondences

from ..third_party.fusionquery.baseline import (
    CASEFusion as _CASEFusion,
    LTMFusion as _LTMFusion,
    TruthFinder as _TruthFinder,
)
from ..third_party.fusionquery.fusion import EMFusioner as _EMFusioner

logger = logging.getLogger(__name__)

ConflictResolutionFunction = Callable[..., Tuple[Any, float, Dict[str, Any]]]


# ---------------------------------------------------------------------------
# Validity + serialisation helpers (inlined; mirrors PyDI's private surface)
# ---------------------------------------------------------------------------


def _is_valid(value: Any) -> bool:
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


def _stringify(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return repr(list(value))
    return str(value)


# ---------------------------------------------------------------------------
# Corpus ingest: PyDI datasets + correspondences -> per-source claim lists
# ---------------------------------------------------------------------------


def _collect_attribute_claims(
    datasets: Sequence[pd.DataFrame],
    correspondences: pd.DataFrame,
    target_attr: str,
    *,
    id_column: Any,
) -> Tuple[
    Dict[str, List[Tuple[str, str, Any]]],
    Dict[str, Dict[str, Any]],
]:
    """Walk every record group and collect per-source claims for ``target_attr``.

    Returns
    -------
    claims_by_source
        ``{source_dataset_name: [(group_id, ans_string, original_value), ...]}``
        — one entry per (source, group) cell whose target-attribute value is
        valid (non-None, non-NaN, non-empty list).
    group_to_per_source
        ``{group_id: {source_name: original_value}}`` — kept so the closure can
        offer per-source diagnostics if needed.
    """
    groups = build_record_groups_from_correspondences(
        datasets=list(datasets),
        correspondences=correspondences,
        id_column=id_column,
    )

    # Skip singletons — a record with no peers in the correspondence graph
    # has only one source claim by construction and contributes nothing to
    # truth-discovery (no inter-source conflict to resolve). Including them
    # blows up the iteration count: on full games (93k records, 528
    # correspondences) accusim's max_iter=20 loop walks 1.8M groups with
    # only ~500 of them informative.
    groups = [g for g in groups if len(g.records) > 1]

    claims_by_source: Dict[str, List[Tuple[str, str, Any]]] = {}
    group_to_per_source: Dict[str, Dict[str, Any]] = {}

    for group in groups:
        gid = group.group_id
        per_source: Dict[str, Any] = {}
        for record in group.records:
            rid = record.get("_id")
            ds = group.source_datasets.get(rid, "unknown")
            value = record.get(target_attr)
            if not _is_valid(value):
                continue
            per_source[ds] = value
            ans_str = _stringify(value)
            claims_by_source.setdefault(ds, []).append((gid, ans_str, value))
        if per_source:
            group_to_per_source[gid] = per_source

    return claims_by_source, group_to_per_source


def _build_cand_answer(
    claims_by_source: Dict[str, List[Tuple[str, str, Any]]],
) -> Tuple[
    Dict[int, List[Tuple[str, float]]],
    List[str],
    List[Tuple[str, str, Any]],
]:
    """Pack per-source claims into the upstream's ``cand_answer`` shape.

    The order of ``flat_claims`` matches the order in which the upstream's
    ``prepare_for_fusion`` will enumerate ``cand_answer`` (outer loop over
    sources, inner loop over the source's pair list), so its index lines up
    with ``self.ans_set``.
    """
    cand_answer: Dict[int, List[Tuple[str, float]]] = {}
    source_names: List[str] = []
    flat_claims: List[Tuple[str, str, Any]] = []

    for src_idx, src_name in enumerate(sorted(claims_by_source.keys())):
        claims = claims_by_source[src_name]
        cand_answer[src_idx] = [(c[1], 1.0) for c in claims]
        source_names.append(src_name)
        flat_claims.extend(claims)

    return cand_answer, source_names, flat_claims


def _pick_winners_per_group(
    veracity: np.ndarray,
    flat_claims: List[Tuple[str, str, Any]],
) -> Dict[str, Tuple[Any, float]]:
    """Group the per-claim veracity by group_id and pick the highest-scoring
    distinct answer per group.

    Duplicate answer strings within the same group sum their veracity (so two
    sources independently claiming "Apple Inc." reinforce each other). Ties
    break stably on first-occurrence order.
    """
    if veracity is None or len(veracity) == 0:
        return {}
    arr = np.asarray(veracity, dtype=float)

    per_group: Dict[str, Dict[str, Tuple[float, Any, int]]] = {}
    # ans_str -> (cumulative_score, original_value, first_seen_index)
    for idx, ((gid, ans_str, original), score) in enumerate(
        zip(flat_claims, arr.tolist())
    ):
        if not np.isfinite(score):
            continue
        bucket = per_group.setdefault(gid, {})
        if ans_str in bucket:
            prev_score, prev_val, prev_idx = bucket[ans_str]
            bucket[ans_str] = (prev_score + float(score), prev_val, prev_idx)
        else:
            bucket[ans_str] = (float(score), original, idx)

    winners: Dict[str, Tuple[Any, float]] = {}
    for gid, bucket in per_group.items():
        # Argmax on cumulative score, tie-break on smaller first-seen index.
        best_str, (best_score, best_val, _) = max(
            bucket.items(), key=lambda kv: (kv[1][0], -kv[1][2])
        )
        total = sum(s for s, _, _ in bucket.values())
        confidence = float(np.clip(best_score / total, 0.0, 1.0)) if total > 0 else 0.0
        winners[gid] = (best_val, confidence)
    return winners


# ---------------------------------------------------------------------------
# Closure factory
# ---------------------------------------------------------------------------


def _make_lookup_resolver(
    rule_name: str,
    winners: Dict[str, Tuple[Any, float]],
    extras: Optional[Dict[str, Any]] = None,
) -> ConflictResolutionFunction:
    """Wrap a precomputed ``{group_id: (value, confidence)}`` dict in a
    PyDI-compatible per-cell ``ConflictResolutionFunction``.

    For groups not present in ``winners`` (typically singletons that the engine
    excludes from group construction, or rare rerun shape mismatches) the
    closure falls back to "first valid value" — keeps the engine's per-cell
    loop progressing instead of crashing.
    """
    extras = dict(extras or {})

    def resolver(values: List[Any], **kwargs: Any) -> Tuple[Any, float, Dict[str, Any]]:
        gid = kwargs.get("group_id")
        if gid is not None and gid in winners:
            value, confidence = winners[gid]
            return (
                value,
                confidence,
                {
                    "rule": rule_name,
                    "source": "batch_lookup",
                    "group_id": gid,
                    **extras,
                },
            )
        for v in values:
            if _is_valid(v):
                return (
                    v,
                    0.5,
                    {
                        "rule": rule_name,
                        "source": "fallback_first_valid",
                        "group_id": gid,
                    },
                )
        return None, 0.0, {"rule": rule_name, "reason": "no_valid_values"}

    return resolver


# ---------------------------------------------------------------------------
# AccuSim batch fit (paper reimplementation, no upstream code)
# ---------------------------------------------------------------------------


def _dice_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 1.0 if a == b else 0.0
    if a == b:
        return 1.0
    cnt_a = Counter(a)
    cnt_b = Counter(b)
    inter = sum((cnt_a & cnt_b).values())
    return 2.0 * inter / (len(a) + len(b))


def _accusim_default_sim(v1: Any, v2: Any) -> float:
    if v1 == v2:
        return 1.0
    try:
        a = float(v1)
        b = float(v2)
        denom = max(abs(a), abs(b), 1.0)
        return float(max(0.0, 1.0 - abs(a - b) / denom))
    except (TypeError, ValueError):
        pass
    if isinstance(v1, (list, tuple)) and isinstance(v2, (list, tuple)):
        ca = Counter(v1)
        cb = Counter(v2)
        inter = sum((ca & cb).values())
        union = sum((ca | cb).values())
        return float(inter / union) if union > 0 else 0.0
    return _dice_similarity(str(v1), str(v2))


def _accusim_batch_fit(
    claims_by_source: Dict[str, List[Tuple[str, str, Any]]],
    *,
    accuracy_prior: float,
    n_competing_values: int,
    epsilon: float,
    max_iter: int,
    sim_threshold: float,
    similarity: Callable[[Any, Any], float],
) -> Tuple[Dict[str, Tuple[Any, float]], Dict[str, float], int]:
    """Run AccuSim's iterative source-accuracy + per-entity log-odds update
    across the entire attribute corpus, returning per-group winners + final
    per-source accuracies.
    """
    # Index sources alphabetically (stable across runs).
    source_names = sorted(claims_by_source.keys())
    src_idx = {s: i for i, s in enumerate(source_names)}
    K = len(source_names)

    # Per-group bucket of (source_idx, original_value).
    per_group: Dict[str, List[Tuple[int, Any]]] = {}
    for src_name, claims in claims_by_source.items():
        sidx = src_idx[src_name]
        for gid, _ans_str, original in claims:
            per_group.setdefault(gid, []).append((sidx, original))

    A = np.full(K, float(accuracy_prior), dtype=float)
    eps = 1e-9
    lo = 1.0 / n_competing_values + eps
    hi = 1.0 - eps

    n_iter = 0
    final_winners: Dict[str, Tuple[Any, float]] = {}

    for step in range(max_iter):
        n_iter = step + 1
        per_source_logit = np.log(
            n_competing_values * np.clip(A, lo, hi) / (1.0 - np.clip(A, lo, hi))
        )

        # For source-accuracy update: source gets credit if its claim was
        # sim_threshold-similar to the chosen truth on the entity.
        per_source_correct = np.zeros(K, dtype=float)
        per_source_total = np.zeros(K, dtype=float)
        winners_this_iter: Dict[str, Tuple[Any, float]] = {}

        for gid, entries in per_group.items():
            distinct: Dict[str, Tuple[Any, float, List[int]]] = {}
            for sidx, original in entries:
                key = _stringify(original)
                if key in distinct:
                    val, _, src_list = distinct[key]
                    src_list.append(sidx)
                else:
                    distinct[key] = (original, 0.0, [sidx])

            keys = list(distinct.keys())
            M = len(keys)
            if M == 0:
                continue

            # M x M similarity matrix.
            sim_mat = np.eye(M, dtype=float)
            for a in range(M):
                for b in range(a + 1, M):
                    s = float(similarity(distinct[keys[a]][0], distinct[keys[b]][0]))
                    s = max(0.0, min(1.0, s))
                    sim_mat[a, b] = sim_mat[b, a] = s

            # Per-value log-odds = sum_w sim[v, w] * sum_{src in distinct[w]} per_source_logit
            value_logits = np.array(
                [sum(per_source_logit[s] for s in distinct[k][2]) for k in keys],
                dtype=float,
            )
            log_odds = sim_mat @ value_logits
            truth_idx = int(np.argmax(log_odds))
            truth_key = keys[truth_idx]
            truth_value = distinct[truth_key][0]

            # Confidence = softmax over log_odds.
            shifted = log_odds - log_odds.max()
            weights = np.exp(shifted)
            confidence = float(weights[truth_idx] / weights.sum())
            winners_this_iter[gid] = (truth_value, float(np.clip(confidence, 0.0, 1.0)))

            truth_sim_row = sim_mat[truth_idx]
            for v_idx, key in enumerate(keys):
                contributing_sources = distinct[key][2]
                credit = 1.0 if truth_sim_row[v_idx] >= sim_threshold else 0.0
                for sidx in contributing_sources:
                    per_source_correct[sidx] += credit
                    per_source_total[sidx] += 1.0

        new_A = np.where(
            per_source_total > 0, per_source_correct / per_source_total, accuracy_prior
        )
        # Smooth toward prior so binary accuracies do not collapse logits.
        new_A = 0.5 * new_A + 0.5 * accuracy_prior
        new_A = np.clip(new_A, lo, hi)

        delta = float(np.max(np.abs(new_A - A)))
        A = new_A
        final_winners = winners_this_iter
        if delta < epsilon:
            break

    src_accuracy = {source_names[i]: float(A[i]) for i in range(K)}
    return final_winners, src_accuracy, n_iter


# ---------------------------------------------------------------------------
# Per-method factories
# ---------------------------------------------------------------------------


def _trivial_short_circuit(
    flat_claims: List[Tuple[str, str, Any]], rule: str
) -> ConflictResolutionFunction:
    winners = {gid: (val, 1.0) for (gid, _ans, val) in flat_claims}
    return _make_lookup_resolver(
        rule, winners, extras={"note": "single_source_short_circuit"}
    )


# All four global-batch truth-discovery makers (fusionquery/truthfinder/ltm/
# casefusion) pool every claim for an attribute into one ``ans_set`` of length V
# and then iterate an EM/Gibbs fit over it. Neither scales to a high-cardinality
# corpus:
#   * FusionQuery's EMFusioner builds a dense (source_num, V, V) float64 array
#     (fusion.py:74/110) -> papers 'title' V~159k => 3*159k^2*8B ~= 564 GiB,
#     which OOM-kills the job (on large-RAM nodes the cgroup SIGKILLs mid-
#     allocation, uncatchable -- so we must refuse BEFORE allocating).
#   * TruthFinder/LTM/CASEFusion are only O(source*V) in memory but peg a single
#     core for many hours at that V (observed: a papers run stuck 10.7h, ~100%
#     CPU, in the per-attribute-optimal sweep).
# So above a scale cutoff we refuse the fit; the C12 runner's per-member
# try/except then skips the candidate and the per-attribute-optimal sweep falls
# back to the conventional resolvers (voting/longest_string/prefer_higher_trust/
# median/...) + the bounded accusim. Calibration: the largest V across the
# committed 4-domain runs is ~23k (FusionQuery matrix ~13 GiB, fits, fast); the
# cutoff at 46k sits well above that and well below papers' ~159k, so this never
# changes the committed domains -- only papers.
_TD_MAX_CLAIMS = 46_000


def _guard_td_scale(rule: str, n_claims: int) -> None:
    """Refuse a global-batch truth-discovery fit that does not scale to
    ``n_claims`` pooled claims. Raises ``ValueError`` so the C12 runner's
    per-member ``try/except`` skips this candidate (falling back to the
    conventional resolvers) instead of OOM-killing or hanging the pipeline."""
    if n_claims > _TD_MAX_CLAIMS:
        raise ValueError(
            f"{rule}: refusing batch truth-discovery on {n_claims} pooled "
            f"claims (> {_TD_MAX_CLAIMS} cutoff) -- does not scale (FusionQuery "
            f"OOMs at ~{3 * n_claims * n_claims * 8 / 1024**3:.0f} GiB; "
            f"TruthFinder/LTM/CASEFusion peg one core for hours); skipping this "
            f"member so the sweep falls back to conventional resolvers."
        )


def make_truthfinder_resolver(
    datasets: Sequence[pd.DataFrame],
    correspondences: pd.DataFrame,
    target_attr: str,
    *,
    id_column: Any = None,
    init_trust: float = 0.9,
    gamma: float = 0.3,
    rho: float = 0.5,
    max_iters: int = 10,
    early_stop: float = 1e-3,
    seed: int = 42,
    **_: Any,
) -> ConflictResolutionFunction:
    """Batch-fit TruthFinder + return a per-cell lookup closure."""
    claims_by_source, _gps = _collect_attribute_claims(
        datasets, correspondences, target_attr, id_column=id_column
    )
    if not claims_by_source:
        return _make_lookup_resolver("truthfinder_batch", {})

    cand_answer, source_names, flat_claims = _build_cand_answer(claims_by_source)
    if len(cand_answer) < 2:
        return _trivial_short_circuit(flat_claims, "truthfinder_batch")

    _guard_td_scale("truthfinder_batch", len(flat_claims))

    np.random.seed(seed)
    finder = _TruthFinder(
        source_num=len(cand_answer),
        init_trust=init_trust,
        gamma=gamma,
        rho=rho,
        max_iters=max_iters,
        early_stop=early_stop,
    )
    finder.prepare_for_fusion(cand_answer)
    finder.iterate_fusion(threshold=0.5)

    winners = _pick_winners_per_group(finder.veracity, flat_claims)
    src_trust = {
        source_names[i]: float(t) for i, t in enumerate(finder.src_trust.tolist())
    }
    logger.info(
        "TruthFinder batch fit on %r: %d sources, %d claims, %d groups, src_trust=%s",
        target_attr,
        len(cand_answer),
        len(flat_claims),
        len(winners),
        src_trust,
    )
    return _make_lookup_resolver(
        "truthfinder_batch",
        winners,
        extras={"src_trust": src_trust, "n_groups": len(winners)},
    )


def make_ltm_resolver(
    datasets: Sequence[pd.DataFrame],
    correspondences: pd.DataFrame,
    target_attr: str,
    *,
    id_column: Any = None,
    alpha_0: Sequence[float] = (50.0, 10.0),
    alpha_1: Sequence[float] = (10.0, 10.0),
    beta: Sequence[float] = (10.0, 10.0),
    max_iters: int = 50,
    burnin: int = 10,
    thin: int = 2,
    seed: int = 42,
    **_: Any,
) -> ConflictResolutionFunction:
    """Batch-fit LTM + return a per-cell lookup closure."""
    claims_by_source, _gps = _collect_attribute_claims(
        datasets, correspondences, target_attr, id_column=id_column
    )
    if not claims_by_source:
        return _make_lookup_resolver("ltm_batch", {})

    cand_answer, source_names, flat_claims = _build_cand_answer(claims_by_source)
    if len(cand_answer) < 2:
        return _trivial_short_circuit(flat_claims, "ltm_batch")

    _guard_td_scale("ltm_batch", len(flat_claims))

    np.random.seed(seed)
    model = _LTMFusion(
        source_num=len(cand_answer),
        alpha_0=list(alpha_0),
        alpha_1=list(alpha_1),
        beta=list(beta),
        max_iters=max_iters,
        burnin=burnin,
        thin=thin,
    )
    model.prepare_for_fusion(cand_answer)
    model.iterate_fusion(threshold=0.5)

    # LTM's ``ans_set`` collapses duplicate strings into single fact ids
    # — ``self.facts`` maps ans_string -> fact_id. Unlike TruthFinder/CASE,
    # ``self.veracity`` is therefore indexed by *fact*, not by *(source, claim)*.
    # We have to map each entity's claims to facts to find that entity's local
    # winner.
    fact_by_str = dict(model.facts)
    veracity = np.asarray(model.veracity, dtype=float)

    per_group: Dict[str, List[Tuple[str, Any]]] = {}
    for gid, ans_str, original in flat_claims:
        per_group.setdefault(gid, []).append((ans_str, original))

    winners: Dict[str, Tuple[Any, float]] = {}
    for gid, entries in per_group.items():
        scored: Dict[str, Tuple[float, Any]] = {}
        for ans_str, original in entries:
            fact_id = fact_by_str.get(ans_str)
            if fact_id is None:
                continue
            score = float(veracity[fact_id])
            if ans_str in scored:
                prev_score, prev_val = scored[ans_str]
                scored[ans_str] = (max(prev_score, score), prev_val)
            else:
                scored[ans_str] = (score, original)
        if not scored:
            continue
        best_str, (best_score, best_val) = max(
            scored.items(), key=lambda kv: (kv[1][0], kv[0])
        )
        total = sum(s for s, _ in scored.values()) or 1.0
        confidence = float(np.clip(best_score / total, 0.0, 1.0))
        winners[gid] = (best_val, confidence)

    logger.info(
        "LTM batch fit on %r: %d sources, %d claims, %d groups, %d unique facts",
        target_attr,
        len(cand_answer),
        len(flat_claims),
        len(winners),
        len(fact_by_str),
    )
    return _make_lookup_resolver(
        "ltm_batch",
        winners,
        extras={"n_groups": len(winners), "n_facts": len(fact_by_str)},
    )


def make_casefusion_resolver(
    datasets: Sequence[pd.DataFrame],
    correspondences: pd.DataFrame,
    target_attr: str,
    *,
    id_column: Any = None,
    dimension: int = 10,
    alpha: float = 1.1,
    beta: float = 1.1,
    lr: float = 0.05,
    converge_rate: float = 1e-5,
    max_iters: int = 50,
    seed: int = 42,
    **_: Any,
) -> ConflictResolutionFunction:
    """Batch-fit CASEFusion + return a per-cell lookup closure."""
    claims_by_source, _gps = _collect_attribute_claims(
        datasets, correspondences, target_attr, id_column=id_column
    )
    if not claims_by_source:
        return _make_lookup_resolver("casefusion_batch", {})

    cand_answer, source_names, flat_claims = _build_cand_answer(claims_by_source)
    if len(cand_answer) < 2:
        return _trivial_short_circuit(flat_claims, "casefusion_batch")

    _guard_td_scale("casefusion_batch", len(flat_claims))

    np.random.seed(seed)
    _stdlib_random.seed(seed)
    model = _CASEFusion(
        source_num=len(cand_answer),
        dimension=dimension,
        alpha=alpha,
        beta=beta,
        lr=lr,
        converge_rate=converge_rate,
        max_iters=max_iters,
    )
    model.prepare_for_fusion(cand_answer)
    model.iterate_fusion(threshold=0.5)

    winners = _pick_winners_per_group(model.veracity, flat_claims)
    logger.info(
        "CASEFusion batch fit on %r: %d sources, %d claims, %d groups",
        target_attr,
        len(cand_answer),
        len(flat_claims),
        len(winners),
    )
    return _make_lookup_resolver(
        "casefusion_batch", winners, extras={"n_groups": len(winners)}
    )


def make_fusionquery_resolver(
    datasets: Sequence[pd.DataFrame],
    correspondences: pd.DataFrame,
    target_attr: str,
    *,
    id_column: Any = None,
    max_iters: int = 5,
    theta: float = 3e-5,
    init_trust: float = 0.95,
    history_size: float = 50.0,
    temperature: float = 0.5,
    threshold: float = 0.7,
    seed: int = 42,
    **_: Any,
) -> ConflictResolutionFunction:
    """Batch-fit FusionQuery's EMFusioner + return a per-cell lookup closure.

    The class-level ``EMFusioner.his_data_size`` carries source-trust history;
    we reset it per (attribute, strategy) call so two attributes do not
    cross-contaminate, but during the single batch fit it accumulates as
    designed across the attribute's many entities.
    """
    claims_by_source, _gps = _collect_attribute_claims(
        datasets, correspondences, target_attr, id_column=id_column
    )
    if not claims_by_source:
        return _make_lookup_resolver("fusionquery_batch", {})

    cand_answer, source_names, flat_claims = _build_cand_answer(claims_by_source)
    if len(cand_answer) < 2:
        return _trivial_short_circuit(flat_claims, "fusionquery_batch")

    # Refuse the (source_num, V, V) blowup before allocating (see _guard_td_scale).
    _guard_td_scale("fusionquery_batch", len(flat_claims))

    np.random.seed(seed)
    _EMFusioner.his_data_size = None
    model = _EMFusioner(
        source_num=len(cand_answer),
        max_iters=max_iters,
        theta=theta,
        init_trust=init_trust,
        history_size=history_size,
        temperature=temperature,
    )
    model.prepare_for_fusion(cand_answer)
    model.iterate_fusion(threshold=threshold)

    winners = _pick_winners_per_group(model.veracity, flat_claims)
    src_trust = {
        source_names[i]: float(t) for i, t in enumerate(model.src_trust.tolist())
    }
    logger.info(
        "FusionQuery batch fit on %r: %d sources, %d claims, %d groups, src_trust=%s",
        target_attr,
        len(cand_answer),
        len(flat_claims),
        len(winners),
        src_trust,
    )
    return _make_lookup_resolver(
        "fusionquery_batch",
        winners,
        extras={"src_trust": src_trust, "n_groups": len(winners)},
    )


def make_accusim_resolver(
    datasets: Sequence[pd.DataFrame],
    correspondences: pd.DataFrame,
    target_attr: str,
    *,
    id_column: Any = None,
    accuracy_prior: float = 0.8,
    n_competing_values: int = 10,
    epsilon: float = 1e-3,
    max_iter: int = 20,
    sim_threshold: float = 0.7,
    similarity: Optional[Callable[[Any, Any], float]] = None,
    **_: Any,
) -> ConflictResolutionFunction:
    """Batch-fit AccuSim (paper reimplementation) + per-cell lookup closure."""
    if not 0.0 < accuracy_prior < 1.0:
        raise ValueError(f"accuracy_prior must be in (0, 1), got {accuracy_prior!r}")
    if n_competing_values < 2:
        raise ValueError(f"n_competing_values must be >= 2, got {n_competing_values!r}")
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter!r}")
    if not 0.0 <= sim_threshold <= 1.0:
        raise ValueError(f"sim_threshold must be in [0, 1], got {sim_threshold!r}")

    sim_fn = similarity or _accusim_default_sim

    claims_by_source, _gps = _collect_attribute_claims(
        datasets, correspondences, target_attr, id_column=id_column
    )
    if not claims_by_source:
        return _make_lookup_resolver("accusim_batch", {})

    if len(claims_by_source) < 2:
        # Only one source has claims — short-circuit.
        flat_claims = [
            (gid, ans_str, val)
            for src in claims_by_source.values()
            for (gid, ans_str, val) in src
        ]
        return _trivial_short_circuit(flat_claims, "accusim_batch")

    winners, src_accuracy, n_iter = _accusim_batch_fit(
        claims_by_source,
        accuracy_prior=accuracy_prior,
        n_competing_values=n_competing_values,
        epsilon=epsilon,
        max_iter=max_iter,
        sim_threshold=sim_threshold,
        similarity=sim_fn,
    )
    logger.info(
        "AccuSim batch fit on %r: %d sources, %d groups, n_iter=%d, src_accuracy=%s",
        target_attr,
        len(claims_by_source),
        len(winners),
        n_iter,
        src_accuracy,
    )
    return _make_lookup_resolver(
        "accusim_batch",
        winners,
        extras={
            "src_accuracy": src_accuracy,
            "n_iter": n_iter,
            "n_groups": len(winners),
        },
    )


__all__ = [
    "make_truthfinder_resolver",
    "make_ltm_resolver",
    "make_casefusion_resolver",
    "make_fusionquery_resolver",
    "make_accusim_resolver",
]
