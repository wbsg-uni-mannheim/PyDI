"""RRF density scorer for Knob 02 — Entity Niche Density.

Implements ``knobs/knob_02_niche_density.md`` §"Niche-density scorer
(removal path)":

    rrf_score(e, n) = Σ_m  1 / (k₀ + rank_m(n | e))

    density(e) = Σ_{n in top_K_rrf(e)} rrf_score(e, n)
                 · 𝟙[agreement_count(e, n) >= c_min]
                 + boost_label_collision · 𝟙[e in label-collision group]

The scorer is consensus-biased (contrast with the corner-case pair
miner, which is recall-biased). A neighbour supported by only a single
metric contributes zero to density. Label-collision adds a fixed boost
so colliding entities deterministically inherit cluster density even
when the ranking metrics disagree.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


NeighbourList = list[tuple[int, float]]


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------


@dataclass
class MetricNeighbourhoods:
    """Per-metric ranked neighbour lists for every entity.

    Parameters
    ----------
    name : str
        Metric name (e.g. ``"ext_jaccard"``, ``"tfidf"``).
    top_k : list of list of (int, float)
        ``top_k[i]`` is the top-K neighbours for entity ``i`` as a
        descending list of ``(neighbour_index, raw_similarity)`` tuples.
    """

    name: str
    top_k: list[NeighbourList]


def reciprocal_rank_fusion(
    metric_lists: Sequence[MetricNeighbourhoods],
    *,
    k0: int = 60,
    n_entities: int,
) -> list[dict[int, tuple[float, int]]]:
    """Fuse per-metric top-K lists via RRF for every entity.

    For every ``(entity, neighbour)`` pair, sums
    ``1 / (k0 + rank_m(neighbour))`` across every metric whose top-K
    list for *entity* contains *neighbour*. Also records the number of
    metrics that supported each neighbour (the "agreement count"), so
    the caller can apply the ``c_min`` consensus rule.

    Parameters
    ----------
    metric_lists : sequence of MetricNeighbourhoods
        One per metric participating in the fusion.
    k0 : int, default 60
        Standard RRF damping constant.
    n_entities : int
        Total entity count (used to size the output list).

    Returns
    -------
    list of dict
        One entry per entity: a mapping
        ``neighbour_index -> (rrf_score, agreement_count)``.
    """
    if k0 <= 0:
        raise ValueError(f"k0 must be positive, got {k0}")

    fused: list[dict[int, tuple[float, int]]] = [dict() for _ in range(n_entities)]

    for metric in metric_lists:
        if len(metric.top_k) != n_entities:
            raise ValueError(
                f"Metric {metric.name!r} has {len(metric.top_k)} rows, "
                f"expected {n_entities}"
            )
        for ent_idx, neigh_list in enumerate(metric.top_k):
            for rank, (neigh_idx, _sim) in enumerate(neigh_list):
                contribution = 1.0 / (k0 + rank + 1)  # rank is 0-based
                entry = fused[ent_idx].get(neigh_idx)
                if entry is None:
                    fused[ent_idx][neigh_idx] = (contribution, 1)
                else:
                    score, agree = entry
                    fused[ent_idx][neigh_idx] = (
                        score + contribution,
                        agree + 1,
                    )
    return fused


# ---------------------------------------------------------------------------
# Density aggregation
# ---------------------------------------------------------------------------


@dataclass
class EntityDensity:
    """Density score for a single entity.

    Parameters
    ----------
    index : int
        Row index in the canonical frame.
    density : float
        Aggregate RRF density (including label-collision boost).
    rrf_component : float
        Pre-boost RRF contribution.
    label_collision_component : float
        Boost added when the entity sits in a label-collision group.
    agreement_counts : dict
        ``neighbour_index -> agreement_count`` for every contributing
        neighbour (post-``c_min`` filter).
    neighbour_count : int
        Number of neighbours that contributed to ``rrf_component``.
    """

    index: int
    density: float
    rrf_component: float
    label_collision_component: float
    agreement_counts: dict[int, int] = field(default_factory=dict)
    neighbour_count: int = 0


def compute_rrf_density(
    metric_lists: Sequence[MetricNeighbourhoods],
    *,
    n_entities: int,
    k0: int = 60,
    c_min: int = 2,
    label_collision_groups: dict[str, list[int]] | None = None,
    boost_label_collision: float = 5.0,
) -> list[EntityDensity]:
    """Aggregate per-entity RRF density with label-collision boost.

    Parameters
    ----------
    metric_lists : sequence of MetricNeighbourhoods
        Metrics participating in the fusion (typically 4 — label
        collision is handled separately via *label_collision_groups*).
    n_entities : int
        Total entity count.
    k0 : int, default 60
        RRF damping constant.
    c_min : int, default 2
        Minimum number of metrics that must agree on a neighbour before
        it contributes to density.
    label_collision_groups : dict or None
        Output of :func:`niche_metrics.label_collision_index` —
        ``normalised_label -> [row_index, ...]``. Entities inside any
        such group receive *boost_label_collision* added to density.
    boost_label_collision : float, default 5.0
        Additive density boost for label-colliding entities.

    Returns
    -------
    list of EntityDensity
        Densities in row order (length ``n_entities``).
    """
    if c_min < 1:
        raise ValueError(f"c_min must be >= 1, got {c_min}")

    fused = reciprocal_rank_fusion(
        metric_lists, k0=k0, n_entities=n_entities
    )

    colliding: set[int] = set()
    if label_collision_groups:
        for members in label_collision_groups.values():
            if len(members) >= 2:
                colliding.update(members)

    out: list[EntityDensity] = []
    for ent_idx in range(n_entities):
        rrf_component = 0.0
        agree_map: dict[int, int] = {}
        count = 0
        for neigh_idx, (score, agree) in fused[ent_idx].items():
            if agree < c_min:
                continue
            rrf_component += score
            agree_map[neigh_idx] = agree
            count += 1
        boost = (
            boost_label_collision
            if ent_idx in colliding and boost_label_collision > 0
            else 0.0
        )
        out.append(
            EntityDensity(
                index=ent_idx,
                density=rrf_component + boost,
                rrf_component=rrf_component,
                label_collision_component=boost,
                agreement_counts=agree_map,
                neighbour_count=count,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Ranking + removal selection
# ---------------------------------------------------------------------------


def rank_entities_by_density(
    densities: Sequence[EntityDensity],
    rng: np.random.Generator,
) -> list[int]:
    """Return entity indices sorted by descending density.

    Ties are broken deterministically by permuting the tie group with
    the supplied *rng*. The ordering is stable given a fixed seed.

    Parameters
    ----------
    densities : sequence of EntityDensity
    rng : numpy.random.Generator
        Seeded RNG used for tie-break ordering.

    Returns
    -------
    list of int
        Entity indices in descending density order.
    """
    # Bucket by density value (rounded to 9 decimals for float stability).
    buckets: dict[float, list[int]] = {}
    for d in densities:
        key = round(d.density, 9)
        buckets.setdefault(key, []).append(d.index)

    ordered_keys = sorted(buckets.keys(), reverse=True)
    out: list[int] = []
    for k in ordered_keys:
        bucket = buckets[k]
        if len(bucket) > 1:
            # Permute deterministically — stable with a fixed rng state.
            perm = rng.permutation(len(bucket))
            out.extend(bucket[int(p)] for p in perm)
        else:
            out.extend(bucket)
    return out


def select_for_removal(
    ranked_indices: Sequence[int],
    *,
    protection_flags: Sequence[bool],
    removal_fraction_cap: float,
) -> list[int]:
    """Return the candidate removal queue up to a hard cap.

    The caller is expected to consume the queue iteratively and re-run
    the corner-case miner after each removal to hit the target ratio.

    Parameters
    ----------
    ranked_indices : sequence of int
        Entity indices sorted by descending density.
    protection_flags : sequence of bool
        ``protection_flags[i]`` is True when entity index *i* is in
        ``expanded_positives`` and MUST NOT be removed.
    removal_fraction_cap : float
        Upper bound on the fraction of non-protected entities that may
        be removed (``removal_fraction_cap * total_nonprotected``).

    Returns
    -------
    list of int
        Entity indices eligible for removal, in iteration order
        (highest density first). Length is bounded by the cap.
    """
    if removal_fraction_cap < 0.0 or removal_fraction_cap > 1.0:
        raise ValueError(
            f"removal_fraction_cap must be in [0, 1], got {removal_fraction_cap}"
        )

    n_total = len(ranked_indices)
    n_protected = sum(1 for flag in protection_flags if flag)
    n_nonprotected = n_total - n_protected
    cap = int(removal_fraction_cap * n_nonprotected)

    queue: list[int] = []
    for idx in ranked_indices:
        if idx >= len(protection_flags):
            continue
        if protection_flags[idx]:
            continue
        queue.append(idx)
        if len(queue) >= cap:
            break
    return queue
