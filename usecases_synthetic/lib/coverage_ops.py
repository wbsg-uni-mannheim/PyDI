"""Coverage histogram measurement and entity-row selection for Knob 04.

Implements the primitives that Knob 04 — Per-entity Source Coverage Skew —
uses to measure, plan, and apply entity-row additions/removals.

See ``knobs/knob_04_coverage_skew.md`` for the full specification.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _source_tiebreak(entity_id: str, source: str) -> str:
    """Deterministic per-``(entity_id, source)`` tie-break key (R10-E).

    Replaces the alphabetical source tie-break in
    :func:`select_removal_candidates` so genuine rank ties spread
    demotions across sources instead of locking to the
    alphabetically-first source.
    """
    return hashlib.sha256(f"{entity_id}|{source}".encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Entity view
# ---------------------------------------------------------------------------


@dataclass
class EntityView:
    """Cross-source view of all entities (multi-source + singletons).

    Parameters
    ----------
    members : dict[str, dict[str, tuple[int, str]]]
        Mapping ``entity_id -> {source: (row_idx, record_id)}``. Each
        entity has at most one base row per source; within-source
        duplicates (from ``within_source_duplicate_rate``) are tracked
        separately on the ``extras`` list.
    extras : dict[str, list[tuple[str, int, str]]]
        Mapping ``entity_id -> list of (source, row_idx, record_id)``
        additional rows emitted by K4 (fabrications / within-source
        duplicates). Counted for coverage only via distinct sources on
        ``members``.
    source_count : int
        Total sources in the domain. Coverage histogram bins
        are ``1..source_count``.
    singleton_source : dict[str, str]
        Mapping ``entity_id -> source`` recording the synthetic singleton
        origin, if the entity was a singleton seed from that source.
        Only populated for the synthetic entity ids built from
        unlinked records.
    """

    members: dict[str, dict[str, tuple[int, str]]] = field(default_factory=dict)
    extras: dict[str, list[tuple[str, int, str]]] = field(default_factory=dict)
    source_count: int = 0
    singleton_source: dict[str, str] = field(default_factory=dict)

    def coverage(self, entity_id: str) -> int:
        """Return number of distinct sources currently covering an entity."""
        return len(self.members.get(entity_id, {}))

    def sources_of(self, entity_id: str) -> set[str]:
        """Return the set of sources currently covering an entity."""
        return set(self.members.get(entity_id, {}).keys())

    def __len__(self) -> int:
        return len(self.members)


def build_entity_view(
    linkage_groups: dict[str, list[tuple[str, str]]],
    sources: dict[str, pd.DataFrame],
    id_columns: dict[str, str],
    source_count: int,
) -> EntityView:
    """Build an :class:`EntityView` from EM linkage groups plus singletons.

    Every record in every source appears in exactly one entity:

    * Records named in ``linkage_groups`` (from the EM union-find) join
      the canonical group's entity.
    * Records not in any group become their own synthetic singleton
      entity with id ``__singleton__:<source>:<record_id>``.

    Parameters
    ----------
    linkage_groups : dict[str, list[tuple[str, str]]]
        Multi-source entity groups: ``group_id -> [(source, record_id)]``.
    sources : dict[str, DataFrame]
        Source DataFrames.
    id_columns : dict[str, str]
        Per-source ID column name.
    source_count : int
        Number of sources in the domain.

    Returns
    -------
    EntityView
        Fully populated entity view covering every row in every source.
    """
    view = EntityView(source_count=source_count)
    linked_records: set[tuple[str, str]] = set()

    # Source-local record lookups (one-pass).
    source_lookup: dict[str, dict[str, int]] = {}
    for source_name, df in sources.items():
        id_col = id_columns.get(source_name)
        if id_col is None or id_col not in df.columns:
            source_lookup[source_name] = {}
            continue
        lookup: dict[str, int] = {}
        for row_idx, rid in zip(df.index, df[id_col].astype(str)):
            lookup[rid] = int(row_idx)
        source_lookup[source_name] = lookup

    # Multi-source groups.
    for group_id, members in linkage_groups.items():
        per_source: dict[str, tuple[int, str]] = {}
        for source_name, record_id in members:
            lookup = source_lookup.get(source_name, {})
            if record_id not in lookup:
                continue
            linked_records.add((source_name, record_id))
            # First occurrence per source wins.
            if source_name not in per_source:
                per_source[source_name] = (lookup[record_id], record_id)
        if per_source:
            view.members[group_id] = per_source

    # Singletons (records not covered by any multi-source group).
    for source_name, df in sources.items():
        id_col = id_columns.get(source_name)
        if id_col is None or id_col not in df.columns:
            continue
        for row_idx, rid_val in zip(df.index, df[id_col].astype(str)):
            if (source_name, rid_val) in linked_records:
                continue
            sing_id = f"__singleton__:{source_name}:{rid_val}"
            view.members[sing_id] = {source_name: (int(row_idx), rid_val)}
            view.singleton_source[sing_id] = source_name

    return view


# ---------------------------------------------------------------------------
# Histogram measurement
# ---------------------------------------------------------------------------


def measure_coverage_histogram(
    view: EntityView, *, include_distractor_singletons: bool = False
) -> dict[int, float]:
    """Compute ``H[k] = fraction of entities covered by exactly k sources``.

    The knob operates on *matchable* entities — entities that come from
    the pool-unioned-with-gold linkage. Synthetic distractor singletons
    created by :func:`build_entity_view` for records that never appear
    in any linkage group are immovable by K4 (they cannot be fabricated
    cross-source because there is no ground-truth cross-source mate,
    and removing them would simply delete orphaned rows). Counting them
    in the denominator inflates ``H[1]`` on distractor-heavy corpora
    (e.g. companies measured ``H[1]=0.919`` EM-gold-only,
    ``H[1]=0.853`` pool-unioned — both dominated by ~11 000 distractor
    rows with no cross-source mate). The histogram therefore excludes
    distractor-origin singletons by default, matching the knob card's
    semantics ("fraction of *matchable* entities covered by k sources").

    Parameters
    ----------
    view : EntityView
    include_distractor_singletons : bool, optional
        When ``True``, revert to legacy behaviour and count synthetic
        distractor singletons (ids prefixed with ``__singleton__:``)
        in the histogram. Defaults to ``False``. Tests that construct
        an ``EntityView`` directly from synthetic linkage groups (no
        distractor singletons) are unaffected by this default.

    Returns
    -------
    dict[int, float]
        Mapping ``k -> fraction`` for ``k in 1..source_count``. Empty
        entities (coverage == 0) and — by default — distractor-origin
        synthetic singletons are excluded from the denominator.
    """
    if view.source_count == 0 or len(view) == 0:
        return {k: 0.0 for k in range(1, max(view.source_count, 1) + 1)}

    counts: dict[int, int] = {k: 0 for k in range(1, view.source_count + 1)}
    total = 0
    for entity_id, ent_members in view.members.items():
        if not include_distractor_singletons and entity_id in view.singleton_source:
            continue
        k = len(ent_members)
        if 1 <= k <= view.source_count:
            counts[k] += 1
            total += 1
    if total == 0:
        return {k: 0.0 for k in counts}
    return {k: counts[k] / total for k in counts}


def histogram_to_dataframe(hist: dict[int, float], label: str) -> pd.DataFrame:
    """Flatten a histogram to a 3-column DataFrame for serialisation.

    Parameters
    ----------
    hist : dict[int, float]
        Histogram mapping ``k -> fraction``.
    label : str
        Label recorded in the ``label`` column (e.g. ``"baseline"``).
    """
    rows = [
        {"label": label, "coverage": int(k), "fraction": float(v)}
        for k, v in sorted(hist.items())
    ]
    return pd.DataFrame(rows, columns=["label", "coverage", "fraction"])


def validate_target_histogram(target: dict[int, float], source_count: int) -> None:
    """Validate a target histogram sums to 1.0 over bins ``1..N``.

    Parameters
    ----------
    target : dict[int, float]
        Target histogram.
    source_count : int
        Number of sources (the histogram must cover bins ``1..N``).

    Raises
    ------
    ValueError
        If the histogram has missing bins or does not sum to 1.
    """
    bins = set(target.keys())
    expected = set(range(1, source_count + 1))
    if bins != expected:
        raise ValueError(
            f"Target histogram bins {sorted(bins)} != expected {sorted(expected)}"
        )
    total = sum(target.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Target histogram does not sum to 1.0 (got {total:.4f})")
    for k, v in target.items():
        if v < 0.0 or v > 1.0 + 1e-9:
            raise ValueError(f"Target histogram bin {k} = {v} is outside [0,1]")


# ---------------------------------------------------------------------------
# Shift planning (how many demotions / promotions per bin)
# ---------------------------------------------------------------------------


def plan_demotions(
    base_hist: dict[int, float],
    target_hist: dict[int, float],
    total_entities: int,
    source_count: int,
) -> dict[int, int]:
    """Return the number of bin-to-(bin-1) demotions required at each k.

    Solves the flow equation

        N[k] + d[k+1] - d[k] = T[k]

    for ``d[k]`` downward from ``k = N`` (where ``d[N+1] = 0``). ``d[k]``
    is the count of entities moved from bin ``k`` to bin ``k-1`` via the
    removal of one source row. Negative values (target demands promotion)
    are clamped to zero and logged by the caller.

    Parameters
    ----------
    base_hist : dict[int, float]
        Baseline coverage histogram.
    target_hist : dict[int, float]
        Target coverage histogram (typically ``H_target.hard``).
    total_entities : int
        Total number of entities (denominator for both histograms).
    source_count : int
        Number of coverage bins.

    Returns
    -------
    dict[int, int]
        Mapping ``k -> d[k]`` for ``k in 2..source_count``. Entries below
        2 are omitted (no demotion from bin 1 is possible — fusion
        survivor floor).
    """
    demotions: dict[int, int] = {}
    d_next = 0
    for k in range(source_count, 1, -1):
        n_k = int(round(base_hist.get(k, 0.0) * total_entities))
        t_k = int(round(target_hist.get(k, 0.0) * total_entities))
        d_k = (n_k - t_k) + d_next
        demotions[k] = max(d_k, 0)
        d_next = demotions[k]
    return demotions


def plan_promotions(
    base_hist: dict[int, float],
    target_hist: dict[int, float],
    total_entities: int,
    source_count: int,
) -> dict[int, int]:
    """Return the number of bin-to-(bin+1) promotions required at each k.

    Symmetric counterpart of :func:`plan_demotions` for the easy path.
    ``u[k]`` is the count of entities promoted from bin ``k`` to bin
    ``k+1`` via fabrication of one additional source row. Negative values
    (target demands demotion) are clamped to zero.

    Parameters
    ----------
    base_hist : dict[int, float]
        Baseline coverage histogram.
    target_hist : dict[int, float]
        Target coverage histogram (typically ``H_target.easy``).
    total_entities : int
        Total number of entities.
    source_count : int
        Number of coverage bins.

    Returns
    -------
    dict[int, int]
        Mapping ``k -> u[k]`` for ``k in 1..source_count-1``.
    """
    promotions: dict[int, int] = {}
    u_prev = 0
    for k in range(1, source_count):
        n_k = int(round(base_hist.get(k, 0.0) * total_entities))
        t_k = int(round(target_hist.get(k, 0.0) * total_entities))
        # N[k] + u[k-1] - u[k] = T[k] => u[k] = N[k] + u[k-1] - T[k]
        u_k = (n_k - t_k) + u_prev
        promotions[k] = max(u_k, 0)
        u_prev = promotions[k]
    return promotions


# ---------------------------------------------------------------------------
# Row selection for removal
# ---------------------------------------------------------------------------


@dataclass
class RemovalConstraints:
    """Bundle of constraint inputs for removal candidate eligibility.

    Parameters
    ----------
    fusion_val_test_ids : set[str]
        Record-IDs of fusion val + test entities (the protected fusion
        universe per §"Terminology convention" in plan_s1_scale.md).
        These entities can still be collapsed (e.g. 3-source -> 1-source)
        but the surviving source is chosen by closeness to the fusion
        target value rather than by conflict-preserving rank — see
        :func:`score_target_distance` and the K4 sign-off (Pending #5
        wire-up, 2026-05-07).
    protected_records : set[tuple[str, str]]
        ``(source, record_id)`` pairs that must never be removed.
        Reserved for explicit anchor records (e.g. K2 distractor anchors
        or K1 fix-on-collapse anchors). **Pool-pair endpoints are NOT
        in this set anymore** — their orphan-check moved to
        :func:`_would_break_pool_edge`.
    distractor_entity_ids : set[str]
        Entity ids produced by Knob 2 as single-source distractors; never
        eligible for removal.
    singleton_cap : float
        Maximum fraction of single-source entities allowed at hard.
    """

    fusion_val_test_ids: set[str] = field(default_factory=set)
    protected_records: set[tuple[str, str]] = field(default_factory=set)
    distractor_entity_ids: set[str] = field(default_factory=set)
    singleton_cap: float = 0.70

    @property
    def fusion_gold_entity_ids(self) -> set[str]:
        """Back-compat alias for :attr:`fusion_val_test_ids`.

        Kept temporarily so any external readers (tests / orchestration
        scripts) that still reference the old name continue to work.
        New code should use ``fusion_val_test_ids``.
        """
        return self.fusion_val_test_ids


def score_conflict(
    sources: dict[str, pd.DataFrame],
    members: dict[str, tuple[int, str]],
    primary_cols: dict[str, str],
) -> dict[str, int]:
    """Return per-source redundancy count for an entity.

    Lower score = more "unique" (disagreeing) relative to other sources,
    higher score = more redundant (agrees with another source). Removal
    prefers high-score rows (conflict-preserving removal).

    Parameters
    ----------
    sources : dict[str, DataFrame]
    members : dict[str, tuple[int, str]]
        Current members of an entity.
    primary_cols : dict[str, str]
        Per-source column to compare values on (the "primary" attribute).

    Returns
    -------
    dict[str, int]
        Mapping ``source -> agreement count``.
    """
    values: dict[str, str] = {}
    for source, (row_idx, _) in members.items():
        col = primary_cols.get(source)
        if col is None or col not in sources[source].columns:
            continue
        val = sources[source].at[row_idx, col]
        if pd.isna(val):
            continue
        values[source] = str(val).strip().lower()

    scores: dict[str, int] = {s: 0 for s in members}
    sources_list = list(values.keys())
    for i, s1 in enumerate(sources_list):
        for s2 in sources_list[i + 1 :]:
            if values[s1] == values[s2]:
                scores[s1] = scores.get(s1, 0) + 1
                scores[s2] = scores.get(s2, 0) + 1
    return scores


def _would_break_pool_edge(
    entity_id: str,
    source_to_remove: str,
    view: EntityView,
    pool_pairs_index: dict[tuple[str, str], list[tuple[str, str]]],
    removed_set: set[tuple[str, str]],
) -> bool:
    """Check if removing ``(entity, source)`` would orphan a pool edge.

    A pool pair declares a match between two records on different sources.
    Removing one endpoint is **allowed** as long as the other endpoint is
    still alive (not yet in *removed_set*). Removal is **blocked** only
    when removing this record would orphan a pool pair — i.e. some pool
    pair through this record has its partner already in the removed set,
    so dropping this endpoint would collapse the pool-declared match onto
    zero sources.

    This is the K4 sign-off semantic (rewritten 2026-05-07): the prior
    "block on any pool edge" check was too strict and rejected every
    record participating in any pool pair, making K4 hard a phantom knob
    on every domain. Single-endpoint removal is now allowed; both-endpoint
    removal of the same pool pair is forbidden.

    Parameters
    ----------
    entity_id : str
        Entity being demoted.
    source_to_remove : str
        Source whose record we are considering removing for ``entity_id``.
    view : EntityView
        Read-only entity-view (members / coverage state pre-mutation).
    pool_pairs_index : dict[(source, rid), list[(source, rid)]]
        Per-record reverse index of pool pairs: each record maps to the
        list of partner records it is paired with. Built once by the
        caller from the raw ``pool_pairs`` list.
    removed_set : set[(source, rid)]
        In-progress removal set — every record already chosen for removal
        in this dispatcher pass.

    Returns
    -------
    bool
        ``True`` iff removing ``(source_to_remove, record_id)`` for
        ``entity_id`` would orphan at least one pool pair (i.e. some
        partner of this record is already in ``removed_set``).
    """
    members = view.members.get(entity_id, {})
    if source_to_remove not in members:
        return False
    _, record_id = members[source_to_remove]
    target_key = (source_to_remove, record_id)
    partners = pool_pairs_index.get(target_key, ())
    for partner in partners:
        if partner in removed_set:
            return True
    return False


def _build_pool_pairs_index(
    pool_pairs: list[tuple[tuple[str, str], tuple[str, str]]],
) -> dict[tuple[str, str], list[tuple[str, str]]]:
    """Build the per-record partner index used by :func:`_would_break_pool_edge`."""
    index: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for left, right in pool_pairs:
        index.setdefault(left, []).append(right)
        index.setdefault(right, []).append(left)
    return index


def score_target_distance(
    sources: dict[str, pd.DataFrame],
    members: dict[str, tuple[int, str]],
    entity_id: str,
    fusion_targets: dict[str, dict[str, list[str]]],
    attribute_mapping: dict[str, dict[str, str]],
    domain: str,
    tol_overrides: dict[str, dict[str, float | str]] | None = None,
) -> dict[str, float]:
    """Per-source closeness score against fusion target values.

    For each source's record in *members*, compute the fraction of
    canonical attributes whose source value is "close enough" to a fusion
    target value under
    :func:`usecases_synthetic.lib.protection.is_close_enough` with the
    per-attribute tolerance from
    :func:`usecases_synthetic.lib.protection.fusion_cell_tolerance`.

    Higher score = closer to fusion target = should be **kept** (last to
    remove). Lower score = further from target = removed first. Removal
    selection sorts ascending on this score for fusion val/test entities.

    Parameters
    ----------
    sources : dict[str, DataFrame]
        Source DataFrames.
    members : dict[str, tuple[int, str]]
        Members of an entity: ``{source_name: (row_idx, record_id)}``.
    entity_id : str
        Entity id (used to look up the fusion target).
    fusion_targets : dict[entity_id, dict[canonical_attr, list[str]]]
        Output of
        :func:`usecases_synthetic.lib.protection.load_fusion_target_values`.
    attribute_mapping : dict[source, dict[source_col, canonical_attr]]
        Per-source column-to-canonical-attribute map (the K4 YAML
        ``attribute_mapping`` block when present, falling back to
        ``primary_columns`` semantics for sources without a richer map).
    domain : str
        Domain name (for tolerance resolution).
    tol_overrides : dict or None
        Optional per-attribute tolerance overrides forwarded to
        :func:`fusion_cell_tolerance`.

    Returns
    -------
    dict[str, float]
        ``{source: closeness_score in [0, 1]}``. Sources with no
        evaluable cells (no fusion target available, no source value, or
        no overlap with the entity's tracked attributes) score ``0.0``.
    """
    # Local imports keep coverage_ops.py free of an import-time dependency
    # on protection.py (which only matters when a fusion target is
    # actually being evaluated).
    from usecases_synthetic.lib.protection import (
        fusion_cell_tolerance,
        is_close_enough,
    )

    target_attrs = fusion_targets.get(entity_id, {})
    if not target_attrs:
        return {source: 0.0 for source in members}

    scores: dict[str, float] = {}
    for source, (row_idx, _) in members.items():
        df = sources.get(source)
        if df is None:
            scores[source] = 0.0
            continue
        source_attr_map = attribute_mapping.get(source, {})
        n_total = 0
        n_close = 0
        for src_col, canonical_attr in source_attr_map.items():
            if src_col not in df.columns:
                continue
            target_vals = target_attrs.get(canonical_attr, [])
            if not target_vals:
                continue
            cell_val = df.at[row_idx, src_col]
            if pd.isna(cell_val):
                continue
            n_total += 1
            tol = fusion_cell_tolerance(domain, canonical_attr, tol_overrides)
            for tv in target_vals:
                if is_close_enough(str(cell_val), tv, tol):
                    n_close += 1
                    break
        scores[source] = (n_close / n_total) if n_total > 0 else 0.0
    return scores


def select_removal_candidates(
    view: EntityView,
    demotions: dict[int, int],
    constraints: RemovalConstraints,
    pool_pairs: list[tuple[tuple[str, str], tuple[str, str]]],
    sources: dict[str, pd.DataFrame],
    primary_cols: dict[str, str],
    rng: np.random.Generator,
    fusion_targets: dict[str, dict[str, list[str]]] | None = None,
    attribute_mapping: dict[str, dict[str, str]] | None = None,
    domain: str | None = None,
    tol_overrides: dict[str, dict[str, float | str]] | None = None,
    per_source_demotion_cap: float = 1.0,
) -> list[tuple[str, str, str]]:
    """Select entity-source rows to remove to satisfy the demotion plan.

    Iterates bins from ``k = N`` down to ``k = 2``; at each bin, picks
    ``demotions[k]`` entities currently at coverage ``k`` and chooses
    which source row to drop. The within-entity rank is:

    - **Fusion val/test entities** (when *fusion_targets*, *attribute_mapping*,
      and *domain* are provided): rank ascending by closeness to the
      entity's fusion target values under
      :func:`score_target_distance` — the source furthest from the
      fusion target is removed first, so the surviving source is the
      closest one. K4 sign-off Pending #5 wire-up (2026-05-07).
    - **Other entities**: rank descending by conflict score
      (redundant/agreeing sources removed first to preserve
      disagreement evidence — :func:`score_conflict`).

    Constraint checks:

    1. Fusion floor — never drop to zero sources (handled by ``k>=2``).
    2. Distractor passthrough — never select K2 distractors.
    3. Explicit protected records — never drop anchor records.
    4. Pool-pair orphan check — never drop a record that would orphan
       a pool pair, i.e. its only-still-alive partner. Single-endpoint
       removal is allowed (:func:`_would_break_pool_edge`).

    Parameters
    ----------
    view : EntityView
        Mutable view; this function does not mutate ``view`` — it returns
        the selected triples and leaves mutation to the caller.
    demotions : dict[int, int]
        Plan from :func:`plan_demotions`.
    constraints : RemovalConstraints
    pool_pairs : list[((source, rid), (source, rid))]
        Pool-declared match edges. Indexed once into per-record partner
        lists for the orphan check.
    sources : dict[str, DataFrame]
    primary_cols : dict[str, str]
        Per-source primary column for conflict scoring.
    rng : Generator
        Seeded RNG.
    fusion_targets : dict or None
        Output of
        :func:`usecases_synthetic.lib.protection.load_fusion_target_values`.
        When provided alongside *attribute_mapping* and *domain*,
        fusion val/test entities use closeness ranking.
    attribute_mapping : dict or None
        Per-source ``{col: canonical_attr}`` map for closeness scoring.
    domain : str or None
        Domain name (for tolerance resolution in
        :func:`score_target_distance`).
    tol_overrides : dict or None
        Optional per-attribute tolerance overrides forwarded to
        :func:`fusion_cell_tolerance`.
    per_source_demotion_cap : float, default 1.0
        R10-E: maximum fraction of a source's rows (at K4 entry) that may
        be demoted. Once a source reaches ``cap * baseline_rows`` removals
        it is skipped within each entity's ranked source list, falling
        through to the next-ranked source; if every source in a cluster
        is capped (or otherwise blocked) the entity demotion is skipped.
        The default ``1.0`` is a no-op (a source can never lose more than
        100% of its rows). The per-domain K4 YAML sets ``0.40`` so a
        single source (e.g. products_1, the EM anchor) is not decimated,
        which would silently delete EM-gold edges that are not
        regenerated post-K4.

    Returns
    -------
    list[tuple[str, str, str]]
        List of ``(entity_id, source, record_id)`` triples selected
        for removal, in application order.
    """
    # Build mutable per-bin pools of eligible entities, sorted for
    # deterministic iteration order.
    per_bin: dict[int, list[str]] = {k: [] for k in range(2, view.source_count + 1)}
    for entity_id in sorted(view.members.keys()):
        if entity_id in constraints.distractor_entity_ids:
            continue
        k = view.coverage(entity_id)
        if k >= 2:
            per_bin[k].append(entity_id)

    # Simulated coverage that evolves as we select removals.
    sim_coverage: dict[str, int] = {eid: view.coverage(eid) for eid in view.members}

    # Per-record reverse index of pool pairs (for the orphan check).
    pool_pairs_index = _build_pool_pairs_index(pool_pairs)

    # In-progress removed-set tracked across bins so the orphan check
    # sees previously-selected removals.
    removed_set: set[tuple[str, str]] = set()

    # R10-E: per-source demotion cap. ``baseline_counts`` is the source
    # row count at K4 entry; ``removed_per_source`` accumulates selected
    # removals so a single source can shed at most ``cap *`` its rows.
    baseline_counts: dict[str, int] = {src: int(len(df)) for src, df in sources.items()}
    removed_per_source: dict[str, int] = {}

    closeness_enabled = (
        fusion_targets is not None
        and attribute_mapping is not None
        and domain is not None
    )

    selected: list[tuple[str, str, str]] = []

    for k in range(view.source_count, 1, -1):
        needed = demotions.get(k, 0)
        if needed <= 0:
            continue
        # Stable RNG-driven shuffle of eligible entities.
        pool = list(per_bin.get(k, []))
        order = rng.permutation(len(pool)) if pool else np.array([], dtype=int)
        candidates = [pool[i] for i in order]

        for entity_id in candidates:
            if needed <= 0:
                break
            # Skip if coverage shifted due to higher-bin demotions.
            if sim_coverage.get(entity_id, 0) != k:
                continue
            members = view.members[entity_id]

            # Per-entity rank: closeness for fusion val/test entities
            # (when wired), conflict-preserving rank otherwise.
            is_fusion_protected = (
                closeness_enabled and entity_id in constraints.fusion_val_test_ids
            )
            if is_fusion_protected:
                distance = score_target_distance(
                    sources=sources,
                    members=members,
                    entity_id=entity_id,
                    fusion_targets=fusion_targets,  # type: ignore[arg-type]
                    attribute_mapping=attribute_mapping,  # type: ignore[arg-type]
                    domain=domain,  # type: ignore[arg-type]
                    tol_overrides=tol_overrides,
                )
                # Ascending: low closeness first = removed first;
                # high closeness last = preserved as the survivor.
                # R10-E: hash tie-break (was alphabetical ``s``).
                sorted_sources = sorted(
                    members.keys(),
                    key=lambda s: (
                        distance.get(s, 0.0),
                        _source_tiebreak(entity_id, s),
                    ),
                )
            else:
                conflict = score_conflict(sources, members, primary_cols)
                # Descending conflict: redundant (high-agreement) sources first.
                # R10-E: hash tie-break (was alphabetical ``s``).
                sorted_sources = sorted(
                    members.keys(),
                    key=lambda s: (-conflict.get(s, 0), _source_tiebreak(entity_id, s)),
                )

            chosen: tuple[str, str, str] | None = None
            cap_blocked = False
            for source in sorted_sources:
                row_idx, record_id = members[source]
                # Constraint: explicit protected anchor records.
                if (source, record_id) in constraints.protected_records:
                    continue
                # Constraint: pool-pair orphan check (new orphan-only
                # semantic; single-endpoint removal allowed).
                if _would_break_pool_edge(
                    entity_id, source, view, pool_pairs_index, removed_set
                ):
                    continue
                # R10-E: per-source demotion cap. Skip a source once it
                # has shed ``cap`` of its rows so no single source (e.g.
                # the EM-anchor source) is decimated.
                cap_limit = per_source_demotion_cap * baseline_counts.get(source, 0)
                if removed_per_source.get(source, 0) >= cap_limit:
                    cap_blocked = True
                    continue
                chosen = (entity_id, source, record_id)
                break

            if chosen is None:
                if cap_blocked:
                    logger.debug(
                        "K4 demotion skipped for entity %s (k=%d): "
                        "per_source_cap reached for all eligible sources",
                        entity_id,
                        k,
                    )
                continue
            selected.append(chosen)
            removed_set.add((chosen[1], chosen[2]))
            removed_per_source[chosen[1]] = removed_per_source.get(chosen[1], 0) + 1
            needed -= 1
            sim_coverage[entity_id] = k - 1
            # If demoted entity still needs to demote again, bump it into
            # the lower bin for the next k.
            if (k - 1) in per_bin:
                per_bin[k - 1].append(entity_id)

    return selected


def apply_singleton_cap_rollback(
    view: EntityView,
    selected: list[tuple[str, str, str]],
    cap: float,
) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    """Roll back the most recent removals until the singleton cap holds.

    The cap applies to **matchable** entities only — entities that K4
    can move along the coverage histogram. Synthetic distractor
    singletons (entity ids prefixed ``__singleton__:`` per
    :func:`build_entity_view`) live outside the histogram by
    construction (no cross-source mate exists, so no demotion is
    possible) and counting them in the denominator would inflate the
    realised singleton fraction with rows K4 cannot move. Same fix
    flavour as the K3 follow-up #1 (rollback scoping). 2026-05-07.

    Parameters
    ----------
    view : EntityView
        Unmutated view (baseline coverage is read from here).
    selected : list[(entity_id, source, record_id)]
        Candidate removals, in application order.
    cap : float
        Maximum fraction of single-source entities (over matchable
        entities only).

    Returns
    -------
    accepted : list[(entity_id, source, record_id)]
        Removals to actually apply.
    rolled_back : list[(entity_id, source, record_id)]
        Removals dropped to satisfy the cap.
    """
    matchable_ids = {
        eid for eid in view.members if not eid.startswith("__singleton__:")
    }
    total = len(matchable_ids)
    if total == 0:
        return list(selected), []

    # Simulate final coverage after applying all removals.
    sim_coverage: dict[str, int] = {eid: view.coverage(eid) for eid in matchable_ids}
    for entity_id, _, _ in selected:
        if entity_id in sim_coverage:
            sim_coverage[entity_id] = max(0, sim_coverage[entity_id] - 1)

    def _singleton_frac(cov: dict[str, int]) -> float:
        s = sum(1 for k in cov.values() if k == 1)
        return s / total if total > 0 else 0.0

    accepted = list(selected)
    rolled_back: list[tuple[str, str, str]] = []
    frac = _singleton_frac(sim_coverage)
    while frac > cap and accepted:
        # Roll back the most recent removal that actually produced a
        # singleton (coverage of the entity is currently 1 and would
        # become 2 if we un-remove it).
        rolled_back_this_pass = False
        for i in range(len(accepted) - 1, -1, -1):
            entity_id, source, record_id = accepted[i]
            if entity_id not in sim_coverage:
                continue
            if sim_coverage[entity_id] == 1:
                sim_coverage[entity_id] = 2
                rolled_back.append(accepted.pop(i))
                rolled_back_this_pass = True
                break
        if not rolled_back_this_pass:
            # No more singleton-producing removals; cap cannot shrink further.
            break
        frac = _singleton_frac(sim_coverage)

    return accepted, rolled_back


# ---------------------------------------------------------------------------
# Row fabrication (easy path)
# ---------------------------------------------------------------------------


def select_fabrication_candidates(
    view: EntityView,
    promotions: dict[int, int],
    rng: np.random.Generator,
) -> list[tuple[str, str]]:
    """Select entity-target-source pairs to fabricate rows for.

    Iterates bins from ``k = 1`` up to ``k = N - 1``; at each bin, picks
    ``promotions[k]`` entities currently at coverage ``k`` and chooses a
    missing source as the fabrication target.

    Parameters
    ----------
    view : EntityView
    promotions : dict[int, int]
        Plan from :func:`plan_promotions`.
    rng : Generator

    Returns
    -------
    list[tuple[str, str]]
        ``(entity_id, target_source)`` pairs, in application order.
    """
    all_sources = set()
    for members in view.members.values():
        all_sources.update(members.keys())
    # Prefer the deterministic source ordering from whatever sources the
    # view already sees. Source count is canonical from the config.
    ordered_sources = sorted(all_sources)

    per_bin: dict[int, list[str]] = {k: [] for k in range(1, view.source_count)}
    for entity_id in sorted(view.members.keys()):
        k = view.coverage(entity_id)
        if 1 <= k < view.source_count:
            per_bin[k].append(entity_id)

    sim_coverage: dict[str, int] = {eid: view.coverage(eid) for eid in view.members}
    sim_members: dict[str, set[str]] = {
        eid: set(m.keys()) for eid, m in view.members.items()
    }

    selected: list[tuple[str, str]] = []

    for k in range(1, view.source_count):
        needed = promotions.get(k, 0)
        if needed <= 0:
            continue
        pool = list(per_bin.get(k, []))
        order = rng.permutation(len(pool)) if pool else np.array([], dtype=int)
        candidates = [pool[i] for i in order]

        for entity_id in candidates:
            if needed <= 0:
                break
            if sim_coverage.get(entity_id, 0) != k:
                continue
            missing = [s for s in ordered_sources if s not in sim_members[entity_id]]
            if not missing:
                continue
            # Deterministic pick of the first missing source (sorted order).
            target_source = missing[0]
            selected.append((entity_id, target_source))
            needed -= 1
            sim_coverage[entity_id] = k + 1
            sim_members[entity_id].add(target_source)
            if (k + 1) in per_bin:
                per_bin[k + 1].append(entity_id)

    return selected


def fabricate_row_by_paraphrase(
    sibling_row: pd.Series,
    target_source_columns: list[str],
    managed_columns: list[str],
    paraphrase_fn: Callable[
        [str, str, np.random.Generator], tuple[str, dict[str, Any]]
    ],
    rng: np.random.Generator,
    new_record_id: str,
    target_id_column: str | None,
) -> tuple[pd.Series, dict[str, dict[str, Any]]]:
    """Build a fabricated row by paraphrasing a sibling's values.

    Parameters
    ----------
    sibling_row : Series
        Row from a sibling source carrying the entity. Its values are
        cloned and then paraphrased (per-cell) to break sibling-identity.
    target_source_columns : list[str]
        Column set of the target source (the fabricated row will be
        aligned to these columns; missing columns become NaN).
    managed_columns : list[str]
        Columns that paraphrase may touch (typically primary/key/
        categorical attrs from K1 config).
    paraphrase_fn : callable
        Single-cell paraphrase callable with signature
        ``(attribute_class, value, rng) -> (new_value, params)``.
    rng : Generator
        RNG threaded from Knob 4's seeded generator.
    new_record_id : str
        Stable synthetic record id for the fabricated row.
    target_id_column : str | None
        ID column of the target source (filled with ``new_record_id``).

    Returns
    -------
    row : Series
        The fabricated row (indexed by ``target_source_columns``).
    per_cell_params : dict[str, dict[str, Any]]
        Per-column paraphrase params (empty for untouched columns).
    """
    data: dict[str, Any] = {col: pd.NA for col in target_source_columns}
    per_cell_params: dict[str, dict[str, Any]] = {}

    # Copy sibling values for any column present in the target source.
    for col in target_source_columns:
        if col in sibling_row.index:
            data[col] = sibling_row[col]

    # Paraphrase managed columns in deterministic order.
    for col in sorted(managed_columns):
        if col not in target_source_columns:
            continue
        val = data.get(col)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue
        val_str = str(val)
        new_val, params = paraphrase_fn(col, val_str, rng)
        data[col] = new_val
        per_cell_params[col] = params

    if target_id_column is not None and target_id_column in target_source_columns:
        data[target_id_column] = new_record_id

    return pd.Series(data, index=target_source_columns), per_cell_params
