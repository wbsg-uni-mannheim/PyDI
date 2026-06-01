"""Silver-augmented protection targets (plan_revision.md C9 / step 4b).

Wires the per-domain :mod:`fusion_silver_standard` output into the same
shape :func:`protection.load_fusion_target_values` returns, so the
existing per-cell closeness gate (``cell_has_close_survivor``) can be
swapped from gold-only to silver-augmented protection without touching
its check semantics.

Merge rule (user directive 2026-05-23): **gold wins per-(member,
attribute) where it exists; silver fills the rest.** Concretely:

* For each member id m, for each attribute a:
    - If m appears in the fusion val/test gold AND gold[m][a] is
      non-empty → use the gold value (unchanged from today).
    - Else if m belongs to a silver cluster C → use silver[C][a].
    - Else → no target (vacuous-true under the closeness contract).

The silver universe is much wider than gold (~4 280 clusters for
music vs ~200 fusion val/test entities) so silver-source protection
constrains K1/K6 across the full pool, not just the load-bearing
fusion val/test entities.

K5 (format) and K10 (reliability) are intentionally *not* protected by
this loader — per the K5 design, format-equivalent values fuse to the
same fused result; per the K10 design, reliability reshuffle leaves
the value pool unchanged.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from .fusion_silver_standard import silver_path
from .protection import _load_fusion_protected_ids, load_fusion_target_values

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Silver-side loaders
# ---------------------------------------------------------------------------


def silver_standard_available(domain: str) -> bool:
    """Return True iff the silver standard CSV exists for *domain*."""
    return silver_path(domain, "csv").exists()


def load_silver_cluster_targets(
    domain: str,
) -> dict[str, dict[str, list[str]]]:
    """Load silver as ``{cluster_id: {attribute: [value]}}``.

    The silver standard ships exactly one fused value per
    (cluster_id, attribute); the loader wraps each non-empty value in
    a single-element list so the shape matches
    :func:`protection.load_fusion_target_values`. Empty / NaN silver
    cells (e.g. companies ``keypeople`` for clusters with no dbpedia
    member) are dropped — no target → vacuous-true under the closeness
    contract, matching how the gold-side absent-target case is handled.
    """
    csv = silver_path(domain, "csv")
    if not csv.exists():
        raise FileNotFoundError(
            f"Silver standard not built for {domain!r} at {csv}. "
            f"Run scripts/build_fusion_silver_standard.py --domain {domain}."
        )

    df = pd.read_csv(csv, keep_default_na=False, na_values=[""])
    targets: dict[str, dict[str, list[str]]] = {}
    for _, row in df.iterrows():
        cid = str(row["cluster_id"])
        attr = str(row["attribute"])
        value: Any = row["fused_value"]
        if value is None:
            continue
        if isinstance(value, float) and pd.isna(value):
            continue
        text = str(value).strip()
        if not text:
            continue
        targets.setdefault(cid, {})[attr] = [text]
    return targets


def load_silver_member_to_cluster(domain: str) -> dict[str, str]:
    """Build ``{member_id: cluster_id}`` from the silver's source_ids column.

    Each silver row carries the comma-joined list of source-record ids
    that make up the cluster; this loader expands that to a flat lookup
    so the K1/K6 protection check can pick out any member id and route
    to the right cluster's silver target dict.
    """
    csv = silver_path(domain, "csv")
    df = pd.read_csv(csv, keep_default_na=False, na_values=[""])
    out: dict[str, str] = {}
    seen: set[str] = set()
    for _, row in df.iterrows():
        cid = str(row["cluster_id"])
        if cid in seen:
            continue
        seen.add(cid)
        for member in str(row["source_ids"]).split(","):
            mid = member.strip()
            if mid:
                out[mid] = cid
    return out


# ---------------------------------------------------------------------------
# Combined gold + silver target view
# ---------------------------------------------------------------------------


def load_combined_target_values(
    domain: str,
) -> dict[str, dict[str, list[str]]]:
    """Return ``{member_id: {attribute: [value, ...]}}`` with gold-wins-per-cell.

    Wraps the existing :func:`protection.load_fusion_target_values`
    gold-only loader and grafts silver targets on top: every silver
    cluster member gets its cluster's silver dict as a fallback,
    *except* for cells where the gold already authored a value (gold
    wins per-(member, attribute) per the 2026-05-23 user directive).

    The output shape is identical to
    :func:`protection.load_fusion_target_values` so the existing
    :class:`apply_knob_01_surface._ClosenessContext` /
    :class:`apply_knob_06_noise._ClosenessContext` can swap it in
    transparently.
    """
    gold = load_fusion_target_values(domain)
    silver = load_silver_cluster_targets(domain)
    member_to_cluster = load_silver_member_to_cluster(domain)

    combined: dict[str, dict[str, list[str]]] = {}

    # 1. Gold targets — copied verbatim. Gold wins.
    for entity_id, attrs in gold.items():
        combined[entity_id] = {a: list(v) for a, v in attrs.items()}

    # 2. Silver fallback — every cluster member gets its cluster's silver
    # dict, except for (entity, attribute) cells that gold already
    # authored.
    for member_id, cluster_id in member_to_cluster.items():
        cluster_attrs = silver.get(cluster_id, {})
        if not cluster_attrs:
            continue
        member_bucket = combined.setdefault(member_id, {})
        for attr, values in cluster_attrs.items():
            if attr in member_bucket:
                # Gold already wrote this cell — gold wins.
                continue
            member_bucket[attr] = list(values)

    logger.info(
        "Combined targets for %s: %d gold entities + %d silver-only members "
        "= %d total target buckets",
        domain,
        len(gold),
        len(combined) - len(gold),
        len(combined),
    )
    return combined


def load_silver_protected_ids(domain: str) -> set[str]:
    """Protection universe under silver source: gold ids ∪ every silver-cluster member.

    Matches the universe ``load_combined_target_values`` populates.
    """
    gold_ids = _load_fusion_protected_ids(domain)
    member_to_cluster = load_silver_member_to_cluster(domain)
    return gold_ids | set(member_to_cluster.keys())


def load_intact_silver_clusters(
    domain: str, surviving_record_ids: set[str]
) -> set[str]:
    """Return silver cluster IDs whose entire original member set
    is present in *surviving_record_ids* (plan_revision.md C13).

    A silver cluster's fused target was computed against the cluster's
    original full membership. Once K2 drops any original member the
    cluster is "broken" — the silver target is no longer the value the
    surviving members would actually fuse to, so applying it as a K1/K6
    closeness constraint becomes incoherent. This helper identifies the
    "intact" set: clusters whose original ``source_ids`` list is fully
    contained in *surviving_record_ids*.

    Parameters
    ----------
    domain : str
        Domain name. The silver standard must be built; absent silver
        returns an empty set (no clusters can be intact if none exist).
    surviving_record_ids : set of str
        The record IDs that still exist in the post-K2 source frames.

    Returns
    -------
    set of str
        Cluster IDs of intact silver clusters. Broken clusters and
        clusters not in this domain's silver are NOT included.
    """
    if not silver_standard_available(domain):
        return set()
    member_to_cluster = load_silver_member_to_cluster(domain)
    cluster_to_members: dict[str, set[str]] = {}
    for member, cluster in member_to_cluster.items():
        cluster_to_members.setdefault(cluster, set()).add(member)
    intact: set[str] = set()
    for cluster_id, original_members in cluster_to_members.items():
        if original_members.issubset(surviving_record_ids):
            intact.add(cluster_id)
    return intact


def load_combined_target_values_intact_only(
    domain: str, intact_cluster_ids: set[str]
) -> dict[str, dict[str, list[str]]]:
    """Like :func:`load_combined_target_values` but skips silver targets
    for any cluster NOT in *intact_cluster_ids* (C13 intact-cluster rule).

    Gold targets are always copied verbatim (gold survival protection is
    unconditional). Silver targets are added per cluster member only when
    that member's cluster is intact. Members of broken clusters get
    nothing from silver — their cells fall through to vacuous-true in the
    downstream closeness check, restoring full K1/K6 mutation freedom on
    those records.

    Parameters
    ----------
    domain : str
        Domain name with silver standard built.
    intact_cluster_ids : set of str
        Output of :func:`load_intact_silver_clusters`.

    Returns
    -------
    dict
        ``{entity_id: {attribute: [value, ...]}}`` — same shape as
        :func:`load_combined_target_values` and
        :func:`protection.load_fusion_target_values` so the downstream
        :class:`_ClosenessContext` can swap it in transparently.
    """
    gold = load_fusion_target_values(domain)
    silver = load_silver_cluster_targets(domain)
    member_to_cluster = load_silver_member_to_cluster(domain)

    combined: dict[str, dict[str, list[str]]] = {}
    for entity_id, attrs in gold.items():
        combined[entity_id] = {a: list(v) for a, v in attrs.items()}

    silver_added_members = 0
    silver_skipped_broken = 0
    for member_id, cluster_id in member_to_cluster.items():
        if cluster_id not in intact_cluster_ids:
            silver_skipped_broken += 1
            continue
        cluster_attrs = silver.get(cluster_id, {})
        if not cluster_attrs:
            continue
        member_bucket = combined.setdefault(member_id, {})
        added_any = False
        for attr, values in cluster_attrs.items():
            if attr in member_bucket:
                continue  # gold wins per (member, attribute)
            member_bucket[attr] = list(values)
            added_any = True
        if added_any:
            silver_added_members += 1

    logger.info(
        "Intact-cluster combined targets for %s: %d gold entities + "
        "%d silver-added members (skipped %d members in broken clusters)",
        domain,
        len(gold),
        silver_added_members,
        silver_skipped_broken,
    )
    return combined


# ---------------------------------------------------------------------------
# Public dispatcher (used by _ClosenessContext)
# ---------------------------------------------------------------------------


PROTECTION_SOURCES = ("gold", "silver")


def resolve_protection_sources(
    domain: str,
    protection_source: str,
    surviving_record_ids: set[str] | None = None,
) -> tuple[set[str], dict[str, dict[str, list[str]]]]:
    """Return (protected_ids, target_values) for the requested source.

    Parameters
    ----------
    domain : str
        Domain name (``music`` / ``games`` / ``companies``).
    protection_source : str
        One of :data:`PROTECTION_SOURCES`.

        - ``gold``: gold-only protection set + targets.
        - ``silver``: silver-augmented (gold wins, silver fills the rest).
          Falls back to ``gold`` with a warning when the per-domain
          silver standard is not built.
    surviving_record_ids : set of str or None
        When provided AND ``protection_source == "silver"``, applies the
        C13 intact-cluster rule: silver targets are included only for
        clusters whose entire original member set is present in
        *surviving_record_ids*. Broken-cluster members get no silver
        target (their cells go vacuous-true in the closeness check).
        Required by K1/K6 in production (the caller derives this from
        the post-K2 sources). ``None`` falls back to the legacy
        all-silver-targets behavior used by ablation / debugging callers
        that don't know which clusters K2 broke.
    """
    if protection_source not in PROTECTION_SOURCES:
        raise ValueError(
            f"Unknown protection_source {protection_source!r}; "
            f"expected one of {PROTECTION_SOURCES}"
        )

    if protection_source == "gold":
        return (
            _load_fusion_protected_ids(domain),
            load_fusion_target_values(domain),
        )

    # silver
    if not silver_standard_available(domain):
        logger.warning(
            "Silver standard not built for %s; falling back to gold-only "
            "protection. Build via scripts/build_fusion_silver_standard.py "
            "to enable silver-augmented protection.",
            domain,
        )
        return (
            _load_fusion_protected_ids(domain),
            load_fusion_target_values(domain),
        )

    if surviving_record_ids is None:
        # Legacy silver behavior: silver targets for every cluster
        # member, regardless of K2 drops. Used by ablation pathways
        # that don't have post-K2 sources at hand.
        return (
            load_silver_protected_ids(domain),
            load_combined_target_values(domain),
        )

    # Production silver path (C13): silver targets only for intact
    # clusters. Gold targets unchanged.
    intact = load_intact_silver_clusters(domain, surviving_record_ids)
    return (
        load_silver_protected_ids(domain),
        load_combined_target_values_intact_only(domain, intact),
    )
