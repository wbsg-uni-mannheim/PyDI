"""Tests for the silver-augmented protection-target loader (plan §4b).

Covers the per-domain merge of gold + silver into a single
``{member_id: {attribute: [value]}}`` dict, and the regression invariant
that silver-source protection rejects mutations the gold-only path
would have accepted.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from usecases_synthetic.lib.fusion_silver_targets import (
    PROTECTION_SOURCES,
    load_combined_target_values,
    load_combined_target_values_intact_only,
    load_intact_silver_clusters,
    load_silver_cluster_targets,
    load_silver_member_to_cluster,
    load_silver_protected_ids,
    resolve_protection_sources,
    silver_standard_available,
)
from usecases_synthetic.lib.protection import (
    ToleranceSpec,
    cell_has_close_survivor,
)

# ---------------------------------------------------------------------------
# Public dispatcher contract
# ---------------------------------------------------------------------------


class TestProtectionSourceValidation:
    def test_invalid_source_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown protection_source"):
            resolve_protection_sources("music", "platinum")

    def test_valid_sources_constant(self) -> None:
        assert set(PROTECTION_SOURCES) == {"gold", "silver"}


# ---------------------------------------------------------------------------
# End-to-end: existing music silver standard
# ---------------------------------------------------------------------------


_MUSIC_INPUTS_PRESENT = (
    Path(__file__).resolve().parents[1]
    / "baselines"
    / "music"
    / "fusion_silver_standard.csv"
).exists()


_skip_if_no_music_silver = pytest.mark.skipif(
    not _MUSIC_INPUTS_PRESENT,
    reason="music silver standard not built (run scripts/build_fusion_silver_standard.py --domain music)",
)


@_skip_if_no_music_silver
class TestMusicSilverTargets:
    def test_silver_available(self) -> None:
        assert silver_standard_available("music")

    def test_silver_cluster_targets_shape(self) -> None:
        targets = load_silver_cluster_targets("music")
        # Music has ~4280 clusters with at least one non-empty target.
        assert len(targets) > 1000
        # Spot-check shape: dict[cluster_id, dict[attribute, list[value]]]
        sample_cluster = next(iter(targets.values()))
        assert isinstance(sample_cluster, dict)
        sample_attr_values = next(iter(sample_cluster.values()))
        assert isinstance(sample_attr_values, list)
        assert all(isinstance(v, str) for v in sample_attr_values)

    def test_silver_member_to_cluster_expansion(self) -> None:
        m2c = load_silver_member_to_cluster("music")
        # Music has 12k+ member ids (sum across all clusters of ~3 members each).
        assert len(m2c) > 10000
        # Every value is a cluster id; every cluster has at least one member
        # that points back to itself.
        # (cluster_id IS one of the member ids — highest-trust anchor.)
        for cluster_id in set(m2c.values()):
            assert cluster_id in m2c, f"cluster {cluster_id!r} not self-referential"
            assert m2c[cluster_id] == cluster_id

    def test_silver_protected_ids_universe(self) -> None:
        gold_only_ids, _ = resolve_protection_sources("music", "gold")
        silver_ids = load_silver_protected_ids("music")
        # Silver universe must be strictly larger than gold.
        assert len(silver_ids) > len(gold_only_ids)
        assert gold_only_ids.issubset(silver_ids)

    def test_combined_targets_gold_wins(self) -> None:
        """For any (gold entity, gold attribute), combined returns gold."""
        gold_ids, gold_targets = resolve_protection_sources("music", "gold")
        _, combined = resolve_protection_sources("music", "silver")

        # Sample 25 gold (entity, attribute) pairs; each should be in
        # combined with the exact gold value.
        sample_count = 0
        for eid in list(gold_targets)[:25]:
            for attr, values in gold_targets[eid].items():
                assert eid in combined, f"gold entity {eid!r} dropped from combined"
                assert combined[eid][attr] == values, (
                    f"gold did not win for {eid!r} × {attr!r}: "
                    f"combined={combined[eid][attr]!r} gold={values!r}"
                )
                sample_count += 1
                if sample_count >= 25:
                    return

    def test_combined_targets_silver_fills_non_gold(self) -> None:
        """Pool members not in fusion val/test gold should get silver targets."""
        gold_ids, _ = resolve_protection_sources("music", "gold")
        silver_ids, combined = resolve_protection_sources("music", "silver")
        non_gold_member_ids = silver_ids - gold_ids
        # Pick a handful and verify they have non-empty silver targets.
        sample = list(non_gold_member_ids)[:25]
        assert sample, "expected silver-only member ids"
        fills_seen = 0
        for mid in sample:
            attrs = combined.get(mid, {})
            if attrs:
                fills_seen += 1
        assert (
            fills_seen > 0
        ), "no silver fills found across 25 sampled non-gold members"

    def test_resolve_dispatches_gold_vs_silver(self) -> None:
        gold_ids, gold_targets = resolve_protection_sources("music", "gold")
        silver_ids, silver_targets = resolve_protection_sources("music", "silver")
        # Gold-only result has the original universe; silver expands it.
        assert silver_ids > gold_ids
        assert len(silver_targets) > len(gold_targets)


# ---------------------------------------------------------------------------
# Silver-missing fallback
# ---------------------------------------------------------------------------


class TestSilverMissingFallback:
    def test_silver_unavailable_falls_back_to_gold(self) -> None:
        # Patch silver_standard_available to simulate a domain without
        # a built silver standard. resolve_protection_sources should
        # fall back to gold-only without raising.
        with patch(
            "usecases_synthetic.lib.fusion_silver_targets.silver_standard_available",
            return_value=False,
        ):
            ids, targets = resolve_protection_sources("music", "silver")
            # Should equal the gold-only universe (music gold = 200 entities).
            gold_ids, gold_targets = resolve_protection_sources("music", "gold")
            assert ids == gold_ids
            assert targets.keys() == gold_targets.keys()


# ---------------------------------------------------------------------------
# Regression: silver protection rejects what gold-only would accept
# ---------------------------------------------------------------------------


@_skip_if_no_music_silver
class TestSilverRegressionRejection:
    """Silver-source protection must reject mutations on members the
    gold-only path would have waved through.

    Concretely: a cluster member that is NOT in fusion val/test gold
    (so the gold-only check vacuously accepts every K1 paraphrase
    candidate that touches it) but IS in a silver cluster gets a
    silver target. A candidate value far from the silver target must
    be rejected.
    """

    def test_silver_only_member_rejects_far_candidate(self) -> None:
        gold_ids, gold_targets = resolve_protection_sources("music", "gold")
        silver_ids, silver_targets = resolve_protection_sources("music", "silver")

        # Find a member id that is in silver but NOT in gold, with a
        # non-empty target for the ``name`` attribute (long_string).
        non_gold_in_silver = silver_ids - gold_ids
        sample_mid: str | None = None
        sample_target: list[str] | None = None
        for mid in non_gold_in_silver:
            attrs = silver_targets.get(mid, {})
            if "name" in attrs and attrs["name"]:
                sample_mid = mid
                sample_target = attrs["name"]
                break
        assert (
            sample_mid is not None
        ), "expected at least one silver-only member with a name target"
        assert sample_target is not None

        # Long_string default tolerance: extended Jaccard >= 0.6.
        tolerance = ToleranceSpec(
            kind="long_string", threshold=0.6, inner_token_threshold=0.8
        )

        # Sanity check: a value equal to the silver target trivially passes.
        assert cell_has_close_survivor(
            sample_target, [sample_target[0]], tolerance
        ), "exact-match candidate should pass the close-survivor check"

        # A candidate string with no token overlap with the target must
        # fail. The silver-only member has no gold protection today; the
        # gold-only path would have skipped this candidate (vacuously
        # True) — silver-source protection actually evaluates it.
        far_candidate = "ZZ-COMPLETELY-DIFFERENT-XX-NEVER-MATCHES-YY"
        assert not cell_has_close_survivor(sample_target, [far_candidate], tolerance), (
            "silver-source protection failed to reject a candidate far from "
            f"the silver target {sample_target!r}"
        )

    def test_gold_overrides_silver_value_in_combined(self) -> None:
        """For an entity present in BOTH gold and silver, combined returns gold."""
        gold_ids, gold_targets = resolve_protection_sources("music", "gold")
        silver_targets = load_silver_cluster_targets("music")
        m2c = load_silver_member_to_cluster("music")
        _, combined = resolve_protection_sources("music", "silver")

        # Find a gold entity that is also in a silver cluster.
        overlap = None
        for gid in gold_targets:
            if gid in m2c:
                cluster_id = m2c[gid]
                # Need a (entity, attribute) where both gold and silver
                # have a value, AND they differ — otherwise gold-wins is
                # untestable.
                for attr, gold_vals in gold_targets[gid].items():
                    silver_vals_for_cluster = silver_targets.get(cluster_id, {})
                    silver_attr = silver_vals_for_cluster.get(attr)
                    if silver_attr and silver_attr != gold_vals:
                        overlap = (gid, attr, gold_vals, silver_attr)
                        break
                if overlap:
                    break

        if overlap is None:
            pytest.skip(
                "No (entity, attribute) where gold and silver disagree "
                "— the gold-wins rule is consistent with silver here"
            )

        gid, attr, gold_vals, silver_vals = overlap
        assert combined[gid][attr] == gold_vals, (
            f"gold-wins rule violated for {gid!r} × {attr!r}: "
            f"combined={combined[gid][attr]!r} gold={gold_vals!r} silver={silver_vals!r}"
        )


# ---------------------------------------------------------------------------
# C13: intact-cluster rule (silver targets only for clusters where every
# original member survives K2)
# ---------------------------------------------------------------------------


@_skip_if_no_music_silver
class TestIntactSilverClusters:
    """Regression for plan_revision.md §C13 intact-cluster semantics."""

    def test_all_members_survive_returns_all_clusters(self) -> None:
        """When the survivor set covers every silver-cluster member, all
        clusters are intact."""
        m2c = load_silver_member_to_cluster("music")
        all_members = set(m2c.keys())
        intact = load_intact_silver_clusters("music", all_members)
        all_clusters = set(m2c.values())
        assert intact == all_clusters

    def test_empty_survivors_no_clusters_intact(self) -> None:
        """No survivors → no intact clusters (silver was built with
        ``include_singletons=False`` so every cluster has ≥2 members,
        all of which would be "lost" under an empty survivor set)."""
        intact = load_intact_silver_clusters("music", set())
        assert intact == set()

    def test_dropping_one_member_breaks_only_its_cluster(self) -> None:
        """Removing a single member from the survivors should break
        exactly the cluster that member belonged to; every other
        cluster stays intact."""
        m2c = load_silver_member_to_cluster("music")
        # Pick the first member; remove it from survivors.
        dropped = next(iter(m2c.keys()))
        dropped_cluster = m2c[dropped]

        survivors = set(m2c.keys()) - {dropped}
        intact = load_intact_silver_clusters("music", survivors)

        assert dropped_cluster not in intact
        # Every other cluster should still be intact.
        all_other_clusters = set(m2c.values()) - {dropped_cluster}
        assert all_other_clusters.issubset(intact)

    def test_silver_missing_returns_empty(self) -> None:
        """When the domain has no silver standard, no clusters exist
        to be intact — return empty set without raising."""
        # Use a fake domain via monkeypatching silver_standard_available
        # to be explicit about intent.
        with patch(
            "usecases_synthetic.lib.fusion_silver_targets.silver_standard_available",
            return_value=False,
        ):
            intact = load_intact_silver_clusters("doesnotexist", {"x", "y"})
            assert intact == set()


@_skip_if_no_music_silver
class TestIntactOnlyCombinedTargets:
    """Combined target dict filters silver entries by intact-cluster rule."""

    def test_all_intact_matches_legacy_combined(self) -> None:
        """When every cluster is intact, the intact-only combined dict
        equals the legacy ``load_combined_target_values`` output."""
        m2c = load_silver_member_to_cluster("music")
        all_clusters = set(m2c.values())
        intact_only = load_combined_target_values_intact_only("music", all_clusters)
        legacy = load_combined_target_values("music")
        assert intact_only == legacy

    def test_empty_intact_set_falls_back_to_gold_only(self) -> None:
        """When no clusters are intact, silver contributes nothing — the
        intact-only combined dict matches the gold-only loader."""
        from usecases_synthetic.lib.protection import load_fusion_target_values

        intact_only = load_combined_target_values_intact_only("music", set())
        gold = load_fusion_target_values("music")
        assert intact_only == gold

    def test_broken_cluster_members_lose_silver_targets(self) -> None:
        """When a cluster is broken (not in intact_cluster_ids), every
        member of that cluster loses its silver targets — except cells
        where gold already authored a value (gold protection is
        unconditional)."""
        from usecases_synthetic.lib.protection import load_fusion_target_values

        m2c = load_silver_member_to_cluster("music")
        all_clusters = set(m2c.values())
        # Break one cluster by removing it from the intact set.
        broken_cluster = next(iter(all_clusters))
        intact = all_clusters - {broken_cluster}

        intact_only = load_combined_target_values_intact_only("music", intact)
        gold = load_fusion_target_values("music")

        broken_members = {m for m, c in m2c.items() if c == broken_cluster}
        for member in broken_members:
            if member in gold:
                # Gold-protected: should still have at least the gold attrs.
                for gold_attr, gold_vals in gold[member].items():
                    assert intact_only.get(member, {}).get(gold_attr) == gold_vals, (
                        f"gold target for broken-cluster member {member!r} × "
                        f"{gold_attr!r} was lost"
                    )
            else:
                # No gold, broken cluster → no targets at all.
                assert member not in intact_only or intact_only[member] == {}, (
                    f"broken-cluster, non-gold member {member!r} should have "
                    f"no targets; got {intact_only.get(member)!r}"
                )


@_skip_if_no_music_silver
class TestResolveProtectionSourcesWithSurvivingIds:
    """resolve_protection_sources(domain, 'silver', surviving) dispatches
    to the intact-only target dict (C13)."""

    def test_surviving_none_returns_legacy_silver_targets(self) -> None:
        """``surviving_record_ids=None`` keeps backward compat: returns
        the full silver-augmented combined target dict (every cluster's
        silver targets)."""
        ids_none, targets_none = resolve_protection_sources(
            "music", "silver", surviving_record_ids=None
        )
        ids_legacy, targets_legacy = resolve_protection_sources("music", "silver")
        assert ids_none == ids_legacy
        assert targets_none == targets_legacy

    def test_full_survivors_equals_legacy(self) -> None:
        """``surviving_record_ids = every cluster member`` produces the
        same target dict as the legacy all-silver path."""
        m2c = load_silver_member_to_cluster("music")
        all_members = set(m2c.keys())
        _, targets_intact = resolve_protection_sources(
            "music", "silver", surviving_record_ids=all_members
        )
        _, targets_legacy = resolve_protection_sources("music", "silver")
        assert targets_intact == targets_legacy

    def test_empty_survivors_drops_silver_only_members(self) -> None:
        """``surviving_record_ids = empty`` breaks every cluster →
        non-gold cluster members lose every target (silver was the only
        source for their cells); gold members keep their gold targets."""
        from usecases_synthetic.lib.protection import load_fusion_target_values

        _, targets_intact = resolve_protection_sources(
            "music", "silver", surviving_record_ids=set()
        )
        gold_targets = load_fusion_target_values("music")
        assert targets_intact == gold_targets


class TestResolveProtectionSourcesGoldUnaffectedByIntactGate:
    """The intact-cluster rule only applies under silver mode. Gold mode
    is unchanged by the new ``surviving_record_ids`` parameter."""

    def test_gold_mode_ignores_surviving_ids(self) -> None:
        ids_a, targets_a = resolve_protection_sources(
            "music", "gold", surviving_record_ids=None
        )
        ids_b, targets_b = resolve_protection_sources(
            "music", "gold", surviving_record_ids={"x", "y", "z"}
        )
        assert ids_a == ids_b
        assert targets_a == targets_b
