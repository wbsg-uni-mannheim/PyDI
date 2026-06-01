"""Tests for protection set construction."""

from __future__ import annotations

from usecases_synthetic.lib.protection import (
    build_drop_corner_protection_set,
    build_expanded_positives,
    is_protected,
)


class TestProtectionSet:
    """Tests for expanded_positives and is_protected."""

    def test_is_protected_true(self, mock_protection_set: set[str]) -> None:
        assert is_protected(
            "http://dbpedia.org/resource/Company_0", mock_protection_set
        )
        assert is_protected("fullcontact_0", mock_protection_set)

    def test_is_protected_false(self, mock_protection_set: set[str]) -> None:
        assert not is_protected("unknown_entity_99", mock_protection_set)
        assert not is_protected("fullcontact_999", mock_protection_set)

    def test_build_expanded_positives_companies(self) -> None:
        """expanded_positives for companies contains pooled + EM gold IDs.

        This test requires the actual data files to be present (LFS).
        """
        positives = build_expanded_positives("companies")

        # Must contain pooled positive IDs (2803 pairs → many entity IDs)
        # At minimum we expect some dbpedia and forbes IDs
        dbpedia_ids = {eid for eid in positives if "dbpedia.org" in eid}
        forbes_ids = {eid for eid in positives if "forbes.com" in eid}

        assert len(dbpedia_ids) > 0, "No dbpedia IDs in expanded_positives"
        assert len(forbes_ids) > 0, "No forbes IDs in expanded_positives"

        # Total should be substantial (EM gold + fusion gold + pool)
        assert (
            len(positives) > 100
        ), f"expanded_positives only has {len(positives)} IDs — expected >100"

    def test_build_expanded_positives_includes_pool(self) -> None:
        """Spot-check: a known pooled pair's IDs must appear."""
        positives = build_expanded_positives("companies")

        # From the pooled_positives.csv first data row
        assert "http://dbpedia.org/resource/A2A" in positives
        assert "http://www.forbes.com/companies/a2a/" in positives


class TestDropCornerProtectionSet:
    """Regression for the 2026-05-28 K2 drop-corner zero-drop bug.

    On ``pool_quality: live`` domains (products) BOTH pool and EM gold
    are coextensive with the full record set, so any protection scheme
    that includes EM gold leaves zero droppable entities. The fix
    narrows the drop-corner protection set to fusion val/test only
    under default ``gold``; step 4c / C11's EM-regen handles dropped
    EM-gold members downstream by pruning Set 1 and corner-mining
    Set 2 from the surviving pool. ``silver`` widens back to include
    pool (matches C9 silver-standard semantics).
    """

    def test_gold_protection_is_fusion_only(self) -> None:
        """Under ``protection_source="gold"``, the drop-corner set is
        exactly the fusion val/test gold — narrower than
        ``build_expanded_positives`` which also includes EM gold +
        pool. On ``pool_quality: live`` domains (products), the broader
        set was coextensive with the full pool and made drop-corner a
        noop; the narrowing is what unblocks the operator."""
        from usecases_synthetic.lib.protection import _load_fusion_protected_ids

        gold = build_drop_corner_protection_set("companies", protection_source="gold")
        fusion_only = _load_fusion_protected_ids("companies")
        assert gold == fusion_only, (
            "drop-corner gold protection must equal fusion val/test only "
            "(no EM gold, no pool)"
        )

    def test_gold_protection_is_strict_subset_of_expanded_positives(self) -> None:
        """``gold`` (fusion-only) must be a strict subset of
        ``expanded_positives`` (em ∪ fusion ∪ pool). Confirms the
        narrowing actually drops entities from the protection set,
        which is the whole point of the helper."""
        full = build_expanded_positives("companies")
        gold = build_drop_corner_protection_set("companies", protection_source="gold")
        assert gold.issubset(full)
        assert len(gold) < len(full), (
            f"gold set ({len(gold)}) must be strictly smaller than "
            f"expanded_positives ({len(full)}) — otherwise pool + EM gold "
            "aren't being excluded."
        )

    def test_silver_protection_includes_pool(self) -> None:
        """Under ``protection_source="silver"``, the drop-corner set
        widens to fusion ∪ pool (C9 silver-standard semantics: every
        pool-cluster member is fusion-recoverable, therefore
        protected). EM gold is still NOT included on its own — it
        only contributes via overlap with pool members."""
        from usecases_synthetic.lib.protection import (
            _load_fusion_protected_ids,
            _load_pooled_positive_ids,
        )

        silver = build_drop_corner_protection_set(
            "companies", protection_source="silver"
        )
        expected = _load_fusion_protected_ids("companies") | _load_pooled_positive_ids(
            "companies"
        )
        assert silver == expected

    def test_em_gold_members_droppable_under_gold(self) -> None:
        """EM gold positive ids that are NOT also in fusion val/test
        gold must NOT be in the gold protection set — i.e. drop-corner
        is allowed to drop them. C11 EM-regen rebuilds the EM splits
        from the surviving pool after the drops land."""
        from usecases_synthetic.lib.protection import (
            _load_em_gold_ids,
            _load_fusion_protected_ids,
        )

        em_only = _load_em_gold_ids("companies") - _load_fusion_protected_ids(
            "companies"
        )
        gold = build_drop_corner_protection_set("companies", protection_source="gold")
        # EM-only ids (not in fusion val/test) must be absent from gold.
        # If the test data has no em-only ids, this is vacuously true —
        # but for companies this set should be nonempty (companies has
        # an EM gold of pairs that doesn't fully overlap fusion val/test).
        intersection = em_only & gold
        assert intersection == set(), (
            f"{len(intersection)} EM-gold-only ids leaked into gold protection — "
            "they should be droppable under the C11 EM-regen contract."
        )

    def test_gold_protection_default_when_param_omitted(self) -> None:
        """The default protection_source value is 'gold' — omitting the
        param must produce the narrow set, not the wide one."""
        default = build_drop_corner_protection_set("companies")
        gold = build_drop_corner_protection_set("companies", protection_source="gold")
        assert default == gold
