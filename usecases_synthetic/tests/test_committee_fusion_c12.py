"""Tests for the C12 fusion committee runner.

Covers the new coherent-member roster shape introduced under
plan_revision.md §C12 (decided 2026-05-22). The legacy
per-(attribute, strategy) tests live in
``tests/test_committee_fusion.py`` and stay green via the
:class:`FusionCommitteeRunner.__new__` dispatcher.

Scope of this module:

* YAML parser — good cases + every documented validation error.
* Selection-cache I/O round-trip.
* Per-member ``_selection_attrs_for_member`` (which attributes need
  val-best PyDI for each of the 9 members).
* Dispatcher routing — ``FusionCommitteeRunner`` returns a C12
  instance for ``members:``-shape YAMLs and the legacy instance for
  the per-(attribute, strategy) YAMLs.
* End-to-end execution for the two no-val-selection-needed members
  (``voting_only`` + ``prefer_higher_trust_only``) — they should run
  without a fusion validation set and produce a per-member
  macro_accuracy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from usecases_synthetic.lib.committee_fusion import FusionCommitteeRunner
from usecases_synthetic.lib.committee_fusion_c12 import (
    SUPPORTED_MEMBERS,
    C12FusionCommitteeRunner,
    _NATIVE_TYPES_BY_MEMBER,
    _load_selection_cache,
    _parse_roster,
    _save_selection_cache,
    _selection_attrs_for_member,
    _selection_cache_path,
)
from usecases_synthetic.lib.variant_loader import VariantBundle

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_roster_dict() -> dict[str, Any]:
    """Return a minimal but valid C12 roster dict for parser tests."""
    return {
        "seed": 42,
        "fused_id_column": "id",
        "gold_id_column": "id",
        "trust_scores": {"a": 2.0, "b": 1.0},
        "column_mapping": {"a": {}, "b": {}},
        "evaluation_functions": {"name": "exact_match", "revenue": "exact_match"},
        "attribute_types": {"name": "string", "revenue": "numeric"},
        "pydi_candidates": {
            "string": [
                {
                    "name": "voting",
                    "function": "voting",
                    "module": "PyDI.fusion.conflict_resolution.general",
                }
            ],
            "numeric": [
                {
                    "name": "median",
                    "function": "median",
                    "module": "PyDI.fusion.conflict_resolution.numeric",
                }
            ],
        },
        "members": [
            {"name": "voting_only", "params": {}},
            {"name": "prefer_higher_trust_only", "params": {}},
        ],
    }


def _write_yaml(tmp_path: Path, raw: dict[str, Any]) -> Path:
    """Write *raw* to a fresh roster YAML under tmp_path."""
    p = tmp_path / "fusion_committee_test.yaml"
    p.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return p


def _make_two_source_bundle(
    *,
    domain: str = "test_domain",
    level: str = "baseline",
) -> VariantBundle:
    """Build a minimal two-source bundle with 3 entities for fusion tests.

    Sources ``a`` and ``b`` carry the same 3 ids (mapped via
    correspondences). The gold uses source ``a``'s name + revenue
    values. With trust ``a=2 > b=1`` (in the roster), both
    ``voting_only`` (tied 1-vs-1, falls back to first non-null) and
    ``prefer_higher_trust_only`` (picks ``a``) should fuse to ``a``'s
    values, yielding accuracy 1.0 on ``a``-flavoured gold.
    """
    n = 3
    src_a = pd.DataFrame(
        {
            "id": [f"a_{i}" for i in range(n)],
            "name": [f"A Co {i}" for i in range(n)],
            "revenue": [float(100 * i) for i in range(n)],
        }
    )
    src_a.attrs["dataset_name"] = "a"
    src_b = pd.DataFrame(
        {
            "id": [f"b_{i}" for i in range(n)],
            "name": [f"B Co {i}" for i in range(n)],
            "revenue": [float(100 * i + 5) for i in range(n)],
        }
    )
    src_b.attrs["dataset_name"] = "b"

    correspondences = pd.DataFrame(
        [{"id1": f"a_{i}", "id2": f"b_{i}", "score": 1.0} for i in range(n)]
    )
    em_gold = {
        ("a", "b"): correspondences.assign(label=1),
    }
    fusion_gold = pd.DataFrame(
        {
            "id": [f"a_{i}" for i in range(n)],
            "name": [f"A Co {i}" for i in range(n)],
            "revenue": [float(100 * i) for i in range(n)],
        }
    )

    return VariantBundle(
        domain=domain,
        level=level,
        sources={"a": src_a, "b": src_b},
        target_schema={"properties": {"id": {}, "name": {}, "revenue": {}}},
        sm_mapping=None,
        em_gold=em_gold,
        em_splits={},
        fusion_gold=fusion_gold,
        fusion_validation=None,
        pooled_positives=None,
        variant_root=Path("/tmp/fusion_c12_test"),
    )


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------


class TestParseRoster:
    """Cover the C12 YAML parser's good case + every validation error."""

    def test_parses_minimal_roster(self) -> None:
        raw = _minimal_roster_dict()
        roster = _parse_roster(raw)
        assert [m.name for m in roster.members] == [
            "voting_only",
            "prefer_higher_trust_only",
        ]
        assert roster.attribute_types == {"name": "string", "revenue": "numeric"}
        assert "string" in roster.pydi_candidates_by_type
        assert "numeric" in roster.pydi_candidates_by_type
        assert roster.trust_scores == {"a": 2.0, "b": 1.0}

    def test_parses_all_nine_members(self) -> None:
        raw = _minimal_roster_dict()
        raw["members"] = [{"name": n, "params": {}} for n in sorted(SUPPORTED_MEMBERS)]
        roster = _parse_roster(raw)
        assert {m.name for m in roster.members} == SUPPORTED_MEMBERS

    def test_rejects_unknown_member(self) -> None:
        raw = _minimal_roster_dict()
        raw["members"] = [{"name": "made_up_member"}]
        with pytest.raises(ValueError, match="Unknown member"):
            _parse_roster(raw)

    def test_rejects_missing_attribute_types(self) -> None:
        raw = _minimal_roster_dict()
        del raw["attribute_types"]
        with pytest.raises(ValueError, match="attribute_types"):
            _parse_roster(raw)

    def test_rejects_invalid_attribute_type(self) -> None:
        raw = _minimal_roster_dict()
        raw["attribute_types"]["name"] = "bogus_type"
        with pytest.raises(ValueError, match="bogus_type"):
            _parse_roster(raw)

    def test_rejects_pydi_candidate_missing_keys(self) -> None:
        raw = _minimal_roster_dict()
        raw["pydi_candidates"]["string"][0] = {"name": "voting"}  # no function/module
        with pytest.raises(ValueError, match="missing required keys"):
            _parse_roster(raw)

    def test_rejects_empty_pydi_candidate_list(self) -> None:
        raw = _minimal_roster_dict()
        raw["pydi_candidates"]["string"] = []
        with pytest.raises(ValueError, match="empty"):
            _parse_roster(raw)

    def test_rejects_missing_pydi_candidates_for_used_type(self) -> None:
        raw = _minimal_roster_dict()
        # Remove the numeric candidate but keep ``revenue`` numeric.
        del raw["pydi_candidates"]["numeric"]
        with pytest.raises(ValueError, match="pydi_candidates is missing"):
            _parse_roster(raw)

    def test_rejects_empty_members(self) -> None:
        raw = _minimal_roster_dict()
        raw["members"] = []
        with pytest.raises(ValueError, match="members"):
            _parse_roster(raw)

    def test_parses_music_yaml(self) -> None:
        """The migrated music YAML is the canonical C12 reference."""
        repo_root = Path(__file__).resolve().parents[1]
        yaml_path = repo_root / "config" / "committees" / "fusion_committee_music.yaml"
        with yaml_path.open(encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        roster = _parse_roster(raw)
        assert {m.name for m in roster.members} == SUPPORTED_MEMBERS
        # Music has 8 attributes spanning every supported type.
        assert set(roster.attribute_types.values()) == {
            "string",
            "categorical",
            "date",
            "numeric",
            "list",
        }


# ---------------------------------------------------------------------------
# Selection-cache I/O tests
# ---------------------------------------------------------------------------


class TestSelectionCache:
    """Round-trip the per-domain val-selection cache."""

    def test_path_under_baselines(self) -> None:
        path = _selection_cache_path("music")
        assert path.name == "fusion_committee_selection.json"
        assert "baselines" in path.parts
        assert "music" in path.parts

    def test_load_returns_empty_when_absent(self, tmp_path: Path) -> None:
        # Point the loader at a non-existent domain.
        cache = _load_selection_cache("__no_such_domain__")
        assert cache == {}

    def test_save_then_load_roundtrip(self, tmp_path: Path, monkeypatch) -> None:
        """Save → load returns an equivalent dict.

        Patches the path-resolver to keep tests isolated from the real
        ``baselines/`` tree (no test should write into the canonical
        baselines directory).
        """
        domain = "fusion_c12_test_domain"
        target = tmp_path / "baselines" / domain / "fusion_committee_selection.json"

        from usecases_synthetic.lib import committee_fusion_c12 as mod

        def fake_path(d: str) -> Path:
            return tmp_path / "baselines" / d / "fusion_committee_selection.json"

        monkeypatch.setattr(mod, "_selection_cache_path", fake_path)

        payload = {
            "pydi_per_attribute_optimal": {
                "name": "voting",
                "revenue": "median",
            },
            "fusionquery_only": {
                "revenue": "median",
                "tracks": "union",
            },
        }
        _save_selection_cache(domain, payload)
        assert target.exists()
        loaded = _load_selection_cache(domain)
        assert loaded == payload


# ---------------------------------------------------------------------------
# Per-member selection-attribute calculation
# ---------------------------------------------------------------------------


class TestSelectionAttrsForMember:
    """Members with universal native sets need no val selection; TD
    members need it on non-native types; pydi_per_attribute_optimal needs
    it on every attribute."""

    @pytest.fixture
    def roster(self):  # type: ignore[no-untyped-def]
        raw = _minimal_roster_dict()
        return _parse_roster(raw)

    def test_voting_only_needs_nothing(self, roster) -> None:  # type: ignore[no-untyped-def]
        from usecases_synthetic.lib.committee_fusion_c12 import _MemberSpec

        assert (
            _selection_attrs_for_member(_MemberSpec(name="voting_only"), roster) == []
        )

    def test_prefer_higher_trust_only_needs_nothing(self, roster) -> None:  # type: ignore[no-untyped-def]
        from usecases_synthetic.lib.committee_fusion_c12 import _MemberSpec

        assert (
            _selection_attrs_for_member(
                _MemberSpec(name="prefer_higher_trust_only"), roster
            )
            == []
        )

    def test_llm_only_needs_nothing(self, roster) -> None:  # type: ignore[no-untyped-def]
        from usecases_synthetic.lib.committee_fusion_c12 import _MemberSpec

        assert _selection_attrs_for_member(_MemberSpec(name="llm_only"), roster) == []

    def test_accusim_only_needs_nothing(self, roster) -> None:  # type: ignore[no-untyped-def]
        from usecases_synthetic.lib.committee_fusion_c12 import _MemberSpec

        assert (
            _selection_attrs_for_member(_MemberSpec(name="accusim_only"), roster) == []
        )

    def test_pydi_per_attribute_optimal_needs_every_attribute(
        self, roster
    ) -> None:  # type: ignore[no-untyped-def]
        from usecases_synthetic.lib.committee_fusion_c12 import _MemberSpec

        attrs = _selection_attrs_for_member(
            _MemberSpec(name="pydi_per_attribute_optimal"), roster
        )
        assert set(attrs) == set(roster.attribute_types.keys())

    def test_fusionquery_only_skips_native_string(
        self, roster
    ) -> None:  # type: ignore[no-untyped-def]
        from usecases_synthetic.lib.committee_fusion_c12 import _MemberSpec

        # roster has ``name`` (string, native) and ``revenue`` (numeric, fallback).
        attrs = _selection_attrs_for_member(
            _MemberSpec(name="fusionquery_only"), roster
        )
        assert attrs == ["revenue"]

    def test_ltm_only_skips_list_and_string_keeps_numeric(self) -> None:
        # LTM is native on string/categorical/date + list; numeric is fallback.
        raw = _minimal_roster_dict()
        raw["attribute_types"] = {
            "name": "string",
            "revenue": "numeric",
            "tags": "list",
        }
        raw["pydi_candidates"]["list"] = [
            {
                "name": "union",
                "function": "union",
                "module": "PyDI.fusion.conflict_resolution.list",
            }
        ]
        roster = _parse_roster(raw)
        from usecases_synthetic.lib.committee_fusion_c12 import _MemberSpec

        attrs = _selection_attrs_for_member(_MemberSpec(name="ltm_only"), roster)
        assert attrs == ["revenue"]


# ---------------------------------------------------------------------------
# Dispatcher routing
# ---------------------------------------------------------------------------


class TestDispatcher:
    """``FusionCommitteeRunner(path)`` routes by YAML shape."""

    def test_routes_members_yaml_to_c12(self, tmp_path: Path) -> None:
        yaml_path = _write_yaml(tmp_path, _minimal_roster_dict())
        runner = FusionCommitteeRunner(yaml_path)
        assert isinstance(runner, C12FusionCommitteeRunner)
        assert runner.roster_names == ["voting_only", "prefer_higher_trust_only"]

    def test_routes_legacy_yaml_to_legacy(self, tmp_path: Path) -> None:
        """A legacy-shape YAML (per-attribute ``strategies:`` blocks)
        stays on the legacy path. All four shipped per-domain YAMLs are
        C12 post-2026-05-25, so this test synthesises a minimal legacy
        roster to keep the dispatcher's negative branch covered."""
        legacy_raw = {
            "seed": 42,
            "fused_id_column": "id",
            "gold_id_column": "id",
            "trust_scores": {"a": 2.0, "b": 1.0},
            "column_mapping": {"a": {}, "b": {}},
            "evaluation_functions": {"name": "exact_match"},
            "attributes": {
                "name": {
                    "attribute_class": "primary",
                    "strategies": [
                        {
                            "name": "voting",
                            "function": "voting",
                            "module": "PyDI.fusion.conflict_resolution.general",
                            "strategy_type": "cell_local",
                            "params": {},
                        }
                    ],
                }
            },
        }
        legacy_yaml = _write_yaml(tmp_path, legacy_raw)
        runner = FusionCommitteeRunner(legacy_yaml)
        # Legacy runner is the base class, not the C12 subclass.
        assert not isinstance(runner, C12FusionCommitteeRunner)
        # Legacy uses per-(attr, strat) names like "name_voting".
        assert any("_" in n for n in runner.roster_names)


# ---------------------------------------------------------------------------
# End-to-end execution (no val-selection needed)
# ---------------------------------------------------------------------------


class TestSimpleMemberExecution:
    """End-to-end runs for the two no-val-selection members.

    These members do not need ``fusion_validation``; they can score
    every level (including baseline) without a val sweep. Both should
    produce a fused DataFrame and a macro_accuracy > 0 on the
    trust-aligned fixture.
    """

    def test_voting_and_prefer_higher_trust_run_end_to_end(
        self, tmp_path: Path
    ) -> None:
        # Build a roster with only the two no-val-selection members so
        # the test doesn't accidentally exercise val selection.
        raw = _minimal_roster_dict()
        raw["members"] = [
            {"name": "voting_only"},
            {"name": "prefer_higher_trust_only"},
        ]
        # Trust scores: a > b so prefer_higher_trust picks a.
        raw["trust_scores"] = {"a": 2.0, "b": 1.0}
        yaml_path = _write_yaml(tmp_path, raw)

        runner = FusionCommitteeRunner(yaml_path)
        bundle = _make_two_source_bundle()
        result = runner.run(bundle)

        assert set(result.per_member.keys()) == {
            "voting_only",
            "prefer_higher_trust_only",
        }
        for member_name, member in result.per_member.items():
            assert "macro_accuracy" in member.metrics, member_name
            assert member.metrics["macro_accuracy"] >= 0.0
            assert "f1" in member.metrics  # promoted from macro_accuracy
            # Selection map should be empty for these members.
            assert member.notes.get("selection_map") == {}

        # prefer_higher_trust should pick source ``a`` cleanly →
        # name + revenue both equal ``a``'s gold values → 1.0.
        pht = result.per_member["prefer_higher_trust_only"]
        assert pht.metrics["macro_accuracy"] == pytest.approx(1.0)

    def test_unknown_member_in_yaml_raises(self, tmp_path: Path) -> None:
        raw = _minimal_roster_dict()
        raw["members"].append({"name": "this_member_does_not_exist"})
        yaml_path = _write_yaml(tmp_path, raw)
        with pytest.raises(ValueError, match="Unknown member"):
            FusionCommitteeRunner(yaml_path)


# ---------------------------------------------------------------------------
# Native-type table consistency
# ---------------------------------------------------------------------------


class TestNativeTypesTable:
    """The :data:`_NATIVE_TYPES_BY_MEMBER` table encodes the C12 spec.

    These tests pin the spec so a refactor of the table can't silently
    flip a member's competence profile.
    """

    def test_pydi_per_attribute_optimal_is_empty(self) -> None:
        # Always selects PyDI per attribute; no native methods.
        assert _NATIVE_TYPES_BY_MEMBER["pydi_per_attribute_optimal"] == frozenset()

    def test_llm_only_is_universal(self) -> None:
        # Prompt v2: LLM handles every type natively.
        assert _NATIVE_TYPES_BY_MEMBER["llm_only"] == frozenset(
            {"string", "categorical", "date", "numeric", "list"}
        )

    def test_accusim_only_is_universal(self) -> None:
        # Type-aware similarity hook covers everything.
        assert _NATIVE_TYPES_BY_MEMBER["accusim_only"] == frozenset(
            {"string", "categorical", "date", "numeric", "list"}
        )

    def test_voting_only_and_pht_only_are_universal(self) -> None:
        for member in ("voting_only", "prefer_higher_trust_only"):
            assert _NATIVE_TYPES_BY_MEMBER[member] == frozenset(
                {"string", "categorical", "date", "numeric", "list"}
            ), member

    def test_truthfinder_fusionquery_casefusion_skip_numeric_and_list(self) -> None:
        for member in ("truthfinder_only", "fusionquery_only", "casefusion_only"):
            native = _NATIVE_TYPES_BY_MEMBER[member]
            assert "numeric" not in native, member
            assert "list" not in native, member
            assert native == frozenset({"string", "categorical", "date"}), member

    def test_ltm_only_covers_list_but_not_numeric(self) -> None:
        native = _NATIVE_TYPES_BY_MEMBER["ltm_only"]
        assert "list" in native
        assert "numeric" not in native
        assert native == frozenset({"string", "categorical", "date", "list"})
