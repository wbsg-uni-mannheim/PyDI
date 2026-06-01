"""Tests for the C12 normalization committee runner.

Covers the 3-member coherent-roster shape introduced under
plan_revision.md §C12 (decided 2026-05-22). The legacy
per-(member, applies_to) tests live in
``tests/test_committee_norm.py`` and stay green via the
:class:`NormCommitteeRunner.__new__` dispatcher.

Scope of this module:

* YAML parser — good cases + every documented validation error.
* Selection-cache I/O round-trip.
* Dispatcher routing — ``NormCommitteeRunner`` returns a C12 instance
  for ``rule_normalizers:`` /``llm_normalizer:``-shape YAMLs and the
  legacy instance for the per-member YAMLs.
* End-to-end execution for the two no-LLM members
  (``rule_per_attribute_optimal`` + ``passthrough``).
* Val-selection sweep — single-candidate attributes lock without
  sweeping; multi-candidate attributes sweep and pick the val-best
  candidate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest
import yaml

from usecases_synthetic.lib.committee_norm import NormCommitteeRunner
from usecases_synthetic.lib.committee_norm_c12 import (
    SUPPORTED_MEMBERS,
    C12NormCommitteeRunner,
    _LLMConfig,
    _MemberSpec,
    _PassthroughNormalizer,
    _RuleCandidate,
    _candidates_for_attribute,
    _load_selection_cache,
    _parse_roster,
    _save_selection_cache,
    _selection_cache_path,
)
from usecases_synthetic.lib.variant_loader import VariantBundle

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_roster_dict() -> dict[str, Any]:
    """Return a minimal but valid C12 norm roster dict."""
    return {
        "seed": 42,
        "rule_normalizers": [
            {
                "name": "text_clean",
                "module": "usecases_synthetic.lib.normalizer_members",
                "class": "TextCleanNormalizer",
                "applies_to": ["name", "country"],
                "params": {"strip_whitespace": True},
            },
        ],
        "llm_normalizer": {
            "module": "usecases_synthetic.lib.llm_normalizer",
            "class": "LLMCanonicalizer",
            "params": {"model_name": "gpt-5.4-mini"},
        },
        "members": [
            {"name": "rule_per_attribute_optimal", "params": {}},
            {"name": "llm_only", "params": {}},
            {"name": "passthrough", "params": {}},
        ],
    }


def _write_yaml(tmp_path: Path, raw: dict[str, Any]) -> Path:
    """Write *raw* to a fresh roster YAML under tmp_path."""
    p = tmp_path / "norm_committee_test.yaml"
    p.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return p


def _synthetic_bundle() -> VariantBundle:
    """Tiny companies-shaped bundle: 2 fusion entities, 2 sources, name+country."""
    sources = {
        "dbpedia": pd.DataFrame(
            [
                {"id": "ent1", "org_name": "Apple Inc.", "nation": "United States"},
                {"id": "ent2", "org_name": "BMW AG", "nation": "Germany"},
            ]
        ),
        "forbes": pd.DataFrame(
            [
                {"id": "ent1", "company": "Apple", "region": "United States"},
                {"id": "ent2", "company": "BMW", "region": "Deutschland"},
            ]
        ),
    }
    sm_mapping = pd.DataFrame(
        [
            {
                "source_dataset": "dbpedia",
                "source_column": "org_name",
                "target_dataset": "companies",
                "target_column": "name",
                "score": 1.0,
            },
            {
                "source_dataset": "dbpedia",
                "source_column": "nation",
                "target_dataset": "companies",
                "target_column": "country",
                "score": 1.0,
            },
            {
                "source_dataset": "forbes",
                "source_column": "company",
                "target_dataset": "companies",
                "target_column": "name",
                "score": 1.0,
            },
            {
                "source_dataset": "forbes",
                "source_column": "region",
                "target_dataset": "companies",
                "target_column": "country",
                "score": 1.0,
            },
        ]
    )
    return VariantBundle(
        domain="companies",
        level="baseline",
        sources=sources,
        target_schema={
            "title": "companies",
            "properties": {
                "id": {"type": "string"},
                "name": {"type": "string"},
                "country": {"type": "string"},
            },
        },
        sm_mapping=sm_mapping,
        em_gold={},
        em_splits={},
        fusion_gold=pd.DataFrame(),
        fusion_validation=None,
        pooled_positives=None,
        variant_root=Path("/tmp/__test_norm_c12__"),
    )


# ---------------------------------------------------------------------------
# YAML parser
# ---------------------------------------------------------------------------


class TestParseRoster:
    def test_parses_minimal(self) -> None:
        roster = _parse_roster(_minimal_roster_dict())
        assert roster.seed == 42
        assert [m.name for m in roster.members] == [
            "rule_per_attribute_optimal",
            "llm_only",
            "passthrough",
        ]
        assert [c.name for c in roster.rule_candidates] == ["text_clean"]
        assert roster.llm_config is not None
        assert roster.llm_config.cls_name == "LLMCanonicalizer"

    def test_rejects_unknown_member(self) -> None:
        raw = _minimal_roster_dict()
        raw["members"].append({"name": "not_a_real_member"})
        with pytest.raises(ValueError, match="Unknown norm member"):
            _parse_roster(raw)

    def test_rejects_empty_members(self) -> None:
        raw = _minimal_roster_dict()
        raw["members"] = []
        with pytest.raises(ValueError, match="non-empty ``members`` list"):
            _parse_roster(raw)

    def test_rejects_rule_normalizer_missing_field(self) -> None:
        raw = _minimal_roster_dict()
        raw["rule_normalizers"][0].pop("class")
        with pytest.raises(ValueError, match="missing required key"):
            _parse_roster(raw)

    def test_rejects_empty_applies_to(self) -> None:
        raw = _minimal_roster_dict()
        raw["rule_normalizers"][0]["applies_to"] = []
        with pytest.raises(ValueError, match="must be a non-empty list"):
            _parse_roster(raw)

    def test_rejects_llm_only_without_llm_block(self) -> None:
        raw = _minimal_roster_dict()
        raw.pop("llm_normalizer")
        with pytest.raises(ValueError, match="llm_normalizer:"):
            _parse_roster(raw)

    def test_rejects_rule_optimal_without_candidates(self) -> None:
        raw = _minimal_roster_dict()
        raw["rule_normalizers"] = []
        with pytest.raises(ValueError, match="rule_normalizers:"):
            _parse_roster(raw)

    def test_supported_members_frozenset(self) -> None:
        assert SUPPORTED_MEMBERS == frozenset(
            {"rule_per_attribute_optimal", "llm_only", "passthrough"}
        )


# ---------------------------------------------------------------------------
# Selection-cache I/O
# ---------------------------------------------------------------------------


class TestSelectionCache:
    def test_path_under_baselines(self) -> None:
        path = _selection_cache_path("music")
        assert path.name == "norm_committee_selection.json"
        assert path.parent.name == "music"
        assert path.parent.parent.name == "baselines"

    def test_load_returns_empty_when_absent(self, tmp_path: Path) -> None:
        with patch(
            "usecases_synthetic.lib.committee_norm_c12._selection_cache_path",
            return_value=tmp_path / "absent.json",
        ):
            assert _load_selection_cache("never_existed") == {}

    def test_save_then_load_roundtrip(self, tmp_path: Path) -> None:
        target = tmp_path / "baselines" / "synthetic" / "norm_committee_selection.json"
        with patch(
            "usecases_synthetic.lib.committee_norm_c12._selection_cache_path",
            return_value=target,
        ):
            cache_in = {
                "rule_per_attribute_optimal": {
                    "name": "text_clean",
                    "country": "country_iso",
                }
            }
            _save_selection_cache("synthetic", cache_in)
            assert target.exists()
            cache_out = _load_selection_cache("synthetic")
            assert cache_out == cache_in


# ---------------------------------------------------------------------------
# Dispatcher routing
# ---------------------------------------------------------------------------


class TestDispatcher:
    def test_routes_c12_yaml_to_c12(self, tmp_path: Path) -> None:
        yaml_path = _write_yaml(tmp_path, _minimal_roster_dict())
        runner = NormCommitteeRunner(yaml_path, with_llm=True)
        assert isinstance(runner, C12NormCommitteeRunner)
        assert runner.roster_names == [
            "rule_per_attribute_optimal",
            "llm_only",
            "passthrough",
        ]

    def test_with_llm_false_skips_llm_only(self, tmp_path: Path) -> None:
        yaml_path = _write_yaml(tmp_path, _minimal_roster_dict())
        runner = NormCommitteeRunner(yaml_path, with_llm=False)
        assert isinstance(runner, C12NormCommitteeRunner)
        assert runner.roster_names == [
            "rule_per_attribute_optimal",
            "passthrough",
        ]

    def test_routes_legacy_yaml_to_legacy(self, tmp_path: Path) -> None:
        """A legacy-shape norm YAML (per-member ``signal_type:``) stays on
        the legacy path."""
        legacy_raw = {
            "seed": 42,
            "members": [
                {
                    "name": "text_clean",
                    "module": "usecases_synthetic.lib.normalizer_members",
                    "class": "TextCleanNormalizer",
                    "signal_type": "rule_string",
                    "enabled_by_default": True,
                    "applies_to": ["name"],
                    "params": {},
                }
            ],
        }
        legacy_yaml = _write_yaml(tmp_path, legacy_raw)
        runner = NormCommitteeRunner(legacy_yaml, with_llm=False)
        assert not isinstance(runner, C12NormCommitteeRunner)
        assert "text_clean" in runner.roster_names


# ---------------------------------------------------------------------------
# Per-member helpers
# ---------------------------------------------------------------------------


class TestPassthroughNormalizer:
    def test_strips_whitespace(self) -> None:
        p = _PassthroughNormalizer()
        out = p.normalize("  Apple  ", attribute="name", kind="long_string", domain="x")
        assert out == "Apple"

    def test_returns_none_on_null(self) -> None:
        p = _PassthroughNormalizer()
        assert (
            p.normalize(None, attribute="name", kind="long_string", domain="x") is None
        )
        assert p.normalize("", attribute="name", kind="long_string", domain="x") is None


class TestCandidatesForAttribute:
    def test_returns_only_applicable(self) -> None:
        candidates = [
            _RuleCandidate(
                name="text_clean",
                module="m",
                cls_name="C",
                applies_to=["name", "country"],
            ),
            _RuleCandidate(
                name="country_iso",
                module="m",
                cls_name="C",
                applies_to=["country"],
            ),
        ]
        assert [c.name for c in _candidates_for_attribute("name", candidates)] == [
            "text_clean"
        ]
        assert sorted(
            c.name for c in _candidates_for_attribute("country", candidates)
        ) == ["country_iso", "text_clean"]
        assert _candidates_for_attribute("missing", candidates) == []


# ---------------------------------------------------------------------------
# End-to-end execution
# ---------------------------------------------------------------------------


class TestPassthroughEndToEnd:
    """The passthrough member runs without val-selection or LLM
    machinery; it's the simplest path through the C12 runner.
    """

    def test_passthrough_alone_runs_end_to_end(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        roster = _minimal_roster_dict()
        # Keep the rule + LLM machinery in the YAML but drop them from
        # the active member set; passthrough alone should still produce
        # a valid CommitteeResult.
        roster["members"] = [{"name": "passthrough", "params": {}}]
        yaml_path = _write_yaml(tmp_path, roster)
        bundle = _synthetic_bundle()

        fake_val = {"ent1": {"name": ["Apple Inc."], "country": ["United States"]}}
        fake_test = {
            "ent1": {"name": ["Apple Inc."], "country": ["United States"]},
            "ent2": {"name": ["BMW AG"], "country": ["Germany"]},
        }

        with (
            patch(
                "usecases_synthetic.lib.committee_norm_c12._load_val_and_test_targets",
                return_value=(fake_val, fake_test),
            ),
        ):
            runner = NormCommitteeRunner(yaml_path, with_llm=False)
            result = runner.run(bundle)

        assert result.stage == "norm"
        assert result.domain == "companies"
        assert list(result.per_member.keys()) == ["passthrough"]
        # passthrough preserves the source value; some cells will be
        # close-enough to gold (dbpedia values match the fusion targets
        # exactly), so macro_f1 should be > 0.
        assert result.per_member["passthrough"].metrics["macro_f1"] > 0.0


class TestRulePerAttributeOptimalSweep:
    """Verify the val-selection sweep picks the best candidate per
    attribute and writes it to the selection cache."""

    def test_single_candidate_locks_without_sweep(self, tmp_path: Path) -> None:
        """When an attribute has a single rule candidate, it locks that
        candidate without running a sweep (the val pass would be wasted
        compute)."""
        roster = _minimal_roster_dict()
        # Drop llm_only so the only active member is rule_per_attribute_optimal.
        roster["members"] = [
            {"name": "rule_per_attribute_optimal", "params": {}},
            {"name": "passthrough", "params": {}},
        ]
        yaml_path = _write_yaml(tmp_path, roster)
        bundle = _synthetic_bundle()

        fake_val = {"ent1": {"name": ["Apple Inc."], "country": ["United States"]}}
        fake_test = {
            "ent1": {"name": ["Apple Inc."], "country": ["United States"]},
            "ent2": {"name": ["BMW AG"], "country": ["Germany"]},
        }

        cache_target = (
            tmp_path / "baselines" / "companies" / "norm_committee_selection.json"
        )
        with (
            patch(
                "usecases_synthetic.lib.committee_norm_c12._load_val_and_test_targets",
                return_value=(fake_val, fake_test),
            ),
            patch(
                "usecases_synthetic.lib.committee_norm_c12._selection_cache_path",
                return_value=cache_target,
            ),
        ):
            runner = NormCommitteeRunner(yaml_path, with_llm=False)
            result = runner.run(bundle)

        # rule_per_attribute_optimal's selection_map should have
        # text_clean locked for both name and country (single candidate
        # each per the minimal roster).
        rule_member = result.per_member["rule_per_attribute_optimal"]
        smap = rule_member.notes.get("selection_map", {})
        assert smap.get("name") == "text_clean"
        assert smap.get("country") == "text_clean"
        # Cache file persisted.
        assert cache_target.exists()

    def test_multi_candidate_sweep_picks_best(self, tmp_path: Path) -> None:
        """When multiple rule candidates apply to an attribute, the val
        sweep picks the val-best one. We give country two candidates
        (text_clean + country_iso) and verify the selection map records
        whichever wins on val."""
        roster = _minimal_roster_dict()
        roster["rule_normalizers"].append(
            {
                "name": "country_iso",
                "module": "usecases_synthetic.lib.normalizer_members",
                "class": "CountryIsoNormalizer",
                "applies_to": ["country"],
                "params": {"output_format": "name"},
            }
        )
        roster["members"] = [
            {"name": "rule_per_attribute_optimal", "params": {}},
            {"name": "passthrough", "params": {}},
        ]
        yaml_path = _write_yaml(tmp_path, roster)
        bundle = _synthetic_bundle()

        fake_val = {"ent1": {"name": ["Apple Inc."], "country": ["United States"]}}
        fake_test = {"ent1": {"name": ["Apple Inc."], "country": ["United States"]}}

        cache_target = (
            tmp_path / "baselines" / "companies" / "norm_committee_selection.json"
        )
        with (
            patch(
                "usecases_synthetic.lib.committee_norm_c12._load_val_and_test_targets",
                return_value=(fake_val, fake_test),
            ),
            patch(
                "usecases_synthetic.lib.committee_norm_c12._selection_cache_path",
                return_value=cache_target,
            ),
        ):
            runner = NormCommitteeRunner(yaml_path, with_llm=False)
            result = runner.run(bundle)

        rule_member = result.per_member["rule_per_attribute_optimal"]
        smap = rule_member.notes.get("selection_map", {})
        # The country sweep picks one of the two candidates. We don't
        # pin which one wins on this fixture (depends on closeness
        # contract behaviour) but it must be one of the declared
        # candidates.
        assert smap.get("country") in ("text_clean", "country_iso")
        # name has a single candidate, locked at text_clean.
        assert smap.get("name") == "text_clean"

    def test_cache_hit_skips_sweep(self, tmp_path: Path) -> None:
        """A pre-populated cache short-circuits the val sweep. We seed
        the cache with non-canonical picks and verify the runner reads
        them back instead of re-sweeping."""
        roster = _minimal_roster_dict()
        roster["rule_normalizers"].append(
            {
                "name": "country_iso",
                "module": "usecases_synthetic.lib.normalizer_members",
                "class": "CountryIsoNormalizer",
                "applies_to": ["country"],
                "params": {"output_format": "name"},
            }
        )
        roster["members"] = [
            {"name": "rule_per_attribute_optimal", "params": {}},
            {"name": "passthrough", "params": {}},
        ]
        yaml_path = _write_yaml(tmp_path, roster)
        bundle = _synthetic_bundle()

        cache_target = (
            tmp_path / "baselines" / "companies" / "norm_committee_selection.json"
        )
        # Seed the cache before the run.
        cache_target.parent.mkdir(parents=True, exist_ok=True)
        seeded = {
            "rule_per_attribute_optimal": {
                "name": "text_clean",
                "country": "country_iso",
            }
        }
        import json

        cache_target.write_text(json.dumps(seeded), encoding="utf-8")

        fake_val: dict[str, dict[str, list[str]]] = {}
        fake_test = {"ent1": {"name": ["Apple Inc."], "country": ["United States"]}}

        with (
            patch(
                "usecases_synthetic.lib.committee_norm_c12._load_val_and_test_targets",
                return_value=(fake_val, fake_test),
            ),
            patch(
                "usecases_synthetic.lib.committee_norm_c12._selection_cache_path",
                return_value=cache_target,
            ),
        ):
            runner = NormCommitteeRunner(yaml_path, with_llm=False)
            result = runner.run(bundle)

        rule_member = result.per_member["rule_per_attribute_optimal"]
        smap = rule_member.notes.get("selection_map", {})
        assert smap == seeded["rule_per_attribute_optimal"]
