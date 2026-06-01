"""Tests for the Normalization committee runner + scoring.

Exercises the runner against a synthetic ``VariantBundle`` so we can
verify per-attribute F1 semantics deterministically without a real
domain dataset. Real-data smoke runs are covered by
``measure_baseline.py --domain <d> --stages norm`` per the R6.1 row of
``plan_s1_scale.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

from usecases_synthetic.lib.committee_norm import (
    NormCommitteeRunner,
    _build_source_attribute_index,
)
from usecases_synthetic.lib.committee_norm_scoring import (
    AttributeScore,
    MemberPerAttributeScores,
    score_cell,
)
from usecases_synthetic.lib.normalizer_members import (
    CountryIsoNormalizer,
    DateIsoNormalizer,
    NumberLocaleNormalizer,
    TaxonomyLookupNormalizer,
    TextCleanNormalizer,
)
from usecases_synthetic.lib.protection import (
    ToleranceSpec,
    fusion_cell_tolerance,
)
from usecases_synthetic.lib.variant_loader import VariantBundle

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMITTEE_DIR = REPO_ROOT / "usecases_synthetic" / "config" / "committees"


# ---------------------------------------------------------------------------
# AttributeScore + MemberPerAttributeScores
# ---------------------------------------------------------------------------


class TestAttributeScore:
    def test_empty(self) -> None:
        s = AttributeScore()
        assert s.precision == 0.0
        assert s.recall == 0.0
        assert s.f1 == 0.0

    def test_perfect(self) -> None:
        s = AttributeScore(correct=10, wrong=0, abstained=0, total=10)
        assert s.precision == 1.0
        assert s.recall == 1.0
        assert s.f1 == 1.0

    def test_half_recall_full_precision(self) -> None:
        # Output is correct on every emit, but abstained on half of cells.
        s = AttributeScore(correct=5, wrong=0, abstained=5, total=10)
        assert s.precision == 1.0
        assert s.recall == 0.5
        # F1 = 2*1*0.5 / (1+0.5) ≈ 0.667
        assert s.f1 == pytest.approx(2 / 3, rel=1e-6)

    def test_some_wrong(self) -> None:
        s = AttributeScore(correct=4, wrong=2, abstained=4, total=10)
        assert s.precision == pytest.approx(4 / 6)
        assert s.recall == 0.4


class TestScoreCell:
    def test_abstention(self) -> None:
        tol = fusion_cell_tolerance("companies", "country")
        correct, abstained = score_cell(None, ["United States"], tol)
        assert correct is False
        assert abstained is True

    def test_close_match(self) -> None:
        tol = fusion_cell_tolerance("companies", "country")
        correct, abstained = score_cell("United States", ["United States"], tol)
        assert correct is True
        assert abstained is False

    def test_wrong_output(self) -> None:
        tol = fusion_cell_tolerance("companies", "country")
        correct, abstained = score_cell("Germany", ["United States"], tol)
        assert correct is False
        assert abstained is False

    def test_empty_targets(self) -> None:
        # No ground truth → treated as abstention (cannot score).
        tol = fusion_cell_tolerance("companies", "country")
        correct, abstained = score_cell("anything", [], tol)
        assert correct is False
        assert abstained is True


# ---------------------------------------------------------------------------
# Helper: build_source_attribute_index
# ---------------------------------------------------------------------------


class TestBuildSourceAttributeIndex:
    def test_empty_mapping(self) -> None:
        empty = pd.DataFrame(
            columns=[
                "source_dataset",
                "source_column",
                "target_dataset",
                "target_column",
            ]
        )
        idx = _build_source_attribute_index(empty, knob_08_renames=None)
        assert idx == {}

    def test_basic_index(self) -> None:
        sm = pd.DataFrame(
            [
                {
                    "source_dataset": "dbpedia",
                    "source_column": "org_name",
                    "target_dataset": "companies",
                    "target_column": "name",
                },
                {
                    "source_dataset": "forbes",
                    "source_column": "company",
                    "target_dataset": "companies",
                    "target_column": "name",
                },
            ]
        )
        idx = _build_source_attribute_index(sm, knob_08_renames=None)
        assert idx == {
            ("dbpedia", "name"): ["org_name"],
            ("forbes", "name"): ["company"],
        }


# ---------------------------------------------------------------------------
# Rule-based members
# ---------------------------------------------------------------------------


class TestRuleBasedMembers:
    def test_text_clean(self) -> None:
        m = TextCleanNormalizer(name="text_clean", lowercase=False)
        out = m.normalize(
            "  Apple  Inc.  ", attribute="name", kind="long_string", domain="companies"
        )
        assert out == "Apple Inc."

    def test_text_clean_list(self) -> None:
        m = TextCleanNormalizer(name="text_clean", lowercase=False)
        out = m.normalize(
            "Action, Shooter ; RPG", attribute="genres", kind="list", domain="games"
        )
        assert out == "Action, RPG, Shooter"

    def test_date_iso_year(self) -> None:
        m = DateIsoNormalizer(name="date_iso")
        assert (
            m.normalize("2005", attribute="founded", kind="year", domain="companies")
            == "2005"
        )
        assert (
            m.normalize(
                "2005-01-01", attribute="founded", kind="year", domain="companies"
            )
            == "2005"
        )
        assert (
            m.normalize("garbage", attribute="founded", kind="year", domain="companies")
            is None
        )

    def test_date_iso_date(self) -> None:
        m = DateIsoNormalizer(name="date_iso")
        out = m.normalize(
            "March 5, 2020", attribute="release-date", kind="date", domain="music"
        )
        assert out == "2020-03-05"

    def test_number_locale(self) -> None:
        m = NumberLocaleNormalizer(name="number_locale")
        assert (
            m.normalize(
                "148,700,000,000",
                attribute="assets",
                kind="continuous",
                domain="companies",
            )
            == "148700000000"
        )

    def test_country_iso_name(self) -> None:
        m = CountryIsoNormalizer(name="country_iso", output_format="name")
        assert (
            m.normalize(
                "Germany", attribute="country", kind="nominal", domain="companies"
            )
            == "Germany"
        )
        # pycountry maps "Russia" → official name "Russian Federation".
        assert (
            m.normalize(
                "Russia", attribute="country", kind="nominal", domain="companies"
            )
            == "Russian Federation"
        )
        # Returns None on garbage input (no fuzzy hit).
        assert (
            m.normalize(
                "z__not_a_country__z",
                attribute="country",
                kind="nominal",
                domain="companies",
            )
            is None
        )

    def test_taxonomy_lookup_companies_industry(self) -> None:
        m = TaxonomyLookupNormalizer(
            name="taxonomy_lookup",
            taxonomies={
                "companies": {
                    "industry": {
                        "path": "companies/input/schemamatching/GICS_Industry_Taxonomy.csv",
                        "columns": [
                            "Sector Name",
                            "Industry Group Name",
                            "Industry Name",
                            "Sub-Industry Name",
                        ],
                    }
                }
            },
        )
        # Hit at sector level.
        assert (
            m.normalize(
                "Financials",
                attribute="industry",
                kind="nominal",
                domain="companies",
            )
            == "Financials"
        )
        # Hit at industry level.
        assert (
            m.normalize(
                "Banks", attribute="industry", kind="nominal", domain="companies"
            )
            == "Banks"
        )
        # Miss returns None.
        assert (
            m.normalize(
                "garbage_industry",
                attribute="industry",
                kind="nominal",
                domain="companies",
            )
            is None
        )

    def test_taxonomy_lookup_resolves_aliased_domain(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Aliased domains inherit the source domain's taxonomy bindings.

        Regression for plan_s1_final.md F8: pre-fix, calling
        ``normalize(..., domain="music-small", ...)`` returned None for
        every cell because ``self._taxonomies`` was keyed by the source
        domain ``music``. Result: ``taxonomy_lookup`` scored 0.0 across
        all 418 genre cells in S.6a baseline audit.
        """
        from usecases_synthetic.lib import domain_config

        monkeypatch.setattr(
            domain_config,
            "_resolve_knob_config_alias",
            lambda dom: "companies" if dom == "companies-small" else None,
        )
        m = TaxonomyLookupNormalizer(
            name="taxonomy_lookup",
            taxonomies={
                "companies": {
                    "industry": {
                        "path": (
                            "companies/input/schemamatching/"
                            "GICS_Industry_Taxonomy.csv"
                        ),
                        "columns": ["Sector Name"],
                    }
                }
            },
        )
        # ``companies-small`` is not in ``taxonomies`` directly; alias
        # resolves to ``companies`` and the lookup must succeed.
        assert (
            m.normalize(
                "Financials",
                attribute="industry",
                kind="nominal",
                domain="companies-small",
            )
            == "Financials"
        )
        # An unaliased domain with no binding still returns None.
        assert (
            m.normalize(
                "Financials",
                attribute="industry",
                kind="nominal",
                domain="some_other_domain",
            )
            is None
        )


# ---------------------------------------------------------------------------
# End-to-end runner with a synthetic bundle
# ---------------------------------------------------------------------------


def _synthetic_bundle() -> VariantBundle:
    """Tiny companies-shaped bundle: 1 fusion entity, 2 sources, name+country."""
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
        variant_root=Path("/tmp/__test_norm__"),
    )


class TestNormCommitteeRunnerSyntheticBundle:
    """Run the runner with patched fusion targets so the test is self-contained."""

    def test_text_clean_only_roster(self, tmp_path: Path) -> None:
        # Build a tiny roster YAML that exercises text_clean alone.
        roster = tmp_path / "norm.yaml"
        roster.write_text(
            "seed: 42\n"
            "members:\n"
            "  - name: text_clean\n"
            "    module: usecases_synthetic.lib.normalizer_members\n"
            "    class: TextCleanNormalizer\n"
            "    signal_type: rule_string\n"
            "    enabled_by_default: true\n"
            "    applies_to: [name, country]\n"
            "    params:\n"
            "      lowercase: false\n"
            "      strip_whitespace: true\n"
            "required_axes:\n"
            "  signal_type: [rule_string]\n"
        )
        bundle = _synthetic_bundle()

        # Patch fusion targets: one entity with name + country.
        fake_targets = {
            "ent1": {"name": ["Apple Inc."], "country": ["United States"]},
            "ent2": {"name": ["BMW AG"], "country": ["Germany"]},
        }

        with patch(
            "usecases_synthetic.lib.committee_norm.load_fusion_target_values",
            return_value=fake_targets,
        ):
            runner = NormCommitteeRunner(roster, with_llm=False)
            result = runner.run(bundle)

        assert result.stage == "norm"
        assert result.domain == "companies"
        assert "text_clean" in result.per_member
        text_clean = result.per_member["text_clean"]
        # Two entities × two sources × two attributes = 8 cells total.
        # text_clean strips whitespace; on this fixture it preserves
        # exact-match for dbpedia (Apple Inc., United States, BMW AG,
        # Germany), so its precision/recall should be > 0.
        assert text_clean.metrics["macro_f1"] > 0.0
        assert "name" in result.per_attribute
        assert "country" in result.per_attribute

    def test_runner_raises_on_empty_sm_mapping(self, tmp_path: Path) -> None:
        roster = tmp_path / "norm.yaml"
        roster.write_text(
            "seed: 42\n"
            "members:\n"
            "  - name: text_clean\n"
            "    module: usecases_synthetic.lib.normalizer_members\n"
            "    class: TextCleanNormalizer\n"
            "    signal_type: rule_string\n"
            "    enabled_by_default: true\n"
            "    applies_to: [name]\n"
            "    params: {}\n"
            "required_axes:\n"
            "  signal_type: [rule_string]\n"
        )
        bundle = _synthetic_bundle()
        bundle.sm_mapping = None

        runner = NormCommitteeRunner(roster, with_llm=False)
        with pytest.raises(ValueError, match="No SM mapping"):
            runner.run(bundle)


# ---------------------------------------------------------------------------
# Real-config sanity: every authored YAML loads + parses members
# ---------------------------------------------------------------------------


class TestPerDomainConfigsLoad:
    # Under C12 (plan_revision.md §C12, landed 2026-05-26) every per-
    # domain norm YAML declares the same 3-member roster
    # (rule_per_attribute_optimal / llm_only / passthrough). With
    # ``with_llm=False`` the llm_only member is skipped, leaving the
    # 2-member active set.
    @pytest.mark.parametrize("domain", ["companies", "games", "music"])
    def test_construct_runner(self, domain: str) -> None:
        path = COMMITTEE_DIR / f"normalization_committee_{domain}.yaml"
        runner = NormCommitteeRunner(path, with_llm=False)
        assert runner.roster_names == [
            "rule_per_attribute_optimal",
            "passthrough",
        ]

    @pytest.mark.parametrize("domain", ["companies", "games", "music"])
    def test_construct_runner_with_llm(self, domain: str) -> None:
        path = COMMITTEE_DIR / f"normalization_committee_{domain}.yaml"
        runner = NormCommitteeRunner(path, with_llm=True)
        assert runner.roster_names == [
            "rule_per_attribute_optimal",
            "llm_only",
            "passthrough",
        ]
