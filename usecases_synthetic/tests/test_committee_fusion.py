"""Tests for the fusion committee runner.

Exercises ``FusionCommitteeRunner`` instantiation from a fixture
roster, scoring against a fixture gold standard, and basic signal
directions (voting on identical sources gives accuracy 1.0,
longest_string picks the longest value).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from usecases_synthetic.lib.committee import CommitteeResult, MemberResult
from usecases_synthetic.lib.committee_fusion import (
    FusionCommitteeRunner,
    _build_correspondences_from_bundle,
    _compute_aggregated,
    _compute_per_attribute,
)
from usecases_synthetic.lib.committee_fusion_scoring import score_fusion
from usecases_synthetic.lib.variant_loader import VariantBundle

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_fixture_roster(
    tmp_path: Path,
    *,
    trust_scores: dict[str, float] | None = None,
    attributes: dict[str, Any] | None = None,
) -> Path:
    """Write a minimal fusion roster YAML for testing.

    Defaults: 2 attributes (name, revenue), 2 strategies each.
    """
    if trust_scores is None:
        trust_scores = {"source_a": 3, "source_b": 2, "source_c": 1}

    if attributes is None:
        attributes = {
            "name": {
                "attribute_class": "primary",
                "strategies": [
                    {
                        "name": "voting",
                        "function": "voting",
                        "module": "PyDI.fusion.conflict_resolution.general",
                        "strategy_type": "cell_local",
                        "params": {},
                    },
                    {
                        "name": "longest_string",
                        "function": "longest_string",
                        "module": "PyDI.fusion.conflict_resolution.string",
                        "strategy_type": "cell_local",
                        "params": {},
                    },
                ],
            },
            "revenue": {
                "attribute_class": "secondary",
                "strategies": [
                    {
                        "name": "median",
                        "function": "median",
                        "module": "PyDI.fusion.conflict_resolution.numeric",
                        "strategy_type": "cell_local",
                        "params": {},
                    },
                    {
                        "name": "prefer_higher_trust",
                        "function": "prefer_higher_trust",
                        "module": "PyDI.fusion.conflict_resolution.general",
                        "strategy_type": "trust_weighted",
                        "params": {},
                    },
                ],
            },
        }

    roster = {
        "seed": 42,
        "trust_scores": trust_scores,
        "fused_id_column": "id",
        "gold_id_column": "id",
        "evaluation_functions": {
            "name": "tokenized_match",
            "revenue": "numeric_tolerance_match",
        },
        "evaluation_params": {
            "revenue": {"tolerance": 0.1},
        },
        "attributes": attributes,
        "required_axes": {
            "strategy_type": ["cell_local", "trust_weighted"],
        },
    }

    path = tmp_path / "fusion_committee.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(roster, f)
    return path


def _make_identical_sources(
    n: int = 10,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """Create 3 sources with identical values + correspondences + gold.

    When all sources agree, ``voting`` should produce accuracy 1.0.

    Returns
    -------
    tuple
        (sources, correspondences, fusion_gold)
    """
    names = [f"Company {i}" for i in range(n)]
    revenues = [float(1000 * (i + 1)) for i in range(n)]

    src_a = pd.DataFrame(
        {
            "id": [f"a_{i}" for i in range(n)],
            "name": names,
            "revenue": revenues,
        }
    )
    src_a.attrs["dataset_name"] = "source_a"

    src_b = pd.DataFrame(
        {
            "id": [f"b_{i}" for i in range(n)],
            "name": names,
            "revenue": revenues,
        }
    )
    src_b.attrs["dataset_name"] = "source_b"

    src_c = pd.DataFrame(
        {
            "id": [f"c_{i}" for i in range(n)],
            "name": names,
            "revenue": revenues,
        }
    )
    src_c.attrs["dataset_name"] = "source_c"

    sources = {"source_a": src_a, "source_b": src_b, "source_c": src_c}

    # Correspondences: a_i <-> b_i <-> c_i.
    rows: list[dict[str, Any]] = []
    for i in range(n):
        rows.append({"id1": f"a_{i}", "id2": f"b_{i}", "score": 1.0})
        rows.append({"id1": f"b_{i}", "id2": f"c_{i}", "score": 1.0})
    correspondences = pd.DataFrame(rows)

    # Gold: one fused record per entity, using a_i as fused ID.
    gold = pd.DataFrame(
        {
            "id": [f"a_{i}" for i in range(n)],
            "name": names,
            "revenue": revenues,
        }
    )

    return sources, correspondences, gold


def _make_distinct_length_sources(
    n: int = 5,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """Create 3 sources where names have different lengths.

    Source A has short names, B has medium, C has the longest.  The
    gold uses C's names, so ``longest_string`` for ``name`` should
    achieve accuracy 1.0 on name, while ``voting`` should do worse.

    Returns
    -------
    tuple
        (sources, correspondences, fusion_gold)
    """
    short_names = [f"Co{i}" for i in range(n)]
    medium_names = [f"Company_{i}" for i in range(n)]
    long_names = [f"The Great Company Number {i}" for i in range(n)]
    revenues = [float(1000 * (i + 1)) for i in range(n)]

    src_a = pd.DataFrame(
        {
            "id": [f"a_{i}" for i in range(n)],
            "name": short_names,
            "revenue": revenues,
        }
    )
    src_a.attrs["dataset_name"] = "source_a"

    src_b = pd.DataFrame(
        {
            "id": [f"b_{i}" for i in range(n)],
            "name": medium_names,
            "revenue": revenues,
        }
    )
    src_b.attrs["dataset_name"] = "source_b"

    src_c = pd.DataFrame(
        {
            "id": [f"c_{i}" for i in range(n)],
            "name": long_names,
            "revenue": revenues,
        }
    )
    src_c.attrs["dataset_name"] = "source_c"

    sources = {"source_a": src_a, "source_b": src_b, "source_c": src_c}

    rows: list[dict[str, Any]] = []
    for i in range(n):
        rows.append({"id1": f"a_{i}", "id2": f"b_{i}", "score": 1.0})
        rows.append({"id1": f"b_{i}", "id2": f"c_{i}", "score": 1.0})
    correspondences = pd.DataFrame(rows)

    # Gold uses the longest names.
    gold = pd.DataFrame(
        {
            "id": [f"a_{i}" for i in range(n)],
            "name": long_names,
            "revenue": revenues,
        }
    )

    return sources, correspondences, gold


def _make_bundle(
    sources: dict[str, pd.DataFrame],
    correspondences: pd.DataFrame,
    fusion_gold: pd.DataFrame,
    *,
    level: str = "baseline",
) -> VariantBundle:
    """Build a minimal VariantBundle for fusion testing."""
    # Build em_gold from correspondences so _build_correspondences_from_bundle works.
    em_gold: dict[tuple[str, str], pd.DataFrame] = {}
    src_names = list(sources.keys())
    if len(src_names) >= 2:
        corr_copy = correspondences.copy()
        corr_copy["label"] = 1
        em_gold[(src_names[0], src_names[1])] = corr_copy

    return VariantBundle(
        domain="companies",
        level=level,
        sources=sources,
        target_schema={"properties": {"id": {}, "name": {}, "revenue": {}}},
        sm_mapping=None,
        em_gold=em_gold,
        em_splits={},
        fusion_gold=fusion_gold,
        fusion_validation=None,
        pooled_positives=None,
        variant_root=Path("/tmp/fusion_test"),
    )


# ---------------------------------------------------------------------------
# Scoring unit tests
# ---------------------------------------------------------------------------


class TestScoreFusion:
    """Tests for ``score_fusion``."""

    def test_perfect_score(self) -> None:
        """Identical fused and gold should give accuracy 1.0."""
        n = 5
        df = pd.DataFrame(
            {
                "id": [f"a_{i}" for i in range(n)],
                "name": [f"Company {i}" for i in range(n)],
                "revenue": [float(1000 * i) for i in range(n)],
            }
        )
        gold = pd.DataFrame(
            {
                "id": [f"a_{i}" for i in range(n)],
                "name": [f"Company {i}" for i in range(n)],
                "revenue": [float(1000 * i) for i in range(n)],
            }
        )

        metrics = score_fusion(
            fused_df=df,
            gold_df=gold,
            eval_specs={"name": "tokenized_match", "revenue": "exact_match"},
            fused_id_column="id",
            gold_id_column="id",
        )
        assert metrics["overall_accuracy"] == 1.0

    def test_partial_score(self) -> None:
        """Some correct, some wrong should give partial accuracy."""
        df = pd.DataFrame(
            {
                "id": ["a_0", "a_1"],
                "name": ["Correct Name", "Wrong Name"],
            }
        )
        gold = pd.DataFrame(
            {
                "id": ["a_0", "a_1"],
                "name": ["Correct Name", "Right Name"],
            }
        )

        metrics = score_fusion(
            fused_df=df,
            gold_df=gold,
            eval_specs={"name": "exact_match"},
            fused_id_column="id",
            gold_id_column="id",
        )
        assert metrics["name_accuracy"] == 0.5


# ---------------------------------------------------------------------------
# Per-attribute computation tests
# ---------------------------------------------------------------------------


class TestComputePerAttribute:
    """Tests for ``_compute_per_attribute``."""

    def test_spread_computation(self) -> None:
        """Spread should be max - min of strategy accuracies."""
        attr_strat_acc = {
            "name": {"voting": 0.8, "longest_string": 0.6},
            "revenue": {"median": 0.9, "prefer_higher_trust": 0.4},
        }
        result = _compute_per_attribute(attr_strat_acc)

        assert result["name"]["spread"] == pytest.approx(0.2)
        assert result["name"]["best_strategy_accuracy"] == pytest.approx(0.8)
        assert result["name"]["mean_strategy_accuracy"] == pytest.approx(0.7)

        assert result["revenue"]["spread"] == pytest.approx(0.5)
        assert result["revenue"]["best_strategy_accuracy"] == pytest.approx(0.9)

    def test_empty_strategies(self) -> None:
        """Empty strategies should produce zero metrics."""
        result = _compute_per_attribute({"name": {}})
        assert result["name"]["spread"] == 0.0
        assert result["name"]["best_strategy_accuracy"] == 0.0


class TestComputeAggregated:
    """Tests for ``_compute_aggregated``."""

    def test_overall_accuracy(self) -> None:
        """Overall accuracy is macro of per-attribute best."""
        per_attribute = {
            "name": {
                "best_strategy_accuracy": 0.8,
                "mean_strategy_accuracy": 0.7,
                "spread": 0.2,
            },
            "revenue": {
                "best_strategy_accuracy": 0.6,
                "mean_strategy_accuracy": 0.5,
                "spread": 0.1,
            },
        }
        result = _compute_aggregated(per_attribute)
        assert result["overall_accuracy"] == pytest.approx(0.7)
        assert result["overall_mean_accuracy"] == pytest.approx(0.6)
        assert result["overall_spread"] == pytest.approx(0.15)

    def test_empty(self) -> None:
        result = _compute_aggregated({})
        assert result["overall_accuracy"] == 0.0


# ---------------------------------------------------------------------------
# FusionCommitteeRunner tests
# ---------------------------------------------------------------------------


class TestFusionCommitteeRunner:
    """Tests for the full fusion committee runner."""

    def test_instantiation(self, tmp_path: Path) -> None:
        """Runner loads a fixture roster."""
        roster_path = _write_fixture_roster(tmp_path)
        runner = FusionCommitteeRunner(roster_path)

        # 2 attributes × 2 strategies = 4 members.
        assert len(runner.roster_names) == 4
        assert "name_voting" in runner.roster_names
        assert "name_longest_string" in runner.roster_names
        assert "revenue_median" in runner.roster_names
        assert "revenue_prefer_higher_trust" in runner.roster_names

    def test_voting_identical_sources(self, tmp_path: Path) -> None:
        """Voting on identical sources should give high accuracy."""
        roster_path = _write_fixture_roster(tmp_path)
        runner = FusionCommitteeRunner(roster_path)

        sources, correspondences, gold = _make_identical_sources()
        bundle = _make_bundle(sources, correspondences, gold)

        result = runner.run(bundle, correspondences=correspondences)

        assert isinstance(result, CommitteeResult)
        assert result.stage == "fusion"
        assert result.domain == "companies"

        # Voting on identical sources should get accuracy 1.0 for name.
        name_voting = result.per_member["name_voting"]
        name_acc = name_voting.metrics.get("name_accuracy", 0.0)
        assert (
            name_acc >= 0.9
        ), f"Voting on identical names should have high accuracy, got {name_acc}"

    def test_longest_string_picks_longest(self, tmp_path: Path) -> None:
        """longest_string should outperform voting when gold uses longest names."""
        roster_path = _write_fixture_roster(tmp_path)
        runner = FusionCommitteeRunner(roster_path)

        sources, correspondences, gold = _make_distinct_length_sources()
        bundle = _make_bundle(sources, correspondences, gold)

        result = runner.run(bundle, correspondences=correspondences)

        # longest_string should achieve higher name_accuracy than voting
        # because gold uses the longest names.
        longest_acc = result.per_member["name_longest_string"].metrics.get(
            "name_accuracy", 0.0
        )
        voting_acc = result.per_member["name_voting"].metrics.get("name_accuracy", 0.0)
        assert longest_acc >= voting_acc, (
            f"longest_string ({longest_acc}) should be >= voting ({voting_acc}) "
            "when gold uses longest names"
        )

    def test_per_attribute_spread_populated(self, tmp_path: Path) -> None:
        """per_attribute should contain spread for K10 detection."""
        roster_path = _write_fixture_roster(tmp_path)
        runner = FusionCommitteeRunner(roster_path)

        sources, correspondences, gold = _make_identical_sources()
        bundle = _make_bundle(sources, correspondences, gold)
        result = runner.run(bundle, correspondences=correspondences)

        assert "name" in result.per_attribute
        assert "revenue" in result.per_attribute
        assert "spread" in result.per_attribute["name"]
        assert "best_strategy_accuracy" in result.per_attribute["name"]
        assert "mean_strategy_accuracy" in result.per_attribute["name"]

    def test_per_partition_by_attribute_class(self, tmp_path: Path) -> None:
        """per_partition should group attributes by attribute_class."""
        roster_path = _write_fixture_roster(tmp_path)
        runner = FusionCommitteeRunner(roster_path)

        sources, correspondences, gold = _make_identical_sources()
        bundle = _make_bundle(sources, correspondences, gold)
        result = runner.run(bundle, correspondences=correspondences)

        assert "primary" in result.per_partition
        assert "secondary" in result.per_partition
        assert "n_attributes" in result.per_partition["primary"]

    def test_aggregated_metrics(self, tmp_path: Path) -> None:
        """Aggregated metrics should contain expected keys."""
        roster_path = _write_fixture_roster(tmp_path)
        runner = FusionCommitteeRunner(roster_path)

        sources, correspondences, gold = _make_identical_sources()
        bundle = _make_bundle(sources, correspondences, gold)
        result = runner.run(bundle, correspondences=correspondences)

        assert "overall_accuracy" in result.aggregated
        assert "overall_mean_accuracy" in result.aggregated
        assert "overall_spread" in result.aggregated

    def test_result_as_dict_serializable(self, tmp_path: Path) -> None:
        """CommitteeResult.as_dict() produces a JSON-serialisable dict."""
        roster_path = _write_fixture_roster(tmp_path)
        runner = FusionCommitteeRunner(roster_path)

        sources, correspondences, gold = _make_identical_sources()
        bundle = _make_bundle(sources, correspondences, gold)
        result = runner.run(bundle, correspondences=correspondences)

        payload = result.as_dict()
        # Should not raise.
        json.dumps(payload)

    def test_no_gold_raises(self, tmp_path: Path) -> None:
        """Runner raises ValueError when no fusion gold is available."""
        roster_path = _write_fixture_roster(tmp_path)
        runner = FusionCommitteeRunner(roster_path)

        bundle = _make_bundle(
            sources={},
            correspondences=pd.DataFrame(columns=["id1", "id2", "score"]),
            fusion_gold=pd.DataFrame(),
        )

        with pytest.raises(ValueError, match="No fusion gold"):
            runner.run(bundle)

    def test_member_notes_populated(self, tmp_path: Path) -> None:
        """Each MemberResult.notes should carry strategy metadata."""
        roster_path = _write_fixture_roster(tmp_path)
        runner = FusionCommitteeRunner(roster_path)

        sources, correspondences, gold = _make_identical_sources()
        bundle = _make_bundle(sources, correspondences, gold)
        result = runner.run(bundle, correspondences=correspondences)

        for member_key, member_result in result.per_member.items():
            assert "attribute" in member_result.notes
            assert "strategy" in member_result.notes
            assert "strategy_type" in member_result.notes
            assert "attribute_class" in member_result.notes

    def test_robust_aggregators_wired_via_yaml(self, tmp_path: Path) -> None:
        """C3.4.1 integration: trimmed_mean / huber_m_estimator /
        median_of_means dispatched from a roster YAML referencing
        ``usecases_synthetic.lib.robust_aggregators`` run end-to-end and
        outperform the non-robust baseline on a deliberately-outlier
        numeric scenario.

        Setup: 5 aligned sources with revenues ``[100, 100, 100, 100,
        10000]``. The expected robust value is 100 (the clean reading);
        ``mean`` would return 2080, so the robust aggregators should
        score noticeably higher against a gold of 100.
        """
        attributes = {
            "revenue": {
                "attribute_class": "secondary",
                "strategies": [
                    {
                        "name": "average",
                        "function": "average",
                        "module": "PyDI.fusion.conflict_resolution.numeric",
                        "strategy_type": "cell_local",
                        "params": {},
                    },
                    {
                        "name": "trimmed_mean",
                        "function": "trimmed_mean",
                        "module": "usecases_synthetic.lib.robust_aggregators",
                        "strategy_type": "cell_local",
                        "params": {"trim": 0.2},
                    },
                    {
                        "name": "huber_m_estimator",
                        "function": "huber_m_estimator",
                        "module": "usecases_synthetic.lib.robust_aggregators",
                        "strategy_type": "cell_local",
                        "params": {},
                    },
                    {
                        "name": "median_of_means",
                        "function": "median_of_means",
                        "module": "usecases_synthetic.lib.robust_aggregators",
                        "strategy_type": "cell_local",
                        # n_blocks=5 reduces median_of_means to a plain
                        # median when all 5 sources contribute — each
                        # value forms its own block, so the block-means
                        # median = median of values.
                        "params": {"n_blocks": 5},
                    },
                    {
                        "name": "prefer_higher_trust",
                        "function": "prefer_higher_trust",
                        "module": "PyDI.fusion.conflict_resolution.general",
                        "strategy_type": "trust_weighted",
                        "params": {},
                    },
                ],
            },
        }
        trust_scores: dict[str, float] = {
            "source_a": 3.0,
            "source_b": 3.0,
            "source_c": 3.0,
            "source_d": 3.0,
            "source_e": 1.0,  # outlier source gets low trust
        }
        roster_path = _write_fixture_roster(
            tmp_path,
            trust_scores=trust_scores,
            attributes=attributes,
        )
        # _write_fixture_roster default includes "name" in
        # evaluation_functions; overwrite to drop it since our attributes
        # dict only contains "revenue".
        with open(roster_path, encoding="utf-8") as f:
            roster_yaml = yaml.safe_load(f)
        roster_yaml["evaluation_functions"] = {"revenue": "numeric_tolerance_match"}
        roster_yaml["evaluation_params"] = {"revenue": {"tolerance": 0.05}}
        with open(roster_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(roster_yaml, f)

        # 2 entities, each with 5 aligned sources; source_e is the outlier.
        clean = 100.0
        outlier = 10_000.0
        revenues = [clean, clean, clean, clean, outlier]
        src_ids = ["a", "b", "c", "d", "e"]
        srcs: dict[str, pd.DataFrame] = {}
        for sid, rev in zip(src_ids, revenues, strict=True):
            df = pd.DataFrame(
                {
                    "id": [f"{sid}_0", f"{sid}_1"],
                    "revenue": [rev, rev],
                }
            )
            df.attrs["dataset_name"] = f"source_{sid}"
            srcs[f"source_{sid}"] = df

        corr_rows: list[dict[str, Any]] = []
        for i in range(2):
            for j in range(len(src_ids) - 1):
                corr_rows.append(
                    {
                        "id1": f"{src_ids[j]}_{i}",
                        "id2": f"{src_ids[j + 1]}_{i}",
                        "score": 1.0,
                    }
                )
        correspondences = pd.DataFrame(corr_rows)
        gold = pd.DataFrame(
            {
                "id": ["a_0", "a_1"],
                "revenue": [clean, clean],
            }
        )

        bundle = _make_bundle(srcs, correspondences, gold)
        runner = FusionCommitteeRunner(roster_path)
        result = runner.run(bundle, correspondences=correspondences)

        mean_acc = result.per_member["revenue_average"].metrics["revenue_accuracy"]
        trimmed_acc = result.per_member["revenue_trimmed_mean"].metrics[
            "revenue_accuracy"
        ]
        huber_acc = result.per_member["revenue_huber_m_estimator"].metrics[
            "revenue_accuracy"
        ]
        mom_acc = result.per_member["revenue_median_of_means"].metrics[
            "revenue_accuracy"
        ]

        # Non-robust average is pulled toward the outlier: 2080 vs gold
        # 100 with tolerance 0.05 => accuracy must be 0.0. Robust
        # aggregators recover the clean value and should score 1.0.
        assert mean_acc == pytest.approx(
            0.0
        ), f"average should fail on outlier data, got {mean_acc}"
        assert trimmed_acc == pytest.approx(
            1.0
        ), f"trimmed_mean should recover clean value, got {trimmed_acc}"
        assert huber_acc == pytest.approx(
            1.0
        ), f"huber_m_estimator should recover clean value, got {huber_acc}"
        # median_of_means with default n_blocks on 5 values will partition
        # as blocks of 1 (n_blocks ~= sqrt(n)), so the block median is
        # robust to the single outlier.
        assert mom_acc == pytest.approx(
            1.0
        ), f"median_of_means should recover clean value, got {mom_acc}"


# ---------------------------------------------------------------------------
# Correspondence builder tests
# ---------------------------------------------------------------------------


class TestBuildCorrespondencesFromBundle:
    """Tests for ``_build_correspondences_from_bundle``."""

    def test_filters_to_positives(self) -> None:
        """Only label==1 rows should be kept."""
        gold = pd.DataFrame(
            {
                "id1": ["a_0", "a_1", "a_2"],
                "id2": ["b_0", "b_1", "b_2"],
                "label": [1, 0, 1],
                "score": [1.0, 0.0, 1.0],
            }
        )

        bundle = VariantBundle(
            domain="companies",
            level="baseline",
            sources={},
            target_schema={"properties": {}},
            sm_mapping=None,
            em_gold={("source_a", "source_b"): gold},
            em_splits={},
            fusion_gold=pd.DataFrame(),
            fusion_validation=None,
            pooled_positives=None,
            variant_root=Path("/tmp/test"),
        )

        result = _build_correspondences_from_bundle(bundle)
        assert len(result) == 2
        assert set(result["id1"]) == {"a_0", "a_2"}

    def test_empty_bundle(self) -> None:
        """Empty EM gold should produce empty correspondences."""
        bundle = VariantBundle(
            domain="companies",
            level="baseline",
            sources={},
            target_schema={"properties": {}},
            sm_mapping=None,
            em_gold={},
            em_splits={},
            fusion_gold=pd.DataFrame(),
            fusion_validation=None,
            pooled_positives=None,
            variant_root=Path("/tmp/test"),
        )

        result = _build_correspondences_from_bundle(bundle)
        assert result.empty
        assert "id1" in result.columns
