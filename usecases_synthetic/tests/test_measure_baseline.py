"""Tests for ``usecases_synthetic.scripts.measure_baseline``.

Monkey-patches the three committee runners to return tiny fixture
results so the tests run fast and without real data dependencies.
Verifies JSON output structure, markdown report rendering, round-trip
via ``baseline_loader``, and ``fusion_input_member`` selection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from usecases_synthetic.lib.baseline_loader import load_baseline
from usecases_synthetic.lib.committee import CommitteeResult, MemberResult
from usecases_synthetic.scripts.measure_baseline import (
    _best_em_member,
    _parse_stages,
    measure_baseline,
)

# ---------------------------------------------------------------------------
# Fixture committee results
# ---------------------------------------------------------------------------


def _fixture_sm_result(domain: str = "companies") -> CommitteeResult:
    """Return a small SM CommitteeResult for testing."""
    return CommitteeResult(
        stage="sm",
        domain=domain,
        level="baseline",
        per_member={
            "label_jaccard": MemberResult(
                name="label_jaccard",
                predictions=None,
                metrics={"precision": 0.85, "recall": 0.80, "f1": 0.824},
                runtime_s=0.5,
                notes={"signal_type": "label"},
            ),
            "instance_tfidf": MemberResult(
                name="instance_tfidf",
                predictions=None,
                metrics={"precision": 0.70, "recall": 0.90, "f1": 0.788},
                runtime_s=1.2,
                notes={"signal_type": "instance"},
            ),
        },
        aggregated={
            "macro_f1": 0.806,
            "min_f1": 0.788,
            "max_f1": 0.824,
            "macro_precision": 0.775,
            "macro_recall": 0.850,
        },
        per_attribute={
            "dbpedia.name": {
                "label_jaccard": 1.0,
                "instance_tfidf": 1.0,
                "any_correct": 1.0,
            },
            "dbpedia.country": {
                "label_jaccard": 1.0,
                "instance_tfidf": 0.0,
                "any_correct": 1.0,
            },
        },
        per_partition={
            "dbpedia": {"macro_f1": 0.82, "n_columns": 4.0},
            "forbes": {"macro_f1": 0.79, "n_columns": 3.0},
        },
        runtime_s=1.7,
        roster=["label_jaccard", "instance_tfidf"],
    )


def _fixture_em_blocking_result(domain: str = "companies") -> CommitteeResult:
    """Return a small em_blocking CommitteeResult for testing.

    Mirrors the keys emitted by
    :class:`EMBlockingCommitteeRunner` post-2026-05-13 EM stage split.
    """
    return CommitteeResult(
        stage="em_blocking",
        domain=domain,
        level="baseline",
        per_member={
            "token_blocker": MemberResult(
                name="token_blocker",
                predictions=None,
                metrics={
                    "pair_recall": 0.92,
                    "reduction_ratio": 0.97,
                },
                runtime_s=1.5,
                notes={"blocking_type": "lexical"},
            ),
            "embedding_blocker": MemberResult(
                name="embedding_blocker",
                predictions=None,
                metrics={
                    "pair_recall": 0.98,
                    "reduction_ratio": 0.95,
                },
                runtime_s=2.5,
                notes={"blocking_type": "embedding"},
            ),
        },
        aggregated={
            "macro_pair_recall": 0.95,
            "min_pair_recall": 0.92,
            "max_pair_recall": 0.98,
            "macro_reduction_ratio": 0.96,
            "best_member_name": "embedding_blocker",
            "best_member_pair_recall": 0.98,
            "best_member_reduction_ratio": 0.95,
            "recall_floor": 0.97,
        },
        per_attribute={},
        per_partition={
            "forbes_dbpedia": {
                "macro_pair_recall": 0.94,
                "macro_reduction_ratio": 0.96,
                "n_members": 2.0,
            },
        },
        runtime_s=4.0,
        roster=["token_blocker", "embedding_blocker"],
    )


def _fixture_em_matching_result(domain: str = "companies") -> CommitteeResult:
    """Return a small em_matching CommitteeResult for testing.

    Mirrors the keys emitted by
    :class:`EMMatchingCommitteeRunner` post-2026-05-13 EM stage split
    (closed-set scoring per plan_revision.md §C10).
    """
    return CommitteeResult(
        stage="em_matching",
        domain=domain,
        level="baseline",
        per_member={
            "token_rule": MemberResult(
                name="token_rule",
                predictions=None,
                metrics={
                    "f1": 0.72,
                    "precision": 0.80,
                    "recall": 0.65,
                    "f1_baseline_test": 0.74,
                    "f1_regen_test": 0.72,
                },
                runtime_s=3.0,
                notes={"matching_type": "rule"},
            ),
            "embedding_rule": MemberResult(
                name="embedding_rule",
                predictions=None,
                metrics={
                    "f1": 0.78,
                    "precision": 0.75,
                    "recall": 0.82,
                    "f1_baseline_test": 0.79,
                    "f1_regen_test": 0.78,
                },
                runtime_s=5.0,
                notes={"matching_type": "rule"},
            ),
        },
        aggregated={
            "macro_f1": 0.75,
            "min_f1": 0.72,
            "max_f1": 0.78,
            "macro_precision": 0.775,
            "macro_recall": 0.735,
            "macro_f1_baseline_test": 0.765,
            "macro_f1_regen_test": 0.75,
            "best_member_name": "embedding_rule",
            "best_member_f1": 0.78,
        },
        per_attribute={},
        per_partition={
            "forbes_dbpedia": {
                "macro_f1": 0.76,
                "min_f1": 0.72,
                "max_f1": 0.80,
                "n_members": 2.0,
            },
            "forbes_fullcontact": {
                "macro_f1": 0.74,
                "min_f1": 0.70,
                "max_f1": 0.78,
                "n_members": 2.0,
            },
        },
        runtime_s=8.0,
        roster=["token_rule", "embedding_rule"],
    )


def _fixture_norm_result(domain: str = "companies") -> CommitteeResult:
    """Return a small Normalization CommitteeResult for testing."""
    return CommitteeResult(
        stage="norm",
        domain=domain,
        level="baseline",
        per_member={
            "text_clean": MemberResult(
                name="text_clean",
                predictions=None,
                metrics={
                    "macro_f1": 0.62,
                    "macro_precision": 0.70,
                    "macro_recall": 0.55,
                    "min_f1": 0.55,
                    "max_f1": 0.70,
                    "f1": 0.62,
                    "precision": 0.70,
                    "recall": 0.55,
                    "n_attributes": 3.0,
                    "n_cells": 25.0,
                },
                runtime_s=0.4,
                notes={"signal_type": "rule_string"},
            ),
        },
        aggregated={
            "macro_f1": 0.62,
            "min_f1": 0.62,
            "max_f1": 0.62,
            "macro_precision": 0.70,
            "macro_recall": 0.55,
            "best_member_f1": 0.62,
            "best_member_name_f1": 1.0,
        },
        per_attribute={
            "name": {"text_clean": 0.62, "any_correct": 1.0, "best_member_f1": 0.62},
        },
        per_partition={
            "dbpedia": {"macro_f1": 0.62, "n_attributes": 3.0},
        },
        runtime_s=0.4,
        roster=["text_clean"],
    )


def _fixture_fusion_result(domain: str = "companies") -> CommitteeResult:
    """Return a small Fusion CommitteeResult for testing."""
    return CommitteeResult(
        stage="fusion",
        domain=domain,
        level="baseline",
        per_member={
            "name_voting": MemberResult(
                name="name_voting",
                predictions=None,
                metrics={
                    "overall_accuracy": 0.82,
                    "macro_accuracy": 0.78,
                    "name_accuracy": 0.85,
                },
                runtime_s=2.0,
                notes={"attribute": "name", "strategy": "voting"},
            ),
            "name_longest_string": MemberResult(
                name="name_longest_string",
                predictions=None,
                metrics={
                    "overall_accuracy": 0.75,
                    "macro_accuracy": 0.72,
                    "name_accuracy": 0.78,
                },
                runtime_s=1.8,
                notes={"attribute": "name", "strategy": "longest_string"},
            ),
        },
        aggregated={
            "overall_accuracy": 0.85,
            "overall_mean_accuracy": 0.80,
            "overall_spread": 0.07,
        },
        per_attribute={
            "name": {
                "best_strategy_accuracy": 0.85,
                "mean_strategy_accuracy": 0.815,
                "spread": 0.07,
                "voting": 0.85,
                "longest_string": 0.78,
            },
        },
        per_partition={
            "primary": {
                "mean_best_accuracy": 0.85,
                "mean_spread": 0.07,
                "n_attributes": 1.0,
            },
        },
        runtime_s=3.8,
        roster=["name_voting", "name_longest_string"],
    )


# ---------------------------------------------------------------------------
# Mock setup
# ---------------------------------------------------------------------------


def _patch_runners():
    """Return a context-manager stack that patches all committee runners.

    Patches both ``EMBlockingCommitteeRunner`` and
    ``EMMatchingCommitteeRunner`` introduced by the 2026-05-13 EM stage
    split.
    """
    sm_mock = MagicMock()
    sm_mock.return_value.run.return_value = _fixture_sm_result()

    norm_mock = MagicMock()
    norm_mock.return_value.run.return_value = _fixture_norm_result()

    em_blocking_mock = MagicMock()
    em_blocking_mock.return_value.run.return_value = _fixture_em_blocking_result()

    em_matching_mock = MagicMock()
    em_matching_mock.return_value.run.return_value = _fixture_em_matching_result()

    fusion_mock = MagicMock()
    fusion_mock.return_value.run.return_value = _fixture_fusion_result()

    bundle_mock = MagicMock()
    bundle_mock.return_value.domain = "companies"
    bundle_mock.return_value.level = "baseline"

    return (
        patch("usecases_synthetic.scripts.measure_baseline.SMCommitteeRunner", sm_mock),
        patch(
            "usecases_synthetic.scripts.measure_baseline.NormCommitteeRunner",
            norm_mock,
        ),
        patch(
            "usecases_synthetic.scripts.measure_baseline.EMBlockingCommitteeRunner",
            em_blocking_mock,
        ),
        patch(
            "usecases_synthetic.scripts.measure_baseline.EMMatchingCommitteeRunner",
            em_matching_mock,
        ),
        patch(
            "usecases_synthetic.scripts.measure_baseline.FusionCommitteeRunner",
            fusion_mock,
        ),
        patch("usecases_synthetic.scripts.measure_baseline.load_variant", bundle_mock),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMeasureBaseline:
    """Tests for the ``measure_baseline`` function."""

    def test_writes_json_with_all_stages(self, tmp_path: Path) -> None:
        patches = _patch_runners()
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            result = measure_baseline("companies", out_dir=tmp_path)

        json_path = tmp_path / "baseline_metrics.json"
        assert json_path.exists()

        with open(json_path, encoding="utf-8") as f:
            payload = json.load(f)

        assert payload["domain"] == "companies"
        assert "sm" in payload["per_stage"]
        assert "em_blocking" in payload["per_stage"]
        assert "em_matching" in payload["per_stage"]
        assert "fusion" in payload["per_stage"]

    def test_json_round_trip_via_baseline_loader(self, tmp_path: Path) -> None:
        patches = _patch_runners()
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            measure_baseline("companies", out_dir=tmp_path)

        json_path = tmp_path / "baseline_metrics.json"
        loaded = load_baseline("companies", path_override=json_path)

        assert loaded.domain == "companies"
        # 5 stages: sm + norm + em_blocking + em_matching + fusion.
        assert len(loaded.per_stage) == 5

        sm_agg = loaded.aggregated("sm")
        assert sm_agg["macro_f1"] == pytest.approx(0.806)

        em_blocking_agg = loaded.aggregated("em_blocking")
        assert em_blocking_agg["macro_pair_recall"] == pytest.approx(0.95)

        em_matching_agg = loaded.aggregated("em_matching")
        assert em_matching_agg["macro_f1_regen_test"] == pytest.approx(0.75)

        fusion_agg = loaded.aggregated("fusion")
        assert fusion_agg["overall_accuracy"] == pytest.approx(0.85)

    def test_writes_markdown_report(self, tmp_path: Path) -> None:
        patches = _patch_runners()
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            measure_baseline("companies", out_dir=tmp_path)

        md_path = tmp_path / "baseline_report.md"
        assert md_path.exists()
        text = md_path.read_text(encoding="utf-8")

        assert "Baseline report - companies" in text
        assert "Stage: sm" in text
        assert "Stage: em_blocking" in text
        assert "Stage: em_matching" in text
        assert "Stage: fusion" in text
        assert "SM - aggregated" in text

    def test_subset_stages(self, tmp_path: Path) -> None:
        patches = _patch_runners()
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            result = measure_baseline("companies", stages=["sm"], out_dir=tmp_path)

        assert "sm" in result["per_stage"]
        assert "em_blocking" not in result["per_stage"]
        assert "em_matching" not in result["per_stage"]
        assert "fusion" not in result["per_stage"]

    def test_fusion_input_member_auto_selected(self, tmp_path: Path) -> None:
        patches = _patch_runners()
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            result = measure_baseline("companies", out_dir=tmp_path)

        # embedding_rule has the highest F1 (0.78 > 0.72).
        assert result["meta"]["fusion_input_member"] == "embedding_rule"

    def test_fusion_input_member_override(self, tmp_path: Path) -> None:
        patches = _patch_runners()
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            result = measure_baseline(
                "companies",
                out_dir=tmp_path,
                fusion_input_member="token_rule",
            )

        assert result["meta"]["fusion_input_member"] == "token_rule"

    def test_fusion_input_member_empty_when_em_skipped(self, tmp_path: Path) -> None:
        patches = _patch_runners()
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            result = measure_baseline(
                "companies",
                stages=["sm", "fusion"],
                out_dir=tmp_path,
            )

        assert result["meta"]["fusion_input_member"] == ""

    def test_meta_contains_committee_versions(self, tmp_path: Path) -> None:
        patches = _patch_runners()
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            result = measure_baseline("companies", out_dir=tmp_path)

        versions = result["meta"]["committee_versions"]
        assert "sm" in versions
        assert "em_blocking" in versions
        assert "em_matching" in versions
        assert "fusion" in versions
        for stage_key in ("sm", "em_blocking", "em_matching", "fusion"):
            assert "@" in versions[stage_key]

    def test_per_attribute_in_json(self, tmp_path: Path) -> None:
        patches = _patch_runners()
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            measure_baseline("companies", out_dir=tmp_path)

        json_path = tmp_path / "baseline_metrics.json"
        loaded = load_baseline("companies", path_override=json_path)

        sm_attr = loaded.per_attribute("sm")
        assert "dbpedia.name" in sm_attr

        fusion_attr = loaded.per_attribute("fusion")
        assert "name" in fusion_attr
        assert fusion_attr["name"]["best_strategy_accuracy"] == pytest.approx(0.85)

    def test_per_partition_in_json(self, tmp_path: Path) -> None:
        patches = _patch_runners()
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            measure_baseline("companies", out_dir=tmp_path)

        json_path = tmp_path / "baseline_metrics.json"
        loaded = load_baseline("companies", path_override=json_path)

        em_part = loaded.per_partition("em_matching")
        assert "forbes_dbpedia" in em_part
        assert em_part["forbes_dbpedia"]["macro_f1"] == pytest.approx(0.76)


class TestBestEmMember:
    """Tests for the ``_best_em_member`` helper.

    ``_best_em_member`` is now called against the em_matching
    committee result (the EM stage split made em_matching the
    matching surface).
    """

    def test_selects_highest_f1(self) -> None:
        result = _fixture_em_matching_result()
        assert _best_em_member(result) == "embedding_rule"

    def test_empty_committee(self) -> None:
        result = CommitteeResult(
            stage="em_matching",
            domain="companies",
            level="baseline",
            per_member={},
            aggregated={},
        )
        assert _best_em_member(result) == ""


class TestParseStages:
    """Tests for the ``_parse_stages`` helper."""

    def test_single_stage(self) -> None:
        assert _parse_stages("sm") == ["sm"]

    def test_multiple_stages(self) -> None:
        assert _parse_stages("sm,em_matching,fusion") == [
            "sm",
            "em_matching",
            "fusion",
        ]

    def test_whitespace_tolerant(self) -> None:
        assert _parse_stages("sm , em_matching") == ["sm", "em_matching"]

    def test_invalid_stage_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown stage"):
            _parse_stages("sm,invalid")
