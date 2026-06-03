"""Tests for ``usecases_synthetic.scripts.validate_variant``.

Monkey-patches the three committee runners, ``load_variant``,
``load_baseline``, and ``_committee_version_string`` so the tests run
fast and without real pipeline data. Verifies the metrics.json shape
(with baseline + delta twins), CSV outputs, markdown rendering,
committee-version pinning, ``fusion_input_member`` resolution, and the
``--level baseline`` zero-delta sanity property.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from usecases_synthetic.lib.baseline_loader import BaselineMetrics
from usecases_synthetic.lib.committee import CommitteeResult, MemberResult
from usecases_synthetic.scripts.validate_variant import (
    _augment_flat_with_delta,
    _augment_stage_block,
    _parse_stages,
    validate_variant,
)

# ---------------------------------------------------------------------------
# Fixture committee results
# ---------------------------------------------------------------------------


def _sm_result(domain: str = "companies", level: str = "easy") -> CommitteeResult:
    return CommitteeResult(
        stage="sm",
        domain=domain,
        level=level,
        per_member={
            "label_jaccard": MemberResult(
                name="label_jaccard",
                predictions=None,
                metrics={"precision": 0.60, "recall": 0.55, "f1": 0.574},
                runtime_s=0.4,
                notes={"signal_type": "label"},
            ),
            "instance_tfidf": MemberResult(
                name="instance_tfidf",
                predictions=None,
                metrics={"precision": 0.50, "recall": 0.70, "f1": 0.583},
                runtime_s=1.0,
                notes={"signal_type": "instance"},
            ),
        },
        aggregated={
            "macro_f1": 0.58,
            "min_f1": 0.574,
            "max_f1": 0.583,
            "macro_precision": 0.55,
            "macro_recall": 0.625,
        },
        per_attribute={
            "dbpedia.name": {
                "label_jaccard": 1.0,
                "instance_tfidf": 1.0,
                "any_correct": 1.0,
            },
        },
        per_partition={
            "dbpedia": {"macro_f1": 0.60, "n_columns": 4.0},
        },
        runtime_s=1.4,
        roster=["label_jaccard", "instance_tfidf"],
    )


def _norm_result(
    domain: str = "companies",
    level: str = "easy",
) -> CommitteeResult:
    """Return a tiny Normalization committee result for the test patch stack."""
    return CommitteeResult(
        stage="norm",
        domain=domain,
        level=level,
        per_member={
            "text_clean": MemberResult(
                name="text_clean",
                predictions=None,
                metrics={
                    "precision": 0.55,
                    "recall": 0.50,
                    "f1": 0.524,
                    "macro_f1": 0.524,
                    "macro_precision": 0.55,
                    "macro_recall": 0.50,
                    "min_f1": 0.524,
                    "max_f1": 0.524,
                    "n_attributes": 1.0,
                    "n_cells": 10.0,
                },
                runtime_s=0.2,
                notes={"signal_type": "rule_string"},
            ),
        },
        aggregated={
            "macro_f1": 0.524,
            "min_f1": 0.524,
            "max_f1": 0.524,
            "macro_precision": 0.55,
            "macro_recall": 0.50,
            "best_member_f1": 0.524,
            "best_member_name_f1": 1.0,
        },
        per_attribute={
            "name": {
                "text_clean": 0.524,
                "any_correct": 1.0,
                "best_member_f1": 0.524,
            },
        },
        per_partition={
            "dbpedia": {"macro_f1": 0.524, "n_attributes": 1.0},
        },
        runtime_s=0.2,
        roster=["text_clean"],
    )


def _em_blocking_result(
    domain: str = "companies",
    level: str = "easy",
) -> CommitteeResult:
    """Return an em_blocking CommitteeResult.

    Mirrors the keys emitted by
    :class:`EMBlockingCommitteeRunner` post-2026-05-13 EM stage split:
    per-blocker ``pair_recall`` + ``reduction_ratio``; aggregated
    ``macro_pair_recall`` etc.
    """
    return CommitteeResult(
        stage="em_blocking",
        domain=domain,
        level=level,
        per_member={
            "token_blocker": MemberResult(
                name="token_blocker",
                predictions=None,
                metrics={
                    "pair_recall": 0.88,
                    "reduction_ratio": 0.96,
                },
                runtime_s=1.2,
                notes={"blocking_type": "lexical"},
            ),
            "embedding_blocker": MemberResult(
                name="embedding_blocker",
                predictions=None,
                metrics={
                    "pair_recall": 0.93,
                    "reduction_ratio": 0.94,
                },
                runtime_s=2.4,
                notes={"blocking_type": "embedding"},
            ),
        },
        aggregated={
            "macro_pair_recall": 0.905,
            "min_pair_recall": 0.88,
            "max_pair_recall": 0.93,
            "macro_reduction_ratio": 0.95,
            "best_member_name": "embedding_blocker",
            "best_member_pair_recall": 0.93,
            "best_member_reduction_ratio": 0.94,
            "recall_floor": 0.97,
        },
        per_attribute={},
        per_partition={
            "forbes_dbpedia": {
                "macro_pair_recall": 0.89,
                "macro_reduction_ratio": 0.95,
                "n_members": 2.0,
            },
        },
        runtime_s=3.6,
        roster=["token_blocker", "embedding_blocker"],
    )


def _em_matching_result(
    domain: str = "companies",
    level: str = "easy",
    *,
    retain: bool = True,
    retain_on: str = "embedding_rule",
) -> CommitteeResult:
    """Return an em_matching CommitteeResult.

    Mirrors the keys emitted by
    :class:`EMMatchingCommitteeRunner` post-2026-05-13 EM stage split.
    ``retain`` controls whether predictions are populated; ``retain_on``
    names which member carries them (mimics the runner's
    ``retain_predictions_for`` behaviour).
    """
    preds_per_pair: dict[str, pd.DataFrame] | None = None
    if retain:
        preds_per_pair = {
            "forbes_dbpedia": pd.DataFrame(
                {"id1": ["a"], "id2": ["b"], "score": [0.9]}
            ),
            "forbes_fullcontact": pd.DataFrame(
                {"id1": ["c"], "id2": ["d"], "score": [0.85]}
            ),
        }
    return CommitteeResult(
        stage="em_matching",
        domain=domain,
        level=level,
        per_member={
            "token_rule": MemberResult(
                name="token_rule",
                predictions=preds_per_pair if retain_on == "token_rule" else None,
                metrics={
                    "f1": 0.60,
                    "precision": 0.65,
                    "recall": 0.55,
                    "f1_baseline_test": 0.63,
                    "f1_regen_test": 0.60,
                    "pool_precision": 0.80,
                    "pool_recall": 0.20,
                    "pool_precision_baseline": 0.82,
                    "pool_precision_delta": -0.02,
                },
                runtime_s=2.0,
                notes={
                    "matching_type": "rule",
                    "per_pair": {
                        "forbes_dbpedia": {
                            "f1": 0.62,
                            "precision": 0.66,
                            "recall": 0.58,
                            "f1_baseline_test": 0.65,
                            "f1_regen_test": 0.62,
                            "pool_precision": 0.82,
                            "pool_recall": 0.22,
                        },
                        "forbes_fullcontact": {
                            "f1": 0.58,
                            "precision": 0.64,
                            "recall": 0.52,
                            "f1_baseline_test": 0.61,
                            "f1_regen_test": 0.58,
                            "pool_precision": 0.78,
                            "pool_recall": 0.18,
                        },
                    },
                },
            ),
            "embedding_rule": MemberResult(
                name="embedding_rule",
                predictions=preds_per_pair if retain_on == "embedding_rule" else None,
                metrics={
                    "f1": 0.68,
                    "precision": 0.70,
                    "recall": 0.66,
                    "f1_baseline_test": 0.71,
                    "f1_regen_test": 0.68,
                    "pool_precision": 0.75,
                    "pool_recall": 0.30,
                    "pool_precision_baseline": 0.78,
                    "pool_precision_delta": -0.03,
                },
                runtime_s=3.0,
                notes={
                    "matching_type": "rule",
                    "per_pair": {
                        "forbes_dbpedia": {
                            "f1": 0.70,
                            "precision": 0.72,
                            "recall": 0.68,
                            "f1_baseline_test": 0.73,
                            "f1_regen_test": 0.70,
                            "pool_precision": 0.77,
                            "pool_recall": 0.32,
                        },
                        "forbes_fullcontact": {
                            "f1": 0.66,
                            "precision": 0.68,
                            "recall": 0.64,
                            "f1_baseline_test": 0.69,
                            "f1_regen_test": 0.66,
                            "pool_precision": 0.73,
                            "pool_recall": 0.28,
                        },
                    },
                },
            ),
        },
        aggregated={
            "macro_f1": 0.64,
            "min_f1": 0.60,
            "max_f1": 0.68,
            "macro_precision": 0.675,
            "macro_recall": 0.605,
            "macro_f1_baseline_test": 0.67,
            "macro_f1_regen_test": 0.64,
            "best_member_name": "embedding_rule",
            "best_member_f1": 0.68,
        },
        per_attribute={},
        per_partition={
            "forbes_dbpedia": {
                "macro_f1": 0.66,
                "min_f1": 0.62,
                "max_f1": 0.70,
                "n_members": 2.0,
            },
            "forbes_fullcontact": {
                "macro_f1": 0.62,
                "min_f1": 0.58,
                "max_f1": 0.66,
                "n_members": 2.0,
            },
        },
        runtime_s=5.0,
        roster=["token_rule", "embedding_rule"],
    )


def _fusion_result(domain: str = "companies", level: str = "easy") -> CommitteeResult:
    return CommitteeResult(
        stage="fusion",
        domain=domain,
        level=level,
        per_member={
            "name_voting": MemberResult(
                name="name_voting",
                predictions=None,
                metrics={
                    "overall_accuracy": 0.72,
                    "macro_accuracy": 0.70,
                    "name_accuracy": 0.75,
                },
                runtime_s=1.5,
                notes={"attribute": "name", "strategy": "voting"},
            ),
            "name_longest_string": MemberResult(
                name="name_longest_string",
                predictions=None,
                metrics={
                    "overall_accuracy": 0.68,
                    "macro_accuracy": 0.65,
                    "name_accuracy": 0.70,
                },
                runtime_s=1.3,
                notes={"attribute": "name", "strategy": "longest_string"},
            ),
        },
        aggregated={
            "overall_accuracy": 0.72,
            "overall_mean_accuracy": 0.70,
            "overall_spread": 0.05,
        },
        per_attribute={
            "name": {
                "best_strategy_accuracy": 0.75,
                "mean_strategy_accuracy": 0.725,
                "spread": 0.05,
                "voting": 0.75,
                "longest_string": 0.70,
            },
        },
        per_partition={
            "primary": {
                "mean_best_accuracy": 0.75,
                "mean_spread": 0.05,
                "n_attributes": 1.0,
            },
        },
        runtime_s=2.8,
        roster=["name_voting", "name_longest_string"],
    )


# ---------------------------------------------------------------------------
# Baseline fixture
# ---------------------------------------------------------------------------


def _baseline_metrics(
    *,
    fusion_input_member: str = "embedding_rule",
    with_llm: bool = False,
    committee_versions: dict[str, str] | None = None,
) -> BaselineMetrics:
    """Return a ``BaselineMetrics`` with tiny but realistic per-stage blocks."""
    versions = committee_versions or {
        "sm": "sm_committee.yaml@aaaaaaaaaaaa",
        "norm": "normalization_committee_companies.yaml@dddddddddddd",
        "em_blocking": "em_blocking_committee.yaml@bbbbbbbbbbbb",
        "em_matching": "em_matching_committee.yaml@bbbbbbbbbbbb",
        "fusion": "fusion_committee.yaml@cccccccccccc",
    }
    sm = _sm_result(level="baseline").as_dict()
    # Baseline block should have higher numbers than the variant fixture
    # so that deltas are negative on the "easy" variant.
    sm["aggregated"] = {
        "macro_f1": 0.80,
        "min_f1": 0.78,
        "max_f1": 0.83,
        "macro_precision": 0.78,
        "macro_recall": 0.82,
    }
    sm["per_member"]["label_jaccard"]["metrics"] = {
        "precision": 0.85,
        "recall": 0.80,
        "f1": 0.824,
    }
    sm["per_member"]["instance_tfidf"]["metrics"] = {
        "precision": 0.70,
        "recall": 0.90,
        "f1": 0.788,
    }

    em_blocking = _em_blocking_result(level="baseline").as_dict()
    em_blocking["aggregated"] = {
        "macro_pair_recall": 0.96,
        "min_pair_recall": 0.94,
        "max_pair_recall": 0.98,
        "macro_reduction_ratio": 0.96,
        "best_member_name": "embedding_blocker",
        "best_member_pair_recall": 0.98,
        "best_member_reduction_ratio": 0.95,
        "recall_floor": 0.97,
    }
    em_blocking["per_member"]["token_blocker"]["metrics"] = {
        "pair_recall": 0.94,
        "reduction_ratio": 0.97,
    }
    em_blocking["per_member"]["embedding_blocker"]["metrics"] = {
        "pair_recall": 0.98,
        "reduction_ratio": 0.95,
    }

    em_matching = _em_matching_result(level="baseline", retain=False).as_dict()
    em_matching["aggregated"] = {
        "macro_f1": 0.75,
        "min_f1": 0.72,
        "max_f1": 0.78,
        "macro_precision": 0.775,
        "macro_recall": 0.735,
        "macro_f1_baseline_test": 0.77,
        "macro_f1_regen_test": 0.75,
        "best_member_name": "embedding_rule",
        "best_member_f1": 0.78,
    }
    em_matching["per_member"]["token_rule"]["metrics"] = {
        "f1": 0.72,
        "precision": 0.80,
        "recall": 0.65,
        "f1_baseline_test": 0.74,
        "f1_regen_test": 0.72,
        "pool_precision": 0.90,
        "pool_recall": 0.30,
    }
    em_matching["per_member"]["embedding_rule"]["metrics"] = {
        "f1": 0.78,
        "precision": 0.75,
        "recall": 0.82,
        "f1_baseline_test": 0.80,
        "f1_regen_test": 0.78,
        "pool_precision": 0.85,
        "pool_recall": 0.40,
    }

    fusion = _fusion_result(level="baseline").as_dict()
    fusion["aggregated"] = {
        "overall_accuracy": 0.82,
        "overall_mean_accuracy": 0.80,
        "overall_spread": 0.07,
    }
    fusion["per_attribute"]["name"] = {
        "best_strategy_accuracy": 0.85,
        "mean_strategy_accuracy": 0.815,
        "spread": 0.07,
        "voting": 0.85,
        "longest_string": 0.78,
    }

    norm = _norm_result(level="baseline").as_dict()
    norm["aggregated"] = {
        "macro_f1": 0.65,
        "min_f1": 0.65,
        "max_f1": 0.65,
        "macro_precision": 0.70,
        "macro_recall": 0.62,
        "best_member_f1": 0.65,
        "best_member_name_f1": 1.0,
    }
    norm["per_member"]["text_clean"]["metrics"] = {
        "precision": 0.70,
        "recall": 0.62,
        "f1": 0.65,
        "macro_f1": 0.65,
        "macro_precision": 0.70,
        "macro_recall": 0.62,
        "min_f1": 0.65,
        "max_f1": 0.65,
        "n_attributes": 1.0,
        "n_cells": 10.0,
    }

    return BaselineMetrics(
        domain="companies",
        per_stage={
            "sm": sm,
            "norm": norm,
            "em_blocking": em_blocking,
            "em_matching": em_matching,
            "fusion": fusion,
        },
        meta={
            "written_at": "2026-04-13T20:51:29+00:00",
            "with_llm": with_llm,
            "committee_versions": versions,
            "total_runtime_s": 68.45,
            "fusion_input_member": fusion_input_member,
        },
    )


# ---------------------------------------------------------------------------
# Patch helpers
# ---------------------------------------------------------------------------


def _make_patches(
    *,
    baseline: BaselineMetrics | None = None,
    variant_level: str = "easy",
    em_retain: bool = True,
    current_versions: dict[str, str] | None = None,
):
    """Return a stack of context managers patching validate_variant's deps."""
    baseline = baseline or _baseline_metrics()
    versions = current_versions or dict(baseline.meta.get("committee_versions", {}))

    def _version_fn(stage: str, domain: str) -> str:
        return versions[stage]

    sm_cls = MagicMock()
    sm_cls.return_value.run.return_value = _sm_result(level=variant_level)
    norm_cls = MagicMock()
    norm_cls.return_value.run.return_value = _norm_result(level=variant_level)
    # Honour the baseline's fusion_input_member so the em_matching mock
    # retains predictions on the right member (mimics real runner
    # behaviour).
    fusion_member = str(baseline.meta.get("fusion_input_member", "embedding_rule"))
    em_blocking_cls = MagicMock()
    em_blocking_cls.return_value.run.return_value = _em_blocking_result(
        level=variant_level,
    )
    em_matching_cls = MagicMock()
    em_matching_cls.return_value.run.return_value = _em_matching_result(
        level=variant_level,
        retain=em_retain,
        retain_on=fusion_member,
    )
    fusion_cls = MagicMock()
    fusion_cls.return_value.run.return_value = _fusion_result(level=variant_level)

    bundle = MagicMock()
    bundle.domain = "companies"
    bundle.level = variant_level
    load_variant_mock = MagicMock(return_value=bundle)

    load_baseline_mock = MagicMock(return_value=baseline)

    return (
        patch("usecases_synthetic.scripts.validate_variant.SMCommitteeRunner", sm_cls),
        patch(
            "usecases_synthetic.scripts.validate_variant.NormCommitteeRunner",
            norm_cls,
        ),
        patch(
            "usecases_synthetic.scripts.validate_variant.EMBlockingCommitteeRunner",
            em_blocking_cls,
        ),
        patch(
            "usecases_synthetic.scripts.validate_variant.EMMatchingCommitteeRunner",
            em_matching_cls,
        ),
        patch(
            "usecases_synthetic.scripts.validate_variant.FusionCommitteeRunner",
            fusion_cls,
        ),
        patch(
            "usecases_synthetic.scripts.validate_variant.load_variant",
            load_variant_mock,
        ),
        patch(
            "usecases_synthetic.scripts.validate_variant.load_baseline",
            load_baseline_mock,
        ),
        patch(
            "usecases_synthetic.scripts.validate_variant._committee_version_string",
            side_effect=_version_fn,
        ),
    )


def _enter_all(patches):
    """Enter every context manager in ``patches`` and return their results."""
    return [p.__enter__() for p in patches]


def _exit_all(patches):
    for p in reversed(patches):
        p.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# Tests: end-to-end metrics.json layout
# ---------------------------------------------------------------------------


class TestValidateVariantJson:
    """End-to-end metrics.json assertions."""

    def test_writes_metrics_json_for_easy(self, tmp_path: Path) -> None:
        patches = _make_patches(variant_level="easy")
        _enter_all(patches)
        try:
            payload = validate_variant("companies", "easy", out_dir=tmp_path)
        finally:
            _exit_all(patches)

        json_path = tmp_path / "metrics.json"
        assert json_path.exists()

        with open(json_path, encoding="utf-8") as f:
            doc = json.load(f)
        assert doc["domain"] == "companies"
        assert doc["meta"]["level"] == "easy"
        assert doc["meta"]["fusion_input_member"] == "embedding_rule"
        assert set(doc["per_stage"]) == {
            "sm",
            "norm",
            "em_blocking",
            "em_matching",
            "fusion",
        }
        # payload returned to caller matches what was written.
        assert payload["meta"]["level"] == "easy"

    def test_baseline_twins_populated_for_aggregated(self, tmp_path: Path) -> None:
        patches = _make_patches(variant_level="easy")
        _enter_all(patches)
        try:
            validate_variant("companies", "easy", out_dir=tmp_path)
        finally:
            _exit_all(patches)

        with open(tmp_path / "metrics.json", encoding="utf-8") as f:
            doc = json.load(f)
        sm_agg = doc["per_stage"]["sm"]["aggregated"]
        # Variant: macro_f1=0.58, baseline: 0.80 -> delta = -0.22
        assert sm_agg["macro_f1"] == pytest.approx(0.58)
        assert sm_agg["macro_f1_baseline"] == pytest.approx(0.80)
        assert sm_agg["macro_f1_delta"] == pytest.approx(-0.22, abs=1e-6)

    def test_em_per_member_has_pool_diagnostics(self, tmp_path: Path) -> None:
        patches = _make_patches(variant_level="easy")
        _enter_all(patches)
        try:
            validate_variant("companies", "easy", out_dir=tmp_path)
        finally:
            _exit_all(patches)

        with open(tmp_path / "metrics.json", encoding="utf-8") as f:
            doc = json.load(f)
        for member_name, member in doc["per_stage"]["em_matching"][
            "per_member"
        ].items():
            metrics = member["metrics"]
            assert "pool_precision" in metrics, member_name
            assert "pool_recall" in metrics, member_name
            assert "pool_precision_baseline" in metrics, member_name
            assert "pool_precision_delta" in metrics, member_name

    def test_fusion_input_member_recorded_in_meta(self, tmp_path: Path) -> None:
        baseline = _baseline_metrics(fusion_input_member="token_rule")
        patches = _make_patches(baseline=baseline, variant_level="easy")
        _enter_all(patches)
        try:
            payload = validate_variant(
                "companies",
                "easy",
                out_dir=tmp_path,
            )
        finally:
            _exit_all(patches)
        assert payload["meta"]["fusion_input_member"] == "token_rule"

    def test_norm_scoring_surface_read_from_baseline(self, tmp_path: Path) -> None:
        # The variant must be scored with the same surface the baseline
        # was measured with — read straight from baseline meta.
        baseline = _baseline_metrics()
        baseline.meta["scoring_surface"] = "schema_constraints"
        patches = _make_patches(baseline=baseline, variant_level="easy")
        entered = _enter_all(patches)
        try:
            payload = validate_variant("companies", "easy", out_dir=tmp_path)
        finally:
            _exit_all(patches)

        norm_cls = entered[1]  # SM=0, Norm=1, ... (see _make_patches order)
        assert norm_cls.call_args.kwargs["scoring_surface"] == "schema_constraints"
        assert payload["meta"]["scoring_surface"] == "schema_constraints"

    def test_norm_scoring_surface_defaults_xml_targets_for_legacy_baseline(
        self, tmp_path: Path
    ) -> None:
        # A baseline predating the surface knob carries no key; the variant
        # falls back to xml_targets so it matches the baseline's norm block.
        baseline = _baseline_metrics()
        baseline.meta.pop("scoring_surface", None)
        patches = _make_patches(baseline=baseline, variant_level="easy")
        entered = _enter_all(patches)
        try:
            payload = validate_variant("companies", "easy", out_dir=tmp_path)
        finally:
            _exit_all(patches)

        norm_cls = entered[1]
        assert norm_cls.call_args.kwargs["scoring_surface"] == "xml_targets"
        assert payload["meta"]["scoring_surface"] == "xml_targets"


# ---------------------------------------------------------------------------
# Tests: committee version pinning
# ---------------------------------------------------------------------------


class TestCommitteeVersionPinning:
    """The script must refuse to run when YAML hashes drift from baseline."""

    def test_refuses_on_mismatch(self, tmp_path: Path) -> None:
        baseline = _baseline_metrics(
            committee_versions={
                "sm": "sm_committee.yaml@aaaaaaaaaaaa",
                "norm": "normalization_committee_companies.yaml@dddddddddddd",
                "em_blocking": "em_blocking_committee.yaml@bbbbbbbbbbbb",
                "em_matching": "em_matching_committee.yaml@bbbbbbbbbbbb",
                "fusion": "fusion_committee.yaml@cccccccccccc",
            }
        )
        current = {
            "sm": "sm_committee.yaml@aaaaaaaaaaaa",
            "norm": "normalization_committee_companies.yaml@dddddddddddd",
            "em_blocking": "em_blocking_committee.yaml@bbbbbbbbbbbb",
            # Drift on the matching side — blocking hash matches but
            # matching hash differs.
            "em_matching": "em_matching_committee.yaml@DIFFERENT_HASH",
            "fusion": "fusion_committee.yaml@cccccccccccc",
        }
        patches = _make_patches(baseline=baseline, current_versions=current)
        _enter_all(patches)
        try:
            with pytest.raises(RuntimeError, match="Committee YAML drift"):
                validate_variant("companies", "easy", out_dir=tmp_path)
        finally:
            _exit_all(patches)

    def test_passes_on_match(self, tmp_path: Path) -> None:
        patches = _make_patches()
        _enter_all(patches)
        try:
            result = validate_variant("companies", "easy", out_dir=tmp_path)
        finally:
            _exit_all(patches)
        # Should not raise; provenance stored.
        assert "committee_versions" in result["meta"]


# ---------------------------------------------------------------------------
# Tests: --level baseline sanity (zero deltas)
# ---------------------------------------------------------------------------


class TestLevelBaselineSanity:
    """Running against level=baseline with matching results yields zero deltas."""

    def test_zero_deltas_when_variant_equals_baseline(self, tmp_path: Path) -> None:
        # Arrange: baseline AND variant return identical aggregated metrics.
        baseline = _baseline_metrics()

        # Override the committee mocks to return results whose aggregated
        # values are identical to baseline.
        def _aligned_sm() -> CommitteeResult:
            r = _sm_result(level="baseline")
            r.aggregated = dict(baseline.per_stage["sm"]["aggregated"])
            return r

        def _aligned_em_blocking() -> CommitteeResult:
            r = _em_blocking_result(level="baseline")
            r.aggregated = dict(baseline.per_stage["em_blocking"]["aggregated"])
            return r

        def _aligned_em_matching() -> CommitteeResult:
            r = _em_matching_result(level="baseline", retain=False)
            r.aggregated = dict(baseline.per_stage["em_matching"]["aggregated"])
            return r

        def _aligned_fusion() -> CommitteeResult:
            r = _fusion_result(level="baseline")
            r.aggregated = dict(baseline.per_stage["fusion"]["aggregated"])
            return r

        def _aligned_norm() -> CommitteeResult:
            r = _norm_result(level="baseline")
            r.aggregated = dict(baseline.per_stage["norm"]["aggregated"])
            return r

        sm_cls = MagicMock()
        sm_cls.return_value.run.return_value = _aligned_sm()
        norm_cls = MagicMock()
        norm_cls.return_value.run.return_value = _aligned_norm()
        em_blocking_cls = MagicMock()
        em_blocking_cls.return_value.run.return_value = _aligned_em_blocking()
        em_matching_cls = MagicMock()
        em_matching_cls.return_value.run.return_value = _aligned_em_matching()
        fusion_cls = MagicMock()
        fusion_cls.return_value.run.return_value = _aligned_fusion()

        versions = baseline.meta["committee_versions"]

        with (
            patch(
                "usecases_synthetic.scripts.validate_variant.SMCommitteeRunner",
                sm_cls,
            ),
            patch(
                "usecases_synthetic.scripts.validate_variant.NormCommitteeRunner",
                norm_cls,
            ),
            patch(
                "usecases_synthetic.scripts.validate_variant.EMBlockingCommitteeRunner",
                em_blocking_cls,
            ),
            patch(
                "usecases_synthetic.scripts.validate_variant.EMMatchingCommitteeRunner",
                em_matching_cls,
            ),
            patch(
                "usecases_synthetic.scripts.validate_variant.FusionCommitteeRunner",
                fusion_cls,
            ),
            patch(
                "usecases_synthetic.scripts.validate_variant.load_variant",
                MagicMock(return_value=MagicMock()),
            ),
            patch(
                "usecases_synthetic.scripts.validate_variant.load_baseline",
                MagicMock(return_value=baseline),
            ),
            patch(
                "usecases_synthetic.scripts.validate_variant._committee_version_string",
                side_effect=lambda s, d: versions[s],
            ),
        ):
            validate_variant("companies", "baseline", out_dir=tmp_path)

        with open(tmp_path / "metrics.json", encoding="utf-8") as f:
            doc = json.load(f)

        # All aggregated deltas should be ~0.0.
        for stage in ("sm", "norm", "em_blocking", "em_matching", "fusion"):
            agg = doc["per_stage"][stage]["aggregated"]
            for key, val in agg.items():
                if key.endswith("_delta"):
                    assert val == pytest.approx(0.0, abs=1e-9), f"{stage}.{key}={val}"


# ---------------------------------------------------------------------------
# Tests: markdown + CSV outputs
# ---------------------------------------------------------------------------


class TestReportArtifacts:
    """Check the markdown + CSV side-products."""

    def test_level_report_md_renders_stage_tables(self, tmp_path: Path) -> None:
        patches = _make_patches(variant_level="easy")
        _enter_all(patches)
        try:
            validate_variant("companies", "easy", out_dir=tmp_path)
        finally:
            _exit_all(patches)

        md = (tmp_path / "level_report.md").read_text(encoding="utf-8")
        assert "Validation report - companies / easy" in md
        assert "Stage summary" in md
        assert "Stage: sm - per member" in md
        assert "Stage: em_matching - per member" in md
        assert "Stage: em_matching - per pair" in md
        assert "Stage: fusion - per member" in md
        assert "Stage: fusion - per attribute" in md
        # Delta columns are present.
        assert "f1_delta" in md
        assert "delta" in md

    def test_em_per_pair_csv_written(self, tmp_path: Path) -> None:
        patches = _make_patches(variant_level="easy")
        _enter_all(patches)
        try:
            validate_variant("companies", "easy", out_dir=tmp_path)
        finally:
            _exit_all(patches)

        csv_path = tmp_path / "em_per_pair.csv"
        assert csv_path.exists()
        df = pd.read_csv(csv_path)
        expected_cols = {
            "member",
            "pair",
            "f1",
            "f1_baseline",
            "f1_delta",
            "precision",
            "recall",
            "pool_precision",
            "pool_recall",
            "pool_precision_baseline",
            "pool_recall_baseline",
        }
        assert expected_cols.issubset(df.columns)
        # 2 members x 2 pairs = 4 rows.
        assert len(df) == 4

    def test_fusion_per_attribute_csv_written(self, tmp_path: Path) -> None:
        patches = _make_patches(variant_level="easy")
        _enter_all(patches)
        try:
            validate_variant("companies", "easy", out_dir=tmp_path)
        finally:
            _exit_all(patches)

        csv_path = tmp_path / "fusion_per_attribute.csv"
        assert csv_path.exists()
        df = pd.read_csv(csv_path)
        expected_cols = {
            "attribute",
            "best_accuracy",
            "best_accuracy_baseline",
            "best_accuracy_delta",
            "spread",
            "spread_baseline",
            "spread_delta",
        }
        assert expected_cols.issubset(df.columns)
        assert "name" in df["attribute"].tolist()


# ---------------------------------------------------------------------------
# Tests: small helpers
# ---------------------------------------------------------------------------


class TestAugmentFlatWithDelta:
    def test_basic(self) -> None:
        out = _augment_flat_with_delta(
            {"f1": 0.5, "precision": 0.6},
            {"f1": 0.8, "precision": 0.7},
        )
        assert out["f1"] == pytest.approx(0.5)
        assert out["f1_baseline"] == pytest.approx(0.8)
        assert out["f1_delta"] == pytest.approx(-0.3)

    def test_keys_only_in_one_side_still_present(self) -> None:
        out = _augment_flat_with_delta({"a": 1.0}, {"b": 2.0})
        assert out["a"] == 1.0
        assert out["a_baseline"] == 0.0
        assert out["a_delta"] == 1.0
        assert out["b"] == 0.0
        assert out["b_baseline"] == 2.0
        assert out["b_delta"] == -2.0

    def test_non_numeric_values_skipped(self) -> None:
        out = _augment_flat_with_delta(
            {"f1": 0.5, "note": "hello"},
            {"f1": 0.7},
        )
        assert "note" not in out
        assert "note_delta" not in out


class TestAugmentStageBlock:
    def test_preserves_top_level_fields(self) -> None:
        measured = _sm_result(level="easy").as_dict()
        baseline = _sm_result(level="baseline").as_dict()
        out = _augment_stage_block(measured, baseline)
        assert out["stage"] == "sm"
        assert out["domain"] == "companies"
        assert out["level"] == "easy"
        assert set(out) >= {
            "stage",
            "domain",
            "level",
            "runtime_s",
            "roster",
            "aggregated",
            "per_attribute",
            "per_partition",
            "per_member",
        }


# ---------------------------------------------------------------------------
# Tests: with_llm mismatch + stage subset + parse_stages
# ---------------------------------------------------------------------------


class TestWithLlmAndStages:
    def test_with_llm_mismatch_raises(self, tmp_path: Path) -> None:
        baseline = _baseline_metrics(with_llm=False)
        patches = _make_patches(baseline=baseline)
        _enter_all(patches)
        try:
            with pytest.raises(RuntimeError, match="with_llm"):
                validate_variant(
                    "companies",
                    "easy",
                    with_llm=True,
                    out_dir=tmp_path,
                )
        finally:
            _exit_all(patches)

    def test_stage_subset_sm_only(self, tmp_path: Path) -> None:
        patches = _make_patches()
        _enter_all(patches)
        try:
            result = validate_variant(
                "companies",
                "easy",
                stages=["sm"],
                out_dir=tmp_path,
            )
        finally:
            _exit_all(patches)
        assert "sm" in result["per_stage"]
        assert "em_blocking" not in result["per_stage"]
        assert "em_matching" not in result["per_stage"]
        assert "fusion" not in result["per_stage"]

    def test_invalid_level_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Invalid level"):
            validate_variant("companies", "INVALID", out_dir=tmp_path)


class TestParseStages:
    def test_multiple(self) -> None:
        assert _parse_stages("sm,em_matching,fusion") == [
            "sm",
            "em_matching",
            "fusion",
        ]

    def test_whitespace(self) -> None:
        assert _parse_stages(" sm , em_blocking ") == ["sm", "em_blocking"]

    def test_invalid(self) -> None:
        with pytest.raises(ValueError, match="Unknown stage"):
            _parse_stages("sm,xxx")
