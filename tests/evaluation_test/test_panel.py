"""End-to-end panel orchestrator tests (v3 2D shape)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from PyDI.evaluation.panel import compute_e2e_panel
from PyDI.evaluation.silver_standard import SilverStandard


def _silver_membership(rows):
    return pd.DataFrame(rows, columns=["record_id", "source", "cluster_id"])


def _make_sources():
    src1 = pd.DataFrame(
        [
            {"id": "src1_1", "title": "Album A", "year": 1990},
            {"id": "src1_2", "title": "Album B", "year": 2000},
        ]
    )
    src1.attrs["dataset_name"] = "src1"
    src2 = pd.DataFrame(
        [
            {"id": "src2_1", "title": "Album A", "year": 1990},
            {"id": "src2_2", "title": "Album B", "year": 2000},
        ]
    )
    src2.attrs["dataset_name"] = "src2"
    return [src1, src2]


def _correspondences_perfect():
    return pd.DataFrame(
        [
            {"id1": "src1_1", "id2": "src2_1", "score": 1.0},
            {"id1": "src1_2", "id2": "src2_2", "score": 1.0},
        ]
    )


def _pipe_fused_perfect():
    return pd.DataFrame(
        [
            {"cluster_id": "group_0", "title": "Album A", "year": 1990},
            {"cluster_id": "group_1", "title": "Album B", "year": 2000},
        ]
    )


def _silver_aligned_to_pipe_groups():
    fused = pd.DataFrame(
        [
            {"cluster_id": "group_0", "title": "Album A", "year": 1990},
            {"cluster_id": "group_1", "title": "Album B", "year": 2000},
        ]
    )
    membership = _silver_membership(
        [
            ("src1_1", "src1", "group_0"),
            ("src2_1", "src2", "group_0"),
            ("src1_2", "src1", "group_1"),
            ("src2_2", "src2", "group_1"),
        ]
    )
    return SilverStandard(fused=fused, membership=membership, cell_provenance=None)


COLUMN_TYPES = {
    "title": "text",
    "year": "numerical",
    "cluster_id": "identifier",
}


class TestPanelIdentity:
    def test_perfect_pipeline_lands_perfect_panel(self, tmp_path: Path):
        result = compute_e2e_panel(
            pipe_fused=_pipe_fused_perfect(),
            correspondences_pipe=_correspondences_perfect(),
            sources_pipe=_make_sources(),
            silver=_silver_aligned_to_pipe_groups(),
            column_types=COLUMN_TYPES,
            pipe_source_id_column="id",
            pipe_id_column="cluster_id",
            usecase="toy",
            run_id="t0",
            silver_source_label="synthetic/toy",
        )
        assert result.panel["headline"]["SR"]["bcubed_f1"] == pytest.approx(1.0)
        assert result.panel["headline"]["SR"]["composite_score"] >= 0.99

        # v3 top-level keys
        assert set(result.panel.keys()) >= {
            "coverage",
            "consistency",
            "correctness",
            "task_step",
            "aggregated",
            "warnings",
        }

        # Identity: fact correctness is perfect
        fact_sr = result.panel["correctness"]["fact"]["SR"]
        assert fact_sr["macro_accuracy"] == pytest.approx(1.0)
        assert fact_sr["fully_correct_cluster_rate"] == pytest.approx(1.0)
        assert any("Source-attribution" in w for w in result.warnings)

        # Placeholders present
        assert result.panel["task_step"]["_placeholder"] is True
        assert result.panel["aggregated"]["_placeholder"] is True

        # Artifacts
        result.write(tmp_path)
        assert (tmp_path / "panel.json").exists()
        assert (tmp_path / "panel.csv").exists()
        assert (tmp_path / "cluster_alignment.csv").exists()
        assert (tmp_path / "cluster_attribute_correctness.csv").exists()
        composite_payload = json.loads((tmp_path / "composite_score.json").read_text())
        # composite_score.json is keyed by level — RF + SR present here.
        assert "RF" in composite_payload
        assert "SR" in composite_payload
        assert "GR" not in composite_payload
        assert composite_payload["SR"]["composite_score"] >= 0.99
        # Subscore names match brainstorm subdimensions (see docs/.../metrics.md)
        assert set(composite_payload["SR"]["weights"].keys()) == {
            "entity_coverage",
            "fact_coverage",
            "source_based_fact_coverage",
            "consistency",
            "cluster_correctness",
            "fact_correctness",
        }
        assert set(composite_payload["SR"]["subscores"].keys()) == set(
            composite_payload["SR"]["weights"].keys()
        )
        # RF composite carries only the three RF-applicable subscores.
        assert set(composite_payload["RF"]["weights"].keys()) == {
            "entity_coverage",
            "fact_coverage",
            "consistency",
        }
        assert "caveat" in composite_payload


class TestPanelCoverageEntity:
    def test_perfect_pipeline_recovers_all_entities(self):
        result = compute_e2e_panel(
            pipe_fused=_pipe_fused_perfect(),
            correspondences_pipe=_correspondences_perfect(),
            sources_pipe=_make_sources(),
            silver=_silver_aligned_to_pipe_groups(),
            column_types=COLUMN_TYPES,
            pipe_source_id_column="id",
            pipe_id_column="cluster_id",
        )
        sr = result.panel["coverage"]["entity"]["SR"]
        assert sr["n_recovered"] == 2
        assert sr["n_partial"] == 0
        assert sr["n_lost"] == 0
        assert sr["n_fabricated"] == 0
        assert sr["recovery_rate"] == pytest.approx(1.0)

        # RF subblock carries n_rows_output even on identity
        rf = result.panel["coverage"]["entity"]["RF"]
        assert rf["n_rows_output"] == 2

    def test_over_merge_lowers_recovery_rate(self):
        pipe = pd.DataFrame(
            [{"cluster_id": "group_0", "title": "Album A", "year": 1990}]
        )
        correspondences = pd.DataFrame(
            [
                {"id1": "src1_1", "id2": "src2_1", "score": 1.0},
                {"id1": "src1_1", "id2": "src1_2", "score": 1.0},
                {"id1": "src1_2", "id2": "src2_2", "score": 1.0},
            ]
        )
        result = compute_e2e_panel(
            pipe_fused=pipe,
            correspondences_pipe=correspondences,
            sources_pipe=_make_sources(),
            silver=_silver_aligned_to_pipe_groups(),
            column_types=COLUMN_TYPES,
            pipe_source_id_column="id",
            pipe_id_column="cluster_id",
        )
        sr = result.panel["coverage"]["entity"]["SR"]
        assert sr["n_recovered"] == 0
        assert sr["n_partial"] >= 1
        assert sr["recovery_rate"] < 1.0


class TestPanelCorrectnessCluster:
    def test_over_merge_drops_bcubed(self):
        pipe = pd.DataFrame(
            [{"cluster_id": "group_0", "title": "Album A", "year": 1990}]
        )
        correspondences = pd.DataFrame(
            [
                {"id1": "src1_1", "id2": "src2_1", "score": 1.0},
                {"id1": "src1_1", "id2": "src1_2", "score": 1.0},
                {"id1": "src1_2", "id2": "src2_2", "score": 1.0},
            ]
        )
        result = compute_e2e_panel(
            pipe_fused=pipe,
            correspondences_pipe=correspondences,
            sources_pipe=_make_sources(),
            silver=_silver_aligned_to_pipe_groups(),
            column_types=COLUMN_TYPES,
            pipe_source_id_column="id",
            pipe_id_column="cluster_id",
        )
        cc_sr = result.panel["correctness"]["cluster"]["SR"]
        assert cc_sr["bcubed"]["precision"] < 1.0

        # Source composition now lives under coverage.source_based.SR
        sb_sr = result.panel["coverage"]["source_based"]["SR"]
        assert "same_source_collision_rate" in sb_sr
        assert "per_source_coverage_rate" in sb_sr
        assert "source_mix_distribution_js" in sb_sr

    def test_same_source_collision_surfaces_under_coverage_source_based(self):
        # Same-source over-merge: 2 src1 records merged into one cluster
        pipe = pd.DataFrame(
            [{"cluster_id": "group_0", "title": "Album A", "year": 1990}]
        )
        correspondences = pd.DataFrame(
            [
                {"id1": "src1_1", "id2": "src1_2", "score": 1.0},
                {"id1": "src1_1", "id2": "src2_1", "score": 1.0},
                {"id1": "src1_2", "id2": "src2_2", "score": 1.0},
            ]
        )
        result = compute_e2e_panel(
            pipe_fused=pipe,
            correspondences_pipe=correspondences,
            sources_pipe=_make_sources(),
            silver=_silver_aligned_to_pipe_groups(),
            column_types=COLUMN_TYPES,
            pipe_source_id_column="id",
            pipe_id_column="cluster_id",
        )
        coll = result.panel["coverage"]["source_based"]["SR"][
            "same_source_collision_rate"
        ]
        assert coll["pipe"] > coll["reference"]


class TestPanelCorrectnessFact:
    def test_text_corruption_drops_accuracy(self):
        pipe = pd.DataFrame(
            [
                {"cluster_id": "group_0", "title": "WRONG_X", "year": 1990},
                {"cluster_id": "group_1", "title": "WRONG_Y", "year": 2000},
            ]
        )
        result = compute_e2e_panel(
            pipe_fused=pipe,
            correspondences_pipe=_correspondences_perfect(),
            sources_pipe=_make_sources(),
            silver=_silver_aligned_to_pipe_groups(),
            column_types=COLUMN_TYPES,
            pipe_source_id_column="id",
            pipe_id_column="cluster_id",
        )
        fact_sr = result.panel["correctness"]["fact"]["SR"]
        assert fact_sr["macro_accuracy"] < 1.0
        title = fact_sr["per_attribute"]["title"]
        # Normalization fingerprint is attached for text attributes
        assert "mismatch_fingerprint" in title
        assert "accuracy_similarity_gap" in title

    def test_list_attribute_set_f1_simplified(self):
        pipe = pd.DataFrame(
            [
                {
                    "cluster_id": "group_0",
                    "title": "Album A",
                    "year": 1990,
                    "tracks": ["t1"],
                },
                {
                    "cluster_id": "group_1",
                    "title": "Album B",
                    "year": 2000,
                    "tracks": ["t1", "t2"],
                },
            ]
        )
        silver = _silver_aligned_to_pipe_groups()
        silver.fused["tracks"] = [["t1", "t2"], ["t1", "t2"]]
        result = compute_e2e_panel(
            pipe_fused=pipe,
            correspondences_pipe=_correspondences_perfect(),
            sources_pipe=_make_sources(),
            silver=silver,
            column_types={**COLUMN_TYPES, "tracks": "list"},
            pipe_source_id_column="id",
            pipe_id_column="cluster_id",
        )
        list_metrics = result.panel["correctness"]["fact"]["SR"][
            "list_attribute_set_metrics"
        ]
        # v2 simplification: only set_f1 + set_jaccard + count are surfaced
        assert set(list_metrics["tracks"].keys()) == {"set_f1", "set_jaccard", "count"}
        assert list_metrics["tracks"]["set_f1"] < 1.0


class TestPanelSchemaDiffAsAuditArtifact:
    def test_schema_diff_surfaces_pipe_extra_column(self):
        pipe = pd.DataFrame(
            [
                {
                    "cluster_id": "group_0",
                    "title": "Album A",
                    "year": 1990,
                    "preview_url": "http://x",
                },
                {
                    "cluster_id": "group_1",
                    "title": "Album B",
                    "year": 2000,
                    "preview_url": "http://y",
                },
            ]
        )
        result = compute_e2e_panel(
            pipe_fused=pipe,
            correspondences_pipe=_correspondences_perfect(),
            sources_pipe=_make_sources(),
            silver=_silver_aligned_to_pipe_groups(),
            column_types={**COLUMN_TYPES, "preview_url": "text"},
            pipe_source_id_column="id",
            pipe_id_column="cluster_id",
        )
        # Schema diff audit artifact still surfaces the extra column
        assert "preview_url" in result.schema_diff["columns_pipe_only"]
        assert (
            "preview_url"
            in result.schema_diff["skipped_columns_for_per_column_metrics"]
        )


class TestPanelDiagnosticWarnings:
    def test_id_mismatch_warning(self):
        pipe = pd.DataFrame(
            [
                {"cluster_id": "fused_xyz", "title": "A", "year": 1990},
                {"cluster_id": "fused_abc", "title": "B", "year": 2000},
            ]
        )
        result = compute_e2e_panel(
            pipe_fused=pipe,
            correspondences_pipe=_correspondences_perfect(),
            sources_pipe=_make_sources(),
            silver=_silver_aligned_to_pipe_groups(),
            column_types=COLUMN_TYPES,
            pipe_source_id_column="id",
            pipe_id_column="cluster_id",
        )
        assert any("do not overlap" in w for w in result.warnings)


class TestClusterAttributeCorrectnessCSV:
    def test_csv_emitted_with_per_cluster_rows(self, tmp_path: Path):
        result = compute_e2e_panel(
            pipe_fused=_pipe_fused_perfect(),
            correspondences_pipe=_correspondences_perfect(),
            sources_pipe=_make_sources(),
            silver=_silver_aligned_to_pipe_groups(),
            column_types=COLUMN_TYPES,
            pipe_source_id_column="id",
            pipe_id_column="cluster_id",
        )
        df = result.cluster_attribute_correctness
        assert not df.empty
        assert {"silver_cluster_id", "pipe_cluster_id"} <= set(df.columns)
        assert "n_attributes_correct" in df.columns
        assert "fully_correct" in df.columns
        # Every cluster fully correct on identity
        assert df["fully_correct"].all()

        result.write(tmp_path)
        assert (tmp_path / "cluster_attribute_correctness.csv").exists()


class TestColumnMetricsMissingnessFingerprint:
    def test_nan_rate_delta_surfaces(self):
        pipe = pd.DataFrame(
            [
                {"cluster_id": "group_0", "title": None, "year": 1990},
                {"cluster_id": "group_1", "title": "Album B", "year": 2000},
            ]
        )
        result = compute_e2e_panel(
            pipe_fused=pipe,
            correspondences_pipe=_correspondences_perfect(),
            sources_pipe=_make_sources(),
            silver=_silver_aligned_to_pipe_groups(),
            column_types=COLUMN_TYPES,
            pipe_source_id_column="id",
            pipe_id_column="cluster_id",
        )
        title_rows = result.column_metrics[
            (result.column_metrics["column"] == "title")
            & (result.column_metrics["metric"] == "nan_rate_delta")
        ]
        assert not title_rows.empty
        assert title_rows.iloc[0]["value"] > 0


class TestPanelConsistency:
    def test_validity_surfaces_under_consistency_sr(self):
        result = compute_e2e_panel(
            pipe_fused=_pipe_fused_perfect(),
            correspondences_pipe=_correspondences_perfect(),
            sources_pipe=_make_sources(),
            silver=_silver_aligned_to_pipe_groups(),
            column_types=COLUMN_TYPES,
            pipe_source_id_column="id",
            pipe_id_column="cluster_id",
        )
        cons_sr = result.panel["consistency"]["SR"]
        assert "validity_per_column" in cons_sr
        assert "mean_validity_delta" in cons_sr
        assert cons_sr["mean_validity_delta"] == pytest.approx(0.0)
        assert "_design_extensions_pending" in result.panel["consistency"]

    def test_constraint_violation_drops_validity_and_warns(self):
        # Pipe emits a year outside the declared [1900, 2030] range
        pipe = pd.DataFrame(
            [
                {"cluster_id": "group_0", "title": "Album A", "year": 1990},
                {"cluster_id": "group_1", "title": "Album B", "year": 9999},
            ]
        )
        result = compute_e2e_panel(
            pipe_fused=pipe,
            correspondences_pipe=_correspondences_perfect(),
            sources_pipe=_make_sources(),
            silver=_silver_aligned_to_pipe_groups(),
            column_types=COLUMN_TYPES,
            column_constraints={"year": {"range": [1900, 2030]}},
            pipe_source_id_column="id",
            pipe_id_column="cluster_id",
        )
        year_validity = result.panel["consistency"]["SR"]["validity_per_column"]["year"]
        assert year_validity["validity_rate_reference"] == pytest.approx(1.0)
        assert year_validity["validity_rate_pipe"] == pytest.approx(0.5)
        assert year_validity["delta"] < 0
        assert any(
            "Constraint-validity dropped" in w for w in result.warnings
        ), f"expected constraint-validity warning, got: {result.warnings}"


class TestPanelSemanticValueMatching:
    def test_semantic_callable_emits_semantic_accuracy_and_confirms_fingerprint(
        self,
    ):
        pipe = pd.DataFrame(
            [
                {"cluster_id": "group_0", "title": "United States", "year": 1990},
                {"cluster_id": "group_1", "title": "DE", "year": 2000},
            ]
        )

        def fake_sim(a: str, b: str) -> float:
            return 0.99

        result = compute_e2e_panel(
            pipe_fused=pipe,
            correspondences_pipe=_correspondences_perfect(),
            sources_pipe=_make_sources(),
            silver=_silver_aligned_to_pipe_groups(),
            column_types=COLUMN_TYPES,
            semantic_value_similarity=fake_sim,
            pipe_source_id_column="id",
            pipe_id_column="cluster_id",
        )
        title = result.panel["correctness"]["fact"]["SR"]["per_attribute"]["title"]
        assert title["accuracy"] == pytest.approx(0.0)
        assert title["semantic_accuracy"] == pytest.approx(1.0)
        assert title["mismatch_fingerprint"] == "normalization_difference_confirmed"
        assert any(
            "Normalization differences *confirmed*" in w for w in result.warnings
        )

    def test_no_semantic_callable_keeps_suspected_fingerprint(self):
        pipe = pd.DataFrame(
            [
                {"cluster_id": "group_0", "title": "Album A_suffix", "year": 1990},
                {"cluster_id": "group_1", "title": "Album B_suffix", "year": 2000},
            ]
        )
        result = compute_e2e_panel(
            pipe_fused=pipe,
            correspondences_pipe=_correspondences_perfect(),
            sources_pipe=_make_sources(),
            silver=_silver_aligned_to_pipe_groups(),
            column_types=COLUMN_TYPES,
            pipe_source_id_column="id",
            pipe_id_column="cluster_id",
        )
        title = result.panel["correctness"]["fact"]["SR"]["per_attribute"]["title"]
        assert title["mismatch_fingerprint"] == "normalization_difference_suspected"
        assert "semantic_accuracy" not in title


class TestPanelResourceUsage:
    def test_resource_usage_omitted_when_not_provided(self):
        result = compute_e2e_panel(
            pipe_fused=_pipe_fused_perfect(),
            correspondences_pipe=_correspondences_perfect(),
            sources_pipe=_make_sources(),
            silver=_silver_aligned_to_pipe_groups(),
            column_types=COLUMN_TYPES,
            pipe_source_id_column="id",
            pipe_id_column="cluster_id",
        )
        assert "resource_usage" not in result.panel

    def test_resource_usage_present_when_provided(self):
        result = compute_e2e_panel(
            pipe_fused=_pipe_fused_perfect(),
            correspondences_pipe=_correspondences_perfect(),
            sources_pipe=_make_sources(),
            silver=_silver_aligned_to_pipe_groups(),
            column_types=COLUMN_TYPES,
            pipe_source_id_column="id",
            pipe_id_column="cluster_id",
            pipeline_duration_seconds=123.4,
            pipeline_peak_memory_mb=512.0,
            pipeline_api_cost=2.50,
            pipeline_api_cost_currency="EUR",
        )
        ru = result.panel["resource_usage"]
        assert ru["duration_seconds"] == pytest.approx(123.4)
        assert ru["peak_memory_mb"] == pytest.approx(512.0)
        assert ru["api_cost"] == pytest.approx(2.50)
        assert ru["api_cost_currency"] == "EUR"

    def test_partial_resource_usage(self):
        result = compute_e2e_panel(
            pipe_fused=_pipe_fused_perfect(),
            correspondences_pipe=_correspondences_perfect(),
            sources_pipe=_make_sources(),
            silver=_silver_aligned_to_pipe_groups(),
            column_types=COLUMN_TYPES,
            pipe_source_id_column="id",
            pipe_id_column="cluster_id",
            pipeline_duration_seconds=42.0,
        )
        ru = result.panel["resource_usage"]
        assert ru == {"duration_seconds": 42.0}

    def test_resource_usage_api_tokens_present_when_provided(self):
        result = compute_e2e_panel(
            pipe_fused=_pipe_fused_perfect(),
            correspondences_pipe=_correspondences_perfect(),
            sources_pipe=_make_sources(),
            silver=_silver_aligned_to_pipe_groups(),
            column_types=COLUMN_TYPES,
            pipe_source_id_column="id",
            pipe_id_column="cluster_id",
            pipeline_api_tokens={
                "n_calls": 3012,
                "input_tokens": 18485598,
                "output_tokens": 1442858,
                "total_tokens": 19928456,
            },
            pipeline_api_notes=(
                "counted: ie=always-on (19928456 tokens, 3012 calls) | "
                "skipped: sm=winner=duplicate_majority non-LLM "
                "(122395 tokens, 4 calls)"
            ),
        )
        ru = result.panel["resource_usage"]
        assert ru["api_tokens"] == {
            "n_calls": 3012,
            "input_tokens": 18485598,
            "output_tokens": 1442858,
            "total_tokens": 19928456,
        }
        assert "ie=always-on" in ru["api_notes"]
        assert "skipped: sm=winner=duplicate_majority non-LLM" in ru["api_notes"]

    def test_resource_usage_absent_when_no_cost_fields(self):
        # No api_tokens and no other cost fields → resource_usage block
        # omitted entirely.
        result = compute_e2e_panel(
            pipe_fused=_pipe_fused_perfect(),
            correspondences_pipe=_correspondences_perfect(),
            sources_pipe=_make_sources(),
            silver=_silver_aligned_to_pipe_groups(),
            column_types=COLUMN_TYPES,
            pipe_source_id_column="id",
            pipe_id_column="cluster_id",
        )
        assert "resource_usage" not in result.panel

    def test_api_notes_round_trips_without_tokens(self):
        # api_notes alone is enough to surface the resource_usage block
        # (the panel doesn't gate notes on having tokens).
        result = compute_e2e_panel(
            pipe_fused=_pipe_fused_perfect(),
            correspondences_pipe=_correspondences_perfect(),
            sources_pipe=_make_sources(),
            silver=_silver_aligned_to_pipe_groups(),
            column_types=COLUMN_TYPES,
            pipe_source_id_column="id",
            pipe_id_column="cluster_id",
            pipeline_api_notes="no LLM steps in this pipeline",
        )
        ru = result.panel["resource_usage"]
        assert ru == {"api_notes": "no LLM steps in this pipeline"}
        assert "api_tokens" not in ru

    def test_api_tokens_empty_dict_omitted(self):
        # An empty mapping should not emit api_tokens (peer of api_cost's
        # None-skip behavior).
        result = compute_e2e_panel(
            pipe_fused=_pipe_fused_perfect(),
            correspondences_pipe=_correspondences_perfect(),
            sources_pipe=_make_sources(),
            silver=_silver_aligned_to_pipe_groups(),
            column_types=COLUMN_TYPES,
            pipe_source_id_column="id",
            pipe_id_column="cluster_id",
            pipeline_api_tokens={},
        )
        assert "resource_usage" not in result.panel


class TestPanelTaskStepAndAggregatedPlaceholders:
    def test_placeholders_present_and_well_formed(self):
        result = compute_e2e_panel(
            pipe_fused=_pipe_fused_perfect(),
            correspondences_pipe=_correspondences_perfect(),
            sources_pipe=_make_sources(),
            silver=_silver_aligned_to_pipe_groups(),
            column_types=COLUMN_TYPES,
            pipe_source_id_column="id",
            pipe_id_column="cluster_id",
        )
        ts = result.panel["task_step"]
        assert ts["_placeholder"] is True
        assert "_design_intent" in ts

        agg = result.panel["aggregated"]
        assert agg["_placeholder"] is True
        assert "_design_intent" in agg

    def test_task_step_metrics_replace_placeholder(self):
        per_stage = {
            "schema_matching": {"f1": 0.91, "n_pairs": 250},
            "blocking": {"pair_completeness": 0.98, "reduction_ratio": 0.997},
            "entity_matching": {"f1": 0.88},
        }
        result = compute_e2e_panel(
            pipe_fused=_pipe_fused_perfect(),
            correspondences_pipe=_correspondences_perfect(),
            sources_pipe=_make_sources(),
            silver=_silver_aligned_to_pipe_groups(),
            column_types=COLUMN_TYPES,
            pipe_source_id_column="id",
            pipe_id_column="cluster_id",
            task_step_metrics=per_stage,
        )
        ts = result.panel["task_step"]
        assert ts == per_stage
        assert "_placeholder" not in ts


def _gold_subset_of_silver():
    """A small gold reference — half the silver, same cluster ids."""
    fused = pd.DataFrame([{"cluster_id": "group_0", "title": "Album A", "year": 1990}])
    membership = _silver_membership(
        [
            ("src1_1", "src1", "group_0"),
            ("src2_1", "src2", "group_0"),
        ]
    )
    return SilverStandard(fused=fused, membership=membership, cell_provenance=None)


class TestPanelRFOnlyMode:
    """RF-only mode — caller supplies no silver and no gold."""

    def test_no_reference_emits_only_rf_blocks(self):
        result = compute_e2e_panel(
            pipe_fused=_pipe_fused_perfect(),
            sources_pipe=_make_sources(),
            column_types=COLUMN_TYPES,
        )
        # coverage sections have RF only — no SR, no GR
        for section in ("entity", "fact", "source_based"):
            block = result.panel["coverage"][section]
            assert "RF" in block
            assert "SR" not in block
            assert "GR" not in block

        # consistency has only RF
        cons = result.panel["consistency"]
        assert "RF" in cons
        assert "SR" not in cons
        assert "GR" not in cons

        # correctness section is empty (no reference → no signal)
        assert result.panel["correctness"] == {}

    def test_no_reference_emits_rf_composite_only(self, tmp_path: Path):
        result = compute_e2e_panel(
            pipe_fused=_pipe_fused_perfect(),
            sources_pipe=_make_sources(),
            column_types=COLUMN_TYPES,
        )
        # RF composite is always computed (structural-only).
        assert result.composite is not None
        assert "RF" in result.composite
        assert "SR" not in result.composite
        assert "GR" not in result.composite
        rf_composite = result.composite["RF"]["composite_score"]
        assert isinstance(rf_composite, float)
        assert 0.0 <= rf_composite <= 1.0

        # Headline carries the RF composite_score; no SR/GR sub-block.
        assert result.panel["headline"]["RF"]["composite_score"] == pytest.approx(
            rf_composite
        )
        assert "SR" not in result.panel["headline"]
        assert "GR" not in result.panel["headline"]

        # E2EPanel reference-bearing optional fields are None
        assert result.schema_diff is None
        assert result.column_metrics is None
        assert result.cluster_alignment_table is None
        assert result.cluster_attribute_correctness is None

        # write() emits panel.json + panel.csv + composite_score.json
        # (the RF-only composite still surfaces a ranking number).
        result.write(tmp_path)
        assert (tmp_path / "panel.json").exists()
        assert (tmp_path / "panel.csv").exists()
        assert (tmp_path / "composite_score.json").exists()
        composite_payload = json.loads((tmp_path / "composite_score.json").read_text())
        assert set(composite_payload.keys()) == {"RF", "caveat"}
        assert not (tmp_path / "schema_diff.json").exists()
        assert not (tmp_path / "column_metrics.csv").exists()
        assert not (tmp_path / "cluster_alignment.csv").exists()
        assert not (tmp_path / "cluster_attribute_correctness.csv").exists()

    def test_no_reference_warns(self):
        result = compute_e2e_panel(
            pipe_fused=_pipe_fused_perfect(),
            sources_pipe=_make_sources(),
            column_types=COLUMN_TYPES,
        )
        assert any(
            "No reference supplied" in w for w in result.warnings
        ), f"expected RF-only warning, got: {result.warnings}"

    def test_rf_consistency_pipe_only_validity(self):
        # Pipe with one invalid year (9999) under a [1900, 2030] constraint
        pipe = pd.DataFrame(
            [
                {"cluster_id": "g0", "title": "A", "year": 1990},
                {"cluster_id": "g1", "title": "B", "year": 9999},
            ]
        )
        result = compute_e2e_panel(
            pipe_fused=pipe,
            sources_pipe=_make_sources(),
            column_types=COLUMN_TYPES,
            column_constraints={"year": {"range": [1900, 2030]}},
        )
        validity = result.panel["consistency"]["RF"]["validity_per_column"]
        assert validity["year"]["validity_rate_pipe"] == pytest.approx(0.5)
        assert validity["year"]["constraint_failures_pipe"] == 1


class TestPanelReferenceFree:
    def test_rf_entity_block_present(self):
        result = compute_e2e_panel(
            pipe_fused=_pipe_fused_perfect(),
            correspondences_pipe=_correspondences_perfect(),
            sources_pipe=_make_sources(),
            silver=_silver_aligned_to_pipe_groups(),
            column_types=COLUMN_TYPES,
            pipe_source_id_column="id",
            pipe_id_column="cluster_id",
        )
        rf = result.panel["coverage"]["entity"]["RF"]
        # 2 fused rows, each input has 2 rows → row_gain = 0
        assert rf["n_rows_output"] == 2
        assert rf["n_rows_largest_input"] == 2
        assert rf["row_gain_vs_largest_input"] == pytest.approx(0.0)

    def test_rf_fact_density_computed(self):
        # All non-null fields in fused → density = 1.0
        result = compute_e2e_panel(
            pipe_fused=_pipe_fused_perfect(),
            correspondences_pipe=_correspondences_perfect(),
            sources_pipe=_make_sources(),
            silver=_silver_aligned_to_pipe_groups(),
            column_types=COLUMN_TYPES,
            pipe_source_id_column="id",
            pipe_id_column="cluster_id",
        )
        rf = result.panel["coverage"]["fact"]["RF"]
        assert rf["density_output"] == pytest.approx(1.0)

    def test_rf_source_based_winning_distribution_from_fusion_metadata(self):
        # When pipe_fused has _fusion_metadata, the panel auto-builds
        # cell_provenance and the RF winning_source_distribution surfaces.
        pipe_fused = _pipe_fused_perfect().assign(
            _fusion_metadata=[
                {"title_sources": ["src1_1"], "year_sources": ["src1_1", "src2_1"]},
                {"title_sources": ["src2_2"], "year_sources": ["src2_2"]},
            ]
        )
        result = compute_e2e_panel(
            pipe_fused=pipe_fused,
            correspondences_pipe=_correspondences_perfect(),
            sources_pipe=_make_sources(),
            silver=_silver_aligned_to_pipe_groups(),
            column_types=COLUMN_TYPES,
            pipe_source_id_column="id",
            pipe_id_column="cluster_id",
            source_prefix_map={"src1_": "src1", "src2_": "src2"},
        )
        rf_sb = result.panel["coverage"]["source_based"]["RF"]
        assert "winning_source_distribution_per_attribute" in rf_sb
        dist = rf_sb["winning_source_distribution_per_attribute"]
        # title: src1 wins one cell, src2 wins one → 0.5/0.5
        assert dist["title"]["src1"] == pytest.approx(0.5)
        assert dist["title"]["src2"] == pytest.approx(0.5)
        # year: cell 1 split 0.5/0.5 between src1 and src2; cell 2 all src2
        #   → src1 = 0.5/2 = 0.25, src2 = (0.5 + 1)/2 = 0.75
        assert dist["year"]["src1"] == pytest.approx(0.25)
        assert dist["year"]["src2"] == pytest.approx(0.75)

    def test_rf_source_based_absent_when_no_fusion_metadata(self):
        # Without _fusion_metadata in pipe_fused, winning_source_distribution
        # is not present in the RF block.
        result = compute_e2e_panel(
            pipe_fused=_pipe_fused_perfect(),
            correspondences_pipe=_correspondences_perfect(),
            sources_pipe=_make_sources(),
            silver=_silver_aligned_to_pipe_groups(),
            column_types=COLUMN_TYPES,
            pipe_source_id_column="id",
            pipe_id_column="cluster_id",
        )
        rf_sb = result.panel["coverage"]["source_based"]["RF"]
        assert "winning_source_distribution_per_attribute" not in rf_sb


class TestPanelGlossaryCompanion:
    """The panel writer copies the canonical glossary alongside panel.json."""

    def test_glossary_emitted_alongside_panel_json(self, tmp_path: Path):
        result = compute_e2e_panel(
            pipe_fused=_pipe_fused_perfect(),
            correspondences_pipe=_correspondences_perfect(),
            sources_pipe=_make_sources(),
            silver=_silver_aligned_to_pipe_groups(),
            column_types=COLUMN_TYPES,
            pipe_source_id_column="id",
            pipe_id_column="cluster_id",
        )
        result.write(tmp_path)

        panel_path = tmp_path / "panel.json"
        glossary_path = tmp_path / "panel_glossary.json"
        assert panel_path.exists()
        assert glossary_path.exists()

        panel = json.loads(panel_path.read_text())
        glossary = json.loads(glossary_path.read_text())

        # Sample paths in panel.json have non-empty string descriptions
        # at the same nested path in the glossary.
        sample_paths = [
            ("coverage", "entity", "SR", "recovery_rate"),
            ("correctness", "fact", "SR", "macro_accuracy"),
        ]
        for path in sample_paths:
            panel_block: object = panel
            for key in path:
                assert isinstance(
                    panel_block, dict
                ), f"panel path {path} broke at {key}"
                assert key in panel_block, f"panel missing {path}"
                panel_block = panel_block[key]

            glossary_block: object = glossary
            for key in path:
                assert isinstance(
                    glossary_block, dict
                ), f"glossary path {path} broke at {key}"
                assert key in glossary_block, f"glossary missing {path}"
                glossary_block = glossary_block[key]
            assert isinstance(glossary_block, str)
            assert glossary_block.strip()

    def test_glossary_emitted_in_rf_only_mode(self, tmp_path: Path):
        result = compute_e2e_panel(
            pipe_fused=_pipe_fused_perfect(),
            sources_pipe=_make_sources(),
            column_types=COLUMN_TYPES,
        )
        result.write(tmp_path)
        assert (tmp_path / "panel_glossary.json").exists()


class TestPanelGoldReference:
    def test_gr_blocks_absent_when_no_gold(self):
        result = compute_e2e_panel(
            pipe_fused=_pipe_fused_perfect(),
            correspondences_pipe=_correspondences_perfect(),
            sources_pipe=_make_sources(),
            silver=_silver_aligned_to_pipe_groups(),
            column_types=COLUMN_TYPES,
            pipe_source_id_column="id",
            pipe_id_column="cluster_id",
        )
        # SR blocks present, GR blocks absent
        assert "SR" in result.panel["coverage"]["entity"]
        assert "GR" not in result.panel["coverage"]["entity"]
        assert "GR" not in result.panel["coverage"]["fact"]
        assert "GR" not in result.panel["coverage"]["source_based"]
        assert "GR" not in result.panel["consistency"]
        assert "GR" not in result.panel["correctness"]["cluster"]
        assert "GR" not in result.panel["correctness"]["fact"]
        assert "GR" not in result.panel["headline"]

    def test_gr_blocks_present_when_gold_provided(self):
        result = compute_e2e_panel(
            pipe_fused=_pipe_fused_perfect(),
            correspondences_pipe=_correspondences_perfect(),
            sources_pipe=_make_sources(),
            silver=_silver_aligned_to_pipe_groups(),
            gold=_gold_subset_of_silver(),
            column_types=COLUMN_TYPES,
            pipe_source_id_column="id",
            pipe_id_column="cluster_id",
            gold_source_label="usecases/toy/input/fusion/test_set.xml",
        )
        # GR blocks land under every applicable section
        for path in [
            ("coverage", "entity", "GR"),
            ("coverage", "fact", "GR"),
            ("coverage", "source_based", "GR"),
            ("consistency", "GR"),
            ("correctness", "cluster", "GR"),
            ("correctness", "fact", "GR"),
        ]:
            block = result.panel
            for k in path:
                assert k in block, f"missing path {path}"
                block = block[k]

        # GR-flavoured headline scores
        assert "bcubed_f1" in result.panel["headline"]["GR"]
        assert "macro_accuracy" in result.panel["headline"]["GR"]
        assert result.panel["gold_source"].endswith("test_set.xml")

    def test_gr_artifacts_written_when_gold_provided(self, tmp_path: Path):
        result = compute_e2e_panel(
            pipe_fused=_pipe_fused_perfect(),
            correspondences_pipe=_correspondences_perfect(),
            sources_pipe=_make_sources(),
            silver=_silver_aligned_to_pipe_groups(),
            gold=_gold_subset_of_silver(),
            column_types=COLUMN_TYPES,
            pipe_source_id_column="id",
            pipe_id_column="cluster_id",
        )
        result.write(tmp_path)
        assert (tmp_path / "cluster_alignment.csv").exists()
        assert (tmp_path / "cluster_alignment_gold.csv").exists()
        assert (tmp_path / "cluster_attribute_correctness.csv").exists()
        assert (tmp_path / "cluster_attribute_correctness_gold.csv").exists()

    def test_perfect_pipe_perfect_gold_matches_silver_metrics(self):
        # When gold ≡ silver, GR sub-blocks should mirror SR sub-blocks.
        silver = _silver_aligned_to_pipe_groups()
        gold = _silver_aligned_to_pipe_groups()
        result = compute_e2e_panel(
            pipe_fused=_pipe_fused_perfect(),
            correspondences_pipe=_correspondences_perfect(),
            sources_pipe=_make_sources(),
            silver=silver,
            gold=gold,
            column_types=COLUMN_TYPES,
            pipe_source_id_column="id",
            pipe_id_column="cluster_id",
        )
        sr = result.panel["correctness"]["cluster"]["SR"]["bcubed"]
        gr = result.panel["correctness"]["cluster"]["GR"]["bcubed"]
        assert sr["f1"] == pytest.approx(gr["f1"])
        # macro accuracies match
        sr_macro = result.panel["correctness"]["fact"]["SR"]["macro_accuracy"]
        gr_macro = result.panel["correctness"]["fact"]["GR"]["macro_accuracy"]
        assert sr_macro == pytest.approx(gr_macro)

    def test_gold_with_fewer_clusters_has_smaller_n_reference(self):
        silver = _silver_aligned_to_pipe_groups()  # 2 clusters
        gold = _gold_subset_of_silver()  # 1 cluster
        result = compute_e2e_panel(
            pipe_fused=_pipe_fused_perfect(),
            correspondences_pipe=_correspondences_perfect(),
            sources_pipe=_make_sources(),
            silver=silver,
            gold=gold,
            column_types=COLUMN_TYPES,
            pipe_source_id_column="id",
            pipe_id_column="cluster_id",
        )
        assert result.panel["coverage"]["entity"]["SR"]["n_reference"] == 2
        assert result.panel["coverage"]["entity"]["GR"]["n_reference"] == 1
