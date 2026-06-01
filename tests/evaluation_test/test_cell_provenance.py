"""Unit tests for build_cell_provenance_from_fused."""

from __future__ import annotations

import pandas as pd
import pytest

from PyDI.evaluation.cell_provenance import build_cell_provenance_from_fused


class TestBuildCellProvenanceFromFused:
    def test_reshapes_fusion_metadata(self):
        fused = pd.DataFrame(
            [
                {
                    "_fusion_group_id": "group_0",
                    "title": "Album A",
                    "year": 1990,
                    "_fusion_metadata": {
                        "title_rule": "longest_string",
                        "title_sources": ["src1_1"],
                        "year_rule": "first_non_null",
                        "year_sources": ["src1_1", "src2_1"],
                    },
                },
                {
                    "_fusion_group_id": "group_1",
                    "title": "Album B",
                    "year": 2000,
                    "_fusion_metadata": {
                        "title_rule": "longest_string",
                        "title_sources": ["src2_2"],
                        "year_rule": "first_non_null",
                        "year_sources": ["src2_2"],
                    },
                },
            ]
        )
        out = build_cell_provenance_from_fused(fused)
        assert set(out.columns) == {"cluster_id", "attribute", "source_ids"}
        assert len(out) == 4
        title_g0 = out[(out["cluster_id"] == "group_0") & (out["attribute"] == "title")]
        assert title_g0.iloc[0]["source_ids"] == ["src1_1"]
        year_g0 = out[(out["cluster_id"] == "group_0") & (out["attribute"] == "year")]
        assert year_g0.iloc[0]["source_ids"] == ["src1_1", "src2_1"]

    def test_missing_metadata_returns_empty(self):
        fused = pd.DataFrame([{"_fusion_group_id": "group_0", "title": "X"}])
        out = build_cell_provenance_from_fused(fused)
        assert out.empty
        assert list(out.columns) == ["cluster_id", "attribute", "source_ids"]

    def test_attribute_filter(self):
        fused = pd.DataFrame(
            [
                {
                    "_fusion_group_id": "group_0",
                    "_fusion_metadata": {
                        "title_sources": ["a"],
                        "year_sources": ["b"],
                        "label_sources": ["c"],
                    },
                }
            ]
        )
        out = build_cell_provenance_from_fused(
            fused, attribute_columns=["title", "label"]
        )
        assert set(out["attribute"]) == {"title", "label"}

    def test_string_source_coerced_to_list(self):
        fused = pd.DataFrame(
            [
                {
                    "_fusion_group_id": "group_0",
                    "_fusion_metadata": {"title_sources": "src1_1"},  # bare string
                }
            ]
        )
        out = build_cell_provenance_from_fused(fused)
        assert out.iloc[0]["source_ids"] == ["src1_1"]

    def test_skips_rows_with_missing_cluster_id(self):
        fused = pd.DataFrame(
            [
                {
                    "_fusion_group_id": None,
                    "_fusion_metadata": {"title_sources": ["a"]},
                },
                {
                    "_fusion_group_id": "group_1",
                    "_fusion_metadata": {"title_sources": ["b"]},
                },
            ]
        )
        out = build_cell_provenance_from_fused(fused)
        assert len(out) == 1
        assert out.iloc[0]["cluster_id"] == "group_1"

    def test_custom_id_column(self):
        fused = pd.DataFrame(
            [
                {
                    "cluster_id": "C1",
                    "_fusion_metadata": {"title_sources": ["a"]},
                }
            ]
        )
        out = build_cell_provenance_from_fused(fused, pipe_id_column="cluster_id")
        assert out.iloc[0]["cluster_id"] == "C1"
