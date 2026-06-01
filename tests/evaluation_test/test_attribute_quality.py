"""Unit tests for fused-attribute quality metrics."""

from __future__ import annotations

import pandas as pd
import pytest

from PyDI.evaluation.attribute_quality import (
    fused_attribute_quality,
    list_attribute_set_metrics,
    per_attribute_density_delta,
    per_cluster_fully_correct_rate,
    source_attribution_metrics,
)


def _alignment(rows):
    return pd.DataFrame(
        rows,
        columns=[
            "silver_cluster_id",
            "best_pipe_cluster_id",
            "overlap_count",
            "silver_size",
            "pipe_size",
            "jaccard",
        ],
    )


class TestFusedAttributeQuality:
    def test_identity_returns_perfect_accuracy(self):
        pipe = pd.DataFrame(
            {"cluster_id": ["c1", "c2"], "title": ["X", "Y"], "year": [1990, 2000]}
        )
        silver = pipe.copy()
        alignment = _alignment([("c1", "c1", 2, 2, 2, 1.0), ("c2", "c2", 2, 2, 2, 1.0)])
        result = fused_attribute_quality(
            pipe,
            silver,
            alignment,
            column_types={"title": "text", "year": "numerical"},
        )
        assert result["macro_accuracy"] == pytest.approx(1.0)
        assert result["micro_accuracy"] == pytest.approx(1.0)
        assert result["per_attribute"]["title"]["accuracy"] == pytest.approx(1.0)

    def test_text_mismatch_reduces_accuracy(self):
        pipe = pd.DataFrame({"cluster_id": ["c1"], "title": ["Wrong"]})
        silver = pd.DataFrame({"cluster_id": ["c1"], "title": ["Right"]})
        alignment = _alignment([("c1", "c1", 1, 1, 1, 1.0)])
        result = fused_attribute_quality(
            pipe, silver, alignment, column_types={"title": "text"}
        )
        assert result["macro_accuracy"] == pytest.approx(0.0)
        assert result["per_attribute"]["title"]["similarity_mean"] == pytest.approx(
            0.0, abs=0.4
        )


class TestListAttributeSetMetrics:
    def test_perfect_set_match(self):
        pipe = pd.DataFrame({"cluster_id": ["c1"], "tracks": [["a", "b", "c"]]})
        silver = pd.DataFrame({"cluster_id": ["c1"], "tracks": [["a", "b", "c"]]})
        alignment = _alignment([("c1", "c1", 1, 1, 1, 1.0)])
        metrics = list_attribute_set_metrics(
            pipe, silver, alignment, {"tracks": "list"}
        )
        assert metrics["tracks"]["set_f1"] == pytest.approx(1.0)

    def test_missing_element_reduces_recall(self):
        pipe = pd.DataFrame({"cluster_id": ["c1"], "tracks": [["a", "b"]]})
        silver = pd.DataFrame({"cluster_id": ["c1"], "tracks": [["a", "b", "c"]]})
        alignment = _alignment([("c1", "c1", 1, 1, 1, 1.0)])
        metrics = list_attribute_set_metrics(
            pipe, silver, alignment, {"tracks": "list"}
        )
        assert metrics["tracks"]["set_recall"] == pytest.approx(2 / 3)


class TestPerAttributeDensityDelta:
    def test_zero_delta_when_no_missing(self):
        pipe = pd.DataFrame({"a": [1, 2, 3]})
        silver = pd.DataFrame({"a": [1, 2, 3]})
        delta = per_attribute_density_delta(pipe, silver, {"a": "numerical"})
        assert delta["a"]["delta"] == pytest.approx(0.0)


class TestPerClusterFullyCorrect:
    def test_all_correct_returns_one(self):
        df = pd.DataFrame(
            {
                "silver_cluster_id": ["c1", "c2"],
                "pipe_cluster_id": ["c1", "c2"],
                "title": [True, True],
                "year": [True, True],
            }
        )
        assert per_cluster_fully_correct_rate(df, ["title", "year"]) == pytest.approx(
            1.0
        )

    def test_any_incorrect_drops_rate(self):
        df = pd.DataFrame(
            {
                "silver_cluster_id": ["c1", "c2"],
                "pipe_cluster_id": ["c1", "c2"],
                "title": [True, False],
                "year": [True, True],
            }
        )
        assert per_cluster_fully_correct_rate(df, ["title", "year"]) == pytest.approx(
            0.5
        )


class TestSourceAttributionMetrics:
    def test_synthesis_rate_split_mass(self):
        silver_provenance = pd.DataFrame(
            [
                {
                    "cluster_id": "c1",
                    "attribute": "title",
                    "source_ids": ["src1_1", "src2_1"],
                },
                {
                    "cluster_id": "c2",
                    "attribute": "title",
                    "source_ids": ["src1_2"],
                },
            ]
        )
        pipe_provenance = pd.DataFrame(
            [
                {
                    "cluster_id": "p1",
                    "attribute": "title",
                    "source_ids": ["src1_1"],
                },
                {
                    "cluster_id": "p2",
                    "attribute": "title",
                    "source_ids": ["src1_2"],
                },
            ]
        )
        alignment = pd.DataFrame(
            [
                {"silver_cluster_id": "c1", "best_pipe_cluster_id": "p1"},
                {"silver_cluster_id": "c2", "best_pipe_cluster_id": "p2"},
            ]
        )
        out = source_attribution_metrics(silver_provenance, pipe_provenance, alignment)
        synth = out["synthesis_rate_per_attribute"]["title"]
        assert synth["silver"] == pytest.approx(0.5)
        assert synth["pipe"] == pytest.approx(0.0)
        assert synth["delta"] == pytest.approx(-0.5)
        assert out["source_attribution_js_per_attribute"]["title"] >= 0.0
