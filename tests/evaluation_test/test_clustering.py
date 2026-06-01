"""Unit tests for the alignment-based clustering metrics."""

from __future__ import annotations

import pandas as pd
import pytest

from PyDI.evaluation.clustering import (
    adjusted_rand_index,
    bcubed_scores,
    cluster_alignment,
    cluster_sizes,
    normalized_mutual_information,
    pairwise_scores,
)


def _membership(rows):
    return pd.DataFrame(rows, columns=["record_id", "source", "cluster_id"])


class TestBCubed:
    def test_perfect_clustering(self):
        pipe = _membership(
            [
                ("a", "src1", "c1"),
                ("b", "src2", "c1"),
                ("c", "src1", "c2"),
                ("d", "src2", "c2"),
            ]
        )
        silver = pipe.copy()
        scores = bcubed_scores(pipe, silver)
        assert scores["precision"] == pytest.approx(1.0)
        assert scores["recall"] == pytest.approx(1.0)
        assert scores["f1"] == pytest.approx(1.0)

    def test_over_merge_drops_precision(self):
        pipe = _membership(
            [
                ("a", "src1", "c1"),
                ("b", "src2", "c1"),
                ("c", "src1", "c1"),
                ("d", "src2", "c1"),
            ]
        )
        silver = _membership(
            [
                ("a", "src1", "s1"),
                ("b", "src2", "s1"),
                ("c", "src1", "s2"),
                ("d", "src2", "s2"),
            ]
        )
        scores = bcubed_scores(pipe, silver)
        assert scores["precision"] < 1.0
        assert scores["recall"] == pytest.approx(1.0)

    def test_split_clusters_drops_recall(self):
        pipe = _membership(
            [
                ("a", "src1", "c1"),
                ("b", "src2", "c2"),
                ("c", "src1", "c3"),
                ("d", "src2", "c4"),
            ]
        )
        silver = _membership(
            [
                ("a", "src1", "s1"),
                ("b", "src2", "s1"),
                ("c", "src1", "s2"),
                ("d", "src2", "s2"),
            ]
        )
        scores = bcubed_scores(pipe, silver)
        assert scores["recall"] < 1.0


class TestARIAndNMI:
    def test_ari_perfect(self):
        membership = _membership(
            [
                ("a", "src1", "c1"),
                ("b", "src2", "c1"),
                ("c", "src1", "c2"),
                ("d", "src2", "c2"),
            ]
        )
        assert adjusted_rand_index(membership, membership.copy()) == pytest.approx(1.0)

    def test_nmi_perfect(self):
        membership = _membership(
            [
                ("a", "src1", "c1"),
                ("b", "src2", "c1"),
                ("c", "src1", "c2"),
            ]
        )
        assert normalized_mutual_information(
            membership, membership.copy()
        ) == pytest.approx(1.0)


class TestPairwise:
    def test_perfect_clustering(self):
        membership = _membership(
            [
                ("a", "src1", "c1"),
                ("b", "src2", "c1"),
                ("c", "src1", "c2"),
                ("d", "src2", "c2"),
            ]
        )
        scores = pairwise_scores(membership, membership.copy())
        assert scores["f1"] == pytest.approx(1.0)


class TestClusterAlignment:
    def test_alignment_table_emits_one_row_per_silver_cluster(self):
        pipe = _membership(
            [
                ("a", "src1", "c1"),
                ("b", "src2", "c1"),
                ("c", "src1", "c2"),
            ]
        )
        silver = _membership(
            [
                ("a", "src1", "s1"),
                ("b", "src2", "s1"),
                ("c", "src1", "s2"),
            ]
        )
        out = cluster_alignment(pipe, silver)
        table = out["table"]
        assert len(table) == 2
        s1_row = table[table["silver_cluster_id"] == "s1"].iloc[0]
        assert s1_row["best_pipe_cluster_id"] == "c1"
        assert s1_row["jaccard"] == pytest.approx(1.0)
        assert out["mean_jaccard"] == pytest.approx(1.0)

    def test_matched_cluster_rate_uses_threshold(self):
        pipe = _membership(
            [
                ("a", "src1", "c1"),
                ("b", "src2", "c1"),
                ("c", "src1", "c1"),
                ("d", "src2", "c1"),
            ]
        )
        silver = _membership(
            [
                ("a", "src1", "s1"),
                ("b", "src2", "s1"),
                ("c", "src1", "s2"),
                ("d", "src2", "s2"),
            ]
        )
        out = cluster_alignment(pipe, silver, matched_threshold=0.75)
        assert out["matched_cluster_rate_at_threshold"] == pytest.approx(0.0)


class TestClusterSizes:
    def test_empty(self):
        assert cluster_sizes(pd.DataFrame()) == []

    def test_counts_unique_records(self):
        membership = _membership(
            [
                ("a", "src1", "c1"),
                ("b", "src2", "c1"),
                ("c", "src1", "c2"),
            ]
        )
        assert cluster_sizes(membership) == [2, 1]
