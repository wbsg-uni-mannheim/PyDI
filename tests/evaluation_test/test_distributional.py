"""Unit tests for the distributional metrics module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from PyDI.evaluation.distributional import (
    cluster_size_summary,
    column_drift,
    column_drift_panel,
    compute_type_routed_metrics,
    jensen_shannon_divergence,
    schema_diff,
    total_variation_distance,
    wasserstein_1d,
)


class TestDivergences:
    def test_js_identical_is_zero(self):
        p = {"a": 0.5, "b": 0.5}
        assert jensen_shannon_divergence(p, p) == pytest.approx(0.0, abs=1e-12)

    def test_js_disjoint_is_one_at_base_two(self):
        p = {"a": 1.0}
        q = {"b": 1.0}
        assert jensen_shannon_divergence(p, q) == pytest.approx(1.0, abs=1e-9)

    def test_tv_bounded_zero_to_one(self):
        p = {"a": 1.0}
        q = {"a": 0.0, "b": 1.0}
        assert total_variation_distance(p, q) == pytest.approx(1.0)
        assert total_variation_distance(p, p) == pytest.approx(0.0)


class TestWasserstein:
    def test_identical_zero(self):
        assert wasserstein_1d([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)

    def test_shifted_one(self):
        assert wasserstein_1d([1, 2, 3], [2, 3, 4]) == pytest.approx(1.0)


class TestSchemaDiff:
    def test_identical_schema(self):
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        diff = schema_diff(df, df.copy())
        assert diff["columns_shared"] == ["a", "b"]
        assert diff["columns_pipe_only"] == []
        assert diff["columns_silver_only"] == []
        assert diff["dtype_mismatches"] == []

    def test_extra_pipe_column(self):
        pipe = pd.DataFrame({"a": [1], "preview_url": ["http://x"]})
        silver = pd.DataFrame({"a": [1]})
        diff = schema_diff(pipe, silver)
        assert diff["columns_pipe_only"] == ["preview_url"]
        assert diff["columns_silver_only"] == []

    def test_dtype_mismatch(self):
        pipe = pd.DataFrame({"a": [1, 2]})
        silver = pd.DataFrame({"a": ["1", "2"]})
        diff = schema_diff(pipe, silver)
        assert len(diff["dtype_mismatches"]) == 1
        assert diff["dtype_mismatches"][0]["column"] == "a"


class TestClusterSizeSummary:
    def test_identity_returns_zero_distances(self):
        sizes = [1, 1, 2, 3, 5]
        summary = cluster_size_summary(sizes, sizes)
        assert summary["wasserstein_1"] == pytest.approx(0.0)
        assert summary["js_divergence"] == pytest.approx(0.0)
        assert summary["singleton_rate_delta"] == pytest.approx(0.0)
        assert summary["mean_size_pipe"] == pytest.approx(np.mean(sizes))

    def test_over_merge_shifts_singletons_down(self):
        silver = [1, 1, 1, 1, 1]
        pipe = [1, 1, 3]
        summary = cluster_size_summary(pipe, silver)
        assert summary["singleton_rate_pipe"] < summary["singleton_rate_silver"]
        assert summary["wasserstein_1"] > 0.0
        assert summary["max_size_pipe"] == 3


class TestTypeRoutedMetrics:
    def _frames(self):
        pipe = pd.DataFrame(
            {
                "country": ["US", "DE", "US", "FR"],
                "duration": [100.0, 200.0, 105.0, 198.0],
                "release": pd.to_datetime(
                    ["1996-01-01", "1999-06-15", "1996-02-01", "2001-12-25"]
                ),
                "id": ["a", "b", "c", "d"],
            }
        )
        silver = pd.DataFrame(
            {
                "country": ["US", "DE", "US", "FR"],
                "duration": [100.0, 200.0, 100.0, 200.0],
                "release": pd.to_datetime(
                    ["1996-01-01", "1999-06-15", "1996-01-01", "2001-12-25"]
                ),
                "id": ["a", "b", "c", "d"],
            }
        )
        column_types = {
            "country": "categorical",
            "duration": "numerical",
            "release": "datetime",
            "id": "identifier",
        }
        return pipe, silver, column_types

    def test_identifier_is_skipped(self):
        pipe, silver, column_types = self._frames()
        rows = compute_type_routed_metrics(pipe, silver, column_types)
        columns_emitted = {r["column"] for r in rows}
        assert "id" not in columns_emitted

    def test_per_column_nan_rate_delta_is_zero_when_neither_side_nan(self):
        pipe, silver, column_types = self._frames()
        rows = compute_type_routed_metrics(pipe, silver, column_types)
        for r in rows:
            if r["metric"] == "nan_rate_delta":
                assert r["value"] == pytest.approx(0.0)

    def test_numerical_drift_increases_column_drift(self):
        pipe, silver, column_types = self._frames()
        rows = compute_type_routed_metrics(pipe, silver, column_types)
        duration_drift = [
            r["value"]
            for r in rows
            if r["column"] == "duration" and r["metric"] == "column_drift"
        ]
        assert duration_drift and duration_drift[0] > 0


class TestColumnDrift:
    def test_drift_identical_is_zero(self):
        s = pd.Series(["x", "y", "z"])
        assert column_drift(s, s) == pytest.approx(0.0)

    def test_drift_disjoint_is_one(self):
        a = pd.Series(["x"])
        b = pd.Series(["y"])
        assert column_drift(a, b) == pytest.approx(1.0, abs=1e-9)

    def test_numeric_drift_uses_shared_edges(self):
        silver = pd.Series(np.arange(1000, dtype=float))
        # A small shift must NOT max out the metric: this would happen if
        # each side built its own bin edges from its own min/max.
        pipe_small_shift = silver + 5.0
        drift_small = column_drift(pipe_small_shift, silver, is_numeric=True)
        assert drift_small < 0.5, f"small shift maxed out drift at {drift_small}"

        # A large shift past the silver range moves it substantially upward,
        # but must not exceed 1.0.
        pipe_big_shift = silver + 10_000.0
        drift_big = column_drift(pipe_big_shift, silver, is_numeric=True)
        assert drift_big > drift_small
        assert drift_big <= 1.0 + 1e-9

    def test_panel_mean_present(self):
        pipe = pd.DataFrame({"x": ["a", "b"], "y": [1, 2]})
        silver = pd.DataFrame({"x": ["a", "b"], "y": [1, 2]})
        panel = column_drift_panel(pipe, silver, {"x": "categorical", "y": "numerical"})
        assert panel["mean"] == pytest.approx(0.0)
