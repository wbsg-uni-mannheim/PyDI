"""Tests for ``usecases_synthetic.lib.validation_metrics``."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from usecases_synthetic.lib.validation_metrics import (
    collapse_flag,
    delta,
    f1,
    macro_f1,
    per_attribute_accuracy,
    precision_recall_f1,
)


class TestF1:
    def test_zero_when_all_zero(self) -> None:
        assert f1(0, 0, 0) == 0.0

    def test_perfect(self) -> None:
        assert f1(10, 0, 0) == 1.0

    def test_no_precision_or_recall(self) -> None:
        assert f1(0, 5, 5) == 0.0

    def test_matches_formula(self) -> None:
        # tp=6, fp=2, fn=4 -> precision=0.75, recall=0.6, f1=0.6667
        value = f1(6, 2, 4)
        assert math.isclose(value, 2 * 0.75 * 0.6 / (0.75 + 0.6), rel_tol=1e-9)


class TestPrecisionRecallF1:
    def test_full_overlap(self) -> None:
        result = precision_recall_f1({1, 2, 3}, {1, 2, 3})
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0
        assert result["tp"] == 3.0
        assert result["fp"] == 0.0
        assert result["fn"] == 0.0

    def test_disjoint(self) -> None:
        result = precision_recall_f1({1, 2}, {3, 4})
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0
        assert result["tp"] == 0.0

    def test_empty(self) -> None:
        result = precision_recall_f1([], [])
        assert result == {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "tp": 0.0,
            "fp": 0.0,
            "fn": 0.0,
        }

    def test_partial_overlap(self) -> None:
        result = precision_recall_f1({1, 2, 3, 4}, {3, 4, 5})
        # tp=2, fp=2, fn=1
        assert result["tp"] == 2.0
        assert result["fp"] == 2.0
        assert result["fn"] == 1.0
        assert math.isclose(result["precision"], 0.5)
        assert math.isclose(result["recall"], 2 / 3)

    def test_duplicates_collapsed(self) -> None:
        result = precision_recall_f1([1, 1, 2], [1, 2, 2])
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0


class TestMacroF1:
    def test_empty(self) -> None:
        assert macro_f1({}) == 0.0

    def test_ignores_missing_f1(self) -> None:
        # partition "b" has no f1 key and should be ignored
        partitions = {
            "a": {"f1": 0.8},
            "b": {"precision": 0.5},
        }
        assert macro_f1(partitions) == 0.8

    def test_mean(self) -> None:
        partitions = {
            "a": {"f1": 0.6},
            "b": {"f1": 0.8},
            "c": {"f1": 0.7},
        }
        assert math.isclose(macro_f1(partitions), 0.7)


class TestDelta:
    def test_simple(self) -> None:
        baseline = {"f1": 0.8, "recall": 0.9}
        measured = {"f1": 0.5, "recall": 0.6}
        out = delta(baseline, measured)
        assert math.isclose(out["f1"], -0.3)
        assert math.isclose(out["recall"], -0.3)

    def test_key_union(self) -> None:
        out = delta({"a": 1.0}, {"b": 2.0})
        assert math.isclose(out["a"], -1.0)
        assert math.isclose(out["b"], 2.0)


class TestCollapseFlag:
    def test_no_collapse(self) -> None:
        baseline = {"f1": 0.8}
        measured = {"f1": 0.7}
        assert collapse_flag(measured, baseline) is False

    def test_big_drop(self) -> None:
        baseline = {"f1": 0.9}
        measured = {"f1": 0.2}
        assert collapse_flag(measured, baseline) is True

    def test_below_floor(self) -> None:
        baseline = {"f1": 0.3}
        measured = {"f1": 0.10}
        assert collapse_flag(measured, baseline) is True

    def test_custom_threshold(self) -> None:
        baseline = {"f1": 0.8}
        measured = {"f1": 0.5}
        assert collapse_flag(measured, baseline, max_drop=0.2) is True
        assert collapse_flag(measured, baseline, max_drop=0.4) is False


class TestPerAttributeAccuracy:
    def _frames(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        pred = pd.DataFrame(
            {
                "id": ["a", "b", "c"],
                "name": ["Alice", "Bob", "Carol"],
                "city": ["NYC", "LA", None],
            }
        )
        gold = pd.DataFrame(
            {
                "id": ["a", "b", "c"],
                "name": ["Alice", "Bobby", "Carol"],
                "city": ["NYC", "LA", None],
            }
        )
        return pred, gold

    def test_mixed_accuracy(self) -> None:
        pred, gold = self._frames()
        out = per_attribute_accuracy(pred, gold, ["name", "city"])
        assert math.isclose(out["name"], 2 / 3)
        assert math.isclose(out["city"], 1.0)  # both NaN counts as match

    def test_missing_column(self) -> None:
        pred, gold = self._frames()
        out = per_attribute_accuracy(pred, gold, ["unknown"])
        assert out["unknown"] == 0.0

    def test_missing_id_column(self) -> None:
        pred = pd.DataFrame({"name": ["a"]})
        gold = pd.DataFrame({"id": ["a"], "name": ["a"]})
        out = per_attribute_accuracy(pred, gold, ["name"])
        assert out["name"] == 0.0

    def test_no_overlap(self) -> None:
        pred = pd.DataFrame({"id": ["x"], "name": ["x"]})
        gold = pd.DataFrame({"id": ["y"], "name": ["y"]})
        out = per_attribute_accuracy(pred, gold, ["name"])
        assert out["name"] == 0.0
