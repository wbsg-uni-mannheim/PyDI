"""Unit tests for the constraint-validity module."""

from __future__ import annotations

import pandas as pd
import pytest

from PyDI.evaluation.constraint_validity import (
    column_validity_rate,
    compare_column_validity,
    mean_validity_delta,
)


class TestColumnValidityRate:
    def test_numerical_parse(self):
        s = pd.Series([1.0, 2.0, "not a number", None])
        out = column_validity_rate(s, "numerical")
        assert out["n_evaluated"] == 3
        assert out["parse_failures"] == 1
        assert out["validity_rate"] == pytest.approx(2 / 3)

    def test_numerical_range_constraint(self):
        s = pd.Series([0, 50, 100, 150])
        out = column_validity_rate(s, "numerical", {"range": [0, 100]})
        assert out["parse_failures"] == 0
        assert out["constraint_failures"] == 1
        assert out["validity_rate"] == pytest.approx(3 / 4)

    def test_datetime_format_constraint(self):
        s = pd.Series(["2024-01-01", "01-01-2024", "2024-12-31"])
        out = column_validity_rate(s, "datetime", {"format": "%Y-%m-%d"})
        assert out["constraint_failures"] == 1

    def test_datetime_range_constraint(self):
        s = pd.Series(["1880-01-01", "2024-01-01", "2050-12-31"])
        out = column_validity_rate(
            s, "datetime", {"range": ["1900-01-01", "2030-01-01"]}
        )
        # 1880 (below) + 2050 (above) → 2 violations
        assert out["constraint_failures"] == 2

    def test_datetime_overflow_is_handled_gracefully(self):
        # pandas Timestamp max is ~2262 — values past that overflow.
        # The parse layer catches it first (pd.to_datetime returns NaT
        # under errors="coerce") so it surfaces as a parse_failure,
        # not a crash and not a constraint_failure.
        s = pd.Series(["2024-01-01", "9999-01-01"])
        out = column_validity_rate(
            s, "datetime", {"range": ["1900-01-01", "2030-01-01"]}
        )
        assert out["parse_failures"] == 1
        assert out["constraint_failures"] == 0
        assert out["validity_rate"] == pytest.approx(0.5)

    def test_categorical_enum_constraint(self):
        s = pd.Series(["US", "DE", "MARS"])
        out = column_validity_rate(s, "categorical", {"enum": ["US", "DE", "FR"]})
        assert out["constraint_failures"] == 1

    def test_text_regex_constraint(self):
        s = pd.Series(["abc-123", "abc-456", "no_dash"])
        out = column_validity_rate(s, "text", {"regex": r"^[a-z]+-\d+$"})
        assert out["constraint_failures"] == 1

    def test_text_length_constraint(self):
        s = pd.Series(["a", "abc", "longer"])
        out = column_validity_rate(s, "text", {"min_length": 2, "max_length": 4})
        assert out["constraint_failures"] == 2

    def test_list_size_constraint(self):
        s = pd.Series([["a"], ["a", "b", "c"], ["a", "b", "c", "d", "e"]])
        out = column_validity_rate(s, "list", {"min_size": 2, "max_size": 4})
        assert out["constraint_failures"] == 2

    def test_identifier_always_valid(self):
        s = pd.Series(["any", "value", 42])
        out = column_validity_rate(s, "identifier")
        assert out["validity_rate"] == pytest.approx(1.0)
        assert out["n_evaluated"] == 0  # not evaluated by design


class TestCompareColumnValidity:
    def test_regression_surfaces_in_delta(self):
        silver = pd.DataFrame({"year": [1990, 2000, 2010]})
        pipe = pd.DataFrame({"year": [1990, 2000, 9999]})
        out = compare_column_validity(
            pipe,
            silver,
            column_types={"year": "numerical"},
            column_constraints={"year": {"range": [1900, 2030]}},
        )
        assert out["year"]["validity_rate_reference"] == pytest.approx(1.0)
        assert out["year"]["validity_rate_pipe"] == pytest.approx(2 / 3)
        assert out["year"]["delta"] < 0

    def test_only_type_validity_when_no_constraints(self):
        silver = pd.DataFrame({"dur": [100.0, 200.0]})
        pipe = pd.DataFrame({"dur": [100.0, "broken"]})
        out = compare_column_validity(pipe, silver, column_types={"dur": "numerical"})
        assert out["dur"]["parse_failures_pipe"] == 1
        assert out["dur"]["constraint_failures_pipe"] == 0


class TestMeanValidityDelta:
    def test_empty_returns_zero(self):
        assert mean_validity_delta({}) == pytest.approx(0.0)

    def test_averages_per_column_deltas(self):
        per_column = {
            "a": {"delta": -0.10, "n_evaluated_pipe": 100},
            "b": {"delta": 0.00, "n_evaluated_pipe": 100},
        }
        assert mean_validity_delta(per_column) == pytest.approx(-0.05)

    def test_skips_columns_with_zero_evaluated(self):
        per_column = {
            "a": {"delta": -0.50, "n_evaluated_pipe": 0},
            "b": {"delta": -0.10, "n_evaluated_pipe": 100},
        }
        assert mean_validity_delta(per_column) == pytest.approx(-0.10)
