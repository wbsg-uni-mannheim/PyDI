"""Functional tests for ``ComaSchemaMatcher``.

Covers the C1.6 ``coma_hybrid`` committee member — the COMA 3.0 CE
adapter backed by Valentine's pure-Python ``ComaPy``.

Scope:
- Shape / typing of the returned mapping frame.
- Deterministic behaviour (same inputs → same scores across calls).
- Sensible scoring on obvious matches (strong matches > weak matches).
- NaN tolerance in the source / target frames.
- Edge cases: empty source, empty target, empty both, PyDI id columns
  auto-excluded.
- Provenance-metadata plumbing (``dataset_name`` propagates, PyDI ID
  columns stripped via ``get_schema_columns``).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from usecases_synthetic.lib.coma_sm_matcher import ComaSchemaMatcher

# ``valentine`` emits noisy DeprecationWarnings from its internals
# (pkg_resources, etc.) that are unrelated to our code — silence at
# import time so test output stays readable.
warnings.filterwarnings("ignore", category=DeprecationWarning)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def matcher() -> ComaSchemaMatcher:
    return ComaSchemaMatcher(max_n=1, use_instances=True, use_schema=True)


@pytest.fixture()
def source_df() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "Company": ["Apple Inc", "Google LLC", "Microsoft Corp"],
            "Country": ["US", "US", "US"],
            "Founded": [1976, 1998, 1975],
            "Sector": ["Tech", "Tech", "Tech"],
        }
    )
    df.attrs["dataset_name"] = "forbes"
    return df


@pytest.fixture()
def target_df() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "name": ["Apple", "Google", "Microsoft"],
            "country": ["United States", "United States", "United States"],
            "founded": [1976, 1998, 1975],
            "industry": ["Technology", "Technology", "Technology"],
            "sector": ["Tech", "Tech", "Tech"],
        }
    )
    df.attrs["dataset_name"] = "companies"
    return df


# ---------------------------------------------------------------------------
# Shape + typing
# ---------------------------------------------------------------------------


class TestShape:
    def test_returns_dataframe(
        self,
        matcher: ComaSchemaMatcher,
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
    ) -> None:
        result = matcher.match(source_df, target_df, threshold=0.1)
        assert isinstance(result, pd.DataFrame)

    def test_columns_match_schema_mapping_contract(
        self,
        matcher: ComaSchemaMatcher,
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
    ) -> None:
        result = matcher.match(source_df, target_df, threshold=0.1)
        expected = [
            "source_dataset",
            "source_column",
            "target_dataset",
            "target_column",
            "score",
            "notes",
        ]
        assert list(result.columns) == expected

    def test_dataset_names_propagate(
        self,
        matcher: ComaSchemaMatcher,
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
    ) -> None:
        result = matcher.match(source_df, target_df, threshold=0.1)
        assert (result["source_dataset"] == "forbes").all()
        assert (result["target_dataset"] == "companies").all()

    def test_score_dtype_is_float(
        self,
        matcher: ComaSchemaMatcher,
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
    ) -> None:
        result = matcher.match(source_df, target_df, threshold=0.1)
        assert result["score"].dtype.kind == "f"
        assert ((result["score"] >= 0.0) & (result["score"] <= 1.0)).all()


# ---------------------------------------------------------------------------
# Functional correctness — sensible scores on obvious matches
# ---------------------------------------------------------------------------


class TestScoringSanity:
    """Score-producing members must have a sanity assertion (strong
    positive pairs above a floor)."""

    def test_identical_column_names_score_high(
        self,
        matcher: ComaSchemaMatcher,
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
    ) -> None:
        """``Founded→founded``, ``Sector→sector``, ``Country→country``
        are case-only differences on identical semantics; COMA should
        rank each as the best match for its source column."""
        result = matcher.match(source_df, target_df, threshold=0.0)
        pairs = {
            (row.source_column, row.target_column): row.score
            for row in result.itertuples()
        }
        assert ("Founded", "founded") in pairs
        assert ("Country", "country") in pairs
        assert ("Sector", "sector") in pairs
        for src, tgt in (
            ("Founded", "founded"),
            ("Country", "country"),
            ("Sector", "sector"),
        ):
            assert pairs[(src, tgt)] >= 0.4, (
                f"Obvious match {src}→{tgt} scored {pairs[(src, tgt)]:.3f} "
                f"(expected >= 0.4)"
            )

    def test_max_n_caps_matches_per_source(
        self,
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
    ) -> None:
        """``max_n=1`` means each source column appears at most once."""
        matcher = ComaSchemaMatcher(max_n=1)
        result = matcher.match(source_df, target_df, threshold=0.0)
        counts = result["source_column"].value_counts()
        assert (
            counts <= 1
        ).all(), f"max_n=1 violated: source_column counts = {counts.to_dict()}"

    def test_threshold_filters_output(
        self,
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
    ) -> None:
        """Raising ``threshold`` strictly shrinks the candidate set."""
        matcher = ComaSchemaMatcher(max_n=0)  # no cap
        low = matcher.match(source_df, target_df, threshold=0.0)
        high = matcher.match(source_df, target_df, threshold=0.5)
        assert len(high) <= len(low)
        if not high.empty:
            assert (high["score"] >= 0.5).all()


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Deterministic members must produce exact outputs across calls."""

    def test_same_inputs_yield_same_scores(
        self,
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
    ) -> None:
        m1 = ComaSchemaMatcher()
        m2 = ComaSchemaMatcher()
        r1 = (
            m1.match(source_df, target_df, threshold=0.0)
            .sort_values(["source_column", "target_column"])
            .reset_index(drop=True)
        )
        r2 = (
            m2.match(source_df, target_df, threshold=0.0)
            .sort_values(["source_column", "target_column"])
            .reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(r1, r2)

    def test_repeat_calls_on_same_instance(
        self,
        matcher: ComaSchemaMatcher,
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
    ) -> None:
        """Calling ``match`` twice on the same instance must give the
        same scores (no hidden state leaking between calls)."""
        r1 = matcher.match(source_df, target_df, threshold=0.0)
        r2 = matcher.match(source_df, target_df, threshold=0.0)
        # Sort for positional comparison.
        r1 = r1.sort_values(["source_column", "target_column"]).reset_index(drop=True)
        r2 = r2.sort_values(["source_column", "target_column"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(r1, r2)


# ---------------------------------------------------------------------------
# NaN tolerance
# ---------------------------------------------------------------------------


class TestNaNTolerance:
    """COMA must survive NaN values without raising."""

    def test_nans_in_source_values(
        self,
        matcher: ComaSchemaMatcher,
        target_df: pd.DataFrame,
    ) -> None:
        source = pd.DataFrame(
            {
                "Company": ["Apple Inc", None, "Microsoft Corp"],
                "Country": ["US", "US", np.nan],
                "Founded": [1976, 1998, 1975],
                "Sector": ["Tech", "Tech", "Tech"],
            }
        )
        source.attrs["dataset_name"] = "forbes"
        result = matcher.match(source, target_df, threshold=0.0)
        assert not result.empty
        assert result["score"].notna().all()

    def test_all_nan_column_survives(
        self,
        matcher: ComaSchemaMatcher,
        target_df: pd.DataFrame,
    ) -> None:
        """A source column consisting entirely of NaN must not crash
        ComaPy; it should simply contribute no instance-based signal."""
        source = pd.DataFrame(
            {
                "Company": ["Apple Inc", "Google LLC", "Microsoft Corp"],
                "Country": [None, None, None],
                "Founded": [1976, 1998, 1975],
            }
        )
        source.attrs["dataset_name"] = "forbes"
        result = matcher.match(source, target_df, threshold=0.0)
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_source(
        self,
        matcher: ComaSchemaMatcher,
        target_df: pd.DataFrame,
    ) -> None:
        source = pd.DataFrame()
        source.attrs["dataset_name"] = "forbes"
        result = matcher.match(source, target_df, threshold=0.0)
        assert result.empty
        assert list(result.columns) == [
            "source_dataset",
            "source_column",
            "target_dataset",
            "target_column",
            "score",
            "notes",
        ]

    def test_empty_target(
        self,
        matcher: ComaSchemaMatcher,
        source_df: pd.DataFrame,
    ) -> None:
        target = pd.DataFrame()
        target.attrs["dataset_name"] = "companies"
        result = matcher.match(source_df, target, threshold=0.0)
        assert result.empty

    def test_empty_both(self, matcher: ComaSchemaMatcher) -> None:
        source = pd.DataFrame()
        source.attrs["dataset_name"] = "s"
        target = pd.DataFrame()
        target.attrs["dataset_name"] = "t"
        result = matcher.match(source, target, threshold=0.0)
        assert result.empty

    def test_zero_row_target_still_matches(
        self,
        matcher: ComaSchemaMatcher,
        source_df: pd.DataFrame,
    ) -> None:
        """The SM committee runner builds a zero-row target from the
        variant's target schema. COMA must still produce label-based
        matches in that case."""
        target = pd.DataFrame(
            columns=["name", "country", "founded", "industry", "sector"]
        )
        target.attrs["dataset_name"] = "companies"
        result = matcher.match(source_df, target, threshold=0.0)
        assert not result.empty
        # At least one of the obvious matches should land.
        pairs = {(row.source_column, row.target_column) for row in result.itertuples()}
        obvious = {("Founded", "founded"), ("Sector", "sector"), ("Country", "country")}
        assert pairs & obvious, f"No obvious matches landed on zero-row target: {pairs}"

    def test_pydi_id_column_excluded(
        self,
        matcher: ComaSchemaMatcher,
        target_df: pd.DataFrame,
    ) -> None:
        """``get_schema_columns`` must strip PyDI-generated ID columns so
        COMA never scores ``forbes_id`` against the target schema."""
        source = pd.DataFrame(
            {
                "forbes_id": ["f1", "f2", "f3"],
                "Company": ["Apple Inc", "Google LLC", "Microsoft Corp"],
                "Founded": [1976, 1998, 1975],
            }
        )
        source.attrs["dataset_name"] = "forbes"
        source.attrs["provenance"] = {"id_column_name": "forbes_id"}
        result = matcher.match(source, target_df, threshold=0.0)
        assert "forbes_id" not in set(result["source_column"])


# ---------------------------------------------------------------------------
# API compatibility
# ---------------------------------------------------------------------------


class TestApiContract:
    def test_preprocess_kwarg_accepted_and_ignored(
        self,
        matcher: ComaSchemaMatcher,
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The ``preprocess`` kwarg is part of the ``BaseSchemaMatcher``
        contract; COMA accepts it, logs a warning, and ignores it."""
        import logging

        caplog.set_level(logging.WARNING)
        result = matcher.match(
            source_df, target_df, preprocess=str.lower, threshold=0.0
        )
        assert not result.empty
        assert any(
            "preprocess" in rec.message for rec in caplog.records
        ), "Expected a warning when preprocess is supplied"

    def test_notes_column_tags_coma(
        self,
        matcher: ComaSchemaMatcher,
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
    ) -> None:
        """The ``notes`` column is the provenance marker the committee
        runner uses to attribute predictions to a specific matcher."""
        result = matcher.match(source_df, target_df, threshold=0.0)
        assert (result["notes"].str.startswith("coma_py:")).all()
