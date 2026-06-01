"""Functional tests for :class:`MagellanMatcher` (EM matching committee).

Beyond a shape-only smoke test per the process requirement in
``plans/plan_committee_finalization.md`` §"Process requirement for every
implementation row".  Exercises: API contract (columns, types), training
lifecycle (lazy train-on-first-call + caching), ranking sanity (strong
positive pairs score higher than strong negatives), determinism (same
seed → same outputs), NaN tolerance, and edge cases (empty frames,
missing files, degenerate gold, invalid params).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from usecases_synthetic.lib.magellan_em_matcher import (
    MagellanMatcher,
    _coerce_labels,
    _import_class,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _paired_corpora() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two tiny company-like corpora with one obvious match per left row.

    Matches are deliberately brand-synonymous so name-string comparators
    produce a wide separation between positive and negative pairs.
    Mirrors the BM25 blocker fixture.
    """
    left = pd.DataFrame(
        {
            "id": ["a1", "a2", "a3", "a4", "a5"],
            "name": [
                "ACME Corp",
                "Globex Incorporated",
                "Initech Systems",
                "Hooli Incorporated",
                "Pied Piper",
            ],
            "country": ["US", "US", "US", "US", "US"],
        }
    )
    right = pd.DataFrame(
        {
            "id": ["b1", "b2", "b3", "b4", "b5"],
            "name": [
                "ACME Corporation",
                "Globex International",
                "Initech Holdings",
                "Hooli Inc",
                "Pied Piper Compression",
            ],
            "country": ["US", "US", "US", "US", "US"],
        }
    )
    return left, right


def _canonical_comparator_specs() -> list[dict[str, Any]]:
    return [
        {
            "class": "StringComparator",
            "module": "PyDI.entitymatching.comparators",
            "params": {
                "column": "name",
                "similarity_function": "jaccard",
            },
        },
        {
            "class": "StringComparator",
            "module": "PyDI.entitymatching.comparators",
            "params": {
                "column": "country",
                "similarity_function": "jaccard",
            },
        },
    ]


def _write_training_gold(
    tmp_path: Path,
    *,
    positives: list[tuple[str, str]] | None = None,
    negatives: list[tuple[str, str]] | None = None,
    label_style: str = "string",
) -> Path:
    """Write a tiny training gold CSV to ``tmp_path``.

    Parameters
    ----------
    tmp_path : Path
        pytest ``tmp_path`` fixture.
    positives, negatives : list of tuple, optional
        Pairs to mark positive / negative.  Defaults cover the
        5-row paired corpora above: every (an, bn) is positive and
        every (an, b(n+1)) is negative.
    label_style : {"string", "bool"}
        How labels are encoded.  Emit both to confirm both paths parse.
    """
    if positives is None:
        positives = [
            ("a1", "b1"),
            ("a2", "b2"),
            ("a3", "b3"),
            ("a4", "b4"),
            ("a5", "b5"),
        ]
    if negatives is None:
        negatives = [
            ("a1", "b2"),
            ("a2", "b3"),
            ("a3", "b4"),
            ("a4", "b5"),
            ("a5", "b1"),
        ]
    rows: list[dict[str, Any]] = []
    for id1, id2 in positives:
        rows.append(
            {"id1": id1, "id2": id2, "label": True if label_style == "bool" else "true"}
        )
    for id1, id2 in negatives:
        rows.append(
            {
                "id1": id1,
                "id2": id2,
                "label": False if label_style == "bool" else "false",
            }
        )
    out = tmp_path / "tiny_train.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def _all_pairs(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """All-vs-all candidate pairs (the smallest 'blocker' possible)."""
    left_ids = left["id"].tolist()
    right_ids = right["id"].tolist()
    return pd.DataFrame([{"id1": l, "id2": r} for l in left_ids for r in right_ids])


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    """Direct tests of the two module-private helpers."""

    def test_import_class_resolves_sklearn(self) -> None:
        cls = _import_class("sklearn.ensemble.RandomForestClassifier")
        from sklearn.ensemble import RandomForestClassifier

        assert cls is RandomForestClassifier

    def test_import_class_rejects_bare_name(self) -> None:
        with pytest.raises(ValueError, match="dotted path"):
            _import_class("RandomForestClassifier")

    def test_coerce_labels_string(self) -> None:
        series = pd.Series(["true", "false", "TRUE", "  true  ", "False"])
        out = _coerce_labels(series, positive_label="true")
        assert out.tolist() == [1, 0, 1, 1, 0]

    def test_coerce_labels_bool(self) -> None:
        series = pd.Series([True, False, True])
        out = _coerce_labels(series)
        assert out.tolist() == [1, 0, 1]


# ---------------------------------------------------------------------------
# API contract
# ---------------------------------------------------------------------------


class TestMagellanAPIContract:
    """Shape / schema checks for the adapter output."""

    def test_output_columns(self, tmp_path: Path) -> None:
        left, right = _paired_corpora()
        gold = _write_training_gold(tmp_path)
        matcher = MagellanMatcher(
            comparators=_canonical_comparator_specs(),
            training_gold_path=gold,
        )
        out = matcher.match(
            left, right, _all_pairs(left, right), id_column="id", threshold=0.0
        )
        assert list(out.columns) == ["id1", "id2", "score", "notes"]
        assert pd.api.types.is_numeric_dtype(out["score"])
        assert (out["notes"] == "magellan").all()

    def test_empty_candidate_batch_returns_empty(self, tmp_path: Path) -> None:
        left, right = _paired_corpora()
        gold = _write_training_gold(tmp_path)
        matcher = MagellanMatcher(
            comparators=_canonical_comparator_specs(),
            training_gold_path=gold,
        )
        empty = pd.DataFrame(columns=["id1", "id2"])
        out = matcher.match(left, right, empty, id_column="id", threshold=0.0)
        assert out.empty
        assert list(out.columns) == ["id1", "id2", "score", "notes"]

    def test_iterable_batches_supported(self, tmp_path: Path) -> None:
        """Committee runners pass ``list[DataFrame]`` iterables — handle both."""
        left, right = _paired_corpora()
        gold = _write_training_gold(tmp_path)
        matcher = MagellanMatcher(
            comparators=_canonical_comparator_specs(),
            training_gold_path=gold,
        )
        all_pairs = _all_pairs(left, right)
        batch_a = all_pairs.iloc[:10].reset_index(drop=True)
        batch_b = all_pairs.iloc[10:].reset_index(drop=True)
        out_single = matcher.match(
            left, right, all_pairs, id_column="id", threshold=0.0
        )
        out_batched = matcher.match(
            left, right, [batch_a, batch_b], id_column="id", threshold=0.0
        )
        # Row counts match — batching is just a chunking detail.
        assert len(out_single) == len(out_batched)

    def test_ignores_runner_comparator_kwargs(self, tmp_path: Path) -> None:
        """Committee runner forwards ``comparators``/``weights`` — must be
        ignored rather than raising or double-instantiating the comparators.
        """
        left, right = _paired_corpora()
        gold = _write_training_gold(tmp_path)
        matcher = MagellanMatcher(
            comparators=_canonical_comparator_specs(),
            training_gold_path=gold,
        )
        out = matcher.match(
            left,
            right,
            _all_pairs(left, right),
            id_column="id",
            threshold=0.0,
            comparators=["ignored"],  # forwarded by committee runner
            weights=[1.0, 0.5],  # forwarded by committee runner
        )
        assert not out.empty


# ---------------------------------------------------------------------------
# Training lifecycle
# ---------------------------------------------------------------------------


class TestMagellanTraining:
    """Lazy training, caching, and error-path coverage."""

    def test_no_training_before_first_match(self, tmp_path: Path) -> None:
        gold = _write_training_gold(tmp_path)
        matcher = MagellanMatcher(
            comparators=_canonical_comparator_specs(),
            training_gold_path=gold,
        )
        assert matcher._classifier is None
        assert matcher._feature_extractor is None

    def test_first_match_trains_and_caches(self, tmp_path: Path) -> None:
        left, right = _paired_corpora()
        gold = _write_training_gold(tmp_path)
        matcher = MagellanMatcher(
            comparators=_canonical_comparator_specs(),
            training_gold_path=gold,
        )
        matcher.match(
            left, right, _all_pairs(left, right), id_column="id", threshold=0.0
        )
        assert matcher._classifier is not None
        first = matcher._classifier
        # Second call reuses the cached classifier.
        matcher.match(
            left, right, _all_pairs(left, right), id_column="id", threshold=0.0
        )
        assert matcher._classifier is first

    def test_missing_gold_raises_file_not_found(self, tmp_path: Path) -> None:
        left, right = _paired_corpora()
        matcher = MagellanMatcher(
            comparators=_canonical_comparator_specs(),
            training_gold_path=tmp_path / "does_not_exist.csv",
        )
        with pytest.raises(FileNotFoundError, match="training gold not found"):
            matcher.match(
                left,
                right,
                _all_pairs(left, right),
                id_column="id",
                threshold=0.0,
            )

    def test_insufficient_support_raises(self, tmp_path: Path) -> None:
        """min_positive_support=3 refuses a gold with only 1 positive."""
        left, right = _paired_corpora()
        gold = _write_training_gold(
            tmp_path,
            positives=[("a1", "b1")],
            negatives=[("a1", "b2"), ("a1", "b3"), ("a1", "b4"), ("a1", "b5")],
        )
        matcher = MagellanMatcher(
            comparators=_canonical_comparator_specs(),
            training_gold_path=gold,
        )
        with pytest.raises(ValueError, match="insufficient class support"):
            matcher.match(
                left,
                right,
                _all_pairs(left, right),
                id_column="id",
                threshold=0.0,
            )

    def test_missing_column_raises(self, tmp_path: Path) -> None:
        """Gold missing 'label' column raises a clear error."""
        left, right = _paired_corpora()
        bad_gold = tmp_path / "bad.csv"
        pd.DataFrame({"id1": ["a1"], "id2": ["b1"]}).to_csv(bad_gold, index=False)
        matcher = MagellanMatcher(
            comparators=_canonical_comparator_specs(),
            training_gold_path=bad_gold,
        )
        with pytest.raises(ValueError, match="missing column 'label'"):
            matcher.match(
                left,
                right,
                _all_pairs(left, right),
                id_column="id",
                threshold=0.0,
            )

    def test_empty_gold_raises(self, tmp_path: Path) -> None:
        left, right = _paired_corpora()
        empty_gold = tmp_path / "empty.csv"
        pd.DataFrame(columns=["id1", "id2", "label"]).to_csv(empty_gold, index=False)
        matcher = MagellanMatcher(
            comparators=_canonical_comparator_specs(),
            training_gold_path=empty_gold,
        )
        with pytest.raises(ValueError, match="empty"):
            matcher.match(
                left,
                right,
                _all_pairs(left, right),
                id_column="id",
                threshold=0.0,
            )

    def test_bool_label_style_accepted(self, tmp_path: Path) -> None:
        """Gold CSVs can use boolean labels (True/False) alongside strings."""
        left, right = _paired_corpora()
        gold = _write_training_gold(tmp_path, label_style="bool")
        matcher = MagellanMatcher(
            comparators=_canonical_comparator_specs(),
            training_gold_path=gold,
        )
        out = matcher.match(
            left, right, _all_pairs(left, right), id_column="id", threshold=0.0
        )
        assert not out.empty


# ---------------------------------------------------------------------------
# Scoring sanity
# ---------------------------------------------------------------------------


class TestMagellanScoringSanity:
    """Non-trivial behaviour: trained classifier ranks positives above negatives."""

    def test_positive_pairs_score_above_negatives(self, tmp_path: Path) -> None:
        """Strong positive pair (ACME→ACME) scores above a clear mismatch."""
        left, right = _paired_corpora()
        gold = _write_training_gold(tmp_path)
        matcher = MagellanMatcher(
            comparators=_canonical_comparator_specs(),
            training_gold_path=gold,
        )
        out = matcher.match(
            left, right, _all_pairs(left, right), id_column="id", threshold=0.0
        )
        pair_score = dict(zip(zip(out["id1"], out["id2"]), out["score"].tolist()))
        # The true matches — each must score strictly higher than the
        # cross-pair that doesn't share a brand token.
        assert pair_score[("a1", "b1")] > pair_score[("a1", "b2")]
        assert pair_score[("a2", "b2")] > pair_score[("a2", "b3")]
        assert pair_score[("a3", "b3")] > pair_score[("a3", "b4")]

    def test_threshold_filters_low_scores(self, tmp_path: Path) -> None:
        left, right = _paired_corpora()
        gold = _write_training_gold(tmp_path)
        matcher = MagellanMatcher(
            comparators=_canonical_comparator_specs(),
            training_gold_path=gold,
        )
        all_pairs = _all_pairs(left, right)
        out_low = matcher.match(left, right, all_pairs, id_column="id", threshold=0.0)
        out_high = matcher.match(left, right, all_pairs, id_column="id", threshold=0.95)
        assert len(out_high) <= len(out_low)
        # Every retained row meets the threshold.
        assert (out_high["score"] >= 0.95).all()

    def test_predict_path_without_probabilities(self, tmp_path: Path) -> None:
        """With ``use_probabilities=False`` the scores are 0/1 class labels."""
        left, right = _paired_corpora()
        gold = _write_training_gold(tmp_path)
        matcher = MagellanMatcher(
            comparators=_canonical_comparator_specs(),
            training_gold_path=gold,
            use_probabilities=False,
        )
        out = matcher.match(
            left, right, _all_pairs(left, right), id_column="id", threshold=0.0
        )
        assert set(out["score"].unique()).issubset({0.0, 1.0})


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestMagellanDeterminism:
    """Seeded RandomForest on the same gold and candidates must repeat exactly."""

    def test_two_instances_same_seed_identical(self, tmp_path: Path) -> None:
        left, right = _paired_corpora()
        gold = _write_training_gold(tmp_path)
        m1 = MagellanMatcher(
            comparators=_canonical_comparator_specs(),
            training_gold_path=gold,
            seed=42,
        )
        m2 = MagellanMatcher(
            comparators=_canonical_comparator_specs(),
            training_gold_path=gold,
            seed=42,
        )
        out1 = (
            m1.match(
                left, right, _all_pairs(left, right), id_column="id", threshold=0.0
            )
            .sort_values(["id1", "id2"])
            .reset_index(drop=True)
        )
        out2 = (
            m2.match(
                left, right, _all_pairs(left, right), id_column="id", threshold=0.0
            )
            .sort_values(["id1", "id2"])
            .reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(out1, out2)


# ---------------------------------------------------------------------------
# NaN tolerance
# ---------------------------------------------------------------------------


class TestMagellanNanTolerance:
    """Training / inference must tolerate NaN in comparator inputs.

    Every numeric feature column is filled with 0.0 before reaching the
    classifier, so missing string fields never crash the sklearn fit.
    """

    def test_nan_in_source_does_not_crash(self, tmp_path: Path) -> None:
        left, right = _paired_corpora()
        left.loc[2, "country"] = None  # a3 has no country
        right.loc[3, "name"] = None  # b4 has no name
        gold = _write_training_gold(tmp_path)
        matcher = MagellanMatcher(
            comparators=_canonical_comparator_specs(),
            training_gold_path=gold,
        )
        out = matcher.match(
            left, right, _all_pairs(left, right), id_column="id", threshold=0.0
        )
        assert isinstance(out, pd.DataFrame)
        # Even with NaN fields, the clear positive pairs must still score
        # above zero (the fillna(0) path gives them credit via the name
        # comparator).
        scores = dict(zip(zip(out["id1"], out["id2"]), out["score"].tolist()))
        assert scores[("a1", "b1")] > 0.0


# ---------------------------------------------------------------------------
# Edge cases / validation
# ---------------------------------------------------------------------------


class TestMagellanValidation:
    """Parameter-validation paths."""

    def test_empty_comparators_triggers_auto_features(self, tmp_path: Path) -> None:
        """Empty / omitted ``comparators`` enables auto-feature-gen mode.

        Previously this raised ValueError; the new contract is that
        omitting comparators delegates feature construction to
        :func:`magellan_auto_features.auto_generate_comparators` so the
        Magellan philosophy (auto-features + RandomForest implicit
        feature selection) works out of the box.
        """
        left, right = _paired_corpora()
        gold = _write_training_gold(tmp_path)
        matcher = MagellanMatcher(training_gold_path=gold)
        out = matcher.match(
            left,
            right,
            _all_pairs(left, right),
            id_column="id",
            threshold=0.0,
        )
        assert set(out.columns) >= {"id1", "id2", "score", "notes"}
        assert len(out) > 0
        # Auto-feature-gen builds >= 10 features per shared string col
        # (name + country here); confirm the extractor was populated.
        assert matcher._feature_columns is not None
        assert len(matcher._feature_columns) >= 10

    def test_bad_comparator_spec_raises(self, tmp_path: Path) -> None:
        left, right = _paired_corpora()
        gold = _write_training_gold(tmp_path)
        matcher = MagellanMatcher(
            comparators=[{"class": "StringComparator"}],  # missing 'module'
            training_gold_path=gold,
        )
        with pytest.raises(ValueError, match="Comparator spec must have"):
            matcher.match(
                left,
                right,
                _all_pairs(left, right),
                id_column="id",
                threshold=0.0,
            )

    def test_unknown_preprocess_raises(self, tmp_path: Path) -> None:
        left, right = _paired_corpora()
        gold = _write_training_gold(tmp_path)
        matcher = MagellanMatcher(
            comparators=_canonical_comparator_specs(),
            training_gold_path=gold,
            preprocess="does_not_exist",
        )
        with pytest.raises(ValueError, match="Unknown preprocess function"):
            matcher.match(
                left,
                right,
                _all_pairs(left, right),
                id_column="id",
                threshold=0.0,
            )

    def test_normalize_text_preprocess_accepted(self, tmp_path: Path) -> None:
        """The only supported preprocess name resolves without error."""
        left, right = _paired_corpora()
        gold = _write_training_gold(tmp_path)
        matcher = MagellanMatcher(
            comparators=_canonical_comparator_specs(),
            training_gold_path=gold,
            preprocess="normalize_text",
        )
        out = matcher.match(
            left, right, _all_pairs(left, right), id_column="id", threshold=0.0
        )
        assert not out.empty

    def test_missing_id_column_raises(self, tmp_path: Path) -> None:
        left = pd.DataFrame({"ident": ["a1"], "name": ["ACME"], "country": ["US"]})
        right = pd.DataFrame({"ident": ["b1"], "name": ["ACME"], "country": ["US"]})
        gold = _write_training_gold(tmp_path)
        matcher = MagellanMatcher(
            comparators=_canonical_comparator_specs(),
            training_gold_path=gold,
        )
        with pytest.raises(ValueError, match="must have 'id' column"):
            matcher.match(
                left,
                right,
                pd.DataFrame({"id1": ["a1"], "id2": ["b1"]}),
                id_column="id",
                threshold=0.0,
            )

    def test_candidate_missing_ids_raises(self, tmp_path: Path) -> None:
        left, right = _paired_corpora()
        gold = _write_training_gold(tmp_path)
        matcher = MagellanMatcher(
            comparators=_canonical_comparator_specs(),
            training_gold_path=gold,
        )
        bad_batch = pd.DataFrame({"x1": ["a1"], "x2": ["b1"]})
        with pytest.raises(ValueError, match="candidate batch must have"):
            matcher.match(left, right, bad_batch, id_column="id", threshold=0.0)
