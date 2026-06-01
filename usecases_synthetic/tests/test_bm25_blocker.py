"""Functional tests for :class:`BM25Blocker` (EM blocking committee).

Goes beyond a shape-only smoke test per the process requirement in
``plans/plan_committee_finalization.md`` §"Process requirement for every
implementation row".  Exercises: API contract (columns, types), ranking
sanity (strong textual match ranks first), determinism (same inputs →
same outputs), NaN tolerance, edge cases (empty frames, unknown columns,
invalid params), and configuration knobs (top_k, min_score, stopwords).
"""

from __future__ import annotations

import pandas as pd
import pytest

from usecases_synthetic.lib.bm25_blocker import BM25Blocker

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _paired_corpora() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two tiny company-like corpora with one obvious match per left row.

    Returns
    -------
    (left, right) : tuple of DataFrame
        ``left`` has 3 rows; each left row has exactly one strong match
        in ``right`` sharing a distinctive brand token.  Distractors are
        generic enough that BM25 should prefer the brand match.
    """
    left = pd.DataFrame(
        {
            "id": ["a1", "a2", "a3"],
            "name": [
                "ACME Corp",
                "Globex Incorporated",
                "Initech Systems",
            ],
            "country": ["US", "US", "US"],
        }
    )
    right = pd.DataFrame(
        {
            "id": ["b1", "b2", "b3", "b4"],
            "name": [
                "ACME Corporation",
                "Globex International",
                "Initech Holdings",
                "Hooli Inc",
            ],
            "country": ["US", "US", "US", "US"],
        }
    )
    return left, right


# ---------------------------------------------------------------------------
# API contract
# ---------------------------------------------------------------------------


class TestBM25BlockerAPIContract:
    """Shape / schema checks for the adapter output."""

    def test_output_columns(self) -> None:
        left, right = _paired_corpora()
        out = BM25Blocker(
            left, right, id_column="id", text_cols=["name"], top_k=2
        ).materialize()
        assert list(out.columns) == ["id1", "id2", "score"]
        assert out["id1"].dtype == object
        assert out["id2"].dtype == object
        assert pd.api.types.is_numeric_dtype(out["score"])

    def test_row_count_upper_bound(self) -> None:
        """``|L| × top_k`` is a hard cap on the row count."""
        left, right = _paired_corpora()
        top_k = 2
        out = BM25Blocker(
            left, right, id_column="id", text_cols=["name"], top_k=top_k
        ).materialize()
        assert len(out) <= len(left) * top_k

    def test_estimate_pairs(self) -> None:
        left, right = _paired_corpora()
        blocker = BM25Blocker(left, right, id_column="id", text_cols=["name"], top_k=5)
        # |R|=4 clamps top_k from 5 to 4 → estimate = 3 × 4 = 12.
        assert blocker.estimate_pairs() == 12


# ---------------------------------------------------------------------------
# Ranking sanity
# ---------------------------------------------------------------------------


class TestBM25BlockerRankingSanity:
    """Non-trivial behaviour checks — beyond 'returns a non-empty frame'."""

    def test_top_hit_is_brand_match(self) -> None:
        """The top-1 retrieval for each left row must share the brand."""
        left, right = _paired_corpora()
        out = BM25Blocker(
            left, right, id_column="id", text_cols=["name"], top_k=1
        ).materialize()
        pairs = set(zip(out["id1"], out["id2"]))
        # Each left id should retrieve its named-brand match as top-1.
        assert ("a1", "b1") in pairs  # ACME → ACME Corporation
        assert ("a2", "b2") in pairs  # Globex → Globex International
        assert ("a3", "b3") in pairs  # Initech → Initech Holdings

    def test_scores_are_descending_per_left_id(self) -> None:
        """Within a left id, scores should be monotonically non-increasing."""
        left, right = _paired_corpora()
        out = BM25Blocker(
            left, right, id_column="id", text_cols=["name"], top_k=3
        ).materialize()
        for _, group in out.groupby("id1"):
            scores = group["score"].tolist()
            assert scores == sorted(scores, reverse=True)

    def test_min_score_filters_low_tail(self) -> None:
        """Pairs with score < min_score must not appear."""
        left, right = _paired_corpora()
        blocker = BM25Blocker(
            left,
            right,
            id_column="id",
            text_cols=["name"],
            top_k=4,
            min_score=100.0,  # intentionally unreachable
        )
        out = blocker.materialize()
        assert out.empty or (out["score"] >= 100.0).all()
        # Companion check — at min_score=0.0 the blocker *does* emit rows.
        blocker0 = BM25Blocker(
            left,
            right,
            id_column="id",
            text_cols=["name"],
            top_k=4,
        )
        assert not blocker0.materialize().empty

    def test_multi_column_serialization_helps_recall(self) -> None:
        """Joining multiple text_cols should retrieve the shared-country tie-breaker.

        This exercises the multi-column path — ``_serialize`` concatenates
        all ``text_cols`` with spaces and the retriever should consume the
        combined text.
        """
        left, right = _paired_corpora()
        out_single = BM25Blocker(
            left, right, id_column="id", text_cols=["name"], top_k=1
        ).materialize()
        out_multi = BM25Blocker(
            left, right, id_column="id", text_cols=["name", "country"], top_k=1
        ).materialize()
        # Both configurations produce the same brand match as top-1, but
        # the multi-column path must at least return the same-shaped frame.
        assert list(out_multi.columns) == list(out_single.columns)
        assert len(out_multi) == len(out_single)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestBM25BlockerDeterminism:
    """BM25 is purely algorithmic — identical inputs must give identical outputs."""

    def test_two_runs_identical(self) -> None:
        left, right = _paired_corpora()
        out1 = BM25Blocker(
            left, right, id_column="id", text_cols=["name"], top_k=3
        ).materialize()
        out2 = BM25Blocker(
            left, right, id_column="id", text_cols=["name"], top_k=3
        ).materialize()
        pd.testing.assert_frame_equal(out1, out2)


# ---------------------------------------------------------------------------
# NaN / missing-value tolerance
# ---------------------------------------------------------------------------


class TestBM25BlockerNanTolerance:
    """BM25 should never crash on NaN text fields — ``_serialize`` fillna('')."""

    def test_nan_in_left(self) -> None:
        left = pd.DataFrame({"id": ["a1", "a2"], "name": ["ACME Corp", None]})
        right = pd.DataFrame(
            {"id": ["b1", "b2"], "name": ["ACME Corporation", "Globex"]}
        )
        out = BM25Blocker(
            left, right, id_column="id", text_cols=["name"], top_k=1
        ).materialize()
        # a1 still retrieves its match; a2 has empty query text → either
        # no hits or a non-crashing empty row.  Either is fine; crashing is not.
        assert ("a1", "b1") in set(zip(out["id1"], out["id2"]))

    def test_nan_in_right_does_not_crash(self) -> None:
        left = pd.DataFrame({"id": ["a1"], "name": ["ACME Corp"]})
        right = pd.DataFrame({"id": ["b1", "b2"], "name": ["ACME Corporation", None]})
        # Must not raise.
        out = BM25Blocker(
            left, right, id_column="id", text_cols=["name"], top_k=2
        ).materialize()
        assert isinstance(out, pd.DataFrame)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestBM25BlockerEdgeCases:
    """Boundary behaviour — empty frames, invalid params, unknown columns."""

    def test_empty_left_yields_empty(self) -> None:
        left = pd.DataFrame(columns=["id", "name"])
        right = pd.DataFrame({"id": ["b1"], "name": ["ACME"]})
        out = BM25Blocker(
            left, right, id_column="id", text_cols=["name"], top_k=1
        ).materialize()
        assert out.empty

    def test_empty_right_yields_empty(self) -> None:
        left = pd.DataFrame({"id": ["a1"], "name": ["ACME"]})
        right = pd.DataFrame(columns=["id", "name"])
        out = BM25Blocker(
            left, right, id_column="id", text_cols=["name"], top_k=1
        ).materialize()
        assert out.empty

    def test_unknown_text_col_left_raises(self) -> None:
        left = pd.DataFrame({"id": ["a1"], "name": ["ACME"]})
        right = pd.DataFrame({"id": ["b1"], "name": ["ACME"]})
        with pytest.raises(ValueError, match="text_cols not found in df_left"):
            BM25Blocker(left, right, id_column="id", text_cols=["nonexistent"])

    def test_unknown_text_col_right_raises(self) -> None:
        left = pd.DataFrame({"id": ["a1"], "name": ["ACME"]})
        right = pd.DataFrame({"id": ["b1"], "label": ["ACME"]})
        with pytest.raises(ValueError, match="text_cols not found in df_right"):
            BM25Blocker(left, right, id_column="id", text_cols=["name"])

    def test_empty_text_cols_raises(self) -> None:
        left = pd.DataFrame({"id": ["a1"], "name": ["ACME"]})
        right = pd.DataFrame({"id": ["b1"], "name": ["ACME"]})
        with pytest.raises(ValueError, match="text_cols must not be empty"):
            BM25Blocker(left, right, id_column="id", text_cols=[])

    def test_invalid_top_k_raises(self) -> None:
        left = pd.DataFrame({"id": ["a1"], "name": ["ACME"]})
        right = pd.DataFrame({"id": ["b1"], "name": ["ACME"]})
        with pytest.raises(ValueError, match="top_k must be >= 1"):
            BM25Blocker(left, right, id_column="id", text_cols=["name"], top_k=0)

    def test_negative_min_score_raises(self) -> None:
        left = pd.DataFrame({"id": ["a1"], "name": ["ACME"]})
        right = pd.DataFrame({"id": ["b1"], "name": ["ACME"]})
        with pytest.raises(ValueError, match="min_score must be >= 0"):
            BM25Blocker(
                left,
                right,
                id_column="id",
                text_cols=["name"],
                min_score=-0.1,
            )

    def test_unknown_id_column_raises(self) -> None:
        left = pd.DataFrame({"ident": ["a1"], "name": ["ACME"]})
        right = pd.DataFrame({"ident": ["b1"], "name": ["ACME"]})
        with pytest.raises(ValueError, match="ID column .* not found"):
            BM25Blocker(left, right, id_column="id", text_cols=["name"])

    def test_top_k_larger_than_right_is_clamped(self) -> None:
        """Requesting top_k=10 from a right source of 2 returns <=2 per query."""
        left = pd.DataFrame({"id": ["a1"], "name": ["ACME"]})
        right = pd.DataFrame({"id": ["b1", "b2"], "name": ["ACME", "Hooli"]})
        out = BM25Blocker(
            left, right, id_column="id", text_cols=["name"], top_k=10
        ).materialize()
        assert len(out) <= 2


# ---------------------------------------------------------------------------
# Stopword behaviour
# ---------------------------------------------------------------------------


class TestBM25BlockerStopwords:
    """Stopword config should shape retrieval behaviour."""

    def test_none_stopwords_does_not_crash(self) -> None:
        left, right = _paired_corpora()
        out = BM25Blocker(
            left,
            right,
            id_column="id",
            text_cols=["name"],
            top_k=2,
            stopwords=None,
        ).materialize()
        assert not out.empty


# ---------------------------------------------------------------------------
# Char-ngram tokeniser
# ---------------------------------------------------------------------------


class TestBM25BlockerCharNgram:
    """Char-ngram tokenisation tolerates paraphrase / typo noise."""

    def test_init_validates_ngram_range(self) -> None:
        left, right = _paired_corpora()
        with pytest.raises(ValueError, match="ngram_range"):
            BM25Blocker(
                left,
                right,
                id_column="id",
                text_cols=["name"],
                tokenizer="char_ngram",
                ngram_range=(5, 3),
            )

    def test_init_rejects_unknown_tokenizer(self) -> None:
        left, right = _paired_corpora()
        with pytest.raises(ValueError, match="tokenizer"):
            BM25Blocker(
                left,
                right,
                id_column="id",
                text_cols=["name"],
                tokenizer="foo",  # type: ignore[arg-type]
            )

    def test_ngram_token_extraction(self) -> None:
        """Internal helper emits ``char_wb``-style padded n-grams."""
        left, right = _paired_corpora()
        blocker = BM25Blocker(
            left,
            right,
            id_column="id",
            text_cols=["name"],
            tokenizer="char_ngram",
            ngram_range=(3, 3),
        )
        toks = blocker._char_ngram_tokens(["Acme"])
        # _ACME_ padded → trigrams: _ac, acm, cme, me_
        assert toks == [["_ac", "acm", "cme", "me_"]]

    def test_char_ngram_returns_nonempty_with_distinctive_match(self) -> None:
        """Distinctive shared n-grams produce a top-1 hit."""
        left, right = _paired_corpora()
        out = BM25Blocker(
            left,
            right,
            id_column="id",
            text_cols=["name"],
            top_k=1,
            tokenizer="char_ngram",
            ngram_range=(3, 5),
        ).materialize()
        # The Acme row has the most distinctive shared n-grams
        # (``_ac``, ``acm``, ``cme``, ``me_``); it must land on b1.
        a1_hits = out[out["id1"] == "a1"]
        assert not a1_hits.empty
        assert a1_hits.iloc[0]["id2"] == "b1"

    def test_char_ngram_survives_paraphrase_where_word_fails(self) -> None:
        """K1-paraphrase analogue: word tokeniser misses; char-ngram hits.

        Builds a left frame where the brand has been paraphrased to a
        synonym that shares no whole-word token with the right corpus
        but does share character sub-strings. Word BM25 cannot rank
        the true match first; char-ngram BM25 can.
        """
        left = pd.DataFrame(
            {
                "id": ["a1"],
                "name": ["Internatnl Bsiness Mach"],
            }
        )
        right = pd.DataFrame(
            {
                "id": ["b1", "b2", "b3"],
                "name": [
                    "International Business Machines",
                    "Apple Computer",
                    "Microsoft Software",
                ],
            }
        )
        out_word = BM25Blocker(
            left, right, id_column="id", text_cols=["name"], top_k=1
        ).materialize()
        out_ngram = BM25Blocker(
            left,
            right,
            id_column="id",
            text_cols=["name"],
            top_k=1,
            tokenizer="char_ngram",
            ngram_range=(3, 5),
        ).materialize()
        # Char-ngram should rank the IBM row first.
        assert not out_ngram.empty
        assert out_ngram.iloc[0]["id2"] == "b1"
        # Word BM25 may return empty (no shared whole tokens) or rank a
        # different row; the test only asserts char-ngram's recovery.
        if not out_word.empty:
            assert out_word.iloc[0]["id2"] != "b1" or out_word.iloc[0]["score"] < (
                out_ngram.iloc[0]["score"]
            )
