"""Functional tests for :class:`SCBlockBlocker` (EM blocking committee).

Goes beyond a shape-only smoke test per the process requirement in
``plans/plan_committee_finalization.md`` §"Process requirement for every
implementation row".

Because a real SC-Block checkpoint is not available in CI, the tests use
the adapter's ``encoder`` injection hook — a user-supplied
``list[str] -> np.ndarray`` — to exercise the end-to-end
serialise → encode → index → query → emit-pairs path without pulling
any transformer. A deterministic hashing embedder (class-level, stable
across runs) stands in for the trained encoder; its output is designed
so that records sharing a distinctive brand token cluster together in
embedding space, giving the top-1 retrieval the same brand-match
behaviour as the real SC-Block checkpoint.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import pytest

from usecases_synthetic.lib.sc_block_blocker import SCBlockBlocker

# ---------------------------------------------------------------------------
# Deterministic stand-in encoder
# ---------------------------------------------------------------------------


class _TokenBagEncoder:
    """Deterministic bag-of-tokens encoder over a fixed vocabulary.

    Each token in ``vocab`` owns one dimension; the embedding of a text
    is a 0/1 multi-hot over tokens that appear in it after lowercasing.
    Brand tokens (``acme``, ``globex``, ``initech``) dominate the
    company-like fixtures, so two records sharing a brand token land at
    a high cosine similarity while records with no shared tokens land
    near zero.

    The class is callable (``encoder(texts) -> np.ndarray``) to match
    the adapter's ``EncoderFn`` signature; it also exposes ``calls``
    (number of invocations) and ``last_texts`` (last batch seen) so the
    tests can assert on the encoder being hit or skipped.
    """

    def __init__(self, vocab: Sequence[str]) -> None:
        self.vocab = tuple(vocab)
        self._index = {t: i for i, t in enumerate(self.vocab)}
        self.calls: int = 0
        self.last_texts: list[str] = []

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        import re

        self.calls += 1
        self.last_texts = list(texts)
        vecs = np.zeros((len(texts), len(self.vocab)), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = set(re.findall(r"[a-z]+", text.lower()))
            for token, idx in self._index.items():
                if token in tokens:
                    vecs[i, idx] = 1.0
        return vecs


@pytest.fixture()
def brand_encoder() -> _TokenBagEncoder:
    return _TokenBagEncoder(
        vocab=(
            "acme",
            "globex",
            "initech",
            "hooli",
            "corp",
            "corporation",
            "systems",
            "holdings",
            "international",
            "incorporated",
        )
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _paired_corpora() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two tiny company-like corpora with one obvious match per left row."""
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


class TestSCBlockBlockerAPIContract:
    """Shape / schema checks for the adapter output."""

    def test_output_columns(self, brand_encoder: _TokenBagEncoder) -> None:
        left, right = _paired_corpora()
        out = SCBlockBlocker(
            left,
            right,
            id_column="id",
            text_cols=["name"],
            top_k=2,
            encoder=brand_encoder,
            index_backend="sklearn",
        ).materialize()
        assert list(out.columns) == ["id1", "id2", "score"]
        assert out["id1"].dtype == object
        assert out["id2"].dtype == object
        assert pd.api.types.is_numeric_dtype(out["score"])

    def test_row_count_upper_bound(self, brand_encoder: _TokenBagEncoder) -> None:
        """``|L| × top_k`` is a hard cap on the row count."""
        left, right = _paired_corpora()
        top_k = 2
        out = SCBlockBlocker(
            left,
            right,
            id_column="id",
            text_cols=["name"],
            top_k=top_k,
            encoder=brand_encoder,
            index_backend="sklearn",
        ).materialize()
        assert len(out) <= len(left) * top_k

    def test_estimate_pairs(self, brand_encoder: _TokenBagEncoder) -> None:
        """|L|=3 × min(5, |R|=4) = 12."""
        left, right = _paired_corpora()
        blocker = SCBlockBlocker(
            left,
            right,
            id_column="id",
            text_cols=["name"],
            top_k=5,
            encoder=brand_encoder,
            index_backend="sklearn",
        )
        assert blocker.estimate_pairs() == 12

    def test_encoder_sees_both_sides(self, brand_encoder: _TokenBagEncoder) -> None:
        """The adapter encodes left + right separately (2 calls)."""
        left, right = _paired_corpora()
        SCBlockBlocker(
            left,
            right,
            id_column="id",
            text_cols=["name"],
            top_k=2,
            encoder=brand_encoder,
            index_backend="sklearn",
        ).materialize()
        assert brand_encoder.calls == 2


# ---------------------------------------------------------------------------
# Ranking sanity
# ---------------------------------------------------------------------------


class TestSCBlockBlockerRankingSanity:
    """Non-trivial behaviour checks — beyond 'returns a non-empty frame'."""

    def test_top_hit_is_brand_match(self, brand_encoder: _TokenBagEncoder) -> None:
        """The top-1 retrieval for each left row must share the brand."""
        left, right = _paired_corpora()
        out = SCBlockBlocker(
            left,
            right,
            id_column="id",
            text_cols=["name"],
            top_k=1,
            encoder=brand_encoder,
            index_backend="sklearn",
        ).materialize()
        pairs = set(zip(out["id1"], out["id2"]))
        assert ("a1", "b1") in pairs  # ACME -> ACME Corporation
        assert ("a2", "b2") in pairs  # Globex -> Globex International
        assert ("a3", "b3") in pairs  # Initech -> Initech Holdings

    def test_positive_pairs_score_above_floor(
        self, brand_encoder: _TokenBagEncoder
    ) -> None:
        """Real-brand matches must land above a strong-positive floor."""
        left, right = _paired_corpora()
        out = SCBlockBlocker(
            left,
            right,
            id_column="id",
            text_cols=["name"],
            top_k=1,
            encoder=brand_encoder,
            index_backend="sklearn",
        ).materialize()
        pair_scores = {(r.id1, r.id2): r.score for r in out.itertuples(index=False)}
        assert pair_scores[("a1", "b1")] >= 0.5
        assert pair_scores[("a2", "b2")] >= 0.5
        assert pair_scores[("a3", "b3")] >= 0.5

    def test_negative_pair_scores_below_ceiling(
        self, brand_encoder: _TokenBagEncoder
    ) -> None:
        """Records with no token overlap must not score near 1.0."""
        left = pd.DataFrame({"id": ["a1"], "name": ["Hooli Inc"]})
        right = pd.DataFrame({"id": ["b1"], "name": ["ACME Corp"]})
        out = SCBlockBlocker(
            left,
            right,
            id_column="id",
            text_cols=["name"],
            top_k=1,
            encoder=brand_encoder,
            index_backend="sklearn",
        ).materialize()
        assert not out.empty
        # No shared tokens in the vocabulary → cosine similarity floor.
        assert (out["score"] < 0.5).all()

    def test_scores_are_descending_per_left_id(
        self, brand_encoder: _TokenBagEncoder
    ) -> None:
        """Within a left id, scores should be non-increasing."""
        left, right = _paired_corpora()
        out = SCBlockBlocker(
            left,
            right,
            id_column="id",
            text_cols=["name"],
            top_k=3,
            encoder=brand_encoder,
            index_backend="sklearn",
        ).materialize()
        for _, group in out.groupby("id1"):
            scores = group["score"].tolist()
            assert scores == sorted(scores, reverse=True)

    def test_threshold_drops_weak_hits(self, brand_encoder: _TokenBagEncoder) -> None:
        """``threshold=0.99`` should retain only near-identical pairs."""
        left, right = _paired_corpora()
        out = SCBlockBlocker(
            left,
            right,
            id_column="id",
            text_cols=["name"],
            top_k=4,
            threshold=0.99,
            encoder=brand_encoder,
            index_backend="sklearn",
        ).materialize()
        # Brand-token-only overlap is below 0.99; every pair should be dropped
        # or score very high (tie-case — both hold).
        assert out.empty or (out["score"] >= 0.99).all()

    def test_multi_column_serialization(self, brand_encoder: _TokenBagEncoder) -> None:
        """Multi-column text_cols must not crash and must still block."""
        left, right = _paired_corpora()
        out = SCBlockBlocker(
            left,
            right,
            id_column="id",
            text_cols=["name", "country"],
            top_k=1,
            encoder=brand_encoder,
            index_backend="sklearn",
        ).materialize()
        pairs = set(zip(out["id1"], out["id2"]))
        assert ("a1", "b1") in pairs


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestSCBlockBlockerDeterminism:
    """Two independent runs on the same input must agree bit-for-bit."""

    def test_two_runs_identical(self, brand_encoder: _TokenBagEncoder) -> None:
        left, right = _paired_corpora()
        encoder2 = _TokenBagEncoder(vocab=brand_encoder.vocab)
        out1 = SCBlockBlocker(
            left,
            right,
            id_column="id",
            text_cols=["name"],
            top_k=3,
            encoder=brand_encoder,
            index_backend="sklearn",
        ).materialize()
        out2 = SCBlockBlocker(
            left,
            right,
            id_column="id",
            text_cols=["name"],
            top_k=3,
            encoder=encoder2,
            index_backend="sklearn",
        ).materialize()
        pd.testing.assert_frame_equal(out1, out2)


# ---------------------------------------------------------------------------
# NaN / missing-value tolerance
# ---------------------------------------------------------------------------


class TestSCBlockBlockerNanTolerance:
    """SC-Block must never crash on NaN text fields (serialise → empty VAL)."""

    def test_nan_in_left(self, brand_encoder: _TokenBagEncoder) -> None:
        left = pd.DataFrame({"id": ["a1", "a2"], "name": ["ACME Corp", None]})
        right = pd.DataFrame(
            {"id": ["b1", "b2"], "name": ["ACME Corporation", "Globex"]}
        )
        out = SCBlockBlocker(
            left,
            right,
            id_column="id",
            text_cols=["name"],
            top_k=1,
            encoder=brand_encoder,
            index_backend="sklearn",
        ).materialize()
        # a1 still retrieves its match; a2 has empty text so its hit is
        # not sanity-checked, only non-crashing is required.
        assert ("a1", "b1") in set(zip(out["id1"], out["id2"]))

    def test_nan_in_right_does_not_crash(self, brand_encoder: _TokenBagEncoder) -> None:
        left = pd.DataFrame({"id": ["a1"], "name": ["ACME Corp"]})
        right = pd.DataFrame({"id": ["b1", "b2"], "name": ["ACME Corporation", None]})
        out = SCBlockBlocker(
            left,
            right,
            id_column="id",
            text_cols=["name"],
            top_k=2,
            encoder=brand_encoder,
            index_backend="sklearn",
        ).materialize()
        assert isinstance(out, pd.DataFrame)
        assert ("a1", "b1") in set(zip(out["id1"], out["id2"]))


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestSCBlockBlockerEdgeCases:
    """Boundary behaviour — empty frames, invalid params, unknown columns."""

    def test_empty_left_yields_empty(self, brand_encoder: _TokenBagEncoder) -> None:
        left = pd.DataFrame(columns=["id", "name"])
        right = pd.DataFrame({"id": ["b1"], "name": ["ACME"]})
        out = SCBlockBlocker(
            left,
            right,
            id_column="id",
            text_cols=["name"],
            top_k=1,
            encoder=brand_encoder,
            index_backend="sklearn",
        ).materialize()
        assert out.empty

    def test_empty_right_yields_empty(self, brand_encoder: _TokenBagEncoder) -> None:
        left = pd.DataFrame({"id": ["a1"], "name": ["ACME"]})
        right = pd.DataFrame(columns=["id", "name"])
        out = SCBlockBlocker(
            left,
            right,
            id_column="id",
            text_cols=["name"],
            top_k=1,
            encoder=brand_encoder,
            index_backend="sklearn",
        ).materialize()
        assert out.empty

    def test_missing_text_col_left_filled_not_raised(
        self, brand_encoder: _TokenBagEncoder, caplog: pytest.LogCaptureFixture
    ) -> None:
        """R10-I: a text_col absent from a source is filled (warn), not raised,
        so wide-scope blocking works across heterogeneous-schema sources."""
        left = pd.DataFrame({"id": ["a1"], "name": ["ACME"]})
        right = pd.DataFrame({"id": ["b1"], "name": ["ACME"]})
        with caplog.at_level("WARNING"):
            blk = SCBlockBlocker(
                left,
                right,
                id_column="id",
                text_cols=["name", "nonexistent"],
                top_k=1,
                encoder=brand_encoder,
                index_backend="sklearn",
            )
        # No raise; the missing col is filled empty + a warning logged.
        assert "missing" in caplog.text.lower()
        out = blk.materialize()
        assert not out.empty  # 'name' still matches ACME<->ACME

    def test_missing_text_col_right_filled_not_raised(
        self, brand_encoder: _TokenBagEncoder
    ) -> None:
        left = pd.DataFrame({"id": ["a1"], "name": ["ACME"]})
        right = pd.DataFrame({"id": ["b1"], "name": ["ACME"]})
        blk = SCBlockBlocker(
            left,
            right,
            id_column="id",
            text_cols=["name", "only_in_left"],
            top_k=1,
            encoder=brand_encoder,
            index_backend="sklearn",
        )
        out = blk.materialize()
        assert not out.empty

    def test_empty_text_cols_raises(self, brand_encoder: _TokenBagEncoder) -> None:
        left = pd.DataFrame({"id": ["a1"], "name": ["ACME"]})
        right = pd.DataFrame({"id": ["b1"], "name": ["ACME"]})
        with pytest.raises(ValueError, match="text_cols must not be empty"):
            SCBlockBlocker(
                left,
                right,
                id_column="id",
                text_cols=[],
                encoder=brand_encoder,
            )

    def test_invalid_top_k_raises(self, brand_encoder: _TokenBagEncoder) -> None:
        left = pd.DataFrame({"id": ["a1"], "name": ["ACME"]})
        right = pd.DataFrame({"id": ["b1"], "name": ["ACME"]})
        with pytest.raises(ValueError, match="top_k must be >= 1"):
            SCBlockBlocker(
                left,
                right,
                id_column="id",
                text_cols=["name"],
                top_k=0,
                encoder=brand_encoder,
            )

    def test_threshold_out_of_range_raises(
        self, brand_encoder: _TokenBagEncoder
    ) -> None:
        left = pd.DataFrame({"id": ["a1"], "name": ["ACME"]})
        right = pd.DataFrame({"id": ["b1"], "name": ["ACME"]})
        with pytest.raises(ValueError, match="threshold must be in"):
            SCBlockBlocker(
                left,
                right,
                id_column="id",
                text_cols=["name"],
                threshold=1.5,
                encoder=brand_encoder,
            )

    def test_bad_index_backend_raises(self, brand_encoder: _TokenBagEncoder) -> None:
        left = pd.DataFrame({"id": ["a1"], "name": ["ACME"]})
        right = pd.DataFrame({"id": ["b1"], "name": ["ACME"]})
        with pytest.raises(ValueError, match="index_backend must be one of"):
            SCBlockBlocker(
                left,
                right,
                id_column="id",
                text_cols=["name"],
                index_backend="elasticsearch",  # type: ignore[arg-type]
                encoder=brand_encoder,
            )

    def test_bad_pooling_raises(self, brand_encoder: _TokenBagEncoder) -> None:
        left = pd.DataFrame({"id": ["a1"], "name": ["ACME"]})
        right = pd.DataFrame({"id": ["b1"], "name": ["ACME"]})
        with pytest.raises(ValueError, match="pooling must be"):
            SCBlockBlocker(
                left,
                right,
                id_column="id",
                text_cols=["name"],
                pooling="max",  # type: ignore[arg-type]
                encoder=brand_encoder,
            )

    def test_no_checkpoint_no_encoder_raises(self) -> None:
        left = pd.DataFrame({"id": ["a1"], "name": ["ACME"]})
        right = pd.DataFrame({"id": ["b1"], "name": ["ACME"]})
        with pytest.raises(
            ValueError, match="requires either checkpoint_path .* or encoder"
        ):
            SCBlockBlocker(left, right, id_column="id", text_cols=["name"])

    def test_missing_checkpoint_defers_error_to_first_use(self) -> None:
        """Construction with a bogus path is fine; ``materialize`` must raise."""
        left = pd.DataFrame({"id": ["a1"], "name": ["ACME"]})
        right = pd.DataFrame({"id": ["b1"], "name": ["ACME"]})
        blocker = SCBlockBlocker(
            left,
            right,
            id_column="id",
            text_cols=["name"],
            checkpoint_path="/tmp/sc_block_does_not_exist",
            index_backend="sklearn",
        )
        with pytest.raises(FileNotFoundError, match="checkpoint not found"):
            blocker.materialize()

    def test_top_k_larger_than_right_is_clamped(
        self, brand_encoder: _TokenBagEncoder
    ) -> None:
        """Requesting top_k=10 from a right source of 2 returns <=2 per query."""
        left = pd.DataFrame({"id": ["a1"], "name": ["ACME Corp"]})
        right = pd.DataFrame(
            {"id": ["b1", "b2"], "name": ["ACME Corporation", "Hooli"]}
        )
        out = SCBlockBlocker(
            left,
            right,
            id_column="id",
            text_cols=["name"],
            top_k=10,
            encoder=brand_encoder,
            index_backend="sklearn",
        ).materialize()
        assert len(out) <= 2

    def test_unknown_id_column_raises(self, brand_encoder: _TokenBagEncoder) -> None:
        left = pd.DataFrame({"ident": ["a1"], "name": ["ACME"]})
        right = pd.DataFrame({"ident": ["b1"], "name": ["ACME"]})
        with pytest.raises(ValueError, match="ID column .* not found"):
            SCBlockBlocker(
                left,
                right,
                id_column="id",
                text_cols=["name"],
                encoder=brand_encoder,
            )


# ---------------------------------------------------------------------------
# Pooling + backend variants
# ---------------------------------------------------------------------------


class TestSCBlockBlockerBackends:
    """All three ANN backends must agree on the top-1 brand match."""

    @pytest.mark.parametrize("backend", ["sklearn", "faiss"])
    def test_backend_agrees_on_top1(
        self, brand_encoder: _TokenBagEncoder, backend: str
    ) -> None:
        pytest.importorskip("faiss") if backend == "faiss" else None
        left, right = _paired_corpora()
        out = SCBlockBlocker(
            left,
            right,
            id_column="id",
            text_cols=["name"],
            top_k=1,
            encoder=brand_encoder,
            index_backend=backend,  # type: ignore[arg-type]
        ).materialize()
        pairs = set(zip(out["id1"], out["id2"]))
        assert ("a1", "b1") in pairs
        assert ("a2", "b2") in pairs
        assert ("a3", "b3") in pairs


# ---------------------------------------------------------------------------
# Serialisation format
# ---------------------------------------------------------------------------


class TestSCBlockBlockerSerialisation:
    """``[COL] field [VAL] value`` tags are preserved in the encoder input."""

    def test_serialisation_uses_col_val_tags(
        self, brand_encoder: _TokenBagEncoder
    ) -> None:
        left = pd.DataFrame({"id": ["a1"], "name": ["ACME Corp"], "country": ["US"]})
        right = pd.DataFrame(
            {"id": ["b1"], "name": ["ACME Corporation"], "country": ["US"]}
        )
        SCBlockBlocker(
            left,
            right,
            id_column="id",
            text_cols=["name", "country"],
            top_k=1,
            encoder=brand_encoder,
            index_backend="sklearn",
        ).materialize()
        # Last batch seen is the left side — 1 row.
        assert len(brand_encoder.last_texts) == 1
        text = brand_encoder.last_texts[0]
        assert "[COL] name [VAL] ACME Corp" in text
        assert "[COL] country [VAL] US" in text

    def test_serialisation_empty_val_on_nan(
        self, brand_encoder: _TokenBagEncoder
    ) -> None:
        left = pd.DataFrame({"id": ["a1"], "name": [None]})
        right = pd.DataFrame({"id": ["b1"], "name": ["ACME Corporation"]})
        SCBlockBlocker(
            left,
            right,
            id_column="id",
            text_cols=["name"],
            top_k=1,
            encoder=brand_encoder,
            index_backend="sklearn",
        ).materialize()
        text = brand_encoder.last_texts[0]
        # NaN rows still carry the column tag with an empty [VAL].
        assert text.startswith("[COL] name [VAL]")
