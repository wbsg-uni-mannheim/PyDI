"""BM25 blocker adapter for the EM blocking committee.

Wraps the `bm25s <https://github.com/xhluca/bm25s>`_ backend in PyDI's
:class:`BaseBlocker` interface.  Each record on both sides is serialised
to a single text string over ``text_cols``; the right-hand corpus is
indexed once; each left-hand record issues a top-``k`` query and the
retrieved right-hand IDs become candidate pairs.

Design points (frozen in ``knobs/committee_review/blocking_shortlist.md``)
-------------------------------------------------------------------------
- **Backend is bm25s** (not ``rank_bm25``) for large-dataset throughput
  — the sparse-matrix scorer is ~500x faster on the ``companies`` scale.
- **Dedicated tokenizer**: ``bm25s.tokenize`` with English stopwords on
  by default, optional stemming.  Deliberately *not* the shared
  ``preprocess_text`` hook — BM25 recall is highly sensitive to the
  stopword policy and the IDF term degrades without it.
- **Inference-only** in the committee: the index is built from the right
  source on every ``_iter_batches`` call; there is no persisted model.

The adapter is synthetic-local infrastructure, not a general PyDI
feature.  If a second caller surfaces it can be promoted to
``PyDI.entitymatching.blocking.bm25``.

Example
-------
>>> import pandas as pd
>>> left = pd.DataFrame({"id": ["a1", "a2"], "name": ["ACME Corp", "Globex"]})
>>> right = pd.DataFrame({"id": ["b1", "b2"], "name": ["ACME Corporation", "Initech"]})
>>> blocker = BM25Blocker(left, right, id_column="id", text_cols=["name"], top_k=2)
>>> blocker.materialize()  # doctest: +SKIP
"""

from __future__ import annotations

import logging
from typing import Iterator, Literal, Sequence

import numpy as np
import pandas as pd

from PyDI.entitymatching.blocking.base import BaseBlocker, CandidateBatch

logger = logging.getLogger(__name__)


_DEFAULT_TOP_K = 50
"""Top-k retrieval count per left record if the caller doesn't specify."""

_DEFAULT_STOPWORDS = "english"
"""Default stopword list key accepted by ``bm25s.tokenize``."""

_DEFAULT_NGRAM_RANGE: tuple[int, int] = (3, 5)
"""Default character n-gram range when ``tokenizer="char_ngram"``."""

TokenizerMode = Literal["word", "char_ngram"]


class BM25Blocker(BaseBlocker):
    """Okapi BM25 blocker with top-``k`` retrieval per left record.

    Parameters
    ----------
    df_left : DataFrame
        Left source.  Must contain ``id_column`` and every column in
        ``text_cols``.
    df_right : DataFrame
        Right source.  Must contain ``id_column`` and every column in
        ``text_cols``.  Indexed as the BM25 corpus.
    id_column : str
        Name of the identifier column (must be present in both sources).
    text_cols : sequence of str
        Columns joined (space-separated) into the text representation
        used by the tokenizer and the BM25 scorer.
    top_k : int, default=50
        Number of right-source matches retrieved per left-source record.
    min_score : float, default=0.0
        Minimum BM25 score for a candidate pair to be emitted.  Set above
        zero to drop the long tail; the default keeps every hit.
    stopwords : str or list of str or None, default="english"
        Stopword list passed to ``bm25s.tokenize``.  Pass ``None`` to
        disable stopword removal entirely. Ignored when
        ``tokenizer="char_ngram"``.
    stemmer : callable or None, default=None
        Optional stemmer callable (e.g. ``PyStemmer.Stemmer("english").stemWords``).
        ``None`` disables stemming. Ignored when ``tokenizer="char_ngram"``.
    tokenizer : {"word", "char_ngram"}, default="word"
        ``"word"`` runs the original whitespace+stopwords path via
        ``bm25s.tokenize``. ``"char_ngram"`` emits character n-grams
        (sklearn ``char_wb`` style: each whitespace-separated word is
        padded with ``_`` before n-grams are extracted), making the
        scorer tolerant to paraphrase / typos. Use ``"char_ngram"`` when
        the blocker has to survive K1 (paraphrase) noise at hard.
    ngram_range : tuple of int, default=(3, 5)
        Inclusive ``(min_n, max_n)`` for char n-gram extraction. Only
        consulted when ``tokenizer="char_ngram"``.
    k1 : float, default=1.5
        BM25 ``k1`` (term frequency saturation).
    b : float, default=0.75
        BM25 ``b`` (length normalisation).
    batch_size : int, default=100_000
        Maximum candidate rows yielded per batch.

    Attributes
    ----------
    text_cols : tuple of str
        Resolved ``text_cols`` (stored as a tuple for determinism).
    top_k : int
    min_score : float

    Notes
    -----
    The blocker is order-preserving: within a single left record, hits
    are emitted in descending BM25 score; ties fall back to ``bm25s``'s
    internal order (stable per-call).  Across left records the order
    follows ``df_left.iterrows()``.
    """

    def __init__(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        id_column: str,
        *,
        text_cols: Sequence[str],
        top_k: int = _DEFAULT_TOP_K,
        min_score: float = 0.0,
        stopwords: str | Sequence[str] | None = _DEFAULT_STOPWORDS,
        stemmer: object | None = None,
        tokenizer: TokenizerMode = "word",
        ngram_range: tuple[int, int] = _DEFAULT_NGRAM_RANGE,
        k1: float = 1.5,
        b: float = 0.75,
        batch_size: int = 100_000,
    ) -> None:
        super().__init__(df_left, df_right, id_column, batch_size=batch_size)
        if not text_cols:
            raise ValueError("text_cols must not be empty")
        missing_left = [c for c in text_cols if c not in df_left.columns]
        missing_right = [c for c in text_cols if c not in df_right.columns]
        if missing_left:
            raise ValueError(f"text_cols not found in df_left: {missing_left}")
        if missing_right:
            raise ValueError(f"text_cols not found in df_right: {missing_right}")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if min_score < 0:
            raise ValueError(f"min_score must be >= 0, got {min_score}")
        if tokenizer not in ("word", "char_ngram"):
            raise ValueError(
                f"tokenizer must be 'word' or 'char_ngram', got {tokenizer!r}"
            )
        lo, hi = ngram_range
        if tokenizer == "char_ngram" and (lo < 1 or hi < lo):
            raise ValueError(
                f"ngram_range must satisfy 1 <= lo <= hi, got {ngram_range}"
            )

        self.text_cols: tuple[str, ...] = tuple(text_cols)
        self.top_k = int(top_k)
        self.min_score = float(min_score)
        self._stopwords = stopwords
        self._stemmer = stemmer
        self._tokenizer: TokenizerMode = tokenizer
        self._ngram_range = (int(lo), int(hi))
        self._k1 = float(k1)
        self._b = float(b)

        # Lazy bm25s import so the module can be loaded without the
        # extra installed (fails fast at first use with a clear hint).
        try:
            import bm25s  # noqa: F401
        except ImportError as exc:  # pragma: no cover - import-time guard
            raise ImportError(
                "BM25Blocker requires the `bm25s` package. Install via "
                "`uv pip install -e '.[bm25]' --python pydi-dev/bin/python`."
            ) from exc

        logger.info(
            "BM25Blocker initialised: |L|=%d |R|=%d top_k=%d "
            "text_cols=%s tokenizer=%s%s",
            len(self.df_left),
            len(self.df_right),
            self.top_k,
            self.text_cols,
            self._tokenizer,
            (
                f" ngram_range={self._ngram_range}"
                if self._tokenizer == "char_ngram"
                else ""
            ),
        )

    # ------------------------------------------------------------------
    # Text serialisation
    # ------------------------------------------------------------------

    def _serialize(self, df: pd.DataFrame) -> list[str]:
        """Concatenate ``text_cols`` into a single string per row."""
        cols = [df[c].astype(str).fillna("") for c in self.text_cols]
        joined = cols[0]
        for extra in cols[1:]:
            joined = joined.str.cat(extra, sep=" ")
        return joined.tolist()

    def _char_ngram_tokens(self, texts: list[str]) -> list[list[str]]:
        """Tokenise *texts* as character n-grams (``char_wb`` style).

        Each whitespace-separated word is padded with a single ``_``
        on each side so the n-grams capture word boundaries (a leading
        or trailing n-gram is distinguishable from a mid-word one).
        Tokens are lowercased.
        """
        lo, hi = self._ngram_range
        out: list[list[str]] = []
        for text in texts:
            tokens: list[str] = []
            for word in text.lower().split():
                padded = f"_{word}_"
                for n in range(lo, hi + 1):
                    if len(padded) < n:
                        continue
                    for i in range(len(padded) - n + 1):
                        tokens.append(padded[i : i + n])
            out.append(tokens)
        return out

    # ------------------------------------------------------------------
    # Index build
    # ------------------------------------------------------------------

    def _tokenize(self, texts: list[str]) -> object:
        """Tokenise *texts* according to ``self._tokenizer``.

        Word mode delegates to ``bm25s.tokenize`` (returns a
        ``Tokenized`` object honoured natively by ``BM25.index`` /
        ``BM25.retrieve``). Char-ngram mode emits ``list[list[str]]``
        which ``bm25s`` accepts directly.
        """
        import bm25s

        if self._tokenizer == "char_ngram":
            return self._char_ngram_tokens(texts)

        tokenize_kwargs: dict[str, object] = {
            "lower": True,
            "show_progress": False,
        }
        if self._stopwords is not None:
            tokenize_kwargs["stopwords"] = self._stopwords
        if self._stemmer is not None:
            tokenize_kwargs["stemmer"] = self._stemmer
        return bm25s.tokenize(texts, return_ids=False, **tokenize_kwargs)

    def _build_index(self) -> tuple[object, object]:
        """Build the BM25 index over the right corpus.

        Returns
        -------
        (retriever, corpus_tokens) : tuple
            ``retriever`` is a populated ``bm25s.BM25`` instance.
            ``corpus_tokens`` is the tokenised right corpus (retained
            only for debugging — not consumed downstream).
        """
        import bm25s

        corpus_text = self._serialize(self.df_right)
        corpus_tokens = self._tokenize(corpus_text)

        retriever = bm25s.BM25(k1=self._k1, b=self._b)
        retriever.index(corpus_tokens, show_progress=False)
        return retriever, corpus_tokens

    # ------------------------------------------------------------------
    # Iter batches
    # ------------------------------------------------------------------

    def _iter_batches(self) -> Iterator[CandidateBatch]:
        if self.df_left.empty or self.df_right.empty:
            return

        retriever, _ = self._build_index()
        right_ids = self.df_right[self.id_column].astype(str).tolist()
        left_ids = self.df_left[self.id_column].astype(str).tolist()
        query_text = self._serialize(self.df_left)
        query_tokens = self._tokenize(query_text)

        # bm25s.retrieve returns (indices, scores), each shape (n_queries, k).
        k = min(self.top_k, len(right_ids))
        indices, scores = retriever.retrieve(query_tokens, k=k, show_progress=False)
        indices = np.asarray(indices)
        scores = np.asarray(scores)

        rows: list[tuple[str, str, float]] = []
        for i, id1 in enumerate(left_ids):
            hit_idx_row = indices[i]
            score_row = scores[i]
            for j, hit_idx in enumerate(hit_idx_row):
                score = float(score_row[j])
                if score < self.min_score:
                    continue
                id2 = right_ids[int(hit_idx)]
                rows.append((id1, id2, score))
                if len(rows) >= self.batch_size:
                    yield self._emit_batch(
                        pd.DataFrame(rows, columns=["id1", "id2", "score"])
                    )
                    rows = []

        if rows:
            yield self._emit_batch(pd.DataFrame(rows, columns=["id1", "id2", "score"]))

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def estimate_pairs(self) -> int:
        """Upper-bound pair count: ``|L| * min(top_k, |R|)``."""
        return len(self.df_left) * min(self.top_k, len(self.df_right))


__all__ = ["BM25Blocker"]
