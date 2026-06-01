"""SC-Block supervised-contrastive blocker (synthetic-local infrastructure).

Wraps a fine-tuned transformer encoder produced by the SC-Block training
recipe (Brinkmann et al., ESWC 2024 — see
``literature-search-generation/scblock_supervised_contrastive_blocking/``)
in PyDI's :class:`BaseBlocker` interface. Inference-only: the encoder is
loaded once per instance, records on both sides are serialised with
Ditto-style ``[COL] <field> [VAL] <value>`` tags, encoded to 768-d
vectors, L2-normalised, and queried against a dense ANN index built on
the right side.

Design points (frozen in
``knobs/committee_review/blocking_shortlist.md`` and mirrored in
``plans/plan_committee_finalization.md`` §C2.4a/b)
---------------------------------------------------------------------------
- **Checkpoint format**: any local directory that
  ``transformers.AutoModel.from_pretrained`` / ``AutoTokenizer.from_pretrained``
  can load — SC-Block saves its encoders via ``model.save_pretrained`` so
  the loader is the HF standard. No custom SC-Block package is vendored;
  the adapter only needs the trained encoder weights.
- **Serialisation**: ``[COL] <field> [VAL] <value>`` per field, fields in
  the order passed in ``text_cols``. Matches the SC-Block paper's record
  representation and Ditto's ``serialize_entity``. NaN / missing values
  are rendered as an empty value (``[COL] name [VAL]``) so the encoder
  still sees the column tag.
- **Pooling**: CLS token by default (SC-Block's trained head). ``mean``
  and ``last_mean_pooled`` (attention-masked mean) are provided for
  flexibility but should not change the frozen-checkpoint behaviour.
- **Normalisation**: L2-normalise embeddings so inner-product search is
  equivalent to cosine similarity.
- **ANN backend**: FAISS ``IndexFlatIP`` by default (exact, deterministic,
  CPU-tractable). ``hnsw`` is available via the ``hnsw`` extra;
  ``sklearn`` is the pure-Python fallback. All three produce pair
  outputs ranked by descending cosine similarity.
- **Injection hook**: an optional ``encoder`` callable
  (``list[str] -> np.ndarray``) bypasses HuggingFace loading entirely.
  Used by the functional test suite to exercise the end-to-end blocking
  path without pulling a transformer checkpoint at test time.

The adapter is synthetic-local infrastructure, not a general PyDI
feature. If a second caller surfaces it can be promoted to
``PyDI.entitymatching.blocking.sc_block``.

Example
-------
>>> import pandas as pd
>>> left = pd.DataFrame({"id": ["a1"], "name": ["ACME Corp"]})
>>> right = pd.DataFrame({"id": ["b1"], "name": ["ACME Corporation"]})
>>> blocker = SCBlockBlocker(  # doctest: +SKIP
...     left, right, id_column="id",
...     text_cols=["name"],
...     checkpoint_path="usecases_synthetic/cache/sc_block_checkpoints/companies/best",
... )
>>> blocker.materialize()  # doctest: +SKIP
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Iterator, Literal, Sequence

import numpy as np
import pandas as pd

from PyDI.entitymatching.blocking.base import BaseBlocker, CandidateBatch

logger = logging.getLogger(__name__)


_DEFAULT_TOP_K = 50
_DEFAULT_MAX_LEN = 256
_DEFAULT_ENCODE_BATCH_SIZE = 64
_DEFAULT_INDEX_BACKEND: Literal["faiss", "hnsw", "sklearn"] = "faiss"
_DEFAULT_POOLING: Literal["cls", "mean"] = "cls"


EncoderFn = Callable[[Sequence[str]], np.ndarray]
"""Type of the optional encoder injection hook.

Accepts an iterable of serialised record strings; returns a float-valued
2-D array of shape ``(len(texts), dim)``. The caller is responsible for
L2-normalisation if desired — ``SCBlockBlocker`` performs the
normalisation itself when ``normalize=True`` (the default).
"""


class SCBlockBlocker(BaseBlocker):
    """Supervised-contrastive dense blocker with FAISS/HNSW/sklearn ANN.

    Parameters
    ----------
    df_left : DataFrame
        Left source. Must contain ``id_column`` and every column in
        ``text_cols``.
    df_right : DataFrame
        Right source. Indexed as the ANN target.
    id_column : str
        Identifier column present in both frames.
    text_cols : sequence of str
        Columns serialised into the ``[COL] <field> [VAL] <value>``
        representation fed to the encoder.
    checkpoint_path : str or Path, optional
        Directory containing a HuggingFace-format encoder (``config.json``,
        ``pytorch_model.bin`` / ``model.safetensors``, tokenizer files).
        Required unless ``encoder`` is provided.
    top_k : int, default=50
        Number of right-source matches retrieved per left-source record.
    threshold : float or None, default=None
        Optional cosine-similarity floor (0-1). Pairs below the threshold
        are dropped. ``None`` keeps every top-``k`` hit.
    max_len : int, default=256
        Token-level sequence length cap passed to the tokenizer.
    batch_size : int, default=100_000
        Maximum candidate rows yielded per batch.
    encode_batch_size : int, default=64
        Records encoded per tokenizer+forward-pass call.
    device : str or None, default=None
        Torch device (``"cpu"`` / ``"cuda"`` / ``"mps"``). ``None``
        auto-detects CUDA → MPS → CPU.
    index_backend : {"faiss", "hnsw", "sklearn"}, default="faiss"
        ANN backend. FAISS is exact + deterministic; HNSW needs the
        ``hnsw`` extra; sklearn is the pure-Python fallback.
    pooling : {"cls", "mean"}, default="cls"
        Token-pooling strategy. SC-Block's frozen checkpoint is trained
        on CLS so leave at the default unless you know the checkpoint
        was trained differently.
    normalize : bool, default=True
        L2-normalise embeddings before indexing / querying. Required for
        cosine-similarity equivalence with inner-product search.
    encoder : callable or None, default=None
        Optional injection hook that bypasses HuggingFace loading. Takes
        a list of serialised strings, returns a 2-D float array. Used by
        the functional test suite; production committee runs leave this
        as ``None``.

    Attributes
    ----------
    text_cols : tuple of str
    top_k : int
    threshold : float or None
    checkpoint_path : Path or None

    Notes
    -----
    The blocker is order-preserving: within a left record, hits are
    emitted in descending cosine similarity; across left records the
    order follows ``df_left`` row order.
    """

    def __init__(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        id_column: str,
        *,
        text_cols: Sequence[str],
        checkpoint_path: str | Path | None = None,
        top_k: int = _DEFAULT_TOP_K,
        threshold: float | None = None,
        max_len: int = _DEFAULT_MAX_LEN,
        batch_size: int = 100_000,
        encode_batch_size: int = _DEFAULT_ENCODE_BATCH_SIZE,
        device: str | None = None,
        index_backend: Literal["faiss", "hnsw", "sklearn"] = _DEFAULT_INDEX_BACKEND,
        pooling: Literal["cls", "mean"] = _DEFAULT_POOLING,
        normalize: bool = True,
        encoder: EncoderFn | None = None,
    ) -> None:
        if not text_cols:
            raise ValueError("text_cols must not be empty")
        # R10-I: tolerate text_cols absent from a heterogeneous source
        # (e.g. companies forbes lacks city/founded under the wide scope).
        # Fill missing columns with empty so the [COL]/[VAL] serialisation
        # is well-defined and identical to the SC-Block trainer (which
        # likewise fills missing text_cols with NA). Warn so a genuine
        # typo in text_cols is still visible.
        missing_left = [c for c in text_cols if c not in df_left.columns]
        missing_right = [c for c in text_cols if c not in df_right.columns]
        if missing_left or missing_right:
            logger.warning(
                "SCBlockBlocker: text_cols missing from sources "
                "(left=%s, right=%s); filling with empty for serialisation",
                missing_left,
                missing_right,
            )
            if missing_left:
                df_left = df_left.copy()
                for col in missing_left:
                    df_left[col] = ""
            if missing_right:
                df_right = df_right.copy()
                for col in missing_right:
                    df_right[col] = ""

        super().__init__(df_left, df_right, id_column, batch_size=batch_size)

        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if threshold is not None and not 0.0 <= float(threshold) <= 1.0:
            raise ValueError(f"threshold must be in [0, 1] when set, got {threshold}")
        if max_len < 1:
            raise ValueError(f"max_len must be >= 1, got {max_len}")
        if encode_batch_size < 1:
            raise ValueError(f"encode_batch_size must be >= 1, got {encode_batch_size}")
        if index_backend not in {"faiss", "hnsw", "sklearn"}:
            raise ValueError(
                f"index_backend must be one of faiss/hnsw/sklearn, got {index_backend!r}"
            )
        if pooling not in {"cls", "mean"}:
            raise ValueError(f"pooling must be 'cls' or 'mean', got {pooling!r}")
        if checkpoint_path is None and encoder is None:
            raise ValueError(
                "SCBlockBlocker requires either checkpoint_path (for "
                "HuggingFace loading) or encoder (for test injection)"
            )

        self.text_cols: tuple[str, ...] = tuple(text_cols)
        self.checkpoint_path: Path | None = (
            Path(checkpoint_path) if checkpoint_path is not None else None
        )
        self.top_k = int(top_k)
        self.threshold = float(threshold) if threshold is not None else None
        self.max_len = int(max_len)
        self.encode_batch_size = int(encode_batch_size)
        self._device_arg = device
        self.index_backend: Literal["faiss", "hnsw", "sklearn"] = index_backend
        self.pooling: Literal["cls", "mean"] = pooling
        self.normalize = bool(normalize)
        self._encoder_override: EncoderFn | None = encoder

        # Lazy-loaded state.
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._device: Any | None = None

        logger.info(
            "SCBlockBlocker initialised: |L|=%d |R|=%d top_k=%d "
            "text_cols=%s backend=%s pooling=%s",
            len(self.df_left),
            len(self.df_right),
            self.top_k,
            self.text_cols,
            self.index_backend,
            self.pooling,
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def _serialize(self, df: pd.DataFrame) -> list[str]:
        """Build ``[COL] field [VAL] value`` strings, one per row.

        Missing values render as an empty ``[VAL]`` chunk so the encoder
        still sees every declared column tag.
        """
        out: list[str] = []
        cols = list(self.text_cols)
        for _, row in df.iterrows():
            parts: list[str] = []
            for col in cols:
                value = row[col]
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    value_str = ""
                else:
                    value_str = str(value)
                parts.append(f"[COL] {col} [VAL] {value_str}".strip())
            out.append(" ".join(parts))
        return out

    # ------------------------------------------------------------------
    # Encoder
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Lazy-load checkpoint and tokenizer on first use."""
        if self._encoder_override is not None:
            return
        if self._model is not None and self._tokenizer is not None:
            return
        if self.checkpoint_path is None or not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"SCBlockBlocker checkpoint not found: {self.checkpoint_path}"
            )
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - import-time guard
            raise ImportError(
                "SCBlockBlocker requires `torch` and `transformers`. "
                "Install via `uv pip install -e '.[plm]' --python pydi-dev/bin/python`."
            ) from exc

        if self._device_arg is not None:
            self._device = torch.device(self._device_arg)
        else:
            self._device = _auto_select_device(torch)

        self._tokenizer = AutoTokenizer.from_pretrained(str(self.checkpoint_path))
        model = AutoModel.from_pretrained(str(self.checkpoint_path))
        model.eval()
        model.to(self._device)
        self._model = model
        logger.info(
            "SCBlockBlocker loaded checkpoint %s on %s",
            self.checkpoint_path,
            self._device,
        )

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encode ``texts`` to a 2-D ``float32`` array (L2-normalised)."""
        if len(texts) == 0:
            return np.zeros((0, 1), dtype=np.float32)

        if self._encoder_override is not None:
            vecs = np.asarray(self._encoder_override(list(texts)), dtype=np.float32)
            if vecs.ndim != 2 or vecs.shape[0] != len(texts):
                raise ValueError(
                    "encoder must return a 2-D array with shape "
                    f"(len(texts), dim); got {vecs.shape}"
                )
        else:
            self._ensure_loaded()
            assert self._tokenizer is not None and self._model is not None
            assert self._device is not None
            import torch

            batches: list[np.ndarray] = []
            for start in range(0, len(texts), self.encode_batch_size):
                chunk = list(texts[start : start + self.encode_batch_size])
                enc = self._tokenizer(
                    chunk,
                    max_length=self.max_len,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                )
                input_ids = enc["input_ids"].to(self._device)
                attention_mask = enc["attention_mask"].to(self._device)
                with torch.no_grad():
                    outputs = self._model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                hidden = outputs.last_hidden_state  # (B, T, H)
                if self.pooling == "cls":
                    pooled = hidden[:, 0, :]
                else:  # mean
                    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
                    summed = (hidden * mask).sum(dim=1)
                    denom = mask.sum(dim=1).clamp(min=1e-6)
                    pooled = summed / denom
                batches.append(pooled.cpu().float().numpy())
            vecs = np.vstack(batches)

        if self.normalize:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            vecs = vecs / norms
        return vecs.astype(np.float32, copy=False)

    # ------------------------------------------------------------------
    # ANN index
    # ------------------------------------------------------------------

    def _build_index(self, right_vecs: np.ndarray) -> tuple[Any, str]:
        """Build the ANN index on the right-side embeddings."""
        backend = self.index_backend
        if backend == "faiss":
            try:
                import faiss  # type: ignore[import-untyped]
            except ImportError:
                logger.warning(
                    "faiss unavailable — falling back to sklearn backend. "
                    "Install via `uv pip install -e '.[faiss]' --python pydi-dev/bin/python`."
                )
                backend = "sklearn"
        if backend == "faiss":
            import faiss  # type: ignore[import-untyped]

            dim = right_vecs.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(np.ascontiguousarray(right_vecs))
            return index, "faiss"

        if backend == "hnsw":
            try:
                import hnswlib
            except ImportError as exc:
                raise ImportError(
                    "hnsw backend requires `hnswlib`. Install via "
                    "`uv pip install -e '.[hnsw]' --python pydi-dev/bin/python`."
                ) from exc
            dim = right_vecs.shape[1]
            index = hnswlib.Index(space="cosine", dim=dim)
            index.init_index(
                max_elements=max(len(right_vecs), 1),
                M=16,
                ef_construction=200,
            )
            index.add_items(right_vecs, np.arange(len(right_vecs)))
            index.set_ef(max(50, self.top_k))
            return index, "hnsw"

        # sklearn fallback.
        from sklearn.neighbors import NearestNeighbors

        index = NearestNeighbors(
            n_neighbors=min(self.top_k, max(len(right_vecs), 1)),
            metric="cosine",
            n_jobs=-1,
        )
        index.fit(right_vecs)
        return index, "sklearn"

    def _query(
        self,
        index: Any,
        backend: str,
        left_vecs: np.ndarray,
        k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(indices, similarities)`` arrays of shape ``(n, k)``."""
        if backend == "faiss":
            scores, indices = index.search(np.ascontiguousarray(left_vecs), k)
            return np.asarray(indices), np.asarray(scores)
        if backend == "hnsw":
            indices, distances = index.knn_query(left_vecs, k=k)
            similarities = 1.0 - np.asarray(distances)
            return np.asarray(indices), similarities
        # sklearn
        distances, indices = index.kneighbors(left_vecs, n_neighbors=k)
        similarities = 1.0 - np.asarray(distances)
        return np.asarray(indices), similarities

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def _iter_batches(self) -> Iterator[CandidateBatch]:
        if self.df_left.empty or self.df_right.empty:
            return

        right_texts = self._serialize(self.df_right)
        right_vecs = self._encode(right_texts)
        index, backend = self._build_index(right_vecs)

        left_texts = self._serialize(self.df_left)
        left_vecs = self._encode(left_texts)

        k = min(self.top_k, len(right_vecs))
        if k < 1:
            return
        indices, similarities = self._query(index, backend, left_vecs, k)

        right_ids = self.df_right[self.id_column].astype(str).tolist()
        left_ids = self.df_left[self.id_column].astype(str).tolist()

        rows: list[tuple[str, str, float]] = []
        for i, id1 in enumerate(left_ids):
            hit_indices = indices[i]
            hit_scores = similarities[i]
            for j, right_idx in enumerate(hit_indices):
                idx = int(right_idx)
                if idx < 0 or idx >= len(right_ids):
                    continue
                score = float(hit_scores[j])
                if self.threshold is not None and score < self.threshold:
                    continue
                rows.append((id1, right_ids[idx], score))
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


def _auto_select_device(torch: Any) -> Any:
    """Select the best available torch device (CUDA → MPS → CPU).

    Accepts the ``torch`` module as an argument so the import stays lazy
    and the top of the file can be imported on systems without torch (an
    ``ImportError`` is only raised when a real HF checkpoint is loaded).
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if (
        mps_backend is not None
        and getattr(mps_backend, "is_available", lambda: False)()
        and getattr(mps_backend, "is_built", lambda: False)()
    ):
        return torch.device("mps")
    return torch.device("cpu")


__all__ = ["SCBlockBlocker", "EncoderFn"]
