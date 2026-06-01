"""Sentence-transformer schema matcher (synthetic-local infrastructure).

This module lives under ``usecases_synthetic/lib/`` rather than
``PyDI/schemamatching/`` because it was introduced by the schema-matching
committee finalisation (C1 in ``plans/plan_committee_finalization.md``) for
the synthetic benchmark pipeline. If the adapter later generalises beyond
the synthetic committee, the promotion path is to move
:class:`EmbeddingBasedSchemaMatcher` into
``PyDI/schemamatching/embedding_based.py`` and re-export it from
``PyDI.schemamatching``.

Used as the ``embedding_sbert`` member of the SM committee
(see ``config/committees/sm_committee.yaml``). Deterministic:
a pinned sentence-transformer model, a single CPU inference path (GPU is
picked up automatically if available), no random state beyond the model
weights themselves.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from PyDI.schemamatching.base import (
    BaseSchemaMatcher,
    SchemaMapping,
    get_schema_columns,
)

logger = logging.getLogger(__name__)


_DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingBasedSchemaMatcher(BaseSchemaMatcher):
    """Schema matcher backed by a sentence-transformer encoder.

    Each source and target column is encoded as a single sentence that
    concatenates (a) the column name and (b) up to ``max_sample_size`` of
    its non-null string values. Cosine similarity between the resulting
    embedding vectors produces the schema-match score.

    This matcher fills the ``embedding`` axis of the synthetic
    schema-matching committee — the non-learned label and instance
    matchers miss semantic column-name equivalences ("released" vs
    "release_year") and value overlap that is token-disjoint but
    semantically close ("United States" vs "USA").

    Parameters
    ----------
    model_name : str, optional
        Hugging Face model identifier loaded via
        :class:`sentence_transformers.SentenceTransformer`. Default
        ``"sentence-transformers/all-MiniLM-L6-v2"`` — 22 M parameters,
        384-dim, CPU-tractable.
    max_sample_size : int, optional
        Maximum number of values sampled from each column to construct
        the encoded sentence. Default 20 (keeps the prompt short and
        deterministic across the typical schema-matching committee
        call shapes; Magneto's retrieval phase uses 10-50).
    device : str, optional
        Torch device string (``"cpu"``, ``"cuda"``, ``"mps"``). If
        ``None`` the sentence-transformer default auto-detect logic is
        used.
    value_separator : str, optional
        String inserted between the column name and the concatenated
        values, and between individual values. Default ``" | "``.
    random_state : int, optional
        Seed used when down-sampling values from a column that is longer
        than ``max_sample_size``. Default 42.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL_NAME,
        max_sample_size: int = 20,
        device: Optional[str] = None,
        value_separator: str = " | ",
        random_state: int = 42,
    ) -> None:
        self.model_name = model_name
        self.max_sample_size = int(max_sample_size)
        self.device = device
        self.value_separator = value_separator
        self.random_state = int(random_state)
        self._model: Any | None = None

    # ------------------------------------------------------------------
    # Lazy model loader
    # ------------------------------------------------------------------

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        from sentence_transformers import SentenceTransformer

        logger.info(
            "Loading SentenceTransformer %s (device=%s) for schema matching",
            self.model_name,
            self.device or "auto",
        )
        model = SentenceTransformer(self.model_name, device=self.device)
        self._model = model
        return model

    # ------------------------------------------------------------------
    # Sentence construction
    # ------------------------------------------------------------------

    def _column_sentence(
        self,
        df: pd.DataFrame,
        column: str,
        preprocess: Optional[Callable[[str], str]] = None,
    ) -> str:
        values = df[column].dropna()
        if len(values) > self.max_sample_size:
            values = values.sample(
                n=self.max_sample_size, random_state=self.random_state
            )
        rendered: list[str] = []
        for val in values:
            text = str(val).strip()
            if not text or text.lower() == "nan":
                continue
            if preprocess is not None:
                text = preprocess(text)
            rendered.append(text)
        name_part = column
        if preprocess is not None:
            name_part = preprocess(name_part)
        if rendered:
            return f"{name_part}{self.value_separator}{self.value_separator.join(rendered)}"
        return name_part

    # ------------------------------------------------------------------
    # Main entrypoint
    # ------------------------------------------------------------------

    def match(
        self,
        source_dataset: pd.DataFrame,
        target_dataset: pd.DataFrame,
        preprocess: Optional[Callable[[str], str]] = None,
        threshold: float = 0.5,
        **kwargs: Any,
    ) -> SchemaMapping:
        """Score all source-target column pairs by cosine similarity.

        Parameters
        ----------
        source_dataset : pandas.DataFrame
            Source frame. Must set ``attrs['dataset_name']``.
        target_dataset : pandas.DataFrame
            Target frame. Must set ``attrs['dataset_name']``.
        preprocess : callable, optional
            String transformer applied to column names and sample values
            before encoding.
        threshold : float, optional
            Minimum cosine similarity required to include a
            correspondence in the output. Default 0.5.

        Returns
        -------
        SchemaMapping
            Frame with columns ``source_dataset``, ``source_column``,
            ``target_dataset``, ``target_column``, ``score`` and
            ``notes``.
        """
        source_name = source_dataset.attrs.get("dataset_name", "source")
        target_name = target_dataset.attrs.get("dataset_name", "target")

        source_columns = get_schema_columns(source_dataset)
        target_columns = get_schema_columns(target_dataset)

        if not source_columns or not target_columns:
            logger.info(
                "EmbeddingBasedSchemaMatcher: empty column list "
                "(source=%d, target=%d) — returning empty mapping",
                len(source_columns),
                len(target_columns),
            )
            return pd.DataFrame(
                columns=[
                    "source_dataset",
                    "source_column",
                    "target_dataset",
                    "target_column",
                    "score",
                    "notes",
                ]
            )

        source_texts = [
            self._column_sentence(source_dataset, c, preprocess) for c in source_columns
        ]
        target_texts = [
            self._column_sentence(target_dataset, c, preprocess) for c in target_columns
        ]

        model = self._ensure_model()
        source_emb = model.encode(
            source_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        target_emb = model.encode(
            target_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        # fp32 matmul can trigger spurious RuntimeWarnings on some BLAS
        # builds (divide-by-zero / overflow / invalid) even when inputs
        # are finite unit vectors — ignore them locally.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            sim_matrix: np.ndarray[Any, Any] = (
                np.asarray(source_emb) @ np.asarray(target_emb).T
            )

        results: list[dict[str, Any]] = []
        for i, src_col in enumerate(source_columns):
            for j, tgt_col in enumerate(target_columns):
                score = float(sim_matrix[i, j])
                if score >= threshold:
                    results.append(
                        {
                            "source_dataset": source_name,
                            "source_column": src_col,
                            "target_dataset": target_name,
                            "target_column": tgt_col,
                            "score": score,
                            "notes": (
                                f"embedding_model={self.model_name},"
                                f"max_sample_size={self.max_sample_size}"
                            ),
                        }
                    )

        return pd.DataFrame(results)
