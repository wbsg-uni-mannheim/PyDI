"""
Base interfaces and utils for streaming candidate generators (blockers).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from typing import Iterator, List, Optional

import pandas as pd


CandidateBatch = pd.DataFrame


class BaseBlocker(ABC):
    """Abstract base class for streaming candidate generators.

    Contract:
    - Initialized with left/right DataFrames and strategy params
    - Iterable yielding DataFrames with columns ["id1", "id2", "block_key?"].
    - Each yielded batch should be <= `batch_size` rows.
    - `estimate_pairs()` may return an estimate or None.
    - `stats()` returns simple diagnostics.
    - `materialize()` returns all candidates as a single DataFrame (for small data).
    """

    def __init__(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        id_column: str,
        *,
        batch_size: int = 100_000,
    ) -> None:
        # Validate that ID column exists in both DataFrames
        if id_column not in df_left.columns:
            raise ValueError(f"ID column '{id_column}' not found in left DataFrame")
        if id_column not in df_right.columns:
            raise ValueError(f"ID column '{id_column}' not found in right DataFrame")

        self.df_left = df_left
        self.df_right = df_right
        self.id_column = id_column
        self.batch_size = int(batch_size)

        # Setup logging
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")

        # Cached indexes for faster lookup in some strategies
        self._left_indexed = df_left.set_index(id_column, drop=False)
        self._right_indexed = df_right.set_index(id_column, drop=False)

        self._pairs_emitted = 0
        self._batches_emitted = 0
        

    def __iter__(self) -> Iterator[CandidateBatch]:  # pragma: no cover - interface
        return self._iter_batches()

    @abstractmethod
    def _iter_batches(self) -> Iterator[CandidateBatch]:
        """Yield candidate batches as DataFrames with id1, id2 (and optional block_key)."""

    def estimate_pairs(self) -> Optional[int]:  # pragma: no cover - default
        return None

    def stats(self) -> dict:
        return {
            "pairs_emitted": self._pairs_emitted,
            "batches_emitted": self._batches_emitted,
            "batch_size": self.batch_size,
        }

    def materialize(self) -> pd.DataFrame:
        """Collect all candidates into a single DataFrame. For small datasets only."""
        frames: List[pd.DataFrame] = []
        for batch in self:
            if not batch.empty:
                frames.append(batch)
        if frames:
            return pd.concat(frames, ignore_index=True)
        return pd.DataFrame(columns=["id1", "id2"])  # empty

    # Helper to emit batches with bookkeeping
    def _emit_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        self._batches_emitted += 1
        self._pairs_emitted += len(df)
        return df


__all__ = [
    "BaseBlocker",
    "CandidateBatch",
]
