"""
Base interfaces and utils for streaming candidate generators (blockers).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from typing import Iterator, List, Optional

import pandas as pd

from ..base import ensure_record_ids


CandidateBatch = pd.DataFrame


def _ensure_id_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy indexed by `_id` if not already indexed so.

    Leaves original DataFrame unchanged. Ensures `_id` column exists.
    """
    df = ensure_record_ids(df)
    if df.index.name == "_id":
        return df
    return df.set_index("_id", drop=False)


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
        *,
        batch_size: int = 100_000,
    ) -> None:
        self.df_left = ensure_record_ids(df_left)
        self.df_right = ensure_record_ids(df_right)
        self.batch_size = int(batch_size)
        
        # Setup logging
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")

        # Cached indexes for faster lookup in some strategies
        self._left_indexed = _ensure_id_index(self.df_left)
        self._right_indexed = _ensure_id_index(self.df_right)

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
