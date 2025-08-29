"""
Standard (equality) blocking on one or more key columns.
"""

from __future__ import annotations

from typing import Iterator, List, Optional, Tuple

import pandas as pd

from .base import BaseBlocker, CandidateBatch


class StandardBlocking(BaseBlocker):
    """Equality-based blocking on one or more key columns.

    Pairs records whose values are exactly equal across the provided columns.
    This is implemented as a grouped join over block keys and streamed in batches.
    """

    def __init__(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        on: List[str],
        *,
        batch_size: int = 100_000,
    ) -> None:
        super().__init__(df_left, df_right, batch_size=batch_size)
        if not on:
            raise ValueError(
                "StandardBlocking requires at least one column in 'on'")
        self.on = list(on)

        # Precompute block keys; normalize to strings for consistent grouping
        self._left_blocks = self._build_blocks(self.df_left, self.on)
        self._right_blocks = self._build_blocks(self.df_right, self.on)

        # Intersect non-empty keys only
        self._common_keys = [
            k for k in self._left_blocks.keys() if k in self._right_blocks]

    def estimate_pairs(self) -> Optional[int]:
        est = 0
        for key in self._common_keys:
            est += len(self._left_blocks[key]) * len(self._right_blocks[key])
        return est

    @staticmethod
    def _build_blocks(df: pd.DataFrame, cols: List[str]) -> dict:
        for col in cols:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found for blocking")

        # Create a block key column tuple; cast to string to avoid NaN issues
        key_series = df[cols].astype(str).agg("||".join, axis=1)
        blocks: dict[str, List[str]] = {}
        for _id, key in zip(df["_id"], key_series):
            if key not in blocks:
                blocks[key] = []
            blocks[key].append(_id)
        return blocks

    def _iter_batches(self) -> Iterator[CandidateBatch]:
        if not self._common_keys:
            return

        batch_pairs: List[Tuple[str, str]] = []

        for key in self._common_keys:
            left_ids = self._left_blocks[key]
            right_ids = self._right_blocks[key]

            # Cartesian within the block
            for lid in left_ids:
                for rid in right_ids:
                    batch_pairs.append((lid, rid))
                    if len(batch_pairs) >= self.batch_size:
                        yield self._emit_batch(
                            pd.DataFrame(batch_pairs, columns=[
                                         "id1", "id2"]).assign(block_key=key)
                        )
                        batch_pairs = []

        if batch_pairs:
            yield self._emit_batch(pd.DataFrame(batch_pairs, columns=["id1", "id2"]))


__all__ = ["StandardBlocking"]


