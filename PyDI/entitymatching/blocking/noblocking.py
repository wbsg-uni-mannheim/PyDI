"""
NoBlocker: emits full Cartesian product in streaming batches.
"""

from __future__ import annotations

from typing import Iterator

import pandas as pd
import numpy as np

from .base import BaseBlocker, CandidateBatch


class NoBlocker(BaseBlocker):
    """Generate all pairs in the Cartesian product (streamed in batches).

    Suitable for small datasets or as a baseline. For n=|L|, m=|R|, this yields
    n*m pairs split across batches of at most `batch_size` rows.
    """

    def estimate_pairs(self) -> int | None:
        return len(self.df_left) * len(self.df_right)

    def _iter_batches(self) -> Iterator[CandidateBatch]:
        if self.df_left.empty or self.df_right.empty:
            return

        left_ids = self.df_left[self.id_column].to_numpy()
        right_ids = self.df_right[self.id_column].to_numpy()

        if len(right_ids) == 0 or len(left_ids) == 0:
            return

        left_chunk_size = max(1, self.batch_size // max(1, len(right_ids)))
        for start in range(0, len(left_ids), left_chunk_size):
            left_chunk = left_ids[start: start + left_chunk_size]

            right_segment_size = max(
                1, self.batch_size // max(1, len(left_chunk)))
            for r_start in range(0, len(right_ids), right_segment_size):
                right_segment = right_ids[r_start: r_start +
                                          right_segment_size]

                batch = pd.DataFrame(
                    {
                        "id1": pd.Series(left_chunk).repeat(len(right_segment)).to_numpy(),
                        "id2": np.tile(right_segment, len(left_chunk)),
                    }
                )

                if not batch.empty:
                    yield self._emit_batch(batch)


__all__ = ["NoBlocker"]
