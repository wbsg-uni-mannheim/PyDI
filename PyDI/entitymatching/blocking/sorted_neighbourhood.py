"""
Sorted neighbourhood blocking with a sliding window over a sort key.
"""

from __future__ import annotations

from typing import Iterator, List, Tuple

import pandas as pd

from .base import BaseBlocker, CandidateBatch


class SortedNeighbourhood(BaseBlocker):
    """Sorted neighbourhood blocking using a sliding window over a sort key.

    Records from both datasets are combined into a single ordering based on the
    provided key. Then, within a window of size `window`, cross-dataset pairs are
    generated. The algorithm is linearithmic due to sorting.
    """

    def __init__(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        key: str,
        window: int,
        *,
        batch_size: int = 100_000,
    ) -> None:
        super().__init__(df_left, df_right, batch_size=batch_size)
        if key not in self.df_left.columns or key not in self.df_right.columns:
            raise ValueError(f"Key '{key}' must exist in both datasets")
        if window < 1:
            raise ValueError("window must be >= 1")
        self.key = key
        self.window = int(window)

        left_tmp = self.df_left[["_id", key]].copy()
        left_tmp["__side"] = "L"
        right_tmp = self.df_right[["_id", key]].copy()
        right_tmp["__side"] = "R"
        combined = pd.concat([left_tmp, right_tmp], ignore_index=True)

        if pd.api.types.is_string_dtype(combined[key]):
            combined["__sort_key"] = combined[key].fillna(
                "").astype(str).str.lower()
        else:
            combined["__sort_key"] = combined[key]

        self._combined_sorted = combined.sort_values(
            "__sort_key", kind="mergesort").reset_index(drop=True)

    def estimate_pairs(self) -> int | None:  # heuristic
        n = len(self._combined_sorted)
        return max(0, (n * min(self.window, max(0, n - 1))) // 2)

    def _iter_batches(self) -> Iterator[CandidateBatch]:
        ids = self._combined_sorted["_id"].to_list()
        sides = self._combined_sorted["__side"].to_list()
        batch: List[Tuple[str, str]] = []

        for i in range(len(ids)):
            for j in range(i + 1, min(i + 1 + self.window, len(ids))):
                if sides[i] == sides[j]:
                    continue
                id_left, id_right = (
                    ids[i], ids[j]) if sides[i] == "L" else (ids[j], ids[i])
                batch.append((id_left, id_right))
                if len(batch) >= self.batch_size:
                    yield self._emit_batch(pd.DataFrame(batch, columns=["id1", "id2"]))
                    batch = []

        if batch:
            yield self._emit_batch(pd.DataFrame(batch, columns=["id1", "id2"]))


__all__ = ["SortedNeighbourhood"]
