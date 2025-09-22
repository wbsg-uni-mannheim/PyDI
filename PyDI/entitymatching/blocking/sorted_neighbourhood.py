"""
Sorted neighbourhood blocking with a sliding window over a sort key.
"""

from __future__ import annotations

import logging
import os
from typing import Iterator, List, Tuple, Optional, Callable
from collections import Counter

import pandas as pd

from .base import BaseBlocker, CandidateBatch


class SortedNeighbourhoodBlocker(BaseBlocker):
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
        id_column: str,
        window: int,
        *,
        batch_size: int = 100_000,
        output_dir: str = "output",
        preprocess: Optional[Callable[[str], str]] = None,
    ) -> None:
        super().__init__(df_left, df_right, id_column, batch_size=batch_size)
        if key not in self.df_left.columns or key not in self.df_right.columns:
            raise ValueError(f"Key '{key}' must exist in both datasets")
        if window < 1:
            raise ValueError("window must be >= 1")
        self.key = key
        self.window = int(window)
        self.output_dir = output_dir
        self.preprocess = preprocess

        # Setup logging
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        
        # Log DEBUG: Creating sort keys
        self.logger.debug(f"Creating sort keys for dataset1: {len(self.df_left)} records")
        left_tmp = self.df_left[[self.id_column, key]].copy()
        left_tmp["__side"] = "L"

        self.logger.debug(f"Creating sort keys for dataset2: {len(self.df_right)} records")
        right_tmp = self.df_right[[self.id_column, key]].copy()
        right_tmp["__side"] = "R"
        
        combined = pd.concat([left_tmp, right_tmp], ignore_index=True)

        if pd.api.types.is_string_dtype(combined[key]):
            if self.preprocess:
                combined["__sort_key"] = combined[key].fillna("").astype(str).apply(self.preprocess)
            else:
                combined["__sort_key"] = combined[key].fillna("").astype(str).str.lower()
        else:
            combined["__sort_key"] = combined[key]

        self.logger.debug(f"Sorting combined dataset with {len(combined)} records")
        self._combined_sorted = combined.sort_values(
            "__sort_key", kind="mergesort").reset_index(drop=True)
        
        # Log INFO: sorted neighbourhood setup
        self.logger.info(f"created sorted neighbourhood with window size {self.window}")
        self.logger.info(f"created 1 sorted sequence from {len(combined)} records")
        
        # Write debug CSV file (like Winter framework)
        self._write_debug_file()

    def estimate_pairs(self) -> int | None:  # heuristic
        n = len(self._combined_sorted)
        return max(0, (n * min(self.window, max(0, n - 1))) // 2)

    def _write_debug_file(self) -> None:
        """Write debug CSV file with sort key prefixes and frequencies like Winter framework."""
        if self._combined_sorted.empty:
            return
            
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Extract sort key prefixes (first 6 characters like Winter seems to do)
        sort_keys = self._combined_sorted["__sort_key"].astype(str)
        prefixes = sort_keys.str[:6].str.upper()  # Convert to uppercase like Winter
        
        # Count frequencies of each prefix
        prefix_counts = prefixes.value_counts()
        
        # Prepare debug data
        debug_data = []
        for prefix, freq in prefix_counts.items():
            debug_data.append({"Blocking Key Value": prefix, "Frequency": freq})
        
        # Write to CSV file
        debug_file = os.path.join(self.output_dir, "debugResultsBlocking_SortedNeighbourhoodBlocker.csv")
        debug_df = pd.DataFrame(debug_data)
        debug_df.to_csv(debug_file, index=False)
        
        self.logger.info(f"Debug results written to file: {debug_file}")

    def _iter_batches(self) -> Iterator[CandidateBatch]:
        # Log DEBUG: Creating candidate pairs 
        self.logger.debug(f"Creating candidate record pairs from sorted neighbourhood with window {self.window}")
        
        ids = self._combined_sorted[self.id_column].to_list()
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


__all__ = ["SortedNeighbourhoodBlocker"]
