"""
Standard (equality) blocking on one or more key columns.
"""

from __future__ import annotations

import logging
import os
from typing import Iterator, List, Optional, Tuple, Callable
from collections import Counter

import pandas as pd

from .base import BaseBlocker, CandidateBatch


class StandardBlocker(BaseBlocker):
    """Equality-based blocking on one or more key columns.

    Pairs records whose values are exactly equal across the provided columns.
    This is implemented as a grouped join over block keys and streamed in batches.
    """

    def __init__(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        on: List[str],
        id_column: str,
        *,
        batch_size: int = 100_000,
        output_dir: str = "output",
        preprocess: Optional[Callable[[str], str]] = None,
    ) -> None:
        super().__init__(df_left, df_right, id_column, batch_size=batch_size)
        if not on:
            raise ValueError(
                "StandardBlocker requires at least one column in 'on'")
        self.on = list(on)
        self.output_dir = output_dir
        self.preprocess = preprocess

        # Setup logging (same as BaseBlocker pattern)
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        
        # Log DEBUG: Creating blocking key values
        self.logger.debug(f"Creating blocking key values for dataset1: {len(self.df_left)} records")
        self._left_blocks = self._build_blocks(self.df_left, self.on, self.id_column, self.preprocess)

        self.logger.debug(f"Creating blocking key values for dataset2: {len(self.df_right)} records")
        self._right_blocks = self._build_blocks(self.df_right, self.on, self.id_column, self.preprocess)
        
        # Log INFO: blocking key counts
        self.logger.info(f"created {len(self._left_blocks)} blocking keys for first dataset")
        self.logger.info(f"created {len(self._right_blocks)} blocking keys for second dataset")

        # Intersect non-empty keys only
        self._common_keys = [
            k for k in self._left_blocks.keys() if k in self._right_blocks]
        
        # Log DEBUG: joining info
        self.logger.debug(f"Joining blocking key values: {len(self._left_blocks)} x {len(self._right_blocks)} blocks")
        
        # Log INFO: final block count
        self.logger.info(f"created {len(self._common_keys)} blocks from blocking keys")
        
        # Log DEBUG: block size distribution and key values
        self._log_block_statistics()
        
        # Write debug CSV file (like Winter framework)
        self._write_debug_file()

    def estimate_pairs(self) -> Optional[int]:
        est = 0
        for key in self._common_keys:
            est += len(self._left_blocks[key]) * len(self._right_blocks[key])
        return est

    @staticmethod
    def _build_blocks(df: pd.DataFrame, cols: List[str], id_column: str, preprocess: Optional[Callable[[str], str]] = None) -> dict:
        for col in cols:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found for blocking")

        # Create a block key column tuple; cast to string and apply preprocessing to avoid NaN issues
        key_series = df[cols].astype(str)
        if preprocess:
            key_series = key_series.apply(lambda x: x.apply(preprocess))
        else:
            key_series = key_series.apply(lambda x: x.str.lower())
        key_series = key_series.agg("||".join, axis=1)
        blocks: dict[str, List[str]] = {}
        for record_id, key in zip(df[id_column], key_series):
            if key not in blocks:
                blocks[key] = []
            blocks[key].append(record_id)
        return blocks

    def _log_block_statistics(self) -> None:
        """Log DEBUG-level block size distribution and key values like Winter framework."""
        if not self._common_keys:
            return
            
        # Calculate block sizes (number of pairs per block)
        block_sizes = []
        for key in self._common_keys:
            block_size = len(self._left_blocks[key]) * len(self._right_blocks[key])
            block_sizes.append(block_size)
        
        # Log block size distribution
        size_counter = Counter(block_sizes)
        self.logger.debug("Block size distribution:")
        self.logger.debug("Size Frequency")
        # Sort by frequency descending, then by element size descending
        for size, freq in sorted(size_counter.items(), key=lambda x: (-x[1], -x[0])):
            self.logger.debug(f"{freq:<11} {size}")
        
        # Log blocking key values with frequencies (top entries only to avoid spam)
        self.logger.debug("Blocking key values:")
        self.logger.debug("BlockingKeyValue\tFrequency")
        
        # Sort by frequency descending, show top entries
        key_freqs = [(key, len(self._left_blocks[key]) * len(self._right_blocks[key])) 
                    for key in self._common_keys]
        key_freqs.sort(key=lambda x: -x[1])
        
        for key, freq in key_freqs[:20]:  # Show top 20 like Winter
            # Truncate very long keys for readability
            display_key = key if len(key) <= 50 else key[:47] + "..."
            self.logger.debug(f"{display_key}\t\t\t{freq}")

    def _write_debug_file(self) -> None:
        """Write debug CSV file with blocking key values and frequencies like Winter framework."""
        if not self._common_keys:
            return
            
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Calculate frequencies for each blocking key
        debug_data = []
        for key in self._common_keys:
            freq = len(self._left_blocks[key]) * len(self._right_blocks[key])
            debug_data.append({"Blocking Key Value": key, "Frequency": freq})
        
        # Sort by frequency descending (like Winter)
        debug_data.sort(key=lambda x: -x["Frequency"])
        
        # Write to CSV file
        debug_file = os.path.join(self.output_dir, "debugResultsBlocking_StandardBlocker.csv")
        debug_df = pd.DataFrame(debug_data)
        debug_df.to_csv(debug_file, index=False)
        
        self.logger.info(f"Debug results written to file: {debug_file}")

    def _iter_batches(self) -> Iterator[CandidateBatch]:
        if not self._common_keys:
            return

        # Log DEBUG: Creating candidate pairs
        self.logger.debug(f"Creating candidate record pairs from {len(self._common_keys)} blocks")
        
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


__all__ = ["StandardBlocker"]


