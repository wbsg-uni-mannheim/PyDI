"""
Token-based blocking with simple alphanumeric tokenizer and overlap.
"""

from __future__ import annotations

import logging
import os
from typing import Callable, Iterator, List, Optional, Tuple
from collections import Counter

import pandas as pd

from .base import BaseBlocker, CandidateBatch


class TokenBlocking(BaseBlocker):
    """Token-based blocking using token overlap on a text column.

    Each record is assigned to blocks by its tokens. Candidate pairs are
    generated when left/right records share at least one token. A simple
    tokenizer is used by default, but a custom callable may be provided.
    """

    def __init__(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        column: str,
        id_column: str,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
        *,
        batch_size: int = 100_000,
        min_token_len: int = 2,
        output_dir: str = "output",
    ) -> None:
        super().__init__(df_left, df_right, id_column, batch_size=batch_size)
        if column not in self.df_left.columns or column not in self.df_right.columns:
            raise ValueError(f"Column '{column}' must exist in both datasets")
        self.column = column
        self.tokenizer = tokenizer or self._default_tokenizer
        self.min_token_len = int(min_token_len)
        self.output_dir = output_dir

        # Setup logging
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        
        # Log DEBUG: Creating token index
        self.logger.debug(f"Creating token index for dataset1: {len(self.df_left)} records")
        self._left_tokens = self._build_token_index(self.df_left, self.column, self.id_column)

        self.logger.debug(f"Creating token index for dataset2: {len(self.df_right)} records")
        self._right_tokens = self._build_token_index(
            self.df_right, self.column, self.id_column)
            
        # Log INFO: token counts
        self.logger.info(f"created {len(self._left_tokens)} token keys for first dataset")
        self.logger.info(f"created {len(self._right_tokens)} token keys for second dataset")

        self._common_tokens = [
            t for t in self._left_tokens.keys() if t in self._right_tokens]
        
        # Log DEBUG: joining info
        self.logger.debug(f"Joining token keys: {len(self._left_tokens)} x {len(self._right_tokens)} tokens")
        
        # Log INFO: final token count
        self.logger.info(f"created {len(self._common_tokens)} blocks from token keys")
        
        # Log DEBUG: token statistics
        self._log_token_statistics()
        
        # Write debug CSV file (like Winter framework)
        self._write_debug_file()

    @staticmethod
    def _default_tokenizer(text: str) -> List[str]:
        if text is None:
            return []
        s = str(text).lower()
        tokens = []
        current = []
        for ch in s:
            if ch.isalnum():
                current.append(ch)
            else:
                if current:
                    tokens.append("".join(current))
                    current = []
        if current:
            tokens.append("".join(current))
        uniq = []
        seen = set()
        for t in tokens:
            if len(t) >= 2 and t not in seen:
                seen.add(t)
                uniq.append(t)
        return uniq

    def _build_token_index(self, df: pd.DataFrame, column: str, id_column: str) -> dict:
        index: dict[str, List[str]] = {}
        for record_id, val in zip(df[id_column], df[column]):
            for tok in self.tokenizer(val):
                if len(tok) < self.min_token_len:
                    continue
                if tok not in index:
                    index[tok] = []
                index[tok].append(record_id)
        return index

    def _log_token_statistics(self) -> None:
        """Log DEBUG-level token statistics like Winter framework."""
        if not self._common_tokens:
            return
            
        # Calculate token frequencies (number of pairs per token)
        token_freqs = []
        for tok in self._common_tokens:
            freq = len(self._left_tokens[tok]) * len(self._right_tokens[tok])
            token_freqs.append(freq)
        
        # Log token frequency distribution
        freq_counter = Counter(token_freqs)
        self.logger.debug("Token frequency distribution:")
        self.logger.debug("Frequency   Element")
        # Sort by frequency descending, then by element size descending
        for freq, count in sorted(freq_counter.items(), key=lambda x: (-x[1], -x[0])):
            self.logger.debug(f"{count:<11} {freq}")
        
        # Log token values with frequencies (top entries only)
        self.logger.debug("Token values:")
        self.logger.debug("Token\t\t\tFrequency")
        
        # Sort by frequency descending, show top entries
        token_freq_pairs = [(tok, len(self._left_tokens[tok]) * len(self._right_tokens[tok])) 
                           for tok in self._common_tokens]
        token_freq_pairs.sort(key=lambda x: -x[1])
        
        for tok, freq in token_freq_pairs[:20]:  # Show top 20 like Winter
            # Truncate very long tokens for readability
            display_tok = tok if len(tok) <= 20 else tok[:17] + "..."
            self.logger.debug(f"{display_tok}\t\t\t{freq}")

    def _write_debug_file(self) -> None:
        """Write debug CSV file with token values and frequencies like Winter framework."""
        if not self._common_tokens:
            return
            
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Calculate frequencies for each token
        debug_data = []
        for tok in self._common_tokens:
            freq = len(self._left_tokens[tok]) * len(self._right_tokens[tok])
            debug_data.append({"Blocking Key Value": tok, "Frequency": freq})
        
        # Sort by frequency descending (like Winter)
        debug_data.sort(key=lambda x: -x["Frequency"])
        
        # Write to CSV file
        debug_file = os.path.join(self.output_dir, "debugResultsBlocking_TokenBlocking.csv")
        debug_df = pd.DataFrame(debug_data)
        debug_df.to_csv(debug_file, index=False)
        
        self.logger.info(f"Debug results written to file: {debug_file}")

    def estimate_pairs(self) -> int | None:
        est = 0
        for tok in self._common_tokens:
            est += len(self._left_tokens[tok]) * len(self._right_tokens[tok])
        return est

    def _iter_batches(self) -> Iterator[CandidateBatch]:
        if not self._common_tokens:
            return

        # Log DEBUG: Creating candidate pairs
        self.logger.debug(f"Creating candidate record pairs from {len(self._common_tokens)} token blocks")

        batch_pairs: List[Tuple[str, str]] = []
        # avoid duplicates across tokens
        seen_pairs: set[Tuple[str, str]] = set()

        for tok in self._common_tokens:
            left_ids = self._left_tokens[tok]
            right_ids = self._right_tokens[tok]
            for lid in left_ids:
                for rid in right_ids:
                    pair = (lid, rid)
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    batch_pairs.append(pair)
                    if len(batch_pairs) >= self.batch_size:
                        yield self._emit_batch(
                            pd.DataFrame(batch_pairs, columns=[
                                         "id1", "id2"]).assign(block_key=tok)
                        )
                        batch_pairs = []

        if batch_pairs:
            yield self._emit_batch(pd.DataFrame(batch_pairs, columns=["id1", "id2"]))


__all__ = ["TokenBlocking"]
