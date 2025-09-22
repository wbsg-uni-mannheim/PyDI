"""
Token-based blocking with simple alphanumeric tokenizer and overlap.
"""

from __future__ import annotations

import logging
import os
from typing import Callable, Iterator, List, Optional, Tuple, Literal
from collections import Counter

import pandas as pd

from .base import BaseBlocker, CandidateBatch


class TokenBlocker(BaseBlocker):
    """Token-based blocking using token overlap on a text column.

    Each record is assigned to blocks by its tokens. Candidate pairs are
    generated when left/right records share at least one token. A simple
    tokenizer is used by default, but a custom callable may be provided.

    Also supports character and word ngrams for more flexible blocking strategies.
    When using ngrams, records are blocked based on overlapping character sequences
    or word sequences of the specified size.
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
        ngram_size: Optional[int] = None,
        ngram_type: Optional[Literal["character", "word"]] = None,
        preprocess: Optional[Callable[[str], str]] = None,
    ) -> None:
        """Initialize TokenBlocker with optional ngram support.

        Args:
            df_left: Left dataset
            df_right: Right dataset
            column: Column name to use for blocking
            id_column: Column name containing record IDs
            tokenizer: Custom tokenizer function (overrides ngram settings if provided)
            batch_size: Size of candidate batches
            min_token_len: Minimum token/ngram length to consider
            output_dir: Directory for debug output files
            ngram_size: Size of ngrams to generate (requires ngram_type)
            ngram_type: Type of ngrams - 'character' or 'word' (requires ngram_size)
        """
        super().__init__(df_left, df_right, id_column, batch_size=batch_size)
        if column not in self.df_left.columns or column not in self.df_right.columns:
            raise ValueError(f"Column '{column}' must exist in both datasets")

        # Validate ngram parameters
        if ngram_size is not None and ngram_type is None:
            raise ValueError("ngram_type must be specified when ngram_size is provided")
        if ngram_type is not None and ngram_size is None:
            raise ValueError("ngram_size must be specified when ngram_type is provided")
        if ngram_size is not None and ngram_size < 1:
            raise ValueError("ngram_size must be >= 1")

        self.column = column
        self.ngram_size = ngram_size
        self.ngram_type = ngram_type
        self.min_token_len = int(min_token_len)
        self.output_dir = output_dir
        self.preprocess = preprocess

        # Select appropriate tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif ngram_size is not None:
            if ngram_type == "character":
                self.tokenizer = self._character_ngram_tokenizer
            elif ngram_type == "word":
                self.tokenizer = self._word_ngram_tokenizer
        else:
            self.tokenizer = self._default_tokenizer

        # Setup logging
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        
        # Log DEBUG: Creating token index
        self.logger.debug(f"Creating token index for dataset1: {len(self.df_left)} records")
        self._left_tokens = self._build_token_index(self.df_left, self.column, self.id_column, self.preprocess)

        self.logger.debug(f"Creating token index for dataset2: {len(self.df_right)} records")
        self._right_tokens = self._build_token_index(
            self.df_right, self.column, self.id_column, self.preprocess)
            
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

    def _character_ngram_tokenizer(self, text: str) -> List[str]:
        """Generate character ngrams from text.

        Args:
            text: Input text to tokenize

        Returns:
            List of character ngrams of specified size
        """
        if text is None or self.ngram_size is None:
            return []

        s = str(text).lower()
        # Remove non-alphanumeric for character ngrams
        clean_text = ''.join(ch for ch in s if ch.isalnum())

        if len(clean_text) < self.ngram_size:
            return []

        ngrams = []
        for i in range(len(clean_text) - self.ngram_size + 1):
            ngram = clean_text[i:i + self.ngram_size]
            if len(ngram) >= self.min_token_len:
                ngrams.append(ngram)

        # Remove duplicates while preserving order
        seen = set()
        unique_ngrams = []
        for ngram in ngrams:
            if ngram not in seen:
                seen.add(ngram)
                unique_ngrams.append(ngram)

        return unique_ngrams

    def _word_ngram_tokenizer(self, text: str) -> List[str]:
        """Generate word ngrams from text.

        Args:
            text: Input text to tokenize

        Returns:
            List of word ngrams of specified size
        """
        if text is None or self.ngram_size is None:
            return []

        # First get individual tokens using the default tokenizer
        tokens = self._default_tokenizer(text)

        if len(tokens) < self.ngram_size:
            return []

        ngrams = []
        for i in range(len(tokens) - self.ngram_size + 1):
            ngram = ' '.join(tokens[i:i + self.ngram_size])
            if len(ngram) >= self.min_token_len:
                ngrams.append(ngram)

        # Remove duplicates while preserving order
        seen = set()
        unique_ngrams = []
        for ngram in ngrams:
            if ngram not in seen:
                seen.add(ngram)
                unique_ngrams.append(ngram)

        return unique_ngrams

    def _build_token_index(self, df: pd.DataFrame, column: str, id_column: str, preprocess: Optional[Callable[[str], str]] = None) -> dict:
        index: dict[str, List[str]] = {}
        for record_id, val in zip(df[id_column], df[column]):
            processed_val = preprocess(str(val)) if preprocess and val is not None else val
            for tok in self.tokenizer(processed_val):
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
        self.logger.debug("Size Frequency")
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
        debug_file = os.path.join(self.output_dir, "debugResultsBlocking_TokenBlocker.csv")
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


__all__ = ["TokenBlocker"]
