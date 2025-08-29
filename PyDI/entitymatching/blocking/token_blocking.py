"""
Token-based blocking with simple alphanumeric tokenizer and overlap.
"""

from __future__ import annotations

from typing import Callable, Iterator, List, Optional, Tuple

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
        tokenizer: Optional[Callable[[str], List[str]]] = None,
        *,
        batch_size: int = 100_000,
        min_token_len: int = 2,
    ) -> None:
        super().__init__(df_left, df_right, batch_size=batch_size)
        if column not in self.df_left.columns or column not in self.df_right.columns:
            raise ValueError(f"Column '{column}' must exist in both datasets")
        self.column = column
        self.tokenizer = tokenizer or self._default_tokenizer
        self.min_token_len = int(min_token_len)

        self._left_tokens = self._build_token_index(self.df_left, self.column)
        self._right_tokens = self._build_token_index(
            self.df_right, self.column)
        self._common_tokens = [
            t for t in self._left_tokens.keys() if t in self._right_tokens]

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

    def _build_token_index(self, df: pd.DataFrame, column: str) -> dict:
        index: dict[str, List[str]] = {}
        for _id, val in zip(df["_id"], df[column]):
            for tok in self.tokenizer(val):
                if len(tok) < self.min_token_len:
                    continue
                if tok not in index:
                    index[tok] = []
                index[tok].append(_id)
        return index

    def estimate_pairs(self) -> int | None:
        est = 0
        for tok in self._common_tokens:
            est += len(self._left_tokens[tok]) * len(self._right_tokens[tok])
        return est

    def _iter_batches(self) -> Iterator[CandidateBatch]:
        if not self._common_tokens:
            return

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
