"""
Blocking and entity matching for PyDI.

This module defines abstractions for blocking (reducing the number of record
comparisons) and matching (scoring candidate pairs). The provided
implementations are simple and intended as a starting point for more
sophisticated algorithms.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, List, Optional

try:
    import networkx as nx  # type: ignore[import-not-found]
except ImportError:
    nx = None
import pandas as pd


class BaseBlocker(ABC):
    """Abstract base class for blocking strategies.

    Blockers produce candidate pairs of record identifiers from two
    datasets. They may stream candidates in batches to avoid materialising
    the full cross‑product. Implementations should override ``__iter__``.
    """

    def __init__(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        *,
        batch_size: int = 100_000,
    ) -> None:
        self.df_left = df_left
        self.df_right = df_right
        self.batch_size = batch_size

    @abstractmethod
    def __iter__(self) -> Iterator[pd.DataFrame]:
        pass

    def estimate_pairs(self) -> Optional[int]:
        """Estimate the total number of candidate pairs or return None.
        Override in subclasses if a cheap estimate can be computed.
        """
        return None

    def stats(self) -> dict:
        """Return a dictionary of blocker-specific statistics.
        By default returns an empty dict.
        """
        return {}

    def materialize(self) -> pd.DataFrame:
        """Materialise all candidate pairs into a single DataFrame.
        Use with caution on large datasets.
        """
        batches = list(self)
        return pd.concat(batches, ignore_index=True) if batches else pd.DataFrame([])


class NoBlocking(BaseBlocker):
    """A trivial blocker that yields the full cross‑product as a single batch."""

    def __iter__(self) -> Iterator[pd.DataFrame]:
        left_ids = self.df_left["_id"].tolist()
        right_ids = self.df_right["_id"].tolist()
        records = [{"id1": l, "id2": r} for l in left_ids for r in right_ids]
        yield pd.DataFrame(records)


class SortedNeighbourhood(BaseBlocker):
    """Sorted neighbourhood blocking.

    Records are sorted on a given key and candidate pairs are generated
    within a sliding window. This implementation is simplified: it
    produces candidate pairs by comparing records within a window of
    ``window`` positions after sorting both datasets separately. The
    results are yielded in a single batch.
    """

    def __init__(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        *,
        key: str,
        window: int = 5,
        batch_size: int = 100_000,
    ) -> None:
        super().__init__(df_left, df_right, batch_size=batch_size)
        self.key = key
        self.window = window

    def __iter__(self) -> Iterator[pd.DataFrame]:
        # sort both datasets on the key; missing keys get NaN which sort to the end
        left_sorted = self.df_left.sort_values(self.key, kind="mergesort").reset_index(drop=True)
        right_sorted = self.df_right.sort_values(self.key, kind="mergesort").reset_index(drop=True)
        records = []
        for i, rec_left in left_sorted.iterrows():
            # compute window in right_sorted
            start = max(0, i - self.window)
            end = min(len(right_sorted), i + self.window + 1)
            for _, rec_right in right_sorted.iloc[start:end].iterrows():
                records.append({"id1": rec_left["_id"], "id2": rec_right["_id"]})
                if len(records) >= self.batch_size:
                    yield pd.DataFrame(records)
                    records = []
        if records:
            yield pd.DataFrame(records)


class CorrespondenceSet:
    """Wrap a DataFrame of record correspondences and provide utilities.

    Attributes
    ----------
    df : pandas.DataFrame
        A DataFrame with at least the columns ``id1``, ``id2`` and ``score``.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def max_weight_bipartite_matching(self) -> "CorrespondenceSet":
        """Return a new CorrespondenceSet with a maximum weight matching.

        This method attempts to import ``networkx`` on demand. If the library
        is unavailable, an :class:`ImportError` is raised with a helpful
        message. The matching is computed on a graph where left IDs are
        prefixed with ``"L_"`` and right IDs with ``"R_"``; edge weights are
        taken from the ``score`` column.
        """
        if nx is None:
            raise ImportError(
                "networkx is required for max_weight_bipartite_matching; install it via pip"
            )
        G = nx.Graph()
        # add weighted edges between id1 and id2
        for _, row in self.df.iterrows():
            G.add_edge(f"L_{row['id1']}", f"R_{row['id2']}", weight=row.get("score", 1.0))
        matching = nx.algorithms.matching.max_weight_matching(G, maxcardinality=False)
        records = []
        for u, v in matching:
            if u.startswith("L_"):
                id1 = u[2:]
                id2 = v[2:]
            else:
                id1 = v[2:]
                id2 = u[2:]
            score = (
                self.df.loc[(self.df["id1"] == id1) & (self.df["id2"] == id2), "score"].max()
            )
            records.append({"id1": id1, "id2": id2, "score": score})
        return CorrespondenceSet(pd.DataFrame(records))

    def to_dataframe(self) -> pd.DataFrame:
        """Return the underlying DataFrame."""
        return self.df


class RuleBasedMatcher:
    """Match candidate record pairs using weighted comparators.

    The rule-based matcher takes an iterable of candidate pairs, a set of
    comparator functions, optional weights and a threshold. For each
    candidate pair, it computes a weighted sum of comparator scores. If
    the aggregated score exceeds ``threshold``, the pair is included in
    the result.
    """

    def match(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        candidates: Iterable[pd.DataFrame],
        comparators: List[Callable[[pd.Series, pd.Series], float]],
        *,
        weights: Optional[List[float]] = None,
        threshold: float = 0.0,
    ) -> CorrespondenceSet:
        if not comparators:
            raise ValueError("At least one comparator function must be provided.")
        if weights and len(weights) != len(comparators):
            raise ValueError("Length of weights must match number of comparators.")
        # normalise weights
        if weights is None:
            weights = [1.0 / len(comparators)] * len(comparators)
        total = []
        for batch in candidates:
            for _, row in batch.iterrows():
                id1 = row["id1"]
                id2 = row["id2"]
                rec1 = df_left[df_left["_id"] == id1]
                rec2 = df_right[df_right["_id"] == id2]
                if rec1.empty or rec2.empty:
                    continue
                rec1 = rec1.iloc[0]
                rec2 = rec2.iloc[0]
                score = 0.0
                for comp, w in zip(comparators, weights):
                    try:
                        s = comp(rec1, rec2)
                    except Exception:
                        s = 0.0
                    score += w * s
                if score >= threshold:
                    total.append({"id1": id1, "id2": id2, "score": score})
        return CorrespondenceSet(pd.DataFrame(total))