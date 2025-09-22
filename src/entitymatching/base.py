"""
Base classes and data structures for entity matching.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Optional
import logging

import pandas as pd

# Core data structures
CorrespondenceSet = pd.DataFrame


class BaseMatcher(ABC):
    """Abstract base class for entity matchers.

    Entity matchers take candidate pairs and return correspondences
    (entity matches) with similarity scores.
    """

    @abstractmethod
    def match(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        candidates: Iterable[pd.DataFrame],
        id_column: str,
        threshold: float = 0.0,
        **kwargs,
    ) -> CorrespondenceSet:
        """Find entity correspondences between two datasets.

        Parameters
        ----------
        df_left : pandas.DataFrame
            Left dataset with specified ID column.
        df_right : pandas.DataFrame
            Right dataset with specified ID column.
        candidates : Iterable[pandas.DataFrame]
            Iterable of candidate pair batches. Each batch should have
            columns id1, id2 representing candidate pairs to compare.
        id_column : str
            Name of the column containing record identifiers.
        threshold : float, optional
            Minimum similarity score to include in results. Default is 0.0.
        **kwargs
            Additional keyword arguments specific to the matcher implementation.

        Returns
        -------
        CorrespondenceSet
            DataFrame with columns id1, id2, score, notes containing
            entity correspondences above the threshold.
        """
        raise NotImplementedError

    def _validate_inputs(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        id_column: str,
    ) -> None:
        """Validate input DataFrames.

        Parameters
        ----------
        df_left : pandas.DataFrame
            Left dataset.
        df_right : pandas.DataFrame
            Right dataset.
        id_column : str
            Name of the ID column.

        Raises
        ------
        ValueError
            If required columns are missing.
        """
        for name, df in [("left", df_left), ("right", df_right)]:
            if id_column not in df.columns:
                raise ValueError(f"{name} dataset must have '{id_column}' column")

    def _log_matching_info(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        candidates: Iterable[pd.DataFrame],
    ) -> None:
        """Log information about the matching process.

        Parameters
        ----------
        df_left : pandas.DataFrame
            Left dataset.
        df_right : pandas.DataFrame
            Right dataset.
        candidates : Iterable[pd.DataFrame]
            Candidate pairs.
        """
        left_name = df_left.attrs.get("dataset_name", "left")
        right_name = df_right.attrs.get("dataset_name", "right")

        logging.info(
            f"Entity matching: {left_name} ({len(df_left)} records) "
            f"<-> {right_name} ({len(df_right)} records)"
        )

        # Try to count candidates if it's materialized
        try:
            if hasattr(candidates, "__len__"):
                logging.info(f"Processing {len(candidates)} candidate batches")
            elif isinstance(candidates, pd.DataFrame):
                logging.info(f"Processing {len(candidates)} candidate pairs")
        except:
            logging.info("Processing candidate pairs (count unknown)")


class BaseComparator:
    """Base class for attribute comparators.

    Comparators compute similarity between specific attributes
    of two records.
    """

    def __init__(self, name: str):
        """Initialize the comparator.

        Parameters
        ----------
        name : str
            Name of this comparator for debugging and logging.
        """
        self.name = name

    def compare(
        self,
        record1: pd.Series,
        record2: pd.Series,
    ) -> float:
        """Compare two records and return similarity score.

        Parameters
        ----------
        record1 : pandas.Series
            First record to compare.
        record2 : pandas.Series
            Second record to compare.

        Returns
        -------
        float
            Similarity score, typically in [0, 1].
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
