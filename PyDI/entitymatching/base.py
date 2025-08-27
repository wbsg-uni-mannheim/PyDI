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


def ensure_record_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame has global record IDs in _id column.
    
    PyDI requires global record IDs in the format {dataset_name}_{i}
    for entity matching operations.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to check for record IDs.
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with _id column ensured.
        
    Raises
    ------
    ValueError
        If dataset_name is not set in df.attrs or _id column is missing.
    """
    if "_id" not in df.columns:
        dataset_name = df.attrs.get("dataset_name")
        if not dataset_name:
            raise ValueError("DataFrame must have 'dataset_name' in df.attrs to generate IDs")
        
        df = df.copy()
        df["_id"] = [f"{dataset_name}_{i:06d}" for i in range(len(df))]
        
        # Update provenance
        provenance = df.attrs.get("provenance", [])
        provenance.append({
            "op": "ensure_record_ids",
            "params": {"dataset_name": dataset_name},
        })
        df.attrs["provenance"] = provenance
        
    return df


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
        threshold: float = 0.0,
        **kwargs,
    ) -> CorrespondenceSet:
        """Find entity correspondences between two datasets.
        
        Parameters
        ----------
        df_left : pandas.DataFrame
            Left dataset with _id column. Must have dataset_name in attrs.
        df_right : pandas.DataFrame  
            Right dataset with _id column. Must have dataset_name in attrs.
        candidates : Iterable[pandas.DataFrame]
            Iterable of candidate pair batches. Each batch should have
            columns id1, id2 representing candidate pairs to compare.
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
    ) -> None:
        """Validate input DataFrames.
        
        Parameters
        ----------
        df_left : pandas.DataFrame
            Left dataset.
        df_right : pandas.DataFrame
            Right dataset.
            
        Raises
        ------
        ValueError
            If required columns or metadata are missing.
        """
        for name, df in [("left", df_left), ("right", df_right)]:
            if "_id" not in df.columns:
                raise ValueError(f"{name} dataset must have '_id' column")
            
            dataset_name = df.attrs.get("dataset_name") 
            if not dataset_name:
                raise ValueError(f"{name} dataset must have 'dataset_name' in df.attrs")
                
        # Check for ID format consistency
        left_dataset = df_left.attrs["dataset_name"]
        right_dataset = df_right.attrs["dataset_name"]
        
        if left_dataset == right_dataset:
            logging.warning(
                f"Both datasets have the same dataset_name '{left_dataset}'. "
                "This may cause issues in correspondence tracking."
            )
    
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
        
        logging.info(f"Entity matching: {left_name} ({len(df_left)} records) "
                    f"<-> {right_name} ({len(df_right)} records)")
        
        # Try to count candidates if it's materialized
        try:
            if hasattr(candidates, '__len__'):
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