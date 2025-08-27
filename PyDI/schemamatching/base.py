"""
Base classes for schema matching.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, List, Optional

import pandas as pd

SchemaMapping = pd.DataFrame


def get_schema_columns(df: pd.DataFrame) -> List[str]:
    """Get columns from a DataFrame excluding PyDI-generated ID columns.
    
    PyDI automatically adds unique identifier columns during data loading
    (typically named {dataset_name}_id). These should be excluded from 
    schema matching as they are not part of the original data schema.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to extract schema columns from.
        
    Returns
    -------
    List[str]
        List of column names excluding PyDI ID columns.
    """
    columns = list(df.columns)
    
    # Check provenance metadata for the ID column name
    provenance = df.attrs.get("provenance", {})
    id_column_name = provenance.get("id_column_name")
    
    if id_column_name and id_column_name in columns:
        columns.remove(id_column_name)
    else:
        # Fallback: remove columns that match {dataset_name}_id pattern
        dataset_name = df.attrs.get("dataset_name")
        if dataset_name:
            expected_id_col = f"{dataset_name}_id"
            if expected_id_col in columns:
                columns.remove(expected_id_col)
    
    return columns


class BaseSchemaMatcher(ABC):
    """Abstract base class for schema matchers.

    Schema matchers take two DataFrames and return a
    ``SchemaMapping`` containing column correspondences.
    """

    @abstractmethod
    def match(
        self,
        source_dataset: pd.DataFrame,
        target_dataset: pd.DataFrame,
        preprocess: Optional[Callable[[str], str]] = None,
        threshold: float = 0.8,
        **kwargs,
    ) -> SchemaMapping:
        """Find attribute correspondences between two datasets.

        Parameters
        ----------
        source_dataset : pandas.DataFrame
            The source dataset. Must have a meaningful ``dataset_name`` entry in ``df.attrs``.
        target_dataset : pandas.DataFrame
            The target dataset. Must have a meaningful ``dataset_name`` entry in ``df.attrs``.
        preprocess : callable, optional
            A function applied to values before comparison (e.g., ``str.lower``).
            Applied to column names in label-based matching, to data values in
            instance-based matching, and to record values in duplicate-based matching.
        threshold : float, optional
            Minimum similarity/confidence score required to include a mapping. 
            Default is 0.8.
        **kwargs
            Additional keyword arguments specific to the matcher implementation.
            For example, DuplicateBasedSchemaMatcher requires a ``correspondences``
            parameter, and LabelBasedSchemaMatcher accepts a ``method`` parameter.

        Returns
        -------
        SchemaMapping
            A DataFrame with columns ``source_dataset``, ``source_column``,
            ``target_dataset``, ``target_column``, ``score`` and optional ``notes``.
        """
        raise NotImplementedError
