"""
Base classes for data translation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import pandas as pd

# Import SchemaMapping type from schemamatching
from ..schemamatching.base import SchemaMapping


class BaseTranslator(ABC):
    """Abstract base class for schema translation.
    
    Translators take a DataFrame and a SchemaMapping and return a transformed
    DataFrame with columns renamed, added, or otherwise modified according to
    the mapping.
    """

    @abstractmethod
    def translate(self, df: pd.DataFrame, corr: SchemaMapping) -> pd.DataFrame:
        """Translate a DataFrame according to a schema mapping.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame to translate. Must have a meaningful ``dataset_name`` 
            entry in ``df.attrs``.
        corr : SchemaMapping
            Schema mapping DataFrame with columns ``source_dataset``, 
            ``source_column``, ``target_dataset``, ``target_column``, and 
            ``score``. Only mappings where ``source_dataset`` matches 
            ``df.attrs["dataset_name"]`` are applied.

        Returns
        -------
        pandas.DataFrame
            A new DataFrame with schema transformations applied. The returned
            DataFrame preserves the original dataset's attrs and adds provenance
            information about the translation operation.
            
        Raises
        ------
        ValueError
            If the input DataFrame is missing required metadata or if the
            schema mapping is invalid.
        """
        raise NotImplementedError