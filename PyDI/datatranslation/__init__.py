"""
Data translation utilities for PyDI.

This subpackage defines abstract base classes and simple implementations
for translating column names using schema mappings. Most classes are 
intentionally skeletal and are intended to be extended to support more 
complex use cases.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseTranslator(ABC):
    """Abstract base class for schema translation."""

    @abstractmethod
    def translate(self, df: pd.DataFrame, corr: pd.DataFrame) -> pd.DataFrame:
        """Translate column names according to a schema mapping.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame to translate.
        corr : pandas.DataFrame
            Schema mapping with columns ``source_dataset``, ``source_column``,
            ``target_dataset`` and ``target_column``. Only mappings where
            ``source_dataset`` matches ``df.attrs["dataset_name"]`` are used.

        Returns
        -------
        pandas.DataFrame
            A copy of ``df`` with columns renamed.
        """
        raise NotImplementedError


class MappingTranslator(BaseTranslator):
    """Translate column names based on an explicit mapping."""

    def translate(self, df: pd.DataFrame, corr: pd.DataFrame) -> pd.DataFrame:
        dataset_name = df.attrs.get("dataset_name")
        if dataset_name is None:
            raise ValueError("DataFrame is missing 'dataset_name' in attrs")
        mapping = {}
        for _, row in corr.iterrows():
            if row["source_dataset"] == dataset_name:
                mapping[row["source_column"]] = row["target_column"]
        return df.rename(columns=mapping, copy=True)
