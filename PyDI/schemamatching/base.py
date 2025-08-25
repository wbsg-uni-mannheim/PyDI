"""
Base classes for schema matching.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, List, Optional

import pandas as pd

SchemaMapping = pd.DataFrame


class BaseSchemaMatcher(ABC):
    """Abstract base class for schema matchers.

    Schema matchers take a list of DataFrames and return a
    ``SchemaMapping`` containing column correspondences.
    """

    @abstractmethod
    def match(
        self,
        datasets: List[pd.DataFrame],
        method: str = "label",
        preprocess: Optional[Callable[[str], str]] = None,
        threshold: float = 0.8,
    ) -> SchemaMapping:
        """Find attribute correspondences between multiple datasets.

        Parameters
        ----------
        datasets : list of pandas.DataFrame
            The datasets whose schemata should be matched. Each DataFrame must
            have a meaningful ``dataset_name`` entry in ``df.attrs``.
        method : str, optional
            Matching strategy. Currently only ``"label"`` is supported.
        preprocess : callable, optional
            A function applied to column names before comparison (e.g., ``str.lower``).
        threshold : float, optional
            Minimum similarity score required to include a mapping. Default is 0.8.

        Returns
        -------
        SchemaMapping
            A DataFrame with columns ``source_dataset``, ``source_column``,
            ``target_dataset``, ``target_column``, ``score`` and optional ``notes``.
        """
        raise NotImplementedError
