"""
Simple schema matcher based on exact label matching.
"""

from __future__ import annotations

import itertools
from typing import Callable, List, Optional

import pandas as pd

from .base import BaseSchemaMatcher, SchemaMapping


class SimpleSchemaMatcher(BaseSchemaMatcher):
    """A naive schema matcher based on column label equality.

    This matcher compares column names across datasets. It can apply
    optional preprocessing (e.g., lowercasing) and returns correspondences
    with a score of 1.0 when names match exactly. It is intended as a
    starting point for more sophisticated matching algorithms.
    """

    def match(
        self,
        datasets: List[pd.DataFrame],
        method: str = "label",
        preprocess: Optional[Callable[[str], str]] = None,
        threshold: float = 0.8,
    ) -> SchemaMapping:
        """Find schema correspondences using exact label matching.
        
        Parameters
        ----------
        datasets : list of pandas.DataFrame
            The datasets whose schemata should be matched.
        method : str, optional
            Matching method. Only "label" is supported.
        preprocess : callable, optional
            Preprocessing function for column names.
        threshold : float, optional
            Minimum similarity score for correspondences.
            
        Returns
        -------
        SchemaMapping
            DataFrame with schema correspondences.
        """
        if method != "label":
            raise ValueError(f"Unsupported method '{method}'. Only 'label' is supported.")
        results = []
        # pairwise combinations
        for i, j in itertools.combinations(range(len(datasets)), 2):
            df_i = datasets[i]
            df_j = datasets[j]
            name_i = df_i.attrs.get("dataset_name", f"ds{i}")
            name_j = df_j.attrs.get("dataset_name", f"ds{j}")
            for col_i in df_i.columns:
                for col_j in df_j.columns:
                    col_i_proc = preprocess(col_i) if preprocess else col_i
                    col_j_proc = preprocess(col_j) if preprocess else col_j
                    similarity = 1.0 if col_i_proc == col_j_proc else 0.0
                    if similarity >= threshold:
                        results.append(
                            {
                                "source_dataset": name_i,
                                "source_column": col_i,
                                "target_dataset": name_j,
                                "target_column": col_j,
                                "score": similarity,
                            }
                        )
        return pd.DataFrame(results)