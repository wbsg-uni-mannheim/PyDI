"""
Schema matching tools for PyDI.

This module defines a simple DataFrame‑based representation of schema
mappings and provides a baseline schema matcher as well as evaluation
utilities. The goal is to avoid complex class hierarchies and to keep
matching logic inspectable and extensible.
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from typing import Callable, Iterable, List, Optional

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


class SchemaMappingEvaluator:
    """Evaluate schema mapping quality against a gold standard.

    Methods in this class compute precision, recall and F1 scores
    comparing an automatically produced mapping against a reference
    (test) mapping. Optionally, metrics can be aggregated by dataset
    or column.
    """

    @staticmethod
    def evaluate(
        corr: SchemaMapping,
        test_set: SchemaMapping,
        *,
        threshold: Optional[float] = None,
        by: Optional[List[str]] = None,
    ) -> dict:
        """Compute precision, recall and F1 for a mapping.

        Parameters
        ----------
        corr : SchemaMapping
            The correspondences produced by a matcher.
        test_set : SchemaMapping
            The gold standard mapping.
        threshold : float, optional
            If provided, ignore correspondences with a score below this value.
        by : list of str, optional
            If provided, aggregate metrics by these columns (e.g.,
            ``["source_dataset"]``).

        Returns
        -------
        dict
            A dictionary with precision, recall and F1. If ``by`` is
            provided, the dictionary contains nested metrics per group.
        """
        if threshold is not None:
            corr = corr[corr["score"] >= threshold]
        # create tuples for comparison
        corr_set = set(
            zip(
                corr["source_dataset"],
                corr["source_column"],
                corr["target_dataset"],
                corr["target_column"],
            )
        )
        test_set_pairs = set(
            zip(
                test_set["source_dataset"],
                test_set["source_column"],
                test_set["target_dataset"],
                test_set["target_column"],
            )
        )
        tp = len(corr_set & test_set_pairs)
        fp = len(corr_set - test_set_pairs)
        fn = len(test_set_pairs - corr_set)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        metrics = {"precision": precision, "recall": recall, "f1": f1}
        # group metrics
        if by:
            grouped = {}
            for key in by:
                groups = corr.groupby(key)
                test_groups = test_set.groupby(key)
                for group_key in set(groups.groups.keys()) | set(test_groups.groups.keys()):
                    corr_group = groups.get_group(group_key) if group_key in groups.groups else pd.DataFrame([])
                    test_group = test_groups.get_group(group_key) if group_key in test_groups.groups else pd.DataFrame([])
                    grouped[group_key] = SchemaMappingEvaluator.evaluate(
                        corr_group, test_group, threshold=threshold, by=None
                    )
            metrics["by"] = grouped
        return metrics

    @staticmethod
    def sweep_thresholds(
        corr: SchemaMapping,
        gold: SchemaMapping,
        *,
        thresholds: Iterable[float],
    ) -> pd.DataFrame:
        """Compute precision and recall for multiple thresholds.

        Parameters
        ----------
        corr : SchemaMapping
            The correspondences produced by a matcher.
        gold : SchemaMapping
            The gold standard mapping.
        thresholds : iterable of float
            A sequence of threshold values to evaluate.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with columns ``threshold``, ``precision``, ``recall`` and ``f1``.
        """
        records = []
        for t in thresholds:
            m = SchemaMappingEvaluator.evaluate(corr, gold, threshold=t)
            records.append({"threshold": t, **m})
        return pd.DataFrame(records)