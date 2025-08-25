"""
Schema mapping evaluation utilities.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

import pandas as pd

from .base import SchemaMapping


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