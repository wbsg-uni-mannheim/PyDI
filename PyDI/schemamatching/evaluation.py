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
    (test) mapping.
    """

    @staticmethod
    def evaluate(
        corr: SchemaMapping,
        test_set: SchemaMapping,
        *,
        threshold: Optional[float] = None,
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

        Returns
        -------
        dict
            A dictionary with precision, recall and F1.
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
        return {"precision": precision, "recall": recall, "f1": f1}

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