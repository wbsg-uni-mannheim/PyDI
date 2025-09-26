"""
Schema mapping evaluation utilities.
"""

from __future__ import annotations

from typing import Iterable, List, Optional
import logging

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

from .base import SchemaMapping


logger = logging.getLogger(__name__)


class SchemaMappingEvaluator:
    """Evaluate schema mapping quality against an evaluation set.

    Methods in this class compute precision, recall and F1 scores
    comparing an automatically produced mapping against a reference
    (evaluation) mapping.
    """

    @staticmethod
    def evaluate(
        corr: SchemaMapping,
        evaluation_set: SchemaMapping,
        *,
        threshold: Optional[float] = None,
        complete: bool = False,
        label_column: str = "label",
    ) -> dict:
        """Compute precision, recall and F1 for a mapping.

        Parameters
        ----------
        corr : SchemaMapping
            The correspondences produced by a matcher.
        evaluation_set : SchemaMapping  
            The evaluation mapping. Can contain both positive and negative examples.
            If it has a label column with True/False values, negatives will be used.
            Otherwise, all correspondences are assumed to be positive examples.
        threshold : float, optional
            If provided, ignore correspondences with a score below this value.
        complete : bool, default False
            Whether the evaluation_set represents a complete evaluation set. If True,
            any correspondence not in the evaluation_set is considered a negative.
            If False, only explicit negatives in evaluation_set are used.
        label_column : str, default "label"
            Name of the column containing True/False labels for positive/negative examples.

        Returns
        -------
        dict
            A dictionary with precision, recall, F1, and detailed counts.
        """
        if threshold is not None:
            corr = corr[corr["score"] >= threshold]
        
        # Create tuples for comparison
        corr_set = set(
            zip(
                corr["source_dataset"],
                corr["source_column"], 
                corr["target_dataset"],
                corr["target_column"],
            )
        )
        
        # Determine positive and negative examples from evaluation_set
        has_labels = label_column in evaluation_set.columns
        if has_labels:
            # Separate positive and negative examples based on label column
            positive_eval = evaluation_set[evaluation_set[label_column] == True]
            negative_eval = evaluation_set[evaluation_set[label_column] == False] 
        else:
            # All evaluation_set examples are positive
            positive_eval = evaluation_set
            negative_eval = pd.DataFrame(columns=evaluation_set.columns)
        
        positive_set = set(
            zip(
                positive_eval["source_dataset"],
                positive_eval["source_column"],
                positive_eval["target_dataset"], 
                positive_eval["target_column"],
            )
        )
        
        negative_set = set(
            zip(
                negative_eval["source_dataset"],
                negative_eval["source_column"],
                negative_eval["target_dataset"],
                negative_eval["target_column"],
            )
        ) if not negative_eval.empty else set()
        
        # Count matches following Winter's logic
        correct = 0  # True positives
        matched = 0  # Total predicted correspondences evaluated
        missing_positives = positive_set.copy()
        
        matched_positives = set()

        for corr_tuple in corr_set:
            # Check both directions for symmetry
            reverse_tuple = (corr_tuple[2], corr_tuple[3], corr_tuple[0], corr_tuple[1])

            if corr_tuple in positive_set or reverse_tuple in positive_set:
                matched += 1
                canonical = corr_tuple if corr_tuple in positive_set else reverse_tuple

                if canonical not in matched_positives:
                    # True positive (only count once per ground-truth correspondence)
                    correct += 1
                    matched_positives.add(canonical)
                else:
                    logger.debug(f"[duplicate-correct] {corr_tuple}")

                # Remove from missing positives (check both directions)
                missing_positives.discard(corr_tuple)
                missing_positives.discard(reverse_tuple)

                logger.debug(f"[correct] {corr_tuple}")
                
            elif (complete or 
                  corr_tuple in negative_set or 
                  reverse_tuple in negative_set):
                # False positive (either complete evaluation set or explicit negative)
                matched += 1
                logger.debug(f"[wrong] {corr_tuple}")
        
        # Log missing positive examples
        for missing in missing_positives:
            logger.debug(f"[missing] {missing}")
        
        # Calculate metrics
        correct_total = len(positive_set)
        precision = correct / matched if matched > 0 else 0.0
        recall = correct / correct_total if correct_total > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        
        return {
            "precision": precision,
            "recall": recall, 
            "f1": f1,
            "correct": correct,
            "matched": matched,
            "correct_total": correct_total,
            "missing": len(missing_positives)
        }

    @staticmethod
    def sweep_thresholds(
        corr: SchemaMapping,
        evaluation_set: SchemaMapping,
        *,
        thresholds: Iterable[float],
        complete: bool = False,
        label_column: str = "label",
    ) -> pd.DataFrame:
        """Compute precision and recall for multiple thresholds.

        Parameters
        ----------
        corr : SchemaMapping
            The correspondences produced by a matcher.
        evaluation_set : SchemaMapping
            The evaluation mapping.
        thresholds : iterable of float
            A sequence of threshold values to evaluate.
        complete : bool, default False
            Whether the evaluation_set represents a complete evaluation set.
        label_column : str, default "label"
            Name of the column containing True/False labels for positive/negative examples.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with columns ``threshold``, ``precision``, ``recall``, ``f1``,
            and additional evaluation metrics.
        """
        records = []
        for t in thresholds:
            m = SchemaMappingEvaluator.evaluate(corr, evaluation_set, threshold=t, complete=complete, label_column=label_column)
            records.append({"threshold": t, **m})
        return pd.DataFrame(records)
