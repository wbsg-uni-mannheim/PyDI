"""
Evaluation framework for data fusion in PyDI.

This module provides tools for evaluating the quality of fusion results
against gold standard data.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import logging

from .base import FusionContext
from .strategy import DataFusionStrategy, EvaluationRule


class DataFusionEvaluator:
    """Evaluator for data fusion results against gold standard.

    Parameters
    ----------
    strategy : DataFusionStrategy
        The fusion strategy containing evaluation rules.
    """

    def __init__(self, strategy: DataFusionStrategy):
        self.strategy = strategy
        self._logger = logging.getLogger(__name__)

    def evaluate(
        self,
        fused_df: pd.DataFrame,
        fused_id_column: str,
        gold_df: pd.DataFrame,
        gold_id_column: str,
    ) -> Dict[str, float]:
        """Evaluate fused results against gold standard.

        Parameters
        ----------
        fused_df : pd.DataFrame
            The fused dataset to evaluate.
        fused_id_column : str
            ID column name in the fused dataset.
        gold_df : pd.DataFrame
            The gold standard dataset.
        gold_id_column : str
            ID column name in the gold dataset.

        Returns
        -------
        Dict[str, float]
            Dictionary of evaluation metrics.
        """
        self._logger.info("Starting fusion evaluation")

        # Align datasets by their respective ID columns
        aligned_fused, aligned_gold = self._align_datasets_two_ids(
            fused_df, fused_id_column, gold_df, gold_id_column
        )

        if aligned_fused.empty or aligned_gold.empty:
            self._logger.warning(
                "No matching records found between fused and gold datasets")
            return {"overall_accuracy": 0.0, "num_evaluated_records": 0}

        # Get attributes to evaluate
        attributes = self._get_evaluable_attributes(
            aligned_fused, aligned_gold, fused_id_column, gold_id_column)

        if not attributes:
            self._logger.warning("No common attributes found for evaluation")
            return {"overall_accuracy": 0.0, "num_evaluated_records": len(aligned_fused)}

        # Evaluate each attribute
        attribute_results = {}
        total_correct = 0
        total_evaluated = 0

        for attribute in attributes:
            results = self._evaluate_attribute(
                aligned_fused, aligned_gold, attribute
            )
            attribute_results[attribute] = results
            total_correct += results["correct_count"]
            total_evaluated += results["total_count"]

            self._logger.debug(
                f"Attribute '{attribute}': {results['accuracy']:.3f} "
                f"({results['correct_count']}/{results['total_count']})"
            )

        # Calculate overall metrics
        overall_accuracy = total_correct / total_evaluated if total_evaluated > 0 else 0.0

        # Calculate macro-average (average of individual attribute accuracies)
        individual_accuracies = [
            results["accuracy"] for results in attribute_results.values()
            if results["total_count"] > 0
        ]
        macro_accuracy = np.mean(
            individual_accuracies) if individual_accuracies else 0.0

        # Prepare result dictionary
        evaluation_results = {
            "overall_accuracy": overall_accuracy,
            "macro_accuracy": macro_accuracy,
            "micro_accuracy": overall_accuracy,  # Same as overall for this implementation
            "num_evaluated_records": len(aligned_fused),
            "num_evaluated_attributes": len(attributes),
            "total_evaluations": total_evaluated,
            "total_correct": total_correct,
        }

        # Add per-attribute results
        for attr, results in attribute_results.items():
            evaluation_results[f"{attr}_accuracy"] = results["accuracy"]
            evaluation_results[f"{attr}_count"] = results["total_count"]

        self._logger.info(
            f"Evaluation complete: {overall_accuracy:.3f} overall accuracy "
            f"({total_correct}/{total_evaluated})"
        )

        return evaluation_results

    def _align_datasets_two_ids(
        self,
        fused_df: pd.DataFrame,
        fused_id_column: str,
        gold_df: pd.DataFrame,
        gold_id_column: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Align two datasets by possibly different ID columns.

        Returns aligned DataFrames with matching records only.
        """
        # Find common IDs
        fused_ids = set(fused_df[fused_id_column].dropna().astype(str))
        gold_ids = set(gold_df[gold_id_column].dropna().astype(str))
        common_ids = fused_ids.intersection(gold_ids)

        if not common_ids:
            return pd.DataFrame(), pd.DataFrame()

        # Filter to common IDs and sort for consistent ordering
        aligned_fused = fused_df[fused_df[fused_id_column].astype(
            str).isin(common_ids)].copy()
        aligned_gold = gold_df[gold_df[gold_id_column].astype(
            str).isin(common_ids)].copy()

        aligned_fused = aligned_fused.sort_values(
            fused_id_column).reset_index(drop=True)
        aligned_gold = aligned_gold.sort_values(
            gold_id_column).reset_index(drop=True)

        return aligned_fused, aligned_gold

    def _get_evaluable_attributes(
        self,
        fused_df: pd.DataFrame,
        gold_df: pd.DataFrame,
        fused_id_column: str,
        gold_id_column: str,
    ) -> List[str]:
        """Get attributes that can be evaluated (present in both datasets)."""
        fused_attrs = set(fused_df.columns)
        gold_attrs = set(gold_df.columns)

        # Find common attributes, excluding metadata columns
        common_attrs = fused_attrs.intersection(gold_attrs)

        # Filter out metadata and ID columns
        evaluable_attrs = [
            attr for attr in common_attrs
            if not attr.startswith("_fusion_") and attr not in {fused_id_column, gold_id_column}
        ]

        return evaluable_attrs

    def _evaluate_attribute(
        self,
        fused_df: pd.DataFrame,
        gold_df: pd.DataFrame,
        attribute: str,
    ) -> Dict[str, Any]:
        """Evaluate a single attribute."""
        # Get evaluation rule for this attribute
        eval_rule = self.strategy.get_evaluation_rule(attribute)
        if eval_rule is None:
            # Use default exact equality
            eval_rule = EvaluationRule("default")

        correct_count = 0
        total_count = 0

        # Create fusion context (minimal for evaluation)
        context = FusionContext(group_id="eval", attribute=attribute)

        # Compare values row by row
        for i in range(len(fused_df)):
            fused_value = fused_df.iloc[i][attribute]
            gold_value = gold_df.iloc[i][attribute]

            # Skip if either value is missing (robust to arrays/lists)
            if self._is_missing(fused_value) or self._is_missing(gold_value):
                continue

            total_count += 1

            # Evaluate using the rule
            if eval_rule.evaluate(fused_value, gold_value, context):
                correct_count += 1

        # Calculate accuracy
        accuracy = correct_count / total_count if total_count > 0 else 0.0

        return {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": total_count,
            "rule_used": eval_rule.name,
        }

    @staticmethod
    def _is_missing(value: Any) -> bool:
        """Return True if the value should be treated as missing.

        Handles scalars, numpy arrays, pandas NA, and Python sequences.
        """
        try:
            # Pandas/NumPy aware check
            if pd.isna(value):
                return True
        except Exception:
            pass

        # Handle sequences (e.g., lists/arrays): consider missing if empty
        if isinstance(value, (list, tuple, set)):
            return len(value) == 0
        try:
            import numpy as np  # already imported at top but guard anyway
            if isinstance(value, np.ndarray):
                return value.size == 0 or np.all(pd.isna(value))
        except Exception:
            pass

        return False


class FusionQualityMetrics:
    """Calculate quality metrics for fusion results."""

    @staticmethod
    def calculate_consistency_metrics(fused_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate consistency metrics for a fused dataset.

        Parameters
        ----------
        fused_df : pd.DataFrame
            The fused dataset with fusion metadata.

        Returns
        -------
        Dict[str, float]
            Dictionary of consistency metrics.
        """
        metrics = {}

        # Overall confidence statistics
        if "_fusion_confidence" in fused_df.columns:
            confidences = fused_df["_fusion_confidence"].dropna()
            metrics["mean_confidence"] = confidences.mean()
            metrics["std_confidence"] = confidences.std()
            metrics["min_confidence"] = confidences.min()
            metrics["max_confidence"] = confidences.max()
        else:
            metrics.update({
                "mean_confidence": 0.0,
                "std_confidence": 0.0,
                "min_confidence": 0.0,
                "max_confidence": 0.0,
            })

        # Count multi-source vs single-source records
        if "_fusion_sources" in fused_df.columns:
            source_counts = fused_df["_fusion_sources"].apply(len)
            metrics["multi_source_records"] = (source_counts > 1).sum()
            metrics["single_source_records"] = (source_counts == 1).sum()
            metrics["mean_sources_per_record"] = source_counts.mean()
        else:
            metrics.update({
                "multi_source_records": 0,
                "single_source_records": len(fused_df),
                "mean_sources_per_record": 1.0,
            })

        # Fusion rule usage statistics
        if "_fusion_metadata" in fused_df.columns:
            rule_usage = {}
            for metadata in fused_df["_fusion_metadata"].dropna():
                if isinstance(metadata, dict):
                    for key, value in metadata.items():
                        if key.endswith("_rule"):
                            rule_usage[value] = rule_usage.get(value, 0) + 1

            metrics["rule_usage"] = rule_usage
            metrics["num_unique_rules"] = len(rule_usage)
        else:
            metrics["rule_usage"] = {}
            metrics["num_unique_rules"] = 0

        return metrics

    @staticmethod
    def calculate_coverage_metrics(
        datasets: List[pd.DataFrame],
        fused_df: pd.DataFrame,
    ) -> Dict[str, float]:
        """Calculate coverage metrics comparing input to output.

        Parameters
        ----------
        datasets : List[pd.DataFrame]
            Original input datasets.
        fused_df : pd.DataFrame
            The fused result dataset.

        Returns
        -------
        Dict[str, float]
            Dictionary of coverage metrics.
        """
        metrics = {}

        # Record coverage
        total_input_records = sum(len(df) for df in datasets)
        output_records = len(fused_df)
        metrics["record_coverage"] = output_records / \
            total_input_records if total_input_records > 0 else 0.0

        # Attribute coverage
        all_input_attrs = set()
        for df in datasets:
            all_input_attrs.update(df.columns)

        output_attrs = set(fused_df.columns)
        # Exclude fusion metadata columns
        output_data_attrs = {
            col for col in output_attrs if not col.startswith("_fusion_")}

        if all_input_attrs:
            metrics["attribute_coverage"] = len(
                output_data_attrs.intersection(all_input_attrs)) / len(all_input_attrs)
        else:
            metrics["attribute_coverage"] = 0.0

        metrics["total_input_records"] = total_input_records
        metrics["total_output_records"] = output_records
        metrics["total_input_attributes"] = len(all_input_attrs)
        metrics["total_output_attributes"] = len(output_data_attrs)

        return metrics
