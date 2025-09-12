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

from .base import FusionContext, get_callable_name
from .strategy import DataFusionStrategy
from ..utils.similarity_registry import SimilarityRegistry


def _is_missing_value(value) -> bool:
    """Helper function to check if a value should be treated as missing.

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
        import numpy as np
        if isinstance(value, np.ndarray):
            return value.size == 0 or np.all(pd.isna(value))
    except Exception:
        pass

    return False


def exact_match(fused_value, gold_value) -> bool:
    """Default evaluation function using exact equality."""
    return fused_value == gold_value


def tokenized_match(fused_value, gold_value, threshold: float = 1.0) -> bool:
    """Evaluation function using tokenized comparison with similarity threshold.

    For lists: Uses Jaccard similarity between lists (order doesn't matter).
    For strings: Tokenizes and uses Jaccard similarity between token sets.
    Useful for actor lists and titles where order and partial matches matter.

    Parameters
    ----------
    fused_value : Any
        The fused value to compare.
    gold_value : Any  
        The gold standard value to compare against.
    threshold : float, default 1.0
        Minimum similarity threshold (0.0 to 1.0). 
        1.0 requires exact match, lower values allow partial matches.

    Returns
    -------
    bool
        True if similarity >= threshold, False otherwise.
    """
    # Check for missing values using the same logic as DataFusionEvaluator
    if _is_missing_value(fused_value) and _is_missing_value(gold_value):
        return True
    if _is_missing_value(fused_value) or _is_missing_value(gold_value):
        return False

    # Get Jaccard similarity function from registry
    jaccard_sim = SimilarityRegistry.get_function('jaccard')

    # Handle lists of strings - use Jaccard similarity
    if isinstance(fused_value, list) and isinstance(gold_value, list):
        # Use Jaccard similarity between sets (order doesn't matter)
        similarity = jaccard_sim(set(fused_value), set(gold_value))
        return similarity >= threshold

    # Handle mixed list/string by converting both to lists
    if isinstance(fused_value, list) or isinstance(gold_value, list):
        # Convert both to lists, then use Jaccard similarity
        fused_list = fused_value if isinstance(
            fused_value, list) else [str(fused_value)]
        gold_list = gold_value if isinstance(
            gold_value, list) else [str(gold_value)]
        similarity = jaccard_sim(set(fused_list), set(gold_list))
        return similarity >= threshold

    # String tokenization logic - clean and use Jaccard similarity
    import string

    def clean_tokens(text):
        # Split into words and remove punctuation
        words = str(text).lower().split()
        clean_words = []
        for word in words:
            # Remove punctuation from each word
            cleaned = word.translate(str.maketrans('', '', string.punctuation))
            if cleaned:  # Only keep non-empty words
                clean_words.append(cleaned)
        return set(clean_words)

    fused_tokens = clean_tokens(fused_value)
    gold_tokens = clean_tokens(gold_value)

    # Use Jaccard similarity between token sets
    if len(fused_tokens) == 0 and len(gold_tokens) == 0:
        return True  # Both empty
    if len(fused_tokens) == 0 or len(gold_tokens) == 0:
        return False  # One empty, one not

    similarity = jaccard_sim(fused_tokens, gold_tokens)
    return similarity >= threshold


def year_only_match(fused_value, gold_value) -> bool:
    """Evaluation function comparing only the year part of dates.

    Useful for dates where "2010-01-01" should match "2010-12-31".
    """
    if _is_missing_value(fused_value) and _is_missing_value(gold_value):
        return True
    if _is_missing_value(fused_value) or _is_missing_value(gold_value):
        return False

    # Extract year from date strings (first 4 characters)
    fused_year = str(fused_value)[:4] if len(str(fused_value)) >= 4 else None
    gold_year = str(gold_value)[:4] if len(str(gold_value)) >= 4 else None

    return fused_year == gold_year and fused_year is not None


def numeric_tolerance_match(fused_value, gold_value, tolerance: float = 0.01) -> bool:
    """Evaluation function for numeric values with tolerance."""
    if _is_missing_value(fused_value) and _is_missing_value(gold_value):
        return True
    if _is_missing_value(fused_value) or _is_missing_value(gold_value):
        return False

    try:
        return abs(float(fused_value) - float(gold_value)) <= tolerance
    except (ValueError, TypeError):
        return str(fused_value).strip() == str(gold_value).strip()


def set_equality_match(fused_value, gold_value) -> bool:
    """Evaluation function for set equality (order-independent).

    Useful for lists where order doesn't matter.
    """
    if _is_missing_value(fused_value) and _is_missing_value(gold_value):
        return True
    if _is_missing_value(fused_value) or _is_missing_value(gold_value):
        return False

    try:
        if isinstance(fused_value, (list, tuple, set)) and isinstance(gold_value, (list, tuple, set)):
            return set(fused_value) == set(gold_value)
        return fused_value == gold_value
    except (TypeError, ValueError):
        return str(fused_value) == str(gold_value)


def boolean_match(fused_value, gold_value) -> bool:
    """Evaluation function for boolean values with flexible interpretation.

    Handles various boolean representations:
    - True/False, true/false, yes/no, 1/0, "true"/"false", etc.
    """
    if _is_missing_value(fused_value) and _is_missing_value(gold_value):
        return True
    if _is_missing_value(fused_value) or _is_missing_value(gold_value):
        return False

    def normalize_boolean(value):
        """Convert various boolean representations to True/False."""
        if isinstance(value, bool):
            return value

        # Convert to string and normalize
        str_val = str(value).lower().strip()

        # True values
        if str_val in ['true', 'yes', '1', 'y', 't']:
            return True
        # False values
        elif str_val in ['false', 'no', '0', 'n', 'f', '']:
            return False
        # Handle None/null values
        elif str_val in ['none', 'null', 'nan']:
            return None
        else:
            # Try to convert to bool directly
            try:
                return bool(value)
            except:
                return None

    # Normalize both values
    fused_bool = normalize_boolean(fused_value)
    gold_bool = normalize_boolean(gold_value)

    # If either couldn't be normalized, fall back to string comparison
    if fused_bool is None or gold_bool is None:
        return str(fused_value).strip().lower() == str(gold_value).strip().lower()

    return fused_bool == gold_bool


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
        # Get evaluation function for this attribute
        eval_function = self.strategy.get_evaluation_function(attribute)
        if eval_function is None:
            # Use default exact equality
            eval_function = exact_match

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

            # Evaluate using the function
            if eval_function(fused_value, gold_value):
                correct_count += 1

        # Calculate accuracy
        accuracy = correct_count / total_count if total_count > 0 else 0.0

        return {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": total_count,
            "rule_used": get_callable_name(eval_function),
        }

    @staticmethod
    def _is_missing(value: Any) -> bool:
        """Delegate to the module-level missing-value check."""
        # Use the module-level helper defined above
        return _is_missing_value(value)


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
