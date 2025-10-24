"""Evaluation framework for information extraction in PyDI.

This module provides an evaluator to compare extracted (predicted)
attributes against an expected/validation dataset with common IR-style metrics.

It supports per-attribute evaluation functions (e.g., exact match,
tokenized string match, numeric tolerance) and produces both attribute-level
and dataset-level (micro/macro) metrics, including precision, recall, F1, and
non-null accuracy.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path
import json
import logging

import pandas as pd

# Reuse flexible evaluation functions from the fusion module
from PyDI.fusion.evaluation import (
    exact_match,
    tokenized_match,
    numeric_tolerance_match,
    set_equality_match,
    boolean_match,
    year_only_match,
)


logger = logging.getLogger(__name__)

# Type alias for evaluation functions (return True when values "match")
EvaluationFunction = Callable[[Any, Any], bool]


def _is_missing_value(value: Any) -> bool:
    """Return True if the value should be treated as missing.

    Handles scalars, pandas NA, numpy arrays, and Python sequences
    (treats empty sequences as missing).
    """
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass

    # Empty sequences are considered missing
    if isinstance(value, (list, tuple, set)):
        return len(value) == 0

    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return value.size == 0 or (hasattr(value, "dtype") and pd.isna(value).all())
    except Exception:
        pass

    return False


def _align_two_by_id(
    pred_df: pd.DataFrame,
    pred_id_column: str,
    expected_df: pd.DataFrame,
    expected_id_column: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Align two DataFrames on possibly different ID columns (inner join by IDs)."""
    pred_ids = set(pred_df[pred_id_column].dropna().astype(str))
    gold_ids = set(expected_df[expected_id_column].dropna().astype(str))
    common_ids = pred_ids & gold_ids
    if not common_ids:
        return pd.DataFrame(), pd.DataFrame()

    pred_aligned = pred_df[pred_df[pred_id_column].astype(str).isin(common_ids)].copy()
    gold_aligned = expected_df[expected_df[expected_id_column].astype(str).isin(common_ids)].copy()

    pred_aligned = pred_aligned.sort_values(pred_id_column).reset_index(drop=True)
    gold_aligned = gold_aligned.sort_values(expected_id_column).reset_index(drop=True)
    return pred_aligned, gold_aligned


def _micro_metrics_from_counts(total_counts: Dict[str, int]) -> Dict[str, float]:
    """Compute precision, recall, F1, and accuracy (non-null) from category counts.

    Category semantics:
    - VC: predicted present and correct (TP)
    - VW: predicted present and wrong value (FP)
    - VN: predicted missing but expected present (FN)
    - NV: predicted present but expected missing (FP)
    - NN: both missing (TN)
    """
    tp = total_counts.get("VC", 0)
    fp = total_counts.get("VW", 0) + total_counts.get("NV", 0)
    fn = total_counts.get("VN", 0)
    tn = total_counts.get("NN", 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Non-null accuracy excludes cases where both are missing (TN=NN)
    denom_non_null = tp + fp + fn
    accuracy = tp / denom_non_null if denom_non_null > 0 else 0.0

    # Overall accuracy (includes TN); can be misleading if many NN
    total = tp + fp + fn + tn
    accuracy_overall = (tp + tn) / total if total > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "accuracy_overall": accuracy_overall,
    }


class InformationExtractionEvaluator:
    """Evaluate information extraction results against an expected/validation dataset.

    Parameters
    ----------
    evaluation_functions : Optional[Dict[str, EvaluationFunction]]
        Optional mapping from attribute name to evaluation function.
        Defaults to exact equality for unspecified attributes.
    default_function : Optional[EvaluationFunction]
        Default evaluation function when none is registered for an attribute.
        If None, uses exact_match.
    """

    def __init__(
        self,
        evaluation_functions: Optional[Dict[str, EvaluationFunction]] = None,
        *,
        default_function: Optional[EvaluationFunction] = None,
    ) -> None:
        self._eval_fns: Dict[str, EvaluationFunction] = dict(evaluation_functions or {})
        self._default_fn: EvaluationFunction = default_function or exact_match
        self._logger = logging.getLogger(__name__)

    def set_evaluation_function(self, attribute: str, fn: EvaluationFunction) -> None:
        self._eval_fns[attribute] = fn

    def get_evaluation_function(self, attribute: str) -> EvaluationFunction:
        return self._eval_fns.get(attribute, self._default_fn)

    def evaluate(
        self,
        predictions_df: pd.DataFrame,
        gold_df: Optional[pd.DataFrame] = None,
        *,
        pred_id_column: str,
        gold_id_column: Optional[str] = None,
        attributes: Optional[List[str]] = None,
        debug_mismatches: bool = False,
        debug_file: Optional[Union[str, Path]] = None,
        debug_format: str = "text",
        expected_df: Optional[pd.DataFrame] = None,
        expected_id_column: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate predictions against an expected/validation dataset.

        Returns a dictionary with attribute-level results and micro/macro metrics.
        """
        self._logger.info("Starting information extraction evaluation")

        # Backward-compat: support old gold_* parameter names
        if expected_df is None and gold_df is not None:
            expected_df = gold_df
        if expected_id_column is None and gold_id_column is not None:
            expected_id_column = gold_id_column

        # Align
        pred_aligned, gold_aligned = _align_two_by_id(
            predictions_df, pred_id_column, expected_df, expected_id_column
        )
        if pred_aligned.empty or gold_aligned.empty:
            self._logger.warning("No matching records found between predictions and expected/validation dataset")
            return {
                "micro": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0, "accuracy_overall": 0.0},
                "macro": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0, "accuracy_overall": 0.0},
                "attributes": {},
                "num_evaluated_records": 0,
            }

        # Determine attributes to evaluate
        if attributes is None:
            common = set(pred_aligned.columns) & set(gold_aligned.columns)
            attributes = [
                a for a in common if a not in {pred_id_column, expected_id_column}
            ]

        # Setup mismatch logging if requested
        mismatch_log: Optional[Path] = None
        write_json = False
        if debug_mismatches:
            log_path = Path(debug_file) if debug_file is not None else Path("ie_eval_mismatches.log")
            mismatch_log = log_path
            try:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("w", encoding="utf-8") as f:
                    if debug_format == "json":
                        write_json = True
                        f.write(json.dumps({"type": "header", "format": "jsonl"}) + "\n")
                    else:
                        f.write("=== IE Evaluation Mismatches ===\n\n")
            except Exception as e:
                self._logger.warning(f"Could not initialize mismatch log: {e}")
                mismatch_log = None

        def emit_mismatch(entry: Dict[str, Any]) -> None:
            if mismatch_log is None:
                return
            try:
                with mismatch_log.open("a", encoding="utf-8") as f:
                    if write_json:
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    else:
                        rid = entry.get("record_id", "?")
                        attr = entry.get("attribute", "?")
                        evf = entry.get("evaluation_function", "?")
                        pred = entry.get("predicted_value")
                        gold = entry.get("expected_value")
                        f.write(
                            f"--- Record {rid} | Attribute '{attr}' ---\n"
                            f"Eval function: {evf}\n"
                            f"Predicted: {pred!r}\n"
                            f"Expected:  {gold!r}\n\n"
                        )
            except Exception:
                pass

        total_counts = {"VC": 0, "VW": 0, "VN": 0, "NV": 0, "NN": 0}
        attr_results: Dict[str, Any] = {}
        per_attr_metrics: List[Dict[str, float]] = []

        # Evaluate per attribute
        for attr in attributes:
            if attr not in pred_aligned.columns or attr not in gold_aligned.columns:
                continue

            eval_fn = self.get_evaluation_function(attr)
            counts = {"VC": 0, "VW": 0, "VN": 0, "NV": 0, "NN": 0}

            for i in range(len(pred_aligned)):
                pred_val = pred_aligned.iloc[i][attr]
                gold_val = gold_aligned.iloc[i][attr]

                pred_missing = _is_missing_value(pred_val)
                gold_missing = _is_missing_value(gold_val)

                if pred_missing and gold_missing:
                    counts["NN"] += 1
                elif pred_missing and not gold_missing:
                    counts["VN"] += 1  # expected present, prediction missing -> FN
                    emit_mismatch({
                        "record_id": pred_aligned.iloc[i][pred_id_column],
                        "attribute": attr,
                        "evaluation_function": getattr(eval_fn, "__name__", str(eval_fn)),
                        "predicted_value": None,
                        "expected_value": gold_val,
                    })
                elif not pred_missing and gold_missing:
                    counts["NV"] += 1  # prediction present, expected missing -> FP
                    emit_mismatch({
                        "record_id": pred_aligned.iloc[i][pred_id_column],
                        "attribute": attr,
                        "evaluation_function": getattr(eval_fn, "__name__", str(eval_fn)),
                        "predicted_value": pred_val,
                        "expected_value": None,
                    })
                else:
                    # both present; use evaluation fn
                    try:
                        match_value = eval_fn(pred_val, gold_val)
                    except Exception:
                        match_value = str(pred_val).strip().lower() == str(gold_val).strip().lower()

                    # Coerce result to a plain bool robustly
                    match_bool: bool
                    try:
                        if isinstance(match_value, bool):
                            match_bool = match_value
                        else:
                            try:
                                import numpy as np  # type: ignore
                                if isinstance(match_value, np.bool_):
                                    match_bool = bool(match_value)
                                elif hasattr(match_value, "item"):
                                    match_bool = bool(match_value.item())  # pandas/NumPy scalar
                                else:
                                    match_bool = bool(match_value)
                            except Exception:
                                # Fallback to strict string comparison
                                match_bool = (str(pred_val).strip().lower() == str(gold_val).strip().lower())
                    except Exception:
                        match_bool = False

                    if match_bool:
                        counts["VC"] += 1
                    else:
                        counts["VW"] += 1
                        emit_mismatch({
                            "record_id": pred_aligned.iloc[i][pred_id_column],
                            "attribute": attr,
                            "evaluation_function": getattr(eval_fn, "__name__", str(eval_fn)),
                            "predicted_value": pred_val,
                            "expected_value": gold_val,
                        })

            # Aggregate to totals
            for k, v in counts.items():
                total_counts[k] += v

            # Metrics for this attribute
            metrics = _micro_metrics_from_counts(counts)
            attr_results[attr] = {"counts": counts, "metrics": metrics, "rule": getattr(eval_fn, "__name__", str(eval_fn))}
            per_attr_metrics.append(metrics)

        # Micro metrics across all attributes
        micro = _micro_metrics_from_counts(total_counts)

        # Macro metrics: mean of metrics across attributes (if any)
        if per_attr_metrics:
            macro = {
                k: sum(m[k] for m in per_attr_metrics) / len(per_attr_metrics)
                for k in per_attr_metrics[0].keys()
            }
        else:
            macro = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0, "accuracy_overall": 0.0}

        results = {
            "micro": micro,
            "macro": macro,
            "attributes": attr_results,
            "num_evaluated_records": len(pred_aligned),
            "total_counts": total_counts,
        }

        self._logger.info(
            f"IE evaluation done on {len(pred_aligned)} records, micro F1={micro['f1']:.3f}, accuracy={micro['accuracy']:.3f}"
        )
        return results
