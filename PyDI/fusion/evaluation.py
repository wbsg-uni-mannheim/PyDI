"""
Evaluation framework for data fusion in PyDI.

This module provides tools for evaluating the quality of fusion results
against validation/test datasets.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple, Union, Iterable
from collections import Counter
from datetime import datetime, date
import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path

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


def exact_match(fused_value, expected_value) -> bool:
    """Default evaluation function using exact equality."""
    return fused_value == expected_value


def tokenized_match(fused_value, expected_value, threshold: float = 1.0) -> bool:
    """Evaluation function using tokenized comparison with similarity threshold.

    For lists: Uses Jaccard similarity between lists (order doesn't matter).
    For strings: Tokenizes and uses Jaccard similarity between token sets.
    Useful for actor lists and titles where order and partial matches matter.

    Parameters
    ----------
    fused_value : Any
        The fused value to compare.
    expected_value : Any
        The expected value to compare against.
    threshold : float, default 1.0
        Minimum similarity threshold (0.0 to 1.0). 
        1.0 requires exact match, lower values allow partial matches.

    Returns
    -------
    bool
        True if similarity >= threshold, False otherwise.
    """
    # Check for missing values using the same logic as DataFusionEvaluator
    if _is_missing_value(fused_value) and _is_missing_value(expected_value):
        return True
    if _is_missing_value(fused_value) or _is_missing_value(expected_value):
        return False

    # Get Jaccard similarity function from registry
    jaccard_sim = SimilarityRegistry.get_function('jaccard')

    # Handle lists of strings - use Jaccard similarity
    if isinstance(fused_value, list) and isinstance(expected_value, list):
        # Use Jaccard similarity between sets (order doesn't matter)
        similarity = jaccard_sim(set(fused_value), set(expected_value))
        return similarity >= threshold

    # Handle mixed list/string by converting both to lists
    if isinstance(fused_value, list) or isinstance(expected_value, list):
        # Convert both to lists, then use Jaccard similarity
        fused_list = fused_value if isinstance(
            fused_value, list) else [str(fused_value)]
        expected_list = expected_value if isinstance(
            expected_value, list) else [str(expected_value)]
        similarity = jaccard_sim(set(fused_list), set(expected_list))
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
    expected_tokens = clean_tokens(expected_value)

    # Use Jaccard similarity between token sets
    if len(fused_tokens) == 0 and len(expected_tokens) == 0:
        return True  # Both empty
    if len(fused_tokens) == 0 or len(expected_tokens) == 0:
        return False  # One empty, one not

    similarity = jaccard_sim(fused_tokens, expected_tokens)
    return similarity >= threshold


def year_only_match(fused_value, expected_value) -> bool:
    """Evaluation function comparing only the year part of dates.

    If the inputs are strings, attempt to parse to dates; if they are
    ``datetime``/``date``/timestamp-like, compare their ``year`` values.
    On unhandled types or failed parsing, log an error and return False.
    """
    if _is_missing_value(fused_value) and _is_missing_value(expected_value):
        return True
    if _is_missing_value(fused_value) or _is_missing_value(expected_value):
        return False

    logger = logging.getLogger(__name__)

    def _to_date(value: Any) -> Optional[date]:
        # Native datetime/date
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value

        # pandas / numpy timestamp-like
        try:
            if isinstance(value, pd.Timestamp):
                return value.date()
        except Exception:
            pass

        # Strings and other coercible types via pandas
        try:
            parsed = pd.to_datetime(value, errors="coerce")
            if pd.isna(parsed):
                return None
            return parsed.date()
        except Exception:
            return None

    d1 = _to_date(fused_value)
    d2 = _to_date(expected_value)

    if d1 is None or d2 is None:
        logger.error(
            "year_only_match: could not convert values to date (fused=%r, expected=%r)",
            fused_value,
            expected_value,
        )
        return False

    return d1.year == d2.year


def numeric_tolerance_match(fused_value, expected_value, tolerance: float = 0.01) -> bool:
    """Evaluation function for numeric values with tolerance."""
    if _is_missing_value(fused_value) and _is_missing_value(expected_value):
        return True
    if _is_missing_value(fused_value) or _is_missing_value(expected_value):
        return False

    try:
        return abs(float(fused_value) - float(expected_value)) <= tolerance
    except (ValueError, TypeError):
        return str(fused_value).strip() == str(expected_value).strip()


def set_equality_match(fused_value, expected_value) -> bool:
    """Evaluation function for set equality (order-independent).

    Useful for lists where order doesn't matter.
    """
    if _is_missing_value(fused_value) and _is_missing_value(expected_value):
        return True
    if _is_missing_value(fused_value) or _is_missing_value(expected_value):
        return False

    try:
        if isinstance(fused_value, (list, tuple, set)) and isinstance(expected_value, (list, tuple, set)):
            return set(fused_value) == set(expected_value)
        return fused_value == expected_value
    except (TypeError, ValueError):
        return str(fused_value) == str(expected_value)


def boolean_match(fused_value, expected_value) -> bool:
    """Evaluation function for boolean values with flexible interpretation.

    Handles various boolean representations:
    - True/False, true/false, yes/no, 1/0, "true"/"false", etc.
    """
    if _is_missing_value(fused_value) and _is_missing_value(expected_value):
        return True
    if _is_missing_value(fused_value) or _is_missing_value(expected_value):
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
    expected_bool = normalize_boolean(expected_value)

    # If either couldn't be normalized, fall back to string comparison
    if fused_bool is None or expected_bool is None:
        return str(fused_value).strip().lower() == str(expected_value).strip().lower()

    return fused_bool == expected_bool


class DataFusionEvaluator:
    """Evaluator for data fusion results against validation/test datasets.

    Parameters
    ----------
    strategy : DataFusionStrategy
        The fusion strategy containing evaluation rules.
    """

    def __init__(
        self,
        strategy: DataFusionStrategy,
        *,
        debug: bool = False,
        debug_file: Optional[Union[str, Path]] = None,
        debug_format: str = "json",
        fusion_debug_logs: Optional[Union[Path, str, Iterable[Union[Path, str]]]] = None,
    ):
        self.strategy = strategy
        self._logger = logging.getLogger(__name__)
        self._debug_enabled = bool(debug)
        self._debug_format = debug_format if debug_format in {"text", "json"} else "json"
        self._debug_file: Optional[Path] = None
        self._fusion_debug_map: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._current_mismatch_counter: Optional[Counter[str]] = None
        if self._debug_enabled:
            path = Path(debug_file) if debug_file is not None else Path(
                "fusion_evaluation_debug.jsonl" if self._debug_format == "json" else "fusion_evaluation_debug.log"
            )
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("w", encoding="utf-8") as f:
                    header = {
                        "type": "evaluation_debug_header",
                        "strategy": self.strategy.name,
                        "format": self._debug_format,
                    }
                    if self._debug_format == "json":
                        f.write(json.dumps(header, ensure_ascii=False) + "\n")
                    else:
                        f.write(
                            "=== Fusion Evaluation Debug Log ===\n"
                            f"Strategy: {self.strategy.name}\n"
                            "Only mismatches and evaluation errors are recorded.\n\n"
                        )
                self._debug_file = path
                self._logger.info(
                    "Fusion evaluation debug logging enabled; refer to %s for mismatch details.",
                    path,
                )
            except Exception as exc:
                self._logger.warning(
                    "Could not initialize evaluation debug log '%s': %s",
                    path,
                    exc,
                )
                self._debug_file = None

        if fusion_debug_logs is not None:
            self.set_fusion_debug_logs(fusion_debug_logs)

    def set_fusion_debug_logs(
        self,
        logs: Optional[Union[Path, str, Iterable[Union[Path, str]]]],
    ) -> None:
        """Load fusion debug log files to recover conflict inputs for evaluation."""
        self._fusion_debug_map = self._load_fusion_debug_inputs(logs) if logs else {}

    def evaluate(
        self,
        fused_df: pd.DataFrame,
        fused_id_column: str,
        expected_df: Optional[pd.DataFrame] = None,
        expected_id_column: Optional[str] = None,
        gold_df: Optional[pd.DataFrame] = None,
        gold_id_column: Optional[str] = None,
    ) -> Dict[str, float]:
        """Evaluate fused results against an expected/validation dataset.

        Parameters
        ----------
        fused_df : pd.DataFrame
            The fused dataset to evaluate.
        fused_id_column : str
            ID column name in the fused dataset.
        expected_df : pd.DataFrame, optional
            The expected/validation/test dataset.
        expected_id_column : str, optional
            ID column name in the expected dataset.
        gold_df : pd.DataFrame, optional
            Deprecated alias for expected_df.
        gold_id_column : str, optional
            Deprecated alias for expected_id_column.

        Returns
        -------
        Dict[str, float]
            Dictionary of evaluation metrics.
        """
        self._logger.info("Starting fusion evaluation")

        mismatch_counter: Optional[Counter[str]] = Counter() if self._debug_enabled else None

        previous_mismatch_counter = self._current_mismatch_counter
        self._current_mismatch_counter = mismatch_counter

        attribute_results: Dict[str, Dict[str, Any]] = {}

        try:
            # Backward-compat: support old gold_* parameter names
            if expected_df is None and gold_df is not None:
                expected_df = gold_df
            if expected_id_column is None and gold_id_column is not None:
                expected_id_column = gold_id_column

            # Align datasets by their respective ID columns
            aligned_fused, aligned_expected = self._align_datasets_two_ids(
                fused_df, fused_id_column, expected_df, expected_id_column
            )

            if aligned_fused.empty or aligned_expected.empty:
                self._logger.warning(
                    "No matching records found between fused and expected datasets")
                return {"overall_accuracy": 0.0, "num_evaluated_records": 0}

            # Get attributes to evaluate
            attributes = self._get_evaluable_attributes(
                aligned_fused, aligned_expected, fused_id_column, expected_id_column)

            if not attributes:
                self._logger.warning("No common attributes found for evaluation")
                return {"overall_accuracy": 0.0, "num_evaluated_records": len(aligned_fused)}

            # Evaluate each attribute
            total_correct = 0
            total_evaluated = 0

            for attribute in attributes:
                results = self._evaluate_attribute(
                    aligned_fused,
                    aligned_expected,
                    attribute,
                    fused_id_column,
                    expected_id_column,
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
        finally:
            if mismatch_counter is not None:
                total_mismatches = sum(mismatch_counter.values())
                if total_mismatches:
                    sorted_items = sorted(
                        mismatch_counter.items(), key=lambda item: item[1], reverse=True
                    )
                    max_attr_len = max(len(attr) for attr, _ in sorted_items)
                    attr_col_width = max(32, min(max_attr_len, 52))

                    def _shorten(text: str) -> str:
                        if len(text) <= attr_col_width:
                            return text
                        return text[: attr_col_width - 1] + "…"

                    header = (
                        f"{'Attribute':<{attr_col_width}} | {'Errors':>7} | {'Percentage':>10}"
                    )
                    separator = "─" * len(header)

                    self._logger.info(
                        "Evaluation mismatches by attribute (debug): %d total",
                        total_mismatches,
                    )
                    self._logger.info("\t%s", header)
                    self._logger.info("\t%s", separator)

                    row_template = f"{{:<{attr_col_width}}} | {{:>7}} | {{:>9.2f}}%%"
                    for attr, count in sorted_items:
                        percentage = (count / total_mismatches) * 100
                        line = row_template.format(
                            _shorten(attr), count, percentage
                        )
                        self._logger.info("\t%s", line)
                else:
                    self._logger.info(
                        "Evaluation mismatches by attribute (debug): none recorded"
                    )
            self._current_mismatch_counter = previous_mismatch_counter

    def _align_datasets_two_ids(
        self,
        fused_df: pd.DataFrame,
        fused_id_column: str,
        expected_df: pd.DataFrame,
        expected_id_column: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Align two datasets by possibly different ID columns.

        Returns aligned DataFrames with matching records only.
        """
        fused_clean = fused_df.dropna(subset=[fused_id_column]).copy()
        expected_clean = expected_df.dropna(subset=[expected_id_column]).copy()

        if expected_clean.empty:
            return pd.DataFrame(), pd.DataFrame()

        fused_clean["__eval_id"] = fused_clean[fused_id_column].astype(str)
        expected_clean["__eval_id"] = expected_clean[expected_id_column].astype(str)

        expected_id_order = expected_clean["__eval_id"].tolist()
        fused_id_set = set(fused_clean["__eval_id"])

        missing_ids = [gid for gid in expected_id_order if gid not in fused_id_set]
        if missing_ids:
            preview = ", ".join(missing_ids[:5])
            if len(missing_ids) > 5:
                preview += ", ..."
            self._logger.warning(
                "Missing %d expected/reference records in fused dataset: %s",
                len(missing_ids),
                preview,
            )

        aligned_expected = expected_clean.set_index("__eval_id").loc[expected_id_order]
        aligned_fused = fused_clean.set_index("__eval_id").reindex(expected_id_order)

        aligned_expected = aligned_expected.reset_index(drop=True)
        aligned_fused = aligned_fused.reset_index(drop=True)

        aligned_expected = aligned_expected.drop(columns="__eval_id", errors="ignore")
        aligned_fused = aligned_fused.drop(columns="__eval_id", errors="ignore")

        return aligned_fused, aligned_expected

    def _get_evaluable_attributes(
        self,
        fused_df: pd.DataFrame,
        expected_df: pd.DataFrame,
        fused_id_column: str,
        expected_id_column: str,
    ) -> List[str]:
        """Get attributes that can be evaluated (present in both datasets)."""
        fused_attrs = set(fused_df.columns)
        expected_attrs = set(expected_df.columns)

        # Find common attributes, excluding metadata columns
        common_attrs = fused_attrs.intersection(expected_attrs)

        # Filter out metadata and ID columns
        evaluable_attrs = [
            attr for attr in common_attrs
            if not attr.startswith("_fusion_") and attr not in {fused_id_column, expected_id_column}
        ]

        return evaluable_attrs

    def _evaluate_attribute(
        self,
        fused_df: pd.DataFrame,
        expected_df: pd.DataFrame,
        attribute: str,
        fused_id_column: str,
        expected_id_column: str,
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
        if len(fused_df) != len(expected_df):
            self._logger.warning(
                "Aligned dataframes have different lengths (%d vs %d) for attribute '%s'",
                len(fused_df),
                len(expected_df),
                attribute,
            )

        num_rows = min(len(fused_df), len(expected_df))

        for i in range(num_rows):
            fused_row = fused_df.iloc[i]
            expected_row = expected_df.iloc[i]
            fused_value = fused_row.get(attribute)
            expected_value = expected_row.get(attribute)

            fused_missing = self._is_missing(fused_value)
            expected_missing = self._is_missing(expected_value)

            fused_id = fused_row.get(fused_id_column)
            expected_id = expected_row.get(expected_id_column)

            metadata = fused_row.get("_fusion_metadata")
            conflict_rule = None
            inputs = None
            if isinstance(metadata, dict):
                conflict_rule = metadata.get(f"{attribute}_rule")
                inputs = metadata.get(f"{attribute}_inputs")
            if conflict_rule is None:
                # Fallback to direct columns if metadata column was removed upstream
                conflict_rule = fused_row.get(f"{attribute}_rule")
            if inputs is None:
                raw_inputs = fused_row.get(f"{attribute}_inputs")
                if raw_inputs is not None:
                    inputs = raw_inputs

            debug_entry = None
            if (inputs is None or conflict_rule is None) and self._fusion_debug_map:
                debug_entry = self._find_debug_entry(
                    fused_row, fused_id_column, attribute
                )
                if debug_entry:
                    if inputs is None:
                        inputs = debug_entry.get("inputs")
                    if conflict_rule is None:
                        conflict_rule = debug_entry.get("conflict_rule")

            if conflict_rule is None:
                fuser = self.strategy.get_attribute_fuser(attribute)
                if fuser is not None:
                    conflict_rule = get_callable_name(fuser.resolver)
                else:
                    conflict_rule = "default"

            if inputs is None:
                fused_sources = fused_row.get("_fusion_sources")
                fused_source_datasets = fused_row.get("_fusion_source_datasets")

                if isinstance(fused_sources, (list, tuple, set)):
                    normalized_sources = list(fused_sources)
                    dataset_list = (
                        list(fused_source_datasets)
                        if isinstance(fused_source_datasets, (list, tuple, set))
                        else []
                    )

                    inputs = []
                    for idx, src in enumerate(normalized_sources):
                        record_id = src
                        dataset_name = None

                        if isinstance(src, dict):
                            record_id = src.get("record_id")
                            dataset_name = src.get("dataset")

                        if dataset_name is None and dataset_list:
                            dataset_name = dataset_list[min(idx, len(dataset_list) - 1)]

                        if dataset_name is None and not isinstance(src, dict):
                            dataset_name = str(src)

                        inputs.append(
                            {
                                "record_id": record_id,
                                "dataset": dataset_name,
                                "value": "<metadata unavailable>",
                            }
                        )

            if isinstance(inputs, list):
                input_records = inputs
            elif isinstance(inputs, tuple):
                input_records = list(inputs)
            else:
                input_records = []

            serialized_inputs = input_records if input_records else []

            # No expected value -> cannot evaluate this row
            if expected_missing and fused_missing:
                continue
            if expected_missing:
                continue


            total_count += 1

            if fused_missing:
                # Count as incorrect when expected value exists but fused is missing
                self._emit_mismatch(
                    attribute=attribute,
                    fused_id=fused_id,
                    expected_id=expected_id,
                    fused_value=None,
                    expected_value=expected_value,
                    evaluation_rule=eval_function,
                    conflict_rule=conflict_rule,
                    inputs=serialized_inputs,
                    reason="missing_fused_value",
                )
                continue

            # Evaluate using the function
            try:
                matched = eval_function(fused_value, expected_value)
            except Exception as exc:
                self._emit_mismatch(
                    attribute=attribute,
                    fused_id=fused_id,
                    expected_id=expected_id,
                    fused_value=fused_value,
                    expected_value=expected_value,
                    evaluation_rule=eval_function,
                    conflict_rule=conflict_rule,
                    inputs=serialized_inputs,
                    reason="evaluation_exception",
                    error=str(exc),
                )
                continue

            if matched:
                correct_count += 1
                continue

            self._emit_mismatch(
                attribute=attribute,
                fused_id=fused_id,
                expected_id=expected_id,
                fused_value=fused_value,
                expected_value=expected_value,
                evaluation_rule=eval_function,
                conflict_rule=conflict_rule,
                inputs=serialized_inputs,
                reason="mismatch",
            )

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

    def _load_fusion_debug_inputs(
        self,
        logs: Union[Path, str, Iterable[Union[Path, str]]],
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        mapping: Dict[str, Dict[str, Dict[str, Any]]] = {}

        if isinstance(logs, (str, Path)):
            paths = [Path(logs)]
        else:
            paths = [Path(p) for p in logs]

        for path in paths:
            try:
                with Path(path).open("r", encoding="utf-8") as handle:
                    for raw_line in handle:
                        line = raw_line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                        except json.JSONDecodeError:
                            # Skip non-JSON lines (e.g., headers in text logs)
                            continue
                        if not isinstance(entry, dict):
                            continue

                        group_id = entry.get("group_id")
                        attribute = entry.get("attribute")
                        inputs = entry.get("inputs")
                        if not group_id or not attribute:
                            continue

                        record = {
                            "inputs": inputs,
                            "conflict_rule": entry.get("conflict_resolution_function")
                            or entry.get("conflict_rule"),
                        }

                        keys = []
                        fused_id = entry.get("fused_id")
                        if fused_id is not None:
                            keys.append(str(fused_id))
                        # Store both raw group_id and fused_ variant
                        keys.append(str(group_id))
                        keys.append(f"fused_{group_id}")

                        for key in keys:
                            attr_map = mapping.setdefault(str(key), {})
                            attr_map[attribute] = record
            except FileNotFoundError:
                self._logger.warning(
                    "Fusion debug log '%s' not found; skipping.", path
                )
            except Exception as exc:
                self._logger.warning(
                    "Failed to read fusion debug log '%s': %s", path, exc
                )

        return mapping

    def _find_debug_entry(
        self,
        fused_row: pd.Series,
        fused_id_column: str,
        attribute: str,
    ) -> Optional[Dict[str, Any]]:
        if not self._fusion_debug_map:
            return None

        candidates: List[Any] = []

        fused_id = fused_row.get(fused_id_column)
        if fused_id is not None:
            candidates.extend([fused_id, str(fused_id)])

        group_id = fused_row.get("_fusion_group_id")
        if group_id is not None:
            candidates.extend([group_id, str(group_id), f"fused_{group_id}"])

        row_index = fused_row.name
        if row_index is not None:
            candidates.extend([row_index, str(row_index)])

        for key in candidates:
            if key is None:
                continue
            attr_map = self._fusion_debug_map.get(str(key))
            if not attr_map:
                continue
            entry = attr_map.get(attribute)
            if entry:
                return entry

        return None

    def _emit_mismatch(
        self,
        *,
        attribute: str,
        fused_id: Any,
        expected_id: Any,
        fused_value: Any,
        expected_value: Any,
        evaluation_rule,
        conflict_rule: Optional[str],
        inputs: Optional[Any],
        reason: str,
        error: Optional[str] = None,
    ) -> None:
        if self._current_mismatch_counter is not None and attribute:
            self._current_mismatch_counter[str(attribute)] += 1

        if not self._debug_enabled or self._debug_file is None:
            return

        entry = {
            "type": "evaluation_mismatch",
            "attribute": attribute,
            "fused_id": self._serialize_value(fused_id),
            "expected_id": self._serialize_value(expected_id),
            "fused_value": self._serialize_value(fused_value),
            "expected_value": self._serialize_value(expected_value),
            "evaluation_rule": get_callable_name(evaluation_rule),
            "conflict_rule": conflict_rule,
            "inputs": self._serialize_value(inputs),
            "reason": reason,
        }
        if error is not None:
            entry["error"] = error

        self._emit_debug(entry)

    def _emit_debug(self, entry: Dict[str, Any]) -> None:
        if not self._debug_enabled or self._debug_file is None:
            return

        try:
            with self._debug_file.open("a", encoding="utf-8") as f:
                if self._debug_format == "json":
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                else:
                    block = self._format_debug_block(entry)
                    f.write(block)
        except Exception as exc:
            self._logger.debug(
                "Failed to write evaluation debug entry: %s", exc
            )


    @staticmethod
    def _format_debug_block(entry: Dict[str, Any]) -> str:
        parts = [
            f"Attribute: {entry.get('attribute')}\n",
            f"  Fused ID: {entry.get('fused_id')}\n",
            f"  Expected ID: {entry.get('expected_id')}\n",
            f"  Conflict Rule: {entry.get('conflict_rule')}\n",
            f"  Evaluation Rule: {entry.get('evaluation_rule')}\n",
            f"  Reason: {entry.get('reason')}\n",
            f"  Fused Value: {entry.get('fused_value')}\n",
            f"  Expected Value: {entry.get('expected_value')}\n",
        ]
        if entry.get("inputs") is not None:
            parts.append(f"  Inputs: {entry.get('inputs')}\n")
        if "error" in entry:
            parts.append(f"  Error: {entry.get('error')}\n")
        parts.append("\n")
        return "".join(parts)

    @staticmethod
    def _serialize_value(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (list, tuple)):
            return [DataFusionEvaluator._serialize_value(v) for v in value]
        if isinstance(value, dict):
            return {
                str(k): DataFusionEvaluator._serialize_value(v)
                for k, v in value.items()
            }
        try:
            return value.item()  # type: ignore[attr-defined]
        except Exception:
            return repr(value)


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
