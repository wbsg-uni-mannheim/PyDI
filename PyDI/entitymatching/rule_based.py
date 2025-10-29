"""
Rule-based entity matching using weighted linear combination of comparators.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd

from .base import BaseMatcher, BaseComparator, CorrespondenceSet


class RuleBasedMatcher(BaseMatcher):
    """Rule-based matcher using weighted linear combination of attribute similarities.
    
    This matcher implements the same functionality as Winter's LinearCombinationMatchingRule
    but with PyDI's simpler DataFrame-first design. It computes a weighted sum of 
    similarity scores from multiple comparators to determine entity matches.
    
    The comparators and weights are passed directly to the match() method following
    the design principle of keeping the matcher stateless.
    
    Example
    -------
    >>> from PyDI.entitymatching import RuleBasedMatcher
    >>> from PyDI.entitymatching.comparators import jaccard, date_within_years
    >>>
    >>> matcher = RuleBasedMatcher()
    >>> matches = matcher.match(
    ...     df_left, df_right, candidates,
    ...     comparators=[jaccard("title"), date_within_years("date", 2)],
    ...     weights=[2, 1],  # relative importance (normalized to [0.67, 0.33])
    ...     threshold=0.7
    ... )
    """
    
    def match(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        candidates: Union[pd.DataFrame, Iterable[pd.DataFrame]],
        id_column: str,
        comparators: List[Union[BaseComparator, Callable, Dict[str, Union[Callable, float]]]],
        weights: Optional[List[float]] = None,
        threshold: float = 0.0,
        debug: bool = False,
        **kwargs,
    ) -> Union[CorrespondenceSet, Tuple[CorrespondenceSet, pd.DataFrame]]:
        """Find entity correspondences using rule-based matching.

        Parameters
        ----------
        df_left : pandas.DataFrame
            Left dataset with specified ID column.
        df_right : pandas.DataFrame
            Right dataset with specified ID column.
        candidates : pandas.DataFrame or Iterable[pandas.DataFrame]
            Single DataFrame or iterable of candidate pair batches with id1, id2 columns.
        id_column : str
            Name of the column containing record identifiers.
        comparators : List[callable] or List[dict]
            List of comparator functions/objects, or list of dicts with
            'comparator' and 'weight' keys. Each comparator should accept
            (record1: pd.Series, record2: pd.Series) and return float.
        weights : List[float], optional
            Relative weights for each comparator. Will be normalized to sum to 1.0.
            If None, equal weights are used. Example: [1, 3, 2] becomes [0.166, 0.5, 0.333].
            Ignored if comparators contain weights.
        threshold : float, optional
            Minimum similarity score to include. Default is 0.0.
        debug : bool, optional
            If True, captures detailed comparator results for debugging.
            Returns tuple of (correspondences, debug_results). Default is False.
        **kwargs
            Additional arguments (ignored).
            
        Returns
        -------
        CorrespondenceSet or Tuple[CorrespondenceSet, pandas.DataFrame]
            If debug=False: DataFrame with columns id1, id2, score, notes.
            If debug=True: Tuple of (correspondences, debug_results) where
            debug_results contains detailed comparator information.
        """
        # Setup logger
        logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        
        # Log start of entity matching
        logger.info("Starting Entity Matching")
        start_time = time.time()
        
        # Validate inputs
        self._validate_inputs(df_left, df_right, id_column)
        
        if not comparators:
            raise ValueError("No comparators provided")
        
        # Parse comparators and weights
        parsed_comparators = self._parse_comparators(comparators, weights)
        
        # Validate that the specified ID column exists
        if id_column not in df_left.columns:
            raise ValueError(f"Left dataset missing required ID column: '{id_column}'")
        if id_column not in df_right.columns:
            raise ValueError(f"Right dataset missing required ID column: '{id_column}'")
        
        # Normalize candidates to iterable of DataFrames
        if isinstance(candidates, pd.DataFrame):
            candidate_batches = [candidates]
        else:
            candidate_batches = candidates
        
        # Log blocking info (similar to Winter's blocking log)
        logger.info(f"Blocking {len(df_left)} x {len(df_right)} elements")
        
        # Create lookup dictionaries for fast record access
        left_lookup = df_left.set_index(id_column)
        right_lookup = df_right.set_index(id_column)
        
        # Count total candidate pairs for reduction ratio calculation
        total_pairs_processed = sum(len(batch) for batch in candidate_batches if not batch.empty)
        total_possible_pairs = len(df_left) * len(df_right)
        reduction_ratio = 1 - (total_pairs_processed / total_possible_pairs) if total_possible_pairs > 0 else 0
        
        # Calculate elapsed time for blocking phase
        blocking_time = time.time() - start_time
        blocking_time_str = f"{blocking_time:.3f}"
        
        # Log matching phase info with reduction ratio
        logger.info(f"Matching {len(df_left)} x {len(df_right)} elements after 0:00:{blocking_time_str}; "
                   f"{total_pairs_processed} blocked pairs (reduction ratio: {reduction_ratio})")
        
        results = []
        debug_results = [] if debug else None
        
        # Process candidate batches
        for batch in candidate_batches:
            if batch.empty:
                continue
                
            if debug:
                batch_results, batch_debug = self._process_batch(
                    batch, left_lookup, right_lookup, parsed_comparators, threshold, debug=True
                )
                debug_results.extend(batch_debug)
            else:
                batch_results = self._process_batch(
                    batch, left_lookup, right_lookup, parsed_comparators, threshold, debug=False
                )
            
            results.extend(batch_results)
        
        # Calculate total elapsed time
        total_time = time.time() - start_time
        total_time_str = f"{total_time:.3f}"
        
        # Log completion info
        logger.info(f"Entity Matching finished after 0:00:{total_time_str}; found {len(results)} correspondences.")
        
        # Create correspondence set
        correspondences = pd.DataFrame(results) if results else pd.DataFrame(columns=["id1", "id2", "score", "notes"])
        
        if debug:
            debug_df = pd.DataFrame(debug_results) if debug_results else pd.DataFrame(columns=[
                "id1", "id2", "comparator_name", "record1_value", "record2_value",
                "record1_preprocessed", "record2_preprocessed", "similarity", "postprocessed_similarity"
            ])
            return correspondences, debug_df
        else:
            return correspondences
    
    def _parse_comparators(
        self,
        comparators: List[Union[BaseComparator, Callable, Dict[str, Union[Callable, float]]]],
        weights: Optional[List[float]],
    ) -> List[Dict[str, Union[Callable, float]]]:
        """Parse comparators and weights into normalized format.

        Parameters
        ----------
        comparators : List
            List of comparators or dicts with comparator/weight.
        weights : List[float], optional
            Weights for comparators.

        Returns
        -------
        List[Dict]
            List of dicts with 'comparator' and 'weight' keys.
            Weights are normalized to sum to 1.0.
        """
        parsed = []

        # First pass: collect comparators and raw weights
        for i, comp in enumerate(comparators):
            if isinstance(comp, dict):
                # Dict format: {"comparator": func, "weight": 0.5}
                if "comparator" not in comp or "weight" not in comp:
                    raise ValueError(f"Comparator dict at index {i} must have 'comparator' and 'weight' keys")

                if comp["weight"] <= 0.0:
                    raise ValueError(f"Weight at index {i} must be > 0.0")

                parsed.append({
                    "comparator": comp["comparator"],
                    "weight": comp["weight"],
                })
            else:
                # Function/object format - need weights
                if weights is None:
                    # Use 1.0 as default (will be normalized)
                    weight = 1.0
                else:
                    if i >= len(weights):
                        raise ValueError(f"Not enough weights provided for {len(comparators)} comparators")
                    weight = weights[i]
                    if weight <= 0.0:
                        raise ValueError(f"Weight at index {i} must be > 0.0")

                parsed.append({
                    "comparator": comp,
                    "weight": weight,
                })

        # Second pass: normalize weights to sum to 1.0
        total_weight = sum(p["weight"] for p in parsed)
        if total_weight == 0:
            raise ValueError("Total weight must be greater than 0")

        for p in parsed:
            p["weight"] /= total_weight

        return parsed
    
    def _process_batch(
        self,
        batch: pd.DataFrame,
        left_lookup: pd.DataFrame,
        right_lookup: pd.DataFrame,
        comparators: List[Dict[str, Union[Callable, float]]],
        threshold: float,
        debug: bool = False,
    ) -> Union[List[Dict[str, Union[str, float]]], Tuple[List[Dict[str, Union[str, float]]], List[Dict]]]:
        """Process a batch of candidate pairs.
        
        Parameters
        ----------
        batch : pandas.DataFrame
            Candidate pairs with id1, id2 columns.
        left_lookup : pandas.DataFrame
            Left dataset indexed by _id.
        right_lookup : pandas.DataFrame
            Right dataset indexed by _id.
        comparators : List[Dict]
            Parsed comparators with weights.
        threshold : float
            Minimum similarity threshold.
        debug : bool, optional
            If True, captures detailed comparator results.
            
        Returns
        -------
        List[Dict] or Tuple[List[Dict], List[Dict]]
            If debug=False: List of correspondence dictionaries.
            If debug=True: Tuple of (correspondence_list, debug_results_list).
        """
        results = []
        debug_results = [] if debug else None
        
        for _, row in batch.iterrows():
            id1, id2 = row["id1"], row["id2"]
            
            # Get records
            try:
                record1 = left_lookup.loc[id1]
                record2 = right_lookup.loc[id2]
                
                # Handle case where .loc returns DataFrame due to duplicate indices
                if isinstance(record1, pd.DataFrame):
                    record1 = record1.iloc[0]
                if isinstance(record2, pd.DataFrame):
                    record2 = record2.iloc[0]
                    
            except KeyError:
                logging.warning(f"Record not found: {id1} or {id2}")
                continue
            
            # Compute similarity
            if debug:
                similarity, pair_debug_results = self._compute_similarity_with_debug(
                    record1, record2, comparators, id1, id2
                )
                debug_results.extend(pair_debug_results)
            else:
                similarity = self._compute_similarity(record1, record2, comparators)
            
            # Add to results if above threshold
            if similarity >= threshold:
                results.append({
                    "id1": id1,
                    "id2": id2,
                    "score": similarity,
                    "notes": f"comparators={len(comparators)}",
                })
        
        if debug:
            return results, debug_results
        else:
            return results
    
    def _compute_similarity(
        self, 
        record1: pd.Series, 
        record2: pd.Series,
        comparators: List[Dict[str, Union[Callable, float]]],
    ) -> float:
        """Compute weighted similarity between two records.
        
        Parameters
        ----------
        record1 : pandas.Series
            First record.
        record2 : pandas.Series
            Second record.
        comparators : List[Dict]
            Parsed comparators with weights.
            
        Returns
        -------
        float
            Weighted similarity score.
        """
        weighted_sum = 0.0
        
        for comp_info in comparators:
            comparator = comp_info["comparator"]
            weight = comp_info["weight"]
            
            # Compute similarity using comparator
            if isinstance(comparator, BaseComparator):
                similarity = comparator.compare(record1, record2)
            else:
                # Assume it's a callable
                similarity = comparator(record1, record2)
            
            weighted_sum += similarity * weight
        
        return weighted_sum
    
    def _compute_similarity_with_debug(
        self,
        record1: pd.Series,
        record2: pd.Series,
        comparators: List[Dict[str, Union[Callable, float]]],
        id1: str,
        id2: str,
    ) -> Tuple[float, List[Dict]]:
        """Compute weighted similarity with detailed debug information.
        
        Parameters
        ----------
        record1 : pandas.Series
            First record.
        record2 : pandas.Series
            Second record.
        comparators : List[Dict]
            Parsed comparators with weights.
        id1, id2 : str
            Record identifiers for debug output.
            
        Returns
        -------
        Tuple[float, List[Dict]]
            Tuple of (weighted_similarity, debug_results_list).
        """
        weighted_sum = 0.0
        debug_results = []
        
        for comp_info in comparators:
            comparator = comp_info["comparator"]
            weight = comp_info["weight"]
            
            # Get comparator name - only show list_strategy if lists were actually processed
            if isinstance(comparator, BaseComparator):
                # Create a cleaner name by conditionally including list_strategy
                if hasattr(comparator, 'column'):
                    class_name = comparator.__class__.__name__
                    column = comparator.column
                    
                    # Check if this comparator actually processed lists
                    list_was_used = False
                    if hasattr(comparator, 'list_strategy'):
                        try:
                            # Check if either record has a list for this column
                            val1 = record1.get(column)
                            val2 = record2.get(column)
                            list_was_used = self._is_actual_list(val1) or self._is_actual_list(val2)
                        except:
                            list_was_used = False
                    
                    # Build the name based on comparator type and whether lists were used
                    if hasattr(comparator, 'similarity_function'):
                        # StringComparator
                        if list_was_used:
                            comparator_name = f"{class_name}({column}, {comparator.similarity_function}, list_strategy={comparator.list_strategy})"
                        else:
                            comparator_name = f"{class_name}({column}, {comparator.similarity_function})"
                    elif hasattr(comparator, 'method'):
                        # NumericComparator
                        if list_was_used:
                            comparator_name = f"{class_name}({column}, {comparator.method}, list_strategy={comparator.list_strategy})"
                        else:
                            comparator_name = f"{class_name}({column}, {comparator.method})"
                    else:
                        # Other comparators like DateComparator
                        if list_was_used:
                            comparator_name = f"{class_name}({column}, list_strategy={comparator.list_strategy})"
                        else:
                            comparator_name = f"{class_name}({column})"
                else:
                    # Fallback to original name
                    comparator_name = comparator.name
            elif hasattr(comparator, '__name__'):
                comparator_name = comparator.__name__
            else:
                comparator_name = str(comparator)
            
            # Extract the values being compared by the comparator
            record1_value = ""
            record2_value = ""
            record1_preprocessed = ""
            record2_preprocessed = ""
            
            # Try to extract the actual column values that the comparator uses
            if isinstance(comparator, BaseComparator) and hasattr(comparator, 'column'):
                # Most PyDI comparators have a 'column' attribute
                column_name = comparator.column
                try:
                    raw_val1 = record1.get(column_name, "")
                    raw_val2 = record2.get(column_name, "")
                    record1_value = str(raw_val1) if raw_val1 is not None and not self._is_null_value(raw_val1) else ""
                    record2_value = str(raw_val2) if raw_val2 is not None and not self._is_null_value(raw_val2) else ""
                    
                    # Apply preprocessing if the comparator has it
                    if hasattr(comparator, 'preprocess') and comparator.preprocess is not None:
                        try:
                            record1_preprocessed = str(comparator.preprocess(raw_val1)) if raw_val1 is not None and not self._is_null_value(raw_val1) else ""
                            record2_preprocessed = str(comparator.preprocess(raw_val2)) if raw_val2 is not None and not self._is_null_value(raw_val2) else ""
                        except:
                            record1_preprocessed = record1_value
                            record2_preprocessed = record2_value
                    else:
                        record1_preprocessed = record1_value
                        record2_preprocessed = record2_value
                        
                except (KeyError, AttributeError):
                    # Column not found, leave values empty
                    pass
            elif hasattr(comparator, '__name__'):
                # For simple callable comparators, try to infer from function name
                # This is a fallback for custom comparators
                record1_value = str(record1)[:100] + "..." if len(str(record1)) > 100 else str(record1)
                record2_value = str(record2)[:100] + "..." if len(str(record2)) > 100 else str(record2)
                record1_preprocessed = record1_value
                record2_preprocessed = record2_value
            
            # Compute similarity using comparator
            if isinstance(comparator, BaseComparator):
                similarity = comparator.compare(record1, record2)
            else:
                # Assume it's a callable
                similarity = comparator(record1, record2)
            
            # For now, postprocessed similarity is the same as raw similarity
            postprocessed_similarity = similarity
            
            # Store debug information
            debug_results.append({
                "id1": id1,
                "id2": id2,
                "comparator_name": comparator_name,
                "record1_value": record1_value,
                "record2_value": record2_value,
                "record1_preprocessed": record1_preprocessed,
                "record2_preprocessed": record2_preprocessed,
                "similarity": similarity,
                "postprocessed_similarity": postprocessed_similarity,
            })
            
            weighted_sum += similarity * weight
        
        return weighted_sum, debug_results
    
    def _is_null_value(self, val):
        """Check if value is null, handling both scalars and arrays."""
        if val is None:
            return True
        try:
            # For arrays/series, check if all values are null
            if hasattr(val, '__iter__') and not isinstance(val, str):
                return pd.isna(val).all() if hasattr(pd.isna(val), 'all') else pd.isna(val)
            # For scalars
            return pd.isna(val)
        except (TypeError, ValueError):
            return False
    
    def _is_actual_list(self, val):
        """Check if value is actually a list/array (not a scalar or string)."""
        if val is None or pd.isna(val):
            return False
        
        # Check if it's iterable but not a string
        if not hasattr(val, '__iter__') or isinstance(val, str):
            return False
            
        try:
            # For pandas Series/arrays, check if they have multiple elements
            if hasattr(val, '__len__'):
                return len(val) > 1
            # For other iterables, try to count elements (avoid consuming iterators)
            return hasattr(val, '__getitem__')  # Has indexing, likely a list-like structure
        except (TypeError, ValueError):
            return False
    
    def __repr__(self) -> str:
        return f"RuleBasedMatcher()"