"""
Rule-based entity matching using weighted linear combination of comparators.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd

from .base import BaseMatcher, BaseComparator, CorrespondenceSet, ensure_record_ids


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
    ...     weights=[0.5, 0.5],
    ...     threshold=0.7
    ... )
    """
    
    def match(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        candidates: Iterable[pd.DataFrame],
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
            Left dataset with _id column.
        df_right : pandas.DataFrame
            Right dataset with _id column.
        candidates : Iterable[pandas.DataFrame]
            Candidate pair batches with id1, id2 columns.
        comparators : List[callable] or List[dict]
            List of comparator functions/objects, or list of dicts with
            'comparator' and 'weight' keys. Each comparator should accept
            (record1: pd.Series, record2: pd.Series) and return float.
        weights : List[float], optional
            Weights for each comparator. If None and comparators are not dicts,
            equal weights are used. Ignored if comparators contain weights.
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
        # Validate inputs
        self._validate_inputs(df_left, df_right)
        
        if not comparators:
            raise ValueError("No comparators provided")
        
        # Parse comparators and weights
        parsed_comparators = self._parse_comparators(comparators, weights)
        
        # Ensure record IDs
        df_left = ensure_record_ids(df_left)
        df_right = ensure_record_ids(df_right)
        
        # Log matching info
        self._log_matching_info(df_left, df_right, candidates)
        
        # Create lookup dictionaries for fast record access
        left_lookup = df_left.set_index("_id")
        right_lookup = df_right.set_index("_id")
        
        results = []
        debug_results = [] if debug else None
        total_pairs_processed = 0
        
        # Process candidate batches
        for batch in candidates:
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
            total_pairs_processed += len(batch)
        
        logging.info(f"Processed {total_pairs_processed} candidate pairs, "
                    f"found {len(results)} matches above threshold {threshold}")
        
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
        """
        parsed = []
        
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
                    # Equal weights
                    weight = 1.0 / len(comparators)
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
            
            # Get comparator name
            if isinstance(comparator, BaseComparator):
                comparator_name = comparator.name
            elif hasattr(comparator, '__name__'):
                comparator_name = comparator.__name__
            else:
                comparator_name = str(comparator)
            
            # For debug, we need to capture the values being compared
            # This is a simplified approach - in reality, comparators might use
            # different attributes, but for now we'll capture what we can
            
            # Try to extract relevant values (this will depend on how comparators work)
            # For now, we'll use a generic approach
            record1_value = str(record1.get(comparator_name, ""))
            record2_value = str(record2.get(comparator_name, ""))
            
            # For preprocessing, we'll assume minimal preprocessing for now
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
    
    def __repr__(self) -> str:
        return f"RuleBasedMatcher()"