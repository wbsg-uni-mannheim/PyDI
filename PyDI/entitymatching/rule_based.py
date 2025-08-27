"""
Rule-based entity matching using weighted linear combination of comparators.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Iterable, List, Optional, Union

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
        **kwargs,
    ) -> CorrespondenceSet:
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
        **kwargs
            Additional arguments (ignored).
            
        Returns
        -------
        CorrespondenceSet
            DataFrame with columns id1, id2, score, notes.
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
        total_pairs_processed = 0
        
        # Process candidate batches
        for batch in candidates:
            if batch.empty:
                continue
                
            batch_results = self._process_batch(
                batch, left_lookup, right_lookup, parsed_comparators, threshold
            )
            results.extend(batch_results)
            total_pairs_processed += len(batch)
        
        logging.info(f"Processed {total_pairs_processed} candidate pairs, "
                    f"found {len(results)} matches above threshold {threshold}")
        
        # Create correspondence set
        if results:
            return pd.DataFrame(results)
        else:
            return pd.DataFrame(columns=["id1", "id2", "score", "notes"])
    
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
    ) -> List[Dict[str, Union[str, float]]]:
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
            
        Returns
        -------
        List[Dict]
            List of correspondence dictionaries.
        """
        results = []
        
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
            similarity = self._compute_similarity(record1, record2, comparators)
            
            # Add to results if above threshold
            if similarity >= threshold:
                results.append({
                    "id1": id1,
                    "id2": id2,
                    "score": similarity,
                    "notes": f"comparators={len(comparators)}",
                })
        
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
    
    def __repr__(self) -> str:
        return f"RuleBasedMatcher()"