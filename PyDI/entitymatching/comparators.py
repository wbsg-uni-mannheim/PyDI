"""
Comparator classes for entity matching.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import pandas as pd

from .base import BaseComparator
from ..utils import SimilarityRegistry


class StringComparator(BaseComparator):
    """String similarity comparator using textdistance metrics.

    Parameters
    ----------
    column : str
        Column name to compare.
    similarity_function : str
        Name of textdistance similarity function.
    tokenization : str or callable, optional
        Tokenization strategy for token-based similarity functions. Options:
        - "word": Whitespace tokenization
        - "char": Character-level tokenization
        - "ngram_2", "ngram_3": N-gram tokenization
        - callable: Custom tokenizer function
        Default: "word" for token-based functions (jaccard, sorensen_dice, cosine, etc.),
        "char" for edit-based functions (jaro_winkler, levenshtein, etc.).
    preprocess : callable, optional
        Function to preprocess values before comparison.
    list_strategy : str, optional
        Strategy for handling lists ("concatenate", "best_match", "set_jaccard", "set_overlap").
        Default is "concatenate".
    """
    
    def __init__(
        self,
        column: str,
        similarity_function: str = "jaro_winkler",
        tokenization: Optional[str] = None,
        preprocess: Optional[Callable[[str], str]] = None,
        list_strategy: str = None,
    ):
        self.column = column
        self.similarity_function = similarity_function
        self.preprocess = preprocess
        self.list_strategy = list_strategy

        # Set intelligent default tokenization based on function type
        if tokenization is None:
            self.tokenization = "word" if SimilarityRegistry.is_tokenizable(similarity_function) else "char"
        else:
            self.tokenization = tokenization

        super().__init__(f"StringComparator({column}, {similarity_function}, tokenization={self.tokenization}, list_strategy={list_strategy})")

        if list_strategy is not None and list_strategy not in ["concatenate", "best_match", "set_jaccard", "set_overlap"]:
            raise ValueError(f"Unknown list_strategy: {list_strategy}")

        # Get similarity function with tokenization
        try:
            self._sim_func = SimilarityRegistry.get_function(similarity_function, self.tokenization)
        except ValueError as e:
            available = SimilarityRegistry.get_recommended_functions("entity_matching")
            available_tokenization = SimilarityRegistry.list_tokenization_strategies()
            raise ValueError(f"{e}. Recommended functions: {available}. Available tokenization: {available_tokenization}")
    
    def compare(self, record1: pd.Series, record2: pd.Series) -> float:
        """Compare string values in specified column, handling both single values and lists."""
        try:
            val1 = record1[self.column]
            val2 = record2[self.column]
            
            # Handle missing values - properly handle arrays/series
            if self._is_null_value(val1) or self._is_null_value(val2):
                return 0.0
            
            # Check if we have list values but no list_strategy
            if self.list_strategy is None and (self._is_list_value(val1) or self._is_list_value(val2)):
                raise ValueError(f"List values detected in column '{self.column}' but list_strategy is None. Please specify a list_strategy.")
            
            # Handle comparison based on whether we have lists or single values
            if self.list_strategy is None:
                # Both are single values - compare directly
                str1 = str(val1) if val1 is not None else ""
                str2 = str(val2) if val2 is not None else ""
                if self.preprocess:
                    str1 = self.preprocess(str1)
                    str2 = self.preprocess(str2)
                result = float(self._sim_func(str1, str2))
            else:
                # Apply list strategy to get comparison values
                result = self._apply_list_strategy(val1, val2)
            
            return result
            
        except KeyError:
            logging.warning(f"Column '{self.column}' not found in one or both records")
            return 0.0
        except Exception as e:
            logging.warning(f"Error in StringComparator: {e}")
            return 0.0
    
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
    
    def _is_list_value(self, val):
        """Check if value is a list/array (not a scalar string)."""
        if val is None:
            return False
        return hasattr(val, '__iter__') and not isinstance(val, str)
    
    def _normalize_to_string_list(self, val):
        """Convert value to list of strings."""
        if val is None:
            return []
        
        # Handle lists/arrays
        if hasattr(val, '__iter__') and not isinstance(val, str):
            try:
                return [str(item) for item in val if item is not None and not pd.isna(item)]
            except (TypeError, ValueError):
                return [str(val)]
        
        # Handle single values
        return [str(val)]
    
    def _apply_list_strategy(self, val1, val2):
        """Apply the list strategy to get comparison result."""
        if self.list_strategy == "concatenate":
            # Convert to concatenated strings
            str1 = self._concatenate_list(val1)
            str2 = self._concatenate_list(val2)
            
            # Preprocess if needed
            if self.preprocess:
                str1 = self.preprocess(str1)
                str2 = self.preprocess(str2)
            
            return float(self._sim_func(str1, str2))
        
        elif self.list_strategy == "best_match":
            # Get string lists
            list1 = self._normalize_to_string_list(val1)
            list2 = self._normalize_to_string_list(val2)
            
            if not list1 or not list2:
                return 0.0
            
            # Find best matching pair
            max_sim = 0.0
            for s1 in list1:
                for s2 in list2:
                    # Preprocess if needed
                    proc_s1 = self.preprocess(s1) if self.preprocess else s1
                    proc_s2 = self.preprocess(s2) if self.preprocess else s2
                    
                    sim = float(self._sim_func(proc_s1, proc_s2))
                    max_sim = max(max_sim, sim)
            
            return max_sim
        
        elif self.list_strategy == "set_jaccard":
            # Get string sets
            set1 = set(self._normalize_to_string_list(val1))
            set2 = set(self._normalize_to_string_list(val2))
            
            if self.preprocess:
                set1 = {self.preprocess(s) for s in set1}
                set2 = {self.preprocess(s) for s in set2}
            
            if not set1 and not set2:
                return 1.0
            if not set1 or not set2:
                return 0.0
            
            # Jaccard similarity
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            return intersection / union if union > 0 else 0.0
        
        elif self.list_strategy == "set_overlap":
            # Get string sets
            set1 = set(self._normalize_to_string_list(val1))
            set2 = set(self._normalize_to_string_list(val2))
            
            if self.preprocess:
                set1 = {self.preprocess(s) for s in set1}
                set2 = {self.preprocess(s) for s in set2}
            
            if not set1 and not set2:
                return 1.0
            if not set1 or not set2:
                return 0.0
            
            # Overlap coefficient
            intersection = len(set1 & set2)
            min_size = min(len(set1), len(set2))
            return intersection / min_size if min_size > 0 else 0.0
        
        else:
            # Default to concatenate
            str1 = self._concatenate_list(val1)
            str2 = self._concatenate_list(val2)
            
            if self.preprocess:
                str1 = self.preprocess(str1)
                str2 = self.preprocess(str2)
            
            return float(self._sim_func(str1, str2))
    
    def _concatenate_list(self, val):
        """Convert value to concatenated string."""
        if val is None or (hasattr(val, '__len__') and len(val) == 0):
            return ""
        
        # Handle lists/arrays
        if hasattr(val, '__iter__') and not isinstance(val, str):
            try:
                # Convert list to space-separated string
                return ' '.join(str(item) for item in val if item is not None and not pd.isna(item))
            except (TypeError, ValueError):
                return str(val)
        
        # Handle single values
        return str(val)


class NumericComparator(BaseComparator):
    """Numeric similarity comparator.

    Parameters
    ----------
    column : str
        Column name to compare.
    method : str
        Similarity method:
        - "absolute_difference": Similarity based on absolute difference.
        - "relative_difference": Similarity based on percentage difference.
        - "within_range": Binary similarity (1.0 if within range, 0.0 otherwise).
    max_difference : float, optional
        Maximum difference for normalization or range checking:
        - For "absolute_difference": Normalizes similarity to [0, 1] range.
        - For "relative_difference": Threshold above which similarity is 0 (Winter-compatible).
          Similarity scales linearly from 1.0 to 0.0 as relative_diff goes from 0 to max_difference.
        - For "within_range": Required; defines the acceptable range.
    list_strategy : str, optional
        Strategy for handling lists ("average", "best_match", "range_overlap", "set_jaccard").
        Default is "average".

    Notes
    -----
    The "relative_difference" method with max_difference is equivalent to Winter's
    PercentageSimilarity. For values differing by X%, where X < max_difference*100:
        similarity = 1 - (X/100) / max_difference
    If X >= max_difference*100, similarity = 0.0.
    """
    
    def __init__(
        self, 
        column: str, 
        method: str = "absolute_difference",
        max_difference: Optional[float] = None,
        list_strategy: str = None,
    ):
        super().__init__(f"NumericComparator({column}, {method}, list_strategy={list_strategy})")
        self.column = column
        self.method = method
        self.max_difference = max_difference
        self.list_strategy = list_strategy
        
        if method not in ["absolute_difference", "relative_difference", "within_range"]:
            raise ValueError(f"Unknown method: {method}")
        
        if list_strategy is not None and list_strategy not in ["average", "best_match", "range_overlap", "set_jaccard"]:
            raise ValueError(f"Unknown list_strategy: {list_strategy}")
    
    def compare(self, record1: pd.Series, record2: pd.Series) -> float:
        """Compare numeric values in specified column, handling both single values and lists."""
        try:
            val1 = record1[self.column]
            val2 = record2[self.column]
            
            # Handle missing values - properly handle arrays/series
            if self._is_null_value(val1) or self._is_null_value(val2):
                return 0.0
            
            # Check if we have list values but no list_strategy
            if self.list_strategy is None and (self._is_list_value(val1) or self._is_list_value(val2)):
                raise ValueError(f"List values detected in column '{self.column}' but list_strategy is None. Please specify a list_strategy.")
            
            # Handle comparison based on whether we have lists or single values
            if self.list_strategy is None:
                # Both are single values - compare directly
                try:
                    num1 = float(val1)
                    num2 = float(val2)
                    return self._compute_similarity(num1, num2)
                except (TypeError, ValueError):
                    return 0.0
            else:
                # Normalize values to numeric lists
                nums1 = self._normalize_to_numeric_list(val1)
                nums2 = self._normalize_to_numeric_list(val2)
                
                if not nums1 or not nums2:
                    return 0.0
                
                # Apply list strategy to get single values for comparison
                comp_val1, comp_val2 = self._apply_list_strategy(nums1, nums2)
                
                # Apply the comparison method
                return self._compute_similarity(comp_val1, comp_val2)
            
        except KeyError:
            logging.warning(f"Column '{self.column}' not found in one or both records")
            return 0.0
        except Exception as e:
            logging.warning(f"Error in NumericComparator: {e}")
            return 0.0
    
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
    
    def _is_list_value(self, val):
        """Check if value is a list/array (not a scalar string)."""
        if val is None:
            return False
        return hasattr(val, '__iter__') and not isinstance(val, str)
    
    def _normalize_to_numeric_list(self, val):
        """Convert value to list of numeric values."""
        if val is None:
            return []
        
        # Handle lists/arrays
        if hasattr(val, '__iter__') and not isinstance(val, str):
            try:
                return [float(item) for item in val if item is not None and not pd.isna(item)]
            except (TypeError, ValueError):
                return []
        
        # Handle single values
        try:
            return [float(val)]
        except (TypeError, ValueError):
            return []
    
    def _apply_list_strategy(self, nums1, nums2):
        """Apply the list strategy to get comparison values."""
        if self.list_strategy == "average":
            return sum(nums1) / len(nums1), sum(nums2) / len(nums2)
        
        elif self.list_strategy == "best_match":
            # Find the pair with minimum difference
            min_diff = float('inf')
            best_val1, best_val2 = nums1[0], nums2[0]
            
            for n1 in nums1:
                for n2 in nums2:
                    diff = abs(n1 - n2)
                    if diff < min_diff:
                        min_diff = diff
                        best_val1, best_val2 = n1, n2
            
            return best_val1, best_val2
        
        elif self.list_strategy == "range_overlap":
            # Use range boundaries for comparison
            min1, max1 = min(nums1), max(nums1)
            min2, max2 = min(nums2), max(nums2)
            
            # Calculate overlap ratio
            overlap_start = max(min1, min2)
            overlap_end = min(max1, max2)
            
            if overlap_start <= overlap_end:
                overlap = overlap_end - overlap_start
                total_range = max(max1, max2) - min(min1, min2)
                return overlap, total_range if total_range > 0 else 1.0
            else:
                return 0.0, 1.0  # No overlap
        
        elif self.list_strategy == "set_jaccard":
            # Use set intersection/union for Jaccard similarity
            set1, set2 = set(nums1), set(nums2)
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            return intersection, union if union > 0 else 1.0
        
        else:
            # Default to average
            return sum(nums1) / len(nums1), sum(nums2) / len(nums2)
    
    def _compute_similarity(self, val1, val2):
        """Compute similarity based on the comparison method."""
        if self.method == "absolute_difference":
            diff = abs(val1 - val2)
            if self.max_difference is not None:
                return max(0.0, 1.0 - diff / self.max_difference)
            else:
                # Return inverse similarity (closer to 0 = more similar)
                return 1.0 / (1.0 + diff)
        
        elif self.method == "relative_difference":
            if val1 == 0 and val2 == 0:
                return 1.0
            max_val = max(abs(val1), abs(val2))
            if max_val == 0:
                return 1.0
            # Calculate relative difference as percentage
            relative_diff = abs(val1 - val2) / max_val

            # Apply max_difference threshold if specified (Winter-compatible behavior)
            if self.max_difference is not None:
                # If difference exceeds threshold, similarity is 0
                if relative_diff >= self.max_difference:
                    return 0.0
                # Scale similarity linearly from 1.0 to 0.0 within threshold
                return 1.0 - (relative_diff / self.max_difference)
            else:
                # Without threshold, use simple inverse similarity
                return max(0.0, 1.0 - relative_diff)
        
        elif self.method == "within_range":
            if self.max_difference is None:
                raise ValueError("max_difference required for within_range method")
            return 1.0 if abs(val1 - val2) <= self.max_difference else 0.0
        
        else:
            return 0.0


class DateComparator(BaseComparator):
    """Date similarity comparator.
    
    Parameters
    ----------
    column : str
        Column name to compare.
    max_days_difference : int, optional
        Maximum days difference for normalization.
    list_strategy : str, optional
        Strategy for handling lists ("closest_dates", "range_overlap", "average_dates", "latest_dates", "earliest_dates").
        Default is "closest_dates".
    """
    
    def __init__(self, column: str, max_days_difference: Optional[int] = None, list_strategy: str = None):
        super().__init__(f"DateComparator({column}, list_strategy={list_strategy})")
        self.column = column
        self.max_days_difference = max_days_difference
        self.list_strategy = list_strategy
        
        if list_strategy is not None and list_strategy not in ["closest_dates", "range_overlap", "average_dates", "latest_dates", "earliest_dates"]:
            raise ValueError(f"Unknown list_strategy: {list_strategy}")
    
    def compare(self, record1: pd.Series, record2: pd.Series) -> float:
        """Compare date values in specified column, handling both single values and lists."""
        try:
            val1 = record1[self.column]
            val2 = record2[self.column]
            
            # Handle missing values - properly handle arrays/series
            if self._is_null_value(val1) or self._is_null_value(val2):
                return 0.0
            
            # Check if we have list values but no list_strategy
            if self.list_strategy is None and (self._is_list_value(val1) or self._is_list_value(val2)):
                raise ValueError(f"List values detected in column '{self.column}' but list_strategy is None. Please specify a list_strategy.")
            
            # Handle comparison based on whether we have lists or single values
            if self.list_strategy is None:
                # Both are single values - compare directly
                try:
                    date1 = pd.to_datetime(val1)
                    date2 = pd.to_datetime(val2)
                    diff_days = abs((date1 - date2).days)
                    return self._compute_similarity(diff_days)
                except (TypeError, ValueError):
                    return 0.0
            else:
                # Normalize values to date lists
                dates1 = self._normalize_to_date_list(val1)
                dates2 = self._normalize_to_date_list(val2)
                
                if not dates1 or not dates2:
                    return 0.0
                
                # Apply list strategy to get comparison values
                comp_result = self._apply_list_strategy(dates1, dates2)
                
                # Apply the comparison method
                return self._compute_similarity(comp_result)
            
        except KeyError:
            logging.warning(f"Column '{self.column}' not found in one or both records")
            return 0.0
        except Exception as e:
            logging.warning(f"Error in DateComparator: {e}")
            return 0.0
    
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
    
    def _is_list_value(self, val):
        """Check if value is a list/array (not a scalar string)."""
        if val is None:
            return False
        return hasattr(val, '__iter__') and not isinstance(val, str)
    
    def _normalize_to_date_list(self, val):
        """Convert value to list of datetime objects."""
        if val is None:
            return []
        
        # Handle lists/arrays
        if hasattr(val, '__iter__') and not isinstance(val, str):
            dates = []
            for item in val:
                if item is not None and not pd.isna(item):
                    try:
                        dates.append(pd.to_datetime(item))
                    except (TypeError, ValueError):
                        continue
            return dates
        
        # Handle single values
        try:
            return [pd.to_datetime(val)]
        except (TypeError, ValueError):
            return []
    
    def _apply_list_strategy(self, dates1, dates2):
        """Apply the list strategy to get comparison values."""
        if self.list_strategy == "closest_dates":
            # Find the pair with minimum difference
            min_diff = float('inf')
            best_date1, best_date2 = dates1[0], dates2[0]
            
            for d1 in dates1:
                for d2 in dates2:
                    diff = abs((d1 - d2).days)
                    if diff < min_diff:
                        min_diff = diff
                        best_date1, best_date2 = d1, d2
            
            return abs((best_date1 - best_date2).days)
        
        elif self.list_strategy == "range_overlap":
            # Calculate date range overlap
            min1, max1 = min(dates1), max(dates1)
            min2, max2 = min(dates2), max(dates2)
            
            overlap_start = max(min1, min2)
            overlap_end = min(max1, max2)
            
            if overlap_start <= overlap_end:
                overlap_days = (overlap_end - overlap_start).days
                total_range_days = (max(max1, max2) - min(min1, min2)).days
                return (overlap_days, total_range_days) if total_range_days > 0 else (1.0, 1.0)
            else:
                return (0.0, 1.0)  # No overlap
        
        elif self.list_strategy == "average_dates":
            # Compare average dates
            avg1 = pd.to_datetime(sum([d.value for d in dates1]) / len(dates1))
            avg2 = pd.to_datetime(sum([d.value for d in dates2]) / len(dates2))
            return abs((avg1 - avg2).days)
        
        elif self.list_strategy == "latest_dates":
            # Compare latest dates
            latest1, latest2 = max(dates1), max(dates2)
            return abs((latest1 - latest2).days)
        
        elif self.list_strategy == "earliest_dates":
            # Compare earliest dates
            earliest1, earliest2 = min(dates1), min(dates2)
            return abs((earliest1 - earliest2).days)
        
        else:
            # Default to closest dates
            min_diff = float('inf')
            for d1 in dates1:
                for d2 in dates2:
                    diff = abs((d1 - d2).days)
                    if diff < min_diff:
                        min_diff = diff
            return min_diff
    
    def _compute_similarity(self, comp_result):
        """Compute similarity based on the comparison result."""
        if self.list_strategy == "range_overlap":
            overlap_days, total_range_days = comp_result
            if total_range_days == 0:
                return 1.0
            return overlap_days / total_range_days
        
        else:
            # For all other strategies, comp_result is days difference
            diff_days = comp_result
            
            if self.max_days_difference is not None:
                return max(0.0, 1.0 - diff_days / self.max_days_difference)
            else:
                # Return inverse similarity
                return 1.0 / (1.0 + diff_days)