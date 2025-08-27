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
    preprocess : callable, optional
        Function to preprocess values before comparison.
    """
    
    def __init__(
        self, 
        column: str, 
        similarity_function: str = "jaro_winkler",
        preprocess: Optional[Callable[[str], str]] = None,
    ):
        super().__init__(f"StringComparator({column}, {similarity_function})")
        self.column = column
        self.similarity_function = similarity_function
        self.preprocess = preprocess
        
        # Get similarity function
        try:
            self._sim_func = SimilarityRegistry.get_function(similarity_function)
        except ValueError as e:
            available = SimilarityRegistry.get_recommended_functions("entity_matching")
            raise ValueError(f"{e}. Recommended: {available}")
    
    def compare(self, record1: pd.Series, record2: pd.Series) -> float:
        """Compare string values in specified column."""
        try:
            val1 = record1[self.column]
            val2 = record2[self.column]
            
            # Handle missing values
            if pd.isna(val1) or pd.isna(val2):
                return 0.0
            
            # Convert to strings
            val1, val2 = str(val1), str(val2)
            
            # Preprocess if needed
            if self.preprocess:
                val1 = self.preprocess(val1)
                val2 = self.preprocess(val2)
            
            return float(self._sim_func(val1, val2))
            
        except KeyError:
            logging.warning(f"Column '{self.column}' not found in one or both records")
            return 0.0
        except Exception as e:
            logging.warning(f"Error in StringComparator: {e}")
            return 0.0


class NumericComparator(BaseComparator):
    """Numeric similarity comparator.
    
    Parameters
    ----------
    column : str
        Column name to compare.
    method : str
        Similarity method ("absolute_difference", "relative_difference", "within_range").
    max_difference : float, optional
        Maximum difference for normalization or range checking.
    """
    
    def __init__(
        self, 
        column: str, 
        method: str = "absolute_difference",
        max_difference: Optional[float] = None,
    ):
        super().__init__(f"NumericComparator({column}, {method})")
        self.column = column
        self.method = method
        self.max_difference = max_difference
        
        if method not in ["absolute_difference", "relative_difference", "within_range"]:
            raise ValueError(f"Unknown method: {method}")
    
    def compare(self, record1: pd.Series, record2: pd.Series) -> float:
        """Compare numeric values in specified column."""
        try:
            val1 = record1[self.column]
            val2 = record2[self.column]
            
            # Handle missing values
            if pd.isna(val1) or pd.isna(val2):
                return 0.0
            
            # Convert to float
            val1, val2 = float(val1), float(val2)
            
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
                diff = abs(val1 - val2) / max_val
                return max(0.0, 1.0 - diff)
            
            elif self.method == "within_range":
                if self.max_difference is None:
                    raise ValueError("max_difference required for within_range method")
                return 1.0 if abs(val1 - val2) <= self.max_difference else 0.0
            
        except KeyError:
            logging.warning(f"Column '{self.column}' not found in one or both records")
            return 0.0
        except Exception as e:
            logging.warning(f"Error in NumericComparator: {e}")
            return 0.0


class DateComparator(BaseComparator):
    """Date similarity comparator.
    
    Parameters
    ----------
    column : str
        Column name to compare.
    max_days_difference : int, optional
        Maximum days difference for normalization.
    """
    
    def __init__(self, column: str, max_days_difference: Optional[int] = None):
        super().__init__(f"DateComparator({column})")
        self.column = column
        self.max_days_difference = max_days_difference
    
    def compare(self, record1: pd.Series, record2: pd.Series) -> float:
        """Compare date values in specified column."""
        try:
            val1 = record1[self.column]
            val2 = record2[self.column]
            
            # Handle missing values
            if pd.isna(val1) or pd.isna(val2):
                return 0.0
            
            # Convert to datetime
            date1 = pd.to_datetime(val1)
            date2 = pd.to_datetime(val2)
            
            # Calculate difference in days
            diff_days = abs((date1 - date2).days)
            
            if self.max_days_difference is not None:
                return max(0.0, 1.0 - diff_days / self.max_days_difference)
            else:
                # Return inverse similarity
                return 1.0 / (1.0 + diff_days)
            
        except KeyError:
            logging.warning(f"Column '{self.column}' not found in one or both records")
            return 0.0
        except Exception as e:
            logging.warning(f"Error in DateComparator: {e}")
            return 0.0