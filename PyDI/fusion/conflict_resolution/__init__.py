"""
Conflict resolution functions for PyDI data fusion.

This package provides built-in conflict resolution functions for different
data types and fusion scenarios.
"""

from .string import longest_string, shortest_string, most_complete
from .numeric import average, median, maximum, minimum, sum_values
from .date import most_recent, earliest
from .list import union, intersection, intersection_k_sources
from .general import voting, favour_sources, random_value, weighted_voting, prefer_higher_trust

__all__ = [
    # String rules
    "longest_string", "shortest_string", "most_complete",
    # Numeric rules
    "average", "median", "maximum", "minimum", "sum_values",
    # Date rules
    "most_recent", "earliest",
    # List rules
    "union", "intersection", "intersection_k_sources",
    # General rules
    "voting", "favour_sources", "random_value", "weighted_voting", "prefer_higher_trust",
]
