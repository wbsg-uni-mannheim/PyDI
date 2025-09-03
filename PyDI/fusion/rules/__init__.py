"""
Conflict resolution functions for PyDI data fusion.

This package provides built-in conflict resolution functions for different
data types and fusion scenarios.
"""

from .string import LongestString, ShortestString
from .numeric import Average, Median, Maximum, Minimum
from .date import MostRecent, Earliest
from .list import Union, Intersection, IntersectionKSources
from .general import Voting, FavourSources, RandomValue

__all__ = [
    # String rules
    "LongestString", "ShortestString",
    # Numeric rules
    "Average", "Median", "Maximum", "Minimum", 
    # Date rules
    "MostRecent", "Earliest",
    # List rules
    "Union", "Intersection", "IntersectionKSources",
    # General rules
    "Voting", "FavourSources", "RandomValue",
]