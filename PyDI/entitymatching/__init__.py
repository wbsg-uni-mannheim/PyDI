"""
Blocking and entity matching tools for PyDI.

This module provides blocking and entity matching algorithms for finding
duplicate and similar records across datasets. It includes matchers and
comparators for computing entity correspondences.
"""

# Base classes and types
from .base import BaseMatcher, BaseComparator, CorrespondenceSet, ensure_record_ids

# Matching algorithms
from .rule_based import RuleBasedMatcher
from .ml_based import MLBasedMatcher

# Feature extraction
from .feature_extraction import FeatureExtractor, VectorFeatureExtractor

# Comparators
from .comparators import StringComparator, NumericComparator, DateComparator

# Evaluation tools
from .evaluation import EntityMatchingEvaluator

__all__ = [
    "BaseMatcher",
    "BaseComparator", 
    "CorrespondenceSet",
    "ensure_record_ids",
    "RuleBasedMatcher",
    "MLBasedMatcher",
    "FeatureExtractor",
    "VectorFeatureExtractor",
    "StringComparator",
    "NumericComparator",
    "DateComparator",
    "EntityMatchingEvaluator",
]