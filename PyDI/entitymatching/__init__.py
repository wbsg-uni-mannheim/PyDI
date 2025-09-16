"""
Blocking and entity matching tools for PyDI.

This module provides blocking and entity matching algorithms for finding
duplicate and similar records across datasets. It includes matchers and
comparators for computing entity correspondences.
"""

# Base classes and types
from .base import BaseMatcher, BaseComparator, CorrespondenceSet, ensure_record_ids

# Blocking strategies (subpackage)
from .blocking import (
    BaseBlocker,
    NoBlocking,
    StandardBlocking,
    SortedNeighbourhood,
    TokenBlocking,
    EmbeddingBlocking,
)

# Matching algorithms
from .rule_based import RuleBasedMatcher
from .ml_based import MLBasedMatcher
from .llm_based import LLMBasedMatcher

# Feature extraction
from .feature_extraction import FeatureExtractor, VectorFeatureExtractor

# Comparators
from .comparators import StringComparator, NumericComparator, DateComparator

# Evaluation tools
from .evaluation import EntityMatchingEvaluator, year_only_match
from .blocking.blocking_evaluation import BlockingEvaluator

__all__ = [
    "BaseMatcher",
    "BaseComparator",
    "CorrespondenceSet",
    "ensure_record_ids",
    "BaseBlocker",
    "NoBlocking",
    "StandardBlocking",
    "SortedNeighbourhood",
    "TokenBlocking",
    "EmbeddingBlocking",
    "RuleBasedMatcher",
    "MLBasedMatcher",
    "LLMBasedMatcher",
    "FeatureExtractor",
    "VectorFeatureExtractor",
    "StringComparator",
    "NumericComparator",
    "DateComparator",
    "EntityMatchingEvaluator",
    "year_only_match",
    "BlockingEvaluator",
]
