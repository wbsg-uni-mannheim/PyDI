"""
Blocking and entity matching tools for PyDI.

This module provides blocking and entity matching algorithms for finding
duplicate and similar records across datasets. It includes matchers and
comparators for computing entity correspondences.
"""

# Base classes and types
from .base import BaseMatcher, BaseComparator, CorrespondenceSet

# Blocking strategies (subpackage)
from .blocking import (
    BaseBlocker,
    NoBlocker,
    StandardBlocker,
    SortedNeighbourhoodBlocker,
    TokenBlocker,
    EmbeddingBlocker,
)

# Matching algorithms
from .rule_based import RuleBasedMatcher
from .ml_based import MLBasedMatcher
from .llm_based import LLMBasedMatcher
from .plm_based import PLMBasedMatcher

# Feature extraction and text formatting
from .feature_extraction import FeatureExtractor, VectorFeatureExtractor
from .text_formatting import TextFormatter

# Comparators
from .comparators import StringComparator, NumericComparator, DateComparator

# Evaluation tools
from .evaluation import EntityMatchingEvaluator

__all__ = [
    "BaseMatcher",
    "BaseComparator",
    "CorrespondenceSet",
    "BaseBlocker",
    "NoBlocker",
    "StandardBlocker",
    "SortedNeighbourhoodBlocker",
    "TokenBlocker",
    "EmbeddingBlocker",
    "RuleBasedMatcher",
    "MLBasedMatcher",
    "LLMBasedMatcher",
    "PLMBasedMatcher",
    "FeatureExtractor",
    "VectorFeatureExtractor",
    "TextFormatter",
    "StringComparator",
    "NumericComparator",
    "DateComparator",
    "EntityMatchingEvaluator",
]
