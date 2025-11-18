"""
Schema matching tools for PyDI.

This module provides schema matching algorithms for finding correspondences
between database schemas. It includes label-based, instance-based, and
duplicate-based matching strategies.
"""

# Base classes and types
from .base import BaseSchemaMatcher, SchemaMapping, get_schema_columns

# Matching algorithms
from .label_based import LabelBasedSchemaMatcher
from .instance_based import InstanceBasedSchemaMatcher
from .duplicate_based import DuplicateBasedSchemaMatcher
from .llm_based import LLMBasedSchemaMatcher

# Evaluation utilities
from .evaluation import SchemaMappingEvaluator

__all__ = [
    "BaseSchemaMatcher",
    "SchemaMapping",
    "get_schema_columns",
    "LabelBasedSchemaMatcher",
    "InstanceBasedSchemaMatcher",
    "DuplicateBasedSchemaMatcher",
    "LLMBasedSchemaMatcher",
    "SchemaMappingEvaluator",
]
