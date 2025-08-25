"""
Schema matching tools for PyDI.

This module provides schema matching algorithms for finding correspondences
between database schemas. It includes label-based, instance-based, and
duplicate-based matching strategies.
"""

# Base classes and types
from .base import BaseSchemaMatcher, SchemaMapping

# Matching algorithms
from .label_based import LabelBasedSchemaMatcher
from .instance_based import InstanceBasedSchemaMatcher
from .duplicate_based import DuplicateBasedSchemaMatcher

# Evaluation utilities
from .evaluation import SchemaMappingEvaluator

__all__ = [
    "BaseSchemaMatcher",
    "SchemaMapping",
    "LabelBasedSchemaMatcher",
    "InstanceBasedSchemaMatcher",
    "DuplicateBasedSchemaMatcher",
    "SchemaMappingEvaluator",
]
