"""
Data translation utilities for PyDI.

This module provides tools for translating DataFrames according to schema
mappings. It includes abstract base classes and concrete implementations
for applying column transformations based on schema correspondence information.

The primary use case is applying schema mappings produced by the schemamatching
module to align DataFrames to a common target schema.
"""

# Base classes
from .base import BaseTranslator

# Concrete implementations  
from .mapping_translator import MappingTranslator

__all__ = [
    "BaseTranslator",
    "MappingTranslator",
]
