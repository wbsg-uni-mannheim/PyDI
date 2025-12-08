"""
Utility functions for PyDI.

This module exposes generic helper functions that can be reused across
modules, including a similarity registry for textdistance metrics.
"""

from __future__ import annotations

# Import similarity registry utilities
from .similarity_registry import SimilarityRegistry, get_similarity_function, list_similarity_functions

__all__ = [
    "SimilarityRegistry",
    "get_similarity_function",
    "list_similarity_functions",
]
