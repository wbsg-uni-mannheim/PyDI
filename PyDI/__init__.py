"""
PyDI: Python Data Integration Framework
=======================================

This package provides tools for end‑to‑end data integration, including
schema matching, blocking, entity matching, information extraction,
normalization and validation, and data fusion. It is inspired by the WInte.r Java framework but takes a
pandas‑first approach. All modules aim to expose simple, composable functions
and classes with stable signatures and rich docstrings.

Subpackages
-----------

``schemamatching``
    Schema matching algorithms, evaluation utilities, and translation.
``entitymatching``
    Blocking and entity matching strategies.
``informationextraction``
    Extraction components for feature engineering.
``normalization``
    Normalization and validation components.
``fusion``
    Conflict resolution functions and data fusion engine.
``utils``
    Generic utilities such as comparators, logging helpers, and data profiling.

See the documentation in `docs/high_level_design.md` for detailed design
guidelines and the high‑level API.
"""

__all__ = [
    "io",
    "schemamatching",
    "entitymatching",
    "informationextraction",
    "normalization",
    "fusion",
    "utils",
]
