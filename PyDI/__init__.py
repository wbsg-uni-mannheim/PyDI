"""
PyDI: Python Data Integration Framework
=======================================

This package provides tools for end‑to‑end data integration, including profiling,
schema matching, blocking, entity matching, data translation, information extraction,
normalization and validation, and data fusion. It is inspired by the WInte.r Java framework but takes a
pandas‑first approach. All modules aim to expose simple, composable functions
and classes with stable signatures and rich docstrings.

Subpackages
-----------

``profiling``
    Wrappers around ydata‑profiling and Sweetviz for dataset profiling.
``schemamatching``
    Schema matching algorithms and evaluation utilities.
``entitymatching``
    Blocking and entity matching strategies.
``datatranslation``
    Translation components for schema mapping.
``informationextraction``
    Extraction components for feature engineering.
``normalization``
    Normalization and validation components.
``fusion``
    Conflict resolution functions and data fusion engine.
``utils``
    Generic utilities such as comparators and logging helpers.

See the documentation in `docs/high_level_design.md` for detailed design
guidelines and the high‑level API.
"""

__all__ = [
    "io",
    "profiling",
    "schemamatching",
    "entitymatching",
    "datatranslation",
    "informationextraction",
    "normalization",
    "fusion",
    "utils",
]
