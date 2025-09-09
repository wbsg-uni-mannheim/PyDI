"""
Profiling tools for PyDI.

This module provides data profiling capabilities using popular libraries
(ydata-profiling and Sweetviz) to generate HTML reports for individual 
datasets and comparisons between two datasets. A quick summary method 
provides basic statistics without creating heavy reports.
"""

from .profiler import DataProfiler

__all__ = [
    "DataProfiler",
]
