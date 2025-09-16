"""
Data fusion tools for PyDI.

This module provides comprehensive data fusion capabilities including:
- Record grouping from correspondences
- Attribute-level conflict resolution
- Strategy-based fusion management
- Quality evaluation and reporting
- Provenance tracking

The implementation is designed to be pandas-first, modular, and extensible.
"""

from __future__ import annotations

# New fusion framework components
from .base import (
    ConflictResolutionFunction,
    AttributeValueFuser, 
    FusionContext,
    FusionResult,
    RecordGroup,
)
from .strategy import DataFusionStrategy
from .engine import DataFusionEngine, build_record_groups_from_correspondences
from .evaluation import (
    DataFusionEvaluator, 
    calculate_consistency_metrics, 
    calculate_coverage_metrics,
    exact_match,
    tokenized_match,
    year_only_match,
    numeric_tolerance_match,
    set_equality_match,
    boolean_match,
)
from .reporting import FusionReport
from .provenance import ProvenanceTracker, ProvenanceInfo

# Import built-in conflict resolution functions
from .conflict_resolution import (
    # String functions
    longest_string, shortest_string, most_complete,
    # Numeric functions
    average, median, maximum, minimum, sum_values,
    # Date functions
    most_recent, earliest,
    # List functions
    union, intersection, intersection_k_sources,
    # General functions
    voting, favour_sources, random_value, weighted_voting, prefer_higher_trust,
)

# Import analysis utilities
from .analysis import (
    analyze_attribute_coverage,
    compare_dataset_schemas,
    detect_attribute_conflicts,
    analyze_conflicts_preview,
    print_conflict_preview,
    AttributeCoverageAnalyzer,
)



# Define what's available when importing * from this module
# Convenience functions for creating strategies

def create_empty_strategy(name: str = "empty") -> DataFusionStrategy:
    """Create an empty fusion strategy.
    
    Parameters
    ----------
    name : str
        Name for the strategy.
        
    Returns
    -------
    DataFusionStrategy
        An empty strategy ready for custom rule configuration.
    """
    return DataFusionStrategy(name)


# Define what's available when importing * from this module
__all__ = [
    # Framework components
    "DataFusionEngine", "DataFusionStrategy", "DataFusionEvaluator",
    "ConflictResolutionFunction", "AttributeValueFuser",
    "FusionContext", "FusionResult", "RecordGroup", "FusionReport",
    "ProvenanceTracker", "ProvenanceInfo", 
    "calculate_consistency_metrics", "calculate_coverage_metrics",
    "build_record_groups_from_correspondences",
    
    # Conflict resolution functions
    "longest_string", "shortest_string", "most_complete",
    "average", "median", "maximum", "minimum", "sum_values",
    "most_recent", "earliest",
    "union", "intersection", "intersection_k_sources",
    "voting", "favour_sources", "random_value", "weighted_voting", "prefer_higher_trust",
    
    # Analysis utilities
    "compare_dataset_schemas", "detect_attribute_conflicts",
    "analyze_conflicts_preview", "print_conflict_preview",
    "AttributeCoverageAnalyzer",
    
    # Evaluation functions
    "exact_match", "tokenized_match", "year_only_match", 
    "numeric_tolerance_match", "set_equality_match", "boolean_match",
    
    # Utility functions
    "create_empty_strategy",
]
