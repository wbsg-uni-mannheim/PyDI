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

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

# New fusion framework components
from .base import (
    ConflictResolutionFunction,
    AttributeValueFuser, 
    FusionContext,
    FusionResult,
    RecordGroup,
)
from .strategy import DataFusionStrategy, EvaluationRule
from .engine import DataFusionEngine, build_record_groups_from_correspondences
from .evaluation import DataFusionEvaluator, FusionQualityMetrics
from .reporting import FusionReport
from .provenance import ProvenanceTracker, ProvenanceInfo

# Import built-in conflict resolution rules (class-based)
from .rules import (
    # String rules
    LongestString, ShortestString,
    # Numeric rules
    Average, Median, Maximum, Minimum,
    # Date rules
    MostRecent, Earliest,
    # List rules
    Union, Intersection, IntersectionKSources,
    # General rules
    Voting, FavourSources, RandomValue,
)

# Import function-based fusion rules
from .fusion_rules import (
    # String rules  
    longest_string, shortest_string, most_complete,
    # Numeric rules
    average, median, maximum, minimum, sum_values,
    # Date rules
    most_recent, earliest,
    # List rules
    union, intersection, intersection_k_sources,
    # Meta rules
    voting, favour_sources, random_value, weighted_voting,
    # Convenient aliases
    LONGEST, SHORTEST, MOST_COMPLETE,
    AVG, AVERAGE, MEDIAN, MAX, MAXIMUM, MIN, MINIMUM, SUM,
    LATEST, MOST_RECENT, EARLIEST,
    UNION, INTERSECTION, INTERSECTION_K,
    VOTE, VOTING, FAVOUR, RANDOM, WEIGHTED_VOTE,
    # Utility functions
    create_function_resolver,
)

# Import analysis utilities
from .analysis import (
    analyze_attribute_coverage,
    compare_dataset_schemas,
    detect_attribute_conflicts,
    AttributeCoverageAnalyzer,
)


# Backward compatibility: maintain legacy DataFuser and FusionRule classes

@dataclass
class FusionRule:
    """Represent a fusion rule for a single attribute (legacy interface).

    Parameters
    ----------
    strategy : str
        Name of the built‑in strategy (e.g., ``"longest"``, ``"most_recent"``).
    function : callable, optional
        A custom function that takes a list of values and returns a fused
        value. If provided, it overrides ``strategy``.
    
    Notes
    -----
    This class is maintained for backward compatibility. New code should
    use the DataFusionStrategy and ConflictResolutionFunction framework.
    """

    strategy: str
    function: Optional[Callable[[List[Any]], Any]] = None


class DataFuser:
    """Fuse multiple datasets into a single DataFrame (legacy interface).

    The current implementation is deliberately simple: it merges rows
    specified in the correspondences and applies fusion rules per
    attribute. Rows without correspondences are appended as‑is. 
    
    Notes
    -----
    This class is maintained for backward compatibility. New code should
    use the DataFusionEngine with DataFusionStrategy framework.
    """

    def fuse(
        self,
        datasets: List[pd.DataFrame],
        correspondences: pd.DataFrame,
        *,
        rules: Dict[str, FusionRule],
    ) -> pd.DataFrame:
        # Build a mapping from id to row for quick lookup
        id_to_row: Dict[str, pd.Series] = {}
        for df in datasets:
            id_to_row.update({row["_id"]: row for _, row in df.iterrows()})
        used_ids = set()
        fused_records: List[Dict[str, Any]] = []
        # Fuse matched pairs
        for _, corr in correspondences.to_dataframe().iterrows():
            id1, id2 = corr["id1"], corr["id2"]
            used_ids.add(id1)
            used_ids.add(id2)
            row1 = id_to_row.get(id1)
            row2 = id_to_row.get(id2)
            if row1 is None or row2 is None:
                continue
            fused: Dict[str, Any] = {}
            columns = set(row1.index) | set(row2.index)
            for col in columns:
                val1 = row1.get(col)
                val2 = row2.get(col)
                if col in rules:
                    rule = rules[col]
                    values = [v for v in [val1, val2] if pd.notna(v)]
                    if rule.function:
                        fused[col] = rule.function(values)
                    else:
                        strategy = rule.strategy
                        if strategy == "longest":
                            fused[col] = max(values, key=lambda x: len(
                                str(x))) if values else None
                        elif strategy == "shortest":
                            fused[col] = min(values, key=lambda x: len(
                                str(x))) if values else None
                        elif strategy == "most_recent":
                            # assumes values are comparable (e.g., datetime or sortable strings)
                            fused[col] = max(values) if values else None
                        elif strategy == "union":
                            fused[col] = list({val for val in values})
                        elif strategy == "majority":
                            from collections import Counter

                            fused[col] = Counter(values).most_common(1)[
                                0][0] if values else None
                        else:
                            fused[col] = values[0] if values else None
                else:
                    # default: prefer non-null, fall back to any
                    fused[col] = val1 if pd.notna(val1) else val2
            # assign a new fused ID
            fused["_id"] = f"fused_{id1}_{id2}"
            fused_records.append(fused)
        # Append unmatched rows
        for id_, row in id_to_row.items():
            if id_ not in used_ids:
                fused_records.append(row.to_dict())
        return pd.DataFrame(fused_records)


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
    # New framework components
    "DataFusionEngine", "DataFusionStrategy", "DataFusionEvaluator",
    "ConflictResolutionFunction", "AttributeValueFuser", "EvaluationRule",
    "FusionContext", "FusionResult", "RecordGroup", "FusionReport",
    "ProvenanceTracker", "ProvenanceInfo", "FusionQualityMetrics",
    "build_record_groups_from_correspondences",
    
    # Built-in conflict resolution rules (class-based)
    "LongestString", "ShortestString", 
    "Average", "Median", "Maximum", "Minimum",
    "MostRecent", "Earliest",
    "Union", "Intersection", "IntersectionKSources",
    "Voting", "FavourSources", "RandomValue",
    
    # Function-based fusion rules
    "longest_string", "shortest_string", "most_complete",
    "average", "median", "maximum", "minimum", "sum_values",
    "most_recent", "earliest",
    "union", "intersection", "intersection_k_sources",
    "voting", "favour_sources", "random_value", "weighted_voting",
    
    # Convenient function aliases
    "LONGEST", "SHORTEST", "MOST_COMPLETE",
    "AVG", "AVERAGE", "MEDIAN", "MAX", "MAXIMUM", "MIN", "MINIMUM", "SUM",
    "LATEST", "MOST_RECENT", "EARLIEST",
    "UNION", "INTERSECTION", "INTERSECTION_K",
    "VOTE", "VOTING", "FAVOUR", "RANDOM", "WEIGHTED_VOTE",
    
    # Utility functions
    "create_function_resolver", "create_empty_strategy",
    
    # Analysis utilities
    "analyze_attribute_coverage", "compare_dataset_schemas", "detect_attribute_conflicts",
    "AttributeCoverageAnalyzer",
    
    # Legacy components (for backward compatibility)
    "DataFuser", "FusionRule",
]