"""
Function-based conflict resolution rules for data fusion.

This module provides simple functions for resolving conflicts during data fusion,
inspired by Winter's comprehensive rule set. Each function takes a list of values
and returns a tuple of (resolved_value, confidence, metadata).
"""

from datetime import datetime
from typing import Any, List, Tuple, Dict, Optional, Union, Set
import pandas as pd
import numpy as np
from collections import Counter
import random


# Type alias for fusion rule result
FusionResult = Tuple[Any, float, Dict[str, Any]]


def _filter_valid_values(values: List[Any]) -> List[Any]:
    """Filter out None and NaN values."""
    return [v for v in values if v is not None and pd.notna(v)]


def _calculate_tie_confidence(winner_count: int, second_count: int, total_values: int) -> float:
    """Calculate confidence when there might be ties."""
    if total_values <= 1:
        return 1.0
    if winner_count > second_count:
        return 0.5 + (winner_count - second_count) / total_values * 0.5
    else:
        return 0.5  # Perfect tie


# =============================================================================
# STRING RULES
# =============================================================================

def longest_string(values: List[Any], **kwargs) -> FusionResult:
    """
    Select the longest string value.
    
    Parameters
    ----------
    values : List[Any]
        List of values to resolve conflicts from
    **kwargs
        Additional context (ignored)
        
    Returns
    -------
    FusionResult
        Tuple of (resolved_value, confidence, metadata)
    """
    valid_values = _filter_valid_values(values)
    if not valid_values:
        return None, 0.0, {"reason": "no_valid_values"}
    
    # Convert to strings and calculate lengths
    string_values = [(str(v), len(str(v))) for v in valid_values]
    string_values.sort(key=lambda x: (-x[1], x[0]))  # Longest first, then alphabetical
    
    winner = string_values[0]
    winner_length = winner[1]
    
    # Calculate confidence
    if len(string_values) > 1:
        second_length = string_values[1][1]
        if winner_length > second_length:
            confidence = 0.5 + (winner_length - second_length) / winner_length * 0.5
        else:
            confidence = 0.5  # Tie
    else:
        confidence = 1.0
    
    return winner[0], confidence, {
        "rule": "longest_string",
        "selected_length": winner_length,
        "num_candidates": len(string_values),
        "all_lengths": [x[1] for x in string_values]
    }


def shortest_string(values: List[Any], **kwargs) -> FusionResult:
    """
    Select the shortest string value.
    
    Parameters
    ----------
    values : List[Any]
        List of values to resolve conflicts from
    **kwargs
        Additional context (ignored)
        
    Returns
    -------
    FusionResult
        Tuple of (resolved_value, confidence, metadata)
    """
    valid_values = _filter_valid_values(values)
    if not valid_values:
        return None, 0.0, {"reason": "no_valid_values"}
    
    # Convert to strings and calculate lengths
    string_values = [(str(v), len(str(v))) for v in valid_values]
    string_values.sort(key=lambda x: (x[1], x[0]))  # Shortest first, then alphabetical
    
    winner = string_values[0]
    winner_length = winner[1]
    
    # Calculate confidence
    if len(string_values) > 1:
        second_length = string_values[1][1]
        if winner_length < second_length:
            confidence = 0.5 + (second_length - winner_length) / second_length * 0.5
        else:
            confidence = 0.5  # Tie
    else:
        confidence = 1.0
    
    return winner[0], confidence, {
        "rule": "shortest_string",
        "selected_length": winner_length,
        "num_candidates": len(string_values),
        "all_lengths": [x[1] for x in string_values]
    }


def most_complete(values: List[Any], **kwargs) -> FusionResult:
    """
    Select the string with most non-whitespace characters.
    
    Parameters
    ----------
    values : List[Any]
        List of values to resolve conflicts from
    **kwargs
        Additional context (ignored)
        
    Returns
    -------
    FusionResult
        Tuple of (resolved_value, confidence, metadata)
    """
    valid_values = _filter_valid_values(values)
    if not valid_values:
        return None, 0.0, {"reason": "no_valid_values"}
    
    # Calculate non-whitespace character counts
    completeness = []
    for v in valid_values:
        s = str(v)
        non_ws_count = len(s.replace(' ', '').replace('\t', '').replace('\n', ''))
        completeness.append((v, non_ws_count))
    
    completeness.sort(key=lambda x: (-x[1], str(x[0])))  # Most complete first
    
    winner = completeness[0]
    winner_count = winner[1]
    
    # Calculate confidence
    if len(completeness) > 1:
        second_count = completeness[1][1]
        confidence = _calculate_tie_confidence(winner_count, second_count, len(completeness))
    else:
        confidence = 1.0
    
    return winner[0], confidence, {
        "rule": "most_complete",
        "selected_completeness": winner_count,
        "num_candidates": len(completeness),
        "all_completeness": [x[1] for x in completeness]
    }


# =============================================================================
# NUMERIC RULES
# =============================================================================

def average(values: List[Any], **kwargs) -> FusionResult:
    """
    Calculate the average of numeric values.
    
    Parameters
    ----------
    values : List[Any]
        List of values to resolve conflicts from
    **kwargs
        Additional context (ignored)
        
    Returns
    -------
    FusionResult
        Tuple of (resolved_value, confidence, metadata)
    """
    valid_values = _filter_valid_values(values)
    if not valid_values:
        return None, 0.0, {"reason": "no_valid_values"}
    
    # Convert to numeric values
    numeric_values = []
    for v in valid_values:
        try:
            numeric_values.append(float(v))
        except (ValueError, TypeError):
            continue
    
    if not numeric_values:
        return None, 0.0, {"reason": "no_numeric_values"}
    
    result = np.mean(numeric_values)
    
    # Confidence based on variance (lower variance = higher confidence)
    if len(numeric_values) > 1:
        variance = np.var(numeric_values)
        mean_abs = abs(result) if result != 0 else 1
        confidence = max(0.1, min(1.0, 1.0 - (np.sqrt(variance) / mean_abs)))
    else:
        confidence = 1.0
    
    return result, confidence, {
        "rule": "average",
        "num_values": len(numeric_values),
        "variance": float(np.var(numeric_values)) if len(numeric_values) > 1 else 0.0,
        "range": [float(min(numeric_values)), float(max(numeric_values))]
    }


def median(values: List[Any], **kwargs) -> FusionResult:
    """
    Calculate the median of numeric values.
    
    Parameters
    ----------
    values : List[Any]
        List of values to resolve conflicts from
    **kwargs
        Additional context (ignored)
        
    Returns
    -------
    FusionResult
        Tuple of (resolved_value, confidence, metadata)
    """
    valid_values = _filter_valid_values(values)
    if not valid_values:
        return None, 0.0, {"reason": "no_valid_values"}
    
    # Convert to numeric values
    numeric_values = []
    for v in valid_values:
        try:
            numeric_values.append(float(v))
        except (ValueError, TypeError):
            continue
    
    if not numeric_values:
        return None, 0.0, {"reason": "no_numeric_values"}
    
    result = np.median(numeric_values)
    
    # Confidence based on how close values are to median
    if len(numeric_values) > 1:
        deviations = [abs(v - result) for v in numeric_values]
        mean_deviation = np.mean(deviations)
        max_value = max(abs(v) for v in numeric_values)
        if max_value > 0:
            confidence = max(0.1, 1.0 - (mean_deviation / max_value))
        else:
            confidence = 1.0
    else:
        confidence = 1.0
    
    return result, confidence, {
        "rule": "median",
        "num_values": len(numeric_values),
        "mean_deviation": float(np.mean([abs(v - result) for v in numeric_values])),
        "range": [float(min(numeric_values)), float(max(numeric_values))]
    }


def maximum(values: List[Any], **kwargs) -> FusionResult:
    """
    Select the maximum numeric value.
    
    Parameters
    ----------
    values : List[Any]
        List of values to resolve conflicts from
    **kwargs
        Additional context (ignored)
        
    Returns
    -------
    FusionResult
        Tuple of (resolved_value, confidence, metadata)
    """
    valid_values = _filter_valid_values(values)
    if not valid_values:
        return None, 0.0, {"reason": "no_valid_values"}
    
    # Convert to numeric values
    numeric_values = []
    for v in valid_values:
        try:
            numeric_values.append(float(v))
        except (ValueError, TypeError):
            continue
    
    if not numeric_values:
        return None, 0.0, {"reason": "no_numeric_values"}
    
    result = max(numeric_values)
    
    # Count how many values equal the maximum
    max_count = sum(1 for v in numeric_values if v == result)
    confidence = _calculate_tie_confidence(max_count, max_count, len(numeric_values))
    
    return result, confidence, {
        "rule": "maximum",
        "num_values": len(numeric_values),
        "num_max_values": max_count,
        "range": [float(min(numeric_values)), float(max(numeric_values))]
    }


def minimum(values: List[Any], **kwargs) -> FusionResult:
    """
    Select the minimum numeric value.
    
    Parameters
    ----------
    values : List[Any]
        List of values to resolve conflicts from
    **kwargs
        Additional context (ignored)
        
    Returns
    -------
    FusionResult
        Tuple of (resolved_value, confidence, metadata)
    """
    valid_values = _filter_valid_values(values)
    if not valid_values:
        return None, 0.0, {"reason": "no_valid_values"}
    
    # Convert to numeric values
    numeric_values = []
    for v in valid_values:
        try:
            numeric_values.append(float(v))
        except (ValueError, TypeError):
            continue
    
    if not numeric_values:
        return None, 0.0, {"reason": "no_numeric_values"}
    
    result = min(numeric_values)
    
    # Count how many values equal the minimum
    min_count = sum(1 for v in numeric_values if v == result)
    confidence = _calculate_tie_confidence(min_count, min_count, len(numeric_values))
    
    return result, confidence, {
        "rule": "minimum",
        "num_values": len(numeric_values),
        "num_min_values": min_count,
        "range": [float(min(numeric_values)), float(max(numeric_values))]
    }


def sum_values(values: List[Any], **kwargs) -> FusionResult:
    """
    Calculate the sum of numeric values.
    
    Parameters
    ----------
    values : List[Any]
        List of values to resolve conflicts from
    **kwargs
        Additional context (ignored)
        
    Returns
    -------
    FusionResult
        Tuple of (resolved_value, confidence, metadata)
    """
    valid_values = _filter_valid_values(values)
    if not valid_values:
        return None, 0.0, {"reason": "no_valid_values"}
    
    # Convert to numeric values
    numeric_values = []
    for v in valid_values:
        try:
            numeric_values.append(float(v))
        except (ValueError, TypeError):
            continue
    
    if not numeric_values:
        return None, 0.0, {"reason": "no_numeric_values"}
    
    result = sum(numeric_values)
    
    # Confidence is always high for sum (it's deterministic)
    confidence = 1.0
    
    return result, confidence, {
        "rule": "sum",
        "num_values": len(numeric_values),
        "range": [float(min(numeric_values)), float(max(numeric_values))]
    }


# =============================================================================
# DATE RULES
# =============================================================================

def most_recent(values: List[Any], **kwargs) -> FusionResult:
    """
    Select the most recent date value.
    
    Parameters
    ----------
    values : List[Any]
        List of values to resolve conflicts from
    **kwargs
        Additional context (ignored)
        
    Returns
    -------
    FusionResult
        Tuple of (resolved_value, confidence, metadata)
    """
    valid_values = _filter_valid_values(values)
    if not valid_values:
        return None, 0.0, {"reason": "no_valid_values"}
    
    # Convert to datetime objects
    datetime_values = []
    for v in valid_values:
        try:
            if isinstance(v, datetime):
                datetime_values.append((v, v))
            else:
                dt = pd.to_datetime(v)
                datetime_values.append((v, dt))
        except (ValueError, TypeError):
            continue
    
    if not datetime_values:
        return None, 0.0, {"reason": "no_date_values"}
    
    # Find most recent
    datetime_values.sort(key=lambda x: x[1], reverse=True)  # Most recent first
    winner = datetime_values[0]
    
    # Count ties
    most_recent_dt = winner[1]
    recent_count = sum(1 for _, dt in datetime_values if dt == most_recent_dt)
    confidence = _calculate_tie_confidence(recent_count, recent_count, len(datetime_values))
    
    return winner[0], confidence, {
        "rule": "most_recent",
        "num_values": len(datetime_values),
        "num_recent_values": recent_count,
        "date_range": [str(datetime_values[-1][1]), str(datetime_values[0][1])]
    }


def earliest(values: List[Any], **kwargs) -> FusionResult:
    """
    Select the earliest date value.
    
    Parameters
    ----------
    values : List[Any]
        List of values to resolve conflicts from
    **kwargs
        Additional context (ignored)
        
    Returns
    -------
    FusionResult
        Tuple of (resolved_value, confidence, metadata)
    """
    valid_values = _filter_valid_values(values)
    if not valid_values:
        return None, 0.0, {"reason": "no_valid_values"}
    
    # Convert to datetime objects
    datetime_values = []
    for v in valid_values:
        try:
            if isinstance(v, datetime):
                datetime_values.append((v, v))
            else:
                dt = pd.to_datetime(v)
                datetime_values.append((v, dt))
        except (ValueError, TypeError):
            continue
    
    if not datetime_values:
        return None, 0.0, {"reason": "no_date_values"}
    
    # Find earliest
    datetime_values.sort(key=lambda x: x[1])  # Earliest first
    winner = datetime_values[0]
    
    # Count ties
    earliest_dt = winner[1]
    early_count = sum(1 for _, dt in datetime_values if dt == earliest_dt)
    confidence = _calculate_tie_confidence(early_count, early_count, len(datetime_values))
    
    return winner[0], confidence, {
        "rule": "earliest",
        "num_values": len(datetime_values),
        "num_early_values": early_count,
        "date_range": [str(datetime_values[0][1]), str(datetime_values[-1][1])]
    }


# =============================================================================
# LIST RULES
# =============================================================================

def union(values: List[Any], separator: str = None, **kwargs) -> FusionResult:
    """
    Create union of all list values.
    
    Parameters
    ----------
    values : List[Any]
        List of values to resolve conflicts from
    separator : str, optional
        If provided, split string values by this separator
    **kwargs
        Additional context (ignored)
        
    Returns
    -------
    FusionResult
        Tuple of (resolved_value, confidence, metadata)
    """
    valid_values = _filter_valid_values(values)
    if not valid_values:
        return None, 0.0, {"reason": "no_valid_values"}
    
    # Convert all values to lists
    all_items = set()
    for v in valid_values:
        if isinstance(v, (list, tuple)):
            all_items.update(v)
        elif isinstance(v, str) and separator:
            all_items.update(v.split(separator))
        else:
            all_items.add(v)
    
    result = sorted(list(all_items))  # Sort for consistency
    confidence = 1.0  # Union is deterministic
    
    return result, confidence, {
        "rule": "union",
        "num_sources": len(valid_values),
        "num_unique_items": len(result),
        "total_items": sum(len(str(v).split(separator)) if separator and isinstance(v, str) 
                          else len(v) if isinstance(v, (list, tuple)) else 1 
                          for v in valid_values)
    }


def intersection(values: List[Any], separator: str = None, **kwargs) -> FusionResult:
    """
    Create intersection of all list values.
    
    Parameters
    ----------
    values : List[Any]
        List of values to resolve conflicts from
    separator : str, optional
        If provided, split string values by this separator
    **kwargs
        Additional context (ignored)
        
    Returns
    -------
    FusionResult
        Tuple of (resolved_value, confidence, metadata)
    """
    valid_values = _filter_valid_values(values)
    if not valid_values:
        return None, 0.0, {"reason": "no_valid_values"}
    
    if len(valid_values) == 1:
        # Only one source, return as-is
        v = valid_values[0]
        if isinstance(v, (list, tuple)):
            result = list(v)
        elif isinstance(v, str) and separator:
            result = v.split(separator)
        else:
            result = [v]
        return result, 1.0, {
            "rule": "intersection",
            "num_sources": 1,
            "num_common_items": len(result)
        }
    
    # Convert all values to sets
    sets = []
    for v in valid_values:
        if isinstance(v, (list, tuple)):
            sets.append(set(v))
        elif isinstance(v, str) and separator:
            sets.append(set(v.split(separator)))
        else:
            sets.append({v})
    
    # Find intersection
    result_set = sets[0]
    for s in sets[1:]:
        result_set = result_set.intersection(s)
    
    result = sorted(list(result_set))  # Sort for consistency
    
    # Confidence based on how much of the original data is preserved
    total_unique_items = len(set().union(*sets))
    confidence = len(result) / total_unique_items if total_unique_items > 0 else 0.0
    
    return result, confidence, {
        "rule": "intersection",
        "num_sources": len(valid_values),
        "num_common_items": len(result),
        "num_total_unique_items": total_unique_items
    }


def intersection_k_sources(values: List[Any], k: int = 2, separator: str = None, **kwargs) -> FusionResult:
    """
    Keep items that appear in at least k sources.
    
    Parameters
    ----------
    values : List[Any]
        List of values to resolve conflicts from
    k : int
        Minimum number of sources an item must appear in
    separator : str, optional
        If provided, split string values by this separator
    **kwargs
        Additional context (ignored)
        
    Returns
    -------
    FusionResult
        Tuple of (resolved_value, confidence, metadata)
    """
    valid_values = _filter_valid_values(values)
    if not valid_values:
        return None, 0.0, {"reason": "no_valid_values"}
    
    k = min(k, len(valid_values))  # Can't require more sources than we have
    
    # Count occurrences of each item across sources
    item_counts = Counter()
    for v in valid_values:
        items = set()
        if isinstance(v, (list, tuple)):
            items = set(v)
        elif isinstance(v, str) and separator:
            items = set(v.split(separator))
        else:
            items = {v}
        
        for item in items:
            item_counts[item] += 1
    
    # Keep items that appear in at least k sources
    result = sorted([item for item, count in item_counts.items() if count >= k])
    
    # Confidence based on the threshold
    total_items = len(item_counts)
    confidence = len(result) / total_items if total_items > 0 else 0.0
    
    return result, confidence, {
        "rule": "intersection_k_sources",
        "k": k,
        "num_sources": len(valid_values),
        "num_qualifying_items": len(result),
        "num_total_items": total_items,
        "item_counts": dict(item_counts)
    }


# =============================================================================
# META RULES
# =============================================================================

def voting(values: List[Any], **kwargs) -> FusionResult:
    """
    Select the most frequent value (majority vote).
    
    Parameters
    ----------
    values : List[Any]
        List of values to resolve conflicts from
    **kwargs
        Additional context (ignored)
        
    Returns
    -------
    FusionResult
        Tuple of (resolved_value, confidence, metadata)
    """
    valid_values = _filter_valid_values(values)
    if not valid_values:
        return None, 0.0, {"reason": "no_valid_values"}
    
    # Count votes
    vote_counts = Counter(valid_values)
    most_common = vote_counts.most_common()
    
    winner_value = most_common[0][0]
    winner_count = most_common[0][1]
    
    # Calculate confidence
    if len(most_common) > 1:
        second_count = most_common[1][1]
        confidence = _calculate_tie_confidence(winner_count, second_count, len(valid_values))
    else:
        confidence = 1.0
    
    return winner_value, confidence, {
        "rule": "voting",
        "winner_votes": winner_count,
        "total_votes": len(valid_values),
        "vote_distribution": dict(vote_counts),
        "is_majority": winner_count > len(valid_values) / 2
    }


def favour_sources(values: List[Any], source_preferences: List[str] = None, 
                  source_column: str = "_source", **kwargs) -> FusionResult:
    """
    Select value from preferred sources.
    
    Parameters
    ----------
    values : List[Any]
        List of values to resolve conflicts from
    source_preferences : List[str], optional
        Ordered list of preferred source names
    source_column : str
        Column name that contains source information
    **kwargs
        Additional context containing source information
        
    Returns
    -------
    FusionResult
        Tuple of (resolved_value, confidence, metadata)
    """
    valid_values = _filter_valid_values(values)
    if not valid_values:
        return None, 0.0, {"reason": "no_valid_values"}
    
    # If no source preferences, just return first value
    if not source_preferences:
        return valid_values[0], 0.5, {
            "rule": "favour_sources",
            "reason": "no_source_preferences"
        }
    
    # Try to get source information from context
    sources = kwargs.get('sources', [])
    if len(sources) != len(values):
        # Fallback to first value
        return valid_values[0], 0.5, {
            "rule": "favour_sources",
            "reason": "source_info_unavailable"
        }
    
    # Find value from highest preference source
    for preferred_source in source_preferences:
        for value, source in zip(values, sources):
            if source == preferred_source and pd.notna(value):
                preference_rank = source_preferences.index(preferred_source)
                confidence = 1.0 - (preference_rank / len(source_preferences)) * 0.5
                return value, confidence, {
                    "rule": "favour_sources",
                    "selected_source": source,
                    "preference_rank": preference_rank,
                    "available_sources": sources
                }
    
    # No preferred source found, return first valid value
    return valid_values[0], 0.3, {
        "rule": "favour_sources",
        "reason": "no_preferred_source_available",
        "available_sources": sources
    }


def random_value(values: List[Any], seed: int = None, **kwargs) -> FusionResult:
    """
    Select a random value.
    
    Parameters
    ----------
    values : List[Any]
        List of values to resolve conflicts from
    seed : int, optional
        Random seed for reproducibility
    **kwargs
        Additional context (ignored)
        
    Returns
    -------
    FusionResult
        Tuple of (resolved_value, confidence, metadata)
    """
    valid_values = _filter_valid_values(values)
    if not valid_values:
        return None, 0.0, {"reason": "no_valid_values"}
    
    if seed is not None:
        random.seed(seed)
    
    result = random.choice(valid_values)
    confidence = 1.0 / len(valid_values)  # Lower confidence with more options
    
    return result, confidence, {
        "rule": "random_value",
        "num_candidates": len(valid_values),
        "seed": seed
    }


def weighted_voting(values: List[Any], weights: List[float] = None, **kwargs) -> FusionResult:
    """
    Select value using weighted voting.
    
    Parameters
    ----------
    values : List[Any]
        List of values to resolve conflicts from
    weights : List[float], optional
        Weights for each value. If None, uses uniform weights.
    **kwargs
        Additional context (ignored)
        
    Returns
    -------
    FusionResult
        Tuple of (resolved_value, confidence, metadata)
    """
    valid_values = _filter_valid_values(values)
    if not valid_values:
        return None, 0.0, {"reason": "no_valid_values"}
    
    # Use uniform weights if not provided
    if weights is None:
        weights = [1.0] * len(values)
    
    # Align weights with valid values
    valid_weights = []
    for i, v in enumerate(values):
        if v in valid_values:
            valid_weights.append(weights[i] if i < len(weights) else 1.0)
    
    # Calculate weighted votes
    weighted_counts = {}
    for value, weight in zip(valid_values, valid_weights):
        if value in weighted_counts:
            weighted_counts[value] += weight
        else:
            weighted_counts[value] = weight
    
    # Find winner
    winner = max(weighted_counts.items(), key=lambda x: x[1])
    winner_value = winner[0]
    winner_weight = winner[1]
    
    total_weight = sum(weighted_counts.values())
    confidence = winner_weight / total_weight if total_weight > 0 else 0.0
    
    return winner_value, confidence, {
        "rule": "weighted_voting",
        "winner_weight": winner_weight,
        "total_weight": total_weight,
        "weight_distribution": weighted_counts
    }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_function_resolver(func, **default_kwargs):
    """
    Create a ConflictResolutionFunction wrapper for a fusion rule function.
    
    This allows function-based rules to be used with the existing class-based API.
    """
    from .base import ConflictResolutionFunction, FusionContext, FusionResult as ClassFusionResult
    
    class FunctionWrapper(ConflictResolutionFunction):
        def __init__(self, func, default_kwargs):
            self.func = func
            self.default_kwargs = default_kwargs
            
        @property
        def name(self) -> str:
            return self.func.__name__
            
        def resolve(self, values: List[Any], context: FusionContext) -> ClassFusionResult:
            # Call the function with merged kwargs
            kwargs = {**self.default_kwargs, **context.__dict__}
            resolved_value, confidence, metadata = self.func(values, **kwargs)
            
            return ClassFusionResult(
                value=resolved_value,
                confidence=confidence,
                rule_used=self.name,
                metadata=metadata
            )
    
    return FunctionWrapper(func, default_kwargs)


# =============================================================================
# COMMON RULE PRESETS
# =============================================================================

# String rules
LONGEST = longest_string
SHORTEST = shortest_string
MOST_COMPLETE = most_complete

# Numeric rules  
AVG = AVERAGE = average
MEDIAN = median
MAX = MAXIMUM = maximum
MIN = MINIMUM = minimum
SUM = sum_values

# Date rules
LATEST = MOST_RECENT = most_recent
EARLIEST = earliest

# List rules
UNION = union
INTERSECTION = intersection
INTERSECTION_K = intersection_k_sources

# Meta rules
VOTE = VOTING = voting
FAVOUR = favour_sources
RANDOM = random_value
WEIGHTED_VOTE = weighted_voting