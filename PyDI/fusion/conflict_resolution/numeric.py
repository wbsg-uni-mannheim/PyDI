"""Numeric conflict resolution functions."""

from typing import Any, List, Tuple, Dict
import numpy as np

from .utils import _filter_valid_values, _calculate_tie_confidence


# Type alias for fusion rule result
FusionResult = Tuple[Any, float, Dict[str, Any]]


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
    
    metadata = {
        "rule": "average",
        "num_values": len(numeric_values),
        "variance": float(np.var(numeric_values)) if len(numeric_values) > 1 else 0.0,
        "range": [float(min(numeric_values)), float(max(numeric_values))]
    }
    
    return result, confidence, metadata


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
