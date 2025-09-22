"""Date-based conflict resolution functions."""

from datetime import datetime
from typing import Any, List, Tuple, Dict
import pandas as pd

from ..base import _is_valid_value
from .utils import _filter_valid_values, _calculate_tie_confidence


# Type alias for fusion rule result
FusionResult = Tuple[Any, float, Dict[str, Any]]


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
