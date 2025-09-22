"""String-based conflict resolution functions."""

from typing import Any, List, Tuple, Dict

from .utils import _filter_valid_values


# Type alias for fusion rule result
FusionResult = Tuple[Any, float, Dict[str, Any]]


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
    
    metadata = {
        "rule": "longest_string",
        "selected_length": winner_length,
        "num_candidates": len(string_values),
        "all_lengths": [x[1] for x in string_values]
    }
    
    return winner[0], confidence, metadata


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
        total_values = len(completeness)
        if winner_count > second_count:
            confidence = 0.5 + (winner_count - second_count) / total_values * 0.5
        else:
            confidence = 0.5  # Perfect tie
    else:
        confidence = 1.0
    
    return winner[0], confidence, {
        "rule": "most_complete",
        "selected_completeness": winner_count,
        "num_candidates": len(completeness),
        "all_completeness": [x[1] for x in completeness]
    }
