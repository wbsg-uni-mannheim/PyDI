"""List-based conflict resolution functions."""

from typing import Any, List, Tuple, Dict, Union, Set
from collections import Counter

from .utils import _filter_valid_values


# Type alias for fusion rule result
FusionResult = Tuple[Any, float, Dict[str, Any]]


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
            split_items = v.split(separator)
            all_items.update(split_items)
        else:
            all_items.add(v)

    result = sorted(list(all_items))  # Sort for consistency
    confidence = 1.0  # Union is deterministic
    
    metadata = {
        "rule": "union",
        "num_sources": len(valid_values),
        "num_unique_items": len(result),
        "total_items": sum(len(str(v).split(separator)) if separator and isinstance(v, str) 
                          else len(v) if isinstance(v, (list, tuple)) else 1 
                          for v in valid_values)
    }
    
    return result, confidence, metadata


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
