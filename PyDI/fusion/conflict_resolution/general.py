"""General conflict resolution functions."""

from typing import Any, List, Tuple, Dict
import pandas as pd
from collections import Counter
import random

from ..base import _is_valid_value


# Type alias for fusion rule result
FusionResult = Tuple[Any, float, Dict[str, Any]]


def _filter_valid_values(values: List[Any]) -> List[Any]:
    """Filter out None, NaN, and empty list values."""
    return [v for v in values if _is_valid_value(v)]


def _calculate_tie_confidence(winner_count: int, second_count: int, total_values: int) -> float:
    """Calculate confidence when there might be ties."""
    if total_values <= 1:
        return 1.0
    if winner_count > second_count:
        return 0.5 + (winner_count - second_count) / total_values * 0.5
    else:
        return 0.5  # Perfect tie


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
    import logging
    logger = logging.getLogger(__name__)
    
    logger.debug(f"voting: input values={repr(values)}")
    valid_values = _filter_valid_values(values)
    logger.debug(f"voting: valid values after filtering={repr(valid_values)}")
    
    if not valid_values:
        logger.debug("voting: no valid values, returning None")
        return None, 0.0, {"reason": "no_valid_values"}
    
    # Count votes
    vote_counts = Counter(valid_values)
    most_common = vote_counts.most_common()
    logger.debug(f"voting: vote counts={dict(vote_counts)}")
    
    winner_value = most_common[0][0]
    winner_count = most_common[0][1]
    
    # Calculate confidence
    if len(most_common) > 1:
        second_count = most_common[1][1]
        confidence = _calculate_tie_confidence(winner_count, second_count, len(valid_values))
        logger.debug(f"voting: winner={repr(winner_value)} ({winner_count} votes), runner-up has {second_count} votes, confidence={confidence:.3f}")
    else:
        confidence = 1.0
        logger.debug(f"voting: unanimous winner={repr(winner_value)} ({winner_count} votes), confidence={confidence:.3f}")
    
    metadata = {
        "rule": "voting",
        "winner_votes": winner_count,
        "total_votes": len(valid_values),
        "vote_distribution": dict(vote_counts),
        "is_majority": winner_count > len(valid_values) / 2
    }
    
    return winner_value, confidence, metadata


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
    import logging
    logger = logging.getLogger(__name__)
    
    logger.debug(f"favour_sources: input values={repr(values)}, preferences={source_preferences}")
    valid_values = _filter_valid_values(values)
    logger.debug(f"favour_sources: valid values after filtering={repr(valid_values)}")
    
    if not valid_values:
        logger.debug("favour_sources: no valid values, returning None")
        return None, 0.0, {"reason": "no_valid_values"}
    
    # If no source preferences, just return first value
    if not source_preferences:
        logger.debug(f"favour_sources: no source preferences, using first value={repr(valid_values[0])}")
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
            if source == preferred_source and _is_valid_value(value):
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