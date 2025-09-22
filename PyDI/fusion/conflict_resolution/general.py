"""General conflict resolution functions."""

from typing import Any, List, Tuple, Dict, Optional
from collections import Counter
import random

from ..base import _is_valid_value
from .utils import _filter_valid_values, _calculate_tie_confidence


# Type alias for fusion rule result
FusionResult = Tuple[Any, float, Dict[str, Any]]


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


def prefer_higher_trust(
    values: List[Any],
    *,
    sources: List[str] = None,
    source_datasets: Dict[str, str] = None,
    trust_map: Optional[Dict[str, float]] = None,
    default_trust: float = 1.0,
    trust_key: str = "trust",
    tie_breaker: str = "first",
    **kwargs,
) -> FusionResult:
    """
    Prefer the value coming from the dataset with the highest trust level.

    Parameters
    ----------
    values : List[Any]
        Candidate values for the attribute.
    sources : List[str]
        Record IDs corresponding 1:1 to ``values`` (provided by the engine).
    source_datasets : Dict[str, str]
        Map from record ID to dataset name (provided by the engine).
    trust_map : Dict[str, float], optional
        Map from dataset name to trust score/level. Higher means more trusted.
    default_trust : float, default 1.0
        Trust to assume for datasets not in ``trust_map``.
    trust_key : str, default "trust"
        Key name to use for the trust value in the emitted metadata.
    tie_breaker : str, default "first"
        How to break ties when multiple datasets share the highest trust.
        Options: "first" (preserve input order).

    Returns
    -------
    FusionResult
        Tuple (value, confidence, metadata).
    """
    # Allow engine-provided trust_map via kwargs if not explicitly passed
    if trust_map is None:
        trust_map = kwargs.get('trust_map')

    valid_values = []
    meta_rows = []

    # Validate alignment info
    if sources is None or source_datasets is None or len(sources) != len(values):
        # Fallback: take first valid
        vv = _filter_valid_values(values)
        if not vv:
            return None, 0.0, {"reason": "no_valid_values"}
        return vv[0], 0.5, {"rule": "prefer_higher_trust", "reason": "alignment_missing"}

    # Build aligned rows with dataset and trust
    for val, rid in zip(values, sources):
        if not _is_valid_value(val):
            continue
        ds = source_datasets.get(rid, "unknown")
        trust = default_trust
        if trust_map and ds in trust_map:
            trust = trust_map[ds]
        meta_rows.append({"value": val, "record_id": rid, "dataset": ds, trust_key: float(trust)})
        valid_values.append(val)

    if not meta_rows:
        return None, 0.0, {"reason": "no_valid_values"}

    # Find highest trust
    max_trust = max(r[trust_key] for r in meta_rows)
    top_rows = [r for r in meta_rows if r[trust_key] == max_trust]

    # Choose among top trusted candidates
    chosen = top_rows[0]
    if tie_breaker != "first" and len(top_rows) > 1:
        # Future: support other tie-breakers like longest_string, voting, most_recent
        pass

    # Confidence: unique top -> 1.0, else diluted by tie size
    tie_count = len(top_rows)
    confidence = 1.0 if tie_count == 1 else 1.0 / tie_count

    # Prepare metadata
    trust_distribution = {}
    for r in meta_rows:
        ds = r["dataset"]
        t = r[trust_key]
        trust_distribution[ds] = max(trust_distribution.get(ds, t), t)

    metadata = {
        "rule": "prefer_higher_trust",
        "selected_dataset": chosen["dataset"],
        "selected_record_id": chosen["record_id"],
        "selected_trust_field": trust_key,
        f"selected_{trust_key}": chosen[trust_key],
        "selected_trust": chosen[trust_key],
        "max_trust": max_trust,
        f"max_{trust_key}": max_trust,
        "tie_count": tie_count,
        "trust_distribution": trust_distribution,
        f"{trust_key}_distribution": trust_distribution,
        "available_sources": [r["dataset"] for r in meta_rows],
    }

    return chosen["value"], confidence, metadata
