"""Shared utilities for conflict resolution functions."""

from typing import Any, List

from ..base import _is_valid_value


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

