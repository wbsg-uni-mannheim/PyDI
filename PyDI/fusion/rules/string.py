"""
String-based conflict resolution functions.
"""

from typing import Any, List
import pandas as pd

from ..base import ConflictResolutionFunction, FusionContext, FusionResult


class LongestString(ConflictResolutionFunction):
    """Resolve conflicts by choosing the longest string value."""
    
    @property
    def name(self) -> str:
        return "longest_string"
    
    def resolve(self, values: List[Any], context: FusionContext) -> FusionResult:
        """Choose the longest string value."""
        if not values:
            return FusionResult(
                value=None,
                confidence=0.0,
                rule_used=self.name,
                metadata={"reason": "no_values"}
            )
        
        # Convert to strings and find longest
        string_values = []
        for v in values:
            if v is not None and pd.notna(v):
                string_values.append((str(v), len(str(v))))
        
        if not string_values:
            return FusionResult(
                value=None,
                confidence=0.0,
                rule_used=self.name,
                metadata={"reason": "all_null"}
            )
        
        # Sort by length (descending) then by value for consistency
        string_values.sort(key=lambda x: (-x[1], x[0]))
        longest_value = string_values[0][0]
        longest_length = string_values[0][1]
        
        # Calculate confidence based on how much longer the winner is
        if len(string_values) > 1:
            second_length = string_values[1][1]
            if longest_length > second_length:
                confidence = min(1.0, 0.5 + (longest_length - second_length) / longest_length * 0.5)
            else:
                confidence = 0.5  # Tie
        else:
            confidence = 1.0  # Only one value
        
        return FusionResult(
            value=longest_value,
            confidence=confidence,
            rule_used=self.name,
            metadata={
                "selected_length": longest_length,
                "num_candidates": len(string_values),
                "all_lengths": [x[1] for x in string_values]
            }
        )


class ShortestString(ConflictResolutionFunction):
    """Resolve conflicts by choosing the shortest string value."""
    
    @property
    def name(self) -> str:
        return "shortest_string"
    
    def resolve(self, values: List[Any], context: FusionContext) -> FusionResult:
        """Choose the shortest string value."""
        if not values:
            return FusionResult(
                value=None,
                confidence=0.0,
                rule_used=self.name,
                metadata={"reason": "no_values"}
            )
        
        # Convert to strings and find shortest
        string_values = []
        for v in values:
            if v is not None and pd.notna(v):
                string_values.append((str(v), len(str(v))))
        
        if not string_values:
            return FusionResult(
                value=None,
                confidence=0.0,
                rule_used=self.name,
                metadata={"reason": "all_null"}
            )
        
        # Sort by length (ascending) then by value for consistency
        string_values.sort(key=lambda x: (x[1], x[0]))
        shortest_value = string_values[0][0]
        shortest_length = string_values[0][1]
        
        # Calculate confidence
        if len(string_values) > 1:
            second_length = string_values[1][1]
            if shortest_length < second_length:
                confidence = min(1.0, 0.5 + (second_length - shortest_length) / second_length * 0.5)
            else:
                confidence = 0.5  # Tie
        else:
            confidence = 1.0  # Only one value
        
        return FusionResult(
            value=shortest_value,
            confidence=confidence,
            rule_used=self.name,
            metadata={
                "selected_length": shortest_length,
                "num_candidates": len(string_values),
                "all_lengths": [x[1] for x in string_values]
            }
        )