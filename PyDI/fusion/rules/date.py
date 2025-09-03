"""
Date/time conflict resolution functions.
"""

from typing import Any, List
import pandas as pd

from ..base import ConflictResolutionFunction, FusionContext, FusionResult


class MostRecent(ConflictResolutionFunction):
    """Resolve conflicts by choosing the most recent date/time value."""
    
    @property
    def name(self) -> str:
        return "most_recent"
    
    def resolve(self, values: List[Any], context: FusionContext) -> FusionResult:
        """Choose the most recent date/time value."""
        if not values:
            return FusionResult(
                value=None,
                confidence=0.0,
                rule_used=self.name,
                metadata={"reason": "no_values"}
            )
        
        # Convert to datetime values
        datetime_values = []
        for v in values:
            if v is not None and pd.notna(v):
                try:
                    dt_val = pd.to_datetime(v)
                    datetime_values.append(dt_val)
                except (ValueError, TypeError):
                    continue
        
        if not datetime_values:
            return FusionResult(
                value=None,
                confidence=0.0,
                rule_used=self.name,
                metadata={"reason": "no_datetime_values"}
            )
        
        most_recent = max(datetime_values)
        
        # Calculate confidence based on time gap
        if len(datetime_values) > 1:
            sorted_values = sorted(datetime_values, reverse=True)
            if len(sorted_values) > 1:
                time_gap = (sorted_values[0] - sorted_values[1]).total_seconds()
                # Higher confidence for larger gaps
                if time_gap > 86400:  # More than 1 day
                    confidence = 0.9
                elif time_gap > 3600:  # More than 1 hour
                    confidence = 0.8
                elif time_gap > 60:   # More than 1 minute
                    confidence = 0.7
                else:
                    confidence = 0.5  # Very close times
            else:
                confidence = 0.5
        else:
            confidence = 1.0
        
        return FusionResult(
            value=most_recent,
            confidence=confidence,
            rule_used=self.name,
            metadata={
                "num_values": len(datetime_values),
                "time_range_seconds": (max(datetime_values) - min(datetime_values)).total_seconds() if len(datetime_values) > 1 else 0
            }
        )


class Earliest(ConflictResolutionFunction):
    """Resolve conflicts by choosing the earliest date/time value."""
    
    @property
    def name(self) -> str:
        return "earliest"
    
    def resolve(self, values: List[Any], context: FusionContext) -> FusionResult:
        """Choose the earliest date/time value."""
        if not values:
            return FusionResult(
                value=None,
                confidence=0.0,
                rule_used=self.name,
                metadata={"reason": "no_values"}
            )
        
        # Convert to datetime values
        datetime_values = []
        for v in values:
            if v is not None and pd.notna(v):
                try:
                    dt_val = pd.to_datetime(v)
                    datetime_values.append(dt_val)
                except (ValueError, TypeError):
                    continue
        
        if not datetime_values:
            return FusionResult(
                value=None,
                confidence=0.0,
                rule_used=self.name,
                metadata={"reason": "no_datetime_values"}
            )
        
        earliest = min(datetime_values)
        
        # Calculate confidence based on time gap
        if len(datetime_values) > 1:
            sorted_values = sorted(datetime_values)
            if len(sorted_values) > 1:
                time_gap = (sorted_values[1] - sorted_values[0]).total_seconds()
                # Higher confidence for larger gaps
                if time_gap > 86400:  # More than 1 day
                    confidence = 0.9
                elif time_gap > 3600:  # More than 1 hour
                    confidence = 0.8
                elif time_gap > 60:   # More than 1 minute
                    confidence = 0.7
                else:
                    confidence = 0.5  # Very close times
            else:
                confidence = 0.5
        else:
            confidence = 1.0
        
        return FusionResult(
            value=earliest,
            confidence=confidence,
            rule_used=self.name,
            metadata={
                "num_values": len(datetime_values),
                "time_range_seconds": (max(datetime_values) - min(datetime_values)).total_seconds() if len(datetime_values) > 1 else 0
            }
        )