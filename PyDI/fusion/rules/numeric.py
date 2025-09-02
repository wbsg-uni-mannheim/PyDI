"""
Numeric conflict resolution functions.
"""

from typing import Any, List
import statistics
import pandas as pd

from ..base import ConflictResolutionFunction, FusionContext, FusionResult


class Average(ConflictResolutionFunction):
    """Resolve conflicts by taking the arithmetic mean of numeric values."""
    
    @property
    def name(self) -> str:
        return "average"
    
    def resolve(self, values: List[Any], context: FusionContext) -> FusionResult:
        """Calculate the arithmetic mean of numeric values."""
        if not values:
            return FusionResult(
                value=None,
                confidence=0.0,
                rule_used=self.name,
                metadata={"reason": "no_values"}
            )
        
        # Convert to numeric values
        numeric_values = []
        for v in values:
            if v is not None and pd.notna(v):
                try:
                    numeric_values.append(float(v))
                except (ValueError, TypeError):
                    continue
        
        if not numeric_values:
            return FusionResult(
                value=None,
                confidence=0.0,
                rule_used=self.name,
                metadata={"reason": "no_numeric_values"}
            )
        
        mean_value = statistics.mean(numeric_values)
        
        # Calculate confidence based on variance
        if len(numeric_values) > 1:
            variance = statistics.variance(numeric_values)
            # Lower variance -> higher confidence
            confidence = max(0.1, min(1.0, 1.0 / (1.0 + variance)))
        else:
            confidence = 1.0
        
        return FusionResult(
            value=mean_value,
            confidence=confidence,
            rule_used=self.name,
            metadata={
                "num_values": len(numeric_values),
                "variance": statistics.variance(numeric_values) if len(numeric_values) > 1 else 0.0,
                "range": max(numeric_values) - min(numeric_values) if numeric_values else 0.0
            }
        )


class Median(ConflictResolutionFunction):
    """Resolve conflicts by taking the median of numeric values."""
    
    @property
    def name(self) -> str:
        return "median"
    
    def resolve(self, values: List[Any], context: FusionContext) -> FusionResult:
        """Calculate the median of numeric values."""
        if not values:
            return FusionResult(
                value=None,
                confidence=0.0,
                rule_used=self.name,
                metadata={"reason": "no_values"}
            )
        
        # Convert to numeric values
        numeric_values = []
        for v in values:
            if v is not None and pd.notna(v):
                try:
                    numeric_values.append(float(v))
                except (ValueError, TypeError):
                    continue
        
        if not numeric_values:
            return FusionResult(
                value=None,
                confidence=0.0,
                rule_used=self.name,
                metadata={"reason": "no_numeric_values"}
            )
        
        median_value = statistics.median(numeric_values)
        
        # Calculate confidence
        if len(numeric_values) > 2:
            # For median, confidence is higher when values cluster around median
            deviations = [abs(v - median_value) for v in numeric_values]
            mean_deviation = statistics.mean(deviations)
            confidence = max(0.1, min(1.0, 1.0 / (1.0 + mean_deviation)))
        else:
            confidence = 0.8  # Reasonable confidence for small samples
        
        return FusionResult(
            value=median_value,
            confidence=confidence,
            rule_used=self.name,
            metadata={
                "num_values": len(numeric_values),
                "mean_deviation": statistics.mean([abs(v - median_value) for v in numeric_values]),
                "range": max(numeric_values) - min(numeric_values) if numeric_values else 0.0
            }
        )


class Maximum(ConflictResolutionFunction):
    """Resolve conflicts by choosing the maximum numeric value."""
    
    @property
    def name(self) -> str:
        return "maximum"
    
    def resolve(self, values: List[Any], context: FusionContext) -> FusionResult:
        """Choose the maximum numeric value."""
        if not values:
            return FusionResult(
                value=None,
                confidence=0.0,
                rule_used=self.name,
                metadata={"reason": "no_values"}
            )
        
        # Convert to numeric values
        numeric_values = []
        for v in values:
            if v is not None and pd.notna(v):
                try:
                    numeric_values.append(float(v))
                except (ValueError, TypeError):
                    continue
        
        if not numeric_values:
            return FusionResult(
                value=None,
                confidence=0.0,
                rule_used=self.name,
                metadata={"reason": "no_numeric_values"}
            )
        
        max_value = max(numeric_values)
        
        # Calculate confidence based on how much larger the max is
        if len(numeric_values) > 1:
            sorted_values = sorted(numeric_values, reverse=True)
            if len(sorted_values) > 1 and sorted_values[0] > sorted_values[1]:
                gap = sorted_values[0] - sorted_values[1]
                range_val = max(numeric_values) - min(numeric_values)
                if range_val > 0:
                    confidence = min(1.0, 0.5 + gap / range_val * 0.5)
                else:
                    confidence = 0.5  # All values are the same
            else:
                confidence = 0.5  # Tie for maximum
        else:
            confidence = 1.0
        
        return FusionResult(
            value=max_value,
            confidence=confidence,
            rule_used=self.name,
            metadata={
                "num_values": len(numeric_values),
                "range": max(numeric_values) - min(numeric_values) if numeric_values else 0.0
            }
        )


class Minimum(ConflictResolutionFunction):
    """Resolve conflicts by choosing the minimum numeric value."""
    
    @property
    def name(self) -> str:
        return "minimum"
    
    def resolve(self, values: List[Any], context: FusionContext) -> FusionResult:
        """Choose the minimum numeric value."""
        if not values:
            return FusionResult(
                value=None,
                confidence=0.0,
                rule_used=self.name,
                metadata={"reason": "no_values"}
            )
        
        # Convert to numeric values
        numeric_values = []
        for v in values:
            if v is not None and pd.notna(v):
                try:
                    numeric_values.append(float(v))
                except (ValueError, TypeError):
                    continue
        
        if not numeric_values:
            return FusionResult(
                value=None,
                confidence=0.0,
                rule_used=self.name,
                metadata={"reason": "no_numeric_values"}
            )
        
        min_value = min(numeric_values)
        
        # Calculate confidence based on how much smaller the min is
        if len(numeric_values) > 1:
            sorted_values = sorted(numeric_values)
            if len(sorted_values) > 1 and sorted_values[0] < sorted_values[1]:
                gap = sorted_values[1] - sorted_values[0]
                range_val = max(numeric_values) - min(numeric_values)
                if range_val > 0:
                    confidence = min(1.0, 0.5 + gap / range_val * 0.5)
                else:
                    confidence = 0.5  # All values are the same
            else:
                confidence = 0.5  # Tie for minimum
        else:
            confidence = 1.0
        
        return FusionResult(
            value=min_value,
            confidence=confidence,
            rule_used=self.name,
            metadata={
                "num_values": len(numeric_values),
                "range": max(numeric_values) - min(numeric_values) if numeric_values else 0.0
            }
        )