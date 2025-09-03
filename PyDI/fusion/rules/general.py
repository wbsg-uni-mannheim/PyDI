"""
General-purpose conflict resolution functions.
"""

from typing import Any, List, Dict
from collections import Counter
import random
import pandas as pd

from ..base import ConflictResolutionFunction, FusionContext, FusionResult


class Voting(ConflictResolutionFunction):
    """Resolve conflicts by majority voting."""
    
    @property
    def name(self) -> str:
        return "voting"
    
    def resolve(self, values: List[Any], context: FusionContext) -> FusionResult:
        """Choose the most frequent value."""
        if not values:
            return FusionResult(
                value=None,
                confidence=0.0,
                rule_used=self.name,
                metadata={"reason": "no_values"}
            )
        
        # Filter out null values and count occurrences
        valid_values = [v for v in values if v is not None and pd.notna(v)]
        
        if not valid_values:
            return FusionResult(
                value=None,
                confidence=0.0,
                rule_used=self.name,
                metadata={"reason": "all_null"}
            )
        
        # Count votes
        vote_counts = Counter(valid_values)
        most_common = vote_counts.most_common(1)[0]
        winning_value, winning_count = most_common
        
        # Calculate confidence based on vote margin
        total_votes = len(valid_values)
        if total_votes == 1:
            confidence = 1.0
        else:
            # Get second-place count
            if len(vote_counts) > 1:
                second_count = vote_counts.most_common(2)[1][1]
                margin = winning_count - second_count
                confidence = min(1.0, 0.5 + (margin / total_votes) * 0.5)
            else:
                # Unanimous vote
                confidence = 1.0
        
        return FusionResult(
            value=winning_value,
            confidence=confidence,
            rule_used=self.name,
            metadata={
                "total_votes": total_votes,
                "winning_votes": winning_count,
                "vote_counts": dict(vote_counts),
                "margin": winning_count / total_votes if total_votes > 0 else 0.0
            }
        )


class FavourSources(ConflictResolutionFunction):
    """Resolve conflicts by favoring specific sources in priority order."""
    
    def __init__(self, source_priority: List[str], trust_scores: Dict[str, float] = None):
        """Initialize with source priority and optional trust scores.
        
        Parameters
        ----------
        source_priority : List[str]
            List of source names in priority order (first = highest priority).
        trust_scores : Dict[str, float], optional
            Trust scores for each source (0.0 to 1.0).
        """
        self.source_priority = source_priority
        self.trust_scores = trust_scores or {}
    
    @property
    def name(self) -> str:
        return "favour_sources"
    
    def resolve(self, values: List[Any], context: FusionContext) -> FusionResult:
        """Choose value from highest priority source."""
        if not values:
            return FusionResult(
                value=None,
                confidence=0.0,
                rule_used=self.name,
                metadata={"reason": "no_values"}
            )
        
        # We need source information from context
        source_datasets = getattr(context, 'source_datasets', {})
        
        # Map values to sources - this is tricky without record-level tracking
        # For now, we'll assume sources are provided in metadata or use simple heuristics
        
        # Find the best source among available ones
        best_value = None
        best_source = None
        best_priority = float('inf')
        best_trust = 0.0
        
        # If we have only the values, choose the first non-null one
        # In a real implementation, we'd need better source tracking
        valid_values = [v for v in values if v is not None and pd.notna(v)]
        
        if not valid_values:
            return FusionResult(
                value=None,
                confidence=0.0,
                rule_used=self.name,
                metadata={"reason": "all_null"}
            )
        
        # Use first value as fallback (would be improved with proper source tracking)
        chosen_value = valid_values[0]
        
        # Calculate confidence based on trust score and priority
        confidence = 0.7  # Default confidence when source tracking is limited
        
        return FusionResult(
            value=chosen_value,
            confidence=confidence,
            rule_used=self.name,
            metadata={
                "source_priority": self.source_priority,
                "available_values": len(valid_values),
                "chosen_index": 0  # Would be actual source index in full implementation
            }
        )


class RandomValue(ConflictResolutionFunction):
    """Resolve conflicts by randomly choosing a value."""
    
    def __init__(self, seed: int = None):
        """Initialize with optional random seed for reproducibility.
        
        Parameters
        ----------
        seed : int, optional
            Random seed for reproducible results.
        """
        self.seed = seed
        if seed is not None:
            random.seed(seed)
    
    @property
    def name(self) -> str:
        return "random_value"
    
    def resolve(self, values: List[Any], context: FusionContext) -> FusionResult:
        """Randomly choose a value."""
        if not values:
            return FusionResult(
                value=None,
                confidence=0.0,
                rule_used=self.name,
                metadata={"reason": "no_values"}
            )
        
        # Filter out null values
        valid_values = [v for v in values if v is not None and pd.notna(v)]
        
        if not valid_values:
            return FusionResult(
                value=None,
                confidence=0.0,
                rule_used=self.name,
                metadata={"reason": "all_null"}
            )
        
        # Choose randomly
        chosen_value = random.choice(valid_values)
        
        # Confidence is inversely related to number of choices
        confidence = 1.0 / len(valid_values) if valid_values else 0.0
        confidence = max(0.1, confidence)  # Minimum confidence
        
        return FusionResult(
            value=chosen_value,
            confidence=confidence,
            rule_used=self.name,
            metadata={
                "num_choices": len(valid_values),
                "total_values": len(values),
                "seed": self.seed
            }
        )


class WeightedVoting(ConflictResolutionFunction):
    """Resolve conflicts by weighted voting based on source trust scores."""
    
    def __init__(self, trust_scores: Dict[str, float]):
        """Initialize with trust scores for sources.
        
        Parameters
        ----------
        trust_scores : Dict[str, float]
            Trust scores for each source (0.0 to 1.0).
        """
        self.trust_scores = trust_scores
    
    @property
    def name(self) -> str:
        return "weighted_voting"
    
    def resolve(self, values: List[Any], context: FusionContext) -> FusionResult:
        """Choose value using weighted voting based on trust scores."""
        if not values:
            return FusionResult(
                value=None,
                confidence=0.0,
                rule_used=self.name,
                metadata={"reason": "no_values"}
            )
        
        # Filter out null values
        valid_values = [v for v in values if v is not None and pd.notna(v)]
        
        if not valid_values:
            return FusionResult(
                value=None,
                confidence=0.0,
                rule_used=self.name,
                metadata={"reason": "all_null"}
            )
        
        # Count weighted votes
        # Note: In a full implementation, we'd need proper source tracking
        # For now, we'll simulate with equal weights
        weighted_counts = {}
        total_weight = 0.0
        
        for value in valid_values:
            # In real implementation, would look up source for this value
            weight = 1.0  # Default weight when source tracking is limited
            weighted_counts[value] = weighted_counts.get(value, 0.0) + weight
            total_weight += weight
        
        # Find winner
        if not weighted_counts:
            return FusionResult(
                value=None,
                confidence=0.0,
                rule_used=self.name,
                metadata={"reason": "no_weighted_values"}
            )
        
        winning_value = max(weighted_counts.items(), key=lambda x: x[1])[0]
        winning_weight = weighted_counts[winning_value]
        
        # Calculate confidence
        if total_weight > 0:
            confidence = winning_weight / total_weight
        else:
            confidence = 0.0
        
        return FusionResult(
            value=winning_value,
            confidence=confidence,
            rule_used=self.name,
            metadata={
                "weighted_counts": weighted_counts,
                "total_weight": total_weight,
                "winning_weight": winning_weight,
                "trust_scores_available": len(self.trust_scores)
            }
        )