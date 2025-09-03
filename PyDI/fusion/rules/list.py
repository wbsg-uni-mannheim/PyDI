"""
List-based conflict resolution functions.
"""

from typing import Any, List, Set
import pandas as pd

from ..base import ConflictResolutionFunction, FusionContext, FusionResult


class Union(ConflictResolutionFunction):
    """Resolve conflicts by taking the union of all list values."""
    
    @property
    def name(self) -> str:
        return "union"
    
    def resolve(self, values: List[Any], context: FusionContext) -> FusionResult:
        """Take the union of all list values."""
        if not values:
            return FusionResult(
                value=[],
                confidence=0.0,
                rule_used=self.name,
                metadata={"reason": "no_values"}
            )
        
        # Convert values to lists and collect all items
        all_items = set()
        valid_lists = 0
        
        for v in values:
            if v is not None and pd.notna(v):
                try:
                    if isinstance(v, (list, tuple, set)):
                        all_items.update(v)
                        valid_lists += 1
                    elif isinstance(v, str) and v.strip():
                        # Try to split strings on common delimiters
                        items = [item.strip() for item in v.split(',') if item.strip()]
                        if not items:  # Try semicolon
                            items = [item.strip() for item in v.split(';') if item.strip()]
                        if not items:  # Treat as single item
                            items = [v.strip()]
                        all_items.update(items)
                        valid_lists += 1
                    else:
                        # Treat as single item
                        all_items.add(str(v))
                        valid_lists += 1
                except (ValueError, TypeError):
                    continue
        
        if not all_items:
            return FusionResult(
                value=[],
                confidence=0.0,
                rule_used=self.name,
                metadata={"reason": "no_valid_lists"}
            )
        
        # Convert back to sorted list for consistency
        union_result = sorted(list(all_items))
        
        # Confidence based on overlap between sources
        confidence = min(1.0, 0.5 + (valid_lists / len(values)) * 0.5) if values else 1.0
        
        return FusionResult(
            value=union_result,
            confidence=confidence,
            rule_used=self.name,
            metadata={
                "num_sources": valid_lists,
                "total_unique_items": len(all_items),
                "total_input_values": len(values)
            }
        )


class Intersection(ConflictResolutionFunction):
    """Resolve conflicts by taking the intersection of all list values."""
    
    @property
    def name(self) -> str:
        return "intersection"
    
    def resolve(self, values: List[Any], context: FusionContext) -> FusionResult:
        """Take the intersection of all list values."""
        if not values:
            return FusionResult(
                value=[],
                confidence=0.0,
                rule_used=self.name,
                metadata={"reason": "no_values"}
            )
        
        # Convert values to sets
        value_sets = []
        valid_lists = 0
        
        for v in values:
            if v is not None and pd.notna(v):
                try:
                    if isinstance(v, (list, tuple, set)):
                        value_sets.append(set(v))
                        valid_lists += 1
                    elif isinstance(v, str) and v.strip():
                        # Try to split strings
                        items = [item.strip() for item in v.split(',') if item.strip()]
                        if not items:  # Try semicolon
                            items = [item.strip() for item in v.split(';') if item.strip()]
                        if not items:  # Treat as single item
                            items = [v.strip()]
                        value_sets.append(set(items))
                        valid_lists += 1
                    else:
                        # Treat as single item
                        value_sets.append({str(v)})
                        valid_lists += 1
                except (ValueError, TypeError):
                    continue
        
        if not value_sets:
            return FusionResult(
                value=[],
                confidence=0.0,
                rule_used=self.name,
                metadata={"reason": "no_valid_lists"}
            )
        
        # Calculate intersection
        intersection = value_sets[0]
        for s in value_sets[1:]:
            intersection = intersection.intersection(s)
        
        # Convert back to sorted list
        intersection_result = sorted(list(intersection))
        
        # Confidence based on intersection size relative to union size
        union_size = len(set.union(*value_sets))
        if union_size > 0:
            confidence = len(intersection) / union_size
        else:
            confidence = 1.0
        
        return FusionResult(
            value=intersection_result,
            confidence=confidence,
            rule_used=self.name,
            metadata={
                "num_sources": valid_lists,
                "intersection_size": len(intersection),
                "union_size": union_size,
                "total_input_values": len(values)
            }
        )


class IntersectionKSources(ConflictResolutionFunction):
    """Resolve conflicts by taking items that appear in at least K sources."""
    
    def __init__(self, k: int = 2):
        """Initialize with minimum number of sources required.
        
        Parameters
        ----------
        k : int
            Minimum number of sources that must contain an item for it to be included.
        """
        self.k = k
    
    @property
    def name(self) -> str:
        return f"intersection_{self.k}_sources"
    
    def resolve(self, values: List[Any], context: FusionContext) -> FusionResult:
        """Take items that appear in at least K sources."""
        if not values:
            return FusionResult(
                value=[],
                confidence=0.0,
                rule_used=self.name,
                metadata={"reason": "no_values"}
            )
        
        # Convert values to sets and count occurrences
        item_counts = {}
        valid_lists = 0
        
        for v in values:
            if v is not None and pd.notna(v):
                try:
                    items = set()
                    if isinstance(v, (list, tuple, set)):
                        items = set(v)
                    elif isinstance(v, str) and v.strip():
                        # Try to split strings
                        split_items = [item.strip() for item in v.split(',') if item.strip()]
                        if not split_items:  # Try semicolon
                            split_items = [item.strip() for item in v.split(';') if item.strip()]
                        if not split_items:  # Treat as single item
                            split_items = [v.strip()]
                        items = set(split_items)
                    else:
                        # Treat as single item
                        items = {str(v)}
                    
                    for item in items:
                        item_counts[item] = item_counts.get(item, 0) + 1
                    
                    valid_lists += 1
                except (ValueError, TypeError):
                    continue
        
        if not item_counts:
            return FusionResult(
                value=[],
                confidence=0.0,
                rule_used=self.name,
                metadata={"reason": "no_valid_lists"}
            )
        
        # Filter items that appear in at least K sources
        k_sources_items = [item for item, count in item_counts.items() if count >= self.k]
        k_sources_items.sort()  # Sort for consistency
        
        # Confidence based on how many items meet the threshold
        total_unique_items = len(item_counts)
        if total_unique_items > 0:
            confidence = len(k_sources_items) / total_unique_items
        else:
            confidence = 0.0
        
        # Boost confidence if K requirement is strict relative to available sources
        if valid_lists > 0:
            strictness_bonus = min(0.3, self.k / valid_lists * 0.3)
            confidence = min(1.0, confidence + strictness_bonus)
        
        return FusionResult(
            value=k_sources_items,
            confidence=confidence,
            rule_used=self.name,
            metadata={
                "k": self.k,
                "num_sources": valid_lists,
                "total_unique_items": total_unique_items,
                "items_meeting_threshold": len(k_sources_items),
                "item_counts": dict(item_counts)
            }
        )