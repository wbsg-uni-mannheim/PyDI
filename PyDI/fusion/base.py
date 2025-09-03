"""
Base classes and data structures for data fusion in PyDI.

This module defines the core abstractions for fusing records from multiple
datasets, including conflict resolution functions, attribute fusers, and
fusion contexts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Set, Union, Tuple
from dataclasses import dataclass, field
import logging

import pandas as pd


@dataclass
class FusionContext:
    """Context information passed to fusers during fusion.
    
    Parameters
    ----------
    group_id : str
        Unique identifier for the current record group.
    attribute : str
        Name of the attribute being fused.
    source_datasets : Dict[str, str]
        Mapping from record ID to dataset name.
    timestamp : Optional[pd.Timestamp]
        Timestamp for this fusion operation.
    metadata : Dict[str, Any]
        Additional metadata for the fusion operation.
    """
    
    group_id: str
    attribute: str
    source_datasets: Dict[str, str] = field(default_factory=dict)
    timestamp: Optional[pd.Timestamp] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FusionResult:
    """Result of a fusion operation for a single attribute.
    
    Parameters
    ----------
    value : Any
        The fused value.
    sources : Set[str]
        Record IDs that contributed to this fused value.
    confidence : float
        Confidence score for the fusion result (0.0 to 1.0).
    rule_used : str
        Name of the conflict resolution rule that was applied.
    metadata : Dict[str, Any]
        Additional metadata about the fusion operation.
    """
    
    value: Any
    sources: Set[str] = field(default_factory=set)
    confidence: float = 1.0
    rule_used: str = "default"
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConflictResolutionFunction(ABC):
    """Abstract base class for conflict resolution functions.
    
    A conflict resolution function takes a list of values and resolves
    conflicts to produce a single fused value.
    """
    
    @abstractmethod
    def resolve(self, values: List[Any], context: FusionContext) -> FusionResult:
        """Resolve conflicts between multiple values.
        
        Parameters
        ----------
        values : List[Any]
            List of values to resolve conflicts for.
        context : FusionContext
            Context information for the fusion operation.
            
        Returns
        -------
        FusionResult
            The resolved fusion result.
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this conflict resolution function."""
        pass


class AttributeValueFuser:
    """Fuser for a specific attribute using a conflict resolution function.
    
    Parameters
    ----------
    resolver : Union[ConflictResolutionFunction, Callable]
        The conflict resolution function to use. Can be either a class-based
        ConflictResolutionFunction or a simple function that returns 
        (value, confidence, metadata).
    accessor : Optional[Callable]
        Function to extract values from records. If None, uses direct
        attribute access.
    required : bool
        Whether this attribute is required for fusion.
    resolver_kwargs : Dict[str, Any]
        Additional keyword arguments to pass to function-based resolvers.
    """
    
    def __init__(
        self,
        resolver: Union[ConflictResolutionFunction, Callable],
        accessor: Optional[Callable[[pd.Series], Any]] = None,
        required: bool = False,
        **resolver_kwargs
    ):
        self.resolver = resolver
        self.accessor = accessor
        self.required = required
        self.resolver_kwargs = resolver_kwargs
        
        # Check if resolver is a function or class
        self._is_function_resolver = callable(resolver) and not isinstance(resolver, ConflictResolutionFunction)
    
    def has_value(self, record: pd.Series, context: FusionContext) -> bool:
        """Check if a record has a value for this attribute.
        
        Parameters
        ----------
        record : pd.Series
            The record to check.
        context : FusionContext
            Context information.
            
        Returns
        -------
        bool
            True if the record has a non-null value for this attribute.
        """
        try:
            value = self.accessor(record) if self.accessor else record.get(context.attribute)
            return value is not None and pd.notna(value)
        except (KeyError, AttributeError):
            return False
    
    def get_value(self, record: pd.Series, context: FusionContext) -> Any:
        """Extract the value for this attribute from a record.
        
        Parameters
        ----------
        record : pd.Series
            The record to extract from.
        context : FusionContext
            Context information.
            
        Returns
        -------
        Any
            The extracted value, or None if not available.
        """
        try:
            return self.accessor(record) if self.accessor else record.get(context.attribute)
        except (KeyError, AttributeError):
            return None
    
    def fuse(self, records: List[pd.Series], context: FusionContext) -> FusionResult:
        """Fuse values from multiple records for this attribute.
        
        Parameters
        ----------
        records : List[pd.Series]
            List of records to fuse.
        context : FusionContext
            Context information.
            
        Returns
        -------
        FusionResult
            The fusion result.
        """
        # Extract values that are present
        values_with_sources = []
        for record in records:
            if self.has_value(record, context):
                value = self.get_value(record, context)
                record_id = record.get("_id", "unknown")
                values_with_sources.append((value, record_id))
        
        if not values_with_sources:
            # No values available
            resolver_name = getattr(self.resolver, 'name', self.resolver.__name__ if hasattr(self.resolver, '__name__') else 'unknown')
            return FusionResult(
                value=None,
                sources=set(),
                confidence=0.0,
                rule_used=resolver_name,
                metadata={"reason": "no_values"}
            )
        
        # Extract just the values for conflict resolution
        values = [item[0] for item in values_with_sources]
        
        # Resolve conflicts based on resolver type
        if self._is_function_resolver:
            # Function-based resolver
            try:
                # Prepare context kwargs for function
                context_kwargs = {
                    'sources': [item[1] for item in values_with_sources],
                    'group_id': context.group_id,
                    'attribute': context.attribute,
                    'source_datasets': context.source_datasets,
                    **self.resolver_kwargs
                }
                
                # Call function resolver
                resolved_value, confidence, metadata = self.resolver(values, **context_kwargs)
                
                result = FusionResult(
                    value=resolved_value,
                    sources={item[1] for item in values_with_sources},
                    confidence=confidence,
                    rule_used=self.resolver.__name__,
                    metadata=metadata
                )
            except Exception as e:
                logging.warning(f"Function resolver {self.resolver.__name__} failed: {e}")
                result = FusionResult(
                    value=values[0] if values else None,
                    sources={item[1] for item in values_with_sources},
                    confidence=0.1,
                    rule_used=self.resolver.__name__,
                    metadata={"error": str(e), "fallback": "first_value"}
                )
        else:
            # Class-based resolver
            result = self.resolver.resolve(values, context)
            # Add source information
            result.sources = {item[1] for item in values_with_sources}
        
        return result


@dataclass
class RecordGroup:
    """A group of records that should be fused together.
    
    Parameters
    ----------
    group_id : str
        Unique identifier for this group.
    records : List[pd.Series]
        List of records in this group.
    source_datasets : Dict[str, str]
        Mapping from record ID to dataset name.
    """
    
    group_id: str
    records: List[pd.Series] = field(default_factory=list)
    source_datasets: Dict[str, str] = field(default_factory=dict)
    
    def add_record(self, record: pd.Series, dataset_name: str):
        """Add a record to this group.
        
        Parameters
        ----------
        record : pd.Series
            The record to add.
        dataset_name : str
            Name of the dataset this record comes from.
        """
        self.records.append(record)
        record_id = record.get("_id", "unknown")
        self.source_datasets[record_id] = dataset_name
    
    def get_all_attributes(self) -> Set[str]:
        """Get all attributes present across records in this group.
        
        Returns
        -------
        Set[str]
            Set of all attribute names.
        """
        attributes = set()
        for record in self.records:
            attributes.update(record.index)
        return attributes