"""
Base classes and data structures for data fusion in PyDI.

This module defines the core abstractions for fusing records from multiple
datasets, including conflict resolution functions, attribute fusers, and
fusion contexts.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Set, Union, Tuple
from dataclasses import dataclass, field
import logging
from functools import partial as _partial

import pandas as pd
import numpy as np
def _is_valid_value(value: Any) -> bool:
    """Check if a value is valid (not None, not NA, not empty list).
    
    This helper function properly handles both scalar values and lists/arrays
    that may be returned from loaders with aggregate mode.
    
    Parameters
    ----------
    value : Any
        The value to check.
        
    Returns
    -------
    bool
        True if the value is valid and usable for fusion.
    """
    if value is None:
        return False
    
    # Handle lists/arrays - consider empty lists as invalid
    if isinstance(value, (list, tuple)):
        return len(value) > 0
    
    # Handle numpy arrays
    try:
        if isinstance(value, np.ndarray):
            return value.size > 0
    except Exception:
        pass
    
    # For scalars, check with pd.isna (inverse of pd.notna)
    try:
        return not pd.isna(value)
    except Exception:
        # If pd.isna fails (e.g., complex objects), assume valid
        return True
    
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
    # Debugging controls
    debug: bool = False
    # Callable to emit a structured debug record for this attribute fusion
    debug_emit: Optional[Callable[[Dict[str, Any]], None]] = None


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


def get_callable_name(fn: Any) -> str:
    """Return a readable name for a callable.

    Handles simple functions, functools.partial, objects with 'name', and
    callable instances by falling back to their type name.
    """
    try:
        # Custom 'name' attribute takes precedence
        name = getattr(fn, "name", None)
        if isinstance(name, str) and name:
            return name

        # Unwrap functools.partial to the underlying function
        if isinstance(fn, _partial):
            return get_callable_name(fn.func)

        # Regular functions and callables
        if hasattr(fn, "__name__"):
            return getattr(fn, "__name__")  # type: ignore[no-any-return]

        # Callable instances: use class name
        return type(fn).__name__
    except Exception:
        return "unknown"


# Type alias for conflict resolution functions
# Accepts variable args to support optional kwargs in call sites.
ConflictResolutionFunction = Callable[..., Tuple[Any, float, Dict[str, Any]]]


class AttributeValueFuser:
    """Fuser for a specific attribute using a conflict resolution function.
    
    Parameters
    ----------
    resolver : Callable
        The conflict resolution function to use. Should accept (values, **kwargs)
        and return (resolved_value, confidence, metadata).
    accessor : Optional[Callable]
        Function to extract values from records. If None, uses direct
        attribute access.
    required : bool
        Whether this attribute is required for fusion.
    resolver_kwargs : Dict[str, Any]
        Additional keyword arguments to pass to the resolver function.
    """
    
    def __init__(
        self,
        resolver: Callable,
        accessor: Optional[Callable[[pd.Series], Any]] = None,
        required: bool = False,
        **resolver_kwargs
    ):
        self.resolver = resolver
        self.accessor = accessor
        self.required = required
        self.resolver_kwargs = resolver_kwargs
    
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
            return _is_valid_value(value)
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
        logger = logging.getLogger(__name__)
        resolver_name = get_callable_name(self.resolver)

        # Extract values that are present
        values_with_sources: List[Tuple[Any, str]] = []
        for record in records:
            if self.has_value(record, context):
                value = self.get_value(record, context)
                record_id = record.get("_id", "unknown")
                values_with_sources.append((value, record_id))

        if not values_with_sources:
            # No values available
            result = FusionResult(
                value=None,
                sources=set(),
                confidence=0.0,
                rule_used=resolver_name,
                metadata={"reason": "no_values"}
            )
            if context.debug and context.debug_emit is not None:
                try:
                    context.debug_emit({
                        "group_id": context.group_id,
                        "attribute": context.attribute,
                        "conflict_resolution_function": resolver_name,
                        "inputs": [],
                        "resolver_kwargs": dict(self.resolver_kwargs),
                        "output": {
                            "value": None,
                            "confidence": 0.0,
                            "metadata": result.metadata,
                        },
                        "error": None,
                    })
                except Exception:
                    pass
            return result

        # Extract just the values for conflict resolution
        values = [item[0] for item in values_with_sources]

        # Call function resolver
        try:
            # Prepare context kwargs for function
            context_kwargs = {
                'sources': [item[1] for item in values_with_sources],
                'group_id': context.group_id,
                'attribute': context.attribute,
                'source_datasets': context.source_datasets,
                **self.resolver_kwargs
            }
            # Pass engine-level metadata (e.g., trust_map) when available
            try:
                if isinstance(context.metadata, dict) and 'trust_map' in context.metadata:
                    context_kwargs.setdefault('trust_map', context.metadata['trust_map'])
            except Exception:
                pass

            # Call function resolver
            resolved_value, confidence, metadata = self.resolver(values, **context_kwargs)

            result = FusionResult(
                value=resolved_value,
                sources={item[1] for item in values_with_sources},
                confidence=confidence,
                rule_used=resolver_name,
                metadata=metadata
            )
            # Emit structured debug record if requested
            if context.debug and context.debug_emit is not None:
                try:
                    context.debug_emit({
                        "group_id": context.group_id,
                        "attribute": context.attribute,
                        "conflict_resolution_function": resolver_name,
                        "inputs": [
                            {
                                "record_id": rid,
                                "dataset": context.source_datasets.get(rid, "unknown"),
                                "value": val,
                            }
                            for (val, rid) in values_with_sources
                        ],
                        "resolver_kwargs": dict(self.resolver_kwargs),
                        "output": {
                            "value": resolved_value,
                            "confidence": confidence,
                            "metadata": metadata,
                        },
                        "error": None,
                    })
                except Exception:
                    # Ensure debug emission never breaks fusion
                    pass
        except Exception as e:
            logger.warning(f"Function resolver {resolver_name} failed: {e}")
            result = FusionResult(
                value=values[0] if values else None,
                sources={item[1] for item in values_with_sources},
                confidence=0.1,
                rule_used=resolver_name,
                metadata={"error": str(e), "fallback": "first_value"}
            )
            # Emit structured debug record including error if requested
            if context.debug and context.debug_emit is not None:
                try:
                    context.debug_emit({
                        "group_id": context.group_id,
                        "attribute": context.attribute,
                        "conflict_resolution_function": resolver_name,
                        "inputs": [
                            {
                                "record_id": rid,
                                "dataset": context.source_datasets.get(rid, "unknown"),
                                "value": val,
                            }
                            for (val, rid) in values_with_sources
                        ],
                        "resolver_kwargs": dict(self.resolver_kwargs),
                        "output": {
                            "value": result.value,
                            "confidence": result.confidence,
                            "metadata": result.metadata,
                        },
                        "error": str(e),
                    })
                except Exception:
                    pass
        
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
