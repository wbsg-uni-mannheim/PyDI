"""
Data fusion strategy management for PyDI.

This module defines the DataFusionStrategy class for registering and managing
attribute-level fusers and evaluation rules.
"""

from __future__ import annotations

from typing import Dict, Optional, Set, Callable, Any, Union
import logging
from functools import partial

from .base import AttributeValueFuser, ConflictResolutionFunction, FusionContext, get_callable_name


# Type alias for evaluation functions
# Functions should return bool and typically accept (fused_value, expected_value),
# but may also accept additional keyword parameters (e.g., threshold).
EvaluationFunction = Callable[..., bool]


class DataFusionStrategy:
    """Strategy for data fusion that manages attribute fusers and evaluation rules.

    A strategy defines how to fuse each attribute by registering AttributeValueFuser
    instances and optional evaluation rules for quality assessment.
    """

    def __init__(self, name: str = "default"):
        """Initialize a new data fusion strategy.

        Parameters
        ----------
        name : str
            Name of this strategy.
        """
        self.name = name
        self._attribute_fusers: Dict[str, AttributeValueFuser] = {}
        self._evaluation_rules: Dict[str, EvaluationFunction] = {}
        self._logger = logging.getLogger(__name__)

    def add_attribute_fuser(
        self,
        attribute: str,
        fuser_or_resolver: Union[AttributeValueFuser, ConflictResolutionFunction],
        evaluation_function: Optional[EvaluationFunction] = None,
        **fuser_kwargs: Any,
    ) -> None:
        """Register how to fuse an attribute (unified API).

        Accepts either a pre-built AttributeValueFuser (advanced usage) or a
        resolver function (simple usage). When a resolver function is provided,
        it is wrapped into an AttributeValueFuser with any extra keyword args
        supplied via ``fuser_kwargs`` (e.g., accessor=..., required=True).

        Parameters
        ----------
        attribute : str
            Name of the attribute to fuse.
        fuser_or_resolver : Union[AttributeValueFuser, ConflictResolutionFunction]
            Either an AttributeValueFuser instance or a resolver function that
            resolves conflicts and returns (value, confidence, metadata).
        evaluation_function : Optional[EvaluationFunction]
            Optional evaluation rule for quality assessment.
        **fuser_kwargs : Any
            Extra options used when a resolver function is provided, passed to
            AttributeValueFuser (e.g., accessor=callable, required=True).
        """
        if isinstance(fuser_or_resolver, AttributeValueFuser):
            fuser = fuser_or_resolver
        else:
            # Assume callable resolver and wrap it
            fuser = AttributeValueFuser(fuser_or_resolver, **fuser_kwargs)

        self._attribute_fusers[attribute] = fuser
        if evaluation_function:
            self._evaluation_rules[attribute] = evaluation_function

        resolver_name = get_callable_name(fuser.resolver)
        self._logger.info(
            f"Registered fuser for attribute '{attribute}' using rule '{resolver_name}'"
        )


    def get_attribute_fuser(self, attribute: str) -> Optional[AttributeValueFuser]:
        """Get the fuser registered for a specific attribute.

        Parameters
        ----------
        attribute : str
            Name of the attribute.

        Returns
        -------
        Optional[AttributeValueFuser]
            The registered fuser, or None if not found.
        """
        return self._attribute_fusers.get(attribute)

    def get_evaluation_function(self, attribute: str) -> Optional[EvaluationFunction]:
        """Get the evaluation rule registered for a specific attribute.

        Parameters
        ----------
        attribute : str
            Name of the attribute.

        Returns
        -------
        Optional[EvaluationFunction]
            The registered evaluation function, or None if not found.
        """
        return self._evaluation_rules.get(attribute)

    def add_evaluation_function(
        self,
        attribute: str,
        evaluation_function: EvaluationFunction,
        **kwargs: Any,
    ) -> None:
        """Register an evaluation function for an attribute.

        Supports optional keyword parameters which will be bound to the
        function using ``functools.partial``.
        
        Parameters
        ----------
        attribute : str
            Name of the attribute.
        evaluation_function : EvaluationFunction
            Function to evaluate fusion results for this attribute. Typically
            has signature ``(fused_value, expected_value, **params) -> bool``.
        **kwargs : Any
            Optional parameters to bind to the evaluation function (e.g.,
            ``threshold=0.7``).

        Examples
        --------
        Register without parameters (exact signature):
        >>> strategy.add_evaluation_function("title", tokenized_match)

        Register with parameters using keyword args (will be bound via partial):
        >>> strategy.add_evaluation_function("director_name", tokenized_match, threshold=0.7)
        >>> strategy.add_evaluation_function("actors", tokenized_match, threshold=0.5)
        """
        bound_function: EvaluationFunction = (
            partial(evaluation_function, **kwargs) if kwargs else evaluation_function
        )
        self._evaluation_rules[attribute] = bound_function
        self._logger.info(
            f"Registered evaluation function for attribute '{attribute}'"
            + (f" with params {kwargs}" if kwargs else "")
        )

    def get_registered_attributes(self) -> Set[str]:
        """Get all attributes that have registered fusers.

        Returns
        -------
        Set[str]
            Set of attribute names with registered fusers.
        """
        return set(self._attribute_fusers.keys())

    def has_fuser_for(self, attribute: str) -> bool:
        """Check if a fuser is registered for the given attribute.

        Parameters
        ----------
        attribute : str
            Name of the attribute to check.

        Returns
        -------
        bool
            True if a fuser is registered for this attribute.
        """
        return attribute in self._attribute_fusers

    def remove_attribute_fuser(self, attribute: str) -> bool:
        """Remove the fuser for a specific attribute.

        Parameters
        ----------
        attribute : str
            Name of the attribute.

        Returns
        -------
        bool
            True if a fuser was removed, False if none was registered.
        """
        removed_fuser = self._attribute_fusers.pop(attribute, None)
        # Also remove evaluation rule
        self._evaluation_rules.pop(attribute, None)

        if removed_fuser:
            self._logger.info(f"Removed fuser for attribute '{attribute}'")
            return True
        return False

    def clear(self) -> None:
        """Remove all registered fusers and evaluation rules."""
        self._attribute_fusers.clear()
        self._evaluation_rules.clear()
        self._logger.info("Cleared all registered fusers and evaluation rules")

    def __repr__(self) -> str:
        return f"DataFusionStrategy(name='{self.name}', attributes={len(self._attribute_fusers)})"
