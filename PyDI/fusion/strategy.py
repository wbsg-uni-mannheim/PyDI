"""
Data fusion strategy management for PyDI.

This module defines the DataFusionStrategy class for registering and managing
attribute-level fusers and evaluation rules.
"""

from __future__ import annotations

from typing import Dict, Optional, Set
import logging

from .base import AttributeValueFuser, ConflictResolutionFunction, FusionContext


class EvaluationRule:
    """Base class for evaluation rules used to assess fusion quality.

    Parameters
    ----------
    name : str
        Name of this evaluation rule.
    """

    def __init__(self, name: str):
        self.name = name

    def evaluate(self, fused_value, gold_value, context: FusionContext) -> bool:
        """Evaluate if a fused value matches the gold standard.

        Parameters
        ----------
        fused_value : Any
            The fused value to evaluate.
        gold_value : Any
            The gold standard value.
        context : FusionContext
            Context information.

        Returns
        -------
        bool
            True if the fused value is considered correct.
        """
        # Default: exact equality
        return fused_value == gold_value


class StringEqualityRule(EvaluationRule):
    """Evaluation rule for exact string equality."""

    def __init__(self):
        super().__init__("string_equality")


class NumericToleranceRule(EvaluationRule):
    """Evaluation rule for numeric values with tolerance.

    Parameters
    ----------
    tolerance : float
        Absolute tolerance for numeric comparison.
    """

    def __init__(self, tolerance: float = 0.01):
        super().__init__("numeric_tolerance")
        self.tolerance = tolerance

    def evaluate(self, fused_value, gold_value, context: FusionContext) -> bool:
        """Evaluate with numeric tolerance."""
        try:
            return abs(float(fused_value) - float(gold_value)) <= self.tolerance
        except (ValueError, TypeError):
            return fused_value == gold_value


class SetEqualityRule(EvaluationRule):
    """Evaluation rule for set equality (order-independent)."""

    def __init__(self):
        super().__init__("set_equality")

    def evaluate(self, fused_value, gold_value, context: FusionContext) -> bool:
        """Evaluate as sets (order-independent)."""
        try:
            if isinstance(fused_value, (list, tuple, set)) and isinstance(gold_value, (list, tuple, set)):
                return set(fused_value) == set(gold_value)
            return fused_value == gold_value
        except (TypeError, ValueError):
            return fused_value == gold_value


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
        self._evaluation_rules: Dict[str, EvaluationRule] = {}
        self._logger = logging.getLogger(__name__)

    def add_attribute_fuser(
        self,
        attribute: str,
        fuser: AttributeValueFuser,
        evaluation_rule: Optional[EvaluationRule] = None,
    ) -> None:
        """Register an attribute fuser for a specific attribute.

        Parameters
        ----------
        attribute : str
            Name of the attribute to fuse.
        fuser : AttributeValueFuser
            The fuser to use for this attribute.
        evaluation_rule : Optional[EvaluationRule]
            Optional evaluation rule for quality assessment.
        """
        self._attribute_fusers[attribute] = fuser
        if evaluation_rule:
            self._evaluation_rules[attribute] = evaluation_rule

        # Be robust to both class-based and function-based resolvers
        resolver_name = getattr(
            fuser.resolver,
            'name',
            getattr(fuser.resolver, '__name__', type(fuser.resolver).__name__),
        )
        self._logger.info(
            f"Registered fuser for attribute '{attribute}' using rule '{resolver_name}'"
        )

    def add_attribute_fuser_from_resolver(
        self,
        attribute: str,
        resolver: ConflictResolutionFunction,
        evaluation_rule: Optional[EvaluationRule] = None,
        **fuser_kwargs,
    ) -> None:
        """Register an attribute fuser from a conflict resolution function.

        Parameters
        ----------
        attribute : str
            Name of the attribute to fuse.
        resolver : ConflictResolutionFunction
            The conflict resolution function to use.
        evaluation_rule : Optional[EvaluationRule]
            Optional evaluation rule for quality assessment.
        **fuser_kwargs
            Additional keyword arguments for AttributeValueFuser.
        """
        fuser = AttributeValueFuser(resolver, **fuser_kwargs)
        self.add_attribute_fuser(attribute, fuser, evaluation_rule)

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

    def get_evaluation_rule(self, attribute: str) -> Optional[EvaluationRule]:
        """Get the evaluation rule registered for a specific attribute.

        Parameters
        ----------
        attribute : str
            Name of the attribute.

        Returns
        -------
        Optional[EvaluationRule]
            The registered evaluation rule, or None if not found.
        """
        return self._evaluation_rules.get(attribute)

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
