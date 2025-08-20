"""
Normalization and validation utilities for PyDI.

This subpackage defines abstract base classes and simple implementations
for normalising values and validating data. Most classes are intentionally 
skeletal and are intended to be extended to support more complex use cases, 
such as pydantic-driven validation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict

import pandas as pd


class BaseNormalizer(ABC):
    """Abstract base class for normalisers."""

    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class Normalizer(BaseNormalizer):
    """Apply a set of normalisation functions to columns.

    Parameters
    ----------
    rules : dict
        A mapping from column names to callables that take a value and return a normalised value.
    """

    def __init__(self, rules: Dict[str, Callable[[Any], Any]]) -> None:
        self.rules = rules

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        for col, func in self.rules.items():
            if col in result.columns:
                result[col] = result[col].apply(func)
        return result


class BaseValidator(ABC):
    """Abstract base class for validators."""

    @abstractmethod
    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class PydanticValidator(BaseValidator):
    """Validate rows against a Pydantic model.

    Parameters
    ----------
    model : pydantic.BaseModel
        A Pydantic model used to validate each row. Invalid rows are dropped.
    """

    def __init__(self, model: Any) -> None:
        from pydantic import BaseModel

        if not issubclass(model, BaseModel):
            raise TypeError("model must be a subclass of pydantic.BaseModel")
        self.model = model

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        from pydantic import ValidationError

        records = []
        for _, row in df.iterrows():
            try:
                validated = self.model(**row.to_dict())
                records.append(validated.model_dump())
            except ValidationError:
                continue
        return pd.DataFrame(records)
