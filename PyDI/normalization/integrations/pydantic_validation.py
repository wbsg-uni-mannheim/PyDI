"""
Pydantic integration for schema-based validation.

This module wraps Pydantic to provide:
- DataFrame validation against Pydantic models
- Row-by-row validation with error collection
- Conversion between DataFrames and Pydantic models
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Type, TypeVar

import pandas as pd

logger = logging.getLogger(__name__)

# Pydantic is optional
try:
    from pydantic import BaseModel, ValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = None  # type: ignore
    ValidationError = None  # type: ignore

T = TypeVar("T")


@dataclass
class PydanticValidationResult:
    """Result of validating a DataFrame against a Pydantic model."""

    total_rows: int
    valid_rows: int
    errors: list[dict[str, Any]] = field(default_factory=list)
    validated_data: list[Any] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Check if all rows passed validation."""
        return len(self.errors) == 0

    @property
    def error_rate(self) -> float:
        """Calculate error rate as percentage."""
        if self.total_rows == 0:
            return 0.0
        return (len(self.errors) / self.total_rows) * 100

    @property
    def validity_rate(self) -> float:
        """Calculate validity rate as percentage."""
        if self.total_rows == 0:
            return 100.0
        return (self.valid_rows / self.total_rows) * 100

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_rows": self.total_rows,
            "valid_rows": self.valid_rows,
            "error_count": len(self.errors),
            "error_rate": round(self.error_rate, 2),
            "validity_rate": round(self.validity_rate, 2),
        }


def is_pydantic_available() -> bool:
    """Check if Pydantic is installed."""
    return PYDANTIC_AVAILABLE


def validate_dataframe(
    df: pd.DataFrame,
    model: Type[T],
    collect_valid: bool = False,
) -> PydanticValidationResult:
    """
    Validate a DataFrame row-by-row against a Pydantic model.

    Args:
        df: DataFrame to validate
        model: Pydantic model class to validate against
        collect_valid: If True, collect successfully validated model instances

    Returns:
        PydanticValidationResult with validation details

    Raises:
        ImportError: If Pydantic is not installed

    Examples:
        >>> from pydantic import BaseModel
        >>> class User(BaseModel):
        ...     name: str
        ...     age: int
        >>> df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [30, "invalid"]})
        >>> result = validate_dataframe(df, User)
        >>> result.valid_rows
        1
    """
    if not PYDANTIC_AVAILABLE:
        raise ImportError(
            "Pydantic is required for schema validation. "
            "Install it with: pip install pydantic"
        )

    result = PydanticValidationResult(
        total_rows=len(df),
        valid_rows=0,
        errors=[],
        validated_data=[],
    )

    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        try:
            validated = model(**row_dict)
            result.valid_rows += 1
            if collect_valid:
                result.validated_data.append(validated)
        except ValidationError as e:
            for error in e.errors():
                result.errors.append({
                    "row": idx,
                    "field": error.get("loc", [None])[0],
                    "message": error.get("msg", "Validation error"),
                    "type": error.get("type", "unknown"),
                    "value": row_dict.get(error.get("loc", [None])[0]),
                })

    return result


def validate_dict(
    data: dict[str, Any],
    model: Type[T],
) -> tuple[T | None, list[dict[str, Any]]]:
    """
    Validate a dictionary against a Pydantic model.

    Args:
        data: Dictionary to validate
        model: Pydantic model class

    Returns:
        Tuple of (validated model instance or None, list of errors)

    Raises:
        ImportError: If Pydantic is not installed
    """
    if not PYDANTIC_AVAILABLE:
        raise ImportError(
            "Pydantic is required for schema validation. "
            "Install it with: pip install pydantic"
        )

    try:
        validated = model(**data)
        return validated, []
    except ValidationError as e:
        errors = []
        for error in e.errors():
            errors.append({
                "field": error.get("loc", [None])[0],
                "message": error.get("msg", "Validation error"),
                "type": error.get("type", "unknown"),
                "value": data.get(error.get("loc", [None])[0]),
            })
        return None, errors


def dataframe_to_models(
    df: pd.DataFrame,
    model: Type[T],
    skip_invalid: bool = True,
) -> list[T]:
    """
    Convert a DataFrame to a list of Pydantic model instances.

    Args:
        df: DataFrame to convert
        model: Pydantic model class
        skip_invalid: If True, skip rows that fail validation

    Returns:
        List of validated model instances

    Raises:
        ImportError: If Pydantic is not installed
        ValidationError: If skip_invalid is False and any row fails validation
    """
    if not PYDANTIC_AVAILABLE:
        raise ImportError(
            "Pydantic is required for schema validation. "
            "Install it with: pip install pydantic"
        )

    models = []
    for _, row in df.iterrows():
        try:
            validated = model(**row.to_dict())
            models.append(validated)
        except ValidationError:
            if not skip_invalid:
                raise
            logger.debug(f"Skipping invalid row: {row.to_dict()}")

    return models


def models_to_dataframe(models: list[T]) -> pd.DataFrame:
    """
    Convert a list of Pydantic models to a DataFrame.

    Args:
        models: List of Pydantic model instances

    Returns:
        DataFrame with model data
    """
    if not models:
        return pd.DataFrame()

    # Use model_dump for Pydantic v2, dict for v1
    if hasattr(models[0], "model_dump"):
        data = [m.model_dump() for m in models]
    else:
        data = [m.dict() for m in models]  # type: ignore

    return pd.DataFrame(data)


def get_model_fields(model: Type[T]) -> dict[str, Any]:
    """
    Get field information from a Pydantic model.

    Args:
        model: Pydantic model class

    Returns:
        Dictionary with field names and their types/constraints
    """
    if not PYDANTIC_AVAILABLE:
        raise ImportError("Pydantic is required")

    # Pydantic v2
    if hasattr(model, "model_fields"):
        return {
            name: {
                "type": str(info.annotation),
                "required": info.is_required(),
                "default": info.default if info.default is not None else None,
            }
            for name, info in model.model_fields.items()
        }
    # Pydantic v1
    elif hasattr(model, "__fields__"):
        return {
            name: {
                "type": str(field.outer_type_),
                "required": field.required,
                "default": field.default,
            }
            for name, field in model.__fields__.items()  # type: ignore
        }
    return {}
