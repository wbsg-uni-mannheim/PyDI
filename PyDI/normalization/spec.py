"""
Normalization specification for user-defined transformations.

This module allows users to specify how columns should be normalized
based on the profile report. Specifications can be created manually
or auto-generated from profile suggestions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

from .profile import DataFrameProfile


@dataclass
class ColumnSpec:
    """Specification for normalizing a single column."""

    # General options
    output_type: Literal["string", "float", "int", "bool", "datetime", "keep"] = "keep"

    # Failure handling
    # keep: Keep original value when transformation fails (default)
    # null: Set to None when transformation fails
    # raise: Raise an error when transformation fails
    on_failure: Literal["keep", "null", "raise"] = "keep"

    # Unit handling
    target_unit: str | None = None
    expand_scale_modifiers: bool = False

    # Percentage handling
    # to_decimal: '50%' → 0.5, or keeps 0.5 as-is
    # to_percent: 0.5 → 50, or '50%' → 50 (removes % symbol)
    convert_percentage: Literal["to_decimal", "to_percent", "keep"] | None = None

    # Country/currency normalization
    country_format: Literal["alpha_2", "alpha_3", "numeric", "name", "keep"] | None = None
    currency_format: Literal["alpha_3", "name", "keep"] | None = None

    # Phone number formatting
    phone_format: Literal["e164", "international", "national", "keep"] | None = None
    phone_default_region: str = "US"

    # Email normalization
    normalize_email: bool = False

    # Standard number formatting
    stdnum_format: bool = False  # Format to canonical form

    # Text options
    case: Literal["lower", "upper", "title", "keep"] | None = None
    strip_whitespace: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_type": self.output_type,
            "on_failure": self.on_failure,
            "target_unit": self.target_unit,
            "expand_scale_modifiers": self.expand_scale_modifiers,
            "convert_percentage": self.convert_percentage,
            "country_format": self.country_format,
            "currency_format": self.currency_format,
            "phone_format": self.phone_format,
            "phone_default_region": self.phone_default_region,
            "normalize_email": self.normalize_email,
            "stdnum_format": self.stdnum_format,
            "case": self.case,
            "strip_whitespace": self.strip_whitespace,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ColumnSpec:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class NormalizationSpec:
    """Specification for normalizing an entire DataFrame."""

    columns: dict[str, ColumnSpec] = field(default_factory=dict)

    def set_column(self, column_name: str, **kwargs: Any) -> NormalizationSpec:
        """
        Set specification for a column.

        Args:
            column_name: Name of the column
            **kwargs: ColumnSpec parameters

        Returns:
            Self for chaining

        Examples:
            >>> spec = NormalizationSpec()
            >>> spec.set_column("revenue", expand_scale_modifiers=True, output_type="float")
            >>> spec.set_column("country", country_format="alpha_2")
        """
        self.columns[column_name] = ColumnSpec(**kwargs)
        return self

    def get_column(self, column_name: str) -> ColumnSpec | None:
        """Get specification for a column."""
        return self.columns.get(column_name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": {name: spec.to_dict() for name, spec in self.columns.items()},
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NormalizationSpec:
        spec = cls()
        for col_name, col_data in data.get("columns", {}).items():
            spec.columns[col_name] = ColumnSpec.from_dict(col_data)
        return spec

    @classmethod
    def from_json(cls, json_str: str) -> NormalizationSpec:
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_profile(
        cls,
        profile: DataFrameProfile,
        auto_apply_suggestions: bool = True,
    ) -> NormalizationSpec:
        """
        Create a specification from a DataFrame profile.

        Args:
            profile: Profile from profile_dataframe()
            auto_apply_suggestions: Whether to auto-apply detected normalizations

        Returns:
            NormalizationSpec with suggested transformations

        Examples:
            >>> profile = profile_dataframe(df)
            >>> spec = NormalizationSpec.from_profile(profile)
        """
        spec = cls()

        if not auto_apply_suggestions:
            return spec

        for col_name, col_profile in profile.columns.items():
            col_spec = ColumnSpec()

            # Apply suggestions based on detected type
            if col_profile.detected_type == "unit_quantity":
                col_spec.output_type = "float"
                # If units are consistent, normalize them
                if col_profile.unit_info:
                    units = col_profile.unit_info.get("units_detected", {})
                    if units:
                        # Use the most common unit as target
                        most_common = max(units.keys(), key=lambda u: units[u])
                        col_spec.target_unit = most_common

            elif col_profile.detected_type == "scaled_number":
                col_spec.expand_scale_modifiers = True
                col_spec.output_type = "float"

            elif col_profile.detected_type == "country":
                col_spec.country_format = "alpha_2"

            elif col_profile.detected_type == "currency":
                col_spec.currency_format = "alpha_3"

            elif col_profile.detected_type == "phone":
                col_spec.phone_format = "e164"

            elif col_profile.detected_type == "email":
                col_spec.normalize_email = True

            elif col_profile.detected_type == "stdnum":
                col_spec.stdnum_format = True

            elif col_profile.detected_type == "percentage":
                # If detected as percentages with % symbol, convert to decimal
                if col_profile.percentage_info:
                    if col_profile.percentage_info.get("format") == "symbol":
                        col_spec.convert_percentage = "to_decimal"
                        col_spec.output_type = "float"

            # Only add spec if we have something to do
            if col_spec != ColumnSpec():
                spec.columns[col_name] = col_spec

        return spec


__all__ = [
    "ColumnSpec",
    "NormalizationSpec",
]
