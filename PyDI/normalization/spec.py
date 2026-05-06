"""
Normalization specification for user-defined transformations.

This module allows users to specify how columns should be normalized
based on the profile report. Specifications can be created manually,
auto-generated from profile suggestions, or imported from JSON Schema.

Note: Validation is handled separately by the validators module.
This module focuses only on data transformation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .profile import DataFrameProfile


@dataclass
class ColumnSpec:
    """
    Specification for normalizing a single column.

    This class defines how a column should be transformed. It focuses purely
    on data transformation - validation is handled separately by the
    validators module.

    Attributes:
        output_type: Target data type for the column.
        on_failure: What to do when transformation fails.
        target_unit: Target unit for unit conversion (e.g., "m", "kg").
        expand_scale_modifiers: Expand scale modifiers like "5 MEO" -> 5000000.
        convert_percentage: How to handle percentage values.
        country_format: Output format for country codes.
        currency_format: Output format for currency codes.
        phone_format: Output format for phone numbers.
        phone_default_region: Default region for phone parsing.
        normalize_email: Whether to normalize email addresses.
        stdnum_format: Whether to format standard numbers.
        case: Text case transformation.
        strip_whitespace: Whether to strip leading/trailing whitespace.

    Examples:
        >>> # Basic type conversion
        >>> spec = ColumnSpec(output_type="float")

        >>> # Country normalization
        >>> spec = ColumnSpec(country_format="alpha_2")
    """

    output_type: Literal["string", "float", "int", "bool", "datetime", "keep"] = "keep"
    """Target data type for the column. 'keep' preserves original type."""

    on_failure: Literal["keep", "null", "raise"] = "keep"
    """
    What to do when transformation fails.

    - 'keep': Keep original value (default)
    - 'null': Set to None
    - 'raise': Raise an error
    """

    # Unit handling
    target_unit: str | None = None
    """Target unit for unit conversion (e.g., 'm', 'kg', 'USD')."""

    expand_scale_modifiers: bool = False
    """Expand scale modifiers like '5 MEO' -> 5000000, '3 MEUR' -> 3000000."""

    # Percentage handling
    convert_percentage: Literal["to_decimal", "to_percent", "keep"] | None = None
    """
    How to handle percentage values.

    - 'to_decimal': '50%' -> 0.5, or keeps 0.5 as-is
    - 'to_percent': 0.5 -> 50, or '50%' -> 50 (removes % symbol)
    - 'keep': No conversion
    """

    # Country/currency normalization
    country_format: Literal["alpha_2", "alpha_3", "numeric", "name", "keep"] | None = (
        None
    )
    """Output format for country codes (ISO 3166)."""

    currency_format: Literal["alpha_3", "name", "keep"] | None = None
    """Output format for currency codes (ISO 4217)."""

    # Phone number formatting
    phone_format: Literal["e164", "international", "national", "keep"] | None = None
    """Output format for phone numbers."""

    phone_default_region: str = "US"
    """Default region for parsing phone numbers without country code."""

    # Email normalization
    normalize_email: bool = False
    """Whether to normalize email addresses (lowercase, etc.)."""

    # Standard number formatting
    stdnum_format: bool = False
    """Whether to format standard numbers (ISBN, IBAN, VAT, etc.)."""

    # Date/datetime options
    date_format: str | None = None
    """Format string for parsing dates (e.g., '%Y' for year-only, '%Y-%m-%d')."""

    # Text options
    case: Literal["lower", "upper", "title", "keep"] | None = None
    """Text case transformation to apply."""

    strip_whitespace: bool = False
    """Whether to strip leading/trailing whitespace."""

    # Taxonomy normalization
    taxonomy_path: str | None = None
    """Path to CSV file containing taxonomy values for categorical normalization."""

    taxonomy_column: str | None = None
    """Column name in taxonomy CSV to use for mapping values. If None, uses first column."""

    taxonomy_mapping_path: str | None = None
    """Path to JSON file for caching the source-to-taxonomy mapping."""

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary representation.

        Returns:
            Dictionary with all field values.
        """
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
            "date_format": self.date_format,
            "case": self.case,
            "strip_whitespace": self.strip_whitespace,
            "taxonomy_path": self.taxonomy_path,
            "taxonomy_column": self.taxonomy_column,
            "taxonomy_mapping_path": self.taxonomy_mapping_path,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ColumnSpec:
        """
        Create ColumnSpec from dictionary.

        Args:
            data: Dictionary with ColumnSpec fields.

        Returns:
            New ColumnSpec instance.
        """
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
