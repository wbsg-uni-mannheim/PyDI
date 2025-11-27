"""
DataFrame profiling for normalization.

This module analyzes DataFrame columns to detect:
- Data types and patterns
- Units and scale modifiers
- Standard number formats (ISBN, IBAN, etc.)
- Country/currency codes
- Phone numbers and emails

The profile report helps users understand their data and specify
how columns should be normalized.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

import pandas as pd

from .integrations import (
    parse_quantity,
    detect_unit_in_text,
    is_valid_unit,
    detect_stdnum_type,
    validate_phone,
    is_valid_email,
    lookup_country,
    lookup_currency,
)
from .scale import detect_scale_modifier
from .types import (
    BooleanParser,
    LinkNormalizer,
    DateNormalizer,
    CoordinateParser,
)

logger = logging.getLogger(__name__)

DetectedType = Literal[
    "numeric",
    "text",
    "date",
    "boolean",
    "email",
    "phone",
    "url",
    "coordinate",
    "country",
    "currency",
    "unit_quantity",
    "scaled_number",
    "percentage",
    "stdnum",
    "mixed",
    "empty",
    "unknown",
]


class DataTypeExtended(Enum):
    """Extended data types for column analysis.

    This enum provides a richer set of types than DetectedType,
    including types like COORDINATE, PERCENTAGE, LIST that are
    detected through pattern matching.
    """

    NUMERIC = "numeric"
    STRING = "string"
    DATE = "date"
    DATETIME = "datetime"
    BOOLEAN = "bool"
    COORDINATE = "coordinate"
    LINK = "link"
    EMAIL = "email"
    PHONE = "phone"
    CURRENCY = "currency"
    PERCENTAGE = "percentage"
    LIST = "list"
    UNIT = "unit"
    UNKNOWN = "unknown"

    @classmethod
    def from_detected_type(cls, detected_type: DetectedType) -> "DataTypeExtended":
        """Convert DetectedType string to DataTypeExtended enum."""
        mapping = {
            "numeric": cls.NUMERIC,
            "text": cls.STRING,
            "date": cls.DATE,
            "boolean": cls.BOOLEAN,
            "email": cls.EMAIL,
            "phone": cls.PHONE,
            "url": cls.LINK,
            "coordinate": cls.COORDINATE,
            "country": cls.STRING,
            "currency": cls.CURRENCY,
            "unit_quantity": cls.UNIT,
            "scaled_number": cls.NUMERIC,
            "stdnum": cls.STRING,
            "mixed": cls.STRING,
            "empty": cls.UNKNOWN,
            "unknown": cls.UNKNOWN,
        }
        return mapping.get(detected_type, cls.UNKNOWN)


@dataclass
class ColumnProfile:
    """Profile information for a single column."""

    name: str
    dtype: str
    total_count: int
    null_count: int
    unique_count: int
    detected_type: DetectedType
    sample_values: list[Any]
    # Type-specific details
    unit_info: dict[str, Any] | None = None
    scale_info: dict[str, Any] | None = None
    percentage_info: dict[str, Any] | None = None
    stdnum_info: dict[str, Any] | None = None
    country_info: dict[str, Any] | None = None
    suggestions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "total_count": self.total_count,
            "null_count": self.null_count,
            "unique_count": self.unique_count,
            "detected_type": self.detected_type,
            "sample_values": self.sample_values,
            "unit_info": self.unit_info,
            "scale_info": self.scale_info,
            "percentage_info": self.percentage_info,
            "stdnum_info": self.stdnum_info,
            "country_info": self.country_info,
            "suggestions": self.suggestions,
        }


@dataclass
class DataFrameProfile:
    """Profile information for an entire DataFrame."""

    row_count: int
    column_count: int
    columns: dict[str, ColumnProfile]

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_count": self.row_count,
            "column_count": self.column_count,
            "columns": {name: col.to_dict() for name, col in self.columns.items()},
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            f"DataFrame Profile: {self.row_count} rows, {self.column_count} columns",
            "=" * 60,
        ]
        for name, col in self.columns.items():
            lines.append(f"\n{name}:")
            lines.append(f"  Type: {col.detected_type}")

            # Show sample values
            if col.sample_values:
                samples_str = str(col.sample_values[:3])
                lines.append(f"  Samples: {samples_str}")

            # Show scale modifier details
            if col.scale_info:
                lines.append(f"  Scale modifiers: {col.scale_info}")

            # Show percentage info details
            if col.percentage_info:
                lines.append(f"  Percentage: {col.percentage_info}")

            # Show unit info details
            if col.unit_info:
                lines.append(f"  Units: {col.unit_info}")

            # Show stdnum info details
            if col.stdnum_info:
                lines.append(f"  Standard numbers: {col.stdnum_info}")

            # Show country info details
            if col.country_info:
                lines.append(f"  Country formats: {col.country_info}")

            # Show null count only if significant
            if col.null_count > 0:
                null_pct = col.null_count / col.total_count * 100
                lines.append(f"  Nulls: {col.null_count}/{col.total_count} ({null_pct:.1f}%)")

            # Show suggestions
            if col.suggestions:
                for suggestion in col.suggestions:
                    lines.append(f"  Suggestion: {suggestion}")

        return "\n".join(lines)


def _sample_non_null(series: pd.Series, n: int = 5) -> list[Any]:
    """Get sample non-null values from a series."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return []
    sample = non_null.head(n).tolist()
    return [str(v)[:100] for v in sample]  # Truncate long values


def _detect_units(series: pd.Series, sample_size: int = 100) -> dict[str, Any] | None:
    """Detect units in column values."""
    sample = series.dropna().astype(str).head(sample_size)
    if len(sample) == 0:
        return None

    units_found: Counter[str] = Counter()
    dimensionalities: Counter[str] = Counter()

    for value in sample:
        parsed = parse_quantity(value)
        if parsed and parsed.unit != "dimensionless":
            units_found[parsed.unit] += 1
            dimensionalities[parsed.dimensionality] += 1
        else:
            # Try detecting unit in text
            unit = detect_unit_in_text(value)
            if unit:
                units_found[unit] += 1

    if not units_found:
        return None

    return {
        "units_detected": dict(units_found.most_common(5)),
        "dimensionalities": dict(dimensionalities.most_common(3)),
        "coverage": sum(units_found.values()) / len(sample),
    }


def _detect_scale_modifiers(series: pd.Series, sample_size: int = 100) -> dict[str, Any] | None:
    """Detect scale modifiers (MEO, million, etc.) in column values."""
    sample = series.dropna().astype(str).head(sample_size)
    if len(sample) == 0:
        return None

    modifiers_found: Counter[str] = Counter()

    for value in sample:
        modifier = detect_scale_modifier(value)
        if modifier:
            modifiers_found[modifier.name] += 1

    if not modifiers_found:
        return None

    return {
        "modifiers_detected": dict(modifiers_found.most_common(5)),
        "coverage": sum(modifiers_found.values()) / len(sample),
    }


def _detect_percentages(series: pd.Series, sample_size: int = 100) -> dict[str, Any] | None:
    """Detect percentage values in column (e.g., '50%', '0.5', '50 %')."""
    import re

    sample = series.dropna().astype(str).head(sample_size)
    if len(sample) == 0:
        return None

    # Pattern for percentage with % symbol
    percent_symbol_pattern = re.compile(r'^-?\d+(?:[.,]\d+)?\s*%$')
    # Pattern for decimal values that look like percentages (0.0-1.0 range)
    decimal_pattern = re.compile(r'^-?0?\.\d+$|^1\.0*$|^0$|^1$')

    percent_symbol_count = 0
    decimal_range_count = 0

    for value in sample:
        v = value.strip()

        # Check for % symbol
        if percent_symbol_pattern.match(v):
            percent_symbol_count += 1
        # Check for decimal values in 0-1 range (potential percentages already as decimals)
        elif decimal_pattern.match(v):
            try:
                num = float(v.replace(',', '.'))
                if 0 <= num <= 1:
                    decimal_range_count += 1
            except ValueError:
                pass

    total_matches = percent_symbol_count + decimal_range_count

    if total_matches == 0:
        return None

    coverage = total_matches / len(sample)
    if coverage < 0.3:  # Less than 30% match - probably not percentage data
        return None

    return {
        "with_symbol": percent_symbol_count,
        "decimal_range": decimal_range_count,
        "coverage": coverage,
        "format": "symbol" if percent_symbol_count > decimal_range_count else "decimal",
    }


def _detect_stdnum(series: pd.Series, sample_size: int = 100) -> dict[str, Any] | None:
    """Detect standard number formats (ISBN, IBAN, etc.)."""
    sample = series.dropna().astype(str).head(sample_size)
    if len(sample) == 0:
        return None

    types_found: Counter[str] = Counter()

    for value in sample:
        stdnum_type = detect_stdnum_type(value)
        if stdnum_type:
            types_found[stdnum_type] += 1

    if not types_found:
        return None

    return {
        "types_detected": dict(types_found.most_common(5)),
        "coverage": sum(types_found.values()) / len(sample),
    }


def _detect_country_codes(series: pd.Series, sample_size: int = 100) -> dict[str, Any] | None:
    """Detect country codes or names."""
    sample = series.dropna().astype(str).head(sample_size)
    if len(sample) == 0:
        return None

    matches = 0
    formats_found: Counter[str] = Counter()

    for value in sample:
        value = value.strip()
        info = lookup_country(value)
        if info:
            matches += 1
            # Determine input format
            if len(value) == 2:
                formats_found["alpha_2"] += 1
            elif len(value) == 3 and value.isalpha():
                formats_found["alpha_3"] += 1
            elif value.isdigit():
                formats_found["numeric"] += 1
            else:
                formats_found["name"] += 1

    if matches == 0:
        return None

    coverage = matches / len(sample)
    if coverage < 0.3:  # Less than 30% match - probably not country data
        return None

    return {
        "formats_detected": dict(formats_found.most_common()),
        "coverage": coverage,
    }


def _detect_column_type(
    series: pd.Series,
    sample_size: int = 100,
) -> tuple[DetectedType, dict[str, Any]]:
    """
    Detect the semantic type of a column.

    Uses specialized parsers from types.py for accurate detection of:
    - Booleans (including string representations like "yes", "no", "true", "false")
    - URLs (with proper validation)
    - Coordinates (lat/lon in various formats)
    - Dates (including string dates not yet parsed)

    Returns tuple of (detected_type, extra_info).
    """
    extra_info: dict[str, Any] = {}

    # Check for empty
    non_null = series.dropna()
    if len(non_null) == 0:
        return ("empty", extra_info)

    sample = non_null.astype(str).head(sample_size)

    # Check for emails (using integration)
    email_count = sum(1 for v in sample if is_valid_email(v))
    if email_count / len(sample) > 0.5:
        return ("email", {"coverage": email_count / len(sample)})

    # Check for phone numbers (using integration)
    phone_count = sum(1 for v in sample if validate_phone(v))
    if phone_count / len(sample) > 0.5:
        return ("phone", {"coverage": phone_count / len(sample)})

    # Check for boolean strings early (before country codes which may match "no", "yes")
    bool_parser = BooleanParser()
    bool_count = sum(1 for v in sample if bool_parser.parse_boolean(v) is not None)
    if bool_count / len(sample) > 0.5:
        return ("boolean", {"coverage": bool_count / len(sample)})

    # Check for standard numbers (ISBN, IBAN, etc.)
    stdnum_info = _detect_stdnum(series, sample_size)
    if stdnum_info and stdnum_info["coverage"] > 0.5:
        extra_info["stdnum_info"] = stdnum_info
        return ("stdnum", extra_info)

    # Check for country codes
    country_info = _detect_country_codes(series, sample_size)
    if country_info and country_info["coverage"] > 0.5:
        extra_info["country_info"] = country_info
        return ("country", extra_info)

    # Check for currency codes
    currency_matches = sum(1 for v in sample if lookup_currency(v.strip()))
    if currency_matches / len(sample) > 0.5:
        return ("currency", {"coverage": currency_matches / len(sample)})

    # Check for units
    unit_info = _detect_units(series, sample_size)
    if unit_info and unit_info["coverage"] > 0.3:
        extra_info["unit_info"] = unit_info
        return ("unit_quantity", extra_info)

    # Check for scale modifiers
    scale_info = _detect_scale_modifiers(series, sample_size)
    if scale_info and scale_info["coverage"] > 0.3:
        extra_info["scale_info"] = scale_info
        return ("scaled_number", extra_info)

    # Check for percentages (e.g., '50%', '0.5')
    percentage_info = _detect_percentages(series, sample_size)
    if percentage_info and percentage_info["coverage"] > 0.5:
        extra_info["percentage_info"] = percentage_info
        return ("percentage", extra_info)

    # Check pandas dtype first for already-typed columns
    if pd.api.types.is_numeric_dtype(series):
        return ("numeric", extra_info)

    if pd.api.types.is_datetime64_any_dtype(series):
        return ("date", extra_info)

    if pd.api.types.is_bool_dtype(series):
        return ("boolean", extra_info)

    # Use specialized parsers for string columns
    # Check for coordinates using CoordinateParser
    coord_parser = CoordinateParser()
    coord_count = sum(1 for v in sample if coord_parser.parse_coordinate(v) is not None)
    if coord_count / len(sample) > 0.5:
        return ("coordinate", {"coverage": coord_count / len(sample)})

    # Check for URLs using LinkNormalizer (more accurate than prefix check)
    link_normalizer = LinkNormalizer()
    url_count = sum(1 for v in sample if link_normalizer.is_valid_url(v))
    if url_count / len(sample) > 0.5:
        return ("url", {"coverage": url_count / len(sample)})

    # Check for date strings using DateNormalizer
    date_normalizer = DateNormalizer()
    date_count = sum(1 for v in sample if date_normalizer.parse_date(v) is not None)
    if date_count / len(sample) > 0.5:
        return ("date", {"coverage": date_count / len(sample)})

    # Default to text
    return ("text", extra_info)


def _generate_suggestions(col_profile: ColumnProfile) -> list[str]:
    """Generate normalization suggestions based on detected type."""
    suggestions = []

    if col_profile.detected_type == "unit_quantity":
        suggestions.append("Consider normalizing units to a standard (e.g., meters, kilograms)")

    if col_profile.detected_type == "scaled_number":
        suggestions.append("Consider expanding scale modifiers (e.g., '5 MEO' → 5000000)")

    if col_profile.detected_type == "percentage":
        if col_profile.percentage_info and col_profile.percentage_info.get("format") == "symbol":
            suggestions.append("Consider converting percentages to decimals (e.g., '50%' → 0.5)")
        else:
            suggestions.append("Detected decimal values in 0-1 range (likely percentages)")

    if col_profile.detected_type == "stdnum":
        suggestions.append("Consider validating and formatting standard numbers")

    if col_profile.detected_type == "country":
        suggestions.append("Consider normalizing to ISO 3166 alpha-2 codes")

    if col_profile.detected_type == "currency":
        suggestions.append("Consider normalizing to ISO 4217 codes")

    if col_profile.detected_type == "phone":
        suggestions.append("Consider normalizing to E.164 format")

    if col_profile.detected_type == "email":
        suggestions.append("Consider validating and normalizing email addresses")

    if col_profile.detected_type == "coordinate":
        suggestions.append("Consider normalizing to decimal degrees format")

    if col_profile.detected_type == "url":
        suggestions.append("Consider validating and normalizing URLs")

    if col_profile.detected_type == "boolean":
        suggestions.append("Consider converting to native boolean type")

    if col_profile.detected_type == "date":
        suggestions.append("Consider normalizing to ISO 8601 date format")

    if col_profile.null_count > 0:
        null_pct = col_profile.null_count / col_profile.total_count * 100
        if null_pct > 10:
            suggestions.append(f"High null rate ({null_pct:.1f}%) - consider handling missing values")

    return suggestions


def profile_column(series: pd.Series, sample_size: int = 100) -> ColumnProfile:
    """
    Profile a single column.

    Args:
        series: Pandas Series to profile
        sample_size: Number of values to sample for type detection

    Returns:
        ColumnProfile with detected information
    """
    detected_type, extra_info = _detect_column_type(series, sample_size)

    profile = ColumnProfile(
        name=str(series.name),
        dtype=str(series.dtype),
        total_count=len(series),
        null_count=int(series.isna().sum()),
        unique_count=int(series.nunique()),
        detected_type=detected_type,
        sample_values=_sample_non_null(series),
        unit_info=extra_info.get("unit_info"),
        scale_info=extra_info.get("scale_info"),
        percentage_info=extra_info.get("percentage_info"),
        stdnum_info=extra_info.get("stdnum_info"),
        country_info=extra_info.get("country_info"),
    )

    profile.suggestions = _generate_suggestions(profile)

    return profile


def profile_dataframe(
    df: pd.DataFrame,
    sample_size: int = 100,
) -> DataFrameProfile:
    """
    Profile an entire DataFrame.

    Args:
        df: DataFrame to profile
        sample_size: Number of values to sample per column for type detection

    Returns:
        DataFrameProfile with column-level information

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     "revenue": ["5 MEO", "10 MEO", "2.5 MEO"],
        ...     "country": ["DE", "FR", "US"],
        ... })
        >>> profile = profile_dataframe(df)
        >>> profile.columns["revenue"].detected_type
        'scaled_number'
        >>> profile.columns["country"].detected_type
        'country'
    """
    columns = {}

    for col_name in df.columns:
        columns[col_name] = profile_column(df[col_name], sample_size)

    return DataFrameProfile(
        row_count=len(df),
        column_count=len(df.columns),
        columns=columns,
    )
