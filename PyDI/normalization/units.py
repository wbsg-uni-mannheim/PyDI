"""
Unit detection, parsing, and conversion utilities for PyDI.

This module provides unit-related functionality using Pint as the backend
for physical unit conversions. It handles:
- Quantity parsing from text
- Unit detection and conversion
- Scale modifier expansion (via the scale module)

For physical units, this module delegates to Pint which provides:
- Comprehensive unit database
- Correct temperature conversions
- Compound unit support (m/s, km/h, etc.)
"""

from __future__ import annotations

import logging

import pandas as pd

from .integrations.pint_units import (
    ParsedQuantity,
    parse_quantity as pint_parse_quantity,
    convert_units as pint_convert_units,
    normalize_to_base,
    get_unit_dimensionality,
    is_compatible,
    is_valid_unit,
    detect_unit_in_text,
    get_compatible_units,
    list_units,
)
from .scale import (
    detect_scale_modifier,
    expand_scale,
    parse_scaled_number,
    ScaleModifier,
    ScaleResult,
)

logger = logging.getLogger(__name__)


def parse_quantity(text: str) -> ParsedQuantity | None:
    """
    Parse a quantity string like '5 km' or '100 miles'.

    This delegates to Pint for parsing.

    Args:
        text: String containing a number and unit

    Returns:
        ParsedQuantity with magnitude, unit, and dimensionality, or None if parsing fails

    Examples:
        >>> result = parse_quantity("5 kilometers")
        >>> result.magnitude
        5.0
        >>> result.unit
        'kilometer'
    """
    return pint_parse_quantity(text)


def convert_units(value: float, from_unit: str, to_unit: str) -> float | None:
    """
    Convert a value from one unit to another.

    This delegates to Pint for conversion.

    Args:
        value: Numeric value to convert
        from_unit: Source unit (e.g., "km", "miles", "celsius")
        to_unit: Target unit (e.g., "m", "km", "fahrenheit")

    Returns:
        Converted value, or None if conversion fails

    Examples:
        >>> convert_units(5, "km", "m")
        5000.0
        >>> convert_units(100, "fahrenheit", "celsius")
        37.77...
    """
    return pint_convert_units(value, from_unit, to_unit)


def detect_unit(text: str) -> str | None:
    """
    Detect a unit in text.

    Args:
        text: Text that may contain a unit

    Returns:
        Detected unit string, or None

    Examples:
        >>> detect_unit("The distance is 5 km")
        'kilometer'
    """
    return detect_unit_in_text(text)


def normalize_quantity(
    text: str,
    target_unit: str | None = None,
    expand_scales: bool = True,
) -> tuple[float, str] | None:
    """
    Normalize a quantity with optional unit conversion and scale expansion.

    Args:
        text: Text containing a quantity (e.g., "5 km", "10 MEO")
        target_unit: Target unit for conversion (if None, keeps original or converts to base)
        expand_scales: Whether to expand scale modifiers (MEO, million, etc.)

    Returns:
        Tuple of (value, unit) or None if parsing fails

    Examples:
        >>> normalize_quantity("5 km", target_unit="m")
        (5000.0, 'meter')
        >>> normalize_quantity("10 MEO", expand_scales=True)
        (10000000.0, 'dimensionless')
    """
    if not text or not isinstance(text, str):
        return None

    text = text.strip()

    # First, try to parse as a quantity with units
    parsed = parse_quantity(text)

    if parsed:
        value = parsed.magnitude
        unit = parsed.unit

        # Check for scale modifiers in the text
        if expand_scales:
            modifier = detect_scale_modifier(text)
            if modifier:
                value *= modifier.multiplier

        # Convert to target unit if specified
        if target_unit and unit != "dimensionless":
            converted = convert_units(value, unit, target_unit)
            if converted is not None:
                return (converted, target_unit)

        return (value, unit)

    # If Pint couldn't parse, try scale modifier parsing
    if expand_scales:
        result = parse_scaled_number(text)
        if result:
            value, modifier = result
            return (value, "dimensionless")

    return None


def normalize_column(
    series: pd.Series,
    target_unit: str | None = None,
    expand_scales: bool = True,
) -> pd.DataFrame:
    """
    Normalize a column containing quantities.

    Args:
        series: Pandas Series with quantity values
        target_unit: Target unit for conversion
        expand_scales: Whether to expand scale modifiers

    Returns:
        DataFrame with columns: value, unit, original

    Examples:
        >>> import pandas as pd
        >>> s = pd.Series(["5 km", "10 miles", "2.5 MEO"])
        >>> df = normalize_column(s, target_unit="m")
    """
    results = []

    for val in series:
        original = str(val) if pd.notna(val) else None
        normalized = normalize_quantity(val, target_unit, expand_scales) if pd.notna(val) else None

        if normalized:
            results.append({
                "value": normalized[0],
                "unit": normalized[1],
                "original": original,
            })
        else:
            results.append({
                "value": None,
                "unit": None,
                "original": original,
            })

    return pd.DataFrame(results, index=series.index)


def get_dimensionality(unit: str) -> str | None:
    """
    Get the dimensionality of a unit.

    Args:
        unit: Unit string

    Returns:
        Dimensionality string (e.g., "[length]")

    Examples:
        >>> get_dimensionality("km")
        '[length]'
        >>> get_dimensionality("m/s")
        '[length] / [time]'
    """
    return get_unit_dimensionality(unit)


def are_compatible(unit1: str, unit2: str) -> bool:
    """
    Check if two units can be converted between each other.

    Args:
        unit1: First unit
        unit2: Second unit

    Returns:
        True if compatible

    Examples:
        >>> are_compatible("km", "miles")
        True
        >>> are_compatible("kg", "miles")
        False
    """
    return is_compatible(unit1, unit2)


def list_compatible_units(unit: str) -> list[str]:
    """
    List all units compatible with the given unit.

    Args:
        unit: Unit to find compatible units for

    Returns:
        List of compatible unit names

    Examples:
        >>> "mile" in list_compatible_units("meter")
        True
    """
    return sorted(get_compatible_units(unit))


# Re-export from integrations for convenience
__all__ = [
    # Main functions
    "parse_quantity",
    "convert_units",
    "detect_unit",
    "normalize_quantity",
    "normalize_column",
    "get_dimensionality",
    "are_compatible",
    "list_compatible_units",
    # From integrations
    "ParsedQuantity",
    "normalize_to_base",
    "is_valid_unit",
    "list_units",
    # Scale handling
    "detect_scale_modifier",
    "expand_scale",
    "parse_scaled_number",
    "ScaleModifier",
    "ScaleResult",
]
