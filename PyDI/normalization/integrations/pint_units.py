"""
Pint integration for physical unit conversions.

This module wraps the Pint library to provide:
- Quantity parsing from strings ("5 km" -> (5000, "meter"))
- Unit conversion (km -> miles)
- Base unit normalization (to SI units)
- Unit compatibility checking
- Unit discovery and listing
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pint

logger = logging.getLogger(__name__)

# Create a singleton unit registry
_ureg = pint.UnitRegistry()

# Enable string formatting for quantities
_ureg.formatter.default_format = "~P"  # Short pretty format


@dataclass
class ParsedQuantity:
    """Result of parsing a quantity string."""

    magnitude: float
    unit: str
    original: str
    dimensionality: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "magnitude": self.magnitude,
            "unit": self.unit,
            "original": self.original,
            "dimensionality": self.dimensionality,
        }


def get_registry() -> pint.UnitRegistry:
    """Get the shared Pint unit registry."""
    return _ureg


def parse_quantity(text: str) -> ParsedQuantity | None:
    """
    Parse a quantity string like '5 km' or '100 miles'.

    Args:
        text: String containing a number and unit

    Returns:
        ParsedQuantity with magnitude, unit, and dimensionality, or None if parsing fails

    Examples:
        >>> parse_quantity("5 kilometers")
        ParsedQuantity(magnitude=5.0, unit='kilometer', ...)
        >>> parse_quantity("100 mph")
        ParsedQuantity(magnitude=100.0, unit='mile / hour', ...)
        >>> parse_quantity("25 °C")
        ParsedQuantity(magnitude=25.0, unit='degree_Celsius', ...)
    """
    if not text or not isinstance(text, str):
        return None

    text = text.strip()
    if not text:
        return None

    try:
        q = _ureg.parse_expression(text)
        # Handle dimensionless quantities
        if isinstance(q, (int, float)):
            return ParsedQuantity(
                magnitude=float(q),
                unit="dimensionless",
                original=text,
                dimensionality="dimensionless",
            )
        return ParsedQuantity(
            magnitude=float(q.magnitude),
            unit=str(q.units),
            original=text,
            dimensionality=str(q.dimensionality),
        )
    except (pint.UndefinedUnitError, pint.DimensionalityError, ValueError) as e:
        logger.debug(f"Failed to parse quantity '{text}': {e}")
        return None
    except Exception as e:
        logger.debug(f"Unexpected error parsing quantity '{text}': {e}")
        return None


def convert_units(
    value: float, from_unit: str, to_unit: str
) -> float | None:
    """
    Convert a value from one unit to another.

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
        >>> convert_units(1, "kg", "lb")
        2.204...
    """
    try:
        # Use Quantity constructor to properly handle offset units like temperature
        # (see https://pint.readthedocs.io/en/stable/user/nonmult.html)
        q = _ureg.Quantity(value, from_unit)
        result = q.to(to_unit)
        return float(result.magnitude)
    except (pint.UndefinedUnitError, pint.DimensionalityError) as e:
        logger.debug(f"Failed to convert {value} {from_unit} to {to_unit}: {e}")
        return None
    except Exception as e:
        logger.debug(f"Unexpected error in conversion: {e}")
        return None


def normalize_to_base(value: float, unit: str) -> tuple[float, str] | None:
    """
    Convert a value to its SI base units.

    Args:
        value: Numeric value
        unit: Source unit

    Returns:
        Tuple of (magnitude, base_unit_string), or None if conversion fails

    Examples:
        >>> normalize_to_base(5, "km")
        (5000.0, 'meter')
        >>> normalize_to_base(1, "hour")
        (3600.0, 'second')
    """
    try:
        q = value * _ureg(unit)
        base = q.to_base_units()
        return (float(base.magnitude), str(base.units))
    except (pint.UndefinedUnitError, pint.DimensionalityError) as e:
        logger.debug(f"Failed to normalize {value} {unit}: {e}")
        return None
    except Exception as e:
        logger.debug(f"Unexpected error in normalization: {e}")
        return None


def get_unit_dimensionality(unit: str) -> str | None:
    """
    Get the dimensionality of a unit.

    Args:
        unit: Unit string (e.g., "km", "m/s", "kg")

    Returns:
        Dimensionality string (e.g., "[length]", "[length] / [time]"), or None

    Examples:
        >>> get_unit_dimensionality("km")
        '[length]'
        >>> get_unit_dimensionality("m/s")
        '[length] / [time]'
        >>> get_unit_dimensionality("kg")
        '[mass]'
    """
    try:
        u = _ureg(unit)
        return str(u.dimensionality)
    except pint.UndefinedUnitError:
        return None


def is_compatible(unit1: str, unit2: str) -> bool:
    """
    Check if two units are compatible (can be converted between).

    Args:
        unit1: First unit
        unit2: Second unit

    Returns:
        True if units are compatible, False otherwise

    Examples:
        >>> is_compatible("km", "miles")
        True
        >>> is_compatible("kg", "lb")
        True
        >>> is_compatible("km", "kg")
        False
    """
    try:
        u1 = _ureg(unit1)
        u2 = _ureg(unit2)
        return u1.is_compatible_with(u2)
    except pint.UndefinedUnitError:
        return False


def get_compatible_units(unit_or_dimensionality: str) -> frozenset[str]:
    """
    Get all units compatible with a given unit or dimensionality.

    Uses Pint's built-in get_compatible_units method.

    Args:
        unit_or_dimensionality: Either a unit string (e.g., "meter") or
                                a dimensionality string (e.g., "[length]")

    Returns:
        Frozenset of compatible unit names

    Examples:
        >>> "mile" in get_compatible_units("meter")
        True
        >>> "kilogram" in get_compatible_units("[mass]")
        True
    """
    try:
        compatible = _ureg.get_compatible_units(unit_or_dimensionality)
        # Extract unit names from Unit objects
        return frozenset(str(u) for u in compatible)
    except (pint.UndefinedUnitError, KeyError):
        return frozenset()


def list_units(dimensionality: str | None = None) -> list[str]:
    """
    List available units, optionally filtered by dimensionality.

    Args:
        dimensionality: Filter by dimensionality (e.g., "[length]", "[mass]")
                       If None, returns units for common dimensionalities

    Returns:
        List of unit names

    Examples:
        >>> "meter" in list_units("[length]")
        True
        >>> "kilogram" in list_units("[mass]")
        True
    """
    if dimensionality is not None:
        return sorted(get_compatible_units(dimensionality))

    # Return units for common dimensionalities
    common_dims = [
        "[length]",
        "[mass]",
        "[time]",
        "[temperature]",
        "[substance]",
        "[current]",
        "[luminosity]",
    ]
    all_units: set[str] = set()
    for dim in common_dims:
        all_units.update(get_compatible_units(dim))
    return sorted(all_units)


def list_dimensionalities() -> list[str]:
    """
    List common dimensionalities supported by Pint.

    Returns:
        List of dimensionality strings

    Examples:
        >>> "[length]" in list_dimensionalities()
        True
        >>> "[mass]" in list_dimensionalities()
        True
    """
    return [
        "[length]",
        "[mass]",
        "[time]",
        "[temperature]",
        "[current]",
        "[substance]",
        "[luminosity]",
        "[length] ** 2",  # area
        "[length] ** 3",  # volume
        "[length] / [time]",  # velocity
        "[length] / [time] ** 2",  # acceleration
        "[mass] * [length] ** 2 / [time] ** 2",  # energy
        "[mass] * [length] ** 2 / [time] ** 3",  # power
        "[mass] / [length] / [time] ** 2",  # pressure
    ]


def is_valid_unit(unit: str) -> bool:
    """
    Check if a string is a valid unit in Pint's registry.

    Args:
        unit: Unit string to check

    Returns:
        True if valid, False otherwise

    Examples:
        >>> is_valid_unit("meter")
        True
        >>> is_valid_unit("not_a_unit")
        False
    """
    if not unit or not isinstance(unit, str):
        return False
    try:
        _ureg(unit)
        return True
    except (pint.UndefinedUnitError, pint.DimensionalityError, AssertionError, Exception):
        return False


def detect_unit_in_text(text: str) -> str | None:
    """
    Attempt to detect a unit mentioned in text using Pint's parser.

    This tries to extract the unit portion from a quantity string.

    Args:
        text: Text that may contain a unit

    Returns:
        Detected unit string, or None

    Examples:
        >>> detect_unit_in_text("The distance is 5 km")
        'kilometer'
        >>> detect_unit_in_text("Speed: 100 mph")
        'mile / hour'
    """
    if not text:
        return None

    # Try parsing the full text as a quantity
    parsed = parse_quantity(text)
    if parsed and parsed.unit != "dimensionless":
        return parsed.unit

    # Try to find numeric + unit patterns
    import re
    # Look for number followed by potential unit
    pattern = r"[\d.,]+\s*([a-zA-Z°/²³]+(?:\s*/\s*[a-zA-Z°²³]+)?)"
    matches = re.findall(pattern, text)

    for match in matches:
        if is_valid_unit(match.strip()):
            try:
                u = _ureg(match.strip())
                return str(u.units)
            except Exception:
                continue

    return None
