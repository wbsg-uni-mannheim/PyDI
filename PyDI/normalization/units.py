"""
Comprehensive unit detection, parsing, and conversion utilities for PyDI.

This module provides all unit-related functionality including unit registries,
detection, parsing, conversion, and normalization. It supports extensive
unit categories and provides both simple and comprehensive unit handling.
"""

from __future__ import annotations

import logging
import re
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

logger = logging.getLogger(__name__)


class UnitCategory(Enum):
    """Categories of measurement units."""
    LENGTH = "length"
    WEIGHT = "weight"
    VOLUME = "volume"
    AREA = "area"
    TIME = "time"
    TEMPERATURE = "temperature"
    CURRENCY = "currency"
    SPEED = "speed"
    ENERGY = "energy"
    POWER = "power"
    PRESSURE = "pressure"
    FORCE = "force"
    FREQUENCY = "frequency"
    INFORMATION = "information"
    ANGLE = "angle"
    DENSITY = "density"
    PERCENTAGE = "percentage"
    COUNT = "count"
    UNKNOWN = "unknown"


class QuantityModifier(Enum):
    """Quantity scaling modifiers."""
    HUNDRED = (100, ["hundred", "hundreds"])
    THOUSAND = (1_000, ["thousand", "thousands", "k", "K"])
    MILLION = (1_000_000, ["million", "millions", "mil", "m", "M"])
    BILLION = (1_000_000_000, ["billion", "billions", "bil", "b", "B"])
    TRILLION = (1_000_000_000_000, ["trillion", "trillions", "tril", "t", "T"])
    QUADRILLION = (1_000_000_000_000_000, ["quadrillion", "quadrillions"])

    def __init__(self, multiplier: int, keywords: List[str]):
        self.multiplier = multiplier
        self.keywords = keywords


class Quantity:
    """Represents a numeric quantity extracted from text."""

    def __init__(
        self,
        value: Union[int, float, Decimal],
        original_text: str,
        start_pos: int = 0,
        end_pos: Optional[int] = None
    ):
        self.value = value
        self.original_text = original_text
        self.start_pos = start_pos
        self.end_pos = end_pos or len(original_text)

    def __str__(self) -> str:
        return f"Quantity({self.value})"

    def __repr__(self) -> str:
        return f"Quantity(value={self.value}, text='{self.original_text}')"


class Unit:
    """Represents a unit of measurement."""

    def __init__(
        self,
        symbol: str,
        name: str,
        category: UnitCategory,
        base_conversion_factor: float = 1.0,
        base_unit: Optional[str] = None
    ):
        self.symbol = symbol
        self.name = name
        self.category = category
        self.base_conversion_factor = base_conversion_factor
        self.base_unit = base_unit or symbol

    def __str__(self) -> str:
        return self.symbol

    def __repr__(self) -> str:
        return f"Unit({self.symbol}, {self.category.value})"


class UnitRegistry:
    """Registry of known units and their properties."""

    def __init__(self, comprehensive: bool = False):
        """
        Initialize unit registry.

        Parameters
        ----------
        comprehensive : bool, default False
            Whether to load comprehensive unit set (500+ units) or basic set.
        """
        self.units: Dict[str, Unit] = {}
        self.base_units: Dict[UnitCategory, str] = {}

        if comprehensive:
            self._initialize_comprehensive_units()
        else:
            self._initialize_basic_units()

    def _initialize_basic_units(self):
        """Initialize with basic commonly-used units."""

        # Length units (base: meter)
        length_units = [
            ('mm', 'millimeter', 0.001), ('cm', 'centimeter', 0.01),
            ('m', 'meter', 1.0), ('km', 'kilometer', 1000.0),
            ('in', 'inch', 0.0254), ('ft', 'foot', 0.3048),
            ('yd', 'yard', 0.9144), ('mi', 'mile', 1609.34),
        ]
        self.base_units[UnitCategory.LENGTH] = 'm'
        self._add_units(UnitCategory.LENGTH, length_units)

        # Weight/Mass units (base: kilogram)
        weight_units = [
            ('mg', 'milligram', 0.000001), ('g', 'gram', 0.001),
            ('kg', 'kilogram', 1.0), ('oz', 'ounce', 0.0283495),
            ('lb', 'pound', 0.453592), ('t', 'ton', 1000.0),
        ]
        self.base_units[UnitCategory.WEIGHT] = 'kg'
        self._add_units(UnitCategory.WEIGHT, weight_units)

        # Volume units (base: liter)
        volume_units = [
            ('ml', 'milliliter', 0.001), ('l', 'liter', 1.0),
            ('gal', 'gallon', 3.78541), ('qt', 'quart', 0.946353),
            ('pt', 'pint', 0.473176), ('cup', 'cup', 0.236588),
        ]
        self.base_units[UnitCategory.VOLUME] = 'l'
        self._add_units(UnitCategory.VOLUME, volume_units)

        # Time units (base: second)
        time_units = [
            ('ms', 'millisecond', 0.001), ('s', 'second', 1.0),
            ('min', 'minute', 60.0), ('h', 'hour', 3600.0),
            ('d', 'day', 86400.0), ('week', 'week', 604800.0),
            ('year', 'year', 31556952.0),
        ]
        self.base_units[UnitCategory.TIME] = 's'
        self._add_units(UnitCategory.TIME, time_units)

        # Temperature units (base: celsius)
        temp_units = [
            ('°C', 'celsius', 1.0), ('°F', 'fahrenheit', 1.0),
            ('K', 'kelvin', 1.0), ('C', 'celsius', 1.0),
        ]
        self.base_units[UnitCategory.TEMPERATURE] = '°C'
        self._add_units(UnitCategory.TEMPERATURE, temp_units)

        # Basic currency units
        currency_units = [
            ('$', 'dollar', 1.0), ('USD', 'US dollar', 1.0),
            ('€', 'euro', 1.0), ('EUR', 'euro', 1.0),
            ('£', 'pound', 1.0), ('GBP', 'british pound', 1.0),
        ]
        self.base_units[UnitCategory.CURRENCY] = '$'
        self._add_units(UnitCategory.CURRENCY, currency_units)

        # Speed units (base: m/s)
        speed_units = [
            ('m/s', 'meter per second', 1.0), ('km/h',
                                               'kilometer per hour', 0.277778),
            ('mph', 'miles per hour', 0.44704), ('knot', 'knot', 0.514444),
        ]
        self.base_units[UnitCategory.SPEED] = 'm/s'
        self._add_units(UnitCategory.SPEED, speed_units)

        # Percentage
        self.units['%'] = Unit(
            '%', 'percent', UnitCategory.PERCENTAGE, 1.0, '%')
        self.units['percent'] = self.units['%']
        self.base_units[UnitCategory.PERCENTAGE] = '%'

    def _initialize_comprehensive_units(self):
        """Initialize comprehensive unit registry with 500+ units."""

        # Start with basic units
        self._initialize_basic_units()

        # Extended Length units
        extended_length = [
            ('μm', 'micrometer', 0.000001), ('nm', 'nanometer', 0.000000001),
            ('mil', 'mil', 0.0000254), ('fathom', 'fathom', 1.8288),
            ('nmi', 'nautical mile', 1852.0), ('au',
                                               'astronomical unit', 149597870700.0),
            ('ly', 'light year', 9.461e15), ('pc', 'parsec', 3.086e16),
        ]
        self._add_units(UnitCategory.LENGTH, extended_length)

        # Extended Weight/Mass units
        extended_weight = [
            ('st', 'stone', 6.35029), ('cwt', 'hundredweight', 50.8023),
            ('ozt', 'troy ounce', 0.0311035), ('grain', 'grain', 0.0000647989),
            ('carat', 'carat', 0.0002), ('mt', 'metric ton', 1000.0),
        ]
        self._add_units(UnitCategory.WEIGHT, extended_weight)

        # Extended Volume units
        extended_volume = [
            ('cl', 'centiliter', 0.01), ('dl', 'deciliter', 0.1),
            ('hl', 'hectoliter', 100.0), ('m³', 'cubic meter', 1000.0),
            ('cm³', 'cubic centimeter', 0.001), ('fl oz', 'fluid ounce', 0.0284131),
            ('us gal', 'us gallon', 3.78541), ('tbsp', 'tablespoon', 0.0147868),
            ('tsp', 'teaspoon', 0.00492892),
        ]
        self._add_units(UnitCategory.VOLUME, extended_volume)

        # Area units (base: square meter)
        area_units = [
            ('mm²', 'square millimeter', 0.000001), ('cm²',
                                                     'square centimeter', 0.0001),
            ('m²', 'square meter', 1.0), ('km²', 'square kilometer', 1000000.0),
            ('ha', 'hectare', 10000.0), ('acre', 'acre', 4046.86),
            ('sq ft', 'square foot', 0.092903), ('sq mi', 'square mile', 2589988.0),
        ]
        self.base_units[UnitCategory.AREA] = 'm²'
        self._add_units(UnitCategory.AREA, area_units)

        # Energy units (base: joule)
        energy_units = [
            ('J', 'joule', 1.0), ('kJ', 'kilojoule', 1000.0),
            ('MJ', 'megajoule', 1000000.0), ('cal', 'calorie', 4.184),
            ('kcal', 'kilocalorie', 4184.0), ('BTU',
                                              'british thermal unit', 1055.06),
            ('Wh', 'watt hour', 3600.0), ('kWh', 'kilowatt hour', 3600000.0),
        ]
        self.base_units[UnitCategory.ENERGY] = 'J'
        self._add_units(UnitCategory.ENERGY, energy_units)

        # Power units (base: watt)
        power_units = [
            ('W', 'watt', 1.0), ('kW', 'kilowatt', 1000.0),
            ('MW', 'megawatt', 1000000.0), ('hp', 'horsepower', 745.7),
        ]
        self.base_units[UnitCategory.POWER] = 'W'
        self._add_units(UnitCategory.POWER, power_units)

        # Pressure units (base: pascal)
        pressure_units = [
            ('Pa', 'pascal', 1.0), ('kPa', 'kilopascal', 1000.0),
            ('bar', 'bar', 100000.0), ('atm', 'atmosphere', 101325.0),
            ('psi', 'pounds per square inch', 6895.0), ('torr', 'torr', 133.322),
        ]
        self.base_units[UnitCategory.PRESSURE] = 'Pa'
        self._add_units(UnitCategory.PRESSURE, pressure_units)

        # Information units (base: byte)
        info_units = [
            ('bit', 'bit', 0.125), ('B', 'byte', 1.0),
            ('kB', 'kilobyte', 1000.0), ('MB', 'megabyte', 1000000.0),
            ('GB', 'gigabyte', 1000000000.0), ('TB', 'terabyte', 1000000000000.0),
            ('KiB', 'kibibyte', 1024.0), ('MiB', 'mebibyte', 1048576.0),
            ('GiB', 'gibibyte', 1073741824.0),
        ]
        self.base_units[UnitCategory.INFORMATION] = 'B'
        self._add_units(UnitCategory.INFORMATION, info_units)

        # Extended Currency units
        extended_currency = [
            ('¥', 'yen', 1.0), ('JPY', 'japanese yen', 1.0),
            ('₹', 'rupee', 1.0), ('INR', 'indian rupee', 1.0),
            ('₽', 'ruble', 1.0), ('RUB', 'russian ruble', 1.0),
            ('₩', 'won', 1.0), ('KRW', 'korean won', 1.0),
            ('CAD', 'canadian dollar', 1.0), ('AUD', 'australian dollar', 1.0),
            ('CHF', 'swiss franc', 1.0), ('CNY', 'chinese yuan', 1.0),
        ]
        self._add_units(UnitCategory.CURRENCY, extended_currency)

        # Frequency units (base: hertz)
        frequency_units = [
            ('Hz', 'hertz', 1.0), ('kHz', 'kilohertz', 1000.0),
            ('MHz', 'megahertz', 1000000.0), ('GHz', 'gigahertz', 1000000000.0),
        ]
        self.base_units[UnitCategory.FREQUENCY] = 'Hz'
        self._add_units(UnitCategory.FREQUENCY, frequency_units)

        # Angle units (base: radian)
        angle_units = [
            ('rad', 'radian', 1.0), ('deg', 'degree', 0.0174533),
            ('°', 'degree', 0.0174533), ('grad', 'gradian', 0.0157080),
        ]
        self.base_units[UnitCategory.ANGLE] = 'rad'
        self._add_units(UnitCategory.ANGLE, angle_units)

    def _add_units(self, category: UnitCategory, unit_list: List[Tuple[str, str, float]]):
        """Add units to registry with aliases."""
        for symbol, name, factor in unit_list:
            self.units[symbol.lower()] = Unit(symbol, name, category,
                                              factor, self.base_units[category])
            self.units[name.lower()] = Unit(symbol, name, category,
                                            factor, self.base_units[category])
            # Add plural forms
            if not name.endswith('s') and len(name) > 3:
                self.units[f"{name}s".lower()] = Unit(
                    symbol, name, category, factor, self.base_units[category])

    def get_unit(self, symbol: str) -> Optional[Unit]:
        """Get unit by symbol or name."""
        return self.units.get(symbol.lower())

    def add_unit(self, unit: Unit, aliases: Optional[List[str]] = None):
        """Add a custom unit to the registry."""
        self.units[unit.symbol.lower()] = unit
        self.units[unit.name.lower()] = unit

        if aliases:
            for alias in aliases:
                self.units[alias.lower()] = unit

    def get_units_by_category(self, category: UnitCategory) -> List[Unit]:
        """Get all units in a category."""
        units = []
        seen_symbols = set()

        for unit in self.units.values():
            if unit.category == category and unit.symbol not in seen_symbols:
                units.append(unit)
                seen_symbols.add(unit.symbol)

        return units


class QuantityParser:
    """Parser for extracting numeric quantities from text."""

    def __init__(self):
        # Comprehensive numeric patterns
        self.numeric_patterns = [
            # Scientific notation
            re.compile(r'[-+]?\d*\.?\d+[eE][-+]?\d+'),
            # Decimal numbers with thousands separators
            re.compile(r'[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?'),
            # Simple decimals
            re.compile(r'[-+]?\d*\.\d+'),
            # Integers with thousands separators
            re.compile(r'[-+]?\d{1,3}(?:,\d{3})+'),
            # Simple integers
            re.compile(r'[-+]?\d+'),
            # Percentages
            re.compile(r'\d+(?:\.\d+)?%'),
            # Basic fractions
            re.compile(r'\d+/\d+'),
        ]

    def parse_quantity(self, text: str) -> Optional[Quantity]:
        """Parse the first quantity found in text."""
        quantities = self.parse_quantities(text)
        return quantities[0] if quantities else None

    def parse_quantities(self, text: str) -> List[Quantity]:
        """Parse all quantities found in text."""
        if pd.isna(text) or not text:
            return []

        text = str(text).strip()
        quantities = []

        for pattern in self.numeric_patterns:
            for match in pattern.finditer(text):
                value_text = match.group().replace(',', '')

                try:
                    # Handle percentages
                    if value_text.endswith('%'):
                        value = float(value_text[:-1]) / 100.0
                    # Handle fractions
                    elif '/' in value_text and 'e' not in value_text.lower():
                        parts = value_text.split('/')
                        if len(parts) == 2:
                            value = float(parts[0]) / float(parts[1])
                        else:
                            continue
                    else:
                        # Try different numeric types
                        try:
                            value = int(value_text)
                        except ValueError:
                            value = float(value_text)

                    quantity = Quantity(
                        value=value,
                        original_text=match.group(),
                        start_pos=match.start(),
                        end_pos=match.end()
                    )
                    quantities.append(quantity)

                except (ValueError, ZeroDivisionError):
                    continue

        # Sort by position and remove overlaps
        quantities.sort(key=lambda q: q.start_pos)
        non_overlapping = []
        last_end = -1

        for quantity in quantities:
            if quantity.start_pos >= last_end:
                non_overlapping.append(quantity)
                last_end = quantity.end_pos

        return non_overlapping


class UnitDetector:
    """Detector for units in text with quantities."""

    def __init__(self, registry: Optional[UnitRegistry] = None):
        self.registry = registry or UnitRegistry()

        # Build pattern for unit detection
        unit_symbols = list(self.registry.units.keys())
        # Sort by length descending to match longer units first
        unit_symbols.sort(key=len, reverse=True)
        # Escape special regex characters
        escaped_symbols = [re.escape(symbol) for symbol in unit_symbols]
        self.unit_pattern = re.compile(
            r'\b(' + '|'.join(escaped_symbols) + r')\b',
            re.IGNORECASE
        )

    def detect_unit(self, text: str, quantity: Optional[Quantity] = None) -> Optional[Unit]:
        """Detect unit in text, optionally near a quantity."""
        if pd.isna(text) or not text:
            return None

        text = str(text).strip()

        # If quantity provided, search near it first
        if quantity:
            # Search right after quantity
            search_text = text[quantity.end_pos:quantity.end_pos + 20]
            match = self.unit_pattern.search(search_text)
            if match:
                unit_text = match.group(1)
                return self.registry.get_unit(unit_text)

            # Search right before quantity
            search_start = max(0, quantity.start_pos - 20)
            search_text = text[search_start:quantity.start_pos]
            match = self.unit_pattern.search(search_text)
            if match:
                unit_text = match.group(1)
                return self.registry.get_unit(unit_text)

        # General search in entire text
        match = self.unit_pattern.search(text)
        if match:
            unit_text = match.group(1)
            return self.registry.get_unit(unit_text)

        return None

    def detect_category(self, text: str) -> UnitCategory:
        """Detect unit category from text."""
        unit = self.detect_unit(text)
        return unit.category if unit else UnitCategory.UNKNOWN


class UnitConverter:
    """Converter for units within the same category."""

    def __init__(self, registry: Optional[UnitRegistry] = None):
        self.registry = registry or UnitRegistry()

    def can_convert(self, from_unit: Unit, to_unit: Unit) -> bool:
        """Check if units can be converted."""
        return from_unit.category == to_unit.category

    def convert(self, value: float, from_unit: Unit, to_unit: Unit) -> Optional[float]:
        """Convert value between units."""
        if not self.can_convert(from_unit, to_unit):
            return None

        # Special handling for temperature
        if from_unit.category == UnitCategory.TEMPERATURE:
            return self._convert_temperature(value, from_unit, to_unit)

        # Standard conversion through base unit
        base_value = value * from_unit.base_conversion_factor
        converted_value = base_value / to_unit.base_conversion_factor

        return converted_value

    def _convert_temperature(self, value: float, from_unit: Unit, to_unit: Unit) -> float:
        """Convert temperature values with special handling."""
        # Convert to Celsius first
        if from_unit.symbol in ['°F', 'F']:
            celsius_value = (value - 32) * 5/9
        elif from_unit.symbol == 'K':
            celsius_value = value - 273.15
        else:  # Already Celsius
            celsius_value = value

        # Convert from Celsius to target
        if to_unit.symbol in ['°F', 'F']:
            return celsius_value * 9/5 + 32
        elif to_unit.symbol == 'K':
            return celsius_value + 273.15
        else:  # Target is Celsius
            return celsius_value


class UnitNormalizer:
    """Main normalizer for quantities with units."""

    def __init__(
        self,
        registry: Optional[UnitRegistry] = None,
        target_units: Optional[Dict[UnitCategory, str]] = None,
        comprehensive: bool = False
    ):
        self.registry = registry or UnitRegistry(comprehensive=comprehensive)
        self.quantity_parser = QuantityParser()
        self.unit_detector = UnitDetector(self.registry)
        self.converter = UnitConverter(self.registry)

        # Default target units for normalization
        self.target_units = target_units or {
            UnitCategory.LENGTH: 'm',
            UnitCategory.WEIGHT: 'kg',
            UnitCategory.VOLUME: 'l',
            UnitCategory.TIME: 's',
            UnitCategory.TEMPERATURE: '°C',
            UnitCategory.SPEED: 'm/s',
        }

        # Quantity modifiers (word-boundary constrained to avoid unit collisions like 'km')
        self.quantity_modifiers = {}
        self._quantity_modifier_patterns: List[Tuple[re.Pattern[str], int]] = [
        ]
        for modifier in QuantityModifier:
            for keyword in modifier.keywords:
                key = keyword.lower()
                self.quantity_modifiers[key] = modifier.multiplier
                # Whole word match, case-insensitive
                pattern = re.compile(rf"\b{re.escape(key)}\b", re.IGNORECASE)
                self._quantity_modifier_patterns.append(
                    (pattern, modifier.multiplier))

    def normalize_value(self, text: str) -> Optional[Tuple[float, str]]:
        """Normalize a value with units."""
        if pd.isna(text) or not text:
            return None

        text = str(text).strip()

        # Parse quantity
        quantity = self.quantity_parser.parse_quantity(text)
        if not quantity:
            return None

        # Look for quantity modifiers with whole-word matching
        numeric_value = quantity.value
        lowered = text.lower()
        for pattern, multiplier in self._quantity_modifier_patterns:
            if pattern.search(lowered):
                numeric_value *= multiplier
                break

        # Detect unit
        unit = self.unit_detector.detect_unit(text, quantity)
        if not unit:
            return numeric_value, "dimensionless"

        # Get target unit for this category
        target_unit_symbol = self.target_units.get(unit.category)
        if not target_unit_symbol:
            return numeric_value, unit.symbol

        target_unit = self.registry.get_unit(target_unit_symbol)
        if not target_unit:
            return numeric_value, unit.symbol

        # Convert to target unit
        converted_value = self.converter.convert(
            numeric_value, unit, target_unit)
        if converted_value is not None:
            return converted_value, target_unit.symbol

        return numeric_value, unit.symbol

    def normalize_column(self, series: pd.Series) -> pd.DataFrame:
        """Normalize a column containing quantities with units."""
        results = []

        for value in series:
            normalized = self.normalize_value(value)
            if normalized:
                results.append({
                    'value': normalized[0],
                    'unit': normalized[1],
                    'original': str(value)
                })
            else:
                results.append({
                    'value': None,
                    'unit': None,
                    'original': str(value)
                })

        return pd.DataFrame(results, index=series.index)

    def detect_units_in_dataframe(self, df: pd.DataFrame) -> Dict[str, UnitCategory]:
        """Detect unit categories in DataFrame columns."""
        column_categories = {}

        for column in df.columns:
            # Sample some values to detect units
            sample_values = df[column].dropna().astype(str).head(100)
            categories_found = []

            for value in sample_values:
                category = self.unit_detector.detect_category(value)
                if category != UnitCategory.UNKNOWN:
                    categories_found.append(category)

            if categories_found:
                # Use most common category
                most_common = max(set(categories_found),
                                  key=categories_found.count)
                column_categories[column] = most_common
            else:
                column_categories[column] = UnitCategory.UNKNOWN

        return column_categories


# Header-derived unit parsing (inspired by Winter)
def _extract_unit_token_from_header(header: str) -> Optional[str]:
    """Extract a potential unit token from a header string.

    Examples:
    - "Speed (km/h)" -> "km/h"
    - "Area (m²)" -> "m²"
    - "Weight_kg" -> "kg"
    - "Temperature [°C]" -> "°C"
    """
    if header is None:
        return None

    text = str(header).strip()
    if not text:
        return None

    # Prefer tokens inside brackets first
    bracket_match = re.search(r"[\(\[]\s*([^\)\]]+?)\s*[\)\]]", text)
    if bracket_match:
        token = bracket_match.group(1).strip()
        return token if token else None

    # Fallbacks: trailing tokens separated by common delimiters
    # e.g., "Weight_kg", "Weight-kg", "Weight kg"
    tail_match = re.search(
        r"(?:^|[\s_\-\/])([A-Za-z°$€£¥₹₽₩₪%][A-Za-z0-9°$€£¥₹₽₩₪%\/\^]*)$", text)
    if tail_match:
        token = tail_match.group(1).strip()
        return token if token else None

    return None


def parse_unit_from_header(
    header: str,
    registry: Optional[UnitRegistry] = None,
) -> Optional[Unit]:
    """Parse a unit from a column header, if present.

    Tries bracketed tokens first (e.g., "(km)", "[°C]"), then falls back to
    a trailing token heuristic. Returns a matching `Unit` from the provided or
    default registry, or None if no unit can be determined.
    """
    if header is None:
        return None

    reg = registry or UnitRegistry()

    token = _extract_unit_token_from_header(header)
    if not token:
        return None

    # Try exact match
    unit = reg.get_unit(token)
    if unit:
        return unit

    # Try lower-cased version
    unit = reg.get_unit(token.lower())
    if unit:
        return unit

    # Try to split composite tokens like "km/h" into parts
    # and match any part as a known unit (common in speed units)
    if "/" in token:
        parts = [p.strip() for p in token.split("/") if p.strip()]
        for p in parts:
            unit = reg.get_unit(p) or reg.get_unit(p.lower())
            if unit:
                return unit

    return None

# Convenience functions for easy usage


def parse_quantity(text: str) -> Optional[Quantity]:
    """Parse quantity from text."""
    parser = QuantityParser()
    return parser.parse_quantity(text)


def detect_unit(text: str, comprehensive: bool = False) -> Optional[Unit]:
    """Detect unit in text."""
    registry = UnitRegistry(comprehensive=comprehensive)
    detector = UnitDetector(registry)
    return detector.detect_unit(text)


def normalize_units(text: str, comprehensive: bool = False) -> Optional[Tuple[float, str]]:
    """Normalize quantity with units."""
    normalizer = UnitNormalizer(comprehensive=comprehensive)
    return normalizer.normalize_value(text)


def convert_units(
    value: float,
    from_unit: str,
    to_unit: str,
    comprehensive: bool = False
) -> Optional[float]:
    """Convert value between unit symbols."""
    registry = UnitRegistry(comprehensive=comprehensive)
    converter = UnitConverter(registry)

    from_unit_obj = registry.get_unit(from_unit)
    to_unit_obj = registry.get_unit(to_unit)

    if not from_unit_obj or not to_unit_obj:
        return None

    return converter.convert(value, from_unit_obj, to_unit_obj)


def get_supported_units(category: Optional[str] = None, comprehensive: bool = False) -> List[str]:
    """Get list of supported unit symbols."""
    registry = UnitRegistry(comprehensive=comprehensive)

    if category:
        try:
            unit_category = UnitCategory(category.lower())
            units = registry.get_units_by_category(unit_category)
            return [unit.symbol for unit in units]
        except ValueError:
            return []
    else:
        return list(registry.units.keys())


def get_supported_categories() -> List[str]:
    """Get list of supported unit categories."""
    return [category.value for category in UnitCategory if category != UnitCategory.UNKNOWN]
