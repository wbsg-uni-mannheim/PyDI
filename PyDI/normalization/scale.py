"""
Scale modifier detection and expansion for numeric values.

This module handles scale modifiers like:
- Generic: thousand, million, billion, k, M, B
- Currency-specific: MEO, MEUR, kEUR, MUSD, kUSD

No existing library handles the reverse direction (parsing "5 million" → 5,000,000)
or currency-specific modifiers like MEO, so we implement this ourselves.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class ScaleModifier:
    """A scale modifier pattern and its multiplier."""

    name: str
    multiplier: float
    pattern: re.Pattern[str]


@dataclass
class ScaleResult:
    """Result of scale detection/expansion."""

    original: str
    value: float
    scaled_value: float
    modifier: ScaleModifier | None
    cleaned_text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "original": self.original,
            "value": self.value,
            "scaled_value": self.scaled_value,
            "modifier_name": self.modifier.name if self.modifier else None,
            "multiplier": self.modifier.multiplier if self.modifier else None,
            "cleaned_text": self.cleaned_text,
        }


# Currency-specific scale modifiers (checked first, higher priority)
CURRENCY_SCALES = [
    # Euros - Portuguese style
    ScaleModifier("MEO", 1_000_000, re.compile(r"\bMEO\b")),
    # Euros - standard abbreviations
    ScaleModifier("MEUR", 1_000_000, re.compile(r"\bM\.?EUR\b", re.IGNORECASE)),
    ScaleModifier("kEUR", 1_000, re.compile(r"\bk\.?EUR\b", re.IGNORECASE)),
    ScaleModifier("TEUR", 1_000, re.compile(r"\bT\.?EUR\b", re.IGNORECASE)),
    # USD
    ScaleModifier("MUSD", 1_000_000, re.compile(r"\bM\.?USD\b", re.IGNORECASE)),
    ScaleModifier("kUSD", 1_000, re.compile(r"\bk\.?USD\b", re.IGNORECASE)),
    # GBP
    ScaleModifier("MGBP", 1_000_000, re.compile(r"\bM\.?GBP\b", re.IGNORECASE)),
    ScaleModifier("kGBP", 1_000, re.compile(r"\bk\.?GBP\b", re.IGNORECASE)),
]

# Generic scale words (checked after currency-specific)
GENERIC_SCALES = [
    ScaleModifier("hundred", 100, re.compile(r"\bhundreds?\b", re.IGNORECASE)),
    ScaleModifier("thousand", 1_000, re.compile(r"\bthousands?\b", re.IGNORECASE)),
    ScaleModifier("million", 1_000_000, re.compile(r"\bmillions?\b", re.IGNORECASE)),
    ScaleModifier("billion", 1_000_000_000, re.compile(r"\bbillions?\b", re.IGNORECASE)),
    ScaleModifier("trillion", 1_000_000_000_000, re.compile(r"\btrillions?\b", re.IGNORECASE)),
    # Abbreviations - only match when standalone (not part of a unit like km)
    ScaleModifier("k", 1_000, re.compile(r"(?<![a-zA-Z])[kK](?![a-zA-Z])")),
    ScaleModifier("M", 1_000_000, re.compile(r"(?<![a-zA-Z])[mM](?![a-zA-Z])")),
    ScaleModifier("B", 1_000_000_000, re.compile(r"(?<![a-zA-Z])[bB](?![a-zA-Z])")),
]

ALL_SCALES = CURRENCY_SCALES + GENERIC_SCALES


def detect_scale_modifier(text: str) -> ScaleModifier | None:
    """
    Detect a scale modifier in text.

    Currency-specific modifiers are checked first (higher priority).

    Args:
        text: Text to search for scale modifiers

    Returns:
        ScaleModifier if found, None otherwise

    Examples:
        >>> detect_scale_modifier("5 MEO revenue").name
        'MEO'
        >>> detect_scale_modifier("Revenue: 5 million").name
        'million'
        >>> detect_scale_modifier("5 km distance")  # km is a unit, not scale
        None
    """
    if not text:
        return None

    for modifier in ALL_SCALES:
        if modifier.pattern.search(text):
            return modifier

    return None


def expand_scale(text: str, value: float) -> ScaleResult:
    """
    Expand scale modifier and return scaled value.

    Args:
        text: Original text containing potential scale modifier
        value: Numeric value to scale

    Returns:
        ScaleResult with scaled value and cleaned text

    Examples:
        >>> result = expand_scale("5 MEO", 5.0)
        >>> result.scaled_value
        5000000.0
        >>> result = expand_scale("Revenue: 2.5 million EUR", 2.5)
        >>> result.scaled_value
        2500000.0
    """
    if not text:
        return ScaleResult(
            original=text or "",
            value=value,
            scaled_value=value,
            modifier=None,
            cleaned_text=text or "",
        )

    modifier = detect_scale_modifier(text)

    if modifier:
        scaled_value = value * modifier.multiplier
        cleaned_text = modifier.pattern.sub("", text)
        cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

        return ScaleResult(
            original=text,
            value=value,
            scaled_value=scaled_value,
            modifier=modifier,
            cleaned_text=cleaned_text,
        )

    return ScaleResult(
        original=text,
        value=value,
        scaled_value=value,
        modifier=None,
        cleaned_text=text,
    )


def parse_scaled_number(text: str) -> tuple[float, ScaleModifier | None] | None:
    """
    Parse a number with optional scale modifier from text.

    Args:
        text: Text containing a number and optional scale modifier

    Returns:
        Tuple of (scaled_value, modifier) or None if no number found

    Examples:
        >>> value, mod = parse_scaled_number("5 MEO")
        >>> value
        5000000.0
        >>> value, mod = parse_scaled_number("2.5 million")
        >>> value
        2500000.0
        >>> value, mod = parse_scaled_number("100")
        >>> value
        100.0
    """
    if not text:
        return None

    text = text.strip()

    # Extract number (handles: 5, 5.5, 5,000, 5,000.50, -5, +5)
    number_pattern = re.compile(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[-+]?\d*\.?\d+")
    match = number_pattern.search(text)

    if not match:
        return None

    try:
        value = float(match.group().replace(",", ""))
    except ValueError:
        return None

    modifier = detect_scale_modifier(text)

    if modifier:
        return (value * modifier.multiplier, modifier)

    return (value, None)
