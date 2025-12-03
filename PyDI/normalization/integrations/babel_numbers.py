"""
Babel integration for locale-aware numeric parsing.

This module wraps the Babel library to provide:
- Locale-aware decimal parsing
- Locale inference from numeric strings
- Multi-locale parsing attempts

Note: Currency symbol stripping is handled here for numeric parsing purposes.
For currency code normalization (EUR, USD, etc.), see the country module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional, Set

logger = logging.getLogger(__name__)

# Babel is optional
try:
    from babel.numbers import parse_decimal as _babel_parse_decimal
    from babel.numbers import NumberFormatError
    BABEL_AVAILABLE = True
except ImportError:
    BABEL_AVAILABLE = False
    _babel_parse_decimal = None  # type: ignore
    NumberFormatError = Exception  # type: ignore


# Currency symbols for stripping during numeric parsing
# (different from currency code normalization in country.py)
CURRENCY_SYMBOLS: Set[str] = {'$', '€', '£', '¥', '₹', '₽', '₩', '₪', '₴', '฿', '₫'}

# Default candidate locales for inference
DEFAULT_CANDIDATE_LOCALES: List[str] = [
    'en_US', 'de_DE', 'fr_FR', 'it_IT', 'es_ES', 'sv_SE', 'de_CH', 'fr_CH'
]


@dataclass
class BabelParseResult:
    """Result of parsing a numeric string with Babel."""

    value: Optional[float]
    locale: Optional[str]
    original: str
    success: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "locale": self.locale,
            "original": self.original,
            "success": self.success,
        }


def is_babel_available() -> bool:
    """Check if Babel is installed."""
    return BABEL_AVAILABLE


def order_locales_for_value(
    text: str,
    candidate_locales: Optional[List[str]] = None,
) -> List[str]:
    """
    Order candidate locales based on heuristics from the value.

    If the last punctuation is a comma, prefer EU-style locales first.
    Otherwise prefer US-style locales first.

    Args:
        text: The numeric string to analyze
        candidate_locales: List of locales to consider

    Returns:
        Ordered list of locales to try
    """
    candidates = candidate_locales or DEFAULT_CANDIDATE_LOCALES

    s = str(text)
    last_comma = s.rfind(',')
    last_dot = s.rfind('.')

    eu_first = ['de_DE', 'fr_FR', 'it_IT', 'es_ES', 'sv_SE', 'de_CH', 'fr_CH', 'en_US']
    us_first = ['en_US', 'de_CH', 'fr_CH', 'de_DE', 'fr_FR', 'it_IT', 'es_ES', 'sv_SE']

    if last_comma > last_dot:
        order = eu_first
    else:
        order = us_first

    allowed = set(candidates)
    return [loc for loc in order if loc in allowed] or list(candidates)


def parse_decimal(
    text: str,
    locale: Optional[str] = None,
    candidate_locales: Optional[List[str]] = None,
    strip_currency: bool = True,
    handle_percent: bool = True,
    handle_parentheses: bool = True,
) -> BabelParseResult:
    """
    Parse a decimal number using Babel with locale awareness.

    Args:
        text: The string to parse
        locale: Specific locale to use (if None, tries multiple)
        candidate_locales: Locales to try if locale is not specified
        strip_currency: Remove currency symbols before parsing
        handle_percent: Handle percentage values (divide by 100)
        handle_parentheses: Treat (value) as negative

    Returns:
        BabelParseResult with parsed value and metadata

    Examples:
        >>> result = parse_decimal("1.234,56", locale="de_DE")
        >>> result.value
        1234.56
        >>> result = parse_decimal("1,234.56", locale="en_US")
        >>> result.value
        1234.56
    """
    if not BABEL_AVAILABLE:
        return BabelParseResult(
            value=None,
            locale=None,
            original=text if text else "",
            success=False,
        )

    if not text or not isinstance(text, str):
        return BabelParseResult(
            value=None,
            locale=None,
            original=str(text) if text else "",
            success=False,
        )

    s = str(text).strip()
    original = s

    # Handle parentheses as negative
    is_negative = False
    if handle_parentheses and s.startswith('(') and s.endswith(')'):
        is_negative = True
        s = s[1:-1].strip()

    # Handle percentage
    is_percent = False
    if handle_percent and s.endswith('%'):
        is_percent = True
        s = s[:-1].strip()

    # Strip currency symbols
    if strip_currency:
        for symbol in CURRENCY_SYMBOLS:
            s = s.replace(symbol, '')
        s = s.strip()

    # Determine locales to try
    if locale:
        locales_to_try = [locale]
    else:
        locales_to_try = order_locales_for_value(s, candidate_locales)

    for loc in locales_to_try:
        try:
            val = float(_babel_parse_decimal(s, locale=loc))
            if is_percent:
                val /= 100.0
            if is_negative:
                val = -val
            return BabelParseResult(
                value=val,
                locale=loc,
                original=original,
                success=True,
            )
        except Exception:
            # Try with common grouping characters stripped
            try:
                sanitized = s
                for grp in (" ", "\xa0", "'", "'"):
                    sanitized = sanitized.replace(grp, '')
                val = float(_babel_parse_decimal(sanitized, locale=loc))
                if is_percent:
                    val /= 100.0
                if is_negative:
                    val = -val
                return BabelParseResult(
                    value=val,
                    locale=loc,
                    original=original,
                    success=True,
                )
            except Exception:
                continue

    return BabelParseResult(
        value=None,
        locale=None,
        original=original,
        success=False,
    )


def infer_locale(
    values: List[str],
    candidate_locales: Optional[List[str]] = None,
    sample_size: int = 500,
) -> Optional[str]:
    """
    Infer the most likely locale for a collection of numeric strings.

    Tries parsing each value with each candidate locale and returns
    the locale with the highest success count.

    Args:
        values: List of numeric strings to analyze
        candidate_locales: Locales to consider
        sample_size: Maximum number of values to sample

    Returns:
        The locale with highest parse success, or None

    Examples:
        >>> infer_locale(["1.234,56", "2.345,67", "3.456,78"])
        'de_DE'
        >>> infer_locale(["1,234.56", "2,345.67"])
        'en_US'
    """
    if not BABEL_AVAILABLE:
        return None

    locales = candidate_locales or DEFAULT_CANDIDATE_LOCALES
    if not locales:
        return None

    # Filter and sample values
    data = [
        str(v) for v in values
        if isinstance(v, (str, bytes)) and str(v).strip()
    ][:sample_size]

    if not data:
        return None

    best_locale = None
    best_score = -1

    for loc in locales:
        score = 0
        for v in data:
            s = v.strip().replace('\xa0', ' ')
            # Remove percent and currency for inference
            if s.endswith('%'):
                s = s[:-1].strip()
            for symbol in CURRENCY_SYMBOLS:
                s = s.replace(symbol, '')
            try:
                _babel_parse_decimal(s, locale=loc)
                score += 1
            except Exception:
                pass

        if score > best_score:
            best_score = score
            best_locale = loc

    return best_locale
