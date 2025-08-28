"""
Comprehensive type conversion utilities for PyDI.

This module provides all type conversion functionality including coordinate parsing,
boolean parsing, URL normalization, date handling, and numeric parsing. It includes
both basic and advanced type conversion capabilities.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any, List, Optional, Tuple, Union
from urllib.parse import urlparse

import pandas as pd

logger = logging.getLogger(__name__)

# Optional: Babel for locale-aware numeric parsing
try:
    from babel.numbers import parse_decimal as babel_parse_decimal  # type: ignore
    from babel.numbers import NumberFormatError  # type: ignore
    BABEL_AVAILABLE = True
except Exception:
    BABEL_AVAILABLE = False


class CoordinateParser:
    """
    Parser and validator for geographic coordinates.

    Supports various coordinate formats including decimal degrees,
    degrees/minutes/seconds, and different notation styles.
    """

    def __init__(self):
        # Decimal degrees pattern (e.g., "40.7589, -73.9851")
        self.decimal_pattern = re.compile(
            r'^[-+]?([1-8]?\d(?:\.\d+)?|90(?:\.0+)?)[,\s]+[-+]?(180(?:\.0+)?|(?:1[0-7]\d|[1-9]?\d)(?:\.\d+)?)$'
        )

        # Degrees Minutes Seconds pattern (e.g., "40°45'32.1\"N 73°59'6.4\"W")
        self.dms_pattern = re.compile(
            r'(\d{1,3})°(?:(\d{1,2})\')?(?:(\d{1,2}(?:\.\d+)?)\")?([NSEW])\s*'
            r'(\d{1,3})°(?:(\d{1,2})\')?(?:(\d{1,2}(?:\.\d+)?)\")?([NSEW])'
        )

        # Degrees Minutes pattern (e.g., "40°45.535'N 73°59.107'W")
        self.dm_pattern = re.compile(
            r'(\d{1,3})°(\d{1,2}\.\d+)\'([NSEW])\s*(\d{1,3})°(\d{1,2}\.\d+)\'([NSEW])'
        )

    def parse_coordinate(self, coord_str: str) -> Optional[Tuple[float, float]]:
        """
        Parse coordinate string into latitude, longitude tuple.

        Parameters
        ----------
        coord_str : str
            Coordinate string to parse.

        Returns
        -------
        Optional[Tuple[float, float]]
            (latitude, longitude) tuple or None if parsing fails.
        """
        if pd.isna(coord_str) or not coord_str:
            return None

        coord_str = str(coord_str).strip()

        # Try decimal degrees format
        match = self.decimal_pattern.match(coord_str)
        if match:
            try:
                lat, lon = map(float, coord_str.replace(',', ' ').split())
                if self.is_valid_coordinate(lat, lon):
                    return lat, lon
            except ValueError:
                pass

        # Try DMS format
        match = self.dms_pattern.match(coord_str)
        if match:
            try:
                lat_deg, lat_min, lat_sec, lat_dir, lon_deg, lon_min, lon_sec, lon_dir = match.groups()

                lat = self._dms_to_decimal(
                    int(lat_deg),
                    int(lat_min or 0),
                    float(lat_sec or 0),
                    lat_dir
                )
                lon = self._dms_to_decimal(
                    int(lon_deg),
                    int(lon_min or 0),
                    float(lon_sec or 0),
                    lon_dir
                )

                if self.is_valid_coordinate(lat, lon):
                    return lat, lon
            except (ValueError, TypeError):
                pass

        # Try DM format
        match = self.dm_pattern.match(coord_str)
        if match:
            try:
                lat_deg, lat_min, lat_dir, lon_deg, lon_min, lon_dir = match.groups()

                lat = self._dm_to_decimal(
                    int(lat_deg), float(lat_min), lat_dir)
                lon = self._dm_to_decimal(
                    int(lon_deg), float(lon_min), lon_dir)

                if self.is_valid_coordinate(lat, lon):
                    return lat, lon
            except (ValueError, TypeError):
                pass

        # Winter compatibility: accept single numeric value as coordinate
        try:
            if re.fullmatch(r'[-+]?\d+(?:\.\d+)?', coord_str):
                val = float(coord_str)
                # Accept if in valid lat/lon domain
                if -180.0 < val < 180.0:
                    # If within latitude range, assume (lat, 0.0); else (0.0, lon)
                    if -90.0 <= val <= 90.0:
                        return val, 0.0
                    else:
                        return 0.0, val
        except Exception:
            pass

        return None

    def _dms_to_decimal(self, degrees: int, minutes: int, seconds: float, direction: str) -> float:
        """Convert degrees/minutes/seconds to decimal degrees."""
        decimal = degrees + minutes/60 + seconds/3600
        if direction.upper() in ['S', 'W']:
            decimal = -decimal
        return decimal

    def _dm_to_decimal(self, degrees: int, minutes: float, direction: str) -> float:
        """Convert degrees/minutes to decimal degrees."""
        decimal = degrees + minutes/60
        if direction.upper() in ['S', 'W']:
            decimal = -decimal
        return decimal

    def is_valid_coordinate(self, lat: float, lon: float) -> bool:
        """Validate latitude and longitude ranges."""
        return -90 <= lat <= 90 and -180 <= lon <= 180

    def format_coordinate(
        self,
        lat: float,
        lon: float,
        format_type: str = 'decimal'
    ) -> str:
        """
        Format coordinates as string.

        Parameters
        ----------
        lat : float
            Latitude.
        lon : float
            Longitude.
        format_type : str, default 'decimal'
            Format type: 'decimal', 'dms', or 'dm'.

        Returns
        -------
        str
            Formatted coordinate string.
        """
        if not self.is_valid_coordinate(lat, lon):
            raise ValueError(f"Invalid coordinates: {lat}, {lon}")

        if format_type == 'decimal':
            return f"{lat:.6f}, {lon:.6f}"
        elif format_type == 'dms':
            lat_dms = self._decimal_to_dms(lat, 'NS')
            lon_dms = self._decimal_to_dms(lon, 'EW')
            return f"{lat_dms} {lon_dms}"
        elif format_type == 'dm':
            lat_dm = self._decimal_to_dm(lat, 'NS')
            lon_dm = self._decimal_to_dm(lon, 'EW')
            return f"{lat_dm} {lon_dm}"
        else:
            raise ValueError(f"Unknown format type: {format_type}")

    def _decimal_to_dms(self, decimal: float, directions: str) -> str:
        """Convert decimal degrees to DMS format."""
        direction = directions[0] if decimal >= 0 else directions[1]
        decimal = abs(decimal)
        degrees = int(decimal)
        minutes = int((decimal - degrees) * 60)
        seconds = ((decimal - degrees) * 60 - minutes) * 60
        return f"{degrees}°{minutes}'{seconds:.1f}\"{direction}"

    def _decimal_to_dm(self, decimal: float, directions: str) -> str:
        """Convert decimal degrees to DM format."""
        direction = directions[0] if decimal >= 0 else directions[1]
        decimal = abs(decimal)
        degrees = int(decimal)
        minutes = (decimal - degrees) * 60
        return f"{degrees}°{minutes:.3f}'{direction}"


class BooleanParser:
    """
    Enhanced boolean parser supporting various representations.

    Supports multiple languages and formats for boolean values.
    """

    def __init__(self, case_sensitive: bool = False):
        self.case_sensitive = case_sensitive

        # Extended boolean mappings
        self.true_values = {
            # English
            'true', 't', 'yes', 'y', 'on', 'enabled', 'active', 'ok', 'okay',
            # Numeric
            '1', '1.0',
            # German
            'ja', 'wahr',
            # French
            'oui', 'vrai',
            # Spanish
            'sí', 'si', 'verdadero',
            # Other common
            'pass', 'passed', 'success', 'successful', 'valid', 'correct'
        }

        self.false_values = {
            # English
            'false', 'f', 'no', 'n', 'off', 'disabled', 'inactive', 'not ok',
            # Numeric
            '0', '0.0',
            # German
            'nein', 'falsch',
            # French
            'non', 'faux',
            # Spanish
            'no', 'falso',
            # Other common
            'fail', 'failed', 'failure', 'invalid', 'incorrect', 'error'
        }

        if not case_sensitive:
            self.true_values = {v.lower() for v in self.true_values}
            self.false_values = {v.lower() for v in self.false_values}

    def parse_boolean(self, value: Any) -> Optional[bool]:
        """Parse boolean value from various representations."""
        if pd.isna(value):
            return None

        # Handle existing boolean types
        if isinstance(value, bool):
            return value

        # Handle numeric types
        if isinstance(value, (int, float)):
            if value == 1 or value == 1.0:
                return True
            elif value == 0 or value == 0.0:
                return False
            else:
                return None

        # Handle string representations
        if isinstance(value, str):
            test_value = value.strip()
            if not self.case_sensitive:
                test_value = test_value.lower()

            if test_value in self.true_values:
                return True
            elif test_value in self.false_values:
                return False

        return None

    def is_boolean(self, value: Any) -> bool:
        """Check if value can be interpreted as boolean."""
        return self.parse_boolean(value) is not None


class LinkNormalizer:
    """URL and link normalization and validation."""

    def __init__(
        self,
        add_protocol: bool = True,
        default_protocol: str = 'https',
        normalize_case: bool = True
    ):
        self.add_protocol = add_protocol
        self.default_protocol = default_protocol
        self.normalize_case = normalize_case

        # URL pattern for validation
        self.url_pattern = re.compile(
            r'^https?://[^\s/$.?#].[^\s]*$|^www\.[^\s/$.?#].[^\s]*$|^[^\s/$.?#]+\.[a-z]{2,}[/\w\-._~:/?#[\]@!$&\'()*+,;=]*$',
            re.IGNORECASE
        )

        # Domain pattern
        self.domain_pattern = re.compile(
            r'^[a-zA-Z0-9][a-zA-Z0-9-]{1,61}[a-zA-Z0-9]\.[a-zA-Z]{2,}$')

    def normalize_url(self, url: str) -> Optional[str]:
        """Normalize URL format."""
        if pd.isna(url) or not url:
            return None

        url = str(url).strip()

        # Remove surrounding quotes
        if (url.startswith('"') and url.endswith('"')) or (url.startswith("'") and url.endswith("'")):
            url = url[1:-1]

        # Add protocol if missing
        if self.add_protocol and not url.startswith(('http://', 'https://')):
            if url.startswith('www.') or self.domain_pattern.match(url):
                url = f"{self.default_protocol}://{url}"

        # Validate URL structure
        if not self.is_valid_url(url):
            return None

        try:
            parsed = urlparse(url)

            # Normalize domain case
            if self.normalize_case and parsed.netloc:
                netloc = parsed.netloc.lower()
            else:
                netloc = parsed.netloc

            # Reconstruct URL
            normalized = f"{parsed.scheme}://{netloc}{parsed.path}"
            if parsed.params:
                normalized += f";{parsed.params}"
            if parsed.query:
                normalized += f"?{parsed.query}"
            if parsed.fragment:
                normalized += f"#{parsed.fragment}"

            return normalized

        except Exception:
            return None

    def is_valid_url(self, url: str) -> bool:
        """Validate URL format."""
        if pd.isna(url) or not url:
            return False

        url = str(url).strip()
        return bool(self.url_pattern.match(url))

    def extract_domain(self, url: str) -> Optional[str]:
        """Extract domain from URL."""
        normalized = self.normalize_url(url)
        if not normalized:
            return None

        try:
            parsed = urlparse(normalized)
            return parsed.netloc
        except Exception:
            return None


class NumericParser:
    """
    Enhanced numeric parser with locale support and advanced format detection.

    Handles various numeric formats including different decimal separators,
    thousands separators, scientific notation, and currency formats.
    """

    def __init__(
        self,
        decimal_separator: str = '.',
        thousands_separator: str = ',',
        handle_currency: bool = True,
        handle_percentages: bool = True,
        auto_detect_locale: bool = True,
        extra_thousands_separators: Optional[List[str]] = None,
        *,
        use_babel: bool = True,
        babel_locale: Optional[str] = None,
        babel_candidate_locales: Optional[List[str]] = None,
    ):
        self.decimal_separator = decimal_separator
        self.thousands_separator = thousands_separator
        self.handle_currency = handle_currency
        self.handle_percentages = handle_percentages
        self.auto_detect_locale = auto_detect_locale
        self.extra_thousands_separators = set(extra_thousands_separators or [
                                              "'", "’", " ", "\xa0"])  # apostrophe, NBSP, space

        # Babel configuration (optional dependency)
        self.use_babel = use_babel and BABEL_AVAILABLE
        self.babel_locale = babel_locale
        self.babel_candidate_locales = babel_candidate_locales or [
            'en_US', 'de_DE', 'fr_FR', 'it_IT', 'es_ES', 'sv_SE', 'de_CH', 'fr_CH'
        ]

        # Currency symbols
        self.currency_symbols = {'$', '€', '£', '¥', '₹', '₽', '₩', '₪'}

        # Build patterns
        self._build_patterns()

    def _build_patterns(self):
        """Build regex patterns for number detection."""
        dec_sep = re.escape(self.decimal_separator)
        thou_sep = re.escape(self.thousands_separator)

        # Scientific notation
        self.scientific_pattern = re.compile(
            r'[-+]?\d*\.?\d+[eE][-+]?\d+',
            re.IGNORECASE
        )

        # Decimal with thousands separators
        self.decimal_pattern = re.compile(
            f'[-+]?\\d{{1,3}}(?:{thou_sep}\\d{{3}})*(?:{dec_sep}\\d+)?'
        )

        # Simple decimal
        self.simple_decimal_pattern = re.compile(
            f'[-+]?\\d*{dec_sep}\\d+'
        )

        # Integer with thousands
        self.integer_thousands_pattern = re.compile(
            f'[-+]?\\d{{1,3}}(?:{thou_sep}\\d{{3}})+'
        )

        # Simple integer
        self.simple_integer_pattern = re.compile(r'[-+]?\d+')

        # Percentage
        if self.handle_percentages:
            self.percentage_pattern = re.compile(
                f'\\d+(?:{dec_sep}\\d+)?%'
            )

        # Currency
        if self.handle_currency:
            currency_chars = ''.join(re.escape(s)
                                     for s in self.currency_symbols)
            self.currency_pattern = re.compile(
                f'[{currency_chars}]?\\s*\\d+(?:{thou_sep}\\d{{3}})*(?:{dec_sep}\\d{{2}})?\\s*[{currency_chars}]?'
            )

    def _ordered_babel_locales_for_value(self, text: str) -> List[str]:
        """Heuristic ordering of candidate locales for a given value."""
        # Prefer EU-style first if last punctuation is comma
        s = str(text)
        last_comma = s.rfind(',')
        last_dot = s.rfind('.')
        eu_first = ['de_DE', 'fr_FR', 'it_IT', 'es_ES',
                    'sv_SE', 'de_CH', 'fr_CH', 'en_US']
        us_first = ['en_US', 'de_CH', 'fr_CH', 'de_DE',
                    'fr_FR', 'it_IT', 'es_ES', 'sv_SE']
        if last_comma > last_dot:
            order = eu_first
        else:
            order = us_first
        # Keep only those present in configured candidates, preserve order
        allowed = set(self.babel_candidate_locales)
        return [loc for loc in order if loc in allowed] or self.babel_candidate_locales

    def _parse_with_babel(self, text: str) -> Optional[Union[int, float]]:
        """Try parsing using Babel with an optionally known or inferred locale."""
        if not self.use_babel:
            return None
        if pd.isna(text) or not text:
            return None

        s = str(text).strip()

        # Parentheses negative
        negative = s.startswith('(') and s.endswith(')')
        if negative:
            s = s[1:-1].strip()

        # Percentage
        is_percent = s.endswith('%')
        if is_percent:
            s = s[:-1].strip()

        # Strip currency symbols
        for symbol in self.currency_symbols:
            s = s.replace(symbol, '')
        s = s.strip()

        locales_to_try: List[str]
        if self.babel_locale:
            locales_to_try = [self.babel_locale]
        else:
            locales_to_try = self._ordered_babel_locales_for_value(s)

        for loc in locales_to_try:
            try:
                # Attempt direct parse first
                val = float(babel_parse_decimal(s, locale=loc))
                if is_percent:
                    val /= 100.0
                if negative:
                    val = -val
                return val
            except Exception:
                # Fallback: strip common grouping characters and retry
                try:
                    sanitized = s
                    for grp in (" ", "\xa0", "'", "’"):
                        sanitized = sanitized.replace(grp, '')
                    val = float(babel_parse_decimal(sanitized, locale=loc))
                    if is_percent:
                        val /= 100.0
                    if negative:
                        val = -val
                    return val
                except Exception:
                    continue
        return None

    def infer_babel_locale(
        self,
        values: List[str],
        candidate_locales: Optional[List[str]] = None,
        *,
        sample_size: int = 500,
    ) -> Optional[str]:
        """Infer the most likely locale for a collection of numeric strings using Babel.

        Returns the locale with the highest parse success count.
        """
        if not self.use_babel:
            return None
        locales = candidate_locales or self.babel_candidate_locales
        if not locales:
            return None

        # Sample to limit cost
        data = [str(v) for v in values if isinstance(
            v, (str, bytes)) and str(v).strip()][:sample_size]
        if not data:
            return None

        best_locale = None
        best_score = -1

        for loc in locales:
            score = 0
            for v in data:
                s = v.strip().replace('\xa0', ' ')
                # Remove currency and percent for inference
                if s.endswith('%'):
                    s = s[:-1].strip()
                for symbol in self.currency_symbols:
                    s = s.replace(symbol, '')
                try:
                    babel_parse_decimal(s, locale=loc)
                    score += 1
                except Exception:
                    pass
            if score > best_score:
                best_score = score
                best_locale = loc

        return best_locale

    def parse_numeric(self, text: str) -> Optional[Union[int, float]]:
        """Parse numeric value from text."""
        if pd.isna(text) or not text:
            return None

        text = str(text).strip()

        # Parentheses denote negative numbers in some sources: (1.234,56)
        is_negative_parentheses = False
        if text.startswith('(') and text.endswith(')'):
            is_negative_parentheses = True
            text = text[1:-1].strip()

        # Try Babel-backed parsing first (handles %, currency, parentheses)
        babel_value = self._parse_with_babel(text)
        if babel_value is not None:
            return babel_value

        # Handle percentages (fallback path without Babel)
        if self.handle_percentages and text.endswith('%'):
            try:
                numeric_part = text[:-1]
                # Remove any known thousands separators
                numeric_part = numeric_part.replace(
                    self.thousands_separator, '')
                for sep in self.extra_thousands_separators:
                    numeric_part = numeric_part.replace(sep, '')
                # Normalize decimal separator
                if self.decimal_separator != '.':
                    numeric_part = numeric_part.replace(
                        self.decimal_separator, '.')
                value = float(numeric_part) / 100.0
                return -value if is_negative_parentheses else value
            except ValueError:
                pass

        # Handle currency
        if self.handle_currency and any(symbol in text for symbol in self.currency_symbols):
            # Remove currency symbols
            clean_text = text
            for symbol in self.currency_symbols:
                clean_text = clean_text.replace(symbol, '')
            clean_text = clean_text.strip()
            parsed = self.parse_numeric(clean_text)
            if parsed is not None and is_negative_parentheses:
                parsed = -parsed
            return parsed

        # Try different patterns in order of specificity
        patterns = [
            self.scientific_pattern,
            self.decimal_pattern,
            self.simple_decimal_pattern,
            self.integer_thousands_pattern,
            self.simple_integer_pattern
        ]

        # First try to match the entire text exactly (for full numeric strings)
        for pattern in patterns:
            match = pattern.fullmatch(text)
            if match:
                try:
                    value_str = match.group()
                    # Remove configured thousands separator and common alternates
                    value_str = value_str.replace(self.thousands_separator, '')
                    for sep in self.extra_thousands_separators:
                        value_str = value_str.replace(sep, '')
                    # Convert configured decimal separator to '.'
                    if self.decimal_separator != '.':
                        value_str = value_str.replace(
                            self.decimal_separator, '.')
                    # Try integer first, then float
                    try:
                        if '.' not in value_str and 'e' not in value_str.lower():
                            val = int(value_str)
                        else:
                            val = float(value_str)
                        return -val if is_negative_parentheses else val
                    except ValueError:
                        continue
                except Exception:
                    continue

        # If no full match found, try to find embedded numbers using search
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                try:
                    value_str = match.group()
                    # Remove configured thousands separator and common alternates
                    value_str = value_str.replace(self.thousands_separator, '')
                    for sep in self.extra_thousands_separators:
                        value_str = value_str.replace(sep, '')
                    # Convert configured decimal separator to '.'
                    if self.decimal_separator != '.':
                        value_str = value_str.replace(
                            self.decimal_separator, '.')
                    # Try integer first, then float
                    try:
                        if '.' not in value_str and 'e' not in value_str.lower():
                            val = int(value_str)
                        else:
                            val = float(value_str)
                        return -val if is_negative_parentheses else val
                    except ValueError:
                        continue
                except Exception:
                    continue

        # Auto-detect locale if enabled and direct patterns failed
        if self.auto_detect_locale:
            variants = [
                ('.', ','),  # US
                (',', '.'),  # EU
                (',', ' '),  # EU with space group
                (',', '\xa0'),  # NBSP grouping
                ('.', ' '),  # dot decimal with space grouping
                ('.', "'"),  # Swiss grouping
                (',', "'"),
                ('.', '’'),
                (',', '’'),
            ]
            txt = text
            for dec, thou in variants:
                try:
                    candidate = txt
                    # Remove grouping separators
                    if thou:
                        candidate = candidate.replace(thou, '')
                    # Also strip common extras
                    for sep in self.extra_thousands_separators:
                        if sep != dec:  # avoid stripping decimal
                            candidate = candidate.replace(sep, '')
                    # Normalize decimal
                    if dec != '.':
                        candidate = candidate.replace(dec, '.')
                    # Remove stray spaces
                    candidate = candidate.strip()
                    # Validate: only one decimal point allowed
                    if candidate.count('.') > 1:
                        continue
                    # Ensure it matches a basic numeric pattern now
                    if not re.fullmatch(r'[-+]?\d*(?:\.\d+)?(?:[eE][-+]?\d+)?', candidate):
                        continue
                    # Parse
                    val = float(candidate) if (
                        '.' in candidate or 'e' in candidate.lower()) else int(candidate)
                    return -val if is_negative_parentheses else val
                except Exception:
                    continue

        return None

    def is_numeric(self, text: str) -> bool:
        """Check if text represents a numeric value."""
        return self.parse_numeric(text) is not None


class DateNormalizer:
    """
    Date and datetime normalization and standardization.

    Provides comprehensive date parsing and normalization capabilities.
    """

    def __init__(
        self,
        target_format: str = '%Y-%m-%d',
        parse_formats: Optional[List[str]] = None,
        handle_timezone: bool = True,
        target_timezone: str = 'UTC',
    ):
        self.target_format = target_format
        self.parse_formats = parse_formats or [
            '%Y-%m-%d', '%Y/%m/%d', '%m/%d/%Y', '%m-%d-%Y',
            '%d/%m/%Y', '%d-%m-%Y', '%Y-%m-%d %H:%M:%S',
            '%Y/%m/%d %H:%M:%S', '%m/%d/%Y %H:%M:%S',
            '%B %d, %Y', '%b %d, %Y', '%d %B %Y',
            '%Y%m%d', '%m%d%Y'
        ]
        self.handle_timezone = handle_timezone
        self.target_timezone = target_timezone

    def parse_date(self, date_str: str) -> Optional[pd.Timestamp]:
        """Parse a date string using multiple format attempts."""
        if pd.isna(date_str) or not str(date_str).strip():
            return None

        date_str = str(date_str).strip()

        # Normalize leading plus on years and trailing Z to +00:00
        if date_str.startswith('+'):
            date_str = date_str[1:]
        if date_str.endswith('Z') and 'T' in date_str:
            date_str = date_str[:-1] + '+00:00'

        # First try pandas' built-in parsing
        try:
            return pd.to_datetime(date_str, infer_datetime_format=True)
        except:
            pass

        # Try each format explicitly
        for fmt in self.parse_formats:
            try:
                return pd.to_datetime(date_str, format=fmt)
            except:
                continue

        logger.debug(f"Could not parse date: {date_str}")
        return None

    def normalize_date(self, date_value: Any) -> Optional[str]:
        """Normalize date value to target format."""
        if pd.isna(date_value):
            return None

        # Parse the date
        if isinstance(date_value, str):
            parsed_date = self.parse_date(date_value)
        else:
            try:
                parsed_date = pd.to_datetime(date_value)
            except:
                return None

        if parsed_date is None:
            return None

        # Handle timezone if requested
        if self.handle_timezone:
            try:
                if parsed_date.tz is None:
                    parsed_date = parsed_date.tz_localize(
                        'UTC').tz_convert(self.target_timezone)
                else:
                    parsed_date = parsed_date.tz_convert(self.target_timezone)
            except:
                pass

        # Format to target format
        try:
            return parsed_date.strftime(self.target_format)
        except:
            return str(parsed_date)

    def normalize_column(self, series: pd.Series) -> pd.Series:
        """Normalize date column."""
        return series.apply(self.normalize_date)


class TypeConverter:
    """
    Main type converter combining all specialized parsers.

    Provides centralized type conversion with comprehensive parsing capabilities.
    """

    def __init__(
        self,
        coordinate_parser: Optional[CoordinateParser] = None,
        boolean_parser: Optional[BooleanParser] = None,
        link_normalizer: Optional[LinkNormalizer] = None,
        numeric_parser: Optional[NumericParser] = None,
        date_normalizer: Optional[DateNormalizer] = None
    ):
        self.coordinate_parser = coordinate_parser or CoordinateParser()
        self.boolean_parser = boolean_parser or BooleanParser()
        self.link_normalizer = link_normalizer or LinkNormalizer()
        self.numeric_parser = numeric_parser or NumericParser()
        self.date_normalizer = date_normalizer or DateNormalizer()

    def convert_coordinate(self, value: str) -> Optional[str]:
        """Convert coordinate value."""
        parsed = self.coordinate_parser.parse_coordinate(value)
        if parsed:
            return f"{parsed[0]:.6f}, {parsed[1]:.6f}"
        return None

    def convert_boolean(self, value: Any) -> Optional[bool]:
        """Convert boolean value."""
        return self.boolean_parser.parse_boolean(value)

    def convert_url(self, value: str) -> Optional[str]:
        """Convert URL value."""
        return self.link_normalizer.normalize_url(value)

    def convert_numeric(self, value: str) -> Optional[Union[int, float]]:
        """Convert numeric value."""
        return self.numeric_parser.parse_numeric(value)

    def convert_date(self, value: Any) -> Optional[str]:
        """Convert date value."""
        return self.date_normalizer.normalize_date(value)

    def convert_column(self, series: pd.Series, target_type: str) -> pd.Series:
        """Convert entire column to target type."""
        if target_type.lower() == 'coordinate':
            return series.apply(self.convert_coordinate)
        elif target_type.lower() == 'boolean':
            return series.apply(self.convert_boolean)
        elif target_type.lower() == 'url':
            return series.apply(self.convert_url)
        elif target_type.lower() == 'numeric':
            return series.apply(self.convert_numeric)
        elif target_type.lower() == 'date':
            return series.apply(self.convert_date)
        else:
            logger.warning(f"Unknown target type: {target_type}")
            return series


# Convenience functions for easy usage
def parse_coordinate(coord_str: str) -> Optional[Tuple[float, float]]:
    """Parse coordinate string."""
    parser = CoordinateParser()
    return parser.parse_coordinate(coord_str)


def parse_boolean(value: Any) -> Optional[bool]:
    """Parse boolean value."""
    parser = BooleanParser()
    return parser.parse_boolean(value)


def normalize_url(url: str) -> Optional[str]:
    """Normalize URL."""
    normalizer = LinkNormalizer()
    return normalizer.normalize_url(url)


def parse_number(text: str) -> Optional[Union[int, float]]:
    """Parse numeric value."""
    parser = NumericParser()
    return parser.parse_numeric(text)


def normalize_date(date_value: Any) -> Optional[str]:
    """Normalize date value."""
    normalizer = DateNormalizer()
    return normalizer.normalize_date(date_value)


def convert_column_type(series: pd.Series, target_type: str) -> pd.Series:
    """Convert column to target type."""
    converter = TypeConverter()
    return converter.convert_column(series, target_type)
