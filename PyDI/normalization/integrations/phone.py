"""
phonenumbers integration for phone number parsing and formatting.

This module wraps Google's phonenumbers library to provide:
- Phone number parsing
- Format conversion (E.164, international, national)
- Validation
- Country detection
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

import phonenumbers
from phonenumbers import PhoneNumberFormat, NumberParseException, geocoder

logger = logging.getLogger(__name__)

PhoneFormat = Literal["E164", "INTERNATIONAL", "NATIONAL", "RFC3966"]


@dataclass
class PhoneInfo:
    """Information about a parsed phone number."""

    original: str
    e164: str | None
    international: str | None
    national: str | None
    country_code: int | None
    region: str | None
    is_valid: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "original": self.original,
            "e164": self.e164,
            "international": self.international,
            "national": self.national,
            "country_code": self.country_code,
            "region": self.region,
            "is_valid": self.is_valid,
        }


def parse_phone(
    value: str,
    default_region: str | None = None,
) -> PhoneInfo | None:
    """
    Parse a phone number and extract information.

    Args:
        value: Phone number string
        default_region: Default country code (e.g., "US", "DE") for numbers without country code

    Returns:
        PhoneInfo with parsed data, or None if parsing fails completely

    Examples:
        >>> info = parse_phone("+49 30 12345678")
        >>> info.region
        'DE'
        >>> info.e164
        '+493012345678'
        >>> parse_phone("030 12345678", default_region="DE").e164
        '+493012345678'
    """
    if not value or not isinstance(value, str):
        return None

    value = value.strip()
    if not value:
        return None

    # Convert international dialing prefix 00 to +
    if value.startswith("00") and len(value) > 2 and value[2].isdigit():
        value = "+" + value[2:]

    try:
        parsed = phonenumbers.parse(value, default_region)

        # Get various formats
        e164 = None
        international = None
        national = None

        try:
            e164 = phonenumbers.format_number(parsed, PhoneNumberFormat.E164)
        except Exception:
            pass

        try:
            international = phonenumbers.format_number(parsed, PhoneNumberFormat.INTERNATIONAL)
        except Exception:
            pass

        try:
            national = phonenumbers.format_number(parsed, PhoneNumberFormat.NATIONAL)
        except Exception:
            pass

        # Get region code (ISO country code)
        region = phonenumbers.region_code_for_number(parsed)

        return PhoneInfo(
            original=value,
            e164=e164,
            international=international,
            national=national,
            country_code=parsed.country_code,
            region=region,
            is_valid=phonenumbers.is_valid_number(parsed),
        )

    except NumberParseException as e:
        logger.debug(f"Failed to parse phone number '{value}': {e}")
        return PhoneInfo(
            original=value,
            e164=None,
            international=None,
            national=None,
            country_code=None,
            region=None,
            is_valid=False,
        )
    except Exception as e:
        logger.debug(f"Unexpected error parsing phone number '{value}': {e}")
        return None


def format_phone(
    value: str,
    format: PhoneFormat = "E164",
    default_region: str | None = None,
) -> str | None:
    """
    Format a phone number in a specific format.

    Args:
        value: Phone number string
        format: Output format - "E164", "INTERNATIONAL", "NATIONAL", or "RFC3966"
        default_region: Default country code for numbers without country code

    Returns:
        Formatted phone number, or None if parsing fails

    Examples:
        >>> format_phone("+49 30 12345678", "E164")
        '+493012345678'
        >>> format_phone("+49 30 12345678", "INTERNATIONAL")
        '+49 30 12345678'
        >>> format_phone("+49 30 12345678", "NATIONAL")
        '030 12345678'
    """
    if not value or not isinstance(value, str):
        return None

    value = value.strip()

    # Convert international dialing prefix 00 to +
    if value.startswith("00") and len(value) > 2 and value[2].isdigit():
        value = "+" + value[2:]

    try:
        parsed = phonenumbers.parse(value, default_region)

        format_map = {
            "E164": PhoneNumberFormat.E164,
            "INTERNATIONAL": PhoneNumberFormat.INTERNATIONAL,
            "NATIONAL": PhoneNumberFormat.NATIONAL,
            "RFC3966": PhoneNumberFormat.RFC3966,
        }

        pn_format = format_map.get(format.upper(), PhoneNumberFormat.E164)
        return phonenumbers.format_number(parsed, pn_format)

    except NumberParseException:
        return None
    except Exception:
        return None


def validate_phone(
    value: str,
    default_region: str | None = None,
) -> bool:
    """
    Check if a phone number is valid.

    Args:
        value: Phone number string
        default_region: Default country code for numbers without country code

    Returns:
        True if valid, False otherwise

    Examples:
        >>> validate_phone("+49 30 12345678")
        True
        >>> validate_phone("invalid")
        False
    """
    info = parse_phone(value, default_region)
    return info.is_valid if info else False


def get_phone_info(
    value: str,
    default_region: str | None = None,
) -> dict[str, Any] | None:
    """
    Get full phone number information as a dictionary.

    Args:
        value: Phone number string
        default_region: Default country code

    Returns:
        Dictionary with phone info, or None if parsing fails
    """
    info = parse_phone(value, default_region)
    return info.to_dict() if info else None
