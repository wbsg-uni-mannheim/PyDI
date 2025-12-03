"""
python-stdnum integration for standard number format validation and normalization.

This module wraps the python-stdnum library to provide:
- Detection of standard number types (ISBN, IBAN, VAT, etc.)
- Validation of standard numbers
- Formatting/normalization of standard numbers

Supported formats include:
- ISBN: International Standard Book Number
- IBAN: International Bank Account Number
- ISSN: International Standard Serial Number
- EAN: International Article Number (barcode)
- IMEI: International Mobile Equipment Identity
- ISIN: International Securities Identification Number
- VAT: Value Added Tax identification numbers (business tax IDs, EU + others)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from stdnum import iban, isbn, issn, ean, imei, isin
from stdnum.eu import vat
from stdnum.exceptions import InvalidChecksum, InvalidComponent, InvalidFormat, InvalidLength

logger = logging.getLogger(__name__)


@dataclass
class StdnumResult:
    """Result of stdnum validation/normalization."""

    value: str
    compact: str
    formatted: str | None
    stdnum_type: str
    is_valid: bool
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "compact": self.compact,
            "formatted": self.formatted,
            "stdnum_type": self.stdnum_type,
            "is_valid": self.is_valid,
            "error": self.error,
        }


# Mapping of type names to modules
STDNUM_MODULES: dict[str, tuple[str, Any]] = {
    "isbn": ("ISBN", isbn),
    "iban": ("IBAN", iban),
    "issn": ("ISSN", issn),
    "ean": ("EAN", ean),
    "imei": ("IMEI", imei),
    "isin": ("ISIN", isin),
    "vat": ("VAT", vat),
}


def detect_stdnum_type(value: str) -> str | None:
    """
    Detect the type of standard number.

    Args:
        value: The value to check

    Returns:
        Type name (e.g., "ISBN", "IBAN", "VAT"), or None if not detected

    Examples:
        >>> detect_stdnum_type("978-0-306-40615-7")
        'ISBN'
        >>> detect_stdnum_type("DE89370400440532013000")
        'IBAN'
        >>> detect_stdnum_type("DE123456789")
        'VAT'
    """
    if not value or not isinstance(value, str):
        return None

    value = value.strip()
    if not value:
        return None

    # Try each standard module
    for type_name, module in STDNUM_MODULES.values():
        try:
            module.validate(value)
            return type_name
        except Exception:
            pass

    return None


def validate_stdnum(value: str, stdnum_type: str | None = None) -> StdnumResult:
    """
    Validate a standard number.

    Args:
        value: The value to validate
        stdnum_type: Type to validate as (e.g., "ISBN", "IBAN"). If None, auto-detect.

    Returns:
        StdnumResult with validation result

    Examples:
        >>> result = validate_stdnum("978-0-306-40615-7", "ISBN")
        >>> result.is_valid
        True
        >>> result.compact
        '9780306406157'
    """
    if not value or not isinstance(value, str):
        return StdnumResult(
            value=value or "",
            compact="",
            formatted=None,
            stdnum_type=stdnum_type or "unknown",
            is_valid=False,
            error="Empty or invalid input",
        )

    value = value.strip()

    # Auto-detect type if not specified
    if stdnum_type is None:
        stdnum_type = detect_stdnum_type(value)
        if stdnum_type is None:
            return StdnumResult(
                value=value,
                compact=value,
                formatted=None,
                stdnum_type="unknown",
                is_valid=False,
                error="Could not detect standard number type",
            )

    stdnum_type_upper = stdnum_type.upper()
    stdnum_type_lower = stdnum_type.lower()

    # Get the appropriate module
    if stdnum_type_lower not in STDNUM_MODULES:
        return StdnumResult(
            value=value,
            compact=value,
            formatted=None,
            stdnum_type=stdnum_type,
            is_valid=False,
            error=f"Unsupported stdnum type: {stdnum_type}",
        )

    _, module = STDNUM_MODULES[stdnum_type_lower]

    try:
        # Validate
        compact = module.validate(value)
        # Try to format
        formatted = None
        if hasattr(module, "format"):
            try:
                formatted = module.format(value)
            except Exception:
                pass

        return StdnumResult(
            value=value,
            compact=compact,
            formatted=formatted,
            stdnum_type=stdnum_type_upper,
            is_valid=True,
            error=None,
        )
    except InvalidChecksum as e:
        return StdnumResult(
            value=value,
            compact=value.replace(" ", "").replace("-", ""),
            formatted=None,
            stdnum_type=stdnum_type_upper,
            is_valid=False,
            error=f"Invalid checksum: {e}",
        )
    except InvalidLength as e:
        return StdnumResult(
            value=value,
            compact=value.replace(" ", "").replace("-", ""),
            formatted=None,
            stdnum_type=stdnum_type_upper,
            is_valid=False,
            error=f"Invalid length: {e}",
        )
    except InvalidFormat as e:
        return StdnumResult(
            value=value,
            compact=value.replace(" ", "").replace("-", ""),
            formatted=None,
            stdnum_type=stdnum_type_upper,
            is_valid=False,
            error=f"Invalid format: {e}",
        )
    except InvalidComponent as e:
        return StdnumResult(
            value=value,
            compact=value.replace(" ", "").replace("-", ""),
            formatted=None,
            stdnum_type=stdnum_type_upper,
            is_valid=False,
            error=f"Invalid component: {e}",
        )
    except Exception as e:
        return StdnumResult(
            value=value,
            compact=value.replace(" ", "").replace("-", ""),
            formatted=None,
            stdnum_type=stdnum_type_upper,
            is_valid=False,
            error=str(e),
        )


def format_stdnum(value: str, stdnum_type: str | None = None) -> str | None:
    """
    Format a standard number in its canonical form.

    Args:
        value: The value to format
        stdnum_type: Type of number (auto-detected if None)

    Returns:
        Formatted value, or None if formatting fails

    Examples:
        >>> format_stdnum("9780306406157", "ISBN")
        '978-0-306-40615-7'
        >>> format_stdnum("DE89370400440532013000", "IBAN")
        'DE89 3704 0044 0532 0130 00'
    """
    result = validate_stdnum(value, stdnum_type)
    return result.formatted if result.is_valid else None


def normalize_stdnum(value: str, stdnum_type: str | None = None) -> str | None:
    """
    Normalize a standard number to its compact form.

    Args:
        value: The value to normalize
        stdnum_type: Type of number (auto-detected if None)

    Returns:
        Compact normalized value, or None if validation fails

    Examples:
        >>> normalize_stdnum("978-0-306-40615-7", "ISBN")
        '9780306406157'
        >>> normalize_stdnum("DE89 3704 0044 0532 0130 00", "IBAN")
        'DE89370400440532013000'
    """
    result = validate_stdnum(value, stdnum_type)
    return result.compact if result.is_valid else None


def list_supported_formats() -> list[str]:
    """
    List all supported standard number formats.

    Returns:
        List of format names with descriptions
    """
    return [
        "ISBN - International Standard Book Number",
        "IBAN - International Bank Account Number",
        "ISSN - International Standard Serial Number",
        "EAN - International Article Number (barcode)",
        "IMEI - International Mobile Equipment Identity",
        "ISIN - International Securities Identification Number",
        "VAT - Value Added Tax identification number",
    ]


def is_valid_isbn(value: str) -> bool:
    """Check if value is a valid ISBN."""
    try:
        isbn.validate(value)
        return True
    except Exception:
        return False


def is_valid_iban(value: str) -> bool:
    """Check if value is a valid IBAN."""
    try:
        iban.validate(value)
        return True
    except Exception:
        return False


def is_valid_vat(value: str) -> bool:
    """
    Check if value is a valid VAT number.

    VAT = Value Added Tax identification number, used by businesses
    in the EU and other countries for tax purposes.

    Args:
        value: VAT number to check

    Returns:
        True if valid, False otherwise
    """
    try:
        vat.validate(value)
        return True
    except Exception:
        return False


def get_iban_country(value: str) -> str | None:
    """
    Get the country code from an IBAN.

    The first two characters of an IBAN are always the ISO 3166-1 alpha-2
    country code. Use the country module to get full country info.

    Args:
        value: IBAN value

    Returns:
        Two-letter country code, or None if invalid

    Examples:
        >>> get_iban_country("DE89370400440532013000")
        'DE'
    """
    try:
        compact = iban.validate(value)
        return compact[:2]
    except Exception:
        return None


def get_vat_country(value: str) -> str | None:
    """
    Get the country code from a VAT number.

    EU VAT numbers typically start with a two-letter country code.
    Use the country module to get full country info.

    Args:
        value: VAT number

    Returns:
        Two-letter country code, or None if invalid/not detected

    Examples:
        >>> get_vat_country("DE123456789")
        'DE'
    """
    try:
        compact = vat.validate(value)
        # VAT numbers typically start with country code
        if len(compact) >= 2 and compact[:2].isalpha():
            return compact[:2].upper()
        return None
    except Exception:
        return None
