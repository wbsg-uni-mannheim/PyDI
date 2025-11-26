"""
pycountry integration for country, currency, and language normalization.

This module wraps the pycountry library to provide:
- Country name/code normalization (ISO 3166)
- Currency code normalization (ISO 4217)
- Language code normalization (ISO 639)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

import pycountry

logger = logging.getLogger(__name__)

CountryFormat = Literal["alpha_2", "alpha_3", "numeric", "name", "official_name"]
CurrencyFormat = Literal["alpha_3", "numeric", "name"]
LanguageFormat = Literal["alpha_2", "alpha_3", "name"]


@dataclass
class CountryInfo:
    """Information about a country."""

    alpha_2: str
    alpha_3: str
    numeric: str
    name: str
    official_name: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha_2": self.alpha_2,
            "alpha_3": self.alpha_3,
            "numeric": self.numeric,
            "name": self.name,
            "official_name": self.official_name,
        }


@dataclass
class CurrencyInfo:
    """Information about a currency."""

    alpha_3: str
    numeric: str
    name: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha_3": self.alpha_3,
            "numeric": self.numeric,
            "name": self.name,
        }


@dataclass
class LanguageInfo:
    """Information about a language."""

    alpha_2: str | None
    alpha_3: str
    name: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha_2": self.alpha_2,
            "alpha_3": self.alpha_3,
            "name": self.name,
        }


def lookup_country(value: str) -> CountryInfo | None:
    """
    Look up a country by name, alpha-2, alpha-3, or numeric code.

    Args:
        value: Country identifier (name, code, etc.)

    Returns:
        CountryInfo object, or None if not found

    Examples:
        >>> lookup_country("DE")
        CountryInfo(alpha_2='DE', alpha_3='DEU', ...)
        >>> lookup_country("Germany")
        CountryInfo(alpha_2='DE', ...)
        >>> lookup_country("276")
        CountryInfo(alpha_2='DE', ...)
    """
    if not value or not isinstance(value, str):
        return None

    value = value.strip()
    if not value:
        return None

    country = None

    # Try exact lookups first
    try:
        country = pycountry.countries.get(alpha_2=value.upper())
    except (KeyError, LookupError):
        pass

    if not country:
        try:
            country = pycountry.countries.get(alpha_3=value.upper())
        except (KeyError, LookupError):
            pass

    if not country:
        try:
            country = pycountry.countries.get(numeric=value.zfill(3))
        except (KeyError, LookupError):
            pass

    if not country:
        try:
            country = pycountry.countries.get(name=value)
        except (KeyError, LookupError):
            pass

    if not country:
        try:
            country = pycountry.countries.get(official_name=value)
        except (KeyError, LookupError):
            pass

    # Try fuzzy search as fallback
    if not country:
        try:
            results = pycountry.countries.search_fuzzy(value)
            if results:
                country = results[0]
        except LookupError:
            pass

    if country:
        return CountryInfo(
            alpha_2=country.alpha_2,
            alpha_3=country.alpha_3,
            numeric=country.numeric,
            name=country.name,
            official_name=getattr(country, "official_name", None),
        )

    return None


def normalize_country(
    value: str,
    output_format: CountryFormat = "alpha_2",
) -> str | None:
    """
    Normalize country name/code to standard format.

    Args:
        value: Country name, alpha-2, alpha-3, or numeric code
        output_format: Output format - "alpha_2", "alpha_3", "numeric", "name", or "official_name"

    Returns:
        Normalized country identifier, or None if not found

    Examples:
        >>> normalize_country("Germany")
        'DE'
        >>> normalize_country("DEU")
        'DE'
        >>> normalize_country("276")
        'DE'
        >>> normalize_country("deutschland")
        'DE'
        >>> normalize_country("Germany", "name")
        'Germany'
        >>> normalize_country("Germany", "alpha_3")
        'DEU'
    """
    info = lookup_country(value)
    if not info:
        return None

    if output_format == "alpha_2":
        return info.alpha_2
    elif output_format == "alpha_3":
        return info.alpha_3
    elif output_format == "numeric":
        return info.numeric
    elif output_format == "name":
        return info.name
    elif output_format == "official_name":
        return info.official_name or info.name
    else:
        return info.alpha_2


def lookup_currency(value: str) -> CurrencyInfo | None:
    """
    Look up a currency by code or name.

    Args:
        value: Currency identifier (alpha-3 code or name)

    Returns:
        CurrencyInfo object, or None if not found

    Examples:
        >>> lookup_currency("EUR")
        CurrencyInfo(alpha_3='EUR', name='Euro', ...)
        >>> lookup_currency("Euro")
        CurrencyInfo(alpha_3='EUR', ...)
    """
    if not value or not isinstance(value, str):
        return None

    value = value.strip()
    if not value:
        return None

    currency = None

    # Try exact lookups
    try:
        currency = pycountry.currencies.get(alpha_3=value.upper())
    except (KeyError, LookupError):
        pass

    if not currency:
        try:
            currency = pycountry.currencies.get(name=value)
        except (KeyError, LookupError):
            pass

    if not currency:
        try:
            currency = pycountry.currencies.get(numeric=value.zfill(3))
        except (KeyError, LookupError):
            pass

    # Try case-insensitive name search
    if not currency:
        value_lower = value.lower()
        for curr in pycountry.currencies:
            if curr.name.lower() == value_lower:
                currency = curr
                break

    if currency:
        return CurrencyInfo(
            alpha_3=currency.alpha_3,
            numeric=getattr(currency, "numeric", ""),
            name=currency.name,
        )

    return None


def normalize_currency(
    value: str,
    output_format: CurrencyFormat = "alpha_3",
) -> str | None:
    """
    Normalize currency code/name to ISO 4217.

    Args:
        value: Currency code or name
        output_format: Output format - "alpha_3", "numeric", or "name"

    Returns:
        Normalized currency identifier, or None if not found

    Examples:
        >>> normalize_currency("EUR")
        'EUR'
        >>> normalize_currency("Euro")
        'EUR'
        >>> normalize_currency("euro")
        'EUR'
        >>> normalize_currency("EUR", "name")
        'Euro'
    """
    info = lookup_currency(value)
    if not info:
        return None

    if output_format == "alpha_3":
        return info.alpha_3
    elif output_format == "numeric":
        return info.numeric
    elif output_format == "name":
        return info.name
    else:
        return info.alpha_3


def lookup_language(value: str) -> LanguageInfo | None:
    """
    Look up a language by code or name.

    Args:
        value: Language identifier (alpha-2, alpha-3, or name)

    Returns:
        LanguageInfo object, or None if not found

    Examples:
        >>> lookup_language("en")
        LanguageInfo(alpha_2='en', alpha_3='eng', name='English')
        >>> lookup_language("English")
        LanguageInfo(alpha_2='en', ...)
    """
    if not value or not isinstance(value, str):
        return None

    value = value.strip()
    if not value:
        return None

    language = None

    # Try exact lookups
    try:
        language = pycountry.languages.get(alpha_2=value.lower())
    except (KeyError, LookupError):
        pass

    if not language:
        try:
            language = pycountry.languages.get(alpha_3=value.lower())
        except (KeyError, LookupError):
            pass

    if not language:
        try:
            language = pycountry.languages.get(name=value)
        except (KeyError, LookupError):
            pass

    # Try case-insensitive name search
    if not language:
        value_lower = value.lower()
        for lang in pycountry.languages:
            if lang.name.lower() == value_lower:
                language = lang
                break

    if language:
        return LanguageInfo(
            alpha_2=getattr(language, "alpha_2", None),
            alpha_3=language.alpha_3,
            name=language.name,
        )

    return None


def normalize_language(
    value: str,
    output_format: LanguageFormat = "alpha_2",
) -> str | None:
    """
    Normalize language code/name to ISO 639.

    Args:
        value: Language code or name
        output_format: Output format - "alpha_2", "alpha_3", or "name"

    Returns:
        Normalized language identifier, or None if not found

    Examples:
        >>> normalize_language("English")
        'en'
        >>> normalize_language("eng")
        'en'
        >>> normalize_language("en", "name")
        'English'
    """
    info = lookup_language(value)
    if not info:
        return None

    if output_format == "alpha_2":
        return info.alpha_2
    elif output_format == "alpha_3":
        return info.alpha_3
    elif output_format == "name":
        return info.name
    else:
        return info.alpha_2


def get_country_info(value: str) -> dict[str, Any] | None:
    """
    Get full country information as a dictionary.

    Args:
        value: Country identifier

    Returns:
        Dictionary with all country info, or None if not found
    """
    info = lookup_country(value)
    return info.to_dict() if info else None


def list_countries() -> list[CountryInfo]:
    """List all countries."""
    return [
        CountryInfo(
            alpha_2=c.alpha_2,
            alpha_3=c.alpha_3,
            numeric=c.numeric,
            name=c.name,
            official_name=getattr(c, "official_name", None),
        )
        for c in pycountry.countries
    ]


def list_currencies() -> list[CurrencyInfo]:
    """List all currencies."""
    return [
        CurrencyInfo(
            alpha_3=c.alpha_3,
            numeric=getattr(c, "numeric", ""),
            name=c.name,
        )
        for c in pycountry.currencies
    ]


def list_languages() -> list[LanguageInfo]:
    """List all languages."""
    return [
        LanguageInfo(
            alpha_2=getattr(lang, "alpha_2", None),
            alpha_3=lang.alpha_3,
            name=lang.name,
        )
        for lang in pycountry.languages
    ]
