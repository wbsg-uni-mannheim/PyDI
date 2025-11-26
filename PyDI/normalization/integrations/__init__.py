"""
Integration wrappers for external normalization libraries.

This module provides unified interfaces to external libraries:
- pint_units: Physical unit conversions via Pint
- country: Country/currency/language codes via pycountry
- stdnum: Standard number formats (ISBN, IBAN, VAT) via python-stdnum
- phone: Phone number parsing via phonenumbers
- email: Email validation via email-validator
"""

from .pint_units import (
    ParsedQuantity,
    get_registry,
    parse_quantity,
    convert_units,
    normalize_to_base,
    get_unit_dimensionality,
    is_compatible,
    get_compatible_units,
    list_units,
    list_dimensionalities,
    is_valid_unit,
    detect_unit_in_text,
)

from .country import (
    CountryInfo,
    CurrencyInfo,
    LanguageInfo,
    normalize_country,
    normalize_currency,
    normalize_language,
    lookup_country,
    lookup_currency,
    lookup_language,
    get_country_info,
    list_countries,
    list_currencies,
    list_languages,
)

from .stdnum import (
    StdnumResult,
    detect_stdnum_type,
    validate_stdnum,
    format_stdnum,
    normalize_stdnum,
    list_supported_formats,
    is_valid_isbn,
    is_valid_iban,
    is_valid_vat,
    get_iban_country,
    get_vat_country,
)

from .phone import (
    PhoneInfo,
    parse_phone,
    format_phone,
    validate_phone,
    get_phone_info,
)

from .email import (
    EmailInfo,
    validate_email,
    normalize_email,
    is_valid_email,
)

__all__ = [
    # pint_units
    "ParsedQuantity",
    "get_registry",
    "parse_quantity",
    "convert_units",
    "normalize_to_base",
    "get_unit_dimensionality",
    "is_compatible",
    "get_compatible_units",
    "list_units",
    "list_dimensionalities",
    "is_valid_unit",
    "detect_unit_in_text",
    # country
    "CountryInfo",
    "CurrencyInfo",
    "LanguageInfo",
    "normalize_country",
    "normalize_currency",
    "normalize_language",
    "lookup_country",
    "lookup_currency",
    "lookup_language",
    "get_country_info",
    "list_countries",
    "list_currencies",
    "list_languages",
    # stdnum
    "StdnumResult",
    "detect_stdnum_type",
    "validate_stdnum",
    "format_stdnum",
    "normalize_stdnum",
    "list_supported_formats",
    "is_valid_isbn",
    "is_valid_iban",
    "is_valid_vat",
    "get_iban_country",
    "get_vat_country",
    # phone
    "PhoneInfo",
    "parse_phone",
    "format_phone",
    "validate_phone",
    "get_phone_info",
    # email
    "EmailInfo",
    "validate_email",
    "normalize_email",
    "is_valid_email",
]
