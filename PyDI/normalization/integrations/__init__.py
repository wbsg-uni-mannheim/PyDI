"""
Integration wrappers for external normalization libraries.

This module provides unified interfaces to external libraries:
- pint_units: Physical unit conversions via Pint
- country: Country/currency/language codes via pycountry
- stdnum: Standard number formats (ISBN, IBAN, VAT) via python-stdnum
- phone: Phone number parsing via phonenumbers
- email: Email validation via email-validator
- babel_numbers: Locale-aware numeric parsing via Babel
- pydantic_validation: Schema validation via Pydantic
- ftfy_text: Text encoding fixing via ftfy
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

from .pydantic_validation import (
    PydanticValidationResult,
    is_pydantic_available,
    validate_dataframe as validate_dataframe_pydantic,
    validate_dict as validate_dict_pydantic,
    dataframe_to_models,
    models_to_dataframe,
    get_model_fields,
)

from .babel_numbers import (
    BabelParseResult,
    BABEL_AVAILABLE,
    CURRENCY_SYMBOLS,
    DEFAULT_CANDIDATE_LOCALES,
    is_babel_available,
    parse_decimal as parse_decimal_babel,
    infer_locale as infer_locale_babel,
    order_locales_for_value,
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
    # pydantic
    "PydanticValidationResult",
    "is_pydantic_available",
    "validate_dataframe_pydantic",
    "validate_dict_pydantic",
    "dataframe_to_models",
    "models_to_dataframe",
    "get_model_fields",
    # babel
    "BabelParseResult",
    "BABEL_AVAILABLE",
    "CURRENCY_SYMBOLS",
    "DEFAULT_CANDIDATE_LOCALES",
    "is_babel_available",
    "parse_decimal_babel",
    "infer_locale_babel",
    "order_locales_for_value",
]
