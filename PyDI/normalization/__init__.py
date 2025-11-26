"""
Normalization and validation utilities for PyDI.

This subpackage provides tools for data normalization, type detection,
validation, and quality assessment. It integrates external libraries
for specific normalization tasks:

- **Pint**: Physical unit conversions (length, weight, temperature, etc.)
- **pycountry**: Country/currency/language code normalization (ISO standards)
- **python-stdnum**: Standard number formats (ISBN, IBAN, VAT, etc.)
- **phonenumbers**: Phone number parsing and formatting
- **email-validator**: Email validation and normalization

Key Components
--------------

Profiling
~~~~~~~~~
profile_dataframe
    Analyze DataFrame columns to detect types, units, scale modifiers, etc.
ColumnProfile, DataFrameProfile
    Profile result objects with to_dict()/to_json() methods.

Unit Handling (Pint-backed)
~~~~~~~~~~~~~~~~~~~~~~~~~~~
parse_quantity
    Parse "5 km" → ParsedQuantity(magnitude=5, unit="kilometer")
convert_units
    Convert between compatible units (km → miles, celsius → fahrenheit)
normalize_quantity
    Parse and optionally convert units + expand scale modifiers
normalize_column
    Normalize a pandas Series of quantities

Scale Modifiers
~~~~~~~~~~~~~~~
detect_scale_modifier
    Detect MEO, MEUR, million, thousand, etc. in text
expand_scale
    Expand "5 MEO" → 5,000,000
parse_scaled_number
    Parse number with scale modifier

Integrations (External Libraries)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
normalize_country, normalize_currency, normalize_language
    ISO code normalization via pycountry
validate_stdnum, format_stdnum, normalize_stdnum
    Standard number handling via python-stdnum
parse_phone, format_phone, validate_phone
    Phone number handling via phonenumbers
validate_email, normalize_email
    Email handling via email-validator

Usage Examples
--------------

Profile a DataFrame:
>>> from PyDI.normalization import profile_dataframe
>>> profile = profile_dataframe(df)
>>> print(profile.summary())

Normalize quantities:
>>> from PyDI.normalization import normalize_quantity
>>> normalize_quantity("5 km", target_unit="m")
(5000.0, 'meter')
>>> normalize_quantity("10 MEO")
(10000000.0, 'dimensionless')

Convert units:
>>> from PyDI.normalization import convert_units
>>> convert_units(100, "fahrenheit", "celsius")
37.77...

Normalize country codes:
>>> from PyDI.normalization.integrations import normalize_country
>>> normalize_country("Germany")
'DE'
>>> normalize_country("DEU", output_format="name")
'Germany'
"""

from __future__ import annotations

# Profiling
from .profile import (
    ColumnProfile,
    DataFrameProfile,
    profile_dataframe,
    profile_column,
)

# Unit handling (Pint-backed)
from .units import (
    ParsedQuantity,
    parse_quantity,
    convert_units,
    detect_unit,
    normalize_quantity,
    normalize_column,
    normalize_to_base,
    get_dimensionality,
    are_compatible,
    list_compatible_units,
    is_valid_unit,
    list_units,
)

# Scale modifiers
from .scale import (
    ScaleModifier,
    ScaleResult,
    detect_scale_modifier,
    expand_scale,
    parse_scaled_number,
)

# Integrations - re-export commonly used functions
from .integrations import (
    # Country/currency/language (pycountry)
    normalize_country,
    normalize_currency,
    normalize_language,
    lookup_country,
    lookup_currency,
    CountryInfo,
    CurrencyInfo,
    # Standard numbers (python-stdnum)
    detect_stdnum_type,
    validate_stdnum,
    format_stdnum,
    normalize_stdnum,
    is_valid_isbn,
    is_valid_iban,
    is_valid_vat,
    # Phone numbers (phonenumbers)
    parse_phone,
    format_phone,
    validate_phone,
    PhoneInfo,
    # Email (email-validator)
    validate_email,
    normalize_email,
    is_valid_email,
    EmailInfo,
)

# Text normalization utilities
from .text import (
    TextNormalizer,
    HeaderNormalizer,
    TokenizationNormalizer,
    WebTableNormalizer,
    BracketContentHandler,
)

# Type detection and conversion
from .types import (
    CoordinateParser,
    BooleanParser,
    LinkNormalizer,
    NumericParser,
    DateNormalizer,
    TypeConverter,
    parse_coordinate,
    parse_boolean,
    normalize_url,
    parse_number,
)

# Value-level normalization
from .values import (
    AdvancedValueNormalizer,
    NullValueHandler,
    normalize_numeric,
    normalize_date,
    normalize_boolean,
    clean_nulls,
)

# Column analysis and type detection
from .columns import (
    DataTypeExtended,
    ValueDetectionType,
    AdvancedTypeDetector,
    ColumnTypeInference,
    detect_column_types,
    analyze_column_quality,
    detect_dataframe_types,
    infer_column_types,
)

# Dataset-level normalization orchestration
from .datasets import (
    NormalizationConfig,
    ColumnNormalizationResult,
    DatasetNormalizationResult,
    DatasetNormalizer,
    normalize_dataset,
    create_normalization_config,
    load_normalization_config,
    save_normalization_config,
)

# Validators (if they exist)
try:
    from .validators import (
        ValidationResult,
        BaseValidator,
        EmailValidator,
        RangeValidator,
        PatternValidator,
        CompletenessValidator,
        UniqueValidator,
        DataQualityChecker,
        SchemaValidator,
        validate_emails,
        validate_ranges,
        validate_completeness,
        validate_schema,
    )
except ImportError:
    pass

# Detectors (if they exist)
try:
    from .detectors import DataType, NullDetector, OutlierDetector, DuplicateDetector
except ImportError:
    pass


__all__ = [
    # Profiling
    "ColumnProfile",
    "DataFrameProfile",
    "profile_dataframe",
    "profile_column",
    # Unit handling
    "ParsedQuantity",
    "parse_quantity",
    "convert_units",
    "detect_unit",
    "normalize_quantity",
    "normalize_column",
    "normalize_to_base",
    "get_dimensionality",
    "are_compatible",
    "list_compatible_units",
    "is_valid_unit",
    "list_units",
    # Scale modifiers
    "ScaleModifier",
    "ScaleResult",
    "detect_scale_modifier",
    "expand_scale",
    "parse_scaled_number",
    # Country/currency (pycountry)
    "normalize_country",
    "normalize_currency",
    "normalize_language",
    "lookup_country",
    "lookup_currency",
    "CountryInfo",
    "CurrencyInfo",
    # Standard numbers (python-stdnum)
    "detect_stdnum_type",
    "validate_stdnum",
    "format_stdnum",
    "normalize_stdnum",
    "is_valid_isbn",
    "is_valid_iban",
    "is_valid_vat",
    # Phone numbers
    "parse_phone",
    "format_phone",
    "validate_phone",
    "PhoneInfo",
    # Email
    "validate_email",
    "normalize_email",
    "is_valid_email",
    "EmailInfo",
    # Text normalization
    "TextNormalizer",
    "HeaderNormalizer",
    "TokenizationNormalizer",
    "WebTableNormalizer",
    "BracketContentHandler",
    # Type conversion
    "CoordinateParser",
    "BooleanParser",
    "LinkNormalizer",
    "NumericParser",
    "DateNormalizer",
    "TypeConverter",
    "parse_coordinate",
    "parse_boolean",
    "normalize_url",
    "parse_number",
    # Value normalization
    "AdvancedValueNormalizer",
    "NullValueHandler",
    "normalize_numeric",
    "normalize_date",
    "normalize_boolean",
    "clean_nulls",
    # Column analysis
    "DataTypeExtended",
    "ValueDetectionType",
    "AdvancedTypeDetector",
    "ColumnTypeInference",
    "detect_column_types",
    "analyze_column_quality",
    "detect_dataframe_types",
    "infer_column_types",
    # Dataset normalization
    "NormalizationConfig",
    "ColumnNormalizationResult",
    "DatasetNormalizationResult",
    "DatasetNormalizer",
    "normalize_dataset",
    "create_normalization_config",
    "load_normalization_config",
    "save_normalization_config",
]
