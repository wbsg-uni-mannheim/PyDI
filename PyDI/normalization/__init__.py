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

Specification & Transformation (New API)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
NormalizationSpec
    Define how columns should be normalized
ColumnSpec
    Per-column normalization settings
transform_dataframe
    Apply transformations according to spec
normalize_dataframe
    Main entry point with optional auto-detection

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

Full workflow with spec:
>>> from PyDI.normalization import (
...     profile_dataframe, NormalizationSpec, transform_dataframe
... )
>>> profile = profile_dataframe(df)
>>> spec = NormalizationSpec.from_profile(profile)
>>> # Or manually:
>>> spec = NormalizationSpec()
>>> spec.set_column("revenue", expand_scale_modifiers=True, output_type="float")
>>> spec.set_column("country", country_format="alpha_2")
>>> result = transform_dataframe(df, spec)
>>> normalized_df = result.dataframe

Auto-normalization:
>>> from PyDI.normalization import normalize_dataframe
>>> normalized_df = normalize_dataframe(df, auto=True)
"""

from __future__ import annotations

# Profiling
from .profile import (
    ColumnProfile,
    DataFrameProfile,
    DataTypeExtended,
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

# Specification and transformation (new API)
from .spec import (
    ColumnSpec,
    NormalizationSpec,
)

# JSON Schema integration
from .json_schema import (
    load_schema,
    load_normalization_spec,
    load_validation_spec,
)

from .transform import (
    TransformResult,
    DataFrameTransformResult,
    transform_column,
    transform_dataframe,
    normalize_dataframe,
)

# Taxonomy normalization
from .taxonomy import (
    TaxonomyMappingResult,
    TaxonomyLoader,
    TaxonomyMapper,
    load_mapping_cache,
    save_mapping_cache,
    apply_taxonomy_mapping,
)

# Validators
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
    PydanticSchemaValidator,
    validate_emails,
    validate_ranges,
    validate_completeness,
    validate_schema,
    validate_with_pydantic,
)



__all__ = [
    # Profiling
    "ColumnProfile",
    "DataFrameProfile",
    "DataTypeExtended",
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
    # Dataset normalization
    "NormalizationConfig",
    "ColumnNormalizationResult",
    "DatasetNormalizationResult",
    "DatasetNormalizer",
    "normalize_dataset",
    "create_normalization_config",
    "load_normalization_config",
    "save_normalization_config",
    # Specification and transformation (new API)
    "ColumnSpec",
    "NormalizationSpec",
    "TransformResult",
    "DataFrameTransformResult",
    "transform_column",
    "transform_dataframe",
    "normalize_dataframe",
    # Taxonomy normalization
    "TaxonomyMappingResult",
    "TaxonomyLoader",
    "TaxonomyMapper",
    "load_mapping_cache",
    "save_mapping_cache",
    "apply_taxonomy_mapping",
    # JSON Schema integration
    "load_schema",
    "load_normalization_spec",
    "load_validation_spec",
    # Validators
    "ValidationResult",
    "BaseValidator",
    "EmailValidator",
    "RangeValidator",
    "PatternValidator",
    "CompletenessValidator",
    "UniqueValidator",
    "DataQualityChecker",
    "SchemaValidator",
    "PydanticSchemaValidator",
    "validate_emails",
    "validate_ranges",
    "validate_completeness",
    "validate_schema",
    "validate_with_pydantic",
]
