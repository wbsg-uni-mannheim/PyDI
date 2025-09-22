"""
Normalization and validation utilities for PyDI.

This subpackage provides comprehensive tools for data normalization, type detection,
validation, and quality assessment inspired by the Winter framework. It includes 
automatic type detection, specialized normalizers for different data types, unit 
handling, advanced text processing, and validation utilities for data quality assurance.

Key Components
--------------
- Advanced type detection and pattern matching with Winter framework patterns
- Comprehensive null/missing value detection and handling 
- Text normalization with tokenization, stemming, and web table cleaning
- Numeric normalization with unit detection, parsing, and conversion
- Date/time parsing and standardization
- Coordinate parsing and validation
- URL normalization and validation
- Enhanced boolean parsing with multi-language support
- Dataset-level normalization pipelines
- Data validation and quality checking
- Pandas-first approach with metadata preservation

Main Classes
------------

Core Normalization
~~~~~~~~~~~~~~~~~~
DatasetNormalizer
    Comprehensive dataset normalization with configurable pipelines and unit-aware value handling.

Type Detection and Conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
DataType
    Enumeration of supported data types.
TypeDetectionResult
    Result object containing type detection information and confidence scores.
NullDetector
    Comprehensive null/missing value detector with extensive pattern matching.
TypeConverter
    Main type converter with unit support and specialized parsers.

Text Normalization
~~~~~~~~~~~~~~~~~~
TextNormalizer
    Basic text cleaning and normalization.
HeaderNormalizer
    Specialized normalizer for column headers with HTML cleaning.
TokenizationNormalizer
    Advanced text tokenization with stemming and stop word removal.
WebTableNormalizer
    Specialized normalizer for web-scraped table data.
BracketContentHandler
    Utility for handling content in brackets.

Numeric and Unit Handling
~~~~~~~~~~~~~~~~~~~~~~~~~
UnitNormalizer
    Main normalizer for quantities with units.
UnitRegistry
    Registry of known units and their conversion factors.
UnitDetector
    Detector for units in text with quantities.
UnitConverter
    Converter for units within the same category.
QuantityParser
    Parser for extracting numeric quantities from text.

Specialized Type Converters
~~~~~~~~~~~~~~~~~~~~~~~~~~~
CoordinateParser
    Parser and validator for geographic coordinates.
BooleanParser
    Boolean parser supporting multiple languages and formats.
LinkNormalizer
    URL and link normalization and validation.
NumericParser
    Numeric parser with locale support and format detection.

Date and Time
~~~~~~~~~~~~~
DateNormalizer
    Date parsing, formatting, and standardization.

Validation and Quality
~~~~~~~~~~~~~~~~~~~~~~
DataQualityChecker
    Comprehensive data quality assessment tool.
EmailValidator, RangeValidator, PatternValidator, CompletenessValidator, UniqueValidator
    Specialized validators for different data quality checks.
SchemaValidator
    Validator for DataFrame schema compliance.
ValidationResult
    Result object containing validation errors, warnings, and quality metrics.

Dataset-Level Operations
~~~~~~~~~~~~~~~~~~~~~~~~
ColumnTypeInference
    Enhanced column type inference with confidence scoring.

Winter-Equivalent Components
~~~~~~~~~~~~~~~~~~~~~~~~~~~
AdvancedTypeDetector
    Sophisticated type detector with majority voting and pattern recognition.
AdvancedValueNormalizer
    Advanced value normalizer with unit conversion and quantity scaling.
ValueDetectionType
    Comprehensive type detection result with confidence and metadata.
QuantityModifier
    Quantity scaling modifiers (thousand, million, billion, etc.).

Convenience Functions
~~~~~~~~~~~~~~~~~~~~
normalize_dataset
    Quick dataset normalization with sensible defaults.
detect_column_types, detect_dataframe_types
    Column type detection (basic and advanced).
parse_quantity, detect_unit, normalize_units, convert_units
    Unit-related convenience functions.
parse_coordinate, parse_boolean, normalize_url, parse_number, normalize_date
    Type-specific parsing functions.

Usage Examples
--------------

Basic normalization:
>>> import pandas as pd
>>> from PyDI.normalization import normalize_dataset
>>> df = pd.read_csv("data.csv")
>>> normalized_df, result = normalize_dataset(df)

Advanced normalization with custom settings:
>>> from PyDI.normalization import DatasetNormalizer, create_normalization_config
>>> config = create_normalization_config(enable_unit_conversion=True, enable_quantity_scaling=True)
>>> normalizer = DatasetNormalizer(config)
>>> normalized_df, result = normalizer.normalize_dataset(df)

Web table normalization:
>>> from PyDI.normalization import WebTableNormalizer
>>> web_normalizer = WebTableNormalizer(remove_brackets_content=True)
>>> cleaned_df = web_normalizer.normalize_dataframe(df)

Unit normalization:
>>> from PyDI.normalization import UnitNormalizer
>>> unit_normalizer = UnitNormalizer()
>>> normalized_value, unit = unit_normalizer.normalize_value("5.2 km")

Type detection:
>>> from PyDI.normalization import AdvancedTypeDetector
>>> detector = AdvancedTypeDetector()
>>> types = detector.detect_dataframe_types(df)

Data quality assessment:
>>> from PyDI.normalization import DataQualityChecker, CompletenessValidator
>>> checker = DataQualityChecker()
>>> checker.add_validator(CompletenessValidator(required_columns=['id', 'name']))
>>> quality_result = checker.assess_quality(df)

Enterprise-grade comprehensive normalization:
>>> from PyDI.normalization import DatasetNormalizer, create_normalization_config
>>> config = create_normalization_config()
>>> normalizer = DatasetNormalizer(config)
>>> normalized_df, result = normalizer.normalize_dataset(df)

Advanced unit handling:
>>> from PyDI.normalization import UnitRegistry, AdvancedValueNormalizer
>>> registry = UnitRegistry(comprehensive=True)
>>> normalizer = AdvancedValueNormalizer(registry)
>>> normalized = normalizer.normalize_value("5.2 thousand km", data_type='numeric')
>>> # Returns normalized value in base units (metres)

For more examples, see the examples/ directory.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

# Text normalization utilities
from .text import (
    TextNormalizer,
    HeaderNormalizer,
    TokenizationNormalizer,
    WebTableNormalizer,
    BracketContentHandler
)

# Unit handling and conversion
from .units import (
    UnitCategory,
    QuantityModifier,
    Unit,
    UnitRegistry,
    QuantityParser,
    UnitDetector,
    UnitConverter,
    UnitNormalizer,
    # Convenience functions
    parse_quantity,
    detect_unit,
    normalize_units,
    convert_units
)

# Type detection and conversion
from .types import (
    CoordinateParser,
    BooleanParser,
    LinkNormalizer,
    NumericParser,
    DateNormalizer,
    TypeConverter,
    # Convenience functions
    parse_coordinate,
    parse_boolean,
    normalize_url,
    parse_number
)

# Value-level normalization
from .values import (
    AdvancedValueNormalizer,
    NullValueHandler,
    # Convenience functions
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
    # Convenience functions
    detect_column_types,
    analyze_column_quality,
    detect_dataframe_types,
    infer_column_types
)

# Dataset-level normalization orchestration
from .datasets import (
    NormalizationConfig,
    ColumnNormalizationResult,
    DatasetNormalizationResult,
    DatasetNormalizer,
    # Convenience functions
    normalize_dataset,
    create_normalization_config,
    load_normalization_config,
    save_normalization_config
)

# Legacy validators (if they exist)
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
        # Convenience functions
        validate_emails,
        validate_ranges,
        validate_completeness,
        validate_schema
    )
except ImportError:
    # Validators module doesn't exist yet
    pass

# Legacy detectors (if they exist)
try:
    from .detectors import DataType, NullDetector, OutlierDetector, DuplicateDetector
except ImportError:
    # Detectors module doesn't exist yet
    pass

# Define what gets exported when using "from PyDI.normalization import *"
__all__ = [
    # Text normalization utilities
    'TextNormalizer',
    'HeaderNormalizer',
    'TokenizationNormalizer',
    'WebTableNormalizer',
    'BracketContentHandler',

    # Unit handling and conversion
    'UnitCategory',
    'QuantityModifier',
    'Unit',
    'UnitRegistry',
    'QuantityParser',
    'UnitDetector',
    'UnitConverter',
    'UnitNormalizer',

    # Type detection and conversion
    'CoordinateParser',
    'BooleanParser',
    'LinkNormalizer',
    'NumericParser',
    'DateNormalizer',
    'TypeConverter',

    # Value-level normalization
    'AdvancedValueNormalizer',
    'NullValueHandler',

    # Column analysis and type detection
    'DataTypeExtended',
    'ValueDetectionType',
    'AdvancedTypeDetector',
    'ColumnTypeInference',

    # Dataset-level normalization orchestration
    'NormalizationConfig',
    'ColumnNormalizationResult',
    'DatasetNormalizationResult',
    'DatasetNormalizer',

    # Convenience functions - Unit handling
    'parse_quantity',
    'detect_unit',
    'normalize_units',
    'convert_units',

    # Convenience functions - Type conversion
    'parse_coordinate',
    'parse_boolean',
    'normalize_url',
    'parse_number',

    # Convenience functions - Value normalization
    'normalize_numeric',
    'normalize_date',
    'normalize_boolean',
    'clean_nulls',

    # Convenience functions - Column analysis
    'detect_column_types',
    'analyze_column_quality',
    'detect_dataframe_types',
    'infer_column_types',

    # Convenience functions - Dataset normalization
    'normalize_dataset',
    'create_normalization_config',
    'load_normalization_config',
    'save_normalization_config',

]
