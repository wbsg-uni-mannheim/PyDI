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
NormalizationPipeline
    Chain multiple normalizers with automatic type detection.
DataSetNormalizer
    Comprehensive dataset normalization with configurable pipelines.
DataFrameNormalizer
    Enterprise-grade normalizer matching Winter's DataSetNormalizer scope and capabilities.
normalize_dataframe
    Convenience function for quick DataFrame normalization.
normalize_dataframe_comprehensive
    Winter-equivalent comprehensive normalization with sophisticated type detection and unit conversion.

Type Detection and Conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
TypeDetector
    Enhanced automatic detection of column data types using comprehensive patterns.
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
NumericNormalizer
    Numeric data scaling, outlier handling, and conversion.
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
EnhancedBooleanParser
    Enhanced boolean parser supporting multiple languages and formats.
LinkNormalizer
    URL and link normalization and validation.
EnhancedNumericParser
    Enhanced numeric parser with locale support and format detection.

Date and Time
~~~~~~~~~~~~~
DateNormalizer
    Date parsing, formatting, and standardization.

Validation and Quality
~~~~~~~~~~~~~~~~~~~~~~
DataQualityChecker
    Comprehensive data quality assessment tool.
BaseValidator
    Abstract base class for validators.
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
BatchNormalizationEngine
    Engine for coordinating multiple normalization operations.

Winter-Equivalent Components
~~~~~~~~~~~~~~~~~~~~~~~~~~~
DataFrameNormalizer
    Comprehensive normalizer matching Winter's DataSetNormalizer with enterprise features.
ComprehensiveUnitRegistry
    Extensive unit registry with 20+ categories and 500+ units.
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
normalize_dataframe
    Quick dataset normalization with sensible defaults.
detect_column_types
    Simple column type detection.
get_quality_report
    Quick data quality assessment.
parse_quantity, detect_unit, normalize_units
    Unit-related convenience functions.
parse_coordinate, parse_boolean, normalize_url, parse_number
    Type-specific parsing functions.

Usage Examples
--------------

Basic normalization:
>>> import pandas as pd
>>> from PyDI.normalization import normalize_dataframe
>>> df = pd.read_csv("data.csv")
>>> normalized_df = normalize_dataframe(df)

Advanced normalization with custom settings:
>>> from PyDI.normalization import DataSetNormalizer
>>> normalizer = DataSetNormalizer(
...     normalize_headers=True,
...     normalize_web_content=True,
...     normalize_units=True,
...     output_dir="normalization_output"
... )
>>> result = normalizer.normalize_dataset(df)

Web table normalization:
>>> from PyDI.normalization import WebTableNormalizer
>>> web_normalizer = WebTableNormalizer(remove_brackets_content=True)
>>> cleaned_df = web_normalizer.normalize_dataframe(df)

Unit normalization:
>>> from PyDI.normalization import UnitNormalizer
>>> unit_normalizer = UnitNormalizer()
>>> normalized_value, unit = unit_normalizer.normalize_value("5.2 km")

Type detection:
>>> from PyDI.normalization import TypeDetector
>>> detector = TypeDetector()
>>> types = detector.detect_dataframe_types(df)

Data quality assessment:
>>> from PyDI.normalization import DataQualityChecker, CompletenessValidator
>>> checker = DataQualityChecker()
>>> checker.add_validator(CompletenessValidator(required_columns=['id', 'name']))
>>> quality_result = checker.assess_quality(df)

Enterprise-grade comprehensive normalization (Winter equivalent):
>>> from PyDI.normalization import DataFrameNormalizer, normalize_dataframe_comprehensive
>>> # Quick comprehensive normalization
>>> normalized_df = normalize_dataframe_comprehensive(df, output_dir="normalization_results")
>>> 
>>> # Advanced usage with custom configuration
>>> normalizer = DataFrameNormalizer(
...     confidence_threshold=0.8,
...     enable_logging=True,
...     output_dir="detailed_normalization"
... )
>>> result = normalizer.normalize_dataset(df)
>>> 
>>> # Access comprehensive normalization metadata
>>> print(result.attrs['normalization']['column_types'])
>>> print(f"Normalization took {result.attrs['normalization']['duration_seconds']:.2f} seconds")

Advanced unit handling:
>>> from PyDI.normalization import ComprehensiveUnitRegistry, AdvancedValueNormalizer
>>> registry = ComprehensiveUnitRegistry()
>>> normalizer = AdvancedValueNormalizer(registry)
>>> normalized = normalizer.normalize_value("5.2 thousand kilometers", detection_type)
>>> # Returns normalized value in base units (metres)

For more examples, see the examples/ directory.
"""

from __future__ import annotations

import logging
import re
import string
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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
    ValueNormalizer,
    AdvancedValueNormalizer,
    ListValueProcessor,
    NullValueHandler,
    # Convenience functions
    normalize_value,
    normalize_numeric,
    normalize_date,
    normalize_boolean,
    normalize_list,
    normalize_coordinate,
    clean_nulls,
    advanced_normalize_value
)

# Column analysis and type detection
from .columns import (
    DataTypeExtended,
    ValueDetectionType,
    TypeDetector,
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

logger = logging.getLogger(__name__)

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
    'ValueNormalizer',
    'AdvancedValueNormalizer',
    'ListValueProcessor',
    'NullValueHandler',

    # Column analysis and type detection
    'DataTypeExtended',
    'ValueDetectionType',
    'TypeDetector',
    'AdvancedTypeDetector',
    'ColumnTypeInference',

    # Dataset-level normalization orchestration
    'NormalizationConfig',
    'ColumnNormalizationResult',
    'DatasetNormalizationResult',
    'DatasetNormalizer',

    # Legacy classes (backward compatibility)
    'NumericNormalizer',
    'NormalizationPipeline',
    'Normalizer',
    'PydanticValidator',

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
    'normalize_value',
    'normalize_numeric',
    'normalize_date',
    'normalize_boolean',
    'normalize_list',
    'normalize_coordinate',
    'clean_nulls',
    'advanced_normalize_value',

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

    # Legacy convenience functions
    'normalize_dataframe_simple',
]


class BaseNormalizer(ABC):
    """Abstract base class for normalisers."""

    @abstractmethod
    def apply(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        raise NotImplementedError


class Normalizer(BaseNormalizer):
    """Apply a set of normalisation functions to columns.

    Parameters
    ----------
    rules : dict
        A mapping from column names to callables that take a value and return a normalised value.
    """

    def __init__(self, rules: Dict[str, Callable[[Any], Any]]) -> None:
        self.rules = rules

    def apply(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        result = df.copy()

        # Use specified columns or all columns in rules
        target_columns = columns if columns is not None else list(
            self.rules.keys())

        for col in target_columns:
            if col in result.columns and col in self.rules:
                result[col] = result[col].apply(self.rules[col])
        return result


class TextNormalizer(BaseNormalizer):
    """Comprehensive text normalization inspired by Winter's string normalization.

    Parameters
    ----------
    lowercase : bool, default True
        Convert text to lowercase.
    strip_whitespace : bool, default True
        Remove leading/trailing whitespace and normalize internal whitespace.
    remove_html : bool, default True
        Remove HTML tags and entities.
    remove_punctuation : bool, default False
        Remove punctuation characters.
    fix_encoding : bool, default True
        Fix common encoding issues.
    normalize_unicode : bool, default True
        Normalize Unicode characters to standard forms.
    """

    def __init__(
        self,
        lowercase: bool = True,
        strip_whitespace: bool = True,
        remove_html: bool = True,
        remove_punctuation: bool = False,
        fix_encoding: bool = True,
        normalize_unicode: bool = True,
    ):
        self.lowercase = lowercase
        self.strip_whitespace = strip_whitespace
        self.remove_html = remove_html
        self.remove_punctuation = remove_punctuation
        self.fix_encoding = fix_encoding
        self.normalize_unicode = normalize_unicode

        # HTML cleaning patterns
        self.html_pattern = re.compile(r'<[^>]+>')
        self.html_entity_pattern = re.compile(r'&[^;]+;')

        # Common HTML entities
        self.html_entities = {
            '&nbsp;': ' ', '&amp;': '&', '&lt;': '<', '&gt;': '>',
            '&quot;': '"', '&apos;': "'", '&ndash;': '-', '&mdash;': '-'
        }

    def _clean_text(self, text: str) -> str:
        """Apply all configured text cleaning operations."""
        if pd.isna(text):
            return text

        text = str(text)

        # Fix encoding issues
        if self.fix_encoding:
            try:
                import ftfy
                text = ftfy.fix_text(text)
            except ImportError:
                # Fallback: basic encoding fixes
                text = text.encode('utf-8', errors='ignore').decode('utf-8')

        # Normalize Unicode
        if self.normalize_unicode:
            import unicodedata
            text = unicodedata.normalize('NFKC', text)

        # Remove HTML tags and entities
        if self.remove_html:
            text = self.html_pattern.sub('', text)
            for entity, replacement in self.html_entities.items():
                text = text.replace(entity, replacement)
            # Remove remaining entities
            text = self.html_entity_pattern.sub(' ', text)

        # Normalize whitespace
        if self.strip_whitespace:
            text = re.sub(r'\s+', ' ', text).strip()

        # Convert to lowercase
        if self.lowercase:
            text = text.lower()

        # Remove punctuation
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))

        return text

    def apply(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Apply text normalization to specified columns or all text columns."""
        result = df.copy()

        if columns is None:
            # Apply to all object/string columns
            columns = result.select_dtypes(include=['object']).columns.tolist()

        for col in columns:
            if col in result.columns:
                logger.info(f"Normalizing text in column: {col}")
                result[col] = result[col].apply(self._clean_text)

        return result


class NumericNormalizer(BaseNormalizer):
    """Numeric data normalization and scaling.

    Parameters
    ----------
    method : str, default 'standard'
        Normalization method: 'standard', 'minmax', 'robust', or 'none'.
    handle_outliers : bool, default False
        Whether to clip outliers before normalization.
    outlier_method : str, default 'iqr'
        Method for outlier detection: 'iqr', 'zscore', or 'percentile'.
    outlier_threshold : float, default 1.5
        Threshold for outlier detection.
    """

    def __init__(
        self,
        method: str = 'standard',
        handle_outliers: bool = False,
        outlier_method: str = 'iqr',
        outlier_threshold: float = 1.5,
    ):
        self.method = method
        self.handle_outliers = handle_outliers
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.scalers = {}

    def _handle_outliers(self, series: pd.Series) -> pd.Series:
        """Clip outliers in numeric series."""
        if self.outlier_method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - self.outlier_threshold * IQR
            upper = Q3 + self.outlier_threshold * IQR
            return series.clip(lower=lower, upper=upper)
        elif self.outlier_method == 'zscore':
            mean = series.mean()
            std = series.std()
            lower = mean - self.outlier_threshold * std
            upper = mean + self.outlier_threshold * std
            return series.clip(lower=lower, upper=upper)
        elif self.outlier_method == 'percentile':
            lower = series.quantile(self.outlier_threshold / 100)
            upper = series.quantile(1 - self.outlier_threshold / 100)
            return series.clip(lower=lower, upper=upper)
        else:
            return series

    def apply(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Apply numeric normalization to specified columns or all numeric columns."""
        result = df.copy()

        if columns is None:
            # Apply to all numeric columns
            columns = result.select_dtypes(
                include=[np.number]).columns.tolist()

        for col in columns:
            if col in result.columns:
                logger.info(
                    f"Normalizing numeric column: {col} using {self.method}")

                # Convert to numeric, coercing errors to NaN
                numeric_series = pd.to_numeric(result[col], errors='coerce')

                # Handle outliers if requested
                if self.handle_outliers:
                    numeric_series = self._handle_outliers(numeric_series)

                # Apply normalization
                if self.method == 'standard':
                    scaler = StandardScaler()
                elif self.method == 'minmax':
                    scaler = MinMaxScaler()
                elif self.method == 'robust':
                    from sklearn.preprocessing import RobustScaler
                    scaler = RobustScaler()
                elif self.method == 'none':
                    result[col] = numeric_series
                    continue
                else:
                    raise ValueError(
                        f"Unknown normalization method: {self.method}")

                # Fit and transform non-null values
                mask = ~numeric_series.isna()
                if mask.sum() > 0:  # Only if there are non-null values
                    values = numeric_series[mask].values.reshape(-1, 1)
                    scaled_values = scaler.fit_transform(values).flatten()
                    numeric_series.loc[mask] = scaled_values
                    self.scalers[col] = scaler

                result[col] = numeric_series

        return result


class DateNormalizer(BaseNormalizer):
    """Date and datetime normalization and standardization.

    Parameters
    ----------
    target_format : str, default '%Y-%m-%d'
        Target date format for output.
    parse_formats : List[str], optional
        List of date formats to try when parsing. If None, uses pandas inference.
    handle_timezone : bool, default True
        Whether to handle timezone conversion.
    target_timezone : str, default 'UTC'
        Target timezone for normalized dates.
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
            '%Y/%m/%d %H:%M:%S', '%m/%d/%Y %H:%M:%S'
        ]
        self.handle_timezone = handle_timezone
        self.target_timezone = target_timezone

    def _parse_date(self, date_str: str) -> Optional[pd.Timestamp]:
        """Parse a date string using multiple format attempts."""
        if pd.isna(date_str) or not str(date_str).strip():
            return None

        date_str = str(date_str).strip()

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

        logger.warning(f"Could not parse date: {date_str}")
        return None

    def apply(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Apply date normalization to specified columns."""
        result = df.copy()

        if columns is None:
            # Try to detect date columns automatically
            type_detector = TypeDetector()
            type_results = type_detector.detect_dataframe_types(df)
            columns = [
                col for col, result_obj in type_results.items()
                if result_obj.data_type in [DataType.DATE, DataType.DATETIME]
            ]

        for col in columns:
            if col in result.columns:
                logger.info(f"Normalizing date column: {col}")

                # Parse dates
                parsed_dates = result[col].apply(self._parse_date)

                # Handle timezone if requested
                if self.handle_timezone:
                    # Localize naive timestamps and convert to target timezone
                    parsed_dates = parsed_dates.apply(
                        lambda x: x.tz_localize('UTC').tz_convert(
                            self.target_timezone)
                        if x is not None and x.tz is None else x
                    )

                # Format to target format
                result[col] = parsed_dates.apply(
                    lambda x: x.strftime(
                        self.target_format) if x is not None else None
                )

        return result


class NormalizationPipeline:
    """Chain multiple normalizers with automatic type detection.

    Parameters
    ----------
    auto_detect_types : bool, default True
        Whether to automatically detect column types and apply appropriate normalizers.
    null_detector : NullDetector, optional
        Custom null detector. If None, uses default.
    """

    def __init__(
        self,
        auto_detect_types: bool = True,
        null_detector: Optional[NullDetector] = None,
    ):
        self.auto_detect_types = auto_detect_types
        self.null_detector = null_detector or NullDetector()
        self.type_detector = TypeDetector(self.null_detector)
        self.normalizers = []
        self.type_results = {}

    def add_normalizer(self, normalizer: BaseNormalizer) -> 'NormalizationPipeline':
        """Add a normalizer to the pipeline."""
        self.normalizers.append(normalizer)
        return self

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all normalizers in the pipeline."""
        result = df.copy()

        # Detect types if requested
        if self.auto_detect_types:
            logger.info("Detecting column types...")
            self.type_results = self.type_detector.detect_dataframe_types(df)

            # Log detected types
            for col, type_result in self.type_results.items():
                logger.info(
                    f"Column '{col}': {type_result.data_type.value} (confidence: {type_result.confidence:.2f})")

            # Apply type-specific normalization
            text_columns = [col for col, res in self.type_results.items()
                            if res.data_type == DataType.STRING]
            if text_columns:
                text_normalizer = TextNormalizer()
                result = text_normalizer.apply(result, text_columns)

            numeric_columns = [col for col, res in self.type_results.items()
                               if res.data_type in [DataType.INTEGER, DataType.FLOAT]]
            if numeric_columns:
                numeric_normalizer = NumericNormalizer()
                result = numeric_normalizer.apply(result, numeric_columns)

            date_columns = [col for col, res in self.type_results.items()
                            if res.data_type in [DataType.DATE, DataType.DATETIME]]
            if date_columns:
                date_normalizer = DateNormalizer()
                result = date_normalizer.apply(result, date_columns)

        # Apply custom normalizers
        for normalizer in self.normalizers:
            logger.info(f"Applying {normalizer.__class__.__name__}...")
            result = normalizer.apply(result)

        # Preserve metadata
        if hasattr(df, 'attrs'):
            result.attrs = df.attrs.copy()
            if 'provenance' not in result.attrs:
                result.attrs['provenance'] = []
            result.attrs['provenance'].append({
                'op': 'normalize',
                'params': {
                    'auto_detect_types': self.auto_detect_types,
                    'normalizer_count': len(self.normalizers)
                },
                'ts': datetime.now().isoformat(),
            })

        return result


class BaseValidator(ABC):
    """Abstract base class for validators."""

    @abstractmethod
    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class PydanticValidator(BaseValidator):
    """Validate rows against a Pydantic model.

    Parameters
    ----------
    model : pydantic.BaseModel
        A Pydantic model used to validate each row. Invalid rows are dropped.
    """

    def __init__(self, model: Any) -> None:
        from pydantic import BaseModel

        if not issubclass(model, BaseModel):
            raise TypeError("model must be a subclass of pydantic.BaseModel")
        self.model = model

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        from pydantic import ValidationError

        records = []
        for _, row in df.iterrows():
            try:
                validated = self.model(**row.to_dict())
                records.append(validated.model_dump())
            except ValidationError:
                continue
        return pd.DataFrame(records)


# Backward compatibility - preserve original convenience function
def normalize_dataframe_simple(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Convenience function for quick DataFrame normalization.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to normalize.
    **kwargs
        Arguments passed to NormalizationPipeline.

    Returns
    -------
    pd.DataFrame
        Normalized DataFrame.
    """
    pipeline = NormalizationPipeline(**kwargs)
    return pipeline.normalize(df)
