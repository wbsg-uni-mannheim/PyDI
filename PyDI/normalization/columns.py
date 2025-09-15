"""
Column-level analysis and type detection utilities for PyDI.

This module provides comprehensive column analysis including type detection,
pattern analysis, and column-specific normalization. It includes both basic
and advanced type detection capabilities with confidence scoring.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from .detectors import DataType, NullDetector, TypeDetectionResult, OutlierDetector
from .units import UnitCategory, UnitDetector, UnitRegistry, parse_unit_from_header
from .values import AdvancedValueNormalizer

logger = logging.getLogger(__name__)


class DataTypeExtended(Enum):
    """Extended data types for comprehensive column analysis."""
    NUMERIC = "numeric"
    STRING = "string"
    DATE = "date"
    DATETIME = "datetime"
    BOOLEAN = "bool"
    COORDINATE = "coordinate"
    LINK = "link"
    EMAIL = "email"
    PHONE = "phone"
    CURRENCY = "currency"
    PERCENTAGE = "percentage"
    LIST = "list"
    UNIT = "unit"
    UNKNOWN = "unknown"


class ValueDetectionType:
    """
    Comprehensive type detection result for columns.

    Contains detailed information about detected type, confidence,
    unit information, and analysis metadata.
    """

    def __init__(
        self,
        data_type: DataTypeExtended,
        confidence: float = 0.0,
        samples_analyzed: int = 0,
        type_counts: Optional[Dict[DataTypeExtended, int]] = None,
        unit_category: Optional[UnitCategory] = None,
        specific_unit: Optional[str] = None,
        format_pattern: Optional[str] = None,
        null_count: int = 0,
        total_count: int = 0,
        sample_values: Optional[List[str]] = None
    ):
        self.data_type = data_type
        self.confidence = confidence
        self.samples_analyzed = samples_analyzed
        self.type_counts = type_counts or {}
        self.unit_category = unit_category
        self.specific_unit = specific_unit
        self.format_pattern = format_pattern
        self.null_count = null_count
        self.total_count = total_count
        self.sample_values = sample_values or []

    @property
    def null_percentage(self) -> float:
        """Calculate null percentage."""
        return (self.null_count / self.total_count) * 100 if self.total_count > 0 else 0.0

    def __repr__(self) -> str:
        parts = [f"{self.data_type.value}"]
        if self.unit_category:
            parts.append(f"unit={self.unit_category.value}")
        if self.specific_unit:
            parts.append(f"specific={self.specific_unit}")
        parts.append(f"confidence={self.confidence:.3f}")
        return f"ValueDetectionType({', '.join(parts)})"




class AdvancedTypeDetector:
    """
    Sophisticated type detector with majority voting and comprehensive analysis.

    Provides Winter-equivalent type detection with confidence scoring,
    unit detection, and pattern analysis.
    """

    def __init__(
        self,
        unit_registry: Optional[UnitRegistry] = None,
        null_detector: Optional[NullDetector] = None
    ):
        self.unit_registry = unit_registry or UnitRegistry(comprehensive=True)
        self.unit_detector = UnitDetector(self.unit_registry)
        self.null_detector = null_detector or NullDetector()
        self._initialize_comprehensive_patterns()

    def _initialize_comprehensive_patterns(self):
        """Initialize comprehensive detection patterns."""
        # Numeric patterns with units and quantities
        self.numeric_patterns = [
            # Scientific notation
            re.compile(r'[-+]?\d*\.?\d+[eE][-+]?\d+', re.IGNORECASE),
            # Numbers with units
            re.compile(
                r'[-+]?\d+(?:\.\d+)?\s*[a-zA-Z°$€£¥₹₽₩₪%]+', re.IGNORECASE),
            # Numbers with quantity modifiers
            re.compile(
                r'[-+]?\d+(?:\.\d+)?\s*(thousand|million|billion|k|m|b)\b', re.IGNORECASE),
            # Decimal with thousands separators
            re.compile(r'[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?'),
            # Simple decimals and integers
            re.compile(r'[-+]?\d*\.\d+'), re.compile(r'[-+]?\d+'),
        ]

        # List pattern (Winter format: {value1|value2|value3})
        self.list_pattern = re.compile(r'^\{([^}]+)\}$')

        # Enhanced email pattern
        self.email_pattern = re.compile(
            r'^[a-zA-Z0-9]([a-zA-Z0-9._-]*[a-zA-Z0-9])?@[a-zA-Z0-9]([a-zA-Z0-9.-]*[a-zA-Z0-9])?\.[a-zA-Z]{2,}$'
        )

        # Enhanced URL pattern
        self.url_pattern = re.compile(
            r'^https?://[^\s/$.?#].[^\s]*$|^www\.[^\s/$.?#].[^\s]*$|^[a-zA-Z0-9][a-zA-Z0-9-]{0,61}[a-zA-Z0-9]\.[a-zA-Z]{2,}[/\w\-._~:/?#[\]@!$&\'()*+,;=]*$',
            re.IGNORECASE
        )

        # Phone pattern
        self.phone_pattern = re.compile(
            r'^[\+]?[1-9]?[\d\s\-\(\)\.\+]{7,20}$|^\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}$'
        )

        # Comprehensive date patterns
        self.date_patterns = [
            re.compile(r'^\d{4}-\d{1,2}-\d{1,2}$'),  # ISO
            re.compile(r'^\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}$'),  # US/European
            re.compile(r'^\d{4}\d{2}\d{2}$'),  # YYYYMMDD
            # Month DD, YYYY
            re.compile(r'^[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}$'),
            re.compile(r'^\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}$'),  # DD Month YYYY
        ]

        # DateTime patterns
        self.datetime_patterns = [
            re.compile(r'^\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}'),  # ISO
            # Common
            re.compile(
                r'^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\s+\d{1,2}:\d{2}(:\d{2})?'),
        ]

        # Boolean pattern (multi-language)
        self.boolean_pattern = re.compile(
            r'^(true|false|yes|no|y|n|t|f|1|0|ja|nein|oui|non|sí|verdadero|falso)$',
            re.IGNORECASE
        )

        # Coordinate patterns
        self.coordinate_pattern = re.compile(
            r'^[-+]?([1-8]?\d(?:\.\d+)?|90(?:\.0+)?)[,\s]+[-+]?(180(?:\.0+)?|(?:1[0-7]\d|[1-9]?\d)(?:\.\d+)?)$'
        )

        # Currency patterns
        self.currency_pattern = re.compile(
            r'^[$€£¥₹₽₩₪]?\s*\d{1,3}(,\d{3})*(\.\d{2,4})?\s*(USD|EUR|GBP|JPY|CAD|AUD)?$',
            re.IGNORECASE
        )

        # Percentage patterns
        self.percentage_pattern = re.compile(
            r'^\d+(\.\d+)?\s*%$|^\d+(\.\d+)?\s*percent$',
            re.IGNORECASE
        )

    def detect_type_for_column(
        self,
        values: List[str],
        column_name: str,
        confidence_threshold: float = 0.7
    ) -> ValueDetectionType:
        """
        Detect type for entire column using majority voting.

        Parameters
        ----------
        values : List[str]
            All values in the column.
        column_name : str
            Column identifier.
        confidence_threshold : float, default 0.7
            Minimum confidence for type assignment.

        Returns
        -------
        ValueDetectionType
            Comprehensive detection result.
        """
        if not values:
            return ValueDetectionType(DataTypeExtended.UNKNOWN)

        # Clean and filter values
        clean_values = []
        null_count = 0

        for v in values:
            if v is None or pd.isna(v):
                null_count += 1
            elif self.null_detector.is_null(v):
                null_count += 1
            else:
                clean_str = str(v).strip()
                if clean_str:
                    clean_values.append(clean_str)

        if not clean_values:
            return ValueDetectionType(
                data_type=DataTypeExtended.UNKNOWN,
                null_count=null_count,
                total_count=len(values)
            )

        # Header-derived unit hint (prefer strong signal like "(km/h)")
        header_unit = parse_unit_from_header(column_name, self.unit_registry)

        # Count type occurrences for each value
        type_counts: Dict[DataTypeExtended, int] = {}
        unit_info: Dict[str, int] = {}  # Track units found
        format_patterns: Dict[str, int] = {}  # Track formats

        for value in clean_values:
            detected_type, unit, format_pattern = self._detect_single_value_comprehensive(
                value)

            type_counts[detected_type] = type_counts.get(detected_type, 0) + 1

            if unit:
                unit_info[unit] = unit_info.get(unit, 0) + 1

            if format_pattern:
                format_patterns[format_pattern] = format_patterns.get(
                    format_pattern, 0) + 1

        # Majority voting with deterministic ordering
        sorted_types = sorted(type_counts.items(),
                              key=lambda x: (-x[1], x[0].value))
        winner_type, winner_count = sorted_types[0]

        # Calculate confidence
        confidence = winner_count / len(clean_values)

        # Determine unit category and specific unit
        unit_category = None
        specific_unit = None

        # Prefer header unit if found
        if header_unit is not None:
            unit_obj = header_unit
            unit_category = unit_obj.category
            specific_unit = unit_obj.symbol
        elif unit_info:
            # Guard against spurious units in mostly-text columns (e.g., film titles
            # like "127 Hours"). Only attach a unit when the winner type is numeric
            # and the unit appears in at least half of the non-null values.
            most_common_unit = max(
                unit_info.keys(), key=lambda k: unit_info[k])
            unit_support = unit_info[most_common_unit]
            unit_ratio = unit_support / max(len(clean_values), 1)
            if winner_type == DataTypeExtended.NUMERIC and unit_ratio >= 0.5:
                unit_obj = self.unit_registry.get_unit(most_common_unit)
                if unit_obj:
                    unit_category = unit_obj.category
                    specific_unit = unit_obj.symbol

        # Determine most common format
        format_pattern = None
        if format_patterns:
            format_pattern = max(format_patterns.keys(),
                                 key=lambda k: format_patterns[k])

        return ValueDetectionType(
            data_type=winner_type,
            confidence=confidence,
            samples_analyzed=len(clean_values),
            type_counts=type_counts,
            unit_category=unit_category,
            specific_unit=specific_unit,
            format_pattern=format_pattern,
            null_count=null_count,
            total_count=len(values),
            sample_values=clean_values[:10]
        )

    def _detect_single_value_comprehensive(
        self,
        value: str
    ) -> Tuple[DataTypeExtended, Optional[str], Optional[str]]:
        """Detect type for single value with unit and format information."""

        # Check for list format first
        if self.list_pattern.match(value):
            return DataTypeExtended.LIST, None, "list"

        # Email check
        if self.email_pattern.match(value):
            return DataTypeExtended.EMAIL, None, "email"

        # URL check
        if self.url_pattern.match(value):
            return DataTypeExtended.LINK, None, "url"

        # Phone check
        if self.phone_pattern.match(value):
            return DataTypeExtended.PHONE, None, "phone"

        # Boolean check
        if self.boolean_pattern.match(value):
            return DataTypeExtended.BOOLEAN, None, "boolean"

        # Coordinate check
        if self.coordinate_pattern.match(value):
            return DataTypeExtended.COORDINATE, None, "coordinate"

        # Currency check
        if self.currency_pattern.match(value):
            return DataTypeExtended.CURRENCY, None, "currency"

        # Percentage check
        if self.percentage_pattern.match(value):
            return DataTypeExtended.PERCENTAGE, None, "percentage"

        # DateTime checks
        for pattern in self.datetime_patterns:
            if pattern.match(value):
                return DataTypeExtended.DATETIME, None, "datetime"

        # Date checks
        for pattern in self.date_patterns:
            if pattern.match(value):
                return DataTypeExtended.DATE, None, "date"

        # Numeric checks with unit detection
        for pattern in self.numeric_patterns:
            if pattern.match(value):
                # Try to extract unit
                unit = self._extract_unit(value)
                if unit:
                    return DataTypeExtended.NUMERIC, unit, "numeric_with_unit"
                else:
                    return DataTypeExtended.NUMERIC, None, "numeric"

        # Default to string
        return DataTypeExtended.STRING, None, "string"

    def _extract_unit(self, value: str) -> Optional[str]:
        """Extract unit from numeric value."""
        # Look for units at the end of the value
        unit_match = re.search(r'([a-zA-Z°$€£¥₹₽₩₪%]+)$', value.strip())
        if unit_match:
            potential_unit = unit_match.group(1).lower()
            if self.unit_registry.get_unit(potential_unit):
                return potential_unit
        return None

    def detect_dataframe_types(
        self,
        df: pd.DataFrame,
        confidence_threshold: float = 0.7
    ) -> Dict[str, ValueDetectionType]:
        """Detect types for all columns in DataFrame."""
        results = {}

        for column in df.columns:
            logger.info(f"Detecting type for column: {column}")

            # Get all values as strings
            column_values = df[column].astype(str).tolist()

            # Use advanced type detection
            detection_result = self.detect_type_for_column(
                column_values,
                column,
                confidence_threshold
            )
            results[column] = detection_result

            logger.info(f"Column '{column}' detected as {detection_result}")

        return results


class ColumnTypeInference:
    """
    Enhanced column type inference with confidence scoring and metadata.

    Provides comprehensive column analysis including type detection,
    quality assessment, and normalization recommendations.
    """

    def __init__(
        self,
        type_detector: Optional[AdvancedTypeDetector] = None,
        confidence_threshold: float = 0.7,
        sample_size: int = 1000
    ):
        self.type_detector = type_detector or AdvancedTypeDetector()
        self.confidence_threshold = confidence_threshold
        self.sample_size = sample_size

    def infer_column_types(self, df: pd.DataFrame) -> Dict[str, ValueDetectionType]:
        """Infer types for all columns with comprehensive analysis."""
        return self.type_detector.detect_dataframe_types(df, self.confidence_threshold)

    def get_type_summary(self, type_results: Dict[str, ValueDetectionType]) -> pd.DataFrame:
        """Create summary DataFrame of type detection results."""
        summary_data = []

        for column, result in type_results.items():
            summary_data.append({
                'column': column,
                'detected_type': result.data_type.value,
                'confidence': round(result.confidence, 3),
                'null_percentage': round(result.null_percentage, 2),
                'samples_analyzed': result.samples_analyzed,
                'unit_category': result.unit_category.value if result.unit_category else None,
                'specific_unit': result.specific_unit,
                'format_pattern': result.format_pattern,
                'sample_values': ', '.join(result.sample_values[:3])
            })

        return pd.DataFrame(summary_data)

    def get_normalization_recommendations(
        self,
        type_results: Dict[str, ValueDetectionType]
    ) -> Dict[str, List[str]]:
        """Generate normalization recommendations based on type detection."""
        recommendations = {}

        for column, result in type_results.items():
            column_recommendations = []

            if result.confidence < self.confidence_threshold:
                column_recommendations.append(
                    f"Low confidence ({result.confidence:.2f}) - verify data type")

            if result.null_percentage > 20:
                column_recommendations.append(
                    f"High null rate ({result.null_percentage:.1f}%) - consider imputation")

            if result.data_type == DataTypeExtended.NUMERIC and result.unit_category:
                column_recommendations.append(
                    f"Contains units ({result.specific_unit}) - consider unit normalization")

            if result.data_type == DataTypeExtended.STRING and result.samples_analyzed > 10:
                column_recommendations.append(
                    "Text data - consider text normalization")

            if result.data_type == DataTypeExtended.DATE:
                column_recommendations.append(
                    "Date data - consider date standardization")

            recommendations[column] = column_recommendations

        return recommendations


# Convenience functions for column analysis
def detect_column_types(df: pd.DataFrame) -> Dict[str, str]:
    """Detect types for all columns using AdvancedTypeDetector.

    Returns mapping of column name to detected type string.
    """
    detector = AdvancedTypeDetector()
    results = detector.detect_dataframe_types(df)
    return {col: result.data_type.value for col, result in results.items()}


def detect_dataframe_types(
    df: pd.DataFrame,
    confidence_threshold: float = 0.7,
) -> Dict[str, ValueDetectionType]:
    """Convenience wrapper to detect types for all columns using AdvancedTypeDetector.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to analyze.
    confidence_threshold : float, default 0.7
        Minimum confidence for type assignment.

    Returns
    -------
    Dict[str, ValueDetectionType]
        Mapping of column name to detection result.
    """
    detector = AdvancedTypeDetector()
    return detector.detect_dataframe_types(df, confidence_threshold)


def infer_column_types(
    df: pd.DataFrame,
    confidence_threshold: float = 0.7,
    sample_size: int = 1000,
) -> Dict[str, ValueDetectionType]:
    """Convenience wrapper for ColumnTypeInference.infer_column_types."""
    inferencer = ColumnTypeInference(
        confidence_threshold=confidence_threshold, sample_size=sample_size
    )
    return inferencer.infer_column_types(df)


def analyze_column_quality(series: pd.Series) -> Dict[str, Any]:
    """
    Analyze quality metrics for a single column.

    Parameters
    ----------
    series : pd.Series
        Series to analyze.

    Returns
    -------
    Dict[str, Any]
        Quality metrics including null rate, uniqueness, etc.
    """
    null_detector = NullDetector()
    outlier_detector = OutlierDetector()

    total_count = len(series)
    null_count = null_detector.detect_nulls(series).sum()
    unique_count = series.nunique()

    quality_metrics = {
        'total_values': total_count,
        'null_count': null_count,
        'null_percentage': (null_count / total_count) * 100 if total_count > 0 else 0,
        'unique_count': unique_count,
        'uniqueness_ratio': unique_count / total_count if total_count > 0 else 0,
        'completeness': (total_count - null_count) / total_count if total_count > 0 else 0
    }

    # Add numeric-specific metrics if applicable
    try:
        numeric_series = pd.to_numeric(series, errors='coerce')
        if not numeric_series.isna().all():
            outliers = outlier_detector.detect_outliers(numeric_series)
            quality_metrics.update({
                'outlier_count': outliers.sum(),
                'outlier_percentage': (outliers.sum() / total_count) * 100 if total_count > 0 else 0,
                'mean': numeric_series.mean(),
                'std': numeric_series.std(),
                'min': numeric_series.min(),
                'max': numeric_series.max()
            })
    except:
        pass

    return quality_metrics


def get_column_recommendations(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Get normalization recommendations for all columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to analyze.

    Returns
    -------
    Dict[str, List[str]]
        Recommendations for each column.
    """
    inferencer = ColumnTypeInference()
    type_results = inferencer.infer_column_types(df)
    return inferencer.get_normalization_recommendations(type_results)
