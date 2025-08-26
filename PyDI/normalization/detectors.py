"""
Data type and quality detection utilities for PyDI.

This module provides classes for automatically detecting data types, missing values,
outliers, and other data quality issues. It's inspired by Winter framework's 
type detection but adapted for pandas DataFrames.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataType(Enum):
    """Enumeration of supported data types for normalization."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    DATE = "date"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    EMAIL = "email"
    URL = "url"
    PHONE = "phone"
    COORDINATE = "coordinate"
    CURRENCY = "currency"
    PERCENTAGE = "percentage"
    UNKNOWN = "unknown"


class TypeDetectionResult:
    """Result of type detection for a column."""
    
    def __init__(
        self,
        data_type: DataType,
        confidence: float,
        patterns: Dict[str, int],
        sample_values: List[str],
        null_count: int,
        total_count: int,
    ):
        self.data_type = data_type
        self.confidence = confidence
        self.patterns = patterns
        self.sample_values = sample_values
        self.null_count = null_count
        self.total_count = total_count
        
    @property
    def null_percentage(self) -> float:
        """Percentage of null values in the column."""
        return (self.null_count / self.total_count) * 100 if self.total_count > 0 else 0.0
        
    def __repr__(self) -> str:
        return f"TypeDetectionResult({self.data_type.value}, {self.confidence:.2f}, {self.null_percentage:.1f}% null)"


class NullDetector:
    """
    Detector for various representations of missing/null values.
    
    Inspired by Winter's WebTablesStringNormalizer null detection.
    """
    
    # Extended null representations (case-insensitive) - includes Winter framework patterns
    DEFAULT_NULL_PATTERNS = [
        "",           # Empty string
        "null",       # Literal null
        "none",       # Python None equivalent
        "nil",        # Common in some systems
        "na",         # Not Available
        "n/a",        # Not Available
        "nan",        # Not a Number
        "missing",    # Explicit missing
        "unknown",    # Unknown value
        "undefined",  # Undefined value
        "void",       # Void value
        "-",          # Single dash
        "--",         # Double dash
        "___",        # Underscores
        "__",         # Double underscore (from Winter)
        "_",          # Single underscore (from Winter)
        "...",        # Ellipsis
        "?",          # Question mark
        "??",         # Double question (from Winter)
        "•",          # Bullet point (from Winter)
        ".",          # Single dot (from Winter)
        "- -",        # Spaced dashes (from Winter)
        "- - -",      # Spaced triple dash (from Winter)
        "(n/a)",      # Parenthesized n/a (from Winter)
        "n.a.",       # Dotted n.a.
        "not available", # Full phrase
        "not applicable", # Full phrase
        "no data",    # Full phrase
        "no info",    # No information
        "tbd",        # To be determined
        "tba",        # To be announced
        # Additional Winter framework patterns
        "n.d.",       # No data
        "nd",         # No data short
        "n\\a",       # Escaped n/a
        "---",        # Triple dash
        "n/d",        # No data
        "no value",   # No value
        "empty",      # Empty
        "blank",      # Blank
        "not specified", # Not specified
        "not set",    # Not set
        "not given",  # Not given
        "not provided", # Not provided
        "not entered", # Not entered
        "not found",  # Not found
        "not recorded", # Not recorded
        "not listed", # Not listed
        "not mentioned", # Not mentioned
        "not stated", # Not stated
        "not indicated", # Not indicated
        "no entry",   # No entry
        "no record",  # No record
        "no information", # No information (full)
        "information not available", # Full phrase
        "data not available", # Full phrase
        "value not available", # Full phrase
        "not available/applicable", # Combined
        "n/a - not available", # Extended n/a
        "n/a - not applicable", # Extended n/a
        # Language variations
        "nicht verfügbar", # German: not available
        "nicht angegeben", # German: not specified
        "unbekannt",  # German: unknown
        "pas disponible", # French: not available
        "non disponible", # French: not available
        "inconnu",    # French: unknown
        "no disponible", # Spanish: not available
        "desconocido", # Spanish: unknown
        # Technical variations
        "null value", # Technical null
        "null entry", # Technical null
        "nil value",  # Technical nil
        "void value", # Technical void
        "empty value", # Technical empty
        "missing value", # Technical missing
        "undefined value", # Technical undefined
        # Web/HTML related
        "&nbsp;",     # HTML non-breaking space
        "nbsp;",      # HTML entity
        "&ndash;",    # HTML en dash
        "&mdash;",    # HTML em dash
        # Database related
        "null record", # Database null
        "no record found", # Database query result
        "record not found", # Database query result
        "field empty", # Database field
        "column empty", # Database column
    ]
    
    def __init__(self, custom_patterns: Optional[List[str]] = None):
        """
        Initialize null detector.
        
        Parameters
        ----------
        custom_patterns : List[str], optional
            Additional null patterns to recognize.
        """
        self.null_patterns = set(pattern.lower() for pattern in self.DEFAULT_NULL_PATTERNS)
        if custom_patterns:
            self.null_patterns.update(pattern.lower() for pattern in custom_patterns)
    
    def is_null(self, value: Any) -> bool:
        """
        Check if a value should be considered null.
        
        Parameters
        ----------
        value : Any
            Value to check for nullness.
            
        Returns
        -------
        bool
            True if value should be considered null.
        """
        # Standard pandas null checks
        if pd.isna(value) or value is None:
            return True
            
        # String-based null pattern matching
        if isinstance(value, str):
            cleaned = value.strip().lower()
            return cleaned in self.null_patterns
            
        return False
    
    def detect_nulls(self, series: pd.Series) -> pd.Series:
        """
        Detect null values in a pandas Series.
        
        Parameters
        ----------
        series : pd.Series
            Series to analyze for null values.
            
        Returns
        -------
        pd.Series
            Boolean series indicating null values.
        """
        return series.apply(self.is_null)
    
    def null_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate a summary of null values across DataFrame columns.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to analyze.
            
        Returns
        -------
        pd.DataFrame
            Summary with columns: null_count, null_percentage, sample_nulls
        """
        summary_data = []
        
        for column in df.columns:
            null_mask = self.detect_nulls(df[column])
            null_count = null_mask.sum()
            null_percentage = (null_count / len(df)) * 100
            
            # Sample some null representations
            null_values = df.loc[null_mask, column].dropna().astype(str).unique()[:5]
            sample_nulls = list(null_values) if len(null_values) > 0 else []
            
            summary_data.append({
                'column': column,
                'null_count': null_count,
                'null_percentage': round(null_percentage, 2),
                'sample_nulls': sample_nulls
            })
        
        return pd.DataFrame(summary_data)


class TypeDetector:
    """
    Automatic data type detector based on pattern matching.
    
    Inspired by Winter's PatternbasedTypeDetector but optimized for pandas.
    """
    
    # Enhanced regex patterns for different data types (based on Winter framework patterns)
    PATTERNS = {
        # Email patterns - comprehensive email validation
        DataType.EMAIL: re.compile(
            r'^[a-zA-Z0-9]([a-zA-Z0-9._-]*[a-zA-Z0-9])?@[a-zA-Z0-9]([a-zA-Z0-9.-]*[a-zA-Z0-9])?\.[a-zA-Z]{2,}$'
        ),
        
        # URL patterns - enhanced to cover more URL formats
        DataType.URL: re.compile(
            r'^https?://[^\s/$.?#].[^\s]*$|^www\.[^\s/$.?#].[^\s]*$|^[a-zA-Z0-9][a-zA-Z0-9-]{0,61}[a-zA-Z0-9]\.[a-zA-Z]{2,}[/\w\-._~:/?#[\]@!$&\'()*+,;=]*$',
            re.IGNORECASE
        ),
        
        # Phone patterns - international formats
        DataType.PHONE: re.compile(
            r'^[\+]?[1-9]?[\d\s\-\(\)\.\+]{7,20}$|^\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}$'
        ),
        
        # Date patterns - various formats including Winter's comprehensive patterns
        DataType.DATE: re.compile(
            r'^\d{1,4}[-/\.]\d{1,2}[-/\.]\d{1,4}$|'  # Basic date formats
            r'^\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}$|'  # US/European formats
            r'^\d{4}\d{2}\d{2}$|'                     # YYYYMMDD
            r'^\d{2}\d{2}\d{4}$|'                     # MMDDYYYY or DDMMYYYY
            r'^[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}$|'  # Month DD, YYYY
            r'^\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}$|'    # DD Month YYYY
            r'^[A-Za-z]{3,9}\s+\d{4}$'               # Month YYYY
        ),
        
        # DateTime patterns - ISO and common formats
        DataType.DATETIME: re.compile(
            r'^\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})?$|'  # ISO format
            r'^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\s+\d{1,2}:\d{2}(:\d{2})?(\s*(AM|PM))?$',   # Common formats
            re.IGNORECASE
        ),
        
        # Coordinate patterns - decimal degrees and DMS formats
        DataType.COORDINATE: re.compile(
            r'^[-+]?([1-8]?\d(\.\d+)?|90(\.0+)?)[,\s]+[-+]?(180(\.0+)?|((1[0-7]\d)|([1-9]?\d))(\.\d+)?)$|'  # Decimal
            r'^\d{1,3}°\d{1,2}\'(\d{1,2}(\.\d+)?\")?[NSEW]\s*\d{1,3}°\d{1,2}\'(\d{1,2}(\.\d+)?\")?[NSEW]$'  # DMS
        ),
        
        # Currency patterns - international currency symbols and formats
        DataType.CURRENCY: re.compile(
            r'^[$€£¥₹₽₩₪]?\s*\d{1,3}(,\d{3})*(\.\d{2,4})?\s*(USD|EUR|GBP|JPY|CAD|AUD|INR|RUB|KRW|ILS)?$|'
            r'^\d+(\.\d{2,4})?\s*(USD|EUR|GBP|JPY|CAD|AUD|INR|RUB|KRW|ILS|\$|€|£|¥|₹|₽|₩|₪)$',
            re.IGNORECASE
        ),
        
        # Percentage patterns - various percentage formats
        DataType.PERCENTAGE: re.compile(
            r'^\d+(\.\d+)?\s*%$|^\d+(\.\d+)?\s*percent$|^\d+(\.\d+)?\s*pct$',
            re.IGNORECASE
        ),
        
        # Integer patterns - including thousands separators and signs
        DataType.INTEGER: re.compile(
            r'^[-+]?\d+$|^[-+]?\d{1,3}(,\d{3})+$'
        ),
        
        # Float patterns - comprehensive decimal formats including scientific notation
        DataType.FLOAT: re.compile(
            r'^[-+]?\d*\.\d+$|'                      # .123 or 123.456
            r'^[-+]?\d+\.\d*$|'                      # 123. or 123.456
            r'^[-+]?\d+[eE][-+]?\d+$|'               # Scientific notation
            r'^[-+]?\d*\.\d+[eE][-+]?\d+$|'          # Decimal scientific
            r'^[-+]?\d{1,3}(,\d{3})*\.\d+$'          # With thousands separators
        ),
        
        # Enhanced Boolean patterns - multi-language and extended formats
        DataType.BOOLEAN: re.compile(
            r'^(true|false|yes|no|y|n|t|f|1|0|on|off|enabled|disabled|active|inactive|'
            r'ja|nein|oui|non|sí|si|verdadero|falso|wahr|falsch|vrai|faux|'
            r'pass|fail|passed|failed|success|failure|valid|invalid|correct|incorrect|'
            r'ok|not ok|okay|good|bad)$',
            re.IGNORECASE
        ),
    }
    
    def __init__(self, null_detector: Optional[NullDetector] = None):
        """
        Initialize type detector.
        
        Parameters
        ----------
        null_detector : NullDetector, optional
            Null detector to use. If None, creates default detector.
        """
        self.null_detector = null_detector or NullDetector()
    
    def _clean_value(self, value: Any) -> str:
        """Clean a value for pattern matching."""
        if pd.isna(value) or self.null_detector.is_null(value):
            return ""
        return str(value).strip()
    
    def _match_patterns(self, values: List[str]) -> Dict[DataType, int]:
        """
        Match values against type patterns.
        
        Parameters
        ----------
        values : List[str]
            List of cleaned string values to match.
            
        Returns
        -------
        Dict[DataType, int]
            Count of matches for each data type.
        """
        matches = {data_type: 0 for data_type in DataType}
        
        for value in values:
            if not value:  # Skip empty values
                continue
                
            for data_type, pattern in self.PATTERNS.items():
                if pattern.match(value):
                    matches[data_type] += 1
        
        return matches
    
    def detect_column_type(
        self, 
        series: pd.Series, 
        sample_size: int = 1000,
        confidence_threshold: float = 0.7
    ) -> TypeDetectionResult:
        """
        Detect the data type of a pandas Series column.
        
        Parameters
        ----------
        series : pd.Series
            Series to analyze.
        sample_size : int, default 1000
            Maximum number of values to sample for type detection.
        confidence_threshold : float, default 0.7
            Minimum confidence required for type detection.
            
        Returns
        -------
        TypeDetectionResult
            Result containing detected type and confidence metrics.
        """
        # Sample values for analysis
        non_null_values = series.dropna()
        if len(non_null_values) == 0:
            return TypeDetectionResult(
                data_type=DataType.UNKNOWN,
                confidence=0.0,
                patterns={},
                sample_values=[],
                null_count=len(series),
                total_count=len(series)
            )
        
        # Sample for efficiency
        if len(non_null_values) > sample_size:
            sampled_values = non_null_values.sample(n=sample_size, random_state=42)
        else:
            sampled_values = non_null_values
        
        # Clean values and remove nulls
        cleaned_values = []
        for value in sampled_values:
            if not self.null_detector.is_null(value):
                cleaned_values.append(self._clean_value(value))
        
        # Remove empty strings
        cleaned_values = [v for v in cleaned_values if v]
        
        if not cleaned_values:
            return TypeDetectionResult(
                data_type=DataType.STRING,
                confidence=0.0,
                patterns={},
                sample_values=[],
                null_count=len(series) - len(non_null_values),
                total_count=len(series)
            )
        
        # Match against patterns
        pattern_matches = self._match_patterns(cleaned_values)
        
        # Find best match
        best_type = DataType.STRING  # Default fallback
        best_score = 0
        
        for data_type, match_count in pattern_matches.items():
            if match_count > best_score:
                best_score = match_count
                best_type = data_type
        
        # Calculate confidence
        confidence = best_score / len(cleaned_values) if cleaned_values else 0.0
        
        # If confidence is too low, fallback to string
        if confidence < confidence_threshold and best_type != DataType.STRING:
            best_type = DataType.STRING
            confidence = 1.0  # We're confident it's a string
        
        return TypeDetectionResult(
            data_type=best_type,
            confidence=confidence,
            patterns={dt.value: count for dt, count in pattern_matches.items() if count > 0},
            sample_values=cleaned_values[:10],  # First 10 samples
            null_count=len(series) - len(non_null_values),
            total_count=len(series)
        )
    
    def detect_dataframe_types(
        self, 
        df: pd.DataFrame,
        sample_size: int = 1000,
        confidence_threshold: float = 0.7
    ) -> Dict[str, TypeDetectionResult]:
        """
        Detect data types for all columns in a DataFrame.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to analyze.
        sample_size : int, default 1000
            Maximum number of values to sample per column.
        confidence_threshold : float, default 0.7
            Minimum confidence required for type detection.
            
        Returns
        -------
        Dict[str, TypeDetectionResult]
            Dictionary mapping column names to type detection results.
        """
        results = {}
        
        for column in df.columns:
            logger.info(f"Detecting type for column: {column}")
            result = self.detect_column_type(
                df[column], 
                sample_size=sample_size,
                confidence_threshold=confidence_threshold
            )
            results[column] = result
            logger.info(f"Column '{column}' detected as {result.data_type.value} (confidence: {result.confidence:.2f})")
        
        return results


class OutlierDetector:
    """Statistical outlier detector for numeric data."""
    
    def __init__(self, method: str = "iqr", threshold: float = 1.5):
        """
        Initialize outlier detector.
        
        Parameters
        ----------
        method : str, default "iqr"
            Method for outlier detection: "iqr", "zscore", or "modified_zscore"
        threshold : float, default 1.5
            Threshold for outlier detection (method-specific)
        """
        self.method = method
        self.threshold = threshold
    
    def detect_outliers(self, series: pd.Series) -> pd.Series:
        """
        Detect outliers in a numeric series.
        
        Parameters
        ----------
        series : pd.Series
            Numeric series to analyze.
            
        Returns
        -------
        pd.Series
            Boolean series indicating outliers.
        """
        # Convert to numeric, coercing errors to NaN
        numeric_series = pd.to_numeric(series, errors='coerce')
        
        if self.method == "iqr":
            return self._iqr_outliers(numeric_series)
        elif self.method == "zscore":
            return self._zscore_outliers(numeric_series)
        elif self.method == "modified_zscore":
            return self._modified_zscore_outliers(numeric_series)
        else:
            raise ValueError(f"Unknown outlier detection method: {self.method}")
    
    def _iqr_outliers(self, series: pd.Series) -> pd.Series:
        """Detect outliers using Interquartile Range method."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - self.threshold * IQR
        upper_bound = Q3 + self.threshold * IQR
        return (series < lower_bound) | (series > upper_bound)
    
    def _zscore_outliers(self, series: pd.Series) -> pd.Series:
        """Detect outliers using Z-score method."""
        z_scores = np.abs((series - series.mean()) / series.std())
        return z_scores > self.threshold
    
    def _modified_zscore_outliers(self, series: pd.Series) -> pd.Series:
        """Detect outliers using Modified Z-score method."""
        median = series.median()
        mad = np.median(np.abs(series - median))
        modified_z_scores = 0.6745 * (series - median) / mad
        return np.abs(modified_z_scores) > self.threshold


class DuplicateDetector:
    """Detector for duplicate and near-duplicate values."""
    
    def __init__(self, case_sensitive: bool = False):
        """
        Initialize duplicate detector.
        
        Parameters
        ----------
        case_sensitive : bool, default False
            Whether to consider case when detecting duplicates.
        """
        self.case_sensitive = case_sensitive
    
    def detect_exact_duplicates(self, series: pd.Series) -> pd.Series:
        """
        Detect exact duplicate values.
        
        Parameters
        ----------
        series : pd.Series
            Series to analyze.
            
        Returns
        -------
        pd.Series
            Boolean series indicating duplicate values.
        """
        if not self.case_sensitive and series.dtype == 'object':
            return series.str.lower().duplicated(keep=False)
        return series.duplicated(keep=False)
    
    def duplicate_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate a summary of duplicates across DataFrame columns.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to analyze.
            
        Returns
        -------
        pd.DataFrame
            Summary with duplicate statistics per column.
        """
        summary_data = []
        
        for column in df.columns:
            duplicates = self.detect_exact_duplicates(df[column])
            duplicate_count = duplicates.sum()
            unique_count = df[column].nunique()
            total_count = len(df)
            duplicate_percentage = (duplicate_count / total_count) * 100 if total_count > 0 else 0
            
            summary_data.append({
                'column': column,
                'total_values': total_count,
                'unique_values': unique_count,
                'duplicate_values': duplicate_count,
                'duplicate_percentage': round(duplicate_percentage, 2),
            })
        
        return pd.DataFrame(summary_data)