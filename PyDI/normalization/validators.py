"""
Data validation utilities for PyDI.

This module provides comprehensive data validation tools including format validators,
range checkers, completeness assessments, and quality metrics. It supports both
schema-based and rule-based validation approaches.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ValidationResult:
    """Result of data validation containing errors and quality metrics."""
    
    def __init__(self):
        self.errors: List[Dict[str, Any]] = []
        self.warnings: List[Dict[str, Any]] = []
        self.quality_metrics: Dict[str, Any] = {}
        self.valid_rows: int = 0
        self.total_rows: int = 0
    
    def add_error(self, column: str, row_index: int, message: str, value: Any = None):
        """Add a validation error."""
        self.errors.append({
            'column': column,
            'row': row_index,
            'message': message,
            'value': value
        })
    
    def add_warning(self, column: str, row_index: int, message: str, value: Any = None):
        """Add a validation warning."""
        self.warnings.append({
            'column': column,
            'row': row_index,
            'message': message,
            'value': value
        })
    
    def set_quality_metric(self, name: str, value: Any):
        """Set a quality metric."""
        self.quality_metrics[name] = value
    
    @property
    def is_valid(self) -> bool:
        """Check if validation passed (no errors)."""
        return len(self.errors) == 0
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate as percentage."""
        if self.total_rows == 0:
            return 0.0
        return (len(self.errors) / self.total_rows) * 100
    
    @property
    def validity_rate(self) -> float:
        """Calculate validity rate as percentage."""
        if self.total_rows == 0:
            return 100.0
        return (self.valid_rows / self.total_rows) * 100
    
    def summary(self) -> Dict[str, Any]:
        """Generate a summary of validation results."""
        return {
            'total_rows': self.total_rows,
            'valid_rows': self.valid_rows,
            'error_count': len(self.errors),
            'warning_count': len(self.warnings),
            'error_rate': round(self.error_rate, 2),
            'validity_rate': round(self.validity_rate, 2),
            'quality_metrics': self.quality_metrics
        }
    
    def errors_df(self) -> pd.DataFrame:
        """Convert errors to DataFrame for analysis."""
        return pd.DataFrame(self.errors)
    
    def warnings_df(self) -> pd.DataFrame:
        """Convert warnings to DataFrame for analysis."""
        return pd.DataFrame(self.warnings)


class BaseValidator(ABC):
    """Abstract base class for validators."""
    
    @abstractmethod
    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """Validate a DataFrame and return results."""
        pass


class EmailValidator(BaseValidator):
    """Validator for email addresses."""
    
    def __init__(self, strict: bool = True):
        """
        Initialize email validator.
        
        Parameters
        ----------
        strict : bool, default True
            Whether to use strict email validation.
        """
        self.strict = strict
        # Basic email regex pattern
        self.email_pattern = re.compile(
            r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        )
        # More lenient pattern
        self.lenient_pattern = re.compile(r'.+@.+\..+')
    
    def _is_valid_email(self, email: str) -> bool:
        """Check if email is valid."""
        if pd.isna(email) or not isinstance(email, str):
            return False
        
        email = email.strip()
        if not email:
            return False
        
        pattern = self.email_pattern if self.strict else self.lenient_pattern
        return bool(pattern.match(email))
    
    def validate(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> ValidationResult:
        """Validate email columns in DataFrame."""
        result = ValidationResult()
        result.total_rows = len(df)
        
        if columns is None:
            # Try to detect email columns by name or content
            columns = [col for col in df.columns if 'email' in col.lower()]
        
        valid_rows = set(range(len(df)))
        
        for col in columns:
            if col not in df.columns:
                continue
                
            logger.info(f"Validating email column: {col}")
            
            for idx, value in df[col].items():
                if not self._is_valid_email(value):
                    result.add_error(col, idx, f"Invalid email format", value)
                    valid_rows.discard(idx)
        
        result.valid_rows = len(valid_rows)
        return result


class RangeValidator(BaseValidator):
    """Validator for numeric and date ranges."""
    
    def __init__(
        self, 
        ranges: Dict[str, Dict[str, Any]], 
        allow_null: bool = True
    ):
        """
        Initialize range validator.
        
        Parameters
        ----------
        ranges : Dict[str, Dict[str, Any]]
            Column ranges in format: {'column': {'min': value, 'max': value}}
        allow_null : bool, default True
            Whether to allow null values.
        """
        self.ranges = ranges
        self.allow_null = allow_null
    
    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """Validate ranges in DataFrame."""
        result = ValidationResult()
        result.total_rows = len(df)
        
        valid_rows = set(range(len(df)))
        
        for col, range_spec in self.ranges.items():
            if col not in df.columns:
                continue
                
            logger.info(f"Validating range for column: {col}")
            
            min_val = range_spec.get('min')
            max_val = range_spec.get('max')
            
            for idx, value in df[col].items():
                if pd.isna(value):
                    if not self.allow_null:
                        result.add_error(col, idx, "Null value not allowed", value)
                        valid_rows.discard(idx)
                    continue
                
                # Convert to numeric if possible
                try:
                    num_value = pd.to_numeric(value)
                except (ValueError, TypeError):
                    try:
                        # Try datetime conversion
                        num_value = pd.to_datetime(value)
                    except:
                        result.add_error(col, idx, "Cannot validate non-numeric value", value)
                        valid_rows.discard(idx)
                        continue
                
                # Check range
                if min_val is not None and num_value < min_val:
                    result.add_error(col, idx, f"Value below minimum ({min_val})", value)
                    valid_rows.discard(idx)
                
                if max_val is not None and num_value > max_val:
                    result.add_error(col, idx, f"Value above maximum ({max_val})", value)
                    valid_rows.discard(idx)
        
        result.valid_rows = len(valid_rows)
        return result


class PatternValidator(BaseValidator):
    """Validator for regex patterns."""
    
    def __init__(self, patterns: Dict[str, str]):
        """
        Initialize pattern validator.
        
        Parameters
        ----------
        patterns : Dict[str, str]
            Column patterns in format: {'column': 'regex_pattern'}
        """
        self.patterns = {col: re.compile(pattern) for col, pattern in patterns.items()}
    
    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """Validate patterns in DataFrame."""
        result = ValidationResult()
        result.total_rows = len(df)
        
        valid_rows = set(range(len(df)))
        
        for col, pattern in self.patterns.items():
            if col not in df.columns:
                continue
                
            logger.info(f"Validating pattern for column: {col}")
            
            for idx, value in df[col].items():
                if pd.isna(value):
                    continue
                
                value_str = str(value).strip()
                if not pattern.match(value_str):
                    result.add_error(col, idx, f"Value doesn't match required pattern", value)
                    valid_rows.discard(idx)
        
        result.valid_rows = len(valid_rows)
        return result


class CompletenessValidator(BaseValidator):
    """Validator for data completeness requirements."""
    
    def __init__(
        self, 
        required_columns: Optional[List[str]] = None,
        min_completeness: float = 0.0,
        column_completeness: Optional[Dict[str, float]] = None
    ):
        """
        Initialize completeness validator.
        
        Parameters
        ----------
        required_columns : List[str], optional
            Columns that cannot have any null values.
        min_completeness : float, default 0.0
            Minimum completeness rate (0-1) for all columns.
        column_completeness : Dict[str, float], optional
            Specific completeness requirements per column.
        """
        self.required_columns = required_columns or []
        self.min_completeness = min_completeness
        self.column_completeness = column_completeness or {}
    
    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """Validate completeness in DataFrame."""
        result = ValidationResult()
        result.total_rows = len(df)
        
        valid_rows = set(range(len(df)))
        
        # Check required columns (no nulls allowed)
        for col in self.required_columns:
            if col not in df.columns:
                result.add_warning('dataset', -1, f"Required column '{col}' not found")
                continue
            
            null_mask = df[col].isna()
            null_indices = df[null_mask].index.tolist()
            
            for idx in null_indices:
                result.add_error(col, idx, "Required field cannot be null", None)
                valid_rows.discard(idx)
        
        # Check minimum completeness for all columns
        for col in df.columns:
            null_count = df[col].isna().sum()
            completeness = 1 - (null_count / len(df))
            
            # Store completeness metric
            result.set_quality_metric(f"{col}_completeness", round(completeness, 3))
            
            # Check against minimum requirement
            min_required = self.column_completeness.get(col, self.min_completeness)
            
            if completeness < min_required:
                message = f"Completeness {completeness:.1%} below required {min_required:.1%}"
                result.add_warning(col, -1, message)
        
        result.valid_rows = len(valid_rows)
        return result


class UniqueValidator(BaseValidator):
    """Validator for uniqueness constraints."""
    
    def __init__(self, unique_columns: List[str]):
        """
        Initialize uniqueness validator.
        
        Parameters
        ----------
        unique_columns : List[str]
            Columns that must contain unique values.
        """
        self.unique_columns = unique_columns
    
    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """Validate uniqueness constraints in DataFrame."""
        result = ValidationResult()
        result.total_rows = len(df)
        
        valid_rows = set(range(len(df)))
        
        for col in self.unique_columns:
            if col not in df.columns:
                result.add_warning('dataset', -1, f"Unique column '{col}' not found")
                continue
            
            logger.info(f"Validating uniqueness for column: {col}")
            
            # Find duplicates
            duplicates = df[col].duplicated(keep=False)
            duplicate_indices = df[duplicates].index.tolist()
            
            for idx in duplicate_indices:
                value = df.loc[idx, col]
                result.add_error(col, idx, f"Duplicate value found", value)
                valid_rows.discard(idx)
            
            # Store uniqueness metric
            unique_count = df[col].nunique()
            total_count = len(df[col].dropna())
            uniqueness = unique_count / total_count if total_count > 0 else 1.0
            result.set_quality_metric(f"{col}_uniqueness", round(uniqueness, 3))
        
        result.valid_rows = len(valid_rows)
        return result


class DataQualityChecker:
    """Comprehensive data quality assessment tool."""
    
    def __init__(self):
        self.validators = []
    
    def add_validator(self, validator: BaseValidator) -> 'DataQualityChecker':
        """Add a validator to the checker."""
        self.validators.append(validator)
        return self
    
    def assess_quality(self, df: pd.DataFrame) -> ValidationResult:
        """Perform comprehensive quality assessment."""
        combined_result = ValidationResult()
        combined_result.total_rows = len(df)
        
        valid_rows = set(range(len(df)))
        
        # Run all validators
        for validator in self.validators:
            logger.info(f"Running {validator.__class__.__name__}...")
            result = validator.validate(df)
            
            # Combine errors and warnings
            combined_result.errors.extend(result.errors)
            combined_result.warnings.extend(result.warnings)
            
            # Combine quality metrics
            combined_result.quality_metrics.update(result.quality_metrics)
            
            # Track valid rows (intersection of all validator results)
            validator_valid_rows = set(range(len(df)))
            for error in result.errors:
                if error['row'] >= 0:  # Skip dataset-level errors
                    validator_valid_rows.discard(error['row'])
            
            valid_rows &= validator_valid_rows
        
        combined_result.valid_rows = len(valid_rows)
        
        # Add overall quality metrics
        self._add_overall_metrics(df, combined_result)
        
        return combined_result
    
    def _add_overall_metrics(self, df: pd.DataFrame, result: ValidationResult):
        """Add overall data quality metrics."""
        # Overall completeness
        total_cells = df.shape[0] * df.shape[1]
        null_cells = df.isna().sum().sum()
        overall_completeness = 1 - (null_cells / total_cells) if total_cells > 0 else 1.0
        result.set_quality_metric('overall_completeness', round(overall_completeness, 3))
        
        # Column statistics
        result.set_quality_metric('total_columns', df.shape[1])
        result.set_quality_metric('total_rows', df.shape[0])
        
        # Data type distribution
        dtype_counts = df.dtypes.value_counts().to_dict()
        result.set_quality_metric('column_types', {str(k): v for k, v in dtype_counts.items()})
        
        # Memory usage
        memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        result.set_quality_metric('memory_usage_mb', round(memory_mb, 2))


class SchemaValidator(BaseValidator):
    """Validator that checks DataFrame against an expected schema."""
    
    def __init__(self, schema: Dict[str, Any]):
        """
        Initialize schema validator.
        
        Parameters
        ----------
        schema : Dict[str, Any]
            Expected schema specification with column names and constraints.
            
        Example schema:
        {
            'columns': ['id', 'name', 'email', 'age'],
            'required_columns': ['id', 'name'],
            'column_types': {'id': 'int64', 'age': 'int64'},
            'unique_columns': ['id'],
            'patterns': {'email': r'.+@.+\..+'},
            'ranges': {'age': {'min': 0, 'max': 150}}
        }
        """
        self.schema = schema
    
    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """Validate DataFrame against schema."""
        result = ValidationResult()
        result.total_rows = len(df)
        
        # Check expected columns exist
        expected_columns = self.schema.get('columns', [])
        missing_columns = set(expected_columns) - set(df.columns)
        extra_columns = set(df.columns) - set(expected_columns)
        
        for col in missing_columns:
            result.add_error('schema', -1, f"Missing expected column: {col}")
        
        for col in extra_columns:
            result.add_warning('schema', -1, f"Unexpected column found: {col}")
        
        # Validate column types
        expected_types = self.schema.get('column_types', {})
        for col, expected_type in expected_types.items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                if actual_type != expected_type:
                    result.add_warning(col, -1, f"Type mismatch: expected {expected_type}, got {actual_type}")
        
        # Use specific validators for detailed checks
        validators = []
        
        # Add completeness validator
        required_columns = self.schema.get('required_columns', [])
        if required_columns:
            validators.append(CompletenessValidator(required_columns=required_columns))
        
        # Add uniqueness validator
        unique_columns = self.schema.get('unique_columns', [])
        if unique_columns:
            validators.append(UniqueValidator(unique_columns=unique_columns))
        
        # Add pattern validator
        patterns = self.schema.get('patterns', {})
        if patterns:
            validators.append(PatternValidator(patterns=patterns))
        
        # Add range validator
        ranges = self.schema.get('ranges', {})
        if ranges:
            validators.append(RangeValidator(ranges=ranges))
        
        # Run all sub-validators
        valid_rows = set(range(len(df)))
        for validator in validators:
            sub_result = validator.validate(df)
            result.errors.extend(sub_result.errors)
            result.warnings.extend(sub_result.warnings)
            result.quality_metrics.update(sub_result.quality_metrics)
            
            # Update valid rows
            for error in sub_result.errors:
                if error['row'] >= 0:
                    valid_rows.discard(error['row'])
        
        result.valid_rows = len(valid_rows)
        return result


# Convenience functions for common validation scenarios
def validate_emails(df: pd.DataFrame, columns: List[str], strict: bool = True) -> ValidationResult:
    """Convenience function to validate email columns."""
    validator = EmailValidator(strict=strict)
    return validator.validate(df, columns)


def validate_ranges(df: pd.DataFrame, ranges: Dict[str, Dict[str, Any]]) -> ValidationResult:
    """Convenience function to validate numeric ranges."""
    validator = RangeValidator(ranges)
    return validator.validate(df)


def validate_completeness(df: pd.DataFrame, required_columns: List[str]) -> ValidationResult:
    """Convenience function to validate required fields."""
    validator = CompletenessValidator(required_columns=required_columns)
    return validator.validate(df)


def validate_schema(df: pd.DataFrame, schema: Dict[str, Any]) -> ValidationResult:
    """Convenience function to validate against a schema."""
    validator = SchemaValidator(schema)
    return validator.validate(df)