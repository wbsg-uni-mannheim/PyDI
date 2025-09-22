"""
Value-level normalization utilities for PyDI.

This module provides comprehensive value normalization logic including
value-specific normalizers and the advanced value normalizer that
handles complex transformations with unit conversion and quantity scaling.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from .types import TypeConverter, BooleanParser, NumericParser, DateNormalizer
from .units import UnitNormalizer, UnitRegistry, QuantityModifier, UnitCategory

logger = logging.getLogger(__name__)


class AdvancedValueNormalizer:
    """
    Sophisticated value normalizer with comprehensive normalization capabilities.
    
    Matches Winter's AdvancedValueNormalizer with unit conversion, quantity
    scaling, and type-specific transformations.
    """
    
    def __init__(
        self,
        unit_registry: Optional[UnitRegistry] = None,
        type_converter: Optional[TypeConverter] = None,
        enable_unit_conversion: bool = True,
        enable_quantity_scaling: bool = True
    ):
        self.unit_registry = unit_registry or UnitRegistry(comprehensive=True)
        self.type_converter = type_converter or TypeConverter()
        self.enable_unit_conversion = enable_unit_conversion
        self.enable_quantity_scaling = enable_quantity_scaling
        
        if self.enable_unit_conversion:
            self.unit_normalizer = UnitNormalizer(self.unit_registry)
        
        # Quantity modifiers mapping
        if self.enable_quantity_scaling:
            self.quantity_modifiers = {}
            for modifier in QuantityModifier:
                for keyword in modifier.keywords:
                    self.quantity_modifiers[keyword.lower()] = modifier.multiplier
    
    def normalize_value(
        self, 
        value: Any, 
        data_type: str,
        unit_category: Optional[str] = None,
        specific_unit: Optional[str] = None
    ) -> Optional[Any]:
        """
        Normalize value with comprehensive type-specific handling.
        
        Parameters
        ----------
        value : Any
            Original value to normalize.
        data_type : str
            Detected data type ('numeric', 'string', 'date', etc.).
        unit_category : str, optional
            Unit category if applicable.
        specific_unit : str, optional
            Specific unit detected.
            
        Returns
        -------
        Optional[Any]
            Normalized value or None if normalization fails.
        """
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        
        value_str = str(value).strip()
        
        try:
            if data_type == 'numeric':
                return self._normalize_numeric(value_str, unit_category, specific_unit)
            elif data_type == 'date':
                return self._normalize_date(value_str)
            elif data_type == 'boolean':
                return self._normalize_boolean(value_str)
            elif data_type == 'list':
                return self._normalize_list(value_str)
            elif data_type == 'coordinate':
                return self._normalize_coordinate(value_str)
            elif data_type == 'url':
                return self._normalize_url(value_str)
            elif data_type == 'email':
                return self._normalize_email(value_str)
            else:
                # Keep as string for other types
                return self._normalize_string(value_str)
                
        except Exception as e:
            logger.debug(f"Normalization failed for value '{value}': {e}")
            return value_str
    
    def _normalize_numeric(
        self, 
        value: str, 
        unit_category: Optional[str] = None, 
        specific_unit: Optional[str] = None
    ) -> Optional[Union[int, float]]:
        """Normalize numeric values with unit conversion and quantity scaling."""
        
        # First, try unit normalization if enabled and units detected
        if (self.enable_unit_conversion and 
            unit_category and specific_unit and 
            unit_category != 'unknown'):
            
            normalized_unit = self.unit_normalizer.normalize_value(value)
            if normalized_unit:
                numeric_value, _ = normalized_unit
                return numeric_value
        
        # Extract numeric part and potential quantity modifier
        numeric_match = re.search(r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', value)
        if not numeric_match:
            return None
        
        numeric_str = numeric_match.group(1)
        try:
            numeric_value = float(numeric_str)
        except ValueError:
            return None
        
        # Apply quantity scaling if enabled
        if self.enable_quantity_scaling:
            for modifier_keyword, multiplier in self.quantity_modifiers.items():
                if modifier_keyword in value.lower():
                    numeric_value *= multiplier
                    break
        
        # Return as int if it's a whole number, otherwise float
        if numeric_value.is_integer():
            return int(numeric_value)
        return numeric_value
    
    def _normalize_date(self, value: str) -> Optional[str]:
        """Normalize date values."""
        return self.type_converter.convert_date(value)
    
    def _normalize_boolean(self, value: str) -> Optional[bool]:
        """Normalize boolean values."""
        return self.type_converter.convert_boolean(value)
    
    def _normalize_list(self, value: str) -> List[str]:
        """Normalize list values (format: {val1|val2|val3})."""
        list_match = re.match(r'^\{([^}]+)\}$', value)
        if list_match:
            items = list_match.group(1).split('|')
            return [item.strip() for item in items]
        return [value]
    
    def _normalize_coordinate(self, value: str) -> Optional[str]:
        """Normalize coordinate values."""
        return self.type_converter.convert_coordinate(value)
    
    def _normalize_url(self, value: str) -> Optional[str]:
        """Normalize URL values."""
        return self.type_converter.convert_url(value)
    
    def _normalize_email(self, value: str) -> str:
        """Normalize email values."""
        # Basic email normalization (lowercase domain)
        if '@' in value:
            local, domain = value.rsplit('@', 1)
            return f"{local}@{domain.lower()}"
        return value.lower()
    
    def _normalize_string(self, value: str) -> str:
        """Normalize string values."""
        text_normalizer = TextNormalizer()
        return text_normalizer.clean_text(value)


class NullValueHandler:
    """
    Handler for various null value representations.
    
    Provides comprehensive null detection and standardization.
    """
    
    def __init__(self, null_replacement: str = None):
        self.null_replacement = null_replacement
        
        # Comprehensive null patterns
        self.null_patterns = {
            '', '__', '-', '_', '?', 'unknown', '- -', 'n/a', '•', 
            '- - -', '.', '??', '(n/a)', 'null', 'none', 'nil', 'na',
            'missing', 'undefined', 'void', 'tbd', 'tba', 'not available',
            'not applicable', 'no data', 'no info', '---', '___', '...',
            'n.a.', 'n.d.', 'nd', 'n\\a', 'empty', 'blank',
            'not specified', 'not set', 'not given', 'not provided',
            'not entered', 'not found', 'not recorded', 'not listed',
            'no entry', 'no record', 'no information'
        }
    
    def is_null_value(self, value: Any) -> bool:
        """Check if value should be considered null."""
        # Standard pandas null checks
        if pd.isna(value) or value is None:
            return True
        
        # String-based null pattern matching
        if isinstance(value, str):
            cleaned = value.strip().lower()
            return cleaned in self.null_patterns
        
        return False
    
    def normalize_nulls(self, value: Any) -> Any:
        """Replace null values with standard representation."""
        if self.is_null_value(value):
            return self.null_replacement if self.null_replacement is not None else None
        return value
    
    def normalize_column_nulls(self, series: pd.Series) -> pd.Series:
        """Normalize nulls in entire column."""
        return series.apply(self.normalize_nulls)


def normalize_numeric(value: str) -> Optional[Union[int, float]]:
    """Normalize numeric value."""
    parser = NumericParser()
    return parser.parse_numeric(value)


def normalize_date(value: Any) -> Optional[str]:
    """Normalize date value."""
    normalizer = DateNormalizer()
    return normalizer.normalize_date(value)


def normalize_boolean(value: Any) -> Optional[bool]:
    """Normalize boolean value."""
    parser = BooleanParser()
    return parser.parse_boolean(value)


def clean_nulls(series: pd.Series, replacement: Any = None) -> pd.Series:
    """Clean null values in a series."""
    handler = NullValueHandler(replacement)
    return handler.normalize_column_nulls(series)


    
