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

from .text import TextNormalizer
from .types import TypeConverter, BooleanParser, NumericParser, DateNormalizer
from .units import UnitNormalizer, UnitRegistry, QuantityModifier, UnitCategory

logger = logging.getLogger(__name__)


class ValueNormalizer:
    """
    Simple value normalizer for basic normalization tasks.
    
    Provides straightforward value normalization without complex
    type detection or unit conversion.
    """
    
    def __init__(
        self,
        text_normalizer: Optional[TextNormalizer] = None,
        numeric_parser: Optional[NumericParser] = None,
        boolean_parser: Optional[BooleanParser] = None,
        date_normalizer: Optional[DateNormalizer] = None
    ):
        self.text_normalizer = text_normalizer or TextNormalizer()
        self.numeric_parser = numeric_parser or NumericParser()
        self.boolean_parser = boolean_parser or BooleanParser()
        self.date_normalizer = date_normalizer or DateNormalizer()
    
    def normalize_value(self, value: Any, target_type: str = 'auto') -> Any:
        """
        Normalize a single value.
        
        Parameters
        ----------
        value : Any
            Value to normalize.
        target_type : str, default 'auto'
            Target type: 'text', 'numeric', 'boolean', 'date', or 'auto'.
            
        Returns
        -------
        Any
            Normalized value.
        """
        if pd.isna(value):
            return value
        
        if target_type == 'auto':
            # Auto-detect and normalize
            return self._auto_normalize(value)
        elif target_type == 'text':
            return self.text_normalizer.clean_text(str(value))
        elif target_type == 'numeric':
            return self.numeric_parser.parse_numeric(str(value))
        elif target_type == 'boolean':
            return self.boolean_parser.parse_boolean(value)
        elif target_type == 'date':
            return self.date_normalizer.normalize_date(value)
        else:
            logger.warning(f"Unknown target type: {target_type}")
            return value
    
    def _auto_normalize(self, value: Any) -> Any:
        """Auto-detect type and normalize accordingly."""
        value_str = str(value).strip()
        
        # Try numeric first
        numeric_result = self.numeric_parser.parse_numeric(value_str)
        if numeric_result is not None:
            return numeric_result
        
        # Try boolean
        boolean_result = self.boolean_parser.parse_boolean(value)
        if boolean_result is not None:
            return boolean_result
        
        # Try date
        date_result = self.date_normalizer.normalize_date(value)
        if date_result is not None:
            return date_result
        
        # Default to text normalization
        return self.text_normalizer.clean_text(value_str)
    
    def normalize_column(self, series: pd.Series, target_type: str = 'auto') -> pd.Series:
        """Normalize entire column."""
        return series.apply(lambda x: self.normalize_value(x, target_type))


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


class ListValueProcessor:
    """
    Specialized processor for list-format values.
    
    Handles Winter's list format: {value1|value2|value3}
    """
    
    def __init__(self, item_normalizer: Optional[ValueNormalizer] = None):
        self.item_normalizer = item_normalizer or ValueNormalizer()
        self.list_pattern = re.compile(r'^\{([^}]+)\}$')
    
    def is_list_format(self, value: str) -> bool:
        """Check if value is in list format."""
        if pd.isna(value) or not value:
            return False
        return bool(self.list_pattern.match(str(value).strip()))
    
    def parse_list(self, value: str) -> List[str]:
        """Parse list value into individual items."""
        if not self.is_list_format(value):
            return [value]
        
        match = self.list_pattern.match(value.strip())
        if match:
            items = match.group(1).split('|')
            return [item.strip() for item in items if item.strip()]
        return []
    
    def normalize_list_items(self, items: List[str], target_type: str = 'auto') -> List[Any]:
        """Normalize individual list items."""
        normalized_items = []
        for item in items:
            normalized = self.item_normalizer.normalize_value(item, target_type)
            normalized_items.append(normalized)
        return normalized_items
    
    def process_list_value(self, value: str, target_type: str = 'auto') -> Union[List[Any], Any]:
        """Process list value or return single normalized value."""
        if self.is_list_format(value):
            items = self.parse_list(value)
            return self.normalize_list_items(items, target_type)
        else:
            return self.item_normalizer.normalize_value(value, target_type)


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


# Convenience functions for value normalization
def normalize_value(value: Any, target_type: str = 'auto') -> Any:
    """
    Quick value normalization with auto-detection.
    
    Parameters
    ----------
    value : Any
        Value to normalize.
    target_type : str, default 'auto'
        Target type for normalization.
        
    Returns
    -------
    Any
        Normalized value.
    """
    normalizer = ValueNormalizer()
    return normalizer.normalize_value(value, target_type)


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


def normalize_list(value: str) -> Union[List[Any], Any]:
    """Normalize list-format value."""
    processor = ListValueProcessor()
    return processor.process_list_value(value)


def normalize_coordinate(value: str) -> Optional[str]:
    """Normalize coordinate value."""
    from .types import parse_coordinate
    parsed = parse_coordinate(value)
    if parsed:
        return f"{parsed[0]:.6f}, {parsed[1]:.6f}"
    return None


def clean_nulls(series: pd.Series, replacement: Any = None) -> pd.Series:
    """Clean null values in a series."""
    handler = NullValueHandler(replacement)
    return handler.normalize_column_nulls(series)


def advanced_normalize_value(
    value: Any, 
    data_type: str, 
    unit_category: Optional[str] = None,
    specific_unit: Optional[str] = None
) -> Any:
    """
    Advanced value normalization with type-specific handling.
    
    Parameters
    ----------
    value : Any
        Value to normalize.
    data_type : str
        Detected data type.
    unit_category : str, optional
        Unit category if applicable.
    specific_unit : str, optional
        Specific unit if detected.
        
    Returns
    -------
    Any
        Normalized value.
    """
    normalizer = AdvancedValueNormalizer()
    return normalizer.normalize_value(value, data_type, unit_category, specific_unit)