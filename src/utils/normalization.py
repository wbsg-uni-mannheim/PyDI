"""
Normalization utility functions for PyDI.

This module provides utility functions for common normalization tasks that can
be reused across the framework. It includes text cleaning, number parsing,
date handling, and other data preprocessing utilities.
"""

from __future__ import annotations

import html
import re
import string
import unicodedata
from datetime import datetime
from typing import Any, List, Optional, Union

import pandas as pd


def clean_text(
    text: str,
    lowercase: bool = True,
    remove_punctuation: bool = False,
    remove_numbers: bool = False,
    normalize_whitespace: bool = True,
) -> str:
    """
    Clean and normalize text data.
    
    Parameters
    ----------
    text : str
        Text to clean.
    lowercase : bool, default True
        Convert to lowercase.
    remove_punctuation : bool, default False
        Remove punctuation characters.
    remove_numbers : bool, default False
        Remove numeric characters.
    normalize_whitespace : bool, default True
        Normalize whitespace (multiple spaces to single).
        
    Returns
    -------
    str
        Cleaned text.
    """
    if pd.isna(text) or not isinstance(text, str):
        return text
    
    # Basic cleaning
    cleaned = text.strip()
    
    # Normalize Unicode
    cleaned = unicodedata.normalize('NFKC', cleaned)
    
    # Convert to lowercase
    if lowercase:
        cleaned = cleaned.lower()
    
    # Remove punctuation
    if remove_punctuation:
        cleaned = cleaned.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    if remove_numbers:
        cleaned = re.sub(r'\d+', '', cleaned)
    
    # Normalize whitespace
    if normalize_whitespace:
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned


def remove_html_tags(text: str) -> str:
    """
    Remove HTML tags and decode HTML entities from text.
    
    Parameters
    ----------
    text : str
        Text containing HTML.
        
    Returns
    -------
    str
        Text with HTML removed.
    """
    if pd.isna(text) or not isinstance(text, str):
        return text
    
    # Remove HTML tags
    cleaned = re.sub(r'<[^>]+>', '', text)
    
    # Decode HTML entities
    cleaned = html.unescape(cleaned)
    
    # Clean up extra whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned


def normalize_phone_number(phone: str, country_code: str = 'US') -> Optional[str]:
    """
    Normalize phone number to a standard format.
    
    Parameters
    ----------
    phone : str
        Phone number to normalize.
    country_code : str, default 'US'
        Country code for phone number formatting.
        
    Returns
    -------
    str or None
        Normalized phone number or None if invalid.
    """
    if pd.isna(phone) or not isinstance(phone, str):
        return None
    
    # Remove all non-digit characters
    digits = re.sub(r'\D', '', phone)
    
    if not digits:
        return None
    
    # Handle US numbers
    if country_code == 'US':
        if len(digits) == 10:
            return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
        elif len(digits) == 11 and digits[0] == '1':
            return f"+1 ({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
    
    # For other countries or invalid formats, return digits with + prefix
    if len(digits) >= 7:
        return f"+{digits}"
    
    return None


def parse_currency(value: str, currency_symbol: str = '$') -> Optional[float]:
    """
    Parse currency string to float value.
    
    Parameters
    ----------
    value : str
        Currency string to parse.
    currency_symbol : str, default '$'
        Expected currency symbol.
        
    Returns
    -------
    float or None
        Parsed currency value or None if invalid.
    """
    if pd.isna(value) or not isinstance(value, str):
        return None
    
    # Remove currency symbols and spaces
    cleaned = value.strip()
    for symbol in ['$', '€', '£', '¥', currency_symbol]:
        cleaned = cleaned.replace(symbol, '')
    
    # Remove commas (thousands separators)
    cleaned = cleaned.replace(',', '')
    
    # Handle parentheses for negative values
    if cleaned.startswith('(') and cleaned.endswith(')'):
        cleaned = '-' + cleaned[1:-1]
    
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_percentage(value: str) -> Optional[float]:
    """
    Parse percentage string to float (as decimal).
    
    Parameters
    ----------
    value : str
        Percentage string to parse.
        
    Returns
    -------
    float or None
        Parsed percentage as decimal (e.g., 0.15 for "15%") or None if invalid.
    """
    if pd.isna(value) or not isinstance(value, str):
        return None
    
    cleaned = value.strip().replace('%', '')
    
    try:
        percentage = float(cleaned)
        return percentage / 100.0
    except ValueError:
        return None


def normalize_boolean(value: Any) -> Optional[bool]:
    """
    Normalize various boolean representations to True/False.
    
    Parameters
    ----------
    value : Any
        Value to convert to boolean.
        
    Returns
    -------
    bool or None
        Normalized boolean or None if cannot be determined.
    """
    if pd.isna(value):
        return None
    
    if isinstance(value, bool):
        return value
    
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        elif value == 0:
            return False
        else:
            return None
    
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in ['true', 't', 'yes', 'y', '1', 'on']:
            return True
        elif cleaned in ['false', 'f', 'no', 'n', '0', 'off']:
            return False
    
    return None


def standardize_country_name(country: str, country_mapping: Optional[dict] = None) -> Optional[str]:
    """
    Standardize country names using a mapping dictionary.
    
    Parameters
    ----------
    country : str
        Country name to standardize.
    country_mapping : dict, optional
        Custom mapping dictionary. If None, uses basic mappings.
        
    Returns
    -------
    str or None
        Standardized country name or None if not found.
    """
    if pd.isna(country) or not isinstance(country, str):
        return None
    
    # Default country mappings
    default_mapping = {
        'usa': 'United States',
        'us': 'United States',
        'united states of america': 'United States',
        'uk': 'United Kingdom',
        'britain': 'United Kingdom',
        'great britain': 'United Kingdom',
        'england': 'United Kingdom',
        'deutschland': 'Germany',
        'de': 'Germany',
    }
    
    mapping = country_mapping or default_mapping
    
    cleaned = country.strip().lower()
    
    # Check direct mapping
    if cleaned in mapping:
        return mapping[cleaned]
    
    # Check partial matches
    for key, value in mapping.items():
        if key in cleaned or cleaned in key:
            return value
    
    # Return title case if no mapping found
    return country.strip().title()


def extract_numeric(text: str, return_first: bool = True) -> Union[List[float], Optional[float]]:
    """
    Extract numeric values from text.
    
    Parameters
    ----------
    text : str
        Text to extract numbers from.
    return_first : bool, default True
        If True, return first number found. If False, return all numbers.
        
    Returns
    -------
    float or List[float] or None
        Extracted number(s) or None if no numbers found.
    """
    if pd.isna(text) or not isinstance(text, str):
        return None if return_first else []
    
    # Pattern to match integers and floats (including negative)
    pattern = r'-?\d+\.?\d*'
    matches = re.findall(pattern, text)
    
    if not matches:
        return None if return_first else []
    
    try:
        numbers = [float(match) for match in matches if match not in ['-', '.']]
        if return_first:
            return numbers[0] if numbers else None
        return numbers
    except ValueError:
        return None if return_first else []


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text (remove extra spaces, tabs, newlines).
    
    Parameters
    ----------
    text : str
        Text to normalize.
        
    Returns
    -------
    str
        Text with normalized whitespace.
    """
    if pd.isna(text) or not isinstance(text, str):
        return text
    
    # Replace multiple whitespace characters with single space
    normalized = re.sub(r'\s+', ' ', text.strip())
    
    return normalized


def remove_accents(text: str) -> str:
    """
    Remove accents and diacritical marks from text.
    
    Parameters
    ----------
    text : str
        Text with potential accents.
        
    Returns
    -------
    str
        Text without accents.
    """
    if pd.isna(text) or not isinstance(text, str):
        return text
    
    # Normalize to NFD (decomposed form) and filter out combining characters
    normalized = unicodedata.normalize('NFD', text)
    without_accents = ''.join(
        char for char in normalized 
        if unicodedata.category(char) != 'Mn'
    )
    
    return without_accents


def standardize_case(text: str, case: str = 'lower') -> str:
    """
    Standardize text case.
    
    Parameters
    ----------
    text : str
        Text to standardize.
    case : str, default 'lower'
        Target case: 'lower', 'upper', 'title', 'sentence'.
        
    Returns
    -------
    str
        Text in specified case.
    """
    if pd.isna(text) or not isinstance(text, str):
        return text
    
    if case == 'lower':
        return text.lower()
    elif case == 'upper':
        return text.upper()
    elif case == 'title':
        return text.title()
    elif case == 'sentence':
        return text.capitalize()
    else:
        return text


def detect_encoding_issues(text: str) -> bool:
    """
    Detect potential encoding issues in text.
    
    Parameters
    ----------
    text : str
        Text to check.
        
    Returns
    -------
    bool
        True if encoding issues detected.
    """
    if pd.isna(text) or not isinstance(text, str):
        return False
    
    # Common indicators of encoding issues
    issues = [
        '�',  # Replacement character
        'â€™',  # Curly apostrophe encoded as UTF-8 but decoded as Latin-1
        'â€œ',  # Left double quotation mark
        'â€',   # Right double quotation mark
        'â€"',  # Em dash
        'Ã¡', 'Ã©', 'Ã­', 'Ã³', 'Ãº',  # Common accented characters
    ]
    
    return any(issue in text for issue in issues)


def fix_encoding(text: str) -> str:
    """
    Attempt to fix common encoding issues in text.
    
    Parameters
    ----------
    text : str
        Text with potential encoding issues.
        
    Returns
    -------
    str
        Text with encoding issues fixed.
    """
    if pd.isna(text) or not isinstance(text, str):
        return text
    
    # Try using ftfy library if available
    try:
        import ftfy
        return ftfy.fix_text(text)
    except ImportError:
        pass
    
    # Fallback: manual fixes for common issues
    fixes = {
        'â€™': "'",
        'â€œ': '"',
        'â€': '"',
        'â€"': '—',
        'Ã¡': 'á',
        'Ã©': 'é',
        'Ã­': 'í',
        'Ã³': 'ó',
        'Ãº': 'ú',
    }
    
    fixed = text
    for wrong, right in fixes.items():
        fixed = fixed.replace(wrong, right)
    
    return fixed


# Convenience functions for applying utilities to pandas Series/DataFrames

def apply_text_cleaning(series: pd.Series, **kwargs) -> pd.Series:
    """Apply text cleaning to a pandas Series."""
    return series.apply(lambda x: clean_text(x, **kwargs))


def apply_html_removal(series: pd.Series) -> pd.Series:
    """Apply HTML tag removal to a pandas Series."""
    return series.apply(remove_html_tags)


def apply_phone_normalization(series: pd.Series, country_code: str = 'US') -> pd.Series:
    """Apply phone number normalization to a pandas Series."""
    return series.apply(lambda x: normalize_phone_number(x, country_code))


def apply_currency_parsing(series: pd.Series, currency_symbol: str = '$') -> pd.Series:
    """Apply currency parsing to a pandas Series."""
    return series.apply(lambda x: parse_currency(x, currency_symbol))


def apply_boolean_normalization(series: pd.Series) -> pd.Series:
    """Apply boolean normalization to a pandas Series."""
    return series.apply(normalize_boolean)


def apply_encoding_fixes(series: pd.Series) -> pd.Series:
    """Apply encoding fixes to a pandas Series."""
    return series.apply(fix_encoding)