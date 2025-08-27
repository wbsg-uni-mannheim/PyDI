"""
Utility functions for PyDI.

This module exposes generic helper functions that can be reused across
modules, such as tokenizers and comparators for string similarity, date
proximity functions, normalization utilities, and a similarity registry for textdistance metrics.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Callable, Optional, Set

import pandas as pd

# Import normalization utilities for convenient access
from .normalization import (
    apply_boolean_normalization,
    apply_currency_parsing,
    apply_encoding_fixes,
    apply_html_removal,
    apply_phone_normalization,
    apply_text_cleaning,
    clean_text,
    extract_numeric,
    fix_encoding,
    normalize_boolean,
    normalize_phone_number,
    normalize_whitespace,
    parse_currency,
    parse_percentage,
    remove_accents,
    remove_html_tags,
    standardize_case,
    standardize_country_name,
)


def _tokenize(text: str) -> Set[str]:
    """Tokenise a string into a set of lowercase alphanumeric tokens."""
    return set(re.findall(r"\w+", str(text).lower()))


def jaccard(column: str) -> Callable[[pd.Series, pd.Series], float]:
    """Return a comparator that computes Jaccard similarity between two values.

    Parameters
    ----------
    column : str
        The name of the column to compare.

    Returns
    -------
    callable
        A function that takes two pandas Series (representing rows) and
        returns a float in [0,1] measuring the Jaccard similarity of
        tokenised values in the specified column.
    """

    def comparator(rec1: pd.Series, rec2: pd.Series) -> float:
        s1 = _tokenize(rec1.get(column, ""))
        s2 = _tokenize(rec2.get(column, ""))
        if not s1 and not s2:
            return 1.0
        return len(s1 & s2) / len(s1 | s2)

    return comparator


def date_within_years(column: str, years: int) -> Callable[[pd.Series, pd.Series], float]:
    """Return a comparator that scores dates based on proximity within a number of years.

    Parameters
    ----------
    column : str
        The name of the date column to compare.
    years : int
        The maximum difference in years for a similarity of 1. Differences
        greater than ``years`` result in a similarity of 0.

    Returns
    -------
    callable
        A function that takes two pandas Series and returns a float in
        [0,1] where 1 means the dates are within ``years``, and 0 means they
        are farther apart.
    """

    def comparator(rec1: pd.Series, rec2: pd.Series) -> float:
        date1 = rec1.get(column)
        date2 = rec2.get(column)
        try:
            dt1 = pd.to_datetime(date1, errors="coerce")
            dt2 = pd.to_datetime(date2, errors="coerce")
        except Exception:
            return 0.0
        if pd.isna(dt1) or pd.isna(dt2):
            return 0.0
        diff_years = abs((dt1 - dt2).days) / 365.25
        return 1.0 if diff_years <= years else 0.0

    return comparator


def lowercase(text: str) -> str:
    """Lowercase a string; returns ``None`` for non‑string inputs."""
    return str(text).lower() if isinstance(text, str) else None


def strip(text: str) -> str:
    """Strip leading and trailing whitespace from a string; returns ``None`` for non‑string inputs."""
    return str(text).strip() if isinstance(text, str) else None


def remove_punctuation(text: str) -> str:
    """Remove punctuation characters from a string.

    Non‑string inputs return ``None``. This function uses the standard
    ``string.punctuation`` set to remove ASCII punctuation. For more
    advanced Unicode punctuation removal, consider using the ``regex``
    package.
    """
    import string

    if not isinstance(text, str):
        return None
    return text.translate(str.maketrans("", "", string.punctuation))


# Import similarity registry utilities
from .similarity_registry import SimilarityRegistry, get_similarity_function, list_similarity_functions

__all__ = [
    "jaccard",
    "date_within_years", 
    "lowercase",
    "strip",
    "remove_punctuation",
    "SimilarityRegistry",
    "get_similarity_function",
    "list_similarity_functions",
]