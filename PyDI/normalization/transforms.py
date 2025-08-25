"""
Discoverable, fast, per-column transformations for PyDI normalization.

Usage
-----
from PyDI.normalization.transforms import Transforms as T, list_transforms

transforms = {
    'title': [T.strip(), T.normalize_whitespace(), T.lower()],
    'year': T.to_numeric(),
    ('gross', 'budget'): T.to_numeric(),
    'released': T.to_datetime(infer=True, dayfirst=True),
    'country': T.replace({'U.S.': 'USA', 'UK': 'United Kingdom'}),
}

See list_transforms() for available transforms.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import pandas as pd


class Transforms:
    """Factory methods returning vectorized Series→Series transformations.

    Each method returns a callable that accepts a pandas Series and returns a
    transformed Series. Methods are designed to be fast and chainable.
    """

    # String cleanup
    @staticmethod
    def lower() -> Callable[[pd.Series], pd.Series]:
        """Lowercase strings (safe for mixed dtypes)."""
        return lambda s: s.astype("string").str.lower()

    @staticmethod
    def upper() -> Callable[[pd.Series], pd.Series]:
        """Uppercase strings (safe for mixed dtypes)."""
        return lambda s: s.astype("string").str.upper()

    @staticmethod
    def strip() -> Callable[[pd.Series], pd.Series]:
        """Trim leading/trailing whitespace."""
        return lambda s: s.astype("string").str.strip()

    @staticmethod
    def normalize_whitespace() -> Callable[[pd.Series], pd.Series]:
        """Collapse internal whitespace and strip."""
        return lambda s: s.astype("string").str.replace(r"\s+", " ", regex=True).str.strip()

    @staticmethod
    def drop_non_ascii() -> Callable[[pd.Series], pd.Series]:
        """Remove non-ASCII characters."""
        return lambda s: s.astype("string").str.encode("ascii", "ignore").str.decode("ascii")

    # Type conversion
    @staticmethod
    def to_numeric(errors: str = "coerce") -> Callable[[pd.Series], pd.Series]:
        """Convert to numeric; default coerces invalid values to NaN."""
        def _fn(s: pd.Series) -> pd.Series:
            # Strip common thousands separators before conversion
            ss = s.astype("string").str.replace(
                r"[\s\u00A0,'’]", "", regex=True)
            return pd.to_numeric(ss, errors=errors)

        return _fn

    @staticmethod
    def to_datetime(errors: str = "coerce", infer: bool = True, dayfirst: Optional[bool] = None) -> Callable[[pd.Series], pd.Series]:
        """Convert to datetime with common defaults."""
        def _fn(s: pd.Series) -> pd.Series:
            return pd.to_datetime(s, errors=errors, infer_datetime_format=infer, dayfirst=dayfirst)

        return _fn

    # NA handling
    @staticmethod
    def fill_na(value: Any) -> Callable[[pd.Series], pd.Series]:
        """Fill NA/NaN with a constant value."""
        return lambda s: s.fillna(value)

    # Generic replace/map
    @staticmethod
    def replace(mapping: Dict[Any, Any]) -> Callable[[pd.Series], pd.Series]:
        """Replace values using a mapping dict."""
        return lambda s: s.replace(mapping)

    @staticmethod
    def regex_replace(pattern: str, repl: str, flags: int = 0) -> Callable[[pd.Series], pd.Series]:
        """Regex replace pattern with repl."""
        return lambda s: s.astype("string").str.replace(pattern, repl, regex=True, flags=flags)

    @staticmethod
    def map(func: Callable[[Any], Any]) -> Callable[[pd.Series], pd.Series]:
        """Apply a Python function element-wise (vectorized where possible)."""
        return lambda s: s.map(func)


# Built-in transform registry (default configurations)
BUILTIN_TRANSFORMS: Dict[str, Callable[[pd.Series], pd.Series]] = {
    "lower": Transforms.lower(),
    "upper": Transforms.upper(),
    "strip": Transforms.strip(),
    "normalize_whitespace": Transforms.normalize_whitespace(),
    "to_numeric": Transforms.to_numeric(),
    "to_datetime": Transforms.to_datetime(),
    "fill_na_empty": Transforms.fill_na(""),
    "fill_na_zero": Transforms.fill_na(0),
    "drop_non_ascii": Transforms.drop_non_ascii(),
}


def list_transforms() -> List[Dict[str, Any]]:
    """Return a list of available built-in transforms with descriptions.

    Each entry includes: name, summary, and usage hints.
    """
    return [
        {"name": "lower", "summary": "Lowercase strings"},
        {"name": "upper", "summary": "Uppercase strings"},
        {"name": "strip", "summary": "Trim leading/trailing whitespace"},
        {"name": "normalize_whitespace",
            "summary": "Collapse internal whitespace and strip"},
        {"name": "to_numeric",
            "summary": "Convert to numeric (thousands stripped, errors=coerce)"},
        {"name": "to_datetime",
            "summary": "Convert to datetime (infer formats)"},
        {"name": "fill_na_empty", "summary": "Fill NA with empty string"},
        {"name": "fill_na_zero", "summary": "Fill NA with 0"},
        {"name": "drop_non_ascii", "summary": "Remove non-ASCII characters"},
    ]


def get_transform(name: str) -> Optional[Callable[[pd.Series], pd.Series]]:
    """Get a built-in transform by name.

    For parameterized variants, use the factory methods on Transforms.
    """
    return BUILTIN_TRANSFORMS.get(name)


# Lightweight alias for ergonomics
T = Transforms
