"""
Information extraction utilities for PyDI.

This subpackage defines abstract base classes and simple implementations
for extracting values via regular expressions or code. Most classes are 
intentionally skeletal and are intended to be extended to support more 
complex use cases, such as LLM-based extraction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict

import pandas as pd


class BaseExtractor(ABC):
    """Abstract base class for feature extractors."""

    @abstractmethod
    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class RegexExtractor(BaseExtractor):
    """Extract new columns using regular expressions.

    Parameters
    ----------
    rules : dict
        A mapping from new column names to regular expression patterns.
    source_column : str
        The name of the source column on which to apply the patterns.
    """

    def __init__(self, rules: Dict[str, str], source_column: str) -> None:
        self.rules = rules
        self.source_column = source_column

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        import re
        for new_col, pattern in self.rules.items():
            result[new_col] = result[self.source_column].astype(str).apply(
                lambda x: re.search(pattern, x).group(0) if re.search(pattern, x) else None
            )
        return result


class CodeExtractor(BaseExtractor):
    """Extract new columns using callables.

    Parameters
    ----------
    funcs : dict
        A mapping from new column names to callables taking a row and returning a value.
    """

    def __init__(self, funcs: Dict[str, Callable[[pd.Series], Any]]) -> None:
        self.funcs = funcs

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        for new_col, func in self.funcs.items():
            result[new_col] = result.apply(func, axis=1)
        return result
