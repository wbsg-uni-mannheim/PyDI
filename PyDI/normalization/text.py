"""
Text normalization utilities for PyDI.

This module provides text-related normalization functionality:
- TextNormalizer: Basic text cleaning (HTML, whitespace, unicode)
- HeaderNormalizer: Column header normalization
- WebTableNormalizer: Web-scraped table cleaning (delegates to TextNormalizer + HeaderNormalizer)
- BracketContentHandler: Extract/remove bracket content

For advanced tokenization with stemming/stopwords, use the tokenizers in:
- PyDI.utils.SimilarityRegistry.TOKENIZATION_STRATEGIES
- PyDI.entitymatching.blocking.TokenBlocker
"""

from __future__ import annotations

import html
import logging
import re
import string
import unicodedata
from typing import List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


class TextNormalizer:
    """
    Text cleaning and normalization.

    Provides text cleaning capabilities including HTML removal, Unicode
    normalization, case conversion, and whitespace handling.

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
        Fix common encoding issues (uses ftfy if available).
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

    def clean_text(self, text: str) -> str:
        """
        Apply all configured text cleaning operations.

        Parameters
        ----------
        text : str
            Text to clean.

        Returns
        -------
        str
            Cleaned text.
        """
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

    def normalize_column(self, series: pd.Series) -> pd.Series:
        """
        Apply text normalization to a pandas Series.

        Parameters
        ----------
        series : pd.Series
            Series to normalize.

        Returns
        -------
        pd.Series
            Series with normalized text.
        """
        return series.apply(self.clean_text)


class HeaderNormalizer:
    """
    Specialized normalizer for column headers.

    Cleans column headers by removing HTML entities, special characters,
    and standardizing format for better schema matching.

    Parameters
    ----------
    lowercase : bool, default True
        Convert headers to lowercase.
    remove_special_chars : bool, default True
        Remove special characters like dots, dollar signs.
    remove_html : bool, default True
        Remove HTML tags and entities.
    remove_brackets : bool, default False
        Remove content in brackets.
    null_value : str, default "NULL"
        Value to use for null/empty headers.
    replace_whitespace_with_underscore : bool, default False
        If True, collapse whitespace and replace spaces with underscores `_`.
    """

    def __init__(
        self,
        lowercase: bool = True,
        remove_special_chars: bool = True,
        remove_html: bool = True,
        remove_brackets: bool = False,
        null_value: str = "NULL",
        replace_whitespace_with_underscore: bool = False
    ):
        self.lowercase = lowercase
        self.remove_special_chars = remove_special_chars
        self.remove_html = remove_html
        self.remove_brackets = remove_brackets
        self.null_value = null_value
        self.replace_whitespace_with_underscore = replace_whitespace_with_underscore

        # HTML entities mapping (extended)
        self.html_entities = {
            '&nbsp;': ' ', '&nbsp': ' ', 'nbsp': ' ',
            '&amp;': '&', '&lt;': '<', '&gt;': '>',
            '&quot;': '"', '&apos;': "'", '&ndash;': '-',
            '&mdash;': '-', '&hellip;': '...', '&copy;': '(c)',
            '&reg;': '(r)', '&trade;': 'tm'
        }

        # Null value patterns
        self.null_patterns = {
            '', '__', '-', '_', '?', 'unknown', '- -',
            'n/a', '•', '- - -', '.', '??', '(n/a)',
            'null', 'none', 'nil', 'na', 'missing', 'undefined'
        }

        # HTML tag pattern
        self.html_tag_pattern = re.compile(r'<.*?>')
        # Bracket pattern
        self.bracket_pattern = re.compile(r'\(.*?\)')

    def normalize_header(self, header: str) -> str:
        """
        Normalize a single header string.

        Parameters
        ----------
        header : str
            Header string to normalize.

        Returns
        -------
        str
            Normalized header string.
        """
        if header is None:
            return self.null_value

        # Convert to string and handle unicode escaping
        header = str(header)

        # Decode HTML entities
        if self.remove_html:
            header = html.unescape(header)
            for entity, replacement in self.html_entities.items():
                header = header.replace(entity, replacement)
            # Remove HTML tags
            header = self.html_tag_pattern.sub('', header)

        # Clean specific characters
        header = header.replace('"', '')
        header = header.replace('|', ' ')
        header = header.replace(',', '')
        header = header.replace('{', '')
        header = header.replace('}', '')
        header = header.replace('\n', ' ')
        header = header.replace('\r', ' ')
        header = header.replace('\t', ' ')

        # Remove brackets if requested
        if self.remove_brackets:
            header = self.bracket_pattern.sub('', header)

        # Convert to lowercase
        if self.lowercase:
            header = header.lower()

        # Trim whitespace and normalize internal whitespace
        header = re.sub(r'\s+', ' ', header.strip())

        # Optionally replace whitespace with underscores
        if self.replace_whitespace_with_underscore:
            header = header.replace(' ', '_')

        # Remove special characters
        if self.remove_special_chars:
            header = header.replace('.', '')
            header = header.replace('$', '')
            # Remove other punctuation except spaces and underscores
            header = re.sub(r'[^\w\s]', '', header)

        # Check for null patterns
        if header.lower().strip() in self.null_patterns:
            return self.null_value

        return header

    def normalize_headers(self, headers: List[str]) -> List[str]:
        """
        Normalize a list of header strings.

        Parameters
        ----------
        headers : List[str]
            List of header strings to normalize.

        Returns
        -------
        List[str]
            List of normalized header strings.
        """
        return [self.normalize_header(header) for header in headers]

    def normalize_dataframe_headers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply header normalization to DataFrame column names.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to normalize headers for.

        Returns
        -------
        pd.DataFrame
            DataFrame with normalized column names.
        """
        result = df.copy()
        normalized_columns = self.normalize_headers(list(df.columns))
        result.columns = normalized_columns
        return result


class WebTableNormalizer:
    """
    Specialized normalizer for web-scraped table data.

    Delegates to TextNormalizer for value cleaning and HeaderNormalizer
    for header cleaning, with additional null value detection.

    Parameters
    ----------
    remove_brackets_content : bool, default False
        Whether to remove content inside brackets.
    handle_html_entities : bool, default True
        Whether to decode HTML entities.
    null_value : str, default "NULL"
        Value to use for null/empty cells.
    custom_null_patterns : List[str], optional
        Additional null value patterns to recognize.
    """

    def __init__(
        self,
        remove_brackets_content: bool = False,
        handle_html_entities: bool = True,
        null_value: str = "NULL",
        custom_null_patterns: Optional[List[str]] = None
    ):
        self.remove_brackets_content = remove_brackets_content
        self.handle_html_entities = handle_html_entities
        self.null_value = null_value

        # Extended null patterns
        self.null_patterns = {
            '', '__', '-', '_', '?', 'unknown', '- -', 'n/a', '•',
            '- - -', '.', '??', '(n/a)', 'null', 'none', 'nil', 'na',
            'missing', 'undefined', 'void', 'tbd', 'tba', 'not available',
            'not applicable', 'no data', 'no info', '---', '___', '...',
            'n.a.', 'n.d.', 'nd', 'n\\a'
        }

        if custom_null_patterns:
            self.null_patterns.update(custom_null_patterns)

        # Delegate to TextNormalizer for value cleaning
        self._text_normalizer = TextNormalizer(
            lowercase=True,
            strip_whitespace=True,
            remove_html=handle_html_entities,
            remove_punctuation=False,
            fix_encoding=True,
            normalize_unicode=True,
        )

        # Delegate to HeaderNormalizer for header cleaning
        self._header_normalizer = HeaderNormalizer(
            null_value=null_value,
            remove_brackets=remove_brackets_content,
        )

        # Bracket pattern for optional content removal
        self.bracket_pattern = re.compile(r'\(.*?\)')

    def normalize_value(self, value: str) -> str:
        """
        Normalize a cell value for web tables.

        Parameters
        ----------
        value : str
            Cell value to normalize.

        Returns
        -------
        str
            Normalized cell value.
        """
        if pd.isna(value):
            return self.null_value

        try:
            # Use TextNormalizer for basic cleaning
            value = self._text_normalizer.clean_text(value)

            # Check for null patterns
            if value.lower().strip() in self.null_patterns:
                return self.null_value

            # Remove bracket content if requested
            if self.remove_brackets_content:
                value = self.bracket_pattern.sub('', value).strip()

        except Exception as e:
            logger.warning(f"Error normalizing value '{value}': {e}")
            return self.null_value

        return value if value else self.null_value

    def normalize_column(self, series: pd.Series) -> pd.Series:
        """
        Normalize all values in a pandas Series.

        Parameters
        ----------
        series : pd.Series
            Series to normalize.

        Returns
        -------
        pd.Series
            Series with normalized values.
        """
        return series.apply(self.normalize_value)

    def normalize_dataframe(
        self,
        df: pd.DataFrame,
        normalize_headers: bool = True,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Normalize an entire DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to normalize.
        normalize_headers : bool, default True
            Whether to also normalize column headers.
        columns : List[str], optional
            Specific columns to normalize. If None, normalizes all columns.

        Returns
        -------
        pd.DataFrame
            Normalized DataFrame.
        """
        result = df.copy()

        # Normalize headers if requested (using HeaderNormalizer)
        if normalize_headers:
            result = self._header_normalizer.normalize_dataframe_headers(result)

        # Normalize values
        target_columns = columns if columns else result.columns.tolist()

        for col in target_columns:
            if col in result.columns:
                logger.debug(f"Normalizing web table column: {col}")
                result[col] = self.normalize_column(result[col])

        return result


class BracketContentHandler:
    """
    Utility class for handling content in brackets.

    Provides options to remove, extract, or transform bracketed content.

    Parameters
    ----------
    bracket_types : str, default '()[]{}'
        Types of brackets to handle.
    """

    def __init__(self, bracket_types: str = '()[]{}'):
        self.bracket_types = bracket_types

        # Build patterns for different bracket types
        self.bracket_patterns = {}
        bracket_pairs = [('(', ')'), ('[', ']'), ('{', '}'), ('<', '>')]

        for open_br, close_br in bracket_pairs:
            if open_br in bracket_types and close_br in bracket_types:
                pattern = re.compile(f'\\{open_br}.*?\\{close_br}')
                self.bracket_patterns[f'{open_br}{close_br}'] = pattern

    def remove_content(self, text: str, keep_brackets: bool = False) -> str:
        """
        Remove content inside brackets.

        Parameters
        ----------
        text : str
            Text to process.
        keep_brackets : bool, default False
            Whether to keep the bracket characters themselves.

        Returns
        -------
        str
            Text with bracket content removed.
        """
        if pd.isna(text):
            return text

        result = str(text)

        for bracket_type, pattern in self.bracket_patterns.items():
            if keep_brackets:
                # Replace content but keep brackets
                replacement = bracket_type[0] + bracket_type[1]
            else:
                # Remove everything including brackets
                replacement = ''

            result = pattern.sub(replacement, result)

        return result

    def extract_content(self, text: str, bracket_type: str = '()') -> List[str]:
        """
        Extract content from inside brackets.

        Parameters
        ----------
        text : str
            Text to process.
        bracket_type : str, default '()'
            Type of brackets to extract from.

        Returns
        -------
        List[str]
            List of content found inside brackets.
        """
        if pd.isna(text) or bracket_type not in self.bracket_patterns:
            return []

        text = str(text)
        pattern = self.bracket_patterns[bracket_type]
        matches = pattern.findall(text)

        # Remove the bracket characters from matches
        open_br, close_br = bracket_type[0], bracket_type[1]
        content = []
        for match in matches:
            if match.startswith(open_br) and match.endswith(close_br):
                inner_content = match[1:-1].strip()
                if inner_content:
                    content.append(inner_content)

        return content

    def process_column(
        self,
        series: pd.Series,
        operation: str = 'remove',
        **kwargs
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Apply bracket handling to a pandas Series.

        Parameters
        ----------
        series : pd.Series
            Series to process.
        operation : str, default 'remove'
            Operation to perform: 'remove' or 'extract'.
        **kwargs
            Additional arguments for the operation.

        Returns
        -------
        Union[pd.Series, pd.DataFrame]
            Processed series or DataFrame with extracted content.
        """
        if operation == 'remove':
            return series.apply(lambda x: self.remove_content(x, **kwargs))
        elif operation == 'extract':
            extracted = series.apply(
                lambda x: self.extract_content(x, **kwargs))
            return pd.DataFrame(extracted.tolist(), index=series.index)
        else:
            raise ValueError(f"Unknown operation: {operation}")


# Convenience functions for easy usage
def normalize_text(
    text: Union[str, pd.Series],
    lowercase: bool = True,
    remove_html: bool = True,
    **kwargs
) -> Union[str, pd.Series]:
    """
    Quick text normalization with common settings.

    Parameters
    ----------
    text : Union[str, pd.Series]
        Text or Series to normalize.
    lowercase : bool, default True
        Convert to lowercase.
    remove_html : bool, default True
        Remove HTML content.
    **kwargs
        Additional arguments for TextNormalizer.

    Returns
    -------
    Union[str, pd.Series]
        Normalized text or Series.
    """
    normalizer = TextNormalizer(
        lowercase=lowercase,
        remove_html=remove_html,
        **kwargs
    )

    if isinstance(text, str):
        return normalizer.clean_text(text)
    else:
        return normalizer.normalize_column(text)


def clean_headers(headers: Union[List[str], pd.DataFrame], **kwargs) -> Union[List[str], pd.DataFrame]:
    """
    Clean column headers or DataFrame headers.

    Parameters
    ----------
    headers : Union[List[str], pd.DataFrame]
        Headers to clean or DataFrame with headers to clean.
    **kwargs
        Additional arguments for HeaderNormalizer.

    Returns
    -------
    Union[List[str], pd.DataFrame]
        Cleaned headers or DataFrame with cleaned headers.
    """
    normalizer = HeaderNormalizer(**kwargs)

    if isinstance(headers, list):
        return normalizer.normalize_headers(headers)
    else:
        return normalizer.normalize_dataframe_headers(headers)


def clean_web_data(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Clean web-scraped table data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with web data to clean.
    **kwargs
        Additional arguments for WebTableNormalizer.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    normalizer = WebTableNormalizer(**kwargs)
    return normalizer.normalize_dataframe(df)
