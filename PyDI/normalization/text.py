"""
Comprehensive text normalization utilities for PyDI.

This module provides all text-related normalization functionality, from basic
text cleaning to advanced tokenization and web table processing. It includes
both simple and sophisticated text processing capabilities.
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

# Try to import optional dependencies for advanced features
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    NLTK_AVAILABLE = True

    # Download required NLTK data if not present
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        logger.info("Downloading required NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)

except ImportError:
    NLTK_AVAILABLE = False
    logger.warning(
        "NLTK not available. Advanced tokenization features will be limited.")


class TextNormalizer:
    """
    Comprehensive text cleaning and normalization.

    Provides both basic and advanced text cleaning capabilities including
    HTML removal, Unicode normalization, case conversion, and whitespace handling.

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
        Fix common encoding issues.
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


class TokenizationNormalizer:
    """
    Advanced tokenization and text normalization with stemming support.

    Provides sophisticated text processing including tokenization, stemming,
    stop word removal, and case change splitting.

    Parameters
    ----------
    use_stemming : bool, default False
        Whether to apply stemming to tokens.
    remove_stopwords : bool, default True
        Whether to remove stop words.
    min_token_length : int, default 1
        Minimum token length to keep.
    split_on_case_change : bool, default True
        Split tokens on case changes (camelCase).
    split_numbers : bool, default True
        Split alphanumeric tokens.
    language : str, default 'english'
        Language for stop words and stemming.
    """

    def __init__(
        self,
        use_stemming: bool = False,
        remove_stopwords: bool = True,
        min_token_length: int = 1,
        split_on_case_change: bool = True,
        split_numbers: bool = True,
        language: str = 'english'
    ):
        self.use_stemming = use_stemming
        self.remove_stopwords = remove_stopwords
        self.min_token_length = min_token_length
        self.split_on_case_change = split_on_case_change
        self.split_numbers = split_numbers
        self.language = language

        # Initialize NLTK components if available
        if NLTK_AVAILABLE:
            try:
                self.stemmer = PorterStemmer() if use_stemming else None
                self.stop_words = set(stopwords.words(
                    language)) if remove_stopwords else set()
            except OSError:
                logger.warning(
                    f"Stop words for '{language}' not available. Disabling stop word removal.")
                self.stop_words = set()
                self.stemmer = None
        else:
            self.stemmer = None
            # Fallback English stop words
            self.stop_words = {
                'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
                'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
                'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
                'had', 'what', 'said', 'each', 'which', 'their', 'we', 'all'
            } if remove_stopwords else set()

        # Tokenization patterns
        self.case_change_pattern = re.compile(r'(?<=[a-z])(?=[A-Z])')
        self.alphanumeric_split_pattern = re.compile(
            r'(?<=[a-zA-Z])(?=\d)|(?<=\d)(?=[a-zA-Z])')
        self.word_boundary_pattern = re.compile(r'\W+')

    def _split_camel_case(self, token: str) -> List[str]:
        """Split camelCase tokens."""
        if self.split_on_case_change:
            return self.case_change_pattern.sub(' ', token).split()
        return [token]

    def _split_alphanumeric(self, token: str) -> List[str]:
        """Split alphanumeric tokens."""
        if self.split_numbers:
            return self.alphanumeric_split_pattern.sub(' ', token).split()
        return [token]

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text with advanced processing.

        Parameters
        ----------
        text : str
            Text to tokenize.

        Returns
        -------
        List[str]
            List of processed tokens.
        """
        if pd.isna(text) or not text:
            return []

        text = str(text)

        # Remove brackets but keep content
        text = text.replace('(', ' ').replace(')', ' ')

        # Basic tokenization on word boundaries
        initial_tokens = self.word_boundary_pattern.split(text.strip())
        initial_tokens = [t for t in initial_tokens if t]

        # Advanced token splitting
        tokens = []
        for token in initial_tokens:
            if not token:
                continue

            # Split on case changes
            case_split_tokens = self._split_camel_case(token)

            # Split alphanumeric
            for case_token in case_split_tokens:
                alpha_split_tokens = self._split_alphanumeric(case_token)
                tokens.extend(alpha_split_tokens)

        # Process tokens
        processed_tokens = []
        for token in tokens:
            if len(token) < self.min_token_length:
                continue

            # Convert to lowercase
            token = token.lower()

            # Remove stop words
            if self.remove_stopwords and token in self.stop_words:
                continue

            # Apply stemming
            if self.stemmer and self.use_stemming:
                try:
                    token = self.stemmer.stem(token)
                except:
                    pass  # Keep original token if stemming fails

            processed_tokens.append(token)

        return processed_tokens

    def normalize(self, text: str) -> str:
        """
        Tokenize text and rejoin with spaces.

        Parameters
        ----------
        text : str
            Text to normalize.

        Returns
        -------
        str
            Normalized text with tokens rejoined.
        """
        tokens = self.tokenize(text)
        return ' '.join(tokens)

    def normalize_column(self, series: pd.Series) -> pd.Series:
        """
        Apply tokenization normalization to a pandas Series.

        Parameters
        ----------
        series : pd.Series
            Series to normalize.

        Returns
        -------
        pd.Series
            Series with normalized text.
        """
        return series.apply(self.normalize)


class WebTableNormalizer:
    """
    Specialized normalizer for web-scraped table data.

    Combines functionality for cleaning messy web data with extensive
    HTML entity handling and null value detection.

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

        # HTML entity patterns
        self.html_entities = {
            '&nbsp;': ' ', '&nbsp': ' ', 'nbsp': ' ',
            '&amp;': '&', '&lt;': '<', '&gt;': '>',
            '&quot;': '"', '&apos;': "'", '&ndash;': '-',
            '&mdash;': '-', '&hellip;': '...', '&copy;': '(c)',
            '&reg;': '(r)', '&trade;': 'tm', '&cent;': 'c',
            '&pound;': 'GBP', '&yen;': 'JPY', '&euro;': 'EUR'
        }

        # Compiled patterns
        self.bracket_pattern = re.compile(r'\(.*?\)')
        self.html_tag_pattern = re.compile(r'<.*?>')
        self.html_entity_pattern = re.compile(r'&[#\w]+;')
        self.numeric_entity_pattern = re.compile(r'[&\\?]#[0-9]{1,3};')

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
            value = str(value)

            # Remove newlines and normalize whitespace
            value = value.replace('\n', ' ').replace(
                '\r', ' ').replace('\t', ' ')

            # Handle HTML entities
            if self.handle_html_entities:
                # Replace known entities
                for entity, replacement in self.html_entities.items():
                    value = value.replace(entity, replacement)

                # Remove numeric HTML entities
                value = self.numeric_entity_pattern.sub(' ', value)

                # Final HTML entity decode
                value = html.unescape(value)

            # Remove HTML tags
            value = self.html_tag_pattern.sub('', value)

            # Convert to lowercase and trim
            value = value.lower().strip()

            # Check for null patterns
            if value in self.null_patterns:
                return self.null_value

            # Remove bracket content if requested
            if self.remove_brackets_content:
                value = self.bracket_pattern.sub('', value).strip()

            # Final whitespace normalization
            value = re.sub(r'\s+', ' ', value).strip()

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

        # Normalize headers if requested
        if normalize_headers:
            header_normalizer = HeaderNormalizer(null_value=self.null_value)
            result = header_normalizer.normalize_dataframe_headers(result)

        # Normalize values
        target_columns = columns if columns else result.columns.tolist()

        for col in target_columns:
            if col in result.columns:
                logger.info(f"Normalizing web table column: {col}")
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


def tokenize_text(text: str, use_stemming: bool = False, **kwargs) -> List[str]:
    """
    Tokenize text with optional stemming.

    Parameters
    ----------
    text : str
        Text to tokenize.
    use_stemming : bool, default False
        Whether to apply stemming.
    **kwargs
        Additional arguments for TokenizationNormalizer.

    Returns
    -------
    List[str]
        List of tokens.
    """
    normalizer = TokenizationNormalizer(use_stemming=use_stemming, **kwargs)
    return normalizer.tokenize(text)


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
