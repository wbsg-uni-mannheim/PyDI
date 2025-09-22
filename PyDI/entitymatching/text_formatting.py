"""
Text formatting utilities for transformer-based entity matching.

This module provides text formatting capabilities to convert entity pairs
into formatted text strings suitable for pre-trained language models.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Union

import pandas as pd


class TextFormatter:
    """Text formatter for converting entity pairs to formatted strings.

    This class handles the conversion of entity pairs into formatted text
    strings suitable for transformer-based entity matching. It supports
    configurable text templates, field selection, and handles missing values.

    Parameters
    ----------
    text_fields : list of str
        Column names to include in the formatted text.
    template : str, optional
        Template string for formatting entity pairs. Use {left} and {right}
        as placeholders for left and right entity text. Default uses [SEP] token.
    single_template : str, optional
        Template string for formatting individual entities. Use field names
        as placeholders (e.g., "{title} ({year})"). If None, uses space-separated values.
    max_length : int, optional
        Maximum length for formatted text. Default is 512.
    handle_missing : str, optional
        How to handle missing values: 'skip', 'empty', or 'placeholder'.
        Default is 'skip'.
    missing_placeholder : str, optional
        Placeholder text for missing values when handle_missing='placeholder'.
        Default is '[MISSING]'.

    Examples
    --------
    >>> formatter = TextFormatter(
    ...     text_fields=['title', 'year', 'director'],
    ...     template="{left} [SEP] {right}",
    ...     single_template="{title} ({year}) by {director}"
    ... )
    >>> texts = formatter.format_pairs(df_left, df_right, pairs, 'id')
    """

    def __init__(
        self,
        text_fields: List[str],
        template: str = "{left} [SEP] {right}",
        single_template: Optional[str] = None,
        max_length: int = 512,
        handle_missing: str = "skip",
        missing_placeholder: str = "[MISSING]",
    ):
        """Initialize text formatter.

        Parameters
        ----------
        text_fields : list of str
            Column names to include in the formatted text.
        template : str, optional
            Template for entity pairs. Default is "{left} [SEP] {right}".
        single_template : str, optional
            Template for individual entities. If None, uses space-separated values.
        max_length : int, optional
            Maximum text length. Default is 512.
        handle_missing : str, optional
            How to handle missing values. Default is 'skip'.
        missing_placeholder : str, optional
            Placeholder for missing values. Default is '[MISSING]'.
        """
        if not text_fields:
            raise ValueError("text_fields cannot be empty")

        if handle_missing not in ["skip", "empty", "placeholder"]:
            raise ValueError("handle_missing must be 'skip', 'empty', or 'placeholder'")

        self.text_fields = text_fields
        self.template = template
        self.single_template = single_template
        self.max_length = max_length
        self.handle_missing = handle_missing
        self.missing_placeholder = missing_placeholder

        # Setup logger
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")

    def format_pairs(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        pairs: pd.DataFrame,
        id_column: str,
    ) -> List[str]:
        """Format entity pairs into text strings.

        Parameters
        ----------
        df_left : pandas.DataFrame
            Left dataset with entity records.
        df_right : pandas.DataFrame
            Right dataset with entity records.
        pairs : pandas.DataFrame
            Entity pairs with id1, id2 columns.
        id_column : str
            Name of the ID column in the datasets.

        Returns
        -------
        list of str
            Formatted text strings for each entity pair.
        """
        if pairs.empty:
            return []

        # Validate required columns
        if "id1" not in pairs.columns or "id2" not in pairs.columns:
            raise ValueError("pairs must contain 'id1' and 'id2' columns")

        # Check if text fields exist in datasets
        missing_left = [f for f in self.text_fields if f not in df_left.columns]
        missing_right = [f for f in self.text_fields if f not in df_right.columns]

        if missing_left:
            self.logger.warning(f"Missing fields in left dataset: {missing_left}")
        if missing_right:
            self.logger.warning(f"Missing fields in right dataset: {missing_right}")

        formatted_texts = []

        for _, pair in pairs.iterrows():
            id1, id2 = pair["id1"], pair["id2"]

            # Get entity records
            left_record = df_left[df_left[id_column] == id1]
            right_record = df_right[df_right[id_column] == id2]

            if left_record.empty or right_record.empty:
                self.logger.warning(f"Missing records for pair ({id1}, {id2})")
                continue

            # Format individual entity texts
            left_text = self._format_single_entity(left_record.iloc[0])
            right_text = self._format_single_entity(right_record.iloc[0])

            # Apply pair template
            formatted_text = self.template.format(left=left_text, right=right_text)

            # Truncate if necessary
            if len(formatted_text) > self.max_length:
                formatted_text = formatted_text[:self.max_length]

            formatted_texts.append(formatted_text)

        self.logger.info(f"Formatted {len(formatted_texts)} entity pairs")
        return formatted_texts

    def format_single_entities(
        self,
        df: pd.DataFrame,
        entity_ids: List[Any],
        id_column: str,
    ) -> List[str]:
        """Format individual entities into text strings.

        This method is useful for embedding-based approaches where individual
        entity representations are needed.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataset containing entity records.
        entity_ids : list
            List of entity IDs to format.
        id_column : str
            Name of the ID column in the dataset.

        Returns
        -------
        list of str
            Formatted text strings for each entity.
        """
        if not entity_ids:
            return []

        # Check if text fields exist in dataset
        missing_fields = [f for f in self.text_fields if f not in df.columns]
        if missing_fields:
            self.logger.warning(f"Missing fields in dataset: {missing_fields}")

        formatted_texts = []

        for entity_id in entity_ids:
            record = df[df[id_column] == entity_id]

            if record.empty:
                self.logger.warning(f"Missing record for entity {entity_id}")
                continue

            formatted_text = self._format_single_entity(record.iloc[0])

            # Truncate if necessary
            if len(formatted_text) > self.max_length:
                formatted_text = formatted_text[:self.max_length]

            formatted_texts.append(formatted_text)

        self.logger.info(f"Formatted {len(formatted_texts)} individual entities")
        return formatted_texts

    def _format_single_entity(self, record: pd.Series) -> str:
        """Format a single entity record into text.

        Parameters
        ----------
        record : pandas.Series
            Entity record with field values.

        Returns
        -------
        str
            Formatted text for the entity.
        """
        if self.single_template:
            # Use custom template
            format_dict = {}
            for field in self.text_fields:
                if field in record.index:
                    value = record[field]
                    if pd.isna(value):
                        format_dict[field] = self._handle_missing_value()
                    else:
                        format_dict[field] = str(value)
                else:
                    format_dict[field] = self._handle_missing_value()

            try:
                return self.single_template.format(**format_dict)
            except KeyError as e:
                self.logger.warning(f"Template formatting error: {e}")
                # Fall back to space-separated format
                return self._format_space_separated(record)
        else:
            # Use space-separated format
            return self._format_space_separated(record)

    def _format_space_separated(self, record: pd.Series) -> str:
        """Format entity as space-separated field values.

        Parameters
        ----------
        record : pandas.Series
            Entity record.

        Returns
        -------
        str
            Space-separated field values.
        """
        values = []
        for field in self.text_fields:
            if field in record.index:
                value = record[field]
                if pd.isna(value):
                    if self.handle_missing != "skip":
                        values.append(self._handle_missing_value())
                else:
                    values.append(str(value))
            else:
                if self.handle_missing != "skip":
                    values.append(self._handle_missing_value())

        return " ".join(values)

    def _handle_missing_value(self) -> str:
        """Handle missing values based on configuration.

        Returns
        -------
        str
            Replacement value for missing data.
        """
        if self.handle_missing == "empty":
            return ""
        elif self.handle_missing == "placeholder":
            return self.missing_placeholder
        else:  # skip
            return ""

    def get_feature_names(self) -> List[str]:
        """Get feature names for compatibility with other extractors.

        Returns
        -------
        list of str
            List containing single feature name for text input.
        """
        return ["formatted_text"]

    def __repr__(self) -> str:
        return (
            f"TextFormatter(text_fields={self.text_fields}, "
            f"template='{self.template}', max_length={self.max_length})"
        )