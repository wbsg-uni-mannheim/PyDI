"""Code-based information extraction using custom functions."""

import logging
from typing import Any, Callable, Dict, Optional, Union

import pandas as pd

from .base import BaseExtractor


logger = logging.getLogger(__name__)


class CodeExtractor(BaseExtractor):
    """Extract information using custom Python functions.

    This extractor applies user-defined functions to DataFrame columns or rows
    to extract structured information. Functions can operate on individual text
    values or entire DataFrame rows.

    Parameters
    ----------
    functions : Dict[str, Union[Callable, Dict[str, Any]]]
        Extraction functions mapping field names to either:
        - Callable: Function that accepts pd.Series (row-based extraction)
        - Dict with keys:
            - 'function': Callable that accepts str (text-based extraction)
            - 'source_column': str, column name to extract from
    vectorize : bool, optional
        Whether to vectorize string functions using pandas, by default True
    default_source : str, optional
        Default source column for text-based functions (deprecated, use explicit config)
    out_dir : str, optional
        Output directory for artifacts, by default "output/informationextraction"
    debug : bool, optional
        Enable debug mode, by default False

    Examples
    --------
    >>> def extract_price(text):
    ...     match = re.search(r'\\$([0-9,]+(?:\\.[0-9]{2})?)', text)
    ...     return float(match.group(1).replace(',', '')) if match else None
    ...
    >>> def extract_category(row):
    ...     if 'electronics' in row['description'].lower():
    ...         return 'Electronics'
    ...     return 'Other'
    ...
    >>> functions = {
    ...     'price': {'function': extract_price, 'source_column': 'description'},
    ...     'category': extract_category  # row-based function
    ... }
    >>> extractor = CodeExtractor(functions)
    >>> result_df = extractor.extract(df)
    """

    def __init__(
        self,
        functions: Dict[str, Union[Callable, Dict[str, Any]]],
        *,
        vectorize: bool = True,
        default_source: Optional[str] = None,
        out_dir: str = "output/informationextraction",
        debug: bool = False
    ):
        super().__init__(out_dir=out_dir, debug=debug)
        self.functions = functions
        self.vectorize = vectorize
        self.default_source = default_source
        self._process_function_configs()

    def _process_function_configs(self) -> None:
        """Process function configurations into normalized format."""
        self.function_info = {}

        for field_name, config in self.functions.items():
            try:
                if callable(config):
                    # Simple callable - assume row-based function
                    self.function_info[field_name] = {
                        'function': config,
                        'type': 'row',
                        'source_column': None
                    }
                elif isinstance(config, dict):
                    # Dictionary config with explicit settings
                    func = config.get('function')
                    source_column = config.get('source_column')
                    
                    if not callable(func):
                        logger.error(f"Function config '{field_name}' missing or invalid 'function' key")
                        continue
                    
                    self.function_info[field_name] = {
                        'function': func,
                        'type': 'text' if source_column else 'row',
                        'source_column': source_column
                    }
                else:
                    logger.error(f"Invalid function config for '{field_name}': expected callable or dict")
                    continue

                logger.debug(
                    f"Function '{field_name}' configured as '{self.function_info[field_name]['type']}' "
                    f"with source_column: {self.function_info[field_name]['source_column']}")

            except Exception as e:
                logger.error(f"Failed to process function config '{field_name}': {e}")
                continue

    def _apply_text_function(
        self,
        df: pd.DataFrame,
        field_name: str,
        func_info: Dict[str, Any]
    ) -> pd.Series:
        """Apply a text-based function to a column."""
        func = func_info['function']
        source_column = func_info['source_column']

        if self.vectorize:
            try:
                # Try vectorized application first
                return df[source_column].apply(func)
            except Exception as e:
                logger.warning(f"Vectorized application failed for '{field_name}': {e}. "
                               f"Falling back to element-wise application.")

        # Element-wise application with error handling
        results = []
        error_count = 0

        for i, value in enumerate(df[source_column]):
            try:
                result = func(value)
                results.append(result)
            except Exception as e:
                logger.debug(f"Function '{field_name}' failed on row {i}: {e}")
                results.append(None)
                error_count += 1

        if error_count > 0:
            logger.warning(
                f"Function '{field_name}' failed on {error_count}/{len(df)} rows")

        return pd.Series(results, index=df.index)

    def _apply_row_function(
        self,
        df: pd.DataFrame,
        field_name: str,
        func_info: Dict[str, Any]
    ) -> pd.Series:
        """Apply a row-based function to the DataFrame."""
        func = func_info['function']
        results = []
        error_count = 0

        for i, (_, row) in enumerate(df.iterrows()):
            try:
                result = func(row)
                results.append(result)
            except Exception as e:
                logger.debug(f"Function '{field_name}' failed on row {i}: {e}")
                results.append(None)
                error_count += 1

        if error_count > 0:
            logger.warning(
                f"Function '{field_name}' failed on {error_count}/{len(df)} rows")

        return pd.Series(results, index=df.index)

    def extract(
        self,
        df: pd.DataFrame,
        *,
        source_column: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Extract structured information using custom functions.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        source_column : str, optional
            Source column for text-based functions. If None, uses default_source

        Returns
        -------
        pd.DataFrame
            DataFrame with extracted columns added
        """
        # Validate input DataFrame (basic checks)
        if df.empty:
            logger.error("Input DataFrame is empty")
            return df

        # Determine source column for text-based functions
        text_source = source_column or self.default_source

        result_df = df.copy()
        extraction_stats = {}

        for field_name, func_info in self.function_info.items():
            logger.debug(
                f"Applying function '{field_name}' (type: {func_info['type']})")

            try:
                if func_info['type'] == 'text':
                    source_col = func_info['source_column'] or text_source
                    if not source_col:
                        logger.error(
                            f"No source column specified for text function '{field_name}'")
                        continue

                    if source_col not in result_df.columns:
                        logger.error(
                            f"Source column '{source_col}' not found for function '{field_name}'")
                        continue

                    # Update func_info with the resolved source column
                    func_info_copy = func_info.copy()
                    func_info_copy['source_column'] = source_col
                    
                    extracted_series = self._apply_text_function(
                        result_df, field_name, func_info_copy
                    )

                else:  # row-based function
                    extracted_series = self._apply_row_function(
                        result_df, field_name, func_info
                    )

                # Add extracted column
                result_df[field_name] = extracted_series

                # Calculate stats
                non_null_count = extracted_series.notna().sum()
                success_rate = non_null_count / \
                    len(result_df) if len(result_df) > 0 else 0.0

                extraction_stats[field_name] = {
                    'function_type': func_info['type'],
                    'source_column': func_info['source_column'] if func_info['type'] == 'text' else 'row-based',
                    'total_rows': len(result_df),
                    'successful_extractions': int(non_null_count),
                    'success_rate': success_rate
                }

                logger.info(f"Function '{field_name}': {non_null_count}/{len(result_df)} successful extractions "
                            f"({success_rate:.2%})")

            except Exception as e:
                logger.error(f"Failed to apply function '{field_name}': {e}")
                continue

        # Log overall stats
        extracted_columns = list(self.function_info.keys())
        self._log_extraction_stats(df, result_df, extracted_columns)

        if self.debug:
            # Write sample data
            sample_df = result_df.head(100)
            self._write_artifact("samples.csv", sample_df)

            # Write extraction stats
            self._write_artifact("function_stats.json", extraction_stats)

            # Write function info
            function_summary = {
                field_name: {
                    'type': info['type'],
                    'source_column': info['source_column']
                }
                for field_name, info in self.function_info.items()
            }
            self._write_artifact("functions_config.json", function_summary)

        return result_df
