"""Code-based information extraction using custom functions."""

import logging
from typing import Any, Callable, Dict, Optional

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
    functions : Dict[str, Callable]
        Extraction functions mapping field names to callables.
        Functions should accept either:
        - str -> Any (for text-based extraction)  
        - pd.Series -> Any (for row-based extraction)
    vectorize : bool, optional
        Whether to vectorize string functions using pandas, by default True
    default_source : str, optional
        Default source column for text-based functions
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
    ...     'price': extract_price,
    ...     'category': extract_category
    ... }
    >>> extractor = CodeExtractor(functions, default_source="description")
    >>> result_df = extractor.extract(df)
    """

    def __init__(
        self,
        functions: Dict[str, Callable],
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
        self._analyze_functions()

    def _analyze_functions(self) -> None:
        """Analyze function signatures to determine how to call them."""
        import inspect

        self.function_info = {}

        for field_name, func in self.functions.items():
            try:
                sig = inspect.signature(func)
                params = list(sig.parameters.values())

                if len(params) == 0:
                    logger.warning(
                        f"Function '{field_name}' takes no parameters, skipping")
                    continue
                elif len(params) == 1:
                    param_name = params[0].name
                    # Try to infer if it expects text or row based on parameter name
                    if param_name.lower() in ['row', 'series', 'record']:
                        func_type = 'row'
                    else:
                        func_type = 'text'
                else:
                    # Multiple parameters - assume row-based
                    func_type = 'row'

                self.function_info[field_name] = {
                    'function': func,
                    'type': func_type,
                    'signature': str(sig)
                }

                logger.debug(
                    f"Function '{field_name}' classified as '{func_type}' with signature: {sig}")

            except Exception as e:
                logger.error(f"Failed to analyze function '{field_name}': {e}")
                continue

    def _apply_text_function(
        self,
        df: pd.DataFrame,
        field_name: str,
        func_info: Dict[str, Any],
        source_column: str
    ) -> pd.Series:
        """Apply a text-based function to a column."""
        func = func_info['function']

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
        # Validate input DataFrame
        try:
            validated_source = self._validate_input(df, source_column)
        except ValueError as e:
            logger.error(f"Input validation failed: {e}")
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
                    if not text_source:
                        logger.error(
                            f"No source column specified for text function '{field_name}'")
                        continue

                    if text_source not in result_df.columns:
                        logger.error(
                            f"Source column '{text_source}' not found for function '{field_name}'")
                        continue

                    extracted_series = self._apply_text_function(
                        result_df, field_name, func_info, text_source
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
                    'source_column': text_source if func_info['type'] == 'text' else 'row-based',
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

            # Write function info (signatures only)
            function_summary = {
                field_name: {
                    'type': info['type'],
                    'signature': info['signature']
                }
                for field_name, info in self.function_info.items()
            }
            self._write_artifact("functions_config.json", function_summary)

        return result_df
