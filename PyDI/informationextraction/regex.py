"""Regex-based information extraction."""

import logging
import re
from typing import Any, Callable, Dict, Optional, Union

import pandas as pd

from .base import BaseExtractor
from .rules import TRANSFORMATIONS


logger = logging.getLogger(__name__)


class RegexExtractor(BaseExtractor):
    """Extract information using regular expressions.

    This extractor applies regex patterns to text columns and extracts structured
    information based on configurable rules. Each rule specifies a field name,
    source column, pattern(s), and optional post-processing.

    Parameters
    ----------
    rules : Dict[str, Dict[str, Any]]
        Extraction rules. Each rule should have:
        - source_column : str, optional - Column to extract from
        - pattern : str or List[str] - Regex pattern(s) to apply  
        - flags : int, optional - Regex flags (default: 0)
        - group : int or tuple, optional - Regex group(s) to capture (default: 0)
        - postprocess : str or Callable, optional - Post-processing function
    default_source : str, optional
        Default source column if not specified in rules
    out_dir : str, optional
        Output directory for artifacts, by default "output/informationextraction"
    debug : bool, optional
        Enable debug mode, by default False

    Examples
    --------
    >>> rules = {
    ...     "price": {
    ...         "source_column": "description", 
    ...         "pattern": r"\\$([0-9,]+(?:\\.[0-9]{2})?)",
    ...         "group": 1,
    ...         "postprocess": "parse_money"
    ...     },
    ...     "email": {
    ...         "pattern": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}",
    ...         "postprocess": "lower"
    ...     }
    ... }
    >>> extractor = RegexExtractor(rules, default_source="text")
    >>> result_df = extractor.extract(df)
    """

    def __init__(
        self,
        rules: Dict[str, Dict[str, Any]],
        *,
        default_source: Optional[str] = None,
        out_dir: str = "output/informationextraction",
        debug: bool = False
    ):
        super().__init__(out_dir=out_dir, debug=debug)
        self.rules = rules
        self.default_source = default_source
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Pre-compile regex patterns for efficiency."""
        self.compiled_rules = {}

        for field_name, rule in self.rules.items():
            compiled_rule = rule.copy()

            patterns = rule.get('pattern', [])
            if isinstance(patterns, str):
                patterns = [patterns]

            flags = rule.get('flags', 0)
            compiled_patterns = []

            for pattern in patterns:
                try:
                    compiled_patterns.append(re.compile(pattern, flags))
                except re.error as e:
                    logger.error(
                        f"Invalid regex pattern for field '{field_name}': {pattern}. Error: {e}")
                    continue

            if compiled_patterns:
                compiled_rule['compiled_patterns'] = compiled_patterns
                self.compiled_rules[field_name] = compiled_rule
            else:
                logger.warning(
                    f"No valid patterns found for field '{field_name}', skipping")

    def _get_postprocessor(self, postprocess: Union[str, Callable, None]) -> Optional[Callable]:
        """Get post-processing function."""
        if postprocess is None:
            return None
        elif isinstance(postprocess, str):
            if postprocess in TRANSFORMATIONS:
                return TRANSFORMATIONS[postprocess]
            else:
                logger.warning(f"Unknown transformation: {postprocess}")
                return None
        elif callable(postprocess):
            return postprocess
        else:
            logger.warning(f"Invalid postprocessor type: {type(postprocess)}")
            return None

    def _extract_from_text(self, text: str, rule: Dict[str, Any]) -> Any:
        """Extract information from a single text string."""
        if not isinstance(text, str) or not text.strip():
            return None

        compiled_patterns = rule.get('compiled_patterns', [])
        group = rule.get('group', 0)
        postprocessor = self._get_postprocessor(rule.get('postprocess'))

        for pattern in compiled_patterns:
            match = pattern.search(text)
            if match:
                try:
                    if isinstance(group, tuple):
                        # Multiple groups - return tuple
                        extracted = tuple(match.group(g) for g in group)
                    else:
                        # Single group
                        extracted = match.group(group)

                    # Apply post-processing
                    if postprocessor:
                        if isinstance(extracted, tuple):
                            extracted = tuple(postprocessor(
                                x) if x else x for x in extracted)
                        else:
                            extracted = postprocessor(extracted)

                    return extracted

                except IndexError as e:
                    logger.warning(
                        f"Group index error for pattern {pattern.pattern}: {e}")
                    continue
                except Exception as e:
                    logger.warning(
                        f"Error processing match for pattern {pattern.pattern}: {e}")
                    continue

        return None

    def extract(
        self,
        df: pd.DataFrame,
        *,
        source_column: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Extract structured information using regex patterns.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        source_column : str, optional
            Column to extract from. If None, uses default_source or rule-specific sources

        Returns
        -------
        pd.DataFrame
            DataFrame with extracted columns added
        """
        # Validate input DataFrame (basic checks)
        if df.empty:
            logger.error("Input DataFrame is empty")
            return df

        result_df = df.copy()
        extraction_stats = {}

        for field_name, rule in self.compiled_rules.items():
            # Determine source column for this rule
            rule_source = rule.get(
                'source_column') or source_column or self.default_source

            if not rule_source:
                logger.error(
                    f"No source column specified for field '{field_name}'")
                continue

            if rule_source not in result_df.columns:
                logger.error(
                    f"Source column '{rule_source}' not found for field '{field_name}'")
                continue

            logger.debug(
                f"Extracting field '{field_name}' from column '{rule_source}'")

            # Apply extraction to each row
            extracted_values = []
            success_count = 0

            for text in result_df[rule_source]:
                extracted = self._extract_from_text(text, rule)
                extracted_values.append(extracted)
                if extracted is not None:
                    success_count += 1

            # Add extracted column
            result_df[field_name] = extracted_values

            # Track stats
            extraction_stats[field_name] = {
                'source_column': rule_source,
                'total_rows': len(result_df),
                'successful_extractions': success_count,
                'success_rate': success_count / len(result_df) if len(result_df) > 0 else 0.0
            }

            logger.info(f"Field '{field_name}': {success_count}/{len(result_df)} successful extractions "
                        f"({extraction_stats[field_name]['success_rate']:.2%})")

        # Log overall stats
        extracted_columns = list(self.compiled_rules.keys())
        self._log_extraction_stats(df, result_df, extracted_columns)

        if self.debug:
            # Write sample data
            sample_df = result_df.head(100)
            self._write_artifact("samples.csv", sample_df)

            # Write detailed extraction stats
            self._write_artifact("rule_stats.json", extraction_stats)

            # Write rule configuration
            serializable_rules = {}
            for field_name, rule in self.rules.items():
                serializable_rule = rule.copy()
                # Remove non-serializable compiled patterns
                serializable_rule.pop('compiled_patterns', None)
                serializable_rules[field_name] = serializable_rule
            self._write_artifact("rules_config.json", serializable_rules)

        return result_df
