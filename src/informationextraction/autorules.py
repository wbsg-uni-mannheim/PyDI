"""Auto-discovery of useful extraction rules based on built-in patterns and coverage."""

import json
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

import pandas as pd

from .base import BaseExtractor
from .regex import RegexExtractor
from .rules import built_in_rules, TRANSFORMATIONS


logger = logging.getLogger(__name__)


class RuleDiscovery(BaseExtractor):
    """Automatically discover useful extraction fields by applying built-in rules
    and filtering by coverage threshold.
    
    This class applies all available built-in regex rules to a text column,
    computes coverage for each extracted field, and returns only fields that
    meet minimum coverage thresholds. This is useful for exploratory data
    analysis when you don't know what structured information might be present
    in unstructured text.
    
    Parameters
    ----------
    out_dir : str, optional
        Output directory for debug artifacts, by default "output/informationextraction"
    debug : bool, optional
        Enable debug mode with verbose logging and artifacts, by default False
    
    Examples
    --------
    >>> from PyDI.informationextraction import RuleDiscovery
    >>> discovery = RuleDiscovery(debug=True)
    >>> df = pd.DataFrame({'text': ['Call us at (555) 123-4567', 'Visit http://example.com']})
    >>> result = discovery.extract_and_select(
    ...     df, 
    ...     source_column='text',
    ...     coverage_threshold=0.5
    ... )
    >>> print(result.columns.tolist())  # ['phone_us', 'url']
    """
    
    def __init__(self, *, out_dir: str = "output/informationextraction", debug: bool = False):
        super().__init__(out_dir=out_dir, debug=debug)
        self.default_source = None  # RuleDiscovery requires explicit source_column
    
    def build_all_rules(
        self, 
        source_column: str, 
        *, 
        categories: Optional[List[str]] = None,
        namespacing: str = "category__name",
        include_postprocess: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """Build regex rules dictionary from all built-in rules.
        
        Parameters
        ----------
        source_column : str
            Source column name to extract from
        categories : list of str, optional
            List of categories to include. If None, includes all categories
        namespacing : str, optional
            Field naming format, by default "category__name"
        include_postprocess : bool, optional
            Whether to include postprocessing functions, by default True
            
        Returns
        -------
        dict
            Rules dictionary compatible with RegexExtractor
        """
        if categories is None:
            categories = list(built_in_rules.keys())
        
        rules = {}
        
        for category in categories:
            if category not in built_in_rules:
                logger.warning(f"Category '{category}' not found in built_in_rules")
                continue
                
            for rule_name, rule_config in built_in_rules[category].items():
                # Create field name using namespacing
                if namespacing == "category__name":
                    field_name = f"{category}__{rule_name}"
                elif namespacing == "name_only":
                    field_name = rule_name
                else:
                    field_name = f"{category}_{rule_name}"
                
                # Build rule configuration
                rule_dict = {
                    "source_column": source_column,
                    "pattern": rule_config["pattern"],
                    "flags": rule_config.get("flags", 0),
                    "group": rule_config.get("group", 0)
                }
                
                # Add postprocessing if available and requested
                if include_postprocess and "postprocess" in rule_config:
                    postprocess_name = rule_config["postprocess"]
                    if postprocess_name in TRANSFORMATIONS:
                        rule_dict["postprocess"] = TRANSFORMATIONS[postprocess_name]
                
                rules[field_name] = rule_dict
        
        return rules
    
    def _compute_coverage(self, df: pd.DataFrame, extracted_fields: List[str]) -> Dict[str, float]:
        """Compute coverage statistics for extracted fields.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with extracted fields
        extracted_fields : list of str
            List of field names to compute coverage for
            
        Returns
        -------
        dict
            Coverage statistics {field_name: coverage_rate}
        """
        coverage = {}
        total_rows = len(df)
        
        for field in extracted_fields:
            if field in df.columns:
                non_null_count = df[field].notna().sum()
                coverage[field] = non_null_count / total_rows if total_rows > 0 else 0.0
            else:
                coverage[field] = 0.0
                
        return coverage
    
    def _filter_fields_by_coverage(
        self,
        coverage: Dict[str, float],
        *,
        coverage_threshold: float = 0.25,
        min_non_null: Optional[int] = None,
        top_k: Optional[int] = None
    ) -> List[str]:
        """Filter fields based on coverage criteria.
        
        Parameters
        ----------
        coverage : dict
            Coverage statistics {field_name: coverage_rate}
        coverage_threshold : float, optional
            Minimum coverage threshold (0.0-1.0), by default 0.25
        min_non_null : int, optional
            Minimum number of non-null values required, by default None
        top_k : int, optional
            Maximum number of fields to return (sorted by coverage), by default None
            
        Returns
        -------
        list of str
            List of selected field names
        """
        # Start with all fields meeting coverage threshold
        selected = [
            field for field, cov in coverage.items() 
            if cov >= coverage_threshold
        ]
        
        # Apply min_non_null filter if specified
        if min_non_null is not None:
            selected = [
                field for field in selected
                if coverage[field] * len(coverage) >= min_non_null  # Approximate non-null count
            ]
        
        # Sort by coverage (descending) and limit to top_k
        selected.sort(key=lambda x: coverage[x], reverse=True)
        if top_k is not None:
            selected = selected[:top_k]
        
        return selected
    
    def _save_debug_artifacts(
        self,
        df: pd.DataFrame,
        coverage: Dict[str, float],
        selected_fields: List[str],
        source_column: str
    ) -> None:
        """Save debug artifacts to disk.
        
        Parameters
        ----------
        df : pd.DataFrame
            Result DataFrame
        coverage : dict
            Coverage statistics
        selected_fields : list of str
            Selected field names
        source_column : str
            Source column name
        """
        if not self.debug:
            return
            
        # Save coverage statistics
        coverage_file = self.run_dir / "autorules_coverage.json"
        with open(coverage_file, 'w') as f:
            json.dump(coverage, f, indent=2)
        logger.info(f"Saved coverage statistics to {coverage_file}")
        
        # Save selected fields
        selected_file = self.run_dir / "autorules_selected_fields.json"
        with open(selected_file, 'w') as f:
            json.dump({
                "source_column": source_column,
                "selected_fields": selected_fields,
                "selection_criteria": {
                    "total_fields_evaluated": len(coverage),
                    "fields_selected": len(selected_fields)
                }
            }, f, indent=2)
        logger.info(f"Saved field selection to {selected_file}")
        
        # Save sample of results
        if not df.empty:
            sample_size = min(100, len(df))
            sample_df = df.sample(n=sample_size) if len(df) > sample_size else df
            sample_file = self.run_dir / "autorules_samples.csv"
            sample_df.to_csv(sample_file, index=False)
            logger.info(f"Saved sample results to {sample_file}")
    
    def extract_and_select(
        self,
        df: pd.DataFrame,
        *,
        source_column: str,
        categories: Optional[List[str]] = None,
        coverage_threshold: float = 0.25,
        top_k: Optional[int] = None,
        min_non_null: Optional[int] = None,
        include_original: bool = False,
        return_meta: bool = False
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]:
        """Extract fields using built-in rules and select by coverage.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        source_column : str
            Column to extract from
        categories : list of str, optional
            List of rule categories to apply. If None, applies all categories
        coverage_threshold : float, optional
            Minimum coverage threshold (0.0-1.0), by default 0.25
        top_k : int, optional
            Maximum number of fields to return (sorted by coverage), by default None
        min_non_null : int, optional
            Minimum number of non-null values required, by default None
        include_original : bool, optional
            Whether to include original DataFrame columns, by default False
        return_meta : bool, optional
            Whether to return metadata about extraction, by default False
            
        Returns
        -------
        pd.DataFrame or tuple
            DataFrame with selected extracted fields.
            If return_meta=True, returns (DataFrame, metadata_dict)
        """
        # Validate input
        source_col = self._validate_input(df, source_column)
        
        if self.debug:
            logger.info(f"Starting rule discovery on column '{source_col}' with {len(df)} rows")
            logger.info(f"Categories: {categories or 'all'}")
            logger.info(f"Coverage threshold: {coverage_threshold}")
        
        # Build all rules
        all_rules = self.build_all_rules(
            source_col,
            categories=categories,
            include_postprocess=True
        )
        
        if self.debug:
            logger.info(f"Built {len(all_rules)} rules from {len(categories or built_in_rules)} categories")
        
        # Apply regex extraction
        extractor = RegexExtractor(
            all_rules,
            default_source=source_col,
            out_dir=str(self.out_dir),
            debug=self.debug
        )
        
        extracted_df = extractor.extract(df, source_column=source_col)
        
        # Get list of newly extracted fields (exclude original columns)
        original_columns = set(df.columns)
        extracted_fields = [col for col in extracted_df.columns if col not in original_columns]
        
        if self.debug:
            logger.info(f"Extracted {len(extracted_fields)} fields from regex patterns")
        
        # Compute coverage
        coverage = self._compute_coverage(extracted_df, extracted_fields)
        
        # Select fields by coverage
        selected_fields = self._filter_fields_by_coverage(
            coverage,
            coverage_threshold=coverage_threshold,
            min_non_null=min_non_null,
            top_k=top_k
        )
        
        if self.debug:
            logger.info(f"Selected {len(selected_fields)} fields meeting criteria")
            if selected_fields:
                logger.info("Selected fields with coverage:")
                for field in selected_fields:
                    logger.info(f"  {field}: {coverage[field]:.3f}")
        
        # Build result DataFrame
        if include_original:
            result_columns = list(df.columns) + selected_fields
            result_df = extracted_df[result_columns]
        else:
            result_df = extracted_df[selected_fields] if selected_fields else pd.DataFrame()
        
        # Save debug artifacts
        self._save_debug_artifacts(result_df, coverage, selected_fields, source_col)
        
        # Return results
        if return_meta:
            metadata = {
                "coverage": coverage,
                "selected_fields": selected_fields,
                "total_fields_evaluated": len(extracted_fields),
                "source_column": source_col,
                "categories_used": categories or list(built_in_rules.keys())
            }
            return result_df, metadata
        
        return result_df
    
    def extract(
        self, 
        df: pd.DataFrame, 
        *, 
        source_column: Optional[str] = None, 
        **kwargs
    ) -> pd.DataFrame:
        """Extract method required by BaseExtractor interface.
        
        This is a simplified interface that calls extract_and_select with
        default parameters. For more control, use extract_and_select directly.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        source_column : str, optional
            Column to extract from
        **kwargs
            Additional parameters passed to extract_and_select
            
        Returns
        -------
        pd.DataFrame
            DataFrame with extracted and selected fields
        """
        if source_column is None:
            raise ValueError("source_column must be specified for RuleDiscovery")
        
        return self.extract_and_select(df, source_column=source_column, **kwargs)


def discover_fields(
    df: pd.DataFrame,
    *,
    source_column: str,
    categories: Optional[List[str]] = None,
    coverage_threshold: float = 0.25,
    top_k: Optional[int] = None,
    min_non_null: Optional[int] = None,
    include_original: bool = False,
    out_dir: str = "output/informationextraction",
    debug: bool = False
) -> pd.DataFrame:
    """Convenience function for field discovery using built-in rules.
    
    This is a functional interface to RuleDiscovery that creates an extractor
    instance and runs field discovery in a single call.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    source_column : str
        Column to extract from
    categories : list of str, optional
        List of rule categories to apply. If None, applies all categories
    coverage_threshold : float, optional
        Minimum coverage threshold (0.0-1.0), by default 0.25
    top_k : int, optional
        Maximum number of fields to return (sorted by coverage), by default None
    min_non_null : int, optional
        Minimum number of non-null values required, by default None
    include_original : bool, optional
        Whether to include original DataFrame columns, by default False
    out_dir : str, optional
        Output directory for debug artifacts, by default "output/informationextraction"
    debug : bool, optional
        Enable debug mode, by default False
        
    Returns
    -------
    pd.DataFrame
        DataFrame with selected extracted fields
        
    Examples
    --------
    >>> import pandas as pd
    >>> from PyDI.informationextraction import discover_fields
    >>> df = pd.DataFrame({
    ...     'text': [
    ...         'Contact us at john@example.com or call (555) 123-4567',
    ...         'Visit our website at https://example.com for $99.99 deals',
    ...         'Free shipping on orders over €50.00'
    ...     ]
    ... })
    >>> result = discover_fields(
    ...     df,
    ...     source_column='text',
    ...     categories=['contact', 'money'],
    ...     coverage_threshold=0.3,
    ...     include_original=True
    ... )
    >>> print(result.columns.tolist())
    """
    discovery = RuleDiscovery(out_dir=out_dir, debug=debug)
    
    return discovery.extract_and_select(
        df,
        source_column=source_column,
        categories=categories,
        coverage_threshold=coverage_threshold,
        top_k=top_k,
        min_non_null=min_non_null,
        include_original=include_original
    )