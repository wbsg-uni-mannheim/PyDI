"""
Label-based schema matching using string similarity metrics.
"""

from __future__ import annotations

import logging
import re
from typing import Callable, List, Optional, Union

import pandas as pd

from .base import BaseSchemaMatcher, SchemaMapping, get_schema_columns
from ..utils import SimilarityRegistry


class LabelBasedSchemaMatcher(BaseSchemaMatcher):
    """Label-based schema matcher using string similarity metrics.
    
    This matcher compares column names using various string similarity
    functions including Jaccard, Levenshtein, and other metrics from
    textdistance. It supports preprocessing and configurable similarity
    thresholds.
    """
    
    def __init__(
        self,
        similarity_function: str = "jaccard",
        preprocess: Optional[Callable[[str], str]] = None,
        tokenize: bool = True,
    ) -> None:
        """Initialize the label-based schema matcher.
        
        Parameters
        ----------
        similarity_function : str, optional
            Similarity function to use. Any function from SimilarityRegistry.
            Recommended: "jaccard", "levenshtein", "jaro_winkler", "cosine", "overlap".
            Default is "jaccard".
        preprocess : callable, optional
            Function to preprocess column names before comparison.
        tokenize : bool, optional
            Whether to tokenize strings before comparison. Default is True.
        """
        self.similarity_function = similarity_function
        self.preprocess = preprocess
        self.tokenize = tokenize
        
        # Get similarity function from registry
        try:
            self._sim_func = SimilarityRegistry.get_function(similarity_function)
        except ValueError as e:
            available_funcs = SimilarityRegistry.get_recommended_functions("schema_matching", "label")
            raise ValueError(f"{e}. Recommended functions for label-based matching: {available_funcs}")
    
    def _prepare_string(self, text: str) -> Union[str, List[str]]:
        """Prepare string for similarity calculation."""
        if self.preprocess:
            text = self.preprocess(text)
        
        if self.tokenize:
            # Tokenize by extracting alphabetic sequences (i.e., contiguous letters)
            tokens = re.findall(r'[a-zA-Z]+', text.lower())
            return tokens
        return text
    
    def match(
        self,
        source_dataset: pd.DataFrame,
        target_dataset: pd.DataFrame,
        preprocess: Optional[Callable[[str], str]] = None,
        threshold: float = 0.8,
        method: str = "label",
        **kwargs,
    ) -> SchemaMapping:
        """Find schema correspondences using label-based matching.
        
        Parameters
        ----------
        source_dataset : pandas.DataFrame
            The source dataset.
        target_dataset : pandas.DataFrame
            The target dataset.
        preprocess : callable, optional
            Preprocessing function (overrides instance setting).
        threshold : float, optional
            Minimum similarity score for correspondences.
        method : str, optional
            Matching method. Only "label" is supported.
        **kwargs
            Additional keyword arguments (ignored).
            
        Returns
        -------
        SchemaMapping
            DataFrame with schema correspondences.
        """
        if method != "label":
            raise ValueError(f"Unsupported method '{method}'. Only 'label' is supported.")
        
        # Use instance preprocessing if not provided
        if preprocess is None:
            preprocess = self.preprocess
            
        results = []
        
        # Get dataset names
        source_name = source_dataset.attrs.get("dataset_name", "source")
        target_name = target_dataset.attrs.get("dataset_name", "target")
        
        # Get schema columns excluding PyDI-generated ID columns
        source_columns = get_schema_columns(source_dataset)
        target_columns = get_schema_columns(target_dataset)
        
        logging.info(f"Matching schemas: {source_name} -> {target_name}")
        logging.info(f"Source columns for matching: {source_columns}")
        logging.info(f"Target columns for matching: {target_columns}")
        
        for source_col in source_columns:
            for target_col in target_columns:
                # Prepare strings for comparison
                source_str = self._prepare_string(source_col)
                target_str = self._prepare_string(target_col)
                
                # Calculate similarity
                similarity = self._sim_func(source_str, target_str)
                
                if similarity >= threshold:
                    results.append({
                        "source_dataset": source_name,
                        "source_column": source_col,
                        "target_dataset": target_name,
                        "target_column": target_col,
                        "score": float(similarity),
                        "notes": f"similarity_function={self.similarity_function}"
                    })
                    
                    logging.debug(f"Match: {source_col} -> {target_col} ({similarity:.4f})")
        
        return pd.DataFrame(results)