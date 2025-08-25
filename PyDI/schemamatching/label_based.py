"""
Label-based schema matching using string similarity metrics.
"""

from __future__ import annotations

import logging
import re
from typing import Callable, List, Optional, Union

import pandas as pd
import textdistance

from .base import BaseSchemaMatcher, SchemaMapping


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
            Similarity function to use. Options: "jaccard", "levenshtein", 
            "jaro_winkler", "cosine", "overlap". Default is "jaccard".
        preprocess : callable, optional
            Function to preprocess column names before comparison.
        tokenize : bool, optional
            Whether to tokenize strings before comparison. Default is True.
        """
        self.similarity_function = similarity_function
        self.preprocess = preprocess
        self.tokenize = tokenize
        
        # Initialize similarity function
        if similarity_function == "jaccard":
            self._sim_func = textdistance.jaccard
        elif similarity_function == "levenshtein":
            self._sim_func = textdistance.levenshtein.normalized_similarity
        elif similarity_function == "jaro_winkler":
            self._sim_func = textdistance.jaro_winkler
        elif similarity_function == "cosine":
            self._sim_func = textdistance.cosine
        elif similarity_function == "overlap":
            self._sim_func = textdistance.overlap.normalized_similarity
        else:
            raise ValueError(f"Unsupported similarity function: {similarity_function}")
    
    def _prepare_string(self, text: str) -> Union[str, List[str]]:
        """Prepare string for similarity calculation."""
        if self.preprocess:
            text = self.preprocess(text)
        
        if self.tokenize:
            # Tokenize on non-alphabetic characters (including underscores, numbers, etc.)
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
        
        logging.info(f"Matching schemas: {source_name} -> {target_name}")
        
        for source_col in source_dataset.columns:
            for target_col in target_dataset.columns:
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