"""
Instance-based schema matching using value distributions.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Set, Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .base import BaseSchemaMatcher, SchemaMapping, get_schema_columns
from ..utils import SimilarityRegistry


class InstanceBasedSchemaMatcher(BaseSchemaMatcher):
    """Instance-based schema matcher using value distributions.
    
    This matcher analyzes the actual values in columns to determine
    correspondences. It creates vector representations of columns based
    on their value distributions and uses similarity measures from the
    textdistance package for improved performance and consistency.
    
    For binary vectors, textdistance functions are used directly.
    For weighted vectors (term frequencies, TF-IDF), custom implementations
    are retained for accuracy with weighted calculations.
    """
    
    def __init__(
        self,
        vector_creation_method: str = "term_frequencies",
        similarity_function: str = "cosine",
        max_sample_size: int = 1000,
        min_non_null_ratio: float = 0.1,
    ) -> None:
        """Initialize the instance-based schema matcher.
        
        Parameters
        ----------
        vector_creation_method : str, optional
            Method for creating vectors. Options: "term_frequencies", 
            "binary_occurrence", "tfidf". Default is "term_frequencies".
        similarity_function : str, optional
            Similarity function for vectors. Any function from SimilarityRegistry.
            Recommended: "cosine", "jaccard", "overlap". Default is "cosine".
        max_sample_size : int, optional
            Maximum number of values to sample from each column for analysis.
        min_non_null_ratio : float, optional
            Minimum ratio of non-null values required for matching.
        """
        self.vector_creation_method = vector_creation_method
        self.similarity_function = similarity_function
        self.max_sample_size = max_sample_size
        self.min_non_null_ratio = min_non_null_ratio
        
        if vector_creation_method not in ["term_frequencies", "binary_occurrence", "tfidf"]:
            raise ValueError(f"Unsupported vector creation method: {vector_creation_method}")
        
        # Validate similarity function exists in registry
        try:
            SimilarityRegistry.get_function(similarity_function)
        except ValueError as e:
            available_funcs = SimilarityRegistry.get_recommended_functions("schema_matching", "instance")
            raise ValueError(f"{e}. Recommended functions for instance-based matching: {available_funcs}")
    
    def _extract_column_values(self, df: pd.DataFrame, column: str, preprocess: Optional[Callable[[str], str]] = None) -> List[str]:
        """Extract and preprocess values from a column."""
        # Get non-null values
        values = df[column].dropna()
        
        # Sample if too many values
        if len(values) > self.max_sample_size:
            values = values.sample(n=self.max_sample_size, random_state=42)
        
        # Convert to strings and clean
        str_values = []
        for val in values:
            str_val = str(val).strip().lower()
            if str_val and str_val != "nan":
                # Apply preprocessing if provided
                if preprocess:
                    str_val = preprocess(str_val)
                str_values.append(str_val)
        
        return str_values
    
    def _create_term_frequency_vector(self, values: List[str]) -> Dict[str, float]:
        """Create term frequency vector from values."""
        if not values:
            return {}
        
        # Tokenize all values
        all_tokens = []
        for value in values:
            # Simple tokenization
            tokens = value.split()
            all_tokens.extend(tokens)
        
        # Count frequencies
        token_counts = Counter(all_tokens)
        total_tokens = len(all_tokens)
        
        # Normalize to frequencies
        return {token: count / total_tokens for token, count in token_counts.items()}
    
    def _create_binary_vector(self, values: List[str]) -> Dict[str, float]:
        """Create binary occurrence vector from values."""
        if not values:
            return {}
        
        # Get unique tokens across all values
        unique_tokens = set()
        for value in values:
            tokens = value.split()
            unique_tokens.update(tokens)
        
        return {token: 1.0 for token in unique_tokens}
    
    def _create_tfidf_vectors(self, all_column_values: Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
        """Create TF-IDF vectors for all columns."""
        if not all_column_values:
            return {}
        
        # Prepare documents (one per column)
        documents = []
        column_names = []
        
        for col_name, values in all_column_values.items():
            if values:
                # Join all values into one document
                doc = " ".join(values)
                documents.append(doc)
                column_names.append(col_name)
        
        if not documents:
            return {}
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(max_features=1000, stop_words=None)
        tfidf_matrix = vectorizer.fit_transform(documents)
        feature_names = vectorizer.get_feature_names_out()
        
        # Convert to dictionary format
        vectors = {}
        for i, col_name in enumerate(column_names):
            vector_values = tfidf_matrix[i].toarray().flatten()
            vectors[col_name] = {
                feature_names[j]: float(vector_values[j])
                for j in range(len(feature_names))
                if vector_values[j] > 0
            }
        
        return vectors
    
    def _calculate_cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """Calculate cosine similarity between two vectors.
        
        For binary vectors, uses textdistance.cosine for efficiency.
        For weighted vectors, uses custom implementation for accuracy.
        """
        if not vec1 or not vec2:
            return 0.0
        
        # For binary vectors (all values are 1.0), use similarity registry
        if all(val == 1.0 for val in vec1.values()) and all(val == 1.0 for val in vec2.values()):
            cosine_func = SimilarityRegistry.get_function("cosine")
            return float(cosine_func(vec1.keys(), vec2.keys()))
        
        # For weighted vectors, use original implementation
        # Get common terms
        common_terms = set(vec1.keys()) & set(vec2.keys())
        if not common_terms:
            return 0.0
        
        # Calculate dot product and magnitudes
        dot_product = sum(vec1[term] * vec2[term] for term in common_terms)
        
        magnitude1 = np.sqrt(sum(val ** 2 for val in vec1.values()))
        magnitude2 = np.sqrt(sum(val ** 2 for val in vec2.values()))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def _calculate_jaccard_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """Calculate Jaccard similarity between two vectors using textdistance."""
        if not vec1 or not vec2:
            return 0.0
        
        # Use similarity registry for Jaccard calculations
        jaccard_func = SimilarityRegistry.get_function("jaccard")
        return float(jaccard_func(vec1.keys(), vec2.keys()))
    
    def _calculate_containment_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """Calculate containment similarity using textdistance overlap function."""
        if not vec1 or not vec2:
            return 0.0
        
        # Use similarity registry overlap function
        overlap_func = SimilarityRegistry.get_function("overlap")
        return float(overlap_func(vec1.keys(), vec2.keys()))
    
    def match(
        self,
        source_dataset: pd.DataFrame,
        target_dataset: pd.DataFrame,
        preprocess: Optional[Callable[[str], str]] = None,
        threshold: float = 0.8,
        **kwargs,
    ) -> SchemaMapping:
        """Find schema correspondences using instance-based matching.
        
        Parameters
        ----------
        source_dataset : pandas.DataFrame
            The source dataset.
        target_dataset : pandas.DataFrame
            The target dataset.
        preprocess : callable, optional
            Function to preprocess string values before analysis.
        threshold : float, optional
            Minimum similarity score for correspondences.
        **kwargs
            Additional keyword arguments (ignored).
            
        Returns
        -------
        SchemaMapping
            DataFrame with schema correspondences.
        """
        results = []
        
        # Get dataset names
        source_name = source_dataset.attrs.get("dataset_name", "source")
        target_name = target_dataset.attrs.get("dataset_name", "target")
        
        # Get schema columns excluding PyDI-generated ID columns
        source_columns = get_schema_columns(source_dataset)
        target_columns = get_schema_columns(target_dataset)
        
        logging.info(f"Instance-based matching: {source_name} -> {target_name}")
        logging.info(f"Source columns for matching: {source_columns}")
        logging.info(f"Target columns for matching: {target_columns}")
        
        # Extract values for all columns in both datasets
        all_column_values = {}
        
        # Source dataset columns
        for col in source_columns:
            values = self._extract_column_values(source_dataset, col, preprocess)
            non_null_ratio = len(values) / len(source_dataset) if len(source_dataset) > 0 else 0
            
            if non_null_ratio >= self.min_non_null_ratio:
                all_column_values[f"{source_name}.{col}"] = values
        
        # Target dataset columns  
        for col in target_columns:
            values = self._extract_column_values(target_dataset, col, preprocess)
            non_null_ratio = len(values) / len(target_dataset) if len(target_dataset) > 0 else 0
            
            if non_null_ratio >= self.min_non_null_ratio:
                all_column_values[f"{target_name}.{col}"] = values
        
        # Create vectors based on method
        if self.vector_creation_method == "tfidf":
            vectors = self._create_tfidf_vectors(all_column_values)
        else:
            vectors = {}
            for col_key, values in all_column_values.items():
                if self.vector_creation_method == "term_frequencies":
                    vectors[col_key] = self._create_term_frequency_vector(values)
                elif self.vector_creation_method == "binary_occurrence":
                    vectors[col_key] = self._create_binary_vector(values)
        
        # Compare columns between datasets
        for source_col in source_columns:
            source_key = f"{source_name}.{source_col}"
            if source_key not in vectors:
                continue
                
            for target_col in target_columns:
                target_key = f"{target_name}.{target_col}"
                if target_key not in vectors:
                    continue
                
                # Calculate similarity
                if self.similarity_function == "cosine":
                    similarity = self._calculate_cosine_similarity(vectors[source_key], vectors[target_key])
                elif self.similarity_function == "jaccard":
                    similarity = self._calculate_jaccard_similarity(vectors[source_key], vectors[target_key])
                elif self.similarity_function == "overlap":
                    similarity = self._calculate_containment_similarity(vectors[source_key], vectors[target_key])
                else:
                    # For any other similarity function from registry
                    sim_func = SimilarityRegistry.get_function(self.similarity_function)
                    if all(val == 1.0 for val in vectors[source_key].values()) and all(val == 1.0 for val in vectors[target_key].values()):
                        # Binary vectors - use function directly on keys
                        similarity = float(sim_func(vectors[source_key].keys(), vectors[target_key].keys()))
                    else:
                        # Weighted vectors - fall back to cosine for non-supported functions
                        similarity = self._calculate_cosine_similarity(vectors[source_key], vectors[target_key])
                
                if similarity >= threshold:
                    results.append({
                        "source_dataset": source_name,
                        "source_column": source_col,
                        "target_dataset": target_name,
                        "target_column": target_col,
                        "score": float(similarity),
                        "notes": f"vector_method={self.vector_creation_method},similarity={self.similarity_function}"
                    })
                    
                    logging.debug(f"Instance match: {source_col} -> {target_col} ({similarity:.4f})")
        
        return pd.DataFrame(results)