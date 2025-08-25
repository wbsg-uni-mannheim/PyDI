"""
Instance-based schema matching using value distributions.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .base import BaseSchemaMatcher, SchemaMapping


class InstanceBasedSchemaMatcher(BaseSchemaMatcher):
    """Instance-based schema matcher using value distributions.
    
    This matcher analyzes the actual values in columns to determine
    correspondences. It creates vector representations of columns based
    on their value distributions and uses vector space similarity measures.
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
            Similarity function for vectors. Options: "cosine", "jaccard",
            "containment". Default is "cosine".
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
        
        if similarity_function not in ["cosine", "jaccard", "containment"]:
            raise ValueError(f"Unsupported similarity function: {similarity_function}")
    
    def _extract_column_values(self, df: pd.DataFrame, column: str) -> List[str]:
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
        """Calculate cosine similarity between two vectors."""
        if not vec1 or not vec2:
            return 0.0
        
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
        """Calculate Jaccard similarity between two vectors."""
        if not vec1 or not vec2:
            return 0.0
        
        terms1 = set(vec1.keys())
        terms2 = set(vec2.keys())
        
        intersection = len(terms1 & terms2)
        union = len(terms1 | terms2)
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_containment_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """Calculate containment similarity (maximum of containment in both directions)."""
        if not vec1 or not vec2:
            return 0.0
        
        terms1 = set(vec1.keys())
        terms2 = set(vec2.keys())
        
        if not terms1 or not terms2:
            return 0.0
        
        # Calculate containment in both directions
        containment_1_in_2 = len(terms1 & terms2) / len(terms1)
        containment_2_in_1 = len(terms1 & terms2) / len(terms2)
        
        return max(containment_1_in_2, containment_2_in_1)
    
    def match(
        self,
        datasets: List[pd.DataFrame],
        method: str = "label",
        preprocess: Optional[Any] = None,
        threshold: float = 0.8,
    ) -> SchemaMapping:
        """Find schema correspondences using instance-based matching.
        
        Parameters
        ----------
        datasets : list of pandas.DataFrame
            The datasets whose schemata should be matched.
        method : str, optional
            Matching method (not used, kept for compatibility).
        preprocess : Any, optional
            Not used in instance-based matching.
        threshold : float, optional
            Minimum similarity score for correspondences.
            
        Returns
        -------
        SchemaMapping
            DataFrame with schema correspondences.
        """
        results = []
        
        # Process datasets in pairs
        import itertools
        for i, j in itertools.combinations(range(len(datasets)), 2):
            df_i = datasets[i]
            df_j = datasets[j]
            name_i = df_i.attrs.get("dataset_name", f"ds{i}")
            name_j = df_j.attrs.get("dataset_name", f"ds{j}")
            
            logging.info(f"Instance-based matching: {name_i} <-> {name_j}")
            
            # Extract values for all columns in both datasets
            all_column_values = {}
            
            # Dataset i columns
            for col in df_i.columns:
                values = self._extract_column_values(df_i, col)
                non_null_ratio = len(values) / len(df_i) if len(df_i) > 0 else 0
                
                if non_null_ratio >= self.min_non_null_ratio:
                    all_column_values[f"{name_i}.{col}"] = values
            
            # Dataset j columns  
            for col in df_j.columns:
                values = self._extract_column_values(df_j, col)
                non_null_ratio = len(values) / len(df_j) if len(df_j) > 0 else 0
                
                if non_null_ratio >= self.min_non_null_ratio:
                    all_column_values[f"{name_j}.{col}"] = values
            
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
            for col_i in df_i.columns:
                key_i = f"{name_i}.{col_i}"
                if key_i not in vectors:
                    continue
                    
                for col_j in df_j.columns:
                    key_j = f"{name_j}.{col_j}"
                    if key_j not in vectors:
                        continue
                    
                    # Calculate similarity
                    if self.similarity_function == "cosine":
                        similarity = self._calculate_cosine_similarity(vectors[key_i], vectors[key_j])
                    elif self.similarity_function == "jaccard":
                        similarity = self._calculate_jaccard_similarity(vectors[key_i], vectors[key_j])
                    elif self.similarity_function == "containment":
                        similarity = self._calculate_containment_similarity(vectors[key_i], vectors[key_j])
                    
                    if similarity >= threshold:
                        results.append({
                            "source_dataset": name_i,
                            "source_column": col_i,
                            "target_dataset": name_j,
                            "target_column": col_j,
                            "score": float(similarity),
                            "notes": f"vector_method={self.vector_creation_method},similarity={self.similarity_function}"
                        })
                        
                        logging.debug(f"Instance match: {col_i} <-> {col_j} ({similarity:.4f})")
        
        return pd.DataFrame(results)