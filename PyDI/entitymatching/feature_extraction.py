"""
Feature extraction for machine learning-based entity matching.

This module provides tools to convert entity pairs and labels into 
scikit-learn ready DataFrames using various feature extraction methods
including similarity comparators and vector-based approaches.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd

from .base import BaseComparator


class FeatureExtractor:
    """Extract features from entity pairs for machine learning.
    
    This class converts entity pairs into feature vectors suitable for 
    scikit-learn classifiers. It supports both traditional similarity-based
    features (using comparators) and vector-based features (embeddings).
    
    Parameters
    ----------
    feature_functions : List[Union[BaseComparator, callable, dict]]
        List of feature extraction functions. Can be:
        - BaseComparator instances
        - Callable functions that take (record1, record2) -> float
        - Dicts with 'function' and 'name' keys for custom naming
        
    Examples
    --------
    >>> from PyDI.entitymatching import FeatureExtractor
    >>> from PyDI.entitymatching.comparators import StringComparator, NumericComparator
    >>> 
    >>> extractor = FeatureExtractor([
    ...     StringComparator("title", "jaro_winkler"),
    ...     NumericComparator("year", "absolute_difference"),
    ...     lambda r1, r2: jaccard_similarity(r1["description"], r2["description"])
    ... ])
    >>> 
    >>> features = extractor.create_features(df_left, df_right, pairs, labels)
    """
    
    def __init__(
        self, 
        feature_functions: List[Union[BaseComparator, Callable, Dict[str, Any]]]
    ):
        if not feature_functions:
            raise ValueError("At least one feature function must be provided")
        
        self.feature_functions = self._parse_feature_functions(feature_functions)
        
    def _parse_feature_functions(
        self, 
        feature_functions: List[Union[BaseComparator, Callable, Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Parse and normalize feature functions."""
        parsed = []
        
        for i, func in enumerate(feature_functions):
            if isinstance(func, dict):
                if "function" not in func:
                    raise ValueError(f"Feature function dict at index {i} must have 'function' key")
                
                parsed.append({
                    "function": func["function"],
                    "name": func.get("name", f"feature_{i}")
                })
            elif isinstance(func, BaseComparator):
                parsed.append({
                    "function": func.compare,
                    "name": func.name
                })
            elif callable(func):
                # Try to get a meaningful name from the function
                if hasattr(func, '__name__'):
                    name = func.__name__
                elif hasattr(func, 'name'):
                    name = func.name
                else:
                    name = f"feature_{i}"
                    
                parsed.append({
                    "function": func,
                    "name": name
                })
            else:
                raise ValueError(f"Feature function at index {i} must be callable, BaseComparator, or dict")
        
        return parsed

    def create_features(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        pairs: pd.DataFrame,
        id_column: str,
        labels: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """Create feature matrix from entity pairs.

        Parameters
        ----------
        df_left : pandas.DataFrame
            Left dataset with specified ID column.
        df_right : pandas.DataFrame
            Right dataset with specified ID column.
        pairs : pandas.DataFrame
            DataFrame with id1, id2 columns representing entity pairs.
        id_column : str
            Name of the column containing record identifiers.
        labels : pandas.Series, optional
            Binary labels for pairs (1 for match, 0 for non-match).
            If provided, adds 'label' column to output.
            
        Returns
        -------
        pandas.DataFrame
            Feature matrix with columns:
            - One column per feature function (named using function names)
            - 'label' column (if labels provided)
            - 'id1', 'id2' columns for reference
            
        Raises
        ------
        ValueError
            If required columns are missing or data shapes don't match.
        """
        # Input validation
        if pairs.empty:
            raise ValueError("Empty pairs DataFrame provided")
            
        required_cols = ["id1", "id2"]
        for col in required_cols:
            if col not in pairs.columns:
                raise ValueError(f"Pairs DataFrame missing required column: {col}")
                
        if labels is not None and len(labels) != len(pairs):
            raise ValueError(f"Labels length ({len(labels)}) doesn't match pairs length ({len(pairs)})")
        
        # Validate that ID column exists
        if id_column not in df_left.columns:
            raise ValueError(f"Left dataset missing ID column: {id_column}")
        if id_column not in df_right.columns:
            raise ValueError(f"Right dataset missing ID column: {id_column}")

        # Create lookup dictionaries for fast record access
        left_lookup = df_left.set_index(id_column)
        right_lookup = df_right.set_index(id_column)
        
        logging.info(f"Extracting features from {len(pairs)} pairs using {len(self.feature_functions)} functions")
        
        # Initialize feature matrix
        feature_data = []
        
        # Process each pair
        for idx, pair in pairs.iterrows():
            id1, id2 = pair["id1"], pair["id2"]
            
            try:
                record1 = left_lookup.loc[id1]
                record2 = right_lookup.loc[id2]
                
                # Handle case where .loc returns DataFrame due to duplicate indices
                if isinstance(record1, pd.DataFrame):
                    record1 = record1.iloc[0]
                if isinstance(record2, pd.DataFrame):
                    record2 = record2.iloc[0]
                    
            except KeyError as e:
                logging.warning(f"Record not found: {e}")
                continue
                
            # Extract features for this pair
            pair_features = {"id1": id1, "id2": id2}
            
            for func_info in self.feature_functions:
                func = func_info["function"]
                name = func_info["name"]
                
                try:
                    feature_value = func(record1, record2)
                    # Ensure numeric value
                    if not isinstance(feature_value, (int, float, np.number)):
                        logging.warning(f"Non-numeric feature value from {name}: {feature_value}")
                        feature_value = 0.0
                    pair_features[name] = float(feature_value)
                except Exception as e:
                    logging.warning(f"Error extracting feature {name} for pair {id1}-{id2}: {e}")
                    pair_features[name] = 0.0
            
            # Add label if provided
            if labels is not None:
                pair_features["label"] = labels.iloc[idx] if hasattr(labels, 'iloc') else labels[idx]
            
            feature_data.append(pair_features)
        
        # Create DataFrame
        feature_df = pd.DataFrame(feature_data)
        
        if feature_df.empty:
            raise ValueError("No valid features could be extracted from the provided pairs")
        
        # Log feature extraction summary
        feature_columns = [f["name"] for f in self.feature_functions]
        logging.info(f"Feature extraction complete: {len(feature_df)} pairs, {len(feature_columns)} features")
        
        if labels is not None:
            pos_labels = sum(feature_df["label"])
            neg_labels = len(feature_df) - pos_labels
            logging.info(f"Label distribution: {pos_labels} positive, {neg_labels} negative")
        
        return feature_df
    
    def get_feature_names(self) -> List[str]:
        """Get names of all features that will be extracted.
        
        Returns
        -------
        List[str]
            List of feature names in the order they appear in feature matrix.
        """
        return [f["name"] for f in self.feature_functions]
    
    def __repr__(self) -> str:
        return f"FeatureExtractor(n_features={len(self.feature_functions)})"


class VectorFeatureExtractor:
    """Extract vector-based features using embeddings and distance metrics.
    
    This class provides vector-based feature extraction using sentence transformers
    or other embedding models to create high-dimensional feature representations
    of text data.
    
    Parameters
    ----------
    embedding_model : str or object
        Either a string name of a sentence transformer model (e.g., 'all-MiniLM-L6-v2')
        or a pre-loaded embedding model with an encode() method.
    columns : List[str]
        List of column names to create embeddings for.
    distance_metrics : List[str], optional
        List of distance metrics to compute. Options: 'cosine', 'euclidean', 'manhattan'.
        Default is ['cosine'].
    pooling_strategy : str, optional
        How to combine multiple column embeddings. Options: 'concatenate', 'mean', 'max'.
        Default is 'concatenate'.
        
    Examples
    --------
    >>> extractor = VectorFeatureExtractor(
    ...     embedding_model='all-MiniLM-L6-v2',
    ...     columns=['title', 'description'],
    ...     distance_metrics=['cosine', 'euclidean']
    ... )
    >>> features = extractor.create_features(df_left, df_right, pairs)
    """
    
    def __init__(
        self,
        embedding_model: Union[str, Any],
        columns: List[str],
        distance_metrics: List[str] = None,
        pooling_strategy: str = "concatenate",
    ):
        if not columns:
            raise ValueError("At least one column must be specified")
            
        self.columns = columns
        self.distance_metrics = distance_metrics or ["cosine"]
        self.pooling_strategy = pooling_strategy
        
        # Initialize embedding model
        if isinstance(embedding_model, str):
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(embedding_model)
                self.model_name = embedding_model
            except ImportError:
                raise ImportError("sentence-transformers package required for vector features")
        else:
            if not hasattr(embedding_model, 'encode'):
                raise ValueError("Embedding model must have an 'encode' method")
            self.model = embedding_model
            self.model_name = str(embedding_model)
        
        # Validate distance metrics
        valid_metrics = {"cosine", "euclidean", "manhattan"}
        for metric in self.distance_metrics:
            if metric not in valid_metrics:
                raise ValueError(f"Unsupported distance metric: {metric}. Valid options: {valid_metrics}")
                
        logging.info(f"Initialized VectorFeatureExtractor with model {self.model_name}")
    
    def _compute_embedding(self, record: pd.Series) -> np.ndarray:
        """Compute embedding for a record by combining specified columns."""
        # Extract text from specified columns
        texts = []
        for col in self.columns:
            value = record.get(col, "")
            if pd.notna(value):
                texts.append(str(value))
            else:
                texts.append("")
        
        if self.pooling_strategy == "concatenate":
            # Concatenate all text
            combined_text = " ".join(texts)
            return self.model.encode([combined_text])[0]
        elif self.pooling_strategy == "mean":
            # Compute embedding for each column and average
            embeddings = []
            for text in texts:
                if text.strip():
                    embeddings.append(self.model.encode([text])[0])
            if embeddings:
                return np.mean(embeddings, axis=0)
            else:
                # Return zero vector if no valid text
                return np.zeros(self.model.get_sentence_embedding_dimension())
        else:
            raise ValueError(f"Unsupported pooling strategy: {self.pooling_strategy}")
    
    def _compute_distance(self, emb1: np.ndarray, emb2: np.ndarray, metric: str) -> float:
        """Compute distance between two embeddings."""
        if metric == "cosine":
            # Cosine similarity (convert to 0-1 range)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(np.dot(emb1, emb2) / (norm1 * norm2))
        elif metric == "euclidean":
            # Euclidean distance (convert to similarity)
            dist = np.linalg.norm(emb1 - emb2)
            return float(1.0 / (1.0 + dist))
        elif metric == "manhattan":
            # Manhattan distance (convert to similarity)
            dist = np.sum(np.abs(emb1 - emb2))
            return float(1.0 / (1.0 + dist))
        else:
            raise ValueError(f"Unsupported distance metric: {metric}")
    
    def create_features(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        pairs: pd.DataFrame,
        id_column: str,
        labels: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """Create vector-based features from entity pairs.

        Parameters
        ----------
        df_left : pandas.DataFrame
            Left dataset with specified ID column.
        df_right : pandas.DataFrame
            Right dataset with specified ID column.
        pairs : pandas.DataFrame
            DataFrame with id1, id2 columns.
        labels : pandas.Series, optional
            Binary labels for pairs.
            
        Returns
        -------
        pandas.DataFrame
            Feature matrix with distance-based features.
        """
        # Input validation
        if pairs.empty:
            raise ValueError("Empty pairs DataFrame provided")
            
        # Validate columns exist
        for col in self.columns:
            if col not in df_left.columns:
                raise ValueError(f"Column '{col}' not found in left dataset")
            if col not in df_right.columns:
                raise ValueError(f"Column '{col}' not found in right dataset")
        
        # Validate that ID column exists
        if id_column not in df_left.columns:
            raise ValueError(f"Left dataset missing ID column: {id_column}")
        if id_column not in df_right.columns:
            raise ValueError(f"Right dataset missing ID column: {id_column}")

        # Create lookup dictionaries
        left_lookup = df_left.set_index(id_column)
        right_lookup = df_right.set_index(id_column)
        
        logging.info(f"Computing vector features for {len(pairs)} pairs")
        
        # Pre-compute embeddings for efficiency
        logging.info("Computing embeddings for left dataset...")
        left_embeddings = {}
        for id_val, record in left_lookup.iterrows():
            left_embeddings[id_val] = self._compute_embedding(record)
        
        logging.info("Computing embeddings for right dataset...")
        right_embeddings = {}
        for id_val, record in right_lookup.iterrows():
            right_embeddings[id_val] = self._compute_embedding(record)
        
        # Process pairs
        feature_data = []
        for idx, pair in pairs.iterrows():
            id1, id2 = pair["id1"], pair["id2"]
            
            if id1 not in left_embeddings or id2 not in right_embeddings:
                logging.warning(f"Missing embedding for pair {id1}-{id2}")
                continue
            
            emb1 = left_embeddings[id1]
            emb2 = right_embeddings[id2]
            
            # Compute distance features
            pair_features = {"id1": id1, "id2": id2}
            
            for metric in self.distance_metrics:
                feature_name = f"vector_{metric}"
                try:
                    distance = self._compute_distance(emb1, emb2, metric)
                    pair_features[feature_name] = distance
                except Exception as e:
                    logging.warning(f"Error computing {metric} for pair {id1}-{id2}: {e}")
                    pair_features[feature_name] = 0.0
            
            # Add label if provided
            if labels is not None:
                pair_features["label"] = labels.iloc[idx] if hasattr(labels, 'iloc') else labels[idx]
            
            feature_data.append(pair_features)
        
        feature_df = pd.DataFrame(feature_data)
        
        if feature_df.empty:
            raise ValueError("No valid vector features could be extracted")
        
        logging.info(f"Vector feature extraction complete: {len(feature_df)} pairs, {len(self.distance_metrics)} features")
        
        return feature_df
    
    def get_feature_names(self) -> List[str]:
        """Get names of vector features."""
        return [f"vector_{metric}" for metric in self.distance_metrics]