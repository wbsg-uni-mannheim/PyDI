"""
Duplicate-based schema matching using known record correspondences.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from .base import BaseSchemaMatcher, SchemaMapping, get_schema_columns
from ..utils import SimilarityRegistry


class DuplicateBasedSchemaMatcher(BaseSchemaMatcher):
    """Duplicate-based schema matcher using known record correspondences.
    
    This matcher uses known duplicate records (correspondences) between datasets
    to determine schema correspondences. For each pair of duplicate records,
    it compares attribute values and aggregates votes to determine which
    attributes correspond across schemas.
    
    Supports both exact and fuzzy string matching using the textdistance package,
    enabling more flexible matching of similar but not identical values.
    """
    
    def __init__(
        self,
        vote_aggregation: str = "majority",
        value_comparison: str = "exact",
        min_votes: int = 1,
        ignore_zero_values: bool = True,
        similarity_function: Optional[str] = None,
        similarity_threshold: float = 0.8,
    ) -> None:
        """Initialize the duplicate-based schema matcher.
        
        Parameters
        ----------
        vote_aggregation : str, optional
            Method for aggregating votes. Options: "majority", "weighted".
            Default is "majority".
        value_comparison : str, optional
            Method for comparing values. Options: "exact", "normalized", "fuzzy".
            Default is "exact".
        min_votes : int, optional
            Minimum number of votes required for a correspondence.
        ignore_zero_values : bool, optional
            Whether to ignore zero or empty values when voting.
        similarity_function : str, optional
            Similarity function for fuzzy matching when value_comparison="fuzzy".
            Any function from SimilarityRegistry. Recommended: "levenshtein", "jaro_winkler", "jaccard".
            Default is None (exact matching).
        similarity_threshold : float, optional
            Minimum similarity score for fuzzy matching. Default is 0.8.
        """
        self.vote_aggregation = vote_aggregation
        self.value_comparison = value_comparison
        self.min_votes = min_votes
        self.ignore_zero_values = ignore_zero_values
        self.similarity_function = similarity_function
        self.similarity_threshold = similarity_threshold
        
        if vote_aggregation not in ["majority", "weighted"]:
            raise ValueError(f"Unsupported vote aggregation: {vote_aggregation}")
        
        if value_comparison not in ["exact", "normalized", "fuzzy"]:
            raise ValueError(f"Unsupported value comparison: {value_comparison}")
            
        if value_comparison == "fuzzy" and similarity_function is None:
            raise ValueError("similarity_function must be specified when using fuzzy value comparison")
            
        # Initialize similarity function for fuzzy matching
        if similarity_function:
            try:
                self._sim_func = SimilarityRegistry.get_function(similarity_function)
            except ValueError as e:
                available_funcs = SimilarityRegistry.get_recommended_functions("schema_matching", "duplicate")
                raise ValueError(f"{e}. Recommended functions for duplicate-based matching: {available_funcs}")
        else:
            self._sim_func = None
    
    def _normalize_value(self, value: Any, preprocess: Optional[Callable[[str], str]] = None) -> str:
        """Normalize a value for comparison."""
        if pd.isna(value):
            return ""
        
        str_val = str(value).strip().lower()
        
        if self.value_comparison == "normalized":
            # Remove extra whitespace and punctuation
            import re
            str_val = re.sub(r'[^\w\s]', '', str_val)
            str_val = ' '.join(str_val.split())
        
        # Apply preprocessing if provided
        if preprocess:
            str_val = preprocess(str_val)
            
        return str_val
    
    def _values_match(self, val1: Any, val2: Any, preprocess: Optional[Callable[[str], str]] = None) -> bool:
        """Check if two values match according to comparison method."""
        norm_val1 = self._normalize_value(val1, preprocess)
        norm_val2 = self._normalize_value(val2, preprocess)
        
        # Skip empty/zero values if configured
        if self.ignore_zero_values:
            if norm_val1 in ["", "0", "0.0", "nan", "null", "none"]:
                return False
            if norm_val2 in ["", "0", "0.0", "nan", "null", "none"]:
                return False
        
        # Exact matching (original behavior)
        if self.value_comparison in ["exact", "normalized"]:
            return norm_val1 == norm_val2
        
        # Fuzzy matching using textdistance
        elif self.value_comparison == "fuzzy":
            if not norm_val1 or not norm_val2:
                return False
            similarity = self._sim_func(norm_val1, norm_val2)
            return similarity >= self.similarity_threshold
        
        return False
    
    def _collect_votes(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        correspondences: pd.DataFrame,
        preprocess: Optional[Callable[[str], str]] = None
    ) -> Dict[Tuple[str, str], int]:
        """Collect votes for schema correspondences from record correspondences."""
        votes = defaultdict(int)
        
        logging.info(f"Collecting votes from {len(correspondences)} record correspondences")
        
        for _, corr in correspondences.iterrows():
            # Get the matching records
            id1 = corr.get("id1", corr.get("source_id", corr.get("first_id")))
            id2 = corr.get("id2", corr.get("target_id", corr.get("second_id")))
            
            if id1 is None or id2 is None:
                logging.warning("Could not find ID columns in correspondence")
                continue
            
            # Find records in dataframes
            # Try different common ID column names
            id_cols_1 = [col for col in df1.columns if 'id' in col.lower()]
            id_cols_2 = [col for col in df2.columns if 'id' in col.lower()]
            
            record1 = None
            record2 = None
            
            # Try to find matching records
            id_cols_1_extended = id_cols_1 + ([df1.index.name] if df1.index.name else [])
            for id_col in id_cols_1_extended:
                if id_col:
                    if id_col in df1.columns:
                        matching_rows = df1[df1[id_col] == id1]
                        if not matching_rows.empty:
                            record1 = matching_rows.iloc[0]
                            break
                    elif df1.index.name == id_col:
                        # Match using the index
                        try:
                            # For MultiIndex, get_level_values
                            matching_idx = df1.index.get_level_values(id_col) == id1
                        except (KeyError, AttributeError, TypeError):
                            # For single index or unnamed index
                            matching_idx = df1.index == id1
                        matching_rows = df1[matching_idx]
                        if not matching_rows.empty:
                            record1 = matching_rows.iloc[0]
                            break
            
            id_cols_2_extended = id_cols_2 + ([df2.index.name] if df2.index.name else [])
            for id_col in id_cols_2_extended:
                if id_col:
                    if id_col in df2.columns:
                        matching_rows = df2[df2[id_col] == id2]
                        if not matching_rows.empty:
                            record2 = matching_rows.iloc[0]
                            break
                    elif df2.index.name == id_col:
                        try:
                            matching_idx = df2.index.get_level_values(id_col) == id2
                        except (KeyError, AttributeError, TypeError):
                            matching_idx = df2.index == id2
                        matching_rows = df2[matching_idx]
                        if not matching_rows.empty:
                            record2 = matching_rows.iloc[0]
                            break
            
            if record1 is None or record2 is None:
                logging.debug(f"Could not find records for correspondence {id1} <-> {id2}")
                continue
            
            # Compare all attribute pairs (excluding PyDI ID columns)
            schema_cols_1 = get_schema_columns(df1)
            schema_cols_2 = get_schema_columns(df2)
            
            for attr1 in schema_cols_1:
                for attr2 in schema_cols_2:
                    val1 = record1[attr1]
                    val2 = record2[attr2]
                    
                    if self._values_match(val1, val2, preprocess):
                        votes[(attr1, attr2)] += 1
        
        return votes
    
    def _aggregate_votes(
        self,
        votes: Dict[Tuple[str, str], int],
        threshold: float
    ) -> List[Dict[str, Any]]:
        """Aggregate votes to determine final correspondences."""
        results = []
        
        if self.vote_aggregation == "majority":
            # Simple majority voting
            for (attr1, attr2), vote_count in votes.items():
                if vote_count >= self.min_votes:
                    confidence = min(vote_count / max(votes.values()), 1.0) if votes else 0.0
                    
                    if confidence >= threshold:
                        results.append({
                            "source_column": attr1,
                            "target_column": attr2,
                            "score": confidence,
                            "votes": vote_count
                        })
        
        elif self.vote_aggregation == "weighted":
            # Weighted voting (could be extended with more sophisticated weighting)
            total_votes = sum(votes.values())
            
            for (attr1, attr2), vote_count in votes.items():
                if vote_count >= self.min_votes:
                    confidence = vote_count / total_votes if total_votes > 0 else 0.0
                    
                    if confidence >= threshold:
                        results.append({
                            "source_column": attr1,
                            "target_column": attr2,
                            "score": confidence,
                            "votes": vote_count
                        })
        
        return results
    
    def match(
        self,
        source_dataset: pd.DataFrame,
        target_dataset: pd.DataFrame,
        preprocess: Optional[Callable[[str], str]] = None,
        threshold: float = 0.1,
        correspondences: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> SchemaMapping:
        """Find schema correspondences using duplicate-based matching.
        
        Parameters
        ----------
        source_dataset : pandas.DataFrame
            The source dataset.
        target_dataset : pandas.DataFrame
            The target dataset.
        preprocess : callable, optional
            Function to preprocess values before comparison.
        threshold : float, optional
            Minimum confidence score for correspondences.
        correspondences : pandas.DataFrame, optional
            Record correspondences between the datasets. Must contain columns
            identifying matching records (e.g., 'id1', 'id2' or 'source_id', 'target_id').
        **kwargs
            Additional keyword arguments (ignored).
            
        Returns
        -------
        SchemaMapping
            DataFrame with schema correspondences.
            
        Raises
        ------
        ValueError
            If correspondences are not provided.
        """
        if correspondences is None or correspondences.empty:
            raise ValueError("Duplicate-based matching requires record correspondences")
        
        source_name = source_dataset.attrs.get("dataset_name", "source")
        target_name = target_dataset.attrs.get("dataset_name", "target")
        
        # Get schema columns excluding PyDI-generated ID columns  
        source_columns = get_schema_columns(source_dataset)
        target_columns = get_schema_columns(target_dataset)
        
        logging.info(f"Duplicate-based matching: {source_name} -> {target_name}")
        logging.info(f"Source columns for matching: {source_columns}")
        logging.info(f"Target columns for matching: {target_columns}")
        
        # Collect votes from record correspondences
        votes = self._collect_votes(source_dataset, target_dataset, correspondences, preprocess)
        
        logging.info(f"Collected {len(votes)} attribute pair votes")
        
        # Aggregate votes to get final correspondences
        aggregated_results = self._aggregate_votes(votes, threshold)
        
        # Format results
        results = []
        for result in aggregated_results:
            results.append({
                "source_dataset": source_name,
                "source_column": result["source_column"],
                "target_dataset": target_name,
                "target_column": result["target_column"],
                "score": result["score"],
                "notes": f"votes={result['votes']},method=duplicate_based"
            })
            
            logging.debug(
                f"Duplicate match: {result['source_column']} <-> {result['target_column']} "
                f"({result['score']:.4f}, {result['votes']} votes)"
            )
        
        return pd.DataFrame(results)