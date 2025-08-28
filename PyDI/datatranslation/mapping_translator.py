"""
Schema translation using explicit column mappings.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, Optional

import pandas as pd

from .base import BaseTranslator
from ..schemamatching.base import SchemaMapping


class MappingTranslator(BaseTranslator):
    """Translate column names based on an explicit schema mapping.
    
    This translator applies column renaming transformations based on a
    SchemaMapping DataFrame. It filters the mapping to only apply transformations
    relevant to the input dataset and preserves metadata with provenance tracking.
    """
    
    def __init__(self, strategy: str = "rename") -> None:
        """Initialize the mapping translator.
        
        Parameters
        ----------
        strategy : str, optional
            Translation strategy. Currently only "rename" is supported.
            Default is "rename".
        """
        if strategy != "rename":
            raise ValueError(f"Unsupported translation strategy: '{strategy}'. Only 'rename' is supported.")
        
        self.strategy = strategy
    
    def translate(self, df: pd.DataFrame, corr: SchemaMapping) -> pd.DataFrame:
        """Translate column names according to a schema mapping.
        
        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame to translate. Must have ``dataset_name`` in attrs.
        corr : SchemaMapping
            Schema mapping DataFrame with the required columns.
            
        Returns
        -------
        pandas.DataFrame
            A new DataFrame with columns renamed according to the mapping.
            
        Raises
        ------
        ValueError
            If DataFrame is missing dataset_name or if schema mapping is invalid.
        """
        # Validate input DataFrame
        dataset_name = df.attrs.get("dataset_name")
        if dataset_name is None:
            raise ValueError("DataFrame is missing 'dataset_name' in attrs")
            
        # Validate schema mapping
        required_columns = {"source_dataset", "source_column", "target_dataset", "target_column"}
        if not required_columns.issubset(corr.columns):
            missing = required_columns - set(corr.columns)
            raise ValueError(f"SchemaMapping is missing required columns: {missing}")
            
        # Extract mappings for this dataset
        relevant_mappings = corr[corr["source_dataset"] == dataset_name]
        
        if relevant_mappings.empty:
            logging.info(f"No schema mappings found for dataset '{dataset_name}', returning unchanged DataFrame")
            return df.copy()
            
        # Build column mapping dictionary, selecting best mapping by score
        mapping: Dict[str, str] = {}
        best_scores: Dict[str, float] = {}
        unmapped_columns = []
        
        # Check if score column exists in the mapping
        has_score = "score" in relevant_mappings.columns
        
        for _, row in relevant_mappings.iterrows():
            source_col = row["source_column"]
            target_col = row["target_column"]
            score = row.get("score", 1.0) if has_score else 1.0
            
            if source_col in df.columns:
                if source_col in mapping:
                    # Compare scores and keep the better mapping
                    if has_score and source_col in best_scores and score > best_scores[source_col]:
                        logging.info(f"Better mapping found for column '{source_col}' in dataset '{dataset_name}': "
                                   f"'{target_col}' (score={score:.4f}) replacing '{mapping[source_col]}' "
                                   f"(score={best_scores[source_col]:.4f})")
                        mapping[source_col] = target_col
                        best_scores[source_col] = score
                    elif has_score and source_col not in best_scores:
                        # First time seeing this column with score, replace no-score mapping
                        logging.info(f"Adding score to mapping for column '{source_col}' in dataset '{dataset_name}': "
                                   f"'{target_col}' (score={score:.4f}) replacing '{mapping[source_col]}'")
                        mapping[source_col] = target_col
                        best_scores[source_col] = score
                    elif not has_score:
                        logging.warning(f"Duplicate mapping for column '{source_col}' in dataset '{dataset_name}', "
                                      f"using '{target_col}' (was '{mapping[source_col]}') - no scores available")
                        mapping[source_col] = target_col
                    else:
                        logging.debug(f"Keeping existing mapping for column '{source_col}': "
                                    f"'{mapping[source_col]}' (score={best_scores[source_col]:.4f}) over "
                                    f"'{target_col}' (score={score:.4f})")
                else:
                    mapping[source_col] = target_col
                    if has_score:
                        best_scores[source_col] = score
            else:
                unmapped_columns.append(source_col)
                
        if unmapped_columns:
            logging.warning(f"Schema mapping references columns not found in dataset '{dataset_name}': "
                          f"{unmapped_columns}")
        
        # Apply translation
        if not mapping:
            logging.info(f"No applicable mappings for dataset '{dataset_name}', returning unchanged DataFrame")
            return df.copy()
            
        logging.info(f"Translating {len(mapping)} columns for dataset '{dataset_name}': {mapping}")
        
        # Store original column attributes before rename (pandas doesn't preserve them)
        original_column_attrs = {}
        for source_col, target_col in mapping.items():
            if source_col in df.columns and hasattr(df[source_col], 'attrs'):
                original_column_attrs[target_col] = df[source_col].attrs.copy()
        
        # Create translated DataFrame
        translated_df = df.rename(columns=mapping, copy=True)
        
        # Preserve and update metadata
        translated_df.attrs = df.attrs.copy()
        
        # Add provenance information
        provenance_entry = {
            "op": "schema_translate",
            "params": {
                "strategy": self.strategy,
                "mappings": mapping,
                "translator": self.__class__.__name__,
            },
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        
        # Initialize or extend provenance list
        if "provenance" not in translated_df.attrs:
            translated_df.attrs["provenance"] = []
        elif not isinstance(translated_df.attrs["provenance"], list):
            # Convert existing provenance to list format if it's not already
            translated_df.attrs["provenance"] = [translated_df.attrs["provenance"]]
        translated_df.attrs["provenance"].append(provenance_entry)
        
        # Restore column-level attributes and add provenance for renamed columns
        for source_col, target_col in mapping.items():
            if source_col in df.columns and target_col in translated_df.columns:
                # Restore original column attributes
                if target_col in original_column_attrs:
                    translated_df[target_col].attrs = original_column_attrs[target_col]
                else:
                    # Initialize attrs if source didn't have any
                    translated_df[target_col].attrs = {}
                    
                # Add column-level provenance
                if "provenance" not in translated_df[target_col].attrs:
                    translated_df[target_col].attrs["provenance"] = []
                elif not isinstance(translated_df[target_col].attrs["provenance"], list):
                    # Convert existing provenance to list format if it's not already
                    translated_df[target_col].attrs["provenance"] = [translated_df[target_col].attrs["provenance"]]
                
                column_provenance = {
                    "op": "schema_transform",
                    "params": {
                        "name_old": source_col,
                        "name_new": target_col,
                    },
                    "ts": datetime.now(timezone.utc).isoformat(),
                }
                translated_df[target_col].attrs["provenance"].append(column_provenance)
        
        return translated_df