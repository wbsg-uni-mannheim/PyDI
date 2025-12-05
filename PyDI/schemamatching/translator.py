"""
Schema translation using explicit column mappings.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict

import pandas as pd

from .base import SchemaMapping


class SchemaTranslator:
    """Translate column names based on a schema mapping.

    This is the final step of schema matching: applying the discovered
    column correspondences to rename columns in the source DataFrame
    to match the target schema.
    """

    def translate(self, df: pd.DataFrame, mapping: SchemaMapping) -> pd.DataFrame:
        """Translate column names according to a schema mapping.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame to translate. Must have ``dataset_name`` in attrs.
        mapping : SchemaMapping
            Schema mapping DataFrame with columns ``source_dataset``,
            ``source_column``, ``target_dataset``, ``target_column``,
            and optionally ``score``.

        Returns
        -------
        pandas.DataFrame
            A new DataFrame with columns renamed according to the mapping.

        Raises
        ------
        ValueError
            If DataFrame is missing dataset_name or if schema mapping is invalid.
        """
        dataset_name = df.attrs.get("dataset_name")
        if dataset_name is None:
            raise ValueError("DataFrame is missing 'dataset_name' in attrs")

        required_columns = {"source_dataset", "source_column", "target_dataset", "target_column"}
        if not required_columns.issubset(mapping.columns):
            missing = required_columns - set(mapping.columns)
            raise ValueError(f"SchemaMapping is missing required columns: {missing}")

        relevant = mapping[mapping["source_dataset"] == dataset_name]

        if relevant.empty:
            logging.info(f"No schema mappings found for dataset '{dataset_name}'")
            return df.copy()

        # Build column rename dict, picking best score if duplicates exist
        rename_map: Dict[str, str] = {}
        best_scores: Dict[str, float] = {}
        has_score = "score" in relevant.columns

        for _, row in relevant.iterrows():
            src = row["source_column"]
            tgt = row["target_column"]
            score = row.get("score", 1.0) if has_score else 1.0

            if src not in df.columns:
                logging.warning(f"Column '{src}' not found in dataset '{dataset_name}'")
                continue

            if src not in rename_map or (has_score and score > best_scores.get(src, 0)):
                rename_map[src] = tgt
                if has_score:
                    best_scores[src] = score

        if not rename_map:
            logging.info(f"No applicable mappings for dataset '{dataset_name}'")
            return df.copy()

        logging.info(f"Translating {len(rename_map)} columns for '{dataset_name}'")

        # Store original column attrs before rename
        original_attrs = {
            tgt: df[src].attrs.copy()
            for src, tgt in rename_map.items()
            if hasattr(df[src], 'attrs')
        }

        translated = df.rename(columns=rename_map, copy=True)
        translated.attrs = df.attrs.copy()

        # Add provenance
        provenance_entry = {
            "op": "schema_translate",
            "params": {"mappings": rename_map},
            "ts": datetime.now(timezone.utc).isoformat(),
        }

        if "provenance" not in translated.attrs:
            translated.attrs["provenance"] = []
        elif not isinstance(translated.attrs["provenance"], list):
            translated.attrs["provenance"] = [translated.attrs["provenance"]]
        translated.attrs["provenance"].append(provenance_entry)

        # Restore column attrs and add column-level provenance
        for src, tgt in rename_map.items():
            if tgt in translated.columns:
                translated[tgt].attrs = original_attrs.get(tgt, {})

                if "provenance" not in translated[tgt].attrs:
                    translated[tgt].attrs["provenance"] = []
                elif not isinstance(translated[tgt].attrs["provenance"], list):
                    translated[tgt].attrs["provenance"] = [translated[tgt].attrs["provenance"]]

                translated[tgt].attrs["provenance"].append({
                    "op": "schema_transform",
                    "params": {"name_old": src, "name_new": tgt},
                    "ts": datetime.now(timezone.utc).isoformat(),
                })

        return translated