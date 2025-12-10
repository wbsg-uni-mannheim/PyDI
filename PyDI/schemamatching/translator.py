"""
Schema translation using explicit column mappings.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, Literal, Union

import pandas as pd

from .base import SchemaMapping

# Lazy imports for normalization to avoid circular dependencies
_normalization_imported = False
_NormalizationSpec = None
_transform_dataframe = None
_profile_dataframe = None


def _import_normalization():
    """Lazy import normalization module components."""
    global _normalization_imported, _NormalizationSpec, _transform_dataframe, _profile_dataframe
    if not _normalization_imported:
        from ..normalization import NormalizationSpec, transform_dataframe, profile_dataframe
        _NormalizationSpec = NormalizationSpec
        _transform_dataframe = transform_dataframe
        _profile_dataframe = profile_dataframe
        _normalization_imported = True


class SchemaTranslator:
    """Translate column names based on a schema mapping.

    This is the final step of schema matching: applying the discovered
    column correspondences to rename columns in the source DataFrame
    to match the target schema. Optionally applies value normalization.
    """

    def translate(
        self,
        df: pd.DataFrame,
        mapping: SchemaMapping,
        normalize: Union["NormalizationSpec", bool, None] = None,
        on_failure: Literal["keep", "null", "raise"] = "keep",
        return_result: bool = False,
    ) -> Union[pd.DataFrame, tuple[pd.DataFrame, "DataFrameTransformResult"]]:
        """Translate column names according to a schema mapping.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame to translate. Must have ``dataset_name`` in attrs.
        mapping : SchemaMapping
            Schema mapping DataFrame with columns ``source_dataset``,
            ``source_column``, ``target_dataset``, ``target_column``,
            and optionally ``score``.
        normalize : NormalizationSpec, bool, or None, optional
            Controls value normalization after column renaming:

            - ``None`` (default): No normalization, only rename columns.
            - ``True``: Auto-detect normalizations by profiling the DataFrame
              and generating a spec via ``NormalizationSpec.from_profile()``.
            - ``NormalizationSpec``: Use the provided spec for normalization.
              Column names in the spec should be the **target** column names
              (after renaming).
        on_failure : {"keep", "null", "raise"}, default "keep"
            Default behavior when a value normalization fails:

            - ``"keep"``: Keep the original value unchanged.
            - ``"null"``: Replace with ``None``/``NaN``.
            - ``"raise"``: Raise a ``ValueError`` immediately.

            Individual ``ColumnSpec`` entries in the normalization spec can
            override this default via their own ``on_failure`` setting.
        return_result : bool, default False
            If True, return a tuple of (DataFrame, DataFrameTransformResult)
            to access per-column transformation errors and statistics.

        Returns
        -------
        pandas.DataFrame or tuple
            A new DataFrame with columns renamed (and optionally normalized)
            according to the mapping. If ``return_result=True``, returns
            ``(DataFrame, DataFrameTransformResult)``.

        Raises
        ------
        ValueError
            If DataFrame is missing dataset_name, if schema mapping is invalid,
            or if ``on_failure="raise"`` and a normalization fails.

        Examples
        --------
        Basic column renaming only:

        >>> translator = SchemaTranslator()
        >>> df_translated = translator.translate(source_df, mapping)

        With auto-detected normalization:

        >>> df_translated = translator.translate(source_df, mapping, normalize=True)

        With explicit normalization spec:

        >>> from PyDI.normalization import NormalizationSpec
        >>> spec = NormalizationSpec()
        >>> spec.set_column("country", country_format="alpha_2")
        >>> spec.set_column("revenue", expand_scale_modifiers=True)
        >>> df_translated = translator.translate(
        ...     source_df, mapping, normalize=spec, on_failure="null"
        ... )
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

        # Build a mapping from string column names to actual column names
        # This handles integer columns (0, 1, 2) that may be referenced as strings
        col_str_to_actual = {str(col): col for col in df.columns}

        for _, row in relevant.iterrows():
            src = row["source_column"]
            tgt = row["target_column"]
            score = row.get("score", 1.0) if has_score else 1.0

            # Look up the actual column name (handles int columns referenced as strings)
            actual_src = col_str_to_actual.get(str(src))
            if actual_src is None:
                logging.warning(f"Column '{src}' not found in dataset '{dataset_name}'")
                continue

            if actual_src not in rename_map or (has_score and score > best_scores.get(actual_src, 0)):
                rename_map[actual_src] = tgt
                if has_score:
                    best_scores[actual_src] = score

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

        # Check for duplicate columns (can happen if multiple source cols map to same target)
        if translated.columns.duplicated().any():
            dup_cols = translated.columns[translated.columns.duplicated(keep=False)].unique().tolist()
            logging.warning(
                f"Multiple source columns mapped to same target: {dup_cols}. "
                "Keeping first occurrence of each duplicate."
            )
            translated = translated.loc[:, ~translated.columns.duplicated(keep='first')]

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

        # Apply normalization if requested
        if normalize is not None:
            translated, transform_result = self._apply_normalization(translated, normalize, on_failure)
            if return_result:
                return translated, transform_result

        return translated

    def _apply_normalization(
        self,
        df: pd.DataFrame,
        normalize: Union["NormalizationSpec", bool],
        on_failure: Literal["keep", "null", "raise"],
    ) -> tuple[pd.DataFrame, "DataFrameTransformResult"]:
        """Apply value normalization to the DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame to normalize (already has renamed columns).
        normalize : NormalizationSpec or bool
            If True, auto-detect normalizations. If NormalizationSpec, use it.
        on_failure : {"keep", "null", "raise"}
            Default failure behavior for columns without explicit on_failure.

        Returns
        -------
        tuple
            (normalized DataFrame, DataFrameTransformResult)
        """
        _import_normalization()

        # Get or create the normalization spec
        if normalize is True:
            # Auto-detect normalizations from profile
            profile = _profile_dataframe(df)
            spec = _NormalizationSpec.from_profile(profile)
            logging.info(f"Auto-detected normalization for {len(spec.columns)} columns")
        else:
            # Use provided spec
            spec = normalize

        # If no columns to normalize, return unchanged
        if not spec.columns:
            logging.info("No columns to normalize")
            # Return empty result
            from ..normalization import DataFrameTransformResult
            return df, DataFrameTransformResult(df, {}, 0, 0)

        # Apply on_failure default to columns that don't have it explicitly set
        for col_name, col_spec in spec.columns.items():
            # ColumnSpec default is "keep", so we only override if user passed
            # a different default and the spec hasn't been explicitly set
            # We check if on_failure is still at its default value
            if col_spec.on_failure == "keep" and on_failure != "keep":
                col_spec.on_failure = on_failure

        # Transform the DataFrame
        result = _transform_dataframe(df, spec)

        logging.info(
            f"Normalization complete: {result.total_transformed} values transformed, "
            f"{result.total_failed} values failed"
        )

        # Add normalization provenance
        normalized_df = result.dataframe
        provenance_entry = {
            "op": "value_normalize",
            "params": {
                "columns": list(spec.columns.keys()),
                "on_failure": on_failure,
                "transformed": result.total_transformed,
                "failed": result.total_failed,
            },
            "ts": datetime.now(timezone.utc).isoformat(),
        }

        if "provenance" not in normalized_df.attrs:
            normalized_df.attrs["provenance"] = []
        elif not isinstance(normalized_df.attrs["provenance"], list):
            normalized_df.attrs["provenance"] = [normalized_df.attrs["provenance"]]
        normalized_df.attrs["provenance"].append(provenance_entry)

        return normalized_df, result