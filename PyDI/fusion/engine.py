"""
Data fusion engine for PyDI.

This module defines the DataFusionEngine class and utilities for record grouping
and executing fusion strategies.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple, Union, Any
from collections import defaultdict
import logging
import time
import pandas as pd
from pathlib import Path
import json

from .base import RecordGroup, FusionContext, FusionResult, _is_valid_value
from .provenance import extract_source_trust_scores
from .strategy import DataFusionStrategy


def build_record_groups_from_correspondences(
    datasets: List[pd.DataFrame],
    correspondences: pd.DataFrame,
    id_column: Optional[Union[str, Dict[str, str]]] = None,
) -> List[RecordGroup]:
    """Build record groups from correspondences using connected components.

    Parameters
    ----------
    datasets : List[pd.DataFrame]
        List of input datasets. Each must have 'dataset_name' in attrs. If
        `id_column` is provided, this function will ensure a string `_id`
        column is created from the specified column(s).
    correspondences : pd.DataFrame
        Correspondences with columns 'id1', 'id2', and optionally 'score'.
        This is compatible with PyDI's CorrespondenceSet format.
    id_column : Optional[Union[str, Dict[str, str]]]
        Identifier column name(s). If a string, the same column is used for
        all datasets. If a dict, it should map dataset name -> ID column for
        that dataset.

    Returns
    -------
    List[RecordGroup]
        List of record groups representing connected components.
    """
    logger = logging.getLogger(__name__)

    # Normalize datasets to ensure `_id` exists and is string-typed
    normalized_datasets: List[pd.DataFrame] = []
    for df in datasets:
        dataset_name = df.attrs.get("dataset_name")
        if not dataset_name:
            raise ValueError(
                "Each dataset must have 'dataset_name' in df.attrs")

        selected_id_col: Optional[str] = None
        if isinstance(id_column, dict):
            selected_id_col = id_column.get(dataset_name)
        elif isinstance(id_column, str):
            selected_id_col = id_column

        df_copy = df.copy()
        try:
            df_copy.attrs = dict(df.attrs)
        except Exception:
            pass

        if selected_id_col:
            if selected_id_col not in df_copy.columns:
                raise ValueError(
                    f"ID column '{selected_id_col}' not found in dataset '{dataset_name}'"
                )
            df_copy["_id"] = df_copy[selected_id_col].astype(str)
        elif "_id" in df_copy.columns:
            df_copy["_id"] = df_copy["_id"].astype(str)
        elif "id" in df_copy.columns:
            df_copy["_id"] = df_copy["id"].astype(str)
        else:
            raise ValueError(
                f"Dataset '{dataset_name}' lacks an ID column. Provide 'id_column' or include '_id'/'id'."
            )

        normalized_datasets.append(df_copy)

    # Normalize correspondences ID dtypes to strings
    correspondences = correspondences.copy()
    if "id1" not in correspondences.columns or "id2" not in correspondences.columns:
        raise ValueError(
            "Correspondences must contain 'id1' and 'id2' columns")
    correspondences["id1"] = correspondences["id1"].astype(str)
    correspondences["id2"] = correspondences["id2"].astype(str)

    # Build mapping from record ID to record and dataset name
    id_to_record: Dict[str, pd.Series] = {}
    id_to_dataset: Dict[str, str] = {}

    for df in normalized_datasets:
        dataset_name = df.attrs.get("dataset_name")
        if not dataset_name:
            raise ValueError(
                "Each dataset must have 'dataset_name' in df.attrs")

        for _, record in df.iterrows():
            record_id = record.get("_id")
            if not record_id:
                raise ValueError("Each record must have '_id' column")

            id_to_record[record_id] = record
            id_to_dataset[record_id] = dataset_name

    # Build graph of correspondences
    graph = defaultdict(set)
    for _, corr in correspondences.iterrows():
        id1, id2 = corr["id1"], corr["id2"]
        graph[id1].add(id2)
        graph[id2].add(id1)

    # Find connected components using DFS
    visited = set()
    groups = []
    group_counter = 0

    def dfs(node_id: str, component: Set[str]):
        """Depth-first search to find connected component."""
        if node_id in visited:
            return
        visited.add(node_id)
        component.add(node_id)
        for neighbor in graph.get(node_id, set()):
            dfs(neighbor, component)

    # Process all nodes that appear in correspondences
    all_correspondence_ids = set()
    for _, corr in correspondences.iterrows():
        all_correspondence_ids.update([corr["id1"], corr["id2"]])

    for record_id in all_correspondence_ids:
        if record_id not in visited:
            component = set()
            dfs(record_id, component)

            if component:
                group = RecordGroup(group_id=f"group_{group_counter}")
                group_counter += 1

                for rid in component:
                    if rid in id_to_record:
                        group.add_record(id_to_record[rid], id_to_dataset[rid])

                groups.append(group)

    # Add singleton groups for records not in any correspondence
    all_record_ids = set(id_to_record.keys())
    unmatched_ids = all_record_ids - all_correspondence_ids

    for record_id in unmatched_ids:
        group = RecordGroup(group_id=f"singleton_{record_id}")
        group.add_record(id_to_record[record_id], id_to_dataset[record_id])
        groups.append(group)

    logger.info(
        f"Created {len(groups)} record groups from {len(correspondences)} correspondences")
    logger.info(f"Groups: {len([g for g in groups if len(g.records) > 1])} multi-record, "
                f"{len([g for g in groups if len(g.records) == 1])} singleton")

    return groups


def apply_schema_correspondences(
    groups: List[RecordGroup],
    schema_correspondences: pd.DataFrame,
) -> List[RecordGroup]:
    """Apply schema correspondences to align attributes across datasets.

    Parameters
    ----------
    groups : List[RecordGroup]
        List of record groups to process.
    schema_correspondences : pd.DataFrame
        Schema correspondences with columns like 'source_attribute', 'target_attribute', 
        'dataset_name' to map dataset-specific columns to canonical attributes.

    Returns
    -------
    List[RecordGroup]
        Modified record groups with aligned schemas.
    """
    logger = logging.getLogger(__name__)

    # Build mapping from (dataset, source_attr) -> target_attr
    schema_map = {}
    for _, corr in schema_correspondences.iterrows():
        key = (corr.get("dataset_name"), corr.get("source_attribute"))
        schema_map[key] = corr.get("target_attribute")

    modified_groups = []

    for group in groups:
        new_records = []

        for record in group.records:
            record_id = record.get("_id", "unknown")
            dataset_name = group.source_datasets.get(record_id)

            # Create new record with mapped attributes
            new_record = record.copy()

            for source_attr in record.index:
                target_attr = schema_map.get((dataset_name, source_attr))
                if target_attr and target_attr != source_attr:
                    # Rename attribute
                    new_record[target_attr] = new_record[source_attr]
                    new_record = new_record.drop(source_attr)

            new_records.append(new_record)

        # Create new group with modified records
        new_group = RecordGroup(
            group_id=group.group_id,
            records=new_records,
            source_datasets=group.source_datasets.copy()
        )
        modified_groups.append(new_group)

    logger.info(f"Applied schema correspondences to {len(groups)} groups")
    return modified_groups


class DataFusionEngine:
    """Engine for executing data fusion strategies on record groups.

    Parameters
    ----------
    strategy : DataFusionStrategy
        The fusion strategy to use.
    """

    def __init__(self, strategy: DataFusionStrategy, *, debug: bool = False, debug_file: Optional[Union[str, Path]] = None, debug_format: str = "text"):
        self.strategy = strategy
        self._logger = logging.getLogger(__name__)
        self._debug_enabled = bool(debug)
        self._debug_format = debug_format if debug_format in {"text", "json"} else "text"
        # Default debug file if enabled but not provided
        self._debug_file: Optional[Path] = None
        if self._debug_enabled:
            default_name = "fusion_debug.jsonl" if self._debug_format == "json" else "fusion_debug.log"
            path = Path(debug_file) if debug_file is not None else Path(default_name)
            self._debug_file = path
            # Ensure we start with a fresh file per engine instance
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("w", encoding="utf-8") as f:
                    ts = time.strftime("%Y-%m-%d %H:%M:%S")
                    if self._debug_format == "text":
                        f.write(f"=== Fusion Debug Log ===\n")
                        f.write(f"Timestamp: {ts}\n")
                        f.write(f"Strategy: {self.strategy.name}\n")
                        f.write("\n")
                    else:
                        header = {
                            "type": "header",
                            "timestamp": ts,
                            "strategy": self.strategy.name,
                            "format": "jsonl",
                        }
                        f.write(json.dumps(header, ensure_ascii=False) + "\n")
            except Exception as e:
                self._logger.warning(f"Could not initialize debug log file '{path}': {e}")
                self._debug_file = None

    # Internal: format and emit a debug block
    def _emit_debug(self, entry: Dict[str, Any]) -> None:
        if not self._debug_enabled or self._debug_file is None:
            return
        try:
            with self._debug_file.open("a", encoding="utf-8") as f:
                if self._debug_format == "json":
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                else:
                    block = self._format_debug_block(entry)
                    f.write(block)
        except Exception as e:
            # Never let debug writing break fusion
            self._logger.debug(f"Failed to write debug block: {e}")

    @staticmethod
    def _format_value(val: Any, max_len: int = 200) -> str:
        s = repr(val)
        if len(s) > max_len:
            return s[: max_len - 3] + "..."
        return s

    def _format_debug_block(self, entry: Dict[str, Any]) -> str:
        gid = entry.get("group_id", "?")
        attr = entry.get("attribute", "?")
        resolver = entry.get("conflict_resolution_function", "?")
        inputs = entry.get("inputs", [])
        kwargs = entry.get("resolver_kwargs", {})
        output = entry.get("output", {})
        error = entry.get("error")

        datasets = []
        for i in inputs:
            ds = i.get("dataset", "unknown")
            if ds not in datasets:
                datasets.append(ds)

        lines = []
        lines.append(f"--- Group {gid} | Attribute '{attr}' ---\n")
        lines.append(f"Conflict resolution function: {resolver}\n")
        if datasets:
            lines.append(f"Input datasets: {', '.join(datasets)}\n")
        lines.append("Inputs:\n")
        if not inputs:
            lines.append("  (none)\n")
        else:
            for i in inputs:
                rid = i.get("record_id", "?")
                ds = i.get("dataset", "unknown")
                val = self._format_value(i.get("value"))
                lines.append(f"  - {rid} [{ds}]: {val}\n")
        if kwargs:
            lines.append(f"Function kwargs: {kwargs}\n")
        out_val = self._format_value(output.get("value"))
        conf = output.get("confidence")
        meta = output.get("metadata")
        lines.append(f"Output: value={out_val}, confidence={conf}, metadata={meta}\n")
        if error:
            lines.append(f"Error: {error}\n")
        lines.append("\n")
        return "".join(lines)

    def run(
        self,
        datasets: List[pd.DataFrame],
        correspondences: pd.DataFrame,
        schema_correspondences: Optional[pd.DataFrame] = None,
        id_column: Optional[Union[str, Dict[str, str]]] = None,
        include_singletons: bool = True,
    ) -> pd.DataFrame:
        """Run the data fusion process.

        Parameters
        ----------
        datasets : List[pd.DataFrame]
            List of input datasets to fuse.
        correspondences : pd.DataFrame
            Record correspondences (CorrespondenceSet format).
        schema_correspondences : Optional[pd.DataFrame]
            Optional schema correspondences for attribute alignment.
        id_column : Optional[Union[str, Dict[str, str]]]
            Identifier column name(s). If a string, the same column is used for
            all datasets. If a dict, it should map dataset name -> ID column
            for that dataset. When provided, each dataset is normalized to have
            a string `_id` column derived from the specified column. If not
            provided, the engine will use an existing `_id` column if present,
            otherwise fall back to `id`.
        include_singletons : bool, default True
            Whether to include singleton groups (records not referenced in any
            correspondence) in the fused output. Set to False to return only
            records that participate in at least one correspondence (group size > 1).

        Returns
        -------
        pd.DataFrame
            The fused dataset.
        """
        self._logger.info(
            f"Starting data fusion with strategy '{self.strategy.name}'")
        start_time = time.time()

        # Normalize datasets to ensure `_id` is present and string-typed
        normalized_datasets: List[pd.DataFrame] = []
        for df in datasets:
            dataset_name = df.attrs.get("dataset_name")
            if not dataset_name:
                raise ValueError(
                    "Each dataset must have 'dataset_name' in df.attrs")
            # Determine which column to use for IDs
            selected_id_col: Optional[str] = None
            if isinstance(id_column, dict):
                selected_id_col = id_column.get(dataset_name)
            elif isinstance(id_column, str):
                selected_id_col = id_column

            # Create a shallow copy to avoid mutating caller's DataFrame
            df_copy = df.copy()
            # Ensure attrs are preserved on the copy (pandas may drop or not deep-copy attrs)
            try:
                df_copy.attrs = dict(df.attrs)
            except Exception:
                pass

            if selected_id_col:
                if selected_id_col not in df_copy.columns:
                    raise ValueError(
                        f"ID column '{selected_id_col}' not found in dataset '{dataset_name}'"
                    )
                df_copy["_id"] = df_copy[selected_id_col].astype(str)
            elif "_id" in df_copy.columns:
                df_copy["_id"] = df_copy["_id"].astype(str)
            elif "id" in df_copy.columns:
                df_copy["_id"] = df_copy["id"].astype(str)
            else:
                raise ValueError(
                    f"Dataset '{dataset_name}' lacks an ID column. Provide 'id_column' or include '_id'/'id'."
                )

            # Warn if id_column mapping did not specify this dataset explicitly
            if isinstance(id_column, dict) and dataset_name not in id_column:
                self._logger.warning(
                    f"id_column mapping has no entry for dataset '{dataset_name}'. Using existing '_id' or 'id' column."
                )

            normalized_datasets.append(df_copy)

        # Normalize correspondences ID types to strings for consistent matching
        normalized_correspondences = correspondences.copy()
        if "id1" not in normalized_correspondences.columns or "id2" not in normalized_correspondences.columns:
            raise ValueError(
                "Correspondences must contain 'id1' and 'id2' columns")
        normalized_correspondences["id1"] = normalized_correspondences["id1"].astype(
            str)
        normalized_correspondences["id2"] = normalized_correspondences["id2"].astype(
            str)

        # Diagnostics: measure how many correspondence IDs match dataset IDs
        known_ids: Set[str] = set()
        for df in normalized_datasets:
            if "_id" in df.columns:
                known_ids.update(df["_id"].astype(str).tolist())
        corr_ids: Set[str] = set(normalized_correspondences["id1"]) | set(
            normalized_correspondences["id2"])
        matched_ids = corr_ids & known_ids
        unmatched_ids = corr_ids - known_ids
        self._logger.info(
            f"Correspondence ID coverage: matched {len(matched_ids)} of {len(corr_ids)} unique IDs"
        )
        if len(corr_ids) > 0 and (len(matched_ids) / len(corr_ids)) < 0.5:
            self._logger.warning(
                "Less than 50% of correspondence IDs were found in the input datasets. "
                "Check ID normalization/mapping (id_column) and correspondence ID values."
            )

        # Build record groups
        groups = build_record_groups_from_correspondences(
            normalized_datasets, normalized_correspondences)

        # Optionally filter out singleton groups (unmatched records)
        if not include_singletons:
            groups = [g for g in groups if len(g.records) > 1]

        # Apply schema correspondences if provided
        if schema_correspondences is not None:
            groups = apply_schema_correspondences(
                groups, schema_correspondences)

        # Fuse each group
        fused_records = []
        for group in groups:
            fused_record = self._fuse_group(group)
            if fused_record is not None:
                fused_records.append(fused_record)

        # Create result DataFrame
        result = pd.DataFrame(fused_records)

        # Add metadata
        result.attrs["fusion_strategy"] = self.strategy.name
        result.attrs["num_input_datasets"] = len(normalized_datasets)
        result.attrs["num_correspondences"] = len(correspondences)
        result.attrs["num_groups"] = len(groups)

        self._logger.info(
            f"Fusion complete: {len(result)} records from {len(groups)} groups")
        self._logger.info(
            f"Fusion time: {time.time() - start_time:.2f} seconds")
        return result

    def _fuse_group(self, group: RecordGroup) -> Optional[Dict]:
        """Fuse a single record group.

        Parameters
        ----------
        group : RecordGroup
            The record group to fuse.

        Returns
        -------
        Optional[Dict]
            The fused record as a dictionary, or None if fusion failed.
        """
        if not group.records:
            return None

        # If only one record, return it with minimal processing
        if len(group.records) == 1:
            record = group.records[0].to_dict()
            record["_fusion_group_id"] = group.group_id
            record["_fusion_sources"] = list(
                group.source_datasets.values())[:1]
            record["_fusion_confidence"] = 1.0
            return record

        # Get all attributes present in this group
        all_attributes = group.get_all_attributes()

        # Create fusion context
        # Build a trust map from dataset attrs/provenance so resolvers can use it
        try:
            trust_map = extract_source_trust_scores(normalized_datasets)
        except Exception:
            trust_map = {}
        context = FusionContext(
            group_id=group.group_id,
            attribute="",  # Will be set per attribute
            source_datasets=group.source_datasets,
            timestamp=pd.Timestamp.now(),
            metadata={"trust_map": trust_map},
            debug=self._debug_enabled,
            debug_emit=self._emit_debug,
        )

        fused_record = {
            "_id": f"fused_{group.group_id}",
            "_fusion_group_id": group.group_id,
            "_fusion_sources": list(set(group.source_datasets.values())),
        }

        # Track overall confidence and metadata
        attribute_confidences = []
        fusion_metadata = {}

        # Fuse each attribute
        for attribute in all_attributes:
            if attribute.startswith("_fusion_"):
                continue  # Skip our own metadata attributes

            context.attribute = attribute

            # Check if we have a specific fuser for this attribute
            fuser = self.strategy.get_attribute_fuser(attribute)

            if fuser:
                # Use registered fuser
                self._logger.debug(f"Fusing attribute '{attribute}' for group '{group.group_id}' using fuser: {fuser.resolver.__name__}")
                result = fuser.fuse(group.records, context)
                fused_record[attribute] = result.value
                attribute_confidences.append(result.confidence)
                fusion_metadata[f"{attribute}_rule"] = result.rule_used
                fusion_metadata[f"{attribute}_sources"] = list(result.sources)
                self._logger.debug(f"  Fused '{attribute}': {repr(result.value)} (confidence: {result.confidence:.3f})")
            else:
                # Default fusion: prefer non-null values, first available
                self._logger.debug(f"Fusing attribute '{attribute}' for group '{group.group_id}' using default (first_non_null)")
                values = []
                for record in group.records:
                    value = record.get(attribute)
                    if _is_valid_value(value):
                        values.append(value)

                if values:
                    fused_record[attribute] = values[0]  # Take first non-null
                    attribute_confidences.append(0.5)  # Default confidence
                    fusion_metadata[f"{attribute}_rule"] = "first_non_null"
                    self._logger.debug(f"  Fused '{attribute}': {repr(values[0])} (default, confidence: 0.5)")
                    # Emit debug block for default resolver if enabled
                    if self._debug_enabled and self._debug_file is not None:
                        try:
                            inputs = []
                            for rec in group.records:
                                val = rec.get(attribute)
                                if _is_valid_value(val):
                                    rid = rec.get("_id", "unknown")
                                    inputs.append({
                                        "record_id": rid,
                                        "dataset": group.source_datasets.get(rid, "unknown"),
                                        "value": val,
                                    })
                            self._emit_debug({
                                "group_id": group.group_id,
                                "attribute": attribute,
                                "conflict_resolution_function": "first_non_null",
                                "inputs": inputs,
                                "resolver_kwargs": {},
                                "output": {
                                    "value": values[0],
                                    "confidence": 0.5,
                                    "metadata": {},
                                },
                                "error": None,
                            })
                        except Exception:
                            pass
                else:
                    fused_record[attribute] = None
                    attribute_confidences.append(0.0)
                    fusion_metadata[f"{attribute}_rule"] = "no_value"
                    self._logger.debug(f"  Fused '{attribute}': None (no values available)")

        # Calculate overall confidence
        if attribute_confidences:
            fused_record["_fusion_confidence"] = sum(
                attribute_confidences) / len(attribute_confidences)
        else:
            fused_record["_fusion_confidence"] = 0.0

        # Add fusion metadata
        fused_record["_fusion_metadata"] = fusion_metadata

        return fused_record
