"""
Helpers for deriving per-cell provenance from PyDI fusion outputs.

The fusion engine ([PyDI.fusion.engine.DataFusionEngine][]) always
writes per-attribute provenance into each fused row's
``_fusion_metadata`` column. The keys used are:

* ``_fusion_metadata[f"{attribute}_rule"]``    — conflict-resolution rule fired
* ``_fusion_metadata[f"{attribute}_sources"]`` — list of source record IDs
                                                  that contributed the winning value
* ``_fusion_metadata[f"{attribute}_inputs"]``  — full per-source input record

This module reshapes those per-row dicts into the long-form
``(cluster_id, attribute, source_ids)`` DataFrame the
end-to-end panel consumes as ``cell_provenance_pipe``. That allows
the source-attribution and synthesis-rate metrics (§3.7.2, §3.7.7
in the panel) to fire automatically on any pipeline that used the
fusion engine — no separate "opt-in" toggle exists or is needed.
"""

from __future__ import annotations

import logging
from typing import Any, List, Sequence

import pandas as pd

logger = logging.getLogger(__name__)


def build_cell_provenance_from_fused(
    pipe_fused: pd.DataFrame,
    *,
    pipe_id_column: str = "_fusion_group_id",
    attribute_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Reshape ``_fusion_metadata`` into long-form cell provenance.

    Parameters
    ----------
    pipe_fused : DataFrame
        Output of :func:`PyDI.fusion.engine.DataFusionEngine.run`.
        Must contain a ``_fusion_metadata`` column. Rows where the
        metadata is missing are silently skipped.
    pipe_id_column : str, default ``"_fusion_group_id"``
        Cluster id column on the fused frame. Used to identify the
        cluster each cell belongs to.
    attribute_columns : sequence of str, optional
        Restrict the output to these attributes. When empty (the
        default), the function emits a row for every
        ``f"{attr}_sources"`` key it finds in ``_fusion_metadata``.

    Returns
    -------
    DataFrame
        Long-form ``(cluster_id, attribute, source_ids)``. The
        ``source_ids`` column carries a ``list[str]`` per row. The
        DataFrame is empty when ``_fusion_metadata`` is missing or
        carries no source keys.
    """
    if pipe_fused.empty or "_fusion_metadata" not in pipe_fused.columns:
        return pd.DataFrame(columns=["cluster_id", "attribute", "source_ids"])

    allowed = set(attribute_columns) if attribute_columns else None
    rows: List[dict] = []

    for _, row in pipe_fused.iterrows():
        cluster_id = row.get(pipe_id_column)
        if cluster_id is None or (
            isinstance(cluster_id, float) and pd.isna(cluster_id)
        ):
            continue
        metadata = row.get("_fusion_metadata")
        if not isinstance(metadata, dict):
            continue
        for key, value in metadata.items():
            if not isinstance(key, str) or not key.endswith("_sources"):
                continue
            attribute = key[: -len("_sources")]
            if allowed is not None and attribute not in allowed:
                continue
            if value is None:
                continue
            source_ids = _normalize_source_ids(value)
            if not source_ids:
                continue
            rows.append(
                {
                    "cluster_id": str(cluster_id),
                    "attribute": attribute,
                    "source_ids": source_ids,
                }
            )

    if not rows:
        logger.debug(
            "build_cell_provenance_from_fused: no %s_sources entries found in _fusion_metadata",
            "attribute",
        )
    return pd.DataFrame(rows, columns=["cluster_id", "attribute", "source_ids"])


def _normalize_source_ids(value: Any) -> List[str]:
    """Coerce a metadata value into a list of source-id strings."""
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple, set)):
        out: List[str] = []
        for v in value:
            if v is None:
                continue
            try:
                if pd.isna(v):
                    continue
            except (TypeError, ValueError):
                pass
            out.append(str(v))
        return out
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass
    return [str(value)]
