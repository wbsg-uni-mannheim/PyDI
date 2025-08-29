"""
Data fusion tools for PyDI.

This module defines classes and utilities for merging multiple datasets
based on record correspondences and conflict resolution rules. The
provided implementation is simplistic and intended as a template for
more sophisticated fusion strategies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import pandas as pd


@dataclass
class FusionRule:
    """Represent a fusion rule for a single attribute.

    Parameters
    ----------
    strategy : str
        Name of the built‑in strategy (e.g., ``"longest"``, ``"most_recent"``).
    function : callable, optional
        A custom function that takes a list of values and returns a fused
        value. If provided, it overrides ``strategy``.
    """

    strategy: str
    function: Optional[Callable[[List[Any]], Any]] = None


class DataFuser:
    """Fuse multiple datasets into a single DataFrame using correspondences.

    The current implementation is deliberately simple: it merges rows
    specified in the correspondences and applies fusion rules per
    attribute. Rows without correspondences are appended as‑is. Users
    should extend this class to handle more complex conflict resolution
    logic and provenance tracking.
    """

    def fuse(
        self,
        datasets: List[pd.DataFrame],
        correspondences: "CorrespondenceSet",
        *,
        rules: Dict[str, FusionRule],
    ) -> pd.DataFrame:
        # Build a mapping from id to row for quick lookup
        id_to_row: Dict[str, pd.Series] = {}
        for df in datasets:
            id_to_row.update({row["_id"]: row for _, row in df.iterrows()})
        used_ids = set()
        fused_records: List[Dict[str, Any]] = []
        # Fuse matched pairs
        for _, corr in correspondences.to_dataframe().iterrows():
            id1, id2 = corr["id1"], corr["id2"]
            used_ids.add(id1)
            used_ids.add(id2)
            row1 = id_to_row.get(id1)
            row2 = id_to_row.get(id2)
            if row1 is None or row2 is None:
                continue
            fused: Dict[str, Any] = {}
            columns = set(row1.index) | set(row2.index)
            for col in columns:
                val1 = row1.get(col)
                val2 = row2.get(col)
                if col in rules:
                    rule = rules[col]
                    values = [v for v in [val1, val2] if pd.notna(v)]
                    if rule.function:
                        fused[col] = rule.function(values)
                    else:
                        strategy = rule.strategy
                        if strategy == "longest":
                            fused[col] = max(values, key=lambda x: len(
                                str(x))) if values else None
                        elif strategy == "shortest":
                            fused[col] = min(values, key=lambda x: len(
                                str(x))) if values else None
                        elif strategy == "most_recent":
                            # assumes values are comparable (e.g., datetime or sortable strings)
                            fused[col] = max(values) if values else None
                        elif strategy == "union":
                            fused[col] = list({val for val in values})
                        elif strategy == "majority":
                            from collections import Counter

                            fused[col] = Counter(values).most_common(1)[
                                0][0] if values else None
                        else:
                            fused[col] = values[0] if values else None
                else:
                    # default: prefer non-null, fall back to any
                    fused[col] = val1 if pd.notna(val1) else val2
            # assign a new fused ID
            fused["_id"] = f"fused_{id1}_{id2}"
            fused_records.append(fused)
        # Append unmatched rows
        for id_, row in id_to_row.items():
            if id_ not in used_ids:
                fused_records.append(row.to_dict())
        return pd.DataFrame(fused_records)
