"""Row-level provenance tracking for knob augmentations.

The provenance schema follows ``knobs/cross_cutting.md`` §"Per-value
provenance (mandatory)":

    (entity_id, source, attribute, original_value, new_value,
     transform_fn, transform_params, knob, level)

Entity-scoped or column-scoped records omit the ``attribute`` /
``original_value`` / ``new_value`` fields as appropriate.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


PROVENANCE_COLUMNS: list[str] = [
    "entity_id",
    "source",
    "attribute",
    "original_value",
    "new_value",
    "transform_fn",
    "transform_params",
    "knob",
    "level",
]


@dataclass
class ProvenanceRow:
    """A single provenance record.

    Parameters
    ----------
    entity_id : str
        Identifier of the entity being augmented.
    source : str
        Source dataset name (e.g. ``"dbpedia"``).
    attribute : str
        Column/attribute being modified. Empty string for entity-scoped ops.
    original_value : str
        Value before augmentation. Empty string for entity-scoped ops.
    new_value : str
        Value after augmentation. Empty string for entity-scoped ops.
    transform_fn : str
        Name of the transform function applied (e.g. ``"typo_insert"``).
    transform_params : str
        JSON-encoded parameters for the transform.
    knob : int
        Knob number (1-10).
    level : str
        Difficulty level (``"easy"``, ``"medium"``, ``"hard"``).
    """

    entity_id: str
    source: str
    attribute: str
    original_value: str
    new_value: str
    transform_fn: str
    transform_params: str
    knob: int
    level: str

    def as_dict(self) -> dict[str, Any]:
        """Return the row as an ordered dictionary matching the CSV schema."""
        return {
            "entity_id": self.entity_id,
            "source": self.source,
            "attribute": self.attribute,
            "original_value": self.original_value,
            "new_value": self.new_value,
            "transform_fn": self.transform_fn,
            "transform_params": self.transform_params,
            "knob": self.knob,
            "level": self.level,
        }


class ProvenanceLog:
    """Accumulates provenance rows and flushes them to CSV.

    Parameters
    ----------
    knob : int
        Knob number this log belongs to.
    level : str
        Difficulty level (``"easy"``, ``"medium"``, ``"hard"``).

    Examples
    --------
    >>> log = ProvenanceLog(knob=6, level="hard")
    >>> log.append(
    ...     entity_id="dbpedia_42",
    ...     source="dbpedia",
    ...     attribute="name",
    ...     original_value="Apple Inc.",
    ...     new_value="Aple Inc.",
    ...     transform_fn="typo_insert",
    ...     transform_params={"error_rate": 0.05},
    ... )
    >>> log.flush(Path("output/provenance/knob_06.csv"))
    """

    def __init__(self, knob: int, level: str) -> None:
        self.knob = knob
        self.level = level
        self._rows: list[ProvenanceRow] = []

    def __len__(self) -> int:
        return len(self._rows)

    def append(
        self,
        *,
        entity_id: str,
        source: str,
        attribute: str = "",
        original_value: str = "",
        new_value: str = "",
        transform_fn: str,
        transform_params: dict[str, Any] | str = "",
    ) -> None:
        """Append a single provenance row.

        Parameters
        ----------
        entity_id : str
            Identifier of the entity being augmented.
        source : str
            Source dataset name.
        attribute : str
            Column name. Empty for entity-scoped operations.
        original_value : str
            Pre-augmentation value. Empty for entity-scoped operations.
        new_value : str
            Post-augmentation value. Empty for entity-scoped operations.
        transform_fn : str
            Name of the transform function.
        transform_params : dict or str
            Parameters dict (will be JSON-encoded) or pre-encoded string.
        """
        if isinstance(transform_params, dict):
            params_str = json.dumps(transform_params, ensure_ascii=False)
        else:
            params_str = transform_params

        self._rows.append(
            ProvenanceRow(
                entity_id=str(entity_id),
                source=str(source),
                attribute=str(attribute),
                original_value=str(original_value),
                new_value=str(new_value),
                transform_fn=str(transform_fn),
                transform_params=params_str,
                knob=self.knob,
                level=self.level,
            )
        )

    def flush(self, path: Path, *, append: bool = False) -> int:
        """Write accumulated rows to a CSV file.

        Parameters
        ----------
        path : Path
            Output CSV path. Parent directories are created if needed.
        append : bool, default False
            If True, append to an existing file (without re-writing the
            header). If False, overwrite.

        Returns
        -------
        int
            Number of rows written.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        write_header = not append or not path.exists() or path.stat().st_size == 0

        n = len(self._rows)
        with open(path, mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=PROVENANCE_COLUMNS)
            if write_header:
                writer.writeheader()
            for row in self._rows:
                writer.writerow(row.as_dict())

        self._rows.clear()
        return n

    @staticmethod
    def read(path: Path) -> pd.DataFrame:
        """Read a provenance CSV back into a DataFrame.

        Parameters
        ----------
        path : Path
            Path to the provenance CSV.

        Returns
        -------
        pandas.DataFrame
            DataFrame with columns matching ``PROVENANCE_COLUMNS``.
        """
        return pd.read_csv(
            path, dtype={"knob": int, "level": str}, keep_default_na=False
        )

    @staticmethod
    def merge(paths: list[Path], output: Path) -> pd.DataFrame:
        """Merge multiple provenance CSVs into one.

        Parameters
        ----------
        paths : list of Path
            Input provenance CSV files.
        output : Path
            Output path for the merged CSV.

        Returns
        -------
        pandas.DataFrame
            The merged DataFrame.
        """
        frames = [ProvenanceLog.read(p) for p in paths if p.exists()]
        if not frames:
            merged = pd.DataFrame(columns=PROVENANCE_COLUMNS)
        else:
            merged = pd.concat(frames, ignore_index=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(output, index=False)
        return merged
