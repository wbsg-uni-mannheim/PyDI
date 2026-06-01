"""Cell-collision coordination index.

Implements ``knobs/cross_cutting.md`` §"Cell-collision coordination":
tracks which ``(entity_id, source, attribute)`` triples have already been
touched by a prior knob, so downstream knobs can skip or apply exception
rules (e.g. K6 skips unless the cell was fabricated by K4).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .provenance import PROVENANCE_COLUMNS, ProvenanceLog

logger = logging.getLogger(__name__)


# Minimum set of columns a CSV must expose to be treated as a provenance
# file by the collision index. Diagnostic / audit CSVs in the same
# directory (e.g. ``knob_02_niche_scores.csv``,
# ``joint_values_audit.csv``) do not satisfy this contract and are
# skipped with a debug log.
_REQUIRED_PROVENANCE_COLUMNS: frozenset[str] = frozenset(
    ["entity_id", "source", "attribute", "transform_fn", "knob"]
)


class CollisionIndex:
    """Reads provenance CSVs and tracks touched cells.

    Parameters
    ----------
    provenance_dir : Path
        Directory containing per-knob provenance CSVs
        (e.g. ``output/provenance/``).
    """

    def __init__(self, provenance_dir: Path) -> None:
        self._provenance_dir = provenance_dir
        self._touched: set[tuple[str, str, str]] = set()
        self._k4_fabricated: set[tuple[str, str, str]] = set()
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Lazy-load all provenance CSVs on first access.

        Files in the provenance directory that do not expose the full
        provenance schema (e.g. diagnostic score dumps, joint audit
        tables) are skipped with a debug log. This keeps the index
        robust to co-located auxiliary artifacts written by individual
        knobs.
        """
        if self._loaded:
            return
        self._loaded = True

        if not self._provenance_dir.exists():
            return

        for csv_path in sorted(self._provenance_dir.glob("*.csv")):
            df = ProvenanceLog.read(csv_path)
            missing = _REQUIRED_PROVENANCE_COLUMNS - set(df.columns)
            if missing:
                logger.debug(
                    "CollisionIndex skipping non-provenance CSV %s "
                    "(missing columns: %s)",
                    csv_path.name,
                    sorted(missing),
                )
                continue
            for _, row in df.iterrows():
                key = (
                    str(row["entity_id"]),
                    str(row["source"]),
                    str(row["attribute"]),
                )
                self._touched.add(key)

                # Track K4-fabricated cells for K6's exception rule
                if int(row["knob"]) == 4 and "fabricat" in str(
                    row["transform_fn"]
                ).lower():
                    self._k4_fabricated.add(key)

    def is_touched(
        self, entity_id: str, source: str, attribute: str
    ) -> bool:
        """Check if a cell has been modified by any prior knob.

        Parameters
        ----------
        entity_id : str
            Entity identifier.
        source : str
            Source dataset name.
        attribute : str
            Attribute/column name.

        Returns
        -------
        bool
            True if any provenance record exists for this cell.
        """
        self._ensure_loaded()
        return (entity_id, source, attribute) in self._touched

    def is_k4_fabricated(
        self, entity_id: str, source: str, attribute: str
    ) -> bool:
        """Check if a cell was fabricated by Knob 4.

        K6 uses this to allow noise injection on fabricated cells even
        when they would otherwise be skipped due to prior modification.

        Parameters
        ----------
        entity_id : str
            Entity identifier.
        source : str
            Source dataset name.
        attribute : str
            Attribute/column name.

        Returns
        -------
        bool
            True if the cell was fabricated by K4.
        """
        self._ensure_loaded()
        return (entity_id, source, attribute) in self._k4_fabricated

    def reload(self) -> None:
        """Force a reload of provenance data from disk."""
        self._touched.clear()
        self._k4_fabricated.clear()
        self._loaded = False
        self._ensure_loaded()
