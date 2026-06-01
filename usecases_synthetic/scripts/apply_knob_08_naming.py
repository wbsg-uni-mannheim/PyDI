#!/usr/bin/env python3
"""Apply Knob 08 — Schema Naming Divergence.

Renames source DataFrame column headers according to a per-domain rename
table and the requested difficulty level. Regenerates schema matching
mapping artifacts so the SM pipeline stage sees consistent column names.

See ``knobs/knob_08_schema_naming.md`` for the full specification.

Usage
-----
::

    python usecases_synthetic/scripts/apply_knob_08_naming.py \\
        --domain companies --level easy

Inputs
------
- Source DataFrames from ``usecases/<domain>/input/data/``
- Per-domain rename YAML at ``usecases_synthetic/config/knob_08_naming/<domain>.yaml``

Outputs (under *output_dir*)
------
- Renamed source DataFrames (returned in-memory; not written to disk by
  this script — the orchestrator handles serialisation)
- SM mapping CSV at ``<output_dir>/input/schemamatching/sm_mapping.csv``
- Provenance CSV at ``<output_dir>/output/provenance/knob_08_naming.csv``
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Ensure the repo root is on sys.path so both ``usecases_synthetic.lib``
# and ``PyDI`` are importable when running the script directly.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from usecases_synthetic.lib.domain_config import (
    CONFIG_DIR,
    VALID_LEVELS,
    load_domain_config,
)
from usecases_synthetic.lib.loaders import load_domain_sources
from usecases_synthetic.lib.provenance import ProvenanceLog

logger = logging.getLogger(__name__)

# Valid transform_fn values for Knob 8 provenance rows.
VALID_TRANSFORM_FNS = frozenset(
    {
        "rename_descriptive",
        "rename_abbreviated",
        "rename_cryptic",
        "rename_anonymize",
        "rename_up_from_mapping",
    }
)

# Rung name -> provenance transform_fn.
_RUNG_TO_FN: dict[str, str] = {
    "descriptive": "rename_descriptive",
    "abbreviated": "rename_abbreviated",
    "cryptic": "rename_cryptic",
    "anonymized": "rename_anonymize",
}


# ---- Config loading -------------------------------------------------------


def load_knob_08_config(domain: str) -> dict[str, Any]:
    """Load the Knob 08 rename config for *domain*.

    Parameters
    ----------
    domain : str
        Domain name (e.g. ``"companies"``).

    Returns
    -------
    dict
        Parsed YAML with keys ``rename_table``, ``level_assignments``,
        ``sm_mapping``.

    Raises
    ------
    FileNotFoundError
        If the per-domain YAML does not exist.
    """
    path = CONFIG_DIR / "knob_08_naming" / f"{domain}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Knob 08 rename config not found: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _infer_baseline_rung(
    original_col: str,
    rungs: dict[str, str],
) -> str:
    """Determine the baseline rung of *original_col*.

    The baseline is the rung whose value equals the original column name.
    If no exact match exists (the original name is not in any rung), we
    treat it as sitting between rungs and return ``"original"``.

    Parameters
    ----------
    original_col : str
        The original column name in the source DataFrame.
    rungs : dict
        ``{descriptive: str, abbreviated: str, cryptic: str,
        anonymized: str}`` from the rename table.

    Returns
    -------
    str
        One of ``"descriptive"``, ``"abbreviated"``, ``"cryptic"``,
        ``"anonymized"``, or ``"original"`` if no rung matches.
    """
    for rung_name, rung_value in rungs.items():
        if rung_value == original_col:
            return rung_name
    return "original"


# ---- Core logic -----------------------------------------------------------


def apply_knob_08(
    domain: str,
    level: Literal["easy", "medium", "hard"],
    sources: dict[str, pd.DataFrame],
    config: dict[str, Any],
    per_source_override: dict[str, str] | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """Rename source columns per the Knob 08 config at *level*.

    Parameters
    ----------
    domain : str
        Domain name.
    level : {"easy", "medium", "hard"}
        Difficulty level.
    sources : dict[str, DataFrame]
        Source DataFrames keyed by source name.
    config : dict
        Parsed Knob 08 YAML (from :func:`load_knob_08_config`).
    per_source_override : dict or None
        Optional ``{source: rung}`` overrides for individual sources.

    Returns
    -------
    renamed_sources : dict[str, DataFrame]
        DataFrames with renamed columns.  ``attrs["dataset_name"]`` is
        preserved from the input.
    sm_mapping_df : DataFrame
        Regenerated SM mapping with columns ``source_dataset``,
        ``source_column``, ``target_dataset``, ``target_column``, ``score``.
    provenance_df : DataFrame
        Provenance log as a DataFrame.
    """
    if level not in VALID_LEVELS:
        raise ValueError(f"Invalid level: {level!r}. Valid: {VALID_LEVELS}")

    rename_table: dict[str, dict[str, dict[str, str]]] = config["rename_table"]
    level_assignments: dict[str, dict[str, str]] = config["level_assignments"]
    sm_ground_truth: dict[str, dict[str, str]] = config["sm_mapping"]

    overrides = per_source_override or {}
    prov = ProvenanceLog(knob=8, level=level)

    renamed: dict[str, pd.DataFrame] = {}
    sm_rows: list[dict[str, str | float]] = []

    # Build set of id columns per source from sm_mapping. Id columns
    # MUST NOT be renamed — downstream stages (EM correspondences,
    # fusion engine id_column lookup) reference records by id.
    id_columns_by_source: dict[str, set[str]] = {
        src: {col for col, target in mapping.items() if target == "id"}
        for src, mapping in sm_ground_truth.items()
    }
    for src, id_cols in id_columns_by_source.items():
        rt = rename_table.get(src, {})
        bad = id_cols & set(rt.keys())
        if bad:
            raise ValueError(
                f"Knob 08 config error: source {src!r} has id column(s) "
                f"{sorted(bad)} in rename_table. Id columns "
                f"(sm_mapping target == 'id') must never be renamed. "
                f"Remove them from rename_table in the domain config."
            )

    for source_name, df in sources.items():
        if source_name not in rename_table:
            logger.warning(
                "Source %r not in rename table for domain %r — identity pass",
                source_name,
                domain,
            )
            renamed[source_name] = df.copy()
            continue

        target_rung = overrides.get(
            source_name,
            level_assignments[level][source_name],
        )
        source_table = rename_table[source_name]
        source_sm = sm_ground_truth.get(source_name, {})
        id_cols = id_columns_by_source.get(source_name, set())

        col_rename_map: dict[str, str] = {}

        for original_col in df.columns:
            if original_col in id_cols:
                # Defensive: never rename id columns.
                logger.debug(
                    "Column %r in source %r is an id column — kept as-is",
                    original_col,
                    source_name,
                )
                continue
            if original_col not in source_table:
                # Column not in rename table — keep as-is (e.g. added by
                # earlier knobs or unknown columns).
                logger.debug(
                    "Column %r in source %r not in rename table — kept as-is",
                    original_col,
                    source_name,
                )
                continue

            rungs = source_table[original_col]
            new_col = rungs[target_rung]
            baseline_rung = _infer_baseline_rung(original_col, rungs)

            if new_col == original_col:
                # Identity rename — no provenance row needed.
                continue

            col_rename_map[original_col] = new_col

            # Determine transform_fn.
            if target_rung == "descriptive" and original_col in source_sm:
                transform_fn = "rename_up_from_mapping"
                oracle = "sm_mapping"
            else:
                transform_fn = _RUNG_TO_FN[target_rung]
                oracle = "rename_table"

            prov.append(
                entity_id="",
                source=source_name,
                attribute=original_col,
                original_value=original_col,
                new_value=new_col,
                transform_fn=transform_fn,
                transform_params={
                    "baseline_rung": baseline_rung,
                    "target_rung": target_rung,
                    "oracle": oracle,
                },
            )

        # Apply renames.
        new_df = df.rename(columns=col_rename_map)
        # Preserve attrs.
        new_df.attrs = df.attrs.copy()
        renamed[source_name] = new_df

        # Build SM mapping rows for this source (using renamed column names).
        for orig_col, target_col in source_sm.items():
            # The source column in the SM mapping is the *renamed* name.
            renamed_col = col_rename_map.get(orig_col, orig_col)
            sm_rows.append(
                {
                    "source_dataset": source_name,
                    "source_column": renamed_col,
                    "target_dataset": domain,
                    "target_column": target_col,
                    "score": 1.0,
                }
            )

    sm_mapping_df = pd.DataFrame(
        sm_rows,
        columns=[
            "source_dataset",
            "source_column",
            "target_dataset",
            "target_column",
            "score",
        ],
    )

    # Build provenance DataFrame from the log's internal rows.
    from usecases_synthetic.lib.provenance import PROVENANCE_COLUMNS

    if len(prov) > 0:
        provenance_df = pd.DataFrame(
            [row.as_dict() for row in prov._rows],
            columns=PROVENANCE_COLUMNS,
        )
    else:
        provenance_df = pd.DataFrame(columns=PROVENANCE_COLUMNS)

    return renamed, sm_mapping_df, provenance_df


def write_outputs(
    sm_mapping_df: pd.DataFrame,
    provenance_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Write SM mapping and provenance artifacts to *output_dir*.

    Parameters
    ----------
    sm_mapping_df : DataFrame
        Regenerated schema matching mapping.
    provenance_df : DataFrame
        Provenance log DataFrame.
    output_dir : Path
        Variant directory root.
    """
    sm_dir = output_dir / "input" / "schemamatching"
    sm_dir.mkdir(parents=True, exist_ok=True)
    sm_mapping_df.to_csv(sm_dir / "sm_mapping.csv", index=False)
    logger.info("Wrote SM mapping to %s", sm_dir / "sm_mapping.csv")

    prov_dir = output_dir / "output" / "provenance"
    prov_dir.mkdir(parents=True, exist_ok=True)
    provenance_df.to_csv(prov_dir / "knob_08_naming.csv", index=False)
    logger.info(
        "Wrote provenance (%d rows) to %s",
        len(provenance_df),
        prov_dir / "knob_08_naming.csv",
    )


# ---- CLI -------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Apply Knob 08 — Schema Naming Divergence",
    )
    parser.add_argument(
        "--domain",
        required=True,
        help="Domain name (e.g. companies)",
    )
    parser.add_argument(
        "--level",
        required=True,
        choices=VALID_LEVELS,
        help="Difficulty level",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Variant output directory (default: usecases_synthetic/output/<domain>/<level>)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    domain: str = args.domain
    level: str = args.level

    output_dir: Path = args.output_dir or (
        REPO_ROOT / "usecases_synthetic" / "output" / domain / level
    )

    logger.info("Knob 08: domain=%s level=%s output=%s", domain, level, output_dir)

    config = load_knob_08_config(domain)
    sources = load_domain_sources(domain)

    renamed, sm_mapping_df, provenance_df = apply_knob_08(
        domain=domain,
        level=level,
        sources=sources,
        config=config,
    )

    write_outputs(sm_mapping_df, provenance_df, output_dir)

    # Summary.
    for src_name, df in renamed.items():
        logger.info("  %s columns: %s", src_name, list(df.columns))
    logger.info("SM mapping: %d rows", len(sm_mapping_df))
    logger.info("Provenance: %d rows", len(provenance_df))


if __name__ == "__main__":
    main()
