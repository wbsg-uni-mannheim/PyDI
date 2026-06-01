#!/usr/bin/env python3
"""Apply Knob 05 — Format/Unit Diversity.

Structured-format and unit rewriting for dates, numbers, currencies.
Classifies attributes by format family, draws format assignments per
(source, attr) at easy/medium or per (row, attr) at hard, applies
operators, verifies round-trip, and writes provenance.

See ``knobs/knob_05_format_unit.md`` for the full specification.

Usage
-----
::

    python usecases_synthetic/scripts/apply_knob_05_format.py \\
        --domain companies --level easy

Inputs
------
- Source DataFrames from ``usecases/<domain>/input/data/``
- Per-domain config at ``usecases_synthetic/config/knob_05_format/<domain>.yaml``
- Shared operator tables under ``usecases_synthetic/config/knob_05_format/_tables/``

Outputs (under *output_dir*)
------
- Reformatted source DataFrames (returned in-memory)
- Provenance CSV at ``<output_dir>/output/provenance/knob_05_format_unit.csv``
- Skipped-cell audit at ``<output_dir>/output/provenance/knob_05_skipped.csv``
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Ensure repo root is on sys.path.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from usecases_synthetic.lib.collision_index import CollisionIndex
from usecases_synthetic.lib.domain_config import (
    CONFIG_DIR,
    VALID_LEVELS,
    load_domain_config,
)
from usecases_synthetic.lib.format_operators import (
    _parse_date_flexible,
    _parse_number,
    format_duration,
    parse_duration,
    reconvert_currency,
    reconvert_unit,
    reformat_date,
    reformat_number,
    reformat_number_suffix,
)
from usecases_synthetic.lib.loaders import load_domain_sources
from usecases_synthetic.lib.provenance import PROVENANCE_COLUMNS, ProvenanceLog
from usecases_synthetic.lib.rate_tables import get_date_format, is_denied_date_format
from usecases_synthetic.lib.rng import make_rng

logger = logging.getLogger(__name__)

# Valid transform_fn values for Knob 5 provenance rows.
VALID_TRANSFORM_FNS = frozenset(
    {
        "reformat_date",
        "reformat_number",
        "reconvert_unit",
        "reconvert_currency",
        "relocale",
    }
)

# Skipped-cell reason codes.
SKIP_ROUNDTRIP = "roundtrip_parse_fail"
SKIP_COLLISION_PRIOR = "cell_collision_with_prior_knob"
SKIP_COLLISION_K4_FAB = "cell_collision_with_k4_fabricated"
SKIP_DENYLIST = "denylist_locale_ambiguous"
SKIP_UNPARSEABLE = "unparseable_value"

SKIPPED_COLUMNS = [
    "entity_id",
    "source",
    "attribute",
    "original_value",
    "attempted_format",
    "reason",
    "knob",
    "level",
]


# ---- Config loading ---------------------------------------------------------


def load_knob_05_config(domain: str) -> dict[str, Any]:
    """Load the Knob 05 format config for *domain*.

    Parameters
    ----------
    domain : str
        Domain name (e.g. ``"companies"``).

    Returns
    -------
    dict
        Parsed YAML with keys ``attribute_classes``,
        ``format_pools_per_level``, etc.
    """
    path = CONFIG_DIR / "knob_05_format" / f"{domain}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Knob 05 format config not found: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---- Skipped-cell audit -----------------------------------------------------


class SkippedLog:
    """Accumulates skipped-cell audit rows."""

    def __init__(self, knob: int, level: str) -> None:
        self.knob = knob
        self.level = level
        self._rows: list[dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self._rows)

    def append(
        self,
        *,
        entity_id: str,
        source: str,
        attribute: str,
        original_value: str,
        attempted_format: str,
        reason: str,
    ) -> None:
        """Record a skipped cell."""
        self._rows.append(
            {
                "entity_id": entity_id,
                "source": source,
                "attribute": attribute,
                "original_value": original_value,
                "attempted_format": attempted_format,
                "reason": reason,
                "knob": self.knob,
                "level": self.level,
            }
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Return as a DataFrame."""
        if not self._rows:
            return pd.DataFrame(columns=SKIPPED_COLUMNS)
        return pd.DataFrame(self._rows, columns=SKIPPED_COLUMNS)

    def flush(self, path: Path) -> int:
        """Write to CSV."""
        path.parent.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        n = len(self._rows)
        self._rows.clear()
        return n


# ---- Core dispatcher --------------------------------------------------------


def _get_managed_attrs(
    config: dict[str, Any],
) -> dict[str, dict[str, str]]:
    """Extract the per-source attribute-to-family mapping.

    Returns
    -------
    dict[str, dict[str, str]]
        ``{source_name: {column_name: format_family}}``.
    """
    return config.get("attribute_classes", {})


def _draw_format_assignment(
    rng: np.random.Generator,
    pool: list[str],
    consistency: str,
    n_rows: int,
) -> list[str] | str:
    """Draw format assignments per the consistency rule.

    Parameters
    ----------
    rng : Generator
        Seeded RNG.
    pool : list[str]
        Available format IDs.
    consistency : str
        ``"source"`` (one draw for all rows) or ``"row"`` (per-row draw).
    n_rows : int
        Number of rows in the source.

    Returns
    -------
    str or list[str]
        Single format ID (source consistency) or list of per-row IDs.
    """
    if consistency == "source":
        return str(rng.choice(pool))
    else:
        return [str(rng.choice(pool)) for _ in range(n_rows)]


def _draw_unit_assignment(
    rng: np.random.Generator,
    unit_pool: list[str],
    consistency: str,
    n_rows: int,
) -> list[str] | str:
    """Draw unit assignments analogous to format assignments."""
    if consistency == "source":
        return str(rng.choice(unit_pool))
    else:
        return [str(rng.choice(unit_pool)) for _ in range(n_rows)]


def apply_knob_05(
    domain: str,
    level: Literal["easy", "medium", "hard"],
    sources: dict[str, pd.DataFrame],
    config: dict[str, Any],
    collision_index: CollisionIndex | None = None,
    seed: int = 42,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """Apply Knob 05 format/unit diversity to source DataFrames.

    Parameters
    ----------
    domain : str
        Domain name.
    level : {"easy", "medium", "hard"}
        Difficulty level.
    sources : dict[str, DataFrame]
        Source DataFrames keyed by source name.
    config : dict
        Parsed Knob 05 YAML (from :func:`load_knob_05_config`).
    collision_index : CollisionIndex or None
        Cell-collision index from prior knobs. K5 defensively skips any
        cell already touched by K1 or K4 (including K4-fabricated cells),
        per ``knobs/cross_cutting.md`` § "Cell-collision coordination".
    seed : int, default 42
        Master seed.

    Returns
    -------
    reformatted_sources : dict[str, DataFrame]
        DataFrames with reformatted values. ``attrs`` preserved.
    provenance_df : DataFrame
        Provenance log.
    skipped_df : DataFrame
        Skipped-cell audit.
    """
    if level not in VALID_LEVELS:
        raise ValueError(f"Invalid level: {level!r}. Valid: {VALID_LEVELS}")

    rng = make_rng(domain, level, knob=5, master_seed=seed)
    prov = ProvenanceLog(knob=5, level=level)
    skipped = SkippedLog(knob=5, level=level)

    managed_attrs = _get_managed_attrs(config)
    format_pools = config["format_pools_per_level"][level]
    consistency = config["within_source_consistency"][level]
    id_columns = config.get("id_columns", {})

    # Unit/magnitude pools.
    unit_pools = config.get("unit_pool_per_level", {}).get(level, {})
    locale_pool = config.get("locale_pool_per_level", {}).get(level, ["en_US"])
    source_mag_ctx = config.get("source_magnitude_context", {})

    reformatted: dict[str, pd.DataFrame] = {}

    for source_name in sorted(sources.keys()):
        df = sources[source_name]
        new_df = df.copy()
        new_df.attrs = df.attrs.copy()

        source_attrs = managed_attrs.get(source_name, {})
        id_col = id_columns.get(source_name)

        for col in sorted(source_attrs.keys()):
            if col not in df.columns:
                logger.warning(
                    "Column %r not in source %r — skipping",
                    col,
                    source_name,
                )
                continue

            family = source_attrs[col]
            pool = format_pools.get(family, [])
            if not pool:
                logger.debug(
                    "No format pool for family %r at level %r — skipping %s.%s",
                    family,
                    level,
                    source_name,
                    col,
                )
                continue

            # Cast numeric columns to object so string-formatted values
            # can be assigned without FutureWarning.
            if new_df[col].dtype != object:
                new_df[col] = new_df[col].astype(object)

            # Draw format assignments.
            fmt_assignment = _draw_format_assignment(rng, pool, consistency, len(df))

            # Draw unit/currency assignments for unit-bearing classes. The
            # ``money`` branch covers currency rotation + magnitude scale
            # (existing). ``file_size`` (added 2026-05-22, plan_revision
            # §K5 follow-up) covers byte-quantity unit rotation
            # (GB↔MB↔TB↔GiB) for attributes like ``vram_gb`` /
            # ``storage_gb``. Each unit-bearing class has its own pool
            # keyed by family in ``unit_pool_per_level`` — no shared pool,
            # so a domain with both ``price`` (money) and ``vram_gb``
            # (file_size) does not conflate GBP/USD/EUR with GB/MB/GiB.
            unit_assignment: list[str] | str | None = None
            mag_assignment: list[str] | str | None = None
            if family == "money":
                money_cfg = unit_pools.get("money", {})
                currencies = money_cfg.get("currencies", ["USD"])
                magnitudes = money_cfg.get("magnitude", ["raw"])

                unit_assignment = _draw_unit_assignment(
                    rng, currencies, consistency, len(df)
                )
                mag_assignment = _draw_unit_assignment(
                    rng, magnitudes, consistency, len(df)
                )
            elif family == "file_size":
                fs_cfg = unit_pools.get("file_size", {})
                units_pool = fs_cfg.get("units", ["GB"])
                unit_assignment = _draw_unit_assignment(
                    rng, units_pool, consistency, len(df)
                )
            elif family == "rate":
                # Rate (bandwidth) class — added 2026-05-27 (step 4h cross-knob
                # expansion for products read_speed_mb_s / write_speed_mb_s).
                # Conversion uses the ``rate`` group in unit_factors.yaml
                # (bytes_per_second / KB/s / MB/s / GB/s / TB/s).
                rt_cfg = unit_pools.get("rate", {})
                units_pool = rt_cfg.get("units", ["MB/s"])
                unit_assignment = _draw_unit_assignment(
                    rng, units_pool, consistency, len(df)
                )

            # Apply transforms row by row.
            for idx in range(len(df)):
                cell_value = df.iloc[idx][col]
                entity_id = (
                    str(df.iloc[idx][id_col])
                    if id_col and id_col in df.columns
                    else str(idx)
                )

                # Skip null/empty cells.
                if pd.isna(cell_value):
                    continue
                cell_str = str(cell_value).strip()
                if not cell_str or cell_str.lower() in ("null", "nan", "none", ""):
                    continue

                # Cell-collision check. K5 defensively skips any cell
                # already touched by a prior knob (K1 or K4), including
                # K4-fabricated cells.
                if collision_index is not None:
                    if collision_index.is_touched(entity_id, source_name, col):
                        if collision_index.is_k4_fabricated(
                            entity_id, source_name, col
                        ):
                            reason = SKIP_COLLISION_K4_FAB
                        else:
                            reason = SKIP_COLLISION_PRIOR
                        skipped.append(
                            entity_id=entity_id,
                            source=source_name,
                            attribute=col,
                            original_value=cell_str,
                            attempted_format="",
                            reason=reason,
                        )
                        continue

                # Get the target format for this row.
                if isinstance(fmt_assignment, list):
                    target_fmt = fmt_assignment[idx]
                else:
                    target_fmt = fmt_assignment

                result = _apply_cell_transform(
                    cell_str=cell_str,
                    family=family,
                    target_fmt=target_fmt,
                    entity_id=entity_id,
                    source_name=source_name,
                    col=col,
                    prov=prov,
                    skipped=skipped,
                    unit_assignment=unit_assignment,
                    mag_assignment=mag_assignment,
                    source_mag_ctx=source_mag_ctx.get(source_name, {}),
                    idx=idx,
                )
                if result is not None:
                    new_df.iat[idx, df.columns.get_loc(col)] = result

        reformatted[source_name] = new_df

    # Build provenance DataFrame.
    if len(prov) > 0:
        provenance_df = pd.DataFrame(
            [row.as_dict() for row in prov._rows],
            columns=PROVENANCE_COLUMNS,
        )
    else:
        provenance_df = pd.DataFrame(columns=PROVENANCE_COLUMNS)

    skipped_df = skipped.to_dataframe()

    return reformatted, provenance_df, skipped_df


def _apply_cell_transform(
    *,
    cell_str: str,
    family: str,
    target_fmt: str,
    entity_id: str,
    source_name: str,
    col: str,
    prov: ProvenanceLog,
    skipped: SkippedLog,
    unit_assignment: list[str] | str | None,
    mag_assignment: list[str] | str | None,
    source_mag_ctx: dict[str, Any],
    idx: int,
) -> str | None:
    """Apply a single-cell format transform. Returns the new value or None."""

    if family == "date":
        return _transform_date(
            cell_str=cell_str,
            target_fmt=target_fmt,
            entity_id=entity_id,
            source_name=source_name,
            col=col,
            prov=prov,
            skipped=skipped,
        )
    elif family in ("money", "number"):
        return _transform_number(
            cell_str=cell_str,
            target_fmt=target_fmt,
            entity_id=entity_id,
            source_name=source_name,
            col=col,
            prov=prov,
            skipped=skipped,
            unit_assignment=unit_assignment,
            mag_assignment=mag_assignment,
            source_mag_ctx=source_mag_ctx,
            idx=idx,
        )
    elif family in ("file_size", "rate"):
        return _transform_file_size(
            cell_str=cell_str,
            target_fmt=target_fmt,
            entity_id=entity_id,
            source_name=source_name,
            col=col,
            prov=prov,
            skipped=skipped,
            unit_family=family,
            unit_assignment=unit_assignment,
            source_mag_ctx=source_mag_ctx,
            idx=idx,
        )
    elif family in ("duration", "dimensional"):
        return _transform_duration(
            cell_str=cell_str,
            target_fmt=target_fmt,
            entity_id=entity_id,
            source_name=source_name,
            col=col,
            prov=prov,
            skipped=skipped,
        )
    else:
        logger.debug(
            "Unsupported family %r for %s.%s — skipping", family, source_name, col
        )
        return None


def _transform_date(
    *,
    cell_str: str,
    target_fmt: str,
    entity_id: str,
    source_name: str,
    col: str,
    prov: ProvenanceLog,
    skipped: SkippedLog,
) -> str | None:
    """Apply date reformatting to a single cell."""
    # Check deny-list.
    if is_denied_date_format(target_fmt):
        skipped.append(
            entity_id=entity_id,
            source=source_name,
            attribute=col,
            original_value=cell_str,
            attempted_format=target_fmt,
            reason=SKIP_DENYLIST,
        )
        return None

    result = reformat_date(cell_str, target_fmt)
    if result is None:
        skipped.append(
            entity_id=entity_id,
            source=source_name,
            attribute=col,
            original_value=cell_str,
            attempted_format=target_fmt,
            reason=SKIP_ROUNDTRIP,
        )
        return None

    new_value, params = result
    if new_value == cell_str:
        # Identity — no provenance needed, no change.
        return None

    prov.append(
        entity_id=entity_id,
        source=source_name,
        attribute=col,
        original_value=cell_str,
        new_value=new_value,
        transform_fn="reformat_date",
        transform_params=params,
    )
    return new_value


def _transform_duration(
    *,
    cell_str: str,
    target_fmt: str,
    entity_id: str,
    source_name: str,
    col: str,
    prov: ProvenanceLog,
    skipped: SkippedLog,
) -> str | None:
    """Apply duration reformatting (seconds_int / mm_ss / hh_mm_ss / human_xm_ys)."""
    result = format_duration(cell_str, target_fmt)
    if result is None:
        skipped.append(
            entity_id=entity_id,
            source=source_name,
            attribute=col,
            original_value=cell_str,
            attempted_format=target_fmt,
            reason=SKIP_ROUNDTRIP,
        )
        return None

    new_value, params = result
    if new_value == cell_str:
        return None

    prov.append(
        entity_id=entity_id,
        source=source_name,
        attribute=col,
        original_value=cell_str,
        new_value=new_value,
        transform_fn="reconvert_unit",
        transform_params=params,
    )
    return new_value


def _resolve_column_context(
    source_mag_ctx: dict[str, Any],
    col: str,
) -> tuple[bool, dict[str, Any]]:
    """Resolve per-column overrides on top of a source's magnitude context.

    Supports two ``columns`` forms in ``source_magnitude_context``:

    * **List form (legacy)**: ``columns: [price]`` — every column in the
      list inherits the source-level ``implicit_currency`` /
      ``implicit_magnitude`` / ``implicit_unit`` keys.
    * **Map form (added 2026-05-22, plan_revision §K5 follow-up)**:
      ``columns: {price: {implicit_currency: GBP, implicit_magnitude: raw},
      vram_gb: {implicit_unit: GB}}`` — per-column overrides take
      precedence over source-level defaults. Mixing classes on the same
      source (money + file_size) requires this form because
      ``implicit_currency`` would otherwise apply to both.

    Returns
    -------
    tuple of (bool, dict)
        ``(is_managed, merged_ctx)`` where ``merged_ctx`` carries
        ``implicit_currency`` / ``implicit_magnitude`` / ``implicit_unit``
        for the column (source defaults overridden by per-column entries
        when present).
    """
    columns = source_mag_ctx.get("columns")
    if columns is None:
        return False, {}
    source_defaults = {
        k: v for k, v in source_mag_ctx.items() if k.startswith("implicit_")
    }
    if isinstance(columns, dict):
        if col not in columns:
            return False, {}
        merged = dict(source_defaults)
        per_col = columns.get(col) or {}
        if isinstance(per_col, dict):
            merged.update(per_col)
        return True, merged
    # Legacy list form.
    if col not in columns:
        return False, {}
    return True, source_defaults


def _transform_number(
    *,
    cell_str: str,
    target_fmt: str,
    entity_id: str,
    source_name: str,
    col: str,
    prov: ProvenanceLog,
    skipped: SkippedLog,
    unit_assignment: list[str] | str | None,
    mag_assignment: list[str] | str | None,
    source_mag_ctx: dict[str, Any],
    idx: int,
) -> str | None:
    """Apply number/money reformatting to a single cell."""
    # Determine target locale (format pool for money = locale IDs).
    locale_id = target_fmt

    # Determine magnitude conversion if applicable.
    is_managed, col_ctx = _resolve_column_context(source_mag_ctx, col)
    from_mag = "raw"
    to_mag = "raw"
    if is_managed:
        from_mag = col_ctx.get("implicit_magnitude", "raw")

    if mag_assignment is not None:
        if isinstance(mag_assignment, list):
            to_mag = mag_assignment[idx]
        else:
            to_mag = mag_assignment

    # Determine currency conversion.
    from_ccy = col_ctx.get("implicit_currency", "USD")
    to_ccy = from_ccy
    if unit_assignment is not None:
        if isinstance(unit_assignment, list):
            to_ccy = unit_assignment[idx]
        else:
            to_ccy = unit_assignment

    # Step 1: magnitude conversion if needed.
    working_value = cell_str
    if from_mag != to_mag and from_mag != "raw" and to_mag != "raw":
        mag_result = reconvert_unit(working_value, "magnitude", from_mag, to_mag)
        if mag_result is not None:
            working_value = mag_result[0]
            prov.append(
                entity_id=entity_id,
                source=source_name,
                attribute=col,
                original_value=cell_str,
                new_value=working_value,
                transform_fn="reconvert_unit",
                transform_params={
                    "from_unit": from_mag,
                    "to_unit": to_mag,
                    "rate": mag_result[1]["rate"],
                    "rate_date": "",
                    "magnitude_scale": to_mag,
                },
            )
    elif from_mag != to_mag and from_mag == "raw" and to_mag != "raw":
        # raw -> scaled magnitude (e.g. raw -> billions)
        mag_result = reconvert_unit(working_value, "magnitude", "raw", to_mag)
        if mag_result is not None:
            working_value = mag_result[0]
            prov.append(
                entity_id=entity_id,
                source=source_name,
                attribute=col,
                original_value=cell_str,
                new_value=working_value,
                transform_fn="reconvert_unit",
                transform_params={
                    "from_unit": "raw",
                    "to_unit": to_mag,
                    "rate": mag_result[1]["rate"],
                    "rate_date": "",
                    "magnitude_scale": to_mag,
                },
            )
    elif from_mag != to_mag and to_mag == "raw":
        # scaled -> raw
        mag_result = reconvert_unit(working_value, "magnitude", from_mag, "raw")
        if mag_result is not None:
            working_value = mag_result[0]
            prov.append(
                entity_id=entity_id,
                source=source_name,
                attribute=col,
                original_value=cell_str,
                new_value=working_value,
                transform_fn="reconvert_unit",
                transform_params={
                    "from_unit": from_mag,
                    "to_unit": "raw",
                    "rate": mag_result[1]["rate"],
                    "rate_date": "",
                    "magnitude_scale": "raw",
                },
            )

    # Step 2: currency conversion if needed.
    pre_ccy_value = working_value
    if to_ccy != from_ccy:
        ccy_result = reconvert_currency(working_value, from_ccy, to_ccy)
        if ccy_result is not None:
            working_value = ccy_result[0]
            prov.append(
                entity_id=entity_id,
                source=source_name,
                attribute=col,
                original_value=pre_ccy_value,
                new_value=working_value,
                transform_fn="reconvert_currency",
                transform_params=ccy_result[1],
            )
        else:
            skipped.append(
                entity_id=entity_id,
                source=source_name,
                attribute=col,
                original_value=pre_ccy_value,
                attempted_format=f"ccy:{from_ccy}->{to_ccy}",
                reason=SKIP_ROUNDTRIP,
            )

    # Step 3: locale reformatting.
    pre_locale_value = working_value
    fmt_result = reformat_number(working_value, locale_id)
    if fmt_result is not None:
        new_value, params = fmt_result
        if new_value != pre_locale_value:
            working_value = new_value
            prov.append(
                entity_id=entity_id,
                source=source_name,
                attribute=col,
                original_value=pre_locale_value,
                new_value=new_value,
                transform_fn="reformat_number",
                transform_params=params,
            )
    else:
        skipped.append(
            entity_id=entity_id,
            source=source_name,
            attribute=col,
            original_value=pre_locale_value,
            attempted_format=f"locale:{locale_id}",
            reason=SKIP_ROUNDTRIP,
        )
        return None

    if working_value == cell_str:
        return None  # Identity — no change.
    return working_value


def _transform_file_size(
    *,
    cell_str: str,
    target_fmt: str,
    entity_id: str,
    source_name: str,
    col: str,
    prov: ProvenanceLog,
    skipped: SkippedLog,
    unit_assignment: list[str] | str | None,
    source_mag_ctx: dict[str, Any],
    idx: int,
    unit_family: str = "file_size",
) -> str | None:
    """Apply file-size or rate unit reformatting to a single cell.

    Mirrors :func:`_transform_number` for the byte-quantity unit family:
    the input cell is interpreted as a bare numeric in the column's
    implicit unit (e.g. ``8`` meaning ``8 GB`` for ``vram_gb``), and
    rewritten to a target unit drawn from the level's pool (``GB`` /
    ``MB`` / ``GiB`` / ``TB`` / ...) with a unit suffix appended
    (``8`` → ``8 GB`` or ``8`` → ``8192 MB``). Locale formatting via
    ``target_fmt`` controls the numeric format (decimal separator,
    digit grouping) — identical to the money path.

    Conversion factors come from the ``file_size`` group (bytes / KB /
    MB / GB / TB + binary KiB / MiB / GiB) or the ``rate`` group
    (bytes_per_second / KB/s / MB/s / GB/s / TB/s) in
    ``unit_factors.yaml`` per the ``unit_family`` argument. The ``rate``
    group was added 2026-05-27 (step 4h cross-knob expansion) for
    products ``read_speed_mb_s`` / ``write_speed_mb_s``. When the
    implicit and target units match, only the suffix attach is applied.
    """
    is_managed, col_ctx = _resolve_column_context(source_mag_ctx, col)
    default_unit = "MB/s" if unit_family == "rate" else "GB"
    from_unit = (
        col_ctx.get("implicit_unit", default_unit) if is_managed else default_unit
    )

    to_unit = from_unit
    if unit_assignment is not None:
        if isinstance(unit_assignment, list):
            to_unit = unit_assignment[idx]
        else:
            to_unit = unit_assignment

    # Step 1: unit conversion if needed.
    working_value = cell_str
    if to_unit != from_unit:
        conv = reconvert_unit(working_value, unit_family, from_unit, to_unit)
        if conv is None:
            skipped.append(
                entity_id=entity_id,
                source=source_name,
                attribute=col,
                original_value=cell_str,
                attempted_format=f"{unit_family}:{from_unit}->{to_unit}",
                reason=SKIP_ROUNDTRIP,
            )
            return None
        working_value = conv[0]
        prov.append(
            entity_id=entity_id,
            source=source_name,
            attribute=col,
            original_value=cell_str,
            new_value=working_value,
            transform_fn="reconvert_unit",
            transform_params={
                "from_unit": from_unit,
                "to_unit": to_unit,
                "rate": conv[1]["rate"],
                "rate_date": "",
                "magnitude_scale": "",
            },
        )

    # Step 2: locale reformatting on the numeric portion. ``bare`` is a
    # sentinel target_fmt that suppresses both locale reformat and the
    # unit suffix — emits the converted numeric as-is so the "unit
    # absent" K5 rung is reachable (e.g. ``8 GB`` → ``8000``, raw MB
    # count without "MB" so downstream must infer the unit).
    if target_fmt != "bare":
        pre_locale_value = working_value
        fmt_result = reformat_number(working_value, target_fmt)
        if fmt_result is not None:
            new_numeric, params = fmt_result
            if new_numeric != pre_locale_value:
                working_value = new_numeric
                prov.append(
                    entity_id=entity_id,
                    source=source_name,
                    attribute=col,
                    original_value=pre_locale_value,
                    new_value=new_numeric,
                    transform_fn="reformat_number",
                    transform_params=params,
                )
        else:
            skipped.append(
                entity_id=entity_id,
                source=source_name,
                attribute=col,
                original_value=pre_locale_value,
                attempted_format=f"locale:{target_fmt}",
                reason=SKIP_ROUNDTRIP,
            )
            return None

    # Step 3: append unit suffix unless target_fmt is ``bare``.
    if target_fmt != "bare":
        numeric_value = working_value
        suffixed = f"{numeric_value} {to_unit}"
        if suffixed != cell_str:
            prov.append(
                entity_id=entity_id,
                source=source_name,
                attribute=col,
                original_value=numeric_value,
                new_value=suffixed,
                transform_fn="append_unit_suffix",
                transform_params={
                    "from_unit": "",
                    "to_unit": to_unit,
                    "rate": 1.0,
                    "rate_date": "",
                    "magnitude_scale": "",
                },
            )
        working_value = suffixed

    if working_value == cell_str:
        return None  # Identity — no change.
    return working_value


# ---- Output writing ---------------------------------------------------------


def write_outputs(
    provenance_df: pd.DataFrame,
    skipped_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Write provenance and skipped-cell artifacts to *output_dir*.

    Parameters
    ----------
    provenance_df : DataFrame
        Provenance log DataFrame.
    skipped_df : DataFrame
        Skipped-cell audit DataFrame.
    output_dir : Path
        Variant directory root.
    """
    prov_dir = output_dir / "output" / "provenance"
    prov_dir.mkdir(parents=True, exist_ok=True)

    provenance_df.to_csv(prov_dir / "knob_05_format_unit.csv", index=False)
    logger.info(
        "Wrote provenance (%d rows) to %s",
        len(provenance_df),
        prov_dir / "knob_05_format_unit.csv",
    )

    skipped_df.to_csv(prov_dir / "knob_05_skipped.csv", index=False)
    logger.info(
        "Wrote skipped audit (%d rows) to %s",
        len(skipped_df),
        prov_dir / "knob_05_skipped.csv",
    )


# ---- CLI --------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Apply Knob 05 — Format/Unit Diversity",
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
        help=(
            "Variant output directory "
            "(default: usecases_synthetic/output/<domain>/<level>)"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master RNG seed (default: 42)",
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

    logger.info("Knob 05: domain=%s level=%s output=%s", domain, level, output_dir)

    config = load_knob_05_config(domain)
    sources = load_domain_sources(domain)

    prov_dir = output_dir / "output" / "provenance"
    collision_index = CollisionIndex(prov_dir)

    reformatted, provenance_df, skipped_df = apply_knob_05(
        domain=domain,
        level=level,
        sources=sources,
        config=config,
        collision_index=collision_index,
        seed=args.seed,
    )

    write_outputs(provenance_df, skipped_df, output_dir)

    # Summary.
    for src_name in sorted(reformatted.keys()):
        logger.info("  %s: %d rows", src_name, len(reformatted[src_name]))
    logger.info("Provenance: %d rows", len(provenance_df))
    logger.info("Skipped: %d rows", len(skipped_df))


if __name__ == "__main__":
    main()
