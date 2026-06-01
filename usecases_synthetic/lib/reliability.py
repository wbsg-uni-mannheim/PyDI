"""Knob 10 — Source Reliability Differentiation.

Core library for measuring per-(source, attribute) gold alignment,
identifying reshufflable cells, and permuting gold-carrier source labels.

See ``knobs/knob_10_source_reliability.md`` for the full specification.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml

from .loaders import read_em_gold_csv

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Canonical-form comparators
# ---------------------------------------------------------------------------


def _parse_date_flexible(value: str) -> date | None:
    """Parse a date string into ``datetime.date``.

    Handles ISO, US, EU, long-form, compact, year-only, and datetime
    strings (e.g. ``2005-01-01T00:00:00.000+01:00``).
    """
    value = value.strip()
    if not value or value.lower() in ("null", "nan", "none", ""):
        return None

    # Year-only (e.g. "1908", "2023")
    if re.fullmatch(r"\d{4}", value):
        return date(int(value), 1, 1)

    # Year-month (e.g. "2024-03")
    if re.fullmatch(r"\d{4}-\d{2}", value):
        parts = value.split("-")
        return date(int(parts[0]), int(parts[1]), 1)

    # Compact (YYYYMMDD)
    if re.fullmatch(r"\d{8}", value):
        return datetime.strptime(value, "%Y%m%d").date()

    try:
        import dateutil.parser as du_parser

        return du_parser.parse(value, dayfirst=False).date()
    except (ValueError, OverflowError, ImportError):
        pass

    try:
        import dateutil.parser as du_parser

        return du_parser.parse(value, dayfirst=True).date()
    except (ValueError, OverflowError, ImportError):
        return None


def _parse_number(value: str) -> Decimal | None:
    """Parse a number string to ``Decimal``.

    Handles plain, en_US (1,234.56), de_DE (1.234,56), scientific
    notation (3.5E9), and K/M/B suffixes.
    """
    if not isinstance(value, str):
        try:
            return Decimal(str(value))
        except (InvalidOperation, ValueError):
            return None

    value = value.strip()
    if not value or value.lower() in ("null", "nan", "none", ""):
        return None

    # Strip currency symbols
    value = re.sub(r"^[£€$¥₹]+\s*", "", value)
    value = re.sub(r"\s*[£€$¥₹]+$", "", value)
    value = re.sub(r"^(USD|EUR|GBP|JPY|CHF|CAD|AUD|CNY)\s*", "", value)

    # Handle K/M/B suffixes
    suffix_scales = {"K": Decimal("1E3"), "M": Decimal("1E6"), "B": Decimal("1E9")}
    scale = Decimal("1")
    upper = value.upper().rstrip()
    for sfx, s in suffix_scales.items():
        if upper.endswith(sfx):
            value = value[: -len(sfx)].strip()
            scale = s
            break

    # Detect de_DE/fr_FR style: comma as decimal separator
    # Pattern: digits, optional dot-separated groups, comma, digits
    if re.match(r"^\d{1,3}(\.\d{3})*,\d+$", value):
        value = value.replace(".", "").replace(",", ".")
    elif re.match(r"^\d{1,3}(\s\d{3})*,\d+$", value):
        # fr_FR style: space as thousands, comma as decimal
        value = value.replace(" ", "").replace(",", ".")
    else:
        # en_US style or plain: remove commas
        value = value.replace(",", "")

    try:
        return Decimal(value) * scale
    except (InvalidOperation, ValueError):
        return None


def _canonicalize_string(value: str) -> str:
    """Casefold + collapse whitespace + strip punctuation."""
    s = value.casefold().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s.strip()


def canonicalize(value: str, comparator_class: str) -> Any:
    """Canonicalize a value according to its comparator class.

    Parameters
    ----------
    value : str
        Raw value string.
    comparator_class : str
        One of ``"date"``, ``"number"``, ``"money"``, ``"duration"``,
        ``"dimensional"``, ``"string"``.

    Returns
    -------
    Any
        Canonical form (``date``, ``Decimal``, ``timedelta``, or ``str``).
        Returns ``None`` if parsing fails.
    """
    if pd.isna(value) or not str(value).strip():
        return None
    value = str(value).strip()

    if comparator_class == "date":
        return _parse_date_flexible(value)
    elif comparator_class in ("number", "money", "dimensional"):
        return _parse_number(value)
    elif comparator_class == "duration":
        # Try parsing as milliseconds (common in music domain)
        num = _parse_number(value)
        if num is not None:
            try:
                return timedelta(milliseconds=float(num))
            except (ValueError, OverflowError):
                return None
        return None
    else:
        return _canonicalize_string(value)


def is_gold_aligned(
    value: str,
    gold_value: str,
    comparator_class: str,
) -> bool:
    """Check if a source value is gold-aligned under canonical-form equality.

    Parameters
    ----------
    value : str
        Source cell value.
    gold_value : str
        Fusion gold value for the same (entity, attribute).
    comparator_class : str
        Comparator class routing key.

    Returns
    -------
    bool
        True if the canonical forms are equal.
    """
    if pd.isna(value) or pd.isna(gold_value):
        return False

    val_str = str(value).strip()
    gold_str = str(gold_value).strip()

    if not val_str or not gold_str:
        return False

    c_val = canonicalize(val_str, comparator_class)
    c_gold = canonicalize(gold_str, comparator_class)

    if c_val is None or c_gold is None:
        return False

    # For numeric types, use tolerance-based comparison
    if comparator_class in ("number", "money", "dimensional"):
        if isinstance(c_val, Decimal) and isinstance(c_gold, Decimal):
            if c_gold == Decimal("0"):
                return c_val == c_gold
            try:
                ratio = abs(c_val - c_gold) / abs(c_gold)
                return ratio < Decimal("1E-4")
            except (InvalidOperation, ZeroDivisionError):
                return c_val == c_gold

    return c_val == c_gold


# ---------------------------------------------------------------------------
# Fusion gold loading
# ---------------------------------------------------------------------------


def load_fusion_gold(fusion_gold_path: Path) -> dict[str, dict[str, str]]:
    """Load fusion gold from XML into ``{entity_id: {attribute: value}}``.

    Parameters
    ----------
    fusion_gold_path : Path
        Path to ``test_set.xml``.

    Returns
    -------
    dict[str, dict[str, str]]
        Gold values keyed by entity ID and attribute name.
    """
    tree = ET.parse(fusion_gold_path)
    root = tree.getroot()

    gold: dict[str, dict[str, str]] = {}
    for entity_elem in root:
        id_elem = entity_elem.find("id")
        if id_elem is None or not id_elem.text:
            continue
        eid = id_elem.text.strip()
        attrs: dict[str, str] = {}
        for child in entity_elem:
            if child.tag == "id":
                continue
            text = child.text
            if text is not None:
                text = text.strip()
                if text:
                    attrs[child.tag] = text
        gold[eid] = attrs
    return gold


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Attribute-class reconciliation (Knob 5 → Knob 10)
# ---------------------------------------------------------------------------


def reconcile_attribute_classes(
    k5_attribute_classes: dict[str, dict[str, str]],
    source_names: list[str],
) -> dict[str, str]:
    """Collapse per-source Knob 5 attribute_classes to a flat map.

    For each attribute, take the majority ``format_family`` across
    sources that declare it. Tiebreak by canonical (sorted) source order.

    Parameters
    ----------
    k5_attribute_classes : dict
        ``{source: {column: format_family}}`` from Knob 5 config.
    source_names : list of str
        Canonical source order (sorted).

    Returns
    -------
    dict[str, str]
        ``{attribute: comparator_class}``.  Attributes not declared by
        Knob 5 are absent (caller defaults to ``"string"``).
    """
    # Gather votes: attribute -> list of (source, family)
    votes: dict[str, list[tuple[str, str]]] = {}
    for source in sorted(source_names):
        source_attrs = k5_attribute_classes.get(source, {})
        for col, family in source_attrs.items():
            votes.setdefault(col, []).append((source, family))

    result: dict[str, str] = {}
    for attr, vote_list in sorted(votes.items()):
        families = [f for _, f in vote_list]
        counter = Counter(families)
        # Majority; tiebreak by first in canonical source order
        max_count = max(counter.values())
        winners = [f for f, c in counter.items() if c == max_count]
        if len(winners) > 1:
            # Tiebreak: pick the family of the first source in canonical order
            for source, family in vote_list:
                if family in winners:
                    result[attr] = family
                    logger.warning(
                        "Attribute %r: sources disagree on format_family %s; "
                        "using %r from source %r (tiebreak)",
                        attr,
                        vote_list,
                        family,
                        source,
                    )
                    break
        else:
            result[attr] = winners[0]

    return result


# Map protection.py kinds → existing K10 comparator classes (the comparator
# name passed to ``canonicalize`` / ``is_gold_aligned``). Strict canonical-form
# equality is preserved per knob_10_source_reliability.md §"Self-contained
# baseline"; only the kind taxonomy source-of-truth moves to protection.py
# (closes the K3 sign-off "kind-consistency" hand-off, 2026-05-07).
_KIND_TO_COMPARATOR_CLASS: dict[str, str] = {
    "continuous": "number",
    "year": "date",
    "date": "date",
    "nominal": "string",
    "long_string": "string",
    "free_text": "string",
    "list": "string",
}


def resolve_attribute_kinds(
    domain: str,
    canonical_attributes: list[str],
) -> dict[str, str]:
    """Resolve per-attribute comparator classes from the protection.py kind map.

    Reads :data:`usecases_synthetic.lib.protection._DEFAULT_KIND_BY_DOMAIN_ATTR`
    (the locked per-domain canonical-attribute → kind map from K1/K5/K6
    sign-offs) and maps each kind to the existing K10 comparator-class name
    via :data:`_KIND_TO_COMPARATOR_CLASS`. This replaces the prior K5-driven
    ``reconcile_attribute_classes`` pathway (Pending #5 strict + infra-aligned
    wire-up, 2026-05-07).

    Parameters
    ----------
    domain : str
        Domain name (``"companies"``, ``"games"``, ``"music"``).
    canonical_attributes : list[str]
        Canonical attribute names that K10 needs comparator classes for
        (the keys of the K10 ``attribute_targets`` block).

    Returns
    -------
    dict[str, str]
        ``{canonical_attribute: comparator_class}`` where comparator_class
        is one of ``{"date", "number", "string"}`` — the existing K10
        canonicalize / is_gold_aligned vocabulary. Attributes absent from
        protection's domain map default to ``"string"``.
    """
    from usecases_synthetic.lib.protection import kind_map_for_domain

    domain_kinds = kind_map_for_domain(domain)
    out: dict[str, str] = {}
    for attr in canonical_attributes:
        kind = domain_kinds.get(attr, "long_string")
        out[attr] = _KIND_TO_COMPARATOR_CLASS.get(kind, "string")
    return out


# ---------------------------------------------------------------------------
# Gold-alignment measurement
# ---------------------------------------------------------------------------


def measure_gold_alignment(
    sources: dict[str, pd.DataFrame],
    fusion_gold: dict[str, dict[str, str]],
    attribute_mapping: dict[str, dict[str, str]],
    id_columns: dict[str, str],
    attribute_classes: dict[str, str],
    entity_linkage: dict[str, dict[str, str]],
) -> pd.DataFrame:
    """Measure per-(source, attribute) gold-alignment rate.

    Parameters
    ----------
    sources : dict[str, DataFrame]
        Source DataFrames keyed by source name.
    fusion_gold : dict[str, dict[str, str]]
        Gold values: ``{gold_entity_id: {gold_attr: value}}``.
    attribute_mapping : dict[str, dict[str, str]]
        ``{source: {source_col: gold_attr}}``.
    id_columns : dict[str, str]
        ``{source: id_column_name}``.
    attribute_classes : dict[str, str]
        ``{gold_attr: comparator_class}`` (from reconciliation).
    entity_linkage : dict[str, dict[str, str]]
        ``{gold_entity_id: {source: record_id}}``.

    Returns
    -------
    DataFrame
        Columns: ``source``, ``attribute``, ``baseline_alignment_rate``,
        ``n_cells``, ``n_aligned``.
    """
    rows: list[dict[str, Any]] = []

    for source_name in sorted(sources.keys()):
        df = sources[source_name]
        id_col = id_columns.get(source_name)
        if id_col is None or id_col not in df.columns:
            continue
        source_attr_map = attribute_mapping.get(source_name, {})

        # Build record_id -> row index
        id_to_idx: dict[str, int] = {}
        for idx in range(len(df)):
            rid = str(df.iloc[idx][id_col])
            id_to_idx[rid] = idx

        # Per gold attribute tracked by this source
        for source_col, gold_attr in sorted(source_attr_map.items()):
            if source_col not in df.columns:
                continue

            comp_class = attribute_classes.get(gold_attr, "string")
            n_cells = 0
            n_aligned = 0

            for gold_eid, gold_attrs in fusion_gold.items():
                gold_val = gold_attrs.get(gold_attr)
                if gold_val is None:
                    continue

                # Find this source's record for this gold entity
                source_links = entity_linkage.get(gold_eid, {})
                record_id = source_links.get(source_name)
                if record_id is None:
                    continue

                row_idx = id_to_idx.get(record_id)
                if row_idx is None:
                    continue

                cell_val = df.iloc[row_idx][source_col]
                if pd.isna(cell_val):
                    continue

                n_cells += 1
                if is_gold_aligned(str(cell_val), gold_val, comp_class):
                    n_aligned += 1

            rate = n_aligned / n_cells if n_cells > 0 else 0.0
            rows.append(
                {
                    "source": source_name,
                    "attribute": gold_attr,
                    "baseline_alignment_rate": round(rate, 6),
                    "n_cells": n_cells,
                    "n_aligned": n_aligned,
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "source",
            "attribute",
            "baseline_alignment_rate",
            "n_cells",
            "n_aligned",
        ],
    )


def identify_per_attribute_winner(
    alignment_df: pd.DataFrame,
) -> dict[str, str]:
    """Identify the per-attribute baseline winner ``W[a]``.

    Parameters
    ----------
    alignment_df : DataFrame
        Output of :func:`measure_gold_alignment`.

    Returns
    -------
    dict[str, str]
        ``{gold_attr: winning_source_name}``.
    """
    winners: dict[str, str] = {}
    for attr in alignment_df["attribute"].unique():
        attr_rows = alignment_df[alignment_df["attribute"] == attr]
        if attr_rows.empty:
            continue
        # Max alignment rate; tiebreak by sorted source name
        best_rate = attr_rows["baseline_alignment_rate"].max()
        best_sources = attr_rows[attr_rows["baseline_alignment_rate"] == best_rate][
            "source"
        ].tolist()
        winners[attr] = sorted(best_sources)[0]
    return winners


# ---------------------------------------------------------------------------
# Entity linkage from EM correspondences
# ---------------------------------------------------------------------------


def build_entity_linkage(
    domain_config: Any,
    id_columns: dict[str, str],
) -> dict[str, dict[str, str]]:
    """Build entity linkage: gold_entity_id -> {source: record_id}.

    Uses EM correspondences to map gold entity IDs (which are Forbes IDs
    in the companies domain) to records in each source.

    Parameters
    ----------
    domain_config : DomainConfig
        Domain configuration (for directory paths and source names).
    id_columns : dict[str, str]
        ``{source: id_column_name}``.

    Returns
    -------
    dict[str, dict[str, str]]
        ``{gold_entity_id: {source_name: record_id}}``.
    """
    em_dir = domain_config.em_dir()
    if not em_dir.exists():
        return {}

    # Collect all positive pairs from EM correspondences
    pairs: list[tuple[str, str]] = []
    for csv_path in sorted(em_dir.glob("*_all.csv")):
        df = read_em_gold_csv(csv_path)
        positives = df[df["label"].astype(str).str.lower() == "true"]
        for _, row in positives.iterrows():
            pairs.append((str(row["id1"]), str(row["id2"])))

    if not pairs:
        for suffix in ("_train.csv", "_val.csv", "_test.csv"):
            for csv_path in sorted(em_dir.glob(f"*{suffix}")):
                df = read_em_gold_csv(csv_path)
                positives = df[df["label"].astype(str).str.lower() == "true"]
                for _, row in positives.iterrows():
                    pairs.append((str(row["id1"]), str(row["id2"])))

    # Determine which ID prefix maps to which source
    prefix_to_source: dict[str, str] = {}
    for spec in domain_config.sources:
        if spec.id_prefix:
            prefix_to_source[spec.id_prefix] = spec.name

    def _resolve_source(record_id: str) -> str | None:
        for prefix, source in prefix_to_source.items():
            if record_id.startswith(prefix):
                return source
        return None

    # Build linkage: group by shared entity
    # Gold entity IDs are the canonical IDs (typically from the first source
    # in each pair, e.g. Forbes)
    linkage: dict[str, dict[str, str]] = {}
    for id1, id2 in pairs:
        src1 = _resolve_source(id1)
        src2 = _resolve_source(id2)

        # Use id1 as the canonical gold entity ID (Forbes convention)
        gold_eid = id1
        if gold_eid not in linkage:
            linkage[gold_eid] = {}
        if src1:
            linkage[gold_eid][src1] = id1
        if src2:
            linkage[gold_eid][src2] = id2

    return linkage


# ---------------------------------------------------------------------------
# Reshufflable-cell identification
# ---------------------------------------------------------------------------


def identify_reshufflable_cells(
    sources: dict[str, pd.DataFrame],
    fusion_gold: dict[str, dict[str, str]],
    attribute_mapping: dict[str, dict[str, str]],
    id_columns: dict[str, str],
    attribute_classes: dict[str, str],
    entity_linkage: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    """Identify cells eligible for reshuffling.

    A cell ``(gold_eid, gold_attr)`` is reshufflable iff:
    - >= 2 sources have a value at this cell
    - >= 1 source is gold-aligned
    - >= 1 source is not gold-aligned

    Parameters
    ----------
    sources, fusion_gold, attribute_mapping, id_columns,
    attribute_classes, entity_linkage
        Same as :func:`measure_gold_alignment`.

    Returns
    -------
    list of dict
        Each dict has keys: ``gold_eid``, ``gold_attr``, ``gold_value``,
        ``comparator_class``, ``source_values`` (list of
        ``{source, record_id, source_col, value, row_idx, aligned}``).
        The ``cell_type`` key is one of ``"reshufflable"``,
        ``"all_aligned"``, ``"no_gold_to_route"``, ``"passthrough"``.
    """
    # Build per-source id->idx lookups
    id_to_idx: dict[str, dict[str, int]] = {}
    for source_name, df in sources.items():
        id_col = id_columns.get(source_name)
        if id_col and id_col in df.columns:
            mapping: dict[str, int] = {}
            for idx in range(len(df)):
                mapping[str(df.iloc[idx][id_col])] = idx
            id_to_idx[source_name] = mapping

    cells: list[dict[str, Any]] = []

    for gold_eid, gold_attrs in sorted(fusion_gold.items()):
        source_links = entity_linkage.get(gold_eid, {})
        if not source_links:
            continue

        for gold_attr, gold_value in sorted(gold_attrs.items()):
            comp_class = attribute_classes.get(gold_attr, "string")
            source_values: list[dict[str, Any]] = []

            for source_name in sorted(sources.keys()):
                record_id = source_links.get(source_name)
                if record_id is None:
                    continue

                df = sources[source_name]
                source_idx_map = id_to_idx.get(source_name, {})
                row_idx = source_idx_map.get(record_id)
                if row_idx is None:
                    continue

                # Find the source column for this gold attribute
                source_attr_map = attribute_mapping.get(source_name, {})
                source_col = None
                for sc, ga in source_attr_map.items():
                    if ga == gold_attr:
                        source_col = sc
                        break
                if source_col is None or source_col not in df.columns:
                    continue

                cell_val = df.iloc[row_idx][source_col]
                if pd.isna(cell_val):
                    continue

                aligned = is_gold_aligned(str(cell_val), gold_value, comp_class)
                source_values.append(
                    {
                        "source": source_name,
                        "record_id": record_id,
                        "source_col": source_col,
                        "value": str(cell_val),
                        "row_idx": row_idx,
                        "aligned": aligned,
                    }
                )

            # Classify cell
            if len(source_values) < 2:
                cell_type = "passthrough"
            else:
                n_aligned = sum(1 for sv in source_values if sv["aligned"])
                n_perturbed = len(source_values) - n_aligned
                if n_aligned == 0:
                    cell_type = "no_gold_to_route"
                elif n_perturbed == 0:
                    cell_type = "all_aligned"
                else:
                    cell_type = "reshufflable"

            cells.append(
                {
                    "gold_eid": gold_eid,
                    "gold_attr": gold_attr,
                    "gold_value": gold_value,
                    "comparator_class": comp_class,
                    "source_values": source_values,
                    "cell_type": cell_type,
                }
            )

    return cells


# ---------------------------------------------------------------------------
# Compromised mask generation
# ---------------------------------------------------------------------------


def generate_compromised_mask(
    source_names: list[str],
    entity_ids: list[str],
    compromise_rate: float,
    compromise_rate_overrides: dict[str, float] | None,
    rng: np.random.Generator,
) -> dict[str, set[str]]:
    """Generate per-(source, entity) compromised mask.

    Parameters
    ----------
    source_names : list of str
        Source names (sorted canonical order).
    entity_ids : list of str
        Gold entity IDs (sorted).
    compromise_rate : float
        Default fraction of entities flagged per source.
    compromise_rate_overrides : dict or None
        Per-source rate overrides.
    rng : numpy.random.Generator
        Seeded RNG for the mask stage.

    Returns
    -------
    dict[str, set[str]]
        ``{source: set(entity_ids_compromised)}``.
    """
    mask: dict[str, set[str]] = {}
    n_entities = len(entity_ids)
    entity_arr = np.array(entity_ids)

    for source in sorted(source_names):
        rate = compromise_rate
        if compromise_rate_overrides and source in compromise_rate_overrides:
            rate = compromise_rate_overrides[source]

        n_compromised = int(np.floor(rate * n_entities))
        if n_compromised <= 0:
            mask[source] = set()
            continue

        chosen_indices = rng.choice(n_entities, size=n_compromised, replace=False)
        mask[source] = set(entity_arr[chosen_indices].tolist())

    return mask


# ---------------------------------------------------------------------------
# Core reshuffle
# ---------------------------------------------------------------------------


def reshuffle_cells(
    cells: list[dict[str, Any]],
    sources: dict[str, pd.DataFrame],
    attribute_targets: dict[str, dict[str, float]],
    compromised_mask: dict[str, set[str]],
    corr_strength: float,
    concentration_cap: float,
    rng: np.random.Generator,
    level: str,
) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    """Reshuffle gold-carrier source labels on reshufflable cells.

    Parameters
    ----------
    cells : list of dict
        Output of :func:`identify_reshufflable_cells`.
    sources : dict[str, DataFrame]
        Source DataFrames (will be copied and mutated).
    attribute_targets : dict
        ``{gold_attr: {source_name: target_probability}}`` for current level.
    compromised_mask : dict
        ``{source: set(entity_ids)}``.
    corr_strength : float
        Multiplicative down-weight for compromised sources.
    concentration_cap : float
        Max allowed ``T[a, W[a]]`` (defensive cap).
    rng : numpy.random.Generator
        Seeded RNG for the per-cell sampling stage.
    level : str
        Difficulty level.

    Returns
    -------
    mutated_sources : dict[str, DataFrame]
        Sources with reshuffled values.
    provenance_rows : list of dict
        Provenance records.
    """
    # Deep copy sources
    mutated: dict[str, pd.DataFrame] = {}
    for name, df in sources.items():
        new_df = df.copy()
        new_df.attrs = df.attrs.copy()
        mutated[name] = new_df

    provenance_rows: list[dict[str, Any]] = []
    canonical_source_order = sorted(sources.keys())

    for cell in cells:
        gold_eid = cell["gold_eid"]
        gold_attr = cell["gold_attr"]
        cell_type = cell["cell_type"]
        source_values = cell["source_values"]

        if cell_type == "passthrough":
            continue

        if cell_type == "all_aligned":
            continue

        if cell_type == "no_gold_to_route":
            perturbed_sources = [sv["source"] for sv in source_values]
            provenance_rows.append(
                {
                    "entity_id": gold_eid,
                    "source": "",
                    "attribute": gold_attr,
                    "original_value": "",
                    "new_value": "",
                    "transform_fn": "no_gold_to_route",
                    "transform_params": json.dumps(
                        {
                            "perturbed_sources": perturbed_sources,
                            "reason": "no_aligned_value",
                        }
                    ),
                    "knob": 10,
                    "level": level,
                }
            )
            continue

        # reshufflable: sample gold-carrier
        present_sources = [sv["source"] for sv in source_values]
        aligned_sources = [sv["source"] for sv in source_values if sv["aligned"]]

        # Build per-cell weight vector
        attr_targets = attribute_targets.get(gold_attr, {})
        weights: dict[str, float] = {}
        for sv in source_values:
            src = sv["source"]
            base_w = attr_targets.get(src, 0.0)
            # Cap concentration
            if base_w > concentration_cap:
                base_w = concentration_cap
            # Apply compromised-mask down-weight
            src_mask = compromised_mask.get(src, set())
            if gold_eid in src_mask:
                base_w *= 1.0 - corr_strength
            weights[src] = max(base_w, 1e-10)  # prevent zero

        # Normalize over present sources
        total_w = sum(weights.values())
        probs = {s: w / total_w for s, w in weights.items()}

        # Sample gold carrier
        src_list = sorted(probs.keys())
        prob_arr = np.array([probs[s] for s in src_list])
        # Re-normalize to handle float precision
        prob_arr = prob_arr / prob_arr.sum()
        chosen_idx = rng.choice(len(src_list), p=prob_arr)
        s_gold = src_list[chosen_idx]

        # Determine if we need to swap
        s_gold_sv = next(sv for sv in source_values if sv["source"] == s_gold)

        if s_gold_sv["aligned"]:
            # Identity — sampler chose a source that already has the gold
            provenance_rows.append(
                {
                    "entity_id": gold_eid,
                    "source": s_gold,
                    "attribute": gold_attr,
                    "original_value": s_gold_sv["value"],
                    "new_value": s_gold_sv["value"],
                    "transform_fn": "identity",
                    "transform_params": json.dumps(
                        {"gold_source": s_gold, "sampled": True}
                    ),
                    "knob": 10,
                    "level": level,
                }
            )
            continue

        # Need a swap: s_gold is not aligned, pick s_swap from aligned
        # Lowest-indexed source in S_aligned under canonical source order
        aligned_sorted = sorted(
            aligned_sources, key=lambda s: canonical_source_order.index(s)
        )
        s_swap = aligned_sorted[0]

        # Find the source_value entries
        sv_gold = next(sv for sv in source_values if sv["source"] == s_gold)
        sv_swap = next(sv for sv in source_values if sv["source"] == s_swap)

        # Perform the 2-cycle swap in mutated DataFrames
        df_gold = mutated[s_gold]
        df_swap = mutated[s_swap]

        # Ensure object dtype for string assignment
        gold_col = sv_gold["source_col"]
        swap_col = sv_swap["source_col"]

        if df_gold[gold_col].dtype != object:
            df_gold[gold_col] = df_gold[gold_col].astype(object)
        if df_swap[swap_col].dtype != object:
            df_swap[swap_col] = df_swap[swap_col].astype(object)

        val_before_gold = df_gold.iat[
            sv_gold["row_idx"], df_gold.columns.get_loc(gold_col)
        ]
        val_before_swap = df_swap.iat[
            sv_swap["row_idx"], df_swap.columns.get_loc(swap_col)
        ]

        df_gold.iat[sv_gold["row_idx"], df_gold.columns.get_loc(gold_col)] = (
            val_before_swap
        )
        df_swap.iat[sv_swap["row_idx"], df_swap.columns.get_loc(swap_col)] = (
            val_before_gold
        )

        # Build weight vector for provenance
        weight_info = {s: round(probs[s], 6) for s in src_list}

        # Check if s_gold was in compromised mask
        from_mask = gold_eid in compromised_mask.get(s_gold, set())

        params = {
            "gold_source_before": s_swap,
            "gold_source_after": s_gold,
            "perturbed_sources": [
                sv["source"] for sv in source_values if not sv["aligned"]
            ],
            "weight_vector": weight_info,
            "sampled_from_compromised_mask": from_mask,
        }
        params_json = json.dumps(params)

        # Two provenance rows: old gold-carrier loses, new gold-carrier gains
        provenance_rows.append(
            {
                "entity_id": gold_eid,
                "source": s_swap,
                "attribute": gold_attr,
                "original_value": str(val_before_swap),
                "new_value": str(val_before_gold),
                "transform_fn": "reassign_gold_carrier",
                "transform_params": params_json,
                "knob": 10,
                "level": level,
            }
        )
        provenance_rows.append(
            {
                "entity_id": gold_eid,
                "source": s_gold,
                "attribute": gold_attr,
                "original_value": str(val_before_gold),
                "new_value": str(val_before_swap),
                "transform_fn": "reassign_gold_carrier",
                "transform_params": params_json,
                "knob": 10,
                "level": level,
            }
        )

    return mutated, provenance_rows


# ---------------------------------------------------------------------------
# Multiset-invariant assertion
# ---------------------------------------------------------------------------


def assert_multiset_invariant(
    original_sources: dict[str, pd.DataFrame],
    mutated_sources: dict[str, pd.DataFrame],
    cells: list[dict[str, Any]],
) -> None:
    """Assert the pure-permutation invariant on every reshuffled cell.

    For each cell, ``Counter(values_before) == Counter(values_after)``.

    Raises
    ------
    AssertionError
        If the invariant is violated on any cell.
    """
    for cell in cells:
        if cell["cell_type"] not in ("reshufflable", "no_gold_to_route"):
            continue

        values_before: list[str] = []
        values_after: list[str] = []

        for sv in cell["source_values"]:
            source = sv["source"]
            row_idx = sv["row_idx"]
            source_col = sv["source_col"]

            orig_val = original_sources[source].iloc[row_idx][source_col]
            mut_val = mutated_sources[source].iloc[row_idx][source_col]

            values_before.append(str(orig_val))
            values_after.append(str(mut_val))

        assert Counter(values_before) == Counter(values_after), (
            f"Multiset invariant violated at ({cell['gold_eid']}, "
            f"{cell['gold_attr']}): before={Counter(values_before)}, "
            f"after={Counter(values_after)}"
        )
