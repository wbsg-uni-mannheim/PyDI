#!/usr/bin/env python3
"""Apply Knob 03 — Per-source Attribute Drop Rate.

Parametric cell masking (NaN injection) with monotone nesting via shared
per-cell uniforms: ``D_easy ⊆ D_medium ⊆ D_hard``.

See ``knobs/knob_03_attribute_drop.md`` for the full specification.

Usage
-----
::

    python usecases_synthetic/scripts/apply_knob_03_drop.py \\
        --domain companies --level easy --seed 42

Inputs
------
- Source DataFrames from ``usecases/<domain>/input/data/``
- EM correspondences from ``usecases/<domain>/input/entitymatching/``
- Fusion gold from ``usecases/<domain>/input/fusion/``
- Per-domain config at ``usecases_synthetic/config/knob_03_drop/<domain>.yaml``

Outputs (under *output_dir*)
------
- Mutated source DataFrames (returned in-memory)
- Baseline CSV at ``<output_dir>/output/baselines/knob_03_baseline_missingness.csv``
- Provenance CSV at ``<output_dir>/output/provenance/knob_03_attribute_drop.csv``
- Skipped-cell audit at ``<output_dir>/output/provenance/knob_03_skipped.csv``
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure repo root is on sys.path.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _is_missing(val: Any) -> bool:
    """Return True if a cell value should be treated as missing.

    Wraps :func:`pandas.isna` with defensive handling of list-like
    cell values (e.g. ``dbpedia.founders`` stores a list of founder
    names per row). ``pd.isna`` on an array returns an elementwise
    mask and raises ``ValueError`` when the caller tries to use the
    result as a boolean. List-like cells always count as *present*
    for K3's purposes: a non-empty list is data, an empty list is
    absence of data.
    """
    if isinstance(val, (list, tuple, set, dict)):
        return len(val) == 0
    result = pd.isna(val)
    if isinstance(result, bool):
        return result
    # Pandas sometimes returns np.bool_ etc; coerce.
    return bool(result)


from usecases_synthetic.lib.baseline_measure import (
    baseline_to_dataframe,
    measure_missingness,
)
from usecases_synthetic.lib.domain_config import (
    CONFIG_DIR,
    USECASES_DIR,
    VALID_LEVELS,
    DomainConfig,
    load_domain_config,
    load_knob_config,
)
from usecases_synthetic.lib.loaders import load_domain_sources, read_em_gold_csv
from usecases_synthetic.lib.protection import (
    fusion_cell_tolerance,
    is_close_enough,
    load_fusion_target_values,
)
from usecases_synthetic.lib.provenance import ProvenanceLog
from usecases_synthetic.lib.rng import make_rng

logger = logging.getLogger(__name__)


# ---- Entity linkage -------------------------------------------------------


@dataclass
class EntityLinkage:
    """Cross-source entity linkage built from EM correspondences.

    Parameters
    ----------
    groups : dict[str, list[tuple[str, str]]]
        Mapping from canonical group ID to list of ``(source, record_id)``.
    index : dict[str, str]
        Mapping from any record ID to its canonical group ID.
    """

    groups: dict[str, list[tuple[str, str]]] = field(default_factory=dict)
    index: dict[str, str] = field(default_factory=dict)


def _find(parent: dict[str, str], x: str) -> str:
    """Union-find: find with path compression."""
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _union(parent: dict[str, str], rank: dict[str, int], a: str, b: str) -> None:
    """Union-find: union by rank."""
    ra, rb = _find(parent, a), _find(parent, b)
    if ra == rb:
        return
    if rank[ra] < rank[rb]:
        ra, rb = rb, ra
    parent[rb] = ra
    if rank[ra] == rank[rb]:
        rank[ra] += 1


def build_entity_linkage(
    domain_config: DomainConfig,
    id_columns: dict[str, str],
    sources: dict[str, pd.DataFrame],
) -> EntityLinkage:
    """Build entity linkage from EM gold correspondences.

    Loads all EM ``*_all.csv`` files (falling back to train+val+test union),
    filters to ``label=true``, and builds connected components via
    union-find.

    Parameters
    ----------
    domain_config : DomainConfig
        Domain configuration with path helpers.
    id_columns : dict[str, str]
        Per-source ID column name.
    sources : dict[str, DataFrame]
        Source DataFrames (used to identify source membership of each ID).

    Returns
    -------
    EntityLinkage
        Linkage structure with groups and reverse index.
    """
    em_dir = domain_config.em_dir()
    if not em_dir.exists():
        return EntityLinkage()

    # Collect all positive pairs from EM gold files.
    pairs: list[tuple[str, str]] = []
    for csv_path in sorted(em_dir.glob("*_all.csv")):
        df = read_em_gold_csv(csv_path)
        positives = df[df["label"].astype(str).str.lower() == "true"]
        for _, row in positives.iterrows():
            pairs.append((str(row["id1"]), str(row["id2"])))

    if not pairs:
        # Fall back to train+val+test
        for suffix in ("_train.csv", "_val.csv", "_test.csv"):
            for csv_path in sorted(em_dir.glob(f"*{suffix}")):
                df = read_em_gold_csv(csv_path)
                positives = df[df["label"].astype(str).str.lower() == "true"]
                for _, row in positives.iterrows():
                    pairs.append((str(row["id1"]), str(row["id2"])))

    # Build union-find over all IDs.
    all_ids: set[str] = set()
    for id1, id2 in pairs:
        all_ids.add(id1)
        all_ids.add(id2)

    # Also add all source record IDs (even if they have no match).
    id_to_source: dict[str, str] = {}
    for source_name, df in sources.items():
        id_col = id_columns.get(source_name)
        if id_col and id_col in df.columns:
            for rid in df[id_col].astype(str):
                all_ids.add(rid)
                id_to_source[rid] = source_name

    parent: dict[str, str] = {x: x for x in all_ids}
    rank: dict[str, int] = {x: 0 for x in all_ids}

    for id1, id2 in pairs:
        _union(parent, rank, id1, id2)

    # Build groups (only multi-member groups are interesting for K3).
    from collections import defaultdict

    groups_raw: dict[str, list[str]] = defaultdict(list)
    for rid in all_ids:
        root = _find(parent, rid)
        groups_raw[root].append(rid)

    groups: dict[str, list[tuple[str, str]]] = {}
    index: dict[str, str] = {}
    for root, members in groups_raw.items():
        if len(members) < 2:
            # Single-member group — still index it but don't store the group.
            for m in members:
                index[m] = root
            continue
        group_members: list[tuple[str, str]] = []
        for m in members:
            src = id_to_source.get(m, "unknown")
            group_members.append((src, m))
        groups[root] = group_members
        for m in members:
            index[m] = root

    return EntityLinkage(groups=groups, index=index)


# ---- Target rate computation ----------------------------------------------


def _get_managed_columns(config: dict[str, Any]) -> dict[str, list[str]]:
    """Extract the list of managed columns per source from config."""
    attr_classes: dict[str, dict[str, str]] = config["attribute_classes"]
    return {src: list(cols.keys()) for src, cols in attr_classes.items()}


def _get_attr_class(config: dict[str, Any], source: str, col: str) -> str:
    """Look up the attribute class for a source column."""
    return config["attribute_classes"].get(source, {}).get(col, "secondary")


def _compute_b_min(
    baseline: dict[str, dict[str, float]],
    config: dict[str, Any],
    source: str,
    col: str,
) -> float:
    """Compute B_min[a] — minimum baseline across sources for the same target attribute.

    Uses ``attribute_mapping`` to find cross-source correspondents.
    For columns without mapping, B_min equals their own baseline.
    """
    attr_mapping: dict[str, dict[str, str]] = config.get("attribute_mapping", {})
    source_mapping = attr_mapping.get(source, {})
    target_attr = source_mapping.get(col)

    if target_attr is None:
        # No cross-source correspondent; B_min = own baseline.
        return baseline.get(source, {}).get(col, 0.0)

    # Find all sources that map a column to the same target attribute.
    b_min = baseline.get(source, {}).get(col, 0.0)
    for other_source, other_mapping in attr_mapping.items():
        for other_col, other_target in other_mapping.items():
            if other_target == target_attr:
                other_rate = baseline.get(other_source, {}).get(other_col, 0.0)
                b_min = min(b_min, other_rate)

    return b_min


def compute_target_rates(
    baseline: dict[str, dict[str, float]],
    level: str,
    config: dict[str, Any],
) -> dict[str, dict[str, float]]:
    """Compute target missingness rates per (source, attribute).

    Applies the level-specific transform (compress/identity/stretch)
    to the baseline, then clips to per-class floor rates and
    per-(source, attribute) ceilings.

    Parameters
    ----------
    baseline : dict[str, dict[str, float]]
        Measured baseline from :func:`measure_missingness`.
    level : str
        ``"easy"``, ``"medium"``, or ``"hard"``.
    config : dict
        Parsed K3 config YAML.

    Returns
    -------
    dict[str, dict[str, float]]
        Target rates ``T[source][column]``.
    """
    rates_per_level: dict[str, dict[str, float]] = config["rates_per_level"]
    floors = rates_per_level[level]
    transform = config["transform_per_level"][level]
    compression_factor: float = config.get("compression_factor", 0.7)
    stretch_factor: float = config.get("stretch_factor", 1.5)
    ceiling_delta: float = config.get("per_cell_ceiling_delta", 0.10)
    overrides: dict[str, dict[str, float]] = config.get(
        "per_source_attribute_overrides", {}
    )

    targets: dict[str, dict[str, float]] = {}

    for source, cols in baseline.items():
        targets[source] = {}
        for col, b_rate in cols.items():
            cls = _get_attr_class(config, source, col)
            floor = floors.get(cls, 0.0)
            b_min = _compute_b_min(baseline, config, source, col)

            if transform == "compress":
                t = b_min + (1.0 - compression_factor) * (b_rate - b_min)
            elif transform == "identity":
                t = b_rate
            elif transform == "stretch":
                t = b_min + stretch_factor * (b_rate - b_min)
            else:
                raise ValueError(f"Unknown transform: {transform!r}")

            # Apply floor.
            t = max(t, floor)

            # Apply per-(source, attribute) ceiling.
            source_delta = overrides.get(source, {}).get(col, ceiling_delta)
            ceiling = b_rate + source_delta
            t = min(t, ceiling)

            # Absolute cap.
            t = min(t, 1.0)

            targets[source][col] = t

    return targets


# ---- Shared uniform draws -------------------------------------------------


def draw_shared_uniforms(
    sources: dict[str, pd.DataFrame],
    managed_columns: dict[str, list[str]],
    rng: np.random.Generator,
) -> dict[str, pd.DataFrame]:
    """Draw per-cell uniform ``u[s, a, e] ~ U(0,1)`` for all managed cells.

    The draws are shared across all levels so that ``D_easy ⊆ D_medium
    ⊆ D_hard`` (drop if ``u < T[s, a]``).

    Parameters
    ----------
    sources : dict[str, DataFrame]
        Source DataFrames.
    managed_columns : dict[str, list[str]]
        Columns to manage per source.
    rng : numpy.random.Generator
        Seeded RNG.

    Returns
    -------
    dict[str, DataFrame]
        DataFrames with the same index as *sources*, columns = managed
        columns, values = uniform draws.
    """
    uniforms: dict[str, pd.DataFrame] = {}

    # Deterministic iteration order: sorted sources, then sorted columns.
    for source_name in sorted(sources.keys()):
        df = sources[source_name]
        cols = sorted(managed_columns.get(source_name, []))
        n = len(df)
        if not cols:
            uniforms[source_name] = pd.DataFrame(index=df.index)
            continue

        data = {}
        for col in cols:
            if col in df.columns:
                data[col] = rng.random(n)
        uniforms[source_name] = pd.DataFrame(data, index=df.index)

    return uniforms


# ---- Propagation fill (easy only) -----------------------------------------


def propagate_fill(
    sources: dict[str, pd.DataFrame],
    linkage: EntityLinkage,
    config: dict[str, Any],
    baseline: dict[str, dict[str, float]],
    prov_log: ProvenanceLog,
) -> dict[str, pd.DataFrame]:
    """Fill missing values from the lowest-missingness source (easy only).

    For each multi-source entity group, for each target attribute, if a
    source record has a null value but another source in the group has a
    non-null value, copy the value from the source with the lowest
    baseline missingness for that attribute.

    Parameters
    ----------
    sources : dict[str, DataFrame]
        Source DataFrames (modified in place and returned).
    linkage : EntityLinkage
        Cross-source entity linkage.
    config : dict
        K3 config.
    baseline : dict[str, dict[str, float]]
        Measured baseline missingness.
    prov_log : ProvenanceLog
        Provenance log to append fill records to.

    Returns
    -------
    dict[str, DataFrame]
        The same *sources* dict with filled values.
    """
    attr_mapping: dict[str, dict[str, str]] = config.get("attribute_mapping", {})
    id_columns: dict[str, str] = config.get("id_columns", {})

    if not linkage.groups or not attr_mapping:
        return sources

    # Build reverse lookup: (source, record_id) -> row index.
    source_id_to_idx: dict[str, dict[str, int]] = {}
    for source_name, df in sources.items():
        id_col = id_columns.get(source_name)
        if id_col and id_col in df.columns:
            lookup: dict[str, int] = {}
            for idx, rid in zip(df.index, df[id_col].astype(str)):
                lookup[rid] = idx
            source_id_to_idx[source_name] = lookup

    # Build target_attr -> [(source, column)] reverse mapping.
    target_to_source_cols: dict[str, list[tuple[str, str]]] = {}
    for src, mapping in attr_mapping.items():
        for col, target in mapping.items():
            target_to_source_cols.setdefault(target, []).append((src, col))

    fill_count = 0

    for group_id, members in linkage.groups.items():
        # For each target attribute with cross-source mapping.
        for target_attr, source_cols in target_to_source_cols.items():
            # Collect available values across sources for this entity group.
            available: list[tuple[str, str, int, Any, float]] = []
            missing: list[tuple[str, str, int]] = []

            for member_source, member_id in members:
                if member_source not in source_id_to_idx:
                    continue
                idx_lookup = source_id_to_idx[member_source]
                if member_id not in idx_lookup:
                    continue
                row_idx = idx_lookup[member_id]

                # Find the column in this source for this target attribute.
                src_col = None
                for sc_src, sc_col in source_cols:
                    if sc_src == member_source:
                        src_col = sc_col
                        break
                if src_col is None or src_col not in sources[member_source].columns:
                    continue

                val = sources[member_source].at[row_idx, src_col]
                b_rate = baseline.get(member_source, {}).get(src_col, 0.0)

                if _is_missing(val):
                    missing.append((member_source, member_id, row_idx))
                else:
                    available.append((member_source, member_id, row_idx, val, b_rate))

            if not available or not missing:
                continue

            # Pick the donor: source with lowest baseline missingness.
            donor = min(available, key=lambda x: x[4])
            donor_source, donor_id, _, donor_val, _ = donor

            # Find the donor's column name for this target attribute.
            donor_col = None
            for sc_src, sc_col in source_cols:
                if sc_src == donor_source:
                    donor_col = sc_col
                    break

            # Fill each missing cell.
            for miss_source, miss_id, miss_idx in missing:
                fill_col = None
                for sc_src, sc_col in source_cols:
                    if sc_src == miss_source:
                        fill_col = sc_col
                        break
                if fill_col is None:
                    continue

                sources[miss_source].at[miss_idx, fill_col] = donor_val
                fill_count += 1

                prov_log.append(
                    entity_id=miss_id,
                    source=miss_source,
                    attribute=fill_col,
                    original_value="",
                    new_value=str(donor_val),
                    transform_fn="propagate_fill",
                    transform_params={
                        "source_from": donor_source,
                        "source_to": miss_source,
                        "entity_id": miss_id,
                        "donor_id": donor_id,
                        "donor_column": donor_col or "",
                        "value_copied": str(donor_val),
                    },
                )

    if fill_count > 0:
        logger.info("Propagation fill: %d cells filled", fill_count)

    return sources


# ---- Constraint enforcement -----------------------------------------------


def _build_fusion_gold_ids(domain_config: DomainConfig) -> set[str]:
    """Load fusion-protected entity IDs from validation + test XMLs.

    Reads both fusion files declared by the domain config's
    ``fusion_files`` block (defaults: ``validation_set.xml`` and
    ``test_set.xml``). Per §"Terminology convention" in
    plan_s1_scale.md, both fusion validation and test entities are
    protected at every value- and entity-mutating knob, K3 included.
    """
    import xml.etree.ElementTree as ET

    ids: set[str] = set()
    for fusion_path in domain_config.fusion_paths():
        if not fusion_path.exists():
            continue
        tree = ET.parse(fusion_path)
        for id_elem in tree.getroot().iter("id"):
            if id_elem.text:
                ids.add(id_elem.text.strip())
    return ids


def _compute_protected_cells(
    sources: dict[str, pd.DataFrame],
    linkage: EntityLinkage,
    fusion_gold_ids: set[str],
    config: dict[str, Any],
    domain: str | None = None,
) -> set[tuple[str, int, str]]:
    """Pre-compute level-independent cell protection.

    Computes a set of ``(source, row_idx, column)`` triples that must
    never be dropped at any level, ensuring monotone nested drop sets
    ``D_easy ⊆ D_medium ⊆ D_hard``.

    Two protections are merged:

    1. **Fusion survivor floor**: for each entity group in the fusion
       gold, for each target attribute, protect the non-null cell whose
       value is closest to the fusion target value under
       :func:`protection.is_close_enough` (Pending #5 contract). When
       *domain* is ``None`` or no fusion target is available, fall back
       to the first cell in sorted source order.
    2. **Conflict preservation**: for each entity group with ≥2 distinct
       values on a target attribute, protect one cell per distinct value
       so the disagreement survives at all levels.

    Parameters
    ----------
    sources : dict[str, DataFrame]
        Source DataFrames (original, pre-fill).
    linkage : EntityLinkage
        Cross-source entity linkage.
    fusion_gold_ids : set[str]
        Fusion gold entity IDs.
    config : dict
        K3 config.

    Returns
    -------
    set[tuple[str, int, str]]
        Protected ``(source_name, row_index, column_name)`` triples.
    """
    protected: set[tuple[str, int, str]] = set()

    if not linkage.groups:
        return protected

    id_columns: dict[str, str] = config.get("id_columns", {})
    attr_mapping: dict[str, dict[str, str]] = config.get("attribute_mapping", {})

    # Pending #5 hand-off: load fusion target values + per-attribute
    # tolerance so the survivor floor protects the carrier closest to
    # the fusion target rather than the first source by sorted order.
    fusion_targets: dict[str, dict[str, list[str]]] = {}
    tol_overrides: dict[str, dict[str, float | str]] = {}
    if domain is not None:
        try:
            fusion_targets = load_fusion_target_values(domain)
        except Exception:  # noqa: BLE001 — fall back to first-sorted on any failure
            fusion_targets = {}
        tol_overrides = config.get("fusion_protection_tolerance", {}) or {}

    def _pick_closest_carrier(
        cells: list[tuple[str, int, str, str]],
        entity_id: str,
        target_attr: str,
    ) -> tuple[str, int, str]:
        """Return the (source, idx, col) of the cell closest to the fusion
        target for this (entity, attribute), or the first-sorted cell when
        no target / no closeness can be resolved."""
        target_vals = fusion_targets.get(entity_id, {}).get(target_attr, [])
        if not target_vals:
            return cells[0][0], cells[0][1], cells[0][2]
        tol = fusion_cell_tolerance(domain or "", target_attr, tol_overrides)
        for src, idx, col, val_key in cells:
            for tv in target_vals:
                if is_close_enough(val_key, tv, tol):
                    return src, idx, col
        return cells[0][0], cells[0][1], cells[0][2]

    # Build target_attr -> [(source, column)] mapping.
    target_to_source_cols: dict[str, list[tuple[str, str]]] = {}
    for src, mapping in attr_mapping.items():
        for col, target in mapping.items():
            target_to_source_cols.setdefault(target, []).append((src, col))

    # Build reverse lookup: (source, record_id) -> row index.
    source_id_to_idx: dict[str, dict[str, int]] = {}
    for source_name, df in sources.items():
        id_col = id_columns.get(source_name)
        if id_col and id_col in df.columns:
            lookup: dict[str, int] = {}
            for idx, rid in zip(df.index, df[id_col].astype(str)):
                lookup[rid] = idx
            source_id_to_idx[source_name] = lookup

    for group_id, members in linkage.groups.items():
        # Identify the fusion-gold entity id within this group (if any) so
        # we can look up the per-(entity, attribute) target value below.
        fusion_entity_id: str | None = next(
            (mid for _, mid in members if mid in fusion_gold_ids), None
        )
        group_in_gold = fusion_entity_id is not None

        for target_attr, source_cols in target_to_source_cols.items():
            # Collect non-null cells with their values.
            cells: list[tuple[str, int, str, str]] = []  # (src, idx, col, val_key)
            for member_source, member_id in sorted(members):
                if member_source not in source_id_to_idx:
                    continue
                idx_lookup = source_id_to_idx[member_source]
                if member_id not in idx_lookup:
                    continue
                row_idx = idx_lookup[member_id]

                src_col = None
                for sc_src, sc_col in source_cols:
                    if sc_src == member_source:
                        src_col = sc_col
                        break
                if src_col is None or src_col not in sources[member_source].columns:
                    continue

                val = sources[member_source].at[row_idx, src_col]
                if not _is_missing(val):
                    val_key = str(val).strip().lower()
                    cells.append((member_source, row_idx, src_col, val_key))

            if not cells:
                continue

            # Fusion survivor floor: protect one cell.
            if group_in_gold:
                assert fusion_entity_id is not None  # mypy
                protected.add(
                    _pick_closest_carrier(cells, fusion_entity_id, target_attr)
                )

            # Conflict preservation: if ≥2 distinct values, protect one
            # representative per distinct value.
            distinct_values: dict[str, tuple[str, int, str]] = {}
            for src, ridx, col, val_key in cells:
                if val_key not in distinct_values:
                    distinct_values[val_key] = (src, ridx, col)

            if len(distinct_values) >= 2:
                for cell_triple in distinct_values.values():
                    protected.add(cell_triple)

    return protected


def apply_constraints(
    drop_mask: dict[str, pd.DataFrame],
    sources: dict[str, pd.DataFrame],
    linkage: EntityLinkage,
    fusion_gold_ids: set[str],
    config: dict[str, Any],
    level: str,
    skipped_log: ProvenanceLog,
    live_sources: dict[str, pd.DataFrame] | None = None,
    uniforms: dict[str, pd.DataFrame] | None = None,
    target_rates: dict[str, dict[str, float]] | None = None,
    fusion_protected_cells: set[tuple[str, int, str]] | None = None,
) -> None:
    """Apply hard constraints to the drop mask (in place).

    Constraints (in order):
    1. Fusion survivor floor — at least one source retains each attribute
       for fusion gold entities.
    2. Conflict-preserving drop — preserve ≥2 disagreeing values.
    3. Single-source-survivor cap at hard.
    4. Per-(source, attribute) ceiling (already applied in target computation).

    Parameters
    ----------
    drop_mask : dict[str, DataFrame]
        Boolean DataFrames (True = will be dropped). Modified in place.
    sources : dict[str, DataFrame]
        Source frames used for **value reads** (conflict-preserve checks,
        single-source-survivor cap value lookups). In canonical S1 use
        this is the pre-K2 reference frame so constraint decisions stay
        level-invariant.
    linkage : EntityLinkage
        Entity linkage for cross-source constraint checks.
    fusion_gold_ids : set[str]
        Entity IDs from the fusion gold test set.
    config : dict
        K3 config.
    level : str
        Current difficulty level.
    skipped_log : ProvenanceLog
        Log for recording skipped (un-dropped) cells.
    live_sources : dict[str, DataFrame] or None, default None
        Source frames used for **drop_mask idx lookups**. Drop masks are
        indexed by the live (post-K2, post-K4) frame's index; constraint
        decisions still come from ``sources`` (typically the pre-K2
        reference). When ``live_sources is None`` we fall back to
        ``sources`` for idx lookups too — the legacy behaviour, only
        correct when ``sources`` and the drop_mask share the same index
        space (e.g. standalone K3 invocation without K2 + K4 upstream).
    """
    id_columns: dict[str, str] = config.get("id_columns", {})
    attr_mapping: dict[str, dict[str, str]] = config.get("attribute_mapping", {})
    cap_hard: float = config.get("single_source_survivor_cap_hard", 0.05)

    if not linkage.groups:
        return

    # Build target_attr -> [(source, column)] mapping.
    target_to_source_cols: dict[str, list[tuple[str, str]]] = {}
    for src, mapping in attr_mapping.items():
        for col, target in mapping.items():
            target_to_source_cols.setdefault(target, []).append((src, col))

    # Build TWO id_col-based lookups:
    #   * ``ref_id_to_idx``  — record_id -> reference-frame row index, for
    #     value reads from ``sources`` (the pre-K2 reference).
    #   * ``live_id_to_idx`` — record_id -> live-frame row index, for
    #     drop_mask reads/writes. The drop_mask is indexed by the LIVE
    #     frame's row index (post-K2 reset_index, post-K4 fab/demote).
    #     Without this second lookup, ``drop_mask.at[ref_idx, col]`` would
    #     silently read/write a different row's cell because K2's
    #     ``reset_index(drop=True)`` after row removal makes ref_idx and
    #     live_idx point at different rows.
    ref_id_to_idx: dict[str, dict[str, int]] = {}
    for source_name, df in sources.items():
        id_col = id_columns.get(source_name)
        if id_col and id_col in df.columns:
            lookup: dict[str, int] = {}
            for idx, rid in zip(df.index, df[id_col].astype(str)):
                lookup[rid] = idx
            ref_id_to_idx[source_name] = lookup

    live_frame = live_sources if live_sources is not None else sources
    live_id_to_idx: dict[str, dict[str, int]] = {}
    for source_name, df in live_frame.items():
        id_col = id_columns.get(source_name)
        if id_col and id_col in df.columns:
            lookup = {}
            for idx, rid in zip(df.index, df[id_col].astype(str)):
                lookup[rid] = idx
            live_id_to_idx[source_name] = lookup

    # Build a level-invariant ref-indexed mask if uniforms + target_rates
    # are provided. Constraints 2 + 3 then operate on this mask instead of
    # the live-indexed drop_mask. Decisions become independent of which
    # K3 call is currently running: the iteration scope (linkage.groups)
    # is level-invariant, value reads come from reference_sources, and
    # is_dropping reads come from ref_mask. K4-hard-removed entities
    # appear in ref_mask just like any other reference entity, so
    # Constraint 2's conflict-preserve and Constraint 3's surviving-count
    # both produce identical un-drop decisions across K3 easy / medium /
    # hard calls. Trade-off: phantom protection — if K4 hard later
    # removes a sibling whose value Constraint 2 thought was "going to
    # be dropped", the protection might be unnecessary. Detected by
    # R7.2's realised-vs-configured K3 drop-rate gap.
    use_ref_mask = uniforms is not None and target_rates is not None
    ref_mask: dict[str, pd.DataFrame] = {}
    if use_ref_mask:
        assert uniforms is not None  # mypy
        assert target_rates is not None
        for src_name, ref_df in sources.items():
            udf = uniforms.get(src_name)
            if udf is None:
                continue
            target = target_rates.get(src_name, {})
            mask_data: dict[str, np.ndarray] = {}
            for col in udf.columns:
                t = float(target.get(col, 0.0))
                u_col = udf[col].values
                if col in ref_df.columns:
                    ref_nonnull = ref_df[col].notna().values
                else:
                    ref_nonnull = np.zeros(len(ref_df), dtype=bool)
                mask_data[col] = ref_nonnull & (u_col < t)
            ref_mask[src_name] = pd.DataFrame(mask_data, index=ref_df.index)
        # Mirror Step 4b's fusion protection onto the ref-indexed mask.
        # fusion_protected_cells is already keyed by (src, ref_idx, col).
        if fusion_protected_cells:
            for src, r_idx, col in fusion_protected_cells:
                if src in ref_mask and col in ref_mask[src].columns:
                    if r_idx in ref_mask[src].index:
                        ref_mask[src].at[r_idx, col] = False

    # ``source_id_to_idx`` is the lookup that drives constraint iteration
    # and drop_mask access. When use_ref_mask is True, we use ref_id_to_idx
    # for both iteration scope AND ref_mask access (the ref_mask is keyed
    # by ref_idx). When False (legacy), we fall back to live_id_to_idx.
    source_id_to_idx = ref_id_to_idx if use_ref_mask else live_id_to_idx

    # --- Constraint 1: Fusion survivor floor ---
    # NOTE: The primary fusion survivor protection is pre-computed
    # level-independently in _compute_fusion_protected_cells() and applied
    # in Step 5b of apply_knob_03. This pass handles residual cases where
    # the pre-computed protection is insufficient (e.g., the protected cell
    # was null at baseline and not filled).

    # --- Constraint 2: Conflict-preserving drop ---
    for group_id, members in linkage.groups.items():
        for target_attr, source_cols in target_to_source_cols.items():
            # Collect current values and drop status.
            entries: list[tuple[str, str, int, Any, bool, str]] = []

            for member_source, member_id in members:
                if member_source not in source_id_to_idx:
                    continue
                idx_lookup = source_id_to_idx[member_source]
                if member_id not in idx_lookup:
                    continue
                # When use_ref_mask=True, ``row_idx`` == ref_idx and we
                # index ref_mask. When False (legacy), ``row_idx`` ==
                # live_idx and we index drop_mask. ``ref_idx`` is always
                # the reference-frame idx for value reads (level-invariant).
                row_idx = idx_lookup[member_id]
                ref_idx = ref_id_to_idx.get(member_source, {}).get(member_id)

                if use_ref_mask:
                    target_mask = ref_mask.get(member_source)
                else:
                    target_mask = drop_mask.get(member_source)
                if target_mask is None or row_idx not in target_mask.index:
                    continue

                src_col = None
                for sc_src, sc_col in source_cols:
                    if sc_src == member_source:
                        src_col = sc_col
                        break
                if src_col is None or src_col not in sources[member_source].columns:
                    continue

                if ref_idx is None or ref_idx not in sources[member_source].index:
                    continue
                val = sources[member_source].at[ref_idx, src_col]
                if _is_missing(val):
                    continue

                is_dropping = src_col in target_mask.columns and bool(
                    target_mask.at[row_idx, src_col]
                )
                entries.append(
                    (member_source, member_id, row_idx, val, is_dropping, src_col)
                )

            if len(entries) < 2:
                continue

            # Find distinct values (simple string comparison).
            value_groups: dict[str, list[int]] = {}
            for i, (_, _, _, val, _, _) in enumerate(entries):
                key = str(val).strip().lower()
                value_groups.setdefault(key, []).append(i)

            if len(value_groups) < 2:
                continue  # All agree — no conflict to preserve.

            # Count surviving distinct values.
            surviving_vals: set[str] = set()
            for val_key, indices in value_groups.items():
                for i in indices:
                    if not entries[i][4]:  # not being dropped
                        surviving_vals.add(val_key)
                        break

            # If fewer than 2 distinct values survive, protect one more.
            if len(surviving_vals) < 2:
                for val_key, indices in value_groups.items():
                    if val_key in surviving_vals:
                        continue
                    for i in indices:
                        src, mid, ridx, _, is_drop, col = entries[i]
                        if is_drop:
                            if use_ref_mask:
                                ref_mask[src].at[ridx, col] = False
                            else:
                                drop_mask[src].at[ridx, col] = False
                            skipped_log.append(
                                entity_id=mid,
                                source=src,
                                attribute=col,
                                transform_fn="drop",
                                transform_params={
                                    "reason": "conflict_preserve",
                                    "target_attr": target_attr,
                                },
                            )
                            surviving_vals.add(val_key)
                            break
                    if len(surviving_vals) >= 2:
                        break

    # --- Constraint 3: Single-source-survivor cap at hard ---
    if level == "hard":
        total_cells = 0
        single_survivor_cells = 0

        for group_id, members in linkage.groups.items():
            for target_attr, source_cols in target_to_source_cols.items():
                surviving_count = 0
                group_entries: list[tuple[str, str, int, str]] = []

                for member_source, member_id in members:
                    if member_source not in source_id_to_idx:
                        continue
                    idx_lookup = source_id_to_idx[member_source]
                    if member_id not in idx_lookup:
                        continue
                    row_idx = idx_lookup[member_id]
                    ref_idx = ref_id_to_idx.get(member_source, {}).get(member_id)
                    if use_ref_mask:
                        target_mask = ref_mask.get(member_source)
                    else:
                        target_mask = drop_mask.get(member_source)
                    if target_mask is None or row_idx not in target_mask.index:
                        continue

                    src_col = None
                    for sc_src, sc_col in source_cols:
                        if sc_src == member_source:
                            src_col = sc_col
                            break
                    if src_col is None or src_col not in sources[member_source].columns:
                        continue

                    if ref_idx is None or ref_idx not in sources[member_source].index:
                        continue
                    val = sources[member_source].at[ref_idx, src_col]
                    if _is_missing(val):
                        continue

                    is_dropping = src_col in target_mask.columns and bool(
                        target_mask.at[row_idx, src_col]
                    )
                    if not is_dropping:
                        surviving_count += 1
                    group_entries.append((member_source, member_id, row_idx, src_col))

                if group_entries:
                    total_cells += 1
                    if surviving_count == 1:
                        single_survivor_cells += 1

        if total_cells > 0:
            frac = single_survivor_cells / total_cells
            if frac > cap_hard:
                logger.warning(
                    "Single-source-survivor fraction %.3f exceeds cap %.3f; "
                    "rolling back drops",
                    frac,
                    cap_hard,
                )
                # Roll back: un-drop cells to reduce single-survivor count.
                # Simple strategy: iterate groups and un-drop the most recent
                # drop in single-survivor cells until cap is met.
                target_count = int(cap_hard * total_cells)
                excess = single_survivor_cells - target_count

                for group_id, members in linkage.groups.items():
                    if excess <= 0:
                        break
                    for target_attr, source_cols in target_to_source_cols.items():
                        if excess <= 0:
                            break
                        # Check if this is a single-survivor cell.
                        survivors_here: list[tuple[str, str, int, str]] = []
                        dropped_here: list[tuple[str, str, int, str]] = []

                        for member_source, member_id in members:
                            if member_source not in source_id_to_idx:
                                continue
                            idx_lookup = source_id_to_idx[member_source]
                            if member_id not in idx_lookup:
                                continue
                            row_idx = idx_lookup[member_id]
                            ref_idx = ref_id_to_idx.get(member_source, {}).get(
                                member_id
                            )
                            if use_ref_mask:
                                target_mask = ref_mask.get(member_source)
                            else:
                                target_mask = drop_mask.get(member_source)
                            if target_mask is None or row_idx not in target_mask.index:
                                continue
                            src_col = None
                            for sc_src, sc_col in source_cols:
                                if sc_src == member_source:
                                    src_col = sc_col
                                    break
                            if (
                                src_col is None
                                or src_col not in sources[member_source].columns
                            ):
                                continue
                            if (
                                ref_idx is None
                                or ref_idx not in sources[member_source].index
                            ):
                                continue
                            val = sources[member_source].at[ref_idx, src_col]
                            if _is_missing(val):
                                continue
                            is_dropping = src_col in target_mask.columns and bool(
                                target_mask.at[row_idx, src_col]
                            )
                            if is_dropping:
                                dropped_here.append(
                                    (member_source, member_id, row_idx, src_col)
                                )
                            else:
                                survivors_here.append(
                                    (member_source, member_id, row_idx, src_col)
                                )

                        if len(survivors_here) == 1 and dropped_here:
                            # Single survivor; un-drop one.
                            src, mid, ridx, col = dropped_here[0]
                            if use_ref_mask:
                                ref_mask[src].at[ridx, col] = False
                            else:
                                drop_mask[src].at[ridx, col] = False
                            skipped_log.append(
                                entity_id=mid,
                                source=src,
                                attribute=col,
                                transform_fn="drop",
                                transform_params={
                                    "reason": "single_source_cap",
                                },
                            )
                            excess -= 1

    # ---- Sync ref_mask un-drops back to live-indexed drop_mask ------------
    # Constraints 2 + 3 wrote to ref_mask (when use_ref_mask=True).
    # Translate the un-drops back to drop_mask via id_col → live_idx so
    # that Step 7 (drop execution) sees the constraint decisions. Cells
    # whose entity has no live_idx (K4-removed) are silently dropped from
    # the translation — they don't exist in the live frame so there's
    # nothing to write back.
    if use_ref_mask:
        for src_name, rmask in ref_mask.items():
            dmask = drop_mask.get(src_name)
            if dmask is None:
                continue
            id_lookup_ref = ref_id_to_idx.get(src_name, {})
            id_lookup_live = live_id_to_idx.get(src_name, {})
            for rid, r_idx in id_lookup_ref.items():
                live_idx = id_lookup_live.get(rid)
                if live_idx is None or live_idx not in dmask.index:
                    continue
                if r_idx not in rmask.index:
                    continue
                for col in rmask.columns:
                    if col in dmask.columns:
                        # AND the two: only un-drop (True→False), never
                        # add a drop. Step 4 + Step 4b's live-mask state
                        # is the upper bound; ref_mask's un-drops trim it.
                        dmask.at[live_idx, col] = bool(
                            dmask.at[live_idx, col]
                        ) and bool(rmask.at[r_idx, col])


# ---- Nesting enforcement --------------------------------------------------


def _enforce_nesting(
    easy: dict[str, pd.DataFrame],
    medium: dict[str, pd.DataFrame],
    hard: dict[str, pd.DataFrame],
) -> None:
    """Shrink *easy* and *medium* so ``D_easy ⊆ D_medium ⊆ D_hard``.

    Operates in place. We only un-drop cells (never add), so the
    shrinking step preserves every constraint 2/3 un-drop decision.
    """
    for src, hard_mask in hard.items():
        med_mask = medium.get(src)
        if med_mask is not None:
            for col in med_mask.columns:
                if col in hard_mask.columns:
                    med_mask[col] = med_mask[col].values & hard_mask[col].values
        easy_mask = easy.get(src)
        if easy_mask is not None and med_mask is not None:
            for col in easy_mask.columns:
                if col in med_mask.columns:
                    easy_mask[col] = easy_mask[col].values & med_mask[col].values


# ---- Main entry point -----------------------------------------------------


def apply_knob_03(
    domain: str,
    level: Literal["easy", "medium", "hard"],
    sources: dict[str, pd.DataFrame],
    config: dict[str, Any],
    linkage: EntityLinkage | None = None,
    fusion_gold_ids: set[str] | None = None,
    seed: int = 42,
    reference_sources: dict[str, pd.DataFrame] | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply Knob 03 attribute drop at the given level.

    Parameters
    ----------
    domain : str
        Domain name.
    level : {"easy", "medium", "hard"}
        Difficulty level.
    sources : dict[str, DataFrame]
        Source DataFrames (modified in place).
    config : dict
        Parsed K3 config YAML.
    linkage : EntityLinkage or None
        Cross-source entity linkage. If None, cross-source constraints
        and propagation fill are skipped.
    fusion_gold_ids : set[str] or None
        Fusion gold entity IDs for survivor floor constraint.
    seed : int, default 42
        Master seed for RNG.
    reference_sources : dict[str, DataFrame] or None, default None
        Level-invariant reference sources used for the per-cell
        ``is_non_null`` gate in the drop mask. When generating variants
        across all three difficulty levels, upstream knobs (notably K4
        propagate-and-paraphrase at easy) can render the "currently
        non-null" set level-dependent, which breaks the knob card's
        ``D_easy ⊆ D_medium ⊆ D_hard`` nesting contract. Passing a
        pre-K4 snapshot here restores nesting: cells null in the
        reference are never candidates for K3 drops at any level.
        Alignment is by DataFrame index; rows present in *sources* but
        not in *reference_sources* (e.g. K4-fabricated duplicates) are
        treated as null-in-reference and therefore excluded from K3.
        If None, falls back to the current ``sources`` state (legacy
        behaviour — acceptable for single-level invocations where K4
        is either identity or runs after K3).

    Returns
    -------
    sources : dict[str, DataFrame]
        DataFrames with dropped cells (NaN).
    provenance_df : DataFrame
        Provenance log (drops + fills).
    skipped_df : DataFrame
        Skipped-cell audit log.
    baseline_df : DataFrame
        Measured baseline missingness.
    """
    if level not in VALID_LEVELS:
        raise ValueError(f"Invalid level: {level!r}. Valid: {VALID_LEVELS}")

    if linkage is None:
        linkage = EntityLinkage()
    if fusion_gold_ids is None:
        fusion_gold_ids = set()

    managed_columns = _get_managed_columns(config)
    id_columns: dict[str, str] = config.get("id_columns", {})

    # --- Step 1: Measure baseline ---
    # When *reference_sources* is provided (canonical S1 generation), we
    # measure baseline from the level-invariant pre-K4 snapshot so that
    # target rates are identical across easy/medium/hard invocations —
    # the only source of cross-call drift in T_level is then the
    # deterministic compress/identity/stretch transform, and
    # D_easy ⊆ D_medium ⊆ D_hard holds across separate level calls
    # (not just within a single call, as the card's shared-uniform
    # argument establishes). When reference_sources is None, falls back
    # to live post-K4 measurement per the card's default contract.
    baseline_source = reference_sources if reference_sources is not None else sources
    baseline = measure_missingness(baseline_source, managed_columns)
    baseline_df = baseline_to_dataframe(baseline)
    logger.info(
        "Baseline missingness measured for %d source-attribute pairs", len(baseline_df)
    )

    # --- Step 2: Compute target rates ---
    targets = compute_target_rates(baseline, level, config)

    # --- Step 2b: Pre-compute level-independent fusion survivor protection ---
    # This uses the ORIGINAL sources (before fill) so the protection set is
    # the same at all levels, ensuring D_easy ⊆ D_medium ⊆ D_hard.
    # When reference_sources is provided we resolve the carrier against the
    # pristine pre-knob snapshot so the protected cell set is identical
    # across easy/medium/hard invocations (otherwise the closest-to-gold
    # carrier can drift between calls as K1/K5/K6 perturbations differ).
    protection_sources = reference_sources if reference_sources is not None else sources
    fusion_protected = _compute_protected_cells(
        protection_sources, linkage, fusion_gold_ids, config, domain=domain
    )

    # ``fusion_protected`` carries (source, ref_idx, col) triples where
    # ``ref_idx`` is a *reference-frame* row index. The drop_masks are
    # indexed by the *live-frame* row index (post-K2 reset_index +
    # post-K4 demote/fab). K2's ``reset_index(drop=True)`` after
    # row removal means ref_idx and live_idx point at different rows for
    # any surviving entity. Pre-build a per-source ``ref_idx → live_idx``
    # translation via id_col so Step 4b applies protection at the
    # correct mask position, not at a coincidentally-overlapping idx.
    ref_to_live_idx: dict[str, dict[int, int]] = {}
    if reference_sources is not None:
        for src_name, ref_df in reference_sources.items():
            id_col = id_columns.get(src_name)
            if not id_col or id_col not in ref_df.columns:
                continue
            live_df = sources.get(src_name)
            if live_df is None or id_col not in live_df.columns:
                continue
            live_id_to_idx = {
                rid: idx for idx, rid in zip(live_df.index, live_df[id_col].astype(str))
            }
            ref_to_live_idx[src_name] = {}
            for r_idx, rid in zip(ref_df.index, ref_df[id_col].astype(str)):
                live_idx = live_id_to_idx.get(rid)
                if live_idx is not None:
                    ref_to_live_idx[src_name][int(r_idx)] = int(live_idx)

    # --- Step 3: Draw shared uniforms (same RNG for all levels) ---
    # Uniforms are drawn BEFORE propagate_fill so the per-cell uniforms
    # are independent of fill state; this keeps D_easy ⊆ D_medium ⊆ D_hard
    # nested regardless of outer call level. When *reference_sources* is
    # provided we draw uniforms against the pristine reference (same `n`
    # at all three level invocations), so the uniform draws — and hence
    # the drop masks — are identical across cross-call level variants
    # for cells present in the reference.
    rng = make_rng(domain, variant="shared", knob=3, master_seed=seed)
    uniform_source = reference_sources if reference_sources is not None else sources
    uniforms = draw_shared_uniforms(uniform_source, managed_columns, rng)

    # --- Step 4: Compute initial drop masks for ALL levels ---
    # Uses pre-K4 ``is_non_null`` (from *reference_sources* when provided)
    # so that cells fabricated or filled by an upstream K4 at one level
    # are not eligible for drop at *any* level. Without this, K4's easy
    # propagate-and-paraphrase fills cells that K3 then marks at easy
    # but not at medium/hard, breaking ``D_easy ⊆ D_medium ⊆ D_hard``.
    targets_by_level: dict[str, dict[str, dict[str, float]]] = {
        lvl: compute_target_rates(baseline, lvl, config) for lvl in VALID_LEVELS
    }
    drop_masks: dict[str, dict[str, pd.DataFrame]] = {}
    for lvl, t_by_src in targets_by_level.items():
        mask: dict[str, pd.DataFrame] = {}
        for source_name, udf in uniforms.items():
            target_rates = t_by_src.get(source_name, {})
            mask_data: dict[str, np.ndarray] = {}
            current_df = sources[source_name]
            ref_df = (
                reference_sources.get(source_name)
                if reference_sources is not None
                else None
            )
            # Build per-source id-based lookups so we can align ref values +
            # uniforms to current rows by *record id*, not by index label.
            # K2 calls ``reset_index(drop=True)`` after removing rows and
            # ``pd.concat(..., ignore_index=True)`` after appending niche
            # entities, so post-K2 indices have been re-sequenced. K4 fab
            # then appends new rows at ``new_idx = max(idx) + 1``, which
            # can collide with a *different* pre-K2 row's idx. Aligning by
            # idx would silently treat a K4-fab cell as a real pre-K2 cell
            # — and the mask would think it's drop-eligible. Aligning by
            # ``id_col`` value is index-label invariant: a K4-fab row's id
            # (``k04__fab__<src>__<entity>``) and a K2-niche row's id are
            # both absent from the pre-K2 reference, so they map to NaN
            # and are excluded from the drop mask.
            id_col = id_columns.get(source_name)
            current_ids = (
                current_df[id_col].astype(str).values
                if id_col and id_col in current_df.columns
                else None
            )
            ref_ids = (
                ref_df[id_col].astype(str).values
                if ref_df is not None and id_col and id_col in ref_df.columns
                else None
            )
            for col in udf.columns:
                t = target_rates.get(col, 0.0)
                if (
                    current_ids is not None
                    and ref_ids is not None
                    and col in udf.columns
                ):
                    u_by_id = pd.Series(udf[col].values, index=ref_ids)
                    u_aligned = u_by_id.reindex(current_ids)
                    below_threshold = (u_aligned < t).fillna(False).values
                else:
                    u_aligned = udf[col].reindex(current_df.index)
                    below_threshold = (u_aligned < t).fillna(False).values
                if ref_df is not None and col in ref_df.columns:
                    if current_ids is not None and ref_ids is not None:
                        ref_by_id = pd.Series(ref_df[col].values, index=ref_ids)
                        ref_aligned = ref_by_id.reindex(current_ids)
                        ref_non_null = ref_aligned.notna().values
                    else:
                        ref_non_null = (
                            ref_df[col].reindex(current_df.index).notna().values
                        )
                else:
                    ref_non_null = current_df[col].notna().values
                mask_data[col] = ref_non_null & below_threshold
            mask[source_name] = pd.DataFrame(mask_data, index=current_df.index)
        drop_masks[lvl] = mask
    drop_mask = drop_masks[level]

    # --- Step 4b: Apply level-independent fusion survivor protection ---
    skipped_log = ProvenanceLog(knob=3, level=level)
    for lvl, mask in drop_masks.items():
        is_current_level = lvl == level
        for src, ridx, col in fusion_protected:
            if src not in mask or col not in mask[src].columns:
                continue
            # Translate the reference-frame ridx to the live-frame idx
            # (K2 resets indices after row removal, so they differ for
            # any surviving entity). If the entity no longer exists in
            # the live frame (K2 / K4 removed it), the protection is
            # moot — skip silently.
            live_idx = ref_to_live_idx.get(src, {}).get(int(ridx))
            if live_idx is None:
                # When reference_sources is provided but the entity has
                # no live_idx, it was removed by K2 or K4 — the cell can
                # no longer be dropped (it isn't in the live frame), so
                # protection is moot. Skip silently. Do NOT fall back to
                # using ``ridx`` as live_idx: ridx is a *reference*-frame
                # index and after K2's reset_index it points to a
                # different row in the live frame, so writing
                # mask[live=ridx, col] = False would un-drop the WRONG
                # cell. The fallback below is reserved for legacy callers
                # that pass reference_sources=None (idx spaces coincide).
                if reference_sources is None:
                    if ridx not in mask[src].index:
                        continue
                    live_idx = int(ridx)
                else:
                    continue
            elif live_idx not in mask[src].index:
                continue
            if mask[src].at[live_idx, col]:
                mask[src].at[live_idx, col] = False
                if is_current_level:
                    entity_id = ""
                    id_col = id_columns.get(src)
                    if (
                        id_col
                        and id_col in sources[src].columns
                        and live_idx in sources[src].index
                    ):
                        entity_id = str(sources[src].at[live_idx, id_col])
                    skipped_log.append(
                        entity_id=entity_id,
                        source=src,
                        attribute=col,
                        transform_fn="drop",
                        transform_params={
                            "reason": "fusion_survivor_floor",
                        },
                    )

    # --- Step 5: Apply remaining constraints at every level ---
    # Running constraints at each level lets us enforce nesting by
    # shrinking afterwards. Only the current level's skipped events
    # are logged. When reference_sources is provided we evaluate
    # constraints (conflict-preserving drop, single-source-survivor
    # cap) against the level-invariant reference so the resulting
    # mask is identical across easy/medium/hard invocations.
    constraint_sources = reference_sources if reference_sources is not None else sources
    # Build per-level target-rate maps so apply_constraints can build a
    # level-invariant ref-indexed mask. Without these, apply_constraints
    # falls back to its legacy live-mask path (used by standalone K3
    # tests). The new path closes the cross-K3-call divergence by making
    # Constraints 2 + 3 operate on a level-invariant mask, independent
    # of which K3 call is currently running.
    for lvl, mask in drop_masks.items():
        lvl_skipped = skipped_log if lvl == level else ProvenanceLog(knob=3, level=lvl)
        apply_constraints(
            mask,
            constraint_sources,
            linkage,
            fusion_gold_ids,
            config,
            lvl,
            lvl_skipped,
            live_sources=sources,
            uniforms=uniforms,
            target_rates=targets_by_level.get(lvl),
            fusion_protected_cells=fusion_protected,
        )

    # --- Step 5b: Enforce D_easy ⊆ D_medium ⊆ D_hard by shrinking ---
    # Constraint 2 (conflict-preserve) can protect a cell at a higher
    # level that isn't protected at a lower level; constraint 3 (hard
    # rollback) un-drops cells only at hard. Both break the invariant
    # that D is monotone across levels. Shrinking D_medium and D_easy
    # to subsets of D_hard / D_medium restores the invariant without
    # re-breaking the constraints (we only un-drop cells, never add).
    _enforce_nesting(drop_masks["easy"], drop_masks["medium"], drop_masks["hard"])

    # --- Step 6: Propagation fill (easy only, after mask construction) ---
    # Moved after mask construction so filled cells (which are null at
    # medium/hard) can never be in any level's drop mask. This preserves
    # nesting: drops at easy are a subset of drops at medium.
    prov_log = ProvenanceLog(knob=3, level=level)
    if level == "easy":
        propagate_fill(sources, linkage, config, baseline, prov_log)

    # --- Step 7: Execute drops ---
    for source_name, mask_df in drop_mask.items():
        id_col = id_columns.get(source_name)
        df = sources[source_name]
        for col in mask_df.columns:
            drop_indices = mask_df.index[mask_df[col]]
            for idx in drop_indices:
                original_val = df.at[idx, col]
                entity_id = ""
                if id_col and id_col in df.columns:
                    entity_id = str(df.at[idx, id_col])

                cls = _get_attr_class(config, source_name, col)
                b_rate = baseline.get(source_name, {}).get(col, 0.0)
                t_rate = targets.get(source_name, {}).get(col, 0.0)
                transform = config["transform_per_level"][level]

                reason = (
                    "floor_rate" if transform in ("compress", "identity") else "stretch"
                )

                prov_log.append(
                    entity_id=entity_id,
                    source=source_name,
                    attribute=col,
                    original_value=str(original_val),
                    new_value="",
                    transform_fn="drop",
                    transform_params={
                        "reason": reason,
                        "baseline_rate": round(b_rate, 4),
                        "target_rate": round(t_rate, 4),
                    },
                )

            # Apply the drops.
            df.loc[mask_df[col], col] = np.nan

    # --- Build output DataFrames ---
    from usecases_synthetic.lib.provenance import PROVENANCE_COLUMNS

    if len(prov_log) > 0:
        provenance_df = pd.DataFrame(
            [row.as_dict() for row in prov_log._rows],
            columns=PROVENANCE_COLUMNS,
        )
    else:
        provenance_df = pd.DataFrame(columns=PROVENANCE_COLUMNS)

    if len(skipped_log) > 0:
        skipped_df = pd.DataFrame(
            [row.as_dict() for row in skipped_log._rows],
            columns=PROVENANCE_COLUMNS,
        )
    else:
        skipped_df = pd.DataFrame(columns=PROVENANCE_COLUMNS)

    return sources, provenance_df, skipped_df, baseline_df


# ---- Output writing -------------------------------------------------------


def write_outputs(
    baseline_df: pd.DataFrame,
    provenance_df: pd.DataFrame,
    skipped_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Write K3 artifacts to disk.

    Parameters
    ----------
    baseline_df : DataFrame
        Measured baseline missingness.
    provenance_df : DataFrame
        Provenance log.
    skipped_df : DataFrame
        Skipped-cell audit.
    output_dir : Path
        Variant directory root.
    """
    baseline_dir = output_dir / "output" / "baselines"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    baseline_df.to_csv(baseline_dir / "knob_03_baseline_missingness.csv", index=False)

    prov_dir = output_dir / "output" / "provenance"
    prov_dir.mkdir(parents=True, exist_ok=True)
    provenance_df.to_csv(prov_dir / "knob_03_attribute_drop.csv", index=False)
    skipped_df.to_csv(prov_dir / "knob_03_skipped.csv", index=False)

    logger.info(
        "Wrote K3 outputs: baseline=%d rows, provenance=%d rows, skipped=%d rows",
        len(baseline_df),
        len(provenance_df),
        len(skipped_df),
    )


# ---- CLI -------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Apply Knob 03 — Per-source Attribute Drop Rate",
    )
    parser.add_argument("--domain", required=True, help="Domain name (e.g. companies)")
    parser.add_argument(
        "--level", required=True, choices=VALID_LEVELS, help="Difficulty level"
    )
    parser.add_argument("--seed", type=int, default=42, help="Master RNG seed")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Variant output directory",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    domain: str = args.domain
    level: str = args.level
    seed: int = args.seed
    output_dir: Path = args.output_dir or (
        REPO_ROOT / "usecases_synthetic" / "output" / domain / level
    )

    logger.info(
        "Knob 03: domain=%s level=%s seed=%d output=%s",
        domain,
        level,
        seed,
        output_dir,
    )

    config = load_knob_config(3, domain)
    domain_config = load_domain_config(domain)
    sources = load_domain_sources(domain)

    id_columns: dict[str, str] = config.get("id_columns", {})
    linkage = build_entity_linkage(domain_config, id_columns, sources)
    fusion_gold_ids = _build_fusion_gold_ids(domain_config)

    logger.info(
        "Entity linkage: %d multi-source groups, %d fusion gold IDs",
        len(linkage.groups),
        len(fusion_gold_ids),
    )

    sources, provenance_df, skipped_df, baseline_df = apply_knob_03(
        domain=domain,
        level=level,
        sources=sources,
        config=config,
        linkage=linkage,
        fusion_gold_ids=fusion_gold_ids,
        seed=seed,
    )

    write_outputs(baseline_df, provenance_df, skipped_df, output_dir)

    for src_name, df in sources.items():
        total = df.shape[0] * df.shape[1]
        nulls = int(df.isna().sum().sum())
        logger.info(
            "  %s: %d/%d cells null (%.1f%%)",
            src_name,
            nulls,
            total,
            100.0 * nulls / total if total else 0,
        )


if __name__ == "__main__":
    main()
