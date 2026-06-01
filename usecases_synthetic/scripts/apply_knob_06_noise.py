#!/usr/bin/env python3
"""Apply Knob 06 — Value Noise Injection.

Cell-level corruption using FEBRL/Christen-Vatsalan operators: typos, OCR
confusions, truncations, whitespace/case corruption.  These are **errors**,
not legitimate variants (that's Knob 1).

See ``knobs/knob_06_value_noise.md`` for the full specification.

Usage
-----
::

    python usecases_synthetic/scripts/apply_knob_06_noise.py \\
        --domain companies --level easy

Inputs
------
- Source DataFrames from ``usecases/<domain>/input/data/``
- Fusion gold from ``usecases/<domain>/input/fusion/``
- Per-domain config at ``usecases_synthetic/config/knob_06_noise/<domain>.yaml``
- Shared operator tables under ``usecases_synthetic/config/knob_06_noise/_tables/``

Outputs (under *output_dir*)
------
- Noised source DataFrames (returned in-memory)
- Provenance CSV at ``<output_dir>/output/provenance/knob_06_noise.csv``
- Skipped-cell audit at ``<output_dir>/output/provenance/knob_06_skipped.csv``
"""

from __future__ import annotations

import argparse
import json
import logging
import re
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
    USECASES_DIR,
    VALID_LEVELS,
    load_domain_config,
)
from usecases_synthetic.lib.loaders import load_domain_sources, read_em_gold_csv
from usecases_synthetic.lib.noise_operators import (
    OPERATOR_REGISTRY,
    Taxonomy,
    case_corrupt,
    load_taxonomy,
    numeric_jitter_within_cap,
    ocr_confuse,
    taxonomy_walk,
    truncate,
    typo_substitute,
    whitespace_corrupt,
)
from usecases_synthetic.lib.fusion_silver_targets import (
    PROTECTION_SOURCES,
    resolve_protection_sources,
)
from usecases_synthetic.lib.protection import (
    ToleranceSpec,
    cell_has_close_survivor,
    fusion_cell_tolerance,
)
from usecases_synthetic.lib.provenance import PROVENANCE_COLUMNS, ProvenanceLog
from usecases_synthetic.lib.rng import (
    cell_selection_uniform,
    make_rng,
    spawn_sub_rng,
)

logger = logging.getLogger(__name__)


# ---- Skipped-cell audit ---------------------------------------------------

SKIPPED_COLUMNS = [
    "entity_id",
    "source",
    "attribute",
    "original_value",
    "reason",
    "knob",
    "level",
]


class SkippedLog:
    """Accumulates skipped-cell audit rows for Knob 06."""

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
        reason: str,
    ) -> None:
        self._rows.append(
            {
                "entity_id": entity_id,
                "source": source,
                "attribute": attribute,
                "original_value": original_value,
                "reason": reason,
                "knob": self.knob,
                "level": self.level,
            }
        )

    def to_dataframe(self) -> pd.DataFrame:
        if not self._rows:
            return pd.DataFrame(columns=SKIPPED_COLUMNS)
        return pd.DataFrame(self._rows, columns=SKIPPED_COLUMNS)


# ---- Config loading -------------------------------------------------------


def load_knob_06_config(domain: str) -> dict[str, Any]:
    """Load the Knob 06 noise config for *domain*."""
    path = CONFIG_DIR / "knob_06_noise" / f"{domain}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Knob 06 noise config not found: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---- Entity linkage (reused from K3 pattern) ------------------------------


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
    domain: str,
    id_columns: dict[str, str],
    sources: dict[str, pd.DataFrame],
) -> dict[str, list[tuple[str, str]]]:
    """Build entity groups from EM gold correspondences.

    Returns a mapping from canonical group ID to list of
    ``(source_name, record_id)`` pairs.
    """
    domain_config = load_domain_config(domain)
    em_dir = domain_config.em_dir()
    if not em_dir.exists():
        return {}

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

    all_ids: set[str] = set()
    for id1, id2 in pairs:
        all_ids.add(id1)
        all_ids.add(id2)

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

    from collections import defaultdict

    groups_raw: dict[str, list[str]] = defaultdict(list)
    for rid in all_ids:
        root = _find(parent, rid)
        groups_raw[root].append(rid)

    groups: dict[str, list[tuple[str, str]]] = {}
    for root, members in groups_raw.items():
        group_members: list[tuple[str, str]] = []
        for m in members:
            src = id_to_source.get(m, "unknown")
            group_members.append((src, m))
        groups[root] = group_members

    return groups


def _build_id_to_entity_group(
    entity_groups: dict[str, list[tuple[str, str]]],
) -> dict[str, str]:
    """Build reverse index: record_id -> entity_group_id."""
    index: dict[str, str] = {}
    for group_id, members in entity_groups.items():
        for _src, rid in members:
            index[rid] = group_id
    return index


# ---- Operator selection & dispatch ----------------------------------------


def _normalise_weights(
    operator_mix: dict[str, float],
) -> tuple[list[str], list[float]]:
    """Normalise operator weights to a probability vector."""
    ops = sorted(operator_mix.keys())
    weights = [operator_mix[o] for o in ops]
    total = sum(weights)
    probs = [w / total for w in weights]
    return ops, probs


def _apply_operator(
    operator_name: str,
    value: str,
    rng: np.random.Generator,
    config: dict[str, Any],
    level: str,
    *,
    taxonomy: Taxonomy | None = None,
) -> tuple[str, dict[str, Any]] | None:
    """Dispatch a single noise operator on a cell value."""
    if operator_name == "typo_substitute":
        n_edits = config.get("max_edits_per_cell", {}).get(level, 1)
        return typo_substitute(value, rng, n_edits=n_edits, use_adjacency=True)
    elif operator_name == "ocr_confuse":
        n_chars = config.get("max_ocr_per_cell", {}).get(level, 1)
        return ocr_confuse(value, rng, n_chars=n_chars)
    elif operator_name == "truncate":
        max_chars = config.get("max_truncate_chars", {}).get(level, 3)
        return truncate(value, rng, max_truncate_chars=max_chars)
    elif operator_name == "whitespace_corrupt":
        return whitespace_corrupt(value, rng)
    elif operator_name == "case_corrupt":
        return case_corrupt(value, rng)
    elif operator_name == "taxonomy_walk":
        if taxonomy is None:
            return None
        return taxonomy_walk(value, rng, taxonomy=taxonomy, direction="either")
    else:
        logger.warning("Unknown operator: %s", operator_name)
        return None


# ---- Taxonomy + numeric attribute resolution -----------------------------

# Maximum retries when a numeric jitter exceeds the per-attribute cap. The
# dispatcher draws a fresh operator each retry; on exhaustion the cell is
# logged to skipped with reason=numeric_jitter_exhausted_retries.
_MAX_JITTER_RETRIES = 5


def _resolve_taxonomies(
    config: dict[str, Any],
) -> tuple[dict[tuple[str, str], Taxonomy], dict[str, Taxonomy]]:
    """Resolve per-(source, column) taxonomy bindings declared in *config*.

    Config block shape::

        taxonomies:
          <name>:
            csv: <path-relative-to-USECASES_DIR>
            levels: [Top, Middle, Leaf]
            bind:
              <source>:
                <column>: <name>
              ...

    Returns ``(per_column_map, taxonomies_by_name)``. ``per_column_map`` keys
    are ``(source, column)`` tuples.
    """
    raw = config.get("taxonomies") or {}
    if not raw:
        return {}, {}

    by_name: dict[str, Taxonomy] = {}
    for name, spec in raw.items():
        csv_rel = spec.get("csv")
        levels = spec.get("levels")
        if not csv_rel or not levels:
            continue
        csv_path = USECASES_DIR / csv_rel
        if not csv_path.exists():
            logger.warning(
                "Taxonomy %r CSV not found at %s -- skipping", name, csv_path
            )
            continue
        by_name[name] = load_taxonomy(name=name, csv_path=csv_path, levels=list(levels))

    per_col: dict[tuple[str, str], Taxonomy] = {}
    # Bindings can be declared either inside each taxonomy spec (under
    # ``bind``) or in a top-level ``taxonomy_bindings`` block keyed by
    # source -> column -> taxonomy name. Either form resolves the same way.
    for name, spec in raw.items():
        bind = spec.get("bind") or {}
        for source, cols in bind.items():
            for col, target_name in (cols or {}).items():
                tax = by_name.get(target_name)
                if tax is not None:
                    per_col[(source, col)] = tax
    flat = config.get("taxonomy_bindings") or {}
    for source, cols in flat.items():
        for col, target_name in (cols or {}).items():
            tax = by_name.get(target_name)
            if tax is not None:
                per_col[(source, col)] = tax
    return per_col, by_name


def _resolve_numeric_attributes(
    config: dict[str, Any],
) -> dict[tuple[str, str], str]:
    """Resolve per-(source, column) numeric type tags from config.

    Config block::

        numeric_attributes:
          <source>:
            <column>: continuous | year | date

    Returns a flat ``{(source, column): type}`` map.
    """
    raw = config.get("numeric_attributes") or {}
    out: dict[tuple[str, str], str] = {}
    for source, cols in raw.items():
        for col, kind in (cols or {}).items():
            kstr = str(kind).strip().lower()
            if kstr in {"continuous", "year", "date"}:
                out[(source, col)] = kstr
            else:
                logger.warning(
                    "Unknown numeric type %r for %s.%s -- ignoring",
                    kind,
                    source,
                    col,
                )
    return out


# ---- Floor checks ---------------------------------------------------------


def _check_clean_primary_floor(
    entity_group_id: str,
    entity_groups: dict[str, list[tuple[str, str]]],
    noised_primaries: dict[str, set[str]],
) -> bool:
    """Check if noising this cell would violate the clean-primary floor.

    Returns True if it is SAFE to noise (at least one other source in the
    entity group still has a clean primary).
    """
    members = entity_groups.get(entity_group_id, [])
    if len(members) <= 1:
        # Single-source entity: never noise primary.
        return False
    clean_count = sum(
        1 for src, rid in members if rid not in noised_primaries.get(src, set())
    )
    # Need at least 2 clean (1 surviving after this noise).
    return clean_count >= 2


class _ClosenessContext:
    """Per-call resources for the closeness contract (Pending #5).

    Bundles the fusion-protected entity-id set, per-(entity, canonical
    attribute) target value lookup, the canonical → (source, source-col)
    reverse index, per-source id→row-index lookups, and a memoised
    tolerance resolver. Built once per ``apply_knob_06`` call.
    """

    def __init__(
        self,
        domain: str,
        sources: dict[str, pd.DataFrame],
        attr_mapping: dict[str, dict[str, str]],
        id_columns: dict[str, str],
        tolerance_overrides: dict[str, dict[str, float | str]] | None,
        *,
        protection_source: str = "gold",
        surviving_record_ids: set[str] | None = None,
    ) -> None:
        if protection_source not in PROTECTION_SOURCES:
            raise ValueError(
                f"protection_source must be one of {PROTECTION_SOURCES}; "
                f"got {protection_source!r}"
            )
        self.domain = domain
        self.attr_mapping = attr_mapping
        self.protection_source = protection_source
        self.protected_ids, self.target_values = resolve_protection_sources(
            domain, protection_source, surviving_record_ids
        )
        self.tolerance_overrides = tolerance_overrides or {}

        # canonical_attr -> [(source_name, source_col), ...]
        rev: dict[str, list[tuple[str, str]]] = {}
        for src, mapping in attr_mapping.items():
            for src_col, canonical in (mapping or {}).items():
                rev.setdefault(canonical, []).append((src, src_col))
        self.canonical_to_source_cols = rev

        # source -> {record_id: row_idx}
        self.id_to_idx: dict[str, dict[str, int]] = {}
        for src, df in sources.items():
            id_col = id_columns.get(src)
            if id_col and id_col in df.columns:
                self.id_to_idx[src] = {
                    str(rid): idx for idx, rid in enumerate(df[id_col].astype(str))
                }

        self._tol_cache: dict[str, ToleranceSpec] = {}

    def tolerance_for(self, canonical_attr: str) -> ToleranceSpec:
        cached = self._tol_cache.get(canonical_attr)
        if cached is not None:
            return cached
        spec = fusion_cell_tolerance(
            self.domain, canonical_attr, self.tolerance_overrides
        )
        self._tol_cache[canonical_attr] = spec
        return spec

    def target_for(self, entity_id: str, canonical_attr: str) -> list[str]:
        return self.target_values.get(entity_id, {}).get(canonical_attr, [])


def _check_close_survivor_floor(
    *,
    entity_id: str,
    group_members: list[tuple[str, str]],
    current_source: str,
    current_col: str,
    candidate_value: str,
    sources_in_progress: dict[str, pd.DataFrame],
    ctx: _ClosenessContext,
) -> bool:
    """Closeness contract — Pending #5 (locked 2026-05-06).

    Returns True iff committing *candidate_value* into
    ``(current_source, current_col)`` would still leave ≥ 1 record
    across all sources mapped to the same canonical attribute that is
    "close enough" to a fusion target value for this entity.

    For non-fusion-protected entities (or entities without a target
    value for this canonical attribute), the gate is vacuously True.
    """
    canonical_attr = ctx.attr_mapping.get(current_source, {}).get(current_col)
    if canonical_attr is None:
        return True

    # Find every member of the group whose record IDs map to a
    # source-column under this canonical attribute. Pick the union of
    # protected entity IDs from group members (if any member is
    # protected, the entity is protected).
    protected_entity_id: str | None = None
    for _src, mid in group_members:
        if mid in ctx.protected_ids:
            protected_entity_id = mid
            break
    if protected_entity_id is None:
        return True

    target_values = ctx.target_for(protected_entity_id, canonical_attr)
    if not target_values:
        return True

    tolerance = ctx.tolerance_for(canonical_attr)
    surviving: list[str | None] = []
    src_cols = ctx.canonical_to_source_cols.get(canonical_attr, [])

    for member_src, member_rid in group_members:
        # Find the source-col for this member that maps to the canonical.
        for sc_src, sc_col in src_cols:
            if sc_src != member_src:
                continue
            df = sources_in_progress.get(member_src)
            if df is None or sc_col not in df.columns:
                continue
            idx_lookup = ctx.id_to_idx.get(member_src, {})
            row_idx = idx_lookup.get(member_rid)
            if row_idx is None:
                continue
            if (
                member_src == current_source
                and member_rid == entity_id
                and sc_col == current_col
            ):
                # The cell about to be committed: use the candidate value.
                surviving.append(candidate_value)
            else:
                val = df.iat[row_idx, df.columns.get_loc(sc_col)]
                if pd.isna(val):
                    continue
                surviving.append(str(val))
            break

    return cell_has_close_survivor(target_values, surviving, tolerance)


# ---- Cleanup operator (easy only) ----------------------------------------


def _apply_cleanup_rules(
    sources: dict[str, pd.DataFrame],
    config: dict[str, Any],
    id_columns: dict[str, str],
    prov: ProvenanceLog,
) -> dict[str, pd.DataFrame]:
    """Apply easy-level cleanup rules (revert known baseline noise)."""
    rules = config.get("cleanup_rules", [])
    if not rules:
        return sources

    result: dict[str, pd.DataFrame] = {}
    for source_name, df in sources.items():
        new_df = df.copy()
        new_df.attrs = df.attrs.copy()
        id_col = id_columns.get(source_name)

        for rule in rules:
            if rule["source"] != source_name:
                continue
            attr = rule["attribute"]
            if attr not in new_df.columns:
                continue
            pattern = re.compile(rule["pattern"])
            replacement = rule["replacement"]

            for idx in range(len(new_df)):
                cell_value = new_df.iloc[idx][attr]
                if pd.isna(cell_value):
                    continue
                cell_str = str(cell_value)
                cleaned = pattern.sub(replacement, cell_str)
                if cleaned != cell_str:
                    entity_id = (
                        str(new_df.iloc[idx][id_col])
                        if id_col and id_col in new_df.columns
                        else str(idx)
                    )
                    new_df.iat[idx, new_df.columns.get_loc(attr)] = cleaned
                    prov.append(
                        entity_id=entity_id,
                        source=source_name,
                        attribute=attr,
                        original_value=cell_str,
                        new_value=cleaned,
                        transform_fn="cleanup",
                        transform_params={
                            "pattern": rule["pattern"],
                            "replacement": replacement,
                        },
                    )

        result[source_name] = new_df
    return result


# ---- Core dispatcher ------------------------------------------------------


def apply_knob_06(
    domain: str,
    level: Literal["easy", "medium", "hard"],
    sources: dict[str, pd.DataFrame],
    config: dict[str, Any],
    entity_groups: dict[str, list[tuple[str, str]]] | None = None,
    collision_index: CollisionIndex | None = None,
    seed: int = 42,
    protection_source: str = "gold",
    surviving_record_ids: set[str] | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """Apply Knob 06 value noise injection to source DataFrames.

    Parameters
    ----------
    domain : str
        Domain name.
    level : {"easy", "medium", "hard"}
        Difficulty level.
    sources : dict[str, DataFrame]
        Source DataFrames keyed by source name.
    config : dict
        Parsed Knob 06 YAML.
    entity_groups : dict or None
        Entity linkage groups (built from EM gold). If None, floor checks
        are skipped.
    collision_index : CollisionIndex or None
        Cell-collision index from prior knobs.
    seed : int, default 42
        Master seed.

    Returns
    -------
    noised_sources : dict[str, DataFrame]
        DataFrames with noise injected. ``attrs`` preserved.
    provenance_df : DataFrame
        Provenance log.
    skipped_df : DataFrame
        Skipped-cell audit.
    """
    if level not in VALID_LEVELS:
        raise ValueError(f"Invalid level: {level!r}. Valid: {VALID_LEVELS}")

    rng = make_rng(domain, level, knob=6, master_seed=seed)
    prov = ProvenanceLog(knob=6, level=level)
    skipped = SkippedLog(knob=6, level=level)

    id_columns: dict[str, str] = config.get("id_columns", {})
    attr_classes: dict[str, dict[str, str]] = config.get("attribute_classes", {})
    noise_rates = config["noise_rates_per_level"][level]
    operator_mix = config["operator_mix"][level]
    attr_mapping: dict[str, dict[str, str]] = config.get("attribute_mapping", {})

    op_names, op_probs = _normalise_weights(operator_mix)

    # Resolve taxonomies + numeric attribute typing once per call.
    taxonomies_by_col, _ = _resolve_taxonomies(config)
    numeric_attrs = _resolve_numeric_attributes(config)
    jitter_cap = float(config.get("numeric_jitter_max_relative", 0.02))

    # Closeness contract (Pending #5): replaces the strict clean-survivor
    # floor with an "≥1 surviving record within tolerance" gate.
    # ``protection_source`` selects the target universe: gold-only
    # (legacy) or silver-augmented (plan_revision.md C9, gold wins for
    # fusion val/test entities; silver fills the rest of the pool).
    # ``surviving_record_ids`` enables the C13 intact-cluster rule when
    # silver is active: silver targets only apply to clusters whose
    # entire original member set survived K2.
    closeness_ctx = _ClosenessContext(
        domain=domain,
        sources=sources,
        attr_mapping=attr_mapping,
        id_columns=id_columns,
        tolerance_overrides=config.get("fusion_protection_tolerance"),
        protection_source=protection_source,
        surviving_record_ids=surviving_record_ids,
    )

    # Build reverse index for entity groups.
    id_to_group: dict[str, str] = {}
    if entity_groups:
        id_to_group = _build_id_to_entity_group(entity_groups)

    # Easy-level: apply cleanup rules first.
    if level == "easy":
        sources = _apply_cleanup_rules(sources, config, id_columns, prov)

    # Track noised cells for floor checks.
    noised_primaries: dict[str, set[str]] = {src: set() for src in sources}
    noised_cells: set[tuple[str, str, str]] = set()  # (rid, source, attr)

    # Soft global primary cap tracking.
    primary_cap = config.get("soft_global_primary_cap_hard", 0.35)
    total_entities_with_noised_primary = 0
    total_linked_entities = len(entity_groups) if entity_groups else 0

    noised: dict[str, pd.DataFrame] = {}

    for source_name in sorted(sources.keys()):
        df = sources[source_name]
        new_df = df.copy()
        new_df.attrs = df.attrs.copy()

        source_attrs = attr_classes.get(source_name, {})
        id_col = id_columns.get(source_name)
        sub_rng = spawn_sub_rng(rng, f"source_{source_name}")

        for col in sorted(source_attrs.keys()):
            if col not in df.columns:
                logger.warning(
                    "Column %r not in source %r -- skipping",
                    col,
                    source_name,
                )
                continue

            attr_class = source_attrs[col]
            target_rate = noise_rates.get(attr_class, 0.0)

            if target_rate <= 0:
                continue

            # Cast to object dtype for string assignment.
            if new_df[col].dtype != object:
                new_df[col] = new_df[col].astype(object)

            col_rng = spawn_sub_rng(sub_rng, f"col_{col}")

            for idx in range(len(df)):
                cell_value = df.iloc[idx][col]
                entity_id = (
                    str(df.iloc[idx][id_col])
                    if id_col and id_col in df.columns
                    else str(idx)
                )

                # Skip non-scalar cells (e.g. list-valued columns like
                # ``dbpedia.founders``). K6 operates on text-scale
                # single-value cells; array-like values are not a
                # meaningful noise target and would also crash
                # ``pd.isna``.
                if isinstance(cell_value, (list, tuple, set, dict)):
                    continue

                # Skip null/empty cells.
                if pd.isna(cell_value):
                    continue
                cell_str = str(cell_value).strip()
                if not cell_str or cell_str.lower() in ("null", "nan", "none"):
                    continue

                # Draw whether to noise this cell. R10-A: level-independent
                # per-cell selection so the noised-cell set nests across
                # levels (easy subset of medium subset of hard). Option B --
                # the operator drawn below stays on the level-keyed
                # ``col_rng`` so each level keeps its own operator mix.
                if (
                    cell_selection_uniform(
                        domain, source_name, entity_id, col, knob=6, master_seed=seed
                    )
                    >= target_rate
                ):
                    continue

                # Cell-collision check: skip if touched by prior knob
                # (except K4-fabricated cells).
                if collision_index is not None:
                    if collision_index.is_touched(entity_id, source_name, col):
                        if not collision_index.is_k4_fabricated(
                            entity_id, source_name, col
                        ):
                            skipped.append(
                                entity_id=entity_id,
                                source=source_name,
                                attribute=col,
                                original_value=cell_str,
                                reason="cell_collision_with_prior_knob",
                            )
                            continue

                # Entity group for floor checks.
                group_id = id_to_group.get(entity_id)

                # Clean-primary floor check.
                if attr_class == "primary" and group_id and entity_groups:
                    if not _check_clean_primary_floor(
                        group_id, entity_groups, noised_primaries
                    ):
                        skipped.append(
                            entity_id=entity_id,
                            source=source_name,
                            attribute=col,
                            original_value=cell_str,
                            reason="clean_primary_floor",
                        )
                        continue

                    # Soft global primary cap at hard.
                    if level == "hard" and total_linked_entities > 0:
                        current_frac = (
                            total_entities_with_noised_primary / total_linked_entities
                        )
                        if current_frac >= primary_cap:
                            skipped.append(
                                entity_id=entity_id,
                                source=source_name,
                                attribute=col,
                                original_value=cell_str,
                                reason="soft_global_primary_cap",
                            )
                            continue

                # Resolve per-cell taxonomy + numeric typing once.
                cell_taxonomy = taxonomies_by_col.get((source_name, col))
                cell_numeric_type = numeric_attrs.get((source_name, col))

                # Draw an operator. Retry up to _MAX_JITTER_RETRIES times
                # if a numeric attribute's mutation exceeds the per-cell
                # ±jitter_cap relative cap (Pending #6); each retry draws a
                # fresh operator so we don't loop on a single deterministic
                # over-shoot.
                attempt_log: list[str] = []
                accepted: tuple[str, str, dict[str, Any]] | None = None
                for _attempt in range(_MAX_JITTER_RETRIES):
                    op_name = str(col_rng.choice(op_names, p=op_probs))
                    # taxonomy_walk is only meaningful when the cell column
                    # is bound to a taxonomy — otherwise we'd waste retries
                    # on a guaranteed no-op.
                    if op_name == "taxonomy_walk" and cell_taxonomy is None:
                        attempt_log.append(f"{op_name}=no_taxonomy")
                        continue

                    result = _apply_operator(
                        op_name,
                        cell_str,
                        col_rng,
                        config,
                        level,
                        taxonomy=cell_taxonomy,
                    )
                    if result is None:
                        attempt_log.append(f"{op_name}=no_effect")
                        continue

                    candidate_value, candidate_params = result

                    # ±jitter_cap relative cap for numeric attributes.
                    if cell_numeric_type is not None:
                        if not numeric_jitter_within_cap(
                            cell_str,
                            candidate_value,
                            cell_numeric_type,
                            max_relative=jitter_cap,
                        ):
                            attempt_log.append(f"{op_name}=jitter_exceeds_cap")
                            continue

                    # Closeness contract (Pending #5): for fusion-protected
                    # cells, ≥1 record across the entity's sources must
                    # remain within tolerance of the fusion target value
                    # post-mutation. For non-protected entities the gate
                    # is vacuously True.
                    if group_id and entity_groups:
                        sources_in_progress = dict(sources)
                        sources_in_progress.update(noised)
                        sources_in_progress[source_name] = new_df
                        if not _check_close_survivor_floor(
                            entity_id=entity_id,
                            group_members=entity_groups[group_id],
                            current_source=source_name,
                            current_col=col,
                            candidate_value=candidate_value,
                            sources_in_progress=sources_in_progress,
                            ctx=closeness_ctx,
                        ):
                            attempt_log.append(f"{op_name}=closeness_violation")
                            continue

                    accepted = (op_name, candidate_value, candidate_params)
                    break

                if accepted is None:
                    # All retries exhausted. Reason reflects the final state.
                    if any(a.endswith("=closeness_violation") for a in attempt_log):
                        reason = "closeness_floor_exhausted_retries"
                    elif cell_numeric_type is not None and any(
                        a.endswith("=jitter_exceeds_cap") for a in attempt_log
                    ):
                        reason = "numeric_jitter_exhausted_retries"
                    else:
                        # Most common case: every drawn operator no-op'd
                        # for this cell (e.g. short value, no confusable
                        # chars, no taxonomy match).
                        last_op = (
                            attempt_log[-1].split("=", 1)[0] if attempt_log else "none"
                        )
                        reason = f"operator_{last_op}_no_effect"
                    skipped.append(
                        entity_id=entity_id,
                        source=source_name,
                        attribute=col,
                        original_value=cell_str,
                        reason=reason,
                    )
                    continue

                op_name, new_value, params = accepted

                # Write the noised value.
                new_df.iat[idx, df.columns.get_loc(col)] = new_value

                # Record provenance.
                prov.append(
                    entity_id=entity_id,
                    source=source_name,
                    attribute=col,
                    original_value=cell_str,
                    new_value=new_value,
                    transform_fn=op_name,
                    transform_params=params,
                )

                # Update floor tracking.
                noised_cells.add((entity_id, source_name, col))
                if attr_class == "primary":
                    noised_primaries[source_name].add(entity_id)
                    if group_id:
                        total_entities_with_noised_primary += 1

        noised[source_name] = new_df

    # Build output DataFrames.
    if len(prov) > 0:
        provenance_df = pd.DataFrame(
            [row.as_dict() for row in prov._rows],
            columns=PROVENANCE_COLUMNS,
        )
    else:
        provenance_df = pd.DataFrame(columns=PROVENANCE_COLUMNS)

    skipped_df = skipped.to_dataframe()

    return noised, provenance_df, skipped_df


# ---- Output writing -------------------------------------------------------


def write_outputs(
    provenance_df: pd.DataFrame,
    skipped_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Write provenance and skipped-cell artifacts to *output_dir*."""
    prov_dir = output_dir / "output" / "provenance"
    prov_dir.mkdir(parents=True, exist_ok=True)

    provenance_df.to_csv(prov_dir / "knob_06_noise.csv", index=False)
    logger.info(
        "Wrote provenance (%d rows) to %s",
        len(provenance_df),
        prov_dir / "knob_06_noise.csv",
    )

    skipped_df.to_csv(prov_dir / "knob_06_skipped.csv", index=False)
    logger.info(
        "Wrote skipped audit (%d rows) to %s",
        len(skipped_df),
        prov_dir / "knob_06_skipped.csv",
    )


# ---- CLI ------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Apply Knob 06 -- Value Noise Injection",
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

    logger.info("Knob 06: domain=%s level=%s output=%s", domain, level, output_dir)

    config = load_knob_06_config(domain)
    sources = load_domain_sources(domain)
    id_columns: dict[str, str] = config.get("id_columns", {})

    # Build entity linkage for floor checks.
    entity_groups = build_entity_linkage(domain, id_columns, sources)

    # Build collision index from prior provenance.
    prov_dir = output_dir / "output" / "provenance"
    collision_index = CollisionIndex(prov_dir)

    noised, provenance_df, skipped_df = apply_knob_06(
        domain=domain,
        level=level,
        sources=sources,
        config=config,
        entity_groups=entity_groups,
        collision_index=collision_index,
        seed=args.seed,
    )

    write_outputs(provenance_df, skipped_df, output_dir)

    for src_name in sorted(noised.keys()):
        logger.info("  %s: %d rows", src_name, len(noised[src_name]))
    logger.info("Provenance: %d rows", len(provenance_df))
    logger.info("Skipped: %d rows", len(skipped_df))


if __name__ == "__main__":
    main()
