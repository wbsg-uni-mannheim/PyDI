#!/usr/bin/env python3
"""Apply Knob 01 — Surface Augmentation Intensity.

Value paraphrase — abbreviations, synonymy, token reordering,
reformulation. Distinguishes from Knob 06 (noise/errors) in that these
are legitimate variants (both forms are correct). Tier C hybrid:

- **Easy**: deterministic normalize-to-canonical via
  ``baseline_above_target_rules``.
- **Medium**: deterministic table-driven abbreviation plus EDA
  ``random_swap`` / ``random_delete`` operators.
- **Hard**: medium set ∪ cached LLM paraphrase (with contamination
  guardrails and committee validation).

See ``knobs/knob_01_surface_augmentation.md`` for the full specification.

Usage
-----
::

    python usecases_synthetic/scripts/apply_knob_01_surface.py \\
        --domain companies --level medium

Inputs
------
- Source DataFrames from ``usecases/<domain>/input/data/``
- Per-domain config at ``usecases_synthetic/config/knob_01_surface/<domain>.yaml``
- Shared tables under ``usecases_synthetic/config/knob_01_surface/_tables/``
- Prompt templates under ``usecases_synthetic/config/knob_01_surface/_prompts/``
- (Hard only) LLM cache at
  ``usecases_synthetic/cache/knob_01_paraphrases/<domain>/<level>/``

Outputs (under *output_dir*)
----------------------------
- Paraphrased source DataFrames (returned in-memory)
- Provenance CSV at ``<output_dir>/output/provenance/knob_01_surface.csv``
- Skipped-cell audit at ``<output_dir>/output/provenance/knob_01_skipped.csv``
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Literal

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
    resolve_cache_domain,
)
from usecases_synthetic.lib.llm_cache import LLMCache, LLMCacheMiss
from usecases_synthetic.lib.loaders import load_domain_sources, read_em_gold_csv
from usecases_synthetic.lib.fusion_silver_targets import (
    PROTECTION_SOURCES,
    resolve_protection_sources,
)
from usecases_synthetic.lib.niche_metrics import _levenshtein_ratio
from usecases_synthetic.lib.protection import (
    ToleranceSpec,
    cell_has_close_survivor,
    fusion_cell_tolerance,
    is_close_enough,
)
from usecases_synthetic.lib.provenance import PROVENANCE_COLUMNS, ProvenanceLog
from usecases_synthetic.lib.rng import (
    cell_selection_uniform,
    make_rng,
    spawn_sub_rng,
)
from usecases_synthetic.lib.surface_operators import (
    abbreviate,
    build_first_token_index,
    eda_random_delete,
    eda_random_swap,
    llm_paraphrase,
    normalize_to_canonical,
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


# ---- Realised paraphrase audit --------------------------------------------

REALISED_COLUMNS = [
    "level",
    "paraphrase_attempts",
    "paraphrase_committed",
    "mean_edit_distance",
    "mean_token_jaccard_drop",
    "strict_cache_miss_count",
    "llm_unchanged_count",
    "llm_near_identity_count",
]


def _token_set(value: str) -> set[str]:
    return {t for t in str(value).lower().split() if t}


def _token_jaccard_drop(original: str, new: str) -> float:
    """Return ``1 - token_jaccard(original, new)`` in ``[0, 1]``.

    Tokens are whitespace-split, lowercased, deduplicated. Returns 0.0
    when both token sets are empty (no signal) and 1.0 when one is empty
    and the other is not. Used as the K1 shallow-paraphrase intensity
    proxy: token reorderings (random_swap) leave the set unchanged so
    drop = 0; token deletions / abbreviations / LLM rewrites reduce the
    overlap and drop > 0.
    """
    ta = _token_set(original)
    tb = _token_set(new)
    if not ta and not tb:
        return 0.0
    if not ta or not tb:
        return 1.0
    inter = len(ta & tb)
    union = len(ta | tb)
    if union == 0:
        return 0.0
    return 1.0 - inter / union


def build_realised_df(
    *,
    level: str,
    provenance_df: pd.DataFrame,
    skipped_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build the per-level K1 realised paraphrase audit summary.

    Columns
    -------
    level : str
    paraphrase_attempts : int
        ``len(provenance_df) + len(skipped_df)`` — cells K1 either
        mutated or post-gate rejected (collision, contamination,
        strict-cache miss, closeness floor, …). Rate-gate-silent skips
        are NOT counted because they are pre-attempt by construction.
    paraphrase_committed : int
        ``len(provenance_df)`` — every cell K1 mutated, including the
        easy normalize-down path.
    mean_edit_distance : float
        Mean of ``1 - levenshtein_ratio(original, new)`` over committed
        cells. 0.0 when no commits.
    mean_token_jaccard_drop : float
        Mean of ``1 - token_jaccard(original, new)`` over committed
        cells. 0.0 when no commits.
    strict_cache_miss_count : int
        Count of ``reason == 'strict_cache_miss'`` rows in
        ``skipped_df``. Surfaces K1 cache dormancy (the same cache-miss
        failure mode that landed K2 hard at 0 interpolations on the
        last music/games run; see plan_revision.md G9).
    llm_unchanged_count : int
        Count of ``reason == 'llm_unchanged_sentinel'`` rows -- cells
        where the LLM returned the v2 ``<UNCHANGED>`` sentinel (R10-D).
        Calibration signal: rising rate at hard signals the prompt is
        too permissive about declaring inputs unparaphrasable.
    llm_near_identity_count : int
        Count of ``reason == 'llm_near_identity'`` rows -- cells where
        the LLM output passed ``paraphrase != value`` but shared the
        input's lowercased token set (casing / punctuation /
        whitespace only). Surfaces shallow-paraphrase laziness that
        the v2 post-filter intercepts (R10-D).
    """
    committed = int(len(provenance_df))
    skipped_count = int(len(skipped_df))

    if committed > 0:
        originals = provenance_df["original_value"].astype(str).tolist()
        new_values = provenance_df["new_value"].astype(str).tolist()
        edit_distances = [
            1.0 - _levenshtein_ratio(o, n) for o, n in zip(originals, new_values)
        ]
        jaccard_drops = [
            _token_jaccard_drop(o, n) for o, n in zip(originals, new_values)
        ]
        mean_edit = float(np.mean(edit_distances))
        mean_jacc = float(np.mean(jaccard_drops))
    else:
        mean_edit = 0.0
        mean_jacc = 0.0

    if skipped_count > 0 and "reason" in skipped_df.columns:
        strict_misses = int((skipped_df["reason"] == "strict_cache_miss").sum())
        unchanged_count = int((skipped_df["reason"] == "llm_unchanged_sentinel").sum())
        near_identity_count = int((skipped_df["reason"] == "llm_near_identity").sum())
    else:
        strict_misses = 0
        unchanged_count = 0
        near_identity_count = 0

    return pd.DataFrame(
        [
            {
                "level": level,
                "paraphrase_attempts": committed + skipped_count,
                "paraphrase_committed": committed,
                "mean_edit_distance": round(mean_edit, 6),
                "mean_token_jaccard_drop": round(mean_jacc, 6),
                "strict_cache_miss_count": strict_misses,
                "llm_unchanged_count": unchanged_count,
                "llm_near_identity_count": near_identity_count,
            }
        ],
        columns=REALISED_COLUMNS,
    )


class SkippedLog:
    """Accumulates skipped-cell audit rows for Knob 01."""

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


def load_knob_01_config(domain: str) -> dict[str, Any]:
    """Load the Knob 01 surface augmentation config for *domain*."""
    path = CONFIG_DIR / "knob_01_surface" / f"{domain}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Knob 01 config not found: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_shared_stopwords() -> set[str]:
    """Load the shared EDA stopword list."""
    path = CONFIG_DIR / "knob_01_surface" / "_tables" / "stopwords_en.yaml"
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return {str(s).lower() for s in raw.get("stopwords", [])}


def _load_eda_params() -> dict[str, Any]:
    """Load shared EDA parameters."""
    path = CONFIG_DIR / "knob_01_surface" / "_tables" / "eda_params.yaml"
    if not path.exists():
        return {
            "min_tokens_for_eda": 2,
            "max_operators_per_cell": {"easy": 1, "medium": 1, "hard": 2},
        }
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_contamination_params() -> dict[str, Any]:
    """Load shared contamination-check parameters."""
    path = CONFIG_DIR / "knob_01_surface" / "_tables" / "contamination.yaml"
    if not path.exists():
        return {"ngram_overlap_threshold": 8, "first_token_probe_length": 3}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_prompt_template(name: str) -> str:
    """Load a prompt template by filename."""
    path = CONFIG_DIR / "knob_01_surface" / "_prompts" / name
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


# ---- Entity linkage (reused from K6 pattern) ------------------------------


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

    Returns a mapping ``group_id -> [(source_name, record_id), ...]``.
    Singletons (entities without cross-source links) get their own
    group. Mirrors the helper used in Knob 06.
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


# ---- Operator dispatch ----------------------------------------------------


def _normalise_weights(
    operator_mix: dict[str, float],
) -> tuple[list[str], list[float]]:
    """Normalise operator weights to a probability vector."""
    ops = sorted(operator_mix.keys())
    weights = [float(operator_mix[o]) for o in ops]
    total = sum(weights)
    probs = [w / total for w in weights]
    return ops, probs


def _apply_deterministic_operator(
    operator_name: str,
    value: str,
    rng: np.random.Generator,
    config: dict[str, Any],
    stopwords: set[str],
    key_tokens: set[str],
) -> tuple[str, dict[str, Any]] | None:
    """Dispatch a single easy/medium-level operator on a cell value."""
    if operator_name == "abbreviate":
        return abbreviate(value, config.get("abbreviation_table", {}) or {}, rng=rng)
    elif operator_name == "eda_random_swap":
        return eda_random_swap(
            value, rng, stopwords=stopwords, key_tokens=key_tokens, n_swaps=1
        )
    elif operator_name == "eda_random_delete":
        return eda_random_delete(value, rng, stopwords=stopwords, key_tokens=key_tokens)
    else:
        logger.warning("Unknown deterministic operator: %s", operator_name)
        return None


# ---- Easy path: normalize-down -------------------------------------------


def _apply_baseline_above_target_rules(
    sources: dict[str, pd.DataFrame],
    config: dict[str, Any],
    id_columns: dict[str, str],
    id_to_group: dict[str, str],
    entity_groups: dict[str, list[tuple[str, str]]],
    prov: ProvenanceLog,
    closeness_ctx: "_ClosenessContext | None" = None,
) -> dict[str, pd.DataFrame]:
    """Easy path: replace (source, attribute) cells with a canonical sibling form.

    K1 follow-up #1 (2026-05-07): when a fusion target value is authored
    for the (entity, canonical_attribute) cell, filter siblings to those
    within tolerance of the target before picking. Prevents normalize-
    down from inheriting an obviously-wrong sibling value (companies easy
    pass had ``forbes.region: China → Taiwan`` because the dbpedia
    sibling's ``nation`` was wrong for that entity).
    """
    rules = config.get("baseline_above_target_rules") or []
    if not rules:
        return sources

    attr_mapping: dict[str, dict[str, str]] = config.get("attribute_mapping", {})

    result = {name: df.copy() for name, df in sources.items()}
    for name in result:
        result[name].attrs = sources[name].attrs.copy()

    for rule in rules:
        src_name = rule["source"]
        attr = rule["attribute"]
        canonical_src = rule["canonical_from"]
        canonical_attr = rule["canonical_attribute"]
        strategy = rule.get("strategy", "shortest")

        if src_name not in result:
            continue
        df = result[src_name]
        if attr not in df.columns:
            continue
        if canonical_src not in result:
            continue

        if df[attr].dtype != object:
            df[attr] = df[attr].astype(object)

        canonical_df = result[canonical_src]
        if canonical_attr not in canonical_df.columns:
            continue

        id_col = id_columns.get(src_name)
        canonical_id_col = id_columns.get(canonical_src)
        if not id_col or not canonical_id_col:
            continue

        # Resolve the canonical fusion attribute for closeness lookups.
        canonical_fusion_attr = attr_mapping.get(canonical_src, {}).get(canonical_attr)
        tolerance: ToleranceSpec | None = None
        if closeness_ctx is not None and canonical_fusion_attr is not None:
            tolerance = closeness_ctx.tolerance_for(canonical_fusion_attr)

        canonical_lookup: dict[str, str] = {}
        for _, row in canonical_df.iterrows():
            rid = str(row[canonical_id_col])
            val = row[canonical_attr]
            if pd.notna(val):
                canonical_lookup[rid] = str(val)

        for idx in range(len(df)):
            entity_id = str(df.iloc[idx][id_col])
            current_value = df.iloc[idx][attr]
            if pd.isna(current_value):
                continue
            current_str = str(current_value)

            group_id = id_to_group.get(entity_id)
            if group_id is None:
                continue
            siblings: list[str] = []
            for member_src, member_rid in entity_groups.get(group_id, []):
                if member_src == canonical_src:
                    if member_rid in canonical_lookup:
                        siblings.append(canonical_lookup[member_rid])

            # K1 follow-up #1: gate normalize-down on closeness when a
            # fusion target is authored for this (entity, canonical_attr).
            if (
                closeness_ctx is not None
                and tolerance is not None
                and canonical_fusion_attr is not None
            ):
                # Find a fusion-protected anchor entity-id within the group.
                protected_eid: str | None = None
                for _msrc, mrid in entity_groups.get(group_id, []):
                    if mrid in closeness_ctx.protected_ids:
                        protected_eid = mrid
                        break
                if protected_eid is not None:
                    targets = closeness_ctx.target_for(
                        protected_eid, canonical_fusion_attr
                    )
                    if targets:
                        filtered = [
                            s
                            for s in siblings
                            if any(is_close_enough(s, t, tolerance) for t in targets)
                        ]
                        if not filtered:
                            # No close sibling — leave the cell alone
                            # rather than normalize-down to a wrong value.
                            continue
                        siblings = filtered

            result_tuple = normalize_to_canonical(
                current_str, siblings, strategy=strategy
            )
            if result_tuple is None:
                continue

            new_value, params = result_tuple
            df.iat[idx, df.columns.get_loc(attr)] = new_value
            prov.append(
                entity_id=entity_id,
                source=src_name,
                attribute=attr,
                original_value=current_str,
                new_value=new_value,
                transform_fn="normalize_to_canonical",
                transform_params=params,
            )

    return result


# ---- Floor checks ---------------------------------------------------------


def _check_clean_primary_floor(
    entity_group_id: str,
    entity_groups: dict[str, list[tuple[str, str]]],
    paraphrased_primaries: dict[str, set[str]],
) -> bool:
    """Anchor-survivor floor: >=1 source keeps clean primary per entity.

    Singletons (entities with no cross-source matches) have no anchor to
    preserve — the floor is vacuously True so paraphrase is allowed.
    K1 follow-up #2 from the 2026-05-07 sign-off (was returning False
    pre-Pending #5, which over-fired on ~233/1397/540 cells in the
    medium smoke).
    """
    members = entity_groups.get(entity_group_id, [])
    if len(members) <= 1:
        return True
    clean_count = sum(
        1 for src, rid in members if rid not in paraphrased_primaries.get(src, set())
    )
    return clean_count >= 2


class _ClosenessContext:
    """Per-call resources for the closeness contract (Pending #5).

    ``protection_source`` selects between gold-only protection (the
    original behavior — fusion val/test entities only) and
    silver-augmented protection (gold wins per-(member, attribute);
    silver fills the rest, expanding the protected universe to every
    pool-cluster member). See
    :func:`fusion_silver_targets.resolve_protection_sources` for the
    merge rule.
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

        rev: dict[str, list[tuple[str, str]]] = {}
        for src, mapping in attr_mapping.items():
            for src_col, canonical in (mapping or {}).items():
                rev.setdefault(canonical, []).append((src, src_col))
        self.canonical_to_source_cols = rev

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

    Returns True iff committing *candidate_value* still leaves ≥1
    record across the entity's sources within tolerance of a fusion
    target value. Vacuously True for non-fusion-protected entities or
    cells without a target value.
    """
    canonical_attr = ctx.attr_mapping.get(current_source, {}).get(current_col)
    if canonical_attr is None:
        return True

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
                surviving.append(candidate_value)
            else:
                val = df.iat[row_idx, df.columns.get_loc(sc_col)]
                if pd.isna(val):
                    continue
                surviving.append(str(val))
            break

    return cell_has_close_survivor(target_values, surviving, tolerance)


# ---- Core dispatcher ------------------------------------------------------


LLMClient = Callable[[str, str], str]
CommitteeFn = Callable[[str, str, str, str], bool]


def apply_knob_01(
    domain: str,
    level: Literal["easy", "medium", "hard"],
    sources: dict[str, pd.DataFrame],
    config: dict[str, Any],
    entity_groups: dict[str, list[tuple[str, str]]] | None = None,
    collision_index: CollisionIndex | None = None,
    llm_cache: LLMCache | None = None,
    llm_client: LLMClient | None = None,
    committee_fn: CommitteeFn | None = None,
    strict_cache: bool = False,
    seed: int = 42,
    protection_source: str = "gold",
    surviving_record_ids: set[str] | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply Knob 01 surface augmentation to source DataFrames.

    Parameters
    ----------
    domain : str
        Domain name.
    level : {"easy", "medium", "hard"}
        Difficulty level.
    sources : dict[str, DataFrame]
        Source DataFrames keyed by source name.
    config : dict
        Parsed Knob 01 YAML.
    entity_groups : dict or None
        Entity linkage groups (from EM gold). If None, floor checks are
        skipped.
    collision_index : CollisionIndex or None
        Cell-collision index from prior joint-phase knobs.
    llm_cache : LLMCache or None
        Shared LLM cache. Required at hard level.
    llm_client : callable or None
        Callable ``(prompt_template, value) -> paraphrase_str``. Optional
        at hard level when the cache is pre-populated; required
        otherwise.
    committee_fn : callable or None
        Committee validator for LLM paraphrases. Defaults to accept-all.
    strict_cache : bool, default False
        When True at hard level, a cache miss raises ``LLMCacheMiss``.
    seed : int, default 42
        Master RNG seed.

    Returns
    -------
    paraphrased_sources : dict[str, DataFrame]
        DataFrames with paraphrase applied. ``attrs`` preserved.
    provenance_df : DataFrame
        Provenance log.
    skipped_df : DataFrame
        Skipped-cell audit.
    realised_df : DataFrame
        Per-level K1 audit summary (one row). Columns per
        :data:`REALISED_COLUMNS`. Powers ``knob_01_realised_*`` audit
        rows in ``monotonicity_report.csv``
        (plan_revision.md R-1 / G9 step 4f).
    """
    if level not in VALID_LEVELS:
        raise ValueError(f"Invalid level: {level!r}. Valid: {VALID_LEVELS}")

    rng = make_rng(domain, level, knob=1, master_seed=seed)
    prov = ProvenanceLog(knob=1, level=level)
    skipped = SkippedLog(knob=1, level=level)

    id_columns: dict[str, str] = config.get("id_columns", {})
    attr_classes: dict[str, dict[str, str]] = config.get("attribute_classes", {})
    attr_mapping: dict[str, dict[str, str]] = config.get("attribute_mapping", {})

    rate_per_class: dict[str, dict[str, float]] = {
        "primary": config.get("paraphrase_rate_primary", {}),
        "key": config.get("paraphrase_rate_key", {}),
        "secondary": config.get("paraphrase_rate_secondary", {}),
        "categorical": config.get("paraphrase_rate_categorical", {}),
    }
    operator_mix: dict[str, float] = config.get("operator_mix", {}).get(level, {})
    anchor_floor_cfg: dict[str, bool] = config.get("anchor_survivor_floor", {}) or {}

    # Resolve stopwords: shared list ∪ per-domain extras.
    stopwords = _load_shared_stopwords()
    stopwords.update(str(s).lower() for s in config.get("stopword_list", []) or [])

    # Key-token skiplist per (source, column). Fallback to empty set.
    key_token_skiplist: dict[str, dict[str, set[str]]] = {}
    raw_skiplist = config.get("key_token_skiplist", {}) or {}
    for src_name, cols in raw_skiplist.items():
        key_token_skiplist[src_name] = {
            col: set(tokens or []) for col, tokens in (cols or {}).items()
        }

    # EDA params / min tokens guard.
    eda_params = _load_eda_params()
    min_tokens_for_eda = eda_params.get("min_tokens_for_eda", 2)

    id_to_group: dict[str, str] = {}
    if entity_groups:
        id_to_group = _build_id_to_entity_group(entity_groups)

    # Closeness contract (Pending #5): replaces the strict per-cell
    # clean-survivor floor with an "≥1 record within tolerance" gate.
    # ``protection_source`` selects the target universe: gold-only
    # (legacy) or silver-augmented (plan_revision.md C9, gold wins for
    # fusion val/test entities; silver fills the rest of the pool).
    # ``surviving_record_ids`` enables the C13 intact-cluster rule when
    # silver is active: silver targets only apply to clusters whose
    # entire original member set survived K2. Caller derives it from
    # post-K2 sources.
    closeness_ctx = _ClosenessContext(
        domain=domain,
        sources=sources,
        attr_mapping=attr_mapping,
        id_columns=id_columns,
        tolerance_overrides=config.get("fusion_protection_tolerance"),
        protection_source=protection_source,
        surviving_record_ids=surviving_record_ids,
    )

    # Easy path: normalize-to-canonical pass first (gated on closeness
    # to fix K1 follow-up #1 — siblings must be within tolerance of the
    # fusion target before being eligible).
    if level == "easy":
        sources = _apply_baseline_above_target_rules(
            sources,
            config,
            id_columns,
            id_to_group,
            entity_groups or {},
            prov,
            closeness_ctx=closeness_ctx,
        )

    # Build first-token index for contamination probe (hard level only).
    first_token_index: dict[tuple[str, ...], str] = {}
    if level == "hard":
        records: list[tuple[str, str]] = []
        for src_name, df in sources.items():
            src_attr_classes = attr_classes.get(src_name, {})
            primary_cols = [
                c for c, cls in src_attr_classes.items() if cls == "primary"
            ]
            id_col = id_columns.get(src_name)
            for col in primary_cols:
                if col not in df.columns or not id_col or id_col not in df.columns:
                    continue
                for _, row in df.iterrows():
                    val = row[col]
                    if pd.isna(val):
                        continue
                    records.append((str(row[id_col]), str(val)))
        first_token_index = build_first_token_index(
            records,
            n_tokens=_load_contamination_params().get("first_token_probe_length", 3),
        )

    # Track paraphrased cells for floor checks.
    paraphrased_primaries: dict[str, set[str]] = {src: set() for src in sources}
    paraphrased_cells: set[tuple[str, str, str]] = set()

    # Load prompt templates (hard level only). Resolution is
    # version-aware: ``llm_prompt_version`` selects the template suffix
    # (R10-D: v2 adds a secondary-specific template + <UNCHANGED>
    # escape + minimum-divergence rule). v1 has no secondary template,
    # so secondary attributes fall back to the short template under v1.
    prompt_version = str(config.get("llm_prompt_version", "v1"))
    prompt_short = _load_prompt_template(f"prompt_short_{prompt_version}.txt")
    prompt_categorical = _load_prompt_template(
        f"prompt_categorical_{prompt_version}.txt"
    )
    prompt_secondary = _load_prompt_template(f"prompt_secondary_{prompt_version}.txt")
    if not prompt_secondary:
        # v1 has no secondary template -- fall back to short for
        # back-compat. v2 ships a real secondary template (R10-D).
        prompt_secondary = prompt_short
    ngram_n = _load_contamination_params().get("ngram_overlap_threshold", 8)

    paraphrased: dict[str, pd.DataFrame] = {}

    op_names, op_probs = _normalise_weights(operator_mix) if operator_mix else ([], [])

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
                    "Column %r not in source %r -- skipping", col, source_name
                )
                continue

            attr_class = source_attrs[col]
            target_rate = rate_per_class.get(attr_class, {}).get(level, 0.0)

            if target_rate <= 0 or not op_names:
                continue

            if new_df[col].dtype != object:
                new_df[col] = new_df[col].astype(object)

            col_rng = spawn_sub_rng(sub_rng, f"col_{col}")

            col_key_tokens = key_token_skiplist.get(source_name, {}).get(col, set())

            for idx in range(len(df)):
                raw_cell = df.iloc[idx][col]
                entity_id = (
                    str(df.iloc[idx][id_col])
                    if id_col and id_col in df.columns
                    else str(idx)
                )

                if pd.isna(raw_cell):
                    continue
                cell_str = str(raw_cell).strip()
                if not cell_str or cell_str.lower() in ("null", "nan", "none"):
                    continue

                # R10-A: level-independent per-cell selection so the
                # perturbed-cell set nests across levels (easy subset of
                # medium subset of hard). Option B -- the operator/value
                # drawn below still uses the level-keyed ``col_rng`` so
                # each level keeps its own per-level operator mix.
                if (
                    cell_selection_uniform(
                        domain, source_name, entity_id, col, knob=1, master_seed=seed
                    )
                    >= target_rate
                ):
                    continue

                # Cell-collision check. Unconditional skip (K1 is the
                # first in the joint phase; downstream knobs read K1's
                # provenance).
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
                        # K4-fabricated cells: K1 does not re-paraphrase.
                        skipped.append(
                            entity_id=entity_id,
                            source=source_name,
                            attribute=col,
                            original_value=cell_str,
                            reason="cell_collision_with_k4_fabricated",
                        )
                        continue

                group_id = id_to_group.get(entity_id)

                # Anchor-survivor floor for primary attrs.
                if (
                    attr_class == "primary"
                    and anchor_floor_cfg.get("primary", True)
                    and group_id
                    and entity_groups
                ):
                    if not _check_clean_primary_floor(
                        group_id, entity_groups, paraphrased_primaries
                    ):
                        skipped.append(
                            entity_id=entity_id,
                            source=source_name,
                            attribute=col,
                            original_value=cell_str,
                            reason="anchor_survivor_floor",
                        )
                        continue

                # (per-cell strict floor superseded by Pending #5
                # closeness gate; check happens post-mutation below)

                # Draw an operator.
                op_name = str(col_rng.choice(op_names, p=op_probs))

                # EDA operators require a minimum token count.
                if op_name in ("eda_random_swap", "eda_random_delete"):
                    if len(cell_str.split()) < int(min_tokens_for_eda):
                        skipped.append(
                            entity_id=entity_id,
                            source=source_name,
                            attribute=col,
                            original_value=cell_str,
                            reason="too_short_for_eda",
                        )
                        continue

                result: tuple[str, dict[str, Any]] | None = None
                transform_fn = op_name
                params: dict[str, Any] = {}

                if op_name == "llm_paraphrase":
                    if llm_cache is None:
                        skipped.append(
                            entity_id=entity_id,
                            source=source_name,
                            attribute=col,
                            original_value=cell_str,
                            reason="llm_cache_missing",
                        )
                        continue
                    # Per-class prompt dispatch. v2 distinguishes
                    # secondary (descriptions) from primary/key (short
                    # entity labels). Under v1 prompt_secondary aliases
                    # prompt_short for back-compat.
                    if attr_class == "categorical":
                        prompt = prompt_categorical
                    elif attr_class == "secondary":
                        prompt = prompt_secondary
                    else:
                        prompt = prompt_short
                    try:
                        llm_result = llm_paraphrase(
                            cell_str,
                            source=source_name,
                            attribute=col,
                            attribute_class=attr_class,  # type: ignore[arg-type]
                            cache=llm_cache,
                            prompt_template=prompt,
                            api_client=llm_client,
                            entity_key=entity_id,
                            first_token_index=first_token_index,
                            committee_fn=committee_fn,
                            strict_cache=strict_cache,
                            ngram_n=int(ngram_n),
                        )
                    except LLMCacheMiss:
                        skipped.append(
                            entity_id=entity_id,
                            source=source_name,
                            attribute=col,
                            original_value=cell_str,
                            reason="strict_cache_miss",
                        )
                        continue
                    if llm_result is None:
                        # Contamination or committee failure; fall back
                        # to a medium-level deterministic operator.
                        skipped.append(
                            entity_id=entity_id,
                            source=source_name,
                            attribute=col,
                            original_value=cell_str,
                            reason="llm_contamination_or_committee_fail",
                        )
                        continue
                    new_value, params = llm_result
                    transform_fn = params.pop("transform_fn")
                    # R10-D: <UNCHANGED> sentinel means the LLM judged
                    # the cell unparaphrasable. Skip the cell (no value
                    # change) but record it in provenance under a
                    # dedicated reason so we can count rates.
                    if transform_fn == "llm_paraphrase_unchanged":
                        skipped.append(
                            entity_id=entity_id,
                            source=source_name,
                            attribute=col,
                            original_value=cell_str,
                            reason="llm_unchanged_sentinel",
                        )
                        continue
                    # R10-D: post-filter rejected the LLM output as
                    # near-identity (zero substantive token change).
                    # Same handling as contamination -- skip + log.
                    if transform_fn == "llm_paraphrase_near_identity":
                        skipped.append(
                            entity_id=entity_id,
                            source=source_name,
                            attribute=col,
                            original_value=cell_str,
                            reason="llm_near_identity",
                        )
                        continue
                else:
                    det_result = _apply_deterministic_operator(
                        op_name,
                        cell_str,
                        col_rng,
                        config,
                        stopwords=stopwords,
                        key_tokens=col_key_tokens,
                    )
                    if det_result is None:
                        skipped.append(
                            entity_id=entity_id,
                            source=source_name,
                            attribute=col,
                            original_value=cell_str,
                            reason=f"operator_{op_name}_no_effect",
                        )
                        continue
                    new_value, params = det_result

                if not isinstance(new_value, str):
                    continue

                # Closeness contract (Pending #5): for fusion-protected
                # cells, ≥1 record across the entity's sources must
                # remain within tolerance of the fusion target value
                # post-mutation. K1's operators are deterministic per
                # cell so we don't retry — we just skip when the
                # candidate would orphan the contract.
                if (
                    anchor_floor_cfg.get(attr_class, False)
                    and group_id
                    and entity_groups
                ):
                    sources_in_progress = dict(sources)
                    sources_in_progress.update(paraphrased)
                    sources_in_progress[source_name] = new_df
                    if not _check_close_survivor_floor(
                        entity_id=entity_id,
                        group_members=entity_groups[group_id],
                        current_source=source_name,
                        current_col=col,
                        candidate_value=new_value,
                        sources_in_progress=sources_in_progress,
                        ctx=closeness_ctx,
                    ):
                        skipped.append(
                            entity_id=entity_id,
                            source=source_name,
                            attribute=col,
                            original_value=cell_str,
                            reason="closeness_floor_violation",
                        )
                        continue

                new_df.iat[idx, df.columns.get_loc(col)] = new_value
                prov.append(
                    entity_id=entity_id,
                    source=source_name,
                    attribute=col,
                    original_value=cell_str,
                    new_value=new_value,
                    transform_fn=transform_fn,
                    transform_params=params,
                )

                paraphrased_cells.add((entity_id, source_name, col))
                if attr_class == "primary":
                    paraphrased_primaries[source_name].add(entity_id)

        paraphrased[source_name] = new_df

    # Build output DataFrames.
    if len(prov) > 0:
        provenance_df = pd.DataFrame(
            [row.as_dict() for row in prov._rows],
            columns=PROVENANCE_COLUMNS,
        )
    else:
        provenance_df = pd.DataFrame(columns=PROVENANCE_COLUMNS)

    skipped_df = skipped.to_dataframe()
    realised_df = build_realised_df(
        level=level,
        provenance_df=provenance_df,
        skipped_df=skipped_df,
    )
    return paraphrased, provenance_df, skipped_df, realised_df


# ---- Output writing -------------------------------------------------------


def write_outputs(
    provenance_df: pd.DataFrame,
    skipped_df: pd.DataFrame,
    output_dir: Path,
    realised_df: pd.DataFrame | None = None,
) -> None:
    """Write provenance, skipped-cell, and realised-summary artifacts.

    The realised summary lands at
    ``<output_dir>/output/baselines/knob_01_realised.csv`` when
    *realised_df* is supplied (plan_revision.md R-1 / G9 step 4f —
    powers the K1 monotonicity audit rows).
    """
    prov_dir = output_dir / "output" / "provenance"
    prov_dir.mkdir(parents=True, exist_ok=True)

    provenance_df.to_csv(prov_dir / "knob_01_surface.csv", index=False)
    logger.info(
        "Wrote provenance (%d rows) to %s",
        len(provenance_df),
        prov_dir / "knob_01_surface.csv",
    )

    skipped_df.to_csv(prov_dir / "knob_01_skipped.csv", index=False)
    logger.info(
        "Wrote skipped audit (%d rows) to %s",
        len(skipped_df),
        prov_dir / "knob_01_skipped.csv",
    )

    if realised_df is not None and not realised_df.empty:
        baselines_dir = output_dir / "output" / "baselines"
        baselines_dir.mkdir(parents=True, exist_ok=True)
        realised_path = baselines_dir / "knob_01_realised.csv"
        realised_df.to_csv(realised_path, index=False)
        logger.info("Wrote realised paraphrase summary to %s", realised_path)


# ---- CLI ------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Apply Knob 01 -- Surface Augmentation Intensity",
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
    parser.add_argument(
        "--strict-cache",
        action="store_true",
        help=(
            "Hard level only: raise on LLM cache miss instead of "
            "invoking the API client."
        ),
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

    logger.info("Knob 01: domain=%s level=%s output=%s", domain, level, output_dir)

    config = load_knob_01_config(domain)
    sources = load_domain_sources(domain)
    id_columns: dict[str, str] = config.get("id_columns", {})

    entity_groups = build_entity_linkage(domain, id_columns, sources)

    prov_dir = output_dir / "output" / "provenance"
    collision_index = CollisionIndex(prov_dir)

    llm_cache: LLMCache | None = None
    if level == "hard":
        cache_dir = (
            REPO_ROOT
            / "usecases_synthetic"
            / "cache"
            / "knob_01_paraphrases"
            / resolve_cache_domain(domain)
            / level
        )
        llm_cache = LLMCache(
            cache_dir=cache_dir,
            prompt_version=config.get("llm_prompt_version", "v1"),
            model_id=config.get("llm_model_id", "claude-opus-4-6"),
        )

    # Strict cache is opt-in via --strict-cache. On miss we want a live
    # LLM call (never fail-on-miss), so the CLI must be given an
    # ``llm_client`` by the caller when running with strict_cache=False
    # at hard level. The previous auto-forcing at hard was the root cause
    # of K2 dial-dormancy (plan_revision.md §C1 / Step 2 findings).
    strict_cache = args.strict_cache

    paraphrased, provenance_df, skipped_df, realised_df = apply_knob_01(
        domain=domain,
        level=level,  # type: ignore[arg-type]
        sources=sources,
        config=config,
        entity_groups=entity_groups,
        collision_index=collision_index,
        llm_cache=llm_cache,
        llm_client=None,  # CLI path: strict cache only.
        committee_fn=None,
        strict_cache=strict_cache,
        seed=args.seed,
    )

    write_outputs(provenance_df, skipped_df, output_dir, realised_df=realised_df)

    for src_name in sorted(paraphrased.keys()):
        logger.info("  %s: %d rows", src_name, len(paraphrased[src_name]))
    logger.info("Provenance: %d rows", len(provenance_df))
    logger.info("Skipped: %d rows", len(skipped_df))
    if not realised_df.empty:
        row = realised_df.iloc[0]
        logger.info(
            "Realised K1: attempts=%d committed=%d mean_edit=%.4f "
            "mean_jaccard_drop=%.4f strict_cache_miss=%d "
            "llm_unchanged=%d llm_near_identity=%d",
            int(row["paraphrase_attempts"]),
            int(row["paraphrase_committed"]),
            float(row["mean_edit_distance"]),
            float(row["mean_token_jaccard_drop"]),
            int(row["strict_cache_miss_count"]),
            int(row["llm_unchanged_count"]),
            int(row["llm_near_identity_count"]),
        )


if __name__ == "__main__":
    main()
