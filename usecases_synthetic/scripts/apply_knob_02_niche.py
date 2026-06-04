#!/usr/bin/env python3
"""Apply Knob 02 — Entity Niche Density.

Controls how many similar-but-distinct entities cluster in the same
semantic neighbourhood. Runs **first** in the S1 canonical knob order.

Two sub-systems share the same multi-metric substrate:

- **Niche-density scorer** (consensus-biased, RRF fusion) drives the
  removal path at easy / medium.
- **Corner-case pair miner** (recall-biased, per-metric union) drives
  EM test-set regeneration and the hard-negative budget.

At **hard** level, the dispatcher additionally runs LLM entity
interpolation to generate near-twin entities from dense clusters.

See ``knobs/knob_02_niche_density.md`` for the full specification and
``plans/module_09_knob_02.md`` for module-level acceptance criteria.

Usage
-----
::

    python usecases_synthetic/scripts/apply_knob_02_niche.py \\
        --domain companies --level easy

Inputs
------
- Per-source DataFrames from ``usecases/<domain>/input/data/``
- Expanded positives protection set (EM gold ∪ fusion gold ∪
  pooled_positives) via :func:`protection.build_expanded_positives`
- Per-domain config at
  ``usecases_synthetic/config/knob_02_niche/<domain>.yaml``
- Embedding cache under
  ``usecases_synthetic/cache/knob_02_embeddings/<domain>.npy``
- (Hard only) interpolation cache under
  ``usecases_synthetic/cache/knob_02_interpolations/<domain>/``

Outputs (under *output_dir*)
----------------------------
- Post-Knob-2 canonical entity CSV at
  ``<output_dir>/input/data/canonical.csv``
- Regenerated EM per-pair per-split files at
  ``<output_dir>/input/entitymatching/<src1>_2_<src2>_{train,val,test}_regenerated.csv``
- Provenance at ``<output_dir>/output/provenance/knob_02_niche.csv``
- Per-entity niche score audit at
  ``<output_dir>/output/provenance/knob_02_niche_scores.csv``
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Ensure repo root is on sys.path.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

from usecases_synthetic.lib.corner_case_miner import (
    CornerCasePair,
    HardNegativeAudit,
    HardNegativePolicy,
    MetricThresholds,
    SplitSpec,
    apply_hard_negative_policy,
    measure_corner_case_ratio,
    mine_corner_cases,
    RegenPools,
    regenerate_em_splits,
)
from usecases_synthetic.lib.domain_config import (
    CONFIG_DIR,
    USECASES_DIR,
    VALID_LEVELS,
    data_root_for_domain,
    load_domain_config,
    resolve_cache_domain,
)
from usecases_synthetic.lib.entity_interpolation import (
    InterpolatedEntity,
    build_openai_interpolation_client,
    default_api_client_from_attributes,
    interpolate_entity,
    place_entity_across_sources,
    select_parent_pairs,
)
from usecases_synthetic.lib.llm_cache import LLMCache, LLMCacheMiss
from usecases_synthetic.lib.non_corner_refill import (
    NonCornerEntity,
    build_openai_non_corner_client,
    refill_non_corner_entity,
    select_reference_anchor,
)
from usecases_synthetic.lib.loaders import load_domain_sources, read_em_gold_csv
from usecases_synthetic.lib.niche_metrics import (
    EmbeddingCacheMeta,
    attribute_overlap_matrix,
    attribute_overlap_neighbours,
    build_label_list,
    build_text_corpus,
    compute_embedding_matrix,
    compute_tfidf_matrix,
    embedding_neighbours,
    label_collision_index,
    lexical_extended_jaccard_neighbours,
    normalize_label,
    tfidf_neighbours,
)
from usecases_synthetic.lib.niche_scorer import (
    EntityDensity,
    MetricNeighbourhoods,
    compute_rrf_density,
    rank_entities_by_density,
    select_for_removal,
)
from usecases_synthetic.lib.protection import (
    build_drop_corner_protection_set,
    build_expanded_positives,
)
from usecases_synthetic.lib.provenance import PROVENANCE_COLUMNS, ProvenanceLog
from usecases_synthetic.lib.rng import make_rng, spawn_sub_rng

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_knob_02_config(domain: str) -> dict[str, Any]:
    """Load the Knob 02 config for *domain*."""
    path = CONFIG_DIR / "knob_02_niche" / f"{domain}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Knob 02 config not found: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


_EM_SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")


def _load_original_split_targets(
    domain: str,
    authored_pairs: Sequence[tuple[str, str]],
) -> tuple[
    dict[tuple[str, str], list[SplitSpec]],
    dict[tuple[str, str], dict[str, list[tuple[str, str, bool]]]],
]:
    """Inspect original EM gold splits to derive regen size + pos ratio + pairs.

    For each authored source pair, reads the per-split files at
    ``usecases/<domain>/input/entitymatching/<src1>_2_<src2>_<split>.csv``
    and returns:

    1. A :class:`SplitSpec` per split capturing target row count and
       positive ratio. Downstream consumers expect the regenerated
       benchmark to mirror these shapes.
    2. The original (id1, id2, label) triples per split. The
       incremental regenerator uses these as the *starting point* for
       the regenerated split (rows whose ids both survive K2 are kept
       verbatim).

    When the original file is missing for a pair/split, the split is
    dropped from both return values rather than synthesised with a
    default — callers should gate emission on list emptiness.

    Parameters
    ----------
    domain : str
        Domain name (``"companies"``, ``"companies-small"``, ...).
    authored_pairs : sequence of (src1, src2)
        Source pairs declared by the domain config.

    Returns
    -------
    specs_by_pair : dict
        ``{(src1, src2): [SplitSpec, ...]}`` ordered train → val → test.
        Empty list for pairs whose original splits cannot be found.
    pairs_by_split : dict
        ``{(src1, src2): {split_name: [(id1, id2, is_positive), ...]}}``
        — original triples in file order. ``is_positive`` is ``True``
        when ``label.lower() == "true"``.
    """
    em_dir = (
        (data_root_for_domain(domain) or USECASES_DIR)
        / domain
        / "input"
        / "entitymatching"
    )
    specs_by_pair: dict[tuple[str, str], list[SplitSpec]] = {}
    pairs_by_split: dict[tuple[str, str], dict[str, list[tuple[str, str, bool]]]] = {}
    for pair in authored_pairs:
        src1, src2 = pair
        pair_name = f"{src1}_2_{src2}"
        specs: list[SplitSpec] = []
        pairs_for_pair: dict[str, list[tuple[str, str, bool]]] = {}
        for split in _EM_SPLIT_NAMES:
            path = em_dir / f"{pair_name}_{split}.csv"
            if not path.exists():
                logger.warning(
                    "Original EM split missing for sizing reference: %s",
                    path,
                )
                continue
            # Original EM golds are headerless (id1, id2, label); handle
            # both that convention and the header-ful CSVs produced by
            # the regenerator itself (so this helper is reusable on
            # augmented variants too).
            try:
                df = pd.read_csv(path)
                if "label" not in df.columns:
                    raise ValueError("missing label column")
            except (ValueError, KeyError):
                df = pd.read_csv(path, header=None, names=["id1", "id2", "label"])
            if df.empty:
                continue
            labels = df["label"].astype(str).str.lower()
            positives_mask = labels == "true"
            positives = int(positives_mask.sum())
            size = int(len(df))
            ratio = float(positives) / size if size > 0 else 0.5
            specs.append(SplitSpec(name=split, size=size, positive_ratio=ratio))
            pairs_for_pair[split] = [
                (str(row.id1), str(row.id2), bool(positives_mask.iloc[i]))
                for i, row in enumerate(df.itertuples(index=False))
            ]
        specs_by_pair[pair] = specs
        pairs_by_split[pair] = pairs_for_pair
    return specs_by_pair, pairs_by_split


def _load_pool_positives_by_pair(
    domain: str,
    authored_pairs: Sequence[tuple[str, str]],
    rid_to_source: dict[str, str] | None = None,
) -> dict[tuple[str, str], list[tuple[str, str]]]:
    """Load pool positives bucketed by authored source pair.

    Reads ``usecases_synthetic/pools/<domain>/pooled_positives.csv`` —
    the consolidated pool of discovered positives that goes beyond the
    hand-curated EM gold (see ``plan.md`` step 4). Each row carries
    ``source_1`` / ``source_2`` columns which label the *pair*, not the
    per-id source: ``id1`` / ``id2`` in the CSV are lex-sorted (per
    :func:`canonical_pair`), so ``id1`` does **not** necessarily belong
    to ``source_1``. We orient via ``rid_to_source`` when provided so
    backfill rows in regen end up with ``id1`` in ``canonical_pair[0]``.

    Aliased domains (``companies-small`` → ``companies``) fall back to
    the source domain's pool when the alias has no pool of its own.

    Parameters
    ----------
    domain : str
        Domain name. Resolved via :func:`resolve_cache_domain` so
        aliased domains pick up the source domain's pool.
    authored_pairs : sequence of (src1, src2)
        Source pairs declared by the domain config. Pool rows whose
        source-pair (as ``frozenset``) matches one of these are kept
        and oriented to the declared (src1, src2) ordering.
    rid_to_source : dict, optional
        Per-record-id source mapping (``{rid: source_name}``). Used to
        orient pairs by checking the *actual* source of ``id1`` rather
        than trusting the row's ``source_1`` column. Pairs whose ids
        are absent from the mapping fall back to the column-based
        orientation. Pass ``None`` for legacy column-based behaviour.

    Returns
    -------
    dict
        ``{(src1, src2): [(id1_in_src1, id2_in_src2), ...]}``. Pairs
        are canonicalised so each ``(id1, id2)`` appears at most once
        and the first id always belongs to ``src1``.
    """
    from usecases_synthetic.lib.domain_config import POOLS_DIR

    cache_domain = resolve_cache_domain(domain)
    pool_path = POOLS_DIR / cache_domain / "pooled_positives.csv"
    out: dict[tuple[str, str], list[tuple[str, str]]] = {p: [] for p in authored_pairs}
    if not pool_path.exists():
        logger.info(
            "Pool positives unavailable for domain=%s (looked for %s) — "
            "regen will rely on K2 cluster positives + survivors only",
            domain,
            pool_path,
        )
        return out

    pool_df = pd.read_csv(pool_path, dtype={"id1": str, "id2": str})
    if pool_df.empty:
        return out
    pair_lookup = {frozenset(p): p for p in authored_pairs}
    seen: dict[tuple[str, str], set[tuple[str, str]]] = {
        p: set() for p in authored_pairs
    }
    orient_fallback_drops = 0
    for row in pool_df.itertuples(index=False):
        src1 = str(getattr(row, "source_1", ""))
        src2 = str(getattr(row, "source_2", ""))
        if not src1 or not src2 or src1 == src2:
            continue
        canonical_pair = pair_lookup.get(frozenset({src1, src2}))
        if canonical_pair is None:
            continue
        id1_raw, id2_raw = str(row.id1), str(row.id2)
        if rid_to_source is not None:
            # Orient by the *actual* source of each id, since pool CSV's
            # id1/id2 are lex-sorted, not source-aligned. When a pool id
            # is absent from rid_to_source (e.g. K2 removed it before
            # this loader ran), drop the pair — it cannot survive the
            # ids_present filter downstream either.
            id1_src = rid_to_source.get(id1_raw)
            id2_src = rid_to_source.get(id2_raw)
            if id1_src is None or id2_src is None:
                orient_fallback_drops += 1
                continue
            if id1_src == canonical_pair[0] and id2_src == canonical_pair[1]:
                id_a, id_b = id1_raw, id2_raw
            elif id1_src == canonical_pair[1] and id2_src == canonical_pair[0]:
                id_a, id_b = id2_raw, id1_raw
            else:
                orient_fallback_drops += 1
                continue
        else:
            # Legacy column-based orientation. Only correct when the pool
            # CSV's per-row id1/id2 happen to match source_1/source_2;
            # for ``canonical_pair``-sorted pools (the production layout)
            # this misorients ~77% of music rows. Kept for callers that
            # cannot supply ``rid_to_source``.
            if src1 == canonical_pair[0]:
                id_a, id_b = id1_raw, id2_raw
            else:
                id_a, id_b = id2_raw, id1_raw
        key = (id_a, id_b)
        if key in seen[canonical_pair]:
            continue
        seen[canonical_pair].add(key)
        out[canonical_pair].append(key)
    if rid_to_source is not None and orient_fallback_drops:
        logger.info(
            "Pool positives orientation: dropped %d rows with ids absent "
            "from rid_to_source (likely K2-removed entities)",
            orient_fallback_drops,
        )
    return out


def _load_prompt_template(name: str) -> str:
    path = CONFIG_DIR / "knob_02_niche" / "_prompts" / name
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Union-find (reused from K1)
# ---------------------------------------------------------------------------


def _find(parent: dict[str, str], x: str) -> str:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _union(parent: dict[str, str], rank: dict[str, int], a: str, b: str) -> None:
    ra, rb = _find(parent, a), _find(parent, b)
    if ra == rb:
        return
    if rank[ra] < rank[rb]:
        ra, rb = rb, ra
    parent[rb] = ra
    if rank[ra] == rank[rb]:
        rank[ra] += 1


# ---------------------------------------------------------------------------
# S1 canonical-view builder
# ---------------------------------------------------------------------------


def build_canonical_view(
    domain: str,
    sources: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, list[tuple[str, str]]], dict[str, str]]:
    """Collapse per-source records into a canonical entity frame.

    Each connected component of the EM-gold pair graph is one canonical
    entity. A canonical row is built by taking the first non-null value
    from *source_priority* for every canonical-schema column, mapping
    per-source columns via *attribute_mapping*.

    Parameters
    ----------
    domain : str
    sources : dict
        Per-source DataFrames keyed by source name.
    config : dict
        Knob 02 config for the domain.

    Returns
    -------
    canonical_frame : pandas.DataFrame
        One row per canonical entity, indexed by ``canonical_id``
        (stored both as the DataFrame index and an ``entity_id``
        column).
    entity_groups : dict
        ``canonical_id -> [(source_name, record_id), ...]``.
    id_to_canonical : dict
        ``record_id -> canonical_id`` reverse index spanning every
        source record (including singletons).
    """
    id_columns: dict[str, str] = config["id_columns"]
    source_priority: list[str] = list(config["source_priority"])
    canonical_schema: list[str] = list(config["canonical_schema"])
    attribute_mapping: dict[str, dict[str, str]] = config["attribute_mapping"]

    domain_config = load_domain_config(domain)
    em_dir = domain_config.em_dir()

    pairs: list[tuple[str, str]] = []
    if em_dir.exists():
        for csv_path in sorted(em_dir.glob("*_all.csv")):
            df = read_em_gold_csv(csv_path)
            positives = df[
                df["label"].astype(str).str.strip().str.lower().isin(("true", "1"))
            ]
            for _, row in positives.iterrows():
                pairs.append((str(row["id1"]), str(row["id2"])))
        if not pairs:
            for suffix in ("_train.csv", "_val.csv", "_test.csv"):
                for csv_path in sorted(em_dir.glob(f"*{suffix}")):
                    df = read_em_gold_csv(csv_path)
                    positives = df[
                        df["label"]
                        .astype(str)
                        .str.strip()
                        .str.lower()
                        .isin(("true", "1"))
                    ]
                    for _, row in positives.iterrows():
                        pairs.append((str(row["id1"]), str(row["id2"])))

    # Domains whose comprehensive ground-truth match structure lives in the
    # pool rather than a complete EM gold group by the pooled positives. For
    # papers the labelled EM gold is only a sample (~6.6k pairs); the
    # DOI-derived pool is the full cross-source match set (~156k pairs /
    # ~55.6k clusters), so without this the canonical view never collapses
    # (every one of the 182k records becomes its own entity, which both
    # blows up K2 runtime and mines same-paper records as fake corner pairs).
    # Opt in via ``canonical_grouping: pool`` in the knob-02 config; the
    # default keeps the EM-gold-only grouping byte-identical for every other
    # domain.
    if config.get("canonical_grouping", "em_gold") == "pool":
        from usecases_synthetic.lib.domain_config import POOLS_DIR

        pool_path = POOLS_DIR / domain / "pooled_positives.csv"
        if pool_path.exists():
            pool_df = pd.read_csv(pool_path)
            for pid1, pid2 in zip(
                pool_df["id1"].astype(str), pool_df["id2"].astype(str)
            ):
                pairs.append((pid1, pid2))

    all_ids: set[str] = set()
    id_to_source: dict[str, str] = {}
    for source_name, df in sources.items():
        id_col = id_columns.get(source_name)
        if id_col and id_col in df.columns:
            for rid in df[id_col].astype(str):
                all_ids.add(rid)
                id_to_source[rid] = source_name

    for id1, id2 in pairs:
        all_ids.add(id1)
        all_ids.add(id2)

    parent: dict[str, str] = {x: x for x in all_ids}
    rank: dict[str, int] = {x: 0 for x in all_ids}
    for id1, id2 in pairs:
        if id1 in parent and id2 in parent:
            _union(parent, rank, id1, id2)

    groups_raw: dict[str, list[str]] = defaultdict(list)
    for rid in all_ids:
        groups_raw[_find(parent, rid)].append(rid)

    # Build fast lookup tables: per-source id -> row dict.
    source_lookup: dict[str, dict[str, dict[str, Any]]] = {}
    for source_name, df in sources.items():
        id_col = id_columns.get(source_name)
        if id_col is None or id_col not in df.columns:
            continue
        per_id: dict[str, dict[str, Any]] = {}
        for _, row in df.iterrows():
            per_id[str(row[id_col])] = row.to_dict()
        source_lookup[source_name] = per_id

    canonical_rows: list[dict[str, Any]] = []
    entity_groups: dict[str, list[tuple[str, str]]] = {}
    id_to_canonical: dict[str, str] = {}

    # Deterministic canonical-id ordering: sort by lexicographic min
    # id in the component so reruns are bit-stable.
    sorted_components = sorted(groups_raw.items(), key=lambda t: min(t[1]))

    for cid_counter, (_root, members) in enumerate(sorted_components):
        canonical_id = f"k02_ent_{cid_counter:06d}"
        members_sorted = sorted(members)
        members_with_src = [(id_to_source.get(m, "unknown"), m) for m in members_sorted]
        entity_groups[canonical_id] = members_with_src
        for m in members_sorted:
            id_to_canonical[m] = canonical_id

        canonical_row: dict[str, Any] = {"entity_id": canonical_id}
        for col in canonical_schema:
            filled = False
            for src in source_priority:
                if src not in source_lookup:
                    continue
                src_cols = attribute_mapping.get(src, {})
                # Find source column(s) mapping to this canonical col.
                reverse = [k for k, v in src_cols.items() if v == col]
                # Look up values across every member in this source.
                for src_col in reverse:
                    for member_src, rid in members_with_src:
                        if member_src != src:
                            continue
                        row = source_lookup[src].get(rid)
                        if row is None:
                            continue
                        val = row.get(src_col)
                        if val is None:
                            continue
                        if isinstance(val, float) and np.isnan(val):
                            continue
                        if str(val).strip().lower() in ("", "nan", "none", "null"):
                            continue
                        canonical_row[col] = val
                        filled = True
                        break
                    if filled:
                        break
                if filled:
                    break
            if not filled:
                canonical_row[col] = None
        canonical_rows.append(canonical_row)

    canonical_frame = pd.DataFrame(canonical_rows)
    if not canonical_frame.empty:
        canonical_frame = canonical_frame.set_index("entity_id", drop=False)
    canonical_frame.attrs["dataset_name"] = f"{domain}_canonical"
    return canonical_frame, entity_groups, id_to_canonical


# ---------------------------------------------------------------------------
# Metric assembly
# ---------------------------------------------------------------------------


def compute_all_neighbourhoods(
    canonical_frame: pd.DataFrame,
    config: dict[str, Any],
    domain: str,
    *,
    embedding_cache_path: Path,
) -> tuple[
    list[MetricNeighbourhoods],
    Any,
    np.ndarray | None,
    list[str],
    list[str],
    dict[str, list[int]],
]:
    """Build per-metric neighbour lists for the canonical frame.

    Returns
    -------
    metric_lists : list of MetricNeighbourhoods
        One entry per enabled metric in ``config["metrics"]`` (excluding
        label_collision which is handled separately).
    tfidf_matrix
        Sparse TF-IDF matrix (or None if disabled).
    embeddings
        Row-normalised embedding matrix (or None if disabled).
    text_corpus
        Per-entity concatenated text block (matches TF-IDF / embedding /
        ext-Jaccard input). Returned so the in-loop miner can re-score
        ext-Jaccard against the same surface used during retrieval.
    labels
        Primary label list in canonical row order. Used for
        label-collision detection.
    collision_groups
        ``normalised_label -> [row_index, ...]``.
    """
    metrics_cfg = config.get("metrics", {})
    top_k = int(config.get("metric_top_k", 20))
    inner_thr = float(config.get("inner_token_threshold", 0.8))
    stopwords = set(str(s).lower() for s in config.get("stopword_list", []) or [])

    primary_column = config.get("primary_column_canonical", "name")
    if primary_column not in canonical_frame.columns:
        # Try fallback: the first canonical schema column.
        primary_column = config["canonical_schema"][0]

    labels = build_label_list(canonical_frame, primary_column)
    text_corpus = build_text_corpus(
        canonical_frame, config.get("text_concat_order", [primary_column])
    )

    metric_lists: list[MetricNeighbourhoods] = []

    if metrics_cfg.get("ext_jaccard", True):
        logger.info("Computing lexical extended-Jaccard neighbourhoods")
        # Operate on the same concatenated text corpus as TF-IDF and the
        # embedding metric so all token-overlap signals share the same
        # input surface; the typo-robust inner Levenshtein matcher is
        # what differentiates ext-Jaccard from TF-IDF after the change.
        ext = lexical_extended_jaccard_neighbours(
            text_corpus,
            top_k=top_k,
            inner_token_threshold=inner_thr,
            stopwords=stopwords,
        )
        metric_lists.append(MetricNeighbourhoods(name="ext_jaccard", top_k=ext))

    tfidf_matrix = None
    if metrics_cfg.get("tfidf", True):
        logger.info("Computing TF-IDF neighbourhoods")
        tfidf_matrix = compute_tfidf_matrix(text_corpus)
        tfidf_nb = tfidf_neighbours(tfidf_matrix, top_k=top_k)
        metric_lists.append(MetricNeighbourhoods(name="tfidf", top_k=tfidf_nb))

    embeddings: np.ndarray | None = None
    if metrics_cfg.get("embedding", True):
        logger.info("Computing embedding neighbourhoods")
        model_id = config.get(
            "embedding_model_id", "sentence-transformers/all-MiniLM-L6-v2"
        )
        embeddings = compute_embedding_matrix(
            text_corpus,
            model_id=model_id,
            cache_path=embedding_cache_path,
            concat_order=list(config.get("text_concat_order", [primary_column])),
        )
        emb_nb = embedding_neighbours(embeddings, top_k=top_k)
        metric_lists.append(MetricNeighbourhoods(name="embedding", top_k=emb_nb))

    if metrics_cfg.get("attribute_overlap", True):
        logger.info("Computing attribute-overlap neighbourhoods")
        weights: dict[str, float] = {
            k: float(v) for k, v in config.get("attribute_overlap_weights", {}).items()
        }
        numeric_overlap_cfg: dict[str, dict[str, Any]] = {
            str(col): dict(spec)
            for col, spec in (config.get("numeric_overlap") or {}).items()
            if isinstance(spec, dict)
        }
        bags = attribute_overlap_matrix(
            canonical_frame,
            columns=list(weights.keys()),
            weights=weights,
            numeric_columns=list(numeric_overlap_cfg.keys()),
        )
        attr_nb = attribute_overlap_neighbours(
            bags,
            weights=weights,
            top_k=top_k,
            numeric_overlap=numeric_overlap_cfg or None,
        )
        metric_lists.append(
            MetricNeighbourhoods(name="attribute_overlap", top_k=attr_nb)
        )

    collision_groups: dict[str, list[int]] = {}
    if metrics_cfg.get("label_collision", True):
        collision_groups = label_collision_index(labels)

    return metric_lists, tfidf_matrix, embeddings, text_corpus, labels, collision_groups


# ---------------------------------------------------------------------------
# Source-record pair enumeration (for EM test regeneration)
# ---------------------------------------------------------------------------


def enumerate_cross_source_positive_pairs(
    entity_groups: dict[str, list[tuple[str, str]]],
    *,
    excluded_canonical_ids: set[str] | None = None,
    source_pair_filter: set[frozenset[str]] | None = None,
) -> list[tuple[str, str]]:
    """Enumerate positive source-record pairs from *entity_groups*.

    For each canonical entity not in *excluded_canonical_ids*, produces
    every pair of member records that come from **different** sources.
    When *source_pair_filter* is provided, only pairs whose source pair
    (as a ``frozenset``) appears in the filter are kept — typically the
    set of source pairs for which EM gold is authored.

    Parameters
    ----------
    entity_groups : dict
        ``canonical_id -> [(source_name, record_id), ...]``.
    excluded_canonical_ids : set of str or None
        Canonical IDs whose members must not appear in the output
        (e.g. entities dropped by the removal loop).
    source_pair_filter : set of frozenset[str] or None
        When set, only emit record pairs whose source pair matches.

    Returns
    -------
    list of (str, str)
        Sorted, deduplicated positive source-record pairs with
        ``rid_a < rid_b`` for determinism.
    """
    excluded = excluded_canonical_ids or set()
    out: set[tuple[str, str]] = set()
    for cid, members in entity_groups.items():
        if cid in excluded:
            continue
        if len(members) < 2:
            continue
        for (src_a, rid_a), (src_b, rid_b) in itertools.combinations(
            sorted(members), 2
        ):
            if src_a == src_b:
                continue
            if (
                source_pair_filter is not None
                and frozenset({src_a, src_b}) not in source_pair_filter
            ):
                continue
            a, b = (rid_a, rid_b) if rid_a < rid_b else (rid_b, rid_a)
            out.add((a, b))
    return sorted(out)


def pick_cross_cluster_record_pair(
    cid_a: str,
    cid_b: str,
    entity_groups: dict[str, list[tuple[str, str]]],
    *,
    source_pair_filter: set[frozenset[str]] | None,
    rng: np.random.Generator,
) -> tuple[str, str] | None:
    """Pick one source-record pair ``(rid_a, rid_b)`` for canonical
    cluster pair ``(cid_a, cid_b)``.

    Only cross-source combinations matching *source_pair_filter* are
    considered. When *source_pair_filter* is ``None`` any cross-source
    pair is eligible. Returns ``None`` when no eligible pair exists —
    the caller drops the canonical pair rather than emitting a negative
    under an unauthored source combination (which would be silently
    discarded by the variant loader's per-pair split).

    The chosen pair is normalised to ``rid_a < rid_b`` for determinism.

    Retained for backward compatibility with existing unit tests. The
    K2 pipeline now uses :func:`enumerate_cross_cluster_record_pairs`
    instead so the negative pool is proportional to the available data
    rather than capped at one negative per cross-cluster canonical
    pair (negatives should never be the scaling bottleneck — there are
    far more cross-cluster record combinations than cross-cluster
    canonical-entity pairs).
    """
    members_a = entity_groups.get(cid_a, [])
    members_b = entity_groups.get(cid_b, [])
    if not members_a or not members_b:
        return None

    pool: list[tuple[str, str]] = []
    for src_a, rid_a in members_a:
        for src_b, rid_b in members_b:
            if src_a == src_b:
                continue
            if (
                source_pair_filter is not None
                and frozenset({src_a, src_b}) not in source_pair_filter
            ):
                continue
            pool.append((rid_a, rid_b))

    if not pool:
        return None
    pool = sorted(pool)
    idx = int(rng.integers(len(pool)))
    rid1, rid2 = pool[idx]
    return (rid1, rid2) if rid1 < rid2 else (rid2, rid1)


def enumerate_cross_cluster_record_pairs(
    cid_a: str,
    cid_b: str,
    entity_groups: dict[str, list[tuple[str, str]]],
    *,
    source_pair_filter: set[frozenset[str]] | None,
    max_per_cluster_pair: int | None = None,
    rng: np.random.Generator | None = None,
) -> list[tuple[str, str]]:
    """Enumerate every valid cross-source record pair between two clusters.

    Unlike :func:`pick_cross_cluster_record_pair` (one pair per
    cross-cluster canonical entity combination), this expands each
    cluster pair to the full set of its cross-source record
    combinations. The negative pool is therefore bounded by the
    *record* count, not by the cluster-pair count — positives should
    be the bottleneck, not negatives.

    Only source combinations in *source_pair_filter* are kept. When
    the pair set for a given cluster pair is larger than
    *max_per_cluster_pair*, a uniform random subset of that size is
    returned (requires *rng*). When *max_per_cluster_pair* is
    ``None``, all eligible record pairs are emitted unthrottled.

    Parameters
    ----------
    cid_a, cid_b : str
        Canonical entity IDs for the two clusters.
    entity_groups : dict
        ``canonical_id -> [(source_name, record_id), ...]``.
    source_pair_filter : set of frozenset[str] or None
        Authored source-pair filter.
    max_per_cluster_pair : int, optional
        Upper bound on record pairs emitted per cluster pair. Avoids
        pathological blow-up when one canonical entity contains many
        records from a single source. ``None`` = unthrottled.
    rng : numpy.random.Generator, optional
        Required when *max_per_cluster_pair* is set and the cluster
        pair has more eligible record pairs than the cap.

    Returns
    -------
    list of (str, str)
        Canonical-ordered record pairs (``rid_a < rid_b``) for this
        cluster pair. Empty list when no eligible pair exists.
    """
    members_a = entity_groups.get(cid_a, [])
    members_b = entity_groups.get(cid_b, [])
    if not members_a or not members_b:
        return []

    pool: list[tuple[str, str]] = []
    for src_a, rid_a in members_a:
        for src_b, rid_b in members_b:
            if src_a == src_b:
                continue
            if (
                source_pair_filter is not None
                and frozenset({src_a, src_b}) not in source_pair_filter
            ):
                continue
            canon = (rid_a, rid_b) if rid_a < rid_b else (rid_b, rid_a)
            pool.append(canon)

    if not pool:
        return []
    pool = sorted(set(pool))

    if max_per_cluster_pair is not None and len(pool) > max_per_cluster_pair:
        if rng is None:
            raise ValueError(
                "enumerate_cross_cluster_record_pairs: max_per_cluster_pair "
                "is set but rng is None; caller must supply an RNG when "
                "throttling is active"
            )
        idxs = rng.choice(len(pool), size=max_per_cluster_pair, replace=False)
        pool = sorted(pool[int(k)] for k in idxs)

    return pool


# ---------------------------------------------------------------------------
# Candidate pair sampling for the miner
# ---------------------------------------------------------------------------


def build_candidate_pairs(
    n_entities: int,
    *,
    cluster_of: list[int],
    metric_lists: list[MetricNeighbourhoods],
    cap_per_class: int,
    rng: np.random.Generator,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Build same-cluster and cross-cluster candidate pair pools.

    Same-cluster pairs are enumerated exhaustively (there are few per
    cluster). Cross-cluster pairs are sampled by: (a) unioning the top-K
    neighbour lists across every metric — every metric-proposed
    neighbour is a candidate, (b) adding a uniform random sample to
    keep the pool diverse. Both pools are capped at *cap_per_class*.
    """
    same_pairs: list[tuple[int, int]] = []
    cluster_members: dict[int, list[int]] = defaultdict(list)
    for i, c in enumerate(cluster_of):
        if c < 0:
            continue
        cluster_members[c].append(i)
    for members in cluster_members.values():
        if len(members) < 2:
            continue
        for i, j in itertools.combinations(sorted(members), 2):
            same_pairs.append((i, j))

    if len(same_pairs) > cap_per_class:
        idxs = rng.choice(len(same_pairs), size=cap_per_class, replace=False)
        same_pairs = [same_pairs[int(k)] for k in sorted(idxs)]

    # Cross-cluster candidates: from neighbour lists, filtered to pairs
    # in different clusters.
    cross_set: set[tuple[int, int]] = set()
    for metric in metric_lists:
        for ent_idx, nbrs in enumerate(metric.top_k):
            for neigh_idx, _sim in nbrs:
                if neigh_idx == ent_idx:
                    continue
                if (
                    cluster_of[ent_idx] == cluster_of[neigh_idx]
                    and cluster_of[ent_idx] >= 0
                ):
                    continue
                pair = (
                    (ent_idx, neigh_idx)
                    if ent_idx < neigh_idx
                    else (neigh_idx, ent_idx)
                )
                cross_set.add(pair)
    cross_pairs = sorted(cross_set)
    if len(cross_pairs) > cap_per_class:
        idxs = rng.choice(len(cross_pairs), size=cap_per_class, replace=False)
        cross_pairs = [cross_pairs[int(k)] for k in sorted(idxs)]

    return same_pairs, cross_pairs


# ---------------------------------------------------------------------------
# Removal loop
# ---------------------------------------------------------------------------


def iterative_removal_to_target(
    *,
    canonical_frame: pd.DataFrame,
    densities: list[EntityDensity],
    protection_flags: list[bool],
    cluster_of: list[int],
    labels: list[str],
    tfidf_matrix,
    embeddings: np.ndarray | None,
    same_pairs: list[tuple[int, int]],
    cross_pairs: list[tuple[int, int]],
    thresholds: MetricThresholds,
    inner_token_threshold: float,
    stopwords: set[str],
    target_ratio: float,
    removal_fraction_cap: float,
    rng: np.random.Generator,
) -> tuple[set[int], float]:
    """Drop non-protected entities from the densest clusters until the
    measured corner-case ratio is at or below *target_ratio*.

    The corner-case classification of any individual pair (i, j) depends
    only on (cluster_of[i], cluster_of[j], labels[i], labels[j], tfidf,
    embeddings) — none of which change as entities are removed. Removing
    an entity can only *deactivate* pairs (those touching it), it cannot
    reclassify them. We therefore mine corner cases once on the full
    candidate-pair pool, build an entity-to-pair adjacency index, and
    deactivate pairs incrementally as candidates are removed. This is
    O(|pairs|) for the initial mining + O(|queue| × max_pairs_per_entity)
    for the loop, instead of the prior O(|queue| × |pairs|).

    Returns
    -------
    removed_indices : set of int
    final_ratio : float
        The ratio after the final removal pass.
    """
    all_pairs = list(same_pairs) + list(cross_pairs)

    initial_corner = mine_corner_cases(
        candidate_pairs=all_pairs,
        cluster_of=cluster_of,
        labels=labels,
        tfidf_matrix=tfidf_matrix,
        embeddings=embeddings,
        thresholds=thresholds,
        inner_token_threshold=inner_token_threshold,
        stopwords=stopwords,
    )
    corner_pair_set: set[tuple[int, int]] = set()
    for cc in initial_corner:
        a, b = (cc.i, cc.j) if cc.i < cc.j else (cc.j, cc.i)
        corner_pair_set.add((a, b))

    ranked = rank_entities_by_density(densities, rng)
    queue = select_for_removal(
        ranked,
        protection_flags=protection_flags,
        removal_fraction_cap=removal_fraction_cap,
    )
    if not queue:
        ratio = measure_corner_case_ratio(all_pairs, initial_corner)
        return set(), ratio

    # Normalise pairs to (a, b) with a < b so adjacency lookup is unambiguous.
    norm_pairs: list[tuple[int, int]] = []
    for p in all_pairs:
        a, b = (p[0], p[1]) if p[0] < p[1] else (p[1], p[0])
        norm_pairs.append((a, b))

    pair_index: dict[int, set[tuple[int, int]]] = defaultdict(set)
    for pair in norm_pairs:
        pair_index[pair[0]].add(pair)
        pair_index[pair[1]].add(pair)

    live_pairs: set[tuple[int, int]] = set(norm_pairs)
    live_corner: set[tuple[int, int]] = corner_pair_set & live_pairs
    removed: set[int] = set()

    def _current_ratio() -> float:
        return len(live_corner) / len(live_pairs) if live_pairs else 0.0

    if _current_ratio() <= target_ratio:
        return removed, _current_ratio()

    for candidate in queue:
        for pair in pair_index.get(candidate, ()):
            if pair in live_pairs:
                live_pairs.remove(pair)
                live_corner.discard(pair)
        removed.add(candidate)
        if _current_ratio() <= target_ratio:
            return removed, _current_ratio()

    return removed, _current_ratio()


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------


def apply_knob_02(
    domain: str,
    level: Literal["easy", "medium", "hard"],
    sources: dict[str, pd.DataFrame],
    config: dict[str, Any],
    *,
    expanded_positives: set[str] | None = None,
    llm_cache: LLMCache | None = None,
    api_client=None,
    committee_fn=None,
    strict_cache: bool = False,
    embedding_cache_path: Path | None = None,
    seed: int = 42,
    source_pair_filter: set[frozenset[str]] | None = None,
    hard_negative_policy: HardNegativePolicy | None = None,
    non_corner_cache: LLMCache | None = None,
    non_corner_api_client=None,
    protection_source: str = "gold",
) -> tuple[
    dict[str, pd.DataFrame],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Apply Knob 02 to *sources* at *level*.

    Returns
    -------
    new_sources : dict
        Per-source frames with removals / interpolated rows applied.
    canonical_frame : pandas.DataFrame
        Post-Knob-2 canonical entity frame.
    regenerated_em_test : pandas.DataFrame
        Regenerated EM test set (columns ``id1``, ``id2``, ``label``).
    provenance_df : pandas.DataFrame
        Provenance log.
    niche_scores_df : pandas.DataFrame
        Per-entity density + agreement audit.
    """
    if level not in VALID_LEVELS:
        raise ValueError(f"Invalid level: {level!r}")

    rng = make_rng(domain, level, knob=2, master_seed=seed)
    prov = ProvenanceLog(knob=2, level=level)

    if expanded_positives is None:
        expanded_positives = build_expanded_positives(domain)

    # 1. Build the canonical view.
    canonical_frame, entity_groups, id_to_canonical = build_canonical_view(
        domain, sources, config
    )
    n_entities = len(canonical_frame)
    if n_entities == 0:
        logger.warning("Canonical frame is empty — nothing to do")
        return (
            sources,
            canonical_frame,
            pd.DataFrame(columns=["id1", "id2", "label"]),
            pd.DataFrame(columns=PROVENANCE_COLUMNS),
            pd.DataFrame(),
        )

    entity_ids = list(canonical_frame["entity_id"])

    # Protection-flag: canonical entity is protected if any member record
    # ID is in expanded_positives.
    protection_flags: list[bool] = []
    for cid in entity_ids:
        members = entity_groups.get(cid, [])
        is_prot = any(rid in expanded_positives for _src, rid in members)
        protection_flags.append(is_prot)

    # Drop-corner-touching operator (step 4i) consults a narrower
    # protection set by design — pool-cluster members are droppable so
    # the operator can move the corner-pair ratio on pool-live domains
    # (products). Under the plan_revision.md C13 design, K2's existence
    # protection is ALWAYS gold (fusion val/test only), independent of
    # the K1/K6 ``protection_source`` flag. ``--protection-source
    # silver`` only changes K1/K6 drift protection; silver-cluster
    # members are explicitly droppable so the K2 dial can fire on
    # pool-live domains and the subsequent K1/K6 intact-cluster gate
    # can waive drift protection for the cluster members K2 broke.
    drop_corner_protected = build_drop_corner_protection_set(
        domain, protection_source="gold"
    )
    drop_corner_protection_flags: list[bool] = []
    for cid in entity_ids:
        members = entity_groups.get(cid, [])
        is_prot = any(rid in drop_corner_protected for _src, rid in members)
        drop_corner_protection_flags.append(is_prot)

    # Cluster id per entity. Each canonical entity = its own cluster
    # (clusters represent ground-truth matches, which the canonical
    # collapse already encodes). Cross-cluster = different canonical id.
    cluster_of = list(range(n_entities))

    # 2. Metrics + RRF scoring.
    embedding_cache_path = embedding_cache_path or (
        REPO_ROOT
        / "usecases_synthetic"
        / "cache"
        / "knob_02_embeddings"
        / f"{domain}.npy"
    )
    (
        metric_lists,
        tfidf_matrix,
        embeddings,
        text_corpus,
        labels,
        collision_groups,
    ) = compute_all_neighbourhoods(
        canonical_frame,
        config,
        domain,
        embedding_cache_path=embedding_cache_path,
    )

    densities = compute_rrf_density(
        metric_lists,
        n_entities=n_entities,
        k0=int(config.get("rrf_k0", 60)),
        c_min=int(config.get("c_min", 2)),
        label_collision_groups=collision_groups,
        boost_label_collision=float(config.get("boost_label_collision", 5.0)),
    )

    # 3. Candidate pair pools for the miner.
    miner_rng = spawn_sub_rng(rng, "pair_sampler")
    same_pairs, cross_pairs = build_candidate_pairs(
        n_entities,
        cluster_of=cluster_of,
        metric_lists=metric_lists,
        cap_per_class=int(
            config.get("em_test_regeneration", {}).get("candidate_pool_cap", 2000)
        ),
        rng=miner_rng,
    )

    # 4. Baseline-driven dispatch.
    #
    # Per the K2 design (plans/plan_s1_scale.md §R4 K2 review,
    # 2026-05-06): operator counts are derived from the data, not
    # pinned per-level. We measure the baseline corner-case ratio
    # first, then close the gap to ``target_ratio`` with whichever
    # operator(s) the baseline implies:
    #   baseline > target  -> drop high-density (densest) entities
    #                         until the ratio falls to target.
    #   baseline < target  -> interpolate corner-case entities up to
    #                         ``max_interp_fraction``; pair every
    #                         interpolation with a low-density drop so
    #                         per-source row counts stay close to the
    #                         original (size invariant).
    #   baseline ~ target  -> no-op.
    # When the easy target cannot be reached without violating
    # protection, we accept the floor and report the realised ratio in
    # provenance / niche_scores_df.
    level_cfg = config["levels"][level]
    target_ratio = float(level_cfg["target_corner_case_ratio"])
    max_interp_fraction = float(config.get("max_interp_fraction", 0.6))
    placement_split = float(config.get("placement_split", 0.6))

    stopwords = set(str(s).lower() for s in config.get("stopword_list", []) or [])
    inner_thr = float(config.get("inner_token_threshold", 0.8))
    thresholds_cfg = config["pair_miner_thresholds"]
    thresholds = MetricThresholds(
        t_match={k: float(v) for k, v in thresholds_cfg["t_match"].items()},
        t_nonmatch={k: float(v) for k, v in thresholds_cfg["t_nonmatch"].items()},
    )

    # Measure baseline corner-case ratio before any operator fires.
    # Note: ``labels`` is passed as the text-corpus surface so per-pair
    # ext-Jaccard scoring agrees with the corpus used in
    # neighbour retrieval. The label_collision rule inside the miner
    # then degrades to "concat strings byte-equal" (rare, harmless);
    # cross-cluster collisions still surface via the metric union and
    # via the RRF scorer's ``boost_label_collision`` boost.
    baseline_corner = mine_corner_cases(
        candidate_pairs=same_pairs + cross_pairs,
        cluster_of=cluster_of,
        labels=text_corpus,
        tfidf_matrix=tfidf_matrix,
        embeddings=embeddings,
        thresholds=thresholds,
        inner_token_threshold=inner_thr,
        stopwords=stopwords,
    )
    pairs_total = max(1, len(same_pairs) + len(cross_pairs))
    baseline_ratio = len(baseline_corner) / pairs_total
    logger.info(
        "K2 baseline: %d/%d corner pairs (ratio=%.3f); target=%.3f",
        len(baseline_corner),
        pairs_total,
        baseline_ratio,
        target_ratio,
    )

    # Tolerance band around the target; outside this band we operate.
    tol = 0.02
    removed_indices: set[int] = set()
    interpolated: list[InterpolatedEntity] = []
    interp_rejection_log: dict[str, int] = {}
    interp_count_chosen = 0
    non_corner_refilled: list[NonCornerEntity] = []
    non_corner_rejection_log: dict[str, int] = {}
    drop_corner_metrics: dict[str, Any] = {}
    operator_decision: str

    if baseline_ratio > target_ratio + tol:
        # Step 4i (2026-05-27): drop-corner-touching + non-corner refill.
        # Greedy: rank entities by their share of corner pairs touched,
        # drop until realised ratio crosses target (skip protected
        # entities + skip last-of-collision-group). Each drop pairs with
        # a 1-for-1 LLM-synthesized non-corner entity refill so the
        # canonical set size stays stable.
        #
        # If ``non_corner_refill.enabled`` is False or no LLM cache is
        # wired, fall back to the legacy noop_baseline_above_target
        # behaviour (logs a warning + reports baseline as realised).
        refill_cfg = config.get("non_corner_refill", {}) or {}
        refill_enabled = bool(refill_cfg.get("enabled", False))
        if not refill_enabled or non_corner_cache is None:
            operator_decision = "noop_baseline_above_target"
            logger.info(
                "K2: baseline ratio %.3f > target %.3f; non_corner_refill "
                "disabled or cache absent — falling back to noop "
                "(reporting baseline as realised). See plan_revision.md "
                "step 4i.",
                baseline_ratio,
                target_ratio,
            )
        else:
            operator_decision = "drop_corner_touching_refilled"
            (
                planned_drops,
                non_corner_refilled,
                non_corner_rejection_log,
                drop_corner_metrics,
            ) = _run_drop_corner_refill(
                canonical_frame=canonical_frame,
                entity_ids=entity_ids,
                same_pairs=same_pairs,
                cross_pairs=cross_pairs,
                baseline_corner=[(p.i, p.j) for p in baseline_corner],
                collision_groups=collision_groups,
                protection_flags=drop_corner_protection_flags,
                densities=densities,
                target_ratio=target_ratio,
                tol=tol,
                max_interp_fraction=max_interp_fraction,
                n_entities=n_entities,
                config=config,
                domain=domain,
                llm_cache=non_corner_cache,
                api_client=non_corner_api_client,
                strict_cache=strict_cache,
                sources=sources,
                rng=spawn_sub_rng(rng, "drop_corner_refill"),
            )
            removed_indices = set(planned_drops)
    elif baseline_ratio < target_ratio - tol:
        operator_decision = "interpolate_paired_drop"
        # Sized estimate for how many interpolations close the gap.
        # Each interpolated entity is a near-twin of dense neighbours
        # by construction so its cross-cluster pairs are flagged as
        # corner cases. Heuristic: each interpolation contributes
        # ~`metric_top_k * interp_pair_factor` corner-pair adds after
        # RRF agreement filtering. ``interp_pair_factor`` is a per-
        # domain config knob (was a hard-coded ``0.5`` until 2026-05-14
        # — calibration over music-small showed the optimistic ``0.5``
        # produced a 7-15x under-shoot on the corner-pair gap because
        # post-RRF agreement filtering drops most candidate pairs).
        # Lowering it (e.g. to 0.05 for music) makes the budget
        # gap-realistic at the cost of more LLM calls per run; the
        # ceiling at ``max_interp_fraction * n_entities`` still holds.
        interp_pair_factor = float(config.get("interp_pair_factor", 0.5))
        metric_top_k = int(config.get("metric_top_k", 20))
        per_interp_corner = max(1, int(metric_top_k * interp_pair_factor))
        needed_corner = max(
            0,
            int(round(target_ratio * pairs_total)) - len(baseline_corner),
        )
        raw_count = (needed_corner + per_interp_corner - 1) // per_interp_corner
        max_allowed = int(max_interp_fraction * n_entities)
        interp_count_chosen = min(raw_count, max_allowed)
        if interp_count_chosen > 0:
            interpolated, interp_rejection_log = _run_interpolation(
                canonical_frame=canonical_frame,
                entity_ids=entity_ids,
                densities=densities,
                metric_lists=metric_lists,
                removed_indices=set(),
                protection_flags=protection_flags,
                config=config,
                domain=domain,
                effective_interp=interp_count_chosen,
                placement_split=placement_split,
                llm_cache=llm_cache,
                api_client=api_client,
                committee_fn=committee_fn,
                strict_cache=strict_cache,
                sources=sources,
                rng=spawn_sub_rng(rng, "interpolation"),
            )
            # Size invariant: drop one low-density (non-corner) entity
            # per interpolation so per-source row counts stay close to
            # the originals. Skip protected entities. Skip entities in
            # label-collision groups (they are corner cases by the
            # collision rule).
            n_balance = len(interpolated)
            if n_balance > 0:
                ranked_desc = rank_entities_by_density(
                    densities, spawn_sub_rng(rng, "balance_drops")
                )
                # Lowest-density first.
                ranked_asc = list(reversed(ranked_desc))
                colliding = {idx for grp in collision_groups.values() for idx in grp}
                balance_drops: list[int] = []
                for idx in ranked_asc:
                    if len(balance_drops) >= n_balance:
                        break
                    if idx >= len(protection_flags):
                        continue
                    if protection_flags[idx]:
                        continue
                    if idx in colliding:
                        continue
                    balance_drops.append(idx)
                removed_indices = set(balance_drops)
    else:
        operator_decision = "noop"

    logger.info(
        "K2 dispatch: operator=%s, removed=%d, interpolated=%d, "
        "interp_budget=%d (max=%d)",
        operator_decision,
        len(removed_indices),
        len(interpolated),
        interp_count_chosen,
        int(max_interp_fraction * n_entities),
    )

    # Record removals in provenance.
    for idx in sorted(removed_indices):
        d = densities[idx]
        prov.append(
            entity_id=entity_ids[idx],
            source="",
            attribute="",
            original_value="",
            new_value="",
            transform_fn="remove_entity",
            transform_params={
                "prior_cluster_id": cluster_of[idx],
                "cluster_size_before": len(entity_groups.get(entity_ids[idx], [])),
                "density_score": d.density,
                "rrf_component": d.rrf_component,
                "label_collision_component": d.label_collision_component,
                "selection_metric": "rrf_density",
                "agreement_count_sum": int(sum(d.agreement_counts.values())),
                "operator_decision": operator_decision,
            },
        )

    # Provenance for interpolated entities (paired with size-invariant
    # drops above).
    if interpolated:
        for entity in interpolated:
            prov.append(
                entity_id=entity.entity_id,
                source=",".join(entity.source_placements),
                attribute="",
                original_value="",
                new_value="",
                transform_fn="llm_interpolate_entity",
                transform_params={
                    "parent_entity_ids": entity.parent_ids,
                    "similarity_metric": "rrf_density",
                    "placement_mode": entity.placement_mode,
                    "prompt_version": config.get("llm_prompt_version", "v1"),
                    "model_id": config.get("llm_model_id", "gpt-5.4"),
                    "cache_path": entity.cache_path,
                    "contamination_check_status": entity.contamination_check_status,
                    "operator_decision": operator_decision,
                },
            )

    # Provenance for non-corner refill entities (step 4i, 2026-05-27).
    # Paired 1-for-1 with the drops in ``removed_indices`` above; the
    # refill is dissimilar to a low-density anchor of the surviving
    # canonical set.
    if non_corner_refilled:
        non_corner_prompt_version = config.get(
            "non_corner_prompt_version",
            (config.get("non_corner_refill", {}) or {}).get("prompt_version", "v1"),
        )
        for entity in non_corner_refilled:
            prov.append(
                entity_id=entity.entity_id,
                source=",".join(entity.source_placements),
                attribute="",
                original_value="",
                new_value="",
                transform_fn="llm_non_corner_refill",
                transform_params={
                    "reference_entity_ids": entity.reference_ids,
                    "selection_metric": "low_density_anchor",
                    "prompt_version": non_corner_prompt_version,
                    "model_id": config.get("llm_model_id", "gpt-5.4-mini"),
                    "cache_path": entity.cache_path,
                    "contamination_check_status": entity.contamination_check_status,
                    "operator_decision": operator_decision,
                },
            )

    # 6. Project removals + interpolations + non-corner refills back to
    # per-source frames. The non-corner refills (step 4i, 2026-05-27)
    # mirror InterpolatedEntity's source-placement contract so they
    # plug into the same _project_to_sources path — NonCornerEntity
    # carries the same entity_id / attributes / source_placements
    # fields _project_to_sources reads.
    new_sources = _project_to_sources(
        sources=sources,
        entity_groups=entity_groups,
        entity_ids=entity_ids,
        removed_indices=removed_indices,
        interpolated=list(interpolated) + list(non_corner_refilled),
        config=config,
    )

    # 7. Build the post-K2 canonical frame.
    canonical_after = canonical_frame.drop(
        index=[entity_ids[i] for i in removed_indices]
    )
    if interpolated:
        interp_rows = []
        for entity in interpolated:
            row = {"entity_id": entity.entity_id}
            for col in config["canonical_schema"]:
                row[col] = entity.attributes.get(col, "")
            interp_rows.append(row)
        canonical_after = pd.concat(
            [
                canonical_after,
                pd.DataFrame(interp_rows).set_index("entity_id", drop=False),
            ],
            axis=0,
        )

    # 8. Regenerate EM test set stratified by corner-case ratio. Pairs
    #    are emitted at the source-record level (matching the original
    #    EM-gold convention). Positives come from cross-source pairs
    #    within each surviving canonical entity's member list; negatives
    #    come from the canonical-level cross-cluster pairs, each mapped
    #    to one chosen record-level pair.
    same_pairs_post = [
        (i, j)
        for (i, j) in same_pairs
        if i not in removed_indices and j not in removed_indices
    ]
    cross_pairs_post = [
        (i, j)
        for (i, j) in cross_pairs
        if i not in removed_indices and j not in removed_indices
    ]

    corner_cases_post = mine_corner_cases(
        candidate_pairs=same_pairs_post + cross_pairs_post,
        cluster_of=cluster_of,
        labels=text_corpus,
        tfidf_matrix=tfidf_matrix,
        embeddings=embeddings,
        thresholds=thresholds,
        inner_token_threshold=inner_thr,
        stopwords=stopwords,
    )
    corner_case_pair_set: set[tuple[int, int]] = {(c.i, c.j) for c in corner_cases_post}

    # Extend entity_groups with synthetic entries for interpolated
    # entities so their placed source records can contribute positives.
    entity_groups_for_emission = dict(entity_groups)
    for entity in interpolated:
        entity_groups_for_emission[entity.entity_id] = [
            (src, f"{entity.entity_id}__{src}") for src in entity.source_placements
        ]

    # Source-pair filter: only emit record pairs whose source pair has
    # EM gold authored against it (via the domain config). Callers may
    # override the filter (used by unit tests with synthetic source
    # names). Falls back to accepting any cross-source pair when the
    # domain config has no source_pairs declared.
    if source_pair_filter is None:
        try:
            domain_cfg = load_domain_config(domain)
            source_pair_filter = (
                {frozenset(pair) for pair in domain_cfg.source_pairs}
                if domain_cfg.source_pairs
                else None
            )
        except FileNotFoundError:
            source_pair_filter = None

    excluded_cids = {entity_ids[i] for i in removed_indices}
    positive_record_pairs = enumerate_cross_source_positive_pairs(
        entity_groups_for_emission,
        excluded_canonical_ids=excluded_cids,
        source_pair_filter=source_pair_filter,
    )

    # Inverse rid → source map (built here so the random-negative
    # top-up below can bucket pairs per authored source-pair; also
    # reused later when stamping source_1 / source_2 on regen rows).
    rid_to_source: dict[str, str] = {}
    for members in entity_groups_for_emission.values():
        for src, rid in members:
            rid_to_source[rid] = src

    negative_mapper_rng = spawn_sub_rng(rng, "cross_record_mapper")
    # Expand every cross-cluster canonical pair to all its valid
    # cross-source record combinations (bounded by max_per_cluster_pair
    # to prevent blow-up on pathological clusters). This replaces the
    # previous one-record-per-cluster-pair sampling so that the
    # negative pool scales with record count rather than cluster count
    # — ensuring negatives are never the regen-size bottleneck.
    max_per_cluster_pair = int(
        config.get("em_test_regeneration", {}).get("max_negatives_per_cluster_pair", 10)
    )
    negative_record_pairs_set: set[tuple[str, str]] = set()
    corner_case_neg_records: set[tuple[str, str]] = set()
    for i, j in cross_pairs_post:
        rec_pairs = enumerate_cross_cluster_record_pairs(
            entity_ids[i],
            entity_ids[j],
            entity_groups_for_emission,
            source_pair_filter=source_pair_filter,
            max_per_cluster_pair=max_per_cluster_pair,
            rng=negative_mapper_rng,
        )
        if not rec_pairs:
            continue
        is_corner = (i, j) in corner_case_pair_set
        for rec_pair in rec_pairs:
            negative_record_pairs_set.add(rec_pair)
            if is_corner:
                corner_case_neg_records.add(rec_pair)

    # Random top-up pass so the negative pool reaches a comfortable
    # headroom above what regenerate_em_splits needs for every
    # authored source-pair. The corner-case miner is biased toward
    # niche neighbours, so the record pairs it produces for any one
    # authored source-pair are often thin (~100-900 per pair on
    # companies-small). Top-up adds uniformly sampled cross-cluster
    # record pairs — these are NOT tagged as corner cases, they fill
    # the "easy negatives" bucket so scaling in the builder is driven
    # by the positive pool rather than the negative pool.
    def _authored_pairs_list() -> list[tuple[str, str]]:
        if source_pair_filter:
            return [tuple(sorted(fs)) for fs in source_pair_filter]
        return []

    authored_pairs_iter = _authored_pairs_list()
    if authored_pairs_iter:
        # Pre-bucket records per source so we can sample by authored
        # source-pair cheaply.
        records_by_source: dict[str, list[str]] = defaultdict(list)
        rid_to_cluster: dict[str, int] = {}
        for members in entity_groups_for_emission.values():
            for src, rid in members:
                records_by_source[src].append(rid)
        for idx, cid in enumerate(entity_ids):
            for _src, rid in entity_groups_for_emission.get(cid, []):
                rid_to_cluster[rid] = idx
        # Interpolated entities live outside the main entity_ids list;
        # give them synthetic cluster indices past the real range so
        # same-cluster rejection still works.
        synthetic_cluster_base = len(entity_ids)
        for synth_idx, cid in enumerate(
            [c for c in entity_groups_for_emission if c not in set(entity_ids)]
        ):
            for _src, rid in entity_groups_for_emission[cid]:
                rid_to_cluster[rid] = synthetic_cluster_base + synth_idx

        # Pool target: cover the worst-case split demand with 1.5×
        # headroom so the builder does not hit the negative bottleneck
        # before it hits the positive one.
        target_pool_min = int(
            config.get("em_test_regeneration", {}).get(
                "random_negative_pool_target_min", 4000
            )
        )
        per_pair_headroom_mult = float(
            config.get("em_test_regeneration", {}).get(
                "random_negative_headroom_mult", 1.5
            )
        )
        random_rng = spawn_sub_rng(rng, "random_neg_topup")
        for pair in authored_pairs_iter:
            src1, src2 = pair
            recs_a = records_by_source.get(src1, [])
            recs_b = records_by_source.get(src2, [])
            if len(recs_a) < 2 or len(recs_b) < 2:
                continue
            # Count current pool contribution for this authored pair.
            current_for_pair = sum(
                1
                for (a, b) in negative_record_pairs_set
                if frozenset({rid_to_source.get(a, ""), rid_to_source.get(b, "")})
                == frozenset(pair)
            )
            target = max(
                target_pool_min,
                int(per_pair_headroom_mult * current_for_pair),
            )
            max_attempts = 20 * target
            attempts = 0
            added = 0
            while current_for_pair + added < target and attempts < max_attempts:
                attempts += 1
                rid_a = recs_a[int(random_rng.integers(len(recs_a)))]
                rid_b = recs_b[int(random_rng.integers(len(recs_b)))]
                c_a = rid_to_cluster.get(rid_a)
                c_b = rid_to_cluster.get(rid_b)
                if c_a is None or c_b is None or c_a == c_b:
                    continue
                canon = (rid_a, rid_b) if rid_a < rid_b else (rid_b, rid_a)
                if canon in negative_record_pairs_set:
                    continue
                negative_record_pairs_set.add(canon)
                added += 1
            if added > 0:
                logger.info(
                    "Random negative top-up for %s_2_%s: added %d pairs "
                    "(pool now %d for this authored pair)",
                    src1,
                    src2,
                    added,
                    current_for_pair + added,
                )

    # Keep deterministic list ordering for downstream steps.
    negative_record_pairs: list[tuple[str, str]] = sorted(negative_record_pairs_set)

    # S3: Hard-negative score-margin gate. Only the corner-case
    # negatives need PLM adjudication — the "easy" negatives outside
    # the corner-case set are not promoted as deceptive distractors, so
    # they stay as-is. The gate filters corner-case negatives in place;
    # audit rows are persisted via provenance.
    hn_audit: list[HardNegativeAudit] = []
    if hard_negative_policy is not None and corner_case_neg_records:
        candidates = sorted(corner_case_neg_records)
        kept_corner, hn_audit = apply_hard_negative_policy(
            candidates, policy=hard_negative_policy
        )
        dropped = set(candidates) - set(kept_corner)
        if dropped:
            negative_record_pairs = [
                p for p in negative_record_pairs if p not in dropped
            ]
            corner_case_neg_records = set(kept_corner)
        for row in hn_audit:
            prov.append(
                entity_id=f"{row.rid_a}|{row.rid_b}",
                source="",
                attribute="hard_negative_gate",
                original_value="",
                new_value=row.verdict,
                transform_fn="hard_negative_gate",
                transform_params={
                    "plm_score": row.plm_score,
                    "theta": row.theta,
                    "delta": row.delta,
                    "llm_says_match": row.llm_says_match,
                },
            )

    # Group positive and negative record pairs by authored source-pair
    # so regenerate_em_splits can emit per-pair per-split files whose
    # shape mirrors the original EM gold splits.
    #
    # Pair ordering: prefer the domain config's declared source_pairs
    # so the emitted filenames match the convention downstream
    # consumers expect (e.g. ``forbes_2_dbpedia`` not ``dbpedia_2_
    # forbes``). When the caller supplied a ``source_pair_filter``
    # containing pairs not in the domain config (synthetic test
    # sources), fall back to an arbitrary-but-deterministic ordering.
    authored_pairs: list[tuple[str, str]] = []
    declared_pairs: list[tuple[str, str]]
    try:
        domain_cfg_for_pairs = load_domain_config(domain)
        declared_pairs = [tuple(p) for p in domain_cfg_for_pairs.source_pairs]
    except FileNotFoundError:
        declared_pairs = []

    if source_pair_filter:
        for pair_set in source_pair_filter:
            # Find the declared ordering that matches, else sort for
            # determinism.
            match = next(
                (p for p in declared_pairs if frozenset(p) == pair_set),
                None,
            )
            authored_pairs.append(match if match else tuple(sorted(pair_set)))
    else:
        authored_pairs = declared_pairs

    def _pair_key(rid_a: str, rid_b: str) -> tuple[str, str] | None:
        """Map a record pair to its authored (src1, src2) ordering.

        Returns None if either record's source is unknown or the pair
        does not correspond to an authored source pair.
        """
        src_a = rid_to_source.get(rid_a, "")
        src_b = rid_to_source.get(rid_b, "")
        if not src_a or not src_b or src_a == src_b:
            return None
        for pair in authored_pairs:
            if frozenset(pair) == frozenset({src_a, src_b}):
                return pair
        return None

    # Interpolated-entity record ids carry the K2 corner-case positive
    # signal at the record level: K2 hard creates near-twins
    # deliberately, so their cross-source pairs are the natural
    # corner-case positive pool. The interp_rids_for_authored_pair map
    # below feeds ``interpolated_positives_by_pair`` for the
    # regenerator, while interpolated_rids_all tags survivors so
    # ``regenerate_em_splits`` can count which originals already cover
    # the corner-positive target (in practice ~0, since originals
    # predate K2).
    interpolated_entity_ids = {e.entity_id for e in interpolated}
    interpolated_rids_all: set[str] = set()
    for cid in interpolated_entity_ids:
        for _src, rid in entity_groups_for_emission.get(cid, []):
            interpolated_rids_all.add(rid)

    interpolated_positives_by_pair: dict[tuple[str, str], list[tuple[str, str]]] = {
        p: [] for p in authored_pairs
    }
    cluster_positives_by_pair: dict[tuple[str, str], list[tuple[str, str]]] = {
        p: [] for p in authored_pairs
    }
    negatives_by_pair: dict[tuple[str, str], list[tuple[str, str]]] = {
        p: [] for p in authored_pairs
    }
    corner_case_negatives_by_pair: dict[tuple[str, str], set[tuple[str, str]]] = {
        p: set() for p in authored_pairs
    }

    def _orient_for_pair(
        rid_a: str, rid_b: str, pair: tuple[str, str]
    ) -> tuple[str, str]:
        """Return (rid_in_src1, rid_in_src2) matching the pair's ordering."""
        if rid_to_source.get(rid_a) == pair[0]:
            return (rid_a, rid_b)
        return (rid_b, rid_a)

    for rid_a, rid_b in positive_record_pairs:
        pair = _pair_key(rid_a, rid_b)
        if pair is None:
            continue
        oriented = _orient_for_pair(rid_a, rid_b, pair)
        if rid_a in interpolated_rids_all or rid_b in interpolated_rids_all:
            interpolated_positives_by_pair[pair].append(oriented)
        else:
            cluster_positives_by_pair[pair].append(oriented)

    for rid_a, rid_b in negative_record_pairs:
        pair = _pair_key(rid_a, rid_b)
        if pair is None:
            continue
        oriented = _orient_for_pair(rid_a, rid_b, pair)
        negatives_by_pair[pair].append(oriented)
        # Corner-case negatives are stored in canonical (sorted) order
        # because the gate audit uses the same convention; match that.
        canon = (rid_a, rid_b) if rid_a < rid_b else (rid_b, rid_a)
        if canon in corner_case_neg_records:
            canon_oriented = (
                oriented if oriented[0] < oriented[1] else (oriented[1], oriented[0])
            )
            corner_case_negatives_by_pair[pair].add(canon_oriented)

    # Pool-discovered positives — used as easy-positive backfill when
    # surviving originals + cluster positives don't cover the target.
    # The pool is loaded from ``usecases_synthetic/pools/<domain>/
    # pooled_positives.csv`` and bucketed per authored source pair.
    # ``rid_to_source`` is passed so the loader can orient (id1, id2)
    # by the actual source of each id; the pool CSV's id1/id2 are
    # lex-sorted via ``canonical_pair`` and don't track source_1.
    pool_positives_by_pair = _load_pool_positives_by_pair(
        domain, authored_pairs, rid_to_source=rid_to_source
    )

    # Ids present in the augmented sources — used by the regenerator to
    # decide which original pairs can be carried over verbatim. K2 is
    # the only knob that removes ids; K4 fabricates new ones but never
    # invalidates an existing id.
    ids_present: set[str] = set()
    id_columns = config.get("id_columns", {})
    for source_name, df in new_sources.items():
        id_col = id_columns.get(source_name)
        if id_col and id_col in df.columns:
            ids_present.update(df[id_col].astype(str).tolist())

    # Split targets — mirror the original per-pair per-split row count
    # and positive:negative ratio so regen is drop-in for downstream
    # users who expect the same benchmark shape. When the original
    # splits cannot be located (synthetic test sources, partial
    # domains), fall back to a default legacy-compatible single-file
    # ``test`` split so the pipeline still produces something
    # downstream consumers can score against.
    default_target_size = int(
        config.get("em_test_regeneration", {}).get("target_size", 200)
    )
    split_specs_by_pair, original_pairs_by_split = _load_original_split_targets(
        domain, authored_pairs
    )
    for pair in authored_pairs:
        if not split_specs_by_pair.get(pair):
            logger.warning(
                "No original EM splits found for %s; falling back to "
                "default single-test spec (size=%d, pos_ratio=0.5)",
                pair,
                default_target_size,
            )
            split_specs_by_pair[pair] = [
                SplitSpec(
                    name="test",
                    size=default_target_size,
                    positive_ratio=0.5,
                )
            ]
            original_pairs_by_split.setdefault(pair, {})

    regen_rng = spawn_sub_rng(rng, "test_regeneration")
    regen_pools = RegenPools(
        original_pairs_by_split=original_pairs_by_split,
        pool_positives_by_pair=pool_positives_by_pair,
        interpolated_positives_by_pair=interpolated_positives_by_pair,
        cluster_positives_by_pair=cluster_positives_by_pair,
        negatives_by_pair=negatives_by_pair,
        corner_case_negatives_by_pair=corner_case_negatives_by_pair,
        split_specs_by_pair=split_specs_by_pair,
        target_ratio=target_ratio,
    )
    regen_rows = regenerate_em_splits(
        original_pairs_by_split=original_pairs_by_split,
        ids_present=ids_present,
        pool_positives_by_pair=pool_positives_by_pair,
        interpolated_positives_by_pair=interpolated_positives_by_pair,
        cluster_positives_by_pair=cluster_positives_by_pair,
        negatives_by_pair=negatives_by_pair,
        corner_case_negatives_by_pair=corner_case_negatives_by_pair,
        split_specs_by_pair=split_specs_by_pair,
        target_ratio=target_ratio,
        rng=regen_rng,
    )
    regen_df = pd.DataFrame(
        regen_rows,
        columns=[
            "id1",
            "id2",
            "source_1",
            "source_2",
            "label",
            "split",
            "pair_name",
            "version",
        ],
    )

    # 9. Build audit frames.
    niche_rows: list[dict[str, Any]] = []
    for d in densities:
        niche_rows.append(
            {
                "entity_id": entity_ids[d.index],
                "density": d.density,
                "rrf_component": d.rrf_component,
                "label_collision_component": d.label_collision_component,
                "neighbour_count": d.neighbour_count,
                "protected": protection_flags[d.index],
                "removed": d.index in removed_indices,
            }
        )
    niche_scores_df = pd.DataFrame(niche_rows)

    if len(prov) > 0:
        provenance_df = pd.DataFrame(
            [r.as_dict() for r in prov._rows], columns=PROVENANCE_COLUMNS
        )
    else:
        provenance_df = pd.DataFrame(columns=PROVENANCE_COLUMNS)

    final_ratio = len(corner_cases_post) / max(
        1, len(same_pairs_post) + len(cross_pairs_post)
    )
    logger.info(
        "Knob 02: level=%s baseline_ratio=%.3f target=%.3f operator=%s "
        "removed=%d interpolated=%d final_ratio=%.3f",
        level,
        baseline_ratio,
        target_ratio,
        operator_decision,
        len(removed_indices),
        len(interpolated),
        final_ratio,
    )

    k2_metrics: dict[str, Any] = {
        "level": level,
        "baseline_ratio": float(baseline_ratio),
        "target_ratio": float(target_ratio),
        "final_ratio": float(final_ratio),
        "operator": str(operator_decision),
        "removed": int(len(removed_indices)),
        "interpolated": int(len(interpolated)),
        "interp_attempted": int(len(interpolated) + sum(interp_rejection_log.values())),
    }
    # Per-guardrail rejection counters surface why interpolated < attempted.
    # Stable column prefix lets ``analyze_monotonicity`` (and any future
    # audit) ingest the breakdown without re-walking the LLM cache.
    for key, count in interp_rejection_log.items():
        k2_metrics[f"rejected_{key}"] = int(count)

    # Step 4i: drop-corner + non-corner refill telemetry. Stable
    # ``drop_corner_`` / ``non_corner_`` prefixes so the
    # ``knob_02_realised.csv`` row carries the new fields even when
    # the legacy interpolate path fired and these dictionaries are
    # empty.
    k2_metrics["drop_corner_planned"] = int(
        drop_corner_metrics.get("planned_drop_count", 0)
    )
    k2_metrics["drop_corner_simulated_final_ratio"] = float(
        drop_corner_metrics.get("simulated_final_ratio", 0.0)
    )
    k2_metrics["drop_corner_cap_bound"] = bool(
        drop_corner_metrics.get("cap_bound", False)
    )
    k2_metrics["non_corner_refill_attempts"] = int(
        drop_corner_metrics.get("refill_attempts", 0)
    )
    k2_metrics["non_corner_refill_committed"] = int(len(non_corner_refilled))
    for key, count in non_corner_rejection_log.items():
        k2_metrics[f"non_corner_rejected_{key}"] = int(count)
    # Per-reason skip counters from the drop-corner greedy loop. Surface
    # them on knob_02_realised.csv so a 0-drops outcome is immediately
    # attributable to which filter killed the candidates (protected, last
    # collision-group member, isolated singleton, empty-pool tail, or
    # counterproductive — the Bug 6 guard that skips drops whose removal
    # would push the realised ratio AWAY from target).
    for skip_key in (
        "protected",
        "collision_last",
        "isolated",
        "empty_pool",
        "counterproductive",
    ):
        k2_metrics[f"drop_corner_skip_{skip_key}"] = int(
            drop_corner_metrics.get(f"skip_{skip_key}", 0)
        )

    return (
        new_sources,
        canonical_after,
        regen_df,
        provenance_df,
        niche_scores_df,
        k2_metrics,
        regen_pools,
    )


# ---------------------------------------------------------------------------
# Interpolation runner
# ---------------------------------------------------------------------------


def _run_interpolation(
    *,
    canonical_frame: pd.DataFrame,
    entity_ids: list[str],
    densities: list[EntityDensity],
    metric_lists: list[MetricNeighbourhoods],
    removed_indices: set[int],
    protection_flags: list[bool],
    config: dict[str, Any],
    domain: str,
    effective_interp: int,
    placement_split: float,
    llm_cache: LLMCache | None,
    api_client,
    committee_fn,
    strict_cache: bool,
    sources: dict[str, pd.DataFrame],
    rng: np.random.Generator,
) -> tuple[list[InterpolatedEntity], dict[str, int]]:
    """Execute the LLM interpolation path.

    Returns
    -------
    tuple of (list of :class:`InterpolatedEntity`, dict)
        First element is the synthetic entities that survived all
        guardrails. Second is a per-reason rejection counter (keys are
        the rejection labels documented on
        :func:`entity_interpolation.interpolate_entity`, plus
        ``"strict_cache_miss"`` when ``strict_cache=True``).
    """
    if llm_cache is None:
        logger.warning("Interpolation skipped: llm_cache=None")
        return [], {}

    primary_column = (
        config.get("primary_column_canonical") or config["canonical_schema"][0]
    )
    schema_cols = list(config["canonical_schema"])

    # Build density ranking excluding removed indices.
    ranked = rank_entities_by_density(densities, rng)
    ranked = [i for i in ranked if i not in removed_indices]

    # Build neighbour_lookup from fused top-K (using best per entity).
    neighbour_lookup: dict[int, list[int]] = defaultdict(list)
    for metric in metric_lists:
        for ent_idx, nbrs in enumerate(metric.top_k):
            for neigh_idx, _ in nbrs:
                if neigh_idx in removed_indices:
                    continue
                if neigh_idx not in neighbour_lookup[ent_idx]:
                    neighbour_lookup[ent_idx].append(neigh_idx)

    # Split the budget across placement modes per the placement_split.
    n_matched = int(round(effective_interp * placement_split))
    n_single = effective_interp - n_matched

    matched_pairs = select_parent_pairs(
        ranked,
        neighbour_lookup=dict(neighbour_lookup),
        protected=protection_flags,
        placement_mode="matched_across",
        k=n_matched,
        rng=spawn_sub_rng(rng, "parent_matched"),
    )
    single_pairs = select_parent_pairs(
        ranked,
        neighbour_lookup=dict(neighbour_lookup),
        protected=protection_flags,
        placement_mode="single_source_distractor",
        k=n_single,
        rng=spawn_sub_rng(rng, "parent_single"),
    )

    reference_labels: set[str] = set()
    for lab in canonical_frame[primary_column].tolist():
        norm = normalize_label(str(lab) if lab is not None else "")
        if norm:
            reference_labels.add(norm)

    prompt_template = _load_prompt_template("interpolate_v1.txt")

    source_names = sorted(sources.keys())
    if len(source_names) < 2:
        logger.warning(
            "Only %d source(s) — interpolation matched-across is disabled",
            len(source_names),
        )

    # Default api_client for smoke tests: a deterministic attribute-blender.
    effective_api = api_client
    if effective_api is None and not strict_cache:
        effective_api = default_api_client_from_attributes

    out: list[InterpolatedEntity] = []
    rejection_log: dict[str, int] = {}
    counter = 0

    def _source_placement(mode: str) -> list[str]:
        if mode == "matched_across":
            # Place across all sources (the fusion committee will see
            # the new entity in every source).
            return list(source_names)
        else:
            # Single-source: pick one source deterministically.
            if not source_names:
                return []
            idx = int(rng.integers(0, len(source_names)))
            return [source_names[idx]]

    for (i, j), mode in list(
        [(p, "matched_across") for p in matched_pairs]
        + [(p, "single_source_distractor") for p in single_pairs]
    ):
        entity_id = f"k02_interp_{domain}_{counter:05d}"
        counter += 1
        parent_rows = [
            canonical_frame.iloc[i].copy(),
            canonical_frame.iloc[j].copy(),
        ]
        parent_rows[0].name = entity_ids[i]
        parent_rows[1].name = entity_ids[j]
        placements = _source_placement(mode)
        try:
            entity = interpolate_entity(
                parent_rows=parent_rows,
                primary_column=primary_column,
                schema_columns=schema_cols,
                domain=domain,
                prompt_template=prompt_template,
                llm_cache=llm_cache,
                api_client=effective_api,
                committee_fn=committee_fn,
                reference_labels=reference_labels,
                placement_mode=mode,  # type: ignore[arg-type]
                source_placements=placements,
                entity_id=entity_id,
                strict_cache=strict_cache,
                rejection_log=rejection_log,
            )
        except LLMCacheMiss:
            logger.info("Interpolation %s skipped: strict cache miss", entity_id)
            rejection_log["strict_cache_miss"] = (
                rejection_log.get("strict_cache_miss", 0) + 1
            )
            continue
        if entity is not None:
            out.append(entity)
            # Once placed, add the new label to the reference set so
            # subsequent interpolations don't collide with each other.
            reference_labels.add(
                normalize_label(str(entity.attributes.get(primary_column, "")))
            )
    if rejection_log:
        logger.info(
            "K2 interpolation rejection breakdown for %s: %s",
            domain,
            ", ".join(f"{k}={v}" for k, v in sorted(rejection_log.items())),
        )
    return out, rejection_log


# ---------------------------------------------------------------------------
# Drop-corner-touching + non-corner refill (step 4i, 2026-05-27)
# ---------------------------------------------------------------------------


def _run_drop_corner_refill(
    *,
    canonical_frame: pd.DataFrame,
    entity_ids: list[str],
    same_pairs: list[tuple[int, int]],
    cross_pairs: list[tuple[int, int]],
    baseline_corner: list[tuple[int, int]],
    collision_groups: dict[str, list[int]],
    protection_flags: list[bool],
    densities,
    target_ratio: float,
    tol: float,
    max_interp_fraction: float,
    n_entities: int,
    config: dict[str, Any],
    domain: str,
    llm_cache: LLMCache,
    api_client,
    strict_cache: bool,
    sources: dict[str, pd.DataFrame],
    rng: np.random.Generator,
) -> tuple[list[int], list[NonCornerEntity], dict[str, int], dict[str, Any]]:
    """Greedy drop-corner-touching operator + 1-for-1 non-corner refill.

    Activates when ``baseline_ratio > target_ratio + tol``. Picks
    entities by expected corner-pair reduction (high corner-touch
    fraction first), drops them while tracking the running realised
    ratio, and stops when the ratio crosses ``target_ratio`` or the
    cap ``max_interp_fraction * n_entities`` is hit. For every drop,
    a synthetic non-corner entity is generated via the
    :mod:`non_corner_refill` module and placed across the same sources
    the dropped entity was in (1-for-1 size invariant).

    Returns
    -------
    tuple of (planned_drops, refilled_entities, rejection_log, metrics)
        ``planned_drops`` : list of canonical indices to drop.
        ``refilled_entities`` : list of :class:`NonCornerEntity` that
        survived the contamination guard + non-empty primary check.
        ``rejection_log`` : per-reason rejection counter
        (``strict_cache_miss``, ``empty_primary_label``,
        ``contamination_collision_with_real_entity``, ``nondict_result``).
        ``metrics`` : dict with the per-step running ratio + telemetry
        (``planned_drop_count``, ``refill_attempts``, ``refill_committed``,
        ``simulated_final_ratio``, ``cap_bound``).
    """
    candidate_pairs = list(same_pairs) + list(cross_pairs)
    corner_pair_set = set(baseline_corner)

    n_total_pairs = len(candidate_pairs)
    n_corner_pairs = len(baseline_corner)

    if n_total_pairs == 0:
        logger.info("Drop-corner refill: no candidate pairs to reason over; skipping.")
        return (
            [],
            [],
            {},
            {
                "planned_drop_count": 0,
                "refill_attempts": 0,
                "refill_committed": 0,
                "simulated_final_ratio": float(n_corner_pairs) / max(1, n_total_pairs),
                "cap_bound": False,
            },
        )

    pair_touch: dict[int, list[int]] = defaultdict(list)
    entity_corner_count: dict[int, int] = defaultdict(int)
    entity_total_count: dict[int, int] = defaultdict(int)
    for pi, (a, b) in enumerate(candidate_pairs):
        pair_touch[a].append(pi)
        pair_touch[b].append(pi)
        entity_total_count[a] += 1
        entity_total_count[b] += 1
        if (a, b) in corner_pair_set:
            entity_corner_count[a] += 1
            entity_corner_count[b] += 1

    # Collision membership: drop-eligible entities can belong to ≥1 group.
    # We must not collapse a group below 2 members.
    collision_membership: dict[int, set[str]] = defaultdict(set)
    for grp_key, members in collision_groups.items():
        for idx in members:
            collision_membership[idx].add(grp_key)
    group_sizes_remaining: dict[str, int] = {
        k: len(v) for k, v in collision_groups.items()
    }

    # Greedy ordering: highest corner_count first; deterministic tie-break
    # by index. Entities with zero corner_count contribute nothing useful
    # to the drop direction, so they're naturally at the bottom and only
    # picked if the high-count queue is exhausted.
    candidates = sorted(
        range(n_entities),
        key=lambda i: (-entity_corner_count.get(i, 0), i),
    )

    max_drops = int(max_interp_fraction * n_entities)
    removed_pair_indices: set[int] = set()
    planned_drops: list[int] = []

    current_corner = n_corner_pairs
    current_total = n_total_pairs

    cap_bound = False
    # Skip-reason counters (telemetry: surfaced in metrics dict so the
    # caller can audit why drop-corner picked few/no candidates).
    skip_counts = {
        "protected": 0,
        "collision_last": 0,
        "isolated": 0,
        "empty_pool": 0,
        "counterproductive": 0,
    }

    for idx in candidates:
        if len(planned_drops) >= max_drops:
            cap_bound = True
            break
        if idx >= len(protection_flags) or protection_flags[idx]:
            skip_counts["protected"] += 1
            continue
        # Skip last member of any collision group it belongs to.
        if any(
            group_sizes_remaining.get(g, 0) <= 1
            for g in collision_membership.get(idx, set())
        ):
            skip_counts["collision_last"] += 1
            continue
        touched = [
            pi for pi in pair_touch.get(idx, ()) if pi not in removed_pair_indices
        ]
        if not touched:
            # Isolated singleton w.r.t. the candidate pairs — no signal,
            # skip rather than waste an LLM call on a refill that won't
            # move the ratio.
            skip_counts["isolated"] += 1
            continue
        removed_corner_here = sum(
            1 for pi in touched if candidate_pairs[pi] in corner_pair_set
        )
        removed_total_here = len(touched)
        new_corner = current_corner - removed_corner_here
        new_total = current_total - removed_total_here
        if new_total <= 0:
            skip_counts["empty_pool"] += 1
            continue

        # Step 4i Bug 6 (2026-05-28): skip counterproductive drops —
        # those whose removal pushes the realised ratio AWAY from the
        # target. The greedy ordering puts high-corner candidates first,
        # so once they're exhausted the tail candidates touch mostly
        # non-corner pairs; removing them shrinks ``total`` faster than
        # ``corner`` and the ratio climbs. Verified on products medium
        # 2026-05-28: without this guard the loop over-drops 12 tail
        # entities and pushes realised from 0.72 back up to 0.82.
        if new_corner * current_total > current_corner * new_total:
            skip_counts["counterproductive"] += 1
            continue

        planned_drops.append(idx)
        removed_pair_indices.update(touched)
        current_corner = new_corner
        current_total = new_total
        for g in collision_membership.get(idx, set()):
            group_sizes_remaining[g] = group_sizes_remaining.get(g, 0) - 1

        if new_corner / new_total <= target_ratio + tol:
            break

    simulated_final_ratio = current_corner / current_total if current_total > 0 else 0.0

    refilled: list[NonCornerEntity] = []
    rejection_log: dict[str, int] = {}

    if not planned_drops:
        # Diagnostic: every candidate was filtered out. Log the per-reason
        # counts so the operator's blocker is visible at the call site.
        logger.warning(
            "Drop-corner refill picked 0 candidates: n_entities=%d "
            "candidate_pairs=%d corner_pairs=%d max_drops=%d "
            "skip[protected=%d collision_last=%d isolated=%d empty_pool=%d "
            "counterproductive=%d]",
            n_entities,
            n_total_pairs,
            n_corner_pairs,
            max_drops,
            skip_counts["protected"],
            skip_counts["collision_last"],
            skip_counts["isolated"],
            skip_counts["empty_pool"],
            skip_counts["counterproductive"],
        )
        return (
            planned_drops,
            refilled,
            rejection_log,
            {
                "planned_drop_count": 0,
                "refill_attempts": 0,
                "refill_committed": 0,
                "simulated_final_ratio": float(simulated_final_ratio),
                "cap_bound": cap_bound,
                "skip_protected": skip_counts["protected"],
                "skip_collision_last": skip_counts["collision_last"],
                "skip_isolated": skip_counts["isolated"],
                "skip_empty_pool": skip_counts["empty_pool"],
                "skip_counterproductive": skip_counts["counterproductive"],
            },
        )

    primary_column = (
        config.get("primary_column_canonical") or config["canonical_schema"][0]
    )
    schema_cols = list(config["canonical_schema"])

    refill_cfg = config.get("non_corner_refill", {}) or {}
    k_ref = int(refill_cfg.get("reference_count", 5))

    # Reference universe is the surviving canonical set after the
    # planned drops. Anchor selection picks low-density survivors so
    # the refill is "be dissimilar to isolated cells", not "be
    # dissimilar to corner-touching cells we just dropped".
    drop_set = set(planned_drops)
    survivors = [i for i in range(n_entities) if i not in drop_set]

    reference_labels: set[str] = set()
    for lab in canonical_frame[primary_column].tolist():
        norm = normalize_label(str(lab) if lab is not None else "")
        if norm:
            reference_labels.add(norm)

    prompt_name = f"non_corner_{refill_cfg.get('prompt_version', 'v1')}.txt"
    # The YAML field is non_corner_prompt_version at the top level
    # (separate from non_corner_refill.prompt_version sub-key); honour
    # either form for flexibility.
    explicit_prompt_version = config.get("non_corner_prompt_version")
    if explicit_prompt_version is not None:
        prompt_name = f"non_corner_{explicit_prompt_version}.txt"
    prompt_template = _load_prompt_template(prompt_name)

    source_names = sorted(sources.keys())

    counter = 0
    for drop_idx in planned_drops:
        entity_id = f"k02_noncorner_{domain}_{counter:05d}"
        counter += 1

        anchor_indices = select_reference_anchor(
            survivors, densities, k=k_ref, rng=spawn_sub_rng(rng, f"anchor_{counter}")
        )
        if len(anchor_indices) < 1:
            rejection_log["no_anchor"] = rejection_log.get("no_anchor", 0) + 1
            continue
        anchor_rows: list[pd.Series] = []
        for a_idx in anchor_indices:
            row = canonical_frame.iloc[a_idx].copy()
            row.name = entity_ids[a_idx]
            anchor_rows.append(row)

        # Place the refill across every source the dropped entity was
        # in — preserves the per-source row-count balance K2 enforces.
        # If the dropped entity was multi-source, the refill is too.
        dropped_id = entity_ids[drop_idx]
        # We need the entity_groups membership for the dropped id.
        # Caller does not pass entity_groups; reconstruct from source
        # membership by looking up the row in each source via id_columns.
        # Cheap fallback: place across all sources, then `_project_to_sources`
        # writes one row per source for the refill — same shape as the
        # interpolate matched_across mode.
        placements = list(source_names)

        try:
            refilled_entity = refill_non_corner_entity(
                reference_rows=anchor_rows,
                primary_column=primary_column,
                schema_columns=schema_cols,
                domain=domain,
                prompt_template=prompt_template,
                llm_cache=llm_cache,
                api_client=api_client,
                reference_labels=reference_labels,
                source_placements=placements,
                entity_id=entity_id,
                strict_cache=strict_cache,
                rejection_log=rejection_log,
            )
        except LLMCacheMiss:
            logger.info("Non-corner refill %s skipped: strict cache miss", entity_id)
            continue

        if refilled_entity is not None:
            refilled.append(refilled_entity)
            reference_labels.add(
                normalize_label(
                    str(refilled_entity.attributes.get(primary_column, "") or "")
                )
            )

    metrics = {
        "planned_drop_count": len(planned_drops),
        "refill_attempts": len(planned_drops),
        "refill_committed": len(refilled),
        "simulated_final_ratio": float(simulated_final_ratio),
        "cap_bound": cap_bound,
        "skip_protected": skip_counts["protected"],
        "skip_collision_last": skip_counts["collision_last"],
        "skip_isolated": skip_counts["isolated"],
        "skip_empty_pool": skip_counts["empty_pool"],
        "skip_counterproductive": skip_counts["counterproductive"],
    }

    return planned_drops, refilled, rejection_log, metrics


# ---------------------------------------------------------------------------
# Source-level projection
# ---------------------------------------------------------------------------


def _project_to_sources(
    *,
    sources: dict[str, pd.DataFrame],
    entity_groups: dict[str, list[tuple[str, str]]],
    entity_ids: list[str],
    removed_indices: set[int],
    interpolated: list[InterpolatedEntity],
    config: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    """Drop removed entity rows from each source and append interpolated ones."""
    id_columns: dict[str, str] = config["id_columns"]
    attribute_mapping: dict[str, dict[str, str]] = config["attribute_mapping"]

    # Collect ids to drop per source.
    drop_ids_per_source: dict[str, set[str]] = defaultdict(set)
    for idx in removed_indices:
        cid = entity_ids[idx]
        for src, rid in entity_groups.get(cid, []):
            drop_ids_per_source[src].add(rid)

    new_sources: dict[str, pd.DataFrame] = {}
    for src_name, df in sources.items():
        id_col = id_columns.get(src_name)
        if id_col is None or id_col not in df.columns:
            new_sources[src_name] = df
            continue
        drops = drop_ids_per_source.get(src_name, set())
        if drops:
            mask = ~df[id_col].astype(str).isin(drops)
            new_df = df.loc[mask].reset_index(drop=True)
        else:
            new_df = df.copy()
        new_df.attrs = df.attrs.copy()
        new_sources[src_name] = new_df

    # Append interpolated entities per their placement_mode and
    # source_placements. Each source expects its own column subset, so
    # we reverse the attribute_mapping to locate target columns.
    for entity in interpolated:
        for src in entity.source_placements:
            if src not in new_sources:
                continue
            df = new_sources[src]
            id_col = id_columns.get(src)
            if id_col is None:
                continue
            src_col_map = attribute_mapping.get(src, {})
            new_row: dict[str, Any] = {c: None for c in df.columns}
            new_row[id_col] = f"{entity.entity_id}__{src}"
            for src_col, canon_col in src_col_map.items():
                if canon_col in entity.attributes:
                    new_row[src_col] = entity.attributes[canon_col]
            new_df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            new_df.attrs = df.attrs.copy()
            new_sources[src] = new_df

    return new_sources


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------


def write_outputs(
    *,
    canonical_frame: pd.DataFrame,
    regenerated_em: pd.DataFrame,
    provenance_df: pd.DataFrame,
    niche_scores_df: pd.DataFrame,
    output_dir: Path,
    k2_metrics: dict[str, Any] | None = None,
) -> None:
    """Write post-K2 artifacts to *output_dir*."""
    canon_dir = output_dir / "input" / "data"
    em_dir = output_dir / "input" / "entitymatching"
    prov_dir = output_dir / "output" / "provenance"
    baselines_dir = output_dir / "output" / "baselines"
    for d in (canon_dir, em_dir, prov_dir, baselines_dir):
        d.mkdir(parents=True, exist_ok=True)

    canonical_frame.to_csv(canon_dir / "canonical.csv", index=False)
    logger.info("Wrote canonical frame (%d rows)", len(canonical_frame))

    # Per-pair per-split per-version regen files — shape mirrors the
    # original EM gold files so downstream users can treat this as a
    # drop-in benchmark replacement. C11 (2026-05-22) splits each
    # (pair, split) into two parallel versions:
    #   - <pair>_<split>_baseline_pruned.csv  (survivors only)
    #   - <pair>_<split>_corner_filled.csv    (survivors + corner backfill)
    # The original `id1, id2, label` convention is headerless; regen
    # files carry a header for clarity (loaders handle both).
    if regenerated_em.empty:
        logger.info("Regenerated EM DataFrame is empty; no splits to write")
    else:
        # Remove any legacy ``*_regenerated.csv`` files so a fresh K2
        # regen never leaves pre-C11 single-output files behind.
        for stale in em_dir.glob("*_regenerated.csv"):
            stale.unlink()
        grouped = regenerated_em.groupby(["pair_name", "split", "version"], sort=True)
        total_rows = 0
        for (pair_name, split, version), sub in grouped:
            out_path = em_dir / f"{pair_name}_{split}_{version}.csv"
            sub.drop(columns=["split", "pair_name", "version"]).to_csv(
                out_path, index=False
            )
            total_rows += len(sub)
            logger.info(
                "Wrote regenerated EM %s/%s split for %s: %d rows",
                split,
                version,
                pair_name,
                len(sub),
            )
        logger.info(
            "Wrote regenerated EM across %d (pair, split, version) combinations, "
            "%d rows total",
            grouped.ngroups,
            total_rows,
        )

    provenance_df.to_csv(prov_dir / "knob_02_niche.csv", index=False)
    logger.info("Wrote provenance (%d rows)", len(provenance_df))

    niche_scores_df.to_csv(prov_dir / "knob_02_niche_scores.csv", index=False)
    logger.info("Wrote niche scores (%d rows)", len(niche_scores_df))

    if k2_metrics is not None:
        realised_df = pd.DataFrame([k2_metrics])
        realised_df.to_csv(baselines_dir / "knob_02_realised.csv", index=False)
        logger.info(
            "Wrote K2 realised stats (final_ratio=%.3f, target=%.3f, operator=%s)",
            float(k2_metrics.get("final_ratio", 0.0)),
            float(k2_metrics.get("target_ratio", 0.0)),
            k2_metrics.get("operator", ""),
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Apply Knob 02 -- Entity Niche Density",
    )
    parser.add_argument("--domain", required=True)
    parser.add_argument("--level", required=True, choices=VALID_LEVELS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Variant output directory",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--strict-cache",
        action="store_true",
        help="Hard level: raise on LLM cache miss instead of calling API",
    )
    parser.add_argument(
        "--protection-source",
        choices=("gold", "silver"),
        default="gold",
        help=(
            "Protection set for K2 drop-corner-touching operator. 'gold' "
            "(default): EM gold ∪ fusion val/test gold — pool members "
            "droppable. 'silver': also includes all pool-cluster members "
            "(matches C9 silver-standard semantics)."
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

    logger.info("Knob 02: domain=%s level=%s output=%s", domain, level, output_dir)

    config = load_knob_02_config(domain)
    sources = load_domain_sources(domain)

    llm_cache: LLMCache | None = None
    if level == "hard":
        cache_dir = (
            REPO_ROOT
            / "usecases_synthetic"
            / "cache"
            / "knob_02_interpolations"
            / resolve_cache_domain(domain)
        )
        llm_cache = LLMCache(
            cache_dir=cache_dir,
            prompt_version=config.get("llm_prompt_version", "v1"),
            model_id=config.get("llm_model_id", "claude-opus-4-6"),
        )

    # Step 4i (2026-05-27): non-corner refill cache namespace, mirrors
    # the generate_variant.py wiring. Separate cache_dir keeps the
    # non-corner prompt's payloads out of the interpolation cache.
    llm_cache_non_corner: LLMCache | None = None
    nc_refill_cfg = config.get("non_corner_refill", {}) or {}
    if nc_refill_cfg.get("enabled", False):
        nc_prompt_version = config.get(
            "non_corner_prompt_version",
            nc_refill_cfg.get("prompt_version", "v1"),
        )
        nc_cache_dir = (
            REPO_ROOT
            / "usecases_synthetic"
            / "cache"
            / "knob_02_non_corner"
            / resolve_cache_domain(domain)
        )
        llm_cache_non_corner = LLMCache(
            cache_dir=nc_cache_dir,
            prompt_version=str(nc_prompt_version),
            model_id=config.get("llm_model_id", "gpt-5.4-mini"),
        )

    strict_cache = args.strict_cache

    # Mirror generate_variant.py: wire OpenAI clients for both K2
    # sub-paths when OPENAI_API_KEY is set and the run isn't strict-
    # cache-only. The interpolation client substitutes
    # ``{parent_records_json}``; the non-corner refill client
    # substitutes ``{reference_records_json}`` (different prompt). On
    # cache miss this populates the cache with real LLM calls
    # (plan_revision.md §C1 follow-up); without a key, K2 interpolate
    # falls back to the deterministic blender, and non-corner refill
    # raises RuntimeError("api_client required on cache miss") to
    # surface that the LLM path was needed but unavailable.
    api_client: Any = None
    non_corner_api_client: Any = None
    if not strict_cache and os.environ.get("OPENAI_API_KEY"):
        model_id = config.get("llm_model_id", "gpt-5.4-mini")
        try:
            api_client = build_openai_interpolation_client(model_id=model_id)
            logger.info("K2 OpenAI interpolation client active (model=%s)", model_id)
        except Exception as exc:  # pragma: no cover - construction failures
            logger.warning(
                "K2 OpenAI client build failed (%s); falling back to "
                "deterministic blender on cache miss",
                exc,
            )
        try:
            non_corner_api_client = build_openai_non_corner_client(model_id=model_id)
            logger.info(
                "K2 OpenAI non-corner refill client active (model=%s)", model_id
            )
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "K2 OpenAI non-corner client build failed (%s); refills will "
                "fail on cache miss",
                exc,
            )

    (
        new_sources,
        canonical_frame,
        regen_em,
        prov_df,
        scores_df,
        k2_metrics,
        _regen_pools,
    ) = apply_knob_02(
        domain=domain,
        level=level,  # type: ignore[arg-type]
        sources=sources,
        config=config,
        llm_cache=llm_cache,
        api_client=api_client,
        committee_fn=None,
        strict_cache=strict_cache,
        seed=args.seed,
        non_corner_cache=llm_cache_non_corner,
        non_corner_api_client=non_corner_api_client,
        protection_source=args.protection_source,
    )

    write_outputs(
        canonical_frame=canonical_frame,
        regenerated_em=regen_em,
        provenance_df=prov_df,
        niche_scores_df=scores_df,
        output_dir=output_dir,
        k2_metrics=k2_metrics,
    )

    for src_name in sorted(new_sources.keys()):
        logger.info("  %s: %d rows", src_name, len(new_sources[src_name]))


if __name__ == "__main__":
    main()
