#!/usr/bin/env python3
"""Apply Knob 04 — Per-entity Source Coverage Skew.

Shifts the per-entity source coverage histogram toward a per-level
target. Hard removes entity rows (long-tail), easy fabricates new rows
(uniform coverage), medium is identity.

See ``knobs/knob_04_coverage_skew.md`` for the full specification.

Usage
-----
::

    python usecases_synthetic/scripts/apply_knob_04_coverage.py \\
        --domain companies --level hard --seed 42

Inputs
------
- Source DataFrames from ``usecases/<domain>/input/data/``
- EM correspondences from ``usecases/<domain>/input/entitymatching/``
- Fusion gold from ``usecases/<domain>/input/fusion/``
- Pooled positives from ``usecases_synthetic/pools/<domain>/``
- Per-domain config at ``usecases_synthetic/config/knob_04_coverage/<domain>.yaml``

Outputs (under *output_dir*)
----------------------------
- Mutated source DataFrames (returned in-memory)
- Baseline CSV at ``<output_dir>/output/baselines/knob_04_baseline_coverage.csv``
- Realised vs target CSV at
  ``<output_dir>/output/baselines/knob_04_realized_vs_target.csv``
- Provenance CSV at ``<output_dir>/output/provenance/knob_04_coverage_skew.csv``
- Skipped-entity audit at ``<output_dir>/output/provenance/knob_04_skipped.csv``
"""

from __future__ import annotations

import argparse
import logging
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure repo root is on sys.path.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from usecases_synthetic.lib.coverage_ops import (
    EntityView,
    RemovalConstraints,
    apply_singleton_cap_rollback,
    build_entity_view,
    fabricate_row_by_paraphrase,
    histogram_to_dataframe,
    measure_coverage_histogram,
    plan_demotions,
    plan_promotions,
    select_fabrication_candidates,
    select_removal_candidates,
    validate_target_histogram,
)
from usecases_synthetic.lib.domain_config import (
    CONFIG_DIR,
    VALID_LEVELS,
    DomainConfig,
    load_domain_config,
    load_knob_config,
)
from usecases_synthetic.lib.loaders import load_domain_sources, read_em_gold_csv
from usecases_synthetic.lib.protection import POOLS_DIR
from usecases_synthetic.lib.provenance import PROVENANCE_COLUMNS, ProvenanceLog
from usecases_synthetic.lib.rng import make_rng, spawn_sub_rng

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EM linkage (reused shape from K3 but inlined to avoid a cross-script dep)
# ---------------------------------------------------------------------------


@dataclass
class EntityLinkage:
    """Cross-source entity linkage built from EM correspondences.

    Parameters
    ----------
    groups : dict[str, list[tuple[str, str]]]
        ``group_id -> [(source, record_id)]`` for multi-source groups.
    index : dict[str, str]
        ``record_id -> group_id`` reverse lookup.
    """

    groups: dict[str, list[tuple[str, str]]] = field(default_factory=dict)
    index: dict[str, str] = field(default_factory=dict)


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


def build_entity_linkage(
    domain_config: DomainConfig,
    id_columns: dict[str, str],
    sources: dict[str, pd.DataFrame],
) -> EntityLinkage:
    """Build entity linkage from pooled positives unioned with EM gold.

    The EM gold in ``usecases/<domain>/input/entitymatching/`` is a
    sampled subset of true matches, not a complete enumeration
    (see plan.md Step 4). Using it alone dramatically undercounts the
    set of matchable entities and inflates the singleton fraction of
    the coverage histogram on domains with many pooled-but-unsampled
    matches (e.g. companies measured ``H_base[1] = 0.919``). This knob
    already treats the pool as authoritative for protection in
    :func:`_load_pool_pairs`; the linkage builder now does the same.

    Loads, in order:

    1. ``usecases_synthetic/pools/<domain>/pooled_positives.csv`` (the
       pooled-positive artifact from plan.md Step 4, if present).
    2. ``usecases/<domain>/input/entitymatching/*_all.csv`` (the hand
       curated EM gold), falling back to per-split files when ``_all``
       is absent.

    Both sources are unioned and fed through union-find to produce the
    connected-component entity groups.
    """
    pairs: list[tuple[str, str]] = []

    # Pooled positives (authoritative for matchable-entity linkage).
    pool_path = POOLS_DIR / domain_config.domain / "pooled_positives.csv"
    if pool_path.exists():
        pool_df = pd.read_csv(pool_path)
        for _, row in pool_df.iterrows():
            pairs.append((str(row["id1"]), str(row["id2"])))

    # EM gold positives (sampled subset; unioned in).
    em_dir = domain_config.em_dir()
    if em_dir.exists():
        all_csvs = sorted(em_dir.glob("*_all.csv"))
        if all_csvs:
            csvs_to_read = all_csvs
        else:
            csvs_to_read = []
            for suffix in ("_train.csv", "_val.csv", "_test.csv"):
                csvs_to_read.extend(sorted(em_dir.glob(f"*{suffix}")))
        for csv_path in csvs_to_read:
            df = read_em_gold_csv(csv_path)
            positives = df[df["label"].astype(str).str.lower() == "true"]
            for _, row in positives.iterrows():
                pairs.append((str(row["id1"]), str(row["id2"])))

    if not pairs:
        return EntityLinkage()

    all_ids: set[str] = set()
    id_to_source: dict[str, str] = {}
    for id1, id2 in pairs:
        all_ids.add(id1)
        all_ids.add(id2)
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
        groups_raw[_find(parent, rid)].append(rid)

    groups: dict[str, list[tuple[str, str]]] = {}
    index: dict[str, str] = {}
    for root, members in groups_raw.items():
        if len(members) < 2:
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


# ---------------------------------------------------------------------------
# Auxiliary: fusion gold ids, pool pairs, protected records
# ---------------------------------------------------------------------------


def _load_fusion_gold_ids(domain_config: DomainConfig) -> set[str]:
    """Load entity IDs from the fusion validation + test set XMLs.

    Reads both fusion files declared by the domain config's
    ``fusion_files`` block (defaults: ``validation_set.xml`` and
    ``test_set.xml``) so that fusion-validation entities are protected
    from K4 demotions / fabrications alongside fusion-test entities.
    Mirrors the ``protection._load_fusion_gold_ids`` semantics shared
    with K2.
    """
    # Delegate to the shared protection loader so XML (pre-2026 domains)
    # and JSONL-by-DOI fusion gold (papers; mapped to per-DOI anchor source
    # ids) are handled identically and in one place.
    from usecases_synthetic.lib.protection import _load_fusion_protected_ids

    return _load_fusion_protected_ids(domain_config.domain)


def _load_pool_pairs(
    domain: str,
    id_columns: dict[str, str],
    sources: dict[str, pd.DataFrame],
) -> list[tuple[tuple[str, str], tuple[str, str]]]:
    """Load pooled-positive pairs as ``((source, rid), (source, rid))``.

    Unions the EM train/val/test positives with the pool artifact.
    Records whose source cannot be resolved from any source DataFrame
    are skipped.
    """
    record_to_source: dict[str, str] = {}
    for source_name, df in sources.items():
        id_col = id_columns.get(source_name)
        if id_col and id_col in df.columns:
            for rid in df[id_col].astype(str):
                record_to_source[rid] = source_name

    pairs: list[tuple[tuple[str, str], tuple[str, str]]] = []
    seen: set[tuple[tuple[str, str], tuple[str, str]]] = set()

    def _add(left_id: str, right_id: str) -> None:
        ls = record_to_source.get(left_id)
        rs = record_to_source.get(right_id)
        if ls is None or rs is None:
            return
        left = (ls, left_id)
        right = (rs, right_id)
        if left > right:
            left, right = right, left
        key = (left, right)
        if key in seen:
            return
        seen.add(key)
        pairs.append(key)

    # EM gold positives (all splits).
    from usecases_synthetic.lib.domain_config import (
        USECASES_DIR,
        data_root_for_domain,
    )

    em_dir = (
        (data_root_for_domain(domain) or USECASES_DIR)
        / domain
        / "input"
        / "entitymatching"
    )
    if em_dir.exists():
        for csv_path in sorted(em_dir.glob("*.csv")):
            try:
                df = read_em_gold_csv(csv_path)
            except Exception:
                continue
            if "label" not in df.columns:
                continue
            positives = df[df["label"].astype(str).str.lower() == "true"]
            for _, row in positives.iterrows():
                _add(str(row["id1"]), str(row["id2"]))

    # Pool artifact.
    pool_path = POOLS_DIR / domain / "pooled_positives.csv"
    if pool_path.exists():
        pool_df = pd.read_csv(pool_path)
        for _, row in pool_df.iterrows():
            _add(str(row["id1"]), str(row["id2"]))

    return pairs


def _build_protected_records(
    pool_pairs: list[tuple[tuple[str, str], tuple[str, str]]],
) -> set[tuple[str, str]]:
    """Flatten pool pairs into a set of protected endpoint records."""
    protected: set[tuple[str, str]] = set()
    for left, right in pool_pairs:
        protected.add(left)
        protected.add(right)
    return protected


# ---------------------------------------------------------------------------
# Paraphrase callable (K1 fallback for easy / within-source duplicates)
# ---------------------------------------------------------------------------


ParaphraseFn = Callable[[str, str, np.random.Generator], tuple[str, dict[str, Any]]]


def build_paraphrase_fn(
    domain: str,
    target_source: str,
    k1_config: dict[str, Any] | None,
) -> ParaphraseFn:
    """Return a single-cell paraphrase callable backed by Knob 01.

    Wraps :func:`usecases_synthetic.lib.surface_operators.paraphrase_value_for_knob_04`
    with the K1 per-domain config and looks up the attribute class per
    column from ``k1_config``. If ``k1_config`` is ``None``, returns a
    deterministic identity callable (no-op paraphrase).

    Parameters
    ----------
    domain : str
        Domain name (forwarded to ``paraphrase_value_for_knob_04``).
    target_source : str
        Target source name (used to look up the column's attribute
        class in K1's ``attribute_classes`` block).
    k1_config : dict or None
        Parsed K1 config YAML; pass ``None`` to disable paraphrase.
    """
    if k1_config is None:

        def _identity(
            col: str, value: str, rng: np.random.Generator
        ) -> tuple[str, dict[str, Any]]:
            del col, rng
            return value, {"transform_fn": "passthrough"}

        return _identity

    from usecases_synthetic.lib.surface_operators import (
        paraphrase_value_for_knob_04,
    )

    attribute_classes: dict[str, dict[str, str]] = k1_config.get(
        "attribute_classes", {}
    )
    source_classes = attribute_classes.get(target_source, {})

    def _wrapped(
        col: str, value: str, rng: np.random.Generator
    ) -> tuple[str, dict[str, Any]]:
        cls = source_classes.get(col, "secondary")
        return paraphrase_value_for_knob_04(
            domain=domain,
            attribute_class=cls,  # type: ignore[arg-type]
            original_value=value,
            config=k1_config,
            rng=rng,
        )

    return _wrapped


def _load_k1_config_safe(domain: str) -> dict[str, Any] | None:
    """Load the K1 config if it exists; return ``None`` otherwise."""
    try:
        return load_knob_config(1, domain)
    except FileNotFoundError:
        return None


# ---------------------------------------------------------------------------
# Row mutation helpers
# ---------------------------------------------------------------------------


def _remove_row(sources: dict[str, pd.DataFrame], source: str, row_idx: int) -> None:
    """Drop a row from a source DataFrame (index-based)."""
    df = sources[source]
    if row_idx in df.index:
        sources[source] = df.drop(index=row_idx)
        sources[source].attrs = df.attrs.copy()


def _new_synthetic_record_id(entity_id: str, target_source: str, kind: str) -> str:
    """Generate a deterministic synthetic record id."""
    safe_ent = entity_id.replace("/", "_").replace(":", "_")
    return f"k04__{kind}__{target_source}__{safe_ent}"


def _append_row(
    sources: dict[str, pd.DataFrame],
    source: str,
    row: pd.Series,
) -> int:
    """Append a row to a source DataFrame. Returns new row index."""
    df = sources[source]
    attrs_copy = df.attrs.copy()
    new_idx = int(df.index.max()) + 1 if len(df) > 0 else 0
    new_row = row.reindex(df.columns)
    new_df = pd.concat(
        [df, pd.DataFrame([new_row.values], columns=df.columns, index=[new_idx])],
        ignore_index=False,
    )
    new_df.attrs = attrs_copy
    sources[source] = new_df
    return new_idx


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------


def apply_knob_04(
    domain: str,
    level: Literal["easy", "medium", "hard"],
    sources: dict[str, pd.DataFrame],
    config: dict[str, Any],
    linkage: EntityLinkage | None = None,
    fusion_gold_ids: set[str] | None = None,
    pool_pairs: list[tuple[tuple[str, str], tuple[str, str]]] | None = None,
    distractor_entity_ids: set[str] | None = None,
    seed: int = 42,
    k1_config: dict[str, Any] | None = None,
    paraphrase_fn_factory: Callable[[str], ParaphraseFn] | None = None,
) -> tuple[
    dict[str, pd.DataFrame],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Apply Knob 04 coverage skew at the given level.

    Parameters
    ----------
    domain : str
        Domain name.
    level : {"easy", "medium", "hard"}
    sources : dict[str, DataFrame]
        Source DataFrames. Mutated in place (and returned).
    config : dict
        Parsed K4 config YAML.
    linkage : EntityLinkage or None
        EM linkage for multi-source groups. If None, only singletons are
        modelled (tests can inject a linkage directly).
    fusion_gold_ids : set[str] or None
        Record IDs present in the fusion gold test set.
    pool_pairs : list of ((source, rid), (source, rid)) or None
        Pool-protected match edges. If None, an empty list is used.
    distractor_entity_ids : set[str] or None
        Entity ids flagged as K2 distractors — never eligible for
        removal or fabrication.
    seed : int
        Master RNG seed.
    k1_config : dict or None
        K1 config used by the paraphrase fallback. If None and
        ``paraphrase_fn_factory`` is None, the fallback becomes an
        identity passthrough.
    paraphrase_fn_factory : callable or None
        ``target_source -> paraphrase_fn``. Overrides the default K1
        factory (useful for tests).

    Returns
    -------
    sources : dict[str, DataFrame]
        Mutated source DataFrames.
    provenance_df : DataFrame
        Provenance log (fabrications, removals, within-source duplicates).
    skipped_df : DataFrame
        Skipped-entity audit log.
    histograms_df : DataFrame
        Long-form baseline + realised + target histograms.
    """
    if level not in VALID_LEVELS:
        raise ValueError(f"Invalid level: {level!r}. Valid: {VALID_LEVELS}")

    if linkage is None:
        linkage = EntityLinkage()
    if fusion_gold_ids is None:
        fusion_gold_ids = set()
    if pool_pairs is None:
        pool_pairs = []
    if distractor_entity_ids is None:
        distractor_entity_ids = set()

    id_columns: dict[str, str] = config["id_columns"]
    primary_columns: dict[str, str] = config.get("primary_columns", {})
    source_count: int = int(config["source_count"])

    target_spec: dict[str, Any] = config["target_coverage_histogram"]

    # --- Step 1: Build entity view and measure baseline ---
    view = build_entity_view(linkage.groups, sources, id_columns, source_count)
    base_hist = measure_coverage_histogram(view)
    # Denominator must match the histogram's population: exclude
    # synthetic distractor singletons that measure_coverage_histogram
    # drops by default. Without this, plan_demotions / plan_promotions
    # multiply the matchable-entity fractions by the full-corpus count
    # and overshoot every demotion/promotion target.
    total_entities = len(view) - len(view.singleton_source)

    # Medium is identity for the target histogram.
    if level == "medium":
        target_hist = dict(base_hist)
    else:
        raw_target = target_spec.get(level)
        if raw_target is None:
            raise ValueError(
                f"K4 config for domain={config.get('domain')} is missing "
                f"target_coverage_histogram.{level}"
            )
        target_hist = {int(k): float(v) for k, v in raw_target.items()}
        validate_target_histogram(target_hist, source_count)

    # --- Cross-level monotonicity check at config load (endpoints only) ---
    _validate_config_monotonicity(target_spec, base_hist, source_count)

    prov_log = ProvenanceLog(knob=4, level=level)
    skipped_log = ProvenanceLog(knob=4, level=level)

    # --- Step 2: Build protected-record set ---
    # Pool-pair endpoints are NOT blanket-protected anymore (per the K4
    # sign-off Pending #5 wire-up, 2026-05-07): the orphan check inside
    # `_would_break_pool_edge` enforces the spec semantic that single-
    # endpoint removal is allowed and only both-endpoint removal of the
    # same pool pair is forbidden. Fusion val/test records are likewise
    # NOT blanket-protected — `score_target_distance` chooses which
    # source survives the demotion (closest-to-target stays alive).
    # ``protected_records`` stays as the explicit anchor-record surface
    # (reserved for K1/K2 anchors); empty by default.
    protected_records: set[tuple[str, str]] = set()

    # Closeness wiring for fusion val/test entities — used by
    # `score_target_distance` inside `select_removal_candidates`.
    # Loaded once per dispatcher pass; falls back to an empty dict if
    # the fusion XML files are unavailable (closeness ranking degrades
    # to "no targets known" -> all sources score 0.0 -> closeness rank
    # ties and the alphabetical tie-break determines order).
    fusion_targets: dict[str, dict[str, list[str]]] = {}
    try:
        from usecases_synthetic.lib.protection import load_fusion_target_values

        fusion_targets = load_fusion_target_values(domain)
    except Exception:  # noqa: BLE001 - fall back to empty targets on any failure
        fusion_targets = {}
    # K6's `fusion_protection_tolerance` block, if authored, overrides
    # the per-attribute tolerance defaults. K4 doesn't author this block;
    # we read K6's domain config when present.
    tol_overrides: dict[str, dict[str, float | str]] = {}
    try:
        import yaml as _yaml

        _k6_path = CONFIG_DIR / "knob_06_noise" / f"{domain}.yaml"
        if _k6_path.exists():
            with open(_k6_path, encoding="utf-8") as _f:
                _k6 = _yaml.safe_load(_f) or {}
            tol_overrides = _k6.get("fusion_protection_tolerance", {}) or {}
    except Exception:  # noqa: BLE001
        tol_overrides = {}
    attribute_mapping: dict[str, dict[str, str]] = config.get("attribute_mapping", {})

    # --- Step 3: Plan removals / fabrications ---
    if level == "hard":
        rng = make_rng(domain, variant=level, knob=4, master_seed=seed)
        demotions = plan_demotions(base_hist, target_hist, total_entities, source_count)
        logger.info(
            "K4 hard demotions plan (bin -> count): %s",
            {k: v for k, v in demotions.items() if v > 0},
        )
        constraints = RemovalConstraints(
            fusion_val_test_ids=set(fusion_gold_ids),
            protected_records=protected_records,
            distractor_entity_ids=set(distractor_entity_ids),
            singleton_cap=float(config.get("singleton_cap_hard", 0.70)),
        )
        selected = select_removal_candidates(
            view=view,
            demotions=demotions,
            constraints=constraints,
            pool_pairs=pool_pairs,
            sources=sources,
            primary_cols=primary_columns,
            rng=rng,
            fusion_targets=fusion_targets,
            attribute_mapping=attribute_mapping,
            domain=domain,
            tol_overrides=tol_overrides,
            per_source_demotion_cap=float(config.get("per_source_demotion_cap", 1.0)),
        )
        # Singleton cap rollback.
        accepted, rolled_back = apply_singleton_cap_rollback(
            view=view,
            selected=selected,
            cap=constraints.singleton_cap,
        )
        for entity_id, source, record_id in rolled_back:
            skipped_log.append(
                entity_id=entity_id,
                source=source,
                attribute="__row__",
                transform_fn="remove_entity_row",
                transform_params={
                    "reason": "singleton_cap",
                    "record_id": record_id,
                },
            )

        # Within-source duplicates (hard only).
        within_rate = float(
            config.get("within_source_duplicate_rate", {}).get("hard", 0.0)
        )
        duplicate_plan = _plan_within_source_duplicates(
            view=view,
            accepted_removals=accepted,
            rate=within_rate,
            rng=rng,
        )

        # Apply removals.
        _apply_removals(
            sources=sources,
            view=view,
            removals=accepted,
            prov_log=prov_log,
            reason="singleton_target",
        )

        # Apply within-source duplicates.
        paraphrase_factory = paraphrase_fn_factory or (
            lambda src: build_paraphrase_fn(domain, src, k1_config)
        )
        _apply_within_source_duplicates(
            sources=sources,
            view=view,
            plan=duplicate_plan,
            paraphrase_factory=paraphrase_factory,
            k1_config=k1_config,
            prov_log=prov_log,
            rng=rng,
        )

    elif level == "easy":
        rng = make_rng(domain, variant=level, knob=4, master_seed=seed)
        promotions = plan_promotions(
            base_hist, target_hist, total_entities, source_count
        )
        logger.info(
            "K4 easy promotions plan (bin -> count): %s",
            {k: v for k, v in promotions.items() if v > 0},
        )
        selected = select_fabrication_candidates(
            view=view,
            promotions=promotions,
            rng=rng,
        )
        paraphrase_factory = paraphrase_fn_factory or (
            lambda src: build_paraphrase_fn(domain, src, k1_config)
        )
        _apply_fabrications(
            sources=sources,
            view=view,
            fabrications=selected,
            id_columns=id_columns,
            paraphrase_factory=paraphrase_factory,
            k1_config=k1_config,
            prov_log=prov_log,
            rng=rng,
        )

    else:  # medium
        logger.info("K4 medium: identity (no rows added or removed)")

    # --- Step 4: Measure realised histogram from the mutated view ---
    # We read from the in-memory view rather than re-building from the
    # mutated sources: rebuilding would interpret K4-fabricated rows (new
    # synthetic record ids) as fresh singletons because they are not
    # present in ``linkage.groups``. The in-memory view has been updated
    # in place with correct per-entity source coverage, so it is the
    # authoritative realised state.
    realised_hist = measure_coverage_histogram(view)

    # --- Step 5: Build output DataFrames ---
    baseline_df = histogram_to_dataframe(base_hist, label="baseline")
    target_df = histogram_to_dataframe(target_hist, label=f"target_{level}")
    realised_df = histogram_to_dataframe(realised_hist, label=f"realised_{level}")
    histograms_df = pd.concat([baseline_df, target_df, realised_df], ignore_index=True)

    provenance_df = _log_to_dataframe(prov_log)
    skipped_df = _log_to_dataframe(skipped_log)

    return sources, provenance_df, skipped_df, histograms_df


def _log_to_dataframe(log: ProvenanceLog) -> pd.DataFrame:
    """Flatten a :class:`ProvenanceLog` to a DataFrame."""
    if len(log) == 0:
        return pd.DataFrame(columns=PROVENANCE_COLUMNS)
    return pd.DataFrame(
        [row.as_dict() for row in log._rows],
        columns=PROVENANCE_COLUMNS,
    )


def _validate_config_monotonicity(
    target_spec: dict[str, Any],
    base_hist: dict[int, float],
    source_count: int,
) -> None:
    """Validate singleton-bin monotonicity across levels.

    Enforces ``H_target.easy[1] <= H_base[1] <= H_target.hard[1]`` (the
    medium target is identity-to-baseline so it lies on the boundary).
    Raises :class:`ValueError` on violation.
    """
    easy_spec = target_spec.get("easy")
    hard_spec = target_spec.get("hard")
    if easy_spec is not None and 1 in easy_spec:
        easy_h1 = float(easy_spec[1])
        if easy_h1 > base_hist.get(1, 0.0) + 1e-9:
            raise ValueError(
                f"K4 config: H_target.easy[1]={easy_h1:.3f} must be "
                f"<= H_base[1]={base_hist.get(1, 0.0):.3f}"
            )
    if hard_spec is not None and 1 in hard_spec:
        hard_h1 = float(hard_spec[1])
        if hard_h1 + 1e-9 < base_hist.get(1, 0.0):
            raise ValueError(
                f"K4 config: H_target.hard[1]={hard_h1:.3f} must be "
                f">= H_base[1]={base_hist.get(1, 0.0):.3f}"
            )


def _apply_removals(
    sources: dict[str, pd.DataFrame],
    view: EntityView,
    removals: list[tuple[str, str, str]],
    prov_log: ProvenanceLog,
    reason: str,
) -> None:
    """Remove rows from sources and log each removal."""
    # Apply in reverse order of (source, row_idx) so index drops don't
    # clobber later-index lookups — but since pandas indices are labels
    # not positions, we can just drop one at a time safely.
    for entity_id, source, record_id in removals:
        members = view.members.get(entity_id, {})
        if source not in members:
            continue
        row_idx, _ = members[source]
        _remove_row(sources, source, row_idx)
        # Update the view so subsequent lookups reflect the state.
        del view.members[entity_id][source]
        prov_log.append(
            entity_id=entity_id,
            source=source,
            attribute="__row__",
            original_value="row_exists",
            new_value="row_absent",
            transform_fn="remove_entity_row",
            transform_params={
                "reason": reason,
                "record_id": record_id,
                "conflict_preserved": True,
            },
        )


def _plan_within_source_duplicates(
    view: EntityView,
    accepted_removals: list[tuple[str, str, str]],
    rate: float,
    rng: np.random.Generator,
) -> list[tuple[str, str, str]]:
    """Select entities to emit within-source duplicate rows for.

    Chooses ``rate * n`` entities that still have ≥1 source after
    removals, and picks one of their surviving sources to duplicate.
    """
    if rate <= 0.0:
        return []
    # Simulate coverage after planned removals.
    removed_set: set[tuple[str, str]] = {(r[0], r[1]) for r in accepted_removals}
    eligible: list[tuple[str, str, str]] = []
    for entity_id, members in view.members.items():
        surviving = [
            (src, rid)
            for src, (_, rid) in members.items()
            if (entity_id, src) not in removed_set
        ]
        if not surviving:
            continue
        # Deterministic source choice: first in sorted order.
        surviving.sort()
        source, record_id = surviving[0]
        eligible.append((entity_id, source, record_id))
    if not eligible:
        return []
    n_select = int(round(rate * len(eligible)))
    if n_select <= 0:
        return []
    order = rng.permutation(len(eligible))
    return [eligible[i] for i in order[:n_select]]


def _apply_within_source_duplicates(
    sources: dict[str, pd.DataFrame],
    view: EntityView,
    plan: list[tuple[str, str, str]],
    paraphrase_factory: Callable[[str], ParaphraseFn],
    k1_config: dict[str, Any] | None,
    prov_log: ProvenanceLog,
    rng: np.random.Generator,
) -> None:
    """Append paraphrased sibling-copy rows back into the same source."""
    if not plan:
        return
    managed_columns_by_source = _managed_columns_from_k1(k1_config)
    for entity_id, source, record_id in plan:
        df = sources[source]
        # Find the template row by its current record_id.
        template_members = view.members.get(entity_id, {})
        if source not in template_members:
            continue
        template_idx, _ = template_members[source]
        if template_idx not in df.index:
            continue
        template_row = df.loc[template_idx]

        sub_rng = spawn_sub_rng(rng, f"within_dup:{entity_id}:{source}")
        paraphrase_fn = paraphrase_factory(source)
        managed_columns = managed_columns_by_source.get(source, [])
        new_record_id = _new_synthetic_record_id(entity_id, source, kind="within_dup")
        id_col = None
        if k1_config is not None:
            id_col = (k1_config.get("id_columns") or {}).get(source)
        new_row, params = fabricate_row_by_paraphrase(
            sibling_row=template_row,
            target_source_columns=list(df.columns),
            managed_columns=list(managed_columns),
            paraphrase_fn=paraphrase_fn,
            rng=sub_rng,
            new_record_id=new_record_id,
            target_id_column=id_col,
        )
        new_idx = _append_row(sources, source, new_row)
        # Reflect in the view as an extra (not counted in coverage).
        view.extras.setdefault(entity_id, []).append((source, new_idx, new_record_id))
        prov_log.append(
            entity_id=entity_id,
            source=source,
            attribute="__row__",
            original_value="row_exists",
            new_value="row_exists",
            transform_fn="within_source_duplicate",
            transform_params={
                "sibling_row_id": record_id,
                "knob_01_paraphrase_params": params,
                "k4_fabricated": True,
            },
        )


def _apply_fabrications(
    sources: dict[str, pd.DataFrame],
    view: EntityView,
    fabrications: list[tuple[str, str]],
    id_columns: dict[str, str],
    paraphrase_factory: Callable[[str], ParaphraseFn],
    k1_config: dict[str, Any] | None,
    prov_log: ProvenanceLog,
    rng: np.random.Generator,
) -> None:
    """Fabricate and insert new rows via the paraphrase fallback path."""
    managed_columns_by_source = _managed_columns_from_k1(k1_config)
    for entity_id, target_source in fabrications:
        members = view.members.get(entity_id, {})
        if not members:
            continue
        if target_source in members:
            continue
        # Pick a sibling source (deterministic: sorted alphabetically).
        sibling_source = sorted(members.keys())[0]
        sibling_idx, sibling_rid = members[sibling_source]
        sibling_df = sources[sibling_source]
        if sibling_idx not in sibling_df.index:
            continue
        sibling_row = sibling_df.loc[sibling_idx]

        target_df = sources[target_source]
        target_columns = list(target_df.columns)
        managed_columns = managed_columns_by_source.get(target_source, [])

        sub_rng = spawn_sub_rng(rng, f"fabricate:{entity_id}:{target_source}")
        paraphrase_fn = paraphrase_factory(target_source)
        new_record_id = _new_synthetic_record_id(entity_id, target_source, kind="fab")
        target_id_col = id_columns.get(target_source)
        new_row, params = fabricate_row_by_paraphrase(
            sibling_row=sibling_row,
            target_source_columns=target_columns,
            managed_columns=list(managed_columns),
            paraphrase_fn=paraphrase_fn,
            rng=sub_rng,
            new_record_id=new_record_id,
            target_id_column=target_id_col,
        )
        new_idx = _append_row(sources, target_source, new_row)
        # Update view so subsequent selections see the new coverage.
        view.members[entity_id][target_source] = (new_idx, new_record_id)
        prov_log.append(
            entity_id=entity_id,
            source=target_source,
            attribute="__row__",
            original_value="row_absent",
            new_value="row_exists",
            transform_fn="propagate_and_paraphrase",
            transform_params={
                "template_source": sibling_source,
                "sibling_source": sibling_source,
                "sibling_row_id": sibling_rid,
                "knob_01_paraphrase_params": params,
                "k4_fabricated": True,
            },
        )


def _managed_columns_from_k1(
    k1_config: dict[str, Any] | None,
) -> dict[str, list[str]]:
    """Extract per-source managed columns from a K1 config."""
    if k1_config is None:
        return {}
    attr_classes: dict[str, dict[str, str]] = k1_config.get("attribute_classes", {})
    return {src: list(cols.keys()) for src, cols in attr_classes.items()}


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------


def write_outputs(
    histograms_df: pd.DataFrame,
    provenance_df: pd.DataFrame,
    skipped_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Write K4 artifacts to disk.

    Parameters
    ----------
    histograms_df : DataFrame
        Long-form baseline + realised + target histograms.
    provenance_df : DataFrame
        Provenance log.
    skipped_df : DataFrame
        Skipped-entity audit.
    output_dir : Path
        Variant directory root.
    """
    baselines_dir = output_dir / "output" / "baselines"
    baselines_dir.mkdir(parents=True, exist_ok=True)

    baseline_rows = histograms_df[histograms_df["label"] == "baseline"]
    baseline_rows.to_csv(baselines_dir / "knob_04_baseline_coverage.csv", index=False)
    histograms_df.to_csv(baselines_dir / "knob_04_realized_vs_target.csv", index=False)

    prov_dir = output_dir / "output" / "provenance"
    prov_dir.mkdir(parents=True, exist_ok=True)
    provenance_df.to_csv(prov_dir / "knob_04_coverage_skew.csv", index=False)
    skipped_df.to_csv(prov_dir / "knob_04_skipped.csv", index=False)

    logger.info(
        "Wrote K4 outputs: histograms=%d rows, provenance=%d rows, skipped=%d rows",
        len(histograms_df),
        len(provenance_df),
        len(skipped_df),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Apply Knob 04 — Per-entity Source Coverage Skew",
    )
    parser.add_argument("--domain", required=True, help="Domain name")
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
        "Knob 04: domain=%s level=%s seed=%d output=%s",
        domain,
        level,
        seed,
        output_dir,
    )

    config = load_knob_config(4, domain)
    domain_config = load_domain_config(domain)
    sources = load_domain_sources(domain)

    id_columns: dict[str, str] = config["id_columns"]
    linkage = build_entity_linkage(domain_config, id_columns, sources)
    fusion_gold_ids = _load_fusion_gold_ids(domain_config)
    pool_pairs = _load_pool_pairs(domain, id_columns, sources)
    k1_config = _load_k1_config_safe(domain)

    logger.info(
        "K4 inputs: %d multi-source groups, %d fusion gold ids, %d pool pairs",
        len(linkage.groups),
        len(fusion_gold_ids),
        len(pool_pairs),
    )

    sources, provenance_df, skipped_df, histograms_df = apply_knob_04(
        domain=domain,
        level=level,
        sources=sources,
        config=config,
        linkage=linkage,
        fusion_gold_ids=fusion_gold_ids,
        pool_pairs=pool_pairs,
        seed=seed,
        k1_config=k1_config,
    )

    write_outputs(histograms_df, provenance_df, skipped_df, output_dir)

    for src_name, df in sources.items():
        logger.info("  %s: %d rows", src_name, len(df))


if __name__ == "__main__":
    main()
