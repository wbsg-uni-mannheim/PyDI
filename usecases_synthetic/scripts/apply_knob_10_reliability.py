#!/usr/bin/env python3
"""Apply Knob 10 — Source Reliability Differentiation.

Pure permutation reshuffling of which source carries the gold-aligned
variant per (entity, attribute) cell.  No gold mutation — fusion gold is
byte-identical before and after.  Controls trust ambiguity for the fusion
stage.

See ``knobs/knob_10_source_reliability.md`` for the full specification.

Usage
-----
::

    python usecases_synthetic/scripts/apply_knob_10_reliability.py \\
        --domain companies --level easy

Inputs
------
- Source DataFrames from ``usecases/<domain>/input/data/``
- Fusion gold from ``usecases/<domain>/input/fusion/{validation_set,test_set}.xml``
  (both files are read; their union is the protected fusion universe per
  the §"Terminology convention" pass in plan_s1_scale.md)
- Per-domain K10 config at ``usecases_synthetic/config/knob_10_reliability/<domain>.yaml``
- Per-attribute kind map from
  :data:`usecases_synthetic.lib.protection._DEFAULT_KIND_BY_DOMAIN_ATTR`
  (Pending #5 strict + infra-aligned wire-up, 2026-05-07)

Outputs (under *output_dir*)
------
- Mutated source DataFrames (returned in-memory; not written to disk by
  default — the orchestrator writes them)
- Provenance CSV at ``<output_dir>/output/provenance/knob_10_reliability.csv``
- Compromised-mask CSV at ``<output_dir>/output/provenance/knob_10_compromised_mask.csv``
- Baseline alignment CSV at ``<output_dir>/output/baselines/knob_10_baseline_alignment.csv``
- Gold-hash sentinel at ``<output_dir>/output/baselines/knob_10_gold_hash.txt``
"""

from __future__ import annotations

import argparse
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

from usecases_synthetic.lib.domain_config import (
    CONFIG_DIR,
    USECASES_DIR,
    VALID_LEVELS,
    load_domain_config,
)
from usecases_synthetic.lib.loaders import load_domain_sources
from usecases_synthetic.lib.provenance import PROVENANCE_COLUMNS, ProvenanceLog
from usecases_synthetic.lib.reliability import (
    assert_multiset_invariant,
    build_entity_linkage,
    generate_compromised_mask,
    identify_per_attribute_winner,
    identify_reshufflable_cells,
    load_fusion_gold,
    measure_gold_alignment,
    reshuffle_cells,
    resolve_attribute_kinds,
    sha256_file,
)
from usecases_synthetic.lib.rng import make_rng

logger = logging.getLogger(__name__)


# ---- Config loading & validation ------------------------------------------


def load_knob_10_config(domain: str) -> dict[str, Any]:
    """Load the Knob 10 reliability config for *domain*."""
    path = CONFIG_DIR / "knob_10_reliability" / f"{domain}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Knob 10 reliability config not found: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def fusion_protected_paths(domain: str) -> list[Path]:
    """Return the list of fusion XML paths whose union is the protected set.

    Reads both fusion files declared by the domain config's
    ``fusion_files`` block (defaults: ``validation_set.xml`` and
    ``test_set.xml``) under ``usecases/<domain>/input/fusion/``. Either
    file may be absent (the other still defines a partial protected set);
    the caller skips non-existent paths via :func:`load_fusion_gold` per
    file.

    Per §"Terminology convention" in plan_s1_scale.md: both fusion val and
    test entities are protected at every value- and entity-mutating knob,
    K10 included.
    """
    domain_config = load_domain_config(domain)
    return domain_config.fusion_paths()


def validate_config(
    config: dict[str, Any],
    alignment_df: pd.DataFrame,
) -> None:
    """Validate Knob 10 config: sums, monotonicity, winner presence.

    Raises
    ------
    ValueError
        On any validation failure.
    """
    attr_targets = config["attribute_targets"]
    compromise_rates = config["compromise_rate_per_level"]
    corr_strengths = config["corr_strength_per_level"]

    # Monotonicity: compromise_rate easy <= medium <= hard
    cr_vals = [compromise_rates[lvl] for lvl in VALID_LEVELS]
    if not (cr_vals[0] <= cr_vals[1] <= cr_vals[2]):
        raise ValueError(
            f"compromise_rate not monotone: {dict(zip(VALID_LEVELS, cr_vals))}"
        )

    # Monotonicity: corr_strength easy <= medium <= hard
    cs_vals = [corr_strengths[lvl] for lvl in VALID_LEVELS]
    if not (cs_vals[0] <= cs_vals[1] <= cs_vals[2]):
        raise ValueError(
            f"corr_strength not monotone: {dict(zip(VALID_LEVELS, cs_vals))}"
        )

    # Per-attribute validation
    winners = identify_per_attribute_winner(alignment_df)

    for attr, level_dists in attr_targets.items():
        for level in VALID_LEVELS:
            dist = level_dists.get(level, {})
            total = sum(dist.values())
            if abs(total - 1.0) > 1e-6:
                raise ValueError(
                    f"attribute_targets[{attr}][{level}] sums to {total}, "
                    f"expected 1.0"
                )

        # Monotonicity: winner share easy >= medium >= hard.
        # The winner is identified from the *current* level's baseline,
        # which can shift when upstream knobs (e.g. K4) alter coverage.
        # Downgrade to a warning because the cross-level constraint is
        # not enforceable when winner identity is level-dependent.
        winner = winners.get(attr)
        if winner and winner in level_dists.get("easy", {}):
            shares = [level_dists.get(lvl, {}).get(winner, 0.0) for lvl in VALID_LEVELS]
            if not (shares[0] >= shares[1] >= shares[2]):
                logger.warning(
                    "attribute_targets[%s] winner=%s shares "
                    "not monotone decreasing: %s (may be "
                    "level-dependent winner shift from upstream knobs)",
                    attr,
                    winner,
                    dict(zip(VALID_LEVELS, shares)),
                )


# ---- Core dispatcher ------------------------------------------------------


def _hash_paths(paths: list[Path]) -> str:
    """SHA-256 over the concatenation of the existing paths' contents.

    Used to assert the union of fusion val + test files is byte-identical
    across the K10 window.
    """
    import hashlib

    h = hashlib.sha256()
    for p in paths:
        if p.exists():
            h.update(sha256_file(p).encode("ascii"))
    return h.hexdigest()


def apply_knob_10(
    domain: str,
    level: Literal["easy", "medium", "hard"],
    sources: dict[str, pd.DataFrame],
    config: dict[str, Any],
    fusion_gold_paths: list[Path],
    seed: int = 42,
) -> tuple[
    dict[str, pd.DataFrame],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Apply Knob 10 source reliability reshuffling.

    Parameters
    ----------
    domain : str
        Domain name.
    level : {"easy", "medium", "hard"}
        Difficulty level.
    sources : dict[str, DataFrame]
        Source DataFrames keyed by source name (post-Knobs 1/5/6/7 and 3).
    config : dict
        Parsed Knob 10 YAML.
    fusion_gold_paths : list[Path]
        Paths to the fusion XML files whose union is the protected set
        (typically ``[validation_set.xml, test_set.xml]``). Each is
        read-only and its SHA-256 is asserted invariant across the K10
        window. Non-existent paths are skipped (the rest still define a
        partial protected set).
    seed : int, default 42
        Master RNG seed.

    Returns
    -------
    mutated_sources : dict[str, DataFrame]
        Sources with reshuffled values.
    provenance_df : DataFrame
        Provenance log.
    compromised_mask_df : DataFrame
        Compromised-mask records (empty at easy).
    baseline_alignment_df : DataFrame
        Freshly measured baseline gold-alignment matrix.
    """
    if level not in VALID_LEVELS:
        raise ValueError(f"Invalid level: {level!r}. Valid: {VALID_LEVELS}")

    # --- SHA-256 over union of fusion gold files before the run ---
    gold_hash_before = _hash_paths(fusion_gold_paths)

    # --- Load fusion gold (val ∪ test) ---
    fusion_gold: dict[str, dict[str, str]] = {}
    for path in fusion_gold_paths:
        if not path.exists():
            logger.info("Fusion gold file not present: %s", path)
            continue
        partial = load_fusion_gold(path)
        # Test wins on conflicting entity IDs (val read first if listed
        # first per fusion_protected_paths).
        fusion_gold.update(partial)
    if not fusion_gold:
        logger.warning("No fusion gold entities found across %s", fusion_gold_paths)
        return (
            {n: df.copy() for n, df in sources.items()},
            pd.DataFrame(columns=PROVENANCE_COLUMNS),
            pd.DataFrame(
                columns=["source", "entity_id", "compromised", "knob", "level"]
            ),
            pd.DataFrame(
                columns=[
                    "source",
                    "attribute",
                    "baseline_alignment_rate",
                    "n_cells",
                    "n_aligned",
                ]
            ),
        )

    # --- Config extraction ---
    id_columns: dict[str, str] = config["id_columns"]
    attribute_mapping: dict[str, dict[str, str]] = config["attribute_mapping"]
    attr_targets_all: dict[str, dict[str, dict[str, float]]] = config[
        "attribute_targets"
    ]
    compromise_rate = config["compromise_rate_per_level"][level]
    compromise_rate_overrides_raw = config.get("compromise_rate_overrides", {})
    compromise_rate_overrides: dict[str, float] | None = None
    if compromise_rate_overrides_raw:
        compromise_rate_overrides = {
            src: overrides[level]
            for src, overrides in compromise_rate_overrides_raw.items()
            if level in overrides
        }
    corr_strength: float = config["corr_strength_per_level"][level]
    concentration_cap: float = config.get("concentration_cap", 0.99)

    # --- Resolve per-attribute kinds from protection.py ---
    # Pending #5 strict + infra-aligned wire-up (2026-05-07): kind taxonomy
    # is sourced from protection._DEFAULT_KIND_BY_DOMAIN_ATTR (the canonical
    # locked map from K1/K5/K6 sign-offs), not from K5 attribute_classes
    # reconciliation. is_gold_aligned semantics are unchanged (canonical-
    # form equality); only the kind source-of-truth moves.
    source_names = sorted(sources.keys())
    attribute_classes = resolve_attribute_kinds(domain, list(attr_targets_all.keys()))

    # --- Build entity linkage ---
    domain_config = load_domain_config(domain)
    entity_linkage = build_entity_linkage(domain_config, id_columns)

    # --- Measure fresh baseline ---
    baseline_alignment_df = measure_gold_alignment(
        sources=sources,
        fusion_gold=fusion_gold,
        attribute_mapping=attribute_mapping,
        id_columns=id_columns,
        attribute_classes=attribute_classes,
        entity_linkage=entity_linkage,
    )
    logger.info(
        "Baseline alignment: %d (source, attribute) pairs measured",
        len(baseline_alignment_df),
    )

    # --- Validate config against measured baseline ---
    validate_config(config, baseline_alignment_df)

    # --- Identify reshufflable cells ---
    cells = identify_reshufflable_cells(
        sources=sources,
        fusion_gold=fusion_gold,
        attribute_mapping=attribute_mapping,
        id_columns=id_columns,
        attribute_classes=attribute_classes,
        entity_linkage=entity_linkage,
    )

    reshufflable_count = sum(1 for c in cells if c["cell_type"] == "reshufflable")
    no_gold_count = sum(1 for c in cells if c["cell_type"] == "no_gold_to_route")
    logger.info(
        "Cells: %d reshufflable, %d no_gold_to_route, %d all_aligned, %d passthrough",
        reshufflable_count,
        no_gold_count,
        sum(1 for c in cells if c["cell_type"] == "all_aligned"),
        sum(1 for c in cells if c["cell_type"] == "passthrough"),
    )

    if reshufflable_count == 0:
        logger.info(
            "No reshufflable cells — Knob 10 is a no-op (upstream knobs "
            "produced no variants or all sources agree)"
        )

    # --- Create RNGs (two independent child streams) ---
    parent_rng = make_rng(domain, level, knob=10, master_seed=seed)
    # Draw two seeds for the two stages
    mask_seed = int(parent_rng.integers(0, 2**63))
    cell_seed = int(parent_rng.integers(0, 2**63))
    mask_rng = np.random.default_rng(np.random.SeedSequence(mask_seed))
    cell_rng = np.random.default_rng(np.random.SeedSequence(cell_seed))

    # --- Generate compromised mask ---
    gold_entity_ids = sorted(fusion_gold.keys())
    compromised_mask = generate_compromised_mask(
        source_names=source_names,
        entity_ids=gold_entity_ids,
        compromise_rate=compromise_rate,
        compromise_rate_overrides=compromise_rate_overrides,
        rng=mask_rng,
    )
    total_compromised = sum(len(v) for v in compromised_mask.values())
    logger.info(
        "Compromised mask: %d (source, entity) pairs across %d sources",
        total_compromised,
        len(source_names),
    )

    # --- Extract current-level attribute targets ---
    attr_targets_level: dict[str, dict[str, float]] = {}
    for attr, levels in attr_targets_all.items():
        attr_targets_level[attr] = levels.get(level, {})

    # --- Keep original copies for invariant check ---
    original_sources: dict[str, pd.DataFrame] = {}
    for name, df in sources.items():
        original_sources[name] = df.copy()

    # --- Reshuffle ---
    mutated_sources, provenance_rows = reshuffle_cells(
        cells=cells,
        sources=sources,
        attribute_targets=attr_targets_level,
        compromised_mask=compromised_mask,
        corr_strength=corr_strength,
        concentration_cap=concentration_cap,
        rng=cell_rng,
        level=level,
    )

    # --- Assert multiset invariant ---
    assert_multiset_invariant(original_sources, mutated_sources, cells)
    logger.info("Multiset invariant verified on all cells")

    # --- Assert gold file byte-identity (val + test union) ---
    gold_hash_after = _hash_paths(fusion_gold_paths)
    assert gold_hash_before == gold_hash_after, (
        f"Fusion gold file mutated! SHA-256 before={gold_hash_before}, "
        f"after={gold_hash_after}"
    )
    logger.info(
        "Fusion gold byte-identity verified across %d files (SHA-256: %s)",
        sum(1 for p in fusion_gold_paths if p.exists()),
        gold_hash_before[:16],
    )

    # --- Build output DataFrames ---
    if provenance_rows:
        provenance_df = pd.DataFrame(provenance_rows, columns=PROVENANCE_COLUMNS)
    else:
        provenance_df = pd.DataFrame(columns=PROVENANCE_COLUMNS)

    # Compromised mask DataFrame
    mask_rows: list[dict[str, Any]] = []
    for source, entity_set in sorted(compromised_mask.items()):
        for eid in sorted(entity_set):
            mask_rows.append(
                {
                    "source": source,
                    "entity_id": eid,
                    "compromised": True,
                    "knob": 10,
                    "level": level,
                }
            )
    compromised_mask_df = pd.DataFrame(
        mask_rows,
        columns=["source", "entity_id", "compromised", "knob", "level"],
    )

    # Realised summary (plan_revision.md R-1 / C3 K10): swap_rate is the
    # rate-based intensity metric that monotonicity Check B needs to
    # detect K10 hard's compromised-mask depopulation against the shrinking
    # post-K3 entity pool. ``swap_cells`` counts distinct (entity, attribute)
    # cells reshuffled (transform_fn=reassign_gold_carrier) — same denominator
    # the audit uses today. ``swap_rate`` normalises by reshufflable_count
    # so the per-level value is invariant to K3's drop.
    swap_cells = 0
    if provenance_rows:
        swapped: set[tuple[str, str]] = set()
        for row in provenance_rows:
            if row.get("transform_fn") == "reassign_gold_carrier":
                swapped.add(
                    (str(row.get("entity_id", "")), str(row.get("attribute", "")))
                )
        swap_cells = len(swapped)
    swap_rate = (
        float(swap_cells) / float(reshufflable_count) if reshufflable_count > 0 else 0.0
    )
    realised_df = pd.DataFrame(
        [
            {
                "level": level,
                "reshufflable_count": int(reshufflable_count),
                "swap_cells": int(swap_cells),
                "swap_rate": float(swap_rate),
                "compromised_mask_count": int(total_compromised),
            }
        ]
    )

    return (
        mutated_sources,
        provenance_df,
        compromised_mask_df,
        baseline_alignment_df,
        realised_df,
    )


# ---- Output writing -------------------------------------------------------


def write_outputs(
    provenance_df: pd.DataFrame,
    compromised_mask_df: pd.DataFrame,
    baseline_alignment_df: pd.DataFrame,
    gold_hash: str,
    output_dir: Path,
    realised_df: pd.DataFrame | None = None,
) -> None:
    """Write K10 artifacts to *output_dir*.

    ``realised_df`` is the per-level swap-rate summary produced by
    :func:`apply_knob_10` (plan_revision.md R-1 / C3 K10). Optional for
    backwards compatibility with older callers; when omitted, the
    ``knob_10_realised.csv`` artifact is skipped.
    """
    # Provenance
    prov_dir = output_dir / "output" / "provenance"
    prov_dir.mkdir(parents=True, exist_ok=True)
    provenance_df.to_csv(prov_dir / "knob_10_reliability.csv", index=False)
    logger.info(
        "Wrote provenance (%d rows) to %s",
        len(provenance_df),
        prov_dir / "knob_10_reliability.csv",
    )

    # Compromised mask
    compromised_mask_df.to_csv(prov_dir / "knob_10_compromised_mask.csv", index=False)
    logger.info(
        "Wrote compromised mask (%d rows) to %s",
        len(compromised_mask_df),
        prov_dir / "knob_10_compromised_mask.csv",
    )

    # Baseline alignment
    baselines_dir = output_dir / "output" / "baselines"
    baselines_dir.mkdir(parents=True, exist_ok=True)
    baseline_alignment_df.to_csv(
        baselines_dir / "knob_10_baseline_alignment.csv", index=False
    )

    # Realised swap-rate (C3 K10): per-level summary for the audit.
    if realised_df is not None and not realised_df.empty:
        realised_df.to_csv(baselines_dir / "knob_10_realised.csv", index=False)
        logger.info(
            "Wrote realised swap-rate summary to %s (swap_rate=%.4f)",
            baselines_dir / "knob_10_realised.csv",
            float(realised_df.iloc[0]["swap_rate"]),
        )
    logger.info(
        "Wrote baseline alignment (%d rows) to %s",
        len(baseline_alignment_df),
        baselines_dir / "knob_10_baseline_alignment.csv",
    )

    # Gold hash sentinel
    gold_hash_path = baselines_dir / "knob_10_gold_hash.txt"
    gold_hash_path.write_text(gold_hash + "\n", encoding="utf-8")
    logger.info("Wrote gold hash to %s", gold_hash_path)


# ---- CLI ------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Apply Knob 10 -- Source Reliability Differentiation",
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

    logger.info("Knob 10: domain=%s level=%s output=%s", domain, level, output_dir)

    # Load config and sources
    config = load_knob_10_config(domain)
    sources = load_domain_sources(domain)

    # Resolve fusion gold paths (val ∪ test, both protected)
    fusion_gold_paths = fusion_protected_paths(domain)

    mutated, provenance_df, mask_df, baseline_df, realised_df = apply_knob_10(
        domain=domain,
        level=level,  # type: ignore[arg-type]
        sources=sources,
        config=config,
        fusion_gold_paths=fusion_gold_paths,
        seed=args.seed,
    )

    # Write outputs (gold hash covers the val + test union)
    gold_hash = _hash_paths(fusion_gold_paths)
    write_outputs(
        provenance_df,
        mask_df,
        baseline_df,
        gold_hash,
        output_dir,
        realised_df=realised_df,
    )

    # Summary
    for src_name in sorted(mutated.keys()):
        logger.info("  %s: %d rows", src_name, len(mutated[src_name]))
    logger.info("Provenance: %d rows", len(provenance_df))
    logger.info("Compromised mask: %d rows", len(mask_df))


if __name__ == "__main__":
    main()
