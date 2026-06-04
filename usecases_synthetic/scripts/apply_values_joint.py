#!/usr/bin/env python3
"""Joint Value Perturbation — orchestrate Knobs 1, 5, 6 sequentially.

Thin orchestration layer that applies the value-perturbation knobs in
the canonical order ``K1 → K5 → K6``, coordinating a shared
``CollisionIndex`` so that no cell is double-touched by K1 and K5 and
so that K6 correctly respects the K4-fabricated exception rule.

See ``plans/module_07_joint_values.md`` for the module specification
and ``knobs/cross_cutting.md`` § "Cell-collision coordination" for the
collision rules:

===== ====== =======================
Knob  Skips  Exception
===== ====== =======================
K1    K4     None (unconditional)
K5    K1, K4 None (defensive skip)
K6    K1, K5 K4-fabricated NOT skipped
===== ====== =======================

The orchestrator assumes K4 (if any) has already been applied and its
provenance written to ``output/provenance/knob_04_*.csv``. It loads the
post-K4 source DataFrames by calling :func:`load_domain_sources` (which
currently reads the unmodified source data — K4 integration will be
wired through Module 8).

Usage
-----
::

    python usecases_synthetic/scripts/apply_values_joint.py \\
        --domain companies --level medium

Inputs
------
- Source DataFrames from ``usecases/<domain>/input/data/``
- Per-knob configs under ``usecases_synthetic/config/knob_{01,05,06}_*/``
- Any pre-existing K4 provenance under ``<output_dir>/output/provenance/``

Outputs (under *output_dir*)
----------------------------
- ``output/provenance/knob_01_surface.csv``
- ``output/provenance/knob_01_skipped.csv``
- ``output/provenance/knob_05_format_unit.csv``
- ``output/provenance/knob_05_skipped.csv``
- ``output/provenance/knob_06_noise.csv``
- ``output/provenance/knob_06_skipped.csv``
- ``output/provenance/joint_values_audit.csv`` — collision audit summary
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import pandas as pd

# ---------------------------------------------------------------------------
# Ensure repo root is on sys.path.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from usecases_synthetic.lib.collision_index import CollisionIndex
from usecases_synthetic.lib.domain_config import VALID_LEVELS, load_knob_config
from usecases_synthetic.lib.llm_cache import LLMCache
from usecases_synthetic.lib.loaders import load_domain_sources
from usecases_synthetic.lib.surface_operators import build_openai_paraphrase_client
from usecases_synthetic.scripts.apply_knob_01_surface import (
    apply_knob_01,
    build_entity_linkage as build_entity_linkage_k1,
    write_outputs as write_outputs_k1,
)
from usecases_synthetic.scripts.apply_knob_05_format import (
    apply_knob_05,
    write_outputs as write_outputs_k5,
)
from usecases_synthetic.scripts.apply_knob_06_noise import (
    apply_knob_06,
    build_entity_linkage as build_entity_linkage_k6,
    write_outputs as write_outputs_k6,
)

logger = logging.getLogger(__name__)


# ---- Collision audit ------------------------------------------------------


AUDIT_COLUMNS = [
    "check",
    "status",
    "detail",
]


def _audit_collisions(
    prov_k1: pd.DataFrame,
    prov_k5: pd.DataFrame,
    prov_k6: pd.DataFrame,
    collision_index_after_k4: CollisionIndex,
) -> pd.DataFrame:
    """Run the post-joint-phase collision audit.

    Verifies:

    1. No ``(entity_id, source, attribute)`` triple appears in more than
       one of K1/K5 provenance outputs.
    2. K4-fabricated cells appear only in K6 provenance (not K1 or K5).
    3. K1, K5, K6 provenance outputs have pairwise disjoint touched sets
       except for the explicit K6-on-K4-fabricated overlap.
    """
    rows: list[dict[str, Any]] = []

    def _triples(df: pd.DataFrame) -> set[tuple[str, str, str]]:
        if df.empty:
            return set()
        return {
            (str(r["entity_id"]), str(r["source"]), str(r["attribute"]))
            for _, r in df.iterrows()
        }

    s1 = _triples(prov_k1)
    s5 = _triples(prov_k5)
    s6 = _triples(prov_k6)

    # Check 1: K1 and K5 disjoint.
    overlap_15 = s1 & s5
    rows.append(
        {
            "check": "k1_k5_disjoint",
            "status": "PASS" if not overlap_15 else "FAIL",
            "detail": f"{len(overlap_15)} overlapping cells",
        }
    )

    # Check 2: K1 and K6 disjoint (K6 skips K1).
    overlap_16 = s1 & s6
    rows.append(
        {
            "check": "k1_k6_disjoint",
            "status": "PASS" if not overlap_16 else "FAIL",
            "detail": f"{len(overlap_16)} overlapping cells",
        }
    )

    # Check 3: K5 and K6 disjoint (K6 skips K5).
    overlap_56 = s5 & s6
    rows.append(
        {
            "check": "k5_k6_disjoint",
            "status": "PASS" if not overlap_56 else "FAIL",
            "detail": f"{len(overlap_56)} overlapping cells",
        }
    )

    # Check 4: K4-fabricated cells only appear in K6 (if anywhere).
    k4_fab = collision_index_after_k4._k4_fabricated  # type: ignore[attr-defined]
    k1_on_k4 = s1 & k4_fab
    k5_on_k4 = s5 & k4_fab
    rows.append(
        {
            "check": "k4_fabricated_not_in_k1",
            "status": "PASS" if not k1_on_k4 else "FAIL",
            "detail": f"{len(k1_on_k4)} K4-fabricated cells touched by K1",
        }
    )
    rows.append(
        {
            "check": "k4_fabricated_not_in_k5",
            "status": "PASS" if not k5_on_k4 else "FAIL",
            "detail": f"{len(k5_on_k4)} K4-fabricated cells touched by K5",
        }
    )

    # Check 5: total rows written.
    rows.append(
        {
            "check": "provenance_row_totals",
            "status": "INFO",
            "detail": (
                f"k1={len(prov_k1)} k5={len(prov_k5)} k6={len(prov_k6)} "
                f"total={len(prov_k1) + len(prov_k5) + len(prov_k6)}"
            ),
        }
    )

    return pd.DataFrame(rows, columns=AUDIT_COLUMNS)


# ---- Orchestration --------------------------------------------------------


def apply_values_joint(
    domain: str,
    level: str,
    sources: dict[str, pd.DataFrame],
    output_dir: Path,
    *,
    seed: int = 42,
    llm_cache_k1: LLMCache | None = None,
    strict_cache_k1: bool = False,
    api_client_k1: Callable[[str, str], str] | None = None,
    levels_override: dict[str, str] | None = None,
    protection_source: str = "gold",
) -> dict[str, Any]:
    """Apply K1, K5, K6 jointly in canonical order with shared collisions.

    Parameters
    ----------
    domain : str
        Domain name (e.g. ``"companies"``).
    level : {"easy", "medium", "hard"}
        Difficulty level.
    sources : dict[str, DataFrame]
        Post-K4 source DataFrames keyed by source name.
    output_dir : Path
        Variant directory root. Per-knob provenance CSVs and the joint
        audit are written under ``<output_dir>/output/provenance/``.
    seed : int, default 42
        Master RNG seed forwarded to each knob.
    llm_cache_k1 : LLMCache or None
        Optional pre-built LLM cache for K1 (hard level).
    strict_cache_k1 : bool, default False
        K1 hard-level strict cache toggle.
    api_client_k1 : callable or None
        Optional live OpenAI paraphrase client
        (``(prompt_template, value) -> paraphrase``) wired into K1's
        ``llm_paraphrase`` operator. Invoked only on cache miss when
        ``strict_cache_k1`` is False. ``None`` (the default) keeps K1 in
        strict-cache-only mode -- a cache miss then degrades to a
        deterministic operator (or ``strict_cache_miss`` skip). Build via
        :func:`surface_operators.build_openai_paraphrase_client`.
    levels_override : dict[str, str] or None
        Optional per-knob level overrides. Keys are ``"knob_01"``,
        ``"knob_05"``, ``"knob_06"``; values must be members of
        ``VALID_LEVELS``. Missing keys fall back to ``level``. Used by
        the ablation pathway to set a single knob to ``hard`` while the
        others stay at ``easy``.

    Returns
    -------
    dict
        ``{"sources": dict[str, DataFrame], "provenance_k1": DataFrame,
        "provenance_k5": DataFrame, "provenance_k6": DataFrame,
        "audit": DataFrame}``.
    """
    overrides = dict(levels_override or {})
    level_k1 = overrides.get("knob_01", level)
    level_k5 = overrides.get("knob_05", level)
    level_k6 = overrides.get("knob_06", level)
    for nm, lv in (
        ("knob_01", level_k1),
        ("knob_05", level_k5),
        ("knob_06", level_k6),
    ):
        if lv not in VALID_LEVELS:
            raise ValueError(f"Invalid level for {nm}: {lv!r}. Valid: {VALID_LEVELS}")

    prov_dir = output_dir / "output" / "provenance"
    prov_dir.mkdir(parents=True, exist_ok=True)

    # C13 intact-cluster rule: derive the set of source-record IDs that
    # still exist in the post-K2+K4 sources passed in. K1 + K6 consult
    # this to decide which silver clusters are intact vs broken. K1 + K5
    # mutate cell values but never drop entities, so the same set is
    # valid for both knobs in this joint step. Only computed when silver
    # mode is active; gold-only callers don't need it.
    surviving_record_ids: set[str] | None = None
    if protection_source == "silver":
        config_id_columns: dict[str, str] = (
            load_knob_config(1, domain).get("id_columns") or {}
        )
        surviving_record_ids = set()
        for src_name, df in sources.items():
            id_col = config_id_columns.get(src_name)
            if id_col and id_col in df.columns:
                surviving_record_ids.update(df[id_col].astype(str))
        logger.info(
            "C13 intact-cluster gate: %d surviving record ids feed K1/K6 "
            "silver-target filtering",
            len(surviving_record_ids),
        )

    # Snapshot the collision index seeded from any pre-existing K4
    # provenance. We keep a reference for the final audit so we can
    # identify K4-fabricated cells after all knobs have run.
    baseline_index = CollisionIndex(prov_dir)
    baseline_index.reload()
    k4_fab_snapshot = set(baseline_index._k4_fabricated)  # type: ignore[attr-defined]
    logger.info(
        "Joint start: %d K4-touched cells, %d K4-fabricated",
        len(baseline_index._touched),  # type: ignore[attr-defined]
        len(k4_fab_snapshot),
    )

    # ---- Step 1: K1 Surface Augmentation ---------------------------------
    logger.info("Joint step 1/3: K1 surface augmentation")
    config_k1 = load_knob_config(1, domain)
    id_columns_k1: dict[str, str] = config_k1.get("id_columns", {})
    entity_groups_k1 = build_entity_linkage_k1(domain, id_columns_k1, sources)

    collision_k1 = CollisionIndex(prov_dir)
    collision_k1.reload()

    # Concurrent LLM prewarm (opt-in via env ``K1_LLM_CONCURRENCY`` > 1).
    # K1's paraphrase cache-miss calls are the dominant LLM cost at scale and
    # are otherwise issued sequentially (~1/sec). Run a throwaway *collect*
    # pass on isolated copies of the sources + collision provenance — cell
    # selection is fixed by the seed + config, so it is identical to the real
    # pass — to enumerate every cache-miss prompt, fill them through a thread
    # pool, then let the real pass below run fully cache-hit. ``temperature=0``
    # keeps the warmed cache byte-identical to a sequential run.
    _k1_concurrency = int(os.environ.get("K1_LLM_CONCURRENCY", "0") or "0")
    if _k1_concurrency > 1 and api_client_k1 is not None and not strict_cache_k1:
        _collect_root = Path(tempfile.mkdtemp(prefix="k1_prewarm_"))
        try:
            _collect_prov = _collect_root / "prov"
            shutil.copytree(prov_dir, _collect_prov)
            _collect_collision = CollisionIndex(_collect_prov)
            _collect_collision.reload()
            _src_copy = {k: v.copy(deep=True) for k, v in sources.items()}
            llm_cache_k1.begin_collect()
            apply_knob_01(
                domain=domain,
                level=level_k1,  # type: ignore[arg-type]
                sources=_src_copy,
                config=config_k1,
                entity_groups=entity_groups_k1,
                collision_index=_collect_collision,
                llm_cache=llm_cache_k1,
                llm_client=api_client_k1,
                committee_fn=None,
                strict_cache=False,
                seed=seed,
                protection_source=protection_source,
                surviving_record_ids=surviving_record_ids,
            )

            # Only cache genuine paraphrases — a failed K1 api call returns
            # ``{"paraphrase": ""}`` (build_openai_paraphrase_client swallows
            # network/rate-limit errors into ""). Skipping those here leaves
            # the cell uncached so the real sequential pass re-issues it at
            # 1/sec (where 429s do not recur), instead of poisoning the warm
            # cache with a permanent deterministic-fallback empty.
            def _k1_result_ok(result: object) -> bool:
                return (
                    isinstance(result, dict)
                    and str(result.get("paraphrase", "")).strip() != ""
                )

            _filled, _skipped = llm_cache_k1.flush_concurrent(
                max_workers=_k1_concurrency,
                result_ok=_k1_result_ok,
            )
            logger.info(
                "[K1] concurrent prewarm: filled %d paraphrase cache entries "
                "(%d workers; %d transient failures left for the sequential "
                "pass)",
                _filled,
                _k1_concurrency,
                _skipped,
            )
        finally:
            llm_cache_k1.end_collect()
            shutil.rmtree(_collect_root, ignore_errors=True)

    sources_after_k1, prov_k1, skipped_k1, realised_k1 = apply_knob_01(
        domain=domain,
        level=level_k1,  # type: ignore[arg-type]
        sources=sources,
        config=config_k1,
        entity_groups=entity_groups_k1,
        collision_index=collision_k1,
        llm_cache=llm_cache_k1,
        llm_client=api_client_k1,
        committee_fn=None,
        strict_cache=strict_cache_k1,
        seed=seed,
        protection_source=protection_source,
        surviving_record_ids=surviving_record_ids,
    )
    write_outputs_k1(prov_k1, skipped_k1, output_dir, realised_df=realised_k1)
    logger.info("K1: %d prov rows, %d skipped", len(prov_k1), len(skipped_k1))

    # ---- Step 2: K5 Format/Unit Diversity --------------------------------
    logger.info("Joint step 2/3: K5 format/unit diversity")
    config_k5 = load_knob_config(5, domain)

    collision_k5 = CollisionIndex(prov_dir)
    collision_k5.reload()  # Now includes K1 + K4.

    sources_after_k5, prov_k5, skipped_k5 = apply_knob_05(
        domain=domain,
        level=level_k5,  # type: ignore[arg-type]
        sources=sources_after_k1,
        config=config_k5,
        collision_index=collision_k5,
        seed=seed,
    )
    write_outputs_k5(prov_k5, skipped_k5, output_dir)
    logger.info("K5: %d prov rows, %d skipped", len(prov_k5), len(skipped_k5))

    # ---- Step 3: K6 Value Noise ------------------------------------------
    logger.info("Joint step 3/3: K6 value noise")
    config_k6 = load_knob_config(6, domain)
    id_columns_k6: dict[str, str] = config_k6.get("id_columns", {})
    entity_groups_k6 = build_entity_linkage_k6(domain, id_columns_k6, sources_after_k5)

    collision_k6 = CollisionIndex(prov_dir)
    collision_k6.reload()  # Now includes K1 + K4 + K5.

    sources_after_k6, prov_k6, skipped_k6 = apply_knob_06(
        domain=domain,
        level=level_k6,  # type: ignore[arg-type]
        sources=sources_after_k5,
        config=config_k6,
        entity_groups=entity_groups_k6,
        collision_index=collision_k6,
        seed=seed,
        protection_source=protection_source,
        surviving_record_ids=surviving_record_ids,
    )
    write_outputs_k6(prov_k6, skipped_k6, output_dir)
    logger.info("K6: %d prov rows, %d skipped", len(prov_k6), len(skipped_k6))

    # ---- Collision audit --------------------------------------------------
    logger.info("Running post-joint collision audit")
    audit_df = _audit_collisions(
        prov_k1=prov_k1,
        prov_k5=prov_k5,
        prov_k6=prov_k6,
        collision_index_after_k4=baseline_index,
    )
    audit_path = prov_dir / "joint_values_audit.csv"
    audit_df.to_csv(audit_path, index=False)
    logger.info("Wrote collision audit to %s", audit_path)

    failures = audit_df[audit_df["status"] == "FAIL"]
    if not failures.empty:
        logger.error(
            "Joint collision audit FAILED:\n%s",
            failures.to_string(index=False),
        )

    return {
        "sources": sources_after_k6,
        "provenance_k1": prov_k1,
        "provenance_k5": prov_k5,
        "provenance_k6": prov_k6,
        "audit": audit_df,
    }


# ---- CLI ------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Apply value perturbation knobs (K1 -> K5 -> K6) jointly",
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
            "K1 hard-level only: raise on LLM cache miss instead of "
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

    logger.info(
        "Joint values: domain=%s level=%s output=%s",
        domain,
        level,
        output_dir,
    )

    sources = load_domain_sources(domain)

    config_k1 = load_knob_config(1, domain)
    model_id_k1 = config_k1.get("llm_model_id", "claude-opus-4-6")
    op_mix_k1 = config_k1.get("operator_mix", {}).get(level, {})
    k1_uses_llm = float(op_mix_k1.get("llm_paraphrase", 0) or 0) > 0

    # Materialise the K1 paraphrase cache whenever the level draws the
    # llm_paraphrase operator (medium + hard for the default configs); easy
    # never draws it, so it stays cache-free.
    llm_cache_k1: LLMCache | None = None
    if k1_uses_llm:
        cache_dir = (
            REPO_ROOT
            / "usecases_synthetic"
            / "cache"
            / "knob_01_paraphrases"
            / domain
            / level
        )
        llm_cache_k1 = LLMCache(
            cache_dir=cache_dir,
            prompt_version=config_k1.get("llm_prompt_version", "v1"),
            model_id=model_id_k1,
        )

    strict_cache_k1 = args.strict_cache or (level == "hard")

    # Wire a live paraphrase client so cache misses call the LLM rather than
    # degrading to a deterministic operator. Skipped under strict cache (the
    # documented hard-level replay path) and when no API key is set.
    api_client_k1: Callable[[str, str], str] | None = None
    if (
        llm_cache_k1 is not None
        and not strict_cache_k1
        and os.environ.get("OPENAI_API_KEY")
    ):
        try:
            api_client_k1 = build_openai_paraphrase_client(model_id=model_id_k1)
            logger.info("[K1] OpenAI paraphrase client active (model=%s)", model_id_k1)
        except Exception as exc:  # pragma: no cover - construction failures
            logger.warning(
                "[K1] OpenAI client build failed (%s); deterministic fallback",
                exc,
            )

    result = apply_values_joint(
        domain=domain,
        level=level,
        sources=sources,
        output_dir=output_dir,
        seed=args.seed,
        llm_cache_k1=llm_cache_k1,
        strict_cache_k1=strict_cache_k1,
        api_client_k1=api_client_k1,
    )

    logger.info("Joint values done")
    for src_name in sorted(result["sources"].keys()):
        logger.info("  %s: %d rows", src_name, len(result["sources"][src_name]))


if __name__ == "__main__":
    main()
