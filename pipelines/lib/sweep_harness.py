"""Per-stage chained-sweep harness.

For each pipeline stage, this module exposes a ``sweep_<stage>``
function that:

1. Takes the upstream :class:`PipelineState` (carrying the winning
   output from the previous stage).
2. Iterates every member's hyperparameter grid from the existing
   ``usecases_synthetic.scripts._tune_<stage>_committee SPECS``
   dicts.
3. Evaluates each (member, HP) cell against this stage's val gold.
4. Returns a dataclass with per-(member, HP) val + test rows, the
   per-member val-best HP, and the cross-member winner.

**Conventions** (per the user-issued directive 2026-05-28):

- **No model reuse** from ``usecases_synthetic/cache/``. Learned
  matchers (Ditto, Magellan classifier) are retrained from scratch
  with checkpoints written under
  ``pipelines/<domain>/checkpoints/<stage>/<member>/<hp_hash>/``.
- **Grids reused, results written elsewhere**. The ``SPECS`` dicts
  are imported as the grid source-of-truth. Sweep result rows are
  written to ``pipelines/<domain>/sweeps/<stage>/sweep.json`` by the
  caller (this module only returns the in-memory result).
"""

from __future__ import annotations

import hashlib
import importlib
import itertools
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

from .bundle import PipelineState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class HpCell:
    """One (member, HP combo) sweep result."""

    member: str
    hp: dict[str, Any]
    hp_id: str
    val_score: float
    test_score: float
    metric_key: str
    runtime_s: float
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class StageSweepResult:
    """Full chained-sweep result for one stage."""

    stage: str
    cells: list[HpCell] = field(default_factory=list)
    # Per-member val-best HP + score.
    per_member_winner: dict[str, HpCell] = field(default_factory=dict)
    # Cross-member winner name.
    cross_member_winner: str = ""
    cross_member_winner_val_score: float = 0.0
    cross_member_winner_test_score: float = 0.0
    metric_key: str = "f1"
    runtime_s: float = 0.0
    notes: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Plain-dict view for JSON serialisation."""
        return {
            "stage": self.stage,
            "metric_key": self.metric_key,
            "cross_member_winner": self.cross_member_winner,
            "cross_member_winner_val_score": self.cross_member_winner_val_score,
            "cross_member_winner_test_score": self.cross_member_winner_test_score,
            "per_member_winner": {
                member: {
                    "hp": cell.hp,
                    "hp_id": cell.hp_id,
                    "val_score": cell.val_score,
                    "test_score": cell.test_score,
                    "runtime_s": cell.runtime_s,
                }
                for member, cell in self.per_member_winner.items()
            },
            "n_cells": len(self.cells),
            "runtime_s": self.runtime_s,
            "notes": self.notes,
        }

    def write_winners_json(self, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "winners.json"
        path.write_text(json.dumps(self.as_dict(), indent=2, default=str))
        return path

    def write_sweep_json(self, out_dir: Path) -> Path:
        """Full per-cell sweep result."""
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "sweep.json"
        payload = {
            "stage": self.stage,
            "metric_key": self.metric_key,
            "cells": [
                {
                    "member": c.member,
                    "hp": c.hp,
                    "hp_id": c.hp_id,
                    "val_score": c.val_score,
                    "test_score": c.test_score,
                    "runtime_s": c.runtime_s,
                    "extras": c.extras,
                }
                for c in self.cells
            ],
        }
        path.write_text(json.dumps(payload, indent=2, default=str))
        return path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def hp_hash(hp: Mapping[str, Any]) -> str:
    """Stable short hash for a HP dict — used as a directory name."""
    payload = json.dumps(hp, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def grid(d: Mapping[str, list[Any]]) -> list[dict[str, Any]]:
    """Cartesian product of a ``{key: [values]}`` dict.

    Empty-dict input → ``[{}]`` (one no-op cell).
    """
    if not d:
        return [{}]
    keys = list(d.keys())
    combos = list(itertools.product(*[d[k] for k in keys]))
    return [dict(zip(keys, c, strict=True)) for c in combos]


def _import_specs(stage: str) -> Any:
    """Pull the ``SPECS`` dict from the existing tune script for ``stage``."""
    mapping = {
        "sm": "usecases_synthetic.scripts._tune_sm_committee",
        "norm": "usecases_synthetic.scripts._tune_norm_committee",
    }
    if stage not in mapping:
        raise ValueError(f"No SPECS available for stage {stage!r}")
    mod = importlib.import_module(mapping[stage])
    return getattr(mod, "SPECS")


def _pick_winners(cells: list[HpCell]) -> tuple[dict[str, HpCell], str, float, float]:
    """Per-member val-best HP and cross-member winner.

    Deterministic tie-break: by member name, then by hp_id (ascending).
    """
    per_member: dict[str, HpCell] = {}
    for cell in cells:
        existing = per_member.get(cell.member)
        if (
            existing is None
            or cell.val_score > existing.val_score
            or (cell.val_score == existing.val_score and cell.hp_id < existing.hp_id)
        ):
            per_member[cell.member] = cell
    if not per_member:
        return {}, "", 0.0, 0.0
    # Cross-member: argmax val, tie-break by member name asc.
    sorted_members = sorted(per_member.items(), key=lambda x: (-x[1].val_score, x[0]))
    winner_name, winner_cell = sorted_members[0]
    return per_member, winner_name, winner_cell.val_score, winner_cell.test_score


# ---------------------------------------------------------------------------
# Stage 1: SM sweep
# ---------------------------------------------------------------------------


def sweep_sm(
    state: PipelineState,
    *,
    sm_yaml_path: Path,
    out_dir: Path,
    skip_members: set[str] | None = None,
    with_llm: bool = False,
) -> StageSweepResult:
    """Sweep SM committee on the bundle's source frames.

    For SM, "upstream state" is the raw bundle — SM is the first stage.
    Each member's grid is the cartesian (init × match_kwargs) from
    ``_tune_sm_committee.SPECS``. Selection metric is F1 against
    ``bundle.sm_mapping`` (which is the products SM gold).

    There is no held-out SM test split for products baseline; we
    report ``test_score = val_score`` and flag it in notes.

    Members skipped by default: ``duplicate_majority`` (no useful grid
    on baseline; the SPECS entry exists but produces a single
    locked-in choice). LLM members (``llm_openai``,
    ``magneto_slm_llm``) are gated on ``with_llm`` to control cost.
    """
    t0 = time.monotonic()
    specs = _import_specs("sm")
    skip_members = skip_members or {"duplicate_majority"}

    from usecases_synthetic.lib.committee_sm import (
        score_sm_mapping,
        _target_df_from_schema,
    )

    bundle = state.bundle
    gold = bundle.sm_mapping
    if gold is None or gold.empty:
        raise ValueError(
            f"No SM gold for {bundle.domain}; refusing to sweep without "
            "a scoring surface."
        )
    gold_target_name = (
        str(gold["target_dataset"].iloc[0])
        if "target_dataset" in gold.columns
        else None
    )

    fusion_frames: list[pd.DataFrame] = []
    if bundle.fusion_validation is not None:
        fusion_frames.append(bundle.fusion_validation)
    if bundle.fusion_gold is not None:
        fusion_frames.append(bundle.fusion_gold)
    target_df = _target_df_from_schema(
        bundle.target_schema,
        bundle.sources,
        target_name=gold_target_name,
        fusion_frames=fusion_frames or None,
    )

    cells: list[HpCell] = []
    for member_name, spec in specs.items():
        if member_name in skip_members:
            logger.info("Skipping SM member %s (in skip_members)", member_name)
            continue
        (mod_path, cls_name), init_grid_d, match_grid_d = spec
        # LLM gating: SM SPECS doesn't carry signal_type, but the LLM
        # members are named ``llm_openai`` and ``magneto_slm_llm``.
        is_llm = member_name in {"llm_openai", "magneto_slm_llm"}
        if is_llm and not with_llm:
            logger.info("Skipping SM LLM member %s (with_llm=False)", member_name)
            continue

        init_combos = grid(init_grid_d)
        match_combos = grid(match_grid_d)
        logger.info(
            "Sweeping SM %s: %d init × %d match = %d cells",
            member_name,
            len(init_combos),
            len(match_combos),
            len(init_combos) * len(match_combos),
        )

        cls = getattr(importlib.import_module(mod_path), cls_name)
        for init_params in init_combos:
            try:
                matcher = cls(**init_params)
            except Exception:
                logger.exception(
                    "SM %s init failed for %s; skipping cell", member_name, init_params
                )
                continue
            for match_kwargs in match_combos:
                hp = {
                    **{f"init.{k}": v for k, v in init_params.items()},
                    **{f"match.{k}": v for k, v in match_kwargs.items()},
                }
                hid = hp_hash(hp)
                t_cell = time.monotonic()
                all_maps: list[pd.DataFrame] = []
                for source_name, source_df in bundle.sources.items():
                    try:
                        mapping = matcher.match(source_df, target_df, **match_kwargs)
                        all_maps.append(mapping)
                    except Exception:
                        logger.exception(
                            "SM %s match failed on source %s with hp %s",
                            member_name,
                            source_name,
                            hp,
                        )
                if all_maps:
                    combined = pd.concat(all_maps, ignore_index=True)
                else:
                    combined = pd.DataFrame(
                        columns=[
                            "source_dataset",
                            "source_column",
                            "target_dataset",
                            "target_column",
                            "score",
                        ]
                    )
                metrics = score_sm_mapping(combined, gold)
                val = float(metrics.get("f1", 0.0))
                cells.append(
                    HpCell(
                        member=member_name,
                        hp=hp,
                        hp_id=hid,
                        val_score=val,
                        test_score=val,  # no held-out test for baseline
                        metric_key="f1",
                        runtime_s=time.monotonic() - t_cell,
                    )
                )

    per_member, winner, val_w, test_w = _pick_winners(cells)
    result = StageSweepResult(
        stage="sm",
        cells=cells,
        per_member_winner=per_member,
        cross_member_winner=winner,
        cross_member_winner_val_score=val_w,
        cross_member_winner_test_score=test_w,
        metric_key="f1",
        runtime_s=time.monotonic() - t0,
        notes={
            "test_eq_val": True,
            "test_eq_val_reason": "no held-out SM test split for baseline",
            "skipped_members": sorted(skip_members),
        },
    )
    if out_dir is not None:
        result.write_sweep_json(out_dir)
        result.write_winners_json(out_dir)
    return result


# ---------------------------------------------------------------------------
# Stage 2: Norm sweep
# ---------------------------------------------------------------------------


def sweep_norm(
    state: PipelineState,
    *,
    norm_yaml_path: Path,
    out_dir: Path,
    skip_members: set[str] | None = None,
    with_llm: bool = False,
) -> StageSweepResult:
    """Sweep Norm committee on stage-1 (SM) winner state.

    Iterates ``_tune_norm_committee.SPECS`` per-member init-grid.
    Each (member, HP) cell runs the normalizer against
    fusion-protected cells (the per-domain context built by
    ``_domain_context``); selection metric is macro_f1 across
    attributes.

    LLM normalizers (``llm_canonicalize``) are gated by ``with_llm``
    because the grid has 4 ``num_examples`` choices × ~600 cells
    each → thousands of LLM calls per combo.

    Writes per-cell sweep.json + per-member-winner winners.json to
    ``out_dir`` when non-None.
    """
    t0 = time.monotonic()
    skip_members = skip_members or set()

    # Reuse the synthetic tune script's helpers directly. They build a
    # per-domain context (fusion-protected cells, attr_index, eligible
    # attributes) and score one normalizer instance end-to-end. We loop
    # the grid ourselves so we can write per-cell artifacts under the
    # pipeline-isolated out_dir.
    tune_mod = importlib.import_module(
        "usecases_synthetic.scripts._tune_norm_committee"
    )
    specs = getattr(tune_mod, "SPECS")
    domain_context_fn = getattr(tune_mod, "_domain_context")
    score_member_fn = getattr(tune_mod, "_score_member")
    instantiate_fn = getattr(tune_mod, "_instantiate")
    expand_grid = getattr(tune_mod, "_expand_grid")

    domain = state.bundle.domain
    try:
        ctx = domain_context_fn(domain)
    except Exception:
        logger.exception(
            "Norm sweep: failed to build per-domain context for %s; "
            "Norm sweep cannot run.",
            domain,
        )
        return StageSweepResult(
            stage="norm",
            cells=[],
            per_member_winner={},
            cross_member_winner="",
            metric_key="macro_f1",
            runtime_s=time.monotonic() - t0,
            notes={"error": "context build failed"},
        )

    cells: list[HpCell] = []
    for member_name, ((module_path, cls_name), init_grid) in specs.items():
        if member_name in skip_members:
            logger.info("Norm sweep: skipping %s (in skip_members)", member_name)
            continue
        if member_name == "llm_canonicalize" and not with_llm:
            logger.info("Norm sweep: skipping %s (with_llm=False)", member_name)
            continue

        init_combos = expand_grid(init_grid)
        logger.info("Norm sweep: member=%s grid_size=%d", member_name, len(init_combos))
        for init in init_combos:
            t_cell = time.monotonic()
            try:
                member = instantiate_fn(
                    module_path,
                    cls_name,
                    init,
                    member_name=member_name,
                    domain=domain,
                )
            except Exception:
                logger.exception(
                    "Norm sweep: instantiation failed for %s with %s",
                    member_name,
                    init,
                )
                continue
            try:
                scores = score_member_fn(member, ctx, member_name=member_name)
            except Exception:
                logger.exception(
                    "Norm sweep: scoring failed for %s with %s",
                    member_name,
                    init,
                )
                continue
            metrics = scores.macro_metrics()
            val_f1 = float(metrics.get("macro_f1", 0.0))
            hp = {f"init.{k}": v for k, v in init.items()}
            cells.append(
                HpCell(
                    member=member_name,
                    hp=hp,
                    hp_id=hp_hash(hp),
                    val_score=val_f1,
                    test_score=val_f1,  # no held-out test surface for Norm
                    metric_key="macro_f1",
                    runtime_s=time.monotonic() - t_cell,
                    extras={
                        "per_attribute": {
                            a: s.f1 for a, s in scores.by_attribute.items()
                        }
                    },
                )
            )

    per_member, winner, val_w, test_w = _pick_winners(cells)
    result = StageSweepResult(
        stage="norm",
        cells=cells,
        per_member_winner=per_member,
        cross_member_winner=winner,
        cross_member_winner_val_score=val_w,
        cross_member_winner_test_score=test_w,
        metric_key="macro_f1",
        runtime_s=time.monotonic() - t0,
        notes={
            "test_eq_val": True,
            "test_eq_val_reason": "Norm has no held-out test surface; "
            "fusion-protected cells are the only ground truth.",
            "skipped_members": sorted(skip_members),
            "llm_enabled": with_llm,
        },
    )
    if out_dir is not None:
        result.write_sweep_json(out_dir)
        result.write_winners_json(out_dir)
    return result


# ---------------------------------------------------------------------------
# Stage 3 + 4: EM blocking + matching sweep
# ---------------------------------------------------------------------------


def sweep_em_blocking(
    state: PipelineState,
    *,
    blocking_yaml_path: Path,
    out_dir: Path,
    sc_block_checkpoint_override: Path | None = None,
) -> StageSweepResult:
    """Sweep EM blocking on stage-2 (Norm) winner state.

    Reuses the three sub-sweep helpers from
    ``_tune_em_blocking_committee``: ``_run_embedding_sweep`` (5-model
    panel), ``_run_standard_sweep`` (per-domain key panel),
    ``_run_sn_sweep`` (key/window panel). Each sub-sweep yields rows
    with ``mean_recall`` / ``mean_rr`` / ``per_pair``. We fold them
    into a flat ``HpCell`` list keyed by sub-sweep + params.

    The sc_block sub-sweep is run with a pipeline-isolated
    ``checkpoint_path`` (from ``sc_block_checkpoint_override``);
    if ``None``, sc_block is skipped LOUDLY.

    Selection metric is ``mean_recall`` × ``mean_rr`` (the existing
    composition strategy applies the same recall-floor logic
    per-pair; the sweep's job is to identify the val-best HP per
    member, not to pick the cross-member winner).
    """
    t0 = time.monotonic()
    tune_mod = importlib.import_module(
        "usecases_synthetic.scripts._tune_em_blocking_committee"
    )
    domain_context_fn = getattr(tune_mod, "_domain_context")
    score_blocker_fn = getattr(tune_mod, "_score_blocker_config")
    domain = state.bundle.domain

    try:
        ctx = domain_context_fn(domain)
    except Exception:
        logger.exception("EM blocking sweep: context build failed for %s", domain)
        return StageSweepResult(
            stage="em_blocking",
            cells=[],
            per_member_winner={},
            cross_member_winner="",
            metric_key="reduction_ratio",
            runtime_s=time.monotonic() - t0,
            notes={"error": "context build failed"},
        )

    contexts = {domain: ctx}
    cells: list[HpCell] = []

    # Embedding sub-sweep (5 models × per-domain text_cols)
    try:
        embedding_rows = tune_mod._run_embedding_sweep(contexts)
        for row in embedding_rows:
            per_dom = row.get("per_domain", {}).get(domain, {})
            recall = float(per_dom.get("mean_pair_recall", row.get("mean_recall", 0.0)))
            rr = float(per_dom.get("mean_reduction_ratio", row.get("mean_rr", 0.0)))
            params = row.get("params", {})
            hp = {f"params.{k}": v for k, v in params.items()}
            cells.append(
                HpCell(
                    member="embedding_blocker",
                    hp=hp,
                    hp_id=hp_hash(hp),
                    val_score=recall * rr,
                    test_score=recall * rr,
                    metric_key="recall_x_rr",
                    runtime_s=0.0,
                    extras={"mean_recall": recall, "mean_rr": rr},
                )
            )
    except Exception:
        logger.exception("EM blocking sweep: embedding sub-sweep failed")

    # Standard sub-sweep
    try:
        standard_rows = tune_mod._run_standard_sweep(contexts)
        for row in standard_rows:
            if row.get("domain") != domain:
                continue
            recall = float(row.get("mean_recall", 0.0))
            rr = float(row.get("mean_rr", 0.0))
            hp = {
                "on_cols": list(row.get("on_cols") or []),
                "key_spec": list(row.get("key_spec") or []),
            }
            cells.append(
                HpCell(
                    member="standard_blocker",
                    hp=hp,
                    hp_id=hp_hash(hp),
                    val_score=recall * rr,
                    test_score=recall * rr,
                    metric_key="recall_x_rr",
                    runtime_s=0.0,
                    extras={"mean_recall": recall, "mean_rr": rr},
                )
            )
    except Exception:
        logger.exception("EM blocking sweep: standard sub-sweep failed")

    # Sorted-neighbourhood sub-sweep
    try:
        sn_rows = tune_mod._run_sn_sweep(contexts)
        for row in sn_rows:
            if row.get("domain") != domain:
                continue
            recall = float(row.get("mean_recall", 0.0))
            rr = float(row.get("mean_rr", 0.0))
            hp = {
                "key": row.get("key_col"),
                "key_spec": dict(row.get("key_spec") or {}),
                "window": row.get("window"),
            }
            cells.append(
                HpCell(
                    member="sorted_neighbourhood_blocker",
                    hp=hp,
                    hp_id=hp_hash(hp),
                    val_score=recall * rr,
                    test_score=recall * rr,
                    metric_key="recall_x_rr",
                    runtime_s=0.0,
                    extras={"mean_recall": recall, "mean_rr": rr},
                )
            )
    except Exception:
        logger.exception("EM blocking sweep: SN sub-sweep failed")

    # sc_block sub-sweep with pipeline-isolated checkpoint
    if sc_block_checkpoint_override is None:
        logger.error(
            "EM blocking sweep: sc_block_checkpoint_override is None; "
            "sc_block sub-sweep skipped (would silently reuse "
            "usecases_synthetic/cache/sc_block_checkpoints)."
        )
    else:
        try:
            from usecases_synthetic.lib.sc_block_train import DOMAIN_TEXT_COLS

            text_cols = DOMAIN_TEXT_COLS.get(domain) or ["title"]
            for top_k in getattr(tune_mod, "SCBLOCK_TOP_KS"):
                for threshold in getattr(tune_mod, "SCBLOCK_THRESHOLDS"):
                    params = {
                        "text_cols": text_cols,
                        "checkpoint_path": str(sc_block_checkpoint_override),
                        "top_k": top_k,
                        "threshold": threshold,
                        "index_backend": "sklearn",
                    }
                    try:
                        res = score_blocker_fn(
                            ctx,
                            blocker_class=(
                                "usecases_synthetic.lib.sc_block_blocker",
                                "SCBlockBlocker",
                            ),
                            blocker_params=params,
                        )
                    except Exception:
                        logger.exception(
                            "sc_block sweep cell failed top_k=%d threshold=%.2f",
                            top_k,
                            threshold,
                        )
                        continue
                    recall = float(res["mean_pair_recall"])
                    rr = float(res["mean_reduction_ratio"])
                    hp = {"top_k": top_k, "threshold": threshold}
                    cells.append(
                        HpCell(
                            member="sc_block",
                            hp=hp,
                            hp_id=hp_hash(hp),
                            val_score=recall * rr,
                            test_score=recall * rr,
                            metric_key="recall_x_rr",
                            runtime_s=float(res.get("elapsed_s", 0.0)),
                            extras={
                                "mean_recall": recall,
                                "mean_rr": rr,
                                "checkpoint_path": str(sc_block_checkpoint_override),
                            },
                        )
                    )
        except Exception:
            logger.exception("EM blocking sweep: sc_block sub-sweep failed")

    per_member, winner, val_w, test_w = _pick_winners(cells)
    result = StageSweepResult(
        stage="em_blocking",
        cells=cells,
        per_member_winner=per_member,
        cross_member_winner=winner,
        cross_member_winner_val_score=val_w,
        cross_member_winner_test_score=test_w,
        metric_key="recall_x_rr",
        runtime_s=time.monotonic() - t0,
        notes={
            "sub_sweeps": ["embedding", "standard", "sn"]
            + (["sc_block"] if sc_block_checkpoint_override else []),
            "sc_block_checkpoint": (
                str(sc_block_checkpoint_override)
                if sc_block_checkpoint_override
                else None
            ),
        },
    )
    if out_dir is not None:
        result.write_sweep_json(out_dir)
        result.write_winners_json(out_dir)
    return result


def sweep_em_matching(
    state: PipelineState,
    *,
    matching_yaml_path: Path,
    out_dir: Path,
    with_llm: bool = False,
) -> StageSweepResult:
    """Sweep EM matching using the existing
    ``_tune_em_matching_committee._run_magellan_sweep`` for Magellan's
    12-cell classifier grid. Per the existing tune script's docstring:

      ditto_plm: no sweep (R2 LR×class-balance is the source of truth;
                 checkpoint at <pipeline-isolated path> is picked).
      magellan:  classifier sweep only (n_estimators × max_depth ×
                 class_weight).
      matchgpt:  no sweep (locked at zero-shot).
      comem:     no sweep (locked at paper defaults).

    For pipeline-isolated Ditto: the orchestrator's
    ``ditto_checkpoint_override`` rewrites the YAML's checkpoint_path
    before calling the EM committee runner. This sweep harness writes
    Magellan grid cells; the locked members (ditto/matchgpt/comem)
    don't contribute additional cells.
    """
    t0 = time.monotonic()
    tune_mod = importlib.import_module(
        "usecases_synthetic.scripts._tune_em_matching_committee"
    )
    domain_context_fn = getattr(tune_mod, "_domain_context")
    domain = state.bundle.domain

    try:
        ctx = domain_context_fn(domain)
    except Exception:
        logger.exception("EM matching sweep: context build failed for %s", domain)
        return StageSweepResult(
            stage="em_matching",
            cells=[],
            per_member_winner={},
            cross_member_winner="",
            metric_key="f1",
            runtime_s=time.monotonic() - t0,
            notes={"error": "context build failed"},
        )

    cells: list[HpCell] = []
    contexts = {domain: ctx}
    try:
        magellan_rows = tune_mod._run_magellan_sweep(contexts)
        for row in magellan_rows:
            if row.get("domain") != domain:
                continue
            f1 = float(row.get("mean_f1", 0.0))
            hp = {
                k: v
                for k, v in row.items()
                if k
                not in {
                    "member",
                    "domain",
                    "mean_f1",
                    "min_f1",
                    "config_id",
                    "per_pair",
                }
            }
            cells.append(
                HpCell(
                    member="magellan",
                    hp=hp,
                    hp_id=hp_hash(hp),
                    val_score=f1,
                    test_score=f1,  # sweep uses gold-pairs closed-set; no separate test
                    metric_key="f1",
                    runtime_s=0.0,
                    extras={"per_pair": row.get("per_pair", {})},
                )
            )
    except Exception:
        logger.exception("EM matching sweep: Magellan sub-sweep failed")

    per_member, winner, val_w, test_w = _pick_winners(cells)
    result = StageSweepResult(
        stage="em_matching",
        cells=cells,
        per_member_winner=per_member,
        cross_member_winner=winner,
        cross_member_winner_val_score=val_w,
        cross_member_winner_test_score=test_w,
        metric_key="f1",
        runtime_s=time.monotonic() - t0,
        notes={
            "swept_members": ["magellan"],
            "locked_members": ["ditto_plm", "matchgpt", "comem"],
            "ditto_isolation": (
                "ditto_plm checkpoint_path is rewritten by the orchestrator's "
                "_maybe_filter_matching_yaml to the pipeline-isolated path."
            ),
        },
    )
    if out_dir is not None:
        result.write_sweep_json(out_dir)
        result.write_winners_json(out_dir)
    return result


# ---------------------------------------------------------------------------
# Stage 5: Refinement sweep (already grid-light: 3 methods)
# ---------------------------------------------------------------------------


def sweep_refinement(
    state: PipelineState,
    *,
    methods: list[str],
    out_dir: Path,
) -> StageSweepResult:
    """Refinement is already a 3-method sweep — delegate to the existing
    :func:`stage_runners.run_refinement` and convert its output.

    This is the only stage where the v1 ``stage_runners`` behaviour
    already matches the chained-sweep semantics: each method is
    evaluated against val gold (using the EM matcher winner's
    per-pair predictions which are upstream state).
    """
    from .stage_runners import run_refinement

    t0 = time.monotonic()
    sel = run_refinement(state, methods=methods)
    cells = [
        HpCell(
            member=method,
            hp={"method": method},
            hp_id=hp_hash({"method": method}),
            val_score=sel.per_member_val.get(method, 0.0),
            test_score=sel.per_member_test.get(method, 0.0),
            metric_key="f1",
            runtime_s=0.0,  # not separately tracked per method
        )
        for method in sel.per_member_val
    ]
    per_member, winner, val_w, test_w = _pick_winners(cells)
    result = StageSweepResult(
        stage="refinement",
        cells=cells,
        per_member_winner=per_member,
        cross_member_winner=winner,
        cross_member_winner_val_score=val_w,
        cross_member_winner_test_score=test_w,
        metric_key="f1",
        runtime_s=time.monotonic() - t0,
        notes={
            "methods_competed": methods,
            "n_correspondences_after_winner": (
                len(state.correspondences) if state.correspondences is not None else 0
            ),
        },
    )
    if out_dir is not None:
        result.write_sweep_json(out_dir)
        result.write_winners_json(out_dir)
    return result


# ---------------------------------------------------------------------------
# Stage 6: Fusion sweep
# ---------------------------------------------------------------------------


def sweep_fusion(
    state: PipelineState,
    *,
    fusion_yaml_path: Path,
    out_dir: Path,
    sub_sweeps: list[str] | None = None,
    with_llm: bool = False,
) -> StageSweepResult:
    """Sweep fusion sub-sweeps from ``_tune_fusion_committee`` on the
    refinement-winner correspondences.

    Sub-sweeps (per the existing tune script):
      - trust, tolerance, trim, list_threshold (small grids over
        member-config knobs)
      - truthfinder (gamma × init_trust), accusim (accuracy_prior ×
        sim_threshold), casefusion (alpha × lr),
        fusionquery (temperature × threshold), ltm (alpha_0 × alpha_1)
      - llm_judge (enabled/disabled) — gated on with_llm

    Each cell runs ``FusionCommitteeRunner`` end-to-end via the
    existing ``_score_run`` helper, mutating the C12 member's params
    block. Scores are the C12 ``aggregated.macro_accuracy``.

    NOTE: this calls C12 end-to-end per cell; if the C12 hang from
    v3/v4/v5 hasn't been fixed, each cell will hang. Run only after
    the hang is diagnosed.
    """
    t0 = time.monotonic()
    sub_sweeps = sub_sweeps or [
        "trust",
        "tolerance",
        "trim",
        "list_threshold",
        "truthfinder",
        "accusim",
        "casefusion",
        "fusionquery",
        "ltm",
    ]
    if with_llm:
        sub_sweeps.append("llm_judge")

    tune_mod = importlib.import_module(
        "usecases_synthetic.scripts._tune_fusion_committee"
    )
    load_roster_yaml = getattr(tune_mod, "_load_roster_yaml")
    domain = state.bundle.domain
    base_path, base_yaml = load_roster_yaml(domain)

    # The sub_* helpers each take (base_yaml, domain, bundle,
    # correspondences) and return a list of {params, score} dicts.
    sub_runners: dict[str, Any] = {
        name: getattr(tune_mod, f"_sub_{name}", None) for name in sub_sweeps
    }
    cells: list[HpCell] = []
    bundle = state.bundle
    correspondences = state.correspondences

    for sub_name, sub_fn in sub_runners.items():
        if sub_fn is None:
            logger.warning("fusion sub-sweep %s not found in tune module", sub_name)
            continue
        try:
            rows = sub_fn(base_yaml, domain, bundle, correspondences)
        except Exception:
            logger.exception("fusion sub-sweep %s failed end-to-end", sub_name)
            continue
        for row in rows or []:
            agg = row.get("aggregated") or row.get("result", {}).get("aggregated", {})
            macro_acc = float(agg.get("macro_accuracy", 0.0)) if agg else 0.0
            params = row.get("params", {})
            hp = {
                "sub_sweep": sub_name,
                **{f"params.{k}": v for k, v in params.items()},
            }
            cells.append(
                HpCell(
                    member=row.get("member") or sub_name,
                    hp=hp,
                    hp_id=hp_hash(hp),
                    val_score=macro_acc,
                    test_score=macro_acc,
                    metric_key="macro_accuracy",
                    runtime_s=float(row.get("runtime_s", 0.0)),
                    extras={"per_attribute": row.get("per_attribute", {})},
                )
            )

    per_member, winner, val_w, test_w = _pick_winners(cells)
    result = StageSweepResult(
        stage="fusion",
        cells=cells,
        per_member_winner=per_member,
        cross_member_winner=winner,
        cross_member_winner_val_score=val_w,
        cross_member_winner_test_score=test_w,
        metric_key="macro_accuracy",
        runtime_s=time.monotonic() - t0,
        notes={
            "sub_sweeps_run": list(sub_sweeps),
            "llm_enabled": with_llm,
            "warn": (
                "fusion sub-sweeps each invoke the C12 FusionCommitteeRunner "
                "end-to-end. If the C12 hang is unresolved, each cell will hang."
            ),
        },
    )
    if out_dir is not None:
        result.write_sweep_json(out_dir)
        result.write_winners_json(out_dir)
    return result


__all__ = [
    "HpCell",
    "StageSweepResult",
    "grid",
    "hp_hash",
    "sweep_sm",
    "sweep_norm",
    "sweep_em_blocking",
    "sweep_em_matching",
    "sweep_refinement",
    "sweep_fusion",
]
