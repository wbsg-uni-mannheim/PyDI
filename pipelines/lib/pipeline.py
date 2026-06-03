"""Best-of-breed pipeline orchestrator.

Threads the stage runners in :mod:`pipelines.lib.stage_runners` into
a single end-to-end pipeline, then emits the four-tier metric panel
(``PyDI.evaluation.panel.compute_e2e_panel``) against a silver
standard.

The pipeline is deliberately greedy: at each stage, the highest-val
committee member wins, and only its output flows to the next stage.
Per the plan's §8.2 caveat, this is locally-optimal but may be
globally suboptimal — flagged in the per-stage report.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

from PyDI.evaluation.panel import E2EPanel, compute_e2e_panel
from PyDI.evaluation.silver_standard import (
    SilverStandard,
    load_workflow_silver,
)
from usecases_synthetic.lib.committee_paths import resolve_committee_path

from ._resource_tracking import process_lifetime_peak_mb
from .bundle import PipelineState, load_pipeline_bundle
from .stage_runners import (
    StageSelection,
    run_em,
    run_fusion,
    run_norm,
    run_refinement,
    run_sm,
)
from .sweep_harness import (
    StageSweepResult,
    sweep_refinement,
    sweep_sm,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config + result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """Parsed pipeline config (the YAML at ``pipelines/configs/<domain>.yaml``)."""

    domain: str
    bundle_source: str
    stages: dict[str, dict[str, Any]]
    column_types: dict[str, str]
    panel_tolerance_default: float
    panel_tolerance_overrides: dict[str, float]
    composite_weights: dict[str, float]
    source_prefix_map: dict[str, str]

    @classmethod
    def from_yaml(cls, path: Path) -> "PipelineConfig":
        """Load a pipeline config YAML."""
        raw = yaml.safe_load(path.read_text())
        tol = raw.get("panel_tolerance", {}) or {}
        return cls(
            domain=raw["domain"],
            bundle_source=raw.get("bundle_source", "synthetic_baseline"),
            stages=raw.get("stages", {}),
            column_types=raw.get("column_types", {}),
            panel_tolerance_default=float(tol.get("default", 0.04)),
            panel_tolerance_overrides={
                k: float(v) for k, v in (tol.get("overrides") or {}).items()
            },
            composite_weights={
                k: float(v) for k, v in (raw.get("composite_weights") or {}).items()
            },
            source_prefix_map=dict(raw.get("source_prefix_map") or {}),
        )


@dataclass
class PipelineRunResult:
    """Bundle of artifacts produced by one pipeline run.

    Parameters
    ----------
    state : PipelineState
        Final state after all stages ran.
    stage_selections : list[StageSelection]
        Per-stage selection records, in execution order.
    sweep_results : list[StageSweepResult]
        Per-stage chained-sweep results when ``--mode sweep`` was
        used. Empty when running with YAML defaults (replay mode).
    panel : E2EPanel or None
        End-to-end metric panel (``None`` if fusion failed).
    mode : str
        Either ``"sweep"`` or ``"replay"``.
    total_runtime_s : float
        End-to-end wall-clock runtime.
    peak_memory_mb : float
        Maximum per-stage peak RSS (MB) across ``stage_selections``.
        ``0.0`` when no stage produced a peak (e.g. psutil missing).
    lifetime_peak_memory_mb : float
        Process-lifetime peak RSS (MB), sampled once after all stages
        finished via :func:`resource.getrusage`. Reflects the largest
        peak any phase of the process has reached, not just the
        in-pipeline phase.
    """

    state: PipelineState
    stage_selections: list[StageSelection] = field(default_factory=list)
    sweep_results: list[StageSweepResult] = field(default_factory=list)
    panel: E2EPanel | None = None
    mode: str = "replay"
    total_runtime_s: float = 0.0
    peak_memory_mb: float = 0.0
    lifetime_peak_memory_mb: float = 0.0


# ---------------------------------------------------------------------------
# Sweep ↔ StageSelection bridge
# ---------------------------------------------------------------------------


def _selection_from_sweep(sweep: StageSweepResult) -> StageSelection:
    """Convert a ``StageSweepResult`` into a ``StageSelection`` so the
    rest of the orchestrator (artifact writers, comparison report)
    can consume sweep + replay modes uniformly.
    """
    per_member_val = {m: cell.val_score for m, cell in sweep.per_member_winner.items()}
    per_member_test = {
        m: cell.test_score for m, cell in sweep.per_member_winner.items()
    }
    return StageSelection(
        stage=sweep.stage,
        winner=sweep.cross_member_winner,
        val_score=sweep.cross_member_winner_val_score,
        test_score=sweep.cross_member_winner_test_score,
        per_member_val=per_member_val,
        per_member_test=per_member_test,
        metric_key=sweep.metric_key,
        runtime_s=sweep.runtime_s,
        notes={**sweep.notes, "n_sweep_cells": len(sweep.cells), "from_sweep": True},
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class BestOfBreedPipeline:
    """End-to-end best-of-breed pipeline.

    Parameters
    ----------
    config : PipelineConfig
        Parsed pipeline config.
    committee_dir : Path
        Directory holding the committee YAMLs (defaults to
        ``usecases_synthetic/config/committees/``).
    with_llm_sm : bool
        Whether to include LLM-backed schema-matching members
        (``llm_matcher``). Default ``True``.
    with_llm_em : bool
        Whether to include LLM-backed EM-matching members
        (``llm_matcher``, ``matchgpt``, ``comem``). Default ``False``
        per the 2026-06-01 directive to drop LLM from entity matching.
    with_llm_fusion : bool
        Whether to include LLM-backed fusion members (``llm_only``,
        ``llm_judge``). Default ``False`` per the 2026-06-01 directive
        to drop LLM from fusion.
    with_llm : bool or None
        Deprecated. When supplied, overrides all three per-stage
        flags to the same value. Kept for backward compatibility
        with the v1 single-flag API.
    mode : {"sweep", "replay"}
        Pipeline execution mode. In ``"sweep"`` mode each stage runs
        the chained-sweep harness (sweep_harness.sweep_<stage>) where
        implemented, falling back to stage_runners with YAML defaults
        for stub stages. In ``"replay"`` mode (default) every stage
        runs the YAML-default committee runners. Sweep mode writes
        per-stage ``sweep.json`` + ``winners.json`` artifacts under
        ``<out_dir>/sweeps/<stage>/``.
    ditto_checkpoint_override : Path or None
        REQUIRED to satisfy the no-model-reuse policy when ditto
        participates. Rewrites the ``ditto_plm`` member's
        ``checkpoint_path`` in an in-memory copy of the matching
        YAML. The canonical
        ``usecases_synthetic/cache/ditto_checkpoints/...`` path is
        NEVER read by the pipeline; when this is ``None`` and the
        YAML's default path is under ``usecases_synthetic/cache/``,
        ditto is dropped from the roster with a clear warning.
    sc_block_checkpoint_override : Path or None
        Same semantics for the ``sc_block`` blocker member.
    out_dir : Path or None
        Output directory for sweep artifacts + pipeline state. When
        ``None`` no artifacts are written during ``run()`` — caller
        is expected to invoke :func:`report.write_run_artifacts`.
    """

    def __init__(
        self,
        config: PipelineConfig,
        *,
        committee_dir: Path,
        with_llm_sm: bool = True,
        with_llm_em: bool = False,
        with_llm_fusion: bool = False,
        with_llm: bool | None = None,
        mode: str = "replay",
        level: str = "baseline",
        ditto_checkpoint_override: Path | None = None,
        sc_block_checkpoint_override: Path | None = None,
        out_dir: Path | None = None,
        fusion_members: set[str] | None = None,
    ) -> None:
        if mode not in {"sweep", "replay"}:
            raise ValueError(f"Unknown mode {mode!r}; expected 'sweep' or 'replay'")
        self.config = config
        self.committee_dir = committee_dir
        # Back-compat: a legacy `with_llm` kwarg forces all three per-stage
        # flags to the same value.
        if with_llm is not None:
            with_llm_sm = with_llm
            with_llm_em = with_llm
            with_llm_fusion = with_llm
        self.with_llm_sm = with_llm_sm
        self.with_llm_em = with_llm_em
        self.with_llm_fusion = with_llm_fusion
        self.mode = mode
        if level not in {"baseline", "easy", "medium", "hard"}:
            raise ValueError(
                f"Unknown level {level!r}; expected one of "
                "{baseline, easy, medium, hard}."
            )
        self.level = level
        self.ditto_checkpoint_override = ditto_checkpoint_override
        self.sc_block_checkpoint_override = sc_block_checkpoint_override
        self.out_dir = out_dir
        # Explicit allowlist of fusion C12 members. When ``None``,
        # only ``with_llm``-gated filtering applies. When a set, every
        # member NOT in the set is disabled in the rewritten YAML —
        # used to escape the v3/v4 fusion stalls on non-LLM members
        # that hang on hidden network deps (see pipelines/products/STATUS.md).
        self.fusion_members = fusion_members

    def run(self) -> PipelineRunResult:
        """Run all enabled stages end-to-end."""
        t0_total = time.monotonic()

        bundle = load_pipeline_bundle(
            self.config.domain,
            level=self.level,
            bundle_source=self.config.bundle_source,
        )
        state = PipelineState(bundle=bundle)
        stage_selections: list[StageSelection] = []
        sweep_results: list[StageSweepResult] = []

        # Stage 1: SM
        if self._stage_enabled("sm"):
            sm_yaml = resolve_committee_path(
                "sm_committee", self.config.domain, committee_dir=self.committee_dir
            )
            if self.mode == "sweep":
                sweep_dir = self._sweep_out_dir("sm")
                sweep_result = sweep_sm(
                    state,
                    sm_yaml_path=sm_yaml,
                    out_dir=sweep_dir,
                    with_llm=self.with_llm_sm,
                )
                sweep_results.append(sweep_result)
                sel = _selection_from_sweep(sweep_result)
                # Apply the SM winner's predictions to state so downstream
                # stages can rely on state.sm_winner / state.sm_mapping_df.
                if sweep_result.cross_member_winner:
                    state.sm_winner = sweep_result.cross_member_winner
                    # Re-run the winner once to get its mapping (the sweep
                    # cells don't retain predictions for memory reasons).
                    winning_sel = run_sm(
                        state, sm_yaml=sm_yaml, with_llm=self.with_llm_sm
                    )
                    if winning_sel.winner == sweep_result.cross_member_winner:
                        # Same winner under YAML defaults — keep its mapping.
                        pass
                    else:
                        logger.warning(
                            "SM sweep winner (%s) ≠ YAML-default winner (%s); "
                            "using YAML-default mapping for state. To use the "
                            "sweep winner's mapping the runner needs HP "
                            "override threaded into run_sm.",
                            sweep_result.cross_member_winner,
                            winning_sel.winner,
                        )
            else:
                sel = run_sm(state, sm_yaml=sm_yaml, with_llm=self.with_llm_sm)
            stage_selections.append(sel)

            # NOTE: post-SM source translation is intentionally NOT
            # applied here anymore (2026-06-02). The canonical_loader
            # returns sources with their RAW per-source column names
            # (manufacturer / brandName / Brand / mfr ...). Downstream
            # committees translate via their own YAML ``column_mapping:``
            # blocks (em_blocking_committee_products.yaml +
            # em_matching_committee_products.yaml +
            # fusion_committee_products.yaml). Norm reads source columns
            # via the SM gold's ``source_column`` (raw name) directly.
            # Running an extra translation here would double-rename and
            # trip ``apply_column_mapping``'s collision-drop logic,
            # leaving downstream stages with only an ``id`` column.
            # The ``_apply_sm_gold_translation_if_needed`` method stays
            # in place as a no-op fallback for any future canonical
            # loader that emits the ``needs_sm_column_translation``
            # attr AND has no YAML column_mapping downstream.

        # Stage 2: Norm
        if self._stage_enabled("norm"):
            norm_yaml = resolve_committee_path(
                "normalization_committee",
                self.config.domain,
                committee_dir=self.committee_dir,
            )
            stage_cfg = self.config.stages.get("norm", {})
            if self.mode == "sweep":
                from .sweep_harness import sweep_norm

                # Norm shares the SM LLM toggle (norm's only LLM member is
                # llm_canonicalize, conceptually closest to schema-level work).
                sweep_result = sweep_norm(
                    state,
                    norm_yaml_path=norm_yaml,
                    out_dir=self._sweep_out_dir("norm"),
                    with_llm=self.with_llm_sm,
                )
                sweep_results.append(sweep_result)
                sel = _selection_from_sweep(sweep_result)
            else:
                sel = run_norm(
                    state,
                    norm_yaml=norm_yaml,
                    vacuous_epsilon=float(stage_cfg.get("vacuous_epsilon", 0.005)),
                    apply_winner=bool(stage_cfg.get("apply_winner", False)),
                    scoring_surface=str(
                        stage_cfg.get("scoring_surface", "xml_targets")
                    ),
                )
            stage_selections.append(sel)

        # Stage 3 + 4: EM (blocking + matching as one joint runner).
        # The YAML in-memory filters rewrite ditto_plm + sc_block
        # checkpoint paths to pipeline-isolated locations (when
        # *_checkpoint_override is set). Members are NEVER silently
        # dropped — if no pipeline-isolated checkpoint is supplied
        # and the YAML default points under usecases_synthetic/cache/,
        # the filter LOUDLY disables the member and tells the user to
        # retrain (per the 2026-05-28 "no silent dropping" directive).
        if self._stage_enabled("em"):
            blocking_yaml = resolve_committee_path(
                "em_blocking_committee",
                self.config.domain,
                committee_dir=self.committee_dir,
            )
            matching_yaml = resolve_committee_path(
                "em_matching_committee",
                self.config.domain,
                committee_dir=self.committee_dir,
            )
            # Apply in-memory rewrites when needed.
            effective_blocking_yaml = self._maybe_filter_blocking_yaml(blocking_yaml)
            effective_matching_yaml = self._maybe_filter_matching_yaml(matching_yaml)
            stage_cfg = self.config.stages.get("em", {})
            clustering = str(stage_cfg.get("clustering", "greedy"))

            if self.mode == "sweep":
                from .sweep_harness import sweep_em_blocking, sweep_em_matching

                blocking_sweep = sweep_em_blocking(
                    state,
                    blocking_yaml_path=effective_blocking_yaml,
                    out_dir=self._sweep_out_dir("em_blocking"),
                    sc_block_checkpoint_override=self.sc_block_checkpoint_override,
                )
                sweep_results.append(blocking_sweep)
                matching_sweep = sweep_em_matching(
                    state,
                    matching_yaml_path=effective_matching_yaml,
                    out_dir=self._sweep_out_dir("em_matching"),
                    with_llm=self.with_llm_em,
                )
                sweep_results.append(matching_sweep)
                # Run the committee anyway to produce per-pair correspondences
                # for the downstream refinement + fusion stages — sweeps tell
                # us HP val-best per member; the actual per-pair winner is
                # picked by the committee's composition logic with those HPs.
                blocking_sel, matching_sel = run_em(
                    state,
                    blocking_yaml=effective_blocking_yaml,
                    matching_yaml=effective_matching_yaml,
                    with_llm=self.with_llm_em,
                    clustering=clustering,
                )
                # Attach sweep metadata to the selections.
                blocking_sel.notes["sweep_n_cells"] = len(blocking_sweep.cells)
                matching_sel.notes["sweep_n_cells"] = len(matching_sweep.cells)
            else:
                blocking_sel, matching_sel = run_em(
                    state,
                    blocking_yaml=effective_blocking_yaml,
                    matching_yaml=effective_matching_yaml,
                    with_llm=self.with_llm_em,
                    clustering=clustering,
                )
            stage_selections.append(blocking_sel)
            stage_selections.append(matching_sel)

        # Stage 5: Refinement
        if self._stage_enabled("refinement"):
            stage_cfg = self.config.stages.get("refinement", {})
            methods = list(stage_cfg.get("methods", ["baseline", "greedy", "mbm"]))
            if self.mode == "sweep":
                sweep_result = sweep_refinement(
                    state,
                    methods=methods,
                    out_dir=self._sweep_out_dir("refinement"),
                )
                sweep_results.append(sweep_result)
                sel = _selection_from_sweep(sweep_result)
            else:
                sel = run_refinement(state, methods=methods)
            stage_selections.append(sel)

        # Stage 6: Fusion
        if self._stage_enabled("fusion"):
            fusion_yaml = resolve_committee_path(
                "fusion_committee",
                self.config.domain,
                committee_dir=self.committee_dir,
            )
            effective_fusion_yaml = self._maybe_filter_fusion_yaml(fusion_yaml)
            if self.mode == "sweep":
                from .sweep_harness import sweep_fusion

                fusion_sweep = sweep_fusion(
                    state,
                    fusion_yaml_path=effective_fusion_yaml,
                    out_dir=self._sweep_out_dir("fusion"),
                    with_llm=self.with_llm_fusion,
                )
                sweep_results.append(fusion_sweep)
            # Always run the actual fusion stage to produce the fused
            # output the panel needs; sweep cells inform per-member HP
            # but the runner picks the winner.
            sel = run_fusion(state, fusion_yaml=effective_fusion_yaml)
            if self.mode == "sweep" and sweep_results:
                sel.notes["sweep_n_cells"] = len(fusion_sweep.cells)
            stage_selections.append(sel)

        # Stage 7: e2e panel
        panel = self._compute_panel(state) if state.fused is not None else None

        total = time.monotonic() - t0_total
        peak_mb = max(
            (sel.peak_memory_mb for sel in stage_selections),
            default=0.0,
        )
        lifetime_peak_mb = process_lifetime_peak_mb()
        logger.info(
            "Pipeline finished in %.1f s (per-stage peak=%.1f MB, lifetime peak=%.1f MB)",
            total,
            peak_mb,
            lifetime_peak_mb,
        )
        return PipelineRunResult(
            state=state,
            stage_selections=stage_selections,
            sweep_results=sweep_results,
            panel=panel,
            mode=self.mode,
            total_runtime_s=total,
            peak_memory_mb=peak_mb,
            lifetime_peak_memory_mb=lifetime_peak_mb,
        )

    def _stage_enabled(self, name: str) -> bool:
        stage_cfg = self.config.stages.get(name, {})
        return bool(stage_cfg.get("enabled", True))

    def _sweep_out_dir(self, stage: str) -> Path | None:
        """Resolve where stage k's sweep artifacts go.

        Returns ``None`` when ``self.out_dir`` is unset (in that case
        the sweep harness keeps results in memory only).
        """
        if self.out_dir is None:
            return None
        return self.out_dir / "sweeps" / stage

    def _apply_sm_gold_translation_if_needed(self, state: PipelineState) -> None:
        """Translate sources to canonical-target-schema column names
        using the SM gold mapping, for sources tagged by the canonical
        loader with ``needs_sm_column_translation``.

        Sources start with raw per-source column names
        (e.g. ``manufacturer`` / ``brandName`` / ``Brand`` / ``mfr``)
        so the SM committee scores its members against varied real
        column names. After SM scoring, downstream stages (Norm, EM,
        Fusion) expect canonical column names — we apply the GOLD
        mapping here. Using the gold (not the winner's predicted
        mapping) is deterministic and keeps downstream input clean
        even when SM under-performs.
        """
        gold = state.bundle.sm_mapping
        if gold is None or gold.empty:
            return
        if not any(
            (df.attrs.get("needs_sm_column_translation") is True)
            for df in state.bundle.sources.values()
        ):
            return

        gold_by_source: dict[str, dict[str, str]] = {}
        for _, row in gold.iterrows():
            src = str(row["source_dataset"])
            src_col = str(row["source_column"])
            tgt_col = str(row["target_column"])
            gold_by_source.setdefault(src, {})[src_col] = tgt_col

        translated: dict[str, "pd.DataFrame"] = {}
        for name, df in state.bundle.sources.items():
            if not df.attrs.get("needs_sm_column_translation"):
                translated[name] = df
                continue
            rename = gold_by_source.get(name, {})
            if not rename:
                logger.warning(
                    "SM gold has no entries for source %s; leaving columns raw.",
                    name,
                )
                translated[name] = df
                continue
            keep_cols = [c for c in df.columns if c in rename]
            new_df = df[keep_cols].rename(columns=rename)
            # Preserve attrs.
            new_df.attrs = dict(df.attrs)
            new_df.attrs["needs_sm_column_translation"] = False
            translated[name] = new_df
            logger.info(
                "Translated %s columns via SM gold "
                "(%d cols renamed; %d cols dropped).",
                name,
                len(rename),
                len(df.columns) - len(keep_cols),
            )
        state.bundle.sources = translated

        # Rewrite the in-memory SM gold so source_column == target_column
        # (identity). Downstream committees (NormCommitteeRunner builds
        # an ``{(source, canonical_attribute): [source_col]}`` index from
        # the gold and reads bundle.sources[source][source_col]); after
        # the translation above, the source DataFrames hold canonical
        # column names, so the gold must point at canonical names too.
        # The original raw-to-canonical gold lives on disk and is the
        # surface the SM committee scored against — that scoring is
        # already complete.
        gold_identity = state.bundle.sm_mapping.copy()
        gold_identity["source_column"] = gold_identity["target_column"]
        # De-duplicate: when the original gold had multiple raw columns
        # mapping to the same canonical attribute (rare), the identity
        # form would have duplicate rows.
        gold_identity = gold_identity.drop_duplicates(
            subset=[
                "source_dataset",
                "source_column",
                "target_dataset",
                "target_column",
            ]
        )
        state.bundle.sm_mapping = gold_identity
        logger.info(
            "Rewrote in-memory SM gold to identity form for downstream "
            "committee scoring (%d rows).",
            len(gold_identity),
        )

    def _maybe_filter_matching_yaml(self, matching_yaml: Path) -> Path:
        """Rewrite ditto_plm's checkpoint_path to the pipeline-isolated
        location. Never silently disables the member.

        Policy (2026-05-28 user directive: no silent dropping):
        - If ``ditto_checkpoint_override`` is set → rewrite
          ``checkpoint_path`` to that location.
        - Else if the YAML's default ``checkpoint_path`` is under
          ``usecases_synthetic/cache/`` → ditto would silently reuse a
          committee-side model. Disable the member with a LOUD warning
          telling the user to pass ``--ditto-checkpoint-override`` or
          run the retrain script.
        - Else (YAML already points outside usecases_synthetic/cache/)
          → pass through, ditto runs as configured.
        """
        import yaml as _yaml

        raw = _yaml.safe_load(matching_yaml.read_text()) or {}
        members = raw.get("members") or []
        rewrote = False
        for member in members:
            if member.get("name") != "ditto_plm":
                continue
            params = member.setdefault("matcher", {}).setdefault("params", {})
            current_ckpt = params.get("checkpoint_path", "")
            if self.ditto_checkpoint_override is not None:
                params["checkpoint_path"] = str(self.ditto_checkpoint_override)
                rewrote = True
                logger.info(
                    "ditto_plm checkpoint_path rewritten to %s (pipeline-isolated).",
                    self.ditto_checkpoint_override,
                )
            elif "usecases_synthetic/cache" in str(current_ckpt):
                member["enabled_by_default"] = False
                rewrote = True
                logger.error(
                    "ditto_plm checkpoint_path points to %s, which is under "
                    "usecases_synthetic/cache/. The no-model-reuse policy "
                    "requires a pipeline-isolated checkpoint. DROPPING ditto_plm "
                    "for this run. Retrain via "
                    "`python usecases_synthetic/scripts/ditto/train.py "
                    "--domain products --output-dir pipelines/products/checkpoints/em_matching/ditto/runs` "
                    "then re-run with `--ditto-checkpoint-override "
                    "pipelines/products/checkpoints/em_matching/ditto/runs/best`.",
                    current_ckpt,
                )
        if not rewrote:
            return matching_yaml

        out_dir = (
            (self.out_dir / "effective_committees")
            if self.out_dir is not None
            else Path("/tmp/bob_effective_committees")
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        rewritten = out_dir / matching_yaml.name
        rewritten.write_text(_yaml.safe_dump(raw, sort_keys=False))
        return rewritten

    def _maybe_filter_fusion_yaml(self, fusion_yaml: Path) -> Path:
        """Drop LLM-backed fusion C12 members when ``with_llm=False``.

        **Important:** the C12 ``FusionCommitteeRunner`` ignores
        ``enabled_by_default`` on individual members — it iterates the
        full ``members:`` list regardless. So filtering by flipping
        the flag is a no-op (caused the v3–v7c "fusion hang at member
        2" which was actually ``llm_only`` silently running 5000+ LLM
        calls). To actually drop a member we **remove** it from the
        ``members:`` list.

        The canonical file under
        ``usecases_synthetic/config/committees/`` is never modified.
        """
        if self.with_llm_fusion and self.fusion_members is None:
            return fusion_yaml

        import yaml as _yaml

        raw = _yaml.safe_load(fusion_yaml.read_text()) or {}
        members = raw.get("members") or []
        rewrote = False
        llm_member_names = {"llm_only"}
        kept: list[dict[str, Any]] = []
        for member in members:
            name = member.get("name")
            drop_for_llm = (not self.with_llm_fusion) and (name in llm_member_names)
            drop_for_allowlist = (
                self.fusion_members is not None and name not in self.fusion_members
            )
            if drop_for_llm:
                rewrote = True
                logger.warning(
                    "with_llm_fusion=False: REMOVING fusion member %s from "
                    "roster (C12 runner ignores enabled_by_default).",
                    name,
                )
                continue
            if drop_for_allowlist:
                rewrote = True
                logger.warning(
                    "fusion_members allowlist: REMOVING fusion member %s from "
                    "roster (not in {%s}).",
                    name,
                    ", ".join(sorted(self.fusion_members)),
                )
                continue
            kept.append(member)

        if not rewrote:
            return fusion_yaml

        raw["members"] = kept
        out_dir = (
            (self.out_dir / "effective_committees")
            if self.out_dir is not None
            else Path("/tmp/bob_effective_committees")
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        rewritten = out_dir / fusion_yaml.name
        rewritten.write_text(_yaml.safe_dump(raw, sort_keys=False))
        return rewritten

    def _maybe_filter_blocking_yaml(self, blocking_yaml: Path) -> Path:
        """Drop ``sc_block`` from the blocking roster when the no-reuse
        policy is in effect.

        Same policy as ditto_plm: rewrite the checkpoint_path to the
        pipeline-isolated location when override is set; LOUDLY disable
        the member when the YAML default is under
        ``usecases_synthetic/cache/`` and no override was supplied.
        """
        import yaml as _yaml

        raw = _yaml.safe_load(blocking_yaml.read_text()) or {}
        members = raw.get("members") or []
        rewrote = False
        for member in members:
            if member.get("name") != "sc_block":
                continue
            params = member.setdefault("blocker", {}).setdefault("params", {})
            current_ckpt = params.get("checkpoint_path", "")
            if self.sc_block_checkpoint_override is not None:
                params["checkpoint_path"] = str(self.sc_block_checkpoint_override)
                rewrote = True
                logger.info(
                    "sc_block checkpoint_path rewritten to %s (pipeline-isolated).",
                    self.sc_block_checkpoint_override,
                )
            elif "usecases_synthetic/cache" in str(current_ckpt):
                member["enabled_by_default"] = False
                rewrote = True
                logger.error(
                    "sc_block checkpoint_path points to %s, which is under "
                    "usecases_synthetic/cache/. The no-model-reuse policy "
                    "requires a pipeline-isolated checkpoint. DROPPING sc_block "
                    "for this run. Retrain via "
                    "`python usecases_synthetic/scripts/sc_block/train.py "
                    "--domain products --output-dir pipelines/products/checkpoints/em_blocking/sc_block/` "
                    "then re-run with `--sc-block-checkpoint-override "
                    "pipelines/products/checkpoints/em_blocking/sc_block/best`.",
                    current_ckpt,
                )
        if not rewrote:
            return blocking_yaml

        out_dir = (
            (self.out_dir / "effective_committees")
            if self.out_dir is not None
            else Path("/tmp/bob_effective_committees")
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        rewritten = out_dir / blocking_yaml.name
        rewritten.write_text(_yaml.safe_dump(raw, sort_keys=False))
        return rewritten

    def _compute_panel(self, state: PipelineState) -> E2EPanel:
        """Compute the e2e metric panel."""
        # Both load_workflow_silver and compute_e2e_panel expect
        # source_prefix_map as {prefix: source_name}, matching the YAML
        # shape. The default for "products" in PyDI ships the upstream
        # (alternate/buy/ebay/newegg) scheme; the synthetic-side
        # products tree uses (products_1_/products_2_/...) so we pass
        # the explicit map from config.
        prefix_map = self.config.source_prefix_map or None
        # Products canonical: tree has fusion CSVs (no test_set.xml).
        # Use the canonical-CSV silver builder; other domains keep the
        # XML loader.
        if (
            self.config.domain == "products"
            and self.config.bundle_source == "canonical"
        ):
            from .canonical_loader import load_canonical_products_workflow_silver

            gold = load_canonical_products_workflow_silver()
        else:
            gold = load_workflow_silver(
                state.bundle.variant_root,
                domain=self.config.domain,
                prefix_map=prefix_map,
            )

        sources_pipe = [df for df in state.bundle.sources.values()]

        # Filter column_types to columns the fused output actually has —
        # avoids spurious schema_diff entries for declared-but-absent
        # columns.
        fused_cols = set(state.fused.columns)
        col_types = {
            k: v for k, v in self.config.column_types.items() if k in fused_cols
        }

        logger.info(
            "Computing e2e panel: pipe_fused=%d rows, gold=%d clusters",
            len(state.fused),
            len(gold.fused),
        )

        # The fusion engine's output uses ``_id`` as the canonical fused
        # row id (per the runner). Build a pipe_membership where each
        # row's cluster_id matches the fused frame's ``_id`` so the panel
        # can align them. Without this, the auto-built membership uses
        # ``group_N`` ids that don't match the source-id-derived ``_id``,
        # which collapses Tier 4 (value_correctness) to 0.
        pipe_id_column = "_id"
        pipe_membership = self._build_pipe_membership_from_fused(state)

        return compute_e2e_panel(
            pipe_fused=state.fused,
            correspondences_pipe=state.correspondences,
            sources_pipe=sources_pipe,
            gold=gold,
            column_types=col_types,
            pipe_id_column=pipe_id_column,
            gold_id_column="cluster_id",
            pipe_membership=pipe_membership,
            numerical_tolerance=self.config.panel_tolerance_default,
            numerical_tolerance_overrides=self.config.panel_tolerance_overrides,
            composite_weights=self.config.composite_weights or None,
            source_prefix_map=self.config.source_prefix_map or None,
            usecase=self.config.domain,
            gold_source_label="fusion_test_set.xml",
        )

    def _build_pipe_membership_from_fused(self, state: PipelineState) -> "pd.DataFrame":
        """Build a long-form ``(record_id, source, cluster_id)`` table where
        ``cluster_id`` matches the fused frame's ``_id``.

        The fused output stores group membership in the
        ``_fusion_sources`` column (list of source ``_id``s) and
        ``_fusion_source_datasets`` (parallel list of source-dataset
        names). For each fused row, we emit one membership row per
        listed source. The cluster id is the fused row's ``_id``.
        """
        import ast as _ast

        import pandas as pd

        fused = state.fused
        if fused is None or fused.empty:
            return pd.DataFrame(columns=["record_id", "source", "cluster_id"])

        def _coerce_list(value: Any) -> list[Any]:
            if isinstance(value, list):
                return value
            if isinstance(value, str):
                try:
                    return _ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    return [value]
            return [value] if value is not None else []

        rows: list[dict[str, Any]] = []
        for _, fused_row in fused.iterrows():
            cluster_id = str(fused_row["_id"])
            sources = _coerce_list(fused_row.get("_fusion_sources"))
            datasets = _coerce_list(fused_row.get("_fusion_source_datasets"))
            # Pad datasets to match sources length when datasets is missing.
            if len(datasets) != len(sources):
                datasets = list(datasets) + ["unknown"] * (len(sources) - len(datasets))
            for record_id, source in zip(sources, datasets, strict=False):
                rows.append(
                    {
                        "record_id": str(record_id),
                        "source": str(source),
                        "cluster_id": cluster_id,
                    }
                )
        return pd.DataFrame(rows, columns=["record_id", "source", "cluster_id"])


__all__ = [
    "BestOfBreedPipeline",
    "PipelineConfig",
    "PipelineRunResult",
]
