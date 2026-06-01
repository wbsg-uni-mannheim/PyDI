"""Artifact writers for the best-of-breed pipeline.

Writes per-stage selection JSON files, a per-stage summary CSV, the
fused output + correspondences, and the e2e panel artifacts under
the run output directory.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from .pipeline import PipelineConfig, PipelineRunResult
from .stage_runners import StageSelection

logger = logging.getLogger(__name__)


def write_run_artifacts(
    result: PipelineRunResult,
    *,
    out_dir: Path,
    config: PipelineConfig,
) -> None:
    """Write the full set of artifacts for one pipeline run.

    Parameters
    ----------
    result : PipelineRunResult
        The pipeline output.
    out_dir : Path
        Directory to write under. Created if absent.
    config : PipelineConfig
        Source config (for the summary header).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-stage selection JSONs
    for i, sel in enumerate(result.stage_selections, start=1):
        path = out_dir / f"stage_{i}_{sel.stage}_selection.json"
        _write_json(path, sel.as_dict())

    # Per-stage summary CSV
    _write_summary_csv(out_dir / "per_stage_summary.csv", result.stage_selections)

    # Fused output + correspondences
    if result.state.fused is not None and not result.state.fused.empty:
        result.state.fused.to_csv(out_dir / "fused.csv", index=False)
    if (
        result.state.correspondences is not None
        and not result.state.correspondences.empty
    ):
        result.state.correspondences.to_csv(
            out_dir / "correspondences.csv", index=False
        )

    # e2e panel
    if result.panel is not None:
        result.panel.write(out_dir / "e2e_panel")

    # Run-level summary markdown
    _write_summary_md(
        out_dir / "summary.md",
        config=config,
        result=result,
    )

    logger.info("Wrote pipeline artifacts to %s", out_dir)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON file with stable formatting."""
    with path.open("w") as f:
        json.dump(payload, f, indent=2, default=str, sort_keys=False)


def _write_summary_csv(path: Path, selections: list[StageSelection]) -> None:
    """One row per stage: winner, val, test, runtime."""
    rows = [
        {
            "stage": sel.stage,
            "winner": sel.winner,
            "metric_key": sel.metric_key,
            "val_score": sel.val_score,
            "test_score": sel.test_score,
            "runtime_s": sel.runtime_s,
        }
        for sel in selections
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_summary_md(
    path: Path,
    *,
    config: PipelineConfig,
    result: PipelineRunResult,
) -> None:
    """Human-readable run summary."""
    lines: list[str] = []
    lines.append(f"# Best-of-breed pipeline run — {config.domain}")
    lines.append("")
    lines.append(f"Total runtime: {result.total_runtime_s:.1f} s")
    lines.append("")
    lines.append("## Per-stage winners")
    lines.append("")
    lines.append("| Stage | Winner | Metric | Val score | Test score | Runtime (s) |")
    lines.append("|---|---|---|---|---|---|")
    for sel in result.stage_selections:
        lines.append(
            f"| {sel.stage} | `{sel.winner}` | {sel.metric_key} | "
            f"{sel.val_score:.4f} | {sel.test_score:.4f} | {sel.runtime_s:.1f} |"
        )
    lines.append("")

    # Panel headline (if available)
    if result.panel is not None:
        lines.append("## End-to-end metric panel")
        lines.append("")
        composite = result.panel.composite.get("composite_score")
        if composite is not None:
            lines.append(f"- **Composite score:** {composite:.4f}")
        for tier_name, subscore in (
            result.panel.composite.get("tier_subscores") or {}
        ).items():
            lines.append(f"- **{tier_name}:** {subscore:.4f}")
        lines.append("")

        warnings = result.panel.warnings or []
        if warnings:
            lines.append("## Panel warnings")
            lines.append("")
            for w in warnings:
                lines.append(f"- {w}")
            lines.append("")

    lines.append("## Caveats")
    lines.append("")
    lines.append(
        "- Greedy per-stage selection is locally optimal; no joint search "
        "across stages. See `plans/plan_best_of_breed_pipeline.md` §8.2."
    )
    norm_sel = next((s for s in result.stage_selections if s.stage == "norm"), None)
    if norm_sel is not None and norm_sel.notes.get("vacuous"):
        lines.append(
            "- **Norm selection was vacuous** "
            f"(spread={norm_sel.notes.get('spread', 0.0):.4f} < epsilon). "
            "Norm members produced near-identical outputs on this input."
        )

    path.write_text("\n".join(lines) + "\n")


__all__ = ["write_run_artifacts"]
