#!/usr/bin/env python3
"""Aggregate a pipeline run's per-stage JSONs + e2e_panel into a single
``final_report.md`` summarising the whole run for a reviewer.

Reads from ``--run-dir`` (the output dir of ``run_best_of_breed.py``):
- ``stage_<n>_<stage>_selection.json``
- ``per_stage_summary.csv``
- ``e2e_panel/panel.json``
- ``e2e_panel/composite_score.json``
- ``comparison.md`` (if present from ``compare_to_human_baseline.py``)
- ``summary.md`` (terse stage table written by report.py)

Emits ``<run-dir>/final_report.md`` formatted for a human reviewer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def _read_text(path: Path) -> str:
    return path.read_text() if path.exists() else ""


def _stage_json_paths(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("stage_*_selection.json"))


def write_final_report(run_dir: Path, *, domain: str) -> Path:
    panel_dir = run_dir / "e2e_panel"
    panel = _read_json(panel_dir / "panel.json")
    composite = _read_json(panel_dir / "composite_score.json")

    lines: list[str] = []
    lines.append(f"# Best-of-breed pipeline — final report ({domain})")
    lines.append("")
    lines.append(f"- Run dir: `{run_dir}`")
    lines.append(f"- Mode: {panel.get('run_id') or 'see run command'}")
    lines.append("")

    # --- Stage table --------------------------------------------------------
    lines.append("## Per-stage winners")
    lines.append("")
    lines.append("| # | Stage | Winner | Metric | Val | Test | Runtime (s) |")
    lines.append("|---|---|---|---|---|---|---|")
    for i, path in enumerate(_stage_json_paths(run_dir), start=1):
        data = _read_json(path)
        lines.append(
            f"| {i} | {data.get('stage')} | `{data.get('winner') or '—'}` | "
            f"{data.get('metric_key', '?')} | "
            f"{data.get('val_score', float('nan')):.4f} | "
            f"{data.get('test_score', float('nan')):.4f} | "
            f"{data.get('runtime_s', 0.0):.1f} |"
        )
    lines.append("")

    # --- Headline panel -----------------------------------------------------
    if composite:
        lines.append("## End-to-end metric panel")
        lines.append("")
        cs = composite.get("composite_score")
        if cs is not None:
            lines.append(f"- **composite_score:** {cs:.4f}")
        for name, sub in (composite.get("subscores") or {}).items():
            try:
                lines.append(f"- **{name}:** {sub:.4f}")
            except (TypeError, ValueError):
                lines.append(f"- **{name}:** {sub}")
        lines.append("")
        if composite.get("caveat"):
            lines.append(f"> {composite.get('caveat')}")
            lines.append("")

    # --- Diagnostic warnings ------------------------------------------------
    warnings = panel.get("warnings") or []
    if warnings:
        lines.append("## Panel warnings")
        lines.append("")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")

    # --- Notebook comparison (if available) ---------------------------------
    cmp_path = run_dir / "comparison.md"
    if cmp_path.exists():
        lines.append("## Comparison vs human baseline")
        lines.append("")
        lines.append(f"See [`comparison.md`]({cmp_path.name}).")
        lines.append("")

    # --- Provenance ---------------------------------------------------------
    lines.append("## Provenance + caveats")
    lines.append("")
    lines.append(
        "- **No model reuse from `usecases_synthetic/cache/`.** ditto_plm + "
        "sc_block dropped at instantiation by the orchestrator's YAML "
        "in-memory filters; their checkpoints under "
        "`usecases_synthetic/cache/{ditto,sc_block}_checkpoints/` were not "
        "consumed."
    )
    lines.append(
        "- **Sweep mode (`--mode sweep`)** runs chained HP sweeps where the "
        "harness implementation is complete (SM, refinement). Other stages "
        "(Norm, EM blocking, EM matching, Fusion) run the existing committee "
        "runners with YAML-locked defaults pending the per-stage sweep "
        "harness completion."
    )
    lines.append(
        "- Greedy per-stage selection is locally optimal; cross-stage "
        "joint search is out of scope for v1."
    )
    lines.append("")

    out = run_dir / "final_report.md"
    out.write_text("\n".join(lines) + "\n")
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--domain", default="products")
    args = p.parse_args()
    out = write_final_report(args.run_dir, domain=args.domain)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
