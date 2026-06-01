"""JSON + markdown writers for validation metrics.

Thin adapters so M5, M7, M8, M9, M10 can dump a consistent on-disk
format without each script reinventing the layout. The JSON format is
round-trippable with :mod:`usecases_synthetic.lib.baseline_loader`.

The markdown output intentionally keeps the rendering minimal — no
HTML, no plotting. Just per-stage and per-attribute tables so the
rollup is human-readable.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .validation_metrics import collapse_flag, delta


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_metrics_json(
    path: Path,
    domain: str,
    per_stage: Mapping[str, Mapping[str, Any]],
    *,
    meta: Mapping[str, Any] | None = None,
) -> Path:
    """Serialise a metrics payload to JSON.

    The on-disk shape is::

        {
          "domain": "...",
          "meta": { ... },
          "per_stage": {
            "sm":     { "aggregated": {...}, "per_attribute": {...}, ... },
            "em":     { ... },
            "fusion": { ... }
          }
        }

    Parameters
    ----------
    path : Path
        Destination path. Parent directories are created.
    domain : str
        Domain name.
    per_stage : mapping
        Per-stage committee result dicts (as produced by
        :meth:`usecases_synthetic.lib.committee.CommitteeResult.as_dict`).
    meta : mapping, optional
        Extra metadata (timestamp, git SHA, runtime, roster version).
        Defaults to ``{"written_at": <utc isoformat>}``.

    Returns
    -------
    Path
        The written path.
    """
    _ensure_parent(path)

    meta_out: dict[str, Any] = {"written_at": datetime.now(timezone.utc).isoformat()}
    if meta:
        meta_out.update(meta)

    payload: dict[str, Any] = {
        "domain": domain,
        "meta": meta_out,
        "per_stage": {stage: dict(block) for stage, block in per_stage.items()},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    return path


def _json_default(obj: Any) -> Any:
    """JSON fallback for numpy scalars and sets."""
    try:
        import numpy as np

        if isinstance(obj, np.generic):
            return obj.item()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def _format_float(value: Any) -> str:
    """Render a float with 4 decimal places, pass-through otherwise."""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _aggregated_table(
    stage: str,
    aggregated: Mapping[str, float],
    baseline_aggregated: Mapping[str, float] | None,
) -> list[str]:
    """Render the per-stage aggregated table (with optional delta column)."""
    lines: list[str] = [f"### {stage.upper()} - aggregated"]
    if baseline_aggregated is not None:
        lines.append("| metric | measured | baseline | delta |")
        lines.append("|---|---|---|---|")
        diffs = delta(baseline_aggregated, aggregated)
        for metric in sorted(aggregated):
            lines.append(
                "| "
                + " | ".join(
                    [
                        metric,
                        _format_float(aggregated[metric]),
                        _format_float(baseline_aggregated.get(metric, 0.0)),
                        _format_float(diffs.get(metric, 0.0)),
                    ]
                )
                + " |"
            )
        collapsed = collapse_flag(aggregated, baseline_aggregated)
        lines.append("")
        lines.append(f"**Collapse flag:** {'YES' if collapsed else 'no'}")
    else:
        lines.append("| metric | value |")
        lines.append("|---|---|")
        for metric in sorted(aggregated):
            lines.append(f"| {metric} | {_format_float(aggregated[metric])} |")
    lines.append("")
    return lines


def _attribute_table(
    stage: str,
    per_attribute: Mapping[str, Mapping[str, float]],
) -> list[str]:
    """Render the per-attribute table."""
    if not per_attribute:
        return []
    columns: set[str] = set()
    for metrics in per_attribute.values():
        columns.update(metrics.keys())
    ordered = sorted(columns)

    lines: list[str] = [f"### {stage.upper()} - per attribute"]
    lines.append("| attribute | " + " | ".join(ordered) + " |")
    lines.append("|" + "|".join(["---"] * (len(ordered) + 1)) + "|")
    for attr in sorted(per_attribute):
        row = [attr]
        for metric in ordered:
            row.append(_format_float(per_attribute[attr].get(metric, "")))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return lines


def write_report_md(
    path: Path,
    domain: str,
    per_stage: Mapping[str, Mapping[str, Any]],
    *,
    baseline_per_stage: Mapping[str, Mapping[str, Any]] | None = None,
    title: str | None = None,
) -> Path:
    """Render a per-stage markdown rollup.

    Parameters
    ----------
    path : Path
        Destination path. Parent directories are created.
    domain : str
        Domain name (used in the header).
    per_stage : mapping
        Per-stage committee result dicts.
    baseline_per_stage : mapping, optional
        When supplied, an extra delta column and a collapse flag are
        rendered per stage.
    title : str, optional
        Override the top-level ``#`` title. Default ``"Validation
        report — <domain>"``.

    Returns
    -------
    Path
        The written path.
    """
    _ensure_parent(path)

    lines: list[str] = []
    lines.append(f"# {title or f'Validation report - {domain}'}")
    lines.append("")
    lines.append(f"_Generated at {datetime.now(timezone.utc).isoformat()}_")
    lines.append("")

    # Stage iteration order: SM → Norm → EM blocking → EM matching → Fusion.
    # The 2026-05-13 EM stage split (plan_revision.md C10/C11) replaced
    # the single ``em`` stage with ``em_blocking`` + ``em_matching``;
    # ``em`` is retained as a legacy alias for any prior baseline_metrics.json
    # files still on disk.
    for stage in ("sm", "norm", "em_blocking", "em_matching", "em", "fusion"):
        block = per_stage.get(stage)
        if not block:
            continue
        aggregated: Mapping[str, float] = block.get("aggregated", {})
        per_attribute: Mapping[str, Mapping[str, float]] = block.get(
            "per_attribute", {}
        )

        baseline_agg: Mapping[str, float] | None = None
        if baseline_per_stage is not None:
            baseline_block = baseline_per_stage.get(stage, {})
            raw_baseline = baseline_block.get("aggregated", {})
            if raw_baseline:
                baseline_agg = raw_baseline

        lines.append(f"## Stage: {stage}")
        lines.append("")
        lines.extend(_aggregated_table(stage, aggregated, baseline_agg))
        lines.extend(_attribute_table(stage, per_attribute))

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")
    return path
