#!/usr/bin/env python3
"""M9 — per-knob ablation analyzer.

Consumes the baseline metrics (M5), the per-level hard metrics (M7),
and the per-knob ablation metrics produced by
``run_ablation_validation.py``, then answers:

1. Did each knob move its *primary-stage* metric in the direction the
   knob card predicts?
2. Did each knob leave the *non-primary* stages approximately flat?
3. Is the ablation's primary-stage effect comparable to the full-hard
   variant's primary-stage effect (under-signal / over-signal flags)?

Writes

- ``usecases_synthetic/validation/<domain>/ablation/ablation_report.md``
- ``usecases_synthetic/validation/<domain>/ablation/ablation_report.csv``

M9 surfaces problems. M10 does the triage. No knob re-configuration
happens here.

Usage
-----
::

    python usecases_synthetic/scripts/analyze_ablation.py --domain companies

Assumes ``run_ablation_validation.py`` has already written per-knob
``metrics.json`` files under
``usecases_synthetic/validation/<domain>/ablation/knob_<id>/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from usecases_synthetic.lib.baseline_loader import baseline_path  # noqa: E402
from usecases_synthetic.lib.domain_config import SYNTHETIC_DIR  # noqa: E402
from usecases_synthetic.lib.monotonicity import (  # noqa: E402
    DEFAULT_FLAT_TOLERANCE,
    Direction,
    SignalExpectation,
    load_knob_expected_signals,
    resolve_metric,
)

logger = logging.getLogger(__name__)

VALIDATION_DIR: Path = SYNTHETIC_DIR / "validation"
EXPECTATIONS_YAML: Path = SYNTHETIC_DIR / "config" / "knob_expected_signals.yaml"

# Flag thresholds.
DEFAULT_LEAKAGE_TOLERANCE: float = 0.05
DEFAULT_UNDER_SIGNAL_RATIO: float = 0.5
DEFAULT_OVER_SIGNAL_RATIO: float = 1.1
DEFAULT_MIN_PRIMARY_DELTA: float = 0.02


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class AblationSignal:
    """Outcome of evaluating a single knob's expectation on its ablation.

    Parameters
    ----------
    expectation : SignalExpectation
        The signal prediction from ``knob_expected_signals.yaml``.
    is_primary : bool
        ``True`` iff ``expectation.stage`` equals the knob's
        ``primary_stage`` field from the YAML.
    baseline : float
        Metric value on the baseline variant.
    ablation : float
        Metric value on this knob's ablation variant.
    hard : float
        Metric value on the full-hard variant (for over/under checks).
        ``math.nan`` when the hard metrics file is missing.
    delta_vs_baseline : float
        ``ablation - baseline``. ``math.nan`` on missing data.
    hard_delta_vs_baseline : float
        ``hard - baseline``. ``math.nan`` on missing data.
    direction_match : bool
        ``True`` iff the observed delta respects the predicted direction
        (with ``flat_tolerance`` slack for ``"flat"``).
    flags : list[str]
        Interaction flags, any subset of ``{"cross_stage_leakage",
        "primary_under_signal", "primary_over_signal",
        "direction_mismatch"}``.
    reason : str
        Short human-readable explanation.
    """

    expectation: SignalExpectation
    is_primary: bool
    baseline: float
    ablation: float
    hard: float
    delta_vs_baseline: float
    hard_delta_vs_baseline: float
    direction_match: bool
    flags: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class KnobAblation:
    """Aggregated ablation result for a single knob."""

    knob: str
    label: str
    primary_stage: str
    signals: list[AblationSignal]
    metrics_path: Path
    source_card: str

    def any_flag(self, name: str) -> bool:
        """Return ``True`` if any signal has ``name`` in its flags."""
        return any(name in sig.flags for sig in self.signals)

    def primary_signals(self) -> list[AblationSignal]:
        """Return the signals tagged as primary-stage."""
        return [s for s in self.signals if s.is_primary]

    def secondary_signals(self) -> list[AblationSignal]:
        """Return the signals tagged as non-primary-stage."""
        return [s for s in self.signals if not s.is_primary]


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict[str, Any]:
    """Read ``path`` as JSON; raise ``FileNotFoundError`` if missing."""
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    with open(path, encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
    return data


def ablation_metrics_path(domain: str, knob_id: str) -> Path:
    """Canonical per-knob ablation metrics path."""
    return VALIDATION_DIR / domain / "ablation" / knob_id / "metrics.json"


def hard_metrics_path(domain: str) -> Path:
    """Canonical full-hard metrics path written by M7."""
    return VALIDATION_DIR / domain / "hard" / "metrics.json"


def _load_primary_stages(expectations_path: Path) -> dict[str, str]:
    """Read only the ``primary_stage`` mapping from the expectations YAML."""
    import yaml

    with open(expectations_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    out: dict[str, str] = {}
    for key, block in raw.items():
        if not isinstance(block, Mapping) or not key.startswith("knob_"):
            continue
        stage = block.get("primary_stage")
        if isinstance(stage, str):
            out[key] = stage
    return out


def _stage_block(metrics: Mapping[str, Any], stage: str) -> Mapping[str, Any]:
    """Return the ``per_stage[stage]`` block or an empty dict."""
    per_stage = metrics.get("per_stage", {}) or {}
    block = per_stage.get(stage, {}) or {}
    if not isinstance(block, Mapping):
        return {}
    return block


# ---------------------------------------------------------------------------
# Signal evaluation
# ---------------------------------------------------------------------------


def _direction_matches(
    delta: float,
    direction: Direction,
    *,
    flat_tolerance: float,
) -> bool:
    """Return ``True`` iff ``delta`` respects ``direction``."""
    if math.isnan(delta):
        return False
    if direction == "down":
        return delta <= flat_tolerance  # allow near-zero as weakly down
    if direction == "up":
        return delta >= -flat_tolerance
    if direction == "flat":
        return abs(delta) <= flat_tolerance
    raise ValueError(f"Unknown direction: {direction!r}")


def _compute_flags(
    exp: SignalExpectation,
    *,
    is_primary: bool,
    delta: float,
    hard_delta: float,
    flat_tolerance: float,
    under_ratio: float,
    over_ratio: float,
    min_primary_delta: float,
) -> list[str]:
    """Decide which interaction flags apply to one signal."""
    flags: list[str] = []
    if math.isnan(delta):
        return flags

    # Direction mismatch (explicit sign disagreement, only for non-flat).
    if exp.direction == "down" and delta > flat_tolerance:
        flags.append("direction_mismatch")
    elif exp.direction == "up" and delta < -flat_tolerance:
        flags.append("direction_mismatch")
    elif exp.direction == "flat" and abs(delta) > flat_tolerance:
        # For a non-primary "flat" expectation, flat violation is cross-stage
        # leakage rather than a generic mismatch — handled below.
        if is_primary:
            flags.append("direction_mismatch")

    # Cross-stage leakage: non-primary signals expected flat (or any
    # direction) but moving more than flat_tolerance.
    if not is_primary and exp.direction == "flat" and abs(delta) > flat_tolerance:
        flags.append("cross_stage_leakage")

    # Under / over signal: only meaningful for primary-stage signals
    # with a monotone prediction and a valid hard_delta reference.
    if is_primary and exp.direction in ("down", "up") and not math.isnan(hard_delta):
        # Use absolute magnitudes. Compare ablation to full-hard
        # displacement. Both should be in the same direction in the
        # normal case; we still flag on magnitude.
        abs_delta = abs(delta)
        abs_hard = abs(hard_delta)
        if abs_hard >= min_primary_delta:
            if abs_delta < under_ratio * abs_hard:
                flags.append("primary_under_signal")
            if abs_delta > over_ratio * abs_hard:
                flags.append("primary_over_signal")
        else:
            # Full-hard barely moved this metric — report the ablation
            # magnitude in isolation.
            if abs_delta < min_primary_delta:
                flags.append("primary_under_signal")

    return flags


def _fmt_delta(delta: float) -> str:
    """Human-friendly delta with explicit sign."""
    if math.isnan(delta):
        return "NaN"
    return f"{delta:+.3f}"


def evaluate_knob_ablation(
    domain: str,
    knob_id: str,
    expectations: list[SignalExpectation],
    primary_stage: str,
    baseline_metrics: Mapping[str, Any],
    hard_metrics: Mapping[str, Any] | None,
    *,
    flat_tolerance: float = DEFAULT_FLAT_TOLERANCE,
    under_ratio: float = DEFAULT_UNDER_SIGNAL_RATIO,
    over_ratio: float = DEFAULT_OVER_SIGNAL_RATIO,
    min_primary_delta: float = DEFAULT_MIN_PRIMARY_DELTA,
) -> KnobAblation:
    """Load a knob's ablation metrics and evaluate each expectation.

    Parameters
    ----------
    domain : str
        Domain name.
    knob_id : str
        Canonical knob id (e.g. ``"knob_08"``).
    expectations : list of SignalExpectation
        Expectations for this knob from the YAML.
    primary_stage : str
        The knob's primary stage (``"sm"``, ``"em"``, or ``"fusion"``).
    baseline_metrics, hard_metrics : mapping
        Full ``metrics.json`` payloads. ``hard_metrics`` may be ``None``
        when the full-hard run has not been produced yet, in which case
        over/under-signal flags fall back to a minimum-magnitude check.
    flat_tolerance, under_ratio, over_ratio, min_primary_delta : float
        Thresholds for the flags. See module docstring.

    Returns
    -------
    KnobAblation
        Populated aggregate. Raises ``FileNotFoundError`` if the knob's
        metrics file is missing.
    """
    path = ablation_metrics_path(domain, knob_id)
    ablation_metrics = _load_json(path)
    source_card = expectations[0].source if expectations else ""

    signals: list[AblationSignal] = []
    for exp in expectations:
        base_block = _stage_block(baseline_metrics, exp.stage)
        abl_block = _stage_block(ablation_metrics, exp.stage)
        hard_block: Mapping[str, Any] = (
            _stage_block(hard_metrics, exp.stage) if hard_metrics else {}
        )

        baseline_val = resolve_metric(base_block, exp.metric)
        ablation_val = resolve_metric(abl_block, exp.metric)
        hard_val = resolve_metric(hard_block, exp.metric) if hard_block else math.nan

        if math.isnan(baseline_val) or math.isnan(ablation_val):
            delta = math.nan
        else:
            delta = ablation_val - baseline_val
        if math.isnan(baseline_val) or math.isnan(hard_val):
            hard_delta = math.nan
        else:
            hard_delta = hard_val - baseline_val

        is_primary = exp.stage == primary_stage
        direction_match = _direction_matches(
            delta, exp.direction, flat_tolerance=flat_tolerance
        )
        flags = _compute_flags(
            exp,
            is_primary=is_primary,
            delta=delta,
            hard_delta=hard_delta,
            flat_tolerance=flat_tolerance,
            under_ratio=under_ratio,
            over_ratio=over_ratio,
            min_primary_delta=min_primary_delta,
        )
        reason = _explain(
            exp=exp,
            is_primary=is_primary,
            baseline=baseline_val,
            ablation=ablation_val,
            delta=delta,
            direction_match=direction_match,
        )
        signals.append(
            AblationSignal(
                expectation=exp,
                is_primary=is_primary,
                baseline=baseline_val,
                ablation=ablation_val,
                hard=hard_val,
                delta_vs_baseline=delta,
                hard_delta_vs_baseline=hard_delta,
                direction_match=direction_match,
                flags=flags,
                reason=reason,
            )
        )

    label = ablation_metrics.get("meta", {}).get("level", knob_id)
    return KnobAblation(
        knob=knob_id,
        label=str(label),
        primary_stage=primary_stage,
        signals=signals,
        metrics_path=path,
        source_card=source_card,
    )


def _explain(
    *,
    exp: SignalExpectation,
    is_primary: bool,
    baseline: float,
    ablation: float,
    delta: float,
    direction_match: bool,
) -> str:
    """Short reason string describing the check outcome."""
    if math.isnan(baseline) or math.isnan(ablation):
        missing: list[str] = []
        if math.isnan(baseline):
            missing.append("baseline")
        if math.isnan(ablation):
            missing.append("ablation")
        return "metric missing: " + ", ".join(missing)
    qualifier = "primary" if is_primary else "secondary"
    verdict = "ok" if direction_match else "off-direction"
    return (
        f"{qualifier} {exp.stage}: {baseline:.3f} -> {ablation:.3f} "
        f"(delta {delta:+.3f}, predicted {exp.direction}, {verdict})"
    )


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def _fmt(value: float) -> str:
    """Format a float, returning ``"NaN"`` on missing."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NaN"
    return f"{value:.3f}"


def _check_symbol(ok: bool) -> str:
    """ASCII-only check / cross glyphs (no emoji per CLAUDE.md)."""
    return "[ok]" if ok else "[!!]"


def _relpath_to_repo(path: Path) -> str:
    """Render ``path`` relative to the repo root when possible."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def build_signal_rows(results: list[KnobAblation]) -> list[dict[str, Any]]:
    """Flatten :class:`KnobAblation` records into CSV-friendly row dicts."""
    rows: list[dict[str, Any]] = []
    for result in results:
        for sig in result.signals:
            exp = sig.expectation
            rows.append(
                {
                    "knob": exp.knob,
                    "signal_id": exp.signal_id,
                    "stage": exp.stage,
                    "is_primary": sig.is_primary,
                    "metric": exp.metric,
                    "direction": exp.direction,
                    "baseline": sig.baseline,
                    "ablation": sig.ablation,
                    "hard": sig.hard,
                    "delta_vs_baseline": sig.delta_vs_baseline,
                    "hard_delta_vs_baseline": sig.hard_delta_vs_baseline,
                    "direction_match": sig.direction_match,
                    "flags": "|".join(sig.flags),
                    "reason": sig.reason,
                    "source_card": exp.source,
                }
            )
    return rows


def render_markdown(
    domain: str,
    results: list[KnobAblation],
    *,
    hard_available: bool,
    knob_order: list[str],
) -> str:
    """Render the per-knob ablation report."""
    lines: list[str] = []
    lines.append(f"# Ablation Report — {domain}")
    lines.append("")
    lines.append(
        "Per-knob ablation validation: each knob set to `hard` with all "
        "others at `easy` (identity). See "
        "`plans/validation/module_09_ablation.md` and "
        "`knobs/ablations.md` for the independent-togglability requirement."
    )
    lines.append("")
    if not hard_available:
        lines.append(
            "> NOTE: full-hard metrics are not available; over- / "
            "under-signal flags fall back to minimum-magnitude checks "
            "instead of ratios against the full-hard displacement."
        )
        lines.append("")

    # ---------- Executive summary ----------
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(
        "| Knob | Primary stage | Primary delta | Direction ok | "
        "Leakage | Under | Over | Mismatch | Card |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for knob in knob_order:
        result = _find(results, knob)
        if result is None:
            lines.append(f"| {knob} | — | — | — | — | — | — | — | missing metrics |")
            continue
        primary = result.primary_signals()
        if primary:
            dominant = max(primary, key=lambda s: abs(s.delta_vs_baseline or 0.0))
            primary_delta = _fmt_delta(dominant.delta_vs_baseline)
            direction_ok = all(s.direction_match for s in primary)
        else:
            primary_delta = "—"
            direction_ok = True
        leakage = result.any_flag("cross_stage_leakage")
        under = result.any_flag("primary_under_signal")
        over = result.any_flag("primary_over_signal")
        mismatch = result.any_flag("direction_mismatch")
        lines.append(
            f"| {knob} | {result.primary_stage} | {primary_delta} | "
            f"{_check_symbol(direction_ok)} | "
            f"{_check_symbol(not leakage)} | "
            f"{_check_symbol(not under)} | "
            f"{_check_symbol(not over)} | "
            f"{_check_symbol(not mismatch)} | "
            f"[card]({result.source_card}) |"
        )
    lines.append("")

    # ---------- Per-signal table ----------
    lines.append("## Per-Signal Results")
    lines.append("")
    lines.append(
        "| Knob | Signal | Stage | Primary | Metric | Dir | baseline |"
        " ablation | full-hard | delta | hard delta | Dir ok | Flags |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for knob in knob_order:
        result = _find(results, knob)
        if result is None:
            continue
        for sig in result.signals:
            exp = sig.expectation
            flags_str = ",".join(sig.flags) if sig.flags else "—"
            lines.append(
                f"| {exp.knob} | {exp.signal_id} | {exp.stage} | "
                f"{'yes' if sig.is_primary else 'no'} | "
                f"`{exp.metric}` | {exp.direction} | "
                f"{_fmt(sig.baseline)} | {_fmt(sig.ablation)} | "
                f"{_fmt(sig.hard)} | {_fmt_delta(sig.delta_vs_baseline)} | "
                f"{_fmt_delta(sig.hard_delta_vs_baseline)} | "
                f"{_check_symbol(sig.direction_match)} | {flags_str} |"
            )
    lines.append("")

    # ---------- Interaction flags commentary ----------
    lines.append("## Interaction Flags")
    lines.append("")
    lines.append(
        "- `cross_stage_leakage` — non-primary stage moved more than the "
        "flat tolerance. Usually indicates a variant-packaging bug (e.g. "
        "renamed headers breaking fusion comparators)."
    )
    lines.append(
        "- `primary_under_signal` — primary-stage delta is materially "
        "smaller than the full-hard displacement. Knob may be dominated "
        "by another knob at hard level. Usually fine; log for M10."
    )
    lines.append(
        "- `primary_over_signal` — primary-stage delta exceeds the "
        "full-hard displacement. Indicates cancellation between knobs "
        "at full-hard; worth a scheduling review."
    )
    lines.append(
        "- `direction_mismatch` — primary-stage signal moved opposite "
        "the card's predicted direction. Treat as a knob-authoring bug."
    )
    lines.append("")

    # ---------- Provenance ----------
    lines.append("## Provenance")
    lines.append("")
    lines.append(f"- Domain: {domain}")
    lines.append(f"- Expectations: `{_relpath_to_repo(EXPECTATIONS_YAML)}`")
    lines.append(f"- Baseline: `{_relpath_to_repo(baseline_path(domain))}`")
    if hard_available:
        lines.append(
            f"- Full-hard metrics: `{_relpath_to_repo(hard_metrics_path(domain))}`"
        )
    for result in results:
        lines.append(
            f"- {result.knob} metrics: " f"`{_relpath_to_repo(result.metrics_path)}`"
        )
    lines.append("")
    return "\n".join(lines)


def _find(results: list[KnobAblation], knob: str) -> KnobAblation | None:
    """Return the :class:`KnobAblation` for ``knob`` or ``None``."""
    for result in results:
        if result.knob == knob:
            return result
    return None


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def analyze_domain(
    domain: str,
    *,
    knobs: list[str] | None = None,
    expectations_path: Path = EXPECTATIONS_YAML,
    out_dir: Path | None = None,
    flat_tolerance: float = DEFAULT_FLAT_TOLERANCE,
    under_ratio: float = DEFAULT_UNDER_SIGNAL_RATIO,
    over_ratio: float = DEFAULT_OVER_SIGNAL_RATIO,
    min_primary_delta: float = DEFAULT_MIN_PRIMARY_DELTA,
) -> dict[str, Any]:
    """Run the full M9 analyzer for one domain.

    Parameters
    ----------
    domain : str
        Domain name.
    knobs : list of str, optional
        Canonical knob ids to analyze. Defaults to every knob with
        expectations that also has a metrics.json on disk.
    expectations_path : Path
        Override for ``knob_expected_signals.yaml``.
    out_dir : Path, optional
        Output directory. Defaults to
        ``usecases_synthetic/validation/<domain>/ablation/``.
    flat_tolerance, under_ratio, over_ratio, min_primary_delta : float
        Flag thresholds.

    Returns
    -------
    dict
        ``{"results": [...], "report_md": Path, "report_csv": Path,
        "out_dir": Path}``.

    Raises
    ------
    FileNotFoundError
        If the baseline metrics file is missing.
    """
    knob_expectations = load_knob_expected_signals(expectations_path)
    primary_stages = _load_primary_stages(expectations_path)

    baseline_metrics = _load_json(baseline_path(domain))
    hard_path = hard_metrics_path(domain)
    hard_metrics: dict[str, Any] | None
    if hard_path.exists():
        hard_metrics = _load_json(hard_path)
    else:
        logger.warning(
            "Full-hard metrics not found at %s; over/under-signal flags "
            "will fall back to minimum-magnitude checks.",
            hard_path,
        )
        hard_metrics = None

    if knobs is None:
        candidates = list(knob_expectations.keys())
    else:
        candidates = list(knobs)

    results: list[KnobAblation] = []
    missing: list[str] = []
    for knob_id in candidates:
        exps = knob_expectations.get(knob_id, [])
        if not exps:
            logger.warning("No expectations for %s; skipping", knob_id)
            continue
        primary = primary_stages.get(knob_id, "")
        if not primary:
            logger.warning("No primary_stage for %s; skipping", knob_id)
            continue
        try:
            result = evaluate_knob_ablation(
                domain=domain,
                knob_id=knob_id,
                expectations=exps,
                primary_stage=primary,
                baseline_metrics=baseline_metrics,
                hard_metrics=hard_metrics,
                flat_tolerance=flat_tolerance,
                under_ratio=under_ratio,
                over_ratio=over_ratio,
                min_primary_delta=min_primary_delta,
            )
        except FileNotFoundError as exc:
            logger.warning("Skipping %s: %s", knob_id, exc)
            missing.append(knob_id)
            continue
        results.append(result)

    target_dir = out_dir or (VALIDATION_DIR / domain / "ablation")
    target_dir.mkdir(parents=True, exist_ok=True)

    csv_path = target_dir / "ablation_report.csv"
    _write_signal_csv(csv_path, results)

    md_path = target_dir / "ablation_report.md"
    md_path.write_text(
        render_markdown(
            domain,
            results,
            hard_available=hard_metrics is not None,
            knob_order=candidates,
        ),
        encoding="utf-8",
    )

    return {
        "results": results,
        "report_md": md_path,
        "report_csv": csv_path,
        "out_dir": target_dir,
        "missing": missing,
    }


def _write_signal_csv(path: Path, results: list[KnobAblation]) -> None:
    """Write the per-signal CSV."""
    rows = build_signal_rows(results)
    fieldnames = [
        "knob",
        "signal_id",
        "stage",
        "is_primary",
        "metric",
        "direction",
        "baseline",
        "ablation",
        "hard",
        "delta_vs_baseline",
        "hard_delta_vs_baseline",
        "direction_match",
        "flags",
        "reason",
        "source_card",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_knobs(raw: str | None) -> list[str] | None:
    """Parse a comma-separated knob list into canonical ids."""
    if raw is None:
        return None
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    out: list[str] = []
    for tok in tokens:
        if tok.startswith("knob_"):
            out.append(tok)
        else:
            try:
                out.append(f"knob_{int(tok):02d}")
            except ValueError as exc:
                raise ValueError(f"Invalid knob token {tok!r}") from exc
    return out


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(
        description=(
            "Per-knob ablation analyzer (M9). Reads per-knob metrics "
            "written by run_ablation_validation.py, compares against "
            "baseline and full-hard, and writes ablation_report.md + .csv."
        )
    )
    parser.add_argument("--domain", required=True, help="Domain name.")
    parser.add_argument(
        "--knobs",
        type=str,
        default=None,
        help=(
            "Comma-separated knob ids to analyze (e.g. '1,8,10'). "
            "Default: all knobs with expectations."
        ),
    )
    parser.add_argument(
        "--expectations",
        type=Path,
        default=EXPECTATIONS_YAML,
        help="Path to knob_expected_signals.yaml.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Override output directory (default: "
            "usecases_synthetic/validation/<domain>/ablation/)."
        ),
    )
    parser.add_argument(
        "--flat-tolerance",
        type=float,
        default=DEFAULT_FLAT_TOLERANCE,
        help="Absolute delta tolerance for flat / leakage checks.",
    )
    parser.add_argument(
        "--under-ratio",
        type=float,
        default=DEFAULT_UNDER_SIGNAL_RATIO,
        help=(
            "Ablation primary delta below this fraction of the "
            "full-hard delta triggers primary_under_signal."
        ),
    )
    parser.add_argument(
        "--over-ratio",
        type=float,
        default=DEFAULT_OVER_SIGNAL_RATIO,
        help=(
            "Ablation primary delta above this multiple of the "
            "full-hard delta triggers primary_over_signal."
        ),
    )
    parser.add_argument(
        "--min-primary-delta",
        type=float,
        default=DEFAULT_MIN_PRIMARY_DELTA,
        help=(
            "Absolute magnitude below which the primary signal is "
            "considered too weak (used when full-hard is unavailable)."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default: INFO).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        knobs = _parse_knobs(args.knobs)
    except ValueError as exc:
        logger.error("%s", exc)
        return 2
    try:
        result = analyze_domain(
            args.domain,
            knobs=knobs,
            expectations_path=args.expectations,
            out_dir=args.out_dir,
            flat_tolerance=args.flat_tolerance,
            under_ratio=args.under_ratio,
            over_ratio=args.over_ratio,
            min_primary_delta=args.min_primary_delta,
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 2

    results: list[KnobAblation] = result["results"]
    logger.info(
        "Analyzed %d knob ablation(s); report=%s",
        len(results),
        result["report_md"],
    )
    if result["missing"]:
        logger.warning(
            "Missing metrics for %d knob(s): %s",
            len(result["missing"]),
            result["missing"],
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
