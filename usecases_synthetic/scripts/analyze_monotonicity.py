#!/usr/bin/env python3
"""M8 — cross-level monotonicity + collapse analyzer.

Consumes the baseline (M5) and per-level validation (M7) metrics for
one domain and answers:

1. Is each knob's predicted signal monotone across
   easy -> medium -> hard in the direction the card predicts?
   (Baseline is shown in the report as a reference point but does not
   participate in the slope verdict — see ``monotonicity.SLOPE_LEVELS``.)
2. Has any member collapsed (F1 < 0.15 or drop > 0.5)?
3. For EM collapses, does the pool-agreement diagnostic corroborate
   (``real_collapse``) or contradict (``hidden_positive_noise``) the
   test-gold drop?

Writes

- ``usecases_synthetic/validation/<domain>/monotonicity_report.md``
- ``usecases_synthetic/validation/<domain>/monotonicity_report.csv``

M8 surfaces problems. M10 does the triage. No knob re-configuration
happens here.

Usage
-----
::

    python usecases_synthetic/scripts/analyze_monotonicity.py --domain companies
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from usecases_synthetic.lib.baseline_loader import baseline_path  # noqa: E402
from usecases_synthetic.lib.domain_config import SYNTHETIC_DIR  # noqa: E402
from usecases_synthetic.lib.monotonicity import (  # noqa: E402
    LEVELS,
    BestMemberCheck,
    Collapse,
    SignalCheck,
    SignalExpectation,
    detect_collapses,
    load_knob_expected_signals,
    compute_ceiling_responsiveness,
    match_best_member_monotonicity,
    match_signals,
    resolve_metric,
)

logger = logging.getLogger(__name__)

VALIDATION_DIR: Path = SYNTHETIC_DIR / "validation"
EXPECTATIONS_YAML: Path = SYNTHETIC_DIR / "config" / "knob_expected_signals.yaml"


def _load_json(path: Path) -> dict[str, Any]:
    """Read ``path`` as JSON; raise ``FileNotFoundError`` if missing."""
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    with open(path, encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
    return data


def _load_level_metrics(domain: str) -> dict[str, dict[str, Any]]:
    """Load baseline + per-level validation metrics for ``domain``.

    Parameters
    ----------
    domain : str
        Domain name.

    Returns
    -------
    dict[str, dict]
        ``{level: metrics_dict}`` with ``level`` in :data:`LEVELS`.

    Raises
    ------
    FileNotFoundError
        If baseline or any of the three level metrics files is
        missing.
    """
    out: dict[str, dict[str, Any]] = {}
    out["baseline"] = _load_json(baseline_path(domain))
    for level in ("easy", "medium", "hard"):
        level_path = VALIDATION_DIR / domain / level / "metrics.json"
        out[level] = _load_json(level_path)
    return out


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def _fmt(value: float) -> str:
    """Format a float value, returning ``"NaN"`` for missing values."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NaN"
    return f"{value:.3f}"


def _fmt_range(
    target_range: tuple[float, float] | None,
) -> str:
    """Render a target_delta_range tuple for the report."""
    if target_range is None:
        return "qualitative"
    return f"[{target_range[0]:.2f}, {target_range[1]:.2f}]"


def _check_symbol(ok: bool) -> str:
    """ASCII-only check / cross glyphs (no emoji per CLAUDE.md)."""
    return "[ok]" if ok else "[!!]"


def _relpath_to_repo(path: Path) -> str:
    """Render ``path`` relative to the repo root when possible.

    Falls back to the absolute path when ``path`` lives outside the
    repo (e.g. a temp dir in tests).
    """
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def build_signal_rows(
    checks: list[SignalCheck],
    *,
    ceiling_responsiveness: dict[tuple[str, str, str], float] | None = None,
) -> list[dict[str, Any]]:
    """Flatten :class:`SignalCheck` list into CSV-friendly row dicts.

    ``ceiling_responsiveness``, when provided, is a
    ``{(knob, signal_id, stage): pearson_r}`` lookup produced by
    :func:`compute_ceiling_responsiveness`. Missing keys default to
    ``math.nan``.
    """
    rows: list[dict[str, Any]] = []
    for check in checks:
        exp = check.expectation
        row: dict[str, Any] = {
            "knob": exp.knob,
            "signal_id": exp.signal_id,
            "stage": exp.stage,
            "metric": exp.metric,
            "direction": exp.direction,
            "qualitative_only": exp.qualitative_only,
            "target_delta_range": (
                ""
                if exp.target_delta_range is None
                else f"{exp.target_delta_range[0]:.3f}|{exp.target_delta_range[1]:.3f}"
            ),
            "is_monotone": check.is_monotone,
            "within_range": check.within_range,
            "baseline_position_ok": check.baseline_position_ok,
            "observed_delta": check.observed_delta,
            "reason": check.reason,
            "source_card": exp.source,
        }
        for level in LEVELS:
            row[level] = check.values.get(level, math.nan)
        if ceiling_responsiveness is not None:
            row["ceiling_responsiveness"] = ceiling_responsiveness.get(
                (exp.knob, exp.signal_id, exp.stage), math.nan
            )
        else:
            row["ceiling_responsiveness"] = math.nan
        rows.append(row)
    return rows


def build_best_member_rows(
    checks: list[BestMemberCheck],
) -> list[dict[str, Any]]:
    """Flatten :class:`BestMemberCheck` list into CSV-friendly row dicts.

    Used to emit ``monotonicity_best_member.csv`` so reviewers can
    inspect the per-level ceiling (the metric the user actually
    consumes) alongside the committee-mean signal.
    """
    rows: list[dict[str, Any]] = []
    for c in checks:
        row: dict[str, Any] = {
            "stage": c.stage,
            "is_non_increasing": c.is_non_increasing,
            "observed_delta": c.observed_delta,
            "reason": c.reason,
        }
        for level in LEVELS:
            row[f"{level}_value"] = c.values.get(level, math.nan)
            row[f"{level}_winner"] = c.winners.get(level, "")
        rows.append(row)
    return rows


def build_collapse_rows(collapses: list[Collapse]) -> list[dict[str, Any]]:
    """Flatten :class:`Collapse` list into CSV-friendly row dicts."""
    rows: list[dict[str, Any]] = []
    for c in collapses:
        rows.append(
            {
                "level": c.level,
                "stage": c.stage,
                "member": c.member,
                "baseline_f1": c.baseline_f1,
                "measured_f1": c.measured_f1,
                "delta": c.delta,
                "classification": c.classification,
                "pool_agreement_delta": (
                    ""
                    if c.pool_agreement_delta is None
                    else f"{c.pool_agreement_delta:.3f}"
                ),
                "recommended_action": c.recommended_action,
            }
        )
    return rows


# Per-stage load-bearing committee metric for the cumulative cross-level
# slope. This is the C2-contract verdict: committee scores decrease weakly
# easy -> medium -> hard, with baseline a reference value that must land no
# harder than medium. Read from the cumulative all-knobs-on variant levels,
# so it is the honest difficulty signal -- unlike the per-knob expectations,
# which assume single-knob isolation that cumulative variants don't provide.
CROSS_LEVEL_HEADLINE: tuple[tuple[str, str, str], ...] = (
    ("sm", "aggregated.macro_f1", "committee macro F1"),
    ("norm", "aggregated.macro_f1", "committee macro F1"),
    ("em_blocking", "aggregated.macro_pair_recall", "committee macro pair-recall"),
    (
        "em_matching",
        "aggregated.macro_f1_variant_model_on_regen_test",
        "variant-model macro F1 on regen test",
    ),
    ("fusion", "aggregated.overall_accuracy", "best-member overall accuracy"),
)

# Tolerance for "weakly" decreasing / baseline-position checks.
_SLOPE_TOL: float = 0.005


def build_cross_level_slope(
    level_metrics: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Per-stage load-bearing committee metric across baseline/easy/medium/hard.

    The honest difficulty verdict on the **cumulative** variants (all knobs
    on per level): the C2 contract asks committee scores to decrease weakly
    across ``easy -> medium -> hard``, with baseline a reference that must
    land no harder than medium (i.e. baseline score >= medium score). Makes
    no single-knob isolation assumption, so it -- not the per-knob signals --
    is the section to read for "is the difficulty monotone".
    """
    rows: list[dict[str, Any]] = []
    for stage, metric, label in CROSS_LEVEL_HEADLINE:
        vals: dict[str, float] = {}
        for level in LEVELS:
            per_stage = (level_metrics.get(level, {}) or {}).get("per_stage", {}) or {}
            block = per_stage.get(stage, {}) or {}
            try:
                vals[level] = resolve_metric(block, metric)
            except Exception:
                vals[level] = math.nan
        e = vals.get("easy", math.nan)
        m = vals.get("medium", math.nan)
        h = vals.get("hard", math.nan)
        b = vals.get("baseline", math.nan)
        slope_ok = (
            not any(math.isnan(x) for x in (e, m, h))
            and e + _SLOPE_TOL >= m
            and m + _SLOPE_TOL >= h
        )
        # Baseline must land no harder than medium: a harder level has the
        # LOWER committee score, so baseline_ok iff baseline >= medium - tol.
        baseline_ok = math.isnan(b) or math.isnan(m) or b + _SLOPE_TOL >= m
        rows.append(
            {
                "stage": stage,
                "metric": metric,
                "label": label,
                "values": vals,
                "delta_easy_hard": (
                    h - e if not (math.isnan(h) or math.isnan(e)) else math.nan
                ),
                "slope_ok": slope_ok,
                "baseline_ok": baseline_ok,
            }
        )
    return rows


def render_markdown(
    domain: str,
    checks: list[SignalCheck],
    collapses: list[Collapse],
    knob_order: list[str],
    best_member_checks: list[BestMemberCheck] | None = None,
    cross_level_slope: list[dict[str, Any]] | None = None,
) -> str:
    """Render the Markdown report."""
    lines: list[str] = []
    lines.append(f"# Monotonicity + Collapse Report — {domain}")
    lines.append("")
    lines.append(
        "Cross-level analysis of easy/medium/hard variants against the "
        "baseline. See `knobs/cross_cutting.md` for the protocol."
    )
    lines.append("")

    # ---------- Cumulative cross-level slope (load-bearing) ----------
    lines.append("## Cumulative Cross-Level Slope (load-bearing verdict)")
    lines.append("")
    lines.append(
        "Per-stage committee metric across the **cumulative** variant levels "
        "(every knob on at each level). This is the C2-contract verdict: "
        "committee scores should weakly decrease easy -> medium -> hard, and "
        "baseline (a reference value) should land no harder than medium. It "
        "makes no single-knob isolation assumption, so this -- not the "
        "per-knob signals below -- is the headline difficulty verdict."
    )
    lines.append("")
    lines.append(
        "| Stage | Metric | baseline | easy | medium | hard | easy->hard | "
        "Mono (e>=m>=h) | BasePos |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in cross_level_slope or []:
        v = r["values"]
        lines.append(
            f"| {r['stage']} | `{r['metric']}` | "
            f"{_fmt(v.get('baseline', math.nan))} | "
            f"{_fmt(v.get('easy', math.nan))} | "
            f"{_fmt(v.get('medium', math.nan))} | "
            f"{_fmt(v.get('hard', math.nan))} | "
            f"{_fmt(r['delta_easy_hard'])} | "
            f"{_check_symbol(r['slope_ok'])} | "
            f"{_check_symbol(r['baseline_ok'])} |"
        )
    lines.append("")

    # ---------- Per-knob expected signals (indicative) ----------
    lines.append("## Per-Knob Expected Signals (indicative)")
    lines.append("")
    lines.append(
        "> These are **per-knob** expectations from "
        "`knob_expected_signals.yaml`, evaluated against the **cumulative** "
        "variants (every knob on at each level). They cannot isolate one "
        "knob, so a `flat` expectation for a stage that *another* knob also "
        "drives reads `[!!]` by construction (e.g. SM is not flat for "
        "knob_01 because K8 naming is also on). Treat this as indicative of "
        "combined effect; the load-bearing verdict is the Cumulative "
        "Cross-Level Slope above. For true per-knob isolation, run "
        "`generate_variant --only-knob <K>` ablations."
    )
    lines.append("")
    per_knob: dict[str, list[SignalCheck]] = {k: [] for k in knob_order}
    for check in checks:
        per_knob.setdefault(check.expectation.knob, []).append(check)
    lines.append("| Knob | Signals | OK direction | OK range | Notes |")
    lines.append("|---|---|---|---|---|")
    for knob in knob_order:
        knob_checks = per_knob.get(knob, [])
        if not knob_checks:
            lines.append(f"| {knob} | 0 | — | — | no expectations |")
            continue
        ok_dir = sum(1 for c in knob_checks if c.is_monotone)
        ok_rng = sum(
            1
            for c in knob_checks
            if c.expectation.target_delta_range is not None and c.within_range
        )
        rng_total = sum(
            1 for c in knob_checks if c.expectation.target_delta_range is not None
        )
        card = knob_checks[0].expectation.source
        lines.append(
            f"| {knob} | {len(knob_checks)} | "
            f"{ok_dir}/{len(knob_checks)} "
            f"{_check_symbol(ok_dir == len(knob_checks))} | "
            f"{ok_rng}/{rng_total if rng_total else '—'} | "
            f"[card]({card}) |"
        )
    lines.append("")

    # ---------- Per-knob signal tables ----------
    lines.append("## Per-Signal Results")
    lines.append("")
    lines.append(
        "| Knob | Signal | Stage | Metric | Dir | baseline | easy | medium |"
        " hard | delta | target | Mono | Rng | BasePos | Reason |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for knob in knob_order:
        for check in per_knob.get(knob, []):
            exp = check.expectation
            row = (
                f"| {exp.knob} | {exp.signal_id} | {exp.stage} | "
                f"`{exp.metric}` | {exp.direction} | "
                f"{_fmt(check.values.get('baseline', math.nan))} | "
                f"{_fmt(check.values.get('easy', math.nan))} | "
                f"{_fmt(check.values.get('medium', math.nan))} | "
                f"{_fmt(check.values.get('hard', math.nan))} | "
                f"{_fmt(check.observed_delta)} | "
                f"{_fmt_range(exp.target_delta_range)} | "
                f"{_check_symbol(check.is_monotone)} | "
                f"{_check_symbol(check.within_range)} | "
                f"{_check_symbol(check.baseline_position_ok)} | "
                f"{check.reason} |"
            )
            lines.append(row)
    lines.append("")

    # ---------- Collapse table ----------
    lines.append("## Collapses")
    lines.append("")
    if not collapses:
        lines.append("No members fell below the collapse threshold.")
    else:
        lines.append(
            "Members with F1 < 0.15 or drop > 0.5 from baseline. EM "
            "collapses are classified against the pool-agreement "
            "diagnostic (`hidden_positive_noise` means pool is stable "
            "while test-gold moved — see `knobs/cross_cutting.md` § "
            "Protection set semantics)."
        )
        lines.append("")
        lines.append(
            "| Level | Stage | Member | baseline F1 | measured F1 | delta |"
            " classification | pool delta | action |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for c in collapses:
            pool = (
                "—" if c.pool_agreement_delta is None else _fmt(c.pool_agreement_delta)
            )
            lines.append(
                f"| {c.level} | {c.stage} | `{c.member}` | "
                f"{_fmt(c.baseline_f1)} | {_fmt(c.measured_f1)} | "
                f"{_fmt(c.delta)} | {c.classification} | {pool} | "
                f"{c.recommended_action} |"
            )
    lines.append("")

    # ---------- Best-member-F1 monotonicity (P8) ----------
    if best_member_checks:
        lines.append("## Best-Member Ceiling (P8)")
        lines.append("")
        lines.append(
            "Per-stage best-member F1 across baseline -> easy -> medium -> "
            "hard. A valid difficulty signal must depress the *ceiling* "
            "(the user-attainable member), not just the committee mean. "
            "A flat / rising ceiling means the user-selected matcher "
            "never sees the synthetic difficulty (committee-mean drift "
            "can be masked by weak-member degradation alone)."
        )
        lines.append("")
        lines.append(
            "| Stage | baseline | easy | medium | hard | delta | "
            "non-increasing | winner trail | Reason |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for c in best_member_checks:
            trail = " -> ".join(c.winners.get(lvl, "?") or "?" for lvl in LEVELS)
            lines.append(
                f"| {c.stage} | "
                f"{_fmt(c.values.get('baseline', math.nan))} | "
                f"{_fmt(c.values.get('easy', math.nan))} | "
                f"{_fmt(c.values.get('medium', math.nan))} | "
                f"{_fmt(c.values.get('hard', math.nan))} | "
                f"{_fmt(c.observed_delta)} | "
                f"{_check_symbol(c.is_non_increasing)} | "
                f"`{trail}` | {c.reason} |"
            )
        lines.append("")

    # ---------- Open questions ----------
    lines.append("## Open Questions")
    lines.append("")
    qualitative_fails = [
        check
        for check in checks
        if check.expectation.qualitative_only
        and check.is_monotone
        and check.expectation.target_delta_range is None
    ]
    if qualitative_fails:
        lines.append(
            "Signals that are direction-correct but magnitude-unspecified "
            "by the knob card. M10 should decide whether the measured "
            "delta is 'strong enough' to count as validation."
        )
        lines.append("")
        lines.append("| Knob | Signal | Stage | observed delta | Card notes |")
        lines.append("|---|---|---|---|---|")
        for check in qualitative_fails:
            exp = check.expectation
            notes = exp.notes.replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {exp.knob} | {exp.signal_id} | {exp.stage} | "
                f"{_fmt(check.observed_delta)} | {notes} |"
            )
    else:
        lines.append(
            "No pending magnitude decisions — every qualitative signal "
            "passed or failed on direction alone."
        )
    lines.append("")

    lines.append("## Provenance")
    lines.append("")
    lines.append(f"- Domain: {domain}")
    lines.append(f"- Expectations: `{_relpath_to_repo(EXPECTATIONS_YAML)}`")
    lines.append(f"- Baseline: `{_relpath_to_repo(baseline_path(domain))}`")
    lines.append(
        f"- Per-level metrics: `usecases_synthetic/validation/{domain}/<level>/metrics.json`"
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def analyze_domain(
    domain: str,
    *,
    expectations_path: Path = EXPECTATIONS_YAML,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the full M8 pipeline for one domain.

    Parameters
    ----------
    domain : str
        Domain name.
    expectations_path : Path, optional
        YAML file with per-knob expectations. Defaults to
        ``usecases_synthetic/config/knob_expected_signals.yaml``.
    out_dir : Path, optional
        Output directory. Defaults to
        ``usecases_synthetic/validation/<domain>/``.

    Returns
    -------
    dict
        Summary with keys ``signal_checks`` (list of :class:`SignalCheck`),
        ``collapses`` (list of :class:`Collapse`), and ``out_dir``.
    """
    knob_expectations: dict[str, list[SignalExpectation]] = load_knob_expected_signals(
        expectations_path
    )
    knob_order = list(knob_expectations.keys())
    all_expectations: list[SignalExpectation] = [
        exp for exps in knob_expectations.values() for exp in exps
    ]
    level_metrics = _load_level_metrics(domain)
    checks = match_signals(level_metrics, all_expectations)
    collapses = detect_collapses(level_metrics)
    # P8: best-member-F1 ceiling per stage. Surfaces difficulty signals
    # that depress committee macro_f1 while leaving the user-attainable
    # best member flat. Stages match the SM/Norm/EM-blocking/EM-matching
    # /Fusion roster emitted by measure_baseline.py + validate_variant.py.
    best_member_checks = match_best_member_monotonicity(level_metrics)
    # C6: per-(knob, signal, stage) Pearson r between the signal's
    # per-level realised metric and the stage's best-member F1. Near-0
    # values flag noop knobs for the ceiling that the user actually
    # consumes (vs. the committee mean).
    ceiling_responsiveness = compute_ceiling_responsiveness(checks, best_member_checks)

    target_dir = out_dir or (VALIDATION_DIR / domain)
    target_dir.mkdir(parents=True, exist_ok=True)

    csv_path = target_dir / "monotonicity_report.csv"
    _write_signal_csv(csv_path, checks, ceiling_responsiveness=ceiling_responsiveness)

    collapse_csv = target_dir / "monotonicity_collapses.csv"
    _write_collapse_csv(collapse_csv, collapses)

    best_csv = target_dir / "monotonicity_best_member.csv"
    _write_best_member_csv(best_csv, best_member_checks)

    cross_level_slope = build_cross_level_slope(level_metrics)
    slope_csv = target_dir / "cross_level_slope.csv"
    _write_cross_level_slope_csv(slope_csv, cross_level_slope)

    md_path = target_dir / "monotonicity_report.md"
    md_path.write_text(
        render_markdown(
            domain,
            checks,
            collapses,
            knob_order,
            best_member_checks,
            cross_level_slope=cross_level_slope,
        ),
        encoding="utf-8",
    )

    return {
        "signal_checks": checks,
        "collapses": collapses,
        "best_member_checks": best_member_checks,
        "cross_level_slope": cross_level_slope,
        "out_dir": target_dir,
        "report_md": md_path,
        "report_csv": csv_path,
        "collapse_csv": collapse_csv,
        "best_member_csv": best_csv,
        "cross_level_slope_csv": slope_csv,
    }


def _write_signal_csv(
    path: Path,
    checks: list[SignalCheck],
    *,
    ceiling_responsiveness: dict[tuple[str, str, str], float] | None = None,
) -> None:
    """Write signal-check rows to ``path`` in CSV form."""
    rows = build_signal_rows(checks, ceiling_responsiveness=ceiling_responsiveness)
    fieldnames = [
        "knob",
        "signal_id",
        "stage",
        "metric",
        "direction",
        "qualitative_only",
        "target_delta_range",
        *LEVELS,
        "observed_delta",
        "is_monotone",
        "within_range",
        "baseline_position_ok",
        "ceiling_responsiveness",
        "reason",
        "source_card",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_collapse_csv(path: Path, collapses: list[Collapse]) -> None:
    """Write collapse rows to ``path`` in CSV form."""
    fieldnames = [
        "level",
        "stage",
        "member",
        "baseline_f1",
        "measured_f1",
        "delta",
        "classification",
        "pool_agreement_delta",
        "recommended_action",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in build_collapse_rows(collapses):
            writer.writerow(row)


def _write_best_member_csv(path: Path, checks: list[BestMemberCheck]) -> None:
    """Write best-member-F1 check rows to ``path`` (P8)."""
    fieldnames = ["stage"]
    for level in LEVELS:
        fieldnames.extend([f"{level}_value", f"{level}_winner"])
    fieldnames.extend(["observed_delta", "is_non_increasing", "reason"])
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in build_best_member_rows(checks):
            writer.writerow(row)


def _write_cross_level_slope_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the cumulative cross-level slope (load-bearing verdict) to CSV."""
    fieldnames = [
        "stage",
        "metric",
        "label",
        *LEVELS,
        "delta_easy_hard",
        "slope_ok",
        "baseline_ok",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            out = {
                "stage": r["stage"],
                "metric": r["metric"],
                "label": r["label"],
                "delta_easy_hard": r["delta_easy_hard"],
                "slope_ok": r["slope_ok"],
                "baseline_ok": r["baseline_ok"],
            }
            for level in LEVELS:
                out[level] = r["values"].get(level, math.nan)
            writer.writerow(out)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(
        description=(
            "Cross-level monotonicity + collapse analyzer (M8). "
            "Reads baseline metrics and per-level validation outputs; "
            "writes monotonicity_report.md + .csv."
        )
    )
    parser.add_argument(
        "--domain",
        required=True,
        help="Domain name (e.g. 'companies').",
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
            "Override the output directory (default: "
            "usecases_synthetic/validation/<domain>/)."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default: INFO).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        result = analyze_domain(
            args.domain,
            expectations_path=args.expectations,
            out_dir=args.out_dir,
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 2

    checks: list[SignalCheck] = result["signal_checks"]
    collapses: list[Collapse] = result["collapses"]
    logger.info(
        "analysed %d signal expectations; %d collapses; report=%s",
        len(checks),
        len(collapses),
        result["report_md"],
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
