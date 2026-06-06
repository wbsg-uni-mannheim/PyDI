"""Cross-level monotonicity and collapse analysis.

Pure functions used by M8 (``scripts/analyze_monotonicity.py``) to turn
the per-level ``metrics.json`` files from M7 into a signed-off
"did-the-knob-move-the-committee?" report.

The module is deliberately free of I/O concerns — it operates on
already-loaded JSON dicts so the CLI entry point stays small.

Key concepts
------------
- **Signal expectation.** A machine-readable record loaded from
  ``usecases_synthetic/config/knob_expected_signals.yaml``. It names a
  stage, a metric path, and a predicted direction (down/up/flat).
  Magnitudes are qualitative by design — no knob card gives numeric
  ranges.
- **Metric path.** Either a dotted path into a stage block (for
  example ``aggregated.macro_f1`` or
  ``per_member.standard_rule.metrics.f1``), or a spread spec
  ``spread:<path_a>:<path_b>`` whose value is ``path_a - path_b``.
- **Collapse.** A committee member whose F1 fell below ``floor`` or
  dropped by more than ``max_drop`` from the baseline. Classified as
  ``hidden_positive_noise`` when the pool-agreement diagnostic is
  stable despite the test-gold F1 collapsing, ``real_collapse``
  otherwise (per ``knobs/cross_cutting.md`` § Protection set
  semantics).

Nothing in this module writes files or fixes collapses — M8 surfaces,
M10 triages.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import yaml

Direction = Literal["down", "up", "flat"]
LEVELS: tuple[str, ...] = ("baseline", "easy", "medium", "hard")
# Levels that participate in the monotonicity slope check.
# baseline is a reference point shown on charts but does NOT take part
# in the slope verdict — it is expected to sit somewhere between easy
# and medium for most stages (e.g. K2-survivable subset makes easy
# easier than baseline by construction). The contract is that
# easy -> medium -> hard moves monotonically in the predicted
# direction.
SLOPE_LEVELS: tuple[str, ...] = ("easy", "medium", "hard")
DEFAULT_FLAT_TOLERANCE: float = 0.05
DEFAULT_COLLAPSE_DROP: float = 0.5
DEFAULT_COLLAPSE_FLOOR: float = 0.15
POOL_STABLE_TOLERANCE: float = 0.1


@dataclass
class SignalExpectation:
    """A single predicted signal parsed from the expectations YAML.

    Parameters
    ----------
    knob : str
        Knob identifier (e.g. ``"knob_08"``).
    signal_id : str
        Unique-within-knob short id (e.g. ``"sm_spread_is_signal"``).
    stage : str
        ``"sm"``, ``"em"``, or ``"fusion"``.
    metric : str
        Either a dotted path into the stage block, or a spread spec
        ``"spread:<path_a>:<path_b>"``.
    direction : str
        ``"down"``, ``"up"``, or ``"flat"``.
    qualitative_only : bool
        ``True`` if the card gives direction but no numeric magnitude.
    target_delta_range : tuple[float, float] | None
        Bounds on ``(hard - baseline)`` when known. ``None`` when
        qualitative.
    pool_check : bool
        Whether collapse flags on this metric should be cross-checked
        against the EM pool-agreement diagnostic.
    notes : str
        Free-form explanation for human readers.
    source : str
        Provenance string (card path or card path + section).
    """

    knob: str
    signal_id: str
    stage: str
    metric: str
    direction: Direction
    qualitative_only: bool
    target_delta_range: tuple[float, float] | None
    pool_check: bool
    notes: str
    source: str


@dataclass
class SignalCheck:
    """Outcome of evaluating one :class:`SignalExpectation` across levels.

    Parameters
    ----------
    expectation : SignalExpectation
        The input prediction.
    values : dict[str, float]
        ``{level: value}`` for each of baseline/easy/medium/hard. A
        missing metric is represented as ``math.nan``.
    is_monotone : bool
        ``True`` iff the values respect the predicted direction across
        easy -> medium -> hard (ties allowed). Baseline is a reference
        point shown in ``values`` but does not gate the verdict — see
        ``SLOPE_LEVELS``.
    within_range : bool
        ``True`` iff ``(hard - baseline)`` falls inside
        ``target_delta_range``. Always ``True`` when the range is
        ``None``.
    observed_delta : float
        ``hard - baseline``. ``math.nan`` if either endpoint is NaN.
    baseline_position_ok : bool
        ``True`` iff the baseline value is NOT harder than medium.
        Baseline is allowed to sit between easy and medium, or to be
        easier than easy (typical case for K2-survivable easy
        regeneration), but it must not be harder than medium. For
        ``direction="down"`` this means ``baseline >= medium``; for
        ``direction="up"``, ``baseline <= medium``; for
        ``direction="flat"``, within ``flat_tolerance`` of medium.
        ``True`` when either value is NaN (cannot check).
    reason : str
        Short human-readable justification for ``is_monotone``.
    """

    expectation: SignalExpectation
    values: dict[str, float]
    is_monotone: bool
    within_range: bool
    observed_delta: float
    baseline_position_ok: bool
    reason: str


@dataclass
class Collapse:
    """A member/stage that has fallen below usable signal at some level.

    Parameters
    ----------
    level : str
        ``"easy"``, ``"medium"``, or ``"hard"``.
    stage : str
        Stage name.
    member : str
        Committee member name within the stage.
    baseline_f1 : float
        Baseline F1 (or analogous primary metric).
    measured_f1 : float
        Measured F1 at ``level``.
    delta : float
        ``measured_f1 - baseline_f1``.
    classification : str
        ``"hidden_positive_noise"`` if pool agreement is stable while
        the test-gold F1 collapsed, ``"real_collapse"`` otherwise.
        ``"unknown"`` when the pool diagnostic is unavailable (SM /
        Fusion stages).
    recommended_action : str
        Short hint pointing at the per-knob fix-strategy table in
        ``knobs/cross_cutting.md``.
    pool_agreement_delta : float | None
        ``measured_pool_precision - baseline_pool_precision`` for EM
        stages; ``None`` elsewhere.
    """

    level: str
    stage: str
    member: str
    baseline_f1: float
    measured_f1: float
    delta: float
    classification: Literal["hidden_positive_noise", "real_collapse", "unknown"]
    recommended_action: str
    pool_agreement_delta: float | None = None


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def _resolve_dotted_path(block: Mapping[str, Any], path: str) -> float:
    """Look up ``path`` inside ``block`` (dotted segments)."""
    node: Any = block
    for segment in path.split("."):
        if isinstance(node, Mapping) and segment in node:
            node = node[segment]
            continue
        return math.nan
    try:
        return float(node)
    except (TypeError, ValueError):
        return math.nan


def resolve_metric(stage_block: Mapping[str, Any], metric: str) -> float:
    """Return the numeric value of ``metric`` against ``stage_block``.

    Parameters
    ----------
    stage_block : mapping
        The ``per_stage[stage]`` sub-dict (i.e. the committee result
        block for a single stage and level).
    metric : str
        Either a dotted path (``"aggregated.macro_f1"``) or a spread
        spec (``"spread:<path_a>:<path_b>"``). A spread resolves to
        ``value_at_path_a - value_at_path_b``; it is ``NaN`` when either
        side is missing.

    Returns
    -------
    float
        Numeric value, or ``math.nan`` if the lookup fails.
    """
    if metric.startswith("spread:"):
        body = metric[len("spread:") :]
        parts = body.split(":")
        if len(parts) != 2:
            raise ValueError(
                f"Spread metric must be 'spread:<path_a>:<path_b>': got {metric!r}"
            )
        a = _resolve_dotted_path(stage_block, parts[0])
        b = _resolve_dotted_path(stage_block, parts[1])
        if math.isnan(a) or math.isnan(b):
            return math.nan
        return a - b
    return _resolve_dotted_path(stage_block, metric)


# ---------------------------------------------------------------------------
# Direction logic
# ---------------------------------------------------------------------------


def check_monotone(
    values: Sequence[float],
    direction: Direction,
    *,
    flat_tolerance: float = DEFAULT_FLAT_TOLERANCE,
) -> bool:
    """Return ``True`` iff ``values`` follow ``direction`` weakly.

    Parameters
    ----------
    values : sequence of float
        Values in level order. Callers in this module pass slope-level
        values (easy, medium, hard); baseline is excluded by design
        (see ``SLOPE_LEVELS``). ``NaN`` entries cause the check to
        return ``False``.
    direction : str
        ``"down"``, ``"up"``, or ``"flat"``. ``"flat"`` requires every
        value to be within ``flat_tolerance`` of the first.
    flat_tolerance : float, optional
        Absolute tolerance for the ``"flat"`` case. Default ``0.05``.

    Returns
    -------
    bool
        Weakly monotone (non-strict) in the requested direction. Ties
        are allowed for ``"down"`` and ``"up"``.
    """
    if len(values) < 2:
        return False
    if any(math.isnan(v) for v in values):
        return False
    if direction == "down":
        return all(b <= a for a, b in zip(values, values[1:]))
    if direction == "up":
        return all(b >= a for a, b in zip(values, values[1:]))
    if direction == "flat":
        anchor = values[0]
        return all(abs(v - anchor) <= flat_tolerance for v in values)
    raise ValueError(f"Unknown direction: {direction!r}")


def baseline_within_allowed_position(
    baseline_val: float,
    medium_val: float,
    direction: Direction,
    *,
    flat_tolerance: float = DEFAULT_FLAT_TOLERANCE,
) -> bool:
    """Return ``True`` iff baseline is NOT harder than medium.

    Baseline can land anywhere between easy and medium, or be easier
    than easy (e.g. K2-survivable easy regeneration is intrinsically
    easier than raw baseline). The contract is only that baseline
    must not be *harder than medium* — that would invert the implicit
    difficulty ordering for any reader inspecting the report.

    Direction semantics:

    - ``"down"`` (e.g. macro_f1): higher value = easier. ``baseline``
      must be ``>= medium`` (i.e. easier-or-equal).
    - ``"up"`` (e.g. error rate): higher value = harder. ``baseline``
      must be ``<= medium`` (i.e. easier-or-equal).
    - ``"flat"``: ``abs(baseline - medium) <= flat_tolerance``.

    Returns ``True`` if either value is NaN (cannot check).
    """
    if math.isnan(baseline_val) or math.isnan(medium_val):
        return True
    if direction == "down":
        return baseline_val >= medium_val
    if direction == "up":
        return baseline_val <= medium_val
    if direction == "flat":
        return abs(baseline_val - medium_val) <= flat_tolerance
    raise ValueError(f"Unknown direction: {direction!r}")


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------


def load_knob_expected_signals(path: Path) -> dict[str, list[SignalExpectation]]:
    """Parse ``knob_expected_signals.yaml`` into expectation records.

    Parameters
    ----------
    path : Path
        Location of the YAML file.

    Returns
    -------
    dict[str, list[SignalExpectation]]
        ``{knob_id: [expectations]}``.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the file is missing required fields.
    """
    if not path.exists():
        raise FileNotFoundError(f"Knob expectations YAML not found: {path}")
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    out: dict[str, list[SignalExpectation]] = {}
    for key, block in raw.items():
        if not isinstance(block, Mapping):
            continue
        if not key.startswith("knob_"):
            continue
        signals_raw = block.get("signals") or []
        if not isinstance(signals_raw, list):
            raise ValueError(f"{key}: 'signals' must be a list")
        card = str(block.get("source_card", ""))
        items: list[SignalExpectation] = []
        for entry in signals_raw:
            if not isinstance(entry, Mapping):
                raise ValueError(f"{key}: signal entries must be mappings")
            try:
                direction = entry["direction"]
                stage = entry["stage"]
                metric = entry["metric"]
                signal_id = entry["id"]
            except KeyError as exc:
                raise ValueError(
                    f"{key}: signal is missing required field {exc.args[0]!r}"
                ) from exc
            if direction not in ("down", "up", "flat"):
                raise ValueError(
                    f"{key}.{signal_id}: direction must be down/up/flat, "
                    f"got {direction!r}"
                )
            raw_range = entry.get("target_delta_range")
            target_range: tuple[float, float] | None
            if raw_range is None:
                target_range = None
            else:
                if (
                    not isinstance(raw_range, list)
                    or len(raw_range) != 2
                    or not all(isinstance(v, (int, float)) for v in raw_range)
                ):
                    raise ValueError(
                        f"{key}.{signal_id}: target_delta_range must be "
                        f"[min, max] or null, got {raw_range!r}"
                    )
                target_range = (float(raw_range[0]), float(raw_range[1]))
            items.append(
                SignalExpectation(
                    knob=key,
                    signal_id=str(signal_id),
                    stage=str(stage),
                    metric=str(metric),
                    direction=direction,
                    qualitative_only=bool(entry.get("qualitative_only", False)),
                    target_delta_range=target_range,
                    pool_check=bool(entry.get("pool_check", False)),
                    notes=str(entry.get("notes", "")).strip(),
                    source=card,
                )
            )
        # point-0 (variant audit): the frozen-baseline EM difficulty signal is
        # primarily reported on the corner_filled (regen) test surface. The
        # corner_filled surface backfills negatives and its positive set
        # composition drifts per level, which can mask injected difficulty
        # (e.g. games-hard looks EASIER on corner_filled but HARDER on the
        # pruned surface). The baseline_pruned surface is composition-stable
        # (baseline gold pruned to surviving records only), so report BOTH:
        # clone every em_matching frozen-regen signal into a *_pruned sibling
        # pointing at the baseline_pruned aggregate. pool_check is dropped on
        # the sibling so the collapse cross-check is not double-counted.
        _REGEN = "aggregated.macro_f1_baseline_model_on_regen_test"
        _PRUNED = "aggregated.macro_f1_baseline_model_on_baseline_test"
        pruned_siblings = [
            replace(
                exp,
                signal_id=f"{exp.signal_id}_pruned",
                metric=_PRUNED,
                pool_check=False,
                notes=(exp.notes + "\n[baseline_pruned surface companion]").strip(),
            )
            for exp in items
            if exp.stage == "em_matching" and exp.metric == _REGEN
        ]
        items.extend(pruned_siblings)
        out[key] = items
    return out


# ---------------------------------------------------------------------------
# Signal matching
# ---------------------------------------------------------------------------


def _stage_block(metrics: Mapping[str, Any], stage: str) -> Mapping[str, Any]:
    """Return the ``per_stage[stage]`` block or an empty dict."""
    per_stage = metrics.get("per_stage", {}) or {}
    block = per_stage.get(stage, {}) or {}
    if not isinstance(block, Mapping):
        return {}
    return block


def collect_level_values(
    level_metrics: Mapping[str, Mapping[str, Any]],
    expectation: SignalExpectation,
) -> dict[str, float]:
    """Read the expectation's metric from each level's metrics dict.

    Parameters
    ----------
    level_metrics : mapping
        ``{level: metrics_dict}`` covering at least ``baseline``, ``easy``,
        ``medium``, ``hard``. ``metrics_dict`` is the full
        ``metrics.json`` payload.
    expectation : SignalExpectation
        Signal to evaluate.

    Returns
    -------
    dict[str, float]
        ``{level: value}`` for every level in :data:`LEVELS`. Missing or
        unparseable metrics become ``math.nan``.
    """
    values: dict[str, float] = {}
    for level in LEVELS:
        metrics = level_metrics.get(level, {}) or {}
        block = _stage_block(metrics, expectation.stage)
        values[level] = resolve_metric(block, expectation.metric)
    return values


def match_signals(
    level_metrics: Mapping[str, Mapping[str, Any]],
    expectations: Sequence[SignalExpectation],
    *,
    flat_tolerance: float = DEFAULT_FLAT_TOLERANCE,
) -> list[SignalCheck]:
    """Evaluate every expectation against the supplied per-level metrics.

    Parameters
    ----------
    level_metrics : mapping
        ``{level: metrics_dict}``. Missing levels are tolerated but
        produce NaN values.
    expectations : sequence of SignalExpectation
        Predictions to check.
    flat_tolerance : float, optional
        Absolute tolerance for the ``"flat"`` direction. Default
        ``0.05``.

    Returns
    -------
    list[SignalCheck]
        One per expectation, in input order.
    """
    checks: list[SignalCheck] = []
    for exp in expectations:
        values = collect_level_values(level_metrics, exp)
        # Slope check runs on easy -> medium -> hard only. Baseline is a
        # reference point (shown in `values` and in the delta below) but
        # does not gate the monotonicity verdict — see SLOPE_LEVELS.
        ordered_slope = [values[level] for level in SLOPE_LEVELS]
        is_monotone = check_monotone(
            ordered_slope, exp.direction, flat_tolerance=flat_tolerance
        )
        baseline_val = values["baseline"]
        hard_val = values["hard"]
        medium_val = values["medium"]
        if math.isnan(baseline_val) or math.isnan(hard_val):
            observed_delta = math.nan
        else:
            observed_delta = hard_val - baseline_val
        within_range = True
        if exp.target_delta_range is not None and not math.isnan(observed_delta):
            lo, hi = exp.target_delta_range
            within_range = lo <= observed_delta <= hi
        baseline_position_ok = baseline_within_allowed_position(
            baseline_val, medium_val, exp.direction, flat_tolerance=flat_tolerance
        )
        reason = _explain_check(ordered_slope, exp, is_monotone)
        checks.append(
            SignalCheck(
                expectation=exp,
                values=values,
                is_monotone=is_monotone,
                within_range=within_range,
                observed_delta=observed_delta,
                baseline_position_ok=baseline_position_ok,
                reason=reason,
            )
        )
    return checks


def _explain_check(
    ordered: Sequence[float],
    exp: SignalExpectation,
    is_monotone: bool,
) -> str:
    """Build a short reason string for a :class:`SignalCheck`."""
    # ordered carries slope-level values (easy/medium/hard); baseline is
    # tracked separately in SignalCheck.values and not part of the slope.
    if any(math.isnan(v) for v in ordered):
        missing_levels = [
            level for level, value in zip(SLOPE_LEVELS, ordered) if math.isnan(value)
        ]
        return "metric missing at levels: " + ", ".join(missing_levels)
    values_str = " -> ".join(f"{v:.3f}" for v in ordered)
    if is_monotone:
        return f"weakly {exp.direction}: {values_str}"
    return f"not weakly {exp.direction}: {values_str}"


# ---------------------------------------------------------------------------
# Collapse detection
# ---------------------------------------------------------------------------


def _member_f1(member_block: Mapping[str, Any]) -> float:
    """Extract the primary F1 (or overall_accuracy for fusion) of a member."""
    metrics = member_block.get("metrics", {}) or {}
    for key in ("f1", "overall_accuracy"):
        if key in metrics:
            try:
                return float(metrics[key])
            except (TypeError, ValueError):
                return math.nan
    return math.nan


def _member_pool_precision(member_block: Mapping[str, Any]) -> float | None:
    """Extract the member's pool_precision if present (EM only)."""
    metrics = member_block.get("metrics", {}) or {}
    if "pool_precision" not in metrics:
        return None
    try:
        return float(metrics["pool_precision"])
    except (TypeError, ValueError):
        return None


def _fix_action_for(stage: str) -> str:
    """Return a short action hint keyed off stage.

    The exact per-knob fix-strategy table lives in
    ``knobs/cross_cutting.md`` § Per-knob fix-strategy defaults. M8
    does not auto-apply fixes; this string is a signpost only.
    """
    return {
        "sm": "see knobs/cross_cutting.md § Per-knob fix-strategy defaults (SM row)",
        "em": "see knobs/cross_cutting.md § Per-knob fix-strategy defaults (EM row)",
        "fusion": (
            "see knobs/cross_cutting.md § Per-knob fix-strategy defaults "
            "(fusion row)"
        ),
    }.get(stage, "see knobs/cross_cutting.md § Per-knob fix-strategy defaults")


def detect_collapses(
    level_metrics: Mapping[str, Mapping[str, Any]],
    *,
    max_drop: float = DEFAULT_COLLAPSE_DROP,
    floor: float = DEFAULT_COLLAPSE_FLOOR,
    pool_stable_tolerance: float = POOL_STABLE_TOLERANCE,
) -> list[Collapse]:
    """Flag every (level, stage, member) whose primary metric collapsed.

    A collapse fires when the member's ``f1`` (or ``overall_accuracy``
    for fusion) at a given level is below ``floor`` OR has dropped by
    more than ``max_drop`` from the baseline value. For EM members the
    pool-agreement diagnostic disambiguates the collapse into
    ``hidden_positive_noise`` (pool agreement steady → test gold
    grew stale) versus ``real_collapse`` (both test gold and pool
    agreement moved together).

    Parameters
    ----------
    level_metrics : mapping
        ``{level: metrics_dict}``. ``baseline`` is required; other
        levels are scanned if present.
    max_drop : float, optional
        Baseline minus measured threshold. Default ``0.5``.
    floor : float, optional
        Absolute threshold below which a metric is considered collapsed.
        Default ``0.15``.
    pool_stable_tolerance : float, optional
        Absolute tolerance on ``pool_precision`` considered "stable"
        relative to baseline. Default ``0.1``.

    Returns
    -------
    list[Collapse]
        One entry per detected collapse, in (level, stage, member)
        order.
    """
    baseline = level_metrics.get("baseline")
    if baseline is None:
        return []

    out: list[Collapse] = []
    for level in ("easy", "medium", "hard"):
        metrics = level_metrics.get(level)
        if metrics is None:
            continue
        for stage in ("sm", "em", "fusion"):
            stage_block = _stage_block(metrics, stage)
            baseline_stage = _stage_block(baseline, stage)
            members = stage_block.get("per_member", {}) or {}
            baseline_members = baseline_stage.get("per_member", {}) or {}
            for member_name, member_block in members.items():
                if not isinstance(member_block, Mapping):
                    continue
                measured_f1 = _member_f1(member_block)
                if math.isnan(measured_f1):
                    continue
                baseline_block = baseline_members.get(member_name, {}) or {}
                baseline_f1 = _member_f1(baseline_block)
                if math.isnan(baseline_f1):
                    continue
                collapsed = measured_f1 < floor or (
                    (baseline_f1 - measured_f1) > max_drop
                )
                if not collapsed:
                    continue
                classification: Literal[
                    "hidden_positive_noise", "real_collapse", "unknown"
                ]
                pool_delta: float | None = None
                if stage == "em":
                    measured_pool = _member_pool_precision(member_block)
                    baseline_pool = _member_pool_precision(baseline_block)
                    if measured_pool is None or baseline_pool is None:
                        classification = "unknown"
                    else:
                        pool_delta = measured_pool - baseline_pool
                        if abs(pool_delta) <= pool_stable_tolerance:
                            classification = "hidden_positive_noise"
                        else:
                            classification = "real_collapse"
                else:
                    classification = "unknown"
                out.append(
                    Collapse(
                        level=level,
                        stage=stage,
                        member=member_name,
                        baseline_f1=baseline_f1,
                        measured_f1=measured_f1,
                        delta=measured_f1 - baseline_f1,
                        classification=classification,
                        recommended_action=_fix_action_for(stage),
                        pool_agreement_delta=pool_delta,
                    )
                )
    return out


# ---------------------------------------------------------------------------
# Best-member-F1 monotonicity (P8)
# ---------------------------------------------------------------------------


# Per-stage key metric for "best member at this level". When the stage's
# members emit a different primary, look it up here. ``f1`` is the
# default and covers SM + EM matching + Norm-per-attr (Norm members
# report ``macro_f1`` at the member level, special-cased below).
#
# R7b dual-model dual-test (2026-05-27): the EM matching / blocking
# headline is the variant-trained model on the corner_filled test
# surface (cleanly isolates intrinsic difficulty from
# transfer-learning gap — see plan_revision.md R7b). Falls back to the
# legacy per-pair ``f1_regen_test`` / ``pair_recall`` aliases when the
# new explicit keys are absent (older committee runs predating R7b),
# then to the bare ``f1`` / ``pair_recall`` headline.
_BEST_MEMBER_METRIC: dict[str, tuple[str, ...]] = {
    "sm": ("f1",),
    "norm": ("macro_f1",),
    "em_blocking": (
        "pair_recall_variant_model_on_regen_test",
        "pair_recall",
    ),
    "em_matching": (
        "f1_variant_model_on_regen_test",
        "f1_regen_test",
        "f1",
    ),
    "em": (
        "f1_variant_model_on_regen_test",
        "f1_regen_test",
        "f1",
    ),
    "fusion": ("macro_accuracy", "overall_accuracy"),
}


def _stage_best_member_value(
    stage_block: Mapping[str, Any], stage: str
) -> tuple[str, float]:
    """Return ``(member_name, primary_value)`` for the best member.

    The "primary" metric is selected from :data:`_BEST_MEMBER_METRIC`
    per-stage. Members missing the primary metric (or with ``NaN``) are
    ignored. When no member has a finite value the function returns
    ``("", math.nan)``.

    Parameters
    ----------
    stage_block : mapping
        The ``per_stage[stage]`` block from a metrics dict.
    stage : str
        Stage name (``"sm"`` / ``"norm"`` / ``"em_blocking"`` /
        ``"em_matching"`` / ``"fusion"``).

    Returns
    -------
    tuple[str, float]
        Winner member name and its primary metric value.
    """
    per_member = stage_block.get("per_member", {}) or {}
    metric_keys = _BEST_MEMBER_METRIC.get(stage, ("f1",))
    best_name = ""
    best_val = math.nan
    for member_name, member_block in per_member.items():
        if not isinstance(member_block, Mapping):
            continue
        metrics = member_block.get("metrics", {}) or {}
        value: float | None = None
        for key in metric_keys:
            if key in metrics:
                try:
                    value = float(metrics[key])
                except (TypeError, ValueError):
                    value = None
                break
        if value is None or math.isnan(value):
            continue
        if math.isnan(best_val) or value > best_val:
            best_val = value
            best_name = member_name
    return (best_name, best_val)


@dataclass
class BestMemberCheck:
    """Best-member-F1 monotonicity outcome for one stage.

    Pinned to the *per-level best member*: the user-attainable ceiling
    of the committee at each difficulty level. A difficulty signal that
    depresses committee macro_f1 (the mean) but leaves best-member F1
    flat is invalid — the user picks the best member and never sees
    the mean drop. P8.

    Parameters
    ----------
    stage : str
        Stage name.
    values : dict[str, float]
        ``{level: best_member_value}`` for each level in :data:`LEVELS`.
    winners : dict[str, str]
        ``{level: best_member_name}``; the winner can change across
        levels and that's fine (the metric tracks the ceiling, not a
        fixed member).
    is_non_increasing : bool
        ``True`` iff the values respect ``baseline >= easy >= medium >=
        hard`` (with ``flat_tolerance`` slack). ``False`` means the
        ceiling did not move with difficulty.
    observed_delta : float
        ``hard - baseline``. ``NaN`` if either endpoint is NaN.
    reason : str
        Short human-readable justification.
    """

    stage: str
    values: dict[str, float]
    winners: dict[str, str]
    is_non_increasing: bool
    observed_delta: float
    reason: str


def collect_best_member_per_level(
    level_metrics: Mapping[str, Mapping[str, Any]], stage: str
) -> tuple[dict[str, float], dict[str, str]]:
    """Compute the per-level best member's value + name for one stage.

    Parameters
    ----------
    level_metrics : mapping
        ``{level: metrics_dict}``. Missing levels are tolerated and
        yield ``NaN`` / ``""``.
    stage : str
        Stage name.

    Returns
    -------
    tuple[dict, dict]
        ``({level: value}, {level: member_name})``.
    """
    values: dict[str, float] = {}
    winners: dict[str, str] = {}
    for level in LEVELS:
        metrics = level_metrics.get(level, {}) or {}
        block = _stage_block(metrics, stage)
        name, value = _stage_best_member_value(block, stage)
        values[level] = value
        winners[level] = name
    return values, winners


def match_best_member_monotonicity(
    level_metrics: Mapping[str, Mapping[str, Any]],
    stages: Sequence[str] = ("sm", "norm", "em_blocking", "em_matching", "fusion"),
    *,
    flat_tolerance: float = DEFAULT_FLAT_TOLERANCE,
) -> list[BestMemberCheck]:
    """Evaluate best-member-F1 monotonicity across levels per stage.

    For each stage, find the *best member* at every level (the winner
    is allowed to change across levels) and verify that the
    user-attainable ceiling does not stay flat or rise as difficulty
    increases.

    Direction is fixed to ``baseline >= easy >= medium >= hard``: a
    valid synthetic difficulty *depresses* the ceiling. Stages that
    measure recall / accuracy / F1 all share this expected direction;
    the function does not currently support stages where higher value
    = harder (none in the current pipeline).

    Parameters
    ----------
    level_metrics : mapping
        ``{level: metrics_dict}``.
    stages : sequence of str, optional
        Stages to evaluate. Default covers the full committee roster.
    flat_tolerance : float, optional
        Absolute slack allowed before a flat/rising sequence is
        flagged as non-monotone. Default 0.05 (5 F1 points).

    Returns
    -------
    list[BestMemberCheck]
        One entry per stage, in input order.
    """
    checks: list[BestMemberCheck] = []
    for stage in stages:
        values, winners = collect_best_member_per_level(level_metrics, stage)
        ordered = [values[level] for level in LEVELS]
        # ``check_monotone`` uses the SignalExpectation direction
        # vocabulary (``"down"`` / ``"up"`` / ``"flat"``); P8's
        # "non-increasing" direction maps to ``"down"`` (ties allowed).
        is_non_increasing = check_monotone(
            ordered, "down", flat_tolerance=flat_tolerance
        )
        if math.isnan(values["baseline"]) or math.isnan(values["hard"]):
            observed_delta = math.nan
        else:
            observed_delta = values["hard"] - values["baseline"]
        if any(math.isnan(v) for v in ordered):
            missing = [lvl for lvl, v in zip(LEVELS, ordered) if math.isnan(v)]
            reason = "best-member ceiling missing at levels: " + ", ".join(missing)
        else:
            values_str = " -> ".join(f"{v:.3f}" for v in ordered)
            winner_trail = " -> ".join(winners[lvl] or "?" for lvl in LEVELS)
            if is_non_increasing:
                reason = f"best-member ceiling non-increasing: {values_str}  ({winner_trail})"
            else:
                reason = (
                    f"best-member ceiling did NOT decline: {values_str}  "
                    f"({winner_trail}) — difficulty dial may be invisible to "
                    f"the user-selected matcher"
                )
        checks.append(
            BestMemberCheck(
                stage=stage,
                values=values,
                winners=winners,
                is_non_increasing=is_non_increasing,
                observed_delta=observed_delta,
                reason=reason,
            )
        )
    return checks


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    """Pearson correlation for two same-length finite sequences.

    Returns ``math.nan`` if fewer than two non-NaN paired observations
    survive, or if either series has zero variance after filtering NaNs.
    Pure-Python — no numpy dependency. With only four data points
    (baseline → easy → medium → hard) this is fast and exact.
    """
    if len(x) != len(y):
        return math.nan
    pairs = [(xi, yi) for xi, yi in zip(x, y) if not (math.isnan(xi) or math.isnan(yi))]
    if len(pairs) < 2:
        return math.nan
    n = len(pairs)
    mx = sum(p[0] for p in pairs) / n
    my = sum(p[1] for p in pairs) / n
    cov = sum((p[0] - mx) * (p[1] - my) for p in pairs)
    vx = sum((p[0] - mx) ** 2 for p in pairs)
    vy = sum((p[1] - my) ** 2 for p in pairs)
    if vx <= 0.0 or vy <= 0.0:
        return math.nan
    return cov / math.sqrt(vx * vy)


def compute_ceiling_responsiveness(
    checks: Sequence[SignalCheck],
    best_member_checks: Sequence[BestMemberCheck],
) -> dict[tuple[str, str, str], float]:
    """Per-(knob, signal_id, stage) Pearson r vs the stage's best-member F1.

    The plan's C6 metric: how much does the user-attainable ceiling
    (best-member F1) move with this knob's realised metric across
    baseline → easy → medium → hard? Values near zero flag a *noop
    knob for this stage's ceiling* — the dial may move committee mean,
    but the user picks the best member and that member is immune.

    The four levels share an implicit total order; the metric direction
    encoded on the :class:`SignalExpectation` is ignored — we let the
    sign of the correlation surface naturally. A signal whose expected
    direction is ``"down"`` and whose ceiling also falls with difficulty
    yields a positive correlation. ``"up"`` signals (rare) yield
    negative correlations under the same ceiling-falls-with-difficulty
    pattern. Use ``abs(r)`` if you only care about magnitude.

    Parameters
    ----------
    checks : sequence of SignalCheck
        Per-signal monotonicity results; each carries
        ``check.values`` keyed by level and ``check.expectation``
        (knob, signal_id, stage).
    best_member_checks : sequence of BestMemberCheck
        Per-stage best-member F1 across the four levels.

    Returns
    -------
    dict
        ``{(knob, signal_id, stage): pearson_r}``. The Pearson r is
        ``math.nan`` if the stage has no best-member entry or either
        series degenerates (zero variance / too many NaNs).
    """
    best_by_stage: dict[str, BestMemberCheck] = {
        bm.stage: bm for bm in best_member_checks
    }
    out: dict[tuple[str, str, str], float] = {}
    for check in checks:
        exp = check.expectation
        bm = best_by_stage.get(exp.stage)
        if bm is None:
            out[(exp.knob, exp.signal_id, exp.stage)] = math.nan
            continue
        xs = [check.values.get(level, math.nan) for level in LEVELS]
        ys = [bm.values.get(level, math.nan) for level in LEVELS]
        out[(exp.knob, exp.signal_id, exp.stage)] = _pearson(xs, ys)
    return out


__all__ = [
    "LEVELS",
    "BestMemberCheck",
    "Collapse",
    "SignalCheck",
    "SignalExpectation",
    "check_monotone",
    "collect_best_member_per_level",
    "collect_level_values",
    "compute_ceiling_responsiveness",
    "detect_collapses",
    "load_knob_expected_signals",
    "match_best_member_monotonicity",
    "match_signals",
    "resolve_metric",
]
