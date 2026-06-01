"""
Composite scoring for the end-to-end evaluation panel.

The panel is organised by **Quality dimension × Reference level**
(see ``docs/tutorial/e2e_evaluation/metrics.md``). A composite is
emitted *per reference level* (``RF``, ``SR``, ``GR``) — each is a
weighted average of the quality subdimensions available at that
level.

* **RF** composite reads only the RF sub-blocks. It is a
  structural-only signal — it does NOT incorporate
  ``cluster_correctness`` or ``fact_correctness`` (those require a
  reference). A pipeline can have a high RF composite while
  fabricating values; only SR/GR composites detect that.
* **SR** / **GR** composites read the SR / GR sub-blocks
  respectively and span all six subdimensions.

Default weights add to 1.0. Per-subscore weights are overridable via
the ``weights`` argument of :func:`composite_score`. Three levels are
emitted whenever the corresponding sub-block is available in the
input panel payload.

The composite is documented as "use for ranking pipelines, not for
diagnosing them". The 2D weighting design (Quality × Reference) for
``panel.aggregated`` is still pending — see ``plans/plan_e2e_metrics_v3.md``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default weights per reference level
# ---------------------------------------------------------------------------


DEFAULT_WEIGHTS_RF: Dict[str, float] = {
    "entity_coverage": 0.30,
    "fact_coverage": 0.40,
    "consistency": 0.30,
}


DEFAULT_WEIGHTS_SR: Dict[str, float] = {
    "entity_coverage": 0.10,
    "fact_coverage": 0.10,
    "source_based_fact_coverage": 0.10,
    "consistency": 0.10,
    "cluster_correctness": 0.30,
    "fact_correctness": 0.30,
}


DEFAULT_WEIGHTS_GR: Dict[str, float] = {
    "entity_coverage": 0.10,
    "fact_coverage": 0.10,
    "source_based_fact_coverage": 0.10,
    "consistency": 0.10,
    "cluster_correctness": 0.30,
    "fact_correctness": 0.30,
}


# Back-compat alias for external consumers that still import the old
# ``DEFAULT_WEIGHTS`` constant — these match the SR weights.
DEFAULT_WEIGHTS: Dict[str, float] = DEFAULT_WEIGHTS_SR


_DEFAULT_WEIGHTS_BY_LEVEL: Dict[str, Dict[str, float]] = {
    "RF": DEFAULT_WEIGHTS_RF,
    "SR": DEFAULT_WEIGHTS_SR,
    "GR": DEFAULT_WEIGHTS_GR,
}


_CAVEAT_TEXT = (
    "Composite is a ranking number, not a diagnostic — inspect per-level "
    "subscores and per-attribute metrics under coverage / consistency / "
    "correctness to understand failures. Three composites are emitted "
    "(one per reference level present): RF reflects only structural "
    "signals (entity row-gain shape, output density, format/constraint "
    "validity) and is NOT a correctness score — a pipeline can have a "
    "high RF composite while fabricating values; only SR and GR composites "
    "incorporate cluster_correctness and fact_correctness against a "
    "reference. Compare composites within the same reference level only "
    "— RF, SR, and GR scores are not interchangeable. The 2D aggregated "
    "(Quality × Reference) weighting design remains pending — see "
    "panel.aggregated placeholder."
)


def _clip01(value: Optional[float]) -> float:
    if value is None:
        return 0.0
    try:
        if np.isnan(value):
            return 0.0
    except TypeError:
        return 0.0
    return max(0.0, min(1.0, float(value)))


# ---------------------------------------------------------------------------
# Per-subdimension subscores
#
# Each helper reads the requested reference-level sub-block. Subscores
# that don't apply at a given level (e.g. cluster_correctness at RF)
# return ``None`` so the caller can drop them from the weighted mean.
# ---------------------------------------------------------------------------


def _entity_coverage_subscore(
    coverage: Mapping[str, Any], level: str
) -> Optional[float]:
    """Entity-coverage subscore at the given level."""
    block = (coverage.get("entity") or {}).get(level)
    if block is None:
        return None
    if level == "RF":
        # Structural signal: row_gain penalty (over-merge OR fabrication).
        row_gain = block.get("row_gain_vs_largest_input")
        if row_gain is None:
            return None
        return _clip01(1.0 - min(1.0, abs(float(row_gain))))
    # SR / GR: row diff + recovery + fabricated rate.
    rf = (coverage.get("entity") or {}).get("RF") or {}
    components = []
    rel_diff = block.get("row_count_rel_diff")
    if rel_diff is not None:
        components.append(_clip01(1.0 - abs(rel_diff)))
    recovery_rate = block.get("recovery_rate")
    if recovery_rate is not None:
        components.append(_clip01(recovery_rate))
    n_fabricated = block.get("n_fabricated")
    n_pipe = rf.get("n_rows_output")
    if n_fabricated is not None and n_pipe:
        components.append(_clip01(1.0 - n_fabricated / n_pipe))
    return float(np.mean(components)) if components else 0.0


def _fact_coverage_subscore(coverage: Mapping[str, Any], level: str) -> Optional[float]:
    """Fact-coverage subscore at the given level."""
    block = (coverage.get("fact") or {}).get(level)
    if block is None:
        return None
    if level == "RF":
        density = block.get("density_output")
        if density is None:
            return None
        return _clip01(density)
    # SR / GR: 1 - mean(per_column_drift_normalized).
    per_column = block.get("per_column_drift_normalized") or {}
    drifts = [_clip01(v) for v in per_column.values() if v is not None]
    if not drifts:
        return 1.0  # nothing to compare → don't drag composite down
    return _clip01(1.0 - float(np.mean(drifts)))


def _source_based_fact_coverage_subscore(
    coverage: Mapping[str, Any], level: str
) -> Optional[float]:
    """Source-based fact-coverage subscore at the given level.

    Not defined at RF — the RF source_based block carries only a
    descriptive histogram (``winning_source_distribution_per_attribute``)
    with no single [0, 1] signal.
    """
    if level == "RF":
        return None
    block = (coverage.get("source_based") or {}).get(level)
    if block is None:
        return None
    components = []
    coll = block.get("same_source_collision_rate") or {}
    if "pipe" in coll:
        components.append(_clip01(1.0 - coll["pipe"]))
    js = block.get("source_mix_distribution_js")
    if js is not None:
        components.append(_clip01(1.0 - js))
    attr_js = block.get("source_attribution_js_per_attribute")
    if attr_js:
        components.append(_clip01(1.0 - float(np.mean(list(attr_js.values())))))
    return float(np.mean(components)) if components else 1.0


def _consistency_subscore(
    consistency: Mapping[str, Any], level: str
) -> Optional[float]:
    """Consistency subscore at the given level."""
    block = consistency.get(level)
    if block is None:
        return None
    validity = block.get("validity_per_column") or {}
    if not validity:
        return 1.0
    if level == "RF":
        # No silver-side comparison — use the pipe-only validity rate.
        rates = [
            _clip01(v.get("validity_rate_pipe"))
            for v in validity.values()
            if v.get("n_evaluated_pipe", 0) > 0
        ]
        if not rates:
            return 1.0
        return _clip01(float(np.mean(rates)))
    # SR / GR: only negative deltas penalise — pipeline more strict than
    # reference isn't punished.
    penalties = [
        max(0.0, -v.get("delta", 0.0))
        for v in validity.values()
        if v.get("n_evaluated_pipe", 0) > 0
    ]
    if not penalties:
        return 1.0
    return _clip01(1.0 - float(np.mean(penalties)))


def _cluster_correctness_subscore(
    correctness: Mapping[str, Any], level: str
) -> Optional[float]:
    """Cluster-correctness subscore — not defined at RF."""
    if level == "RF":
        return None
    block = (correctness.get("cluster") or {}).get(level)
    if block is None:
        return None
    components = []
    bcubed = block.get("bcubed") or {}
    if "f1" in bcubed:
        components.append(_clip01(bcubed["f1"]))
    align = block.get("alignment") or {}
    if "mean_jaccard" in align:
        components.append(_clip01(align["mean_jaccard"]))
    return float(np.mean(components)) if components else 0.0


def _fact_correctness_subscore(
    correctness: Mapping[str, Any], level: str
) -> Optional[float]:
    """Fact-correctness subscore — not defined at RF."""
    if level == "RF":
        return None
    block = (correctness.get("fact") or {}).get(level)
    if block is None:
        return None
    components = []
    if "macro_accuracy" in block:
        components.append(_clip01(block["macro_accuracy"]))
    conflict_per_attribute = block.get("conflict_only_per_attribute") or {}
    if "conflict_only_accuracy" in block and conflict_per_attribute:
        components.append(_clip01(block["conflict_only_accuracy"]))
    if "fully_correct_cluster_rate" in block:
        components.append(_clip01(block["fully_correct_cluster_rate"]))
    return float(np.mean(components)) if components else 0.0


# Mapping from subscore name → helper. Helpers read from one of the
# three top-level sections (coverage / consistency / correctness).
_SUBSCORE_HELPERS = {
    "entity_coverage": ("coverage", _entity_coverage_subscore),
    "fact_coverage": ("coverage", _fact_coverage_subscore),
    "source_based_fact_coverage": ("coverage", _source_based_fact_coverage_subscore),
    "consistency": ("consistency", _consistency_subscore),
    "cluster_correctness": ("correctness", _cluster_correctness_subscore),
    "fact_correctness": ("correctness", _fact_correctness_subscore),
}


_SUBSCORE_RECIPES: Dict[str, Dict[str, str]] = {
    "RF": {
        "entity_coverage": (
            "1 - min(1, |coverage.entity.RF.row_gain_vs_largest_input|) — "
            "penalises both over-merge (row_gain << 0) and fabrication "
            "/ under-merge (row_gain >> 0)."
        ),
        "fact_coverage": (
            "coverage.fact.RF.density_output — fraction of non-null cells "
            "across evaluable columns. No drift comparison available "
            "without a reference."
        ),
        "consistency": (
            "mean over columns with n_evaluated_pipe > 0 of "
            "consistency.RF.validity_per_column[col].validity_rate_pipe. "
            "Returns 1.0 when no column was evaluable."
        ),
    },
    "SR": {
        "entity_coverage": (
            "mean(1 - |coverage.entity.SR.row_count_rel_diff|, "
            "coverage.entity.SR.recovery_rate, "
            "1 - coverage.entity.SR.n_fabricated/coverage.entity.RF.n_rows_output)"
        ),
        "fact_coverage": ("1 - mean(coverage.fact.SR.per_column_drift_normalized)"),
        "source_based_fact_coverage": (
            "mean(1 - coverage.source_based.SR.same_source_collision_rate.pipe, "
            "1 - coverage.source_based.SR.source_mix_distribution_js, "
            "1 - mean(coverage.source_based.SR.source_attribution_js_per_attribute))"
        ),
        "consistency": ("1 - mean(max(0, -consistency.SR.validity_per_column.delta))"),
        "cluster_correctness": (
            "mean(correctness.cluster.SR.bcubed.f1, "
            "correctness.cluster.SR.alignment.mean_jaccard)"
        ),
        "fact_correctness": (
            "mean(correctness.fact.SR.macro_accuracy, "
            "correctness.fact.SR.conflict_only_accuracy, "
            "correctness.fact.SR.fully_correct_cluster_rate)"
        ),
    },
    "GR": {
        "entity_coverage": (
            "mean(1 - |coverage.entity.GR.row_count_rel_diff|, "
            "coverage.entity.GR.recovery_rate, "
            "1 - coverage.entity.GR.n_fabricated/coverage.entity.RF.n_rows_output)"
        ),
        "fact_coverage": ("1 - mean(coverage.fact.GR.per_column_drift_normalized)"),
        "source_based_fact_coverage": (
            "mean(1 - coverage.source_based.GR.same_source_collision_rate.pipe, "
            "1 - coverage.source_based.GR.source_mix_distribution_js, "
            "1 - mean(coverage.source_based.GR.source_attribution_js_per_attribute))"
        ),
        "consistency": ("1 - mean(max(0, -consistency.GR.validity_per_column.delta))"),
        "cluster_correctness": (
            "mean(correctness.cluster.GR.bcubed.f1, "
            "correctness.cluster.GR.alignment.mean_jaccard)"
        ),
        "fact_correctness": (
            "mean(correctness.fact.GR.macro_accuracy, "
            "correctness.fact.GR.conflict_only_accuracy, "
            "correctness.fact.GR.fully_correct_cluster_rate)"
        ),
    },
}


# ---------------------------------------------------------------------------
# Top-level composite (per level)
# ---------------------------------------------------------------------------


def _resolve_level_weights(
    level: str,
    weights: Optional[Mapping[str, Any]],
) -> Dict[str, float]:
    """Resolve the effective subscore→weight mapping for *level*.

    Supports two input shapes for ``weights``:

    * **Per-level nested**: ``{'RF': {...}, 'SR': {...}, 'GR': {...}}`` —
      the entry for *level* takes precedence.
    * **Flat**: ``{subscore: weight}`` — applied to SR + GR. Ignored
      for RF (RF subscore set differs and has its own structural
      defaults).
    """
    defaults = dict(_DEFAULT_WEIGHTS_BY_LEVEL[level])
    if not weights:
        return defaults

    is_per_level = all(k in {"RF", "SR", "GR"} for k in weights.keys()) and all(
        isinstance(v, Mapping) for v in weights.values()
    )

    if is_per_level:
        override = weights.get(level)
        if override:
            for k, v in override.items():
                if k in defaults:
                    defaults[k] = float(v)
        return defaults

    # Flat — applied to SR/GR only.
    if level == "RF":
        return defaults
    for k, v in weights.items():
        if k in defaults:
            defaults[k] = float(v)
    return defaults


def _compute_one_level(
    *,
    level: str,
    coverage: Mapping[str, Any],
    consistency: Mapping[str, Any],
    correctness: Mapping[str, Any],
    weights: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Compute the composite payload for a single reference level."""
    resolved_weights = _resolve_level_weights(level, weights)
    sections = {
        "coverage": coverage,
        "consistency": consistency,
        "correctness": correctness,
    }

    subscores: Dict[str, float] = {}
    effective_weights: Dict[str, float] = {}
    for name, weight in resolved_weights.items():
        section_name, helper = _SUBSCORE_HELPERS[name]
        value = helper(sections[section_name], level)
        if value is None:
            continue
        subscores[name] = float(value)
        effective_weights[name] = float(weight)

    total_weight = sum(effective_weights.values()) or 1.0
    composite = (
        sum(effective_weights[name] * subscores[name] for name in subscores)
        / total_weight
    )

    return {
        "composite_score": float(composite),
        "weights": effective_weights,
        "subscores": subscores,
        "subscore_recipe": {name: _SUBSCORE_RECIPES[level][name] for name in subscores},
    }


def composite_score(
    *,
    coverage: Mapping[str, Any],
    consistency: Mapping[str, Any],
    correctness: Mapping[str, Any],
    weights: Optional[Mapping[str, Any]] = None,
    levels: Sequence[str] = ("SR",),
) -> Dict[str, Dict[str, Any]]:
    """Compute the weighted composite score per reference level.

    Parameters
    ----------
    coverage, consistency, correctness : mapping
        Top-level v3 panel sections (same shape
        :func:`panel.compute_e2e_panel` emits).
    weights : mapping, optional
        Either a per-level nested dict
        ``{'RF': {...}, 'SR': {...}, 'GR': {...}}`` or a flat
        ``{subscore: weight}`` dict that applies to SR + GR only.
        Missing subscore keys fall back to the level's defaults
        (:data:`DEFAULT_WEIGHTS_RF`, :data:`DEFAULT_WEIGHTS_SR`,
        :data:`DEFAULT_WEIGHTS_GR`).
    levels : sequence of str, default ``("SR",)``
        Reference levels to compute composites for. Each must be one
        of ``"RF"``, ``"SR"``, ``"GR"``. ``"RF"`` is always computable
        (it only needs the pipeline-side RF sub-blocks); ``"SR"`` /
        ``"GR"`` require the corresponding sub-block to be present.

    Returns
    -------
    dict
        Keyed by level (``"RF"`` / ``"SR"`` / ``"GR"``). Each value
        is a dict with ``composite_score``, ``weights``, ``subscores``,
        and ``subscore_recipe``. A top-level ``"caveat"`` key carries
        the shared caveat text once.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for level in levels:
        if level not in _DEFAULT_WEIGHTS_BY_LEVEL:
            raise ValueError(
                f"Unknown reference level {level!r}; expected one of "
                f"{list(_DEFAULT_WEIGHTS_BY_LEVEL)}"
            )
        out[level] = _compute_one_level(
            level=level,
            coverage=coverage,
            consistency=consistency,
            correctness=correctness,
            weights=weights,
        )

    out["caveat"] = _CAVEAT_TEXT  # type: ignore[assignment]
    return out
