#!/usr/bin/env python3
"""Phase 3 per-variant validation runner.

Runs the SM, EM, and Fusion committees against a packaged variant
(produced by M6 / ``generate_variant.py`` + ``package_variant.py``),
compares each stage's metrics to the baseline (from M5), and persists
per-level metrics, per-pair / per-attribute CSVs, and a human-readable
markdown rollup under
``usecases_synthetic/validation/<domain>/<level>/``.

This is PIPELINE.md Phase 3 (currently ``[todo]``). It is deliberately
measurement-only: monotonicity analysis (M8), collapse handling, and
ablation (M9) live in downstream modules.

Usage
-----
::

    python usecases_synthetic/scripts/validate_variant.py --domain companies --level easy
    python usecases_synthetic/scripts/validate_variant.py --domain companies --level medium
    python usecases_synthetic/scripts/validate_variant.py --domain companies --level hard
    python usecases_synthetic/scripts/validate_variant.py --domain companies --level baseline  # sanity check

Outputs
-------
- ``usecases_synthetic/validation/<domain>/<level>/metrics.json``
- ``usecases_synthetic/validation/<domain>/<level>/level_report.md``
- ``usecases_synthetic/validation/<domain>/<level>/em_per_pair.csv``
- ``usecases_synthetic/validation/<domain>/<level>/fusion_per_attribute.csv``
"""

from __future__ import annotations

# faiss-cpu's libomp collides with torch's libomp on macOS arm64
# (Darwin 25.x): the faiss search loop crashes with
# ``OMP: Error #179: pthread_mutex_init failed`` once any prior import
# has initialised an OpenMP thread pool. Forcing single-threaded
# operation skips the pool init entirely. Set before any other import
# so the env is in place when faiss / torch load downstream.
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import csv
import hashlib
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

# ---------------------------------------------------------------------------
# Ensure repo root is on sys.path.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Load OPENAI_API_KEY (and any other secrets) from .env so committee
# members backed by ChatOpenAI can authenticate.
try:
    from dotenv import load_dotenv  # noqa: E402

    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

import pandas as pd

from usecases_synthetic.lib.baseline_loader import (
    BaselineMetrics,
    load_baseline,
)
from usecases_synthetic.lib.committee import CommitteeResult, Stage
from usecases_synthetic.lib.committee_em import (
    EMBlockingCommitteeRunner,
    EMCommitteeRunner,
    EMMatchingCommitteeRunner,
)
from usecases_synthetic.lib.committee_fusion import FusionCommitteeRunner
from usecases_synthetic.lib.fusion_perfect_clusters import (
    build_perfect_clusters_correspondences,
)
from usecases_synthetic.lib.committee_norm import NormCommitteeRunner
from usecases_synthetic.lib.committee_paths import resolve_committee_path
from usecases_synthetic.lib.committee_sm import SMCommitteeRunner
from usecases_synthetic.lib.domain_config import SYNTHETIC_DIR
from usecases_synthetic.lib.validation_report import write_metrics_json
from usecases_synthetic.lib.variant_loader import VariantBundle, load_variant

logger = logging.getLogger(__name__)

COMMITTEE_DIR: Path = REPO_ROOT / "usecases_synthetic" / "config" / "committees"
VALIDATION_DIR: Path = SYNTHETIC_DIR / "validation"

ALL_STAGES: list[Stage] = ["sm", "norm", "em_blocking", "em_matching", "fusion"]
VALID_LEVELS = {"baseline", "easy", "medium", "hard"}


# ---------------------------------------------------------------------------
# Committee-version pinning
# ---------------------------------------------------------------------------


def _file_sha256(path: Path) -> str:
    """Return the hex SHA-256 digest of a file's contents.

    Parameters
    ----------
    path : Path
        File to hash.

    Returns
    -------
    str
        Hex digest.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# Keep in sync with ``measure_baseline._STAGE_YAML_BASE_NAMES``.  EM is a
# pair of YAMLs after the C2.4b split; both files feed the EM runtime
# and must be hashed so drift detection catches edits to either side.
# Filenames are *base names* (no ``.yaml`` suffix); per S10 of
# ``plans/plan_s1_scale.md``, ``resolve_committee_path`` picks the
# canonical companies file or the per-domain fork.
_STAGE_YAML_BASE_NAMES: dict[Stage, tuple[str, ...]] = {
    "sm": ("sm_committee",),
    "norm": ("normalization_committee",),
    "em_blocking": ("em_blocking_committee",),
    "em_matching": ("em_matching_committee",),
    "fusion": ("fusion_committee",),
}


def _committee_version_string(stage: Stage, domain: str) -> str:
    """Compute the current on-disk ``file@sha12`` marker for a stage.

    Parameters
    ----------
    stage : str
        ``"sm"``, ``"em"``, or ``"fusion"``.
    domain : str
        Domain name. Used to resolve per-domain committee YAML forks via
        :func:`usecases_synthetic.lib.committee_paths.resolve_committee_path`.

    Returns
    -------
    str
        Version marker.  Single-file stages: ``"sm_committee.yaml@12e296681e83"``.
        Multi-file stages (EM): ``"em_blocking_committee.yaml@...+em_matching_committee.yaml@..."``.
        For per-domain forks the filenames carry the ``_<domain>`` suffix
        (e.g. ``"em_blocking_committee_games.yaml@..."``).
    """
    parts: list[str] = []
    for base_name in _STAGE_YAML_BASE_NAMES[stage]:
        path = resolve_committee_path(base_name, domain, committee_dir=COMMITTEE_DIR)
        sha = _file_sha256(path)[:12]
        parts.append(f"{path.name}@{sha}")
    return "+".join(parts)


def _check_committee_versions(
    baseline: BaselineMetrics,
    stages: list[Stage],
    domain: str,
) -> dict[str, str]:
    """Verify current committee YAMLs match the versions recorded in the baseline.

    Parameters
    ----------
    baseline : BaselineMetrics
        Loaded baseline metrics.
    stages : list of str
        Stages we are about to run.

    Returns
    -------
    dict[str, str]
        Current ``{stage: version}`` mapping for provenance.

    Raises
    ------
    RuntimeError
        If any on-disk YAML has drifted from the baseline's recorded version.
    """
    baseline_versions: Mapping[str, Any] = baseline.meta.get("committee_versions", {})
    current: dict[str, str] = {}
    mismatches: list[str] = []
    for stage in stages:
        current_version = _committee_version_string(stage, domain)
        current[stage] = current_version
        expected = baseline_versions.get(stage)
        if expected is None:
            mismatches.append(
                f"{stage}: baseline has no recorded version but stage is in use"
            )
            continue
        if expected != current_version:
            mismatches.append(
                f"{stage}: baseline={expected!r}, on-disk={current_version!r}"
            )
    if mismatches:
        raise RuntimeError(
            "Committee YAML drift detected vs baseline_metrics.json; "
            "re-run measure_baseline.py before validating variants. "
            "Mismatches: " + "; ".join(mismatches)
        )
    return current


# ---------------------------------------------------------------------------
# Delta augmentation
# ---------------------------------------------------------------------------


def _augment_flat_with_delta(
    measured: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> dict[str, float]:
    """Return a dict combining ``measured`` values with ``_baseline`` and ``_delta`` twins.

    Parameters
    ----------
    measured : mapping
        Measured flat metric dict (strings to numbers).
    baseline : mapping
        Baseline flat metric dict.

    Returns
    -------
    dict[str, float]
        Each numeric key from either mapping gets three entries: the
        measured value, ``<key>_baseline``, and ``<key>_delta``.
    """
    out: dict[str, float] = {}
    keys = set(measured) | set(baseline)
    for key in keys:
        try:
            m_val = float(measured.get(key, 0.0))
        except (TypeError, ValueError):
            continue
        try:
            b_val = float(baseline.get(key, 0.0))
        except (TypeError, ValueError):
            b_val = 0.0
        out[key] = m_val
        out[f"{key}_baseline"] = b_val
        out[f"{key}_delta"] = m_val - b_val
    return out


def _augment_nested_with_delta(
    measured: Mapping[str, Mapping[str, Any]],
    baseline: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    """Return a per-group dict with ``_baseline`` and ``_delta`` twins."""
    out: dict[str, dict[str, float]] = {}
    keys = set(measured) | set(baseline)
    for key in keys:
        m_inner = measured.get(key, {}) or {}
        b_inner = baseline.get(key, {}) or {}
        out[key] = _augment_flat_with_delta(m_inner, b_inner)
    return out


def _augment_per_member_with_delta(
    measured_per_member: Mapping[str, Mapping[str, Any]],
    baseline_per_member: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Return per-member dict enriched with ``_baseline`` / ``_delta`` twins.

    Each value of the ``metrics`` sub-dict gains the two twins. Other
    keys (``runtime_s``, ``notes``) pass through unchanged.

    Parameters
    ----------
    measured_per_member : mapping
        ``CommitteeResult.as_dict()["per_member"]`` output.
    baseline_per_member : mapping
        Same shape from the baseline JSON.

    Returns
    -------
    dict[str, dict[str, Any]]
        Augmented per-member dict.
    """
    out: dict[str, dict[str, Any]] = {}
    names = set(measured_per_member) | set(baseline_per_member)
    for name in names:
        m_block = measured_per_member.get(name, {}) or {}
        b_block = baseline_per_member.get(name, {}) or {}
        m_metrics = m_block.get("metrics", {}) or {}
        b_metrics = b_block.get("metrics", {}) or {}
        augmented_metrics = _augment_flat_with_delta(m_metrics, b_metrics)
        out[name] = {
            "metrics": augmented_metrics,
            "runtime_s": m_block.get("runtime_s", 0.0),
            "notes": m_block.get("notes", {}) or {},
        }
    return out


def _augment_stage_block(
    measured_block: Mapping[str, Any],
    baseline_block: Mapping[str, Any],
) -> dict[str, Any]:
    """Augment a single stage's ``CommitteeResult.as_dict()`` with baseline+delta twins.

    Parameters
    ----------
    measured_block : mapping
        Current-variant stage dict.
    baseline_block : mapping
        Baseline stage dict (from baseline_metrics.json).

    Returns
    -------
    dict[str, Any]
        Augmented stage block suitable for metrics.json.
    """
    aggregated = _augment_flat_with_delta(
        measured_block.get("aggregated", {}) or {},
        baseline_block.get("aggregated", {}) or {},
    )
    per_attribute = _augment_nested_with_delta(
        measured_block.get("per_attribute", {}) or {},
        baseline_block.get("per_attribute", {}) or {},
    )
    per_partition = _augment_nested_with_delta(
        measured_block.get("per_partition", {}) or {},
        baseline_block.get("per_partition", {}) or {},
    )
    per_member = _augment_per_member_with_delta(
        measured_block.get("per_member", {}) or {},
        baseline_block.get("per_member", {}) or {},
    )
    return {
        "stage": measured_block.get("stage"),
        "domain": measured_block.get("domain"),
        "level": measured_block.get("level"),
        "runtime_s": measured_block.get("runtime_s", 0.0),
        "roster": list(measured_block.get("roster", []) or []),
        "aggregated": aggregated,
        "per_attribute": per_attribute,
        "per_partition": per_partition,
        "per_member": per_member,
    }


# ---------------------------------------------------------------------------
# Per-stage runners
# ---------------------------------------------------------------------------


def _run_sm(
    bundle: VariantBundle,
    *,
    with_llm: bool,
) -> CommitteeResult:
    """Run the SM committee against ``bundle``."""
    runner = SMCommitteeRunner(
        resolve_committee_path(
            "sm_committee", bundle.domain, committee_dir=COMMITTEE_DIR
        ),
        with_llm=with_llm,
    )
    return runner.run(bundle)


def _run_norm(
    bundle: VariantBundle,
    *,
    with_llm: bool,
    scoring_surface: str,
) -> CommitteeResult:
    """Run the Normalization committee against ``bundle``.

    ``scoring_surface`` is read from the baseline meta by the caller so
    every variant is scored against the same surface the baseline was
    measured with (``"schema_constraints"`` compares to the canonical
    target_schema constraints; ``"xml_targets"`` is the legacy fusion
    val/test comparison).
    """
    runner = NormCommitteeRunner(
        resolve_committee_path(
            "normalization_committee", bundle.domain, committee_dir=COMMITTEE_DIR
        ),
        with_llm=with_llm,
        scoring_surface=scoring_surface,
    )
    return runner.run(bundle)


def _run_em_blocking(
    bundle: VariantBundle,
) -> CommitteeResult:
    """Run the EM blocking committee against ``bundle``.

    Mirrors ``measure_baseline.py``'s split EM pipeline so variant
    metrics line up with the baseline. Blocking emits per-blocker
    ``pair_recall`` / ``reduction_ratio``; the matching half is run
    separately by :func:`_run_em_matching` so LLM matchers are fed the
    labelled gold pairs directly (not the full blocker output —
    plan_s1_final.md S.7 path).
    """
    runner = EMBlockingCommitteeRunner(
        resolve_committee_path(
            "em_blocking_committee", bundle.domain, committee_dir=COMMITTEE_DIR
        ),
    )
    return runner.run(bundle)


def _run_em_matching(
    bundle: VariantBundle,
    *,
    with_llm: bool,
) -> CommitteeResult:
    """Run the EM matching committee against ``bundle``.

    Uses the perfect-prior-step matching runner: each matcher sees the
    labelled ``_val.csv`` + ``_test.csv`` pair set as candidates (not the
    blocker output), so LLM matchers cost O(|gold|) prompts rather than
    O(|blocker candidates|). Score primary headline is
    ``macro_f1_regen_test`` per the closed-set semantic on the
    corner-filled test split (plan_revision.md C10/C11).
    """
    runner = EMMatchingCommitteeRunner(
        resolve_committee_path(
            "em_matching_committee", bundle.domain, committee_dir=COMMITTEE_DIR
        ),
        with_llm=with_llm,
    )
    return runner.run(bundle)


def _run_fusion(
    bundle: VariantBundle,
    correspondences: pd.DataFrame | None,
) -> CommitteeResult:
    """Run the Fusion committee. ``correspondences=None`` uses gold pairs."""
    runner = FusionCommitteeRunner(
        resolve_committee_path(
            "fusion_committee", bundle.domain, committee_dir=COMMITTEE_DIR
        )
    )
    return runner.run(bundle, correspondences=correspondences)


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------


def _write_em_per_pair_csv(
    path: Path,
    measured_block: Mapping[str, Any],
    baseline_block: Mapping[str, Any],
) -> Path:
    """Write the per-pair EM CSV: one row per (member, pair) with F1 + pool diag.

    Parameters
    ----------
    path : Path
        Destination file.
    measured_block : mapping
        Measured EM stage block (``CommitteeResult.as_dict()``).
    baseline_block : mapping
        Baseline EM stage block.

    Returns
    -------
    Path
        The written path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    measured_members = measured_block.get("per_member", {}) or {}
    baseline_members = baseline_block.get("per_member", {}) or {}
    for member_name, member_block in measured_members.items():
        m_per_pair = (member_block.get("notes", {}) or {}).get("per_pair", {}) or {}
        b_member = baseline_members.get(member_name, {}) or {}
        b_per_pair = (b_member.get("notes", {}) or {}).get("per_pair", {}) or {}
        for pair_key, pair_metrics in m_per_pair.items():
            b_pair_metrics = b_per_pair.get(pair_key, {}) or {}
            f1 = float(pair_metrics.get("f1", 0.0))
            f1_baseline = float(b_pair_metrics.get("f1", 0.0))
            rows.append(
                {
                    "member": member_name,
                    "pair": pair_key,
                    "f1": f1,
                    "f1_baseline": f1_baseline,
                    "f1_delta": f1 - f1_baseline,
                    "precision": float(pair_metrics.get("precision", 0.0)),
                    "recall": float(pair_metrics.get("recall", 0.0)),
                    # R7b dual-model dual-test (4 cells per pair × member).
                    "f1_baseline_model_on_baseline_test": float(
                        pair_metrics.get(
                            "f1_baseline_model_on_baseline_test", float("nan")
                        )
                    ),
                    "f1_baseline_model_on_regen_test": float(
                        pair_metrics.get(
                            "f1_baseline_model_on_regen_test", float("nan")
                        )
                    ),
                    "f1_variant_model_on_baseline_test": float(
                        pair_metrics.get(
                            "f1_variant_model_on_baseline_test", float("nan")
                        )
                    ),
                    "f1_variant_model_on_regen_test": float(
                        pair_metrics.get("f1_variant_model_on_regen_test", float("nan"))
                    ),
                    "variant_model_distinct": float(
                        pair_metrics.get("variant_model_distinct", 0.0)
                    ),
                    # Pre-R7b legacy aliases (= baseline-model surfaces).
                    "f1_baseline_test": float(
                        pair_metrics.get("f1_baseline_test", float("nan"))
                    ),
                    "f1_regen_test": float(
                        pair_metrics.get("f1_regen_test", float("nan"))
                    ),
                    "f1_vs_pool": float(pair_metrics.get("f1_vs_pool", float("nan"))),
                    "pool_precision": float(pair_metrics.get("pool_precision", 0.0)),
                    "pool_recall": float(pair_metrics.get("pool_recall", 0.0)),
                    "pool_precision_baseline": float(
                        b_pair_metrics.get("pool_precision", 0.0)
                    ),
                    "pool_recall_baseline": float(
                        b_pair_metrics.get("pool_recall", 0.0)
                    ),
                }
            )
    fieldnames = [
        "member",
        "pair",
        "f1",
        "f1_baseline",
        "f1_delta",
        "precision",
        "recall",
        # R7b dual-model dual-test.
        "f1_baseline_model_on_baseline_test",
        "f1_baseline_model_on_regen_test",
        "f1_variant_model_on_baseline_test",
        "f1_variant_model_on_regen_test",
        "variant_model_distinct",
        # Pre-R7b legacy aliases.
        "f1_baseline_test",
        "f1_regen_test",
        "f1_vs_pool",
        "pool_precision",
        "pool_recall",
        "pool_precision_baseline",
        "pool_recall_baseline",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _write_fusion_per_attribute_csv(
    path: Path,
    measured_block: Mapping[str, Any],
    baseline_block: Mapping[str, Any],
) -> Path:
    """Write the per-attribute fusion CSV with best-strategy accuracy + deltas.

    Parameters
    ----------
    path : Path
        Destination file.
    measured_block : mapping
        Measured fusion stage block.
    baseline_block : mapping
        Baseline fusion stage block.

    Returns
    -------
    Path
        The written path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    measured_attrs = measured_block.get("per_attribute", {}) or {}
    baseline_attrs = baseline_block.get("per_attribute", {}) or {}
    rows: list[dict[str, Any]] = []
    for attr, metrics in measured_attrs.items():
        b_metrics = baseline_attrs.get(attr, {}) or {}
        best = float(metrics.get("best_strategy_accuracy", 0.0))
        best_baseline = float(b_metrics.get("best_strategy_accuracy", 0.0))
        spread = float(metrics.get("spread", 0.0))
        spread_baseline = float(b_metrics.get("spread", 0.0))
        mean_acc = float(metrics.get("mean_strategy_accuracy", 0.0))
        mean_baseline = float(b_metrics.get("mean_strategy_accuracy", 0.0))
        rows.append(
            {
                "attribute": attr,
                "best_accuracy": best,
                "best_accuracy_baseline": best_baseline,
                "best_accuracy_delta": best - best_baseline,
                "mean_accuracy": mean_acc,
                "mean_accuracy_baseline": mean_baseline,
                "mean_accuracy_delta": mean_acc - mean_baseline,
                "spread": spread,
                "spread_baseline": spread_baseline,
                "spread_delta": spread - spread_baseline,
            }
        )
    fieldnames = [
        "attribute",
        "best_accuracy",
        "best_accuracy_baseline",
        "best_accuracy_delta",
        "mean_accuracy",
        "mean_accuracy_baseline",
        "mean_accuracy_delta",
        "spread",
        "spread_baseline",
        "spread_delta",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def _format_float(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _stage_summary_row(
    stage: str,
    measured: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> list[str]:
    """Return a single Markdown table row describing the macro metric of a stage.

    Parameters
    ----------
    stage : str
        ``"sm"``, ``"em_blocking"``, ``"em_matching"``, or ``"fusion"``.
    measured : mapping
        Measured aggregated metrics for this stage.
    baseline : mapping
        Baseline aggregated metrics for this stage.

    Returns
    -------
    list of str
        Cells: ``[stage, metric_name, measured, baseline, delta]``.
    """
    # R7b: EM stage headline = variant_model_on_regen_test (the
    # load-bearing surface for monotonicity per plan_revision.md R7b).
    # Pre-R7b runs that don't carry the new key fall back to the legacy
    # ``macro_f1_regen_test`` / ``macro_pair_recall`` aliases.
    if stage == "em_blocking":
        if "macro_pair_recall_variant_model_on_regen_test" in measured:
            metric_name = "macro_pair_recall_variant_model_on_regen_test"
        else:
            metric_name = "macro_pair_recall"
    elif stage == "em_matching":
        if "macro_f1_variant_model_on_regen_test" in measured:
            metric_name = "macro_f1_variant_model_on_regen_test"
        else:
            metric_name = "macro_f1_regen_test"
    elif stage in ("sm", "norm"):
        metric_name = "macro_f1"
    else:
        metric_name = "overall_accuracy"
    m_val = measured.get(metric_name, 0.0)
    b_val = baseline.get(metric_name, 0.0)
    try:
        d_val = float(m_val) - float(b_val)
    except (TypeError, ValueError):
        d_val = 0.0
    return [
        stage,
        metric_name,
        _format_float(m_val),
        _format_float(b_val),
        _format_float(d_val),
    ]


def _write_level_report_md(
    path: Path,
    *,
    domain: str,
    level: str,
    with_llm: bool,
    measured_per_stage: Mapping[str, Mapping[str, Any]],
    baseline_per_stage: Mapping[str, Mapping[str, Any]],
    committee_versions: Mapping[str, str],
) -> Path:
    """Render the human-readable level report.

    Parameters
    ----------
    path : Path
        Destination file.
    domain, level : str
        Identifiers for the variant.
    with_llm : bool
        Whether LLM members were enabled.
    measured_per_stage : mapping
        Per-stage measured dicts.
    baseline_per_stage : mapping
        Per-stage baseline dicts.
    committee_versions : mapping
        ``{stage: "<filename>@<sha12>"}`` provenance.

    Returns
    -------
    Path
        The written path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append(f"# Validation report - {domain} / {level}")
    lines.append("")
    lines.append(f"_Generated at {datetime.now(timezone.utc).isoformat()}_")
    lines.append("")
    lines.append(f"- domain: `{domain}`")
    lines.append(f"- level: `{level}`")
    lines.append(f"- with_llm: `{with_llm}`")
    versions_str = ", ".join(
        f"{stage}=`{v}`" for stage, v in committee_versions.items()
    )
    lines.append(f"- committee_versions: {versions_str}")
    lines.append("")

    # Stage summary table
    lines.append("## Stage summary")
    lines.append("")
    lines.append("| stage | metric | measured | baseline | delta |")
    lines.append("|---|---|---|---|---|")
    for stage in ("sm", "norm", "em_blocking", "em_matching", "fusion"):
        measured_block = measured_per_stage.get(stage)
        if not measured_block:
            continue
        baseline_block = baseline_per_stage.get(stage, {}) or {}
        row = _stage_summary_row(
            stage,
            measured_block.get("aggregated", {}) or {},
            baseline_block.get("aggregated", {}) or {},
        )
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Per-stage per-member tables
    for stage in ("sm", "norm", "em_blocking", "em_matching", "fusion"):
        measured_block = measured_per_stage.get(stage)
        if not measured_block:
            continue
        baseline_block = baseline_per_stage.get(stage, {}) or {}
        lines.extend(_per_member_table(stage, measured_block, baseline_block))
        if stage == "em_matching":
            lines.extend(_em_per_pair_table(measured_block, baseline_block))
        if stage == "fusion":
            lines.extend(_fusion_per_attribute_table(measured_block, baseline_block))

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")
    return path


def _per_member_table(
    stage: str,
    measured_block: Mapping[str, Any],
    baseline_block: Mapping[str, Any],
) -> list[str]:
    """Markdown table: one row per committee member with the headline metric."""
    lines: list[str] = []
    lines.append(f"## Stage: {stage} - per member")
    lines.append("")

    if stage == "sm":
        metric_name = "f1"
    elif stage == "norm":
        metric_name = "macro_f1"
    elif stage == "em_blocking":
        metric_name = "pair_recall"
    elif stage == "em_matching":
        metric_name = "f1"
    else:
        metric_name = "overall_accuracy"

    extra_cols: list[str] = []
    if stage == "em_matching":
        extra_cols = ["f1_baseline_test", "f1_regen_test"]
    elif stage == "em_blocking":
        extra_cols = ["reduction_ratio"]

    header = [
        "member",
        f"{metric_name}",
        f"{metric_name}_baseline",
        f"{metric_name}_delta",
    ]
    for extra in extra_cols:
        header.extend([extra, f"{extra}_baseline"])
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    measured_members = measured_block.get("per_member", {}) or {}
    baseline_members = baseline_block.get("per_member", {}) or {}
    for name in sorted(measured_members):
        m_metrics = measured_members[name].get("metrics", {}) or {}
        b_metrics = (baseline_members.get(name, {}) or {}).get("metrics", {}) or {}
        m_val = float(m_metrics.get(metric_name, 0.0))
        b_val = float(b_metrics.get(metric_name, 0.0))
        row = [
            name,
            _format_float(m_val),
            _format_float(b_val),
            _format_float(m_val - b_val),
        ]
        for extra in extra_cols:
            row.extend(
                [
                    _format_float(m_metrics.get(extra, 0.0)),
                    _format_float(b_metrics.get(extra, 0.0)),
                ]
            )
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return lines


def _em_per_pair_table(
    measured_block: Mapping[str, Any],
    baseline_block: Mapping[str, Any],
) -> list[str]:
    """Markdown table: per-(pair, member) EM F1 with deltas."""
    lines: list[str] = [
        "## Stage: em_matching - per pair",
        "",
        "| pair | member | f1 | f1_baseline | f1_delta |",
        "|---|---|---|---|---|",
    ]
    measured_members = measured_block.get("per_member", {}) or {}
    baseline_members = baseline_block.get("per_member", {}) or {}
    rows: list[tuple[str, str, float, float]] = []
    for name in sorted(measured_members):
        m_pp = (measured_members[name].get("notes", {}) or {}).get("per_pair", {}) or {}
        b_pp = ((baseline_members.get(name, {}) or {}).get("notes", {}) or {}).get(
            "per_pair", {}
        ) or {}
        for pair_key in sorted(m_pp):
            f1 = float(m_pp[pair_key].get("f1", 0.0))
            f1_b = float((b_pp.get(pair_key, {}) or {}).get("f1", 0.0))
            rows.append((pair_key, name, f1, f1_b))
    rows.sort(key=lambda r: (r[0], r[1]))
    for pair_key, name, f1, f1_b in rows:
        lines.append(
            f"| {pair_key} | {name} | {_format_float(f1)} | "
            f"{_format_float(f1_b)} | {_format_float(f1 - f1_b)} |"
        )
    lines.append("")
    return lines


def _fusion_per_attribute_table(
    measured_block: Mapping[str, Any],
    baseline_block: Mapping[str, Any],
) -> list[str]:
    """Markdown table: per-attribute fusion best accuracy with spread + deltas."""
    lines: list[str] = [
        "## Stage: fusion - per attribute",
        "",
        "| attribute | best_accuracy | baseline | delta | spread | spread_baseline | spread_delta |",
        "|---|---|---|---|---|---|---|",
    ]
    measured_attrs = measured_block.get("per_attribute", {}) or {}
    baseline_attrs = baseline_block.get("per_attribute", {}) or {}
    for attr in sorted(measured_attrs):
        m = measured_attrs[attr]
        b = baseline_attrs.get(attr, {}) or {}
        best = float(m.get("best_strategy_accuracy", 0.0))
        best_b = float(b.get("best_strategy_accuracy", 0.0))
        spread = float(m.get("spread", 0.0))
        spread_b = float(b.get("spread", 0.0))
        lines.append(
            "| "
            + " | ".join(
                [
                    attr,
                    _format_float(best),
                    _format_float(best_b),
                    _format_float(best - best_b),
                    _format_float(spread),
                    _format_float(spread_b),
                    _format_float(spread - spread_b),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Main validation logic
# ---------------------------------------------------------------------------


def validate_variant(
    domain: str,
    level: str,
    *,
    stages: list[Stage] | None = None,
    with_llm: bool = False,
    fusion_input_member: str | None = None,
    out_dir: Path | None = None,
    variant_root: Path | None = None,
) -> dict[str, Any]:
    """Run committees against a packaged variant and persist per-level metrics.

    Parameters
    ----------
    domain : str
        Domain name (e.g. ``"companies"``).
    level : str
        One of ``"baseline"``, ``"easy"``, ``"medium"``, ``"hard"``.
    stages : list of str, optional
        Stages to run. Default: all three.
    with_llm : bool
        Toggle LLM committee members. Must match the baseline's
        ``with_llm`` flag, else :class:`RuntimeError` is raised.
    fusion_input_member : str, optional
        Override the EM member whose correspondences feed Fusion.
        Default: the baseline's recorded ``fusion_input_member``.
    out_dir : Path, optional
        Destination directory. Default:
        ``usecases_synthetic/validation/<domain>/<level>/``.
    variant_root : Path, optional
        Override the variant root directory to validate against. When
        set, ``level`` is used only to gate committee-version checks
        and caching behaviour (typically ``"hard"`` for ablation runs);
        the actual bundle is loaded from ``variant_root``.

    Returns
    -------
    dict[str, Any]
        The payload written to ``metrics.json``.

    Raises
    ------
    ValueError
        If ``level`` is invalid.
    RuntimeError
        On committee-version drift, ``with_llm`` mismatch, or missing
        baseline fusion_input_member when EM is requested.
    """
    if level not in VALID_LEVELS:
        raise ValueError(
            f"Invalid level {level!r}; expected one of {sorted(VALID_LEVELS)}"
        )

    active_stages: list[Stage] = stages or list(ALL_STAGES)
    if out_dir is None:
        out_dir = VALIDATION_DIR / domain / level

    logger.info("Loading baseline metrics for domain=%s", domain)
    baseline = load_baseline(domain)

    # Cross-check with_llm.
    baseline_with_llm = bool(baseline.meta.get("with_llm", False))
    if baseline_with_llm != with_llm:
        raise RuntimeError(
            f"with_llm mismatch: baseline={baseline_with_llm}, "
            f"requested={with_llm}. Re-run measure_baseline.py with the "
            "same flag."
        )

    # Resolve fusion_input_member from baseline.
    baseline_fusion_input: str = str(baseline.meta.get("fusion_input_member", ""))
    active_fusion_input: str = (
        fusion_input_member
        if fusion_input_member is not None
        else baseline_fusion_input
    )

    # Resolve the Norm scoring surface from the baseline so the variant is
    # scored against the exact surface the baseline was measured with
    # (no independent override — variant scoring must equal baseline
    # scoring). Baselines predating the surface knob carry no key and
    # fall back to the legacy "xml_targets", matching their norm numbers.
    norm_scoring_surface: str = str(baseline.meta.get("scoring_surface", "xml_targets"))

    # Verify committee YAMLs haven't drifted.
    current_versions = _check_committee_versions(baseline, active_stages, domain)

    if variant_root is not None:
        logger.info(
            "Loading variant bundle for domain=%s level=%s root=%s",
            domain,
            level,
            variant_root,
        )
        bundle = load_variant(domain, level=level, root_override=variant_root)
    else:
        logger.info("Loading variant bundle for domain=%s level=%s", domain, level)
        bundle = load_variant(domain, level=level)

    measured_per_stage: dict[str, dict[str, Any]] = {}
    em_blocking_result: CommitteeResult | None = None
    em_matching_result: CommitteeResult | None = None
    total_t0 = time.monotonic()

    # --- SM ---
    if "sm" in active_stages:
        logger.info("Running SM committee...")
        sm_result = _run_sm(bundle, with_llm=with_llm)
        measured_per_stage["sm"] = sm_result.as_dict()
        logger.info(
            "SM done: macro_f1=%.4f best=%s (f1=%.4f) (%.1fs)",
            sm_result.aggregated.get("macro_f1", 0.0),
            sm_result.aggregated.get("best_member_name", "?"),
            sm_result.aggregated.get("best_member_f1", 0.0),
            sm_result.runtime_s,
        )

    # --- Normalization ---
    if "norm" in active_stages:
        logger.info(
            "Running Normalization committee (scoring_surface=%s)...",
            norm_scoring_surface,
        )
        norm_result = _run_norm(
            bundle, with_llm=with_llm, scoring_surface=norm_scoring_surface
        )
        measured_per_stage["norm"] = norm_result.as_dict()
        logger.info(
            "Norm done: macro_f1=%.4f best_member_f1=%.4f (%.1fs)",
            norm_result.aggregated.get("macro_f1", 0.0),
            norm_result.aggregated.get("best_member_f1", 0.0),
            norm_result.runtime_s,
        )

    # --- EM blocking ---
    if "em_blocking" in active_stages:
        logger.info("Running EM blocking committee...")
        em_blocking_result = _run_em_blocking(bundle)
        measured_per_stage["em_blocking"] = em_blocking_result.as_dict()
        # R7b: log the variant-model-on-regen surface (load-bearing for
        # monotonicity), falling back to the legacy macro_pair_recall
        # alias for older committee outputs.
        em_blk_agg = em_blocking_result.aggregated
        em_blk_headline = em_blk_agg.get(
            "macro_pair_recall_variant_model_on_regen_test",
            em_blk_agg.get("macro_pair_recall", 0.0),
        )
        logger.info(
            "EM blocking done: macro_pair_recall_variant_model_on_regen_test=%.4f"
            " (%.1fs)",
            em_blk_headline,
            em_blocking_result.runtime_s,
        )

    # --- EM matching ---
    if "em_matching" in active_stages:
        logger.info("Running EM matching committee...")
        em_matching_result = _run_em_matching(bundle, with_llm=with_llm)
        measured_per_stage["em_matching"] = em_matching_result.as_dict()
        em_mat_agg = em_matching_result.aggregated
        em_mat_headline = em_mat_agg.get(
            "macro_f1_variant_model_on_regen_test",
            em_mat_agg.get("macro_f1_regen_test", 0.0),
        )
        logger.info(
            "EM matching done: macro_f1_variant_model_on_regen_test=%.4f (%.1fs)",
            em_mat_headline,
            em_matching_result.runtime_s,
        )

    # --- Fusion ---
    # Per R5 Fusion design (plans/plan_s1_scale.md, 2026-05-12): every
    # committee is evaluated against the **perfect** output of the prior
    # pipeline step. For fusion that means assuming EM produced the
    # ground-truth clusters declared in the R3 pool — record IDs survive
    # every K-knob mutation per the variant provenance contract, so the
    # same pool defines perfect clusters for both baseline AND every
    # variant of the same domain.
    if "fusion" in active_stages:
        logger.info("Running Fusion committee...")
        correspondences = build_perfect_clusters_correspondences(domain, bundle)
        logger.info(
            "Fusion using perfect-cluster correspondences: n=%d",
            len(correspondences),
        )
        fusion_result = _run_fusion(bundle, correspondences)
        measured_per_stage["fusion"] = fusion_result.as_dict()
        logger.info(
            "Fusion done: overall_accuracy=%.4f (%.1fs)",
            fusion_result.aggregated.get("overall_accuracy", 0.0),
            fusion_result.runtime_s,
        )

    total_runtime = time.monotonic() - total_t0

    # Augment measured blocks with baseline / delta twins.
    augmented_per_stage: dict[str, dict[str, Any]] = {}
    for stage, block in measured_per_stage.items():
        baseline_block = baseline.per_stage.get(stage, {}) or {}
        augmented_per_stage[stage] = _augment_stage_block(block, baseline_block)

    # Canonical baseline path, for provenance.
    from usecases_synthetic.lib.baseline_loader import baseline_path

    baseline_source = str(baseline_path(domain).relative_to(REPO_ROOT))

    meta: dict[str, Any] = {
        "level": level,
        "with_llm": with_llm,
        "baseline_source": baseline_source,
        "committee_versions": current_versions,
        "fusion_input_member": active_fusion_input,
        "total_runtime_s": round(total_runtime, 2),
    }
    # Provenance: the Norm surface this variant was scored against (== the
    # baseline's, by construction). Stamped only when norm actually ran.
    if "norm" in active_stages:
        meta["scoring_surface"] = norm_scoring_surface

    json_path = out_dir / "metrics.json"
    write_metrics_json(
        json_path,
        domain=domain,
        per_stage=augmented_per_stage,
        meta=meta,
    )
    logger.info("Wrote %s", json_path)

    # Per-stage CSVs.
    if "em_matching" in active_stages:
        csv_path = out_dir / "em_per_pair.csv"
        _write_em_per_pair_csv(
            csv_path,
            measured_per_stage["em_matching"],
            baseline.per_stage.get("em_matching", {}) or {},
        )
        logger.info("Wrote %s", csv_path)
    if "fusion" in active_stages:
        csv_path = out_dir / "fusion_per_attribute.csv"
        _write_fusion_per_attribute_csv(
            csv_path,
            measured_per_stage["fusion"],
            baseline.per_stage.get("fusion", {}) or {},
        )
        logger.info("Wrote %s", csv_path)

    # Markdown rollup.
    md_path = out_dir / "level_report.md"
    _write_level_report_md(
        md_path,
        domain=domain,
        level=level,
        with_llm=with_llm,
        measured_per_stage=measured_per_stage,
        baseline_per_stage=baseline.per_stage,
        committee_versions=current_versions,
    )
    logger.info("Wrote %s", md_path)

    return {
        "domain": domain,
        "meta": meta,
        "per_stage": augmented_per_stage,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_stages(raw: str) -> list[Stage]:
    """Parse a comma-separated stages string.

    Parameters
    ----------
    raw : str
        e.g. ``"sm,em"``.

    Returns
    -------
    list of Stage
        Validated stage names.

    Raises
    ------
    ValueError
        If an unknown stage is specified.
    """
    parts = [s.strip() for s in raw.split(",") if s.strip()]
    for s in parts:
        if s not in ALL_STAGES:
            raise ValueError(f"Unknown stage: {s!r}. Valid: {ALL_STAGES}")
    return parts  # type: ignore[return-value]


def main(argv: list[str] | None = None) -> None:
    """CLI entry point.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments. Default: ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(
        description="Run committees against a packaged variant and "
        "persist per-level metrics with deltas vs the baseline.",
    )
    parser.add_argument(
        "--domain",
        required=True,
        help="Domain name (e.g. 'companies').",
    )
    parser.add_argument(
        "--level",
        required=True,
        choices=sorted(VALID_LEVELS),
        help="Variant level.",
    )
    parser.add_argument(
        "--with-llm",
        action="store_true",
        default=False,
        help="Include LLM committee members. Must match baseline.",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default=None,
        help="Comma-separated stages (e.g. 'sm,em'). Default: all three.",
    )
    parser.add_argument(
        "--fusion-input-member",
        type=str,
        default=None,
        help="Override the EM member feeding Fusion. Default: baseline value.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Override output directory. Default: "
        "usecases_synthetic/validation/<domain>/<level>/.",
    )
    parser.add_argument(
        "--variant-root",
        type=Path,
        default=None,
        help=(
            "Override the variant root directory to load. Used by the "
            "ablation runner to point validate_variant at "
            "usecases/<domain>-augmented/ablation_knob_<id>/ while "
            "still running as level=hard. Default: derived from --level."
        ),
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    stages: list[Stage] | None = None
    if args.stages is not None:
        stages = _parse_stages(args.stages)

    validate_variant(
        domain=args.domain,
        level=args.level,
        stages=stages,
        with_llm=args.with_llm,
        fusion_input_member=args.fusion_input_member,
        out_dir=args.out_dir,
        variant_root=args.variant_root,
    )


if __name__ == "__main__":
    main()
