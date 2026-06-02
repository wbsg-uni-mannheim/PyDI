#!/usr/bin/env python3
"""Measure baseline committee metrics on original (unaugmented) data.

Phase 1 of ``usecases_synthetic/PIPELINE.md``: run SM, EM, and Fusion
committees on the original ``usecases/<domain>/`` data and persist the
per-stage, per-member, per-attribute metrics as the reference point for
all subsequent variant validation.

The output is committed to the repo so validation runs (M7/M8) are
reproducible without re-measuring.

Usage
-----
::

    python usecases_synthetic/scripts/measure_baseline.py --domain companies
    python usecases_synthetic/scripts/measure_baseline.py --domain companies --with-llm
    python usecases_synthetic/scripts/measure_baseline.py --domain companies --stages sm,em

Outputs
-------
- ``usecases_synthetic/baselines/<domain>/baseline_metrics.json``
- ``usecases_synthetic/baselines/<domain>/baseline_report.md``
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
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

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

from usecases_synthetic.lib.baseline_loader import BASELINES_DIR, load_baseline
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
from usecases_synthetic.lib.validation_report import write_metrics_json, write_report_md
from usecases_synthetic.lib.variant_loader import load_variant

logger = logging.getLogger(__name__)

COMMITTEE_DIR: Path = REPO_ROOT / "usecases_synthetic" / "config" / "committees"

ALL_STAGES: list[Stage] = ["sm", "norm", "em_blocking", "em_matching", "fusion"]


# ---------------------------------------------------------------------------
# Helpers
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


# Files that actually drive each committee's runtime.  EM is a pair after
# the C2.4b split (see plans/plan_committee_finalization.md): both YAMLs
# must be hashed so drift detection catches edits to either file.  The
# combined ``file@sha+file@sha`` format keeps a single ``em`` stage entry
# while pinning both files; ``_check_committee_versions`` in
# ``validate_variant.py`` computes the same string so exact-match checks
# continue to work. Filenames here are *base names* (no ``.yaml`` suffix);
# ``resolve_committee_path`` from ``committee_paths`` picks the canonical
# (companies) or per-domain fork (per S10).
_STAGE_YAML_BASE_NAMES: dict[Stage, tuple[str, ...]] = {
    "sm": ("sm_committee",),
    "norm": ("normalization_committee",),
    # ``em`` (legacy bundled stage) and the split ``em_blocking`` /
    # ``em_matching`` stages all hash the same two YAMLs; the split
    # surface so downstream drift detection treats them as different
    # stage names without re-implementing the hash machinery.
    "em": ("em_blocking_committee", "em_matching_committee"),
    "em_blocking": ("em_blocking_committee",),
    "em_matching": ("em_matching_committee",),
    "fusion": ("fusion_committee",),
}


def _committee_versions(stages: list[Stage], domain: str) -> dict[str, str]:
    """Build a ``{stage: "file@sha[+file@sha]"}`` dict for provenance.

    Parameters
    ----------
    stages : list of str
        Stages being measured.
    domain : str
        Domain name. Used to resolve per-domain committee YAML forks via
        :func:`usecases_synthetic.lib.committee_paths.resolve_committee_path`.

    Returns
    -------
    dict[str, str]
        Version strings keyed by stage name.  Multi-file stages (``em``
        after the C2.4b split) emit ``file1@sha1+file2@sha2`` so both
        source files are pinned under a single stage key. Hashed file is
        the resolved per-domain fork (or canonical companies file when
        no fork exists for that domain).
    """
    versions: dict[str, str] = {}
    for stage in stages:
        parts: list[str] = []
        for base_name in _STAGE_YAML_BASE_NAMES[stage]:
            path = resolve_committee_path(
                base_name, domain, committee_dir=COMMITTEE_DIR
            )
            if not path.exists():
                continue
            sha = _file_sha256(path)[:12]
            parts.append(f"{path.name}@{sha}")
        if parts:
            versions[stage] = "+".join(parts)
    return versions


def _best_em_member(em_result: CommitteeResult) -> str:
    """Return the name of the EM member with the highest macro F1.

    Parameters
    ----------
    em_result : CommitteeResult
        The EM committee result.

    Returns
    -------
    str
        Best member name. Empty string if no members ran.
    """
    best_name = ""
    best_f1 = -1.0
    for name, member in em_result.per_member.items():
        f1 = member.metrics.get("f1", 0.0)
        if f1 > best_f1:
            best_f1 = f1
            best_name = name
    return best_name


# ---------------------------------------------------------------------------
# Main measurement logic
# ---------------------------------------------------------------------------


def measure_baseline(
    domain: str,
    *,
    stages: list[Stage] | None = None,
    with_llm: bool = False,
    out_dir: Path | None = None,
    fusion_input_member: str | None = None,
) -> dict[str, Any]:
    """Run committee measurement on the original (baseline) data.

    Parameters
    ----------
    domain : str
        Domain name (e.g. ``"companies"``).
    stages : list of str, optional
        Stages to measure. Default: all three (``sm``, ``em``,
        ``fusion``).
    with_llm : bool
        Include LLM committee members. Default ``False``.
    out_dir : Path, optional
        Override output directory. Default:
        ``usecases_synthetic/baselines/<domain>/``.
    fusion_input_member : str, optional
        Override auto-selection of EM member for fusion input
        recording. Default: highest-F1 EM member.

    Returns
    -------
    dict[str, Any]
        The full payload written to ``baseline_metrics.json``.
    """
    stages = stages or list(ALL_STAGES)
    out_dir = out_dir or (BASELINES_DIR / domain)

    logger.info("Loading baseline bundle for domain=%s", domain)
    bundle = load_variant(domain, level="baseline")

    per_stage: dict[str, dict[str, Any]] = {}
    total_t0 = time.monotonic()
    em_result: CommitteeResult | None = None

    # --- SM ---
    if "sm" in stages:
        logger.info("Running SM committee...")
        sm_runner = SMCommitteeRunner(
            resolve_committee_path("sm_committee", domain, committee_dir=COMMITTEE_DIR),
            with_llm=with_llm,
        )
        sm_result = sm_runner.run(bundle)
        per_stage["sm"] = sm_result.as_dict()
        logger.info(
            "SM done: macro_f1=%.4f best=%s (f1=%.4f) (%.1fs)",
            sm_result.aggregated.get("macro_f1", 0.0),
            sm_result.aggregated.get("best_member_name", "?"),
            sm_result.aggregated.get("best_member_f1", 0.0),
            sm_result.runtime_s,
        )

    # --- Normalization ---
    if "norm" in stages:
        logger.info("Running Normalization committee...")
        norm_runner = NormCommitteeRunner(
            resolve_committee_path(
                "normalization_committee", domain, committee_dir=COMMITTEE_DIR
            ),
            with_llm=with_llm,
        )
        norm_result = norm_runner.run(bundle)
        per_stage["norm"] = norm_result.as_dict()
        logger.info(
            "Norm done: macro_f1=%.4f best_member_f1=%.4f (%.1fs)",
            norm_result.aggregated.get("macro_f1", 0.0),
            norm_result.aggregated.get("best_member_f1", 0.0),
            norm_result.runtime_s,
        )

    # --- EM ---
    # EM stage split (2026-05-13, perfect-prior-step design):
    # - ``em_blocking`` measures blockers on full sources (recall +
    #   reduction_ratio) — no matching runs.
    # - ``em_matching`` measures matchers on labelled (id1, id2) pairs
    #   from the EM CSVs under the closed-set semantic — no blocking.
    # Backward compat: ``em`` legacy stage expands to the two new ones.
    if "em" in stages:
        if "em_blocking" not in stages:
            stages = list(stages) + ["em_blocking"]
        if "em_matching" not in stages:
            stages = list(stages) + ["em_matching"]
    if "em_blocking" in stages:
        logger.info("Running EM blocking committee...")
        em_blocking_runner = EMBlockingCommitteeRunner(
            resolve_committee_path(
                "em_blocking_committee", domain, committee_dir=COMMITTEE_DIR
            ),
        )
        em_blocking_result = em_blocking_runner.run(bundle)
        per_stage["em_blocking"] = em_blocking_result.as_dict()
        logger.info(
            "EM blocking done: macro_pair_recall=%.4f best_member=%s "
            "(recall=%.4f, rr=%.4f) (%.1fs)",
            em_blocking_result.aggregated.get("macro_pair_recall", 0.0),
            em_blocking_result.aggregated.get("best_member_name", "?"),
            em_blocking_result.aggregated.get("best_member_pair_recall", 0.0),
            em_blocking_result.aggregated.get("best_member_reduction_ratio", 0.0),
            em_blocking_result.runtime_s,
        )

    if "em_matching" in stages:
        logger.info("Running EM matching committee...")
        em_matching_runner = EMMatchingCommitteeRunner(
            resolve_committee_path(
                "em_matching_committee", domain, committee_dir=COMMITTEE_DIR
            ),
            with_llm=with_llm,
        )
        em_matching_result = em_matching_runner.run(bundle)
        per_stage["em_matching"] = em_matching_result.as_dict()
        em_result = em_matching_result
        logger.info(
            "EM matching done: macro_f1=%.4f best=%s (f1=%.4f) (%.1fs)",
            em_matching_result.aggregated.get("macro_f1", 0.0),
            em_matching_result.aggregated.get("best_member_name", "?"),
            em_matching_result.aggregated.get("best_member_f1", 0.0),
            em_matching_result.runtime_s,
        )

    # --- Fusion ---
    # Per R5 Fusion design (plans/plan_s1_scale.md, 2026-05-12): each
    # committee is evaluated against the **perfect** output of the prior
    # pipeline step, isolating its own signal. For fusion that means
    # assuming the EM step produced the ground-truth clusters declared in
    # the fusion validation + test XMLs. Perfect-cluster correspondences
    # are derived from provenance attributes (companies + music) or
    # EM-gold-positive partners (games, which ships fusion XML without
    # provenance).
    if "fusion" in stages:
        logger.info("Running Fusion committee...")
        fusion_runner = FusionCommitteeRunner(
            resolve_committee_path(
                "fusion_committee", domain, committee_dir=COMMITTEE_DIR
            ),
        )
        perfect_correspondences = build_perfect_clusters_correspondences(domain, bundle)
        logger.info(
            "Fusion using perfect-cluster correspondences: n=%d",
            len(perfect_correspondences),
        )
        fusion_result = fusion_runner.run(
            bundle, correspondences=perfect_correspondences
        )
        per_stage["fusion"] = fusion_result.as_dict()
        logger.info(
            "Fusion done: overall_accuracy=%.4f (%.1fs)",
            fusion_result.aggregated.get("overall_accuracy", 0.0),
            fusion_result.runtime_s,
        )

    total_runtime = time.monotonic() - total_t0

    # Determine fusion_input_member.
    selected_fusion_input: str = fusion_input_member or ""
    if not selected_fusion_input and em_result is not None:
        selected_fusion_input = _best_em_member(em_result)

    # Build metadata.
    meta: dict[str, Any] = {
        "with_llm": with_llm,
        "committee_versions": _committee_versions(stages, domain),
        "total_runtime_s": round(total_runtime, 2),
        "fusion_input_member": selected_fusion_input,
    }

    # Merge with any prior baseline_metrics.json. Partial-stage runs
    # (e.g. ``--stages fusion``) should preserve other stages already
    # measured — replacing the whole file would discard their data, so
    # we keep prior per_stage entries that aren't being re-measured this
    # run. committee_versions is merged the same way so the SHA digest
    # for un-rerun stages is preserved.
    json_path = out_dir / "baseline_metrics.json"
    merged_per_stage: dict[str, Any] = {}
    merged_versions: dict[str, str] = {}
    if json_path.exists():
        try:
            with open(json_path, encoding="utf-8") as f:
                prior = json.load(f)
            merged_per_stage = dict(prior.get("per_stage") or {})
            prior_meta = prior.get("meta") or {}
            merged_versions = dict(prior_meta.get("committee_versions") or {})
        except (OSError, json.JSONDecodeError):
            logger.warning(
                "Could not read existing %s for merge; overwriting.", json_path
            )
            merged_per_stage = {}
            merged_versions = {}
    merged_per_stage.update(per_stage)
    merged_versions.update(meta["committee_versions"])
    meta["committee_versions"] = merged_versions

    # Write JSON.
    write_metrics_json(json_path, domain=domain, per_stage=merged_per_stage, meta=meta)
    logger.info("Wrote %s", json_path)

    # Write markdown report — uses the full merged per_stage so the
    # report reflects everything measured to date, not just this run's
    # stages.
    md_path = out_dir / "baseline_report.md"
    write_report_md(
        md_path,
        domain=domain,
        per_stage=merged_per_stage,
        title=f"Baseline report - {domain}",
    )
    logger.info("Wrote %s", md_path)

    # Verify round-trip.
    loaded = load_baseline(domain, path_override=json_path)
    logger.info(
        "Round-trip check OK: loaded %d stages for %s",
        len(loaded.per_stage),
        loaded.domain,
    )

    return {
        "domain": domain,
        "meta": meta,
        "per_stage": per_stage,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_stages(raw: str) -> list[Stage]:
    """Parse a comma-separated stages string.

    Parameters
    ----------
    raw : str
        Comma-separated stage names (e.g. ``"sm,em"``).

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
        description="Measure baseline committee metrics on original data.",
    )
    parser.add_argument(
        "--domain",
        required=True,
        help="Domain name (e.g. 'companies').",
    )
    parser.add_argument(
        "--with-llm",
        action="store_true",
        default=False,
        help="Include LLM committee members (adds cost and latency).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Override output directory. "
            "Default: usecases_synthetic/baselines/<domain>/."
        ),
    )
    parser.add_argument(
        "--stages",
        type=str,
        default=None,
        help=(
            "Comma-separated stages to run (e.g. 'sm,em'). "
            "Default: all (sm, em, fusion)."
        ),
    )
    parser.add_argument(
        "--fusion-input-member",
        type=str,
        default=None,
        help=(
            "Override auto-selection of EM member recorded as fusion "
            "input. Default: highest-F1 EM member."
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

    measure_baseline(
        domain=args.domain,
        stages=stages,
        with_llm=args.with_llm,
        out_dir=args.out_dir,
        fusion_input_member=args.fusion_input_member,
    )


if __name__ == "__main__":
    main()
