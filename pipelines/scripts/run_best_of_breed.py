#!/usr/bin/env python3
"""Best-of-breed pipeline CLI.

Usage
-----
::

    python pipelines/scripts/run_best_of_breed.py \\
        --config pipelines/configs/products.yaml \\
        --out pipelines/products/run_<timestamp>/

Outputs (under ``--out``):

- ``stage_<n>_<stage>_selection.json`` — per-stage winner + scores
- ``per_stage_summary.csv`` — one row per stage
- ``fused.csv`` — final fused output
- ``correspondences.csv`` — post-refinement correspondences fed to fusion
- ``e2e_panel/`` — six metric panel artifacts
- ``summary.md`` — human-readable run summary
"""

from __future__ import annotations

# faiss-cpu's libomp collides with torch's libomp on macOS arm64 — same
# guard ``usecases_synthetic/scripts/measure_baseline.py`` uses.
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

from pipelines.lib.pipeline import BestOfBreedPipeline, PipelineConfig
from pipelines.lib.report import write_run_artifacts


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the best-of-breed data integration pipeline."
    )
    p.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Pipeline config YAML (e.g. pipelines/configs/products.yaml).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output directory. Defaults to " "pipelines/<domain>/run_<UTC timestamp>/."
        ),
    )
    p.add_argument(
        "--committee-dir",
        type=Path,
        default=REPO_ROOT / "usecases_synthetic" / "config" / "committees",
        help="Directory holding committee YAMLs.",
    )
    p.add_argument(
        "--mode",
        choices=["sweep", "replay"],
        default="replay",
        help=(
            "sweep: run chained HP sweeps at each stage where implemented; "
            "replay: use YAML-default committee runners (faster, "
            "deterministic). Default replay."
        ),
    )
    p.add_argument(
        "--llm-sm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Toggle LLM-backed schema-matching members (llm_matcher). "
            "Default: ON. Pass --no-llm-sm to disable."
        ),
    )
    p.add_argument(
        "--llm-em",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Toggle LLM-backed EM-matching members (llm_matcher, matchgpt, "
            "comem). Default: OFF — per the 2026-06-01 directive to drop "
            "LLM from entity matching. Pass --llm-em to enable."
        ),
    )
    p.add_argument(
        "--llm-fusion",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Toggle LLM-backed fusion members (llm_only, llm_judge). "
            "Default: OFF — per the 2026-06-01 directive to drop LLM "
            "from fusion. Pass --llm-fusion to enable."
        ),
    )
    p.add_argument(
        "--no-llm",
        action="store_true",
        help=(
            "Legacy global kill-switch: when set, forces --no-llm-sm AND "
            "--no-llm-em AND --no-llm-fusion regardless of their per-stage "
            "values. Use the per-stage flags above for fine-grained control."
        ),
    )
    p.add_argument(
        "--ditto-checkpoint-override",
        type=Path,
        default=None,
        help=(
            "Path to a pipeline-isolated Ditto checkpoint dir (must contain "
            "model_config.json + model.pt). The committee YAML's default "
            "checkpoint_path (under usecases_synthetic/cache/) is NEVER read; "
            "if this flag is omitted and the YAML default lives in "
            "usecases_synthetic/cache/, ditto_plm is LOUDLY disabled for the "
            "run with retrain instructions."
        ),
    )
    p.add_argument(
        "--sc-block-checkpoint-override",
        type=Path,
        default=None,
        help=(
            "Path to a pipeline-isolated sc_block checkpoint dir (must contain "
            "config.json + model.safetensors). Same no-reuse semantics as "
            "--ditto-checkpoint-override."
        ),
    )
    p.add_argument(
        "--fusion-members",
        default=None,
        help=(
            "Comma-separated allowlist of C12 fusion roster member names "
            "(e.g. 'voting_only,prefer_higher_trust_only,pydi_per_attribute_optimal'). "
            "Members not in the list are dropped from the rewritten fusion "
            "YAML. Useful to skip members with hidden network dependencies."
        ),
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(levelname)s] %(name)s - %(message)s",
    )

    config = PipelineConfig.from_yaml(args.config)
    out_dir = args.out or (
        REPO_ROOT
        / "pipelines"
        / config.domain
        / f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    )

    # Per-stage LLM toggles. The legacy --no-llm flag forces all three off.
    with_llm_sm = args.llm_sm and not args.no_llm
    with_llm_em = args.llm_em and not args.no_llm
    with_llm_fusion = args.llm_fusion and not args.no_llm
    logging.info(
        "Running best-of-breed pipeline: domain=%s mode=%s "
        "llm_sm=%s llm_em=%s llm_fusion=%s "
        "ditto_override=%s sc_block_override=%s out=%s",
        config.domain,
        args.mode,
        with_llm_sm,
        with_llm_em,
        with_llm_fusion,
        args.ditto_checkpoint_override,
        args.sc_block_checkpoint_override,
        out_dir,
    )
    fusion_members = None
    if args.fusion_members:
        fusion_members = {
            name.strip() for name in args.fusion_members.split(",") if name.strip()
        }
    pipeline = BestOfBreedPipeline(
        config,
        committee_dir=args.committee_dir,
        with_llm_sm=with_llm_sm,
        with_llm_em=with_llm_em,
        with_llm_fusion=with_llm_fusion,
        mode=args.mode,
        ditto_checkpoint_override=args.ditto_checkpoint_override,
        sc_block_checkpoint_override=args.sc_block_checkpoint_override,
        out_dir=out_dir,
        fusion_members=fusion_members,
    )
    result = pipeline.run()

    write_run_artifacts(result, out_dir=out_dir, config=config)
    print(f"\nRun complete: {out_dir}")
    print(f"  - {len(result.stage_selections)} stages")
    print(f"  - total runtime: {result.total_runtime_s:.1f}s")
    if result.panel is not None:
        composite = result.panel.composite.get("composite_score")
        print(
            f"  - composite_score: {composite:.4f}"
            if composite is not None
            else "  - composite_score: N/A"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
