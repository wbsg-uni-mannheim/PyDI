#!/usr/bin/env python3
"""Recompute per-pair EM matching F1 for a baseline best-of-breed run.

Re-runs ONLY the EM stage (blocking + matching committees) using the
*effective* committee YAMLs the original run persisted (which already
carry the pipeline-isolated ditto / sc_block checkpoint paths), then
dumps the winning matcher's per-pair test F1.

This faithfully reproduces the pipeline's EM stage without re-running
SM / norm / fusion, because for the baseline the EM committee operates
on the *raw* bundle sources:

  - post-SM source translation is a no-op (pipeline.py:310-325 — the
    committees translate via their own YAML ``column_mapping``), and
  - norm runs with ``apply_winner: false`` in every domain config, so
    it scores but never mutates the sources handed to EM.

The EM matchers are deterministic (ditto eval-mode inference; magellan
RandomForest ``random_state=42``), so the recomputed *average* F1 must
match the persisted ``stage_4_em_matching_selection.json`` ``test_score``
— a built-in sanity check printed at the end.

Usage
-----
::

    python pipelines/scripts/recompute_em_per_pair.py \
        --domain papers \
        --run-dir pipelines/papers/run_slurm_baseline_256772/
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

import yaml

from pipelines.lib.bundle import PipelineState, load_pipeline_bundle
from pipelines.lib.stage_runners import run_em

logging.basicConfig(
    level=logging.INFO, format="[%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("recompute_em_per_pair")


def _one(eff_dir: Path, prefix: str) -> Path:
    hits = sorted(eff_dir.glob(f"{prefix}*.yaml"))
    if not hits:
        raise FileNotFoundError(f"no {prefix}*.yaml under {eff_dir}")
    if len(hits) > 1:
        raise RuntimeError(f"ambiguous {prefix}*.yaml under {eff_dir}: {hits}")
    return hits[0]


def main() -> int:
    p = argparse.ArgumentParser(
        description="Recompute the winning matcher's per-pair EM test F1."
    )
    p.add_argument("--domain", required=True)
    p.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Baseline run dir holding effective_committees/.",
    )
    p.add_argument(
        "--bundle-source",
        default=None,
        help="Override; else read from pipelines/configs/<domain>.yaml.",
    )
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    cfg = yaml.safe_load(
        (REPO_ROOT / "pipelines" / "configs" / f"{args.domain}.yaml").read_text()
    )
    bundle_source = args.bundle_source or cfg.get(
        "bundle_source", "synthetic_baseline"
    )
    clustering = str(
        (cfg.get("stages", {}).get("em", {}) or {}).get("clustering", "greedy")
    )

    eff = args.run_dir / "effective_committees"
    blocking_yaml = _one(eff, "em_blocking_committee")
    matching_yaml = _one(eff, "em_matching_committee")

    logger.info(
        "domain=%s bundle_source=%s clustering=%s", args.domain, bundle_source, clustering
    )
    logger.info("blocking_yaml=%s", blocking_yaml)
    logger.info("matching_yaml=%s", matching_yaml)

    bundle = load_pipeline_bundle(
        args.domain, level="baseline", bundle_source=bundle_source
    )
    state = PipelineState(bundle=bundle)

    _blocking_sel, matching_sel = run_em(
        state,
        blocking_yaml=blocking_yaml,
        matching_yaml=matching_yaml,
        with_llm=False,
        clustering=clustering,
    )

    winner = matching_sel.winner
    per_pair = (matching_sel.notes or {}).get("per_pair_test_f1", {})

    # Sanity check: recomputed average must match the persisted test_score.
    persisted = None
    stage4 = args.run_dir / "stage_4_em_matching_selection.json"
    if stage4.exists():
        persisted = json.loads(stage4.read_text()).get("test_score")

    out = {
        "domain": args.domain,
        "winner": winner,
        "winner_per_pair_test_f1": per_pair.get(winner, {}),
        "all_members_per_pair_test_f1": per_pair,
        "per_member_test_f1_avg": matching_sel.per_member_test,
        "recomputed_test_score_avg": matching_sel.test_score,
        "persisted_test_score_avg": persisted,
        "matches_persisted": (
            persisted is not None
            and abs(float(matching_sel.test_score) - float(persisted)) < 1e-6
        ),
    }

    out_path = args.out or (args.run_dir / "em_per_pair_test_f1.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    logger.info("wrote %s", out_path)
    if not out["matches_persisted"]:
        logger.warning(
            "Recomputed avg (%s) != persisted test_score (%s) — investigate "
            "before trusting the per-pair split.",
            matching_sel.test_score,
            persisted,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
