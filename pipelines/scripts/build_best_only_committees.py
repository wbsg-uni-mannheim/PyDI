#!/usr/bin/env python3
"""Build 'best-only' committee YAMLs + a config for timing the final best
pipeline (Option B).

We already know the winning member per stage from a prior full-committee run.
This reads those winners from ``<ref_run>/per_stage_summary.csv``, copies the
committee dir, and filters each stage's roster (``members:``) down to the single
winning member, so the best-of-breed pipeline run against this dir executes
ONLY the best members. Refinement is config-driven (not a committee YAML), so we
also emit a config copy with ``stages.refinement.methods`` restricted to the
winning refiner. Run the timed pipeline with::

    python pipelines/scripts/run_best_of_breed.py \
        --config <out_config> --committee-dir <out_committee_dir> \
        --out <besttime_run_dir> --variant baseline ...
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from usecases_synthetic.lib.committee_paths import resolve_committee_path

# stage -> committee base filename (refinement is config-driven, handled below)
STAGE_BASE = {
    "sm": "sm_committee",
    "norm": "normalization_committee",
    "em_blocking": "em_blocking_committee",
    "em_matching": "em_matching_committee",
    "fusion": "fusion_committee",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--ref-run", required=True, help="run dir with per_stage_summary.csv")
    ap.add_argument("--out-committee-dir", required=True)
    ap.add_argument("--out-config", required=True)
    a = ap.parse_args()

    winners: dict[str, str] = {}
    with open(Path(a.ref_run) / "per_stage_summary.csv") as f:
        for r in csv.DictReader(f):
            winners[r["stage"]] = r["winner"]
    print(f"[best-only] winners for {a.domain}: {winners}")

    src_dir = REPO_ROOT / "usecases_synthetic" / "config" / "committees"
    out_dir = Path(a.out_committee_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(src_dir, out_dir)

    for stage, base in STAGE_BASE.items():
        winner = winners.get(stage)
        if not winner:
            print(f"[best-only] {stage}: no winner recorded; leaving full roster")
            continue
        path = resolve_committee_path(base, a.domain, committee_dir=out_dir)
        if not path.exists():
            print(f"[best-only] {stage}: committee file missing: {path}")
            continue
        doc = yaml.safe_load(path.read_text())
        members = doc.get("members") or []
        kept = [m for m in members if str(m.get("name")) == winner]
        if not kept:
            print(
                f"[best-only] WARN {stage}: winner {winner!r} not among "
                f"{[m.get('name') for m in members]} in {path.name}; leaving roster"
            )
            continue
        doc["members"] = kept
        path.write_text(yaml.safe_dump(doc, sort_keys=False))
        print(f"[best-only] {stage}: {path.name} -> [{winner}]")

    # Refinement: restrict the config's method list to the winner (baseline).
    refine_winner = winners.get("refinement", "baseline")
    cfg_path = REPO_ROOT / "pipelines" / "configs" / f"{a.domain}.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg.setdefault("stages", {}).setdefault("refinement", {})["methods"] = [refine_winner]
    out_cfg = Path(a.out_config)
    out_cfg.parent.mkdir(parents=True, exist_ok=True)
    out_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"[best-only] refinement methods -> [{refine_winner}]; config -> {out_cfg}")
    print("[best-only] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
