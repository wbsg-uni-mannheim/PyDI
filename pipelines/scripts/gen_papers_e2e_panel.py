#!/usr/bin/env python3
"""Generate the full RF+SR+GR e2e panel for papers BASE.

Papers was skipped by the in-pipeline panel (no membership/provenance silver).
This standalone driver:
  1. Caches the human-baseline notebook output (SR silver) by executing
     papers_workflow_minimal.ipynb headless with an appended save cell, if
     pipelines/papers/baselines/notebook_fused.csv is missing.
  2. Builds the GR gold silver from the DOI-keyed fusion gold
     (canonical_loader.load_canonical_papers_workflow_silver).
  3. Computes the panel for an existing papers BASE run (fused.csv +
     correspondences.csv) with BOTH silver and gold -> RF, SR, GR tiers in one
     composite, written to <run_dir>/e2e_panel_fixed/ (matching the dir the
     DI_Bench tab:e2e-detailed table reads).

Run via sbatch (compute: the notebook executes the human-baseline SM/EM/fusion).
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import logging
import subprocess
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

import pandas as pd

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("gen_papers_e2e_panel")


def _cache_notebook_baseline(cache_dir: Path) -> None:
    """Execute papers_workflow_minimal.ipynb headless, appending a cell that
    saves fused_v5 + all_correspondences to the cache dir."""
    fused_csv = cache_dir / "notebook_fused.csv"
    if fused_csv.exists():
        logger.info("Notebook baseline cache present (%s); skipping execution.", fused_csv)
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    nb_path = REPO_ROOT / "usecases" / "papers" / "papers_workflow_minimal.ipynb"
    nb = json.loads(nb_path.read_text())
    save_src = (
        "from pathlib import Path as _P\n"
        f"_cache = _P(r'{cache_dir}')\n"
        "_cache.mkdir(parents=True, exist_ok=True)\n"
        "fused_v5.to_csv(_cache / 'notebook_fused.csv', index=False)\n"
        "try:\n"
        "    all_correspondences.to_csv(_cache / 'notebook_correspondences.csv', index=False)\n"
        "except NameError:\n"
        "    pass\n"
        "print('SAVED notebook baseline to', _cache)\n"
    )
    import nbformat
    from nbclient import NotebookClient

    nbf = nbformat.from_dict(nb)
    # nbclient needs cell.source as a str; JSON notebooks store it as a list.
    for _c in nbf.cells:
        if isinstance(_c.get("source"), list):
            _c["source"] = "".join(_c["source"])
    nbf.cells.append(nbformat.v4.new_code_cell(save_src))
    logger.info("Executing notebook headless via nbclient (cwd=%s)", REPO_ROOT)
    client = NotebookClient(
        nbf,
        timeout=7200,
        kernel_name="python3",
        resources={"metadata": {"path": str(REPO_ROOT)}},
    )
    client.execute()
    if not fused_csv.exists():
        raise RuntimeError(f"Notebook ran but {fused_csv} was not produced.")
    logger.info("Cached notebook baseline -> %s", fused_csv)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="papers BASE run dir (has fused.csv)")
    ap.add_argument("--config", default="pipelines/configs/papers.yaml")
    args = ap.parse_args()

    from pipelines.lib.pipeline import PipelineConfig
    from pipelines.lib.canonical_loader import (
        load_canonical_papers_bundle,
        load_canonical_papers_workflow_silver,
    )
    from pipelines.scripts.compare_to_human_baseline import (
        build_silver_standard_from_notebook,
    )
    from PyDI.evaluation.panel import compute_e2e_panel

    run_dir = Path(args.run_dir)
    cache_dir = REPO_ROOT / "pipelines" / "papers" / "baselines"

    # 1. SR silver: cache + build from the human-baseline notebook.
    _cache_notebook_baseline(cache_dir)
    config = PipelineConfig.from_yaml(Path(args.config))
    bundle = load_canonical_papers_bundle()
    silver = build_silver_standard_from_notebook(
        cache_dir, domain="papers", config=config, bundle=bundle
    )
    logger.info("SR silver: %d clusters", len(silver.fused))

    # 2. GR gold silver from the DOI-keyed fusion gold.
    gold = load_canonical_papers_workflow_silver("test")
    logger.info("GR gold: %d clusters", len(gold.fused))

    # 3. Pipe output from the existing run.
    fused = pd.read_csv(run_dir / "fused.csv")
    corr_path = run_dir / "correspondences.csv"
    corr = pd.read_csv(corr_path) if corr_path.exists() else None

    # sources_pipe: canonical papers sources renamed via the (identity) SM gold,
    # mirroring pipeline._compute_panel.
    sm_map = bundle.sm_mapping
    sources_pipe = []
    for df in bundle.sources.values():
        name = df.attrs.get("dataset_name")
        rename = {}
        if sm_map is not None and not sm_map.empty:
            for _, r in sm_map.iterrows():
                if str(r.get("source_dataset")) == str(name):
                    sc, tc = r.get("source_column"), r.get("target_column")
                    if sc and tc and sc != tc:
                        rename[str(sc)] = str(tc)
        d2 = df.rename(columns=rename) if rename else df
        d2.attrs["dataset_name"] = name
        sources_pipe.append(d2)

    fused_cols = set(fused.columns)
    col_types = {k: v for k, v in config.column_types.items() if k in fused_cols}

    # pipe_membership from fused _fusion_sources / _fusion_source_datasets.
    import ast

    def _coerce(v):
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            try:
                return ast.literal_eval(v)
            except (ValueError, SyntaxError):
                return [v]
        return [v] if v is not None else []

    rows = []
    for _, fr in fused.iterrows():
        cid = str(fr["_id"])
        srcs = _coerce(fr.get("_fusion_sources"))
        dss = _coerce(fr.get("_fusion_source_datasets"))
        if len(dss) != len(srcs):
            dss = list(dss) + ["unknown"] * (len(srcs) - len(dss))
        for rid, s in zip(srcs, dss):
            rows.append({"record_id": str(rid), "source": str(s), "cluster_id": cid})
    pipe_membership = pd.DataFrame(rows, columns=["record_id", "source", "cluster_id"])

    schema_path = bundle.variant_root / "input" / "schemamatching" / "target_schema.json"

    panel = compute_e2e_panel(
        pipe_fused=fused,
        correspondences_pipe=corr,
        sources_pipe=sources_pipe,
        silver=silver,
        gold=gold,
        column_types=col_types,
        target_schema=schema_path,
        taxonomy_base_path=bundle.variant_root,
        pipe_id_column="_id",
        silver_id_column="cluster_id",
        gold_id_column="cluster_id",
        pipe_membership=pipe_membership,
        numerical_tolerance=config.panel_tolerance_default,
        numerical_tolerance_overrides=config.panel_tolerance_overrides,
        composite_weights=config.composite_weights or None,
        source_prefix_map=config.source_prefix_map or None,
        usecase="papers",
        silver_source_label="papers_workflow_minimal.ipynb",
        gold_source_label="fusion_test.jsonl",
    )

    out = run_dir / "e2e_panel_fixed"
    panel.write(out)
    logger.info("Wrote panel -> %s", out)
    comp = panel.composite
    for tier in ("RF", "SR", "GR"):
        t = comp.get(tier, {})
        cs = t.get("composite_score") if isinstance(t, dict) else None
        print(f"{tier}: composite_score = {cs}")
    print("panel.csv written; subscores in", out / "composite_score.json")


if __name__ == "__main__":
    main()
