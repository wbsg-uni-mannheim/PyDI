#!/usr/bin/env python3
"""Execute a human-baseline workflow notebook headless and cache its fused output.

``compare_to_human_baseline.py``'s SR panel needs the notebook's ``fused`` +
``all_correspondences`` DataFrames persisted to
``pipelines/<domain>/baselines/notebook_{fused,correspondences}.csv`` — but the
script's ``cache_notebook_output`` is a documented manual step (raises
NotImplementedError). This driver automates it: it appends a save-cell to a
copy of the notebook and runs the whole thing with nbclient.

The workflow notebooks resolve their inputs via ``NOTEBOOK_DIR = Path(".")``,
so the kernel cwd MUST be the notebook's own directory (``usecases/<domain>/``);
the save-cell writes to an absolute cache path so it lands in ``pipelines/``.

Usage::

    python pipelines/scripts/cache_notebook.py \\
        --notebook usecases/music/music_workflow.ipynb \\
        --cache-dir pipelines/music/baselines
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--notebook", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument(
        "--kernel",
        default="pydi-dev",
        help="ipykernel name backing the pydi-dev venv (register with "
        "`python -m ipykernel install --user --name pydi-dev`).",
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=10800,
        help="Per-cell execution timeout in seconds (default 3h).",
    )
    args = ap.parse_args()

    cache = args.cache_dir.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    fused_csv = cache / "notebook_fused.csv"
    corr_csv = cache / "notebook_correspondences.csv"

    nb = nbformat.read(str(args.notebook), as_version=4)
    save_src = (
        "# --- injected by cache_notebook.py: persist fused output for SR ---\n"
        f"fused.to_csv(r'{fused_csv}', index=False)\n"
        f"all_correspondences.to_csv(r'{corr_csv}', index=False)\n"
        f"print('CACHED', len(fused), 'fused rows ->', r'{fused_csv}')\n"
    )
    nb.cells.append(nbformat.v4.new_code_cell(save_src))

    run_path = str(args.notebook.resolve().parent)
    print(f"Executing {args.notebook} (cwd={run_path}, kernel={args.kernel}) ...")
    client = NotebookClient(
        nb,
        timeout=args.timeout,
        kernel_name=args.kernel,
        resources={"metadata": {"path": run_path}},
    )
    client.execute()

    if not fused_csv.exists():
        print(
            f"ERROR: {fused_csv} not written — notebook may not define `fused`.",
            file=sys.stderr,
        )
        return 1
    print(f"OK: cached {fused_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
