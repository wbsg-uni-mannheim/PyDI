#!/usr/bin/env python3
"""Programmatically execute the human-baseline notebook and cache its
final fused output for the panel comparison.

The human-baseline notebook (e.g.
``usecases/products/products_workflow_minimal.ipynb``) doesn't persist
its final ``fused`` + ``all_correspondences`` DataFrames to disk.
This script:

1. Copies the notebook to a temp location.
2. Appends an export cell that writes ``fused.to_csv(...)`` +
   ``all_correspondences.to_csv(...)`` to the target cache dir.
3. Executes the copy via ``jupyter nbconvert --execute``.
4. Verifies the two CSVs landed.

Cost: re-executing the notebook is slow (LLM-backed SM + EM matchers
+ fusion) and consumes API tokens. One-off — re-run only when the
notebook itself changes.

Usage
-----
::

    python pipelines/scripts/extract_notebook_baseline.py \\
        --notebook usecases/products/products_workflow_minimal.ipynb \\
        --cache-dir pipelines/products/baselines
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger(__name__)


_EXPORT_CELL_TEMPLATE = """
# ---------------------------------------------------------------------
# Auto-injected by pipelines/scripts/extract_notebook_baseline.py.
# Persists the final fused output + all_correspondences for the
# best-of-breed panel comparison.
# ---------------------------------------------------------------------
import json as _json
from pathlib import Path as _Path
_cache = _Path({cache_dir!r})
_cache.mkdir(parents=True, exist_ok=True)

# fused: the final DataFusionEngine output (set in Part 3 of the notebook).
try:
    fused.to_csv(_cache / 'notebook_fused.csv', index=False)
    _meta = {{'n_rows': len(fused), 'columns': list(fused.columns)}}
except Exception as _e:
    _meta = {{'error': str(_e)}}

# all_correspondences: the concatenated refined-pair links (set in Part 2).
try:
    all_correspondences.to_csv(_cache / 'notebook_correspondences.csv', index=False)
    _meta['n_correspondences'] = len(all_correspondences)
except Exception as _e:
    _meta.setdefault('error', str(_e))

with open(_cache / 'notebook_export_meta.json', 'w') as _f:
    _json.dump(_meta, _f, indent=2, default=str)

print('Exported notebook baseline:', _meta)
"""


def _append_export_cell(notebook_path: Path, *, cache_dir: Path) -> Path:
    """Copy the notebook to /tmp and append the export cell."""
    nb_dict = json.loads(notebook_path.read_text())
    cells = nb_dict.get("cells") or []
    cells.append(
        {
            "cell_type": "code",
            "metadata": {},
            "outputs": [],
            "source": _EXPORT_CELL_TEMPLATE.format(cache_dir=str(cache_dir)),
            "execution_count": None,
        }
    )
    nb_dict["cells"] = cells

    target = Path("/tmp") / f"_bob_export_{notebook_path.name}"
    target.write_text(json.dumps(nb_dict, indent=1))
    return target


def _execute_notebook(
    patched_notebook: Path,
    *,
    cwd: Path,
    timeout_s: int = 3600,
) -> None:
    """Run ``jupyter nbconvert --execute`` on the patched notebook."""
    cmd = [
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        str(patched_notebook),
        "--output",
        str(patched_notebook.with_suffix(".executed.ipynb").name),
        "--output-dir",
        str(patched_notebook.parent),
        "--ExecutePreprocessor.timeout=" + str(timeout_s),
    ]
    logger.info("Running: %s (cwd=%s)", " ".join(cmd), cwd)
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error("nbconvert stdout:\n%s", result.stdout[-4000:])
        logger.error("nbconvert stderr:\n%s", result.stderr[-4000:])
        raise RuntimeError(f"nbconvert failed with exit {result.returncode}")
    logger.info("nbconvert finished cleanly.")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--notebook", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--timeout-s", type=int, default=3600)
    p.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(levelname)s] %(name)s - %(message)s",
    )

    if not args.notebook.exists():
        print(f"ERROR: notebook not found: {args.notebook}", file=sys.stderr)
        return 2

    cache_dir = args.cache_dir.resolve()
    patched = _append_export_cell(args.notebook, cache_dir=cache_dir)

    # The notebook uses relative paths anchored at its own directory
    # (e.g. ``Path('.').resolve() / 'input' / ...``). Run nbconvert from
    # the notebook's parent dir so those resolve as the author expects.
    notebook_parent = args.notebook.parent.resolve()

    # Move the patched copy into the same parent so its relative paths
    # resolve identically.
    target = notebook_parent / patched.name
    shutil.copy2(patched, target)

    try:
        _execute_notebook(target, cwd=notebook_parent, timeout_s=args.timeout_s)
    finally:
        target.unlink(missing_ok=True)
        target.with_suffix(".executed.ipynb").unlink(missing_ok=True)

    expected = [
        cache_dir / "notebook_fused.csv",
        cache_dir / "notebook_correspondences.csv",
    ]
    missing = [p for p in expected if not p.exists()]
    if missing:
        print(f"ERROR: expected files missing: {missing}", file=sys.stderr)
        return 3

    meta = json.loads((cache_dir / "notebook_export_meta.json").read_text())
    print("\nNotebook baseline cached:")
    for k, v in meta.items():
        print(f"  - {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
