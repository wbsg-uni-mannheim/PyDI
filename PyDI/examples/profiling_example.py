#!/usr/bin/env python3
"""
PyDI Profiling Example
======================

Load a dataset using PyDI's provenance-aware I/O, then generate a profiling
report using ydata-profiling. Artifacts are written to output/profiling.

Usage:
    python -m PyDI.examples.profiling_example
"""

import logging
from pathlib import Path

from PyDI.io import load_csv
from PyDI.profiling import DataProfiler


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    # Locate example CSV
    repo_root = Path(__file__).parent.parent.parent
    csv_path = repo_root / "input/schemamatching/data/movie_list.csv"
    out_dir = repo_root / "output/profiling"

    if not csv_path.exists():
        print(f"CSV file not found: {csv_path}")
        return

    # Load with provenance-aware loader
    df = load_csv(csv_path, name="movies")
    print(f"Loaded {len(df)} rows from {csv_path.name}")
    print(f"Columns: {list(df.columns)[:8]} ...")

    # Profile
    profiler = DataProfiler()
    try:
        report_path = profiler.profile(df, str(out_dir))
    except ImportError as e:
        logger.error(
            "Profiling requires optional dependencies. Please install: 'ydata-profiling' (and optionally 'sweetviz'). Error: %s",
            e,
        )
        return

    print("\nProfiling report written to:")
    print(report_path)


if __name__ == "__main__":
    main()
