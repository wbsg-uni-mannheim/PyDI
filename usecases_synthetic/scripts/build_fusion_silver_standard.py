#!/usr/bin/env python3
"""Build the per-domain fusion silver standard (plan_revision.md C9 / step 4b).

Runs the human-baseline notebook's fusion stack against every cluster
in the pool, persisting the per-cluster fused value per attribute as
the protection target for variant generation.

Usage
-----
::

    python usecases_synthetic/scripts/build_fusion_silver_standard.py --domain music
    python usecases_synthetic/scripts/build_fusion_silver_standard.py --domain music --sample 10

Outputs
-------
- ``usecases_synthetic/baselines/<domain>/fusion_silver_standard.csv``
- ``usecases_synthetic/baselines/<domain>/fusion_silver_standard.json``
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from usecases_synthetic.lib.fusion_silver_standard import (  # noqa: E402
    build_silver_standard,
    silver_path,
    supported_domains,
    write_silver_standard,
)

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="[%(levelname)-5s] %(name)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--domain",
        required=True,
        help=f"Domain to build the silver standard for. Supported: {supported_domains()}",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help=(
            "Optional: emit only the first N clusters as a quick spot-check "
            "(written to a sibling .sample.csv/.sample.json so it does not "
            "clobber the full artifact). When omitted, the full silver is "
            "built and persisted."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Build the silver standard but do not persist artifacts.",
    )
    args = parser.parse_args(argv)

    _setup_logging(args.verbose)

    if args.domain not in supported_domains():
        parser.error(
            f"Domain {args.domain!r} is not yet wired. "
            f"Supported: {supported_domains()}."
        )

    logger.info("Building fusion silver standard for domain=%s", args.domain)
    silver = build_silver_standard(args.domain)
    logger.info(
        "Silver standard rows=%d unique_clusters=%d",
        len(silver),
        silver["cluster_id"].nunique() if not silver.empty else 0,
    )

    if args.sample is not None and not silver.empty:
        sample_ids = silver["cluster_id"].drop_duplicates().head(args.sample).tolist()
        silver = silver[silver["cluster_id"].isin(sample_ids)].reset_index(drop=True)
        logger.info("Sampled %d clusters (%d rows)", len(sample_ids), len(silver))

    if args.no_write:
        logger.info("--no-write: skipping persistence")
        return 0

    if args.sample is not None:
        # Write to *.sample.* so spot-check runs don't clobber the canonical.
        base_csv = silver_path(args.domain, "csv")
        base_json = silver_path(args.domain, "json")
        sample_csv = base_csv.with_name("fusion_silver_standard.sample.csv")
        sample_json = base_json.with_name("fusion_silver_standard.sample.json")
        sample_csv.parent.mkdir(parents=True, exist_ok=True)
        # Use the same writer logic by passing an explicit out_dir + rename.
        paths = write_silver_standard(args.domain, silver)
        # write_silver_standard always writes to the canonical name; move
        # the result to the .sample.* path so the canonical isn't created.
        paths["csv"].rename(sample_csv)
        paths["json"].rename(sample_json)
        logger.info("Wrote sample silver to %s + %s", sample_csv, sample_json)
    else:
        paths = write_silver_standard(args.domain, silver)
        logger.info("Wrote silver to %s + %s", paths["csv"], paths["json"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
