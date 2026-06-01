"""Materialize the products domain at the synthetic-side data_root.

The original ``usecases/products/`` notebook workflow uses bare integer
ids (each ``products_<n>.json`` file's native ``id`` field is drawn
from disjoint ranges). The synthetic variant-generation pipeline
requires source-prefixed string ids that are unambiguously cross-source
unique, and a different EM gold directory + file naming convention.

Rather than touch the upstream notebook artefacts at all, this script
reads from ``usecases/products/input/`` and **writes** the
synthetic-side copy at ``usecases_synthetic/usecases/products/input/``
with:

* Source JSON files rewritten with ``id = "products_<n>_<original_int>"``.
* EM gold CSVs renamed from ``prod<N>_to_prod<M>_<split>.csv`` to
  ``products_<N>_2_products_<M>_<split>.csv`` and ids rewritten to the
  same source-prefixed strings.
* The cross-pair ungrouped ``train_gt.csv`` / ``val_gt.csv`` /
  ``test_gt.csv`` are NOT copied — they are notebook artefacts, not
  part of the canonical pipeline EM gold layout.

Idempotent — re-running on already-materialized files leaves them
unchanged. The output paths are derived from the products domain YAML
via :func:`usecases_synthetic.lib.domain_config.data_root_for_domain`,
so they automatically follow the configured ``data_root``.

Run::

    source pydi-dev/bin/activate
    python usecases_synthetic/scripts/_rewrite_products_ids.py
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from usecases_synthetic.lib.domain_config import data_root_for_domain  # noqa: E402

UPSTREAM_INPUT_DIR = REPO_ROOT / "usecases" / "products" / "input"
DOMAIN = "products"

SOURCE_NAMES = ("products_1", "products_2", "products_3", "products_4")
EM_PAIR_STEMS = {
    "prod1_to_prod2": ("products_1", "products_2", "products_1_2_products_2"),
    "prod1_to_prod3": ("products_1", "products_3", "products_1_2_products_3"),
    "prod1_to_prod4": ("products_1", "products_4", "products_1_2_products_4"),
}
PER_PAIR_SPLITS = ("train", "val", "test", "all")

logger = logging.getLogger("rewrite_products_ids")


def _synthetic_input_dir() -> Path:
    """Resolve the synthetic-side input dir from the domain YAML."""
    from usecases_synthetic.lib.domain_config import USECASES_DIR

    root = data_root_for_domain(DOMAIN) or USECASES_DIR
    return root / DOMAIN / "input"


def _prefix(source: str, raw_id: object) -> str:
    """Return ``source_<raw_id>``; passes through already-prefixed ids."""
    text = str(raw_id)
    if text.startswith(f"{source}_"):
        return text
    return f"{source}_{text}"


def _rewrite_source_json(
    source: str,
    upstream_path: Path,
    synthetic_path: Path,
    *,
    dry_run: bool,
) -> int:
    """Read upstream JSON, rewrite ids, write to synthetic-side location.

    Returns the count of records whose id was rewritten (records already
    carrying the source prefix are passed through).
    """
    with upstream_path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"{upstream_path}: top-level JSON must be a list")
    changed = 0
    for record in data:
        if "id" not in record:
            raise KeyError(f"{upstream_path}: record missing 'id': {record}")
        old = record["id"]
        new = _prefix(source, old)
        if str(old) != new:
            record["id"] = new
            changed += 1
    if dry_run:
        logger.info(
            "[dry-run] would write %s (%d ids rewritten)",
            synthetic_path,
            changed,
        )
        return changed
    synthetic_path.parent.mkdir(parents=True, exist_ok=True)
    with synthetic_path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    logger.info("wrote %s (%d ids rewritten)", synthetic_path, changed)
    return changed


def _rewrite_em_gold_csv(
    upstream_path: Path,
    synthetic_path: Path,
    source_left: str,
    source_right: str,
    *,
    dry_run: bool,
) -> int:
    """Read upstream EM gold CSV, rewrite ids, write to synthetic-side.

    Returns count of id cells rewritten.
    """
    df = pd.read_csv(upstream_path, dtype={"id1": str, "id2": str, "label": str})
    if not {"id1", "id2", "label"}.issubset(df.columns):
        raise ValueError(
            f"{upstream_path}: expected id1/id2/label columns; got {list(df.columns)}"
        )
    before = df.copy()
    df["id1"] = df["id1"].map(lambda v: _prefix(source_left, v))
    df["id2"] = df["id2"].map(lambda v: _prefix(source_right, v))
    # The synthetic EM-gold convention matches music/games/companies:
    # headerless CSVs with lowercase boolean labels (``false`` / ``true``).
    # Products ships ``label`` as ``0`` / ``1`` with a header row.
    label_map = {"0": "false", "1": "true", "false": "false", "true": "true"}
    df["label"] = df["label"].str.strip().str.lower().map(label_map)
    if df["label"].isna().any():
        bad = before.loc[df["label"].isna(), "label"].unique().tolist()
        raise ValueError(f"{upstream_path}: unrecognised label values: {bad!r}")
    changed = int(
        (before["id1"] != df["id1"]).sum() + (before["id2"] != df["id2"]).sum()
    )
    if dry_run:
        logger.info(
            "[dry-run] would write %s (%d id cells rewritten)",
            synthetic_path,
            changed,
        )
        return changed
    synthetic_path.parent.mkdir(parents=True, exist_ok=True)
    df[["id1", "id2", "label"]].to_csv(synthetic_path, index=False, header=False)
    logger.info("wrote %s (%d id cells rewritten)", synthetic_path, changed)
    return changed


def _copy_target_schema(synthetic_input: Path, *, dry_run: bool) -> None:
    """Copy the products target schema from upstream to synthetic side.

    Writes BOTH:
    * ``products_target_schema.json`` (preserves the upstream filename
      for callers that reference it explicitly), and
    * ``target_schema.json`` (the canonical music/games/companies
      filename that ``variant_loader._load_target_schema`` and the SM
      committee runner expect).

    Idempotent — overwrites the synthetic-side copy each run.
    """
    upstream = UPSTREAM_INPUT_DIR / "schemamatching" / "products_target_schema.json"
    if not upstream.exists():
        logger.warning("Upstream target schema missing: %s", upstream)
        return
    out_dir = synthetic_input / "schemamatching"
    targets = [
        out_dir / "products_target_schema.json",
        out_dir / "target_schema.json",
    ]
    if dry_run:
        for target in targets:
            logger.info("[dry-run] would copy %s -> %s", upstream.name, target)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for target in targets:
        shutil.copy2(upstream, target)
        logger.info("copied %s -> %s", upstream.name, target)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize the products domain at the synthetic-side "
            "data_root (rewrite ids to source-prefixed strings + "
            "rename EM gold to canonical layout). Reads from the "
            "upstream usecases/products/ directory but never writes "
            "to it."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect changes without writing them.",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    synthetic_input = _synthetic_input_dir()
    logger.info("Synthetic-side input dir: %s", synthetic_input)

    logger.info("Phase 1: rewrite source JSON ids")
    upstream_data = UPSTREAM_INPUT_DIR / "data"
    for name in SOURCE_NAMES:
        _rewrite_source_json(
            name,
            upstream_data / f"{name}.json",
            synthetic_input / "data" / f"{name}.json",
            dry_run=args.dry_run,
        )

    logger.info("Phase 2: rewrite EM gold ids + canonicalise names")
    upstream_em = UPSTREAM_INPUT_DIR / "entity_matching_gt"
    synthetic_em = synthetic_input / "entitymatching"
    for legacy_stem, (left, right, canon_stem) in EM_PAIR_STEMS.items():
        for split in PER_PAIR_SPLITS:
            upstream_csv = upstream_em / f"{legacy_stem}_{split}.csv"
            if not upstream_csv.exists():
                logger.warning("Skip missing upstream EM gold: %s", upstream_csv)
                continue
            synthetic_csv = synthetic_em / f"{canon_stem}_{split}.csv"
            _rewrite_em_gold_csv(
                upstream_csv,
                synthetic_csv,
                left,
                right,
                dry_run=args.dry_run,
            )

    logger.info("Phase 3: mirror target schema for SM committee")
    _copy_target_schema(synthetic_input, dry_run=args.dry_run)

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
