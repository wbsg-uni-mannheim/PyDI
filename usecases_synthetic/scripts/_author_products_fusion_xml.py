"""Author the canonical fusion XML for products at the synthetic-side data_root.

The upstream products notebook workflow ships fusion gold as flat CSVs
(one row per matched cluster, with all canonical attribute values inline):

* ``usecases/products/input/fusion/fusion_validation_set.csv``
* ``usecases/products/input/fusion/fusion_test_set.csv``

The synthetic pipeline expects fusion gold in the music/games/companies
XML format (one ``<product>`` element per fused entity with a
``provenance`` attribute on each canonical-attribute child element).
This script reads the upstream CSVs and writes the canonical-format XML
to the synthetic-side data_root:

* ``usecases_synthetic/usecases/products/input/fusion/validation_set.xml``
* ``usecases_synthetic/usecases/products/input/fusion/test_set.xml``

The upstream ``usecases/products/`` directory is never modified.

Provenance policy: every canonical-attribute cell is tagged with the
union of the left + right source ids (``"<id_left>+<id_right>"``).
This matches the policy described in plan_s1_products.md §"Hard
blockers" P0.5; the upstream CSV does not record per-cell attribution.

ID rewriting: the upstream CSV uses bare-int ``id_left`` / ``id_right``
and short source labels ``p1`` / ``p2`` / ``p3`` / ``p4``. The
synthetic side carries source-prefixed string ids
(``products_<n>_<int>``); this script applies the same prefix here so
the XML provenance ids resolve in the synthetic-side source data.

The fused entity ``<id>`` is always the left-source id; ``p1``
(products_1) is the canonical anchor for every CSV row today.

Run::

    source pydi-dev/bin/activate
    python usecases_synthetic/scripts/_author_products_fusion_xml.py
"""

from __future__ import annotations

import argparse
import logging
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from usecases_synthetic.lib.domain_config import (  # noqa: E402
    USECASES_DIR,
    data_root_for_domain,
    load_domain_config,
)

UPSTREAM_FUSION_DIR = REPO_ROOT / "usecases" / "products" / "input" / "fusion"
DOMAIN = "products"


def _canonical_attrs() -> tuple[str, ...]:
    """Full canonical attribute scope for the products fusion XML.

    Sourced from the products domain config (``attribute_classes``) so the
    fusion gold always carries the full wide schema (plan_revision R1 /
    R10-C) and never drifts from the YAML. Was previously a hardcoded
    5-tuple (title/brand/description/price/priceCurrency) which silently
    emitted a narrow 6-tag fusion set even though the upstream CSV is fully
    populated across all 19 canonical attributes — the staleness this fix
    addresses.
    """
    return tuple(load_domain_config(DOMAIN).attribute_classes.keys())


SHORT_TO_FULL = {
    "p1": "products_1",
    "p2": "products_2",
    "p3": "products_3",
    "p4": "products_4",
}

logger = logging.getLogger("author_products_fusion_xml")


def _synthetic_fusion_dir() -> Path:
    """Resolve the synthetic-side fusion dir from the domain YAML."""
    root = data_root_for_domain(DOMAIN) or USECASES_DIR
    return root / DOMAIN / "input" / "fusion"


def _prefix(short_source: str, raw_id: object) -> str:
    """Return ``products_<n>_<raw_id>`` from short label + raw id."""
    full = SHORT_TO_FULL[short_source]
    text = str(raw_id)
    if text.startswith(f"{full}_"):
        return text
    return f"{full}_{text}"


def _format_cell(value: object) -> str:
    """Render a fused-cell value for XML emission; NaN/None -> empty."""
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value)


def _build_xml(df: pd.DataFrame) -> ET.ElementTree:
    """Build the canonical fusion XML tree from a fusion CSV DataFrame."""
    canonical_attrs = _canonical_attrs()
    root = ET.Element("products")
    for _, row in df.iterrows():
        product = ET.SubElement(root, "product")
        left_full = _prefix(row["source_left"], row["id_left"])
        right_full = _prefix(row["source_right"], row["id_right"])
        provenance = f"{left_full}+{right_full}"

        id_el = ET.SubElement(product, "id")
        id_el.text = left_full

        for attr in canonical_attrs:
            el = ET.SubElement(product, attr)
            text = _format_cell(row.get(attr))
            if text:
                el.text = text
                el.set("provenance", provenance)
            else:
                el.set("provenance", "")
    return ET.ElementTree(root)


def _write_xml(tree: ET.ElementTree, path: Path) -> None:
    """Write the XML tree with the canonical declaration + 2-space indent."""
    ET.indent(tree, space="  ", level=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def convert(csv_path: Path, xml_path: Path) -> int:
    """Convert ``csv_path`` to ``xml_path``. Returns row count."""
    df = pd.read_csv(csv_path)
    tree = _build_xml(df)
    _write_xml(tree, xml_path)
    logger.info("Wrote %d products -> %s", len(df), xml_path)
    return len(df)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Author the canonical fusion XML for products at the "
            "synthetic-side data_root. Reads upstream "
            "usecases/products/input/fusion/*.csv but never writes to "
            "the upstream directory."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan without writing.",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    synthetic_fusion = _synthetic_fusion_dir()
    pairs = [
        (
            UPSTREAM_FUSION_DIR / "fusion_validation_set.csv",
            synthetic_fusion / "validation_set.xml",
        ),
        (
            UPSTREAM_FUSION_DIR / "fusion_test_set.csv",
            synthetic_fusion / "test_set.xml",
        ),
    ]
    for csv_path, xml_path in pairs:
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing upstream fusion CSV: {csv_path}")
        if args.dry_run:
            logger.info("[dry-run] would convert %s -> %s", csv_path.name, xml_path)
            continue
        convert(csv_path, xml_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
