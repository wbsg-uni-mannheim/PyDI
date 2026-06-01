"""Canonical bundle loader for the best-of-breed pipeline.

Reads each domain's authoritative data directly from
``usecases/<domain>/`` (the PyDI canonical tree the human-baseline
notebooks read) and returns a :class:`VariantBundle` shaped exactly
like the synthetic-side ``variant_loader.load_variant(domain,
"baseline")`` output.

Why this exists
---------------
``usecases_synthetic/config/domains/products.yaml`` ships a
``data_root: usecases_synthetic/usecases`` override so the synthetic
pipeline doesn't perturb the notebook's input tree. That override
also leaks into the best-of-breed pipeline (``load_pipeline_bundle``
goes through ``load_variant``), making the BoB pipeline read products
data from ``usecases_synthetic/usecases/products/``.

For music / games / companies the bundle already resolves to
canonical ``usecases/<domain>/`` (no synthetic ``data_root``
override). The audit at ``pipelines/scripts/audit_data_sources.py``
confirms this.

The 2026-06-01 directive is that ALL domains read evaluation gold
from canonical ``usecases/<domain>/``. This module supplies the
products-specific canonical loader. ``music``, ``games``,
``companies`` go through the default ``load_variant`` path which
already lands on canonical.

Layout differences handled in-memory
------------------------------------
The canonical ``usecases/products/`` tree differs from the synthetic
loader's expected layout in three places. The loader translates each
in-memory; no data is forked or copied to a new location.

1. **Sources** — canonical sources live under
   ``input/data_cleaned_final/dataset_<n>_normalized.json`` (the
   27-attribute hardware schema the notebook uses), with bare-integer
   IDs. The loader applies the standard ``products_<n>_`` prefix at
   load time so downstream stages see the prefixed IDs the rest of
   the pipeline expects.

2. **EM gold** — canonical files live under
   ``input/entity_matching_gt/`` (note the underscore + ``_gt``
   suffix) with names ``prod<n>_to_prod<m>_<split>.csv`` and a
   ``id1,id2,label`` header where label is ``0``/``1``. The loader
   reads them, prefixes both IDs, and converts the label to the
   ``true``/``false`` string form synthetic-side files use.

3. **Fusion silver** — canonical files live under ``input/fusion/`` as
   CSVs (``fusion_test_set.csv``, ``fusion_validation_set.csv``) with
   a pair-based shape (id_left + id_right + ~27 attribute columns +
   a ``filled`` flag). The loader filters ``filled == 'y'``, prefixes
   ``id_left`` into the ``products_1_`` scheme, and reshapes to flat
   fused-record form (one row per fused entity, ``id`` column = anchor
   product_1 id). This is the SAME 27-attribute hardware silver the
   workflow notebook scores against.

SM gold
-------
``usecases/products/input/schemamatching/sm_mapping_gold.csv`` was
copied from the synthetic side (hand-authored, applies regardless of
ID scheme). The loader reads it from the canonical location.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from usecases_synthetic.lib.variant_loader import VariantBundle

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]


_PRODUCTS_SOURCE_FILES = {
    "products_1": "dataset_1_normalized.json",
    "products_2": "dataset_2_normalized.json",
    "products_3": "dataset_3_normalized.json",
    "products_4": "dataset_4_normalized.json",
}

_PRODUCTS_EM_PAIRS: tuple[tuple[str, str, str], ...] = (
    # (src1, src2, canonical-filename-stem)
    ("products_1", "products_2", "prod1_to_prod2"),
    ("products_1", "products_3", "prod1_to_prod3"),
    ("products_1", "products_4", "prod1_to_prod4"),
)

# Pair-based fusion silver columns that don't belong on a flat
# fused-record frame.
_FUSION_DROP_COLS = {
    "id_right",
    "source_left",
    "source_right",
    "filled",
    "gt_source_url",
    "gt_source_2",
    "title_left_raw",
    "title_right_raw",
    "desc_left_raw",
    "desc_right_raw",
    "sampling_type",
}


def _prefix_ids(df: pd.DataFrame, column: str, prefix: str) -> pd.DataFrame:
    df = df.copy()
    df[column] = prefix + df[column].astype(str)
    return df


def _load_canonical_sources(
    products_root: Path,
) -> dict[str, pd.DataFrame]:
    """Load the 4 canonical product sources with in-memory id prefix."""
    sources: dict[str, pd.DataFrame] = {}
    for source_name, fname in _PRODUCTS_SOURCE_FILES.items():
        path = products_root / "input" / "data_cleaned_final" / fname
        if not path.exists():
            raise FileNotFoundError(
                f"Canonical products source missing: {path}. "
                f"Check usecases/products/input/data_cleaned_final/."
            )
        with path.open() as f:
            records = json.load(f)
        df = pd.DataFrame(records)
        prefix = f"{source_name}_"
        df = _prefix_ids(df, "id", prefix)
        df.attrs["dataset_name"] = source_name
        sources[source_name] = df
    return sources


def _read_canonical_em_csv(path: Path, src1: str, src2: str) -> pd.DataFrame:
    """Read a canonical EM CSV and translate to the synthetic shape:
    prefix id1+id2 and lowercase the label to ``true``/``false``."""
    df = pd.read_csv(path)
    if not {"id1", "id2", "label"}.issubset(df.columns):
        raise ValueError(
            f"Canonical EM CSV {path} missing required cols id1/id2/label; "
            f"got {list(df.columns)}"
        )
    df = _prefix_ids(df, "id1", f"{src1}_")
    df = _prefix_ids(df, "id2", f"{src2}_")
    # Canonical labels are int 0/1; synthetic uses string false/true.
    df["label"] = df["label"].apply(lambda v: "true" if int(v) == 1 else "false")
    return df[["id1", "id2", "label"]]


def _load_canonical_em(
    products_root: Path,
) -> tuple[
    dict[tuple[str, str], pd.DataFrame],
    dict[tuple[str, str], dict[str, pd.DataFrame]],
]:
    """Build em_gold + em_splits from canonical entity_matching_gt/."""
    em_dir = products_root / "input" / "entity_matching_gt"
    em_gold: dict[tuple[str, str], pd.DataFrame] = {}
    em_splits: dict[tuple[str, str], dict[str, pd.DataFrame]] = {}
    for src1, src2, stem in _PRODUCTS_EM_PAIRS:
        all_path = em_dir / f"{stem}_all.csv"
        if not all_path.exists():
            logger.warning(
                "Canonical EM gold missing for %s-%s at %s; skipping pair.",
                src1,
                src2,
                all_path,
            )
            continue
        em_gold[(src1, src2)] = _read_canonical_em_csv(all_path, src1, src2)

        pair_splits: dict[str, pd.DataFrame] = {}
        for split in ("train", "val", "test", "all"):
            sp_path = em_dir / f"{stem}_{split}.csv"
            if sp_path.exists():
                pair_splits[split] = _read_canonical_em_csv(sp_path, src1, src2)
        if pair_splits:
            em_splits[(src1, src2)] = pair_splits
    return em_gold, em_splits


def _load_canonical_fusion(
    products_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Read the canonical fusion silver CSVs and reshape to the flat
    fused-record form ``bundle.fusion_gold`` expects.

    Both CSVs are pair-based with id_left (products_1 anchor),
    id_right (paired source), ~27 attribute columns, and a
    ``filled`` flag. Keep ``filled == 'y'`` rows, drop pair-only
    metadata, and rename ``id_left`` -> ``id`` with the products_1
    prefix.
    """
    fusion_dir = products_root / "input" / "fusion"

    def _load_one(name: str) -> pd.DataFrame | None:
        path = fusion_dir / name
        if not path.exists():
            return None
        df = pd.read_csv(path)
        if "filled" in df.columns:
            df = df[df["filled"] == "y"].copy()
        df = df.drop(
            columns=[c for c in _FUSION_DROP_COLS if c in df.columns],
            errors="ignore",
        )
        df = df.rename(columns={"id_left": "id"})
        df = _prefix_ids(df, "id", "products_1_")
        return df

    test = _load_one("fusion_test_set.csv")
    if test is None:
        raise FileNotFoundError(
            f"Canonical products fusion test silver missing at "
            f"{fusion_dir / 'fusion_test_set.csv'}"
        )
    validation = _load_one("fusion_validation_set.csv")
    return test, validation


def _load_canonical_sm_gold(products_root: Path) -> pd.DataFrame | None:
    """Read the canonical SM gold (copied here from synthetic)."""
    path = products_root / "input" / "schemamatching" / "sm_mapping_gold.csv"
    if not path.exists():
        logger.warning(
            "Canonical SM gold not found at %s; SM evaluation will be "
            "unavailable for this run.",
            path,
        )
        return None
    return pd.read_csv(path)


def _load_canonical_target_schema(products_root: Path) -> dict:
    """Read the products target schema. Prefers ``target_schema.json``,
    falls back to ``products_target_schema.json``."""
    sm_dir = products_root / "input" / "schemamatching"
    for fname in ("target_schema.json", "products_target_schema.json"):
        path = sm_dir / fname
        if path.exists():
            with path.open() as f:
                return json.load(f)
    raise FileNotFoundError(
        f"No target_schema.json or products_target_schema.json under {sm_dir}"
    )


def load_canonical_products_bundle() -> VariantBundle:
    """Return a baseline :class:`VariantBundle` for products sourced
    from canonical ``usecases/products/``."""
    products_root = REPO_ROOT / "usecases" / "products"
    if not products_root.exists():
        raise FileNotFoundError(f"Canonical products root not found at {products_root}")

    sources = _load_canonical_sources(products_root)
    em_gold, em_splits = _load_canonical_em(products_root)
    fusion_gold, fusion_validation = _load_canonical_fusion(products_root)
    sm_mapping = _load_canonical_sm_gold(products_root)
    target_schema = _load_canonical_target_schema(products_root)

    logger.info(
        "Loaded canonical products bundle from %s "
        "(sources=%d, em_pairs=%d, fusion_test_rows=%d, sm_gold=%s)",
        products_root,
        len(sources),
        len(em_gold),
        len(fusion_gold),
        "yes" if sm_mapping is not None else "no",
    )

    return VariantBundle(
        domain="products",
        level="baseline",
        sources=sources,
        target_schema=target_schema,
        sm_mapping=sm_mapping,
        em_gold=em_gold,
        em_splits=em_splits,
        fusion_gold=fusion_gold,
        fusion_validation=fusion_validation,
        pooled_positives=None,
        variant_root=products_root,
    )
