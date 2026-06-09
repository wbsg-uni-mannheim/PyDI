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
from typing import Any

import pandas as pd

from usecases_synthetic.lib.variant_loader import (
    VariantBundle,
    _load_sm_mapping as _load_sm_gold,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]


_PRODUCTS_SOURCE_FILES = {
    "products_1": "dataset_1.json",
    "products_2": "dataset_2.json",
    "products_3": "dataset_3.json",
    "products_4": "dataset_4.json",
}

# Per-source raw-column -> canonical-column mapping mirrored verbatim
# from usecases/products/products_workflow_minimal.ipynb (the
# notebook's ``SCHEMA_MATCHES`` dict). The canonical product sources
# ship varied column naming styles (snake_case / camelCase / PascalCase
# / terse ERP codes) precisely to exercise schema matching. The
# canonical loader applies this mapping in-memory so downstream stages
# see the canonical target-schema column names, while the BoB SM
# committee still scores against the matching gold at
# ``input/schemamatching/sm_mapping_gold.csv`` (derived from the same
# dict — every (source_dataset, source_column) row maps to the same
# target_column listed here).
_PRODUCTS_SCHEMA_MATCHES: dict[str, dict[str, str]] = {
    "products_1": {
        "id": "id",
        "manufacturer": "brand",
        "product_name": "title",
        "product_description": "description",
        "list_price": "price",
        "currency_code": "priceCurrency",
        "cluster_id": "cluster_id",
        "product_url": "url",
        "name_and_description": "title_description",
        "model_name": "model",
        "manufacturer_part_number": "model_number",
        "category": "product_type",
        "gpu_chipset": "chipset_name",
        "video_memory_gb": "vram_gb",
        "capacity_gb": "storage_gb",
        "sequential_read_mb_s": "read_speed_mb_s",
        "sequential_write_mb_s": "write_speed_mb_s",
        "bus_standard": "bus_type",
        "interface": "interface_type",
        "width_millimeters": "width_mm",
        "length_millimeters": "length_mm",
        "height_millimeters": "height_mm",
        "weight_grams": "weight_g",
        "connector": "storage_connection_type",
        "memory_technology": "memory_type",
        "colour": "color",
        "form_factor": "form_factor",
    },
    "products_2": {
        "id": "id",
        "brandName": "brand",
        "name": "title",
        "descriptionText": "description",
        "priceAmount": "price",
        "currency": "priceCurrency",
        "cluster_id": "cluster_id",
        "productUrl": "url",
        "titleAndDescription": "title_description",
        "modelName": "model",
        "mpn": "model_number",
        "productCategory": "product_type",
        "chipset": "chipset_name",
        "vramGb": "vram_gb",
        "capacityGb": "storage_gb",
        "readSpeedMbps": "read_speed_mb_s",
        "writeSpeedMbps": "write_speed_mb_s",
        "busType": "bus_type",
        "interfaceType": "interface_type",
        "widthMm": "width_mm",
        "depthMm": "length_mm",
        "heightMm": "height_mm",
        "weightG": "weight_g",
        "connectionType": "storage_connection_type",
        "memoryType": "memory_type",
        "color": "color",
        "formFactor": "form_factor",
    },
    "products_3": {
        "id": "id",
        "Brand": "brand",
        "ProductTitle": "title",
        "Details": "description",
        "Price": "price",
        "Currency": "priceCurrency",
        "cluster_id": "cluster_id",
        "Link": "url",
        "TitleDetails": "title_description",
        "Model": "model",
        "PartNo": "model_number",
        "Type": "product_type",
        "Chipset": "chipset_name",
        "MemorySizeGB": "vram_gb",
        "CapacityGB": "storage_gb",
        "ReadMBs": "read_speed_mb_s",
        "WriteMBs": "write_speed_mb_s",
        "Bus": "bus_type",
        "Interface": "interface_type",
        "WidthMM": "width_mm",
        "LengthMM": "length_mm",
        "HeightMM": "height_mm",
        "WeightG": "weight_g",
        "Connector": "storage_connection_type",
        "MemoryType": "memory_type",
        "Colour": "color",
        "FormFactor": "form_factor",
    },
    "products_4": {
        "id": "id",
        "mfr": "brand",
        "name": "title",
        "desc": "description",
        "amt": "price",
        "cur": "priceCurrency",
        "cluster_id": "cluster_id",
        "link": "url",
        "name_desc": "title_description",
        "mdl": "model",
        "pn": "model_number",
        "cat": "product_type",
        "chip": "chipset_name",
        "vram": "vram_gb",
        "cap_gb": "storage_gb",
        "rd_mbs": "read_speed_mb_s",
        "wr_mbs": "write_speed_mb_s",
        "bus": "bus_type",
        "iface": "interface_type",
        "w_mm": "width_mm",
        "l_mm": "length_mm",
        "h_mm": "height_mm",
        "wt_g": "weight_g",
        "conn": "storage_connection_type",
        "mem": "memory_type",
        "clr": "color",
        "ff": "form_factor",
    },
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
    """Load the 4 canonical product sources with their RAW per-source
    column names intact (manufacturer / brandName / Brand / mfr, ...).

    The SM committee scores these raw columns against the canonical
    SM gold; the orchestrator applies the gold (or the SM winner's)
    mapping after SM scoring to translate sources to canonical column
    names for downstream stages. IDs are prefixed so they remain
    cross-source unique.
    """
    sources: dict[str, pd.DataFrame] = {}
    for source_name, fname in _PRODUCTS_SOURCE_FILES.items():
        path = products_root / "input" / "data" / fname
        if not path.exists():
            raise FileNotFoundError(
                f"Canonical products source missing: {path}. "
                f"Check usecases/products/input/data/."
            )
        with path.open() as f:
            records = json.load(f)
        df = pd.DataFrame(records)
        if "id" not in df.columns:
            raise KeyError(
                f"Canonical source {source_name} has no 'id' column; cannot "
                "prefix. Check the dataset file."
            )
        prefix = f"{source_name}_"
        df = _prefix_ids(df, "id", prefix)
        df.attrs["dataset_name"] = source_name
        # Tag the source so the orchestrator knows it needs post-SM
        # gold-based column translation before EM / Norm / Fusion.
        df.attrs["needs_sm_column_translation"] = True
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
    """Read the canonical SM gold (copied here from synthetic).

    Prefers ``sm_mapping_gold.json`` (``kind: pydi_schema_mapping_gold``) over
    the legacy CSV. The canonical/BoB path keeps *raw* source column names,
    which the JSON gold is authored against, so it is consumed as-is here (no
    source-column reconciliation, unlike the synthetic ``load_variant`` path,
    which renames columns and therefore reconciles the gold).
    """
    sm_dir = products_root / "input" / "schemamatching"
    mapping = _load_sm_gold(sm_dir, baseline=True)
    if mapping is None:
        logger.warning(
            "Canonical SM gold not found at %s; SM evaluation will be "
            "unavailable for this run.",
            sm_dir / "sm_mapping_gold.json",
        )
    return mapping


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


def load_canonical_products_workflow_silver() -> "Any":
    """Build a :class:`SilverStandard` for products directly from the
    canonical wide-format ``fusion_test_set.csv``.

    The PyDI ``load_workflow_silver`` loader expects
    ``input/fusion/test_set.xml``; the canonical products tree only
    has CSV (id_left/id_right/source_left/source_right/cluster_id +
    27 attribute columns). This helper reads that CSV directly and
    assembles the same :class:`SilverStandard` shape (fused frame
    keyed by ``cluster_id`` + membership table).
    """
    from PyDI.evaluation.silver_standard import SilverStandard

    products_root = REPO_ROOT / "usecases" / "products"
    csv_path = products_root / "input" / "fusion" / "fusion_test_set.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    if "filled" in df.columns:
        df = df[df["filled"] == "y"].copy()
    if "cluster_id" not in df.columns:
        raise ValueError(
            f"Canonical fusion silver missing cluster_id column at {csv_path}"
        )

    # Build the fused frame: one row per cluster_id with the canonical
    # expected attribute values. Drop pair-only metadata columns; what
    # remains is the cluster-level fused state.
    drop_cols = _FUSION_DROP_COLS | {
        "id_left",
        "id_right",
        "source_left",
        "source_right",
    }
    keep_cols = [c for c in df.columns if c not in drop_cols]
    grouped = df.drop_duplicates(subset=["cluster_id"]).copy()
    fused = grouped[keep_cols].reset_index(drop=True)
    # cluster_id stays as the canonical cluster id; no prefixing needed.

    # Build membership from the pair rows: each (id_left, source_left,
    # cluster_id) and (id_right, source_right, cluster_id) becomes one
    # row. The CSV uses short source codes ``p1``/``p2``/``p3``/``p4``
    # and bare-int ids; translate them to the pipeline's
    # ``products_<n>`` convention with ``products_<n>_<id>`` prefixed
    # record ids so membership aligns with bundle.sources keys.
    _short_to_full = {
        "p1": "products_1",
        "p2": "products_2",
        "p3": "products_3",
        "p4": "products_4",
    }

    def _full_source(short: Any) -> str | None:
        if pd.isna(short):
            return None
        return _short_to_full.get(str(short), str(short))

    def _prefix_record_id(record_id: Any, full_source: str | None) -> str | None:
        if pd.isna(record_id) or full_source is None:
            return None
        return f"{full_source}_{record_id}"

    left_rows = df[["id_left", "source_left", "cluster_id"]].rename(
        columns={"id_left": "record_id", "source_left": "source"}
    )
    right_rows = df[["id_right", "source_right", "cluster_id"]].rename(
        columns={"id_right": "record_id", "source_right": "source"}
    )
    membership = pd.concat([left_rows, right_rows], ignore_index=True).dropna(
        subset=["record_id", "source"]
    )
    membership["source"] = [_full_source(s) for s in membership["source"]]
    membership = membership.dropna(subset=["source"])
    membership["record_id"] = [
        _prefix_record_id(rid, src)
        for rid, src in zip(membership["record_id"], membership["source"])
    ]
    membership = (
        membership.dropna(subset=["record_id"])
        .drop_duplicates(subset=["record_id", "source", "cluster_id"])
        .reset_index(drop=True)
    )

    logger.info(
        "Loaded canonical products workflow silver: %d clusters, %d "
        "membership rows (from %s)",
        len(fused),
        len(membership),
        csv_path.relative_to(REPO_ROOT),
    )
    return SilverStandard(
        fused=fused,
        membership=membership,
        cell_provenance=None,
    )


# ===========================================================================
# Papers
# ===========================================================================
#
# Papers is a new (2026 pull) domain not yet wired into
# usecases_synthetic. It has 3 sources (dblp, crossref, open_alex) all
# read from JSONL with PyDI.io.load_json(add_index=True), which
# generates ids of the form ``<source>-<NNNNN>`` (dash separator,
# zero-padded). The EM gold under input/entitymatching/ uses the
# domain-specific columns ``id_dblp + id_<other>`` (NOT the standard
# ``id1/id2/label``); the loader renames them. Fusion gold is JSONL
# (NOT XML/CSV) and joins on ``doi`` — the loader returns it
# verbatim. SM gold is not authored yet.

_PAPERS_SOURCE_FILES = {
    "dblp": "dblp.jsonl",
    "crossref": "crossref.jsonl",
    "open_alex": "open_alex.jsonl",
}

# EM pair declarations: (src1, src2, em-csv-stem).
# Filenames are ``<stem>_{train,val,test}.csv``.
_PAPERS_EM_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("dblp", "crossref", "dblp_crossref"),
    ("dblp", "open_alex", "dblp_openalex"),
)


def _load_canonical_papers_sources(
    papers_root: Path,
) -> dict[str, pd.DataFrame]:
    """Load papers' 3 JSONL sources via :func:`PyDI.io.load_json` so
    the auto-generated ids match the dash-prefixed format the EM gold
    references (``dblp-NNNNN``, ``crossref-NNNNN``, ``open_alex-NNNNN``)."""
    from PyDI.io import load_json

    sources: dict[str, pd.DataFrame] = {}
    for name, fname in _PAPERS_SOURCE_FILES.items():
        path = papers_root / "input" / "data" / fname
        if not path.exists():
            raise FileNotFoundError(
                f"Canonical papers source missing: {path}. "
                f"Check usecases/papers/input/data/."
            )
        df = load_json(path, add_index=True, lines=True, name=name)
        id_col = f"{name}_id"
        if id_col in df.columns and "id" not in df.columns:
            df = df.rename(columns={id_col: "id"})
        # Canonicalize the heterogeneous per-source papers schema (dblp
        # ``publication_title`` / crossref ``title_text`` / open_alex
        # ``display_title`` -> ``title``; ``author_list`` etc. -> ``authors``;
        # ...) so the EM/fusion committees see the canonical columns their
        # ``blocking_name_column: title`` / ``text_cols: [title, ...]`` and
        # ditto fields expect. This matches the *variant* papers loader (which
        # already ships canonical columns) and the synthetic
        # ``normalize_loaded_source``. Without it, the committee
        # ``column_mapping: {}`` (papers P1 fix, correct for the canonicalizing
        # loader) leaves blocking with no ``title`` column -> zero candidates
        # -> empty EM. papers SM gold is unauthored (SM scoring is skipped), so
        # there is no raw-name learning to preserve here.
        from usecases_synthetic.lib.loaders import _PAPERS_SOURCE_COLUMN_MAP

        canon = {
            k: v
            for k, v in _PAPERS_SOURCE_COLUMN_MAP.get(name, {}).items()
            if k in df.columns and v not in df.columns
        }
        if canon:
            df = df.rename(columns=canon)
        # Authors ship as JSON arrays, so ``load_json`` loads them as
        # Python lists. List-valued cells break every scalar-assuming
        # SM/EM matcher (coma_hybrid: pd.notna(list) array-truthiness;
        # magneto: list in NULL_REPRESENTATIONS unhashable). The SM
        # committee LEARNS the raw->canonical mapping, so it sees the
        # raw column names (``author_list``/``contributor_names``/
        # ``authors_list``) -- not yet renamed to canonical ``authors``.
        # Stringify any column that contains list cells; the fusion
        # runner re-parses via literal_eval (fusion_committee_papers
        # declares gold_list_columns: [authors]). Mirrors
        # usecases_synthetic.lib.loaders.normalize_loaded_source:495-508
        # which the BoB pipeline bypasses by calling ``load_json``
        # directly.
        for col in df.columns:
            if df[col].apply(lambda v: isinstance(v, list)).any():
                df[col] = df[col].apply(lambda v: str(v) if isinstance(v, list) else v)
        df.attrs["dataset_name"] = name
        sources[name] = df
    return sources


def _read_papers_em_csv(path: Path, src1: str, src2: str) -> pd.DataFrame:
    """Read a papers EM CSV. The columns are ``id_<src1>,id_<src2>,label``;
    rename to ``id1,id2,label`` and convert 0/1 labels to ``true``/``false``
    matching the synthetic-side convention. IDs are already prefixed."""
    df = pd.read_csv(path)
    col1, col2 = f"id_{src1}", f"id_{src2}"
    # The crossref+openalex files use the non-anchor source's prefix
    # directly in the column name; accept short aliases too.
    if col1 not in df.columns or col2 not in df.columns:
        # Fall back to whatever id_* columns are present, with src1
        # matching the column whose values start with src1+"-".
        id_cols = [c for c in df.columns if c.startswith("id_")]
        if len(id_cols) != 2:
            raise ValueError(
                f"Papers EM CSV {path} has unexpected columns {list(df.columns)}"
            )
        col1, col2 = id_cols
    rename = {col1: "id1", col2: "id2"}
    df = df.rename(columns=rename)
    if "label" not in df.columns:
        raise ValueError(f"Papers EM CSV {path} missing label column")
    df["label"] = df["label"].apply(lambda v: "true" if int(v) == 1 else "false")
    return df[["id1", "id2", "label"]]


def _load_canonical_papers_em(
    papers_root: Path,
) -> tuple[
    dict[tuple[str, str], pd.DataFrame],
    dict[tuple[str, str], dict[str, pd.DataFrame]],
]:
    em_dir = papers_root / "input" / "entitymatching"
    em_gold: dict[tuple[str, str], pd.DataFrame] = {}
    em_splits: dict[tuple[str, str], dict[str, pd.DataFrame]] = {}
    for src1, src2, stem in _PAPERS_EM_PAIRS:
        # Treat the union of train+val+test as ``all`` since the
        # canonical papers tree doesn't ship an explicit ``_all.csv``.
        pair_splits: dict[str, pd.DataFrame] = {}
        for split in ("train", "val", "test"):
            sp_path = em_dir / f"{stem}_{split}.csv"
            if sp_path.exists():
                pair_splits[split] = _read_papers_em_csv(sp_path, src1, src2)
        if not pair_splits:
            logger.warning(
                "Papers EM gold missing for %s-%s under %s; skipping pair.",
                src1,
                src2,
                em_dir,
            )
            continue
        all_df = pd.concat(pair_splits.values(), ignore_index=True)
        pair_splits["all"] = all_df
        em_gold[(src1, src2)] = all_df
        em_splits[(src1, src2)] = pair_splits
    return em_gold, em_splits


def _load_canonical_papers_fusion(
    papers_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Read the JSONL fusion silvers; both have one row per fused
    paper keyed on ``doi``."""
    fusion_dir = papers_root / "input" / "fusion"
    test_path = fusion_dir / "fusion_test.jsonl"
    if not test_path.exists():
        raise FileNotFoundError(
            f"Canonical papers fusion test silver missing at {test_path}"
        )
    def _read_jsonl(path: Path) -> pd.DataFrame:
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return pd.DataFrame(records)

    test = _read_jsonl(test_path)
    val_path = fusion_dir / "fusion_val.jsonl"
    validation = _read_jsonl(val_path) if val_path.exists() else None
    return test, validation


def load_canonical_papers_bundle() -> VariantBundle:
    """Return a baseline :class:`VariantBundle` for papers sourced
    from canonical ``usecases/papers/``.

    The papers SM gold (``sm_mapping_gold.json``) is authored against
    the RAW on-disk source columns, so it is reconciled here to the
    canonical column names that :func:`_load_canonical_papers_sources`
    produces — otherwise every SM committee member scores 0.0 from a
    raw-vs-canonical column mismatch.
    """
    papers_root = REPO_ROOT / "usecases" / "papers"
    if not papers_root.exists():
        raise FileNotFoundError(f"Canonical papers root not found at {papers_root}")

    sources = _load_canonical_papers_sources(papers_root)
    em_gold, em_splits = _load_canonical_papers_em(papers_root)
    fusion_gold, fusion_validation = _load_canonical_papers_fusion(papers_root)
    target_schema_path = papers_root / "input" / "schemamatching" / "target_schema.json"
    if not target_schema_path.exists():
        raise FileNotFoundError(target_schema_path)
    with target_schema_path.open() as f:
        target_schema = json.load(f)

    sm_mapping = _load_sm_gold(papers_root / "input" / "schemamatching", baseline=True)

    # The committed papers SM gold (sm_mapping_gold.json) is authored against the
    # RAW on-disk source columns (doi_value/display_title/work_kind/...), but
    # _load_canonical_papers_sources renames every source column to its canonical
    # name via _PAPERS_SOURCE_COLUMN_MAP before any committee sees the frames.
    # Reconcile the gold's source_column to the same canonical names so the SM
    # committee scores its (canonical-named) predictions against a canonical-keyed
    # gold; without this every SM member scores 0.0 from a raw-vs-canonical tuple
    # mismatch. Mirrors variant_loader._reconcile_sm_gold_source_columns; rows
    # whose source_column is already canonical (none, here) pass through unchanged.
    if sm_mapping is not None and not sm_mapping.empty:
        from usecases_synthetic.lib.loaders import _PAPERS_SOURCE_COLUMN_MAP

        sm_mapping = sm_mapping.copy()
        sm_mapping["source_column"] = [
            _PAPERS_SOURCE_COLUMN_MAP.get(str(ds), {}).get(str(sc), sc)
            for ds, sc in zip(
                sm_mapping["source_dataset"], sm_mapping["source_column"]
            )
        ]

    logger.info(
        "Loaded canonical papers bundle from %s "
        "(sources=%d, em_pairs=%d, fusion_test_rows=%d, sm_gold=%s)",
        papers_root,
        len(sources),
        len(em_gold),
        len(fusion_gold),
        "yes" if sm_mapping is not None else "no",
    )

    return VariantBundle(
        domain="papers",
        level="baseline",
        sources=sources,
        target_schema=target_schema,
        sm_mapping=sm_mapping,
        em_gold=em_gold,
        em_splits=em_splits,
        fusion_gold=fusion_gold,
        fusion_validation=fusion_validation,
        pooled_positives=None,
        variant_root=papers_root,
    )
