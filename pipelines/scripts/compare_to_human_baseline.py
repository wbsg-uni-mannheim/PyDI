#!/usr/bin/env python3
"""Compute the e2e panel against the human-baseline notebook output.

The human baseline lives at
``usecases/<domain>/<domain>_workflow_minimal.ipynb``. Its fused
output is not persisted on disk by the notebook itself, so this
script reads a *cached* copy of that output (the notebook author
runs the notebook once and saves ``fused.csv`` +
``correspondences.csv`` under ``pipelines/<domain>/baselines/``).

The cached output uses the notebook's native ID scheme (e.g. bare-int
IDs for products); the silver standard uses source-prefixed IDs
(e.g. ``products_1_<n>``). This script translates the notebook output
into the prefixed scheme by joining against the synthetic-side
sources (where the prefix → bare-int mapping is unambiguous via the
``cluster_id`` column).

Then it computes the panel against the same silver standard as the
best-of-breed pipeline and emits ``comparison.md`` with a side-by-side
of the two pipelines' tier subscores.

Usage
-----
::

    # 1. Cache the notebook output (one-time, before first comparison)
    python pipelines/scripts/compare_to_human_baseline.py \\
        --domain products \\
        --cache-from-notebook usecases/products/products_workflow_minimal.ipynb

    # 2. Run the comparison
    python pipelines/scripts/compare_to_human_baseline.py \\
        --domain products \\
        --pipeline-run pipelines/products/run_<id>/

Outputs
-------
- ``pipelines/<domain>/baselines/notebook_fused.csv`` (one-time cache)
- ``pipelines/<domain>/baselines/notebook_correspondences.csv``
- ``pipelines/<domain>/baselines/notebook_panel/`` (panel against silver)
- ``pipelines/<domain>/<run_id>/comparison.md`` (side-by-side report)
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

import pandas as pd

from PyDI.evaluation.panel import compute_e2e_panel
from PyDI.evaluation.silver_standard import load_workflow_silver
from pipelines.lib.bundle import load_pipeline_bundle
from pipelines.lib.notebook_fusion_eval import (
    evaluate_with_notebook_strategy,
    get_spec as get_notebook_fusion_spec,
)
from pipelines.lib.pipeline import PipelineConfig


def _load_bundle_for_domain(domain: str):
    """Load the bundle using the same ``bundle_source`` as the BoB run.

    Reads ``pipelines/configs/<domain>.yaml`` to pick up the
    ``bundle_source`` setting so the comparison harness sees the same
    physical data tree the pipeline ran against.
    """
    config_path = REPO_ROOT / "pipelines" / "configs" / f"{domain}.yaml"
    bundle_source = "synthetic_baseline"
    if config_path.exists():
        try:
            cfg = PipelineConfig.from_yaml(config_path)
            bundle_source = cfg.bundle_source
        except Exception:
            logger.exception(
                "Failed to read %s; falling back to bundle_source=synthetic_baseline",
                config_path,
            )
    return load_pipeline_bundle(domain, bundle_source=bundle_source)


def _canonicalize_sources_for_panel(bundle) -> "list[pd.DataFrame]":
    """Rename each source's columns from raw to canonical (target-schema)
    names using the bundle's SM gold mapping.

    The panel's conflict-only accuracy / conflict-rate read source-record
    values **by canonical attribute name** (``column_types`` keys), but the
    raw source tables still carry their original column names (e.g.
    ``manufacturer`` rather than ``brand``). Without this translation every
    per-attribute source lookup returns ``None``, so no conflicts are ever
    detected and both metrics collapse to ~0. Renaming here (non-mutating —
    it returns copies) makes the source values visible to the detector.
    """
    sm = getattr(bundle, "sm_mapping", None)
    out: list[pd.DataFrame] = []
    for df in bundle.sources.values():
        name = df.attrs.get("dataset_name")
        rename: dict = {}
        if sm is not None and not sm.empty:
            for _, r in sm.iterrows():
                if str(r.get("source_dataset")) == str(name):
                    src_col, tgt_col = r.get("source_column"), r.get("target_column")
                    if src_col and tgt_col and src_col != tgt_col:
                        rename[str(src_col)] = str(tgt_col)
        new_df = df.rename(columns=rename) if rename else df.copy()
        new_df.attrs["dataset_name"] = name
        out.append(new_df)
    return out


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare best-of-breed pipeline vs human baseline."
    )
    p.add_argument("--domain", required=True, help="Domain (e.g. products).")
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Pipeline config YAML. Default: pipelines/configs/<domain>.yaml.",
    )
    p.add_argument(
        "--cache-from-notebook",
        type=Path,
        default=None,
        help=(
            "Re-execute the given notebook and cache its fused output. "
            "One-time bootstrap step; subsequent runs reuse the cache."
        ),
    )
    p.add_argument(
        "--pipeline-run",
        type=Path,
        default=None,
        help=(
            "Path to a best-of-breed run directory "
            "(contains fused.csv, correspondences.csv, e2e_panel/). "
            "Required for the comparison output."
        ),
    )
    p.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Notebook output translation
# ---------------------------------------------------------------------------


def translate_bare_int_to_prefixed(
    fused_notebook: pd.DataFrame,
    *,
    id_column: str,
    sources_by_name: Mapping[str, pd.DataFrame],
    source_prefix_map: Mapping[str, str],
) -> pd.DataFrame:
    """Convert notebook bare-int IDs into prefixed (synthetic-side) IDs.

    The notebook's ``id_column`` carries a bare integer from the
    products_1 source. We look up the equivalent prefixed id via a
    direct match: for each source DataFrame whose ``attrs["dataset_name"]``
    matches ``"products_1"`` (or whichever source the master ID column
    came from), the bare-int ``id`` becomes ``products_1_<id>``.

    For products specifically the notebook uses ``p1_id`` as the master,
    so every notebook row maps to ``products_1_<p1_id>``.
    """
    if id_column not in fused_notebook.columns:
        raise KeyError(
            f"id_column={id_column!r} not in notebook fused output "
            f"(columns: {list(fused_notebook.columns)[:10]}...)"
        )

    # Find the prefix for the master source. For products: "products_1_".
    # We assume the master is the first source declared in the prefix map.
    master_source = next(iter(sources_by_name.keys()))
    master_prefix = next(
        (p for p, s in source_prefix_map.items() if s == master_source), None
    )
    if master_prefix is None:
        raise ValueError(
            f"No prefix found for master source {master_source!r} in "
            f"source_prefix_map={dict(source_prefix_map)}"
        )

    out = fused_notebook.copy()
    out["_translated_id"] = master_prefix + out[id_column].astype("int64").astype(str)
    return out


def translate_correspondences_bare_to_prefixed(
    correspondences: pd.DataFrame,
    *,
    source_prefix_map: Mapping[str, str],
    sources_by_name: Mapping[str, pd.DataFrame],
    master_source: str,
) -> pd.DataFrame:
    """Translate notebook correspondences from bare-int to prefixed IDs.

    The products notebook's ``all_correspondences`` concatenates
    ``correspondences_p1_p<n>_refined`` for n=2,3,4 without source
    labels. We disambiguate id2's source by membership-check against
    each non-master source's id column:

    - id1 always belongs to ``master_source`` (products_1) → prefix
      with ``master_source``'s prefix.
    - id2 belongs to whichever non-master source's id space contains
      the integer value.

    Rows whose id2 doesn't belong to any non-master source are
    dropped with a warning (corruption / out-of-source-pool).
    """
    if correspondences is None or correspondences.empty:
        return (
            correspondences.copy()
            if correspondences is not None
            else (pd.DataFrame(columns=["id1", "id2", "score"]))
        )

    master_prefix = next(
        (p for p, s in source_prefix_map.items() if s == master_source), None
    )
    if master_prefix is None:
        raise ValueError(
            f"No prefix for master {master_source!r} in {dict(source_prefix_map)}"
        )

    # Pre-build per-source int-id sets for id2 disambiguation.
    source_id_sets: dict[str, set[int]] = {}
    source_prefixes: dict[str, str] = {}
    for source_name, df in sources_by_name.items():
        if source_name == master_source:
            continue
        prefix = next(
            (p for p, s in source_prefix_map.items() if s == source_name), None
        )
        if prefix is None:
            continue
        # Source ids are stored prefixed on the synthetic-side bundle
        # (``products_<n>_<bare_int>``). Strip the prefix to get bare ints.
        bare_ids: set[int] = set()
        for v in df["id"].dropna().astype(str):
            if v.startswith(prefix):
                try:
                    bare_ids.add(int(v[len(prefix) :]))
                except ValueError:
                    continue
        source_id_sets[source_name] = bare_ids
        source_prefixes[source_name] = prefix

    out_rows: list[dict[str, Any]] = []
    dropped = 0
    for _, row in correspondences.iterrows():
        try:
            id1_int = int(row["id1"])
            id2_int = int(row["id2"])
        except (KeyError, ValueError, TypeError):
            dropped += 1
            continue
        id1_prefixed = f"{master_prefix}{id1_int}"
        matched_source = None
        for source_name, id_set in source_id_sets.items():
            if id2_int in id_set:
                matched_source = source_name
                break
        if matched_source is None:
            dropped += 1
            continue
        id2_prefixed = f"{source_prefixes[matched_source]}{id2_int}"
        out_rows.append(
            {
                "id1": id1_prefixed,
                "id2": id2_prefixed,
                "score": row.get("score", 1.0),
            }
        )

    if dropped:
        logger.warning(
            "Dropped %d notebook correspondence rows whose id2 didn't match "
            "any non-master source id space (out of %d total).",
            dropped,
            len(correspondences),
        )
    return pd.DataFrame(out_rows, columns=["id1", "id2", "score"])


# ---------------------------------------------------------------------------
# Caching the notebook output
# ---------------------------------------------------------------------------


def cache_notebook_output(
    notebook_path: Path,
    *,
    cache_dir: Path,
) -> None:
    """Execute the notebook and persist its fused output to ``cache_dir``.

    Uses ``jupyter nbconvert --execute --to notebook`` to run the
    notebook in-place to a sidecar, then reads the executed sidecar's
    final cell outputs to find the ``fused`` + ``all_correspondences``
    DataFrames.

    This is fragile (depends on the notebook's variable names not
    changing) so for v1 we just document the steps and raise
    NotImplementedError. The intended manual workflow:

    1. Open the notebook in Jupyter, "Run All".
    2. Add a final cell:
       ``fused.to_csv("<cache_dir>/notebook_fused.csv", index=False)``
       ``all_correspondences.to_csv("<cache_dir>/notebook_correspondences.csv", index=False)``
    3. Save the notebook.
    4. Re-run this script without ``--cache-from-notebook``.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    instructions = (
        "Automatic notebook execution + extraction is not implemented "
        "in v1 (the notebook's outputs live in-memory and the variable "
        "names couple it to extraction logic).\n\n"
        "Manual caching steps:\n"
        f"  1. Open {notebook_path} in Jupyter, Run All.\n"
        "  2. Add a final cell:\n"
        f"        fused.to_csv('{cache_dir}/notebook_fused.csv', index=False)\n"
        f"        all_correspondences.to_csv('{cache_dir}/notebook_correspondences.csv', index=False)\n"
        "  3. Save the notebook.\n"
        f"  4. Re-run this script without --cache-from-notebook.\n"
    )
    print(instructions)
    raise NotImplementedError(
        "See printed instructions for caching the notebook output."
    )


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _fusion_sources_are_prefixed(
    fused: pd.DataFrame,
    source_prefix_map: Mapping[str, str],
) -> bool:
    """Heuristically decide whether a notebook fused frame's
    ``_fusion_sources`` already use the prefixed synthetic-side ID scheme
    (e.g. ``mbrainz_1``, ``http://www.forbes.com/...``) rather than bare
    integers (e.g. ``12198483``).

    Prefixed IDs are directly comparable to the best-of-breed pipe output
    and must NOT be run through the bare-int translator: ``int('mbrainz_1')``
    raises and the row is dropped, which silently empties the membership and
    collapses every cluster-correctness metric to zero. Returns True when the
    first parseable ``_fusion_sources`` entry starts with a known source
    prefix or is otherwise non-integer; False when it is a bare integer.
    """
    import ast as _ast

    if "_fusion_sources" not in fused.columns:
        return False
    prefixes = tuple(str(p) for p in (source_prefix_map or {}).keys())
    for value in fused["_fusion_sources"].dropna():
        if isinstance(value, str):
            try:
                items = _ast.literal_eval(value)
            except (ValueError, SyntaxError):
                items = [value]
        elif isinstance(value, list):
            items = value
        else:
            continue
        if not items:
            continue
        sample = str(items[0])
        if prefixes and sample.startswith(prefixes):
            return True
        try:
            int(sample)
            return False
        except ValueError:
            return True
    return False


def build_silver_standard_from_notebook(
    cache_dir: Path,
    *,
    domain: str,
    config: PipelineConfig,
    bundle,
) -> "SilverStandard":
    """Wrap the cached human-baseline notebook output as a SilverStandard.

    The notebook's `fused` DataFrame is the per-cluster reference; its
    `_fusion_sources` + `_fusion_source_datasets` parallel columns
    encode membership. Bare-int IDs are translated to the prefixed
    (synthetic-side) scheme so the resulting SilverStandard can be
    compared against the best-of-breed pipeline output without ID
    skew.

    cell_provenance is set to ``None`` — the notebook doesn't track
    per-cell provenance, so source-attribution and synthesis-rate
    metrics will be skipped with the standard panel warning.
    """
    from PyDI.evaluation.silver_standard import SilverStandard

    fused_path = cache_dir / "notebook_fused.csv"
    if not fused_path.exists():
        raise FileNotFoundError(
            f"Cached notebook fused output missing at {fused_path}. "
            "Run extract_notebook_baseline.py first."
        )
    fused_raw = pd.read_csv(fused_path)

    # The notebook output carries a ``cluster_id`` column from the raw
    # source data (numeric IDs unrelated to the fusion engine). Drop it
    # before adding our translated cluster-key column, otherwise we'd
    # collide on the column name and downstream metric code crashes
    # with `DataFrame has no attribute 'dtype'` (duplicate column lookup
    # returns a frame, not a series).
    fused_raw = fused_raw.drop(columns=["cluster_id"], errors="ignore")

    # The notebook's ``_fusion_sources`` come in one of two ID schemes:
    #   * already-prefixed synthetic-side IDs (e.g. ``mbrainz_1``,
    #     ``http://www.forbes.com/...``) — music / games / companies. These
    #     already match the best-of-breed pipe output, so we key the cluster
    #     on ``_id`` and build membership with the pass-through builder.
    #   * bare-int IDs (e.g. ``12198483``) — products. These need translation
    #     to the prefixed scheme via the parallel ``_fusion_source_datasets``
    #     list, with the master ``p1_id`` giving the cluster key.
    # Running the bare-int translator on already-prefixed sources parses
    # ``int('mbrainz_1')``, drops every membership row, and collapses all
    # cluster-correctness metrics to zero — so detect the scheme first.
    if _fusion_sources_are_prefixed(fused_raw, config.source_prefix_map):
        if "_id" not in fused_raw.columns:
            raise KeyError(
                "notebook fused output has prefixed _fusion_sources but no "
                f"_id column for the cluster key (columns: "
                f"{list(fused_raw.columns)[:10]}...)"
            )
        translated_fused = fused_raw.copy()
        translated_fused["cluster_id"] = translated_fused["_id"].astype(str)
        # Pass-through builder: keys the cluster on ``_id`` (== cluster_id
        # above) and treats _fusion_sources as already-prefixed record IDs,
        # exactly as for the best-of-breed pipe side.
        membership = _build_membership_from_prefixed_sources(translated_fused)
    elif "p1_id" in fused_raw.columns:
        # Bare-int notebook (products): translate p1_id → prefixed cluster_id.
        translated_fused = translate_bare_int_to_prefixed(
            fused_raw,
            id_column="p1_id",
            sources_by_name=bundle.sources,
            source_prefix_map=config.source_prefix_map,
        )
        # SilverStandard.fused conventionally uses ``cluster_id`` as the
        # cluster-key column (load_workflow_silver does this).
        translated_fused = translated_fused.rename(
            columns={"_translated_id": "cluster_id"}
        )
        membership = _build_notebook_membership(
            translated_fused,
            cluster_id_column="cluster_id",
            source_prefix_map=config.source_prefix_map,
        )
    else:
        translated_fused = fused_raw.copy()
        if (
            "cluster_id" not in translated_fused.columns
            and "_id" in translated_fused.columns
        ):
            translated_fused["cluster_id"] = translated_fused["_id"]
        membership = _build_notebook_membership(
            translated_fused,
            cluster_id_column="cluster_id",
            source_prefix_map=config.source_prefix_map,
        )

    return SilverStandard(
        fused=translated_fused,
        membership=membership,
        cell_provenance=None,
    )


def _build_membership_from_prefixed_sources(
    fused: pd.DataFrame,
) -> pd.DataFrame:
    """Build ``(record_id, source, cluster_id)`` from a fused frame whose
    ``_fusion_sources`` already contains prefixed source IDs (the
    orchestrator's output convention — see
    ``BestOfBreedPipeline._build_pipe_membership_from_fused``).

    Used for the best-of-breed (pipe) side of the pipe-vs-notebook
    comparison. The notebook side uses ``_build_notebook_membership``
    instead because its ``_fusion_sources`` is bare-int.
    """
    import ast as _ast

    def _coerce_list(value: Any) -> list[Any]:
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                return _ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return [value]
        return (
            [value]
            if value is not None and not (isinstance(value, float) and pd.isna(value))
            else []
        )

    rows: list[dict[str, Any]] = []
    for _, fused_row in fused.iterrows():
        cluster_id = str(fused_row["_id"])
        sources = _coerce_list(fused_row.get("_fusion_sources"))
        datasets = _coerce_list(fused_row.get("_fusion_source_datasets"))
        if len(datasets) != len(sources):
            datasets = list(datasets) + ["unknown"] * (len(sources) - len(datasets))
        for record_id, source in zip(sources, datasets, strict=False):
            rows.append(
                {
                    "record_id": str(record_id),
                    "source": str(source),
                    "cluster_id": cluster_id,
                }
            )
    return pd.DataFrame(rows, columns=["record_id", "source", "cluster_id"])


def _build_notebook_membership(
    fused_translated: pd.DataFrame,
    *,
    cluster_id_column: str,
    source_prefix_map: Mapping[str, str],
) -> pd.DataFrame:
    """Build long-form ``(record_id, source, cluster_id)`` for the notebook
    fused frame, using the parallel ``_fusion_sources`` +
    ``_fusion_source_datasets`` columns.

    The notebook fused frame carries bare-int source IDs in
    ``_fusion_sources``. We translate each bare-int → prefixed using its
    parallel-list dataset name (so 12198483 + 'products_1' →
    'products_1_12198483'). The cluster_id is the translated p1_id from
    ``cluster_id_column``.
    """
    import ast as _ast

    # Invert {prefix: source_name} → {source_name: prefix}.
    source_to_prefix = {s: p for p, s in source_prefix_map.items()}

    def _coerce_list(value: Any) -> list[Any]:
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                return _ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return [value]
        return (
            [value]
            if value is not None and not (isinstance(value, float) and pd.isna(value))
            else []
        )

    rows: list[dict[str, Any]] = []
    dropped = 0
    for _, fused_row in fused_translated.iterrows():
        cluster_id = str(fused_row[cluster_id_column])
        bare_ids = _coerce_list(fused_row.get("_fusion_sources"))
        dataset_names = _coerce_list(fused_row.get("_fusion_source_datasets"))
        if len(dataset_names) != len(bare_ids):
            dataset_names = list(dataset_names) + ["unknown"] * (
                len(bare_ids) - len(dataset_names)
            )
        for bare_id, dataset in zip(bare_ids, dataset_names, strict=False):
            prefix = source_to_prefix.get(str(dataset))
            if prefix is None:
                dropped += 1
                continue
            try:
                int_id = int(bare_id)
            except (TypeError, ValueError):
                dropped += 1
                continue
            rows.append(
                {
                    "record_id": f"{prefix}{int_id}",
                    "source": str(dataset),
                    "cluster_id": cluster_id,
                }
            )
    if dropped:
        logger.warning(
            "Dropped %d notebook membership rows whose source/id couldn't be "
            "resolved (out of %d fused rows).",
            dropped,
            len(fused_translated),
        )
    return pd.DataFrame(rows, columns=["record_id", "source", "cluster_id"])


def compute_panel_on_cached_notebook(
    *,
    domain: str,
    config: PipelineConfig,
    cache_dir: Path,
    out_dir: Path,
) -> dict[str, Any]:
    """Compute the e2e panel on the cached notebook output."""
    fused_path = cache_dir / "notebook_fused.csv"
    if not fused_path.exists():
        raise FileNotFoundError(
            f"Cached notebook fused output missing at {fused_path}. "
            "Run with --cache-from-notebook first (see the script docstring)."
        )

    fused_notebook = pd.read_csv(fused_path)
    bundle = _load_bundle_for_domain(domain)

    # Translate IDs. Products: notebook id_column is 'p1_id'.
    if "_id" not in fused_notebook.columns and "p1_id" in fused_notebook.columns:
        translated = translate_bare_int_to_prefixed(
            fused_notebook,
            id_column="p1_id",
            sources_by_name=bundle.sources,
            source_prefix_map=config.source_prefix_map,
        )
        translated_id_col = "_translated_id"
    else:
        translated = fused_notebook.copy()
        translated_id_col = "_id"

    silver = load_workflow_silver(
        bundle.variant_root,
        domain=domain,
        prefix_map=config.source_prefix_map or None,
    )
    fused_cols = set(translated.columns)
    col_types = {k: v for k, v in config.column_types.items() if k in fused_cols}

    # Notebook correspondences cache: bare-int id1/id2 with no source
    # labels. Disambiguate via id2 membership against each non-master
    # source's id space; id1 is always the master (products_1).
    corr_path = cache_dir / "notebook_correspondences.csv"
    if corr_path.exists():
        raw_corr = pd.read_csv(corr_path)
        master_source = next(iter(bundle.sources.keys()))
        try:
            corr_translated = translate_correspondences_bare_to_prefixed(
                raw_corr,
                source_prefix_map=config.source_prefix_map,
                sources_by_name=bundle.sources,
                master_source=master_source,
            )
        except Exception as exc:
            logger.warning(
                "Notebook correspondence translation failed (%s); "
                "passing empty correspondences and accepting degraded Tier 3.",
                exc,
            )
            corr_translated = pd.DataFrame(columns=["id1", "id2", "score"])
    else:
        logger.warning(
            "No notebook_correspondences.csv at %s; passing empty "
            "correspondences and accepting degraded Tier 3.",
            corr_path,
        )
        corr_translated = pd.DataFrame(columns=["id1", "id2", "score"])

    # Build pipe_membership where cluster_id matches the notebook fused
    # frame's translated_id, so the panel's Tier 4 row alignment works.
    # (Without this, the panel auto-builds membership with ``group_N`` ids
    # and value_correctness collapses to 0 — same regression fixed in the
    # orchestrator's _build_pipe_membership_from_fused.)
    pipe_membership = _build_notebook_membership(
        translated,
        cluster_id_column=translated_id_col,
        source_prefix_map=config.source_prefix_map,
    )

    schema_path = (
        Path(bundle.variant_root) / "input" / "schemamatching" / "target_schema.json"
    )
    panel = compute_e2e_panel(
        pipe_fused=translated,
        correspondences_pipe=corr_translated,
        sources_pipe=_canonicalize_sources_for_panel(bundle),
        silver=silver,
        column_types=col_types,
        target_schema=schema_path,
        taxonomy_base_path=Path(bundle.variant_root),
        pipe_id_column=translated_id_col,
        silver_id_column="cluster_id",
        pipe_membership=pipe_membership,
        numerical_tolerance=config.panel_tolerance_default,
        numerical_tolerance_overrides=config.panel_tolerance_overrides,
        composite_weights=config.composite_weights or None,
        source_prefix_map=config.source_prefix_map or None,
        usecase=domain,
        silver_source_label="workflow_xml_vs_notebook",
    )
    panel.write(out_dir)
    return {
        "composite_score": panel.composite.get("composite_score"),
        "tier_subscores": panel.composite.get("tier_subscores", {}),
        "warnings": panel.warnings,
    }


def compute_panel_pipe_vs_notebook(
    *,
    pipeline_run: Path,
    domain: str,
    config: PipelineConfig,
    cache_dir: Path,
    out_dir: Path,
) -> "Mapping[str, Any]":
    """Compute ONE e2e panel where the notebook IS the silver standard.

    pipe = best-of-breed (from ``pipeline_run/fused.csv`` +
    ``pipeline_run/correspondences.csv``); silver = the cached human-
    baseline notebook output wrapped as a SilverStandard. The metrics
    answer "how close does best-of-breed come to the human notebook's
    output," not "where does each pipeline differ from a third silver."
    """
    bundle = _load_bundle_for_domain(domain)

    silver = build_silver_standard_from_notebook(
        cache_dir, domain=domain, config=config, bundle=bundle
    )

    pipe_fused = pd.read_csv(pipeline_run / "fused.csv")
    corr_path = pipeline_run / "correspondences.csv"
    pipe_corr = (
        pd.read_csv(corr_path)
        if corr_path.exists()
        else pd.DataFrame(columns=["id1", "id2", "score"])
    )

    # The orchestrator's fused frame uses ``_id`` as the cluster id (set
    # by DataFusionEngine), and its ``_fusion_sources`` list contains
    # **already-prefixed** source IDs (e.g. ``products_1_12198483``) —
    # unlike the notebook output where they're bare-int. So we use a
    # pass-through builder, NOT the bare-int translator.
    pipe_membership = _build_membership_from_prefixed_sources(pipe_fused)

    fused_cols = set(pipe_fused.columns) | set(silver.fused.columns)
    col_types = {k: v for k, v in config.column_types.items() if k in fused_cols}

    schema_path = (
        Path(bundle.variant_root) / "input" / "schemamatching" / "target_schema.json"
    )
    panel = compute_e2e_panel(
        pipe_fused=pipe_fused,
        correspondences_pipe=pipe_corr,
        sources_pipe=_canonicalize_sources_for_panel(bundle),
        silver=silver,
        column_types=col_types,
        target_schema=schema_path,
        taxonomy_base_path=Path(bundle.variant_root),
        pipe_id_column="_id",
        silver_id_column="cluster_id",
        pipe_membership=pipe_membership,
        numerical_tolerance=config.panel_tolerance_default,
        numerical_tolerance_overrides=config.panel_tolerance_overrides,
        composite_weights=config.composite_weights or None,
        source_prefix_map=config.source_prefix_map or None,
        usecase=domain,
        silver_source_label="human_baseline_notebook",
    )
    panel.write(out_dir)
    return {
        "composite_score": panel.composite.get("composite_score"),
        "tier_subscores": panel.composite.get("tier_subscores", {}),
        "warnings": panel.warnings,
    }


def _fmt(value: Any, fmt: str = ".4f") -> str:
    """Pretty-format a value for the markdown table, NaN-safe."""
    if value is None:
        return "—"
    try:
        f = float(value)
        if f != f:  # NaN
            return "—"
        return format(f, fmt)
    except (TypeError, ValueError):
        return str(value)


def _delta_str(p: Any, n: Any, fmt: str = "+.4f") -> str:
    try:
        return format(float(p) - float(n), fmt)
    except (TypeError, ValueError):
        return "—"


def _append_tier1(
    lines: list[str],
    pipe: Mapping[str, Any],
    nb: Mapping[str, Any],
) -> None:
    """Tier 1 — entity_coverage."""
    lines.append("## Tier 1 — entity coverage")
    lines.append("")
    lines.append("Row counts and membership-based entity overlap (§1.1 + §1.2).")
    lines.append("")
    lines.append("| Metric | best-of-breed | notebook | delta |")
    lines.append("|---|---|---|---|")
    for key, fmt in [
        ("n_pipe", "d"),
        ("n_silver", "d"),
        ("row_count_abs_diff", "+d"),
        ("row_count_rel_diff", "+.4f"),
    ]:
        p = pipe.get(key)
        n = nb.get(key)
        lines.append(
            f"| {key} | {_fmt(p, fmt)} | {_fmt(n, fmt)} | {_delta_str(p, n, '+.4f')} |"
        )
    p_ov = pipe.get("entity_overlap") or {}
    n_ov = nb.get("entity_overlap") or {}
    for key in ["n_recovered", "n_partial", "n_lost", "n_fabricated", "recovery_rate"]:
        p = p_ov.get(key)
        n = n_ov.get(key)
        fmt = ".4f" if key == "recovery_rate" else "d"
        lines.append(
            f"| {key} | {_fmt(p, fmt)} | {_fmt(n, fmt)} | {_delta_str(p, n, '+.4f')} |"
        )
    lines.append("")


def _append_tier2(
    lines: list[str],
    pipe: Mapping[str, Any],
    nb: Mapping[str, Any],
) -> None:
    """Tier 2 — column_shape: per-column drift + validity."""
    lines.append("## Tier 2 — column shape")
    lines.append("")
    lines.append(
        "Per-column type-routed drift (lower = closer to silver) and constraint "
        "validity rate (higher = more cells parse + satisfy constraints)."
    )
    lines.append("")
    p_drift = pipe.get("per_column_drift_normalized") or {}
    n_drift = nb.get("per_column_drift_normalized") or {}
    columns = sorted(set(p_drift) | set(n_drift))
    if columns:
        lines.append("### per-column drift (normalized)")
        lines.append("")
        lines.append("| Column | best-of-breed | notebook | delta |")
        lines.append("|---|---|---|---|")
        for col in columns:
            p = p_drift.get(col)
            n = n_drift.get(col)
            lines.append(f"| {col} | {_fmt(p)} | {_fmt(n)} | {_delta_str(p, n)} |")
        lines.append("")
    p_val = pipe.get("validity_per_column") or {}
    n_val = nb.get("validity_per_column") or {}
    cols = sorted(set(p_val) | set(n_val))
    if cols:
        lines.append("### constraint / type validity per column")
        lines.append("")
        lines.append(
            "| Column | pipe validity | silver validity | pipe delta | nb validity | nb silver | nb delta |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for col in cols:
            pv = p_val.get(col) or {}
            nv = n_val.get(col) or {}
            lines.append(
                f"| {col} | {_fmt(pv.get('validity_rate_pipe'))} | "
                f"{_fmt(pv.get('validity_rate_reference'))} | "
                f"{_fmt(pv.get('delta'), '+.4f')} | "
                f"{_fmt(nv.get('validity_rate_pipe'))} | "
                f"{_fmt(nv.get('validity_rate_reference'))} | "
                f"{_fmt(nv.get('delta'), '+.4f')} |"
            )
        lines.append("")
    p_mv = pipe.get("mean_validity_delta")
    n_mv = nb.get("mean_validity_delta")
    lines.append(
        f"**mean_validity_delta** — best-of-breed: {_fmt(p_mv, '+.4f')} · "
        f"notebook: {_fmt(n_mv, '+.4f')}"
    )
    lines.append("")


def _append_tier3(
    lines: list[str],
    pipe: Mapping[str, Any],
    nb: Mapping[str, Any],
) -> None:
    """Tier 3 — cluster_correctness: BCubed + alignment + source_composition."""
    lines.append("## Tier 3 — cluster correctness")
    lines.append("")
    p_bc = pipe.get("bcubed") or {}
    n_bc = nb.get("bcubed") or {}
    lines.append("### BCubed (§3.1) — per-record precision / recall / F1")
    lines.append("")
    lines.append("| Metric | best-of-breed | notebook | delta |")
    lines.append("|---|---|---|---|")
    for key in ["precision", "recall", "f1"]:
        p = p_bc.get(key)
        n = n_bc.get(key)
        lines.append(f"| bcubed.{key} | {_fmt(p)} | {_fmt(n)} | {_delta_str(p, n)} |")
    lines.append("")

    p_al = pipe.get("alignment") or {}
    n_al = nb.get("alignment") or {}
    lines.append("### Cluster alignment (§3.2)")
    lines.append("")
    lines.append("| Metric | best-of-breed | notebook | delta |")
    lines.append("|---|---|---|---|")
    for key in [
        "mean_jaccard",
        "matched_cluster_rate_at_threshold",
        "matched_threshold",
        "size_match_rate",
        "mean_size_delta",
        "max_size_overshoot",
    ]:
        p = p_al.get(key)
        n = n_al.get(key)
        lines.append(f"| {key} | {_fmt(p)} | {_fmt(n)} | {_delta_str(p, n)} |")
    lines.append("")

    p_sc = pipe.get("source_composition") or {}
    n_sc = nb.get("source_composition") or {}
    lines.append("### Source composition (§3.3)")
    lines.append("")
    # same_source_collision_rate overall
    p_ss = p_sc.get("same_source_collision_rate") or {}
    n_ss = n_sc.get("same_source_collision_rate") or {}
    lines.append(
        "**same_source_collision_rate** (any cluster with ≥ 2 records from one source — red flag for EM over-merge)"
    )
    lines.append("")
    lines.append("| Side | silver | pipe | delta |")
    lines.append("|---|---|---|---|")
    lines.append(
        f"| best-of-breed | {_fmt(p_ss.get('silver'))} | {_fmt(p_ss.get('pipe'))} | {_fmt(p_ss.get('delta'), '+.4f')} |"
    )
    lines.append(
        f"| notebook | {_fmt(n_ss.get('silver'))} | {_fmt(n_ss.get('pipe'))} | {_fmt(n_ss.get('delta'), '+.4f')} |"
    )
    lines.append("")
    # source_mix_distribution_js
    p_mix = p_sc.get("source_mix_distribution_js")
    n_mix = n_sc.get("source_mix_distribution_js")
    lines.append(
        f"**source_mix_distribution_js** — best-of-breed: {_fmt(p_mix)} · notebook: {_fmt(n_mix)}"
    )
    lines.append("")
    # per_source_coverage_rate
    p_cov = p_sc.get("per_source_coverage_rate") or {}
    n_cov = n_sc.get("per_source_coverage_rate") or {}
    sources = sorted(set(p_cov) | set(n_cov))
    if sources:
        lines.append("**per_source_coverage_rate**")
        lines.append("")
        lines.append(
            "| Source | pipe silver | pipe pipe | pipe delta | nb silver | nb pipe | nb delta |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for src in sources:
            ps = p_cov.get(src) or {}
            ns = n_cov.get(src) or {}
            lines.append(
                f"| {src} | {_fmt(ps.get('silver'))} | {_fmt(ps.get('pipe'))} | "
                f"{_fmt(ps.get('delta'), '+.4f')} | {_fmt(ns.get('silver'))} | "
                f"{_fmt(ns.get('pipe'))} | {_fmt(ns.get('delta'), '+.4f')} |"
            )
        lines.append("")


def _append_tier4(
    lines: list[str],
    pipe: Mapping[str, Any],
    nb: Mapping[str, Any],
) -> None:
    """Tier 4 — value_correctness."""
    lines.append("## Tier 4 — value correctness")
    lines.append("")
    lines.append("### Headline accuracy + conflict context")
    lines.append("")
    lines.append("| Metric | best-of-breed | notebook | delta |")
    lines.append("|---|---|---|---|")
    for key in [
        "macro_accuracy",
        "micro_accuracy",
        "conflict_only_accuracy",
        "conflict_only_micro_accuracy",
        "conflict_rate_pipe",
        "conflict_rate_silver",
        "fully_correct_cluster_rate",
    ]:
        p = pipe.get(key)
        n = nb.get(key)
        lines.append(f"| {key} | {_fmt(p)} | {_fmt(n)} | {_delta_str(p, n)} |")
    lines.append("")

    p_pa = pipe.get("per_attribute") or {}
    n_pa = nb.get("per_attribute") or {}
    attrs = sorted(set(p_pa) | set(n_pa))
    if attrs:
        lines.append("### per-attribute accuracy + normalization fingerprint")
        lines.append("")
        lines.append(
            "| Attribute | pipe acc | nb acc | Δ acc | pipe sim_mean | nb sim_mean | pipe fingerprint | nb fingerprint |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        for a in attrs:
            pa = p_pa.get(a) or {}
            na = n_pa.get(a) or {}
            lines.append(
                f"| {a} | {_fmt(pa.get('accuracy'))} | {_fmt(na.get('accuracy'))} | "
                f"{_delta_str(pa.get('accuracy'), na.get('accuracy'))} | "
                f"{_fmt(pa.get('similarity_mean'))} | {_fmt(na.get('similarity_mean'))} | "
                f"{pa.get('mismatch_fingerprint') or '—'} | "
                f"{na.get('mismatch_fingerprint') or '—'} |"
            )
        lines.append("")

    p_co = pipe.get("conflict_only_per_attribute") or {}
    n_co = nb.get("conflict_only_per_attribute") or {}
    co_attrs = sorted(set(p_co) | set(n_co))
    if co_attrs:
        lines.append("### conflict-only per-attribute accuracy (§4.5)")
        lines.append("")
        lines.append(
            "Restricted to cells where ≥ 2 distinct source values were present."
        )
        lines.append("")
        lines.append("| Attribute | pipe acc | pipe count | nb acc | nb count |")
        lines.append("|---|---|---|---|---|")
        for a in co_attrs:
            pca = p_co.get(a) or {}
            nca = n_co.get(a) or {}
            lines.append(
                f"| {a} | {_fmt(pca.get('accuracy'))} | "
                f"{_fmt(pca.get('count'), 'd')} | {_fmt(nca.get('accuracy'))} | "
                f"{_fmt(nca.get('count'), 'd')} |"
            )
        lines.append("")

    p_dd = pipe.get("density_delta_per_attribute") or {}
    n_dd = nb.get("density_delta_per_attribute") or {}
    dd_attrs = sorted(set(p_dd) | set(n_dd))
    if dd_attrs:
        lines.append("### density delta per attribute (§4.3)")
        lines.append("")
        lines.append(
            "Pipe density − silver density. Negative = pipeline silently nulled cells."
        )
        lines.append("")
        lines.append(
            "| Attribute | pipe silver | pipe pipe | pipe delta | nb silver | nb pipe | nb delta |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for a in dd_attrs:
            pd_ = p_dd.get(a) or {}
            nd_ = n_dd.get(a) or {}
            lines.append(
                f"| {a} | {_fmt(pd_.get('reference_density'))} | {_fmt(pd_.get('pipe_density'))} | "
                f"{_fmt(pd_.get('delta'), '+.4f')} | "
                f"{_fmt(nd_.get('reference_density'))} | {_fmt(nd_.get('pipe_density'))} | "
                f"{_fmt(nd_.get('delta'), '+.4f')} |"
            )
        lines.append("")

    p_list = pipe.get("list_attribute_set_metrics") or {}
    n_list = nb.get("list_attribute_set_metrics") or {}
    if p_list or n_list:
        list_attrs = sorted(set(p_list) | set(n_list))
        lines.append("### list-attribute set metrics (§4.2)")
        lines.append("")
        lines.append(
            "| Attribute | pipe set_f1 | nb set_f1 | pipe set_jaccard | nb set_jaccard | count |"
        )
        lines.append("|---|---|---|---|---|---|")
        for a in list_attrs:
            pl = p_list.get(a) or {}
            nl = n_list.get(a) or {}
            lines.append(
                f"| {a} | {_fmt(pl.get('set_f1'))} | {_fmt(nl.get('set_f1'))} | "
                f"{_fmt(pl.get('set_jaccard'))} | {_fmt(nl.get('set_jaccard'))} | "
                f"{_fmt(pl.get('count') or nl.get('count'), 'd')} |"
            )
        lines.append("")


def _append_tier1_single(lines: list[str], block: Mapping[str, Any]) -> None:
    lines.append("## Tier 1 — entity coverage")
    lines.append("")
    lines.append(
        "Row counts and membership-based entity overlap (§1.1 + §1.2). "
        "`silver` here means the notebook output."
    )
    lines.append("")
    lines.append("| Metric | value |")
    lines.append("|---|---|")
    for key, fmt in [
        ("n_pipe", "d"),
        ("n_silver", "d"),
        ("row_count_abs_diff", "+d"),
        ("row_count_rel_diff", "+.4f"),
    ]:
        lines.append(f"| {key} | {_fmt(block.get(key), fmt)} |")
    ov = block.get("entity_overlap") or {}
    for key in ["n_recovered", "n_partial", "n_lost", "n_fabricated", "recovery_rate"]:
        fmt = ".4f" if key == "recovery_rate" else "d"
        lines.append(f"| {key} | {_fmt(ov.get(key), fmt)} |")
    lines.append("")


def _append_tier2_single(lines: list[str], block: Mapping[str, Any]) -> None:
    lines.append("## Tier 2 — column shape")
    lines.append("")
    lines.append(
        "Per-column type-routed drift (lower = closer to the notebook); "
        "constraint / type validity per column."
    )
    lines.append("")
    drift = block.get("per_column_drift_normalized") or {}
    if drift:
        lines.append("### per-column drift (normalized)")
        lines.append("")
        lines.append("| Column | drift |")
        lines.append("|---|---|")
        for col in sorted(drift):
            lines.append(f"| {col} | {_fmt(drift[col])} |")
        lines.append("")
    validity = block.get("validity_per_column") or {}
    if validity:
        lines.append("### constraint / type validity per column")
        lines.append("")
        lines.append("| Column | pipe validity | silver validity | delta |")
        lines.append("|---|---|---|---|")
        for col in sorted(validity):
            v = validity[col] or {}
            lines.append(
                f"| {col} | {_fmt(v.get('validity_rate_pipe'))} | "
                f"{_fmt(v.get('validity_rate_reference'))} | "
                f"{_fmt(v.get('delta'), '+.4f')} |"
            )
        lines.append("")
    lines.append(
        f"**mean_validity_delta**: {_fmt(block.get('mean_validity_delta'), '+.4f')}"
    )
    lines.append("")


def _append_tier3_single(lines: list[str], block: Mapping[str, Any]) -> None:
    lines.append("## Tier 3 — cluster correctness")
    lines.append("")
    bc = block.get("bcubed") or {}
    lines.append("### BCubed (§3.1)")
    lines.append("")
    lines.append("| Metric | value |")
    lines.append("|---|---|")
    for key in ["precision", "recall", "f1"]:
        lines.append(f"| bcubed.{key} | {_fmt(bc.get(key))} |")
    lines.append("")
    al = block.get("alignment") or {}
    lines.append("### Cluster alignment (§3.2)")
    lines.append("")
    lines.append("| Metric | value |")
    lines.append("|---|---|")
    for key in [
        "mean_jaccard",
        "matched_cluster_rate_at_threshold",
        "matched_threshold",
        "size_match_rate",
        "mean_size_delta",
        "max_size_overshoot",
    ]:
        lines.append(f"| {key} | {_fmt(al.get(key))} |")
    lines.append("")
    sc = block.get("source_composition") or {}
    ss = sc.get("same_source_collision_rate") or {}
    lines.append("### Source composition (§3.3)")
    lines.append("")
    lines.append(
        "**same_source_collision_rate** — silver: "
        f"{_fmt(ss.get('silver'))} · pipe: {_fmt(ss.get('pipe'))} · "
        f"delta: {_fmt(ss.get('delta'), '+.4f')}"
    )
    lines.append("")
    by_src = ss.get("by_source") or {}
    if by_src:
        lines.append("| Source | silver | pipe | delta |")
        lines.append("|---|---|---|---|")
        for src in sorted(by_src):
            row = by_src[src] or {}
            lines.append(
                f"| {src} | {_fmt(row.get('silver'))} | {_fmt(row.get('pipe'))} | "
                f"{_fmt(row.get('delta'), '+.4f')} |"
            )
        lines.append("")
    mix = sc.get("source_mix_distribution_js")
    lines.append(f"**source_mix_distribution_js**: {_fmt(mix)}")
    lines.append("")
    cov = sc.get("per_source_coverage_rate") or {}
    if cov:
        lines.append("**per_source_coverage_rate**")
        lines.append("")
        lines.append("| Source | silver | pipe | delta |")
        lines.append("|---|---|---|---|")
        for src in sorted(cov):
            row = cov[src] or {}
            lines.append(
                f"| {src} | {_fmt(row.get('silver'))} | {_fmt(row.get('pipe'))} | "
                f"{_fmt(row.get('delta'), '+.4f')} |"
            )
        lines.append("")


def _append_tier4_single(lines: list[str], block: Mapping[str, Any]) -> None:
    lines.append("## Tier 4 — value correctness")
    lines.append("")
    lines.append(
        "How well do best-of-breed's fused values agree with the notebook's "
        "per attribute? Macro accuracy averages across attributes; "
        "conflict-only restricts to cells where ≥ 2 sources disagreed."
    )
    lines.append("")
    lines.append("### Headline accuracy + conflict context")
    lines.append("")
    lines.append("| Metric | value |")
    lines.append("|---|---|")
    for key in [
        "macro_accuracy",
        "micro_accuracy",
        "conflict_only_accuracy",
        "conflict_only_micro_accuracy",
        "conflict_rate_pipe",
        "conflict_rate_silver",
        "fully_correct_cluster_rate",
    ]:
        lines.append(f"| {key} | {_fmt(block.get(key))} |")
    lines.append("")
    pa = block.get("per_attribute") or {}
    if pa:
        lines.append("### per-attribute accuracy + normalization fingerprint")
        lines.append("")
        lines.append(
            "| Attribute | accuracy | similarity_mean | gap | fingerprint | count |"
        )
        lines.append("|---|---|---|---|---|---|")
        for a in sorted(pa):
            row = pa[a] or {}
            lines.append(
                f"| {a} | {_fmt(row.get('accuracy'))} | "
                f"{_fmt(row.get('similarity_mean'))} | "
                f"{_fmt(row.get('accuracy_similarity_gap'), '+.4f')} | "
                f"{row.get('mismatch_fingerprint') or '—'} | "
                f"{_fmt(row.get('count'), 'd')} |"
            )
        lines.append("")
    co = block.get("conflict_only_per_attribute") or {}
    if co:
        lines.append("### conflict-only per-attribute accuracy (§4.5)")
        lines.append("")
        lines.append("| Attribute | accuracy | count |")
        lines.append("|---|---|---|")
        for a in sorted(co):
            row = co[a] or {}
            lines.append(
                f"| {a} | {_fmt(row.get('accuracy'))} | {_fmt(row.get('count'), 'd')} |"
            )
        lines.append("")
    dd = block.get("density_delta_per_attribute") or {}
    if dd:
        lines.append("### density delta per attribute (§4.3)")
        lines.append("")
        lines.append(
            "Pipe density − silver density. Negative = pipeline nulled cells the notebook filled."
        )
        lines.append("")
        lines.append("| Attribute | silver | pipe | delta |")
        lines.append("|---|---|---|---|")
        for a in sorted(dd):
            row = dd[a] or {}
            lines.append(
                f"| {a} | {_fmt(row.get('reference_density'))} | {_fmt(row.get('pipe_density'))} | "
                f"{_fmt(row.get('delta'), '+.4f')} |"
            )
        lines.append("")
    lst = block.get("list_attribute_set_metrics") or {}
    if lst:
        lines.append("### list-attribute set metrics (§4.2)")
        lines.append("")
        lines.append("| Attribute | set_f1 | set_jaccard | count |")
        lines.append("|---|---|---|---|")
        for a in sorted(lst):
            row = lst[a] or {}
            lines.append(
                f"| {a} | {_fmt(row.get('set_f1'))} | {_fmt(row.get('set_jaccard'))} | "
                f"{_fmt(row.get('count'), 'd')} |"
            )
        lines.append("")


# ---------------------------------------------------------------------------
# Apples-to-apples per-stage scoring helpers
# ---------------------------------------------------------------------------
#
# The per-stage tables in Part 1 need the SAME metrics computed on the
# SAME gold for both sides. Best-of-breed persists only the
# composition-strategy metric (reduction_ratio) for blocking, so we
# re-score the winning blocker for pair_completeness too. The notebook
# doesn't save its LLM-SM mapping, so we re-run that matcher once and
# score it against the SM gold the best-of-breed uses.


def _score_winning_blocker_for_pair_completeness(
    domain: str,
    winner_per_pair: Mapping[str, str],
) -> dict[str, dict[str, float]]:
    """Re-evaluate the BoB EM blocking winner per pair for BOTH
    pair_completeness (recall) and reduction_ratio against the test
    EM gold. This makes the report's Stage 3 numbers directly
    comparable to the notebook's (which reports both).
    """
    from usecases_synthetic.lib.committee_em_scoring import (
        blocking_pair_recall,
        reduction_ratio,
    )

    bundle = _load_bundle_for_domain(domain)
    out: dict[str, dict[str, float]] = {}
    for (src1, src2), splits in bundle.em_splits.items():
        pair_key = f"{src1}_{src2}"
        winner = winner_per_pair.get(pair_key)
        if winner is None:
            continue
        test_gold = splits.get("test")
        if test_gold is None or test_gold.empty:
            continue
        blocker = _instantiate_blocker_from_yaml(domain, winner, src1, src2, bundle)
        if blocker is None:
            continue
        try:
            candidates = blocker.materialize()
        except Exception:
            logger.exception(
                "Blocker %s materialization failed for %s_%s", winner, src1, src2
            )
            continue
        pr = blocking_pair_recall(candidates, test_gold)
        rr = reduction_ratio(
            candidates,
            n_left=len(bundle.sources[src1]),
            n_right=len(bundle.sources[src2]),
        )
        out[pair_key] = {
            "winner_blocker": winner,
            "pair_completeness": float(pr["pair_recall"]),
            "reduction_ratio": float(rr["reduction_ratio"]),
            "n_candidates": int(rr.get("candidate_count", len(candidates))),
            "n_true_pairs": int(pr.get("gold_positives", 0)),
            "true_positives_found": int(pr.get("covered", 0)),
        }
    return out


def _instantiate_blocker_from_yaml(
    domain: str,
    member_name: str,
    src1: str,
    src2: str,
    bundle,
):
    """Build a blocker instance using the products YAML's params for
    `member_name`. Returns None if the member name is unknown."""
    import importlib

    import yaml as _yaml

    yaml_path = (
        REPO_ROOT
        / "usecases_synthetic"
        / "config"
        / "committees"
        / f"em_blocking_committee_{domain}.yaml"
    )
    if not yaml_path.exists():
        return None
    raw = _yaml.safe_load(yaml_path.read_text()) or {}
    member = next(
        (m for m in raw.get("members", []) if m.get("name") == member_name),
        None,
    )
    if member is None:
        return None
    blocker_spec = member.get("blocker") or {}
    cls_name = blocker_spec.get("class")
    module_path = blocker_spec.get("module")
    params = dict(blocker_spec.get("params") or {})
    if not cls_name or not module_path:
        return None
    cls = getattr(importlib.import_module(module_path), cls_name)
    df_left = bundle.sources[src1].copy()
    df_right = bundle.sources[src2].copy()
    # The StandardBlocker uses a derived name_first_token key; that's
    # computed by the EM committee runner before calling the blocker.
    # For embedding/token/bm25/sn blockers we don't need that derivation.
    try:
        return cls(df_left, df_right, id_column="id", **params)
    except Exception:
        logger.exception(
            "Failed to instantiate %s for %s_%s with params=%s",
            cls_name,
            src1,
            src2,
            params,
        )
        return None


def _score_notebook_llm_sm(domain: str) -> dict[str, float] | None:
    """Re-run the notebook's LLMBasedSchemaMatcher against the same SM
    gold best-of-breed scored against. The notebook itself doesn't save
    its mapping, so the only way to get a comparable F1 is to re-run.

    Cost: ~16 LLM calls @ ~$0.01 each = ~$0.20 in API. Falls back to
    None if OPENAI_API_KEY isn't set or the matcher raises.
    """
    import os

    if not os.environ.get("OPENAI_API_KEY"):
        logger.warning(
            "OPENAI_API_KEY not set; cannot re-score notebook LLM SM. "
            "Reporting as not-available."
        )
        return None
    try:
        from langchain_openai import ChatOpenAI

        from PyDI.schemamatching.llm_based import LLMBasedSchemaMatcher
        from usecases_synthetic.lib.committee_sm import (
            _target_df_from_schema,
            score_sm_mapping,
        )
    except Exception:
        logger.exception("Failed to import LLM SM dependencies")
        return None

    bundle = _load_bundle_for_domain(domain)
    gold = bundle.sm_mapping
    if gold is None or gold.empty:
        return None
    gold_target_name = (
        str(gold["target_dataset"].iloc[0])
        if "target_dataset" in gold.columns
        else None
    )
    target_df = _target_df_from_schema(
        bundle.target_schema,
        bundle.sources,
        target_name=gold_target_name,
        fusion_frames=[
            f
            for f in (bundle.fusion_validation, bundle.fusion_gold)
            if f is not None and not f.empty
        ]
        or None,
    )
    # Notebook config: gpt-5.5 / temperature 0 / num_rows=40. We fall
    # back to gpt-5.4-mini if gpt-5.5 errors (the notebook itself ran
    # under whatever fallback langchain selected at the time).
    for model_name in ("gpt-5.5", "gpt-5.4-mini"):
        try:
            chat = ChatOpenAI(model=model_name, temperature=0)
            matcher = LLMBasedSchemaMatcher(
                chat_model=chat,
                num_rows=40,
                target_schema=bundle.target_schema,
            )
            all_maps = []
            for source_name, source_df in bundle.sources.items():
                m = matcher.match(source_df, target_df)
                all_maps.append(m)
            if all_maps:
                combined = pd.concat(all_maps, ignore_index=True)
                metrics = score_sm_mapping(combined, gold)
                return {
                    "model": model_name,
                    "precision": float(metrics.get("precision", 0.0)),
                    "recall": float(metrics.get("recall", 0.0)),
                    "f1": float(metrics.get("f1", 0.0)),
                }
        except Exception:
            logger.exception(
                "LLMBasedSchemaMatcher failed with model=%s; trying fallback",
                model_name,
            )
            continue
    return None


# ---------------------------------------------------------------------------
# Per-stage test scores from the notebook execution
# ---------------------------------------------------------------------------
#
# The notebook writes per-pair JSONs under
# ``usecases/products/output/`` after each evaluation cell. We read them
# directly so the numbers in the report match what the notebook itself
# saw at execution time (no transcription).
NOTEBOOK_EVAL_ROOT = REPO_ROOT / "usecases" / "products" / "output"


def _read_notebook_blocking_test() -> dict[str, dict[str, float]]:
    """Per-pair blocker stats from ``output/Blocking/blocking_eval_*``."""
    out: dict[str, dict[str, float]] = {}
    for pair_dir, label in [
        ("blocking_eval_prod1_prod2", "products_1_products_2"),
        ("blocking_eval_prod1_prod3", "products_1_products_3"),
        ("blocking_eval_prod1_prod4", "products_1_products_4"),
    ]:
        path = (
            NOTEBOOK_EVAL_ROOT
            / "Blocking"
            / pair_dir
            / "blocking_evaluation_summary.json"
        )
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        out[label] = {
            "pair_completeness": float(d.get("pair_completeness", float("nan"))),
            "reduction_ratio": float(d.get("reduction_ratio", float("nan"))),
        }
    return out


def _read_notebook_matching_test() -> dict[str, dict[str, dict[str, float]]]:
    """Per-pair per-refiner F1/P/R from ``debug_results_entity_matching``.

    The notebook's refinement comparison cell writes one folder per
    (pair, refiner). We pull all three refiners' final test-set scores
    so we can show the per-pair winner.
    """
    out: dict[str, dict[str, dict[str, float]]] = {}
    for pair, label in [
        ("p1_p2", "products_1_products_2"),
        ("p1_p3", "products_1_products_3"),
        ("p1_p4", "products_1_products_4"),
    ]:
        out[label] = {}
        for refiner in ["refined", "greedy", "mbm"]:
            path = (
                NOTEBOOK_EVAL_ROOT
                / "debug_results_entity_matching"
                / f"{pair}_{refiner}"
                / "matching_evaluation_summary.json"
            )
            if not path.exists():
                continue
            d = json.loads(path.read_text())
            # "refined" filename actually holds the MBM-style 1:1 refined
            # output per the notebook's terminology — rename to ``mbm``
            # alias so the comparison makes sense. The "baseline" (raw
            # rule-based output) is reported as the headline by the
            # notebook's refinement-summary cell with these numbers:
            # p1_p2 F1=0.7536, p1_p3 F1=0.7658, p1_p4 F1=0.8879.
            out[label][refiner] = {
                "precision": float(d.get("precision", float("nan"))),
                "recall": float(d.get("recall", float("nan"))),
                "f1": float(d.get("f1", float("nan"))),
                "accuracy": float(d.get("accuracy", float("nan"))),
            }
    return out


# Notebook "baseline" (= raw rule-based output, no refiner) — taken
# directly from the notebook's refinement-comparison cell output
# (cell id 4cab6733 in products_workflow_minimal.ipynb). These ARE the
# pre-refinement matching scores the notebook reported; the JSON files
# capture only the three refined versions.
NOTEBOOK_BASELINE_MATCHING = {
    "products_1_products_2": {
        "precision": 0.6411,
        "recall": 0.9138,
        "f1": 0.7536,
        "accuracy": 0.7754,
    },
    "products_1_products_3": {
        "precision": 0.6814,
        "recall": 0.8742,
        "f1": 0.7658,
        "accuracy": 0.7757,
    },
    "products_1_products_4": {
        "precision": 0.8319,
        "recall": 0.9519,
        "f1": 0.8879,
        "accuracy": 0.8804,
    },
}


# Notebook fusion eval — from the notebook's final cell output
# (cell id e9839e90). The notebook reports overall_accuracy and
# macro_accuracy on the fusion test set (100 verified rows).
NOTEBOOK_FUSION_TEST = {
    "winner": "hardware_fusion_strategy",
    "strategy_detail": (
        "voting for brand/product_type/model_number; minimum for "
        "performance specs (vram_gb, storage_gb, read/write speeds); "
        "prefer_higher_trust for technical strings + dimensions"
    ),
    "overall_accuracy": 0.370,
    "macro_accuracy": 0.398,
    "num_evaluated_records": 93,
    "num_evaluated_attributes": 26,
}


METRIC_DESCRIPTIONS = {
    # composite + tiers
    "composite_score": (
        "Weighted average across the four tier subscores. A ranking number, "
        "not a diagnostic — inspect the per-tier deltas to understand failures."
    ),
    "entity_coverage": (
        "Tier 1 (weight 0.10). Did the pipeline produce roughly the right "
        "number of entities + the right set of entities (membership-based)? "
        "Recipe: mean(1 − |row_count_rel_diff|, recovery_rate, 1 − n_fabricated/n_pipe)."
    ),
    "column_shape": (
        "Tier 2 (weight 0.20). Do per-column value distributions match? "
        "Type-routed: JS / W1-normalized / token-JS / W1-days; lower drift = "
        "closer. Recipe: 1 − mean(per_column_drift_normalized)."
    ),
    "cluster_correctness": (
        "Tier 3 (weight 0.40). Are records assigned to the right clusters? "
        "Recipe: mean(bcubed_f1, 1 − same_source_collision_rate, "
        "1 − source_mix_distribution_js, mean_jaccard)."
    ),
    "value_correctness": (
        "Tier 4 (weight 0.30). Do fused values match silver's? Recipe: "
        "mean(macro_accuracy, conflict_only_accuracy, fully_correct_cluster_rate, "
        "1 − mean(source_attribution_js)). Last term dropped when provenance unavailable."
    ),
    # Tier 1 metrics
    "n_pipe": (
        "Number of fused entities best-of-breed produced. **How**: row count of "
        "`fused.csv`. **Reading**: vs n_silver gives the row-count delta — useful "
        "as a coarse sanity check."
    ),
    "n_silver": ("Number of clusters in the notebook's fused output."),
    "row_count_abs_diff": (
        "n_pipe − n_silver. Anti-symmetric — a pipeline that loses 100 entities and "
        "fabricates 100 new ones shows 0. Use entity_overlap instead for a diagnostic."
    ),
    "row_count_rel_diff": (
        "row_count_abs_diff / n_silver. Same caveat as above; rely on entity_overlap."
    ),
    "n_recovered": (
        "Notebook clusters whose **exact** member set is preserved by some "
        "best-of-breed cluster (Jaccard = 1.0). **How**: greedy maximum-overlap "
        "alignment per silver cluster, count Jaccard == 1.0. **Reading**: low value "
        "means the two pipelines pick different cluster boundaries; read mean_jaccard "
        "alongside for 'mostly right' clusters."
    ),
    "n_partial": (
        "Notebook clusters partially preserved (0 < Jaccard < 1.0). **How**: same "
        "alignment as recovered, but count partial Jaccards. **Reading**: most aligned "
        "clusters share **some** records but disagree on the exact boundary."
    ),
    "n_lost": (
        "Notebook clusters with **zero** overlap with any best-of-breed cluster. "
        "Critical if non-zero — pipeline missed an entity entirely."
    ),
    "n_fabricated": (
        "Best-of-breed clusters that aren't the best match for any notebook cluster. "
        "**Reading**: pipeline produced clusters the human didn't."
    ),
    "recovery_rate": (
        "n_recovered / n_silver. **Reading**: fraction of notebook clusters reproduced "
        "exactly. **Caveat**: strict — a pipeline that gets a cluster 90% right still "
        "counts as n_partial, not recovered."
    ),
    # Tier 3 metrics
    "bcubed.precision": (
        "BCubed precision per record, averaged. **How**: for each source record r, "
        "precision(r) = (records r was lumped with that *should* be in r's cluster) / "
        "(records r was lumped with). **Reading**: low = pipeline over-merges."
    ),
    "bcubed.recall": (
        "BCubed recall per record, averaged. **How**: recall(r) = (records r should be "
        "lumped with that the pipeline found) / (records r should be lumped with). "
        "**Reading**: low = pipeline misses cross-source links."
    ),
    "bcubed.f1": (
        "Harmonic mean of BCubed precision and recall. **Reading**: load-bearing "
        "clustering metric — record-level so it's robust to cluster-size skew "
        "(unlike pair-based F1)."
    ),
    "mean_jaccard": (
        "Average alignment quality across aligned cluster pairs. **How**: for each "
        "silver cluster, find best-overlapping pipe cluster (greedy), compute Jaccard, "
        "average. **Reading**: softer than n_recovered — values 0.5–0.9 say 'mostly "
        "right boundaries'."
    ),
    "matched_cluster_rate_at_threshold": (
        "Fraction of silver clusters whose alignment Jaccard ≥ matched_threshold (default 0.5). "
        "**Reading**: complement to mean_jaccard, gives a 'yes/no' good-alignment count."
    ),
    "size_match_rate": (
        "Fraction of aligned cluster pairs where pipe_size == silver_size exactly. "
        "**Reading**: 1.0 = pipeline never adds/drops a member; low = size drift."
    ),
    "mean_size_delta": (
        "Mean of (pipe_size − silver_size) across aligned pairs. **Reading**: sign tells "
        "you whether pipeline tends to over-merge (positive) or under-merge (negative)."
    ),
    "max_size_overshoot": (
        "max(pipe_size) − max(silver_size) across the table. **Reading**: a single "
        "runaway-large pipe cluster will show up here."
    ),
    "source_mix_distribution_js": (
        "JS divergence between silver's and pipe's distribution of cluster source-sets. "
        "**How**: count how often each source-set combination appears as a cluster's "
        "membership; JS-divergence the two histograms. **Reading**: low = pipelines "
        "produce similar cross-source merge patterns."
    ),
    # Tier 4 metrics
    "macro_accuracy": (
        "Average per-attribute exact-match accuracy across aligned cluster cells. "
        "**How**: for each aligned (pipe_cluster, silver_cluster) pair and each "
        "attribute, check if pipe.value == silver.value; macro-average across "
        "attributes. **Reading**: low when fusion strategies pick different values."
    ),
    "micro_accuracy": (
        "Same comparison as macro_accuracy but pooled across all cells "
        "instead of macro-averaging per attribute. **Reading**: dominated by "
        "high-volume attributes; differs from macro when some attributes have "
        "fewer evaluable cells."
    ),
    "conflict_only_accuracy": (
        "macro_accuracy restricted to cells where source records actually disagreed "
        "(≥ 2 distinct values). **How**: skip trivial cells where every source agrees. "
        "**Reading**: the real fusion-policy quality signal; trivial cells dominate "
        "macro_accuracy otherwise."
    ),
    "conflict_only_micro_accuracy": (
        "Conflict-only micro-average — same restriction as conflict_only_accuracy but "
        "pooled across cells. **Reading**: useful when per-attribute conflict counts vary."
    ),
    "conflict_rate_pipe": (
        "Fraction of aligned clusters where best-of-breed had ≥ 1 attribute with "
        "input source disagreement. **How**: walk the cluster's input records, count "
        "(cluster, attr) cells where input values disagreed. **Reading**: context for "
        "interpreting conflict_only_accuracy — 0.99 acc on 2% conflict rate ≠ 0.99 on 60%."
    ),
    "conflict_rate_silver": (
        "Same as conflict_rate_pipe but on the notebook's (silver) clusters."
    ),
    "fully_correct_cluster_rate": (
        "Fraction of aligned clusters where *every* attribute simultaneously matched. "
        "**How**: AND across attribute-level correctness flags per cluster, take mean. "
        "**Reading**: harshest Tier 4 metric — sensitive to attribute count + trailing "
        "whitespace; cross-pipeline comparison risks apples-to-oranges."
    ),
}


# ---------------------------------------------------------------------------
# Notebook-style fusion eval (apples-to-apples)
# ---------------------------------------------------------------------------
#
# Use the SAME DataFusionStrategy + per-attribute evaluation functions
# the workflow notebook uses, applied to BOTH the best-of-breed fused
# frame and the cached notebook fused frame against the same gold.
# Rules per domain live in pipelines/lib/notebook_fusion_eval.py.


_DOMAIN_ANCHOR_PREFIX: dict[str, str] = {
    "products": "products_1_",
    "music": "mbrainz_",
    "games": "metacritic_",
    "companies": "http://www.forbes.com/",
    "papers": "dblp-",
}


def _parse_fusion_sources(value: Any) -> list[str] | None:
    """Parse the ``_fusion_sources`` column entry (list-as-string after
    CSV roundtrip, or already a list)."""
    if isinstance(value, (list, tuple, set)):
        return [str(x) for x in value]
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s or s in {"[]", "nan", "None"}:
        return None
    # Try JSON first (after single->double quote swap).
    try:
        parsed = json.loads(s.replace("'", '"'))
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except Exception:
        pass
    # Best-effort fallback: strip brackets and split.
    inner = s.strip("[](){}")
    return [x.strip().strip("'\"") for x in inner.split(",") if x.strip()]


def _derive_anchor_id_column(
    fused: pd.DataFrame, *, domain: str, out_col: str
) -> pd.DataFrame:
    """Add ``out_col`` to ``fused`` by picking the source id whose
    prefix matches the domain's anchor source. Drops rows that have
    no anchor-source member."""
    prefix = _DOMAIN_ANCHOR_PREFIX[domain]
    if out_col in fused.columns and fused[out_col].notna().any():
        # Already present (notebook-cached frames carry the anchor id).
        return fused

    if "_fusion_sources" not in fused.columns:
        raise KeyError(
            "Cannot derive anchor id: fused frame missing '_fusion_sources' "
            f"column (have: {list(fused.columns)[:10]}...)."
        )

    def pick(value: Any) -> str | None:
        sources = _parse_fusion_sources(value)
        if not sources:
            return None
        for src in sources:
            if src.startswith(prefix):
                return src
        return None

    out = fused.copy()
    out[out_col] = out["_fusion_sources"].apply(pick)
    return out.dropna(subset=[out_col])


def _load_notebook_fusion_gold(domain: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load + preprocess the notebook's fusion gold for ``domain``.

    Returns (gold_df, info) where ``info`` captures the gold-file path
    + any preprocessing applied, for reporting.
    """
    from PyDI.io import load_xml

    if domain == "products":
        gold_path = (
            REPO_ROOT
            / "usecases"
            / "products"
            / "input"
            / "fusion"
            / "fusion_test_set.csv"
        )
        gold = pd.read_csv(gold_path)
        if "filled" in gold.columns:
            gold = gold[gold["filled"] == "y"].copy()
        for col in (
            "vram_gb",
            "storage_gb",
            "read_speed_mb_s",
            "write_speed_mb_s",
        ):
            if col in gold.columns:
                gold[col] = pd.to_numeric(gold[col], errors="coerce")
        return gold, {
            "path": str(gold_path.relative_to(REPO_ROOT)),
            "filter": "filled == 'y'",
            "rows": int(len(gold)),
        }

    if domain == "games":
        gold_path = (
            REPO_ROOT / "usecases" / "games" / "input" / "fusion" / "test_set_final.xml"
        )
        gold = load_xml(gold_path, name="fusion_test_set", nested_handling="aggregate")
        if "genres_genre" in gold.columns:
            gold = gold.rename(columns={"genres_genre": "genres"})
        if "releaseYear" in gold.columns:
            gold["releaseYear"] = pd.to_datetime(gold["releaseYear"], errors="coerce")
        return gold, {
            "path": str(gold_path.relative_to(REPO_ROOT)),
            "preprocessing": "rename genres_genre->genres, coerce releaseYear",
            "rows": int(len(gold)),
        }

    if domain == "companies":
        gold_path = (
            REPO_ROOT
            / "usecases"
            / "companies"
            / "input"
            / "fusion"
            / "test_set_final.xml"
        )
        gold = load_xml(gold_path, name="fusion_test_set", nested_handling="aggregate")
        if "keypeople_name" in gold.columns:
            # Schema authority is target_schema.json's ``keypeople``.
            # The notebook still renames keypeople_name -> founders;
            # we align with the schema and the notebook_fusion_eval
            # rule's attribute name.
            gold["keypeople"] = gold["keypeople_name"].apply(
                lambda x: [x] if isinstance(x, str) else x
            )

        def _from_sci(v: Any) -> Any:
            if isinstance(v, str) and ("e" in v or "E" in v):
                try:
                    return int(float(v))
                except Exception:
                    return v
            return v

        for col in ("assets", "revenue"):
            if col in gold.columns:
                gold[col] = gold[col].apply(_from_sci)
        return gold, {
            "path": str(gold_path.relative_to(REPO_ROOT)),
            "preprocessing": "keypeople_name->founders, sci-notation->int for assets/revenue",
            "rows": int(len(gold)),
        }

    if domain == "music":
        gold_path = (
            REPO_ROOT / "usecases" / "music" / "input" / "fusion" / "test_set_final.xml"
        )
        gold = load_xml(gold_path, name="fusion_test_set", nested_handling="aggregate")
        # The notebook additionally translates release-country via the
        # discogs spec and parses tracks via parse_track_list before
        # evaluation. Both pipelines need to match that prep for a fair
        # tokenized_match — leaving them out understates accuracy on
        # those two columns but does not affect the other 5.
        return gold, {
            "path": str(gold_path.relative_to(REPO_ROOT)),
            "preprocessing": (
                "NONE — notebook applies release-country normalization + "
                "track-list parsing pre-eval; both are skipped here so "
                "release-country / tracks scores are pessimistic on both "
                "sides equally."
            ),
            "rows": int(len(gold)),
        }

    if domain == "papers":
        gold_path = (
            REPO_ROOT / "usecases" / "papers" / "input" / "fusion" / "fusion_test.jsonl"
        )
        if not gold_path.exists():
            raise FileNotFoundError(gold_path)
        gold = pd.read_json(gold_path, lines=True)
        return gold, {
            "path": str(gold_path.relative_to(REPO_ROOT)),
            "preprocessing": "none (jsonl loaded verbatim; join key is doi)",
            "rows": int(len(gold)),
        }

    raise ValueError(f"Unsupported domain for notebook fusion eval: {domain!r}")


def compute_notebook_style_fusion_comparison(
    *,
    domain: str,
    pipeline_run: Path,
    cache_dir: Path,
) -> dict[str, Any]:
    """Run the notebook's DataFusionEvaluator + strategy on BOTH the
    best-of-breed fused frame and the cached notebook fused frame
    against the same gold.

    Returns ``{"pipe_scores": ..., "notebook_scores": ..., ...}``.
    Raises if either fused frame is missing; callers should wrap in
    try/except.
    """
    spec = get_notebook_fusion_spec(domain)
    gold, gold_info = _load_notebook_fusion_gold(domain)

    pipe_fused_path = pipeline_run / "fused.csv"
    nb_fused_path = cache_dir / "notebook_fused.csv"
    if not pipe_fused_path.exists():
        raise FileNotFoundError(pipe_fused_path)
    if not nb_fused_path.exists():
        raise FileNotFoundError(nb_fused_path)

    pipe_fused = pd.read_csv(pipe_fused_path)
    nb_fused = pd.read_csv(nb_fused_path)

    # Pipe-side: derive anchor id column if absent.
    pipe_fused = _derive_anchor_id_column(
        pipe_fused, domain=domain, out_col=spec.fused_id_column
    )

    # Notebook-side: same derivation in case the cache doesn't carry it.
    nb_fused = _derive_anchor_id_column(
        nb_fused, domain=domain, out_col=spec.fused_id_column
    )

    pipe_scores = evaluate_with_notebook_strategy(
        pipe_fused, domain=domain, gold_df=gold
    )
    nb_scores = evaluate_with_notebook_strategy(nb_fused, domain=domain, gold_df=gold)

    return {
        "domain": domain,
        "gold_info": gold_info,
        "rules": [
            {
                "attribute": a,
                "function": fn.__name__,
                "kwargs": kw,
            }
            for a, fn, kw in spec.rules
        ],
        "fused_id_column": spec.fused_id_column,
        "gold_id_column": spec.gold_id_column,
        "pipe_rows": int(len(pipe_fused)),
        "notebook_rows": int(len(nb_fused)),
        "pipe_scores": pipe_scores,
        "notebook_scores": nb_scores,
    }


def _append_notebook_style_fusion_section(
    lines: list[str], result: dict[str, Any] | None
) -> None:
    """Append the apples-to-apples notebook-style fusion eval section."""
    lines.append("")
    lines.append(
        "### Stage 6 supplement — notebook-style fusion eval (apples-to-apples)"
    )
    lines.append("")
    if result is None:
        lines.append(
            "_Notebook-style fusion eval unavailable (gold file or fused "
            "cache missing; see logs)._ "
        )
        lines.append("")
        return

    gi = result["gold_info"]
    lines.append(
        "Both fused outputs scored with the **same notebook DataFusionEvaluator "
        f"strategy** ({len(result['rules'])} per-attribute rules, gold = "
        f"`{gi['path']}`, {gi['rows']} rows"
        + (f", filter: `{gi['filter']}`" if gi.get("filter") else "")
        + (f", preprocessing: {gi['preprocessing']}" if gi.get("preprocessing") else "")
        + ")."
    )
    lines.append("")
    lines.append(
        f"- best-of-breed fused rows aligned to anchor id `{result['fused_id_column']}` = "
        f"{result['pipe_rows']}"
    )
    lines.append(
        f"- notebook fused rows aligned to anchor id `{result['fused_id_column']}` = "
        f"{result['notebook_rows']}"
    )
    lines.append("")

    lines.append("**Per-attribute accuracy (same matcher + kwargs on both sides):**")
    lines.append("")
    lines.append(
        "| Attribute | Matcher | kwargs | best-of-breed | notebook | Δ (BoB − nb) |"
    )
    lines.append("|---|---|---|---|---|---|")
    pipe_scores = result["pipe_scores"]
    nb_scores = result["notebook_scores"]
    for rule in result["rules"]:
        attr = rule["attribute"]
        fn = rule["function"]
        kw = rule["kwargs"]
        kw_str = ", ".join(f"{k}={v}" for k, v in kw.items()) if kw else "—"
        # DataFusionEvaluator publishes per-attribute scores as
        # ``<attr>_accuracy`` (with ``<attr>_count`` siblings) — not as
        # bare attribute keys.
        p = pipe_scores.get(f"{attr}_accuracy")
        n = nb_scores.get(f"{attr}_accuracy")
        lines.append(
            f"| {attr} | `{fn}` | {kw_str} | "
            f"{_fmt(p)} | {_fmt(n)} | {_delta_str(p, n)} |"
        )
    # Overall accuracy row if available.
    p_overall = pipe_scores.get("overall_accuracy")
    n_overall = nb_scores.get("overall_accuracy")
    if p_overall is not None or n_overall is not None:
        lines.append(
            f"| **overall_accuracy** | — | — | "
            f"{_fmt(p_overall)} | {_fmt(n_overall)} | {_delta_str(p_overall, n_overall)} |"
        )
    lines.append("")
    lines.append(
        "_This table replays the notebook's exact evaluation surface on "
        "both fused outputs, so positive Δ ⇒ best-of-breed beats the "
        "notebook on that attribute under the matcher the notebook "
        "itself uses._"
    )
    lines.append("")


def write_full_comparison_report(
    *,
    pipeline_run: Path,
    panel_dir: Path,
    out_path: Path,
    domain: str,
) -> None:
    """Comprehensive report: per-stage test metrics + end-to-end panel
    with metric descriptions. The single canonical artifact to share
    with collaborators.
    """
    panel = json.loads((panel_dir / "panel.json").read_text())
    composite = json.loads((panel_dir / "composite_score.json").read_text())

    # Read best-of-breed per-stage info.
    stage_files = sorted(pipeline_run.glob("stage_*_selection.json"))
    bob_stages = {
        json.loads(p.read_text())["stage"]: json.loads(p.read_text())
        for p in stage_files
    }

    # Read notebook per-stage test info from its written eval files.
    nb_blocking = _read_notebook_blocking_test()
    nb_matching = _read_notebook_matching_test()

    # ---- Apples-to-apples helpers (compute missing scores live) ----
    # 1. BoB winning blocker re-scored for pair_completeness (the metric
    #    notebook reports). The persisted JSON only carries reduction_ratio.
    em_blk_stage = bob_stages.get("em_blocking") or {}
    winner_per_pair = (em_blk_stage.get("notes") or {}).get("per_pair_winner") or {}
    bob_blocker_per_pair: dict[str, dict[str, float]] = {}
    if winner_per_pair:
        try:
            bob_blocker_per_pair = _score_winning_blocker_for_pair_completeness(
                domain, winner_per_pair
            )
        except Exception:
            logger.exception(
                "Failed to re-score BoB blocker for pair_completeness; "
                "blocking table will show only persisted reduction_ratio."
            )

    # 2. Notebook LLM SM re-scored against the same SM gold so Stage 1
    #    isn't asymmetric (BoB has F1, notebook had '—').
    try:
        nb_llm_sm = _score_notebook_llm_sm(domain)
    except Exception:
        logger.exception(
            "Failed to re-score notebook LLM SM; Stage 1 will note absence"
        )
        nb_llm_sm = None

    lines: list[str] = []

    # -------------------------------------------------------------------
    # Header
    # -------------------------------------------------------------------
    lines.append(f"# Best-of-breed vs human-baseline notebook — {domain}")
    lines.append("")
    lines.append(
        "Comparison report covering (1) per-stage test metrics with the "
        "winning algorithm called out on each side, and (2) the end-to-end "
        "panel from [docs/tutorial/e2e_evaluation/e2e_evaluation_metrics.md](../../../docs/tutorial/e2e_evaluation/e2e_evaluation_metrics.md) "
        "where the notebook's fused output is treated as the silver standard."
    )
    lines.append("")
    lines.append(f"- Best-of-breed run: `{pipeline_run}`")
    lines.append(
        f"- Notebook silver wrapper from: `pipelines/{domain}/baselines/notebook_fused.csv`"
    )
    lines.append("")

    # -------------------------------------------------------------------
    # Part 1 — Per-stage test metrics
    # -------------------------------------------------------------------
    lines.append("---")
    lines.append("")
    lines.append("# Part 1 — Per-stage test metrics")
    lines.append("")
    lines.append(
        "Each pipeline stage produces a test-set score for the chosen "
        "algorithm. Best-of-breed picks each winner by **validation-set** "
        "score and reports test alongside; the notebook hand-picks each "
        "stage's algorithm a priori and reports the test result."
    )
    lines.append("")

    # ----- Stage 1 — Schema matching -----
    sm = bob_stages.get("sm") or {}
    lines.append("## Stage 1 — Schema matching")
    lines.append("")
    lines.append(
        "**Apples-to-apples comparison surface**: F1 against the same SM gold "
        "(`sm_mapping_gold.csv`). Best-of-breed scored every committee "
        "member; we re-ran the notebook's `LLMBasedSchemaMatcher` against the "
        "same gold to fill in the notebook column."
    )
    lines.append("")
    lines.append("| Side | Algorithm | Metric | Test F1 |")
    lines.append("|---|---|---|---|")
    lines.append(
        f"| best-of-breed | `{sm.get('winner', '—')}` | F1 vs sm_mapping_gold.csv "
        f"| {_fmt(sm.get('test_score'))} |"
    )
    if nb_llm_sm:
        lines.append(
            f"| notebook | `LLMBasedSchemaMatcher (model={nb_llm_sm['model']}, "
            f"num_rows=40)` | F1 vs sm_mapping_gold.csv | "
            f"{_fmt(nb_llm_sm['f1'])} |"
        )
        lines.append("")
        lines.append(
            f"Notebook LLM SM precision = {_fmt(nb_llm_sm['precision'])}, "
            f"recall = {_fmt(nb_llm_sm['recall'])} (re-run against the same "
            "SM gold; notebook itself doesn't save its mapping)."
        )
    else:
        lines.append(
            "| notebook | `LLMBasedSchemaMatcher (gpt-5.5, num_rows=40)` | F1 vs sm_mapping_gold.csv "
            "| — (OPENAI_API_KEY not set or re-run failed; see logs) |"
        )
    lines.append("")
    lines.append("**Best-of-breed per-member F1** (all candidates considered):")
    lines.append("")
    lines.append("| Member | val | test |")
    lines.append("|---|---|---|")
    for m in sorted((sm.get("per_member_val") or {}).keys()):
        lines.append(
            f"| {m} | {_fmt(sm['per_member_val'].get(m))} | "
            f"{_fmt(sm.get('per_member_test', {}).get(m))} |"
        )
    lines.append("")

    # ----- Stage 2 — Normalization -----
    norm = bob_stages.get("norm") or {}
    lines.append("## Stage 2 — Normalization")
    lines.append("")
    lines.append(
        "**Apples-to-apples note**: the notebook starts from "
        "`data_cleaned_final/` (which is the post-cleaning state), so its "
        "effective normalization is the **same identity transform** as "
        "best-of-breed's winning `passthrough` member. The committee's "
        "macro_f1 metric — fraction of fusion-protected cells that "
        "normalize to the canonical value — is therefore identical on both "
        "sides because both pipelines hand the downstream stages the same "
        "rows."
    )
    lines.append("")
    lines.append("| Side | Algorithm | Metric | Test macro_f1 |")
    lines.append("|---|---|---|---|")
    lines.append(
        f"| best-of-breed | `{norm.get('winner', '—')}` (won a vacuous tie) | "
        f"macro_f1 vs fusion-protected cells | {_fmt(norm.get('test_score'))} |"
    )
    lines.append(
        "| notebook | `passthrough` (equivalent: data_cleaned_final used as-is) | "
        f"macro_f1 vs fusion-protected cells | {_fmt(norm.get('test_score'))} |"
    )
    lines.append("")
    notes = norm.get("notes") or {}
    if notes.get("vacuous"):
        lines.append(
            f"*Best-of-breed flagged this stage as **vacuous** (per-member spread = "
            f"{notes.get('spread', 0):.4f} < epsilon {notes.get('vacuous_epsilon', 0):.4f}) — "
            f"all normalizers produced identical scores on the pre-cleaned input.*"
        )
        lines.append("")

    # ----- Stage 3 — EM blocking -----
    em_blk = bob_stages.get("em_blocking") or {}
    lines.append("## Stage 3 — Entity-matching blocking")
    lines.append("")
    lines.append(
        "**Apples-to-apples comparison surface**: per-pair "
        "**pair_completeness** (recall — fraction of true gold pairs the "
        "blocker keeps) and **reduction_ratio** (fraction of all possible "
        "pairs the blocker prunes) against each pair's `*_test.csv` EM gold. "
        "Best-of-breed persists pair_completeness (primary) and "
        "reduction_ratio (secondary, surfaced under "
        "`notes.per_member_reduction_ratio_*`) per the committee's "
        "composition strategy; we re-ran the winning blocker against the "
        "test gold for the per-pair pair_completeness / reduction_ratio "
        "breakdown."
    )
    lines.append("")
    lines.append("**Per-side winner — averaged across the 3 source pairs:**")
    lines.append("")
    lines.append("| Side | Algorithm | pair_completeness | reduction_ratio |")
    lines.append("|---|---|---|---|")
    # BoB row
    bob_pc_avg = None
    bob_rr_avg = None
    if bob_blocker_per_pair:
        pcs = [v["pair_completeness"] for v in bob_blocker_per_pair.values()]
        rrs = [v["reduction_ratio"] for v in bob_blocker_per_pair.values()]
        bob_pc_avg = sum(pcs) / len(pcs)
        bob_rr_avg = sum(rrs) / len(rrs)
        lines.append(
            f"| best-of-breed | `{em_blk.get('winner', '—')}` (selected per-pair "
            f"via composition) | {bob_pc_avg:.4f} | {bob_rr_avg:.4f} |"
        )
    else:
        # Fallback when the per-pair re-eval didn't run / failed. After
        # the metric-flip, em_blk['test_score'] carries recall
        # (pair_completeness), not reduction_ratio. Pull the side-metric
        # RR from notes when available; the winner is the stage-level
        # winner string, which may not match any single per-pair winner.
        em_notes = em_blk.get("notes") or {}
        rr_side = em_notes.get("per_member_reduction_ratio_test") or {}
        winner_name = em_blk.get("winner", "")
        rr_fallback = rr_side.get(winner_name) if winner_name else None
        lines.append(
            f"| best-of-breed | `{em_blk.get('winner', '—')}` (selected per-pair "
            f"via composition) | {_fmt(em_blk.get('test_score'))} | "
            f"{_fmt(rr_fallback)} |"
        )
    if nb_blocking:
        rr_avg = sum(d["reduction_ratio"] for d in nb_blocking.values()) / len(
            nb_blocking
        )
        pc_avg = sum(d["pair_completeness"] for d in nb_blocking.values()) / len(
            nb_blocking
        )
        lines.append(
            f"| notebook | `StandardBlocker(on=['product_type'])` | "
            f"{pc_avg:.4f} | {rr_avg:.4f} |"
        )
    lines.append("")

    # BoB per-pair table (apples-to-apples with the notebook per-pair table below)
    if bob_blocker_per_pair:
        lines.append(
            "**Best-of-breed per-pair blocking** (winning blocker re-run against "
            "each pair's `*_test.csv` EM gold):"
        )
        lines.append("")
        lines.append(
            "| Pair | winner blocker | pair_completeness | reduction_ratio | "
            "n_candidates | true_pairs_kept / total_true |"
        )
        lines.append("|---|---|---|---|---|---|")
        for pair_key in sorted(bob_blocker_per_pair):
            r = bob_blocker_per_pair[pair_key]
            lines.append(
                f"| {pair_key} | `{r['winner_blocker']}` | "
                f"{_fmt(r['pair_completeness'])} | {_fmt(r['reduction_ratio'])} | "
                f"{r['n_candidates']:,} | "
                f"{r['true_positives_found']} / {r['n_true_pairs']} |"
            )
        lines.append("")

    lines.append(
        "**Best-of-breed per-member pair_completeness (recall) — "
        "primary** and **reduction_ratio — secondary** (val=test on the "
        "blocking surface since the committee scores against em_gold, "
        "which is the test split):"
    )
    lines.append("")
    lines.append("| Member | recall (val) | recall (test) | RR (val) | RR (test) |")
    lines.append("|---|---|---|---|---|")
    em_notes = em_blk.get("notes") or {}
    rr_val_map = em_notes.get("per_member_reduction_ratio_val") or {}
    rr_test_map = em_notes.get("per_member_reduction_ratio_test") or {}
    for m in sorted((em_blk.get("per_member_val") or {}).keys()):
        lines.append(
            f"| {m} | {_fmt(em_blk['per_member_val'].get(m))} | "
            f"{_fmt(em_blk.get('per_member_test', {}).get(m))} | "
            f"{_fmt(rr_val_map.get(m))} | {_fmt(rr_test_map.get(m))} |"
        )
    lines.append("")
    if nb_blocking:
        lines.append(
            "**Notebook per-pair blocking** (single algorithm — `StandardBlocker(on=['product_type'])`):"
        )
        lines.append("")
        lines.append("| Pair | pair_completeness | reduction_ratio |")
        lines.append("|---|---|---|")
        for pair, scores in nb_blocking.items():
            lines.append(
                f"| {pair} | {_fmt(scores['pair_completeness'])} | "
                f"{_fmt(scores['reduction_ratio'])} |"
            )
        lines.append("")

    # EM matching
    em_m = bob_stages.get("em_matching") or {}
    lines.append("## Stage 4 — Entity-matching")
    lines.append("")
    lines.append("**Per-side winner**")
    lines.append("")
    lines.append("| Side | Algorithm | Metric | Test score |")
    lines.append("|---|---|---|---|")
    lines.append(
        f"| best-of-breed | `{em_m.get('winner', '—')}` (pipeline-isolated checkpoint "
        f"retrained for this run) | F1 vs per-pair EM test gold | "
        f"{_fmt(em_m.get('test_score'))} |"
    )
    # Notebook's "baseline" (raw rule-based) is the winner per the notebook's
    # own refinement comparison.
    if NOTEBOOK_BASELINE_MATCHING:
        f1_avg = sum(d["f1"] for d in NOTEBOOK_BASELINE_MATCHING.values()) / len(
            NOTEBOOK_BASELINE_MATCHING
        )
        lines.append(
            f"| notebook | `RuleBasedMatcher` (hand-weighted comparators on "
            f"title/brand/product_type/storage_gb; weights [0.30, 0.30, 0.30, 0.10], "
            f"threshold 0.70) | F1 (macro across 3 pairs) | {_fmt(f1_avg)} |"
        )
    lines.append("")
    lines.append("**Best-of-breed per-member F1**:")
    lines.append("")
    lines.append("| Member | val | test |")
    lines.append("|---|---|---|")
    for m in sorted((em_m.get("per_member_val") or {}).keys()):
        lines.append(
            f"| {m} | {_fmt(em_m['per_member_val'].get(m))} | "
            f"{_fmt(em_m.get('per_member_test', {}).get(m))} |"
        )
    lines.append("")
    if NOTEBOOK_BASELINE_MATCHING:
        lines.append(
            "**Notebook per-pair matching** (single rule-based matcher, "
            "winning over greedy/mbm refinement per the notebook's own "
            "comparison and over ML matcher):"
        )
        lines.append("")
        lines.append("| Pair | precision | recall | F1 | accuracy |")
        lines.append("|---|---|---|---|---|")
        for pair, scores in NOTEBOOK_BASELINE_MATCHING.items():
            lines.append(
                f"| {pair} | {_fmt(scores['precision'])} | {_fmt(scores['recall'])} | "
                f"{_fmt(scores['f1'])} | {_fmt(scores['accuracy'])} |"
            )
        lines.append("")

    # Refinement
    ref = bob_stages.get("refinement") or {}
    lines.append("## Stage 5 — Post-clustering refinement")
    lines.append("")
    lines.append("**Per-side winner**")
    lines.append("")
    lines.append("| Side | Algorithm | Metric | Test score |")
    lines.append("|---|---|---|---|")
    lines.append(
        f"| best-of-breed | `{ref.get('winner', '—')}` (no refinement — raw matcher "
        f"output passes through) | F1 vs val gold | {_fmt(ref.get('test_score'))} |"
    )
    lines.append(
        "| notebook | `baseline` (no refinement — won per-pair vs greedy + mbm in "
        "the notebook's own comparison) | F1 | (same as Stage 4) |"
    )
    lines.append("")
    lines.append("**Best-of-breed per-method F1**:")
    lines.append("")
    lines.append("| Method | val | test |")
    lines.append("|---|---|---|")
    for m in sorted((ref.get("per_member_val") or {}).keys()):
        lines.append(
            f"| {m} | {_fmt(ref['per_member_val'].get(m))} | "
            f"{_fmt(ref.get('per_member_test', {}).get(m))} |"
        )
    lines.append("")
    if nb_matching:
        lines.append(
            "**Notebook per-pair refinement** (raw `baseline` wins all 3 pairs):"
        )
        lines.append("")
        lines.append("| Pair | refiner | precision | recall | F1 | accuracy |")
        lines.append("|---|---|---|---|---|---|")
        for pair in (
            "products_1_products_2",
            "products_1_products_3",
            "products_1_products_4",
        ):
            nb_baseline = NOTEBOOK_BASELINE_MATCHING.get(pair, {})
            if nb_baseline:
                lines.append(
                    f"| {pair} | baseline (no refinement) | "
                    f"{_fmt(nb_baseline['precision'])} | {_fmt(nb_baseline['recall'])} | "
                    f"**{_fmt(nb_baseline['f1'])}** | {_fmt(nb_baseline['accuracy'])} |"
                )
            for refiner in ("greedy", "mbm"):
                row = nb_matching.get(pair, {}).get(refiner)
                if not row:
                    continue
                lines.append(
                    f"| {pair} | {refiner} | {_fmt(row['precision'])} | "
                    f"{_fmt(row['recall'])} | {_fmt(row['f1'])} | "
                    f"{_fmt(row['accuracy'])} |"
                )
        lines.append("")

    # Fusion
    fus = bob_stages.get("fusion") or {}
    lines.append("## Stage 6 — Data fusion")
    lines.append("")
    lines.append("**Per-side winner**")
    lines.append("")
    lines.append("| Side | Algorithm | Metric | Test score |")
    lines.append("|---|---|---|---|")
    lines.append(
        f"| best-of-breed | `{fus.get('winner', '—')}` (val-best PyDI strategy per "
        f"attribute, locked once + replayed) | macro_accuracy vs fusion test gold "
        f"| {_fmt(fus.get('test_score'))} |"
    )
    lines.append(
        f"| notebook | `{NOTEBOOK_FUSION_TEST['winner']}` "
        f"({NOTEBOOK_FUSION_TEST['strategy_detail']}) | overall_accuracy / "
        f"macro_accuracy | {_fmt(NOTEBOOK_FUSION_TEST['overall_accuracy'])} / "
        f"{_fmt(NOTEBOOK_FUSION_TEST['macro_accuracy'])} |"
    )
    lines.append("")
    lines.append("**Best-of-breed per-member macro_accuracy**:")
    lines.append("")
    lines.append("| Member | val | test |")
    lines.append("|---|---|---|")
    for m in sorted((fus.get("per_member_val") or {}).keys()):
        lines.append(
            f"| {m} | {_fmt(fus['per_member_val'].get(m))} | "
            f"{_fmt(fus.get('per_member_test', {}).get(m))} |"
        )
    lines.append("")

    # Notebook-style fusion eval (apples-to-apples) — same matcher +
    # kwargs the notebook itself uses, on both fused outputs.
    try:
        nb_style_eval = compute_notebook_style_fusion_comparison(
            domain=domain,
            pipeline_run=pipeline_run,
            cache_dir=REPO_ROOT / "pipelines" / domain / "baselines",
        )
    except Exception:
        logger.exception(
            "Failed to compute notebook-style fusion eval; section will note absence."
        )
        nb_style_eval = None
    _append_notebook_style_fusion_section(lines, nb_style_eval)

    # -------------------------------------------------------------------
    # Part 2 — End-to-end metrics
    # -------------------------------------------------------------------
    lines.append("---")
    lines.append("")
    lines.append("# Part 2 — End-to-end metrics (panel)")
    lines.append("")
    lines.append(
        "Single panel computed on the best-of-breed fused output against "
        "the notebook's fused output treated as silver. Every metric in "
        'this section answers "**how close did best-of-breed come to the '
        'notebook\'s fused output?**" — _not_ "how close to a third silver."'
    )
    lines.append("")
    lines.append(
        "Definitions and recipes are paraphrased from "
        "[e2e_evaluation_metrics.md](../../../docs/tutorial/e2e_evaluation/e2e_evaluation_metrics.md). "
        "Each metric carries a description (what it answers), a method "
        "(how it's calculated), and a reading (what the value tells us "
        "about the best-of-breed vs notebook comparison)."
    )
    lines.append("")

    # Composite + tier subscores
    cs = composite.get("composite_score")
    lines.append(f"## Composite score = {_fmt(cs)}")
    lines.append("")
    lines.append(f"*{METRIC_DESCRIPTIONS['composite_score']}*")
    lines.append("")
    weights = composite.get("weights") or {}
    tiers = composite.get("tier_subscores") or {}
    lines.append("### Tier subscores")
    lines.append("")
    lines.append("| Tier (weight) | value | what it measures |")
    lines.append("|---|---|---|")
    for tier in [
        "entity_coverage",
        "column_shape",
        "cluster_correctness",
        "value_correctness",
    ]:
        w = weights.get(tier)
        w_str = f" ({w:.2f})" if isinstance(w, (int, float)) else ""
        lines.append(
            f"| **{tier}**{w_str} | {_fmt(tiers.get(tier))} | "
            f"{METRIC_DESCRIPTIONS.get(tier, '—')} |"
        )
    lines.append("")

    # Tier 1
    _append_tier_with_descriptions(
        lines,
        tier_name="Tier 1 — entity coverage",
        block=panel.get("entity_coverage") or {},
        ordered_keys=[
            "n_pipe",
            "n_silver",
            "row_count_abs_diff",
            "row_count_rel_diff",
        ],
        nested_block_key="entity_overlap",
        nested_keys=[
            "n_recovered",
            "n_partial",
            "n_lost",
            "n_fabricated",
            "recovery_rate",
        ],
    )

    # Tier 2 — per-column tables only, with a doc paragraph
    lines.append("## Tier 2 — column shape (per-column distributional comparison)")
    lines.append("")
    lines.append(
        "**What it measures**: per column, how different are the value "
        "distributions between best-of-breed and the notebook? Each column "
        "is routed to a type-appropriate metric (JS for categorical, "
        "Wasserstein-1 for numerical, length-W1 + token-JS for text, "
        "W1-days for datetime). The drift values are normalized so they "
        "can be averaged. **Reading**: low values across the board → best-"
        "of-breed picks similar values to the notebook on average. Doesn't "
        "see per-record assignment errors (those live in Tier 4)."
    )
    lines.append("")
    drift = (panel.get("column_shape") or {}).get("per_column_drift_normalized") or {}
    if drift:
        lines.append("### per-column drift (normalized)")
        lines.append("")
        lines.append("| Column | drift |")
        lines.append("|---|---|")
        for col in sorted(drift):
            lines.append(f"| {col} | {_fmt(drift[col])} |")
        lines.append("")
    validity = (panel.get("column_shape") or {}).get("validity_per_column") or {}
    if validity:
        lines.append("### constraint / type validity per column")
        lines.append("")
        lines.append(
            "**What it measures**: per column, what fraction of non-null cells parse "
            "to the declared type and satisfy declared constraints. **Reading**: a "
            "negative pipe delta = best-of-breed emits cells violating the column's "
            "type/constraints that the notebook didn't."
        )
        lines.append("")
        lines.append("| Column | pipe validity | silver validity | delta |")
        lines.append("|---|---|---|---|")
        for col in sorted(validity):
            v = validity[col] or {}
            lines.append(
                f"| {col} | {_fmt(v.get('validity_rate_pipe'))} | "
                f"{_fmt(v.get('validity_rate_reference'))} | "
                f"{_fmt(v.get('delta'), '+.4f')} |"
            )
        lines.append("")
    mv = (panel.get("column_shape") or {}).get("mean_validity_delta")
    lines.append(f"**mean_validity_delta**: {_fmt(mv, '+.4f')}")
    lines.append("")

    # Tier 3
    cc = panel.get("cluster_correctness") or {}
    bc = cc.get("bcubed") or {}
    lines.append("## Tier 3 — cluster correctness")
    lines.append("")
    lines.append("### BCubed (§3.1)")
    lines.append("")
    lines.append("| Metric | value | description |")
    lines.append("|---|---|---|")
    for key in ["precision", "recall", "f1"]:
        lines.append(
            f"| bcubed.{key} | {_fmt(bc.get(key))} | "
            f"{METRIC_DESCRIPTIONS.get(f'bcubed.{key}', '—')} |"
        )
    lines.append("")
    al = cc.get("alignment") or {}
    lines.append("### Cluster alignment (§3.2)")
    lines.append("")
    lines.append("| Metric | value | description |")
    lines.append("|---|---|---|")
    for key in [
        "mean_jaccard",
        "matched_cluster_rate_at_threshold",
        "matched_threshold",
        "size_match_rate",
        "mean_size_delta",
        "max_size_overshoot",
    ]:
        lines.append(
            f"| {key} | {_fmt(al.get(key))} | " f"{METRIC_DESCRIPTIONS.get(key, '—')} |"
        )
    lines.append("")
    sc = cc.get("source_composition") or {}
    ss = sc.get("same_source_collision_rate") or {}
    lines.append("### Source composition (§3.3)")
    lines.append("")
    lines.append(
        "**What it measures**: in a multi-source data integration pipeline, each "
        "cluster has a *source signature* (the multiset of source datasets its "
        "records come from). Silver typically has one record from each of N "
        "sources. Pipeline mistakes produce characteristic signature deviations. "
        "**Reading**: same_source_collision_rate is the highest-value single signal — "
        "a non-zero delta means best-of-breed merges multiple records from the same "
        "source into one cluster (EM over-merge red flag)."
    )
    lines.append("")
    lines.append(
        "**same_source_collision_rate** — silver: "
        f"{_fmt(ss.get('silver'))} · pipe: {_fmt(ss.get('pipe'))} · "
        f"delta: {_fmt(ss.get('delta'), '+.4f')}"
    )
    lines.append("")
    mix = sc.get("source_mix_distribution_js")
    lines.append(
        f"**source_mix_distribution_js**: {_fmt(mix)} — "
        f"{METRIC_DESCRIPTIONS.get('source_mix_distribution_js', '—')}"
    )
    lines.append("")
    cov = sc.get("per_source_coverage_rate") or {}
    if cov:
        lines.append(
            "**per_source_coverage_rate** — per source, fraction of clusters that "
            "contain ≥ 1 record from it. A negative delta means best-of-breed clusters "
            "touch that source's records less often than notebook clusters do (could "
            "indicate a blocking-recall or matcher-recall regression for that source)."
        )
        lines.append("")
        lines.append("| Source | silver | pipe | delta |")
        lines.append("|---|---|---|---|")
        for src in sorted(cov):
            row = cov[src] or {}
            lines.append(
                f"| {src} | {_fmt(row.get('silver'))} | "
                f"{_fmt(row.get('pipe'))} | {_fmt(row.get('delta'), '+.4f')} |"
            )
        lines.append("")

    # Tier 4
    vc = panel.get("value_correctness") or {}
    lines.append("## Tier 4 — value correctness")
    lines.append("")
    lines.append("### Headline accuracy + conflict context")
    lines.append("")
    lines.append("| Metric | value | description |")
    lines.append("|---|---|---|")
    for key in [
        "macro_accuracy",
        "micro_accuracy",
        "conflict_only_accuracy",
        "conflict_only_micro_accuracy",
        "conflict_rate_pipe",
        "conflict_rate_silver",
        "fully_correct_cluster_rate",
    ]:
        lines.append(
            f"| {key} | {_fmt(vc.get(key))} | " f"{METRIC_DESCRIPTIONS.get(key, '—')} |"
        )
    lines.append("")
    pa = vc.get("per_attribute") or {}
    if pa:
        lines.append("### per-attribute accuracy + normalization fingerprint")
        lines.append("")
        lines.append(
            "**What it measures**: per attribute, exact-match accuracy of fused "
            "values against the notebook's fused values. For **text** attributes, "
            "also computes similarity_mean (normalized Levenshtein) and the "
            "accuracy↔similarity gap. When the gap > 0.10 the fingerprint reads "
            "`normalization_difference_suspected` — wrong values are close-to-right, "
            "suggesting the two pipelines normalize the same fact differently. "
            "**Reading**: high `accuracy` = exact agreement; high `similarity_mean` "
            "with low `accuracy` = semantic agreement but normalization drift. The "
            "fix per doc §4.1b is to pass a `semantic_value_similarity` callable."
        )
        lines.append("")
        lines.append(
            "| Attribute | accuracy | similarity_mean | gap | fingerprint | count |"
        )
        lines.append("|---|---|---|---|---|---|")
        for a in sorted(pa):
            row = pa[a] or {}
            lines.append(
                f"| {a} | {_fmt(row.get('accuracy'))} | "
                f"{_fmt(row.get('similarity_mean'))} | "
                f"{_fmt(row.get('accuracy_similarity_gap'), '+.4f')} | "
                f"{row.get('mismatch_fingerprint') or '—'} | "
                f"{_fmt(row.get('count'), 'd')} |"
            )
        lines.append("")
    co = vc.get("conflict_only_per_attribute") or {}
    if co:
        lines.append("### conflict-only per-attribute accuracy")
        lines.append("")
        lines.append(
            "**What it measures**: per attribute, accuracy restricted to cells where "
            "≥ 2 distinct source values were present (i.e. fusion actually had to "
            "decide). **Reading**: low values indicate the two pipelines' fusion "
            "policies disagree on contested cells."
        )
        lines.append("")
        lines.append("| Attribute | accuracy | count |")
        lines.append("|---|---|---|")
        for a in sorted(co):
            row = co[a] or {}
            lines.append(
                f"| {a} | {_fmt(row.get('accuracy'))} | "
                f"{_fmt(row.get('count'), 'd')} |"
            )
        lines.append("")
    dd = vc.get("density_delta_per_attribute") or {}
    if dd:
        lines.append("### density delta per attribute")
        lines.append("")
        lines.append(
            "**What it measures**: per attribute, fraction of fused rows with a "
            "non-null value, pipe vs silver. **Reading**: negative delta = pipeline "
            "silently nulled cells the notebook filled; positive delta = pipeline "
            "filled cells the notebook left null. Magnitudes near zero = both "
            "pipelines fill the column at similar rates."
        )
        lines.append("")
        lines.append("| Attribute | silver | pipe | delta |")
        lines.append("|---|---|---|---|")
        for a in sorted(dd):
            row = dd[a] or {}
            lines.append(
                f"| {a} | {_fmt(row.get('reference_density'))} | "
                f"{_fmt(row.get('pipe_density'))} | "
                f"{_fmt(row.get('delta'), '+.4f')} |"
            )
        lines.append("")

    # -------------------------------------------------------------------
    # Warnings + caveats
    # -------------------------------------------------------------------
    warnings = panel.get("warnings") or []
    if warnings:
        lines.append("## Panel diagnostic warnings")
        lines.append("")
        lines.append(
            "Per [e2e_evaluation_metrics.md §Diagnostic warnings](../../../docs/tutorial/e2e_evaluation/e2e_evaluation_metrics.md), "
            "the panel fires pattern-based warnings when metric combinations "
            "match known failure modes:"
        )
        lines.append("")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")

    lines.append("## Caveats")
    lines.append("")
    lines.append(
        "- **Silver = notebook fused output** (`pipelines/products/baselines/"
        "notebook_fused.csv`), wrapped as a SilverStandard. Bare-int IDs from the "
        "notebook are translated to the prefixed (synthetic-side) scheme so "
        "best-of-breed and the notebook share a coordinate system."
    )
    lines.append(
        "- Notebook outputs carry no per-cell provenance, so source-attribution + "
        "synthesis-rate metrics (§4.6) are skipped."
    )
    lines.append(
        "- v7d ran `--no-llm` with **pipeline-isolated** ditto + sc_block "
        "checkpoints (retrained for this run under `pipelines/products/checkpoints/`)."
        " The notebook's own pipeline uses LLM SM + a rule-based EM matcher."
    )
    lines.append(
        "- Notebook per-stage numbers come from JSON files the notebook itself "
        "wrote during execution under `usecases/products/output/`; matching "
        "Stage-4 numbers reflect the raw rule-based matcher output (the "
        "notebook's `baseline` refinement winner)."
    )
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n")


def _append_tier_with_descriptions(
    lines: list[str],
    *,
    tier_name: str,
    block: Mapping[str, Any],
    ordered_keys: list[str],
    nested_block_key: str | None = None,
    nested_keys: list[str] | None = None,
) -> None:
    lines.append(f"## {tier_name}")
    lines.append("")
    lines.append("| Metric | value | description |")
    lines.append("|---|---|---|")
    for key in ordered_keys:
        lines.append(
            f"| {key} | {_fmt(block.get(key))} | "
            f"{METRIC_DESCRIPTIONS.get(key, '—')} |"
        )
    if nested_block_key:
        nested = block.get(nested_block_key) or {}
        for key in nested_keys or []:
            fmt = ".4f" if "rate" in key else "d"
            lines.append(
                f"| {nested_block_key}.{key} | {_fmt(nested.get(key), fmt)} | "
                f"{METRIC_DESCRIPTIONS.get(key, '—')} |"
            )
    lines.append("")


def write_single_panel_md(
    *,
    pipeline_run: Path,
    panel_dir: Path,
    out_path: Path,
    domain: str,
) -> None:
    """Single-panel report: best-of-breed measured against notebook-as-silver."""
    panel_path = panel_dir / "panel.json"
    composite_path = panel_dir / "composite_score.json"
    panel = json.loads(panel_path.read_text()) if panel_path.exists() else {}
    composite = (
        json.loads(composite_path.read_text()) if composite_path.exists() else {}
    )

    lines: list[str] = []
    lines.append(f"# Best-of-breed vs human-baseline notebook — {domain}")
    lines.append("")
    lines.append(
        "Single panel: **`pipe = best-of-breed (v7d)`** measured against "
        "**`silver = the cached human-baseline notebook output`**. "
        'Every metric below answers "how close did best-of-breed come to '
        "the notebook's fused output?\""
    )
    lines.append("")
    lines.append(f"- Best-of-breed run: `{pipeline_run}`")
    lines.append(
        f"- Notebook silver wrapper built from: `pipelines/{domain}/baselines/notebook_fused.csv`"
    )
    lines.append("")

    cs = composite.get("composite_score")
    lines.append("## Composite score")
    lines.append("")
    lines.append(
        f"**`composite_score` = {_fmt(cs)}** (best-of-breed vs notebook-as-silver)"
    )
    lines.append("")
    weights = composite.get("weights") or {}
    tiers = composite.get("tier_subscores") or {}
    lines.append("### Tier subscores")
    lines.append("")
    lines.append("| Tier (weight) | value |")
    lines.append("|---|---|")
    for tier in [
        "entity_coverage",
        "column_shape",
        "cluster_correctness",
        "value_correctness",
    ]:
        w = weights.get(tier)
        w_str = f" ({w:.2f})" if isinstance(w, (int, float)) else ""
        lines.append(f"| {tier}{w_str} | {_fmt(tiers.get(tier))} |")
    lines.append("")

    _append_tier1_single(lines, panel.get("entity_coverage") or {})
    _append_tier2_single(lines, panel.get("column_shape") or {})
    _append_tier3_single(lines, panel.get("cluster_correctness") or {})
    _append_tier4_single(lines, panel.get("value_correctness") or {})

    warnings = panel.get("warnings") or []
    if warnings:
        lines.append("## Panel warnings")
        lines.append("")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")

    lines.append("## Caveats")
    lines.append("")
    lines.append(
        "- The notebook is wrapped as a SilverStandard from "
        "`notebook_fused.csv` (the cached human-baseline output). Bare-int "
        "IDs are translated to the prefixed (synthetic-side) scheme so the "
        "best-of-breed output and the notebook share a coordinate system."
    )
    lines.append(
        "- The notebook output carries no per-cell provenance, so "
        "source-attribution + synthesis-rate metrics (§4.6) are skipped "
        "with the standard panel warning."
    )
    lines.append(
        "- v7d ran `--no-llm` with pipeline-isolated ditto + sc_block "
        "checkpoints. The notebook's own pipeline uses LLM SM + a rule-"
        "based EM matcher. The comparison is fair-vs-fair only in the "
        "sense that both produced a complete fused dataset for the same "
        "4 source datasets."
    )

    out_path.write_text("\n".join(lines) + "\n")


def write_comparison_md(
    *,
    pipeline_run: Path,
    notebook_panel_dir: Path,
    out_path: Path,
    domain: str,
) -> None:
    """Side-by-side composite + tier table."""
    # Panel composite + tier subscores live in ``composite_score.json``,
    # not ``panel.json``. Read both files: panel.json carries warnings +
    # raw tier blocks; composite_score.json carries the weighted headline.
    pipe_panel_path = pipeline_run / "e2e_panel" / "panel.json"
    pipe_composite_path = pipeline_run / "e2e_panel" / "composite_score.json"
    nb_panel_path = notebook_panel_dir / "panel.json"
    nb_composite_path = notebook_panel_dir / "composite_score.json"

    pipe = json.loads(pipe_panel_path.read_text()) if pipe_panel_path.exists() else {}
    nb = json.loads(nb_panel_path.read_text()) if nb_panel_path.exists() else {}
    pipe_comp = (
        json.loads(pipe_composite_path.read_text())
        if pipe_composite_path.exists()
        else {}
    )
    nb_comp = (
        json.loads(nb_composite_path.read_text()) if nb_composite_path.exists() else {}
    )

    pipe_composite = pipe_comp.get("composite_score", float("nan"))
    nb_composite = nb_comp.get("composite_score", float("nan"))

    pipe_tiers = pipe_comp.get("tier_subscores", {}) or {}
    nb_tiers = nb_comp.get("tier_subscores", {}) or {}

    lines = []
    lines.append(f"# Pipeline vs human baseline — {domain}")
    lines.append("")
    lines.append(f"- Best-of-breed run: `{pipeline_run}`")
    lines.append(f"- Human baseline panel: `{notebook_panel_dir}`")
    lines.append("")
    lines.append("## Composite scores")
    lines.append("")
    lines.append("| Side | composite_score |")
    lines.append("|---|---|")
    lines.append(f"| best-of-breed | {pipe_composite:.4f} |")
    lines.append(f"| notebook (human baseline) | {nb_composite:.4f} |")
    lines.append("")
    lines.append("## Tier subscores")
    lines.append("")
    lines.append("| Tier (weight) | best-of-breed | notebook | delta |")
    lines.append("|---|---|---|---|")
    weights = pipe_comp.get("weights") or nb_comp.get("weights") or {}
    for tier in [
        "entity_coverage",
        "column_shape",
        "cluster_correctness",
        "value_correctness",
    ]:
        p = pipe_tiers.get(tier, float("nan"))
        n = nb_tiers.get(tier, float("nan"))
        w = weights.get(tier)
        w_str = f" ({w:.2f})" if isinstance(w, (int, float)) else ""
        try:
            delta = p - n
            delta_str = f"{delta:+.4f}"
        except TypeError:
            delta_str = "—"
        lines.append(f"| {tier}{w_str} | {p:.4f} | {n:.4f} | {delta_str} |")
    lines.append("")

    # ----- Tier 1: entity_coverage detail -----
    _append_tier1(
        lines, pipe.get("entity_coverage") or {}, nb.get("entity_coverage") or {}
    )
    # ----- Tier 2: column_shape detail -----
    _append_tier2(lines, pipe.get("column_shape") or {}, nb.get("column_shape") or {})
    # ----- Tier 3: cluster_correctness detail -----
    _append_tier3(
        lines,
        pipe.get("cluster_correctness") or {},
        nb.get("cluster_correctness") or {},
    )
    # ----- Tier 4: value_correctness detail -----
    _append_tier4(
        lines, pipe.get("value_correctness") or {}, nb.get("value_correctness") or {}
    )

    pipe_warnings = pipe.get("warnings") or []
    nb_warnings = nb.get("warnings") or []
    if pipe_warnings or nb_warnings:
        lines.append("## Panel warnings")
        lines.append("")
        if pipe_warnings:
            lines.append("**best-of-breed:**")
            for w in pipe_warnings:
                lines.append(f"- {w}")
            lines.append("")
        if nb_warnings:
            lines.append("**notebook:**")
            for w in nb_warnings:
                lines.append(f"- {w}")
            lines.append("")

    lines.append("## Caveats")
    lines.append("")
    lines.append(
        "- Notebook correspondences are reconstructed by disambiguating "
        "each (id1, id2) row against the synthetic-side source id spaces "
        "(id1 always → master products_1; id2 → products_{2,3,4} by "
        "membership lookup). Rows whose id2 falls outside any non-master "
        "source's id space are dropped; the count is logged. Tier 3 "
        "(cluster_correctness) is reliable as long as that drop count is small."
    )
    lines.append(
        "- IDs are translated bare-int → prefixed via the convention that "
        "products_1 is the master source (notebook's `p1_id` column)."
    )
    lines.append(
        "- A composite delta is a ranking signal, not a quality verdict. "
        "Inspect the per-tier deltas + per-column metrics to understand failures."
    )
    lines.append(
        "- Best-of-breed run was conducted under the no-model-reuse policy "
        "with pipeline-isolated checkpoints: ditto_plm retrained to "
        "`pipelines/<domain>/checkpoints/em_matching/ditto/runs/<ts>/checkpoints/best`, "
        "sc_block retrained to "
        "`pipelines/<domain>/checkpoints/em_blocking/sc_block/best`. The "
        "v7d run was --no-llm, so llm_matcher / comem / matchgpt / "
        "llm_only were excluded from the competition; see the run's "
        "`effective_committees/` for the exact roster."
    )

    out_path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(levelname)s] %(name)s - %(message)s",
    )

    config_path = args.config or (
        REPO_ROOT / "pipelines" / "configs" / f"{args.domain}.yaml"
    )
    config = PipelineConfig.from_yaml(config_path)
    cache_dir = REPO_ROOT / "pipelines" / args.domain / "baselines"

    if args.cache_from_notebook is not None:
        cache_notebook_output(args.cache_from_notebook, cache_dir=cache_dir)
        return 0

    if args.pipeline_run is None:
        print("ERROR: must supply --pipeline-run when not caching.", file=sys.stderr)
        return 2

    # Single-panel comparison: pipe = best-of-breed, silver = notebook.
    panel_dir = args.pipeline_run / "vs_notebook_panel"
    headline = compute_panel_pipe_vs_notebook(
        pipeline_run=args.pipeline_run,
        domain=args.domain,
        config=config,
        cache_dir=cache_dir,
        out_dir=panel_dir,
    )
    logger.info(
        "Best-of-breed vs notebook-as-silver: composite=%s",
        headline["composite_score"],
    )

    out_path = args.pipeline_run / "comparison.md"
    write_full_comparison_report(
        pipeline_run=args.pipeline_run,
        panel_dir=panel_dir,
        out_path=out_path,
        domain=args.domain,
    )
    print(f"Wrote {out_path}")
    print(f"Panel artifacts: {panel_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
