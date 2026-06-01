#!/usr/bin/env python3
"""Package a single-level augmented variant into the final S1 directory layout.

Assembles the variant directory from an intermediate work directory that
holds per-knob provenance, baselines, regenerated EM test set, and the
K8 SM mapping. Serialises the augmented source DataFrames into the
variant's ``input/data/`` folder, copies original SM/EM/fusion artifacts
that survive unchanged, and writes the ``config/difficulty.yaml``
summary.

This script is meant to be called by :mod:`generate_variant` after all
knob scripts have run, but is exposed as a CLI for re-packaging from an
existing work directory.

Variant layout produced (per ``plan.md`` §"Scenario 1: Augmented use
cases")::

    usecases/<domain>-augmented/<level>/
      input/
        data/              <source>.csv per source (renamed columns kept)
        schemamatching/    sm_mapping.csv (from K8) + target_schema.json
        entitymatching/    *all.csv / *train.csv / ... (original) + *_{train,val,test}_{baseline_pruned,corner_filled}.csv
        fusion/            test_set.xml + validation_set.xml (renamed from
                           the source's configured ``fusion_files`` block —
                           e.g. ``*_set_final.xml`` for games/music)
      output/
        provenance/        consolidated per-knob CSVs + provenance_all.csv
        baselines/         per-knob baseline measurements (K3/K4/K10)
      config/
        difficulty.yaml    knob parameters + seeds used for this variant

Usage
-----
::

    python usecases_synthetic/scripts/package_variant.py \\
        --domain companies --level easy \\
        --work-dir usecases_synthetic/output/companies/easy \\
        --variant-dir usecases/companies-augmented/easy
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Ensure repo root is importable.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from usecases_synthetic.lib.domain_config import (
    USECASES_DIR,
    VALID_LEVELS,
    DomainConfig,
    data_root_for_domain,
    load_domain_config,
)
from usecases_synthetic.lib.provenance import PROVENANCE_COLUMNS, ProvenanceLog

logger = logging.getLogger(__name__)


# Provenance filenames emitted by individual knobs. The master provenance
# CSV will union all of these into ``provenance_all.csv``.
KNOB_PROVENANCE_FILES: tuple[str, ...] = (
    "knob_02_niche.csv",
    "knob_02_niche_scores.csv",
    "knob_04_coverage_skew.csv",
    "knob_04_skipped.csv",
    "knob_01_surface.csv",
    "knob_01_skipped.csv",
    "knob_05_format_unit.csv",
    "knob_05_skipped.csv",
    "knob_06_noise.csv",
    "knob_06_skipped.csv",
    "knob_03_attribute_drop.csv",
    "knob_03_skipped.csv",
    "knob_10_reliability.csv",
    "knob_10_compromised_mask.csv",
    "knob_08_naming.csv",
    "joint_values_audit.csv",
)

BASELINE_FILES: tuple[str, ...] = (
    "knob_01_realised.csv",
    "knob_02_realised.csv",
    "knob_03_baseline_missingness.csv",
    "knob_04_baseline_coverage.csv",
    "knob_04_realized_vs_target.csv",
    "knob_10_baseline_alignment.csv",
    "knob_10_realised.csv",
    "knob_10_gold_hash.txt",
)


def default_variant_dir(domain: str, level: str) -> Path:
    """Return the default variant directory for ``<domain>`` / ``<level>``.

    Parameters
    ----------
    domain : str
        Domain name.
    level : str
        Difficulty level.

    Returns
    -------
    Path
        ``usecases/<domain>-augmented/<level>`` under the repo root.
    """
    # Augmented outputs always land at ``usecases/<domain>-augmented/<level>``
    # for cross-domain consistency; the per-domain ``data_root`` override
    # applies only to *input* data (see ``variant_loader._variant_root``
    # for the symmetric read path).
    return USECASES_DIR / f"{domain}-augmented" / level


def default_work_dir(domain: str, level: str) -> Path:
    """Return the default orchestrator work directory.

    Parameters
    ----------
    domain : str
        Domain name.
    level : str
        Difficulty level.

    Returns
    -------
    Path
        ``usecases_synthetic/output/<domain>/<level>`` under the repo root.
    """
    return REPO_ROOT / "usecases_synthetic" / "output" / domain / level


# ---------------------------------------------------------------------------
# Source serialisation
# ---------------------------------------------------------------------------


def write_sources_as_csv(
    sources: dict[str, pd.DataFrame],
    data_dir: Path,
) -> list[Path]:
    """Serialise each augmented source as CSV to ``data_dir``.

    The original upstream formats (XML, JSON, TSV) are collapsed to CSV
    so the variant can be loaded via :func:`PyDI.io.load_csv` without
    special reader kwargs. ``DataFrame.attrs["dataset_name"]`` is set on
    the input frames; the on-disk filename uses the same source label.

    Parameters
    ----------
    sources : dict[str, DataFrame]
        Augmented source DataFrames keyed by source name.
    data_dir : Path
        Destination directory. Created if absent.

    Returns
    -------
    list of Path
        The written file paths (one per source).
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, df in sources.items():
        out = data_dir / f"{name}.csv"
        df.to_csv(out, index=False)
        written.append(out)
        logger.info(
            "Wrote augmented source %r to %s (%d rows x %d cols)",
            name,
            out,
            len(df),
            df.shape[1],
        )
    return written


# ---------------------------------------------------------------------------
# Copy helpers
# ---------------------------------------------------------------------------


def _copy_if_exists(src: Path, dst: Path) -> bool:
    """Copy ``src`` to ``dst`` if the source exists. Returns True on copy."""
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def copy_original_em_correspondences(
    domain_config: DomainConfig,
    em_out_dir: Path,
) -> list[str]:
    """Copy the original EM correspondence CSVs into the variant directory.

    The original gold (``*_train.csv``, ``*_val.csv``, ``*_test.csv``,
    ``*_all.csv``) is preserved unchanged. The regenerated test set
    (from Knob 2) is copied separately by :func:`copy_regenerated_em`.

    Parameters
    ----------
    domain_config : DomainConfig
        Parsed domain config (for the source EM directory).
    em_out_dir : Path
        Destination directory.

    Returns
    -------
    list of str
        Filenames copied.
    """
    em_out_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    src_em = domain_config.em_dir()
    if not src_em.exists():
        logger.warning("EM source dir missing: %s", src_em)
        return copied
    for csv_path in sorted(src_em.glob("*.csv")):
        dest = em_out_dir / csv_path.name
        shutil.copy2(csv_path, dest)
        copied.append(csv_path.name)
    logger.info("Copied %d original EM correspondence files", len(copied))
    return copied


def copy_regenerated_em(
    work_dir: Path,
    em_out_dir: Path,
) -> list[str]:
    """Copy Knob 2's per-pair per-split regenerated EM files if present.

    Looks for files matching ``*_{train,val,test}_{baseline_pruned,
    corner_filled}.csv`` in the orchestrator work directory's
    entitymatching folder and copies each into the variant output
    directory. Since C11 (plan_revision.md, 2026-05-22) the regen writer
    emits two parallel versions per (pair, split) — ``baseline_pruned``
    (Set 1, survivors only) and ``corner_filled`` (Set 2, survivors +
    corner backfill) — named ``<src1>_2_<src2>_<split>_<version>.csv`` by
    :func:`generate_variant.write_regenerated_em_splits`. The per-pair
    per-split shape mirrors the original EM gold so downstream consumers
    can treat the variant as a drop-in benchmark replacement.

    R10-F (2026-05-29): this previously globbed ``*_regenerated.csv`` —
    a suffix the C11 writer never emits — so the regen files silently
    never reached the variant directory and every dual-test surface fell
    back to the baseline gold.

    Parameters
    ----------
    work_dir : Path
        Orchestrator work directory containing
        ``input/entitymatching/*_{baseline_pruned,corner_filled}.csv``
        files.
    em_out_dir : Path
        Variant entitymatching directory.

    Returns
    -------
    list of str
        Copied filenames.
    """
    src_dir = work_dir / "input" / "entitymatching"
    copied: list[str] = []
    if not src_dir.exists():
        logger.info("No regenerated EM source dir at %s (skipping)", src_dir)
        return copied
    patterns = ("*_baseline_pruned.csv", "*_corner_filled.csv")
    seen: set[str] = set()
    for pattern in patterns:
        for path in sorted(src_dir.glob(pattern)):
            if path.name in seen:
                continue
            seen.add(path.name)
            dst = em_out_dir / path.name
            shutil.copy2(path, dst)
            copied.append(path.name)
    if copied:
        logger.info(
            "Copied %d regenerated EM split file(s) to %s",
            len(copied),
            em_out_dir,
        )
    else:
        logger.info("No regenerated EM split files in %s (skipping)", src_dir)
    return copied


def copy_fusion_gold(
    domain_config: DomainConfig,
    fusion_out_dir: Path,
) -> list[str]:
    """Copy the original fusion gold files unchanged.

    Parameters
    ----------
    domain_config : DomainConfig
        Parsed domain config.
    fusion_out_dir : Path
        Variant fusion directory.

    Returns
    -------
    list of str
        Filenames copied.
    """
    fusion_out_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    src_fusion = domain_config.fusion_dir()
    if not src_fusion.exists():
        logger.warning("Fusion source dir missing: %s", src_fusion)
        return copied
    # Resolve the configured source filenames (e.g. ``*_set_final.xml``
    # for games + music) but always emit the canonical
    # ``test_set.xml`` / ``validation_set.xml`` names into the variant
    # so ``variant_loader._load_fusion`` and downstream notebooks read a
    # stable filename regardless of source-side naming drift.
    canonical_targets = {
        "validation": "validation_set.xml",
        "test": "test_set.xml",
    }
    for key, dst_name in canonical_targets.items():
        src = src_fusion / domain_config.fusion_files[key]
        if src.exists():
            shutil.copy2(src, fusion_out_dir / dst_name)
            copied.append(dst_name)
    logger.info("Copied %d fusion gold files", len(copied))
    return copied


def copy_schemamatching(
    domain_config: DomainConfig,
    work_dir: Path,
    sm_out_dir: Path,
) -> list[str]:
    """Assemble the schema matching directory.

    Copies:

    - The original ``target_schema.json`` (unchanged).
    - ``sm_mapping.csv`` produced by Knob 8 in the work directory.

    Parameters
    ----------
    domain_config : DomainConfig
        Parsed domain config.
    work_dir : Path
        Orchestrator work directory.
    sm_out_dir : Path
        Variant SM directory.

    Returns
    -------
    list of str
        Filenames copied.
    """
    sm_out_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []

    # Original target schema, if present.
    src_sm = (
        (data_root_for_domain(domain_config.domain) or USECASES_DIR)
        / domain_config.domain
        / "input"
        / "schemamatching"
    )
    target_schema = src_sm / "target_schema.json"
    if _copy_if_exists(target_schema, sm_out_dir / "target_schema.json"):
        copied.append("target_schema.json")

    # K8 mapping.
    k8_mapping = work_dir / "input" / "schemamatching" / "sm_mapping.csv"
    if _copy_if_exists(k8_mapping, sm_out_dir / "sm_mapping.csv"):
        copied.append("sm_mapping.csv")
    else:
        logger.warning(
            "Knob 8 SM mapping missing at %s (K8 may not have run)",
            k8_mapping,
        )
    logger.info("SM directory populated with %d files", len(copied))
    return copied


# ---------------------------------------------------------------------------
# Provenance consolidation
# ---------------------------------------------------------------------------


def consolidate_provenance(
    work_dir: Path,
    variant_dir: Path,
) -> pd.DataFrame:
    """Copy per-knob provenance CSVs and build a master ``provenance_all.csv``.

    Parameters
    ----------
    work_dir : Path
        Orchestrator work directory whose ``output/provenance/`` contains
        per-knob CSVs.
    variant_dir : Path
        Target variant directory. Provenance goes under
        ``output/provenance/`` inside this directory.

    Returns
    -------
    pandas.DataFrame
        The merged provenance (standard-schema rows only — audit/score
        CSVs are copied but not concatenated).
    """
    src_prov = work_dir / "output" / "provenance"
    dst_prov = variant_dir / "output" / "provenance"
    dst_prov.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    to_merge: list[Path] = []
    for name in KNOB_PROVENANCE_FILES:
        src = src_prov / name
        if not src.exists():
            continue
        dst = dst_prov / name
        shutil.copy2(src, dst)
        copied.append(name)
        # Only concatenate CSVs that follow the standard provenance
        # schema. Audit + score + skipped files carry auxiliary columns.
        if name in {
            "knob_02_niche.csv",
            "knob_04_coverage_skew.csv",
            "knob_01_surface.csv",
            "knob_05_format_unit.csv",
            "knob_06_noise.csv",
            "knob_03_attribute_drop.csv",
            "knob_10_reliability.csv",
            "knob_08_naming.csv",
        }:
            to_merge.append(dst)

    logger.info("Copied %d provenance files", len(copied))

    merged = ProvenanceLog.merge(to_merge, dst_prov / "provenance_all.csv")
    logger.info("Master provenance: %d rows", len(merged))
    return merged


def consolidate_baselines(
    work_dir: Path,
    variant_dir: Path,
) -> list[str]:
    """Copy baseline measurement files produced by K3/K4/K10.

    Parameters
    ----------
    work_dir : Path
        Orchestrator work directory.
    variant_dir : Path
        Target variant directory.

    Returns
    -------
    list of str
        Filenames copied.
    """
    src_base = work_dir / "output" / "baselines"
    dst_base = variant_dir / "output" / "baselines"
    dst_base.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    if not src_base.exists():
        return copied
    for name in BASELINE_FILES:
        if _copy_if_exists(src_base / name, dst_base / name):
            copied.append(name)
    logger.info("Copied %d baseline files", len(copied))
    return copied


# ---------------------------------------------------------------------------
# difficulty.yaml
# ---------------------------------------------------------------------------


def write_difficulty_yaml(
    variant_dir: Path,
    summary: dict[str, Any],
) -> Path:
    """Write the ``config/difficulty.yaml`` summary for this variant.

    Parameters
    ----------
    variant_dir : Path
        Target variant directory.
    summary : dict
        Arbitrary summary dict (domain, level, master_seed, per-knob
        parameters, generated_at, knob order).

    Returns
    -------
    Path
        The written YAML path.
    """
    cfg_dir = variant_dir / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    out = cfg_dir / "difficulty.yaml"
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, sort_keys=False, default_flow_style=False)
    logger.info("Wrote %s", out)
    return out


# ---------------------------------------------------------------------------
# Top-level packager
# ---------------------------------------------------------------------------


def package_variant(
    domain: str,
    level: str,
    sources: dict[str, pd.DataFrame],
    work_dir: Path,
    variant_dir: Path,
    difficulty_summary: dict[str, Any],
) -> dict[str, Any]:
    """Assemble a full variant directory from in-memory sources + work-dir artifacts.

    Parameters
    ----------
    domain : str
        Domain name.
    level : str
        Difficulty level.
    sources : dict[str, DataFrame]
        Final augmented source DataFrames (post-K8).
    work_dir : Path
        Orchestrator work directory containing per-knob provenance,
        baselines, SM mapping, and the regenerated EM test set.
    variant_dir : Path
        Output variant directory.
    difficulty_summary : dict
        Summary written to ``config/difficulty.yaml`` verbatim.

    Returns
    -------
    dict
        Keys: ``data_files``, ``em_files``, ``fusion_files``,
        ``sm_files``, ``provenance_rows``, ``difficulty_yaml``.
    """
    # The level string is validated by upstream callers (generate_variant
    # and the standalone CLI in this module). Ablation mode deliberately
    # uses labels like ``ablation_knob_08`` that are not in VALID_LEVELS,
    # so we accept any non-empty string here.
    if not isinstance(level, str) or not level:
        raise ValueError(f"Invalid level: {level!r}")

    domain_config = load_domain_config(domain)

    # input/data — serialise augmented sources.
    data_files = write_sources_as_csv(sources, variant_dir / "input" / "data")

    # input/schemamatching — target schema + K8 mapping.
    sm_files = copy_schemamatching(
        domain_config, work_dir, variant_dir / "input" / "schemamatching"
    )

    # input/entitymatching — original gold + K2 regenerated per-pair
    # per-split files (train/val/test).
    em_out = variant_dir / "input" / "entitymatching"
    em_files = copy_original_em_correspondences(domain_config, em_out)
    em_files.extend(copy_regenerated_em(work_dir, em_out))

    # input/fusion — unchanged copy.
    fusion_files = copy_fusion_gold(domain_config, variant_dir / "input" / "fusion")

    # output/provenance + output/baselines
    prov_df = consolidate_provenance(work_dir, variant_dir)
    consolidate_baselines(work_dir, variant_dir)

    # config/difficulty.yaml
    yaml_path = write_difficulty_yaml(variant_dir, difficulty_summary)

    return {
        "data_files": [p.name for p in data_files],
        "em_files": em_files,
        "fusion_files": fusion_files,
        "sm_files": sm_files,
        "provenance_rows": len(prov_df),
        "difficulty_yaml": str(yaml_path),
    }


# ---------------------------------------------------------------------------
# Stand-alone CLI
# ---------------------------------------------------------------------------


def _load_sources_from_work_dir(domain: str, work_dir: Path) -> dict[str, pd.DataFrame]:
    """Load augmented source DataFrames from a work directory.

    Looks under ``<work_dir>/input/data/<source>.csv``. Falls back to
    the original use case data directory when the work directory does
    not hold per-source serialisations (this is normal because the
    orchestrator passes sources in-memory and only the final packager
    serialises them).

    Parameters
    ----------
    domain : str
        Domain name.
    work_dir : Path
        Orchestrator work directory.

    Returns
    -------
    dict[str, DataFrame]
        Source DataFrames keyed by source name, with
        ``attrs["dataset_name"]`` set.
    """
    domain_config = load_domain_config(domain)
    data_dir = work_dir / "input" / "data"
    sources: dict[str, pd.DataFrame] = {}
    for spec in domain_config.sources:
        csv_path = data_dir / f"{spec.name}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
        else:
            # Fall back to the original source via the standard loader.
            from usecases_synthetic.lib.loaders import load_source

            df = load_source(
                domain=domain,
                source_name=spec.name,
                source_file=spec.file,
                source_format=spec.format,
                reader_kwargs=spec.reader_kwargs,
            )
        df.attrs["dataset_name"] = spec.name
        sources[spec.name] = df
    return sources


def main() -> None:
    """CLI entry point: re-package from an existing work directory."""
    parser = argparse.ArgumentParser(
        description="Package an S1 variant directory from a work directory.",
    )
    parser.add_argument("--domain", required=True)
    parser.add_argument("--level", required=True, choices=VALID_LEVELS)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Orchestrator work directory "
        "(default: usecases_synthetic/output/<domain>/<level>)",
    )
    parser.add_argument(
        "--variant-dir",
        type=Path,
        default=None,
        help="Variant output directory "
        "(default: usecases/<domain>-augmented/<level>)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    domain: str = args.domain
    level: str = args.level
    work_dir: Path = args.work_dir or default_work_dir(domain, level)
    variant_dir: Path = args.variant_dir or default_variant_dir(domain, level)

    logger.info(
        "Packaging variant: domain=%s level=%s work_dir=%s variant_dir=%s",
        domain,
        level,
        work_dir,
        variant_dir,
    )

    sources = _load_sources_from_work_dir(domain, work_dir)

    summary: dict[str, Any] = {
        "domain": domain,
        "level": level,
        "source": "package_variant_standalone",
        "note": (
            "difficulty.yaml written by the standalone packager — the "
            "orchestrator overwrites this with the full knob parameter "
            "summary when called via generate_variant.py."
        ),
    }
    package_variant(
        domain=domain,
        level=level,
        sources=sources,
        work_dir=work_dir,
        variant_dir=variant_dir,
        difficulty_summary=summary,
    )
    logger.info("Variant packaging complete: %s", variant_dir)


if __name__ == "__main__":
    main()
