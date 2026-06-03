"""Regression guard: every per-knob YAML's ``id_columns`` and column
references must match the post-loader-rename column convention.

Background (2026-05-07): K3, K4, K5 sign-offs shipped with broken
``id_columns`` referencing pre-rename column names (e.g. ``entity_uri``,
``forbes_url``, ``Attribute_1``, ``wiki_ref``, ``mc_id``, ``rec_id``,
``identifier``, ``rel_id``). The loader at
:func:`usecases_synthetic.lib.loaders.load_source` renames every source's
primary id column to ``id`` (search ``df.rename(columns={id_column: "id"})``
in the loader). Dispatchers silently degrade when ``id_col`` is wrong
(``if id_col and id_col in df.columns: ... else: continue``), so smoke
runs and unit tests don't catch the bug — only end-to-end functional
checks do. This test is the canonical regression guard.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from usecases_synthetic.lib.loaders import load_domain_sources

CONFIG_DIR: Path = Path(__file__).resolve().parents[1] / "config"

# Knobs whose YAMLs author per-source ``id_columns`` blocks. Knob 8 has no
# id_columns block (header-only knob). Knob 2 lives under knob_02_niche.
KNOBS_WITH_ID_COLUMNS = (
    "knob_01_surface",
    "knob_02_niche",
    "knob_03_drop",
    "knob_04_coverage",
    "knob_05_format",
    "knob_06_noise",
    "knob_10_reliability",
)

# Active S1 domains (movies + products are descoped per plan_s1_scale.md).
# papers (2026) folds in here so its per-knob YAMLs are validated against
# the actual loaded source columns (jsonl sources, dash-minted ids).
ACTIVE_DOMAINS = ("companies", "games", "music", "papers")


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict), f"{path} did not parse as a dict"
    return data


@pytest.mark.parametrize("domain", ACTIVE_DOMAINS)
def test_id_columns_match_loader_rename(domain: str) -> None:
    """Every per-knob YAML's ``id_columns`` must use ``id`` for every source.

    The loader collapses all source primary id columns to ``id``. Authoring
    raw CSV column names (e.g. ``entity_uri``, ``rec_id``, ``rel_id``)
    causes the dispatcher to silently no-op the load-bearing branch.
    """
    sources = load_domain_sources(domain)
    expected_id_col = "id"
    for source_name, df in sources.items():
        assert expected_id_col in df.columns, (
            f"Loader contract violation: source {source_name!r} for domain "
            f"{domain!r} loads without an 'id' column. Got: {list(df.columns)}"
        )

    for knob in KNOBS_WITH_ID_COLUMNS:
        cfg_path = CONFIG_DIR / knob / f"{domain}.yaml"
        if not cfg_path.exists():
            # Some knobs may not have a per-domain config for every domain.
            continue
        cfg = _load_yaml(cfg_path)
        id_columns = cfg.get("id_columns")
        if id_columns is None:
            # Knob doesn't author id_columns — skip.
            continue
        assert isinstance(
            id_columns, dict
        ), f"{cfg_path}: id_columns must be a dict, got {type(id_columns)}"
        for source_name, id_col in sorted(id_columns.items()):
            assert id_col == expected_id_col, (
                f"{cfg_path}: id_columns[{source_name!r}] = {id_col!r}; "
                f"must be {expected_id_col!r} (loader renames every source's "
                f"primary id column to 'id'). See "
                f"feedback_synth_id_columns_convention.md and "
                f"plan_s1_scale.md K3 sign-off bug + fix table."
            )


@pytest.mark.parametrize("domain", ACTIVE_DOMAINS)
def test_attribute_classes_and_mappings_reference_real_columns(
    domain: str,
) -> None:
    """Per-knob YAML attribute_classes / attribute_mapping / primary_columns
    references must point at columns that exist in the loaded DataFrames.

    Mirrors the id_columns guard: same silent-degradation pattern fires
    when these references mismatch (most dispatchers gate on
    ``col in df.columns`` and skip on miss).
    """
    sources = load_domain_sources(domain)
    cols_per_source: dict[str, set[str]] = {
        n: set(df.columns) for n, df in sources.items()
    }

    for knob in KNOBS_WITH_ID_COLUMNS:
        cfg_path = CONFIG_DIR / knob / f"{domain}.yaml"
        if not cfg_path.exists():
            continue
        cfg = _load_yaml(cfg_path)

        # attribute_classes / attribute_mapping / primary_columns all use
        # the source -> {col: …} shape. Validate every key is a real column.
        for block_name in (
            "attribute_classes",
            "attribute_mapping",
        ):
            block = cfg.get(block_name)
            if not isinstance(block, dict):
                continue
            for source_name, col_map in block.items():
                if not isinstance(col_map, dict):
                    continue
                if source_name not in cols_per_source:
                    continue
                source_cols = cols_per_source[source_name]
                for col in col_map.keys():
                    assert col in source_cols, (
                        f"{cfg_path}: {block_name}[{source_name!r}] "
                        f"references column {col!r} which does not exist "
                        f"in the loaded DataFrame. Available columns: "
                        f"{sorted(source_cols)}"
                    )

        # primary_columns is source -> col (scalar value, not nested).
        primary_columns = cfg.get("primary_columns")
        if isinstance(primary_columns, dict):
            for source_name, col in primary_columns.items():
                if source_name not in cols_per_source:
                    continue
                if not isinstance(col, str):
                    continue
                assert col in cols_per_source[source_name], (
                    f"{cfg_path}: primary_columns[{source_name!r}] = "
                    f"{col!r} which does not exist in the loaded DataFrame. "
                    f"Available columns: {sorted(cols_per_source[source_name])}"
                )
