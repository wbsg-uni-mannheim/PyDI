"""Tests for Knob 08 — Schema Naming Divergence.

Acceptance criteria (from module_01_knob_08.md):
1. Companies easy: all 3 sources have descriptive headers matching target_schema.json
2. Companies hard: >=1 source fully anonymized; no source descriptive
3. Provenance CSV has one row per renamed column with valid transform_fn
4. SM mapping file regenerated with updated LHS column names, RHS unchanged
5. Cell values unchanged: diff of all values before/after is empty
6. pytest passes
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from usecases_synthetic.scripts.apply_knob_08_naming import (
    VALID_TRANSFORM_FNS,
    _infer_baseline_rung,
    apply_knob_08,
    load_knob_08_config,
    write_outputs,
)

# ---- Fixtures ---------------------------------------------------------------


@pytest.fixture
def companies_config() -> dict[str, Any]:
    """Load the real companies Knob 08 config."""
    return load_knob_08_config("companies")


@pytest.fixture
def small_sources() -> dict[str, pd.DataFrame]:
    """Small DataFrames with refreshed companies column names (post-loader)."""
    dbpedia = pd.DataFrame(
        {
            "id": ["db_1", "db_2", "db_3"],
            "org_name": ["Acme", "Globex", "Initech"],
            "established": ["1950", "1961", "1999"],
            "nation": ["United States", "Germany", "Japan"],
            "headquarters": ["New York", "Berlin", "Tokyo"],
            "sector": ["Tech", "Energy", "Finance"],
            "keypeople_name": ["J. Doe", "H. Muller", "T. Sato"],
            "total_assets_val": [1000, 2000, 3000],
            "annual_income": [500, 600, 700],
        }
    )
    dbpedia.attrs["dataset_name"] = "dbpedia"

    forbes = pd.DataFrame(
        {
            "id": ["fb_1", "fb_2", "fb_3"],
            "company": ["Acme Corp", "Globex Inc", "Initech LLC"],
            "url": ["acme.com", "globex.com", "initech.com"],
            "region": ["USA", "DEU", "JPN"],
            "business_segment": ["Technology", "Energy", "Financial"],
            "asset_value": [100.0, 200.0, 150.0],
            "sales_figure": [10.0, 20.0, 15.0],
        }
    )
    forbes.attrs["dataset_name"] = "forbes"

    fullcontact = pd.DataFrame(
        {
            "id": ["fc_1", "fc_2", "fc_3"],
            "Attribute_2": ["Acme", "Globex", "Initech"],
            "Attribute_3": ["United States", "Germany", "Japan"],
            "Attribute_4": ["New York", "Berlin", "Tokyo"],
            "Attribute_5": ["Tech", "Energy", "Finance"],
            "Attribute_6": ["1950", "1961", "1999"],
        }
    )
    fullcontact.attrs["dataset_name"] = "fullcontact"

    return {"dbpedia": dbpedia, "forbes": forbes, "fullcontact": fullcontact}


# ---- Config loading tests ---------------------------------------------------


class TestLoadConfig:
    """Config loading and structure."""

    def test_loads_companies(self, companies_config: dict[str, Any]) -> None:
        assert "rename_table" in companies_config
        assert "level_assignments" in companies_config
        assert "sm_mapping" in companies_config

    def test_all_sources_present(self, companies_config: dict[str, Any]) -> None:
        for section in ("rename_table", "sm_mapping"):
            for source in ("dbpedia", "forbes", "fullcontact"):
                assert (
                    source in companies_config[section]
                ), f"{source} missing from {section}"

    def test_all_levels_assigned(self, companies_config: dict[str, Any]) -> None:
        for level in ("easy", "medium", "hard"):
            assignments = companies_config["level_assignments"][level]
            for source in ("dbpedia", "forbes", "fullcontact"):
                assert (
                    source in assignments
                ), f"{source} missing from level_assignments[{level}]"

    def test_four_rungs_per_column(self, companies_config: dict[str, Any]) -> None:
        expected_rungs = {"descriptive", "abbreviated", "cryptic", "anonymized"}
        for source, cols in companies_config["rename_table"].items():
            for col, rungs in cols.items():
                assert (
                    set(rungs.keys()) == expected_rungs
                ), f"{source}.{col} rungs: {set(rungs.keys())}"

    def test_no_rung_collisions_within_source(
        self, companies_config: dict[str, Any]
    ) -> None:
        """No two original columns map to the same name at any rung."""
        for source, cols in companies_config["rename_table"].items():
            for rung in ("descriptive", "abbreviated", "cryptic", "anonymized"):
                names = [rungs[rung] for rungs in cols.values()]
                assert len(names) == len(
                    set(names)
                ), f"Collision in {source} at rung {rung}: {names}"

    def test_missing_domain_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_knob_08_config("nonexistent_domain")


# ---- Easy level tests -------------------------------------------------------


class TestEasyLevel:
    """At easy, all sources should have descriptive (target-like) headers."""

    def test_dbpedia_descriptive_headers(
        self,
        small_sources: dict[str, pd.DataFrame],
        companies_config: dict[str, Any],
    ) -> None:
        """At easy, all non-id mapped columns use target schema names.

        Id columns are exempt — they keep their original header.
        """
        renamed, _, _ = apply_knob_08(
            "companies", "easy", small_sources, companies_config
        )
        db_cols = set(renamed["dbpedia"].columns)
        target_cols = {
            target
            for target in companies_config["sm_mapping"]["dbpedia"].values()
            if target != "id"
        }
        assert target_cols.issubset(db_cols)

    def test_forbes_descriptive_headers(
        self,
        small_sources: dict[str, pd.DataFrame],
        companies_config: dict[str, Any],
    ) -> None:
        renamed, _, _ = apply_knob_08(
            "companies", "easy", small_sources, companies_config
        )
        f_cols = set(renamed["forbes"].columns)
        target_cols = {
            target
            for target in companies_config["sm_mapping"]["forbes"].values()
            if target != "id"
        }
        assert target_cols.issubset(f_cols)

    def test_fullcontact_descriptive_headers(
        self,
        small_sources: dict[str, pd.DataFrame],
        companies_config: dict[str, Any],
    ) -> None:
        renamed, _, _ = apply_knob_08(
            "companies", "easy", small_sources, companies_config
        )
        fc_cols = set(renamed["fullcontact"].columns)
        target_cols = {
            target
            for target in companies_config["sm_mapping"]["fullcontact"].values()
            if target != "id"
        }
        assert target_cols.issubset(fc_cols)

    def test_sm_mapping_at_easy(
        self,
        small_sources: dict[str, pd.DataFrame],
        companies_config: dict[str, Any],
    ) -> None:
        """At easy, SM source_column should equal target_column for all mapped cols.

        Id columns (target == 'id') are exempt: they are never renamed,
        so a source's id column keeps its original header even at easy.
        """
        renamed, sm_df, _ = apply_knob_08(
            "companies", "easy", small_sources, companies_config
        )
        for _, row in sm_df.iterrows():
            if row["target_column"] == "id":
                continue
            assert row["source_column"] == row["target_column"], (
                f"{row['source_dataset']}: {row['source_column']} != "
                f"{row['target_column']}"
            )


# ---- Hard level tests -------------------------------------------------------


class TestHardLevel:
    """At hard, >=1 source anonymized; no source descriptive."""

    def test_at_least_one_anonymized(
        self,
        small_sources: dict[str, pd.DataFrame],
        companies_config: dict[str, Any],
    ) -> None:
        assignments = companies_config["level_assignments"]["hard"]
        assert any(
            rung == "anonymized" for rung in assignments.values()
        ), "No source at anonymized rung at hard level"

    def test_no_descriptive_at_hard(
        self,
        small_sources: dict[str, pd.DataFrame],
        companies_config: dict[str, Any],
    ) -> None:
        assignments = companies_config["level_assignments"]["hard"]
        assert all(
            rung != "descriptive" for rung in assignments.values()
        ), "A source is still descriptive at hard level"

    def test_fullcontact_anonymized_headers(
        self,
        small_sources: dict[str, pd.DataFrame],
        companies_config: dict[str, Any],
    ) -> None:
        """All non-id fullcontact columns anonymized at hard.

        The id column (fullcontact.id) is exempt from renaming to keep
        downstream stages (EM correspondences, fusion id_column lookup)
        stable.
        """
        renamed, _, _ = apply_knob_08(
            "companies", "hard", small_sources, companies_config
        )
        fc_cols = list(renamed["fullcontact"].columns)
        id_cols = {
            col
            for col, target in companies_config["sm_mapping"]
            .get("fullcontact", {})
            .items()
            if target == "id"
        }
        non_id_cols = [c for c in fc_cols if c not in id_cols]
        attribute_cols = [c for c in non_id_cols if c.startswith("Attribute_")]
        assert len(attribute_cols) == len(
            non_id_cols
        ), f"Expected all non-id columns anonymized, got {fc_cols}"


# ---- Provenance tests -------------------------------------------------------


class TestProvenance:
    """Provenance rows are correct and complete."""

    def test_provenance_non_empty_at_easy(
        self,
        small_sources: dict[str, pd.DataFrame],
        companies_config: dict[str, Any],
    ) -> None:
        _, _, prov = apply_knob_08("companies", "easy", small_sources, companies_config)
        assert len(prov) > 0, "Expected provenance rows at easy level"

    def test_provenance_valid_transform_fns(
        self,
        small_sources: dict[str, pd.DataFrame],
        companies_config: dict[str, Any],
    ) -> None:
        for level in ("easy", "medium", "hard"):
            _, _, prov = apply_knob_08(
                "companies", level, small_sources, companies_config
            )
            fns = set(prov["transform_fn"].unique())
            assert fns.issubset(
                VALID_TRANSFORM_FNS
            ), f"Invalid transform_fn at {level}: {fns - VALID_TRANSFORM_FNS}"

    def test_provenance_transform_params_json(
        self,
        small_sources: dict[str, pd.DataFrame],
        companies_config: dict[str, Any],
    ) -> None:
        _, _, prov = apply_knob_08("companies", "easy", small_sources, companies_config)
        for _, row in prov.iterrows():
            params = json.loads(row["transform_params"])
            assert "baseline_rung" in params
            assert "target_rung" in params
            assert "oracle" in params

    def test_provenance_row_count_matches_renames(
        self,
        small_sources: dict[str, pd.DataFrame],
        companies_config: dict[str, Any],
    ) -> None:
        """Provenance rows = total number of actually-renamed columns."""
        for level in ("easy", "medium", "hard"):
            renamed, _, prov = apply_knob_08(
                "companies", level, small_sources, companies_config
            )
            # Count columns that changed name.
            actual_renames = 0
            for src_name, df_orig in small_sources.items():
                df_new = renamed[src_name]
                for old_c, new_c in zip(df_orig.columns, df_new.columns):
                    if old_c != new_c:
                        actual_renames += 1
            assert len(prov) == actual_renames, (
                f"level={level}: prov rows ({len(prov)}) != "
                f"actual renames ({actual_renames})"
            )


# ---- Value preservation tests -----------------------------------------------


class TestValuePreservation:
    """Cell values must not change — only column headers."""

    def test_values_unchanged_all_levels(
        self,
        small_sources: dict[str, pd.DataFrame],
        companies_config: dict[str, Any],
    ) -> None:
        for level in ("easy", "medium", "hard"):
            renamed, _, _ = apply_knob_08(
                "companies", level, small_sources, companies_config
            )
            for src_name in small_sources:
                orig = small_sources[src_name]
                new = renamed[src_name]
                # Compare values ignoring column names.
                assert orig.shape == new.shape, f"{src_name} shape changed at {level}"
                for i in range(orig.shape[1]):
                    pd.testing.assert_series_equal(
                        orig.iloc[:, i].reset_index(drop=True),
                        new.iloc[:, i].reset_index(drop=True),
                        check_names=False,
                    )

    def test_attrs_preserved(
        self,
        small_sources: dict[str, pd.DataFrame],
        companies_config: dict[str, Any],
    ) -> None:
        for level in ("easy", "medium", "hard"):
            renamed, _, _ = apply_knob_08(
                "companies", level, small_sources, companies_config
            )
            for src_name in small_sources:
                assert renamed[src_name].attrs["dataset_name"] == src_name


# ---- SM mapping tests -------------------------------------------------------


class TestSMMapping:
    """Regenerated SM mapping is consistent with renamed columns."""

    def test_sm_mapping_columns(
        self,
        small_sources: dict[str, pd.DataFrame],
        companies_config: dict[str, Any],
    ) -> None:
        _, sm_df, _ = apply_knob_08(
            "companies", "easy", small_sources, companies_config
        )
        assert set(sm_df.columns) == {
            "source_dataset",
            "source_column",
            "target_dataset",
            "target_column",
            "score",
        }

    def test_sm_target_columns_unchanged(
        self,
        small_sources: dict[str, pd.DataFrame],
        companies_config: dict[str, Any],
    ) -> None:
        """RHS (target) column names are the same across all levels."""
        mappings = {}
        for level in ("easy", "medium", "hard"):
            _, sm_df, _ = apply_knob_08(
                "companies", level, small_sources, companies_config
            )
            mappings[level] = set(sm_df["target_column"].unique())

        assert mappings["easy"] == mappings["medium"] == mappings["hard"]

    def test_sm_source_columns_exist_in_renamed_df(
        self,
        small_sources: dict[str, pd.DataFrame],
        companies_config: dict[str, Any],
    ) -> None:
        """Every source_column in the SM mapping exists in the renamed DataFrame."""
        for level in ("easy", "medium", "hard"):
            renamed, sm_df, _ = apply_knob_08(
                "companies", level, small_sources, companies_config
            )
            for _, row in sm_df.iterrows():
                src = row["source_dataset"]
                col = row["source_column"]
                assert col in renamed[src].columns, (
                    f"SM mapping references {src}.{col} at {level}, "
                    f"but columns are {list(renamed[src].columns)}"
                )

    def test_sm_row_count(
        self,
        small_sources: dict[str, pd.DataFrame],
        companies_config: dict[str, Any],
    ) -> None:
        """SM mapping has the correct number of rows (sum of mapped columns)."""
        expected = sum(len(cols) for cols in companies_config["sm_mapping"].values())
        _, sm_df, _ = apply_knob_08(
            "companies", "easy", small_sources, companies_config
        )
        assert len(sm_df) == expected


# ---- Output writing tests ---------------------------------------------------


class TestWriteOutputs:
    """Artifacts land on disk correctly."""

    def test_write_creates_files(
        self,
        small_sources: dict[str, pd.DataFrame],
        companies_config: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        _, sm_df, prov_df = apply_knob_08(
            "companies", "easy", small_sources, companies_config
        )
        write_outputs(sm_df, prov_df, tmp_path)
        assert (tmp_path / "input" / "schemamatching" / "sm_mapping.csv").exists()
        assert (tmp_path / "output" / "provenance" / "knob_08_naming.csv").exists()

    def test_written_sm_round_trips(
        self,
        small_sources: dict[str, pd.DataFrame],
        companies_config: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        _, sm_df, prov_df = apply_knob_08(
            "companies", "easy", small_sources, companies_config
        )
        write_outputs(sm_df, prov_df, tmp_path)
        loaded = pd.read_csv(tmp_path / "input" / "schemamatching" / "sm_mapping.csv")
        pd.testing.assert_frame_equal(sm_df, loaded)


# ---- Per-source override tests ----------------------------------------------


class TestOverrides:
    """Per-source rung overrides work."""

    def test_override_single_source(
        self,
        small_sources: dict[str, pd.DataFrame],
        companies_config: dict[str, Any],
    ) -> None:
        renamed, _, _ = apply_knob_08(
            "companies",
            "easy",
            small_sources,
            companies_config,
            per_source_override={"forbes": "anonymized"},
        )
        # Forbes non-id columns should all be Attribute_N
        # (id is never renamed by design).
        fc = list(renamed["forbes"].columns)
        non_id = [c for c in fc if c != "id"]
        attribute_cols = [c for c in non_id if c.startswith("Attribute_")]
        assert len(attribute_cols) == len(non_id)


# ---- Edge case tests --------------------------------------------------------


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_invalid_level_raises(
        self,
        small_sources: dict[str, pd.DataFrame],
        companies_config: dict[str, Any],
    ) -> None:
        with pytest.raises(ValueError, match="Invalid level"):
            apply_knob_08(
                "companies", "extreme", small_sources, companies_config  # type: ignore[arg-type]
            )

    def test_unknown_source_identity_pass(
        self,
        companies_config: dict[str, Any],
    ) -> None:
        """A source not in the rename table passes through unchanged."""
        extra = pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]})
        extra.attrs["dataset_name"] = "unknown_source"
        sources = {"unknown_source": extra}
        renamed, sm_df, prov = apply_knob_08(
            "companies", "easy", sources, companies_config
        )
        pd.testing.assert_frame_equal(renamed["unknown_source"], extra)
        assert len(prov) == 0

    def test_infer_baseline_rung_exact_match(self) -> None:
        rungs = {
            "descriptive": "country",
            "abbreviated": "ctry",
            "cryptic": "geo_1",
            "anonymized": "Attribute_4",
        }
        assert _infer_baseline_rung("country", rungs) == "descriptive"
        assert _infer_baseline_rung("ctry", rungs) == "abbreviated"
        assert _infer_baseline_rung("Attribute_4", rungs) == "anonymized"

    def test_infer_baseline_rung_no_match(self) -> None:
        rungs = {
            "descriptive": "country",
            "abbreviated": "ctry",
            "cryptic": "geo_1",
            "anonymized": "Attribute_4",
        }
        assert _infer_baseline_rung("some_other_name", rungs) == "original"
