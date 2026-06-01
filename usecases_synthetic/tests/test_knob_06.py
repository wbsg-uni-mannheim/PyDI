"""Tests for Knob 06 — Value Noise Injection.

Acceptance criteria (from module_04_knob_06.md):
1. Each of 5 operators produces visibly corrupted output on 100 test values
2. For every fusion-gold entity, >=1 source retains clean primary even at hard
3. Collision index correctly skips cells with prior K1/K5 provenance
4. K4-fabricated cells ARE corrupted (not skipped)
5. Determinism: same seed + same input = identical output
6. pytest passes
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from usecases_synthetic.lib.collision_index import CollisionIndex
from usecases_synthetic.lib.noise_operators import (
    OPERATOR_REGISTRY,
    VALID_TRANSFORM_FNS,
    case_corrupt,
    ocr_confuse,
    truncate,
    typo_substitute,
    whitespace_corrupt,
)
from usecases_synthetic.lib.provenance import PROVENANCE_COLUMNS, ProvenanceLog
from usecases_synthetic.scripts.apply_knob_06_noise import (
    SKIPPED_COLUMNS,
    SkippedLog,
    apply_knob_06,
    load_knob_06_config,
    write_outputs,
)

# ---- Fixtures ---------------------------------------------------------------


@pytest.fixture
def companies_config() -> dict[str, Any]:
    """Load the real companies Knob 06 config."""
    return load_knob_06_config("companies")


@pytest.fixture
def small_sources() -> dict[str, pd.DataFrame]:
    """Small DataFrames with companies-like schema for noise testing."""
    rng = np.random.default_rng(99)
    n = 20

    names = [f"Company_{i}" for i in range(n)]
    countries = rng.choice(
        ["United States", "Germany", "Japan", "China", "Brazil"], size=n
    ).tolist()
    cities = rng.choice(
        ["New York", "Berlin", "Tokyo", "Beijing", "Sao Paulo"], size=n
    ).tolist()

    dbpedia = pd.DataFrame(
        {
            "identifier": [f"db_{i}" for i in range(n)],
            "name": names,
            "countryName": countries,
            "cityName": cities,
            "revenue": [
                f"{rng.integers(100, 999)},{rng.integers(100, 999)},000"
                for _ in range(n)
            ],
        }
    )
    dbpedia.attrs["dataset_name"] = "dbpedia"

    forbes = pd.DataFrame(
        {
            "Identifier": [f"fb_{i}" for i in range(n)],
            "Company": [n.replace("_", " ") for n in names],
            "Country": countries,
            "Assets": [f"{rng.integers(1, 99)} billion" for _ in range(n)],
        }
    )
    forbes.attrs["dataset_name"] = "forbes"

    fullcontact = pd.DataFrame(
        {
            "id": [f"fc_{i}" for i in range(n)],
            "name": names,
            "country": countries,
            "locality": cities,
        }
    )
    fullcontact.attrs["dataset_name"] = "fullcontact"

    return {"dbpedia": dbpedia, "forbes": forbes, "fullcontact": fullcontact}


@pytest.fixture
def small_entity_groups() -> dict[str, list[tuple[str, str]]]:
    """Entity groups linking the 3 sources for entities 0-9."""
    groups: dict[str, list[tuple[str, str]]] = {}
    for i in range(10):
        groups[f"group_{i}"] = [
            ("dbpedia", f"db_{i}"),
            ("forbes", f"fb_{i}"),
            ("fullcontact", f"fc_{i}"),
        ]
    # Entities 10-19 are singletons (no cross-source link).
    for i in range(10, 20):
        groups[f"group_{i}"] = [("dbpedia", f"db_{i}")]
    return groups


@pytest.fixture
def small_config() -> dict[str, Any]:
    """Minimal config for testing the dispatcher."""
    return {
        "id_columns": {
            "dbpedia": "identifier",
            "forbes": "Identifier",
            "fullcontact": "id",
        },
        "attribute_classes": {
            "dbpedia": {
                "name": "primary",
                "countryName": "key",
                "cityName": "key",
                "revenue": "secondary",
            },
            "forbes": {
                "Company": "primary",
                "Country": "key",
                "Assets": "secondary",
            },
            "fullcontact": {
                "name": "primary",
                "country": "key",
                "locality": "key",
            },
        },
        "attribute_mapping": {
            "dbpedia": {
                "name": "name",
                "countryName": "country",
                "cityName": "city",
            },
            "forbes": {
                "Company": "name",
                "Country": "country",
            },
            "fullcontact": {
                "name": "name",
                "country": "country",
                "locality": "city",
            },
        },
        "noise_rates_per_level": {
            "easy": {"primary": 0.0, "key": 0.0, "secondary": 0.01},
            "medium": {"primary": 0.0, "key": 0.10, "secondary": 0.30},
            "hard": {"primary": 0.05, "key": 0.30, "secondary": 0.60},
        },
        "operator_mix": {
            "easy": {"whitespace_corrupt": 1.0, "case_corrupt": 1.0},
            "medium": {
                "whitespace_corrupt": 1.0,
                "case_corrupt": 1.0,
                "typo_substitute": 1.5,
                "ocr_confuse": 1.0,
            },
            "hard": {
                "whitespace_corrupt": 0.5,
                "case_corrupt": 0.5,
                "typo_substitute": 2.0,
                "ocr_confuse": 1.5,
                "truncate": 1.0,
            },
        },
        "max_edits_per_cell": {"easy": 1, "medium": 1, "hard": 3},
        "max_ocr_per_cell": {"easy": 1, "medium": 1, "hard": 2},
        "max_truncate_chars": {"easy": 1, "medium": 2, "hard": 3},
        "soft_global_primary_cap_hard": 0.35,
        "cleanup_rules": [],
    }


# ---- Individual operator tests -----------------------------------------------


class TestTypoSubstitute:
    """Tests for the typo_substitute operator."""

    def test_produces_output(self) -> None:
        rng = np.random.default_rng(42)
        result = typo_substitute("Hello World", rng, n_edits=1)
        assert result is not None
        new_val, params = result
        assert new_val != "Hello World"
        assert len(new_val) == len("Hello World")
        assert "positions" in params

    def test_multiple_edits(self) -> None:
        rng = np.random.default_rng(42)
        result = typo_substitute("Hello World", rng, n_edits=3)
        assert result is not None
        new_val, params = result
        assert len(params["positions"]) <= 3

    def test_preserves_case(self) -> None:
        rng = np.random.default_rng(42)
        result = typo_substitute("HELLO", rng, n_edits=1)
        assert result is not None
        new_val, params = result
        for i, c in enumerate(new_val):
            if i in params["positions"]:
                assert c.isupper() or c.isdigit()

    def test_too_short_returns_none(self) -> None:
        rng = np.random.default_rng(42)
        # No alphanumeric characters.
        result = typo_substitute("...", rng, n_edits=1)
        assert result is None

    def test_100_values(self) -> None:
        """Each of 100 test values is visibly corrupted."""
        rng = np.random.default_rng(42)
        values = [f"TestCompany_{i:03d}" for i in range(100)]
        corrupted_count = 0
        for v in values:
            result = typo_substitute(v, rng, n_edits=1)
            if result is not None and result[0] != v:
                corrupted_count += 1
        assert corrupted_count >= 90  # At least 90% should be corrupted.

    def test_without_adjacency(self) -> None:
        rng = np.random.default_rng(42)
        result = typo_substitute("Hello", rng, n_edits=1, use_adjacency=False)
        assert result is not None
        new_val, params = result
        assert new_val != "Hello"
        assert params["use_adjacency"] is False


class TestOcrConfuse:
    """Tests for the ocr_confuse operator."""

    def test_single_char_confusion(self) -> None:
        rng = np.random.default_rng(42)
        # "O" -> "0" is in the table.
        result = ocr_confuse("ORACLE", rng, n_chars=1)
        assert result is not None
        new_val, params = result
        assert new_val != "ORACLE"
        assert "positions" in params

    def test_pair_confusion(self) -> None:
        rng = np.random.default_rng(42)
        # "rn" -> "m" is in the table.
        result = ocr_confuse("turning", rng, n_chars=1)
        assert result is not None
        new_val, params = result
        assert new_val != "turning"

    def test_no_confusable_chars(self) -> None:
        rng = np.random.default_rng(42)
        result = ocr_confuse("yyy", rng, n_chars=1)
        assert result is None

    def test_100_values(self) -> None:
        """Test on values containing known OCR-confusable characters."""
        rng = np.random.default_rng(42)
        # Values with O, l, I, S, B — all confusable.
        values = [f"OlISB_{i:03d}" for i in range(100)]
        corrupted_count = 0
        for v in values:
            result = ocr_confuse(v, rng, n_chars=1)
            if result is not None and result[0] != v:
                corrupted_count += 1
        assert corrupted_count >= 90


class TestTruncate:
    """Tests for the truncate operator."""

    def test_basic(self) -> None:
        rng = np.random.default_rng(42)
        result = truncate("Hello World", rng, max_truncate_chars=3)
        assert result is not None
        new_val, params = result
        assert len(new_val) < len("Hello World")
        assert "Hello World".startswith(new_val)

    def test_short_value(self) -> None:
        rng = np.random.default_rng(42)
        result = truncate("ab", rng, max_truncate_chars=3)
        assert result is None  # Too short.

    def test_100_values(self) -> None:
        rng = np.random.default_rng(42)
        values = [f"LongCompanyName_{i:03d}" for i in range(100)]
        corrupted_count = 0
        for v in values:
            result = truncate(v, rng, max_truncate_chars=3)
            if result is not None and result[0] != v:
                corrupted_count += 1
        assert corrupted_count >= 95


class TestWhitespaceCorrupt:
    """Tests for the whitespace_corrupt operator."""

    def test_space_insert(self) -> None:
        rng = np.random.default_rng(42)
        # Run multiple times to get a space_insert.
        found = False
        for seed in range(100):
            r = np.random.default_rng(seed)
            result = whitespace_corrupt("Hello", r)
            if result is not None and result[1].get("sub_op") == "space_insert":
                assert " " in result[0]
                found = True
                break
        assert found, "space_insert sub-op never triggered"

    def test_space_delete(self) -> None:
        rng = np.random.default_rng(42)
        result = whitespace_corrupt("Hello World", rng)
        assert result is not None

    def test_punct_collapse(self) -> None:
        found = False
        for seed in range(100):
            r = np.random.default_rng(seed)
            result = whitespace_corrupt("New York, USA", r)
            if result is not None and result[1].get("sub_op") == "punct_collapse":
                found = True
                break
        assert found, "punct_collapse sub-op never triggered"

    def test_100_values(self) -> None:
        rng = np.random.default_rng(42)
        values = [f"City Name-{i}" for i in range(100)]
        corrupted_count = 0
        for v in values:
            result = whitespace_corrupt(v, rng)
            if result is not None and result[0] != v:
                corrupted_count += 1
        assert corrupted_count >= 80


class TestCaseCorrupt:
    """Tests for the case_corrupt operator."""

    def test_basic(self) -> None:
        rng = np.random.default_rng(42)
        result = case_corrupt("Hello", rng)
        assert result is not None
        new_val, params = result
        assert new_val != "Hello"
        assert new_val.lower() == "Hello".lower() or any(
            a != b for a, b in zip(new_val, "Hello")
        )

    def test_no_alpha(self) -> None:
        rng = np.random.default_rng(42)
        result = case_corrupt("123", rng)
        assert result is None

    def test_100_values(self) -> None:
        rng = np.random.default_rng(42)
        values = [f"Company_{i}" for i in range(100)]
        corrupted_count = 0
        for v in values:
            result = case_corrupt(v, rng)
            if result is not None and result[0] != v:
                corrupted_count += 1
        assert corrupted_count >= 90


# ---- Operator registry test --------------------------------------------------


class TestOperatorRegistry:
    """Tests for the operator registry and valid transform_fns."""

    def test_all_operators_in_registry(self) -> None:
        expected = {
            "typo_substitute",
            "ocr_confuse",
            "truncate",
            "whitespace_corrupt",
            "case_corrupt",
            "taxonomy_walk",
        }
        assert set(OPERATOR_REGISTRY.keys()) == expected

    def test_valid_transform_fns_superset(self) -> None:
        # VALID_TRANSFORM_FNS includes cleanup and rollback_for_committee
        assert set(OPERATOR_REGISTRY.keys()).issubset(VALID_TRANSFORM_FNS)
        assert "cleanup" in VALID_TRANSFORM_FNS
        assert "rollback_for_committee" in VALID_TRANSFORM_FNS
        assert "taxonomy_walk" in VALID_TRANSFORM_FNS


# ---- Dispatcher / integration tests -----------------------------------------


class TestApplyKnob06:
    """Integration tests for the Knob 06 dispatcher."""

    def test_easy_no_primary_noise(
        self,
        small_sources: dict[str, pd.DataFrame],
        small_config: dict[str, Any],
    ) -> None:
        """At easy, primary attributes are never noised (rate=0)."""
        noised, prov_df, skipped_df = apply_knob_06(
            domain="companies",
            level="easy",
            sources=small_sources,
            config=small_config,
            seed=42,
        )
        if len(prov_df) > 0:
            primary_cols = set()
            for src, cols in small_config["attribute_classes"].items():
                for col, cls in cols.items():
                    if cls == "primary":
                        primary_cols.add((src, col))

            for _, row in prov_df.iterrows():
                pair = (row["source"], row["attribute"])
                assert (
                    pair not in primary_cols
                ), f"Primary attribute noised at easy: {pair}"

    def test_medium_no_primary_noise(
        self,
        small_sources: dict[str, pd.DataFrame],
        small_config: dict[str, Any],
    ) -> None:
        """At medium, primary attributes are never noised (rate=0)."""
        noised, prov_df, skipped_df = apply_knob_06(
            domain="companies",
            level="medium",
            sources=small_sources,
            config=small_config,
            seed=42,
        )
        if len(prov_df) > 0:
            primary_cols = set()
            for src, cols in small_config["attribute_classes"].items():
                for col, cls in cols.items():
                    if cls == "primary":
                        primary_cols.add((src, col))

            for _, row in prov_df.iterrows():
                pair = (row["source"], row["attribute"])
                assert pair not in primary_cols

    def test_hard_more_noise_than_medium(
        self,
        small_sources: dict[str, pd.DataFrame],
        small_config: dict[str, Any],
    ) -> None:
        """Hard level produces more noise than medium."""
        _, prov_m, _ = apply_knob_06(
            domain="companies",
            level="medium",
            sources=small_sources,
            config=small_config,
            seed=42,
        )
        _, prov_h, _ = apply_knob_06(
            domain="companies",
            level="hard",
            sources=small_sources,
            config=small_config,
            seed=42,
        )
        assert len(prov_h) >= len(prov_m)

    def test_determinism(
        self,
        small_sources: dict[str, pd.DataFrame],
        small_config: dict[str, Any],
    ) -> None:
        """Same seed + same input = identical output."""
        noised_1, prov_1, skipped_1 = apply_knob_06(
            domain="companies",
            level="hard",
            sources=small_sources,
            config=small_config,
            seed=42,
        )
        noised_2, prov_2, skipped_2 = apply_knob_06(
            domain="companies",
            level="hard",
            sources=small_sources,
            config=small_config,
            seed=42,
        )

        for src in noised_1:
            pd.testing.assert_frame_equal(noised_1[src], noised_2[src])

        pd.testing.assert_frame_equal(prov_1, prov_2)
        pd.testing.assert_frame_equal(skipped_1, skipped_2)

    def test_attrs_preserved(
        self,
        small_sources: dict[str, pd.DataFrame],
        small_config: dict[str, Any],
    ) -> None:
        """DataFrame.attrs['dataset_name'] survives noise injection."""
        noised, _, _ = apply_knob_06(
            domain="companies",
            level="medium",
            sources=small_sources,
            config=small_config,
            seed=42,
        )
        for src_name, df in noised.items():
            assert df.attrs.get("dataset_name") == src_name

    def test_provenance_schema(
        self,
        small_sources: dict[str, pd.DataFrame],
        small_config: dict[str, Any],
    ) -> None:
        """Provenance DataFrame has the correct schema."""
        _, prov_df, _ = apply_knob_06(
            domain="companies",
            level="hard",
            sources=small_sources,
            config=small_config,
            seed=42,
        )
        assert list(prov_df.columns) == PROVENANCE_COLUMNS
        if len(prov_df) > 0:
            for _, row in prov_df.iterrows():
                assert row["knob"] == 6
                assert row["level"] == "hard"
                assert row["transform_fn"] in VALID_TRANSFORM_FNS
                # transform_params should be valid JSON.
                params = json.loads(row["transform_params"])
                assert isinstance(params, dict)

    def test_skipped_schema(
        self,
        small_sources: dict[str, pd.DataFrame],
        small_config: dict[str, Any],
    ) -> None:
        """Skipped DataFrame has the correct schema."""
        _, _, skipped_df = apply_knob_06(
            domain="companies",
            level="hard",
            sources=small_sources,
            config=small_config,
            seed=42,
        )
        assert list(skipped_df.columns) == SKIPPED_COLUMNS


class TestCleanPrimaryFloor:
    """Tests for the per-entity clean-primary floor constraint."""

    def test_floor_holds_at_hard(
        self,
        small_sources: dict[str, pd.DataFrame],
        small_config: dict[str, Any],
        small_entity_groups: dict[str, list[tuple[str, str]]],
    ) -> None:
        """For every linked entity, at least 1 source retains clean primary."""
        # Use high primary rate to stress the floor.
        config = {**small_config}
        config["noise_rates_per_level"] = {
            **config["noise_rates_per_level"],
            "hard": {"primary": 0.90, "key": 0.50, "secondary": 0.80},
        }

        noised, prov_df, _ = apply_knob_06(
            domain="companies",
            level="hard",
            sources=small_sources,
            config=config,
            entity_groups=small_entity_groups,
            seed=42,
        )

        # Build set of (entity_id, source) that had their primary noised.
        primary_cols = {}
        for src, cols in config["attribute_classes"].items():
            for col, cls in cols.items():
                if cls == "primary":
                    primary_cols[src] = col

        noised_primary_ids: dict[str, set[str]] = {src: set() for src in noised}
        if len(prov_df) > 0:
            for _, row in prov_df.iterrows():
                src = row["source"]
                attr = row["attribute"]
                if primary_cols.get(src) == attr:
                    noised_primary_ids[src].add(row["entity_id"])

        # Check: for every multi-source entity group, at least 1 source
        # has a clean primary.
        for group_id, members in small_entity_groups.items():
            if len(members) <= 1:
                continue
            clean_count = sum(
                1
                for src, rid in members
                if rid not in noised_primary_ids.get(src, set())
            )
            assert clean_count >= 1, (
                f"Entity group {group_id} has no clean primary: "
                f"members={members}, noised={noised_primary_ids}"
            )


class TestCollisionIndex:
    """Tests for cell-collision index integration."""

    def test_skips_touched_cells(
        self,
        small_sources: dict[str, pd.DataFrame],
        small_config: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Cells with prior provenance are skipped."""
        # Write a fake K5 provenance to simulate prior knob.
        prov_dir = tmp_path / "output" / "provenance"
        prov_dir.mkdir(parents=True)

        fake_prov = ProvenanceLog(knob=5, level="medium")
        fake_prov.append(
            entity_id="db_0",
            source="dbpedia",
            attribute="revenue",
            original_value="100",
            new_value="100.00",
            transform_fn="reformat_number",
            transform_params={"from_locale": "en_US", "to_locale": "de_DE"},
        )
        fake_prov.flush(prov_dir / "knob_05_format_unit.csv")

        collision_idx = CollisionIndex(prov_dir)

        # Use high rates to ensure the cell would normally be noised.
        config = {**small_config}
        config["noise_rates_per_level"] = {
            **config["noise_rates_per_level"],
            "hard": {"primary": 0.99, "key": 0.99, "secondary": 0.99},
        }

        _, _, skipped_df = apply_knob_06(
            domain="companies",
            level="hard",
            sources=small_sources,
            config=config,
            collision_index=collision_idx,
            seed=42,
        )

        # Check that db_0 revenue was skipped for collision.
        collision_skips = skipped_df[
            skipped_df["reason"] == "cell_collision_with_prior_knob"
        ]
        found = any(
            (row["entity_id"] == "db_0" and row["attribute"] == "revenue")
            for _, row in collision_skips.iterrows()
        )
        assert found, "db_0 revenue should have been skipped for collision"

    def test_k4_fabricated_not_skipped(
        self,
        small_sources: dict[str, pd.DataFrame],
        small_config: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """K4-fabricated cells are NOT skipped (exception rule)."""
        prov_dir = tmp_path / "output" / "provenance"
        prov_dir.mkdir(parents=True)

        fake_prov = ProvenanceLog(knob=4, level="medium")
        fake_prov.append(
            entity_id="db_0",
            source="dbpedia",
            attribute="revenue",
            original_value="",
            new_value="500000",
            transform_fn="fabricate_coverage",
            transform_params={"k4_fabricated": True},
        )
        fake_prov.flush(prov_dir / "knob_04_coverage.csv")

        collision_idx = CollisionIndex(prov_dir)

        config = {**small_config}
        config["noise_rates_per_level"] = {
            **config["noise_rates_per_level"],
            "hard": {"primary": 0.99, "key": 0.99, "secondary": 0.99},
        }

        _, prov_df, skipped_df = apply_knob_06(
            domain="companies",
            level="hard",
            sources=small_sources,
            config=config,
            collision_index=collision_idx,
            seed=42,
        )

        # db_0 revenue should NOT be skipped (K4-fabricated exception).
        collision_skips = skipped_df[
            skipped_df["reason"] == "cell_collision_with_prior_knob"
        ]
        blocked = any(
            (row["entity_id"] == "db_0" and row["attribute"] == "revenue")
            for _, row in collision_skips.iterrows()
        )
        assert (
            not blocked
        ), "db_0 revenue should NOT be skipped — K4-fabricated exception"


class TestCleanupRules:
    """Tests for easy-level cleanup transform."""

    def test_cleanup_applied(self) -> None:
        """Cleanup rules revert known baseline noise at easy."""
        sources = {
            "forbes": pd.DataFrame(
                {
                    "Identifier": ["fb_0", "fb_1"],
                    "Company": ["Acme", "Globex"],
                    "Country": ["United States [a]", "Germany"],
                    "Assets": ["50 billion", "20 billion"],
                }
            ),
        }
        sources["forbes"].attrs["dataset_name"] = "forbes"

        config: dict[str, Any] = {
            "id_columns": {"forbes": "Identifier"},
            "attribute_classes": {
                "forbes": {
                    "Company": "primary",
                    "Country": "key",
                    "Assets": "secondary",
                },
            },
            "attribute_mapping": {"forbes": {"Company": "name", "Country": "country"}},
            "noise_rates_per_level": {
                "easy": {"primary": 0.0, "key": 0.0, "secondary": 0.01},
            },
            "operator_mix": {
                "easy": {"whitespace_corrupt": 1.0, "case_corrupt": 1.0},
            },
            "max_edits_per_cell": {"easy": 1},
            "max_ocr_per_cell": {"easy": 1},
            "max_truncate_chars": {"easy": 1},
            "cleanup_rules": [
                {
                    "source": "forbes",
                    "attribute": "Country",
                    "pattern": "\\s*\\[a\\]\\s*$",
                    "replacement": "",
                },
            ],
        }

        noised, prov_df, _ = apply_knob_06(
            domain="companies",
            level="easy",
            sources=sources,
            config=config,
            seed=42,
        )

        # The "[a]" should be removed from "United States [a]".
        assert noised["forbes"].iloc[0]["Country"] == "United States"

        # Check provenance has a cleanup row.
        cleanup_rows = prov_df[prov_df["transform_fn"] == "cleanup"]
        assert len(cleanup_rows) >= 1


class TestMonotoneNoiseCounts:
    """Tests that noise counts are monotone across levels."""

    def test_monotone_per_class(
        self,
        small_sources: dict[str, pd.DataFrame],
        small_config: dict[str, Any],
    ) -> None:
        """Total corrupted cell count: hard >= medium >= easy per attr class."""
        counts: dict[str, dict[str, int]] = {}

        for level in ("easy", "medium", "hard"):
            _, prov_df, _ = apply_knob_06(
                domain="companies",
                level=level,
                sources=small_sources,
                config=small_config,
                seed=42,
            )

            # Classify each provenance row by attribute class.
            level_counts: dict[str, int] = {
                "primary": 0,
                "key": 0,
                "secondary": 0,
            }
            for _, row in prov_df.iterrows():
                if row["transform_fn"] == "cleanup":
                    continue
                src = row["source"]
                attr = row["attribute"]
                cls = (
                    small_config["attribute_classes"]
                    .get(src, {})
                    .get(attr, "secondary")
                )
                level_counts[cls] += 1

            counts[level] = level_counts

        for cls in ("primary", "key", "secondary"):
            assert counts["hard"][cls] >= counts["medium"][cls], (
                f"{cls}: hard ({counts['hard'][cls]}) < "
                f"medium ({counts['medium'][cls]})"
            )
            assert counts["medium"][cls] >= counts["easy"][cls], (
                f"{cls}: medium ({counts['medium'][cls]}) < "
                f"easy ({counts['easy'][cls]})"
            )


class TestWriteOutputs:
    """Tests for the output writing function."""

    def test_write_and_read_round_trip(self, tmp_path: Path) -> None:
        """Provenance and skipped CSVs round-trip correctly."""
        prov_df = pd.DataFrame(
            {
                "entity_id": ["db_0"],
                "source": ["dbpedia"],
                "attribute": ["name"],
                "original_value": ["Acme"],
                "new_value": ["Acne"],
                "transform_fn": ["typo_substitute"],
                "transform_params": ['{"positions": [2], "chars": ["n"]}'],
                "knob": [6],
                "level": ["hard"],
            }
        )
        skipped_df = pd.DataFrame(columns=SKIPPED_COLUMNS)

        write_outputs(prov_df, skipped_df, tmp_path)

        written_prov = pd.read_csv(
            tmp_path / "output" / "provenance" / "knob_06_noise.csv",
            keep_default_na=False,
        )
        assert len(written_prov) == 1
        assert written_prov.iloc[0]["entity_id"] == "db_0"
        assert written_prov.iloc[0]["transform_fn"] == "typo_substitute"


class TestLoadConfig:
    """Tests for config loading."""

    def test_load_companies_config(self) -> None:
        """Real companies config loads without error."""
        config = load_knob_06_config("companies")
        assert config["domain"] == "companies"
        assert "noise_rates_per_level" in config
        assert "operator_mix" in config
        assert "attribute_classes" in config

    def test_monotonicity(self) -> None:
        """Noise rates are monotone across levels."""
        config = load_knob_06_config("companies")
        rates = config["noise_rates_per_level"]
        for cls in ("primary", "key", "secondary"):
            e = rates["easy"][cls]
            m = rates["medium"][cls]
            h = rates["hard"][cls]
            assert e <= m <= h, f"Non-monotone {cls}: easy={e}, medium={m}, hard={h}"

    def test_operator_mix_non_shrinking(self) -> None:
        """Operator sets are non-shrinking across levels."""
        config = load_knob_06_config("companies")
        mix = config["operator_mix"]
        easy_ops = set(mix["easy"].keys())
        medium_ops = set(mix["medium"].keys())
        hard_ops = set(mix["hard"].keys())
        assert easy_ops.issubset(
            medium_ops
        ), f"Easy ops not subset of medium: {easy_ops - medium_ops}"
        assert medium_ops.issubset(
            hard_ops
        ), f"Medium ops not subset of hard: {medium_ops - hard_ops}"


class TestDifferentSeeds:
    """Tests that different seeds produce different outputs."""

    def test_different_seeds_different_output(
        self,
        small_sources: dict[str, pd.DataFrame],
        small_config: dict[str, Any],
    ) -> None:
        noised_a, _, _ = apply_knob_06(
            domain="companies",
            level="hard",
            sources=small_sources,
            config=small_config,
            seed=42,
        )
        noised_b, _, _ = apply_knob_06(
            domain="companies",
            level="hard",
            sources=small_sources,
            config=small_config,
            seed=99,
        )

        # At least one source should differ.
        any_diff = False
        for src in noised_a:
            if not noised_a[src].equals(noised_b[src]):
                any_diff = True
                break
        assert any_diff, "Different seeds should produce different outputs"
