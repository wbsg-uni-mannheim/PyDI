"""Tests for Knob 10 — Source Reliability Differentiation.

Acceptance criteria (from module_03_knob_10.md):
1. Fusion gold values byte-identical before and after reshuffling
2. Per-attribute gold-carrier concentration measured >=85% at easy
3. At hard, error_correlation produces entity-level burst patterns
4. No-op when all values identical across sources (no variants to reshuffle)
5. Baseline measured fresh every run (self-contained)
6. Multiset invariant: Counter(values_before) == Counter(values_after) per cell
7. Determinism: re-running with same seed produces bit-identical outputs
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from usecases_synthetic.lib.reliability import (
    assert_multiset_invariant,
    build_entity_linkage,
    canonicalize,
    generate_compromised_mask,
    identify_per_attribute_winner,
    identify_reshufflable_cells,
    is_gold_aligned,
    load_fusion_gold,
    measure_gold_alignment,
    reconcile_attribute_classes,
    reshuffle_cells,
)
from usecases_synthetic.lib.provenance import PROVENANCE_COLUMNS


# ---- Test fixtures --------------------------------------------------------


@pytest.fixture
def k10_config() -> dict[str, Any]:
    """Minimal K10 config for testing with 3 sources."""
    return {
        "domain": "test",
        "id_columns": {
            "source_a": "id",
            "source_b": "id",
            "source_c": "id",
        },
        "attribute_mapping": {
            "source_a": {"name": "name", "country": "country", "revenue": "revenue"},
            "source_b": {"name": "name", "country": "country", "revenue": "revenue"},
            "source_c": {"name": "name", "country": "country"},
        },
        "attribute_targets": {
            "name": {
                "easy": {"source_a": 0.90, "source_b": 0.05, "source_c": 0.05},
                "medium": {"source_a": 0.70, "source_b": 0.15, "source_c": 0.15},
                "hard": {"source_a": 0.40, "source_b": 0.35, "source_c": 0.25},
            },
            "country": {
                "easy": {"source_a": 0.05, "source_b": 0.90, "source_c": 0.05},
                "medium": {"source_a": 0.15, "source_b": 0.70, "source_c": 0.15},
                "hard": {"source_a": 0.35, "source_b": 0.40, "source_c": 0.25},
            },
            "revenue": {
                "easy": {"source_a": 0.10, "source_b": 0.90},
                "medium": {"source_a": 0.30, "source_b": 0.70},
                "hard": {"source_a": 0.60, "source_b": 0.40},
            },
        },
        "compromise_rate_per_level": {
            "easy": 0.0,
            "medium": 0.05,
            "hard": 0.15,
        },
        "corr_strength_per_level": {
            "easy": 0.0,
            "medium": 0.20,
            "hard": 0.50,
        },
        "concentration_cap": 0.99,
    }


def _make_sources(
    n_entities: int = 50,
    *,
    all_identical: bool = False,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    """Build synthetic source DataFrames, fusion gold, and entity linkage.

    Each entity is linked across all 3 sources. Source_a carries the
    gold-aligned values. Source_b and source_c carry perturbed values
    (unless ``all_identical=True``).
    """
    ids_a = [f"a_{i}" for i in range(n_entities)]
    ids_b = [f"b_{i}" for i in range(n_entities)]
    ids_c = [f"c_{i}" for i in range(n_entities)]

    gold_names = [f"Entity {i}" for i in range(n_entities)]
    gold_countries = [f"Country {i}" for i in range(n_entities)]
    gold_revenues = [f"{1000 + i}" for i in range(n_entities)]

    if all_identical:
        # All sources carry gold values
        src_a = pd.DataFrame({
            "id": ids_a,
            "name": gold_names,
            "country": gold_countries,
            "revenue": gold_revenues,
        })
        src_b = pd.DataFrame({
            "id": ids_b,
            "name": gold_names,
            "country": gold_countries,
            "revenue": gold_revenues,
        })
        src_c = pd.DataFrame({
            "id": ids_c,
            "name": gold_names,
            "country": gold_countries,
        })
    else:
        src_a = pd.DataFrame({
            "id": ids_a,
            "name": gold_names,
            "country": gold_countries,
            "revenue": gold_revenues,
        })
        src_b = pd.DataFrame({
            "id": ids_b,
            "name": [f"Entity_{i}_alt" for i in range(n_entities)],
            "country": gold_countries,  # same as gold
            "revenue": [f"{2000 + i}" for i in range(n_entities)],
        })
        src_c = pd.DataFrame({
            "id": ids_c,
            "name": [f"Entity-{i}-variant" for i in range(n_entities)],
            "country": [f"Country_{i}_diff" for i in range(n_entities)],
        })

    for df, name in [(src_a, "source_a"), (src_b, "source_b"), (src_c, "source_c")]:
        df.attrs["dataset_name"] = name

    sources = {"source_a": src_a, "source_b": src_b, "source_c": src_c}

    # Fusion gold keyed by source_a IDs (convention: first source is canonical)
    fusion_gold: dict[str, dict[str, str]] = {}
    for i in range(n_entities):
        fusion_gold[f"a_{i}"] = {
            "name": gold_names[i],
            "country": gold_countries[i],
            "revenue": gold_revenues[i],
        }

    # Entity linkage: gold_eid -> {source: record_id}
    entity_linkage: dict[str, dict[str, str]] = {}
    for i in range(n_entities):
        entity_linkage[f"a_{i}"] = {
            "source_a": f"a_{i}",
            "source_b": f"b_{i}",
            "source_c": f"c_{i}",
        }

    return sources, fusion_gold, entity_linkage


@pytest.fixture
def synthetic_data() -> tuple[
    dict[str, pd.DataFrame],
    dict[str, dict[str, str]],
    dict[str, dict[str, str]],
]:
    """Return sources, fusion_gold, entity_linkage for 50 entities."""
    return _make_sources(n_entities=50)


@pytest.fixture
def identical_data() -> tuple[
    dict[str, pd.DataFrame],
    dict[str, dict[str, str]],
    dict[str, dict[str, str]],
]:
    """Return sources where all values are identical (all gold-aligned)."""
    return _make_sources(n_entities=20, all_identical=True)


# ---- Canonical-form comparator tests -------------------------------------


class TestCanonicalComparators:
    """Test the canonical-form comparator functions."""

    def test_string_casefold(self) -> None:
        assert canonicalize("Hello World", "string") == "hello world"

    def test_string_collapse_ws(self) -> None:
        assert canonicalize("  hello   world  ", "string") == "hello world"

    def test_string_strip_punct(self) -> None:
        assert canonicalize("hello, world!", "string") == "hello world"

    def test_date_iso(self) -> None:
        from datetime import date

        assert canonicalize("2005-01-01", "date") == date(2005, 1, 1)

    def test_date_datetime_string(self) -> None:
        from datetime import date

        result = canonicalize("2005-01-01T00:00:00.000+01:00", "date")
        assert result == date(2005, 1, 1)

    def test_date_year_only(self) -> None:
        from datetime import date

        assert canonicalize("2005", "date") == date(2005, 1, 1)

    def test_number_plain(self) -> None:
        from decimal import Decimal

        assert canonicalize("1234.56", "number") == Decimal("1234.56")

    def test_number_scientific(self) -> None:
        from decimal import Decimal

        assert canonicalize("3.5E9", "number") == Decimal("3.5E9")

    def test_money_same_as_number(self) -> None:
        from decimal import Decimal

        assert canonicalize("3.5E9", "money") == Decimal("3.5E9")

    def test_null_returns_none(self) -> None:
        assert canonicalize("", "string") is None
        assert canonicalize("null", "date") is None


class TestIsGoldAligned:
    """Test gold-alignment checking."""

    def test_exact_match(self) -> None:
        assert is_gold_aligned("Apple Inc.", "Apple Inc.", "string")

    def test_casefold_match(self) -> None:
        assert is_gold_aligned("apple inc.", "Apple Inc.", "string")

    def test_no_match(self) -> None:
        assert not is_gold_aligned("Google", "Apple Inc.", "string")

    def test_date_match(self) -> None:
        assert is_gold_aligned(
            "2005", "2005-01-01T00:00:00.000+01:00", "date"
        )

    def test_number_tolerance(self) -> None:
        assert is_gold_aligned("3.5E9", "3500000000", "money")

    def test_null_not_aligned(self) -> None:
        assert not is_gold_aligned("", "Apple", "string")


# ---- Gold-alignment measurement tests ------------------------------------


class TestMeasureGoldAlignment:
    """Test baseline gold-alignment measurement."""

    def test_basic_measurement(
        self,
        synthetic_data: tuple[dict[str, pd.DataFrame], dict[str, dict[str, str]], dict[str, dict[str, str]]],
        k10_config: dict[str, Any],
    ) -> None:
        sources, fusion_gold, entity_linkage = synthetic_data
        attribute_mapping = k10_config["attribute_mapping"]
        id_columns = k10_config["id_columns"]

        df = measure_gold_alignment(
            sources=sources,
            fusion_gold=fusion_gold,
            attribute_mapping=attribute_mapping,
            id_columns=id_columns,
            attribute_classes={},  # all string
            entity_linkage=entity_linkage,
        )

        assert len(df) > 0
        assert set(df.columns) == {
            "source",
            "attribute",
            "baseline_alignment_rate",
            "n_cells",
            "n_aligned",
        }

        # Source_a should be gold-aligned on everything
        sa_rows = df[df["source"] == "source_a"]
        for _, row in sa_rows.iterrows():
            assert row["baseline_alignment_rate"] == 1.0, (
                f"source_a should be fully aligned on {row['attribute']}"
            )

    def test_identifies_winner(
        self,
        synthetic_data: tuple[dict[str, pd.DataFrame], dict[str, dict[str, str]], dict[str, dict[str, str]]],
        k10_config: dict[str, Any],
    ) -> None:
        sources, fusion_gold, entity_linkage = synthetic_data
        df = measure_gold_alignment(
            sources=sources,
            fusion_gold=fusion_gold,
            attribute_mapping=k10_config["attribute_mapping"],
            id_columns=k10_config["id_columns"],
            attribute_classes={},
            entity_linkage=entity_linkage,
        )
        winners = identify_per_attribute_winner(df)
        # Source_a has perfect alignment, so it should win everything
        for attr, winner in winners.items():
            assert winner == "source_a", f"Expected source_a to win {attr}"


# ---- Reshufflable cell identification tests -------------------------------


class TestIdentifyReshufflableCells:
    """Test cell classification."""

    def test_reshufflable_cells_exist(
        self,
        synthetic_data: tuple[dict[str, pd.DataFrame], dict[str, dict[str, str]], dict[str, dict[str, str]]],
        k10_config: dict[str, Any],
    ) -> None:
        sources, fusion_gold, entity_linkage = synthetic_data
        cells = identify_reshufflable_cells(
            sources=sources,
            fusion_gold=fusion_gold,
            attribute_mapping=k10_config["attribute_mapping"],
            id_columns=k10_config["id_columns"],
            attribute_classes={},
            entity_linkage=entity_linkage,
        )
        reshufflable = [c for c in cells if c["cell_type"] == "reshufflable"]
        assert len(reshufflable) > 0

    def test_no_reshufflable_when_identical(
        self,
        identical_data: tuple[dict[str, pd.DataFrame], dict[str, dict[str, str]], dict[str, dict[str, str]]],
        k10_config: dict[str, Any],
    ) -> None:
        """No-op when all values identical across sources."""
        sources, fusion_gold, entity_linkage = identical_data
        cells = identify_reshufflable_cells(
            sources=sources,
            fusion_gold=fusion_gold,
            attribute_mapping=k10_config["attribute_mapping"],
            id_columns=k10_config["id_columns"],
            attribute_classes={},
            entity_linkage=entity_linkage,
        )
        reshufflable = [c for c in cells if c["cell_type"] == "reshufflable"]
        assert len(reshufflable) == 0, (
            "Should have no reshufflable cells when all sources agree"
        )

    def test_cell_types_exhaustive(
        self,
        synthetic_data: tuple[dict[str, pd.DataFrame], dict[str, dict[str, str]], dict[str, dict[str, str]]],
        k10_config: dict[str, Any],
    ) -> None:
        sources, fusion_gold, entity_linkage = synthetic_data
        cells = identify_reshufflable_cells(
            sources=sources,
            fusion_gold=fusion_gold,
            attribute_mapping=k10_config["attribute_mapping"],
            id_columns=k10_config["id_columns"],
            attribute_classes={},
            entity_linkage=entity_linkage,
        )
        valid_types = {"reshufflable", "all_aligned", "no_gold_to_route", "passthrough"}
        for c in cells:
            assert c["cell_type"] in valid_types


# ---- Compromised mask tests -----------------------------------------------


class TestCompromisedMask:
    """Test compromised-mask generation."""

    def test_easy_empty_mask(self) -> None:
        rng = np.random.default_rng(42)
        mask = generate_compromised_mask(
            source_names=["a", "b", "c"],
            entity_ids=[f"e_{i}" for i in range(100)],
            compromise_rate=0.0,
            compromise_rate_overrides=None,
            rng=rng,
        )
        for src, entities in mask.items():
            assert len(entities) == 0, f"Easy should have empty mask for {src}"

    def test_hard_mask_size(self) -> None:
        rng = np.random.default_rng(42)
        n_entities = 100
        rate = 0.15
        mask = generate_compromised_mask(
            source_names=["a", "b", "c"],
            entity_ids=[f"e_{i}" for i in range(n_entities)],
            compromise_rate=rate,
            compromise_rate_overrides=None,
            rng=rng,
        )
        expected = int(np.floor(rate * n_entities))
        for src, entities in mask.items():
            assert len(entities) == expected, (
                f"Source {src}: expected {expected} compromised, got {len(entities)}"
            )

    def test_deterministic(self) -> None:
        args = dict(
            source_names=["a", "b", "c"],
            entity_ids=[f"e_{i}" for i in range(100)],
            compromise_rate=0.15,
            compromise_rate_overrides=None,
        )
        rng1 = np.random.default_rng(42)
        mask1 = generate_compromised_mask(rng=rng1, **args)
        rng2 = np.random.default_rng(42)
        mask2 = generate_compromised_mask(rng=rng2, **args)
        for src in ["a", "b", "c"]:
            assert mask1[src] == mask2[src]


# ---- Core reshuffle tests -------------------------------------------------


class TestReshuffle:
    """Test the core reshuffle logic."""

    def _run_reshuffle(
        self,
        synthetic_data: tuple[dict[str, pd.DataFrame], dict[str, dict[str, str]], dict[str, dict[str, str]]],
        k10_config: dict[str, Any],
        level: str,
    ) -> tuple[
        dict[str, pd.DataFrame],
        list[dict[str, Any]],
        list[dict[str, Any]],
        dict[str, pd.DataFrame],
    ]:
        """Helper to run the full reshuffle pipeline."""
        sources, fusion_gold, entity_linkage = synthetic_data
        attr_mapping = k10_config["attribute_mapping"]
        id_columns = k10_config["id_columns"]

        cells = identify_reshufflable_cells(
            sources=sources,
            fusion_gold=fusion_gold,
            attribute_mapping=attr_mapping,
            id_columns=id_columns,
            attribute_classes={},
            entity_linkage=entity_linkage,
        )

        rate = k10_config["compromise_rate_per_level"][level]
        corr = k10_config["corr_strength_per_level"][level]
        cap = k10_config["concentration_cap"]

        gold_eids = sorted(fusion_gold.keys())
        source_names = sorted(sources.keys())

        mask_rng = np.random.default_rng(100)
        cell_rng = np.random.default_rng(200)

        mask = generate_compromised_mask(
            source_names=source_names,
            entity_ids=gold_eids,
            compromise_rate=rate,
            compromise_rate_overrides=None,
            rng=mask_rng,
        )

        attr_targets = {
            attr: levels[level]
            for attr, levels in k10_config["attribute_targets"].items()
        }

        # Keep originals
        originals = {n: df.copy() for n, df in sources.items()}

        mutated, prov_rows = reshuffle_cells(
            cells=cells,
            sources=sources,
            attribute_targets=attr_targets,
            compromised_mask=mask,
            corr_strength=corr,
            concentration_cap=cap,
            rng=cell_rng,
            level=level,
        )

        return mutated, prov_rows, cells, originals

    def test_multiset_invariant(
        self,
        synthetic_data: tuple[dict[str, pd.DataFrame], dict[str, dict[str, str]], dict[str, dict[str, str]]],
        k10_config: dict[str, Any],
    ) -> None:
        """Counter(values_before) == Counter(values_after) per cell."""
        mutated, _, cells, originals = self._run_reshuffle(
            synthetic_data, k10_config, "hard"
        )
        # Should not raise
        assert_multiset_invariant(originals, mutated, cells)

    def test_no_op_when_all_identical(
        self,
        identical_data: tuple[dict[str, pd.DataFrame], dict[str, dict[str, str]], dict[str, dict[str, str]]],
        k10_config: dict[str, Any],
    ) -> None:
        """When all sources carry gold values, reshuffle is a no-op."""
        sources, fusion_gold, entity_linkage = identical_data
        cells = identify_reshufflable_cells(
            sources=sources,
            fusion_gold=fusion_gold,
            attribute_mapping=k10_config["attribute_mapping"],
            id_columns=k10_config["id_columns"],
            attribute_classes={},
            entity_linkage=entity_linkage,
        )

        mask_rng = np.random.default_rng(100)
        cell_rng = np.random.default_rng(200)

        mask = generate_compromised_mask(
            source_names=sorted(sources.keys()),
            entity_ids=sorted(fusion_gold.keys()),
            compromise_rate=0.15,
            compromise_rate_overrides=None,
            rng=mask_rng,
        )

        attr_targets = {
            attr: levels["hard"]
            for attr, levels in k10_config["attribute_targets"].items()
        }

        mutated, prov_rows = reshuffle_cells(
            cells=cells,
            sources=sources,
            attribute_targets=attr_targets,
            compromised_mask=mask,
            corr_strength=0.5,
            concentration_cap=0.99,
            rng=cell_rng,
            level="hard",
        )

        # No swaps should have occurred
        swap_rows = [r for r in prov_rows if r["transform_fn"] == "reassign_gold_carrier"]
        assert len(swap_rows) == 0, "No swaps expected when all values identical"

    def test_provenance_row_counts(
        self,
        synthetic_data: tuple[dict[str, pd.DataFrame], dict[str, dict[str, str]], dict[str, dict[str, str]]],
        k10_config: dict[str, Any],
    ) -> None:
        """Provenance rows = 2*N_swap + 1*N_identity + 1*N_no_gold."""
        _, prov_rows, cells, _ = self._run_reshuffle(
            synthetic_data, k10_config, "medium"
        )

        n_swap = sum(
            1 for r in prov_rows if r["transform_fn"] == "reassign_gold_carrier"
        )
        n_identity = sum(
            1 for r in prov_rows if r["transform_fn"] == "identity"
        )
        n_no_gold = sum(
            1 for r in prov_rows if r["transform_fn"] == "no_gold_to_route"
        )

        # Each swap produces 2 rows, identity 1, no_gold 1
        assert n_swap % 2 == 0, "reassign_gold_carrier rows should come in pairs"
        assert len(prov_rows) == n_swap + n_identity + n_no_gold

    def test_determinism(
        self,
        synthetic_data: tuple[dict[str, pd.DataFrame], dict[str, dict[str, str]], dict[str, dict[str, str]]],
        k10_config: dict[str, Any],
    ) -> None:
        """Re-running with same seed produces identical outputs."""
        mut1, prov1, _, _ = self._run_reshuffle(
            synthetic_data, k10_config, "hard"
        )

        # Re-create sources (need fresh copies)
        sources2, fusion_gold2, entity_linkage2 = _make_sources(n_entities=50)
        synthetic2 = (sources2, fusion_gold2, entity_linkage2)
        mut2, prov2, _, _ = self._run_reshuffle(
            synthetic2, k10_config, "hard"
        )

        # Compare mutated source values
        for src_name in sorted(mut1.keys()):
            pd.testing.assert_frame_equal(mut1[src_name], mut2[src_name])

        # Compare provenance
        assert len(prov1) == len(prov2)
        for r1, r2 in zip(prov1, prov2):
            assert r1["entity_id"] == r2["entity_id"]
            assert r1["transform_fn"] == r2["transform_fn"]


# ---- Concentration monotonicity tests -------------------------------------


class TestConcentrationMonotonicity:
    """Test that per-attribute winner concentration is monotone."""

    def _measure_realized_concentration(
        self,
        mutated_sources: dict[str, pd.DataFrame],
        fusion_gold: dict[str, dict[str, str]],
        entity_linkage: dict[str, dict[str, str]],
        attribute_mapping: dict[str, dict[str, str]],
        id_columns: dict[str, str],
    ) -> dict[str, dict[str, float]]:
        """Measure realized per-(source, attribute) gold alignment after reshuffle."""
        alignment = measure_gold_alignment(
            sources=mutated_sources,
            fusion_gold=fusion_gold,
            attribute_mapping=attribute_mapping,
            id_columns=id_columns,
            attribute_classes={},
            entity_linkage=entity_linkage,
        )
        result: dict[str, dict[str, float]] = {}
        for _, row in alignment.iterrows():
            attr = row["attribute"]
            src = row["source"]
            result.setdefault(attr, {})[src] = row["baseline_alignment_rate"]
        return result

    def test_monotone_easy_to_hard(
        self,
        k10_config: dict[str, Any],
    ) -> None:
        """Winner concentration decreases easy -> medium -> hard."""
        concentrations: dict[str, dict[str, float]] = {}  # level -> attr -> max_rate

        for level in ["easy", "medium", "hard"]:
            sources, fusion_gold, entity_linkage = _make_sources(n_entities=50)
            attr_mapping = k10_config["attribute_mapping"]
            id_columns = k10_config["id_columns"]

            cells = identify_reshufflable_cells(
                sources=sources,
                fusion_gold=fusion_gold,
                attribute_mapping=attr_mapping,
                id_columns=id_columns,
                attribute_classes={},
                entity_linkage=entity_linkage,
            )

            rate = k10_config["compromise_rate_per_level"][level]
            corr = k10_config["corr_strength_per_level"][level]

            mask_rng = np.random.default_rng(100)
            cell_rng = np.random.default_rng(200)

            mask = generate_compromised_mask(
                source_names=sorted(sources.keys()),
                entity_ids=sorted(fusion_gold.keys()),
                compromise_rate=rate,
                compromise_rate_overrides=None,
                rng=mask_rng,
            )

            attr_targets = {
                attr: levels[level]
                for attr, levels in k10_config["attribute_targets"].items()
            }

            mutated, _ = reshuffle_cells(
                cells=cells,
                sources=sources,
                attribute_targets=attr_targets,
                compromised_mask=mask,
                corr_strength=corr,
                concentration_cap=0.99,
                rng=cell_rng,
                level=level,
            )

            realized = self._measure_realized_concentration(
                mutated, fusion_gold, entity_linkage, attr_mapping, id_columns
            )

            # Max concentration per attribute (the winner)
            level_max: dict[str, float] = {}
            for attr, src_rates in realized.items():
                if src_rates:
                    level_max[attr] = max(src_rates.values())
            concentrations[level] = level_max

        # Check monotonicity: easy >= medium >= hard per attribute (with tolerance)
        tolerance = 0.10  # slightly generous for small sample
        for attr in concentrations.get("easy", {}):
            if attr in concentrations.get("medium", {}) and attr in concentrations.get("hard", {}):
                e = concentrations["easy"][attr]
                m = concentrations["medium"][attr]
                h = concentrations["hard"][attr]
                assert e >= m - tolerance, (
                    f"{attr}: easy({e:.2f}) < medium({m:.2f}) - tol({tolerance})"
                )
                assert m >= h - tolerance, (
                    f"{attr}: medium({m:.2f}) < hard({h:.2f}) - tol({tolerance})"
                )


# ---- Burst pattern tests (hard level) ------------------------------------


class TestBurstPatterns:
    """Test that hard-level compromised mask creates correlated errors."""

    def test_hard_has_compromised_entities(self) -> None:
        """At hard, the mask should have non-empty entity sets."""
        rng = np.random.default_rng(42)
        mask = generate_compromised_mask(
            source_names=["a", "b", "c"],
            entity_ids=[f"e_{i}" for i in range(100)],
            compromise_rate=0.15,
            compromise_rate_overrides=None,
            rng=rng,
        )
        total = sum(len(v) for v in mask.values())
        # 3 sources * floor(0.15 * 100) = 3 * 15 = 45
        assert total == 45

    def test_easy_has_no_compromised(self) -> None:
        """At easy, no entities are compromised."""
        rng = np.random.default_rng(42)
        mask = generate_compromised_mask(
            source_names=["a", "b", "c"],
            entity_ids=[f"e_{i}" for i in range(100)],
            compromise_rate=0.0,
            compromise_rate_overrides=None,
            rng=rng,
        )
        total = sum(len(v) for v in mask.values())
        assert total == 0


# ---- Attribute-class reconciliation tests ---------------------------------


class TestReconcileAttributeClasses:
    """Test Knob 5 -> Knob 10 attribute class reconciliation."""

    def test_majority_vote(self) -> None:
        k5 = {
            "src_a": {"col1": "date", "col2": "money"},
            "src_b": {"col1": "date", "col2": "number"},
            "src_c": {"col1": "date"},
        }
        result = reconcile_attribute_classes(k5, ["src_a", "src_b", "src_c"])
        assert result["col1"] == "date"

    def test_tiebreak_by_source_order(self) -> None:
        k5 = {
            "src_a": {"col1": "money"},
            "src_b": {"col1": "number"},
        }
        result = reconcile_attribute_classes(k5, ["src_a", "src_b"])
        # src_a is first in canonical order, its family (money) wins
        assert result["col1"] == "money"

    def test_empty_returns_empty(self) -> None:
        result = reconcile_attribute_classes({}, ["src_a"])
        assert result == {}
