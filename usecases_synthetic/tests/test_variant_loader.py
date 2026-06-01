"""Tests for ``usecases_synthetic.lib.variant_loader``."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from usecases_synthetic.lib.variant_loader import (
    VariantBundle,
    _load_em_gold,
    load_variant,
    variant_root,
)

# ---------------------------------------------------------------------------
# Fixture factory — builds a tiny companies-shaped augmented variant on disk
# ---------------------------------------------------------------------------


def _minimal_target_schema() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema#",
        "title": "Company",
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "name": {"type": "string"},
            "country": {"type": "string"},
        },
        "required": ["id", "name"],
    }


def _write_em_gold_csv(path: Path, rows: list[tuple[str, str, str]]) -> None:
    """Write a header-less EM gold CSV (id1, id2, label)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for id1, id2, label in rows:
            f.write(f"{id1},{id2},{label}\n")


def _build_augmented_fixture(root: Path) -> Path:
    """Create a full companies-augmented/easy variant under ``root``.

    Returns the variant directory (``root/input``'s parent).
    """
    variant = root
    input_dir = variant / "input"
    data = input_dir / "data"
    sm = input_dir / "schemamatching"
    em = input_dir / "entitymatching"
    fusion = input_dir / "fusion"
    for d in (data, sm, em, fusion):
        d.mkdir(parents=True, exist_ok=True)

    # Source CSVs (package_variant.py serialises every source as CSV).
    for name in ("dbpedia", "forbes", "fullcontact"):
        df = pd.DataFrame(
            {
                "id": [f"{name}_0", f"{name}_1"],
                "name": [f"{name}_name_0", f"{name}_name_1"],
                "country": ["USA", "Germany"],
            }
        )
        df.to_csv(data / f"{name}.csv", index=False)

    # Schema matching: target schema + K8 mapping.
    with open(sm / "target_schema.json", "w", encoding="utf-8") as f:
        json.dump(_minimal_target_schema(), f)

    mapping = pd.DataFrame(
        {
            "source": ["dbpedia", "forbes"],
            "source_col": ["name", "name"],
            "target_col": ["name", "name"],
        }
    )
    mapping.to_csv(sm / "sm_mapping.csv", index=False)

    # EM gold per source pair with all splits.
    for pair in (("forbes", "dbpedia"), ("forbes", "fullcontact")):
        src1, src2 = pair
        for split in ("all", "train", "val", "test"):
            _write_em_gold_csv(
                em / f"{src1}_2_{src2}_{split}.csv",
                [(f"{src1}_0", f"{src2}_0", "true")],
            )

    # Fusion gold + validation as small XML files.
    test_xml = (
        "<companies>\n"
        "<company><id>c1</id><name>Alpha</name></company>\n"
        "<company><id>c2</id><name>Beta</name></company>\n"
        "</companies>\n"
    )
    (fusion / "test_set.xml").write_text(test_xml, encoding="utf-8")
    (fusion / "validation_set.xml").write_text(test_xml, encoding="utf-8")

    return variant


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestVariantRoot:
    def test_baseline_points_at_original(self) -> None:
        root = variant_root("companies", "baseline")
        assert root.name == "companies"
        assert root.parent.name == "usecases"

    def test_augmented_level(self) -> None:
        root = variant_root("companies", "easy")
        assert root.name == "easy"
        assert root.parent.name == "companies-augmented"


class TestLoadVariantAugmentedFixture:
    """Acceptance criterion 3: fixture augmented variant loads end-to-end."""

    @pytest.fixture
    def fixture_variant(self, tmp_path: Path) -> Path:
        return _build_augmented_fixture(tmp_path / "companies-augmented" / "easy")

    def test_load_augmented_fixture(self, fixture_variant: Path) -> None:
        bundle = load_variant(
            "companies",
            level="easy",
            root_override=fixture_variant,
        )
        assert isinstance(bundle, VariantBundle)
        assert bundle.domain == "companies"
        assert bundle.level == "easy"

        # Sources loaded with dataset_name.
        assert set(bundle.sources) == {"dbpedia", "forbes", "fullcontact"}
        for name, df in bundle.sources.items():
            assert df.attrs.get("dataset_name") == name
            assert len(df) == 2

        # SM mapping present for augmented.
        assert bundle.sm_mapping is not None
        assert "source_col" in bundle.sm_mapping.columns

        # Target schema parsed.
        assert bundle.target_schema["title"] == "Company"

        # EM gold per pair.
        assert set(bundle.em_gold.keys()) == {
            ("forbes", "dbpedia"),
            ("forbes", "fullcontact"),
        }
        for frame in bundle.em_gold.values():
            assert list(frame.columns) == ["id1", "id2", "label"]
            assert len(frame) == 1

        # EM splits contain all four splits.
        for pair, splits in bundle.em_splits.items():
            assert set(splits.keys()) == {"train", "val", "test", "all"}

        # Fusion gold loaded (XML parser collapses to 2 rows).
        assert len(bundle.fusion_gold) == 2
        assert bundle.fusion_validation is not None

        assert bundle.variant_root == fixture_variant

    def test_missing_fusion_raises(self, fixture_variant: Path) -> None:
        (fixture_variant / "input" / "fusion" / "test_set.xml").unlink()
        with pytest.raises(FileNotFoundError, match="Fusion gold missing"):
            load_variant(
                "companies",
                level="easy",
                root_override=fixture_variant,
            )

    def test_missing_target_schema_raises(self, fixture_variant: Path) -> None:
        (fixture_variant / "input" / "schemamatching" / "target_schema.json").unlink()
        with pytest.raises(FileNotFoundError, match="Target schema"):
            load_variant(
                "companies",
                level="easy",
                root_override=fixture_variant,
            )


class TestLoadBaselineCompanies:
    """Acceptance criterion 2: baseline loads the untouched original dir.

    This test actually reads the repo's companies use case via the
    standard PyDI loaders — it is the end-to-end check that
    ``level="baseline"`` is wired correctly.
    """

    def test_load_baseline(self) -> None:
        bundle = load_variant("companies", level="baseline")
        assert bundle.level == "baseline"
        # Baseline loads sm_mapping_gold.csv (hand-authored gold mapping).
        assert bundle.sm_mapping is not None
        assert "source_column" in bundle.sm_mapping.columns
        assert set(bundle.sources) == {"dbpedia", "forbes", "fullcontact"}
        for name, df in bundle.sources.items():
            assert df.attrs.get("dataset_name") == name
        # Source pairs from the domain config should all have test gold.
        assert set(bundle.em_gold.keys()) == {
            ("forbes", "dbpedia"),
            ("forbes", "fullcontact"),
        }
        for pair_gold in bundle.em_gold.values():
            assert list(pair_gold.columns) == ["id1", "id2", "label"]
            assert len(pair_gold) > 0
        assert len(bundle.fusion_gold) > 0
        # Pool exists for companies.
        assert bundle.pooled_positives is not None
        assert "id1" in bundle.pooled_positives.columns

    def test_invalid_level_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid level"):
            load_variant("companies", level="bogus")


class TestLoadEMGoldDirectionTolerance:
    """``_load_em_gold`` must accept either on-disk pair orientation.

    Regression test for plan_revision_step4g_findings.md §1 — the games
    domain declared ``source_pairs: [[metacritic, dbpedia]]`` but the
    test gold lived at ``dbpedia_2_metacritic_test.csv``, and the old
    loader silently dropped the pair because it checked only the
    declared direction. Mirrors ``_load_em_gold_regenerated``'s existing
    direction tolerance.
    """

    def test_loads_when_file_matches_declared_direction(self, tmp_path: Path) -> None:
        em_dir = tmp_path / "entitymatching"
        em_dir.mkdir(parents=True)
        _write_em_gold_csv(
            em_dir / "metacritic_2_dbpedia_test.csv",
            [("metacritic_1", "dbpedia_1", "true")],
        )

        out = _load_em_gold(em_dir, [("metacritic", "dbpedia")])

        assert set(out.keys()) == {("metacritic", "dbpedia")}
        assert list(out[("metacritic", "dbpedia")].columns) == ["id1", "id2", "label"]
        assert len(out[("metacritic", "dbpedia")]) == 1

    def test_loads_when_file_is_reverse_direction(self, tmp_path: Path) -> None:
        em_dir = tmp_path / "entitymatching"
        em_dir.mkdir(parents=True)
        # File is dbpedia_2_metacritic_test.csv but declared pair is
        # (metacritic, dbpedia) — the games-on-disk situation that
        # silently dropped the pair before the direction-tolerant fix.
        _write_em_gold_csv(
            em_dir / "dbpedia_2_metacritic_test.csv",
            [("dbpedia_1", "metacritic_1", "true")],
        )

        out = _load_em_gold(em_dir, [("metacritic", "dbpedia")])

        assert set(out.keys()) == {("metacritic", "dbpedia")}
        frame = out[("metacritic", "dbpedia")]
        assert len(frame) == 1
        # id1/id2 MUST be swapped to match the declared pair direction —
        # downstream consumers (committee_em.py: _score_predictions, the
        # matcher input) look up id1 in df_left = sources[src1]. For
        # pair (metacritic, dbpedia) that means id1 must hold metacritic
        # ids; the on-disk file had dbpedia in id1, so the loader swaps.
        # Regression check for the 2026-05-26 silent F1=0 + magellan
        # crash on games metacritic_dbpedia.
        assert frame.iloc[0]["id1"] == "metacritic_1"
        assert frame.iloc[0]["id2"] == "dbpedia_1"
        assert frame.iloc[0]["label"] == "true"

    def test_prefers_all_over_test(self, tmp_path: Path) -> None:
        em_dir = tmp_path / "entitymatching"
        em_dir.mkdir(parents=True)
        _write_em_gold_csv(
            em_dir / "forbes_2_dbpedia_test.csv",
            [("forbes_1", "dbpedia_1", "true")],
        )
        _write_em_gold_csv(
            em_dir / "forbes_2_dbpedia_all.csv",
            [
                ("forbes_1", "dbpedia_1", "true"),
                ("forbes_2", "dbpedia_2", "false"),
            ],
        )

        out = _load_em_gold(em_dir, [("forbes", "dbpedia")])

        # _all preferred over _test (2 rows vs 1).
        assert len(out[("forbes", "dbpedia")]) == 2

    def test_prefers_declared_direction_when_both_exist(self, tmp_path: Path) -> None:
        em_dir = tmp_path / "entitymatching"
        em_dir.mkdir(parents=True)
        # Both orientations exist — declared direction wins (file size
        # 2 vs 1 makes the choice observable).
        _write_em_gold_csv(
            em_dir / "metacritic_2_dbpedia_test.csv",
            [
                ("metacritic_1", "dbpedia_1", "true"),
                ("metacritic_2", "dbpedia_2", "true"),
            ],
        )
        _write_em_gold_csv(
            em_dir / "dbpedia_2_metacritic_test.csv",
            [("dbpedia_1", "metacritic_1", "true")],
        )

        out = _load_em_gold(em_dir, [("metacritic", "dbpedia")])

        assert len(out[("metacritic", "dbpedia")]) == 2

    def test_missing_pair_silently_skipped(self, tmp_path: Path) -> None:
        em_dir = tmp_path / "entitymatching"
        em_dir.mkdir(parents=True)
        # No gold for declared pair (either direction). Loader should
        # skip without raising — matches pre-fix behaviour for the
        # truly-missing case.
        out = _load_em_gold(em_dir, [("metacritic", "dbpedia")])
        assert out == {}

    def test_all_split_falls_back_to_reverse_test_when_no_forward_test(
        self, tmp_path: Path
    ) -> None:
        """Cross-orientation fallback works for the ``_test`` split too.

        Edge case: ``_all`` doesn't exist in either direction; ``_test``
        exists only in the reverse direction. Pre-fix the pair was
        skipped; post-fix the reverse ``_test`` is loaded.
        """
        em_dir = tmp_path / "entitymatching"
        em_dir.mkdir(parents=True)
        _write_em_gold_csv(
            em_dir / "dbpedia_2_metacritic_test.csv",
            [("dbpedia_1", "metacritic_1", "true")],
        )

        out = _load_em_gold(em_dir, [("metacritic", "dbpedia")])

        assert set(out.keys()) == {("metacritic", "dbpedia")}
        frame = out[("metacritic", "dbpedia")]
        assert len(frame) == 1
        # Swap also applies to the reverse `_test` fallback path.
        assert frame.iloc[0]["id1"] == "metacritic_1"
        assert frame.iloc[0]["id2"] == "dbpedia_1"
