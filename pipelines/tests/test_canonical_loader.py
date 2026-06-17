"""Regression tests for the canonical products bundle loader.

The 2026-06-01 directive is that the BoB pipeline evaluates against
canonical PyDI ``usecases/<domain>/`` train/val/test sets. Music /
games / companies already do via the default loader; products needs
the dedicated ``canonical_loader`` because the synthetic
``data_root`` override otherwise diverts it to
``usecases_synthetic/usecases/products/``.

These tests assert that:
1. ``load_pipeline_bundle(domain, bundle_source="canonical")`` for
   products lands on the canonical tree.
2. The bundle's content matches the synthetic translation modulo
   ID prefix + label format (row counts, key consistency).
3. The fusion silver carries the FULL hardware schema (the 27-attr
   silver the workflow notebook scores against), not the 6-attr
   synthetic XML silver.
4. All four domains resolve to canonical when the config says so.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pipelines.lib.bundle import load_pipeline_bundle

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_products_canonical_bundle_lands_on_usecases_tree() -> None:
    bundle = load_pipeline_bundle("products", bundle_source="canonical")
    assert bundle.variant_root == REPO_ROOT / "usecases" / "products"
    assert bundle.domain == "products"
    assert bundle.level == "baseline"


def test_products_canonical_sources_are_id_prefixed_and_raw_named() -> None:
    """Sources keep RAW per-source column names so the SM committee
    can score against varied real schemas (manufacturer / brandName /
    Brand / mfr). The pipeline applies the SM gold mapping after SM
    scoring to translate to canonical column names for downstream."""
    bundle = load_pipeline_bundle("products", bundle_source="canonical")
    assert set(bundle.sources.keys()) == {
        "products_1",
        "products_2",
        "products_3",
        "products_4",
    }
    expected_raw_brand = {
        "products_1": "manufacturer",
        "products_2": "brandName",
        "products_3": "Brand",
        "products_4": "mfr",
    }
    for name, df in bundle.sources.items():
        prefix = f"{name}_"
        bad = df[~df["id"].astype(str).str.startswith(prefix)]
        assert bad.empty, f"{name}: {len(bad)} ids missing prefix {prefix!r}"
        assert df.attrs.get("dataset_name") == name
        # Raw schema preserved — the canonical loader no longer
        # renames columns at load time.
        assert expected_raw_brand[name] in df.columns, (
            f"{name}: expected raw brand col "
            f"{expected_raw_brand[name]!r} in {list(df.columns)[:10]}..."
        )
        # And canonical name absent until post-SM translation.
        assert "brand" not in df.columns, (
            f"{name}: canonical 'brand' should not be present pre-SM; "
            "the loader leaves columns raw."
        )
        assert df.attrs.get("needs_sm_column_translation") is True


def test_products_canonical_em_gold_matches_synthetic_translation() -> None:
    """Canonical EM gold should be byte-equivalent (modulo prefix +
    label format) to the synthetic-translated copy. Lock this in so
    a future refresh of the canonical tree without a synthetic resync
    is detected loudly."""
    canon = load_pipeline_bundle("products", bundle_source="canonical")
    synth = load_pipeline_bundle("products", bundle_source="synthetic_baseline")
    assert set(canon.em_gold.keys()) == set(synth.em_gold.keys())
    for pair in canon.em_gold:
        c = canon.em_gold[pair].sort_values(["id1", "id2"]).reset_index(drop=True)
        s = synth.em_gold[pair].sort_values(["id1", "id2"]).reset_index(drop=True)
        assert len(c) == len(s), f"{pair}: canonical {len(c)} vs synth {len(s)}"
        # IDs + labels should be byte-identical after canonical loader's
        # in-memory translation.
        pd.testing.assert_frame_equal(c, s, check_dtype=False)


def test_products_canonical_em_splits_present() -> None:
    bundle = load_pipeline_bundle("products", bundle_source="canonical")
    assert ("products_1", "products_2") in bundle.em_splits
    for pair in bundle.em_splits:
        splits = bundle.em_splits[pair]
        # Canonical ships train/val/test/all for products.
        assert {"train", "val", "test", "all"}.issubset(
            splits.keys()
        ), f"{pair}: only got {sorted(splits.keys())}"


def test_products_canonical_fusion_silver_carries_full_hardware_schema() -> None:
    """The 2026-06-01 fix: BoB now scores fusion against the canonical
    27-attribute hardware silver (matching the workflow notebook), not
    the 6-attribute synthetic XML silver."""
    bundle = load_pipeline_bundle("products", bundle_source="canonical")
    fg = bundle.fusion_gold
    # Notebook-scored hardware attributes that MUST be present.
    required = {
        "id",
        "brand",
        "product_type",
        "vram_gb",
        "storage_gb",
        "chipset_name",
        "bus_type",
        "interface_type",
        "memory_type",
    }
    missing = required - set(fg.columns)
    assert not missing, f"canonical fusion silver missing {missing}"
    # IDs should be products_1 anchor-prefixed (id_left was renamed
    # to id).
    bad_ids = fg[~fg["id"].astype(str).str.startswith("products_1_")]
    assert bad_ids.empty, f"{len(bad_ids)} fusion-silver ids not products_1 prefixed"
    # `filled` was the row filter; should be dropped from the silver.
    assert "filled" not in fg.columns


def test_products_canonical_sm_gold_present_at_canonical_path() -> None:
    sm_path = (
        REPO_ROOT
        / "usecases"
        / "products"
        / "input"
        / "schemamatching"
        / "sm_mapping_gold.json"
    )
    assert sm_path.exists(), (
        "SM gold must live at canonical path "
        "usecases/products/input/schemamatching/sm_mapping_gold.json "
        "(see pipelines/lib/canonical_loader.py)."
    )
    bundle = load_pipeline_bundle("products", bundle_source="canonical")
    assert bundle.sm_mapping is not None
    assert not bundle.sm_mapping.empty


@pytest.mark.parametrize("domain", ["music", "games", "companies"])
def test_other_domains_resolve_to_canonical_via_default_loader(
    domain: str,
) -> None:
    """For music / games / companies, both bundle_source values land
    on canonical because no synthetic ``data_root`` override applies."""
    bundle = load_pipeline_bundle(domain, bundle_source="synthetic_baseline")
    assert bundle.variant_root == REPO_ROOT / "usecases" / domain


def test_games_canonical_em_validation_splits_present_and_stratified() -> None:
    bundle = load_pipeline_bundle("games", bundle_source="canonical")
    expected = {
        ("dbpedia", "sales"): {
            "train": {"total": 474, "FALSE": 311, "TRUE": 163},
            "val": {"total": 119, "FALSE": 78, "TRUE": 41},
            "test": {"total": 402, "FALSE": 286, "TRUE": 116},
        },
        ("metacritic", "dbpedia"): {
            "train": {"total": 460, "FALSE": 313, "TRUE": 147},
            "val": {"total": 115, "FALSE": 78, "TRUE": 37},
            "test": {"total": 337, "FALSE": 231, "TRUE": 106},
        },
    }
    assert set(expected).issubset(bundle.em_splits)
    for pair, split_expectations in expected.items():
        splits = bundle.em_splits[pair]
        assert {"train", "val", "test"}.issubset(splits)
        assert "all" not in splits

        for split, counts in split_expectations.items():
            frame = splits[split]
            label_counts = frame["label"].astype(str).str.upper().value_counts()
            assert len(frame) == counts["total"]
            assert int(label_counts.get("FALSE", 0)) == counts["FALSE"]
            assert int(label_counts.get("TRUE", 0)) == counts["TRUE"]

        train_pairs = set(map(tuple, splits["train"][["id1", "id2"]].values))
        val_pairs = set(map(tuple, splits["val"][["id1", "id2"]].values))
        assert train_pairs.isdisjoint(val_pairs)


def test_load_pipeline_bundle_rejects_unknown_bundle_source() -> None:
    with pytest.raises(ValueError, match="Unknown bundle_source"):
        load_pipeline_bundle("products", bundle_source="bogus")
