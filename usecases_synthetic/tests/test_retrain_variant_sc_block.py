"""R10-G phase 1 smoke tests for the variant SC-Block retrain script.

The heavy trainer (``_invoke_scblock_train``, which calls
``sc_block.train.train``) is monkeypatched to a stub. What is exercised:
the K8-resolved variant-source mapping, collection of the per-pair
corner_filled splits into the ``data_override`` payload, and that the
trainer is invoked with the variant output directory so its ``best``
symlink lands where the committee runner reads it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import usecases_synthetic.scripts.sc_block.retrain_variant as rv
from usecases_synthetic.lib.sc_block_train import DOMAIN_TEXT_COLS
from usecases_synthetic.lib.variant_loader import VariantBundle


def _tiny_bundle(level: str, variant_root: Path) -> VariantBundle:
    pair = ("dbpedia", "forbes")
    gold = pd.DataFrame(
        {"id1": ["d1", "d2"], "id2": ["f1", "f2"], "label": ["true", "false"]}
    )
    # Sources already carry the canonical text_cols (name, country) so the
    # blocking column_mapping is effectively a pass-through here.
    sources = {
        "dbpedia": pd.DataFrame(
            {"id": ["d1", "d2"], "name": ["a", "b"], "country": ["US", "DE"]}
        ),
        "forbes": pd.DataFrame(
            {"id": ["f1", "f2"], "name": ["a", "b"], "country": ["US", "DE"]}
        ),
    }
    return VariantBundle(
        domain="companies",
        level=level,
        sources=sources,
        target_schema={},
        sm_mapping=None,
        em_gold={pair: gold},
        em_splits={},
        em_gold_regenerated={
            pair: {
                "train": {"corner_filled": gold},
                "val": {"corner_filled": gold.head(1)},
            }
        },
        fusion_gold=pd.DataFrame(),
        fusion_validation=None,
        pooled_positives=None,
        variant_root=variant_root,
    )


class TestRetrainVariantScBlockSmoke:
    def test_builds_data_invokes_trainer_and_returns_variant_best(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bundle = _tiny_bundle("hard", tmp_path / "variant")
        monkeypatch.setattr(
            rv,
            "load_variant",
            lambda domain, level, *, root_override=None: bundle,
        )
        ckpt_parent = tmp_path / "ckpt"
        monkeypatch.setattr(
            rv,
            "_scblock_variant_dir",
            lambda domain, level: ckpt_parent / domain / f"variant_{level}",
        )

        calls: dict[str, Any] = {}

        def fake_train(
            domain: str,
            eval_pair: tuple[str, str],
            output_dir: Path,
            data_override: Any,
        ) -> dict[str, Any]:
            calls.update(
                domain=domain,
                eval_pair=eval_pair,
                output_dir=output_dir,
                data_override=data_override,
            )
            best = output_dir / "best"
            best.mkdir(parents=True, exist_ok=True)
            (best / "config.json").write_text("{}", encoding="utf-8")
            return {"best_val_recall": 1.0}

        monkeypatch.setattr(rv, "_invoke_scblock_train", fake_train)

        out = rv.retrain_variant_sc_block("companies", "hard")

        # Trainer invoked with the variant output dir + a 3-tuple override.
        assert calls["domain"] == "companies"
        assert calls["output_dir"] == ckpt_parent / "companies" / "variant_hard"
        sources_mapped, em_train_by_pair, em_splits_by_pair = calls["data_override"]
        # Variant sources carry the canonical text_cols.
        for col in DOMAIN_TEXT_COLS["companies"]:
            assert col in sources_mapped["dbpedia"].columns
        # corner_filled train + val collected for the pair.
        assert ("dbpedia", "forbes") in em_train_by_pair
        assert "train" in em_splits_by_pair[("dbpedia", "forbes")]
        assert "val" in em_splits_by_pair[("dbpedia", "forbes")]

        # Returned path is the variant best the runner will read.
        assert out == ckpt_parent / "companies" / "variant_hard" / "best"
        assert (out / "config.json").exists()

    def test_out_dir_override_routes_checkpoint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``out_dir`` sends the trainer output to the pipeline-isolated tree,
        bypassing ``_scblock_variant_dir`` (committee cache)."""
        bundle = _tiny_bundle("hard", tmp_path / "variant")
        monkeypatch.setattr(
            rv, "load_variant", lambda domain, level, *, root_override=None: bundle
        )

        def _boom(domain: str, level: str) -> Path:  # pragma: no cover
            raise AssertionError("committee cache path must not be used")

        monkeypatch.setattr(rv, "_scblock_variant_dir", _boom)

        calls: dict[str, Any] = {}

        def fake_train(domain, eval_pair, output_dir, data_override) -> dict[str, Any]:
            calls["output_dir"] = output_dir
            best = output_dir / "best"
            best.mkdir(parents=True, exist_ok=True)
            (best / "config.json").write_text("{}", encoding="utf-8")
            return {"best_val_recall": 1.0}

        monkeypatch.setattr(rv, "_invoke_scblock_train", fake_train)

        isolated = tmp_path / "pipelines" / "companies" / "ckpt" / "variant_hard"
        out = rv.retrain_variant_sc_block("companies", "hard", out_dir=isolated)
        assert calls["output_dir"] == isolated
        assert out == isolated / "best"

    def test_baseline_level_rejected(self) -> None:
        with pytest.raises(ValueError, match="baseline"):
            rv.retrain_variant_sc_block("companies", "baseline")


class TestBuildVariantData:
    def test_maps_sources_and_collects_corner_filled(self, tmp_path: Path) -> None:
        bundle = _tiny_bundle("easy", tmp_path / "variant")
        sources_mapped, em_train_by_pair, em_splits_by_pair = rv._build_variant_data(
            "companies", bundle
        )
        for col in DOMAIN_TEXT_COLS["companies"]:
            assert col in sources_mapped["forbes"].columns
        assert list(em_train_by_pair.keys()) == [("dbpedia", "forbes")]
        assert set(em_splits_by_pair[("dbpedia", "forbes")]) == {"train", "val"}

    def test_missing_train_split_skipped(self, tmp_path: Path) -> None:
        bundle = _tiny_bundle("easy", tmp_path / "variant")
        # Drop the corner_filled train -> the pair should be skipped.
        bundle.em_gold_regenerated[("dbpedia", "forbes")]["train"] = {}
        _, em_train_by_pair, _ = rv._build_variant_data("companies", bundle)
        assert em_train_by_pair == {}
