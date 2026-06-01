"""R10-G phase 1 / R10-I smoke tests for the variant Ditto retrain script.

The heavy trainer (``_invoke_ditto_train``, which shells out to
``ditto/train.py``) is monkeypatched to a stub, and the wide
committee-scope record builder is stubbed so the test does not depend on
the real per-domain schema. What is exercised: pooling + dedup, the
``train.json.gz`` / ``val.json.gz`` build, that the trainer is invoked with
the *wide* committee field scope (R10-I, not knob-02 canonical_schema),
that the K8-resolved committee column_mapping is passed to the builder, and
placement of the variant checkpoint symlink.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import usecases_synthetic.scripts.ditto.retrain_variant as rv
from usecases_synthetic.lib.domain_config import load_knob_config
from usecases_synthetic.lib.variant_loader import VariantBundle


def _tiny_bundle(level: str, variant_root: Path) -> VariantBundle:
    pair = ("dbpedia", "forbes")
    gold = pd.DataFrame(
        {"id1": ["d1", "d2"], "id2": ["f1", "f2"], "label": ["true", "false"]}
    )
    sources = {
        "dbpedia": pd.DataFrame({"identifier": ["d1", "d2"], "name": ["a", "b"]}),
        "forbes": pd.DataFrame({"Identifier": ["f1", "f2"], "Company": ["a", "b"]}),
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


def _fake_records(
    gold: pd.DataFrame,
    domain: str,
    src1: str,
    src2: str,
    *,
    sources: dict[str, pd.DataFrame] | None = None,
    fields: Any = None,
    column_mapping: Any = None,
    normalize: bool = False,
    id_column: str = "id",
) -> list[dict[str, Any]]:
    return [
        {
            "id_left": str(row.id1),
            "id_right": str(row.id2),
            "pair_id": f"{row.id1}__{row.id2}",
            "label": 1 if str(row.label).lower() == "true" else 0,
        }
        for row in gold.itertuples(index=False)
    ]


def _read_json_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class TestRetrainVariantDittoSmoke:
    def test_builds_data_invokes_trainer_and_places_checkpoint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bundle = _tiny_bundle("medium", tmp_path / "variant")
        monkeypatch.setattr(
            rv,
            "load_variant",
            lambda domain, level, *, root_override=None: bundle,
        )

        calls: dict[str, Any] = {}

        def _capturing_records(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
            # Record the builder kwargs so the test can assert the wide
            # field scope + the K8-resolved committee column_mapping flow
            # through, then delegate to the lightweight fake.
            calls.setdefault("builder_fields", kwargs.get("fields"))
            calls.setdefault("builder_column_mapping", kwargs.get("column_mapping"))
            calls.setdefault("builder_sources", kwargs.get("sources"))
            return _fake_records(*args, **kwargs)

        monkeypatch.setattr(
            rv, "build_ditto_pair_records_committee_scope", _capturing_records
        )
        ckpt_parent = tmp_path / "ckpt"
        monkeypatch.setattr(
            rv,
            "_ditto_variant_dir",
            lambda domain, level: ckpt_parent / domain / f"variant_{level}",
        )

        def fake_train(
            train_json: Path,
            val_json: Path,
            run_parent: Path,
            *,
            fields: str,
            batch_size: int,
            max_len: int,
            max_field_len: int,
            config_path: Path,
        ) -> Path:
            calls.update(
                train_json=train_json,
                val_json=val_json,
                fields=fields,
                batch_size=batch_size,
                max_len=max_len,
                max_field_len=max_field_len,
            )
            best = run_parent / "run_test" / "checkpoints" / "best"
            best.mkdir(parents=True)
            (best / "config.json").write_text("{}", encoding="utf-8")
            return best

        monkeypatch.setattr(rv, "_invoke_ditto_train", fake_train)

        work = tmp_path / "work"
        out = rv.retrain_variant_ditto("companies", "medium", work_dir=work)

        # train.json.gz / val.json.gz built with the pooled records.
        train_recs = _read_json_gz(work / "train.json.gz")
        val_recs = _read_json_gz(work / "val.json.gz")
        assert len(train_recs) == 2  # d1/d2 deduped to 2 rows
        assert len(val_recs) == 1
        assert train_recs[0]["pair_id"] == "d1__f1"

        # R10-I: trainer invoked with the WIDE committee field scope (not
        # knob-02 canonical_schema), and knob-02 still supplies the PLM
        # hyperparameters.
        from usecases_synthetic.scripts.ditto.prepare_em_training_data import (
            committee_ditto_fields,
        )

        knob02 = load_knob_config(2, "companies")
        expected_fields = committee_ditto_fields("companies")
        assert calls["fields"] == ",".join(expected_fields)
        assert calls["builder_fields"] == expected_fields
        assert calls["batch_size"] == int(knob02.get("plm_batch_size", 16))
        assert calls["max_field_len"] == int(knob02.get("plm_max_field_len", 350))

        # The committee column_mapping (K8-resolved via the bundle) flows
        # into the builder; the raw variant sources are passed through.
        assert "dbpedia" in calls["builder_column_mapping"]
        assert calls["builder_sources"] is bundle.sources

        # Variant checkpoint placed as a symlink to the produced best dir.
        assert out == ckpt_parent / "companies" / "variant_medium" / "best"
        assert out.is_symlink()
        assert (out / "config.json").exists()

    def test_baseline_level_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="baseline"):
            rv.retrain_variant_ditto("companies", "baseline")


class TestDedupeRecords:
    def test_unordered_pair_dedup(self) -> None:
        records = [
            {"id_left": "a", "id_right": "b"},
            {"id_left": "b", "id_right": "a"},  # same unordered pair
            {"id_left": "a", "id_right": "c"},
        ]
        out = rv._dedupe_records(records)
        assert len(out) == 2
