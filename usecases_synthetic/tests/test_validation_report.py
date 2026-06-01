"""Tests for ``usecases_synthetic.lib.validation_report`` and ``baseline_loader``."""

from __future__ import annotations

import json
from pathlib import Path

from usecases_synthetic.lib.baseline_loader import (
    BaselineMetrics,
    load_baseline,
)
from usecases_synthetic.lib.validation_report import (
    write_metrics_json,
    write_report_md,
)


def _sample_per_stage() -> dict:
    return {
        "sm": {
            "aggregated": {"precision": 0.9, "recall": 0.8, "f1": 0.8471},
            "per_attribute": {
                "name": {"f1": 0.95},
                "country": {"f1": 0.88},
            },
            "per_partition": {
                "dbpedia": {"f1": 0.9},
                "forbes": {"f1": 0.8},
            },
        },
        "em": {
            "aggregated": {"f1": 0.75},
            "per_attribute": {},
            "per_partition": {
                "forbes__dbpedia": {"f1": 0.72},
                "forbes__fullcontact": {"f1": 0.78},
            },
        },
        "fusion": {
            "aggregated": {"accuracy": 0.82},
            "per_attribute": {
                "name": {"accuracy": 0.9},
                "country": {"accuracy": 0.7},
            },
            "per_partition": {},
        },
    }


class TestWriteMetricsJson:
    def test_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "metrics.json"
        payload = _sample_per_stage()
        written = write_metrics_json(
            path,
            domain="companies",
            per_stage=payload,
            meta={"run_id": "abc123"},
        )
        assert written == path
        assert path.exists()

        with open(path, encoding="utf-8") as f:
            loaded = json.load(f)

        assert loaded["domain"] == "companies"
        assert loaded["meta"]["run_id"] == "abc123"
        assert "written_at" in loaded["meta"]
        assert (
            loaded["per_stage"]["sm"]["aggregated"]["precision"]
            == 0.9
        )

    def test_round_trip_via_baseline_loader(self, tmp_path: Path) -> None:
        path = tmp_path / "baseline_metrics.json"
        payload = _sample_per_stage()
        write_metrics_json(path, domain="companies", per_stage=payload)

        loaded = load_baseline("companies", path_override=path)
        assert isinstance(loaded, BaselineMetrics)
        assert loaded.domain == "companies"

        agg_sm = loaded.aggregated("sm")
        assert agg_sm["precision"] == 0.9
        assert agg_sm["f1"] == 0.8471

        per_attr_fusion = loaded.per_attribute("fusion")
        assert per_attr_fusion["name"]["accuracy"] == 0.9

        per_part_em = loaded.per_partition("em")
        assert per_part_em["forbes__dbpedia"]["f1"] == 0.72

        # Missing stage returns empty dicts, not KeyError.
        assert loaded.aggregated("does_not_exist") == {}
        assert loaded.per_attribute("does_not_exist") == {}


class TestWriteReportMd:
    def test_renders_tables(self, tmp_path: Path) -> None:
        path = tmp_path / "report.md"
        payload = _sample_per_stage()
        write_report_md(path, domain="companies", per_stage=payload)
        text = path.read_text(encoding="utf-8")

        # Header and stage sections.
        assert "# Validation report - companies" in text
        assert "## Stage: sm" in text
        assert "## Stage: em" in text
        assert "## Stage: fusion" in text

        # Per-stage aggregated table + per-attribute table.
        assert "SM - aggregated" in text
        assert "SM - per attribute" in text

        # Formatted float values.
        assert "0.9000" in text
        assert "0.8471" in text

    def test_renders_delta_column_with_baseline(self, tmp_path: Path) -> None:
        path = tmp_path / "report.md"
        measured = _sample_per_stage()
        baseline = _sample_per_stage()
        # Simulate a drop on EM.
        baseline["em"]["aggregated"]["f1"] = 0.95

        write_report_md(
            path,
            domain="companies",
            per_stage=measured,
            baseline_per_stage=baseline,
        )
        text = path.read_text(encoding="utf-8")

        # Delta column header present.
        assert "| metric | measured | baseline | delta |" in text
        # EM drop 0.75 - 0.95 = -0.2 renders as "-0.2000".
        assert "-0.2000" in text

    def test_empty_stages_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "report.md"
        payload = {"sm": {"aggregated": {"f1": 0.5}, "per_attribute": {}}}
        write_report_md(path, domain="companies", per_stage=payload)
        text = path.read_text(encoding="utf-8")
        assert "Stage: sm" in text
        assert "Stage: em" not in text
