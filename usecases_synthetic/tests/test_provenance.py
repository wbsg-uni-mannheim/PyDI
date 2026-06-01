"""Tests for ProvenanceLog: append, flush, read round-trip, schema, perf."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pytest

from usecases_synthetic.lib.provenance import PROVENANCE_COLUMNS, ProvenanceLog


class TestProvenanceLog:
    """ProvenanceLog append/flush/read round-trip tests."""

    def test_append_and_flush_round_trip(self, tmp_output_dir: Path) -> None:
        log = ProvenanceLog(knob=6, level="hard")
        log.append(
            entity_id="dbpedia_42",
            source="dbpedia",
            attribute="name",
            original_value="Apple Inc.",
            new_value="Aple Inc.",
            transform_fn="typo_insert",
            transform_params={"error_rate": 0.05},
        )
        log.append(
            entity_id="forbes_10",
            source="forbes",
            attribute="revenue",
            original_value="1000000",
            new_value="1000001",
            transform_fn="numeric_jitter",
            transform_params={"magnitude": 1},
        )

        csv_path = tmp_output_dir / "knob_06.csv"
        n = log.flush(csv_path)
        assert n == 2
        assert len(log) == 0  # flushed

        df = ProvenanceLog.read(csv_path)
        assert list(df.columns) == PROVENANCE_COLUMNS
        assert len(df) == 2
        assert df.iloc[0]["entity_id"] == "dbpedia_42"
        assert df.iloc[0]["knob"] == 6
        assert df.iloc[0]["level"] == "hard"
        assert df.iloc[1]["transform_fn"] == "numeric_jitter"

    def test_csv_schema_matches_spec(self, tmp_output_dir: Path) -> None:
        log = ProvenanceLog(knob=1, level="easy")
        log.append(
            entity_id="e1",
            source="src",
            attribute="col",
            original_value="a",
            new_value="b",
            transform_fn="fn",
            transform_params="{}",
        )
        csv_path = tmp_output_dir / "schema_test.csv"
        log.flush(csv_path)

        df = pd.read_csv(csv_path)
        assert list(df.columns) == PROVENANCE_COLUMNS

    def test_append_mode(self, tmp_output_dir: Path) -> None:
        csv_path = tmp_output_dir / "append_test.csv"

        log1 = ProvenanceLog(knob=3, level="medium")
        log1.append(
            entity_id="e1", source="s", transform_fn="drop",
        )
        log1.flush(csv_path)

        log2 = ProvenanceLog(knob=3, level="medium")
        log2.append(
            entity_id="e2", source="s", transform_fn="drop",
        )
        log2.flush(csv_path, append=True)

        df = ProvenanceLog.read(csv_path)
        assert len(df) == 2

    def test_merge(self, tmp_output_dir: Path) -> None:
        for i, knob in enumerate([1, 6]):
            log = ProvenanceLog(knob=knob, level="hard")
            log.append(
                entity_id=f"e{i}", source="s", transform_fn=f"fn{i}",
            )
            log.flush(tmp_output_dir / f"knob_{knob:02d}.csv")

        merged_path = tmp_output_dir / "merged.csv"
        merged = ProvenanceLog.merge(
            [tmp_output_dir / "knob_01.csv", tmp_output_dir / "knob_06.csv"],
            merged_path,
        )
        assert len(merged) == 2
        assert set(merged["knob"]) == {1, 6}

    def test_1000_row_performance(self, tmp_output_dir: Path) -> None:
        """Acceptance criterion: 1000 rows append+flush+read in <1s."""
        log = ProvenanceLog(knob=6, level="hard")
        start = time.perf_counter()
        for i in range(1000):
            log.append(
                entity_id=f"entity_{i}",
                source="source_a",
                attribute=f"attr_{i % 10}",
                original_value=f"orig_{i}",
                new_value=f"new_{i}",
                transform_fn="typo_insert",
                transform_params={"idx": i},
            )
        csv_path = tmp_output_dir / "perf_test.csv"
        log.flush(csv_path)
        df = ProvenanceLog.read(csv_path)
        elapsed = time.perf_counter() - start

        assert len(df) == 1000
        assert elapsed < 1.0, f"1000-row round-trip took {elapsed:.2f}s (>1s)"

    def test_empty_values_preserved(self, tmp_output_dir: Path) -> None:
        """Entity-scoped ops have empty attribute/original/new fields."""
        log = ProvenanceLog(knob=2, level="hard")
        log.append(
            entity_id="e1",
            source="s",
            transform_fn="entity_remove",
        )
        csv_path = tmp_output_dir / "empty_test.csv"
        log.flush(csv_path)

        df = ProvenanceLog.read(csv_path)
        assert df.iloc[0]["attribute"] == ""
        assert df.iloc[0]["original_value"] == ""
        assert df.iloc[0]["new_value"] == ""
