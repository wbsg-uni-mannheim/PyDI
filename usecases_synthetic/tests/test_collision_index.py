"""Tests for CollisionIndex."""

from __future__ import annotations

from pathlib import Path

from usecases_synthetic.lib.collision_index import CollisionIndex
from usecases_synthetic.lib.provenance import ProvenanceLog


class TestCollisionIndex:
    """Tests for cell-collision tracking."""

    def test_is_touched_after_provenance(self, tmp_output_dir: Path) -> None:
        log = ProvenanceLog(knob=6, level="hard")
        log.append(
            entity_id="e1",
            source="dbpedia",
            attribute="name",
            original_value="Apple",
            new_value="Aple",
            transform_fn="typo_insert",
            transform_params={"rate": 0.05},
        )
        log.flush(tmp_output_dir / "knob_06.csv")

        idx = CollisionIndex(tmp_output_dir)
        assert idx.is_touched("e1", "dbpedia", "name")
        assert not idx.is_touched("e1", "dbpedia", "revenue")
        assert not idx.is_touched("e2", "dbpedia", "name")

    def test_is_k4_fabricated(self, tmp_output_dir: Path) -> None:
        log = ProvenanceLog(knob=4, level="hard")
        log.append(
            entity_id="e1",
            source="forbes",
            attribute="revenue",
            original_value="",
            new_value="5000000",
            transform_fn="fabricate_value",
            transform_params={"method": "llm"},
        )
        log.flush(tmp_output_dir / "knob_04.csv")

        idx = CollisionIndex(tmp_output_dir)
        assert idx.is_k4_fabricated("e1", "forbes", "revenue")
        assert idx.is_touched("e1", "forbes", "revenue")
        # Non-fabricated cell
        assert not idx.is_k4_fabricated("e1", "forbes", "name")

    def test_empty_provenance_dir(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty_prov"
        empty_dir.mkdir()

        idx = CollisionIndex(empty_dir)
        assert not idx.is_touched("e1", "s", "a")

    def test_nonexistent_dir(self, tmp_path: Path) -> None:
        idx = CollisionIndex(tmp_path / "nonexistent")
        assert not idx.is_touched("e1", "s", "a")

    def test_reload(self, tmp_output_dir: Path) -> None:
        idx = CollisionIndex(tmp_output_dir)
        assert not idx.is_touched("e1", "s", "a")

        # Write provenance after initial load
        log = ProvenanceLog(knob=1, level="easy")
        log.append(
            entity_id="e1",
            source="s",
            attribute="a",
            original_value="x",
            new_value="y",
            transform_fn="paraphrase",
        )
        log.flush(tmp_output_dir / "knob_01.csv")

        # Still stale
        assert not idx.is_touched("e1", "s", "a")

        # After reload
        idx.reload()
        assert idx.is_touched("e1", "s", "a")
