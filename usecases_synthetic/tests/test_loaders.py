"""Tests for the synthetic-side post-load adjustments in
``usecases_synthetic/lib/loaders.py``: XML namespace stripping and
opt-in ``id`` injection for source files that ship without explicit
identifier columns.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pandas as pd
import pytest

from usecases_synthetic.lib.loaders import (
    _ensure_id_column,
    _strip_xml_namespaces,
    load_source,
)


class TestStripXmlNamespaces:
    def test_strips_leading_braced_segment(self) -> None:
        df = pd.DataFrame(
            {
                "{http://example.org/ns}rel_id": ["a", "b"],
                "{http://example.org/ns}title": ["x", "y"],
            }
        )
        out = _strip_xml_namespaces(df)
        assert list(out.columns) == ["rel_id", "title"]

    def test_idempotent_on_clean_columns(self) -> None:
        df = pd.DataFrame({"id": [1, 2], "name": ["a", "b"]})
        out = _strip_xml_namespaces(df)
        assert list(out.columns) == ["id", "name"]
        # Should be a no-op (same object returned when no rename happens).
        assert out is df

    def test_strips_embedded_namespace_segments(self) -> None:
        # PyDI's XML aggregator concatenates parent/child element paths
        # with ``_`` and leaves the namespace embedded mid-string
        # (e.g. ``medium-list_{ns}medium_{ns}position``). Every
        # ``{...}`` segment must be removed, not just the leading one.
        df = pd.DataFrame(
            {
                "medium-list_{http://example.org/ns}medium_{http://example.org/ns}position": [
                    1,
                    2,
                ],
            }
        )
        out = _strip_xml_namespaces(df)
        assert list(out.columns) == ["medium-list_medium_position"]


class TestEnsureIdColumn:
    def test_preserves_existing_id_column(self) -> None:
        df = pd.DataFrame({"id": ["existing_1", "existing_2"], "x": [1, 2]})
        out = _ensure_id_column(df, "ignored_source_name")
        assert list(out["id"]) == ["existing_1", "existing_2"]
        # Column order is preserved when an existing id is found.
        assert list(out.columns) == ["id", "x"]

    def test_injects_when_missing(self) -> None:
        df = pd.DataFrame({"name": ["a", "b", "c"], "x": [1, 2, 3]})
        out = _ensure_id_column(df, "dbpedia")
        assert list(out["id"]) == ["dbpedia_1", "dbpedia_2", "dbpedia_3"]
        # Injected column lands in front.
        assert list(out.columns) == ["id", "name", "x"]

    def test_one_based_indexing(self) -> None:
        # Confirms the convention matches existing EM gold IDs
        # (e.g. games dbpedia gold references ``dbpedia_52062``,
        # which is the 52062nd row 1-indexed on a 65000-row source).
        df = pd.DataFrame({"name": ["row1"]})
        out = _ensure_id_column(df, "src")
        assert list(out["id"]) == ["src_1"]

    def test_does_not_mutate_input(self) -> None:
        df = pd.DataFrame({"name": ["a", "b"]})
        cols_before = list(df.columns)
        _ensure_id_column(df, "dbpedia")
        assert list(df.columns) == cols_before


class TestLoadSourceXmlNamespaceStrip:
    def test_load_source_strips_namespaces_on_xml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Build a minimal namespaced XML on disk and route load_source
        # at it via a stand-in domain directory.
        domain_dir = tmp_path / "usecases" / "synthtest" / "input" / "data"
        domain_dir.mkdir(parents=True)
        xml_path = domain_dir / "ns.xml"
        xml_path.write_text(textwrap.dedent("""\
                <metadata xmlns="http://example.org/ns/v1#">
                  <release>
                    <title>One</title>
                    <rel_id>r1</rel_id>
                  </release>
                  <release>
                    <title>Two</title>
                    <rel_id>r2</rel_id>
                  </release>
                </metadata>
                """))

        # Point USECASES_DIR at our tmp tree for this one call.
        monkeypatch.setattr(
            "usecases_synthetic.lib.loaders.USECASES_DIR",
            tmp_path / "usecases",
        )

        df = load_source(
            domain="synthtest",
            source_name="ns_source",
            source_file="ns.xml",
            source_format="xml",
        )
        # Namespace-prefixed columns must have been stripped.
        assert "title" in df.columns
        assert "rel_id" in df.columns
        assert not any(c.startswith("{") for c in df.columns)
        assert list(df["rel_id"]) == ["r1", "r2"]


class TestLoadSourceIdInjection:
    def test_load_source_injects_id_when_inject_id_true(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Build a JSON source that lacks an ``id`` column.
        domain_dir = tmp_path / "usecases" / "synthtest" / "input" / "data"
        domain_dir.mkdir(parents=True)
        json_path = domain_dir / "no_id.json"
        json_path.write_text('[{"title":"A"},{"title":"B"},{"title":"C"}]')

        monkeypatch.setattr(
            "usecases_synthetic.lib.loaders.USECASES_DIR",
            tmp_path / "usecases",
        )

        df = load_source(
            domain="synthtest",
            source_name="src",
            source_file="no_id.json",
            source_format="json",
            inject_id=True,
        )
        assert list(df["id"]) == ["src_1", "src_2", "src_3"]

    def test_load_source_no_injection_by_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Same source, no inject_id flag — the column must remain
        # absent (preserves backward-compat with companies, where
        # the canonical ID is ``identifier`` / ``Identifier``).
        domain_dir = tmp_path / "usecases" / "synthtest" / "input" / "data"
        domain_dir.mkdir(parents=True)
        json_path = domain_dir / "no_id.json"
        json_path.write_text('[{"title":"A"},{"title":"B"}]')

        monkeypatch.setattr(
            "usecases_synthetic.lib.loaders.USECASES_DIR",
            tmp_path / "usecases",
        )

        df = load_source(
            domain="synthtest",
            source_name="src",
            source_file="no_id.json",
            source_format="json",
        )
        assert "id" not in df.columns


class TestLoadSourceIdColumnRename:
    """Tests for the ``id_column`` rename path used by the 2026-05-04
    refreshed CSV sources whose native id columns ship under semantic
    names (e.g. ``entity_uri`` / ``mc_id``)."""

    def _write_csv(self, tmp_path: Path) -> Path:
        domain_dir = tmp_path / "usecases" / "synthtest" / "input" / "data"
        domain_dir.mkdir(parents=True)
        csv_path = domain_dir / "with_id.csv"
        csv_path.write_text(
            "wiki_ref,title\nsynth_1,One\nsynth_2,Two\n", encoding="utf-8"
        )
        return csv_path

    def test_renames_id_column_to_id(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._write_csv(tmp_path)
        monkeypatch.setattr(
            "usecases_synthetic.lib.loaders.USECASES_DIR",
            tmp_path / "usecases",
        )
        df = load_source(
            domain="synthtest",
            source_name="src",
            source_file="with_id.csv",
            source_format="csv",
            id_column="wiki_ref",
        )
        assert "id" in df.columns
        assert "wiki_ref" not in df.columns
        assert list(df["id"]) == ["synth_1", "synth_2"]

    def test_missing_id_column_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._write_csv(tmp_path)
        monkeypatch.setattr(
            "usecases_synthetic.lib.loaders.USECASES_DIR",
            tmp_path / "usecases",
        )
        with pytest.raises(ValueError, match="not present"):
            load_source(
                domain="synthtest",
                source_name="src",
                source_file="with_id.csv",
                source_format="csv",
                id_column="does_not_exist",
            )

    def test_id_column_already_named_id_is_no_op(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # When the on-disk column is already named ``id``, passing
        # ``id_column="id"`` is allowed and a no-op (rename only fires
        # when the source name differs from the destination name).
        domain_dir = tmp_path / "usecases" / "synthtest" / "input" / "data"
        domain_dir.mkdir(parents=True)
        csv_path = domain_dir / "native_id.csv"
        csv_path.write_text("id,title\nrow_1,One\nrow_2,Two\n", encoding="utf-8")
        monkeypatch.setattr(
            "usecases_synthetic.lib.loaders.USECASES_DIR",
            tmp_path / "usecases",
        )
        df = load_source(
            domain="synthtest",
            source_name="src",
            source_file="native_id.csv",
            source_format="csv",
            id_column="id",
        )
        assert list(df["id"]) == ["row_1", "row_2"]

    def test_collision_with_existing_id_column_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Both ``wiki_ref`` and ``id`` present → renaming would
        # silently shadow the existing ``id``; loader must refuse.
        domain_dir = tmp_path / "usecases" / "synthtest" / "input" / "data"
        domain_dir.mkdir(parents=True)
        csv_path = domain_dir / "collide.csv"
        csv_path.write_text("wiki_ref,id,title\na,b,One\nc,d,Two\n", encoding="utf-8")
        monkeypatch.setattr(
            "usecases_synthetic.lib.loaders.USECASES_DIR",
            tmp_path / "usecases",
        )
        with pytest.raises(ValueError, match="already exists"):
            load_source(
                domain="synthtest",
                source_name="src",
                source_file="collide.csv",
                source_format="csv",
                id_column="wiki_ref",
            )
