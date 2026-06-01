"""Unit tests for silver_standard loaders."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pandas as pd
import pytest

from PyDI.evaluation.silver_standard import (
    SilverStandard,
    load_synthetic_silver,
    load_workflow_silver,
)

SAMPLE_XML = textwrap.dedent("""<?xml version='1.0' encoding='utf-8'?>
    <releases>
      <release>
        <id>mbrainz_1</id>
        <name provenance="mbrainz_1">Album A</name>
        <artist provenance="mbrainz_1+discogs_1">Artist A</artist>
        <tracks provenance="discogs_1">['T1', 'T2']</tracks>
      </release>
      <release>
        <id>mbrainz_2</id>
        <name provenance="mbrainz_2">Album B</name>
        <artist provenance="mbrainz_2">Artist B</artist>
      </release>
    </releases>
    """).strip()


class TestLoadWorkflowSilver:
    def test_round_trips_fields(self, tmp_path: Path):
        fusion_dir = tmp_path / "input" / "fusion"
        fusion_dir.mkdir(parents=True)
        (fusion_dir / "test_set.xml").write_text(SAMPLE_XML)

        silver = load_workflow_silver(
            tmp_path,
            prefix_map={"mbrainz_": "musicbrainz", "discogs_": "discogs"},
        )

        assert isinstance(silver, SilverStandard)
        assert sorted(silver.fused["cluster_id"]) == ["mbrainz_1", "mbrainz_2"]
        assert set(silver.fused.columns) >= {"cluster_id", "name", "artist", "tracks"}

        tracks_row = silver.fused.set_index("cluster_id").loc["mbrainz_1"]
        assert tracks_row["tracks"] == ["T1", "T2"]

        member_sources = (
            silver.membership.groupby("cluster_id")["source"].apply(set).to_dict()
        )
        assert member_sources["mbrainz_1"] == {"musicbrainz", "discogs"}

    def test_provenance_table_carries_composite(self, tmp_path: Path):
        fusion_dir = tmp_path / "input" / "fusion"
        fusion_dir.mkdir(parents=True)
        (fusion_dir / "test_set.xml").write_text(SAMPLE_XML)

        silver = load_workflow_silver(
            tmp_path,
            prefix_map={"mbrainz_": "musicbrainz", "discogs_": "discogs"},
        )
        prov = silver.cell_provenance
        assert prov is not None
        artist_row = prov[
            (prov["cluster_id"] == "mbrainz_1") & (prov["attribute"] == "artist")
        ].iloc[0]
        assert artist_row["source_ids"] == ["mbrainz_1", "discogs_1"]


class TestLoadSyntheticSilver:
    def test_pivots_long_csv_and_emits_membership(self, tmp_path: Path):
        baselines_dir = tmp_path / "baselines"
        domain_dir = baselines_dir / "toydomain"
        domain_dir.mkdir(parents=True)

        long_silver = pd.DataFrame(
            [
                {
                    "cluster_id": "src1_1",
                    "attribute": "title",
                    "fused_value": "X",
                    "source_ids": "src1_1,src2_1",
                },
                {
                    "cluster_id": "src1_1",
                    "attribute": "year",
                    "fused_value": "1990",
                    "source_ids": "src1_1,src2_1",
                },
                {
                    "cluster_id": "src1_2",
                    "attribute": "title",
                    "fused_value": "Y",
                    "source_ids": "src1_2",
                },
                {
                    "cluster_id": "src1_2",
                    "attribute": "year",
                    "fused_value": "2000",
                    "source_ids": "src1_2",
                },
            ]
        )
        long_silver.to_csv(domain_dir / "fusion_silver_standard.csv", index=False)

        silver = load_synthetic_silver(
            "toydomain",
            baselines_dir=baselines_dir,
            prefix_map={"src1_": "src1", "src2_": "src2"},
        )

        assert isinstance(silver, SilverStandard)
        assert silver.cell_provenance is None
        assert sorted(silver.fused["cluster_id"]) == ["src1_1", "src1_2"]
        assert set(silver.fused.columns) == {"cluster_id", "title", "year"}
        cluster1 = silver.fused.set_index("cluster_id").loc["src1_1"]
        assert cluster1["title"] == "X"

        member_lookup = (
            silver.membership.groupby("cluster_id")["record_id"].apply(set).to_dict()
        )
        assert member_lookup["src1_1"] == {"src1_1", "src2_1"}
        assert member_lookup["src1_2"] == {"src1_2"}
