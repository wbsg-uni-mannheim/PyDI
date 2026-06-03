"""Tests for the fusion silver standard builder (plan_revision.md §4b)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from usecases_synthetic.lib import fusion_silver_standard as fss
from usecases_synthetic.lib.fusion_silver_standard import (
    _COMPANIES_STACK,
    _GAMES_STACK,
    _MUSIC_STACK,
    _build_correspondences,
    _fix_discogs_zero_date,
    _normalize_sources,
    _parse_track_list,
    _source_for_id,
    _sum_duration,
    _value_repr,
    build_pool_clusters,
    build_silver_standard,
    canonical_cluster_id,
    prefer_track_list_by_source,
    silver_path,
    supported_domains,
    write_silver_standard,
)

# ---------------------------------------------------------------------------
# Helpers + parsing utilities (verbatim from the music notebook)
# ---------------------------------------------------------------------------


class TestParseTrackList:
    def test_python_list_literal(self) -> None:
        assert _parse_track_list("['Track A', 'Track B']") == ["Track A", "Track B"]

    def test_actual_list_passthrough(self) -> None:
        assert _parse_track_list(["Track A", "Track B"]) == ["Track A", "Track B"]

    def test_dedup_casefold(self) -> None:
        # Notebook collapses near-duplicates by casefolded whitespace-normalised key.
        assert _parse_track_list("['Track A', 'TRACK  A']") == ["Track A"]

    def test_pipe_separated_fallback(self) -> None:
        # When the literal-eval fails, splits on '|'.
        assert _parse_track_list("Track A | Track B") == ["Track A", "Track B"]

    def test_nan_returns_empty(self) -> None:
        assert _parse_track_list(np.nan) == []
        assert _parse_track_list(None) == []
        assert _parse_track_list("") == []

    def test_skips_blank_items(self) -> None:
        assert _parse_track_list("['Track A', '', '   ']") == ["Track A"]


class TestFixDiscogsZeroDate:
    def test_replaces_only_zero_segments(self) -> None:
        assert _fix_discogs_zero_date("1998-00-00") == "1998-01-01"
        assert _fix_discogs_zero_date("1998-12-00") == "1998-12-01"
        assert _fix_discogs_zero_date("1998-12-14") == "1998-12-14"

    def test_non_string_passthrough(self) -> None:
        assert _fix_discogs_zero_date(None) is None
        assert _fix_discogs_zero_date(1998) == 1998


class TestSumDuration:
    def test_list_sum(self) -> None:
        assert _sum_duration(["10", "20", "30"]) == 60

    def test_int_passthrough(self) -> None:
        assert _sum_duration(42) == 42
        assert _sum_duration("42") == 42

    def test_invalid_returns_nan(self) -> None:
        result = _sum_duration("not a number")
        assert isinstance(result, float) and np.isnan(result)


class TestPreferTrackListBySource:
    def test_picks_highest_priority_source(self) -> None:
        # musicbrainz (priority 0) wins over discogs (1) and lastfm (2).
        values = [["A", "B"], ["X", "Y", "Z"], ["P"]]
        sources = ["mbrainz_1", "discogs_3", "lastFM_1"]
        source_datasets = {
            "mbrainz_1": "musicbrainz",
            "discogs_3": "discogs",
            "lastFM_1": "lastfm",
        }
        value, conf, meta = prefer_track_list_by_source(
            values, sources=sources, source_datasets=source_datasets
        )
        assert value == ["A", "B"]
        assert conf == 1.0
        assert meta["selected_dataset"] == "musicbrainz"
        assert meta["selected_record_id"] == "mbrainz_1"

    def test_falls_through_to_lower_priority(self) -> None:
        # No musicbrainz available; discogs wins.
        values = [None, ["X", "Y", "Z"], ["P"]]
        sources = ["mbrainz_1", "discogs_3", "lastFM_1"]
        source_datasets = {
            "mbrainz_1": "musicbrainz",
            "discogs_3": "discogs",
            "lastFM_1": "lastfm",
        }
        value, conf, meta = prefer_track_list_by_source(
            values, sources=sources, source_datasets=source_datasets
        )
        assert value == ["X", "Y", "Z"]
        # Confidence drops at priority 1: 1 - 0.2*1 = 0.8.
        assert conf == pytest.approx(0.8)
        assert meta["selected_dataset"] == "discogs"

    def test_no_valid_tracks_returns_none(self) -> None:
        value, conf, meta = prefer_track_list_by_source(
            [None, None, ""], sources=["a", "b", "c"], source_datasets={}
        )
        assert value is None
        assert conf == 0.0
        assert meta["reason"] == "no_valid_track_lists"


# ---------------------------------------------------------------------------
# Cluster construction + canonical id picker
# ---------------------------------------------------------------------------


class TestSourceForId:
    def test_music_prefixes(self) -> None:
        prefix_map = _MUSIC_STACK.id_prefix_to_source
        assert _source_for_id("mbrainz_42", prefix_map) == "musicbrainz"
        assert _source_for_id("discogs_42", prefix_map) == "discogs"
        assert _source_for_id("lastFM_42", prefix_map) == "lastfm"
        assert _source_for_id("unknown_42", prefix_map) is None


class TestCanonicalClusterId:
    def test_picks_highest_trust_source(self) -> None:
        # musicbrainz trust=3, lastfm=2, discogs=1.
        cluster = ["discogs_5", "lastFM_2", "mbrainz_1"]
        assert canonical_cluster_id(cluster, _MUSIC_STACK) == "mbrainz_1"

    def test_picks_lex_min_within_highest_trust(self) -> None:
        # Two musicbrainz members -> pick the lex-smaller id.
        cluster = ["mbrainz_2", "mbrainz_1", "discogs_5"]
        assert canonical_cluster_id(cluster, _MUSIC_STACK) == "mbrainz_1"

    def test_falls_back_to_lex_when_no_known_source(self) -> None:
        cluster = ["foo_2", "foo_1"]
        assert canonical_cluster_id(cluster, _MUSIC_STACK) == "foo_1"

    def test_empty_cluster_raises(self) -> None:
        with pytest.raises(ValueError):
            canonical_cluster_id([], _MUSIC_STACK)


class TestBuildCorrespondences:
    def test_hub_and_spoke(self) -> None:
        clusters = {
            "a": {"a", "b", "c"},
            "x": {"x", "y"},
        }
        df = _build_correspondences(clusters)
        # Hub a -> {b, c}; hub x -> {y}. 3 edges total.
        assert len(df) == 3
        assert set(df.columns) == {"id1", "id2", "score"}
        assert (df["score"] == 1.0).all()
        # Hub-and-spoke means id1 is repeated for the hub:
        assert "a" in set(df["id1"]) and "x" in set(df["id1"])

    def test_singleton_emits_self_edge(self) -> None:
        clusters = {"a": {"a"}}
        df = _build_correspondences(clusters)
        assert len(df) == 1
        assert df.iloc[0]["id1"] == df.iloc[0]["id2"] == "a"

    def test_empty_returns_empty_frame(self) -> None:
        df = _build_correspondences({})
        assert df.empty
        assert list(df.columns) == ["id1", "id2", "score"]


# ---------------------------------------------------------------------------
# Value-repr (CSV display safety)
# ---------------------------------------------------------------------------


class TestValueRepr:
    def test_nan_returns_empty_string(self) -> None:
        assert _value_repr(float("nan")) == ""
        assert _value_repr(None) == ""

    def test_list_uses_json(self) -> None:
        assert _value_repr(["a", "b"]) == '["a", "b"]'

    def test_timestamp_iso(self) -> None:
        ts = pd.Timestamp("2024-01-02T03:04:05")
        assert _value_repr(ts) == "2024-01-02T03:04:05"

    def test_scalar_str(self) -> None:
        assert _value_repr(1055.0) == "1055.0"
        assert _value_repr("hello") == "hello"


# ---------------------------------------------------------------------------
# End-to-end build determinism (music)
# ---------------------------------------------------------------------------


def _domain_inputs_present(domain: str, source_file: str) -> bool:
    repo = Path(__file__).resolve().parents[2]
    return (repo / "usecases" / domain / "input" / "data" / source_file).exists() and (
        repo / "usecases_synthetic" / "pools" / domain / "pooled_positives.csv"
    ).exists()


_skip_if_no_music_inputs = pytest.mark.skipif(
    not _domain_inputs_present("music", "musicbrainz.csv"),
    reason="music baseline sources or pool not present in this checkout",
)
_skip_if_no_games_inputs = pytest.mark.skipif(
    not _domain_inputs_present("games", "dbpedia.csv"),
    reason="games baseline sources or pool not present in this checkout",
)
_skip_if_no_companies_inputs = pytest.mark.skipif(
    not _domain_inputs_present("companies", "dbpedia.csv"),
    reason="companies baseline sources or pool not present in this checkout",
)


class TestSupportedDomains:
    def test_wired_domains(self) -> None:
        # products wired with the 2026-06-02 data_cleaned_final schema;
        # papers wired 2026-06-03 (jsonl sources, Version_5 fusion stack).
        assert supported_domains() == [
            "companies",
            "games",
            "music",
            "papers",
            "products",
        ]

    def test_unsupported_raises_friendly(self) -> None:
        # movies has no fusion stack (no per-domain workflow notebook).
        with pytest.raises(NotImplementedError, match="not yet wired"):
            build_silver_standard("movies")


@_skip_if_no_music_inputs
class TestMusicSilverBuild:
    """End-to-end build of the music silver standard.

    Slow tests — they load the full music source set + pool. Skipped
    when the music inputs aren't present (e.g. CI subset checkouts).
    """

    @pytest.fixture(scope="class")
    def silver(self) -> pd.DataFrame:
        return build_silver_standard("music")

    def test_silver_has_expected_columns(self, silver: pd.DataFrame) -> None:
        expected = {
            "cluster_id",
            "attribute",
            "fused_value",
            "fused_value_repr",
            "fusion_rule",
            "confidence",
            "source_ids",
            "num_sources",
        }
        assert expected.issubset(silver.columns)

    def test_silver_is_nonempty(self, silver: pd.DataFrame) -> None:
        # Music pool yields ~4280 clusters × 7 attrs ≈ 29960 rows.
        assert len(silver) > 0
        assert silver["cluster_id"].nunique() > 0

    def test_silver_covers_all_attributes(self, silver: pd.DataFrame) -> None:
        # The fusion stack covers all 7 music attributes.
        assert set(silver["attribute"].unique()) == set(_MUSIC_STACK.attributes)

    def test_silver_fusion_rules_match_stack(self, silver: pd.DataFrame) -> None:
        # Every attribute's fusion_rule should be exactly one value
        # (one strategy per attribute in the silver build).
        rule_per_attr = silver.groupby("attribute")["fusion_rule"].unique()
        for attr, rules in rule_per_attr.items():
            assert len(rules) == 1, f"attribute {attr!r} has multiple rules: {rules}"

    def test_silver_cluster_ids_are_mbrainz_when_available(
        self, silver: pd.DataFrame
    ) -> None:
        # Most pool clusters contain a musicbrainz member (the canonical
        # anchor source). We assert the canonical id starts with the
        # mbrainz prefix in the majority of cases.
        unique_clusters = silver["cluster_id"].drop_duplicates()
        mbrainz_share = unique_clusters.str.startswith("mbrainz_").mean()
        assert mbrainz_share > 0.5, (
            f"expected most clusters to be anchored at musicbrainz; "
            f"got mbrainz share = {mbrainz_share:.2f}"
        )

    def test_silver_build_is_deterministic(self, silver: pd.DataFrame) -> None:
        # Run a second build and assert equality on the load-bearing
        # columns. The fusion stack is fully deterministic when no LLM
        # member is involved (which is the case for music's silver).
        second = build_silver_standard("music")
        join_cols = ["cluster_id", "attribute"]
        merged = silver.merge(
            second,
            on=join_cols,
            suffixes=("_a", "_b"),
            how="outer",
            indicator=True,
        )
        assert (
            merged["_merge"] == "both"
        ).all(), "second build produced different cluster_id × attribute keys"
        # Values may include lists/Timestamps; compare the display repr.
        assert (merged["fused_value_repr_a"] == merged["fused_value_repr_b"]).all()
        assert (merged["fusion_rule_a"] == merged["fusion_rule_b"]).all()

    def test_silver_release_country_canonicalized(self, silver: pd.DataFrame) -> None:
        # Notebook's spec normalises release-country to the canonical
        # name form (e.g. discogs "UK" -> "United Kingdom"). At least
        # one cluster sourced from discogs should land on a canonical
        # name, never the discogs raw "UK" abbreviation.
        country_rows = silver[silver["attribute"] == "release-country"]
        countries = country_rows["fused_value_repr"].dropna().unique()
        assert (
            "UK" not in countries
        ), "discogs 'UK' leaked into silver; country canonicalization failed"

    def test_silver_tracks_are_parsed_lists(self, silver: pd.DataFrame) -> None:
        track_rows = silver[silver["attribute"] == "tracks"]
        # Each value should be a Python list, never a stringified list.
        sample = track_rows.head(20)
        for v in sample["fused_value"]:
            assert isinstance(v, list), f"tracks fused_value is not a list: {v!r}"

    def test_persistence_roundtrip(self, silver: pd.DataFrame, tmp_path: Path) -> None:
        out_dir = tmp_path / "baselines" / "music"
        paths = write_silver_standard("music", silver, out_dir=out_dir)
        assert paths["csv"].exists() and paths["json"].exists()

        # CSV repr column equals the silver's fused_value_repr.
        on_disk = pd.read_csv(paths["csv"], keep_default_na=False, na_values=[""])
        assert len(on_disk) == len(silver)
        assert set(on_disk.columns) == {
            "cluster_id",
            "attribute",
            "fused_value",
            "fusion_rule",
            "confidence",
            "source_ids",
            "num_sources",
        }

        # JSON is well-formed and matches the cluster count.
        with paths["json"].open() as f:
            nested = json.load(f)
        assert len(nested) == silver["cluster_id"].nunique()


@_skip_if_no_games_inputs
class TestGamesSilverBuild:
    """End-to-end build of the games silver standard."""

    @pytest.fixture(scope="class")
    def silver(self) -> pd.DataFrame:
        return build_silver_standard("games")

    def test_silver_covers_all_attributes(self, silver: pd.DataFrame) -> None:
        assert set(silver["attribute"].unique()) == set(_GAMES_STACK.attributes)

    def test_silver_fusion_rules_match_stack(self, silver: pd.DataFrame) -> None:
        # Per attribute, the rule should be exactly the configured strategy
        # name. The empty-string sentinel ``""`` is also allowed because
        # the engine emits no metadata key when the attribute is absent
        # from every cluster member's source (e.g. criticScore for clusters
        # whose only members are from dbpedia, which does not carry
        # criticScore per the gold mapping).
        expected = {
            "name": "voting",
            "platform": "voting",
            "developer": "voting",
            "releaseYear": "voting",
            "ESRB": "prefer_higher_trust",
            "criticScore": "prefer_higher_trust",
            "userScore": "average",
            "genres": "union",
        }
        for attr, rule in expected.items():
            rules = set(silver[silver["attribute"] == attr]["fusion_rule"].unique())
            allowed = {rule, ""}
            assert rules.issubset(allowed), (
                f"games attribute {attr!r} has unexpected rules: {rules}; "
                f"allowed: {allowed}"
            )

    def test_silver_cluster_ids_are_metacritic_when_available(
        self, silver: pd.DataFrame
    ) -> None:
        unique_clusters = silver["cluster_id"].drop_duplicates()
        metacritic_share = unique_clusters.str.startswith("metacritic_").mean()
        assert metacritic_share > 0.5, (
            f"expected most games clusters to be anchored at metacritic; "
            f"got metacritic share = {metacritic_share:.2f}"
        )

    def test_silver_genres_are_parsed_lists(self, silver: pd.DataFrame) -> None:
        genre_rows = silver[silver["attribute"] == "genres"]
        sample = genre_rows.head(20)
        for v in sample["fused_value"]:
            assert isinstance(v, list), f"genres fused_value is not a list: {v!r}"


@_skip_if_no_companies_inputs
class TestCompaniesSilverBuild:
    """End-to-end build of the companies silver standard."""

    @pytest.fixture(scope="class")
    def silver(self) -> pd.DataFrame:
        return build_silver_standard("companies")

    def test_silver_covers_all_attributes(self, silver: pd.DataFrame) -> None:
        assert set(silver["attribute"].unique()) == set(_COMPANIES_STACK.attributes)

    def test_silver_fusion_rules_match_stack(self, silver: pd.DataFrame) -> None:
        # Per attribute, the rule should be exactly the configured strategy
        # name. The empty-string sentinel ``""`` is also allowed because
        # the engine emits no metadata key when the attribute is absent
        # from every cluster member's source (e.g. keypeople on a
        # forbes+fullcontact cluster — gold mapping only routes keypeople
        # from dbpedia).
        expected = {
            "name": "voting",
            "assets": "prefer_higher_trust",
            "revenue": "prefer_higher_trust",
            "keypeople": "union",
            "founded": "prefer_higher_trust",
            "country": "voting",
            "city": "shortest_string",
        }
        for attr, rule in expected.items():
            rules = set(silver[silver["attribute"] == attr]["fusion_rule"].unique())
            allowed = {rule, ""}
            assert rules.issubset(allowed), (
                f"companies attribute {attr!r} has unexpected rules: {rules}; "
                f"allowed: {allowed}"
            )

    def test_silver_cluster_ids_prefer_highest_trust_source(
        self, silver: pd.DataFrame
    ) -> None:
        # Companies clustering pairs are (forbes, dbpedia) and (forbes,
        # fullcontact) — so each cluster contains exactly one non-forbes
        # member at minimum. The cluster_id picker should never anchor
        # on forbes when dbpedia or fullcontact is available.
        unique_clusters = silver["cluster_id"].drop_duplicates()
        forbes_anchored = unique_clusters.str.startswith("http://www.forbes.com/")
        # Cross-reference with source_ids: a cluster anchored on forbes
        # must have ONLY forbes members (no dbpedia, no fullcontact).
        forbes_clusters = silver[
            silver["cluster_id"].isin(unique_clusters[forbes_anchored])
        ].drop_duplicates("cluster_id")
        for _, row in forbes_clusters.iterrows():
            members = str(row["source_ids"]).split(",")
            non_forbes = [
                m for m in members if not m.startswith("http://www.forbes.com/")
            ]
            assert not non_forbes, (
                f"cluster anchored on forbes {row['cluster_id']!r} has "
                f"non-forbes members {non_forbes} — cluster_id picker "
                "should have chosen the higher-trust source"
            )

    def test_silver_keypeople_is_a_list(self, silver: pd.DataFrame) -> None:
        kp_rows = silver[silver["attribute"] == "keypeople"]
        # Most cells will be empty (only dbpedia provides keypeople per the
        # gold mapping) — but every non-empty value should be a list.
        non_empty = [
            v
            for v in kp_rows["fused_value"]
            if v not in (None, "") and not (isinstance(v, float) and np.isnan(v))
        ]
        assert non_empty, "expected at least some keypeople fused values"
        for v in non_empty[:20]:
            assert isinstance(v, list), f"keypeople fused_value is not a list: {v!r}"

    def test_silver_country_canonicalized(self, silver: pd.DataFrame) -> None:
        # spec.country_format=name routes through pycountry's primary name.
        # The raw forbes/dbpedia data uses some short forms (e.g. "USA")
        # that should not survive into the silver.
        country_rows = silver[silver["attribute"] == "country"]
        countries = country_rows["fused_value_repr"].dropna().unique()
        assert (
            "USA" not in countries
        ), "raw 'USA' leaked into silver; country canonicalization failed"


# ---------------------------------------------------------------------------
# Path / config tests
# ---------------------------------------------------------------------------


class TestSilverPath:
    def test_csv_path(self) -> None:
        p = silver_path("music", "csv")
        assert p.name == "fusion_silver_standard.csv"
        assert p.parent.name == "music"

    def test_json_path(self) -> None:
        p = silver_path("music", "json")
        assert p.name == "fusion_silver_standard.json"

    def test_unknown_ext_raises(self) -> None:
        with pytest.raises(ValueError):
            silver_path("music", "xml")
