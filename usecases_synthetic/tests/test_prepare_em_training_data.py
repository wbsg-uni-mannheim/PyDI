"""Tests for ``usecases_synthetic.scripts.ditto.prepare_em_training_data``.

Focused on the ``value_normalize`` hook on ``_canonical_record`` and the
``normalize`` flag on ``build_ditto_pair_records_from_gold`` — the
plumbing that makes the Ditto A/B retrain on normalized games data
runnable (plan_revision_step4g_findings.md §2).
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from usecases_synthetic.scripts.ditto.prepare_em_training_data import (
    _canonical_record,
    build_ditto_pair_records_committee_scope,
    build_ditto_pair_records_from_gold,
    committee_column_mapping,
    committee_ditto_fields,
    write_committee_fields_sidecar,
)


class TestCanonicalRecordNormalizeHook:
    """``_canonical_record`` with value_normalize applies per-field transforms."""

    def _games_setup(self) -> tuple[dict[str, dict[str, str]], list[str]]:
        attribute_mapping = {
            "dbpedia": {
                "title": "name",
                "system": "platform",
                "genre": "genres",
            }
        }
        canonical_schema = ["name", "platform", "genres"]
        return attribute_mapping, canonical_schema

    def test_no_normalize_yields_raw_string_values(self) -> None:
        attribute_mapping, canonical_schema = self._games_setup()
        row = pd.Series(
            {
                "title": "Doom (2016 video game)",
                "system": "Playstation 3",
                "genre": "Shooter",
            }
        )

        record = _canonical_record(row, "dbpedia", attribute_mapping, canonical_schema)

        assert record == {
            "name": "Doom (2016 video game)",
            "platform": "Playstation 3",
            "genres": "Shooter",
        }

    def test_normalize_applies_per_field_transforms(self) -> None:
        attribute_mapping, canonical_schema = self._games_setup()
        row = pd.Series(
            {
                "title": "Doom (2016 video game)",
                "system": "Playstation 3",
                "genre": "Shooter",
            }
        )
        value_normalize = {
            "platform": lambda v: str(v).lower().replace("playstation 3", "ps3"),
            "name": lambda v: str(v).split(" (")[0].lower(),
        }

        record = _canonical_record(
            row,
            "dbpedia",
            attribute_mapping,
            canonical_schema,
            value_normalize,
        )

        # platform + name transformed; genres untouched (not in map).
        assert record["name"] == "doom"
        assert record["platform"] == "ps3"
        assert record["genres"] == "Shooter"

    def test_normalize_handles_missing_source_columns(self) -> None:
        """value_normalize must not crash on fields whose source column is missing."""
        attribute_mapping, canonical_schema = self._games_setup()
        # Drop the 'system' column — platform should land as "".
        row = pd.Series({"title": "Halo", "genre": "Shooter"})
        value_normalize = {
            "platform": lambda v: f"{v}!",
            "name": lambda v: str(v).upper(),
        }

        record = _canonical_record(
            row,
            "dbpedia",
            attribute_mapping,
            canonical_schema,
            value_normalize,
        )

        # Platform stays empty (column absent → not run through normaliser).
        assert record["platform"] == ""
        assert record["name"] == "HALO"
        assert record["genres"] == "Shooter"

    def test_normalize_handles_nan_input(self) -> None:
        attribute_mapping, canonical_schema = self._games_setup()
        row = pd.Series({"title": "Halo", "system": float("nan"), "genre": "Shooter"})
        value_normalize = {"platform": lambda v: f"normalized:{v}"}

        record = _canonical_record(
            row,
            "dbpedia",
            attribute_mapping,
            canonical_schema,
            value_normalize,
        )

        # NaN is short-circuited before the normaliser runs (preserves
        # the empty-string semantics for missing values).
        assert record["platform"] == ""

    def test_normalize_treats_empty_string_result_as_empty(self) -> None:
        """A normaliser returning empty string is the same as a missing value.

        Matters because games' ``normalize_games_platform`` returns ``""``
        for inputs it can't canonicalise (None / NaN / whitespace) — the
        downstream COL/VAL serialisation must omit the field cleanly
        rather than emit ``platform=""``.
        """
        attribute_mapping, canonical_schema = self._games_setup()
        row = pd.Series({"title": "Halo", "system": "   ", "genre": "Shooter"})
        value_normalize = {
            "platform": lambda v: str(v).strip(),  # collapses "   " → ""
        }

        record = _canonical_record(
            row,
            "dbpedia",
            attribute_mapping,
            canonical_schema,
            value_normalize,
        )

        # Empty string is still recorded (the normaliser ran); the
        # downstream record-serialisation logic in serialise_record_for_pool
        # skips empty cells. Test documents current behaviour — adjust
        # if upstream signature changes.
        assert record["platform"] == ""


class TestBuildDittoPairRecordsNormalize:
    """``build_ditto_pair_records_from_gold(normalize=True)`` plumbing."""

    def test_unknown_domain_normalize_raises(self) -> None:
        """A domain without a configured normaliser must fail loudly.

        Prevents silently shipping raw values when the caller asked for
        normalisation but forgot to wire a domain-specific spec.
        """
        gold = pd.DataFrame({"id1": [], "id2": [], "label": []})
        with patch(
            "usecases_synthetic.scripts.ditto.prepare_em_training_data."
            "_load_knob02_config",
            return_value={
                "id_columns": {"src1": "id", "src2": "id"},
                "attribute_mapping": {},
                "canonical_schema": ["name"],
            },
        ):
            with pytest.raises(ValueError, match="no value normaliser configured"):
                build_ditto_pair_records_from_gold(
                    gold,
                    "domain_without_normaliser",
                    "src1",
                    "src2",
                    normalize=True,
                )

    def test_games_normalize_yields_canonical_platform_and_title(self) -> None:
        """End-to-end: load games sources via the real loader, normalize=True.

        Picks two ids with known platform values that differ in the alias
        map (e.g. "Microsoft Windows" → "pc"). Verifies the normalised
        Ditto record carries the canonical form.
        """
        # Pick gold pairs that exercise the alias map. dbpedia has
        # "Microsoft Windows", "Playstation 3", etc. metacritic uses
        # variants like "PC", "PS3".
        # Use one synthetic positive pair where the platform appears in
        # the alias map. We don't depend on a specific id pair surviving
        # source refreshes — instead, mock the source frames.

        attribute_mapping = {
            "dbpedia": {
                "title": "name",
                "system": "platform",
            },
            "metacritic": {
                "game_title": "name",
                "console": "platform",
            },
        }
        canonical_schema = ["name", "platform"]
        sources = {
            "dbpedia": pd.DataFrame(
                {
                    "id": ["dbpedia_1"],
                    "title": ["Doom (2016 video game)"],
                    "system": ["Microsoft Windows"],
                }
            ),
            "metacritic": pd.DataFrame(
                {
                    "id": ["metacritic_1"],
                    "game_title": ["Doom"],
                    "console": ["PC"],
                }
            ),
        }
        gold = pd.DataFrame(
            {"id1": ["dbpedia_1"], "id2": ["metacritic_1"], "label": ["true"]}
        )

        with patch(
            "usecases_synthetic.scripts.ditto.prepare_em_training_data."
            "_load_knob02_config",
            return_value={
                "id_columns": {"dbpedia": "id", "metacritic": "id"},
                "attribute_mapping": attribute_mapping,
                "canonical_schema": canonical_schema,
            },
        ), patch(
            "usecases_synthetic.scripts.ditto.prepare_em_training_data."
            "load_domain_sources",
            return_value=sources,
        ):
            records_raw = build_ditto_pair_records_from_gold(
                gold, "games", "dbpedia", "metacritic", normalize=False
            )
            records_norm = build_ditto_pair_records_from_gold(
                gold, "games", "dbpedia", "metacritic", normalize=True
            )

        assert len(records_raw) == len(records_norm) == 1

        # Raw: platforms differ ("Microsoft Windows" vs "PC"); names
        # carry the parenthetical.
        assert records_raw[0]["platform_left"] == "Microsoft Windows"
        assert records_raw[0]["platform_right"] == "PC"
        assert records_raw[0]["name_left"] == "Doom (2016 video game)"
        assert records_raw[0]["name_right"] == "Doom"

        # Normalized: both platforms canonicalise to "pc"; the
        # parenthetical (including the year) is stripped by the title
        # regex — both sides converge on "doom".
        assert records_norm[0]["platform_left"] == "pc"
        assert records_norm[0]["platform_right"] == "pc"
        assert records_norm[0]["name_left"] == "doom"
        assert records_norm[0]["name_right"] == "doom"

        # Labels + ids preserved across the two modes.
        assert records_raw[0]["label"] == records_norm[0]["label"] == 1
        assert (
            records_raw[0]["pair_id"]
            == records_norm[0]["pair_id"]
            == "dbpedia_1__metacritic_1"
        )


# ---------------------------------------------------------------------------
# R10-I: committee-scope (wide) WDC record builder + helpers.
# ---------------------------------------------------------------------------


class TestCommitteeDittoFields:
    """``committee_ditto_fields`` == DOMAIN_TEXT_COLS minus reserved names."""

    def test_products_is_full_wide_list(self) -> None:
        from usecases_synthetic.lib.sc_block_train import DOMAIN_TEXT_COLS

        fields = committee_ditto_fields("products")
        assert fields == DOMAIN_TEXT_COLS["products"]
        assert "price" in fields and "form_factor" in fields and len(fields) == 19

    def test_music_drops_reserved_label(self) -> None:
        from usecases_synthetic.lib.sc_block_train import DOMAIN_TEXT_COLS

        fields = committee_ditto_fields("music")
        # 'label' is the music record-label attribute AND a Ditto WDC
        # reserved metadata key — it must be dropped from the Ditto scope.
        assert "label" in DOMAIN_TEXT_COLS["music"]
        assert "label" not in fields
        assert fields == [f for f in DOMAIN_TEXT_COLS["music"] if f != "label"]
        assert "name" in fields and "tracks" in fields

    def test_games_companies_have_no_reserved_collision(self) -> None:
        from usecases_synthetic.lib.sc_block_train import DOMAIN_TEXT_COLS

        for d in ("games", "companies"):
            assert committee_ditto_fields(d) == DOMAIN_TEXT_COLS[d]

    def test_alias_domain_resolves(self) -> None:
        assert committee_ditto_fields("companies-small") == committee_ditto_fields(
            "companies"
        )

    def test_unknown_domain_raises(self) -> None:
        with pytest.raises(KeyError, match="DOMAIN_TEXT_COLS"):
            committee_ditto_fields("not_a_domain")


class TestCommitteeColumnMapping:
    """``committee_column_mapping`` reads the matching committee YAML block."""

    def test_companies_maps_heterogeneous_columns(self) -> None:
        cm = committee_column_mapping("companies")
        assert cm["dbpedia"]["org_name"] == "name"
        assert cm["dbpedia"]["nation"] == "country"
        assert cm["forbes"]["company"] == "name"

    def test_products_is_identity(self) -> None:
        cm = committee_column_mapping("products")
        # products sources share the canonical schema → empty per-source map.
        assert all(m == {} for m in cm.values())


class TestBuildCommitteeScope:
    """``build_ditto_pair_records_committee_scope`` projects the wide scope."""

    def _sources(self) -> dict[str, pd.DataFrame]:
        return {
            "dbpedia": pd.DataFrame(
                {
                    "id": ["d1", "d2"],
                    "org_name": ["ACME", "Beta"],
                    "nation": ["US", "DE"],
                }
            ),
            "forbes": pd.DataFrame(
                {
                    "id": ["f1", "f2"],
                    "company": ["ACME Inc", "Beta"],
                    "region": ["US", "DE"],
                }
            ),
        }

    def _gold(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"id1": ["d1", "d2"], "id2": ["f1", "f2"], "label": ["true", "false"]}
        )

    def test_column_mapping_and_wide_projection(self) -> None:
        recs = build_ditto_pair_records_committee_scope(
            self._gold(), "companies", "dbpedia", "forbes", sources=self._sources()
        )
        assert len(recs) == 2
        r0 = recs[0]
        # Heterogeneous source columns renamed onto canonical names.
        assert r0["name_left"] == "ACME" and r0["name_right"] == "ACME Inc"
        assert r0["country_left"] == "US"
        # A canonical field absent on a source serialises as empty.
        assert r0["industry_left"] == "" and r0["industry_right"] == ""
        # Every wide committee field present as {field}_left / _right.
        for fld in committee_ditto_fields("companies"):
            assert f"{fld}_left" in r0 and f"{fld}_right" in r0
        # Labels normalised; WDC metadata present.
        assert r0["label"] == 1 and recs[1]["label"] == 0
        assert r0["pair_id"] == "d1__f1" and r0["is_hard_negative"] == 0

    def test_explicit_fields_and_mapping_override(self) -> None:
        recs = build_ditto_pair_records_committee_scope(
            self._gold(),
            "companies",
            "dbpedia",
            "forbes",
            sources=self._sources(),
            fields=["name", "country"],
            column_mapping={
                "dbpedia": {"org_name": "name", "nation": "country"},
                "forbes": {"company": "name", "region": "country"},
            },
        )
        r0 = recs[0]
        # Only the two explicit fields are serialized (plus the WDC
        # metadata keys id_left/id_right, which are not field columns).
        field_left = {k for k in r0 if k.endswith("_left") and k not in {"id_left"}}
        assert field_left == {"name_left", "country_left"}
        assert "description_left" not in r0 and "industry_left" not in r0

    def test_unknown_id_pairs_dropped(self) -> None:
        gold = pd.DataFrame({"id1": ["d1", "dX"], "id2": ["f1", "f1"], "label": [1, 1]})
        recs = build_ditto_pair_records_committee_scope(
            gold,
            "companies",
            "dbpedia",
            "forbes",
            sources=self._sources(),
            fields=["name"],
            column_mapping={
                "dbpedia": {"org_name": "name"},
                "forbes": {"company": "name"},
            },
        )
        assert len(recs) == 1 and recs[0]["id_left"] == "d1"

    def test_missing_source_raises(self) -> None:
        with pytest.raises(KeyError, match="sources missing"):
            build_ditto_pair_records_committee_scope(
                self._gold(),
                "companies",
                "dbpedia",
                "forbes",
                sources={"dbpedia": self._sources()["dbpedia"]},
                fields=["name"],
                column_mapping={},
            )

    def test_normalize_applies(self) -> None:
        sources = {
            "dbpedia": pd.DataFrame(
                {"id": ["d1"], "name": ["Doom"], "platform": ["Microsoft Windows"]}
            ),
            "metacritic": pd.DataFrame(
                {"id": ["m1"], "name": ["Doom"], "platform": ["PC"]}
            ),
        }
        gold = pd.DataFrame({"id1": ["d1"], "id2": ["m1"], "label": ["true"]})
        recs = build_ditto_pair_records_committee_scope(
            gold,
            "games",
            "dbpedia",
            "metacritic",
            sources=sources,
            fields=["name", "platform"],
            column_mapping={},
            normalize=True,
        )
        # games platform alias map canonicalises both sides to "pc".
        assert recs[0]["platform_left"] == "pc" and recs[0]["platform_right"] == "pc"


class TestFieldsSidecar:
    def test_writes_comma_joined_fields(self, tmp_path) -> None:
        path = write_committee_fields_sidecar(tmp_path, "music")
        assert path == tmp_path / "fields.txt"
        content = path.read_text().strip()
        assert content == ",".join(committee_ditto_fields("music"))
        assert "label" not in content.split(",")


class TestTrainInferenceSerializationEquivalence:
    """The load-bearing R10-I property: a WDC record built for training
    serializes byte-identically to what :class:`DittoMatcher` emits at
    inference off the same column-mapped sources. If this drifts, the
    checkpoint trains on a different surface than it scores."""

    def _equivalence(
        self,
        domain: str,
        src1: str,
        src2: str,
        sources: dict[str, pd.DataFrame],
        gold: pd.DataFrame,
        yaml_fields: list[str],
    ) -> None:
        from usecases_synthetic.lib.column_mapping import apply_column_mapping
        from usecases_synthetic.lib.ditto_matcher import DittoMatcher
        from usecases_synthetic.third_party.ditto_modern.data import (
            wdc_to_pair_examples,
        )

        cm = committee_column_mapping(domain)
        train_fields = committee_ditto_fields(domain)

        # Training-side serialization.
        recs = build_ditto_pair_records_committee_scope(
            gold, domain, src1, src2, sources=sources
        )
        examples = wdc_to_pair_examples(
            pd.DataFrame(recs), fields=train_fields, max_field_len=350
        )

        # Inference-side serialization (committee maps sources, matcher
        # serializes the verbatim YAML field list).
        dfl = (
            apply_column_mapping(sources[src1], cm.get(src1, {}))
            if cm.get(src1)
            else sources[src1].copy()
        )
        dfr = (
            apply_column_mapping(sources[src2], cm.get(src2, {}))
            if cm.get(src2)
            else sources[src2].copy()
        )
        matcher = DittoMatcher(
            checkpoint_path="/tmp/nonexistent-ckpt",
            fields=yaml_fields,
            cache_dir=False,
        )
        li = dfl.set_index("id", drop=False)
        ri = dfr.set_index("id", drop=False)
        for ex, (_, g) in zip(examples, gold.iterrows(), strict=True):
            left, right = matcher._pair_text(
                li.loc[str(g["id1"])], ri.loc[str(g["id2"])]
            )
            assert ex.left == left, (domain, ex.left, left)
            assert ex.right == right, (domain, ex.right, right)

    def test_companies_heterogeneous_mapping(self) -> None:
        sources = {
            "dbpedia": pd.DataFrame(
                {
                    "id": ["d1"],
                    "org_name": ["ACME"],
                    "nation": ["US"],
                    "established": ["1990"],
                }
            ),
            "forbes": pd.DataFrame(
                {
                    "id": ["f1"],
                    "company": ["ACME Inc"],
                    "region": ["US"],
                    "sales_figure": ["5B"],
                }
            ),
        }
        gold = pd.DataFrame({"id1": ["d1"], "id2": ["f1"], "label": ["true"]})
        self._equivalence(
            "companies",
            "dbpedia",
            "forbes",
            sources,
            gold,
            yaml_fields=[
                "name",
                "country",
                "city",
                "industry",
                "sector",
                "founded",
                "keypeople",
                "assets",
                "revenue",
            ],
        )

    def test_music_reserved_label_dropped_both_ends(self) -> None:
        sources = {
            "musicbrainz": pd.DataFrame(
                {
                    "id": ["m1"],
                    "name": ["Thriller"],
                    "artist": ["MJ"],
                    "label": ["Epic"],
                    "genre": ["Pop"],
                }
            ),
            "discogs": pd.DataFrame(
                {
                    "id": ["g1"],
                    "name": ["Thriller"],
                    "artist": ["Michael J"],
                    "label": ["Epic Rec"],
                    "genre": ["Pop"],
                }
            ),
        }
        gold = pd.DataFrame({"id1": ["m1"], "id2": ["g1"], "label": ["true"]})
        self._equivalence(
            "music",
            "musicbrainz",
            "discogs",
            sources,
            gold,
            yaml_fields=[
                "name",
                "artist",
                "release-date",
                "release-country",
                "duration",
                "label",
                "genre",
                "tracks",
            ],
        )

    def test_missing_cells_drop_consistently(self) -> None:
        """Float NaN and None cells must be DROPPED at both ends (never
        serialized as 'nan'/'<NA>'). The wide products scope has many sparse
        numeric columns (price, vram_gb, ...) loaded as float NaN."""
        import numpy as np

        sources = {
            "products_1": pd.DataFrame(
                {
                    "id": ["p1"],
                    "title": ["GPU A"],
                    "price": [np.nan],  # float NaN
                    "brand": [None],  # None
                    "vram_gb": [8.0],
                }
            ),
            "products_2": pd.DataFrame(
                {
                    "id": ["q1"],
                    "title": ["GPU A"],
                    "price": [299.0],
                    "brand": ["ACME"],
                    "vram_gb": [np.nan],
                }
            ),
        }
        gold = pd.DataFrame({"id1": ["p1"], "id2": ["q1"], "label": ["true"]})
        # products committee fields verbatim (identity column_mapping).
        yaml_fields = committee_ditto_fields("products")
        self._equivalence(
            "products", "products_1", "products_2", sources, gold, yaml_fields
        )
        # And assert the NaN/None fields are genuinely absent from the text
        # (not serialized as "nan"): build one record + serialize.
        from usecases_synthetic.lib.ditto_matcher import DittoMatcher

        m = DittoMatcher(
            checkpoint_path="/tmp/nonexistent-ckpt", fields=yaml_fields, cache_dir=False
        )
        left, _ = m._pair_text(
            sources["products_1"].set_index("id", drop=False).loc["p1"],
            sources["products_2"].set_index("id", drop=False).loc["q1"],
        )
        assert "nan" not in left.lower() and "price" not in left and "brand" not in left
        assert "COL title VAL GPU A" in left and "COL vram_gb VAL 8" in left
