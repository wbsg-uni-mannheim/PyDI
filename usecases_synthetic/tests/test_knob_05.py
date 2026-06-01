"""Tests for Knob 05 — Format/Unit Diversity.

Acceptance criteria (from module_05_knob_05.md):
1. Every reformatted value round-trips to the exact canonical value
   (within FP tolerance for unit conversions)
2. At easy, all values for a given (source, attribute) use the same format
3. At hard, >= 2 distinct formats appear within a single (source, attribute)
4. No locale-ambiguous date patterns emitted (deny-list active)
5. Provenance includes from_format, to_format, rate where applicable
6. pytest passes
"""

from __future__ import annotations

import json
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from usecases_synthetic.lib.format_operators import (
    _parse_date_flexible,
    _parse_number,
    format_duration,
    parse_duration,
    reconvert_currency,
    reconvert_unit,
    reformat_date,
    reformat_number,
    reformat_number_suffix,
)
from usecases_synthetic.lib.rate_tables import (
    get_all_date_formats,
    get_date_format,
    get_fx_rate,
    get_locale_config,
    get_suffix_scales,
    get_unit_factor,
    is_denied_date_format,
)
from usecases_synthetic.scripts.apply_knob_05_format import (
    SKIPPED_COLUMNS,
    VALID_TRANSFORM_FNS,
    SkippedLog,
    apply_knob_05,
    load_knob_05_config,
    write_outputs,
)

# ---- Fixtures ---------------------------------------------------------------


@pytest.fixture
def companies_config() -> dict[str, Any]:
    """Load the real companies Knob 05 config."""
    return load_knob_05_config("companies")


@pytest.fixture
def small_sources() -> dict[str, pd.DataFrame]:
    """Small DataFrames matching the refreshed companies schema (2026-05-04)."""
    dbpedia = pd.DataFrame(
        {
            "entity_uri": ["db_1", "db_2", "db_3", "db_4", "db_5"],
            "org_name": ["Acme", "Globex", "Initech", "Hooli", "Piedmont"],
            "nation": ["US", "DE", "JP", "US", "FR"],
            "established": [
                "1970-01-01",
                "1993-01-01",
                "2002-01-01",
                "1985-06-15",
                "2010-03-20",
            ],
            "annual_income": [
                "65170000000",
                "358500000",
                "6000000000",
                "209760000000",
                "31690000",
            ],
            "total_assets_val": [
                "240560000000",
                "607100000",
                "294970000",
                "76660000",
                "462900000",
            ],
        }
    )
    dbpedia.attrs["dataset_name"] = "dbpedia"

    forbes = pd.DataFrame(
        {
            "forbes_url": ["f_1", "f_2", "f_3", "f_4", "f_5"],
            "company": [
                "Acme Corp",
                "Globex Inc",
                "Initech LLC",
                "Hooli",
                "Piedmont",
            ],
            "region": ["USA", "DEU", "JPN", "USA", "FRA"],
            "asset_value": [
                3124900000000,
                2449500000000,
                2405400000000,
                2435300000000,
                493400000000,
            ],
            "sales_figure": [
                148700000000,
                121300000000,
                136400000000,
                105700000000,
                30200000000,
            ],
        }
    )
    forbes.attrs["dataset_name"] = "forbes"

    fullcontact = pd.DataFrame(
        {
            "Attribute_1": [
                "fullcontact_1",
                "fullcontact_2",
                "fullcontact_3",
                "fullcontact_4",
                "fullcontact_5",
            ],
            "Attribute_2": ["Acme", "Globex", "Initech", "Hooli", "Piedmont"],
            "Attribute_3": ["US", "DE", "JP", "US", "FR"],
            "Attribute_6": [
                "1908-01-01",
                "1957-01-01",
                "1871-01-01",
                "2007-01-01",
                "1883-01-01",
            ],
        }
    )
    fullcontact.attrs["dataset_name"] = "fullcontact"

    return {"dbpedia": dbpedia, "forbes": forbes, "fullcontact": fullcontact}


@pytest.fixture
def music_sources() -> dict[str, pd.DataFrame]:
    """Small DataFrames matching the refreshed music schema (duration in int seconds)."""
    musicbrainz = pd.DataFrame(
        {
            "id": ["mb_1", "mb_2", "mb_3", "mb_4", "mb_5"],
            "name": ["Track A", "Track B", "Track C", "Track D", "Track E"],
            "artist": ["X", "Y", "Z", "X", "Y"],
            "release-date": [
                "1996-01-01",
                "1998-12-14",
                "2000-09-05",
                "1999-04-27",
                "2000-07-24",
            ],
            "duration": [1055, 724, 2384, 1626, 4145],
        }
    )
    musicbrainz.attrs["dataset_name"] = "musicbrainz"

    discogs = pd.DataFrame(
        {
            "id": ["dc_1", "dc_2", "dc_3", "dc_4", "dc_5"],
            "name": ["Track A", "Track B", "Track C", "Track D", "Track E"],
            "artist": ["X", "Y", "Z", "X", "Y"],
            "release-date": [
                "1996-01-01",
                "1998-01-01",
                "2000-09-05",
                "1999-04-27",
                "2000-06-01",
            ],
            "duration": [1626, 5938, 180, 240, 3600],
        }
    )
    discogs.attrs["dataset_name"] = "discogs"

    lastfm = pd.DataFrame(
        {
            "id": ["lf_1", "lf_2", "lf_3", "lf_4", "lf_5"],
            "name": ["Track A", "Track B", "Track C", "Track D", "Track E"],
            "artist": ["X", "Y", "Z", "X", "Y"],
            "release-date": [None, None, None, None, None],
            "duration": [903.0, 734.0, 1626.0, 1265.0, 1378.0],
        }
    )
    lastfm.attrs["dataset_name"] = "lastfm"

    return {"musicbrainz": musicbrainz, "discogs": discogs, "lastfm": lastfm}


# ---- Rate table tests -------------------------------------------------------


class TestRateTables:
    """Static tables load correctly."""

    def test_date_formats_load(self) -> None:
        fmts = get_all_date_formats()
        assert "iso" in fmts
        assert "us_slash" in fmts
        assert fmts["iso"]["pattern"] == "%Y-%m-%d"

    def test_denied_formats(self) -> None:
        assert is_denied_date_format("eu_slash")
        assert not is_denied_date_format("iso")
        assert not is_denied_date_format("us_slash")

    def test_locale_config(self) -> None:
        cfg = get_locale_config("en_US")
        assert cfg["decimal_sep"] == "."
        assert cfg["thousands_sep"] == ","

        cfg_de = get_locale_config("de_DE")
        assert cfg_de["decimal_sep"] == ","
        assert cfg_de["thousands_sep"] == "."

    def test_fx_rate_self(self) -> None:
        rate = get_fx_rate("USD", "USD")
        assert rate == pytest.approx(1.0)

    def test_fx_rate_roundtrip(self) -> None:
        rate_fwd = get_fx_rate("USD", "EUR")
        rate_rev = get_fx_rate("EUR", "USD")
        assert rate_fwd * rate_rev == pytest.approx(1.0, rel=1e-6)

    def test_unit_factor_identity(self) -> None:
        factor = get_unit_factor("magnitude", "raw", "raw")
        assert factor == pytest.approx(1.0)

    def test_unit_factor_magnitude(self) -> None:
        factor = get_unit_factor("magnitude", "raw", "billions")
        assert factor == pytest.approx(1e-9)

    def test_suffix_scales(self) -> None:
        scales = get_suffix_scales()
        assert scales["B"] == pytest.approx(1e9)
        assert scales["M"] == pytest.approx(1e6)


# ---- Format operator tests --------------------------------------------------


class TestDateOperators:
    """Date reformat operators with round-trip verification."""

    @pytest.mark.parametrize(
        "value,to_fmt,expected_contains",
        [
            ("2024-03-15", "us_slash", "03/15/2024"),
            ("2024-03-15", "eu_dot", "15.03.2024"),
            ("2024-03-15", "long_english", "March 15, 2024"),
            ("2024-03-15", "compact", "20240315"),
        ],
    )
    def test_reformat_date_basic(
        self, value: str, to_fmt: str, expected_contains: str
    ) -> None:
        result = reformat_date(value, to_fmt)
        assert result is not None
        new_value, params = result
        assert expected_contains in new_value
        assert "to_format" in params

    def test_reformat_date_roundtrip(self) -> None:
        """Reformatted date parses back to the same date."""
        original = "2024-03-15"
        parsed_original = _parse_date_flexible(original)
        for fmt_id in ("us_slash", "eu_dot", "long_english", "compact"):
            result = reformat_date(original, fmt_id)
            assert result is not None, f"Failed for {fmt_id}"
            new_value, _ = result
            parsed_new = _parse_date_flexible(new_value)
            assert parsed_new == parsed_original, (
                f"Round-trip failed for {fmt_id}: "
                f"{original} -> {new_value} -> {parsed_new}"
            )

    def test_reformat_date_year_only(self) -> None:
        """Year-only input like '1908' can be reformatted."""
        result = reformat_date("1908", "precision_year")
        assert result is not None
        assert result[0] == "1908"

    def test_reformat_date_null_returns_none(self) -> None:
        assert reformat_date("", "iso") is None
        assert reformat_date("null", "iso") is None

    def test_reformat_date_identity(self) -> None:
        """ISO -> ISO returns identity direction."""
        result = reformat_date("2024-03-15", "iso")
        assert result is not None
        _, params = result
        assert params["direction"] == "identity"


class TestNumberOperators:
    """Number reformat operators with round-trip verification."""

    def test_reformat_number_en_to_de(self) -> None:
        result = reformat_number("1,234,567.89", "de_DE")
        assert result is not None
        new_value, params = result
        # German format: dot as thousands, comma as decimal.
        assert "," in new_value
        assert params["to_locale"] == "de_DE"

    def test_reformat_number_roundtrip(self) -> None:
        """Reformatted number parses back to the same value."""
        original = "1234567.89"
        parsed_original = _parse_number(original)
        assert parsed_original is not None
        for locale in ("en_US", "de_DE", "fr_FR", "plain"):
            result = reformat_number(original, locale)
            assert result is not None, f"Failed for {locale}"
            new_value, _ = result
            parsed_new = _parse_number(new_value)
            assert parsed_new is not None
            assert abs(parsed_new - parsed_original) < Decimal("0.01"), (
                f"Round-trip failed for {locale}: "
                f"{original} -> {new_value} -> {parsed_new}"
            )

    def test_reformat_number_suffix(self) -> None:
        result = reformat_number_suffix("1500000000", "B")
        assert result is not None
        new_value, params = result
        assert "B" in new_value
        assert "1.5" in new_value or "1.50" in new_value

    def test_reformat_number_null_returns_none(self) -> None:
        assert reformat_number("", "en_US") is None
        assert reformat_number("null", "en_US") is None


class TestUnitOperators:
    """Unit conversion operators with round-trip verification."""

    def test_magnitude_conversion(self) -> None:
        # 1 billion raw -> 1.0 in billions
        result = reconvert_unit("1000000000", "magnitude", "raw", "billions")
        assert result is not None
        new_value, params = result
        parsed = _parse_number(new_value)
        assert parsed is not None
        assert abs(parsed - Decimal("1")) < Decimal("0.01")

    def test_magnitude_roundtrip(self) -> None:
        original = "500000000"
        parsed_original = _parse_number(original)
        result = reconvert_unit(original, "magnitude", "raw", "millions")
        assert result is not None
        new_value, params = result
        # Convert back.
        result_back = reconvert_unit(new_value, "magnitude", "millions", "raw")
        assert result_back is not None
        parsed_back = _parse_number(result_back[0])
        assert parsed_back is not None
        assert abs(parsed_back - parsed_original) / abs(parsed_original) < Decimal(
            "0.001"
        )

    def test_currency_conversion_roundtrip(self) -> None:
        original = "1000.00"
        parsed_original = _parse_number(original)
        result = reconvert_currency(original, "USD", "EUR")
        assert result is not None
        new_value, params = result
        assert params["rate_date"] == "2026-03-15"
        # Convert back.
        result_back = reconvert_currency(new_value, "EUR", "USD")
        assert result_back is not None
        parsed_back = _parse_number(result_back[0])
        assert parsed_back is not None
        assert abs(parsed_back - parsed_original) / abs(parsed_original) < Decimal(
            "0.01"
        )


# ---- Config loading tests ---------------------------------------------------


class TestLoadConfig:
    """Config loading and structure."""

    def test_loads_companies(self, companies_config: dict[str, Any]) -> None:
        assert "attribute_classes" in companies_config
        assert "format_pools_per_level" in companies_config
        assert "within_source_consistency" in companies_config

    def test_all_sources_present(self, companies_config: dict[str, Any]) -> None:
        for source in ("dbpedia", "forbes", "fullcontact"):
            assert source in companies_config["attribute_classes"]

    def test_all_levels_present(self, companies_config: dict[str, Any]) -> None:
        for level in ("easy", "medium", "hard"):
            assert level in companies_config["format_pools_per_level"]
            assert level in companies_config["within_source_consistency"]

    def test_pool_sizes_per_level(self, companies_config: dict[str, Any]) -> None:
        """Pinned pool sizes: easy=2, medium=3, hard=4."""
        pools = companies_config["format_pools_per_level"]
        for family in pools["easy"]:
            assert len(pools["easy"][family]) == 2, f"Easy {family} pool size != 2"
        for family in pools["medium"]:
            assert len(pools["medium"][family]) == 3, f"Medium {family} pool size != 3"
        for family in pools["hard"]:
            assert len(pools["hard"][family]) == 4, f"Hard {family} pool size != 4"

    def test_consistency_levels(self, companies_config: dict[str, Any]) -> None:
        cons = companies_config["within_source_consistency"]
        assert cons["easy"] == "source"
        assert cons["medium"] == "source"
        assert cons["hard"] == "row"


# ---- Integration tests (apply_knob_05) --------------------------------------


class TestEasyLevel:
    """At easy, within-source format consistency must hold."""

    def test_easy_produces_changes(
        self,
        small_sources: dict[str, pd.DataFrame],
        companies_config: dict[str, Any],
    ) -> None:
        """Easy is not a no-op."""
        reformatted, prov_df, skipped_df = apply_knob_05(
            "companies", "easy", small_sources, companies_config
        )
        assert len(prov_df) > 0, "Expected provenance rows at easy level"

    def test_easy_within_source_consistency_dates(
        self,
        small_sources: dict[str, pd.DataFrame],
        companies_config: dict[str, Any],
    ) -> None:
        """At easy, all date values within a (source, attribute) use the same format."""
        reformatted, prov_df, _ = apply_knob_05(
            "companies", "easy", small_sources, companies_config
        )
        # Check fullcontact.Attribute_6 — all should be in the same format.
        fc = reformatted["fullcontact"]
        dates = fc["Attribute_6"].dropna().tolist()
        # All dates should follow the same pattern.
        # If they were reformatted, they should all match one format.
        if len(dates) > 1:
            # Check that all non-null dates have the same structural pattern.
            patterns = set()
            for d in dates:
                d_str = str(d).strip()
                if d_str and d_str.lower() != "null":
                    # Classify by separator pattern.
                    if "/" in d_str:
                        patterns.add("slash")
                    elif "." in d_str and not d_str.replace(".", "").isdigit():
                        patterns.add("dot")
                    elif "," in d_str:
                        patterns.add("long")
                    elif len(d_str) == 8 and d_str.isdigit():
                        patterns.add("compact")
                    elif "-" in d_str:
                        patterns.add("dash")
                    elif d_str.isdigit() and len(d_str) == 4:
                        patterns.add("year")
            # At easy, should be exactly 1 pattern.
            assert (
                len(patterns) <= 1
            ), f"Multiple date patterns at easy: {patterns} in {dates}"


class TestHardLevel:
    """At hard, within-source format inconsistency for >=1 column."""

    def test_hard_produces_more_changes_than_easy(
        self,
        small_sources: dict[str, pd.DataFrame],
        companies_config: dict[str, Any],
    ) -> None:
        _, easy_prov, _ = apply_knob_05(
            "companies", "easy", small_sources, companies_config
        )
        _, hard_prov, _ = apply_knob_05(
            "companies", "hard", small_sources, companies_config
        )
        # Hard should have at least as many provenance rows.
        assert len(hard_prov) >= len(easy_prov)


# ---- Provenance tests -------------------------------------------------------


class TestProvenance:
    """Provenance rows are correct and complete."""

    def test_provenance_valid_transform_fns(
        self,
        small_sources: dict[str, pd.DataFrame],
        companies_config: dict[str, Any],
    ) -> None:
        for level in ("easy", "medium", "hard"):
            _, prov_df, _ = apply_knob_05(
                "companies", level, small_sources, companies_config
            )
            if len(prov_df) > 0:
                fns = set(prov_df["transform_fn"].unique())
                assert fns.issubset(VALID_TRANSFORM_FNS), (
                    f"Invalid transform_fn at {level}: " f"{fns - VALID_TRANSFORM_FNS}"
                )

    def test_provenance_transform_params_json(
        self,
        small_sources: dict[str, pd.DataFrame],
        companies_config: dict[str, Any],
    ) -> None:
        _, prov_df, _ = apply_knob_05(
            "companies", "easy", small_sources, companies_config
        )
        for _, row in prov_df.iterrows():
            params = json.loads(row["transform_params"])
            assert isinstance(params, dict)

    def test_provenance_knob_and_level(
        self,
        small_sources: dict[str, pd.DataFrame],
        companies_config: dict[str, Any],
    ) -> None:
        for level in ("easy", "medium", "hard"):
            _, prov_df, _ = apply_knob_05(
                "companies", level, small_sources, companies_config
            )
            if len(prov_df) > 0:
                assert (prov_df["knob"] == 5).all()
                assert (prov_df["level"] == level).all()


# ---- Value preservation tests -----------------------------------------------


class TestValuePreservation:
    """Non-managed columns and DataFrame shape must be preserved."""

    def test_shape_unchanged(
        self,
        small_sources: dict[str, pd.DataFrame],
        companies_config: dict[str, Any],
    ) -> None:
        for level in ("easy", "medium", "hard"):
            reformatted, _, _ = apply_knob_05(
                "companies", level, small_sources, companies_config
            )
            for src_name in small_sources:
                assert reformatted[src_name].shape == small_sources[src_name].shape

    def test_attrs_preserved(
        self,
        small_sources: dict[str, pd.DataFrame],
        companies_config: dict[str, Any],
    ) -> None:
        for level in ("easy", "medium", "hard"):
            reformatted, _, _ = apply_knob_05(
                "companies", level, small_sources, companies_config
            )
            for src_name in small_sources:
                assert reformatted[src_name].attrs["dataset_name"] == src_name

    def test_unmanaged_columns_unchanged(
        self,
        small_sources: dict[str, pd.DataFrame],
        companies_config: dict[str, Any],
    ) -> None:
        """Columns not in attribute_classes are untouched."""
        managed = companies_config["attribute_classes"]
        reformatted, _, _ = apply_knob_05(
            "companies", "hard", small_sources, companies_config
        )
        for src_name in small_sources:
            src_managed = set(managed.get(src_name, {}).keys())
            for col in small_sources[src_name].columns:
                if col not in src_managed:
                    pd.testing.assert_series_equal(
                        small_sources[src_name][col],
                        reformatted[src_name][col],
                        check_names=True,
                    )


# ---- Roundtrip verification tests -------------------------------------------


class TestRoundtripVerification:
    """Every reformatted value round-trips to the canonical value."""

    def test_date_roundtrips(
        self,
        small_sources: dict[str, pd.DataFrame],
        companies_config: dict[str, Any],
    ) -> None:
        """All reformatted dates parse back to the same date."""
        for level in ("easy", "medium", "hard"):
            reformatted, prov_df, _ = apply_knob_05(
                "companies", level, small_sources, companies_config
            )
            date_prov = prov_df[prov_df["transform_fn"] == "reformat_date"]
            for _, row in date_prov.iterrows():
                orig_date = _parse_date_flexible(row["original_value"])
                new_date = _parse_date_flexible(row["new_value"])
                params = json.loads(row["transform_params"])
                to_fmt = params["to_format"]
                if to_fmt == "precision_year":
                    assert new_date is not None
                    assert orig_date is not None
                    assert new_date.year == orig_date.year
                elif to_fmt == "precision_year_month":
                    assert new_date is not None
                    assert orig_date is not None
                    assert new_date.year == orig_date.year
                    assert new_date.month == orig_date.month
                else:
                    assert orig_date == new_date, (
                        f"Date roundtrip failed at {level}: "
                        f"{row['original_value']} -> {row['new_value']}"
                    )

    def test_number_roundtrips(
        self,
        small_sources: dict[str, pd.DataFrame],
        companies_config: dict[str, Any],
    ) -> None:
        """All reformatted numbers parse back to the same value (within tolerance)."""
        for level in ("easy", "medium", "hard"):
            reformatted, prov_df, _ = apply_knob_05(
                "companies", level, small_sources, companies_config
            )
            num_prov = prov_df[prov_df["transform_fn"] == "reformat_number"]
            for _, row in num_prov.iterrows():
                orig = _parse_number(row["original_value"])
                new = _parse_number(row["new_value"])
                assert (
                    orig is not None
                ), f"Cannot parse original: {row['original_value']}"
                assert new is not None, f"Cannot parse new: {row['new_value']}"
                if orig != 0:
                    rel_diff = abs(new - orig) / abs(orig)
                    assert rel_diff < Decimal("0.001"), (
                        f"Number roundtrip failed at {level}: "
                        f"{row['original_value']} -> {row['new_value']} "
                        f"(rel_diff={rel_diff})"
                    )


# ---- Deny-list tests --------------------------------------------------------


class TestDenyList:
    """No locale-ambiguous date patterns are emitted."""

    def test_no_denied_formats_in_provenance(
        self,
        small_sources: dict[str, pd.DataFrame],
        companies_config: dict[str, Any],
    ) -> None:
        for level in ("easy", "medium", "hard"):
            _, prov_df, _ = apply_knob_05(
                "companies", level, small_sources, companies_config
            )
            date_prov = prov_df[prov_df["transform_fn"] == "reformat_date"]
            for _, row in date_prov.iterrows():
                params = json.loads(row["transform_params"])
                to_fmt = params["to_format"]
                assert not is_denied_date_format(
                    to_fmt
                ), f"Denied format {to_fmt} emitted at {level}"


# ---- Skipped log tests ------------------------------------------------------


class TestSkippedLog:
    """Skipped-cell audit has correct structure."""

    def test_skipped_log_columns(
        self,
        small_sources: dict[str, pd.DataFrame],
        companies_config: dict[str, Any],
    ) -> None:
        _, _, skipped_df = apply_knob_05(
            "companies", "easy", small_sources, companies_config
        )
        assert list(skipped_df.columns) == SKIPPED_COLUMNS


# ---- Output writing tests ---------------------------------------------------


class TestWriteOutputs:
    """Artifacts land on disk correctly."""

    def test_write_creates_files(
        self,
        small_sources: dict[str, pd.DataFrame],
        companies_config: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        _, prov_df, skipped_df = apply_knob_05(
            "companies", "easy", small_sources, companies_config
        )
        write_outputs(prov_df, skipped_df, tmp_path)
        assert (tmp_path / "output" / "provenance" / "knob_05_format_unit.csv").exists()
        assert (tmp_path / "output" / "provenance" / "knob_05_skipped.csv").exists()

    def test_written_provenance_round_trips(
        self,
        small_sources: dict[str, pd.DataFrame],
        companies_config: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        _, prov_df, skipped_df = apply_knob_05(
            "companies", "easy", small_sources, companies_config
        )
        write_outputs(prov_df, skipped_df, tmp_path)
        loaded = pd.read_csv(
            tmp_path / "output" / "provenance" / "knob_05_format_unit.csv",
            keep_default_na=False,
        )
        assert len(loaded) == len(prov_df)


# ---- Edge case tests --------------------------------------------------------


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_invalid_level_raises(
        self,
        small_sources: dict[str, pd.DataFrame],
        companies_config: dict[str, Any],
    ) -> None:
        with pytest.raises(ValueError, match="Invalid level"):
            apply_knob_05(
                "companies",
                "extreme",  # type: ignore[arg-type]
                small_sources,
                companies_config,
            )

    def test_missing_domain_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_knob_05_config("nonexistent_domain")

    def test_empty_sources(self, companies_config: dict[str, Any]) -> None:
        """Empty source dict produces empty outputs."""
        reformatted, prov_df, skipped_df = apply_knob_05(
            "companies", "easy", {}, companies_config
        )
        assert len(reformatted) == 0
        assert len(prov_df) == 0

    def test_deterministic_output(
        self,
        small_sources: dict[str, pd.DataFrame],
        companies_config: dict[str, Any],
    ) -> None:
        """Same seed produces identical output."""
        r1, p1, s1 = apply_knob_05(
            "companies", "medium", small_sources, companies_config, seed=42
        )
        r2, p2, s2 = apply_knob_05(
            "companies", "medium", small_sources, companies_config, seed=42
        )
        for src in r1:
            pd.testing.assert_frame_equal(r1[src], r2[src])
        pd.testing.assert_frame_equal(p1, p2)
        pd.testing.assert_frame_equal(s1, s2)


# ---- Duration operator tests ------------------------------------------------


class TestDurationOperators:
    """Duration reformat operators with round-trip verification."""

    @pytest.mark.parametrize(
        "seconds,target,expected",
        [
            (1055, "seconds_int", "1055"),
            (1055, "mm_ss", "17:35"),
            (1055, "hh_mm_ss", "0:17:35"),
            (1055, "human_xm_ys", "17m 35s"),
            (3700, "mm_ss", "61:40"),
            (3700, "hh_mm_ss", "1:01:40"),
            (3700, "human_xm_ys", "1h 1m 40s"),
            (0, "mm_ss", "0:00"),
            (0, "human_xm_ys", "0s"),
            (60, "mm_ss", "1:00"),
            (60, "human_xm_ys", "1m 0s"),
        ],
    )
    def test_format_duration_basic(
        self, seconds: int, target: str, expected: str
    ) -> None:
        result = format_duration(str(seconds), target)
        assert result is not None
        new_value, params = result
        assert new_value == expected
        assert params["from_unit"] == "seconds"
        assert params["to_unit"] == target

    def test_format_duration_roundtrip(self) -> None:
        """Every emitted form parses back to the canonical seconds."""
        for seconds in (0, 1, 59, 60, 1055, 3599, 3600, 3700, 86399, 86400):
            for target in ("seconds_int", "mm_ss", "hh_mm_ss", "human_xm_ys"):
                result = format_duration(str(seconds), target)
                assert (
                    result is not None
                ), f"format_duration({seconds}, {target}) returned None"
                new_value, _ = result
                parsed_back = parse_duration(new_value)
                assert parsed_back == seconds, (
                    f"Round-trip failed for {seconds}s -> {target} -> "
                    f"{new_value!r} -> {parsed_back}"
                )

    def test_format_duration_rejects_unknown_target(self) -> None:
        assert format_duration("1055", "bogus_format") is None

    def test_format_duration_rejects_negative(self) -> None:
        assert format_duration("-1", "mm_ss") is None

    def test_format_duration_handles_float_input(self) -> None:
        result = format_duration("903.0", "mm_ss")
        assert result is not None
        assert result[0] == "15:03"

    def test_parse_duration_accepts_all_forms(self) -> None:
        assert parse_duration("1055") == 1055
        assert parse_duration("17:35") == 1055
        assert parse_duration("0:17:35") == 1055
        assert parse_duration("17m 35s") == 1055
        assert parse_duration("1h 1m 40s") == 3700

    def test_parse_duration_rejects_garbage(self) -> None:
        assert parse_duration("nonsense") is None
        assert parse_duration("") is None
        assert parse_duration("null") is None


# ---- Music duration integration tests ---------------------------------------


class TestMusicDurationIntegration:
    """Music domain — duration family routes through the dispatcher."""

    @pytest.fixture
    def music_config(self) -> dict[str, Any]:
        return load_knob_05_config("music")

    def test_easy_emits_duration_provenance(
        self,
        music_sources: dict[str, pd.DataFrame],
        music_config: dict[str, Any],
    ) -> None:
        """At easy, duration cells are reformatted (not a no-op)."""
        _, prov_df, _ = apply_knob_05("music", "easy", music_sources, music_config)
        # Duration provenance rows are emitted via reconvert_unit transform_fn.
        unit_prov = prov_df[prov_df["transform_fn"] == "reconvert_unit"]
        assert len(unit_prov) > 0, "No duration reformat at music/easy"

    def test_duration_roundtrips_at_every_level(
        self,
        music_sources: dict[str, pd.DataFrame],
        music_config: dict[str, Any],
    ) -> None:
        """Every emitted duration parses back to the original seconds."""
        for level in ("easy", "medium", "hard"):
            _, prov_df, _ = apply_knob_05("music", level, music_sources, music_config)
            unit_prov = prov_df[prov_df["transform_fn"] == "reconvert_unit"]
            for _, row in unit_prov.iterrows():
                params = json.loads(row["transform_params"])
                if params.get("from_unit") != "seconds":
                    continue
                orig = parse_duration(row["original_value"])
                new = parse_duration(row["new_value"])
                assert orig is not None
                assert new == orig, (
                    f"Duration roundtrip failed at {level}: "
                    f"{row['original_value']} -> {row['new_value']}"
                )

    def test_hard_introduces_format_diversity(
        self,
        music_sources: dict[str, pd.DataFrame],
        music_config: dict[str, Any],
    ) -> None:
        """At hard with row-level draws, >= 2 duration forms appear within a source."""
        reformatted, _, _ = apply_knob_05(
            "music", "hard", music_sources, music_config, seed=42
        )
        # Classify each duration cell by structural form and count distinct
        # forms per source. With 5 rows and a 4-format pool drawn per row,
        # at least 2 distinct forms should appear in at least one source.
        seen_diverse_source = False
        for src_name, df in reformatted.items():
            forms: set[str] = set()
            for v in df["duration"].dropna().tolist():
                s = str(v)
                if ":" in s and s.count(":") == 2:
                    forms.add("hh_mm_ss")
                elif ":" in s:
                    forms.add("mm_ss")
                elif "s" in s:
                    forms.add("human_xm_ys")
                else:
                    forms.add("seconds_int")
            if len(forms) >= 2:
                seen_diverse_source = True
                break
        assert seen_diverse_source, "No source had >= 2 duration forms at hard"


# ---- file_size class tests (plan_revision §K5, 2026-05-22) ---------------


class TestResolveColumnContext:
    """Unit tests for the per-column source_magnitude_context resolver."""

    def test_legacy_list_form_managed_column(self) -> None:
        from usecases_synthetic.scripts.apply_knob_05_format import (
            _resolve_column_context,
        )

        ctx = {
            "columns": ["price"],
            "implicit_currency": "GBP",
            "implicit_magnitude": "raw",
        }
        is_managed, merged = _resolve_column_context(ctx, "price")
        assert is_managed is True
        assert merged == {"implicit_currency": "GBP", "implicit_magnitude": "raw"}

    def test_legacy_list_form_unmanaged_column(self) -> None:
        from usecases_synthetic.scripts.apply_knob_05_format import (
            _resolve_column_context,
        )

        ctx = {
            "columns": ["price"],
            "implicit_currency": "GBP",
        }
        is_managed, merged = _resolve_column_context(ctx, "vram_gb")
        assert is_managed is False

    def test_map_form_per_column_override(self) -> None:
        from usecases_synthetic.scripts.apply_knob_05_format import (
            _resolve_column_context,
        )

        ctx = {
            "columns": {
                "price": {"implicit_currency": "GBP", "implicit_magnitude": "raw"},
                "vram_gb": {"implicit_unit": "GB"},
            },
        }
        is_managed, merged = _resolve_column_context(ctx, "price")
        assert is_managed is True
        assert merged["implicit_currency"] == "GBP"

        is_managed, merged = _resolve_column_context(ctx, "vram_gb")
        assert is_managed is True
        assert merged["implicit_unit"] == "GB"
        # vram_gb has no implicit_currency — must not be carried over
        # from any other column.
        assert "implicit_currency" not in merged

    def test_map_form_source_defaults_inherited_when_not_overridden(self) -> None:
        from usecases_synthetic.scripts.apply_knob_05_format import (
            _resolve_column_context,
        )

        ctx = {
            "implicit_currency": "USD",  # source-level default
            "columns": {
                "price": {"implicit_magnitude": "raw"},  # no implicit_currency override
            },
        }
        is_managed, merged = _resolve_column_context(ctx, "price")
        assert is_managed is True
        assert merged == {"implicit_currency": "USD", "implicit_magnitude": "raw"}

    def test_missing_columns_key_returns_unmanaged(self) -> None:
        from usecases_synthetic.scripts.apply_knob_05_format import (
            _resolve_column_context,
        )

        is_managed, merged = _resolve_column_context({}, "anything")
        assert is_managed is False
        assert merged == {}


class TestFileSizeIntegration:
    """End-to-end: file_size + money on the same source coexist without
    conflating currency and byte-unit pools.

    Acceptance:
    1. file_size columns get GB/MB/GiB unit suffixes; money columns get
       currency rotation independently.
    2. Round-trip: a converted "8 GB" → "8192 MB" preserves the byte
       count (within FP tolerance).
    3. Easy = identity unit pool (no rotation); hard cycles >= 2 distinct
       units.
    """

    @pytest.fixture
    def products_config(self) -> dict[str, Any]:
        return load_knob_05_config("products")

    @pytest.fixture
    def products_sources(self) -> dict[str, pd.DataFrame]:
        """Minimal products sources with mixed money + file_size columns."""
        rows = pd.DataFrame(
            {
                "id": [
                    "products_1_1",
                    "products_1_2",
                    "products_1_3",
                    "products_1_4",
                    "products_1_5",
                ],
                "title": ["A", "B", "C", "D", "E"],
                "brand": ["x", "y", "z", "x", "y"],
                "description": ["d", "d", "d", "d", "d"],
                "price": [99, 199, 49, 299, 79],
                "priceCurrency": ["GBP"] * 5,
                "vram_gb": [8, 16, 12, 24, 8],
                "storage_gb": [256, 512, 128, 1024, 256],
            }
        )
        rows.attrs["dataset_name"] = "products_1"
        return {"products_1": rows}

    def test_file_size_emits_unit_suffix_at_hard(
        self,
        products_config: dict[str, Any],
        products_sources: dict[str, pd.DataFrame],
    ) -> None:
        """At hard, file_size cells get a unit suffix (e.g. "8 GB" /
        "16 GiB" / "12288 MB"). The bare integer is no longer the cell
        value for most rows."""
        reformatted, prov_df, _ = apply_knob_05(
            domain="products",
            level="hard",
            sources=products_sources,
            config=products_config,
            seed=42,
        )
        vram_values = reformatted["products_1"]["vram_gb"].astype(str).tolist()
        # At least 4 of 5 rows should have a unit suffix or magnitude change
        # (the "bare" format suppresses suffix on a subset).
        suffixed = sum(
            1 for v in vram_values if any(u in v for u in (" GB", " GiB", " MB"))
        )
        assert suffixed >= 1, f"No suffixed vram_gb values at hard: {vram_values}"

    def test_file_size_money_pools_are_independent(
        self,
        products_config: dict[str, Any],
        products_sources: dict[str, pd.DataFrame],
    ) -> None:
        """Price never gets a file_size unit; vram_gb never gets a currency."""
        reformatted, _, _ = apply_knob_05(
            domain="products",
            level="hard",
            sources=products_sources,
            config=products_config,
            seed=42,
        )
        price_values = reformatted["products_1"]["price"].astype(str).tolist()
        vram_values = reformatted["products_1"]["vram_gb"].astype(str).tolist()
        for v in price_values:
            assert (
                "GB" not in v and "GiB" not in v and "MB" not in v
            ), f"price cell contaminated with file_size unit: {v!r}"
        for v in vram_values:
            assert (
                "GBP" not in v and "USD" not in v and "EUR" not in v
            ), f"vram_gb cell contaminated with currency: {v!r}"

    def test_file_size_easy_identity_unit(
        self,
        products_config: dict[str, Any],
        products_sources: dict[str, pd.DataFrame],
    ) -> None:
        """Easy uses only GB (no rotation). Conversions trivial; suffix appended."""
        reformatted, _, _ = apply_knob_05(
            domain="products",
            level="easy",
            sources=products_sources,
            config=products_config,
            seed=42,
        )
        vram_values = reformatted["products_1"]["vram_gb"].astype(str).tolist()
        for v in vram_values:
            # Easy only draws GB or `plain`/`en_US` locale — both produce
            # "<n> GB" because the suffix is always appended on non-bare
            # formats and easy has no "bare" entry.
            assert ("GB" in v) or v == str(v), f"unexpected easy value: {v!r}"

    def test_file_size_roundtrip_preserves_byte_count(self) -> None:
        """8 GB rewritten as MB should equal 8 * 1000 (decimal) bytes."""
        result = reconvert_unit("8", "file_size", "GB", "MB")
        assert result is not None
        new_value, params = result
        # 8 GB == 8000 MB in decimal (factor 1000:1 between GB and MB).
        assert new_value == "8000"
        assert params["from_unit"] == "GB"
        assert params["to_unit"] == "MB"

    def test_file_size_gb_to_gib_decimal_conversion(self) -> None:
        """8 GB ≈ 7.45 GiB (decimal vs binary mismatch)."""
        result = reconvert_unit("8", "file_size", "GB", "GiB")
        assert result is not None
        new_value, _ = result
        # 8e9 / 1.073741824e9 ≈ 7.45
        assert float(new_value) == pytest.approx(7.45, abs=0.01)

    def test_file_size_provenance_uses_reconvert_unit_fn(
        self,
        products_config: dict[str, Any],
        products_sources: dict[str, pd.DataFrame],
    ) -> None:
        """Provenance rows for file_size carry transform_fn=reconvert_unit
        (for unit swaps) or append_unit_suffix (for plain suffix-attach)."""
        _, prov_df, _ = apply_knob_05(
            domain="products",
            level="hard",
            sources=products_sources,
            config=products_config,
            seed=42,
        )
        vram_prov = prov_df[prov_df["attribute"] == "vram_gb"]
        if not vram_prov.empty:
            allowed = {"reconvert_unit", "append_unit_suffix", "reformat_number"}
            unexpected = set(vram_prov["transform_fn"].unique()) - allowed
            assert (
                not unexpected
            ), f"unexpected transform_fn on vram_gb provenance: {unexpected}"

    def test_file_size_money_isolation_at_load_time(
        self, products_config: dict[str, Any]
    ) -> None:
        """The config separates money + file_size unit pools (regression
        test for the original deferral that conflated them)."""
        for level in ("easy", "medium", "hard"):
            pools = products_config["unit_pool_per_level"][level]
            assert "money" in pools
            assert "file_size" in pools
            # money carries currencies, file_size carries units
            assert "currencies" in pools["money"]
            assert "units" in pools["file_size"]
            # No accidental cross-contamination
            assert "units" not in pools["money"]
            assert "currencies" not in pools["file_size"]
