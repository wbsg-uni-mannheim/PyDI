"""Tests for ``usecases_synthetic.lib.domain_value_norm``.

Covers the per-domain value normaliser module that feeds the Ditto A/B
retrain experiment (plan_revision_step4g_findings.md §2). Mirrors the
notebook's preprocessing exactly so the Ditto training data sees the
same value distribution the human-baseline matcher sees.
"""

from __future__ import annotations

import pytest

from usecases_synthetic.lib.domain_value_norm import (
    GAMES_PLATFORM_ALIASES,
    get_value_normalizer,
    normalize_games_platform,
    normalize_games_title,
)


class TestNormalizeGamesPlatform:
    """Mirrors regen_human_baseline._normalize_platform."""

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("PS3", "ps3"),
            ("Playstation 3", "ps3"),
            ("PS4", "ps4"),
            ("Playstation 4", "ps4"),
            ("PSX", "psx"),  # not in the map — passes through lowercased
            ("PSV", "ps vita"),
            ("Xbox One", "xbox one"),
            ("XOne", "xbox one"),
            ("X360", "xbox 360"),
            ("Microsoft Windows", "pc"),
            ("Windows", "pc"),
            ("PC", "pc"),
            ("Nintendo Switch", "switch"),
            ("Switch", "switch"),
            ("Nintendo GameCube", "gamecube"),
        ],
    )
    def test_alias_map_canonicalises(self, raw: str, expected: str) -> None:
        assert normalize_games_platform(raw) == expected

    def test_unknown_platform_lowercases_passes_through(self) -> None:
        # Caller expects "no surprises" — unknown values are still
        # canonicalised by case + whitespace, so K8/K10 transforms that
        # introduce e.g. trailing whitespace don't desync inference.
        assert normalize_games_platform("  STeAm Deck  ") == "steam deck"

    def test_none_returns_empty_string(self) -> None:
        assert normalize_games_platform(None) == ""

    def test_nan_returns_empty_string(self) -> None:
        assert normalize_games_platform(float("nan")) == ""

    def test_empty_string_returns_empty(self) -> None:
        assert normalize_games_platform("") == ""
        assert normalize_games_platform("   ") == ""

    def test_alias_map_size_matches_notebook(self) -> None:
        # Drift guard: regen_human_baseline.py:222-262 ships 39 alias
        # entries collapsing to 21 canonical platform names. If the
        # notebook gains a new platform variant, this test fails before
        # the Ditto retrain so we know to sync both modules.
        assert len(GAMES_PLATFORM_ALIASES) == 39
        assert len(set(GAMES_PLATFORM_ALIASES.values())) == 21


class TestNormalizeGamesTitle:
    """Mirrors regen_human_baseline._normalize_match_title."""

    def test_strips_video_game_parenthetical(self) -> None:
        # Regex matches the whole "(... video game ...)" group → the year
        # inside the parens goes with it. Matches notebook behaviour
        # exactly (regen_human_baseline._normalize_match_title).
        assert normalize_games_title("Doom (2016 video game)") == "doom"

    def test_strips_edition_suffixes(self) -> None:
        assert (
            normalize_games_title("Red Dead Redemption Special Edition")
            == "red dead redemption"
        )
        assert normalize_games_title("Skyrim Definitive Edition") == "skyrim"
        assert normalize_games_title("Resident Evil 4 HD") == "resident evil 4"

    def test_collapses_punctuation_and_whitespace(self) -> None:
        assert normalize_games_title("Grand-Theft_Auto:V") == "grand theft auto v"

    def test_lowercases(self) -> None:
        assert normalize_games_title("SUPER MARIO 64") == "super mario 64"

    def test_none_returns_empty(self) -> None:
        assert normalize_games_title(None) == ""

    def test_nan_returns_empty(self) -> None:
        assert normalize_games_title(float("nan")) == ""

    def test_empty_string_returns_empty(self) -> None:
        assert normalize_games_title("") == ""


class TestGetValueNormalizer:
    def test_games_returns_platform_plus_name_normalizers(self) -> None:
        result = get_value_normalizer("games")
        assert result is not None
        assert set(result.keys()) == {"platform", "name"}
        # Smoke-check the dispatcher actually wires the right functions.
        assert result["platform"]("PS3") == "ps3"
        assert result["name"]("Doom (2016 video game)") == "doom"

    def test_unknown_domain_returns_none(self) -> None:
        assert get_value_normalizer("companies") is None
        assert get_value_normalizer("music") is None
        assert get_value_normalizer("products") is None
        assert get_value_normalizer("nonexistent") is None
