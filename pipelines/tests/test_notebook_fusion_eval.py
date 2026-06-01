"""Tests for the per-domain notebook fusion-evaluator wiring."""

from __future__ import annotations

import pandas as pd
import pytest

from PyDI.fusion import (
    exact_match,
    intersection,
    numeric_tolerance_match,
    set_equality_match,
    tokenized_match,
    year_only_match,
)
from pipelines.lib.notebook_fusion_eval import (
    build_notebook_strategy,
    evaluate_with_notebook_strategy,
    get_spec,
    hardware_strict_spec_match,
)


def test_get_spec_known_domains() -> None:
    for d in ("products", "music", "games", "companies"):
        spec = get_spec(d)
        assert spec.domain == d
        assert spec.rules, f"empty rules for {d}"


def test_get_spec_unknown_domain_raises() -> None:
    with pytest.raises(ValueError, match="No notebook fusion-eval spec"):
        get_spec("movies")


def test_products_strategy_rule_count_matches_notebook() -> None:
    spec = get_spec("products")
    # Notebook cell 43 registers 14 add_evaluation_function calls
    # (2 exact + 8 numeric_tolerance + 4 hardware_strict_spec).
    # model_number is a FUSER in the notebook but has NO eval
    # function — it MUST NOT appear in the rules.
    attrs = [a for a, _, _ in spec.rules]
    assert "model_number" not in attrs
    assert len(spec.rules) == 14
    # The 4 strict-spec attrs.
    strict_attrs = {a for a, fn, _ in spec.rules if fn is hardware_strict_spec_match}
    assert strict_attrs == {"chipset_name", "bus_type", "interface_type", "memory_type"}
    # The 8 numeric_tolerance attrs all use 0.15.
    num_tols = [
        kw["tolerance"] for _, fn, kw in spec.rules if fn is numeric_tolerance_match
    ]
    assert len(num_tols) == 8
    assert all(t == 0.15 for t in num_tols)


def test_music_strategy_matches_notebook() -> None:
    spec = get_spec("music")
    assert {a for a, _, _ in spec.rules} == {
        "name",
        "artist",
        "duration",
        "release-date",
        "release-country",
        "label",
        "tracks",
    }
    # release-date uses year_only_match (NOT numeric_tolerance_match).
    for attr, fn, kw in spec.rules:
        if attr == "release-date":
            assert fn is year_only_match
            assert kw == {}
        if attr == "duration":
            assert fn is numeric_tolerance_match
            assert kw == {"tolerance": 10}


def test_games_strategy_uses_intersection_for_genres() -> None:
    spec = get_spec("games")
    rules_by_attr = {a: (fn, kw) for a, fn, kw in spec.rules}
    assert rules_by_attr["name"][0] is exact_match
    assert rules_by_attr["releaseYear"][0] is year_only_match
    assert rules_by_attr["criticScore"] == (numeric_tolerance_match, {"tolerance": 2})
    assert rules_by_attr["userScore"] == (numeric_tolerance_match, {"tolerance": 0.2})
    assert rules_by_attr["genres"][0] is intersection


def test_companies_strategy_double_registers_assets() -> None:
    """The notebook registers `assets` twice: tokenized_match first,
    then numeric_tolerance_match(tolerance=0.1). The DataFusionStrategy
    stores eval functions by attribute, so the LATER registration wins.
    Mirror that.
    """
    spec = get_spec("companies")
    assets_calls = [(fn, kw) for a, fn, kw in spec.rules if a == "assets"]
    assert assets_calls == [
        (tokenized_match, {}),
        (numeric_tolerance_match, {"tolerance": 0.1}),
    ]


def test_companies_strategy_uses_set_equality_for_founders() -> None:
    spec = get_spec("companies")
    rules_by_attr = {a: (fn, kw) for a, fn, kw in spec.rules}
    assert rules_by_attr["founders"][0] is set_equality_match


def test_build_strategy_registers_all_rules() -> None:
    """build_notebook_strategy applies every rule to the DataFusionStrategy."""
    for d in ("products", "music", "games", "companies"):
        strategy = build_notebook_strategy(d)
        spec = get_spec(d)
        # Every unique attribute in the rules should resolve to a
        # registered eval function (later registrations win, e.g.
        # companies.assets).
        rule_attrs = {a for a, _, _ in spec.rules}
        for attr in rule_attrs:
            assert (
                strategy.get_evaluation_function(attr) is not None
            ), f"{d}: {attr} not registered"


def test_hardware_strict_spec_match_pcie_x8_vs_x16() -> None:
    """The custom matcher must reject PCIE x8 vs PCIE x16 (different digits)."""
    assert hardware_strict_spec_match("PCIE x16", "PCIE x16") is True
    assert hardware_strict_spec_match("PCIE x8", "PCIE x16") is False
    # Case-insensitive + punctuation-insensitive equality.
    assert hardware_strict_spec_match("pcie-x16", "PCIE x16") is True
    # NaN handling.
    assert hardware_strict_spec_match(None, "PCIE x16") is False


def test_evaluate_with_notebook_strategy_runs_on_synthetic_fused() -> None:
    """End-to-end sanity: build a tiny fused frame + gold and confirm
    the evaluator returns per-attribute accuracies."""
    fused = pd.DataFrame(
        {
            "metacritic_id": ["m1", "m2", "m3"],
            "name": ["Halo", "Mario", "Tetris"],
            "platform": ["Xbox", "Switch", "GB"],
            "developer": ["Bungie", "Nintendo", "AB"],
            "releaseYear": [2001, 1985, 1984],
            "ESRB": ["M", "E", "E"],
            "criticScore": [97, 95, 90],
            "userScore": [8.5, 9.0, 9.5],
            "genres": [["FPS"], ["Platformer"], ["Puzzle"]],
        }
    )
    gold = pd.DataFrame(
        {
            "id": ["m1", "m2", "m3"],
            "name": ["Halo", "Mario", "Tetris"],
            "platform": ["Xbox", "Switch", "GB"],
            "developer": ["Bungie", "Nintendo", "AB"],
            "releaseYear": [2001, 1985, 1984],
            "ESRB": ["M", "E", "E"],
            "criticScore": [98, 95, 90],  # within tol=2
            "userScore": [8.4, 9.0, 9.5],  # within tol=0.2
            "genres": [["FPS"], ["Platformer"], ["Puzzle"]],
        }
    )
    scores = evaluate_with_notebook_strategy(fused, domain="games", gold_df=gold)
    # Every per-attribute accuracy should be 1.0 (within tolerance).
    for attr in [
        "name",
        "platform",
        "developer",
        "releaseYear",
        "ESRB",
        "criticScore",
        "userScore",
    ]:
        key = f"{attr}_accuracy"
        assert scores.get(key) == pytest.approx(1.0), f"{key}: {scores.get(key)}"
