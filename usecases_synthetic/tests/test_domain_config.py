"""Tests for domain config loading and monotonicity validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from usecases_synthetic.lib.domain_config import (
    DomainConfig,
    load_domain_config,
    validate_knob_config_monotonicity,
    validate_monotonicity,
)


class TestLoadDomainConfig:
    """Tests for YAML domain config loading."""

    def test_load_companies(self) -> None:
        config = load_domain_config("companies")
        assert config.domain == "companies"
        assert len(config.sources) == 3
        assert config.source_names == ["dbpedia", "forbes", "fullcontact"]
        assert config.master_seed == 42

    def test_attribute_classes(self) -> None:
        config = load_domain_config("companies")
        assert config.attribute_classes["name"] == "primary"
        assert config.attribute_classes["country"] == "key"
        assert config.attribute_classes["revenue"] == "secondary"

    def test_source_pairs(self) -> None:
        config = load_domain_config("companies")
        assert ("forbes", "dbpedia") in config.source_pairs

    def test_paths(self) -> None:
        config = load_domain_config("companies")
        assert config.data_dir().name == "data"
        assert config.em_dir().name == "entitymatching"
        assert config.fusion_dir().name == "fusion"
        assert config.pool_dir().name == "companies"

    def test_unknown_domain_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown domain"):
            load_domain_config("nonexistent")


class TestMonotonicity:
    """Tests for monotonicity validation."""

    def test_valid_increasing(self) -> None:
        validate_monotonicity(
            {"easy": 0.0, "medium": 0.05, "hard": 0.15}, direction="increasing"
        )

    def test_valid_decreasing(self) -> None:
        validate_monotonicity(
            {"easy": 0.15, "medium": 0.05, "hard": 0.0}, direction="decreasing"
        )

    def test_equal_values_valid(self) -> None:
        validate_monotonicity(
            {"easy": 0.1, "medium": 0.1, "hard": 0.1}, direction="increasing"
        )

    def test_non_monotone_increasing_raises(self) -> None:
        with pytest.raises(ValueError, match="Non-monotone"):
            validate_monotonicity(
                {"easy": 0.1, "medium": 0.05, "hard": 0.15},
                direction="increasing",
            )

    def test_non_monotone_decreasing_raises(self) -> None:
        with pytest.raises(ValueError, match="Non-monotone"):
            validate_monotonicity(
                {"easy": 0.0, "medium": 0.05, "hard": 0.15},
                direction="decreasing",
            )

    def test_validate_knob_config_monotonicity(self) -> None:
        config = {
            "noise_rate_primary": {"easy": 0.0, "medium": 0.0, "hard": 0.02},
            "noise_rate_key": {"easy": 0.0, "medium": 0.02, "hard": 0.08},
            "noise_rate_secondary": {"easy": 0.01, "medium": 0.05, "hard": 0.15},
        }
        errors = validate_knob_config_monotonicity(
            config,
            ["noise_rate_primary", "noise_rate_key", "noise_rate_secondary"],
        )
        assert errors == []

    def test_validate_knob_config_catches_violation(self) -> None:
        config = {
            "noise_rate_key": {"easy": 0.1, "medium": 0.05, "hard": 0.15},
        }
        errors = validate_knob_config_monotonicity(config, ["noise_rate_key"])
        assert len(errors) == 1
        assert "noise_rate_key" in errors[0]
