"""Tests for the SM committee runner.

Exercises ``SMCommitteeRunner`` instantiation from a fixture roster,
scoring against a fixture gold mapping, and the K8 signal direction
(label-based matcher degrades when headers are renamed).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from usecases_synthetic.lib.committee import CommitteeResult, MemberResult
from usecases_synthetic.lib.committee_sm import (
    SMCommitteeRunner,
    score_sm_mapping,
    score_sm_per_attribute,
)
from usecases_synthetic.lib.variant_loader import VariantBundle


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "sm"


def _write_fixture_roster(
    tmp_path: Path,
    *,
    include_duplicate: bool = False,
) -> Path:
    """Write a minimal SM roster YAML for testing.

    Includes label_based (jaccard) and instance_based (tfidf/cosine).
    Optionally includes duplicate_based.
    """
    members: list[dict[str, Any]] = [
        {
            "name": "label_jaccard",
            "module": "PyDI.schemamatching.label_based",
            "class": "LabelBasedSchemaMatcher",
            "signal_type": "label",
            "enabled_by_default": True,
            "params": {"similarity_function": "jaccard", "tokenize": True},
            "match_kwargs": {"threshold": 0.3},
        },
        {
            "name": "instance_tfidf",
            "module": "PyDI.schemamatching.instance_based",
            "class": "InstanceBasedSchemaMatcher",
            "signal_type": "instance",
            "enabled_by_default": True,
            "params": {
                "vector_creation_method": "tfidf",
                "similarity_function": "cosine",
                "max_sample_size": 100,
            },
            "match_kwargs": {"threshold": 0.05},
        },
    ]
    if include_duplicate:
        members.append(
            {
                "name": "duplicate_majority",
                "module": "PyDI.schemamatching.duplicate_based",
                "class": "DuplicateBasedSchemaMatcher",
                "signal_type": "duplicate",
                "enabled_by_default": True,
                "params": {
                    "vote_aggregation": "majority",
                    "value_comparison": "exact",
                },
                "match_kwargs": {"threshold": 0.1},
            }
        )

    roster = {
        "seed": 42,
        "members": members,
        "required_axes": {"signal_type": ["label", "instance"]},
    }

    path = tmp_path / "sm_committee.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(roster, f)
    return path


def _make_source_identical(
    n: int = 15,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any], pd.DataFrame]:
    """Create 2 sources with IDENTICAL column names to the target schema.

    Returns (sources, target_schema, gold_mapping).
    """
    rng = np.random.default_rng(42)
    target_schema: dict[str, Any] = {
        "title": "Company",
        "properties": {
            "id": {"type": "string"},
            "name": {"type": "string"},
            "country": {"type": "string"},
            "revenue": {"type": "integer"},
        },
    }

    src_a = pd.DataFrame(
        {
            "id": [f"a_{i}" for i in range(n)],
            "name": [f"Company_{i}" for i in range(n)],
            "country": rng.choice(["US", "DE", "JP"], size=n).tolist(),
            "revenue": (rng.random(n) * 1e9).astype(int).tolist(),
        }
    )
    src_a.attrs["dataset_name"] = "source_a"

    src_b = pd.DataFrame(
        {
            "id": [f"b_{i}" for i in range(n)],
            "name": [f"Firm_{i}" for i in range(n)],
            "country": rng.choice(["US", "DE", "JP"], size=n).tolist(),
            "revenue": (rng.random(n) * 1e9).astype(int).tolist(),
        }
    )
    src_b.attrs["dataset_name"] = "source_b"

    gold = pd.DataFrame(
        [
            ("source_a", "id", "company", "id", 1.0),
            ("source_a", "name", "company", "name", 1.0),
            ("source_a", "country", "company", "country", 1.0),
            ("source_a", "revenue", "company", "revenue", 1.0),
            ("source_b", "id", "company", "id", 1.0),
            ("source_b", "name", "company", "name", 1.0),
            ("source_b", "country", "company", "country", 1.0),
            ("source_b", "revenue", "company", "revenue", 1.0),
        ],
        columns=[
            "source_dataset",
            "source_column",
            "target_dataset",
            "target_column",
            "score",
        ],
    )

    sources = {"source_a": src_a, "source_b": src_b}
    return sources, target_schema, gold


def _make_source_renamed(
    n: int = 15,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any], pd.DataFrame]:
    """Create 2 sources where one has columns renamed to ``Attribute_N``.

    This simulates K8 hard-level renaming — label-based matchers should
    fail on the renamed source.

    Returns (sources, target_schema, gold_mapping).
    """
    rng = np.random.default_rng(42)
    target_schema: dict[str, Any] = {
        "title": "Company",
        "properties": {
            "id": {"type": "string"},
            "name": {"type": "string"},
            "country": {"type": "string"},
            "revenue": {"type": "integer"},
        },
    }

    # Source A: identical column names — easy.
    src_a = pd.DataFrame(
        {
            "id": [f"a_{i}" for i in range(n)],
            "name": [f"Company_{i}" for i in range(n)],
            "country": rng.choice(["US", "DE", "JP"], size=n).tolist(),
            "revenue": (rng.random(n) * 1e9).astype(int).tolist(),
        }
    )
    src_a.attrs["dataset_name"] = "source_a"

    # Source B: anonymised column names — hard for label-based.
    src_b = pd.DataFrame(
        {
            "Attribute_1": [f"b_{i}" for i in range(n)],
            "Attribute_2": [f"Firm_{i}" for i in range(n)],
            "Attribute_3": rng.choice(["US", "DE", "JP"], size=n).tolist(),
            "Attribute_4": (rng.random(n) * 1e9).astype(int).tolist(),
        }
    )
    src_b.attrs["dataset_name"] = "source_b"

    gold = pd.DataFrame(
        [
            ("source_a", "id", "company", "id", 1.0),
            ("source_a", "name", "company", "name", 1.0),
            ("source_a", "country", "company", "country", 1.0),
            ("source_a", "revenue", "company", "revenue", 1.0),
            ("source_b", "Attribute_1", "company", "id", 1.0),
            ("source_b", "Attribute_2", "company", "name", 1.0),
            ("source_b", "Attribute_3", "company", "country", 1.0),
            ("source_b", "Attribute_4", "company", "revenue", 1.0),
        ],
        columns=[
            "source_dataset",
            "source_column",
            "target_dataset",
            "target_column",
            "score",
        ],
    )

    sources = {"source_a": src_a, "source_b": src_b}
    return sources, target_schema, gold


def _make_bundle(
    sources: dict[str, pd.DataFrame],
    target_schema: dict[str, Any],
    gold: pd.DataFrame,
    *,
    level: str = "baseline",
) -> VariantBundle:
    """Build a minimal VariantBundle for SM testing."""
    return VariantBundle(
        domain="companies",
        level=level,
        sources=sources,
        target_schema=target_schema,
        sm_mapping=gold,
        em_gold={},
        em_splits={},
        fusion_gold=pd.DataFrame(),
        fusion_validation=None,
        pooled_positives=None,
        variant_root=Path("/tmp/sm_test"),
    )


# ---------------------------------------------------------------------------
# Scoring unit tests
# ---------------------------------------------------------------------------


class TestScoreSMMapping:
    """Tests for ``score_sm_mapping``."""

    def test_perfect_score(self) -> None:
        gold = pd.DataFrame(
            [
                ("src", "a", "tgt", "x", 1.0),
                ("src", "b", "tgt", "y", 1.0),
            ],
            columns=[
                "source_dataset",
                "source_column",
                "target_dataset",
                "target_column",
                "score",
            ],
        )
        metrics = score_sm_mapping(gold, gold)
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0

    def test_partial_score(self) -> None:
        gold = pd.DataFrame(
            [
                ("src", "a", "tgt", "x", 1.0),
                ("src", "b", "tgt", "y", 1.0),
            ],
            columns=[
                "source_dataset",
                "source_column",
                "target_dataset",
                "target_column",
                "score",
            ],
        )
        # Predict only one correct + one wrong.
        pred = pd.DataFrame(
            [
                ("src", "a", "tgt", "x", 0.9),
                ("src", "c", "tgt", "z", 0.8),
            ],
            columns=[
                "source_dataset",
                "source_column",
                "target_dataset",
                "target_column",
                "score",
            ],
        )
        metrics = score_sm_mapping(pred, gold)
        assert metrics["precision"] == 0.5  # 1 correct / 2 predicted
        assert metrics["recall"] == 0.5  # 1 correct / 2 gold
        assert 0.49 < metrics["f1"] < 0.51

    def test_empty_prediction(self) -> None:
        gold = pd.DataFrame(
            [("src", "a", "tgt", "x", 1.0)],
            columns=[
                "source_dataset",
                "source_column",
                "target_dataset",
                "target_column",
                "score",
            ],
        )
        pred = pd.DataFrame(
            columns=[
                "source_dataset",
                "source_column",
                "target_dataset",
                "target_column",
                "score",
            ]
        )
        metrics = score_sm_mapping(pred, gold)
        assert metrics["f1"] == 0.0
        assert metrics["fn"] == 1.0


class TestScoreSMPerAttribute:
    """Tests for ``score_sm_per_attribute``."""

    def test_all_correct(self) -> None:
        gold = pd.DataFrame(
            [
                ("src", "a", "tgt", "x", 1.0),
                ("src", "b", "tgt", "y", 1.0),
            ],
            columns=[
                "source_dataset",
                "source_column",
                "target_dataset",
                "target_column",
                "score",
            ],
        )
        result = score_sm_per_attribute(gold, gold)
        assert result["src.a"]["correct"] == 1.0
        assert result["src.b"]["correct"] == 1.0

    def test_one_wrong(self) -> None:
        gold = pd.DataFrame(
            [
                ("src", "a", "tgt", "x", 1.0),
                ("src", "b", "tgt", "y", 1.0),
            ],
            columns=[
                "source_dataset",
                "source_column",
                "target_dataset",
                "target_column",
                "score",
            ],
        )
        pred = pd.DataFrame(
            [
                ("src", "a", "tgt", "x", 1.0),
                ("src", "b", "tgt", "z", 1.0),  # wrong target
            ],
            columns=[
                "source_dataset",
                "source_column",
                "target_dataset",
                "target_column",
                "score",
            ],
        )
        result = score_sm_per_attribute(pred, gold)
        assert result["src.a"]["correct"] == 1.0
        assert result["src.b"]["correct"] == 0.0


# ---------------------------------------------------------------------------
# SMCommitteeRunner tests
# ---------------------------------------------------------------------------


class TestSMCommitteeRunner:
    """Tests for the full SM committee runner."""

    def test_instantiation(self, tmp_path: Path) -> None:
        """Runner loads a fixture roster and instantiates matchers."""
        roster_path = _write_fixture_roster(tmp_path)
        runner = SMCommitteeRunner(roster_path)

        assert runner.roster_names == ["label_jaccard", "instance_tfidf"]
        assert len(runner.roster) == 2

    def test_run_identical_headers(self, tmp_path: Path) -> None:
        """On identical headers the label-based matcher should score ~1.0."""
        roster_path = _write_fixture_roster(tmp_path)
        runner = SMCommitteeRunner(roster_path)

        sources, schema, gold = _make_source_identical()
        bundle = _make_bundle(sources, schema, gold)

        result = runner.run(bundle)

        assert isinstance(result, CommitteeResult)
        assert result.stage == "sm"
        assert result.domain == "companies"
        assert result.level == "baseline"
        assert set(result.per_member) == {"label_jaccard", "instance_tfidf"}
        assert result.roster == ["label_jaccard", "instance_tfidf"]

        # Label-based matcher on identical headers should get high F1.
        label_f1 = result.per_member["label_jaccard"].metrics["f1"]
        assert label_f1 >= 0.8, (
            f"Label-based matcher on identical headers should have "
            f"high F1, got {label_f1}"
        )

        # Aggregated metrics populated.
        assert "macro_f1" in result.aggregated
        assert "min_f1" in result.aggregated
        assert "max_f1" in result.aggregated

    def test_run_renamed_headers_label_degrades(
        self, tmp_path: Path
    ) -> None:
        """K8 signal direction: label-based F1 drops on anonymised headers."""
        roster_path = _write_fixture_roster(tmp_path)
        runner = SMCommitteeRunner(roster_path)

        # Baseline: identical headers.
        src_id, schema_id, gold_id = _make_source_identical()
        bundle_id = _make_bundle(src_id, schema_id, gold_id)
        result_id = runner.run(bundle_id)
        label_f1_baseline = result_id.per_member["label_jaccard"].metrics["f1"]

        # Variant: source_b has anonymised headers.
        src_rn, schema_rn, gold_rn = _make_source_renamed()
        bundle_rn = _make_bundle(src_rn, schema_rn, gold_rn, level="hard")
        result_rn = runner.run(bundle_rn)
        label_f1_variant = result_rn.per_member["label_jaccard"].metrics["f1"]

        # The label-based matcher should do worse on renamed headers.
        assert label_f1_variant < label_f1_baseline, (
            f"Label-based matcher should degrade on renamed headers: "
            f"baseline={label_f1_baseline}, variant={label_f1_variant}"
        )

    def test_per_attribute_populated(self, tmp_path: Path) -> None:
        """per_attribute should have entries for each gold column."""
        roster_path = _write_fixture_roster(tmp_path)
        runner = SMCommitteeRunner(roster_path)

        sources, schema, gold = _make_source_identical()
        bundle = _make_bundle(sources, schema, gold)
        result = runner.run(bundle)

        # Gold has 8 entries (4 per source × 2 sources).
        assert len(result.per_attribute) == 8
        for attr_key, attr_vals in result.per_attribute.items():
            assert "any_correct" in attr_vals, (
                f"Missing 'any_correct' for {attr_key}"
            )

    def test_per_partition_populated(self, tmp_path: Path) -> None:
        """per_partition should have one entry per source."""
        roster_path = _write_fixture_roster(tmp_path)
        runner = SMCommitteeRunner(roster_path)

        sources, schema, gold = _make_source_identical()
        bundle = _make_bundle(sources, schema, gold)
        result = runner.run(bundle)

        assert "source_a" in result.per_partition
        assert "source_b" in result.per_partition
        for source, metrics in result.per_partition.items():
            assert "macro_f1" in metrics
            assert "n_columns" in metrics

    def test_result_as_dict_serializable(self, tmp_path: Path) -> None:
        """CommitteeResult.as_dict() produces a JSON-serialisable dict."""
        import json

        roster_path = _write_fixture_roster(tmp_path)
        runner = SMCommitteeRunner(roster_path)

        sources, schema, gold = _make_source_identical()
        bundle = _make_bundle(sources, schema, gold)
        result = runner.run(bundle)

        payload = result.as_dict()
        # Should not raise.
        json.dumps(payload)

    def test_no_gold_raises(self, tmp_path: Path) -> None:
        """Runner raises ValueError when no gold mapping is available."""
        roster_path = _write_fixture_roster(tmp_path)
        runner = SMCommitteeRunner(roster_path)

        bundle = _make_bundle(
            sources={},
            target_schema={"properties": {}},
            gold=None,  # type: ignore[arg-type]
        )
        bundle.sm_mapping = None

        with pytest.raises(ValueError, match="No SM gold mapping"):
            runner.run(bundle)
