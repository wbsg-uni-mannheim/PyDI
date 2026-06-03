"""Tests for the schema-constraint Norm scorer + its C12 runner wiring."""

from __future__ import annotations

from pathlib import Path

import pytest

from pipelines.lib.schema_constraint_scorer import (
    AttributeConstraints,
    SchemaConstraintScores,
    parse_target_schema,
    value_passes,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_products_schema() -> Path:
    """Schema file was renamed ``products_target_schema.json`` ->
    ``target_schema.json`` for naming consistency (commit f9bbe06c).
    Honor either filename so the test survives both checkouts."""
    base = REPO_ROOT / "usecases" / "products" / "input" / "schemamatching"
    for fname in ("target_schema.json", "products_target_schema.json"):
        p = base / fname
        if p.exists():
            return p
    raise FileNotFoundError(f"No target schema under {base}")


PRODUCTS_SCHEMA = _resolve_products_schema()


@pytest.fixture(scope="module")
def products_constraints() -> dict[str, AttributeConstraints]:
    return parse_target_schema(PRODUCTS_SCHEMA)


def test_parse_covers_every_property(
    products_constraints: dict[str, AttributeConstraints],
) -> None:
    # 26 attributes in the products target schema per the 2026 refresh.
    assert len(products_constraints) >= 20
    # A few markers that must be present.
    for k in (
        "price",
        "priceCurrency",
        "product_type",
        "vram_gb",
        "storage_gb",
        "bus_type",
        "memory_type",
    ):
        assert k in products_constraints, f"{k} missing from parsed constraints"


def test_price_range(products_constraints) -> None:
    c = products_constraints["price"]
    assert c.minimum == 0.0
    assert value_passes(10, c, {}) is True
    assert value_passes(-1, c, {}) is False
    assert value_passes(None, c, {}) is None  # missing -> abstain


def test_iso_4217_currency(products_constraints) -> None:
    c = products_constraints["priceCurrency"]
    assert value_passes("USD", c, {}) is True
    assert value_passes("EUR", c, {}) is True
    assert value_passes("ZZZ", c, {}) is False  # passes regex, not ISO
    assert value_passes("usd", c, {}) is False  # case-sensitive on pattern


def test_product_type_enum(products_constraints) -> None:
    c = products_constraints["product_type"]
    for v in ("GPU", "SSD", "HDD", "USB_STICK"):
        assert value_passes(v, c, {}) is True
    assert value_passes("Foo", c, {}) is False


def test_field_applicability_vram_gb(products_constraints) -> None:
    c = products_constraints["vram_gb"]
    # Applicable + in range.
    assert value_passes(16, c, {"product_type": "GPU"}) is True
    # Applicable but out of range.
    assert value_passes(200, c, {"product_type": "GPU"}) is False
    # Inapplicable rowtype + value present -> violation.
    assert value_passes(16, c, {"product_type": "SSD"}) is False
    # Inapplicable rowtype + missing -> correctly absent -> abstain.
    assert value_passes(None, c, {"product_type": "SSD"}) is None


def test_open_taxonomy_substring(products_constraints) -> None:
    c = products_constraints["bus_type"]
    # exhaustive=False -> any value containing a family substring passes.
    assert value_passes("PCIe x16", c, {}) is True
    assert value_passes("SATA III", c, {}) is True
    assert value_passes("random gibberish", c, {}) is False


def test_open_taxonomy_with_applies_to(products_constraints) -> None:
    # memory_type applies only to GPU / SSD and uses open_taxonomy.
    c = products_constraints["memory_type"]
    assert value_passes("GDDR6", c, {"product_type": "GPU"}) is True
    assert value_passes("GDDR6", c, {"product_type": "HDD"}) is False  # inapplicable
    assert value_passes(None, c, {"product_type": "HDD"}) is None


def test_scores_macro_metrics_shape() -> None:
    """Aggregator must expose macro_f1/macro_precision/macro_recall so
    the C12 runner can swap it in without changing aggregation code."""
    s = SchemaConstraintScores(member="dummy")
    c_price = AttributeConstraints(
        name="price", json_type="number", minimum=0.0, has_any_constraint=True
    )
    # Two correct, one wrong, one abstain.
    s.record("price", 10, c_price, {})
    s.record("price", 5, c_price, {})
    s.record("price", -3, c_price, {})
    s.record("price", None, c_price, {})
    m = s.macro_metrics()
    assert set(m.keys()) == {
        "macro_f1",
        "macro_precision",
        "macro_recall",
        "num_attributes_scored",
    }
    assert m["num_attributes_scored"] == 1
    # 2 correct, 1 wrong, 1 abstain -> precision = 2/3, recall = 2/4 = 0.5
    assert m["macro_precision"] == pytest.approx(2 / 3)
    assert m["macro_recall"] == pytest.approx(0.5)


def test_per_attribute_counts_expose_total_for_predictions_frame() -> None:
    """The C12 runner's _member_predictions_frame reads
    .correct/.wrong/.abstained/.total/.precision/.recall/.f1."""
    s = SchemaConstraintScores(member="dummy")
    c = AttributeConstraints(
        name="price", json_type="number", minimum=0.0, has_any_constraint=True
    )
    s.record("price", 1, c, {})
    s.record("price", -1, c, {})
    s.record("price", None, c, {})
    counts = s.by_attribute["price"]
    assert counts.correct == 1
    assert counts.wrong == 1
    assert counts.abstained == 1
    assert counts.total == 3
    assert isinstance(counts.precision, float)
    assert isinstance(counts.recall, float)
    assert isinstance(counts.f1, float)


def test_c12_runner_accepts_scoring_surface_kwarg() -> None:
    """The NormCommitteeRunner dispatch must forward scoring_surface
    to the C12 runner."""
    from usecases_synthetic.lib.committee_norm import NormCommitteeRunner

    yaml_path = (
        REPO_ROOT
        / "usecases_synthetic"
        / "config"
        / "committees"
        / "normalization_committee_products.yaml"
    )
    runner = NormCommitteeRunner(yaml_path, scoring_surface="schema_constraints")
    assert runner._scoring_surface == "schema_constraints"
    runner2 = NormCommitteeRunner(yaml_path)
    assert runner2._scoring_surface == "xml_targets"


def test_c12_runner_rejects_unknown_scoring_surface() -> None:
    from usecases_synthetic.lib.committee_norm import NormCommitteeRunner

    yaml_path = (
        REPO_ROOT
        / "usecases_synthetic"
        / "config"
        / "committees"
        / "normalization_committee_products.yaml"
    )
    with pytest.raises(ValueError, match="Unknown scoring_surface"):
        NormCommitteeRunner(yaml_path, scoring_surface="bogus")
