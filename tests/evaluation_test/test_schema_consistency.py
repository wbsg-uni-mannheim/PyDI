"""Unit tests for schema-driven consistency evaluation."""

from __future__ import annotations

import pandas as pd
import pytest

from PyDI.evaluation.schema_consistency import (
    evaluate_schema_consistency,
    write_schema_consistency_report,
)


def test_native_constraints_use_filled_cell_weighted_score():
    schema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "minLength": 2,
                "maxLength": 5,
                "pattern": "^[A-Z]+$",
            },
            "assets": {"type": "integer", "minimum": 0, "maximum": 10},
            "empty": {"type": "string", "minLength": 1},
        },
    }
    df = pd.DataFrame(
        {
            "name": ["AB", "abc", ""],
            "assets": [5, 11, "9"],
            "empty": [None, "", None],
        }
    )

    result = evaluate_schema_consistency(df, schema)

    assert result["n_evaluated"] == 5
    assert result["n_consistent"] == 3
    assert result["consistency_score"] == pytest.approx(3 / 5)
    assert result["per_column"]["name"]["consistency_score"] == pytest.approx(1 / 2)
    assert result["per_column"]["assets"]["consistency_score"] == pytest.approx(2 / 3)
    assert result["per_column"]["empty"]["consistency_score"] is None
    assert result["n_empty_columns_excluded"] == 1


def test_exhaustive_taxonomy_accepts_alias_columns(tmp_path):
    taxonomy = tmp_path / "countries.csv"
    taxonomy.write_text(
        "Country Name,Country Short Name\nGermany,DE\nUnited States,US\n",
        encoding="utf-8",
    )
    schema = {
        "type": "object",
        "properties": {
            "country": {
                "type": "string",
                "x-pydi-taxonomy": "countries.csv",
                "x-pydi-taxonomy-column": "Country Name",
                "x-pydi-taxonomy-alias-columns": ["Country Short Name"],
                "x-pydi-taxonomy-exhaustive": True,
            }
        },
    }
    df = pd.DataFrame({"country": ["Germany", "DE", "France", ""]})

    result = evaluate_schema_consistency(
        df,
        schema,
        taxonomy_base_path=tmp_path,
    )

    assert result["per_column"]["country"]["n_evaluated"] == 3
    assert result["per_column"]["country"]["n_consistent"] == 2
    assert result["per_column"]["country"]["failure_counts"] == {"taxonomy": 1}


def test_non_exhaustive_taxonomy_does_not_fail_outside_values(tmp_path):
    taxonomy = tmp_path / "genres.csv"
    taxonomy.write_text("Genre\nAction\nPuzzle\n", encoding="utf-8")
    schema = {
        "type": "object",
        "properties": {
            "genre": {
                "type": "string",
                "x-pydi-taxonomy": "genres.csv",
                "x-pydi-taxonomy-column": "Genre",
                "x-pydi-taxonomy-exhaustive": False,
            }
        },
    }
    df = pd.DataFrame({"genre": ["Action", "New Genre"]})

    result = evaluate_schema_consistency(df, schema, taxonomy_base_path=tmp_path)

    assert result["per_column"]["genre"]["consistency_score"] == pytest.approx(1.0)
    assert result["per_column"]["genre"]["diagnostics"]["taxonomy"] == {
        "exhaustive": False,
        "membership_enforced": False,
        "n_values_checked": 2,
        "n_known_values": 1,
        "n_unknown_values": 1,
        "known_value_rate": pytest.approx(0.5),
    }


def test_identifier_columns_are_excluded_from_dataset_score():
    schema = {
        "type": "object",
        "properties": {
            "id": {"type": "string", "pattern": "^[A-Z]+$"},
            "name": {"type": "string", "minLength": 1},
        },
    }
    df = pd.DataFrame({"id": [123, 456], "name": ["SAP", "IBM"]})

    result = evaluate_schema_consistency(df, schema)

    assert result["consistency_score"] == pytest.approx(1.0)
    assert result["n_evaluated"] == 2
    assert result["per_column"]["id"]["skipped"] is True
    assert result["per_column"]["id"]["skip_reason"] == "identifier_column"


def test_custom_rules_for_dates_pages_currency_applicability_and_zero_missing():
    schema = {
        "type": "object",
        "properties": {
            "product_type": {"type": "string", "enum": ["GPU", "SSD"]},
            "vram_gb": {
                "type": "number",
                "minimum": 1,
                "maximum": 128,
                "x-pydi-consistency": {
                    "rule": "field_applicability",
                    "appliesToProductTypes": ["GPU"],
                },
            },
            "currency": {
                "type": "string",
                "pattern": "^[A-Z]{3}$",
                "x-pydi-consistency": {"rule": "currency_code"},
            },
            "duration": {
                "type": "number",
                "minimum": 1,
                "maximum": 100,
                "x-pydi-consistency": {"rule": "zero_as_missing"},
            },
            "release_date": {
                "type": "string",
                "format": "date",
                "pattern": "^[0-9]{4}-[0-9]{2}-[0-9]{2}$",
                "x-pydi-consistency": {
                    "rule": "date_range",
                    "minimumDate": "2020-01-01",
                    "maximumDate": "2024-12-31",
                },
            },
            "first_page": {
                "type": "string",
                "pattern": "^[A-Za-z0-9][A-Za-z0-9._-]*$",
                "x-pydi-consistency": {
                    "rule": "page_locator",
                    "numericMinimum": 1,
                    "numericMaximum": 1000,
                },
            },
            "last_page": {
                "type": "string",
                "pattern": "^[A-Za-z0-9][A-Za-z0-9._-]*$",
                "x-pydi-consistency": {
                    "rule": "page_locator",
                    "numericMinimum": 1,
                    "numericMaximum": 1000,
                    "numericComparison": {"operator": ">=", "field": "first_page"},
                },
            },
        },
    }
    df = pd.DataFrame(
        {
            "product_type": ["GPU", "SSD"],
            "vram_gb": [16, 16],
            "currency": ["USD", "ZZZ"],
            "duration": [0, 5],
            "release_date": ["2024-01-01", "2025-01-01"],
            "first_page": ["10", "20"],
            "last_page": ["12", "10"],
        }
    )

    result = evaluate_schema_consistency(df, schema)

    assert result["per_column"]["duration"]["n_evaluated"] == 1
    assert result["per_column"]["vram_gb"]["failure_counts"] == {
        "field_applicability": 1
    }
    assert result["per_column"]["currency"]["failure_counts"] == {"currency_code": 1}
    assert result["per_column"]["release_date"]["failure_counts"] == {"date_range": 1}
    assert result["per_column"]["last_page"]["failure_counts"] == {
        "page_locator.numeric_comparison": 1
    }


def test_open_taxonomy_is_diagnostic_when_not_exhaustive():
    schema = {
        "type": "object",
        "properties": {
            "bus_type": {
                "type": "string",
                "x-pydi-consistency": {
                    "rule": "open_taxonomy",
                    "exhaustive": False,
                    "expectedFamilies": ["PCIe", "USB"],
                },
            }
        },
    }
    df = pd.DataFrame({"bus_type": ["PCIe x16", "Custom connector"]})

    result = evaluate_schema_consistency(df, schema)

    assert result["per_column"]["bus_type"]["consistency_score"] == pytest.approx(1.0)
    assert result["per_column"]["bus_type"]["diagnostics"]["open_taxonomy"] == {
        "exhaustive": False,
        "membership_enforced": False,
        "n_values_checked": 2,
        "n_expected_family_matches": 1,
        "n_outside_expected_families": 1,
        "expected_family_match_rate": pytest.approx(0.5),
    }


def test_array_items_and_delimited_terms_are_checked():
    schema = {
        "type": "object",
        "properties": {
            "genres": {
                "type": "array",
                "minItems": 1,
                "maxItems": 2,
                "items": {"type": "string", "minLength": 2},
            },
            "keywords": {
                "type": "string",
                "x-pydi-consistency": {
                    "rule": "delimited_open_text",
                    "separator": ",",
                    "maxTerms": 2,
                },
            },
        },
    }
    df = pd.DataFrame(
        {
            "genres": [["Action"], ["A"], ["Action", "Puzzle", "RPG"]],
            "keywords": ["alpha, beta", "alpha, beta, gamma", ""],
        }
    )

    result = evaluate_schema_consistency(df, schema)

    assert result["per_column"]["genres"]["n_consistent"] == 1
    assert result["per_column"]["genres"]["failure_counts"] == {
        "items.minLength": 1,
        "maxItems": 1,
    }
    assert result["per_column"]["keywords"]["failure_counts"] == {
        "delimited_open_text.maxTerms": 1
    }


def test_write_schema_consistency_report(tmp_path):
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string", "minLength": 1}},
    }
    result = evaluate_schema_consistency(pd.DataFrame({"name": ["SAP"]}), schema)
    output_path = tmp_path / "metrics" / "consistency.json"

    written = write_schema_consistency_report(
        result,
        output_path,
        metadata={"usecase": "companies"},
    )

    assert written == output_path
    payload = output_path.read_text(encoding="utf-8")
    assert '"metric": "consistency"' in payload
    assert '"usecase": "companies"' in payload
    assert '"consistency_score": 1.0' in payload
