"""
Unit tests for taxonomy-based normalization.

Tests cover:
1. TaxonomyLoader - loading taxonomy values from CSV files
2. TaxonomyMapper - LLM-based mapping creation (with mocked LLM)
3. Cache functions - saving and loading taxonomy mappings
4. Integration with transform_dataframe
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from PyDI.normalization import (
    ColumnSpec,
    NormalizationSpec,
    transform_dataframe,
)
from PyDI.normalization.taxonomy import (
    TaxonomyLoader,
    TaxonomyMapper,
    TaxonomyMappingResult,
    apply_taxonomy_mapping,
    load_mapping_cache,
    save_mapping_cache,
)


class TestTaxonomyLoader:
    """Tests for TaxonomyLoader class."""

    def test_load_csv_first_column(self, tmp_path):
        """Test loading taxonomy with default first column."""
        csv_path = tmp_path / "taxonomy.csv"
        csv_path.write_text("Industry,Code\nSoftware,45\nBanks,40\nEnergy,10\n")

        loader = TaxonomyLoader()
        values = loader.load(csv_path)

        assert len(values) == 3
        assert "Software" in values
        assert "Banks" in values
        assert "Energy" in values

    def test_load_csv_specified_column(self, tmp_path):
        """Test loading taxonomy from a specified column."""
        csv_path = tmp_path / "taxonomy.csv"
        csv_path.write_text(
            "Sector,Industry Group,Industry\n"
            "Tech,Software,Application Software\n"
            "Tech,Software,Systems Software\n"
            "Finance,Banks,Regional Banks\n"
        )

        loader = TaxonomyLoader()
        values = loader.load(csv_path, column="Industry")

        assert len(values) == 3
        assert "Application Software" in values
        assert "Systems Software" in values
        assert "Regional Banks" in values

    def test_load_csv_unique_values(self, tmp_path):
        """Test that loader returns unique values only."""
        csv_path = tmp_path / "taxonomy.csv"
        csv_path.write_text("Industry\nSoftware\nSoftware\nBanks\nSoftware\n")

        loader = TaxonomyLoader()
        values = loader.load(csv_path)

        assert len(values) == 2
        assert "Software" in values
        assert "Banks" in values

    def test_load_csv_missing_file(self, tmp_path):
        """Test error when file doesn't exist."""
        loader = TaxonomyLoader()

        with pytest.raises(FileNotFoundError):
            loader.load(tmp_path / "nonexistent.csv")

    def test_load_csv_empty_file(self, tmp_path):
        """Test error when file is empty."""
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("")

        loader = TaxonomyLoader()

        with pytest.raises(ValueError, match="empty"):
            loader.load(csv_path)

    def test_load_csv_missing_column(self, tmp_path):
        """Test error when specified column doesn't exist."""
        csv_path = tmp_path / "taxonomy.csv"
        csv_path.write_text("Industry,Code\nSoftware,45\n")

        loader = TaxonomyLoader()

        with pytest.raises(ValueError, match="not found"):
            loader.load(csv_path, column="NonExistent")

    def test_load_csv_with_base_path(self, tmp_path):
        """Test loading with relative path and base_path."""
        # Create subdirectory structure
        tax_dir = tmp_path / "taxonomies"
        tax_dir.mkdir()
        csv_path = tax_dir / "industries.csv"
        csv_path.write_text("Name\nSoftware\nBanks\n")

        loader = TaxonomyLoader()
        values = loader.load("taxonomies/industries.csv", base_path=tmp_path)

        assert len(values) == 2
        assert "Software" in values

    def test_load_full_csv(self, tmp_path):
        """Test loading full CSV content as string."""
        csv_content = "Sector,Industry\nTech,Software\nFinance,Banks\n"
        csv_path = tmp_path / "taxonomy.csv"
        csv_path.write_text(csv_content)

        loader = TaxonomyLoader()
        content = loader.load_full_csv(csv_path)

        assert content == csv_content

    def test_caching(self, tmp_path):
        """Test that loader caches results."""
        csv_path = tmp_path / "taxonomy.csv"
        csv_path.write_text("Industry\nSoftware\nBanks\n")

        loader = TaxonomyLoader()

        # Load twice
        values1 = loader.load(csv_path)
        values2 = loader.load(csv_path)

        assert values1 == values2

        # Clear cache and reload
        loader.clear_cache()
        values3 = loader.load(csv_path)
        assert values3 == values1


class TestTaxonomyMappingResult:
    """Tests for TaxonomyMappingResult dataclass."""

    def test_create_result(self):
        """Test creating a mapping result."""
        result = TaxonomyMappingResult(
            mapping={"Tech": "Software", "Banking": "Banks", "Unknown": None},
            unmapped=["Unknown"],
            taxonomy_values=["Software", "Banks", "Energy"],
            taxonomy_column="Industry",
            llm_model="gpt-4o-mini",
        )

        assert result.mapping["Tech"] == "Software"
        assert result.mapping["Unknown"] is None
        assert "Unknown" in result.unmapped
        assert len(result.taxonomy_values) == 3
        assert result.taxonomy_column == "Industry"


class TestCacheFunctions:
    """Tests for cache save/load functions."""

    def test_save_and_load_cache(self, tmp_path):
        """Test saving and loading a mapping cache."""
        cache_path = tmp_path / "mapping.json"

        result = TaxonomyMappingResult(
            mapping={"Tech": "Software", "Banking": "Banks"},
            unmapped=[],
            taxonomy_values=["Software", "Banks"],
            taxonomy_column="Industry",
            llm_model="gpt-4o-mini",
        )

        # Save
        save_mapping_cache(cache_path, result)
        assert cache_path.exists()

        # Load
        loaded = load_mapping_cache(cache_path)
        assert loaded is not None
        assert loaded["Tech"] == "Software"
        assert loaded["Banking"] == "Banks"

    def test_load_nonexistent_cache(self, tmp_path):
        """Test loading from nonexistent file returns None."""
        cache_path = tmp_path / "nonexistent.json"
        loaded = load_mapping_cache(cache_path)
        assert loaded is None

    def test_load_invalid_cache(self, tmp_path):
        """Test loading invalid cache returns None."""
        cache_path = tmp_path / "invalid.json"
        cache_path.write_text("not json")

        loaded = load_mapping_cache(cache_path)
        assert loaded is None

    def test_load_cache_wrong_structure(self, tmp_path):
        """Test loading cache with wrong structure returns None."""
        cache_path = tmp_path / "wrong.json"
        cache_path.write_text('{"wrong_key": "value"}')

        loaded = load_mapping_cache(cache_path)
        assert loaded is None

    def test_cache_creates_directories(self, tmp_path):
        """Test that save creates parent directories."""
        cache_path = tmp_path / "nested" / "dir" / "mapping.json"

        result = TaxonomyMappingResult(
            mapping={"A": "B"},
            unmapped=[],
            taxonomy_values=["B"],
        )

        save_mapping_cache(cache_path, result)
        assert cache_path.exists()


class TestApplyTaxonomyMapping:
    """Tests for apply_taxonomy_mapping function."""

    def test_apply_mapping(self):
        """Test applying a mapping to a Series."""
        series = pd.Series(["Tech", "Banking", "Software", None, "Unknown"])
        mapping = {"Tech": "Software", "Banking": "Banks", "Software": "Software"}

        result = apply_taxonomy_mapping(series, mapping)

        assert result[0] == "Software"  # Tech -> Software
        assert result[1] == "Banks"  # Banking -> Banks
        assert result[2] == "Software"  # Software -> Software (already valid)
        assert pd.isna(result[3])  # None stays None
        assert result[4] == "Unknown"  # Not in mapping, keep original

    def test_apply_mapping_with_nulls(self):
        """Test that null mappings keep original values."""
        series = pd.Series(["Tech", "Unknown"])
        mapping = {"Tech": "Software", "Unknown": None}

        result = apply_taxonomy_mapping(series, mapping)

        assert result[0] == "Software"
        assert result[1] == "Unknown"  # Mapped to None, keep original

    def test_apply_mapping_empty_strings(self):
        """Test handling of empty strings."""
        series = pd.Series(["Tech", "", "  "])
        mapping = {"Tech": "Software"}

        result = apply_taxonomy_mapping(series, mapping)

        assert result[0] == "Software"
        assert result[1] == ""
        assert result[2] == "  "


class TestTaxonomyMapper:
    """Tests for TaxonomyMapper class with mocked LLM."""

    def test_create_mapping_empty_values(self):
        """Test creating mapping with empty source values."""
        mock_model = MagicMock()

        mapper = TaxonomyMapper(mock_model)
        result = mapper.create_mapping(
            source_values=[],
            taxonomy_csv_content="Industry\nSoftware\nBanks\n",
            taxonomy_column="Industry",
        )

        assert result.mapping == {}
        assert result.unmapped == []
        # LLM should not be called for empty values
        mock_model.invoke.assert_not_called()

    def test_create_mapping_with_llm(self):
        """Test creating mapping calls LLM correctly."""
        # Mock the LLM response
        mock_response = MagicMock()
        mock_response.content = '{"mappings": {"Tech": "Software", "Banking": "Banks"}}'

        mock_model = MagicMock()
        mock_model.invoke.return_value = mock_response

        mapper = TaxonomyMapper(mock_model, batch_size=100)
        result = mapper.create_mapping(
            source_values=["Tech", "Banking"],
            taxonomy_csv_content="Industry\nSoftware\nBanks\nEnergy\n",
            taxonomy_column="Industry",
        )

        assert result.mapping["Tech"] == "Software"
        assert result.mapping["Banking"] == "Banks"
        mock_model.invoke.assert_called_once()

    def test_create_mapping_handles_null_response(self):
        """Test handling of null values in LLM response."""
        mock_response = MagicMock()
        mock_response.content = '{"mappings": {"Tech": "Software", "Unknown": null}}'

        mock_model = MagicMock()
        mock_model.invoke.return_value = mock_response

        mapper = TaxonomyMapper(mock_model)
        result = mapper.create_mapping(
            source_values=["Tech", "Unknown"],
            taxonomy_csv_content="Industry\nSoftware\n",
            taxonomy_column="Industry",
        )

        assert result.mapping["Tech"] == "Software"
        assert result.mapping["Unknown"] is None
        assert "Unknown" in result.unmapped

    def test_create_mapping_batching(self):
        """Test that large value sets are batched."""
        mock_response = MagicMock()

        # First batch response
        mock_response.content = '{"mappings": {"A": "X", "B": "Y"}}'

        mock_model = MagicMock()
        mock_model.invoke.return_value = mock_response

        mapper = TaxonomyMapper(mock_model, batch_size=2)
        result = mapper.create_mapping(
            source_values=["A", "B", "C", "D"],
            taxonomy_csv_content="Industry\nX\nY\nZ\n",
            taxonomy_column="Industry",
        )

        # Should be called twice (2 batches of 2)
        assert mock_model.invoke.call_count == 2

    def test_parse_response_with_code_fences(self):
        """Test parsing response with markdown code fences."""
        mock_response = MagicMock()
        mock_response.content = '```json\n{"mappings": {"Tech": "Software"}}\n```'

        mock_model = MagicMock()
        mock_model.invoke.return_value = mock_response

        mapper = TaxonomyMapper(mock_model)
        result = mapper.create_mapping(
            source_values=["Tech"],
            taxonomy_csv_content="Industry\nSoftware\n",
            taxonomy_column="Industry",
        )

        assert result.mapping["Tech"] == "Software"

    def test_retry_on_failure(self):
        """Test that mapper retries on LLM failure."""
        mock_model = MagicMock()

        # First call fails, second succeeds
        mock_model.invoke.side_effect = [
            Exception("API error"),
            MagicMock(content='{"mappings": {"Tech": "Software"}}'),
        ]

        mapper = TaxonomyMapper(mock_model, max_retries=1, retry_delay=0.01)
        result = mapper.create_mapping(
            source_values=["Tech"],
            taxonomy_csv_content="Industry\nSoftware\n",
            taxonomy_column="Industry",
        )

        assert result.mapping["Tech"] == "Software"
        assert mock_model.invoke.call_count == 2


class TestColumnSpecTaxonomy:
    """Tests for ColumnSpec taxonomy fields."""

    def test_column_spec_taxonomy_fields(self):
        """Test ColumnSpec has taxonomy fields."""
        spec = ColumnSpec(
            taxonomy_path="taxonomies/industries.csv",
            taxonomy_column="Industry Name",
            taxonomy_mapping_path="output/mapping.json",
        )

        assert spec.taxonomy_path == "taxonomies/industries.csv"
        assert spec.taxonomy_column == "Industry Name"
        assert spec.taxonomy_mapping_path == "output/mapping.json"

    def test_column_spec_to_dict_includes_taxonomy(self):
        """Test to_dict includes taxonomy fields."""
        spec = ColumnSpec(taxonomy_path="test.csv")
        data = spec.to_dict()

        assert "taxonomy_path" in data
        assert data["taxonomy_path"] == "test.csv"


class TestJsonSchemaExtensions:
    """Tests for JSON Schema x-pydi-taxonomy extensions."""

    def test_load_schema_with_taxonomy(self, tmp_path):
        """Test loading schema with taxonomy extensions."""
        from PyDI.normalization import load_normalization_spec

        schema_path = tmp_path / "schema.json"
        schema_path.write_text(
            json.dumps(
                {
                    "type": "object",
                    "properties": {
                        "industry": {
                            "type": "string",
                            "x-pydi-taxonomy": "taxonomies/gics.csv",
                            "x-pydi-taxonomy-column": "Industry Name",
                        }
                    },
                }
            )
        )

        spec = load_normalization_spec(schema_path)

        assert "industry" in spec.columns
        assert spec.columns["industry"].taxonomy_path == "taxonomies/gics.csv"
        assert spec.columns["industry"].taxonomy_column == "Industry Name"


class TestTransformDataframeIntegration:
    """Integration tests for transform_dataframe with taxonomy."""

    def test_transform_with_cached_mapping(self, tmp_path):
        """Test transform_dataframe uses cached mapping."""
        # Create taxonomy CSV
        tax_path = tmp_path / "taxonomy.csv"
        tax_path.write_text("Industry\nSoftware\nBanks\nEnergy\n")

        # Create cached mapping
        cache_path = tmp_path / "mapping.json"
        cache_data = {
            "version": "1.0",
            "created_at": "2024-01-01T00:00:00Z",
            "llm_model": "test",
            "taxonomy_column": "Industry",
            "taxonomy_values": ["Software", "Banks", "Energy"],
            "mappings": {"Tech": "Software", "Banking": "Banks"},
            "unmapped": [],
        }
        cache_path.write_text(json.dumps(cache_data))

        # Create spec and dataframe
        spec = NormalizationSpec()
        spec.set_column(
            "industry",
            taxonomy_path=str(tax_path),
            taxonomy_column="Industry",
            taxonomy_mapping_path=str(cache_path),
        )

        df = pd.DataFrame({"industry": ["Tech", "Banking", "Software", "Unknown"]})

        # Transform (should use cache, no LLM needed)
        result = transform_dataframe(df, spec)

        assert result.dataframe["industry"][0] == "Software"
        assert result.dataframe["industry"][1] == "Banks"
        assert result.dataframe["industry"][2] == "Software"
        assert result.dataframe["industry"][3] == "Unknown"  # Not in mapping

    def test_transform_without_chat_model_keeps_unmapped_values(self, tmp_path):
        """Test taxonomy normalization is optional when no chat_model is supplied."""
        tax_path = tmp_path / "taxonomy.csv"
        tax_path.write_text("Industry\nSoftware\n")

        spec = NormalizationSpec()
        spec.set_column(
            "industry",
            taxonomy_path=str(tax_path),
            taxonomy_column="Industry",
        )

        df = pd.DataFrame({"industry": ["Tech", "Software"]})

        result = transform_dataframe(df, spec)

        assert result.dataframe["industry"][0] == "Tech"
        assert result.dataframe["industry"][1] == "Software"

    def test_transform_with_exact_matches(self, tmp_path):
        """Test that exact taxonomy matches don't need LLM."""
        tax_path = tmp_path / "taxonomy.csv"
        tax_path.write_text("Industry\nSoftware\nBanks\n")

        cache_path = tmp_path / "mapping.json"

        spec = NormalizationSpec()
        spec.set_column(
            "industry",
            taxonomy_path=str(tax_path),
            taxonomy_column="Industry",
            taxonomy_mapping_path=str(cache_path),
        )

        # All values already match taxonomy exactly
        df = pd.DataFrame({"industry": ["Software", "Banks", "Software"]})

        # Should work without chat_model since all values match
        result = transform_dataframe(df, spec)

        assert result.dataframe["industry"][0] == "Software"
        assert result.dataframe["industry"][1] == "Banks"

        # Cache should be created with direct mappings
        assert cache_path.exists()
