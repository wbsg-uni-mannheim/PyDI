"""Tests for MappingTranslator class."""

import pandas as pd
import pytest
from datetime import datetime, timezone

from PyDI.datatranslation.mapping_translator import MappingTranslator
from PyDI.datatranslation.base import BaseTranslator


class TestMappingTranslator:
    """Test the MappingTranslator class."""
    
    def test_inheritance(self):
        """Test that MappingTranslator inherits from BaseTranslator."""
        translator = MappingTranslator()
        assert isinstance(translator, BaseTranslator)
    
    def test_initialization_default_strategy(self):
        """Test default initialization with rename strategy."""
        translator = MappingTranslator()
        assert translator.strategy == "rename"
    
    def test_initialization_explicit_strategy(self):
        """Test initialization with explicit rename strategy."""
        translator = MappingTranslator(strategy="rename")
        assert translator.strategy == "rename"
    
    def test_initialization_invalid_strategy(self):
        """Test that invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported translation strategy"):
            MappingTranslator(strategy="invalid_strategy")
    
    def test_translate_basic_functionality(self, movies_df_for_translation, sample_schema_mapping):
        """Test basic translation functionality."""
        translator = MappingTranslator()
        result = translator.translate(movies_df_for_translation, sample_schema_mapping)
        
        # Should return a DataFrame
        assert isinstance(result, pd.DataFrame)
        
        # Should have renamed columns according to mapping
        expected_columns = {"id", "name", "release_year", "genre"}
        assert set(result.columns) == expected_columns
        
        # Should preserve data values
        assert len(result) == len(movies_df_for_translation)
        assert result["id"].tolist() == [1, 2, 3]
        assert result["name"].tolist() == ["The Matrix", "Inception", "Pulp Fiction"]
    
    def test_translate_no_matching_dataset(self, movies_df_for_translation):
        """Test translation when no mappings match the dataset."""
        # Create mapping for different dataset
        mapping_data = {
            "source_dataset": ["other_dataset", "another_dataset"],
            "source_column": ["col1", "col2"], 
            "target_dataset": ["unified", "unified"],
            "target_column": ["new_col1", "new_col2"],
            "score": [0.9, 0.8],
            "notes": ["test", "test"]
        }
        mapping = pd.DataFrame(mapping_data)
        
        translator = MappingTranslator()
        result = translator.translate(movies_df_for_translation, mapping)
        
        # Should return unchanged DataFrame
        pd.testing.assert_frame_equal(result, movies_df_for_translation)
    
    def test_translate_missing_dataset_name(self, df_without_dataset_name, sample_schema_mapping):
        """Test that missing dataset_name raises ValueError."""
        translator = MappingTranslator()
        
        with pytest.raises(ValueError, match="DataFrame is missing 'dataset_name' in attrs"):
            translator.translate(df_without_dataset_name, sample_schema_mapping)
    
    def test_translate_invalid_schema_mapping(self, movies_df_for_translation, invalid_schema_mapping):
        """Test that invalid schema mapping raises ValueError."""
        translator = MappingTranslator()
        
        with pytest.raises(ValueError, match="SchemaMapping is missing required columns"):
            translator.translate(movies_df_for_translation, invalid_schema_mapping)
    
    def test_translate_empty_mapping(self, movies_df_for_translation, empty_schema_mapping):
        """Test translation with empty schema mapping."""
        translator = MappingTranslator()
        result = translator.translate(movies_df_for_translation, empty_schema_mapping)
        
        # Should return unchanged DataFrame
        pd.testing.assert_frame_equal(result, movies_df_for_translation)
    
    def test_translate_partial_mapping(self, movies_df_for_translation):
        """Test translation when only some columns have mappings."""
        # Create mapping for only 2 out of 4 columns
        mapping_data = {
            "source_dataset": ["movies", "movies"],
            "source_column": ["movie_id", "title"],
            "target_dataset": ["unified", "unified"], 
            "target_column": ["id", "name"],
            "score": [0.95, 0.90],
            "notes": ["exact", "similar"]
        }
        mapping = pd.DataFrame(mapping_data)
        
        translator = MappingTranslator()
        result = translator.translate(movies_df_for_translation, mapping)
        
        # Should have 2 renamed columns + 2 original columns
        expected_columns = {"id", "name", "year", "genre"}
        assert set(result.columns) == expected_columns
    
    def test_translate_nonexistent_source_columns(self, movies_df_for_translation):
        """Test translation when mapping references non-existent columns."""
        # Create mapping with columns not in the DataFrame
        mapping_data = {
            "source_dataset": ["movies", "movies"],
            "source_column": ["nonexistent_col", "another_missing_col"],
            "target_dataset": ["unified", "unified"],
            "target_column": ["target1", "target2"],
            "score": [0.9, 0.8],
            "notes": ["test", "test"]
        }
        mapping = pd.DataFrame(mapping_data)
        
        translator = MappingTranslator()
        result = translator.translate(movies_df_for_translation, mapping)
        
        # Should return unchanged DataFrame (no applicable mappings)
        pd.testing.assert_frame_equal(result, movies_df_for_translation)
    
    def test_translate_preserves_attrs(self, movies_df_for_translation, sample_schema_mapping):
        """Test that DataFrame attrs are preserved during translation."""
        # Add some additional attrs
        movies_df_for_translation.attrs["custom_attr"] = "test_value"
        movies_df_for_translation.attrs["another_attr"] = {"nested": "value"}
        
        translator = MappingTranslator()
        result = translator.translate(movies_df_for_translation, sample_schema_mapping)
        
        # Should preserve original attrs
        assert result.attrs["dataset_name"] == "movies"
        assert result.attrs["custom_attr"] == "test_value"
        assert result.attrs["another_attr"] == {"nested": "value"}
        
        # Should add provenance
        assert "provenance" in result.attrs
        assert len(result.attrs["provenance"]) == 1
    
    def test_translate_adds_provenance(self, movies_df_for_translation, sample_schema_mapping):
        """Test that provenance information is added."""
        translator = MappingTranslator()
        result = translator.translate(movies_df_for_translation, sample_schema_mapping)
        
        # Should add provenance entry
        assert "provenance" in result.attrs
        provenance = result.attrs["provenance"]
        assert len(provenance) == 1
        
        entry = provenance[0]
        assert entry["op"] == "schema_translate"
        assert entry["params"]["strategy"] == "rename"
        assert entry["params"]["translator"] == "MappingTranslator"
        assert "mappings" in entry["params"]
        assert "ts" in entry
    
    def test_translate_column_level_provenance(self, df_with_column_attrs):
        """Test that column-level attributes and provenance are preserved."""
        # Create mapping for the test DataFrame
        mapping_data = {
            "source_dataset": ["test_dataset"],
            "source_column": ["col1"],
            "target_dataset": ["unified"],
            "target_column": ["new_col1"],
            "score": [0.9],
            "notes": ["test"]
        }
        mapping = pd.DataFrame(mapping_data)
        
        translator = MappingTranslator()
        result = translator.translate(df_with_column_attrs, mapping)
        
        # Check that column attributes were preserved for renamed column
        assert "unit" in result["new_col1"].attrs
        assert result["new_col1"].attrs["unit"] == "count"
        
        # Check that column provenance was added
        assert "provenance" in result["new_col1"].attrs
        col_provenance = result["new_col1"].attrs["provenance"]
        assert len(col_provenance) == 1
        assert col_provenance[0]["op"] == "schema_transform"
        assert col_provenance[0]["params"]["name_old"] == "col1"
        assert col_provenance[0]["params"]["name_new"] == "new_col1"
    
    def test_translate_duplicate_mappings(self, movies_df_for_translation):
        """Test behavior when there are duplicate mappings for the same column."""
        # Create mapping with duplicate source columns  
        mapping_data = {
            "source_dataset": ["movies", "movies"],
            "source_column": ["title", "title"],  # duplicate
            "target_dataset": ["unified", "unified"],
            "target_column": ["name", "title"],  # different targets
            "score": [0.9, 0.8],
            "notes": ["first", "second"]
        }
        mapping = pd.DataFrame(mapping_data)
        
        translator = MappingTranslator()
        result = translator.translate(movies_df_for_translation, mapping)
        
        # Should use the last mapping (title -> title)
        assert "title" in result.columns
        assert "name" not in result.columns
    
    def test_multiple_datasets_in_mapping(self, movies_df_for_translation, films_df_for_translation, sample_schema_mapping):
        """Test translation works correctly when mapping contains multiple datasets."""
        translator = MappingTranslator()
        
        # Translate movies dataset
        movies_result = translator.translate(movies_df_for_translation, sample_schema_mapping)
        expected_movies_columns = {"id", "name", "release_year", "genre"}
        assert set(movies_result.columns) == expected_movies_columns
        
        # Translate films dataset
        films_result = translator.translate(films_df_for_translation, sample_schema_mapping)
        expected_films_columns = {"id", "name", "release_year"}
        assert set(films_result.columns) == expected_films_columns