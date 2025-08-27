"""Tests for base schema matching classes."""

import pandas as pd
import pytest
from abc import ABC

from PyDI.schemamatching.base import BaseSchemaMatcher, SchemaMapping


class ConcreteSchemaMatcher(BaseSchemaMatcher):
    """Concrete implementation for testing the abstract base class."""
    
    def match(self, source_dataset, target_dataset, preprocess=None, threshold=0.8, **kwargs):
        """Simple test implementation that returns empty mapping."""
        return pd.DataFrame(columns=[
            "source_dataset", "source_column", 
            "target_dataset", "target_column", 
            "score", "notes"
        ])


class TestBaseSchemaMatcher:
    """Test the BaseSchemaMatcher abstract base class."""
    
    def test_abstract_class_cannot_be_instantiated(self):
        """Test that BaseSchemaMatcher cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseSchemaMatcher()
    
    def test_concrete_implementation_can_be_instantiated(self):
        """Test that concrete implementations can be instantiated."""
        matcher = ConcreteSchemaMatcher()
        assert isinstance(matcher, BaseSchemaMatcher)
    
    def test_match_method_signature(self, sample_movies_df, sample_films_df):
        """Test that the match method has the correct signature and returns SchemaMapping."""
        matcher = ConcreteSchemaMatcher()
        result = matcher.match(sample_movies_df, sample_films_df)
        
        # Should return a DataFrame (SchemaMapping)
        assert isinstance(result, pd.DataFrame)
        
        # Should have the required columns
        expected_columns = [
            "source_dataset", "source_column", 
            "target_dataset", "target_column", 
            "score", "notes"
        ]
        for col in expected_columns:
            assert col in result.columns
    
    def test_match_with_optional_parameters(self, sample_movies_df, sample_films_df, preprocessing_func):
        """Test that match method accepts optional parameters."""
        matcher = ConcreteSchemaMatcher()
        
        # Should not raise any errors
        result = matcher.match(
            sample_movies_df, 
            sample_films_df,
            preprocess=preprocessing_func,
            threshold=0.5,
            extra_param="test"
        )
        
        assert isinstance(result, pd.DataFrame)


class TestSchemaMapping:
    """Test the SchemaMapping type alias."""
    
    def test_schema_mapping_is_dataframe(self):
        """Test that SchemaMapping is a DataFrame type."""
        # Create a sample mapping
        mapping_data = {
            "source_dataset": ["movies"],
            "source_column": ["title"],
            "target_dataset": ["films"],
            "target_column": ["film_name"],
            "score": [0.95],
            "notes": ["test"]
        }
        mapping = pd.DataFrame(mapping_data)
        
        # Should be a valid SchemaMapping (which is just a DataFrame)
        assert isinstance(mapping, SchemaMapping)
        assert isinstance(mapping, pd.DataFrame)
    
    def test_schema_mapping_expected_structure(self):
        """Test that SchemaMapping has expected structure."""
        mapping_data = {
            "source_dataset": ["movies", "movies"],
            "source_column": ["title", "year"],
            "target_dataset": ["films", "films"],
            "target_column": ["film_name", "release_year"],
            "score": [0.95, 0.85],
            "notes": ["exact_match", "year_match"]
        }
        mapping = pd.DataFrame(mapping_data)
        
        # Check expected columns exist
        assert "source_dataset" in mapping.columns
        assert "source_column" in mapping.columns
        assert "target_dataset" in mapping.columns
        assert "target_column" in mapping.columns
        assert "score" in mapping.columns
        
        # Check data types
        assert len(mapping) == 2
        assert all(isinstance(score, (int, float)) for score in mapping["score"])
    
    def test_empty_schema_mapping(self):
        """Test that empty SchemaMapping is valid."""
        empty_mapping = pd.DataFrame(columns=[
            "source_dataset", "source_column", 
            "target_dataset", "target_column", 
            "score", "notes"
        ])
        
        assert isinstance(empty_mapping, SchemaMapping)
        assert len(empty_mapping) == 0
        assert list(empty_mapping.columns) == [
            "source_dataset", "source_column", 
            "target_dataset", "target_column", 
            "score", "notes"
        ]