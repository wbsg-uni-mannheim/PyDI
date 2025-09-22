"""Tests for base entity matching classes and functions."""

import pandas as pd
import pytest
from typing import Iterable

from PyDI.entitymatching.base import (
    BaseMatcher, 
    BaseComparator, 
    CorrespondenceSet,
    ensure_record_ids
)


class ConcreteMatcher(BaseMatcher):
    """Concrete implementation for testing the abstract base class."""
    
    def match(self, df_left, df_right, candidates, threshold=0.0, **kwargs):
        """Simple test implementation that returns empty correspondence set."""
        return pd.DataFrame(columns=["id1", "id2", "score", "notes"])


class ConcreteComparator(BaseComparator):
    """Concrete implementation for testing the abstract base class."""
    
    def compare(self, record1: pd.Series, record2: pd.Series) -> float:
        """Simple test implementation that returns 0.5."""
        return 0.5


class TestBaseMatcher:
    """Test the BaseMatcher abstract base class."""
    
    def test_abstract_class_cannot_be_instantiated(self):
        """Test that BaseMatcher cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseMatcher()
    
    def test_concrete_implementation_can_be_instantiated(self):
        """Test that concrete implementations can be instantiated."""
        matcher = ConcreteMatcher()
        assert isinstance(matcher, BaseMatcher)
    
    def test_match_method_signature(self, sample_movies_left, sample_movies_right, sample_candidate_pairs):
        """Test that the match method has the correct signature and returns CorrespondenceSet."""
        matcher = ConcreteMatcher()
        result = matcher.match(
            sample_movies_left, 
            sample_movies_right, 
            [sample_candidate_pairs]
        )
        
        # Should return a DataFrame (CorrespondenceSet)
        assert isinstance(result, pd.DataFrame)
        assert isinstance(result, CorrespondenceSet)
        
        # Should have the required columns
        expected_columns = ["id1", "id2", "score", "notes"]
        for col in expected_columns:
            assert col in result.columns
    
    def test_validate_inputs_valid_data(self, sample_movies_left, sample_movies_right):
        """Test _validate_inputs with valid data."""
        matcher = ConcreteMatcher()
        
        # Add _id columns
        sample_movies_left = ensure_record_ids(sample_movies_left)
        sample_movies_right = ensure_record_ids(sample_movies_right)
        
        # Should not raise any errors
        matcher._validate_inputs(sample_movies_left, sample_movies_right)
    
    def test_validate_inputs_missing_id_column(self, sample_movies_left, sample_movies_right):
        """Test _validate_inputs with missing _id column."""
        matcher = ConcreteMatcher()
        
        # Remove _id columns to test validation
        left_no_id = sample_movies_left.drop('_id', axis=1)
        right_no_id = sample_movies_right.drop('_id', axis=1)
        
        with pytest.raises(ValueError, match="must have '_id' column"):
            matcher._validate_inputs(left_no_id, right_no_id)
    
    def test_validate_inputs_missing_dataset_name(self, sample_movies_left, sample_movies_right):
        """Test _validate_inputs with missing dataset_name."""
        matcher = ConcreteMatcher()
        
        # Add _id columns but remove dataset_name
        sample_movies_left = ensure_record_ids(sample_movies_left)
        sample_movies_right = ensure_record_ids(sample_movies_right)
        
        del sample_movies_left.attrs["dataset_name"]
        
        with pytest.raises(ValueError, match="must have 'dataset_name' in df.attrs"):
            matcher._validate_inputs(sample_movies_left, sample_movies_right)
    
    def test_validate_inputs_same_dataset_names(self, sample_movies_left, sample_movies_right):
        """Test _validate_inputs with same dataset names (should warn)."""
        matcher = ConcreteMatcher()
        
        # Add _id columns and set same dataset name
        sample_movies_left = ensure_record_ids(sample_movies_left)
        sample_movies_right = ensure_record_ids(sample_movies_right)
        sample_movies_right.attrs["dataset_name"] = "academy_awards"  # Same as left
        
        # Should not raise error but might log warning
        matcher._validate_inputs(sample_movies_left, sample_movies_right)
    
    def test_log_matching_info(self, sample_movies_left, sample_movies_right, sample_candidate_pairs):
        """Test _log_matching_info doesn't crash."""
        matcher = ConcreteMatcher()
        
        sample_movies_left = ensure_record_ids(sample_movies_left)
        sample_movies_right = ensure_record_ids(sample_movies_right)
        
        # Should not raise any errors
        matcher._log_matching_info(
            sample_movies_left, 
            sample_movies_right, 
            [sample_candidate_pairs]
        )


class TestBaseComparator:
    """Test the BaseComparator base class."""
    
    def test_can_be_instantiated(self):
        """Test that BaseComparator can be instantiated."""
        comparator = ConcreteComparator("test_comparator")
        assert isinstance(comparator, BaseComparator)
        assert comparator.name == "test_comparator"
    
    def test_compare_method_signature(self):
        """Test that compare method works correctly."""
        comparator = ConcreteComparator("test")
        
        record1 = pd.Series({"title": "Movie A", "year": 2020})
        record2 = pd.Series({"title": "Movie B", "year": 2021})
        
        result = comparator.compare(record1, record2)
        assert isinstance(result, float)
        assert result == 0.5  # Our concrete implementation returns 0.5
    
    def test_repr(self):
        """Test string representation."""
        comparator = ConcreteComparator("test_name")
        repr_str = repr(comparator)
        assert "ConcreteComparator" in repr_str
        assert "test_name" in repr_str


class TestEnsureRecordIds:
    """Test the ensure_record_ids function."""
    
    def test_adds_id_column_when_missing(self, df_without_record_ids):
        """Test that _id column is added when missing."""
        result = ensure_record_ids(df_without_record_ids)
        
        # Should have _id column
        assert "_id" in result.columns
        
        # IDs should be in correct format
        expected_ids = ["test_movies_000000", "test_movies_000001", "test_movies_000002"]
        assert result["_id"].tolist() == expected_ids
        
        # Original data should be preserved
        assert list(result.columns) == ["title", "year", "genre", "_id"]
        assert len(result) == 3
    
    def test_preserves_existing_id_column(self, df_with_record_ids):
        """Test that existing _id column is preserved."""
        original_ids = df_with_record_ids["_id"].tolist()
        result = ensure_record_ids(df_with_record_ids)
        
        # Should preserve existing IDs
        assert result["_id"].tolist() == original_ids
        assert len(result) == 3
    
    def test_requires_dataset_name_in_attrs(self):
        """Test that dataset_name is required in attrs."""
        df = pd.DataFrame({"col1": [1, 2, 3]})
        # No dataset_name in attrs
        
        with pytest.raises(ValueError, match="must have 'dataset_name' in df.attrs"):
            ensure_record_ids(df)
    
    def test_adds_provenance(self, df_without_record_ids):
        """Test that provenance is added to attrs."""
        result = ensure_record_ids(df_without_record_ids)
        
        # Should have provenance
        assert "provenance" in result.attrs
        provenance = result.attrs["provenance"]
        assert len(provenance) == 1
        assert provenance[0]["op"] == "ensure_record_ids"
        assert provenance[0]["params"]["dataset_name"] == "test_movies"
    
    def test_preserves_existing_provenance(self):
        """Test that existing provenance is preserved and extended."""
        df = pd.DataFrame({"col1": [1, 2, 3]})
        df.attrs["dataset_name"] = "test"
        df.attrs["provenance"] = [{"op": "previous_op", "params": {}}]
        
        result = ensure_record_ids(df)
        
        # Should preserve and extend provenance
        assert "provenance" in result.attrs
        provenance = result.attrs["provenance"]
        assert len(provenance) == 2
        assert provenance[0]["op"] == "previous_op"
        assert provenance[1]["op"] == "ensure_record_ids"
    
    def test_returns_copy_not_reference(self, df_without_record_ids):
        """Test that function returns copy, not reference."""
        original_columns = list(df_without_record_ids.columns)
        result = ensure_record_ids(df_without_record_ids)
        
        # Original DataFrame should be unchanged
        assert list(df_without_record_ids.columns) == original_columns
        assert "_id" not in df_without_record_ids.columns
        
        # Result should have _id column
        assert "_id" in result.columns
    
    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        df.attrs["dataset_name"] = "empty_test"
        
        result = ensure_record_ids(df)
        
        # Should have _id column even if empty
        assert "_id" in result.columns
        assert len(result) == 0


class TestCorrespondenceSet:
    """Test the CorrespondenceSet type alias."""
    
    def test_correspondence_set_is_dataframe(self):
        """Test that CorrespondenceSet is a DataFrame type."""
        # Create a sample correspondence set
        corr_data = {
            "id1": ["academy_awards_1", "academy_awards_2"],
            "id2": ["actors_1", "actors_2"],
            "score": [0.95, 0.87],
            "notes": ["high_confidence", "medium_confidence"]
        }
        correspondence_set = pd.DataFrame(corr_data)
        
        # Should be a valid CorrespondenceSet (which is just a DataFrame)
        assert isinstance(correspondence_set, CorrespondenceSet)
        assert isinstance(correspondence_set, pd.DataFrame)
    
    def test_expected_structure(self):
        """Test that CorrespondenceSet has expected structure."""
        corr_data = {
            "id1": ["academy_awards_1", "academy_awards_2", "academy_awards_3"],
            "id2": ["actors_1", "actors_2", "actors_3"],
            "score": [0.95, 0.87, 0.92],
            "notes": ["exact_match", "fuzzy_match", "high_confidence"]
        }
        correspondence_set = pd.DataFrame(corr_data)
        
        # Check expected columns exist
        assert "id1" in correspondence_set.columns
        assert "id2" in correspondence_set.columns
        assert "score" in correspondence_set.columns
        assert "notes" in correspondence_set.columns
        
        # Check data types
        assert len(correspondence_set) == 3
        assert all(isinstance(score, (int, float)) for score in correspondence_set["score"])
    
    def test_empty_correspondence_set(self, empty_correspondence_set):
        """Test that empty CorrespondenceSet is valid."""
        assert isinstance(empty_correspondence_set, CorrespondenceSet)
        assert len(empty_correspondence_set) == 0
        assert list(empty_correspondence_set.columns) == ["id1", "id2", "score", "notes"]