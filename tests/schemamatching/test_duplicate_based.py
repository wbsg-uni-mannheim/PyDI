"""Tests for duplicate-based schema matching."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from PyDI.schemamatching.duplicate_based import DuplicateBasedSchemaMatcher


class TestDuplicateBasedSchemaMatcher:
    """Test the DuplicateBasedSchemaMatcher class."""
    
    def test_initialization_default(self):
        """Test default initialization."""
        matcher = DuplicateBasedSchemaMatcher()
        assert matcher.vote_aggregation == "majority"
        assert matcher.value_comparison == "exact"
        assert matcher.min_votes == 1
        assert matcher.ignore_zero_values is True
    
    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        matcher = DuplicateBasedSchemaMatcher(
            vote_aggregation="weighted",
            value_comparison="normalized",
            min_votes=3,
            ignore_zero_values=False
        )
        assert matcher.vote_aggregation == "weighted"
        assert matcher.value_comparison == "normalized"
        assert matcher.min_votes == 3
        assert matcher.ignore_zero_values is False
    
    def test_unsupported_vote_aggregation(self):
        """Test that unsupported vote aggregation methods raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported vote aggregation"):
            DuplicateBasedSchemaMatcher(vote_aggregation="invalid_method")
    
    def test_unsupported_value_comparison(self):
        """Test that unsupported value comparison methods raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported value comparison"):
            DuplicateBasedSchemaMatcher(value_comparison="invalid_method")
    
    def test_normalize_value_basic(self):
        """Test basic value normalization."""
        matcher = DuplicateBasedSchemaMatcher()
        
        # Test string normalization
        assert matcher._normalize_value("  Hello World  ") == "hello world"
        assert matcher._normalize_value("UPPERCASE") == "uppercase"
        assert matcher._normalize_value(123) == "123"
        assert matcher._normalize_value(12.34) == "12.34"
    
    def test_normalize_value_with_nulls(self):
        """Test value normalization with null values."""
        matcher = DuplicateBasedSchemaMatcher()
        
        assert matcher._normalize_value(None) == ""
        assert matcher._normalize_value(pd.NA) == ""
        assert matcher._normalize_value(np.nan) == ""
    
    def test_normalize_value_normalized_comparison(self):
        """Test value normalization with normalized comparison method."""
        matcher = DuplicateBasedSchemaMatcher(value_comparison="normalized")
        
        # Should remove punctuation and extra whitespace
        result = matcher._normalize_value("Hello, World!   Extra   Spaces")
        assert result == "hello world extra spaces"
    
    def test_normalize_value_with_preprocessing(self):
        """Test value normalization with preprocessing function."""
        matcher = DuplicateBasedSchemaMatcher()
        preprocess_func = lambda x: x.replace(" ", "_")
        
        result = matcher._normalize_value("hello world", preprocess_func)
        assert result == "hello_world"
    
    def test_values_match_exact(self):
        """Test exact value matching."""
        matcher = DuplicateBasedSchemaMatcher(value_comparison="exact")
        
        assert matcher._values_match("hello", "hello") is True
        assert matcher._values_match("Hello", "hello") is True  # Case normalized
        assert matcher._values_match("  hello  ", "hello") is True  # Whitespace normalized
        assert matcher._values_match("hello", "world") is False
        assert matcher._values_match(123, "123") is True  # Type conversion
    
    def test_values_match_normalized(self):
        """Test normalized value matching."""
        matcher = DuplicateBasedSchemaMatcher(value_comparison="normalized")
        
        assert matcher._values_match("hello, world!", "hello world") is True
        assert matcher._values_match("hello", "world") is False
    
    def test_values_match_ignore_zero_values(self):
        """Test value matching with ignore_zero_values option."""
        matcher = DuplicateBasedSchemaMatcher(ignore_zero_values=True)
        
        # Should return False for zero/empty values
        assert matcher._values_match("", "anything") is False
        assert matcher._values_match("0", "0") is False
        assert matcher._values_match(None, None) is False
        assert matcher._values_match("null", "null") is False
        
        # Non-zero values should work normally
        assert matcher._values_match("hello", "hello") is True
    
    def test_values_match_include_zero_values(self):
        """Test value matching with ignore_zero_values=False."""
        matcher = DuplicateBasedSchemaMatcher(ignore_zero_values=False)
        
        # Should match zero/empty values
        assert matcher._values_match("0", "0") is True
        assert matcher._values_match("", "") is True
    
    def test_collect_votes_basic(self, sample_movies_df, sample_films_df, sample_correspondences_df):
        """Test basic vote collection from correspondences."""
        matcher = DuplicateBasedSchemaMatcher()
        
        votes = matcher._collect_votes(sample_movies_df, sample_films_df, sample_correspondences_df)
        
        # Should have collected some votes
        assert len(votes) > 0
        assert isinstance(votes, dict)
        
        # Votes should be for attribute pairs
        for (attr1, attr2), vote_count in votes.items():
            assert isinstance(attr1, str)
            assert isinstance(attr2, str)
            assert isinstance(vote_count, int)
            assert vote_count > 0
    
    def test_collect_votes_no_correspondences(self, sample_movies_df, sample_films_df):
        """Test vote collection with empty correspondences."""
        empty_correspondences = pd.DataFrame(columns=["id1", "id2"])
        matcher = DuplicateBasedSchemaMatcher()
        
        votes = matcher._collect_votes(sample_movies_df, sample_films_df, empty_correspondences)
        
        # Should return empty votes dictionary
        assert len(votes) == 0
    
    def test_collect_votes_missing_records(self):
        """Test vote collection when correspondence records are not found."""
        df1 = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})
        df2 = pd.DataFrame({"id": [3, 4], "name": ["Charlie", "David"]})
        
        # Correspondences refer to non-existent records
        correspondences = pd.DataFrame({"id1": [99, 100], "id2": [999, 1000]})
        
        matcher = DuplicateBasedSchemaMatcher()
        votes = matcher._collect_votes(df1, df2, correspondences)
        
        # Should return empty or minimal votes
        assert len(votes) == 0
    
    def test_collect_votes_alternative_id_columns(self):
        """Test vote collection with alternative ID column names."""
        df1 = pd.DataFrame({"source_id": [1, 2], "name": ["Alice", "Bob"]})
        df2 = pd.DataFrame({"target_id": [1, 2], "name": ["Alice", "Bob"]})
        
        # Use alternative correspondence column names
        correspondences = pd.DataFrame({"first_id": [1, 2], "second_id": [1, 2]})
        
        matcher = DuplicateBasedSchemaMatcher()
        votes = matcher._collect_votes(df1, df2, correspondences)
        
        # Should still collect votes (or handle gracefully)
        assert isinstance(votes, dict)
    
    def test_aggregate_votes_majority(self):
        """Test vote aggregation with majority method."""
        matcher = DuplicateBasedSchemaMatcher(vote_aggregation="majority", min_votes=2)
        
        votes = {
            ("col1", "col2"): 5,
            ("col1", "col3"): 3,
            ("col2", "col3"): 1,  # Below min_votes
        }
        
        results = matcher._aggregate_votes(votes, threshold=0.5)
        
        # Should include results above min_votes and threshold
        assert len(results) >= 1
        
        # Check structure
        for result in results:
            assert "source_column" in result
            assert "target_column" in result
            assert "score" in result
            assert "votes" in result
            assert result["votes"] >= matcher.min_votes
    
    def test_aggregate_votes_weighted(self):
        """Test vote aggregation with weighted method."""
        matcher = DuplicateBasedSchemaMatcher(vote_aggregation="weighted", min_votes=1)
        
        votes = {
            ("col1", "col2"): 6,  # 6/10 = 0.6
            ("col1", "col3"): 3,  # 3/10 = 0.3
            ("col2", "col3"): 1,  # 1/10 = 0.1
        }
        
        results = matcher._aggregate_votes(votes, threshold=0.25)
        
        # Should include results above threshold
        assert len(results) >= 2  # col1->col2 and col1->col3 should qualify
        
        # Check that scores are weighted properly
        score_map = {(r["source_column"], r["target_column"]): r["score"] for r in results}
        assert abs(score_map.get(("col1", "col2"), 0) - 0.6) < 1e-10
        assert abs(score_map.get(("col1", "col3"), 0) - 0.3) < 1e-10
    
    def test_aggregate_votes_empty(self):
        """Test vote aggregation with empty votes."""
        matcher = DuplicateBasedSchemaMatcher()
        
        results = matcher._aggregate_votes({}, threshold=0.5)
        
        assert results == []
    
    def test_match_basic_functionality(self, sample_movies_df, sample_films_df, sample_correspondences_df):
        """Test basic matching functionality."""
        matcher = DuplicateBasedSchemaMatcher()
        
        result = matcher.match(
            sample_movies_df,
            sample_films_df,
            correspondences=sample_correspondences_df,
            threshold=0.1
        )
        
        # Should return a DataFrame with correct structure
        assert isinstance(result, pd.DataFrame)
        
        expected_columns = [
            "source_dataset", "source_column", 
            "target_dataset", "target_column", 
            "score", "notes"
        ]
        for col in expected_columns:
            assert col in result.columns
    
    def test_match_no_correspondences_raises_error(self, sample_movies_df, sample_films_df):
        """Test that matching without correspondences raises ValueError."""
        matcher = DuplicateBasedSchemaMatcher()
        
        with pytest.raises(ValueError, match="requires record correspondences"):
            matcher.match(sample_movies_df, sample_films_df)
        
        with pytest.raises(ValueError, match="requires record correspondences"):
            matcher.match(sample_movies_df, sample_films_df, correspondences=pd.DataFrame())
    
    def test_match_with_preprocessing(self):
        """Test matching with preprocessing function."""
        df1 = pd.DataFrame({"id": [1, 2], "name": ["Alice Smith", "Bob Jones"]})
        df1.attrs["dataset_name"] = "source"
        
        df2 = pd.DataFrame({"id": [1, 2], "name": ["ALICE SMITH", "BOB JONES"]})
        df2.attrs["dataset_name"] = "target"
        
        correspondences = pd.DataFrame({"id1": [1, 2], "id2": [1, 2]})
        
        preprocess_func = str.upper
        matcher = DuplicateBasedSchemaMatcher()
        
        result = matcher.match(
            df1, df2, 
            correspondences=correspondences, 
            preprocess=preprocess_func,
            threshold=0.1
        )
        
        # Should find matches with preprocessing
        assert len(result) > 0
    
    def test_match_different_vote_aggregations(self):
        """Test matching with different vote aggregation methods."""
        df1 = pd.DataFrame({"id": [1, 2], "value": ["A", "B"]})
        df1.attrs["dataset_name"] = "source"
        
        df2 = pd.DataFrame({"id": [1, 2], "value": ["A", "B"]})
        df2.attrs["dataset_name"] = "target"
        
        correspondences = pd.DataFrame({"id1": [1, 2], "id2": [1, 2]})
        
        methods = ["majority", "weighted"]
        
        for method in methods:
            matcher = DuplicateBasedSchemaMatcher(vote_aggregation=method)
            result = matcher.match(df1, df2, correspondences=correspondences, threshold=0.1)
            
            # Should not raise errors
            assert isinstance(result, pd.DataFrame)
    
    def test_match_different_value_comparisons(self):
        """Test matching with different value comparison methods."""
        df1 = pd.DataFrame({"id": [1, 2], "text": ["hello, world!", "test data"]})
        df1.attrs["dataset_name"] = "source"
        
        df2 = pd.DataFrame({"id": [1, 2], "text": ["hello world", "test data"]})
        df2.attrs["dataset_name"] = "target"
        
        correspondences = pd.DataFrame({"id1": [1, 2], "id2": [1, 2]})
        
        methods = ["exact", "normalized"]
        
        for method in methods:
            matcher = DuplicateBasedSchemaMatcher(value_comparison=method)
            result = matcher.match(df1, df2, correspondences=correspondences, threshold=0.1)
            
            # Should not raise errors
            assert isinstance(result, pd.DataFrame)
    
    def test_match_threshold_filtering(self):
        """Test that threshold parameter filters results correctly."""
        df1 = pd.DataFrame({"id": [1, 2, 3], "col1": ["A", "B", "C"]})
        df1.attrs["dataset_name"] = "source"
        
        df2 = pd.DataFrame({"id": [1, 2, 3], "col2": ["A", "B", "X"]})  # Partial match
        df2.attrs["dataset_name"] = "target"
        
        correspondences = pd.DataFrame({"id1": [1, 2, 3], "id2": [1, 2, 3]})
        
        matcher = DuplicateBasedSchemaMatcher()
        
        result_low = matcher.match(df1, df2, correspondences=correspondences, threshold=0.1)
        result_high = matcher.match(df1, df2, correspondences=correspondences, threshold=0.9)
        
        # High threshold should return fewer results
        assert len(result_high) <= len(result_low)
    
    def test_match_min_votes_filtering(self):
        """Test that min_votes parameter filters results correctly."""
        df1 = pd.DataFrame({"id": [1, 2], "col1": ["A", "B"]})
        df1.attrs["dataset_name"] = "source"
        
        df2 = pd.DataFrame({"id": [1, 2], "col2": ["A", "B"]})
        df2.attrs["dataset_name"] = "target"
        
        correspondences = pd.DataFrame({"id1": [1], "id2": [1]})  # Only one correspondence
        
        matcher_low = DuplicateBasedSchemaMatcher(min_votes=1)
        matcher_high = DuplicateBasedSchemaMatcher(min_votes=3)
        
        result_low = matcher_low.match(df1, df2, correspondences=correspondences, threshold=0.1)
        result_high = matcher_high.match(df1, df2, correspondences=correspondences, threshold=0.1)
        
        # High min_votes should return fewer results
        assert len(result_high) <= len(result_low)
    
    def test_match_dataset_name_handling(self):
        """Test that dataset names are correctly included in results."""
        df1 = pd.DataFrame({"id": [1], "col1": ["A"]})
        df1.attrs["dataset_name"] = "custom_source"
        
        df2 = pd.DataFrame({"id": [1], "col2": ["A"]})
        df2.attrs["dataset_name"] = "custom_target"
        
        correspondences = pd.DataFrame({"id1": [1], "id2": [1]})
        
        matcher = DuplicateBasedSchemaMatcher()
        result = matcher.match(df1, df2, correspondences=correspondences, threshold=0.1)
        
        if len(result) > 0:
            assert result.iloc[0]["source_dataset"] == "custom_source"
            assert result.iloc[0]["target_dataset"] == "custom_target"
    
    def test_match_missing_dataset_names(self):
        """Test behavior when dataset names are missing from attrs."""
        df1 = pd.DataFrame({"id": [1], "col1": ["A"]})
        # No dataset_name in attrs
        
        df2 = pd.DataFrame({"id": [1], "col2": ["A"]})
        # No dataset_name in attrs
        
        correspondences = pd.DataFrame({"id1": [1], "id2": [1]})
        
        matcher = DuplicateBasedSchemaMatcher()
        result = matcher.match(df1, df2, correspondences=correspondences, threshold=0.1)
        
        if len(result) > 0:
            assert result.iloc[0]["source_dataset"] == "source"  # Default
            assert result.iloc[0]["target_dataset"] == "target"  # Default
    
    def test_match_result_structure_and_notes(self):
        """Test that result has correct structure and informative notes."""
        df1 = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})
        df1.attrs["dataset_name"] = "people1"
        
        df2 = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})
        df2.attrs["dataset_name"] = "people2"
        
        correspondences = pd.DataFrame({"id1": [1, 2], "id2": [1, 2]})
        
        matcher = DuplicateBasedSchemaMatcher()
        result = matcher.match(df1, df2, correspondences=correspondences, threshold=0.1)
        
        if len(result) > 0:
            # Check data types
            assert all(isinstance(score, (int, float)) for score in result["score"])
            assert all(0 <= score <= 1 for score in result["score"])
            
            # Check that notes contain method info
            assert all("votes=" in note for note in result["notes"])
            assert all("method=duplicate_based" in note for note in result["notes"])
    
    def test_edge_case_single_correspondence(self):
        """Test behavior with single correspondence."""
        df1 = pd.DataFrame({"id": [1], "data": ["test"]})
        df1.attrs["dataset_name"] = "source"
        
        df2 = pd.DataFrame({"id": [1], "data": ["test"]})
        df2.attrs["dataset_name"] = "target"
        
        correspondences = pd.DataFrame({"id1": [1], "id2": [1]})
        
        matcher = DuplicateBasedSchemaMatcher(min_votes=1)
        result = matcher.match(df1, df2, correspondences=correspondences, threshold=0.1)
        
        # Should handle single correspondence gracefully
        assert isinstance(result, pd.DataFrame)
    
    def test_edge_case_no_matching_values(self):
        """Test behavior when correspondences exist but values don't match."""
        df1 = pd.DataFrame({"id": [1, 2], "data": ["A", "B"]})
        df1.attrs["dataset_name"] = "source"
        
        df2 = pd.DataFrame({"id": [1, 2], "data": ["X", "Y"]})  # Different values
        df2.attrs["dataset_name"] = "target"
        
        correspondences = pd.DataFrame({"id1": [1, 2], "id2": [1, 2]})
        
        matcher = DuplicateBasedSchemaMatcher()
        result = matcher.match(df1, df2, correspondences=correspondences, threshold=0.1)
        
        # Should return empty result when no values match
        assert len(result) == 0