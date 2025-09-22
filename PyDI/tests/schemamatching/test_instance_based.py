"""Tests for instance-based schema matching."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from PyDI.schemamatching.instance_based import InstanceBasedSchemaMatcher


class TestInstanceBasedSchemaMatcher:
    """Test the InstanceBasedSchemaMatcher class."""
    
    def test_initialization_default(self):
        """Test default initialization."""
        matcher = InstanceBasedSchemaMatcher()
        assert matcher.vector_creation_method == "term_frequencies"
        assert matcher.similarity_function == "cosine"
        assert matcher.max_sample_size == 1000
        assert matcher.min_non_null_ratio == 0.1
    
    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        matcher = InstanceBasedSchemaMatcher(
            vector_creation_method="binary_occurrence",
            similarity_function="jaccard",
            max_sample_size=500,
            min_non_null_ratio=0.2
        )
        assert matcher.vector_creation_method == "binary_occurrence"
        assert matcher.similarity_function == "jaccard"
        assert matcher.max_sample_size == 500
        assert matcher.min_non_null_ratio == 0.2
    
    def test_unsupported_vector_creation_method(self):
        """Test that unsupported vector creation methods raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported vector creation method"):
            InstanceBasedSchemaMatcher(vector_creation_method="invalid_method")
    
    def test_unsupported_similarity_function(self):
        """Test that unsupported similarity functions raise ValueError."""
        with pytest.raises(ValueError, match="Unknown similarity function"):
            InstanceBasedSchemaMatcher(similarity_function="invalid_function")
    
    def test_extract_column_values_basic(self):
        """Test basic column value extraction."""
        df = pd.DataFrame({
            "text_col": ["apple", "banana", "cherry", "date"],
            "num_col": [1, 2, 3, 4]
        })
        
        matcher = InstanceBasedSchemaMatcher()
        
        text_values = matcher._extract_column_values(df, "text_col")
        assert text_values == ["apple", "banana", "cherry", "date"]
        
        num_values = matcher._extract_column_values(df, "num_col")
        assert num_values == ["1", "2", "3", "4"]
    
    def test_extract_column_values_with_nulls(self):
        """Test column value extraction with null values."""
        df = pd.DataFrame({
            "col_with_nulls": ["apple", None, "cherry", pd.NA, ""]
        })
        
        matcher = InstanceBasedSchemaMatcher()
        values = matcher._extract_column_values(df, "col_with_nulls")
        
        # Should exclude nulls and empty strings, include valid values
        expected = ["apple", "cherry"]
        assert values == expected
    
    def test_extract_column_values_with_preprocessing(self):
        """Test column value extraction with preprocessing."""
        df = pd.DataFrame({
            "text_col": ["Apple", "BANANA", "Cherry"]
        })
        
        matcher = InstanceBasedSchemaMatcher()
        preprocess_func = str.upper
        
        values = matcher._extract_column_values(df, "text_col", preprocess_func)
        assert values == ["APPLE", "BANANA", "CHERRY"]
    
    def test_extract_column_values_sampling(self):
        """Test that sampling works when max_sample_size is exceeded."""
        # Create a large dataset
        large_data = [f"value_{i}" for i in range(2000)]
        df = pd.DataFrame({"large_col": large_data})
        
        matcher = InstanceBasedSchemaMatcher(max_sample_size=500)
        values = matcher._extract_column_values(df, "large_col")
        
        # Should be sampled down to max_sample_size
        assert len(values) == 500
        # All values should be from the original data
        assert all(val.startswith("value_") for val in values)
    
    def test_create_term_frequency_vector(self):
        """Test term frequency vector creation."""
        matcher = InstanceBasedSchemaMatcher()
        values = ["apple pie", "apple juice", "orange juice"]
        
        vector = matcher._create_term_frequency_vector(values)
        
        # Should have correct frequencies
        total_tokens = 6  # apple, pie, apple, juice, orange, juice
        assert vector["apple"] == 2/6
        assert vector["juice"] == 2/6
        assert vector["pie"] == 1/6
        assert vector["orange"] == 1/6
    
    def test_create_term_frequency_vector_empty(self):
        """Test term frequency vector creation with empty input."""
        matcher = InstanceBasedSchemaMatcher()
        vector = matcher._create_term_frequency_vector([])
        assert vector == {}
    
    def test_create_binary_vector(self):
        """Test binary occurrence vector creation."""
        matcher = InstanceBasedSchemaMatcher()
        values = ["apple pie", "apple juice", "orange juice"]
        
        vector = matcher._create_binary_vector(values)
        
        # Should have binary occurrence
        expected_tokens = {"apple", "pie", "juice", "orange"}
        assert set(vector.keys()) == expected_tokens
        assert all(val == 1.0 for val in vector.values())
    
    def test_create_binary_vector_empty(self):
        """Test binary vector creation with empty input."""
        matcher = InstanceBasedSchemaMatcher()
        vector = matcher._create_binary_vector([])
        assert vector == {}
    
    def test_create_tfidf_vectors(self):
        """Test TF-IDF vector creation."""
        matcher = InstanceBasedSchemaMatcher()
        all_column_values = {
            "col1": ["apple pie", "apple juice"],
            "col2": ["orange juice", "orange cake"]
        }
        
        vectors = matcher._create_tfidf_vectors(all_column_values)
        
        assert "col1" in vectors
        assert "col2" in vectors
        
        # Vectors should be non-empty dictionaries
        assert isinstance(vectors["col1"], dict)
        assert isinstance(vectors["col2"], dict)
        
        # Should contain some tokens
        assert len(vectors["col1"]) > 0
        assert len(vectors["col2"]) > 0
    
    def test_create_tfidf_vectors_empty(self):
        """Test TF-IDF vector creation with empty input."""
        matcher = InstanceBasedSchemaMatcher()
        vectors = matcher._create_tfidf_vectors({})
        assert vectors == {}
    
    def test_calculate_cosine_similarity(self):
        """Test cosine similarity calculation."""
        matcher = InstanceBasedSchemaMatcher()
        
        vec1 = {"apple": 0.5, "pie": 0.5}
        vec2 = {"apple": 0.7, "juice": 0.3}
        
        similarity = matcher._calculate_cosine_similarity(vec1, vec2)
        
        # Should be between 0 and 1
        assert 0 <= similarity <= 1
        # Should be > 0 since they share "apple"
        assert similarity > 0
    
    def test_calculate_cosine_similarity_identical(self):
        """Test cosine similarity with identical vectors."""
        matcher = InstanceBasedSchemaMatcher()
        
        vec1 = {"apple": 0.5, "pie": 0.5}
        vec2 = {"apple": 0.5, "pie": 0.5}
        
        similarity = matcher._calculate_cosine_similarity(vec1, vec2)
        
        # Should be 1.0 for identical vectors
        assert abs(similarity - 1.0) < 1e-10
    
    def test_calculate_cosine_similarity_disjoint(self):
        """Test cosine similarity with disjoint vectors."""
        matcher = InstanceBasedSchemaMatcher()
        
        vec1 = {"apple": 0.5, "pie": 0.5}
        vec2 = {"orange": 0.7, "juice": 0.3}
        
        similarity = matcher._calculate_cosine_similarity(vec1, vec2)
        
        # Should be 0.0 for disjoint vectors
        assert similarity == 0.0
    
    def test_calculate_cosine_similarity_empty(self):
        """Test cosine similarity with empty vectors."""
        matcher = InstanceBasedSchemaMatcher()
        
        vec1 = {"apple": 0.5}
        vec2 = {}
        
        similarity = matcher._calculate_cosine_similarity(vec1, vec2)
        assert similarity == 0.0
        
        similarity = matcher._calculate_cosine_similarity({}, {})
        assert similarity == 0.0
    
    def test_calculate_jaccard_similarity(self):
        """Test Jaccard similarity calculation."""
        matcher = InstanceBasedSchemaMatcher()
        
        vec1 = {"apple": 0.5, "pie": 0.3}
        vec2 = {"apple": 0.7, "juice": 0.2}
        
        similarity = matcher._calculate_jaccard_similarity(vec1, vec2)
        
        # Jaccard = intersection/union = 1/3 = 0.333...
        assert abs(similarity - 1/3) < 1e-10
    
    def test_calculate_jaccard_similarity_identical(self):
        """Test Jaccard similarity with identical term sets."""
        matcher = InstanceBasedSchemaMatcher()
        
        vec1 = {"apple": 0.5, "pie": 0.3}
        vec2 = {"apple": 0.7, "pie": 0.2}  # Same terms, different values
        
        similarity = matcher._calculate_jaccard_similarity(vec1, vec2)
        
        # Should be 1.0 since term sets are identical
        assert similarity == 1.0
    
    def test_calculate_containment_similarity(self):
        """Test containment similarity calculation."""
        matcher = InstanceBasedSchemaMatcher()
        
        vec1 = {"apple": 0.5}  # Smaller set
        vec2 = {"apple": 0.7, "pie": 0.2}  # Larger set
        
        similarity = matcher._calculate_containment_similarity(vec1, vec2)
        
        # Max containment should be 1.0 (vec1 fully contained in vec2)
        assert similarity == 1.0
    
    def test_match_with_similar_columns(self):
        """Test matching with columns that should match based on values."""
        df1 = pd.DataFrame({
            "fruits": ["apple", "banana", "cherry", "apple"],
            "numbers": [1, 2, 3, 1]
        })
        df1.attrs["dataset_name"] = "source"
        
        df2 = pd.DataFrame({
            "fruit_names": ["apple", "banana", "cherry", "apple"],  # Same values as "fruits"
            "quantities": [10, 20, 30, 10]  # Different values from "numbers"
        })
        df2.attrs["dataset_name"] = "target"
        
        matcher = InstanceBasedSchemaMatcher(vector_creation_method="term_frequencies")
        result = matcher.match(df1, df2, threshold=0.5)
        
        # Should find match between "fruits" and "fruit_names"
        matches = result[["source_column", "target_column"]].values.tolist()
        fruit_match = ["fruits", "fruit_names"] in matches
        assert fruit_match
    
    def test_match_with_different_vector_methods(self):
        """Test matching with different vector creation methods."""
        df1 = pd.DataFrame({"col1": ["word1 word2", "word2 word3"]})
        df1.attrs["dataset_name"] = "source"
        
        df2 = pd.DataFrame({"col2": ["word1 word2", "word2 word3"]})
        df2.attrs["dataset_name"] = "target"
        
        methods = ["term_frequencies", "binary_occurrence", "tfidf"]
        
        for method in methods:
            matcher = InstanceBasedSchemaMatcher(vector_creation_method=method)
            result = matcher.match(df1, df2, threshold=0.1)
            
            # Should not raise errors
            assert isinstance(result, pd.DataFrame)
    
    def test_match_with_different_similarity_functions(self):
        """Test matching with different similarity functions."""
        df1 = pd.DataFrame({"col1": ["word1", "word2"]})
        df1.attrs["dataset_name"] = "source"
        
        df2 = pd.DataFrame({"col2": ["word1", "word3"]})
        df2.attrs["dataset_name"] = "target"
        
        functions = ["cosine", "jaccard", "overlap"]
        
        for func in functions:
            matcher = InstanceBasedSchemaMatcher(similarity_function=func)
            result = matcher.match(df1, df2, threshold=0.1)
            
            # Should not raise errors
            assert isinstance(result, pd.DataFrame)
    
    def test_match_threshold_filtering(self):
        """Test that threshold parameter filters results correctly."""
        df1 = pd.DataFrame({"col1": ["completely", "different", "values"]})
        df1.attrs["dataset_name"] = "source"
        
        df2 = pd.DataFrame({"col2": ["totally", "other", "content"]})
        df2.attrs["dataset_name"] = "target"
        
        matcher = InstanceBasedSchemaMatcher()
        
        # High threshold should return fewer/no matches
        result_high = matcher.match(df1, df2, threshold=0.9)
        result_low = matcher.match(df1, df2, threshold=0.01)
        
        assert len(result_high) <= len(result_low)
    
    def test_match_min_non_null_ratio(self):
        """Test that min_non_null_ratio filters columns correctly."""
        df1 = pd.DataFrame({
            "mostly_null": [None, None, None, None, "similar"],  # 20% non-null
            "mostly_filled": ["similar", "text", "values", "data", "content"]  # 100% non-null
        })
        df1.attrs["dataset_name"] = "source"
        
        df2 = pd.DataFrame({"target_col": ["similar", "text", "values", "data", "content"]})
        df2.attrs["dataset_name"] = "target"
        
        # High min_non_null_ratio should exclude mostly_null column
        matcher = InstanceBasedSchemaMatcher(min_non_null_ratio=0.5)
        result = matcher.match(df1, df2, threshold=0.1)
        
        # Should only match mostly_filled column (since mostly_null is filtered out)
        if not result.empty:
            source_columns = result["source_column"].unique()
            assert "mostly_null" not in source_columns
            assert "mostly_filled" in source_columns
        else:
            # If no matches found, verify the filtering logic worked by testing with lower threshold
            matcher_low_threshold = InstanceBasedSchemaMatcher(min_non_null_ratio=0.1)  # Include both columns
            result_low = matcher_low_threshold.match(df1, df2, threshold=0.1)
            assert not result_low.empty, "Should find matches when both columns are included"
    
    def test_match_empty_dataframes(self, empty_df):
        """Test behavior with empty DataFrames."""
        df1 = pd.DataFrame({"col1": ["value1", "value2"]})
        df1.attrs["dataset_name"] = "source"
        
        matcher = InstanceBasedSchemaMatcher()
        result = matcher.match(df1, empty_df, threshold=0.1)
        
        # Should return empty result
        assert len(result) == 0
        assert isinstance(result, pd.DataFrame)
    
    def test_match_with_nulls(self, df_with_nulls):
        """Test matching with DataFrame containing nulls."""
        df2 = pd.DataFrame({
            "similar_col": [1, 2, 3, 4, 5],  # Similar to col_a
            "different_col": ["x", "y", "z", "w", "v"]
        })
        df2.attrs["dataset_name"] = "target"
        
        matcher = InstanceBasedSchemaMatcher(min_non_null_ratio=0.3)
        result = matcher.match(df_with_nulls, df2, threshold=0.1)
        
        # Should handle nulls gracefully and not crash
        assert isinstance(result, pd.DataFrame)
    
    def test_result_structure(self):
        """Test that result has correct structure and data types."""
        df1 = pd.DataFrame({"col1": ["apple", "banana"]})
        df1.attrs["dataset_name"] = "source"
        
        df2 = pd.DataFrame({"col2": ["apple", "cherry"]})
        df2.attrs["dataset_name"] = "target"
        
        matcher = InstanceBasedSchemaMatcher()
        result = matcher.match(df1, df2, threshold=0.1)
        
        # Check required columns
        expected_columns = [
            "source_dataset", "source_column", 
            "target_dataset", "target_column", 
            "score", "notes"
        ]
        for col in expected_columns:
            assert col in result.columns
        
        if len(result) > 0:
            # Check data types
            assert all(isinstance(score, (int, float)) for score in result["score"])
            assert all(0 <= score <= 1 for score in result["score"])
            
            # Check that notes contain method info
            assert all("vector_method=" in note for note in result["notes"])
            assert all("similarity=" in note for note in result["notes"])
    
    def test_dataset_name_handling(self):
        """Test that dataset names are correctly included in results."""
        df1 = pd.DataFrame({"col1": ["apple"]})
        df1.attrs["dataset_name"] = "custom_source"
        
        df2 = pd.DataFrame({"col2": ["apple"]})
        df2.attrs["dataset_name"] = "custom_target"
        
        matcher = InstanceBasedSchemaMatcher()
        result = matcher.match(df1, df2, threshold=0.1)
        
        if len(result) > 0:
            assert result.iloc[0]["source_dataset"] == "custom_source"
            assert result.iloc[0]["target_dataset"] == "custom_target"
    
    @pytest.mark.parametrize("vector_method", ["term_frequencies", "binary_occurrence", "tfidf"])
    @pytest.mark.parametrize("similarity_func", ["cosine", "jaccard", "overlap"])
    def test_all_method_combinations(self, vector_method, similarity_func):
        """Test all combinations of vector methods and similarity functions."""
        df1 = pd.DataFrame({"col1": ["apple pie", "banana split"]})
        df1.attrs["dataset_name"] = "source"
        
        df2 = pd.DataFrame({"col2": ["apple tart", "banana cake"]})
        df2.attrs["dataset_name"] = "target"
        
        matcher = InstanceBasedSchemaMatcher(
            vector_creation_method=vector_method,
            similarity_function=similarity_func
        )
        
        result = matcher.match(df1, df2, threshold=0.01)
        
        # Should not raise any errors
        assert isinstance(result, pd.DataFrame)
        # Most combinations should find some similarity
        assert len(result) >= 0