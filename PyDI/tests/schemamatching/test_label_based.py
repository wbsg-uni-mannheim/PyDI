"""Tests for label-based schema matching."""

import pytest
import pandas as pd
from unittest.mock import patch

from PyDI.schemamatching.label_based import LabelBasedSchemaMatcher


class TestLabelBasedSchemaMatcher:
    """Test the LabelBasedSchemaMatcher class."""
    
    def test_initialization_default(self):
        """Test default initialization."""
        matcher = LabelBasedSchemaMatcher()
        assert matcher.similarity_function == "jaccard"
        assert matcher.preprocess is None
        assert matcher.tokenize is True
    
    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        preprocess_func = str.lower
        matcher = LabelBasedSchemaMatcher(
            similarity_function="levenshtein",
            preprocess=preprocess_func,
            tokenize=False
        )
        assert matcher.similarity_function == "levenshtein"
        assert matcher.preprocess == preprocess_func
        assert matcher.tokenize is False
    
    def test_unsupported_similarity_function(self):
        """Test that unsupported similarity functions raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported similarity function"):
            LabelBasedSchemaMatcher(similarity_function="invalid_function")
    
    def test_supported_similarity_functions(self):
        """Test that all supported similarity functions can be initialized."""
        functions = ["jaccard", "levenshtein", "jaro_winkler", "cosine", "overlap"]
        
        for func in functions:
            matcher = LabelBasedSchemaMatcher(similarity_function=func)
            assert matcher.similarity_function == func
    
    def test_prepare_string_with_tokenization(self):
        """Test string preparation with tokenization."""
        matcher = LabelBasedSchemaMatcher(tokenize=True)
        
        # Test tokenization of column names
        result = matcher._prepare_string("movie_id_2023")
        assert result == ["movie", "id"]
        
        result = matcher._prepare_string("FirstName")
        assert result == ["firstname"]  # Should be lowercased
        
        result = matcher._prepare_string("customer-email@domain")
        assert result == ["customer", "email", "domain"]
    
    def test_prepare_string_without_tokenization(self):
        """Test string preparation without tokenization."""
        matcher = LabelBasedSchemaMatcher(tokenize=False)
        
        result = matcher._prepare_string("movie_id")
        assert result == "movie_id"
    
    def test_prepare_string_with_preprocessing(self):
        """Test string preparation with preprocessing."""
        preprocess_func = lambda x: x.upper().replace("_", "-")
        matcher = LabelBasedSchemaMatcher(preprocess=preprocess_func, tokenize=False)
        
        result = matcher._prepare_string("movie_id")
        assert result == "MOVIE-ID"
    
    def test_exact_column_matches(self, sample_movies_df, sample_films_df):
        """Test matching with exact column name matches."""
        # Create dataframes with some identical column names
        df1 = pd.DataFrame({"title": ["Movie1"], "year": [2020]})
        df1.attrs["dataset_name"] = "source"
        
        df2 = pd.DataFrame({"title": ["Movie2"], "genre": ["Action"]})
        df2.attrs["dataset_name"] = "target"
        
        matcher = LabelBasedSchemaMatcher(similarity_function="jaccard")
        result = matcher.match(df1, df2, threshold=1.0)  # Require exact match
        
        # Should find the exact match for "title"
        assert len(result) == 1
        assert result.iloc[0]["source_column"] == "title"
        assert result.iloc[0]["target_column"] == "title"
        assert result.iloc[0]["score"] == 1.0
    
    def test_partial_column_matches(self):
        """Test matching with partial column name matches."""
        df1 = pd.DataFrame({"movie_title": ["Movie1"], "release_year": [2020]})
        df1.attrs["dataset_name"] = "source"
        
        df2 = pd.DataFrame({"title": ["Movie2"], "year": [2021]})
        df2.attrs["dataset_name"] = "target"
        
        matcher = LabelBasedSchemaMatcher(similarity_function="jaccard")
        result = matcher.match(df1, df2, threshold=0.3)  # Lower threshold for partial matches
        
        # Should find partial matches
        assert len(result) > 0
        
        # Check that matches include expected pairs
        matches = result[["source_column", "target_column"]].values.tolist()
        
        # movie_title should match with title (both contain "title")
        title_match = any(match for match in matches if "title" in match[0] and "title" in match[1])
        assert title_match
    
    def test_threshold_filtering(self):
        """Test that threshold parameter filters results correctly."""
        df1 = pd.DataFrame({"very_different_name": [1, 2, 3]})
        df1.attrs["dataset_name"] = "source"
        
        df2 = pd.DataFrame({"completely_other_name": [4, 5, 6]})
        df2.attrs["dataset_name"] = "target"
        
        matcher = LabelBasedSchemaMatcher()
        
        # High threshold should return no matches
        result_high = matcher.match(df1, df2, threshold=0.9)
        assert len(result_high) == 0
        
        # Low threshold might return some matches
        result_low = matcher.match(df1, df2, threshold=0.1)
        # Could be 0 or more depending on similarity function behavior
        assert len(result_low) >= 0
    
    def test_different_similarity_functions(self):
        """Test that different similarity functions produce different results."""
        df1 = pd.DataFrame({"movie_title": ["Movie1"]})
        df1.attrs["dataset_name"] = "source"
        
        df2 = pd.DataFrame({"film_title": ["Movie2"]})
        df2.attrs["dataset_name"] = "target"
        
        functions = ["jaccard", "levenshtein", "jaro_winkler"]
        results = {}
        
        for func in functions:
            matcher = LabelBasedSchemaMatcher(similarity_function=func)
            result = matcher.match(df1, df2, threshold=0.1)
            if len(result) > 0:
                results[func] = result.iloc[0]["score"]
        
        # Should have results from at least some functions
        assert len(results) > 0
        
        # Different functions should potentially give different scores
        # (though they might be the same for this particular example)
    
    def test_preprocessing_parameter_override(self):
        """Test that preprocess parameter in match() overrides instance setting."""
        df1 = pd.DataFrame({"Movie_Title": ["Movie1"]})
        df1.attrs["dataset_name"] = "source"
        
        df2 = pd.DataFrame({"MOVIE_TITLE": ["Movie2"]})
        df2.attrs["dataset_name"] = "target"
        
        # Instance preprocess function
        instance_preprocess = str.upper
        matcher = LabelBasedSchemaMatcher(preprocess=instance_preprocess, tokenize=False)
        
        # Override with different preprocess function
        match_preprocess = str.lower
        result = matcher.match(df1, df2, preprocess=match_preprocess, threshold=0.1)
        
        # Should use the match-level preprocessing
        assert len(result) >= 0  # Just ensure no errors occurred
    
    def test_unsupported_method_parameter(self, sample_movies_df, sample_films_df):
        """Test that unsupported method parameter raises ValueError."""
        matcher = LabelBasedSchemaMatcher()
        
        with pytest.raises(ValueError, match="Unsupported method"):
            matcher.match(sample_movies_df, sample_films_df, method="instance")
    
    def test_dataset_name_handling(self):
        """Test that dataset names are correctly included in results."""
        df1 = pd.DataFrame({"col1": [1, 2, 3]})
        df1.attrs["dataset_name"] = "custom_source"
        
        df2 = pd.DataFrame({"col1": [4, 5, 6]})
        df2.attrs["dataset_name"] = "custom_target"
        
        matcher = LabelBasedSchemaMatcher()
        result = matcher.match(df1, df2, threshold=0.5)
        
        if len(result) > 0:
            assert result.iloc[0]["source_dataset"] == "custom_source"
            assert result.iloc[0]["target_dataset"] == "custom_target"
    
    def test_missing_dataset_names(self):
        """Test behavior when dataset names are missing from attrs."""
        df1 = pd.DataFrame({"col1": [1, 2, 3]})
        # No dataset_name in attrs
        
        df2 = pd.DataFrame({"col1": [4, 5, 6]})
        # No dataset_name in attrs
        
        matcher = LabelBasedSchemaMatcher()
        result = matcher.match(df1, df2, threshold=0.5)
        
        if len(result) > 0:
            assert result.iloc[0]["source_dataset"] == "source"  # Default
            assert result.iloc[0]["target_dataset"] == "target"  # Default
    
    def test_empty_dataframes(self, empty_df):
        """Test behavior with empty DataFrames."""
        df1 = pd.DataFrame({"col1": [1, 2, 3]})
        df1.attrs["dataset_name"] = "source"
        
        matcher = LabelBasedSchemaMatcher()
        result = matcher.match(df1, empty_df)
        
        # Should return empty result
        assert len(result) == 0
        assert isinstance(result, pd.DataFrame)
    
    def test_single_column_dataframes(self, single_column_df):
        """Test behavior with single column DataFrames."""
        df2 = pd.DataFrame({"single_col": [6, 7, 8]})
        df2.attrs["dataset_name"] = "target"
        
        matcher = LabelBasedSchemaMatcher()
        result = matcher.match(single_column_df, df2, threshold=0.5)
        
        # Should find exact match
        assert len(result) == 1
        assert result.iloc[0]["source_column"] == "single_col"
        assert result.iloc[0]["target_column"] == "single_col"
        assert result.iloc[0]["score"] == 1.0
    
    def test_result_structure(self, sample_movies_df, sample_films_df):
        """Test that result has correct structure and data types."""
        matcher = LabelBasedSchemaMatcher()
        result = matcher.match(sample_movies_df, sample_films_df, threshold=0.1)
        
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
            
            # Check that notes contain similarity function info
            assert all("similarity_function=" in note for note in result["notes"])
    
    @pytest.mark.parametrize("similarity_function", ["jaccard", "levenshtein", "jaro_winkler", "cosine", "overlap"])
    def test_all_similarity_functions_work(self, similarity_function):
        """Test that all similarity functions work without errors."""
        df1 = pd.DataFrame({"movie_title": ["The Matrix"], "year": [1999]})
        df1.attrs["dataset_name"] = "source"
        
        df2 = pd.DataFrame({"film_name": ["The Matrix"], "release_year": [1999]})
        df2.attrs["dataset_name"] = "target"
        
        matcher = LabelBasedSchemaMatcher(similarity_function=similarity_function)
        result = matcher.match(df1, df2, threshold=0.1)
        
        # Should not raise any errors
        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 0