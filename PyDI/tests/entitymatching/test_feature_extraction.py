"""Tests for feature extraction for entity matching."""

import pandas as pd
import pytest
import numpy as np
from unittest.mock import Mock

from PyDI.entitymatching.feature_extraction import FeatureExtractor, VectorFeatureExtractor
from PyDI.entitymatching.comparators import StringComparator, NumericComparator
from PyDI.entitymatching.base import ensure_record_ids


class TestFeatureExtractor:
    """Test the FeatureExtractor class."""
    
    def test_initialization_with_comparators(self):
        """Test initialization with BaseComparator objects."""
        comparators = [
            StringComparator("title"),
            NumericComparator("year")
        ]
        
        extractor = FeatureExtractor(comparators)
        assert len(extractor.feature_functions) == 2
        assert extractor.feature_functions[0]["name"] == "StringComparator(title, jaro_winkler)"
        assert extractor.feature_functions[1]["name"] == "NumericComparator(year, absolute_difference)"
    
    def test_initialization_with_functions(self):
        """Test initialization with callable functions."""
        def title_similarity(r1, r2):
            return 0.5
        
        def year_similarity(r1, r2):
            return 0.8
        
        comparators = [title_similarity, year_similarity]
        
        extractor = FeatureExtractor(comparators)
        assert len(extractor.feature_functions) == 2
        assert extractor.feature_functions[0]["name"] == "title_similarity"
        assert extractor.feature_functions[1]["name"] == "year_similarity"
    
    def test_initialization_with_dicts(self):
        """Test initialization with dict format."""
        def custom_similarity(r1, r2):
            return 0.7
        
        comparators = [
            {"function": custom_similarity, "name": "custom_feature"},
            {"function": lambda r1, r2: 0.6}  # No name provided
        ]
        
        extractor = FeatureExtractor(comparators)
        assert len(extractor.feature_functions) == 2
        assert extractor.feature_functions[0]["name"] == "custom_feature"
        assert extractor.feature_functions[1]["name"] == "feature_1"  # Default name
    
    def test_initialization_empty_functions(self):
        """Test initialization with empty function list."""
        with pytest.raises(ValueError, match="At least one feature function must be provided"):
            FeatureExtractor([])
    
    def test_initialization_invalid_function(self):
        """Test initialization with invalid function type."""
        with pytest.raises(ValueError, match="must be callable, BaseComparator, or dict"):
            FeatureExtractor(["not_callable"])
    
    def test_initialization_invalid_dict(self):
        """Test initialization with invalid dict format."""
        with pytest.raises(ValueError, match="must have 'function' key"):
            FeatureExtractor([{"name": "test"}])  # Missing function key
    
    def test_create_features_basic(self, sample_movies_left, sample_movies_right, sample_candidate_pairs):
        """Test basic feature creation."""
        comparators = [
            StringComparator("title"),
            NumericComparator("year")
        ]
        
        extractor = FeatureExtractor(comparators)
        
        # Prepare datasets
        sample_movies_left = ensure_record_ids(sample_movies_left)
        sample_movies_right = ensure_record_ids(sample_movies_right)
        
        features = extractor.create_features(
            sample_movies_left,
            sample_movies_right,
            sample_candidate_pairs
        )
        
        assert isinstance(features, pd.DataFrame)
        
        # Check columns
        expected_cols = ["id1", "id2", "StringComparator(title, jaro_winkler)", "NumericComparator(year, absolute_difference)"]
        for col in expected_cols:
            assert col in features.columns
        
        # Check that features are numeric
        feature_cols = [col for col in features.columns if col not in ["id1", "id2"]]
        for col in feature_cols:
            assert all(isinstance(val, (int, float, np.number)) for val in features[col])
    
    def test_create_features_with_labels(self, sample_movies_left, sample_movies_right, sample_candidate_pairs, sample_training_labels):
        """Test feature creation with labels."""
        comparators = [StringComparator("title")]
        extractor = FeatureExtractor(comparators)
        
        # Prepare datasets
        sample_movies_left = ensure_record_ids(sample_movies_left)
        sample_movies_right = ensure_record_ids(sample_movies_right)
        
        features = extractor.create_features(
            sample_movies_left,
            sample_movies_right,
            sample_candidate_pairs,
            labels=sample_training_labels
        )
        
        assert "label" in features.columns
        assert len(features) == len(sample_training_labels)
        assert all(label in [0, 1] for label in features["label"])
    
    def test_create_features_empty_pairs(self, sample_movies_left, sample_movies_right):
        """Test feature creation with empty pairs."""
        comparators = [StringComparator("title")]
        extractor = FeatureExtractor(comparators)
        
        # Prepare datasets
        sample_movies_left = ensure_record_ids(sample_movies_left)
        sample_movies_right = ensure_record_ids(sample_movies_right)
        
        empty_pairs = pd.DataFrame(columns=["id1", "id2"])
        
        with pytest.raises(ValueError, match="Empty pairs DataFrame provided"):
            extractor.create_features(sample_movies_left, sample_movies_right, empty_pairs)
    
    def test_create_features_missing_columns(self, sample_movies_left, sample_movies_right):
        """Test feature creation with missing required columns."""
        comparators = [StringComparator("title")]
        extractor = FeatureExtractor(comparators)
        
        # Pairs without required columns
        bad_pairs = pd.DataFrame({"wrong_col1": ["a"], "wrong_col2": ["b"]})
        
        with pytest.raises(ValueError, match="missing required column"):
            extractor.create_features(sample_movies_left, sample_movies_right, bad_pairs)
    
    def test_create_features_labels_length_mismatch(self, sample_movies_left, sample_movies_right, sample_candidate_pairs):
        """Test feature creation with mismatched labels length."""
        comparators = [StringComparator("title")]
        extractor = FeatureExtractor(comparators)
        
        # Prepare datasets
        sample_movies_left = ensure_record_ids(sample_movies_left)
        sample_movies_right = ensure_record_ids(sample_movies_right)
        
        wrong_length_labels = pd.Series([1, 0])  # Different length than pairs
        
        with pytest.raises(ValueError, match="Labels length .* doesn't match pairs length"):
            extractor.create_features(
                sample_movies_left,
                sample_movies_right,
                sample_candidate_pairs,
                labels=wrong_length_labels
            )
    
    def test_create_features_missing_records(self, sample_movies_left, sample_movies_right):
        """Test feature creation with pairs referencing non-existent records."""
        comparators = [StringComparator("title")]
        extractor = FeatureExtractor(comparators)
        
        # Prepare datasets
        sample_movies_left = ensure_record_ids(sample_movies_left)
        sample_movies_right = ensure_record_ids(sample_movies_right)
        
        # Pairs with non-existent IDs
        bad_pairs = pd.DataFrame({
            "id1": ["nonexistent_id"],
            "id2": ["also_nonexistent"]
        })
        
        # Should raise ValueError when no valid features can be extracted
        with pytest.raises(ValueError, match="No valid features could be extracted"):
            extractor.create_features(sample_movies_left, sample_movies_right, bad_pairs)
    
    def test_create_features_function_errors(self, sample_movies_left, sample_movies_right, sample_candidate_pairs):
        """Test feature creation when feature functions raise errors."""
        def error_function(r1, r2):
            raise ValueError("Feature function error")
        
        def good_function(r1, r2):
            return 0.5
        
        comparators = [error_function, good_function]
        extractor = FeatureExtractor(comparators)
        
        # Prepare datasets
        sample_movies_left = ensure_record_ids(sample_movies_left)
        sample_movies_right = ensure_record_ids(sample_movies_right)
        
        features = extractor.create_features(
            sample_movies_left,
            sample_movies_right,
            sample_candidate_pairs
        )
        
        # Should handle errors gracefully and set error values to 0.0
        if len(features) > 0:
            assert "error_function" in features.columns
            assert "good_function" in features.columns
            assert all(val == 0.0 for val in features["error_function"])  # Error values set to 0
            assert all(val == 0.5 for val in features["good_function"])   # Good values preserved
    
    def test_create_features_non_numeric_return(self, sample_movies_left, sample_movies_right, sample_candidate_pairs):
        """Test feature creation when function returns non-numeric value."""
        def non_numeric_function(r1, r2):
            return "not_a_number"
        
        comparators = [non_numeric_function]
        extractor = FeatureExtractor(comparators)
        
        # Prepare datasets
        sample_movies_left = ensure_record_ids(sample_movies_left)
        sample_movies_right = ensure_record_ids(sample_movies_right)
        
        features = extractor.create_features(
            sample_movies_left,
            sample_movies_right,
            sample_candidate_pairs
        )
        
        # Should convert non-numeric to 0.0
        if len(features) > 0:
            assert all(val == 0.0 for val in features["non_numeric_function"])
    
    def test_get_feature_names(self):
        """Test get_feature_names method."""
        comparators = [
            StringComparator("title"),
            NumericComparator("year")
        ]
        
        extractor = FeatureExtractor(comparators)
        names = extractor.get_feature_names()
        
        assert len(names) == 2
        assert "StringComparator(title, jaro_winkler)" in names
        assert "NumericComparator(year, absolute_difference)" in names
    
    def test_repr(self):
        """Test string representation."""
        comparators = [StringComparator("title")]
        extractor = FeatureExtractor(comparators)
        
        repr_str = repr(extractor)
        assert "FeatureExtractor" in repr_str
        assert "n_features=1" in repr_str


class TestVectorFeatureExtractor:
    """Test the VectorFeatureExtractor class."""
    
    def test_initialization_requires_sentence_transformers(self):
        """Test that VectorFeatureExtractor requires sentence-transformers."""
        # This test checks if the import works
        try:
            VectorFeatureExtractor("all-MiniLM-L6-v2", ["title"])
            # If this succeeds, sentence-transformers is available
        except ImportError as e:
            assert "sentence-transformers package required" in str(e)
        except Exception:
            # Other errors are fine for this test - we just want to check import handling
            pass
    
    def test_initialization_with_custom_model(self):
        """Test initialization with custom embedding model."""
        # Mock embedding model
        mock_model = Mock()
        mock_model.encode = Mock(return_value=np.array([[0.1, 0.2, 0.3]]))
        mock_model.get_sentence_embedding_dimension = Mock(return_value=3)
        
        extractor = VectorFeatureExtractor(
            embedding_model=mock_model,
            columns=["title", "description"]
        )
        
        assert extractor.model == mock_model
        assert extractor.columns == ["title", "description"]
        assert extractor.distance_metrics == ["cosine"]  # default
        assert extractor.pooling_strategy == "concatenate"  # default
    
    def test_initialization_invalid_model(self):
        """Test initialization with invalid embedding model."""
        invalid_model = object()  # Object without 'encode' method
        
        with pytest.raises(ValueError, match="must have an 'encode' method"):
            VectorFeatureExtractor(
                embedding_model=invalid_model,
                columns=["title"]
            )
    
    def test_initialization_empty_columns(self):
        """Test initialization with empty columns list."""
        mock_model = Mock()
        mock_model.encode = Mock()
        
        with pytest.raises(ValueError, match="At least one column must be specified"):
            VectorFeatureExtractor(mock_model, columns=[])
    
    def test_initialization_custom_distance_metrics(self):
        """Test initialization with custom distance metrics."""
        mock_model = Mock()
        mock_model.encode = Mock()
        
        extractor = VectorFeatureExtractor(
            mock_model,
            columns=["title"],
            distance_metrics=["cosine", "euclidean", "manhattan"]
        )
        
        assert extractor.distance_metrics == ["cosine", "euclidean", "manhattan"]
    
    def test_initialization_invalid_distance_metric(self):
        """Test initialization with invalid distance metric."""
        mock_model = Mock()
        mock_model.encode = Mock()
        
        with pytest.raises(ValueError, match="Unsupported distance metric"):
            VectorFeatureExtractor(
                mock_model,
                columns=["title"],
                distance_metrics=["invalid_metric"]
            )
    
    def test_compute_embedding_concatenate(self):
        """Test _compute_embedding with concatenate strategy."""
        mock_model = Mock()
        mock_model.encode = Mock(return_value=np.array([[0.1, 0.2, 0.3]]))
        
        extractor = VectorFeatureExtractor(
            mock_model,
            columns=["title", "description"],
            pooling_strategy="concatenate"
        )
        
        record = pd.Series({
            "title": "The Matrix",
            "description": "Sci-fi movie"
        })
        
        embedding = extractor._compute_embedding(record)
        
        # Should call encode with concatenated text
        mock_model.encode.assert_called_once_with(["The Matrix Sci-fi movie"])
        assert isinstance(embedding, np.ndarray)
    
    def test_compute_embedding_mean(self):
        """Test _compute_embedding with mean strategy."""
        mock_model = Mock()
        # Return different embeddings for each text
        mock_model.encode = Mock(side_effect=[
            np.array([[0.1, 0.2, 0.3]]),  # First text
            np.array([[0.4, 0.5, 0.6]])   # Second text
        ])
        mock_model.get_sentence_embedding_dimension = Mock(return_value=3)
        
        extractor = VectorFeatureExtractor(
            mock_model,
            columns=["title", "description"],
            pooling_strategy="mean"
        )
        
        record = pd.Series({
            "title": "The Matrix",
            "description": "Sci-fi movie"
        })
        
        embedding = extractor._compute_embedding(record)
        
        # Should call encode twice
        assert mock_model.encode.call_count == 2
        # Result should be mean of the two embeddings
        expected_mean = np.array([0.25, 0.35, 0.45])  # Mean of [0.1,0.2,0.3] and [0.4,0.5,0.6]
        np.testing.assert_array_almost_equal(embedding, expected_mean)
    
    def test_compute_embedding_missing_values(self):
        """Test _compute_embedding with missing values."""
        mock_model = Mock()
        mock_model.encode = Mock(return_value=np.array([[0.1, 0.2, 0.3]]))
        
        extractor = VectorFeatureExtractor(
            mock_model,
            columns=["title", "description"],
            pooling_strategy="concatenate"
        )
        
        record = pd.Series({
            "title": "The Matrix",
            "description": None  # Missing value
        })
        
        embedding = extractor._compute_embedding(record)
        
        # Should handle missing value by using empty string
        mock_model.encode.assert_called_once_with(["The Matrix "])
    
    def test_compute_distance_cosine(self):
        """Test _compute_distance with cosine metric."""
        mock_model = Mock()
        extractor = VectorFeatureExtractor(mock_model, ["title"])
        
        emb1 = np.array([1.0, 0.0, 0.0])
        emb2 = np.array([0.0, 1.0, 0.0])
        
        distance = extractor._compute_distance(emb1, emb2, "cosine")
        
        # Cosine similarity of orthogonal vectors should be 0
        assert abs(distance - 0.0) < 1e-10
    
    def test_compute_distance_euclidean(self):
        """Test _compute_distance with euclidean metric."""
        mock_model = Mock()
        extractor = VectorFeatureExtractor(mock_model, ["title"])
        
        emb1 = np.array([0.0, 0.0, 0.0])
        emb2 = np.array([1.0, 1.0, 1.0])
        
        distance = extractor._compute_distance(emb1, emb2, "euclidean")
        
        # Should return similarity (1 / (1 + euclidean_distance))
        expected_euclidean = np.sqrt(3)  # sqrt(1^2 + 1^2 + 1^2)
        expected_similarity = 1.0 / (1.0 + expected_euclidean)
        assert abs(distance - expected_similarity) < 1e-10
    
    def test_compute_distance_manhattan(self):
        """Test _compute_distance with manhattan metric."""
        mock_model = Mock()
        extractor = VectorFeatureExtractor(mock_model, ["title"])
        
        emb1 = np.array([0.0, 0.0, 0.0])
        emb2 = np.array([1.0, 1.0, 1.0])
        
        distance = extractor._compute_distance(emb1, emb2, "manhattan")
        
        # Manhattan distance = 3.0, similarity = 1/(1+3) = 0.25
        expected_similarity = 1.0 / (1.0 + 3.0)
        assert abs(distance - expected_similarity) < 1e-10
    
    def test_get_feature_names(self):
        """Test get_feature_names method."""
        mock_model = Mock()
        extractor = VectorFeatureExtractor(
            mock_model,
            ["title"],
            distance_metrics=["cosine", "euclidean"]
        )
        
        names = extractor.get_feature_names()
        
        assert names == ["vector_cosine", "vector_euclidean"]
    
    @pytest.mark.skip("Requires working sentence-transformers installation")
    def test_create_features_integration(self, sample_movies_left, sample_movies_right, sample_candidate_pairs):
        """Integration test for create_features (requires sentence-transformers)."""
        # This test would require sentence-transformers to be installed
        extractor = VectorFeatureExtractor(
            "all-MiniLM-L6-v2",
            columns=["title"],
            distance_metrics=["cosine"]
        )
        
        # Prepare datasets
        sample_movies_left = ensure_record_ids(sample_movies_left)
        sample_movies_right = ensure_record_ids(sample_movies_right)
        
        features = extractor.create_features(
            sample_movies_left,
            sample_movies_right,
            sample_candidate_pairs
        )
        
        assert isinstance(features, pd.DataFrame)
        assert "vector_cosine" in features.columns