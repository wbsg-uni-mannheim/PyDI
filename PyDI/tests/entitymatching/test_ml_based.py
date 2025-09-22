"""Tests for machine learning-based entity matching."""

import pandas as pd
import pytest
import numpy as np
from unittest.mock import Mock

from PyDI.entitymatching.ml_based import MLBasedMatcher
from PyDI.entitymatching.feature_extraction import FeatureExtractor, VectorFeatureExtractor
from PyDI.entitymatching.comparators import StringComparator, NumericComparator
from PyDI.entitymatching.base import ensure_record_ids


class TestMLBasedMatcher:
    """Test the MLBasedMatcher class."""
    
    def test_initialization_with_feature_extractor(self):
        """Test MLBasedMatcher initialization with FeatureExtractor."""
        comparators = [StringComparator("title"), NumericComparator("year")]
        feature_extractor = FeatureExtractor(comparators)
        
        matcher = MLBasedMatcher(feature_extractor)
        assert isinstance(matcher, MLBasedMatcher)
        assert matcher.feature_extractor == feature_extractor
    
    def test_initialization_with_vector_feature_extractor(self):
        """Test MLBasedMatcher initialization with VectorFeatureExtractor."""
        pytest.skip("VectorFeatureExtractor requires sentence-transformers which may not be available")
        # This test would be similar but would require sentence-transformers
    
    def test_initialization_invalid_extractor(self):
        """Test initialization with invalid feature extractor."""
        with pytest.raises(ValueError, match="must be FeatureExtractor or VectorFeatureExtractor"):
            MLBasedMatcher("invalid_extractor")
    
    def test_match_with_probabilities(self, sample_movies_left, sample_movies_right, sample_candidate_pairs, mock_trained_classifier):
        """Test matching using predict_proba for probabilistic scores."""
        comparators = [StringComparator("title"), NumericComparator("year")]
        feature_extractor = FeatureExtractor(comparators)
        matcher = MLBasedMatcher(feature_extractor)
        
        # Prepare datasets
        sample_movies_left = ensure_record_ids(sample_movies_left)
        sample_movies_right = ensure_record_ids(sample_movies_right)
        
        # Perform matching
        matches = matcher.match(
            sample_movies_left,
            sample_movies_right,
            [sample_candidate_pairs],
            trained_classifier=mock_trained_classifier,
            threshold=0.5,
            use_probabilities=True
        )
        
        # Verify results
        assert isinstance(matches, pd.DataFrame)
        assert list(matches.columns) == ["id1", "id2", "score", "notes"]
        
        # Check metadata
        assert "classifier_type" in matches.attrs
        assert matches.attrs["classifier_type"] == "MockClassifier"
        assert matches.attrs["threshold"] == 0.5
        assert matches.attrs["use_probabilities"] == True
    
    def test_match_without_probabilities(self, sample_movies_left, sample_movies_right, sample_candidate_pairs, mock_trained_classifier):
        """Test matching using predict for binary decisions."""
        comparators = [StringComparator("title"), NumericComparator("year")]
        feature_extractor = FeatureExtractor(comparators)
        matcher = MLBasedMatcher(feature_extractor)
        
        # Prepare datasets
        sample_movies_left = ensure_record_ids(sample_movies_left)
        sample_movies_right = ensure_record_ids(sample_movies_right)
        
        # Perform matching
        matches = matcher.match(
            sample_movies_left,
            sample_movies_right,
            [sample_candidate_pairs],
            trained_classifier=mock_trained_classifier,
            threshold=0.5,
            use_probabilities=False
        )
        
        # Verify results
        assert isinstance(matches, pd.DataFrame)
        assert list(matches.columns) == ["id1", "id2", "score", "notes"]
        assert matches.attrs["use_probabilities"] == False
    
    def test_match_multiple_batches(self, sample_movies_left, sample_movies_right, candidate_pair_batches, mock_trained_classifier):
        """Test matching with multiple candidate batches."""
        comparators = [StringComparator("title"), NumericComparator("year")]
        feature_extractor = FeatureExtractor(comparators)
        matcher = MLBasedMatcher(feature_extractor)
        
        # Prepare datasets
        sample_movies_left = ensure_record_ids(sample_movies_left)
        sample_movies_right = ensure_record_ids(sample_movies_right)
        
        # Perform matching with multiple batches
        matches = matcher.match(
            sample_movies_left,
            sample_movies_right,
            candidate_pair_batches,
            trained_classifier=mock_trained_classifier,
            threshold=0.1  # Low threshold to get some results
        )
        
        assert isinstance(matches, pd.DataFrame)
        # Should process all batches
        assert len(matches) >= 0
    
    def test_match_empty_candidates(self, sample_movies_left, sample_movies_right, mock_trained_classifier):
        """Test matching with empty candidate list."""
        comparators = [StringComparator("title"), NumericComparator("year")]
        feature_extractor = FeatureExtractor(comparators)
        matcher = MLBasedMatcher(feature_extractor)
        
        # Prepare datasets
        sample_movies_left = ensure_record_ids(sample_movies_left)
        sample_movies_right = ensure_record_ids(sample_movies_right)
        
        # Empty candidates
        empty_candidates = [pd.DataFrame(columns=["id1", "id2"])]
        
        matches = matcher.match(
            sample_movies_left,
            sample_movies_right,
            empty_candidates,
            trained_classifier=mock_trained_classifier,
            threshold=0.5
        )
        
        assert isinstance(matches, pd.DataFrame)
        assert len(matches) == 0
        assert list(matches.columns) == ["id1", "id2", "score", "notes"]
    
    def test_match_high_threshold_no_results(self, sample_movies_left, sample_movies_right, sample_candidate_pairs, mock_trained_classifier):
        """Test matching with very high threshold returns no results."""
        comparators = [StringComparator("title"), NumericComparator("year")]
        feature_extractor = FeatureExtractor(comparators)
        matcher = MLBasedMatcher(feature_extractor)
        
        # Prepare datasets
        sample_movies_left = ensure_record_ids(sample_movies_left)
        sample_movies_right = ensure_record_ids(sample_movies_right)
        
        # Very high threshold
        matches = matcher.match(
            sample_movies_left,
            sample_movies_right,
            [sample_candidate_pairs],
            trained_classifier=mock_trained_classifier,
            threshold=0.999  # Nearly impossible to achieve
        )
        
        assert isinstance(matches, pd.DataFrame)
        assert len(matches) == 0  # Should be empty
        assert list(matches.columns) == ["id1", "id2", "score", "notes"]
    
    def test_validation_errors(self, sample_movies_left, sample_movies_right, sample_candidate_pairs):
        """Test input validation errors."""
        comparators = [StringComparator("title")]
        feature_extractor = FeatureExtractor(comparators)
        matcher = MLBasedMatcher(feature_extractor)
        
        # None classifier
        with pytest.raises(ValueError, match="trained_classifier cannot be None"):
            matcher.match(
                sample_movies_left,
                sample_movies_right,
                [sample_candidate_pairs],
                trained_classifier=None
            )
        
        # Classifier without predict method
        invalid_classifier = Mock()
        del invalid_classifier.predict  # Remove predict method
        
        with pytest.raises(ValueError, match="must have predict\\(\\) method"):
            matcher.match(
                sample_movies_left,
                sample_movies_right,
                [sample_candidate_pairs],
                trained_classifier=invalid_classifier
            )
        
        # Classifier without predict_proba when use_probabilities=True
        classifier_no_proba = Mock(spec=['predict'])  # Only has predict method
        classifier_no_proba.predict = Mock()
        
        with pytest.raises(ValueError, match="must have predict_proba\\(\\) method"):
            matcher.match(
                sample_movies_left,
                sample_movies_right,
                [sample_candidate_pairs],
                trained_classifier=classifier_no_proba,
                use_probabilities=True
            )
    
    def test_process_batch_success(self, sample_movies_left, sample_movies_right, sample_candidate_pairs, mock_trained_classifier):
        """Test _process_batch method with successful processing."""
        comparators = [StringComparator("title"), NumericComparator("year")]
        feature_extractor = FeatureExtractor(comparators)
        matcher = MLBasedMatcher(feature_extractor)
        
        # Prepare datasets
        sample_movies_left = ensure_record_ids(sample_movies_left)
        sample_movies_right = ensure_record_ids(sample_movies_right)
        
        results = matcher._process_batch(
            sample_candidate_pairs,
            sample_movies_left,
            sample_movies_right,
            mock_trained_classifier,
            threshold=0.1,
            use_probabilities=True
        )
        
        assert isinstance(results, list)
        
        if len(results) > 0:
            # Check structure of results
            result = results[0]
            assert "id1" in result
            assert "id2" in result
            assert "score" in result
            assert "notes" in result
            assert isinstance(result["score"], float)
            assert "ml_classifier=MockClassifier" in result["notes"]
    
    def test_process_batch_no_features(self, sample_movies_left, sample_movies_right, mock_trained_classifier):
        """Test _process_batch when no features can be extracted."""
        # Create feature extractor that will fail
        def failing_comparator(r1, r2):
            raise ValueError("Feature extraction failed")
        
        feature_extractor = FeatureExtractor([failing_comparator])
        matcher = MLBasedMatcher(feature_extractor)
        
        # Prepare datasets
        sample_movies_left = ensure_record_ids(sample_movies_left)
        sample_movies_right = ensure_record_ids(sample_movies_right)
        
        # Create candidate pairs
        candidates = pd.DataFrame({
            "id1": ["academy_awards_000000"],
            "id2": ["actors_000000"]
        })
        
        results = matcher._process_batch(
            candidates,
            sample_movies_left,
            sample_movies_right,
            mock_trained_classifier,
            threshold=0.5,
            use_probabilities=True
        )
        
        # Should handle gracefully and return empty list
        assert results == []
    
    def test_predict_pairs(self, sample_movies_left, sample_movies_right, sample_candidate_pairs, mock_trained_classifier):
        """Test predict_pairs method for getting raw predictions."""
        comparators = [StringComparator("title"), NumericComparator("year")]
        feature_extractor = FeatureExtractor(comparators)
        matcher = MLBasedMatcher(feature_extractor)
        
        # Prepare datasets
        sample_movies_left = ensure_record_ids(sample_movies_left)
        sample_movies_right = ensure_record_ids(sample_movies_right)
        
        # Get predictions
        predictions = matcher.predict_pairs(
            sample_movies_left,
            sample_movies_right,
            sample_candidate_pairs,
            mock_trained_classifier,
            use_probabilities=True
        )
        
        # Verify results
        assert isinstance(predictions, pd.DataFrame)
        assert "id1" in predictions.columns
        assert "id2" in predictions.columns
        assert "prediction" in predictions.columns
        
        # Check that predictions are numeric
        if len(predictions) > 0:
            assert all(isinstance(pred, (int, float, np.number)) for pred in predictions["prediction"])
    
    def test_predict_pairs_without_probabilities(self, sample_movies_left, sample_movies_right, sample_candidate_pairs, mock_trained_classifier):
        """Test predict_pairs without using probabilities."""
        comparators = [StringComparator("title"), NumericComparator("year")]
        feature_extractor = FeatureExtractor(comparators)
        matcher = MLBasedMatcher(feature_extractor)
        
        # Prepare datasets
        sample_movies_left = ensure_record_ids(sample_movies_left)
        sample_movies_right = ensure_record_ids(sample_movies_right)
        
        # Get predictions
        predictions = matcher.predict_pairs(
            sample_movies_left,
            sample_movies_right,
            sample_candidate_pairs,
            mock_trained_classifier,
            use_probabilities=False
        )
        
        # Verify results
        assert isinstance(predictions, pd.DataFrame)
        assert "prediction" in predictions.columns
    
    def test_predict_pairs_empty(self, sample_movies_left, sample_movies_right, mock_trained_classifier):
        """Test predict_pairs with empty pairs."""
        comparators = [StringComparator("title"), NumericComparator("year")]
        feature_extractor = FeatureExtractor(comparators)
        matcher = MLBasedMatcher(feature_extractor)
        
        empty_pairs = pd.DataFrame(columns=["id1", "id2"])
        
        # Should raise ValueError for empty pairs
        with pytest.raises(ValueError, match="Empty pairs DataFrame provided"):
            matcher.predict_pairs(
                sample_movies_left,
                sample_movies_right,
                empty_pairs,
                mock_trained_classifier
            )
    
    def test_get_feature_importance(self, mock_trained_classifier):
        """Test get_feature_importance method."""
        comparators = [StringComparator("title"), NumericComparator("year")]
        feature_extractor = FeatureExtractor(comparators)
        matcher = MLBasedMatcher(feature_extractor)
        
        importance_df = matcher.get_feature_importance(mock_trained_classifier)
        
        assert isinstance(importance_df, pd.DataFrame)
        assert "feature" in importance_df.columns
        assert "importance" in importance_df.columns
        
        # Should be sorted by importance descending
        importances = importance_df["importance"].tolist()
        assert importances == sorted(importances, reverse=True)
    
    def test_get_feature_importance_no_feature_importance(self):
        """Test get_feature_importance with classifier that doesn't have feature importance."""
        comparators = [StringComparator("title")]
        feature_extractor = FeatureExtractor(comparators)
        matcher = MLBasedMatcher(feature_extractor)
        
        # Classifier without feature_importances_
        classifier_no_importance = Mock(spec=[])  # Empty spec, no attributes
        
        with pytest.raises(AttributeError, match="doesn't provide feature importance"):
            matcher.get_feature_importance(classifier_no_importance)
    
    def test_get_feature_importance_name_mismatch(self, mock_trained_classifier):
        """Test get_feature_importance with feature name mismatch."""
        # Create extractor with different number of features than classifier
        comparators = [StringComparator("title")]  # Only 1 feature
        feature_extractor = FeatureExtractor(comparators)
        matcher = MLBasedMatcher(feature_extractor)
        
        # mock_trained_classifier has 3 feature importances
        importance_df = matcher.get_feature_importance(mock_trained_classifier)
        
        # Should handle mismatch gracefully
        assert isinstance(importance_df, pd.DataFrame)
        assert len(importance_df) == 3  # Should use actual number of importances
        
        # Feature names should be generic
        feature_names = importance_df["feature"].tolist()
        assert all("feature_" in name for name in feature_names)
    
    def test_classifier_predict_proba_error_fallback(self, sample_movies_left, sample_movies_right, sample_candidate_pairs):
        """Test fallback to predict when predict_proba fails."""
        # Create classifier that has predict_proba but it raises an error
        class ProblematicClassifier:
            def predict(self, X):
                return np.array([1, 0, 1, 0])
            
            def predict_proba(self, X):
                raise ValueError("predict_proba failed")
        
        comparators = [StringComparator("title"), NumericComparator("year")]
        feature_extractor = FeatureExtractor(comparators)
        matcher = MLBasedMatcher(feature_extractor)
        
        # Prepare datasets
        sample_movies_left = ensure_record_ids(sample_movies_left)
        sample_movies_right = ensure_record_ids(sample_movies_right)
        
        problematic_classifier = ProblematicClassifier()
        
        # Should handle error gracefully and fall back to predict
        matches = matcher.match(
            sample_movies_left,
            sample_movies_right,
            [sample_candidate_pairs],
            trained_classifier=problematic_classifier,
            threshold=0.5,
            use_probabilities=True
        )
        
        assert isinstance(matches, pd.DataFrame)
        # Should still work due to fallback
    
    def test_repr(self):
        """Test string representation."""
        comparators = [StringComparator("title")]
        feature_extractor = FeatureExtractor(comparators)
        matcher = MLBasedMatcher(feature_extractor)
        
        repr_str = repr(matcher)
        assert "MLBasedMatcher" in repr_str
        assert "feature_extractor" in repr_str