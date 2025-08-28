"""Tests for schema mapping evaluation utilities."""

import pytest
import pandas as pd
import numpy as np

from PyDI.schemamatching.evaluation import SchemaMappingEvaluator


class TestSchemaMappingEvaluator:
    """Test the SchemaMappingEvaluator class."""
    
    def test_perfect_evaluation(self):
        """Test evaluation with perfect precision and recall."""
        # Perfect correspondences - all predictions are correct and complete
        predicted_mapping = pd.DataFrame({
            "source_dataset": ["movies", "movies"],
            "source_column": ["title", "year"],
            "target_dataset": ["films", "films"],
            "target_column": ["film_name", "release_year"],
            "score": [1.0, 0.9]
        })
        
        evaluation_set = pd.DataFrame({
            "source_dataset": ["movies", "movies"],
            "source_column": ["title", "year"],
            "target_dataset": ["films", "films"],
            "target_column": ["film_name", "release_year"],
            "label": [True, True]
        })
        
        result = SchemaMappingEvaluator.evaluate(predicted_mapping, evaluation_set)
        
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0
        assert result["correct"] == 2
        assert result["matched"] == 2
        assert result["correct_total"] == 2
        assert result["missing"] == 0
    
    def test_partial_precision_perfect_recall(self):
        """Test evaluation with partial precision but perfect recall."""
        # Extra incorrect prediction
        predicted_mapping = pd.DataFrame({
            "source_dataset": ["movies", "movies", "movies"],
            "source_column": ["title", "year", "genre"],
            "target_dataset": ["films", "films", "films"],
            "target_column": ["film_name", "release_year", "director"],  # genre->director is wrong
            "score": [1.0, 0.9, 0.8]
        })
        
        evaluation_set = pd.DataFrame({
            "source_dataset": ["movies", "movies", "movies"],
            "source_column": ["title", "year", "genre"],
            "target_dataset": ["films", "films", "films"],
            "target_column": ["film_name", "release_year", "category"],  # genre should match category
            "label": [True, True, True]
        })
        
        result = SchemaMappingEvaluator.evaluate(predicted_mapping, evaluation_set, complete=True)
        
        assert result["precision"] == 2/3  # 2 correct out of 3 predictions
        assert result["recall"] == 2/3     # 2 correct out of 3 total correct
        assert result["correct"] == 2
        assert result["matched"] == 3
        assert result["correct_total"] == 3
        assert result["missing"] == 1
    
    def test_perfect_precision_partial_recall(self):
        """Test evaluation with perfect precision but partial recall."""
        # Missing some correct correspondences
        predicted_mapping = pd.DataFrame({
            "source_dataset": ["movies"],
            "source_column": ["title"],
            "target_dataset": ["films"],
            "target_column": ["film_name"],
            "score": [1.0]
        })
        
        evaluation_set = pd.DataFrame({
            "source_dataset": ["movies", "movies"],
            "source_column": ["title", "year"],
            "target_dataset": ["films", "films"],
            "target_column": ["film_name", "release_year"],
            "label": [True, True]
        })
        
        result = SchemaMappingEvaluator.evaluate(predicted_mapping, evaluation_set)
        
        assert result["precision"] == 1.0  # All predictions correct
        assert result["recall"] == 0.5     # Only found 1 out of 2 correct
        assert result["f1"] == 2/3         # Harmonic mean of 1.0 and 0.5
        assert result["correct"] == 1
        assert result["matched"] == 1
        assert result["correct_total"] == 2
        assert result["missing"] == 1
    
    def test_evaluation_with_threshold(self):
        """Test evaluation with score threshold filtering."""
        predicted_mapping = pd.DataFrame({
            "source_dataset": ["movies", "movies"],
            "source_column": ["title", "year"],
            "target_dataset": ["films", "films"],
            "target_column": ["film_name", "release_year"],
            "score": [0.9, 0.3]  # Second one below threshold
        })
        
        evaluation_set = pd.DataFrame({
            "source_dataset": ["movies", "movies"],
            "source_column": ["title", "year"],
            "target_dataset": ["films", "films"],
            "target_column": ["film_name", "release_year"],
            "label": [True, True]
        })
        
        result = SchemaMappingEvaluator.evaluate(predicted_mapping, evaluation_set, threshold=0.5)
        
        # Only the high-scoring correspondence should be evaluated
        assert result["precision"] == 1.0  # 1 correct out of 1 evaluated
        assert result["recall"] == 0.5     # 1 correct out of 2 total
        assert result["correct"] == 1
        assert result["matched"] == 1
        assert result["missing"] == 1
    
    def test_evaluation_with_explicit_negatives(self):
        """Test evaluation with explicit negative examples."""
        predicted_mapping = pd.DataFrame({
            "source_dataset": ["movies", "movies"],
            "source_column": ["title", "year"],
            "target_dataset": ["films", "films"],
            "target_column": ["film_name", "director"],  # year->director is incorrect
            "score": [1.0, 0.8]
        })
        
        evaluation_set = pd.DataFrame({
            "source_dataset": ["movies", "movies", "movies"],
            "source_column": ["title", "year", "year"],
            "target_dataset": ["films", "films", "films"],
            "target_column": ["film_name", "release_year", "director"],
            "label": [True, True, False]  # year->director is explicitly negative
        })
        
        result = SchemaMappingEvaluator.evaluate(predicted_mapping, evaluation_set)
        
        assert result["precision"] == 0.5  # 1 correct out of 2 predictions
        assert result["recall"] == 0.5     # 1 correct out of 2 positives
        assert result["correct"] == 1
        assert result["matched"] == 2
        assert result["correct_total"] == 2
    
    def test_evaluation_without_label_column(self):
        """Test evaluation when evaluation set has no label column."""
        predicted_mapping = pd.DataFrame({
            "source_dataset": ["movies"],
            "source_column": ["title"],
            "target_dataset": ["films"],
            "target_column": ["film_name"],
            "score": [1.0]
        })
        
        # No label column - all are assumed positive
        evaluation_set = pd.DataFrame({
            "source_dataset": ["movies", "movies"],
            "source_column": ["title", "year"],
            "target_dataset": ["films", "films"],
            "target_column": ["film_name", "release_year"]
        })
        
        result = SchemaMappingEvaluator.evaluate(predicted_mapping, evaluation_set)
        
        assert result["precision"] == 1.0  # All predictions are in evaluation set
        assert result["recall"] == 0.5     # Found 1 out of 2 positives
        assert result["correct"] == 1
        assert result["matched"] == 1
        assert result["correct_total"] == 2
    
    def test_evaluation_with_symmetry(self):
        """Test that evaluation handles symmetric correspondences."""
        # Predicted in one direction
        predicted_mapping = pd.DataFrame({
            "source_dataset": ["movies"],
            "source_column": ["title"],
            "target_dataset": ["films"],
            "target_column": ["film_name"],
            "score": [1.0]
        })
        
        # Evaluation set has reverse direction
        evaluation_set = pd.DataFrame({
            "source_dataset": ["films"],
            "source_column": ["film_name"],
            "target_dataset": ["movies"],
            "target_column": ["title"],
            "label": [True]
        })
        
        result = SchemaMappingEvaluator.evaluate(predicted_mapping, evaluation_set)
        
        # Should still find the match due to symmetry
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["correct"] == 1
    
    def test_evaluation_empty_predictions(self, sample_evaluation_mapping):
        """Test evaluation with empty predictions."""
        empty_mapping = pd.DataFrame(columns=[
            "source_dataset", "source_column", 
            "target_dataset", "target_column", "score"
        ])
        
        result = SchemaMappingEvaluator.evaluate(empty_mapping, sample_evaluation_mapping)
        
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0
        assert result["correct"] == 0
        assert result["matched"] == 0
        assert result["correct_total"] == 4  # All in evaluation set are positive
        assert result["missing"] == 4
    
    def test_evaluation_empty_evaluation_set(self):
        """Test evaluation with empty evaluation set."""
        predicted_mapping = pd.DataFrame({
            "source_dataset": ["movies"],
            "source_column": ["title"],
            "target_dataset": ["films"],
            "target_column": ["film_name"],
            "score": [1.0]
        })
        
        empty_evaluation = pd.DataFrame(columns=[
            "source_dataset", "source_column", 
            "target_dataset", "target_column", "label"
        ])
        
        result = SchemaMappingEvaluator.evaluate(predicted_mapping, empty_evaluation)
        
        # No positive examples, so recall is undefined (0.0)
        # No matched predictions since nothing in evaluation set
        assert result["precision"] == 0.0  # No evaluation available
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0
        assert result["correct"] == 0
        assert result["matched"] == 0
        assert result["correct_total"] == 0
    
    def test_evaluation_complete_mode(self):
        """Test evaluation in complete mode."""
        predicted_mapping = pd.DataFrame({
            "source_dataset": ["movies", "movies"],
            "source_column": ["title", "genre"],
            "target_dataset": ["films", "films"],
            "target_column": ["film_name", "director"],  # genre->director is wrong
            "score": [1.0, 0.8]
        })
        
        # Evaluation set only has positive example for title
        evaluation_set = pd.DataFrame({
            "source_dataset": ["movies"],
            "source_column": ["title"],
            "target_dataset": ["films"],
            "target_column": ["film_name"],
            "label": [True]
        })
        
        # In complete mode, anything not in evaluation set is considered negative
        result = SchemaMappingEvaluator.evaluate(predicted_mapping, evaluation_set, complete=True)
        
        assert result["precision"] == 0.5  # 1 correct out of 2 predictions
        assert result["recall"] == 1.0     # Found the 1 positive example
        assert result["correct"] == 1
        assert result["matched"] == 2
        assert result["correct_total"] == 1
    
    def test_evaluation_custom_label_column(self):
        """Test evaluation with custom label column name."""
        predicted_mapping = pd.DataFrame({
            "source_dataset": ["movies"],
            "source_column": ["title"],
            "target_dataset": ["films"],
            "target_column": ["film_name"],
            "score": [1.0]
        })
        
        evaluation_set = pd.DataFrame({
            "source_dataset": ["movies"],
            "source_column": ["title"],
            "target_dataset": ["films"],
            "target_column": ["film_name"],
            "is_correct": [True]  # Custom column name
        })
        
        result = SchemaMappingEvaluator.evaluate(
            predicted_mapping, evaluation_set, label_column="is_correct"
        )
        
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
    
    def test_sweep_thresholds(self, sample_evaluation_mapping):
        """Test threshold sweeping functionality."""
        predicted_mapping = pd.DataFrame({
            "source_dataset": ["movies", "movies", "movies", "movies"],
            "source_column": ["movie_id", "title", "year", "genre"],
            "target_dataset": ["films", "films", "films", "films"],
            "target_column": ["film_id", "film_name", "release_year", "category"],
            "score": [0.9, 0.8, 0.6, 0.3]
        })
        
        thresholds = [0.2, 0.5, 0.7, 0.9]
        result = SchemaMappingEvaluator.sweep_thresholds(
            predicted_mapping, sample_evaluation_mapping, thresholds=thresholds
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(thresholds)
        
        # Check required columns
        expected_columns = ["threshold", "precision", "recall", "f1", "correct", "matched", "correct_total", "missing"]
        for col in expected_columns:
            assert col in result.columns
        
        # Thresholds should match input
        assert result["threshold"].tolist() == thresholds
        
        # As threshold increases, matched should decrease or stay same
        matched_counts = result["matched"].tolist()
        for i in range(1, len(matched_counts)):
            assert matched_counts[i] <= matched_counts[i-1]
    
    def test_sweep_thresholds_empty_thresholds(self, sample_evaluation_mapping):
        """Test threshold sweeping with empty threshold list."""
        predicted_mapping = pd.DataFrame({
            "source_dataset": ["movies"],
            "source_column": ["title"],
            "target_dataset": ["films"],
            "target_column": ["film_name"],
            "score": [1.0]
        })
        
        result = SchemaMappingEvaluator.sweep_thresholds(
            predicted_mapping, sample_evaluation_mapping, thresholds=[]
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
    
    def test_edge_case_zero_division(self):
        """Test that zero division is handled gracefully."""
        # No predictions
        empty_mapping = pd.DataFrame(columns=[
            "source_dataset", "source_column", 
            "target_dataset", "target_column", "score"
        ])
        
        # Empty evaluation set
        empty_evaluation = pd.DataFrame(columns=[
            "source_dataset", "source_column", 
            "target_dataset", "target_column", "label"
        ])
        
        result = SchemaMappingEvaluator.evaluate(empty_mapping, empty_evaluation)
        
        # Should handle gracefully without division by zero
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0
    
    def test_evaluation_result_data_types(self, sample_evaluation_mapping):
        """Test that evaluation results have correct data types."""
        predicted_mapping = pd.DataFrame({
            "source_dataset": ["movies"],
            "source_column": ["title"],
            "target_dataset": ["films"],
            "target_column": ["film_name"],
            "score": [1.0]
        })
        
        result = SchemaMappingEvaluator.evaluate(predicted_mapping, sample_evaluation_mapping)
        
        # Check data types
        assert isinstance(result["precision"], float)
        assert isinstance(result["recall"], float)
        assert isinstance(result["f1"], float)
        assert isinstance(result["correct"], int)
        assert isinstance(result["matched"], int)
        assert isinstance(result["correct_total"], int)
        assert isinstance(result["missing"], int)
        
        # Check value ranges
        assert 0 <= result["precision"] <= 1
        assert 0 <= result["recall"] <= 1
        assert 0 <= result["f1"] <= 1
    
    @pytest.mark.parametrize("complete", [True, False])
    def test_evaluation_modes(self, complete):
        """Test evaluation in both complete and incomplete modes."""
        predicted_mapping = pd.DataFrame({
            "source_dataset": ["movies"],
            "source_column": ["title"],
            "target_dataset": ["films"],
            "target_column": ["film_name"],
            "score": [1.0]
        })
        
        evaluation_set = pd.DataFrame({
            "source_dataset": ["movies"],
            "source_column": ["title"],
            "target_dataset": ["films"],
            "target_column": ["film_name"],
            "label": [True]
        })
        
        result = SchemaMappingEvaluator.evaluate(
            predicted_mapping, evaluation_set, complete=complete
        )
        
        # Both modes should work without errors
        assert isinstance(result, dict)
        assert "precision" in result
        assert "recall" in result
        assert "f1" in result