"""Tests for entity matching evaluation."""

import pandas as pd
import pytest
import json
import numpy as np
from pathlib import Path

from PyDI.entitymatching.evaluation import EntityMatchingEvaluator


class TestEntityMatchingEvaluator:
    """Test the EntityMatchingEvaluator class."""
    
    def test_evaluate_basic(self, sample_correspondences, sample_test_pairs):
        """Test basic evaluation functionality."""
        results = EntityMatchingEvaluator.evaluate(
            sample_correspondences,
            sample_test_pairs
        )
        
        # Check basic structure
        assert isinstance(results, dict)
        
        # Check required metrics
        required_metrics = [
            "precision", "recall", "f1", "true_positives", 
            "false_positives", "false_negatives", "evaluation_timestamp"
        ]
        for metric in required_metrics:
            assert metric in results
        
        # Check metric ranges
        assert 0.0 <= results["precision"] <= 1.0
        assert 0.0 <= results["recall"] <= 1.0
        assert 0.0 <= results["f1"] <= 1.0
        
        # Check counts are non-negative integers
        assert results["true_positives"] >= 0
        assert results["false_positives"] >= 0
        assert results["false_negatives"] >= 0
    
    def test_evaluate_with_threshold(self, sample_correspondences, sample_test_pairs):
        """Test evaluation with similarity threshold."""
        results = EntityMatchingEvaluator.evaluate(
            sample_correspondences,
            sample_test_pairs,
            threshold=0.9
        )
        
        assert "threshold_used" in results
        assert results["threshold_used"] == 0.9
        assert "total_correspondences" in results
        assert "filtered_correspondences" in results
        
        # Filtered should be <= total
        assert results["filtered_correspondences"] <= results["total_correspondences"]
    
    def test_evaluate_with_candidate_pairs(self, sample_correspondences, sample_test_pairs, sample_candidate_pairs):
        """Test evaluation with candidate pairs for candidate recall."""
        results = EntityMatchingEvaluator.evaluate(
            sample_correspondences,
            sample_test_pairs,
            candidate_pairs=sample_candidate_pairs
        )
        
        assert "candidate_recall" in results
        assert "total_candidates" in results
        assert 0.0 <= results["candidate_recall"] <= 1.0
        assert results["total_candidates"] >= 0
    
    def test_evaluate_with_total_pairs(self, sample_correspondences, sample_test_pairs):
        """Test evaluation with total possible pairs for pair reduction."""
        total_pairs = 1000
        
        results = EntityMatchingEvaluator.evaluate(
            sample_correspondences,
            sample_test_pairs,
            total_possible_pairs=total_pairs
        )
        
        assert "pair_reduction" in results
        assert "total_possible_pairs" in results
        assert 0.0 <= results["pair_reduction"] <= 1.0
        assert results["total_possible_pairs"] == total_pairs
    
    def test_evaluate_with_output_directory(self, sample_correspondences, sample_test_pairs, temp_output_dir):
        """Test evaluation with output directory for file writing."""
        out_dir = str(temp_output_dir)
        
        results = EntityMatchingEvaluator.evaluate(
            sample_correspondences,
            sample_test_pairs,
            out_dir=out_dir
        )
        
        assert "output_files" in results
        assert isinstance(results["output_files"], list)
        
        # Check that files were created
        for file_path in results["output_files"]:
            assert Path(file_path).exists()
        
        # Check specific files
        summary_file = temp_output_dir / "evaluation_summary.json"
        detailed_file = temp_output_dir / "detailed_results.csv"
        
        assert summary_file.exists()
        assert detailed_file.exists()
        
        # Verify JSON content
        with open(summary_file) as f:
            json_data = json.load(f)
            assert "precision" in json_data
            assert "recall" in json_data
    
    def test_evaluate_empty_correspondences(self, sample_test_pairs):
        """Test evaluation with empty correspondence set."""
        empty_corr = pd.DataFrame(columns=["id1", "id2", "score", "notes"])
        
        results = EntityMatchingEvaluator.evaluate(empty_corr, sample_test_pairs)
        
        # With no correspondences, precision is undefined but should be handled
        assert results["precision"] >= 0.0
        assert results["recall"] == 0.0  # No true positives found
        assert results["true_positives"] == 0
        assert results["false_positives"] == 0
    
    def test_evaluate_empty_test_pairs(self, sample_correspondences):
        """Test evaluation with empty test pairs."""
        empty_test = pd.DataFrame(columns=["id1", "id2", "label"])
        
        with pytest.raises(ValueError, match="Empty test_pairs DataFrame"):
            EntityMatchingEvaluator.evaluate(sample_correspondences, empty_test)
    
    def test_evaluate_missing_columns(self):
        """Test evaluation with missing required columns."""
        # Correspondences missing score column
        bad_corr = pd.DataFrame({
            "id1": ["a"],
            "id2": ["b"],
            "notes": ["test"]
            # Missing "score" column
        })
        
        test_pairs = pd.DataFrame({
            "id1": ["a"],
            "id2": ["b"],
            "label": [1]
        })
        
        with pytest.raises(ValueError, match="missing required column"):
            EntityMatchingEvaluator.evaluate(bad_corr, test_pairs)
    
    def test_evaluate_test_pairs_without_labels(self, sample_correspondences):
        """Test evaluation with test pairs without label column (assumes all positive)."""
        test_pairs_no_labels = pd.DataFrame({
            "id1": ["academy_awards_1", "academy_awards_2"],
            "id2": ["actors_1", "actors_2"]
            # No label column - should assume all positive
        })
        
        results = EntityMatchingEvaluator.evaluate(sample_correspondences, test_pairs_no_labels)
        
        # Should work without labels
        assert isinstance(results, dict)
        assert "precision" in results
        assert "recall" in results
        assert results["accuracy"] is None  # No accuracy without negatives
    
    def test_evaluate_with_negative_labels(self, sample_correspondences):
        """Test evaluation with explicit negative labels."""
        test_pairs_with_negatives = pd.DataFrame({
            "id1": ["academy_awards_1", "academy_awards_2", "academy_awards_3", "academy_awards_4"],
            "id2": ["actors_1", "actors_2", "actors_3", "actors_4"],
            "label": [1, 1, 1, 0]  # Last one is negative
        })
        
        results = EntityMatchingEvaluator.evaluate(sample_correspondences, test_pairs_with_negatives)
        
        # Should compute accuracy when negatives are available
        assert results["accuracy"] is not None
        assert 0.0 <= results["accuracy"] <= 1.0
        assert results["true_negatives"] >= 0
    
    def test_create_cluster_consistency_report_basic(self, sample_correspondences):
        """Test basic cluster consistency analysis."""
        report = EntityMatchingEvaluator.create_cluster_consistency_report(
            sample_correspondences
        )
        
        assert isinstance(report, pd.DataFrame)
        
        # Check required columns
        expected_columns = [
            "cluster_id", "cluster_size", "total_edges", "expected_edges",
            "consistency_ratio", "is_consistent", "avg_similarity",
            "min_similarity", "max_similarity", "entities"
        ]
        for col in expected_columns:
            assert col in report.columns
        
        # Check data types and ranges
        if len(report) > 0:
            assert (report["cluster_size"] >= 1).all()
            assert (report["total_edges"] >= 0).all()
            assert (report["expected_edges"] >= 0).all()
            assert ((report["consistency_ratio"] >= 0.0) & (report["consistency_ratio"] <= 1.0)).all()
            assert all(isinstance(consistent, bool) for consistent in report["is_consistent"])
    
    def test_create_cluster_consistency_report_with_output(self, sample_correspondences, temp_output_dir):
        """Test cluster consistency report with output directory."""
        out_dir = str(temp_output_dir)
        
        report = EntityMatchingEvaluator.create_cluster_consistency_report(
            sample_correspondences,
            out_dir=out_dir
        )
        
        assert isinstance(report, pd.DataFrame)
        
        # Check that files were created
        csv_file = temp_output_dir / "cluster_consistency_report.csv"
        json_file = temp_output_dir / "cluster_analysis_summary.json"
        
        assert csv_file.exists()
        assert json_file.exists()
        
        # Verify JSON content
        with open(json_file) as f:
            json_data = json.load(f)
            assert "total_clusters" in json_data
            assert "consistent_clusters" in json_data
            assert "inconsistent_clusters" in json_data
    
    def test_create_cluster_consistency_report_empty(self):
        """Test cluster consistency report with empty correspondences."""
        empty_corr = pd.DataFrame(columns=["id1", "id2", "score", "notes"])
        
        with pytest.raises(ValueError, match="Empty correspondence set"):
            EntityMatchingEvaluator.create_cluster_consistency_report(empty_corr)
    
    def test_create_cluster_consistency_report_missing_columns(self):
        """Test cluster consistency report with missing columns."""
        bad_corr = pd.DataFrame({
            "id1": ["a"],
            "id2": ["b"]
            # Missing "score" column
        })
        
        with pytest.raises(ValueError, match="missing required column"):
            EntityMatchingEvaluator.create_cluster_consistency_report(bad_corr)
    
    def test_write_record_groups_by_consistency(self, sample_correspondences, temp_output_dir):
        """Test writing record groups organized by consistency."""
        output_file = temp_output_dir / "record_groups.json"
        
        result_path = EntityMatchingEvaluator.write_record_groups_by_consistency(
            str(output_file),
            sample_correspondences
        )
        
        assert result_path == str(output_file)
        assert output_file.exists()
        
        # Verify JSON structure
        with open(output_file) as f:
            data = json.load(f)
            
            assert "metadata" in data
            assert "consistent_clusters" in data
            assert "inconsistent_clusters" in data
            
            # Check metadata
            metadata = data["metadata"]
            assert "generated_at" in metadata
            assert "total_correspondences" in metadata
            assert "total_clusters" in metadata
    
    def test_write_record_groups_empty_correspondences(self, temp_output_dir):
        """Test writing record groups with empty correspondences."""
        output_file = temp_output_dir / "empty_groups.json"
        empty_corr = pd.DataFrame(columns=["id1", "id2", "score", "notes"])
        
        with pytest.raises(ValueError, match="Empty correspondence set"):
            EntityMatchingEvaluator.write_record_groups_by_consistency(
                str(output_file),
                empty_corr
            )
    
    def test_threshold_sweep(self, sample_correspondences, sample_test_pairs):
        """Test threshold sweep analysis."""
        thresholds = [0.0, 0.5, 0.8, 1.0]
        
        sweep_results = EntityMatchingEvaluator.threshold_sweep(
            sample_correspondences,
            sample_test_pairs,
            thresholds=thresholds
        )
        
        assert isinstance(sweep_results, pd.DataFrame)
        
        # Check required columns
        expected_columns = [
            "threshold", "precision", "recall", "f1", "true_positives",
            "false_positives", "false_negatives", "correspondences_count"
        ]
        for col in expected_columns:
            assert col in sweep_results.columns
        
        # Should have one row per threshold (minus any that failed)
        assert len(sweep_results) <= len(thresholds)
        
        # Check that thresholds are as expected
        if len(sweep_results) > 0:
            assert all(thresh in thresholds for thresh in sweep_results["threshold"])
    
    def test_threshold_sweep_default_thresholds(self, sample_correspondences, sample_test_pairs):
        """Test threshold sweep with default threshold range."""
        sweep_results = EntityMatchingEvaluator.threshold_sweep(
            sample_correspondences,
            sample_test_pairs
        )
        
        assert isinstance(sweep_results, pd.DataFrame)
        
        # Default should be 0.0 to 1.0 in 0.1 steps (11 thresholds)
        assert len(sweep_results) <= 11
    
    def test_threshold_sweep_with_output(self, sample_correspondences, sample_test_pairs, temp_output_dir):
        """Test threshold sweep with output directory."""
        out_dir = str(temp_output_dir)
        
        sweep_results = EntityMatchingEvaluator.threshold_sweep(
            sample_correspondences,
            sample_test_pairs,
            out_dir=out_dir
        )
        
        assert isinstance(sweep_results, pd.DataFrame)
        
        # Check that output file was created
        output_file = temp_output_dir / "threshold_sweep.csv"
        assert output_file.exists()
        
        # Verify CSV content
        loaded_results = pd.read_csv(output_file)
        assert len(loaded_results) == len(sweep_results)
    
    def test_create_cluster_size_distribution(self, sample_correspondences):
        """Test cluster size distribution analysis."""
        distribution = EntityMatchingEvaluator.create_cluster_size_distribution(
            sample_correspondences
        )
        
        assert isinstance(distribution, pd.DataFrame)
        
        # Check required columns
        expected_columns = ["cluster_size", "frequency", "percentage"]
        for col in expected_columns:
            assert col in distribution.columns
        
        if len(distribution) > 0:
            # Check data validity
            assert all(distribution["cluster_size"] >= 1)
            assert all(distribution["frequency"] >= 1)
            assert all(distribution["percentage"] >= 0.0)
            
            # Total percentage should sum to 100 (approximately)
            total_percentage = distribution["percentage"].sum()
            assert abs(total_percentage - 100.0) < 1e-10
    
    def test_create_cluster_size_distribution_with_output(self, sample_correspondences, temp_output_dir):
        """Test cluster size distribution with output directory."""
        out_dir = str(temp_output_dir)
        
        distribution = EntityMatchingEvaluator.create_cluster_size_distribution(
            sample_correspondences,
            out_dir=out_dir
        )
        
        assert isinstance(distribution, pd.DataFrame)
        
        # Check that output file was created
        output_file = temp_output_dir / "cluster_size_distribution.csv"
        assert output_file.exists()
        
        # Verify CSV content
        loaded_distribution = pd.read_csv(output_file)
        assert len(loaded_distribution) == len(distribution)
    
    def test_create_cluster_size_distribution_empty(self):
        """Test cluster size distribution with empty correspondences."""
        empty_corr = pd.DataFrame(columns=["id1", "id2", "score", "notes"])
        
        with pytest.raises(ValueError, match="Empty correspondence set"):
            EntityMatchingEvaluator.create_cluster_size_distribution(empty_corr)
    
    
    def test_perfect_precision_recall(self):
        """Test evaluation with perfect precision and recall."""
        # Create correspondences that exactly match test pairs
        perfect_corr = pd.DataFrame({
            "id1": ["academy_awards_1", "academy_awards_2"],
            "id2": ["actors_1", "actors_2"],
            "score": [1.0, 1.0],
            "notes": ["perfect", "perfect"]
        })
        
        perfect_test = pd.DataFrame({
            "id1": ["academy_awards_1", "academy_awards_2"],
            "id2": ["actors_1", "actors_2"],
            "label": [1, 1]
        })
        
        results = EntityMatchingEvaluator.evaluate(perfect_corr, perfect_test)
        
        # Should have perfect precision and recall
        assert results["precision"] == 1.0
        assert results["recall"] == 1.0
        assert results["f1"] == 1.0
        assert results["true_positives"] == 2
        assert results["false_positives"] == 0
        assert results["false_negatives"] == 0
    
    def test_zero_division_protection(self):
        """Test that zero division is handled gracefully."""
        # Create correspondences with no matches
        no_match_corr = pd.DataFrame({
            "id1": ["academy_awards_999"],
            "id2": ["actors_999"],
            "score": [0.9],
            "notes": ["no_match"]
        })
        
        test_pairs = pd.DataFrame({
            "id1": ["academy_awards_1"],
            "id2": ["actors_1"],
            "label": [1]
        })
        
        results = EntityMatchingEvaluator.evaluate(no_match_corr, test_pairs)
        
        # Should handle zero division gracefully
        assert 0.0 <= results["precision"] <= 1.0
        assert 0.0 <= results["recall"] <= 1.0
        assert 0.0 <= results["f1"] <= 1.0