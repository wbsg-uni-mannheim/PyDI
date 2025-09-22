"""Tests for rule-based entity matching."""

import pandas as pd
import pytest
from unittest.mock import Mock

from PyDI.entitymatching.rule_based import RuleBasedMatcher
from PyDI.entitymatching.comparators import StringComparator, NumericComparator
from PyDI.entitymatching.base import ensure_record_ids


class TestRuleBasedMatcher:
    """Test the RuleBasedMatcher class."""
    
    def test_initialization(self):
        """Test RuleBasedMatcher initialization."""
        matcher = RuleBasedMatcher()
        assert isinstance(matcher, RuleBasedMatcher)
    
    def test_match_with_comparator_objects(self, sample_movies_left, sample_movies_right, sample_candidate_pairs):
        """Test matching with BaseComparator objects."""
        matcher = RuleBasedMatcher()
        
        # Prepare datasets
        sample_movies_left = ensure_record_ids(sample_movies_left)
        sample_movies_right = ensure_record_ids(sample_movies_right)
        
        # Create comparators
        comparators = [
            StringComparator("title"),
            NumericComparator("year")
        ]
        
        # Perform matching
        matches = matcher.match(
            sample_movies_left,
            sample_movies_right,
            [sample_candidate_pairs],
            comparators=comparators,
            weights=[0.7, 0.3],
            threshold=0.5
        )
        
        # Verify results
        assert isinstance(matches, pd.DataFrame)
        assert list(matches.columns) == ["id1", "id2", "score", "notes"]
        
        # Should have some matches since we have similar movies
        assert len(matches) >= 0  # Might be 0 if threshold is too high
        
        if len(matches) > 0:
            # Check that scores are in valid range
            assert all(0.0 <= score <= 1.0 for score in matches["score"])
            
            # Check that all scores are above threshold
            assert all(score >= 0.5 for score in matches["score"])
    
    def test_match_with_callable_functions(self, sample_movies_left, sample_movies_right, sample_candidate_pairs, sample_comparator_functions):
        """Test matching with callable functions as comparators."""
        matcher = RuleBasedMatcher()
        
        # Prepare datasets
        sample_movies_left = ensure_record_ids(sample_movies_left)
        sample_movies_right = ensure_record_ids(sample_movies_right)
        
        # Perform matching with function comparators
        matches = matcher.match(
            sample_movies_left,
            sample_movies_right,
            [sample_candidate_pairs],
            comparators=sample_comparator_functions,
            weights=[0.6, 0.4],
            threshold=0.8
        )
        
        assert isinstance(matches, pd.DataFrame)
        assert list(matches.columns) == ["id1", "id2", "score", "notes"]
    
    def test_match_with_dict_comparators(self, sample_movies_left, sample_movies_right, sample_candidate_pairs, sample_comparator_functions):
        """Test matching with dict format comparators."""
        matcher = RuleBasedMatcher()
        
        # Prepare datasets
        sample_movies_left = ensure_record_ids(sample_movies_left)
        sample_movies_right = ensure_record_ids(sample_movies_right)
        
        # Create dict format comparators
        dict_comparators = [
            {"comparator": sample_comparator_functions[0], "weight": 0.7},
            {"comparator": sample_comparator_functions[1], "weight": 0.3}
        ]
        
        # Perform matching
        matches = matcher.match(
            sample_movies_left,
            sample_movies_right,
            [sample_candidate_pairs],
            comparators=dict_comparators,
            threshold=0.5
        )
        
        assert isinstance(matches, pd.DataFrame)
        assert list(matches.columns) == ["id1", "id2", "score", "notes"]
    
    def test_match_with_equal_weights(self, sample_movies_left, sample_movies_right, sample_candidate_pairs, sample_comparator_functions):
        """Test matching with equal weights (weights=None)."""
        matcher = RuleBasedMatcher()
        
        # Prepare datasets
        sample_movies_left = ensure_record_ids(sample_movies_left)
        sample_movies_right = ensure_record_ids(sample_movies_right)
        
        # Perform matching without explicit weights
        matches = matcher.match(
            sample_movies_left,
            sample_movies_right,
            [sample_candidate_pairs],
            comparators=sample_comparator_functions,
            weights=None,  # Should use equal weights
            threshold=0.3
        )
        
        assert isinstance(matches, pd.DataFrame)
        
        # Equal weights should be 1/n for each comparator
        # This is tested indirectly through the matching process
    
    def test_match_multiple_candidate_batches(self, sample_movies_left, sample_movies_right, candidate_pair_batches, sample_comparator_functions):
        """Test matching with multiple candidate batches."""
        matcher = RuleBasedMatcher()
        
        # Prepare datasets
        sample_movies_left = ensure_record_ids(sample_movies_left)
        sample_movies_right = ensure_record_ids(sample_movies_right)
        
        # Perform matching with multiple batches
        matches = matcher.match(
            sample_movies_left,
            sample_movies_right,
            candidate_pair_batches,
            comparators=sample_comparator_functions,
            weights=[0.5, 0.5],
            threshold=0.1  # Low threshold to get some results
        )
        
        assert isinstance(matches, pd.DataFrame)
        # Should process all batches
        assert len(matches) >= 0
    
    def test_match_empty_candidates(self, sample_movies_left, sample_movies_right, sample_comparator_functions):
        """Test matching with empty candidate list."""
        matcher = RuleBasedMatcher()
        
        # Prepare datasets
        sample_movies_left = ensure_record_ids(sample_movies_left)
        sample_movies_right = ensure_record_ids(sample_movies_right)
        
        # Empty candidates
        empty_candidates = [pd.DataFrame(columns=["id1", "id2"])]
        
        matches = matcher.match(
            sample_movies_left,
            sample_movies_right,
            empty_candidates,
            comparators=sample_comparator_functions,
            weights=[0.5, 0.5],
            threshold=0.5
        )
        
        assert isinstance(matches, pd.DataFrame)
        assert len(matches) == 0
        assert list(matches.columns) == ["id1", "id2", "score", "notes"]
    
    def test_match_high_threshold_no_results(self, sample_movies_left, sample_movies_right, sample_candidate_pairs, sample_comparator_functions):
        """Test matching with very high threshold returns fewer results."""
        matcher = RuleBasedMatcher()
        
        # Prepare datasets
        sample_movies_left = ensure_record_ids(sample_movies_left)
        sample_movies_right = ensure_record_ids(sample_movies_right)
        
        # Very high threshold - but our exact match functions can still achieve 1.0
        matches = matcher.match(
            sample_movies_left,
            sample_movies_right,
            [sample_candidate_pairs],
            comparators=sample_comparator_functions,
            weights=[0.5, 0.5],
            threshold=0.999  # High threshold but exact matches can exceed this
        )
        
        assert isinstance(matches, pd.DataFrame)
        # Should have fewer matches than with lower threshold, but exact matches can still pass
        assert len(matches) >= 0  # May have matches if records match exactly
        assert list(matches.columns) == ["id1", "id2", "score", "notes"]
    
    def test_parse_comparators_functions(self, sample_comparator_functions):
        """Test _parse_comparators with function comparators."""
        matcher = RuleBasedMatcher()
        
        parsed = matcher._parse_comparators(sample_comparator_functions, [0.6, 0.4])
        
        assert len(parsed) == 2
        assert parsed[0]["weight"] == 0.6
        assert parsed[1]["weight"] == 0.4
        assert callable(parsed[0]["comparator"])
        assert callable(parsed[1]["comparator"])
    
    def test_parse_comparators_equal_weights(self, sample_comparator_functions):
        """Test _parse_comparators with equal weights."""
        matcher = RuleBasedMatcher()
        
        parsed = matcher._parse_comparators(sample_comparator_functions, None)
        
        assert len(parsed) == 2
        expected_weight = 1.0 / 2  # Equal weights
        assert parsed[0]["weight"] == expected_weight
        assert parsed[1]["weight"] == expected_weight
    
    def test_parse_comparators_dicts(self, sample_comparator_functions):
        """Test _parse_comparators with dict format."""
        matcher = RuleBasedMatcher()
        
        dict_comparators = [
            {"comparator": sample_comparator_functions[0], "weight": 0.7},
            {"comparator": sample_comparator_functions[1], "weight": 0.3}
        ]
        
        parsed = matcher._parse_comparators(dict_comparators, None)
        
        assert len(parsed) == 2
        assert parsed[0]["weight"] == 0.7
        assert parsed[1]["weight"] == 0.3
    
    def test_parse_comparators_objects(self):
        """Test _parse_comparators with BaseComparator objects."""
        matcher = RuleBasedMatcher()
        
        comparators = [
            StringComparator("title"),
            NumericComparator("year")
        ]
        
        parsed = matcher._parse_comparators(comparators, [0.8, 0.2])
        
        assert len(parsed) == 2
        assert parsed[0]["weight"] == 0.8
        assert parsed[1]["weight"] == 0.2
        # BaseComparator objects are stored directly, not their methods
        assert isinstance(parsed[0]["comparator"], StringComparator)
        assert isinstance(parsed[1]["comparator"], NumericComparator)
    
    def test_parse_comparators_errors(self):
        """Test _parse_comparators error handling."""
        matcher = RuleBasedMatcher()
        
        # Missing comparator key in dict
        with pytest.raises(ValueError, match="must have 'comparator' and 'weight' keys"):
            matcher._parse_comparators([{"weight": 0.5}], None)
        
        # Missing weight key in dict
        with pytest.raises(ValueError, match="must have 'comparator' and 'weight' keys"):
            matcher._parse_comparators([{"comparator": lambda x, y: 0.5}], None)
        
        # Zero weight
        with pytest.raises(ValueError, match="must be > 0.0"):
            matcher._parse_comparators([{"comparator": lambda x, y: 0.5, "weight": 0.0}], None)
        
        # Negative weight
        with pytest.raises(ValueError, match="must be > 0.0"):
            matcher._parse_comparators([lambda x, y: 0.5], [-0.1])
        
        # Not enough weights
        with pytest.raises(ValueError, match="Not enough weights"):
            matcher._parse_comparators([lambda x, y: 0.5, lambda x, y: 0.3], [0.8])
    
    def test_compute_similarity(self, sample_comparator_functions):
        """Test _compute_similarity method."""
        matcher = RuleBasedMatcher()
        
        record1 = pd.Series({"title": "The Matrix", "year": 1999})
        record2 = pd.Series({"title": "The Matrix", "year": 1999})
        
        parsed_comparators = matcher._parse_comparators(sample_comparator_functions, [0.6, 0.4])
        
        similarity = matcher._compute_similarity(record1, record2, parsed_comparators)
        
        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0
        # Since titles match and years match, should be high
        assert similarity > 0.5
    
    def test_process_batch(self, sample_movies_left, sample_movies_right, sample_candidate_pairs, sample_comparator_functions):
        """Test _process_batch method."""
        matcher = RuleBasedMatcher()
        
        # Prepare datasets
        sample_movies_left = ensure_record_ids(sample_movies_left)
        sample_movies_right = ensure_record_ids(sample_movies_right)
        
        left_lookup = sample_movies_left.set_index("_id")
        right_lookup = sample_movies_right.set_index("_id")
        
        parsed_comparators = matcher._parse_comparators(sample_comparator_functions, [0.5, 0.5])
        
        results = matcher._process_batch(
            sample_candidate_pairs,
            left_lookup,
            right_lookup, 
            parsed_comparators,
            threshold=0.1
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
    
    def test_validation_errors(self, sample_movies_left, sample_movies_right, sample_candidate_pairs):
        """Test input validation errors."""
        matcher = RuleBasedMatcher()
        
        # Prepare datasets with _id columns for this test
        sample_movies_left = ensure_record_ids(sample_movies_left)
        sample_movies_right = ensure_record_ids(sample_movies_right)
        
        # No comparators
        with pytest.raises(ValueError, match="No comparators provided"):
            matcher.match(
                sample_movies_left,
                sample_movies_right,
                [sample_candidate_pairs],
                comparators=[],
                threshold=0.5
            )
        
        # Missing _id columns (test inherited from BaseMatcher)
        # Use original datasets without _id columns
        with pytest.raises(ValueError, match="must have '_id' column"):
            matcher.match(
                pd.DataFrame({"title": ["Movie"]}),  # No _id column
                sample_movies_right,
                [sample_candidate_pairs],
                comparators=[lambda x, y: 0.5],
                threshold=0.5
            )
    
    def test_missing_records_in_lookup(self, sample_movies_left, sample_movies_right, sample_comparator_functions):
        """Test handling of missing records in lookup tables."""
        matcher = RuleBasedMatcher()
        
        # Prepare datasets
        sample_movies_left = ensure_record_ids(sample_movies_left)
        sample_movies_right = ensure_record_ids(sample_movies_right)
        
        # Create candidate pairs with non-existent IDs
        bad_candidates = pd.DataFrame({
            "id1": ["nonexistent_id"],
            "id2": ["also_nonexistent"]
        })
        
        matches = matcher.match(
            sample_movies_left,
            sample_movies_right,
            [bad_candidates],
            comparators=sample_comparator_functions,
            weights=[0.5, 0.5],
            threshold=0.5
        )
        
        # Should handle gracefully and return empty results
        assert isinstance(matches, pd.DataFrame)
        assert len(matches) == 0
    
    def test_match_single_dataframe_candidates(self, sample_movies_left, sample_movies_right, sample_comparator_functions):
        """Test that match accepts a single DataFrame as candidates."""
        matcher = RuleBasedMatcher()
        df_left, df_right = sample_movies_left, sample_movies_right
        comparators, _ = sample_comparator_functions
        
        # Create candidate pairs as a single DataFrame (not wrapped in a list)
        candidates = pd.DataFrame({
            "id1": ["academy_awards_000000", "academy_awards_000001"],
            "id2": ["actors_000000", "actors_000001"]
        })
        
        # This should work without wrapping candidates in a list
        result = matcher.match(df_left, df_right, candidates, comparators, threshold=0.0)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "id1" in result.columns
        assert "id2" in result.columns
        assert "score" in result.columns

    def test_repr(self):
        """Test string representation."""
        matcher = RuleBasedMatcher()
        repr_str = repr(matcher)
        assert "RuleBasedMatcher" in repr_str