"""Tests for entity matching comparators."""

import pandas as pd
import pytest
import numpy as np
from datetime import datetime

from PyDI.entitymatching.comparators import (
    StringComparator,
    NumericComparator, 
    DateComparator
)


class TestStringComparator:
    """Test the StringComparator class."""
    
    def test_initialization_default(self):
        """Test StringComparator initialization with defaults."""
        comparator = StringComparator("title")
        
        assert comparator.column == "title"
        assert comparator.similarity_function == "jaro_winkler"
        assert comparator.preprocess is None
        assert "StringComparator" in comparator.name
        assert "title" in comparator.name
        assert "jaro_winkler" in comparator.name
    
    def test_initialization_custom_similarity(self):
        """Test StringComparator initialization with custom similarity function."""
        try:
            comparator = StringComparator("title", similarity_function="jaccard")
            assert comparator.similarity_function == "jaccard"
        except ValueError:
            # If jaccard is not available, that's okay
            pytest.skip("jaccard similarity function not available")
    
    def test_initialization_with_preprocessing(self):
        """Test StringComparator initialization with preprocessing function."""
        def preprocess(text):
            return text.lower().strip()
        
        comparator = StringComparator("title", preprocess=preprocess)
        assert comparator.preprocess == preprocess
    
    def test_compare_exact_match(self):
        """Test comparison with exact string match."""
        comparator = StringComparator("title")
        
        record1 = pd.Series({"title": "The Matrix", "year": 1999})
        record2 = pd.Series({"title": "The Matrix", "year": 2000})
        
        similarity = comparator.compare(record1, record2)
        assert isinstance(similarity, float)
        assert similarity >= 0.9  # Should be very high for exact match
    
    def test_compare_no_match(self):
        """Test comparison with completely different strings."""
        comparator = StringComparator("title")
        
        record1 = pd.Series({"title": "AAAA", "year": 1999})
        record2 = pd.Series({"title": "ZZZZ", "year": 2010})
        
        similarity = comparator.compare(record1, record2)
        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0
        assert similarity < 0.8  # Should be lower for completely different strings
    
    def test_compare_with_preprocessing(self):
        """Test comparison with preprocessing."""
        def normalize(text):
            return text.lower().replace(" ", "")
        
        comparator = StringComparator("title", preprocess=normalize)
        
        record1 = pd.Series({"title": "The Matrix"})
        record2 = pd.Series({"title": "THE MATRIX"})
        
        similarity = comparator.compare(record1, record2)
        assert similarity >= 0.9  # Should be high after normalization
    
    def test_compare_missing_values(self):
        """Test comparison with missing values."""
        comparator = StringComparator("title")
        
        # One missing value
        record1 = pd.Series({"title": "The Matrix"})
        record2 = pd.Series({"title": None})
        
        similarity = comparator.compare(record1, record2)
        assert similarity == 0.0
        
        # Both missing values
        record1 = pd.Series({"title": None})
        record2 = pd.Series({"title": None})
        
        similarity = comparator.compare(record1, record2)
        assert similarity == 0.0
    
    def test_compare_missing_column(self):
        """Test comparison when column is missing from records."""
        comparator = StringComparator("nonexistent_column")
        
        record1 = pd.Series({"title": "The Matrix"})
        record2 = pd.Series({"title": "Inception"})
        
        # Should handle gracefully and return 0.0
        similarity = comparator.compare(record1, record2)
        assert similarity == 0.0
    
    def test_invalid_similarity_function(self):
        """Test initialization with invalid similarity function."""
        with pytest.raises(ValueError, match="Recommended"):
            StringComparator("title", similarity_function="nonexistent_function")


class TestNumericComparator:
    """Test the NumericComparator class."""
    
    def test_initialization_default(self):
        """Test NumericComparator initialization with defaults."""
        comparator = NumericComparator("year")
        
        assert comparator.column == "year"
        assert comparator.method == "absolute_difference"
        assert comparator.max_difference is None
        assert "NumericComparator" in comparator.name
        assert "year" in comparator.name
        assert "absolute_difference" in comparator.name
    
    def test_initialization_custom_method(self):
        """Test initialization with custom method."""
        comparator = NumericComparator("rating", method="relative_difference")
        assert comparator.method == "relative_difference"
    
    def test_initialization_with_max_difference(self):
        """Test initialization with max_difference parameter."""
        comparator = NumericComparator("year", max_difference=5.0)
        assert comparator.max_difference == 5.0
    
    def test_initialization_invalid_method(self):
        """Test initialization with invalid method."""
        with pytest.raises(ValueError, match="Unknown method"):
            NumericComparator("year", method="invalid_method")
    
    def test_compare_absolute_difference_exact_match(self):
        """Test absolute difference with exact match."""
        comparator = NumericComparator("year", method="absolute_difference")
        
        record1 = pd.Series({"year": 2010})
        record2 = pd.Series({"year": 2010})
        
        similarity = comparator.compare(record1, record2)
        assert similarity == 1.0
    
    def test_compare_absolute_difference_with_max(self):
        """Test absolute difference with max_difference."""
        comparator = NumericComparator("year", method="absolute_difference", max_difference=10.0)
        
        # Small difference
        record1 = pd.Series({"year": 2010})
        record2 = pd.Series({"year": 2015})
        
        similarity = comparator.compare(record1, record2)
        assert 0.0 <= similarity <= 1.0
        assert similarity == 0.5  # (10 - 5) / 10
        
        # Large difference
        record1 = pd.Series({"year": 2010})
        record2 = pd.Series({"year": 2025})
        
        similarity = comparator.compare(record1, record2)
        assert similarity == 0.0  # Difference > max_difference
    
    def test_compare_absolute_difference_without_max(self):
        """Test absolute difference without max_difference."""
        comparator = NumericComparator("year", method="absolute_difference")
        
        record1 = pd.Series({"year": 2010})
        record2 = pd.Series({"year": 2015})
        
        similarity = comparator.compare(record1, record2)
        assert 0.0 <= similarity <= 1.0
        expected = 1.0 / (1.0 + 5)  # 1 / (1 + diff)
        assert abs(similarity - expected) < 0.01
    
    def test_compare_relative_difference(self):
        """Test relative difference method."""
        comparator = NumericComparator("rating", method="relative_difference")
        
        # Similar values
        record1 = pd.Series({"rating": 8.0})
        record2 = pd.Series({"rating": 8.4})
        
        similarity = comparator.compare(record1, record2)
        assert 0.0 <= similarity <= 1.0
        
        # Both zero
        record1 = pd.Series({"rating": 0.0})
        record2 = pd.Series({"rating": 0.0})
        
        similarity = comparator.compare(record1, record2)
        assert similarity == 1.0
    
    def test_compare_within_range(self):
        """Test within_range method."""
        comparator = NumericComparator("year", method="within_range", max_difference=5.0)
        
        # Within range
        record1 = pd.Series({"year": 2010})
        record2 = pd.Series({"year": 2013})
        
        similarity = comparator.compare(record1, record2)
        assert similarity == 1.0
        
        # Outside range
        record1 = pd.Series({"year": 2010})
        record2 = pd.Series({"year": 2020})
        
        similarity = comparator.compare(record1, record2)
        assert similarity == 0.0
    
    def test_within_range_requires_max_difference(self):
        """Test that within_range method requires max_difference."""
        comparator = NumericComparator("year", method="within_range")
        
        record1 = pd.Series({"year": 2010})
        record2 = pd.Series({"year": 2013})
        
        # Should handle error gracefully and return 0.0
        similarity = comparator.compare(record1, record2)
        assert similarity == 0.0
    
    def test_compare_missing_values(self):
        """Test comparison with missing values."""
        comparator = NumericComparator("year")
        
        record1 = pd.Series({"year": 2010})
        record2 = pd.Series({"year": None})
        
        similarity = comparator.compare(record1, record2)
        assert similarity == 0.0
        
        # Both missing
        record1 = pd.Series({"year": None})
        record2 = pd.Series({"year": None})
        
        similarity = comparator.compare(record1, record2)
        assert similarity == 0.0
    
    def test_compare_missing_column(self):
        """Test comparison when column is missing."""
        comparator = NumericComparator("nonexistent_column")
        
        record1 = pd.Series({"year": 2010})
        record2 = pd.Series({"year": 2015})
        
        similarity = comparator.compare(record1, record2)
        assert similarity == 0.0


class TestDateComparator:
    """Test the DateComparator class."""
    
    def test_initialization_default(self):
        """Test DateComparator initialization with defaults."""
        comparator = DateComparator("date")
        
        assert comparator.column == "date"
        assert comparator.max_days_difference is None
        assert "DateComparator" in comparator.name
        assert "date" in comparator.name
    
    def test_initialization_with_max_days(self):
        """Test initialization with max_days_difference."""
        comparator = DateComparator("date", max_days_difference=30)
        assert comparator.max_days_difference == 30
    
    def test_compare_same_date(self):
        """Test comparison with same date."""
        comparator = DateComparator("date")
        
        record1 = pd.Series({"date": "2010-01-01"})
        record2 = pd.Series({"date": "2010-01-01"})
        
        similarity = comparator.compare(record1, record2)
        assert similarity == 1.0
    
    def test_compare_different_dates_with_max_days(self):
        """Test comparison with different dates and max_days_difference."""
        comparator = DateComparator("date", max_days_difference=10)
        
        # Small difference
        record1 = pd.Series({"date": "2010-01-01"})
        record2 = pd.Series({"date": "2010-01-06"})  # 5 days difference
        
        similarity = comparator.compare(record1, record2)
        assert 0.0 <= similarity <= 1.0
        expected = 1.0 - (5 / 10)  # 0.5
        assert abs(similarity - expected) < 0.01
        
        # Large difference
        record1 = pd.Series({"date": "2010-01-01"})
        record2 = pd.Series({"date": "2010-01-20"})  # 19 days difference
        
        similarity = comparator.compare(record1, record2)
        assert similarity == 0.0
    
    def test_compare_different_dates_without_max_days(self):
        """Test comparison without max_days_difference."""
        comparator = DateComparator("date")
        
        record1 = pd.Series({"date": "2010-01-01"})
        record2 = pd.Series({"date": "2010-01-06"})  # 5 days difference
        
        similarity = comparator.compare(record1, record2)
        assert 0.0 <= similarity <= 1.0
        expected = 1.0 / (1.0 + 5)  # Inverse similarity
        assert abs(similarity - expected) < 0.01
    
    def test_compare_different_formats(self):
        """Test comparison with different date formats."""
        comparator = DateComparator("date")
        
        record1 = pd.Series({"date": "2010-01-01"})
        record2 = pd.Series({"date": "01/01/2010"})
        
        # Should parse both and find they're the same
        similarity = comparator.compare(record1, record2)
        assert similarity >= 0.9  # Should be high (parsing might introduce small differences)
    
    def test_compare_datetime_objects(self):
        """Test comparison with datetime objects."""
        comparator = DateComparator("date")
        
        date1 = datetime(2010, 1, 1)
        date2 = datetime(2010, 1, 1)
        
        record1 = pd.Series({"date": date1})
        record2 = pd.Series({"date": date2})
        
        similarity = comparator.compare(record1, record2)
        assert similarity == 1.0
    
    def test_compare_missing_values(self):
        """Test comparison with missing values."""
        comparator = DateComparator("date")
        
        record1 = pd.Series({"date": "2010-01-01"})
        record2 = pd.Series({"date": None})
        
        similarity = comparator.compare(record1, record2)
        assert similarity == 0.0
        
        # Both missing
        record1 = pd.Series({"date": None})
        record2 = pd.Series({"date": None})
        
        similarity = comparator.compare(record1, record2)
        assert similarity == 0.0
    
    def test_compare_invalid_dates(self):
        """Test comparison with invalid date strings."""
        comparator = DateComparator("date")
        
        record1 = pd.Series({"date": "2010-01-01"})
        record2 = pd.Series({"date": "invalid_date"})
        
        # Should handle gracefully and return 0.0
        similarity = comparator.compare(record1, record2)
        assert similarity == 0.0
    
    def test_compare_missing_column(self):
        """Test comparison when column is missing."""
        comparator = DateComparator("nonexistent_column")
        
        record1 = pd.Series({"date": "2010-01-01"})
        record2 = pd.Series({"date": "2010-01-02"})
        
        similarity = comparator.compare(record1, record2)
        assert similarity == 0.0


class TestComparatorIntegration:
    """Integration tests for comparators working together."""
    
    def test_multiple_comparators_same_records(self, sample_movies_left, sample_movies_right):
        """Test multiple comparators on the same records."""
        string_comp = StringComparator("title")
        numeric_comp = NumericComparator("year")
        
        # Get first record from each dataset
        record1 = sample_movies_left.iloc[0]
        record2 = sample_movies_right.iloc[0]  # Should be same movie (Biutiful)
        
        title_sim = string_comp.compare(record1, record2)
        year_sim = numeric_comp.compare(record1, record2)
        
        # Both should be high similarity since it's the same movie
        assert title_sim >= 0.9
        assert year_sim >= 0.9
    
    def test_comparators_with_preprocessing(self, preprocessing_function):
        """Test comparators with shared preprocessing function."""
        comparator = StringComparator("title", preprocess=preprocessing_function)
        
        record1 = pd.Series({"title": "The Matrix"})
        record2 = pd.Series({"title": "THE MATRIX"})
        
        similarity = comparator.compare(record1, record2)
        assert similarity >= 0.9  # Should be high after preprocessing
    
    def test_comparator_error_handling(self):
        """Test that comparators handle errors gracefully."""
        comparators = [
            StringComparator("title"),
            NumericComparator("year"),
            DateComparator("date")
        ]
        
        # Record with problematic data
        record1 = pd.Series({"title": "Movie", "year": "not_a_number", "date": "invalid_date"})
        record2 = pd.Series({"title": "Film", "year": 2020, "date": "2020-01-01"})
        
        for comp in comparators:
            # Should not crash, should return valid similarity scores
            similarity = comp.compare(record1, record2)
            assert isinstance(similarity, (int, float))
            assert 0.0 <= similarity <= 1.0