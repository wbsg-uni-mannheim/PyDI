"""Tests for autorules module."""

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from PyDI.informationextraction.autorules import RuleDiscovery, discover_fields
from PyDI.informationextraction.rules import built_in_rules


class TestRuleDiscovery:
    """Test cases for RuleDiscovery class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample DataFrame with mixed structured data."""
        data = {
            'description': [
                'Contact us at john@example.com or call (555) 123-4567 for $99.99 deals',
                'Visit https://example.com - Special offer: $49.95 until 2024-12-31',
                'Our office: 123 Main St, ZIP 12345. Email: info@company.com',
                'Product XYZ123 costs €75.50 (available in red, blue colors)',
                'No structured data in this text entry',
                'Call +1-800-555-0123 or visit www.test.org for more info'
            ]
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    def test_initialization(self, temp_output_dir):
        """Test RuleDiscovery initialization."""
        discovery = RuleDiscovery(out_dir=temp_output_dir, debug=True)
        
        assert discovery.out_dir == Path(temp_output_dir)
        assert discovery.debug is True
        assert discovery.default_source is None
        
        # Check that output directory is created
        assert discovery.run_dir.exists()
    
    def test_build_all_rules(self, temp_output_dir):
        """Test building rules from built-in patterns."""
        discovery = RuleDiscovery(out_dir=temp_output_dir)
        
        # Test with specific categories
        rules = discovery.build_all_rules(
            'description', 
            categories=['contact', 'money'],
            include_postprocess=True
        )
        
        # Should have rules from both categories
        assert len(rules) > 0
        
        # Check that rules have expected structure
        for field_name, rule_config in rules.items():
            assert 'source_column' in rule_config
            assert 'pattern' in rule_config
            assert 'flags' in rule_config
            assert 'group' in rule_config
            assert rule_config['source_column'] == 'description'
        
        # Check category namespacing
        contact_fields = [f for f in rules.keys() if f.startswith('contact__')]
        money_fields = [f for f in rules.keys() if f.startswith('money__')]
        assert len(contact_fields) > 0
        assert len(money_fields) > 0
    
    def test_build_all_rules_all_categories(self, temp_output_dir):
        """Test building rules with all categories."""
        discovery = RuleDiscovery(out_dir=temp_output_dir)
        
        rules = discovery.build_all_rules('description', categories=None)
        
        # Should have rules from all categories
        expected_categories = set(built_in_rules.keys())
        found_categories = set()
        
        for field_name in rules.keys():
            category = field_name.split('__')[0]
            found_categories.add(category)
        
        assert found_categories == expected_categories
    
    def test_compute_coverage(self, sample_data, temp_output_dir):
        """Test coverage computation."""
        discovery = RuleDiscovery(out_dir=temp_output_dir)
        
        # Add some test columns with different coverage levels
        test_df = sample_data.copy()
        test_df['high_coverage'] = ['val1', 'val2', 'val3', 'val4', 'val5', 'val6']  # 100%
        test_df['medium_coverage'] = ['val1', 'val2', 'val3', None, None, None]  # 50%
        test_df['low_coverage'] = ['val1', None, None, None, None, None]  # ~17%
        test_df['no_coverage'] = [None, None, None, None, None, None]  # 0%
        
        coverage = discovery._compute_coverage(
            test_df, 
            ['high_coverage', 'medium_coverage', 'low_coverage', 'no_coverage']
        )
        
        assert coverage['high_coverage'] == 1.0
        assert coverage['medium_coverage'] == 0.5
        assert abs(coverage['low_coverage'] - (1/6)) < 0.01
        assert coverage['no_coverage'] == 0.0
    
    def test_filter_fields_by_coverage(self, temp_output_dir):
        """Test field filtering by coverage criteria."""
        discovery = RuleDiscovery(out_dir=temp_output_dir)
        
        coverage = {
            'high_field': 0.8,
            'medium_field': 0.4,
            'low_field': 0.1,
            'zero_field': 0.0
        }
        
        # Test coverage threshold filtering
        selected = discovery._filter_fields_by_coverage(
            coverage, 
            coverage_threshold=0.3
        )
        assert set(selected) == {'high_field', 'medium_field'}
        
        # Test top_k limiting
        selected = discovery._filter_fields_by_coverage(
            coverage, 
            coverage_threshold=0.0,
            top_k=2
        )
        assert len(selected) == 2
        assert selected[0] == 'high_field'  # Should be sorted by coverage desc
        assert selected[1] == 'medium_field'
    
    def test_extract_and_select_basic(self, sample_data, temp_output_dir):
        """Test basic extract_and_select functionality."""
        discovery = RuleDiscovery(out_dir=temp_output_dir, debug=True)
        
        result = discovery.extract_and_select(
            sample_data,
            source_column='description',
            categories=['contact', 'money'],
            coverage_threshold=0.25,
            include_original=False
        )
        
        # Should return DataFrame with extracted fields
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_data)
        
        # Should have some extracted fields (emails, phones, prices should be found)
        assert len(result.columns) > 0
        
        # Original columns should not be included
        assert 'description' not in result.columns
    
    def test_extract_and_select_with_original(self, sample_data, temp_output_dir):
        """Test extract_and_select with include_original=True."""
        discovery = RuleDiscovery(out_dir=temp_output_dir)
        
        result = discovery.extract_and_select(
            sample_data,
            source_column='description',
            categories=['contact'],
            coverage_threshold=0.1,
            include_original=True
        )
        
        # Should include original columns
        assert 'description' in result.columns
        
        # Should have extracted fields as well
        assert len(result.columns) > 1
    
    def test_extract_and_select_with_metadata(self, sample_data, temp_output_dir):
        """Test extract_and_select with return_meta=True."""
        discovery = RuleDiscovery(out_dir=temp_output_dir)
        
        result, metadata = discovery.extract_and_select(
            sample_data,
            source_column='description',
            categories=['contact', 'money'],
            coverage_threshold=0.1,
            return_meta=True
        )
        
        # Check result DataFrame
        assert isinstance(result, pd.DataFrame)
        
        # Check metadata structure
        assert isinstance(metadata, dict)
        assert 'coverage' in metadata
        assert 'selected_fields' in metadata
        assert 'total_fields_evaluated' in metadata
        assert 'source_column' in metadata
        assert 'categories_used' in metadata
        
        assert metadata['source_column'] == 'description'
        assert set(metadata['categories_used']) == {'contact', 'money'}
    
    def test_extract_and_select_empty_result(self, temp_output_dir):
        """Test extract_and_select with high threshold returning empty result."""
        discovery = RuleDiscovery(out_dir=temp_output_dir)
        
        # Data with no structured information
        df = pd.DataFrame({'text': ['just plain text', 'no patterns here']})
        
        result = discovery.extract_and_select(
            df,
            source_column='text',
            coverage_threshold=0.9,  # Very high threshold
            include_original=False
        )
        
        # Should return empty DataFrame (no columns selected)
        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) == 0 or result.empty
    
    def test_top_k_limiting(self, sample_data, temp_output_dir):
        """Test top_k field limiting."""
        discovery = RuleDiscovery(out_dir=temp_output_dir)
        
        result = discovery.extract_and_select(
            sample_data,
            source_column='description',
            categories=['contact', 'money', 'dates'],
            coverage_threshold=0.1,  # Low threshold to get many fields
            top_k=3,  # Limit to top 3
            include_original=False
        )
        
        # Should have at most 3 fields
        assert len(result.columns) <= 3
    
    def test_debug_artifacts(self, sample_data, temp_output_dir):
        """Test that debug artifacts are created when debug=True."""
        discovery = RuleDiscovery(out_dir=temp_output_dir, debug=True)
        
        discovery.extract_and_select(
            sample_data,
            source_column='description',
            categories=['contact'],
            coverage_threshold=0.1
        )
        
        # Check for debug files
        coverage_file = discovery.run_dir / "autorules_coverage.json"
        selected_file = discovery.run_dir / "autorules_selected_fields.json"
        sample_file = discovery.run_dir / "autorules_samples.csv"
        
        assert coverage_file.exists()
        assert selected_file.exists()
        # Sample file should exist if we have data
        if not sample_data.empty:
            assert sample_file.exists()
    
    def test_extract_method(self, sample_data, temp_output_dir):
        """Test BaseExtractor interface extract method."""
        discovery = RuleDiscovery(out_dir=temp_output_dir)
        
        # Should require source_column
        with pytest.raises(ValueError, match="source_column must be specified"):
            discovery.extract(sample_data)
        
        # Should work with source_column
        result = discovery.extract(sample_data, source_column='description')
        assert isinstance(result, pd.DataFrame)


class TestDiscoverFields:
    """Test cases for discover_fields convenience function."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample DataFrame."""
        data = {
            'product_info': [
                'iPhone 14 Pro - $999.99 - Available in blue, black colors',
                'Contact support@apple.com for technical issues',
                'MacBook Air starts at €1299.00 (13-inch display)',
                'Free shipping on orders over $50.00'
            ]
        }
        return pd.DataFrame(data)
    
    def test_discover_fields_basic(self, sample_data):
        """Test basic discover_fields functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = discover_fields(
                sample_data,
                source_column='product_info',
                categories=['money', 'contact', 'product'],
                coverage_threshold=0.25,
                out_dir=tmpdir,
                debug=False
            )
            
            assert isinstance(result, pd.DataFrame)
            assert len(result) == len(sample_data)
            # Should find some fields (prices, emails, colors should have good coverage)
            assert len(result.columns) > 0
    
    def test_discover_fields_with_original(self, sample_data):
        """Test discover_fields with include_original=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = discover_fields(
                sample_data,
                source_column='product_info',
                categories=['money', 'contact'],
                coverage_threshold=0.1,
                include_original=True,
                out_dir=tmpdir
            )
            
            # Should include original column
            assert 'product_info' in result.columns
            assert len(result.columns) > 1
    
    def test_discover_fields_all_categories(self, sample_data):
        """Test discover_fields with all categories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = discover_fields(
                sample_data,
                source_column='product_info',
                coverage_threshold=0.1,  # Low threshold to catch more patterns
                top_k=10,  # Limit results
                out_dir=tmpdir
            )
            
            assert isinstance(result, pd.DataFrame)
            # With all categories and low threshold, should find multiple fields
            assert len(result.columns) > 0


class TestIntegration:
    """Integration tests combining autorules with other extractors."""
    
    def test_with_existing_workflow(self):
        """Test autorules as part of larger extraction workflow."""
        # Create mixed data with various patterns
        data = {
            'text': [
                'Email us at support@example.com or call (555) 123-4567',
                'Visit https://example.com for deals starting at $19.99',
                'Located at 123 Main St, ZIP 12345 - Open since 2020',
                'Product model ABC123 in red color - €45.00 with 2 year warranty'
            ]
        }
        df = pd.DataFrame(data)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use autorules to discover high-coverage fields
            discovery_result = discover_fields(
                df,
                source_column='text',
                categories=['contact', 'money', 'dates', 'product'],
                coverage_threshold=0.25,
                top_k=8,
                include_original=True,
                out_dir=tmpdir
            )
            
            # Should have original column plus discovered fields
            assert 'text' in discovery_result.columns
            assert len(discovery_result.columns) > 1
            
            # All rows should be preserved
            assert len(discovery_result) == len(df)
            
            # Some fields should have been discovered
            discovered_fields = [col for col in discovery_result.columns if col != 'text']
            assert len(discovered_fields) > 0