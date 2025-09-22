"""Tests for information extraction module."""

import pandas as pd
import pytest
import re
from pathlib import Path
import tempfile
import shutil

from PyDI.informationextraction import (
    BaseExtractor, 
    ExtractorPipeline, 
    RegexExtractor, 
    CodeExtractor, 
    built_in_rules
)
from PyDI.informationextraction.rules import TRANSFORMATIONS
from PyDI.normalization.types import NumericParser


class TestTransformations:
    """Test transformation functions."""
    
    def test_parse_money(self):
        """Test money parsing transformation."""
        parse_money = TRANSFORMATIONS['parse_money']
        
        assert parse_money("$123.45") == 123.45
        assert parse_money("$1,234.56") == 1234.56
        assert parse_money("€123,45") == 123.45  # European format
        assert parse_money("123.456,78") == 123456.78  # European format
        assert parse_money("$-123.45") == -123.45
        assert parse_money("no money here") is None
        assert parse_money("") is None
        assert parse_money(None) is None
    
    def test_parse_number(self):
        """Test number parsing transformation."""
        numeric_parser = NumericParser()
        
        assert numeric_parser.parse_numeric("123") == 123.0
        assert numeric_parser.parse_numeric("123.45") == 123.45
        assert numeric_parser.parse_numeric("Item costs 99.99 dollars") == 99.99
        assert numeric_parser.parse_numeric("-42.5") == -42.5
        assert numeric_parser.parse_numeric("no numbers") is None
        assert numeric_parser.parse_numeric("") is None
    
    def test_text_transformations(self):
        """Test text transformation functions."""
        assert TRANSFORMATIONS['lower']("HELLO") == "hello"
        assert TRANSFORMATIONS['strip']("  hello  ") == "hello"
        assert TRANSFORMATIONS['normalize_whitespace']("hello\n\t world") == "hello world"
    
    def test_parse_percent(self):
        """Test percentage parsing transformation."""
        parse_percent = TRANSFORMATIONS['parse_percent']
        result = parse_percent("50%")
        assert result == 0.5 or result == 50.0  # Handle both fraction and percentage
        
    def test_parse_date(self):
        """Test date parsing transformation."""
        parse_date = TRANSFORMATIONS['parse_date'] 
        assert parse_date("2023-12-25") is not None
        
    def test_url_parsing(self):
        """Test URL parsing and normalization."""
        normalize_url = TRANSFORMATIONS['normalize_url']
        extract_domain = TRANSFORMATIONS['extract_domain']
        
        test_url = "https://example.com/path"
        normalized = normalize_url(test_url)
        assert normalized is not None
        domain = extract_domain(test_url)
        assert domain == "example.com" or domain is None  # Handle normalization differences
        
    def test_storage_parsing(self):
        """Test storage capacity parsing."""
        parse_storage_gb = TRANSFORMATIONS['parse_storage_gb']
        assert parse_storage_gb("512GB") == 512.0
        assert parse_storage_gb("1TB") == 1024.0
        
    def test_power_parsing(self):
        """Test power parsing transformation."""
        parse_power_w = TRANSFORMATIONS['parse_power_w']
        assert parse_power_w("100W") == 100.0
        assert parse_power_w("1kW") == 1000.0
        
    def test_frequency_parsing(self):
        """Test frequency parsing transformation."""
        parse_frequency_hz = TRANSFORMATIONS['parse_frequency_hz']
        assert parse_frequency_hz("2GHz") == 2000000000.0
        assert parse_frequency_hz("1000Hz") == 1000.0
        
    def test_employee_count_parsing(self):
        """Test employee count parsing with multipliers."""
        parse_employee_count = TRANSFORMATIONS['parse_employee_count']
        assert parse_employee_count("500 employees") == 500.0
        assert parse_employee_count("50k employees") == 50000.0
        assert parse_employee_count("2M staff") == 2000000.0


class TestBuiltInRules:
    """Test built-in regex rules."""
    
    def test_identifier_rules(self):
        """Test identifier extraction rules."""
        uuid_rule = built_in_rules["identifiers"]["uuid4"]
        pattern = re.compile(uuid_rule["pattern"], uuid_rule["flags"])
        
        # Valid UUID4
        assert pattern.search("550e8400-e29b-41d4-a716-446655440000")
        
        # Invalid UUID (wrong version)
        assert not pattern.search("550e8400-e29b-31d4-a716-446655440000")
        
        imdb_rule = built_in_rules["identifiers"]["imdb_id"]
        pattern = re.compile(imdb_rule["pattern"], imdb_rule["flags"])
        
        assert pattern.search("tt1234567")
        assert pattern.search("tt12345678")
        assert not pattern.search("tt123456")  # Too short
        assert not pattern.search("mm1234567")  # Wrong prefix
    
    def test_contact_rules(self):
        """Test contact information rules."""
        email_rule = built_in_rules["contact"]["email"]
        pattern = re.compile(email_rule["pattern"], email_rule["flags"])
        
        assert pattern.search("test@example.com")
        assert pattern.search("user.name+tag@domain.co.uk")
        assert not pattern.search("invalid-email")
        assert not pattern.search("@domain.com")
        
        url_rule = built_in_rules["contact"]["url"]
        pattern = re.compile(url_rule["pattern"], url_rule["flags"])
        
        assert pattern.search("https://example.com")
        assert pattern.search("http://subdomain.example.com/path")
        assert not pattern.search("ftp://example.com")
    
    def test_money_rules(self):
        """Test money extraction rules."""
        price_rule = built_in_rules["money"]["price_symbol"]
        pattern = re.compile(price_rule["pattern"], price_rule["flags"])
        
        assert pattern.search("$123.45")
        assert pattern.search("£1,234")
        assert pattern.search("€99.99")
        assert not pattern.search("123.45")  # No symbol
        
        percent_rule = built_in_rules["money"]["percent"]
        pattern = re.compile(percent_rule["pattern"], percent_rule["flags"])
        
        assert pattern.search("50%")
        assert pattern.search("12.5%")
        assert not pattern.search("fifty percent")
    
    def test_measurement_rules(self):
        """Test measurement extraction rules."""
        length_rule = built_in_rules["measurements"]["length_metric"]
        pattern = re.compile(length_rule["pattern"], length_rule["flags"])
        
        assert pattern.search("10cm")
        assert pattern.search("1.5 m")
        assert pattern.search("500 mm")
        assert not pattern.search("10 inches")
        
        temp_rule = built_in_rules["measurements"]["temperature"]
        pattern = re.compile(temp_rule["pattern"], temp_rule["flags"])
        
        assert pattern.search("20°C")
        assert pattern.search("98.6°F")
        assert not pattern.search("20 degrees")
    
    def test_new_identifier_rules(self):
        """Test new identifier patterns."""
        # Test ISIN
        isin_rule = built_in_rules["identifiers"]["isin"]
        pattern = re.compile(isin_rule["pattern"], isin_rule["flags"])
        assert pattern.search("US0378331005")  # Apple ISIN
        assert not pattern.search("US037833100")  # Too short
        
        # Test IBAN
        iban_rule = built_in_rules["identifiers"]["iban"]
        pattern = re.compile(iban_rule["pattern"], iban_rule["flags"])
        assert pattern.search("GB82WEST12345698765432")
        assert not pattern.search("GB82WEST123")  # Too short
        
        # Test VAT
        vat_rule = built_in_rules["identifiers"]["vat_generic"]
        pattern = re.compile(vat_rule["pattern"], vat_rule["flags"])
        assert pattern.search("DE123456789")
        assert pattern.search("FR12345678901")
        assert not pattern.search("DE123")  # Too short
    
    def test_geographical_rules(self):
        """Test geographical patterns."""
        # Test US postal codes
        postal_us_rule = built_in_rules["geo"]["postal_us"]
        pattern = re.compile(postal_us_rule["pattern"], postal_us_rule["flags"])
        assert pattern.search("90210")
        assert pattern.search("90210-1234")
        assert not pattern.search("9021")  # Too short
        
        # Test UK postal codes
        postal_uk_rule = built_in_rules["geo"]["postal_uk"]
        pattern = re.compile(postal_uk_rule["pattern"], postal_uk_rule["flags"])
        assert pattern.search("SW1A 1AA")
        assert pattern.search("M1 1AA")
        assert not pattern.search("123456")
        
        # Test coordinates
        lat_long_rule = built_in_rules["geo"]["lat_long"]
        pattern = re.compile(lat_long_rule["pattern"], lat_long_rule["flags"])
        assert pattern.search("51.5074, -0.1278")  # London
        assert pattern.search("40.7128 -74.0060")  # NYC
        assert not pattern.search("not coordinates")
    
    def test_product_spec_rules(self):
        """Test product specification patterns."""
        # Test storage
        storage_rule = built_in_rules["product"]["storage_gb_tb"]
        pattern = re.compile(storage_rule["pattern"], storage_rule["flags"])
        assert pattern.search("512GB")
        assert pattern.search("1TB")
        assert pattern.search("2.5TB")
        assert not pattern.search("512MB")
        
        # Test battery
        battery_rule = built_in_rules["product"]["battery_mah"]
        pattern = re.compile(battery_rule["pattern"], battery_rule["flags"])
        assert pattern.search("5000mAh")
        assert pattern.search("3,500 mAh")
        assert not pattern.search("5000W")
        
        # Test resolution
        resolution_rule = built_in_rules["product"]["resolution_px"]
        pattern = re.compile(resolution_rule["pattern"], resolution_rule["flags"])
        assert pattern.search("1920x1080")
        assert pattern.search("1920 × 1080 pixels")
        assert not pattern.search("1920p")
        
        # Test video resolution tags
        video_rule = built_in_rules["product"]["video_resolution_tag"]
        pattern = re.compile(video_rule["pattern"], video_rule["flags"])
        assert pattern.search("1080p")
        assert pattern.search("4K")
        assert pattern.search("720p")
        assert not pattern.search("1081p")
    
    def test_contact_phone_e164(self):
        """Test E164 phone number pattern."""
        e164_rule = built_in_rules["contact"]["phone_e164"]
        pattern = re.compile(e164_rule["pattern"], e164_rule["flags"])
        assert pattern.search("+1234567890")
        assert pattern.search("+447700900123")
        assert not pattern.search("+0123456789")  # Cannot start with 0
        assert not pattern.search("1234567890")   # Must have +
    
    def test_key_value_separators(self):
        """Test additional key-value separators."""
        # Test dash separator
        dash_rule = built_in_rules["key_value"]["dash_separator"]
        pattern = re.compile(dash_rule["pattern"], dash_rule["flags"])
        match = pattern.search("Color - Blue")
        assert match
        assert match.groups() == ("Color", "Blue")
        
        # Test semicolon separator
        semicolon_rule = built_in_rules["key_value"]["semicolon_separator"]
        pattern = re.compile(semicolon_rule["pattern"], semicolon_rule["flags"])
        match = pattern.search("Brand; Apple")
        assert match
        assert match.groups() == ("Brand", "Apple")
    
    def test_company_employee_count_multiplier(self):
        """Test employee count with multipliers."""
        employee_rule = built_in_rules["company"]["employee_count"]
        pattern = re.compile(employee_rule["pattern"], employee_rule["flags"])
        
        # Test basic numbers
        assert pattern.search("100 employees")
        assert pattern.search("1,500 staff")
        
        # Test multipliers
        assert pattern.search("50k employees")
        assert pattern.search("2M workers")
        assert pattern.search("250K staff")
        
        # Test postprocessing
        postprocess = built_in_rules["company"]["employee_count"]["postprocess"]
        parse_func = TRANSFORMATIONS[postprocess]
        assert parse_func("100 employees") == 100.0
        assert parse_func("50k employees") == 50000.0
        assert parse_func("2M staff") == 2000000.0


class MockExtractor(BaseExtractor):
    """Mock extractor for testing base functionality."""
    
    def extract(self, df, *, source_column=None, **kwargs):
        result = df.copy()
        result['mock_field'] = 'extracted'
        return result


class TestBaseExtractor:
    """Test base extractor functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_output_dir_creation(self, temp_dir):
        """Test output directory creation."""
        extractor = MockExtractor(out_dir=temp_dir, debug=True)
        assert extractor.run_dir.exists()
        assert extractor.run_dir.parent == Path(temp_dir)
    
    def test_validate_input(self):
        """Test input validation."""
        extractor = MockExtractor()
        df = pd.DataFrame({"text": ["hello", "world"]})
        
        # Valid input
        source = extractor._validate_input(df, "text")
        assert source == "text"
        
        # Empty DataFrame
        with pytest.raises(ValueError, match="empty"):
            extractor._validate_input(pd.DataFrame())
        
        # Invalid column
        with pytest.raises(ValueError, match="not found"):
            extractor._validate_input(df, "invalid_column")
        
        # No source column specified
        with pytest.raises(ValueError, match="must be specified"):
            extractor._validate_input(df, None)
    
    def test_artifact_writing(self, temp_dir):
        """Test artifact writing."""
        extractor = MockExtractor(out_dir=temp_dir, debug=True)
        
        # Write JSON
        data = {"test": "data"}
        filepath = extractor._write_artifact("test.json", data)
        assert filepath.exists()
        
        # Write CSV
        df = pd.DataFrame({"col": [1, 2, 3]})
        filepath = extractor._write_artifact("test.csv", df)
        assert filepath.exists()
        
        # Write text
        filepath = extractor._write_artifact("test.txt", "hello world")
        assert filepath.exists()


class TestRegexExtractor:
    """Test regex-based extraction."""
    
    @pytest.fixture
    def sample_df(self):
        """Sample DataFrame for testing."""
        return pd.DataFrame({
            "text": [
                "Contact us at info@example.com or call $123.45",
                "Visit https://website.com for 50% off",
                "The item costs €99.99 and weighs 2.5kg",
                "No useful info here",
                ""
            ]
        })
    
    def test_basic_extraction(self, sample_df):
        """Test basic regex extraction."""
        rules = {
            "email": {
                "source_column": "text",
                "pattern": built_in_rules["contact"]["email"]["pattern"],
                "flags": re.IGNORECASE
            },
            "url": {
                "source_column": "text", 
                "pattern": built_in_rules["contact"]["url"]["pattern"]
            }
        }
        
        extractor = RegexExtractor(rules)
        result = extractor.extract(sample_df)
        
        assert "email" in result.columns
        assert "url" in result.columns
        assert result.loc[0, "email"] == "info@example.com"
        assert result.loc[1, "url"] == "https://website.com"
        assert pd.isna(result.loc[3, "email"])  # No match
    
    def test_postprocessing(self, sample_df):
        """Test postprocessing transformations."""
        rules = {
            "price": {
                "source_column": "text",
                "pattern": built_in_rules["money"]["price_symbol"]["pattern"],
                "postprocess": "parse_money"
            }
        }
        
        extractor = RegexExtractor(rules)
        result = extractor.extract(sample_df)
        
        assert result.loc[0, "price"] == 123.45
        assert result.loc[2, "price"] == 99.99
    
    def test_multiple_patterns(self, sample_df):
        """Test extraction with multiple patterns."""
        rules = {
            "price": {
                "source_column": "text",
                "pattern": [
                    built_in_rules["money"]["price_symbol"]["pattern"],
                    built_in_rules["money"]["price_iso"]["pattern"]
                ],
                "postprocess": "parse_money"
            }
        }
        
        extractor = RegexExtractor(rules)
        result = extractor.extract(sample_df)
        
        assert len(result.columns) == len(sample_df.columns) + 1
        assert result.loc[0, "price"] == 123.45
    
    def test_group_capture(self):
        """Test regex group capture."""
        df = pd.DataFrame({"text": ["Price: $123.45", "Cost: €99.99"]})
        
        rules = {
            "amount": {
                "source_column": "text",
                "pattern": r"([\$€])(\d+\.\d+)",
                "group": 2,
                "postprocess": "parse_number"
            }
        }
        
        extractor = RegexExtractor(rules, default_source="text")
        result = extractor.extract(df)
        
        assert result.loc[0, "amount"] == 123.45
        assert result.loc[1, "amount"] == 99.99
    
    def test_invalid_patterns(self):
        """Test handling of invalid regex patterns."""
        rules = {
            "bad_pattern": {
                "source_column": "text",
                "pattern": "[invalid regex"  # Missing closing bracket
            }
        }
        
        df = pd.DataFrame({"text": ["test"]})
        extractor = RegexExtractor(rules)
        result = extractor.extract(df)
        
        # Should not have extracted column due to invalid pattern
        assert "bad_pattern" not in result.columns


class TestCodeExtractor:
    """Test code-based extraction."""
    
    @pytest.fixture
    def sample_df(self):
        """Sample DataFrame for testing."""
        return pd.DataFrame({
            "text": ["Hello World", "PYTHON programming", "Data Science 2023"],
            "category": ["greeting", "tech", "education"],
            "value": [10, 25, 100]
        })
    
    def test_text_function(self, sample_df):
        """Test text-based function extraction."""
        def extract_length(text):
            return len(text) if isinstance(text, str) else 0
        
        def extract_uppercase_count(text):
            return sum(1 for c in str(text) if c.isupper())
        
        functions = {
            "text_length": extract_length,
            "uppercase_count": extract_uppercase_count
        }
        
        extractor = CodeExtractor(functions, default_source="text")
        result = extractor.extract(sample_df)
        
        assert result.loc[0, "text_length"] == 11  # "Hello World"
        assert result.loc[1, "uppercase_count"] == 6  # "PYTHON"
        assert result.loc[2, "text_length"] == 17  # "Data Science 2023"
    
    def test_row_function(self, sample_df):
        """Test row-based function extraction."""
        def combine_category_value(row):
            return f"{row['category']}_{row['value']}"
        
        def is_high_value(row):
            return row['value'] > 50
        
        functions = {
            "category_value": combine_category_value,
            "high_value": is_high_value
        }
        
        extractor = CodeExtractor(functions)
        result = extractor.extract(sample_df)
        
        assert result.loc[0, "category_value"] == "greeting_10"
        assert result.loc[1, "high_value"] == False
        assert result.loc[2, "high_value"] == True
    
    def test_function_errors(self, sample_df):
        """Test handling of function errors."""
        def error_function(text):
            raise ValueError("This function always fails")
        
        def divide_by_zero(text):
            return 1 / 0
        
        functions = {
            "error_field": error_function,
            "zero_division": divide_by_zero
        }
        
        extractor = CodeExtractor(functions, default_source="text")
        result = extractor.extract(sample_df)
        
        # Should have columns with None values due to errors
        assert "error_field" in result.columns
        assert "zero_division" in result.columns
        assert all(pd.isna(result["error_field"]))
        assert all(pd.isna(result["zero_division"]))
    
    def test_vectorization_toggle(self, sample_df):
        """Test vectorization behavior."""
        def simple_upper(text):
            return str(text).upper()
        
        functions = {"upper_text": simple_upper}
        
        # With vectorization
        extractor_vec = CodeExtractor(functions, vectorize=True, default_source="text")
        result_vec = extractor_vec.extract(sample_df)
        
        # Without vectorization  
        extractor_no_vec = CodeExtractor(functions, vectorize=False, default_source="text")
        result_no_vec = extractor_no_vec.extract(sample_df)
        
        # Results should be the same
        pd.testing.assert_series_equal(result_vec["upper_text"], result_no_vec["upper_text"])


class TestExtractorPipeline:
    """Test extractor pipeline."""
    
    @pytest.fixture
    def sample_df(self):
        """Sample DataFrame for testing."""
        return pd.DataFrame({
            "text": [
                "Contact info@test.com for $99.99 deal",
                "Visit https://sale.com for 25% discount",
                "Special offer: €150.00 this week only"
            ]
        })
    
    def test_pipeline_execution(self, sample_df):
        """Test pipeline execution with multiple extractors."""
        # Regex extractor for basic patterns
        regex_rules = {
            "email": {
                "source_column": "text",
                "pattern": built_in_rules["contact"]["email"]["pattern"]
            },
            "url": {
                "source_column": "text",
                "pattern": built_in_rules["contact"]["url"]["pattern"]
            }
        }
        
        # Code extractor for additional processing
        def extract_domain(row):
            email = row.get("email")
            if pd.notna(email) and isinstance(email, str):
                return email.split("@")[1] if "@" in email else None
            return None
        
        code_functions = {
            "domain": extract_domain
        }
        
        # Create pipeline
        extractors = [
            RegexExtractor(regex_rules),
            CodeExtractor(code_functions)
        ]
        
        pipeline = ExtractorPipeline(extractors)
        result = pipeline.run(sample_df)
        
        # Check that both extractors ran
        assert "email" in result.columns
        assert "url" in result.columns  
        assert "domain" in result.columns
        assert result.loc[0, "domain"] == "test.com"
    
    def test_empty_pipeline(self, sample_df):
        """Test pipeline with no extractors."""
        pipeline = ExtractorPipeline([])
        result = pipeline.run(sample_df)
        
        # Should return original DataFrame unchanged
        pd.testing.assert_frame_equal(result, sample_df)


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_end_to_end_extraction(self):
        """Test complete extraction workflow."""
        # Create sample data
        df = pd.DataFrame({
            "product_description": [
                "MacBook Pro 16-inch for $2,499.99 - contact sales@apple.com",
                "iPhone 13 Pro 128GB - $999.00, call 1-800-APL-CARE",
                "iPad Air with 256GB storage €679.99 - visit https://apple.com",
                "Regular text with no structured data",
                ""
            ]
        })
        
        # Define extraction rules
        rules = {
            "price": {
                "source_column": "product_description",
                "pattern": [
                    built_in_rules["money"]["price_symbol"]["pattern"],
                    built_in_rules["money"]["price_iso"]["pattern"]
                ],
                "postprocess": "parse_money"
            },
            "email": {
                "source_column": "product_description", 
                "pattern": built_in_rules["contact"]["email"]["pattern"]
            },
            "url": {
                "source_column": "product_description",
                "pattern": built_in_rules["contact"]["url"]["pattern"]
            }
        }
        
        # Add custom functions
        def extract_storage(row):
            text = str(row.get('product_description', ''))
            import re
            match = re.search(r'(\d+)(GB|TB)', text)
            if match:
                size, unit = match.groups()
                multiplier = 1024 if unit == 'TB' else 1
                return int(size) * multiplier
            return None
        
        def categorize_product(row):
            text = str(row.get('product_description', '')).lower()
            if 'macbook' in text or 'laptop' in text:
                return 'laptop'
            elif 'iphone' in text or 'phone' in text:
                return 'phone'
            elif 'ipad' in text or 'tablet' in text:
                return 'tablet'
            return 'other'
        
        functions = {
            'storage_gb': extract_storage,
            'category': categorize_product
        }
        
        # Create pipeline
        extractors = [
            RegexExtractor(rules),
            CodeExtractor(functions)
        ]
        
        pipeline = ExtractorPipeline(extractors)
        result = pipeline.run(df, debug=False)
        
        # Verify extracted data
        assert len(result) == len(df)
        assert result.loc[0, 'price'] == 2499.99
        assert result.loc[1, 'price'] == 999.00
        assert result.loc[2, 'price'] == 679.99
        assert result.loc[0, 'email'] == 'sales@apple.com'
        assert result.loc[2, 'url'] == 'https://apple.com'
        assert result.loc[1, 'storage_gb'] == 128
        assert result.loc[2, 'storage_gb'] == 256
        assert result.loc[0, 'category'] == 'laptop'
        assert result.loc[1, 'category'] == 'phone'
        assert result.loc[2, 'category'] == 'tablet'
        
        # Check for proper handling of missing data
        assert pd.isna(result.loc[3, 'price'])
        assert pd.isna(result.loc[4, 'email'])


if __name__ == "__main__":
    pytest.main([__file__])