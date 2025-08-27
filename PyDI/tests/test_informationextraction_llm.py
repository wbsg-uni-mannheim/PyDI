"""Tests for LLM-based information extraction."""

import json
from unittest.mock import Mock, MagicMock
from typing import Optional

import pandas as pd
import pytest

try:
    from pydantic import BaseModel
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

try:
    from PyDI.informationextraction.llm import LLMExtractor, LANGCHAIN_AVAILABLE
except ImportError:
    LANGCHAIN_AVAILABLE = False


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain dependencies not available")
@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
class TestLLMExtractor:
    """Test suite for LLMExtractor."""
    
    @pytest.fixture
    def sample_schema(self):
        """Create a sample Pydantic schema for testing."""
        class Product(BaseModel):
            brand: Optional[str] = None
            model: Optional[str] = None
            price: Optional[float] = None
        return Product
    
    @pytest.fixture 
    def dict_schema(self):
        """Create a sample dictionary schema for testing."""
        return {
            "brand": str,
            "model": str, 
            "price": float
        }
    
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing."""
        return pd.DataFrame({
            'description': [
                'Apple iPhone 14 Pro - $999.99',
                'Samsung Galaxy S23 Ultra 256GB for $1199',
                'Invalid text with no products',
                ''  # Empty string
            ]
        })
    
    @pytest.fixture
    def mock_chat_model(self):
        """Create a mock LangChain chat model."""
        mock_model = Mock()
        mock_model.model_name = "test-model"
        
        # Mock successful responses
        mock_responses = [
            Mock(content='{"brand": "Apple", "model": "iPhone 14 Pro", "price": 999.99}'),
            Mock(content='{"brand": "Samsung", "model": "Galaxy S23 Ultra", "price": 1199.0}'),
            Mock(content='{"brand": null, "model": null, "price": null}'),
            Mock(content='{"brand": null, "model": null, "price": null}')
        ]
        mock_model.invoke.side_effect = mock_responses
        
        return mock_model
    
    @pytest.fixture
    def failing_chat_model(self):
        """Create a mock chat model that returns malformed JSON initially."""
        mock_model = Mock()
        mock_model.model_name = "test-model"
        
        # First call fails, second succeeds
        mock_responses = [
            Mock(content='Invalid JSON response'),
            Mock(content='{"brand": "Apple", "model": "iPhone 14", "price": 999.0}')
        ]
        mock_model.invoke.side_effect = mock_responses
        
        return mock_model
    
    def test_init_pydantic_schema(self, mock_chat_model, sample_schema):
        """Test initialization with Pydantic schema."""
        extractor = LLMExtractor(
            chat_model=mock_chat_model,
            schema=sample_schema,
            source_column="description",
            system_prompt="Extract product info"
        )
        
        assert extractor.is_pydantic is True
        assert extractor.schema_fields == ['brand', 'model', 'price']
        assert extractor.output_parser is not None
    
    def test_init_dict_schema(self, mock_chat_model, dict_schema):
        """Test initialization with dictionary schema."""
        extractor = LLMExtractor(
            chat_model=mock_chat_model,
            schema=dict_schema,
            source_column="description", 
            system_prompt="Extract product info"
        )
        
        assert extractor.is_pydantic is False
        assert extractor.schema_fields == ['brand', 'model', 'price']
        assert extractor.output_parser is None
    
    def test_extract_basic(self, mock_chat_model, sample_schema, sample_df):
        """Test basic extraction functionality."""
        extractor = LLMExtractor(
            chat_model=mock_chat_model,
            schema=sample_schema,
            source_column="description",
            system_prompt="Extract product information as JSON"
        )
        
        result_df = extractor.extract(sample_df)
        
        # Check that schema fields were added
        for field in ['brand', 'model', 'price']:
            assert field in result_df.columns
        
        # Check specific extractions
        assert result_df.iloc[0]['brand'] == 'Apple'
        assert result_df.iloc[0]['model'] == 'iPhone 14 Pro'
        assert result_df.iloc[0]['price'] == 999.99
        
        assert result_df.iloc[1]['brand'] == 'Samsung'
        assert result_df.iloc[1]['model'] == 'Galaxy S23 Ultra'
        assert result_df.iloc[1]['price'] == 1199.0
        
        # Check null handling for invalid/empty text
        assert pd.isna(result_df.iloc[2]['brand'])
        assert pd.isna(result_df.iloc[3]['brand'])
    
    def test_extract_with_few_shots(self, mock_chat_model, sample_schema, sample_df):
        """Test extraction with few-shot examples."""
        few_shots = [
            ("Sony WH-1000XM4 headphones $349", '{"brand": "Sony", "model": "WH-1000XM4", "price": 349.0}')
        ]
        
        extractor = LLMExtractor(
            chat_model=mock_chat_model,
            schema=sample_schema,
            source_column="description",
            system_prompt="Extract product info",
            few_shots=few_shots
        )
        
        result_df = extractor.extract(sample_df)
        
        # Should still work with few-shots
        assert 'brand' in result_df.columns
        assert result_df.iloc[0]['brand'] == 'Apple'
    
    def test_retry_mechanism(self, failing_chat_model, sample_schema):
        """Test retry mechanism with initially failing model."""
        df = pd.DataFrame({'description': ['Apple iPhone 14 - $999']})
        
        extractor = LLMExtractor(
            chat_model=failing_chat_model,
            schema=sample_schema,
            source_column="description",
            system_prompt="Extract product info",
            retries=2
        )
        
        result_df = extractor.extract(df)
        
        # Should succeed after retry
        assert result_df.iloc[0]['brand'] == 'Apple'
        assert result_df.iloc[0]['model'] == 'iPhone 14'
        assert result_df.iloc[0]['price'] == 999.0
        
        # Should have made 2 calls (first failed, second succeeded)
        assert failing_chat_model.invoke.call_count == 2
    
    def test_validation_failure_handling(self, mock_chat_model, sample_schema):
        """Test handling of Pydantic validation failures."""
        # Mock model to return invalid data types
        mock_chat_model.invoke.return_value = Mock(
            content='{"brand": 123, "model": [], "price": "invalid"}'
        )
        
        df = pd.DataFrame({'description': ['Test product']})
        
        extractor = LLMExtractor(
            chat_model=mock_chat_model,
            schema=sample_schema,
            source_column="description",
            system_prompt="Extract product info"
        )
        
        result_df = extractor.extract(df)
        
        # Should handle validation failure gracefully
        assert all(pd.isna(result_df.iloc[0][field]) for field in ['brand', 'model', 'price'])
    
    def test_debug_artifacts(self, mock_chat_model, sample_schema, tmp_path):
        """Test that debug artifacts are written correctly."""
        df = pd.DataFrame({'description': ['Apple iPhone - $999']})
        
        extractor = LLMExtractor(
            chat_model=mock_chat_model,
            schema=sample_schema,
            source_column="description",
            system_prompt="Extract product info",
            debug=True,
            out_dir=str(tmp_path / "test_llm")
        )
        
        result_df = extractor.extract(df)
        
        # Check that debug files exist
        out_dir = tmp_path / "test_llm"
        assert (out_dir / "llm_config.json").exists()
        assert (out_dir / "llm_stats.json").exists()
        assert (out_dir / "samples.csv").exists()
        
        # Check prompt and response artifacts
        prompts_dir = out_dir / "prompts"
        responses_dir = out_dir / "responses"
        
        if prompts_dir.exists():
            prompt_files = list(prompts_dir.glob("*.txt"))
            assert len(prompt_files) > 0
        
        if responses_dir.exists():
            response_files = list(responses_dir.glob("*.json"))
            assert len(response_files) > 0
    
    def test_json_extraction_fallback(self, mock_chat_model, sample_schema):
        """Test JSON extraction from response with extra text."""
        # Mock response with JSON embedded in text
        mock_chat_model.invoke.return_value = Mock(
            content='Here is the extracted information: {"brand": "Apple", "price": 999.99} hope this helps!'
        )
        
        df = pd.DataFrame({'description': ['Apple iPhone']})
        
        extractor = LLMExtractor(
            chat_model=mock_chat_model,
            schema=sample_schema,
            source_column="description",
            system_prompt="Extract info"
        )
        
        result_df = extractor.extract(df)
        
        # Should extract JSON from within the text
        assert result_df.iloc[0]['brand'] == 'Apple'
        assert result_df.iloc[0]['price'] == 999.99
    
    def test_missing_source_column(self, mock_chat_model, sample_schema, sample_df):
        """Test handling of missing source column."""
        extractor = LLMExtractor(
            chat_model=mock_chat_model,
            schema=sample_schema,
            source_column="nonexistent_column",
            system_prompt="Extract info"
        )
        
        result_df = extractor.extract(sample_df)
        
        # Should return original DataFrame unchanged
        assert result_df.equals(sample_df)
    
    def test_temperature_and_max_tokens(self, mock_chat_model, sample_schema, sample_df):
        """Test that temperature and max_tokens are passed to model."""
        extractor = LLMExtractor(
            chat_model=mock_chat_model,
            schema=sample_schema,
            source_column="description",
            system_prompt="Extract info",
            temperature=0.5,
            max_tokens=100
        )
        
        result_df = extractor.extract(sample_df)
        
        # Check that invoke was called with correct parameters
        call_args = mock_chat_model.invoke.call_args_list[0]
        assert call_args[1]['temperature'] == 0.5
        assert call_args[1]['max_tokens'] == 100
    
    def test_source_column_override(self, mock_chat_model, sample_schema):
        """Test overriding source column in extract method."""
        df = pd.DataFrame({
            'description': ['Apple iPhone'],
            'text': ['Samsung Galaxy']
        })
        
        extractor = LLMExtractor(
            chat_model=mock_chat_model,
            schema=sample_schema,
            source_column="description",
            system_prompt="Extract info"
        )
        
        # Override source column to 'text'
        result_df = extractor.extract(df, source_column="text")
        
        # Should use 'text' column instead of 'description'
        # We can verify this by checking that the mock was called
        assert mock_chat_model.invoke.called
        

@pytest.mark.skipif(LANGCHAIN_AVAILABLE, reason="Skip when LangChain is available")
def test_import_error_without_langchain():
    """Test that LLMExtractor raises ImportError when LangChain is not available."""
    # This test only runs when LangChain is NOT available
    with pytest.raises(ImportError, match="LangChain dependencies not available"):
        from PyDI.informationextraction.llm import LLMExtractor
        LLMExtractor(None, {}, "text", "prompt")


def test_langchain_availability_flag():
    """Test the LANGCHAIN_AVAILABLE flag."""
    from PyDI.informationextraction.llm import LANGCHAIN_AVAILABLE
    # Should be boolean
    assert isinstance(LANGCHAIN_AVAILABLE, bool)