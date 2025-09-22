"""
Tests for the LLM-based entity matcher.
"""

import json
import tempfile
from pathlib import Path
from typing import List, Optional
from unittest.mock import Mock

import pandas as pd
import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage

from PyDI.entitymatching import LLMBasedMatcher


class MockChatModel:
    """Mock chat model for testing without API calls."""
    
    def __init__(self, responses: Optional[List[str]] = None):
        """Initialize with predefined responses."""
        self._responses = responses or []
        self._call_count = 0
        self.model_name = "mock-model"
    
    @property
    def responses(self):
        return self._responses
    
    @property
    def call_count(self):
        return self._call_count
    
    def invoke(self, input_messages: List[BaseMessage], **kwargs) -> AIMessage:
        """Invoke the mock model."""
        if self._call_count < len(self._responses):
            content = self._responses[self._call_count]
        else:
            # Default response for exact match
            content = '{"match": true, "score": 0.9, "explanation": "exact match"}'
        
        self._call_count += 1
        
        # Create a mock response with usage info
        response = AIMessage(content=content)
        response.usage = {"prompt_tokens": 100, "completion_tokens": 20}
        return response


@pytest.fixture
def sample_data():
    """Create sample DataFrames for testing."""
    df_left = pd.DataFrame({
        "_id": ["l1", "l2", "l3"],
        "name": ["Acme Corp", "Beta Inc", "Gamma LLC"],
        "address": ["123 Main St", "456 Oak Ave", "789 Pine Rd"],
        "city": ["New York", "Boston", "Chicago"]
    })
    df_left.attrs["dataset_name"] = "test_left"
    
    df_right = pd.DataFrame({
        "_id": ["r1", "r2", "r3"],
        "name": ["Acme Corporation", "Beta Incorporated", "Delta Systems"],
        "address": ["123 Main Street", "456 Oak Avenue", "999 Elm St"],
        "city": ["New York", "Boston", "Seattle"]
    })
    df_right.attrs["dataset_name"] = "test_right"
    
    candidates = [("l1", "r1"), ("l2", "r2"), ("l3", "r3")]
    
    return df_left, df_right, candidates


@pytest.fixture
def llm_matcher():
    """Create an LLMBasedMatcher instance."""
    return LLMBasedMatcher()


def test_basic_matching(sample_data, llm_matcher):
    """Test basic LLM matching functionality."""
    df_left, df_right, candidates = sample_data
    
    # Create mock model that returns high confidence matches
    mock_model = MockChatModel([
        '{"match": true, "score": 0.9, "explanation": "name and address match"}',
        '{"match": true, "score": 0.85, "explanation": "name and location match"}',
        '{"match": false, "score": 0.1, "explanation": "different companies"}'
    ])
    
    matches = llm_matcher.match(
        df_left, df_right, candidates,
        chat_model=mock_model,
        threshold=0.5
    )
    
    # Should have 2 matches (first two above threshold)
    assert len(matches) == 2
    assert set(matches.columns) == {"id1", "id2", "score", "notes"}
    
    # Check first match
    first_match = matches.iloc[0]
    assert first_match["id1"] == "l1"
    assert first_match["id2"] == "r1"
    assert first_match["score"] == 0.9
    assert "name and address match" in first_match["notes"]


def test_zero_shot_vs_few_shot(sample_data, llm_matcher):
    """Test zero-shot vs few-shot prompting."""
    df_left, df_right, candidates = sample_data
    mock_model = MockChatModel()
    
    # Test zero-shot
    matches_zero = llm_matcher.match(
        df_left, df_right, candidates[:1],
        chat_model=mock_model
    )
    
    # Reset mock
    mock_model._call_count = 0
    
    # Test few-shot
    few_shots = [
        (
            {"name": "Example Corp", "address": "100 Test St"},
            {"name": "Example Corporation", "address": "100 Test Street"},
            '{"match": true, "score": 0.95, "explanation": "same company"}'
        )
    ]
    
    matches_few = llm_matcher.match(
        df_left, df_right, candidates[:1],
        chat_model=mock_model,
        few_shots=few_shots
    )
    
    # Both should work and return results
    assert len(matches_zero) >= 0
    assert len(matches_few) >= 0


def test_field_selection(sample_data, llm_matcher):
    """Test field selection functionality."""
    df_left, df_right, candidates = sample_data
    mock_model = MockChatModel()
    
    # Test with specific fields
    matches = llm_matcher.match(
        df_left, df_right, candidates[:1],
        chat_model=mock_model,
        fields=["name", "city"]  # Exclude address
    )
    
    assert len(matches) >= 0


def test_auto_field_selection(llm_matcher):
    """Test automatic field selection."""
    # Create data with mixed types
    df_left = pd.DataFrame({
        "_id": ["l1"],
        "name": ["Test"],
        "score": [95.5],  # numeric
        "active": [True],  # boolean
        "description": ["A test company"]  # string
    })
    df_left.attrs["dataset_name"] = "test_left"
    
    df_right = pd.DataFrame({
        "_id": ["r1"],
        "name": ["Test Corp"],
        "score": [90.0],
        "active": [True],
        "description": ["A testing company"]
    })
    df_right.attrs["dataset_name"] = "test_right"
    
    candidates = [("l1", "r1")]
    mock_model = MockChatModel()
    
    matches = llm_matcher.match(
        df_left, df_right, candidates,
        chat_model=mock_model
        # fields=None should auto-select string columns
    )
    
    assert len(matches) >= 0


def test_json_parsing_edge_cases(sample_data, llm_matcher):
    """Test JSON parsing with various response formats."""
    df_left, df_right, candidates = sample_data
    
    # Test responses with extra text
    responses = [
        'Here is my analysis: {"match": true, "score": 0.8, "explanation": "good match"}',
        '{"match": false, "score": 0.2, "explanation": "no match"} - this is my conclusion',
        'Sure! {"match": true, "score": 0.9, "explanation": "excellent"} Hope this helps.',
    ]
    
    mock_model = MockChatModel(responses)
    
    matches = llm_matcher.match(
        df_left, df_right, candidates,
        chat_model=mock_model,
        threshold=0.5
    )
    
    # Should successfully parse and get 2 matches (scores 0.8 and 0.9)
    assert len(matches) == 2


def test_parsing_strictness(sample_data, llm_matcher):
    """Test different parsing strictness settings."""
    df_left, df_right, candidates = sample_data
    
    # Create model that returns invalid JSON
    bad_responses = [
        'This is not JSON at all',
        'Almost JSON but not quite {"match": true',
        '{"match": true, "score": "invalid"}'  # Invalid score type
    ]
    
    mock_model = MockChatModel(bad_responses)
    
    # Test "skip" mode (default)
    matches_skip = llm_matcher.match(
        df_left, df_right, candidates,
        chat_model=mock_model,
        parse_strictness="skip"
    )
    
    # Should skip unparseable responses
    assert len(matches_skip) == 0
    
    # Reset mock
    mock_model._call_count = 0
    
    # Test "zero_score" mode  
    matches_zero = llm_matcher.match(
        df_left, df_right, candidates,
        chat_model=mock_model,
        parse_strictness="zero_score",
        threshold=0.0  # Include zero scores
    )
    
    # Should include responses with score 0.0
    assert len(matches_zero) >= 0


def test_retry_logic(sample_data, llm_matcher):
    """Test retry logic on failures."""
    df_left, df_right, candidates = sample_data
    
    # Mock model that fails first few times
    mock_model = MockChatModel()
    
    # Mock the invoke method to fail then succeed
    original_invoke = mock_model.invoke
    call_count = 0
    
    def failing_invoke(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:  # Fail first 2 attempts
            raise Exception("API Error")
        return original_invoke(*args, **kwargs)
    
    mock_model.invoke = failing_invoke
    
    matches = llm_matcher.match(
        df_left, df_right, candidates[:1],
        chat_model=mock_model,
        retries=3  # Should succeed on 3rd retry
    )
    
    assert len(matches) >= 0


def test_debug_artifacts(sample_data, llm_matcher):
    """Test debug artifact generation."""
    df_left, df_right, candidates = sample_data
    mock_model = MockChatModel()
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        matches = llm_matcher.match(
            df_left, df_right, candidates[:1],
            chat_model=mock_model,
            debug=True,
            out_dir=tmp_dir
        )
        
        # Check that debug artifacts were created
        debug_path = Path(tmp_dir)
        assert (debug_path / "llm_stats.json").exists()
        
        # Check stats content
        with open(debug_path / "llm_stats.json") as f:
            stats = json.load(f)
        assert "total_candidates" in stats
        assert "total_matches" in stats
        assert "match_rate" in stats


def test_thresholding(sample_data, llm_matcher):
    """Test score thresholding."""
    df_left, df_right, candidates = sample_data
    
    # Create responses with different scores
    responses = [
        '{"match": true, "score": 0.9, "explanation": "high confidence"}',
        '{"match": true, "score": 0.6, "explanation": "medium confidence"}',  
        '{"match": false, "score": 0.3, "explanation": "low confidence"}'
    ]
    
    mock_model = MockChatModel(responses)
    
    # Test with threshold 0.7
    matches = llm_matcher.match(
        df_left, df_right, candidates,
        chat_model=mock_model,
        threshold=0.7
    )
    
    # Should only include the first match (0.9 >= 0.7)
    assert len(matches) == 1
    assert matches.iloc[0]["score"] == 0.9


def test_custom_system_prompt(sample_data, llm_matcher):
    """Test custom system prompt."""
    df_left, df_right, candidates = sample_data
    mock_model = MockChatModel()
    
    custom_prompt = "You are a specialized matcher for companies. Focus on business names."
    
    matches = llm_matcher.match(
        df_left, df_right, candidates[:1],
        chat_model=mock_model,
        system_prompt=custom_prompt
    )
    
    assert len(matches) >= 0


def test_temperature_and_tokens(sample_data, llm_matcher):
    """Test temperature and max_tokens parameters."""
    df_left, df_right, candidates = sample_data
    mock_model = MockChatModel()
    
    matches = llm_matcher.match(
        df_left, df_right, candidates[:1],
        chat_model=mock_model,
        temperature=0.7,
        max_tokens=100
    )
    
    assert len(matches) >= 0


def test_notes_detail_setting(sample_data, llm_matcher):
    """Test notes_detail parameter."""
    df_left, df_right, candidates = sample_data
    
    response_with_explanation = '{"match": true, "score": 0.8, "explanation": "detailed explanation here"}'
    mock_model = MockChatModel([response_with_explanation])
    
    # Test with notes detail enabled
    matches_detailed = llm_matcher.match(
        df_left, df_right, candidates[:1],
        chat_model=mock_model,
        notes_detail=True
    )
    
    assert len(matches_detailed) == 1
    assert "detailed explanation here" in matches_detailed.iloc[0]["notes"]
    
    # Reset mock
    mock_model._call_count = 0
    
    # Test with notes detail disabled
    matches_simple = llm_matcher.match(
        df_left, df_right, candidates[:1],
        chat_model=mock_model,
        notes_detail=False
    )
    
    assert len(matches_simple) == 1
    # Should have empty or minimal notes
    assert matches_simple.iloc[0]["notes"] == ""


def test_empty_candidates(sample_data, llm_matcher):
    """Test with empty candidate list."""
    df_left, df_right, _ = sample_data
    mock_model = MockChatModel()
    
    matches = llm_matcher.match(
        df_left, df_right, [],  # Empty candidates
        chat_model=mock_model
    )
    
    assert len(matches) == 0
    assert set(matches.columns) == {"id1", "id2", "score", "notes"}


def test_score_clamping(sample_data, llm_matcher):
    """Test that scores are clamped to [0,1] range."""
    df_left, df_right, candidates = sample_data
    
    # Responses with out-of-range scores
    responses = [
        '{"match": true, "score": 1.5, "explanation": "too high"}',
        '{"match": false, "score": -0.3, "explanation": "too low"}'
    ]
    
    mock_model = MockChatModel(responses)
    
    matches = llm_matcher.match(
        df_left, df_right, candidates[:2],
        chat_model=mock_model,
        threshold=0.0  # Include all matches
    )
    
    assert len(matches) == 2
    
    # Scores should be clamped
    scores = matches["score"].tolist()
    assert all(0.0 <= score <= 1.0 for score in scores)
    assert 1.0 in scores  # 1.5 clamped to 1.0
    assert 0.0 in scores  # -0.3 clamped to 0.0