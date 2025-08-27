"""Shared fixtures for data translation tests."""

import pandas as pd
import pytest


@pytest.fixture
def sample_schema_mapping():
    """Create a sample schema mapping for testing translation."""
    data = {
        "source_dataset": ["movies", "movies", "movies", "films", "films"],
        "source_column": ["movie_id", "title", "year", "film_id", "film_name"],
        "target_dataset": ["unified", "unified", "unified", "unified", "unified"],
        "target_column": ["id", "name", "release_year", "id", "name"],
        "score": [0.95, 0.90, 0.85, 0.92, 0.88],
        "notes": ["exact_match", "label_match", "similar", "exact_match", "similar"]
    }
    return pd.DataFrame(data)


@pytest.fixture
def movies_df_for_translation():
    """Create a movies DataFrame for translation testing."""
    data = {
        "movie_id": [1, 2, 3],
        "title": ["The Matrix", "Inception", "Pulp Fiction"],
        "year": [1999, 2010, 1994],
        "genre": ["Sci-Fi", "Thriller", "Crime"]
    }
    df = pd.DataFrame(data)
    df.attrs["dataset_name"] = "movies"
    return df


@pytest.fixture
def films_df_for_translation():
    """Create a films DataFrame for translation testing."""
    data = {
        "film_id": [101, 102, 103],
        "film_name": ["The Matrix", "Inception", "Pulp Fiction"],
        "release_year": [1999, 2010, 1994]
    }
    df = pd.DataFrame(data)
    df.attrs["dataset_name"] = "films"
    return df


@pytest.fixture
def df_without_dataset_name():
    """Create a DataFrame without dataset_name in attrs."""
    data = {"col1": [1, 2, 3], "col2": ["a", "b", "c"]}
    return pd.DataFrame(data)


@pytest.fixture
def invalid_schema_mapping():
    """Create an invalid schema mapping missing required columns."""
    data = {
        "source_dataset": ["movies", "films"],
        "source_column": ["movie_id", "film_id"],
        # Missing target_dataset and target_column
        "score": [0.95, 0.92]
    }
    return pd.DataFrame(data)


@pytest.fixture
def empty_schema_mapping():
    """Create an empty schema mapping."""
    return pd.DataFrame(columns=[
        "source_dataset", "source_column", 
        "target_dataset", "target_column", 
        "score", "notes"
    ])


@pytest.fixture
def df_with_column_attrs():
    """Create a DataFrame with column-level attributes for testing provenance."""
    data = {
        "col1": [1, 2, 3],
        "col2": ["a", "b", "c"]
    }
    df = pd.DataFrame(data)
    df.attrs["dataset_name"] = "test_dataset"
    
    # Add column-level attributes
    df["col1"].attrs = {"unit": "count", "datatype": "integer"}
    df["col2"].attrs = {"datatype": "string", "category": "label"}
    
    return df