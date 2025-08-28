"""Shared fixtures for schema matching tests."""

import pandas as pd
import pytest
import tempfile
import os
from pathlib import Path


@pytest.fixture
def sample_movies_df():
    """Create a sample movies dataset for testing."""
    data = {
        "movie_id": [1, 2, 3, 4, 5],
        "title": ["The Matrix", "Inception", "Pulp Fiction", "The Godfather", "Star Wars"],
        "year": [1999, 2010, 1994, 1972, 1977],
        "genre": ["Sci-Fi", "Thriller", "Crime", "Drama", "Adventure"],
        "rating": [8.7, 8.8, 8.9, 9.2, 8.6]
    }
    df = pd.DataFrame(data)
    df.attrs["dataset_name"] = "movies"
    return df


@pytest.fixture
def sample_films_df():
    """Create a sample films dataset with similar but different schema."""
    data = {
        "film_id": [101, 102, 103, 104, 105],
        "film_name": ["The Matrix", "Inception", "Pulp Fiction", "The Godfather", "Star Wars"],
        "release_year": [1999, 2010, 1994, 1972, 1977],
        "category": ["Science Fiction", "Thriller", "Crime Drama", "Drama", "Space Adventure"],
        "imdb_score": [8.7, 8.8, 8.9, 9.2, 8.6]
    }
    df = pd.DataFrame(data)
    df.attrs["dataset_name"] = "films"
    return df


@pytest.fixture
def sample_awards_df():
    """Create a sample awards dataset."""
    data = {
        "award_id": [1, 2, 3, 4],
        "movie_title": ["The Godfather", "Pulp Fiction", "The Matrix", "Inception"],
        "award_year": [1973, 1995, 2000, 2011],
        "award_type": ["Best Picture", "Palme d'Or", "Best Visual Effects", "Best Cinematography"],
        "winner": [True, True, True, False]
    }
    df = pd.DataFrame(data)
    df.attrs["dataset_name"] = "awards"
    return df


@pytest.fixture
def sample_correspondences_df():
    """Create sample record correspondences between movies and films."""
    data = {
        "id1": [1, 2, 3, 4, 5],
        "id2": [101, 102, 103, 104, 105],
        "confidence": [1.0, 1.0, 1.0, 1.0, 1.0]
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_evaluation_mapping():
    """Create a sample evaluation mapping for testing."""
    data = {
        "source_dataset": ["movies", "movies", "movies", "movies"],
        "source_column": ["movie_id", "title", "year", "genre"],
        "target_dataset": ["films", "films", "films", "films"],
        "target_column": ["film_id", "film_name", "release_year", "category"],
        "label": [True, True, True, True]
    }
    return pd.DataFrame(data)


@pytest.fixture
def empty_df():
    """Create an empty DataFrame for testing edge cases."""
    df = pd.DataFrame()
    df.attrs["dataset_name"] = "empty"
    return df


@pytest.fixture
def single_column_df():
    """Create a DataFrame with a single column."""
    data = {"single_col": [1, 2, 3, 4, 5]}
    df = pd.DataFrame(data)
    df.attrs["dataset_name"] = "single"
    return df


@pytest.fixture
def df_with_nulls():
    """Create a DataFrame with null values."""
    data = {
        "col_a": [1, None, 3, None, 5],
        "col_b": ["a", "b", None, "d", None],
        "col_c": [1.1, 2.2, 3.3, None, 5.5]
    }
    df = pd.DataFrame(data)
    df.attrs["dataset_name"] = "nulls"
    return df


@pytest.fixture
def identical_schema_df1():
    """Create first DataFrame with identical schema for testing."""
    data = {
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35]
    }
    df = pd.DataFrame(data)
    df.attrs["dataset_name"] = "identical1"
    return df


@pytest.fixture
def identical_schema_df2():
    """Create second DataFrame with identical schema for testing."""
    data = {
        "id": [4, 5, 6],
        "name": ["David", "Eve", "Frank"],
        "age": [40, 45, 50]
    }
    df = pd.DataFrame(data)
    df.attrs["dataset_name"] = "identical2"
    return df


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def preprocessing_func():
    """Sample preprocessing function for testing."""
    def preprocess(text: str) -> str:
        return text.lower().replace("_", "").replace(" ", "")
    return preprocess