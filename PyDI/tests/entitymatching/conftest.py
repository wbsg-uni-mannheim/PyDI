"""Shared fixtures for entity matching tests."""

import pandas as pd
import pytest
import tempfile
import numpy as np
from pathlib import Path
from typing import List, Dict, Any


@pytest.fixture
def sample_movies_left():
    """Create left movies dataset for entity matching testing."""
    data = {
        "_id": ["academy_awards_000000", "academy_awards_000001", "academy_awards_000002", "academy_awards_000003"],
        "title": ["Biutiful", "True Grit", "The Social Network", "Inception"],
        "director": ["Alejandro González Iñárritu", "Joel Coen", "David Fincher", "Christopher Nolan"],
        "year": [2010, 2010, 2010, 2010],
        "genre": ["Drama", "Western", "Drama", "Sci-Fi"],
        "rating": [7.2, 7.6, 7.8, 8.8]
    }
    df = pd.DataFrame(data)
    df.attrs["dataset_name"] = "academy_awards"
    return df


@pytest.fixture
def sample_movies_right():
    """Create right movies dataset for entity matching testing."""
    data = {
        "_id": ["actors_000000", "actors_000001", "actors_000002", "actors_000003"],
        "title": ["Biutiful", "True Grit", "Social Network", "Inception"],
        "actors": ["Javier Bardem", "Hailee Steinfeld", "Jesse Eisenberg", "Leonardo DiCaprio"],
        "year": [2010, 2010, 2010, 2010],
        "category": ["Drama", "Adventure", "Drama", "Science Fiction"],
        "score": [7.2, 7.6, 7.8, 8.8]
    }
    df = pd.DataFrame(data)
    df.attrs["dataset_name"] = "actors"
    return df


@pytest.fixture
def sample_candidate_pairs():
    """Create sample candidate pairs for testing."""
    data = {
        "id1": ["academy_awards_000000", "academy_awards_000001", "academy_awards_000002", "academy_awards_000003"],
        "id2": ["actors_000000", "actors_000001", "actors_000002", "actors_000003"]
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_correspondences():
    """Create sample correspondences for testing evaluation."""
    data = {
        "id1": ["academy_awards_000000", "academy_awards_000001", "academy_awards_000002"],
        "id2": ["actors_000000", "actors_000001", "actors_000002"],
        "score": [0.95, 0.87, 0.92],
        "notes": ["high_confidence", "medium_confidence", "high_confidence"]
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_test_pairs():
    """Create ground truth test pairs for evaluation."""
    data = {
        "id1": ["academy_awards_000000", "academy_awards_000001", "academy_awards_000002", "academy_awards_000003"],
        "id2": ["actors_000000", "actors_000001", "actors_000002", "actors_000003"],
        "label": [1, 1, 1, 0]  # first 3 are matches, last is not
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_training_labels():
    """Create training labels for ML testing."""
    return pd.Series([1, 1, 1, 0], name="label")


@pytest.fixture
def empty_correspondence_set():
    """Create empty correspondence set for testing."""
    return pd.DataFrame(columns=["id1", "id2", "score", "notes"])


@pytest.fixture
def inconsistent_correspondences():
    """Create correspondences with cluster inconsistencies for testing."""
    # Create a triangle where A matches B, B matches C, but A doesn't match C
    data = {
        "id1": ["academy_awards_000000", "academy_awards_000001"],
        "id2": ["actors_000000", "actors_000001"],
        "score": [0.9, 0.8],
        "notes": ["test", "test"]
    }
    return pd.DataFrame(data)


@pytest.fixture
def df_without_record_ids():
    """Create DataFrame without _id column for testing ensure_record_ids."""
    data = {
        "title": ["Movie A", "Movie B", "Movie C"],
        "year": [2020, 2021, 2022],
        "genre": ["Action", "Drama", "Comedy"]
    }
    df = pd.DataFrame(data)
    df.attrs["dataset_name"] = "test_movies"
    return df


@pytest.fixture
def df_with_record_ids():
    """Create DataFrame with _id column."""
    data = {
        "_id": ["test_movies_000001", "test_movies_000002", "test_movies_000003"],
        "title": ["Movie A", "Movie B", "Movie C"],
        "year": [2020, 2021, 2022],
        "genre": ["Action", "Drama", "Comedy"]
    }
    df = pd.DataFrame(data)
    df.attrs["dataset_name"] = "test_movies"
    return df


@pytest.fixture
def df_with_missing_values():
    """Create DataFrame with missing values for testing comparators."""
    data = {
        "_id": ["test_1", "test_2", "test_3"],
        "title": ["Movie A", None, "Movie C"],
        "year": [2020, 2021, None],
        "rating": [7.5, None, 8.2]
    }
    df = pd.DataFrame(data)
    df.attrs["dataset_name"] = "test_missing"
    return df


@pytest.fixture
def sample_feature_data():
    """Create sample feature data for ML testing."""
    data = {
        "id1": ["academy_awards_000000", "academy_awards_000001", "academy_awards_000002"],
        "id2": ["actors_000000", "actors_000001", "actors_000002"],
        "title_similarity": [0.95, 0.87, 0.92],
        "year_similarity": [1.0, 1.0, 1.0],
        "genre_similarity": [0.8, 0.6, 0.9],
        "label": [1, 1, 1]
    }
    return pd.DataFrame(data)


@pytest.fixture
def temp_output_dir():
    """Create temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_trained_classifier():
    """Create a mock trained classifier for ML testing."""
    class MockClassifier:
        def __init__(self):
            self.feature_importances_ = np.array([0.4, 0.3, 0.3])
            
        def predict(self, X):
            # Simple mock: predict 1 if first feature > 0.5
            if hasattr(X, 'iloc'):
                return (X.iloc[:, 0] > 0.5).astype(int).values
            else:
                return (X[:, 0] > 0.5).astype(int)
            
        def predict_proba(self, X):
            predictions = self.predict(X)
            # Return probabilities: [prob_class_0, prob_class_1]
            probas = np.zeros((len(predictions), 2))
            probas[:, 1] = predictions * 0.8 + 0.1  # Probability of class 1
            probas[:, 0] = 1 - probas[:, 1]  # Probability of class 0
            return probas
    
    return MockClassifier()


@pytest.fixture
def candidate_pair_batches():
    """Create multiple batches of candidate pairs for testing."""
    batch1 = pd.DataFrame({
        "id1": ["academy_awards_000000", "academy_awards_000001"],
        "id2": ["actors_000000", "actors_000001"]
    })
    batch2 = pd.DataFrame({
        "id1": ["academy_awards_000002", "academy_awards_000003"], 
        "id2": ["actors_000002", "actors_000003"]
    })
    return [batch1, batch2]


@pytest.fixture
def sample_xml_data_movies():
    """Load actual movie data from XML files for integration testing."""
    try:
        import xml.etree.ElementTree as ET
        
        # Try to load academy awards data
        academy_path = Path("input/movies/entitymatching/data/academy_awards.xml")
        if academy_path.exists():
            tree = ET.parse(academy_path)
            root = tree.getroot()
            
            data = []
            for movie in root.findall('movie'):
                movie_data = {
                    "id": movie.find('id').text if movie.find('id') is not None else "",
                    "title": movie.find('title').text if movie.find('title') is not None else "",
                    "date": movie.find('date').text if movie.find('date') is not None else ""
                }
                
                # Extract director
                director_elem = movie.find('director')
                if director_elem is not None:
                    director_name = director_elem.find('name')
                    movie_data["director"] = director_name.text if director_name is not None else ""
                else:
                    movie_data["director"] = ""
                    
                # Extract first actor
                actors_elem = movie.find('actors')
                if actors_elem is not None:
                    first_actor = actors_elem.find('actor')
                    if first_actor is not None:
                        actor_name = first_actor.find('name')
                        movie_data["actor"] = actor_name.text if actor_name is not None else ""
                    else:
                        movie_data["actor"] = ""
                else:
                    movie_data["actor"] = ""
                    
                data.append(movie_data)
            
            df = pd.DataFrame(data)
            df.attrs["dataset_name"] = "academy_awards"
            return df
            
    except Exception:
        pass
    
    # Return empty DataFrame if file loading fails
    return pd.DataFrame(columns=["id", "title", "date", "director", "actor"])


@pytest.fixture
def sample_correspondence_files():
    """Load correspondence files for integration testing."""
    try:
        train_path = Path("input/movies/entitymatching/splits/gs_academy_awards_2_actors_training.csv")
        test_path = Path("input/movies/entitymatching/splits/gs_academy_awards_2_actors_test.csv")
        
        data = {}
        if train_path.exists():
            train_df = pd.read_csv(train_path, names=["id1", "id2", "label"])
            train_df["label"] = train_df["label"].map({"TRUE": 1, "FALSE": 0})
            data["train"] = train_df
            
        if test_path.exists():
            test_df = pd.read_csv(test_path, names=["id1", "id2", "label"])  
            test_df["label"] = test_df["label"].map({"TRUE": 1, "FALSE": 0})
            data["test"] = test_df
            
        return data if data else {"train": pd.DataFrame(), "test": pd.DataFrame()}
        
    except Exception:
        return {"train": pd.DataFrame(), "test": pd.DataFrame()}


@pytest.fixture
def preprocessing_function():
    """Sample preprocessing function for testing."""
    def preprocess_text(text: str) -> str:
        """Normalize text by lowercasing and removing punctuation."""
        if pd.isna(text):
            return ""
        return str(text).lower().strip()
    return preprocess_text


@pytest.fixture
def sample_comparator_functions():
    """Create sample comparator functions for testing."""
    def title_similarity(r1: pd.Series, r2: pd.Series) -> float:
        """Simple title similarity based on exact match."""
        t1 = str(r1.get('title', '')).lower()
        t2 = str(r2.get('title', '')).lower()
        return 1.0 if t1 == t2 else 0.0
    
    def year_similarity(r1: pd.Series, r2: pd.Series) -> float:
        """Year similarity based on exact match."""
        y1 = r1.get('year', 0)
        y2 = r2.get('year', 0)
        return 1.0 if y1 == y2 else 0.0
    
    return [title_similarity, year_similarity]