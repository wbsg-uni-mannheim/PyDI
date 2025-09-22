"""
Tests for EmbeddingBlocking class.

This module tests the embedding-based blocking functionality including
initialization, validation, embedding generation, and nearest neighbor search.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock

from PyDI.entitymatching.blocking.embedding import EmbeddingBlocking


@pytest.fixture
def sample_data():
    """Create sample datasets for testing."""
    df_left = pd.DataFrame({
        "title": ["The Matrix", "The Matrix Reloaded", "Star Wars", "Avatar"],
        "description": ["Sci-fi movie", "Sequel to Matrix", "Space opera", "Blue aliens"],
        "_id": ["l1", "l2", "l3", "l4"]
    })
    
    df_right = pd.DataFrame({
        "title": ["Matrix", "The Matrix 2", "Star Trek", "Avatar Movie"],
        "description": ["Cyberpunk film", "Matrix sequel", "Space adventure", "Pandora story"],
        "_id": ["r1", "r2", "r3", "r4"]
    })
    
    return df_left, df_right


@pytest.fixture
def small_embeddings():
    """Create small test embeddings."""
    left_embeddings = np.array([
        [0.1, 0.2, 0.3],
        [0.15, 0.25, 0.35],
        [0.9, 0.1, 0.1],
        [0.2, 0.8, 0.1]
    ], dtype=np.float32)
    
    right_embeddings = np.array([
        [0.12, 0.22, 0.32],  # Similar to left[0]
        [0.16, 0.26, 0.36],  # Similar to left[1]  
        [0.8, 0.15, 0.15],   # Similar to left[2]
        [0.18, 0.82, 0.12]   # Similar to left[3]
    ], dtype=np.float32)
    
    return left_embeddings, right_embeddings


class TestEmbeddingBlockingInit:
    """Test EmbeddingBlocking initialization and validation."""
    
    def test_basic_initialization(self, sample_data):
        """Test basic initialization with default parameters."""
        df_left, df_right = sample_data
        
        blocker = EmbeddingBlocking(
            df_left, df_right,
            text_cols=["title", "description"]
        )
        
        assert blocker.text_cols == ["title", "description"]
        assert blocker.model == "sentence-transformers/all-MiniLM-L6-v2"
        assert blocker.index_backend == "sklearn"
        assert blocker.metric == "cosine"
        assert blocker.top_k == 50
        assert blocker.threshold == 0.3
        assert blocker.normalize is True
        assert blocker.batch_size == 25_000
        assert blocker.query_batch_size == 2048
    
    def test_custom_parameters(self, sample_data):
        """Test initialization with custom parameters."""
        df_left, df_right = sample_data
        
        blocker = EmbeddingBlocking(
            df_left, df_right,
            text_cols=["title"],
            model="custom-model",
            index_backend="faiss",
            metric="dot",
            top_k=10,
            threshold=0.5,
            normalize=False,
            batch_size=1000,
            query_batch_size=100
        )
        
        assert blocker.text_cols == ["title"]
        assert blocker.model == "custom-model"
        assert blocker.index_backend == "faiss"
        assert blocker.metric == "dot"
        assert blocker.top_k == 10
        assert blocker.threshold == 0.5
        assert blocker.normalize is False
        assert blocker.batch_size == 1000
        assert blocker.query_batch_size == 100
    
    def test_empty_text_cols_error(self, sample_data):
        """Test error when text_cols is empty."""
        df_left, df_right = sample_data
        
        with pytest.raises(ValueError, match="text_cols cannot be empty"):
            EmbeddingBlocking(df_left, df_right, text_cols=[])
    
    def test_missing_columns_error(self, sample_data):
        """Test error when text_cols don't exist in datasets."""
        df_left, df_right = sample_data
        
        with pytest.raises(ValueError, match="not found in left dataset"):
            EmbeddingBlocking(df_left, df_right, text_cols=["missing_col"])
        
        # Remove column from right dataset
        df_right_missing = df_right.drop(columns=["title"])
        with pytest.raises(ValueError, match="not found in right dataset"):
            EmbeddingBlocking(df_left, df_right_missing, text_cols=["title"])
    
    def test_invalid_parameters_error(self, sample_data):
        """Test error with invalid parameters."""
        df_left, df_right = sample_data
        
        with pytest.raises(ValueError, match="top_k must be positive"):
            EmbeddingBlocking(df_left, df_right, text_cols=["title"], top_k=0)
        
        with pytest.raises(ValueError, match="threshold must be between 0 and 1"):
            EmbeddingBlocking(df_left, df_right, text_cols=["title"], threshold=-0.1)
        
        with pytest.raises(ValueError, match="batch_size must be positive"):
            EmbeddingBlocking(df_left, df_right, text_cols=["title"], batch_size=-1)
    
    def test_precomputed_embeddings_validation(self, sample_data, small_embeddings):
        """Test validation of precomputed embeddings."""
        df_left, df_right = sample_data
        left_embeddings, right_embeddings = small_embeddings
        
        # Test mismatched shapes
        wrong_shape = np.random.random((2, 3)).astype(np.float32)
        with pytest.raises(ValueError, match="doesn't match left dataset length"):
            EmbeddingBlocking(df_left, df_right, text_cols=["title"], left_embeddings=wrong_shape)
        
        # Test only one embedding provided
        with pytest.raises(ValueError, match="Both left_embeddings and right_embeddings must be provided"):
            EmbeddingBlocking(df_left, df_right, text_cols=["title"], left_embeddings=left_embeddings)
        
        # Test mismatched dimensions
        wrong_dim = np.random.random((4, 5)).astype(np.float32)
        with pytest.raises(ValueError, match="Embedding dimensions don't match"):
            EmbeddingBlocking(df_left, df_right, text_cols=["title"], 
                            left_embeddings=left_embeddings, right_embeddings=wrong_dim)
    
    def test_valid_precomputed_embeddings(self, sample_data, small_embeddings):
        """Test valid precomputed embeddings."""
        df_left, df_right = sample_data
        left_embeddings, right_embeddings = small_embeddings
        
        blocker = EmbeddingBlocking(
            df_left, df_right,
            text_cols=["title"],
            left_embeddings=left_embeddings,
            right_embeddings=right_embeddings
        )
        
        assert np.array_equal(blocker.left_embeddings, left_embeddings)
        assert np.array_equal(blocker.right_embeddings, right_embeddings)


class TestEmbeddingGeneration:
    """Test embedding generation functionality."""
    
    def test_combine_text_columns(self, sample_data):
        """Test text column combination."""
        df_left, df_right = sample_data
        
        blocker = EmbeddingBlocking(df_left, df_right, text_cols=["title", "description"])
        
        combined = blocker._combine_text_columns(df_left)
        expected = [
            "The Matrix Sci-fi movie",
            "The Matrix Reloaded Sequel to Matrix", 
            "Star Wars Space opera",
            "Avatar Blue aliens"
        ]
        
        assert combined == expected
    
    def test_combine_text_with_missing_values(self):
        """Test text combination with missing values."""
        df = pd.DataFrame({
            "title": ["Movie A", None, "Movie C"],
            "description": ["Desc A", "Desc B", None],
            "_id": ["1", "2", "3"]
        })
        
        blocker = EmbeddingBlocking(df, df, text_cols=["title", "description"])
        combined = blocker._combine_text_columns(df)
        
        expected = ["Movie A Desc A", " Desc B", "Movie C "]
        assert combined == expected
    
    @patch('PyDI.entitymatching.blocking.embedding.EmbeddingBlocking._get_sentence_transformer')
    def test_compute_embeddings_with_sentence_transformer(self, mock_get_transformer, sample_data):
        """Test embedding computation using sentence transformers."""
        df_left, df_right = sample_data
        
        # Mock sentence transformer
        mock_transformer = Mock()
        mock_transformer.encode.return_value = np.random.random((4, 384)).astype(np.float32)
        mock_get_transformer.return_value = mock_transformer
        
        blocker = EmbeddingBlocking(df_left, df_right, text_cols=["title"])
        embeddings = blocker._compute_embeddings(df_left)
        
        assert embeddings.shape == (4, 384)
        assert embeddings.dtype == np.float32
        mock_transformer.encode.assert_called_once()
    
    def test_compute_embeddings_with_custom_embedder(self, sample_data):
        """Test embedding computation with custom embedder."""
        df_left, df_right = sample_data
        
        def custom_embedder(df, text_cols):
            return np.random.random((len(df), 100)).astype(np.float32)
        
        blocker = EmbeddingBlocking(df_left, df_right, text_cols=["title"], embedder=custom_embedder)
        embeddings = blocker._compute_embeddings(df_left)
        
        assert embeddings.shape == (4, 100)
        assert embeddings.dtype == np.float32
    
    def test_normalization(self, sample_data, small_embeddings):
        """Test embedding normalization."""
        df_left, df_right = sample_data
        left_embeddings, _ = small_embeddings
        
        blocker = EmbeddingBlocking(df_left, df_right, text_cols=["title"], normalize=True)
        
        # Mock the embedder to return unnormalized embeddings  
        def mock_embedder(df, text_cols):
            return left_embeddings
        
        blocker.embedder = mock_embedder
        result = blocker._compute_embeddings(df_left)
            
        # Check that vectors are normalized
        result_norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(result_norms, 1.0, rtol=1e-6)


class TestNearestNeighborIndexing:
    """Test nearest neighbor indexing backends."""
    
    def test_sklearn_index_creation(self, sample_data, small_embeddings):
        """Test sklearn index creation."""
        df_left, df_right = sample_data
        _, right_embeddings = small_embeddings
        
        with patch('sklearn.neighbors.NearestNeighbors') as mock_nn:
            mock_index = Mock()
            mock_nn.return_value = mock_index
            
            blocker = EmbeddingBlocking(df_left, df_right, text_cols=["title"], 
                                      index_backend="sklearn", metric="cosine")
            result = blocker._build_sklearn_index(right_embeddings)
            
            mock_nn.assert_called_once_with(
                n_neighbors=4,  # min(top_k=50, len(embeddings)=4)
                metric="cosine",
                n_jobs=-1
            )
            mock_index.fit.assert_called_once_with(right_embeddings)
            assert result == mock_index
    
    @patch('faiss.IndexFlatIP')
    def test_faiss_index_creation(self, mock_faiss_index, sample_data, small_embeddings):
        """Test FAISS index creation."""
        df_left, df_right = sample_data
        _, right_embeddings = small_embeddings
        
        mock_index = Mock()
        mock_faiss_index.return_value = mock_index
        
        blocker = EmbeddingBlocking(df_left, df_right, text_cols=["title"],
                                  index_backend="faiss", metric="cosine")
        result = blocker._build_faiss_index(right_embeddings)
        
        mock_faiss_index.assert_called_once_with(3)  # embedding dimension
        mock_index.add.assert_called_once_with(right_embeddings)
        assert result == mock_index
    
    @patch('hnswlib.Index')
    def test_hnsw_index_creation(self, mock_hnsw_index, sample_data, small_embeddings):
        """Test HNSW index creation."""
        df_left, df_right = sample_data
        _, right_embeddings = small_embeddings
        
        mock_index = Mock()
        mock_hnsw_index.return_value = mock_index
        
        blocker = EmbeddingBlocking(df_left, df_right, text_cols=["title"],
                                  index_backend="hnsw", metric="cosine")
        result = blocker._build_hnsw_index(right_embeddings)
        
        mock_hnsw_index.assert_called_once_with(space="cosine", dim=3)
        mock_index.init_index.assert_called_once_with(max_elements=4, M=16, ef_construction=200)
        mock_index.add_items.assert_called_once()
        mock_index.set_ef.assert_called_once_with(50)
        assert result == mock_index
    
    def test_unknown_backend_error(self, sample_data, small_embeddings):
        """Test error with unknown backend."""
        df_left, df_right = sample_data
        _, right_embeddings = small_embeddings
        
        blocker = EmbeddingBlocking(df_left, df_right, text_cols=["title"],
                                  index_backend="unknown")
        
        with pytest.raises(ValueError, match="Unknown index_backend: unknown"):
            blocker._build_nn_index(right_embeddings)


class TestBlockingExecution:
    """Test the main blocking execution."""
    
    def test_iter_batches_with_precomputed_embeddings(self, sample_data, small_embeddings):
        """Test iteration with precomputed embeddings."""
        df_left, df_right = sample_data
        left_embeddings, right_embeddings = small_embeddings
        
        # Mock sklearn index to return predictable results
        with patch('sklearn.neighbors.NearestNeighbors') as mock_nn:
            mock_index = Mock()
            # Return indices and distances for each query
            mock_index.kneighbors.return_value = (
                np.array([[0.1, 0.2], [0.15, 0.25], [0.3, 0.4], [0.2, 0.3]]),  # distances
                np.array([[0, 1], [1, 0], [2, 3], [3, 2]])  # indices
            )
            mock_nn.return_value = mock_index
            
            blocker = EmbeddingBlocking(
                df_left, df_right,
                text_cols=["title"],
                left_embeddings=left_embeddings,
                right_embeddings=right_embeddings,
                index_backend="sklearn",
                threshold=0.7,  # 1 - 0.3 = 0.7, so distances <= 0.3 pass
                batch_size=10
            )
            
            batches = list(blocker._iter_batches())
            
            assert len(batches) >= 1
            assert all("id1" in batch.columns and "id2" in batch.columns for batch in batches)
    
    def test_empty_datasets(self):
        """Test with empty datasets."""
        df_left = pd.DataFrame({"title": [], "_id": []})
        df_right = pd.DataFrame({"title": [], "_id": []})
        
        blocker = EmbeddingBlocking(df_left, df_right, text_cols=["title"])
        
        # Should handle empty datasets gracefully
        batches = list(blocker._iter_batches())
        assert len(batches) == 0
    
    def test_estimate_pairs_with_precomputed(self, sample_data, small_embeddings):
        """Test pair estimation with precomputed embeddings."""
        df_left, df_right = sample_data
        left_embeddings, right_embeddings = small_embeddings
        
        with patch('sklearn.neighbors.NearestNeighbors') as mock_nn:
            mock_index = Mock()
            # Mock return values for sampling
            mock_index.kneighbors.return_value = (
                np.array([[0.1, 0.2]]),  # distances 
                np.array([[0, 1]])       # indices
            )
            mock_nn.return_value = mock_index
            
            blocker = EmbeddingBlocking(
                df_left, df_right,
                text_cols=["title"],
                left_embeddings=left_embeddings,
                right_embeddings=right_embeddings,
                threshold=0.7  # 1 - 0.3 = 0.7
            )
            
            estimate = blocker.estimate_pairs()
            
            assert isinstance(estimate, int)
            assert estimate >= 0
    
    def test_estimate_pairs_empty_datasets(self):
        """Test pair estimation with empty datasets."""
        df_left = pd.DataFrame({"title": [], "_id": []})
        df_right = pd.DataFrame({"title": [], "_id": []})
        
        blocker = EmbeddingBlocking(df_left, df_right, text_cols=["title"])
        
        estimate = blocker.estimate_pairs()
        assert estimate == 0


class TestIntegration:
    """Integration tests with real components."""
    
    @pytest.mark.skipif(
        not pytest.importorskip("sklearn.neighbors", reason="sklearn not available"),
        reason="sklearn not available"
    )
    def test_sklearn_integration(self, sample_data):
        """Test integration with sklearn backend."""
        df_left, df_right = sample_data
        
        # Create simple mock embeddings
        left_emb = np.random.random((4, 10)).astype(np.float32)
        right_emb = np.random.random((4, 10)).astype(np.float32)
        
        blocker = EmbeddingBlocking(
            df_left, df_right,
            text_cols=["title"],
            left_embeddings=left_emb,
            right_embeddings=right_emb,
            index_backend="sklearn",
            top_k=2,
            threshold=0.0,  # Accept all pairs
            batch_size=5
        )
        
        # Should be able to iterate without errors
        batches = list(blocker._iter_batches())
        
        # Check basic structure
        if batches:
            batch = batches[0]
            assert "id1" in batch.columns
            assert "id2" in batch.columns
            assert len(batch) > 0
    
    def test_materialize_and_stats(self, sample_data, small_embeddings):
        """Test materialize and stats methods inherited from BaseBlocker."""
        df_left, df_right = sample_data
        left_embeddings, right_embeddings = small_embeddings
        
        with patch('sklearn.neighbors.NearestNeighbors') as mock_nn:
            mock_index = Mock()
            mock_index.kneighbors.return_value = (
                np.array([[0.1], [0.2], [0.3], [0.4]]),  # distances
                np.array([[0], [1], [2], [3]])            # indices
            )
            mock_nn.return_value = mock_index
            
            blocker = EmbeddingBlocking(
                df_left, df_right,
                text_cols=["title"],
                left_embeddings=left_embeddings,
                right_embeddings=right_embeddings,
                threshold=0.5  # 1 - 0.5 = 0.5
            )
            
            # Test materialize
            result = blocker.materialize()
            assert isinstance(result, pd.DataFrame)
            assert "id1" in result.columns
            assert "id2" in result.columns
            
            # Test stats
            stats = blocker.stats()
            assert isinstance(stats, dict)
            assert "total_pairs" in stats