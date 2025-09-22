"""
Embedding-based blocking for entity matching using nearest neighbor search.

This module provides EmbeddingBlocker, which generates candidate pairs by computing
text embeddings and performing nearest neighbor search to find similar records.
"""

import logging
import os
from typing import Iterator, Optional, Union, Callable, Literal
import warnings

import numpy as np
import pandas as pd

from .base import BaseBlocker, CandidateBatch

logger = logging.getLogger(__name__)


class EmbeddingBlocker(BaseBlocker):
    """
    Embedding-based blocking using nearest neighbor search over text embeddings.
    
    This blocker combines text columns into a single string representation,
    computes embeddings using sentence transformers or a custom embedder,
    and uses nearest neighbor search to find candidate pairs above a similarity threshold.
    
    Parameters
    ----------
    df_left : pd.DataFrame
        Left dataset for blocking
    df_right : pd.DataFrame  
        Right dataset for blocking
    text_cols : list[str]
        Column names to use for text embedding (must exist in both datasets)
    model : str, default "sentence-transformers/all-MiniLM-L6-v2"
        Model name for sentence transformers embedding
    index_backend : {"sklearn", "faiss", "hnsw"}, default "sklearn"
        Backend to use for nearest neighbor index
    metric : {"cosine", "dot"}, default "cosine"
        Distance metric for similarity computation
    top_k : int, default 50
        Maximum number of nearest neighbors to retrieve per query
    threshold : float, default 0.3
        Minimum similarity threshold for candidate pairs (0-1 for cosine)
    normalize : bool, default True
        Whether to L2-normalize embeddings (recommended for cosine similarity)
    batch_size : int, default 25000
        Number of candidate pairs to yield per batch
    query_batch_size : int, default 2048
        Number of records to process at once during embedding/querying
    embedder : callable, optional
        Custom embedding function: embedder(df, text_cols) -> np.ndarray
    left_embeddings : np.ndarray, optional
        Pre-computed embeddings for left dataset
    right_embeddings : np.ndarray, optional
        Pre-computed embeddings for right dataset
    device : str, optional
        Device for embedding computation (e.g., "cpu", "cuda:0")
    """
    
    def __init__(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        text_cols: list[str],
        id_column: str,
        *,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_backend: Literal["sklearn", "faiss", "hnsw"] = "sklearn",
        metric: Literal["cosine", "dot"] = "cosine",
        top_k: int = 50,
        threshold: float = 0.3,
        normalize: bool = True,
        batch_size: int = 25_000,
        query_batch_size: int = 2048,
        embedder: Optional[Callable[[pd.DataFrame, list[str]], np.ndarray]] = None,
        left_embeddings: Optional[np.ndarray] = None,
        right_embeddings: Optional[np.ndarray] = None,
        device: Optional[str] = None,
        output_dir: str = "output",
        preprocess: Optional[Callable[[str], str]] = None,
    ):
        super().__init__(df_left, df_right, id_column, batch_size=batch_size)
        
        # Setup logging (consistent with other blockers)
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        
        # Validate parameters
        self._validate_params(text_cols, index_backend, metric, top_k, threshold, 
                            batch_size, query_batch_size, left_embeddings, right_embeddings)
        
        self.text_cols = text_cols
        self.model = model
        self.index_backend = index_backend
        self.metric = metric
        self.top_k = top_k
        self.threshold = threshold
        self.normalize = normalize
        self.batch_size = batch_size
        self.query_batch_size = query_batch_size
        self.embedder = embedder
        self.device = device
        self.output_dir = output_dir
        self.preprocess = preprocess
        
        # Initialize embeddings
        self.left_embeddings = left_embeddings
        self.right_embeddings = right_embeddings
        
        # Lazy-loaded components
        self._sentence_transformer = None
        self._nn_index = None
        
        self.logger.info(f"Initialized EmbeddingBlocker with {index_backend} backend, "
                        f"top_k={top_k}, threshold={threshold}")
        
        # Write debug file will be called during iteration when embeddings are ready
    
    def _validate_params(
        self,
        text_cols: list[str],
        index_backend: str,
        metric: str,
        top_k: int,
        threshold: float,
        batch_size: int,
        query_batch_size: int,
        left_embeddings: Optional[np.ndarray],
        right_embeddings: Optional[np.ndarray],
    ) -> None:
        """Validate initialization parameters."""
        if not text_cols:
            raise ValueError("text_cols cannot be empty")
            
        # Check text columns exist in both datasets
        missing_left = set(text_cols) - set(self.df_left.columns)
        missing_right = set(text_cols) - set(self.df_right.columns)
        
        if missing_left:
            raise ValueError(f"text_cols {missing_left} not found in left dataset")
        if missing_right:
            raise ValueError(f"text_cols {missing_right} not found in right dataset")
        
        # Validate numeric parameters
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not 0 <= threshold <= 1 and metric == "cosine":
            raise ValueError("threshold must be between 0 and 1 for cosine metric")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if query_batch_size <= 0:
            raise ValueError("query_batch_size must be positive")
            
        # Validate embeddings if provided
        if left_embeddings is not None:
            if left_embeddings.shape[0] != len(self.df_left):
                raise ValueError(f"left_embeddings shape {left_embeddings.shape} doesn't match "
                               f"left dataset length {len(self.df_left)}")
                               
        if right_embeddings is not None:
            if right_embeddings.shape[0] != len(self.df_right):
                raise ValueError(f"right_embeddings shape {right_embeddings.shape} doesn't match "
                               f"right dataset length {len(self.df_right)}")
                               
        if (left_embeddings is not None) != (right_embeddings is not None):
            raise ValueError("Both left_embeddings and right_embeddings must be provided together")
            
        if left_embeddings is not None and right_embeddings is not None:
            if left_embeddings.shape[1] != right_embeddings.shape[1]:
                raise ValueError(f"Embedding dimensions don't match: left={left_embeddings.shape[1]}, "
                               f"right={right_embeddings.shape[1]}")
    
    def _get_sentence_transformer(self):
        """Lazy-load sentence transformer model."""
        if self._sentence_transformer is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for embedding generation. "
                    "Install with: pip install sentence-transformers"
                )
            
            self._sentence_transformer = SentenceTransformer(self.model, device=self.device)
            self.logger.info(f"Loaded sentence transformer model: {self.model}")
            
        return self._sentence_transformer
    
    def _combine_text_columns(self, df: pd.DataFrame) -> list[str]:
        """Combine text columns into single strings for embedding."""
        combined_texts = []

        for _, row in df.iterrows():
            text_parts = []
            for col in self.text_cols:
                value = row[col]
                if pd.isna(value):
                    value = ""
                value_str = str(value)
                if self.preprocess:
                    value_str = self.preprocess(value_str)
                text_parts.append(value_str)
            combined_texts.append(" ".join(text_parts))

        return combined_texts
    
    def _compute_embeddings(self, df: pd.DataFrame) -> np.ndarray:
        """Compute embeddings for a dataset."""
        if self.embedder is not None:
            # Use custom embedder
            embeddings = self.embedder(df, self.text_cols)
        else:
            # Use sentence transformers
            transformer = self._get_sentence_transformer()
            texts = self._combine_text_columns(df)
            
            # Compute embeddings in batches
            all_embeddings = []
            for i in range(0, len(texts), self.query_batch_size):
                batch_texts = texts[i:i + self.query_batch_size]
                batch_embeddings = transformer.encode(
                    batch_texts, 
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
                all_embeddings.append(batch_embeddings)
            
            embeddings = np.vstack(all_embeddings)
        
        # Convert to float32 and normalize if requested
        embeddings = embeddings.astype(np.float32)
        
        if self.normalize:
            # L2 normalize
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)  # Avoid division by zero
            embeddings = embeddings / norms
            
        return embeddings
    
    def _ensure_embeddings(self) -> tuple[np.ndarray, np.ndarray]:
        """Ensure both left and right embeddings are computed."""
        if self.left_embeddings is None or self.right_embeddings is None:
            self.logger.debug("Computing embeddings for datasets...")
            
            if self.left_embeddings is None:
                self.logger.debug(f"Creating embeddings for dataset1: {len(self.df_left)} records")
                self.left_embeddings = self._compute_embeddings(self.df_left)
                self.logger.info(f"created {self.left_embeddings.shape[1]}d embeddings for first dataset")
                
            if self.right_embeddings is None:
                self.logger.debug(f"Creating embeddings for dataset2: {len(self.df_right)} records")
                self.right_embeddings = self._compute_embeddings(self.df_right)
                self.logger.info(f"created {self.right_embeddings.shape[1]}d embeddings for second dataset")
                
        return self.left_embeddings, self.right_embeddings
    
    def _build_nn_index(self, embeddings: np.ndarray):
        """Build nearest neighbor index for right-side embeddings."""
        if self.index_backend == "sklearn":
            return self._build_sklearn_index(embeddings)
        elif self.index_backend == "faiss":
            return self._build_faiss_index(embeddings)
        elif self.index_backend == "hnsw":
            return self._build_hnsw_index(embeddings)
        else:
            raise ValueError(f"Unknown index_backend: {self.index_backend}")
    
    def _build_sklearn_index(self, embeddings: np.ndarray):
        """Build sklearn NearestNeighbors index."""
        try:
            from sklearn.neighbors import NearestNeighbors
        except ImportError:
            raise ImportError("sklearn is required for sklearn backend")
            
        # Map metric names
        sklearn_metric = "cosine" if self.metric == "cosine" else "euclidean"
        
        index = NearestNeighbors(
            n_neighbors=min(self.top_k, len(embeddings)),
            metric=sklearn_metric,
            n_jobs=-1
        )
        index.fit(embeddings)
        
        self.logger.info(f"created similarity index with {len(embeddings)} vectors, metric={sklearn_metric}")
        return index
    
    def _build_faiss_index(self, embeddings: np.ndarray):
        """Build FAISS index."""
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "faiss-cpu is required for faiss backend. "
                "Install with: pip install faiss-cpu"
            )
            
        dim = embeddings.shape[1]
        
        if self.metric == "cosine" or (self.metric == "dot" and self.normalize):
            # Use inner product index (works for normalized vectors)
            index = faiss.IndexFlatIP(dim)
        else:
            # Use L2 distance
            index = faiss.IndexFlatL2(dim)
            
        index.add(embeddings)
        
        self.logger.info(f"created similarity index with {len(embeddings)} vectors, metric={self.metric}")
        return index
    
    def _build_hnsw_index(self, embeddings: np.ndarray):
        """Build HNSW index."""
        try:
            import hnswlib
        except ImportError:
            raise ImportError(
                "hnswlib is required for hnsw backend. "
                "Install with: pip install hnswlib"
            )
            
        dim = embeddings.shape[1]
        space = "cosine" if self.metric == "cosine" else "ip"
        
        index = hnswlib.Index(space=space, dim=dim)
        index.init_index(max_elements=len(embeddings), M=16, ef_construction=200)
        index.add_items(embeddings, np.arange(len(embeddings)))
        index.set_ef(50)  # Controls recall/speed tradeoff
        
        self.logger.info(f"created similarity index with {len(embeddings)} vectors, space={space}")
        return index
    
    def _query_nn_index(self, query_embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Query nearest neighbor index and return indices and similarities."""
        if self.index_backend == "sklearn":
            return self._query_sklearn_index(query_embeddings)
        elif self.index_backend == "faiss":
            return self._query_faiss_index(query_embeddings)
        elif self.index_backend == "hnsw":
            return self._query_hnsw_index(query_embeddings)
        else:
            raise ValueError(f"Unknown index_backend: {self.index_backend}")
    
    def _query_sklearn_index(self, query_embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Query sklearn index."""
        distances, indices = self._nn_index.kneighbors(query_embeddings)
        
        # Convert distances to similarities
        if self.metric == "cosine":
            similarities = 1 - distances
        else:  # euclidean for dot product
            similarities = -distances  # Higher distance = lower similarity
            
        return indices, similarities
    
    def _query_faiss_index(self, query_embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Query FAISS index."""
        scores, indices = self._nn_index.search(query_embeddings, min(self.top_k, self._nn_index.ntotal))
        
        # FAISS returns similarities (higher = better)
        similarities = scores
        
        return indices, similarities
    
    def _query_hnsw_index(self, query_embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Query HNSW index."""
        all_indices = []
        all_similarities = []
        
        for query_vec in query_embeddings:
            indices, distances = self._nn_index.knn_query(query_vec, k=min(self.top_k, self._nn_index.get_current_count()))
            
            # Convert distances to similarities based on space
            if self.metric == "cosine":
                similarities = 1 - distances
            else:  # inner product
                similarities = distances
                
            all_indices.append(indices)
            all_similarities.append(similarities)
        
        return np.array(all_indices), np.array(all_similarities)
    
    def _write_debug_file(self, left_embeddings: np.ndarray, right_embeddings: np.ndarray) -> None:
        """Write debug CSV file with embedding similarity statistics like Winter framework."""
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Sample a subset for debug analysis to avoid expensive computation
        sample_size = min(100, len(left_embeddings))
        sample_indices = np.random.choice(len(left_embeddings), size=sample_size, replace=False)
        sample_embeddings = left_embeddings[sample_indices]
        
        # Query for similarity distribution
        neighbor_indices, similarities = self._query_nn_index(sample_embeddings)
        
        # Create debug data with similarity ranges
        debug_data = []
        
        # Analyze similarity distribution in bins
        all_sims = similarities.flatten()
        valid_sims = all_sims[all_sims >= self.threshold]
        
        if len(valid_sims) > 0:
            # Create similarity bins
            sim_bins = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
            for i in range(len(sim_bins) - 1):
                low, high = sim_bins[i], sim_bins[i + 1]
                count = np.sum((valid_sims >= low) & (valid_sims < high))
                if count > 0:
                    range_str = f"{low:.2f}-{high:.2f}"
                    debug_data.append({"Blocking Key Value": range_str, "Frequency": int(count)})
        
        if not debug_data:
            debug_data.append({"Blocking Key Value": "no_similarities", "Frequency": 0})
        
        # Sort by similarity range (approximate ordering)
        debug_data.sort(key=lambda x: -x["Frequency"])
        
        # Write to CSV file
        debug_file = os.path.join(self.output_dir, "debugResultsBlocking_EmbeddingBlocker.csv")
        debug_df = pd.DataFrame(debug_data)
        debug_df.to_csv(debug_file, index=False)
        
        self.logger.info(f"Debug results written to file: {debug_file}")
    
    def _iter_batches(self) -> Iterator[CandidateBatch]:
        """Generate batches of candidate pairs."""
        # Ensure embeddings are computed
        left_embeddings, right_embeddings = self._ensure_embeddings()
        
        # Build NN index on right embeddings
        if self._nn_index is None:
            self.logger.debug(f"Building similarity index for {len(right_embeddings)} vectors")
            self._nn_index = self._build_nn_index(right_embeddings)
            
            # Write debug file after embeddings and index are ready
            self._write_debug_file(left_embeddings, right_embeddings)
        
        # Log DEBUG: Creating candidate pairs
        self.logger.debug(f"Creating candidate record pairs from embedding similarity with threshold {self.threshold}")
        
        # Accumulator for candidate pairs
        pairs_accumulator = []
        
        # Process left embeddings in batches
        for start_idx in range(0, len(left_embeddings), self.query_batch_size):
            end_idx = min(start_idx + self.query_batch_size, len(left_embeddings))
            query_batch = left_embeddings[start_idx:end_idx]
            
            # Query NN index
            neighbor_indices, similarities = self._query_nn_index(query_batch)
            
            # Process results
            for i, (neighbors, sims) in enumerate(zip(neighbor_indices, similarities)):
                left_idx = start_idx + i
                left_id = self._left_indexed.iloc[left_idx][self.id_column]
                
                # Filter by threshold
                valid_mask = sims >= self.threshold
                valid_neighbors = neighbors[valid_mask]
                
                # Add valid pairs to accumulator
                for right_idx in valid_neighbors:
                    if 0 <= right_idx < len(self._right_indexed):  # Safety check
                        right_id = self._right_indexed.iloc[right_idx][self.id_column]
                        pairs_accumulator.append((left_id, right_id))
                        
                        # Emit batch if we've accumulated enough pairs
                        if len(pairs_accumulator) >= self.batch_size:
                            batch_df = pd.DataFrame(pairs_accumulator, columns=["id1", "id2"])
                            yield self._emit_batch(batch_df)
                            pairs_accumulator.clear()
        
        # Emit remaining pairs
        if pairs_accumulator:
            batch_df = pd.DataFrame(pairs_accumulator, columns=["id1", "id2"])
            yield self._emit_batch(batch_df)
        
    
    def estimate_pairs(self) -> Optional[int]:
        """Estimate number of candidate pairs using sampling."""
        if len(self.df_left) == 0 or len(self.df_right) == 0:
            return 0
            
        # Ensure embeddings are computed
        left_embeddings, right_embeddings = self._ensure_embeddings()
        
        # Build NN index if needed
        if self._nn_index is None:
            self._nn_index = self._build_nn_index(right_embeddings)
        
        # Sample from left dataset
        sample_size = min(1000, len(left_embeddings))
        sample_indices = np.random.choice(len(left_embeddings), size=sample_size, replace=False)
        sample_embeddings = left_embeddings[sample_indices]
        
        # Query for samples
        neighbor_indices, similarities = self._query_nn_index(sample_embeddings)
        
        # Count valid neighbors per sample
        total_valid_neighbors = 0
        for sims in similarities:
            total_valid_neighbors += np.sum(sims >= self.threshold)
        
        if sample_size == 0:
            return None
            
        # Extrapolate to full dataset
        avg_neighbors_per_query = total_valid_neighbors / sample_size
        estimated_pairs = int(avg_neighbors_per_query * len(self.df_left))
        
        self.logger.debug(f"Estimated {estimated_pairs} candidate pairs from {sample_size} samples")
        return estimated_pairs


__all__ = ["EmbeddingBlocker"]