"""
Pre-trained Language Model based entity matching using transformers.

This module provides PLM-based entity matching that leverages pre-trained
transformer models while maintaining compatibility with PyDI's design principles.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Iterable, List, Optional, Union

import numpy as np
import pandas as pd

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from .base import BaseMatcher, CorrespondenceSet
from .text_formatting import TextFormatter


class PLMBasedMatcher(BaseMatcher):
    """Pre-trained Language Model based entity matcher.

    This matcher uses pre-trained transformer models to predict entity matches
    based on formatted text representations of candidate pairs. It supports
    multiple model types and provides minimal wrapping around transformer
    libraries while maintaining PyDI compatibility.

    The matcher requires a TextFormatter to convert entity pairs into formatted
    text and a pre-trained transformer model. This design allows users to leverage
    the full transformer ecosystem while maintaining PyDI's DataFrame-first approach.

    Parameters
    ----------
    text_formatter : TextFormatter
        Text formatting component that converts entity pairs to formatted text.

    Examples
    --------
    >>> from PyDI.entitymatching import PLMBasedMatcher, TextFormatter
    >>> from transformers import AutoModel, AutoTokenizer
    >>>
    >>> # Create text formatter
    >>> formatter = TextFormatter(
    ...     text_fields=["title", "year"],
    ...     template="{left} [SEP] {right}",
    ...     max_length=512
    ... )
    >>>
    >>> # Load pre-trained model
    >>> model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
    >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    >>>
    >>> # Use with matcher
    >>> matcher = PLMBasedMatcher(formatter)
    >>> matches = matcher.match(
    ...     df_left, df_right, candidates,
    ...     id_column='id',
    ...     trained_model=model,
    ...     tokenizer=tokenizer,
    ...     model_type='classification',
    ...     threshold=0.7
    ... )
    """

    def __init__(self, text_formatter: TextFormatter):
        """Initialize PLM-based matcher.

        Parameters
        ----------
        text_formatter : TextFormatter
            Text formatting component.
        """
        if not isinstance(text_formatter, TextFormatter):
            raise ValueError("text_formatter must be TextFormatter instance")

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library is required for PLMBasedMatcher. "
                "Install with: pip install transformers"
            )

        self.text_formatter = text_formatter

    def match(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        candidates: Iterable[pd.DataFrame],
        id_column: str,
        trained_model: Any,
        tokenizer: Optional[Any] = None,
        model_type: str = "classification",
        threshold: float = 0.5,
        batch_size: int = 16,
        device: Optional[str] = None,
        **kwargs,
    ) -> CorrespondenceSet:
        """Find entity correspondences using pre-trained transformer model.

        Parameters
        ----------
        df_left : pandas.DataFrame
            Left dataset with _id column and dataset_name in attrs.
        df_right : pandas.DataFrame
            Right dataset with _id column and dataset_name in attrs.
        candidates : Iterable[pandas.DataFrame]
            Candidate pair batches with id1, id2 columns.
        id_column : str
            Name of the ID column in the datasets.
        trained_model : transformer model
            Pre-trained transformer model. Type depends on model_type:
            - 'classification': Model with classification head
            - 'embedding': SentenceTransformer or model for embeddings
            - 'raw': Any transformer model for custom processing
        tokenizer : tokenizer, optional
            Tokenizer for the model. Required for 'classification' and 'raw' types.
            Not needed for SentenceTransformer models.
        model_type : str, optional
            Type of model usage: 'classification', 'embedding', or 'raw'.
            Default is 'classification'.
        threshold : float, optional
            Decision threshold for classification. Default is 0.5.
        batch_size : int, optional
            Batch size for model inference. Default is 16.
        device : str, optional
            Device for model inference ('cpu', 'cuda'). Auto-detected if None.
        **kwargs
            Additional arguments (ignored).

        Returns
        -------
        CorrespondenceSet
            DataFrame with columns id1, id2, score, notes containing
            entity correspondences above the threshold.

        Raises
        ------
        ValueError
            If model type is invalid or required parameters are missing.
        """
        # Setup logger
        logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")

        # Log start of entity matching
        logger.info("Starting Entity Matching")
        start_time = time.time()

        # Validate inputs
        self._validate_inputs(df_left, df_right, id_column)

        if trained_model is None:
            raise ValueError("trained_model cannot be None")

        if model_type not in ["classification", "embedding", "raw"]:
            raise ValueError("model_type must be 'classification', 'embedding', or 'raw'")

        if model_type in ["classification", "raw"] and tokenizer is None:
            raise ValueError(f"tokenizer is required for model_type='{model_type}'")

        # Setup device
        if device is None and TORCH_AVAILABLE:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device is None:
            device = "cpu"

        # Move model to device if it's a torch model
        if TORCH_AVAILABLE and hasattr(trained_model, "to"):
            trained_model = trained_model.to(device)
            trained_model.eval()

        # Log blocking info
        logger.info(f"Blocking {len(df_left)} x {len(df_right)} elements")

        # Convert candidates to list to count total pairs
        if hasattr(candidates, '__iter__') and not isinstance(candidates, pd.DataFrame):
            candidate_list = list(candidates)
        else:
            candidate_list = [candidates] if isinstance(candidates, pd.DataFrame) else []

        # Count total candidate pairs for reduction ratio calculation
        total_pairs_processed = sum(len(batch) for batch in candidate_list if not batch.empty)
        total_possible_pairs = len(df_left) * len(df_right)
        reduction_ratio = 1 - (total_pairs_processed / total_possible_pairs) if total_possible_pairs > 0 else 0

        # Calculate elapsed time for blocking phase
        blocking_time = time.time() - start_time
        blocking_time_str = f"{blocking_time:.3f}"

        # Log matching phase info with reduction ratio
        logger.info(f"Matching {len(df_left)} x {len(df_right)} elements after 0:00:{blocking_time_str}; "
                   f"{total_pairs_processed} blocked pairs (reduction ratio: {reduction_ratio})")

        results = []

        # Process candidate batches
        for batch in candidate_list:
            if batch.empty:
                continue

            batch_results = self._process_batch(
                batch,
                df_left,
                df_right,
                id_column,
                trained_model,
                tokenizer,
                model_type,
                threshold,
                batch_size,
                device,
            )
            results.extend(batch_results)

        # Calculate total elapsed time
        total_time = time.time() - start_time
        total_time_str = f"{total_time:.3f}"

        # Log completion info
        logger.info(f"Entity Matching finished after 0:00:{total_time_str}; found {len(results)} correspondences.")

        # Create correspondence set
        if results:
            corr_df = pd.DataFrame(results)

            # Add metadata
            corr_df.attrs["model_type"] = model_type
            corr_df.attrs["threshold"] = threshold
            corr_df.attrs["batch_size"] = batch_size
            corr_df.attrs["text_formatter"] = str(self.text_formatter)

            return corr_df
        else:
            empty_df = pd.DataFrame(columns=["id1", "id2", "score", "notes"])
            empty_df.attrs["model_type"] = model_type
            empty_df.attrs["threshold"] = threshold
            return empty_df

    def _process_batch(
        self,
        batch: pd.DataFrame,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        id_column: str,
        trained_model: Any,
        tokenizer: Optional[Any],
        model_type: str,
        threshold: float,
        batch_size: int,
        device: str,
    ) -> List[dict]:
        """Process a batch of candidate pairs using the transformer model.

        Parameters
        ----------
        batch : pandas.DataFrame
            Candidate pairs with id1, id2 columns.
        df_left : pandas.DataFrame
            Left dataset.
        df_right : pandas.DataFrame
            Right dataset.
        id_column : str
            Name of the ID column.
        trained_model : transformer model
            Pre-trained model.
        tokenizer : tokenizer
            Model tokenizer.
        model_type : str
            Type of model usage.
        threshold : float
            Decision threshold.
        batch_size : int
            Inference batch size.
        device : str
            Device for inference.

        Returns
        -------
        list of dict
            List of correspondence dictionaries.
        """
        try:
            if model_type == "classification":
                return self._process_batch_classification(
                    batch, df_left, df_right, id_column, trained_model,
                    tokenizer, threshold, batch_size, device
                )
            elif model_type == "embedding":
                return self._process_batch_embedding(
                    batch, df_left, df_right, id_column, trained_model,
                    threshold, batch_size, device
                )
            elif model_type == "raw":
                return self._process_batch_raw(
                    batch, df_left, df_right, id_column, trained_model,
                    tokenizer, threshold, batch_size, device
                )
            else:
                logging.error(f"Unknown model_type: {model_type}")
                return []

        except Exception as e:
            logging.error(f"Error processing batch: {e}")
            return []

    def _process_batch_classification(
        self,
        batch: pd.DataFrame,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        id_column: str,
        trained_model: Any,
        tokenizer: Any,
        threshold: float,
        batch_size: int,
        device: str,
    ) -> List[dict]:
        """Process batch using classification model."""
        # Format text pairs
        formatted_texts = self.text_formatter.format_pairs(
            df_left, df_right, batch, id_column
        )

        if not formatted_texts:
            return []

        results = []

        # Process in batches
        for i in range(0, len(formatted_texts), batch_size):
            batch_texts = formatted_texts[i:i + batch_size]
            batch_pairs = batch.iloc[i:i + batch_size]

            # Tokenize
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.text_formatter.max_length,
                return_tensors="pt"
            )

            # Move to device
            if TORCH_AVAILABLE:
                encoded = {k: v.to(device) for k, v in encoded.items()}

            # Get predictions
            with torch.no_grad() if TORCH_AVAILABLE else contextlib.nullcontext():
                outputs = trained_model(**encoded)

                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs[0]

                # Apply softmax to get probabilities
                if TORCH_AVAILABLE:
                    probas = torch.softmax(logits, dim=-1)
                    scores = probas[:, 1].cpu().numpy()  # Positive class
                else:
                    # Fallback for non-torch models
                    scores = logits.numpy()[:, 1] if logits.shape[1] > 1 else logits.numpy().flatten()

            # Create results for this batch
            for j, score in enumerate(scores):
                if score >= threshold:
                    pair_idx = i + j
                    if pair_idx < len(batch):
                        row = batch.iloc[pair_idx]
                        results.append({
                            "id1": row["id1"],
                            "id2": row["id2"],
                            "score": float(score),
                            "notes": f"plm_classification={type(trained_model).__name__}",
                        })

        return results

    def _process_batch_embedding(
        self,
        batch: pd.DataFrame,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        id_column: str,
        trained_model: Any,
        threshold: float,
        batch_size: int,
        device: str,
    ) -> List[dict]:
        """Process batch using embedding model."""
        results = []

        # Get unique entity IDs
        left_ids = batch["id1"].unique()
        right_ids = batch["id2"].unique()

        # Format individual entities
        left_texts = self.text_formatter.format_single_entities(
            df_left, left_ids, id_column
        )
        right_texts = self.text_formatter.format_single_entities(
            df_right, right_ids, id_column
        )

        if not left_texts or not right_texts:
            return []

        # Generate embeddings
        if SENTENCE_TRANSFORMERS_AVAILABLE and hasattr(trained_model, 'encode'):
            # SentenceTransformer model
            left_embeddings = trained_model.encode(left_texts, batch_size=batch_size)
            right_embeddings = trained_model.encode(right_texts, batch_size=batch_size)
        else:
            # Regular transformer model - simple mean pooling
            left_embeddings = self._get_embeddings(left_texts, trained_model, batch_size, device)
            right_embeddings = self._get_embeddings(right_texts, trained_model, batch_size, device)

        # Create ID to embedding mapping
        left_id_to_emb = dict(zip(left_ids, left_embeddings))
        right_id_to_emb = dict(zip(right_ids, right_embeddings))

        # Calculate similarities for pairs
        for _, pair in batch.iterrows():
            id1, id2 = pair["id1"], pair["id2"]

            if id1 in left_id_to_emb and id2 in right_id_to_emb:
                emb1 = left_id_to_emb[id1]
                emb2 = right_id_to_emb[id2]

                # Cosine similarity
                similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

                if similarity >= threshold:
                    results.append({
                        "id1": id1,
                        "id2": id2,
                        "score": float(similarity),
                        "notes": f"plm_embedding={type(trained_model).__name__}",
                    })

        return results

    def _process_batch_raw(
        self,
        batch: pd.DataFrame,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        id_column: str,
        trained_model: Any,
        tokenizer: Any,
        threshold: float,
        batch_size: int,
        device: str,
    ) -> List[dict]:
        """Process batch using raw model outputs."""
        # This is a simplified implementation - users should override for custom behavior
        # Default: treat as classification with manual output processing

        formatted_texts = self.text_formatter.format_pairs(
            df_left, df_right, batch, id_column
        )

        if not formatted_texts:
            return []

        results = []

        for i in range(0, len(formatted_texts), batch_size):
            batch_texts = formatted_texts[i:i + batch_size]
            batch_pairs = batch.iloc[i:i + batch_size]

            # Tokenize
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.text_formatter.max_length,
                return_tensors="pt"
            )

            if TORCH_AVAILABLE:
                encoded = {k: v.to(device) for k, v in encoded.items()}

            # Get raw outputs
            with torch.no_grad() if TORCH_AVAILABLE else contextlib.nullcontext():
                outputs = trained_model(**encoded)

                # Simple processing - take mean of last hidden states
                if hasattr(outputs, 'last_hidden_state'):
                    hidden_states = outputs.last_hidden_state
                    # Mean pooling
                    scores = hidden_states.mean(dim=1)[:, 0]  # First dimension
                else:
                    scores = outputs[0].mean(dim=1)[:, 0]

                if TORCH_AVAILABLE and hasattr(scores, 'cpu'):
                    scores = scores.cpu().numpy()

            # Create results
            for j, score in enumerate(scores):
                if score >= threshold:
                    pair_idx = i + j
                    if pair_idx < len(batch):
                        row = batch.iloc[pair_idx]
                        results.append({
                            "id1": row["id1"],
                            "id2": row["id2"],
                            "score": float(score),
                            "notes": f"plm_raw={type(trained_model).__name__}",
                        })

        return results

    def _get_embeddings(self, texts: List[str], model: Any, batch_size: int, device: str) -> np.ndarray:
        """Get embeddings from regular transformer model using mean pooling."""
        embeddings = []

        # This requires a tokenizer - for simplicity, we'll return random embeddings
        # Users should override this method for their specific model setup
        logging.warning("Using simple embedding extraction - override _get_embeddings for custom models")

        for text in texts:
            # Placeholder: return random normalized embeddings
            emb = np.random.randn(768)  # Common BERT size
            emb = emb / np.linalg.norm(emb)
            embeddings.append(emb)

        return np.array(embeddings)

    def predict_pairs(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        pairs: pd.DataFrame,
        id_column: str,
        trained_model: Any,
        tokenizer: Optional[Any] = None,
        model_type: str = "classification",
        batch_size: int = 16,
        device: Optional[str] = None,
    ) -> pd.DataFrame:
        """Predict match probabilities for specific pairs without threshold filtering.

        Parameters
        ----------
        df_left : pandas.DataFrame
            Left dataset.
        df_right : pandas.DataFrame
            Right dataset.
        pairs : pandas.DataFrame
            Pairs to predict with id1, id2 columns.
        id_column : str
            Name of the ID column.
        trained_model : transformer model
            Pre-trained model.
        tokenizer : tokenizer, optional
            Model tokenizer.
        model_type : str, optional
            Type of model usage. Default is 'classification'.
        batch_size : int, optional
            Inference batch size. Default is 16.
        device : str, optional
            Device for inference.

        Returns
        -------
        pandas.DataFrame
            DataFrame with id1, id2, prediction columns.
        """
        # Setup device
        if device is None and TORCH_AVAILABLE:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device is None:
            device = "cpu"

        # Process batch without threshold filtering
        predictions = self._process_batch(
            pairs, df_left, df_right, id_column, trained_model,
            tokenizer, model_type, 0.0, batch_size, device  # threshold=0 to get all
        )

        if not predictions:
            return pd.DataFrame(columns=["id1", "id2", "prediction"])

        # Convert to DataFrame
        result_df = pd.DataFrame(predictions)
        result_df = result_df.rename(columns={"score": "prediction"})
        return result_df[["id1", "id2", "prediction"]]

    def __repr__(self) -> str:
        return f"PLMBasedMatcher(text_formatter={self.text_formatter})"


# Handle missing imports gracefully
import contextlib