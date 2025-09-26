"""
Base classes for post-clustering and global optimization algorithms.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Tuple, Set, Dict, Any, Optional

import pandas as pd
import numpy as np

from ..base import CorrespondenceSet


class BasePostClusterer(ABC):
    """Abstract base class for post-clustering assignment algorithms.

    Post-clustering algorithms take entity correspondences and refine them
    by solving various assignment problems. This includes clustering-based
    methods (grouping similar entities) and global optimization methods
    (enforcing constraints like one-to-one matching).

    All implementations maintain compatibility with PyDI's CorrespondenceSet
    format for seamless integration with evaluation and downstream processing.
    """

    def __init__(
        self,
        threshold: Optional[float] = None,
        min_cluster_size: int = 2,
        preserve_scores: bool = True,
        **kwargs
    ):
        """Initialize the post-clustering algorithm.

        Parameters
        ----------
        threshold : float, optional
            Similarity threshold for clustering decisions. If None, uses all
            correspondences regardless of similarity score.
        min_cluster_size : int, default=2
            Minimum number of entities required to form a cluster.
        preserve_scores : bool, default=True
            Whether to preserve original similarity scores in the output.
            If False, may assign uniform scores to clustered entities.
        **kwargs
            Algorithm-specific parameters (e.g., linkage_mode, num_clusters).
        """
        self.threshold = threshold
        self.min_cluster_size = min_cluster_size
        self.preserve_scores = preserve_scores
        self.algorithm_params = kwargs

        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """Validate initialization parameters."""
        if self.threshold is not None and (self.threshold < 0 or self.threshold > 1):
            raise ValueError("Threshold must be between 0 and 1")
        if self.min_cluster_size < 1:
            raise ValueError("min_cluster_size must be at least 1")

    @abstractmethod
    def cluster(self, correspondences: CorrespondenceSet) -> CorrespondenceSet:
        """Apply clustering algorithm to refine correspondences.

        Parameters
        ----------
        correspondences : CorrespondenceSet
            DataFrame with columns id1, id2, score, notes containing
            entity correspondences to be refined.

        Returns
        -------
        CorrespondenceSet
            Refined correspondences in the same format as input.
            The specific refinement depends on the algorithm:
            - Clustering methods may group entities into equivalence classes
            - Matching methods may enforce one-to-one constraints
        """
        raise NotImplementedError

    def _validate_correspondences(self, correspondences: CorrespondenceSet) -> None:
        """Validate input correspondences format.

        Parameters
        ----------
        correspondences : CorrespondenceSet
            Input correspondences to validate.

        Raises
        ------
        ValueError
            If correspondences don't have required columns.
        """
        required_columns = {'id1', 'id2', 'score'}
        if not required_columns.issubset(correspondences.columns):
            missing = required_columns - set(correspondences.columns)
            raise ValueError(f"Correspondences missing required columns: {missing}")

        if len(correspondences) == 0:
            logging.warning("Empty correspondences provided to post-clustering")

    def _filter_by_threshold(self, correspondences: CorrespondenceSet) -> CorrespondenceSet:
        """Filter correspondences by similarity threshold.

        Parameters
        ----------
        correspondences : CorrespondenceSet
            Input correspondences.

        Returns
        -------
        CorrespondenceSet
            Filtered correspondences above threshold.
        """
        if self.threshold is None:
            return correspondences

        filtered = correspondences[correspondences['score'] >= self.threshold].copy()

        if len(filtered) == 0:
            logging.warning(f"No correspondences above threshold {self.threshold}")
        else:
            logging.info(
                f"Filtered correspondences: {len(correspondences)} -> {len(filtered)} "
                f"(threshold={self.threshold})"
            )

        return filtered

    def _correspondences_to_similarity_graph(
        self, correspondences: CorrespondenceSet
    ) -> List[Tuple[str, str, float]]:
        """Convert PyDI correspondences to similarity graph format.

        Parameters
        ----------
        correspondences : CorrespondenceSet
            Input correspondences.

        Returns
        -------
        List[Tuple[str, str, float]]
            List of (entity1, entity2, similarity_score) tuples.
        """
        return [
            (row['id1'], row['id2'], row['score'])
            for _, row in correspondences.iterrows()
        ]

    def _create_correspondence_dataframe(
        self,
        entity_pairs: List[Tuple[str, str]],
        scores: Optional[List[float]] = None,
        original_correspondences: Optional[CorrespondenceSet] = None
    ) -> CorrespondenceSet:
        """Create CorrespondenceSet from entity pairs and scores.

        Parameters
        ----------
        entity_pairs : List[Tuple[str, str]]
            List of (id1, id2) entity pairs.
        scores : List[float], optional
            Similarity scores for each pair. If None, attempts to retrieve
            from original_correspondences or sets to 1.0.
        original_correspondences : CorrespondenceSet, optional
            Original correspondences to lookup scores and notes from.

        Returns
        -------
        CorrespondenceSet
            DataFrame with id1, id2, score, notes columns.
        """
        if not entity_pairs:
            # Return empty DataFrame with correct schema
            return pd.DataFrame(columns=['id1', 'id2', 'score', 'notes'])

        result_data = []

        # Create lookup for original correspondences if available
        original_lookup = {}
        if original_correspondences is not None:
            for _, row in original_correspondences.iterrows():
                key = (row['id1'], row['id2'])
                original_lookup[key] = row
                # Also add reverse lookup for symmetric correspondences
                reverse_key = (row['id2'], row['id1'])
                if reverse_key not in original_lookup:
                    original_lookup[reverse_key] = row

        for i, (id1, id2) in enumerate(entity_pairs):
            # Determine score
            if scores is not None:
                score = scores[i]
            elif (id1, id2) in original_lookup:
                score = original_lookup[(id1, id2)]['score']
            else:
                score = 1.0  # Default score for new correspondences

            # Determine notes
            notes = ""
            if (id1, id2) in original_lookup and 'notes' in original_lookup[(id1, id2)]:
                notes = original_lookup[(id1, id2)]['notes']

            result_data.append({
                'id1': id1,
                'id2': id2,
                'score': score,
                'notes': notes
            })

        return pd.DataFrame(result_data)

    def _log_algorithm_info(
        self,
        correspondences: CorrespondenceSet,
        result: CorrespondenceSet
    ) -> None:
        """Log information about the post-clustering process.

        Parameters
        ----------
        correspondences : CorrespondenceSet
            Input correspondences.
        result : CorrespondenceSet
            Output correspondences after clustering.
        """
        algorithm_name = self.__class__.__name__

        logging.info(
            f"{algorithm_name}: {len(correspondences)} -> {len(result)} correspondences"
        )

        if len(correspondences) > 0 and len(result) > 0:
            input_entities = set(correspondences['id1']) | set(correspondences['id2'])
            output_entities = set(result['id1']) | set(result['id2'])

            logging.info(
                f"{algorithm_name}: {len(input_entities)} -> {len(output_entities)} entities"
            )