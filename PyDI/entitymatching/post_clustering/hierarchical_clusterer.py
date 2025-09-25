"""
Hierarchical Clusterer for entity matching post-processing.

This implementation is based on Winter's HierarchicalClusterer,
adapted to work with PyDI's CorrespondenceSet format.
"""

from __future__ import annotations

import logging
from typing import Set, List, Tuple, Dict, Optional
from enum import Enum

import pandas as pd
import numpy as np

from .base import BasePostClusterer
from ..base import CorrespondenceSet


class LinkageMode(Enum):
    """Linkage modes for hierarchical clustering."""
    MIN = "min"
    MAX = "max"
    AVG = "avg"


class HierarchicalClusterer(BasePostClusterer):
    """Hierarchical clustering for entity correspondences.

    Implements agglomerative hierarchical clustering that builds a hierarchy
    of entity clusters by iteratively merging the closest pairs of clusters.
    Supports different linkage criteria (min, max, average) and stopping
    criteria (number of clusters or minimum similarity threshold).
    """

    def __init__(
        self,
        linkage_mode: LinkageMode = LinkageMode.MIN,
        num_clusters: Optional[int] = None,
        min_similarity: Optional[float] = None,
        threshold: float = 0.0,
        min_cluster_size: int = 2,
        preserve_scores: bool = True,
        **kwargs
    ):
        """Initialize Hierarchical Clusterer.

        Parameters
        ----------
        linkage_mode : LinkageMode, default=LinkageMode.MIN
            Linkage criterion for merging clusters.
        num_clusters : int, optional
            Target number of clusters to produce.
        min_similarity : float, optional
            Minimum similarity threshold for merging clusters.
        threshold : float, default=0.0
            Minimum similarity score to consider correspondences.
        min_cluster_size : int, default=2
            Minimum size of clusters to include in output.
        preserve_scores : bool, default=True
            Whether to preserve original similarity scores.
        """
        super().__init__(
            threshold=threshold,
            min_cluster_size=min_cluster_size,
            preserve_scores=preserve_scores,
            **kwargs
        )

        self.linkage_mode = linkage_mode
        self.num_clusters = num_clusters
        self.min_similarity = min_similarity

        if num_clusters is None and min_similarity is None:
            raise ValueError("Must specify either num_clusters or min_similarity")

    def cluster(self, correspondences: CorrespondenceSet) -> CorrespondenceSet:
        """Apply hierarchical clustering to correspondences.

        Parameters
        ----------
        correspondences : CorrespondenceSet
            Input correspondences to cluster.

        Returns
        -------
        CorrespondenceSet
            Hierarchically clustered correspondences.
        """
        self._validate_correspondences(correspondences)

        if len(correspondences) == 0:
            return correspondences.copy()

        # Filter by threshold
        filtered_correspondences = self._filter_by_threshold(correspondences)

        if len(filtered_correspondences) == 0:
            return pd.DataFrame(columns=['id1', 'id2', 'score', 'notes'])

        # Apply hierarchical clustering
        clusters = self._hierarchical_clustering_algorithm(filtered_correspondences)

        # Filter by minimum cluster size
        filtered_clusters = [
            cluster for cluster in clusters
            if len(cluster) >= self.min_cluster_size
        ]

        if not filtered_clusters:
            logging.info("No clusters meet minimum size requirement")
            return pd.DataFrame(columns=['id1', 'id2', 'score', 'notes'])

        # Generate correspondences from clusters
        result_pairs = []
        for cluster in filtered_clusters:
            cluster_pairs = self._generate_cluster_correspondences(
                cluster, correspondences if self.preserve_scores else None
            )
            result_pairs.extend(cluster_pairs)

        # Create result DataFrame
        if not result_pairs:
            return pd.DataFrame(columns=['id1', 'id2', 'score', 'notes'])

        result = self._create_correspondence_dataframe(
            [(pair[0], pair[1]) for pair in result_pairs],
            [pair[2] for pair in result_pairs] if self.preserve_scores else None,
            correspondences
        )

        self._log_algorithm_info(correspondences, result)

        return result

    def _hierarchical_clustering_algorithm(
        self, correspondences: CorrespondenceSet
    ) -> List[Set[str]]:
        """Apply hierarchical clustering algorithm.

        Parameters
        ----------
        correspondences : CorrespondenceSet
            Filtered correspondences to cluster.

        Returns
        -------
        List[Set[str]]
            List of entity clusters.
        """
        # Build initial data structures
        entities = list(set(correspondences['id1']) | set(correspondences['id2']))
        n_entities = len(entities)
        entity_to_idx = {entity: i for i, entity in enumerate(entities)}

        # Initialize similarity matrix
        similarity_matrix = self._build_similarity_matrix(
            correspondences, entities, entity_to_idx
        )

        # Initialize each entity as its own cluster
        clusters = {i: {entities[i]} for i in range(n_entities)}
        active_clusters = set(range(n_entities))

        # Perform hierarchical clustering
        while len(active_clusters) > 1:
            # Find closest pair of clusters
            best_sim = float('-inf')
            best_pair = None

            for i in active_clusters:
                for j in active_clusters:
                    if i < j:  # Avoid duplicates and self-comparison
                        sim = self._compute_cluster_similarity(
                            clusters[i], clusters[j], similarity_matrix, entity_to_idx
                        )
                        if sim > best_sim:
                            best_sim = sim
                            best_pair = (i, j)

            if best_pair is None:
                break

            # Check stopping criterion
            if self.min_similarity is not None and best_sim < self.min_similarity:
                break

            if self.num_clusters is not None and len(active_clusters) <= self.num_clusters:
                break

            # Merge clusters
            i, j = best_pair
            clusters[i].update(clusters[j])
            del clusters[j]
            active_clusters.remove(j)

        return list(clusters.values())

    def _build_similarity_matrix(
        self, correspondences: CorrespondenceSet, entities: List[str], entity_to_idx: Dict[str, int]
    ) -> np.ndarray:
        """Build similarity matrix from correspondences.

        Parameters
        ----------
        correspondences : CorrespondenceSet
            Input correspondences.
        entities : List[str]
            List of all entities.
        entity_to_idx : Dict[str, int]
            Mapping from entity to matrix index.

        Returns
        -------
        np.ndarray
            Symmetric similarity matrix.
        """
        n = len(entities)
        matrix = np.zeros((n, n))

        for _, row in correspondences.iterrows():
            i = entity_to_idx[row['id1']]
            j = entity_to_idx[row['id2']]
            score = row['score']

            matrix[i, j] = score
            matrix[j, i] = score

        return matrix

    def _compute_cluster_similarity(
        self,
        cluster1: Set[str],
        cluster2: Set[str],
        similarity_matrix: np.ndarray,
        entity_to_idx: Dict[str, int]
    ) -> float:
        """Compute similarity between two clusters based on linkage mode.

        Parameters
        ----------
        cluster1, cluster2 : Set[str]
            Entity clusters to compare.
        similarity_matrix : np.ndarray
            Similarity matrix.
        entity_to_idx : Dict[str, int]
            Entity to index mapping.

        Returns
        -------
        float
            Cluster similarity score.
        """
        similarities = []

        for entity1 in cluster1:
            for entity2 in cluster2:
                i = entity_to_idx[entity1]
                j = entity_to_idx[entity2]
                sim = similarity_matrix[i, j]
                if sim > 0:  # Only consider existing correspondences
                    similarities.append(sim)

        if not similarities:
            return float('-inf')  # No connections between clusters

        if self.linkage_mode == LinkageMode.MIN:
            return min(similarities)
        elif self.linkage_mode == LinkageMode.MAX:
            return max(similarities)
        elif self.linkage_mode == LinkageMode.AVG:
            return sum(similarities) / len(similarities)
        else:
            raise ValueError(f"Unknown linkage mode: {self.linkage_mode}")

    def _generate_cluster_correspondences(
        self,
        cluster: Set[str],
        original_correspondences: CorrespondenceSet = None
    ) -> List[Tuple[str, str, float]]:
        """Generate all-pairs correspondences within a cluster.

        Parameters
        ----------
        cluster : Set[str]
            Set of entity IDs in the cluster.
        original_correspondences : CorrespondenceSet, optional
            Original correspondences to lookup scores from.

        Returns
        -------
        List[Tuple[str, str, float]]
            List of (id1, id2, score) tuples for all pairs in cluster.
        """
        cluster_list = sorted(list(cluster))
        pairs = []

        # Create lookup for original scores
        score_lookup = {}
        if original_correspondences is not None:
            for _, row in original_correspondences.iterrows():
                key1 = (row['id1'], row['id2'])
                key2 = (row['id2'], row['id1'])
                score_lookup[key1] = row['score']
                score_lookup[key2] = row['score']

        # Generate all pairs within cluster
        for i in range(len(cluster_list)):
            for j in range(i + 1, len(cluster_list)):
                id1, id2 = cluster_list[i], cluster_list[j]

                # Get score from original correspondences or default
                if (id1, id2) in score_lookup:
                    score = score_lookup[(id1, id2)]
                elif (id2, id1) in score_lookup:
                    score = score_lookup[(id2, id1)]
                else:
                    # Default score for hierarchical connections
                    score = 1.0 if not self.preserve_scores else 0.6

                pairs.append((id1, id2, score))

        return pairs