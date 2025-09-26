"""
Connected Components Clusterer for entity matching post-processing.

This implementation is based on Winter's ConnectedComponentClusterer,
adapted to work with PyDI's CorrespondenceSet format.
"""

from __future__ import annotations

import logging
from typing import Dict, Set, List, Tuple

import pandas as pd

from .base import BasePostClusterer
from ..base import CorrespondenceSet


class ConnectedComponentClusterer(BasePostClusterer):
    """Groups entities into clusters based on connected components.

    This clusterer finds weakly connected components in the similarity graph
    formed by entity correspondences. All entities that are transitively
    connected (directly or through other entities) are grouped into the
    same cluster.

    This enforces transitivity: if A matches B and B matches C, then A, B, and C
    are all considered equivalent and grouped together.

    Examples
    --------
    >>> clusterer = ConnectedComponentClusterer(threshold=0.7)
    >>> correspondences = pd.DataFrame([
    ...     {'id1': 'a', 'id2': 'b', 'score': 0.9, 'notes': ''},
    ...     {'id1': 'b', 'id2': 'c', 'score': 0.8, 'notes': ''},
    ...     {'id1': 'd', 'id2': 'e', 'score': 0.7, 'notes': ''}
    ... ])
    >>> result = clusterer.cluster(correspondences)
    # Result groups: {a,b,c}, {d,e}
    """

    def __init__(
        self,
        threshold: float = 0.0,
        min_cluster_size: int = 2,
        preserve_scores: bool = True,
        **kwargs
    ):
        """Initialize Connected Components Clusterer.

        Parameters
        ----------
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

    def cluster(self, correspondences: CorrespondenceSet) -> CorrespondenceSet:
        """Apply connected components clustering to correspondences.

        Parameters
        ----------
        correspondences : CorrespondenceSet
            Input correspondences to cluster.

        Returns
        -------
        CorrespondenceSet
            Clustered correspondences where all entities in the same
            connected component are fully connected to each other.
        """
        self._validate_correspondences(correspondences)

        if len(correspondences) == 0:
            return correspondences.copy()

        # Filter by threshold
        filtered_correspondences = self._filter_by_threshold(correspondences)

        if len(filtered_correspondences) == 0:
            return pd.DataFrame(columns=['id1', 'id2', 'score', 'notes'])

        # Build connected components
        clusters = self._find_connected_components(filtered_correspondences)

        # Filter by minimum cluster size
        filtered_clusters = [
            cluster for cluster in clusters
            if len(cluster) >= self.min_cluster_size
        ]

        if not filtered_clusters:
            logging.info("No clusters meet minimum size requirement")
            return pd.DataFrame(columns=['id1', 'id2', 'score', 'notes'])

        # Generate all-pairs correspondences within each cluster
        result_pairs = []
        for cluster in filtered_clusters:
            cluster_pairs = self._generate_cluster_pairs(
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

    def _find_connected_components(
        self, correspondences: CorrespondenceSet
    ) -> List[Set[str]]:
        """Find connected components using Union-Find algorithm.

        Parameters
        ----------
        correspondences : CorrespondenceSet
            Input correspondences to analyze.

        Returns
        -------
        List[Set[str]]
            List of connected component sets.
        """
        # Initialize Union-Find data structure
        parent = {}

        def find(x):
            if x not in parent:
                parent[x] = x
            if parent[x] != x:
                parent[x] = find(parent[x])  # Path compression
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Process all edges (correspondences)
        for _, row in correspondences.iterrows():
            union(row['id1'], row['id2'])

        # Group entities by their root parent
        components = {}
        for entity in parent:
            root = find(entity)
            if root not in components:
                components[root] = set()
            components[root].add(entity)

        return list(components.values())

    def _generate_cluster_pairs(
        self,
        cluster: Set[str],
        original_correspondences: CorrespondenceSet = None
    ) -> List[Tuple[str, str, float]]:
        """Generate all pairs within a cluster with appropriate scores.

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
        cluster_list = sorted(list(cluster))  # Sort for consistent output
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
                    # Default score for transitive connections
                    score = 1.0 if not self.preserve_scores else 0.5

                pairs.append((id1, id2, score))

        return pairs

    def get_clusters(self, correspondences: CorrespondenceSet) -> List[Set[str]]:
        """Get the clusters without generating correspondences.

        This method is useful for analysis and debugging.

        Parameters
        ----------
        correspondences : CorrespondenceSet
            Input correspondences to cluster.

        Returns
        -------
        List[Set[str]]
            List of entity clusters (connected components).
        """
        self._validate_correspondences(correspondences)

        if len(correspondences) == 0:
            return []

        filtered_correspondences = self._filter_by_threshold(correspondences)
        clusters = self._find_connected_components(filtered_correspondences)

        return [
            cluster for cluster in clusters
            if len(cluster) >= self.min_cluster_size
        ]