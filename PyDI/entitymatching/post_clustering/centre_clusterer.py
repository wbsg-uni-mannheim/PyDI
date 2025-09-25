"""
Centre Clusterer for entity matching post-processing.

This implementation is based on Winter's CentreClusterer (CENTER algorithm),
adapted to work with PyDI's CorrespondenceSet format.
"""

from __future__ import annotations

import logging
from typing import Set, List, Tuple, Dict

import pandas as pd

from .base import BasePostClusterer
from ..base import CorrespondenceSet


class CentreClusterer(BasePostClusterer):
    """Creates star-shaped clusters using the CENTER algorithm.

    The CENTER algorithm finds star-shaped clusters by:
    1. Sorting all correspondences by similarity score (descending)
    2. For each correspondence, checking if either entity is already a center
    3. If neither is assigned, creating a new cluster with the first entity as center
    4. If one entity is a center, adding the other to that cluster
    5. Resulting clusters have maximum diameter of 2 (star shape)

    This is useful for creating hub-based entity groups where one entity
    serves as the canonical representative.
    """

    def __init__(
        self,
        threshold: float = 0.0,
        min_cluster_size: int = 2,
        preserve_scores: bool = True,
        **kwargs
    ):
        """Initialize Centre Clusterer.

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
        """Apply centre clustering to correspondences.

        Parameters
        ----------
        correspondences : CorrespondenceSet
            Input correspondences to cluster.

        Returns
        -------
        CorrespondenceSet
            Star-shaped clusters where each cluster has a center entity.
        """
        self._validate_correspondences(correspondences)

        if len(correspondences) == 0:
            return correspondences.copy()

        # Filter by threshold
        filtered_correspondences = self._filter_by_threshold(correspondences)

        if len(filtered_correspondences) == 0:
            return pd.DataFrame(columns=['id1', 'id2', 'score', 'notes'])

        # Apply center clustering algorithm
        clusters = self._center_clustering_algorithm(filtered_correspondences)

        # Filter by minimum cluster size
        filtered_clusters = {
            center: cluster for center, cluster in clusters.items()
            if len(cluster) >= self.min_cluster_size
        }

        if not filtered_clusters:
            logging.info("No clusters meet minimum size requirement")
            return pd.DataFrame(columns=['id1', 'id2', 'score', 'notes'])

        # Generate correspondences from clusters
        result_pairs = []
        for center, cluster in filtered_clusters.items():
            cluster_pairs = self._generate_star_correspondences(
                center, cluster, correspondences if self.preserve_scores else None
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

    def _center_clustering_algorithm(
        self, correspondences: CorrespondenceSet
    ) -> Dict[str, Set[str]]:
        """Apply the CENTER clustering algorithm.

        Parameters
        ----------
        correspondences : CorrespondenceSet
            Filtered correspondences to cluster.

        Returns
        -------
        Dict[str, Set[str]]
            Mapping from center entity to its cluster members.
        """
        # Sort correspondences by similarity score (descending)
        sorted_correspondences = correspondences.sort_values(
            by='score', ascending=False
        ).reset_index(drop=True)

        assigned_entities = set()
        clusters = {}  # center -> set of cluster members

        for _, row in sorted_correspondences.iterrows():
            id1, id2 = row['id1'], row['id2']

            # Check assignment status
            id1_assigned = id1 in assigned_entities
            id2_assigned = id2 in assigned_entities

            if not id1_assigned and not id2_assigned:
                # Neither entity is assigned - create new cluster with id1 as center
                clusters[id1] = {id1, id2}
                assigned_entities.add(id1)
                assigned_entities.add(id2)

            elif not id1_assigned and id2_assigned:
                # id2 is assigned, check if it's a center
                if id2 in clusters:
                    # id2 is a center - add id1 to its cluster
                    clusters[id2].add(id1)
                    assigned_entities.add(id1)
                # If id2 is not a center, skip this correspondence

            elif id1_assigned and not id2_assigned:
                # id1 is assigned, check if it's a center
                if id1 in clusters:
                    # id1 is a center - add id2 to its cluster
                    clusters[id1].add(id2)
                    assigned_entities.add(id2)
                # If id1 is not a center, skip this correspondence

            # If both are assigned, skip this correspondence

        return clusters

    def _generate_star_correspondences(
        self,
        center: str,
        cluster: Set[str],
        original_correspondences: CorrespondenceSet = None
    ) -> List[Tuple[str, str, float]]:
        """Generate star-shaped correspondences for a cluster.

        In star clusters, all non-center entities are connected to the center,
        but not necessarily to each other.

        Parameters
        ----------
        center : str
            Center entity ID.
        cluster : Set[str]
            Set of all cluster members (including center).
        original_correspondences : CorrespondenceSet, optional
            Original correspondences to lookup scores from.

        Returns
        -------
        List[Tuple[str, str, float]]
            List of (id1, id2, score) tuples for star correspondences.
        """
        pairs = []

        # Create lookup for original scores
        score_lookup = {}
        if original_correspondences is not None:
            for _, row in original_correspondences.iterrows():
                key1 = (row['id1'], row['id2'])
                key2 = (row['id2'], row['id1'])
                score_lookup[key1] = row['score']
                score_lookup[key2] = row['score']

        # Connect all non-center entities to the center
        for entity in sorted(cluster):  # Sort for consistent output
            if entity != center:
                # Get score from original correspondences
                if (center, entity) in score_lookup:
                    score = score_lookup[(center, entity)]
                elif (entity, center) in score_lookup:
                    score = score_lookup[(entity, center)]
                else:
                    # Default score for center connections
                    score = 1.0 if not self.preserve_scores else 0.8

                pairs.append((center, entity, score))

        return pairs

    def get_clusters_with_centers(self, correspondences: CorrespondenceSet) -> Dict[str, Set[str]]:
        """Get clusters with their center entities.

        This method is useful for analysis and understanding cluster structure.

        Parameters
        ----------
        correspondences : CorrespondenceSet
            Input correspondences to cluster.

        Returns
        -------
        Dict[str, Set[str]]
            Mapping from center entity ID to set of cluster members.
        """
        self._validate_correspondences(correspondences)

        if len(correspondences) == 0:
            return {}

        filtered_correspondences = self._filter_by_threshold(correspondences)
        clusters = self._center_clustering_algorithm(filtered_correspondences)

        return {
            center: cluster for center, cluster in clusters.items()
            if len(cluster) >= self.min_cluster_size
        }