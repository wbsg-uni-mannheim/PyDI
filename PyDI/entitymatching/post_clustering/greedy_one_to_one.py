"""
Greedy One-to-One Matching Algorithm for entity matching post-processing.

This implementation is based on Winter's GreedyOneToOneMatchingAlgorithm,
adapted to work with PyDI's CorrespondenceSet format.
"""

from __future__ import annotations

import logging
from typing import Set, List, Tuple

import pandas as pd

from .base import BasePostClusterer
from ..base import CorrespondenceSet


class GreedyOneToOneMatchingAlgorithm(BasePostClusterer):
    """Enforces one-to-one matching using a greedy approach.

    This algorithm ensures that each entity participates in at most one
    correspondence by greedily selecting the highest-scoring matches first
    and removing conflicting lower-scoring matches.

    The algorithm:
    1. Sorts all correspondences by similarity score (descending)
    2. Greedily selects correspondences, ensuring no entity appears twice
    3. Returns the subset of correspondences with maximum total weight

    This is not guaranteed to return the globally optimal solution, but
    provides a good approximation efficiently.

    Examples
    --------
    >>> matcher = GreedyOneToOneMatchingAlgorithm()
    >>> correspondences = pd.DataFrame([
    ...     {'id1': 'a', 'id2': 'b', 'score': 0.9, 'notes': ''},
    ...     {'id1': 'a', 'id2': 'c', 'score': 0.8, 'notes': ''},
    ...     {'id1': 'd', 'id2': 'b', 'score': 0.8, 'notes': ''},
    ...     {'id1': 'd', 'id2': 'c', 'score': 0.1, 'notes': ''}
    ... ])
    >>> result = matcher.cluster(correspondences)
    # Result: a-b (0.9), d-c (0.1) - greedy selection ensures one-to-one
    """

    def __init__(
        self,
        threshold: float = 0.0,
        preserve_scores: bool = True,
        group_by_dataset: bool = False,
        **kwargs
    ):
        """Initialize Greedy One-to-One Matching Algorithm.

        Parameters
        ----------
        threshold : float, default=0.0
            Minimum similarity score to consider correspondences.
        preserve_scores : bool, default=True
            Whether to preserve original similarity scores.
        group_by_dataset : bool, default=False
            Whether to group correspondences by dataset before matching.
            Useful when matching multiple datasets simultaneously.
        """
        super().__init__(
            threshold=threshold,
            preserve_scores=preserve_scores,
            **kwargs
        )
        self.group_by_dataset = group_by_dataset

    def cluster(self, correspondences: CorrespondenceSet) -> CorrespondenceSet:
        """Apply greedy one-to-one matching to correspondences.

        Parameters
        ----------
        correspondences : CorrespondenceSet
            Input correspondences to process.

        Returns
        -------
        CorrespondenceSet
            One-to-one matched correspondences where each entity appears
            in at most one correspondence.
        """
        self._validate_correspondences(correspondences)

        if len(correspondences) == 0:
            return correspondences.copy()

        # Filter by threshold
        filtered_correspondences = self._filter_by_threshold(correspondences)

        if len(filtered_correspondences) == 0:
            return pd.DataFrame(columns=['id1', 'id2', 'score', 'notes'])

        # Apply greedy matching
        if self.group_by_dataset:
            result = self._match_with_dataset_grouping(filtered_correspondences)
        else:
            result = self._greedy_one_to_one_match(filtered_correspondences)

        self._log_algorithm_info(correspondences, result)

        return result

    def _greedy_one_to_one_match(
        self, correspondences: CorrespondenceSet
    ) -> CorrespondenceSet:
        """Apply greedy one-to-one matching to correspondences.

        Parameters
        ----------
        correspondences : CorrespondenceSet
            Filtered correspondences to match.

        Returns
        -------
        CorrespondenceSet
            One-to-one matched correspondences.
        """
        # Sort correspondences by similarity score (descending)
        sorted_correspondences = correspondences.sort_values(
            by='score', ascending=False
        ).reset_index(drop=True)

        matched_entities = set()
        selected_correspondences = []

        for _, row in sorted_correspondences.iterrows():
            id1, id2 = row['id1'], row['id2']

            # Check if either entity is already matched
            if id1 not in matched_entities and id2 not in matched_entities:
                # Select this correspondence
                selected_correspondences.append(row)
                matched_entities.add(id1)
                matched_entities.add(id2)

        if not selected_correspondences:
            return pd.DataFrame(columns=['id1', 'id2', 'score', 'notes'])

        result = pd.DataFrame(selected_correspondences).reset_index(drop=True)

        logging.info(
            f"Greedy matching: {len(correspondences)} -> {len(result)} correspondences "
            f"({len(matched_entities)} entities matched)"
        )

        return result

    def _match_with_dataset_grouping(
        self, correspondences: CorrespondenceSet
    ) -> CorrespondenceSet:
        """Apply matching with dataset grouping.

        This groups correspondences by dataset identifiers and applies
        one-to-one matching within each group.

        Parameters
        ----------
        correspondences : CorrespondenceSet
            Correspondences to process.

        Returns
        -------
        CorrespondenceSet
            One-to-one matched correspondences grouped by dataset.
        """
        # For now, assume all correspondences are in the same group
        # This can be extended to handle dataset identifiers if available
        logging.info("Dataset grouping requested but not implemented - using global matching")
        return self._greedy_one_to_one_match(correspondences)

    def get_matching_statistics(self, correspondences: CorrespondenceSet) -> dict:
        """Get statistics about the matching process.

        Parameters
        ----------
        correspondences : CorrespondenceSet
            Input correspondences.

        Returns
        -------
        dict
            Statistics including reduction ratio, entity counts, etc.
        """
        if len(correspondences) == 0:
            return {
                'input_correspondences': 0,
                'output_correspondences': 0,
                'reduction_ratio': 0.0,
                'input_entities': 0,
                'matched_entities': 0
            }

        filtered = self._filter_by_threshold(correspondences)
        result = self._greedy_one_to_one_match(filtered)

        input_entities = set(correspondences['id1']) | set(correspondences['id2'])
        matched_entities = set(result['id1']) | set(result['id2']) if len(result) > 0 else set()

        return {
            'input_correspondences': len(correspondences),
            'filtered_correspondences': len(filtered),
            'output_correspondences': len(result),
            'reduction_ratio': 1 - (len(result) / len(correspondences)) if len(correspondences) > 0 else 0,
            'input_entities': len(input_entities),
            'matched_entities': len(matched_entities),
            'matching_coverage': len(matched_entities) / len(input_entities) if len(input_entities) > 0 else 0
        }