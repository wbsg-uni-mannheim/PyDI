"""
Stable Matching Algorithm for entity matching post-processing.

This implementation is based on Winter's BestChoiceMatching (stable marriage-style),
adapted to work with PyDI's CorrespondenceSet format.
"""

from __future__ import annotations

import logging
from typing import Set, List, Tuple, Dict, Optional

import pandas as pd

from .base import BasePostClusterer
from ..base import CorrespondenceSet


class StableMatching(BasePostClusterer):
    """Finds stable matches using preference-based selection.

    This algorithm ensures that the selected matches are stable in the sense
    that no entity would prefer to switch to a different partner. It implements
    a variant of the stable marriage problem for entity matching.

    The algorithm:
    1. For each entity, find its best potential match
    2. Check if this match is mutual (both entities prefer each other)
    3. Only include matches that are stable (mutually preferred)

    This guarantees that no entity in the result would prefer to switch partners
    with any other entity in the result.
    """

    def __init__(
        self,
        threshold: float = 0.0,
        preserve_scores: bool = True,
        force_one_to_one: bool = True,
        **kwargs
    ):
        """Initialize Stable Matching Algorithm.

        Parameters
        ----------
        threshold : float, default=0.0
            Minimum similarity score to consider correspondences.
        preserve_scores : bool, default=True
            Whether to preserve original similarity scores.
        force_one_to_one : bool, default=True
            Whether to enforce one-to-one matching constraint.
        """
        super().__init__(
            threshold=threshold,
            preserve_scores=preserve_scores,
            **kwargs
        )
        self.force_one_to_one = force_one_to_one

    def cluster(self, correspondences: CorrespondenceSet) -> CorrespondenceSet:
        """Apply stable matching to correspondences.

        Parameters
        ----------
        correspondences : CorrespondenceSet
            Input correspondences to process.

        Returns
        -------
        CorrespondenceSet
            Stable matches where each entity is matched to its mutually
            preferred partner.
        """
        self._validate_correspondences(correspondences)

        if len(correspondences) == 0:
            return correspondences.copy()

        # Filter by threshold
        filtered_correspondences = self._filter_by_threshold(correspondences)

        if len(filtered_correspondences) == 0:
            return pd.DataFrame(columns=['id1', 'id2', 'score', 'notes'])

        # Apply stable matching algorithm
        result = self._stable_matching_algorithm(filtered_correspondences)

        self._log_algorithm_info(correspondences, result)

        return result

    def _stable_matching_algorithm(
        self, correspondences: CorrespondenceSet
    ) -> CorrespondenceSet:
        """Apply stable matching algorithm.

        Parameters
        ----------
        correspondences : CorrespondenceSet
            Filtered correspondences to process.

        Returns
        -------
        CorrespondenceSet
            Stable matched correspondences.
        """
        # Build correspondence lookup dictionary for O(1) access
        correspondence_lookup = self._build_correspondence_lookup(correspondences)

        # Build preference lists for each entity
        preferences = self._build_preference_lists(correspondences)

        # Find stable matches
        stable_matches = []
        already_matched = set()

        # Sort entities by ID for consistent processing order
        left_entities = sorted(set(correspondences['id1']))

        for entity in left_entities:
            if self.force_one_to_one and entity in already_matched:
                continue

            # Find best stable match for this entity
            best_match = self._find_best_stable_match(
                entity, preferences, correspondence_lookup, already_matched
            )

            if best_match is not None:
                stable_matches.append(best_match)
                if self.force_one_to_one:
                    already_matched.add(best_match['id1'])
                    already_matched.add(best_match['id2'])

        if not stable_matches:
            return pd.DataFrame(columns=['id1', 'id2', 'score', 'notes'])

        result = pd.DataFrame(stable_matches).reset_index(drop=True)

        logging.info(
            f"Stable matching: {len(correspondences)} -> {len(result)} correspondences "
            f"({len(already_matched)} entities matched)"
        )

        return result

    def _build_correspondence_lookup(
        self, correspondences: CorrespondenceSet
    ) -> Dict[Tuple[str, str], dict]:
        """Build lookup dictionary for fast correspondence access.

        Parameters
        ----------
        correspondences : CorrespondenceSet
            Input correspondences.

        Returns
        -------
        Dict[Tuple[str, str], dict]
            Dictionary mapping (id1, id2) tuples to correspondence rows.
            Includes both forward and reverse lookups.
        """
        lookup = {}

        # Convert to list of dicts once (faster than repeated iterrows)
        corr_list = correspondences.to_dict('records')

        for row in corr_list:
            id1, id2 = row['id1'], row['id2']
            # Add forward lookup
            lookup[(id1, id2)] = row
            # Add reverse lookup for symmetric matching
            lookup[(id2, id1)] = row

        return lookup

    def _build_preference_lists(
        self, correspondences: CorrespondenceSet
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Build preference lists for each entity.

        Parameters
        ----------
        correspondences : CorrespondenceSet
            Input correspondences.

        Returns
        -------
        Dict[str, List[Tuple[str, float]]]
            Mapping from entity to sorted list of (partner, score) preferences.
        """
        preferences = {}

        # Convert to list of dicts once (much faster than iterrows)
        corr_list = correspondences.to_dict('records')

        # Build preferences for both sides
        for row in corr_list:
            id1, id2, score = row['id1'], row['id2'], row['score']

            # Add preference for id1 -> id2
            if id1 not in preferences:
                preferences[id1] = []
            preferences[id1].append((id2, score))

            # Add preference for id2 -> id1 (symmetric)
            if id2 not in preferences:
                preferences[id2] = []
            preferences[id2].append((id1, score))

        # Sort preference lists by score (descending)
        for entity in preferences:
            preferences[entity].sort(key=lambda x: x[1], reverse=True)

        return preferences

    def _find_best_stable_match(
        self,
        entity: str,
        preferences: Dict[str, List[Tuple[str, float]]],
        correspondence_lookup: Dict[Tuple[str, str], dict],
        already_matched: Set[str]
    ) -> Optional[dict]:
        """Find the best stable match for an entity.

        Parameters
        ----------
        entity : str
            Entity to find match for.
        preferences : Dict[str, List[Tuple[str, float]]]
            Preference lists for all entities.
        correspondence_lookup : Dict[Tuple[str, str], dict]
            Dictionary for O(1) correspondence lookup.
        already_matched : Set[str]
            Set of already matched entities.

        Returns
        -------
        Optional[dict]
            Best stable match as correspondence row, or None if no stable match.
        """
        if entity not in preferences:
            return None

        entity_preferences = preferences[entity]

        for candidate, score in entity_preferences:
            if self.force_one_to_one and candidate in already_matched:
                continue

            # Check if this match is stable
            if self._is_stable_match(entity, candidate, score, preferences, already_matched):
                # Use O(1) lookup instead of O(n) search
                match_row = correspondence_lookup.get((entity, candidate))
                if match_row is not None:
                    return match_row

        return None

    def _is_stable_match(
        self,
        entity1: str,
        entity2: str,
        score: float,
        preferences: Dict[str, List[Tuple[str, float]]],
        already_matched: Set[str]
    ) -> bool:
        """Check if a match is stable.

        A match is stable if both entities mutually prefer each other over
        any other available options.

        Parameters
        ----------
        entity1, entity2 : str
            Entities to check for stability.
        score : float
            Similarity score between entities.
        preferences : Dict[str, List[Tuple[str, float]]]
            Preference lists for all entities.
        already_matched : Set[str]
            Set of already matched entities.

        Returns
        -------
        bool
            True if match is stable, False otherwise.
        """
        # Check if entity2 also prefers entity1 among available options
        if entity2 not in preferences:
            return False

        entity2_preferences = preferences[entity2]

        # Find entity1 in entity2's preference list
        entity1_rank = None
        entity1_score = None
        for i, (candidate, candidate_score) in enumerate(entity2_preferences):
            if candidate == entity1:
                entity1_rank = i
                entity1_score = candidate_score
                break

        if entity1_rank is None:
            return False

        # Check if entity2 has any better available options
        for i, (candidate, candidate_score) in enumerate(entity2_preferences):
            if i >= entity1_rank:
                break  # No better options

            if self.force_one_to_one and candidate in already_matched:
                continue

            # entity2 prefers this candidate over entity1, so match is not stable
            return False

        return True

    def get_stability_analysis(self, correspondences: CorrespondenceSet) -> dict:
        """Analyze the stability of matches.

        Parameters
        ----------
        correspondences : CorrespondenceSet
            Input correspondences.

        Returns
        -------
        dict
            Stability analysis including number of stable matches,
            preference satisfaction, etc.
        """
        if len(correspondences) == 0:
            return {
                'total_correspondences': 0,
                'stable_matches': 0,
                'stability_ratio': 0.0
            }

        filtered = self._filter_by_threshold(correspondences)
        stable_result = self._stable_matching_algorithm(filtered)

        # Compare with greedy matching
        from .greedy_one_to_one import GreedyOneToOneMatchingAlgorithm
        greedy_matcher = GreedyOneToOneMatchingAlgorithm(threshold=0.0)
        greedy_result = greedy_matcher._greedy_one_to_one_match(filtered)

        return {
            'total_correspondences': len(correspondences),
            'filtered_correspondences': len(filtered),
            'stable_matches': len(stable_result),
            'greedy_matches': len(greedy_result),
            'stability_ratio': len(stable_result) / len(greedy_result) if len(greedy_result) > 0 else 0,
            'stable_total_weight': stable_result['score'].sum() if len(stable_result) > 0 else 0,
            'greedy_total_weight': greedy_result['score'].sum() if len(greedy_result) > 0 else 0
        }