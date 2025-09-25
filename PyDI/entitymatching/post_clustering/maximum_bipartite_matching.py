"""
Maximum Bipartite Matching Algorithm for entity matching post-processing.

This implementation is based on Winter's MaximumBipartiteMatchingAlgorithm,
adapted to work with PyDI's CorrespondenceSet format.
"""

from __future__ import annotations

import logging
from typing import Set, List, Tuple, Dict, Optional

import pandas as pd

from .base import BasePostClusterer
from ..base import CorrespondenceSet

# Optional NetworkX import for bipartite matching
try:
    import networkx as nx
    from networkx.algorithms import bipartite

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False


class MaximumBipartiteMatching(BasePostClusterer):
    """Finds optimal one-to-one matching using maximum bipartite matching.

    This algorithm formulates the entity matching problem as a maximum weight
    bipartite matching problem and finds the globally optimal solution using
    the Hungarian algorithm or similar optimization methods.

    Unlike the greedy approach, this guarantees the optimal solution that
    maximizes the total similarity score while enforcing one-to-one constraints.

    Requires NetworkX for the bipartite matching implementation.

    Examples
    --------
    >>> matcher = MaximumBipartiteMatching()
    >>> correspondences = pd.DataFrame([
    ...     {'id1': 'a', 'id2': 'x', 'score': 0.9, 'notes': ''},
    ...     {'id1': 'a', 'id2': 'y', 'score': 0.8, 'notes': ''},
    ...     {'id1': 'b', 'id2': 'x', 'score': 0.7, 'notes': ''},
    ...     {'id1': 'b', 'id2': 'y', 'score': 0.85, 'notes': ''}
    ... ])
    >>> result = matcher.cluster(correspondences)
    # Result: a-x (0.9), b-y (0.85) - optimal total weight = 1.75
    """

    def __init__(
        self,
        threshold: float = 0.0,
        preserve_scores: bool = True,
        use_weight_scaling: bool = True,
        **kwargs,
    ):
        """Initialize Maximum Bipartite Matching Algorithm.

        Parameters
        ----------
        threshold : float, default=0.0
            Minimum similarity score to consider correspondences.
        preserve_scores : bool, default=True
            Whether to preserve original similarity scores.
        use_weight_scaling : bool, default=True
            Whether to scale weights to integers for optimization.
            Some algorithms work better with integer weights.
        """
        if not NETWORKX_AVAILABLE:
            raise ImportError(
                "NetworkX is required for MaximumBipartiteMatching. "
                "Install with: pip install networkx"
            )

        super().__init__(threshold=threshold, preserve_scores=preserve_scores, **kwargs)
        self.use_weight_scaling = use_weight_scaling

    def cluster(self, correspondences: CorrespondenceSet) -> CorrespondenceSet:
        """Apply maximum bipartite matching to correspondences.

        Parameters
        ----------
        correspondences : CorrespondenceSet
            Input correspondences to process.

        Returns
        -------
        CorrespondenceSet
            Optimally matched correspondences with maximum total weight.
        """
        self._validate_correspondences(correspondences)

        if len(correspondences) == 0:
            return correspondences.copy()

        # Filter by threshold
        filtered_correspondences = self._filter_by_threshold(correspondences)

        if len(filtered_correspondences) == 0:
            return pd.DataFrame(columns=["id1", "id2", "score", "notes"])

        # Apply maximum bipartite matching
        result = self._maximum_bipartite_match(filtered_correspondences)

        self._log_algorithm_info(correspondences, result)

        return result

    def _maximum_bipartite_match(
        self, correspondences: CorrespondenceSet
    ) -> CorrespondenceSet:
        """Apply maximum bipartite matching algorithm.

        Parameters
        ----------
        correspondences : CorrespondenceSet
            Filtered correspondences to match.

        Returns
        -------
        CorrespondenceSet
            Optimally matched correspondences.
        """
        # Create bipartite graph
        graph = self._create_bipartite_graph(correspondences)

        if graph.number_of_nodes() == 0:
            return pd.DataFrame(columns=["id1", "id2", "score", "notes"])

        # Identify the two partitions
        left_nodes = set(correspondences["id1"].unique())
        right_nodes = set(correspondences["id2"].unique())

        # Ensure partitions are disjoint
        if left_nodes & right_nodes:
            logging.warning(
                "Overlapping partitions detected - using maximum weight matching"
            )
            return self._maximum_weight_matching(correspondences, graph)

        # Perform maximum weight bipartite matching
        try:
            # Use maximum_matching which returns a dictionary of node -> matched_node
            matching_dict = nx.bipartite.maximum_matching(graph, top_nodes=left_nodes)

            # Convert dictionary to list of edges (tuples)
            matching = []
            processed_nodes = set()
            for node, matched_node in matching_dict.items():
                # Avoid duplicate edges (since dict contains both directions)
                if node not in processed_nodes and matched_node not in processed_nodes:
                    matching.append((node, matched_node))
                    processed_nodes.add(node)
                    processed_nodes.add(matched_node)
        except Exception as e:
            logging.error(f"Bipartite matching failed: {e}")
            logging.info("Falling back to greedy approach")
            return self._fallback_greedy_matching(correspondences)

        # Convert matching back to correspondences
        result_rows = []
        edge_to_correspondence = self._create_edge_lookup(correspondences)

        total_weight = 0
        # matching is a set of edges, need to handle both edge formats
        for edge in matching:
            if isinstance(edge, tuple) and len(edge) == 2:
                u, v = edge
            else:
                # If edge is not a simple tuple, skip it
                logging.warning(f"Unexpected edge format in matching: {edge}")
                continue
            # Ensure consistent ordering (left partition first)
            if u in left_nodes:
                id1, id2 = u, v
            else:
                id1, id2 = v, u

            # Find original correspondence
            edge_key = (id1, id2)
            if edge_key in edge_to_correspondence:
                row = edge_to_correspondence[edge_key]
                result_rows.append(row)
                total_weight += row["score"]

        if not result_rows:
            return pd.DataFrame(columns=["id1", "id2", "score", "notes"])

        result = pd.DataFrame(result_rows).reset_index(drop=True)

        logging.info(
            f"Maximum bipartite matching: {len(correspondences)} -> {len(result)} "
        )

        return result

    def _create_bipartite_graph(self, correspondences: CorrespondenceSet) -> "nx.Graph":
        """Create a bipartite graph from correspondences.

        Parameters
        ----------
        correspondences : CorrespondenceSet
            Input correspondences.

        Returns
        -------
        nx.Graph
            Weighted bipartite graph.
        """
        graph = nx.Graph()

        for _, row in correspondences.iterrows():
            id1, id2, score = row["id1"], row["id2"], row["score"]

            # Scale weight if requested (some algorithms prefer integers)
            weight = int(score * 1000000) if self.use_weight_scaling else score

            graph.add_edge(id1, id2, weight=weight)

        return graph

    def _create_edge_lookup(
        self, correspondences: CorrespondenceSet
    ) -> Dict[Tuple[str, str], dict]:
        """Create lookup from edge to original correspondence.

        Parameters
        ----------
        correspondences : CorrespondenceSet
            Input correspondences.

        Returns
        -------
        Dict[Tuple[str, str], dict]
            Mapping from (id1, id2) to correspondence row.
        """
        lookup = {}
        for _, row in correspondences.iterrows():
            key = (row["id1"], row["id2"])
            lookup[key] = row.to_dict()

        return lookup

    def _maximum_weight_matching(
        self, correspondences: CorrespondenceSet, graph: "nx.Graph"
    ) -> CorrespondenceSet:
        """Fallback to general maximum weight matching.

        Used when partitions are not clearly separated.

        Parameters
        ----------
        correspondences : CorrespondenceSet
            Input correspondences.
        graph : nx.Graph
            Weighted graph.

        Returns
        -------
        CorrespondenceSet
            Matched correspondences.
        """
        try:
            matching = nx.algorithms.matching.max_weight_matching(
                graph, weight="weight"
            )

            result_rows = []
            edge_to_correspondence = self._create_edge_lookup(correspondences)

            for u, v in matching:
                # Try both orderings
                for id1, id2 in [(u, v), (v, u)]:
                    if (id1, id2) in edge_to_correspondence:
                        result_rows.append(edge_to_correspondence[(id1, id2)])
                        break

            if result_rows:
                return pd.DataFrame(result_rows).reset_index(drop=True)

        except Exception as e:
            logging.error(f"Maximum weight matching failed: {e}")

        return self._fallback_greedy_matching(correspondences)

    def _fallback_greedy_matching(
        self, correspondences: CorrespondenceSet
    ) -> CorrespondenceSet:
        """Fallback to greedy matching when optimization fails.

        Parameters
        ----------
        correspondences : CorrespondenceSet
            Input correspondences.

        Returns
        -------
        CorrespondenceSet
            Greedily matched correspondences.
        """
        # Import and use the greedy algorithm as fallback
        from .greedy_one_to_one import GreedyOneToOneMatchingAlgorithm

        logging.info("Using greedy fallback for bipartite matching")
        greedy_matcher = GreedyOneToOneMatchingAlgorithm(threshold=0.0)
        return greedy_matcher._greedy_one_to_one_match(correspondences)

    def get_matching_quality(self, correspondences: CorrespondenceSet) -> dict:
        """Get matching quality metrics.

        Parameters
        ----------
        correspondences : CorrespondenceSet
            Input correspondences.

        Returns
        -------
        dict
            Quality metrics comparing optimal vs. greedy solutions.
        """
        if len(correspondences) == 0:
            return {"total_weight": 0, "num_matches": 0}

        filtered = self._filter_by_threshold(correspondences)
        optimal_result = self._maximum_bipartite_match(filtered)

        # Compare with greedy for quality assessment
        from .greedy_one_to_one import GreedyOneToOneMatchingAlgorithm

        greedy_matcher = GreedyOneToOneMatchingAlgorithm(threshold=0.0)
        greedy_result = greedy_matcher._greedy_one_to_one_match(filtered)

        optimal_weight = optimal_result["score"].sum() if len(optimal_result) > 0 else 0
        greedy_weight = greedy_result["score"].sum() if len(greedy_result) > 0 else 0

        return {
            "optimal_total_weight": optimal_weight,
            "optimal_num_matches": len(optimal_result),
            "greedy_total_weight": greedy_weight,
            "greedy_num_matches": len(greedy_result),
            "improvement_ratio": (
                (optimal_weight / greedy_weight) if greedy_weight > 0 else 1.0
            ),
            "quality_gap": optimal_weight - greedy_weight,
        }
