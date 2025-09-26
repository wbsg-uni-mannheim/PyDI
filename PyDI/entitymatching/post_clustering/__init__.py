"""
Post-clustering and global optimization methods for entity matching.

This module provides algorithms to refine entity correspondences through
clustering-based grouping and global optimization approaches. All methods
take correspondences as input and return refined correspondences in the
same format for seamless integration with PyDI's evaluation framework.
"""

# Base classes
from .base import BasePostClusterer

# Clustering-based methods
from .connected_components import ConnectedComponentClusterer
from .centre_clusterer import CentreClusterer
from .hierarchical_clusterer import HierarchicalClusterer

# Global optimization methods
from .greedy_one_to_one import GreedyOneToOneMatchingAlgorithm
from .maximum_bipartite_matching import MaximumBipartiteMatching
from .stable_matching import StableMatching

__all__ = [
    "BasePostClusterer",
    "ConnectedComponentClusterer",
    "CentreClusterer",
    "HierarchicalClusterer",
    "GreedyOneToOneMatchingAlgorithm",
    "MaximumBipartiteMatching",
    "StableMatching",
]