"""
Blocking subpackage: base interfaces and simple blocking strategies.
"""

from .base import BaseBlocker
from .noblocking import NoBlocker
from .standard import StandardBlocker
from .sorted_neighbourhood import SortedNeighbourhoodBlocker
from .token_blocking import TokenBlocker
from .embedding import EmbeddingBlocker

__all__ = [
    "BaseBlocker",
    "NoBlocker",
    "StandardBlocker",
    "SortedNeighbourhoodBlocker",
    "TokenBlocker",
    "EmbeddingBlocker",
]


