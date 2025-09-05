"""
Blocking subpackage: base interfaces and simple blocking strategies.
"""

from .base import BaseBlocker
from .noblocking import NoBlocking
from .standard import StandardBlocking
from .sorted_neighbourhood import SortedNeighbourhood
from .token_blocking import TokenBlocking
from .embedding import EmbeddingBlocking

__all__ = [
    "BaseBlocker",
    "NoBlocking",
    "StandardBlocking",
    "SortedNeighbourhood",
    "TokenBlocking",
    "EmbeddingBlocking",
]


