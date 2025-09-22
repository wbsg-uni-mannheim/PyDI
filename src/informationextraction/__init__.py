"""Information extraction utilities for PyDI.

This subpackage provides extractors for structured information extraction from
DataFrame text columns using regex patterns, custom functions, and built-in rules.
"""

from .autorules import RuleDiscovery, discover_fields
from .base import BaseExtractor, ExtractorPipeline
from .code import CodeExtractor
from .regex import RegexExtractor
from .rules import built_in_rules
from .evaluation import InformationExtractionEvaluator

try:
    from .llm import LLMExtractor
    _llm_available = True
except ImportError:
    _llm_available = False

__all__ = [
    'BaseExtractor',
    'ExtractorPipeline',
    'RegexExtractor',
    'CodeExtractor',
    'RuleDiscovery',
    'discover_fields',
    'built_in_rules',
    'InformationExtractionEvaluator'
]

if _llm_available:
    __all__.append('LLMExtractor')
