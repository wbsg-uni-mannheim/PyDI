"""
Comprehensive similarity function registry for textdistance metrics.

This module provides a centralized registry of all textdistance similarity
metrics, organized by category and optimized for data integration use cases
across schema matching, entity matching, and other PyDI modules.
"""

from __future__ import annotations

import logging
import re
from typing import Callable, Dict, List, Optional, Set, Union

import textdistance


class SimilarityRegistry:
    """Registry for textdistance similarity metrics.
    
    Provides a centralized way to access and validate textdistance similarity
    functions, categorized by their characteristics and use cases across
    all PyDI modules.
    """
    
    # Edit-based algorithms - good for typos and character-level variations
    EDIT_BASED = {
        "hamming": textdistance.hamming,
        "levenshtein": textdistance.levenshtein.normalized_similarity,
        "damerau_levenshtein": textdistance.damerau_levenshtein.normalized_similarity,
        "jaro_winkler": textdistance.jaro_winkler,
        "jaro": textdistance.jaro,
        "strcmp95": textdistance.strcmp95,
        "needleman_wunsch": textdistance.needleman_wunsch.normalized_similarity,
        "gotoh": textdistance.gotoh.normalized_similarity,
        "smith_waterman": textdistance.smith_waterman.normalized_similarity,
        "mlipns": textdistance.mlipns.normalized_similarity,
        "editex": textdistance.editex.normalized_similarity,
    }
    
    # Token-based algorithms - good for multi-word strings and sets
    TOKEN_BASED = {
        "jaccard": textdistance.jaccard,
        "sorensen_dice": textdistance.sorensen,
        "tversky": textdistance.tversky,
        "overlap": textdistance.overlap.normalized_similarity,
        "tanimoto": textdistance.tanimoto,
        "cosine": textdistance.cosine,
        "monge_elkan": textdistance.monge_elkan,
        "bag": textdistance.bag.normalized_similarity,
    }
    
    # Sequence-based algorithms - good for substring matching
    SEQUENCE_BASED = {
        "lcsseq": textdistance.lcsseq.normalized_similarity,
        "lcsstr": textdistance.lcsstr.normalized_similarity,
        "ratcliff_obershelp": textdistance.ratcliff_obershelp,
    }
    
    # Simple algorithms - good for specific patterns
    SIMPLE = {
        "prefix": textdistance.prefix.normalized_similarity,
        "postfix": textdistance.postfix.normalized_similarity,
        "length": textdistance.length.normalized_similarity,
        "identity": textdistance.identity,
    }
    
    # Phonetic algorithms - good for similar-sounding strings
    PHONETIC = {
        "mra": textdistance.mra,
    }
    
    # All algorithms combined
    ALL_ALGORITHMS = {
        **EDIT_BASED,
        **TOKEN_BASED,
        **SEQUENCE_BASED,
        **SIMPLE,
        **PHONETIC,
    }
    
    # Categories mapping
    CATEGORIES = {
        "edit": EDIT_BASED,
        "token": TOKEN_BASED,
        "sequence": SEQUENCE_BASED,
        "simple": SIMPLE,
        "phonetic": PHONETIC,
        "all": ALL_ALGORITHMS,
    }
    
    # Recommended algorithms for different use cases
    SCHEMA_MATCHING_RECOMMENDED = {
        "label": [
            "jaro_winkler", "levenshtein", "damerau_levenshtein", 
            "jaccard", "sorensen_dice", "prefix", "postfix"
        ],
        "instance": [
            "jaccard", "cosine", "sorensen_dice", "jaro_winkler", 
            "levenshtein", "overlap", "ratcliff_obershelp"
        ],
        "duplicate": [
            "jaro_winkler", "levenshtein", "damerau_levenshtein",
            "jaccard", "sorensen_dice", "monge_elkan"
        ]
    }
    
    ENTITY_MATCHING_RECOMMENDED = [
        "jaro_winkler", "levenshtein", "jaccard", "sorensen_dice",
        "monge_elkan", "cosine", "overlap", "damerau_levenshtein"
    ]

    # Functions that benefit from tokenization (token-based algorithms)
    TOKENIZABLE_FUNCTIONS = {
        "jaccard", "sorensen_dice", "tversky", "overlap", "tanimoto",
        "cosine", "monge_elkan", "bag"
    }

    # Default tokenization strategies
    TOKENIZATION_STRATEGIES = {
        "char": lambda text: text,  # Character-level (no tokenization)
        "word": lambda text: text.split() if isinstance(text, str) else text,
        "ngram_2": lambda text: [text[i:i+2] for i in range(len(text)-1)] if isinstance(text, str) else text,
        "ngram_3": lambda text: [text[i:i+3] for i in range(len(text)-2)] if isinstance(text, str) else text,
    }

    @classmethod
    def get_function(cls, name: str, tokenization: str = "char") -> Callable:
        """Get a similarity function by name with optional tokenization.

        Parameters
        ----------
        name : str
            Name of the similarity function.
        tokenization : str or callable, optional
            Tokenization strategy. Can be:
            - "char": Character-level (default, no tokenization)
            - "word": Whitespace tokenization
            - "ngram_2", "ngram_3": N-gram tokenization
            - Callable: Custom tokenizer function

        Returns
        -------
        Callable
            The textdistance similarity function, optionally wrapped with tokenization.

        Raises
        ------
        ValueError
            If the function name or tokenization strategy is not recognized.
        """
        if name not in cls.ALL_ALGORITHMS:
            raise ValueError(
                f"Unknown similarity function: {name}. "
                f"Available functions: {list(cls.ALL_ALGORITHMS.keys())}"
            )

        base_func = cls.ALL_ALGORITHMS[name]

        # Apply tokenization if requested and function supports it
        if tokenization != "char" and name in cls.TOKENIZABLE_FUNCTIONS:
            return cls._wrap_with_tokenization(base_func, tokenization)
        elif tokenization != "char" and name not in cls.TOKENIZABLE_FUNCTIONS:
            logging.warning(
                f"Tokenization '{tokenization}' not applicable to function '{name}' "
                f"(only supported for: {sorted(cls.TOKENIZABLE_FUNCTIONS)}). "
                "Using character-level processing."
            )

        return base_func

    @classmethod
    def _wrap_with_tokenization(cls, base_func: Callable, tokenization: Union[str, Callable]) -> Callable:
        """Wrap a textdistance function with tokenization.

        Parameters
        ----------
        base_func : Callable
            The base textdistance function.
        tokenization : str or callable
            Tokenization strategy or custom tokenizer function.

        Returns
        -------
        Callable
            Wrapped function that applies tokenization before similarity computation.
        """
        # Get tokenizer function
        if callable(tokenization):
            tokenizer = tokenization
        elif tokenization in cls.TOKENIZATION_STRATEGIES:
            tokenizer = cls.TOKENIZATION_STRATEGIES[tokenization]
        else:
            raise ValueError(
                f"Unknown tokenization strategy: {tokenization}. "
                f"Available strategies: {list(cls.TOKENIZATION_STRATEGIES.keys())} "
                "or provide a callable tokenizer."
            )

        def tokenized_similarity(s1: str, s2: str) -> float:
            """Compute similarity with tokenization."""
            try:
                # Handle None/empty inputs
                if not s1 or not s2:
                    return 0.0 if s1 != s2 else 1.0

                # Apply tokenization
                tokens1 = tokenizer(s1)
                tokens2 = tokenizer(s2)

                # Handle empty token lists
                if not tokens1 and not tokens2:
                    return 1.0
                elif not tokens1 or not tokens2:
                    return 0.0

                # Compute similarity on tokens
                return float(base_func(tokens1, tokens2))

            except Exception as e:
                logging.warning(f"Error in tokenized similarity computation: {e}")
                return 0.0

        return tokenized_similarity

    @classmethod
    def get_functions_by_category(cls, category: str) -> Dict[str, Callable]:
        """Get all functions in a specific category.
        
        Parameters
        ----------
        category : str
            Category name ("edit", "token", "sequence", "simple", "phonetic", "all").
            
        Returns
        -------
        Dict[str, Callable]
            Dictionary mapping function names to functions.
            
        Raises
        ------
        ValueError
            If the category is not recognized.
        """
        if category not in cls.CATEGORIES:
            raise ValueError(
                f"Unknown category: {category}. "
                f"Available categories: {list(cls.CATEGORIES.keys())}"
            )
        return cls.CATEGORIES[category].copy()
    
    @classmethod
    def get_recommended_functions(cls, use_case: str, matcher_type: Optional[str] = None) -> List[str]:
        """Get recommended functions for a specific use case.
        
        Parameters
        ----------
        use_case : str
            Use case ("schema_matching", "entity_matching").
        matcher_type : str, optional
            For schema matching: "label", "instance", "duplicate".
            Ignored for other use cases.
            
        Returns
        -------
        List[str]
            List of recommended function names.
            
        Raises
        ------
        ValueError
            If the use case or matcher type is not recognized.
        """
        if use_case == "schema_matching":
            if matcher_type not in cls.SCHEMA_MATCHING_RECOMMENDED:
                raise ValueError(
                    f"Unknown schema matcher type: {matcher_type}. "
                    f"Available types: {list(cls.SCHEMA_MATCHING_RECOMMENDED.keys())}"
                )
            return cls.SCHEMA_MATCHING_RECOMMENDED[matcher_type].copy()
        elif use_case == "entity_matching":
            return cls.ENTITY_MATCHING_RECOMMENDED.copy()
        else:
            raise ValueError(
                f"Unknown use case: {use_case}. "
                f"Available use cases: schema_matching, entity_matching"
            )
    
    @classmethod
    def list_available_functions(cls, category: Optional[str] = None) -> List[str]:
        """List all available similarity function names.
        
        Parameters
        ----------
        category : str, optional
            Specific category to list. If None, lists all functions.
            
        Returns
        -------
        List[str]
            Sorted list of function names.
        """
        if category is None:
            return sorted(cls.ALL_ALGORITHMS.keys())
        else:
            return sorted(cls.get_functions_by_category(category).keys())
    
    @classmethod
    def validate_functions(cls, function_names: Union[str, List[str]]) -> List[str]:
        """Validate and normalize function names.
        
        Parameters
        ----------
        function_names : str or List[str]
            Single function name or list of function names.
            
        Returns
        -------
        List[str]
            List of validated function names.
            
        Raises
        ------
        ValueError
            If any function name is not recognized.
        """
        if isinstance(function_names, str):
            function_names = [function_names]
        
        for name in function_names:
            if name not in cls.ALL_ALGORITHMS:
                raise ValueError(
                    f"Unknown similarity function: {name}. "
                    f"Available functions: {list(cls.ALL_ALGORITHMS.keys())}"
                )
        
        return function_names
    
    @classmethod
    def get_function_info(cls, name: str) -> Dict[str, str]:
        """Get information about a specific function.
        
        Parameters
        ----------
        name : str
            Name of the similarity function.
            
        Returns
        -------
        Dict[str, str]
            Dictionary with function information.
        """
        if name not in cls.ALL_ALGORITHMS:
            raise ValueError(f"Unknown similarity function: {name}")
        
        # Determine category
        category = None
        for cat_name, functions in cls.CATEGORIES.items():
            if cat_name != "all" and name in functions:
                category = cat_name
                break
        
        return {
            "name": name,
            "category": category or "unknown",
            "description": cls._get_function_description(name),
        }
    
    @classmethod
    def list_tokenization_strategies(cls) -> List[str]:
        """List all available tokenization strategies.

        Returns
        -------
        List[str]
            Sorted list of tokenization strategy names.
        """
        return sorted(cls.TOKENIZATION_STRATEGIES.keys())

    @classmethod
    def is_tokenizable(cls, function_name: str) -> bool:
        """Check if a function supports tokenization.

        Parameters
        ----------
        function_name : str
            Name of the similarity function.

        Returns
        -------
        bool
            True if the function supports tokenization, False otherwise.
        """
        return function_name in cls.TOKENIZABLE_FUNCTIONS

    @classmethod
    def _get_function_description(cls, name: str) -> str:
        """Get description for a similarity function."""
        descriptions = {
            # Edit-based
            "hamming": "Character-level edit distance for strings of equal length",
            "levenshtein": "Standard edit distance with insertions, deletions, substitutions",
            "damerau_levenshtein": "Edit distance with transpositions allowed",
            "jaro_winkler": "String similarity with prefix scaling bonus",
            "jaro": "String similarity based on matching characters",
            "strcmp95": "Enhanced Jaro-Winkler with additional adjustments", 
            "needleman_wunsch": "Global sequence alignment algorithm",
            "gotoh": "Sequence alignment with affine gap penalties",
            "smith_waterman": "Local sequence alignment algorithm",
            "mlipns": "Modified LCS-based similarity",
            "editex": "Phonetic-aware edit distance",
            
            # Token-based
            "jaccard": "Set intersection over union",
            "sorensen_dice": "Overlap coefficient with different weighting",
            "tversky": "Generalized Jaccard with asymmetric weights",
            "overlap": "Shared elements divided by minimum set size",
            "tanimoto": "Variant of Jaccard similarity",
            "cosine": "Vector space cosine similarity",
            "monge_elkan": "Token-level approximate string matching",
            "bag": "Frequency-based token similarity",
            
            # Sequence-based
            "lcsseq": "Longest Common Subsequence for sequences",
            "lcsstr": "Longest Common Substring similarity",
            "ratcliff_obershelp": "Pattern recognition based similarity",
            
            # Simple
            "prefix": "Common prefix length ratio",
            "postfix": "Common suffix length ratio", 
            "length": "Length-based similarity",
            "identity": "Exact string identity check",
            
            # Phonetic
            "mra": "Match Rating Approach phonetic similarity",
        }
        
        return descriptions.get(name, "No description available")


def get_similarity_function(name: str, tokenization: str = "char") -> Callable:
    """Convenience function to get a similarity function by name with optional tokenization.

    Parameters
    ----------
    name : str
        Name of the similarity function.
    tokenization : str or callable, optional
        Tokenization strategy (default: "char").

    Returns
    -------
    Callable
        The textdistance similarity function, optionally with tokenization.
    """
    return SimilarityRegistry.get_function(name, tokenization)


def list_similarity_functions(category: Optional[str] = None) -> List[str]:
    """Convenience function to list available similarity functions.
    
    Parameters
    ----------
    category : str, optional
        Specific category to list.
        
    Returns
    -------
    List[str]
        Sorted list of function names.
    """
    return SimilarityRegistry.list_available_functions(category)