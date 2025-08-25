#!/usr/bin/env python3
"""
Simple test script for schema matching functionality
"""

import sys
import pandas as pd
from pathlib import Path

# Add PyDI to path
sys.path.insert(0, str(Path(__file__).parent))

from PyDI.schemamatching import LabelBasedSchemaMatcher, InstanceBasedSchemaMatcher


def create_test_datasets():
    """Create simple test datasets for schema matching."""
    # Dataset 1 - Movies from IMDB
    df1 = pd.DataFrame({
        'movie_id': ['m1', 'm2', 'm3'],
        'title': ['The Matrix', 'Inception', 'Pulp Fiction'],
        'director': ['Wachowski', 'Nolan', 'Tarantino'],
        'year': [1999, 2010, 1994],
        'genre': ['Sci-Fi', 'Sci-Fi', 'Crime']
    })
    df1.attrs["dataset_name"] = "imdb"
    
    # Dataset 2 - Movies from Rotten Tomatoes  
    df2 = pd.DataFrame({
        'film_id': ['rt1', 'rt2', 'rt3'],
        'name': ['The Matrix', 'Inception', 'Pulp Fiction'], 
        'director_name': ['Lana Wachowski', 'Christopher Nolan', 'Quentin Tarantino'],
        'release_year': [1999, 2010, 1994],
        'category': ['Science Fiction', 'Science Fiction', 'Crime Drama']
    })
    df2.attrs["dataset_name"] = "rottentomatoes"
    
    return [df1, df2]


def test_label_based_matching():
    """Test label-based schema matching."""
    print("\n=== TESTING LABEL-BASED MATCHING ===")
    
    datasets = create_test_datasets()
    
    # Test different similarity functions
    similarity_functions = ["jaccard", "levenshtein", "jaro_winkler"]
    
    for sim_func in similarity_functions:
        print(f"\n--- Testing {sim_func} similarity ---")
        
        try:
            matcher = LabelBasedSchemaMatcher(
                similarity_function=sim_func,
                tokenize=True
            )
            
            correspondences = matcher.match(datasets, threshold=0.3)
            print(f"Found {len(correspondences)} correspondences")
            
            if not correspondences.empty:
                for _, match in correspondences.iterrows():
                    print(f"  {match['source_column']} <-> {match['target_column']} "
                          f"(score: {match['score']:.3f})")
            
        except Exception as e:
            print(f"Error with {sim_func}: {e}")


def test_instance_based_matching():
    """Test instance-based schema matching."""
    print("\n=== TESTING INSTANCE-BASED MATCHING ===")
    
    datasets = create_test_datasets()
    
    # Test different vector methods
    methods = [
        ("term_frequencies", "cosine"),
        ("binary_occurrence", "jaccard"),
    ]
    
    for vector_method, similarity in methods:
        print(f"\n--- Testing {vector_method} + {similarity} ---")
        
        try:
            matcher = InstanceBasedSchemaMatcher(
                vector_creation_method=vector_method,
                similarity_function=similarity,
                max_sample_size=100
            )
            
            correspondences = matcher.match(datasets, threshold=0.1)
            print(f"Found {len(correspondences)} correspondences")
            
            if not correspondences.empty:
                for _, match in correspondences.iterrows():
                    print(f"  {match['source_column']} <-> {match['target_column']} "
                          f"(score: {match['score']:.3f})")
            
        except Exception as e:
            print(f"Error with {vector_method}/{similarity}: {e}")


def main():
    """Run all tests."""
    print("PyDI Schema Matching Test")
    print("=" * 50)
    
    # Test label-based matching
    test_label_based_matching()
    
    # Test instance-based matching  
    test_instance_based_matching()
    
    print("\n" + "=" * 50)
    print("Tests completed!")


if __name__ == "__main__":
    main()