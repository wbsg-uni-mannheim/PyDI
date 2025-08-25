#!/usr/bin/env python3
"""
Schema Matching Example Script

This script demonstrates the three schema matching strategies implemented in PyDI:
1. Label-based matching using string similarity
2. Instance-based matching using value distributions
3. Duplicate-based matching using known record correspondences

Usage:
    python schema_matching_example.py [input_directory]
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Add PyDI to path if running as script
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from PyDI.schemamatching import (
    LabelBasedSchemaMatcher,
    InstanceBasedSchemaMatcher,
    DuplicateBasedSchemaMatcher,
    SchemaMappingEvaluator,
)
from PyDI.utils.xml_loader import load_music_datasets

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def run_label_based_matching(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Run label-based schema matching on the datasets."""
    print("\n" + "="*60)
    print("LABEL-BASED SCHEMA MATCHING")
    print("="*60)
    
    # Create matcher with different similarity functions
    matchers = {
        "jaccard": LabelBasedSchemaMatcher(similarity_function="jaccard", tokenize=True),
        "levenshtein": LabelBasedSchemaMatcher(similarity_function="levenshtein", tokenize=False),
        "jaro_winkler": LabelBasedSchemaMatcher(similarity_function="jaro_winkler", tokenize=False),
    }
    
    all_results = []
    
    for similarity_name, matcher in matchers.items():
        print(f"\n--- Using {similarity_name} similarity ---")
        
        # Run matching on all dataset combinations
        dataset_list = list(datasets.values())
        correspondences = matcher.match(dataset_list, threshold=0.3)
        
        print(f"Found {len(correspondences)} correspondences")
        
        if not correspondences.empty:
            # Add similarity function to results
            correspondences["similarity_function"] = similarity_name
            all_results.append(correspondences)
            
            # Show top matches
            top_matches = correspondences.nlargest(10, 'score')
            for _, match in top_matches.iterrows():
                print(f"  {match['source_dataset']}.{match['source_column']} <-> "
                      f"{match['target_dataset']}.{match['target_column']} "
                      f"(score: {match['score']:.3f})")
    
    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()


def run_instance_based_matching(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Run instance-based schema matching on the datasets."""
    print("\n" + "="*60)
    print("INSTANCE-BASED SCHEMA MATCHING")
    print("="*60)
    
    # Create matcher with different vector methods
    matchers = {
        "term_frequencies_cosine": InstanceBasedSchemaMatcher(
            vector_creation_method="term_frequencies",
            similarity_function="cosine",
            max_sample_size=500
        ),
        "binary_jaccard": InstanceBasedSchemaMatcher(
            vector_creation_method="binary_occurrence", 
            similarity_function="jaccard",
            max_sample_size=500
        ),
        "tfidf_cosine": InstanceBasedSchemaMatcher(
            vector_creation_method="tfidf",
            similarity_function="cosine",
            max_sample_size=500
        ),
    }
    
    all_results = []
    
    for method_name, matcher in matchers.items():
        print(f"\n--- Using {method_name} method ---")
        
        # Run matching on all dataset combinations  
        dataset_list = list(datasets.values())
        correspondences = matcher.match(dataset_list, threshold=0.1)
        
        print(f"Found {len(correspondences)} correspondences")
        
        if not correspondences.empty:
            # Add method to results
            correspondences["vector_method"] = method_name
            all_results.append(correspondences)
            
            # Show top matches
            top_matches = correspondences.nlargest(10, 'score')
            for _, match in top_matches.iterrows():
                print(f"  {match['source_dataset']}.{match['source_column']} <-> "
                      f"{match['target_dataset']}.{match['target_column']} "
                      f"(score: {match['score']:.3f})")
    
    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()


def create_sample_correspondences(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create sample record correspondences for duplicate-based matching."""
    # This is a simplified example - in practice, you would load actual correspondences
    correspondences = []
    
    # Create some artificial correspondences based on matching artists/names
    dataset_names = list(datasets.keys())
    
    if len(dataset_names) >= 2:
        df1 = datasets[dataset_names[0]]
        df2 = datasets[dataset_names[1]]
        
        # Try to find potential matches based on artist and name similarity
        for i1, row1 in df1.head(20).iterrows():
            artist1 = str(row1.get('artist', ''))
            name1 = str(row1.get('name', ''))
            
            for i2, row2 in df2.head(20).iterrows():
                artist2 = str(row2.get('artist', ''))
                name2 = str(row2.get('name', ''))
                
                # Simple matching logic
                if (artist1.lower() == artist2.lower() and 
                    artist1 and artist2 and 
                    artist1 != 'nan'):
                    
                    correspondences.append({
                        'id1': row1.get('id', row1.get('_id', f'rec_{i1}')),
                        'id2': row2.get('id', row2.get('_id', f'rec_{i2}')),
                        'score': 1.0
                    })
    
    return pd.DataFrame(correspondences)


def run_duplicate_based_matching(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Run duplicate-based schema matching on the datasets."""
    print("\n" + "="*60)
    print("DUPLICATE-BASED SCHEMA MATCHING")
    print("="*60)
    
    # Create sample correspondences
    correspondences = create_sample_correspondences(datasets)
    
    if correspondences.empty:
        print("No record correspondences available - skipping duplicate-based matching")
        return pd.DataFrame()
    
    print(f"Using {len(correspondences)} record correspondences")
    
    # Create matcher
    matcher = DuplicateBasedSchemaMatcher(
        vote_aggregation="majority",
        value_comparison="normalized",
        min_votes=1
    )
    
    # Run matching on the first two datasets
    dataset_list = list(datasets.values())[:2]
    
    try:
        schema_correspondences = matcher.match(
            dataset_list,
            correspondences=correspondences,
            threshold=0.1
        )
        
        print(f"Found {len(schema_correspondences)} schema correspondences")
        
        if not schema_correspondences.empty:
            # Show all matches
            for _, match in schema_correspondences.iterrows():
                print(f"  {match['source_dataset']}.{match['source_column']} <-> "
                      f"{match['target_dataset']}.{match['target_column']} "
                      f"(score: {match['score']:.3f}, {match.get('notes', '')})")
        
        return schema_correspondences
        
    except Exception as e:
        print(f"Error in duplicate-based matching: {e}")
        return pd.DataFrame()


def save_results(results: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Save matching results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for method_name, df in results.items():
        if not df.empty:
            output_file = output_dir / f"{method_name}_results.csv"
            df.to_csv(output_file, index=False)
            print(f"Saved {method_name} results to: {output_file}")


def main(input_directory: str = None):
    """Main function to run schema matching examples."""
    if input_directory is None:
        # Default to project input directory
        script_dir = Path(__file__).parent
        input_directory = script_dir.parent.parent / "input"
    
    input_dir = Path(input_directory)
    
    print("PyDI Schema Matching Example")
    print("="*60)
    print(f"Input directory: {input_dir}")
    
    # Load datasets
    try:
        datasets = load_music_datasets(input_dir)
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return
    
    if not datasets:
        print("No datasets loaded - exiting")
        return
    
    print(f"\nLoaded {len(datasets)} datasets:")
    for name, df in datasets.items():
        print(f"  {name}: {len(df)} records, {len(df.columns)} columns")
        print(f"    Columns: {list(df.columns)[:10]}...")
    
    # Run all matching methods
    results = {}
    
    # 1. Label-based matching
    try:
        label_results = run_label_based_matching(datasets)
        if not label_results.empty:
            results["label_based"] = label_results
    except Exception as e:
        print(f"Error in label-based matching: {e}")
    
    # 2. Instance-based matching
    try:
        instance_results = run_instance_based_matching(datasets)
        if not instance_results.empty:
            results["instance_based"] = instance_results
    except Exception as e:
        print(f"Error in instance-based matching: {e}")
    
    # 3. Duplicate-based matching
    try:
        duplicate_results = run_duplicate_based_matching(datasets)
        if not duplicate_results.empty:
            results["duplicate_based"] = duplicate_results
    except Exception as e:
        print(f"Error in duplicate-based matching: {e}")
    
    # Save results
    output_dir = Path("output") / "schema_matching"
    save_results(results, output_dir)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for method_name, df in results.items():
        print(f"{method_name}: {len(df)} correspondences found")
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    import sys
    input_dir = sys.argv[1] if len(sys.argv) > 1 else None
    main(input_dir)