#!/usr/bin/env python3
"""Example: Auto-discovery of extraction rules using built-in patterns.

This script demonstrates how to use RuleDiscovery to automatically find useful
structured fields in unstructured text by applying all built-in rules and
filtering by coverage threshold.
"""

import pandas as pd
from PyDI.informationextraction import RuleDiscovery, discover_fields


def main():
    """Demonstrate auto-rule discovery with sample product data."""
    
    # Create sample data with mixed structured information
    sample_data = [
        "iPhone 14 Pro - Space Black - $999.99 - Available at apple.com",
        "Samsung Galaxy S23 Ultra for €1199.99 - Contact support@samsung.com",
        "MacBook Air M2 - Silver - Starting at $1199 (13-inch model)",
        "Dell XPS 13 laptop - Visit dell.com or call 1-800-DELL-CARE",
        "Nintendo Switch OLED - $349.99 - Red/Blue Joy-Con controllers",
        "Contact sales@bestbuy.com for bulk orders over $500.00",
        "Sony WH-1000XM4 headphones - Black - €299.99 - 30-hour battery",
        "Visit https://example.com for special deals until 2024-12-31",
        "Free shipping on orders over $50.00 - Call (555) 123-4567",
        "Product model XYZ-2024 in blue, white, black colors - 2 year warranty"
    ]
    
    df = pd.DataFrame({'product_description': sample_data})
    
    print(f"Starting auto-rule discovery on {len(df)} product descriptions\n")
    
    # Method 1: Using the convenience function
    print("=" * 60)
    print("Method 1: Using discover_fields() convenience function")
    print("=" * 60)
    
    result1 = discover_fields(
        df,
        source_column='product_description',
        categories=['money', 'contact', 'dates', 'product'],
        coverage_threshold=0.25,
        top_k=8,
        include_original=False,
        debug=True,
        out_dir="output/informationextraction/autorules_demo"
    )
    
    print(f"\nDiscovered {len(result1.columns)} high-coverage fields:")
    for col in result1.columns:
        non_null = result1[col].notna().sum()
        coverage = non_null / len(result1) * 100
        print(f"  {col}: {non_null}/{len(result1)} ({coverage:.1f}%)")
    
    print("\nSample extracted values:")
    print(result1.head().to_string(max_colwidth=40))
    
    # Method 2: Using RuleDiscovery class for more control
    print("\n\n" + "=" * 60)
    print("Method 2: Using RuleDiscovery class with metadata")
    print("=" * 60)
    
    discovery = RuleDiscovery(
        debug=True, 
        out_dir="output/informationextraction/autorules_detailed"
    )
    
    result2, metadata = discovery.extract_and_select(
        df,
        source_column='product_description',
        categories=['money', 'contact', 'product', 'measurements'],
        coverage_threshold=0.20,
        top_k=10,
        include_original=True,
        return_meta=True
    )
    
    print(f"\nEvaluation summary:")
    print(f"  Total fields evaluated: {metadata['total_fields_evaluated']}")
    print(f"  Fields selected: {len(metadata['selected_fields'])}")
    print(f"  Categories used: {metadata['categories_used']}")
    
    print(f"\nTop selected fields by coverage:")
    for field in metadata['selected_fields'][:5]:
        coverage = metadata['coverage'][field]
        print(f"  {field}: {coverage:.3f}")
    
    # Method 3: Progressive filtering with different thresholds
    print("\n\n" + "=" * 60)
    print("Method 3: Progressive filtering with different thresholds")
    print("=" * 60)
    
    thresholds = [0.1, 0.25, 0.4, 0.6]
    
    for threshold in thresholds:
        result = discover_fields(
            df,
            source_column='product_description',
            categories=['contact', 'money', 'product'],
            coverage_threshold=threshold,
            include_original=False,
            debug=False
        )
        
        print(f"Threshold {threshold:.1f}: {len(result.columns)} fields selected")
        if len(result.columns) > 0:
            print(f"  Fields: {', '.join(result.columns[:5])}")
            if len(result.columns) > 5:
                print(f"  ... and {len(result.columns) - 5} more")
    
    # Method 4: Category-specific discovery
    print("\n\n" + "=" * 60)
    print("Method 4: Category-specific discovery")
    print("=" * 60)
    
    categories = ['contact', 'money', 'product', 'dates', 'measurements']
    
    for category in categories:
        result = discover_fields(
            df,
            source_column='product_description',
            categories=[category],
            coverage_threshold=0.2,
            include_original=False,
            debug=False
        )
        
        print(f"{category.capitalize()}: {len(result.columns)} fields")
        if len(result.columns) > 0:
            # Show sample values for first field
            first_field = result.columns[0]
            sample_values = result[first_field].dropna().head(3).tolist()
            print(f"  Example ({first_field}): {sample_values}")
    
    # Save final comprehensive result
    comprehensive_result = discover_fields(
        df,
        source_column='product_description',
        coverage_threshold=0.2,
        top_k=15,
        include_original=True,
        debug=True,
        out_dir="output/informationextraction/autorules_final"
    )
    
    output_path = "output/informationextraction/autorules_final/discovered_fields.csv"
    comprehensive_result.to_csv(output_path, index=False)
    
    print(f"\n\nFinal comprehensive result saved to: {output_path}")
    print(f"Columns in final result: {list(comprehensive_result.columns)}")
    print(f"Total fields discovered: {len(comprehensive_result.columns) - 1}")  # Exclude original


if __name__ == "__main__":
    main()