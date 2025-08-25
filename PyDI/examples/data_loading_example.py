#!/usr/bin/env python3
"""
PyDI Data Loading Example
========================

This script demonstrates the provenance-aware data loading capabilities of
PyDI's `io` module. It shows how to load various file formats (CSV, XML, JSON)
and how provenance metadata is attached to the resulting DataFrames.

Usage:
    python -m PyDI.examples.data_loading_example
"""

import logging
from pathlib import Path

from PyDI.io import load_csv, load_xml, load_json

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demonstrate_csv_loading():
    """Demonstrate loading CSV files."""
    print("\n" + "="*60)
    print("CSV Loading Example")
    print("="*60)

    # Use the movie_list.csv from the input directory
    csv_path = Path(__file__).parent.parent.parent / \
        "input/schemamatching/data/movie_list.csv"

    if csv_path.exists():
        # Load CSV with provenance-aware wrapper
        df = load_csv(csv_path, name="movies")

        print(f"Loaded {len(df)} rows from {csv_path.name}")
        prov = df.attrs.get('provenance', {})
        print(f"Dataset name: {prov.get('dataset_name', 'N/A')}")
        print(f"Columns: {list(df.columns)}")
        print(f"Sample data:")
        print(df.head(3))

        # Show metadata
        print(f"\nMetadata:")
        for key, value in prov.items():
            print(f"  {key}: {value}")
    else:
        print(f"CSV file not found: {csv_path}")


def demonstrate_xml_loading():
    """Demonstrate loading and flattening XML files."""
    print("\n" + "="*60)
    print("XML Loading Example (with nested structure flattening)")
    print("="*60)

    # Use the academy_awards.xml from the input directory
    xml_path = Path(__file__).parent.parent.parent / \
        "input/entitymatching/data/academy_awards.xml"

    if xml_path.exists():
        # Load XML with provenance-aware wrapper
        df = load_xml(xml_path, name="academy_awards")

        print(f"Loaded {len(df)} rows from {xml_path.name}")
        prov = df.attrs.get('provenance', {})
        print(f"Dataset name: {prov.get('dataset_name', 'N/A')}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nSample data:")
        print(df.head(6))
    else:
        print(f"XML file not found: {xml_path}")


def demonstrate_json_loading():
    """Demonstrate loading JSON files."""
    print("\n" + "="*60)
    print("JSON Loading Example (with nested structure flattening)")
    print("="*60)

    # Use the testTable.json from the winter directory
    json_path = Path(__file__).parent.parent.parent / \
        "winter/winter-framework/src/test/resource/testTable.json"

    if json_path.exists():
        try:
            df = load_json(json_path, name="hockey_stats")
            print(f"Loaded {len(df)} rows from {json_path.name}")
            prov = df.attrs.get('provenance', {})
            print(f"Dataset name: {prov.get('dataset_name', 'N/A')}")
            print(f"Columns: {list(df.columns)}")
            print(f"\nSample JSON data:")
            print(df.head(3))
        except Exception as e:
            print(f"Failed to load JSON file: {e}")
    else:
        print(f"JSON file not found: {json_path}")


def demonstrate_convenience_function():
    """Demonstrate the convenience load_data function."""
    print("\n" + "="*60)
    print("Convenience Function Example")
    print("="*60)

    # Use any available file
    csv_path = Path(__file__).parent.parent.parent / \
        "input/schemamatching/data/movie_list.csv"

    if csv_path.exists():
        # Use the provenance-aware csv loader
        df = load_csv(csv_path, name="movies_convenience")

        prov = df.attrs.get('provenance', {})
        id_col = prov.get('id_column_name', '<none>')
        print(f"Loaded with CSV loader: {len(df)} rows")
        print(f"Dataset name: {prov.get('dataset_name', 'N/A')}")
        print(f"ID column: {id_col} (present: {id_col in df.columns})")
        print(f"Provenance recorded: {'provenance' in df.attrs}")
    else:
        print(f"File not found for convenience demo: {csv_path}")


def main():
    """Run all data loading examples."""
    print("PyDI Data Loading Examples")
    print("This script demonstrates loading various file formats with automatic flattening.")

    try:
        demonstrate_csv_loading()
        #demonstrate_xml_loading()
        #demonstrate_json_loading()
        #demonstrate_convenience_function()

        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)

    except Exception as e:
        logger.error(f"Example failed: {e}")
        raise


if __name__ == "__main__":
    main()
