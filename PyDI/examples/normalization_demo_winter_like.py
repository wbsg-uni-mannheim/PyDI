"""
Winter-like normalization demo using PyDI on bundled repo data.

This script demonstrates:
- Provenance-aware loading from XML and CSV
- Header normalization and header-derived unit detection
- Locale-aware numeric parsing (commas as decimals, apostrophe/space groupings)
- Unit-aware numeric normalization and type detection
- Dataset-level normalization with a concise summary

Data sources used from the repository:
- input/fusion/data/academy_awards.xml
- input/fusion/data/actors.xml
- input/schemamatching/data/movie_list.csv

Run:
    python -m PyDI.examples.normalization_demo_winter_like
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from PyDI.io.loaders import load_xml, load_csv
from PyDI.normalization.datasets import (
    DatasetNormalizer,
    create_normalization_config,
)
from PyDI.normalization.text import HeaderNormalizer
from PyDI.normalization.columns import ColumnTypeInference
from PyDI.normalization.types import NumericParser


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    # PyDI/examples/ -> repo root two levels up
    return here.parents[2]


def load_demo_frames() -> dict[str, pd.DataFrame]:
    root = _repo_root()

    paths = {
        "academy_awards": root / "input" / "fusion" / "data" / "academy_awards.xml",
        "actors": root / "input" / "fusion" / "data" / "actors.xml",
        "movies_csv": root / "input" / "schemamatching" / "data" / "movie_list.csv",
    }

    frames: dict[str, pd.DataFrame] = {}

    # Load XML with flattening
    frames["academy_awards"] = load_xml(
        paths["academy_awards"], name="academy_awards")
    frames["actors"] = load_xml(paths["actors"], name="actors")

    # Load CSV
    frames["movies_csv"] = load_csv(paths["movies_csv"], name="movies")

    return frames


def demonstrate_header_normalization(df: pd.DataFrame) -> pd.DataFrame:
    print("\n=== Header Normalization ===")
    print("Original columns:", list(df.columns)[:10])
    hn = HeaderNormalizer(lowercase=True, remove_brackets=True)
    cleaned = hn.normalize_dataframe_headers(df)
    print("Normalized columns:", list(cleaned.columns)[:10])
    return cleaned


def demonstrate_locale_numeric_parser():
    print("\n=== Locale-aware Numeric Parsing ===")
    samples = [
        "1,234.56",  # US
        "1.234,56",  # EU
        "1 234,56",  # EU with space grouping
        "1\xa0234,56",  # NBSP grouping
        "1'234.56",  # Swiss
        "(1.234,56)",  # EU with negative parentheses
        "$1,234.56",  # currency US
        "€ 1.234,56",  # currency EU
        "12,5%",  # EU percent
    ]

    parser = NumericParser()
    for s in samples:
        val = parser.parse_numeric(s)
        print(f"{s!r} -> {val}")


def demonstrate_dataset_normalization(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    print("\n=== Dataset Normalization ===")
    config = create_normalization_config(
        enable_unit_conversion=True,
        enable_quantity_scaling=True,
        normalize_text=True,
        standardize_nulls=True,
    )

    normalizer = DatasetNormalizer(config)
    normalized_df, result = normalizer.normalize_dataset(
        df, output_path=out_dir)

    print("Original shape:", result.original_shape)
    print("Normalized shape:", result.normalized_shape)
    print("Overall success rate:", f"{result.overall_success_rate:.1%}")

    # Show a few detected types
    print("\nDetected column types (sample):")
    for cr in result.column_results[:8]:
        unit_info = f", unit={cr.specific_unit}" if cr.specific_unit else ""
        print(
            f"- {cr.normalized_name}: {cr.detected_type.value} (conf={cr.confidence:.2f}{unit_info})")

    return normalized_df


def main():
    frames = load_demo_frames()
    out_dir = _repo_root() / "output" / "examples" / "normalization_demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pick a moderately sized frame: movies_csv
    movies = frames["movies_csv"].copy()
    movies = demonstrate_header_normalization(movies)

    demonstrate_locale_numeric_parser()

    normalized = demonstrate_dataset_normalization(movies, out_dir)
    out_csv = out_dir / "movies_normalized.csv"
    normalized.to_csv(out_csv, index=False)
    print("\nSaved normalized dataset to:", out_csv)


if __name__ == "__main__":
    main()
