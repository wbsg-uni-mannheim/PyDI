"""
Quickstart: Normalize repository data with PyDI.

What this shows:
- Load CSV/XML from the repo with provenance
- Detect column types and units (including header-derived unit)
- Normalize data and save results for sharing
 - Apply per-column transformations before normalization (discoverable API)

Run:
    python -m PyDI.examples.normalization_quickstart
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from PyDI.io.loaders import load_csv, load_xml
from PyDI.normalization.datasets import DatasetNormalizer, create_normalization_config
from PyDI.normalization.columns import ColumnTypeInference
from PyDI.normalization.transforms import Transforms as T, list_transforms


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main():
    root = repo_root()
    csv_path = root / "input" / "schemamatching" / "data" / "movie_list.csv"
    xml_path = root / "input" / "fusion" / "data" / "academy_awards.xml"

    movies = load_csv(csv_path, name="movies")
    awards = load_xml(xml_path, name="academy_awards")

    print("Movies shape:", movies.shape)
    print("Academy awards shape:", awards.shape)

    # Type inference preview
    infer = ColumnTypeInference()
    summary = infer.get_type_summary(
        infer.infer_column_types(movies.head(1000)))
    print("\nType summary (movies, sample):")
    print(summary.to_string(index=False))

    # Show available built-in transforms (discoverable UX)
    print("\nAvailable built-in transforms:")
    for t in list_transforms():
        print(f"- {t['name']}: {t['summary']}")

    # Define per-column transforms (applied BEFORE detection/normalization)
    # Keys can be column names or tuples of names to apply the same transform.
    # 1) Clean text/numerics
    transforms = {
        # clean title text
        'Film': [T.strip(), T.normalize_whitespace()],
        'exclude': T.replace({'y': True, 'n': False, '': None}),
    }

    # 2) Coerce currency-like columns to numeric and scale to millions
    money_cols = [
        'Domestic Gross', 'Foreign Gross', 'Worldwide Gross',
        'Opening Weekend', 'Box Office Average per Cinema',
        'Budget', 'Profit',
    ]
    
    to_millions = T.map(lambda v: v / 1_000_000 if pd.notna(v) else v)
    transforms[tuple(money_cols)] = [T.to_numeric(), to_millions]

    # Normalize movies dataset with transforms
    cfg = create_normalization_config(
        enable_unit_conversion=True, enable_quantity_scaling=True,
    )
    normalizer = DatasetNormalizer(cfg)
    out_dir = root / "output" / "examples" / "quickstart"
    out_dir.mkdir(parents=True, exist_ok=True)
    normalized, result = normalizer.normalize_dataset(
        movies, output_path=out_dir, column_transforms=transforms)

    # Alternative: configure transforms within the config (equivalent)
    # cfg = create_normalization_config(
    #     enable_unit_conversion=True,
    #     enable_quantity_scaling=True,
    #     column_transformations=transforms,
    # )
    # normalizer = DatasetNormalizer(cfg)
    # normalized, result = normalizer.normalize_dataset(movies, output_path=out_dir)

    print("\nNormalized shape:", normalized.shape)
    print("Overall success:", f"{result.overall_success_rate:.1%}")
    (out_dir / "movies_normalized.csv").write_text(normalized.to_csv(index=False))
    print("Saved:", out_dir / "movies_normalized.csv")
    print("\nPreview (Film, year, exclude, Budget[M], Profit[M]):")
    cols = [c for c in ['Film', 'year', 'exclude',
                        'Budget', 'Profit'] if c in normalized.columns]
    print(normalized[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
