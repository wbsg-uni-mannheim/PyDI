"""
Dataset analysis utilities for data fusion in PyDI.

This module provides comprehensive analysis capabilities for datasets before
and after fusion, including attribute coverage, schema comparison, and
conflict detection.
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def analyze_attribute_coverage(
    datasets: List[pd.DataFrame],
    dataset_names: Optional[List[str]] = None,
    include_samples: bool = True,
    max_sample_length: int = 50,
    sample_count: int = 2
) -> pd.DataFrame:
    """
    Analyze attribute coverage across multiple datasets.

    This function creates a comprehensive analysis of which attributes exist
    in which datasets, their coverage percentages, and sample values.

    Parameters
    ----------
    datasets : List[pd.DataFrame]
        List of datasets to analyze.
    dataset_names : Optional[List[str]]
        Names for each dataset. If None, uses dataset_0, dataset_1, etc.
    include_samples : bool, default True
        Whether to include sample values in the analysis.
    max_sample_length : int, default 50
        Maximum length for sample value strings.
    sample_count : int, default 2
        Number of sample values to include per attribute.

    Returns
    -------
    pd.DataFrame
        DataFrame with coverage analysis results.

    Examples
    --------
    >>> coverage_df = analyze_attribute_coverage(
    ...     [df1, df2, df3],
    ...     dataset_names=['Source A', 'Source B', 'Source C']
    ... )
    >>> print(coverage_df)
    """
    if not datasets:
        raise ValueError("At least one dataset must be provided")

    if dataset_names is None:
        dataset_names = [f"dataset_{i}" for i in range(len(datasets))]

    if len(datasets) != len(dataset_names):
        raise ValueError(
            "Number of datasets must match number of dataset names")

    # Get all unique attributes across datasets
    all_attributes = set()
    for df in datasets:
        all_attributes.update(df.columns)

    # Create detailed coverage analysis
    coverage_data = []
    for attr in sorted(all_attributes):
        attr_info = {'attribute': attr}

        for df, name in zip(datasets, dataset_names):
            if attr in df.columns:
                non_null_count = df[attr].count()
                total_count = len(df)
                coverage_pct = non_null_count / total_count if total_count > 0 else 0

                # Count stats
                attr_info[f'{name}_count'] = f"{non_null_count}/{total_count}"
                attr_info[f'{name}_pct'] = f"{coverage_pct:.1%}"
                attr_info[f'{name}_coverage'] = coverage_pct

                # Sample values for context
                if include_samples:
                    sample_values = df[attr].dropna().head(
                        sample_count).tolist()
                    if sample_values:
                        sample_str = str(sample_values)
                        if len(sample_str) > max_sample_length:
                            sample_str = sample_str[:max_sample_length] + "..."
                        attr_info[f'{name}_samples'] = sample_str
                    else:
                        attr_info[f'{name}_samples'] = "[]"
            else:
                attr_info[f'{name}_count'] = "0/0"
                attr_info[f'{name}_pct'] = "0%"
                attr_info[f'{name}_coverage'] = 0.0
                if include_samples:
                    attr_info[f'{name}_samples'] = "N/A"

        # Calculate total coverage across all datasets
        coverage_values = [attr_info[f'{name}_coverage']
                           for name in dataset_names]
        attr_info['avg_coverage'] = np.mean(coverage_values)
        attr_info['max_coverage'] = max(coverage_values)
        attr_info['datasets_with_attribute'] = sum(
            1 for c in coverage_values if c > 0)

        coverage_data.append(attr_info)

    coverage_df = pd.DataFrame(coverage_data)

    # Add metadata
    coverage_df.attrs['analysis_type'] = 'attribute_coverage'
    coverage_df.attrs['dataset_count'] = len(datasets)
    coverage_df.attrs['dataset_names'] = dataset_names
    coverage_df.attrs['total_attributes'] = len(all_attributes)

    logger.info(
        f"Analyzed {len(all_attributes)} attributes across {len(datasets)} datasets")

    return coverage_df


def compare_dataset_schemas(
    datasets: List[pd.DataFrame],
    dataset_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Compare schemas across multiple datasets.

    Parameters
    ----------
    datasets : List[pd.DataFrame]
        List of datasets to compare.
    dataset_names : Optional[List[str]]
        Names for each dataset.

    Returns
    -------
    Dict[str, Any]
        Schema comparison results.
    """
    if dataset_names is None:
        dataset_names = [f"dataset_{i}" for i in range(len(datasets))]

    # Collect schema information
    schemas = {}
    for df, name in zip(datasets, dataset_names):
        schemas[name] = {
            'columns': set(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'shape': df.shape,
            'null_counts': df.isnull().sum().to_dict()
        }

    # Find common and unique attributes
    all_columns = set()
    for schema in schemas.values():
        all_columns.update(schema['columns'])

    common_columns = all_columns.copy()
    for schema in schemas.values():
        common_columns &= schema['columns']

    unique_columns = {}
    for name, schema in schemas.items():
        unique_to_dataset = schema['columns'] - \
            (all_columns - schema['columns'])
        unique_columns[name] = unique_to_dataset

    return {
        'schemas': schemas,
        'all_columns': all_columns,
        'common_columns': common_columns,
        'unique_columns': unique_columns,
        'schema_overlap_matrix': _calculate_schema_overlap(schemas),
        'dtype_conflicts': _detect_dtype_conflicts(schemas, common_columns)
    }


def detect_attribute_conflicts(
    datasets: List[pd.DataFrame],
    dataset_names: Optional[List[str]] = None,
    sample_size: int = 1000
) -> Dict[str, Any]:
    """
    Detect potential conflicts in attribute values across datasets.

    Parameters
    ----------
    datasets : List[pd.DataFrame]
        List of datasets to analyze.
    dataset_names : Optional[List[str]]
        Names for each dataset.
    sample_size : int, default 1000
        Maximum number of records to sample per dataset for analysis.

    Returns
    -------
    Dict[str, Any]
        Conflict analysis results.
    """
    if dataset_names is None:
        dataset_names = [f"dataset_{i}" for i in range(len(datasets))]

    conflicts = {}

    # Find common attributes
    common_attrs = set(datasets[0].columns) if datasets else set()
    for df in datasets[1:]:
        common_attrs &= set(df.columns)

    for attr in common_attrs:
        attr_conflicts = {
            'value_distributions': {},
            'unique_value_counts': {},
            'potential_conflicts': [],
            'data_type_mismatches': []
        }

        # Analyze value distributions
        for df, name in zip(datasets, dataset_names):
            if attr in df.columns:
                # Sample data if dataset is large
                sample_df = df.sample(min(len(df), sample_size)) if len(
                    df) > sample_size else df

                # Value distribution
                if sample_df[attr].dtype in ['object', 'string']:
                    value_counts = sample_df[attr].value_counts().head(10)
                    attr_conflicts['value_distributions'][name] = value_counts.to_dict(
                    )
                else:
                    # For numeric data, show statistics
                    stats = sample_df[attr].describe()
                    attr_conflicts['value_distributions'][name] = stats.to_dict()

                # Unique value counts
                attr_conflicts['unique_value_counts'][name] = sample_df[attr].nunique(
                )

                # Data type
                dtype_str = str(sample_df[attr].dtype)
                if 'data_types' not in attr_conflicts:
                    attr_conflicts['data_types'] = {}
                attr_conflicts['data_types'][name] = dtype_str

        # Detect data type mismatches
        dtypes = list(attr_conflicts.get('data_types', {}).values())
        if len(set(dtypes)) > 1:
            attr_conflicts['data_type_mismatches'] = dtypes

        # Only include attributes with potential conflicts
        if (attr_conflicts['data_type_mismatches'] or
                len(attr_conflicts['value_distributions']) > 1):
            conflicts[attr] = attr_conflicts

    return conflicts


class AttributeCoverageAnalyzer:
    """
    Comprehensive analyzer for attribute coverage across datasets.

    This class provides detailed analysis capabilities and can suggest
    fusion rules based on the coverage patterns.

    Parameters
    ----------
    datasets : List[pd.DataFrame]
        List of datasets to analyze.
    dataset_names : Optional[List[str]]
        Names for each dataset.
    """

    def __init__(
        self,
        datasets: List[pd.DataFrame],
        dataset_names: Optional[List[str]] = None
    ):
        self.datasets = datasets
        self.dataset_names = dataset_names or [
            f"dataset_{i}" for i in range(len(datasets))]

        if len(self.datasets) != len(self.dataset_names):
            raise ValueError(
                "Number of datasets must match number of dataset names")

        # Perform analysis
        self.coverage_df = analyze_attribute_coverage(
            self.datasets, self.dataset_names)
        self.schema_comparison = compare_dataset_schemas(
            self.datasets, self.dataset_names)
        self.conflict_analysis = detect_attribute_conflicts(
            self.datasets, self.dataset_names)

        logger.info(
            f"Initialized AttributeCoverageAnalyzer for {len(self.datasets)} datasets")

    def print_summary(self, max_attributes: int = 20) -> None:
        """
        Print a comprehensive summary of the coverage analysis.

        Parameters
        ----------
        max_attributes : int, default 20
            Maximum number of attributes to show in detail.
        """
        print("Attribute Coverage Analysis Summary")
        print("=" * 50)

        # Dataset overview
        print(f"\nDataset Overview:")
        total_records = sum(len(df) for df in self.datasets)
        for i, (df, name) in enumerate(zip(self.datasets, self.dataset_names)):
            print(
                f"  {i+1}. {name}: {len(df):,} records, {len(df.columns)} attributes")
        print(f"  Total: {total_records:,} records")

        # Attribute statistics
        total_attrs = len(self.coverage_df)
        common_attrs = len([1 for _, row in self.coverage_df.iterrows()
                            if row['datasets_with_attribute'] == len(self.datasets)])

        print(f"\nAttribute Statistics:")
        print(f"  Total unique attributes: {total_attrs}")
        print(f"  Common across all datasets: {common_attrs}")
        print(f"  Dataset-specific attributes: {total_attrs - common_attrs}")

        # Coverage distribution
        coverage_bins = pd.cut(
            self.coverage_df['avg_coverage'],
            bins=[0, 0.25, 0.5, 0.75, 1.0],
            labels=['Low (0-25%)', 'Medium (25-50%)',
                    'High (50-75%)', 'Very High (75-100%)'],
            include_lowest=True
        )

        print(f"\nCoverage Distribution:")
        coverage_dist = coverage_bins.value_counts().sort_index()
        for category, count in coverage_dist.items():
            print(f"  {category}: {count} attributes ({count/total_attrs:.1%})")

        # Top attributes by coverage
        top_attrs = self.coverage_df.nlargest(
            min(10, total_attrs), 'avg_coverage')
        print(f"\nTop Attributes by Coverage:")
        for i, (_, row) in enumerate(top_attrs.iterrows(), 1):
            attr_name = row['attribute']
            avg_cov = row['avg_coverage']
            datasets_count = row['datasets_with_attribute']
            print(
                f"  {i}. {attr_name}: {avg_cov:.1%} avg ({datasets_count}/{len(self.datasets)} datasets)")

        # Show detailed coverage for top attributes
        if max_attributes > 0:
            print(
                f"\nDetailed Coverage (Top {min(max_attributes, total_attrs)}):")
            display_attrs = self.coverage_df.nlargest(
                max_attributes, 'avg_coverage')

            # Create display columns
            display_cols = ['attribute']
            for name in self.dataset_names:
                display_cols.extend([f'{name}_count', f'{name}_pct'])
            display_cols.extend(['avg_coverage', 'datasets_with_attribute'])

            # Filter available columns
            available_cols = [
                col for col in display_cols if col in display_attrs.columns]

            # Format percentages for display
            display_df = display_attrs[available_cols].copy()
            if 'avg_coverage' in display_df.columns:
                display_df['avg_coverage'] = display_df['avg_coverage'].apply(
                    lambda x: f"{x:.1%}")

            print(display_df.to_string(index=False))

        # Conflict analysis summary
        if self.conflict_analysis:
            print(f"\nPotential Conflicts Detected:")
            print(
                f"  Attributes with conflicts: {len(self.conflict_analysis)}")
            for attr, conflicts in list(self.conflict_analysis.items())[:5]:
                conflict_types = []
                if conflicts.get('data_type_mismatches'):
                    conflict_types.append("data type")
                if len(conflicts.get('value_distributions', {})) > 1:
                    conflict_types.append("value distribution")
                print(f"    {attr}: {', '.join(conflict_types)}")

            if len(self.conflict_analysis) > 5:
                print(f"    ... and {len(self.conflict_analysis) - 5} more")

    def export_analysis(self, output_path: Union[str, Path]) -> None:
        """
        Export the complete analysis to files.

        Parameters
        ----------
        output_path : Union[str, Path]
            Directory path to save analysis files.
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Export coverage DataFrame
        self.coverage_df.to_csv(
            output_path / 'attribute_coverage.csv', index=False)

        # Export schema comparison
        with open(output_path / 'schema_comparison.json', 'w') as f:
            # Convert sets to lists for JSON serialization
            exportable_schema = {}
            for key, value in self.schema_comparison.items():
                if isinstance(value, dict):
                    exportable_schema[key] = {}
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, set):
                            exportable_schema[key][subkey] = list(subvalue)
                        else:
                            exportable_schema[key][subkey] = subvalue
                elif isinstance(value, set):
                    exportable_schema[key] = list(value)
                else:
                    exportable_schema[key] = value

            import json
            json.dump(exportable_schema, f, indent=2, default=str)

        # Export conflict analysis
        with open(output_path / 'conflict_analysis.json', 'w') as f:
            import json
            json.dump(self.conflict_analysis, f, indent=2, default=str)

        # Export fusion rule suggestions
        suggestions = self.suggest_fusion_rules()
        with open(output_path / 'fusion_rule_suggestions.json', 'w') as f:
            import json
            json.dump(suggestions, f, indent=2)

        logger.info(f"Analysis exported to {output_path}")


def _calculate_schema_overlap(schemas: Dict[str, Dict]) -> pd.DataFrame:
    """Calculate overlap matrix between dataset schemas."""
    dataset_names = list(schemas.keys())
    overlap_matrix = pd.DataFrame(index=dataset_names, columns=dataset_names)

    for i, name1 in enumerate(dataset_names):
        for j, name2 in enumerate(dataset_names):
            if i == j:
                overlap_matrix.loc[name1, name2] = 1.0
            else:
                cols1 = schemas[name1]['columns']
                cols2 = schemas[name2]['columns']
                overlap = len(cols1 & cols2) / len(cols1 |
                                                   cols2) if cols1 | cols2 else 0.0
                overlap_matrix.loc[name1, name2] = overlap

    return overlap_matrix.astype(float)


def _detect_dtype_conflicts(schemas: Dict[str, Dict], common_columns: set) -> Dict[str, Dict]:
    """Detect data type conflicts in common columns."""
    conflicts = {}

    for col in common_columns:
        dtypes = {}
        for dataset_name, schema in schemas.items():
            if col in schema['dtypes']:
                dtype_str = str(schema['dtypes'][col])
                dtypes[dataset_name] = dtype_str

        # Check if there are different data types
        unique_dtypes = set(dtypes.values())
        if len(unique_dtypes) > 1:
            conflicts[col] = dtypes

    return conflicts
