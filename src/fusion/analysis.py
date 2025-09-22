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

from .base import _is_valid_value

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


def analyze_conflicts_preview(
    datasets: List[pd.DataFrame],
    correspondences: pd.DataFrame,
    sample_size: int = 5,
    conflict_attrs: Optional[List[str]] = None,
    include_samples: bool = True,
    id_columns: Union[str, List[str]] = '_id'
) -> Dict[str, Any]:
    """
    Preview potential conflicts in matched records from correspondences.
    
    This function examines specific matched record pairs to identify concrete
    conflicts that will need to be resolved during fusion. Unlike statistical
    conflict analysis, this provides specific examples of conflicting values.
    
    Parameters
    ----------
    datasets : List[pd.DataFrame]
        List of datasets containing the records.
    correspondences : pd.DataFrame
        DataFrame with matched record pairs. Must have columns: id1, id2, score.
    sample_size : int, default 5
        Number of correspondences to analyze for conflicts.
    conflict_attrs : Optional[List[str]]
        Specific attributes to check for conflicts. If None, checks all common attributes.
    include_samples : bool, default True
        Whether to include sample records in the output.
    id_columns : Union[str, List[str]], default '_id'
        ID column name(s). Can be a single string (same for all datasets) or 
        a list of strings (one per dataset).
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'conflict_examples': List of dictionaries with conflict details
        - 'conflict_summary': Statistics about conflicts found
        - 'attribute_conflicts': Per-attribute conflict counts
        
    Examples
    --------
    >>> conflict_preview = analyze_conflicts_preview(
    ...     datasets=[df1, df2], 
    ...     correspondences=corr_df,
    ...     sample_size=10,
    ...     id_columns=['id1', 'id2']  # Different ID columns per dataset
    ... )
    >>> print(f"Found {len(conflict_preview['conflict_examples'])} conflicts")
    """
    if not datasets:
        raise ValueError("At least one dataset must be provided")
    
    if correspondences.empty:
        logger.warning("No correspondences provided for conflict analysis")
        return {
            'conflict_examples': [],
            'conflict_summary': {'total_matches': 0, 'matches_with_conflicts': 0},
            'attribute_conflicts': {}
        }
    
    # Validate and prepare ID columns
    if isinstance(id_columns, str):
        # Single string - use for all datasets
        dataset_id_columns = [id_columns] * len(datasets)
    elif isinstance(id_columns, list):
        # List - must match dataset count
        if len(id_columns) != len(datasets):
            raise ValueError(f"Length of id_columns ({len(id_columns)}) must match number of datasets ({len(datasets)})")
        dataset_id_columns = id_columns
    else:
        raise ValueError("id_columns must be a string or list of strings")
    
    # Build lookup tables
    id_to_record = {}
    id_to_dataset = {}
    
    for i, df in enumerate(datasets):
        dataset_name = df.attrs.get('dataset_name', f'dataset_{i}')
        id_column = dataset_id_columns[i]
        
        if id_column not in df.columns:
            raise ValueError(f"ID column '{id_column}' not found in dataset '{dataset_name}' (columns: {list(df.columns)})")
        
        for _, record in df.iterrows():
            record_id = record.get(id_column)
            if record_id is not None:
                id_to_record[record_id] = record
                id_to_dataset[record_id] = dataset_name
    
    # Sample correspondences for analysis
    sample_corr = correspondences.head(sample_size)
    
    conflict_examples = []
    attribute_conflict_counts = {}
    matches_with_conflicts = 0
    
    # Diagnostic counters for debugging
    records_not_found = 0
    no_common_attributes = 0
    identical_values = 0
    total_comparisons = 0
    
    logger.info(f"Analyzing conflicts in {len(sample_corr)} correspondence pairs")
    
    # Debug logging if enabled
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Built lookup table with {len(id_to_record)} records")
        logger.debug(f"Sample dataset IDs: {list(id_to_record.keys())[:10]}")
        logger.debug(f"Sample correspondence IDs: id1={sample_corr['id1'].head(5).tolist()}, id2={sample_corr['id2'].head(5).tolist()}")
    
    for i, (_, corr) in enumerate(sample_corr.iterrows(), 1):
        id1, id2 = corr['id1'], corr['id2']
        score = corr.get('score', 'N/A')
        
        record1 = id_to_record.get(id1)
        record2 = id_to_record.get(id2)
        
        if record1 is None or record2 is None:
            records_not_found += 1
            if logger.isEnabledFor(logging.DEBUG):
                missing_ids = [id1 if record1 is None else None, id2 if record2 is None else None]
                logger.debug(f"Missing record(s) for correspondence {i}: {[id for id in missing_ids if id is not None]}")
            continue
            
        dataset1 = id_to_dataset[id1]
        dataset2 = id_to_dataset[id2]
        
        # Determine attributes to check
        common_attrs = set(record1.index) & set(record2.index)
        if conflict_attrs is not None:
            check_attrs = [attr for attr in conflict_attrs if attr in common_attrs]
        else:
            check_attrs = list(common_attrs)
        
        # Track cases with no common attributes
        if not check_attrs:
            no_common_attributes += 1
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"No common attributes for correspondence {i}: {dataset1} cols={list(record1.index)[:5]}, {dataset2} cols={list(record2.index)[:5]}")
            continue
        
        # Find conflicts
        conflicts = []
        for attr in check_attrs:
            if attr.startswith('_'):  # Skip internal attributes
                continue
                
            val1, val2 = record1.get(attr), record2.get(attr)
            
            # Count all comparisons
            if _is_valid_value(val1) and _is_valid_value(val2):
                total_comparisons += 1
                
                # Check if values are different
                if str(val1).strip() != str(val2).strip():
                    conflicts.append({
                        'attribute': attr,
                        'value1': val1,
                        'value2': val2,
                        'dataset1': dataset1,
                        'dataset2': dataset2
                    })
                    
                    # Count attribute-level conflicts
                    attribute_conflict_counts[attr] = attribute_conflict_counts.get(attr, 0) + 1
                else:
                    # Values are identical
                    identical_values += 1
        
        if conflicts:
            matches_with_conflicts += 1
        
        # Build example record
        example = {
            'match_id': i,
            'id1': id1,
            'id2': id2,
            'dataset1': dataset1,
            'dataset2': dataset2,
            'score': score,
            'conflicts': conflicts,
            'has_conflicts': len(conflicts) > 0
        }
        
        if include_samples:
            # Add sample record data (first few attributes for brevity)
            sample_attrs = ['title', 'name', 'director', 'author', 'date', 'year']
            example['record1_sample'] = {
                attr: record1.get(attr, 'N/A') 
                for attr in sample_attrs if attr in record1.index
            }
            example['record2_sample'] = {
                attr: record2.get(attr, 'N/A') 
                for attr in sample_attrs if attr in record2.index
            }
        
        conflict_examples.append(example)
    
    # Compile results
    results = {
        'conflict_examples': conflict_examples,
        'conflict_summary': {
            'total_matches': len(sample_corr),
            'matches_with_conflicts': matches_with_conflicts,
            'conflict_rate': matches_with_conflicts / len(sample_corr) if len(sample_corr) > 0 else 0.0,
            'total_attribute_conflicts': sum(attribute_conflict_counts.values())
        },
        'attribute_conflicts': attribute_conflict_counts,
        'diagnostics': {
            'records_not_found': records_not_found,
            'no_common_attributes': no_common_attributes,
            'identical_values': identical_values,
            'total_comparisons': total_comparisons,
            'lookup_table_size': len(id_to_record),
            'processed_pairs': len(sample_corr) - records_not_found - no_common_attributes
        }
    }
    
    logger.info(
        f"Found conflicts in {matches_with_conflicts}/{len(sample_corr)} matches "
        f"({results['conflict_summary']['conflict_rate']:.1%})"
    )
    
    # Detailed diagnostic logging if debug is enabled
    if logger.isEnabledFor(logging.DEBUG):
        diagnostics = results['diagnostics']
        logger.debug(f"Conflict analysis diagnostics:")
        logger.debug(f"  Records not found: {diagnostics['records_not_found']}")
        logger.debug(f"  No common attributes: {diagnostics['no_common_attributes']}")
        logger.debug(f"  Identical values: {diagnostics['identical_values']}")
        logger.debug(f"  Total comparisons: {diagnostics['total_comparisons']}")
        logger.debug(f"  Processed pairs: {diagnostics['processed_pairs']}")
        
        if diagnostics['records_not_found'] > 0:
            logger.debug(f"TIP: {diagnostics['records_not_found']} records not found - check if correspondence IDs match dataset ID columns")
        if diagnostics['no_common_attributes'] > 0:
            logger.debug(f"TIP: {diagnostics['no_common_attributes']} pairs have no common attributes - check column names between datasets")
        if diagnostics['identical_values'] > diagnostics['total_comparisons'] * 0.8:
            logger.debug(f"TIP: Most values ({diagnostics['identical_values']}/{diagnostics['total_comparisons']}) are identical - datasets may already be standardized")
    
    return results


def print_conflict_preview(
    datasets: List[pd.DataFrame],
    correspondences: pd.DataFrame,
    sample_size: int = 5,
    conflict_attrs: Optional[List[str]] = None,
    id_columns: Union[str, List[str]] = '_id'
) -> None:
    """
    Print a formatted preview of conflicts in matched records.
    
    This is a convenience function that calls analyze_conflicts_preview()
    and prints the results in a human-readable format.
    
    Parameters
    ----------
    datasets : List[pd.DataFrame]
        List of datasets containing the records.
    correspondences : pd.DataFrame
        DataFrame with matched record pairs.
    sample_size : int, default 5
        Number of correspondences to analyze.
    conflict_attrs : Optional[List[str]]
        Specific attributes to check for conflicts.
    id_columns : Union[str, List[str]], default '_id'
        ID column name(s). Can be a single string (same for all datasets) or 
        a list of strings (one per dataset).
    """
    results = analyze_conflicts_preview(
        datasets, correspondences, sample_size, conflict_attrs, include_samples=True, id_columns=id_columns
    )
    
    print(f"Conflict Analysis Preview (First {sample_size} matches):")
    print("=" * 80)
    
    summary = results['conflict_summary']
    print(f"Summary: {summary['matches_with_conflicts']}/{summary['total_matches']} matches have conflicts ({summary['conflict_rate']:.1%})")
    
    if results['attribute_conflicts']:
        print(f"Most conflicted attributes: {dict(sorted(results['attribute_conflicts'].items(), key=lambda x: x[1], reverse=True))}")
    
    print("\nDetailed Examples:")
    print("-" * 40)
    
    for example in results['conflict_examples']:
        dataset1, dataset2 = example['dataset1'], example['dataset2']
        score = example['score']
        
        print(f"\nMatch {example['match_id']}: {dataset1} ↔ {dataset2} (score: {score})")
        
        # Show sample record data
        if 'record1_sample' in example:
            record1_display = " | ".join([f"{v}" for v in example['record1_sample'].values()])
            record2_display = " | ".join([f"{v}" for v in example['record2_sample'].values()])
            print(f"  {dataset1}: {record1_display}")
            print(f"  {dataset2}: {record2_display}")
        
        # Show conflicts
        if example['has_conflicts']:
            conflict_strs = []
            for conflict in example['conflicts']:
                conflict_strs.append(f"{conflict['attribute']}: '{conflict['value1']}' vs '{conflict['value2']}'")
            print(f"  Conflicts: {'; '.join(conflict_strs)}")
        else:
            print(f"  No obvious conflicts")


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
