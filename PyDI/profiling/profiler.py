"""
Data profiling implementation for PyDI.

This module provides the DataProfiler class that wraps popular data
profiling libraries (ydata-profiling and Sweetviz) to generate HTML
reports for individual datasets and comparisons between two datasets.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import pandas as pd


class DataProfiler:
    """Generate profiling reports for pandas DataFrames.

    The profiler uses optional third‑party libraries. If a required library
    is not installed, a clear :class:`ImportError` is raised.

    Methods in this class return the path to the generated HTML file for
    downstream consumption.
    """

    def __init__(self) -> None:
        pass

    def profile(self, df: pd.DataFrame, out_dir: str) -> str:
        """Generate an HTML profiling report for a single DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame to profile.
        out_dir : str
            Directory in which to store the generated report. The directory
            will be created if it does not exist.

        Returns
        -------
        str
            Path to the HTML report.

        Raises
        ------
        ImportError
            If the `ydata_profiling` library is not installed.
        """
        os.makedirs(out_dir, exist_ok=True)
        try:
            from ydata_profiling import ProfileReport
        except ImportError as exc:
            raise ImportError(
                "ydata-profiling is required for DataProfiler.profile; install with pip"
            ) from exc
        dataset_name = df.attrs.get("dataset_name", "dataset")
        report = ProfileReport(
            df,
            title=f"Profile for {dataset_name}",
            infer_dtypes=True,
            explorative=True,
        )
        out_path = os.path.join(out_dir, f"{dataset_name}_profile.html")
        report.to_file(out_path)
        return out_path

    def compare(self, df_a: pd.DataFrame, df_b: pd.DataFrame, out_dir: str) -> str:
        """Generate a comparison report for two DataFrames.

        Parameters
        ----------
        df_a, df_b : pandas.DataFrame
            The DataFrames to compare.
        out_dir : str
            Directory in which to store the generated report.

        Returns
        -------
        str
            Path to the HTML comparison report.

        Raises
        ------
        ImportError
            If the `sweetviz` library is not installed.
        """
        os.makedirs(out_dir, exist_ok=True)
        try:
            import sweetviz as sv
        except ImportError as exc:
            raise ImportError(
                "sweetviz is required for DataProfiler.compare; install with pip"
            ) from exc
        name_a = df_a.attrs.get("dataset_name", "A")
        name_b = df_b.attrs.get("dataset_name", "B")

        # Clean dataframes for sweetviz - convert unhashable types to strings
        df_a_clean = self._clean_for_sweetviz(df_a, name_a)
        df_b_clean = self._clean_for_sweetviz(df_b, name_b)

        report = sv.compare((df_a_clean, name_a), (df_b_clean, name_b))
        out_path = os.path.join(out_dir, f"{name_a}_vs_{name_b}_compare.html")
        report.show_html(out_path)
        return out_path

    def summary(
        self, df: pd.DataFrame, print_summary: bool = True
    ) -> Dict[str, object]:
        """Return a dictionary of basic dataset statistics and optionally print them.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame to summarise.
        print_summary : bool, default True
            Whether to print the summary statistics to console.

        Returns
        -------
        dict
            A dictionary with row count, column count, total null values,
            per‑column null counts and dtypes.
        """
        summary_data = {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "nulls_total": int(df.isnull().sum().sum()),
            "nulls_per_column": df.isnull().sum().to_dict(),
            "dtypes": df.dtypes.apply(lambda x: x.name).to_dict(),
        }

        if print_summary:
            dataset_name = df.attrs.get("dataset_name", "Dataset")
            print(f"{dataset_name}:")
            print(f"  Rows: {summary_data['rows']:,}")
            print(f"  Columns: {summary_data['columns']}")
            print(f"  Total nulls: {summary_data['nulls_total']:,}")
            print(
                f"  Null percentage: {(summary_data['nulls_total'] / (summary_data['rows'] * summary_data['columns']) * 100):.1f}%"
            )

            # Show null counts per column
            nulls_per_col = summary_data["nulls_per_column"]
            if any(nulls_per_col.values()):
                print("  Null counts per column:")
                for col, null_count in nulls_per_col.items():
                    if null_count > 0:
                        print(
                            f"    {col}: {null_count:,} ({null_count/summary_data['rows']*100:.1f}%)"
                        )

            print()

        return summary_data

    def analyze_coverage(
        self,
        datasets: List[pd.DataFrame],
        dataset_names: Optional[List[str]] = None,
        include_samples: bool = True,
        max_sample_length: int = 50,
        sample_count: int = 2,
    ) -> pd.DataFrame:
        """Analyze attribute coverage across multiple datasets.

        This method provides comprehensive analysis of attribute coverage,
        schema overlap, and data quality across multiple datasets - useful
        for data integration planning.

        Parameters
        ----------
        datasets : List[pd.DataFrame]
            List of datasets to analyze.
        dataset_names : Optional[List[str]]
            Names for each dataset. If None, uses dataset names from attrs
            or generates default names.
        include_samples : bool, default True
            Whether to include sample values in the analysis.
        max_sample_length : int, default 50
            Maximum length for sample value strings.
        sample_count : int, default 2
            Number of sample values to include per attribute.

        Returns
        -------
        pd.DataFrame
            DataFrame with detailed coverage analysis including:
            - Per-dataset attribute counts and percentages
            - Sample values for context
            - Cross-dataset coverage statistics

        Examples
        --------
        >>> profiler = DataProfiler()
        >>> coverage = profiler.analyze_coverage([df1, df2, df3])
        >>> print(coverage[['attribute', 'dataset1_pct', 'dataset2_pct']])
        """
        # Import here to avoid circular dependency
        from ..fusion.analysis import analyze_attribute_coverage

        # Use dataset names from attrs if not provided
        if dataset_names is None:
            dataset_names = []
            for i, df in enumerate(datasets):
                name = df.attrs.get("dataset_name", f"dataset_{i}")
                dataset_names.append(name)

        return analyze_attribute_coverage(
            datasets=datasets,
            dataset_names=dataset_names,
            include_samples=include_samples,
            max_sample_length=max_sample_length,
            sample_count=sample_count,
        )

    def _clean_for_sweetviz(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """Clean DataFrame for sweetviz by excluding columns with unhashable types.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame to clean.
        dataset_name : str
            Name of the dataset for logging purposes.

        Returns
        -------
        pandas.DataFrame
            Cleaned DataFrame with unhashable columns excluded.
        """
        df_clean = df.copy()
        columns_to_drop = []

        for col in df_clean.columns:
            # Check if column contains unhashable types (lists, dicts, sets)
            has_unhashable = False

            # Check a larger sample and also check for specific types
            sample_values = df_clean[col].dropna()

            # First check if any values are obviously unhashable types
            for value in sample_values.head(500):  # Check more samples
                if isinstance(value, (list, dict, set)):
                    has_unhashable = True
                    break
                # Try hashing
                try:
                    hash(value)
                except (TypeError, ValueError):
                    has_unhashable = True
                    break

            # Additional check: try to get unique values (this will fail on unhashable types)
            if not has_unhashable:
                try:
                    _ = df_clean[col].nunique()
                except (TypeError, ValueError):
                    has_unhashable = True

            if has_unhashable:
                columns_to_drop.append(col)

        # Drop columns with unhashable types
        if columns_to_drop:
            df_clean = df_clean.drop(columns=columns_to_drop)
            print(f"WARNING: Dataset '{dataset_name}' contains unhashable data types in columns: {columns_to_drop}")
            print(f"         These columns have been excluded from the sweetviz comparison.")

        return df_clean
