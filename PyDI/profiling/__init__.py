"""
Profiling tools for PyDI.

This module exposes a :class:`DataProfiler` that wraps popular data
profiling libraries (ydata-profiling and Sweetviz) to generate HTML
reports for individual datasets and comparisons between two datasets. A
quick summary method provides basic statistics without creating heavy
reports.
"""

from __future__ import annotations

import os
from typing import Dict

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
        report = sv.compare(df_a, df_b, name_a, name_b)
        out_path = os.path.join(out_dir, f"{name_a}_vs_{name_b}_compare.html")
        report.show_html(out_path)
        return out_path

    def summary(self, df: pd.DataFrame) -> Dict[str, object]:
        """Return a dictionary of basic dataset statistics.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame to summarise.

        Returns
        -------
        dict
            A dictionary with row count, column count, total null values,
            per‑column null counts and dtypes.
        """
        return {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "nulls_total": int(df.isnull().sum().sum()),
            "nulls_per_column": df.isnull().sum().to_dict(),
            "dtypes": df.dtypes.apply(lambda x: x.name).to_dict(),
        }