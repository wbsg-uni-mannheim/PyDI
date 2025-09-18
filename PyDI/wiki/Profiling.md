# Profiling

This module generates dataset profiles and quick summaries for exploratory analysis. Profiling helps in understanding the schemata and record values, detect anomalies, and communicate general data quality at any point in the data integration pipeline. PyDI offers methods for quick data profiling in the console as well as printing detailed HTML reports via the ydata-profiling and sweetviz libraries.

When to Use (examples)
- Before schema/entity-matching: understand data, check for missing values, derive insights regarding required pre-processing.
- After data translation/normalization: validate improvements in coverage and consistency.

Module: `PyDI.profiling.profiler`
- `DataProfiler.profile()`: Generates detailed profiling information for a single dataset using ydata‑profiling. Outputs the resulting dataset profiling information as an HTML file.
- `DataProfiler.compare()`: Compares two datasets using the Sweetviz library. Outputs the result of the comparison as an HTML file.
- `DataProfiler.summary()`: Generates Log-style console output giving a quick overview of a dataset regarding number of rows, columns and missing values.
- `DataProfiler.analyze_coverage()`: Gives a more detailed overview over the columns contained in one or more datasets. Returns a dataframe with information about the amount of missing values and column overlap in datasets. Helpful to determine which attributes are useful for entity matching.

Example
```python
from PyDI.profiling import DataProfiler

profiler = DataProfiler()
html_path = profiler.profile(df, out_dir="output/movies/dataset-profiles")
```

Artifacts
- HTML reports written to file (`profile()`, `compare()`)
- Console output and optional JSON objects from `summary()` and `analyze coverage()`