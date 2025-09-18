# Profiling

Generate dataset profiles and quick summaries for exploratory analysis. Profiling helps verify schema assumptions, detect anomalies, and communicate data quality via shareable HTML.

When to Use
- Before matching: check header quality, missingness, and candidate keys.
- After translation/normalization: validate improvements in coverage and consistency.
- During debugging: compare two datasets to understand distribution differences.

Module: `PyDI.profiling.profiler`
- `DataProfiler.profile(df, out_dir) -> str`: ydata‑profiling HTML
- `DataProfiler.compare(df_a, df_b, out_dir) -> str`: Sweetviz comparison HTML
- `DataProfiler.summary(df) -> dict`: quick stats suitable for logs
- `DataProfiler.analyze_coverage(df, required_columns, out_dir=None) -> dict|str`: coverage for required fields

How It Works
- Cleans DataFrames for compatibility (casts list cells to strings, removes problematic dtypes for Sweetviz).
- Generates a single, self‑contained HTML per operation that can be opened locally or shared.

Performance and Practical Tips
- Large DataFrames: consider profiling a sample for speed, then profile full data at milestones.
- Privacy: ydata‑profiling may show example values; use sampling/redaction when needed.
- Headless servers: HTML is fully self‑contained; no extra assets required.

Example
```python
from PyDI.profiling import DataProfiler

profiler = DataProfiler()
html_path = profiler.profile(df, out_dir="output/movies/dataset-profiles")
print("Profile written:", html_path)
```

Artifacts
- HTML reports written to `out_dir` (ydata‑profiling, Sweetviz)
- Optional JSON summaries from `summary()` and coverage analysis
