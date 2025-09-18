# IO

Load tabular and semi‑structured data into pandas DataFrames with consistent IDs, dataset names, and provenance. PyDI’s IO aims for deterministic, script‑friendly ingestion that works well for notebooks, pipelines, and LLM tools.

Design Goals
- DataFrames first: returns `pd.DataFrame` with metadata in `df.attrs`.
- Deterministic IDs: stable `_id` per row for traceability across steps.
- Provenance by default: record the source and operation that produced the data.
- Practical ergonomics: sensible defaults, auto‑detection helpers, and safe conversions for export.

Key Functions (module `PyDI.io.loaders`)
- `load_csv(path, *, dataset_name=None, id_column='_id', delimiter=',', ...) -> pd.DataFrame`
- `load_json(path, *, dataset_name=None, record_path=None, ...) -> pd.DataFrame`
- `load_excel(path, *, dataset_name=None, sheet_name=0, ...) -> pd.DataFrame`
- `load_parquet(path, *, dataset_name=None, ...) -> pd.DataFrame`
- `load_feather(path, *, dataset_name=None, ...) -> pd.DataFrame`
- `load_pickle(path, *, dataset_name=None, ...) -> pd.DataFrame`
- `load_html(path, *, dataset_name=None, ...) -> pd.DataFrame`
- `load_xml(path, *, dataset_name=None, record_tag=None, ...) -> pd.DataFrame`
- `load_table(path, *, dataset_name=None, format=None, ...) -> pd.DataFrame` (auto‑detects by suffix)

Unique ID Scheme
- If `dataset_name` is provided, `_id` is generated as `f"{dataset_name}_{i:06d}"` with zero‑padded index.
- If an `id_column` already exists, PyDI preserves it and may still add `_id` if requested.
- IDs allow consistent joins, matching, and evaluation across modules.

Provenance Model
- `df.attrs.update({"dataset_name": ..., "source": {"path": ..., "format": ...}, "provenance": [{"op": "load_csv", "params": {...}, "ts": ...}]})`
- Downstream modules append their own provenance entries.

Column Names and Types
- Light cleaning (e.g., whitespace trimming and optional normalization utilities) ensures safe column labels.
- List‑typed cells are converted to strings for CSV/HTML exports to avoid lossy serialization.

Format‑Specific Notes
- CSV/TSV: respect delimiter, encoding, and nullable types when possible.
- JSON: flattening and `record_path` help for nested structures.
- Excel: choose sheet by name/index; beware mixed‑type columns.
- XML/HTML: configurable record tags and flattening of element trees.

Large Files and Memory
- Prefer Parquet/Feather for large datasets.
- Read in chunks externally if needed; PyDI focuses on clean load semantics rather than chunked iteration.

Example
```python
from PyDI.io import load_csv

df = load_csv("input/movies.csv", dataset_name="movies")
print(df.attrs["dataset_name"])  # "movies"
print(df.columns[:5])
```

Best Practices
- Always pass `dataset_name` for stable `_id` generation.
- Inspect `df.attrs["source"]` to verify ingestion details during debugging.
- Use Parquet for intermediate results to preserve types.

Artifacts
- None by default; downstream components write artifacts under `output/`.
