# IO

The IO module loads tabular and semi‑structured data into pandas DataFrames, records provenance for each load operation in `df.attrs`, and optionally creates or preserves stable row identifiers for downstream matching and fusion. It standardizes dataset naming (`df.attrs["dataset_name"]`) and keeps source information and reader parameters available for reproducibility.

Supported Formats (module `PyDI.io.loaders`)
- `load_csv(...)`, `load_table(...)` — delimited text files
- `load_json(...)` — JSON with optional `record_path` (flattens nested objects)
- `load_excel(...)` — a single sheet or a dictionary of sheets
- `load_parquet(...)`, `load_feather(...)` — columnar formats
- `load_pickle(...)` — pre‑serialized DataFrames
- `load_html(...)` — list of HTML tables per page
- `load_xml(...)` — XML with record‑level flattening (explode or aggregate repeated elements)

Saving
- To preserve provenance (`df.attrs`) alongside the data, we currently recommend saving DataFrames as pickle files. Other formats (e.g., CSV, Parquet, Feather) typically discard custom attributes; use them only when provenance is not required or is exported separately.
