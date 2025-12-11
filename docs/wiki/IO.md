# IO

The IO module loads tabular and semi-structured data into pandas DataFrames with provenance tracking. Each load operation records metadata in `df.attrs["provenance"]` and standardizes dataset naming via `df.attrs["dataset_name"]`. The module optionally creates stable row identifiers for downstream matching and fusion.

## Available Loaders

The module `PyDI.io` provides the following loaders:

- `load_csv(...)` - CSV and delimited text files
- `load_table(...)` - Generic delimited text with configurable separators
- `load_fwf(...)` - Fixed-width formatted files
- `load_json(...)` - JSON with optional `record_path` (flattens nested objects)
- `load_xml(...)` - XML with record-level flattening (explode or aggregate repeated elements)
- `load_excel(...)` - Excel files (single sheet or dictionary of sheets)
- `load_parquet(...)` - Apache Parquet columnar format
- `load_feather(...)` - Apache Arrow Feather format
- `load_pickle(...)` - Pre-serialized DataFrames
- `load_html(...)` - HTML tables (returns list of DataFrames per page)

## Basic Usage

```python
from PyDI.io import load_csv, load_json, load_xml

# Load CSV with dataset name
df = load_csv("data/companies.csv", name="companies")
print(df.attrs["dataset_name"])  # 'companies'

# Load JSON with nested records
df = load_json("data/products.json", record_path="items")

# Load XML with repeated elements
df = load_xml("data/catalog.xml", record_tag="product")
```

## Provenance Tracking

All loaders automatically attach provenance metadata to `df.attrs["provenance"]`:

```python
df = load_csv("data/movies.csv", name="movies")

# Access provenance
prov = df.attrs["provenance"]
print(prov["source_path"])       # '/path/to/data/movies.csv'
print(prov["reader"])            # 'read_csv'
print(prov["loaded_at_utc_iso"]) # '2024-01-15T10:30:00+00:00'
print(prov["file_size_bytes"])   # 12345
print(prov["sha256_prefix"])     # 'a1b2c3...' (checksum of first 64MB)
```

Custom provenance can be added:

```python
df = load_csv(
    "data/movies.csv",
    name="movies",
    provenance={"source_url": "https://example.com/movies.csv", "version": "2.0"}
)
```

## Unique Identifiers

Add stable row identifiers for entity matching and fusion:

```python
df = load_csv("data/movies.csv", name="movies", add_index=True)

# First column is now 'movies_id' with values like 'movies_0', 'movies_1', ...
print(df.columns[0])  # 'movies_id'
print(df["movies_id"].iloc[0])  # 'movies_0'

# Customize the id column
df = load_csv(
    "data/movies.csv",
    name="movies",
    add_index=True,
    index_column_name="movie_uid",
    id_prefix="mov"
)
# Column 'movie_uid' with values 'mov_0', 'mov_1', ...
```

The id column name is recorded in provenance:

```python
print(df.attrs["provenance"]["id_column_name"])  # 'movies_id'
```

## Loading Specific Formats

### CSV and Delimited Files

```python
from PyDI.io import load_csv, load_table

# Standard CSV
df = load_csv("data/movies.csv", name="movies")

# Tab-separated without headers
df = load_csv("data/actors.tsv", name="actors", sep="\t", header=None)

# Generic delimited file
df = load_table("data/pipe_delimited.txt", name="data", sep="|")
```

### JSON

```python
from PyDI.io import load_json

# Flat JSON array
df = load_json("data/records.json", name="records")

# Nested JSON with record path
df = load_json("data/api_response.json", name="users", record_path="data.users")

# Handle nested arrays (aggregate to strings)
df = load_json("data/products.json", name="products", nested_handling="aggregate")
```

### XML

```python
from PyDI.io import load_xml

# Basic XML loading
df = load_xml("data/catalog.xml", name="products", record_tag="product")

# With attribute extraction
df = load_xml(
    "data/catalog.xml",
    name="products",
    record_tag="product",
    attr_prefix="@"  # Attributes become columns like '@id'
)

# Handle repeated child elements
df = load_xml(
    "data/movies.xml",
    name="movies",
    record_tag="movie",
    repeated_handling="aggregate",  # Join repeated elements with separator
    separator=", "
)
```

### Excel

```python
from PyDI.io import load_excel

# Single sheet (first by default)
df = load_excel("data/report.xlsx", name="report")

# Specific sheet
df = load_excel("data/report.xlsx", name="sales", sheet_name="Sales Data")

# All sheets (returns dict)
sheets = load_excel("data/report.xlsx", name="report", sheet_name=None)
# sheets["Sheet1"], sheets["Sheet2"], etc.
```

### Columnar Formats

```python
from PyDI.io import load_parquet, load_feather

# Parquet (efficient for large datasets)
df = load_parquet("data/large_dataset.parquet", name="dataset")

# Feather (fast read/write)
df = load_feather("data/processed.feather", name="processed")
```

## Preserving Provenance

DataFrame attributes (`df.attrs`) are lost when saving to most formats. To preserve provenance:

```python
import pickle

# Save with provenance
with open("data/movies_with_prov.pkl", "wb") as f:
    pickle.dump(df, f)

# Load preserves attrs
from PyDI.io import load_pickle
df = load_pickle("data/movies_with_prov.pkl", name="movies")
```

For other formats, export provenance separately:

```python
import json

# Save data and provenance
df.to_csv("data/movies.csv", index=False)
with open("data/movies_provenance.json", "w") as f:
    json.dump(df.attrs, f)
```



## Tutorials

- [Data Integration Tutorial](../tutorial/entity_matching_and_fusion/data_integration_tutorial.ipynb) - Data loading with provenance tracking as part of a full integration pipeline
- [Value Normalization Tutorial](../tutorial/normalization/value_normalization/value_normalization_tutorial.ipynb) - Loading CSV data for normalization workflows
