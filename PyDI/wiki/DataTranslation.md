# Data Translation

Translate/align source datasets to a target schema using a schema mapping. Translation prepares datasets for matching and fusion by ensuring consistent attribute names and optionally derived values.

Modules
- `PyDI.datatranslation.base.BaseTranslator`
- `PyDI.datatranslation.mapping_translator.MappingTranslator`

Strategies
- rename: rename columns to target names where a 1:1 mapping exists.
- copy/merge (future extension): create new columns by copying/combining sources.
- derive (via code or IE): compute attributes from existing text/numeric fields.

Handling Missing or Conflicting Mappings
- Unmapped columns are left as‑is by default; downstream steps may ignore them.
- Conflicts (two source columns mapped to same target) can be resolved by explicit rules or left for fusion.

Provenance
- Translation appends provenance to column attrs when available (e.g., old→new name mapping) to support auditability.

Validation
- Combine with `normalization.validators` or pydantic models to ensure aligned data matches expected types and ranges.

Usage
```python
from PyDI.datatranslation import MappingTranslator

translator = MappingTranslator(strategy="rename")
df_aligned = translator.translate(df, corr)
```

Inputs
- `df`: pandas DataFrame with `df.attrs["dataset_name"]`
- `corr`: schema mapping DataFrame (see Schema Matching)

Artifacts
- Aligned DataFrame; write to CSV/Parquet using pandas IO.
