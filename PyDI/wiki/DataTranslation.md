# Data Translation

Translate/align source datasets to a target schema using a schema mapping. Translation prepares datasets for matching and fusion by ensuring consistent attribute names and optionally derived values.

Module: `PyDI.datatranslation`
- `MappingTranslator`: renames dataframe columns to target names if a 1:1 mapping exists given a set of schema correspondences.

Usage
```python
from PyDI.datatranslation import MappingTranslator

translator = MappingTranslator()
df_aligned = translator.translate(df, corr)
```

Artifacts
- Dataframe with aligned column names according to a schema correspondence file.