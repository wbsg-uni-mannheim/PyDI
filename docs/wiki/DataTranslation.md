# Data Translation

The `MappingTranslator` translates a dataset into a target schema using a mapping, e.g. a set of schema correspondences. The mapping can either be the result of automated [schema matching](SchemaMatching.md) or be provided by the user. Data Translation is a pre-processing step that prepares datasets for entity matching and data fusion by normalizing attribute names.

Module: `PyDI.datatranslation`
- `MappingTranslator`: renames dataframe columns to target names using a set of schema correspondences.

## Usage Example
```python
from PyDI.datatranslation import MappingTranslator

translator = MappingTranslator()
df_aligned = translator.translate(df, corr)
```

## Artifacts
- Dataframe with aligned column names according to a schema correspondence file.
