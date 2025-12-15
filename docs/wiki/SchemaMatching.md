# Schema Matching

This module contains methods to automatically find 1-to-1 correspondences between columns across two datasets using label-, instance-, or duplicate-based schema matching methods. The discovered correspondences can be used afterwards to translate data from one schema into the other.

## Target Schema

The target schema defines the unified structure you want to map source datasets to. You can represent it as:

1. **A DataFrame** with the target column names (simplest approach)
2. **A JSON Schema** file with column definitions and descriptions (recommended for LLM-based matching). PyDI supports [JSON Schema Draft-07](https://json-schema.org/draft-07/).

### Using a JSON Schema

When using `LLMBasedSchemaMatcher`, passing a JSON Schema via `target_schema` improves matching accuracy. The LLM sees not just the column names but also their descriptions, which helps with ambiguous mappings.

```python
import json
from PyDI.schemamatching import LLMBasedSchemaMatcher

# Load target schema
with open("target_schema.json") as f:
    target_schema = json.load(f)

# Create empty target DataFrame for column names
target_columns = list(target_schema["properties"].keys())
df_target = pd.DataFrame(columns=target_columns)

# Pass schema to matcher - either in constructor or match() call
matcher = LLMBasedSchemaMatcher(
    chat_model=llm,
    target_schema=target_schema,  # Set default for all match() calls
)
corr = matcher.match(df_source, df_target)

# Or override per match() call
corr = matcher.match(df_source, df_target, target_schema=other_schema)
```

Example `target_schema.json`:
```json
{
  "type": "object",
  "properties": {
    "title": {
      "type": "string",
      "description": "Movie title"
    },
    "date": {
      "type": "string",
      "format": "date",
      "description": "Release year in ISO 8601 format"
    },
    "budget": {
      "type": "number",
      "minimum": 0,
      "description": "Production budget in millions USD"
    }
  }
}
```

The `description` fields help the LLM understand the semantic meaning of each column, leading to better matches when source column names differ significantly from target names.

### Deriving Normalization from Target Schema

The same JSON Schema can be used to derive a `NormalizationSpec` for value transformation after matching. See [From JSON Schema](Normalization.md#from-json-schema) in the Normalization Wiki for supported constructs.

```python
from PyDI.normalization import load_normalization_spec

spec = load_normalization_spec("target_schema.json")
df_aligned = translator.translate(df_source, corr, normalize=spec)
```

## Source Dataset Metadata

For source datasets, you can provide [Schema.org Dataset](https://schema.org/Dataset) metadata to improve LLM matching accuracy. The `variableMeasured` field describes each column, helping the LLM understand ambiguous or abbreviated column names.

```python
import json

# Load Schema.org metadata
with open("dbpedia_metadata.json") as f:
    source_meta = json.load(f)

# Pass metadata in match() call
corr = matcher.match(df_source, df_target, source_metadata=source_meta)
```

Example `dbpedia_metadata.json`:
```json
{
  "@type": "Dataset",
  "name": "DBpedia Companies",
  "description": "Company data extracted from Wikipedia",
  "variableMeasured": [
    {
      "@type": "PropertyValue",
      "name": "companyName",
      "description": "Official name of the company"
    },
    {
      "@type": "PropertyValue",
      "name": "revenue",
      "description": "Annual revenue",
      "unitText": "monetary amount"
    }
  ]
}
```

The LLM sees the column descriptions alongside sample data, improving matches for columns like `foundingDate` → `date` or `companyName` → `name`.

## Available Matchers
The module `PyDI.schemamatching` provides the following matchers, evaluators, and translators:
- `LabelBasedSchemaMatcher`: compares the labels of the columns using similarity metrics to find schema correspondences. Fast and accurate when column labels are meaningful.
- `InstanceBasedSchemaMatcher`: compares the distributions of values per column via TF/TF‑IDF/binary vectors and cosine/Jaccard/containment similarity. Better suited than LabelBasedMatcher if column labels are ambigous.
- `DuplicateBasedSchemaMatcher`: leverage known record correspondences to infer column alignments from co‑occurring values in columns of the corresponding records. Great when a labeled set of matching records between datasets exists.
- `LLMBasedSchemaMatcher`: the matcher prompts hosted large language models, such as GPT or Gemini, to find corespondences. Using the matcher requires a valid API key from a LLM provider.
- `SchemaMappingEvaluator`: offers methods for evaluating a generated schema mapping given a labeled set of schema correspondences.
- `SchemaTranslator`: renames DataFrame columns to target names using schema correspondences, preparing datasets for entity matching and data fusion.

## Schema Matching Example 
```python
from langchain_openai import ChatOpenAI
from PyDI.schemamatching import LLMBasedSchemaMatcher

chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)
matcher = LLMBasedSchemaMatcher(
    chat_model=chat,
    num_rows=10,
    debug=True
    )
corr = matcher.match(source_df, target_df)
```

Example evaluation
```python
from PyDI.schemamatching import SchemaMappingEvaluator

metrics = SchemaMappingEvaluator.evaluate(corr, test_set)
```

## Schema Translation

After matching, use `SchemaTranslator` to apply the discovered correspondences and rename columns in your source DataFrame to match the target schema:

```python
from PyDI.schemamatching import SchemaTranslator

translator = SchemaTranslator()
df_aligned = translator.translate(source_df, corr)
```

The translator:
- Filters mappings to the relevant dataset (based on `dataset_name` in DataFrame attrs)
- Picks the best mapping by score when duplicates exist
- Adds provenance tracking at both DataFrame and column level

### Translation with Value Normalization

`SchemaTranslator` can optionally normalize values after renaming columns. This is useful when the target schema expects specific formats (e.g., ISO country codes, standardized phone numbers).

**Auto-detection:**
```python
# Automatically detect and apply normalizations based on data profiling
df_aligned = translator.translate(source_df, corr, normalize=True)

# Auto-detect with custom failure handling
df_aligned = translator.translate(source_df, corr, normalize=True, on_failure="null")
```

**Explicit normalization spec:**
```python
from PyDI.normalization import NormalizationSpec

spec = NormalizationSpec()
spec.set_column("country", country_format="alpha_2")  # Normalize to ISO 2-letter codes
spec.set_column("revenue", expand_scale_modifiers=True, output_type="float")  # "5 MEO" → 5000000.0
spec.set_column("email", normalize_email=True)

df_aligned = translator.translate(source_df, corr, normalize=spec)
```

### Handling Normalization Failures

The `on_failure` parameter controls what happens when a value cannot be normalized. It works with both auto-detection and explicit specs:

| Value | Behavior |
|-------|----------|
| `"keep"` | Keep the original value (default) |
| `"null"` | Replace with `None`/`NaN` |
| `"raise"` | Raise a `ValueError` |

```python
# Auto-detect normalizations, set failures to null
df_aligned = translator.translate(source_df, corr, normalize=True, on_failure="null")

# With explicit spec, individual columns can override the default
spec = NormalizationSpec()
spec.set_column("country", country_format="alpha_2", on_failure="null")
spec.set_column("phone", phone_format="e164", on_failure="keep")  # Keep invalid phones as-is
df_aligned = translator.translate(source_df, corr, normalize=spec, on_failure="raise")
```

## Tutorials

- [Schema Matching Tutorial](../tutorial/normalization/schema_matching/schema_matching_tutorial.ipynb) - End-to-end workflow: LLM-based matching with JSON Schema integration
