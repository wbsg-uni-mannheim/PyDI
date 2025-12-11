# Schema Matching

This module contains methods to automatically find 1-to-1 correspondences between columns across two datasets using label-, instance-, or duplicate-based schema matching methods. The discovered correspondences can be used afterwards to translate data from one schema into the other.

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

## WDC SMB Evaluation

Use ``run_wdc_smb_benchmark`` to evaluate ``LLMBasedSchemaMatcher`` using the
 [WDC Schema Matching Benchmark](https://webdatacommons.org/structureddata/smb/) (SOTAB-SM and T2D-SM). The helper takes
care of loading the released correspondences, aligning column indices with
their headers, and reporting precision, recall, and F1 via
``SchemaMappingEvaluator``.

Steps:
- Download the desired benchmark zip from the WDC SMB release page and extract
  it locally (e.g. ``SOTAB_SM_V500`` or ``T2D_SM_WH``).
- Instantiate your preferred LangChain ``BaseChatModel`` (for example
  ``ChatOpenAI``) and create ``LLMBasedSchemaMatcher`` with it.
- Call ``run_wdc_smb_benchmark`` with a ``WDCBenchmarkConfig`` pointing to the
  extracted directory and the split you want to evaluate.

Minimal example:
```python
from pathlib import Path
from langchain_openai import ChatOpenAI

from PyDI.schemamatching import (
    LLMBasedSchemaMatcher,
    WDCBenchmarkConfig,
    run_wdc_smb_benchmark,
)

chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)
matcher = LLMBasedSchemaMatcher(chat_model=chat, num_rows=8)

config = WDCBenchmarkConfig(
    dataset_root=Path("/data/WDC/SOTAB_SM_V500"),
    task="sotab",
    split="test",
)

result = run_wdc_smb_benchmark(matcher, config)
print(result["overall"])  # precision / recall / F1
```

There is also a CLI for OpenAI models:
```
python -m PyDI.schemamatching.wdc_smb \
    --dataset-root /data/WDC/SOTAB_SM_V500 \
    --task sotab \
    --split test \
    --openai-model gpt-4o-mini
```

Benchmark data should never be fed into model training. The utility only reads
the provided evaluation splits for on-demand benchmarking.

Artifacts
- Schema correspondences written to file

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
