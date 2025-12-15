# PyDI 0.2.0 Release Notes

We're excited to announce PyDI 0.2.0, a major update that completely rewrites the normalization module and adds a new schema translation component. This release brings a declarative, specification-based approach to data transformation.

## Highlights

- **Rewritten normalization module** with a new Spec/Transform API
- **New SchemaTranslator** for applying schema mappings with optional normalization
- **JSON Schema integration** for defining normalization rules
- **Improved LLM schema matching** with source metadata and target schema context
- **Four end-to-end use cases** demonstrating complete data integration workflows

## New Features

### Declarative Normalization API

The normalization module has been completely rewritten around a specification-based approach:

```python
from PyDI.normalization import (
    NormalizationSpec,
    transform_dataframe,
    profile_dataframe,
)

# Option 1: Manual specification
spec = NormalizationSpec()
spec.set_column("revenue", expand_scale_modifiers=True, output_type="float")
spec.set_column("country", country_format="alpha_2")
spec.set_column("phone", phone_format="e164", phone_default_region="US")
result = transform_dataframe(df, spec)

# Option 2: Auto-detect from data profiling
profile = profile_dataframe(df)
spec = NormalizationSpec.from_profile(profile)

# Option 3: Derive from JSON Schema
spec = load_normalization_spec("target_schema.json")
```

Supported normalizations include:
- **Units**: Convert between units of measurement (length, weight, temperature, etc.)
- **Scale modifiers**: Expand "5M", "2.5 billion", "100K" to numeric values
- **Countries**: Normalize to ISO alpha-2, alpha-3, numeric, or full names
- **Currencies**: Standardize currency codes and names
- **Phone numbers**: Format as E.164, international, or national
- **Standard numbers**: Validate and format ISBN, IBAN, VAT, EAN, etc.
- **Percentages**: Convert between "50%" and 0.5 representations

### Schema Translation

New `SchemaTranslator` class applies schema correspondences to rename columns and optionally normalize values in one step:

```python
from PyDI.schemamatching import SchemaTranslator

translator = SchemaTranslator()

# Rename columns only
df_translated = translator.translate(df_source, mapping)

# Rename and normalize in one step
df_normalized = translator.translate(
    df_source, mapping,
    normalize=spec,
    on_failure="keep"
)
```

### JSON Schema Integration

Define your target schema once and use it for both matching and normalization:

```python
# Load JSON Schema
with open("target_schema.json") as f:
    target_schema = json.load(f)

# Use for schema matching (LLM sees field descriptions)
matcher = LLMBasedSchemaMatcher(chat_model=llm, target_schema=target_schema)

# Derive normalization rules from the same schema
spec = load_normalization_spec("target_schema.json")
```

Supported JSON Schema constructs:
- `type`: string, integer, number, boolean mapped to output types
- `format`: date, email, phone-e164, country-alpha2, etc.
- `x-pydi-*` extensions for PyDI-specific settings

### Improved LLM Schema Matching

The LLM-based matcher now accepts richer context:

```python
# Source metadata (Schema.org Dataset format)
with open("source_metadata.json") as f:
    source_meta = json.load(f)

# Pass metadata per match() call
mapping = matcher.match(
    df_source, df_target,
    source_metadata=source_meta,    # Column descriptions
    target_schema=target_schema     # Override constructor schema
)
```

### Data Profiling

New profiling module detects column types and suggests normalizations:

```python
from PyDI.normalization import profile_dataframe

profile = profile_dataframe(df)
print(profile.summary())
```

Detected types include: numeric, string, date, boolean, unit_quantity, scaled_number, percentage, country, currency, phone, email, url, stdnum, coordinate.

### Pydantic Validation

Validate DataFrames against Pydantic models:

```python
from PyDI.normalization import validate_with_pydantic

class MovieRecord(BaseModel):
    title: str
    year: int
    budget: float

results = validate_with_pydantic(df, MovieRecord)
```

## Use Cases

Four complete end-to-end workflows demonstrate PyDI's capabilities:

| Use Case | Domain | Datasets | Features Demonstrated |
|----------|--------|----------|----------------------|
| [Companies](usecases/companies/) | Business | DBpedia, Wikidata, Forbes | Schema matching, entity matching, fusion |
| [Games](usecases/games/) | Video Games | RAWG, Metacritic, Sales | LLM matching, value normalization |
| [Movies](usecases/movies/) | Entertainment | Academy Awards, Golden Globes | Full pipeline with validation |
| [Music](usecases/music/) | Music | Discogs, LastFM, MusicBrainz | Multi-source fusion |

## Breaking Changes

### Removed Modules

- `PyDI.normalization.columns` - Use `profile_dataframe()` instead
- `PyDI.normalization.detectors` - Functionality moved to `profile.py`
- `PyDI.datatranslation` - Use `SchemaTranslator` instead
- `PyDI.profiling` - Use `PyDI.normalization.profile_dataframe()`

### API Changes

- `EmailValidator`: Constructor parameter changed from `strict: bool` to `check_deliverability: bool`
- `TokenizationNormalizer` removed from `text.py` - use `TokenBlocker` for tokenization

### Migration

```python
# Before (0.1.x)
from PyDI.normalization.columns import detect_column_types
from PyDI.datatranslation import MappingTranslator

# After (0.2.0)
from PyDI.normalization import profile_dataframe
from PyDI.schemamatching import SchemaTranslator
```

## Installation

```bash
pip install uma-pydi==0.2.0
```

## Links

- [GitHub Repository](https://github.com/wbsg-uni-mannheim/PyDI)
- [Documentation Wiki](https://github.com/wbsg-uni-mannheim/PyDI/blob/main/docs/wiki/Home.md)
- [Tutorials](https://github.com/wbsg-uni-mannheim/PyDI/blob/main/docs/tutorial/README.md)
- [Full Changelog](CHANGELOG.md)
