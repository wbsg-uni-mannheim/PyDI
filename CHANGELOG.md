# Changelog

All notable changes to PyDI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-12-15

### Added

- **LLM-Based Schema Matching Enhancements**:
  - `source_metadata` parameter in `match()` for Schema.org Dataset metadata
  - `target_schema` parameter in `match()` to override constructor-level schema
  - Structured prompt template with clear sections (Dataset Description, Provenance, Sample Data)
  - Support for `variableMeasured` field descriptions from Schema.org metadata

- **JSON Schema Integration**:
  - `load_normalization_spec()` derives normalization rules from JSON Schema
  - `load_validation_spec()` extracts validation constraints from JSON Schema
  - Support for `type`, `format`, and custom `x-pydi-*` extensions
  - Target schema descriptions passed to LLM for better matching context

- **SchemaTranslator with Normalization**:
  - `normalize` parameter accepts `NormalizationSpec`, `True` (auto-detect), or `False`
  - `on_failure` parameter controls handling of normalization failures
  - Combined translation and normalization in a single step

- **New Spec/Transform API** for declarative normalization:
  - `NormalizationSpec` - Define DataFrame-level normalization rules
  - `ColumnSpec` - Per-column normalization settings with options for output type, failure handling, units, percentages, country/currency formats, phone formatting, and more
  - `transform_dataframe()` - Apply transformations according to a spec
  - `normalize_dataframe()` - Main entry point with optional auto-detection (`auto=True`)
  - `TransformResult` / `DataFrameTransformResult` - Detailed transformation metadata

- **Enhanced profiling**:
  - `DataTypeExtended` enum for richer type classification
  - Percentage detection (both `50%` and `0.5` formats)
  - Coordinate detection using `CoordinateParser`
  - Boolean string detection (`"yes"`, `"no"`, `"true"`, `"false"`)
  - Improved profile summary output with samples and suggestions

- **New integration modules**:
  - `integrations/babel_numbers.py` - Locale-aware numeric parsing with Babel
  - `integrations/pydantic_validation.py` - DataFrame validation against Pydantic models

- **New validators**:
  - `PydanticSchemaValidator` - Validate DataFrames using Pydantic models
  - `validate_with_pydantic()` - Convenience function for Pydantic validation

- **Documentation**:
  - Schema Matching Wiki with Target Schema and Source Dataset Metadata sections
  - Normalization Wiki with JSON Schema integration guide and type/format mapping tables
  - Updated tutorials with proper content ordering and cross-references
  - End-to-end use cases for companies, games, movies, and music domains

### Changed

- **EmailValidator** now uses `email-validator` library instead of regex patterns
  - Constructor parameter changed from `strict: bool` to `check_deliverability: bool`
  - `validate_emails()` function signature updated accordingly

- **NumericParser** now delegates Babel logic to `integrations/babel_numbers.py`

- **WebTableNormalizer** now delegates to `TextNormalizer` and `HeaderNormalizer` internally

- **AdvancedValueNormalizer** now uses `scale` module and `normalize_quantity()` instead of custom `QuantityModifier` handling

- **DateNormalizer** uses `pd.to_datetime()` without deprecated `infer_datetime_format` parameter

### Removed

- **`columns.py`** module deleted - functionality consolidated into `profile.py`:
  - `AdvancedTypeDetector`
  - `ColumnTypeInference`
  - `ValueDetectionType`
  - `detect_column_types()`
  - `analyze_column_quality()`
  - `detect_dataframe_types()`
  - `infer_column_types()`

- **`detectors.py`** module deleted:
  - `DataType`
  - `NullDetector`
  - `OutlierDetector`
  - `DuplicateDetector`

- **`TokenizationNormalizer`** class removed from `text.py`
  - For tokenization with stemming/stopwords, use:
    - `PyDI.utils.SimilarityRegistry.TOKENIZATION_STRATEGIES`
    - `PyDI.entitymatching.blocking.TokenBlocker`

- **`tokenize_text()`** function removed from `text.py`

### Migration Guide

#### Updating imports

```python
# Before
from PyDI.normalization.columns import DataTypeExtended, detect_column_types

# After
from PyDI.normalization import DataTypeExtended
from PyDI.normalization import profile_dataframe  # Use profiling instead
```

#### Using the new Spec/Transform API

```python
from PyDI.normalization import (
    profile_dataframe,
    NormalizationSpec,
    transform_dataframe,
    normalize_dataframe,
)

# Option 1: Auto-detection
normalized_df = normalize_dataframe(df, auto=True)

# Option 2: From profile
profile = profile_dataframe(df)
spec = NormalizationSpec.from_profile(profile)
result = transform_dataframe(df, spec)
normalized_df = result.dataframe

# Option 3: Manual specification
spec = NormalizationSpec()
spec.set_column("revenue", expand_scale_modifiers=True, output_type="float")
spec.set_column("country", country_format="alpha_2")
spec.set_column("phone", phone_format="e164", phone_default_region="DE")
result = transform_dataframe(df, spec)
```

#### Updating EmailValidator usage

```python
# Before
validator = EmailValidator(strict=True)

# After
validator = EmailValidator(check_deliverability=False)  # or True for DNS checks
```

#### Replacing TokenizationNormalizer

```python
# Before
from PyDI.normalization.text import TokenizationNormalizer
tokenizer = TokenizationNormalizer(use_stemming=True)
tokens = tokenizer.tokenize(text)

# After - use entity matching tokenizers
from PyDI.entitymatching.blocking import TokenBlocker
# Or access tokenization strategies directly
from PyDI.utils import SimilarityRegistry
```
