# PyDI Tutorials

This directory contains tutorials demonstrating PyDI's data integration capabilities.

## Available Tutorials

### Entity Matching and Data Fusion

| Tutorial | Description | Topics Covered |
|----------|-------------|----------------|
| [data_integration_tutorial.ipynb](entity_matching_and_fusion/data_integration_tutorial.ipynb) | End-to-end data integration with movie datasets | Data loading, blocking, entity matching (rule-based, ML-based), post-clustering, data fusion, evaluation |

### Schema Matching and Value Normalization

| Tutorial | Description | Topics Covered |
|----------|-------------|----------------|
| [schema_matching_tutorial.ipynb](normalization/schema_matching/schema_matching_tutorial.ipynb) | Schema matching with JSON Schema-driven normalization | LLM-based schema matching, JSON Schema integration, SchemaTranslator |
| [value_normalization_tutorial.ipynb](normalization/value_normalization/value_normalization_tutorial.ipynb) | Full normalization workflow with messy company data | Profiling, NormalizationSpec, transformations, unit conversion, country/currency codes, phone/email normalization |

## Directory Structure

```
tutorial/
├── entity_matching_and_fusion/
│   ├── data_integration_tutorial.ipynb
│   └── movies/                            # Movie datasets and evaluation sets
├── normalization/
│   ├── value_normalization/
│   │   ├── value_normalization_tutorial.ipynb
│   │   └── challenging_dataset.csv
│   └── schema_matching/
│       ├── schema_matching_tutorial.ipynb
│       └── data/                          # Sample data files
└── benchmark/                             # WDC Schema Matching Benchmark data
```

## Getting Started

1. Install PyDI and dependencies:
   ```bash
   pip install -e .
   ```

2. For LLM-based features (schema matching), set up your API key:
   ```bash
   export OPENAI_API_KEY=your-key-here
   ```

3. Open a tutorial notebook:
   ```bash
   jupyter notebook docs/tutorial/entity_matching_and_fusion/data_integration_tutorial.ipynb
   ```

## Related Documentation

The PyDI [Wiki](../wiki/) covers the complete functionality of each module:

- [IO](../wiki/IO.md) - Data loading with provenance
- [Schema Matching](../wiki/SchemaMatching.md) - Column correspondence discovery
- [Normalization](../wiki/Normalization.md) - Value transformation and cleaning
- [Entity Matching](../wiki/EntityMatching.md) - Record deduplication and linking
- [Data Fusion](../wiki/DataFusion.md) - Conflict resolution and merging
