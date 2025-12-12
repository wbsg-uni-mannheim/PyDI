# Welcome to the PyDI Wiki

PyDI (Python Data Integration) is an end to end data integration framework covering the complete integration process, including schema matching, data translation, entity matching, and data fusion. The framework offers both traditional string-based methods as well as modern embedding- and LLM-based techniques for these tasks.

## General features:
- PyDI consists of composable modules that can be used independently or as a pipeline.
- All modules rely on pandas DataFrames as underlying data structure, ensuring interoperability with third-party packages that also rely on pandas.
- Each module supports result evaluation and can generate detailed logs of its operations. These logs can serve as starting point for debugging and improving data integration pipelines.

## The PyDI Data Integration Pipeline
PyDI supports the following steps of the data integration pipeline:
1. Load data and add provenance metadata
2. Data profiling
3. Schema matching and data translation
4. Information extraction
5. Attribute value normalization
6. Entity matching including blocking
7. Data fusion including conflict resolution

## The PyDI Modules
PyDI consists of the following modules:
- [Schema Matching](SchemaMatching.md) provides matchers to automatically find correspondences between the columns of two datasets, plus translation to apply the mappings. The matchers implement label-based, instance-based, and LLM-based schema matching methods.
- [Normalization](Normalization.md) provides methods for standardizing attribute vales including normalizing of units of measurement.
- [Information Extraction](InformationExtraction.md) provides functionality for splitting a string into attribute values using regex-, code-, or LLM-based information extraction methods.
- [Entity Matching](EntityMatching.md) this module allows you to identify records in multiple datasets that describe the same real-world entity. For this, the module implements various blocking techniques as well as different matchers (rule-based, ML-based, LLM-based).
- [Data Fusion](DataFusion.md) merges sets of records describing the same real-world entity into a single consolidated dataset. Provides methods for resolving data conflicts using attribute-level conflict resolution heuristics.
- [IO](IO.md) offers readers for loading data in different formats, adding identifiers to records, and adding provenance metadata to datasets.
- [Utils](Utils.md) implements functionality that is used by multiple modules, such as data profiling, logging, and similarity computation.

## Tutorials

For hands-on examples, see the [Tutorials](../tutorial/README.md):

- [Data Integration Tutorial](../tutorial/entity_matching_and_fusion/data_integration_tutorial.ipynb) - End-to-end pipeline with movie datasets
- [Value Normalization Tutorial](../tutorial/normalization/value_normalization/value_normalization_tutorial.ipynb) - Profiling, specs, and transformations
- [Schema Matching Tutorial](../tutorial/normalization/schema_matching/schema_matching_tutorial.ipynb) - LLM-based matching with JSON Schema
