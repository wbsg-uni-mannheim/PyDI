# PyDI - Python Data Integration Framework 

The PyDI framework provides methods for end-to-end data integration. The framework covers all steps of the integration process, including schema matching, data translation, entity matching, and data fusion. The framework offers both traditional string-based methods as well as modern LLM- and embedding-based techniques for these tasks. PyDI is designed as a set of independent, composable modules that operate on pandas DataFrames as the underlying data structure, ensuring interoperability with third-party packages that rely on pandas. 

This page provides an overview of the PyDI framework. Further details about the functionality of the framework are found  in the [Wiki](https://github.com/wbsg-uni-mannheim/PyDI/blob/main/docs/wiki/Home.md). In order to learn how to use the framework, please read the [Tutorials](https://github.com/wbsg-uni-mannheim/PyDI/blob/main/docs/tutorial/README.md) or have a look at the [Use Cases](https://github.com/wbsg-uni-mannheim/PyDI/blob/main/usecases/README.md) which illustrate how PyDI is used for end-to-end data integration.

## Installing PyDI

You can install PyDI via pip:

```
pip install uma-pydi
```

## Functionality

The PyDI framework covers all steps of the data integration process, including data loading, schema matching, data translation, entity matching, and data fusion. This section gives an overview of the functionality and the alternative methods that are provided for each of these steps.

**Schema Matching**: Schema matching identifies attributes in multiple schemata that have the same meaning. PyDI provides four schema matching methods which either rely on attribute labels or data values, or exploit an existing mapping of records in order to find attribute correspondences (duplicate-based schema matching). PyDI's schema matching module offers:

-   Label-based schema matching
-   Instance-based schema matching
-   Duplicate-based schema matching
-   LLM-based schema matching
-   Data translation with optional value normalization
-   Evaluation of schema matching results
-   Debug reports about the matching process

**Data Translation**: Translates data from a source schema into a target schema. The translation process may include value normalization and information extraction. PyDI provides the following data translaton methods:

-   Value normalization
    -   Data profiling with automatic type and pattern detection
    -   Unit of measurement conversion (length, weight, temperature, etc.)
    -   Scale modifier expansion (MEO, MEUR, million, billion)
    -   Country, currency, and language code normalization
    -   Number validation (phone, IBAN, VAT, ISBN)
    -   JSON Schema support for defining normalization specs
    -   Data quality validation (ranges, patterns, completeness, uniqueness)
-   Information extraction via
    -   Regex
    -   Python functions
    -   Large language models
-   Evaluation of information extraction results

**Entity Matching**: Entity matching methods identify records in different datasets that describe the same real-world entity. PyDI offers a range of entity matching methods, starting from simple attribute similarity-based rules over machine-learned rules, to Pre-trained Language Models (PLMs) and Large Language Models (LLMs). Entity matching methods rely on blocking in order to reduce the number of record comparisons. PyDI provides the following blocking and entity matching methods:

-	Blocking Methods
	-   Key-based blocking
	-   Sorted-neighbourhood blocking
	-   Token-based blocking
	-   Embedding-based blocking
- 	Entity Matching
	-   Rule-based entity matching (manual or machine learning-based)
	-   PLM-based entity matching
	-   LLM-based entity matching
    -   5 correspondence filtering and clustering methods
-   Evaluation of entity matching and blocking results
-   Debug reports about the matching process

**Data Fusion**: Data fusion combines data from multiple sources into a single, consolidated dataset. Different sources may provide conflicting data values. PyDI allows you to resolve such data conflicts (decide which value to include in the final dataset) by applying different conflict resolution functions. PyDI's fusion module offers the following:

-   15 conflict resolution functions
    -   Strings: longest_string, shortest_string, most_complete
    -   Numbers: average, median, maximum, minimum, sum_values
    -   Dates: most_recent, earliest
    -   Lists: union, intersection, intersection_k_sources
    -   Metadata-based: voting, weighted_voting, favour_sources, prefer_higher_trust
-   Evaluation of data fusion results against ground truth
-   Provenance tracking for fused values
-   Debug reports about the fusion process

**IO**: PyDI provides methods for reading standard data formats into pandas DataFrames with provenance tracking:

-   Supported formats: CSV, JSON, XML, Excel, Parquet, Feather, HTML tables, fixed-width files
-   Automatic provenance metadata (source path, timestamps, checksums)
-   Optional unique identifier generation for downstream matching and fusion

## Tutorials

| Tutorial | Description |
|----------|-------------|
| [Data Integration Tutorial](docs/tutorial/entity_matching_and_fusion/data_integration_tutorial.ipynb) | End-to-end pipeline: loading, blocking, matching, fusion |
| [Value Normalization Tutorial](docs/tutorial/normalization/value_normalization/value_normalization_tutorial.ipynb) | Profiling, specs, unit conversion, data cleaning |
| [Schema Matching Tutorial](docs/tutorial/normalization/schema_matching/schema_matching_tutorial.ipynb) | LLM-based schema matching with JSON Schema |

## Contact

For issues, feature requests, or contributions, please open a GitHub **Issue** or submit a **Pull Request**. For further information about PyDI, please email the maintainers of the framework.

## Acknowledgements

PyDI is developed by the [Web-based Systems Group](https://www.uni-mannheim.de/dws/research/focus-groups/web-based-systems-prof-bizer/) at the [University of Mannheim](http://www.uni-mannheim.de/). The framework is used for projects and exercises in the course [Web Data Integration](https://www.uni-mannheim.de/dws/teaching/course-details/courses-for-master-candidates/ie-670-web-data-integration/) at the University of Mannheim.
