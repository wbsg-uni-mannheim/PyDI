# Python Data Integration Framework (PyDI)

The PyDI framework provides methods for end-to-end data integration. The framework covers all steps of the integration process, including schema matching, data translation, entity matching, and data fusion. The framework offers both traditional string-based methods as well as modern LLM- and embedding-based techniques for these tasks. PyDI is designed as a set of independent, composable modules that operate on pandas DataFrames as the underlying data structure, ensuring interoperability with third-party packages that rely on pandas. 
## Contents

-   [Functionality](#functionality)
-   [Contact](#contact)
-   [License](#license)
-   [Acknowledgements](#acknowledgements)

This page provides an overview of the functionality of the PyDI framework. As alternatives to familiarizing yourself with the framework, you can also read the [PyDI Tutorial](/PyDI/tutorial/PyDI_Tutorial.ipynb) or have a look at the code examples in our [Wiki](/PyDI/wiki/)!

## Installing PyDI

You can install PyDI via pip:

```
pip install uma-pydi
```

## Functionality

The PyDI framework covers all steps of the data integration process, including data loading, schema matching, data translation, entity matching, and data fusion. This section gives an overview of the functionality and the alternative algorithms that are provided for each of these steps.

**[Data Loading](#)**: PyDI provides methods for reading standard data formats such as JSON, XML, and CSV into pandas DataFrames. All read methods can optionally add provenance metadata to the dataframes which is represented using the pandas property `DataFrame.attrs`.

**[Schema Matching](#)**: Schema matching identifies attributes in multiple schemata that have the same meaning. PyDI provides three schema matching methods which either rely on attribute labels or data values, or exploit an existing mapping of records (duplicate-based schema matching) in order to find attribute correspondences. PyDI's schema matching module supports:

-   Label-based schema matching
-   Instance-based schema matching
-   Duplicate-based schema matching
-   Evaluation of schema matching results
-   Reports about the matching process

**[Data Translation](#)**: Translates data from a source schema into a target schema. The translation process may include value normalization and information extraction. PyDI provides the following methods for data translation, value normalization, and information extraction:

-   Data Translation
	-   Mapping-based translation
-   Value normalization
    -   Data type detection
    -   Value & header normalization
    -   Unit of measurement conversion
    -   Data validation
-   Information extraction via
    -   Regex
    -   Python functions
    -   Large language models

**[Entity Matching](#)**: Entity matching methods identify records in different datasets that describe the same real-world entity. PyDI offers a range of entity matching methods, starting from simple attribute similarity-based rules over machine-learned rules, to Pre-trained Language Models (PLMs) and Large Language Models (LLMs). Entity matching methods rely on blocking in order to reduce the number of record comparisons. PyDI provides the following pre-implemented blocking and entity matching methods:

-	Blocking Methods
	-   Key-based blocking
	-   Sorted-neighbourhood blocking
	-   Token-based blocking
	-   Embedding-based blocking
- Entity Matching
	-   Rule-based entity matching (manual or machine learning-based)
	-   PLM-based entity matching
	-   LLM-based entity matching
-   Evaluation of entity matching and blocking results
-   Reports about the matching process

**[Data Fusion](#)**: Data fusion combines data from multiple sources into a single, consolidated dataset. Different sources may provide conflicting data values. PyDI allows you to resolve such data conflicts (decide which value to include in the final dataset) by applying different conflict resolution functions. PyDI's fusion module offers the following:

-   13 value-based conflict resolution functions for strings, numbers, and sets
-   4 metadata-based conflict resolution functions.
-   Evaluation of data fusion results against ground truth
-   Reports about the fusion process

## Contact

If you have questions or need help, please first consult the [PyDI Tutorial](/PyDI/tutorial/PyDI_Tutorial.ipynb), the [Wiki](/PyDI/wiki/), and the project documentation. For issues, feature requests, or contributions, please open a GitHub **Issue** or submit a **Pull Request**.

## License

The PyDI framework can be used under the [Apache 2.0 License](http://www.apache.org/licenses/LICENSE-2.0).

## Acknowledgements

PyDI is developed at the [Data and Web Science Group](http://dws.informatik.uni-mannheim.de/) at the [University of Mannheim](http://www.uni-mannheim.de/).
