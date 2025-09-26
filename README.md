# PyDI - Python Data Integration Framework 

The PyDI framework provides methods for end-to-end data integration. The framework covers all steps of the integration process, including schema matching, data translation, entity matching, and data fusion. The framework offers both traditional string-based methods as well as modern LLM- and embedding-based techniques for these tasks. PyDI is designed as a set of independent, composable modules that operate on pandas DataFrames as the underlying data structure, ensuring interoperability with third-party packages that rely on pandas. 

This page provides an overview of the functionality of the PyDI framework. As alternatives to familiarizing yourself with the framework, you can also read the [PyDI Tutorial](https://github.com/wbsg-uni-mannheim/PyDI/blob/main/docs/tutorial/PyDI_Tutorial.ipynb) or have a look at the code examples in our [Wiki](https://github.com/wbsg-uni-mannheim/PyDI/blob/main/docs/wiki/Home.md)!

## Installing PyDI

You can install PyDI via pip:

```
pip install uma-pydi
```

## Functionality

The PyDI framework covers all steps of the data integration process, including data loading, schema matching, data translation, entity matching, and data fusion. This section gives an overview of the functionality and the alternative algorithms that are provided for each of these steps.

**Schema Matching**: Schema matching identifies attributes in multiple schemata that have the same meaning. PyDI provides three schema matching methods which either rely on attribute labels or data values, or exploit an existing mapping of records (duplicate-based schema matching) in order to find attribute correspondences. PyDI's schema matching module offers:

-   Label-based schema matching
-   Instance-based schema matching
-   Duplicate-based schema matching
-   LLM-based schema matching
-   Evaluation of schema matching results
-   Debug reports about the matching process

**Data Translation**: Translates data from a source schema into a target schema. The translation process may include value normalization and information extraction. PyDI provides the following methods for value normalization and information extraction:

-   Value normalization
    -   Data type detection
    -   Value & header normalization
    -   Unit of measurement conversion
    -   Data validation
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
    -   6 post-clustering methods
-   Evaluation of entity matching and blocking results
-   Debug reports about the matching process

**Data Fusion**: Data fusion combines data from multiple sources into a single, consolidated dataset. Different sources may provide conflicting data values. PyDI allows you to resolve such data conflicts (decide which value to include in the final dataset) by applying different conflict resolution functions. PyDI's fusion module offers the following:

-   13 value-based conflict resolution functions for strings, numbers, and sets
-   4 metadata-based conflict resolution functions.
-   Evaluation of data fusion results against ground truth
-   Debug reports about the fusion process

**IO**: PyDI provides methods for reading standard data formats such as JSON, XML, and CSV into pandas DataFrames. All read methods can optionally add unique identifiers and provenance metadata to the DataFrames.

## Contact

If you have questions or need help, please first consult the [PyDI Tutorial](https://github.com/wbsg-uni-mannheim/PyDI/blob/main/docs/tutorial/PyDI_Tutorial.ipynb), the [Wiki](https://github.com/wbsg-uni-mannheim/PyDI/blob/main/docs/wiki/Home.md), and the project documentation. For issues, feature requests, or contributions, please open a GitHub **Issue** or submit a **Pull Request**. For further information, please email the maintainers of the framework.

## Acknowledgements

PyDI is developed by the [Web-based Systems Group](https://www.uni-mannheim.de/dws/research/focus-groups/web-based-systems-prof-bizer/) at the [University of Mannheim](http://www.uni-mannheim.de/).
