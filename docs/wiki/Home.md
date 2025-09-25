Welcome to the PyDI Wiki
========================

PyDI (Python Data Integration) is an end‑to‑end data integration framework for loading, profiling, matching, and fusing heterogeneous datasets. It combines traditional methods (rule‑based similarity, blocking, voting) with modern approaches (machine learning, deep‑learning embeddings, and LLM‑based extraction/matching). Each component writes out human‑readable artifacts and logs, so results are understandable and improvements can be targeted based on evidence.

PyDI Design Principles
- Composable modules that can be used independently or as a pipeline.
- Data is stored in pandas DataFrames throughout the entire framework, allowing for maximum interoperability with other packages and frameworks that are not part of PyDI.
- Reproducible runs with optional debugging logs allow for detailed performance interpretation and analysis.

End‑to‑End Data Integration Pipeline 
1. Load data and add provenance 
2. Profile datasets 
3. Information extraction 
4. Value normalization
5. Schema Mapping
6. Data Translation   
7. Entity Matching 
8. Data Fusion 


Contents (PyDI Modules)
- [IO](#io) - load data, set IDs, record provenance
- [Profiling](#profiling) - dataset profiles and comparisons (HTML)
- [Information Extraction](#information-extraction) - regex/code/LLM extraction + evaluation
- [Normalization](#normalization) - clean headers/text, standardize values and units
- [Schema Matching](#schema-matching) - label/instance/duplicate‑based matching + evaluation
- [Data Translation](#data-translation) - apply schema mappings to align data
- [Entity Matching](#entity-matching) - candidate generation (standard/sorted‑neighbourhood/token/embedding), rule‑based, ML‑based, and LLM‑based matchers + evaluation
- [Data Fusion](#data-fusion) - conflict resolution rules and evaluation/reporting
- [Utils](#utils) - comparators and similarity registry
- [Tutorial](../tutorial/PyDI_Winter_Tutorial.ipynb) - end‑to‑end example (movie usecase)


The sections below provide a concise, high‑level overview of PyDI’s functionality. For more detailed and interactive exploration, see the linked example notebooks and the tutorial notebook in `PyDI/examples` and `PyDI/tutorial`.


# IO

The IO module loads tabular and semi‑structured data into pandas DataFrames, records provenance for each load operation in `df.attrs`, and optionally creates or preserves stable row identifiers for downstream matching and fusion. It standardizes dataset naming (`df.attrs["dataset_name"]`) and keeps source information and reader parameters available for reproducibility.

Supported Formats (module `PyDI.io.loaders`)
- `load_csv(...)`, `load_table(...)` - delimited text files
- `load_json(...)` - JSON with optional `record_path` (flattens nested objects)
- `load_excel(...)` - a single sheet or a dictionary of sheets
- `load_parquet(...)`, `load_feather(...)` - columnar formats
- `load_pickle(...)` - pre‑serialized DataFrames
- `load_html(...)` - list of HTML tables per page
- `load_xml(...)` - XML with record‑level flattening (explode or aggregate repeated elements)

To preserve provenance (found in `df.attrs`) alongside the data, we currently recommend saving DataFrames as pickle files. Other formats (e.g., CSV, Parquet, Feather) discard `.attrs`, use them only when provenance is not required or is exported separately.


# Profiling

This module generates dataset profiles and quick summaries for exploratory data analysis. Profiling helps in understanding the schemata and record values, detect anomalies, and communicate general data quality at any point in the data integration pipeline. PyDI offers methods for quick data profiling in the console as well as printing detailed HTML reports via the ydata-profiling and sweetviz libraries.

When to Use (examples)
- Exploratory analysis after data loading.
- Before schema/entity-matching: understand data, check for missing values, derive insights regarding required pre-processing.
- After data translation/normalization: validate improvements in coverage and consistency.

Module: `PyDI.profiling.profiler`
- `DataProfiler.profile()`: Generates detailed profiling information for a single dataset using ydata‑profiling. Outputs the resulting dataset profiling information as an HTML file.
- `DataProfiler.compare()`: Compares two datasets using the Sweetviz library. Outputs the result of the comparison as an HTML file.
- `DataProfiler.summary()`: Generates Log-style console output giving a quick overview of a dataset regarding number of rows, columns and missing values.
- `DataProfiler.analyze_coverage()`: Gives a more detailed overview over the columns contained in one or more datasets. Returns a dataframe with information about the amount of missing values and column overlap in datasets. Helpful to determine which attributes are useful for entity matching.

Example
```python
from PyDI.profiling import DataProfiler

profiler = DataProfiler()
profiler.profile(df, out_dir="output/movies/dataset-profiles")
```

Artifacts
- HTML reports written to file (`profile()`, `compare()`)
- Console output and optional JSON objects from `summary()` and `analyze coverage()`


# Information Extraction

The Information Extraction module derives structured, typed attributes from text‑heavy or semi‑structured columns. It supports pattern‑based extraction with regular expressions, custom Python functions for dataset‑specific logic, and model‑based extraction using large language models. Extracted fields are appended as new columns and can be evaluated against a gold standard when available.

Supported Approaches (module `PyDI.informationextraction`)
- Regex‑based extraction: `RegexExtractor` applies rule sets of patterns with optional post‑processing to normalize results (e.g., amounts, dates, units).
- Code‑based extraction: `CodeExtractor` runs user‑defined Python functions on text columns or full rows for flexible, domain‑specific logic.
- LLM‑based extraction: `LLMExtractor` performs schema‑guided extraction via LangChain, supporting many providers and models through a unified chat‑model interface (e.g., OpenAI, Anthropic). Prompts, responses, and errors are persisted for transparency.

Additional Notes
- Normalize outputs to canonical types (e.g., ISO dates, floats) to aid downstream matching and fusion.
- Multiple extractors can be combined using `ExtractorPipeline` to incrementally enrich a dataset.

Evaluation of Extraction
- Module: `PyDI.informationextraction.evaluation.InformationExtractionEvaluator` compares predicted columns with a gold standard aligned by record IDs.
- Metrics: reports attribute‑level results and aggregate micro/macro precision, recall, F1, and non‑null accuracy; supports attribute‑specific rules (e.g., exact match, tokenized text, numeric tolerance, set equality).
- Diagnostics: optional mismatch logs (text or JSONL) to inspect errors and refine rules or prompts.


# Normalization

Normalization prepares datasets for matching and fusion by cleaning headers and text, standardizing values and units, and detecting basic quality issues (nulls, types, outliers, duplicates). It aims to reduce noise while keeping transformations straightforward and reproducible.

Capabilities (module `PyDI.normalization`)
- Text & headers: clean punctuation/HTML/accents, standardize case/whitespace, normalize header tokens.
- Values: parse numbers, booleans, dates, lists, URLs, emails; handle common null markers.
- Units: extract and convert quantities (e.g., MB↔GB, temperature, frequency) to a preferred unit.
- Detection: null rate summaries, simple outlier flags, column/type inference, duplicate checks.
- Validation: lightweight helpers to check ranges and formats before downstream use.
- Normalization can be extended with custom functions (e.g., domain‑specific parsers or validators) and incorporated into your pipeline before schema/entity matching.




# Schema Matching

This module contains methods to automatically find 1-to-1 correspondences between columns across two datasets using label-, instance-, or duplicate-based schema matching strategies. Mapping columns containing corresponding information to a unified schema is important to prepare datasets for a high quality matching and merging during the following entity matching and data fusion steps.

Module: `PyDI.schemamatching`
- `LabelBasedSchemaMatcher`: compares the labels of the columns using similarity metrics to find schema correspondences. Fast and accurate when column labels are meaningful.
- `InstanceBasedSchemaMatcher`: compares the distributions of values per column via TF/TF‑IDF/binary vectors and cosine/Jaccard/containment similarity. Better suited than LabelBasedMatcher if column labels are ambigous.
- `DuplicateBasedSchemaMatcher`: leverage known record correspondences to infer column alignments from co‑occurring values in columns of the corresponding records. Great when a labeled set of matching records between datasets exists.
- `SchemaMappingEvaluator`: offers methods for evaluating a generated schema mapping given a labeled set of schema correspondences.

Example schema matching
```python
from PyDI.schemamatching import InstanceBasedSchemaMatcher

matcher = InstanceBasedSchemaMatcher(method="tfidf", similarity="cosine")
corr = matcher.match([df_a, df_b], threshold=0.75)
```

Example evaluation
```python
from PyDI.schemamatching import SchemaMappingEvaluator

metrics = SchemaMappingEvaluator.evaluate(corr, test_set)
```

Artifacts
- Schema correspondences written to file


# Data Translation

Translate/align source datasets to a target schema using a schema mapping. Translation prepares datasets for matching and fusion by ensuring consistent attribute names and enabling comparison.

Module: `PyDI.datatranslation`
- `MappingTranslator`: renames dataframe columns to target names if a 1:1 mapping exists given a set of schema correspondences.

Usage
```python
from PyDI.datatranslation import MappingTranslator

translator = MappingTranslator()
df_aligned = translator.translate(df, corr)
```

Artifacts
- Dataframe with aligned column names according to a schema correspondence file.


# Entity Matching

Computes record‑level correspondences between datasets using rule‑based, ML‑based, PLM-based, or LLM‑based matchers. Matchers operate on candidate record pairs. These pairs can either be created by building all combinations between records from two datasets (cartesian product) or by using a blocking method for more efficient pair building. PyDI offers a StandardBlocker, SortedNeighborhoodBlocker, TokenBlocker and EmbeddingBlocker. Both blocking and entity matching steps can be separately evaluated and debugged using extensive debug logging capabilities. The entity matchers produce a set of record correspondences between two datasets as output.


Module: `PyDI.entitymatching`

Blockers
- `blocking.NoBlocking`: Creates all possible candidate pairs (cartesian product).
- `blocking.StandardBlocker`: Creates candidate pairs based on defined blocking key.
- `blocking.SortedNeighborhoodBlocker`: Creates candidate pairs based on defined blocking key combined with rolling window over neighborhood.
- `blocking.TokenBlocker`: Blocks based on token or n-gram overlap.
- `blocking.EmbeddingBlocker`: Embeds all records using sentence-transformers and generates candiate pairs via nearest neighbor search in the vector space.

Example (Blocking without matching step)
```python
from PyDI.entitymatching.blocking import StandardBlocking

blocker = StandardBlocking(df_left, df_right, on=["title","year"], batch_size=100_000, out_dir="output/matching/blocking")
for batch in blocker:  # yields DataFrames with [id1, id2, block_key]
    process(batch)
# alternatively calling blocker.materialize() directly materializes all pairs
```

Evaluation (Blocking)
```python
from PyDI.entitymatching.blocking import BlockingEvaluator

report = BlockingEvaluator.evaluate(candidates=blocker, gold=test_pairs, out_dir="output/matching/blocking_eval")
```

Matchers
- `Comparators`: A set of attribute comparators that can be used together with the RuleBasedMatcher and FeatureExtractor. Each comparator consists of an attribute and a similarity metric for comparing two values of that attribute.
- `RuleBasedMatcher`: Composes a set of attribute comparators (e.g., Jaccard on title, date proximity) and assigns weights to calculate record pair similarity. A manually assigned threshold for the rule allows the classification of record pairs into matches and non-matches.
- `FeatureExtractor` and `VectorFeatureExtractor`: Used to generate features for machine-learning based matchers in PyDI. FeatureExtractor uses similarity-based metrics like Jaccard or Levenshtein. VectorFeatureExtractor generates embedding vectors (BoW and sentence-transformers).  
- `MLBasedMatcher`: takes a trained scikit-learn model as input and uses the model to classify record pairs and create a set of correspondences between two datasets.
- `PLMBasedMatcher`: takes an off-the-shelf or fine-tuned huggingface transformer model and performs entity matching. Returns a set of correspondences.
- `LLMBasedMatcher`: calls external hosted LLMs (e.g. OpenAI) to perform the entity matching and return a set of correspondences.
- `EntityMatchingEvaluator`: evaluation methods for blocking (pair completeness, pair quality, reduction ratio) and matching (Accuracy, Precision, Recall, F1). Supports writing detailed console logs and debug files.



Example (Rule‑based Matcher)
```python
from PyDI.entitymatching.rule_based import RuleBasedMatcher
from PyDI.utils import jaccard, date_within_years

matcher = RuleBasedMatcher(comparators=[jaccard("title"), date_within_years("date", 2)], weights=[0.7, 0.3], threshold=0.6, out_dir="output/matching/rule_based")
correspondences = matcher.match(df_left, df_right, candidates=blocker)
```

Evaluation (Matching)
```python
from PyDI.entitymatching.evaluation import EntityMatchingEvaluator

metrics = EntityMatchingEvaluator.evaluate_matching(corr=correspondences, test_pairs=test_pairs, threshold=0.7)
```

Artifacts
- Set of pairwise record correspondences written to file. Optionally detailed debugging logs written to file.


# Data Fusion

Data Fusion merges matched records into one consolidated dataset and then evaluates the result. Conflicts between source values are resolved by per‑attribute rules. Evaluation supports exact and fuzzy comparison.

Modules

-   `PyDI.fusion.engine` — fusion with strategy rules
-   `PyDI.fusion.conflict_resolution` — built‑in resolution functions
-   `PyDI.fusion.strategy.DataFusionStrategy` — register rules and eval functions
-   `PyDI.fusion.evaluation`, `PyDI.fusion.reporting` — metrics and reports

Conflict Resolution Rules (overview)

-   Value‑based (by attribute type)
    -   Strings: `longest_string`, `shortest_string`, `most_complete`
    -   Numerics: `average`, `median`, `maximum`, `minimum`, `sum_values`
    -   Dates: `most_recent`, `earliest`
    -   Lists/Sets: `union`, `intersection`, `intersection_k_sources`
-   Source‑based (use multiple inputs)
    -   `voting` (majority), `weighted_voting` (weights), `favour_sources` (priority order), `random_value` (tie‑break)
-   Custom: any callable that returns a `FusionResult`

Evaluation and Reporting

-   DataFusionEvaluator compares fused vs. gold by ID. It supports attribute‑specific evaluation functions, enabling fuzzy assessment beyond exact equality, e.g. `tokenized_match`, `numeric_tolerance_match`, `year_only_match`.
-   Additional outputs: consistency summaries, rule usage, record/attribute coverage, and JSON/HTML reports via FusionReport.

Example

```python
from PyDI.fusion.strategy import DataFusionStrategy
from PyDI.fusion.engine import DataFuser
from PyDI.fusion.conflict_resolution.numeric import average
from PyDI.fusion.conflict_resolution.date import most_recent
from PyDI.fusion.conflict_resolution.list import union

strategy = DataFusionStrategy()
strategy.add_attribute_fuser("title", union)
strategy.add_attribute_fuser("release_date", most_recent)
strategy.add_attribute_fuser("rating", average)

fuser = DataFuser(strategy=strategy, out_dir="output/fusion")
fused_df = fuser.fuse(datasets=[df_a, df_b], correspondences=matches)
```

Artifacts

-   Fused dataset (CSV/Parquet) and JSON/HTML reports under `out_dir`.


# Utils

Utils provides shared helpers used across the framework, notably a similarity metric registry using the `textdistance` package and consistent logging for LLM invocations.

The Similarity Metric Registry

`PyDI.utils.similarity_registry.SimilarityRegistry` centralizes access to similarity functions with name/category lookup and recommended sets for common use cases.

Available Metrics
- Edit‑based: hamming, levenshtein, damerau_levenshtein, jaro_winkler, jaro, strcmp95, needleman_wunsch, gotoh, smith_waterman, mlipns, editex
- Token‑based: jaccard, sorensen_dice, tversky, overlap, tanimoto, cosine, monge_elkan, bag
- Sequence‑based: lcsseq, lcsstr, ratcliff_obershelp
- Simple: prefix, postfix, length, identity
- Phonetic: mra

LLM Invocation Logging

The LLM logging helpers (`PyDI.utils.llm`) standardize how prompts, responses, token usage, and model/provider details are captured. They enable comparable debugging and usage tracking across LLM‑based extractors and matchers, and integrate with artifact writing so traces can be reviewed alongside other pipeline outputs.


# Tutorial

For a fully implemented example of a data integration workflow for integrating movie datasets from data loading, profiling, over entity matching and data fusion, refer to the [Tutorial Notebook](../tutorial/PyDI_Tutorial.ipynb)
