# Python Data Integration Framework ("PyDI") --- High‑Level Design

> Goal: a lightweight, pandas‑first framework for end‑to‑end data
> integration--- usable by students and researchers, and friendly to LLM
> agents. The design avoids deep class hierarchies and can be used
> sequentially in plain Python scripts. A pipeline orchestrator is
> optional and **not** required for day‑1.

## 1) Purpose & audience

* **Audience:** course participants, researchers, and LLM agents using
  > tool calls.

* **Use cases:** schema matching, data translation, blocking + entity
  > matching, data fusion, evaluation & reporting.

## 2) Design principles

1. **Simplicity over inheritance.** Small classes and functions;
   > minimal abstract bases.

2. **DataFrames first.** All core methods accept/return
   > pandas.DataFrames.

3. **Native metadata.** Store dataset‑level and column‑level metadata
   > in DataFrame.attrs / Series.attrs.

4. **Deterministic and refactor‑safe.** Stable signatures + type
   > hints + rich docstrings for LLMs.

5. **Observability by default.** Every step logs to console **and**
   > writes artifacts to disk (CSV/JSON/HTML, optional XML).

6. **Configurable + reproducible.** JSON/YAML configs and manifest
   > files for each run.

7. **Modular adoption.** Each step can be called directly from a Python
   > script; no orchestrator needed.

8. **Leverage existing libs.** ydata-profiling, Sweetviz, scikit‑learn,
   > networkx, rapidfuzz/textdistance, sentence-transformers.

## 3) Data model & conventions

### 3.1 Data structure

* **Primary structure:** pandas.DataFrame for each dataset.

* **Global record ID:** add a column `_id` with values:
  > f"{dataset_name}_{i}" (e.g., movies_000123).

* **Dataset name:** required string; stored in
  > `df.attrs["dataset_name"] = "movies"`.

### 3.2 Metadata & provenance (stored natively)

* **Dataset‑level (DataFrame.attrs)**:

````
df.attrs.update({
    "dataset_name": "movies",
    "source": {"path": "input/movies.csv", "format": "csv"},
    "score": 0.7,  # optional: source quality/priority
    "date": "2020-01-01",  # optional: data currency
    "provenance": [
        {"op": "load_csv", "params": {"path": ".../movies.csv"}, "ts": "2025-08-11T10:00:00Z"}
    ]
})
````

* **Column‑level (Series.attrs)** may hold semantic hints (datatype,
  > regex, unit).

````
col.attrs.update({
    "provenance": [
        {"op": "schema_transform", "params": {"name_old": "clock_rate", "name_new": "clock_speed"}, "ts": "2025-08-11T10:00:00Z"}
    ],
    "unit": "GHz",
    "datatype": "integer",
})
````

### 3.3 File/location conventions

* **Run directory:** `runs/{timestamp}/`

* **Artifacts per step:** `runs/{ts}/{step_name}/` with CSV/JSON/HTML
  > (and optional XML).

* **Manifest:** `runs/{ts}/manifest.json` summarising inputs, outputs,
  > configs, and metrics.

* **Important:** By default do not include the timestamp in the folder
  > but make this a flag. In the default setting there should just be
  > a simple `input/` and `output/` folder.

## 4) Components (high‑level API)

> Each component is independent; instantiate and call its methods in
> sequence.

### 4.1 Profiling

**Class:** `DataProfiler`  
**Responsibilities:**

* ydata profiling for a single dataset → HTML (+ optional JSON).
* Sweetviz comparison between two datasets → HTML.

**Key methods (suggested signatures):**

```python
profile(df: pd.DataFrame, out_dir: str) -> str  # returns HTML path
compare(df_a: pd.DataFrame, df_b: pd.DataFrame, out_dir: str) -> str
summary(df: pd.DataFrame) -> dict  # quick stats for console
```

### 4.2 Schema matching

**Artifacts:** `SchemaMapping` represented as a DataFrame with columns:  
`source_dataset`, `source_column`, `target_dataset`, `target_column`, `score`, `notes`

**Classes/Functions:**

* `BaseSchemaMatcher`: interface (thin ABC).
* `SimpleSchemaMatcher`: label‑based and value‑based matching.

**Key method:**

```python
match(
    datasets: list[pd.DataFrame],
    method: str = "label",
    preprocess: callable | None = None,
    threshold: float = 0.8,
) -> SchemaMapping
```

**Evaluation (separate object):**

```python
SchemaMappingEvaluator.evaluate(
    corr: SchemaMapping,
    test_set: SchemaMapping,
    *,
    threshold: float | None = None,
    by: list[str] | None = None,
) -> dict  # P/R/F1, coverage by dataset/column, confusion table

SchemaMappingEvaluator.sweep_thresholds(
    corr: SchemaMapping,
    gold: SchemaMapping,
    *,
    thresholds: list[float],
) -> pd.DataFrame  # PR curve & best‑F1
```

### 4.3 Data translation, information extraction & normalization

**Components (matcher‑style):**

* `BaseTranslator` (ABC)
* `MappingTranslator.translate(df: pd.DataFrame, corr: SchemaMapping) -> pd.DataFrame`

* `BaseExtractor` (ABC)  
  `RegexExtractor.extract(df: pd.DataFrame, rules: dict[str, str]) -> pd.DataFrame`  
  `CodeExtractor.extract(df: pd.DataFrame, funcs: dict[str, callable]) -> pd.DataFrame`  
  `LLMExtractor(model: str, prompts: dict[str, str]).extract(df: pd.DataFrame) -> pd.DataFrame`  # optional

* `BaseNormalizer` (ABC)  
  `Normalizer.apply(df: pd.DataFrame, rules: dict[str, callable]) -> pd.DataFrame`  
  `NormalizationUtils`: lowercase, strip, remove_punctuation, normalize_date, …

* `BaseValidator` (ABC)  
  `PydanticValidator.validate(df: pd.DataFrame, model: BaseModel) -> pd.DataFrame`

### 4.4 Blocking & entity matching

**Data structures:**

* *Candidate pairs (small):* DataFrame with `id1`, `id2`, …
* *CandidateBatchStream (large):* Iterable[pd.DataFrame] where each
  batch has required columns [`id1`, `id2`], optional [`block_key` and features]. Use `DataFrame.attrs` for provenance/counters.
* *Correspondences:* DataFrame with `id1`, `id2`, `score`, `notes` (plus helpers as a `CorrespondenceSet` class).

**Blocking strategies (small classes):**

* `BaseBlocker` (ABC)
    * Initialized with `df_left`, `df_right` and strategy params.
    * `__iter__(self) -> Iterator[pd.DataFrame]`: yields candidate batches with columns [`id1`, `id2`, `block_key?`] of size ≤ `batch_size`.
    * `batch_size: int` (default 100_000); `estimate_pairs() -> int | None`; `stats() -> dict`; `materialize() -> pd.DataFrame` (for small data).
* `NoBlocking(df_left, df_right, *, batch_size: int = 100_000)`
* `StandardBlocking(df_left, df_right, on: list[str], *, batch_size: int = 100_000)`
* `SortedNeighbourhood(df_left, df_right, key: str, window: int, *, batch_size: int = 100_000)`
* `TokenBlocking(df_left, df_right, column: str, tokenizer: callable | None = None, *, batch_size: int = 100_000)`
* `EmbeddingBlocking(df_left, df_right, text_cols: list[str], model: str, threshold: float, *, batch_size: int = 25_000)`

Notes:

* All blockers stream pairs; they never materialize the full cross‑product. Each batch DataFrame may carry attrs like `{"pairs": N, "block_key": "k"}`.
* Blockers may persist intermediate indexes under `runs/{ts}/matching/blocking/`.

**Matchers:**

* *Weighted linear combination of attribute similarities (e.g., title
  Jaccard + date proximity)*

```python
RuleBasedMatcher.match(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    candidates: Iterable[pd.DataFrame],
    comparators: list[callable] | list[{comparator: callable, weight: float}],
    weights: list[float] | None = None,
    threshold: float = 0.0,
) -> CorrespondenceSet

MLBasedMatcher.train(pairs: pd.DataFrame, labels: pd.Series, model: str) -> Any
MLBasedMatcher.match(df_left: pd.DataFrame, df_right: pd.DataFrame, candidates: Iterable[pd.DataFrame]) -> CorrespondenceSet
LLMBasedMatcher.match(df_left: pd.DataFrame, df_right: pd.DataFrame, candidates: Iterable[pd.DataFrame], prompt) -> CorrespondenceSet
```

**Post‑processing:**

* `CorrespondenceSet.max_weight_bipartite_matching()` (via networkx)
* `CorrespondenceSet.apply_stable_marriage()` (optional)
* clustering utilities.

**Evaluation:**

```python
EntityMatchingEvaluator.evaluate(
    corr: CorrespondenceSet,
    test_pairs: pd.DataFrame,
    *,
    threshold: float | None = None,
) -> dict  # P/R/F1, candidate recall, pair reduction

EntityMatchingEvaluator.create_cluster_consistency_report(
    correspondences: CorrespondenceSet
) -> pd.DataFrame

EntityMatchingEvaluator.write_record_groups_by_consistency(
    out_path: str,
    correspondences: CorrespondenceSet
) -> str
```

### 4.5 Data fusion

**Concepts:** `ConflictResolutionFunction` per attribute (e.g., majority,
shortest/longest, most_recent, favour_source, union, custom fn).

**Class:** `DataFuser`

**Key methods:**

```python
fuse(
    datasets: list[pd.DataFrame],
    correspondences: CorrespondenceSet,
    rules: dict[str, FusionRule],
) -> pd.DataFrame

FusionEvaluator.evaluate_density(df: pd.DataFrame) -> dict
FusionEvaluator.evaluate_consistency(datasets: list[pd.DataFrame], correspondences: CorrespondenceSet) -> dict
FusionEvaluator.evaluate_accuracy(fused: pd.DataFrame, test_set: pd.DataFrame, rules: dict[str, callable]) -> dict

Exporters: CSV/JSON built‑in; XML writer convenience:
write_xml(df: pd.DataFrame, path: str, root: str = "records", row: str = "record") -> str
```

## 5) LLM‑agent readiness

* **Stable signatures + type hints + docstrings** suitable for tool calling.
* **Every step writes files** (CSV/JSON/HTML/XML) + **returns paths** for downstream tools.
* **Verbose debug mode**: sample inputs/outputs, parameter echo, and warnings captured to `runs/{ts}/{step}`.
* **Config files**: Allow calling `.run(config="config.json")` for reproducibility.
* **Dry‑run**: Output planned actions and expected artifacts without executing heavy steps.

## 6) Minimal sequential example

```python
df_a = load_csv("input/a.csv", dataset_name="a")  # sets _id and attrs
df_b = load_csv("input/b.csv", dataset_name="b")

# Profiling
DataProfiler().profile(df_a, "runs/2025-08-11/profiling")
DataProfiler().profile(df_b, "runs/2025-08-11/profiling")

# Schema matching
corr = SimpleSchemaMatcher().match([df_a, df_b], method="label", preprocess=str.lower, threshold=0.8)

# Data translation (align schemas)
df_a_aligned = MappingTranslator().translate(df_a, corr)
df_b_aligned = MappingTranslator().translate(df_b, corr)

# Blocking + matching
cands = SortedNeighbourhood(df_a_aligned, df_b_aligned, key="title", window=5)
matches = RuleBasedMatcher().match(
    df_a_aligned,
    df_b_aligned,
    cands,
    comparators=[jaccard("title"), date_within_years("date", 2)],
    weights=[0.5, 0.5],
    threshold=0.7,
)
matches = matches.max_weight_bipartite_matching()

# Fusion
fused = DataFuser().fuse(
    [df_a_aligned, df_b_aligned],
    matches,
    rules={
        "title": ConflictResolutionFunction(strategy="longest"),
        "date": ConflictResolutionFunction(strategy="most_recent"),
        "actors": ConflictResolutionFunction(strategy="union"),
    },
)
write_xml(fused, "runs/2025-08-11/fused.xml")
```

### Development artifacts

The `input/` and `output/` folders contain reference datasets and expected results from the WInte.r framework that should be used during PyDI development:

* **Validation:** Compare PyDI outputs against the expected results in `output/` to ensure correctness.
* **Testing:** Use the datasets in `input/` as standard test cases for unit and integration tests.
* **Benchmarking:** Measure PyDI performance against WInte.r using the same input datasets.
* **Examples:** Demonstrate PyDI capabilities using the provided movie datasets (academy awards, actors, golden globes).

The movie datasets represent a typical data integration scenario with entity matching between different sources and subsequent data fusion, making them ideal for end-to-end pipeline development and testing.

## 8) Repository layout (proposal)

```
PyDI/
profiling/ # ydata, sweetviz wrappers
entitymatching/  # blocking, comparators, rule-based, ml, embedding
schemamatching/    # schema matching + mappings
datatranslation/ # schema translation and mapping
informationextraction/ # feature extraction 
normalization/ # normalization and validation
fusion/    # fusion rules, engine, evaluation, reports
utils/     # logging, file manifests, metrics, common types
examples/  # small end-to-end scripts
tests/
docs/

input/     # Example datasets and test cases from WInte.r for development
entitymatching/  # Movie datasets (academy_awards, actors) with train/test splits
  data/          # XML source datasets 
  splits/        # Training and test correspondences for evaluation
fusion/          # Movie datasets with pre-computed correspondences  
  data/          # XML source datasets (academy_awards, actors, golden_globes)
  correspondences/ # Pre-computed entity matches between datasets
  splits/        # Gold standard fused data for evaluation
schemamatching/  # Schema matching test cases
  data/          # Source datasets in various formats
  targetschema/  # Predefined target schemas (XML/XSD) for alignment

output/    # Example outputs from WInte.r for validation and testing
entitymatching/  # Expected results from entity matching pipeline
  # Correspondence files, debug logs, blocking results, matching evaluations
fusion/          # Expected results from data fusion pipeline  
  # Fused datasets, consistency reports, record-level fusion details
```