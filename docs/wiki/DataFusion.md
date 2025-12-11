# Data Fusion

Data Fusion consolidates groups of records that all describe the same real-world entity into a single record by resolving data conflicts. The input of the data fusion process are two or more datasets together with entity correspondences that specify which records refer to the same real-world entity. The output of the process is a single fused dataset.

For resolving data conflicts, PyDI's `DataFusionEngine` uses `DataFusionStrategy`s which specify a conflict resolution function for each attribute. 
The `DataFusionEvaluator` allows you to compare the resulting fused dataset to a ground truth dataset. `FusionReport` generates detailed logs about the data fusion process. The logs can be used as a starting point for improving the `DataFusionStrategy`.


## Requirements

- Each input `DataFrame` must set `df.attrs["dataset_name"]`.
- Records need a stable identifier. Provide `_id`/`id` or pass `id_column` to the engine.
- Correspondences must be a `DataFrame` with at least `id1`, `id2` (see [Entity Matching](EntityMatching.md)).

## Fusion Strategy

The fusion strategy assigns a conflict resolution function to each attribute of the datasets. The conflict resolution function chooses or generates fused value given the set of all values of an attribute in the records that describe a specific entity. 

## Conflict Resolution Functions

PyDI implements the following conflict resolution functions (called resolvers):
- Strings: `longest_string`, `shortest_string`, `most_complete`
- Numerics: `average`, `median`, `maximum`, `minimum`, `sum_values`
- Dates: `most_recent`, `earliest`
- Lists/Sets: `union`, `intersection`, `intersection_k_sources`
- Source-aware: `voting`, `weighted_voting`, `favour_sources`, `prefer_higher_trust`, `random_value`

### Attribute Fusers

An attribute fuser assigns a resolver to an attribute. Use `add_attribute_fuser(attr, resolver, **kwargs)` to add an attribute fuser to a fusion strategy.

```python
from PyDI.fusion import DataFusionStrategy
from PyDI.fusion import longest_string, union, prefer_higher_trust

strategy = DataFusionStrategy("movie_fusion_strategy")
strategy.add_attribute_fuser("title", longest_string)
strategy.add_attribute_fuser("director_name", longest_string)
strategy.add_attribute_fuser("date", prefer_higher_trust, trust_key="trust_score")
strategy.add_attribute_fuser("actors_actor_name", union)
```

### Custom Resolvers

You can extend PyDI by implementing custom resolvers. Resolvers are simple callables: `resolver(values, **kwargs) -> (value, confidence, metadata)`.

```python
def pick_longest_nonempty(values, **kwargs):
    texts = [str(v) for v in values if v is not None and str(v).strip()]
    if not texts:
        return None, 0.0, {"reason": "no_valid_values"}
    winner = max(texts, key=len)
    # Confidence grows with the length gap to second best
    second = max((t for t in texts if t != winner), default="", key=len)
    conf = 1.0 if not second else 0.5 + min(0.5, (len(winner) - len(second)) / max(1, len(winner)))
    return winner, conf, {"rule": "pick_longest_nonempty", "candidates": len(texts)}

strategy.add_attribute_fuser("title", pick_longest_nonempty)
```

## Fusion Engine

The fusion engine executes the data fusion process by 1) grouping records using the provided correspondences, 2) applying the defined attribute fusers to each group of records, and 3) collecting debug metadata about the fusion process.

```python
from PyDI.fusion import DataFusionEngine

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_file="fusion_debug.jsonl",
    debug_format="json",
)

fused = engine.run(
    datasets=[df_a, df_b, df_c],
    correspondences=corr,
    id_column="_id",                    # or {dataset_name: id_col}
    schema_correspondences=None,        # optional column alignment
    include_singletons=True,            # keep unmatched records
)
```

Notes
- Pass dataset trust via `df.attrs` or `_trust` column. The engine supplies a `trust_map` to resolvers like `prefer_higher_trust`.
- Output includes `_fusion_group_id`, `_fusion_sources`, `_fusion_confidence`, and `_fusion_metadata` (rule, sources, inputs per attribute).


## Evaluating Data Fusion Results

The `DataFusionEvaluator` allows you to compare fused datasets to a ground truth dataset and to calculate the accuracy of the fused values. For determining the accuracy of fused values minor differences to ground truth values are often irrelevant, e.g. if the fusion process determines the population of a city to be 1 million persons and the ground truth value is 1.002 million persons, you might still want to consider the fused value as correct. PyDI allows you to define tolerance ranges within which fused values are still considered as correct by specifying evaluation functions.  

## Evaluation Functions

Exact equality often penalizes harmless differences (token order, punctuation, year vs. full date). Bind attribute-specific evaluation rules to get quality metrics.

PyDi provides the following functions that can be used to define tolerance ranges:
- `tokenized_match(threshold=...)` – Jaccard over tokens or sets 
- `year_only_match` – compare only date years
- `numeric_tolerance_match(tolerance=...)` – numeric closeness
- `set_equality_match` – order-independent list/set equality
- `boolean_match` – includes boolean normalization

The code below demonstrates how tolerance ranges are defined for the attributes covered by a `DataFusionStrategy`:

```python
from PyDI.fusion import tokenized_match, numeric_tolerance_match, year_only_match

strategy.add_evaluation_function("title", tokenized_match, threshold=0.9)
strategy.add_evaluation_function("date", year_only_match)
strategy.add_evaluation_function("rating", numeric_tolerance_match, tolerance=0.05)
```

### Custom Evaluation

You can also define a custom evaluation function `(fused, expected, **params) -> bool` tailored to your use case and register it with the strategy.

```python
# Example: prefix match with configurable minimum length
def title_prefix_match(fused, expected, min_prefix: int = 6) -> bool:
    a, b = str(fused).strip().lower(), str(expected).strip().lower()
    n = min(len(a), len(b), max(0, min_prefix))
    return n == 0 or a[:n] == b[:n]

strategy.add_evaluation_function("title", title_prefix_match, min_prefix=8)
```

## Evaluator

The `DataFusionEvaluator` compares a fused dataset to a testset (ground truth) and calcuated the accuracy of the values of the fused dataset using the strategy’s evaluation functions (exact by default). 

```python
from PyDI.fusion import DataFusionEvaluator, tokenized_match, year_only_match

# reuse the fusion strategy from earlier
evaluator = DataFusionEvaluator(strategy, debug=True, fusion_debug_logs="fusion_debug.jsonl")
metrics = evaluator.evaluate(
    fused_df=fused, fused_id_column="_id", expected_df=test_set, expected_id_column="_id"
)
print({k: round(v, 3) for k, v in metrics.items() if isinstance(v, (int, float))})
```


## Tuning with Debug Logs

Evaluation mismatch logs show where fused results diverge from the validation/testset. They include the evaluation rule used, the fusion rule that produced the value, and the exact inputs considered. Subsequently, aiding in the refinement of fusion rules and thresholds.

```json
{
   "type":"evaluation_mismatch",
   "attribute":"title",
   "fused_id":"academy_awards_4270",
   "expected_id":"academy_awards_4270",
   "fused_value":"The Great Zeigfeld",
   "expected_value":"The Great Ziegfeld",
   "evaluation_rule":"tokenized_match",
   "conflict_rule":"longest_string",
   "inputs":[
      {
         "record_id":"actors_9",
         "dataset":"actors",
         "value":"The Great Zeigfeld"
      },
      {
         "record_id":"academy_awards_4270",
         "dataset":"academy_awards",
         "value":"The Great Ziegfeld"
      }
   ],
   "reason":"mismatch"
}
```

## End-to-End Example: Fusing the Movie Datasets 

The end-to-end example below illustrates the application of PyDI's data fusion functionality for fusing to the movie datasets. 

```python
from PyDI.fusion import (
    DataFusionStrategy, DataFusionEngine, FusionReport, DataFusionEvaluator,
    longest_string, union, prefer_higher_trust, tokenized_match,
)

# 1) Configure data fusion strategy for movies
strategy = DataFusionStrategy("movie_fusion_strategy")
# fusion rules
strategy.add_attribute_fuser("title", longest_string)
strategy.add_attribute_fuser("director_name", longest_string)
strategy.add_attribute_fuser("date", prefer_higher_trust, trust_key="trust_score")
strategy.add_attribute_fuser("actors_actor_name", union)
# evaluation rule
strategy.add_evaluation_function("title", tokenized_match, threshold=0.9)

# 2) Configure data fusion engine and run fusion for the three movies datasets
engine = DataFusionEngine(strategy, debug=True, debug_file="fusion_debug.jsonl", debug_format="json")
fused = engine.run([df_a, df_b, df_c], correspondences=corr, id_column="_id")

# 3) Evaluate data fusion results by comparing them to a ground truth dataset
metrics = DataFusionEvaluator(strategy, fusion_debug_logs="fusion_debug.jsonl").evaluate(
    fused_df=fused, fused_id_column="_id", expected_df=test_set, expected_id_column="_id"
)
print("Overall accuracy:", metrics.get("overall_accuracy"))

# 4) Generate data fusion report
report = FusionReport(fused, [df_a, df_b, df_c], strategy.name, correspondences=corr)
report.print_summary()
open("fusion_report.html", "w").write(report.to_html())
```

## Tutorials

- [Data Integration Tutorial](../tutorial/entity_matching_and_fusion/data_integration_tutorial.ipynb) - Complete pipeline including data fusion with conflict resolution strategies and evaluation
