# Data Fusion

PyDI’s Fusion module consolidates matched records from multiple datasets into a single dataset. It resolves conflicting values per attribute using pluggable rules, tracks provenance, and provides evaluation and reporting utilities so you can audit and improve quality.

- Record grouping from correspondences (connected components)
- Strategy object to register per‑attribute rules and evaluation functions
- Attribute‑level conflict resolution via built‑in and custom rules
- Evaluation against a test dataset with exact and fuzzy comparisons
- Optional debug logs capturing inputs, chosen rules, and outputs


## Requirements

- Each input `DataFrame` must set `df.attrs["dataset_name"]`.
- Records need a stable identifier. Provide `_id`/`id` or pass `id_column` to the engine.
- Correspondences must be a `DataFrame` with at least `id1`, `id2` (see [Entity Matching](EntityMatching.md)).

## Fusion Strategy

Configure both fusion and evaluation in one place. The strategy binds per‑attribute rules that resolve conflicts during fusion and binds evaluation functions that assess the quality of those resolutions. 

### Fusion rules

An attribute fuser is a resolver that takes all values for one attribute across sources and returns a single value plus a confidence. Use `add_attribute_fuser(attr, resolver, **kwargs)` to register it.

Common built‑ins
- Strings: `longest_string`, `shortest_string`, `most_complete`
- Numerics: `average`, `median`, `maximum`, `minimum`, `sum_values`
- Dates: `most_recent`, `earliest`
- Lists/Sets: `union`, `intersection`, `intersection_k_sources`
- Source‑aware: `voting`, `weighted_voting`, `favour_sources`, `prefer_higher_trust`, `random_value`

### Custom resolvers

Resolvers are simple callables: `resolver(values, **kwargs) -> (value, confidence, metadata)`.

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

### Register rules

```python
from PyDI.fusion import DataFusionStrategy
from PyDI.fusion import longest_string, union, prefer_higher_trust

strategy = DataFusionStrategy("movie_fusion_strategy")
strategy.add_attribute_fuser("title", longest_string)
strategy.add_attribute_fuser("director_name", longest_string)
strategy.add_attribute_fuser("date", prefer_higher_trust, trust_key="trust_score")
strategy.add_attribute_fuser("actors_actor_name", union)
```


### Evaluation functions

Exact equality often penalizes harmless differences (token order, punctuation, year vs. full date). Bind attribute‑specific evaluation rules to get quality metrics.

Built‑ins
- `tokenized_match(threshold=...)` – Jaccard over tokens or sets 
- `year_only_match` – compare only date years
- `numeric_tolerance_match(tolerance=...)` – numeric closeness
- `set_equality_match` – order‑independent list/set equality
- `boolean_match` – includes boolean normalization

```python
from PyDI.fusion import tokenized_match, numeric_tolerance_match, year_only_match

strategy.add_evaluation_function("title", tokenized_match, threshold=0.9)
strategy.add_evaluation_function("date", year_only_match)
strategy.add_evaluation_function("rating", numeric_tolerance_match, tolerance=0.05)
```

### Custom evaluation

Define an evaluation function `(fused, expected, **params) -> bool` tailored to your use case and register it with the strategy.

```python
# Example: prefix match with configurable minimum length
def title_prefix_match(fused, expected, min_prefix: int = 6) -> bool:
    a, b = str(fused).strip().lower(), str(expected).strip().lower()
    n = min(len(a), len(b), max(0, min_prefix))
    return n == 0 or a[:n] == b[:n]

strategy.add_evaluation_function("title", title_prefix_match, min_prefix=8)
```


## Fusion Engine

Runs fusion end‑to‑end: builds record groups from correspondences, applies your strategy’s attribute fusers, and writes fusion metadata. Enable debug to emit per‑attribute logs that make decisions auditable.

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


## Evaluator

Measures fused vs. testset accuracy using the strategy’s evaluation rules (exact by default). Optionally load fusion debug logs so mismatches show the exact inputs and rules used during fusion.

```python
from PyDI.fusion import DataFusionEvaluator, tokenized_match, year_only_match

# reuse the fusion strategy from earlier
evaluator = DataFusionEvaluator(strategy, debug=True, fusion_debug_logs="fusion_debug.jsonl")
metrics = evaluator.evaluate(
    fused_df=fused, fused_id_column="_id", expected_df=test_set, expected_id_column="_id"
)
print({k: round(v, 3) for k, v in metrics.items() if isinstance(v, (int, float))})
```


## Tuning with debug logs

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

## Example: End‑to‑End (movies)

```python
from PyDI.fusion import (
    DataFusionStrategy, DataFusionEngine, FusionReport, DataFusionEvaluator,
    longest_string, union, prefer_higher_trust, tokenized_match,
)

# 1) Strategy
strategy = DataFusionStrategy("movie_fusion_strategy")
# fusion rules
strategy.add_attribute_fuser("title", longest_string)
strategy.add_attribute_fuser("director_name", longest_string)
strategy.add_attribute_fuser("date", prefer_higher_trust, trust_key="trust_score")
strategy.add_attribute_fuser("actors_actor_name", union)
# evaluation rule
strategy.add_evaluation_function("title", tokenized_match, threshold=0.9)

# 2) Engine
engine = DataFusionEngine(strategy, debug=True, debug_file="fusion_debug.jsonl", debug_format="json")
fused = engine.run([df_a, df_b, df_c], correspondences=corr, id_column="_id")

# 3) Evaluate
metrics = DataFusionEvaluator(strategy, fusion_debug_logs="fusion_debug.jsonl").evaluate(
    fused_df=fused, fused_id_column="_id", expected_df=test_set, expected_id_column="_id"
)
print("Overall accuracy:", metrics.get("overall_accuracy"))

# 4) Report
report = FusionReport(fused, [df_a, df_b, df_c], strategy.name, correspondences=corr)
report.print_summary()
open("fusion_report.html", "w").write(report.to_html())
```
