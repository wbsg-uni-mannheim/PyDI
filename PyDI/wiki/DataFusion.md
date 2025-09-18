# Data Fusion

Data Fusion merges matched records into one consolidated dataset and then evaluates the result. Conflicts between source values are resolved by per‑attribute rules; evaluation supports exact and fuzzy comparison.

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
