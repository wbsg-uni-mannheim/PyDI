# Tutorial

The tutorial mirrors the WInte.r movie example and demonstrates an end‑to‑end PyDI pipeline using the datasets in `input/` and expected outputs in `output/`.

Learning Goals
- Understand the flow from ingestion to fusion.
- See how artifacts (HTML/CSV/JSON) accumulate for observability.
- Practice threshold tuning and blocker choice.

Where to Start
- See `PyDI/tutorial/PyDI_Winter_Tutorial.ipynb` for a notebook walkthrough.
- Example outputs are under `PyDI/tutorial/output/movies/` (dataset profiles, etc.).

Minimal Pipeline (script‑style)
```python
from PyDI.io import load_csv
from PyDI.profiling import DataProfiler
from PyDI.schemamatching import InstanceBasedSchemaMatcher
from PyDI.datatranslation import MappingTranslator
from PyDI.entitymatching.blocking import SortedNeighbourhood
from PyDI.entitymatching.rule_based import RuleBasedMatcher
from PyDI.utils import jaccard, date_within_years
from PyDI.fusion.strategy import DataFusionStrategy
from PyDI.fusion.engine import DataFuser
from PyDI.fusion.conflict_resolution.date import most_recent
from PyDI.fusion.conflict_resolution.list import union

df_a = load_csv("input/movies/academy_awards.csv", dataset_name="academy_awards")
df_b = load_csv("input/movies/actors.csv", dataset_name="actors")

DataProfiler().profile(df_a, "output/movies/dataset-profiles")
DataProfiler().profile(df_b, "output/movies/dataset-profiles")

corr = InstanceBasedSchemaMatcher(method="tfidf").match([df_a, df_b], threshold=0.8)
df_a_aligned = MappingTranslator().translate(df_a, corr)
df_b_aligned = MappingTranslator().translate(df_b, corr)

cands = SortedNeighbourhood(df_a_aligned, df_b_aligned, key="title", window=5)
matches = RuleBasedMatcher(
    comparators=[jaccard("title"), date_within_years("date", 2)],
    weights=[0.5, 0.5],
    threshold=0.7,
    out_dir="output/movies/matching"
).match(df_a_aligned, df_b_aligned, candidates=cands)

strategy = DataFusionStrategy()
strategy.add_attribute_fuser("title", union)
strategy.add_attribute_fuser("date", most_recent)

fused = DataFuser(strategy=strategy, out_dir="output/movies/fusion").fuse([df_a_aligned, df_b_aligned], matches)
```

Suggested Experiments
- Swap `SortedNeighbourhood` for `StandardBlocking` on `(title, year)` and compare candidate recall.
- Try `LabelBasedSchemaMatcher` vs `InstanceBasedSchemaMatcher` and compare PR curves.
- Add a unit normalizer on amounts and observe fusion conflict rates.

Validation
- Compare against the reference outputs under `output/` for sanity checks during development.
