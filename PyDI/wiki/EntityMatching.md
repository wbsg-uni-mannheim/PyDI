# Entity Matching

Compute record‑level correspondences between datasets using rule‑based, ML‑based, or LLM‑based matchers. Matching operates on candidate pairs from Blocking and outputs scored correspondences.

Modules
- `PyDI.entitymatching.base.BaseMatcher`
- `PyDI.entitymatching.rule_based` — weighted similarity of comparators
- `PyDI.entitymatching.ml_based.MLBasedMatcher` — scikit‑learn models
- `PyDI.entitymatching.llm_based.LLMBasedMatcher` — optional LLM calls with artifacts
- `PyDI.entitymatching.comparators` — string/numeric/date comparators
- `PyDI.entitymatching.feature_extraction` — helpers for ML pipelines
- `PyDI.entitymatching.evaluation` — PR/F1, clustering, threshold sweeps

Rule‑Based Matching
- Compose attribute comparators (e.g., Jaccard on title, date proximity) and assign weights.
- Set a threshold to emit correspondences; tune via evaluation.

ML‑Based Matching
- Extract feature vectors for pairs and train a classifier (logreg, random forest, etc.).
- Use probability scores as match scores; calibrate thresholds using validation sets.

LLM‑Based Matching
- Useful for ambiguous records with rich text; more expensive and slower.
- Strong auditing via prompt/response artifacts; consider batching and caching.

Post‑Processing
- Deduplicate pairs, enforce 1:1 via max‑weight bipartite matching, or cluster by connected components before fusion.

Example (Rule‑based)
```python
from PyDI.entitymatching.rule_based import RuleBasedMatcher
from PyDI.utils import jaccard, date_within_years

matcher = RuleBasedMatcher(comparators=[jaccard("title"), date_within_years("date", 2)], weights=[0.7, 0.3], threshold=0.6, out_dir="output/matching/rule_based")
correspondences = matcher.match(df_left, df_right, candidates=blocker)
```

Example (ML‑based)
```python
from PyDI.entitymatching.ml_based import MLBasedMatcher

ml = MLBasedMatcher(model="random_forest", features=[...], out_dir="output/matching/ml")
ml.train(pairs=train_pairs, labels=train_labels)
correspondences = ml.match(df_left, df_right, candidates=blocker)
```

Evaluation
```python
from PyDI.entitymatching.evaluation import EntityMatchingEvaluator

metrics = EntityMatchingEvaluator.evaluate(corr=correspondences, test_pairs=test_pairs, threshold=0.7)
```

Artifacts
- Candidate summaries, match scores, prompts/responses (LLM), eval reports under `out_dir`.
