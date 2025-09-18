# Schema Matching

Find correspondences between columns across datasets using label-, instance-, or duplicate-based strategies. A good schema mapping improves downstream translation, blocking, and fusion quality.

Approaches
- Label‑based: compare normalized column headers (e.g., `title` vs `film_title`). Fast, robust when headers are meaningful.
- Instance‑based: compare distributions of values per column via TF/TF‑IDF/binary vectors and cosine/Jaccard/containment similarity. Strong when headers are ambiguous.
- Duplicate‑based: use known record correspondences to infer column alignments from co‑occurring values. Great when a small gold set exists.

Modules
- `PyDI.schemamatching.base`
  - `BaseSchemaMatcher.match(datasets, method=..., preprocess=None, threshold=0.8) -> pd.DataFrame`
- `PyDI.schemamatching.label_based.LabelBasedSchemaMatcher` — column label similarity
- `PyDI.schemamatching.instance_based.InstanceBasedSchemaMatcher` — value distribution similarity (tf, binary, tf‑idf; cosine/jaccard/containment)
- `PyDI.schemamatching.duplicate_based.DuplicateBasedSchemaMatcher` — leverage known record correspondences
- `PyDI.schemamatching.evaluation.SchemaMappingEvaluator`

Schema Mapping Format (DataFrame)
- Columns: `source_dataset`, `source_column`, `target_dataset`, `target_column`, `score`, `notes`
- One row per proposed correspondence. Multiple candidates may exist per column before selection.

Preprocessing and Similarity
- Use simple preprocessors like `str.lower`, `strip`, or custom functions to normalize labels/values.
- Similarity functions are validated via `SimilarityRegistry` in `PyDI.utils`.

Thresholds and Selection
- Start with a conservative threshold (e.g., 0.7–0.8), then sweep to inspect precision‑recall trade‑offs.
- Resolve many‑to‑one cases by choosing the highest score per source (or per target) depending on use case.

Example
```python
from PyDI.schemamatching import InstanceBasedSchemaMatcher

matcher = InstanceBasedSchemaMatcher(method="tfidf", similarity="cosine")
corr = matcher.match([df_a, df_b], threshold=0.75)
```

Evaluation
```python
from PyDI.schemamatching import SchemaMappingEvaluator

metrics = SchemaMappingEvaluator.evaluate(corr, gold, threshold=0.8)
sweep = SchemaMappingEvaluator.sweep_thresholds(corr, gold, thresholds=[0.5,0.6,0.7,0.8,0.9])
```

Pitfalls and Tips
- Header noise: normalize headers (see Normalization) before label‑based matching.
- Sparse columns: prefer binary/containment similarity over TF‑IDF.
- Synonyms/semantics: consider Information Extraction to derive clearer attributes before matching.

Artifacts
- CSV/JSON exports of correspondences are commonly written by calling code.
