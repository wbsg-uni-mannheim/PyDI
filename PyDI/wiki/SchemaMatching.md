# Schema Matching

This module contains methods to automatically find 1-to-1 correspondences between columns across two datasets using label-, instance-, or duplicate-based schema matching strategies. Mapping columns containing corresponding information to a unified schema is important to prepare datasets for a high quality matching and merging during the following entity matching and data fusion steps.

Module: `PyDI.schemamatching`
- `LabelBasedSchemaMatcher`: compares the labels of the columns using similarity metrics to find schema correspondences. Fast and accurate when column labels are meaningful.
- `InstanceBasedSchemaMatcher`: compares the distributions of values per column via TF/TF‑IDF/binary vectors and cosine/Jaccard/containment similarity. Better suited than LabelBasedMatcher if column labels are ambigous.
- `DuplicateBasedSchemaMatcher`: leverage known record correspondences to infer column alignments from co‑occurring values in columns of the corresponding records. Great when a labeled set of matching records between datasets exists.
- `SchemaMappingEvaluator`: offers methods for evaluating a generated schema mapping given a labeled set of schema correspondences.

Example matching
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