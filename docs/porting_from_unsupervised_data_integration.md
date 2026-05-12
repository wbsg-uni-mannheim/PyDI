# Porting Tracker: unsupervised-data-integration to PyDI

This tracks changes reviewed for migration from
`/Users/aaronsteiner/Documents/GitHub/unsupervised-data-integration` into the
core PyDI package.

## Safe to Move

- [x] Embedding blocker hardening
  - Sanitize non-finite embedding values before indexing/querying.
  - Avoid nondeterministic random vectors for zero-norm embeddings.
  - Suppress only expected sklearn numerical warnings after sanitization.
- [x] Entity matching evaluator performance cleanup
  - Replace `iterrows()` pair-set creation in hot paths with vectorized/zip helpers.
  - Allow blocking evaluation with explicit `total_possible_pairs`.
- [x] Schema matcher prompt and sampling improvements
  - Add column value summaries for placeholder/generic column names.
  - Select sample rows that expose sparse columns.
  - Preserve existing PyDI metadata and target schema override support.
  - Return an empty result with expected columns when no correspondences are found.
- [x] Taxonomy normalization support
  - Add taxonomy fields to `ColumnSpec`.
  - Add JSON Schema extension parsing.
  - Add taxonomy loader/mapper/cache helpers.
  - Wire taxonomy mappings into DataFrame normalization and schema translation.
- [x] Fusion evaluation ID alignment
  - Align expected records through fused `_fusion_sources` and optional `source_ids`.

## Needs Design Before Moving

- [ ] LLM entity matcher behavior
  - Preserve the current `score`/`notes` correspondence contract by default.
  - Add boolean match/difficulty output as compatible opt-in behavior.
- [x] Feature extraction / ML matcher speedups
  - Move cached lookup optimization.
  - Do not expose unused `n_jobs` or import unused parallel tooling.
- [ ] Optional `jellyfish` similarity acceleration
  - Add as an optional dependency and test fallback behavior first.

## Intentionally Not Moving

- [ ] Blocker spec registry (`to_spec`, `from_spec`, `blocker_from_spec`)
- [ ] Breaking fusion return signature changes
- [ ] Nondeterministic embedding random-vector fallback
- [ ] Wholesale schema matcher replacement that removes metadata support
