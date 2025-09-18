# Evaluation

Evaluate matching and fusion quality with built‑in utilities. Consistent metrics and threshold sweeps make it easy to benchmark pipelines and regress changes.

Schema Matching
- `PyDI.schemamatching.evaluation.SchemaMappingEvaluator`
  - `evaluate(corr, test_set, threshold=None, by=None) -> dict`
  - `sweep_thresholds(corr, gold, thresholds) -> pd.DataFrame`
- Guidance: ensure the gold mapping has clear positives/negatives; if labels are available, the evaluator can use them directly.

Entity Matching
- `PyDI.entitymatching.evaluation.EntityMatchingEvaluator`
  - Precision/recall/F1, candidate recall, pair reduction
  - Cluster consistency and size distribution, threshold sweeps
- Tips: start with high candidate recall in blocking; tune thresholds for target precision/recall; consider bipartite matching for 1:1 scenarios.

Blocking
- `PyDI.entitymatching.blocking.blocking_evaluation.BlockingEvaluator`
  - Candidate coverage, reduction ratio, block statistics
- Readouts: look for a high coverage (close to 1.0) with a strong reduction ratio; debug top heavy tokens/blocks.

Data Fusion
- `PyDI.fusion.evaluation`
  - `calculate_consistency_metrics(fused_df)` and `calculate_coverage_metrics(...)`
  - `FusionReport` summaries and HTML reports
- Use gold fused data if available to quantify attribute accuracy.

Artifacts
- Metrics as dict/CSV/JSON; HTML reports under each component’s `out_dir`.
