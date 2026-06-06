# Best-of-breed pipeline run — music

Total runtime: 6434.5 s

## Per-stage winners

| Stage | Winner | Metric | Val score | Test score | Runtime (s) |
|---|---|---|---|---|---|
| sm | `coma_hybrid` | f1 | 1.0000 | 1.0000 | 91.1 |
| norm | `rule_per_attribute_optimal` | macro_f1 | 0.8952 | 0.8952 | 6.1 |
| em_blocking | `bm25_blocker` | pair_completeness (>=0.97 floor; reduction_ratio tiebreak) | 0.9987 | 0.9987 | 4976.5 |
| em_matching | `ditto_plm` | f1 | 0.9486 | 0.9486 | 4976.5 |
| refinement | `baseline` | f1 | 0.9772 | 0.9471 | 1.2 |
| fusion | `truthfinder_only` | macro_accuracy | 0.7106 | 0.8543 | 1340.9 |

## End-to-end metric panel


## Caveats

- Greedy per-stage selection is locally optimal; no joint search across stages. See `plans/plan_best_of_breed_pipeline.md` §8.2.
