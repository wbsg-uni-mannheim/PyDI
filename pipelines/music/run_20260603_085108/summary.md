# Best-of-breed pipeline run — music

Total runtime: 1343.5 s

## Per-stage winners

| Stage | Winner | Metric | Val score | Test score | Runtime (s) |
|---|---|---|---|---|---|
| sm | `coma_hybrid` | f1 | 1.0000 | 1.0000 | 25.0 |
| norm | `rule_per_attribute_optimal` | macro_f1 | 0.8525 | 0.8525 | 6.5 |
| em_blocking | `bm25_blocker` | pair_completeness (>=0.97 floor; reduction_ratio tiebreak) | 0.0000 | 0.0000 | 932.8 |
| em_matching | `ditto_plm` | f1 | 0.9432 | 0.9560 | 932.8 |
| refinement | `baseline` | f1 | 0.9597 | 0.9560 | 0.2 |
| fusion | `accusim_only` | macro_accuracy | 0.7659 | 0.9017 | 374.7 |

## End-to-end metric panel


## Caveats

- Greedy per-stage selection is locally optimal; no joint search across stages. See `plans/plan_best_of_breed_pipeline.md` §8.2.
