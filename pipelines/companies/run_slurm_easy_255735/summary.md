# Best-of-breed pipeline run — companies

Total runtime: 382.3 s

## Per-stage winners

| Stage | Winner | Metric | Val score | Test score | Runtime (s) |
|---|---|---|---|---|---|
| sm | `coma_hybrid` | f1 | 1.0000 | 1.0000 | 23.6 |
| norm | `passthrough` | macro_f1 | 0.6619 | 0.6619 | 1.0 |
| em_blocking | `sc_block` | pair_completeness (>=0.97 floor; reduction_ratio tiebreak) | 0.9752 | 0.9752 | 265.5 |
| em_matching | `magellan` | f1 | 0.9357 | 0.9357 | 265.5 |
| refinement | `baseline` | f1 | 0.9395 | 0.9357 | 0.1 |
| fusion | `pydi_per_attribute_optimal` | macro_accuracy | 0.4200 | 0.4030 | 85.6 |

## End-to-end metric panel


## Caveats

- Greedy per-stage selection is locally optimal; no joint search across stages. See `plans/plan_best_of_breed_pipeline.md` §8.2.
