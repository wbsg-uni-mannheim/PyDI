# Best-of-breed pipeline run — companies

Total runtime: 871.1 s

## Per-stage winners

| Stage | Winner | Metric | Val score | Test score | Runtime (s) |
|---|---|---|---|---|---|
| sm | `coma_hybrid` | f1 | 1.0000 | 1.0000 | 67.0 |
| norm | `passthrough` | macro_f1 | 0.6619 | 0.6619 | 3.2 |
| em_blocking | `sc_block` | pair_completeness (>=0.97 floor; reduction_ratio tiebreak) | 0.9752 | 0.9752 | 564.3 |
| em_matching | `magellan` | f1 | 0.9357 | 0.9357 | 564.3 |
| refinement | `baseline` | f1 | 0.9395 | 0.9357 | 0.5 |
| fusion | `pydi_per_attribute_optimal` | macro_accuracy | 0.4200 | 0.4030 | 221.0 |

## End-to-end metric panel


## Caveats

- Greedy per-stage selection is locally optimal; no joint search across stages. See `plans/plan_best_of_breed_pipeline.md` §8.2.
