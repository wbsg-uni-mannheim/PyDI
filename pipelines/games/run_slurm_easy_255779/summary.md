# Best-of-breed pipeline run — games

Total runtime: 10152.5 s

## Per-stage winners

| Stage | Winner | Metric | Val score | Test score | Runtime (s) |
|---|---|---|---|---|---|
| sm | `coma_hybrid` | f1 | 0.9811 | 0.9811 | 23.8 |
| norm | `passthrough` | macro_f1 | 0.9608 | 0.9608 | 0.6 |
| em_blocking | `standard_blocker` | pair_completeness (>=0.97 floor; reduction_ratio tiebreak) | 0.9729 | 0.9729 | 5914.9 |
| em_matching | `ditto_plm` | f1 | 0.6637 | 0.6637 | 5914.9 |
| refinement | `baseline` | f1 | 0.0000 | 0.6562 | 1.3 |
| fusion | `accusim_only` | macro_accuracy | 0.6678 | 0.6258 | 4185.1 |

## End-to-end metric panel


## Panel warnings

- Source-attribution and synthesis-rate metrics skipped against gold (gold.cell_provenance is None).

## Caveats

- Greedy per-stage selection is locally optimal; no joint search across stages. See `plans/plan_best_of_breed_pipeline.md` §8.2.
