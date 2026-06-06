# Best-of-breed pipeline run — games

Total runtime: 4526.1 s

## Per-stage winners

| Stage | Winner | Metric | Val score | Test score | Runtime (s) |
|---|---|---|---|---|---|
| sm | `llm_openai` | f1 | 1.0000 | 1.0000 | 24.8 |
| norm | `passthrough` | macro_f1 | 0.9809 | 0.9809 | 0.6 |
| em_blocking | `standard_blocker` | pair_completeness (>=0.97 floor; reduction_ratio tiebreak) | 0.9858 | 0.9858 | 1873.3 |
| em_matching | `ditto_plm` | f1 | 0.6449 | 0.6730 | 1873.3 |
| refinement | `baseline` | f1 | 0.0000 | 0.6730 | 0.8 |
| fusion | `accusim_only` | macro_accuracy | 0.6891 | 0.6491 | 2600.3 |

## End-to-end metric panel


## Panel warnings

- Source-attribution and synthesis-rate metrics skipped against gold (gold.cell_provenance is None).

## Caveats

- Greedy per-stage selection is locally optimal; no joint search across stages. See `plans/plan_best_of_breed_pipeline.md` §8.2.
