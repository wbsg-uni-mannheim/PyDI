# Best-of-breed pipeline run — games

Total runtime: 14050.1 s

## Per-stage winners

| Stage | Winner | Metric | Val score | Test score | Runtime (s) |
|---|---|---|---|---|---|
| sm | `llm_openai` | f1 | 0.9811 | 0.9811 | 23.9 |
| norm | `passthrough` | macro_f1 | 0.8576 | 0.8576 | 0.6 |
| em_blocking | `standard_blocker` | pair_completeness (>=0.97 floor; reduction_ratio tiebreak) | 0.5969 | 0.5969 | 12629.5 |
| em_matching | `magellan` | f1 | 0.7708 | 0.7708 | 12629.5 |
| refinement | `baseline` | f1 | 0.0000 | 0.4628 | 0.6 |
| fusion | `ltm_only` | macro_accuracy | 0.5856 | 0.5212 | 1368.8 |

## End-to-end metric panel


## Panel warnings

- Source-attribution and synthesis-rate metrics skipped against gold (gold.cell_provenance is None).

## Caveats

- Greedy per-stage selection is locally optimal; no joint search across stages. See `plans/plan_best_of_breed_pipeline.md` §8.2.
