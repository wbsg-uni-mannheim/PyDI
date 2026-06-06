# Best-of-breed pipeline run — games

Total runtime: 6661.7 s

## Per-stage winners

| Stage | Winner | Metric | Val score | Test score | Runtime (s) |
|---|---|---|---|---|---|
| sm | `llm_openai` | f1 | 0.9811 | 0.9811 | 25.4 |
| norm | `passthrough` | macro_f1 | 0.8576 | 0.8576 | 0.6 |
| em_blocking | `embedding_blocker` | pair_completeness (>=0.97 floor; reduction_ratio tiebreak) | 0.9663 | 0.9663 | 5187.2 |
| em_matching | `magellan` | f1 | 0.7923 | 0.7923 | 5187.2 |
| refinement | `baseline` | f1 | 0.0000 | 0.4807 | 0.6 |
| fusion | `ltm_only` | macro_accuracy | 0.5930 | 0.5251 | 1421.9 |

## End-to-end metric panel


## Panel warnings

- Source-attribution and synthesis-rate metrics skipped against gold (gold.cell_provenance is None).

## Caveats

- Greedy per-stage selection is locally optimal; no joint search across stages. See `plans/plan_best_of_breed_pipeline.md` §8.2.
