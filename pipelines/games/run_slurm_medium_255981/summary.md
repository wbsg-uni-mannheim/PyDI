# Best-of-breed pipeline run — games

Total runtime: 3689.3 s

## Per-stage winners

| Stage | Winner | Metric | Val score | Test score | Runtime (s) |
|---|---|---|---|---|---|
| sm | `llm_openai` | f1 | 0.9811 | 0.9811 | 24.4 |
| norm | `passthrough` | macro_f1 | 0.9442 | 0.9442 | 0.6 |
| em_blocking | `standard_blocker` | pair_completeness (>=0.97 floor; reduction_ratio tiebreak) | 0.9667 | 0.9667 | 1741.0 |
| em_matching | `magellan` | f1 | 0.6969 | 0.6969 | 1741.0 |
| refinement | `baseline` | f1 | 0.0000 | 0.6910 | 0.7 |
| fusion | `prefer_higher_trust_only` | macro_accuracy | 0.6428 | 0.6358 | 1895.0 |

## End-to-end metric panel


## Panel warnings

- Source-attribution and synthesis-rate metrics skipped against gold (gold.cell_provenance is None).

## Caveats

- Greedy per-stage selection is locally optimal; no joint search across stages. See `plans/plan_best_of_breed_pipeline.md` §8.2.
