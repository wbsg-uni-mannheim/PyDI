# Best-of-breed pipeline run — games

Total runtime: 1927.8 s

## Per-stage winners

| Stage | Winner | Metric | Val score | Test score | Runtime (s) |
|---|---|---|---|---|---|
| sm | `llm_openai` | f1 | 0.9811 | 0.9811 | 15.9 |
| norm | `passthrough` | macro_f1 | 0.9526 | 0.9526 | 10.8 |
| em_blocking | `standard_blocker` | pair_completeness (>=0.97 floor; reduction_ratio tiebreak) | 0.0000 | 0.0000 | 1055.2 |
| em_matching | `ditto_plm` | f1 | 0.5829 | 0.6443 | 1055.2 |
| refinement | `baseline` | f1 | 0.0000 | 0.6443 | 0.3 |
| fusion | `accusim_only` | macro_accuracy | 0.6853 | 0.6458 | 836.8 |

## End-to-end metric panel


## Panel warnings

- Source-attribution and synthesis-rate metrics skipped against gold (gold.cell_provenance is None).

## Caveats

- Greedy per-stage selection is locally optimal; no joint search across stages. See `plans/plan_best_of_breed_pipeline.md` §8.2.
