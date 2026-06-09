# Best-of-breed pipeline run — papers

Total runtime: 5585.8 s

## Per-stage winners

| Stage | Winner | Metric | Val score | Test score | Runtime (s) |
|---|---|---|---|---|---|
| sm | `llm_openai` | f1 | 0.9714 | 0.9714 | 143.6 |
| norm | `passthrough` | macro_f1 | 0.8257 | 0.8257 | 9.0 |
| em_blocking | `sc_block` | pair_completeness (>=0.97 floor; reduction_ratio tiebreak) | 1.0000 | 1.0000 | 4935.9 |
| em_matching | `ditto_plm` | f1 | 0.0000 | 0.9970 | 4935.9 |
| refinement | `baseline` | f1 | 1.0000 | 0.9970 | 12.1 |
| fusion | `accusim_only` | macro_accuracy | 0.5990 | 0.6104 | 481.8 |

## Caveats

- Greedy per-stage selection is locally optimal; no joint search across stages. See `plans/plan_best_of_breed_pipeline.md` §8.2.
