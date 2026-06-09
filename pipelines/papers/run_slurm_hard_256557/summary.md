# Best-of-breed pipeline run — papers

Total runtime: 15872.7 s

## Per-stage winners

| Stage | Winner | Metric | Val score | Test score | Runtime (s) |
|---|---|---|---|---|---|
| sm | `llm_openai` | f1 | 0.9286 | 0.9286 | 60.2 |
| norm | `passthrough` | macro_f1 | 0.5682 | 0.5682 | 3.1 |
| em_blocking | `sc_block` | pair_completeness (>=0.97 floor; reduction_ratio tiebreak) | 0.9499 | 0.9499 | 15457.1 |
| em_matching | `ditto_plm` | f1 | 0.9671 | 0.9671 | 15457.1 |
| refinement | `baseline` | f1 | 0.7028 | 0.6827 | 7.0 |
| fusion | `accusim_only` | macro_accuracy | 0.3768 | 0.3779 | 344.0 |

## Caveats

- Greedy per-stage selection is locally optimal; no joint search across stages. See `plans/plan_best_of_breed_pipeline.md` §8.2.
