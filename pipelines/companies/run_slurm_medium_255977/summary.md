# Best-of-breed pipeline run — companies

Total runtime: 390.5 s

## Per-stage winners

| Stage | Winner | Metric | Val score | Test score | Runtime (s) |
|---|---|---|---|---|---|
| sm | `llm_openai` | f1 | 1.0000 | 1.0000 | 33.7 |
| norm | `passthrough` | macro_f1 | 0.7152 | 0.7152 | 4.7 |
| em_blocking | `bm25_blocker` | pair_completeness (>=0.97 floor; reduction_ratio tiebreak) | 0.9574 | 0.9574 | 237.0 |
| em_matching | `ditto_plm` | f1 | 0.8887 | 0.8887 | 237.0 |
| refinement | `baseline` | f1 | 0.9191 | 0.8887 | 0.2 |
| fusion | `prefer_higher_trust_only` | macro_accuracy | 0.4500 | 0.3958 | 108.2 |

## End-to-end metric panel


## Caveats

- Greedy per-stage selection is locally optimal; no joint search across stages. See `plans/plan_best_of_breed_pipeline.md` §8.2.
