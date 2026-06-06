# Best-of-breed pipeline run — companies

Total runtime: 375.5 s

## Per-stage winners

| Stage | Winner | Metric | Val score | Test score | Runtime (s) |
|---|---|---|---|---|---|
| sm | `llm_openai` | f1 | 1.0000 | 1.0000 | 21.2 |
| norm | `rule_per_attribute_optimal` | macro_f1 | 0.7340 | 0.7340 | 0.7 |
| em_blocking | `sc_block` | pair_completeness (>=0.97 floor; reduction_ratio tiebreak) | 0.5347 | 0.5347 | 266.9 |
| em_matching | `ditto_plm` | f1 | 0.9415 | 0.9415 | 266.9 |
| refinement | `baseline` | f1 | 0.6051 | 0.6037 | 0.1 |
| fusion | `prefer_higher_trust_only` | macro_accuracy | 0.3455 | 0.3364 | 80.0 |

## End-to-end metric panel


## Caveats

- Greedy per-stage selection is locally optimal; no joint search across stages. See `plans/plan_best_of_breed_pipeline.md` §8.2.
