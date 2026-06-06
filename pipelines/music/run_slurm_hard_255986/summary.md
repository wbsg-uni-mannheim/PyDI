# Best-of-breed pipeline run — music

Total runtime: 3081.1 s

## Per-stage winners

| Stage | Winner | Metric | Val score | Test score | Runtime (s) |
|---|---|---|---|---|---|
| sm | `llm_openai` | f1 | 0.9200 | 0.9200 | 65.4 |
| norm | `rule_per_attribute_optimal` | macro_f1 | 0.8591 | 0.8591 | 0.9 |
| em_blocking | `sc_block` | pair_completeness (>=0.97 floor; reduction_ratio tiebreak) | 0.9963 | 0.9963 | 2661.9 |
| em_matching | `ditto_plm` | f1 | 0.9433 | 0.9433 | 2661.9 |
| refinement | `baseline` | f1 | 0.6213 | 0.6209 | 0.7 |
| fusion | `accusim_only` | macro_accuracy | 0.6441 | 0.5453 | 337.9 |

## End-to-end metric panel


## Caveats

- Greedy per-stage selection is locally optimal; no joint search across stages. See `plans/plan_best_of_breed_pipeline.md` §8.2.
