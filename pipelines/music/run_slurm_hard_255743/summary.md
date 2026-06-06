# Best-of-breed pipeline run — music

Total runtime: 3109.6 s

## Per-stage winners

| Stage | Winner | Metric | Val score | Test score | Runtime (s) |
|---|---|---|---|---|---|
| sm | `llm_openai` | f1 | 0.9200 | 0.9200 | 45.1 |
| norm | `rule_per_attribute_optimal` | macro_f1 | 0.8591 | 0.8591 | 0.9 |
| em_blocking | `sc_block` | pair_completeness (>=0.97 floor; reduction_ratio tiebreak) | 0.4862 | 0.4862 | 2712.1 |
| em_matching | `ditto_plm` | f1 | 0.9433 | 0.9433 | 2712.1 |
| refinement | `baseline` | f1 | 0.6213 | 0.6209 | 0.7 |
| fusion | `accusim_only` | macro_accuracy | 0.6441 | 0.5453 | 336.3 |

## End-to-end metric panel


## Caveats

- Greedy per-stage selection is locally optimal; no joint search across stages. See `plans/plan_best_of_breed_pipeline.md` §8.2.
