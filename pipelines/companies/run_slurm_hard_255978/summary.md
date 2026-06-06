# Best-of-breed pipeline run — companies

Total runtime: 559.8 s

## Per-stage winners

| Stage | Winner | Metric | Val score | Test score | Runtime (s) |
|---|---|---|---|---|---|
| sm | `magneto_slm_llm` | f1 | 0.9130 | 0.9130 | 33.7 |
| norm | `rule_per_attribute_optimal` | macro_f1 | 0.7340 | 0.7340 | 4.8 |
| em_blocking | `bm25_blocker` | pair_completeness (>=0.97 floor; reduction_ratio tiebreak) | 0.9687 | 0.9687 | 432.6 |
| em_matching | `ditto_plm` | f1 | 0.9452 | 0.9452 | 432.6 |
| refinement | `baseline` | f1 | 0.6092 | 0.6071 | 0.2 |
| fusion | `prefer_higher_trust_only` | macro_accuracy | 0.3455 | 0.3333 | 82.1 |

## End-to-end metric panel


## Caveats

- Greedy per-stage selection is locally optimal; no joint search across stages. See `plans/plan_best_of_breed_pipeline.md` §8.2.
