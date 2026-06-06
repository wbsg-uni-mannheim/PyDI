# Best-of-breed pipeline run — music

Total runtime: 2610.2 s

## Per-stage winners

| Stage | Winner | Metric | Val score | Test score | Runtime (s) |
|---|---|---|---|---|---|
| sm | `llm_openai` | f1 | 1.0000 | 1.0000 | 65.1 |
| norm | `rule_per_attribute_optimal` | macro_f1 | 0.8704 | 0.8704 | 1.1 |
| em_blocking | `bm25_blocker` | pair_completeness (>=0.97 floor; reduction_ratio tiebreak) | 0.9951 | 0.9951 | 1694.0 |
| em_matching | `magellan` | f1 | 0.9324 | 0.9324 | 1694.0 |
| refinement | `baseline` | f1 | 0.9532 | 0.9324 | 0.6 |
| fusion | `voting_only` | macro_accuracy | 0.7213 | 0.6511 | 834.2 |

## End-to-end metric panel


## Caveats

- Greedy per-stage selection is locally optimal; no joint search across stages. See `plans/plan_best_of_breed_pipeline.md` §8.2.
