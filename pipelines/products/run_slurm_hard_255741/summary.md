# Best-of-breed pipeline run — products

Total runtime: 2089.2 s

## Per-stage winners

| Stage | Winner | Metric | Val score | Test score | Runtime (s) |
|---|---|---|---|---|---|
| sm | `llm_openai` | f1 | 0.9811 | 0.9811 | 79.2 |
| norm | `rule_per_attribute_optimal` | macro_f1 | 0.5446 | 0.5446 | 2.2 |
| em_blocking | `embedding_blocker` | pair_completeness (>=0.97 floor; reduction_ratio tiebreak) | 0.5533 | 0.5533 | 1817.6 |
| em_matching | `ditto_plm` | f1 | 0.8528 | 0.8528 | 1817.6 |
| refinement | `baseline` | f1 | 0.6050 | 0.6395 | 0.1 |
| fusion | `pydi_per_attribute_optimal` | macro_accuracy | 0.4494 | 0.4953 | 173.1 |

## End-to-end metric panel


## Caveats

- Greedy per-stage selection is locally optimal; no joint search across stages. See `plans/plan_best_of_breed_pipeline.md` §8.2.
