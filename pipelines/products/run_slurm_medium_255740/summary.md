# Best-of-breed pipeline run — products

Total runtime: 3082.1 s

## Per-stage winners

| Stage | Winner | Metric | Val score | Test score | Runtime (s) |
|---|---|---|---|---|---|
| sm | `llm_openai` | f1 | 0.9811 | 0.9811 | 84.9 |
| norm | `rule_per_attribute_optimal` | macro_f1 | 0.5608 | 0.5608 | 3.0 |
| em_blocking | `bm25_blocker` | pair_completeness (>=0.97 floor; reduction_ratio tiebreak) | 0.4867 | 0.4867 | 2625.9 |
| em_matching | `ditto_plm` | f1 | 0.8808 | 0.8808 | 2625.9 |
| refinement | `baseline` | f1 | 0.5959 | 0.5282 | 0.2 |
| fusion | `pydi_per_attribute_optimal` | macro_accuracy | 0.5746 | 0.5927 | 348.1 |

## End-to-end metric panel


## Caveats

- Greedy per-stage selection is locally optimal; no joint search across stages. See `plans/plan_best_of_breed_pipeline.md` §8.2.
