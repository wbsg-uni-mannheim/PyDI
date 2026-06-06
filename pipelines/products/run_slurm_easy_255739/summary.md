# Best-of-breed pipeline run — products

Total runtime: 2616.2 s

## Per-stage winners

| Stage | Winner | Metric | Val score | Test score | Runtime (s) |
|---|---|---|---|---|---|
| sm | `duplicate_majority` | f1 | 0.9953 | 0.9953 | 83.4 |
| norm | `rule_per_attribute_optimal` | macro_f1 | 0.6786 | 0.6786 | 3.2 |
| em_blocking | `embedding_blocker` | pair_completeness (>=0.97 floor; reduction_ratio tiebreak) | 0.4800 | 0.4800 | 2159.5 |
| em_matching | `magellan` | f1 | 0.9150 | 0.9150 | 2159.5 |
| refinement | `baseline` | f1 | 0.5420 | 0.4986 | 0.6 |
| fusion | `pydi_per_attribute_optimal` | macro_accuracy | 0.5113 | 0.5165 | 353.3 |

## End-to-end metric panel


## Caveats

- Greedy per-stage selection is locally optimal; no joint search across stages. See `plans/plan_best_of_breed_pipeline.md` §8.2.
