# Best-of-breed pipeline run — companies

Total runtime: 343.8 s

## Per-stage winners

| Stage | Winner | Metric | Val score | Test score | Runtime (s) |
|---|---|---|---|---|---|
| sm | `llm_openai` | f1 | 1.0000 | 1.0000 | 28.5 |
| norm | `passthrough` | macro_f1 | 0.9388 | 0.9388 | 0.6 |
| em_blocking | `bm25_blocker` | pair_completeness (>=0.97 floor; reduction_ratio tiebreak) | 0.9596 | 0.9596 | 208.8 |
| em_matching | `magellan` | f1 | 0.6907 | 0.8929 | 208.8 |
| refinement | `baseline` | f1 | 0.9094 | 0.8929 | 0.2 |
| fusion | `prefer_higher_trust_only` | macro_accuracy | 0.5333 | 0.4576 | 100.1 |

## End-to-end metric panel


## Caveats

- Greedy per-stage selection is locally optimal; no joint search across stages. See `plans/plan_best_of_breed_pipeline.md` §8.2.
