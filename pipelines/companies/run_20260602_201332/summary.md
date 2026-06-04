# Best-of-breed pipeline run — companies

Total runtime: 646.4 s

## Per-stage winners

| Stage | Winner | Metric | Val score | Test score | Runtime (s) |
|---|---|---|---|---|---|
| sm | `llm_openai` | f1 | 1.0000 | 1.0000 | 16.1 |
| norm | `passthrough` | macro_f1 | 0.7860 | 0.7860 | 2.1 |
| em_blocking | `bm25_blocker` | pair_completeness (>=0.97 floor; reduction_ratio tiebreak) | 0.0000 | 0.0000 | 594.9 |
| em_matching | `magellan` | f1 | 0.7493 | 0.9019 | 594.9 |
| refinement | `baseline` | f1 | 0.9058 | 0.9019 | 0.1 |
| fusion | `prefer_higher_trust_only` | macro_accuracy | 0.5383 | 0.4618 | 31.6 |

## End-to-end metric panel


## Caveats

- Greedy per-stage selection is locally optimal; no joint search across stages. See `plans/plan_best_of_breed_pipeline.md` §8.2.
