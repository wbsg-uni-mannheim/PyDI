# Best-of-breed pipeline run — papers

Total runtime: 27988.5 s

## Per-stage winners

| Stage | Winner | Metric | Val score | Test score | Runtime (s) |
|---|---|---|---|---|---|
| sm | `llm_openai` | f1 | 0.9048 | 0.9048 | 186.1 |
| norm | `passthrough` | macro_f1 | 0.5981 | 0.5981 | 34.0 |
| em_blocking | `sc_block` | pair_completeness (>=0.97 floor; reduction_ratio tiebreak) | 0.9955 | 0.9955 | 25650.6 |
| em_matching | `magellan` | f1 | 0.9923 | 0.9923 | 25650.6 |
| refinement | `baseline` | f1 | 0.9955 | 0.9923 | 65.8 |
| fusion | `prefer_higher_trust_only` | macro_accuracy | 0.4490 | 0.4766 | 2042.9 |

## Caveats

- Greedy per-stage selection is locally optimal; no joint search across stages. See `plans/plan_best_of_breed_pipeline.md` §8.2.
