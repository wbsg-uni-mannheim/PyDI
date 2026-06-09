# Best-of-breed pipeline run — papers

Total runtime: 4878.0 s

## Per-stage winners

| Stage | Winner | Metric | Val score | Test score | Runtime (s) |
|---|---|---|---|---|---|
| sm | `coma_hybrid` | f1 | 0.9286 | 0.9286 | 97.8 |
| norm | `passthrough` | macro_f1 | 0.6466 | 0.6466 | 7.3 |
| em_blocking | `sc_block` | pair_completeness (>=0.97 floor; reduction_ratio tiebreak) | 0.9985 | 0.9985 | 4326.7 |
| em_matching | `magellan` | f1 | 0.9945 | 0.9945 | 4326.7 |
| refinement | `baseline` | f1 | 0.9924 | 0.9898 | 10.8 |
| fusion | `prefer_higher_trust_only` | macro_accuracy | 0.5551 | 0.5807 | 433.2 |

## Caveats

- Greedy per-stage selection is locally optimal; no joint search across stages. See `plans/plan_best_of_breed_pipeline.md` §8.2.
