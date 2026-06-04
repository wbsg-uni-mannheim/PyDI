# Best-of-breed pipeline run — products

Total runtime: 294.8 s

## Per-stage winners

| Stage | Winner | Metric | Val score | Test score | Runtime (s) |
|---|---|---|---|---|---|
| sm | `llm_openai` | f1 | 0.9811 | 0.9811 | 73.6 |
| norm | `passthrough` | macro_f1 | 0.5902 | 0.5902 | 2.2 |
| em_blocking | `embedding_blocker` | pair_completeness (>=0.97 floor; reduction_ratio tiebreak) | 0.0000 | 0.0000 | 40.2 |
| em_matching | `ditto_plm` | f1 | 0.0000 | 0.7987 | 40.2 |
| refinement | `baseline` | f1 | 0.8326 | 0.7987 | 0.2 |
| fusion | `pydi_per_attribute_optimal` | macro_accuracy | 0.5418 | 0.5587 | 170.7 |

## End-to-end metric panel


## Panel warnings

- Source-attribution and synthesis-rate metrics skipped against gold (gold.cell_provenance is None).

## Caveats

- Greedy per-stage selection is locally optimal; no joint search across stages. See `plans/plan_best_of_breed_pipeline.md` §8.2.
- **Norm selection was vacuous** (spread=0.0000 < epsilon). Norm members produced near-identical outputs on this input.
