# Best-of-breed pipeline run — products

Total runtime: 587.1 s

## Per-stage winners

| Stage | Winner | Metric | Val score | Test score | Runtime (s) |
|---|---|---|---|---|---|
| sm | `llm_openai` | f1 | 1.0000 | 1.0000 | 130.9 |
| norm | `passthrough` | macro_f1 | 0.6188 | 0.6188 | 2.4 |
| em_blocking | `sc_block` | pair_completeness (>=0.97 floor; reduction_ratio tiebreak) | 1.0000 | 1.0000 | 52.8 |
| em_matching | `ditto_plm` | f1 | 0.0000 | 0.8409 | 52.8 |
| refinement | `baseline` | f1 | 0.8357 | 0.8409 | 0.2 |
| fusion | `pydi_per_attribute_optimal` | macro_accuracy | 0.5492 | 0.5668 | 383.7 |

## End-to-end metric panel


## Panel warnings

- Source-attribution and synthesis-rate metrics skipped against gold (gold.cell_provenance is None).

## Caveats

- Greedy per-stage selection is locally optimal; no joint search across stages. See `plans/plan_best_of_breed_pipeline.md` §8.2.
- **Norm selection was vacuous** (spread=0.0000 < epsilon). Norm members produced near-identical outputs on this input.
