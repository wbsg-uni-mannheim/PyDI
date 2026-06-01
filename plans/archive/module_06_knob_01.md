# Module 6: Knob 1 — Surface Augmentation Intensity

## Purpose

Value paraphrase — abbreviations, synonymy, token reordering, reformulation. Distinguishes from Knob 6 (noise/errors) in that these are **legitimate variants** (both forms are correct). Creates the shared `llm_cache.py` used by Knobs 2 and 4. Exports `paraphrase_value_for_knob_04` callable for K4's fallback path.

**Split across 2 sessions:** Session 1 = easy/medium deterministic path + shared infrastructure (`llm_cache.py`). Session 2 = hard LLM paraphrase path + committee integration.

## Spec References

- **Knob card:** [knobs/knob_01_surface_augmentation.md](../../knobs/knob_01_surface_augmentation.md) — full specification including paraphrase_rate per level, operator catalogue, per-attribute-class routing (short/categorical/long), stacking rules at hard, anchor-survivor floor, hard-negative mining from paraphrased non-matches, per-domain baselines
- **Cross-cutting provenance:** [knobs/cross_cutting.md](../../knobs/cross_cutting.md) § "Per-value provenance" — `transform_fn ∈ {eda_substitute, eda_delete, eda_swap, llm_paraphrase, abbreviate, normalize_to_canonical}`
- **Cell-collision coordination:** [knobs/cross_cutting.md](../../knobs/cross_cutting.md) § "Cell-collision coordination" — K1: unconditional skip on any earlier row for same `(entity_id, source, attribute)`
- **Committee fix-on-collapse:** [knobs/cross_cutting.md](../../knobs/cross_cutting.md) § "Committee-validated augmentation" — K1: replace/extend gold to accepted set on collapse
- **LLM hygiene:** [knobs/cross_cutting.md](../../knobs/cross_cutting.md) — pinned `model_id`, `temperature=0`, cached + committed, committee gating, contamination spot-checks

## Key Mechanism (from knob card § "Algorithm selection")

**Per-level behavior:**
- **Easy (~10% cells):** Normalize-to-canonical. Map surface variants back to a single canonical form. Operators: abbreviation expansion, case normalization
- **Medium (~30% cells):** Deterministic EDA operators: `eda_random_swap` (reorder tokens), `eda_random_delete` (drop optional tokens), `abbreviate` (full form → abbreviation from authored table)
- **Hard (~50% cells):** Stacked operators: abbreviation + synonym + LLM reformulation simultaneously. LLM generates near-twin paraphrases seeded from existing matched pairs

**Anchor-survivor floor:** For every fusion-gold entity, ≥1 source retains the non-paraphrased primary value.

**Hard-negative mining:** Paraphrased existing non-matches become synthetic hard negatives for EM training/test sets.

**LLM paraphrase at hard (Tier C):**
- Prompt template: domain-specific, per-attribute-class
- `temperature=0`, pinned model
- Cache key: `sha256(source|attribute|original_value|prompt_version|model_id)`
- Committee validation: accept if semantically equivalent (downstream EM still matches), reject otherwise
- Contamination spot-check: 8-gram overlap with known databases

## Files to Create

| File | Responsibility |
|---|---|
| `usecases_synthetic/scripts/apply_knob_01_surface.py` | CLI: `--domain`, `--level`. Routes by attribute class and level, handles collision avoidance, anchor-survivor floor, writes provenance |
| `usecases_synthetic/lib/surface_operators.py` | `normalize_to_canonical(value, sibling_values)`, `abbreviate(value, abbrev_table)`, `eda_random_swap(value, rng, n_swaps)`, `eda_random_delete(value, rng, n_deletes)`, `llm_paraphrase(value, attribute_class, cache, prompt_template)`. Each returns `(new_value, params_dict)`. Exports `paraphrase_value_for_knob_04(value, rng, ...) -> str` |
| `usecases_synthetic/lib/llm_cache.py` | **Shared by K1/K2/K4.** `LLMCache(cache_dir, prompt_version, model_id)`: `.get(cell_hash)`, `.put(cell_hash, result)`, `.call_or_cache(inputs, api_fn)`. Cache key = `sha256(source\|attribute\|value\|prompt_version\|model_id)`. File-based JSON cache, committed to repo |
| `usecases_synthetic/config/knob_01_surface/companies.yaml` | Per-attribute-class paraphrase rates per level, abbreviation tables, EDA parameters, stopword lists |
| `usecases_synthetic/config/knob_01_surface/_prompts/prompt_short_v1.txt` | LLM prompt for short-field paraphrase |
| `usecases_synthetic/config/knob_01_surface/_prompts/prompt_categorical_v1.txt` | LLM prompt for categorical paraphrase |
| `usecases_synthetic/tests/test_knob_01.py` | normalize_to_canonical, abbreviation table lookup, EDA operators on 100 values, LLM cache hit/miss (mock API), anchor-survivor floor, exported `paraphrase_value_for_knob_04`, collision index skip |

## Acceptance Criteria

1. At easy, all outputs are canonical forms (consistent, normalized)
2. At medium, outputs show abbreviation/reordering/deletion but token overlap ≥50% with original
3. At hard, LLM paraphrases cached and never re-queried on rerun
4. Anchor-survivor floor: ≥1 source retains non-paraphrased primary per fusion-gold entity
5. `paraphrase_value_for_knob_04` callable produces deterministic output given same RNG state
6. Collision index: cells with prior K4 provenance are skipped
7. `pytest usecases_synthetic/tests/test_knob_01.py -v` passes

## Dependencies

Module 0 (provenance, collision index, RNG, loaders).
