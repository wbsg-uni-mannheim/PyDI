# Module 8: Knob 4 — Per-entity Source Coverage Skew

## Purpose

Controls the distribution of how many sources cover each entity. Hard = row removal (long-tail, many entities in only 1-2 sources). Easy = fabrication (uniform coverage, entities present in all sources). Medium = identity. Hybrid tier: hard is Tier A (deterministic pandas), easy is Tier C (LLM fabrication with K1 paraphrase fallback).

**Split across 1.5 sessions:** Session 1 = hard removal path (all constraints). Session 2 = easy fabrication (LLM + fallback).

## Spec References

- **Knob card:** [knobs/knob_04_coverage_skew.md](../../knobs/knob_04_coverage_skew.md) — full specification including target coverage histograms per domain per level, constraint system (fusion-gold floor, pool protection, conflict-preserving removal, singleton cap, K2 distractor passthrough), easy fabrication mechanism, `within_source_duplicate_rate`, joint cell-collision with K1/5/6, per-domain notes (companies at hard baseline, music singleton-heavy)
- **Cross-cutting provenance:** [knobs/cross_cutting.md](../../knobs/cross_cutting.md) § "Per-value provenance" — row-level provenance with `transform_fn ∈ {remove_row, fabricate_row}` + `k4_fabricated=True` flag for fabricated rows
- **Protection set:** [knobs/cross_cutting.md](../../knobs/cross_cutting.md) § "Gold standard incompleteness" — both endpoints of `expanded_positives` pairs must remain in distinct sources
- **LLM hygiene:** [knobs/cross_cutting.md](../../knobs/cross_cutting.md) — same hygiene as K1/K2: pinned model, temperature=0, cached + committed
- **K1 paraphrase fallback:** [knobs/knob_01_surface_augmentation.md](../../knobs/knob_01_surface_augmentation.md) — `paraphrase_value_for_knob_04` callable exported by Module 6

## Key Mechanism (from knob card § "Algorithm selection")

**Target coverage histogram:** `H_target[k]` = fraction of entities covered by exactly k sources. Per-domain, per-level. Mandatory — no cross-domain default.

**Hard (removal):**
1. Measure baseline `H_base[k]`
2. Compute removal targets to shift toward `H_target_hard[k]` (long-tail)
3. Select entity-source rows for removal, subject to constraints:
   - **Fusion-gold floor:** ≥1 source per entity
   - **Pool protection:** both endpoints of `expanded_positives` pairs remain in distinct sources
   - **Conflict-preserving removal:** prefer rows with redundantly-agreeing values
   - **Singleton cap:** ≤70% (companies 55%) single-source entities
   - **K2 distractor passthrough:** single-source distractors from K2 never removed
4. Remove selected rows

**Easy (fabrication):**
1. Compute fabrication targets to shift toward `H_target_easy[k]` (uniform)
2. For each entity needing additional source coverage:
   - **Primary path:** LLM generates source-native representation seeded from entity's gold-consistent values + style/schema/formatting of target source
   - **Fallback:** Copy sibling source row + apply K1 medium-level paraphrase (`paraphrase_value_for_knob_04`)
3. Mark fabricated rows with `k4_fabricated=True` in provenance

**Medium:** Identity — no rows added or removed.

**Per-domain notes:**
- Companies: at hard baseline already; easy fabrication expensive → use `paraphrase_only` mode (no LLM)
- Games: long DBpedia tail; natural easy headroom
- Music: singleton-heavy by design; easy requires aggressive fabrication

## Files to Create

| File | Responsibility |
|---|---|
| `usecases_synthetic/scripts/apply_knob_04_coverage.py` | CLI: `--domain`, `--level`. Measures coverage histogram, applies removal or fabrication, enforces constraints, writes provenance |
| `usecases_synthetic/lib/coverage_ops.py` | `measure_coverage_histogram(dfs, id_column) -> dict[int, float]`, `select_removals(entity_source_matrix, target_hist, constraints, rng) -> list[(entity, source)]`, `fabricate_row(entity, target_source, sibling_sources, llm_cache, paraphrase_fn, rng) -> Series` |
| `usecases_synthetic/config/knob_04_coverage/companies.yaml` | `H_target` per level, `singleton_cap_hard`, `within_source_duplicate_rate`, `paraphrase_only` flag (companies) |
| `usecases_synthetic/tests/test_knob_04.py` | Hard constraints (all 5), stochastic dominance CDF check, fabrication fallback when LLM unavailable, K2 distractor passthrough, `k4_fabricated=True` in provenance |

## Acceptance Criteria

1. Hard: no entity drops to zero sources (fusion-gold floor)
2. Hard: both endpoints of each pooled-positive pair remain in distinct sources
3. Hard: singleton fraction ≤ cap from YAML
4. Easy: fabricated rows have `k4_fabricated=True` in provenance
5. Easy: stochastic dominance `sum(H_easy[k] for k≤j) ≤ sum(H_medium[k] for k≤j) ≤ sum(H_hard[k] for k≤j)` ∀j
6. Medium: identity (zero rows added or removed)
7. `pytest usecases_synthetic/tests/test_knob_04.py -v` passes

## Dependencies

Module 0 (domain config, provenance, RNG, protection set, loaders). Module 6 (exports `paraphrase_value_for_knob_04` + `llm_cache`).

## Follow-ups (post-S1, 2026-04-17)

Landed during S1 validation on `companies-small`. See [plans/validation/plan_s1_validation.md](../validation/plan_s1_validation.md#pre-ablation-fixes-applied-on-companies-small-2026-04-17).

1. **Orchestrator audit now tolerates null-histogram levels.** The `generate_variant --level all` monotonicity audit previously compared K4 row counts across levels and false-triggered whenever a domain YAML declared `target_coverage_histogram: {level: null}` (null = identity by design — no rows added or removed at that level). The audit now consults the knob config and excludes null-histogram levels from the non-decreasing check, reporting `identity-at: <levels>` in the detail column. Implementation: `_k4_histogram_is_null` helper in [scripts/generate_variant.py](../../usecases_synthetic/scripts/generate_variant.py).

2. **Known pre-existing test failures (not fixed).** Seven tests in [usecases_synthetic/tests/test_knob_04.py](../../usecases_synthetic/tests/test_knob_04.py) (`TestHistogram::test_measure_baseline`, `TestMediumIdentity`, `TestHardRemoval::*`, `TestEasyFabrication::*`, `TestStochasticDominance::test_cdf_dominance`) fail because `measure_coverage_histogram` was changed to exclude synthetic distractor singletons by default ([lib/coverage_ops.py:146-171](../../usecases_synthetic/lib/coverage_ops.py#L146-L171)) and the tests build `EntityView` directly from synthetic linkage groups where the default excludes everything. Fix: pass `include_distractor_singletons=True` in the test call sites. Not done yet — orthogonal to the S1 validation path.
