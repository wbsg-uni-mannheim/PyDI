# Module 2: Knob 3 — Per-source Attribute Drop Rate

## Purpose

Parametric cell masking (NaN injection) with constraint enforcement. Introduces the "measure baseline → transform" pattern reused by Knobs 6 and 10. Creates the shared `baseline_measure.py` utility.

## Spec References

- **Knob card:** [knobs/knob_03_attribute_drop.md](../../knobs/knob_03_attribute_drop.md) — full specification including per-attribute-class rates, compress/identity/stretch transforms, constraint system, monotone nesting via shared per-cell uniform, per-domain baselines
- **Cross-cutting provenance:** [knobs/cross_cutting.md](../../knobs/cross_cutting.md) § "Per-value provenance" — row-level provenance for drop and propagation-fill operations
- **Profile model:** [knobs/cross_cutting.md](../../knobs/cross_cutting.md) § "Profile model" — easy may require **adding** values (propagation fill from cleanest source), not just removing
- **Protection set:** [knobs/cross_cutting.md](../../knobs/cross_cutting.md) § "Gold standard incompleteness" — fusion-gold entities need ≥1 source surviving per attribute
- **Attribute classes:** defined in per-domain config — `primary` (canonical label), `key` (blocking-relevant), `secondary` (everything else)

## Key Mechanism (from knob card § "Algorithm selection")

**Monotone nesting:** Draw per-cell uniform `u[s, a, e] ~ U(0,1)` **once** (shared across levels). Then:
- `D_easy = {cells where u < T_easy[s,a]}`
- `D_medium = {cells where u < T_medium[s,a]}`
- `D_hard = {cells where u < T_hard[s,a]}`

Since `T_easy ≤ T_medium ≤ T_hard` (enforced by monotonicity validation), this guarantees `D_easy ⊆ D_medium ⊆ D_hard`.

**Per-level transforms:**
- Easy: compress toward min-missingness → `T_easy[s,a] = B[s,a] * compression_factor` (floor at per-class minimum)
- Medium: identity → `T_medium[s,a] = B[s,a]`
- Hard: stretch → `T_hard[s,a] = B[s,a] + (1 - B[s,a]) * stretch_factor` (ceiling per per-class maximum)

Where `B[s,a]` = measured baseline missingness for source s, attribute a.

**Constraints (enforced in this order):**
1. **Fusion survivor floor:** ≥1 source must retain value for every `(entity, attribute)` in fusion gold
2. **Conflict-preserving drop:** when ≥2 sources disagree, preserve ≥2 conflicting values; drops from redundant-agreement pool first
3. **Single-source-survivor cap at hard:** ≤5% (default) of cells collapse to single surviving source
4. **Per-(source, attribute) ceiling:** realized drop ≤ `B[s,a] + delta` (prevents negative headroom)

**Easy propagation fill:** Copy values from the lowest-missingness source to fill gaps in other sources (only for cells that were already missing at baseline).

## Files to Create

| File | Responsibility |
|---|---|
| `usecases_synthetic/scripts/apply_knob_03_drop.py` | CLI: `--domain`, `--level`. Measures baseline, computes targets, draws shared uniforms, masks cells, enforces constraints, writes provenance |
| `usecases_synthetic/lib/baseline_measure.py` | `measure_missingness(dfs: dict[str, DataFrame], attribute_classes: dict[str, str]) -> dict[str, dict[str, float]]` — per-(source, attribute) null fraction. Reused by K6/K10 |
| `usecases_synthetic/config/knob_03_drop/companies.yaml` | `compression_factor`, `stretch_factor`, per-class rate floors/ceilings, `single_source_survivor_cap_hard`, `per_source_attr_delta` |
| `usecases_synthetic/tests/test_knob_03.py` | Monotone nesting, fusion-survivor floor, conflict preservation, single-source cap, propagation fill correctness |

## Acceptance Criteria

1. `D_easy ⊆ D_medium ⊆ D_hard` verified on 100 random entities
2. No fusion-gold `(entity, attribute)` cell has zero surviving sources at any level
3. Single-source-survivor fraction at hard ≤ cap from YAML
4. Easy propagation fill produces values copied from the lowest-missingness source
5. Baseline measured fresh every run (never cached)
6. Provenance emitted for every drop (`transform_fn=drop`) and every fill (`transform_fn=propagate_fill`)
7. `pytest usecases_synthetic/tests/test_knob_03.py -v` passes

## Dependencies

Module 0 (domain config, provenance, RNG, loaders, protection set).

## Follow-ups (post-S1, 2026-04-17)

Landed during S1 validation on `companies-small`. See [plans/validation/plan_s1_validation.md](../validation/plan_s1_validation.md#pre-ablation-fixes-applied-on-companies-small-2026-04-17).

1. **Nesting enforcement is now structural, not probabilistic.** The shared-uniform approach from Acceptance Criterion #1 was violated in practice by three level-dependent effects: (a) easy `propagate_fill` running before mask construction, (b) conflict-preserve selecting different cells at each level, (c) hard-level single-source-survivor rollback. Fixed by computing drop masks for all 3 levels in one pass, running constraints at each, then shrinking `easy`/`medium` against `hard` via a new `_enforce_nesting` helper in [scripts/apply_knob_03_drop.py](../../usecases_synthetic/scripts/apply_knob_03_drop.py). Propagate_fill now runs **after** nesting so filled cells can never be drop-masked.

2. **Per-domain knob config overrides.** The `single_source_survivor_cap_hard` default (0.25) was too loose for downsampled domains. Rather than cloning the full K3 YAML, a new `knob_config_overrides` block in the domain YAML deep-merges onto the aliased knob config — see `load_knob_config` / `_resolve_knob_config_overrides` in [lib/domain_config.py](../../usecases_synthetic/lib/domain_config.py). `companies-small` overrides the cap to 0.15 in [config/domains/companies-small.yaml](../../usecases_synthetic/config/domains/companies-small.yaml).
