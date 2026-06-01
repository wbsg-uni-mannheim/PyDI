# Module 10: Orchestrator + Packaging

## Purpose

End-to-end CLI that runs all knobs in canonical order for a given `(domain, level)`, packages the result into the variant directory layout from `plan.md`, and validates cross-level monotonicity. Updates `PIPELINE.md` with actual file paths and status.

## Spec References

- **Canonical knob order (S1):** [knobs/README.md](../../knobs/README.md) § "Canonical knob application order" — `K2 → K4 → K1/5/6 (joint) → K3 → K10 → K8`
- **Output format (S1):** [plan.md](../../plan.md) § "Scenario 1: Augmented use cases" — `usecases/<domain>-augmented/input/{data, schemamatching, entitymatching, fusion}` + `config/difficulty.yaml`
- **Profile model:** [knobs/cross_cutting.md](../../knobs/cross_cutting.md) § "Profile model" — four artifacts per domain (baseline + easy/medium/hard), monotone easy→medium→hard required
- **Provenance consolidation:** [knobs/cross_cutting.md](../../knobs/cross_cutting.md) § "Per-value provenance" — all per-knob provenance CSVs consolidated into variant's `output/provenance/`
- **DataFrame attrs:** [CLAUDE.md](../../CLAUDE.md) — `dataset_name` in `DataFrame.attrs` must be set for provenance tracking
- **Pipeline runbook:** [usecases_synthetic/PIPELINE.md](../../usecases_synthetic/PIPELINE.md) — Phase 2 entries to update

## Variant Directory Layout

```
usecases/<domain>-augmented/
  input/
    data/                — original source datasets + augmented records
    schemamatching/      — original mappings (preserved) + renamed headers (K8)
    entitymatching/      — original correspondences + regenerated test set (K2)
    fusion/              — original gold standard (unchanged)
  output/
    provenance/          — consolidated per-knob provenance CSVs
  config/
    difficulty.yaml      — all knob parameters + seeds used for this variant
```

## Files to Create

| File | Responsibility |
|---|---|
| `usecases_synthetic/scripts/generate_variant.py` | Master CLI: `--domain companies --level hard [--seed 42]`. Calls knob scripts in canonical order: K2→K4→joint(K1/K5/K6)→K3→K10→K8. Validates inputs, manages intermediate state directory, consolidates provenance, writes `difficulty.yaml` |
| `usecases_synthetic/scripts/package_variant.py` | Assembles final variant directory from intermediate state. Copies augmented data, SM mappings, EM correspondences, fusion gold. Sets `DataFrame.attrs["dataset_name"]`. Creates empty `output/` (populated by running PyDI pipeline) |
| `usecases_synthetic/tests/test_generate_variant.py` | Integration test: run all 3 levels for companies (can use small subset), verify directory structure, provenance completeness, cross-level monotonicity |

## difficulty.yaml Structure

```yaml
domain: companies
level: hard
master_seed: 42
generated_at: "2026-04-10T..."
knobs:
  knob_02:
    corner_case_ratio: 0.7
    seed: <derived>
  knob_04:
    singleton_cap: 0.55
    seed: <derived>
  knob_01:
    paraphrase_rate: 0.5
    seed: <derived>
  # ... all active knobs
```

## Cross-Level Monotonicity Checks

The orchestrator validates these after generating all 3 levels:
1. **K3 drop nesting:** `D_easy ⊆ D_medium ⊆ D_hard` (via shared per-cell uniforms)
2. **K4 coverage:** stochastic dominance of coverage histograms
3. **K10 concentration:** per-attribute gold-carrier concentration decreasing easy→hard
4. **K2 corner-case ratio:** increasing easy→hard
5. **K8 naming distance:** increasing easy→hard (descriptive→anonymized)
6. **K5 format count:** increasing easy→hard
7. **K6 noise rate:** increasing easy→hard

## Acceptance Criteria

1. `generate_variant.py --domain companies --level easy` produces a valid variant directory
2. All 3 levels produce complete variants with all expected artifacts
3. Cross-level monotonicity holds for all 7 checks above
4. `difficulty.yaml` contains all knob parameters and seeds
5. Provenance CSVs consolidated under `output/provenance/` with no gaps
6. `PIPELINE.md` Phase 2 entries updated to reflect actual implementation
7. `pytest usecases_synthetic/tests/test_generate_variant.py -v` passes

## Dependencies

All preceding modules (0-9). This is the final integration module.

## Follow-ups (post-S1, 2026-04-17)

Landed during S1 validation on `companies-small`. See [plans/validation/plan_s1_validation.md](../validation/plan_s1_validation.md#pre-ablation-fixes-applied-on-companies-small-2026-04-17).

1. **Alias-aware strict-cache defaults for K1/K2.** Strict-cache-at-hard was causing K1/K2 on `companies-small` to silently identity-pass because the aliased cache directory (pointing at the `companies` domain's cache) had no entries yet. Fix in [scripts/generate_variant.py](../../usecases_synthetic/scripts/generate_variant.py): when `knob_config_alias` is set on the domain, strict_cache defaults flip to `False` and K1/K2 `LLMCache` instances are always materialized so cache misses populate the shared cache directory. A companion change in [lib/surface_operators.py](../../usecases_synthetic/lib/surface_operators.py) (`llm_paraphrase`) treats `api_client=None` as effective-strict so a miss raises `LLMCacheMiss` (handled gracefully — skip + log) rather than the cache's generic "api_fn required" `RuntimeError`.

2. **Monotonicity audit metric changes.** The cross-level audit documented in "Cross-Level Monotonicity Checks" above has two updates: **K4** now consults `target_coverage_histogram` per level and skips levels declaring `null` (identity by design) from the non-decreasing row-count check, reporting `identity-at: <levels>` in the detail column. **K8** row-count was confounded by upstream K3 drops; it's now replaced by a summed-`rapidfuzz.distance.Levenshtein` distance over provenance rows (column renamed `knob_08_naming_edit_distance`). Helpers: `_k4_histogram_is_null`, `_k8_naming_distance` in [scripts/generate_variant.py](../../usecases_synthetic/scripts/generate_variant.py).

3. **Per-domain knob-config overrides.** The domain YAML now supports a `knob_config_overrides: {knob_NN_<name>: {key: value, ...}}` block that deep-merges into the (aliased) knob config, so downsampled / specialized domains can tune individual knob knobs without forking the entire config. See [lib/domain_config.py](../../usecases_synthetic/lib/domain_config.py) (`load_knob_config`, `_resolve_knob_config_overrides`). Used by [config/domains/companies-small.yaml](../../usecases_synthetic/config/domains/companies-small.yaml) to tighten K3's `single_source_survivor_cap_hard` from 0.25 to 0.15.
