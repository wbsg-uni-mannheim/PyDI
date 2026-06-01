# S1 Difficulty Validation — Implementation Plan

Top-level tracker for **plan.md Step 7**: verify that the S1 knob settings produce measurable, monotone performance differences in PyDI's pipeline. Covers PIPELINE.md **Phase 1** (baseline measurement) and **Phase 3** (committee validation), plus the prerequisite real end-to-end run of `generate_variant.py` on companies.

**Scope gate — Scenario 2 is blocked on Scenario 1 completion.** Per user direction 2026-04-17, Scenario 2 (plan.md Step 9) does not start until Scenario 1 is fully implemented, validated, and packaged across **every** benchmark domain (plan.md Step 8 — companies, games, music, movies, products). This validation plan is one prerequisite; Step 8 Scale is the other.

Prototype domain: **companies** only. Games, music, movies, and products are deferred to plan.md Step 8 (Scale), since their per-knob domain YAMLs are not populated yet ([usecases_synthetic/PIPELINE.md:90](../../usecases_synthetic/PIPELINE.md#L90)).

**Fast path: `companies-small`.** A downsampled clone of `companies` (~10% of dbpedia rows, all EM/fusion gold preserved) is maintained via [usecases_synthetic/scripts/downsample_domain.py](../../usecases_synthetic/scripts/downsample_domain.py) and aliases companies' per-knob YAMLs via `knob_config_alias`. **Use `companies-small` for M9 (ablation) and for iterating on M7/M8 before running them on full companies** — the full-companies run is the authoritative signal for M10, but the ablation matrix (8 knobs × 3 levels × committee) is only tractable on the small domain. `measure_baseline.py --domain companies-small` is the prerequisite before running `run_ablation_validation.py --domain companies-small`.

## Process

- Git commits, branching, version control handled by the human. The implementer (Claude) focuses on code, tests, config files.
- All commands use the `pydi-dev/` venv: `pydi-dev/bin/pytest`, `pydi-dev/bin/python`, etc. Never bare `python` or `pytest`.
- No emojis in console output. NumPy-style docstrings. mypy strict. Preserve `DataFrame.attrs` ([CLAUDE.md](../../CLAUDE.md)).

## Goal (precise)

After this plan is executed, we must be able to answer — **with evidence** — all four of the following for the companies domain:

1. **Baseline exists.** Per-stage committee metrics on the *original* companies data are measured and stored (SM, EM, Fusion). This is the reference point cross_cutting.md's committee loop subtracts from.
2. **Signals are real.** For each level (easy / medium / hard), committee metrics are measured on the packaged variant, and the level-vs-baseline delta is **monotone** in the direction and stage predicted by each knob card's *Committee expectations* section.
3. **No silent collapses.** Any attribute or stage that collapses to random is flagged, attributed to a specific knob via ablation, and handled via the per-knob fix strategy in [cross_cutting.md §Per-knob fix-strategy defaults](../../knobs/cross_cutting.md#per-knob-fix-strategy-defaults).
4. **Per-knob attribution.** For each of the eight active knobs (K1, K2, K3, K4, K5, K6, K8, K10), the measured delta matches the card's predicted direction and relative strength. This is the empirical check on the algorithm-selection choices made in plan.md Step 5.

Out of scope for this plan: K7 (not built in v1), K9 (S2 only), fully-synthetic Scenario 2 generation (plan.md Step 9 — blocked on Step 8 Scale), Scale to games/music/movies/products (plan.md Step 8), fix-on-collapse that requires modifying S1 knob implementations (goes back to `plans/module_*.md`).

## Key References (read before any module)

| Document | Purpose |
|---|---|
| [../../plan.md](../../plan.md) | Step 7 definition and validation methodology |
| [../../knobs/cross_cutting.md](../../knobs/cross_cutting.md) | Committee mechanism, bootstrap order, fix-on-collapse, pool protection semantics |
| [../../knobs/README.md](../../knobs/README.md) | Canonical knob order, index, dimension back-mapping |
| [../../knobs/ablations.md](../../knobs/ablations.md) | Ablation protocol per knob (each-knob-alone runs) |
| [../../knobs/knob_*.md](../../knobs/) | Per-knob *Committee expectations* sections — empirical predictions to verify |
| [../../usecases_synthetic/PIPELINE.md](../../usecases_synthetic/PIPELINE.md) | Runbook. Phase 1 and Phase 3 are todo; this plan fills them in |
| [../../plan_algorithmselection.md](../../plan_algorithmselection.md) | Algorithm tier classification — informs committee composition |
| [../plan_s1_implementation.md](../plan_s1_implementation.md) | S1 implementation tracker (completed) |
| [../plan_s1_scale.md](../plan_s1_scale.md) | S1 scale-out prerequisites (games, music, movies, products) — tracks what must happen to reach plan.md Step 8 completion |
| [../../tests/companies_test/test_workflow_companies.py](../../tests/companies_test/test_workflow_companies.py) | Reference end-to-end companies pipeline — a starting point for committee composition |

## Output artifacts (what this plan produces)

```
usecases_synthetic/
  baselines/
    companies/
      baseline_metrics.json        # Phase 1 output — per-stage, per-attribute
      baseline_report.md           # human-readable rollup
  validation/
    companies/
      easy/metrics.json            # per-level committee run on packaged variant
      medium/metrics.json
      hard/metrics.json
      monotonicity_report.csv      # per-stage/per-attribute deltas vs baseline
      monotonicity_report.md       # rollup, collapse flags, expected-signal checks
      ablation/                    # each-knob-alone committee runs (K1..K10)
        knob_01/metrics.json
        ...
      final_report.md              # consolidated story: does S1 work? fix-list if not
  config/
    committees/
      sm_committee.yaml            # schema-matching committee roster
      em_committee.yaml            # entity-matching committee roster (+ pool diagnostic)
      fusion_committee.yaml        # fusion committee roster (extends cross_cutting.md draft)
  lib/
    committee.py                   # shared committee runner framework
    committee_sm.py                # SM committee implementation
    committee_em.py                # EM committee implementation
    committee_fusion.py            # Fusion committee implementation
    metrics.py                     # metric helpers (F1, per-attribute accuracy, deltas)
  scripts/
    measure_baseline.py            # Phase 1 entry point
    validate_variant.py            # Phase 3 entry point
    run_ablation_validation.py     # each-knob-alone runner
  tests/
    test_committee_sm.py
    test_committee_em.py
    test_committee_fusion.py
    test_measure_baseline.py
    test_validate_variant.py
    test_monotonicity.py
```

## Progress Tracker

| # | Module | Status | Sub-plan | Sessions |
|---|--------|--------|----------|----------|
| 0 | Core Infrastructure (metrics, runner framework, loaders) | `[x]` done | [module_00_infrastructure.md](module_00_infrastructure.md) | 1 |
| 1 | Committee composition spec + YAML configs | `[x]` done | [module_01_committee_spec.md](module_01_committee_spec.md) | 1 |
| 2 | SM committee runner | `[x]` done | [module_02_sm_committee.md](module_02_sm_committee.md) | 1 |
| 3 | EM committee runner (+ pool diagnostic) | `[x]` done | [module_03_em_committee.md](module_03_em_committee.md) | 1.5 |
| 4 | Fusion committee runner | `[x]` done | [module_04_fusion_committee.md](module_04_fusion_committee.md) | 1 |
| 5 | Phase 1: baseline measurement | `[x]` done | [module_05_baseline.md](module_05_baseline.md) | 1 |
| 6 | Generate real companies variants (exercise orchestrator) | `[x]` done | [module_06_generate_variants.md](module_06_generate_variants.md) | 1 |
| 7 | Phase 3: per-variant validation runner | `[x]` done | [module_07_validate_variant.md](module_07_validate_variant.md) | 1 |
| 8 | Monotonicity + collapse detection | `[x]` done | [module_08_monotonicity.md](module_08_monotonicity.md) | 1 |
| 9 | Per-knob ablation validation (run on `companies-small`) | `[x]` code-done (runner + analyzer + tests; awaiting real-data run on `companies-small`) | [module_09_ablation.md](module_09_ablation.md) | 1.5 |
| 10 | Triage + final report | `[x]` done (companies-small) | [module_10_triage_report.md](module_10_triage_report.md) | 1 |

**Total: ~11-12 sessions for companies.** Games/music are deferred to plan.md Step 9.

## Pre-ablation fixes applied on `companies-small` (2026-04-17)

Structural issues surfaced by M6 (`generate_variant --level all`) on `companies-small` that were fixed **before** the M9 ablation runs proceed. These are companies-small-specific inputs to M9 — they shift the baseline the ablation matrix will be measured against.

| # | Issue | Fix | Code refs |
|---|-------|-----|-----------|
| 1 | K3 `single_source_survivor_cap_hard` (0.25 default) too loose for the downsampled domain → hard variant left too many single-source survivors | Added `knob_config_overrides` mechanism in [domain_config.py:155-215](../../usecases_synthetic/lib/domain_config.py#L155-L215) and set `single_source_survivor_cap_hard: 0.15` in [companies-small.yaml](../../usecases_synthetic/config/domains/companies-small.yaml) | [lib/domain_config.py](../../usecases_synthetic/lib/domain_config.py), [config/domains/companies-small.yaml](../../usecases_synthetic/config/domains/companies-small.yaml) |
| 2 | K3 drop-nesting `D_easy ⊆ D_medium ⊆ D_hard` violated by (a) easy propagate_fill running before mask construction, (b) level-dependent conflict-preserve, (c) hard rollback | Compute masks for all 3 levels, apply constraints at each, then shrink via `_enforce_nesting` at the end. Propagate_fill moved to after nesting | [scripts/apply_knob_03_drop.py](../../usecases_synthetic/scripts/apply_knob_03_drop.py) (`_enforce_nesting` helper + reordered `apply_knob_03` body) |
| 3 | `generate_variant` audits false-triggered on (a) K4 when a level declared a null histogram (null = identity by design), (b) K8 when row counts differed due to upstream K3 drops | K4 audit now skips null-histogram levels and reports `identity-at: <levels>`. K8 audit renamed to `knob_08_naming_edit_distance` and uses summed `rapidfuzz.distance.Levenshtein` over provenance rows | [scripts/generate_variant.py](../../usecases_synthetic/scripts/generate_variant.py) (`_k4_histogram_is_null`, `_k8_naming_distance`) |
| 4 | K1/K2 silently identity-passed on `companies-small` because strict-cache was on at hard and the shared cache (aliased to `companies`) had no entries yet | Alias-aware defaults: when `knob_config_alias` is set, strict_cache flips off for K1/K2 and caches are always materialized so misses populate the shared cache. `surface_operators.llm_paraphrase` treats `api_client=None` as effective-strict so `LLMCacheMiss` fires and K1 degrades gracefully to skip-on-miss | [scripts/generate_variant.py](../../usecases_synthetic/scripts/generate_variant.py) (strict_cache defaulting block), [lib/surface_operators.py:505-520](../../usecases_synthetic/lib/surface_operators.py#L505-L520) |

**Verification status:** unit tests pass for all four areas — K3 (23 tests), K1/K2 (87 tests), generate_variant audits (12 tests). End-to-end regeneration of `companies-small` variants has **not yet been re-run**; that's the prerequisite before M9 proceeds with the real-data ablation sweep.

**Pre-existing unrelated failures noted (not fixed):** 7 tests in [usecases_synthetic/tests/test_knob_04.py](../../usecases_synthetic/tests/test_knob_04.py) (`TestHistogram::test_measure_baseline`, `TestMediumIdentity`, `TestHardRemoval::*`, `TestEasyFabrication::*`, `TestStochasticDominance::test_cdf_dominance`) fail because `measure_coverage_histogram` was changed to exclude distractor singletons by default ([lib/coverage_ops.py:146-171](../../usecases_synthetic/lib/coverage_ops.py#L146-L171)) but the tests were never updated to pass `include_distractor_singletons=True`. Tracked here for future cleanup — orthogonal to this plan.

## Next step when picking up

Run `python usecases_synthetic/scripts/generate_variant.py --domain companies-small --level all` and confirm:
1. Hard variant's `single_source_survivor_cap_hard` audit reports ≤0.15 (the override took effect).
2. `monotonicity_audit.csv` shows `knob_03_drop_nesting` passes at easy+medium.
3. `knob_08_naming_edit_distance` monotonically non-decreasing easy→hard.
4. K1/K2 report real paraphrase/interpolation counts — not all-identity.
5. Shared cache under `pools/companies/llm_cache_*` grows (misses populated it).

Then M9's real-data ablation on `companies-small` is unblocked.

## Dependency Graph

```
                 M0 (infrastructure)
                   │
                   v
                 M1 (committee spec)
                 ┌─┴─┬───┐
                 v   v   v
Wave 1:         M2  M3  M4
                (SM)(EM)(Fus)
                 └─┬─┴───┘
                   v
Wave 2:          M5 (baseline)    ───┐
                                     │
Wave 2':         M6 (real variants) ─┤   (M6 depends only on M0, can run parallel to M1-M5)
                                     │
                                     v
Wave 3:          M7 (validate per variant)
                   │
                   v
Wave 4:          M8 (monotonicity + collapse detection)
                   │
                   v
Wave 5:          M9 (ablation)
                   │
                   v
Wave 6:          M10 (triage + final report)
```

M6 (real variant generation) is structurally independent of M1-M5 and can be done as soon as M0 lands. Recommend running M6 early to surface any orchestrator bugs while the S1 code is fresh — flagged in the plan.md Step 7 rationale.

## Suggested Sequential Order

| Order | Module | Rationale |
|-------|--------|-----------|
| 1 | M0 | Foundation (metrics, runner framework, loaders) |
| 2 | M6 | Early real variant generation — catch orchestrator bugs now |
| 3 | M1 | Committee composition design doc — design decisions |
| 4 | M2 | SM committee (simplest, smallest search space) |
| 5 | M4 | Fusion committee (draft exists in cross_cutting.md) |
| 6 | M3 | EM committee (most complex; benefits from SM/Fusion shape) |
| 7 | M5 | Baseline measurement (runs all three committees on original data) |
| 8 | M7 | Per-variant validation runner (runs all three on easy/med/hard) |
| 9 | M8 | Monotonicity + collapse detection |
| 10 | M9 | Ablation (runs full knob-alone matrix — costly) |
| 11 | M10 | Triage + final report |

## Risk callouts

1. **M6 may uncover orchestrator bugs** that require going back to `plans/module_*.md` (S1 implementation). Budget for this — do not treat M6 as a pure dry run.
2. **SM committee baseline may already be near-ceiling** for companies (the existing SM mappings are hand-authored). Expect the SM delta to live almost entirely on the K8 axis.
3. **Companies has only 3 sources** — EM committee F1 variance will be high. Consider macro-averaging across source pairs and reporting confidence intervals.
4. **Pool-vs-test-gold divergence** is the *expected* signal of cross_cutting.md's committee loop — not a bug. M7 must report both numbers side by side.
5. **Fix-on-collapse loop** is intentionally out of scope as an automatic closed loop. M10 produces a triage document; any actual knob-implementation fixes are separate work that goes back to the S1 modules.

## Open decisions to lock in during M1

- **SM committee roster:** which of `label_based`, `instance_based`, `duplicate_based`, `llm_based` to include. Trade-off: LLM inclusion is validated by knob cards (K8/K9 Committee expectations explicitly call out LLM-matcher spread) but adds cost and non-determinism to every validation run.
- **EM committee roster:** rule-based vs embedding vs learning-based vs LLM. K2/K6/K1 Committee expectations distinguish "lexical vs embedding" — the roster must span that axis.
- **Fusion committee per-attribute routing:** cross_cutting.md drafts a strategy-per-attribute-type map. M1 must pin the exact columns for companies and their attribute classes.
- **Metric normalization:** how to combine per-pair EM F1 across {forbes↔dbpedia, forbes↔fullcontact} into a single summary. Default: macro average, also report per-pair.
- **LLM arbitration toggle:** several Committee expectations sections mention "LLM arbitrate" as the recovery strategy on hard. Is that in-roster or out-of-scope for v1? Default: out of roster (too expensive for every validation run), but available as an opt-in flag on `validate_variant.py`.

These are answered by M1 and become inputs to M2/M3/M4.
