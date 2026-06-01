# S1 Augmented Use Cases — Implementation Plan

Top-level tracker. Each module has its own sub-plan with full references to knob specs and cross-cutting policies. Prototype domain: **companies**, then scale to games/music.

## Process

- Git commits, branching, and version control are handled by the human. The implementer (Claude) focuses on writing code, tests, and config files — does not commit or push.
- All commands use the `pydi-dev/` venv: `pydi-dev/bin/pytest`, `pydi-dev/bin/python`, etc. Never use bare `python` or `pytest`.

## Key References (read before any module)

| Document | Purpose |
|---|---|
| [knobs/README.md](../../knobs/README.md) | Knob index, canonical application order, dimension back-mapping |
| [knobs/cross_cutting.md](../../knobs/cross_cutting.md) | Provenance schema, committee mechanism, protection-set semantics, fix-on-collapse, profile model |
| [difficulty_dimensions.md](../../difficulty_dimensions.md) | Difficulty dimensions per pipeline stage |
| [usecases_synthetic/PIPELINE.md](../../usecases_synthetic/PIPELINE.md) | Pipeline runbook (Phase 0 done, Phases 1-4 todo) |
| [plan_algorithmselection.md](../../plan_algorithmselection.md) | Tier classification, cross-knob audit |
| [plan_s1_scale.md](plan_s1_scale.md) | **Follow-on plan:** scale-out prerequisites (games, music, movies, products) — what must happen before S1 variants can be generated beyond companies |

## S1 Canonical Knob Order

```
K2 (niche) → K4 (coverage) → K1/K5/K6 (joint value perturbations) → K3 (drop) → K10 (reliability) → K8 (naming)
```

## Progress Tracker

| # | Module | Status | Sub-plan | Sessions |
|---|--------|--------|----------|----------|
| 0 | Core Infrastructure | `[x]` done | [module_00_infrastructure.md](module_00_infrastructure.md) | 1 |
| 1 | K8 Schema Naming (Tier A) | `[x]` done | [module_01_knob_08.md](module_01_knob_08.md) | 1 |
| 2 | K3 Attribute Drop (Tier A) | `[x]` done | [module_02_knob_03.md](module_02_knob_03.md) | 1 |
| 3 | K10 Source Reliability (Tier A) | `[x]` done | [module_03_knob_10.md](module_03_knob_10.md) | 1 |
| 4 | K6 Value Noise (Tier B) | `[x]` done | [module_04_knob_06.md](module_04_knob_06.md) | 1 |
| 5 | K5 Format/Unit (Tier B) | `[x]` done | [module_05_knob_05.md](module_05_knob_05.md) | 1 |
| 6 | K1 Surface Augmentation (Tier B/C) | `[x]` done | [module_06_knob_01.md](module_06_knob_01.md) | 2 |
| 7 | Joint Value Perturbation (K1+K5+K6) | `[x]` done | [module_07_joint_values.md](module_07_joint_values.md) | 1 |
| 8 | K4 Coverage Skew (Tier A/C) | `[x]` done | [module_08_knob_04.md](module_08_knob_04.md) | 1.5 |
| 9 | K2 Niche Density (Tier B/C) | `[x]` done | [module_09_knob_02.md](module_09_knob_02.md) | 1 |
| 10 | Orchestrator + Packaging | `[x]` done | [module_10_orchestrator.md](module_10_orchestrator.md) | 1 |

**Total: ~13-15 sessions. Companies first, then games/music configs (~2 more).**

## Dependency Graph

```
                    Module 0 (Infrastructure)
                    ┌──┬──┬──┬──┬──┐
                    │  │  │  │  │  │
                    v  v  v  v  v  v
Wave 1 (parallel):  M1 M2 M3 M4 M5 M6──────┐
                    K8 K3 K10 K6 K5 K1       │
                                  │  │       │
                                  v  v       v
Wave 2:                          Module 7   M8  M9
                                 Joint1/5/6 K4  K2
                                       │    │   │
                                       v    v   v
Wave 3:                           Module 10 (Orchestrator)
```

## Suggested Sequential Order

| Order | Module | Rationale |
|-------|--------|-----------|
| 1 | M0 | Foundation, blocks everything |
| 2 | M1 (K8) | Simplest knob, validates script template |
| 3 | M2 (K3) | Introduces baseline-measure pattern reused by K6/K10 |
| 4 | M5 (K5) | Self-contained format operators |
| 5 | M4 (K6) | Noise operators, reuses baseline_measure |
| 6 | M3 (K10) | Reshuffling, benefits from understanding value knobs |
| 7 | M6 (K1) | Complex; creates llm_cache shared by K2/K4 |
| 8 | M7 | Thin joint orchestrator for K1+K5+K6 |
| 9 | M8 (K4) | Removal + fabrication using K1's exports |
| 10 | M9 (K2) | Most complex; metrics, scoring, interpolation |
| 11 | M10 | Integration + packaging |

## New Directory Structure

```
usecases_synthetic/
  lib/                    # Shared library (new)
  config/                 # Per-knob per-domain YAML configs (new)
    knob_01_surface/
    knob_02_niche/
    knob_03_drop/
    knob_04_coverage/
    knob_05_format/
    knob_06_noise/
    knob_08_naming/
    knob_10_reliability/
  cache/                  # LLM outputs + embeddings (new, committed)
  scripts/                # CLI entry points (existing + new)
    build_pool.py         # (existing, Phase 0)
  tests/                  # Pytest suite (new)
  output/                 # Generated at runtime (not committed)
```

## Verification (after Module 10)

```bash
python usecases_synthetic/scripts/generate_variant.py --domain companies --level easy
python usecases_synthetic/scripts/generate_variant.py --domain companies --level medium
python usecases_synthetic/scripts/generate_variant.py --domain companies --level hard
pytest usecases_synthetic/tests/ -v
```
