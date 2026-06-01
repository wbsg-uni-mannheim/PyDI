# Module 4: Knob 6 — Value Noise Injection

## Purpose

Cell-level corruption using FEBRL/Christen-Vatsalan operators: typos, OCR confusions, truncations, whitespace/case corruption. These are **errors**, not legitimate variants (that's Knob 1). Participates in the joint cell-collision system with Knobs 1 and 5.

## Spec References

- **Knob card:** [knobs/knob_06_value_noise.md](../../knobs/knob_06_value_noise.md) — full specification including per-attribute-class rates, operator catalogue, compress/identity/stretch transforms, per-entity clean-primary floor, per-cell clean-survivor floor, per-domain baselines, keyboard-adjacency bias, committee rollback rules
- **Cross-cutting provenance:** [knobs/cross_cutting.md](../../knobs/cross_cutting.md) § "Per-value provenance" — row-level provenance with `transform_fn ∈ {typo_substitute, ocr_confuse, truncate, whitespace_corrupt, case_corrupt, cleanup, rollback_for_committee}`
- **Cell-collision coordination:** [knobs/cross_cutting.md](../../knobs/cross_cutting.md) § "Cell-collision coordination" — K6 skips cells touched by K1/K5, **except** cells with `k4_fabricated=True` (those ARE fair game for K6)
- **Committee fix-on-collapse:** [knobs/cross_cutting.md](../../knobs/cross_cutting.md) § "Committee-validated augmentation" — K6: **reject** — typos never promoted to gold. Roll back offending mutations
- **Baseline measurement:** reuses `baseline_measure.py` from Module 2 to measure per-(source, attribute) baseline noise rates

## Key Mechanism (from knob card § "Algorithm selection")

**Per-attribute-class rates:**
| Class | Easy | Medium | Hard |
|---|---|---|---|
| Primary (canonical label) | ~0% | ~0% | ~1-2% |
| Key (blocking-relevant) | ~0% | ~1-2% | ~5-8% |
| Secondary (everything else) | ~1% | ~3-5% | ~10-15% |

**Operators (from Christen-Vatsalan + FEBRL):**
- `typo_substitute(value, rng, n_edits)` — keyboard-adjacency weighted character substitution. 1 edit at medium, 1-3 at hard
- `ocr_confuse(value, rng, n_chars)` — OCR confusion table (`l↔1`, `O↔0`, `rn↔m`, etc.). Single-char at medium, multi-char at hard
- `truncate(value, rng, max_fraction)` — Right-truncation. Hard only
- `whitespace_corrupt(value, rng)` — Extra/missing spaces, tab injection. All levels
- `case_corrupt(value, rng)` — Random case changes. All levels
- `cleanup(value)` — Easy-only: fix existing noise from cleanest source

**Constraints:**
1. Per-entity clean-primary floor: ≥1 source retains noise-free primary
2. Per-cell clean-survivor floor: ≥1 source retains noise-free value per fusion-gold `(entity, attribute)`
3. Soft global primary cap at hard: ~30-40% of entities with noised primary (calibrated per domain)

## Files to Create

| File | Responsibility |
|---|---|
| `usecases_synthetic/scripts/apply_knob_06_noise.py` | CLI: `--domain`, `--level`. Classifies cells by attribute class, draws noise/clean per cell, applies operators, enforces constraints, writes provenance |
| `usecases_synthetic/lib/noise_operators.py` | Individual operator functions, each returning `(new_value, params_dict)` for provenance. QWERTY adjacency table. OCR confusion table. All deterministic given RNG |
| `usecases_synthetic/config/knob_06_noise/companies.yaml` | Rate targets per level per class, operator weights per level, compress/stretch factors, primary cap |
| `usecases_synthetic/config/knob_06_noise/_tables/qwerty.yaml` | Keyboard adjacency map |
| `usecases_synthetic/config/knob_06_noise/_tables/ocr.yaml` | OCR confusion pairs |
| `usecases_synthetic/tests/test_knob_06.py` | Each operator independently, clean-primary floor, clean-survivor floor, collision index integration, K4-fabricated exception, rollback path |

## Acceptance Criteria

1. Each of 5 operators produces visibly corrupted output on 100 test values
2. For every fusion-gold entity, ≥1 source retains clean primary value even at hard
3. Collision index correctly skips cells with prior K1/K5 provenance
4. K4-fabricated cells ARE corrupted (not skipped)
5. Determinism: same seed + same input = identical output
6. `pytest usecases_synthetic/tests/test_knob_06.py -v` passes

## Dependencies

Module 0 (provenance, collision index, RNG). Reuses `baseline_measure.py` from Module 2.
