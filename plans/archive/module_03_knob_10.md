# Module 3: Knob 10 — Source Reliability Differentiation

## Purpose

Pure permutation reshuffling of which source carries the gold-aligned variant per `(entity, attribute)` cell. No gold mutation — fusion gold is byte-identical before and after. Controls trust ambiguity for the fusion stage.

## Spec References

- **Knob card:** [knobs/knob_10_source_reliability.md](../../knobs/knob_10_source_reliability.md) — full specification including per_attribute_concentration, error_correlation, canonical-form comparator, per-domain notes, S1 reshuffling mechanism, self-contained baseline measurement
- **Cross-cutting provenance:** [knobs/cross_cutting.md](../../knobs/cross_cutting.md) § "Per-value provenance" — row-level provenance for reshuffle operations
- **Knob 5 attribute_classes taxonomy:** [knobs/knob_05_format_unit.md](../../knobs/knob_05_format_unit.md) § "Attribute classes" — the format family taxonomy that drives the canonical-form comparator routing (date → datetime.date, number → Decimal, money → Decimal after FX, etc.)
- **Committee validation:** [knobs/cross_cutting.md](../../knobs/cross_cutting.md) § "Committee-validated augmentation" — K10 fix-on-collapse: no gold change needed (pure permutation)
- **Fusion gold:** `usecases/<domain>/input/fusion/test_set.xml` — the gold-standard fused records used for baseline agreement measurement

## Key Mechanism (from knob card § "Algorithm selection")

**Self-contained baseline:** Measures `B[s, a]` = per-(source, attribute) gold-alignment rate fresh every run using canonical-form equality. Identifies per-attribute winner `W[a]`.

**Canonical-form comparator routing:**
- date → `datetime.date`
- number → `Decimal`
- money → `Decimal` after FX normalization
- duration → `timedelta`
- dimensional → `Decimal` after unit normalization
- string → `casefold + collapse_ws + strip_punct`

**Reshuffling mechanism:**
1. Identify cells where upstream knobs (K1/5/6) produced ≥2 sources with different values
2. For each such cell, the "gold-aligned variant" is the one matching canonical gold
3. **Easy (~90% concentration):** `W[a]` carries gold most of the time → simple per-attribute bias
4. **Medium (~70% concentration, low correlation):** More uniform spread, independent per cell
5. **Hard (~40/35/25 near-uniform, moderate-high correlation):** Draw per-(source, entity) "compromised" mask → correlated errors across attributes within same entity

**All randomness:** One seeded RNG per `(domain, variant, knob=10)`.

## Files to Create

| File | Responsibility |
|---|---|
| `usecases_synthetic/scripts/apply_knob_10_reliability.py` | CLI: `--domain`, `--level`. Measures baseline agreement, identifies reshufflable cells, applies concentration/correlation targets, writes provenance |
| `usecases_synthetic/lib/reliability.py` | `measure_gold_alignment(dfs, fusion_gold, attribute_classes) -> dict[str, dict[str, float]]` — per-(source, attribute) agreement matrix. `identify_reshufflable_cells(dfs, provenance_index) -> DataFrame` — cells where ≥2 sources differ. `reshuffle(cells, rng, concentration, correlation) -> assignment` — the core permutation. Canonical-form comparator dispatch |
| `usecases_synthetic/config/knob_10_reliability/companies.yaml` | `per_attribute_concentration` per level, `error_correlation` per level, per-attribute winner names |
| `usecases_synthetic/tests/test_knob_10.py` | Gold invariance, concentration measurement, error burst patterns at hard, no-op on identical values |

## Acceptance Criteria

1. Fusion gold values **byte-identical** before and after reshuffling
2. Per-attribute gold-carrier concentration measured ≥85% at easy
3. At hard, error_correlation produces entity-level burst patterns (≥2 attributes from same source wrong on same entity)
4. No-op when all values identical across sources (no variants to reshuffle)
5. Baseline measured fresh every run (self-contained)
6. `pytest usecases_synthetic/tests/test_knob_10.py -v` passes

## Dependencies

Module 0 (domain config, provenance, RNG, loaders). Logically operates on output of Knobs 1/5/6 but unit-testable with synthetic variant DataFrames.
