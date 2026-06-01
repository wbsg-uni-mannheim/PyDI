# Module 5: Knob 5 — Format/Unit Diversity

## Purpose

Structured-format and unit rewriting for dates, numbers, currencies, durations. Introduces the format family taxonomy, round-trip verification pattern, and static rate/conversion tables. Participates in the joint cell-collision system.

## Spec References

- **Knob card:** [knobs/knob_05_format_unit.md](../../knobs/knob_05_format_unit.md) — full specification including format family taxonomy, per-domain baselines, operator catalogue, locale handling, round-trip verification, static table specs (date_formats, number_locales, fx_rates, unit_factors), domain-specific notes
- **Cross-cutting provenance:** [knobs/cross_cutting.md](../../knobs/cross_cutting.md) § "Per-value provenance" — row-level provenance with `transform_fn ∈ {reformat_date, reformat_number, reconvert_unit, relocale}`
- **Cell-collision coordination:** [knobs/cross_cutting.md](../../knobs/cross_cutting.md) § "Cell-collision coordination" — K5: defensive skip on collision with prior K1 provenance
- **Committee fix-on-collapse:** [knobs/cross_cutting.md](../../knobs/cross_cutting.md) § "Committee-validated augmentation" — K5: trivial — canonical comparison handles it (round-trip guaranteed)

## Per-Domain Baselines (from knob card)

| Domain | Baseline | Notes |
|---|---|---|
| Companies | At hard on financials | Mixed magnitudes (billions vs. raw). Easy normalizes down |
| Games | Below easy | Entirely uniform formats. Every level requires active injection |
| Music | At hard on dates/durations | 4 coexisting date families; ms vs mm:ss durations. Easy/medium normalize down |

## Key Mechanism (from knob card § "Algorithm selection")

**Format family taxonomy (per attribute):** date, number, money, duration, dimensional, string.

**Per-level behavior:**
- **Easy:** 1-2 formats per attribute, consistent within source. One unit. One locale. Light reformat (not no-op — always at least one transform for testability)
- **Medium:** 2-3 formats, consistent within source. 1-2 units. Locale split along source lines
- **Hard:** 3+ formats, inconsistent **within** source. Multiple units. Locale mixed within source

**Operators:**
- `reformat_date(value, from_fmt, to_fmt)` — strftime-based, locale-aware
- `reformat_number(value, from_locale, to_locale)` — decimal/thousands separator conversion
- `reconvert_unit(value, from_unit, to_unit, rate)` — dimensional conversion with published rates
- `relocale(value, locale_tag)` — locale-specific formatting

**Round-trip verification:** Every emitted value must parse back to the canonical value (within FP tolerance for unit conversions). Fallback to identity on failure.

**Static tables (immutable once checked in):**
- `date_formats.yaml` — strftime patterns + locale-ambiguous deny-list (e.g., MM/DD/YYYY vs DD/MM/YYYY)
- `number_locales.yaml` — decimal/thousands separator combos per locale
- `fx_rates.yaml` — FX rates as of 2026-03-15, base currency USD
- `unit_factors.yaml` — dimensional conversion factors

## Files to Create

| File | Responsibility |
|---|---|
| `usecases_synthetic/scripts/apply_knob_05_format.py` | CLI: `--domain`, `--level`. Classifies attributes by format family, draws format assignments per (source, attr) or per (row, attr), applies operators, verifies round-trip, writes provenance |
| `usecases_synthetic/lib/format_operators.py` | `reformat_date`, `reformat_number`, `reconvert_unit`, `relocale`; each with built-in round-trip verifier returning `(new_value, params_dict)` or identity on failure |
| `usecases_synthetic/lib/rate_tables.py` | Loads static YAML tables. Exposes `get_fx_rate(from_ccy, to_ccy)`, `get_unit_factor(from_unit, to_unit)`, `get_date_formats(family)`, `get_locale_config(locale)`. Constants, no external calls |
| `usecases_synthetic/config/knob_05_format/companies.yaml` | Per-attribute format family, format pools per level, locale_mix settings |
| `usecases_synthetic/config/knob_05_format/_tables/date_formats.yaml` | strftime patterns + deny-list |
| `usecases_synthetic/config/knob_05_format/_tables/number_locales.yaml` | Locale separator configs |
| `usecases_synthetic/config/knob_05_format/_tables/fx_rates.yaml` | FX rates (2026-03-15) |
| `usecases_synthetic/config/knob_05_format/_tables/unit_factors.yaml` | Dimensional conversions |
| `usecases_synthetic/tests/test_knob_05.py` | Round-trip verification per operator, within-source consistency at easy/medium, within-source inconsistency at hard, deny-list enforcement, normalize-down for music/companies |

## Acceptance Criteria

1. Every reformatted value round-trips to the exact canonical value (within FP tolerance for unit conversions)
2. At easy, all values for a given (source, attribute) use the same format
3. At hard, ≥2 distinct formats appear within a single (source, attribute)
4. No locale-ambiguous date patterns emitted (deny-list active)
5. Provenance includes `from_format`, `to_format`, `rate` where applicable
6. `pytest usecases_synthetic/tests/test_knob_05.py -v` passes

## Dependencies

Module 0 (domain config, provenance, RNG, collision index, loaders).
