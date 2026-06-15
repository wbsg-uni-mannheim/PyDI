# Knobs — Difficulty Generation Spec

Final deliverable from the brainstorm in [../plan_brainstorm.md](../plan_brainstorm.md). Each knob is its own card; this README is the index and the verification artifacts (back-mapping, canonical order, ablations cross-link).

For the brainstorm history, per-step notes, measured baselines, and rationale, see [../summary_perstep_brainstorm.md](../summary_perstep_brainstorm.md). For the underlying dimensions, see [../difficulty_dimensions.md](../difficulty_dimensions.md). For cross-cutting policies that apply to every knob (committee mechanism, profile model, provenance schema, test-set treatment), see [cross_cutting.md](cross_cutting.md).

## Knob index

| # | Knob | Primary stage(s) | Scenario | Card |
|---|---|---|---|---|
| 1 | Surface augmentation intensity | EM, Fusion | S1 + S2 | [knob_01_surface_augmentation.md](knob_01_surface_augmentation.md) |
| 2 | Entity niche density | Blocking, EM | S1 (partial ↓) + S2 | [knob_02_niche_density.md](knob_02_niche_density.md) |
| 3 | Per-source attribute drop rate | Block, EM, Fusion | S1 + S2 | [knob_03_attribute_drop.md](knob_03_attribute_drop.md) |
| 4 | Per-entity source coverage skew | Fusion | S1 + S2 | [knob_04_coverage_skew.md](knob_04_coverage_skew.md) |
| 5 | Format / unit diversity | Normalization | S1 + S2 | [knob_05_format_unit.md](knob_05_format_unit.md) |
| 6 | Value-noise injection rate | Norm, EM, Fusion | S1 + S2 | [knob_06_value_noise.md](knob_06_value_noise.md) |
| 7 | Value ambiguity / collision rate | Norm, Fusion | S1 + S2 (specced, not built v1) | [knob_07_value_ambiguity.md](knob_07_value_ambiguity.md) |
| 8 | Schema naming divergence | SM | S1 + S2 | [knob_08_schema_naming.md](knob_08_schema_naming.md) |
| 9 | Schema completeness / distractors | SM | **S2 only** | [knob_09_schema_completeness.md](knob_09_schema_completeness.md) |
| 10 | Source reliability differentiation | Fusion | S1 + S2 | [knob_10_source_reliability.md](knob_10_source_reliability.md) |

**Out of scope:** inter-source structural divergence (nesting, splitting fields). PyDI is 1:1 schema matching only.

## Canonical knob application order

Single source of truth for the order in which knobs are applied by the generator.

- **Scenario 1 (augmentation):**
  `Knob 2 (niche density) → Knob 4 (coverage skew) → Knobs 1/5/6/7 (value perturbations, jointly per cell) → Knob 3 (attribute drop) → Knob 10 (reliability reshuffle) → Knob 8 (header rename)`
- **Scenario 2 (synthetic):** prepend `Knob 9 (schema completeness / distractors)`. Knob 9 fixes the column set; the S1 order then runs against it. Knob 10 in S2 is baked into source generation rather than applied post-hoc, but logically occupies the same position.

Rationale: Knob 4 takes Knob 2's placements as fixed input; Knobs 1/5/6/7 run before Knob 3 so drops happen on perturbed data; Knob 10 reshuffles among whatever variants 1/5/6/7 produced; Knob 8 is header-only and orthogonal, so last; Knob 9 fixes the column set so it goes first in S2.

## Dimension → profile back-mapping (Step 3 verification)

Every dimension in [../difficulty_dimensions.md](../difficulty_dimensions.md) is covered by ≥1 knob. Each dimension's level under a profile equals the level set on its driving knob(s).

| Pipeline stage | Dimension | Driving knob(s) |
|---|---|---|
| SM | Naming Heterogeneity | 8 |
| SM | Schema Completeness | 9 (S2 only) |
| SM | Semantic Ambiguity | 9 (S2 only) |
| Norm | Format Heterogeneity | 5 |
| Norm | Unit & Scale Diversity | 5 |
| Norm | Noise & Corruption | 6 |
| Norm | Value Ambiguity | 7 (specced, not built v1) |
| Block | Representation Heterogeneity | 1 (primary), 5 (residual) |
| Block | Candidate Density | 2 |
| Block | Blocking Key Completeness | 3 |
| EM | Corner-Case Ratio | 2 |
| EM | Corner-Case Difficulty | 1, 6 |
| EM | Record Completeness | 3 |
| EM | Entity Group Size Variance | 4 |
| Fusion | Source Density | 3, 4 |
| Fusion | Conflict Rate | 4, 6, 7, 10 (indirect) |
| Fusion | Conflict Subtlety | 1, 7 |
| Fusion | Trust Ambiguity | 10 |

**S1-only gaps:** Schema Completeness and Semantic Ambiguity are not exercised in S1 by design (Knob 9 is S2-only). Value Ambiguity / Conflict Subtlety from Knob 7 are under-stressed in v1 (Knob 7 specced but not built — re-opens if the parked accepted-sets discussion lands favorably).

## Ablation candidates

See [ablations.md](ablations.md). Provisional shortlist (final cut deferred to after the easy/medium/hard prototype runs): Knobs **2, 3, 5, 8**, with Knob **1** as alternate.
