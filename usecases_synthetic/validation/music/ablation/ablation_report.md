# Ablation Report — music

Per-knob ablation validation: each knob set to `hard` with all others at `easy` (identity). See `plans/validation/module_09_ablation.md` and `knobs/ablations.md` for the independent-togglability requirement.

## Executive Summary

| Knob | Primary stage | Primary delta | Direction ok | Leakage | Under | Over | Mismatch | Card |
|---|---|---|---|---|---|---|---|---|
| knob_01 | em | -0.106 | [!!] | [!!] | [ok] | [!!] | [ok] | [card](knobs/knob_01_surface_augmentation.md) |
| knob_02 | em | -0.153 | [ok] | [ok] | [ok] | [!!] | [ok] | [card](knobs/knob_02_niche_density.md) |
| knob_03 | em | -0.140 | [ok] | [ok] | [ok] | [!!] | [ok] | [card](knobs/knob_03_attribute_drop.md) |
| knob_04 | fusion | -0.031 | [ok] | [ok] | [!!] | [ok] | [ok] | [card](knobs/knob_04_coverage_skew.md) |
| knob_05 | fusion | -0.031 | [ok] | [!!] | [!!] | [ok] | [ok] | [card](knobs/knob_05_format_unit.md) |
| knob_06 | em | -0.104 | [ok] | [!!] | [ok] | [!!] | [ok] | [card](knobs/knob_06_value_noise.md) |
| knob_08 | sm | -0.164 | [!!] | [!!] | [ok] | [ok] | [ok] | [card](knobs/knob_08_schema_naming.md) |
| knob_10 | fusion | -0.031 | [ok] | [!!] | [!!] | [ok] | [ok] | [card](knobs/knob_10_source_reliability.md) |

## Per-Signal Results

| Knob | Signal | Stage | Primary | Metric | Dir | baseline | ablation | full-hard | delta | hard delta | Dir ok | Flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| knob_01 | sm_flat | sm | no | `aggregated.macro_f1` | flat | 0.362 | 0.470 | 0.176 | +0.108 | -0.186 | [!!] | cross_stage_leakage |
| knob_01 | em_monotone_drop | em | yes | `aggregated.macro_f1_vs_pool` | down | 0.851 | 0.745 | 0.759 | -0.106 | -0.092 | [ok] | primary_over_signal |
| knob_01 | em_blocker_spread | em | yes | `spread:per_member.standard_rule.metrics.f1:per_member.embedding_rule.metrics.f1` | down | NaN | NaN | NaN | NaN | NaN | [!!] | — |
| knob_02 | em_monotone_drop | em | yes | `aggregated.macro_f1_vs_pool` | down | 0.851 | 0.769 | 0.759 | -0.082 | -0.092 | [ok] | — |
| knob_02 | em_pool_precision_tightens | em | yes | `aggregated.macro_pool_precision` | down | 0.915 | 0.761 | 0.813 | -0.153 | -0.102 | [ok] | primary_over_signal |
| knob_02 | fusion_flat | fusion | no | `aggregated.overall_accuracy` | flat | 0.550 | 0.540 | 0.480 | -0.010 | -0.070 | [ok] | — |
| knob_03 | em_pool_recall_drops | em | yes | `aggregated.macro_pool_recall` | down | 0.394 | 0.372 | 0.352 | -0.023 | -0.043 | [ok] | — |
| knob_03 | em_monotone_drop | em | yes | `aggregated.macro_f1_vs_pool` | down | 0.851 | 0.711 | 0.759 | -0.140 | -0.092 | [ok] | primary_over_signal |
| knob_03 | fusion_monotone_drop | fusion | no | `aggregated.overall_accuracy` | down | 0.550 | 0.540 | 0.480 | -0.010 | -0.070 | [ok] | — |
| knob_04 | em_flat_or_shift | em | no | `aggregated.macro_f1_vs_pool` | flat | 0.851 | 0.844 | 0.759 | -0.007 | -0.092 | [ok] | — |
| knob_04 | fusion_monotone_drop | fusion | yes | `aggregated.overall_accuracy` | down | 0.550 | 0.540 | 0.480 | -0.010 | -0.070 | [ok] | primary_under_signal |
| knob_04 | fusion_spread_widens | fusion | yes | `aggregated.overall_spread` | up | 0.043 | 0.012 | 0.075 | -0.031 | +0.031 | [ok] | — |
| knob_05 | sm_flat | sm | no | `aggregated.macro_f1` | flat | 0.362 | 0.446 | 0.176 | +0.084 | -0.186 | [!!] | cross_stage_leakage |
| knob_05 | em_monotone_drop | em | no | `aggregated.macro_f1_vs_pool` | down | 0.851 | 0.769 | 0.759 | -0.082 | -0.092 | [ok] | — |
| knob_05 | em_pool_recall_drops_for_lexical | em | no | `per_member.standard_rule.metrics.pool_recall` | down | NaN | NaN | NaN | NaN | NaN | [!!] | — |
| knob_05 | fusion_spread_is_signal | fusion | yes | `aggregated.overall_spread` | up | 0.043 | 0.012 | 0.075 | -0.031 | +0.031 | [ok] | — |
| knob_05 | fusion_monotone_drop | fusion | yes | `aggregated.overall_accuracy` | down | 0.550 | 0.540 | 0.480 | -0.010 | -0.070 | [ok] | primary_under_signal |
| knob_06 | sm_flat | sm | no | `aggregated.macro_f1` | flat | 0.362 | 0.445 | 0.176 | +0.083 | -0.186 | [!!] | cross_stage_leakage |
| knob_06 | em_monotone_drop | em | yes | `aggregated.macro_f1_vs_pool` | down | 0.851 | 0.747 | 0.759 | -0.104 | -0.092 | [ok] | primary_over_signal |
| knob_06 | em_pool_recall_drops | em | yes | `aggregated.macro_pool_recall` | down | 0.394 | 0.371 | 0.352 | -0.023 | -0.043 | [ok] | — |
| knob_06 | fusion_monotone_drop | fusion | no | `aggregated.overall_accuracy` | down | 0.550 | 0.507 | 0.480 | -0.043 | -0.070 | [ok] | — |
| knob_08 | sm_monotone_drop | sm | yes | `aggregated.macro_f1` | down | 0.362 | 0.198 | 0.176 | -0.164 | -0.186 | [ok] | — |
| knob_08 | sm_label_collapses | sm | yes | `per_member.label_jaccard.metrics.f1` | down | NaN | NaN | NaN | NaN | NaN | [!!] | — |
| knob_08 | sm_spread_is_signal | sm | yes | `spread:per_member.llm_openai.metrics.f1:per_member.label_jaccard.metrics.f1` | up | NaN | NaN | NaN | NaN | NaN | [!!] | — |
| knob_08 | sm_instance_steady | sm | yes | `per_member.instance_tfidf_cosine.metrics.f1` | flat | NaN | NaN | NaN | NaN | NaN | [!!] | — |
| knob_08 | em_flat | em | no | `aggregated.macro_f1_vs_pool` | flat | 0.851 | 0.769 | 0.759 | -0.082 | -0.092 | [!!] | cross_stage_leakage |
| knob_08 | fusion_flat | fusion | no | `aggregated.overall_accuracy` | flat | 0.550 | 0.540 | 0.480 | -0.010 | -0.070 | [ok] | — |
| knob_10 | sm_flat | sm | no | `aggregated.macro_f1` | flat | 0.362 | 0.460 | 0.176 | +0.098 | -0.186 | [!!] | cross_stage_leakage |
| knob_10 | em_flat | em | no | `aggregated.macro_f1_vs_pool` | flat | 0.851 | 0.769 | 0.759 | -0.081 | -0.092 | [!!] | cross_stage_leakage |
| knob_10 | fusion_monotone_drop | fusion | yes | `aggregated.overall_accuracy` | down | 0.550 | 0.519 | 0.480 | -0.031 | -0.070 | [ok] | primary_under_signal |
| knob_10 | fusion_spread_widens | fusion | yes | `aggregated.overall_spread` | up | 0.043 | 0.031 | 0.075 | -0.012 | +0.031 | [ok] | primary_under_signal |
| knob_10 | fusion_voting_drops_faster_than_trust_on_name | fusion | yes | `spread:per_attribute.name.prefer_higher_trust:per_attribute.name.voting` | up | 0.000 | 0.000 | 0.000 | +0.000 | +0.000 | [ok] | primary_under_signal |

## Interaction Flags

- `cross_stage_leakage` — non-primary stage moved more than the flat tolerance. Usually indicates a variant-packaging bug (e.g. renamed headers breaking fusion comparators).
- `primary_under_signal` — primary-stage delta is materially smaller than the full-hard displacement. Knob may be dominated by another knob at hard level. Usually fine; log for M10.
- `primary_over_signal` — primary-stage delta exceeds the full-hard displacement. Indicates cancellation between knobs at full-hard; worth a scheduling review.
- `direction_mismatch` — primary-stage signal moved opposite the card's predicted direction. Treat as a knob-authoring bug.

## Provenance

- Domain: music
- Expectations: `usecases_synthetic/config/knob_expected_signals.yaml`
- Baseline: `usecases_synthetic/baselines/music/baseline_metrics.json`
- Full-hard metrics: `usecases_synthetic/validation/music/hard/metrics.json`
- knob_01 metrics: `usecases_synthetic/validation/music/ablation/knob_01/metrics.json`
- knob_02 metrics: `usecases_synthetic/validation/music/ablation/knob_02/metrics.json`
- knob_03 metrics: `usecases_synthetic/validation/music/ablation/knob_03/metrics.json`
- knob_04 metrics: `usecases_synthetic/validation/music/ablation/knob_04/metrics.json`
- knob_05 metrics: `usecases_synthetic/validation/music/ablation/knob_05/metrics.json`
- knob_06 metrics: `usecases_synthetic/validation/music/ablation/knob_06/metrics.json`
- knob_08 metrics: `usecases_synthetic/validation/music/ablation/knob_08/metrics.json`
- knob_10 metrics: `usecases_synthetic/validation/music/ablation/knob_10/metrics.json`
