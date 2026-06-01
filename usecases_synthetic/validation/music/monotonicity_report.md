# Monotonicity + Collapse Report — music

Cross-level analysis of easy/medium/hard variants against the baseline. See `knobs/cross_cutting.md` for the protocol.

## Executive Summary

| Knob | Signals | OK direction | OK range | Notes |
|---|---|---|---|---|
| knob_01 | 3 | 0/3 [!!] | 0/— | [card](knobs/knob_01_surface_augmentation.md) |
| knob_02 | 3 | 0/3 [!!] | 0/— | [card](knobs/knob_02_niche_density.md) |
| knob_03 | 3 | 1/3 [!!] | 0/— | [card](knobs/knob_03_attribute_drop.md) |
| knob_04 | 3 | 1/3 [!!] | 0/— | [card](knobs/knob_04_coverage_skew.md) |
| knob_05 | 5 | 1/5 [!!] | 0/— | [card](knobs/knob_05_format_unit.md) |
| knob_06 | 4 | 1/4 [!!] | 0/— | [card](knobs/knob_06_value_noise.md) |
| knob_08 | 6 | 0/6 [!!] | 0/— | [card](knobs/knob_08_schema_naming.md) |
| knob_10 | 5 | 1/5 [!!] | 0/— | [card](knobs/knob_10_source_reliability.md) |

## Per-Signal Results

| Knob | Signal | Stage | Metric | Dir | baseline | easy | medium | hard | delta | target | Mono | Rng | Reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| knob_01 | sm_flat | sm | `aggregated.macro_f1` | flat | 0.876 | 0.893 | 0.760 | 0.670 | -0.206 | qualitative | [!!] | [ok] | not weakly flat: 0.876 -> 0.893 -> 0.760 -> 0.670 |
| knob_01 | em_monotone_drop | em | `aggregated.macro_f1_vs_pool` | down | NaN | NaN | NaN | NaN | NaN | qualitative | [!!] | [ok] | metric missing at levels: baseline, easy, medium, hard |
| knob_01 | em_blocker_spread | em | `spread:per_member.standard_rule.metrics.f1:per_member.embedding_rule.metrics.f1` | down | NaN | NaN | NaN | NaN | NaN | qualitative | [!!] | [ok] | metric missing at levels: baseline, easy, medium, hard |
| knob_02 | em_monotone_drop | em | `aggregated.macro_f1_vs_pool` | down | NaN | NaN | NaN | NaN | NaN | qualitative | [!!] | [ok] | metric missing at levels: baseline, easy, medium, hard |
| knob_02 | em_pool_precision_tightens | em | `aggregated.macro_pool_precision` | down | NaN | NaN | NaN | NaN | NaN | qualitative | [!!] | [ok] | metric missing at levels: baseline, easy, medium, hard |
| knob_02 | fusion_flat | fusion | `aggregated.overall_accuracy` | flat | 0.859 | 0.839 | 0.807 | 0.574 | -0.286 | qualitative | [!!] | [ok] | not weakly flat: 0.859 -> 0.839 -> 0.807 -> 0.574 |
| knob_03 | em_pool_recall_drops | em | `aggregated.macro_pool_recall` | down | NaN | NaN | NaN | NaN | NaN | qualitative | [!!] | [ok] | metric missing at levels: baseline, easy, medium, hard |
| knob_03 | em_monotone_drop | em | `aggregated.macro_f1_vs_pool` | down | NaN | NaN | NaN | NaN | NaN | qualitative | [!!] | [ok] | metric missing at levels: baseline, easy, medium, hard |
| knob_03 | fusion_monotone_drop | fusion | `aggregated.overall_accuracy` | down | 0.859 | 0.839 | 0.807 | 0.574 | -0.286 | qualitative | [ok] | [ok] | weakly down: 0.859 -> 0.839 -> 0.807 -> 0.574 |
| knob_04 | em_flat_or_shift | em | `aggregated.macro_f1_vs_pool` | flat | NaN | NaN | NaN | NaN | NaN | qualitative | [!!] | [ok] | metric missing at levels: baseline, easy, medium, hard |
| knob_04 | fusion_monotone_drop | fusion | `aggregated.overall_accuracy` | down | 0.859 | 0.839 | 0.807 | 0.574 | -0.286 | qualitative | [ok] | [ok] | weakly down: 0.859 -> 0.839 -> 0.807 -> 0.574 |
| knob_04 | fusion_spread_widens | fusion | `aggregated.overall_spread` | up | 0.114 | 0.096 | 0.195 | 0.062 | -0.052 | qualitative | [!!] | [ok] | not weakly up: 0.114 -> 0.096 -> 0.195 -> 0.062 |
| knob_05 | sm_flat | sm | `aggregated.macro_f1` | flat | 0.876 | 0.893 | 0.760 | 0.670 | -0.206 | qualitative | [!!] | [ok] | not weakly flat: 0.876 -> 0.893 -> 0.760 -> 0.670 |
| knob_05 | em_monotone_drop | em | `aggregated.macro_f1_vs_pool` | down | NaN | NaN | NaN | NaN | NaN | qualitative | [!!] | [ok] | metric missing at levels: baseline, easy, medium, hard |
| knob_05 | em_pool_recall_drops_for_lexical | em | `per_member.standard_rule.metrics.pool_recall` | down | NaN | NaN | NaN | NaN | NaN | qualitative | [!!] | [ok] | metric missing at levels: baseline, easy, medium, hard |
| knob_05 | fusion_spread_is_signal | fusion | `aggregated.overall_spread` | up | 0.114 | 0.096 | 0.195 | 0.062 | -0.052 | qualitative | [!!] | [ok] | not weakly up: 0.114 -> 0.096 -> 0.195 -> 0.062 |
| knob_05 | fusion_monotone_drop | fusion | `aggregated.overall_accuracy` | down | 0.859 | 0.839 | 0.807 | 0.574 | -0.286 | qualitative | [ok] | [ok] | weakly down: 0.859 -> 0.839 -> 0.807 -> 0.574 |
| knob_06 | sm_flat | sm | `aggregated.macro_f1` | flat | 0.876 | 0.893 | 0.760 | 0.670 | -0.206 | qualitative | [!!] | [ok] | not weakly flat: 0.876 -> 0.893 -> 0.760 -> 0.670 |
| knob_06 | em_monotone_drop | em | `aggregated.macro_f1_vs_pool` | down | NaN | NaN | NaN | NaN | NaN | qualitative | [!!] | [ok] | metric missing at levels: baseline, easy, medium, hard |
| knob_06 | em_pool_recall_drops | em | `aggregated.macro_pool_recall` | down | NaN | NaN | NaN | NaN | NaN | qualitative | [!!] | [ok] | metric missing at levels: baseline, easy, medium, hard |
| knob_06 | fusion_monotone_drop | fusion | `aggregated.overall_accuracy` | down | 0.859 | 0.839 | 0.807 | 0.574 | -0.286 | qualitative | [ok] | [ok] | weakly down: 0.859 -> 0.839 -> 0.807 -> 0.574 |
| knob_08 | sm_monotone_drop | sm | `aggregated.macro_f1` | down | 0.876 | 0.893 | 0.760 | 0.670 | -0.206 | qualitative | [!!] | [ok] | not weakly down: 0.876 -> 0.893 -> 0.760 -> 0.670 |
| knob_08 | sm_label_collapses | sm | `per_member.label_jaccard.metrics.f1` | down | NaN | NaN | NaN | NaN | NaN | qualitative | [!!] | [ok] | metric missing at levels: baseline, easy, medium, hard |
| knob_08 | sm_spread_is_signal | sm | `spread:per_member.llm_openai.metrics.f1:per_member.label_jaccard.metrics.f1` | up | NaN | NaN | NaN | NaN | NaN | qualitative | [!!] | [ok] | metric missing at levels: baseline, easy, medium, hard |
| knob_08 | sm_instance_steady | sm | `per_member.instance_tfidf_cosine.metrics.f1` | flat | NaN | NaN | NaN | NaN | NaN | qualitative | [!!] | [ok] | metric missing at levels: baseline, easy, medium, hard |
| knob_08 | em_flat | em | `aggregated.macro_f1_vs_pool` | flat | NaN | NaN | NaN | NaN | NaN | qualitative | [!!] | [ok] | metric missing at levels: baseline, easy, medium, hard |
| knob_08 | fusion_flat | fusion | `aggregated.overall_accuracy` | flat | 0.859 | 0.839 | 0.807 | 0.574 | -0.286 | qualitative | [!!] | [ok] | not weakly flat: 0.859 -> 0.839 -> 0.807 -> 0.574 |
| knob_10 | sm_flat | sm | `aggregated.macro_f1` | flat | 0.876 | 0.893 | 0.760 | 0.670 | -0.206 | qualitative | [!!] | [ok] | not weakly flat: 0.876 -> 0.893 -> 0.760 -> 0.670 |
| knob_10 | em_flat | em | `aggregated.macro_f1_vs_pool` | flat | NaN | NaN | NaN | NaN | NaN | qualitative | [!!] | [ok] | metric missing at levels: baseline, easy, medium, hard |
| knob_10 | fusion_monotone_drop | fusion | `aggregated.overall_accuracy` | down | 0.859 | 0.839 | 0.807 | 0.574 | -0.286 | qualitative | [ok] | [ok] | weakly down: 0.859 -> 0.839 -> 0.807 -> 0.574 |
| knob_10 | fusion_spread_widens | fusion | `aggregated.overall_spread` | up | 0.114 | 0.096 | 0.195 | 0.062 | -0.052 | qualitative | [!!] | [ok] | not weakly up: 0.114 -> 0.096 -> 0.195 -> 0.062 |
| knob_10 | fusion_voting_drops_faster_than_trust_on_name | fusion | `spread:per_attribute.name.prefer_higher_trust:per_attribute.name.voting` | up | -0.010 | -0.010 | -0.030 | -0.020 | -0.010 | qualitative | [!!] | [ok] | not weakly up: -0.010 -> -0.010 -> -0.030 -> -0.020 |

## Collapses

Members with F1 < 0.15 or drop > 0.5 from baseline. EM collapses are classified against the pool-agreement diagnostic (`hidden_positive_noise` means pool is stable while test-gold moved — see `knobs/cross_cutting.md` § Protection set semantics).

| Level | Stage | Member | baseline F1 | measured F1 | delta | classification | pool delta | action |
|---|---|---|---|---|---|---|---|---|
| hard | sm | `coma_hybrid` | 1.000 | 0.457 | -0.543 | unknown | — | see knobs/cross_cutting.md § Per-knob fix-strategy defaults (SM row) |
| hard | sm | `label_jw` | 1.000 | 0.364 | -0.636 | unknown | — | see knobs/cross_cutting.md § Per-knob fix-strategy defaults (SM row) |

## Best-Member Ceiling (P8)

Per-stage best-member F1 across baseline -> easy -> medium -> hard. A valid difficulty signal must depress the *ceiling* (the user-attainable member), not just the committee mean. A flat / rising ceiling means the user-selected matcher never sees the synthetic difficulty (committee-mean drift can be masked by weak-member degradation alone).

| Stage | baseline | easy | medium | hard | delta | non-increasing | winner trail | Reason |
|---|---|---|---|---|---|---|---|---|
| sm | 1.000 | 1.000 | 1.000 | 0.920 | -0.080 | [ok] | `label_jw -> llm_openai -> llm_openai -> llm_openai` | best-member ceiling non-increasing: 1.000 -> 1.000 -> 1.000 -> 0.920  (label_jw -> llm_openai -> llm_openai -> llm_openai) |
| norm | 0.768 | 0.774 | 0.758 | 0.776 | 0.008 | [!!] | `text_clean -> text_clean -> text_clean -> text_clean` | best-member ceiling did NOT decline: 0.768 -> 0.774 -> 0.758 -> 0.776  (text_clean -> text_clean -> text_clean -> text_clean) — difficulty dial may be invisible to the user-selected matcher |
| em_blocking | 0.999 | 0.999 | 0.999 | 0.590 | -0.409 | [!!] | `sc_block -> sc_block -> token_blocker -> sc_block` | best-member ceiling did NOT decline: 0.999 -> 0.999 -> 0.999 -> 0.590  (sc_block -> sc_block -> token_blocker -> sc_block) — difficulty dial may be invisible to the user-selected matcher |
| em_matching | 0.976 | 0.976 | 0.976 | 0.874 | -0.101 | [ok] | `ditto_plm -> ditto_plm -> ditto_plm -> ditto_plm` | best-member ceiling non-increasing: 0.976 -> 0.976 -> 0.976 -> 0.874  (ditto_plm -> ditto_plm -> ditto_plm -> ditto_plm) |
| fusion | 0.838 | 0.819 | 0.747 | 0.564 | -0.274 | [ok] | `duration_maximum -> duration_huber_m_estimator -> tracks_union -> release-country_llm_judge` | best-member ceiling non-increasing: 0.838 -> 0.819 -> 0.747 -> 0.564  (duration_maximum -> duration_huber_m_estimator -> tracks_union -> release-country_llm_judge) |

## Open Questions

Signals that are direction-correct but magnitude-unspecified by the knob card. M10 should decide whether the measured delta is 'strong enough' to count as validation.

| Knob | Signal | Stage | observed delta | Card notes |
|---|---|---|---|---|
| knob_03 | fusion_monotone_drop | fusion | -0.286 | Fusion accuracy drops monotonically given conflict-preserving constraint and survivor cap. |
| knob_04 | fusion_monotone_drop | fusion | -0.286 | Primary target. Voting-family strategies degrade faster than trust-weighted / single-best-source strategies. |
| knob_05 | fusion_monotone_drop | fusion | -0.286 | Monotone drop for naive strategies. Best-strategy accuracy (the max across strategies) may hold if a canonicalizing strategy is available, but overall_accuracy averages all strategies and therefore drops. |
| knob_06 | fusion_monotone_drop | fusion | -0.286 | Monotone drop given survivor floors. Provenance-aware fusers should be rewarded over cell-local. |
| knob_10 | fusion_monotone_drop | fusion | -0.286 | Primary target. Cell-local strategies (voting, most_complete, per-attribute trust) lose accuracy as unreliability grows. |

## Provenance

- Domain: music
- Expectations: `usecases_synthetic/config/knob_expected_signals.yaml`
- Baseline: `usecases_synthetic/baselines/music/baseline_metrics.json`
- Per-level metrics: `usecases_synthetic/validation/music/<level>/metrics.json`
