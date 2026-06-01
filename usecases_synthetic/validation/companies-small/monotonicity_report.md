# Monotonicity + Collapse Report — companies-small

Cross-level analysis of easy/medium/hard variants against the baseline. See `knobs/cross_cutting.md` for the protocol.

## Executive Summary

| Knob | Signals | OK direction | OK range | Notes |
|---|---|---|---|---|
| knob_01 | 3 | 0/3 [!!] | 0/— | [card](knobs/knob_01_surface_augmentation.md) |
| knob_02 | 3 | 0/3 [!!] | 0/— | [card](knobs/knob_02_niche_density.md) |
| knob_03 | 3 | 0/3 [!!] | 0/— | [card](knobs/knob_03_attribute_drop.md) |
| knob_04 | 3 | 0/3 [!!] | 0/— | [card](knobs/knob_04_coverage_skew.md) |
| knob_05 | 5 | 0/5 [!!] | 0/— | [card](knobs/knob_05_format_unit.md) |
| knob_06 | 4 | 0/4 [!!] | 0/— | [card](knobs/knob_06_value_noise.md) |
| knob_08 | 6 | 0/6 [!!] | 0/— | [card](knobs/knob_08_schema_naming.md) |
| knob_10 | 5 | 0/5 [!!] | 0/— | [card](knobs/knob_10_source_reliability.md) |

## Per-Signal Results

| Knob | Signal | Stage | Metric | Dir | baseline | easy | medium | hard | delta | target | Mono | Rng | Reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| knob_01 | sm_flat | sm | `aggregated.macro_f1` | flat | 0.673 | 0.830 | 0.702 | 0.594 | -0.079 | qualitative | [!!] | [ok] | not weakly flat: 0.673 -> 0.830 -> 0.702 -> 0.594 |
| knob_01 | em_monotone_drop | em | `aggregated.macro_f1_vs_pool` | down | NaN | NaN | NaN | NaN | NaN | qualitative | [!!] | [ok] | metric missing at levels: baseline, easy, medium, hard |
| knob_01 | em_blocker_spread | em | `spread:per_member.standard_rule.metrics.f1:per_member.embedding_rule.metrics.f1` | down | NaN | NaN | NaN | NaN | NaN | qualitative | [!!] | [ok] | metric missing at levels: baseline, easy, medium, hard |
| knob_02 | em_monotone_drop | em | `aggregated.macro_f1_vs_pool` | down | NaN | NaN | NaN | NaN | NaN | qualitative | [!!] | [ok] | metric missing at levels: baseline, easy, medium, hard |
| knob_02 | em_pool_precision_tightens | em | `aggregated.macro_pool_precision` | down | NaN | NaN | NaN | NaN | NaN | qualitative | [!!] | [ok] | metric missing at levels: baseline, easy, medium, hard |
| knob_02 | fusion_flat | fusion | `aggregated.overall_accuracy` | flat | 0.881 | 0.619 | 0.786 | 0.611 | -0.270 | qualitative | [!!] | [ok] | not weakly flat: 0.881 -> 0.619 -> 0.786 -> 0.611 |
| knob_03 | em_pool_recall_drops | em | `aggregated.macro_pool_recall` | down | NaN | NaN | NaN | NaN | NaN | qualitative | [!!] | [ok] | metric missing at levels: baseline, easy, medium, hard |
| knob_03 | em_monotone_drop | em | `aggregated.macro_f1_vs_pool` | down | NaN | NaN | NaN | NaN | NaN | qualitative | [!!] | [ok] | metric missing at levels: baseline, easy, medium, hard |
| knob_03 | fusion_monotone_drop | fusion | `aggregated.overall_accuracy` | down | 0.881 | 0.619 | 0.786 | 0.611 | -0.270 | qualitative | [!!] | [ok] | not weakly down: 0.881 -> 0.619 -> 0.786 -> 0.611 |
| knob_04 | em_flat_or_shift | em | `aggregated.macro_f1_vs_pool` | flat | NaN | NaN | NaN | NaN | NaN | qualitative | [!!] | [ok] | metric missing at levels: baseline, easy, medium, hard |
| knob_04 | fusion_monotone_drop | fusion | `aggregated.overall_accuracy` | down | 0.881 | 0.619 | 0.786 | 0.611 | -0.270 | qualitative | [!!] | [ok] | not weakly down: 0.881 -> 0.619 -> 0.786 -> 0.611 |
| knob_04 | fusion_spread_widens | fusion | `aggregated.overall_spread` | up | 0.286 | 0.095 | 0.198 | 0.119 | -0.167 | qualitative | [!!] | [ok] | not weakly up: 0.286 -> 0.095 -> 0.198 -> 0.119 |
| knob_05 | sm_flat | sm | `aggregated.macro_f1` | flat | 0.673 | 0.830 | 0.702 | 0.594 | -0.079 | qualitative | [!!] | [ok] | not weakly flat: 0.673 -> 0.830 -> 0.702 -> 0.594 |
| knob_05 | em_monotone_drop | em | `aggregated.macro_f1_vs_pool` | down | NaN | NaN | NaN | NaN | NaN | qualitative | [!!] | [ok] | metric missing at levels: baseline, easy, medium, hard |
| knob_05 | em_pool_recall_drops_for_lexical | em | `per_member.standard_rule.metrics.pool_recall` | down | NaN | NaN | NaN | NaN | NaN | qualitative | [!!] | [ok] | metric missing at levels: baseline, easy, medium, hard |
| knob_05 | fusion_spread_is_signal | fusion | `aggregated.overall_spread` | up | 0.286 | 0.095 | 0.198 | 0.119 | -0.167 | qualitative | [!!] | [ok] | not weakly up: 0.286 -> 0.095 -> 0.198 -> 0.119 |
| knob_05 | fusion_monotone_drop | fusion | `aggregated.overall_accuracy` | down | 0.881 | 0.619 | 0.786 | 0.611 | -0.270 | qualitative | [!!] | [ok] | not weakly down: 0.881 -> 0.619 -> 0.786 -> 0.611 |
| knob_06 | sm_flat | sm | `aggregated.macro_f1` | flat | 0.673 | 0.830 | 0.702 | 0.594 | -0.079 | qualitative | [!!] | [ok] | not weakly flat: 0.673 -> 0.830 -> 0.702 -> 0.594 |
| knob_06 | em_monotone_drop | em | `aggregated.macro_f1_vs_pool` | down | NaN | NaN | NaN | NaN | NaN | qualitative | [!!] | [ok] | metric missing at levels: baseline, easy, medium, hard |
| knob_06 | em_pool_recall_drops | em | `aggregated.macro_pool_recall` | down | NaN | NaN | NaN | NaN | NaN | qualitative | [!!] | [ok] | metric missing at levels: baseline, easy, medium, hard |
| knob_06 | fusion_monotone_drop | fusion | `aggregated.overall_accuracy` | down | 0.881 | 0.619 | 0.786 | 0.611 | -0.270 | qualitative | [!!] | [ok] | not weakly down: 0.881 -> 0.619 -> 0.786 -> 0.611 |
| knob_08 | sm_monotone_drop | sm | `aggregated.macro_f1` | down | 0.673 | 0.830 | 0.702 | 0.594 | -0.079 | qualitative | [!!] | [ok] | not weakly down: 0.673 -> 0.830 -> 0.702 -> 0.594 |
| knob_08 | sm_label_collapses | sm | `per_member.label_jaccard.metrics.f1` | down | NaN | NaN | NaN | NaN | NaN | qualitative | [!!] | [ok] | metric missing at levels: baseline, easy, medium, hard |
| knob_08 | sm_spread_is_signal | sm | `spread:per_member.llm_openai.metrics.f1:per_member.label_jaccard.metrics.f1` | up | NaN | NaN | NaN | NaN | NaN | qualitative | [!!] | [ok] | metric missing at levels: baseline, easy, medium, hard |
| knob_08 | sm_instance_steady | sm | `per_member.instance_tfidf_cosine.metrics.f1` | flat | NaN | NaN | NaN | NaN | NaN | qualitative | [!!] | [ok] | metric missing at levels: baseline, easy, medium, hard |
| knob_08 | em_flat | em | `aggregated.macro_f1_vs_pool` | flat | NaN | NaN | NaN | NaN | NaN | qualitative | [!!] | [ok] | metric missing at levels: baseline, easy, medium, hard |
| knob_08 | fusion_flat | fusion | `aggregated.overall_accuracy` | flat | 0.881 | 0.619 | 0.786 | 0.611 | -0.270 | qualitative | [!!] | [ok] | not weakly flat: 0.881 -> 0.619 -> 0.786 -> 0.611 |
| knob_10 | sm_flat | sm | `aggregated.macro_f1` | flat | 0.673 | 0.830 | 0.702 | 0.594 | -0.079 | qualitative | [!!] | [ok] | not weakly flat: 0.673 -> 0.830 -> 0.702 -> 0.594 |
| knob_10 | em_flat | em | `aggregated.macro_f1_vs_pool` | flat | NaN | NaN | NaN | NaN | NaN | qualitative | [!!] | [ok] | metric missing at levels: baseline, easy, medium, hard |
| knob_10 | fusion_monotone_drop | fusion | `aggregated.overall_accuracy` | down | 0.881 | 0.619 | 0.786 | 0.611 | -0.270 | qualitative | [!!] | [ok] | not weakly down: 0.881 -> 0.619 -> 0.786 -> 0.611 |
| knob_10 | fusion_spread_widens | fusion | `aggregated.overall_spread` | up | 0.286 | 0.095 | 0.198 | 0.119 | -0.167 | qualitative | [!!] | [ok] | not weakly up: 0.286 -> 0.095 -> 0.198 -> 0.119 |
| knob_10 | fusion_voting_drops_faster_than_trust_on_name | fusion | `spread:per_attribute.name.prefer_higher_trust:per_attribute.name.voting` | up | -0.167 | 0.167 | -0.056 | -0.167 | 0.000 | qualitative | [!!] | [ok] | not weakly up: -0.167 -> 0.167 -> -0.056 -> -0.167 |

## Collapses

No members fell below the collapse threshold.

## Best-Member Ceiling (P8)

Per-stage best-member F1 across baseline -> easy -> medium -> hard. A valid difficulty signal must depress the *ceiling* (the user-attainable member), not just the committee mean. A flat / rising ceiling means the user-selected matcher never sees the synthetic difficulty (committee-mean drift can be masked by weak-member degradation alone).

| Stage | baseline | easy | medium | hard | delta | non-increasing | winner trail | Reason |
|---|---|---|---|---|---|---|---|---|
| sm | 0.884 | 0.952 | 0.884 | 0.884 | 0.000 | [!!] | `llm_openai -> coma_hybrid -> llm_openai -> llm_openai` | best-member ceiling did NOT decline: 0.884 -> 0.952 -> 0.884 -> 0.884  (llm_openai -> coma_hybrid -> llm_openai -> llm_openai) — difficulty dial may be invisible to the user-selected matcher |
| norm | 0.849 | 0.886 | 0.813 | 0.800 | -0.049 | [!!] | `country_iso -> country_iso -> date_iso -> date_iso` | best-member ceiling did NOT decline: 0.849 -> 0.886 -> 0.813 -> 0.800  (country_iso -> country_iso -> date_iso -> date_iso) — difficulty dial may be invisible to the user-selected matcher |
| em_blocking | 0.980 | 0.980 | 0.980 | 0.532 | -0.449 | [ok] | `embedding_blocker -> embedding_blocker -> embedding_blocker -> embedding_blocker` | best-member ceiling non-increasing: 0.980 -> 0.980 -> 0.980 -> 0.532  (embedding_blocker -> embedding_blocker -> embedding_blocker -> embedding_blocker) |
| em_matching | 0.909 | 0.936 | 0.907 | 0.921 | 0.012 | [!!] | `llm_matcher -> magellan -> magellan -> magellan` | best-member ceiling did NOT decline: 0.909 -> 0.936 -> 0.907 -> 0.921  (llm_matcher -> magellan -> magellan -> magellan) — difficulty dial may be invisible to the user-selected matcher |
| fusion | 0.857 | 0.579 | 0.714 | 0.579 | -0.278 | [!!] | `city_shortest_string -> city_prefer_higher_trust -> revenue_maximum -> country_truthfinder` | best-member ceiling did NOT decline: 0.857 -> 0.579 -> 0.714 -> 0.579  (city_shortest_string -> city_prefer_higher_trust -> revenue_maximum -> country_truthfinder) — difficulty dial may be invisible to the user-selected matcher |

## Open Questions

No pending magnitude decisions — every qualitative signal passed or failed on direction alone.

## Provenance

- Domain: companies-small
- Expectations: `usecases_synthetic/config/knob_expected_signals.yaml`
- Baseline: `usecases_synthetic/baselines/companies-small/baseline_metrics.json`
- Per-level metrics: `usecases_synthetic/validation/companies-small/<level>/metrics.json`
