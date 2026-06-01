# Monotonicity + Collapse Report — games

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
| knob_01 | sm_flat | sm | `aggregated.macro_f1` | flat | 0.727 | 0.861 | 0.748 | 0.678 | -0.049 | qualitative | [!!] | [ok] | not weakly flat: 0.727 -> 0.861 -> 0.748 -> 0.678 |
| knob_01 | em_monotone_drop | em | `aggregated.macro_f1_vs_pool` | down | NaN | NaN | NaN | NaN | NaN | qualitative | [!!] | [ok] | metric missing at levels: baseline, easy, medium, hard |
| knob_01 | em_blocker_spread | em | `spread:per_member.standard_rule.metrics.f1:per_member.embedding_rule.metrics.f1` | down | NaN | NaN | NaN | NaN | NaN | qualitative | [!!] | [ok] | metric missing at levels: baseline, easy, medium, hard |
| knob_02 | em_monotone_drop | em | `aggregated.macro_f1_vs_pool` | down | NaN | NaN | NaN | NaN | NaN | qualitative | [!!] | [ok] | metric missing at levels: baseline, easy, medium, hard |
| knob_02 | em_pool_precision_tightens | em | `aggregated.macro_pool_precision` | down | NaN | NaN | NaN | NaN | NaN | qualitative | [!!] | [ok] | metric missing at levels: baseline, easy, medium, hard |
| knob_02 | fusion_flat | fusion | `aggregated.overall_accuracy` | flat | 0.747 | 0.742 | 0.715 | 0.661 | -0.086 | qualitative | [!!] | [ok] | not weakly flat: 0.747 -> 0.742 -> 0.715 -> 0.661 |
| knob_03 | em_pool_recall_drops | em | `aggregated.macro_pool_recall` | down | NaN | NaN | NaN | NaN | NaN | qualitative | [!!] | [ok] | metric missing at levels: baseline, easy, medium, hard |
| knob_03 | em_monotone_drop | em | `aggregated.macro_f1_vs_pool` | down | NaN | NaN | NaN | NaN | NaN | qualitative | [!!] | [ok] | metric missing at levels: baseline, easy, medium, hard |
| knob_03 | fusion_monotone_drop | fusion | `aggregated.overall_accuracy` | down | 0.747 | 0.742 | 0.715 | 0.661 | -0.086 | qualitative | [ok] | [ok] | weakly down: 0.747 -> 0.742 -> 0.715 -> 0.661 |
| knob_04 | em_flat_or_shift | em | `aggregated.macro_f1_vs_pool` | flat | NaN | NaN | NaN | NaN | NaN | qualitative | [!!] | [ok] | metric missing at levels: baseline, easy, medium, hard |
| knob_04 | fusion_monotone_drop | fusion | `aggregated.overall_accuracy` | down | 0.747 | 0.742 | 0.715 | 0.661 | -0.086 | qualitative | [ok] | [ok] | weakly down: 0.747 -> 0.742 -> 0.715 -> 0.661 |
| knob_04 | fusion_spread_widens | fusion | `aggregated.overall_spread` | up | 0.132 | 0.132 | 0.134 | 0.097 | -0.035 | qualitative | [!!] | [ok] | not weakly up: 0.132 -> 0.132 -> 0.134 -> 0.097 |
| knob_05 | sm_flat | sm | `aggregated.macro_f1` | flat | 0.727 | 0.861 | 0.748 | 0.678 | -0.049 | qualitative | [!!] | [ok] | not weakly flat: 0.727 -> 0.861 -> 0.748 -> 0.678 |
| knob_05 | em_monotone_drop | em | `aggregated.macro_f1_vs_pool` | down | NaN | NaN | NaN | NaN | NaN | qualitative | [!!] | [ok] | metric missing at levels: baseline, easy, medium, hard |
| knob_05 | em_pool_recall_drops_for_lexical | em | `per_member.standard_rule.metrics.pool_recall` | down | NaN | NaN | NaN | NaN | NaN | qualitative | [!!] | [ok] | metric missing at levels: baseline, easy, medium, hard |
| knob_05 | fusion_spread_is_signal | fusion | `aggregated.overall_spread` | up | 0.132 | 0.132 | 0.134 | 0.097 | -0.035 | qualitative | [!!] | [ok] | not weakly up: 0.132 -> 0.132 -> 0.134 -> 0.097 |
| knob_05 | fusion_monotone_drop | fusion | `aggregated.overall_accuracy` | down | 0.747 | 0.742 | 0.715 | 0.661 | -0.086 | qualitative | [ok] | [ok] | weakly down: 0.747 -> 0.742 -> 0.715 -> 0.661 |
| knob_06 | sm_flat | sm | `aggregated.macro_f1` | flat | 0.727 | 0.861 | 0.748 | 0.678 | -0.049 | qualitative | [!!] | [ok] | not weakly flat: 0.727 -> 0.861 -> 0.748 -> 0.678 |
| knob_06 | em_monotone_drop | em | `aggregated.macro_f1_vs_pool` | down | NaN | NaN | NaN | NaN | NaN | qualitative | [!!] | [ok] | metric missing at levels: baseline, easy, medium, hard |
| knob_06 | em_pool_recall_drops | em | `aggregated.macro_pool_recall` | down | NaN | NaN | NaN | NaN | NaN | qualitative | [!!] | [ok] | metric missing at levels: baseline, easy, medium, hard |
| knob_06 | fusion_monotone_drop | fusion | `aggregated.overall_accuracy` | down | 0.747 | 0.742 | 0.715 | 0.661 | -0.086 | qualitative | [ok] | [ok] | weakly down: 0.747 -> 0.742 -> 0.715 -> 0.661 |
| knob_08 | sm_monotone_drop | sm | `aggregated.macro_f1` | down | 0.727 | 0.861 | 0.748 | 0.678 | -0.049 | qualitative | [!!] | [ok] | not weakly down: 0.727 -> 0.861 -> 0.748 -> 0.678 |
| knob_08 | sm_label_collapses | sm | `per_member.label_jaccard.metrics.f1` | down | NaN | NaN | NaN | NaN | NaN | qualitative | [!!] | [ok] | metric missing at levels: baseline, easy, medium, hard |
| knob_08 | sm_spread_is_signal | sm | `spread:per_member.llm_openai.metrics.f1:per_member.label_jaccard.metrics.f1` | up | NaN | NaN | NaN | NaN | NaN | qualitative | [!!] | [ok] | metric missing at levels: baseline, easy, medium, hard |
| knob_08 | sm_instance_steady | sm | `per_member.instance_tfidf_cosine.metrics.f1` | flat | NaN | NaN | NaN | NaN | NaN | qualitative | [!!] | [ok] | metric missing at levels: baseline, easy, medium, hard |
| knob_08 | em_flat | em | `aggregated.macro_f1_vs_pool` | flat | NaN | NaN | NaN | NaN | NaN | qualitative | [!!] | [ok] | metric missing at levels: baseline, easy, medium, hard |
| knob_08 | fusion_flat | fusion | `aggregated.overall_accuracy` | flat | 0.747 | 0.742 | 0.715 | 0.661 | -0.086 | qualitative | [!!] | [ok] | not weakly flat: 0.747 -> 0.742 -> 0.715 -> 0.661 |
| knob_10 | sm_flat | sm | `aggregated.macro_f1` | flat | 0.727 | 0.861 | 0.748 | 0.678 | -0.049 | qualitative | [!!] | [ok] | not weakly flat: 0.727 -> 0.861 -> 0.748 -> 0.678 |
| knob_10 | em_flat | em | `aggregated.macro_f1_vs_pool` | flat | NaN | NaN | NaN | NaN | NaN | qualitative | [!!] | [ok] | metric missing at levels: baseline, easy, medium, hard |
| knob_10 | fusion_monotone_drop | fusion | `aggregated.overall_accuracy` | down | 0.747 | 0.742 | 0.715 | 0.661 | -0.086 | qualitative | [ok] | [ok] | weakly down: 0.747 -> 0.742 -> 0.715 -> 0.661 |
| knob_10 | fusion_spread_widens | fusion | `aggregated.overall_spread` | up | 0.132 | 0.132 | 0.134 | 0.097 | -0.035 | qualitative | [!!] | [ok] | not weakly up: 0.132 -> 0.132 -> 0.134 -> 0.097 |
| knob_10 | fusion_voting_drops_faster_than_trust_on_name | fusion | `spread:per_attribute.name.prefer_higher_trust:per_attribute.name.voting` | up | 0.080 | 0.050 | 0.060 | -0.023 | -0.103 | qualitative | [!!] | [ok] | not weakly up: 0.080 -> 0.050 -> 0.060 -> -0.023 |

## Collapses

No members fell below the collapse threshold.

## Best-Member Ceiling (P8)

Per-stage best-member F1 across baseline -> easy -> medium -> hard. A valid difficulty signal must depress the *ceiling* (the user-attainable member), not just the committee mean. A flat / rising ceiling means the user-selected matcher never sees the synthetic difficulty (committee-mean drift can be masked by weak-member degradation alone).

| Stage | baseline | easy | medium | hard | delta | non-increasing | winner trail | Reason |
|---|---|---|---|---|---|---|---|---|
| sm | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | [ok] | `llm_openai -> llm_openai -> llm_openai -> llm_openai` | best-member ceiling non-increasing: 1.000 -> 1.000 -> 1.000 -> 1.000  (llm_openai -> llm_openai -> llm_openai -> llm_openai) |
| norm | 0.975 | 0.951 | 0.938 | 0.943 | -0.032 | [!!] | `date_iso -> date_iso -> date_iso -> date_iso` | best-member ceiling did NOT decline: 0.975 -> 0.951 -> 0.938 -> 0.943  (date_iso -> date_iso -> date_iso -> date_iso) — difficulty dial may be invisible to the user-selected matcher |
| em_blocking | 1.000 | 1.000 | 1.000 | 0.560 | -0.440 | [ok] | `token_blocker -> bm25_blocker -> embedding_blocker -> token_blocker` | best-member ceiling non-increasing: 1.000 -> 1.000 -> 1.000 -> 0.560  (token_blocker -> bm25_blocker -> embedding_blocker -> token_blocker) |
| em_matching | 0.771 | 0.865 | 0.808 | 0.647 | -0.124 | [!!] | `comem -> magellan -> magellan -> magellan` | best-member ceiling did NOT decline: 0.771 -> 0.865 -> 0.808 -> 0.647  (comem -> magellan -> magellan -> magellan) — difficulty dial may be invisible to the user-selected matcher |
| fusion | 0.728 | 0.717 | 0.688 | 0.648 | -0.080 | [ok] | `genres_union -> genres_union -> genres_union -> genres_prefer_higher_trust` | best-member ceiling non-increasing: 0.728 -> 0.717 -> 0.688 -> 0.648  (genres_union -> genres_union -> genres_union -> genres_prefer_higher_trust) |

## Open Questions

Signals that are direction-correct but magnitude-unspecified by the knob card. M10 should decide whether the measured delta is 'strong enough' to count as validation.

| Knob | Signal | Stage | observed delta | Card notes |
|---|---|---|---|---|
| knob_03 | fusion_monotone_drop | fusion | -0.086 | Fusion accuracy drops monotonically given conflict-preserving constraint and survivor cap. |
| knob_04 | fusion_monotone_drop | fusion | -0.086 | Primary target. Voting-family strategies degrade faster than trust-weighted / single-best-source strategies. |
| knob_05 | fusion_monotone_drop | fusion | -0.086 | Monotone drop for naive strategies. Best-strategy accuracy (the max across strategies) may hold if a canonicalizing strategy is available, but overall_accuracy averages all strategies and therefore drops. |
| knob_06 | fusion_monotone_drop | fusion | -0.086 | Monotone drop given survivor floors. Provenance-aware fusers should be rewarded over cell-local. |
| knob_10 | fusion_monotone_drop | fusion | -0.086 | Primary target. Cell-local strategies (voting, most_complete, per-attribute trust) lose accuracy as unreliability grows. |

## Provenance

- Domain: games
- Expectations: `usecases_synthetic/config/knob_expected_signals.yaml`
- Baseline: `usecases_synthetic/baselines/games/baseline_metrics.json`
- Per-level metrics: `usecases_synthetic/validation/games/<level>/metrics.json`
