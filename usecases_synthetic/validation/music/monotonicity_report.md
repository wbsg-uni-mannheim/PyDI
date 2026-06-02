# Monotonicity + Collapse Report — music

Cross-level analysis of easy/medium/hard variants against the baseline. See `knobs/cross_cutting.md` for the protocol.

## Cumulative Cross-Level Slope (load-bearing verdict)

Per-stage committee metric across the **cumulative** variant levels (every knob on at each level). This is the C2-contract verdict: committee scores should weakly decrease easy -> medium -> hard, and baseline (a reference value) should land no harder than medium. It makes no single-knob isolation assumption, so this -- not the per-knob signals below -- is the headline difficulty verdict.

| Stage | Metric | baseline | easy | medium | hard | easy->hard | Mono (e>=m>=h) | BasePos |
|---|---|---|---|---|---|---|---|---|
| sm | `aggregated.macro_f1` | 0.876 | 0.900 | 0.762 | 0.678 | -0.222 | [ok] | [ok] |
| norm | `aggregated.macro_f1` | 0.658 | 0.618 | 0.580 | 0.532 | -0.086 | [ok] | [ok] |
| em_blocking | `aggregated.macro_pair_recall` | 0.950 | 0.967 | 0.943 | 0.932 | -0.035 | [ok] | [ok] |
| em_matching | `aggregated.macro_f1_variant_model_on_regen_test` | 0.832 | 0.890 | 0.842 | 0.856 | -0.034 | [!!] | [!!] |
| fusion | `aggregated.overall_accuracy` | 0.893 | 0.866 | 0.765 | 0.566 | -0.300 | [ok] | [ok] |

## Per-Knob Expected Signals (indicative)

> These are **per-knob** expectations from `knob_expected_signals.yaml`, evaluated against the **cumulative** variants (every knob on at each level). They cannot isolate one knob, so a `flat` expectation for a stage that *another* knob also drives reads `[!!]` by construction (e.g. SM is not flat for knob_01 because K8 naming is also on). Treat this as indicative of combined effect; the load-bearing verdict is the Cumulative Cross-Level Slope above. For true per-knob isolation, run `generate_variant --only-knob <K>` ablations.

| Knob | Signals | OK direction | OK range | Notes |
|---|---|---|---|---|
| knob_01 | 3 | 0/3 [!!] | 0/— | [card](knobs/knob_01_surface_augmentation.md) |
| knob_02 | 3 | 0/3 [!!] | 0/— | [card](knobs/knob_02_niche_density.md) |
| knob_03 | 3 | 2/3 [!!] | 0/— | [card](knobs/knob_03_attribute_drop.md) |
| knob_04 | 3 | 2/3 [!!] | 0/— | [card](knobs/knob_04_coverage_skew.md) |
| knob_05 | 5 | 2/5 [!!] | 0/— | [card](knobs/knob_05_format_unit.md) |
| knob_06 | 4 | 2/4 [!!] | 0/— | [card](knobs/knob_06_value_noise.md) |
| knob_08 | 6 | 4/6 [!!] | 0/— | [card](knobs/knob_08_schema_naming.md) |
| knob_10 | 5 | 2/5 [!!] | 0/— | [card](knobs/knob_10_source_reliability.md) |

## Per-Signal Results

| Knob | Signal | Stage | Metric | Dir | baseline | easy | medium | hard | delta | target | Mono | Rng | BasePos | Reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| knob_01 | sm_flat | sm | `aggregated.macro_f1` | flat | 0.876 | 0.900 | 0.762 | 0.678 | -0.198 | qualitative | [!!] | [ok] | [!!] | not weakly flat: 0.900 -> 0.762 -> 0.678 |
| knob_01 | em_monotone_drop | em_matching | `aggregated.macro_f1_variant_model_on_regen_test` | down | 0.832 | 0.890 | 0.842 | 0.856 | 0.023 | qualitative | [!!] | [ok] | [!!] | not weakly down: 0.890 -> 0.842 -> 0.856 |
| knob_01 | em_blocker_spread | em_blocking | `spread:per_member.standard_blocker.metrics.pair_recall:per_member.embedding_blocker.metrics.pair_recall` | down | -0.138 | -0.087 | -0.152 | -0.130 | 0.008 | qualitative | [!!] | [ok] | [ok] | not weakly down: -0.087 -> -0.152 -> -0.130 |
| knob_02 | em_monotone_drop | em_matching | `aggregated.macro_f1_variant_model_on_regen_test` | down | 0.832 | 0.890 | 0.842 | 0.856 | 0.023 | qualitative | [!!] | [ok] | [!!] | not weakly down: 0.890 -> 0.842 -> 0.856 |
| knob_02 | em_pool_precision_tightens | em_matching | `aggregated.macro_precision` | down | 0.992 | 0.994 | 0.990 | 0.991 | -0.001 | qualitative | [!!] | [ok] | [ok] | not weakly down: 0.994 -> 0.990 -> 0.991 |
| knob_02 | fusion_flat | fusion | `aggregated.overall_accuracy` | flat | 0.893 | 0.866 | 0.765 | 0.566 | -0.327 | qualitative | [!!] | [ok] | [!!] | not weakly flat: 0.866 -> 0.765 -> 0.566 |
| knob_03 | em_pool_recall_drops | em_blocking | `aggregated.macro_pair_recall` | down | 0.950 | 0.967 | 0.943 | 0.932 | -0.018 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.967 -> 0.943 -> 0.932 |
| knob_03 | em_monotone_drop | em_matching | `aggregated.macro_f1_variant_model_on_regen_test` | down | 0.832 | 0.890 | 0.842 | 0.856 | 0.023 | qualitative | [!!] | [ok] | [!!] | not weakly down: 0.890 -> 0.842 -> 0.856 |
| knob_03 | fusion_monotone_drop | fusion | `aggregated.overall_accuracy` | down | 0.893 | 0.866 | 0.765 | 0.566 | -0.327 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.866 -> 0.765 -> 0.566 |
| knob_04 | em_flat_or_shift | em_matching | `aggregated.macro_f1_variant_model_on_regen_test` | flat | 0.832 | 0.890 | 0.842 | 0.856 | 0.023 | qualitative | [ok] | [ok] | [ok] | weakly flat: 0.890 -> 0.842 -> 0.856 |
| knob_04 | fusion_monotone_drop | fusion | `aggregated.overall_accuracy` | down | 0.893 | 0.866 | 0.765 | 0.566 | -0.327 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.866 -> 0.765 -> 0.566 |
| knob_04 | fusion_spread_widens | fusion | `spread:aggregated.max_accuracy:aggregated.min_accuracy` | up | 0.080 | 0.094 | 0.107 | 0.042 | -0.038 | qualitative | [!!] | [ok] | [ok] | not weakly up: 0.094 -> 0.107 -> 0.042 |
| knob_05 | sm_flat | sm | `aggregated.macro_f1` | flat | 0.876 | 0.900 | 0.762 | 0.678 | -0.198 | qualitative | [!!] | [ok] | [!!] | not weakly flat: 0.900 -> 0.762 -> 0.678 |
| knob_05 | em_monotone_drop | em_matching | `aggregated.macro_f1_variant_model_on_regen_test` | down | 0.832 | 0.890 | 0.842 | 0.856 | 0.023 | qualitative | [!!] | [ok] | [!!] | not weakly down: 0.890 -> 0.842 -> 0.856 |
| knob_05 | em_pool_recall_drops_for_lexical | em_blocking | `per_member.standard_blocker.metrics.pair_recall` | down | 0.859 | 0.911 | 0.842 | 0.842 | -0.017 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.911 -> 0.842 -> 0.842 |
| knob_05 | fusion_spread_is_signal | fusion | `spread:aggregated.max_accuracy:aggregated.min_accuracy` | up | 0.080 | 0.094 | 0.107 | 0.042 | -0.038 | qualitative | [!!] | [ok] | [ok] | not weakly up: 0.094 -> 0.107 -> 0.042 |
| knob_05 | fusion_monotone_drop | fusion | `aggregated.overall_accuracy` | down | 0.893 | 0.866 | 0.765 | 0.566 | -0.327 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.866 -> 0.765 -> 0.566 |
| knob_06 | sm_flat | sm | `aggregated.macro_f1` | flat | 0.876 | 0.900 | 0.762 | 0.678 | -0.198 | qualitative | [!!] | [ok] | [!!] | not weakly flat: 0.900 -> 0.762 -> 0.678 |
| knob_06 | em_monotone_drop | em_matching | `aggregated.macro_f1_variant_model_on_regen_test` | down | 0.832 | 0.890 | 0.842 | 0.856 | 0.023 | qualitative | [!!] | [ok] | [!!] | not weakly down: 0.890 -> 0.842 -> 0.856 |
| knob_06 | em_pool_recall_drops | em_blocking | `aggregated.macro_pair_recall` | down | 0.950 | 0.967 | 0.943 | 0.932 | -0.018 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.967 -> 0.943 -> 0.932 |
| knob_06 | fusion_monotone_drop | fusion | `aggregated.overall_accuracy` | down | 0.893 | 0.866 | 0.765 | 0.566 | -0.327 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.866 -> 0.765 -> 0.566 |
| knob_08 | sm_monotone_drop | sm | `aggregated.macro_f1` | down | 0.876 | 0.900 | 0.762 | 0.678 | -0.198 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.900 -> 0.762 -> 0.678 |
| knob_08 | sm_label_collapses | sm | `per_member.label_jw.metrics.f1` | down | 1.000 | 1.000 | 0.698 | 0.364 | -0.636 | qualitative | [ok] | [ok] | [ok] | weakly down: 1.000 -> 0.698 -> 0.364 |
| knob_08 | sm_spread_is_signal | sm | `spread:per_member.llm_openai.metrics.f1:per_member.label_jw.metrics.f1` | up | 0.000 | 0.000 | 0.302 | 0.556 | 0.556 | qualitative | [ok] | [ok] | [ok] | weakly up: 0.000 -> 0.302 -> 0.556 |
| knob_08 | sm_instance_steady | sm | `per_member.instance_tf_cosine.metrics.f1` | flat | 0.638 | 0.731 | 0.609 | 0.605 | -0.034 | qualitative | [!!] | [ok] | [ok] | not weakly flat: 0.731 -> 0.609 -> 0.605 |
| knob_08 | em_flat | em_matching | `aggregated.macro_f1_variant_model_on_regen_test` | flat | 0.832 | 0.890 | 0.842 | 0.856 | 0.023 | qualitative | [ok] | [ok] | [ok] | weakly flat: 0.890 -> 0.842 -> 0.856 |
| knob_08 | fusion_flat | fusion | `aggregated.overall_accuracy` | flat | 0.893 | 0.866 | 0.765 | 0.566 | -0.327 | qualitative | [!!] | [ok] | [!!] | not weakly flat: 0.866 -> 0.765 -> 0.566 |
| knob_10 | sm_flat | sm | `aggregated.macro_f1` | flat | 0.876 | 0.900 | 0.762 | 0.678 | -0.198 | qualitative | [!!] | [ok] | [!!] | not weakly flat: 0.900 -> 0.762 -> 0.678 |
| knob_10 | em_flat | em_matching | `aggregated.macro_f1_variant_model_on_regen_test` | flat | 0.832 | 0.890 | 0.842 | 0.856 | 0.023 | qualitative | [ok] | [ok] | [ok] | weakly flat: 0.890 -> 0.842 -> 0.856 |
| knob_10 | fusion_monotone_drop | fusion | `aggregated.overall_accuracy` | down | 0.893 | 0.866 | 0.765 | 0.566 | -0.327 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.866 -> 0.765 -> 0.566 |
| knob_10 | fusion_spread_widens | fusion | `spread:aggregated.max_accuracy:aggregated.min_accuracy` | up | 0.080 | 0.094 | 0.107 | 0.042 | -0.038 | qualitative | [!!] | [ok] | [ok] | not weakly up: 0.094 -> 0.107 -> 0.042 |
| knob_10 | fusion_voting_drops_faster_than_trust_on_name | fusion | `spread:per_attribute.name.prefer_higher_trust_only:per_attribute.name.voting_only` | up | -0.010 | 0.010 | -0.010 | 0.000 | 0.010 | qualitative | [!!] | [ok] | [ok] | not weakly up: 0.010 -> -0.010 -> 0.000 |

## Collapses

Members with F1 < 0.15 or drop > 0.5 from baseline. EM collapses are classified against the pool-agreement diagnostic (`hidden_positive_noise` means pool is stable while test-gold moved — see `knobs/cross_cutting.md` § Protection set semantics).

| Level | Stage | Member | baseline F1 | measured F1 | delta | classification | pool delta | action |
|---|---|---|---|---|---|---|---|---|
| hard | sm | `label_jw` | 1.000 | 0.364 | -0.636 | unknown | — | see knobs/cross_cutting.md § Per-knob fix-strategy defaults (SM row) |

## Best-Member Ceiling (P8)

Per-stage best-member F1 across baseline -> easy -> medium -> hard. A valid difficulty signal must depress the *ceiling* (the user-attainable member), not just the committee mean. A flat / rising ceiling means the user-selected matcher never sees the synthetic difficulty (committee-mean drift can be masked by weak-member degradation alone).

| Stage | baseline | easy | medium | hard | delta | non-increasing | winner trail | Reason |
|---|---|---|---|---|---|---|---|---|
| sm | 1.000 | 1.000 | 1.000 | 0.920 | -0.080 | [ok] | `label_jw -> label_jw -> llm_openai -> llm_openai` | best-member ceiling non-increasing: 1.000 -> 1.000 -> 1.000 -> 0.920  (label_jw -> label_jw -> llm_openai -> llm_openai) |
| norm | 0.698 | 0.648 | 0.609 | 0.569 | -0.129 | [ok] | `passthrough -> passthrough -> passthrough -> passthrough` | best-member ceiling non-increasing: 0.698 -> 0.648 -> 0.609 -> 0.569  (passthrough -> passthrough -> passthrough -> passthrough) |
| em_blocking | 1.000 | 1.000 | 1.000 | 0.996 | -0.004 | [ok] | `token_blocker -> token_blocker -> sc_block -> sc_block` | best-member ceiling non-increasing: 1.000 -> 1.000 -> 1.000 -> 0.996  (token_blocker -> token_blocker -> sc_block -> sc_block) |
| em_matching | 0.990 | 0.960 | 0.972 | 0.970 | -0.021 | [!!] | `ditto_plm -> ditto_plm -> ditto_plm -> ditto_plm` | best-member ceiling did NOT decline: 0.990 -> 0.960 -> 0.972 -> 0.970  (ditto_plm -> ditto_plm -> ditto_plm -> ditto_plm) — difficulty dial may be invisible to the user-selected matcher |
| fusion | 0.893 | 0.866 | 0.765 | 0.566 | -0.327 | [ok] | `prefer_higher_trust_only -> truthfinder_only -> truthfinder_only -> llm_only` | best-member ceiling non-increasing: 0.893 -> 0.866 -> 0.765 -> 0.566  (prefer_higher_trust_only -> truthfinder_only -> truthfinder_only -> llm_only) |

## Open Questions

Signals that are direction-correct but magnitude-unspecified by the knob card. M10 should decide whether the measured delta is 'strong enough' to count as validation.

| Knob | Signal | Stage | observed delta | Card notes |
|---|---|---|---|---|
| knob_03 | em_pool_recall_drops | em_blocking | -0.018 | Candidate recall drops monotonically as drop_rate_key rises. Sharper for single-key blockers. |
| knob_03 | fusion_monotone_drop | fusion | -0.327 | Fusion accuracy drops monotonically given conflict-preserving constraint and survivor cap. |
| knob_04 | em_flat_or_shift | em_matching | 0.023 | EM mostly indifferent; any shift is through the singleton-prior, direction unspecified by the card. |
| knob_04 | fusion_monotone_drop | fusion | -0.327 | Primary target. Voting-family strategies degrade faster than trust-weighted / single-best-source strategies. |
| knob_05 | em_pool_recall_drops_for_lexical | em_blocking | -0.017 | Raw-string blocker (lexical+rule) sees a monotone pool_recall drop; embedding+rule blocker holds steadier. |
| knob_05 | fusion_monotone_drop | fusion | -0.327 | Monotone drop for naive strategies. Best-strategy accuracy (the max across strategies) may hold if a canonicalizing strategy is available, but overall_accuracy averages all strategies and therefore drops. |
| knob_06 | em_pool_recall_drops | em_blocking | -0.018 | Lexical/n-gram blockers lose recall sharply; embedding blockers mildly. |
| knob_06 | fusion_monotone_drop | fusion | -0.327 | Monotone drop given survivor floors. Provenance-aware fusers should be rewarded over cell-local. |
| knob_08 | sm_monotone_drop | sm | -0.198 | Primary target. Monotone drop expected. |
| knob_08 | sm_label_collapses | sm | -0.636 | Label-based string-similarity matchers collapse fast on cryptic/anonymized names. |
| knob_08 | sm_spread_is_signal | sm | 0.556 | Per the card, "the spread between matcher types IS the K8 difficulty signal". Instance-based / embedding / LLM matchers hold while label-based collapse. Spread (llm_openai.f1 - label_jaccard.f1) widens monotonically. |
| knob_08 | em_flat | em_matching | 0.023 | Downstream stages unaffected by K8. |
| knob_10 | em_flat | em_matching | 0.023 | Upstream stages unaffected by K10. |
| knob_10 | fusion_monotone_drop | fusion | -0.327 | Primary target. Cell-local strategies (voting, most_complete, per-attribute trust) lose accuracy as unreliability grows. |

## Provenance

- Domain: music
- Expectations: `usecases_synthetic/config/knob_expected_signals.yaml`
- Baseline: `usecases_synthetic/baselines/music/baseline_metrics.json`
- Per-level metrics: `usecases_synthetic/validation/music/<level>/metrics.json`
