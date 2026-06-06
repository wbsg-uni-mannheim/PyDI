# Monotonicity + Collapse Report — companies

Cross-level analysis of easy/medium/hard variants against the baseline. See `knobs/cross_cutting.md` for the protocol.

## Cumulative Cross-Level Slope (load-bearing verdict)

Per-stage committee metric across the **cumulative** variant levels (every knob on at each level). This is the C2-contract verdict: committee scores should weakly decrease easy -> medium -> hard, and baseline (a reference value) should land no harder than medium. It makes no single-knob isolation assumption, so this -- not the per-knob signals below -- is the headline difficulty verdict.

| Stage | Metric | baseline | easy | medium | hard | easy->hard | Mono (e>=m>=h) | BasePos |
|---|---|---|---|---|---|---|---|---|
| sm | `aggregated.macro_f1` | 0.727 | 0.862 | 0.770 | 0.646 | -0.216 | [ok] | [!!] |
| norm | `aggregated.macro_f1` | 0.871 | 0.609 | 0.719 | 0.725 | 0.116 | [!!] | [ok] |
| em_blocking | `aggregated.macro_pair_recall` | 0.988 | 0.983 | 0.982 | 0.984 | 0.001 | [ok] | [ok] |
| em_matching | `aggregated.macro_f1_variant_model_on_regen_test` | 0.884 | 0.907 | 0.872 | 0.874 | -0.033 | [ok] | [ok] |
| fusion | `aggregated.overall_accuracy` | 0.458 | 0.400 | 0.413 | 0.348 | -0.052 | [!!] | [ok] |

## Per-Knob Expected Signals (indicative)

> These are **per-knob** expectations from `knob_expected_signals.yaml`, evaluated against the **cumulative** variants (every knob on at each level). They cannot isolate one knob, so a `flat` expectation for a stage that *another* knob also drives reads `[!!]` by construction (e.g. SM is not flat for knob_01 because K8 naming is also on). Treat this as indicative of combined effect; the load-bearing verdict is the Cumulative Cross-Level Slope above. For true per-knob isolation, run `generate_variant --only-knob <K>` ablations.

| Knob | Signals | OK direction | OK range | Notes |
|---|---|---|---|---|
| knob_01 | 4 | 1/4 [!!] | 0/— | [card](knobs/knob_01_surface_augmentation.md) |
| knob_02 | 4 | 1/4 [!!] | 0/— | [card](knobs/knob_02_niche_density.md) |
| knob_03 | 4 | 1/4 [!!] | 0/— | [card](knobs/knob_03_attribute_drop.md) |
| knob_04 | 4 | 2/4 [!!] | 0/— | [card](knobs/knob_04_coverage_skew.md) |
| knob_05 | 6 | 1/6 [!!] | 0/— | [card](knobs/knob_05_format_unit.md) |
| knob_06 | 5 | 1/5 [!!] | 0/— | [card](knobs/knob_06_value_noise.md) |
| knob_08 | 7 | 6/7 [!!] | 0/— | [card](knobs/knob_08_schema_naming.md) |
| knob_10 | 6 | 2/6 [!!] | 0/— | [card](knobs/knob_10_source_reliability.md) |

## Per-Signal Results

| Knob | Signal | Stage | Metric | Dir | baseline | easy | medium | hard | delta | target | Mono | Rng | BasePos | Reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| knob_01 | sm_flat | sm | `aggregated.macro_f1` | flat | 0.727 | 0.862 | 0.770 | 0.646 | -0.081 | qualitative | [!!] | [ok] | [ok] | not weakly flat: 0.862 -> 0.770 -> 0.646 |
| knob_01 | em_monotone_drop | em_matching | `aggregated.macro_f1_baseline_model_on_regen_test` | down | 0.884 | 0.899 | 0.872 | 0.887 | 0.003 | qualitative | [!!] | [ok] | [ok] | not weakly down: 0.899 -> 0.872 -> 0.887 |
| knob_01 | em_blocker_spread | em_blocking | `spread:per_member.standard_blocker.metrics.pair_recall:per_member.embedding_blocker.metrics.pair_recall` | down | -0.027 | -0.034 | -0.039 | -0.030 | -0.004 | qualitative | [!!] | [ok] | [ok] | not weakly down: -0.034 -> -0.039 -> -0.030 |
| knob_01 | em_monotone_drop_pruned | em_matching | `aggregated.macro_f1_baseline_model_on_baseline_test` | down | 0.884 | 0.899 | 0.872 | 0.868 | -0.016 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.899 -> 0.872 -> 0.868 |
| knob_02 | em_monotone_drop | em_matching | `aggregated.macro_f1_baseline_model_on_regen_test` | down | 0.884 | 0.899 | 0.872 | 0.887 | 0.003 | qualitative | [!!] | [ok] | [ok] | not weakly down: 0.899 -> 0.872 -> 0.887 |
| knob_02 | em_pool_precision_tightens | em_matching | `aggregated.macro_precision` | down | 0.888 | 0.922 | 0.891 | 0.900 | 0.012 | qualitative | [!!] | [ok] | [!!] | not weakly down: 0.922 -> 0.891 -> 0.900 |
| knob_02 | fusion_flat | fusion | `aggregated.overall_accuracy` | flat | 0.458 | 0.400 | 0.413 | 0.348 | -0.109 | qualitative | [!!] | [ok] | [ok] | not weakly flat: 0.400 -> 0.413 -> 0.348 |
| knob_02 | em_monotone_drop_pruned | em_matching | `aggregated.macro_f1_baseline_model_on_baseline_test` | down | 0.884 | 0.899 | 0.872 | 0.868 | -0.016 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.899 -> 0.872 -> 0.868 |
| knob_03 | em_pool_recall_drops | em_blocking | `aggregated.macro_pair_recall` | down | 0.988 | 0.983 | 0.982 | 0.984 | -0.004 | qualitative | [!!] | [ok] | [ok] | not weakly down: 0.983 -> 0.982 -> 0.984 |
| knob_03 | em_monotone_drop | em_matching | `aggregated.macro_f1_baseline_model_on_regen_test` | down | 0.884 | 0.899 | 0.872 | 0.887 | 0.003 | qualitative | [!!] | [ok] | [ok] | not weakly down: 0.899 -> 0.872 -> 0.887 |
| knob_03 | fusion_monotone_drop | fusion | `aggregated.overall_accuracy` | down | 0.458 | 0.400 | 0.413 | 0.348 | -0.109 | qualitative | [!!] | [ok] | [ok] | not weakly down: 0.400 -> 0.413 -> 0.348 |
| knob_03 | em_monotone_drop_pruned | em_matching | `aggregated.macro_f1_baseline_model_on_baseline_test` | down | 0.884 | 0.899 | 0.872 | 0.868 | -0.016 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.899 -> 0.872 -> 0.868 |
| knob_04 | em_flat_or_shift | em_matching | `aggregated.macro_f1_baseline_model_on_regen_test` | flat | 0.884 | 0.899 | 0.872 | 0.887 | 0.003 | qualitative | [ok] | [ok] | [ok] | weakly flat: 0.899 -> 0.872 -> 0.887 |
| knob_04 | fusion_monotone_drop | fusion | `aggregated.overall_accuracy` | down | 0.458 | 0.400 | 0.413 | 0.348 | -0.109 | qualitative | [!!] | [ok] | [ok] | not weakly down: 0.400 -> 0.413 -> 0.348 |
| knob_04 | fusion_spread_widens | fusion | `spread:aggregated.max_accuracy:aggregated.min_accuracy` | up | 0.104 | 0.071 | 0.056 | 0.043 | -0.061 | qualitative | [!!] | [ok] | [!!] | not weakly up: 0.071 -> 0.056 -> 0.043 |
| knob_04 | em_flat_or_shift_pruned | em_matching | `aggregated.macro_f1_baseline_model_on_baseline_test` | flat | 0.884 | 0.899 | 0.872 | 0.868 | -0.016 | qualitative | [ok] | [ok] | [ok] | weakly flat: 0.899 -> 0.872 -> 0.868 |
| knob_05 | sm_flat | sm | `aggregated.macro_f1` | flat | 0.727 | 0.862 | 0.770 | 0.646 | -0.081 | qualitative | [!!] | [ok] | [ok] | not weakly flat: 0.862 -> 0.770 -> 0.646 |
| knob_05 | em_monotone_drop | em_matching | `aggregated.macro_f1_baseline_model_on_regen_test` | down | 0.884 | 0.899 | 0.872 | 0.887 | 0.003 | qualitative | [!!] | [ok] | [ok] | not weakly down: 0.899 -> 0.872 -> 0.887 |
| knob_05 | em_pool_recall_drops_for_lexical | em_blocking | `per_member.standard_blocker.metrics.pair_recall` | down | 0.973 | 0.966 | 0.961 | 0.970 | -0.004 | qualitative | [!!] | [ok] | [ok] | not weakly down: 0.966 -> 0.961 -> 0.970 |
| knob_05 | fusion_spread_is_signal | fusion | `spread:aggregated.max_accuracy:aggregated.min_accuracy` | up | 0.104 | 0.071 | 0.056 | 0.043 | -0.061 | qualitative | [!!] | [ok] | [!!] | not weakly up: 0.071 -> 0.056 -> 0.043 |
| knob_05 | fusion_monotone_drop | fusion | `aggregated.overall_accuracy` | down | 0.458 | 0.400 | 0.413 | 0.348 | -0.109 | qualitative | [!!] | [ok] | [ok] | not weakly down: 0.400 -> 0.413 -> 0.348 |
| knob_05 | em_monotone_drop_pruned | em_matching | `aggregated.macro_f1_baseline_model_on_baseline_test` | down | 0.884 | 0.899 | 0.872 | 0.868 | -0.016 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.899 -> 0.872 -> 0.868 |
| knob_06 | sm_flat | sm | `aggregated.macro_f1` | flat | 0.727 | 0.862 | 0.770 | 0.646 | -0.081 | qualitative | [!!] | [ok] | [ok] | not weakly flat: 0.862 -> 0.770 -> 0.646 |
| knob_06 | em_monotone_drop | em_matching | `aggregated.macro_f1_baseline_model_on_regen_test` | down | 0.884 | 0.899 | 0.872 | 0.887 | 0.003 | qualitative | [!!] | [ok] | [ok] | not weakly down: 0.899 -> 0.872 -> 0.887 |
| knob_06 | em_pool_recall_drops | em_blocking | `aggregated.macro_pair_recall` | down | 0.988 | 0.983 | 0.982 | 0.984 | -0.004 | qualitative | [!!] | [ok] | [ok] | not weakly down: 0.983 -> 0.982 -> 0.984 |
| knob_06 | fusion_monotone_drop | fusion | `aggregated.overall_accuracy` | down | 0.458 | 0.400 | 0.413 | 0.348 | -0.109 | qualitative | [!!] | [ok] | [ok] | not weakly down: 0.400 -> 0.413 -> 0.348 |
| knob_06 | em_monotone_drop_pruned | em_matching | `aggregated.macro_f1_baseline_model_on_baseline_test` | down | 0.884 | 0.899 | 0.872 | 0.868 | -0.016 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.899 -> 0.872 -> 0.868 |
| knob_08 | sm_monotone_drop | sm | `aggregated.macro_f1` | down | 0.727 | 0.862 | 0.770 | 0.646 | -0.081 | qualitative | [ok] | [ok] | [!!] | weakly down: 0.862 -> 0.770 -> 0.646 |
| knob_08 | sm_label_collapses | sm | `per_member.label_jw.metrics.f1` | down | 0.385 | 1.000 | 0.706 | 0.320 | -0.065 | qualitative | [ok] | [ok] | [!!] | weakly down: 1.000 -> 0.706 -> 0.320 |
| knob_08 | sm_spread_is_signal | sm | `spread:per_member.llm_openai.metrics.f1:per_member.label_jw.metrics.f1` | up | 0.615 | 0.000 | 0.294 | 0.680 | 0.065 | qualitative | [ok] | [ok] | [!!] | weakly up: 0.000 -> 0.294 -> 0.680 |
| knob_08 | sm_instance_steady | sm | `per_member.instance_tf_cosine.metrics.f1` | flat | 0.629 | 0.562 | 0.562 | 0.562 | -0.066 | qualitative | [ok] | [ok] | [!!] | weakly flat: 0.562 -> 0.562 -> 0.562 |
| knob_08 | em_flat | em_matching | `aggregated.macro_f1_baseline_model_on_regen_test` | flat | 0.884 | 0.899 | 0.872 | 0.887 | 0.003 | qualitative | [ok] | [ok] | [ok] | weakly flat: 0.899 -> 0.872 -> 0.887 |
| knob_08 | fusion_flat | fusion | `aggregated.overall_accuracy` | flat | 0.458 | 0.400 | 0.413 | 0.348 | -0.109 | qualitative | [!!] | [ok] | [ok] | not weakly flat: 0.400 -> 0.413 -> 0.348 |
| knob_08 | em_flat_pruned | em_matching | `aggregated.macro_f1_baseline_model_on_baseline_test` | flat | 0.884 | 0.899 | 0.872 | 0.868 | -0.016 | qualitative | [ok] | [ok] | [ok] | weakly flat: 0.899 -> 0.872 -> 0.868 |
| knob_10 | sm_flat | sm | `aggregated.macro_f1` | flat | 0.727 | 0.862 | 0.770 | 0.646 | -0.081 | qualitative | [!!] | [ok] | [ok] | not weakly flat: 0.862 -> 0.770 -> 0.646 |
| knob_10 | em_flat | em_matching | `aggregated.macro_f1_baseline_model_on_regen_test` | flat | 0.884 | 0.899 | 0.872 | 0.887 | 0.003 | qualitative | [ok] | [ok] | [ok] | weakly flat: 0.899 -> 0.872 -> 0.887 |
| knob_10 | fusion_monotone_drop | fusion | `aggregated.overall_accuracy` | down | 0.458 | 0.400 | 0.413 | 0.348 | -0.109 | qualitative | [!!] | [ok] | [ok] | not weakly down: 0.400 -> 0.413 -> 0.348 |
| knob_10 | fusion_spread_widens | fusion | `spread:aggregated.max_accuracy:aggregated.min_accuracy` | up | 0.104 | 0.071 | 0.056 | 0.043 | -0.061 | qualitative | [!!] | [ok] | [!!] | not weakly up: 0.071 -> 0.056 -> 0.043 |
| knob_10 | fusion_voting_drops_faster_than_trust_on_name | fusion | `spread:per_attribute.name.prefer_higher_trust_only:per_attribute.name.voting_only` | up | 0.170 | 0.190 | 0.140 | -0.035 | -0.205 | qualitative | [!!] | [ok] | [!!] | not weakly up: 0.190 -> 0.140 -> -0.035 |
| knob_10 | em_flat_pruned | em_matching | `aggregated.macro_f1_baseline_model_on_baseline_test` | flat | 0.884 | 0.899 | 0.872 | 0.868 | -0.016 | qualitative | [ok] | [ok] | [ok] | weakly flat: 0.899 -> 0.872 -> 0.868 |

## Collapses

No members fell below the collapse threshold.

## Best-Member Ceiling (P8)

Per-stage best-member F1 across baseline -> easy -> medium -> hard. A valid difficulty signal must depress the *ceiling* (the user-attainable member), not just the committee mean. A flat / rising ceiling means the user-selected matcher never sees the synthetic difficulty (committee-mean drift can be masked by weak-member degradation alone).

| Stage | baseline | easy | medium | hard | delta | non-increasing | winner trail | Reason |
|---|---|---|---|---|---|---|---|---|
| sm | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | [ok] | `llm_openai -> coma_hybrid -> llm_openai -> llm_openai` | best-member ceiling non-increasing: 1.000 -> 1.000 -> 1.000 -> 1.000  (llm_openai -> coma_hybrid -> llm_openai -> llm_openai) |
| norm | 0.939 | 0.662 | 0.735 | 0.752 | -0.187 | [!!] | `passthrough -> passthrough -> llm_only -> llm_only` | best-member ceiling did NOT decline: 0.939 -> 0.662 -> 0.735 -> 0.752  (passthrough -> passthrough -> llm_only -> llm_only) — difficulty dial may be invisible to the user-selected matcher |
| em_blocking | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | [ok] | `embedding_blocker -> embedding_blocker -> sc_block -> embedding_blocker` | best-member ceiling non-increasing: 1.000 -> 1.000 -> 1.000 -> 1.000  (embedding_blocker -> embedding_blocker -> sc_block -> embedding_blocker) |
| em_matching | 0.895 | 0.929 | 0.881 | 0.905 | 0.009 | [!!] | `comem -> magellan -> magellan -> magellan` | best-member ceiling did NOT decline: 0.895 -> 0.929 -> 0.881 -> 0.905  (comem -> magellan -> magellan -> magellan) — difficulty dial may be invisible to the user-selected matcher |
| fusion | 0.458 | 0.400 | 0.413 | 0.348 | -0.109 | [!!] | `prefer_higher_trust_only -> pydi_per_attribute_optimal -> pydi_per_attribute_optimal -> ltm_only` | best-member ceiling did NOT decline: 0.458 -> 0.400 -> 0.413 -> 0.348  (prefer_higher_trust_only -> pydi_per_attribute_optimal -> pydi_per_attribute_optimal -> ltm_only) — difficulty dial may be invisible to the user-selected matcher |

## Open Questions

Signals that are direction-correct but magnitude-unspecified by the knob card. M10 should decide whether the measured delta is 'strong enough' to count as validation.

| Knob | Signal | Stage | observed delta | Card notes |
|---|---|---|---|---|
| knob_01 | em_monotone_drop_pruned | em_matching | -0.016 | Monotone F1 drop across levels. Sharper for lexical blockers than for embedding matchers. [baseline_pruned surface companion] |
| knob_02 | em_monotone_drop_pruned | em_matching | -0.016 | Monotone drop. Sharper for similarity-threshold matchers than for learned matchers. Niche collisions pressurise precision. [baseline_pruned surface companion] |
| knob_03 | em_monotone_drop_pruned | em_matching | -0.016 | Monotone F1 drop. Learned matchers with missing-value handling degrade less than rule-based comparators. [baseline_pruned surface companion] |
| knob_04 | em_flat_or_shift | em_matching | 0.003 | EM mostly indifferent; any shift is through the singleton-prior, direction unspecified by the card. |
| knob_04 | em_flat_or_shift_pruned | em_matching | -0.016 | EM mostly indifferent; any shift is through the singleton-prior, direction unspecified by the card. [baseline_pruned surface companion] |
| knob_05 | em_monotone_drop_pruned | em_matching | -0.016 | Monotone drop for non-normalizing comparators; minimal for type-aware comparators. [baseline_pruned surface companion] |
| knob_06 | em_monotone_drop_pruned | em_matching | -0.016 | Monotone drop. Sharp for rule-based comparators, mild for learned/embedding matchers. [baseline_pruned surface companion] |
| knob_08 | sm_monotone_drop | sm | -0.081 | Primary target. Monotone drop expected. |
| knob_08 | sm_label_collapses | sm | -0.065 | Label-based string-similarity matchers collapse fast on cryptic/anonymized names. |
| knob_08 | sm_spread_is_signal | sm | 0.065 | Per the card, "the spread between matcher types IS the K8 difficulty signal". Instance-based / embedding / LLM matchers hold while label-based collapse. Spread (llm_openai.f1 - label_jaccard.f1) widens monotonically. |
| knob_08 | sm_instance_steady | sm | -0.066 | Instance-based matchers degrade more gracefully than label-based. Note: baseline instance-based F1 for companies is already 0.0 (no overlap in sampled values). This signal may be untestable on companies because the baseline is already at the floor. |
| knob_08 | em_flat | em_matching | 0.003 | Downstream stages unaffected by K8. |
| knob_08 | em_flat_pruned | em_matching | -0.016 | Downstream stages unaffected by K8. [baseline_pruned surface companion] |
| knob_10 | em_flat | em_matching | 0.003 | Upstream stages unaffected by K10. |
| knob_10 | em_flat_pruned | em_matching | -0.016 | Upstream stages unaffected by K10. [baseline_pruned surface companion] |

## Provenance

- Domain: companies
- Expectations: `usecases_synthetic/config/knob_expected_signals.yaml`
- Baseline: `usecases_synthetic/baselines/companies/baseline_metrics.json`
- Per-level metrics: `usecases_synthetic/validation/companies/<level>/metrics.json`
