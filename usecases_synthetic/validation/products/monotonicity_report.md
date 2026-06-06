# Monotonicity + Collapse Report — products

Cross-level analysis of easy/medium/hard variants against the baseline. See `knobs/cross_cutting.md` for the protocol.

## Cumulative Cross-Level Slope (load-bearing verdict)

Per-stage committee metric across the **cumulative** variant levels (every knob on at each level). This is the C2-contract verdict: committee scores should weakly decrease easy -> medium -> hard, and baseline (a reference value) should land no harder than medium. It makes no single-knob isolation assumption, so this -- not the per-knob signals below -- is the headline difficulty verdict.

| Stage | Metric | baseline | easy | medium | hard | easy->hard | Mono (e>=m>=h) | BasePos |
|---|---|---|---|---|---|---|---|---|
| sm | `aggregated.macro_f1` | 0.676 | 0.745 | 0.686 | 0.621 | -0.124 | [ok] | [!!] |
| norm | `aggregated.macro_f1` | 0.593 | 0.629 | 0.531 | 0.518 | -0.110 | [ok] | [ok] |
| em_blocking | `aggregated.macro_pair_recall` | 0.850 | 0.846 | 0.875 | 0.866 | 0.020 | [!!] | [!!] |
| em_matching | `aggregated.macro_f1_variant_model_on_regen_test` | 0.900 | 0.867 | 0.822 | 0.822 | -0.046 | [ok] | [ok] |
| fusion | `aggregated.overall_accuracy` | 0.698 | 0.541 | 0.611 | 0.514 | -0.027 | [!!] | [ok] |

## Per-Knob Expected Signals (indicative)

> These are **per-knob** expectations from `knob_expected_signals.yaml`, evaluated against the **cumulative** variants (every knob on at each level). They cannot isolate one knob, so a `flat` expectation for a stage that *another* knob also drives reads `[!!]` by construction (e.g. SM is not flat for knob_01 because K8 naming is also on). Treat this as indicative of combined effect; the load-bearing verdict is the Cumulative Cross-Level Slope above. For true per-knob isolation, run `generate_variant --only-knob <K>` ablations.

| Knob | Signals | OK direction | OK range | Notes |
|---|---|---|---|---|
| knob_01 | 4 | 1/4 [!!] | 0/— | [card](knobs/knob_01_surface_augmentation.md) |
| knob_02 | 4 | 1/4 [!!] | 0/— | [card](knobs/knob_02_niche_density.md) |
| knob_03 | 4 | 1/4 [!!] | 0/— | [card](knobs/knob_03_attribute_drop.md) |
| knob_04 | 4 | 1/4 [!!] | 0/— | [card](knobs/knob_04_coverage_skew.md) |
| knob_05 | 6 | 1/6 [!!] | 0/— | [card](knobs/knob_05_format_unit.md) |
| knob_06 | 5 | 1/5 [!!] | 0/— | [card](knobs/knob_06_value_noise.md) |
| knob_08 | 7 | 5/7 [!!] | 0/— | [card](knobs/knob_08_schema_naming.md) |
| knob_10 | 6 | 1/6 [!!] | 0/— | [card](knobs/knob_10_source_reliability.md) |

## Per-Signal Results

| Knob | Signal | Stage | Metric | Dir | baseline | easy | medium | hard | delta | target | Mono | Rng | BasePos | Reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| knob_01 | sm_flat | sm | `aggregated.macro_f1` | flat | 0.676 | 0.745 | 0.686 | 0.621 | -0.056 | qualitative | [!!] | [ok] | [ok] | not weakly flat: 0.745 -> 0.686 -> 0.621 |
| knob_01 | em_monotone_drop | em_matching | `aggregated.macro_f1_baseline_model_on_regen_test` | down | 0.900 | 0.740 | 0.705 | 0.777 | -0.123 | qualitative | [!!] | [ok] | [ok] | not weakly down: 0.740 -> 0.705 -> 0.777 |
| knob_01 | em_blocker_spread | em_blocking | `spread:per_member.standard_blocker.metrics.pair_recall:per_member.embedding_blocker.metrics.pair_recall` | down | -0.340 | -0.412 | -0.314 | -0.330 | 0.010 | qualitative | [!!] | [ok] | [!!] | not weakly down: -0.412 -> -0.314 -> -0.330 |
| knob_01 | em_monotone_drop_pruned | em_matching | `aggregated.macro_f1_baseline_model_on_baseline_test` | down | 0.900 | 0.919 | 0.900 | 0.857 | -0.043 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.919 -> 0.900 -> 0.857 |
| knob_02 | em_monotone_drop | em_matching | `aggregated.macro_f1_baseline_model_on_regen_test` | down | 0.900 | 0.740 | 0.705 | 0.777 | -0.123 | qualitative | [!!] | [ok] | [ok] | not weakly down: 0.740 -> 0.705 -> 0.777 |
| knob_02 | em_pool_precision_tightens | em_matching | `aggregated.macro_precision` | down | 0.903 | 0.853 | 0.802 | 0.828 | -0.075 | qualitative | [!!] | [ok] | [ok] | not weakly down: 0.853 -> 0.802 -> 0.828 |
| knob_02 | fusion_flat | fusion | `aggregated.overall_accuracy` | flat | 0.698 | 0.541 | 0.611 | 0.514 | -0.184 | qualitative | [!!] | [ok] | [!!] | not weakly flat: 0.541 -> 0.611 -> 0.514 |
| knob_02 | em_monotone_drop_pruned | em_matching | `aggregated.macro_f1_baseline_model_on_baseline_test` | down | 0.900 | 0.919 | 0.900 | 0.857 | -0.043 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.919 -> 0.900 -> 0.857 |
| knob_03 | em_pool_recall_drops | em_blocking | `aggregated.macro_pair_recall` | down | 0.850 | 0.846 | 0.875 | 0.866 | 0.016 | qualitative | [!!] | [ok] | [!!] | not weakly down: 0.846 -> 0.875 -> 0.866 |
| knob_03 | em_monotone_drop | em_matching | `aggregated.macro_f1_baseline_model_on_regen_test` | down | 0.900 | 0.740 | 0.705 | 0.777 | -0.123 | qualitative | [!!] | [ok] | [ok] | not weakly down: 0.740 -> 0.705 -> 0.777 |
| knob_03 | fusion_monotone_drop | fusion | `aggregated.overall_accuracy` | down | 0.698 | 0.541 | 0.611 | 0.514 | -0.184 | qualitative | [!!] | [ok] | [ok] | not weakly down: 0.541 -> 0.611 -> 0.514 |
| knob_03 | em_monotone_drop_pruned | em_matching | `aggregated.macro_f1_baseline_model_on_baseline_test` | down | 0.900 | 0.919 | 0.900 | 0.857 | -0.043 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.919 -> 0.900 -> 0.857 |
| knob_04 | em_flat_or_shift | em_matching | `aggregated.macro_f1_baseline_model_on_regen_test` | flat | 0.900 | 0.740 | 0.705 | 0.777 | -0.123 | qualitative | [ok] | [ok] | [!!] | weakly flat: 0.740 -> 0.705 -> 0.777 |
| knob_04 | fusion_monotone_drop | fusion | `aggregated.overall_accuracy` | down | 0.698 | 0.541 | 0.611 | 0.514 | -0.184 | qualitative | [!!] | [ok] | [ok] | not weakly down: 0.541 -> 0.611 -> 0.514 |
| knob_04 | fusion_spread_widens | fusion | `spread:aggregated.max_accuracy:aggregated.min_accuracy` | up | 0.153 | 0.146 | 0.159 | 0.100 | -0.053 | qualitative | [!!] | [ok] | [ok] | not weakly up: 0.146 -> 0.159 -> 0.100 |
| knob_04 | em_flat_or_shift_pruned | em_matching | `aggregated.macro_f1_baseline_model_on_baseline_test` | flat | 0.900 | 0.919 | 0.900 | 0.857 | -0.043 | qualitative | [!!] | [ok] | [ok] | not weakly flat: 0.919 -> 0.900 -> 0.857 |
| knob_05 | sm_flat | sm | `aggregated.macro_f1` | flat | 0.676 | 0.745 | 0.686 | 0.621 | -0.056 | qualitative | [!!] | [ok] | [ok] | not weakly flat: 0.745 -> 0.686 -> 0.621 |
| knob_05 | em_monotone_drop | em_matching | `aggregated.macro_f1_baseline_model_on_regen_test` | down | 0.900 | 0.740 | 0.705 | 0.777 | -0.123 | qualitative | [!!] | [ok] | [ok] | not weakly down: 0.740 -> 0.705 -> 0.777 |
| knob_05 | em_pool_recall_drops_for_lexical | em_blocking | `per_member.standard_blocker.metrics.pair_recall` | down | 0.660 | 0.588 | 0.686 | 0.670 | 0.010 | qualitative | [!!] | [ok] | [!!] | not weakly down: 0.588 -> 0.686 -> 0.670 |
| knob_05 | fusion_spread_is_signal | fusion | `spread:aggregated.max_accuracy:aggregated.min_accuracy` | up | 0.153 | 0.146 | 0.159 | 0.100 | -0.053 | qualitative | [!!] | [ok] | [ok] | not weakly up: 0.146 -> 0.159 -> 0.100 |
| knob_05 | fusion_monotone_drop | fusion | `aggregated.overall_accuracy` | down | 0.698 | 0.541 | 0.611 | 0.514 | -0.184 | qualitative | [!!] | [ok] | [ok] | not weakly down: 0.541 -> 0.611 -> 0.514 |
| knob_05 | em_monotone_drop_pruned | em_matching | `aggregated.macro_f1_baseline_model_on_baseline_test` | down | 0.900 | 0.919 | 0.900 | 0.857 | -0.043 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.919 -> 0.900 -> 0.857 |
| knob_06 | sm_flat | sm | `aggregated.macro_f1` | flat | 0.676 | 0.745 | 0.686 | 0.621 | -0.056 | qualitative | [!!] | [ok] | [ok] | not weakly flat: 0.745 -> 0.686 -> 0.621 |
| knob_06 | em_monotone_drop | em_matching | `aggregated.macro_f1_baseline_model_on_regen_test` | down | 0.900 | 0.740 | 0.705 | 0.777 | -0.123 | qualitative | [!!] | [ok] | [ok] | not weakly down: 0.740 -> 0.705 -> 0.777 |
| knob_06 | em_pool_recall_drops | em_blocking | `aggregated.macro_pair_recall` | down | 0.850 | 0.846 | 0.875 | 0.866 | 0.016 | qualitative | [!!] | [ok] | [!!] | not weakly down: 0.846 -> 0.875 -> 0.866 |
| knob_06 | fusion_monotone_drop | fusion | `aggregated.overall_accuracy` | down | 0.698 | 0.541 | 0.611 | 0.514 | -0.184 | qualitative | [!!] | [ok] | [ok] | not weakly down: 0.541 -> 0.611 -> 0.514 |
| knob_06 | em_monotone_drop_pruned | em_matching | `aggregated.macro_f1_baseline_model_on_baseline_test` | down | 0.900 | 0.919 | 0.900 | 0.857 | -0.043 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.919 -> 0.900 -> 0.857 |
| knob_08 | sm_monotone_drop | sm | `aggregated.macro_f1` | down | 0.676 | 0.745 | 0.686 | 0.621 | -0.056 | qualitative | [ok] | [ok] | [!!] | weakly down: 0.745 -> 0.686 -> 0.621 |
| knob_08 | sm_label_collapses | sm | `per_member.label_jw.metrics.f1` | down | 0.557 | 0.781 | 0.603 | 0.299 | -0.258 | qualitative | [ok] | [ok] | [!!] | weakly down: 0.781 -> 0.603 -> 0.299 |
| knob_08 | sm_spread_is_signal | sm | `spread:per_member.llm_openai.metrics.f1:per_member.label_jw.metrics.f1` | up | 0.424 | 0.200 | 0.378 | 0.683 | 0.258 | qualitative | [ok] | [ok] | [!!] | weakly up: 0.200 -> 0.378 -> 0.683 |
| knob_08 | sm_instance_steady | sm | `per_member.instance_tf_cosine.metrics.f1` | flat | 0.404 | 0.418 | 0.449 | 0.453 | 0.049 | qualitative | [ok] | [ok] | [ok] | weakly flat: 0.418 -> 0.449 -> 0.453 |
| knob_08 | em_flat | em_matching | `aggregated.macro_f1_baseline_model_on_regen_test` | flat | 0.900 | 0.740 | 0.705 | 0.777 | -0.123 | qualitative | [ok] | [ok] | [!!] | weakly flat: 0.740 -> 0.705 -> 0.777 |
| knob_08 | fusion_flat | fusion | `aggregated.overall_accuracy` | flat | 0.698 | 0.541 | 0.611 | 0.514 | -0.184 | qualitative | [!!] | [ok] | [!!] | not weakly flat: 0.541 -> 0.611 -> 0.514 |
| knob_08 | em_flat_pruned | em_matching | `aggregated.macro_f1_baseline_model_on_baseline_test` | flat | 0.900 | 0.919 | 0.900 | 0.857 | -0.043 | qualitative | [!!] | [ok] | [ok] | not weakly flat: 0.919 -> 0.900 -> 0.857 |
| knob_10 | sm_flat | sm | `aggregated.macro_f1` | flat | 0.676 | 0.745 | 0.686 | 0.621 | -0.056 | qualitative | [!!] | [ok] | [ok] | not weakly flat: 0.745 -> 0.686 -> 0.621 |
| knob_10 | em_flat | em_matching | `aggregated.macro_f1_baseline_model_on_regen_test` | flat | 0.900 | 0.740 | 0.705 | 0.777 | -0.123 | qualitative | [ok] | [ok] | [!!] | weakly flat: 0.740 -> 0.705 -> 0.777 |
| knob_10 | fusion_monotone_drop | fusion | `aggregated.overall_accuracy` | down | 0.698 | 0.541 | 0.611 | 0.514 | -0.184 | qualitative | [!!] | [ok] | [ok] | not weakly down: 0.541 -> 0.611 -> 0.514 |
| knob_10 | fusion_spread_widens | fusion | `spread:aggregated.max_accuracy:aggregated.min_accuracy` | up | 0.153 | 0.146 | 0.159 | 0.100 | -0.053 | qualitative | [!!] | [ok] | [ok] | not weakly up: 0.146 -> 0.159 -> 0.100 |
| knob_10 | fusion_voting_drops_faster_than_trust_on_name | fusion | `spread:per_attribute.name.prefer_higher_trust_only:per_attribute.name.voting_only` | up | NaN | NaN | NaN | NaN | NaN | qualitative | [!!] | [ok] | [ok] | metric missing at levels: easy, medium, hard |
| knob_10 | em_flat_pruned | em_matching | `aggregated.macro_f1_baseline_model_on_baseline_test` | flat | 0.900 | 0.919 | 0.900 | 0.857 | -0.043 | qualitative | [!!] | [ok] | [ok] | not weakly flat: 0.919 -> 0.900 -> 0.857 |

## Collapses

No members fell below the collapse threshold.

## Best-Member Ceiling (P8)

Per-stage best-member F1 across baseline -> easy -> medium -> hard. A valid difficulty signal must depress the *ceiling* (the user-attainable member), not just the committee mean. A flat / rising ceiling means the user-selected matcher never sees the synthetic difficulty (committee-mean drift can be masked by weak-member degradation alone).

| Stage | baseline | easy | medium | hard | delta | non-increasing | winner trail | Reason |
|---|---|---|---|---|---|---|---|---|
| sm | 0.981 | 0.995 | 0.981 | 0.981 | 0.000 | [!!] | `llm_openai -> duplicate_majority -> llm_openai -> llm_openai` | best-member ceiling did NOT decline: 0.981 -> 0.995 -> 0.981 -> 0.981  (llm_openai -> duplicate_majority -> llm_openai -> llm_openai) — difficulty dial may be invisible to the user-selected matcher |
| norm | 0.619 | 0.679 | 0.561 | 0.545 | -0.074 | [!!] | `rule_per_attribute_optimal -> rule_per_attribute_optimal -> rule_per_attribute_optimal -> rule_per_attribute_optimal` | best-member ceiling did NOT decline: 0.619 -> 0.679 -> 0.561 -> 0.545  (rule_per_attribute_optimal -> rule_per_attribute_optimal -> rule_per_attribute_optimal -> rule_per_attribute_optimal) — difficulty dial may be invisible to the user-selected matcher |
| em_blocking | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | [ok] | `token_blocker -> sc_block -> embedding_blocker -> token_blocker` | best-member ceiling non-increasing: 1.000 -> 1.000 -> 1.000 -> 1.000  (token_blocker -> sc_block -> embedding_blocker -> token_blocker) |
| em_matching | 0.934 | 0.943 | 0.949 | 0.919 | -0.015 | [!!] | `ditto_plm -> ditto_plm -> ditto_plm -> ditto_plm` | best-member ceiling did NOT decline: 0.934 -> 0.943 -> 0.949 -> 0.919  (ditto_plm -> ditto_plm -> ditto_plm -> ditto_plm) — difficulty dial may be invisible to the user-selected matcher |
| fusion | 0.698 | 0.541 | 0.611 | 0.514 | -0.184 | [!!] | `pydi_per_attribute_optimal -> llm_only -> pydi_per_attribute_optimal -> pydi_per_attribute_optimal` | best-member ceiling did NOT decline: 0.698 -> 0.541 -> 0.611 -> 0.514  (pydi_per_attribute_optimal -> llm_only -> pydi_per_attribute_optimal -> pydi_per_attribute_optimal) — difficulty dial may be invisible to the user-selected matcher |

## Open Questions

Signals that are direction-correct but magnitude-unspecified by the knob card. M10 should decide whether the measured delta is 'strong enough' to count as validation.

| Knob | Signal | Stage | observed delta | Card notes |
|---|---|---|---|---|
| knob_01 | em_monotone_drop_pruned | em_matching | -0.043 | Monotone F1 drop across levels. Sharper for lexical blockers than for embedding matchers. [baseline_pruned surface companion] |
| knob_02 | em_monotone_drop_pruned | em_matching | -0.043 | Monotone drop. Sharper for similarity-threshold matchers than for learned matchers. Niche collisions pressurise precision. [baseline_pruned surface companion] |
| knob_03 | em_monotone_drop_pruned | em_matching | -0.043 | Monotone F1 drop. Learned matchers with missing-value handling degrade less than rule-based comparators. [baseline_pruned surface companion] |
| knob_04 | em_flat_or_shift | em_matching | -0.123 | EM mostly indifferent; any shift is through the singleton-prior, direction unspecified by the card. |
| knob_05 | em_monotone_drop_pruned | em_matching | -0.043 | Monotone drop for non-normalizing comparators; minimal for type-aware comparators. [baseline_pruned surface companion] |
| knob_06 | em_monotone_drop_pruned | em_matching | -0.043 | Monotone drop. Sharp for rule-based comparators, mild for learned/embedding matchers. [baseline_pruned surface companion] |
| knob_08 | sm_monotone_drop | sm | -0.056 | Primary target. Monotone drop expected. |
| knob_08 | sm_label_collapses | sm | -0.258 | Label-based string-similarity matchers collapse fast on cryptic/anonymized names. |
| knob_08 | sm_spread_is_signal | sm | 0.258 | Per the card, "the spread between matcher types IS the K8 difficulty signal". Instance-based / embedding / LLM matchers hold while label-based collapse. Spread (llm_openai.f1 - label_jaccard.f1) widens monotonically. |
| knob_08 | sm_instance_steady | sm | 0.049 | Instance-based matchers degrade more gracefully than label-based. Note: baseline instance-based F1 for companies is already 0.0 (no overlap in sampled values). This signal may be untestable on companies because the baseline is already at the floor. |
| knob_08 | em_flat | em_matching | -0.123 | Downstream stages unaffected by K8. |
| knob_10 | em_flat | em_matching | -0.123 | Upstream stages unaffected by K10. |

## Provenance

- Domain: products
- Expectations: `usecases_synthetic/config/knob_expected_signals.yaml`
- Baseline: `usecases_synthetic/baselines/products/baseline_metrics.json`
- Per-level metrics: `usecases_synthetic/validation/products/<level>/metrics.json`
