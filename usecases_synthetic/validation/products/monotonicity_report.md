# Monotonicity + Collapse Report — products

Cross-level analysis of easy/medium/hard variants against the baseline. See `knobs/cross_cutting.md` for the protocol.

## Cumulative Cross-Level Slope (load-bearing verdict)

Per-stage committee metric across the **cumulative** variant levels (every knob on at each level). This is the C2-contract verdict: committee scores should weakly decrease easy -> medium -> hard, and baseline (a reference value) should land no harder than medium. It makes no single-knob isolation assumption, so this -- not the per-knob signals below -- is the headline difficulty verdict.

| Stage | Metric | baseline | easy | medium | hard | easy->hard | Mono (e>=m>=h) | BasePos |
|---|---|---|---|---|---|---|---|---|
| sm | `aggregated.macro_f1` | 0.654 | 0.659 | 0.583 | 0.509 | -0.150 | [ok] | [ok] |
| norm | `aggregated.macro_f1` | 0.452 | 0.467 | 0.414 | 0.401 | -0.066 | [ok] | [ok] |
| em_blocking | `aggregated.macro_pair_recall` | 0.861 | 0.894 | 0.907 | 0.857 | -0.037 | [!!] | [!!] |
| em_matching | `aggregated.macro_f1_variant_model_on_regen_test` | 0.951 | 0.944 | 0.934 | 0.896 | -0.049 | [ok] | [ok] |
| fusion | `aggregated.overall_accuracy` | 0.822 | 0.702 | 0.662 | 0.527 | -0.174 | [ok] | [ok] |

## Per-Knob Expected Signals (indicative)

> These are **per-knob** expectations from `knob_expected_signals.yaml`, evaluated against the **cumulative** variants (every knob on at each level). They cannot isolate one knob, so a `flat` expectation for a stage that *another* knob also drives reads `[!!]` by construction (e.g. SM is not flat for knob_01 because K8 naming is also on). Treat this as indicative of combined effect; the load-bearing verdict is the Cumulative Cross-Level Slope above. For true per-knob isolation, run `generate_variant --only-knob <K>` ablations.

| Knob | Signals | OK direction | OK range | Notes |
|---|---|---|---|---|
| knob_01 | 3 | 1/3 [!!] | 0/— | [card](knobs/knob_01_surface_augmentation.md) |
| knob_02 | 3 | 2/3 [!!] | 0/— | [card](knobs/knob_02_niche_density.md) |
| knob_03 | 3 | 2/3 [!!] | 0/— | [card](knobs/knob_03_attribute_drop.md) |
| knob_04 | 3 | 2/3 [!!] | 0/— | [card](knobs/knob_04_coverage_skew.md) |
| knob_05 | 5 | 2/5 [!!] | 0/— | [card](knobs/knob_05_format_unit.md) |
| knob_06 | 4 | 2/4 [!!] | 0/— | [card](knobs/knob_06_value_noise.md) |
| knob_08 | 6 | 5/6 [!!] | 0/— | [card](knobs/knob_08_schema_naming.md) |
| knob_10 | 5 | 2/5 [!!] | 0/— | [card](knobs/knob_10_source_reliability.md) |

## Per-Signal Results

| Knob | Signal | Stage | Metric | Dir | baseline | easy | medium | hard | delta | target | Mono | Rng | BasePos | Reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| knob_01 | sm_flat | sm | `aggregated.macro_f1` | flat | 0.654 | 0.659 | 0.583 | 0.509 | -0.145 | qualitative | [!!] | [ok] | [!!] | not weakly flat: 0.659 -> 0.583 -> 0.509 |
| knob_01 | em_monotone_drop | em_matching | `aggregated.macro_f1_variant_model_on_regen_test` | down | 0.951 | 0.944 | 0.934 | 0.896 | -0.055 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.944 -> 0.934 -> 0.896 |
| knob_01 | em_blocker_spread | em_blocking | `spread:per_member.standard_blocker.metrics.pair_recall:per_member.embedding_blocker.metrics.pair_recall` | down | -0.293 | -0.300 | -0.249 | -0.344 | -0.051 | qualitative | [!!] | [ok] | [!!] | not weakly down: -0.300 -> -0.249 -> -0.344 |
| knob_02 | em_monotone_drop | em_matching | `aggregated.macro_f1_variant_model_on_regen_test` | down | 0.951 | 0.944 | 0.934 | 0.896 | -0.055 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.944 -> 0.934 -> 0.896 |
| knob_02 | em_pool_precision_tightens | em_matching | `aggregated.macro_precision` | down | 0.991 | 0.998 | 0.995 | 0.984 | -0.008 | qualitative | [ok] | [ok] | [!!] | weakly down: 0.998 -> 0.995 -> 0.984 |
| knob_02 | fusion_flat | fusion | `aggregated.overall_accuracy` | flat | 0.822 | 0.702 | 0.662 | 0.527 | -0.294 | qualitative | [!!] | [ok] | [!!] | not weakly flat: 0.702 -> 0.662 -> 0.527 |
| knob_03 | em_pool_recall_drops | em_blocking | `aggregated.macro_pair_recall` | down | 0.861 | 0.894 | 0.907 | 0.857 | -0.005 | qualitative | [!!] | [ok] | [!!] | not weakly down: 0.894 -> 0.907 -> 0.857 |
| knob_03 | em_monotone_drop | em_matching | `aggregated.macro_f1_variant_model_on_regen_test` | down | 0.951 | 0.944 | 0.934 | 0.896 | -0.055 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.944 -> 0.934 -> 0.896 |
| knob_03 | fusion_monotone_drop | fusion | `aggregated.overall_accuracy` | down | 0.822 | 0.702 | 0.662 | 0.527 | -0.294 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.702 -> 0.662 -> 0.527 |
| knob_04 | em_flat_or_shift | em_matching | `aggregated.macro_f1_variant_model_on_regen_test` | flat | 0.951 | 0.944 | 0.934 | 0.896 | -0.055 | qualitative | [ok] | [ok] | [ok] | weakly flat: 0.944 -> 0.934 -> 0.896 |
| knob_04 | fusion_monotone_drop | fusion | `aggregated.overall_accuracy` | down | 0.822 | 0.702 | 0.662 | 0.527 | -0.294 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.702 -> 0.662 -> 0.527 |
| knob_04 | fusion_spread_widens | fusion | `spread:aggregated.max_accuracy:aggregated.min_accuracy` | up | 0.422 | 0.320 | 0.304 | 0.160 | -0.262 | qualitative | [!!] | [ok] | [!!] | not weakly up: 0.320 -> 0.304 -> 0.160 |
| knob_05 | sm_flat | sm | `aggregated.macro_f1` | flat | 0.654 | 0.659 | 0.583 | 0.509 | -0.145 | qualitative | [!!] | [ok] | [!!] | not weakly flat: 0.659 -> 0.583 -> 0.509 |
| knob_05 | em_monotone_drop | em_matching | `aggregated.macro_f1_variant_model_on_regen_test` | down | 0.951 | 0.944 | 0.934 | 0.896 | -0.055 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.944 -> 0.934 -> 0.896 |
| knob_05 | em_pool_recall_drops_for_lexical | em_blocking | `per_member.standard_blocker.metrics.pair_recall` | down | 0.699 | 0.700 | 0.751 | 0.651 | -0.047 | qualitative | [!!] | [ok] | [!!] | not weakly down: 0.700 -> 0.751 -> 0.651 |
| knob_05 | fusion_spread_is_signal | fusion | `spread:aggregated.max_accuracy:aggregated.min_accuracy` | up | 0.422 | 0.320 | 0.304 | 0.160 | -0.262 | qualitative | [!!] | [ok] | [!!] | not weakly up: 0.320 -> 0.304 -> 0.160 |
| knob_05 | fusion_monotone_drop | fusion | `aggregated.overall_accuracy` | down | 0.822 | 0.702 | 0.662 | 0.527 | -0.294 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.702 -> 0.662 -> 0.527 |
| knob_06 | sm_flat | sm | `aggregated.macro_f1` | flat | 0.654 | 0.659 | 0.583 | 0.509 | -0.145 | qualitative | [!!] | [ok] | [!!] | not weakly flat: 0.659 -> 0.583 -> 0.509 |
| knob_06 | em_monotone_drop | em_matching | `aggregated.macro_f1_variant_model_on_regen_test` | down | 0.951 | 0.944 | 0.934 | 0.896 | -0.055 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.944 -> 0.934 -> 0.896 |
| knob_06 | em_pool_recall_drops | em_blocking | `aggregated.macro_pair_recall` | down | 0.861 | 0.894 | 0.907 | 0.857 | -0.005 | qualitative | [!!] | [ok] | [!!] | not weakly down: 0.894 -> 0.907 -> 0.857 |
| knob_06 | fusion_monotone_drop | fusion | `aggregated.overall_accuracy` | down | 0.822 | 0.702 | 0.662 | 0.527 | -0.294 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.702 -> 0.662 -> 0.527 |
| knob_08 | sm_monotone_drop | sm | `aggregated.macro_f1` | down | 0.654 | 0.659 | 0.583 | 0.509 | -0.145 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.659 -> 0.583 -> 0.509 |
| knob_08 | sm_label_collapses | sm | `per_member.label_jw.metrics.f1` | down | 0.678 | 0.678 | 0.462 | 0.196 | -0.482 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.678 -> 0.462 -> 0.196 |
| knob_08 | sm_spread_is_signal | sm | `spread:per_member.llm_openai.metrics.f1:per_member.label_jw.metrics.f1` | up | 0.173 | 0.173 | 0.390 | 0.655 | 0.482 | qualitative | [ok] | [ok] | [ok] | weakly up: 0.173 -> 0.390 -> 0.655 |
| knob_08 | sm_instance_steady | sm | `per_member.instance_tf_cosine.metrics.f1` | flat | 0.280 | 0.284 | 0.292 | 0.281 | 0.001 | qualitative | [ok] | [ok] | [ok] | weakly flat: 0.284 -> 0.292 -> 0.281 |
| knob_08 | em_flat | em_matching | `aggregated.macro_f1_variant_model_on_regen_test` | flat | 0.951 | 0.944 | 0.934 | 0.896 | -0.055 | qualitative | [ok] | [ok] | [ok] | weakly flat: 0.944 -> 0.934 -> 0.896 |
| knob_08 | fusion_flat | fusion | `aggregated.overall_accuracy` | flat | 0.822 | 0.702 | 0.662 | 0.527 | -0.294 | qualitative | [!!] | [ok] | [!!] | not weakly flat: 0.702 -> 0.662 -> 0.527 |
| knob_10 | sm_flat | sm | `aggregated.macro_f1` | flat | 0.654 | 0.659 | 0.583 | 0.509 | -0.145 | qualitative | [!!] | [ok] | [!!] | not weakly flat: 0.659 -> 0.583 -> 0.509 |
| knob_10 | em_flat | em_matching | `aggregated.macro_f1_variant_model_on_regen_test` | flat | 0.951 | 0.944 | 0.934 | 0.896 | -0.055 | qualitative | [ok] | [ok] | [ok] | weakly flat: 0.944 -> 0.934 -> 0.896 |
| knob_10 | fusion_monotone_drop | fusion | `aggregated.overall_accuracy` | down | 0.822 | 0.702 | 0.662 | 0.527 | -0.294 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.702 -> 0.662 -> 0.527 |
| knob_10 | fusion_spread_widens | fusion | `spread:aggregated.max_accuracy:aggregated.min_accuracy` | up | 0.422 | 0.320 | 0.304 | 0.160 | -0.262 | qualitative | [!!] | [ok] | [!!] | not weakly up: 0.320 -> 0.304 -> 0.160 |
| knob_10 | fusion_voting_drops_faster_than_trust_on_name | fusion | `spread:per_attribute.name.prefer_higher_trust_only:per_attribute.name.voting_only` | up | NaN | NaN | NaN | NaN | NaN | qualitative | [!!] | [ok] | [ok] | metric missing at levels: easy, medium, hard |

## Collapses

No members fell below the collapse threshold.

## Best-Member Ceiling (P8)

Per-stage best-member F1 across baseline -> easy -> medium -> hard. A valid difficulty signal must depress the *ceiling* (the user-attainable member), not just the committee mean. A flat / rising ceiling means the user-selected matcher never sees the synthetic difficulty (committee-mean drift can be masked by weak-member degradation alone).

| Stage | baseline | easy | medium | hard | delta | non-increasing | winner trail | Reason |
|---|---|---|---|---|---|---|---|---|
| sm | 1.000 | 1.000 | 0.940 | 0.919 | -0.081 | [ok] | `duplicate_majority -> duplicate_majority -> duplicate_majority -> duplicate_majority` | best-member ceiling non-increasing: 1.000 -> 1.000 -> 0.940 -> 0.919  (duplicate_majority -> duplicate_majority -> duplicate_majority -> duplicate_majority) |
| norm | 0.479 | 0.494 | 0.440 | 0.423 | -0.055 | [!!] | `rule_per_attribute_optimal -> passthrough -> passthrough -> rule_per_attribute_optimal` | best-member ceiling did NOT decline: 0.479 -> 0.494 -> 0.440 -> 0.423  (rule_per_attribute_optimal -> passthrough -> passthrough -> rule_per_attribute_optimal) — difficulty dial may be invisible to the user-selected matcher |
| em_blocking | 1.000 | 1.000 | 1.000 | 0.995 | -0.005 | [ok] | `sc_block -> token_blocker -> sc_block -> embedding_blocker` | best-member ceiling non-increasing: 1.000 -> 1.000 -> 1.000 -> 0.995  (sc_block -> token_blocker -> sc_block -> embedding_blocker) |
| em_matching | 0.979 | 0.982 | 0.991 | 0.945 | -0.034 | [!!] | `ditto_plm -> magellan -> ditto_plm -> ditto_plm` | best-member ceiling did NOT decline: 0.979 -> 0.982 -> 0.991 -> 0.945  (ditto_plm -> magellan -> ditto_plm -> ditto_plm) — difficulty dial may be invisible to the user-selected matcher |
| fusion | 0.822 | 0.702 | 0.662 | 0.527 | -0.294 | [ok] | `pydi_per_attribute_optimal -> pydi_per_attribute_optimal -> pydi_per_attribute_optimal -> pydi_per_attribute_optimal` | best-member ceiling non-increasing: 0.822 -> 0.702 -> 0.662 -> 0.527  (pydi_per_attribute_optimal -> pydi_per_attribute_optimal -> pydi_per_attribute_optimal -> pydi_per_attribute_optimal) |

## Open Questions

Signals that are direction-correct but magnitude-unspecified by the knob card. M10 should decide whether the measured delta is 'strong enough' to count as validation.

| Knob | Signal | Stage | observed delta | Card notes |
|---|---|---|---|---|
| knob_01 | em_monotone_drop | em_matching | -0.055 | Monotone F1 drop across levels. Sharper for lexical blockers than for embedding matchers. |
| knob_02 | em_monotone_drop | em_matching | -0.055 | Monotone drop. Sharper for similarity-threshold matchers than for learned matchers. Niche collisions pressurise precision. |
| knob_02 | em_pool_precision_tightens | em_matching | -0.008 | Pool precision tightens at hard: near-duplicates create false matches. |
| knob_03 | em_monotone_drop | em_matching | -0.055 | Monotone F1 drop. Learned matchers with missing-value handling degrade less than rule-based comparators. |
| knob_03 | fusion_monotone_drop | fusion | -0.294 | Fusion accuracy drops monotonically given conflict-preserving constraint and survivor cap. |
| knob_04 | em_flat_or_shift | em_matching | -0.055 | EM mostly indifferent; any shift is through the singleton-prior, direction unspecified by the card. |
| knob_04 | fusion_monotone_drop | fusion | -0.294 | Primary target. Voting-family strategies degrade faster than trust-weighted / single-best-source strategies. |
| knob_05 | em_monotone_drop | em_matching | -0.055 | Monotone drop for non-normalizing comparators; minimal for type-aware comparators. |
| knob_05 | fusion_monotone_drop | fusion | -0.294 | Monotone drop for naive strategies. Best-strategy accuracy (the max across strategies) may hold if a canonicalizing strategy is available, but overall_accuracy averages all strategies and therefore drops. |
| knob_06 | em_monotone_drop | em_matching | -0.055 | Monotone drop. Sharp for rule-based comparators, mild for learned/embedding matchers. |
| knob_06 | fusion_monotone_drop | fusion | -0.294 | Monotone drop given survivor floors. Provenance-aware fusers should be rewarded over cell-local. |
| knob_08 | sm_monotone_drop | sm | -0.145 | Primary target. Monotone drop expected. |
| knob_08 | sm_label_collapses | sm | -0.482 | Label-based string-similarity matchers collapse fast on cryptic/anonymized names. |
| knob_08 | sm_spread_is_signal | sm | 0.482 | Per the card, "the spread between matcher types IS the K8 difficulty signal". Instance-based / embedding / LLM matchers hold while label-based collapse. Spread (llm_openai.f1 - label_jaccard.f1) widens monotonically. |
| knob_08 | sm_instance_steady | sm | 0.001 | Instance-based matchers degrade more gracefully than label-based. Note: baseline instance-based F1 for companies is already 0.0 (no overlap in sampled values). This signal may be untestable on companies because the baseline is already at the floor. |
| knob_08 | em_flat | em_matching | -0.055 | Downstream stages unaffected by K8. |
| knob_10 | em_flat | em_matching | -0.055 | Upstream stages unaffected by K10. |
| knob_10 | fusion_monotone_drop | fusion | -0.294 | Primary target. Cell-local strategies (voting, most_complete, per-attribute trust) lose accuracy as unreliability grows. |

## Provenance

- Domain: products
- Expectations: `usecases_synthetic/config/knob_expected_signals.yaml`
- Baseline: `usecases_synthetic/baselines/products/baseline_metrics.json`
- Per-level metrics: `usecases_synthetic/validation/products/<level>/metrics.json`
