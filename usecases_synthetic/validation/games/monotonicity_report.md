# Monotonicity + Collapse Report — games

Cross-level analysis of easy/medium/hard variants against the baseline. See `knobs/cross_cutting.md` for the protocol.

## Cumulative Cross-Level Slope (load-bearing verdict)

Per-stage committee metric across the **cumulative** variant levels (every knob on at each level). This is the C2-contract verdict: committee scores should weakly decrease easy -> medium -> hard, and baseline (a reference value) should land no harder than medium. It makes no single-knob isolation assumption, so this -- not the per-knob signals below -- is the headline difficulty verdict.

| Stage | Metric | baseline | easy | medium | hard | easy->hard | Mono (e>=m>=h) | BasePos |
|---|---|---|---|---|---|---|---|---|
| sm | `aggregated.macro_f1` | 0.748 | 0.869 | 0.768 | 0.693 | -0.176 | [ok] | [!!] |
| norm | `aggregated.macro_f1` | 0.897 | 0.885 | 0.871 | 0.840 | -0.046 | [ok] | [ok] |
| em_blocking | `aggregated.macro_pair_recall` | 0.956 | 0.955 | 0.940 | 0.734 | -0.221 | [ok] | [ok] |
| em_matching | `aggregated.macro_f1_variant_model_on_regen_test` | 0.609 | 0.613 | 0.657 | 0.556 | -0.057 | [!!] | [!!] |
| fusion | `aggregated.overall_accuracy` | 0.720 | 0.711 | 0.690 | 0.585 | -0.127 | [ok] | [ok] |

## Per-Knob Expected Signals (indicative)

> These are **per-knob** expectations from `knob_expected_signals.yaml`, evaluated against the **cumulative** variants (every knob on at each level). They cannot isolate one knob, so a `flat` expectation for a stage that *another* knob also drives reads `[!!]` by construction (e.g. SM is not flat for knob_01 because K8 naming is also on). Treat this as indicative of combined effect; the load-bearing verdict is the Cumulative Cross-Level Slope above. For true per-knob isolation, run `generate_variant --only-knob <K>` ablations.

| Knob | Signals | OK direction | OK range | Notes |
|---|---|---|---|---|
| knob_01 | 3 | 0/3 [!!] | 0/— | [card](knobs/knob_01_surface_augmentation.md) |
| knob_02 | 3 | 1/3 [!!] | 0/— | [card](knobs/knob_02_niche_density.md) |
| knob_03 | 3 | 2/3 [!!] | 0/— | [card](knobs/knob_03_attribute_drop.md) |
| knob_04 | 3 | 1/3 [!!] | 0/— | [card](knobs/knob_04_coverage_skew.md) |
| knob_05 | 5 | 2/5 [!!] | 0/— | [card](knobs/knob_05_format_unit.md) |
| knob_06 | 4 | 2/4 [!!] | 0/— | [card](knobs/knob_06_value_noise.md) |
| knob_08 | 6 | 4/6 [!!] | 0/— | [card](knobs/knob_08_schema_naming.md) |
| knob_10 | 5 | 1/5 [!!] | 0/— | [card](knobs/knob_10_source_reliability.md) |

## Per-Signal Results

| Knob | Signal | Stage | Metric | Dir | baseline | easy | medium | hard | delta | target | Mono | Rng | BasePos | Reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| knob_01 | sm_flat | sm | `aggregated.macro_f1` | flat | 0.748 | 0.869 | 0.768 | 0.693 | -0.055 | qualitative | [!!] | [ok] | [ok] | not weakly flat: 0.869 -> 0.768 -> 0.693 |
| knob_01 | em_monotone_drop | em_matching | `aggregated.macro_f1_variant_model_on_regen_test` | down | 0.609 | 0.613 | 0.657 | 0.556 | -0.053 | qualitative | [!!] | [ok] | [!!] | not weakly down: 0.613 -> 0.657 -> 0.556 |
| knob_01 | em_blocker_spread | em_blocking | `spread:per_member.standard_blocker.metrics.pair_recall:per_member.embedding_blocker.metrics.pair_recall` | down | 0.047 | 0.028 | 0.029 | -0.003 | -0.051 | qualitative | [!!] | [ok] | [ok] | not weakly down: 0.028 -> 0.029 -> -0.003 |
| knob_02 | em_monotone_drop | em_matching | `aggregated.macro_f1_variant_model_on_regen_test` | down | 0.609 | 0.613 | 0.657 | 0.556 | -0.053 | qualitative | [!!] | [ok] | [!!] | not weakly down: 0.613 -> 0.657 -> 0.556 |
| knob_02 | em_pool_precision_tightens | em_matching | `aggregated.macro_precision` | down | 0.867 | 0.868 | 0.798 | 0.774 | -0.093 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.868 -> 0.798 -> 0.774 |
| knob_02 | fusion_flat | fusion | `aggregated.overall_accuracy` | flat | 0.720 | 0.711 | 0.690 | 0.585 | -0.136 | qualitative | [!!] | [ok] | [ok] | not weakly flat: 0.711 -> 0.690 -> 0.585 |
| knob_03 | em_pool_recall_drops | em_blocking | `aggregated.macro_pair_recall` | down | 0.956 | 0.955 | 0.940 | 0.734 | -0.222 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.955 -> 0.940 -> 0.734 |
| knob_03 | em_monotone_drop | em_matching | `aggregated.macro_f1_variant_model_on_regen_test` | down | 0.609 | 0.613 | 0.657 | 0.556 | -0.053 | qualitative | [!!] | [ok] | [!!] | not weakly down: 0.613 -> 0.657 -> 0.556 |
| knob_03 | fusion_monotone_drop | fusion | `aggregated.overall_accuracy` | down | 0.720 | 0.711 | 0.690 | 0.585 | -0.136 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.711 -> 0.690 -> 0.585 |
| knob_04 | em_flat_or_shift | em_matching | `aggregated.macro_f1_variant_model_on_regen_test` | flat | 0.609 | 0.613 | 0.657 | 0.556 | -0.053 | qualitative | [!!] | [ok] | [ok] | not weakly flat: 0.613 -> 0.657 -> 0.556 |
| knob_04 | fusion_monotone_drop | fusion | `aggregated.overall_accuracy` | down | 0.720 | 0.711 | 0.690 | 0.585 | -0.136 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.711 -> 0.690 -> 0.585 |
| knob_04 | fusion_spread_widens | fusion | `spread:aggregated.max_accuracy:aggregated.min_accuracy` | up | 0.107 | 0.104 | 0.091 | 0.056 | -0.051 | qualitative | [!!] | [ok] | [!!] | not weakly up: 0.104 -> 0.091 -> 0.056 |
| knob_05 | sm_flat | sm | `aggregated.macro_f1` | flat | 0.748 | 0.869 | 0.768 | 0.693 | -0.055 | qualitative | [!!] | [ok] | [ok] | not weakly flat: 0.869 -> 0.768 -> 0.693 |
| knob_05 | em_monotone_drop | em_matching | `aggregated.macro_f1_variant_model_on_regen_test` | down | 0.609 | 0.613 | 0.657 | 0.556 | -0.053 | qualitative | [!!] | [ok] | [!!] | not weakly down: 0.613 -> 0.657 -> 0.556 |
| knob_05 | em_pool_recall_drops_for_lexical | em_blocking | `per_member.standard_blocker.metrics.pair_recall` | down | 0.986 | 0.986 | 0.958 | 0.737 | -0.249 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.986 -> 0.958 -> 0.737 |
| knob_05 | fusion_spread_is_signal | fusion | `spread:aggregated.max_accuracy:aggregated.min_accuracy` | up | 0.107 | 0.104 | 0.091 | 0.056 | -0.051 | qualitative | [!!] | [ok] | [!!] | not weakly up: 0.104 -> 0.091 -> 0.056 |
| knob_05 | fusion_monotone_drop | fusion | `aggregated.overall_accuracy` | down | 0.720 | 0.711 | 0.690 | 0.585 | -0.136 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.711 -> 0.690 -> 0.585 |
| knob_06 | sm_flat | sm | `aggregated.macro_f1` | flat | 0.748 | 0.869 | 0.768 | 0.693 | -0.055 | qualitative | [!!] | [ok] | [ok] | not weakly flat: 0.869 -> 0.768 -> 0.693 |
| knob_06 | em_monotone_drop | em_matching | `aggregated.macro_f1_variant_model_on_regen_test` | down | 0.609 | 0.613 | 0.657 | 0.556 | -0.053 | qualitative | [!!] | [ok] | [!!] | not weakly down: 0.613 -> 0.657 -> 0.556 |
| knob_06 | em_pool_recall_drops | em_blocking | `aggregated.macro_pair_recall` | down | 0.956 | 0.955 | 0.940 | 0.734 | -0.222 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.955 -> 0.940 -> 0.734 |
| knob_06 | fusion_monotone_drop | fusion | `aggregated.overall_accuracy` | down | 0.720 | 0.711 | 0.690 | 0.585 | -0.136 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.711 -> 0.690 -> 0.585 |
| knob_08 | sm_monotone_drop | sm | `aggregated.macro_f1` | down | 0.748 | 0.869 | 0.768 | 0.693 | -0.055 | qualitative | [ok] | [ok] | [!!] | weakly down: 0.869 -> 0.768 -> 0.693 |
| knob_08 | sm_label_collapses | sm | `per_member.label_jw.metrics.f1` | down | 0.343 | 0.981 | 0.619 | 0.303 | -0.040 | qualitative | [ok] | [ok] | [!!] | weakly down: 0.981 -> 0.619 -> 0.303 |
| knob_08 | sm_spread_is_signal | sm | `spread:per_member.llm_openai.metrics.f1:per_member.label_jw.metrics.f1` | up | 0.657 | 0.000 | 0.362 | 0.660 | 0.003 | qualitative | [ok] | [ok] | [!!] | weakly up: 0.000 -> 0.362 -> 0.660 |
| knob_08 | sm_instance_steady | sm | `per_member.instance_tf_cosine.metrics.f1` | flat | 0.720 | 0.653 | 0.638 | 0.667 | -0.053 | qualitative | [ok] | [ok] | [!!] | weakly flat: 0.653 -> 0.638 -> 0.667 |
| knob_08 | em_flat | em_matching | `aggregated.macro_f1_variant_model_on_regen_test` | flat | 0.609 | 0.613 | 0.657 | 0.556 | -0.053 | qualitative | [!!] | [ok] | [ok] | not weakly flat: 0.613 -> 0.657 -> 0.556 |
| knob_08 | fusion_flat | fusion | `aggregated.overall_accuracy` | flat | 0.720 | 0.711 | 0.690 | 0.585 | -0.136 | qualitative | [!!] | [ok] | [ok] | not weakly flat: 0.711 -> 0.690 -> 0.585 |
| knob_10 | sm_flat | sm | `aggregated.macro_f1` | flat | 0.748 | 0.869 | 0.768 | 0.693 | -0.055 | qualitative | [!!] | [ok] | [ok] | not weakly flat: 0.869 -> 0.768 -> 0.693 |
| knob_10 | em_flat | em_matching | `aggregated.macro_f1_variant_model_on_regen_test` | flat | 0.609 | 0.613 | 0.657 | 0.556 | -0.053 | qualitative | [!!] | [ok] | [ok] | not weakly flat: 0.613 -> 0.657 -> 0.556 |
| knob_10 | fusion_monotone_drop | fusion | `aggregated.overall_accuracy` | down | 0.720 | 0.711 | 0.690 | 0.585 | -0.136 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.711 -> 0.690 -> 0.585 |
| knob_10 | fusion_spread_widens | fusion | `spread:aggregated.max_accuracy:aggregated.min_accuracy` | up | 0.107 | 0.104 | 0.091 | 0.056 | -0.051 | qualitative | [!!] | [ok] | [!!] | not weakly up: 0.104 -> 0.091 -> 0.056 |
| knob_10 | fusion_voting_drops_faster_than_trust_on_name | fusion | `spread:per_attribute.name.prefer_higher_trust_only:per_attribute.name.voting_only` | up | 0.070 | 0.040 | 0.060 | 0.036 | -0.034 | qualitative | [!!] | [ok] | [!!] | not weakly up: 0.040 -> 0.060 -> 0.036 |

## Collapses

No members fell below the collapse threshold.

## Best-Member Ceiling (P8)

Per-stage best-member F1 across baseline -> easy -> medium -> hard. A valid difficulty signal must depress the *ceiling* (the user-attainable member), not just the committee mean. A flat / rising ceiling means the user-selected matcher never sees the synthetic difficulty (committee-mean drift can be masked by weak-member degradation alone).

| Stage | baseline | easy | medium | hard | delta | non-increasing | winner trail | Reason |
|---|---|---|---|---|---|---|---|---|
| sm | 1.000 | 0.981 | 0.981 | 0.963 | -0.037 | [ok] | `llm_openai -> llm_openai -> llm_openai -> llm_openai` | best-member ceiling non-increasing: 1.000 -> 0.981 -> 0.981 -> 0.963  (llm_openai -> llm_openai -> llm_openai -> llm_openai) |
| norm | 0.981 | 0.961 | 0.943 | 0.852 | -0.129 | [ok] | `passthrough -> passthrough -> passthrough -> passthrough` | best-member ceiling non-increasing: 0.981 -> 0.961 -> 0.943 -> 0.852  (passthrough -> passthrough -> passthrough -> passthrough) |
| em_blocking | 1.000 | 0.995 | 0.986 | 0.774 | -0.226 | [ok] | `token_blocker -> token_blocker -> token_blocker -> token_blocker` | best-member ceiling non-increasing: 1.000 -> 0.995 -> 0.986 -> 0.774  (token_blocker -> token_blocker -> token_blocker -> token_blocker) |
| em_matching | 0.716 | 0.667 | 0.755 | 0.658 | -0.057 | [!!] | `ditto_plm -> ditto_plm -> magellan -> magellan` | best-member ceiling did NOT decline: 0.716 -> 0.667 -> 0.755 -> 0.658  (ditto_plm -> ditto_plm -> magellan -> magellan) — difficulty dial may be invisible to the user-selected matcher |
| fusion | 0.720 | 0.711 | 0.690 | 0.585 | -0.136 | [ok] | `fusionquery_only -> prefer_higher_trust_only -> truthfinder_only -> fusionquery_only` | best-member ceiling non-increasing: 0.720 -> 0.711 -> 0.690 -> 0.585  (fusionquery_only -> prefer_higher_trust_only -> truthfinder_only -> fusionquery_only) |

## Open Questions

Signals that are direction-correct but magnitude-unspecified by the knob card. M10 should decide whether the measured delta is 'strong enough' to count as validation.

| Knob | Signal | Stage | observed delta | Card notes |
|---|---|---|---|---|
| knob_02 | em_pool_precision_tightens | em_matching | -0.093 | Pool precision tightens at hard: near-duplicates create false matches. |
| knob_03 | em_pool_recall_drops | em_blocking | -0.222 | Candidate recall drops monotonically as drop_rate_key rises. Sharper for single-key blockers. |
| knob_03 | fusion_monotone_drop | fusion | -0.136 | Fusion accuracy drops monotonically given conflict-preserving constraint and survivor cap. |
| knob_04 | fusion_monotone_drop | fusion | -0.136 | Primary target. Voting-family strategies degrade faster than trust-weighted / single-best-source strategies. |
| knob_05 | em_pool_recall_drops_for_lexical | em_blocking | -0.249 | Raw-string blocker (lexical+rule) sees a monotone pool_recall drop; embedding+rule blocker holds steadier. |
| knob_05 | fusion_monotone_drop | fusion | -0.136 | Monotone drop for naive strategies. Best-strategy accuracy (the max across strategies) may hold if a canonicalizing strategy is available, but overall_accuracy averages all strategies and therefore drops. |
| knob_06 | em_pool_recall_drops | em_blocking | -0.222 | Lexical/n-gram blockers lose recall sharply; embedding blockers mildly. |
| knob_06 | fusion_monotone_drop | fusion | -0.136 | Monotone drop given survivor floors. Provenance-aware fusers should be rewarded over cell-local. |
| knob_08 | sm_monotone_drop | sm | -0.055 | Primary target. Monotone drop expected. |
| knob_08 | sm_label_collapses | sm | -0.040 | Label-based string-similarity matchers collapse fast on cryptic/anonymized names. |
| knob_08 | sm_spread_is_signal | sm | 0.003 | Per the card, "the spread between matcher types IS the K8 difficulty signal". Instance-based / embedding / LLM matchers hold while label-based collapse. Spread (llm_openai.f1 - label_jaccard.f1) widens monotonically. |
| knob_08 | sm_instance_steady | sm | -0.053 | Instance-based matchers degrade more gracefully than label-based. Note: baseline instance-based F1 for companies is already 0.0 (no overlap in sampled values). This signal may be untestable on companies because the baseline is already at the floor. |
| knob_10 | fusion_monotone_drop | fusion | -0.136 | Primary target. Cell-local strategies (voting, most_complete, per-attribute trust) lose accuracy as unreliability grows. |

## Provenance

- Domain: games
- Expectations: `usecases_synthetic/config/knob_expected_signals.yaml`
- Baseline: `usecases_synthetic/baselines/games/baseline_metrics.json`
- Per-level metrics: `usecases_synthetic/validation/games/<level>/metrics.json`
