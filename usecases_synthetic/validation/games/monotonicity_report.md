# Monotonicity + Collapse Report — games

Cross-level analysis of easy/medium/hard variants against the baseline. See `knobs/cross_cutting.md` for the protocol.

## Cumulative Cross-Level Slope (load-bearing verdict)

Per-stage committee metric across the **cumulative** variant levels (every knob on at each level). This is the C2-contract verdict: committee scores should weakly decrease easy -> medium -> hard, and baseline (a reference value) should land no harder than medium. It makes no single-knob isolation assumption, so this -- not the per-knob signals below -- is the headline difficulty verdict.

| Stage | Metric | baseline | easy | medium | hard | easy->hard | Mono (e>=m>=h) | BasePos |
|---|---|---|---|---|---|---|---|---|
| sm | `aggregated.macro_f1` | 0.748 | 0.871 | 0.770 | 0.709 | -0.162 | [ok] | [!!] |
| norm | `aggregated.macro_f1` | 0.897 | 0.885 | 0.872 | 0.833 | -0.052 | [ok] | [ok] |
| em_blocking | `aggregated.macro_pair_recall` | 0.956 | 0.959 | 0.954 | 0.947 | -0.011 | [ok] | [ok] |
| em_matching | `aggregated.macro_f1_variant_model_on_regen_test` | 0.609 | 0.627 | 0.672 | 0.726 | 0.099 | [!!] | [!!] |
| fusion | `aggregated.overall_accuracy` | 0.720 | 0.711 | 0.687 | 0.586 | -0.125 | [ok] | [ok] |

## Per-Knob Expected Signals (indicative)

> These are **per-knob** expectations from `knob_expected_signals.yaml`, evaluated against the **cumulative** variants (every knob on at each level). They cannot isolate one knob, so a `flat` expectation for a stage that *another* knob also drives reads `[!!]` by construction (e.g. SM is not flat for knob_01 because K8 naming is also on). Treat this as indicative of combined effect; the load-bearing verdict is the Cumulative Cross-Level Slope above. For true per-knob isolation, run `generate_variant --only-knob <K>` ablations.

| Knob | Signals | OK direction | OK range | Notes |
|---|---|---|---|---|
| knob_01 | 4 | 1/4 [!!] | 0/— | [card](knobs/knob_01_surface_augmentation.md) |
| knob_02 | 4 | 0/4 [!!] | 0/— | [card](knobs/knob_02_niche_density.md) |
| knob_03 | 4 | 2/4 [!!] | 0/— | [card](knobs/knob_03_attribute_drop.md) |
| knob_04 | 4 | 1/4 [!!] | 0/— | [card](knobs/knob_04_coverage_skew.md) |
| knob_05 | 6 | 2/6 [!!] | 0/— | [card](knobs/knob_05_format_unit.md) |
| knob_06 | 5 | 2/5 [!!] | 0/— | [card](knobs/knob_06_value_noise.md) |
| knob_08 | 7 | 4/7 [!!] | 0/— | [card](knobs/knob_08_schema_naming.md) |
| knob_10 | 6 | 1/6 [!!] | 0/— | [card](knobs/knob_10_source_reliability.md) |

## Per-Signal Results

| Knob | Signal | Stage | Metric | Dir | baseline | easy | medium | hard | delta | target | Mono | Rng | BasePos | Reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| knob_01 | sm_flat | sm | `aggregated.macro_f1` | flat | 0.748 | 0.871 | 0.770 | 0.709 | -0.039 | qualitative | [!!] | [ok] | [ok] | not weakly flat: 0.871 -> 0.770 -> 0.709 |
| knob_01 | em_monotone_drop | em_matching | `aggregated.macro_f1_baseline_model_on_regen_test` | down | 0.609 | 0.633 | 0.647 | 0.683 | 0.074 | qualitative | [!!] | [ok] | [!!] | not weakly down: 0.633 -> 0.647 -> 0.683 |
| knob_01 | em_blocker_spread | em_blocking | `spread:per_member.standard_blocker.metrics.pair_recall:per_member.embedding_blocker.metrics.pair_recall` | down | 0.047 | 0.048 | 0.024 | -0.022 | -0.069 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.048 -> 0.024 -> -0.022 |
| knob_01 | em_monotone_drop_pruned | em_matching | `aggregated.macro_f1_baseline_model_on_baseline_test` | down | 0.609 | 0.633 | 0.648 | 0.553 | -0.057 | qualitative | [!!] | [ok] | [!!] | not weakly down: 0.633 -> 0.648 -> 0.553 |
| knob_02 | em_monotone_drop | em_matching | `aggregated.macro_f1_baseline_model_on_regen_test` | down | 0.609 | 0.633 | 0.647 | 0.683 | 0.074 | qualitative | [!!] | [ok] | [!!] | not weakly down: 0.633 -> 0.647 -> 0.683 |
| knob_02 | em_pool_precision_tightens | em_matching | `aggregated.macro_precision` | down | 0.867 | 0.882 | 0.802 | 0.831 | -0.036 | qualitative | [!!] | [ok] | [ok] | not weakly down: 0.882 -> 0.802 -> 0.831 |
| knob_02 | fusion_flat | fusion | `aggregated.overall_accuracy` | flat | 0.720 | 0.711 | 0.687 | 0.586 | -0.134 | qualitative | [!!] | [ok] | [ok] | not weakly flat: 0.711 -> 0.687 -> 0.586 |
| knob_02 | em_monotone_drop_pruned | em_matching | `aggregated.macro_f1_baseline_model_on_baseline_test` | down | 0.609 | 0.633 | 0.648 | 0.553 | -0.057 | qualitative | [!!] | [ok] | [!!] | not weakly down: 0.633 -> 0.648 -> 0.553 |
| knob_03 | em_pool_recall_drops | em_blocking | `aggregated.macro_pair_recall` | down | 0.956 | 0.959 | 0.954 | 0.947 | -0.009 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.959 -> 0.954 -> 0.947 |
| knob_03 | em_monotone_drop | em_matching | `aggregated.macro_f1_baseline_model_on_regen_test` | down | 0.609 | 0.633 | 0.647 | 0.683 | 0.074 | qualitative | [!!] | [ok] | [!!] | not weakly down: 0.633 -> 0.647 -> 0.683 |
| knob_03 | fusion_monotone_drop | fusion | `aggregated.overall_accuracy` | down | 0.720 | 0.711 | 0.687 | 0.586 | -0.134 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.711 -> 0.687 -> 0.586 |
| knob_03 | em_monotone_drop_pruned | em_matching | `aggregated.macro_f1_baseline_model_on_baseline_test` | down | 0.609 | 0.633 | 0.648 | 0.553 | -0.057 | qualitative | [!!] | [ok] | [!!] | not weakly down: 0.633 -> 0.648 -> 0.553 |
| knob_04 | em_flat_or_shift | em_matching | `aggregated.macro_f1_baseline_model_on_regen_test` | flat | 0.609 | 0.633 | 0.647 | 0.683 | 0.074 | qualitative | [!!] | [ok] | [ok] | not weakly flat: 0.633 -> 0.647 -> 0.683 |
| knob_04 | fusion_monotone_drop | fusion | `aggregated.overall_accuracy` | down | 0.720 | 0.711 | 0.687 | 0.586 | -0.134 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.711 -> 0.687 -> 0.586 |
| knob_04 | fusion_spread_widens | fusion | `spread:aggregated.max_accuracy:aggregated.min_accuracy` | up | 0.107 | 0.104 | 0.092 | 0.067 | -0.039 | qualitative | [!!] | [ok] | [!!] | not weakly up: 0.104 -> 0.092 -> 0.067 |
| knob_04 | em_flat_or_shift_pruned | em_matching | `aggregated.macro_f1_baseline_model_on_baseline_test` | flat | 0.609 | 0.633 | 0.648 | 0.553 | -0.057 | qualitative | [!!] | [ok] | [ok] | not weakly flat: 0.633 -> 0.648 -> 0.553 |
| knob_05 | sm_flat | sm | `aggregated.macro_f1` | flat | 0.748 | 0.871 | 0.770 | 0.709 | -0.039 | qualitative | [!!] | [ok] | [ok] | not weakly flat: 0.871 -> 0.770 -> 0.709 |
| knob_05 | em_monotone_drop | em_matching | `aggregated.macro_f1_baseline_model_on_regen_test` | down | 0.609 | 0.633 | 0.647 | 0.683 | 0.074 | qualitative | [!!] | [ok] | [!!] | not weakly down: 0.633 -> 0.647 -> 0.683 |
| knob_05 | em_pool_recall_drops_for_lexical | em_blocking | `per_member.standard_blocker.metrics.pair_recall` | down | 0.986 | 0.990 | 0.967 | 0.944 | -0.042 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.990 -> 0.967 -> 0.944 |
| knob_05 | fusion_spread_is_signal | fusion | `spread:aggregated.max_accuracy:aggregated.min_accuracy` | up | 0.107 | 0.104 | 0.092 | 0.067 | -0.039 | qualitative | [!!] | [ok] | [!!] | not weakly up: 0.104 -> 0.092 -> 0.067 |
| knob_05 | fusion_monotone_drop | fusion | `aggregated.overall_accuracy` | down | 0.720 | 0.711 | 0.687 | 0.586 | -0.134 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.711 -> 0.687 -> 0.586 |
| knob_05 | em_monotone_drop_pruned | em_matching | `aggregated.macro_f1_baseline_model_on_baseline_test` | down | 0.609 | 0.633 | 0.648 | 0.553 | -0.057 | qualitative | [!!] | [ok] | [!!] | not weakly down: 0.633 -> 0.648 -> 0.553 |
| knob_06 | sm_flat | sm | `aggregated.macro_f1` | flat | 0.748 | 0.871 | 0.770 | 0.709 | -0.039 | qualitative | [!!] | [ok] | [ok] | not weakly flat: 0.871 -> 0.770 -> 0.709 |
| knob_06 | em_monotone_drop | em_matching | `aggregated.macro_f1_baseline_model_on_regen_test` | down | 0.609 | 0.633 | 0.647 | 0.683 | 0.074 | qualitative | [!!] | [ok] | [!!] | not weakly down: 0.633 -> 0.647 -> 0.683 |
| knob_06 | em_pool_recall_drops | em_blocking | `aggregated.macro_pair_recall` | down | 0.956 | 0.959 | 0.954 | 0.947 | -0.009 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.959 -> 0.954 -> 0.947 |
| knob_06 | fusion_monotone_drop | fusion | `aggregated.overall_accuracy` | down | 0.720 | 0.711 | 0.687 | 0.586 | -0.134 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.711 -> 0.687 -> 0.586 |
| knob_06 | em_monotone_drop_pruned | em_matching | `aggregated.macro_f1_baseline_model_on_baseline_test` | down | 0.609 | 0.633 | 0.648 | 0.553 | -0.057 | qualitative | [!!] | [ok] | [!!] | not weakly down: 0.633 -> 0.648 -> 0.553 |
| knob_08 | sm_monotone_drop | sm | `aggregated.macro_f1` | down | 0.748 | 0.871 | 0.770 | 0.709 | -0.039 | qualitative | [ok] | [ok] | [!!] | weakly down: 0.871 -> 0.770 -> 0.709 |
| knob_08 | sm_label_collapses | sm | `per_member.label_jw.metrics.f1` | down | 0.343 | 0.981 | 0.619 | 0.303 | -0.040 | qualitative | [ok] | [ok] | [!!] | weakly down: 0.981 -> 0.619 -> 0.303 |
| knob_08 | sm_spread_is_signal | sm | `spread:per_member.llm_openai.metrics.f1:per_member.label_jw.metrics.f1` | up | 0.657 | 0.000 | 0.362 | 0.678 | 0.021 | qualitative | [ok] | [ok] | [!!] | weakly up: 0.000 -> 0.362 -> 0.678 |
| knob_08 | sm_instance_steady | sm | `per_member.instance_tf_cosine.metrics.f1` | flat | 0.720 | 0.653 | 0.638 | 0.667 | -0.053 | qualitative | [ok] | [ok] | [!!] | weakly flat: 0.653 -> 0.638 -> 0.667 |
| knob_08 | em_flat | em_matching | `aggregated.macro_f1_baseline_model_on_regen_test` | flat | 0.609 | 0.633 | 0.647 | 0.683 | 0.074 | qualitative | [!!] | [ok] | [ok] | not weakly flat: 0.633 -> 0.647 -> 0.683 |
| knob_08 | fusion_flat | fusion | `aggregated.overall_accuracy` | flat | 0.720 | 0.711 | 0.687 | 0.586 | -0.134 | qualitative | [!!] | [ok] | [ok] | not weakly flat: 0.711 -> 0.687 -> 0.586 |
| knob_08 | em_flat_pruned | em_matching | `aggregated.macro_f1_baseline_model_on_baseline_test` | flat | 0.609 | 0.633 | 0.648 | 0.553 | -0.057 | qualitative | [!!] | [ok] | [ok] | not weakly flat: 0.633 -> 0.648 -> 0.553 |
| knob_10 | sm_flat | sm | `aggregated.macro_f1` | flat | 0.748 | 0.871 | 0.770 | 0.709 | -0.039 | qualitative | [!!] | [ok] | [ok] | not weakly flat: 0.871 -> 0.770 -> 0.709 |
| knob_10 | em_flat | em_matching | `aggregated.macro_f1_baseline_model_on_regen_test` | flat | 0.609 | 0.633 | 0.647 | 0.683 | 0.074 | qualitative | [!!] | [ok] | [ok] | not weakly flat: 0.633 -> 0.647 -> 0.683 |
| knob_10 | fusion_monotone_drop | fusion | `aggregated.overall_accuracy` | down | 0.720 | 0.711 | 0.687 | 0.586 | -0.134 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.711 -> 0.687 -> 0.586 |
| knob_10 | fusion_spread_widens | fusion | `spread:aggregated.max_accuracy:aggregated.min_accuracy` | up | 0.107 | 0.104 | 0.092 | 0.067 | -0.039 | qualitative | [!!] | [ok] | [!!] | not weakly up: 0.104 -> 0.092 -> 0.067 |
| knob_10 | fusion_voting_drops_faster_than_trust_on_name | fusion | `spread:per_attribute.name.prefer_higher_trust_only:per_attribute.name.voting_only` | up | 0.070 | 0.040 | 0.030 | 0.048 | -0.022 | qualitative | [!!] | [ok] | [!!] | not weakly up: 0.040 -> 0.030 -> 0.048 |
| knob_10 | em_flat_pruned | em_matching | `aggregated.macro_f1_baseline_model_on_baseline_test` | flat | 0.609 | 0.633 | 0.648 | 0.553 | -0.057 | qualitative | [!!] | [ok] | [ok] | not weakly flat: 0.633 -> 0.648 -> 0.553 |

## Collapses

No members fell below the collapse threshold.

## Best-Member Ceiling (P8)

Per-stage best-member F1 across baseline -> easy -> medium -> hard. A valid difficulty signal must depress the *ceiling* (the user-attainable member), not just the committee mean. A flat / rising ceiling means the user-selected matcher never sees the synthetic difficulty (committee-mean drift can be masked by weak-member degradation alone).

| Stage | baseline | easy | medium | hard | delta | non-increasing | winner trail | Reason |
|---|---|---|---|---|---|---|---|---|
| sm | 1.000 | 0.981 | 0.981 | 0.981 | -0.019 | [ok] | `llm_openai -> coma_hybrid -> llm_openai -> llm_openai` | best-member ceiling non-increasing: 1.000 -> 0.981 -> 0.981 -> 0.981  (llm_openai -> coma_hybrid -> llm_openai -> llm_openai) |
| norm | 0.981 | 0.961 | 0.944 | 0.858 | -0.123 | [ok] | `passthrough -> passthrough -> passthrough -> passthrough` | best-member ceiling non-increasing: 0.981 -> 0.961 -> 0.944 -> 0.858  (passthrough -> passthrough -> passthrough -> passthrough) |
| em_blocking | 1.000 | 1.000 | 1.000 | 0.991 | -0.009 | [ok] | `token_blocker -> token_blocker -> token_blocker -> token_blocker` | best-member ceiling non-increasing: 1.000 -> 1.000 -> 1.000 -> 0.991  (token_blocker -> token_blocker -> token_blocker -> token_blocker) |
| em_matching | 0.716 | 0.710 | 0.768 | 0.823 | 0.107 | [!!] | `ditto_plm -> ditto_plm -> ditto_plm -> ditto_plm` | best-member ceiling did NOT decline: 0.716 -> 0.710 -> 0.768 -> 0.823  (ditto_plm -> ditto_plm -> ditto_plm -> ditto_plm) — difficulty dial may be invisible to the user-selected matcher |
| fusion | 0.720 | 0.711 | 0.687 | 0.586 | -0.134 | [ok] | `fusionquery_only -> prefer_higher_trust_only -> truthfinder_only -> fusionquery_only` | best-member ceiling non-increasing: 0.720 -> 0.711 -> 0.687 -> 0.586  (fusionquery_only -> prefer_higher_trust_only -> truthfinder_only -> fusionquery_only) |

## Open Questions

Signals that are direction-correct but magnitude-unspecified by the knob card. M10 should decide whether the measured delta is 'strong enough' to count as validation.

| Knob | Signal | Stage | observed delta | Card notes |
|---|---|---|---|---|
| knob_01 | em_blocker_spread | em_blocking | -0.069 | Spread between the lexical+rule blocker (standard_rule) and embedding+rule blocker (embedding_rule) should shrink or even invert as lexical suffers more than embedding on surface perturbations. (Direction is 'down' because baseline spread is positive with lexical ahead.) |
| knob_03 | em_pool_recall_drops | em_blocking | -0.009 | Candidate recall drops monotonically as drop_rate_key rises. Sharper for single-key blockers. |
| knob_03 | fusion_monotone_drop | fusion | -0.134 | Fusion accuracy drops monotonically given conflict-preserving constraint and survivor cap. |
| knob_04 | fusion_monotone_drop | fusion | -0.134 | Primary target. Voting-family strategies degrade faster than trust-weighted / single-best-source strategies. |
| knob_05 | em_pool_recall_drops_for_lexical | em_blocking | -0.042 | Raw-string blocker (lexical+rule) sees a monotone pool_recall drop; embedding+rule blocker holds steadier. |
| knob_05 | fusion_monotone_drop | fusion | -0.134 | Monotone drop for naive strategies. Best-strategy accuracy (the max across strategies) may hold if a canonicalizing strategy is available, but overall_accuracy averages all strategies and therefore drops. |
| knob_06 | em_pool_recall_drops | em_blocking | -0.009 | Lexical/n-gram blockers lose recall sharply; embedding blockers mildly. |
| knob_06 | fusion_monotone_drop | fusion | -0.134 | Monotone drop given survivor floors. Provenance-aware fusers should be rewarded over cell-local. |
| knob_08 | sm_monotone_drop | sm | -0.039 | Primary target. Monotone drop expected. |
| knob_08 | sm_label_collapses | sm | -0.040 | Label-based string-similarity matchers collapse fast on cryptic/anonymized names. |
| knob_08 | sm_spread_is_signal | sm | 0.021 | Per the card, "the spread between matcher types IS the K8 difficulty signal". Instance-based / embedding / LLM matchers hold while label-based collapse. Spread (llm_openai.f1 - label_jaccard.f1) widens monotonically. |
| knob_08 | sm_instance_steady | sm | -0.053 | Instance-based matchers degrade more gracefully than label-based. Note: baseline instance-based F1 for companies is already 0.0 (no overlap in sampled values). This signal may be untestable on companies because the baseline is already at the floor. |
| knob_10 | fusion_monotone_drop | fusion | -0.134 | Primary target. Cell-local strategies (voting, most_complete, per-attribute trust) lose accuracy as unreliability grows. |

## Provenance

- Domain: games
- Expectations: `usecases_synthetic/config/knob_expected_signals.yaml`
- Baseline: `usecases_synthetic/baselines/games/baseline_metrics.json`
- Per-level metrics: `usecases_synthetic/validation/games/<level>/metrics.json`
