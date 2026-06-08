# Monotonicity + Collapse Report — papers

Cross-level analysis of easy/medium/hard variants against the baseline. See `knobs/cross_cutting.md` for the protocol.

## Cumulative Cross-Level Slope (load-bearing verdict)

Per-stage committee metric across the **cumulative** variant levels (every knob on at each level). This is the C2-contract verdict: committee scores should weakly decrease easy -> medium -> hard, and baseline (a reference value) should land no harder than medium. It makes no single-knob isolation assumption, so this -- not the per-knob signals below -- is the headline difficulty verdict.

| Stage | Metric | baseline | easy | medium | hard | easy->hard | Mono (e>=m>=h) | BasePos |
|---|---|---|---|---|---|---|---|---|
| sm | `aggregated.macro_f1` | 0.834 | 0.759 | 0.582 | 0.494 | -0.264 | [ok] | [ok] |
| norm | `aggregated.macro_f1` | 0.761 | 0.602 | 0.566 | 0.542 | -0.060 | [ok] | [ok] |
| em_blocking | `aggregated.macro_pair_recall` | 0.958 | 0.958 | 0.954 | 0.914 | -0.043 | [ok] | [ok] |
| em_matching | `aggregated.macro_f1_variant_model_on_regen_test` | 0.966 | 0.966 | 0.956 | 0.916 | -0.050 | [ok] | [ok] |
| fusion | `aggregated.overall_accuracy` | 0.610 | 0.583 | 0.477 | 0.435 | -0.147 | [ok] | [ok] |

## Per-Knob Expected Signals (indicative)

> These are **per-knob** expectations from `knob_expected_signals.yaml`, evaluated against the **cumulative** variants (every knob on at each level). They cannot isolate one knob, so a `flat` expectation for a stage that *another* knob also drives reads `[!!]` by construction (e.g. SM is not flat for knob_01 because K8 naming is also on). Treat this as indicative of combined effect; the load-bearing verdict is the Cumulative Cross-Level Slope above. For true per-knob isolation, run `generate_variant --only-knob <K>` ablations.

| Knob | Signals | OK direction | OK range | Notes |
|---|---|---|---|---|
| knob_01 | 4 | 2/4 [!!] | 0/— | [card](knobs/knob_01_surface_augmentation.md) |
| knob_02 | 4 | 2/4 [!!] | 0/— | [card](knobs/knob_02_niche_density.md) |
| knob_03 | 4 | 4/4 [ok] | 0/— | [card](knobs/knob_03_attribute_drop.md) |
| knob_04 | 4 | 2/4 [!!] | 0/— | [card](knobs/knob_04_coverage_skew.md) |
| knob_05 | 6 | 3/6 [!!] | 0/— | [card](knobs/knob_05_format_unit.md) |
| knob_06 | 5 | 4/5 [!!] | 0/— | [card](knobs/knob_06_value_noise.md) |
| knob_08 | 7 | 4/7 [!!] | 0/— | [card](knobs/knob_08_schema_naming.md) |
| knob_10 | 6 | 2/6 [!!] | 0/— | [card](knobs/knob_10_source_reliability.md) |

## Per-Signal Results

| Knob | Signal | Stage | Metric | Dir | baseline | easy | medium | hard | delta | target | Mono | Rng | BasePos | Reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| knob_01 | sm_flat | sm | `aggregated.macro_f1` | flat | 0.834 | 0.759 | 0.582 | 0.494 | -0.340 | qualitative | [!!] | [ok] | [!!] | not weakly flat: 0.759 -> 0.582 -> 0.494 |
| knob_01 | em_monotone_drop | em_matching | `aggregated.macro_f1_baseline_model_on_regen_test` | down | 0.966 | 0.964 | 0.953 | 0.914 | -0.052 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.964 -> 0.953 -> 0.914 |
| knob_01 | em_blocker_spread | em_blocking | `spread:per_member.standard_blocker.metrics.pair_recall:per_member.embedding_blocker.metrics.pair_recall` | down | NaN | NaN | NaN | NaN | NaN | qualitative | [!!] | [ok] | [ok] | metric missing at levels: easy, medium, hard |
| knob_01 | em_monotone_drop_pruned | em_matching | `aggregated.macro_f1_baseline_model_on_baseline_test` | down | 0.966 | 0.965 | 0.953 | 0.922 | -0.044 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.965 -> 0.953 -> 0.922 |
| knob_02 | em_monotone_drop | em_matching | `aggregated.macro_f1_baseline_model_on_regen_test` | down | 0.966 | 0.964 | 0.953 | 0.914 | -0.052 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.964 -> 0.953 -> 0.914 |
| knob_02 | em_pool_precision_tightens | em_matching | `aggregated.macro_precision` | down | 1.000 | 0.999 | 1.000 | 0.988 | -0.011 | qualitative | [!!] | [ok] | [ok] | not weakly down: 0.999 -> 1.000 -> 0.988 |
| knob_02 | fusion_flat | fusion | `aggregated.overall_accuracy` | flat | 0.610 | 0.583 | 0.477 | 0.435 | -0.175 | qualitative | [!!] | [ok] | [!!] | not weakly flat: 0.583 -> 0.477 -> 0.435 |
| knob_02 | em_monotone_drop_pruned | em_matching | `aggregated.macro_f1_baseline_model_on_baseline_test` | down | 0.966 | 0.965 | 0.953 | 0.922 | -0.044 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.965 -> 0.953 -> 0.922 |
| knob_03 | em_pool_recall_drops | em_blocking | `aggregated.macro_pair_recall` | down | 0.958 | 0.958 | 0.954 | 0.914 | -0.044 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.958 -> 0.954 -> 0.914 |
| knob_03 | em_monotone_drop | em_matching | `aggregated.macro_f1_baseline_model_on_regen_test` | down | 0.966 | 0.964 | 0.953 | 0.914 | -0.052 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.964 -> 0.953 -> 0.914 |
| knob_03 | fusion_monotone_drop | fusion | `aggregated.overall_accuracy` | down | 0.610 | 0.583 | 0.477 | 0.435 | -0.175 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.583 -> 0.477 -> 0.435 |
| knob_03 | em_monotone_drop_pruned | em_matching | `aggregated.macro_f1_baseline_model_on_baseline_test` | down | 0.966 | 0.965 | 0.953 | 0.922 | -0.044 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.965 -> 0.953 -> 0.922 |
| knob_04 | em_flat_or_shift | em_matching | `aggregated.macro_f1_baseline_model_on_regen_test` | flat | 0.966 | 0.964 | 0.953 | 0.914 | -0.052 | qualitative | [!!] | [ok] | [ok] | not weakly flat: 0.964 -> 0.953 -> 0.914 |
| knob_04 | fusion_monotone_drop | fusion | `aggregated.overall_accuracy` | down | 0.610 | 0.583 | 0.477 | 0.435 | -0.175 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.583 -> 0.477 -> 0.435 |
| knob_04 | fusion_spread_widens | fusion | `spread:aggregated.max_accuracy:aggregated.min_accuracy` | up | 0.136 | 0.104 | 0.098 | 0.080 | -0.056 | qualitative | [!!] | [ok] | [!!] | not weakly up: 0.104 -> 0.098 -> 0.080 |
| knob_04 | em_flat_or_shift_pruned | em_matching | `aggregated.macro_f1_baseline_model_on_baseline_test` | flat | 0.966 | 0.965 | 0.953 | 0.922 | -0.044 | qualitative | [ok] | [ok] | [ok] | weakly flat: 0.965 -> 0.953 -> 0.922 |
| knob_05 | sm_flat | sm | `aggregated.macro_f1` | flat | 0.834 | 0.759 | 0.582 | 0.494 | -0.340 | qualitative | [!!] | [ok] | [!!] | not weakly flat: 0.759 -> 0.582 -> 0.494 |
| knob_05 | em_monotone_drop | em_matching | `aggregated.macro_f1_baseline_model_on_regen_test` | down | 0.966 | 0.964 | 0.953 | 0.914 | -0.052 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.964 -> 0.953 -> 0.914 |
| knob_05 | em_pool_recall_drops_for_lexical | em_blocking | `per_member.standard_blocker.metrics.pair_recall` | down | NaN | NaN | NaN | NaN | NaN | qualitative | [!!] | [ok] | [ok] | metric missing at levels: easy, medium, hard |
| knob_05 | fusion_spread_is_signal | fusion | `spread:aggregated.max_accuracy:aggregated.min_accuracy` | up | 0.136 | 0.104 | 0.098 | 0.080 | -0.056 | qualitative | [!!] | [ok] | [!!] | not weakly up: 0.104 -> 0.098 -> 0.080 |
| knob_05 | fusion_monotone_drop | fusion | `aggregated.overall_accuracy` | down | 0.610 | 0.583 | 0.477 | 0.435 | -0.175 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.583 -> 0.477 -> 0.435 |
| knob_05 | em_monotone_drop_pruned | em_matching | `aggregated.macro_f1_baseline_model_on_baseline_test` | down | 0.966 | 0.965 | 0.953 | 0.922 | -0.044 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.965 -> 0.953 -> 0.922 |
| knob_06 | sm_flat | sm | `aggregated.macro_f1` | flat | 0.834 | 0.759 | 0.582 | 0.494 | -0.340 | qualitative | [!!] | [ok] | [!!] | not weakly flat: 0.759 -> 0.582 -> 0.494 |
| knob_06 | em_monotone_drop | em_matching | `aggregated.macro_f1_baseline_model_on_regen_test` | down | 0.966 | 0.964 | 0.953 | 0.914 | -0.052 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.964 -> 0.953 -> 0.914 |
| knob_06 | em_pool_recall_drops | em_blocking | `aggregated.macro_pair_recall` | down | 0.958 | 0.958 | 0.954 | 0.914 | -0.044 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.958 -> 0.954 -> 0.914 |
| knob_06 | fusion_monotone_drop | fusion | `aggregated.overall_accuracy` | down | 0.610 | 0.583 | 0.477 | 0.435 | -0.175 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.583 -> 0.477 -> 0.435 |
| knob_06 | em_monotone_drop_pruned | em_matching | `aggregated.macro_f1_baseline_model_on_baseline_test` | down | 0.966 | 0.965 | 0.953 | 0.922 | -0.044 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.965 -> 0.953 -> 0.922 |
| knob_08 | sm_monotone_drop | sm | `aggregated.macro_f1` | down | 0.834 | 0.759 | 0.582 | 0.494 | -0.340 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.759 -> 0.582 -> 0.494 |
| knob_08 | sm_label_collapses | sm | `per_member.label_jw.metrics.f1` | down | 0.861 | 0.812 | 0.514 | 0.262 | -0.599 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.812 -> 0.514 -> 0.262 |
| knob_08 | sm_spread_is_signal | sm | `spread:per_member.llm_openai.metrics.f1:per_member.label_jw.metrics.f1` | up | 0.111 | 0.116 | 0.391 | 0.667 | 0.556 | qualitative | [ok] | [ok] | [ok] | weakly up: 0.116 -> 0.391 -> 0.667 |
| knob_08 | sm_instance_steady | sm | `per_member.instance_tf_cosine.metrics.f1` | flat | 0.415 | 0.394 | 0.458 | 0.406 | -0.009 | qualitative | [!!] | [ok] | [ok] | not weakly flat: 0.394 -> 0.458 -> 0.406 |
| knob_08 | em_flat | em_matching | `aggregated.macro_f1_baseline_model_on_regen_test` | flat | 0.966 | 0.964 | 0.953 | 0.914 | -0.052 | qualitative | [!!] | [ok] | [ok] | not weakly flat: 0.964 -> 0.953 -> 0.914 |
| knob_08 | fusion_flat | fusion | `aggregated.overall_accuracy` | flat | 0.610 | 0.583 | 0.477 | 0.435 | -0.175 | qualitative | [!!] | [ok] | [!!] | not weakly flat: 0.583 -> 0.477 -> 0.435 |
| knob_08 | em_flat_pruned | em_matching | `aggregated.macro_f1_baseline_model_on_baseline_test` | flat | 0.966 | 0.965 | 0.953 | 0.922 | -0.044 | qualitative | [ok] | [ok] | [ok] | weakly flat: 0.965 -> 0.953 -> 0.922 |
| knob_10 | sm_flat | sm | `aggregated.macro_f1` | flat | 0.834 | 0.759 | 0.582 | 0.494 | -0.340 | qualitative | [!!] | [ok] | [!!] | not weakly flat: 0.759 -> 0.582 -> 0.494 |
| knob_10 | em_flat | em_matching | `aggregated.macro_f1_baseline_model_on_regen_test` | flat | 0.966 | 0.964 | 0.953 | 0.914 | -0.052 | qualitative | [!!] | [ok] | [ok] | not weakly flat: 0.964 -> 0.953 -> 0.914 |
| knob_10 | fusion_monotone_drop | fusion | `aggregated.overall_accuracy` | down | 0.610 | 0.583 | 0.477 | 0.435 | -0.175 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.583 -> 0.477 -> 0.435 |
| knob_10 | fusion_spread_widens | fusion | `spread:aggregated.max_accuracy:aggregated.min_accuracy` | up | 0.136 | 0.104 | 0.098 | 0.080 | -0.056 | qualitative | [!!] | [ok] | [!!] | not weakly up: 0.104 -> 0.098 -> 0.080 |
| knob_10 | fusion_voting_drops_faster_than_trust_on_name | fusion | `spread:per_attribute.name.prefer_higher_trust_only:per_attribute.name.voting_only` | up | NaN | NaN | NaN | NaN | NaN | qualitative | [!!] | [ok] | [ok] | metric missing at levels: easy, medium, hard |
| knob_10 | em_flat_pruned | em_matching | `aggregated.macro_f1_baseline_model_on_baseline_test` | flat | 0.966 | 0.965 | 0.953 | 0.922 | -0.044 | qualitative | [ok] | [ok] | [ok] | weakly flat: 0.965 -> 0.953 -> 0.922 |

## Collapses

Members with F1 < 0.15 or drop > 0.5 from baseline. EM collapses are classified against the pool-agreement diagnostic (`hidden_positive_noise` means pool is stable while test-gold moved — see `knobs/cross_cutting.md` § Protection set semantics).

| Level | Stage | Member | baseline F1 | measured F1 | delta | classification | pool delta | action |
|---|---|---|---|---|---|---|---|---|
| medium | sm | `coma_hybrid` | 0.971 | 0.376 | -0.595 | unknown | — | see knobs/cross_cutting.md § Per-knob fix-strategy defaults (SM row) |
| hard | sm | `coma_hybrid` | 0.971 | 0.186 | -0.785 | unknown | — | see knobs/cross_cutting.md § Per-knob fix-strategy defaults (SM row) |
| hard | sm | `label_jw` | 0.861 | 0.262 | -0.599 | unknown | — | see knobs/cross_cutting.md § Per-knob fix-strategy defaults (SM row) |

## Best-Member Ceiling (P8)

Per-stage best-member F1 across baseline -> easy -> medium -> hard. A valid difficulty signal must depress the *ceiling* (the user-attainable member), not just the committee mean. A flat / rising ceiling means the user-selected matcher never sees the synthetic difficulty (committee-mean drift can be masked by weak-member degradation alone).

| Stage | baseline | easy | medium | hard | delta | non-increasing | winner trail | Reason |
|---|---|---|---|---|---|---|---|---|
| sm | 0.971 | 0.929 | 0.905 | 0.929 | -0.043 | [!!] | `llm_openai -> coma_hybrid -> llm_openai -> llm_openai` | best-member ceiling did NOT decline: 0.971 -> 0.929 -> 0.905 -> 0.929  (llm_openai -> coma_hybrid -> llm_openai -> llm_openai) — difficulty dial may be invisible to the user-selected matcher |
| norm | 0.826 | 0.647 | 0.598 | 0.568 | -0.257 | [ok] | `passthrough -> passthrough -> passthrough -> passthrough` | best-member ceiling non-increasing: 0.826 -> 0.647 -> 0.598 -> 0.568  (passthrough -> passthrough -> passthrough -> passthrough) |
| em_blocking | 0.995 | 0.993 | 0.989 | 0.951 | -0.044 | [ok] | `sc_block -> sc_block -> sc_block -> sc_block` | best-member ceiling non-increasing: 0.995 -> 0.993 -> 0.989 -> 0.951  (sc_block -> sc_block -> sc_block -> sc_block) |
| em_matching | 1.000 | 0.999 | 0.999 | 0.987 | -0.012 | [!!] | `magellan -> magellan -> magellan -> ditto_plm` | best-member ceiling did NOT decline: 1.000 -> 0.999 -> 0.999 -> 0.987  (magellan -> magellan -> magellan -> ditto_plm) — difficulty dial may be invisible to the user-selected matcher |
| fusion | 0.610 | 0.583 | 0.477 | 0.435 | -0.175 | [ok] | `accusim_only -> prefer_higher_trust_only -> prefer_higher_trust_only -> llm_only` | best-member ceiling non-increasing: 0.610 -> 0.583 -> 0.477 -> 0.435  (accusim_only -> prefer_higher_trust_only -> prefer_higher_trust_only -> llm_only) |

## Open Questions

Signals that are direction-correct but magnitude-unspecified by the knob card. M10 should decide whether the measured delta is 'strong enough' to count as validation.

| Knob | Signal | Stage | observed delta | Card notes |
|---|---|---|---|---|
| knob_01 | em_monotone_drop | em_matching | -0.052 | Monotone F1 drop across levels. Sharper for lexical blockers than for embedding matchers. |
| knob_01 | em_monotone_drop_pruned | em_matching | -0.044 | Monotone F1 drop across levels. Sharper for lexical blockers than for embedding matchers. [baseline_pruned surface companion] |
| knob_02 | em_monotone_drop | em_matching | -0.052 | Monotone drop. Sharper for similarity-threshold matchers than for learned matchers. Niche collisions pressurise precision. |
| knob_02 | em_monotone_drop_pruned | em_matching | -0.044 | Monotone drop. Sharper for similarity-threshold matchers than for learned matchers. Niche collisions pressurise precision. [baseline_pruned surface companion] |
| knob_03 | em_pool_recall_drops | em_blocking | -0.044 | Candidate recall drops monotonically as drop_rate_key rises. Sharper for single-key blockers. |
| knob_03 | em_monotone_drop | em_matching | -0.052 | Monotone F1 drop. Learned matchers with missing-value handling degrade less than rule-based comparators. |
| knob_03 | fusion_monotone_drop | fusion | -0.175 | Fusion accuracy drops monotonically given conflict-preserving constraint and survivor cap. |
| knob_03 | em_monotone_drop_pruned | em_matching | -0.044 | Monotone F1 drop. Learned matchers with missing-value handling degrade less than rule-based comparators. [baseline_pruned surface companion] |
| knob_04 | fusion_monotone_drop | fusion | -0.175 | Primary target. Voting-family strategies degrade faster than trust-weighted / single-best-source strategies. |
| knob_04 | em_flat_or_shift_pruned | em_matching | -0.044 | EM mostly indifferent; any shift is through the singleton-prior, direction unspecified by the card. [baseline_pruned surface companion] |
| knob_05 | em_monotone_drop | em_matching | -0.052 | Monotone drop for non-normalizing comparators; minimal for type-aware comparators. |
| knob_05 | fusion_monotone_drop | fusion | -0.175 | Monotone drop for naive strategies. Best-strategy accuracy (the max across strategies) may hold if a canonicalizing strategy is available, but overall_accuracy averages all strategies and therefore drops. |
| knob_05 | em_monotone_drop_pruned | em_matching | -0.044 | Monotone drop for non-normalizing comparators; minimal for type-aware comparators. [baseline_pruned surface companion] |
| knob_06 | em_monotone_drop | em_matching | -0.052 | Monotone drop. Sharp for rule-based comparators, mild for learned/embedding matchers. |
| knob_06 | em_pool_recall_drops | em_blocking | -0.044 | Lexical/n-gram blockers lose recall sharply; embedding blockers mildly. |
| knob_06 | fusion_monotone_drop | fusion | -0.175 | Monotone drop given survivor floors. Provenance-aware fusers should be rewarded over cell-local. |
| knob_06 | em_monotone_drop_pruned | em_matching | -0.044 | Monotone drop. Sharp for rule-based comparators, mild for learned/embedding matchers. [baseline_pruned surface companion] |
| knob_08 | sm_monotone_drop | sm | -0.340 | Primary target. Monotone drop expected. |
| knob_08 | sm_label_collapses | sm | -0.599 | Label-based string-similarity matchers collapse fast on cryptic/anonymized names. |
| knob_08 | sm_spread_is_signal | sm | 0.556 | Per the card, "the spread between matcher types IS the K8 difficulty signal". Instance-based / embedding / LLM matchers hold while label-based collapse. Spread (llm_openai.f1 - label_jaccard.f1) widens monotonically. |
| knob_08 | em_flat_pruned | em_matching | -0.044 | Downstream stages unaffected by K8. [baseline_pruned surface companion] |
| knob_10 | fusion_monotone_drop | fusion | -0.175 | Primary target. Cell-local strategies (voting, most_complete, per-attribute trust) lose accuracy as unreliability grows. |
| knob_10 | em_flat_pruned | em_matching | -0.044 | Upstream stages unaffected by K10. [baseline_pruned surface companion] |

## Provenance

- Domain: papers
- Expectations: `usecases_synthetic/config/knob_expected_signals.yaml`
- Baseline: `usecases_synthetic/baselines/papers/baseline_metrics.json`
- Per-level metrics: `usecases_synthetic/validation/papers/<level>/metrics.json`
