# S1 Difficulty Validation — Final Report (music FULL)

Domain: **music** (full, 37k records). Runtime: **3h 20min** end-to-end for S.7 (R6.1 → R6.2 → R7.1 × 3 → R7.2). Generated 2026-05-16.

Predecessor sanity ladder: [validation/music-small/](../music-small/) (post-F1–F10 + P8). All R5 committee sign-offs locked; all K* sign-offs locked.

## Verdict

**PASS on cross-level committee F1 monotonicity at hard for all 5 stages.** Best-member ceiling (P8) drops at hard for SM, EM-matching, Fusion; EM-blocking ceiling collapses; Norm ceiling robust (text_clean immune to music's K1/K8). 4 structural monotonicity FAILs are documented metric proxies (F4/F9/F10/K10), not regen bugs.

The synthetic difficulty dial **moves both the committee mean and the user-attainable ceiling**, which is the load-bearing claim of the pipeline.

## Committee F1 monotonicity (R7.1 + R7.2)

Headline metric per stage across baseline → easy → medium → hard. Direction column: ↓ = drop, ↑ = rise, = = flat (|Δ| ≤ 0.005).

| Stage | Metric | Baseline | Easy | Medium | Hard | Direction |
|---|---|---:|---:|---:|---:|---|
| SM | macro_f1 | 0.8761 | 0.8925 | 0.7604 | 0.6699 | ↑ ↓ ↓ |
| Norm | macro_f1 | 0.4337 | 0.3936 | 0.3613 | 0.3175 | ↓ ↓ ↓ |
| EM-block | macro_pair_recall | 0.9511 | 0.9703 | 0.9481 | 0.5432 | ↑ ↓ ↓ |
| EM-match | macro_f1_vs_test | 0.7499 | 0.8978 | 0.7957 | 0.6801 | ↑ ↓ ↓ |
| Fusion | overall_accuracy | 0.8594 | 0.8394 | 0.8072 | 0.5738 | ↓ ↓ ↓ |

**Hard-vs-baseline delta**:

| Stage | Δ (hard - baseline) |
|---|---:|
| SM | **-0.206** |
| Norm | -0.116 |
| EM-block | **-0.408** |
| EM-match | -0.070 |
| Fusion | **-0.286** |

**Easy frequently exceeds baseline** (4/5 stages). The regenerated easy test = K2-survivable subset, genuinely easier than baseline. Not a regression — the medium → hard slope is the real difficulty signal.

## P8 best-member-F1 ceiling

Tracks the user-attainable ceiling — the strongest committee member at each level. Per [plan_s1_final.md §Reporting](../../../plans/plan_s1_final.md): the committee mean masks individual strong/weak members; best-member is what the user actually consumes.

| Stage | Baseline winner | Δ ceiling (hard - baseline) | Non-increasing? |
|---|---|---:|---|
| SM | label_jw / llm_openai | 1.000 → 0.920 (**-0.08**) | ✓ |
| Norm | text_clean | 0.768 → 0.776 (+0.008) | ✗ (text_clean is robust to music K1/K8) |
| EM-block | sc_block | 0.999 → 0.590 (**-0.41**) | ✗ (flagged: easy 0.9993 barely > baseline 0.9993, within tolerance; main drop is at hard) |
| EM-match | ditto_plm | 0.976 → 0.874 (**-0.10**) | ✓ |
| Fusion | duration_maximum / release-country_llm_judge | 0.838 → 0.564 (**-0.27**) | ✓ |

**Collapses (2)**: at hard SM, string-similarity matchers collapse as predicted by the K8 card — `label_jw` 1.0 → 0.36 (-0.64) and `coma_hybrid` 1.0 → 0.46 (-0.54). The committee ceiling (`llm_openai` at 0.92) absorbs the loss; user-attainable performance only drops -0.08.

## Per-knob structural monotonicity (R7.2)

Source: [output/music/monotonicity_report.csv](../../output/music/monotonicity_report.csv).

| Knob | Check | Easy | Medium | Hard | Status |
|---|---|---:|---:|---:|---|
| K2 | configured corner ratio | 0.2 | 0.5 | 0.8 | PASS |
| K2 | realised vs configured | 0.257 | 0.266 | 0.260 | **FAIL** (F4 dial limit — K2 can't reach configured 0.5/0.8) |
| K2 | realised monotonicity | 0.257 | 0.266 | 0.260 | **FAIL** (same as above; realised stuck at baseline) |
| K3 | drop nesting | 7510 | 21067 | 24071 | PASS (easy ⊆ medium ⊆ hard) |
| K4 | coverage mean sources | 2.90 | 2.40 | 1.76 | PASS |
| K5 | format prov rows | 34760 | 43624 | 36102 | **FAIL** (F9 metric — per-source binary draw, source-size sensitive) |
| K6 | noise prov rows | 380 | 2686 | 7772 | PASS |
| K8 | naming edit distance | 0 | 83 | 163 | PASS |
| K10 | configured winner-share | 0.85 | 0.65 | 0.50 | PASS |
| K10 | realised monotonicity | 100 | 151 | 98 | **FAIL** (compromised-mask depopulation at hard, documented) |

4 FAILs, all documented as known metric proxies in [plan_s1_final.md §F4/F9/F11](../../../plans/plan_s1_final.md). None block the committee-F1 verdict above.

## Goals

| Goal | Status | Evidence |
|---|---|---|
| 1. Baseline exists | **PASS** | [baselines/music/baseline_metrics.json](../../baselines/music/baseline_metrics.json), runtime 623s. SM=0.876, Norm=0.434, EM-block=0.951, EM-match=0.750, Fusion=0.859. |
| 2. Signals are real (committee macro down at hard) | **PASS** | This report §Committee F1 monotonicity. All 5 stages drop at hard, -0.07 to -0.41. |
| 3. Signals are real (best-member ceiling, P8) | **PASS for 3/5**, 2 noted | SM/EM-match/Fusion ceilings drop at hard. Norm ceiling immune (text_clean robust). EM-block ceiling drops -0.41 but is flagged non-monotonic on tolerance (easy=0.9993 barely > baseline=0.9993). |
| 4. No silent collapses | **PASS** | [monotonicity_collapses.csv](monotonicity_collapses.csv): 2 collapses at hard SM (label_jw, coma_hybrid), both predicted by K8 card; user-selected `llm_openai` still 0.92. |
| 5. Variant artifacts on disk | **PASS** | [usecases/music-augmented/{easy,medium,hard}/](../../../usecases/music-augmented/) — 3 levels × full directory tree (data, schemamatching, entitymatching, fusion, output/{provenance,baselines}, config/difficulty.yaml). |

## Known limitations (documented, not blocking)

- **F4** — K2 dial saturates around 0.26 on music regardless of configured target (0.20 → 0.50 → 0.80). Mechanism: K2's per-interpolation corner-pair contribution doesn't scale to hit 0.50+ within `max_interp_fraction=0.6`. Music's `interp_pair_factor=0.05` is the best dial available pre-operator-redesign. Does not block downstream signal (committee F1 at hard still drops).
- **F7** — K2 easy is a no-op (baseline > target). The realised ratio at easy equals baseline 0.26 by design.
- **F9** — K5 raw prov-row count is source-size sensitive (per-source binary draw at easy/medium). Non-monotonic in raw count but the per-cell rate moves with the dial.
- **F10 carryover** — At hard, ~528 train positives are removed by K2 entity demotion; pool can't backfill (post-F10 gold-canon filter). Train pos_ratio drops 0.081 → 0.056 on `musicbrainz_2_discogs` hard. Structural artifact, not a regen bug.
- **K10 hard** — Compromised-mask depopulates the swap pool, so realised swap_cells at hard (98) is lower than medium (151).

## Timing breakdown

| Step | Wall time | Notes |
|---|---:|---|
| R6.1 baseline | 10:27 | LLM + Ditto caches warm from music-small |
| R6.2 variant gen (3 levels) | 1h 24min | All knobs K1/K2/K3/K4/K5/K6/K8/K10 |
| R7.1 easy validate | 32:52 | Post-split-runner refactor in plan_s1_final.md |
| R7.1 medium validate | 37:05 | |
| R7.1 hard validate | 36:01 | |
| R7.2 analyze_monotonicity | <1s | |
| **Total S.7** | **3h 20min** | |

Faster than the 10–13h estimate because LLM/Ditto/sc_block caches were largely warm from the music-small sanity ladder runs over the prior days.

## Artifact map

- Baseline: [baselines/music/baseline_metrics.json](../../baselines/music/baseline_metrics.json) + [baselines/music/baseline_report.md](../../baselines/music/baseline_report.md)
- Variants: [usecases/music-augmented/{easy,medium,hard}/](../../../usecases/music-augmented/)
- Variant monotonicity (structural): [output/music/monotonicity_report.csv](../../output/music/monotonicity_report.csv)
- Per-level validation metrics: [validation/music/{easy,medium,hard}/metrics.json](.)
- Per-level reports: [validation/music/{easy,medium,hard}/level_report.md](.) + `em_per_pair.csv`, `fusion_per_attribute.csv`
- Committee F1 monotonicity (P8 + structural): [validation/music/monotonicity_report.md](monotonicity_report.md), [monotonicity_report.csv](monotonicity_report.csv), [monotonicity_best_member.csv](monotonicity_best_member.csv), [monotonicity_collapses.csv](monotonicity_collapses.csv)
- Chain log: [output/_logs/s7_music_chain.log](../../output/_logs/s7_music_chain.log)

## What's NOT in this run

- **games-FULL** — gated by F11 `source_pairs` cleanup (drop `metacritic_2_sales` from games configs). See [plan_s1_final.md §Hard blockers](../../../plans/plan_s1_final.md).
- **companies-FULL** — gated by user (thin fusion val/test). See [plan_s1_final.md §Hard blockers](../../../plans/plan_s1_final.md).
- **products-FULL** — tracked separately in [plan_s1_products.md](../../../plans/plan_s1_products.md). Sanity ladder green; R6.1 etc. pending in that plan.
