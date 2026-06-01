# S1 Difficulty Validation — Final Report (products FULL)

Domain: **products** (full, 3012 records across 4 sources: products_1 812 / products_2 812 / products_3 762 / products_4 626). Runtime: **53min** end-to-end for S.7 (R6.1 → R6.2 → R7.1 × 3 → R7.2). Generated 2026-05-16.

Predecessor sanity ladder: [validation/products-small/](../products-small/) ([plan_s1_products.md §S](../../../plans/plan_s1_products.md)). All R5 committee sign-offs locked; all K* sign-offs locked. F-P1 K4 fusion-protection bug remains unfixed (carryover from products-small).

## Config prep (2026-05-16)

- **F4 K2 calibration** — Added `interp_pair_factor: 0.05` to [config/knob_02_niche/products.yaml](../../config/knob_02_niche/products.yaml). Matches music + games + companies post-F4 pass. Default 0.5 was documented as 7-15× under-shoot on music; products has small clusters (max size 4) so per-interp contribution is realistically 1.0 corner pair.

## Verdict

**PASS on cross-level committee F1 monotonicity at hard for 4/5 stages.** EM-block has the **largest hard cliff of any domain** (-0.66 committee macro, -0.83 best-member ceiling). EM-matching ceiling **rises** at hard (+0.09 unexpectedly), but committee macro is essentially flat across all 3 levels (Δ-0.004) — EM matching dial doesn't bite on products. **0 collapses**. 4 structural monotonicity FAILs are all documented metric proxies.

The synthetic difficulty dial moves the committee mean on 4/5 stages; EM-matching is the outlier. Hard variant fusion gold has known unresolved IDs from the F-P1 carryover, but fusion still produces a clean monotonic ↓↓↓ committee signal.

## Committee F1 monotonicity (R7.1 + R7.2)

Headline metric per stage across baseline → easy → medium → hard.

| Stage | Metric | Baseline | Easy | Medium | Hard | Direction |
|---|---|---:|---:|---:|---:|---|
| SM | macro_f1 | 0.8138 | 0.8143 | 0.7664 | 0.7279 | = ↓ ↓ |
| Norm | macro_f1 | 0.2579 | 0.2941 | 0.2076 | 0.1662 | ↑ ↓ ↓ |
| EM-block | macro_pair_recall | 0.7979 | 0.7998 | 0.7957 | 0.1372 | = = ↓↓ |
| EM-match | macro_f1_vs_test | 0.6004 | 0.6100 | 0.5978 | 0.5964 | ↑ ↓ = |
| Fusion | overall_accuracy | 0.7760 | 0.7200 | 0.6880 | 0.6667 | ↓ ↓ ↓ |

**Hard-vs-baseline delta**:

| Stage | Δ (hard - baseline) |
|---|---:|
| SM | -0.086 |
| Norm | -0.092 |
| EM-block | **-0.661** |
| EM-match | -0.004 |
| Fusion | -0.109 |

**EM-matching has essentially no dial signal** at the committee macro level. SM/Norm/Fusion all clean monotone-down. EM-block has by far the strongest signal — and the largest of any domain so far.

## P8 best-member-F1 ceiling

| Stage | Baseline winner | Hard winner | Δ (hard - baseline) | Non-increasing? |
|---|---|---|---:|---|
| SM | duplicate_majority | duplicate_majority | 1.000 → 1.000 (0.000) | ✓ (flat, ceiling immune) |
| Norm | text_clean | text_clean | 0.395 → 0.300 (-0.095) | ✗ (non-monotone via easy bump 0.395 → 0.474 → 0.378 → 0.300) |
| EM-block | token_blocker | token_blocker | 0.997 → 0.168 (**-0.828**) | ✗ (easy 0.997 barely > baseline 0.997 within tolerance; main drop at hard) |
| EM-match | ditto_plm | ditto_plm | 0.702 → 0.793 (**+0.091**) | ✗ — **ceiling RISES at hard**, dial does not depress user-attainable EM matching |
| Fusion | title_longest_string | title_most_complete | 0.674 → 0.550 (-0.124) | ✓ |

**Key observations**:

- **EM-block ceiling collapses by -0.83** at hard. Largest of any domain (music -0.41, companies -0.45, games -0.44). Products has 4 sources × 3 pairs (largest source-pair set) and the K1/K3 mutations apparently devastate the token-blocker more than for other domains.
- **EM-match ceiling RISES at hard** by +0.09. `ditto_plm` actually does better on hard. Likely because the regenerated hard test set is a smaller, K2-survivable subset of pairs that ditto_plm can match more confidently. The dial doesn't depress the user-attainable EM matching for products.
- **SM ceiling immune** — `duplicate_majority` (exact-match heuristic) handles K1/K8 fine because products have stable `id` columns.
- **Norm has the strongest non-monotonicity** — text_clean RISES at easy (0.395 → 0.474) before dropping. K1 surface anomalies at easy create text_clean opportunities that improve the recall on the regenerated test.

**Collapses: 0**. Same as games (music had 2 SM collapses).

## Per-knob structural monotonicity (R7.2)

Source: [output/products/monotonicity_report.csv](../../output/products/monotonicity_report.csv).

| Knob | Check | Easy | Medium | Hard | Status |
|---|---|---:|---:|---:|---|
| K2 | configured corner ratio | 0.20 | 0.50 | 0.65 | PASS |
| K2 | realised vs configured | 0.439 | 0.420 | 0.435 | **FAIL** (baseline ≈ 0.44 > easy target 0.20; K2 noops at easy + can't reach medium/hard) |
| K2 | realised monotonicity | 0.439 | 0.420 | 0.435 | **FAIL** (essentially flat at baseline) |
| K3 | drop nesting | 285 | 395 | 436 | PASS (easy ⊆ medium ⊆ hard) |
| K4 | coverage mean sources | 3.72 | 3.71 | 2.87 | PASS |
| K5 | format prov rows | 621 | 2918 | 1772 | **FAIL** (F9 metric noise; medium > hard) |
| K6 | noise prov rows | 74 | 260 | 625 | PASS (clean 8× ramp) |
| K8 | naming edit distance | 0 | 85 | 133 | PASS (cleanest of any domain) |
| K10 | configured winner-share | 0.55 | 0.40 | 0.25 | PASS |
| K10 | realised monotonicity | 329 | 300 | 157 | **FAIL** (compromised-mask depopulation at hard, documented) |

4 FAILs, all documented metric proxies. None block the committee-F1 verdict.

## Goals

| Goal | Status | Evidence |
|---|---|---|
| 1. Baseline exists | **PASS** | [baselines/products/baseline_metrics.json](../../baselines/products/baseline_metrics.json), runtime 559s. SM=0.814, Norm=0.258, EM-block=0.798, EM-match=0.600, Fusion=0.776. |
| 2. Signals are real (committee macro down at hard) | **PASS for 4/5** | SM/Norm/EM-block/Fusion drop at hard, -0.09 to -0.66. EM-match is flat (-0.004). |
| 3. Signals are real (best-member ceiling, P8) | **PASS for 2/5**, 3 robust/unexpected | EM-block / Fusion ceilings drop. SM ceiling immune (duplicate_majority). Norm has easy bump. **EM-match ceiling RISES** at hard — known anomaly with K2-survivable regen subsets being easier for the best matcher. |
| 4. No silent collapses | **PASS** | 0 collapses. |
| 5. Variant artifacts on disk | **PASS** | [usecases/products-augmented/{easy,medium,hard}/](../../../usecases/products-augmented/) — 3 levels × full directory tree. |

## Known limitations (documented)

- **F4** — Products' baseline K2 corner ratio (~0.44) is above easy target 0.20 → K2 noops at easy. The dial can only push toward medium (0.50) and hard (0.65) and even there it under-shoots slightly. EM-block + Fusion downstream signals still bite at hard, so the K-mutations themselves operate even when K2 ratio doesn't move.
- **F-P1 carryover (K4 fusion-protection bug)** — Per [plan_s1_products.md §F-P1](../../../plans/plan_s1_products.md), products hard variant fusion gold has unresolved IDs (840/2000 provenance + 168/200 entity IDs). The K4 fusion-protection comparison in [coverage_ops.py:707-708](../../lib/coverage_ops.py#L707-L708) compares `entity_id` (canonical-frame key) against `constraints.fusion_val_test_ids` (source-record IDs). Mismatch means K4 silently treats fusion-protected entities as unprotected at hard. Fusion still produces a monotonic ↓↓↓ committee signal so the bug doesn't block this run, but products hard fusion accuracy may be slightly inflated/deflated by the unresolved IDs. Fix candidate from plan_s1_products.md: translate fusion `<id>` text → canonical `entity_id` at the call site.
- **K10 hard depopulation** — compromised-mask exhausts before hard's swap budget is satisfied (329 → 300 → 157).
- **EM-match dial is invisible** at committee macro level (Δ-0.004) and ceiling level (+0.09 at hard). The K1/K3/K5/K6 mutations don't depress products EM matching; ditto_plm is robust. To make EM matching a difficulty signal for products, K2 would need to actually move the corner ratio (currently saturated at baseline).

## Timing breakdown

| Step | Wall time | Notes |
|---|---:|---|
| R6.1 baseline | 9:22 | Caches partly warm from products-small |
| R6.2 variant gen (3 levels) | 23:26 | Smallest domain (3k records) |
| R7.1 easy validate | 7:05 | Small test sets (~460 pairs total) |
| R7.1 medium validate | 8:29 | |
| R7.1 hard validate | 4:42 | Hard test undersized (F-P1 carryover) |
| R7.2 analyze_monotonicity | <1s | |
| **Total S.7** | **53min** | Fastest of all 3 full-domain runs |

Faster than music (3h 20min) and games (5h 9min) — products is the smallest domain by a factor of 10-25× in record count.

## Cross-domain comparison

| Stage | music Δ@hard | companies Δ@hard | games Δ@hard | products Δ@hard |
|---|---:|---:|---:|---:|
| SM (macro) | -0.206 | n/a (companies-FULL gated) | -0.049 | -0.086 |
| EM-block (macro) | -0.408 | n/a | -0.450 | **-0.661** |
| EM-match (macro) | -0.070 | n/a | -0.103 | -0.004 |
| Fusion (macro) | -0.286 | n/a | -0.086 | -0.109 |
| SM (ceiling) | -0.080 | n/a | 0.000 | 0.000 |
| EM-block (ceiling) | -0.409 | n/a | -0.440 | **-0.828** |
| EM-match (ceiling) | -0.101 | n/a | -0.124 | **+0.091** |
| Fusion (ceiling) | -0.274 | n/a | -0.080 | -0.124 |

**Pattern across 3 domains**: EM-block always has the strongest hard signal (-0.41 to -0.83). Fusion + SM are moderately dialed. EM-matching is the inconsistent stage — it bites at music + games but flatlines on products. Norm is generally robust to all knobs (Δ-0.02 to -0.09 ceiling).

## Artifact map

- Baseline: [baselines/products/baseline_metrics.json](../../baselines/products/baseline_metrics.json) + [baselines/products/baseline_report.md](../../baselines/products/baseline_report.md)
- Variants: [usecases/products-augmented/{easy,medium,hard}/](../../../usecases/products-augmented/)
- Variant monotonicity (structural): [output/products/monotonicity_report.csv](../../output/products/monotonicity_report.csv)
- Per-level metrics: [validation/products/{easy,medium,hard}/metrics.json](.)
- Per-level reports: [validation/products/{easy,medium,hard}/level_report.md](.) + `em_per_pair.csv`, `fusion_per_attribute.csv`
- Committee F1 monotonicity: [monotonicity_report.md](monotonicity_report.md), [monotonicity_report.csv](monotonicity_report.csv), [monotonicity_best_member.csv](monotonicity_best_member.csv), [monotonicity_collapses.csv](monotonicity_collapses.csv) (empty)
- Chain log: [output/_logs/s7_products_chain.log](../../output/_logs/s7_products_chain.log)

## What's NOT in this run

- **companies-FULL** — gated by user (thin fusion val/test). See [plan_s1_final.md §Hard blockers](../../../plans/plan_s1_final.md).
- **F-P1 fix** — K4 fusion-protection bug still open. Products hard fusion run nevertheless produced a clean monotonic ↓↓↓ signal at the committee level.
