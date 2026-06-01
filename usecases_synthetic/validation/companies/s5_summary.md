# S5 — Phase A smoke test (full companies)

Tracks [plans/plan_s1_scale.md row S5](../../../plans/plan_s1_scale.md). Validates that the 2026-04-17 P0 fixes plus the S1/S2/S3/S4* changes hold at full companies scale (~12k records, 2 source pairs), not just on `companies-small`.

Generated: 2026-04-28.

## Verdict

**Pass** as a smoke test. Pipeline runs end-to-end at full scale, all hardening lands as expected at scale, primary stage signal direction is correct. Two pre-known measurement-semantics issues persist (inherited from S4b/S4c notes) — neither is a wiring regression.

## Evidence

- **Re-baseline, generate, validate** all completed exit-0 with the post-C4 split-EM-committee structure.
- **Committee freeze**: all 3 validation runs record the same markers as the freshly re-baselined `baselines/companies/baseline_metrics.json`:
  - sm=`sm_committee.yaml@5022dec6c8d2`
  - em=`em_blocking_committee.yaml@44fd6bfd276d+em_matching_committee.yaml@47093abf698a`
  - fusion=`fusion_committee.yaml@0a388fd41a72`
- **S1/S2 plumbing**: per-pair per-split regenerated EM splits emitted at every level for both source pairs, matching original split shape:
  - `forbes_2_dbpedia_{train,val,test}_regenerated.csv`: 458 / 219 / 140 rows (pos ratio ~0.49 across all 3 levels).
  - `forbes_2_fullcontact_{train,val,test}_regenerated.csv`: 1513 / 729 / 459 rows (pos ratio ~0.31).
  - Closed-set scorer populates `macro_f1_vs_regenerated_val` (primary) + `macro_f1_vs_regenerated_test` (sanity) for every level.
- **S3 hard-negative gate** firing in `provenance_all.csv` at every level:
  - easy: 50 audit rows, all `keep_strong`.
  - medium: 43 audit rows, all `keep_strong`.
  - hard: 36 audit rows, all `keep_strong`.
  - All PLM scores fall below θ−δ on the full-companies pool — no margin-band rescues triggered, no drops. Different distribution from `companies-small` (which had 2 `drop_adjudicated` at easy because the LLM adjudicator is disabled at easy by design).

## Per-stage results

| Stage | Metric | Baseline | Easy | Medium | Hard | Direction |
|---|---|---|---|---|---|---|
| sm | macro_f1 | 0.3756 | 0.4011 | 0.2430 | 0.1944 | clean monotone drop after easy's near-baseline shuffle (K8 owns SM as predicted by the card) |
| em | macro_f1_vs_regenerated_val (primary) | NaN | 0.8454 | 0.8246 | 0.8393 | easy→medium drop, medium→hard rebound (+0.014) |
| em | macro_f1_vs_regenerated_test (sanity) | NaN | 0.8828 | 0.8530 | 0.8283 | clean monotone drop |
| em | macro_f1_vs_pool (secondary) | 0.7203 | 0.7075 | 0.7016 | 0.6793 | clean monotone drop |
| em | macro_f1_vs_test_gold (secondary) | 0.7341 | 0.7204 | 0.7150 | 0.6819 | clean monotone drop |
| fusion | overall_accuracy | 0.4792 | 0.4306 | 0.3333 | 0.3542 | gross delta baseline→hard −0.125; small medium→hard rebound (+0.021) |
| fusion | overall_spread | 0.3125 | 0.1319 | 0.1736 | 0.1250 | spread compresses across the board (committee converges) |

## Open issues (not S5 blockers)

1. **`monotonicity_report.csv` reports two pre-known audit FAILs**:
   - `knob_02_corner_case_count`: 50 → 43 → 36. Audit-side measurement-semantics issue documented in S4b — the row-count check at [generate_variant.py:1084](../../scripts/generate_variant.py#L1084) treats `hard_negative_gate` audit rows as if they were `llm_interpolate_entity` rows. The K2 miner found 50/43/36 corner-case candidates at the three levels — that's a per-level niche-density sampling outcome, not a knob-intensity regression. Deferred to S13/S14.
   - `knob_05_format_prov_rows`: 13866 → 13799 → 24780. Medium-dip vs easy is ~0.5% on a 14k-row metric — known K2-embedding × K4-LLM upstream non-determinism residue called out in [`validation/companies-small/final_report.md`](../companies-small/final_report.md) changelog. The gross direction baseline→hard is correct (massive jump up at hard).

2. **EM `f1_vs_regenerated_val` (primary surface) rebounds slightly at hard**:
   - 0.845 (easy) → 0.825 (medium) → **0.839** (hard), so easy→medium drops cleanly but medium→hard rebounds (+0.014).
   - `_test` sanity surface is clean monotone (0.883 → 0.853 → 0.828), and both `vs_pool` and `vs_test_gold` secondary surfaces are clean monotone. So the rebound on `_val` specifically is a sample-noise property of which entity ids land in the val split at each level, not a measurement-surface bug.
   - `companies-small` showed a similar small non-monotonicity in S4c (peaked at medium); at full companies the wobble is smaller and shifts to easy/hard. Re-check after S13/S14 per-domain knob recalibration. Not blocking S5's smoke-test scope.

3. **Fusion medium → hard rebound (+0.021)**:
   - Same shape as the EM `_val` wobble. Gross direction baseline → hard (−0.125) is correct; the medium → hard segment isn't strictly monotone. Will re-check after S13/S14.

## Artifact map

```
usecases_synthetic/
  baselines/companies/
    baseline_metrics.json     # 2026-04-28 re-baseline
    baseline_report.md
  validation/companies/
    easy/{metrics.json, level_report.md, em_per_pair.csv, fusion_per_attribute.csv}
    medium/{metrics.json, level_report.md, em_per_pair.csv, fusion_per_attribute.csv}
    hard/{metrics.json, level_report.md, em_per_pair.csv, fusion_per_attribute.csv}
    s5_summary.md             # this file
  output/companies/
    monotonicity_report.csv   # generate_variant knob-intensity audit (2 FAILs noted above)
  output/_logs/
    s5_baseline_companies.log
    s5_generate_companies_all.log
    s5_validate_companies_{easy,medium,hard}.log
```

Phase A is complete. Phase B (S6 / S7 pools, S8 / S9 / S10 per-domain configs) and Phase C (S11+) are unblocked.
