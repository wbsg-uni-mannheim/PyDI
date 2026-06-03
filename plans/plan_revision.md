# plan_revision.md

Pending revisions to apply before the **next** variant-generation run.
Each item lists the concrete files / lines affected and the rationale.

**Reading order:** R-1 (variant-quality improvement) is the load-bearing
item — its outcomes change the contract that R0 (companies-FULL) and
R1 (products schema upgrade) re-run against. Land R-1 fixes first, then
the cascading reruns inherit them automatically. R2-R5 are smaller
cleanup items that can land in any order.

---

## R-1 — Variant-quality improvement on music + games (load-bearing)

**Status:** Step 1 (instrument) landed 2026-05-19. Step 2 (diagnose)
landed 2026-05-19. **Step 3 (calibrate) + step 4 (code-fix) landed
2026-05-22** — user signed off on the four C1/C2/C5/C4 decisions
recorded in [plan_revision_step2_findings.md](plan_revision_step2_findings.md);
code + config changes are in place. Headline correction from step
2: G1's "12 LLM calls, all guardrail-dropped" framing was wrong; the
real cause is K2 hard's default `strict_cache=True` suppressing all
LLM calls. Music K2 standalone replay reports `strict_cache_miss=1080`
(the K2 hard budget) — every attempt missed the cache, so the
original full-music hard run made 0 LLM calls. See updated G1, R4,
C1, C2, C4, C5 below for the final decisions and what landed.

**Additional 2026-05-22 design decisions** (post step-4 fixes,
infrastructure work pending): **C9** (fusion silver standard as
variant-protection target, replacing "one value close to gold");
**C10** (drop open-set EM scoring; committee surfaces two closed-set
EM F1 numbers per pair; only `f1_regen_test` drives monotonicity);
**C11** (EM regen emits two parallel split versions —
`_baseline_pruned` + `_corner_filled` — supersedes C8); **C12**
(restructure normalization + fusion committees so each member is a
coherent end-to-end approach; LLM members get prompt v2 with
LLM-chosen operation + synthesis allowed; TD members get val-best
PyDI fallback for semantically-incompetent types; AccuSim uses
type-aware similarity hooks; LTM keeps native multi-truth on lists).
Also: G8 (regen survivor-bias) **CORRECTED 2026-05-22** — survivors
are corner-biased not easy-biased; dilution source is the easy-fill
backfill, not survivors.

**Step 4b (C9 silver standard) landed 2026-05-23**:
[lib/fusion_silver_standard.py](../usecases_synthetic/lib/fusion_silver_standard.py)
+ [lib/fusion_silver_targets.py](../usecases_synthetic/lib/fusion_silver_targets.py)
+ [scripts/build_fusion_silver_standard.py](../usecases_synthetic/scripts/build_fusion_silver_standard.py).
Music / games / companies silvers built (4 280 / 8 974 / 1 088
clusters); products deferred until R1 (data_cleaned_final schema).
Protection wiring: K1 + K6 dispatchers gain a `protection_source`
flag (default `gold`; `silver` switches to the wider universe with
gold-wins-per-cell merge). 61 new tests pass; user-gate review on
music silver signed off 2026-05-23.

**Step 4c (C11 EM regen split refactor) landed 2026-05-25**:
[lib/corner_case_miner.py:regenerate_em_splits](../usecases_synthetic/lib/corner_case_miner.py)
rewritten to emit two parallel versions per (pair, split) — Set 1
`baseline_pruned` (survivors only) and Set 2 `corner_filled`
(survivors + 100% corner-mined backfill, no easy fills). Output
rows carry a new `version` column; writer emits
`<pair>_<split>_<version>.csv`. [variant_loader.py](../usecases_synthetic/lib/variant_loader.py)
+ `VariantBundle.em_gold_regenerated` upgraded to the 3-level
nested shape `{pair: {split: {version: DataFrame}}}`.
[committee_em.py:_load_labelled_split](../usecases_synthetic/lib/committee_em.py)
accepts `version: str = "corner_filled"` (default preserves
pre-C11 headline F1 semantics; 4d will read both versions
explicitly). Shortfall accepted per C11 option (i) when the corner
pool exhausts. 9 new tests + 4 existing regen tests updated; 231
tests pass across knob_02 / committee_em / committee_fusion /
silver / silver_targets / protection / generate_variant.

**Step 4d (C10 EM committee scoring rewrite) landed 2026-05-25**:
[lib/committee_em.py:_score_predictions](../usecases_synthetic/lib/committee_em.py)
+ `EMMatchingCommitteeRunner.run` rewritten to surface two closed-set
EM F1 numbers per pair — `f1_baseline_test` (closed-set on
`em_test_baseline_pruned.csv`, Set 1) and `f1_regen_test` (closed-set
on `em_test_corner_filled.csv`, Set 2). Open-set `f1_vs_test_gold`
retired (scorer deleted from
[committee_em_scoring.py](../usecases_synthetic/lib/committee_em_scoring.py)).
Headline `f1` fallback chain: `regen_test → baseline_test → pool`.
`regen_val` kept internal as val/test agreement sanity (logged on
divergence > 0.15, not surfaced in per-pair metrics). Aggregated
macro keys renamed to `macro_f1_baseline_test` + `macro_f1_regen_test`.
[analyze_monotonicity.py](../usecases_synthetic/scripts/analyze_monotonicity.py)
+ [knob_expected_signals.yaml](../usecases_synthetic/config/knob_expected_signals.yaml)
switched from `aggregated.macro_f1_vs_pool` to
`aggregated.macro_f1_regen_test`;
[monotonicity.py:_BEST_MEMBER_METRIC](../usecases_synthetic/lib/monotonicity.py)
points at `f1_regen_test` for em_matching + em.
[validate_variant.py](../usecases_synthetic/scripts/validate_variant.py)
+ [build_statistics.py](../usecases_synthetic/scripts/build_statistics.py)
updated to the renamed keys. Sample-checked baseline EM test sets
on music / games / companies / products — all carry labeled
negatives (no positive-only domains, no recall-style degeneracy).
4 new tests + the legacy `TestScoreEMCorrespondences` class
deleted; 47 committee_em tests + 276 across the C10-touched suite
all pass.

**Step 4e (C12 LLM prompt v2) partially landed 2026-05-25**:
[lib/llm_judge_fusion.py](../usecases_synthetic/lib/llm_judge_fusion.py)
and [lib/llm_normalizer.py](../usecases_synthetic/lib/llm_normalizer.py)
rewritten to prompt v2. Fusion judge emits
`{"value", "operation", "confidence", "reasoning"}` over operation set
`{verbatim_pick, aggregation, union, intersection, normalization,
interpolation}` — synthesis allowed; v1 verbatim constraint removed.
Normalizer emits the same shape over
`{vocab_canonicalize, date_normalize, numeric_normalize,
categorical_map, synthesize, abstain}` — synthesis allowed. Default
`prompt_version` bumped to `"v2"` (invalidates existing caches at
`cache/llm_judge_fusion/` + `cache/llm_normalizer/` by construction).
Both modules gain `op_log_path` parameter — when set by the runner per
(member, domain, level), every non-trivial call appends a CSV row with
the operation tag, chosen value, confidence, reasoning, and cache_hit.
Tests: 14 v2-migrated `TestLLMJudge` cases (test_fusion_td_adapters.py)
+ 22 new `LLMCanonicalizer` cases (new test_llm_normalizer.py) — 36
new/updated tests pass; 94 tests across the committee + adapter suite
remain green (backward-compatible signatures).

**Step 4f K1 realised-intensity audit landed 2026-05-26**:
[apply_knob_01_surface.py](../usecases_synthetic/scripts/apply_knob_01_surface.py)
extended to emit `output/baselines/knob_01_realised.csv` (columns
`level / paraphrase_attempts / paraphrase_committed /
mean_edit_distance / mean_token_jaccard_drop / strict_cache_miss_count`);
new helpers `_token_jaccard_drop` + `build_realised_df`; `apply_knob_01`
returns a 4-tuple; `write_outputs` gains `realised_df` kwarg.
[generate_variant.py](../usecases_synthetic/scripts/generate_variant.py)
gains `_k1_realised_metrics` reader + two new audit rows
(`knob_01_realised_rate_monotonicity` on `paraphrase_committed`,
`knob_01_realised_intensity_monotonicity` on
`mean_edit_distance` AND `mean_token_jaccard_drop`). Missing-file
fallback adds explicit FAIL rows. `BASELINE_FILES` in
[package_variant.py](../usecases_synthetic/scripts/package_variant.py)
extended with `knob_01_realised.csv` + `knob_10_realised.csv` (the
latter a latent step-1 omission that silently disabled the K10
rate-monotonicity audit on real variants). 22 new tests
(`TestRealisedAudit` in test_knob_01 + `TestK1RealisedMetricsReader`
/ `TestK1AuditRowsInCheckMonotonicity` in test_generate_variant);
full 1299-test synthetic suite green. Next: re-run R7.2 on
existing music + games variants once the LLM-spend authorisation
lands.

**Step 4e committee restructure landed 2026-05-26**: (i)
[committee_fusion_c12.py](../usecases_synthetic/lib/committee_fusion_c12.py)
runner with the 9-member coherent roster + val-selection plumbing +
per-domain `baselines/<domain>/fusion_committee_selection.json` cache
(landed earlier; verified 2026-05-26); (ii)
[committee_norm_c12.py](../usecases_synthetic/lib/committee_norm_c12.py)
runner with the 3-member coherent roster
(`rule_per_attribute_optimal` / `llm_only` / `passthrough`) +
val-selection plumbing + `baselines/<domain>/norm_committee_selection.json`
cache; [committee_norm.py](../usecases_synthetic/lib/committee_norm.py)
gains a `__new__` dispatcher mirroring fusion; (iii) all four
`fusion_committee*.yaml` (music / games / products / companies) and
all four `normalization_committee_<domain>.yaml` migrated to C12
schema; (iv)
[build_statistics.py](../usecases_synthetic/scripts/build_statistics.py)
gains a new `selection_map` XLSX sheet that reads
`MemberResult.notes["selection_map"]` for the optimized members;
[analyze_monotonicity.py](../usecases_synthetic/scripts/analyze_monotonicity.py)
already reads the C12 keys (`aggregated.macro_f1` for norm,
`aggregated.overall_accuracy` for fusion) without change. 21 new
tests in [test_committee_norm_c12.py](../usecases_synthetic/tests/test_committee_norm_c12.py)
pass; legacy-shape dispatch test added to test_committee_fusion_c12.py;
TestNormCommitteeConfig + TestFusionCommitteeConfig in
test_committee_configs.py updated to be C12-shape-aware. Full 359-test
C12-touched suite green.

**Step 4h (pre-rerun knob tuning review) landed 2026-05-27**: all 8
knobs walked end-to-end with the user; decisions captured in
[plan_revision_step4h_knob_review.md](plan_revision_step4h_knob_review.md).
Headline outcomes: K1 Option B rates + LLM-at-medium + bumped hard;
K2 `placement_split: 0.6 → 0.5`, prompt v1 → v2, `hard_negative_gate`
option (a) full-LLM via new `gate_mode: full_llm` code field; K3
products gains per-source-attribute caps for the R1 + extension sparse
columns; K4 explicit medium histogram (option β conservative
singletons) + ramped `within_source_duplicate_rate`; K5 products gains
new `rate` class for `read_speed_mb_s` + `write_speed_mb_s`; K6
products cross-knob expansion (3 new taxonomy CSVs + 7 categorical /
numeric extension columns into scope across K3/K6/K8/K10);
K1+K6-cleanup overlap corrected (the 2 K1 normalize-down additions
proposed during K1 walk were DROPPED in favour of K6's existing
surgical regex `cleanup_rules`); K8 products rename_table extended +
refactored to YAML anchors; K10 products `attribute_targets` extended
+ companies dbpedia-winner override commented.

**Step 4i (drop-corner-touching operator + non-corner LLM refill)
landed 2026-05-28**: replaces the legacy `noop_baseline_above_target`
K2 branch with a real bidirectional dial. New
[lib/non_corner_refill.py](../usecases_synthetic/lib/non_corner_refill.py)
module + `_run_drop_corner_refill` helper in
[apply_knob_02_niche.py](../usecases_synthetic/scripts/apply_knob_02_niche.py)
+ new
[config/knob_02_niche/_prompts/non_corner_v1.txt](../usecases_synthetic/config/knob_02_niche/_prompts/non_corner_v1.txt)
+ new cache namespace at `cache/knob_02_non_corner/<domain>/`. K2's 3
dispatch branches now: `interpolate_paired_drop` (baseline < target),
`drop_corner_touching_refilled` (baseline > target — NEW), or `noop`
(within tol). 28 new tests in
[tests/test_non_corner_refill.py](../usecases_synthetic/tests/test_non_corner_refill.py)
pass; full 1394-test synthetic suite green.

**Step 4j (pre-rerun cleanup) landed 2026-05-28**: mandatory list +
optional small-augmented swept per user 2026-05-28 sign-off. Deleted:
3 full augmented variant trees (`usecases/{music,games,products}-augmented/`,
~314 MB), 4 small-augmented variant trees
(`usecases/{music,games,products,companies}-small-augmented/`, ~179 MB),
4 stale HPO trees (`cache/{em_blocking,em_matching,fusion,norm}_tuning/`,
~4.3 MB). Total ~497 MB freed. `cache/sm_tuning/` +
`cache/ditto_checkpoints/` retained per the KEEP list. No
`usecases/companies-augmented/` existed (R0 still gated). LLM caches
remain empty on disk (no warming needed — first variant generation
populates).

**Remaining work**: step 5 (products rerun — smoke-test / smallest
domain), then step 6 (music + games full reruns), then companies
(gated on R0's larger fusion gold). R7-baseline-finish + top-level
pool-builder retrain + pool rebuild + per-domain R7c are scheduled
per-domain into steps 5/6 with the per-domain gating semantics
clarified under R7 below.

### Step 1 (instrument) — landed 2026-05-19

- **C1** — per-guardrail rejection counters in
  [entity_interpolation.py:213](../usecases_synthetic/lib/entity_interpolation.py#L213)
  +
  [apply_knob_02_niche.py:1869](../usecases_synthetic/scripts/apply_knob_02_niche.py#L1869);
  counts surface as `rejected_*` columns on `knob_02_realised.csv`.
- **C6** — `compute_ceiling_responsiveness` in
  [monotonicity.py:848](../usecases_synthetic/lib/monotonicity.py#L848)
  + wired through
  [analyze_monotonicity.py](../usecases_synthetic/scripts/analyze_monotonicity.py);
  `monotonicity_report.csv` gains a `ceiling_responsiveness` column
  (Pearson r between the signal's per-level value and the stage's
  best-member F1 across baseline/easy/medium/hard).
- **C3 K5** — `_k5_distinct_format_families` in
  [generate_variant.py:1130](../usecases_synthetic/scripts/generate_variant.py#L1130);
  new audit row `knob_05_distinct_format_families` alongside the
  raw-count check.
- **C3 K8** — `_k8_naming_intensity` in
  [generate_variant.py:1112](../usecases_synthetic/scripts/generate_variant.py#L1112)
  (rung_rank descriptive=0 / abbreviated=1 / cryptic=2 / anonymize=3);
  new audit row `knob_08_naming_intensity` alongside edit-distance.
- **C3 K10** — implemented as a *rate-based audit* (rather than the
  mask-reorder the plan originally specified, which was based on a
  pipeline-order misdiagnosis — see G4 correction below).
  [apply_knob_10_reliability.py](../usecases_synthetic/scripts/apply_knob_10_reliability.py)
  now returns a 5-tuple adding `realised_df` and writes
  `output/baselines/knob_10_realised.csv`
  (`level, reshufflable_count, swap_cells, swap_rate,
  compromised_mask_count`).
  [generate_variant.py:1107](../usecases_synthetic/scripts/generate_variant.py#L1107)
  reads it via `_k10_realised_swap_rate` and adds a new audit row
  `knob_10_realised_rate_monotonicity` as the load-bearing K10 verdict;
  the legacy count-based row is retained as a secondary signal.
- **Tests added:** 19 unit tests across `test_knob_02.py` (rejection
  log), `test_monotonicity.py` (`_pearson`, `compute_ceiling_responsiveness`),
  and `test_generate_variant.py` (K5 distinct-families, K8 intensity,
  K10 realised-rate reader). All 132 tests in the K10 / K2 /
  monotonicity suites pass.

The R7.3 final reports
([music](../usecases_synthetic/validation/music/final_report.md),
[games](../usecases_synthetic/validation/games/final_report.md))
recorded **PASS** on the cross-level committee F1 monotonicity criterion
— but the verdict is charitable. Several knobs aren't contributing the
way the K-cards promised, several monotonicity FAILs were waved through
as "documented metric proxies", and the **easy variant is easier than
baseline** on most stages (anti-goal). The full audit is below.

### Current state vs goals

**Goals** (from [plan_s1_final.md §Conventions+Goals](archive/plan_s1_final.md);
goals 2-4 updated 2026-05-22 to reflect C2 + C5):
1. Baseline exists with all 5 committees scored.
2. Committee macro_f1 monotonically decreases **`easy → medium → hard`** (C2 contract;
   baseline is a reference value, not part of the slope verdict, and must
   land no harder than medium).
3. Best-member ceiling (P8) drops monotonically with difficulty across
   `easy → medium → hard` (same slope contract as goal 2).
4. No silent member collapses (drop > 0.5 or absolute F1 < 0.15 at any
   level) **except** K8-anonymized-driven collapses at hard, which C5
   reclassified as intentional ("string-only matchers die by construction
   at K8 anonymize hard; committee macro_f1 holds via LLM/embedding members").
5. Variant artifacts on disk + per-knob realised metrics match configured.
6. (Implicit, never stated explicitly but the whole point): each of the 8
   active knobs (K1/K2/K3/K4/K5/K6/K8/K10) **demonstrably contributes**
   to the difficulty signal — not just nominally enabled.

**What actually landed on music FULL** (from `monotonicity_report.csv`
+ `final_report.md`):

| Stage | Δ macro_f1 (hard − baseline) | Easy vs baseline | Best-member ceiling Δ |
|---|---:|---|---:|
| SM | -0.206 | **+0.016 (easier)** | -0.08 |
| Norm | -0.116 | -0.040 ✓ | +0.008 (immune) |
| EM-block | -0.408 | **+0.019 (easier)** | -0.41 |
| EM-match | -0.070 | **+0.148 (much easier)** | -0.10 |
| Fusion | -0.286 | -0.020 ✓ | -0.27 |

Per-knob realised: **K2 FAIL** (realised 0.257/0.266/0.260 stuck at
baseline 0.24 — never reaches configured 0.5/0.8), **K5 FAIL** (F9
binary-draw metric, 34760/43624/36102 non-monotone), **K10 FAIL**
(realised cell counts 100/151/98).

**What actually landed on games FULL:**

| Stage | Δ macro_f1 (hard − baseline) | Easy vs baseline | Best-member ceiling Δ |
|---|---:|---|---:|
| SM | -0.049 | **+0.134 (much easier)** | 0.000 (immune) |
| Norm | -0.023 | -0.008 ✓ | -0.032 (small) |
| EM-block | -0.450 | 0.000 (flat) | -0.440 |
| EM-match | -0.103 | **+0.053 (easier)** | -0.124 (winner changes) |
| Fusion | -0.086 | -0.004 ✓ | -0.080 |

Per-knob realised: **K2 FAIL** (realised 0.670/0.660/0.671 — flat at
baseline 0.67; easy can't reach configured 0.20 because baseline already
overshoots — F7 noop), **K8 FAIL** (192/83/137 edit-distance non-monotone),
**K10 FAIL** (2/14/10 realised cells).

### Gap analysis — why goals were missed

**G1. K2 (niche corner-case ratio) — the headline knob is essentially
dormant.**
- Music realised corner ratio: 0.257 → 0.266 → 0.260. Configured was
  0.20 / 0.50 / 0.80. The dial does not move; realised is stuck at
  the natural baseline (0.24).
- Games realised: 0.670 / 0.660 / 0.671. Baseline is already 0.67,
  above easy's 0.20 target. F7 made the dispatcher noop in that case.
- **Cause #1 (medium/hard under-shoot) — REVISED 2026-05-19 by step 2.**
  The interpolation step that was supposed to add corner pairs
  produced 0 entities for both domains. **The original framing — "12
  LLM calls all rejected by `contamination_check`" — was wrong.**
  Music K2 standalone replay
  ([/tmp/k2_diag/music/hard/output/baselines/knob_02_realised.csv](../))
  reports `strict_cache_miss=1080` (the full K2 hard budget) — *every*
  attempt missed the cache, so no LLM call was made and no guardrail
  rejection happened. Root cause:
  [generate_variant.py:792-795](../usecases_synthetic/scripts/generate_variant.py#L792-L795)
  forces `strict_cache_k2 = True` at hard level for non-aliased
  domains, and the existing K2 cache at
  `cache/knob_02_interpolations/music/` (1597 entries, mostly from
  music-small) does not contain the 1080 pair-hashes K2 hard's
  deterministic dispatcher selects on the full-music sources. So the
  dial-dormancy is a *cache-population* problem, not a guardrail
  problem. K4-fab adds entities but those don't count toward "corner"
  pairs.
- **Cause #2 (easy over-shoot):** F7 noop is correct for the constraint
  (K2 has no "drop corner-touching" operator), but it means easy =
  baseline for any domain with high natural corner ratio. The
  difficulty signal at easy depends entirely on K1/K3/K4/K5/K6/K8/K10.

**G2. Easy < baseline for 3-4 of 5 stages on each domain. (Reframed
2026-05-22 by C2 — no longer an anti-goal under the new contract.)**
- Music: SM +0.016, EM-block +0.019, EM-match +0.148. Games: SM +0.134,
  EM-match +0.053.
- **Original framing (pre-C2):** "Easy < baseline is an anti-goal; the
  user should see monotone slopes baseline → easy → medium → hard."
- **C2 reframing (2026-05-22):** the monotonicity contract is
  `easy → medium → hard` only; baseline is a reference value that
  must not be harder than medium but is allowed to land between easy
  and medium (typical) or be easier than easy. Easy < baseline is
  therefore acceptable. The original concern ("readers will ask is
  the pipeline working") is addressed by the new
  [analyze_monotonicity.py](../usecases_synthetic/scripts/analyze_monotonicity.py)
  `baseline_position_ok` column + R7.3 narration explaining the
  contract.
- **Cause** (retained for audit-trail): the regenerated EM test for
  easy draws from the K2-survivable subset (pairs whose both entities
  survive K2 + K3 + K4). At easy, K3 drops < 5% of rows, K4 coverage
  is mild → the surviving test set is heavily weighted toward "easy"
  entities (high attribute coverage, clean values) compared to the
  unrestricted baseline test that includes hard original pairs. Net
  effect: easy test is intrinsically easier than baseline test. Under
  C2 this is no longer a problem; under C11 (`_corner_filled` regen
  surface) the load-bearing per-level F1 is computed against a
  baseline-sized corner-biased pool, which removes the easy-spike at
  easy by construction.

**G3. K5 (format) raw metric is non-monotonic on music — F9 documented
but never fixed.**
- music: 34760 / 43624 / 36102 — hard < medium.
- F9 root cause: K5's per-source format draw at easy/medium has ~50%
  chance of staying at baseline (0 prov rows) vs landing on a variant
  (prov for ALL rows in that source × attribute). When F7 leaves more
  rows in play at easy (K2 noop), K5 easy prov count overtakes K5
  medium. The dial isn't broken; the **measurement** is source-size
  sensitive.
- **Cause:** the monotonicity audit uses a raw row-count proxy. The
  right metric is per-cell intensity (e.g. distinct format families
  touched, or expected prov rate from pool size). F9 noted this and
  punted.

**G4. K10 (reliability) realised non-monotone on both domains.**
- Music: 100 / 151 / 98 (cells with compromised provenance).
- Games: 2 / 14 / 10.
- The "compromised-mask depopulation at hard" hand-wave in the reports
  means: K10's mask references entities that get dropped by K3 at hard,
  so the realised count at hard < medium even though the configured
  winner-share *decreases* monotonically (0.85 → 0.65 → 0.50, PASS).
- **Cause (revised 2026-05-19).** The original diagnosis — "K10 mask is
  applied before K3 drops" — is incorrect. Actual pipeline order is
  **K3 → K10 → K8** (verified at
  [generate_variant.py:924-939](../usecases_synthetic/scripts/generate_variant.py#L924-L939)),
  so K10's mask is already built against post-K3 sources by
  construction. The real cause is *mask depopulation against a shrinking
  pool*: at hard, K10 disperses winner-share more aggressively
  (correct direction) but K3 drops more entities, so the absolute swap
  count can shrink even though K10's dial is monotone. The count-based
  audit is therefore fragile to K3-K10 pool interactions; a *rate*-based
  audit (swap_cells / reshufflable_count) is invariant to K3's drop and
  surfaces K10 monotonicity cleanly. Landed under step 1 (C3 K10) — see
  the new `knob_10_realised_rate_monotonicity` check.

**G5. K8 (naming) edit-distance non-monotonic on games.**
- Games: 192 / 83 / 137. Same direction problem as music's K8 (described
  in F9 context).
- **Cause:** easy descriptive→canonical renames are the largest edits;
  abbreviated/cryptic at medium/hard often share substrings with the
  base name. Edit-distance is the wrong intensity proxy for K8.

**G6. EM-block ceiling collapses ~0.40-0.45 at hard on every domain.**
- Music -0.41, games -0.44, companies -0.45.
- This is a **universal pattern** — every domain's best blocker
  (`sc_block` on music, `token_blocker` on games) drops to ~0.55-0.59
  on hard. The user's blocking recall ceiling falls off a cliff.
- **Cause:** at hard, K1 (paraphrase rate 0.08-0.20) + K8 (column
  rename) + K10 (reliability erosion of key fields) combine to make
  blocking keys unrecognisable across sources. The configured
  blocking-key columns no longer share enough surface form for sparse
  matchers, and the embedding blocker doesn't compensate.

**G7. Music SM has 2 silent member collapses at hard. (Reclassified
2026-05-22 by C5 — intentional, not a defect.)**
- `label_jw` 1.0 → 0.36 (-0.64), `coma_hybrid` 1.0 → 0.46 (-0.54).
- The committee ceiling (`llm_openai` at 0.92) absorbs the loss so the
  user-attainable ceiling only drops -0.08.
- **Cause:** the K8 anonymized column rename (`name` → `Attribute_2`)
  is a label-only signal that string-similarity matchers literally
  cannot use.
- **C5 reclassification (2026-05-22):** hard is the
  "LLM/embedding matchers must carry it; string-only dies by
  construction" level. The collapse is now **expected and intentional**
  per the user's directive. The `detect_collapses` audit still flags
  them (no code change) but R7.3 narration interprets K8-anonymize-
  driven collapses at hard as design, not failure. Goal 4 is relaxed
  for these specific cases.

**G8. Regenerated EM splits — dilution comes from the *easy-fill
backfill*, not from surviving originals.** (Raised 2026-05-19 during
OVERVIEW.md review. **Original "easy-survivor" framing CORRECTED
2026-05-22 after grounding in
[apply_knob_02_niche.py](../usecases_synthetic/scripts/apply_knob_02_niche.py)
+ [niche_scorer.py](../usecases_synthetic/lib/niche_scorer.py)** —
prior framing had the bias direction backwards.)

- **Current regen logic.**
  [lib/corner_case_miner.py:regenerate_em_splits](../usecases_synthetic/lib/corner_case_miner.py)
  carries over every original `(id1, id2, label)` whose ids both still
  exist post-K2 / K4, then backfills to the original
  `(size, positive_ratio)` with new corner positives / negatives sized
  to hit `target_corner_case_ratio` plus **easy fills** covering the
  remaining capacity.
- **Original (wrong) framing.** Previously claimed: "the surviving
  originals are by construction the pairs that aren't corner cases —
  pairs where both entities… weren't restructured into near-twin
  neighborhoods." That description was not grounded in the K2 code
  and inverted the actual bias direction.
- **What K2 actually does at the size-balance step**
  ([apply_knob_02_niche.py:1276-1300](../usecases_synthetic/scripts/apply_knob_02_niche.py#L1276-L1300)):
  for every N interpolated near-twin entities added, drop N existing
  entities to keep per-source row count stable. The N dropped are
  chosen by `rank_entities_by_density` reversed — **lowest density
  first** — skipping protected entities (fusion val/test, EM gold
  protected ids) and entities in label-collision groups. "Density"
  ([niche_scorer.py:115-218](../usecases_synthetic/lib/niche_scorer.py#L115-L218))
  is RRF-fused neighbor agreement across ≥`c_min` metrics plus a
  +5.0 boost for label-collision members. **High density = sits in a
  near-twin neighborhood = corner-case-like. Low density = isolated =
  unambiguous match.**
- **Actual survivor bias.** K2's size-balance drop preferentially
  removes *isolated, low-density (easy)* entities and *protects*
  dense (corner-neighborhood) entities. Original-gold pairs whose
  entities are corner-case in the baseline tend to **survive**; pairs
  whose entities are isolated/easy tend to **drop out**. Set 1
  (`_baseline_pruned` in C11) is therefore biased **toward
  corner-pairs of the original gold**, *not* toward easy pairs. The
  bias *intensifies* with level — at hard, K2 hard runs ~1080 drops
  on music, so the surviving subset is correspondingly more corner-
  heavy.
- **Actual dilution source.** The easy fills the current regen adds
  to reach the original split size after corners-to-hit-target have
  been mined. These are random non-corner pairs from the variant and
  water down the corner ratio of the regen split.
- **C11 implication.** The structural decision (two parallel surfaces)
  stands. Set 2 (`_corner_filled` per C11) fixes the actual dilution
  by spec'ing "filled up to the old size using corner-cases" — no
  easy fills. Both halves of Set 2 (survivors + corner-mined
  backfill) are corner-biased, so the K2-attributable F1 drop should
  surface cleanly. Set 1 is a level-dependent, corner-biased
  subsample of the original gold — **not** a stable cross-level
  reference; its level-by-level numbers drift with K2 intensity. R7.3
  reports must narrate this rather than treat baseline-test as a
  fixed yardstick.
- **Compounds with G1** (K2 dormant): unchanged. If K2's
  interpolation produces 0 entities (the strict_cache=True bug
  diagnosed in step 2), the size-balance drop also produces 0
  entities (`n_balance = len(interpolated) = 0`,
  [apply_knob_02_niche.py:1281](../usecases_synthetic/scripts/apply_knob_02_niche.py#L1281)),
  so the entire variant inherits the baseline survivor composition
  with neither the corner-bias intensification nor the easy-fill
  dilution.

**G9. K1 (surface paraphrase) effect on difficulty is unverified.**
(Raised 2026-05-23.)

- K1 is configured with paraphrase rate 0.08 / 0.14 / 0.20 across
  easy / medium / hard, applied via LLM to entity attributes
  ([apply_knob_01_surface.py](../usecases_synthetic/scripts/apply_knob_01_surface.py)).
- Per-knob realised-intensity audits exist for K2 (corner-case
  ratio), K3 (drop rate), K4 (coverage), K5 (distinct format
  families — post-C3), K8 (rung-weighted naming_intensity —
  post-C3), K10 (realised swap_rate — post-C3). **K1 has no
  analogous realised-intensity audit.** No artifact in
  `output/baselines/` tells us whether the configured paraphrase
  rate translates into a measurable degradation of string overlap
  / token Jaccard / edit distance at the cell level.
- Suspected failure modes:
  - **Cache-dormancy analogous to G1.** K1 also has a strict-cache
    code path
    ([apply_knob_01_surface.py:1124](../usecases_synthetic/scripts/apply_knob_01_surface.py#L1124),
    fixed by C1's strict_cache forcing removal). Without a realised
    audit we don't know whether the K1 cache at
    `cache/knob_01_surface/<domain>/` actually covers the
    pair-hashes K1 hard selects on full-domain sources.
  - **Shallow paraphrase intensity.** Even if K1 fires at the
    configured rate, the LLM may produce paraphrases that don't
    degrade string-similarity matchers meaningfully (casing,
    synonym swaps, trivial reorderings) — much lower effective
    intensity than the rate suggests.
  - **Stage-level invisibility.** G6 attributes the EM-block hard
    ceiling collapse to "K1 + K8 + K10 combined". If K1 is
    shallow, the K8 + K10 components are doing most of that work
    and C4's char-ngram BM25 fix may not close the gap as expected.
- **Verification work** (step 4f in the action plan): add a K1
  realised-intensity audit analogous to C3's K5/K8/K10
  instrumentation. Write `output/baselines/knob_01_realised.csv`
  from
  [apply_knob_01_surface.py](../usecases_synthetic/scripts/apply_knob_01_surface.py)
  carrying `level, paraphrase_attempts, paraphrase_committed,
  mean_edit_distance, mean_token_jaccard_drop,
  strict_cache_miss_count`. Add audit rows
  `knob_01_realised_rate_monotonicity` and
  `knob_01_realised_intensity_monotonicity` to
  [generate_variant.py](../usecases_synthetic/scripts/generate_variant.py)
  + [analyze_monotonicity.py](../usecases_synthetic/scripts/analyze_monotonicity.py).
  Will surface dormancy (rate flat across levels) and shallow-
  paraphrase issues (rate monotone but intensity flat / inverted)
  as distinct verdicts.

**G10. Games EM committee baseline F1 is much worse than the
human-baseline notebook's EM F1 on the same data.** (Raised
2026-05-23.)

- Observation: at baseline (no knobs applied), the games EM
  committee's macro_f1 across the games source pairs is well below
  what the human-baseline Jupyter notebook achieves on the same
  source pairs. The committee should at least match a hand-tuned
  notebook at baseline, since both operate on identical input.
- Possible causes (to be triaged in the diagnosis step):
  - **Member composition.** The committee's EM matchers
    (`ditto_plm`, `magellan`, `llm_matcher`, `comem`) and blockers
    (`sc_block`, `bm25` (now char-ngram per C4), `token_blocker`,
    …) may be suboptimal vs the notebook's hand-tuned per-attribute
    pipeline.
  - **Blocker recall ceiling already low at baseline.** G6 names
    a hard-level EM-block ceiling cliff; the baseline number may
    also be sub-ceiling on games specifically (different schema /
    different blocking-key suitability than music / products).
  - **`text_cols` / blocking-key config mismatch.** The games
    [em_blocking_committee_games.yaml](../usecases_synthetic/config/committees/em_blocking_committee_games.yaml)
    + [em_matching_committee_games.yaml](../usecases_synthetic/config/committees/em_matching_committee_games.yaml)
    may not select the right signal columns (e.g. omits a
    high-signal attribute the notebook uses).
  - **Measurement mismatch.** Open vs closed-set scoring was a
    known issue (C10) but the notebook is internally consistent;
    once C10 lands and `f1_baseline_test` is closed-set on the
    same test gold the notebook uses, the comparison becomes
    apples-to-apples. The gap may shrink or vanish.
  - **Per-attribute optimization in the notebook.** If the notebook
    picks per-attribute methods that the committee's fixed roster
    can't match, the gap is structural. EM matchers don't usually
    optimize per-attribute (they're per-pair), so this is less
    likely than the fusion case but worth ruling out.
- **Diagnosis work** (step 4g in the action plan): locate the
  human-baseline games EM notebook under `usecases_synthetic/`,
  document its method stack (matchers, blockers, scoring), and
  compute committee vs notebook EM F1 on the same baseline games
  test set under closed-set scoring (post-C10, so the comparison
  is apples-to-apples). Attribute the gap to a specific component
  (blocker recall / matcher composition / config tuning / scoring
  semantics). One-time investigation, not a recurring per-level
  audit.
- **Couples with G6** (EM-block ceiling collapse at hard): if the
  baseline gap is dominated by blocker recall, C4's char-ngram
  BM25 may close some of it on the next rerun, and the games
  Step 7 verdict should compare against the notebook's baseline
  rather than against an absolute F1 threshold.

### Brainstorm — necessary changes

> **Decision gate.** Each item below names options + a recommended pick.
> **No code or config changes land without explicit user sign-off on
> each choice.** The recommendations are anchored in the artifact trail,
> but several shift difficulty semantics (e.g. monotonicity slope
> contract, K8 anonymized scope, fusion silver standard, EM closed-set
> scoring, committee restructure, LLM prompt v2 with synthesis) and the
> user owns those trade-offs. Convention: Claude proposes → user
> confirms or redirects → Claude implements. See [plan_s1_final.md
> §Conventions "Committee tuning convention (2026-05-08)"](archive/plan_s1_final.md)
> for the same pattern applied to committee sweeps.
>
> **Sign-off status as of 2026-05-25**: C1, C2, C3, C4, C5, C6, C9, C10,
> C11, C12 are all DECIDED. C3, C4, C6 + the step-1 instrumentation
> have CODE LANDED. C1/C2/C5 have CODE LANDED at the config + slope-
> check level; C1's "real LLM call on miss" follow-up is open. **C9
> CODE LANDED 2026-05-23** (silver standard + protection_source flag).
> **C11 CODE LANDED 2026-05-25** (regenerate_em_splits two-version
> refactor + loader/consumer updates). **C10 CODE LANDED 2026-05-25**
> (closed-set EM committee scoring on both versions; open-set
> retired). C12 remains DECIDED but **implementation is pending**
> (step 4e in the action plan). C7 is the only PENDING decision —
> operational LLM-budget item awaiting the cache pre-population
> authorisation. C8 was SUPERSEDED by C11.

**C1. Fix K2 hard interpolation — DECIDED 2026-05-22: never fail on
cache miss.** The user's directive: on miss the pipeline must make a
call, not raise `LLMCacheMiss`. Implementation:

- [generate_variant.py:792-810](../usecases_synthetic/scripts/generate_variant.py#L792-L810)
  no longer auto-forces `strict_cache_{k1,k2} = True` at hard. Both
  default to `False`; callers can still opt into strict replay via
  `strict_cache_k1=True` / `strict_cache_k2=True` for deliberate
  reproducibility runs.
- [apply_knob_02_niche.py:2229](../usecases_synthetic/scripts/apply_knob_02_niche.py#L2229)
  + [apply_knob_01_surface.py:1124](../usecases_synthetic/scripts/apply_knob_01_surface.py#L1124)
  CLIs: dropped the `or (level == "hard")` forcing. Strict mode is
  now opt-in via `--strict-cache`.
- On miss, the K2 path either calls the supplied `api_client` (real
  LLM if wired) or, when `api_client=None`, falls through to the
  deterministic blender
  (`default_api_client_from_attributes`,
  [apply_knob_02_niche.py:1965-1968](../usecases_synthetic/scripts/apply_knob_02_niche.py#L1965-L1968)).

Cascade work: the orchestrator currently passes `api_client=None`,
which means cache miss → deterministic blender. To get real LLM
calls on miss (the most-faithful path), `generate_variant.py` needs
an OpenAI-backed `LLMInterpolateFn` builder wired in when an API
key is available. **Follow-up work item, not blocking this round**:
add `build_openai_interpolation_client()` and call it from
`_run_knob_02` when `os.environ["OPENAI_API_KEY"]` is set. Until
then, miss → blender, which is itself an upgrade over miss → skip.

The rejection-counter instrumentation (step 1 C1) is now load-bearing
once cache lookups start hitting: it will tell us which guardrail
dominates, at which point `interpolation_count` /
`contamination_check` thresholds become tuning knobs.

**C2. "Easy < baseline" — DECIDED 2026-05-22: baseline is reference-only;
slope is `easy → medium → hard`; baseline must NOT be harder than medium.**
The user's directive in two parts:

1. *Monotonicity slope*: contract is `easy → medium → hard` only.
   Baseline is a reference value shown on charts but does not gate
   the slope verdict.
2. *Baseline position constraint*: baseline must sit between easy
   and medium (typical) or be easier than easy (acceptable). It is
   NOT allowed to land *harder than medium* — that inverts the
   implicit difficulty ordering for any reader inspecting the
   report.

Implementation:
- [monotonicity.py:42-50](../usecases_synthetic/lib/monotonicity.py#L42-L50)
  introduces `SLOPE_LEVELS = ("easy", "medium", "hard")` alongside
  the existing `LEVELS` (which still drives reporting/delta calc).
- [monotonicity.py: match_signals](../usecases_synthetic/lib/monotonicity.py#L424)
  feeds `check_monotone` the slope-level values only; the
  `hard − baseline` delta is unchanged for reporting.
- [monotonicity.py: baseline_within_allowed_position](../usecases_synthetic/lib/monotonicity.py)
  new helper: `True` iff baseline is not harder than medium
  (direction-aware: `>=` for down, `<=` for up, `flat_tolerance` for
  flat).
- `SignalCheck` dataclass gains `baseline_position_ok: bool` field;
  `match_signals` populates it per check.
- [analyze_monotonicity.py](../usecases_synthetic/scripts/analyze_monotonicity.py):
  CSV writer surfaces `baseline_position_ok` column;
  markdown report adds a `BasePos` column to the per-signal table.
- Module docstring on analyze_monotonicity.py updated to describe
  both the slope contract and the position constraint.

13 new tests added (`TestBaselineWithinAllowedPosition` + two
`TestMatchSignals` integration tests for the new field); all 52
monotonicity tests pass. No K1 config changes land (the recommended
(b) option from the original C2 brainstorm is moot under the new
contract).

**C3. Replace raw-count monotonicity proxies with intensity metrics.
LANDED 2026-05-19 under step 1.** (K5 distinct_format_families, K8
rung-weighted naming_intensity, K10 realised swap_rate — all
implemented in [generate_variant.py](../usecases_synthetic/scripts/generate_variant.py)
+ [apply_knob_10_reliability.py](../usecases_synthetic/scripts/apply_knob_10_reliability.py)
with the new audit rows in `monotonicity_report.csv`.)
- K5: count distinct format families touched per level (not prov rows).
- K8: count number of attributes renamed × naming-mode rank
  (descriptive=0, abbreviated=1, cryptic=2, anonymized=3), not edit
  distance.
- K10 (revised 2026-05-19): the original "apply the mask after K3 drops"
  fix is a **no-op** under the actual pipeline order
  (K3 → K10 → K8, verified). The mask is already built against post-K3
  sources by construction. Replacement landed under step 1: K10 now
  writes `output/baselines/knob_10_realised.csv` with
  `swap_rate = swap_cells / reshufflable_count`, and the audit adds a
  rate-based check `knob_10_realised_rate_monotonicity` as the
  load-bearing K10 verdict. Rate is invariant to K3's drop so K10's
  dispersion-dial monotonicity surfaces without the
  compromised-mask-depopulation confound.
- Land in
  [scripts/analyze_monotonicity.py](../usecases_synthetic/scripts/analyze_monotonicity.py)
  + per-knob `apply_knob_*.py` provenance writers.

**C4. EM-block ceiling cliff — DECIDED 2026-05-22: throw away
word-level BM25, replace with char-ngram BM25.** The existing
`bm25_blocker` roster member used `bm25s.tokenize` with English
stopwords on word tokens (`text_cols: [name]`). K1 paraphrase
rewrites whole words → word overlap drops → BM25 recall collapses.
Char-ngrams over a `(3, 5)` window with `char_wb`-style padding
preserve enough sub-string signal to survive paraphrase.

Implementation:
- [bm25_blocker.py](../usecases_synthetic/lib/bm25_blocker.py): added
  `tokenizer: {"word", "char_ngram"}` + `ngram_range: tuple[int, int]`
  params. Word mode is unchanged; char-ngram mode bypasses
  `bm25s.tokenize` and emits `list[list[str]]` of padded n-grams
  directly to `BM25.index` / `BM25.retrieve` (which accepts
  pre-tokenised input). Helper: `_char_ngram_tokens`.
- 4 unit tests added (`TestBM25BlockerCharNgram` in
  [test_bm25_blocker.py](../usecases_synthetic/tests/test_bm25_blocker.py)),
  all 25 BM25 tests pass.
- Committee config swap: word-BM25 retired across
  [em_blocking_committee.yaml](../usecases_synthetic/config/committees/em_blocking_committee.yaml),
  [em_blocking_committee_music.yaml](../usecases_synthetic/config/committees/em_blocking_committee_music.yaml),
  [em_blocking_committee_games.yaml](../usecases_synthetic/config/committees/em_blocking_committee_games.yaml),
  [em_blocking_committee_products.yaml](../usecases_synthetic/config/committees/em_blocking_committee_products.yaml).
  All four now use `tokenizer: char_ngram`, `ngram_range: [3, 5]`.

`sc_block` retrain with K1/K8-augmented data (option a) is NOT
landing in this round. If char-ngram BM25 doesn't lift recall enough
at the next rerun, retrain becomes a follow-up.

**C5. K8 anonymized rename — DECIDED 2026-05-22: keep full anonymize
as intentional.** The user's directive: string-only matchers failing
completely at hard is a feature, not a bug. Hard is the "LLM /
embedding matchers must carry it; string-only dies by construction"
level. No config change; the existing K8 anonymized stage stays at
full intensity.

Goal 4 ("no silent member collapses with drop > 0.5") is therefore
relaxed for K8-driven collapses at hard: a `label_jw 1.0 → 0.36`
drop is now classified as *expected* when K8 anonymized fired, not a
red flag. Future R7.3 final reports should narrate this explicitly
("K8 anonymize at hard kills string-only matchers by design; the
committee's macro_f1 holds via LLM/embedding members"). No code
change required — the `detect_collapses` audit still flags them, the
narration interprets them.

**C6. Track ceiling-immunity as a first-class metric. LANDED
2026-05-19 under step 1.** (`compute_ceiling_responsiveness` in
[monotonicity.py:848](../usecases_synthetic/lib/monotonicity.py#L848)
wired through
[analyze_monotonicity.py](../usecases_synthetic/scripts/analyze_monotonicity.py);
`monotonicity_report.csv` gains a `ceiling_responsiveness` column.
Step-2 finding: at n=4 levels the Pearson r is sparse / noisy and
`monotonicity_best_member.csv` is the more readable artifact for
ceiling-immunity; R7.3 narration should lead from that file.)
- "Best-member is immune (`text_clean` on music Norm)" is currently
  noted in prose but isn't surfaced in any audit. The user picks the
  best member; if a difficulty knob has no effect on the best member
  it should be flagged.
- Add to
  [analyze_monotonicity.py](../usecases_synthetic/scripts/analyze_monotonicity.py):
  per-(stage, knob) `ceiling_responsiveness` = correlation between knob
  realised level and best-member F1. Anything near 0 = noop knob for
  that stage's ceiling.

**C7. Re-baseline LLM cost ceilings. PENDING — operational item, not
decided 2026-05-22.** Becomes load-bearing once the K2 cache
pre-population run is authorised (precondition for C1 to hit cache
on the music/games/products reruns). Re-estimate budget once the C12
prompt-v2 work is sized too (LLM judge + LLM normalizer prompts get
longer; per-call token counts go up; cache hit rates on existing
entries reset to zero via the prompt_version bump).
- The K2 LLM cache stats from the last music run (~1585 small calls,
  12 full calls, ~3500 if we raise count per pair as in C1) at
  `gpt-5.4-mini` are tractable. Re-running with the proposed
  `interpolation_count: 100` × 2 pairs × 3 levels = 600 calls/domain.
  Budget impact noted in the run plan.

**C8. Make the regenerated EM splits reflect cross-level difficulty,
not just the corner-case ratio.** (Addresses G8.) **SUPERSEDED 2026-05-22
by C11** — the two-version split structure (`_baseline_pruned` +
`_corner_filled`) replaces the survivor-throttling question with two
first-class evaluation surfaces. Options (a)/(b)/(c) below are
retained for audit-trail traceability; see C11 for the resolved
structure.

Three options for how to treat surviving originals during regen:

- (a) **Drop survivor carry-over entirely.** Regenerate every split
  from scratch — corner positives, corner negatives, easy positives,
  easy negatives, all freshly drawn against the post-K2 surface.
  Maximally tracks difficulty but loses any cross-level row-identity
  anchor; metrics on the regen test at baseline ≠ metrics on the
  regen test at hard even ignoring matchers, because the underlying
  pairs differ. Open-set originals still provide the legacy anchor.
- (b) **Cap survivors at `1 − target_corner_case_ratio`.** At hard
  with target=0.80, survivors get at most 20% of each split; the
  remaining 80% is freshly drawn corner pairs. Easy survivors get
  throttled first via random subsample. Cheaper to implement than
  (c); preserves some row-identity overlap with the original.
- (c) **Re-evaluate survivors against the post-K2 corner-case miner.**
  Pairs that were easy in baseline but are now corner cases (e.g. a
  surviving entity now sits next to a K2-interpolated near-twin)
  get reclassified as corner; pairs still easy under the post-K2
  miner get throttled per (b). Preserves comparability where the
  underlying signal hasn't changed, lets the regen properly track
  difficulty where K2 has restructured the neighborhood. Most
  expensive but cleanest semantics.
- **Prerequisite measurement (G8 last bullet):** before picking,
  run regen-with-survivors vs regen-from-scratch on music / games
  hard and compare committee macro_f1 + corner-only F1. If the
  delta is small (say <2pp on macro_f1), the survivor dilution
  isn't material and option (b) is sufficient; if it's large, (c)
  earns its complexity.
- **Recommendation, pending the measurement:** **(c)** — methodologically
  cleanest, lets monotonicity attribute drop to dial position rather
  than to a mix of new corners + old easies. If the measurement
  shows the issue isn't material, fall back to **(b)** for the
  cheaper fix.
- **Coupling with C1:** G8 only matters once K2's interpolation
  actually moves the realised corner ratio. Sequence the work after
  C1 lands so we can measure the regen-survivor dilution against a
  K2 that's pulling its weight; otherwise we can't tell K2-dormant
  from regen-dilution as causes.

**C9. Fusion protection — DECIDED 2026-05-22: replace "one value close
to gold" with a fusion silver standard.** The current variant-generation
protection guarantees that for each fusion val/test entity at least one
source value stays "close enough" to the baseline val/test value, which
rules out the trivial case where every source value is destroyed. It
does **not** protect the surrounding values: K5 / K6 / K10 are still
free to corrupt the rest of the cluster's source values, and fusion
functions that depend on the full distribution (median, weighted
average, voting, trimmed-mean, Huber M-estimator, etc.) can still fail
even when one valid sample is present. Net effect: a fusion variant
can pass the current protection check yet make good fusion
unachievable downstream.

**Scope — protection vs. evaluation.** The silver standard is a
*protection target* used during variant generation, **not an
evaluation surface**. Fusion committee monotonicity in R7.2
continues to be evaluated against the **fixed human-authored fusion
val/test set** (~100 + 100 per domain on music/games/products;
companies pending R0) — same entities at every level, source values
progressively corrupted by K5/K6/K10 because the protection set
keeps these entities surviving K2/K3/K4. This parallels EM:

- **EM evaluation surfaces (per level):** two closed-set comparisons
  (see C10 for the full decision):
    - closed-set on the **original baseline test set** at
      `usecases/<domain>/input/entitymatching/<src1>_2_<src2>_test.csv`
      — frozen across baseline/easy/medium/hard, a cross-level
      reference;
    - closed-set on the **regenerated test split** per level via
      [lib/corner_case_miner.py:regenerate_em_splits](../usecases_synthetic/lib/corner_case_miner.py)
      — tracks K2's corner-case ratio per level, which is exactly the
      regen semantics G8 / C8 are reconsidering.
  Open-set `test_gold` scoring is retired (see C10).
- **Fusion evaluation surface (per level):** fixed
  `usecases/<domain>/input/fusion/{validation,test}_set.xml`. Same
  entities every level; macro_accuracy drops monotonically as
  K5/K6/K10 corrupt the source values for those entities.
- **Silver standard's role:** it sits between K2/K3/K4 (which
  protects fusion entity *survival*) and K5/K6/K10 (which corrupts
  the *surrounding distribution*). It ensures the post-K5/K6/K10
  cluster is still fusion-recoverable, so the eventual evaluation
  on the test set isn't degraded by a destroyed distribution rather
  than by genuine difficulty.

The replacement protection target is a per-cluster **fusion silver
standard**, conceptually parallel to pooled-positives for EM:

1. For each cluster in the pooled positives (the silver-standard
   cluster set already built early in the pipeline), apply the fusion
   functions defined in the human-baseline Jupyter notebook to the
   baseline source values for that cluster.
2. The fusion output per cluster becomes the **"correct" fusion value
   for that cluster** — the silver-standard target, treated as ground
   truth for variant-generation protection (independent of the much
   smaller human-authored val/test gold).
3. During variant generation, K5 / K6 / K10 are constrained so that
   the **post-knob fusion** (computed with the same fusion functions)
   still produces a value "close enough" to the silver value — i.e.
   the surrounding distribution must remain fusion-recoverable, not
   just contain one preserved sample.

Critical implementation requirement: the **same normalization** that
the human-baseline notebook applies before fusion must be replicated
both when building the silver values and when comparing post-knob
fusion outputs against them. Otherwise the protection comparison is
not apples-to-apples and any drift between the silver-standard fusion
pipeline and the eventual R7.2 fusion-stage evaluation will manifest
as inflated / deflated fusion accuracy.

Implementation work (prerequisite for the regens in steps 5-6):

- Locate the human-baseline fusion notebook(s) under
  [usecases_synthetic/](../usecases_synthetic/) and extract the
  fusion-function + normalization stack into a reusable callable
  (`lib/fusion_silver_standard.py`).
- For each domain, run the extracted fusion stack against the pooled
  positives on the full baseline dataset → write
  `usecases_synthetic/baselines/<domain>/fusion_silver_standard.{csv,json}`
  keyed by `cluster_id × attribute`.
- Wire the silver-standard check into the variant-generation
  protection pass; replace / augment the current "one value close"
  rule with a "post-knob fusion close to silver value" rule, per
  attribute, with tolerances drawn from the existing
  `fusion_committee_<domain>.yaml` `evaluation_params` (numeric
  `tolerance`, `lexical_extended_jaccard` threshold for strings).
- Couples with C11 (which supersedes C8): the silver standard is
  also the natural reference for the regen survivor / corner
  trade-off — surviving originals that would have shifted the
  cluster's fusion value off-silver get flagged here as well, and
  the silver-recoverable check provides an attribute-level lens that
  complements C11's EM-pair-level Set 1 vs Set 2 split.

Side effect: this is more restrictive than the current rule, so some
K6 / K10 corruptions that were previously allowed at hard will now be
backed off when they would push the cluster off the silver value.
That's the intentional trade-off — hard fusion difficulty is bounded
by "still recoverable by the human-baseline fusion stack", not by
"one preserved sample remains".

**C10. EM committee evaluation — DECIDED 2026-05-22: drop open-set
scoring; the committee surfaces exactly two closed-set EM scores per
source pair, and only the regen-test surface drives monotonicity.**

Current state ([committee_em.py:1140-1261](../usecases_synthetic/lib/committee_em.py#L1140-L1261))
emits three EM surfaces per pair: `f1_vs_regenerated_val` (closed-set,
current primary), `f1_vs_regenerated_test` (closed-set, sanity), and
`f1_vs_test_gold` (**open-set** vs original human gold, "secondary for
cross-paper comparability"). The open-set surface uses
`score_em_correspondences` ([committee_em_scoring.py:178+](../usecases_synthetic/lib/committee_em_scoring.py#L178)),
which counts every prediction outside the gold's positive list as FP.
On large source pairs (games 92k records, music ~50k) the matcher's
prediction set is many times larger than the small human-authored
positive list, so the FP penalty is dominated by "predicted a pair
the gold didn't enumerate" rather than "predicted a real negative".
The number is not interpretable as matching quality and gets cited
nowhere useful.

Replacement: **two** closed-set scores per pair, both via
`score_em_correspondences_closed_set`
([committee_em_scoring.py:112](../usecases_synthetic/lib/committee_em_scoring.py#L112)):

- `f1_baseline_test`: closed-set on the **baseline-pruned test split**
  written per-variant as `em_test_baseline_pruned.csv` (see C11) —
  the original baseline test gold with any pair removed where one or
  both ids no longer exist in the variant. The remaining labeled
  positives + negatives form the judged pair universe; predictions
  outside are out-of-scope (not FP). Smaller **and more corner-biased**
  at higher difficulty levels (K2's size-balance drop preferentially
  removes low-density / easy entities; see G8 for the mechanism).
  A per-level reference number, **not** a monotonicity-verdict
  surface and **not** a stable cross-level baseline — its level-by-
  level numbers drift with K2 intensity.
- `f1_regen_test`: closed-set on the **per-level corner-filled test
  split** written as `em_test_corner_filled.csv` (see C11) — the
  baseline-pruned split, backfilled with corner-case pairs mined from
  the variant to restore the original baseline split's cardinality
  and positive ratio. **This is the load-bearing surface for EM
  committee macro_f1 monotonicity** — the easy → medium → hard
  slope verdict (C2) is computed on `f1_regen_test`.

These are the only two EM F1 numbers in the committee output.

Implementation:

- Replace `score_em_correspondences(preds, gold_df)` at
  [committee_em.py:1192](../usecases_synthetic/lib/committee_em.py#L1192)
  with `score_em_correspondences_closed_set`; rename the metric key
  to `f1_baseline_test`.
- Drop `f1_vs_test_gold` / `precision_vs_test_gold` /
  `recall_vs_test_gold` from per-pair metrics
  ([committee_em.py:1259-1261](../usecases_synthetic/lib/committee_em.py#L1259-L1261))
  and the macro fields
  ([committee_em.py:1492](../usecases_synthetic/lib/committee_em.py#L1492)).
- Demote `f1_vs_regenerated_val` from the headline. Keep regen_val
  internally as a val/test agreement sanity check (logged but not
  surfaced in the public per-member metrics) so we still notice if
  val and test diverge unexpectedly. Promote `f1_regen_test` to the
  headline `f1`.
- Update the headline `f1` fallback chain at
  [committee_em.py:1214-1244](../usecases_synthetic/lib/committee_em.py#L1214-L1244):
  new chain is `regen_test → baseline_test → pool`. Pool-as-gold
  stays as the last-resort fallback (baselines, or domains whose
  regen builder produced nothing).
- [analyze_monotonicity.py](../usecases_synthetic/scripts/analyze_monotonicity.py):
  EM committee monotonicity reads from `f1_regen_test` only. The
  `f1_baseline_test` column is written to `monotonicity_report.csv`
  for reference but excluded from PASS/FAIL verdicts (analogous to
  how legacy raw-count rows were demoted under C3).
- Verify per-domain that the original baseline test sets contain
  labeled negatives — sample-check
  `<src1>_2_<src2>_test.csv` for `label=0` rows on music / games /
  products / companies. If any ships positive-only, that domain's
  `f1_baseline_test` degenerates to a recall-style metric; flag in
  the R7.3 narration when it happens.
- `score_em_correspondences` (open-set scorer) is no longer
  referenced in the committee. After a grep for outside callers
  (tests / other modules), delete it from
  [committee_em_scoring.py](../usecases_synthetic/lib/committee_em_scoring.py)
  if unused.
- R7.3 final reports: rewrite the EM section to narrate the two
  closed-set numbers and explicitly say which one drives monotonicity.
- Statistics XLSX schema updates: `build_statistics.py` columns
  rename to match the new keys.
- Memory update follow-up: `feedback_committee_reporting_conventions`
  should be amended — open-set `test_gold` is fully retired, not just
  "never primary". Both surfaces are closed-set; only `f1_regen_test`
  drives monotonicity. **[done 2026-05-22.]**

**C11. EM split regeneration — DECIDED 2026-05-22 + LANDED 2026-05-25:
emit two parallel versions of every EM split (train / val / test)
mirroring C10's two evaluation surfaces. Supersedes C8.**

Pre-C11 state:
[lib/corner_case_miner.py:regenerate_em_splits](../usecases_synthetic/lib/corner_case_miner.py)
emitted one regenerated split per train/val/test per pair — kept
every original `(id1, id2, label)` whose ids both still exist
post-K2/K4, then backfilled to the original size with corner
positives/negatives sized to hit `target_corner_case_ratio` plus
easy fills.

New structure: each split (train, val, test) is emitted in **two
parallel versions** per source pair:

- **`_baseline_pruned`** (Set 1) — start from the original baseline
  split (`usecases/<domain>/input/entitymatching/<src1>_2_<src2>_<split>.csv`);
  drop any `(id1, id2)` row where either id no longer exists in the
  post-K2/K4/K3 variant. **No backfill.** Smaller than the original;
  per G8 (revised), survivors are biased **toward corner-pairs** of
  the original gold (K2's size-balance drop preferentially removes
  low-density / easy entities). Both the shrinkage and the corner-
  bias intensify with K2 intensity across levels — Set 1 is a
  level-dependent, corner-biased subsample of the original gold,
  **not** a stable cross-level reference.
- **`_corner_filled`** (Set 2) — start from `_baseline_pruned` (Set 1),
  then backfill with corner-case pairs mined from the variant until
  the split reaches the original baseline split's cardinality,
  preserving the original positive ratio. Set 2 is the size-restored,
  difficulty-tracking surface.

**Set 1 ⊂ Set 2 by construction.** Three splits × two versions =
**6 EM artifacts per source pair**:

```
em_train_baseline_pruned.csv   em_train_corner_filled.csv
em_val_baseline_pruned.csv     em_val_corner_filled.csv
em_test_baseline_pruned.csv    em_test_corner_filled.csv
```

For **training** (matchers that consume `pair_train` — ditto / magellan /
sc_block): default to `_corner_filled` (matches what the matcher will
be scored against at evaluation time); `_baseline_pruned` is available
for ablation runs ("matcher trained on baseline survivors only, no
corner augmentation"). LLM-matcher zero-shot doesn't train so the
choice is moot for it.

For **evaluation** (per C10):
- `f1_baseline_test` reads `em_test_baseline_pruned.csv` (Set 1).
- `f1_regen_test` reads `em_test_corner_filled.csv` (Set 2). Monotonicity
  verdicts run on this column only.

Implementation:

- [lib/corner_case_miner.py:regenerate_em_splits](../usecases_synthetic/lib/corner_case_miner.py)
  refactored to emit both versions per split. The current single-output
  path is renamed internally to `_corner_filled` (its behaviour already
  matches Set 2's contract); a new `_baseline_pruned` path implements
  the simpler subset filter.
- File naming: `em_<split>_<version>.csv` under each variant's existing
  `em/<pair>/` directory.
- `VariantBundle` + [committee_em.py:_load_labelled_split](../usecases_synthetic/lib/committee_em.py)
  load both versions per `(pair, split)` and key them by version.
- Committee scoring (C10 implementation) reads the two versions per
  pair: Set 1 → `f1_baseline_test`, Set 2 → `f1_regen_test`.
- Tests: unit coverage for the subset invariant (Set 1 ⊂ Set 2),
  size preservation (|Set 2| = |original|), positive-ratio
  preservation, corner-mining still respects `target_corner_case_ratio`
  on the backfill portion, and the K2/K3/K4 id-existence check that
  drives Set 1 pruning.

**C8 resolution.** The C8 survivor-handling options (a / b / c) are
**superseded by C11**. Under the new structure:

- Survivors are kept by construction in *both* sets — no "throttle"
  question. Set 1 is survivor-only by definition; Set 2 keeps them
  and adds corners on top.
- **G8's original "easy-survivor dilution" framing is corrected, not
  just superseded.** Per the G8 revision (2026-05-22, grounded in
  the K2 size-balance code), the actual dilution source in the
  current regen is the *easy-fill backfill*, not surviving originals
  (which are corner-biased, not easy-biased). C11's `_corner_filled`
  spec — "filled up to the old size using corner-cases", no easy
  fills — removes the actual dilution by construction. Both halves
  of Set 2 (survivors + corner-mined backfill) are corner-heavy, so
  the K2-attributable F1 drop should surface cleanly. The "weighted
  average over easy survivors + new hard corners" framing no longer
  applies.
- One residual sub-question: what to do when the natural corner-fill
  size (`|original| - |Set 1|`) falls short of `target_corner_case_ratio`
  (i.e. K2/K3/K4 didn't remove enough survivors to backfill with the
  configured corner share). Two paths:
  - **(i) Accept the realised ratio shortfall.** Backfill is 100%
    corners up to the cap left by survivor removal; if the resulting
    Set 2 corner ratio < target, log the gap. Matches C3's
    "audit by realised intensity, not configured target" principle.
  - **(ii) Displace some surviving pairs to inject more corners.**
    Keeps target hit but breaks the Set 1 ⊂ Set 2 invariant and
    re-introduces the survivor-throttle question C11 was meant to
    retire.
- **Recommendation: (i).** The C2 monotonicity slope still tracks
  K2's *dial* via the realised corner ratio in Set 2; the cap-driven
  shortfall surfaces as a documented data point, not a quietly
  injected bias.

**C12. Restructure normalization + fusion committees so each member is
a coherent approach producing a full per-record output — not a
per-(attribute, strategy) datum with voting fallback. — DECIDED
2026-05-22.**

Current state (grounded):

- **Fusion** ([lib/committee_fusion.py:491-716](../usecases_synthetic/lib/committee_fusion.py#L491-L716),
  [fusion_committee_music.yaml](../usecases_synthetic/config/committees/fusion_committee_music.yaml)):
  YAML lists per-attribute `strategies:` blocks. The runner emits one
  member per (attribute, strategy) — music has 30+ such — and for
  each member runs a `DataFusionEngine` where the strategy applies to
  *only* its target attribute; every other attribute falls back to
  `voting`. Per-member `macro_accuracy` is therefore mostly voting
  noise plus one attribute's actual behavior. The aggregated
  `overall_best_accuracy` is the per-attribute oracle (best strategy
  per attribute, picked retroactively on whatever data was scored —
  no held-out val).
- **Normalization** ([lib/committee_norm.py:287+](../usecases_synthetic/lib/committee_norm.py#L287),
  [normalization_committee_music.yaml](../usecases_synthetic/config/committees/normalization_committee_music.yaml)):
  Members are coherent classes (TextCleanNormalizer,
  DateIsoNormalizer, …) with `applies_to: [attr, ...]`. Macro_f1 per
  member is across the applies_to subset only. Each member is
  internally coherent, but the subsets vary — cross-member macro_f1
  is not directly comparable (TextClean averages over 3 attrs,
  DateIso over 1).

Replacement structure (both committees): each committee member is a
coherent approach that produces a **complete fused / normalized
output across all attributes**. The member roster names the
approach; per-member macro_accuracy / macro_f1 is therefore directly
comparable and interpretable as "what does this approach achieve
end-to-end".

**Fusion member roster (resolved 2026-05-22).** All non-PyDI members
fall back to **val-best PyDI** (same val-selection mechanism as
`pydi_per_attribute_optimal`, restricted to type-appropriate PyDI
functions) for attribute types where the member's main method is
**semantically incompetent** — not voting. Macro_accuracy is then
comparable across members because every member covers every attribute
and the fallback is principled per domain.

Semantic-competence per type (grounded at
[td_batch_fusion.py:90-93](../usecases_synthetic/lib/td_batch_fusion.py#L90-L93)
+ [llm_judge_fusion.py:238-302](../usecases_synthetic/lib/llm_judge_fusion.py#L238-L302)):

| Type | Methods that handle it natively |
|---|---|
| string / categorical / date | PyDI built-ins, `llm_judge` (prompt v2), all TD methods (TruthFinder / LTM / CaseFusion / FusionQuery / AccuSim), voting, prefer_higher_trust |
| numeric | PyDI numeric ops (median, trimmed_mean, huber_m_estimator, median_of_means, maximum, …), `llm_judge` (prompt v2, interpolation/synthesis permitted), AccuSim with numeric-tolerance similarity (native via the `similarity` hook), voting, prefer_higher_trust |
| list | PyDI list ops (union, intersection, intersection_k_sources), `llm_judge` (prompt v2, LLM-chosen set op), LTM (native multi-truth via `alpha_0` etc.), AccuSim with Jaccard similarity (native via the `similarity` hook), voting, prefer_higher_trust |

`llm_judge` (prompt v2, see implementation row below) is type-aware
via the LLM itself — the prompt names the attribute and presents the
candidates, and the LLM picks its own operation per call (verbatim
pick, aggregation, set op, normalization, **or interpolation /
synthesis of a value not present in any source** — synthesis is
explicitly permitted, the verbatim constraint of prompt v1 is
removed). The chosen operation is part of the JSON response and
logged for audit. **`llm_judge` therefore has no PyDI fallback** —
it covers every attribute type natively.

Members not listed in a row are "semantically incompetent" for that
type (verbatim string-claim only, no aggregation / set semantics /
synthesis) and use val-best PyDI for that type instead.

Roster:

- `pydi_per_attribute_optimal` — for each attribute, sweep PyDI's
  type-applicable built-in conflict resolution functions on the
  **fusion validation set**; lock in the per-attribute winner per
  domain; apply the locked map across baseline + variant levels.
- `llm_only` — `llm_judge` on **every attribute, no fallback**.
  Prompt v2 lets the LLM choose the operation per call (verbatim
  pick, statistical aggregation, set op like union / intersection,
  date normalization, **or interpolation / synthesis of a value not
  present in any source**). The operation choice is logged per
  (entity, attribute) for audit. No val-selection pass needed for
  this member.
- `fusionquery_only` — `fusionquery` on string / categorical / date;
  val-best PyDI on numeric + list.
- `truthfinder_only` — `truthfinder` on string / categorical / date;
  val-best PyDI on numeric + list.
- `ltm_only` — `ltm` on string / categorical / date **and on list**
  attributes (native multi-truth, no fallback); val-best PyDI on
  numeric.
- `casefusion_only` — `casefusion` on string / categorical / date;
  val-best PyDI on numeric + list.
- `accusim_only` — `accusim` on **every** attribute type via the
  type-aware `similarity` hook: Dice on string / categorical / date,
  numeric tolerance on numeric, Jaccard on list. No PyDI fallback;
  the similarity function used per type is logged in the per-member
  selection map.
- `voting_only` — `voting` on every attribute. Coherent baseline.
- `prefer_higher_trust_only` — `prefer_higher_trust` on every
  attribute. Coherent baseline.

Best-member macro_accuracy across this roster = "user-attainable
ceiling": the score a user gets by picking the best coherent fusion
approach. Comparability holds because all members cover all
attributes via either their native method or a val-selected PyDI
fallback. Aligns with `feedback_committee_reporting_conventions`
Rule 1 (best-member + macro).

**Normalization member roster (resolved 2026-05-22).** Same shape
as the fusion roster — coherent end-to-end members; `llm_only` has
no fallback under prompt v2.

- `rule_per_attribute_optimal` — for each attribute, sweep the
  type-applicable rule normalizers (TextCleanNormalizer for string;
  DateIsoNormalizer for date; NumberLocaleNormalizer for numeric;
  CountryIsoNormalizer for codelist; TaxonomyLookupNormalizer for
  nominal-with-taxonomy) on the **val set**; lock in the per-attribute
  winner per domain; apply across baseline + variant levels.
- `llm_only` — `LLMCanonicalizer` on **every attribute, no fallback**.
  Prompt v2 (see implementation row below) lets the LLM choose the
  operation per call (`vocab_canonicalize` / `date_normalize` /
  `numeric_normalize` / `categorical_map` / `synthesize` / `abstain`).
  **Synthesis explicitly allowed** — partial-date → ISO completion,
  unit conversion, abbreviation expansion. The operation choice is
  logged per (entity, attribute) for audit. No val-selection pass
  needed for this member.
- `passthrough` — no normalization. Coherent baseline.

`hybrid_rule_llm` was considered and **retired 2026-05-22** —
once LLM covers every type natively under prompt v2, "rule for
type-matched, LLM for residual" collapses to a degenerate variant
of `rule_per_attribute_optimal` / `llm_only` rivalry. Cleaner to
let the two compete directly.

Implementation:

- New YAML schema for both committees: a top-level `members:` list
  where each entry declares the meta-strategy (e.g.
  `strategy: pydi_per_attribute_optimal` with selection logic) or a
  single-method declaration (e.g. `strategy: llm_only` with the
  member's per-method config). Attribute-type fallback rules are
  per-member.
- [lib/committee_fusion.py](../usecases_synthetic/lib/committee_fusion.py)
  rewrite: one `DataFusionEngine.run()` per member; member produces
  one fused DataFrame; score; record per-member `macro_accuracy` +
  per-attribute accuracies. Retire the (attr, strat) inner loop.
- [lib/committee_norm.py](../usecases_synthetic/lib/committee_norm.py)
  rewrite: members declare meta-strategy + per-method config;
  cover all attributes via the member's internal routing.
- Val-set selection plumbing: needed by `pydi_per_attribute_optimal`
  (every attribute) and by every TD member that has a fallback
  (numeric + list attrs for `fusionquery_only`, `truthfinder_only`,
  `casefusion_only`; numeric attrs for `ltm_only`). One val-evaluation
  pass per member per fallback attribute before the test pass. Cache
  the per-(member, attribute) → method choices in
  `usecases_synthetic/baselines/<domain>/{fusion,norm}_committee_selection.json`
  so monotonicity reruns don't reselect (and a level-shift in K5/K6
  doesn't quietly change a member's identity).
  `llm_only` needs no val selection — `llm_judge` covers every
  attribute natively under prompt v2.
  `accusim_only` needs no val selection either, but the per-type
  similarity function used (Dice / numeric_tolerance / Jaccard) is
  recorded in the same file for audit symmetry.
- **Extend `llm_judge_fusion.py` to prompt v2** ([llm_judge_fusion.py:78-90](../usecases_synthetic/lib/llm_judge_fusion.py#L78-L90)):
  new system prompt that names the attribute and candidates, removes
  the verbatim-only constraint, and asks the LLM to return JSON of the
  form `{"value": <chosen value, may be a list / number / synthesized
  string>, "operation": <"verbatim_pick"|"aggregation"|"union"|"intersection"|
  "normalization"|"interpolation">, "confidence": <0..1>,
  "reasoning": <short text>}`. Bumping the `prompt_version` field
  ([llm_judge_fusion.py:115-121](../usecases_synthetic/lib/llm_judge_fusion.py#L115-L121))
  invalidates the existing cache by construction. The `operation`
  field is the audit trail — surface it in the per-(entity, attribute)
  diagnostic CSV under `output/fusion_diagnostics/<domain>/<level>/llm_only_operations.csv`
  so stats like "for the duration attribute, LLM picked
  `interpolation` 78% of calls, `verbatim_pick` 22%" are computable.
  Determinism preserved via the existing `temperature: 0.0` setting
  in the per-domain YAMLs.
- **Extend `llm_normalizer.py` to prompt v2** ([llm_normalizer.py:50-67](../usecases_synthetic/lib/llm_normalizer.py#L50-L67)),
  symmetric to the fusion LLM judge work: new system prompt that
  names the attribute + attribute type, removes the "closed-vocab
  verbatim" constraint, and asks the LLM to return JSON of the form
  `{"value": <canonical form, may be synthesized>, "operation":
  <"vocab_canonicalize"|"date_normalize"|"numeric_normalize"|"categorical_map"|
  "synthesize"|"abstain">, "confidence": <0..1>, "reasoning":
  <short text>}`. Synthesis explicitly allowed (partial-date → ISO
  completion, unit conversion, abbreviation expansion). Bump the
  `PROMPT_VERSION_V1` constant ([llm_normalizer.py:45](../usecases_synthetic/lib/llm_normalizer.py#L45))
  to `v2` — invalidates the existing
  [cache/llm_normalizer/](../usecases_synthetic/cache/llm_normalizer/)
  by construction. Operation log: per-(entity, attribute, operation)
  written to `output/norm_diagnostics/<domain>/<level>/llm_only_operations.csv`
  for stats parity with the fusion side. Determinism preserved via
  the existing `temperature: 0.0` setting.
- [build_statistics.py:92](../usecases_synthetic/scripts/build_statistics.py#L92):
  per-member `macro_accuracy` becomes the meaningful headline.
  Add a per-member "selection map" column for the optimized members
  so the reader sees which PyDI / rule function was picked per attr.
- [analyze_monotonicity.py](../usecases_synthetic/scripts/analyze_monotonicity.py):
  per-member macro tracked across levels; best-member macro is the
  user-attainable ceiling per stage (parallels the SM / EM
  best-member reporting from Rule 1).
- Retire per-(attr, strat) raw rows from the public statistics
  output. If a per-attribute strategy-comparison view is needed for
  ablation, expose it under a separate `fusion_strategy_sweep.csv`
  artifact, not in the committee output.

Coupling with other items:
- **C9 fusion silver standard:** orthogonal. Silver is a
  variant-generation protection target built from the human-baseline
  notebook's fusion functions; C12 changes how the *committee*
  evaluates members on the post-knob data. The human-baseline
  notebook's fusion functions may overlap with what
  `pydi_per_attribute_optimal` picks on val, but they're different
  artifacts.
- **C10 EM evaluation:** independent — C10 is EM-side, C12 is
  fusion + norm.
- **Cascade:** every domain re-runs SM / Norm / EM / Fusion committee
  evaluation against the new member rosters. Adds val-set selection
  pass before the test pass; ~one-time cost per domain.

Open sub-questions:
- Per-domain member roster — pick the same baseline roster for all
  active domains (music / games / products / companies) or tailor?
  Recommendation: same baseline roster across domains; let
  `pydi_per_attribute_optimal`'s val-selection do the
  domain-specific tuning automatically.
- Attribute-type fallback for the single-method TD members (e.g.
  fusionquery_only on music's `tracks` list-typed attr) — voting
  fallback per member spec, surfaced in the per-member selection
  map so it's auditable rather than silent.

**C13. Silver-as-default protection with intact-cluster semantics —
DECIDED 2026-05-28; implementation pending.**

Going forward, all 4 active domains (music / games / products /
companies) default to `protection_source: silver` in variant
generation — NOT gold-only. The semantics decouple two protection
axes that the binary `gold | silver` switch conflates:

| Axis | Gold | Silver |
|---|---|---|
| **Existence** (entity may not be dropped) | YES (K2 protects) | NO (K2 may drop) |
| **Drift** (post-knob value must stay close to target) | YES (gold target) | YES, **iff entity's silver cluster remained intact through K2** |

A silver cluster is **intact** iff every one of its original
member records is still present in the post-K2 source frames. If K2
drops any member (drop-corner-touching or the size-invariant drop
inside K2-interpolate), the cluster is **broken** — its silver
target was computed against the full original membership, no longer
authoritatively represents what the surviving members would fuse to,
and the K1/K6 closeness check is waived for the cluster's remaining
members. Gold members keep their drift protection unconditionally.

This middle ground solves both failure modes of the binary switch:

- gold-only **under-protection**: K1/K6 free to destroy any non-gold
  pool cluster member, so wider-fusion across the pool drifts
  arbitrarily and silver-recoverability is lost for ~93% of records.
- full-silver **over-protection** (Bug 3 pattern): on
  ``pool_quality: live`` domains (products), silver covers every
  source record → K1/K6 closeness check fires on every cell → dial
  collapses to cosmetic noise.

Under intact-cluster-silver on products: ~487 K2 drops break maybe
300-400 of ~800-900 clusters; the ~800-1500 records in broken
clusters get unconstrained K1/K6 (recovers the difficulty signal);
the ~600-1500 records in intact clusters keep silver-drift
protection (preserves recoverability where the silver target is
still meaningful).

Prerequisites:
- Products silver standard must be built (currently deferred from
  step 4b; `data_cleaned_final` schema landed under step 4h, so the
  silver builder is now unblocked).

Implementation (~2-3 days):

1. **Build products silver standard** —
   `scripts/build_fusion_silver_standard.py --domain products`.
   ~$1-5; one fusion-engine sweep over the ~3012-record pool.
2. **New helper** `load_intact_silver_clusters(domain,
   surviving_record_ids: set[str]) -> set[str]` in
   [lib/fusion_silver_targets.py](../usecases_synthetic/lib/fusion_silver_targets.py).
   Returns cluster IDs whose entire original `source_ids` list is a
   subset of `surviving_record_ids`.
3. **Update K1 + K6 closeness check dispatchers** to consult
   `(target_values, intact_cluster_ids)` instead of just
   `target_values`. Per-cell dispatch:
   - entity ∈ gold → check against gold target (unchanged);
   - else if entity's silver cluster ∈ intact_cluster_ids → check
     against silver target;
   - else → no check, mutation unconstrained.
4. **Generate-variant plumbing**: after K2 fires, derive
   `surviving_record_ids` from the post-K2 source frames; pass to
   the joint K1/K5/K6 step alongside the existing
   `protection_source` flag.
5. **Tests**: intact/broken classification on hand-built cluster
   fixtures, dispatch switch (gold / intact-silver / broken /
   non-cluster) covers each branch, regression test that the
   K1/K6 closeness check waives correctly on broken-cluster
   members.
6. **Cascade reruns**:
   - products first — A/B against the current gold-only run already
     on disk;
   - music / games / companies under the new default (they're
     already running with silver standards built);
   - the 4 silver standards are reused as-is; only the dispatch
     logic changes.

Couplings:
- **Bug 3 (drop-corner protection narrowing)**: stays. Drop-corner
  protects fusion val/test only; silver provides drift protection,
  not existence protection.
- **C9 silver standard infrastructure**: unchanged. The existing
  `build_silver_standard` (with `include_singletons=False`) +
  `load_combined_target_values` + `load_silver_member_to_cluster`
  helpers all reused; only `load_intact_silver_clusters` is new.

### Cascade impact — every active domain reruns

R-1 is not a music/games-only improvement. The C1–C12 changes alter
the variant pipeline + difficulty audit + EM regen + committee
scoring + LLM prompts at the level of `apply_knob_*` /
`generate_variant.py` / `analyze_monotonicity.py` /
`corner_case_miner.py` / `committee_em.py` / `committee_fusion.py` /
`committee_norm.py` / `llm_judge_fusion.py` / `llm_normalizer.py`,
plus the silver standard infrastructure (C9). **Every active domain
has to be regenerated and re-validated against the same fixed
contract** for the verdict to hold. Order:

1. **R1 (products schema upgrade)** — promoted ahead of music + games
   reruns. Rationale: the products redesign authors per-knob configs
   (K3 drop, K5 format, K8 rename_table, K10 reliability) for ~12
   columns of mixed text / categorical / numeric content. Doing that
   work first means (a) the new K8 rename strategy (C5) and new K10
   mask ordering (C3) get exercised on a richer surface before
   music/games rerun against them, and (b) the products numeric
   attributes (`vram_gb`, `storage_gb`, `price`) expose K5 / K10 edge
   cases that music/games can't (music has duration only; games has
   the score columns but no storage units / capacity formats).
   Prerequisite: R3 (train the SC-block checkpoint for products with
   whatever text_cols R1 lands).
2. **music FULL rerun** — primary diagnostic surface for the existing
   knob set. Confirms K2 actually moves, easy ≥ baseline, audit metrics
   pass cleanly under the music distribution (low natural corner
   ratio).
3. **games FULL rerun** — secondary confirmation; checks the fixes
   generalise to the games distribution (high natural corner ratio of
   0.67, different K8 mode mix, two surviving source pairs post-F11).
4. **R0 (companies-FULL)** — gated on the user's larger fusion gold.
   When the gold lands, companies S.7 executes the R-1 fixes natively
   (C2 slope contract, C3 audit metrics, C4 char-ngram BM25, C5 K8
   anonymize-as-intentional, C6 ceiling-responsiveness, C9 silver
   standard, C10/C11 EM closed-set scoring + two-version splits,
   C12 coherent committee members). Without R-1 first, companies-FULL
   would ship with the same dormant K2 + diluted-regen issues we just
   diagnosed on music/games.

If C4 picks option (a) (noise-aware sc_block encoder), retrain across
music + games + products + companies with the same K1/K8-augmented
recipe at the start of each domain's rerun so blockers behave
consistently.

After all four full-domain runs land, refresh
[statistics/](../usecases_synthetic/statistics/) via
`build_statistics.py` so the central XLSX reports reflect the new
variants.

### Action plan

> **Status as of 2026-05-28**: steps 1-4 are done 2026-05-19/22
> (instrument, diagnose, calibrate, fix code for C1/C2/C3/C4/C5/C6).
> Sub-steps 4b-4i all landed 2026-05-23 through 2026-05-28:
> 4b (C9 silver), 4c (C11 EM regen), 4d (C10 EM scoring), 4e (C12
> committee restructure + LLM prompt v2), 4f (K1 audit), 4g (games EM
> diagnosis), 4h (pre-rerun knob review), 4i (drop-corner + non-corner
> refill). 4j (pre-rerun cleanup) landed 2026-05-28 — mandatory list
> + optional small-augmented swept (~497 MB freed); `sm_tuning` +
> `ditto_checkpoints` retained. **Steps 5 onward (variant regens) are NOT
> gated on a separate cache-warming run** — the variant generation
> itself populates the LLM caches as it runs; products is the
> recommended first rerun (smallest domain, ~$35-80 cost, validates
> the full step-4h + step-4i change set end-to-end before music +
> games).

1. **Instrument** (low risk, high info) — **[done 2026-05-19]**:
   - **[done]** Per-guardrail rejection logging in
     [entity_interpolation.py](../usecases_synthetic/lib/entity_interpolation.py)
     + counters surfaced on `knob_02_realised.csv` (C1).
   - **[done]** `ceiling_responsiveness` column added to
     `monotonicity_report.csv` via new
     `compute_ceiling_responsiveness` in
     [monotonicity.py](../usecases_synthetic/lib/monotonicity.py) (C6).
   - **[done]** New intensity audits added beside the existing
     raw-count rows in
     [generate_variant.py](../usecases_synthetic/scripts/generate_variant.py)
     (C3): K5 `distinct_format_families`, K8 `naming_intensity`
     (rung-weighted), K10 `realised_rate_monotonicity` (rate-based,
     replacing the no-op "apply mask after K3" — see G4 correction).
   - **Step 1 details:** see the "Step 1 (instrument) — landed
     2026-05-19" section at the top of R-1.
2. **Diagnose** by re-running R7.2 on existing music + games variants
   with the new instrumentation (no regen needed — read the existing
   variant files). **[done 2026-05-19]** — full findings at
   [plan_revision_step2_findings.md](plan_revision_step2_findings.md).
   Key outcomes:
   - **G1 cause revised**: K2 dial dormancy is a `strict_cache=True`
     cache-miss problem (music: 1080 misses), not guardrail rejection.
     The C1 decision in step 3 now reads as α/β/γ (see C1 above).
   - **K5 + K8 intensity audits PASS** on both domains under the new
     metrics (music K5 2/2/2, K8 0/24/48; games K5 1/2/3, K8 0/3/43).
     The legacy raw-count rows can be demoted in R7.3 narration.
   - **K10 rate audit deferred** to step-3 regen — needs the new
     `knob_10_realised.csv` artifact which requires K10 to re-run
     against post-K3 sources (existing variant dirs predate the
     artifact).
   - **Ceiling-responsiveness column** (C6) lands but is sparse at
     n=4 levels; `monotonicity_best_member.csv` is the more readable
     view for ceiling-immunity (music norm `text_clean`, games SM
     `llm_openai` at 1.0 flat).
   - **G2 / G6 / G7 unchanged** from prior diagnosis.
3. **Calibrate** (config-only, no code) — **[done 2026-05-22]**:
   - C1: never-fail-on-miss adopted; strict_cache defaults flipped
     to False everywhere; no K2 interpolation_count change yet
     (becomes a tuning knob once cache populates).
   - C2: monotonicity contract is easy → medium → hard; baseline is
     reference-only. No K1 easy-floor change.
   - C5: full K8 anonymize kept; "string-only dies at hard" is
     intentional.
4. **Fix code** — **[done 2026-05-22]**:
   - C3 K5/K8/K10 audit metrics already landed under step 1.
   - C4: char-ngram tokenisation added to `BM25Blocker`; all five
     em_blocking committee yamls swapped from word-BM25 to
     char-ngram BM25 with `ngram_range: [3, 5]`. 4 new unit tests
     added.
   - C1 strict_cache forcings removed from `generate_variant.py` +
     both standalone CLIs.
   - C2 `SLOPE_LEVELS` introduced; `match_signals` checks
     monotonicity on `easy → medium → hard` only.
4b. **Build the fusion silver standard** (C9) — **[done 2026-05-23]**:
   - Extracted per-domain fusion stack + normalization into
     [lib/fusion_silver_standard.py](../usecases_synthetic/lib/fusion_silver_standard.py)
     (music / games / companies; products deferred until R1).
   - Built silver artifacts:
     `baselines/{music,games,companies}/fusion_silver_standard.{csv,json}`
     — 4 280 / 8 974 / 1 088 clusters respectively.
   - Replaced "one value close to gold" protection with
     silver-augmented dispatch via
     [lib/fusion_silver_targets.py](../usecases_synthetic/lib/fusion_silver_targets.py);
     K1 + K6 dispatchers gained a `protection_source: str = "gold"`
     flag (CLI flag `--protection-source` on `generate_variant.py`).
     Silver merges with gold-wins-per-(member, attribute) per the
     2026-05-23 user directive — gold values still authoritative for
     fusion val/test entities.
   - K5 + K10 intentionally NOT wired (per user 2026-05-23): K5
     format-equivalent values fuse to the same result; K10 doesn't
     change cell values.
   - 61 new tests (49 silver-standard + 12 silver-targets) pass.
   - **User gate** signed off 2026-05-23 on the music silver CSV
     against `usecases/music/music_workflow.ipynb` cell 40 output.
4c. **EM regen split refactor — emit two parallel versions** (C11) —
    **[done 2026-05-25]**:
   - Refactored [lib/corner_case_miner.py:regenerate_em_splits](../usecases_synthetic/lib/corner_case_miner.py)
     to emit a single row list with a new `version` column; each
     surviving pair appears in both `baseline_pruned` and
     `corner_filled`, each corner-mined backfill pair appears only
     in `corner_filled`. Easy-fill backfill removed entirely;
     `pool_positives_by_pair` / `cluster_positives_by_pair` /
     `target_ratio` kept in the signature for API + audit-trail
     stability but no longer consumed for backfill.
   - Per-version files emitted as `<pair>_<split>_<version>.csv`
     under `input/entitymatching/` (3 splits × 2 versions = 6 files
     per source pair). Legacy `*_regenerated.csv` cleanup glob
     extended in both
     [apply_knob_02_niche.py](../usecases_synthetic/scripts/apply_knob_02_niche.py)
     and [generate_variant.py:_rerun_regen_post_k4](../usecases_synthetic/scripts/generate_variant.py).
   - `_corner_filled` backfill is 100% corner-mined (no easy fills);
     shortfall accepted per C11 option (i) when interp_pool /
     corner_neg_pool exhaust before the target size is reached
     (logged as a warning).
   - [variant_loader.py:_load_em_gold_regenerated](../usecases_synthetic/lib/variant_loader.py)
     returns `dict[pair, dict[split, dict[version, DataFrame]]]`;
     `VariantBundle.em_gold_regenerated` annotation updated to the
     3-level nested shape. Legacy `*_regenerated.csv` files no
     longer recognised (per 2026-05-25 sign-off).
   - [committee_em.py:_load_labelled_split](../usecases_synthetic/lib/committee_em.py)
     accepts `version: str = "corner_filled"`; existing scoring
     reads `corner_filled` to preserve pre-C11 headline F1
     semantics. 4d will explicitly read both versions.
   - 9 new tests (`TestRegenSplitVersionsC11` + `TestVariantLoaderRegenVersions`)
     cover the Set 1 ⊂ Set 2 invariant, size preservation when
     pools cover, undersize accepted when corner pool dry,
     easy-backfill exclusion, K2/K3/K4 id-existence pruning,
     positive-ratio preservation, and loader version dispatch.
     4 existing regen tests updated to filter by version.
   - **Total: 231 tests pass** across knob_02, committee_em,
     committee_fusion, silver, silver_targets, protection,
     generate_variant.
4d. **EM committee scoring rewrite** (C10, depends on 4c) —
    **[done 2026-05-25]**:
   - [committee_em.py:_score_predictions](../usecases_synthetic/lib/committee_em.py)
     rewritten to compute `f1_baseline_test`
     (closed-set on `em_test_baseline_pruned.csv`) and
     `f1_regen_test` (closed-set on `em_test_corner_filled.csv`).
     Open-set surface (`f1_vs_test_gold` + precision/recall) dropped
     from per-pair metrics; the `gold_df` argument is retained but
     unused. Headline `f1` fallback chain:
     `regen_test → baseline_test → pool`. `regen_val_corner` loaded
     and scored internally; on `abs(val_f1 − test_f1) > 0.15` a
     debug log fires, but val is not surfaced in the public dict.
   - `EMMatchingCommitteeRunner.run` updated symmetrically — both
     test versions loaded, matcher runs once against
     `corner_filled` candidates, predictions scored on both
     versions via the closed-set scorer (Set 1 ⊂ Set 2 means no
     second matcher pass). When `corner_filled` is absent the
     runner falls back to running against `baseline_pruned`
     directly. Aggregated keys `macro_f1_baseline_test` +
     `macro_f1_regen_test`; retired `macro_f1_vs_test` /
     `macro_f1_vs_val`.
   - `_MATCHER_AVG_KEYS` + `_compute_aggregated` in
     [committee_em.py](../usecases_synthetic/lib/committee_em.py)
     swapped to the C10 keys.
   - [knob_expected_signals.yaml](../usecases_synthetic/config/knob_expected_signals.yaml):
     all 9 EM monotonicity expectations switched from
     `aggregated.macro_f1_vs_pool` to `aggregated.macro_f1_regen_test`;
     header comment rewritten to describe the C10 closed-set
     surfaces.
   - [monotonicity.py](../usecases_synthetic/lib/monotonicity.py):
     `_BEST_MEMBER_METRIC` for `em_matching` + `em` now uses
     `("f1_regen_test", "f1")` so best-member ceiling reporting
     reads the new key.
   - [validate_variant.py](../usecases_synthetic/scripts/validate_variant.py):
     per-pair CSV writer + summary + per-member tables + log line
     all updated to `f1_baseline_test` / `f1_regen_test` /
     `macro_f1_regen_test`. [build_statistics.py](../usecases_synthetic/scripts/build_statistics.py)
     headline metric for `em_matching` switched to
     `macro_f1_regen_test`.
   - Baseline EM test sets sample-checked: music / games / companies
     / products all carry labeled negatives (no positive-only
     domains; no recall-style degeneracy on `f1_baseline_test`).
   - `score_em_correspondences` deleted from
     [committee_em_scoring.py](../usecases_synthetic/lib/committee_em_scoring.py);
     unused import removed from
     [_tune_em_matching_committee.py](../usecases_synthetic/scripts/_tune_em_matching_committee.py);
     `TestScoreEMCorrespondences` test class removed (5 tests).
   - 4 new tests in `TestEMScoreFallbackChain` (covering
     `regen_test → baseline_test → pool` fallback chain + retired
     macro keys absence); existing val/test fallback tests
     rewritten against the C11 3-level nested
     `em_gold_regenerated` shape. All 47 committee_em tests pass;
     full 276-test C10-touched suite green.
4e. **Committee restructure — fusion + normalization** (C12,
    independent of 4c/4d but lands together for the same rerun):

   **LLM prompt v2 landed 2026-05-25** (sub-piece of 4e):
   - [lib/llm_judge_fusion.py](../usecases_synthetic/lib/llm_judge_fusion.py)
     prompt v2: removes verbatim constraint, JSON output with
     `{value, operation, confidence, reasoning}`, operation set
     `{verbatim_pick, aggregation, union, intersection, normalization,
     interpolation}`. Default `prompt_version="v2"` invalidates
     [cache/llm_judge_fusion/](../usecases_synthetic/cache/llm_judge_fusion/)
     by construction. New `op_log_path` kwarg appends rows to a CSV
     per non-trivial call.
   - [lib/llm_normalizer.py](../usecases_synthetic/lib/llm_normalizer.py)
     prompt v2: symmetric — type-aware operation choice over
     `{vocab_canonicalize, date_normalize, numeric_normalize,
     categorical_map, synthesize, abstain}`, synthesis allowed.
     `PROMPT_VERSION_V1` → `PROMPT_VERSION_V2`; invalidates
     [cache/llm_normalizer/](../usecases_synthetic/cache/llm_normalizer/).
     `op_log_path` constructor param mirrors the fusion side.
   - Tests: 14 v2-migrated `TestLLMJudge` + 22 new
     [tests/test_llm_normalizer.py](../usecases_synthetic/tests/test_llm_normalizer.py)
     pass; full 94-test impacted suite green (signatures stayed
     backward-compatible via keyword-only `op_log_path`).

   **Remaining sub-pieces of 4e:**
   - **Schema**: new YAML schema for both committees — top-level
     `members:` list, per-member meta-strategy or single-method
     declaration, per-member attribute-type fallback rules.
   - **Fusion runner rewrite** ([lib/committee_fusion.py](../usecases_synthetic/lib/committee_fusion.py)):
     one `DataFusionEngine.run()` per member; member produces one
     fused DataFrame; score; record per-member `macro_accuracy` +
     per-attribute accuracies. Retire the per-(attr, strat) inner
     loop. Implement the 9-member roster (`pydi_per_attribute_optimal`,
     `llm_only`, `fusionquery_only`, `truthfinder_only`, `ltm_only`,
     `casefusion_only`, `accusim_only`, `voting_only`,
     `prefer_higher_trust_only`).
   - **Normalization runner rewrite** ([lib/committee_norm.py](../usecases_synthetic/lib/committee_norm.py)):
     coherent members covering every attribute via internal routing.
     Implement the 3-member roster (`rule_per_attribute_optimal`,
     `llm_only`, `passthrough`).
   - **Val-selection plumbing**: one val-evaluation pass per member
     per fallback attribute; cache per-(member, attribute) → method
     choices in `baselines/<domain>/{fusion,norm}_committee_selection.json`.
   - **Prompt v2 for LLM judge** — **[done 2026-05-25]** (see "LLM
     prompt v2 landed" block above).
   - **Prompt v2 for LLM normalizer** — **[done 2026-05-25]** (see
     same block above).
   - **build_statistics.py**: surface per-member `macro_accuracy` /
     `macro_f1` as the meaningful headline; add per-member selection
     map column.
   - **analyze_monotonicity.py**: per-member macro tracked across
     levels; best-member macro is the user-attainable ceiling per
     stage.
   - **Tests** for both runners: roster parsing, per-member
     end-to-end fused/normalized output covers every attribute,
     val-selection caching, prompt v2 schema parse.
4f. **K1 realised-intensity audit** (G9, addresses K1 verification
    gap — symmetric to C3's K5/K8/K10 instrumentation) —
    **[done 2026-05-26]**:
   - [apply_knob_01_surface.py](../usecases_synthetic/scripts/apply_knob_01_surface.py)
     extended to emit `output/baselines/knob_01_realised.csv` per
     level. Schema (`REALISED_COLUMNS`):
     `level, paraphrase_attempts, paraphrase_committed,
     mean_edit_distance, mean_token_jaccard_drop,
     strict_cache_miss_count`. New helpers `_token_jaccard_drop`
     (whitespace-token set Jaccard — catches shallow rewrites where
     edit-distance is high but token-set is unchanged, e.g.
     `eda_random_swap`) and `build_realised_df` (computes
     `attempts = committed + skipped`; pulls `_levenshtein_ratio`
     from `niche_metrics` for edit-distance). `apply_knob_01`
     return signature widened to a 4-tuple
     `(paraphrased, provenance_df, skipped_df, realised_df)`;
     `write_outputs` gains a `realised_df: pd.DataFrame | None = None`
     kwarg that lands the CSV at
     `output/baselines/knob_01_realised.csv` when supplied.
     [apply_values_joint.py:278](../usecases_synthetic/scripts/apply_values_joint.py#L278)
     threads `realised_k1` through to `write_outputs_k1(...,
     realised_df=realised_k1)`.
   - [generate_variant.py](../usecases_synthetic/scripts/generate_variant.py)
     gains `_k1_realised_metrics` reader (parallel to
     `_k10_realised_swap_rate`); `check_monotonicity` emits two new
     audit rows right after the K3-nesting check:
     - `knob_01_realised_rate_monotonicity`: PASS iff
       `paraphrase_committed easy <= medium <= hard`. Detail
       carries `strict_cache_miss_count` per level + the attempts
       count so dormancy (K1 mirroring K2's strict_cache failure
       mode) is one inspection step away when the row FAILs.
     - `knob_01_realised_intensity_monotonicity`: PASS iff BOTH
       `mean_edit_distance` AND `mean_token_jaccard_drop` are
       monotone non-decreasing. Detail carries `edit_ok=...
       jaccard_ok=...` plus the jaccard_drop values so the
       intensity-dimension FAIL ("edit grew but jaccard didn't,
       so the LLM produced word-substitutions that preserved
       token sets") is distinguishable from "neither grew".
     Missing-file fallback adds explicit FAIL rows naming the
     levels with missing artifacts (parallels the K2
     missing-file path).
   - [package_variant.py](../usecases_synthetic/scripts/package_variant.py)
     `BASELINE_FILES` gains `knob_01_realised.csv` so the
     work_dir → variant_dir packaging propagates the new artifact.
     Also added `knob_10_realised.csv` to the same list — that was
     a latent step-1 omission that silently disabled the
     `knob_10_realised_rate_monotonicity` audit row in real runs
     (the row only fires when all three level CSVs are present;
     without packaging the K10 audit was always being skipped on
     real variants).
   - [analyze_monotonicity.py](../usecases_synthetic/scripts/analyze_monotonicity.py)
     **not touched** — that script's SignalCheck system reads the
     YAML expectations file (`knob_expected_signals.yaml`), not
     the `monotonicity_report.csv` audit rows. C3 K5/K8/K10
     instrumentation (step 1) follows the same convention: the
     realised-audit rows live exclusively in `monotonicity_report.csv`.
     The plan note specifying both files was a spec slip; the
     landed pattern is generate_variant-only.
   - 11 new tests in
     [test_knob_01.py:TestRealisedAudit](../usecases_synthetic/tests/test_knob_01.py)
     covering `_token_jaccard_drop` corner cases (identical,
     reorder-is-zero, disjoint, one-empty, both-empty, partial
     overlap), `build_realised_df` (no commits, attempts =
     committed + skipped, strict_cache_miss tallied), the new
     `apply_knob_01` 4-tuple shape, and `write_outputs` with /
     without `realised_df`.
   - 11 new tests in
     [test_generate_variant.py:TestK1RealisedMetricsReader](../usecases_synthetic/tests/test_generate_variant.py)
     + `TestK1AuditRowsInCheckMonotonicity` covering the reader
     (missing file, empty CSV, full row, missing column),
     monotone-passes-both, decreasing-committed-fails-rate,
     flat-committed-allowed (with strict_cache_miss surfaced in
     detail), shallow-paraphrase, inverted edit-distance,
     inverted jaccard-drop, and the missing-CSV fallback.
   - `_seed_variant` fixture extended with `k01_*` kwargs;
     `joint_stub` updated to write `knob_01_realised.csv` so
     `test_all_three_levels_then_monotonicity` exercises the new
     audit row end-to-end. All 1299 tests pass across the
     synthetic-pipeline suite (7 unrelated pre-existing skips).
   - **Next step (gated on the LLM-spend authorisation that
     pre-populates the K2 cache):** re-run R7.2 on the existing
     music + games variants — no full regen needed because the
     new audit row only reads `output/baselines/knob_01_realised.csv`
     and that CSV will be emitted at variant-generation time
     after the rerun. Findings feed back into Step 7
     verification (G9's "K1 dormant / shallow / working"
     verdicts).
4g. **Diagnose games EM committee baseline gap** (G10) —
    **[landed 2026-05-26]**. Findings:
    [plan_revision_step4g_findings.md](plan_revision_step4g_findings.md).
    Primary cause: silent loader bug — `_load_em_gold` only tried the
    declared `<src1>_2_<src2>_*.csv` direction, so the games
    `(metacritic, dbpedia)` pair was dropped (the test file lives at
    `dbpedia_2_metacritic_test.csv` — reverse direction). Secondary:
    no platform-alias value normalization before matching (notebook
    applies one). Tertiary: no per-pair threshold tuning + no
    rule-based matcher in the committee roster. The §1 loader fix
    landed in
    [lib/variant_loader.py:_load_em_gold](../usecases_synthetic/lib/variant_loader.py#L313-L373)
    along with direction-tolerant lookup; the §2 normalization is
    intentionally left out of the committee (per user 2026-05-26 —
    we want the performance drop *without* normalization measurable
    at the committee level), but the pool builder's downstream Ditto
    stream still benefits from it indirectly via the human-baseline
    correspondences stream. The subsequent R6 cascade (committee
    Ditto re-baselining, see below) supersedes the F1-target portion
    of this item.
4h. **Pre-rerun knob tuning review** — **landed 2026-05-27**
    (process check, last gate before any full domain rerun;
    walkthrough captured in
    [plan_revision_step4h_knob_review.md](plan_revision_step4h_knob_review.md)
    + decisions applied to all 4 active domains' knob YAMLs):
   - Walk through every per-domain knob YAML across the 8 active
     knobs ([config/knob_01_surface/](../usecases_synthetic/config/knob_01_surface/),
     [knob_02_niche/](../usecases_synthetic/config/knob_02_niche/),
     [knob_03_drop/](../usecases_synthetic/config/knob_03_drop/),
     [knob_04_coverage/](../usecases_synthetic/config/knob_04_coverage/),
     [knob_05_format/](../usecases_synthetic/config/knob_05_format/),
     [knob_06_noise/](../usecases_synthetic/config/knob_06_noise/),
     [knob_08_naming/](../usecases_synthetic/config/knob_08_naming/),
     [knob_10_reliability/](../usecases_synthetic/config/knob_10_reliability/))
     × 4 active domains (music / games / products / companies).
   - Per (knob, domain) verify:
     - **Per-level parameters monotone in the correct direction**
       (easy → medium → hard, accounting for sign — e.g. K10
       `winner_share` decreases with difficulty because lower share
       = more dispersion; K1 paraphrase rate increases with
       difficulty).
     - **Configured value sized to produce *realised*
       differentiation** across levels — cross-reference step 1's
       K2 realised CSV + step 4f's new K1 realised CSV + the
       K5/K8/K10 intensity audits. If a knob's natural baseline
       overshoots / undershoots the configured easy target (per
       G1 / G3 / G4 / G5), revise.
     - **LLM model_ids consistent with
       `feedback_synth_llm_and_naming` memory** — gpt-5.4-mini for
       K1/K2/K4 unless explicitly bumped. R2 migration (products
       from claude-opus-4-6 → gpt-5.4-mini) landed this session.
     - **K8 anonymized rung lands at hard** (C5 directive — full
       anonymize intentional, "string-only dies at hard").
     - **Per-attribute scope synced with R1's products schema
       upgrade** — K8 rename_table, K5 format rules, K10 corruption
       profiles, K3 drop targets must cover the new product columns
       (`title_description`, `model`, `model_number`, `product_type`,
       `chipset_name`, `vram_gb`, `storage_gb`).
     - **`id_columns` convention** matches the
       `feedback_synth_id_columns_convention` memory: every
       per-knob YAML uses `id_columns: {<source>: id}` (loader
       renames every primary id to "id"; raw CSV column names are
       wrong and silently degrade dispatchers).
   - Document findings in
     `plans/plan_revision_step4h_knob_review.md` — one section per
     (knob, domain) with a verdict (PASS / REVISE) and reason. Any
     REVISE entry includes the proposed new value.
   - **User gate** before any YAML edit — review the proposed
     change set, sign off, then Claude applies the YAMLs and reruns
     unit tests. Step 5 gated on a green test suite.
   - **Couples with G1 (K2) / G9 (K1) / R2 (LLM model_id) / R1
     (products schema)** — drift surfaced by those items gets
     corrected here so step 5 reruns against a fully-tuned config.
4i. **K2 drop-corner-touching operator + non-corner refill** —
    **landed 2026-05-28** (code authorised under step 4h K2 overall
    walkthrough 2026-05-27, implemented + tested 2026-05-28).

   **Why this exists.** Per the F7 finding and the legacy K2 dispatch
   logic, K2 noop'd when `baseline_corner_ratio > target_ratio + tol`
   (operator decision `noop_baseline_above_target`). The legacy
   `drop_high_density` operator dropped least-corner entities and
   shrank the denominator faster than the numerator — moving the
   realised ratio *up*, not down (music-small easy: baseline 0.2375
   → realised 0.266 after 9 825 removals). Without a real "drop
   corner-touching entities" operator the easy target was
   unenforceable on domains where baseline > easy target (music
   baseline 0.257 vs target 0.20; games baseline 0.67 vs target 0.20
   and 0.50).

   The user's step-4h directive (2026-05-27): K2 must be a
   bidirectional dial — realised ratio strictly lower at easy than
   medium than hard, even if the perfect target isn't reached.
   Configuration-only fixes (raising easy targets above baseline
   per-domain) were rejected in favour of building the missing
   operator.

   **New K2 dispatch branch** at `baseline_ratio > target_ratio + tol`:
   `operator_decision = "drop_corner_touching_refilled"`.

   **Operator design.**

   1. *Score candidate drops by expected corner-pair reduction.*
      For each canonical entity, count the corner pairs it
      participates in (`same_pairs + cross_pairs ∩ corner_pairs`)
      relative to its total pair count. High ratio = "dropping this
      entity reduces corner pairs most without proportionally
      reducing total pairs." Sort candidates descending.
   2. *Greedy pick until realised ratio crosses target.* For each
      candidate:
      - **Skip protected** (fusion val/test gold + EM gold positives,
        same `protection_flags` array the existing size-invariant
        drop consults at [apply_knob_02_niche.py:1086-1092](../usecases_synthetic/scripts/apply_knob_02_niche.py#L1086-L1092)).
        Couples with C9 — under `--protection-source silver` the
        protected set expands to every pool-cluster member.
      - **Skip last member of a label-collision group** (dropping
        would collapse the group; the group's entire corner-pair
        contribution would vanish, overshooting the target).
      - Recompute realised ratio after the candidate's pairs are
        removed. Stop when realised ratio ≤ `target_ratio + tol`
        OR all eligible candidates exhausted (cap at
        `max_interp_fraction × n_entities` for symmetry with the
        interpolation upper bound).
   3. *Refill 1-for-1 with synthetic non-corner entities* (size
      invariant preserved per the user 2026-05-27 directive — the
      canonical set size stays at the original n_entities). For
      each drop, generate a synthetic non-corner entity via LLM
      prompt `non_corner_v1.txt`:
      - **Prompt**: "Generate a fictional entity that is **dissimilar**
        to the K reference entities below: distinct semantic niche,
        distinct primary label, attribute values outside the
        reference distribution. Domain: [music/games/companies/
        products]." Mirrors the K2 prompt v2 domain enumeration +
        domain-extension maintenance note.
      - **Reference set**: K=5 lowest-density entities from the
        remaining canonical set (after the drop). These define the
        "what to be dissimilar from" anchor.
      - **Contamination guard**: synthesized entity's normalised
        primary label must not exist in `label_collision_index`
        across the canonical set (would re-introduce a label
        collision and defeat the drop).
      - **Cache namespace**: `cache/knob_02_non_corner/<domain>/`
        (separate from the existing
        `cache/knob_02_interpolations/<domain>/` so the two
        prompts don't collide on cache keys).
      - **LLM model**: `gpt-5.4-mini` (uniform with K2 interpolate).

   **YAML changes** (all 4 K2 YAMLs):
   ```yaml
   non_corner_refill:
     enabled: true
     reference_count: 5            # K reference entities for dissimilarity anchor
     contamination_check:
       label_collision_block: true # synthesized label must not exist in current canonical set
   non_corner_prompt_version: v1
   ```

   **Realised audit metrics** (new fields on the existing
   `knob_02_realised.csv` row, not separate CSVs — the implementation
   went with the simpler additive approach):
   - `drop_corner_planned`: per-level realised drop count.
   - `drop_corner_simulated_final_ratio`: ratio after the greedy
     drop loop simulates the removals.
   - `drop_corner_cap_bound`: bool — True iff the loop exited via the
     `max_interp_fraction × n_entities` cap.
   - `non_corner_refill_attempts`: number of refill attempts (= drop
     count when refill enabled).
   - `non_corner_refill_committed`: how many refills survived
     contamination / non-empty-primary / strict_cache guardrails.
   - `non_corner_rejected_*`: per-reason rejection counter prefix
     (`strict_cache_miss`, `empty_primary_label`,
     `contamination_collision_with_real_entity`, `nondict_result`).

   These columns surface in the same `knob_02_realised.csv` that
   already carries `level`, `baseline_ratio`, `target_ratio`,
   `final_ratio`, `operator`, `removed`, `interpolated`. Downstream
   `analyze_monotonicity.py` audit rows can be added in a follow-up
   if a structured PASS/FAIL signal is needed; not blocking step 5.

   **Files in scope (what actually landed 2026-05-28)**:
   - [scripts/apply_knob_02_niche.py](../usecases_synthetic/scripts/apply_knob_02_niche.py):
     new dispatch branch + `_run_drop_corner_refill` orchestrator;
     `apply_knob_02` signature widened with `non_corner_cache` +
     `non_corner_api_client` kwargs; provenance rows for
     `llm_non_corner_refill` paired with the drops; `k2_metrics`
     extended with the fields above; CLI `main()` wires the new
     cache for standalone invocations.
   - New: [lib/non_corner_refill.py](../usecases_synthetic/lib/non_corner_refill.py)
     — `NonCornerEntity` dataclass + `refill_non_corner_entity` +
     `select_reference_anchor` + `contamination_check` +
     `reference_anchor_hash` + `default_api_client_from_attributes`
     fallback for tests.
   - New: [config/knob_02_niche/_prompts/non_corner_v1.txt](../usecases_synthetic/config/knob_02_niche/_prompts/non_corner_v1.txt)
     — "be dissimilar to" framing with the 4-domain enumeration +
     maintenance note.
   - [scripts/generate_variant.py](../usecases_synthetic/scripts/generate_variant.py):
     new `llm_cache_k2_non_corner` instance pointing at
     `cache/knob_02_non_corner/<domain>/`; threaded through
     `_run_knob_02` → `apply_knob_02`.
   - All four [config/knob_02_niche/{music,games,companies,products}.yaml](../usecases_synthetic/config/knob_02_niche/):
     `non_corner_refill` + `non_corner_prompt_version` blocks added.
   - New: [tests/test_non_corner_refill.py](../usecases_synthetic/tests/test_non_corner_refill.py)
     — 28 tests across `TestSelectReferenceAnchor` (5),
     `TestContaminationCheck` (4), `TestReferenceAnchorHash` (3),
     `TestDefaultApiClient` (3), `TestRefillNonCornerEntity` (8),
     `TestRunDropCornerRefill` (5). Full 1394-test synthetic suite
     green.

   **Deferred (follow-up, not blocking step 5)**:
   - `package_variant.py BASELINE_FILES` expansion to declare the
     new K2 telemetry columns explicitly — currently they ride on
     the existing `knob_02_realised.csv`.
   - `generate_variant.py` audit rows in `monotonicity_report.csv`
     for `knob_02_drop_corner_realised_rate` + `knob_02_non_corner_refill_size_invariant`
     — useful for the analyze_monotonicity PASS/FAIL signal but the
     fields are already inspectable on the realised CSV.

   **Couples with**:
   - **C7** (LLM spend authorisation): drop-corner + refill at easy on
     music + games adds ~`(realised_baseline − target) × n_pairs / k_drop_reduction`
     LLM calls. Music easy needs to drop ~0.05 of ~9k = 450 entities;
     games easy needs to drop ~0.47 of ~80k = ~37 600 entities.
     Caps at `max_interp_fraction × n_entities` (0.60). Worst case ~$10-30
     additional spend. Bundles with the existing C7 authorisation
     request.
   - **G1** (K2 dormancy): orthogonal — G1 was about strict_cache at
     hard suppressing interpolation. Step 4i adds the missing
     operator on the *other side* of the dial (baseline > target).
     Both fixes land before step 5.
   - **C9** (silver-standard protection): the drop-corner operator
     consults the same `protection_flags` array as the existing
     size-invariant interpolation drop, so `--protection-source silver`
     widens the protection set for drops symmetrically.
   - **K2 overall step-4h decisions**: `placement_split: 0.6 → 0.5`,
     `interp_pair_factor: 0.05` unchanged, prompt v2, hard_negative_gate
     option (a) full LLM via new `gate_mode: full_llm`. All five land
     together with step 4i as a coherent K2 update.

   **Actual scope landed 2026-05-28**: ~660 lines of new + modified
   code (lib/non_corner_refill.py ~310 LOC, apply_knob_02_niche.py +
   ~250 LOC for the helper, generate_variant.py + ~35 LOC for the
   cache wiring), 28 new tests, full 1394-test synthetic suite green.
   Time to write + verify: ~3 hours wall-clock. Blocks step 5 was
   correct in spirit — products rerun now exercises a real
   bidirectional K2 dial; previously easy targets below baseline
   would noop on music + games. Cap-bound semantics work for products
   too (small canonical set + 0.60 cap = ~1 807 max drops at easy,
   sufficient for the 0.44 baseline → 0.20 target gap).

4j. **Pre-rerun variant-output + stale-HPO cleanup** —
    **landed 2026-05-28** (housekeeping, authorised under step 4h
    2026-05-27, executed at step-5 launch time): user signed off on
    mandatory + optional small-augmented at execution time; both
    swept (~497 MB freed; see Files-in-scope block below for the
    exact delete list). Step 5 now starts from a clean per-variant
    + per-tuning slate. **Not a "cache-warming" step** — the
    variant generation itself populates LLM caches as it runs;
    there's no separate warming pass.

   **Cache-reuse contract** (from
   [lib/llm_cache.py:6](../usecases_synthetic/lib/llm_cache.py#L6)):
   cache key = `sha256(source|attribute|value|prompt_version|model_id)`.
   With the YAML changes landed (K2 prompt v2, K2 non-corner v1
   prompt, hard_negative_gate v1 adjudicator prompt, `model_id`
   locked to `gpt-5.4-mini`), identical inputs produce identical
   cache hits on every subsequent run. The first variant generation
   per domain populates the cache for that domain; subsequent reruns
   of the same domain hit cache. Per-domain first-run cost estimates
   (cumulative across all LLM-using knobs): products ~$35-80, music
   ~$80-180, games ~$150-300, companies ~$30-50. Subsequent reruns
   of the same domain ~$5-30.

   **Currently no LLM-call caches exist on disk** (the
   `usecases_synthetic/cache/` tree contains only model artifacts +
   tuning JSONs). The "cleanup" required is per-variant outputs +
   pre-C4 / pre-C12 HPO trees.

   **DELETE before step 5 (stale per-variant outputs)**:
   - `usecases/music-augmented/{easy,medium,hard}/` — pre-step-4
     fixes (K2 strict_cache bug, K1 v1 operator pool, old C10/C11
     EM regen shape). Regenerated by next variant pass.
   - `usecases/games-augmented/{easy,medium,hard}/` — same.
   - `usecases/companies-augmented/{easy,medium,hard}/` — same.
   - `usecases/products-augmented/{easy,medium,hard}/` — same.
   - `usecases/{music,games,companies,products}-small-augmented/` —
     pre-step-4 smoke-test variants; clear if no longer needed for
     debugging (confirm at execution time).

   **DELETE before step 5 (stale HPO trees)**:
   - `/cache/em_blocking_tuning/` — pre-C4 word-BM25 sweeps.
     Invalidated by C4 char-ngram switch.
   - `/cache/em_matching_tuning/` — pre-C12 committee sweeps.
     Invalidated by C12 coherent-member restructure.
   - `/cache/fusion_tuning/` — same (pre-C12).
   - `/cache/norm_tuning/` — same (pre-C12).

   **KEEP (untouched by step 4 / C4 / C12)**:
   - `/cache/sm_tuning/sweep.json` — one-off SM committee HPO
     (2026-05-08). C12 restructured fusion + norm but **did NOT
     touch the SM committee** — the roster + member shapes are
     unchanged. The per-member SM hyperparameters baked into the SM
     YAMLs trace back to this sweep file.
   - `/cache/ditto_checkpoints/` — model artifacts (R6 + R7
     maintained, post-R7 retrain pending for music/companies/products
     in the post-step-5 cascade).
   - `usecases_synthetic/cache/sc_block_checkpoints/` — model
     artifacts.
   - `usecases_synthetic/cache/ditto_inference/` — inference cache
     keyed on (pair, model_path); correct across runs as long as
     checkpoints stay in place.
   - `usecases_synthetic/cache/magneto_prompts/`.
   - `usecases_synthetic/baselines/<domain>/fusion_silver_standard.*`
     — C9 silver-standard artifacts (current).

   **NEW LLM-cache namespaces that step 4i / step 4h code changes
   create on first run** (no pre-population needed — first sweep
   populates):
   - `usecases_synthetic/cache/knob_01_paraphrases/<domain>/<level>/`
     (K1 unchanged prompt v1; new cells get populated this run).
   - `usecases_synthetic/cache/knob_02_interpolations/<domain>/`
     (K2 prompt v2 — all entries fresh).
   - `usecases_synthetic/cache/knob_02_non_corner/<domain>/` (NEW —
     step 4i `non_corner_v1.txt` prompt).
   - `usecases_synthetic/cache/knob_02_hng_adjudicator/<domain>/`
     (NEW — step 4h `gate_mode: full_llm` adjudicator prompt v1).
   - `usecases_synthetic/cache/knob_04_fabrications/<domain>/` (K4
     existing path; first populates this run).
   - `usecases_synthetic/cache/llm_judge_fusion/<domain>/<level>/`
     (C12 prompt v2 — already landed in code; populates per committee
     run).
   - `usecases_synthetic/cache/llm_normalizer/<domain>/<level>/`
     (C12 prompt v2 — same).
   - `usecases_synthetic/cache/pool_builder/<domain>/r3_pool_adjudicator/`
     (bucket-C tightened — populates on next pool rebuild).

   **Files in scope (housekeeping only — no code change)**:
   - The 4 augmented variant directory trees under `usecases/`.
   - The 4 stale HPO trees under `/cache/`.
   - (Optional) the 4 small-domain augmented trees under `usecases/`.

   **Execution gate**: user confirms the final delete list at
   execution time; Claude runs the `rm -rf` commands and emits a
   summary log of what was removed + total disk freed. After this
   step, step 5 starts from a clean per-variant + per-tuning slate.

   > **Authoritative execution order.** The per-domain step 5/6 cascades
   > below are the *detail*; the binding order is R10's "Concrete
   > sequencing for the remaining work" (R10 code gate -> R9 sweeps under
   > R10-I wide scope -> per-domain step-5/6 cascade, products first ->
   > publish). When the two diverge, R10's sequencing wins — update both
   > together. Each per-domain cascade additionally embeds R10-H
   > (baseline retrain) + R10-G phase 2 (variant retrain) and runs under
   > R10-I wide scope.
   >
   > **All four domains use `--protection-source silver`** on every
   > `generate_variant.py` invocation (products / music / games /
   > companies). This is the C13 default, but the CLI default is `gold`,
   > so the flag MUST be passed explicitly. Silver enables the C9/C13
   > intact-cluster *drift* protection (K1/K6 closeness) while K2's
   > drop-*existence* protection stays gold-hardcoded. Requires the
   > per-domain `baselines/<domain>/fusion_silver_standard.*` to be
   > current (rebuilt after any pool rebuild — see step 6).

5. **Products rerun first** (the smoke-test / smallest-domain start —
   directive 2026-05-27):
   - **Rationale**: products is the smallest active domain at ~3 012
     entities (vs music 37k / games 75k). K2's per-level cap binds at
     `max_interp_fraction × n = ~1 807` so LLM-call volume is bounded.
     First-sweep cost estimate ~$35-80; subsequent products reruns
     hit cache and cost ~$5-15. Fast iteration if step 4h/4i surfaces
     issues post-landing.
   - **No pool rebuild required for products**: products uses
     `pool_quality: live` with native `cluster_id` linkage via
     [scripts/build_pool_products.py](../usecases_synthetic/scripts/build_pool_products.py).
     The R6 bucket-C tightening (every disagreement → LLM adjudicator)
     applies only to the PLM + rule-based `build_pool.py` path used by
     music/games/companies — products is orthogonal.
   - **R1 schema swap is largely already applied**: the
     `data_cleaned_final` columns are already in the source JSONs +
     domain config; step 4h cross-knob expansion landed the per-knob
     scope additions. No pre-rerun schema work pending.
   - **R3 sc_block checkpoint** at
     `usecases_synthetic/cache/sc_block_checkpoints/products/best`
     already trained (per the R6-5 audit, 2026-05-22 run dir).
     Prerequisite satisfied.
   - Run products variant generation + initial validation:
     ```bash
     python usecases_synthetic/scripts/generate_variant.py --domain products --level all --protection-source silver
     python usecases_synthetic/scripts/validate_variant.py --domain products --level all --with-llm
     ```
   - Verify the R-1 success criteria at each level (see step 7).

   **Optional follow-up: R7c per-variant retrain (post-rerun)** — once
   the products variant has been generated, the K2 dispatcher writes
   `<pair>_train_corner_filled.csv` files under
   `usecases/products-augmented/<level>/input/entitymatching/`. R7c
   trains a Ditto + sc_block checkpoint per (products, level) on those
   files, enabling the R7b dual-model evaluation surface. ~3-6 hours
   MPS. Triggered AFTER the initial variant run + validation, only if
   the dual-model `variant_model_on_regen_test` signal is wanted for
   products' monotonicity headline. Until R7c lands, validate_variant
   produces the 4-cell matrix with variant cells aliasing baseline.

   - **User gate** between products rerun and music/games reruns — if
     products surfaces issues with the step 4h/4i changes, fix before
     kicking off the larger domains.

6. **Music + games full reruns** — prerequisites + execution:

   **Pool rebuild prerequisite (per R6, gated on R7 decision)**:
   Music + games + companies use the PLM + rule-based
   [scripts/build_pool.py](../usecases_synthetic/scripts/build_pool.py)
   path with `pool_quality: two_system`. The bucket-C policy tightened
   2026-05-26 — every disagreement now goes through the LLM
   adjudicator (legacy `score >= theta + delta` auto-include +
   `score < theta - delta` auto-drop paths removed). Pools were last
   built under the legacy auto-paths, so the silver-standard positives
   that K2's `expanded_positives` consults are stale. Rebuilding under
   the new policy adds ~14 314 LLM adjudicator calls (companies 1818,
   games 8134, music 3378 — ~$1-$10 at gpt-5.4-mini pricing; 1-3 hours
   wall-clock per [pool_builder.py:856-887](../usecases_synthetic/lib/pool_builder.py#L856-L887)
   estimates). Bundle the pool rebuild with this step.

   **Silver-standard rebuild (follows the pool rebuild, precedes variant
   gen).** The materialized
   `baselines/<domain>/fusion_silver_standard.{csv,json}` is built from
   the pool's partner graph
   ([fusion_silver_standard.py:786](../usecases_synthetic/lib/fusion_silver_standard.py#L786)
   `build_pool_clusters`), so a pool rebuild makes it stale. It MUST be
   regenerated via `build_fusion_silver_standard.py` before variant
   generation, because `generate_variant --protection-source silver` (the
   C13 default) reads it as the C9/C13 drift-protection target. This is
   distinct from the live-read `expanded_positives` (the K2 existence
   set), which self-heals from the rebuilt `pooled_positives.csv` and
   needs no materialized rebuild. Companies' silver rebuild is
   additionally gated on R0 (larger fusion gold).

   **R10-H baseline Ditto retrain (mandatory; R7-baseline-finish folded
   into R10-H 2026-05-29 — see the R10-H section).** Music + companies +
   products carry baseline Ditto checkpoints from before the R7 padding
   fix; R10-H retrains them under the padded trainer on the **R10-I wide
   committee-scope** fields. Games is **skipped** (its baseline Ditto was
   already retrained at R7 verification 2026-05-27). The commands below
   are R10-correct: `_prep_<domain>.py --train-source pydi` builds wide
   committee-scope records and `train.py --domain <domain>` sources the
   wide field list per R10-I (a narrow `--fields` is now a hard error).
   Per-domain F1 distortion vs the padded retrain: companies **−0.058**
   (load-bearing), music −0.008, products −0.012 (within noise). Must
   finish before this domain's `measure_baseline.py` so the refreshed
   checkpoint is the one loaded into the BL matcher. ~30-60 min MPS/domain.

   **Execution**:
   ```bash
   # 1. Pool rebuild (gated on R6 follow-up; bundles with this step)
   python usecases_synthetic/scripts/build_pool.py --domain music
   python usecases_synthetic/scripts/build_pool.py --domain games
   # (companies pool rebuild gated on R0 — larger fusion gold)

   # 2. Rebuild the materialized silver standard from the NEW pool
   #    (REQUIRED before step 4 — generate_variant --protection-source
   #    silver reads baselines/<domain>/fusion_silver_standard.* as the
   #    C9/C13 drift target; the old artifact is stale once the pool
   #    changes).
   python usecases_synthetic/scripts/build_fusion_silver_standard.py --domain music
   python usecases_synthetic/scripts/build_fusion_silver_standard.py --domain games
   # (companies silver rebuild gated on R0)

   # 3. R10-H baseline Ditto retrain (mandatory; skip games — already
   #    R7-retrained). --domain sources the R10-I wide committee scope.
   python usecases_synthetic/scripts/ditto/_prep_music.py --train-source pydi
   python usecases_synthetic/scripts/ditto/train.py --domain music
   # (same for products; companies similarly — NOT games)

   # 4. Variant generation + validation per domain (C13: silver default)
   python usecases_synthetic/scripts/generate_variant.py --domain music --level all --protection-source silver
   python usecases_synthetic/scripts/validate_variant.py --domain music --level all --with-llm

   python usecases_synthetic/scripts/generate_variant.py --domain games --level all --protection-source silver
   python usecases_synthetic/scripts/validate_variant.py --domain games --level all --with-llm
   ```

   **First-sweep cost estimates**:
   - music: ~$80-180 (variant generation + committee LLM members)
   - games: ~$150-300 (larger entity count drives K1 + K4 LLM volume)
   - Pool rebuild adds ~$1-10 on top, one-time.

   Subsequent reruns of the same domain hit cache and cost ~$10-30.

   **Optional per-domain follow-up: R7c per-variant retrain** — same
   pattern as products in step 5. After each domain's variant
   generation, R7c trains Ditto + sc_block per (domain, level) on the
   K2-emitted `<pair>_train_corner_filled.csv` to enable the R7b
   dual-model evaluation. **Per-domain decision, not a batched sweep**
   — keyed off the monotonicity signal from the initial validation
   run. ~3-6 hours MPS per domain. Until R7c lands for a given domain,
   the variant cells in `validate_variant.py` alias the baseline
   cells (no dual-model signal but the rerun still completes).
7. **Verify** at each rerun (updated 2026-05-23 to reflect C2 + C5 + G9):
   - K2 realised hits configured target on the domain.
   - **K1 realised-intensity audit PASS** (G9 / step 4f): rate
     monotone `easy < medium < hard` AND intensity (edit distance
     / token Jaccard drop) monotone `easy < medium < hard`. Surfaces
     dormancy + shallow-paraphrase issues as distinct verdicts.
   - Committee macro_f1 monotone over `easy → medium → hard` per
     stage (C2 contract — baseline not part of the slope verdict).
   - `baseline_position_ok` is True for every stage (C2 — baseline
     not harder than medium).
   - K5/K8/K10 intensity audits PASS (legacy raw-count rows demoted
     per C3).
   - 0 silent member collapses **except** K8-anonymize-driven
     collapses at hard, which C5 marks as intentional. The
     `detect_collapses` audit still flags them; verify the R7.3
     narration interprets them correctly rather than treating them
     as a defect.
   - **For games specifically (G10 / step 4g)**: EM committee
     baseline F1 within target tolerance of the human-baseline
     notebook on the same pairs. Tolerance is set once the 4g
     diagnosis lands (initially ≤5pp gap; refined per the
     attribution finding).
8. **Update R7.3 final reports** for products + music + games; refresh
   `statistics/` XLSX after each domain so the central reporting
   stays in sync.
9. **Cascade unblock for companies-FULL**: when the user's larger
   companies fusion gold lands, run the full S.7 ladder against the
   already-fixed pipeline; R7.3 + statistics follow.

### Files in scope

**Already landed under steps 1 + 4 (2026-05-19 / 2026-05-22):**
- [entity_interpolation.py](../usecases_synthetic/lib/entity_interpolation.py) — guardrail logging (C1)
- [monotonicity.py](../usecases_synthetic/lib/monotonicity.py) — `SLOPE_LEVELS` (C2), `compute_ceiling_responsiveness` (C6), `baseline_within_allowed_position` (C2)
- [analyze_monotonicity.py](../usecases_synthetic/scripts/analyze_monotonicity.py) — audit metrics + ceiling-responsiveness + `BasePos` column (C2/C3/C6)
- [apply_knob_10_reliability.py](../usecases_synthetic/scripts/apply_knob_10_reliability.py) — writes `knob_10_realised.csv` (C3 K10)
- [generate_variant.py](../usecases_synthetic/scripts/generate_variant.py) — K5/K8/K10 intensity audits (C3); strict_cache forcings dropped (C1)
- [apply_knob_02_niche.py](../usecases_synthetic/scripts/apply_knob_02_niche.py),
  [apply_knob_01_surface.py](../usecases_synthetic/scripts/apply_knob_01_surface.py)
  — strict_cache opt-in via `--strict-cache` (C1)
- [bm25_blocker.py](../usecases_synthetic/lib/bm25_blocker.py) — `tokenizer` + `ngram_range` params for char-ngram mode (C4; **no new `cngram_bm25_blocker.py` file** — extended in place)
- All five `em_blocking_committee*.yaml` — switched to `tokenizer: char_ngram` (C4)

**Already landed — C9 silver standard (step 4b, 2026-05-23):**
- New: [lib/fusion_silver_standard.py](../usecases_synthetic/lib/fusion_silver_standard.py) — per-domain fusion + normalization stack, `build_silver_standard(domain)`, `canonical_cluster_id`, persistence helpers.
- New: [lib/fusion_silver_targets.py](../usecases_synthetic/lib/fusion_silver_targets.py) — silver-augmented protection targets with gold-wins-per-(member, attribute) merge, `resolve_protection_sources("gold" | "silver")`.
- New: [scripts/build_fusion_silver_standard.py](../usecases_synthetic/scripts/build_fusion_silver_standard.py) — CLI: `--domain {music|games|companies}`, `--sample N`.
- New: `usecases_synthetic/baselines/{music,games,companies}/fusion_silver_standard.{csv,json}` — 4 280 / 8 974 / 1 088 clusters respectively.
- [apply_knob_01_surface.py](../usecases_synthetic/scripts/apply_knob_01_surface.py) — `_ClosenessContext` + `apply_knob_01` gain `protection_source: str = "gold"`.
- [apply_knob_06_noise.py](../usecases_synthetic/scripts/apply_knob_06_noise.py) — same.
- [apply_values_joint.py](../usecases_synthetic/scripts/apply_values_joint.py) — threads `protection_source` to K1 + K6.
- [generate_variant.py](../usecases_synthetic/scripts/generate_variant.py) — adds `--protection-source {gold,silver}` CLI flag (default `gold`).
- New tests: [tests/test_fusion_silver_standard.py](../usecases_synthetic/tests/test_fusion_silver_standard.py) (49 tests) + [tests/test_fusion_silver_targets.py](../usecases_synthetic/tests/test_fusion_silver_targets.py) (12 tests).

**Already landed — C11 EM regen refactor (step 4c, 2026-05-25):**
- [lib/corner_case_miner.py](../usecases_synthetic/lib/corner_case_miner.py) — `regenerate_em_splits` emits both versions with a `version` column; new constants `VERSION_BASELINE_PRUNED`, `VERSION_CORNER_FILLED`, `REGEN_VERSIONS`; easy-fill backfill removed; shortfall accepted per C11 option (i).
- [scripts/apply_knob_02_niche.py](../usecases_synthetic/scripts/apply_knob_02_niche.py) — writer groups by (pair, split, version) and emits `<pair>_<split>_<version>.csv`; legacy `*_regenerated.csv` scrubbed.
- [scripts/generate_variant.py](../usecases_synthetic/scripts/generate_variant.py) — `_rerun_regen_post_k4` mirrors the new file naming.
- [lib/variant_loader.py](../usecases_synthetic/lib/variant_loader.py) — `_load_em_gold_regenerated` returns 3-level nested dict; `VariantBundle.em_gold_regenerated` annotation updated.
- [lib/committee_em.py](../usecases_synthetic/lib/committee_em.py) — `_load_labelled_split` accepts `version: str = "corner_filled"`.
- [tests/test_knob_02.py](../usecases_synthetic/tests/test_knob_02.py) — 9 new tests (`TestRegenSplitVersionsC11` + `TestVariantLoaderRegenVersions`); 4 existing regen tests updated to filter by version.

**Already landed — C10 EM committee scoring (step 4d, 2026-05-25):**
- [committee_em.py](../usecases_synthetic/lib/committee_em.py) — `_score_predictions` + `EMMatchingCommitteeRunner.run` emit `f1_baseline_test` + `f1_regen_test`; `f1_vs_test_gold` and `f1_vs_regenerated_*` retired; `_MATCHER_AVG_KEYS` + `_compute_aggregated` swapped to the new keys; aggregated `macro_f1_baseline_test` + `macro_f1_regen_test`.
- [committee_em_scoring.py](../usecases_synthetic/lib/committee_em_scoring.py) — `score_em_correspondences` (open-set) deleted; module docstring + closed-set scorer docstring rewritten.
- [knob_expected_signals.yaml](../usecases_synthetic/config/knob_expected_signals.yaml) — 9 EM monotonicity targets switched to `aggregated.macro_f1_regen_test`; header rewritten to describe C10 surfaces.
- [monotonicity.py](../usecases_synthetic/lib/monotonicity.py) — `_BEST_MEMBER_METRIC` updated to `("f1_regen_test", "f1")` for `em_matching` + `em`.
- [validate_variant.py](../usecases_synthetic/scripts/validate_variant.py) — per-pair CSV writer, stage summary metric, per-member table extras, log lines all on the new keys.
- [build_statistics.py](../usecases_synthetic/scripts/build_statistics.py) — `em_matching` headline = `macro_f1_regen_test`.
- [_tune_em_matching_committee.py](../usecases_synthetic/scripts/_tune_em_matching_committee.py) — dropped unused `score_em_correspondences` import.
- [tests/test_committee_em.py](../usecases_synthetic/tests/test_committee_em.py) — 4 new `TestEMScoreFallbackChain` tests, 5 retired open-set tests removed, fixtures upgraded to the C11 3-level nested shape.

**Already landed — C12 LLM prompt v2 (step 4e sub-piece, 2026-05-25):**
- [llm_judge_fusion.py](../usecases_synthetic/lib/llm_judge_fusion.py) — prompt v2, `VALID_OPERATIONS_V2`, JSON output schema, `_append_op_log` writer, `op_log_path` kwarg on `llm_judge`; default `prompt_version="v2"` invalidates cache.
- [llm_normalizer.py](../usecases_synthetic/lib/llm_normalizer.py) — prompt v2, `PROMPT_VERSION_V2`, `VALID_NORM_OPERATIONS_V2`, JSON output schema, `_append_op_log` writer, `op_log_path` constructor param on `LLMCanonicalizer`.
- [tests/test_fusion_td_adapters.py](../usecases_synthetic/tests/test_fusion_td_adapters.py) — `TestLLMJudge` migrated to v2 schema (14 tests; +2 new: `test_synthesis_allowed_under_v2`, `test_op_log_appended_when_path_set`).
- New: [tests/test_llm_normalizer.py](../usecases_synthetic/tests/test_llm_normalizer.py) — 22 tests covering parser, end-to-end normalize, cache hits, op-log behavior, prompt-version cache invalidation, garbage-response handling.

**Already landed — C12 committee runner restructure (step 4e completion, 2026-05-26):**
- New: [committee_fusion_c12.py](../usecases_synthetic/lib/committee_fusion_c12.py) — 9-member coherent roster + val-selection plumbing
- New: [committee_norm_c12.py](../usecases_synthetic/lib/committee_norm_c12.py) — 3-member coherent roster + val-selection plumbing
- [committee_norm.py](../usecases_synthetic/lib/committee_norm.py) — `__new__` dispatcher mirroring fusion's
- All four `fusion_committee_<domain>.yaml` migrated to C12 `members:` schema
- All four `normalization_committee_<domain>.yaml` migrated to C12 `members:` schema
- New: `usecases_synthetic/baselines/<domain>/{fusion,norm}_committee_selection.json` — val-selection cache
- [build_statistics.py](../usecases_synthetic/scripts/build_statistics.py) — new `selection_map` XLSX sheet for C12 members
- [analyze_monotonicity.py](../usecases_synthetic/scripts/analyze_monotonicity.py) — already reads C12 aggregated keys (`aggregated.macro_f1` for norm, `aggregated.overall_accuracy` for fusion)
- 21 new tests in `tests/test_committee_norm_c12.py`; 359-test C12-touched suite green.

**Already landed — G9 K1 audit (step 4f, 2026-05-26):**
- [apply_knob_01_surface.py](../usecases_synthetic/scripts/apply_knob_01_surface.py) — `REALISED_COLUMNS` constant; helpers `_token_jaccard_drop` + `build_realised_df`; `apply_knob_01` widened to 4-tuple return; `write_outputs` gains `realised_df` kwarg landing `output/baselines/knob_01_realised.csv`.
- [apply_values_joint.py](../usecases_synthetic/scripts/apply_values_joint.py) — threads `realised_k1` to `write_outputs_k1`.
- [generate_variant.py](../usecases_synthetic/scripts/generate_variant.py) — `_k1_realised_metrics` reader; `check_monotonicity` emits `knob_01_realised_rate_monotonicity` (committed monotone non-decreasing, FAIL surfaces cache-miss dormancy) + `knob_01_realised_intensity_monotonicity` (edit_distance AND token_jaccard_drop monotone, FAIL surfaces shallow paraphrases). Missing-file fallback adds explicit FAIL rows.
- [package_variant.py](../usecases_synthetic/scripts/package_variant.py) — `BASELINE_FILES` gains `knob_01_realised.csv` + `knob_10_realised.csv` (the latter was a latent step-1 omission).
- [tests/test_knob_01.py](../usecases_synthetic/tests/test_knob_01.py) — 11 new `TestRealisedAudit` tests covering helper corner cases + 4-tuple shape + write_outputs behaviour; 19 existing call sites updated to the 4-tuple unpack.
- [tests/test_generate_variant.py](../usecases_synthetic/tests/test_generate_variant.py) — 11 new tests across `TestK1RealisedMetricsReader` + `TestK1AuditRowsInCheckMonotonicity`; `_seed_variant` fixture + `joint_stub` extended to emit the new realised CSV.
- analyze_monotonicity.py NOT touched — that script reads the YAML expectations file, not `monotonicity_report.csv` rows (C3 K5/K8/K10 follows the same convention).

**Already landed — G10 games EM diagnosis (step 4g, 2026-05-26):**
- Investigation only; findings recorded in `plans/plan_revision_step4g_findings.md`. The §1 loader fix landed in [lib/variant_loader.py:_load_em_gold](../usecases_synthetic/lib/variant_loader.py#L313-L373) along with direction-tolerant lookup; subsequent R6 cascade (committee Ditto re-baselining) supersedes the F1-target portion of this item.

**Already landed — pre-rerun knob tuning review (step 4h, 2026-05-27):**
- All per-domain knob YAMLs across the 8 active knobs ×
  4 active domains were walked end-to-end and edited per the user's
  sign-off. Decisions captured in
  [plans/plan_revision_step4h_knob_review.md](plan_revision_step4h_knob_review.md).
  Per-knob landed changes:
  - K1: Option B rates + LLM-at-medium + bumped-hard operator_mix +
    dormant-easy comment (4 YAMLs).
  - K2 overall: `placement_split: 0.6 → 0.5`, prompt v1 → v2,
    `gate_mode: full_llm` per-level + top-level (4 YAMLs); per-domain
    target_corner_case_ratio normalised to 0.20/0.50/0.80; companies
    `interp_pair_factor: 0.05 → 0.10`; products per-level extras
    simplified + canonical_schema gains `priceCurrency` + `title_description`.
  - K3: products per_source_attribute_overrides extended for the 4
    sparse R1 attrs + 6 cross-knob extension columns.
  - K4: explicit medium histogram (option β conservative singletons)
    + ramped `within_source_duplicate_rate medium 0.0 → 0.01` +
    paraphrase_only YAML comment (4 YAMLs).
  - K5: products new `rate` class for `read_speed_mb_s` +
    `write_speed_mb_s` (1 YAML) + `rate` group added to
    [unit_factors.yaml](../usecases_synthetic/config/knob_05_format/_tables/unit_factors.yaml).
  - K6: products cross-knob expansion (3 new taxonomy CSVs at
    [usecases/products/input/schemamatching/](../usecases/products/input/schemamatching/)
    + 7 categorical/numeric extension columns into scope across
    K3/K6/K8/K10) + `numeric_attributes` block added +
    `taxonomy_walk` operator added to medium/hard.
  - K8: products rename_table extended +7 attrs × 4 rungs + YAML
    anchor refactor (290 → 153 lines).
  - K10: products `attribute_targets` extended +10 entries using the
    existing `*target_distribution` anchor; companies country/founded
    memory-override comment added.
- New: products domain config (
  [config/domains/products.yaml](../usecases_synthetic/config/domains/products.yaml))
  `attribute_classes` extended +7 (the cross-knob extension columns).
- New file: K2 prompt v2 at
  [config/knob_02_niche/_prompts/interpolate_v2.txt](../usecases_synthetic/config/knob_02_niche/_prompts/interpolate_v2.txt).
- Code changes: K5 `rate` class registration in dispatcher
  ([apply_knob_05_format.py](../usecases_synthetic/scripts/apply_knob_05_format.py));
  K6 `gate_mode` field on `HardNegativePolicy` +
  `apply_hard_negative_policy` full-LLM dispatch branch
  ([lib/corner_case_miner.py](../usecases_synthetic/lib/corner_case_miner.py));
  K6 `gate_mode` wired in `_build_hard_negative_policy`
  ([generate_variant.py](../usecases_synthetic/scripts/generate_variant.py)).
- K1+K6 cleanup overlap correction (the 2 K1 normalize-down additions
  proposed during the K1 walk were DROPPED; K6 cleanup_rules already
  cover those patterns surgically).
- Full 1366-test synthetic suite green post-step-4h.

**Already landed — step 4i drop-corner + non-corner refill (2026-05-28):**
- New: [lib/non_corner_refill.py](../usecases_synthetic/lib/non_corner_refill.py)
  (`NonCornerEntity`, `refill_non_corner_entity`,
  `select_reference_anchor`, `contamination_check`,
  `reference_anchor_hash`, `default_api_client_from_attributes`).
- [scripts/apply_knob_02_niche.py](../usecases_synthetic/scripts/apply_knob_02_niche.py):
  new dispatch branch + `_run_drop_corner_refill` orchestrator;
  `apply_knob_02` signature widened with `non_corner_cache` +
  `non_corner_api_client`; provenance + `k2_metrics` extended; CLI
  `main()` wires the new cache for standalone invocations.
- New: [config/knob_02_niche/_prompts/non_corner_v1.txt](../usecases_synthetic/config/knob_02_niche/_prompts/non_corner_v1.txt).
- [scripts/generate_variant.py](../usecases_synthetic/scripts/generate_variant.py):
  `llm_cache_k2_non_corner` instance + threading through `_run_knob_02`.
- All four K2 YAMLs: `non_corner_refill` + `non_corner_prompt_version` blocks.
- New: [tests/test_non_corner_refill.py](../usecases_synthetic/tests/test_non_corner_refill.py)
  — 28 new tests; full 1394-test synthetic suite green.

**Already landed — step 4j cleanup (2026-05-28):**
- Deleted full augmented variant trees: `usecases/music-augmented/` (125 MB),
  `usecases/games-augmented/` (168 MB), `usecases/products-augmented/`
  (21 MB). No `companies-augmented/` existed (R0 still gated).
- Deleted small-augmented variant trees per user 2026-05-28 sign-off
  (optional in original plan): `usecases/music-small-augmented/`
  (118 MB), `usecases/games-small-augmented/` (17 MB),
  `usecases/products-small-augmented/` (21 MB),
  `usecases/companies-small-augmented/` (23 MB).
- Deleted stale HPO trees:
  `cache/em_blocking_tuning/` (56 KB, pre-C4),
  `cache/em_matching_tuning/` (496 KB, pre-C12),
  `cache/fusion_tuning/` (3.5 MB, pre-C12),
  `cache/norm_tuning/` (212 KB, pre-C12).
- Retained per KEEP list: `cache/sm_tuning/` (C12 didn't touch SM),
  `cache/ditto_checkpoints/` (R6/R7 maintained).
- Total: ~497 MB freed. Step 5 now starts from a clean slate.

---

## R0 — Companies-FULL: rebaseline + variant generation (gated on user)

**Status (2026-05-29): UNBLOCKED.** The user has authored the
expanded companies fusion validation + test sets. R0 is no longer
gated externally — companies-FULL joins music / games / products as
a first-class domain in the R10 / Phase 2 cascade.

Historical context: companies-FULL was held back from the S.7 sweep
while music / games / products went through because the original
companies fusion gold counts (val=25, test=18) were too small to
give the fusion committee a stable accuracy estimate (each entity
worth ~5% of the score; a single fusion error moves macro_accuracy
by 5pp). The new expanded sets land companies in the same
load-bearing zone as the other three domains (target ≥100 / ≥100).

**What R0 means under R10 / Phase 2 (2026-05-29 update):**

R0 is no longer a separate cascade to be queued after the other
three. Companies-FULL slots into the Phase 2 per-domain cascade
just like products / music / games — but **before the cascade
fires for companies, an explicit conventions audit is required**.

**Companies-conventions audit (mandatory pre-step before R0's
cascade, added 2026-05-29):**

Companies-FULL was last touched 2026-05-15 — pre-R5 / R6 / R7 /
R7b / R9 / every R10 item. The configs and gold artifacts may be
inconsistent with the conventions every other domain settled into
this session. Before the Phase-2 cascade fires for companies, audit
and align:

- **Committee YAMLs**: `em_blocking_committee_companies.yaml`,
  `em_matching_committee_companies.yaml`,
  `normalization_committee_companies.yaml`,
  `fusion_committee_companies.yaml` — confirm they match the C12
  members-list shape (not pre-C12 per-attribute strategy shape) and
  carry the R6 Ditto roster decisions.
- **K1/K2/K4 LLM model_id**: confirm `gpt-5.4-mini` per the
  `feedback_synth_llm_and_naming` convention; companies may still
  be on `claude-opus-4-6` from an old run (R2-style migration).
- **K8 rename_table**: audit against the current companies source
  schema; per-source rung assignments should follow the canonical
  descriptive / abbreviated / cryptic / anonymized ladder.
- **Silver standard**: companies silver standard needs (re)building
  against the current canonical_schema if it's stale; the C13
  intact-cluster protection requires a fresh silver per the post-
  R10 pipeline.
- **EM gold + linkage**: confirm `usecases/companies/input/entitymatching/`
  contains the source-prefixed IDs the post-2026-05-15 loader
  expects (`companies_<n>_<original_id>`).
- **R10-C SM/Norm gold scope**: companies' `sm_mapping_gold.csv`
  may still be on the legacy small-attr scope; R10-C's expansion
  applies to companies the same way as to the other domains.
- **R10-I wide scope**: companies' Ditto `fields:` + Magellan
  `attributes:` need expansion to the full companies source
  schema, same as the other domains.
- **R10-D K1 v2 prompts**: `llm_prompt_version: v2` wired into
  companies' `knob_01_surface/companies.yaml`.
- **Pool builder**: companies uses Ditto for pool construction
  (R7-pool-builder retrain item). Confirm the top-level
  `cache/ditto_checkpoints/companies/best` symlink points at a
  post-R7-padding-fix checkpoint, or schedule the retrain inline.

Half-day audit + alignment work; mostly config edits. Land before
the companies cascade fires.

**Phase 2 cascade for companies (after the audit lands):**

1. R9-deferred sweep for companies under the R10-I wide scope.
2. R10-H baseline Ditto retrain for companies (~30-60 min MPS) —
   the −0.058 F1 hit on companies (per R7 "what's deferred") was
   the strongest single-domain argument for R10-H. Now mandatory.
3. `generate_variant.py --domain companies` under R10 A/D/F/E.
4. R10-G phase 2 retrain_variant_cascade for companies.
5. `measure_baseline.py --domain companies --with-llm`.
6. `validate_variant.py --domain companies --level {easy,medium,hard} --with-llm`.
7. `build_statistics.py --domain companies`.

Pre-existing
[usecases_synthetic/validation/companies/{easy,medium,hard}/metrics.json](../usecases_synthetic/validation/companies/)
are pre-block leftovers from an earlier pass against the small
gold; delete or overwrite before the Phase 2 companies cascade
runs.

The cross-domain Phase 2 order (added 2026-05-29 user directive):
~~**products first**, then companies, music, games~~ — **SUPERSEDED.**
Actual order this session (user redirects): **music first** (done /
in-flight 2026-06-01), then **companies** (user 2026-06-02: "put the
companies run right after music"), then **products** (deferred to after
companies, plan R10-M), then games. The companies readiness audit +
refreshed cascade + rigorous music-parity cross-checks are in **R10-N**
(2026-06-02), which supersedes the 2026-05-29 "Companies-conventions
audit" checklist above.

---

## R1 — Products: switch to the richer `data_cleaned_final` schema

**Status:** queued.

The current synthetic products source files at
[usecases_synthetic/usecases/products/input/data/products_<1..4>.json](../usecases_synthetic/usecases/products/input/data/)
carry only the 8 base columns: `id, brand, title, description, price,
priceCurrency, cluster_id, url`.

[usecases/products/input/data_cleaned_final/dataset_<1..4>_normalized.json](../usecases/products/input/data_cleaned_final/)
ships the same row counts with **7 extra structured columns** —
`title_description`, `model`, `model_number`, `product_type`,
`chipset_name`, `vram_gb`, `storage_gb`. Coverage of those extras:

| field                | ds1 | ds2 | ds3 | ds4 |
|----------------------|-----|-----|-----|-----|
| `title_description`  | 100%| 100%| 100%| 100%|
| `product_type`       | 100%| 100%|  ~100% | ~100% |
| `model`              |  92%|  91%|  90%|  92%|
| `storage_gb`         |  70%|  70%|  70%|  69%|
| `model_number`       |  41%|  41%|  39%|  40%|
| `chipset_name`       |  28%|  28%|  29%|  29%|
| `vram_gb`            |  28%|  28%|  28%|  28%|

Adopting this schema gives products a per-attribute surface that's
comparable to music (9 columns) and games (10+) instead of the
4-text-attribute degenerate case the current pipeline operates on.
Numeric attributes (`vram_gb`, `storage_gb`) become K5-format and K10-
reliability surface; categorical attributes (`product_type`,
`chipset_name`) become K8-rename surface; sparse identifiers
(`model_number`) become K3-drop / K4-coverage targets.

**Files to update (in order):**

1. **Replace source files.** Copy
   [usecases/products/input/data_cleaned_final/dataset_<n>_normalized.json](../usecases/products/input/data_cleaned_final/)
   over
   [usecases_synthetic/usecases/products/input/data/products_<n>.json](../usecases_synthetic/usecases/products/input/data/),
   preserving the `products_<n>` filenames the loader expects. Verify
   the `id` column still uses `products_<n>_<num>` IDs that match the
   EM gold (sample-checked: `data_cleaned_final` already does).

2. **Domain config** —
   [usecases_synthetic/config/domains/products.yaml](../usecases_synthetic/config/domains/products.yaml)
   - Lines 17-18: rewrite the "Schema is shared verbatim across all
     four sources" comment to enumerate the new column list.
   - Lines 56-58: drop the *"Optional category-specific fields …
     absent from most rows and stay out of scope"* sentence — those
     fields are now in scope.
   - Lines 59-64: extend `attribute_classes` with the new columns.
     Suggested classification (consistent with music/games):

     ```yaml
     attribute_classes:
       title: primary
       brand: key
       description: secondary
       price: secondary
       priceCurrency: secondary
       title_description: secondary      # derived long-string
       product_type: key                  # categorical partitioner
       model: secondary
       model_number: secondary
       chipset_name: secondary
       vram_gb: secondary
       storage_gb: secondary
     ```

3. **EM committees** —
   [em_blocking_committee_products.yaml](../usecases_synthetic/config/committees/em_blocking_committee_products.yaml),
   [em_matching_committee_products.yaml](../usecases_synthetic/config/committees/em_matching_committee_products.yaml).
   Extend the `fields:` / `text_cols:` lists on ditto_plm, magellan
   (comparator definitions), llm_matcher, comem, sc_block. The
   canonical "ditto fields = sc_block text_cols" invariant must hold,
   so pick the same set for both. Suggested:
   `[title, brand, description, model, model_number, product_type,
   chipset_name, vram_gb, storage_gb]` (drop `price` since the
   `MagellanMatcher` numeric handling lives behind separate
   `numeric_attributes` — see games config for the pattern).

4. **Fusion committee** —
   [fusion_committee_products.yaml](../usecases_synthetic/config/committees/fusion_committee_products.yaml).
   Add per-attribute `strategies` blocks for `model`, `product_type`,
   `chipset_name`, `vram_gb`, `storage_gb`. Numeric attributes get the
   robust-statistic family (`median`, `trimmed_mean`,
   `huber_m_estimator`, `prefer_higher_trust`) per the games
   `criticScore` / `userScore` pattern. Update `trust_scores` if
   relevant and extend the `evaluation_functions` /
   `evaluation_params` blocks (numeric `vram_gb` / `storage_gb` want
   `tolerance` thresholds analogous to `price`).

5. **Knob configs** — extend the per-domain
   [knob_01_surface/products.yaml](../usecases_synthetic/config/knob_01_surface/products.yaml),
   [knob_02_niche/products.yaml](../usecases_synthetic/config/knob_02_niche/products.yaml),
   [knob_03_drop/products.yaml](../usecases_synthetic/config/knob_03_drop/products.yaml),
   [knob_04_coverage/products.yaml](../usecases_synthetic/config/knob_04_coverage/products.yaml),
   [knob_05_format/products.yaml](../usecases_synthetic/config/knob_05_format/products.yaml),
   [knob_06_noise/products.yaml](../usecases_synthetic/config/knob_06_noise/products.yaml),
   [knob_08_naming/products.yaml](../usecases_synthetic/config/knob_08_naming/products.yaml),
   [knob_10_reliability/products.yaml](../usecases_synthetic/config/knob_10_reliability/products.yaml).
   - K8 (naming) needs `rename_table` entries for each new attribute
     across `descriptive` / `abbreviated` / `cryptic` / `anonymized`.
   - K5 (format) needs numeric format rules for `vram_gb` /
     `storage_gb` (e.g. `8` → `8 GB` → `8.00 GiB` → `8192`).
   - K10 (reliability) needs per-attribute corruption profiles.
   - K3 (drop) thresholds should target sparser attributes
     (`model_number`, `chipset_name`, `vram_gb`) — but verify the
     existing per-knob ratios still produce monotonic targets.

6. **EM gold** — confirm the existing
   `usecases_synthetic/usecases/products/input/entitymatching/products_<a>_2_products_<b>_<split>.csv`
   files keep their gold IDs valid against the new source records.
   The IDs are content-independent (`products_<n>_<num>`), so a
   sanity-grep of `id1 / id2` against the new source `id` columns
   should be 100%. No regen needed if it is.

7. **Re-baseline + re-variant.** Once 1-6 land, run:
   - `python usecases_synthetic/scripts/measure_baseline.py --domain products --with-llm`
   - `python usecases_synthetic/scripts/generate_variant.py --domain products --level {easy,medium,hard}`
   - `python usecases_synthetic/scripts/validate_variant.py --domain products --level {easy,medium,hard} --with-llm`
   - `python usecases_synthetic/scripts/build_statistics.py --domain products`

   The previous products baseline metrics + variant metrics + XLSX
   reflect the 4-column schema and will become stale on landing.

---

## R2 — Migrate products K1/K2/K4 LLM model_id to `gpt-5.4-mini` (landed)

**Status:** landed this session (2026-05-28). All three products
knob YAMLs (K1 surface, K2 niche, K4 coverage) now pin
`llm_model_id: gpt-5.4-mini`, matching the
`feedback_synth_llm_and_naming` memory convention used by the other
three active domains.

Files updated:

- [knob_01_surface/products.yaml](../usecases_synthetic/config/knob_01_surface/products.yaml)
- [knob_02_niche/products.yaml](../usecases_synthetic/config/knob_02_niche/products.yaml)
- [knob_04_coverage/products.yaml](../usecases_synthetic/config/knob_04_coverage/products.yaml)

**Effect on caches:** the K2/K4 LLM caches at
[cache/knob_02_interpolations/products/](../usecases_synthetic/cache/knob_02_interpolations/products/),
[cache/llm_cache/](../usecases_synthetic/cache/llm_cache/) etc. are
keyed by `model_id` — the switch invalidated the legacy
`claude-opus-4-6` entries (no source-of-truth loss; regenerable on
next run).

---

## R3 — Train the SC-block checkpoint for products

**Status:** prerequisite for any next run that touches em_blocking.

Earlier in this session
[em_blocking_committee_products.yaml](../usecases_synthetic/config/committees/em_blocking_committee_products.yaml)
was flipped to `sc_block: enabled_by_default: true` to align with
the canonical 6-member blocking roster. The checkpoint at
`usecases_synthetic/cache/sc_block_checkpoints/products/best` **does
not yet exist** (`cache/sc_block_checkpoints/` currently holds only
`companies`, `games`, `music`). `SCBlockBlocker._lazy_load_encoder`
([sc_block_blocker.py:264-266](../usecases_synthetic/lib/sc_block_blocker.py#L264-L266))
raises `FileNotFoundError` if the checkpoint is absent, which would
abort the EM blocking phase.

**Action:** before the next products baseline / variant run, train:

```
python usecases_synthetic/scripts/sc_block/train.py --domain products
```

with `text_cols` matching whatever ditto field-set R1 lands. **The
sc_block and ditto field sets must be identical** (per the canonical
config comment in
[em_blocking_committee.yaml](../usecases_synthetic/config/committees/em_blocking_committee.yaml)).
If R1 lands first the trained checkpoint should target the expanded
field set; if R1 is deferred, train against the existing `[title]`.

---

## R4 — K2 hard interpolation produced 0 entities on the last run

**Status:** **diagnosed 2026-05-19 via step-2 standalone replay.**
Action subsumed by R-1 §C1 (α/β/γ decision). R4 is retained here for
audit-trail traceability of the original investigation.

For both **products** and **music** the K2 hard realised CSV records
`interpolated: 0` despite the per-domain config calling for K2
interpolations:

```
usecases/products-augmented/hard/output/baselines/knob_02_realised.csv
→ baseline_ratio=0.435, target=0.65, final_ratio=0.435, interpolated=0

usecases/music-augmented/hard/output/baselines/knob_02_realised.csv
→ baseline_ratio=0.26, target=0.8, final_ratio=0.26, interpolated=0
```

**Original framing (incorrect): "12 LLM calls all guardrail-dropped".**

**Corrected diagnosis (step 2, 2026-05-19):** both failure modes
collapse to the same root cause — `strict_cache=True` at hard level
+ empty cache for the deterministically-selected pair-hashes:

- `generate_variant.py:792-795` forces `strict_cache_k2 = True` at
  hard for non-aliased domains.
- `apply_knob_02_niche.py:2229` does the same when invoked
  standalone (`strict_cache = args.strict_cache or (level == "hard")`).
- The K2 cache at `cache/knob_02_interpolations/music/` (1597 entries)
  is mostly populated from music-small, which uses a different
  pair-hash space. Music FULL's 1080 deterministic pair-hashes are
  not present.
- Music K2 standalone replay (2026-05-19) confirms:
  `rejected_strict_cache_miss=1080`, `interpolated=0`, no LLM calls.
- Same applies to products and games: cache empty → strict miss →
  loop continues at
  [apply_knob_02_niche.py:1997-1999](../usecases_synthetic/scripts/apply_knob_02_niche.py#L1997)
  → 0 entities produced.

The 1597 entries in the `music/` cache dir are NOT the "12 full-music
calls" the prior investigation claimed; they are music-small calls
parked in the shared per-domain directory.

**Action (subsumed by R-1 C1, decided 2026-05-22):** see R-1 §C1
above for the resolved decision — **never fail on cache miss**.
`strict_cache_{k1,k2}` defaults flipped to `False` in
`generate_variant.py` + both standalone CLIs; on miss the K2 path
either calls a live LLM (when wired) or falls through to the
deterministic blender. The original "instrument the guardrails"
sub-action landed under step 1 (rejection counters are in place);
those counters become load-bearing once cache lookups start hitting
in earnest after the products / music / games reruns.

---

## R5 — Items already applied this session (no action needed)

For audit-trail completeness; these do not need re-application but
their downstream effects depend on R1-R3 landing:

- `em_matching_committee_products.yaml`: magellan / llm_matcher / comem
  flipped to `enabled_by_default: true`; llm_matcher + comem upgraded
  to `gpt-5.4-mini`; llm_matcher stripped of few-shot machinery (now
  zero-shot per the canonical roster); `required_axes` updated to
  `[learned, llm]`.
- `em_blocking_committee_products.yaml`: sc_block flipped to enabled
  (see R3 for the dependency).

The existing
[usecases_synthetic/baselines/products/baseline_metrics.json](../usecases_synthetic/baselines/products/baseline_metrics.json)
and
[usecases_synthetic/validation/products/{easy,medium,hard}/metrics.json](../usecases_synthetic/validation/products/)
were produced before R4 landed and still reflect the
`[ditto_plm, llm_matcher, comem]` roster. They become stale once R1/R3
land and the rerun (R1 step 7) regenerates everything against the
upgraded schema + full 4-member em_matching + 6-member em_blocking
rosters.

---

## R6 — Committee Ditto re-baselining (landed 2026-05-27)

**Status: code + symlinks landed; baseline_metrics.json re-runs pending
R7 trainer-padding decision (see below).**

Investigation of the games EM committee gap (§4g) surfaced three
interlocked issues with the committee Ditto pipeline that affected
all domains. The user's directives 2026-05-27:

- **R6-1**: Audit + replace all stale committee Ditto symlinks /
  caches across domains.
- **R6-2**: Verify whether Ditto is genuinely direction-sensitive at
  inference (user's prior expectation: it is **not**, and the F1=0
  result on `metacritic_dbpedia` must have another cause).
- **R6-3**: ADI is reserved for the pool builder; committee Ditto
  must train on PyDI splits only (committee evaluation distribution =
  training distribution).

All three landed 2026-05-27, plus a fourth issue uncovered during the
work.

### R6-1 — Stale checkpoint symlinks

Before the audit, `usecases_synthetic/cache/ditto_checkpoints/<domain>/best`
pointed at:

| Domain | Old symlink target | Model | Test F1 reported at train time |
|---|---|---|---:|
| games | `run_20260429_161752/checkpoints/best` | distilbert-base-uncased, 3 epochs | 0.831 |
| music | `run_20260429_162024/checkpoints/best` | distilbert-base-uncased, 3 epochs | 0.804 |
| companies | **(no symlink — `best/` was a stale roberta directory)** | roberta-base, 15 epochs (Apr 20) | — |
| products | `sweep_lr_1e-5_unweighted_9field/run_20260522_145821/checkpoints/best` | roberta-base | 0.969 |

Repointed to the canonical PyDI-trained checkpoints:

| Domain | New target | Source | Test F1 (train.py) |
|---|---|---|---:|
| games | `/cache/ditto_checkpoints/games_ab_pydi_raw/run_20260526_175204/checkpoints/best` | fresh PyDI roberta (R6-3) | 0.730 |
| music | `/cache/ditto_checkpoints/music/sweep_lr_1e-4_weighted/run_20260505_063929/checkpoints/best` | R2 winner, already PyDI-trained per `_prep_music.py` | 0.984 |
| companies | `/cache/ditto_checkpoints/companies_ab_pydi_raw/run_20260527_110410/checkpoints/best` | fresh PyDI roberta (R6-3) | 0.860 |
| products | unchanged | already correct | 0.969 |

Deleted ~3 GB+ of stale caches: Apr 29 distilbert runs for games +
music (255 MB each), Apr 20 companies sweep runs (~1.7 GB total),
superseded May 15 products sweep (483 MB), unwired A/B artifacts
(`games_ab_raw`, `games_ab_normalized`, `games_ab_pydi_normalized`,
~1.4 GB combined), and 10 stale Ditto inference cache files (940 KB).
Top-level
[/cache/ditto_checkpoints/](../cache/ditto_checkpoints/) (which the
pool builder reads from per [build_pool.py:170,230,320,365](../usecases_synthetic/scripts/build_pool.py))
is left untouched — its R2 sweep history is the pool builder's
property and `_prep_*.py --train-source adi` remains the supported
path for regenerating it.

### R6-2 — Direction sensitivity verified (loader bug, not model)

Initial post-R6-1 committee run on games showed
`ditto_plm metacritic_dbpedia: F1=0 (tp=0, fp=0, fn=106)` and
`magellan: crash ("No valid features could be extracted")`. The first
hypothesis was Ditto's transformer being position-sensitive — but
magellan, which builds per-attribute features that should be
order-invariant, *also* crashing pointed elsewhere.

Root cause: my §4g loader fix made `_load_em_gold` direction-tolerant
but did NOT swap `id1` / `id2` columns when loading from the reversed
direction. For `pair = (metacritic, dbpedia)` loaded from
`dbpedia_2_metacritic_test.csv`, the returned frame had id1=dbpedia
ids. Downstream
[committee_em.py:2068-2074](../usecases_synthetic/lib/committee_em.py#L2068-L2074)
passes `test_gold[["id1","id2"]]` as candidates to the matcher with
`df_left = sources[src1]` (metacritic) and `df_right = sources[src2]`
(dbpedia). The matcher looks up id1 (dbpedia values) in df_left
(metacritic source) → every lookup fails → magellan crashes during
feature-extraction, Ditto silently predicts negative on every pair.

Fix landed in:
- [lib/variant_loader.py:_load_em_gold (313-373)](../usecases_synthetic/lib/variant_loader.py#L313-L373) — when loading from the reverse direction, swap id1↔id2 so the returned frame matches the declared (src1, src2) pair direction.
- [lib/committee_em.py:_load_labelled_split (1908-1957)](../usecases_synthetic/lib/committee_em.py#L1908-L1957) — same swap (this code path bypasses `_load_em_gold` for variant-level reads).

Test updated in
[tests/test_variant_loader.py:TestLoadEMGoldDirectionTolerance](../usecases_synthetic/tests/test_variant_loader.py)
to assert the post-load id ordering matches the declared pair
direction (regression guard).

Verified after the fix: Ditto produces correct predictions on both
pair directions. The user's expectation was correct: Ditto is
direction-agnostic given correct input.

### R6-3 — ADI removed from committee training defaults

[_prep_companies.py](../usecases_synthetic/scripts/ditto/_prep_companies.py)
and [_prep_games.py](../usecases_synthetic/scripts/ditto/_prep_games.py)
gained a `--train-source {pydi,adi}` flag with `pydi` as the default.
`--train-source pydi`:

- **Companies** — loads `forbes_2_<src>_train.csv` and `_val.csv` from
  [usecases/companies/input/entitymatching/](../usecases/companies/input/entitymatching/).
  Pair `dbpedia_fullcontact` (no PyDI gold) is skipped. Test = PyDI.
  Result: train 1968 / val 948 / test 599 pairs.
- **Games** — loads `<pair>_train.csv` from
  [usecases/games/input/entitymatching/](../usecases/games/input/entitymatching/);
  no on-disk val for games, so train is split 80/20 stratified by
  label with seed=42. Result (post-id-resolution): train 724 / val
  185 / test 739. Note: **22% of PyDI train rows reference dbpedia
  ids that don't exist in the post-2026-05-04 source refresh** —
  silent drops during `build_ditto_pair_records_from_gold`. Test
  CSVs are clean (0% stale). Logged for separate cleanup; not
  blocking R6.
- **Music** — `_prep_music.py` was already PyDI-only; unchanged.
- **Products** — `_prep_products.py` was already PyDI-only (no ADI
  pipeline exists for products); unchanged.

[_prep_games.py PAIRS](../usecases_synthetic/scripts/ditto/_prep_games.py)
also now skips the F11-dropped `metacritic_sales` pair when its test
CSV is missing (previously crashed).

`--train-source adi` remains available as a non-default for the pool
builder's R2 training path. The committee yaml symlinks
(`usecases_synthetic/cache/...`) point only at PyDI-trained
checkpoints.

### R6-4 — DittoMatcher tokenization patched (long-standing bug surfaced)

When verifying R6-3 with a fresh diagnostic, the games PyDI checkpoint
produced F1=0 in committee despite `train.py` reporting F1=0.73 at the
same threshold on the same gold pairs. Direct probing showed the
checkpoint outputs scores ∈ [0.33, 0.43] for the 222 positives via
[DittoMatcher](../usecases_synthetic/lib/ditto_matcher.py) but
∈ [0.003, 0.999] (median 0.94, 145 ≥ 0.5) via `train.py`'s eval loop.

Root cause: training-side
[PairDataset.pad](../usecases_synthetic/third_party/ditto_modern/data.py#L199-L228)
pads sequences with token id `0` (which is RoBERTa's `<s>`/CLS token,
**not** the pad token id=1) and computes `attention_mask = 1 if tok
!= 0 else 0`. This masks the CLS token's attention at training time;
the model learns to classify with CLS masked. The HF-default
`tokenizer(..., padding=True)` call in
[DittoMatcher._score_batch](../usecases_synthetic/lib/ditto_matcher.py)
pads with id=1 and gives CLS attention=1 — different attention pattern
at inference, score distribution shifts. PyDI's narrower training
score distribution made the shift fatal; ADI's wider distribution
absorbed it (which is why the earlier fresh-ADI diagnostic appeared
to "work").

Patched DittoMatcher's `_score_batch` to mimic the trainer's
pad-with-0 + `attention_mask = (tok != 0)` convention. Verified:
PyDI score distribution now matches `train.py`'s predictions.csv
exactly (n=222 positives, median 0.941, 145 ≥ 0.5). The proper
upstream fix (use `tokenizer.pad_token_id`, retrain all checkpoints)
is parked under **R7**.

### R6-5 — Audit of every other committee member (landed 2026-05-27)

User directive (2026-05-27): *"ensure that all other committee members
are trained and evaluated on the pydi train/validation/test sets"* and
*"also ensure that everything is evaluated on the correct test set"*.

Audit conclusion: **all trainable members beyond Ditto already
train + eval on PyDI splits**. R6-1/2/3/4 closed the only outstanding
gap (committee Ditto on games + companies was ADI-trained / stale).
No further changes required for the other stages.

**Trainable members (verified PyDI):**

| Stage | Member | Training data source | Verified |
|---|---|---|---|
| EM blocking | `sc_block` (SCBlockBlocker, RoBERTa+SupCon per-domain) | `<pair>_train.csv` from `usecases/<domain>/input/entitymatching/` via [scripts/sc_block/train.py:_load_em_pair_splits](../usecases_synthetic/scripts/sc_block/train.py) | ✓ |
| EM matching | `magellan` (MagellanMatcher, per-pair RandomForest) | `<pair>_train.csv` from same PyDI dir via [committee_em._resolve_pair_train_path:341-374](../usecases_synthetic/lib/committee_em.py#L341-L374); injected at matcher instantiation per pair | ✓ |
| Norm (C12) | `rule_per_attribute_optimal` (val-sweep over rule normalizers per attribute) | `usecases/<domain>/input/fusion/validation_set.xml` per [committee_norm_c12._load_val_and_test_targets:356-367](../usecases_synthetic/lib/committee_norm_c12.py#L356-L367) | ✓ |
| Fusion (C12) | `pydi_per_attribute_optimal` (val-sweep over fusion strategies per attribute) | `usecases/<domain>/input/fusion/validation_set.xml` | ✓ |

**Existing sc_block checkpoints across domains** (all PyDI-trained, post-2026-05-04 source refresh):

| Domain | Run dir | Date | best_epoch |
|---|---|---|---:|
| companies | `run_20260515-122448` | 2026-05-15 | 1 |
| games | `run_20260510-221941` | 2026-05-10 | 1 |
| music | `run_20260510-221943` | 2026-05-10 | 0 |
| products | `run_20260522-142827` | 2026-05-22 | 2 |

(games + companies sc_block were trained on a PyDI train CSV that
silently drops ~22% of rows referencing stale post-refresh dbpedia
ids — same staleness issue flagged for Ditto. Not blocking; quality
acceptable. Logged for the R6 stale-CSV cleanup follow-up.)

**Non-trainable / distribution-agnostic members** (no training,
inherently correct across distributions):
- EM blocking: TokenBlocker, StandardBlocker, SortedNeighbourhoodBlocker, BM25Blocker (char-ngram post-C4), EmbeddingBlocker (BAAI/bge-base-en-v1.5 pretrained).
- Norm: TextCleanNormalizer, DateIsoNormalizer, NumberLocaleNormalizer, CountryIsoNormalizer, TaxonomyLookupNormalizer.
- Fusion strategies: voting, longest_string, most_complete, median, average, trimmed_mean, huber_m_estimator, intersection, union, fusionquery_only, truthfinder_only, ltm_only, casefusion_only, accusim_only, voting_only, prefer_higher_trust_only.
- SM: LabelBasedSchemaMatcher (jaro-winkler), InstanceBasedSchemaMatcher (TF cosine), EmbeddingBasedSchemaMatcher (pretrained SBERT), DuplicateBasedSchemaMatcher (consumes runtime-injected EM-gold positives from `bundle.em_gold` — already PyDI per [committee_sm.py:572,714](../usecases_synthetic/lib/committee_sm.py)).

**Zero-shot LLM members** (no training, eval-only):
- llm_matcher (MatchGPTMatcher), comem (ComEMMatcher), llm_openai (LLMBasedSchemaMatcher), magneto_slm_llm (MagnetoSchemaMatcher — uses pretrained SLM + LLM), coma_hybrid (ComaSchemaMatcher — Java lib reload, library-based), llm_only members in Norm + Fusion C12.

**Test-set audit — every stage scores against the variant's PyDI gold:**

| Stage | Test gold path | Notes |
|---|---|---|
| SM | `usecases/<domain>/input/schemamatching/target_schema.json` | JSON Schema; ground truth column set used by [committee_sm.score_correspondences](../usecases_synthetic/lib/committee_sm.py) ✓ |
| EM blocking | `bundle.em_gold[pair]` = PyDI `<pair>_test.csv` | post-R6-2 direction-tolerant + id-swapped via [variant_loader._load_em_gold](../usecases_synthetic/lib/variant_loader.py#L313-L373) ✓ |
| EM matching | same as EM blocking; `f1_baseline_test` is closed-set F1 against the gold's labeled (pos + neg) pairs | post-C10 + post-R6-2 ✓ |
| Norm | `usecases/<domain>/input/fusion/test_set.xml` per-attribute targets | [committee_norm_c12._load_val_and_test_targets:356-367](../usecases_synthetic/lib/committee_norm_c12.py) ✓ |
| Fusion | `bundle.fusion_gold` = same `test_set.xml` (games + music use `*_final.xml`; companies + products use canonical names — resolved via `DomainConfig.fusion_files`) | [committee_fusion_c12.py:947](../usecases_synthetic/lib/committee_fusion_c12.py#L947) ✓ |

Variant-level paths (easy/medium/hard) use the K2-regenerated splits
emitted by [apply_knob_02_niche.regenerate_em_splits](../usecases_synthetic/lib/corner_case_miner.py),
which are always written in PyDI's `<src1>_2_<src2>_<split>_<version>.csv`
canonical-direction form (C11 work). The post-R6 loader fix handles
those equivalently; no per-variant changes needed.

**No further action**. The user's directive is satisfied by R6-1/2/3/4
landing — those addressed the only domain (Ditto on games + companies)
where the committee was loading non-PyDI or stale artifacts. The other
stages were already correctly wired.

### Final committee EM F1 with R6 fixes

| Domain | Pre-R6 (old yaml + stale symlink) | Post-R6 (current code + symlinks) | Δ |
|---|---:|---:|---:|
| games | macro_f1 = 0.586 (dbpedia_sales only — loader skipped metacritic_dbpedia) | macro_f1 = **0.740** (ditto_plm both pairs) | +0.154 |
| music | macro_f1 = 0.976 (R2-quality but with stale-ditto-checkpoint artifact) | macro_f1 = **0.977** | +0.001 (correct checkpoint, same R2 winner) |
| companies | em_matching section empty (no checkpoint) | macro_f1 = **0.881** (ditto_plm 0.871 + magellan 0.890) | (was broken) |

Verified per-pair for games: `dbpedia_sales` F1=0.771 (was 0.586),
`metacritic_dbpedia` F1=0.708 (was skipped). Companies per-pair:
`forbes_dbpedia` 0.886, `forbes_fullcontact` 0.857. Music per-pair:
`musicbrainz_discogs` 0.968, `musicbrainz_lastfm` 0.987.

### R6 — Files in scope

**Code:**
- [lib/variant_loader.py:_load_em_gold](../usecases_synthetic/lib/variant_loader.py#L313-L373) — direction-tolerant + id swap.
- [lib/committee_em.py:_load_labelled_split](../usecases_synthetic/lib/committee_em.py#L1908-L1957) — same swap (variant-level path).
- [lib/ditto_matcher.py:_score_batch](../usecases_synthetic/lib/ditto_matcher.py) — pad-with-0-and-mask-cls patch.
- [scripts/ditto/_prep_companies.py](../usecases_synthetic/scripts/ditto/_prep_companies.py) — `--train-source pydi|adi` (default pydi).
- [scripts/ditto/_prep_games.py](../usecases_synthetic/scripts/ditto/_prep_games.py) — same (default pydi).

**Tests:**
- [tests/test_variant_loader.py](../usecases_synthetic/tests/test_variant_loader.py)
  `TestLoadEMGoldDirectionTolerance::test_loads_when_file_is_reverse_direction`
  asserts the id1↔id2 swap. 13 tests pass.

**Checkpoints:**
- New: `/cache/ditto_checkpoints/games_ab_pydi_raw/run_20260526_175204` (games committee).
- New: `/cache/ditto_checkpoints/companies_ab_pydi_raw/run_20260527_110410` (companies committee).
- Symlinks: `usecases_synthetic/cache/ditto_checkpoints/{games,music,companies}/best` repointed; products unchanged.

**Pool rebuild dependency (was the original R6 framing pre-2026-05-27)
— not yet executed**: the bucket-C policy tightening landed (every
disagreement → LLM adjudicator; see
[lib/pool_builder.py:build_buckets](../usecases_synthetic/lib/pool_builder.py)).
Pools were last built under the old confident-pos / confident-neg
auto-paths; rebuilding under the new policy adds ~14 314 LLM
adjudicator calls (companies 1818, games 8134, music 3378 — ~$1-$10
at gpt-5.4 / gpt-5.4-mini pricing; 1-3 hours wall-clock). **Gated on
R7 decision**: if the trainer-padding fix lands and retrain
cascades, the pool builder's Ditto checkpoint composition changes
too; bundle the pool rebuild with that work.

---

## R7 — Trainer padding fix + Ditto retrain (verified 2026-05-27; full sweep deferred to post-step-5)

**Status: code fix landed + verified on games. Per-domain baseline retrains
for music + companies + products bundled with the post-step-5 variant
regen (user directive 2026-05-27: "we will do this after generating
new variants. just verify everything works").**

The Ditto training pipeline
([third_party/ditto_modern/data.py:PairDataset.pad](../usecases_synthetic/third_party/ditto_modern/data.py#L199-L228))
pads with token id `0` (RoBERTa's CLS, not pad) and masks
attention on tokens that equal 0. This trains the model with the CLS
token's attention masked out — confirmed by direct inspection 2026-05-27.
R6-4 patched the inference side (DittoMatcher) to mimic this convention
so committee scores match `train.py`'s reported numbers, but the
underlying training is still wrong:

- All Ditto checkpoints across domains carry this learning artifact.
- Score distributions remain narrower than they should be (more
  pairs near the decision boundary), which makes the model fragile
  to threshold + inference-path variation.
- Any downstream tool relying on standard transformer behavior
  (e.g. embedding extraction, transfer learning) will see degraded
  representations.

### R7 — what landed (code, 2026-05-27)

1. **`third_party/ditto_modern/data.py:PairDataset.pad`** rewritten as
   an instance method that reads `self.tokenizer.pad_token_id` and
   pads with it; `attention_mask = (tok != pad_id)`. CLS is no longer
   masked. Tested via the existing trainer test set + downstream
   committee tests (72/72 pass post-fix).
2. **`third_party/ditto_modern/trainer.py`** updated to pass
   `dataset.pad` (bound method) as the DataLoader `collate_fn` (was
   `PairDataset.pad` unbound).
3. **`lib/ditto_matcher.py:_score_batch`** reverted the R6-4 hack —
   now uses the standard HF `tokenizer(lefts, rights, padding=True,
   ...)` call. Drop-in compatible with HF defaults.
4. **`scripts/sc_block/train.py`**: audited 2026-05-27 — uses
   `tokenizer(batch_texts, padding=True, ...)` already (HF standard).
   **No bug, no change needed.**

### R7 — verification on games (the validation target, 2026-05-27)

Retrained PyDI Ditto on games with the fixed pad. Same hyperparameters
as R6's games_ab_pydi_raw (lr=1e-5, weighted, da=del, fixed_threshold=0.5):

| Surface | Pre-R7 (buggy train + R6-4 hack) | Post-R7 (proper train + std inference) | Δ |
|---|---:|---:|---:|
| Training-time test F1 | 0.730 | **0.771** | +0.041 |
| Best epoch | 6 | 7 | +1 |
| Best val F1 | 0.972 | 0.966 | -0.006 |
| Committee EM diagnostic ditto_plm F1 | 0.740 | **0.766** | +0.026 |

Key observation: post-R7 training-time test F1 (0.771) and committee
inference F1 (0.766) now **match within 0.005** — the previously
catastrophic ~0.6 divergence (R6-4 finding: train.py F1=0.73 vs
DittoMatcher F1=0) is gone. The trainer + inference are calibrated to
the same padding convention.

Per-pair on games:
- `dbpedia_sales`: 0.834 (was 0.771 pre-R7) → +0.063
- `metacritic_dbpedia`: 0.697 (was 0.708 pre-R7) → -0.011

Symlink `usecases_synthetic/cache/ditto_checkpoints/games/best` repointed
to `/cache/ditto_checkpoints/games_r7_verify/run_20260527_130708/checkpoints/best`.

### R7 — transitional state for the other 3 domains

Music, companies, products **still use their pre-R7 buggy checkpoints**.
DittoMatcher is now using standard HF padding, so these 3 domains are
in a train-inference mismatch state until they're retrained. Quantified
2026-05-27 (committee EM diagnostic, baseline level):

| Domain | Pre-R7 ditto_plm F1 (R6 numbers, R6-4 hack) | Transitional ditto_plm F1 (buggy ckpt + std inference) | Δ |
|---|---:|---:|---:|
| music | 0.977 | 0.969 | −0.008 |
| companies | 0.871 | 0.813 | **−0.058** |
| products | 0.969 | 0.957 | −0.012 |

Music + products degrade trivially because their original score
distributions are wide and bimodal (most positives near 1.0, most
negatives near 0); the padding shift moves boundary cases but doesn't
flip many predictions. Companies degrades more visibly because its
training data is smaller and its score distribution is narrower —
boundary-case predictions flip when the attention pattern changes.

The transitional state is functional (no crashes, no F1=0 collapses)
and the degradation is bounded. Per the user directive 2026-05-27 the
3 remaining baseline retrains are deferred to bundle with the
domain-specific variant regeneration runs in steps 5/6.

### R7 — what's deferred (split across domain-specific reruns)

Three pieces of deferred work, with **different gating semantics**:

1. **R7-baseline-finish** (domain-level prerequisite, before variant
   rerun): retrain baseline Ditto for music + companies + products on
   PyDI (same recipe as games). ~30-60 min × 3 domains = 1-3 hours
   MPS. Repoint synthetic-side committee symlinks. **Optional per the
   step-4h `gate_mode: full_llm` insurance** — the transitional
   calibration doesn't propagate into the variant gold under full-LLM
   adjudication. Recommended to land before music + companies reruns
   to clean up the −0.058 F1 hit on companies; trivial for music /
   products (−0.008 / −0.012). Slot into step 6 (music + games) or
   skip for the products run.

2. **Top-level pool-builder Ditto retrains** (per-domain prerequisite,
   pre-variant-rerun, bundles with pool rebuild): same training
   pipeline, ADI data source where applicable (`--train-source adi`);
   retrains the pool builder's Ditto under the corrected padding.
   ~30-60 min × 3 affected domains (music / games / companies) =
   1.5-3 hours MPS. Repoint top-level
   `/cache/ditto_checkpoints/<domain>/best` symlinks. **Bundles with
   the pool rebuild under the bucket-C tightened policy** — the
   rebuild reads Ditto scores, so the new top-level checkpoint should
   be in place first. Products is orthogonal (uses
   `build_pool_products.py` with native cluster_id linkage, no Ditto).

3. **R7c — per-variant Ditto + sc_block retraining** (**per-domain
   POST-variant-rerun**, gated on that domain's variant generation):
   only possible AFTER the variant has been generated for the domain,
   because R7c needs the K2-emitted `<pair>_train_corner_filled.csv`
   files as training data. Train one Ditto + sc_block checkpoint per
   (domain, level) → ~30-60 min × 3 levels × 2 matchers = 3-6 hours
   MPS PER DOMAIN. Write to
   `usecases_synthetic/cache/ditto_checkpoints/<domain>/variant_<level>/best`
   so [committee_em.py:_resolve_variant_checkpoint_path](../usecases_synthetic/lib/committee_em.py)
   picks it up. Once landed, `validate_variant.py` produces meaningful
   4-cell dual-model dual-test metrics (R7b headline metric
   `macro_f1_variant_model_on_regen_test` becomes the load-bearing
   monotonicity F1). Until R7c lands for a domain, the variant cells
   alias the baseline cells in `validate_variant.py` — the run still
   completes, just without the dual-model signal.

   **Per-domain ordering**:
   ```
   variant generation (writes corner_filled CSVs)
     └→ validate_variant.py (initial run; variant cells alias baseline)
        └→ R7c retrain Ditto + sc_block on corner_filled CSVs
           └→ validate_variant.py rerun (4-cell matrix populated;
              dual-model signal live)
   ```

   This is a per-domain optional follow-up, NOT a batched sweep across
   all 4 domains. Each domain's R7c is its own decision keyed off
   that domain's variant + monotonicity signals from the initial
   validation run.

A separate concern flagged during R6: **per-variant Ditto retraining**
(committee members trained on the K2-regenerated splits for each
variant level rather than reusing the baseline checkpoint). The
infrastructure for this isn't in place — the EM committee runner
loads one checkpoint per domain. Per-variant retraining requires a
runner hook + ~30-60 min × 4 levels × 4 domains = ~8-16 hours
additional MPS work. Captured as **R7b** below.

---

## R7b — Dual-model dual-test evaluation infrastructure (landed 2026-05-27)

**Status: code + tests landed; per-variant retraining run gated on R7
(trainer padding fix).**

User directive (2026-05-27): *"Let's have two model evaluations.
Let's evaluate for both test surfaces the original models trained on
the baseline AND the new models trained on the regenerated variant
train sets. All are reported but load-bearing for monotonicity remains
f1_regen_test on the models trained on the regenerated variant train
sets."*

### What R7b changes

At every variant level (easy / medium / hard), each EM committee
member is now evaluated on a **2 × 2 matrix**: two model variants
(baseline-trained, variant-trained) × two test surfaces (baseline_pruned,
corner_filled). Per (pair, member, level) the runner emits 4 F1
metrics for EM matching + 4 pair_recall metrics for EM blocking:

| Cell | Train data | Test gold | Use |
|---|---|---|---|
| `*_baseline_model_on_baseline_test` | original baseline train | original test minus dropped pairs (Set 1) | reference — pre-R7b semantics |
| `*_baseline_model_on_regen_test` | original baseline train | K2-corner-filled test (Set 2) | pre-R7b headline (now reference) |
| `*_variant_model_on_baseline_test` | K2-regenerated variant train | Set 1 | reference |
| **`*_variant_model_on_regen_test`** | K2-regenerated variant train | Set 2 | **load-bearing for monotonicity** |

At baseline level the model axis collapses (no variant train) and the
test axis collapses (no K2 regen), so all 4 cells equal the single
baseline F1. At variant levels where the variant checkpoint hasn't
been trained yet (R7c pending), the variant cells alias the baseline
cells (variant_model_distinct = 0.0).

### Why the variant-trained surface is load-bearing (paper argument)

Variant-trained on regen_test **isolates intrinsic data difficulty
from transfer-learning gap**. The headline claim becomes *"even a
purpose-trained model degrades from easy → hard"* — strictly the
intrinsic-difficulty claim — rather than *"a clean-trained model
fails to transfer to noisy variants"* — which would conflate
distribution shift with intrinsic difficulty. Both claims are valid;
the dual-cell report exposes both. R7b picks the variant-trained
surface as the slope verdict because it answers the stronger, more
defensible claim form.

Matches standard difficulty-benchmark methodology (ImageNet-C,
GLUE-Adversarial, MNIST-Rotate) where the in-distribution surface
(train + test on the same corruption level) is the headline and
zero-shot transfer is reported as a complementary number.

Counter-argument (worth addressing in the paper): "in-distribution
risks the model memorising K2's synthesis quirks." Mitigation —
corner_filled is built from **real near-twin entities mined from the
variant**, not random synthetic noise. K2 doesn't fabricate pairs; it
surfaces real entities that share many attribute values. In-distribution
training therefore tests "can the model learn the corner-heavy
subspace?" which is a meaningful generalisation question.

### Which committee members retrain per variant (under R7c)

| Member | Retrains per variant? | Train file at variant level |
|---|---|---|
| **Ditto (ditto_plm)** | YES — new RoBERTa checkpoint per (domain, level) | pooled `<pair>_train_corner_filled.csv` across pairs |
| **sc_block** (EM blocking) | YES — new RoBERTa+SupCon checkpoint per (domain, level) | same |
| **Magellan** | YES — per-pair RandomForest fit at runtime | `<pair>_train_corner_filled.csv` (now picked up by `_resolve_variant_train_path`) |
| llm_matcher / comem / token / standard / bm25 / etc. | NO (zero-shot or non-trainable) | — — variant_model aliases baseline |

For trainable members, R7b looks for a variant checkpoint at
``usecases_synthetic/cache/<model>_checkpoints/<domain>/variant_<level>/best``
and uses it if present. When absent (today's state — no R7c run yet),
the runner gracefully falls back to the baseline checkpoint, and the
variant cells equal the baseline cells. The infrastructure is in place
so that once R7c trains the variant checkpoints, the dual-model
numbers diverge and monotonicity reads the headline cleanly.

### R7b — what landed

**Code:**
- [lib/committee_em.py](../usecases_synthetic/lib/committee_em.py):
  - New helpers `_resolve_variant_checkpoint_path` (variant ckpt
    lookup with graceful fallback) + `_resolve_variant_train_path`
    (variant train CSV lookup, prefers `<pair>_train_corner_filled.csv`).
  - `_build_matcher` + `_build_blocker` accept a `checkpoint_override`
    kwarg used to swap the variant-trained checkpoint at instantiation.
  - `EMMatchingCommitteeRunner.run` rewritten to build baseline +
    variant matcher instances per (pair, member); each matcher runs
    once against the corner_filled candidate set, then scores against
    both gold surfaces (4-cell matrix); legacy `f1_baseline_test` /
    `f1_regen_test` keys preserved as aliases.
  - `EMBlockingCommitteeRunner.run` rewritten symmetrically for blocking.
  - `_load_labelled_split` lifted to a module-level helper so both
    runners share the same regen + direction-tolerant fallback.
- [lib/monotonicity.py:_BEST_MEMBER_METRIC](../usecases_synthetic/lib/monotonicity.py):
  EM `f1` / `pair_recall` chain now leads with `*_variant_model_on_regen_test`,
  falls back to the legacy `f1_regen_test` / `pair_recall` aliases.
- [config/knob_expected_signals.yaml](../usecases_synthetic/config/knob_expected_signals.yaml):
  8 EM monotonicity signals switched to
  `aggregated.macro_f1_variant_model_on_regen_test`; header comment
  rewritten with the R7b rationale.
- [scripts/validate_variant.py](../usecases_synthetic/scripts/validate_variant.py):
  per-pair CSV writer emits the 4-cell matrix + `variant_model_distinct`
  flag; stage-summary metric picker prefers the new R7b keys with
  legacy fallback; log lines reflect the new headline names.
- [scripts/build_statistics.py](../usecases_synthetic/scripts/build_statistics.py):
  `_STAGE_AGG_KEY` switched to the variant-model headline; reader
  falls back to the legacy alias keys for pre-R7b outputs.

**Tests:**
- [tests/test_committee_em.py](../usecases_synthetic/tests/test_committee_em.py):
  10 new tests across `TestResolveVariantCheckpointPath` + `TestResolveVariantTrainPath`
  covering the helper resolution logic (baseline level, variant level
  with/without sibling, every level, reverse-direction CSV).
- All 187 tests pass across `test_committee_em` (58),
  `test_variant_loader` (13), `test_domain_value_norm` (22),
  `test_pool_builder_buckets` (14), `test_prepare_em_training_data` (7),
  `test_ditto_matcher` (15+), `test_monotonicity` (52+).

### R7b — what's deferred to R7c (now folded into R10-G, mandatory)

**2026-05-29 update**: R7c is no longer an optional per-domain
follow-up. It has been folded into R10 as **R10-G**, mandatory for
every domain's published step-5 numbers. See R10-G for the current
scope, sequencing, scripts, and cost. The historical R7c framing
below is preserved for context but the load-bearing decisions live
under R10-G.

R7c is **per-domain**, gated on that domain's variant generation —
the train data R7c needs (`<pair>_train_corner_filled.csv`) only
exists after step 5/6 runs `generate_variant.py` for the domain. Not
a batched sweep across all 4 domains.

- **Per-variant training script (write once, reuse per domain)**:
  `scripts/ditto/retrain_variant.py` + `scripts/sc_block/retrain_variant.py`
  that take `--domain` + `--level` and:
  1. Load `<pair>_train_corner_filled.csv` for every source pair of
     the domain from `usecases/<domain>-augmented/<level>/input/entitymatching/`.
  2. Pool + dedupe, emit a level-specific json.gz / SupCon train set.
  3. Train via the existing trainer scripts with R2-winner hyperparameters.
  4. Write checkpoint to `usecases_synthetic/cache/<model>_checkpoints/<domain>/variant_<level>/best`.
  Scripts written once; invoked separately per (domain, level) after
  the domain's variant lands.
- **Actual retraining cost** per domain: ~30-60 min × 3 levels × 2
  trainable matchers (Ditto + sc_block) = **3-6 hours MPS per domain**.
- **R7 prerequisite**: variant retraining inherits the trainer-padding
  bug unless R7-baseline-finish for the domain has landed first. The
  recommended order per domain is:
  ```
  R7-baseline-finish for domain (optional, pre-variant-rerun)
    └→ variant generation (step 5 for products / step 6 for music+games)
       └→ initial validate_variant.py (variant cells alias baseline)
          └→ R7c retrain Ditto + sc_block on corner_filled CSVs
             └→ validate_variant.py rerun (dual-model signal live)
  ```

### R7b sequencing relative to step 5/6

**Superseded by R10-G (2026-05-29).** The per-domain choice framing
below is historical. Under R10-G every domain's step-5 cascade
unconditionally invokes the retrain step between regen and final
validate, so every published step-5 number carries the load-bearing
`variant_model_on_regen_test` signal. See R10-G "Phase 2" for the
canonical per-domain cascade sequence.

The R7b dual-model infrastructure is in place; variant cells in
`validate_variant.py` alias the baseline cells until R7c trains the
variant-specific checkpoint for that domain. The initial variant +
validation run **completes either way** — R7c is the optional
follow-up that unlocks the dual-model signal.

Per-domain decision after each rerun:

- If the initial validation surfaces clean monotonicity on
  `baseline_model_on_regen_test` (the pre-R7b legacy headline,
  still produced), R7c is purely an upgrade to a stronger
  load-bearing metric and can be deferred or skipped.
- If the initial run is ambiguous on monotonicity (e.g. the
  baseline-trained Ditto generalises poorly to corner_filled
  test), R7c is the right next move — variant-trained Ditto
  isolates intrinsic data difficulty from transfer gap.

R7c is therefore a **per-domain choice** based on that domain's
post-variant signals, not a global "do all 4 at once" sweep.

---

## R8 — Final-pass rerun: expand committee field scope to the full source schema (2026-05-28)

**Status (2026-05-29): superseded by R10-I.** The "separate cascade
after step-5 publishes" framing was overscoped — the only piece of
R8 that doesn't already happen inside R10 / Phase 2 is the EM matching
+ blocking YAML config edit. That edit lives in **R10-I**; under R10-I
the wide-scope EM committees are in place before R9 sweeps fire, and
Phase 2 publishes wide-scope step-5 numbers directly without needing a
second cascade. The historical R8 framing below is preserved for
context; the live decisions live under R10-I.

**Original status (queued, to land after the current step-5 cascade
on the 9-attribute committee scope).**

**Background.** Both `ditto_plm` and `magellan` in the EM matching
committee are currently configured to a **9-attribute subset** of the
27-column products source schema:

```
title, brand, description, product_type, model, model_number,
chipset_name, vram_gb, storage_gb
```

The 18 excluded columns (`bus_type, color, form_factor, height_mm,
interface_type, length_mm, memory_type, read_speed_mb_s,
storage_connection_type, weight_g, width_mm, write_speed_mb_s,
title_description, priceCurrency, url, cluster_id, id, price`) are
dropped by both matchers via config — Ditto's trainer projects to its
`--fields` set, Magellan's auto-feature generator scopes to
`attributes` in its committee YAML. The original rationale was that
sparse columns (some <30% coverage) would inject noise without adding
match signal.

User direction 2026-05-28: this design choice is **not** the
end-state. After the current cascade lands, we want to validate the
full-attribute alternative — every available source attribute fed to
every matcher independently — as a separate, final rerun.

**What R8 changes.**

1. **Ditto**: re-prep + retrain per domain with `--fields` expanded
   to every attribute present in the per-source DataFrames. Per
   domain:
   - products: 9 → 27 fields
   - music / games / companies: domain-specific full schema; numbers
     TBD when R8 starts (each domain's source DataFrames have
     varying column counts).
   The new checkpoints write to
   `cache/ditto_checkpoints/<domain>/sweep_full_schema/run_<ts>/`
   and the committee symlink at
   `usecases_synthetic/cache/ditto_checkpoints/<domain>/best` repoints
   to the R8 winner. Training cost: ~30-60 min × 4 domains MPS.
2. **Magellan**: extend `attributes` in
   `em_matching_committee_<domain>.yaml` to the full per-domain
   schema; extend `numeric_attributes` to flag every numeric
   per-domain attribute. Re-run the
   `_tune_em_matching_committee.py` sweep on the wider feature
   surface; winners overwrite the YAML's `classifier_params` +
   `threshold`. Sweep cost: ~5-15 min per domain (no LLM).
3. **EM blocking**: blockers are mostly parameter-free or already
   trained, but `bm25_charngram` `ngram_range` + `top_k` could be
   re-tuned on the wider schema if a separate
   `_tune_em_blocking_committee.py` exists or is authored. Decide
   at R8 start whether to run.
4. **Fusion + Norm**: untouched by R8 — C12 fusion + norm members
   already see the full canonical attribute set (per the
   `attribute_classes` block in each `config/domains/<domain>.yaml`).
   The 9-attribute scope was an EM-committee-only constraint.
5. **measure_baseline + validate**: rerun across all 4 domains under
   the new wide-scope committees. Compare committee macro_f1
   (with-llm) to the present 9-attribute baselines side-by-side;
   the wider scope is expected to be at least competitive and may
   lift Ditto's F1 if the sparse columns carry signal.

**Why this is deferred, not done now.**

The step-5 cascade has been calibrated against the existing
9-attribute committee scope (knob YAMLs sized to attribute counts,
silver standards built against the canonical_schema, variant
monotonicity signals validated against this surface). Switching
schema mid-cascade would invalidate every silver standard, force a
products silver rebuild, and re-open the K2 cap / refill tuning
question. Cleaner sequence: finish the cascade on the present scope,
publish those numbers as the "narrow committee" baseline, then run
R8 as a clean "wide committee" delta on top — both numbers reported
side-by-side in the final R7.3 narrative.

**Order of operations under R8 (when it lands):**

1. Rebuild silver standards for all 4 domains against the wider
   canonical_schema (so the K1/K6 closeness check protects the
   expanded surface under intact-cluster silver).
2. Re-prep + retrain Ditto per domain.
3. Re-sweep Magellan per domain.
4. `measure_baseline` per domain.
5. `validate_variant` per (domain, level) — variants don't need
   regeneration; the schema expansion only affects which committee
   members fire on which columns.
6. R7.3 final report extension: dual table — 9-attribute committee
   vs full-schema committee F1 + per-attribute breakdowns.

---

## R9 — Per-domain committee HPO sweep cascade (2026-05-28)

**Status**: products run **in progress** during step 5; the matching
sweeps for music / games / companies are queued behind the products
step-5 measure_baseline + validate cascade. Authored after the user
audit 2026-05-28 of which committee hyperparameters are currently
tuned vs hand-set.

**Context — what HPO state every domain is in entering step 5.** Step
4j deleted the cached results of every committee sweep except SM, so
the C12 YAMLs ship with: SM hyperparameters from the 2026-05-08 sweep
(KEPT), EM matching `magellan` hand-set (R5 lock for the other 3
members), EM blocking parameters carried from pre-C4 YAMLs (the C4
char-ngram switch on `bm25_charngram` was applied YAML-only, no
re-sweep), fusion + norm parameters as-shipped post-C12 restructure
with no sweep. The grids themselves are still present in
`_tune_*_committee.py` script source — only the result caches were
purged.

**Products R9 sequence (this session, after Magellan sweep + before
measure_baseline)**:

1. **Magellan sweep** — currently running across all 4 domains
   (`_tune_em_matching_committee.py --domains products,companies,games,music`).
   Apply winners to all 4 YAMLs when the sweep lands (this part is
   in-scope here; cheap to do everyone in one pass since the LLM
   cost is zero).
2. **Add a products entry to
   [`_tune_em_blocking_committee.py`'s `STANDARD_KEY_PANEL`](../usecases_synthetic/scripts/_tune_em_blocking_committee.py).**
   Current panel covers companies / games / music only.
   Suggested products keys: `[{type: prefix, column: title, n: 5}]`,
   `[{type: token, column: title}]`,
   `[{type: compound, parts: [{type: token, column: title}, {type: value, column: product_type}]}]`.
3. **EM blocking sweep** — run `--domains products
   --sub-sweeps embedding,standard,sn,sc_block`. Embedding + SNB +
   sc_block require no additional config (sc_block uses the existing
   trained checkpoint at
   `usecases_synthetic/cache/sc_block_checkpoints/products/best`,
   sweeping `top_k × threshold` only). Token + BM25 deliberately
   skipped per the script's docstring (line 21).
4. **Fusion sub-sweep smoke-test on companies (1 sub-sweep)** —
   verify `_tune_fusion_committee.py` produces sensible cells under
   the C12 fusion YAML shape before launching the full grid. Pick
   the cheapest sub-sweep (e.g. `_sub_tolerance` — domain-level, no
   LLM). If cells emit, schedule the full fusion sweep for products.
   **VERIFIED 2026-05-28**: `_sub_tolerance` works under C12; other
   9 sub-sweeps are broken (see step 5).
5. **Fusion sweep — products only — TOLERANCE ONLY for now.**
   The full grid is 10 sub-sweeps, but the post-C12 audit
   2026-05-28 found that 9 of them are NOT compatible with the
   C12 `members:` YAML shape. The script's `_mutate_td_params` at
   [_tune_fusion_committee.py:131](../usecases_synthetic/scripts/_tune_fusion_committee.py#L131)
   iterates `mutated["attributes"].values()` — assumes the pre-C12
   YAML had a top-level `attributes:` dict mapping attribute name to
   strategies list. The C12 restructure replaced that with a
   `members:` list (each member is a coherent end-to-end approach).
   Affected sub-sweeps that cannot run as-is: `_sub_trim`,
   `_sub_list_threshold`, `_sub_truthfinder`, `_sub_accusim`,
   `_sub_casefusion`, `_sub_fusionquery`, `_sub_ltm`,
   `_sub_llm_judge` (and `_sub_trust` — needs a re-check). Only
   `_sub_tolerance` is C12-safe because it operates on
   `evaluation_params` / `fusion_protection_tolerance` keys that
   C12 preserved.

   **Refactor required for R9 to deliver a full fusion sweep**:
   rewrite `_mutate_td_params` (and the `_disable_llm_judge` helper
   at `_tune_fusion_committee.py:144` which has the same shape
   assumption) to traverse `mutated["members"]`, find the C12 member
   whose name matches the swept method (e.g. `accusim_only`,
   `truthfinder_only`), and patch its `params:` block. Each member's
   internal config block also needs inspection — C12 collapsed the
   per-(attr, strategy) shape into a per-(member, fallback) shape,
   so a single sub-sweep cell now translates to "one member, one
   set of params" rather than "one attribute, one strategy". This is
   a ~2-3 hour refactor + reverification on each sub-sweep. Defer
   for now; run only tolerance for products under R9 step 5.

   `_sub_llm_judge` carries LLM cost (~$5-15 on products); skip it
   unless the user authorises the spend.
6. **Norm sweep smoke-test on companies** — same shape verification
   under C12 norm YAML (3 coherent members: `rule_per_attribute_optimal`,
   `llm_only`, `passthrough`). If `SPECS` references members that
   no longer exist, the script needs updating before it can run; in
   that case **skip norm sweep entirely for now** (the C12
   `rule_per_attribute_optimal` member already does runtime
   val-selection, and the other two are parameter-free or have only
   prompt-level tunables).
7. **Apply winners** to products YAMLs (em_blocking, fusion, possibly
   norm).
8. **THEN** measure_baseline products → validate_variant per level
   → publish step 5 numbers.

**Deferred to a future R9 pass (music / games / companies)**:

**Blocked by R10** (added 2026-05-28). The cross-domain R9 sweeps
must NOT start until R10-C lands, because R10-C refreshes every
domain's baseline SM (and Norm) gold from the legacy small-attr scope
to the full R1 scope. Running the sweeps now would lock in winners
against the stale scope, then require a redo after R10-C anyway. See
R10 "Order vs R9" for the sequencing rationale.

After products step 5 lands, the same sweep cascade has to be run on
the other 3 domains before their measure_baseline + validate runs
are methodologically sound. Specifically:

- Apply the Magellan sweep winners to music / games / companies
  YAMLs (this is done already as part of step 1 above; the only
  thing missing is the *other* domains' measure_baselines).
- Add per-domain `STANDARD_KEY_PANEL` entries for music / games /
  companies if blocking-key adjustments are wanted, OR confirm the
  existing entries still hold (the panel was authored pre-R6
  Ditto retrains; nothing about R6 invalidates blocking-key choices).
- EM blocking sweep — `--domains music,games,companies`. Same
  4 sub-sweeps; same caveats.
- Fusion sweep — same 10 sub-sweeps, same smoke-test recommendation
  before launching all 3 domains.
- Norm sweep — same C12 smoke-test before commitment.

Cost rough estimate (no LLM unless an LLM-backed sweep cell is
explicitly selected):

| Domain | EM blocking | Fusion | Total local-CPU | LLM cost |
|---|---:|---:|---:|---:|
| products (this session) | 15-30 min | 10-20 min + LLM judge | 25-50 min | $5-15 (LLM judge sub) |
| music | 15-30 min | 10-20 min + LLM judge | 25-50 min | $5-15 |
| games | 15-30 min | 10-20 min + LLM judge | 25-50 min | $5-15 |
| companies | 15-30 min | 10-20 min + LLM judge | 25-50 min | $5-15 |

**LLM judge sub-sweep is optional** — if cost is a concern, skip
`--sub-sweeps llm_judge` on the fusion sweep; the other 9 sub-sweeps
are all local-only.

**Order vs R8** (full-schema rerun): R9 and R8 are independent.
R9 finds optimal hyperparameters under the present 9-attribute
committee scope; R8 expands the scope to all 27 columns (or
whatever the per-domain full schema is) and would require a
separate sweep cycle on the wider feature surface. Natural
sequence: R9 → publish step 5 numbers per domain → R8 → publish
delta. R9 winners may not transfer to the wider R8 scope (n_estimators
etc. likely shift), so R8 will need its own sweep pass at that point.

---

## R10 — Methodology hardening before the next full variant regen (2026-05-28)

**Status**: products step-5 cascade landed (R9 winners applied, baseline
+ easy/medium/hard validated 2026-05-28). Surfaced three methodology
issues that must be fixed **before** the music / games / companies
variant regens land, otherwise their published gradients will inherit
the same artifacts. R10 is the gate between products step-5 publication
and the cross-domain variant cascade.

### R10 — what surfaced during products step-5

Products fusion gradient is clean and monotone
(0.8220 → 0.7000 → 0.6420 → 0.5765 across baseline / easy / medium /
hard, deltas -0.122 / -0.058 / -0.066). EM blocking + EM matching are
**non-monotone** (medium > easy):

| stage | baseline | easy | medium | hard |
|---|---|---|---|---|
| EM blocking macro_pair_recall | 0.8603 | 0.3708 | 0.4108 | 0.1497 |
| EM matching macro_f1 | 0.9608 | 0.6652 | 0.6893 | 0.4508 |

SM / Norm score *above* baseline at every variant level
(SM 0.66 / 0.58 / 0.51 vs baseline 0.47; Norm 0.44 / 0.40 / 0.34
best-member vs baseline 0.48). Three distinct root causes — items A, B,
C below — none of them an outright bug, all worth fixing before the
next regen cycle.

### R10-A — Nested perturbation sets across levels (refactor) — landed 2026-05-29

**Status: landed 2026-05-29.** Implemented as **Option B (cell-set
nesting only)**, **scoped to K1 + K6** — both per-user decisions
2026-05-29 (see "Step R10-A landed" block below). The literal
"committed value = hard value" wording in the original Fix was *not*
adopted; instead the *set* of perturbed cells nests across levels while
each level keeps its own per-level operators/values. K5 is excluded
(no per-cell rate gate; its gradient is format diversity, and it never
touches text blocking keys, so it is not a driver of the medium>easy EM
anomaly). The LLM cache re-key (Fix item 2) is **moot under Option B**:
`llm_cache_k1` is instantiated only at `level == "hard"`
([apply_values_joint.py:453](../usecases_synthetic/scripts/apply_values_joint.py#L453)),
so there is no cross-level paraphrase reuse to optimise.

**Cause.** K1 / K5 / K6 reseed their cell-selection RNGs per level and
sample independently — there is no cumulative-cell invariant across
{easy, medium, hard}. A cell can be more perturbed at medium than at
hard if the level-medium sampler happens to hit a high-information
field that the hard sampler skipped. Direct evidence on products
`products_2_1830308`:

- medium: `title` → "HDD WD 1T.B Passport USB3-Black" (paraphrased + K6 dot);
  `priceCurrency` → "BDT" (was USD); `price` → 70.49 (was 59.99).
- hard: `title` → original "WD My Passport 1TB USB 3.2 ..."; `price`
  → 59.99 (original); only K8 renames + K5 `storage_gb` → "1000000 MB"
  fire on this row.

When per-row perturbation isn't nested, blocking-key-bearing cells
(`title` + `brand`) can survive at hard while being corrupted at
medium, producing the medium > easy non-monotonicity on EM stages
(blocking-time recall + matching-time F1 both correlate strongly with
title-field quality, while fusion's per-attribute val-best selection
averages over the whole frame and stays monotone).

**Fix.** Plan the **hard** perturbation once per (knob × cell), then
take cumulative subsets at lower levels so easy_cells ⊂ medium_cells ⊂
hard_cells and the committed value at each level matches the hard
value for cells in that level's subset. Concretely:

1. For K1 / K5 / K6 the runner already produces a provenance row per
   perturbed cell at every level. Refactor `apply_values_joint.py` so
   the cell sampler is driven by a single "hard plan" generated once,
   then masked down per level (easy = X% of hard rows by deterministic
   shuffle; medium = Y% with easy ⊂ medium ⊂ hard).
2. Cumulative semantics also need plumbing through the LLM caches:
   K1's `llm_cache_k1` is currently keyed by `(domain, level, cell)`;
   under R10-A it would be keyed by `(domain, cell)` with a single
   value used at every level the cell is included in. Avoids
   re-paying for the same paraphrase at higher levels.
3. K3 (attribute drop) and K2 (entity drop) are already cumulative by
   construction (higher-level targets strictly dominate lower-level
   ones in the share schedule), and K8 / K10 produce per-level outputs
   that are level-monotone by config — no refactor needed there.

**Estimated effort**: ~1 day of work in `apply_values_joint.py` +
`apply_knob_0{1,5,6}.py` + cache key migration + a fresh
test_apply_values_joint.py asserting `easy_cells ⊂ medium_cells ⊂
hard_cells`. Should land before any further variant regen runs to
avoid republishing non-monotone EM gradients.

**Step R10-A landed 2026-05-29:**

Two design questions were put to the user before coding, because the
original one-line Fix does not match the code reality:

- **Value model → Option B (cell-set nesting only).** The plan's
  "committed value = hard value" would have collapsed per-cell
  intensity to hard everywhere (only the *fraction* of cells would vary
  per level), making per-level `operator_mix` + the R10-D per-cell
  intensity tuning hard-only. The user instead chose to nest only the
  *set* of perturbed cells and keep each level's own operators/values.
  This is the lighter, less invasive fix and preserves R10-D. Residual
  caveat (accepted): a cell selected at both medium and hard can still
  draw a *harsher* operator at medium than at hard (the operator draw is
  level-keyed); cell-set nesting alone does not forbid that. The
  dominant cause — independent per-level *sampling* — is eliminated.
- **Scope → K1 + K6 only.** K5 has **no per-cell rate gate**: it
  reformats every managed cell at every level
  ([apply_knob_05_format.py:333,412-415](../usecases_synthetic/scripts/apply_knob_05_format.py#L333)),
  its gradient is format *diversity* (`within_source_consistency`:
  source-wide vs per-row), and it only handles money/date/file_size/rate
  families — never text blocking keys. Value-sharing would erase its
  gradient, and it is not a driver of the EM anomaly, so it is excluded.
  K1's `easy` is separately fine: `paraphrase_rate_*.easy = 0.0` in every
  domain YAML, so `easy` is the canonicalize-toward-target pass only and
  the paraphrase nesting reduces to `medium ⊆ hard`.

Mechanism (mirrors K3's shared-uniform pattern, but keyed by cell
identity rather than row position so nesting survives K2/K4 dropping
different rows per level):

- New
  [lib/rng.py:cell_selection_uniform](../usecases_synthetic/lib/rng.py)
  `(domain, source, entity_id, attribute, knob, master_seed) -> float`:
  deterministic, **level-independent** uniform in `[0, 1)` via
  `sha256(domain|source|entity_id|attribute|knob|select|seed)`. Because
  the per-level target rates are monotone non-decreasing and the uniform
  does not depend on level, selecting a cell when `u < rate[level]`
  yields `easy ⊆ medium ⊆ hard` by construction.
- [apply_knob_01_surface.py:980](../usecases_synthetic/scripts/apply_knob_01_surface.py#L980)
  and
  [apply_knob_06_noise.py:767](../usecases_synthetic/scripts/apply_knob_06_noise.py#L767):
  the per-row gate `if col_rng.random() >= target_rate: continue` is
  replaced with `if cell_selection_uniform(...) >= target_rate:
  continue` (knob=1 / knob=6). The level-keyed `col_rng` still drives the
  operator + parameter draws below the gate, so per-level operators are
  retained (Option B). Distinct `knob` keys make K1 and K6 select
  uncorrelated cells.
- `apply_values_joint.py` needed **no change** — the gate fix lives
  inside each knob; the cache instantiation is already hard-only.

Tests (placed in the canonical existing files rather than a new
`test_apply_values_joint.py`, which would duplicate the existing
[tests/test_joint_values.py](../usecases_synthetic/tests/test_joint_values.py)):

- 8 new `TestCellSelectionUniform` cases in
  [tests/test_rng.py](../usecases_synthetic/tests/test_rng.py):
  determinism, `[0,1)` range, no `level`/`variant` parameter,
  knob/identity/seed divergence, ~uniform distribution, and the
  load-bearing `easy ⊆ medium ⊆ hard` selection-set property.
- 4 new `TestLevelNesting` cases in `test_joint_values.py`: end-to-end
  through the real `apply_knob_01` / `apply_knob_06` dispatchers
  (`entity_groups=None`, `collision_index=None` so `provenance ∪ skipped`
  == the gate set regardless of operator success). Asserts K6
  `easy ⊆ medium ⊆ hard` (strict growth), K6 equal medium/hard rates →
  *identical* set (regression guard against per-level resampling), K1
  `easy ⊆ medium ⊆ hard`, and K1/K6 select independent cells.

Verification: 124/124 K1+K6 tests pass (determinism preserved); 29/29
rng+joint_values; full synthetic suite **1451 passed, 7 skipped, 2
failed** — the 2 failures are the pre-existing
`test_fusion_silver_standard.py::TestSupportedDomains` stale 3-domain
assertions documented under R10-D, unrelated to R10-A. `black` clean;
`mypy` clean on `lib/rng.py`. **Empirical confirmation** that the EM
gradient becomes monotone is deferred to the products regen in the R10
cascade (needs R10-F/G/C/I + folder deletion + LLM auth first).

**Cache deletion**: NONE required for R10-A. The selection change is
RNG-only (no cache artifacts); the K1 LLM cache key/path is untouched.

### R10-B — K2 easy realised = 0.477 (configured 0.200) is a protection-floor artifact, not a knob bug

**Investigation.** Products K2 baseline corner ratio is **0.828**
(1655 / 2000 candidate pairs already qualify as corner cases pre-
operator) — way above easy's target 0.200. The operator dispatcher
goes into the `drop_corner_touching_refilled` branch, drops 396
entities (~34% of the canonical surface), and floors out at realised
ratio 0.477 because the remaining corner-touching entities are either
protected (gold + intact-silver per [[C13]]) or last-of-collision-
group (cannot drop without making a label collision unresolvable).

The `max_interp_fraction = 0.6` ceiling (`interp_budget=0
(max=690)` in the regen log) is irrelevant here — we never
interpolate, we drop. So this is the artifact the user predicted:
fusion / silver protection caps how many entities K2 can shed, and
products happens to start so corner-heavy that easy's target is
unreachable.

**Decision.** Accept the floor; no knob change required. R10-B is
the *characterization* sub-item — its only deliverable is to update
the monotonicity-audit docstring + the K2 README to flag the
expected easy-overshoot for domains whose baseline ratio sits far
above the easy target. Mention products as the concrete example.
The other 3 domains' baselines should be re-checked before their
regens land; if any of them also sit well above their easy target,
the same floor will surface in step-5 numbers and we should not
treat it as a regression.

### R10-C — Committee measurement-scope equivalence across baseline + variant — landed 2026-05-29

**Status: landed 2026-05-29.** SM gold refreshed for **products only**
(24 → 80 rows); companies / games / music were already full-scope (the
generator confirmed byte-equivalence — they never had the products R1
schema upgrade). Norm / EM / fusion verified scope-equal by construction
— no change. See "Step R10-C landed" block at the end of this section.

**Cause.** Baseline measurement loads from
`usecases_synthetic/usecases/<domain>/input/...` and scores SM on
`sm_mapping_gold.csv`, which on products lists only the 6 original
attrs `id / title / brand / description / price / priceCurrency`
across 4 sources (25 rows total). Variant SM is scored on the
K8-generated `sm_mapping.csv` covering all 20 R1 attrs across 4
sources (81 rows). Variant per-partition `n_columns=20` vs baseline
`n_columns_baseline=6` — the macro_f1 denominators differ by 3.3x,
so the headline baseline-vs-variant SM delta is meaningless.

The extra 14 R1 attributes (model, model_number, product_type,
chipset_name, ..., write_speed_mb_s) are mostly trivial for SM members
to score: they exist in only some sources, they're typed numerics, and
their instance distributions are sparse enough that several SM members
hit either 1.0 or 0.0. This pads variant macro_f1 vs baseline and
explains why variant SM > baseline on products despite K8 making the
SM task strictly harder.

**Fix — SM**. Refresh every domain's `sm_mapping_gold.csv` so it covers
the same attribute set that the K8 `rename_table` perturbs. Concretely
for products: extend gold from 6 to 20 attrs per source (the K8
rename_table's full key set) — 81 rows total, matching the variant
side. After the refresh the baseline-vs-variant SM macro_f1 delta
becomes apples-to-apples.

**Fix — Norm**. Same root cause, same fix path: baseline scores Norm
on whatever the per-attribute norm gold covers. If it's the 5-attr
original schema, refresh to the full 20-attr R1 schema. Verify by
inspecting `n_attributes_baseline` vs `n_attributes` in the per-stage
JSON.

**Fix — EM blocking / matching**. Verify, don't refactor. Both stages
already train per-pair on whatever columns the source frames carry;
the question is whether the input columns are identical baseline-vs-
variant. Products baseline source JSON has all 27 R1 columns, and the
variant CSV has the same 27 cols (post K8-rename). So input feature
*availability* is identical; the only divergence is K8 renames at
medium / hard, which the runner already handles via the K8 sm_mapping
applied before blocking. Spot-check the blocking-time `Source columns
for matching` log line at each variant level to confirm — if any
column unexpectedly absent at variant time, that's a runner bug
worth investigating.

**Fix — Fusion**. Already aligned: the fusion gold XML
(`validation_set.xml` / `test_set.xml`) is *copied* by `package_variant`
from baseline to variant, so the fusion attribute set is identical at
every level. Products fusion gold has 5 attrs (the original
`title / brand / description / price / priceCurrency`); the data_cleaned_final
additions stay declared but dead in `fusion_committee_products.yaml`
until R8 lands. No R10 work needed for fusion.

**Estimated effort**: 1-2 hours per domain to draft the refreshed gold
mappings + spot-check the per-stage JSON to confirm `n_attributes`
agree. Land before the cross-domain regen cascade so step-5 numbers
publish with comparable SM / Norm scopes everywhere.

**Step R10-C landed 2026-05-29:**

*SM (the only data change).* New
[scripts/build_sm_mapping_gold.py](../usecases_synthetic/scripts/build_sm_mapping_gold.py)
derives the baseline `sm_mapping_gold.csv` from the *same*
`config/knob_08_naming/<domain>.yaml` `sm_mapping` ground-truth block
that [apply_knob_08_naming.py](../usecases_synthetic/scripts/apply_knob_08_naming.py)
generates the variant `sm_mapping.csv` from — using the original
(un-renamed) source column names — so the two sides are
scope-identical by construction. A dry-run diff across all four domains
showed:
- **products**: 24 → **80** rows (4 sources × 20 attrs). Purely additive
  (the 14 R1 `data_cleaned_final` attrs × 4 sources = 56 new rows; **zero**
  dropped). Regenerated on disk
  ([usecases/products/input/schemamatching/sm_mapping_gold.csv](../usecases_synthetic/usecases/products/input/schemamatching/sm_mapping_gold.csv)).
  Verified the baseline products sources carry all 20 columns, so the SM
  members can score the full set.
- **companies (21) / games (27) / music (27)**: generator output is
  byte-equivalent to the on-disk gold (no R1 upgrade → already
  full-scope). Left untouched to avoid churn; the generator is now the
  documented way to rebuild any domain's gold.

*Norm (verified, no change).*
[committee_norm.run](../usecases_synthetic/lib/committee_norm.py) scores
the attribute set drawn from `load_fusion_target_values(domain)` — the
**frozen** fusion XML, which `package_variant` copies byte-identically to
every variant level. So baseline and variant Norm score the same
attributes (products: the 5 fusion attrs) by construction; `n_attributes`
already agrees. Confirmed the SM-gold refresh does **not** change Norm's
scored set — Norm only scores `fusion_targets ∩ sm_mapping`, and all 5
fusion attrs were already in the 6-attr gold, so the baseline Norm number
is unchanged by the refresh.

*EM (verified, no change).* Input feature availability is identical
baseline-vs-variant — both load the full per-domain source schema; the
only divergence is K8 renames at medium/hard, which the runner already
maps back via the K8 `sm_mapping`. The blocking-time "Source columns for
matching" log line is the spot-check at the regen cascade.

*Fusion (no change).* Frozen XML, copied to every variant level.

Tests: new
[tests/test_build_sm_mapping_gold.py](../usecases_synthetic/tests/test_build_sm_mapping_gold.py)
(5) — products full 80-row scope, R1 attrs present under original names
(not K8 tokens), matches the K8 ground truth exactly, the 3 other domains
are content-equivalent to disk, and the committed products gold is the
80-row version. Full synthetic suite **1479 passed, 7 skipped, 2 failed**
(the 2 pre-existing `TestSupportedDomains` stale assertions). `black`
clean. **No cache/folder deletion** for R10-C.

**Note**: the refreshed products baseline SM number will be re-emitted at
the products step-5 re-publish (the gold is wider now); this is the
intended apples-to-apples correction, not a regression.

### R10-D — K1 paraphrase prompt strengthening (`v2`) — landed 2026-05-29

**Status: landed 2026-05-29.** All R10-D code + tests in place; v2
prompts on disk; all four K1 domain YAMLs pin `llm_prompt_version: v2`.
85/85 K1 tests green. Pre-bake of products + companies caches is the
only remaining work and folds into the per-domain Phase 2 cascade.

**Cause.** Monotonicity audit's `knob_01_realised_intensity_monotonicity`
FAIL details: edit_distance hard 0.1395 < medium 0.1641 (non-monotone);
jaccard_drop hard 0.0907 < medium 0.0975. The detail string flags
"shallow paraphrases — rate fires but LLM/operator output is near-
identity (casing, trivial reorder)".

The current prompt (`prompt_short_v1.txt`) lets the LLM return the
input unchanged when "no plausible paraphrase exists", which the model
is using too liberally; combined with no minimum-divergence
constraint, ~30-40% of K1 calls produce near-identity outputs at hard.

**Fix — author `prompt_short_v2.txt` + `prompt_categorical_v2.txt`**.
Concrete changes:

1. Mandate minimum divergence: at least 1 substantive token change
   beyond whitespace / casing / punctuation. Reject zero-change outputs
   server-side too (post-filter on `token_jaccard(input, output) < 1.0`
   and re-prompt or fall back to a non-LLM operator).
2. Replace the "return unchanged if no plausible variant exists" rule
   with a stronger escape: "If no plausible alternate surface form
   exists, return the literal string `<UNCHANGED>` (not the input)" so
   we can distinguish LLM-judged-unparaphrasable from LLM-laziness in
   provenance, then count `<UNCHANGED>` rate per (domain, level,
   attribute) as a calibration signal.
3. Add 2-3 worked examples (GOOD vs BAD) per attribute class —
   primary (short title), key (categorical brand / genre), secondary
   (longer description). Use real examples taken from the source
   data_cleaned_final files. Examples are the strongest prompt lever
   on near-identity drift.
4. Clarify cross-language rule: the v1 rule says "no cross-language
   translation" but products hard produced Polish paraphrases (e.g.
   `products_2_569608: "Dysk SSD M.2 Silicon Power A80 1TB PCI-e"`).
   Either tighten ("stay in the input's primary language") or relax
   explicitly to allow code-switching when the underlying product is
   marketed internationally; current ambiguity lets the LLM drift in
   ways we did not specify.
5. Per-attribute-class operator mix — the current v1 prompts are
   identical across primary / key / secondary; v2 should be split
   so secondary descriptions (which can paraphrase much more
   aggressively without entity drift) don't share the conservative
   primary-name rules.

Wire `llm_prompt_version: v2` in every domain's `knob_01_surface/<domain>.yaml`.
Document the cache invalidation: switching `prompt_version` rotates the
cache key under `usecases_synthetic/cache/llm_paraphrase_*/`.
Pre-bake the products + companies caches under v2 before the regen
cascade so the run itself doesn't pay the LLM cost.

**Estimated effort**: half a day to author the v2 prompts +
post-filter + provenance counter; another half day to pre-bake the
caches across all 4 domains.

**Step R10-D landed 2026-05-29:**
- New: [config/knob_01_surface/_prompts/prompt_short_v2.txt](../usecases_synthetic/config/knob_01_surface/_prompts/prompt_short_v2.txt)
  — primary/key class: minimum-divergence rule + `<UNCHANGED>` sentinel
  + GOOD/BAD examples (Gigabyte RTX 3080, AMD Radeon, Pink Floyd, Sony) +
  explicit "stay in input's primary language" + no-translation rule.
- New: [config/knob_01_surface/_prompts/prompt_categorical_v2.txt](../usecases_synthetic/config/knob_01_surface/_prompts/prompt_categorical_v2.txt)
  — categorical class: same divergence/escape/language rules + GPU/SSD/HDD/
  IT-Services examples.
- New: [config/knob_01_surface/_prompts/prompt_secondary_v2.txt](../usecases_synthetic/config/knob_01_surface/_prompts/prompt_secondary_v2.txt)
  — secondary class (descriptions): more aggressive reword allowed,
  fact-preservation guard, ±30% length bound, GPU-spec / release-date /
  storage-spec examples.
- [scripts/apply_knob_01_surface.py](../usecases_synthetic/scripts/apply_knob_01_surface.py):
  prompt loading made version-aware (`prompt_short_{version}.txt` etc.)
  with v1 → short fallback for `prompt_secondary` back-compat; per-class
  dispatch (`categorical → categorical`, `secondary → secondary`,
  `primary/key → short`); `<UNCHANGED>` sentinel + near-identity skip
  handling under reasons `llm_unchanged_sentinel` + `llm_near_identity`;
  `REALISED_COLUMNS` extended with `llm_unchanged_count` +
  `llm_near_identity_count`; `build_realised_df` counts the two new
  skipped reasons; CLI log line surfaces both.
- [lib/surface_operators.py](../usecases_synthetic/lib/surface_operators.py):
  new `UNCHANGED_SENTINEL` constant + `_is_near_identity` helper;
  `llm_paraphrase` detects sentinel (returns
  `(value, transform_fn=llm_paraphrase_unchanged)`) + near-identity
  (returns `(paraphrase, transform_fn=llm_paraphrase_near_identity)`)
  before existing contamination / committee checks; new
  `llm_paraphrase_secondary` transform_fn for the new prompt class;
  `VALID_TRANSFORM_FNS` extended with the three new tags.
- All four `config/knob_01_surface/{music,games,products,companies}.yaml`:
  `llm_prompt_version: v1 → v2` with comment block explaining the
  R10-D bump + automatic cache rotation via `sha256(... | prompt_version
  | ...)`.
- 17 new tests in [tests/test_knob_01.py](../usecases_synthetic/tests/test_knob_01.py):
  `TestNearIdentityHelper` (8), `TestLLMParaphraseV2Behavior` (5),
  `TestRealisedAuditV2Counters` (2), `TestPromptVersionDispatch` (2 —
  asserts all 4 YAMLs pin v2 + v2 prompts contain the
  `<UNCHANGED>` sentinel string). All 85 K1 tests pass; 1439 of the
  1441 synthetic-suite tests pass (the 2 failures in
  `test_fusion_silver_standard.py::TestSupportedDomains` are unrelated
  to R10-D — products was added to silver-standard support in a prior
  session without updating the legacy 3-domain assertion).

**Cache deletion**: NONE required. The K1 LLM cache directory at
`usecases_synthetic/cache/knob_01_paraphrases/` doesn't currently
exist on disk (step 4j cleanup confirmed the K1 cache is empty), so
there are no v1-keyed entries to remove. The cache key formula
`sha256(source|attribute|value|prompt_version|model_id)` rotates
automatically on the `v1 → v2` bump, so even if v1 entries existed,
they would coexist harmlessly with v2 entries under different keys.

### R10-E — K4 coverage-skew source-selection rebalancing — landed 2026-05-29

**Status: landed 2026-05-29.** Both surgical changes in `coverage_ops.py`
(per-source demotion cap + hash tie-break) + the config default 0.40 on
all four K4 YAMLs + a focused unit test. See "Step R10-E landed" block at
the end of this section. Runner-only change; no measurement-side impact,
so all four domains inherit the rebalanced K4 on the next regen.

**Cause.** K4's hard demotion plan
(`{4: 534, 3: 550, 2: 359}` for products = 1443 candidate demotions
across the three cluster-size bins, 690 actually applied after the
singleton-cap rollback) is concentrated overwhelmingly on a single
source. Products K4 hard provenance from the 2026-05-28 regen:

| source | rows removed at hard |
|---|---:|
| products_1 | 645 |
| products_2 | 31 |
| products_3 | 14 |
| products_4 | 0 |

The selection logic in
[`coverage_ops.py:726-732`](../usecases_synthetic/lib/coverage_ops.py#L726-L732)
sorts each entity's members by **descending `score_conflict`** then
alphabetically. `score_conflict` is the "most agreeable with the rest of
the cluster" metric — by design it identifies the *redundant* member.
On products, **products_1 is the EM anchor and the most cleanly
curated source**, so its values match the consensus across virtually
every cluster, scoring it highest on conflict almost every time. The
alphabetical tie-break then locks in products_1 wherever conflict
scores genuinely tie. The downstream consequence is that products_1
shrinks from 812 → 169 rows at hard while products_2/3/4 stay near
their baseline counts.

**Why this matters end-to-end (not just cosmetic).** EM gold pairs are
fixed at baseline and **not** regenerated post-K4 (only K2 regenerates
EM splits). At hard, ~80% of products EM gold edges reference a
products_1 ID. K4 deletes 645 / 812 = 79% of products_1 records, so a
large fraction of EM gold edges at hard point at entities that no
longer exist in the variant — the matcher cannot match them, and the
recall floor on EM matching at hard tanks beyond what the K1/K5/K6
perturbations alone would explain (hard EM macro_f1=0.451 vs medium
0.689 — sharper than the medium → hard step on any other stage).
"Coverage skew" as currently implemented behaves more like "ablate
products_1" on products.

**Fix — per-source demotion cap + RNG tie-break**. Two surgical
changes in `coverage_ops.py`:

1. Add a `per_source_demotion_cap` knob (default `0.40`) on the K4
   YAML. In `select_removal_candidates` track a running
   `removed_per_source: dict[str, int]` and skip a candidate source
   once its count crosses `cap * baseline_row_count_for_source`,
   falling through to the next-ranked source in the same entity's
   sorted list. If every source in a cluster hits the cap, the entity
   demotion gets skipped (logged as `skip_reason=per_source_cap`)
   rather than forcing a violation.
2. Replace the alphabetical tie-break (the trailing `s` in the sort
   key tuple) with a deterministic per-`(entity_id, source)` hash so
   genuine ties spread across sources instead of locking to the
   first letter.

**Expected effect on products hard** at `per_source_demotion_cap=0.40`:
products_1 capped at ~325 demotions; the remaining ~320 demotions
spread across products_2/3/4 (currently 31/14/0 → ~100-110 each). The
per-source row counts at hard land roughly at products_1 ≈ 490,
products_2 ≈ 710, products_3 ≈ 660, products_4 ≈ 540. Still uneven (the
conflict signal still drives the *ranking*), but no source is
decimated. EM gold edges that reference products_1 stop disappearing
at the rate they currently do, and the hard-EM-matching crater
shallows.

**Why not the alternatives**.

- **Pure quota / round-robin selection** (always rotate sources): loses
  the K4 design intent — the "drop redundant" heuristic is genuinely
  useful when one source's value is uninformative. We want the
  conflict ranking to remain primary.
- **Tie-break-only fix** (just swap alphabetical for RNG): doesn't
  help — products_1's conflict scores aren't *tied* with others, they
  dominate. Tie-breaking the few residual ties would only shift a
  handful of demotions.
- **Regen EM gold post-K4**: tempting but expensive — K2 already
  regenerates EM splits and the regen pipeline is the long pole. We
  can avoid the EM gold staleness by keeping products_1 entities
  alive instead, which the per-source cap does for free.

**Estimated effort**: ~2 hours to implement + a focused unit test on
the cap behavior + a config-default decision (start at `0.40` and
adjust if R10-A's nested-perturbation pass changes the variant-quality
signal). Existing K4 tests should pass unchanged — the cap is a no-op
until the threshold is crossed.

**Step R10-E landed 2026-05-29:**

- [coverage_ops.py:select_removal_candidates](../usecases_synthetic/lib/coverage_ops.py)
  gains a `per_source_demotion_cap: float = 1.0` parameter. A
  `baseline_counts = {src: len(df)}` snapshot (at K4 entry) + a running
  `removed_per_source` counter gate the inner source-selection loop: a
  source is skipped once `removed_per_source[src] >= cap *
  baseline_counts[src]`, falling through to the next-ranked source; if
  every eligible source is capped (or otherwise blocked) the entity
  demotion is skipped (logged at DEBUG as `per_source_cap`). Default
  `1.0` is a no-op (can't shed >100% of rows), so existing behaviour /
  tests are unchanged.
- New helper `_source_tiebreak(entity_id, source)` =
  `sha256(entity_id|source)`; both the fusion-closeness and the
  conflict-ranking sort keys now use it instead of the trailing
  alphabetical `s`, so genuine rank ties spread demotions across sources
  instead of locking to the alphabetically-first source.
- [apply_knob_04_coverage.py](../usecases_synthetic/scripts/apply_knob_04_coverage.py)
  threads `per_source_demotion_cap=float(config.get("per_source_demotion_cap", 1.0))`
  into the call.
- All four `config/knob_04_coverage/{products,companies,games,music}.yaml`
  set `per_source_demotion_cap: 0.40` (verified loaded via
  `load_knob_config(4, ...)`).
- New [tests/test_coverage_demotion_cap.py](../usecases_synthetic/tests/test_coverage_demotion_cap.py)
  (5 tests): uncapped-concentrates-on-only-eligible-source,
  cap-limits-a-dominant-source (10 rows + cap 0.40 → exactly 4 aaa
  removals, rest skipped), cap-lets-remaining-demotions-spread (max
  per-source ≤ 4, all demotions still satisfied), hash-tie-break-spreads
  (≥2 distinct sources picked vs the old alphabetical lock-in), and
  tie-break determinism. 14 existing `test_knob_04` tests stay green
  (cap no-op at default). Full synthetic suite **1474 passed, 7 skipped,
  2 failed** (the 2 pre-existing `TestSupportedDomains` stale
  assertions). `black` clean.

**Cache deletion**: none for R10-E (runner-only; no cache artifacts).

### R10-F — EM dual-test plumbing fixes (package_variant glob + Ditto cache key) — landed 2026-05-29

**Status: landed 2026-05-29.** Both bug fixes + the full EM-gold
verification audit + the smoke regression test are in place. See "Step
R10-F landed" block at the end of this section. Headline: the audit
found the *active* dual-test path
([EMMatchingCommitteeRunner](../usecases_synthetic/lib/committee_em.py) +
[EMBlockingCommitteeRunner](../usecases_synthetic/lib/committee_em.py) +
[variant_loader](../usecases_synthetic/lib/variant_loader.py)) already
loads + consumes the right gold once the glob bug is fixed; the audit's
one false-positive (committee_norm row_idx skip) was rejected per the
plan's own item 2.ii ("current behavior is correct").

**Cause.** Surfaced during the products step-5 statistics audit
(2026-05-28). Two independent bugs that compound to produce identical
EM matching scores across all four dual-test surfaces and all four
levels for Ditto specifically:

1. **`package_variant.copy_regenerated_em` globs the wrong suffix.**
   At [`package_variant.py:271`](../usecases_synthetic/scripts/package_variant.py#L271)
   the function scans for `*_regenerated.csv` files, but K2 (post-C11
   2026-05-25) writes the regenerated EM splits as
   `<pair>_{train,val,test}_{corner_filled,baseline_pruned}.csv`. The
   glob never matches and the variant log emits the giveaway line
   `No regenerated EM split files in <work_dir>/input/entitymatching
   (skipping)`. Result: every variant directory under
   `usecases/<domain>-augmented/<level>/input/entitymatching/`
   contains only the *original* baseline EM gold files. The committee
   evaluator's `bundle.em_gold_regenerated` ends up empty across all
   levels, so all four R7b dual-test surfaces fall back to the
   baseline test gold via the existing fallback chain. Verified
   directly on products: the first row of every level's
   `products_1_2_products_2_test.csv` is identical
   (`products_1_35033699,products_2_14505369,false`).

2. **`DittoMatcher._cache_key` omits the source record values.**
   At [`ditto_matcher.py:215-243`](../usecases_synthetic/lib/ditto_matcher.py#L215-L243)
   the cache hash uses `(checkpoint_path + mtime, fields, max_len,
   max_field_len, sorted (id1, id2) pair set)` — but **not** the
   field values from `df1` / `df2`. When the same pair IDs reappear
   across variant levels under bug #1 (every level scoring against
   the baseline test gold pair set), the cache hits even though the
   underlying records have changed (K1 paraphrase, K5 unit swap, K6
   noise applied to the same source IDs at each level). Ditto
   returns the first level's cached predictions for every subsequent
   level. Other matchers don't share this bug — Magellan refits per
   call, comem / llm_matcher either don't cache or include the values
   in their cache key — which is why magellan / comem / llm_matcher
   move legitimately across levels (0.96 → 0.58 → 0.61 → 0.28 on
   products) while ditto_plm stays flat at exactly 0.9569 across
   baseline / easy / medium / hard.

**Why the bugs compound.** Bug #1 alone would still give ditto
different test pair sets per level (K2 corner_filled vs
baseline_pruned vs baseline gold), so the (id1, id2) cache key would
miss between levels and ditto would re-run inference — values would
move per level. Bug #2 alone, with bug #1 fixed, would still cache-hit
within a level (when the same pair is scored twice) but each level's
distinct pair set would force a fresh cache key. Together they
produce the observed flat ditto curve: every level's pair set is
identical (bug #1) and the cache key ignores the value drift the
other matchers do see (bug #2).

**Fix — Bug #1**: replace the single-pattern glob with a list:

```python
patterns = ("*_corner_filled.csv", "*_baseline_pruned.csv")
copied: list[Path] = []
for pattern in patterns:
    for path in sorted(src_dir.glob(pattern)):
        # existing per-file copy + provenance append
        copied.append(_copy_one(path, em_out))
```

Plus a regression test asserting that a generated variant's
`input/entitymatching/` directory contains both `_corner_filled.csv`
and `_baseline_pruned.csv` files for every authored source pair × the
three splits.

**Fix — Bug #2**: include a content hash of the relevant field values
in `_cache_key`:

```python
def _cache_key(self, candidates, df1, df2):
    h = hashlib.sha256()
    # ... existing checkpoint / fields / pairs hashing ...
    fields = self.fields
    by_id_1 = df1.set_index("id")[fields].astype(str)
    by_id_2 = df2.set_index("id")[fields].astype(str)
    for id1, id2 in pairs:
        for v in by_id_1.loc[id1]:
            h.update(v.encode()); h.update(b"|")
        h.update(b"\n")
        for v in by_id_2.loc[id2]:
            h.update(v.encode()); h.update(b"|")
        h.update(b"\n")
    return h.hexdigest()[:16]
```

This invalidates existing cache files (correct behavior — they were
keyed against stale source content). Cost: O(pairs × fields) string
concatenation per cache key — tens of ms for products' pair sets,
acceptable. Add a unit test asserting that `_cache_key` differs when
the same (id1, id2) candidate set is paired with different `df1` /
`df2` value tables.

**Expected effect.**

- Bug #1 fix alone makes the four dual-test surfaces actually
  distinct on variant levels (corner_filled vs baseline_pruned vs
  baseline test gold differ in pair-set composition + label
  distribution per C11). The R7b dual-model alias still holds
  (variant_model == baseline_model when there's no variant
  checkpoint), so the baseline-trained surfaces would still
  agree internally, but the *test*-side split would finally
  differentiate `_on_baseline_test` from `_on_regen_test`.
- Bug #2 fix isolates ditto. On products step-5 the immediate
  expectation is that ditto_plm shifts from a flat 0.9569 to a
  level-dependent curve roughly in line with the other matchers
  (clean baseline, monotone drop through hard). The exact value at
  each level depends on how much K1/K5/K6 actually move the model's
  embedding inputs — could easily land anywhere in the 0.5-0.85
  band at hard depending on the perturbation distribution.

**Estimated effort**: 30 min for bug #1 (glob + test); 1-2 hours for
bug #2 (cache-key refactor + test + invalidate existing cache
artifacts). Both ship together as R10-F so the regression test exits
on a real dual-test signal.

**Verification audit — every EM-gold reference must be re-checked
that it loads the *right* gold (added 2026-05-29)**. The two bugs
above were found one at a time during a statistics-file spot-check.
The blocker-recall conversation 2026-05-29 surfaced the broader
concern that we cannot be sure we are *actually* evaluating against
the corner_filled / baseline_pruned splits anywhere — we only know
the numbers fall back silently to the baseline gold today. Before
declaring R10-F "done", explicitly audit and assert correct
gold-loading at every code path that touches EM gold:

1. **Test-gold loaders**:
   - [variant_loader.py `_load_em_gold_regenerated`](../usecases_synthetic/lib/variant_loader.py)
     — confirm both `_corner_filled.csv` and `_baseline_pruned.csv`
     are loaded into `bundle.em_gold_regenerated[pair][split][version]`
     for each `version in {corner_filled, baseline_pruned}` and each
     `split in {train, val, test}`. The C11 schema is documented in
     the docstring; the test should assert the dict-of-dict-of-dict
     shape that `committee_em.EMMatchingCommitteeRunner.run` reads at
     line 1355.
   - [variant_loader.py `_load_em_gold_original`](../usecases_synthetic/lib/variant_loader.py)
     — confirm the baseline gold loader STILL fires at baseline level
     (no regen at baseline) but does NOT mask the regen loader at
     variant levels.

2. **Test-gold consumers**:
   - [committee_em.py `EMMatchingCommitteeRunner.run`](../usecases_synthetic/lib/committee_em.py)
     — trace lines 1305-1321 (the C11 4-cell evaluation pattern) and
     confirm `f1_baseline_test` / `f1_regen_test` are computed against
     the `regen_splits.test.baseline_pruned` / `corner_filled`
     DataFrames respectively, **not** the original gold.
   - [committee_em.py `EMBlockingCommitteeRunner.run`](../usecases_synthetic/lib/committee_em.py)
     — same audit for blocking. Specifically: assert that
     `pair_recall_*_on_baseline_test` uses baseline_pruned and
     `_on_regen_test` uses corner_filled, not baseline gold.
   - [`_resolve_variant_train_path`](../usecases_synthetic/lib/committee_em.py)
     (R7b helper) — confirm Magellan's per-pair train data points at
     `<pair>_train_corner_filled.csv` at variant levels.
   - [committee_norm.py `_build_entity_linkage`](../usecases_synthetic/lib/committee_norm.py)
     — the Normalization committee reuses EM gold positives to map
     `fusion_entity_id → {source_name: source_record_id}` so it can
     look up the raw value at the right (source, row) for every
     fusion-protected entity. Today the function reads
     `bundle.em_gold` (the original baseline EM gold copied into the
     variant dir). Audit:
     (i) confirm that's still the intended source post-R10-F — i.e.,
     entity *identity* doesn't change with K-knob perturbations, only
     values do, so the baseline positives are the correct linkage
     substrate;
     (ii) confirm the K2-dropped-entity skip path
     ([committee_norm.py `_score_attribute` lines 567-571](../usecases_synthetic/lib/committee_norm.py#L567-L571))
     gracefully exits when `row_idx is None`, dropping the cell from
     the denominator without recording a wrong/abstain — current
     behavior is correct, but the audit should assert it stays
     correct after the regen splits start landing in
     `bundle.em_gold_regenerated`;
     (iii) consider whether `_build_entity_linkage` should optionally
     prefer `bundle.em_gold_regenerated[pair][test].baseline_pruned`
     positives at variant levels (post-R10-F) — same entity identity,
     but pruned to surviving entities so the linkage dict is smaller
     and never includes K2-dropped references. Functionally
     equivalent under the current skip path; only matters if a future
     stage starts to *iterate* the linkage dict instead of looking up
     by ID.
   - [committee_norm.py `_score_attribute`](../usecases_synthetic/lib/committee_norm.py#L527)
     — confirm the gold values still come from
     `load_fusion_target_values(domain)` (fusion XML, frozen at
     baseline by design — fusion gold is byte-identically copied at
     variant package time). This is correct and load-bearing: Norm
     measures "can the normalizer recover the canonical
     baseline-fusion-XML value from the perturbed variant source
     value?" The R10-F audit does NOT need to migrate this to the
     regen test gold — the fusion XML IS the canonical reference and
     stays frozen until a future R8-style fusion-gold refresh.

3. **Fallback chains**:
   - Every `dict.get(new_key, dict.get(legacy_key, fallback))` chain
     in committee_em / committee_em_scoring should be audited. The
     pattern is OK for forward-compatible reads of legacy outputs,
     but it must NOT mask a missing regen file at variant levels.
   - Concretely: at variant levels where `bundle.em_gold_regenerated[pair][test]`
     is empty (because the files weren't copied), the fallback should
     LOG A WARNING and emit `None` for the regen-test metrics, not
     silently substitute baseline gold. Today the fallback is
     invisible.

4. **Metric emission**:
   - `aggregated` keys in metrics.json — confirm
     `macro_pair_recall_baseline_model_on_baseline_test`,
     `macro_pair_recall_baseline_model_on_regen_test`, etc., are
     each computed against the gold their name claims and not
     populated as `0.0` placeholders (today EM blocking emits 0.0
     for all four; that's a runner issue separate from this audit
     but worth flagging in the audit checklist).
   - `per_member` keys — same audit at the member level.
   - `per_pair.notes` (used by build_statistics) — same.

5. **Downstream consumers**:
   - [validate_variant.py per-pair CSV writer](../usecases_synthetic/scripts/validate_variant.py)
     — confirm the 4-cell matrix emits distinct values per surface
     when the regen files are present.
   - [build_statistics.py `_STAGE_AGG_KEY` reader](../usecases_synthetic/scripts/build_statistics.py)
     — already audited 2026-05-28; the em_blocking quirk is
     understood. Re-audit after R10-F to confirm the 4 dual-test
     surface sheets show distinguishable values when populated by
     the runner.
   - [analyze_monotonicity.py / monotonicity.py](../usecases_synthetic/lib/monotonicity.py)
     — the load-bearing key chain
     `*_variant_model_on_regen_test → legacy alias` must point at
     a *real* regen-test metric post-R10-F. Audit the
     `_BEST_MEMBER_METRIC` chain.

6. **Smoke regression test** (lands as part of R10-F implementation):
   construct a temporary variant directory containing the original
   baseline gold + corner_filled + baseline_pruned files for one
   source pair × one split. Run `validate_variant.py` against it.
   Assert that:
   - `metrics.json.per_stage.em_matching.aggregated.macro_f1_baseline_test`
     reflects the **baseline_pruned** gold (not the original baseline gold).
   - `..._regen_test` reflects the **corner_filled** gold.
   - The two values differ when the synthetic per-pair counts differ
     across baseline_pruned vs corner_filled (use deliberately
     distinct fixture data so the values can't accidentally agree).

This audit is the load-bearing part of R10-F — the glob fix is
mechanical, the cache-key fix is local, but the broader question of
"are we actually measuring what we think we are measuring" is what
the R10-F regression test must answer. Allocate ~half a day to walk
through the code paths above and write the smoke test; the actual
bug fixes are small relative to the audit.

**Step R10-F landed 2026-05-29:**

*Bug #1 — package_variant glob.*
[copy_regenerated_em](../usecases_synthetic/scripts/package_variant.py)
now globs `("*_baseline_pruned.csv", "*_corner_filled.csv")` (de-duped)
instead of the never-emitted `*_regenerated.csv`. Module + function
docstrings updated. New
[tests/test_package_variant.py](../usecases_synthetic/tests/test_package_variant.py)
(`TestCopyRegeneratedEm`, 4 tests): both versions × 3 splits copied,
legacy `_regenerated.csv` + original gold ignored, no double-copy,
missing-dir returns empty. The existing
[test_generate_variant.py](../usecases_synthetic/tests/test_generate_variant.py)
`test_package_variant_creates_full_directory` (which encoded the old
`_regenerated.csv` naming) was updated to the C11 naming + asserts the
legacy suffix is NOT copied.

*Bug #2 — DittoMatcher cache key.*
[ditto_matcher.py](../usecases_synthetic/lib/ditto_matcher.py):
`_cache_key` now takes `(candidates, df_left, df_right, id_column)` and
hashes the per-pair source field *values* (new `_value_lookup` helper:
`str(id) -> "\x1f".join(field values)`, first-wins on dup id, only
fields present in the frame). `_cache_path_for` + the `match` call site
thread the frames through. Same pairs scored against perturbed records
across variant levels now produce distinct keys, so ditto_plm stops
returning the first level's cached scores (the flat 0.9569 curve). 3
new `TestCacheKey` cases (value change invalidates, identical content
→ same key, only pair-referenced records matter); the 7 existing
cache-key/cache-path tests were migrated to the new signature.

*Verification audit (5 parallel auditors over every EM-gold path).*
Confirmed **OK**: (1) the loader
[variant_loader._load_em_gold_regenerated](../usecases_synthetic/lib/variant_loader.py)
builds the 3-level `{pair:{split:{version}}}` dict, both versions, no
masking by the original-gold loader; (2)
EMMatchingCommitteeRunner loads `corner_filled`→regen_test,
`baseline_pruned`→baseline_test and the 4-cell m_bb/m_br/m_vb/m_vr
matrix scores each surface against the right gold; (3)
EMBlockingCommitteeRunner does the same for pair_recall and emits real
`_macro_blocker` values (no 0.0 placeholders). **Rejected** as a false
positive: the committee_norm `row_idx is None: continue` skip — the
plan's own item 2.ii states this is correct (a K2-dropped entity leaves
both numerator and denominator, which is neutral, not F1-inflating).
**Fixed** (defensive / visibility):
- [committee_em.py:1195](../usecases_synthetic/lib/committee_em.py)
  (legacy `EMCommitteeRunner`) now uses `_resolve_variant_train_path`
  so that *if* it ran at a variant level it would train on
  `<pair>_train_corner_filled.csv`. Currently inert (the bundled runner
  only runs at baseline, where the resolver returns the baseline train
  unchanged; the active variant path `EMMatchingCommitteeRunner` already
  used the variant resolver) — landed to close the audit item and
  remove the latent footgun.
- Targeted gold-missing warnings added at both active runners'
  gold-load sites (EMBlocking ~L1888, EMMatching ~L2291): when a
  regenerated test split is absent at a *variant* level the runner now
  logs a WARNING naming the missing version, so a future glob/copy
  regression can no longer silently collapse the dual-test surfaces
  onto baseline gold. (Not added to the R7b model-alias fallback chain
  — that fallback fires by design until R10-G trains variant
  checkpoints, so warning there would be pure noise.) build_statistics'
  legacy-alias reader fallback left as-is (reader-side, low value).

*Smoke regression test (audit item 6).*
[test_committee_em.py](../usecases_synthetic/tests/test_committee_em.py)
`TestR10FDualTestGoldWiring` (2 tests) runs the real
`EMMatchingCommitteeRunner` on a `level="medium"` bundle whose
`baseline_pruned` test gold is all-positive (F1=1.0) and `corner_filled`
adds deceptive negatives (F1=0.75). Asserts the aggregated
`macro_f1_baseline_test` (1.0) and `macro_f1_regen_test` (0.75) are
distinct, correctly ordered, and that the per-pair
`f1_baseline_model_on_baseline_test` vs `..._on_regen_test` map to the
right gold — end-to-end proof the surfaces are distinguishable.

*Verification.* 81/81 across committee_em + ditto_matcher +
package_variant; 37/37 test_generate_variant; full synthetic suite
**1460 passed, 7 skipped, 2 failed** (the 2 are the pre-existing
`TestSupportedDomains` stale 3-domain assertions, unrelated). `black`
clean on all 7 changed files; no new `mypy` errors on the changed
files (committee_em.py / ditto_matcher.py carry only their pre-existing
synthetic-package mypy noise — usecases_synthetic is not mypy-strict).

**Cache deletion REQUIRED (old folder).** Bug #2 rotates every Ditto
inference cache key, so the existing
`usecases_synthetic/cache/ditto_inference/` entries (27 files / ~20 MB)
are now dead — keyed by the old value-blind hash, they will never be
hit again and only waste disk + confuse. Delete the directory before
the next EM run:
```
rm -rf usecases_synthetic/cache/ditto_inference
```
(The R10-A K1 paraphrase cache and the K1 LLM cache are untouched and
need no deletion.)

### R10-G — Variant-trained checkpoint retraining (R7c folded into R10, now mandatory)

**Status: phase 1 (code) landed 2026-05-29.** The three retrain scripts +
two smoke tests are in place; the baseline trainers gained two small
backward-compatible hooks so the variant path reuses them. **Phase 2
(per-domain training runs) is still pending** and executes inside each
domain's step-5 cascade. See "Step R10-G phase 1 landed" block below.

**Cause.** R7b shipped the dual-model dual-test infrastructure as code
+ runner support but left the variant-trained checkpoints unpopulated:
when no `<model>_checkpoints/<domain>/variant_<level>/best` exists,
the variant matcher aliases the baseline matcher and all four R7b
dual-test surfaces collapse to two distinct surfaces (test side only).
The paper headline metric — `variant_model_on_regen_test`, "even a
purpose-trained model degrades from easy → hard" — is then no
different from `baseline_model_on_regen_test` numerically, and the
intrinsic-difficulty claim cannot be made distinct from the
distribution-shift claim.

The original R7c framing left this as an optional per-domain
follow-up. User directive 2026-05-29: **make R7c mandatory and fold
it into R10**, so every domain's published step-5 numbers carry the
load-bearing variant-trained surface, not the alias.

**Scope — what R10-G covers**.

Two trainable EM committee members per the R7b "which members
retrain per variant" table:

| Member | Train data per (domain, level) | Output checkpoint |
|---|---|---|
| **ditto_plm** | pooled `<pair>_train_corner_filled.csv` across pairs → `json.gz` | `cache/ditto_checkpoints/<domain>/variant_<level>/best` |
| **sc_block** (EM blocking) | same pool → SupCon train set | `cache/sc_block_checkpoints/<domain>/variant_<level>/best` |

Magellan is technically trainable but re-fits per-pair at runtime in
the committee runner; R7b's `_resolve_variant_train_path` already
points it at `<pair>_train_corner_filled.csv` when present, so no new
checkpoint artifact needs writing for Magellan — it's covered by the
runtime fit.

Per (domain, level), R10-G needs:
- 1 pooled Ditto train file
- 1 SupCon train file
- 1 Ditto training run (~30-45 min MPS)
- 1 sc_block training run (~30-45 min MPS)
- 2 checkpoints saved under `variant_<level>/`

Three levels × four domains × two matchers = **24 training runs total**
across the cascade. With R10-G scripts written once and invoked per
(domain, level), each domain's R7c cost is **3-6 hours MPS**; across
all four domains the total is **12-24 hours MPS** (sequential).

**Phase 1 (code, lands with the rest of R10)** — author retrain
scripts so they exist before any domain hits the variant-regen step:

- [scripts/ditto/retrain_variant.py](../usecases_synthetic/scripts/ditto/) — takes
  `--domain` + `--level`, loads `<pair>_train_corner_filled.csv` for
  every source pair from
  `usecases/<domain>-augmented/<level>/input/entitymatching/`, pools +
  dedupes, emits `train.json.gz` to a per-(domain, level) work dir,
  invokes the existing Ditto trainer with R2-winner hyperparameters
  (model_id, batch_size, lr, max_len, max_field_len), writes the
  checkpoint to
  `cache/ditto_checkpoints/<domain>/variant_<level>/best`.
- [scripts/sc_block/retrain_variant.py](../usecases_synthetic/scripts/sc_block/) — symmetric for
  sc_block: load the same per-pair train CSVs, build the SupCon
  in-batch contrastive train set, invoke
  [lib/sc_block_train.py](../usecases_synthetic/lib/sc_block_train.py)
  with the existing per-domain hyperparameters, write the checkpoint
  to `cache/sc_block_checkpoints/<domain>/variant_<level>/best`.
- A thin convenience driver
  [scripts/retrain_variant_cascade.py](../usecases_synthetic/scripts/) — takes
  `--domain` (or repeated flag for batched cross-domain), loops
  `easy / medium / hard`, runs the two retrain scripts per level,
  emits a per-level training log under
  `usecases_synthetic/output/<domain>/<level>/r7c_retrain.log` for
  the cascade audit trail.

Phase-1 deliverables ship as code + smoke tests:
- `tests/test_retrain_variant_ditto.py` — a fast smoke test using a
  one-pair toy corner_filled CSV that asserts the script produces
  `train.json.gz` + invokes the trainer entry point + writes a
  checkpoint stub to the expected path.
- Same shape for `tests/test_retrain_variant_sc_block.py`.
- No actual training runs in CI — those gate on MPS and ship as the
  Phase-2 cascade.

**Step R10-G phase 1 landed 2026-05-29:**

Three scripts, both reusing the existing trainers (no duplicated
training/data-prep logic):
- [scripts/ditto/retrain_variant.py](../usecases_synthetic/scripts/ditto/retrain_variant.py)
  — `retrain_variant_ditto(domain, level)`: loads the variant bundle,
  **reverses K8 column renames** back to baseline names
  (`_reverse_k8_sources`) so the perturbed values project onto the
  canonical schema, builds pooled+deduped train/val WDC records via the
  existing `build_ditto_pair_records_from_gold(..., sources=<variant>)`
  + `write_json_gz`, then `_invoke_ditto_train` shells out to
  `ditto/train.py` (knob-02 `plm_*` hyperparameters + canonical-schema
  `--fields` + `default_train.yaml`) and `_place_checkpoint` symlinks
  `cache/ditto_checkpoints/<domain>/variant_<level>/best` → the produced
  run.
- [scripts/sc_block/retrain_variant.py](../usecases_synthetic/scripts/sc_block/retrain_variant.py)
  — `retrain_variant_sc_block(domain, level)`: `_build_variant_data`
  maps the variant sources to `text_cols` via the K8-resolved blocking
  column mapping (`bundle.resolve_column_mapping`) and collects the
  per-pair corner_filled train/val splits, then `_invoke_scblock_train`
  calls the existing `sc_block.train.train(..., output_dir=<variant>,
  data_override=…)` whose own `best` symlink lands at
  `cache/sc_block_checkpoints/<domain>/variant_<level>/best`.
- [scripts/retrain_variant_cascade.py](../usecases_synthetic/scripts/retrain_variant_cascade.py)
  — `--domain` (repeatable) `--levels`; loops easy/medium/hard, runs
  both retrains per level, writes
  `output/<domain>/<level>/r7c_retrain.log`.

Two backward-compatible hooks added to the baseline trainers (default
behaviour unchanged):
- [prepare_em_training_data.py](../usecases_synthetic/scripts/ditto/prepare_em_training_data.py)
  `build_ditto_pair_records_from_gold(..., sources=None)` — optional
  sources override (defaults to `load_domain_sources`).
- [sc_block/train.py](../usecases_synthetic/scripts/sc_block/train.py)
  `train(..., data_override=None)` — when set, skips `_load_domain_data`
  and uses the injected `(sources_mapped, em_train_by_pair,
  em_splits_by_pair)`.

Smoke tests (heavy trainer monkeypatched; no real training):
[tests/test_retrain_variant_ditto.py](../usecases_synthetic/tests/test_retrain_variant_ditto.py)
(5: orchestration + K8-reversal + dedup + baseline-rejected) and
[tests/test_retrain_variant_sc_block.py](../usecases_synthetic/tests/test_retrain_variant_sc_block.py)
(4: orchestration + `_build_variant_data` mapping/collection +
baseline-rejected). 9 new tests pass; existing `test_sc_block_train.py`
(22) + `test_prepare_em_training_data.py` (7) stay green; full synthetic
suite **1469 passed, 7 skipped, 2 failed** (the 2 pre-existing
`TestSupportedDomains` stale assertions). `black` clean on all 7 files.

**Phase 2 (per-domain execution, post-variant-regen)** — invoked as
part of each domain's step-5 cascade right after `generate_variant.py`
lands the variant and before the final `validate_variant.py` pass:

```
generate_variant.py --domain X (regen under R10 A/D/F/E settings)
  └→ retrain_variant_cascade.py --domain X     # R10-G phase 2 for X
     └→ produces variant_<level>/best per (model, level)
        └→ validate_variant.py --domain X      # dual-model signal live
```

Per-domain Phase 2 produces 6 checkpoints (2 matchers × 3 levels),
~3-6 hours MPS. Domains run independently; the cascade can parallel
or serialise based on hardware availability.

**Expected effect on the statistics file**. The four dual-test
surface sheets in [statistics/products.xlsx](../usecases_synthetic/statistics/products.xlsx)
(and equivalents for music / games / companies once they land) stop
showing identical values across surfaces for ditto_plm + sc_block.
The `train=Var` sheets diverge from the `train=BL` sheets by however
much the variant-trained model recovers from the perturbation
distribution it now sees in training. The `variant_model_on_regen_test`
row in committee_summary becomes the load-bearing headline; the other
three R7b surfaces remain in the dedicated sheets as references.

**Sequencing within R10**. R10-G splits across the phase boundary:

- **Phase 1 (code)** lands alongside the other R10 code items, after
  R10-F (which fixes the regen-data plumbing R10-G's training scripts
  read from). Without R10-F, the retrain scripts would find no
  `<pair>_train_corner_filled.csv` files in the variant directory and
  fail-fast.
- **Phase 2 (execution)** runs **per-domain after that domain's
  variant regen**, before the final `validate_variant` pass for that
  domain. It does not gate R10's pre-regen code work — it gates
  publication of that domain's step-5 numbers.

**Estimated effort**. Phase 1: ~1-1.5 days to author the three
scripts + the two smoke tests, given the existing trainer
infrastructure does the heavy lifting. Phase 2: 12-24 hours MPS total
across all four domains, parallelisable.

### R10-H — Baseline Ditto retrain for music / companies / products (R7-baseline-finish folded into R10)

**Cause.** The R7 trainer padding fix landed 2026-05-27 (see "R7 —
what landed"), but only the games-domain baseline Ditto checkpoint
was retrained against it at verification time. The baseline Ditto
checkpoints for music, companies, and products at
`cache/ditto_checkpoints/<domain>/best` still carry the
pre-padding-fix recipe. Measured baseline F1 distortion vs the
padding-fixed retrain (per R7 — what landed):

| Domain | BL Ditto F1 hit vs padding-fixed |
|---|---|
| companies | **−0.058** (load-bearing distortion) |
| music | −0.008 (within noise) |
| products | −0.012 (within noise) |
| games | 0 (already retrained at R7 verification) |

R10-G phase 2 retrains the *variant* checkpoints
(`variant_<level>/best`) but does not touch the baseline checkpoint.
Without R10-H, the dual-test `train=BL` surface sheets — both
`train=BL test=BL` and `train=BL test=Var` — inherit the padding-bug
distortion at the baseline level, especially on companies where it
dominates the variant-vs-baseline delta the paper claims rest on.

**Scope.** Retrain baseline Ditto for music + companies + products
using the R7-padded trainer with the same recipe games already uses:

- Input: pooled baseline EM training data
  (`<pair>_train.csv` across the domain's source pairs), pooled +
  deduped into `train.json.gz` via the existing
  [scripts/ditto/_prep_<domain>.py](../usecases_synthetic/scripts/ditto/)
  prep step.
- Trainer: the existing
  [scripts/ditto/train.py](../usecases_synthetic/scripts/ditto/train.py)
  with R2-winner hyperparameters
  (model_id, batch_size, lr, max_len, max_field_len) per domain.
  No new code — the padding fix is in the trainer already.
- Output: refreshed checkpoint at
  `usecases_synthetic/cache/ditto_checkpoints/<domain>/best` (the
  same path the baseline matcher reads in committee_em.py). The
  existing committee symlinks point at `best`, so no symlink
  repointing is needed once the directory is overwritten in place.

Per-domain: ~30-60 min MPS for prep + train. Three domains × ~45 min
average = **1-3 hours MPS total**. Games is intentionally skipped (its
baseline Ditto was retrained at R7 verification 2026-05-27).

**Baseline sc_block must also be retrained wide (added 2026-05-30).** R10-H
originally named only baseline Ditto, but the committee EM *blocking* member
`sc_block` reads its own baseline checkpoint at
`cache/sc_block_checkpoints/<domain>/best`, and the products one is still the
narrow 9-field R3 checkpoint (2026-05-22). Under R10-I's widened
`sc_block.text_cols`, `measure_baseline` would serialize the wide column set
at inference against a narrow-trained model — the same train/inference
mismatch the R10-I rework fixed for Ditto. So R10-H also runs
[scripts/sc_block/train.py](../usecases_synthetic/scripts/sc_block/train.py)
`--domain <domain>` (writes/symlinks `cache/sc_block_checkpoints/<domain>/best`
on the wide `DOMAIN_TEXT_COLS`). Run the two baseline trainers sequentially
(both MPS) before `measure_baseline`. Variant-side sc_block is separate —
R10-G phase 2's `retrain_variant_cascade.py` already trains variant Ditto +
variant sc_block per level.

**Why a separate R10 item rather than folding into R10-G.** R10-G
retrains the *variant* checkpoints from the K2-regenerated
corner_filled training data; R10-H retrains the *baseline* checkpoint
from the original baseline training data. Different inputs, different
output paths, different load-bearing semantics. Folding both into
the same script with a `--also-baseline` flag is feasible but
conflates two distinct training surfaces under one CLI, and one
domain's R10-H decision (skip / run) is independent of its R10-G
decision (always run under R10). Keeping them as separate R10 items
preserves that independence in the audit trail.

**Sequencing.** Pure-execution item (no Phase 1 code work — uses
existing scripts). Runs per-domain in the step-5 cascade. R10-H is
independent of variant generation, so it can fire as early as the
domain's R9 sweep finishes; it must finish before that domain's
`measure_baseline.py` so the refreshed baseline checkpoint is the one
loaded into the BL matcher instance.

Per-domain cascade with R10-H slotted in:

```
R9 deferred sweep for <domain>   # baseline-side HPO, no checkpoint dep
  └→ R10-H baseline Ditto retrain (30-60 min MPS; skip for games)
     └→ generate_variant.py (under R10 A/D/F/E settings)
        └→ R10-G phase 2 retrain_variant_cascade.py (3-6 hours MPS)
           └→ measure_baseline.py (loads R10-H refreshed BL checkpoint)
              └→ validate_variant.py (4-surface dual-test fully populated)
```

R10-H and R10-G phase 2 are independent training runs (different
inputs, different output paths), so they can run in parallel on
hardware that supports it. Sequentially the per-domain MPS budget is
~3.5-7 hours including R10-H.

**Expected effect.** The four EM matching surface sheets in the
statistics file show baseline F1 shifts:

- **companies**: BL surfaces shift up by ~0.058 F1, finally placing
  baseline-trained Magellan / Ditto on the same load-bearing
  footing as the variant-trained surface. The baseline-vs-variant
  delta on companies becomes interpretable as intrinsic difficulty
  + transfer gap rather than padding-bug + intrinsic + transfer.
- **music**: BL surfaces shift up by ~0.008 F1; within noise but
  consistent with the audit trail.
- **products**: BL surfaces shift up by ~0.012 F1; within noise.

**Estimated effort.** Code: none (R7 trainer landed; prep scripts
exist). Execution: 1-3 hours MPS across the three domains.

### R10-I — EM committee scope expansion to the full per-domain schema (R8 folded into R10)

**Status: config layer landed 2026-05-29; training-data-path rework
PENDING.** User chose **full widening** + **retrain reads the committee
ditto_plm.fields** (2026-05-29). The mapping showed R10-I is *not* "pure
config" — the "ditto fields = sc_block text_cols" invariant ties the
field list across 5 locations, and a deeper blocker surfaced (see "Step
R10-I — what landed / what's pending" below). What landed: all 8
committee YAMLs widened + DOMAIN_TEXT_COLS + SCBlockBlocker
missing-column tolerance + a consistency-guard test (1481 tests pass).
What's pending and REQUIRED before any wide-scope ditto training: the
Ditto training-data pipeline (variant R10-G retrain + baseline R10-H
prep) must build records from the committee-column-mapped sources, NOT
via knob_02 `attribute_mapping` (which covers only ~10 of the 19
products wide fields, so the current builder would train Ditto on
*empty* values for the new wide fields). The config layer is inert until
phase-2 retraining, so it is safe to have landed first.

**Cause.** Today both `ditto_plm` and `magellan` in
`em_matching_committee_<domain>.yaml` are scoped to a 9-attribute
subset of the full per-domain source schema (products example):

```
title, brand, description, product_type, model, model_number,
chipset_name, vram_gb, storage_gb
```

The 18 excluded columns
(`bus_type, color, form_factor, height_mm, interface_type, length_mm,
memory_type, read_speed_mb_s, storage_connection_type, weight_g,
width_mm, write_speed_mb_s, title_description, priceCurrency, url,
cluster_id, id, price`) are dropped via the Ditto `fields:` list and
the Magellan `attributes:` list. R8 (original framing 2026-05-28)
queued this as a *separate cascade* to be run after the 9-attr
step-5 numbers publish. User directive 2026-05-29: **fold R8 into
R10 / Phase 2** so the cross-domain rerun publishes the wide-scope
numbers directly and we don't pay for a second cascade.

**Why R8's original "needs its own cascade" framing was overscoped.**

R8 listed five items it claimed needed to happen distinctly:

1. Ditto re-prep + retrain — **already happening in Phase 2 under
   R10-G + R10-H**. The retrain just needs a wider `--fields` list
   to land per-domain; the trainer doesn't care how wide.
2. Magellan attribute expansion + re-sweep — config edit + folds into
   **R9-deferred** if it runs against the wide scope from the start.
3. EM blocking re-tune — optional; folds into R9-deferred's blocking
   sub-sweep.
4. Fusion + Norm — untouched by R8 (they already see the full
   canonical attribute set per `attribute_classes` in the domain
   YAML).
5. measure_baseline + validate rerun — **already Phase 2**.

The only piece R8 added that R10 / Phase 2 doesn't already cover is
the *config* expansion in the EM matching committee YAMLs. R10-I
captures that. Silver-standard rebuilds — flagged in R8 as a
prerequisite — are **not** required: the silver standard is built
from the source data, which already carries all per-domain
attributes regardless of which subset the matcher reads.

**Scope.**

1. Per-domain: rewrite the `fields:` list on `ditto_plm`,
   `llm_matcher`, `comem` in
   `em_matching_committee_<domain>.yaml` to enumerate every attribute
   present in the canonical source DataFrames (drop `id` /
   `cluster_id` / `url` which are not matchable, keep
   `title_description` which is derived and informative). Same set on
   `sc_block` `text_cols:` in `em_blocking_committee_<domain>.yaml`
   per the canonical "ditto fields = sc_block text_cols" invariant.
2. Per-domain: extend `attributes:` (and `numeric_attributes:` for
   numerics) on the Magellan member in the EM matching committee
   YAML to the same wide set.
3. Per-domain attribute counts after expansion (estimates;
   confirm at edit time):

   | Domain | Current scope | Wide scope |
   |---|---:|---:|
   | products | 9 | ~24 (27 cols - id/cluster_id/url) |
   | music | 9 | domain-specific full schema; check at edit time |
   | games | 10 | same |
   | companies | varies | same |

4. After R10-I lands, the **R9-deferred** sweeps for music / games /
   companies and (separately) a re-sweep for products run against
   the wide scope. Products' existing 9-attr R9 winners are
   discarded.

**Sequencing.** R10-I is pure config + a small follow-on: re-run
products R9 sweeps against the wide scope (the music / games /
companies R9-deferred sweeps were already scheduled and will pick up
the wider scope automatically once they fire). Lands as part of the
R10 code gate (no actual training; the wider Ditto / Magellan
training happens in Phase 2's R10-G + R10-H + per-domain sweeps).

Per-domain effect under Phase 2:

- Products R9 needs a re-sweep against the wide scope (the previous
  9-attr sweep winners no longer apply). ~25-50 min local CPU + the
  optional LLM judge sub-sweep.
- Music / games / companies R9-deferred sweeps fire against the wide
  scope from the start (no extra cost vs the original R9 plan).
- R10-G phase 2 trains the variant Ditto + sc_block per (domain,
  level) on the wide scope automatically because the retrain scripts
  read the Ditto `fields:` list from the YAML.
- R10-H retrains baseline Ditto on the wide scope automatically for
  the same reason.

**Expected effect on published numbers.**

The wider matcher input surface should be at-least competitive with
the 9-attribute baseline and may lift Ditto + Magellan F1 where the
excluded columns carry signal. Per-attribute breakdown in
`per_member` lets the paper report which columns the model actually
uses. Companies in particular likely benefits because the excluded
columns there include the long-tail descriptors the matcher had to
imply from the title.

**Estimated effort.** Config: ~30 min per domain × 4 domains = 2
hours total. R9 re-sweep on products: ~25-50 min local CPU. No new
code. All Phase-2 execution cost (R10-G + R10-H + measure_baseline +
validate) is already accounted for in those items.

**Step R10-I — what landed / what's pending (2026-05-29):**

Wide-scope canonical field list per domain (= K8 `sm_mapping` targets ∪
current matcher fields, minus `id`/`cluster_id`/`url`):
- **products** (19): title, brand, description, price, priceCurrency,
  title_description, product_type, model, model_number, chipset_name,
  vram_gb, storage_gb, bus_type, interface_type, memory_type,
  storage_connection_type, form_factor, read_speed_mb_s, write_speed_mb_s
- **music** (8): name, artist, release-date, release-country, duration,
  label, genre, tracks
- **games** (11): name, releaseYear, developer, platform, genres, series,
  criticScore, userScore, ESRB, publisher, globalSales
- **companies** (9): name, country, city, industry, sector, founded,
  keypeople, assets, revenue  (`sector` is an EM canonical attr absent
  from K8 sm_mapping, so kept via the union)

**Landed (config layer; inert until phase-2 retrain):**
- All 8 committee YAMLs widened to the per-domain wide list: `ditto_plm`,
  `llm_matcher`, `comem` `fields:` + Magellan `attributes:` (products;
  music/games/companies Magellan is auto-gen) in
  `em_matching_committee[_<d>].yaml`; `sc_block.text_cols` in
  `em_blocking_committee[_<d>].yaml`. Products Magellan
  `numeric_attributes` extended to `[price, vram_gb, storage_gb,
  read_speed_mb_s, write_speed_mb_s]` (music/games/companies already
  covered).
- [lib/sc_block_train.py](../usecases_synthetic/lib/sc_block_train.py)
  `DOMAIN_TEXT_COLS` widened to match (the now-outdated companies
  `[name, country]` intersection comment replaced).
- [lib/sc_block_blocker.py](../usecases_synthetic/lib/sc_block_blocker.py)
  `SCBlockBlocker.__init__` no longer raises when a `text_col` is absent
  from a heterogeneous source (e.g. companies forbes lacks city/founded);
  it fills the missing column with empty + warns, matching the trainer's
  NA-fill so train↔inference serialization stays identical. The two
  blocker tests that asserted the old ValueError were updated.
- New guard
  [tests/test_em_field_scope_consistency.py](../usecases_synthetic/tests/test_em_field_scope_consistency.py)
  asserts per domain `ditto_plm.fields == sc_block.text_cols ==
  DOMAIN_TEXT_COLS`, and that `llm_matcher`/`comem` widened to the same
  scope. Full synthetic suite **1481 passed, 7 skipped, 2 failed** (the 2
  pre-existing `TestSupportedDomains` stale assertions). `black` clean.

**Ditto training-data pipeline rework — LANDED 2026-05-29.** The
committee Ditto checkpoints (baseline `best` + variant `variant_<level>/best`)
now train on the **wide committee field scope** (`ditto_plm.fields` ==
`DOMAIN_TEXT_COLS`), column-mapped exactly the way the committee EM runner
maps sources before inference — so training and inference serialize the
same surface. What landed:

- New in
  [prepare_em_training_data.py](../usecases_synthetic/scripts/ditto/prepare_em_training_data.py):
  `committee_ditto_fields(domain)` (= `DOMAIN_TEXT_COLS` minus Ditto WDC
  reserved names), `committee_column_mapping(domain)` (reads the matching
  committee YAML; the runner enforces blocking==matching so this is the
  inference-authoritative mapping), `build_ditto_pair_records_committee_scope(...)`
  (column-maps sources → projects each gold pair onto the wide fields;
  missing/NaN → empty, dropped by Ditto's serializer), and
  `write_committee_fields_sidecar(...)`. The legacy narrow
  `build_ditto_pair_records_from_gold` is unchanged (still used by the
  pool-builder-only ADI path + the standalone CLI converter).
- [retrain_variant.py](../usecases_synthetic/scripts/ditto/retrain_variant.py)
  (R10-G variant path): now builds wide records via
  `build_ditto_pair_records_committee_scope` on `bundle.sources` with the
  committee column_mapping translated through K8 via
  `bundle.resolve_column_mapping(...)`; trains `--fields` = wide list.
  `_reverse_k8_sources` + the knob-02 `canonical_schema` projection are
  gone (knob-02 still supplies the PLM hyperparameters only).
- Baseline `_prep_{products,music,companies,games}.py`: the
  committee-correct (pydi / base) path builds wide records;
  companies/games keep their legacy `--train-source adi` branch on the
  narrow builder (pool-builder only, never wired to the committee —
  honours the directive that *only* the pool-builder Ditto trains on ADI;
  all committee dittos train on base/variant).
- Train-fields wiring:
  [train.py](../usecases_synthetic/scripts/ditto/train.py) gains `--domain`
  (+ `_resolve_field_scope`) — when set, the field scope is sourced from
  the canonical wide list, overriding the stale narrow `fields:` default in
  `default_train.yaml`; an explicit conflicting `--fields` is a hard error.
  The prep scripts print the matching `train.py --domain <d>` command and
  write a `fields.txt` sidecar, so an R10-H baseline retrain can't be run
  on a narrow field set.

**Two corrections surfaced + fixed during the rework:**
1. **Music `label` is a Ditto-reserved WDC key.** R10-I had widened
   `ditto_plm.fields` to include music's `label` (record-label attribute),
   but `label` collides with Ditto's pair-metadata key
   (`RESERVED_SERIALIZATION_FIELDS`) and `wdc_to_pair_examples` rejects it.
   Resolution (matches the YAML header's documented intent): Ditto's
   serialization scope = committee fields **minus reserved names**, applied
   at *both* ends — `committee_ditto_fields` drops them on the training
   side and `DittoMatcher` drops them on the inference side, so music's
   `label` is excluded consistently and the other 3 members keep it. YAMLs /
   `DOMAIN_TEXT_COLS` / the field-scope guard test are unchanged.
2. **NaN cells serialized inconsistently.** Loaded sources hold float
   `NaN` in many (now in-scope) sparse columns (price, vram_gb, durations,
   ...); `DittoMatcher` previously `str()`-ed them into literal
   `COL <field> VAL nan` tokens at inference while the training builder
   dropped them — a latent mismatch the wide scope would have made
   pervasive. `DittoMatcher._pair_text` now drops `None`/float-`NaN`,
   mirroring the builder, so both ends serialize identically.

Tests: new `committee_*` builder/helper tests + a train↔inference
byte-equivalence regression (companies heterogeneous mapping, music
reserved-`label`, products NaN/None) + `train._resolve_field_scope` tests +
extended field-scope guard; `test_retrain_variant_ditto.py` rewritten off
the wide builder (the `_reverse_k8_sources` class removed). Full synthetic
suite green except the 2 pre-existing unrelated `TestSupportedDomains`
fusion-silver assertions. **Phase-2 Ditto training (R10-G phase 2 + R10-H)
is now unblocked on the wide scope.**

**Adversarial verification (2026-05-29, 19-agent workflow):** 15 findings
examined; 13 refuted (field order, NaN/empty equivalence, `max_field_len`
parity, list-cell `str()` serialization, normalization-off for committee
baseline, identical column-mapping, ADI-path isolation, music `label`
filtering all confirmed sound). Net actionable findings on the rework
itself: **none**. One pre-existing robustness gap surfaced + **fixed**
(user-approved 2026-05-29): `committee_em._resolve_column_mapping` now
calls `_validate_no_id_rename`, which rejects any committee `column_mapping`
entry that renames the hardcoded `id` join key — either away
(`{id: other}`, loud-but-confusing downstream failure) or onto it
(`{other: id}`, which `apply_column_mapping` collision-handling would
turn into *silent* id corruption). Identity `{id: id}` is allowed. No
current YAML maps `id`; this is a guard against a future typo. 5 new tests
(`TestResolveColumnMappingIdGuard`).

### R10-J — Surface validation-split EM scores alongside the test-split scores

**Status: PENDING (note 2026-05-30).** During variant validation we currently
surface only the two **test-split** closed-set EM F1 numbers per pair —
`f1_baseline_test` (on `em_test_baseline_pruned.csv`, Set 1) and
`f1_regen_test` (on `em_test_corner_filled.csv`, Set 2) per C10. The
validation-split equivalent (`regen_val`) is computed but kept *internal*
(logged only on val/test divergence, not surfaced in the per-member metrics).

**Decision (user, 2026-05-30): also collect + surface the validation-split
scores that correspond to the test splits used in validation.** For every
test-split metric, emit the parallel val-split metric: `f1_baseline_val`
(closed-set on the regenerated `*_val_baseline_pruned.csv`) and `f1_regen_val`
(closed-set on `*_val_corner_filled.csv`). C11 already emits both val versions
per pair (it regenerates train / val / test × baseline_pruned / corner_filled),
so **no regen-side change is needed** — only the committee scorer + the stats
surfacing.

**Scope:**
- [committee_em.py](../usecases_synthetic/lib/committee_em.py): promote
  `regen_val` from internal-only to a surfaced per-member metric `f1_regen_val`,
  and add the baseline-val counterpart `f1_baseline_val` (closed-set on the
  val splits, mirroring the test-split scorers). Aggregate to
  `macro_f1_baseline_val` + `macro_f1_regen_val`.
- [build_statistics.py](../usecases_synthetic/scripts/build_statistics.py):
  add the val columns to the EM stats sheet alongside the existing test
  columns so every surface shows a val/test pair.
- **Monotonicity verdict is unchanged** — it stays on `f1_regen_test`
  (C10 / C2). The val scores are collected for val/test agreement + tuning
  visibility, not as the slope driver. The internal divergence log from C10
  is subsumed by the now-surfaced pair.

**Sequencing:** validate-side only; independent of the R10 code gate. Land
before the per-domain validate runs publish step-5/6 numbers so the val/test
pairs appear from the first publication.

### R10-K — K1 `llm_paraphrase` operator never executed (wiring fix) — landed 2026-05-30

**Status: landed 2026-05-30.** Surfaced by R10-D's new K1 intensity audit on
the first post-R10 products regen: `knob_01_realised_intensity_monotonicity`
FAILed because hard was *shallower* than medium (edit-dist 0.173 → 0.128).
Root cause: K1's `llm_paraphrase` operator never ran at **any** level —
the R10-D v2 paraphrase prompt was never called. Two coupled defects:

1. **Client hardcoded to None.** `apply_values_joint` passed
   `llm_client=None` to K1 — the C1 "call the live LLM on cache miss" fix was
   applied to K2 but never K1. With no client, misses raised `LLMCacheMiss` →
   `strict_cache_miss` identity fallback (2341 cells at products hard).
2. **Cache built only at hard.** `generate_variant` materialised
   `llm_cache_k1` only when `level_k1 == "hard"`, so at medium every
   `llm_paraphrase` draw short-circuited to `llm_cache_missing`.

Net: K1's difficulty signal came entirely from deterministic operators, whose
hard mix is abbreviation-heavy and shallower than medium → the FAIL.

**Fix (Fix A + Fix B):**
- New `build_openai_paraphrase_client` in
  [surface_operators.py](../usecases_synthetic/lib/surface_operators.py)
  (mirrors the K2 interpolation / non-corner builders;
  `(prompt_template, value) -> paraphrase`, `{value}` substitution,
  `<UNCHANGED>` passthrough, quote-stripping, gpt-5.4-mini).
- [generate_variant.py](../usecases_synthetic/scripts/generate_variant.py)
  builds the client when `not strict_cache_k1 and OPENAI_API_KEY` and threads
  it via `_run_joint` → `apply_values_joint` → `apply_knob_01` (new
  `api_client_k1` param, replacing the hardcoded `None` at
  [apply_values_joint.py](../usecases_synthetic/scripts/apply_values_joint.py)).
  Cache build switched to config-driven `_k1_uses_llm_paraphrase(domain,
  level)` (operator_mix `llm_paraphrase` weight > 0) so medium + hard get a
  cache and easy stays cache-free. The `apply_values_joint` CLI `main()`
  mirrors both.
- +13 tests (client builder, config-driven helper, orchestrator wiring at
  easy/medium/hard, no-key fallback, `apply_values_joint` forwarding). Full
  synthetic suite 1523 passed (the 2 pre-existing `TestSupportedDomains`
  fusion-silver assertions still fail, unrelated).

**Verified on the products regen (2026-05-30):**
`knob_01_realised_intensity_monotonicity` → **PASS** (0.0 / 0.216 / 0.262).
K1 hard: committed 854 → **1546**, mean_edit_distance 0.128 → **0.262**,
token_jaccard_drop 0.085 → **0.348**, `strict_cache_miss` **2341 → 0**. The
fix is **cross-domain** — music / games / companies K1 inherit it on their
regens (their pre-fix K1 caches were also empty). Remaining 4 monotonicity
FAILs are all known/accepted: K2 `realised_vs_configured` (easy R10-B
protection floor; medium/hard on target), K5 `format_prov_rows` (raw-count
proxy; load-bearing `distinct_format_families` PASSES), K10 rate + count
(rate is the load-bearing K10 metric, marginally non-monotone; count is the
demoted secondary).

### R10-L — Monotonicity-gate hardening + known-weak allowlist — landed 2026-06-01

**Status: landed 2026-06-01.** The music post-K2-fix regen exit-1'd on 4
audit FAILs. An investigation workflow (3 readers + adversarial critic)
found two of them were **audit-design artifacts**, not data regressions,
and surfaced two **genuine** inversions the original triage had glossed.
Three fixes to
[generate_variant.py](../usecases_synthetic/scripts/generate_variant.py)
(+ tests, 111 passing):

1. **K5 `_k5_distinct_format_families` bug.** Read `payload["target_fmt"]`
   — a key **no K5 operator emits**. Date rows write `to_format`, unit /
   currency rows `to_unit`, number rows `to_locale`. Every row collapsed
   to `(fn, "")` → flat **music 2/2/2** (passed only on the
   `non_decreasing`-allows-equality technicality; zero escalation
   evidence). Fixed with a priority-key lookup
   `(to_format, to_unit, to_locale, target_fmt)`. Realised, verified on
   real provenance: **music 3/4/8**, **products 3/10/16** — both monotone,
   genuine escalation. `format_prov_rows` demoted to advisory.
2. **K1 intensity sample-gate.** The intensity mean is noise when too few
   cells fire (music easy = **2 cells** → 0.405 overshoots well-sampled
   medium/hard ≈ 0.346). Now compares only levels with
   `paraphrase_committed >= K1_INTENSITY_MIN_COMMITTED (30)` — a proxy for
   "config-active" (easy `paraphrase_rate=0.0` is inactive by design).
   **<2 qualifying levels → FAIL** (not SKIP): the gate keys `exit()` on
   `FAIL` only, and the rate check is non-decreasing (PASSes on flat
   committed) so it is **not** a dormancy backstop — FAIL-on-<2 is.
3. **Two-tier WARN downgrade** (`_apply_status_downgrades`):
   - `ADVISORY_CHECKS` (all domains): structurally K3-shrink-confounded
     raw-count proxies — `knob_05_format_prov_rows`,
     `knob_10_realised_monotonicity` (count). Load-bearing companion is
     the families / rate check.
   - `KNOWN_WEAK_EXCEPTIONS` (per `(domain, check)`): genuine load-bearing
     inversions accepted as documented dial limits. Any **new/unlisted**
     load-bearing non-monotonicity still FAILs.

Result: **music GREEN** (0 FAIL / 3 WARN), **products GREEN** (0 FAIL /
4 WARN).

#### ⚠️ DEFERRED — revisit in a future variant iteration

The three `KNOWN_WEAK_EXCEPTIONS` are **allowlisted, not fixed** — real
dial weaknesses parked to keep the gate green. **Come back to these when
the knobs/metrics are next reworked** (per user, 2026-06-01):

- **`(music, knob_02_realised_monotonicity)`** — K2 is intrinsically
  low-range for music (~0.33 ceiling at `max_interp_fraction=0.60`);
  realised medium 0.258 > hard 0.248 (+0.01 capped-sample dilution).
  *Fix:* widen the K2 reachable range for music or accept it as a
  small-range knob and re-target.
- **`(products, knob_10_realised_rate_monotonicity)`** — `swap_rate`
  denominator `reshufflable_count` shrinks 691/613/496, inverting the
  rate (0.593/0.602/0.587). *Fix:* redesign K10's rate over a
  **stable base** instead of the K3-shrinking reshufflable pool.
- **`(products, knob_02_realised_vs_configured)`** — K2 easy target 0.20
  is below the achievable floor for products: `drop_corner_touching`
  cannot pull the baseline ratio (~0.48) down to 0.20 (realised 0.477).
  *Fix:* raise products K2 easy target or strengthen the drop operator.

Also deferred: the K5 token-key mapping was validated on **music +
products only** — re-validate on companies / games provenance when those
regenerate (their operator vocab may differ).

### R10-M — Music R9 cascade + fusion/norm tooling fixes + products staleness audit — 2026-06-01

Work done driving the **music** R9 → measure/validate chain (continues
the per-domain cascade after R10-G music variant retrain completed: all 3
levels, 6 checkpoints, val-F1 0.970/0.947/0.926).

**R9 tooling fixes (landed 2026-06-01):**
- **Fusion sweep harness** ([committee_fusion_c12.py](../usecases_synthetic/lib/committee_fusion_c12.py)
  + [_tune_fusion_committee.py](../usecases_synthetic/scripts/_tune_fusion_committee.py)),
  4 fixes — the committee RUNTIME was never broken (only the tuning
  sweep): (1) `reselect: bool=False` on `C12FusionCommitteeRunner.run`
  bypasses the persisted `(domain,member,attr)` selection cache per sweep
  cell (the cache froze candidate-param sweeps like `trim` to no-ops);
  default False = byte-identical normal runs. (2) `_score_run` captures
  `per_member` so single-member sub-sweeps are picked by the swept
  member's `macro_accuracy`, not the committee-wide `overall_accuracy`
  (= best-member macro, invariant to the swept member). (3) `_sub_llm_judge`
  skips the enabled cell when `OPENAI_API_KEY` is unset (else `llm_only`
  scores 0.0 and corrupts the comparison). (4) docstring refresh. 223
  fusion tests pass; 2 pre-existing `TestSupportedDomains` failures unrelated.
- **Norm tuner** ([_tune_norm_committee.py](../usecases_synthetic/scripts/_tune_norm_committee.py)):
  the `llm_canonicalize` SPEC swept `max_tokens=[64]` (reasoning-model
  floor is 1024 → every cell errored → 0.0) and `prompt_version=[v1]`
  (committee uses v2). Fixed to `[2048]` / `[v2]` + docstring note. The
  tuner was **NOT** C12-incompatible (an earlier unverified caution): its
  SPECS members (text_clean/date_iso/…) ARE the `rule_normalizers`
  candidates that `rule_per_attribute_optimal` selects among.
- **Products fusion gold XML staleness** ([_author_products_fusion_xml.py:61](../usecases_synthetic/scripts/_author_products_fusion_xml.py#L61)):
  hardcoded 5-attr `CANONICAL_ATTRS` → emitted a 6-tag fusion XML despite
  the upstream CSV carrying all 19. Now derives the scope from
  `load_domain_config('products').attribute_classes`; regenerated both
  XMLs to 19 attrs. Products fusion YAML widened to 19 (eval_functions +
  attribute_types) with numeric `tolerance=0.15` hard-set from the human
  baseline (`products_workflow_v2.ipynb`); `price` left 0.1.
  (numeric_tolerance_match is an ABSOLUTE band, not %.)

**Music R9 winners applied (2026-06-01):**
- EM matching (magellan): winner = current sklearn default (n_estimators=100,
  max_depth=None) → no change.
- EM blocking: **sc_block top_k 50→100, threshold 0.3→0.0** (R9 recall
  0.9993). embedding kept bge-base (bge-small led by +0.0001 = noise);
  standard/sn = current.
- Fusion: re-sweep (trust/trim/truthfinder/accusim/casefusion/fusionquery/ltm)
  **confirms the R5 params within n=161 val noise** (~1 correspondence) →
  no change. Tolerance hard-set `duration=10` (from `music_workflow.ipynb`,
  absolute ±10s). tolerance + list_threshold (eval params) + llm_judge
  (LLM cost) excluded from the sweep.
- Norm rules (n=2288): **text_clean lowercase false→true** (+0.019);
  **country_iso name→official_name** (+0.27). The latter overturns a
  deliberate `name` comment — investigation confirmed the fusion reference
  is dominated by official long-forms ('United Kingdom of Great Britain
  and Northern Ireland' 123 >> 'United Kingdom' 13; 'United States of
  America' 24 >> 'United States' 2), so official_name matches ~150/200 vs
  name ~34/200.
- **llm_canonicalize**: sweep RUNNING (apply best `num_examples` when done).
- **Remaining music steps**: apply llm winner → `measure_baseline --domain
  music --with-llm` → `validate_variant --domain music --level {easy,
  medium,hard} --with-llm` (no `--level all`) → `analyze_monotonicity` +
  `build_statistics`. Headline = build_statistics committee_summary
  em_matching row `macro_f1_variant_model_on_regen_test`.

**Music norm scope note:** music kind_map covers 5 of 7 canonical attrs
(genre/label are discogs-only, by design). Music fusion set is full
(unlike products) — no staleness.

#### Products staleness audit (2026-06-01) + ⚠️ DEFERRED products-phase plan

A 3-agent audit confirmed the wide-schema (19-attr) propagation MOSTLY
worked: SM gold (80 rows, R10-C), both EM committees (R10-I wide scope),
Ditto + sc_block checkpoints (wide 19-field, run_20260530), and knob
configs K3/K6/K8/K10 are all at 19; K1/K4/K5 are narrow **by design**
(step4h review). The fusion XML + norm kind_map were the exceptions.

**New issues found (beyond fusion XML + kind_map):**
1. **K2 generation gap** — [knob_02_niche/products.yaml](../usecases_synthetic/config/knob_02_niche/products.yaml):
   `canonical_schema` widened to 12 but `attribute_mapping` left at 10 →
   **priceCurrency + title_description absent**; K2-synthesized niche
   entities in the products variants silently drop those 2 attrs
   (`apply_knob_02_niche.py` reverse-lookup returns empty). Config fix is
   trivial; **decision pending**: does it warrant a products variant regen
   or is it acceptable on existing variants (affects only K2 niche entities)?
2. **fusion_silver_standard.csv/.json built from wrong schema** — contains
   4 phantom attrs (width/height/length_mm, weight_g — explicitly excluded
   from products.yaml as too-sparse) and misses 5 real ones (title,
   description, price, title_description, model). Regenerate from the
   19-attr gold.
3. **Products fusion R9 winners stale** — `trust_scores` + TD params in
   [fusion_committee_products.yaml](../usecases_synthetic/config/committees/fusion_committee_products.yaml)
   were tuned on the retired 5-attr gold (`sweep_products_full.json`, May
   28); no 19-attr products fusion sweep exists.

**Downstream-stale (all regenerated by the products re-measure):**
`baselines/products/baseline_metrics.json` + report (fusion/norm 5-attr,
written 2026-05-30, pre Jun-1 XML refresh), `fusion_committee_selection.json`
(10/19 attrs), `norm_committee_selection.json` (5/19),
`validation/products/{easy,medium,hard}/metrics.json` (fusion/norm 5-attr),
the monotonicity/cross_level CSVs, and `final_report.md` (2026-05-16, pre-R1).

**Cosmetic (comment-only):** stale fusion-YAML NOTE block (lines 14-21,
contradicts the accurate R10-L block at 51-58), ditto_plm "must be
retrained" TODO (done 2026-05-30), K10 "9 attributes" / S11/S13 comments.

**Products phase — GATED (user 2026-06-01): do NOT auto-start. Stop after
the music chain (validate ×3 → analyze_monotonicity + build_statistics)
completes and wait for the user's next task.** Ordered plan when resumed:

> **STATUS (2026-06-02) — SUPERSEDED by R10-O.** Step 1 (Config) LANDED at
> the FULL 24-attr scope under the new `schema_constraints` methodology
> (not the 19-attr / xml_targets framing below); steps 2-3 DEFERRED per
> user. The 19-attr text below is kept for audit trail. See R10-O.

1. Config: K2 `attribute_mapping` (+priceCurrency,+title_description);
   norm widen — kind_map (protection.py: continuous vram_gb/storage_gb/
   read_speed/write_speed; nominal product_type/bus_type/interface_type/
   memory_type/storage_connection_type/form_factor; long_string model/
   model_number/chipset_name; free_text title_description) +
   normalization_committee_products.yaml rule_normalizers (number_locale
   numerics, text_clean strings/cats, add taxonomy_lookup) + copy the 3
   K6 taxonomies (Product_Type / Storage_Interface / GPU_Memory) into the
   synthetic data_root + confirm Storage_Interface→interface_type vs
   storage_connection_type; cosmetic comment cleanups.
2. Re-sweep/regen: products fusion sweep on 19-attr gold + apply (with the
   fixed harness); products norm sweep (text_clean/number_locale/taxonomy/
   llm_canonicalize) + apply; regenerate fusion_silver_standard.
3. Re-measure: baseline (fusion+norm) + validate ×3 + monotonicity +
   final_report on the 19-attr scope.

**Held uncommitted (per user, 2026-06-01):** the R10-L gate-hardening, the
R10-M fusion/norm tooling fixes + music R9 YAML edits, and (when done) the
products-phase changes. usecases_synthetic was folded into the PyDI repo
2026-06-01 (no longer a nested git repo).

### R10-N — Companies-FULL readiness audit + music-parity cross-checks — 2026-06-02

**Order (user 2026-06-02): companies runs RIGHT AFTER music**, before the
products phase (R10-M). Supersedes the 2026-05-29 R0 "Companies-conventions
audit" (plan ~§R0) — that checklist predates R5/R6/R7/R9 + all R10 items;
this is the current-state re-audit (3-agent workflow) + the rigorous
cross-checks vs the finished music run that the user requested. Companies
is **very out of date — a near-full redo**; most artifacts must be
(re)done or wired new.

#### Current-state audit (2026-06-02)

**BLOCKERS:**
- **`fusion_files` not wired** — `config/domains/companies.yaml` has NO
  `fusion_files:` block, so `domain_config.py:108-113` defaults to
  `validation_set.xml` (25) / `test_set.xml` (18) — the **stale small gold**
  the whole R0 "load-bearing zone" rationale exists to fix. The expanded
  100/100 `*_set_final.xml` (user-authored 2026-05-29) is **never loaded**.
  FIX: add `fusion_files: {validation: validation_set_final.xml, test:
  test_set_final.xml}` (mirror music.yaml:34-36). *Highest-impact item.*
- **`usecases/companies-augmented/` missing entirely** — `generate_variant
  --domain companies` has never run. The variant pipeline must run from scratch.

**HIGH — scope inconsistency + data/checkpoints:**
- **Scope is a mess across artifacts** — `sector` (K2/committees) vs
  `industry` (SM/kind_map/K8/norm-taxonomy) vs `keypeople` (fusion gold) vs
  `founders` (canonical target_schema) vs `website` (canonical only). K2
  `knob_02_niche/companies.yaml` canonical_schema = 6 attrs (drops
  assets/revenue/keypeople, adds phantom `sector`); `domains/companies.yaml`
  attribute_classes = 7 (omits `industry`); fusion gold = 7; committees = 9;
  target_schema.json = 9 data attrs (name,website,founded,country,city,
  industry,assets,revenue,founders). **RESOLUTION (user 2026-06-02,
  CORRECTED): use the FULL target_schema, NOT the fusion-test subset — for
  ALL domains.** So the companies fusion gold must be WIDENED from 7 → the
  full target-schema scope (add industry + website; reconcile keypeople↔
  founders), and K2 canonical_schema/attribute_mapping + attribute_classes +
  silver harmonize to that full set. (My earlier "keep 7-attr" resolution was
  wrong.) Cross-domain coverage vs target_schema: **music 8/8 already full
  (the finished music run is correct — no redo)**; companies 7/9 (missing
  industry, website); games 9/11 (missing globalSales, series); products
  19/24 (missing the 5 sparse dims color/width/length/height_mm/weight_g).
  Games + products widenings happen in their phases.
  **RESOLVED (user 2026-06-02): "if source and fusion gold agree, fix the
  names in the target schema."** Source (`keypeople_name`) + gold
  (`keypeople`, multi-truth nested `<name>` list, 91/100 populated) agree on
  **keypeople**, so `target_schema.json` `founders` was RENAMED →
  `keypeople` (title/desc broadened from founders-only to key-people; DONE
  2026-06-02). The two missing attrs are NOT naming conflicts — they're
  absent from the gold and get ADDED by widening: `industry` (derive per
  cluster from the trusted source — forbes `business_segment` / dbpedia
  `sector`) + `website` (forbes `url`), with provenance, consistent with how
  the gold's existing values are source-derived. **Companies full scope = 9:
  name, website, founded, country, city, industry, assets, revenue,
  keypeople.** K2 canonical_schema/attribute_mapping + attribute_classes +
  silver harmonize to those 9. **General principle (all domains):** where the
  source and fusion gold agree on a name, fix the target_schema to match
  (rather than the gold) — apply when auditing products/games target schemas.
  Tracked in repo-root `companies_run_progress.md`.
- **EM gold ID convention** — `usecases/companies/input/entitymatching/*.csv`
  (dated 2026-05-04) carry raw native IDs (forbes URLs, dbpedia URIs,
  fullcontact_N), not the source-prefixed `companies_<n>_<id>`. Verify the
  current loader's expectation; rewire if it needs the prefix.
- **sc_block checkpoint NARROW** — `text_cols=[name,country]` only (May 15,
  pre-R9/R10-I); must retrain on the wide 9-attr scope like the other domains.
- **R10-H baseline Ditto retrain mandatory** — the −0.058 F1 hit on companies
  was the strongest argument for R10-H. (The `usecases_synthetic`-side Ditto
  best IS valid/post-R7 — May 27 R6-3 PyDI, val_f1 0.93 — but R10-H still
  mandates the fresh baseline retrain. NOTE: `hard_negative_gate.plm_checkpoint`
  + some runners point at the **top-level** `cache/ditto_checkpoints/companies/best`
  = May 4 **ADI-trained** stale checkpoint — repoint to the post-R7 one.)
- **pool STALE** (ADI-era, 2026-05-05, references the May-4 ADI checkpoint) →
  rebuild after the Ditto retrain.
- **fusion_silver_standard STALE** (May 23, pre-final-gold + pre-R10) →
  rebuild under the harmonized canonical scope.
- **Published baseline + validation STALE/pre-C12** — `baselines/companies/
  baseline_metrics.json` (May 12, pre-C12 per-(attr,strategy) names, small
  gold) + `validation/companies/{easy,medium,hard}/` (Apr/May leftovers,
  mixed committee hashes) → delete/regenerate. Old plain `test_set.xml`/
  `validation_set.xml` (Apr 2) → delete.
- **norm sweep NEVER run** — `baselines/companies/norm_committee_selection.json`
  missing (unapplied-sweep-winner class); run the companies norm sweep + apply.

**VERIFY-ONLY / already OK** (no products-style gap): committees are
C12-shaped (companies is the **canonical unsuffixed** domain —
`em_blocking_committee.yaml` / `em_matching_committee.yaml` /
`fusion_committee.yaml` ARE the companies files; only norm is
`_companies`-suffixed; `committee_paths.py`); committee attr scope is the
full R10-I wide 9-attr set; K1/K2/K4 already `gpt-5.4-mini` + prompt v2 (no
claude-opus leftover); norm **kind_map is near-full (8 attrs)** — NOT the
products narrow-kind_map gap; SM gold full scope; K8 full coverage; K10
scope OK; fusion `*_final.xml` fresh + load-bearing-sized (100/100, 7 attrs).

#### Companies cascade (refreshed; runs right after music)

0. **Config-fix pre-step**: wire `fusion_files` → `*_final.xml` (blocker);
   harmonize scope (K2 canonical_schema/attribute_mapping + attribute_classes
   + silver to the agreed authoritative set — DECISION); rewire EM gold IDs
   if needed; repoint `hard_negative_gate` checkpoint to the post-R7 Ditto;
   delete stale published + old plain XML.
1. **R10-H baseline Ditto retrain** (wide scope) + **sc_block retrain** (wide
   9-attr, currently [name,country] only).
2. **Pool rebuild** (post-retrain Ditto) → **silver rebuild** (harmonized scope).
3. **R9-deferred sweep** under R10-I wide scope — EM blocking (incl sc_block,
   with the R10-M `reselect` fix), EM matching (magellan), fusion (R10-M
   harness), **norm (never run)** — apply per-domain winners (NOT music's).
4. `generate_variant --domain companies` (never run) under R10 A/D/E/F.
5. R10-G `retrain_variant_cascade --domain companies`.
6. `measure_baseline --domain companies --with-llm`.
7. `validate_variant --domain companies --level {easy,medium,hard} --with-llm`.
8. `analyze_monotonicity` + `build_statistics --domain companies`.

#### Rigorous music-parity cross-checks

*must-pass-BEFORE-run:* fusion_files → `*_final` (100/100); K1/K2/K4 =
gpt-5.4-mini + v2; committee files C12-shape (companies unsuffixed); EM
matching roster parity (4 members enabled, threshold 0.5, gpt-5.4-mini/2048);
sc_block reflects a **companies** R9 re-sweep (not the frozen R5 sign-off,
and NOT music's 100/0.0 — per-domain); norm llm_canonicalize at
max_tokens=2048/prompt v2; fusion numeric tolerance hard-set from the
**companies** human-baseline notebook (assets/revenue rel-0.1, founded
year_only — NOT music's abs-10); fusion_silver + baseline rebuilt under
current canonical+C12.

*must-pass-AFTER-run:* R10-L monotonicity gate **GREEN (0 FAIL)** with the
shared ADVISORY_CHECKS, + any companies-specific genuine inversion added to
`KNOWN_WEAK_EXCEPTIONS` with a dated justification (companies has none yet);
K1 sample-gate (floor 30) applied + K1 rate check passes; **K5
`distinct_format_families` re-validated on companies provenance** (the
R10-L token-key fix — must be monotone non-decreasing, NOT pinned flat);
R10-F dual-test 4 surfaces emitted + `build_statistics` headline =
`em_matching macro_f1_variant_model_on_regen_test`; **SM committee reports
best_member** (this session's fix); cross-level committee macro_f1 monotone
easy≥medium≥hard (C2 contract).

*sanity-compare:* companies baseline committee F1s land in a plausible band
vs music — **band-check, never equality** (a companies value identical to
music's would itself flag an accidental config copy).

**Companies LEGITIMATELY differs from music — do NOT "correct" to music's
values:** (1) DBpedia is the noisiest source → trust leans Forbes/FullContact
(except GICS industry/sector where DBpedia can win — `feedback_dbpedia_noise_profile`);
(2) GICS industry taxonomy (vs music genre); (3) 3 sources forbes/fullcontact/
dbpedia, forbes-hub pairing, **non-empty column_mapping** (music's is `{}`);
(4) norm choices are *opposite* by design — text_clean `lowercase:false`
(music true), country_iso `output_format:name` (music official_name) — both
domain-justified, must NOT be flipped; (5) fusion tolerance from companies
notebook; (6) magellan `class_weight:balanced/max_depth:20` (companies R5
winner) vs music's null/null; (7) lower absolute F1 band expected (noisier
sources, ~1543 EM pairs). The one real norm gap: llm_canonicalize
`num_examples` is UN-swept (=5) → measure the companies optimum, do NOT copy
music's 0.

**GATE (updated user 2026-06-02):** do NOT stop after music — **auto-continue
directly into the companies cascade** (config-fixes → retrains → pool/silver
→ R9 sweep → generate_variant → R10-G → measure/validate/analyze), then
**stop and wait after companies completes**. Run companies autonomously
(handle the config fixes + scope harmonization per the resolution below);
surface results + any genuine anomalies when stopping after companies.

### R10-O — `schema_constraints` norm surface + products config-prep landing — 2026-06-02

> **FOLLOW-ON (2026-06-02): products SOURCE-NATIVE schema adopted — see
> repo-root `products_native_schema_plan.md` (steps 1-5 EXECUTED) +
> `products_run_progress.md`.** The new upstream gives each of the 4
> products sources its OWN native column vocabulary (same values/IDs;
> SM is now a real task, mirroring companies). Executed: source
> re-materialized native + id-prefixed; `column_mapping` populated in the
> 3 committees (native→canonical); `sm_mapping_gold` ← upstream per-source
> map; all knob per-source `attribute_mapping`/`attribute_classes`/etc.
> + K8 `sm_mapping`/`rename_table` re-keyed native; norm-eligible verified
> 24/24. **No ditto/sc_block retrain** (matchers serialize canonical
> post-`column_mapping`; values+canonical-names unchanged). The
> canonical-keyed config below (kind_map, scorer fix, fusion eval=19 given
> gold, norm committee, domain attribute_classes, EM fields) STANDS; the
> per-source `attribute_mapping`/`rename_table` parts were re-expressed
> native. Compute (sweeps/measure/validate + silver fix) still DEFERRED
> behind the companies run.

Supersedes R10-M's products-phase step 1 (which assumed the legacy
`xml_targets` norm methodology). Two things converged: a new norm-scoring
surface discovered in the working tree, and the products scope decision.

**`schema_constraints` norm-scoring surface (new; discovered + verified
from source 2026-06-02, not previously in this plan).** A working-tree-only
mechanism that scores the Normalization committee against the per-domain
canonical `target_schema.json` constraints instead of the legacy fusion
val/test XML targets:
- New module [schema_constraint_scorer.py](../pipelines/lib/schema_constraint_scorer.py)
  (`parse_target_schema` + `AttributeConstraints` + `SchemaConstraintScores`)
  parses JSON-Schema + `x-pydi-consistency` into per-attribute constraints.
- Uncommitted edits to [committee_norm.py](../usecases_synthetic/lib/committee_norm.py)
  + [committee_norm_c12.py](../usecases_synthetic/lib/committee_norm_c12.py)
  add a `scoring_surface` param; `C12NormCommitteeRunner` gains
  `_score_member_against_schema` (no per-entity gold — the constraint set
  IS the gold) and computes eligible attrs as `sm_resolved ∩ {schema attrs
  with has_any_constraint} ∩ kind_map`.
- [measure_baseline.py](../usecases_synthetic/scripts/measure_baseline.py)
  adds `--norm-scoring-surface` **defaulting to `schema_constraints`** and
  records it in baseline meta; [validate_variant.py](../usecases_synthetic/scripts/validate_variant.py)
  reads it back so every variant scores against the same surface.
- **Cross-domain impact:** every domain measured/validated after this lands
  uses `schema_constraints` by default — including the in-flight companies
  run (the first consumer). Reads `usecases/<domain>/input/schemamatching/
  <domain>_target_schema.json` (preferred) or `target_schema.json`.
- **Scorer bug fixed (2026-06-02):** `AttributeConstraints.has_any_constraint`
  used `v not in (None, False, (), [])`, which via `0.0 == False` silently
  dropped any attribute whose only constraint is a zero numeric bound —
  e.g. products `price` (`minimum: 0`) — from norm scoring. Fixed at BOTH
  compute sites (the dataclass `__post_init__` + the `parse_target_schema`
  recompute) to identity checks. Verified: products `price` now eligible;
  companies(7)/music(7)/games(9) constrained-attr sets UNCHANGED (strict
  no-op for the running companies run); 11 scorer tests pass.

**Products scope decision (user 2026-06-02): 24 (full target_schema), NOT
19 — applied PER-SURFACE.** The 5 sparse dims previously excluded as
too-sparse (R10-M) are brought into the **norm (schema_constraints) +
K-knob + SM** scope to match the authoritative `target_schema.json`
(user-edited 2026-06-02). Source coverage: color 8-15%, width/length/
height_mm 4-6%, weight_g 3% (vs 50-78% for the 19-set). 24 = target_schema
minus id + url. **Fusion is NOT widened: it scores only the GIVEN fusion
val/test gold (validation_set.xml / test_set.xml, 100 records, fixed at 19
attrs) — the gold is not regenerated and attributes absent from it are not
fusion-scored (same principle as companies' industry). EM = 19 fields.**
So per-surface: SM/norm/K-knobs = 24; fusion = given 19-attr gold; EM = 19.

**EM match fields stay at 19 (user 2026-06-02).** The 5 sparse dims are
perturbed (K-knobs) + scored (norm/fusion) but are NOT EM match features
(magellan documents excluding sparse columns; 3-15% coverage won't aid
matching). The 19-field ditto + sc_block checkpoints (run_20260530) stay
valid → no products EM retrain.

**Config-prep LANDED 2026-06-02 (products-only + 2 signed-off shared
touches; companies/music/games files untouched; all 11 configs parse;
24-attr consistency verified). Live tracker: repo-root
`products_run_progress.md`.**
- K2 `attribute_mapping` +priceCurrency,+title_description (R10-M #1, the
  confirmed silent-drop bug).
- [protection.py](../usecases_synthetic/lib/protection.py) products kind_map
  5→24 (shared file, products-key only; companies/music maps untouched).
- [domains/products.yaml](../usecases_synthetic/config/domains/products.yaml)
  attribute_classes 19→24.
- [fusion_committee_products.yaml](../usecases_synthetic/config/committees/fusion_committee_products.yaml):
  stale NOTE-block cleanup only — committee scope stays at the GIVEN
  19-attr gold (an initial +5 widening was REVERTED 2026-06-02 after
  confirming the gold carries exactly 19; sparse dims not fusion-scored).
- [normalization_committee_products.yaml](../usecases_synthetic/config/committees/normalization_committee_products.yaml):
  text_clean→15, number_locale→9, new taxonomy_lookup (product_type→
  Product_Type; bus_type+interface_type→Storage_Interface; memory_type→
  GPU_Memory). **No taxonomy copy needed** — TaxonomyLookupNormalizer
  resolves via `USECASES_DIR`=`usecases/`, where the 3 taxonomies already
  ship (corrects R10-M's "copy into the synthetic data_root" step).
- K3/K6/K8/K10 products configs +5 dims each (K3 tight 0.02 drop caps; K6
  color=string + 4 numerics; K8 rename ladder Attribute_21-25; K10 targets).
  EM ditto_plm/sc_block stale "must be retrained" comments corrected.

**K5 (format) EXCLUDES the sparse dims (user-confirmed 2026-06-02 "leave as
is").** K5's engine hardcodes money/number/file_size/rate families with
`unit_factors.yaml` unit tables; length/weight units would need a SHARED
`apply_knob_05_format.py` change (companies risk), the `number` family is
unused/ill-suited (currency/magnitude path), and at 3-6% coverage the
format dial is marginal. Sparse dims remain in 6 of 7 surfaces.

**DEFERRED (user 2026-06-02 "wait with the compute-heavy stuff") — run only
AFTER the companies cascade completes** (shared scripts/files in active use
+ MPS/CPU contention):
NO fusion gold regen — the val/test gold is the fixed GIVEN 19-attr set.
1. fusion_silver_standard.py `_PRODUCTS_STACK` fix (R10-M #2) — align the
   silver to the GIVEN 19-attr gold: drop the 4 phantom dims (width/length/
   height_mm, weight_g — absent from the gold) + add the missing real gold
   attrs (title/description/price/priceCurrency/title_description/model).
   SHARED file companies' silver build uses right now.
2. Products fusion sweep on the GIVEN 19-attr gold (R10-M #3: trust/TD
   params stale from the retired 5-attr gold) + apply → rebuild
   fusion_silver_standard.
3. Products norm sweep (text_clean/number_locale/taxonomy/llm_canonicalize)
   under schema_constraints (24) + apply.
4. measure_baseline (`--with-llm`) → validate ×3 → analyze_monotonicity +
   build_statistics; likely a variant regen so the K-knobs exercise the
   now-in-scope dims + the K2 fix.

### R10 sequence + gate

Land code in dependency order:
**R10-D → R10-A → R10-F → R10-G (phase 1) → R10-E → R10-C → R10-I → R10-B**.

R10-H has no code phase — pure per-domain execution against the
existing trainer.

1. R10-D (K1 v2 prompt) first because A re-keys the LLM cache by cell
   only, and pre-baking the v2 cache under the old cache key would
   waste LLM spend.
2. R10-A (nested perturbations) second — once it lands, all subsequent
   regens guarantee the cumulative-cell invariant, including the
   v2-cache pre-bake.
3. R10-F (EM dual-test plumbing) third — bugs surface in EM
   evaluation, so the fix needs to be in before the regen cascade
   exercises the dual-test infrastructure for real. Lands after A so
   the regen test can verify both nested-cell invariants AND the
   corner_filled / baseline_pruned files landing in the variant
   directory. Independent of D.
4. R10-G phase 1 (retrain scripts) fourth — the scripts read the
   `<pair>_train_corner_filled.csv` files that R10-F's plumbing fix
   puts in the variant dir, so the scripts depend on F. Lands as
   code-only (no training runs); the training executes per-domain in
   phase 2 during each domain's step-5 cascade.
5. R10-E (K4 source rebalancing) fifth — runner-only change, no
   measurement-side impact; affects every future variant regen and
   should land before the cross-domain cascade so all 4 domains
   inherit the rebalanced K4. Independent of A/D/F/G phase 1 but
   ordered after them so the regen test runs exercise the combined
   post-R10 pipeline.
6. R10-C (scope equivalence) sixth — touches only baseline gold
   files; independent of variant generation but must precede the
   R9-deferred sweeps (see "Order vs R9").
7. R10-I (EM committee scope expansion) seventh — config edit on the
   per-domain EM matching + blocking YAMLs; lands after C so the
   wide-scope sweeps audit against the refreshed gold but before the
   per-domain cascade fires. Must precede R9-deferred + products R9
   re-sweep so all sweeps run against the wide scope from the start.
8. R10-B (documentation only) anytime.

After R10-A / R10-C / R10-D / R10-E / R10-F / R10-G-phase-1 / R10-I
land, kick off the cross-domain variant regen cascade. Each domain's
cascade embeds R10-H (where applicable) before regen and R10-G phase 2
between regen and final validate, all running under R10-I wide scope:

```
R10-H baseline Ditto retrain --domain X      # R10-H (~30-60 min MPS;
                                             #        wide scope, skip for games)
  └→ generate_variant.py --domain X          # under R10 A/D/F/E
     └→ retrain_variant_cascade.py --domain X # R10-G phase 2, wide scope
        │                                    # (~3-6h MPS)
        └→ measure_baseline.py --domain X    # uses R10-H wide-scope BL ckpt
           └→ validate_variant.py --domain X # wide-scope dual-test surface
```

If the EM medium > easy anomaly persists on any domain post-R10-A,
that's the signal to dig deeper (e.g. into K6's character-noise
distribution); the R10-E rebalance should already have taken K4 out
of the suspect list by shrinking the hard products_1 ablation to a
normal-sized perturbation, the R10-F fixes should have made the
dual-test surfaces finally distinguishable, and R10-G phase 2 should
have replaced the variant=BL alias with a real variant-trained signal
for ditto + sc_block.

**Order vs R9 — hard gate, R10 must precede R9-deferred**.

R9 is split in two:
- **R9 products** (done 2026-05-28): products sweep cascade + winner
  application + measure_baseline + validate. Already landed under the
  *current* methodology (6-attr SM gold, non-nested K1/K5/K6 cells, v1
  K1 prompt). Products step-5 numbers will be re-emitted on top of
  R10 once it lands.
- **R9 deferred** (music / games / companies sweep cascades): NOT
  STARTED, and **gated behind R10**. R10-C refreshes the baseline SM
  (and Norm) gold mapping for every domain to cover the full R1
  attribute set; running R9-deferred sweeps before R10-C lands would
  evaluate sweep cells against the stale 6-attr SM gold, requiring a
  redo after R10-C anyway. Even though the individual R9 sub-sweeps
  (Magellan, EM blocking, fusion) don't directly read `sm_mapping_gold.csv`,
  re-running them post-R10-C lets us audit the music / games /
  companies SM gold under the same scope every other measurement
  uses, and avoids a half-step methodology mismatch in the published
  step-5 numbers.

Concrete sequencing for the remaining work:

1. **R10-D → R10-A → R10-F → R10-G phase 1 → R10-E → R10-C → R10-I →
   R10-B** (this gate; see "R10 sequence + gate" above). All
   code-only.
2. **R9 sweeps under R10-I wide scope** —
   - **R9 products re-sweep** against the wide scope (existing 9-attr
     winners discarded). ~25-50 min local CPU + optional LLM judge.
   - **R9 music / games / companies** under the post-R10 + R10-I
     wide scope from the start (no extra cost vs the original R9
     plan).
3. **Per-domain step-5 cascade** for all 4 domains — products first
   as the regression-target lead, then companies / music / games
   (run sequentially in that order, or parallelise non-products in
   parallel if hardware allows). Per user directive 2026-05-29:
   products goes first because the R10 changes were debugged against
   products data and products numbers re-publish first as the
   canonical regression target before the other three domains'
   first-time publications. All four cascades run under R10-I wide
   scope:
   - **R10-H**: baseline Ditto retrain for music / companies /
     products (30-60 min MPS; skip for games). Refreshes
     `cache/ditto_checkpoints/<domain>/best` against the R7-padded
     trainer using the R10-I-expanded `fields:` list.
   - `generate_variant.py` under R10-A nested cells + R10-D v2 prompt +
     R10-F regen-test plumbing + R10-E rebalanced K4. The variant
     data already covers the full source schema; no regen-side
     change from R10-I.
   - **R10-G phase 2**: `retrain_variant_cascade.py --domain X`
     (3-6 hours MPS) — variant Ditto + sc_block trained on the
     wide-scope `fields:` list, producing checkpoints per (level)
     that unlock the load-bearing `variant_model_on_regen_test`
     surface.
   - `measure_baseline.py` under the R10-C-refreshed measurement
     scope. Loads the R10-H refreshed baseline checkpoint and the
     wide-scope Magellan attribute set.
   - `validate_variant.py` per level. The four dual-test surface sheets
     in the statistics file now show distinguishable values (R10-F
     test-side + R10-G train-side + R10-H clean BL anchor) under
     wide-scope EM (R10-I), and the `variant_model_on_regen_test`
     row in committee_summary becomes the load-bearing headline.
4. **Publish step-5 numbers** per domain. Products numbers re-publish
   under the post-R10 methodology + wide scope; music / games /
   companies numbers publish for the first time, all with R7c-trained
   variant matchers
   and R7-padded baseline matchers so every domain's BL + Var dual-test
   surfaces carry the same load-bearing semantics.

**Order vs R8**: R8 is superseded by **R10-I** (2026-05-29). The
wide-scope EM committees land as part of the R10 code gate and the
wide-scope step-5 numbers publish directly from Phase 2 — no separate
post-step-5 cascade.

---
