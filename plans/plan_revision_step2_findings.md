# plan_revision.md R-1 Step 2 — Diagnose (findings)

Step 2 of [plan_revision.md R-1](plan_revision.md): re-run the
monotonicity audit on the existing music + games variants with the
new instrumentation that landed in step 1 (2026-05-19). No regen of
the variant data itself; the new audits read from the existing
provenance + baselines.

Artifacts produced this step:

- `usecases_synthetic/output/{music,games}/monotonicity_report.csv`
  (variant-level audit, rerun via `check_monotonicity`)
- `usecases_synthetic/validation/{music,games}/monotonicity_report.{csv,md}`
  (cross-level analyzer rerun with the new `ceiling_responsiveness`
  column landed)
- `usecases_synthetic/validation/{music,games}/monotonicity_best_member.csv`
  (P8 best-member ceiling, already on disk from R7.2; reproduced here
  for narrative completeness)

---

## Finding 1 — C3 K5 and K8 intensity audits replace the FAIL'd raw proxies

### Music

| Check | easy | medium | hard | Status |
|---|---:|---:|---:|---|
| `knob_05_format_prov_rows` (old raw count) | 34760 | 43624 | 36102 | **FAIL** |
| `knob_05_distinct_format_families` (new) | 2 | 2 | 2 | **PASS** |
| `knob_08_naming_edit_distance` (old) | 0 | 83 | 163 | PASS |
| `knob_08_naming_intensity` (new, rung-weighted) | 0 | 24 | 48 | PASS |

The new K5 intensity check shows **K5 touches a constant 2 distinct
format families across all 3 levels** on music. So the row-count
non-monotonicity (G3) was *purely* a per-source-pool-size draw
artifact, not a knob configuration issue. The intensity reading
matches the K5 card's intent: the dial is "how many format families
to mix" and music's config sits at 2 at every level.

### Games

| Check | easy | medium | hard | Status |
|---|---:|---:|---:|---|
| `knob_05_format_prov_rows` (old) | 53048 | 53048 | 57472 | PASS |
| `knob_05_distinct_format_families` (new) | 1 | 2 | 3 | **PASS** |
| `knob_08_naming_edit_distance` (old) | 192 | 83 | 137 | **FAIL** |
| `knob_08_naming_intensity` (new, rung-weighted) | 0 | 3 | 43 | **PASS** |

Games' K5 *does* increase family count across levels (1 → 2 → 3),
which is what the card promised. The K8 raw edit-distance FAIL (G5)
is fully resolved by the rung-weighted intensity check: 0 / 3 / 43
makes the qualitative jump from "descriptive" → "cryptic / anonymized"
visible. The old edit-distance proxy penalised `abbreviated` (many
small edits) more than `cryptic` (few but conceptually larger), which
inverted the ordering.

**Implication for step 3 / R-1 §C3:** the new audit rows are
load-bearing. The legacy `knob_05_format_prov_rows` and
`knob_08_naming_edit_distance` rows should be demoted to "informational
/ legacy", with the intensity rows taking the audit verdict. No
config change needed for K5 / K8 on the basis of these audits — the
dials are doing what the cards say.

---

## Finding 2 — K10 rate audit needs a regen pass

`knob_10_realised.csv` (the new artifact added under C3 K10) is
absent from the existing music + games variant dirs because the
variants pre-date step 1. The standalone `check_monotonicity` replay
fell back to legacy count-based K10 checks; the rate-based row
(`knob_10_realised_rate_monotonicity`) was skipped.

Counts on the existing variants (legacy view):

| Domain | easy | medium | hard | Legacy verdict |
|---|---:|---:|---:|---|
| music | 100 | 151 | 98 | FAIL (G4) |
| games | 2 | 14 | 10 | FAIL (G4) |

The rate-based audit is the load-bearing K10 verdict under C3 / G4.
To get it on existing variants we need to either (a) re-run
`generate_variant.py --domain {music,games} --level all` (cache-fed,
no LLM calls expected), which refreshes `knob_10_realised.csv`
alongside the new K2 `rejected_*` columns and lets the rate check
emit, or (b) write a smaller "audit-only K10 rerun" that builds
swap_rate from the existing provenance + post-K3 reshufflable count.

Recommendation: **(a)**, as part of the step 3 calibration cycle.
Doing it standalone here would yield the rate but not exercise the
K10 ↔ K3 ordering on the actual variant sources used downstream by
EM / fusion.

---

## Finding 3 — K2 dial dormancy reconfirmed, guardrail breakdown pending

The existing `knob_02_realised.csv` on each variant does not yet
carry the new `rejected_*` columns (those are written by the K2
dispatcher under step 1 / C1). I'm running K2 standalone in
strict-cache mode against music hard + games hard to surface the
breakdown without LLM calls; the result will be appended to this
file under "Finding 3b" once both jobs finish.

What we can already confirm from the on-disk realised CSVs:

| Domain | level | baseline_ratio | target_ratio | final_ratio | operator | interpolated |
|---|---|---:|---:|---:|---|---:|
| music | easy | 0.257 | 0.20 | 0.257 | noop_baseline_above_target | 0 |
| music | medium | 0.266 | 0.50 | 0.266 | interpolate_paired_drop | 0 |
| music | hard | 0.260 | 0.80 | 0.260 | interpolate_paired_drop | 0 |
| games | easy | 0.670 | 0.20 | 0.670 | noop_baseline_above_target | 0 |
| games | medium | 0.660 | 0.50 | 0.660 | noop_baseline_above_target | 0 |
| games | hard | 0.671 | 0.80 | 0.671 | interpolate_paired_drop | 0 |

Three things this already tells us:

1. **Music hard**: the dispatcher chose `interpolate_paired_drop` at
   both medium and hard but produced 0 entities. Step 1's
   `rejected_*` instrumentation will show which guardrail consumed
   each cache hit.
2. **Games hard**: same — `interpolate_paired_drop` chosen, 0 produced.
   Games' baseline corner ratio of 0.67 means easy and medium both
   hit `noop_baseline_above_target` (F7 path) — by design.
3. **Easy levels on both domains noop**: aligns with C2's diagnosis
   that easy difficulty rests on K1 / K3 / K4 / K5 / K6 / K8 / K10
   only. Step 3's C2 decision (K1 floor / pool redesign / drop easy)
   matters here.

### Finding 3b — per-guardrail rejection counts (music landed; games pending)

**Music hard, strict-cache replay** (`/tmp/k2_diag/music/hard/`):

```
level,baseline_ratio,target_ratio,final_ratio,operator,removed,interpolated,interp_attempted,rejected_strict_cache_miss
hard,0.26,0.8,0.26,interpolate_paired_drop,0,0,1080,1080
```

**This contradicts the plan's "12 LLM calls" claim.** The real
diagnosis is:

- K2 hard for music selects **1080 parent pairs** (`interp_count_chosen=1080`,
  with `max=18849` as the per-domain cap).
- All 1080 hit `strict_cache_miss` — none of the 1080 pair-hashes
  are in the existing K2 cache.
- Cause: `generate_variant.py` defaults `strict_cache_k2=True` at
  hard level for non-aliased domains
  ([generate_variant.py:792-795](../usecases_synthetic/scripts/generate_variant.py#L792-L795)).
  The standalone `apply_knob_02_niche.py` CLI applies the same
  `(level == "hard")` → `strict_cache=True` rule. So the original
  full-music K2 hard run **never called the LLM**; the 1080
  cache-miss outcome maps directly to `interpolated=0`.

The 1597 existing entries in `cache/knob_02_interpolations/music/`
are presumably from music-small runs (which use a non-strict cache
path during their first run); they were keyed by the music-small
parent-pair selection, not the music FULL selection.

**Plan claim revision:** the "12 LLM calls, guardrail-dropped" framing
in plan_revision.md §G1 should be replaced with "0 LLM calls because
strict_cache=True at hard suppresses them; cache empty for full-music
hard pair selection". No guardrail rejection happened because no
response reached the guardrails.

**Practical implication for C1 (step 3):** the K2 dial-dormancy fix
isn't "raise `interpolation_count`" or "relax `contamination_check`".
It's one of:

| Option | What | Trade-off |
|---|---|---|
| **(α)** Pre-populate the K2 cache | Run K2 hard once with `strict_cache=False` + live LLM key. Cache the 1080 responses. Subsequent strict runs hit cache. | One-time LLM cost (1080 calls × interpolation_count attempts each per pair — could be 12k–100k calls depending on the per-pair attempt cap). Quality depends on the `contamination_check` rejection rate, which we still can't measure until calls actually land. |
| **(β)** Disable strict_cache at hard | Flip the `(level == "hard")` default to `False` (or add a config flag). K2 falls through to the deterministic blender on cache miss (`default_api_client_from_attributes`). | Deterministic, no LLM cost. Quality is lower — the blender alternates parent tokens / averages numerics. May produce contamination_check failures (collision with real entities) at unknown rates. |
| **(γ)** Lower the K2 budget | Reduce `interp_count_chosen` so K2 picks fewer pairs (or only pairs already in cache). | The dial moves toward target only if some pairs are cached. If 0 cached, this is equivalent to (β) without the blender — i.e. K2 stays dormant. Likely a noop given the current 0 cache hit rate. |

**Recommendation for C1 (step 3):** (α) pre-populate the cache as
the surgical fix. Couple with C1's existing C1 sub-decision —
contamination_check thresholds + interpolation_count per pair —
because once cache populates, guardrail rejections will become the
new bottleneck (or not — we won't know until 1080 responses land).

The same diagnosis very likely applies to **games hard** (same
strict_cache default, same dispatcher path). Awaiting the games
K2 standalone to confirm; this section will be updated when it
lands.

### Finding 3b addendum — games

games K2 standalone was terminated after ~38 min without producing
`knob_02_realised.csv` (the K2 niche scoring on games' ~70k entities
across 4 sources is roughly 2× music's runtime, and the diagnostic
value beyond music's finding is low). The same code path applies:
`apply_knob_02_niche.py` forces `strict_cache=True` at hard, and
`generate_variant.py` does the same via `strict_cache_k2 = (level
== "hard") and not is_aliased`. Games hard is therefore presumed to
exhibit the same `strict_cache_miss=N` pattern, where N is whatever
budget the K2 dispatcher picks for games (likely smaller than
music's 1080 given games's higher natural corner ratio of 0.67 — the
operator may even noop at hard since baseline is closer to target).

The on-disk realised CSV (pre-step-1, no `rejected_*` columns) already
shows:

```
hard,0.6705,0.8,0.6705,interpolate_paired_drop,0,0
```

Operator `interpolate_paired_drop` produced 0 entities — consistent
with the music outcome (cache miss every attempt). To get the explicit
N for games and verify, a fresh standalone run can be retried on a
calmer time-slice, or the diagnostic can be deferred to the step-3
calibration regen which will rerun K2 in non-strict mode under option
(α) and produce the breakdown organically.

---

## Finding 4 — Ceiling responsiveness (C6) lands but is sparse

The new `ceiling_responsiveness` column is populated in both domains'
`monotonicity_report.csv`. With only 4 data points (baseline → easy →
medium → hard) Pearson r is noisy, and many signals end up NaN
because the metric isn't level-keyed at every level (per-member,
spread, and pool metrics).

| Domain | non-NaN responsiveness rows | NaN rows |
|---|---:|---:|
| music | 16 / 32 | 16 |
| games | 11 / 32 | 21 |

Among the non-NaN rows the values are mostly **high** (≥ 0.8) — no
clear ceiling-immune knobs surface from this column alone in either
domain.

**The more readable artifact is `monotonicity_best_member.csv`** (P8),
which already exists and surfaces ceiling-immunity by direction. From
that file:

- **music norm**: ceiling 0.768 → 0.774 → 0.758 → 0.776 (winner
  `text_clean` at every level). **Not non-increasing.** Difficulty
  dial invisible to the user-selected matcher → ceiling-immune knob
  for the norm stage on music.
- **games SM**: 1.0 / 1.0 / 1.0 / 1.0 (winner `llm_openai` at every
  level). **Fully flat.** SM stage ceiling-immune on games.
- **games norm**: 0.975 → 0.951 → 0.938 → 0.943. **Not non-increasing**
  at hard (small rise).
- **music EM-block**: 0.999 / 0.999 / 0.999 / **0.590** (G6 confirmed,
  cliff at hard).
- **games EM-block**: 1.0 / 1.0 / 1.0 / **0.560** (G6 confirmed).
- **games EM-match**: 0.771 / **0.865** / 0.808 / 0.647 (easy beats
  baseline by +0.094 — G2 reconfirmed; ceiling moves *up* easy →
  baseline).

**Implication for step 3:**

- Treat `monotonicity_best_member.csv` as the user-facing
  ceiling-responsiveness artifact; promote it in R7.3 final reports.
- `ceiling_responsiveness` column is still useful for spot-checking
  specific (knob, signal) pairs but should not drive automatic
  PASS / FAIL verdicts at this sample size.

---

## Finding 5 — Easy-easier-than-baseline anti-pattern (G2) reconfirmed

From the per-stage best-member ceilings:

| Domain | Stage | baseline | easy | Easy easier? |
|---|---|---:|---:|---|
| music | sm | 1.0 | 1.0 | flat (winner switch label_jw → llm_openai at easy) |
| music | em_blocking | 0.999 | 0.999 | flat |
| music | em_matching | 0.976 | 0.976 | flat |
| games | em_matching | 0.771 | 0.865 | **easy +0.094** |

Also from the **committee-mean** (the R7.3 monotonicity_report.md
tables that originally surfaced G2):

| Domain | Stage | Δ easy vs baseline |
|---|---|---:|
| music | sm | +0.016 |
| music | em_block | +0.019 |
| music | em_match | +0.148 |
| games | sm | +0.134 |
| games | em_match | +0.053 |

C2 options (a) re-design easy pool, (b) K1 floor, (c) drop easy
level remain on the table. Step 1 instrumentation does not redirect
this decision; it's a step 3 calibration call.

---

## Finding 6 — Music SM silent collapses (G7) still present

`monotonicity_collapses.csv` for music hard:

| Level | Stage | Member | baseline F1 | hard F1 | Δ | classification |
|---|---|---|---:|---:|---:|---|
| hard | sm | `coma_hybrid` | 1.000 | 0.457 | -0.543 | unknown |
| hard | sm | `label_jw` | 1.000 | 0.364 | -0.636 | unknown |

Games has 0 collapses.

The K8 anonymized rename on music (G7 cause) is the load-bearing
input here. Step 3's C5 decision (option A committee gating vs
option B scope-only-to-secondary) directly governs whether these
two members survive on the next run.

---

## Step 3 decision packet (what the user picks)

Per plan_revision.md R-1 §Brainstorm callout, each item is a decision
for the user; Claude implements after sign-off. Step 2 has not changed
the option set; it has *confirmed* the underlying diagnoses. The
unchanged decision list:

| Item | Decision | Options |
|---|---|---|
| **C1** | Raise K2 `interpolation_count` 12 → ? | Depends on Finding 3b guardrail breakdown |
| **C1** | Relax `contamination_check`? | tighter / unchanged / looser |
| **C2** | Fix easy ≥ baseline | (a) regen-pool / (b) K1 easy floor 0.01/0.02/0.04 / (c) drop easy |
| **C3** | Audit metrics | Already adopted: distinct_families (K5), naming_intensity (K8), realised_swap_rate (K10) — these are the new load-bearing rows; step 3 just needs to confirm the legacy rows are demoted in R7.3 final reports |
| **C4** | EM-block ceiling cliff | (a) noise-aware sc_block retrain / (b) char-ngram BM25 blocker / (c) accept the cliff |
| **C5** | K8 anonymized scope | (A) committee gating / (B) anonymize secondary only |
| **C6** | Surface ceiling-responsiveness | Implementation note: best-member CSV is the readable artifact; ceiling_responsiveness column is supplementary at n=4. R7.3 reports should narrate from best-member trail |

**Suggested step-3 order (for the user to confirm):**

1. **C2** decision first — it sets whether the easy K1 floor / new
   pool affects the regen + audit interpretation of K1.
2. **C5** decision second — drives K8 config rewrite for music.
3. **C1** decision third — once Finding 3b lands, the user picks
   `interpolation_count` and whether to loosen contamination_check
   for music. Games may need a different lever (its K2 hard problem
   is partly the high natural 0.67 baseline; raising the LLM count
   alone won't move the dial enough at easy/medium).
4. **C4** decision fourth — code change (new blocker / retrain) is
   the heaviest item and depends on C1 + C2 outcomes (because the
   regenerated test pool composition is downstream of C2).
