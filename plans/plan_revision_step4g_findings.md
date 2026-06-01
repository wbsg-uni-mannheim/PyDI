# plan_revision_step4g_findings.md

Investigation note for G10 — "Games EM committee baseline F1 is much
worse than the human-baseline notebook's EM F1 on the same data."
Per [plan_revision.md §4g](plan_revision.md#L1698-L1719). One-time
investigation; no code change in this note. Findings drive YAML / loader
fixes that must land before the step 6 games rerun.

Run date: 2026-05-26.

## TL;DR

The gap is dominated by **a silent loader skip of the `metacritic ↔
dbpedia` source pair at baseline** (the committee evaluates ONE pair on
games; the notebook evaluates TWO). The lesser cause is **lack of
platform-alias normalization before matching** on the one pair the
committee does evaluate (`dbpedia_sales`). The committee's blocker
recall is fine (~1.0 on the evaluated pair); the gap sits in the matcher
input pipeline, not in blocking.

Recommended fix before the step 6 games rerun:

1. **Loader fix (required, structural)** — make
   `lib/variant_loader.py:_load_em_gold` direction-tolerant, mirroring
   the existing direction-tolerant logic in `_load_em_gold_regenerated`
   ([variant_loader.py:391-394](../usecases_synthetic/lib/variant_loader.py#L391-L394)).
   Restores the `(metacritic, dbpedia)` pair to the games baseline.
2. **Platform-alias preprocessing for games (recommended, scoped)** —
   port the notebook's platform alias map into a preprocessing hook on
   the games column_mapping so committee matchers see normalized
   platform values before scoring. See "Attribution" §2 below for the
   exact map.
3. **Tolerance for step 7 verification** — once (1) lands, expect
   committee per-pair F1 within ≤10pp of notebook on `dbpedia_metacritic`
   (notebook: 0.889; committee target: ≥0.79) and `dbpedia_sales`
   (notebook: 0.900; committee target: ≥0.80). If (2) also lands, the
   tighter target ≤5pp becomes attainable.

## Inputs reviewed

- Notebook: [usecases/games/games_workflow.ipynb](../usecases/games/games_workflow.ipynb)
  (~3.9k lines; cached execution outputs intact — captured
  2026-05-15T12:48-12:50 against the source-side input directory).
- Committee configs:
  [config/committees/em_blocking_committee_games.yaml](../usecases_synthetic/config/committees/em_blocking_committee_games.yaml)
  and [em_matching_committee_games.yaml](../usecases_synthetic/config/committees/em_matching_committee_games.yaml).
- Domain config: [config/domains/games.yaml](../usecases_synthetic/config/domains/games.yaml).
- Committee baseline output:
  [baselines/games/baseline_metrics.json](../usecases_synthetic/baselines/games/baseline_metrics.json)
  (written 2026-05-16T12:31:14Z).
- Gold files on disk: [usecases/games/input/entitymatching/](../usecases/games/input/entitymatching/).
- Loader: [lib/variant_loader.py](../usecases_synthetic/lib/variant_loader.py).

## Notebook method stack

### Source pairs evaluated

Two: `dbpedia × metacritic` ("m2d") and `dbpedia × sales` ("m2s").
`metacritic × sales` is NOT in the notebook (consistent with the F11
drop in games.yaml).

### Per-pair preprocessing (applied BEFORE blocking/matching)

- Platform normalization via `platform_aliases` map covering ~25
  variants → canonical (`NES` → `Nintendo Entertainment System`,
  `PS1/PSX/PS` → `Playstation`, `XOne` → `Xbox One`, `Microsoft Windows`
  → `PC`, etc.). Applied symmetrically to `dbpedia.system`,
  `metacritic.console`, `sales.hw` ([games_workflow.ipynb cell 10](../usecases/games/games_workflow.ipynb)
  + the smaller `platform_aliases` map used by the m2d blocker /
  matcher).
- Franchise suffix scrub on dbpedia (`" (video game)"` regex removal).
- Title normalization in the m2d comparators only — strips
  `"(... video game ...)"`, `"hd"`, `"remaster(ed)"`,
  `"definitive/special/complete/goty edition"`, etc., before
  computing similarity.
- Genre string → list conversion.
- ReleaseYear normalized to year-start datetimes.
- Trust scores stamped: metacritic=3, sales=2, dbpedia=1.

### Blockers (one per pair)

- **m2d**: custom `UnionTitleTokenBlocker` — union of (title-token +
  platform) ∪ (title-token + release-year) blocks. Title tokens
  filtered through a stopword list (`the, of, and, edition, game, video,
  ii, iii, iv` …). NOT a built-in PyDI blocker.
- **m2s**: `StandardBlocker(on=['name_longest_token'])` — derived key
  is the single longest token of the name field.

### Matchers

`RuleBasedMatcher` for both pairs, threshold-tuned per pair:

- **m2d** — custom Python comparators (not PyDI primitives):
  - `name_sequence_similarity` — `difflib.SequenceMatcher` on the
    aggressively-normalized title (weight 0.65).
  - `name_token_overlap_similarity` — token-set overlap / min(|s1|,
    |s2|) on the same normalized title (weight 0.25).
  - `platform_exact_similarity` — 1.0 iff normalized platforms equal,
    0.0 otherwise (weight 0.10).
  - Threshold: **0.98** (high, tuned for the high-recall blocker).

- **m2s** — PyDI primitives:
  - `StringComparator(column='name', similarity_function='jaccard',
    preprocess=str.lower)` (weight 0.6).
  - `StringComparator(column='platform', similarity_function='jaccard')`
    (weight 0.3).
  - `DateComparator(column='releaseYear', max_days_difference=360)`
    (weight 0.1).
  - Threshold: **0.8**.

### Post-processing

`MaximumBipartiteMatching` clusterer applied AFTER the raw matcher on
both pairs (forces 1:1 reduction). Reported numbers go BOTH ways —
notebook prints raw-matcher F1 and post-MBM F1.

### Scoring + test files

- `EntityMatchingEvaluator.evaluate_matching` (closed-set on the test
  gold).
- Test gold: `usecases/games/input/entitymatching/dbpedia_2_metacritic_test.csv`
  + `dbpedia_2_sales_test.csv`. Loaded as positive+negative pairs with
  `add_index=False, header=None, names=['id1','id2','label']`.

### Notebook reported F1 (cached cell outputs)

| Pair | Stage | Precision | Recall | F1 | TP / FP / FN |
|---|---|---:|---:|---:|---:|
| dbpedia × metacritic | raw matcher | 0.957 | 0.830 | **0.889** | 88 / 4 / 18 |
| dbpedia × metacritic | post-MBM | 1.000 | 0.481 | 0.650 | 51 / 0 / 55 |
| dbpedia × sales | raw matcher | 1.000 | 0.819 | **0.900** | 95 / 0 / 21 |
| dbpedia × sales | post-MBM | (≈ same or higher) | | | |

The raw-matcher F1 is the load-bearing notebook number; MBM hurts m2d
recall because dbpedia has many-to-many candidate clusters that 1:1
reduction collapses incorrectly. The committee should be compared to
**raw-matcher F1: 0.889 (m2d) and 0.900 (m2s)**.

## Committee method stack

### Source pairs evaluated

From `config/domains/games.yaml`:

```yaml
source_pairs:
  - [dbpedia, sales]
  - [metacritic, dbpedia]
  # [metacritic, sales] dropped 2026-05-16 — F11 fix removed the EM gold
```

Declared: two pairs. **Actually evaluated at baseline: ONE pair
(`dbpedia_sales`)**. See "Attribution" §1 below for why.

### Per-pair preprocessing

Limited to whatever the column_mapping renames declare
([em_blocking/matching_committee_games.yaml](../usecases_synthetic/config/committees/em_blocking_committee_games.yaml#L148-L174)):
DBpedia `system`/`title`/`launch_yr`/`studio`/`genre`/`franchise` →
canonical names; Metacritic and Sales symmetric. No platform-alias
normalization, no title-suffix scrub, no genre string→list conversion
at the committee level.

### Blockers (committee — `select_best`)

6-member roster, recall_floor=0.97, tie_breaker=reduction_ratio.

| Member | Pair recall @ dbpedia_sales | RR | Notes |
|---|---:|---:|---|
| token_blocker | 1.000 | 0.953 | Lexical |
| **standard_blocker** | **1.000** | **0.999** | **Selected winner** (`name_first_5`) |
| embedding_blocker | 1.000 | 0.994 | BAAI/bge-base-en-v1.5 |
| sorted_neighbourhood_blocker | 0.991 | 0.999 | name_norm + window=40 |
| bm25_blocker | 1.000 | 0.994 | char_ngram (3,5) post-C4 |
| sc_block | 1.000 | 0.994 | RoBERTa SupCon |

Blocking recall is fine — multiple blockers clear 0.97 floor with recall
1.0 on the one pair evaluated. The notebook's own m2s blocker
(StandardBlocker on `name_longest_token`) gets pair_completeness=1.0
with reduction_ratio=0.999 on the same pair — committee matches it.

For m2d the notebook's `UnionTitleTokenBlocker` reports 0.991 pair
completeness on the same gold — directly comparable to the
committee's blocker recall ceiling, and within noise of what the
generic blockers achieve. Blocking is NOT the gap source.

### Matchers (committee — all 4 run)

| Member | Pair | F1 | Precision | Recall | TP / FP / FN |
|---|---|---:|---:|---:|---|
| ditto_plm | dbpedia_sales | 0.586 | 0.426 | 0.940 | 109 / 147 / 7 |
| magellan | dbpedia_sales | 0.602 | 1.000 | 0.431 | 50 / 0 / 66 |
| llm_matcher | dbpedia_sales | 0.760 | 0.961 | 0.629 | 73 / 3 / 43 |
| **comem** (best) | dbpedia_sales | **0.771** | 0.974 | 0.638 | 74 / 2 / 42 |

`aggregated.macro_f1 = 0.6799` (mean across 4 members on the single
evaluated pair).

## Per-pair gap

### dbpedia × metacritic

| Source | F1 |
|---|---:|
| Notebook (raw matcher) | **0.889** |
| Committee baseline | **N/A (silently skipped)** |

### dbpedia × sales

| Source | F1 | Precision | Recall |
|---|---:|---:|---:|
| Notebook (raw matcher) | **0.900** | 1.000 | 0.819 |
| Committee best member (comem) | **0.771** | 0.974 | 0.638 |
| Gap | **−12.9pp F1** | −2.6pp | **−18.1pp recall** |

The committee under-predicts on dbpedia_sales — recall is 18pp worse.
Precision is essentially the same. Each committee matcher member has a
distinct failure profile: ditto_plm over-predicts (low precision),
magellan under-predicts (low recall), the two LLM members hover at the
same precision/recall trade-off near the notebook's profile but ~18pp
short on recall.

## Attribution

### §1. Primary cause — silent loader skip of `(metacritic, dbpedia)` (~half of games EM workload)

**Bug location:**
[lib/variant_loader.py:_load_em_gold (313-342)](../usecases_synthetic/lib/variant_loader.py#L313-L342).
For each declared pair `(src1, src2)`, the function looks ONLY for
`<src1>_2_<src2>_all.csv` (preferred) or `<src1>_2_<src2>_test.csv`
(fallback). It does NOT try the reverse direction `<src2>_2_<src1>_*`.

**State on disk:**
[usecases/games/input/entitymatching/](../usecases/games/input/entitymatching/)
holds `dbpedia_2_metacritic_test.csv` and `metacritic_2_dbpedia_train.csv`
— the test gold for the metacritic↔dbpedia pair exists, but only in
the `dbpedia_2_metacritic` direction. The complementary
`metacritic_2_dbpedia_test.csv` does not exist (it was never created
when the file-naming convention was tightened; the `old/train_test/`
backup has the same asymmetry under the no-`_2_` naming).

**Effect:** for `source_pair = (metacritic, dbpedia)`, the lookup misses,
the pair is silently dropped from `VariantBundle.em_gold` (the
`source_pairs` property returns `list(self.em_gold.keys())`, so
downstream code never sees the pair).
[baseline_metrics.json em_matching.per_partition](../usecases_synthetic/baselines/games/baseline_metrics.json#L2806-L2813)
shows only `dbpedia_sales`; macro_f1 is computed on n=1 pair, not the
declared n=2.

**Contrast:**
[_load_em_gold_regenerated (391-394)](../usecases_synthetic/lib/variant_loader.py#L391-L394)
already handles this correctly — it tries both
`<src1>_2_<src2>_<split>_<version>.csv` and
`<src2>_2_<src1>_<split>_<version>.csv`. The asymmetry between the two
loaders is the root structural defect.

**Why this dominates the gap:** the dbpedia_metacritic pair (notebook
F1 0.889 — the *higher* of the two pairs) is dropped entirely. Even
if the committee matched the notebook exactly on dbpedia_sales (which
it doesn't), the macro-F1 over both pairs would be (0.889 + 0.900) /
2 ≈ 0.895 for the notebook vs the committee's 0.771 single-pair number
= 12.4pp macro gap before considering any other defect.

### §2. Secondary cause — no platform-alias normalization before matching (on dbpedia_sales)

The notebook applies an extensive platform-alias map BEFORE matching
([games_workflow.ipynb cell 10](../usecases/games/games_workflow.ipynb)):
~25 entries spanning all Nintendo/Sony/Microsoft generations:

```
NES → Nintendo Entertainment System
SNES, Super Nintendo Entertainment System → Super Nintendo
N64 → Nintendo 64
GC, Nintendo GameCube → GameCube
Nintendo Wii → Wii
GBC → Game Boy Color
PS1, PSX, PS, Playstation 1 → Playstation
PSP → Playstation Portable
PSV, PS Vita → Playstation Vita
XB, Xbox (console) → Xbox
XOne → Xbox One
X360 → Xbox 360
Microsoft Windows, Windows → PC
```

…plus the m2d blocker uses a parallel-but-stricter alias map that
lowercases and collapses more variants (`pc, microsoft windows,
windows → pc`).

The committee column_mapping just renames source columns (`system → platform`,
`console → platform`, `hw → platform`) — it does NOT canonicalize the
*values*. As a result, the committee matchers see raw values:
metacritic's "PS3" vs dbpedia's "Playstation 3" or sales's "PS3 (console)"
are different strings to every matcher in the roster. Magellan's
comparator vector treats them as a mismatch; ditto_plm's text serialisation
includes the literal strings; LLM-matchers can sometimes recover via
world knowledge but inconsistently (comem at 0.974 precision suggests it
mostly does, but loses recall when the cleaned platform would have
flipped a "no" to a "yes").

Notebook recall on dbpedia_sales: **0.819**. Committee best
(comem) recall: **0.638**. The 18pp recall gap is consistent with
platform-value disagreement on a meaningful fraction of the 116
gold positives — same-game-different-platform-strings would correctly
NOT match, but same-game-same-platform-different-spelling rejected by
the matcher costs recall.

### §3. Minor cause — no per-pair tuning, no rule-based member in committee roster

The notebook hand-tunes the matcher per pair: threshold 0.98 + custom
title comparators for m2d (high-recall blocker demands a strict
matcher), threshold 0.8 + PyDI primitives for m2s (lower-recall
blocker, more lenient matcher). The committee runs the same 4 matchers
× threshold 0.5 on every pair. Single-threshold-fits-all is
structurally weaker than per-pair tuning, but with only two pairs
and the larger §1/§2 effects above, this is a tertiary concern.

The notebook's `RuleBasedMatcher` has no analogue in the committee
roster (ditto/magellan/llm_matcher/comem are all learned or LLM-based).
Whether to add a tunable rule-based member is a structural design
question; it should NOT block the step 6 games rerun, but is worth
revisiting if §1 + §2 close the gap insufficiently.

### §4. Non-cause — scoring semantics (post-C10)

The notebook uses `EntityMatchingEvaluator.evaluate_matching` (closed-
set against the explicit test gold CSV). Post-C10 (landed 2026-05-25)
the committee surfaces `f1_baseline_test` as closed-set on the same
`em_test_baseline_pruned.csv` ([committee_em.py:_score_predictions](../usecases_synthetic/lib/committee_em.py)).
At baseline (no K2), `em_test_baseline_pruned.csv` IS the original test
gold (no regen has happened), so this is apples-to-apples.

The committee's reported `f1_vs_test=0.5860..0.7708` in
[baseline_metrics.json](../usecases_synthetic/baselines/games/baseline_metrics.json#L2814-L2972)
predates C10 but is computed against the same test set the notebook
uses, so the comparison stands.

### §5. Non-cause — blocker recall ceiling at baseline

All committee blockers clear 0.97 floor on the dbpedia_sales pair,
multiple at pair_recall=1.0 — matching the notebook's blocker recall
ceiling on that pair. There is no sub-ceiling at baseline.

(G6 of the gap analysis still stands for higher-difficulty levels —
hard-level EM blocking ceiling collapses ~0.40-0.45 universally — but
that's a separate phenomenon driven by K1+K8+K10, not the baseline
gap investigated here.)

## Decision tree (per the plan's framing)

| Question | Verdict |
|---|---|
| Blocker recall sub-ceiling at baseline? | NO |
| Matcher composition (notebook methods committee lacks)? | PARTIAL — no rule-based + per-pair tuning + clusterer in committee. Tertiary effect. |
| `text_cols` / blocking-key config mismatch? | NO for blocking; YES indirectly for matching (platform values not canonicalized — §2 above). |
| Scoring semantics? | NO (post-C10 apples-to-apples). |
| **Loader / config bug silently skipping pairs?** | **YES (§1) — dominant.** |

## Proposed follow-up

### Required, structural — landing as code changes before step 6

1. **Direction-tolerant `_load_em_gold`** —
   [lib/variant_loader.py](../usecases_synthetic/lib/variant_loader.py).
   Mirror the existing direction-tolerance in `_load_em_gold_regenerated`
   (try both `<src1>_2_<src2>_*` and `<src2>_2_<src1>_*` candidates).
   Restores `(metacritic, dbpedia)` to the games baseline EM workload
   without renaming any on-disk file. Add a unit test exercising the
   reversed-direction lookup. **Estimated effort: ~30 LOC + 2 tests.**

   Note: when the file is found in the reversed direction, the loader
   must NOT swap `id1` / `id2` columns — the gold pairs are symmetric
   (label is unordered), so reading them as-is preserves correctness.
   This matches `_load_em_gold_regenerated`'s current behavior.

### Recommended, scoped — config-only YAML changes before step 6

2. **Platform value canonicalization for games** — add a preprocessing
   hook to the games column_mapping (or to the matcher input pipeline)
   that applies the notebook's platform alias map to `platform` values
   in all three sources before scoring. Concretely: either a new
   `value_normalize: {<column>: <alias_map_yaml>}` block under
   column_mapping in
   [em_matching_committee_games.yaml](../usecases_synthetic/config/committees/em_matching_committee_games.yaml#L114),
   or a tiny callable hook on the games loader. The alias map should
   match the notebook's (cell 10 — ~25 entries). Symmetric for
   `em_blocking_committee_games.yaml` to keep blocking key derivation
   consistent.

   This is more invasive than (1) — it touches the committee runner's
   preprocessing path. If the design committee considers value-level
   canonicalization out of scope for the EM committee (i.e. "that's a
   normalization-committee responsibility"), an equivalent place to put
   it is the **normalization committee** which already has per-domain
   reach. The decision sits with the user; ship (1) regardless.

### Verification tolerance (Step 7 contract for games)

With (1) only: target committee per-pair F1 within **≤10pp** of
notebook on both pairs:
- `dbpedia_metacritic`: notebook 0.889 → committee target ≥ 0.79
- `dbpedia_sales`: notebook 0.900 → committee target ≥ 0.80
  - committee baseline best (comem) currently at 0.771 → already 13pp
    short. Needs (2) or improved matcher recall via threshold tuning to
    close this. If (2) lands, expect ~0.85+ on this pair (notebook's
    primary advantage on m2s was platform comparison going from 0 → 1
    on aliases).

With (1) + (2): tighter target **≤5pp** is attainable:
- `dbpedia_metacritic`: ≥ 0.84
- `dbpedia_sales`: ≥ 0.85

If (1) lands but the gap on dbpedia_sales does not close after (2) is
applied, the residual is the per-pair-tuning / rule-based-matcher gap
(§3) and becomes a separate structural work item — does NOT block step
6, but logs as carry-over.

### Non-blockers — log as separate items if pursued

3. **Rule-based matcher in the committee roster** — open as a structural
   committee-design discussion, not a gating item. Would require
   per-pair comparator + threshold configuration which the current
   committee runner does not support.
4. **`MaximumBipartiteMatching` post-processor** — the notebook uses it
   and it HURTS m2d F1 (0.889 → 0.650). Don't replicate.

## Couples with

- **G6** (EM-block ceiling collapse at hard, plan_revision.md): the
  blocker recall ceiling that holds at baseline collapses at hard via
  K1+K8+K10. The §1 fix above is orthogonal to G6 and should land
  regardless.
- **C10** (EM committee scoring rewrite, landed 2026-05-25): the §4
  non-cause analysis depends on C10's closed-set semantics. The
  baseline_metrics.json file used here predates C10 but the underlying
  test set + scoring formula are equivalent (closed-set on the same
  gold file). After the step 6 rerun, the committee output will report
  the C10 keys (`macro_f1_baseline_test`); the comparison numbers
  remain directly usable.
- **Step 4h knob review**: independent. The §1 / §2 fixes here are
  not knob configuration changes and don't need to wait for or be
  bundled with 4h.
