# Knob 4 — Per-entity source coverage skew

**Status:** LOCKED. **Scenario:** S1 + S2 (fully controllable, bidirectional).

## Definition

Distribution of how many sources cover each entity (uniform vs long-tail). Operates on *entities*, not attributes (Knob 3 handles attribute presence within a row).

## Dimensions controlled

- Entity Group Size Variance (Fusion)
- Source Density (Fusion)
- Conflict Rate (Fusion)

## Sub-parameters

- `target_coverage_histogram` — dict keyed by integer group size, parameterized by source count `N`. For 3-source domains: `{1: x, 2: y, 3: z}` with `x+y+z=1`. Mean group size and variance are *derived* for reporting but not the canonical input.
- `within_source_duplicate_rate` — fraction of (entity, source) cells where the source carries an *additional* row for the same entity. 0 at easy, ~0 at medium, small-but-non-zero at hard. Duplicate rows are produced by passing the existing row through Knob 1's paraphrase pipeline so they look like independent re-entries.

## Easy / Medium / Hard

| Level | Target state | Generator action |
|---|---|---|
| **Easy** | Near-uniform coverage. Most entities covered by (nearly) all sources. Voting strategies always see ≥2 values per attribute. | LLM-fabricate a source-specific representation (style, schema, formatting of the target source) for entities missing from that source. Fabricated reps must be consistent with the fusion gold. |
| **Medium** | Mixed distribution. Substantial fraction fully covered, substantial fraction in 2 sources, small singleton tail. | Identity / baseline. |
| **Hard** | Long-tail. Most entities in only 1–2 sources. Fusion voting fails for much of the test set; fusion must lean on trust signals. | Remove entity rows from sources to create singletons, subject to fusion-gold floor and conflict-preserving removal. |

## Easy fabrication mode

- **Primary:** LLM generates a source-native representation. New row inherits the entity's gold-consistent values, re-expressed in source-native form.
- **Fallback:** "Hybrid" — copy the row from a sibling source, then push it through Knob 1's **medium-level** paraphrase pipeline (table dispatcher + EDA `random_swap` / `random_delete`, deterministic, no LLM) to break sibling-identity. Triggered when the LLM batch fails committee or plausibility checks. The Knob 1 entry point used here is the exported single-cell callable `paraphrase_value_for_knob_04` documented in [knob_01_surface_augmentation.md §Implementation handoff → Exported single-cell paraphrase callable](knob_01_surface_augmentation.md#exported-single-cell-paraphrase-callable-knob-4-fallback-consumer-locked-c3-contract); RNG is threaded from Knob 4's seeded generator via `SeedSequence.spawn()` so Knob 4's reproducibility is preserved.

## Composition

- **Knob 2:** Knob 4 runs **after** Knob 2 and treats Knob 2's placements as fixed input. Single-source distractors stay singletons by construction.
- **Knob 3:** orthogonal granularity. Knob 4 decides *if* a source has a row; Knob 3 decides *which attributes* of that row are NULL.
- **Knob 10:** composes downstream. After Knob 4 fixes coverage, Knob 10 reshuffles which surviving source carries the gold per (entity, attribute).

## Entity linkage and histogram semantics

Knob 4 needs entity-level grouping (which records across sources represent the same real-world entity) to build the presence matrix that feeds `H[k]` measurement. Two implementation-level decisions, surfaced during M6 validation (companies domain):

1. **Pool-based linkage.** Entity linkage is built from the union of EM gold correspondences (`usecases/<domain>/input/entitymatching/{train,val,test}_gold.csv`) and pooled positives (`usecases_synthetic/pools/<domain>/pooled_positives.csv`). EM gold alone is a sampled subset that under-counts cross-source links, inflating the singleton bin and misaligning `H_base` with the knob's intent. Pool-based linkage captures the full cross-source match surface.
2. **Distractor exclusion from the histogram denominator.** Knob 2 creates single-source distractors (records that never had a cross-source counterpart). These are singletons by construction and are not the kind of coverage signal Knob 4 controls. The histogram denominator therefore counts only *matchable* entities (those with at least one cross-source link in the union-find). Distractors are passed through unchanged and are *not* candidates for removal at hard or fabrication at easy. This keeps `H_base[1]` aligned with the authored targets and prevents distractor mass from swamping the coverage distribution.

These semantics are implemented in `usecases_synthetic/lib/coverage_ops.py:measure_coverage_histogram` (parameter `include_distractor_singletons=False`) and in `usecases_synthetic/scripts/apply_knob_04_coverage.py:build_entity_linkage`.

## Test-set treatment

- **Fusion:** entity membership frozen. Easy can fabricate new source reps for fusion-gold entities (gold unchanged); hard can remove rows from a fusion-gold entity's group only if the floor (≥1 near-gold survivor per cell) is preserved. No entity ever drops below 1 source.
- **EM:** test set regenerated per variant. Fabrication creates new positive pairs; removal eliminates some.

## Fusion-monotonicity guards

1. **Conflict-preserving removal at hard.** Same as Knob 3: prefer rows whose values agree with another surviving row.
2. **Committee check.** Inversion → soften removals or fabrication aggressiveness locally.
3. **Singleton cap at hard.** Explicit upper bound on the fraction of entities collapsed to single-source. Prevents fusion from becoming trivially "trust the only source." Default TBD.

## Committee expectations

- **Blocking:** mostly indifferent.
- **EM:** F1 affected mainly through changing the prior on singleton entities.
- **Fusion:** primary target. Voting strategies degrade faster than single-best-source / trust-weighted.

## Per-domain notes

- **Companies:** baseline already skewed (Forbes is the intersection pivot; DBpedia and FullContact cover largely disjoint tails). Sits near **medium→hard**. Easy fabrication is expensive (most of the corpus, not a tail).
- **Games:** uneven EM splits; long DBpedia tail (~26k entities with no Metacritic/Sales counterpart) offers natural easy headroom. **Caveat:** baseline measurement of coverage skew must come from raw record overlap, not EM gold (splits are pre-existing uneven).
- **Music:** strongly singleton-heavy by design. LastFM 9,865 records (not ~22k as metadata claimed). Easy requires aggressive fabrication; hard near baseline.

## Provenance

- **Fabricated row:** `transform_fn ∈ {llm_fabricate_rep, propagate_and_paraphrase}`, `transform_params={template_source, ...}`.
- **Within-source duplicate row:** `transform_fn=within_source_duplicate`, `transform_params={sibling_row_id, knob_01_paraphrase_params}`. (Locked enum extension — driven by the already-locked `within_source_duplicate_rate` sub-parameter; mechanically distinct from `propagate_and_paraphrase` because the sibling row lives in the *same* source.)
- **Removed row:** `transform_fn=remove_entity_row`, `transform_params={reason: "variance" | "density" | "singleton_target"}`.

## Algorithm selection

**Chosen approach.** Hybrid by level. **Medium** is identity (no-op passthrough — takes whatever coverage histogram emerges from the sources after Knob 2 has run). **Hard** is **Tier A** — deterministic in-house pandas + `numpy.random.default_rng` removal of entity rows from sources, driven by a target coverage histogram, constrained by a fusion-survivor floor, a conflict-preserving removal order, and a hard singleton cap. **Easy** is **Tier C** — deterministic baseline (the "hybrid" fallback: copy a sibling source row and push it through Knob 1's paraphrase pipeline) with an optional **LLM fabrication primary path** for source-native representations of entities missing from a target source. Easy fabrication is the only place Knob 4 touches the LLM; the medium/hard paths are pure pandas. The dispatcher (a) measures the baseline coverage histogram `H_base[k]` (`k` = number of sources covering an entity, `k ∈ 1..N`) from the post-Knob-2 entity × source presence matrix at the start of every run (never cached across runs — same reasoning as Knob 3: source data may have changed upstream), (b) computes the target histogram `H_target[k]` from the level + the per-domain target config, (c) computes the per-bin delta `Δ[k] = H_target[k] − H_base[k]` and decides whether to **add** coverage (easy shifts mass toward `H[N]`, the fully-covered bin) or **remove** coverage (hard shifts mass toward `H[1]`, the singleton bin), (d) selects which (entity, source) cells to add/remove subject to constraints, (e) mutates the source DataFrames. One seeded RNG per `(domain, variant, knob=4)` tuple. The function is pure pandas / `numpy` / `pyyaml` / stdlib for medium and hard; easy additionally imports the Knob 1 paraphrase wrapper (fallback path) and an LLM client stub (primary path, cached).

**Mapping to easy/medium/hard.** Monotonicity of *the coverage-skew difficulty signal* runs easy → medium → hard, where easy is **least skewed** (most entities fully covered) and hard is **most skewed** (long-tail, many singletons). Let `N` be the number of sources in the domain (N=3 for companies, games, music) and `H[k]` be the fraction of entities covered by exactly `k` sources.

| Level | Target `H_target[k]` | Generator action | Monotone direction |
|---|---|---|---|
| **Easy** | Authored per domain in YAML (mandatory). N=3 reference shape: `{1: 0.05, 2: 0.15, 3: 0.80}` — long-tail collapsed toward near-full coverage. | **Fabricate** new (entity, source) rows for under-covered entities until `H_target` is hit. Primary: LLM source-native fabrication. Fallback: sibling-copy + Knob 1 paraphrase. | Coverage **increases** vs. baseline (skew ↓). |
| **Medium** | `H_target = H_base` (identity). | No-op. Record `H_base` for the audit log. | Baseline. |
| **Hard** | Authored per domain in YAML (mandatory). N=3 reference shape: `{1: 0.55, 2: 0.30, 3: 0.15}` — long-tail amplified. | **Remove** entity rows from sources until `H_target` is hit, subject to constraints below. Per-level `within_source_duplicate_rate` adds a tiny within-source duplicate tail via the Knob 1 paraphrase pipeline. | Coverage **decreases** vs. baseline (skew ↑). |

`target_coverage_histogram` is **mandatory per domain** — no cross-domain default is provided, because no single histogram is reachable across companies (near medium→hard at baseline), games (long DBpedia tail), and music (singleton-heavy). The example dicts above (`{1: 0.05, 2: 0.15, 3: 0.80}` easy, `{1: 0.55, 2: 0.30, 3: 0.15}` hard) are N=3 reference shapes for visualization only; every domain YAML must author its own values grounded in its measured baseline. The absolute-target-band profile model is mandatory per [cross_cutting.md](cross_cutting.md#profile-model--option-b-absolute-target-bands) — easy is not "baseline minus delta"; it is a fixed target, and for already-skewed domains that means the easy path must fabricate aggressively.

Independent togglability: Knob 4 reads the post-Knob-2 entity × source presence matrix and writes only entity-level row additions/removals. It never touches cell values (Knobs 1/5/6/7), never drops attributes within a row (Knob 3), never reshuffles which source carries the gold (Knob 10), and never touches column headers (Knob 8).

**Monotonicity.** Unlike Knob 3, Knob 4 cannot use a shared per-cell uniform draw to guarantee nested operation sets: easy *adds* rows and hard *removes* rows, so the two levels act in opposite directions from baseline and there is no cell-level nesting between them. The monotonicity guarantee is therefore coarser and histogram-level, not cell-level. It holds along two independent axes:
1. **Authored targets are monotone by construction.** Per-domain YAML must author `H_target.easy`, `H_target.medium = H_base`, `H_target.hard` such that the singleton fraction `H[1]` is monotone non-decreasing easy → medium → hard (and correspondingly the fully-covered fraction `H[N]` is monotone non-increasing). The config loader validates this on load and refuses to run if violated.
2. **Realized histograms may deviate from targets only via constraints**, never above hard or below easy. The cross-level monotonicity smoke-test check is **stochastic dominance over the full ordering** `k = 1..N`, not just the endpoints `H[1]` / `H[N]`: for every `j ∈ 1..N`, `Σ_{k ≤ j} H_realized.easy[k] ≤ Σ_{k ≤ j} H_realized.medium[k] ≤ Σ_{k ≤ j} H_realized.hard[k]`. This pins middle bins as well as the endpoints — the singleton-shifted CDF must dominate at every rank. At hard the fusion-gold floor, pool-protection, conflict-preservation, and singleton-cap constraints can only *reduce* realized skew below the target. At easy the committee-fail-to-fallback path can only *reduce* fabrication quality, not fabrication count. Deviations are logged to `output/baselines/knob_04_realized_vs_target.csv` and surfaced to the committee loop. Cross-level monotonicity is then a property of the realized histograms (not the operation sets): `H_realized.easy[1] ≤ H_realized.medium[1] ≤ H_realized.hard[1]` is checked in the smoke test, and violation triggers the fix-on-collapse path.

**Constraint resolution order at hard** (applied *before* any row is actually removed):

1. **Fusion-gold floor.** For every entity in the fusion gold, at least one source must retain a row — no entity may drop to zero sources. Hard constraint enforced by the removal loop.
2. **Pooled-positives protection.** For every pair in `expanded_positives` (per [cross_cutting.md §Pooled positives set](cross_cutting.md#pooled-positives-set)), both endpoint rows must remain present **and remain in distinct sources** after removal — i.e., removal may not break a pool-declared match edge by collapsing both endpoints onto a single source. Single-source tail entities that happen to be in the pool but have no surviving partner row are unaffected by this constraint.
3. **Conflict-preserving removal.** Same semantics as Knob 3: prefer removing rows whose values are redundantly agreeing with another source. Rows participating in ≥ 2 disagreeing (conflict-graph) values on a fusion-gold entity are removed only after the redundant-agreement pool for that entity is exhausted. Protects downstream Knob 10 conflict signal.
4. **Singleton cap at hard.** After (1)–(3), count the realized fraction of single-source entities. If it exceeds the per-domain cap (default `singleton_cap_hard = 0.60`, authored in YAML), roll back the most recent removals in reverse draw order until the cap holds. Persistent rollbacks emit a calibration warning.
5. **Knob 2 distractor passthrough.** Single-source distractors created by Knob 2 are already singletons — they are passed through unchanged and counted toward `H_target[1]` but are *never* selected as removal candidates (they cannot be de-singletoned further).

**Joint cell-collision index integration (resolves C2 from the Step 5 cross-knob review).** Every cell on a K4-fabricated row carries the LLM/paraphrase touch already; if K1/K5/K7 then re-touched the same cell during the joint `1/5/6/7` phase, the result would be double-augmentation. To prevent that silently, the K4 dispatcher writes one row per fabricated cell into `output/provenance/knob_04_coverage_skew.csv` carrying the flag `k4_fabricated=True` in `transform_params`, *and* the joint cell-collision index that K1/5/6/7 read at dispatch time globs `output/provenance/knob_0{1,4,5,6,7}_*.csv` (not just `knob_0{1,5,6,7}_*.csv`). The composition rules per downstream knob:
- **K1, K5, K7**: skip K4-fabricated cells unconditionally (the cell is already the result of an LLM/paraphrase pass; another pass would be a no-op-best, double-perturbation-worst).
- **K6 (value noise)**: K4-fabricated cells are **fair game** for noise injection. Fabricated rows are new data, not gold carriers, and there is no semantic reason to exempt them from typo / OCR-style corruption — the difficulty signal compounds correctly. K6 reads the joint index but ignores the `k4_fabricated=True` flag specifically.

Knob 1's joint-index glob in [knob_01_surface_augmentation.md:82](knob_01_surface_augmentation.md#L82) and Knob 6's collision contract are mirrored to match.

**At easy the fabrication loop** iterates: for each entity where current coverage `k < H_target`-implied rank, select a target source missing that entity and fabricate a row for it. Primary path: LLM call with a source-native template (style, schema, formatting of the target source), seeded by the entity's fusion-gold values. Fallback path (triggers on LLM batch committee-check failure, contamination spot-check failure, or plausibility failure): copy the row from a sibling source that already carries the entity and push it through Knob 1's paraphrase pipeline to break sibling-identity. Both paths write a new row whose gold-relevant values remain consistent with the fusion gold (gold artifact never mutated); surface-level values are source-native.

**Fix-on-collapse.** Knob 4 is not listed in the [cross_cutting.md §Per-knob fix-strategy defaults](cross_cutting.md#per-knob-fix-strategy-defaults) table, so it inherits the default "soften augmentation locally" path: on committee monotonicity collapse, reduce the offending level's `|Δ[k]|` by a fixed step (default 0.05) and re-run. For easy-specific collapses driven by LLM fabrication quality (contamination, stilted output), an additional fix is to switch the offending entity's fabrication from the LLM primary path to the Knob-1-paraphrase fallback path before softening `Δ`. Fusion gold is never mutated.

**Literature citations.**

- **DAPO** ([../literature-search-generation/dapo_large_scale_data_pollution/paper.md](../literature-search-generation/dapo_large_scale_data_pollution/paper.md)) — *Per-Source Corruption Profiles* and the entity-presence dimension of its pollution taxonomy. Cited as the direct precedent for per-source entity-presence skew as a benchmark-generation knob. Our histogram-target approach is a strict generalization: rather than a per-source presence rate, we target a joint distribution `H[k]` over coverage ranks.
- **WDC Products Benchmark** ([../literature-search-generation/wdc_products_benchmark/paper.md](../literature-search-generation/wdc_products_benchmark/paper.md)) and **EMBench** ([../literature-search-generation/embench/paper.md](../literature-search-generation/embench/paper.md)) — cited for the singleton-tail methodology in the fusion-stage difficulty literature; EMBench's *Provenance-based ground-truth derivation* card in particular motivates the "fusion voting fails" regime at hard, where fabricated/removed rows preserve gold-by-construction.
- **Curated LLM Tabular Augmentation** ([../literature-search-generation/curated_llm_tabular_augmentation/paper.md](../literature-search-generation/curated_llm_tabular_augmentation/paper.md)) — *Serialized Prompting* + committee-validated curation. Cited *only* for the easy-level LLM fabrication primary path, not for the removal loop. Scoped, hygienic LLM use per [../plan_algorithmselection.md](../plan_algorithmselection.md) §Decision framework tiebreaker 3 and matching the Knob 1 LLM-hygiene pattern (pinned prompt version, `temperature=0`, cached + committed outputs, committee gating, two-step contamination check below).

No literature method is ported wholesale. Justification: the knob behavior (controlled, monotone, fusion-safe entity-presence histogram matching with conflict-preservation and pool protection) is a benchmark-engineering operation with no research content beyond the taxonomy anchor; a deterministic pandas implementation is strictly simpler, cheaper, and more reproducible than any ported library. The LLM fabrication sub-task at easy is where literature LLM-augmentation patterns earn their cite.

**Determinism & provenance.**

- **Baseline coverage histogram is measured fresh every run** from the post-Knob-2 entity × source presence matrix (same never-cache reasoning as Knob 3). Written to `output/baselines/knob_04_baseline_coverage.csv` inside the variant directory at the start of every Knob 4 invocation.
- **RNG.** Single `numpy.random.default_rng(seed)` per `(domain, variant, knob=4)` tuple; seed recorded in the variant's `config/difficulty.yaml`. Removal draws iterate in a canonical `entity_id × source` order so that level-unrelated entities do not reshuffle between runs.
- **LLM hygiene (easy path only).** Mirrors Knob 1's locked LLM hygiene contract (single source of truth — see [knob_01_surface_augmentation.md §Determinism & provenance → LLM hygiene](knob_01_surface_augmentation.md)). Specifically: fixed `prompt_version=v1` pinned in `usecases_synthetic/config/knob_04_coverage_skew/_prompts/fabricate_v1.txt`, `model_id` pinned in the per-domain YAML, `temperature=0`, fabricated outputs cached on disk and **committed to the repo** at `usecases_synthetic/cache/knob_04_fabrications/<domain>/<variant>/<entity_id>__<source>.json`, committee validation gating acceptance, and the **same two-step contamination spot-check Knob 1 hard uses**, applied to every fabricated row before admission: (1) **8-gram overlap probe** against the original gold value (per [benchmark_contamination_survey](../literature-search-generation/benchmark_contamination_survey/paper.md)) — fabricated row may not contain an 8-or-more-token contiguous overlap with any source's existing value for the same entity; (2) **first-token memorization probe** (per [elephants_never_forget_tabular_memorization](../literature-search-generation/elephants_never_forget_tabular_memorization/paper.md)) — fabricated entity name/title may not match the first few tokens of any *other* real entity in the same domain. Either failure routes the cell to the paraphrase fallback path. The LLM call is part of generation, not the runtime pipeline being benchmarked.
- **Per-domain config file:** `usecases_synthetic/config/knob_04_coverage_skew/<domain>.yaml`. Keys:
  - `source_count`: integer `N` — number of sources in the domain.
  - `target_coverage_histogram`: `{easy: {1: f, 2: f, ..., N: f}, medium: null, hard: {1: f, ..., N: f}}`. `medium: null` means "use measured baseline". Easy/hard fractions must sum to 1.0.
  - `within_source_duplicate_rate`: `{easy: 0.0, medium: 0.0, hard: 0.02}` (defaults; per-level scalar).
  - `singleton_cap_hard`: scalar. **Resolves the "Default TBD" from the locked card's Fusion-monotonicity guards section** — proposed default `0.70` (widened from the prior 0.60 to give realized-vs-target tolerance + constraint-induced rollbacks comfortable headroom; Companies overridden to `0.55`). Confirm during committee calibration at Step 8; tune downward if the hard variant collapses fusion voting to the point where committee-vs-pool agreement stays high while test-gold F1 crashes (indicates hidden-positive noise rather than real difficulty — see [cross_cutting.md §Protection set semantics](cross_cutting.md#protection-set-semantics-not-replacement-gold) point 3).
  - `delta_softening_step`: scalar (default 0.05) for the fix-on-collapse path.
  - `fabrication_mode`: `{easy: "llm_primary_with_paraphrase_fallback" | "paraphrase_only"}` — `paraphrase_only` disables the LLM primary path for domains where contamination risk is high or committee fails too often.
  - `llm_prompt_version`: string (pinned), `llm_model_id`: string, `llm_temperature`: float (default 0.0).
  - `protected_entity_ids`: auto-populated at runtime from `expanded_positives` plus the fusion gold — not authored by hand.
- **Provenance enum extension.** The locked Provenance section of this card (above) lists three `transform_fn` values: `llm_fabricate_rep`, `propagate_and_paraphrase`, `remove_entity_row`. The algorithm-selection implementation needs a fourth, `within_source_duplicate`, to log rows emitted by the already-locked `within_source_duplicate_rate` sub-parameter. This is an extension of the locked enum driven by an already-locked sub-parameter, not a new capability; flagged here for the cross-knob audit. Alternative considered: reuse `propagate_and_paraphrase` with a `{within_source: true}` flag in `transform_params`. Rejected because within-source duplication is mechanically distinct (sibling row lives in the *same* source, not a different one) and conflating the two would complicate the provenance query path.
- **Provenance written per affected row** to `output/provenance/knob_04_coverage_skew.csv` inside the variant directory, following the [cross_cutting.md §Per-value provenance](cross_cutting.md#per-value-provenance-mandatory) flat-row schema adapted to entity-scoped operations:
  ```
  (entity_id, source, attribute=<row>, original_value=<row_exists|row_absent>,
   new_value=<row_exists|row_absent>,
   transform_fn ∈ {llm_fabricate_rep, propagate_and_paraphrase, remove_entity_row, within_source_duplicate},
   transform_params, knob=4, level)
  ```
  `transform_params` JSON keys by `transform_fn`:
  - `llm_fabricate_rep`: `{template_source, prompt_version, model_id, cache_path, committee_passed: bool, contamination_check_passed: bool}`.
  - `propagate_and_paraphrase`: `{template_source, sibling_source, knob_01_paraphrase_params}`.
  - `remove_entity_row`: `{reason ∈ {"variance", "density", "singleton_target"}, conflict_preserved: bool}`.
  - `within_source_duplicate`: `{sibling_row_id, knob_01_paraphrase_params}`.
  - Skipped (constraint-protected) entities are emitted to `output/provenance/knob_04_skipped.csv` with `{reason ∈ {"fusion_gold_floor", "pool_protection", "conflict_preserve", "singleton_cap", "distractor_passthrough"}}`.
- **Fusion gold file is read-only**; byte-identical before and after every run. Entity membership in the fusion gold is frozen per [cross_cutting.md §Test-set treatment](cross_cutting.md#test-set-treatment).
- **Committee surface.** Fusion committee (primary target — voting strategies degrade faster than single-best-source / trust-weighted). EM committee sees changed positive pair counts due to fabrication/removal. Blocking committee mostly indifferent. The three committees are the ones [cross_cutting.md §Committee-validated augmentation](cross_cutting.md#committee-validated-augmentation) specifies.

**Domain-specific adjustments.**

- **Companies.** Baseline already skewed — Forbes is the intersection pivot; DBpedia and FullContact cover largely disjoint tails. Estimated baseline sits near medium→hard. **Easy fabrication is expensive** on Companies because most of the corpus (not a tail) must be fabricated to reach the easy band. Mitigation: `fabrication_mode: "paraphrase_only"` is the recommended setting to keep LLM cost tractable; the paraphrase fallback is cheap and still moves coverage. Hard target histogram for Companies pins closer to baseline with a modest singleton bump (`singleton_cap_hard: 0.50`), since the baseline is already skewed and the headroom for further removal is limited. Authored in `usecases_synthetic/config/knob_04_coverage_skew/companies.yaml`.
- **Games.** Uneven EM splits and a long DBpedia tail (~26k entities with no Metacritic/Sales counterpart). **Baseline measurement must come from raw record overlap, not EM gold** (see Per-domain notes in the main spec) — the dispatcher measures presence directly from source DataFrames, not from the EM gold, so this caveat is honoured by construction but should be re-verified in the smoke test. Games has generous easy headroom because fabricating Metacritic/Sales reps for DBpedia-only entities is straightforward (rich source-native templates exist).
- **Music.** Strongly singleton-heavy by design (LastFM = 9,865 records, not ~22k as metadata claimed). Baseline sits at-or-near hard. Easy requires aggressive fabrication; hard is near baseline — `H_target.hard` is pinned close to `H_base` with a small amplification of the singleton tier. Because of fabrication cost and contamination risk on Music titles, `fabrication_mode: "llm_primary_with_paraphrase_fallback"` is used but contamination spot-checks are tightened (fail rate threshold 0.05 rather than the default 0.10).
- **Movies, products.** Deferred alongside the Step 6 prototype. The dispatcher no-ops on domains without a `config/knob_04_coverage_skew/<domain>.yaml` (warns in the log). No code change required when those domains come online — only a new YAML plus (for movies) acknowledging the weaker single-source pool protection.

**Rejected alternatives.**

- **Pure LLM fabrication at hard** (prompt the LLM to produce a skewed coverage distribution in one shot). Rejected: the hard path is a parametric sampling task over a fixed presence matrix with monotonicity, fusion-floor, pool-protection, conflict-preservation, and singleton-cap constraints. Deterministic pandas trivially satisfies all of these; LLM use would sacrifice determinism for zero expected quality gain. **LLM not used at hard because the deterministic alternative is sufficient.**
- **Pure paraphrase at easy, no LLM at all.** Rejected as the *primary* path but retained as the *fallback*. Paraphrased sibling-copy rows are structurally identical to the sibling and carry sibling-identity signal that source-native fabrication avoids; for domains where contamination risk is low enough that LLM primary is safe (Games especially), source-native fabrication materially improves realism. Companies, where fabrication must operate on most of the corpus rather than a tail, *does* use paraphrase-only.
- **Heavyweight generative models** (GReaT / CTGAN / TabDDPM conditional generation for fabricated source rows). Rejected under the [plan_algorithmselection.md](../plan_algorithmselection.md) framework rule against heavyweight ML — they violate determinism, validation cost, and dependency weight simultaneously, and they optimize for realistic joint distributions, not for controlled source-native-style fabrication consistent with a *specified* fusion-gold value set.
- **Statistical missingness models at the entity level** (MAR/MNAR for entity presence). Rejected for the same reason Knob 3 rejected them: they *learn* a plausible mechanism from the data and would silently absorb the very skew signal we are trying to control. We want transparent `H_target`, not inferred.
- **Mutating Knob 2's entity placements to hit coverage targets.** Rejected: Knob 2 and Knob 4 are explicitly orthogonal per [README.md](README.md#canonical-knob-application-order) — Knob 4 takes Knob 2's placements as fixed input. Mixing them would entangle niche-density difficulty with coverage-skew difficulty and break independent togglability (ablation hard requirement).
- **Caching the baseline histogram across runs** for speed. Rejected per the same reasoning as Knob 3: source data can change between runs, measurement is cheap (`groupby(entity_id)["source"].nunique()` → `value_counts`), and a stale baseline would silently break monotonicity.
- **Column-level or attribute-level operations.** Rejected: that is Knob 3 (attribute drop within a row) and Knob 9 (column-level schema omission). Knob 4 is entity-row granularity only.

**Implementation handoff.** Everything Step 6 needs to implement this knob without re-reading surrounding cards:

- **Target script:** `usecases_synthetic/scripts/apply_knob_04_coverage_skew.py` (new, convention matches `apply_knob_03_attribute_drop.py`). Standalone runnable from repo root.
- **Function shape (illustrative):**
  ```python
  def apply_knob_04(
      domain: str,
      level: Literal["easy", "medium", "hard"],
      sources: dict[str, pd.DataFrame],           # post-Knob-2, pre-Knobs-1/5/6/7
      fusion_gold: pd.DataFrame,                  # read-only; fusion-gold floor lookup
      expanded_positives: pd.DataFrame,           # pooled_positives ∪ {train,val,test}_gold
      config_path: Path,                          # usecases_synthetic/config/knob_04_coverage_skew/<domain>.yaml
      output_dir: Path,
      seed: int,
      llm_client: LLMClient | None = None,        # easy-path only; None ⇒ paraphrase fallback forced
      knob_01_paraphrase: Callable | None = None, # wrapper around Knob 1's paraphrase pipeline
  ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
      """Returns (mutated_sources, provenance_df, skipped_df, baseline_histogram_df).

      baseline_histogram_df is measured fresh from `sources` at the start of every
      call and written to output/baselines/knob_04_baseline_coverage.csv.
      """
  ```
- **Inputs the script reads:**
  - Source DataFrames (via `PyDI.io.load_*`, preserving `df.attrs["dataset_name"]`), already mutated by Knob 2 upstream in the canonical S1 order.
  - The fusion gold artifact from `usecases/<domain>/input/fusion/` — read-only, used for the fusion-gold floor; **never mutated** (byte-identical before and after).
  - `expanded_positives` — the pool artifact at `usecases/<domain>/input/entitymatching/pooled_positives.csv` unioned with `{train,val,test}_gold` per [cross_cutting.md §Pooled positives set](cross_cutting.md#pooled-positives-set). Already built for companies/games/music per [plan.md](../plan.md) Step 4.
  - Per-domain config at `usecases_synthetic/config/knob_04_coverage_skew/<domain>.yaml`.
  - (Easy path) Knob 1 paraphrase callable (fallback) and an LLM client (primary path).
- **Outputs the script writes** (under the variant directory):
  - Mutated source files in `input/data/` (same format as input — XML/JSON/CSV).
  - Freshly measured baseline histogram at `output/baselines/knob_04_baseline_coverage.csv` (every run; not cached across runs).
  - Provenance log at `output/provenance/knob_04_coverage_skew.csv`.
  - Skipped-entity audit at `output/provenance/knob_04_skipped.csv` (entities spared by fusion-gold / pool-protection / conflict-preserve / singleton-cap / distractor-passthrough).
  - (Easy path) LLM fabrication cache at `usecases_synthetic/cache/knob_04_fabrications/<domain>/<variant>/<entity_id>__<source>.json`, **committed to the repo**.
- **Pipeline integration:** Knob 4 sits at position 2 of the canonical S1 order from [README.md](README.md#canonical-knob-application-order), running after `Knob 2 (niche density)` and before `Knobs 1/5/6/7 (value perturbations)`. It sees Knob 2's entity placements as fixed input and produces the coverage histogram that Knobs 1/5/6/7, Knob 3, Knob 10, Knob 8 then operate on.
- **Dependencies:** stdlib + `pandas` + `numpy` + `pyyaml` for medium/hard. Easy path additionally needs the Knob 1 paraphrase wrapper (in-repo) and an LLM client (OpenAI-compatible, loaded via `python-dotenv` per [CLAUDE.md](../CLAUDE.md#testing-notes)). No new runtime dependencies beyond what Knob 1 already introduces.
- **Authoring task before first run:** populate `usecases_synthetic/config/knob_04_coverage_skew/{companies,games,music}.yaml` with `source_count`, `target_coverage_histogram`, `within_source_duplicate_rate`, `singleton_cap_hard`, `fabrication_mode`, `llm_prompt_version`, `llm_model_id`, `llm_temperature`. Baseline histogram is **measured**, not authored.
- **Smoke test:** for each domain with a config, run the script at all three levels and assert (a) the fusion gold file on disk is byte-identical before and after the run, (b) every fusion-gold entity retains at least one source row, (c) every `expanded_positives`-protected match edge still connects two non-empty source rows, (d) realized coverage histogram matches `H_target` within a per-level tolerance (default ±0.03 per bin, modulo constraint-induced rollbacks), (e) at hard, the singleton fraction does not exceed `singleton_cap_hard`, (f) at medium, `input/data/` source files are unchanged vs. their pre-Knob-4 state (the dispatcher skips re-serialization at medium; only `output/baselines/knob_04_baseline_coverage.csv` is written), (g) the provenance row count equals the number of row-level mutations (fabrications + removals + within-source duplicates), (h) no column is ever added or removed (columns are Knobs 8/9 territory), (i) easy-path LLM cache files exist for every `llm_fabricate_rep` provenance row, (j) re-running with the same seed on the same source snapshot produces bit-identical outputs **when the fabrication cache is populated** — first-run cache-miss invocations hit the LLM and bit-identity is guaranteed only from the second run onward, (k) the baseline histogram CSV mtime is ≥ start-of-run (proves fresh measurement), (l) the realized-vs-target histogram CSV shows `H_realized.easy[1] ≤ H_realized.medium[1] ≤ H_realized.hard[1]` (cross-level histogram-monotonicity check, per the monotonicity guarantee above).
