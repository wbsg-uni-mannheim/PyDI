# Knob 2 — Entity niche density

**Status:** LOCKED. **Scenario:** S1 (bidirectional, downward bounded by fusion gold) + S2 (fully controllable).

## Definition

How many similar-but-distinct entities cluster in the same semantic neighborhood (franchises, artist discographies, subsidiaries, sequels/remasters, label-collisions).

## Dimensions controlled

- Candidate Density (Block)
- Corner-Case Ratio (EM) — explicit sub-parameter
- Corner-Case Difficulty (EM) — shared with Knob 1

## Sub-parameters

- `corner_case_ratio` — explicit fraction of EM pairs that are corner cases. EM test set is regenerated per variant to track this.
- `placement_split` — for interpolated entities, the share placed across multiple sources (matched near-twins → hard positives + hard negatives) vs. inserted into a single source only (unmatched distractors → pure hard negatives at blocking/EM). **Mixed placement (Option c).** Default pinned at algorithm selection: 60/40 at hard. Medium does no interpolation, so `placement_split` is not consulted at medium (the prior "70/30 at medium" entry was a leftover from an interpolation-at-medium draft and is dropped).
- `niche_metric_set` — multiple similarity signals combined: lexical-niche, embedding-niche, attribute-overlap, **and label-collision** (shared primary label across otherwise dissimilar entities — added per the Knob 7 rescoping). Reduces single-metric selection bias.

## Easy / Medium / Hard

| Level | Target state | `corner_case_ratio` | Generator action |
|---|---|---|---|
| **Easy** | Sparse neighborhoods. Most entities have no near-twin. Blockers return small candidate sets dominated by true matches. | ~20% | Niche-aware *removal*: drop non-fusion-gold entities from crowded clusters until clusters thin out. |
| **Medium** | Moderate clustering. A meaningful fraction of entities have 1–2 near-twins. Blockers return mixed candidate sets. | ~50% | Add or remove toward target. |
| **Hard** | Dense neighborhoods. Most entities sit in tight clusters of near-twins. Blockers return large candidate sets where the right match differs only on a few discriminative attributes. | ~60–80% (achieved ratio per-domain; see Algorithm selection level table) | Heavy *interpolation*: LLM-generate near-twin entities seeded from existing ones, with mixed placement. |

## Composition

- **Pooled-positives floor:** entities in `expanded_positives = fusion_gold ∪ EM_gold(train/val/test) ∪ pooled_positives` are never dropped, even at easy, and are never used as parent seeds for single-source distractor interpolation. This replaces the narrower "fusion-gold floor" and protects hidden true matches surfaced by the pooling systems. See [cross_cutting.md](cross_cutting.md#gold-standard-incompleteness-and-pooling). EM test sets are still regenerated per variant, but entity *removability* is governed by the pooled set, not the test split alone.
- **Knob 4:** runs **after** Knob 2 and treats Knob 2's placements as fixed input. Single-source distractors stay singletons by construction.
- **Pipeline order:** **first** in the canonical order (S1 + S2). See [README.md](README.md#canonical-knob-application-order).

## Test-set treatment

EM test set regenerated per variant to hit the explicit `corner_case_ratio`. Regeneration draws from the full set of candidate pairs over the post-Knob-2 canonical set (same-cluster pairs for hard matches, cross-cluster pairs for hard non-matches), stratified to the level target. Reported F1 is still computed against the **original human-annotated test gold** per [cross_cutting.md §Gold standard incompleteness and pooling](cross_cutting.md#gold-standard-incompleteness-and-pooling); the regenerated test set is a *secondary* corner-case-ratio-calibrated evaluation surface, not a replacement. Fusion membership frozen throughout.

## Committee expectations

- **Blocking:** candidate counts scale with the level (easy → low recall pressure, hard → high precision pressure).
- **EM:** monotone F1 drop, sharper for similarity-threshold matchers than learned matchers.
- **Fusion:** largely unaffected (interpolated entities come with their own gold by construction; removed entities take their gold rows with them).

## Per-domain notes

Downward-headroom figures below are computed against the raw EM gold / fusion gold and will **shrink** once the pooled-positives floor is in place. Re-measure during algorithm selection after `pooled_positives.csv` exists per domain.

- **Companies:** measured ~medium for matched space (Forbes EM gold ~half hard negatives). **Downward headroom previously estimated as near-unbounded** — fusion gold is only 43 entities out of ~14k records (Step 5 corrected the earlier "tight floor" assumption). **Caveat:** this estimate is against fusion gold only; once pooling surfaces probable-positive entities in the EM-only population, the effective floor will rise. Expect the largest pooled-set correction of any domain because companies has the most unlabeled headroom to retract.
- **Games:** below medium baseline. **Strongest cluster substrate of the corpus** thanks to DBpedia's explicit `franchise` column (1,299 unique franchises, large clusters). **Implementation hint:** use franchise as primary niche signal, but cross-check via EM matches because DBpedia is generally a noisy source. Pooled-set correction expected to be moderate — EM splits are already incomplete ([summary_perstep_brainstorm.md:691](../summary_perstep_brainstorm.md#L691)) so pooling will fill real gaps.
- **Music:** **lowest density in the corpus** — measured EM positive rate 8–10% across both pair-directions. **Strongest natural label-collision substrate** (artist homonyms like "John Williams", "Crash"). Most upward headroom of any domain. Pooled-set correction expected to be the smallest — music already has thousands of labeled pairs per direction, so pooling adds proportionally less.

## Provenance

Entity-scoped (operates on whole entities, not cells):

- **Interpolated entity:** `transform_fn=llm_interpolate_entity`, `transform_params={parent_entity_ids, similarity_metric, placement_mode: "matched_across" | "single_source_distractor"}`.
- **Removed entity:** `transform_fn=remove_entity`, `transform_params={prior_cluster_id, cluster_size_before, selection_metric}`.

## Algorithm selection

**Chosen approach.** Tier B hybrid. The niche-density *scorer*, the removal path, and the corner-case pair miner are all deterministic in-house code built on cited literature primitives. The interpolation path at **hard** is the single Tier C escalation in this knob — LLM-generated near-twin entities are the only realistic way to hit the 80% corner-case target without reusing real entities and leaking ground truth. The removal path and the interpolation path are independently togglable so the ablation matrix stays clean.

Two distinct sub-systems share the same multi-metric substrate but fuse it differently:
- **Niche-density scorer** (drives the removal path): *consensus-biased*. An entity is niche-dense when **multiple metrics agree** it has many close neighbours. Built on Reciprocal Rank Fusion (RRF) over per-metric top-K lists.
- **Corner-case pair miner** (drives EM test regeneration and hard-negative selection): *recall-biased*. A pair is a corner case when **any single metric** flags it, using per-metric raw similarity thresholds. A pair missed by Jaccard but caught by the embedding metric is still a corner case.

The consensus/recall asymmetry is deliberate: for removal we want to avoid deleting an entity the benchmark actually depends on (false-positive density = irreversible data loss), and for pair mining we want to avoid missing a hard pair the downstream matcher will trip on (false-negative hardness = lost signal).

### Metric set

The five metrics in `niche_metric_set`, all deterministic, used by **both** sub-systems:

| Metric | Signal | Implementation |
|---|---|---|
| `lexical_extended_jaccard` | token overlap on the primary label attribute (name / title / company), robust to per-token typos | Generalised Jaccard: two tokens count as matching when their normalised Levenshtein ratio ≥ `inner_token_threshold` (default 0.8, the WDC Products default). Whitespace-tokenised + lowercased, stopwords removed per domain. Replaces plain token Jaccard so that Knob 6-injected typos do not cause Knob 2 to silently lose near-twins. `python-Levenshtein` + stdlib. |
| `lexical_tfidf` | rare-term-weighted textual similarity on the concatenated text block (primary + secondary text attrs) | Cosine over `sklearn.feature_extraction.text.TfidfVectorizer` fit on the full canonical set; sparse matmul for top-K. |
| `embedding_cosine` | semantic similarity, captures synonymy / paraphrase the lexical metrics miss | `sentence-transformers/all-MiniLM-L6-v2` (384-dim, CPU-viable, explicitly recommended by the pretrained-embeddings-for-ER benchmark as the best-recall small model). Embeddings computed once per domain and cached at `usecases_synthetic/cache/knob_02_embeddings/<domain>.npy`. Exact top-K via `sklearn.neighbors.NearestNeighbors` with cosine metric (corpus sizes 5K–20K make an ANN index unnecessary). |
| `attribute_overlap` | structured-attribute agreement (franchise, label, series, category, release year bucket, country) | Weighted Jaccard over the categorical-attribute bag with per-domain column weights authored in YAML. Pure pandas. |
| `label_collision` | shared normalised primary label across otherwise dissimilar entities (inherits from the Knob 7 rescoping — "John Williams", "Crash") | Exact match on `normalize(name)` (lowercase, strip punctuation, collapse whitespace, drop bracketed suffixes) groups entities with byte-identical normalised labels. Because exact match produces a binary partition rather than a ranking, collision neighbours are *not* fed into RRF directly; instead `label_collision` contributes a **fixed density boost** (`+boost_label_collision` to the scorer density of each colliding entity, default 5.0) and contributes to the pair miner via the union rule below. Pure stdlib. |

### Niche-density scorer (removal path)

For every entity `e`, metrics 1–4 each propose their top-K nearest neighbours with raw similarity scores. The four per-metric ranked lists are fused via RRF:

```
rrf_score(e, n) = Σ_{m ∈ {ext_jaccard, tfidf, embed, attr}}  1 / (k₀ + rank_m(n | e))
```

with `k₀ = 60` (the standard RRF default). Niche density of `e` is computed from the fused neighbour list, **not** from its size:

```
density(e) = ( Σ_{n ∈ top_K_rrf(e)} rrf_score(e, n)  ·  𝟙[agreement_count(e, n) ≥ c_min] )
             + boost_label_collision · 𝟙[e is in a label-collision group of size ≥ 2]
```

where `agreement_count(e, n)` is the number of per-metric lists that actually placed `n` in the top-K for `e`, and `c_min` (default 2) enforces the "consensus" rule — a neighbour supported by only a single metric contributes zero to density. Density is therefore (a) monotone in the number of metrics that agree, (b) monotone in each metric's rank of the agreed-upon neighbours, and (c) boosted by label collisions even though those do not participate in RRF.

Entities are sorted by density descending. Protected entities (those appearing in any `expanded_positives` pair, per [cross_cutting.md §Gold standard incompleteness and pooling](cross_cutting.md#gold-standard-incompleteness-and-pooling)) are flagged but not excluded from scoring — they can still act as **parent seeds** for interpolation at hard, they just cannot be *dropped* at easy and cannot be used as parent seeds for the single-source distractor placement mode (which would fabricate a hidden-positive trap).

### Corner-case pair miner (test regeneration + hard-negative selection)

Operates on raw per-metric similarities, not RRF ranks. For each pair `(e₁, e₂)` and each metric `m`, compute the raw similarity `sim_m(e₁, e₂)` in `[0, 1]`. The pair is classified as:

- `hard_match` if `e₁` and `e₂` are in the same ground-truth cluster **and** `∃ m : sim_m(e₁, e₂) < t_match[m]` (low similarity under some metric → the matcher will find it hard),
- `hard_non_match` if they are in different ground-truth clusters **and** `∃ m : sim_m(e₁, e₂) > t_nonmatch[m]` (high similarity under some metric → looks deceptively like a match).

The union-across-metrics rule is the recall-biased contract. Per-metric thresholds default to the WDC Products / profiling_em_benchmarks defaults (`t_match[ext_jaccard] = 0.5`, `t_nonmatch[ext_jaccard] = 0.5`, tightened to `[0.3, 0.7]` on text-heavy domains); thresholds for `tfidf` and `embedding_cosine` are authored in the per-domain config and pinned during the baseline-measurement pass. The `label_collision` metric contributes a hard deterministic rule: any cross-cluster pair with identical normalised labels is automatically `hard_non_match` regardless of other metrics. The `attribute_overlap` metric is omitted from the pair miner (it adds noise at the pair level — two franchise-mates are not automatically corner cases) and used only in the scorer.

**`label_collision` × `expanded_positives` interaction.** Two entities sharing a normalised primary label are *not* automatically split or merged by the protection set. If both endpoints of a label-collision pair are in `expanded_positives` and the pair is in the same ground-truth cluster, the collision is a *true label-coincident match* — protected by the standard expanded-positives rule, scorer-boosted via `boost_label_collision`, and the corner-case pair miner classifies it as `hard_match` (not `hard_non_match`) regardless of the cross-cluster default rule above. If the endpoints are in *different* ground-truth clusters and at least one endpoint is in `expanded_positives`, the protected endpoint is preserved (cannot be removed at easy) but the unprotected endpoint may be removed if its density rank crosses the cap; the surviving endpoint stays in the canonical set with its label-collision boost intact. The cross-cluster `hard_non_match` auto-classification still fires whenever both endpoints survive into the regenerated EM test set. This rule is enforced in the scorer + miner together, not as a post-hoc patch.

Disabling metrics in the YAML degrades gracefully: with only `lexical_extended_jaccard` enabled, the miner reduces to WDC Products' Corner-Case Pair Mining verbatim (modulo the typo-robust inner token comparator), which is a defensible fallback and insulates the knob from "why not vanilla WDC?" reviewer challenges.

**Mapping to easy/medium/hard.** Monotonicity is enforced at the level of (a) protected-set-aware entity removals and (b) the number of interpolated near-twin entities injected, both parameterised by a single `level` key in the per-domain YAML that picks a frozen `(removal_fraction, interpolation_count, placement_split, target_corner_case_ratio)` tuple.

| Level | Target corner-case ratio | Generator action | Interpolation | Placement split default |
|---|---|---|---|---|
| **Easy** | ~20% | **Niche-aware removal.** Sort non-protected entities by density descending; the dispatcher iteratively drops the next entity from the sorted list and re-runs the corner-case pair miner after each drop, stopping the moment the target ratio is hit. `removal_fraction` is therefore an *upper bound* on the fraction removed (a hard cap, frozen per level in the YAML), not a fixed scalar — the actual removal count is whatever satisfies the target first. The frozen tuple `(removal_fraction_cap, interpolation_count, placement_split, target_corner_case_ratio)` is what the YAML loader monotonicity-checks. Removed entities take their gold rows with them (fusion membership is frozen and fusion-gold entities are in `expanded_positives`, so they are protected from removal by construction). | None. | N/A. |
| **Medium** | ~50% | Removal-only. Per-domain baseline decides the magnitude: domains already above target remove more, domains at or below target remove a small fraction (identity-ish) and let the miner's corner-case sampling from the natural distribution do the rest. No new entities are generated at medium. | None. Deterministic attribute-level crossover was considered and rejected — stitched labels ("Grand Theft Halo") are neither plausible nor citeable. Interpolation stays confined to hard. | N/A. |
| **Hard** | ~60–80%, subject to calibration | **Heavy LLM interpolation.** Seed pairs drawn from the top of the density distribution (parents that already sit in dense clusters, so their children deepen existing clusters instead of creating isolated new ones). Interpolation count bounded by `max_interp_fraction` in the per-domain YAML (default 60% of canonical set size). The 80% upper target extrapolates WDC Products' published 40%+ hard variant; reaching 80% on every domain is not guaranteed and the per-domain YAML records the *achieved* ratio post-calibration. | **Tier C LLM.** See LLM hygiene block below. | 60/40 matched-across vs single-source — the 40% single-source distractors are the primary driver of blocking candidate-set size and EM hard-negative pressure per WDC Products' finding that hard non-matches dominate the difficulty signal. Protected entities are excluded from the single-source path. |

Independent togglability: the removal path, the interpolation path, and each metric in `niche_metric_set` have individual YAML toggles so the ablation matrix can disable any one of them. Disabling interpolation at hard degrades hard to "removal-only at the highest achievable target ratio", which is a soft degradation the ablation matrix can reason about. Disabling the embedding metric reduces the scorer + miner to lexical-only operation, which is itself a defensible WDC-aligned fallback.

**Literature citations.**
- **WDC Products Benchmark** ([../literature-search-generation/wdc_products_benchmark/paper.md](../literature-search-generation/wdc_products_benchmark/paper.md)) — anchor paper. Methods used: *Corner-Case Pair Mining* (Jaccard-on-title + optional brand/description; our RRF union is the multi-metric generalisation), *Multi-Dimensional Variant Construction* (corner_case_ratio as the controllable dimension, 20% / 50% / 80% level targets extrapolated from the paper's 10% / 25% / 40%+ range up to the value published as the "hard" variant's actual corner-case ratio). Cited for corner-case methodology and thresholds.
- **Profiling EM Benchmarks** ([../literature-search-generation/profiling_em_benchmarks/paper.md](../literature-search-generation/profiling_em_benchmarks/paper.md)) — cited for corner-case-ratio as the single strongest difficulty predictor (`r = -0.85` with matcher F1), which justifies using it as Knob 2's primary target state.
- **EMBench** ([../literature-search-generation/embench/paper.md](../literature-search-generation/embench/paper.md)) — cited for the provenance-based ground-truth derivation pattern used by the interpolation path (ground truth known by construction because we control parentage).
- **Pretrained Embeddings for ER** ([../literature-search-generation/pretrained_embeddings_er/paper.md](../literature-search-generation/pretrained_embeddings_er/paper.md)) — cited for the choice of `all-MiniLM-L6-v2` as the embedding backbone. The paper's *Embedding-Based Entity Blocking* method card enumerates this model as a best-recall / low-cost pick for ER-style nearest-neighbour retrieval. Using a pretrained, frozen encoder is *not* the heavyweight-ML escalation rejected by `plan_algorithmselection.md` §Tiebreaker rule 4 — the encoder is a fixed deterministic function (no training, no fine-tuning, no drift between runs) and the knob output is the deterministic RRF rank.
- **Curated LLM Tabular Augmentation** ([../literature-search-generation/curated_llm_tabular_augmentation/paper.md](../literature-search-generation/curated_llm_tabular_augmentation/paper.md)) — cited for the hard-level interpolation, matching the LLM-hygiene pattern shared with Knob 1.
- **DAPO** ([../literature-search-generation/dapo_large_scale_data_pollution/paper.md](../literature-search-generation/dapo_large_scale_data_pollution/paper.md)) — cited again as the precedent for combining a controllable corner-case-density knob with other pollution knobs in the same benchmark run.

**Determinism & provenance.**
- RNG: a single `numpy.random.default_rng(seed)` per `(domain, variant, knob=2)` tuple; seed written into the variant's `config/difficulty.yaml`. Governs the removal draw, the parent-pair selection for interpolation, and the RRF tie-break order. Embeddings are computed once and cached (byte-stable across runs for a fixed model ID + input), so embedding-based neighbour lists are deterministic.
- Per-domain config file: `usecases_synthetic/config/knob_02_niche_density/<domain>.yaml`. Keys: per-metric weight vector and top-K for `niche_metric_set`, per-metric `t_match` / `t_nonmatch` thresholds, `removal_fraction_cap` / `interpolation_count` / `placement_split` per level, `max_interp_fraction` hard cap, `hard_negative_margin_delta` (δ — the minimum raw-similarity gap between a hard non-match and the nearest true-match similarity in the same cluster; **TBD per domain at calibration**, pinned in the YAML during the baseline-measurement pass and mirrored on Knob 1's `paraphrase_rate_*` calibration so that K1 paraphrase budget × K2 δ stay consistent), domain-specific column weights for `attribute_overlap`, normalisation rules for `label_collision`, and pointers to the expanded-positives CSVs.
- Embedding cache: `usecases_synthetic/cache/knob_02_embeddings/<domain>.npy` + a sidecar `<domain>.meta.json` capturing `{model_id, input_column_concat_order, content_hash}` for cache invalidation. Committed to the repo so downstream runs reproduce bit-identically without pulling the model.
- LLM interpolation cache: `usecases_synthetic/cache/knob_02_interpolations/<domain>/<parent_pair_hash>.json` per interpolated entity, committed to the repo. Schema: `{parents: [id_a, id_b], prompt_version, model_id, temperature, response, normalized_entity_row, contamination_check_hits}`.
- LLM hygiene (Tier C, inheriting the Knob 1 pattern once Knob 1 is filled in): fixed `prompt_version` pinned in `usecases_synthetic/config/knob_02_niche_density/_prompts/interpolate_v1.txt`, `model_id` pinned in the per-domain YAML (default: the same model as Knob 1 for consistency), `temperature=0`, responses cached and committed, committee validation gating acceptance, contamination spot-check against a web-search blocklist + normalised-name lookup against real data before the entity is admitted to the canonical set.
- Provenance written per affected entity to `output/provenance/knob_02_niche_density.csv`. **Authoritative `transform_fn` enum (single source of truth, supersedes the top-of-card §Provenance bullet list):** `{remove_entity, llm_interpolate_entity}`. The never-introduced `tier_b_crossover_entity` transform is dropped — medium is removal-only per the level table. Row shape:
  ```
  (entity_id, transform_fn ∈ {remove_entity, llm_interpolate_entity},
   transform_params, knob=2, level)
  ```
  `transform_params` is JSON-encoded and carries, depending on `transform_fn`: `{prior_cluster_id, cluster_size_before, density_score, agreement_count, selection_metric}` for removals; `{parent_entity_ids, similarity_metric, placement_mode, prompt_version, model_id, cache_path, contamination_check_status}` for LLM interpolations.
- Committee surface: the Blocking / EM / Fusion committees see the post-Knob-2 canonical set exactly as written. No metric leakage into the committee harness. The committee-vs-pool diagnostic signal from [cross_cutting.md §Gold standard incompleteness and pooling](cross_cutting.md#gold-standard-incompleteness-and-pooling) is especially load-bearing for Knob 2 calibration — if committee F1 collapses but pool agreement stays high, the interpolation depth was too aggressive and should be softened.

**Domain-specific adjustments.**
- **Companies:** measured at ~medium baseline. Default mid-level action is a small removal; easy triggers larger removal, hard triggers interpolation. Calibration must re-measure removal headroom against `expanded_positives`, not raw fusion gold — the pooled-positives floor is expected to retract the companies headroom the most because of the large unlabeled EM population. `attribute_overlap` weights: `{industry: 0.4, country: 0.3, founded_year_bucket: 0.2, ceo: 0.1}` (authored per the DBpedia+Forbes+FullContact attribute surface).
- **Games:** strongest natural cluster substrate. `attribute_overlap` uses `franchise` as the dominant signal (weight 0.5) with `platform`, `genre`, `release_year_bucket` splitting the rest. The card calls out DBpedia's franchise column as noisy, so the dispatcher cross-checks franchise matches against `pooled_positives` — franchise-only matches that disagree with the pool are downweighted rather than promoted to cluster members. Interpolation at hard seeds primarily within franchise boundaries (sequels / remasters / editions).
- **Music:** lowest density in the corpus and strongest natural `label_collision` substrate. `label_collision` weight is bumped in the RRF fusion (effectively shifting the round-robin so the label-collision metric contributes earlier in the fused rank). `attribute_overlap`: `{artist: 0.4, album: 0.3, genre: 0.2, year_bucket: 0.1}`. Music has the most upward headroom, so medium and hard rely more on interpolation than removal.
- **Movies (weak single-source pool exists):** `usecases_synthetic/pools/movies/` has a single-system pool (not the union-of-two pattern used by companies/games/music) per [plan.md](../plan.md) Step 4. Easy and medium run normally; the **hard** interpolation path is permitted but emits a `pool_quality=single_system` flag on every interpolated-entity provenance row so reviewers can audit the increased risk of a hidden-positive trap. The dispatcher does **not** auto-degrade hard on movies — the single-system pool is judged sufficient for protection given the LLM-hygiene contamination check.
- **Products (no pool):** `usecases_synthetic/pools/products/` does not exist. The dispatcher hard-fails on the **hard** level (does not silently degrade) and warns on easy/medium that the protected set is the raw EM-gold-splits ∪ fusion-gold union only. When the products pool comes online the only change needed is authoring `usecases_synthetic/config/knob_02_niche_density/products.yaml` and dropping the hard-fail guard — no code change to the scorer or miner.

**Rejected alternatives.**
- **Single-metric niche scoring** (Jaccard-only per vanilla WDC Products). Rejected because the card's `niche_metric_set` sub-parameter explicitly calls out single-metric selection bias, and because the WDC paper itself flags that pairs hard by Jaccard may be easy for a deep model (and vice versa). RRF fusion is the minimum-cost fix.
- **Heavyweight clustering / deep metric learning** (GReaT embeddings, fine-tuned bi-encoders, learned blocking models). Rejected under `plan_algorithmselection.md` §Tiebreaker rule 4 — violates determinism (training), validation cost, and dependency weight simultaneously. The frozen `all-MiniLM-L6-v2` encoder is the ceiling of what we pull in, and even that is used only as a static similarity function.
- **LLM-only interpolation for medium.** Rejected to keep medium fully deterministic for the ablation. LLM use is confined to hard, where the deterministic alternatives demonstrably cannot hit the 80% corner-case ratio without either (a) reusing real entities and leaking pool-protected positives or (b) collapsing to a tiny canonical set after removal.
- **LLM not used for the niche scorer itself.** A pretrained encoder plus deterministic RRF is sufficient; an LLM "does this pair look like a corner case?" loop would sacrifice determinism and provide no measurable quality gain over the embedding metric. **LLM not used because the deterministic alternative is sufficient.**
- **Connected-component clustering over the full similarity graph.** Rejected because it conflates the per-entity density signal (what the knob actually needs) with a global partition, and because transitive closure across metrics would let a single spurious embedding neighbour merge two real clusters. Per-entity top-K neighbour sets are both sufficient and easier to audit.

**Implementation handoff.** Everything Step 6 needs to implement this knob without re-reading surrounding cards:

- **Target script:** `usecases_synthetic/scripts/apply_knob_02_niche_density.py` (new, convention matches `apply_knob_06_noise.py`). Standalone runnable from repo root.
- **Scenario-aware input shape.** S1 and S2 present differently:
  - **S2** passes a pre-built `canonical_entities` DataFrame (one row per entity, pre-source-split). This is the native shape — Knob 2 is first in the S2 pipeline and all downstream knobs consume its output.
  - **S1** has no canonical set a priori. The dispatcher builds one by taking the union of per-source records and collapsing them via `expanded_positives` (each connected component of the pair graph = one canonical entity, one attribute per column chosen by a fixed deterministic rule: first non-null in a per-domain source-priority order). The collapsed set is the input to the scorer; removals and interpolations are then projected back to the per-source tables before returning. The S1 projection step is recorded in provenance so the source-level effect of each entity-level removal is auditable.
- **Function shape (illustrative):**
  ```python
  def apply_knob_02(
      domain: str,
      scenario: Literal["s1", "s2"],
      level: Literal["easy", "medium", "hard"],
      sources: dict[str, pd.DataFrame],               # S1: per-source record tables; S2: {"canonical": df}
      expanded_positives: pd.DataFrame,               # id1, id2 pairs — the protection set
      fusion_gold: pd.DataFrame,                      # for the fusion-frozen membership check
      em_gold_splits: dict[str, pd.DataFrame],        # train/val/test — for test regeneration
      attribute_classes: dict[str, Literal["primary", "secondary", "categorical", "text"]],
      source_priority: list[str],                     # S1 only: deterministic non-null fallback order
      config_path: Path,                              # usecases_synthetic/config/knob_02_niche_density/<domain>.yaml
      embedding_cache_dir: Path,                      # usecases_synthetic/cache/knob_02_embeddings/
      llm_cache_dir: Path,                            # usecases_synthetic/cache/knob_02_interpolations/<domain>/
      output_dir: Path,
      seed: int,
  ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
      """Returns (new_sources, regenerated_em_test_set, provenance_df, niche_score_df)."""
  ```
- **Inputs the script reads:**
  - Canonical entity set (pre-Knob-2, pre-source-splitting).
  - `usecases_synthetic/pools/<domain>/pooled_positives.csv` merged with EM gold splits and fusion gold to build `expanded_positives`.
  - Per-domain config at `usecases_synthetic/config/knob_02_niche_density/<domain>.yaml`.
  - Embedding cache (or computes + writes it on first run).
  - LLM interpolation cache (hard level only; committed to the repo).
- **Outputs the script writes** (under the variant directory):
  - Post-Knob-2 canonical entity set in `input/data/canonical.csv`.
  - Regenerated EM test set with the target corner-case ratio in `input/entitymatching/test_gold_regenerated.csv`.
  - Provenance log at `output/provenance/knob_02_niche_density.csv`.
  - Per-entity niche score audit at `output/provenance/knob_02_niche_scores.csv` (entity_id, per-metric rank, RRF rank, cluster size, protected flag).
- **Pipeline integration:** Knob 2 runs **first** in the canonical S1 order from [README.md](README.md#canonical-knob-application-order). All downstream knobs (especially Knob 4, which treats Knob 2 placements as fixed input per this card's Composition section) consume the post-Knob-2 canonical set.
- **Dependencies:** stdlib + `pandas` + `numpy` + `scikit-learn` (TF-IDF, NearestNeighbors) + `sentence-transformers` (already declared under the `[embedding]` extra in `pyproject.toml` per [CLAUDE.md](../CLAUDE.md) — re-use, do not add a new extra) + `python-Levenshtein` (for `lexical_extended_jaccard`'s inner token comparator — new, small) + `pyyaml`. Hard level additionally needs the LLM client already wired for Knob 1 (re-use, don't duplicate).
- **Authoring tasks before first run:**
  1. Populate `usecases_synthetic/config/knob_02_niche_density/{companies,games,music}.yaml` with per-metric weights, top-K, thresholds, level targets, and `attribute_overlap` column weights using the Domain-specific adjustments block above as the source of truth.
  2. Author `usecases_synthetic/config/knob_02_niche_density/_prompts/interpolate_v1.txt` following the Knob 1 prompt authoring convention (pinned once Knob 1 lands).
  3. Compute and commit the embedding cache for `{companies, games, music}`.
  4. Movies / products YAMLs **not** authored at v1; dispatcher no-ops the hard level and warns.
- **Smoke test:** for each domain with a config, run the script at all three levels and assert (a) every entity appearing in any `expanded_positives` pair is present in the post-Knob-2 output (canonical set for S2, union over sources for S1), (b) the regenerated EM test set's corner-case ratio matches the level target within ±5% (hard may be below target if calibration caps out — in that case, assert the *achieved* ratio recorded in the per-domain YAML), (c) no interpolated entity's normalised name collides with a real entity's normalised name, (d) the embedding cache hash matches the committed sidecar, (e) hard-level LLM calls hit the cache and do not make live API requests on reruns, (f) the fusion gold file on disk is byte-identical before and after the run, (g) **monotonicity** — the density distribution at hard stochastically dominates medium which dominates easy (Kolmogorov–Smirnov one-sided, p > 0.05 acceptable).
