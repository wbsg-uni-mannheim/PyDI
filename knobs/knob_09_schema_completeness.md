# Knob 9 — Schema completeness / distractors

**Status:** LOCKED. **Scenario: S2 only.** S1 inherits the original use case's column set as-is.

## Definition

Controls schema-level holes and schema-level noise: how much of the target schema is actually covered by source columns, how many distractor columns each source carries, how much semantic ambiguity exists among candidate source columns for the same target. Header *naming* is Knob 8's job — Knob 9 decides *which* columns exist.

## Dimensions controlled

- Schema Completeness (SM)
- Semantic Ambiguity (SM)

**S1 consequence:** Schema Completeness and Semantic Ambiguity are not exercised at all in S1 by design. Both are S2-exclusive signals.

## Sub-parameters (all independent)

- `target_coverage_ratio` — fraction of target attributes that have ≥1 source column correctly mapping to them. Complement: unmapped target attributes the SM matcher must correctly leave unmapped.
- `distractor_column_rate` per source — ratio of distractor columns to genuine columns (scales with target schema size). Distractors are plausible-looking source columns with no true target correspondence.
- `type_matched_distractor_share` — fraction of distractors that are **type-matched** (share a type with some target attribute, hard for instance-based SM). The remainder are unrelated.
- `semantic_ambiguity_rate` — fraction of target attributes for which **≥2 source columns are plausible candidates** by name + type + value-range overlap, but only one is the true match. Stresses disambiguation rather than rejection.

## Per-source shape

No baseline to measure (S2 generates from scratch). Sub-parameters are set directly per level. Different sources may carry different `distractor_column_rate` values to keep heterogeneity interesting, but the *level* is uniform across sources.

## Easy / Medium / Hard

| Level | `target_coverage_ratio` | `distractor_column_rate` | `type_matched_distractor_share` | `semantic_ambiguity_rate` | Target state |
|---|---|---|---|---|---|
| **Easy** | ~1.0 | 0–10% | 0% (unrelated only, if any) | 0% | Near-bijection. Every target attribute has a source candidate. ≤1 distractor per source, all easy to reject on type. SM is essentially one-shot alignment. |
| **Medium** | ~0.85 | ~30% | ~30% | ~10% | Most targets covered; small unmapped tail. A few distractors per source, mostly unrelated, a third type-matched. Occasional ambiguous target. SM must handle rejection + light disambiguation. |
| **Hard** | ~0.60 | ~100% | ~70% | ~30% | Substantial unmapped target tail. Sources are ~half distractor columns, mostly type-matched. Several target attributes have 3+ near-interchangeable candidates differing only on subtle distributional cues. SM stresses completeness detection + rejection + disambiguation simultaneously. |

## Generator action per level

- **Unmapped targets:** simply omit the target attribute from every source's generated column set.
- **Unrelated distractors:** synthesize source columns with realistic domain-plausible headers (internal flags, derived metrics, alternate IDs, free-text notes) and value distributions whose **type does not match any target attribute**.
- **Type-matched distractors:** synthesize source columns whose **type and value range overlap a target attribute** but whose semantics are unrelated (e.g. a `legacy_score` 0–100 alongside the genuine `criticScore`, generated as random noise in the same range).
- **Semantic ambiguity:** for a target attribute, generate multiple source columns sharing its type and overlapping its value range, where only one is the true semantic match (e.g. `score_1`, `score_2`, `score_3` all 0–100, only one is `criticScore`).

## Composition

- **Knob 8:** compose **Knob 9 first, then Knob 8**. Knob 9 fixes the column set; Knob 8 renames it. Knob 8 `anonymized` on top of Knob 9 hard distractors produces the worst-case SM scenario.
- **All value knobs (1, 3, 5, 6, 7, 10):** orthogonal. Values populate whatever columns Knob 9 creates.
- **Knob 4:** orthogonal — Knob 4 controls entity × source incidence; Knob 9 controls attribute × source incidence.
- **Pipeline order:** **first** in S2 (prepended to the S1 order). See [README.md](README.md#canonical-knob-application-order).

## Test-set treatment

- **SM mappings:** generated per variant (S2 artifact anyway). Unmapped target attributes appear as "no correspondence" rows in the gold mapping; distractors appear as explicit negatives so the SM scorer can credit correct rejection.
- **Fusion gold:** **defined only over the subset of target attributes that have ≥1 source correspondence in the variant.** Unmapped target attributes are excluded from the fusion gold file entirely — neither nulls nor zeros. Keeps Knob 9's signal localized to SM and avoids double-counting with Knob 3.
- **EM:** unaffected.

## Fusion safety

N/A direct — Knob 9 doesn't mutate values, and the fusion gold is scoped to covered attributes only. No committee collapse risk.

## Committee expectations

- **SM:** **primary target.** Monotone drop expected. Diagnostic spread across matcher types is itself a difficulty signal:
  - *Label-based matchers* collapse fast on distractors (especially with Knob 8 anonymization) and cannot disambiguate semantic ambiguity at all.
  - *Instance-based matchers* degrade gracefully on **unrelated** distractors (type-mismatch betrays them) but collapse on **type-matched** distractors and on semantic ambiguity.
  - *Duplicate-based matchers* are mostly orthogonal.
  - *LLM matchers* degrade slowest because they reason over names + samples + context.
- **Blocking / Normalization / EM / Fusion:** flat. Fusion gold is scoped to covered attributes, so there is no downstream amplification.

## Per-domain notes

N/A — Knob 9 is S2-only. The S2 use case is generated against a target schema; the level directly controls the schema composition.

## Provenance

Column-level (not row-level), like Knob 8. Per generated column:

```
(source, column_name, target_attribute_or_NULL,
 role ∈ {genuine, distractor_unrelated, distractor_type_matched,
         ambiguous_candidate_for:<target>, dropped_target},
 generation_params, knob=9, level)
```

**Role ↔ pool mapping** (one-to-one between provenance role and catalog pool, so a debugger can always recover which YAML entry produced a row):
- `distractor_unrelated` ← `unrelated_pool` entry
- `distractor_type_matched` ← `type_matched_pool` entry
- `ambiguous_candidate_for:<target>` ← `ambiguous_pool` entry whose `shadows_target == <target>`
- `genuine` ← pre-Knob-9 SM mapping (no pool)
- `dropped_target` ← omission selected from `level_assignments[<level>][<source>].drop` (no pool)

The `dropped_target` role records a target attribute omitted from a source's column set: `column_name` carries the target attribute name, `target_attribute_or_NULL` is the same value, and `generation_params` is empty. Added during the algorithm-selection pass so omissions and injections share a single provenance surface (the alternative — implicit signal via absence from the SM mapping — loses debuggability).

Plus the regenerated SM mapping file: explicit `NO_MATCH` rows for distractors, ambiguous siblings, **and** dropped target attributes (per §Test-set treatment above — unmapped targets appear as "no correspondence" rows so the SM scorer can credit correct rejection of absent attributes).

## Algorithm selection

**Chosen approach.** Tier A — fully deterministic, S2-only. The "algorithm" is a hand-authored per-domain **distractor catalog** YAML (one entry per candidate distractor column with a frozen synthesis recipe) plus a thin pandas/Python dispatcher that (a) selects which target attributes are dropped from each source, (b) selects which distractor catalog entries are materialized per source, (c) synthesizes each distractor's values with a seeded generator of the declared family, and (d) regenerates the SM mapping with explicit negatives. No clustering, no similarity search, no LLM. The literature contribution lives in the taxonomy of distractor families and the choice to split unrelated vs. type-matched (see citations); the dispatcher is plumbing. Determinism matches Knob 8: same `(domain, source, level, seed)` produces byte-identical column sets and values.

**Mapping to easy/medium/hard.** A single `level` parameter selects the four sub-parameter targets (`target_coverage_ratio`, `distractor_column_rate`, `type_matched_distractor_share`, `semantic_ambiguity_rate`) from the per-domain YAML's `level_profiles` block. The ranges below come directly from the table in §"Easy / Medium / Hard" above; nothing is re-derived here.

| Level | Generator action |
|---|---|
| **Easy** | Coverage ~1.0 (no target drops). ≤1 unrelated distractor per source. No type-matched distractors. No ambiguous sets. Dispatcher picks distractors deterministically from the `unrelated_pool` section of the catalog using `stable_hash(domain, source, "easy", slot_idx) mod |unrelated_pool|`. |
| **Medium** | Coverage ~0.85: drop ~15% of target attributes per source. `distractor_column_rate ≈ 0.30` (scaled against target schema size); ~30% drawn from `type_matched_pool`, ~70% from `unrelated_pool`. One or two target attributes get an ambiguous sibling synthesized from `ambiguous_pool`. |
| **Hard** | Coverage ~0.60: drop ~40% of target attributes per source. `distractor_column_rate ≈ 1.0`; ~70% drawn from `type_matched_pool`, ~30% from `unrelated_pool`. `semantic_ambiguity_rate` moderate: several target attributes each get 2–3 ambiguous siblings. |

**Selection determinism.** Which specific target attributes get dropped, and which pool entries get chosen, is pure `stable_hash(domain, source, level, "drop"/"distractor"/"ambiguous", index)` — no RNG state. Re-runs are bit-identical. The author pre-computes and commits the resulting selection into the catalog's `level_assignments` block so the dispatcher also has an explicit audit trail; the hash is used only at authoring time to seed the initial assignment.

**Value synthesis per distractor family.** The catalog declares each distractor entry's family and parameters:
- `numeric_uniform`: `{low, high, dtype}` — seeded `numpy` uniform draw.
- `numeric_normal`: `{mean, std, dtype, clip}` — seeded normal draw with optional clipping.
- `categorical`: `{vocab: [str, ...], weights: [float, ...] | null}` — seeded categorical draw.
- `boolean_flag`: `{p_true}` — seeded Bernoulli.
- `freetext_template`: `{templates: [str, ...], slots: {slot_name: [str, ...]}}` — seeded template fill; short domain-plausible strings only (e.g. `"internal note: {tag}"`). Not LLM-generated — a small slot dictionary suffices for distractors, which are noise by construction.
- `id_like`: `{pattern: str}` — seeded regex-ish fill (e.g. `ID-\d{6}`) via Python stdlib, no external dep.

Ambiguous candidates (`ambiguous_pool`) declare the **target attribute they shadow** and reuse the same value family as the genuine column, with an independent seeded draw so distributions overlap but values disagree. For numeric targets this is a uniform/normal draw over the observed target value range (the range is pinned at authoring time from the base synthetic data under construction — Knob 9 runs before value knobs, so it reads the pre-perturbation canonical values produced by the S2 generation step preceding it). For categoricals, the ambiguous sibling reuses ~70% of the target vocab and adds plausible-but-wrong extras.

Independent togglability: the dispatcher takes only `level` plus optional per-source overrides; it writes only columns (adds new ones, omits some genuine ones) and the SM mapping artifact, never touches existing cells of surviving columns, and emits a column-scoped provenance file disjoint from every value knob's row-level provenance. **Knob 9 is not invoked in S1 runs at any level** — S1 inherits the original use case's column set as-is per the card-top status line, so the S1 ablation matrix has no Knob 9 dial. Within S2, Knob 9 composes independently of every other knob.

Composition contract: **Knob 9 runs first among the perturbation knobs in S2**, but *after* the S2 base generator that emits canonical genuine-column values for every entity. This ordering is load-bearing: the `ambiguous_pool` entries with `params_source: "inherit_from_target"` need the genuine target column's value range (numeric bounds or categorical vocab) to draw plausibly overlapping but wrong values. The S2 base generator produces those canonical values; Knob 9 reads them to size ambiguous siblings, then emits the augmented column set; downstream value knobs (1, 5, 6, 7, 10) and Knob 3 perturb or drop whatever Knob 9 produced; Knob 8 renames headers last. Knob 8 reads column headers directly from Knob 9's output — **see cross-knob note below** for the Knob 8 × Knob 9 distractor-renaming gap.

**Literature citations.** Two distinct sub-tasks with different citation support:

*Coverage reduction (dropping target attributes per source) — well-supported:*
- **XBenchMatch** ([../literature-search-generation/xbenchmatch_schema_matching/paper.md](../literature-search-generation/xbenchmatch_schema_matching/paper.md)) — *Schema Heterogeneity Classification and Injection*, Level 2 Structural Conflicts, specifically "Missing elements: one schema has attributes the other lacks" and the element-deletion injection operator (paper §Apply structural heterogeneity). Directly supports our `target_coverage_ratio` axis.
- **Valentine** ([../literature-search-generation/valentine_schema_matching/paper.md](../literature-search-generation/valentine_schema_matching/paper.md)) — *Fabricated Benchmark Generation via Table Transformations*, specifically the `column subset` primitive (drop 20–50% of columns randomly, ground truth maps remaining columns by identity). Our coverage-reduction path is a direct analogue: construction-time knowledge of which columns are dropped lets the SM gold emit explicit `NO_MATCH` rows as pure metadata, not inferred.
- **DAPO** ([../literature-search-generation/dapo_large_scale_data_pollution/paper.md](../literature-search-generation/dapo_large_scale_data_pollution/paper.md)) — *Schema Inhomogeneity Injection* with `attribute_drop_rate` per source (paper §Phase 3). Cited as precedent for combining schema-completeness perturbation with the rest of a multi-knob pollution profile, matching Knob 9's role in the S2 stack.

*Distractor injection (unrelated + type-matched extra columns) and ambiguous siblings — **no direct literature method**:*
- **No literature method; justification:** a ripgrep pass over `xbenchmatch_schema_matching/paper.md`, `valentine_schema_matching/paper.md`, `ibench/paper.md`, and `dapo_large_scale_data_pollution/paper.md` turned up zero matches for `distractor|extra.?attribute|type.?match|adversarial|noise attribute`. XBenchMatch's "Type conflicts" operator transforms the **same** attribute's type, not an injected sibling; Valentine's `column_add` primitive does not exist (only `column_subset`); iBench's primitives are structural (copy/projection/join/union), none adversarial. Distractor injection as used here is novel composition: we are porting the *taxonomy* split from XBenchMatch's naming/structural/semantic levels, plus Valentine's construction-time-ground-truth discipline, but the actual distractor-column synthesis is authored from scratch. This is deliberately small (a few hundred lines of dispatcher + a hand-authored catalog per domain) so the novelty cost is bounded.
- **No LLM method cited.** Distractor synthesis is deliberately non-creative — noise by construction, drawn from bounded value families. Using an LLM to paraphrase distractor headers or fill distractor cells would only introduce non-determinism and contamination risk for zero benchmark benefit, because the SM matcher is not being tested on how *realistic* the distractor text is, only on whether it correctly rejects the column.

**Determinism & provenance.**
- Catalogs live at `usecases_synthetic/config/knob_09_schema_completeness/<domain>.yaml`, one file per domain. Structure:
  - `unrelated_pool`: list of entries `{name, family, params, applies_to_sources: [str, ...] | "any"}`. Names are authored to be plausibly domain-native (e.g. for movies: `legacy_rating`, `import_batch`, `archivist_notes`).
  - `type_matched_pool`: same shape plus `shadows_type: {str, int, float, date, categorical, list}`.
  - `ambiguous_pool`: entries of form `{name, shadows_target: str, family, params_source: "inherit_from_target" | explicit_params}`.
  - `level_profiles`: `{easy|medium|hard: {target_coverage_ratio, distractor_column_rate, type_matched_distractor_share, semantic_ambiguity_rate, drop_blocklist: [str, ...]}}`. `drop_blocklist` names target attributes that must never be dropped even at hard (e.g. a primary-key-like attribute required for downstream knobs).
  - `level_assignments`: `{easy|medium|hard: {source_name: {drop: [target_attr, ...], distractors: [pool_entry_name, ...], ambiguous: [pool_entry_name, ...]}}}`. Pre-computed once at authoring time from the `stable_hash` seeding described above, committed verbatim. The dispatcher reads `level_assignments` directly — the hash is the authoring tool, not a runtime dependency.
- A single seeded `numpy.random.default_rng(stable_hash(domain, source, level, "values"))` drives value synthesis for that source-level combination. One RNG per `(source, level)`, drawn from independently so the seed surface stays small and re-runs stay bit-identical.
- No external cache required at apply time — catalogs are static YAML; synthesized values are written straight into the variant's source files. Column count is tiny (tens of columns) and row count is whatever the S2 base generator produced, so the output is regenerable on demand.
- Provenance is **column-scoped** per the card's §Provenance block, written to `output/provenance/knob_09_schema_completeness.csv` inside the variant directory:
  ```
  (source, column_name, target_attribute_or_NULL,
   role ∈ {genuine, distractor_unrelated, distractor_type_matched, ambiguous_candidate_for:<target>, dropped_target},
   generation_params_json, knob=9, level)
  ```
  `dropped_target` rows record an omitted target attribute: `column_name` carries the target attribute name, `source` records which source omitted it, `role = dropped_target`, `generation_params_json = {}`. `generation_params_json` for synthesized columns records `{family, params, rng_seed}` so a debugger can regenerate any single distractor without re-running the whole dispatcher.
- The regenerated SM mapping artifact at `usecases_synthetic/<domain>/<level>/input/schemamatching/<source>_to_target.csv` (canonical variant-rooted path; the `input/schemamatching/...` form used elsewhere in this card is the same path relative to the variant directory) includes:
  - one row per genuine `(source_col → target_col)` pair (the positives),
  - one row per distractor `(source_col → NO_MATCH)` with a sentinel target value the SM scorer recognizes (explicit negative),
  - one row per ambiguous sibling `(source_col → NO_MATCH)` (also an explicit negative; only the genuine column for that target retains the positive),
  - one row per dropped target attribute `(NO_SOURCE → target_col)` with a sentinel source value, per §Test-set treatment above — the SM scorer credits correct rejection of absent attributes, so dropped targets must appear explicitly rather than via absence.
- Fusion gold scoping: the dispatcher emits a companion file `output/provenance/knob_09_covered_targets.csv` listing the target attributes with ≥1 source correspondence in the variant. Downstream fusion gold is filtered against this file (per the §Test-set treatment block). This is the only cross-knob handoff Knob 9 produces beyond its own provenance.

**Domain-specific adjustments.** Knob 9 is S2-only, so there is no pre-existing baseline to normalize against. Per-domain catalogs are authored from scratch using the target schema of the S2 variant as the source of truth for what counts as "a plausible distractor" in that domain. Rough guidance for authoring:
- **Movies:** unrelated pool draws from administrative metadata (`import_batch`, `archivist_notes`, `legacy_id`); type-matched pool shadows numeric targets like `runtime`, `release_year`, `budget` with plausible-looking analogues (`imported_runtime_min`, `catalog_year`, `pre_tax_budget`). Ambiguous pool focuses on `title` (shadowed by `original_title`, `display_title`) and rating-like numerics.
- **Games:** type-matched distractors shadow `release_year`, `metacritic_score`, `user_rating`. Ambiguous pool naturally clusters around score-like attributes (multiple 0–100 numerics). Unrelated pool draws from dev/publisher metadata (`build_id`, `platform_internal_code`).
- **Music:** type-matched distractors shadow `duration_ms`, `year`, `track_number`. Ambiguous pool clusters around title/artist-like string attributes. Unrelated pool draws from catalog/licensing metadata (`catalog_number`, `license_region`, `recording_country`).
- **Companies, products:** catalogs authored during Step 6 prototyping against whatever S2 target schema that domain settles on. No dispatcher change.

Baselines do not apply — S2 starts from an empty source and Knob 9 constructs the column set. There is no "rename-down" analogue from Knob 8 because every Knob 9 level starts from the same zero state.

**Rejected alternatives.**
- **LLM-generated distractor columns / headers / values.** Rejected: distractors are noise by construction; the SM matcher is being tested on rejection, not on distractor realism. Determinism win is large, benchmark quality cost is nil. **LLM not used because the deterministic alternative is sufficient.**
- **Sampling distractor values from unused real-world datasets.** Rejected: re-introduces contamination risk (the LLM-based SM matchers in the committee may have memorized the donor dataset), violates the data-leakage safeguard in [../plan.md](../plan.md#1-llm-entity-interpolation).
- **Auto-discovering ambiguous siblings via embedding similarity over the target schema.** Rejected: embedding-based selection would force us to defend a particular embedding model, re-introduces a similarity computation that the hand-authored `ambiguous_pool` makes unnecessary, and risks drifting across runs if the embedding model is updated. A small hand-authored pool per domain is stable and citeable.
- **Magneto-style LLM-authored column name populations** ([../literature-search-generation/magneto_schema_matching/paper.md](../literature-search-generation/magneto_schema_matching/paper.md)) — overkill for distractor naming, same argument as Knob 8.
- **Heavyweight ML methods (CTGAN, GReaT, TabDDPM, BART-error, diffusion, fine-tuning).** Rejected by the framework rule in [../plan_algorithmselection.md](../plan_algorithmselection.md#decision-framework-deterministic-in-house-vs-literature-method-vs-llm): Knob 9's target behavior is controlled, monotone, per-column column insertion with construction-time ground truth — orthogonal to what generative tabular models optimize, and incompatible with determinism and column-scoped provenance.

**Implementation handoff.** Everything Step 6 needs to implement this knob without re-reading the surrounding cards:

- **Target script:** `usecases_synthetic/scripts/apply_knob_09_schema_completeness.py` (S2 pipeline only — placeholder to be added to `usecases_synthetic/PIPELINE.md` Phase 3 when that phase is drafted). Standalone runnable from repo root, same convention as `apply_knob_08_naming.py`.
- **Function shape (illustrative, not prescriptive):**
  ```python
  def apply_knob_09(
      domain: str,
      level: Literal["easy", "medium", "hard"],
      sources: dict[str, pd.DataFrame],            # source_name -> base-synthetic dataframe (pre-perturbation)
      target_schema: dict[str, str],               # target_attribute -> declared type
      sm_mapping: dict[str, dict[str, str]],       # source_name -> {source_col -> target_col} (genuine only, pre-Knob-9)
      catalog_path: Path,                          # usecases_synthetic/config/knob_09_schema_completeness/<domain>.yaml
      output_dir: Path,                            # variant directory root
      per_source_override: dict[str, dict] | None = None,
  ) -> tuple[
      dict[str, pd.DataFrame],                     # sources with dropped targets + injected distractors + ambiguous siblings
      dict[str, dict[str, str | None]],            # regenerated SM mapping, incl. explicit NO_MATCH entries
      pd.DataFrame,                                # column-scoped provenance
      list[str],                                   # covered target attributes (for fusion gold scoping)
  ]:
      """Returns (augmented_sources, regenerated_sm_mapping, provenance_df, covered_targets)."""
  ```
- **Inputs the script reads:**
  - The S2 base-synthetic source DataFrames (whatever the preceding S2 generation step emits, `df.attrs["dataset_name"]` preserved).
  - The S2 target schema declaration (type map).
  - The pre-Knob-9 SM mapping (genuine-only) for that variant.
  - The per-domain catalog YAML.
  - The active `level` from the variant's `config/difficulty.yaml`.
- **Outputs the script writes** (under the variant directory `usecases_synthetic/<domain>/<level>/` — path to be finalized alongside `usecases_synthetic/PIPELINE.md` Phase 3):
  - Augmented source files in `input/data/` (same format as the S2 base step — XML/JSON/CSV — with some target columns dropped and distractor/ambiguous columns injected).
  - Regenerated SM mapping CSVs in `input/schemamatching/<source>_to_target.csv` including explicit `NO_MATCH` rows for distractors and ambiguous siblings.
  - Provenance log at `output/provenance/knob_09_schema_completeness.csv` with the column schema given above.
  - Covered-targets file at `output/provenance/knob_09_covered_targets.csv` (single column `target_attribute`).
- **Pipeline integration:** Knob 9 runs **first** among S2 perturbation knobs, after the S2 base generator (which produces canonical genuine-column values so ambiguous siblings can inherit target ranges). It writes only column-scoped artifacts, so it composes orthogonally with every downstream value knob; those knobs read whatever column set Knob 9 produced and populate cells without reading Knob 9's provenance. Knob 8 runs last and renames headers Knob 9 created, via the composition rule in `knob_08_schema_naming.md` §Composition.
- **Cross-knob note — Knob 8 × Knob 9 distractor renaming.** Knob 8's per-domain renaming table keys on *original target attribute columns*. Distractor column names authored by Knob 9 are **not** in Knob 8's `rename_table`, so Knob 8's lookup misses for every Knob 9 distractor. Today this means Knob 8 would passthrough distractor headers unchanged, which leaks label signal at *hard* (anonymized genuine columns next to plain-English distractor columns → the matcher can trivially identify distractors by their un-anonymized names, defeating the knob). **Resolution:** Knob 9's catalog must author each distractor entry with a parallel `descriptive / abbreviated / cryptic / anonymized` rung set (mirroring Knob 8's `rename_table` shape), and Knob 9's dispatcher merges those entries into the Knob 8 rename table before Knob 8 runs. The `unrelated_pool`, `type_matched_pool`, and `ambiguous_pool` YAML schemas must therefore carry a `rename_rungs: {descriptive, abbreviated, cryptic, anonymized}` field per entry. Flagged as a cross-cutting follow-up on the Step 5 tracker.
- **Cross-knob note — Knob 9 × EM blocking.** A distractor with `family: id_like` could be mistaken by a downstream EM blocker for a genuine identifier column. Knob 9 does not forbid `id_like` distractors (they are legitimate corner cases for SM), but the per-domain catalog author should avoid placing them on attributes any EM blocker is configured to key on. Surfaces as a lint during Step 6 prototyping, not a dispatcher concern.
- **Dependencies:** stdlib + `pandas` + `numpy` + `pyyaml`. No PyDI extension points (Knob 9 prepares input *for* schema matching, it doesn't *do* schema matching). No LLM API dependency.
- **Authoring task before first run:** populate `usecases_synthetic/config/knob_09_schema_completeness/<domain>.yaml` with `unrelated_pool`, `type_matched_pool`, `ambiguous_pool`, `level_profiles`, and the pre-computed `level_assignments`. Per-domain pool sizing guidance: enough entries to support ~2× the `hard` distractor rate for the largest source, so the dispatcher always has choice. If a domain has no catalog, the dispatcher raises — unlike Knob 8's identity fallback, silent no-ops at Knob 9 would break S2's SM stage (no distractor signal at all) and must fail loud.
- **XML sources:** same XML re-serialization concern as Knob 8. The dispatcher writes the augmented DataFrame back to the source format, re-using `PyDI.io` loaders/writers. For XML, injected distractor columns become new child elements under each record; dropped targets are simply omitted. Element-tag-name correctness is asserted in the smoke test.
- **Smoke test:** for each domain with a catalog, run the script at all three levels and assert:
  - (a) `covered_targets ∪ dropped_targets == target_schema.keys()` (completeness);
  - (b) for each source and level, `|covered_target_cols| / |target_schema| ≈ target_coverage_ratio ± 0.05`;
  - (c) for each source, `|distractor_cols| / |genuine_cols| ≈ distractor_column_rate ± 0.05`, and `|type_matched_distractors| / |distractor_cols| ≈ type_matched_distractor_share ± 0.05`;
  - (d) the regenerated SM mapping has an explicit `NO_MATCH` row for every distractor and every ambiguous sibling, **and** an explicit `(NO_SOURCE → target_col)` row for every dropped target attribute (no implicit-absence signalling);
  - (e) no distractor column name collides with any genuine target attribute name in that source;
  - (f) re-running the dispatcher with the same inputs produces byte-identical outputs (determinism);
  - (g) `drop_blocklist` attributes are never in the dropped set at any level;
  - (h) for XML sources, on-disk element tag names match the returned DataFrame column names.
