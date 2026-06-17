# Knob 1 — Surface augmentation intensity

**Status:** LOCKED. **Scenario:** S1 + S2 (fully controllable, bidirectional).

## Definition

Plausible free-text rewriting of values across sources (paraphrase, abbreviate, reorder). Result is still semantically correct, just expressed differently. Boundary vs Knob 6: paraphrase = legitimate variant; noise = error. No overlap by construction.

## Dimensions controlled

- Representation Heterogeneity (SM, Block, EM)
- Corner-Case Difficulty (EM)
- Conflict Subtlety (Fusion)

## Sub-parameters

Single intensity knob. Per-attribute behavior is handled in code:
- `paraphrase_short` — short fields (titles, names, cities, countries)
- `paraphrase_long` — long-text fields. **Currently dead in the corpus** (none of companies / games / music has a long-text target attribute). Kept in spec for future domains (movies plots, products marketing copy). Implicitly v2-or-future-domain priority.
- `paraphrase_categorical` — categorical labels (industry, ESRB, genre)
- `normalize_to_canonical` — used at easy when baseline is already above target

## Easy / Medium / Hard

| Level | Target state | Generator action |
|---|---|---|
| **Easy** | Sources largely agree on surface form. Short fields effectively verbatim across sources; long fields share most content with only incidental wording differences. | Baseline at target → passthrough. Baseline above (e.g. companies country) → normalize toward a canonical surface form per entity (propagate one source's value, or light rewrite). |
| **Medium** | Recognizable per-source paraphrases with substantial token overlap. Short fields show abbreviations / reordering / dropped subtitles on a meaningful fraction. Long fields show moderate rewording with preserved key terms. | Add or soften toward target. |
| **Hard** | Aggressive per-source paraphrase. Short fields use alternative naming forms (marketing names, aliases, alternate romanizations within the same script); long fields heavily reworded (different sentence structure, synonyms, changed verbosity); categorical labels paraphrased. **No cross-language translation.** | Heavy per-source paraphrase; typically additive (most domains sit well below hard). |

## Composition

- **Knob 6 (noise):** strict separation by intent (variant vs error). See [knob_06_value_noise.md](knob_06_value_noise.md).
- **Knob 5 (formats):** orthogonal — Knob 1 touches free-text fields, Knob 5 touches structured/parseable fields. Categoricals belong to Knob 1.
- **Pipeline order:** runs jointly with Knobs 5/6/7 in the value-perturbation phase, before Knob 3 (drop). See [README.md](README.md#canonical-knob-application-order).

## Fusion safety

Replace-or-extend gold on collapse (per [cross_cutting.md](cross_cutting.md#per-knob-fix-strategy-defaults)).

## Committee expectations

- **SM:** largely unaffected (values, not headers).
- **EM:** monotone drop, sharper for lexical blockers than embedding matchers.
- **Fusion:** needs gold extend-to-accepted-set on hard for long string attributes (when those exist).

## Per-domain notes

- **Companies:** medium baseline (per-source country forms differ — Forbes long official vs DBpedia/FullContact short). No long-text attribute → `paraphrase_long` inactive. Easy on country requires *normalize-down*.
- **Games:** easy→medium baseline. ESRB / system / franchise vocabularies already shared at baseline → easy near-passthrough; medium and hard need active per-source paraphrasing.
- **Music:** easy→medium for short fields (titles, artists, labels).

## Provenance

`transform_fn ∈ {paraphrase_short, paraphrase_long, paraphrase_categorical, normalize_to_canonical}` with `transform_params={style, intensity, model?, prompt_id?}`. Every committee-driven gold extend emits a mirror record on the gold artifact side with `transform_fn=gold_extend_for_committee`.

## Algorithm selection

**Chosen approach.** Tier C — hybrid by level. **Easy** is deterministic Tier A: when a source's baseline sits above the easy target (e.g. Companies per-source country forms), the dispatcher normalizes the offending cell to a canonical per-entity form drawn from a sibling source; otherwise it passes through. **Medium** is deterministic Tier B: a curated per-domain abbreviation / synonym / article-drop dispatcher plus EDA-style token-level operators (random swap within non-stopword positions, random deletion on non-key tokens) realised in pure pandas + regex. No LLM at medium. **Hard** is the single Tier C escalation: an LLM paraphrase pass over short-field and categorical values, applied **per (entity, source, attribute)** with cached outputs, committee validation gating acceptance, and mandatory contamination spot-checks. This knob locks the LLM-hygiene pattern shared with Knob 2 hard interpolation and Knob 4 easy fabrication — subsequent LLM-touched knobs inherit it verbatim.

The two Tier B paths and the Tier C path are independently togglable per the ablation matrix: disabling the LLM path at hard soft-degrades hard to "medium + extended operator pool", disabling the table dispatcher at medium degrades medium to "pure EDA-style operators", and disabling EDA at medium degrades medium to "pure table dispatcher". None of these degradations is a hard failure — each is a defensible, named fallback.

Per-attribute routing is decided once at dispatch time from the attribute class declared in the per-domain YAML:

| Sub-parameter | Attribute class | Easy | Medium | Hard |
|---|---|---|---|---|
| `paraphrase_short` | primary label / short identifier (title, name, city, country) | Passthrough unless baseline-above-target → *normalize-down* (canonical-form selection from sibling source, deterministic) | Table-driven abbreviation / expansion / article-drop **plus** EDA *random swap* and *random deletion* on non-stopword non-key tokens (token-budget capped; stopword + key-token skiplist per domain) | LLM paraphrase prompt `prompt_short_v1.txt` — alternative naming forms, marketing names, aliases, romanization variants (within the same script). Gated by the **Fusion committee** per [cross_cutting.md §Committee composition](cross_cutting.md#committee-composition-fusion-draft): a paraphrase is accepted iff ≥2 of 3 committee members (`longest_string`, `most_frequent`, `shortest_string`) resolve the post-paraphrase `(entity, attribute)` cell to a value that is either the original gold or a member of the accepted-set extension. Cached, contamination-checked. |
| `paraphrase_long` | long descriptive text | **Dead in v1** — none of companies / games / music has a long-text target attribute per the Per-domain notes section. Dispatcher no-ops and logs a skip row. Prompt `prompt_long_v1.txt` intentionally **not authored** at v1. | Same. | Same. Revisit when movies / products land. |
| `paraphrase_categorical` | categorical label (industry, ESRB rating, genre, platform) | Passthrough unless baseline-above-target → *normalize-down* to a canonical vocabulary form (e.g. ESRB `E10+` ↔ `Everyone 10+`) | Table-driven alias / synonym rewrite (per-domain curated vocab map) | LLM paraphrase prompt `prompt_categorical_v1.txt` for genuinely-open categorical values (rare at this granularity) **plus** table-driven rewrite. |
| `normalize_to_canonical` | any | Baseline-above-target only | n/a | n/a |

The `easy → medium → hard` sequence is monotone in the per-cell draw probability **and** in the operator pool. Medium adds the EDA + table operators to the easy set; hard adds the LLM operator to the medium set. Stopwords, key-tokens, and canonical-vocabulary positions are never touched by *any* level so that paraphrase stays a surface rewrite and never a value-change — that is Knob 7's territory (deferred) and Knob 6's wrong-value territory, strictly separated by construction.

**Mapping to easy/medium/hard.** A single `level` key in the per-domain YAML selects a frozen tuple `(paraphrase_rate_primary, paraphrase_rate_key, paraphrase_rate_secondary, operator_mix, per_source_shape)`. Monotonicity is enforced by (a) strictly non-decreasing per-class paraphrase rates, (b) strictly non-shrinking operator sets, (c) the canonical form selection at easy being idempotent on any medium/hard input, (d) `paraphrase_long` passthrough at every level in v1, and (e) the YAML loader refusing to run if any of these are violated.

| Level | Primary rate | Key rate | Secondary rate | Operator set | Per-source shape |
|---|---|---|---|---|---|
| **Easy** | 0 (+ normalize-down hits) | 0 (+ normalize-down hits) | 0 (+ normalize-down hits) | `{normalize_to_canonical}` — reversal-only. `normalize_to_canonical` is exclusive to easy and only fires where the authored baseline-above-target rule names the source × attribute. | Quietest baseline source acts as the canonical reference. |
| **Medium** | ~1–3% | ~3–5% | ~5–10% | Easy set ∪ `{paraphrase_short (table-driven abbreviation / expansion / article-drop), paraphrase_categorical (table-driven alias / synonym rewrite), eda_random_swap (within non-stopword positions, 1–2 tokens), eda_random_delete (non-key tokens only, ≤1 token)}`. Max 1 operator per cell. | Identity — baseline per-source surface shape preserved. |
| **Hard** | ~5–10% | ~10–15% | ~15–25% | Full Medium set ∪ `{llm_paraphrase_short, llm_paraphrase_categorical}`. Long-text LLM operator **not authored** in v1. Up to `max_operators_per_cell = 2` with a mandatory guard that no cell is touched by both the medium table-dispatcher and the LLM paraphrase in the same run. The only permitted pairings are `{eda_random_swap + eda_random_delete}` (both EDA, independent token positions) and `{table_dispatcher + eda_random_swap}` or `{table_dispatcher + eda_random_delete}` (one table rewrite followed by one EDA perturbation on the post-table token stream). `{llm_paraphrase + anything}` is forbidden — LLM paraphrase always runs alone on its cell. | Stretch — per-attribute paraphrase rates on the already-more-rewritten sources scaled up; the quietest source left near-identity to preserve an anchor for the per-entity anchor-survivor floor (see below). |

**Knob 1 ↔ Knob 6 cell-collision contract.** Knob 6 already assumes Knob 1 honours a symmetric "don't double-perturb this cell" check ([knob_06_value_noise.md:98](knob_06_value_noise.md) and the Step 5 cross-knob tracker item). The contract is enforced here: **before** Knob 1 mutates a cell, the dispatcher reads the joint provenance index under `output/provenance/knob_0{1,4,5,6,7}_*.csv` (whichever of 4/5/6/7 have already run in the canonical S1 order — Knob 4 always runs upstream of the joint phase, so its fabrication rows are always present) and skips the cell if any earlier row exists for the same `(entity_id, source, attribute)`. Knob-4-fabricated cells (carrying `k4_fabricated=True` in `transform_params`) are unconditionally skipped — see [knob_04_coverage_skew.md](knob_04_coverage_skew.md#joint-cell-collision-index-integration) for the symmetric C2 contract: K4-fabricated cells already went through an LLM/paraphrase pass during fabrication, so re-paraphrasing here would be double-augmentation. The dispatch order inside the joint phase is fixed at `Knob 1 → Knob 5 → Knob 6 → Knob 7`, so in practice Knob 1 writes first and only reads its own prior index at retry time; the collision check guards the general case where ordering is overridden by an ablation. Skipped cells are logged to `output/provenance/knob_01_skipped.csv` with `reason=cell_collision_with_{1,5,6,7}`.

**Anchor-survivor floor (locked for Knob 1).** Mirrors the Knob 6 per-entity clean-primary floor: for every fusion-gold entity, **at least one source must retain a non-paraphrased primary value** at every level including hard. Hard constraint — guarantees blocking always has a clean anchor and that the fusion committee's `longest_string` / `most_frequent` strategies have a non-paraphrased candidate to vote for. Per-cell clean-survivor floor is the softer variant: for every fusion-gold `(entity, attribute)` cell, ≥1 source retains a non-paraphrased value unless the per-domain YAML explicitly waives it.

**Fix-on-collapse.** Per the locked Fusion-safety section of this card and the [cross_cutting.md §Per-knob fix-strategy defaults](cross_cutting.md#per-knob-fix-strategy-defaults) table, Knob 1's fix on committee collapse is **replace-or-extend gold to accepted set**. Concretely: when the fusion committee monotonicity check collapses on a paraphrased primary/key value whose original gold is no longer recoverable from the surviving sources, the gold artifact for that `(entity, attribute)` cell is extended with the accepted paraphrase as an additional member of an accepted-set column (gold row is not replaced — the original value stays). This is the only knob authorised to mutate the gold artifact; the mutation emits a mirror provenance row `transform_fn=gold_extend_for_committee` on the gold-side provenance file per the locked Provenance section above. If extending the accepted set still does not restore monotonicity (rare), fall back to **soften paraphrase locally** for the offending entities and re-run. Trigger condition for `soften_for_committee`: a second committee-collapse detection on the same `(entity, attribute)` cell after one gold-extend pass, OR any collapse where the committee failed to converge on *any* value (no majority) — in both cases the offending entities have their hard-level LLM paraphrase downgraded to medium table-dispatcher output for that attribute only, with a provenance row recording the downgrade. Never rolls back at hard — rollback is Knob 6's fix mode, not Knob 1's, because paraphrase is by construction a legitimate variant and the gold contract is allowed to grow to accommodate it.

**Literature citations.**

- **LEMONADE** ([../literature-search-generation/lemonade_llm_guided_em/paper.md](../literature-search-generation/lemonade_llm_guided_em/paper.md)) — anchor paper for the hard LLM path. Methods used: *Soft Entity Modification* (the locked "medium" target state maps cleanly onto LEMONADE's abbreviation / minor spelling variation / token reordering / partial omission catalogue, which we implement deterministically at medium) and *Strong Entity Modification* (the hard LLM paraphrase path, restricted to `paraphrase_short` + `paraphrase_categorical` because `paraphrase_long` is dead in v1). LEMONADE's published prompts for companies / music / movies / games are adapted verbatim into `prompt_short_v1.txt` and `prompt_categorical_v1.txt` as the starting point, then pinned under `prompt_version=v1` after spot-checks against the domain's gold.
- **EDA** ([../literature-search-generation/eda_easy_data_augmentation/paper.md](../literature-search-generation/eda_easy_data_augmentation/paper.md)) — medium-level rule-based operators. Methods used: *Random Swap* and *Random Deletion* (the two operators that are safe for entity-matching surface rewrite — *Synonym Replacement* and *Random Insertion* are rejected because WordNet synonyms do not exist for brand names / proper nouns and random insertion changes entity identity too often; EDA's own limitations section calls these out). EDA provides citable, zero-cost medium operators and the `alpha = n/sentence_length` rate convention that we adopt for `paraphrase_rate_*`. Our implementation mirrors the [`nlpaug`](https://github.com/makcedward/nlpaug) parameterisation referenced in the EDA paper's implementation notes but is re-implemented in pandas + stdlib to avoid a runtime dependency.
- **Curated LLM Tabular Augmentation** ([../literature-search-generation/curated_llm_tabular_augmentation/paper.md](../literature-search-generation/curated_llm_tabular_augmentation/paper.md)) — cited for the hard-level LLM prompt shape (*Serialized Prompting*) and the committee-validated acceptance curation layer. Shared with Knob 2 interpolation and Knob 4 easy fabrication.
- **Benchmark Contamination Survey** ([../literature-search-generation/benchmark_contamination_survey/paper.md](../literature-search-generation/benchmark_contamination_survey/paper.md)) and **Elephants Never Forget** ([../literature-search-generation/elephants_never_forget_tabular_memorization/paper.md](../literature-search-generation/elephants_never_forget_tabular_memorization/paper.md)) — cited for the contamination spot-check protocol applied to every LLM-paraphrased cell at hard (n-gram overlap against the source gold + first-token memorization probe, both documented in the Determinism & provenance block below).
- **DAPO** ([../literature-search-generation/dapo_large_scale_data_pollution/paper.md](../literature-search-generation/dapo_large_scale_data_pollution/paper.md)) — precedent for combining a parameterised paraphrase profile with other pollution knobs in a single controlled benchmark run. Cited as "this composition pattern is not novel" alongside Knobs 3 / 4 / 6 / 10.

**Determinism & provenance.**

- **RNG.** Single `numpy.random.default_rng(seed)` per `(domain, variant, knob=1)` tuple; seed recorded in the variant's `config/difficulty.yaml`. Governs the per-cell "paraphrase or not" draw, the operator draw (swap / delete / table / LLM), the token-position draws for EDA operators, and the canonical-source selection at easy. The LLM call itself is not RNG-dependent (see LLM hygiene below).
- **LLM hygiene (hard path only).** Fixed `prompt_version=v1` pinned in the per-domain YAML and under `usecases_synthetic/config/knob_01_surface_augmentation/_prompts/{prompt_short_v1.txt, prompt_categorical_v1.txt}`; `model_id` pinned in the per-domain YAML; `temperature=0` (or `0.0` + fixed seed where the provider supports it); outputs cached and **committed to the repo** at `usecases_synthetic/cache/knob_01_paraphrases/<domain>/<variant>/<cell_hash>.json` where `cell_hash = sha256(source|attribute|original_value|prompt_version|model_id)`; committee validation gating acceptance; contamination spot-check per the two-step protocol below. The LLM call is part of generation, not the runtime pipeline being benchmarked. Either an OpenAI-compatible API call (loaded via `python-dotenv` per [CLAUDE.md](../CLAUDE.md#testing-notes)) or direct generation by the assistant during a Step 6/7 generation run is acceptable, provided outputs land in the cache and the cache is committed. **The committed cache is the sole source of truth on rerun:** the dispatcher first checks `<cell_hash>.json`; if present, it loads and uses the cached paraphrase unconditionally — no API call, no assistant-direct generation, no re-prompting. Spot-check reruns MUST NOT regenerate cached cells. A cache miss on a variant whose `config/difficulty.yaml` is unchanged is a hard error, not a silent regeneration trigger.
- **Contamination spot-check (mandatory at hard).** Two-step, applied to every LLM paraphrase response before the cell is admitted:
  1. **N-gram overlap** (per [benchmark_contamination_survey](../literature-search-generation/benchmark_contamination_survey/paper.md)): the paraphrase must not contain an 8-or-more-token contiguous overlap with the original gold value (threshold lifted from the survey — 8-gram overlap is the canonical contamination signal). Failure routes to the medium fallback operator for that cell.
  2. **First-token memorization probe** (per [elephants_never_forget_tabular_memorization](../literature-search-generation/elephants_never_forget_tabular_memorization/paper.md)): the paraphrase must not match the first few tokens of any *other* real entity in the same domain (cheap lookup against a pre-computed normalised first-token index). Catches the case where the LLM silently aliases one entity onto another. Failure routes to the medium fallback operator.
- **Per-domain config file:** `usecases_synthetic/config/knob_01_surface_augmentation/<domain>.yaml`. Keys:
  - `attribute_classes`: `{source_name: {column: "primary" | "key" | "secondary" | "categorical"}}`. Drives sub-parameter routing.
  - `stopword_list`: per-domain list of tokens the EDA operators must not touch (e.g. "the", "of", plus domain-specific words like "Inc.", "Ltd.").
  - `key_token_skiplist`: per-column list of tokens the EDA operators must not touch (e.g. for music `album` column, the artist name is a key token).
  - `baseline_above_target_rules`: per-(source, attribute) list of canonicalisation rules for the easy normalize-down path. Authored per the Per-domain notes section (e.g. Companies DBpedia `country` → canonical short form).
  - `abbreviation_table`, `synonym_table`, `article_drop_pattern`: per-domain curated tables used by the medium table-driven operator. `abbreviation_table` is bidirectional (expand *and* contract — `"International Business Machines" ↔ "IBM"`).
  - `categorical_vocab_map`: per-domain categorical alias map (e.g. Games `{"ESRB_E10+": ["Everyone 10+", "E10+"]}`).
  - `paraphrase_rate_primary/key/secondary`: per-level rate triples. Authored from the level table above with domain adjustments (see Domain-specific adjustments).
  - `operator_mix`: per-level dict `{operator_name: relative_weight}`.
  - `llm_prompt_version`: string (pinned, default `v1`).
  - `llm_model_id`: string.
  - `llm_temperature`: float (default 0.0).
  - `anchor_survivor_floor`: bool (default true for primary, configurable for key/secondary).
- **Shared tables.** Stopword defaults, EDA position-sampling parameters, and the 8-gram contamination threshold live as shared static YAML at `usecases_synthetic/config/knob_01_surface_augmentation/_tables/{stopwords_en.yaml, eda_params.yaml, contamination.yaml}`.
- **Provenance written per paraphrased cell** to `output/provenance/knob_01_surface_augmentation.csv` inside the variant directory, following the [cross_cutting.md §Per-value provenance](cross_cutting.md#per-value-provenance-mandatory) flat-row schema:
  ```
  (entity_id, source, attribute, original_value, new_value,
   transform_fn ∈ {paraphrase_short, paraphrase_long, paraphrase_categorical,
                   normalize_to_canonical, eda_random_swap, eda_random_delete,
                   llm_paraphrase_short, llm_paraphrase_categorical,
                   gold_extend_for_committee, soften_for_committee},
   transform_params, knob=1, level)
  ```
  `transform_params` is a JSON-encoded string in the CSV column. Keys by `transform_fn`:
  - `paraphrase_short` / `paraphrase_categorical` (table dispatcher): `{table: "abbreviation" | "synonym" | "article_drop" | "categorical_vocab", table_key, direction}`.
  - `eda_random_swap`: `{positions: [i, j], tokens_before: [...], tokens_after: [...]}`.
  - `eda_random_delete`: `{position: i, token_removed}`.
  - `llm_paraphrase_short` / `llm_paraphrase_categorical`: `{prompt_version, model_id, cache_path, committee_passed: bool, contamination_check: {ngram_overlap_passed: bool, first_token_probe_passed: bool}}`.
  - `normalize_to_canonical`: `{template_source, canonical_form_origin: "sibling_source" | "config_rule"}`.
  - `gold_extend_for_committee`: `{committee_collapse_attribute, accepted_set_new_member}`. Emitted on the gold-side provenance file at `output/provenance/knob_01_gold_extend.csv` per the locked Provenance section.
  - `soften_for_committee`: `{original_transform_fn, entity_ids_softened}`.
- **Gold-extend rows are NOT written to the joint K1/5/6/7 cell-collision index.** They land exclusively on the gold-side provenance file (`output/provenance/knob_01_gold_extend.csv`) so downstream knobs in the joint phase do not treat the gold mutation as a "cell already touched" signal — the gold artifact is a separate namespace from per-source surface cells. Knobs 5/6/7 never read the gold-extend log.
- **Gold extend mirror file.** When the fix-on-collapse path extends the accepted set, the mutation is written to the variant's gold artifact *and* mirrored to `output/provenance/knob_01_gold_extend.csv` so readers can reconstruct the pre-extend gold by streaming the provenance log. The gold artifact's baseline version is checksummed at run start and the checksum is recorded; any mutation increments a `gold_revision` counter captured in `output/baselines/knob_01_gold_revision.json`.
- **Skipped-cell audit** at `output/provenance/knob_01_skipped.csv`. Reasons: `cell_collision_with_{1,5,6,7}`, `stopword_only`, `key_token_only`, `llm_contamination_fail`, `llm_committee_fail`, `anchor_survivor_floor`, `per_cell_clean_survivor_floor`.
- **Committee surface.** The SM / Blocking / EM / Fusion committees (per the [cross_cutting.md §Committee composition](cross_cutting.md#committee-composition-fusion-draft) draft) see the paraphrased source files exactly as written. No operator leakage into the committee harness. The committee-vs-pool diagnostic signal ([cross_cutting.md §Protection set semantics](cross_cutting.md#protection-set-semantics-not-replacement-gold) point 3) is load-bearing for Knob 1 hard calibration: if committee F1 collapses but pool agreement stays high, the LLM paraphrase depth was too aggressive on the un-pooled headroom and should soften locally.

**Domain-specific adjustments.**

- **Companies.** Baseline sits at medium for short fields per the Per-domain notes section. Forbes long-official country forms vs. DBpedia / FullContact short forms — easy must **normalize-down** for `country`: the canonical per-entity form is the shortest sibling-source value (config rule pinned in `baseline_above_target_rules`). `paraphrase_long` is **dead** (no long-text target attribute). Key `abbreviation_table` entries: `{"Incorporated": "Inc.", "Corporation": "Corp.", "Limited": "Ltd.", "International Business Machines": "IBM", ...}` — authored by hand during Step 6 bootstrap. Hard LLM paraphrase is guarded by a tight contamination check because the companies corpus has the most web-memorable entities in the three-domain set.
- **Games.** Baseline easy → medium for short fields. ESRB / system / franchise vocabularies already shared at baseline → easy is near-passthrough (no normalize-down rules). Medium and hard need active per-source paraphrasing. `abbreviation_table` heavy on subtitle-drop patterns (e.g. `"The Legend of Zelda: Ocarina of Time" → "Ocarina of Time"`) and platform aliases (`"PlayStation 4" ↔ "PS4"`). Categorical `vocab_map` for ESRB (`E10+ ↔ "Everyone 10+"`) and genre (`"Third-Person Shooter" ↔ "TPS"`). Hard LLM paraphrase is the safest of the three domains because game titles are less web-memorable than companies.
- **Music.** Baseline easy → medium for short fields (titles, artists, labels). No `paraphrase_long`. `abbreviation_table` focuses on article-drop (`"The Dark Side of the Moon" ↔ "Dark Side of the Moon"`) and common artist-name variants. Hard LLM paraphrase uses the tightest contamination check of any domain — music titles are highly web-memorable and the LLM is most at risk of silently aliasing one entity onto another. `llm_temperature=0` is not negotiable on Music.
- **Movies, products.** Deferred alongside the Step 6 prototype. The dispatcher no-ops on domains without a `config/knob_01_surface_augmentation/<domain>.yaml` (warns in the log). When movies lands, `paraphrase_long` activates for the plot field — `prompt_long_v1.txt` must be authored at that point. No code change required when those domains come online — only new YAML + (for movies) the long-text prompt.

**Rejected alternatives.**

- **LLM paraphrase at medium.** Rejected. Medium's target state (recognisable per-source paraphrases with substantial token overlap) is reachable with the curated tables + EDA operators, and reviewing 10 hand-inspected examples per domain during Step 6 design confirmed that rule-based output is indistinguishable from LLM output at this difficulty. Keeping medium deterministic (a) halves the ablation matrix's LLM surface, (b) keeps the medium variant reproducible without API access, and (c) preserves the guiding-principle default of "simplest deterministic thing that realises the target state".
- **LLM paraphrase at easy.** Rejected. Easy is passthrough or normalize-down; there is no creative task. Using an LLM to canonicalise a country form would be comically expensive for a string-replace.
- **BART-error generation / back-translation / contextual word replacement / heavy ML paraphrase models.** Rejected under [plan_algorithmselection.md §Decision framework tiebreaker 4](../plan_algorithmselection.md#decision-framework-deterministic-in-house-vs-literature-method-vs-llm) against heavyweight ML methods — violates determinism, validation cost, and dependency weight simultaneously. LEMONADE via an LLM is cheaper, more controllable, and already validated for ER surface augmentation; back-translation specifically introduces locale drift which would collide with Knob 5's locale axis and break independent togglability.
- **WordNet synonym replacement (EDA Synonym Replacement operator).** Rejected because WordNet has no synonyms for brand names, proper nouns, and ~all entity labels in the three target domains; the operator degenerates to a no-op on the tokens we actually care about. Cited the EDA paper's own limitations section.
- **EDA Random Insertion operator.** Rejected because inserting a random synonym at a random position in an entity label changes entity identity too often (e.g. `"Ocarina of Time" → "comedy Ocarina of Time"`); EDA's own limitations section calls this out for ER use.
- **Ditto span-level operators** ([../literature-search-generation/ditto/paper.md](../literature-search-generation/ditto/paper.md)) at medium. Considered — Ditto's `span_del` and `span_shuffle` are EDA-adjacent and cited in the PIPELINE_MAP. Rejected as redundant once EDA Random Swap / Random Deletion are in place; adding them would duplicate coverage without increasing expressive range.
- **MixER latent interpolation** ([../literature-search-generation/mixer_latent_interpolation_er/paper.md](../literature-search-generation/mixer_latent_interpolation_er/paper.md)). Rejected: latent-space augmentation produces pairs for training, not surface-level paraphrases of source records. Orthogonal problem.
- **Fabricator training-data generation** ([../literature-search-generation/fabricator_training_data_gen/paper.md](../literature-search-generation/fabricator_training_data_gen/paper.md)). Rejected as it generates entire labeled examples rather than per-cell paraphrases; wrong granularity.

**Implementation handoff.** Everything Step 6 needs to implement this knob without re-reading surrounding cards:

- **Target script:** `usecases_synthetic/scripts/apply_knob_01_surface_augmentation.py` (new, convention matches `apply_knob_06_noise.py`). Standalone runnable from repo root.
- **Function shape (illustrative):**
  ```python
  def apply_knob_01(
      domain: str,
      level: Literal["easy", "medium", "hard"],
      sources: dict[str, pd.DataFrame],            # post-Knob-4, pre-Knobs-5/6/7 (joint phase)
      fusion_gold: pd.DataFrame,                   # anchor-survivor floor lookup; mutable via gold-extend on collapse
      attribute_classes: dict[str, dict[str, Literal["primary", "key", "secondary", "categorical"]]],
      joint_provenance_index: pd.DataFrame,        # rows emitted by Knobs 1/5/6/7 so far — cell-collision check
      config_path: Path,                           # usecases_synthetic/config/knob_01_surface_augmentation/<domain>.yaml
      llm_cache_dir: Path,                         # usecases_synthetic/cache/knob_01_paraphrases/<domain>/<variant>/
      prompts_dir: Path,                           # usecases_synthetic/config/knob_01_surface_augmentation/_prompts/
      output_dir: Path,
      seed: int,
      llm_client: LLMClient | None = None,        # hard path only; None ⇒ level must not be "hard"
  ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
      """Returns (paraphrased_sources, provenance_df, skipped_df, gold_extend_df, gold_revision_summary_df)."""
  ```
- **Inputs the script reads:**
  - Source DataFrames (via `PyDI.io.load_*`, preserving `df.attrs["dataset_name"]`), post-Knob-2 and post-Knob-4 per the canonical S1 order.
  - The fusion gold artifact from `usecases/<domain>/input/fusion/` — read-write under the gold-extend-on-collapse path (the only knob authorised to mutate the gold); byte-checksummed at run start and on every mutation.
  - Per-domain config at `usecases_synthetic/config/knob_01_surface_augmentation/<domain>.yaml`.
  - Shared tables at `usecases_synthetic/config/knob_01_surface_augmentation/_tables/{stopwords_en.yaml, eda_params.yaml, contamination.yaml}`.
  - Prompt files at `usecases_synthetic/config/knob_01_surface_augmentation/_prompts/{prompt_short_v1.txt, prompt_categorical_v1.txt}` (hard level only; `prompt_long_v1.txt` intentionally absent in v1).
  - Joint provenance index from any Knobs 1/5/6/7 that have already run in the current variant (cell-collision contract).
  - LLM client (hard level only; must be None for easy/medium).
- **Outputs the script writes** (under the variant directory):
  - Paraphrased source files in `input/data/` (same format as input — XML/JSON/CSV).
  - Provenance log at `output/provenance/knob_01_surface_augmentation.csv`.
  - Skipped-cell audit at `output/provenance/knob_01_skipped.csv`.
  - Gold-extend mirror log at `output/provenance/knob_01_gold_extend.csv` (hard + committee-collapse path only; empty file when no extensions happened).
  - Gold revision summary at `output/baselines/knob_01_gold_revision.json` (pre-run checksum, post-run checksum, extension count).
  - (Hard path) LLM paraphrase cache at `usecases_synthetic/cache/knob_01_paraphrases/<domain>/<variant>/<cell_hash>.json`, **committed to the repo**.
- **Pipeline integration.** Knob 1 sits inside the `Knobs 1/5/6/7` joint phase of the canonical S1 order from [README.md](README.md#canonical-knob-application-order) and is the first to run within that phase. It runs *after* Knob 4 (coverage skew) has fixed entity presence and *before* Knobs 5/6/7 record their per-cell transforms. Because Knob 1 is first in the joint phase, the cell-collision contract degenerates to a no-op on the first run but must still be wired (ablation configurations may reorder the joint phase).
- **Exported single-cell paraphrase callable (Knob 4 fallback consumer, locked C3 contract).** Knob 4's easy-level fabrication fallback path ([knob_04_coverage_skew.md §Easy fabrication mode](knob_04_coverage_skew.md#easy-fabrication-mode)) calls into Knob 1's medium-level table-driven + EDA operator pool to paraphrase a sibling-copied row. The contract is:
  ```python
  def paraphrase_value_for_knob_04(
      domain: str,
      attribute_class: Literal["primary", "key", "secondary", "categorical"],
      original_value: str,
      config: Knob01Config,                       # already-loaded per-domain YAML + shared tables
      rng: numpy.random.Generator,                # threaded from Knob 4's seeded RNG via spawn()
  ) -> tuple[str, dict]:
      """Apply one Knob 1 medium-level operator to a single value.

      Returns (new_value, transform_params_dict). Deterministic given (rng, config).
      Pinned to medium level (table dispatcher + EDA random_swap / random_delete).
      Never invokes the LLM; never reads or writes the joint provenance index
      (Knob 4 owns the provenance row for the fabricated cell).
      """
  ```
  Pinned semantics: **medium level only** (table-driven + EDA), never hard (no LLM call from inside Knob 4 — Knob 4 owns its own LLM primary path separately). RNG is threaded from Knob 4 via `numpy.random.SeedSequence(seed).spawn(...)` so Knob 4's reproducibility is preserved. The callable does not write to `output/provenance/knob_01_*.csv` — Knob 4 records the fabrication on its own provenance row with `transform_fn=propagate_and_paraphrase` and the returned `transform_params_dict` nested under `knob_01_paraphrase_params`. This is the one and only Knob 1 entry point for cross-knob consumers; no other knob calls into Knob 1.
- **Dependencies:** stdlib + `pandas` + `numpy` + `pyyaml` for easy/medium. Hard path additionally needs an LLM client (OpenAI-compatible, loaded via `python-dotenv` per [CLAUDE.md](../CLAUDE.md#testing-notes)) and the contamination-check helper module (new, small, in-repo). No new runtime dependencies beyond what the rest of the pipeline already introduces.
- **Authoring tasks before first run:**
  1. Populate `usecases_synthetic/config/knob_01_surface_augmentation/{companies,games,music}.yaml` with `attribute_classes`, `stopword_list`, `key_token_skiplist`, `baseline_above_target_rules`, `abbreviation_table`, `synonym_table`, `article_drop_pattern`, `categorical_vocab_map`, `paraphrase_rate_primary/key/secondary`, `operator_mix`, `llm_prompt_version=v1`, `llm_model_id`, `llm_temperature=0.0`, `anchor_survivor_floor=true`.
  2. Author `usecases_synthetic/config/knob_01_surface_augmentation/_prompts/{prompt_short_v1.txt, prompt_categorical_v1.txt}` by adapting LEMONADE's published *Soft* and *Strong* modification prompts per domain. `prompt_long_v1.txt` **not authored** in v1.
  3. Copy the English stopword list and EDA parameter defaults from the EDA paper into `_tables/`.
- **Smoke test.** For each domain with a config, run the script at all three levels and assert (a) the anchor-survivor floor holds for every fusion-gold entity (≥1 source retains a non-paraphrased primary), (b) the per-(entity, attribute) clean-survivor floor holds unless the YAML explicitly waives it, (c) total paraphrased cell count at hard > medium > easy per attribute class, (d) the provenance row count equals the number of cell-value mutations, (e) the fusion gold file on disk is byte-identical at easy and medium; at hard, any mutation is accompanied by a matching `knob_01_gold_extend.csv` row and a `gold_revision` increment, (f) every LLM-touched cell has a corresponding cache file and a contamination-check record with both probes passing, (g) re-running with the same seed and a populated cache produces bit-identical outputs on hard, (h) the cell-collision check rejects any cell already present in the joint provenance index (verified by seeding the index with a synthetic row before the run), (i) no `paraphrase_long` rows are emitted in v1, (j) EDA `random_swap` never touches a stopword or key-token position, (k) `normalize_to_canonical` only fires where a `baseline_above_target_rules` entry names the cell.
