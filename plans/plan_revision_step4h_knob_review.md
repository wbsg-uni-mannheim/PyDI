# plan_revision_step4h_knob_review.md

Step 4h of [plan_revision.md](plan_revision.md): pre-rerun knob tuning
review. Walks through every per-domain knob YAML across the 8 active
knobs × 4 active domains (music, games, companies, products) and
verifies six criteria (direction, realised sizing, LLM model_id,
K8 anonymize-at-hard, id_columns convention, per-attribute scope).
**This document is review-only — no YAML edits land until the user
signs off on the proposed change set.**

---

## TL;DR — what needs sign-off

| Domain | PASS | REVISE | Highlights for user gate |
|---|---:|---:|---|
| music | 7 | 1 | K2 easy target 0.20 < realised baseline 0.26 (noop). |
| games | 7 | 1 | K2 easy 0.20 AND medium 0.50 < realised baseline 0.67 (both noop). |
| companies | 6 | 2 | K2 `interp_pair_factor` under-realising (170/211/195 vs 200/500/800); K10 default trust-winner-is-DBpedia conflicts with `feedback_dbpedia_noise_profile`. |
| products | 3 | 5 | K2 + K10 attribute scope gaps (priceCurrency / model_number / title_description); K3 needs sparse-attribute drop-rate overrides for `model_number`/`chipset_name`/`vram_gb`. |
| **Total** | **23** | **9** | |

**Cross-cutting theme: K2 dial calibration.** Every domain except
products has K2 easy targets below natural baseline, which makes easy
a noop by construction. Music + games REVISEs are entirely K2 sizing;
companies adds a budget gap; products' K2 issue is scope (missing
attributes), not sizing.

**Plan-doc drift discovered during this review** (out-of-band, not
itself a knob-config issue but worth recording):

1. **R2 (LLM model_id migration) is already silently applied.** The
   plan claims products + movies K1/K2/K4 still pin
   `claude-opus-4-6`. Verified 2026-05-27:
   [knob_01_surface/products.yaml:116](../usecases_synthetic/config/knob_01_surface/products.yaml#L116),
   [knob_02_niche/products.yaml:148](../usecases_synthetic/config/knob_02_niche/products.yaml#L148),
   [knob_04_coverage/products.yaml:49](../usecases_synthetic/config/knob_04_coverage/products.yaml#L49),
   and all three equivalent movies files already read
   `llm_model_id: gpt-5.4-mini`. R2 can be marked done in
   plan_revision.md.
2. **R1 (products data_cleaned_final schema swap) is largely already
   applied** in the source data + domain config + most knob YAMLs.
   `products_<1..4>.json` carries all 7 R1 columns plus 12 extension
   columns (`read_speed_mb_s`, `bus_type`, `interface_type`,
   `width_mm`, `length_mm`, `height_mm`, `weight_g`,
   `storage_connection_type`, `memory_type`, `color`, `form_factor`).
   The domain config attribute_classes (
   [domains/products.yaml:67-79](../usecases_synthetic/config/domains/products.yaml#L67-L79))
   explicitly classifies the 11 in-scope attributes; the extension
   columns are intentionally left unclassified per the comment at
   [domains/products.yaml:65-66](../usecases_synthetic/config/domains/products.yaml#L65-L66).
   Remaining R1 work is the per-knob scope completion items below
   under the products section (K2 canonical_schema + K10
   attribute_mapping/targets gaps for `priceCurrency` /
   `model_number` / `title_description`), plus the R1 step-7 rerun.

---

## music

### K1 (surface paraphrase)
**Verdict: PASS**

- Direction: `paraphrase_rate_primary` 0.0 → 0.02 → 0.08;
  `paraphrase_rate_key` 0.0 → 0.04 → 0.12; `paraphrase_rate_secondary`
  0.0 → 0.08 → 0.20; `paraphrase_rate_categorical` 0.0 → 0.04 → 0.12 —
  all monotone increasing.
- Realised sizing: per-knob realised CSV does not yet exist on disk
  (step 4f landed; will populate after the next regen).
- model_id: `gpt-5.4-mini` at
  [knob_01_surface/music.yaml:179](../usecases_synthetic/config/knob_01_surface/music.yaml#L179)
  — matches memory.
- id_columns: lines 25–28, all sources → `id`.
- Per-attribute scope: covers name (primary), artist (key),
  release-country (key), release-date (secondary), label (secondary);
  lastfm correctly thinned (no secondary fields).
- Proposed change: none.

### K2 (niche)
**Verdict: REVISE**

- Direction: `target_corner_case_ratio` 0.20 → 0.50 → 0.80 — monotone
  increasing.
- Realised sizing — **concern**: music baseline corner ratio ≈ 0.257
  (per G1 step-2 diagnosis). Easy target 0.20 < baseline 0.257 →
  easy reduces to `noop_baseline_above_target` by construction.
- model_id: `gpt-5.4-mini` at
  [knob_02_niche/music.yaml:150](../usecases_synthetic/config/knob_02_niche/music.yaml#L150).
- id_columns: lines 11–14, all sources → `id`.
- Per-attribute scope: covers name, artist, release-date,
  release-country, duration, genre, label with documented coverage
  gaps; correct.
- **Proposed change**: raise easy target above baseline so easy is no
  longer a noop. Suggest 0.30 (above the 0.257 baseline; gives a
  small but real K2 lift at easy). Medium 0.50 and hard 0.80 stay.

### K3 (drop)
**Verdict: PASS**

- Direction: primary 0.0/0.0/0.03, key 0.02/0.10/0.25, secondary
  0.05/0.15/0.35 — monotone non-decreasing.
- Realised sizing: K3 not instrumented with realised CSV; configured
  rates monotone. Stretch factor 1.5 at hard amplifies.
- model_id: n/a.
- id_columns: lines 11–14, all → `id`.
- Per-attribute scope: correctly excludes schema-level absent
  attributes (musicbrainz {genre,label}; lastfm
  {release-country,release-date,label}).
- Proposed change: none.

### K4 (coverage)
**Verdict: PASS**

- Direction: `target_coverage_histogram` shifts from 3-source-dominant
  (easy 0.90 / hard 0.15) to singleton-dominant (easy 0.0 / hard 0.55)
  — monotone toward harder linkage.
- Realised sizing: K4 targets inherited from companies; recalibration
  deferred to S13 per YAML comment.
- model_id: `gpt-5.4-mini` at
  [knob_04_coverage/music.yaml:46](../usecases_synthetic/config/knob_04_coverage/music.yaml#L46).
- id_columns: lines 12–15, all → `id`.
- Per-attribute scope: K4 is coverage-focused, not attribute-focused;
  scope appropriate.
- Proposed change: none.

### K5 (format)
**Verdict: PASS**

- Direction: distinct format families per level — duration 2 → 3 → 4,
  date 2 → 3 → 4. Monotone increasing.
- Realised sizing: step-2 audit on existing variants showed K5
  music intensity PASS (2/2/2 distinct families — flat; expected to
  hit 2/3/4 on the next regen with the audit running against the
  config above).
- model_id: n/a.
- id_columns: lines 33–36, all → `id`.
- Per-attribute scope: targets date + duration; correct.
- Proposed change: none.

### K6 (noise)
**Verdict: PASS**

- Direction: noise rates per class 0.0/0.0/0.01 (primary),
  0.0/0.02/0.06 (key), 0.01/0.04/0.12 (secondary) — monotone
  non-decreasing. Operator mix and edit caps escalate.
- Realised sizing: not instrumented; deterministic per operator
  weights.
- model_id: n/a.
- id_columns: lines 15–18, all → `id`.
- Per-attribute scope: covers all declared attributes; taxonomy
  bindings conditional per source.
- Proposed change: none.

### K8 (naming)
**Verdict: PASS**

- Direction + C5 anonymize-at-hard: `level_assignments` map easy →
  descriptive, hard reaches anonymized (lastfm rung 3 at
  [knob_08_naming/music.yaml:23](../usecases_synthetic/config/knob_08_naming/music.yaml#L23)).
  Satisfies C5 directive.
- Realised sizing: step-2 audit reported K8 music naming_intensity
  (rung-weighted) PASS (0/24/48) — monotone increasing.
- model_id: n/a.
- id_columns: id columns correctly omitted from rename_table —
  `id: id` only appears in sm_mapping (lines 31, 41, 51); rename_table
  (lines 61–186) does not rename `id`.
- Per-attribute scope: canonical 9 columns across three sources;
  source-conditional bindings.
- Proposed change: none.

### K10 (reliability)
**Verdict: PASS**

- Direction: per-attribute winner share decreases easy→hard (e.g.
  `name`: musicbrainz 0.85 → 0.65 → 0.40) — monotone decreasing
  winner share = monotone increasing dispersion. `compromise_rate`
  0.0/0.05/0.15 and `corr_strength` 0.0/0.20/0.50 amplify.
- Realised sizing: not instrumented pre-step-4f; rate-based audit
  will run post-regen.
- model_id: n/a.
- id_columns: lines 17–20, all → `id`.
- Per-attribute scope: correctly scopes to ≥2-source attributes (name,
  artist, release-date, release-country, duration); excludes
  discogs-only attributes (genre, label).
- Proposed change: none.

### music — summary
**7 PASS, 1 REVISE.** Sole revision is K2 easy target raise
(0.20 → ~0.30) so easy is no longer a noop. All other knobs are
fit-for-purpose for step 5/6 regen.

---

## games

### K1 (surface paraphrase)
**Verdict: PASS**

- Direction: primary 0.0/0.02/0.08, key 0.0/0.04/0.12, secondary
  0.0/0.08/0.20, categorical 0.0/0.04/0.12 — monotone increasing.
- Realised sizing: K1 realised CSV doesn't exist on existing variants
  (step 4f instrumentation lands at next regen). Operator mix
  escalates easy (normalize-only) → hard (abbreviate + EDA + LLM).
- model_id: `gpt-5.4-mini` at
  [knob_01_surface/games.yaml:204](../usecases_synthetic/config/knob_01_surface/games.yaml#L204).
- id_columns: `{dbpedia: id, metacritic: id, sales: id}` (lines 26–29)
  — PASS. F11 dropped the `metacritic_sales` pair but the `sales`
  source itself is still loaded for the `dbpedia_sales` pair, so
  retaining the entry is correct.
- Per-attribute scope: attribute classes match games canonical
  schema (primary name; key platform/ESRB; secondary developer/
  publisher/genres).
- Proposed change: none.

### K2 (niche)
**Verdict: REVISE** (high priority)

- Direction: `target_corner_case_ratio` 0.20 → 0.50 → 0.80 — monotone
  increasing.
- Realised sizing — **two-level violation**: games natural baseline
  corner ratio ≈ 0.67 (very high due to game re-releases / platform
  variants — observed in existing
  [usecases/games-augmented/{easy,medium,hard}/output/baselines/knob_02_realised.csv](../usecases/games-augmented/easy/output/baselines/knob_02_realised.csv)
  reporting realised 0.67 / 0.66 / 0.67). Both easy (0.20 < 0.67) and
  medium (0.50 < 0.67) fall below baseline → both reduce to
  `noop_baseline_above_target`. Only hard (0.80 > 0.67) triggers
  interpolation, and it was blocked last run by the C1 strict_cache
  bug (now fixed).
- model_id: `gpt-5.4-mini` at
  [knob_02_niche/games.yaml:147](../usecases_synthetic/config/knob_02_niche/games.yaml#L147).
- id_columns: lines 10–13, all → `id`.
- Per-attribute scope: canonical_schema lists name/platform/genres/
  developer/releaseYear; K2 works on near-twin density, not
  per-attribute, so omitting score columns is fine.
- **Proposed change**: raise easy AND medium above baseline. Suggest:
  - easy → 0.70 (or pin at baseline 0.67 if the policy is "easy ≥
    baseline strictly"; 0.70 leaves headroom for K2 to do *some*
    interpolation work and not be entirely noop).
  - medium → 0.78.
  - hard → 0.85.
  Alternative if the user prefers minimal disturbance: easy =
  medium = baseline (≈ 0.67) — K2 noops at easy + medium by design;
  difficulty comes from K1/K3/K4/K5/K6/K8/K10. Hard 0.85 stays. This
  is the more conservative read of C2 ("K2 has no `drop corner-
  touching` operator at easy"; G1 cause #2 in plan_revision.md).

### K3 (drop)
**Verdict: PASS**

- Direction: primary 0.0/0.0/0.03, key 0.02/0.12/0.30, secondary
  0.05/0.18/0.40 — monotone non-decreasing.
- Realised sizing: not instrumented; stretch_factor 1.5 at hard
  amplifies.
- model_id: n/a.
- id_columns: lines 11–14, all → `id`.
- Per-attribute scope: covers all sources' attributes (dbpedia 6,
  metacritic 8, sales 10). Complete.
- Proposed change: none.

### K4 (coverage)
**Verdict: PASS**

- Direction: histogram skews from 3-source-dominant (easy 0.90) →
  singleton-dominant (hard 0.55). Monotone.
- Realised sizing: deterministic target histogram; not adaptive.
- model_id: `gpt-5.4-mini` at
  [knob_04_coverage/games.yaml:45](../usecases_synthetic/config/knob_04_coverage/games.yaml#L45).
- id_columns: lines 9–12, all → `id`.
- Per-attribute scope: primary_columns per source; correct.
- Proposed change: none.

### K5 (format)
**Verdict: PASS**

- Direction: date pool 2 → 3 → 4 families; money 2 → 3 → 4
  magnitudes; unit pool grows millions → millions/raw →
  millions/raw/thousands.
- Realised sizing: step-2 audit reported K5 games distinct families
  PASS (1/2/3) on existing variants — monotone increasing.
- model_id: n/a.
- id_columns: lines 26–29, all → `id`.
- Per-attribute scope: targets date attributes (`launch_yr`,
  `year_published`, `launch_dt`) and money (`units_sold_mm`).
- Proposed change: none.

### K6 (noise)
**Verdict: PASS**

- Direction: primary 0.0/0.0/0.01, key 0.0/0.02/0.06, secondary
  0.01/0.04/0.12. Operator caps grow (max_edits 1/1/3, max_ocr 1/1/2,
  max_truncate 1/2/3).
- Realised sizing: not instrumented; monotone by construction.
- model_id: n/a.
- id_columns: lines 10–13, all → `id`.
- Per-attribute scope: full per-source mapping (lines 48–75).
  Taxonomy bindings for platform + genre.
  `numeric_jitter_max_relative=0.02` targets numeric fields.
- Proposed change: none.

### K8 (naming)
**Verdict: PASS**

- Direction + C5 anonymize-at-hard: rung_rank weighted intensity
  0 → 3 → 43 across levels (step-2 audit PASS). Sales source reaches
  anonymized at hard
  ([knob_08_naming/games.yaml:29](../usecases_synthetic/config/knob_08_naming/games.yaml#L29))
  satisfying C5.
- Realised sizing: step-2 audit on existing variants confirmed
  monotone via the new naming_intensity metric (replaces legacy
  edit-distance FAIL of 192/83/137).
- model_id: n/a.
- id_columns: K8 doesn't declare id_columns block (naming renames
  source columns, not entity ids; id passes through unchanged).
- Per-attribute scope: rename_table covers 6 dbpedia + 8 metacritic +
  10 sales columns × 4 rungs each. Comprehensive.
- Proposed change: none.

### K10 (reliability)
**Verdict: PASS**

- Direction: for all 8 attribute_targets the metacritic winner share
  drops 0.85 → 0.65 → 0.40 (or 0.50 for criticScore/userScore/ESRB).
  `compromise_rate` 0.0/0.05/0.15; `corr_strength` 0.0/0.20/0.50.
- Realised sizing: not instrumented pre-step-4f; rate-based audit
  will run post-regen.
- model_id: n/a.
- id_columns: lines 16–19, all → `id`.
- Per-attribute scope: 8 attribute_targets covering ≥2-source
  attributes (name, platform, ESRB, releaseYear, developer, genres,
  criticScore, userScore). Single-source attributes (publisher,
  series, globalSales) correctly excluded.
- Proposed change: none.

### games — summary
**7 PASS, 1 REVISE.** K2 is the load-bearing fix: both easy (0.20)
and medium (0.50) are below the natural baseline 0.67 and therefore
noop. Per-knob proposal above offers two options — strict "easy
above baseline" (0.70 / 0.78 / 0.85) or conservative "K2 noops at
easy + medium, difficulty comes from other knobs"
(baseline / baseline / 0.85). User to pick.

---

## companies

### K1 (surface paraphrase)
**Verdict: REVISE** (deferred — pending step-4f realised data)

- Direction: per-class rates 0.0 → 0.02–0.12 → 0.08–0.20; operator mix
  escalates. Monotone.
- Realised sizing: step-4f K1 realised CSV not yet produced for
  companies (gated on the LLM-spend authorisation that pre-populates
  the K2 cache). Verdict: configuration direction/scope are sound;
  hold for now and re-audit when realised data lands. Listed as
  REVISE only to flag the pending audit, not because a config change
  is required.
- model_id: `gpt-5.4-mini` at
  [knob_01_surface/companies.yaml:203](../usecases_synthetic/config/knob_01_surface/companies.yaml#L203).
- id_columns: `{dbpedia: id, forbes: id, fullcontact: id}`
  (lines 28–31).
- Per-attribute scope: covers name, country, city, industry,
  founders. Numeric attributes (revenue, employees) intentionally
  excluded; not typical paraphrase targets.
- Proposed change: **defer**. Re-audit post-regen.

### K2 (niche)
**Verdict: REVISE** (high priority)

- Direction: 0.20 → 0.50 → 0.80 — monotone increasing.
- Realised sizing — **critical under-realisation**: companies-small
  realised 0.170 / 0.211 / 0.195 vs configured 0.20 / 0.50 / 0.80.
  Even at hard, realised barely moves from baseline. Two causes
  combined: (a) C1 strict_cache bug (now fixed in code, but the
  cache is empty for companies-FULL pair-hashes), and (b)
  `interp_pair_factor: 0.05` may be too small to close the gap once
  the cache populates.
- model_id: `gpt-5.4-mini` at
  [knob_02_niche/companies.yaml:215](../usecases_synthetic/config/knob_02_niche/companies.yaml#L215).
- id_columns: lines 18–21, all → `id`.
- Per-attribute scope: name, country, city, industry, sector,
  founded; coverage refreshed 2026-05-04.
- **Proposed change**: raise `interp_pair_factor` from 0.05 to 0.10
  (doubles the budget). Couples with C7 (LLM spend authorisation) —
  the C1 fix alone won't close the gap without (a) cache pre-
  population AND (b) enough budget to interpolate.

### K3 (drop)
**Verdict: PASS**

- Direction: primary 0.0/0.0/0.03, key 0.02/0.10/0.25, secondary
  0.05/0.15/0.35 — monotone non-decreasing. Stretch 1.5 at hard.
- Realised sizing: not instrumented; rates and transform sound.
  DBpedia caps sensitive attributes (`total_assets_val`,
  `annual_income`, `keypeople_name` at 0.02 ceiling, lines 90–95) —
  guards against stretch overshoot.
- model_id: n/a.
- id_columns: lines 10–13, all → `id`.
- Per-attribute scope: primary (name), key
  (country/city/industry), secondary (founded/assets/revenue/
  keypeople). Sensible coverage on refreshed CSVs.
- Proposed change: none.

### K4 (coverage)
**Verdict: PASS**

- Direction: easy 3-source dominant (0.90) → hard singleton-dominant
  (0.55). Monotone.
- Realised sizing: `fabrication_mode: paraphrase_only` keeps cost
  bounded.
- model_id: `gpt-5.4-mini` at
  [knob_04_coverage/companies.yaml:73](../usecases_synthetic/config/knob_04_coverage/companies.yaml#L73)
  (unused under paraphrase_only mode, kept for consistency).
- id_columns: lines 22–25, all → `id`.
- Per-attribute scope: primary columns only (org_name / company /
  Attribute_2). Correct for K4's entity-level operation.
- Proposed change: none.

### K5 (format)
**Verdict: PASS**

- Direction: date families 2 → 3 → 4; money 2 → 3 → 4. Within-source
  consistency shifts from source-level (easy/medium) → per-row
  (hard).
- Realised sizing: normalize-down threshold 0.85 conservative.
  Magnitude pools sized for companies' mixed billions / raw data
  (DBpedia: mixed; Forbes: raw).
- model_id: n/a.
- id_columns: lines 31–34, all → `id`.
- Per-attribute scope: targets date (`established`, `Attribute_6`)
  and money (`total_assets_val`/`annual_income`, `asset_value`/
  `sales_figure`).
- Proposed change: none.

### K6 (noise)
**Verdict: PASS**

- Direction: primary 0.0/0.0/0.01, key 0.0/0.0/0.06, secondary
  0.01/0.04/0.12 — monotone non-decreasing. Operator caps and
  weights escalate.
- Realised sizing: GICS taxonomy binding for industry; numeric jitter
  ±2% conservative for financials.
- model_id: n/a.
- id_columns: lines 15–18, all → `id`.
- Per-attribute scope: covers primary + key + secondary across three
  sources; cleanup rule for `forbes.region` is easy-only
  normalization.
- Proposed change: none.

### K8 (naming)
**Verdict: PASS**

- Direction + C5 anonymize-at-hard: dbpedia descriptive → abbreviated
  → cryptic; forbes descriptive → descriptive → abbreviated;
  fullcontact descriptive → cryptic → anonymized. Hard reaches
  anonymized on fullcontact
  ([knob_08_naming/companies.yaml:149-164](../usecases_synthetic/config/knob_08_naming/companies.yaml#L149-L164))
  — C5 satisfied.
- Realised sizing: rename_table comprehensive (8 dbpedia / 6 forbes /
  5 fullcontact columns × 4 rungs).
- model_id: n/a.
- id_columns: lines 30–32, all → `id`. id_columns intentionally
  omitted from rename_table (line 59 comment) — correct.
- Per-attribute scope: SM mapping covers 8 canonical attributes.
- Proposed change: none.

### K10 (reliability)
**Verdict: REVISE** (decision required — memory `feedback_dbpedia_noise_profile` interaction)

- Direction: per-attribute winner share monotone decreasing toward
  flatter hard distribution. Mechanically correct.
- Realised sizing: not instrumented; configured targets dimensionally
  sound.
- model_id: n/a.
- id_columns: lines 24–27, all → `id`.
- Per-attribute scope — **flag for user decision**: per
  `feedback_dbpedia_noise_profile`, DBpedia should not default as a
  per-attribute trust winner unless the attribute is GICS / controlled-
  vocab. Current K10 winners:
  - `country` (lines 80–84): dbpedia 0.85 / 0.65 / 0.40 — dbpedia wins
    at easy/medium. Country is *not* a GICS-mapped attribute.
  - `founded` (lines 86–95): dbpedia 0.90 / 0.70 / 0.50 — dbpedia wins
    at easy/medium. Not GICS.
  - `name` (lines 73–78): forbes 0.85 / 0.65 / 0.40 — forbes wins
    (consistent with the memory).
  - `city` (lines 96–104): fullcontact 0.90 / 0.70 / 0.50 — fullcontact
    wins (consistent).
  - `assets`, `revenue` (lines 105–139): forbes wins (consistent).
  The K10 YAML comments at lines 63–65 explicitly justify country /
  founded → dbpedia as "fusion gold was authored from dbpedia"
  (gold-alignment override). This conflicts with the memory's
  source-quality prior.
- **Proposed change**: user decision — either (a) keep current
  gold-alignment winners (no YAML change; add an explicit comment
  noting the memory override), or (b) flip country / founded winners
  away from dbpedia (forbes for country, forbes/fullcontact for
  founded depending on coverage) and accept that the human-authored
  fusion gold may need re-authoring against the new winners. (a) is
  the lower-risk path; (b) is the methodologically purer one.

### companies — summary
**6 PASS, 2 REVISE (+ K1 deferred-audit).** Key items:
1. K2 budget bump (`interp_pair_factor` 0.05 → 0.10) — required for
   K2 to do meaningful work at any level on companies-FULL.
2. K10 dbpedia-as-trust-winner decision — user choice between
   gold-alignment override (keep) vs source-quality prior (flip).
3. K1 realised audit deferred until step-4f K1 CSV is produced
   (gated on LLM-spend authorisation).

Companies-FULL R7 readiness once these land + the larger fusion gold
arrives: configuration is otherwise ready; no R0 (companies-FULL)
blockers remain on the knob side.

---

## products

### K1 (surface paraphrase)
**Verdict: PASS**

- Direction: primary 0.0/0.02/0.08, key 0.0/0.04/0.12, secondary
  0.0/0.08/0.20, categorical 0.0/0.04/0.12 — monotone.
- Realised sizing: K1 realised CSV will populate next regen.
- model_id: `gpt-5.4-mini` at
  [knob_01_surface/products.yaml:116](../usecases_synthetic/config/knob_01_surface/products.yaml#L116).
  **R2 already silently applied** — see TL;DR §1.
- id_columns: lines 11–15, all four products_<n> sources → `id`.
- Per-attribute scope: 11 attributes — title (primary); brand,
  product_type (key); description, price, title_description, model,
  model_number, chipset_name, vram_gb, storage_gb (secondary). Already
  R1-anticipating; covers all the R1 column additions.
- Proposed change: none. **Confirm**: priceCurrency intentionally
  excluded from K1 surface paraphrase scope? It's a 3-letter ISO
  code; paraphrasing it makes no sense, so excluding is reasonable.

### K2 (niche)
**Verdict: REVISE**

- Direction: 0.20 → 0.50 → 0.65 — monotone increasing. Hard target
  (0.65) is lower than music/games/companies (0.80) — calibration
  comment at top of YAML notes products is a smaller domain
  (3012 rows).
- Realised sizing: hard `interpolation_count: 12` may overshoot given
  small cluster sizes (max 4) — YAML self-flags S.5a calibration as
  pending.
- model_id: `gpt-5.4-mini` at
  [knob_02_niche/products.yaml:148](../usecases_synthetic/config/knob_02_niche/products.yaml#L148).
- id_columns: lines 12–16, all four sources → `id`.
- Per-attribute scope — **gap**: canonical_schema (lines 24–34)
  lists 10 attributes but **missing `priceCurrency` and
  `title_description`**. Domain config includes them at
  [domains/products.yaml:72-73](../usecases_synthetic/config/domains/products.yaml#L72-L73);
  K2 should mirror the domain scope so the niche scorer sees them
  in the entity neighborhood.
- **Proposed change**: add `priceCurrency` and `title_description`
  to `canonical_schema:` (after `chipset_name`); replicate in
  `attribute_mapping:` block at lines 38–48; replicate in
  `coverage:` block at lines 32+.

### K3 (drop)
**Verdict: REVISE**

- Direction: primary 0.0/0.0/0.03, key 0.02/0.10/0.25, secondary
  0.05/0.15/0.35 — monotone.
- Realised sizing: per_source_attribute_overrides block is empty
  (line 72). Uniform secondary-class rates apply to all 11
  attributes, including the sparse R1 columns
  (model_number ~40% / chipset_name ~28% / vram_gb ~28%).
- model_id: n/a.
- id_columns: lines 5–9, all → `id`.
- Per-attribute scope: covers all 11 attributes including
  priceCurrency, title_description.
- **Proposed change**: add `per_source_attribute_overrides:` block
  targeting the naturally-sparse R1 columns with higher drop rates
  at medium/hard (e.g. `model_number: {medium: 0.30, hard: 0.45}`,
  `chipset_name: {medium: 0.35, hard: 0.50}`, `vram_gb: {medium:
  0.35, hard: 0.50}`). Or, alternatively, leave K3 uniform and let
  K4 (coverage) carry the sparsity signal. User to pick — the
  per-attribute override is more honest about the realised drop
  shape but adds calibration complexity.

### K4 (coverage)
**Verdict: REVISE** (calibration deferred)

- Direction: histogram skews from 3-source-dominant → singleton
  (45% at hard).
- Realised sizing: products has tiny clusters (max 4). YAML notes
  S.5a calibration pending. Hard target 45% singletons + 30% dual
  may be reasonable; will know once the baseline runs.
- model_id: `gpt-5.4-mini` at
  [knob_04_coverage/products.yaml:49](../usecases_synthetic/config/knob_04_coverage/products.yaml#L49).
- id_columns: lines 11–15, all → `id`.
- Per-attribute scope: primary_columns title only — coverage
  operation is entity-level, doesn't need broader scope.
- **Proposed change**: defer. Add a step-5 sub-task: "after products
  baseline runs, re-audit K4 histogram against realised cluster
  size distribution; recalibrate hard target if needed."

### K5 (format)
**Verdict: PASS**

- Direction: locale pool 2 → 3 → 4; unit pool 1 → 2 → 3 for
  file_size attributes.
- Realised sizing: `file_size` format for vram_gb / storage_gb at
  hard adds bare-bytes representation (`8` → `8192`), maximally
  difficult.
- model_id: n/a.
- id_columns: lines 28–32, all → `id`.
- Per-attribute scope: price (money) + vram_gb / storage_gb
  (file_size) — correctly classified. Other R1 columns are
  text/categorical, no format rules needed.
- Proposed change: none.

### K6 (noise)
**Verdict: PASS**

- Direction: primary 0.0/0.0/0.01, key 0.0/0.02/0.06, secondary
  0.01/0.04/0.12 — monotone. soft_global_primary_cap_hard=0.35
  guards primary.
- Realised sizing: deterministic by operator weights.
- model_id: n/a.
- id_columns: lines 10–14, all → `id`.
- Per-attribute scope: all 11 attributes covered in attribute_classes
  + attribute_mapping including priceCurrency + title_description.
- Proposed change: none.

### K8 (naming)
**Verdict: PASS**

- Direction + C5 anonymize-at-hard: level_assignments escalate;
  hard reaches anonymized on products_3
  ([knob_08_naming/products.yaml:21](../usecases_synthetic/config/knob_08_naming/products.yaml#L21)).
- Realised sizing: rename_table covers all 11 attributes including
  priceCurrency + title_description across all 4 sources × 4 rungs.
  Comprehensive.
- model_id: n/a.
- id_columns: lines 5–9, all → `id`.
- Per-attribute scope: full per-source × per-rung coverage.
- Proposed change: none.

### K10 (reliability)
**Verdict: REVISE**

- Direction: source distribution easy (55/20/15/10) → hard
  (25/25/25/25). Monotone toward flatter (harder) fusion.
- Realised sizing: not instrumented pre-step-4f.
- model_id: n/a.
- id_columns: lines 11–15, all → `id`.
- Per-attribute scope — **gap**: attribute_mapping (lines 18–30)
  covers 9 attributes — `title, brand, description, price,
  product_type, model, chipset_name, vram_gb, storage_gb`. **Missing
  `priceCurrency` and `model_number`**. attribute_targets (lines
  53–61) has the same 9-attribute gap.
- **Proposed change**: add `priceCurrency` and `model_number` to
  both `attribute_mapping:` and `attribute_targets:`. priceCurrency
  is high-coverage (100%) and should have a stable winner;
  model_number is sparse (~40%) and the K10 dispersion should
  reflect this.

### products — summary
**3 PASS, 5 REVISE.**

**Bundled with R1 schema completion** (most of the REVISE items are
about closing the per-knob R1 scope gaps that already partly
landed):
- K2 canonical_schema — add `priceCurrency`, `title_description`.
- K3 per_source_attribute_overrides — optional, for sparse R1
  columns.
- K10 attribute_mapping + attribute_targets — add `priceCurrency`,
  `model_number`.

**Calibration deferred to step-5 baseline run**:
- K4 histogram — re-audit against realised cluster size distribution.
- K2 hard interpolation_count: 12 — may need post-baseline tweak.

**No-change confirmations**:
- K1, K5, K6, K8 — fully R1-ready.
- R2 (LLM model_id migration) — already silently done.

Top 3 revisions to sign off:
1. **K2 canonical_schema gap** — add `priceCurrency` +
   `title_description` so the niche scorer evaluates the right
   neighborhood.
2. **K10 attribute_mapping gap** — add `priceCurrency` +
   `model_number` so reliability dispersion covers all secondary
   attributes.
3. **K3 sparse-attribute override (optional)** — user to pick
   between explicit overrides for `model_number`/`chipset_name`/
   `vram_gb` or uniform secondary-class drop rates.

---

## Cross-cutting themes + decisions for user gate

1. **K2 sizing is the dominant cross-domain revision.** Music, games,
   and companies all need K2 dial adjustments. Music and games are
   easy/medium-target-below-baseline noops; companies is budget-
   under-realisation. **User decision: which K2 strategy?**
   - **Strategy A (strict "easy above baseline")**: raise easy
     above each domain's natural baseline by ~5pp (music 0.20 →
     0.30; games 0.20 → 0.70; companies stays 0.20 but bump
     `interp_pair_factor` 0.05 → 0.10). K2 contributes a small
     lift at easy on each domain.
   - **Strategy B (conservative "K2 noops at easy on high-baseline
     domains")**: leave music easy = 0.20 (still below baseline
     0.26 — easy K2 is noop), games easy/medium = baseline (0.67;
     noop by design), companies same as A. Accept K2-easy-noop as
     intentional per the G1 cause-#2 framing in plan_revision.md.
   **Recommendation: A** — it gives a real, monotone K2 contribution
   on every level on every domain. B is fine if the user prefers
   minimal disturbance from the current configuration but means K2
   contributes nothing to the easy → medium → hard slope on games.

2. **K10 dbpedia-as-trust-winner on companies.** User to choose
   between gold-alignment override (keep current, add comment) vs
   source-quality prior flip per `feedback_dbpedia_noise_profile`.
   **Recommendation: keep current** — the fusion gold was authored
   from dbpedia, so a source-quality flip would require re-authoring
   the gold against the new winners. Add a YAML comment at
   [knob_10_reliability/companies.yaml:63-65](../usecases_synthetic/config/knob_10_reliability/companies.yaml#L63-L65)
   noting the memory override explicitly so the next reviewer sees
   the intent.

3. **K3 products sparse-attribute override.** Optional. User to pick
   between explicit overrides and uniform secondary-class rates.
   **Recommendation: skip the explicit overrides for now** — start
   with uniform rates, add overrides post-baseline if the realised
   drop shape diverges from the modeled secondary class.

4. **Plan-doc drift items to fold back into plan_revision.md**
   (separate work, not blocking step 4h sign-off):
   - Mark R2 as DONE (already silently applied).
   - Mark R1 as PARTIALLY DONE (data swap + most YAMLs done; remaining
     gap is the K2 + K10 scope items above; bundle R1's remaining
     status with the sign-off here).

5. **K1 realised audit (G9) is still gated on LLM-spend authorisation**
   for the K2 cache pre-population run. The K1 audit infrastructure
   landed in step 4f but no realised CSV exists for any domain yet.
   This is a blocker for any K1-specific verdict refinement and
   becomes load-bearing once the rerun authorisation lands.

---

## Sign-off checklist (for the user)

Tick to authorise; Claude applies the YAMLs + reruns the synthetic-
pipeline test suite before kicking off step 5.

### K1 — signed off 2026-05-27 (overall walkthrough)

- [x] **Per-class paraphrase rates (Option B)** — applied to all 4 domains:
  ```
  paraphrase_rate_primary:    easy 0.0  medium 0.04  hard 0.12   (was 0.02 / 0.08)
  paraphrase_rate_key:        easy 0.0  medium 0.08  hard 0.18   (was 0.04 / 0.12)
  paraphrase_rate_secondary:  easy 0.0  medium 0.16  hard 0.30   (was 0.08 / 0.20)
  paraphrase_rate_categorical: easy 0.0 medium 0.08  hard 0.18   (was 0.04 / 0.12)
  ```
- [x] **operator_mix update** — applied to all 4 domains:
  - medium: ADD `llm_paraphrase: 1.0` (was 0; ~20% draw share)
  - hard: bump `llm_paraphrase: 2.0 → 3.0` (~46% draw share)
  - easy: kept `normalize_to_canonical: 1.0` with a NEW YAML comment
    noting it is dormant under `paraphrase_rate_*.easy: 0.0` and is
    a defensive placeholder for any future easy-rate bump.
- [x] **anchor_survivor_floor** — unchanged.
- [x] **LLM config** — unchanged (gpt-5.4-mini, v1 prompts, temp 0.0).
- [x] **baseline_above_target_rules** — domain-specific additions:
  - music: ADD `lastfm.name → musicbrainz.name (shortest)` (strips
    "Artist - " prefix lastfm sometimes adds).
  - games: ADD `dbpedia.title → metacritic.game_title (shortest)`
    (strips Wikipedia parenthetical disambiguation like
    "(video game)").
  - companies: no addition (1 rule sufficient — `forbes.region → dbpedia.nation`).
  - products: no addition (sources already canonical on categorical
    attributes; heterogeneity is K1-paraphrase / K10-reliability
    territory, not normalize-down).

### K2 overall — signed off 2026-05-27

- [x] **`interp_pair_factor: 0.05`** — kept (all 4 domains).
- [x] **`max_interp_fraction: 0.60`** — kept (all 4 domains).
- [x] **`placement_split: 0.6 → 0.5`** — applied to all 4 domains
  (equal train/test distribution of new corner pairs).
- [x] **Neighbor-search params** — kept (`c_min: 2`, `rrf_k0: 60`,
  `boost_label_collision: 5.0`, `inner_token_threshold: 0.8`).
- [x] **LLM prompt v1 → v2** — bump `llm_prompt_version: v1 → v2` in
  all 4 K2 YAMLs. New prompt
  [interpolate_v2.txt](../usecases_synthetic/config/knob_02_niche/_prompts/interpolate_v2.txt)
  enumerates all 4 supported domains (companies / games / music /
  products) and carries a domain-extension maintenance note. Cache
  invalidated by `prompt_version` bump → invalidates
  `cache/knob_02_interpolations/<domain>/` for every domain on next
  run. Couples with C7 (LLM spend authorisation).
- [x] **`hard_negative_gate` option (a) — full LLM via new
  `gate_mode: full_llm`** field. Mirrors bucket-C policy (every PLM
  disagreement → LLM adjudicator) chosen because 3 of 4 domains
  (music / companies / products) will run K2 hard-negative-gate on
  pre-R7-padding-fix Ditto checkpoints during step 5/6 — calibration
  is in transitional state. Cost estimate ~$3-6 / sweep at
  gpt-5.4-mini pricing. Requires both YAML changes AND a code
  change to [lib/corner_case_miner.py:apply_hard_negative_policy](../usecases_synthetic/lib/corner_case_miner.py)
  to honour the new `gate_mode` field. Per-level `use_llm_adjudicator`
  collapsed to `gate_mode: full_llm` at all three levels.

### K3 — signed off 2026-05-27 (overall walkthrough)

- [x] **Per-class rates (Block A)** — kept all 4 domains as-is:
  - easy: primary 0.0 / key 0.02 / secondary 0.05
  - medium: primary 0.0 / key 0.10 (games 0.12) / secondary 0.15 (games 0.18)
  - hard: primary 0.03 / key 0.25 (games 0.30) / secondary 0.35 (games 0.40)
  - Games keeps its slightly aggressive medium/hard rates (justified by domain density per the games YAML comment).
- [x] **Transform + factors (Block B)** — kept: `compress / identity / stretch` with `compression_factor: 0.7`, `stretch_factor: 1.5`. Bidirectional dial (easy reduces cross-source missingness gap via `propagate_fill`; hard amplifies it via stretched target).
- [x] **Safety caps (Block C)** — kept: `single_source_survivor_cap_hard: 0.05`, `per_cell_ceiling_delta: 0.10`.
- [x] **Per-source attribute overrides** — added for products (the R1 sparse columns):
  ```yaml
  # products K3
  per_source_attribute_overrides:
    products_1: &products_attr_caps
      chipset_name: 0.02    # 72% baseline missing
      vram_gb: 0.02         # 72% baseline missing
      model_number: 0.05    # 60% baseline missing
      storage_gb: 0.05      # 30% baseline missing
    products_2: *products_attr_caps
    products_3: *products_attr_caps
    products_4: *products_attr_caps
  ```
  Music + games + companies overrides unchanged.

### K4 — signed off 2026-05-27 (overall walkthrough)

- [x] **Block A — `target_coverage_histogram` medium** — option (β)
  Conservative singletons. Adds explicit medium ramp (previously `null`
  = noop) so K4 contributes monotonically across all 3 levels.
  ```yaml
  # 3-source domains (music / games / companies)
  target_coverage_histogram:
    easy:   {1: 0.0,  2: 0.10, 3: 0.90}    # unchanged
    medium: {1: 0.20, 2: 0.30, 3: 0.50}    # NEW (was null)
    hard:   {1: 0.55, 2: 0.30, 3: 0.15}    # unchanged

  # products (4 sources)
  target_coverage_histogram:
    easy:   {1: 0.0,  2: 0.05, 3: 0.20, 4: 0.75}    # unchanged
    medium: {1: 0.15, 2: 0.25, 3: 0.30, 4: 0.30}    # NEW (was null)
    hard:   {1: 0.45, 2: 0.30, 3: 0.15, 4: 0.10}    # unchanged
  ```
  Singleton ramp 0 → 20 → 55 (3-source) / 0 → 15 → 45 (products).
  Conservative choice avoids saturating fusion-voting difficulty at
  medium and preserves the easy → medium → hard slope.
- [x] **Block B — `within_source_duplicate_rate` ramp** — added
  medium step:
  ```yaml
  within_source_duplicate_rate:
    easy: 0.0       # unchanged
    medium: 0.01    # NEW (was 0.0)
    hard: 0.02      # unchanged
  ```
  Tests EM intra-source dedup + fusion-with-within-source-near-twin.
  2% ceiling matches realistic intra-source dup rate in
  well-curated sources; medium at 1% sits below the "occasional
  contamination" floor.
- [x] **Block C — safety caps** — kept: `singleton_cap_hard: 0.60`,
  `delta_softening_step: 0.05`.
- [x] **Block D — `fabrication_mode: paraphrase_only`** — kept;
  YAML comment added clarifying that the deterministic K1-medium
  operator pool (`abbreviate` / `eda_random_swap` / `eda_random_delete`)
  is reused for K4 fabrication via
  [`paraphrase_value_for_knob_04`](../usecases_synthetic/lib/surface_operators.py#L566)
  and that `llm_*` fields are declared but unused under this mode.

### K5 — signed off 2026-05-27 (overall walkthrough)

- [x] **Block A — `format_pools_per_level` size escalation** (2 / 3 / 4
  per attribute class) — kept across all 4 domains.
- [x] **Block A — products extension (moderate option a)** — bring
  `read_speed_mb_s + write_speed_mb_s` into K5 scope with a new
  `rate` class:
  ```yaml
  # config/domains/products.yaml — bring into attribute_classes
  read_speed_mb_s: secondary    # ~17% coverage, MB/s baseline
  write_speed_mb_s: secondary   # ~14% coverage, MB/s baseline

  # config/knob_05_format/products.yaml — extend K5 scope
  attribute_classes (per source):
    + read_speed_mb_s: rate
    + write_speed_mb_s: rate

  format_pools_per_level (new rate class):
    easy:   [en_US, plain]
    medium: [en_US, de_DE, plain]
    hard:   [en_US, de_DE, plain, bare]

  unit_pool_per_level (new rate class):
    easy:   {units: [MB/s]}
    medium: {units: [MB/s, GB/s]}
    hard:   {units: [MB/s, GB/s, KB/s]}

  source_magnitude_context (per column entry):
    read_speed_mb_s:  {implicit_unit: MB/s}
    write_speed_mb_s: {implicit_unit: MB/s}
  ```
  **Code change required**: K5 dispatcher needs to register the
  new `rate` class label (small — maps to the same numeric formatter
  as money/file_size with per-class unit pool). Extension columns
  (dimensions, weight, color, bus_type, ...) stay out of scope.
- [x] **Block B — `locale_pool_per_level`** (1 / 2 / 3 locales) — kept.
- [x] **Block C — `within_source_consistency`** (`source / source / row`)
  — kept current shape. **Note**: may revisit after step 5/6 rerun
  results — if K5's medium signal is too weak, introduce option (c)
  partial within-source variation at medium ("row at rate r" — would
  require a code change to add the partial-sampling mode).
- [x] **Block D — `normalize_down_threshold: 0.85`** — kept.
- [x] **Block E — `unit_pool_per_level` + `source_magnitude_context`**
  — kept per-domain calibrations.
- [x] **Block F — domain-specific hard format extras**
  (companies `two_digit_year_us` for date; products `bare` for
  file_size + rate) — kept.

### K6 — signed off 2026-05-27 (overall walkthrough)

- [x] **Block A — `noise_rates_per_level`** — kept current shape
  (easy 0/0/0.01, medium 0/0.02/0.04, hard 0.01/0.06/0.12). K6 +
  K1 + K3 already produce aggressive combined secondary disturbance
  at hard; no bump.
- [x] **Block B — transform + factors** (`compress / identity / stretch`,
  `compression_factor: 0.5`, `stretch_factor: 1.5`) — kept.
- [x] **Block C — `operator_mix`** — kept current per-level shape;
  products gains `taxonomy_walk` at medium (1.0) and hard (1.5)
  after the products taxonomies land (Block G below).
- [x] **Block D — Per-cell edit caps** (`max_edits` 1/1/3,
  `max_ocr` 1/1/2, `max_truncate` 1/2/3) — kept.
- [x] **Block E — `soft_global_primary_cap_hard: 0.35`** — kept.
  Verified at
  [apply_knob_06_noise.py:700-810](../usecases_synthetic/scripts/apply_knob_06_noise.py#L700-L810)
  that the cap denominator is **all EM-linked entities**
  (`entity_groups`), independent of `protection_source`. Silver
  expands per-cell Layer 2 protection but doesn't change the
  global runaway cap.
- [x] **Block F — `numeric_jitter_max_relative: 0.02`** — kept on
  music/games/companies; added to products (Block I).
- [x] **Block G — Products taxonomies** — implement 3 new taxonomy
  CSVs + K6 bindings:
  ```
  usecases/products/input/schemamatching/Storage_Interface_Taxonomy.csv
  usecases/products/input/schemamatching/GPU_Memory_Taxonomy.csv
  usecases/products/input/schemamatching/Product_Type_Taxonomy.csv
  ```
  Storage_Interface binds `bus_type / interface_type /
  storage_connection_type`; GPU_Memory binds `memory_type`;
  Product_Type binds `product_type`. CSV content drafted in
  plan_revision_step4h_knob_review.md follow-up. `form_factor`
  intentionally has no taxonomy (mixed drive-bay + M.2 codes,
  no clean hierarchy).
- [x] **Block H — `cleanup_rules`** — kept all existing (music
  lastfm.name prefix strip; games dbpedia.title parenthetical
  strip; companies forbes.region footnote strip).
- [x] **Block I — Products `numeric_attributes` block** — added:
  ```yaml
  numeric_attributes:
    products_1: &products_numeric
      price: continuous
      vram_gb: continuous
      storage_gb: continuous
      read_speed_mb_s: continuous
      write_speed_mb_s: continuous
    products_2..4: alias
  numeric_jitter_max_relative: 0.02
  ```
- [x] **K1+K6 cleanup overlap correction** — **dropping the 2 K1
  `baseline_above_target_rules` additions** I proposed earlier
  (music `lastfm.name → musicbrainz.name shortest`; games
  `dbpedia.title → metacritic.game_title shortest`). K6's surgical
  regex cleanup_rules already cover these patterns and are more
  precise. Music K1 keeps the original 1 rule
  (`musicbrainz.release-country` long → discogs short); games K1
  keeps the original 1 rule (`dbpedia.system` → sales `hw`
  platform). **Supersedes the K1 sign-off entry above.**

### Cross-knob expansion — products categorical + numeric extensions (signed off 2026-05-27)

Step 4h initial per-domain audit operated against the existing
`attribute_classes` blocks per knob, which honored the
[config/domains/products.yaml](../usecases_synthetic/config/domains/products.yaml#L65-L66)
design comment excluding the 12 extension columns. The K6 walkthrough
surfaced that several extensions (high-coverage categoricals + the K5
rate columns) have meaningful difficulty-dial potential. The user
2026-05-27 authorised a full cross-knob expansion bringing 7
extensions into scope:

**5 categorical extensions** (taxonomy_walkable):
- `bus_type` (75% coverage)
- `interface_type` (52%)
- `memory_type` (26%)
- `storage_connection_type` (50%)
- `form_factor` (50%)

**2 numeric extensions** (already added under K5 rate-class expansion):
- `read_speed_mb_s` (18%)
- `write_speed_mb_s` (15%)

**Skipped**:
- `color` (8-15%, mixed-locale chaos)
- `width_mm / length_mm / height_mm / weight_g` (3-6% — too sparse
  for meaningful dial movement)

**Affected knobs**:
- [config/domains/products.yaml](../usecases_synthetic/config/domains/products.yaml)
  `attribute_classes` — add 7 entries (all `secondary`).
- [config/knob_03_drop/products.yaml](../usecases_synthetic/config/knob_03_drop/products.yaml)
  `attribute_classes` + `attribute_mapping` + extended
  `per_source_attribute_overrides` (see K3 sign-off above for the
  full ceiling table covering all 11 sparse attrs).
- [config/knob_05_format/products.yaml](../usecases_synthetic/config/knob_05_format/products.yaml)
  already includes the 2 numerics under K5 rate-class sign-off; no
  further change.
- [config/knob_06_noise/products.yaml](../usecases_synthetic/config/knob_06_noise/products.yaml)
  `attribute_classes` + `attribute_mapping` + `numeric_attributes`
  + 3 new `taxonomies` blocks + `operator_mix` adds `taxonomy_walk`
  at medium + hard.
- [config/knob_08_naming/products.yaml](../usecases_synthetic/config/knob_08_naming/products.yaml)
  `sm_mapping` + `rename_table` adds 7 attrs × 4 rungs each (will
  draft during K8 walkthrough).
- [config/knob_10_reliability/products.yaml](../usecases_synthetic/config/knob_10_reliability/products.yaml)
  `attribute_mapping` + `attribute_targets` adds 7 distribution
  blocks (will draft during K10 walkthrough).
- K4 (coverage skew): no per-attribute scope change (entity-level
  operation only).

### K8 — signed off 2026-05-27 (overall walkthrough)

- [x] **Block A — 4-rung escalation + C5 anonymize-at-hard** — kept.
  Each of the 4 active domains has exactly one source hitting
  `anonymized` at hard (lastfm / sales / fullcontact / products_3).
  Step-2 audit confirmed `naming_intensity` PASS on music (0/24/48)
  and games (0/3/43).
- [x] **Block B — per-source easy→medium noop on ONE source per
  domain** — kept. discogs (music) / metacritic (games) / forbes
  (companies) / products_1 (products) stay at `descriptive` through
  medium. Intentional — SM stage benefits from having at least one
  source retaining canonical naming at medium; cross-source aggregate
  is still monotone via other sources.
- [x] **Block C — `sm_mapping` extension for products** — add 7 new
  entries for the cross-knob expansion (bus_type / interface_type /
  memory_type / storage_connection_type / form_factor /
  read_speed_mb_s / write_speed_mb_s). Anchor-based already; landing
  one block.
- [x] **Block D — products `rename_table` additions** — 7 attrs ×
  4 rungs (descriptive / abbreviated / cryptic / anonymized) using
  Attribute_14..20:
  ```yaml
  bus_type:                 [bus_type, bus_t, bt, Attribute_14]
  interface_type:           [interface_type, iface_t, it, Attribute_15]
  memory_type:              [memory_type, mem_t, mt, Attribute_16]
  storage_connection_type:  [storage_connection_type, stor_conn, sc, Attribute_17]
  form_factor:              [form_factor, form_f, ff, Attribute_18]
  read_speed_mb_s:          [read_speed_mb_s, rd_speed_mbs, rs, Attribute_19]
  write_speed_mb_s:         [write_speed_mb_s, wr_speed_mbs, ws, Attribute_20]
  ```
- [x] **Refactor products `rename_table` to YAML anchors** — collapse
  4 verbose blocks into one anchor + 3 aliases. All 4 products
  sources share the identical rename ladder.
- [x] **Block E — id-column safety** — `id` columns remain omitted
  from rename_table across all domains; apply-time guard preserved.

### K10 — signed off 2026-05-27 (overall walkthrough)

- [x] **Block A — `attribute_targets` shape** — kept per-domain.
  Monotone decreasing winner share across levels (easy ~0.85-0.90,
  medium ~0.65-0.70, hard ~0.40-0.50 convergence). K10 is the
  cleanest bidirectional dial — every parameter movement is strictly
  monotone across all 3 levels.
- [x] **Block B — `compromise_rate_per_level` (0/0.05/0.15) +
  `corr_strength_per_level` (0/0.20/0.50)** — kept.
- [x] **Block C — `concentration_cap: 0.99`** — kept.
- [x] **Block D — Companies dbpedia-as-winner for country + founded**
  — **kept gold-alignment** (current 0.85-0.90 at easy). Add YAML
  comment noting the memory override per `feedback_dbpedia_noise_profile`:
  "country/founded default to dbpedia because the fusion gold was
  authored from dbpedia values for these encyclopedic facts;
  source-quality prior overridden by gold-alignment."
- [x] **Block E — Products anchor bias** (`products_1: 0.55 → 0.40 → 0.25`)
  — kept. Products has no a-priori winner; arbitrary anchor bias is
  the only way to produce a real K10 dial when sources are
  quality-symmetric.
- [x] **Block F — Products extension expansion** — add 10 new
  `attribute_targets` entries to products K10 using the existing
  `*target_distribution` anchor:
  ```yaml
  attribute_mapping (per source, anchor):
    priceCurrency, model_number, title_description,        # 3 R1 columns missing from original
    bus_type, interface_type, memory_type,                  # categorical extensions
    storage_connection_type, form_factor,
    read_speed_mb_s, write_speed_mb_s                       # numeric extensions

  attribute_targets:
    # ...existing 9...
    priceCurrency: *target_distribution
    model_number: *target_distribution
    title_description: *target_distribution
    bus_type: *target_distribution
    interface_type: *target_distribution
    memory_type: *target_distribution
    storage_connection_type: *target_distribution
    form_factor: *target_distribution
    read_speed_mb_s: *target_distribution
    write_speed_mb_s: *target_distribution
  ```
  Products K10 coverage: 9 → 19 attributes.

### K2 — drop-corner + non-corner refill (new step 4i, signed off 2026-05-27)

- [x] **Build the missing "drop corner-touching entities" operator
  + LLM-synthetic non-corner refill** — captured as new
  [step 4i in plan_revision.md](plan_revision.md). K2 currently
  noops at `baseline > target` because the existing `drop_high_density`
  operator goes the wrong direction. Step 4i adds a real drop-corner
  operator (greedy on expected corner-pair reduction, skipping
  protected + last-collision-group-members) + 1-for-1 refill via a
  new LLM prompt `non_corner_v1.txt` (separate cache namespace
  `cache/knob_02_non_corner/<domain>/`). 2-3 days code + tests; blocks
  step 5.

### K2 per-domain — still pending walkthrough

- [ ] Per-domain `target_corner_case_ratio` per level (music / games
      / companies / products — pre-existing REVISE items in the
      domain sections above, plus the bidirectional-dial discussion
      from K2 overall).
- [ ] Per-domain `metric_top_k` (music 30 / games 50 / companies 20
      / products 20 — review needed).
- [ ] Products K2 canonical_schema gap — add `priceCurrency` +
      `title_description`.
- [ ] Products K2 per-level extras
      (`removal_fraction_cap` / `interpolation_count` /
      `placement_split` per-level).

### Other knobs (walkthrough pending)

- [ ] K3 / K4 / K5 / K6 / K8 / K10 overall walkthroughs.
- [ ] Companies K10 country/founded — keep dbpedia winner (add
      comment) OR flip away from dbpedia.
- [ ] Products K10 attribute_mapping gap — add priceCurrency +
      model_number.
- [ ] Products K3 sparse-attribute override — apply or skip.
- [ ] Fold R1/R2 status corrections into plan_revision.md.

After all per-knob walkthroughs land + YAML edits + step 4i code lands
+ test run, step 5 (products redesign rerun) is unblocked. Step 6
(music + games reruns) follows.
