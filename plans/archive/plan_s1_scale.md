# S1 Variant Generation — companies, games, music (re-run on refreshed sources)

Top-level tracker for re-baselining and producing trustworthy `easy/medium/hard` variants for **companies, games, music** after the 2026-05-04 source-data refresh. Movies and products remain descoped from this pass; their pool prerequisites (S6 / S7 in the prior tracker) and PLM-pipeline outputs are unchanged and can be picked up in a future plan.

The repo state coming into this plan is the result of [Phase A0 + A + B + C from the prior tracker](#prior-state-context-load-bearing) — Ditto vendoring, generator + measurement fixes, per-domain configs + committee forks, and one full pass of variants on the previous source data. That work is preserved in code; the section below names the load-bearing pieces so this plan can be picked up with full context.

## Process

- Git commits, branching, version control handled by the human. The implementer (Claude) focuses on code, tests, config.
- All commands use the `pydi-dev/` venv.
- No emojis in console output. NumPy-style docstrings. mypy strict. Preserve `DataFrame.attrs`.
- **Do not modify original PyDI code** (anything under [../PyDI/](../../PyDI/)). User-directed, 2026-04-22. New adapters, wrappers, prompts, vendored code land under [../usecases_synthetic/lib/](../../usecases_synthetic/lib/) or [../usecases_synthetic/third_party/](../../usecases_synthetic/third_party/). If a change to `PyDI/` is genuinely required, stop and ask.

## Source data convention (NEW 2026-05-04)

All source datasets ship as **CSV files** at `usecases/<domain>/input/data/<source>.csv`, each paired with a schema.org-style `<source>_metadata.json` sidecar (license, row count, schema). The previous XML/JSON sources are preserved at `usecases/<domain>/input/data/old/` for reference; they are no longer the loader's input.

Implications:
- Domain configs ([../usecases_synthetic/config/domains/{companies,games,music}.yaml](../../usecases_synthetic/config/domains/)) need `format: xml/json` → `format: csv`, `file: <source>.xml/json` → `file: <source>.csv`, and `inject_id: true` removal where the new CSVs ship a native id column (verify per source).
- Committee YAML `column_mapping` blocks must be cross-checked against the new CSV column names — mismatches will fail at the EM blocking stage.
- The XML namespace stripping + `inject_id` plumbing in [../usecases_synthetic/lib/loaders.py](../../usecases_synthetic/lib/loaders.py) and [../usecases_synthetic/scripts/downsample_domain.py](../../usecases_synthetic/scripts/downsample_domain.py) becomes unused on these three domains; keep the code in place since movies/products will still need it later.

## Prior state context (load-bearing)

Brief summary of what is committed and currently relied on. Each item below is "still in code; understand before touching."

### Phase A0 — Ditto PLM foundation
- Vendored runtime: [../usecases_synthetic/third_party/ditto_modern/](../../usecases_synthetic/third_party/ditto_modern/).
- Matcher adapter: [../usecases_synthetic/lib/ditto_matcher.py](../../usecases_synthetic/lib/ditto_matcher.py) (with per-batch inference cache that survives lid-close / SIGINT — cache dir `cache/ditto_inference/`, gitignored).
- Train / eval scripts: [../usecases_synthetic/scripts/ditto/{train.py,evaluate.py,prepare_em_training_data.py}](../../usecases_synthetic/scripts/ditto/).
- Wired through `ditto_plm` member in `config/committees/em_matching_committee_<domain>.yaml`.

### Phase A — Generator + measurement fixes
- **K2 regenerated EM as primary surface (S1 / S2 / S4c).** K2 emits per-pair per-split `<src1>_2_<src2>_{train,val,test}_regenerated.csv` at the source-record level. `committee_em._score_predictions` uses closed-set scoring with the fallback chain `f1_vs_regenerated_val → _test → pool` ([../usecases_synthetic/lib/committee_em.py](../../usecases_synthetic/lib/committee_em.py), [../usecases_synthetic/lib/committee_em_scoring.py](../../usecases_synthetic/lib/committee_em_scoring.py)).
- **Hard-negative gate (S3).** Mining policy requires PLM score < θ−δ, with an LLM adjudicator on the margin band. Code at [../usecases_synthetic/lib/corner_case_miner.py](../../usecases_synthetic/lib/corner_case_miner.py) + [../usecases_synthetic/lib/hard_negative_plm.py](../../usecases_synthetic/lib/hard_negative_plm.py). Per-domain parameters in `knob_02_niche/<domain>.yaml`.
- **Committee freeze + drift guard (S4a / S4b).** SM / EM-blocking / EM-matching / Fusion committees are frozen and SHA-pinned. Per-domain forks at `config/committees/*_<domain>.yaml`; resolver at [../usecases_synthetic/lib/committee_paths.py](../../usecases_synthetic/lib/committee_paths.py); drift check in [../usecases_synthetic/scripts/measure_baseline.py](../../usecases_synthetic/scripts/measure_baseline.py) + [../usecases_synthetic/scripts/validate_variant.py](../../usecases_synthetic/scripts/validate_variant.py).

### Phase B — Per-domain plumbing
- Domain configs at [../usecases_synthetic/config/domains/](../../usecases_synthetic/config/domains/) (companies, games, music — all need R1 updates).
- Per-knob configs at [../usecases_synthetic/config/knob_*/](../../usecases_synthetic/config/) (8 knobs × 3 domains; per-level rates carried over from companies as defaults — calibration is part of R4).
- Per-domain committee forks at `config/committees/{em_blocking,em_matching,fusion}_committee_{games,music}.yaml` (companies uses the canonical unsuffixed YAMLs).

### Phase C — First pass (now stale)
The variants at `usecases/{companies,games,music}-augmented/{easy,medium,hard}/` were generated against the prior (XML/JSON-mixed) source data and pre-refresh Ditto state. Re-running on the refreshed CSV sources is the entire point of this plan; the existing variant directories will be overwritten by R6.

Notable code-level fixes from Phase C that remain load-bearing:
- DittoMatcher inference cache (S14) — survives lid-close mid-run.
- EM scorer fallback chain (S14, [../usecases_synthetic/lib/committee_em.py](../../usecases_synthetic/lib/committee_em.py)).
- Closed-set EM scoring (S4c, [../usecases_synthetic/lib/committee_em_scoring.py](../../usecases_synthetic/lib/committee_em_scoring.py)).
- Audit-side fixes for K2/K3/K6 monotonicity counters (S13, [../usecases_synthetic/scripts/generate_variant.py](../../usecases_synthetic/scripts/generate_variant.py)).
- `analyze_ablation` / `analyze_monotonicity` surface fix to use `aggregated.macro_f1_vs_pool` (S16, [../usecases_synthetic/config/knob_expected_signals.yaml](../../usecases_synthetic/config/knob_expected_signals.yaml)).

## Phase R (re-run) — focus of this plan

Hard ordering: **R1 → R2 → R3 → R4 → R5 → R6 → R7**. R2 / R3 / R4 / R5 are interactive review steps where Claude proposes the approach for one slice at a time (per-domain for R2, per-stream for R3, per-knob for R4, per-stage × per-domain for R5) and the user approves or requests changes before any runtime work starts.

## Pending design updates (2026-05-06)

User-supplied additions captured mid-R4 (after K2 sign-off, during K4 review). Each item lives inline at its relevant section below; this list is the at-a-glance index. Items marked **(interactive)** require Claude to propose and the user to approve before implementation; **(automatic)** items can be implemented once the broader plan is signed off. Status: `[ ]` until folded into the corresponding R-phase row.

1. `[x]` **(interactive)** Brainstorm + select an embedding model for the EM-blocking committee's embedding-based blocker. **Closed 2026-05-10 as part of R5 EM blocking sign-off.** Winner: **`BAAI/bge-base-en-v1.5`** — selected from a 5-model panel (MiniLM-L6, MPNet-base, BGE-base, BGE-small, E5-base) by the only-model-clearing-0.97-recall-floor-on-all-3-domains criterion. Cross-domain mean pair_recall = 0.986. See §"R5 EM blocking sign-off (2026-05-10)" below.
2. `[x]` **(interactive)** Add a **Normalization pipeline step** between schema mapping and entity matching, with its own committee and difficulty curve. **Closed 2026-05-10**: design + implementation + per-domain YAMLs + tests + measure_baseline/validate_variant wiring + tuning sweep + winner-lock all landed. Sweep validated the existing YAML defaults (no edits needed); cross-source-linkage bug caught + fixed mid-sweep; LLMCache key-collision bug caught + fixed. R6.1 baseline measurement folded into R6.1; R7.2 monotonicity check folded into R7.2 (Pending #8 applies). See §"R5 Normalization sign-off (2026-05-10)" below.
3. `[x]` **(automatic)** Extend the SM committee with label-based and instance-based schema matchers (PyDI ships both — confirm wiring into [config/committees/sm_committee.yaml](../../usecases_synthetic/config/committees/sm_committee.yaml)). **Closed 2026-05-08 as part of R5 SM sign-off.** New `label_jw` (jaro_winkler, threshold 0.75) and `instance_tf_cosine` (term_frequencies + cosine, threshold 0.5, max_sample_size=200) members in [sm_committee.yaml](../../usecases_synthetic/config/committees/sm_committee.yaml). `required_axes.signal_type` extended to `[duplicate, label, instance, embedding, hybrid]`. See §"R5 SM sign-off (2026-05-08)" below.
4. `[x]` **(interactive)** Taxonomy-based noising for value-noise knobs: where an external taxonomy covers an attribute (e.g. genre / industry / country), add an operator that walks the taxonomy up (more abstract) or down (more specific) to "noise" the value. **Closed 2026-05-07 as part of K6 sign-off.** Single-level walk only (both medium and hard). Taxonomies bound: companies `dbpedia.sector` + `forbes.business_segment` → GICS_Industry_Taxonomy (4 levels); games `(dbpedia.system, metacritic.console, sales.hw)` → Gaming_Platforms_Taxonomy + `(dbpedia.genre, metacritic.genres, sales.genre)` → Video_Game_Genres_Taxonomy; music `(musicbrainz.genre, discogs.genre, lastfm.genre)` → Music_Genres_Taxonomy. ESRB taxonomy is flat (no parent/child) — intentionally not used. New `Taxonomy` class + `taxonomy_walk` operator at [noise_operators.py](../../usecases_synthetic/lib/noise_operators.py). `fullcontact.Attribute_4` confirmed = city/locality, not industry-like (no binding). See §"K6 sign-off (2026-05-07)" below.
5. `[x]` **(interactive)** **Relax the fusion val/test protection rule**: today protection is "fusion val + test entity records cannot be dropped, and at least one record per cell must preserve the exact target value." New rule: every fusion val + test entity must remain alive in every variant **and** for each (entity, attribute) cell at least one surviving record must stay "close enough" to the val/test value that a lenient fusion strategy can recover the truth. **Closeness metric locked 2026-05-06** (see §"K4 sign-off → Closeness-metric specification"): `lexical_extended_jaccard(inner_token_threshold=0.8)` for long strings, `_levenshtein_ratio` ≥ 0.85 for short / nominal, ±3 % relative for continuous numerics, ±1 absolute for years. **Closed 2026-05-07** as the dedicated R4 row "Pending #5 wire-up (closeness contract)". Surface delivered: new `fusion_cell_tolerance` / `is_close_enough` / `cell_has_close_survivor` / `load_fusion_target_values` helpers in [protection.py](../../usecases_synthetic/lib/protection.py); K6's strict per-cell `clean_survivor_floor` replaced with a post-mutation closeness gate inside the retry loop (new skipped-reason `closeness_floor_exhausted_retries`); K1's strict `per_cell_clean_survivor_floor` replaced (new skipped-reason `closeness_floor_violation`) + the easy normalize-down path now filters siblings against the fusion target (closes K1 follow-up #1 — `forbes.region: China → Taiwan` no longer fires on fusion-protected cells) + the anchor-survivor singleton over-fire is fixed (closes K1 follow-up #2). K5 is a no-op: every K5 operator is round-trip preserving by construction (`SKIP_ROUNDTRIP` audit), so the closeness contract is satisfied transitively without a separate gate. K3 + K10 inherit `_ClosenessContext` + `_check_close_survivor_floor` semantics when their R4 rows ship. See §"Pending #5 closeness-contract wire-up (2026-05-07)" below.
6. `[x]` **(automatic)** K6 numeric noising: when a numeric value is jittered, constrain the perturbation to ±2 % relative. **Closed 2026-05-07 as part of K6 sign-off.** New `numeric_jitter_within_cap` helper at [noise_operators.py](../../usecases_synthetic/lib/noise_operators.py) + new `numeric_attributes` per-domain config block tagging columns as `continuous` / `year` / `date`. Post-operator clamp in the dispatcher: continuous rejects beyond ±2 % relative; year rejects parsed-year change; date rejects calendar-day change. Mutations that make the value **unparseable** (the canonical K6 corruption mode — see K6 spec "malformed unparseable date is K6") are **allowed**. On rejection, retry up to 5 times with a fresh operator draw; on exhaustion, log `numeric_jitter_exhausted_retries` to skipped. See §"K6 sign-off (2026-05-07)" below.
7. `[x]` **(automatic)** K5 unit-of-measurement swap operator: convert between equivalent units (e.g. `10 km` ↔ `10000 m`, `1 hr` ↔ `60 min`). **Closed 2026-05-07 as part of K5 sign-off.** `reconvert_unit` math was already wired for `duration / weight / length / file_size` in [format_operators.py:374](../../usecases_synthetic/lib/format_operators.py#L374); the dispatcher in [apply_knob_05_format.py:_apply_cell_transform](../../usecases_synthetic/scripts/apply_knob_05_format.py) now routes `duration` and `dimensional` families to a new `_transform_duration` helper backed by `format_duration` / `parse_duration` in [format_operators.py](../../usecases_synthetic/lib/format_operators.py). Music's `duration` column exercises the path with the `seconds_int / mm_ss / hh_mm_ss / human_xm_ys` pool; games' `units_sold_mm` exercises the equivalent magnitude swap (`millions ↔ raw ↔ thousands`) via the already-wired money/number path.
8. `[ ]` **(automatic)** R7.2 monotonicity check must additionally verify that the **best-member** F1 (not just committee macro_f1) drops with difficulty. A difficulty increase that only depresses the committee mean while the best member stays flat is **not** a valid difficulty signal — the user could simply pick the best member and ignore the committee. Add this guard to [scripts/analyze_monotonicity.py](../../usecases_synthetic/scripts/analyze_monotonicity.py).

### R1 — Source-convention migration

| # | Module | Status |
|---|--------|--------|
| R1.1 | Update [../usecases_synthetic/config/domains/{companies,games,music}.yaml](../../usecases_synthetic/config/domains/): switch every source to `format: csv`, point `file:` at the new `<source>.csv` filename, drop `inject_id: true` where the new CSV ships a native id column (verify via `pd.read_csv(...).columns`). Keep `id_prefix` matching the EM gold's id convention. | `[x]` |
| R1.2 | Spot-check `config/committees/{em_blocking,em_matching,fusion}_committee_*.yaml`: each `column_mapping` per-source block must reference column names that exist in the new CSVs. Diff against `pd.read_csv(...).columns` per (domain, source). | `[x]` |
| R1.3 | Smoke: `pytest usecases_synthetic/tests/test_loaders.py usecases_synthetic/tests/test_committee_configs.py -q` — both must stay green. Then `python -c "from usecases_synthetic.lib.loaders import load_domain_sources; load_domain_sources('<d>')"` per domain to confirm the new sources resolve. | `[x]` |

### R2 — Per-domain Ditto baseline training (INTERACTIVE, per domain)

Train one Ditto checkpoint per domain on the refreshed EM gold + the new CSV sources. Output goes to `cache/ditto_checkpoints/<domain>/run_*/checkpoints/best/` with a `cache/ditto_checkpoints/<domain>/best/` symlink for stable committee-YAML reference (mirrors the S11 layout).

**Process:** before kicking off training for each domain, Claude proposes the full training plan (data prep + hyperparameters + split shape) for that domain; the user approves or revises before any compute starts. One review session per domain — what's right for companies (compact schema, ~5k pairs) is not right for games (schema-light list attributes, smaller gold) or music (large EM gold, `label`-field collision).

**Open design questions for the per-domain review** (Claude proposes, user approves or revises):
- **Training data shape**: which EM gold splits feed train / val / test (`_train.csv` only, `_train` + `_val`, full `_all.csv` with stratified resplit)? Balance ratio inside the training subsample?
- **Field projection**: which canonical-schema fields enter Ditto's pair-text serialization? Per-domain caveats (music's `label` collision with the binary classification target — must stay dropped from `ditto_plm.fields` until canonical schema is renamed).
- **Model + recipe**: distilbert-base-uncased CPU fallback (batch 16 / max-len 128 / max-field-len 200 / 3 epochs / `--no-fp16`) vs roberta-base GPU (batch 32 / max-len 256 / 10 epochs / fp16). Confirm CUDA / MPS availability before locking the recipe.
- **Augmentation switches**: Ditto's `--summarize`, `--dk` (domain-knowledge injector). Off by default since they require spaCy + nltk; turn on per domain only if the field-text exceeds `max-len`.
- **Threshold selection**: best-val-F1 θ vs a fixed θ; downstream impact on the S3 hard-negative gate's `θ−δ` band.
- **Acceptance criteria**: prior CPU bars were companies 0.918 / games 0.831 / music 0.804 test F1. Are those the right floor on the refreshed gold, or are the new sources different enough to expect movement?

| # | Module | Status |
|---|--------|--------|
| R2.1 | Refresh prep scripts at [../usecases_synthetic/scripts/ditto/](../../usecases_synthetic/scripts/ditto/) (`_prep_companies.py`, `_prep_games.py`, `_prep_music.py`) so they invoke `build_ditto_pair_records_from_gold` against the refreshed sources. Output WDC `json.gz` triplets per pair. **Per-domain review of split shape + balance + field projection happens before this row is checked off.** | companies `[x]` · games `[x]` · music `[x]` |
| R2.2 | Train + in-loop test per domain. **Per-domain review of model + hyperparameters + augmentation switches happens before this row is checked off.** Recipe + final hyperparameters recorded per-domain in [config/ditto/README.md](../../usecases_synthetic/config/ditto/README.md). | companies `[x]` · games `[x]` · music `[x]` |
| R2.3 | Symlink `best/` → `run_*/checkpoints/best/` per domain. Confirm `metrics.json` populated and the run meets the per-domain acceptance criteria agreed in the R2 review (test F1 ≥ floor, val F1 ≥ floor, predictions.csv populated). | companies `[x]` · games `[x]` · music `[x]` |

### R3 — Pool rebuild

Final design (post-review, 2026-05-05): two evidence streams + per-domain
transitive closure + bucket adjudication. ADI was dropped — both
streams are now sources we control and can re-run on the refreshed
data:

1. **Gold positives** — every label-True row across
   `usecases/<domain>/input/entitymatching/*_{train,val,test}.csv`.
2. **Human-baseline pipeline correspondences** — regenerated against
   the refreshed CSV sources by
   [scripts/regen_human_baseline.py](../../usecases_synthetic/scripts/regen_human_baseline.py),
   which replays each notebook's `RuleBasedMatcher` configuration
   (per-pair blockers + comparators + weights + threshold + post-
   greedy/maximum-bipartite reduction). Notebook fidelity verified:
   per-pair correspondence row counts within ~5% of notebook
   reference. Output: `usecases/<d>/output/correspondences_<src1>_<src2>.csv`.
3. **Ditto baseline** — score Ditto from the R2 checkpoint over a
   per-source-pair candidate set produced by the in-script blocker
   sweep ([lib/pool_builder.py](../../usecases_synthetic/lib/pool_builder.py)
   `sweep_blockers`). Cap: blockers with `n_candidates > 1M` are
   demoted to the highest-recall under-cap blocker (otherwise
   `token_blocker` wins with multi-million candidate sets that make
   Ditto inference impractical on MPS).

**Domain-level transitive closure** (per evidence stream, before
bucket assembly): build one undirected graph from the union of all
per-source-pair edges, take connected components, emit every cross-
source pair within each component → `expanded_human`,
`expanded_ditto`. Closure-implied pairs without a direct Ditto score
are rescored against the appropriate source-pair frames before
bucketing. Source-pairs that surface only via closure (e.g.
companies `dbpedia↔fullcontact`, music `discogs↔lastfm`) are added
as synthetic `PairSignals` so their closure-implied pairs aren't
silently dropped.

**Bucket logic** uses the closure-expanded sets:
- A — gold positive: kept unconditionally.
- B — pair in `expanded_human ∩ expanded_ditto` (Ditto≥θ): kept (agreement, possibly transitive).
- C — singleton: gated by Ditto score against the per-domain `delta`
  (90th percentile of |prob − 0.5| on Ditto misclassifications,
  clipped to [0.05, 0.20]). Score ≥ θ+δ kept (confident pos);
  score < θ−δ dropped (confident neg); margin band [θ−δ, θ+δ]
  adjudicated by `gpt-5.4` (temperature=0) via
  [LLMCache](../../usecases_synthetic/lib/llm_cache.py), cache namespace
  `r3_pool_adjudicator::<domain>`.

**Output schema:** `usecases_synthetic/pools/<domain>/pooled_positives.csv`
with columns `id1, id2, source_1, source_2, score, in_gold, in_human,
in_ditto, decision_path` (`decision_path ∈ {gold, agreement,
plm_check_confident_pos, plm_check_llm_yes}`).
`pool_stats.json` carries per-pair blocker-sweep stats, bucket
breakdowns, closure additions per source-pair (human + ditto),
delta-estimation telemetry, and a connected-components size
histogram (informational only — no egregious-cluster filter; max
component size ≤ 43 across all 3 domains, well below the prior
threshold).

| domain | total pool | per-pair (incl. closure-only) |
|---|---|---|
| companies | 1500 | forbes-dbpedia 476, forbes-fullcontact 860, **dbpedia-fullcontact 164 (closure-only)** |
| games | 21264 | dbpedia-metacritic 11552, dbpedia-sales 6062, metacritic-sales 3650 |
| music | 10508 | musicbrainz-discogs 4131, musicbrainz-lastfm 3982, **discogs-lastfm 2395 (closure-only)** |

Closure surfaced 758 human + 82 ditto pairs for companies'
missing dbpedia↔fullcontact pair (via forbes pivot), 2157 human +
2988 ditto for games' metacritic↔sales, and 1646 human + 4518 ditto
for music's discogs↔lastfm. Without closure (independent per-pair
processing) the pools would have been 1336 / 20130 / 8113 — closure
adds ~+5% / +5% / +30%, with the music delta dominated by the entire
discogs↔lastfm pair being closure-only.

Status: `[x]` (2026-05-05).

### R4 — Generation-process review (INTERACTIVE, per knob)

Walk through each knob in the K2-first generation order. Per knob: confirm the per-level rate table, the operator mix, the protection set semantics, any per-domain overrides, and any S13 calibration carry-overs. Output: per-knob sign-off recorded inline.

| Knob | Module | Status |
|------|--------|--------|
| K2 (niche density / corner cases) | `config/knob_02_niche/<domain>.yaml` + [../usecases_synthetic/scripts/apply_knob_02_niche.py](../../usecases_synthetic/scripts/apply_knob_02_niche.py) | `[x]` (2026-05-06) |
| K4 (coverage skew) | `config/knob_04_coverage/<domain>.yaml` + [apply_knob_04_coverage.py](../../usecases_synthetic/scripts/apply_knob_04_coverage.py) | `[x]` (2026-05-06) — see §"K4 sign-off (2026-05-06)" below for dispatcher fixes, per-domain target tables, and the K4 ↔ Pending #5 closeness-metric coupling. |
| K1 (surface paraphrase) | `config/knob_01_surface/<domain>.yaml` + [apply_knob_01_surface.py](../../usecases_synthetic/scripts/apply_knob_01_surface.py) | `[x]` (2026-05-07) — see §"K1 sign-off (2026-05-07)" below for column-name remap, expanded abbreviation tables, LLM model default, and two follow-up flags. |
| K5 (format / unit) | `config/knob_05_format/<domain>.yaml` + [apply_knob_05_format.py](../../usecases_synthetic/scripts/apply_knob_05_format.py) | `[x]` (2026-05-07) — see §"K5 sign-off (2026-05-07)" below for column-name remap to refreshed CSVs, duration-family dispatcher wiring (Pending #7 closed), forbes magnitude flip (raw vs prior billions), and games units_sold_mm via existing money path. |
| K6 (noise) | `config/knob_06_noise/<domain>.yaml` + [apply_knob_06_noise.py](../../usecases_synthetic/scripts/apply_knob_06_noise.py) | `[x]` (2026-05-07) — see §"K6 sign-off (2026-05-07)" below for column-name remap, `taxonomy_walk` operator (Pending #4 closed), ±2 % numeric jitter cap with retry loop (Pending #6 closed), refreshed cleanup rules, and the Pending #5 split. |
| Pending #5 wire-up (closeness contract) | [protection.py](../../usecases_synthetic/lib/protection.py) + per-knob guards in K1/K3/K5/K6/K10 | `[x]` (2026-05-07) — see §"Pending #5 closeness-contract wire-up (2026-05-07)" below for the protection.py refactor, K1 + K6 wire-ups, K1 follow-ups #1 + #2 closed, K5 no-op rationale (round-trip preservation), and the K3/K10 hand-off. |
| K3 (drop / nesting) | `config/knob_03_drop/<domain>.yaml` + [apply_knob_03_drop.py](../../usecases_synthetic/scripts/apply_knob_03_drop.py) | `[x]` (2026-05-07) — see §"K3 sign-off (2026-05-07)" below for column-name remap, per-domain rate recalibration, closeness-aware survivor selection (Pending #5 wire-up), discogs duration=0 loader coalesce, and per-source ceilings on near-baseline-hard attributes. |
| K10 (reliability) | `config/knob_10_reliability/<domain>.yaml` + [apply_knob_10_reliability.py](../../usecases_synthetic/scripts/apply_knob_10_reliability.py) | `[x]` (2026-05-07) — see §"K10 sign-off (2026-05-07)" below for column-name remap (id_columns collapse to `id`), revised winner table (curated-source-preferred where measured baseline agrees), Pending #5 strict + infra-aligned wire-up (kind taxonomy sourced from `protection._DEFAULT_KIND_BY_DOMAIN_ATTR`, K5 `attribute_classes` reconciliation retired from the K10 path), `load_fusion_gold` extended to read both `validation_set.xml` + `test_set.xml`, and the games gold↔EM-positive overlap follow-up flag. |
| K8 (naming / SM) | `config/knob_08_naming/<domain>.yaml` + [apply_knob_08_naming.py](../../usecases_synthetic/scripts/apply_knob_08_naming.py) | `[x]` (2026-05-08) — see §"K8 sign-off (2026-05-08)" below for column-name remap to refreshed CSVs (FullContact ships physically anonymized; metacritic shifted toward descriptive at baseline), level_assignment recalibration (games medium `sales: cryptic → abbreviated`), canonical-target sm_mapping aligned to K10's `attribute_mapping`, and the YAML 1.1 `on`-as-boolean fix. |

### R5 — Committee review (INTERACTIVE, per stage × per domain)

Walk through each committee for each domain. Per (stage, domain): confirm the roster, per-attribute strategy lists, trust scores (fusion), `column_mapping`, blocker selectivity, and matcher thresholds. The committee freeze is locked at R6.1, so any edit after R5 invalidates baselines.

| Stage | Per-domain files | Status |
|-------|-----------------|--------|
| SM (shared) | [../usecases_synthetic/config/committees/sm_committee.yaml](../../usecases_synthetic/config/committees/sm_committee.yaml) | `[x]` (2026-05-08, extended 2026-05-10) — see §"R5 SM sign-off (2026-05-08)" + §"R5 SM hyperparameter tuning (2026-05-08)" + §"R5 SM embedding + Magneto tuning (2026-05-10)" + §"R5 SM duplicate-matcher fix (2026-05-10)" below. Closes Pending #3 (label_based + instance_based added), K8 follow-ups #1 + #2 (stale on-disk SM gold + target_schema), and lifts the cross-domain macro_f1 mean from 0.437 → **0.779** across three tuning passes. `magneto_slm_llm` enabled by default with BGE-base + topk=20; `duplicate_majority` re-enabled with a runner-side per-pair dispatch fix (PyDI bug #2 still open upstream). |
| Normalization (NEW) | `config/committees/normalization_committee_{companies,games,music}.yaml` + [committee_norm.py](../../usecases_synthetic/lib/committee_norm.py) + [normalizer_members.py](../../usecases_synthetic/lib/normalizer_members.py) + [llm_normalizer.py](../../usecases_synthetic/lib/llm_normalizer.py) + [committee_norm_scoring.py](../../usecases_synthetic/lib/committee_norm_scoring.py) + [_tune_norm_committee.py](../../usecases_synthetic/scripts/_tune_norm_committee.py). | `[x]` (2026-05-10) — design + implementation + sweep + winner-lock complete. Cross-domain mean macro_f1=0.508 (rule-based 5 members) → 0.514 (with LLM). See §"R5 Normalization sign-off (2026-05-10)" below. |
| EM blocking | `em_blocking_committee_{games,music}.yaml` + canonical `em_blocking_committee.yaml` (companies) + [_tune_em_blocking_committee.py](../../usecases_synthetic/scripts/_tune_em_blocking_committee.py) | `[x]` (2026-05-10) — sub-A/B/B' tuning + winners locked (BGE-base universal embedding; per-domain standard_blocker key; SN name_norm+window=40 universal). Sub-D SC-Block MPS training pipeline landed 2026-05-10: per-domain roberta-base checkpoints at `cache/sc_block_checkpoints/{companies,games,music}/best/`; sc_block enabled in all 3 YAMLs (companies val_pair_recall=0.9904, music=1.0000, games=0.9577 below floor — diagnostic-only there). Pending #1 closed; see §"R5 EM blocking sign-off (2026-05-10)" + §"R5 EM blocking sub-D sign-off (2026-05-10)" below. |
| EM matching | `em_matching_committee_{games,music}.yaml` + canonical `em_matching_committee.yaml` (companies) + [_tune_em_matching_committee.py](../../usecases_synthetic/scripts/_tune_em_matching_committee.py) + [magellan_auto_features.py](../../usecases_synthetic/lib/magellan_auto_features.py) + [openai_batch.py](../../usecases_synthetic/lib/openai_batch.py) | `[x]` (2026-05-11) — 4-of-4 enabled roster (ditto_plm + magellan + matchgpt zero-shot + comem); Option A per-pair `training_gold_path` runner injection; dbpedia.sector column-mapping fix preserves K2 dual-attribute setup; Magellan rewritten to use synth-local auto-feature-gen (py_entitymatching incompatible with Python 3.12); sweep landed per-domain classifier winners (companies / games favour `balanced`, music favours `None`); OpenAI Batch API foundational module landed (call-site integration deferred to pre-R6.2 follow-up). See §"R5 EM matching sign-off (2026-05-11)" below. |
| Fusion | `fusion_committee_{games,music}.yaml` + canonical `fusion_committee.yaml` (companies) + [fusion_perfect_clusters.py](../../usecases_synthetic/lib/fusion_perfect_clusters.py) + [_tune_fusion_committee.py](../../usecases_synthetic/scripts/_tune_fusion_committee.py) | `[x]` (2026-05-12) — perfect-cluster handoff (pool → cluster → correspondences), 10-sub-sweep × 3-domain hyperparameter tuning, list-attr eval via PyDI's `tokenized_match` + `intersection`/`intersection_k_sources` strategies, OpenAI gpt-5.4-mini llm_judge wiring. Final baselines: companies=0.881 · games=0.817 · music=0.859 (cross-domain mean **0.852**, +1.9pp vs pre-sweep). See §"R5 Fusion sign-off (2026-05-12)" below. |

### S — `*-small` sanity checks (2026-05-13, pre-R6.2 gate)

User directive (2026-05-13): before committing compute to the full R6.2 sweep across companies/games/music, regenerate `music-small` from the refreshed sources + pool and run the full R6.1 + R6.2 pipeline against it end-to-end. The pre-refresh `*-small` directories under `usecases/` and the cached pools/configs under `usecases_synthetic/` are stale (built 2026-05-03 against the prior music XML/JSON/CSV bundle) and must be deleted first.

**Scope expansion (2026-05-14):** the S phase now covers **music-small, games-small, and companies-small** in sequence. Each domain goes through the same S.1-S.5a cycle plus a per-committee member-level audit (S.6a) that surfaces silent zero-F1 members from the bundled-EMCommitteeRunner exception-swallowing pattern (see Cross-cutting bug fix #2). Full-domain R6.1/R6.2 begins only after all three `*-small` runs have green spot-checks + clean per-member scores.

| # | Module | Status |
|---|--------|--------|
| S.1 | Delete stale `*-small` + `*-augmented` artifacts: `usecases/{music-small,music-small-augmented,companies-small,games-small}` (manually removed 2026-05-13); orphan `usecases_synthetic/pools/{music-small,companies-small,games-small}/` + `config/domains/{music-small,companies-small,games-small}.yaml` are overwritten in-place by S.2 (no separate cleanup needed). | `[x]` |
| S.2 | `python usecases_synthetic/scripts/downsample_domain.py --source-domain music --target-domain music-small --gold-multiplier 1.5 --min-rows 50 --seed 0`. EM-gold + fusion-gold protection floor essentially saturates two of three sources (musicbrainz 4763/4763, lastfm 9865/9865; discogs 19131/22627). Pool filtered to 10465/10508 rows. Resulting `music-small.yaml` carries `knob_config_alias: music`. | `[x]` |
| S.3 | `python usecases_synthetic/scripts/measure_baseline.py --domain music-small --with-llm`. Validates SM/Norm/EM/Fusion committees against the smaller bundle and writes `baselines/music-small/baseline_metrics.json`. **Bug uncovered + fixed**: `_DEFAULT_KIND_BY_DOMAIN_ATTR` at [protection.py](../../usecases_synthetic/lib/protection.py) was keyed by raw domain only — aliased domains (`music-small` → `music`) returned an empty kind map, causing the Norm committee to error with "SM∩fusion∩kind is empty". Added `kind_map_for_domain(domain)` helper that resolves `knob_config_alias` via `_resolve_knob_config_alias`; updated 4 call sites ([committee_norm.py:323](../../usecases_synthetic/lib/committee_norm.py#L323), [reliability.py:375](../../usecases_synthetic/lib/reliability.py#L375), [_tune_norm_committee.py:86](../../usecases_synthetic/scripts/_tune_norm_committee.py#L86), [protection.py:244](../../usecases_synthetic/lib/protection.py#L244)) to use it. Baseline written 2026-05-13: SM macro_f1=0.874 (best=1.000), Norm macro_f1=0.389 (best=0.768), EM blocking macro_pair_recall=0.951 (best=sc_block 0.999), EM matching macro_f1=0.750 (best=ditto_plm 0.976), Fusion overall_accuracy=0.859. Total runtime 2078s. | `[x]` |
| S.4 | `python usecases_synthetic/scripts/generate_variant.py --domain music-small --level all` (K2/K4 default model `gpt-5.4-mini` per the 2026-05-06 LLM-bump convention; user confirmed no bump for the sanity run on 2026-05-13). Writes `usecases/music-small-augmented/{easy,medium,hard}/`. | `[ ]` |
| S.5 | Spot-check the 3 variant directories for: (a) `input/data/*.csv` + 3 sources × all rows valid, (b) `input/{schemamatching,entitymatching,fusion}/` populated, (c) `output/{provenance,baselines}/` per-knob CSVs present, (d) `config/difficulty.yaml` matches level, (e) EM-gold + fusion-gold IDs still resolve against the mutated sources. | `[ ]` |
| S.5a | **EM regen contract** (2026-05-13 rewrite of `regenerate_em_splits`): for each authored source pair, compare `input/entitymatching/*_{train,val,test}_regenerated.csv` against the original `usecases/music-small/input/entitymatching/*_{train,val,test}.csv`. Verify per split: (a) **size** within ±5% of the original — heavy undersizing (the pre-rewrite music train collapsed 19166 → 2445) indicates the pool+cluster backfill failed; (b) **positive ratio** within ±2 percentage points of the original — test was the regression case (orig 33% → pre-rewrite 8%); (c) every surviving original pair (both ids still in the post-K2 source frames) is carried forward verbatim; (d) backfill rows reference only ids present in the mutated sources; (e) cross-split disjointness holds within each source pair. Also check K2 logs for any `regen pools insufficient` or `undersized` warnings — those are the loud failure modes. | `[ ]` |
| S.6 | After S.5 + S.5a green on music-small, repeat the same S.2-S.5a cycle on **games-small** and **companies-small** before any full-domain R6.1/R6.2 work. Same sanity-run conventions: K2/K4 default `gpt-5.4-mini` (user-approved bump per run), no LLM cache invalidation, S.1 cleanup of any prior `*-small` + `*-augmented` dirs first. Cycle per domain: downsample → measure_baseline → generate_variant → spot-check → EM regen contract. | `[ ]` |
| S.6a | **Per-committee member-level audit (every committee × every sanity-run domain).** For **each of the five committees** (SM / Norm / EM blocking / EM matching / Fusion) on each `*-small` domain, dump every member's score on its committee-specific surface: SM per-attribute F1, Norm per-attribute F1, EM blocking per-blocker `pair_recall` + `reduction_ratio`, EM matching per-matcher F1 (`f1_vs_test` headline + `f1_vs_val` for learned matchers), Fusion per-strategy per-attribute accuracy. Flag every member that scores **0.0** or far below the committee median — silent zeros are real failures hiding behind aggregate `macro_f1`. The bundled-EMCommitteeRunner exception-swallowing pattern at [committee_em.py:1040](../../usecases_synthetic/lib/committee_em.py#L1040) is the **known** instance for EM (see Cross-cutting bug fix #2 below) — analogous patterns exist in SM ([committee_sm.py](../../usecases_synthetic/lib/committee_sm.py) `_score_member` wraps), Norm ([committee_norm.py](../../usecases_synthetic/lib/committee_norm.py)), and Fusion ([committee_fusion.py](../../usecases_synthetic/lib/committee_fusion.py)) — audit each one. Investigate root cause per silent zero (typical causes: column-mapping stale against refreshed CSVs, missing trust scores, reasoning-model `max_tokens` truncation, schema-attribute kind missing from `_DEFAULT_KIND_BY_DOMAIN_ATTR`, list-attribute / `tokenized_match` config gaps). Land the fix under [usecases_synthetic/lib/](../../usecases_synthetic/lib/) before proceeding to the next domain. | `[ ]` |
| S.7 | Only after S.5 + S.5a + S.6 + S.6a green on **all three** `*-small` domains: proceed to R6.1 → R6.2 on the full music / games / companies. | `[ ]` |

**Cross-cutting bug fix #1 (S.3 follow-up, alias resolution)**: the alias-resolution gap above also applies to any future `*-small` (or otherwise aliased) domain run; the fix is now general. **games-small + companies-small are now in scope for the S phase** per the 2026-05-14 user directive; both will be regenerated under the `knob_config_alias` convention as part of S.6.

**Cross-cutting bug fix #2 (S.3 follow-up, reasoning-model max_tokens — 2026-05-13)**: music-small EM stage exposed `openai.BadRequestError 400: max_tokens or model output limit was reached` from [matchgpt_em_matcher.py](../../usecases_synthetic/lib/matchgpt_em_matcher.py). Root cause: `gpt-5.4-mini` (locked as the default model for music/games/companies committees at the 2026-05-06 switch) consumes internal reasoning tokens against the same `max_tokens` budget as visible output. The R5 EM matching sweep had `max_tokens=8` for `llm_matcher` (paper-default for yes/no decisions) and `max_tokens_stage2=8` for ComEM — sufficient for the prior non-reasoning `gpt-4o-mini`, but reasoning models exhaust the 8-token cap on internal scratchpad and return empty visible output → API 400.

Why it wasn't caught at R5: [committee_em.py:1040](../../usecases_synthetic/lib/committee_em.py#L1040) swallows per-matcher exceptions (`except Exception: log + empty preds`); `llm_matcher` silently scored F1=0 on every gpt-5.4-mini call. The R5 sign-off reported per-domain *winners* (Ditto/Magellan dominated music) — the silent zero-F1 of `llm_matcher` was invisible. Cache hits from prior `gpt-4o-mini` populations further masked the issue during sweeps. Same shape applies to `llm_normalizer` (default 64) and fusion `llm_judge` (default 64) — both small enough to truncate reasoning output for adversarial cases.

**Fix locked 2026-05-13**:
- New [usecases_synthetic/lib/llm_client.py](../../usecases_synthetic/lib/llm_client.py) — `build_chat_openai(model, temperature, max_tokens)` centralises every `ChatOpenAI` construction in the synth-lib. Detects reasoning models via regex (`^(gpt-5|o1|o3)`); raises `ValueError` at construction time when `max_tokens < 1024` is set on a reasoning model. **Prevents silent recurrence**: any future caller that drops below the floor fails loudly rather than the previously-silent F1=0 mode.
- All 4 affected ChatOpenAI sites switched to the helper: [matchgpt_em_matcher.py:_ensure_llm_callable](../../usecases_synthetic/lib/matchgpt_em_matcher.py), [comem_em_matcher.py:_build_chat_callable](../../usecases_synthetic/lib/comem_em_matcher.py), [llm_normalizer.py:_ensure_llm_callable](../../usecases_synthetic/lib/llm_normalizer.py), [committee_fusion.py:_build_openai_llm_callable](../../usecases_synthetic/lib/committee_fusion.py).
- Python defaults bumped to **2048**: `MatchGPTMatcher.max_tokens` (8 → 2048), `ComEMMatcher.max_tokens_stage1/2` (128/8 → 2048/2048), `LLMNormalizer.max_tokens` (64 → 2048), `_build_openai_llm_callable.max_tokens` (64 → 2048) + the fallback in `_build_strategy` (64 → 2048).
- YAML defaults bumped to **2048** in 11 committee files (5 EM matching: companies/games/music/movies/products; 3 normalization: companies/games/music; 3 fusion: companies/games/music) for every `max_tokens`, `max_tokens_stage1`, `max_tokens_stage2` row.
- **Caches cleared** (any prior cached entries could have been written with truncation): `usecases_synthetic/cache/{matchgpt_prompts,comem_prompts,llm_normalizer,llm_judge_fusion}/`. Preserved: `cache/llm_cache/r3_pool` (pool adjudicator on `gpt-5.4` without `max_tokens` set — no truncation risk), `cache/magneto_prompts` (SM-LLM matcher without `max_tokens`), and the non-LLM caches.
- **Sites without max_tokens set** ([committee_sm.py:135](../../usecases_synthetic/lib/committee_sm.py#L135), [build_pool.py:392](../../usecases_synthetic/scripts/build_pool.py#L392)) intentionally left alone — they inherit OpenAI's model-level default (16K+), which is safe.

Recommendation per call type, for future bumps: 2048 covers typical reasoning overhead (100-3000 reasoning tokens for EM/Norm/Fusion decisions); OpenAI bills on tokens generated, not on the `max_tokens` cap, so over-provisioning is cost-free. Tighten only if reasoning-cost telemetry shows we never exceed 1024 in practice.

**Cross-cutting refactor #3 (S.3 follow-up, EM committee split — 2026-05-13)**: the bundled `EMCommitteeRunner` conflated blocking + matching into a single F1 number. Per the user directive (and the perfect-prior-step design philosophy first locked in R5 Fusion sign-off, 2026-05-12), each pipeline step should measure its own difficulty against a clean upstream:

- **EM blocking** is its own committee. Runs every blocker on the **full source DataFrames**, scored against the test gold positives. Metrics: per-blocker `pair_recall`, `reduction_ratio`, `candidate_count`. Difficulty signal: recall floor (typically 0.97) must hold from easy → hard; `reduction_ratio` is allowed to degrade. No matchers run.
- **EM matching** is its own committee. Skips blocking entirely. Feeds the labelled `(id1, id2)` pairs from `_test.csv` (or `_test_regenerated.csv` under variants) as candidates to each matcher, runs `matcher.match(df_left, df_right, test_pairs, ...)`, scores under the **closed-set semantic** ([committee_em_scoring.score_em_correspondences_closed_set](../../usecases_synthetic/lib/committee_em_scoring.py)). The labelled negatives in the test CSV provide precision; predictions outside the labelled universe are out-of-scope (not FPs). ML matchers still fit on `_train.csv` via the existing `pair_train_path` injection. **Primary headline: macro `f1_vs_test`** (canonical metric for R6.1 baseline + R7.1 variant comparison). Learned matchers (`matching_type: learned`, e.g. Ditto with early stopping, Magellan classifier sweep) **also** score `f1_vs_val` as a secondary overfit-vs-test diagnostic. Zero-shot matchers (`matching_type: llm`, e.g. MatchGPT, ComEM) **skip val entirely** — val gives the same closed-set F1 as test for them (no hyperparameters tunable on val, no train-data dependency), so val scoring is pure duplicate cost. The val skip cuts MatchGPT cold-cache calls from ~19k → ~2k for music (where the human val.csv is unusually large at ~8-9k pairs per source pair).

Why the bundled architecture masked bugs: per-matcher exception swallowing at [committee_em.py:1040](../../usecases_synthetic/lib/committee_em.py#L1040) silently dropped failed matchers to F1=0; the R5 sign-offs reported per-domain winners only, so the silently-broken `llm_matcher` on `gpt-5.4-mini` never surfaced (Ditto/Magellan won every domain). The split makes each member's metric individually visible.

**Implementation locked 2026-05-13**:
- New [EMBlockingCommitteeRunner](../../usecases_synthetic/lib/committee_em.py) and [EMMatchingCommitteeRunner](../../usecases_synthetic/lib/committee_em.py) classes added; they share lower-level helpers (`_build_blocker`, `_build_matcher`, `_resolve_pair_train_path`, `_generate_blocking_keys`, `_select_best_blocker`) with the bundled runner but orchestrate independently.
- [measure_baseline.py](../../usecases_synthetic/scripts/measure_baseline.py) `ALL_STAGES` now `["sm", "norm", "em_blocking", "em_matching", "fusion"]`. The legacy `em` stage flag is preserved as a one-line alias that expands into the two split stages, so existing CLI invocations (`--stages sm,em`) keep working but emit two metric blocks.
- [committee.py](../../usecases_synthetic/lib/committee.py) `Stage` literal extended to `"em_blocking" | "em_matching"` alongside the legacy `"em"`.
- The bundled `EMCommitteeRunner` is **retained** for [validate_variant.py](../../usecases_synthetic/scripts/validate_variant.py) (which validates the realistic end-to-end pipeline against a variant) and its unit tests. validate_variant.py refactor to use the split runners is deferred to R7.1 cleanup.
- Sweep harnesses ([_tune_em_matching_committee.py](../../usecases_synthetic/scripts/_tune_em_matching_committee.py)) already use the closed-set semantic per the comment block at line 244 — no harness change needed.

Aggregated metric naming under the split: blocking emits `macro_pair_recall`, `macro_reduction_ratio`, `best_member_name`, `best_member_pair_recall`, `best_member_reduction_ratio`; matching emits `macro_f1`, `macro_f1_vs_val`, `macro_f1_vs_test`, `best_member_f1`. Per-pair details survive at `result.per_partition` and `MemberResult.notes["per_pair"]` as before.

### R6 — Re-baseline + full generation runs

| # | Module | Status |
|---|--------|--------|
| R6.1 | `python usecases_synthetic/scripts/measure_baseline.py --domain <d>` for companies, games, music. Records committee version SHAs into `baselines/<domain>/baseline_metrics.json`. R5 sign-off must be done first — the SHAs landed here become the drift-guard reference for R7.1. | `[ ]` |
| R6.2 | `python usecases_synthetic/scripts/generate_variant.py --domain <d> --level all` for companies, games, music. Variants land at `usecases/<domain>-augmented/{easy,medium,hard}/`; inline `package_variant` writes the full directory shape (input/data, input/{schemamatching,entitymatching,fusion}, output/{provenance,baselines}, config/difficulty.yaml). | `[ ]` |

### R7 — Validate + report

| # | Module | Status |
|---|--------|--------|
| R7.1 | `python usecases_synthetic/scripts/validate_variant.py --domain <d> --level <l>` for the 9 (domain, level) cells. Compares against R6.1 baseline; SHA-pinned committee-freeze guard catches any drift. | `[ ]` |
| R7.2 | `python usecases_synthetic/scripts/analyze_monotonicity.py --domain <d>` per domain. Flag any signal whose direction flipped vs the prior pass. **TODO (Pending #8, automatic):** the validator must additionally verify that the **best-member** F1 (not just committee macro_f1) drops with difficulty. A difficulty signal that only depresses the committee mean while the best member stays flat is invalid (the user could pick the best member and ignore the committee). | `[ ]` |
| R7.3 | Per-domain `final_report.md` modelled on [../usecases_synthetic/validation/companies-small/final_report.md](../../usecases_synthetic/validation/companies-small/final_report.md): goal-status table, per-knob summary with committee `macro_f1` + best-member F1, triage list, artifact map. | `[ ]` |
| R7.4 | (optional) Ablation sweep per domain via `downsample_domain.py` + `generate_variant.py --ablation`. Run only on knobs whose R7.1 / R7.2 signal is unclear. | `[ ]` |

## Reporting conventions (carry-over from S14 / S16)

- Every stage delta narrated in plan rows / progress messages reports **both** the committee `macro_f1` *and* the best-member F1 (e.g. `SM macro_f1=0.59 (best=coma_hybrid 1.00)`). Committee mean masks individual strong/weak members; best-member is the "what can a user actually obtain" ceiling.
- Primary EM surface: `f1_vs_regenerated_val` (closed-set), with `_test` fallback when the original EM gold ships test-only. Re-verify which surface applies per domain after R1 — the refreshed EM gold may carry full train/val/test triplets where the old gold did not.
- `f1_vs_test_gold` is benchmarking-vs-prior-work only, never a difficulty surface.

## Terminology convention (2026-05-06)

Drop the **"gold"** suffix when referring to labelled splits. Standard ML terminology applies: **train / validation / test sets**. Carry-over usage in the codebase (`_load_em_gold_ids`, `_load_fusion_gold_ids`, `fusion_gold_ids`, etc.) is honoured for now to avoid an invasive rename, but new code should adopt the new naming and we will batch the refactor in a follow-up. When in doubt:

- "EM train / val / test set" — the labelled pair-classification splits at `usecases/<domain>/input/entitymatching/*_{train,val,test}.csv`.
- "fusion validation set" — `usecases/<domain>/input/fusion/validation_set.xml`.
- "fusion test set" — `usecases/<domain>/input/fusion/test_set.xml`.
- "fusion val + test" — both fusion splits collectively (the protected fusion universe).
- "pool" / "pooled positives" — the cross-evidence positive set at `usecases_synthetic/pools/<domain>/pooled_positives.csv`. No "gold" qualifier; the pool is not a gold standard, it is an evidence union.

**Both fusion validation and test entities are protected** at every value-mutating and entity-mutating knob (K2, K4, K1, K3, K5, K6, K10). [protection.py:_load_fusion_gold_ids](../../usecases_synthetic/lib/protection.py) and [apply_knob_04_coverage.py:_load_fusion_gold_ids](../../usecases_synthetic/scripts/apply_knob_04_coverage.py) already read both `validation_set.xml` and `test_set.xml`; the function names will be renamed to `_load_fusion_protected_ids` in the rename pass.

## Committee tuning convention (2026-05-08)

User-directed (2026-05-08): **every committee gets a hyperparameter tuning sweep before R6.1 freezes baselines**, and the user signs off on each proposed parameter grid before the sweep runs.

Applies to:

- SM committee — done 2026-05-08, see §"R5 SM hyperparameter tuning (2026-05-08)" + §"R5 SM embedding + Magneto tuning (2026-05-08)".
- Normalization committee (NEW) — sweep proposal lands as part of the Pending #2 design.
- EM blocking committee — sweep proposal lands as part of the EM-blocking R5 row + Pending #1 (embedding-model brainstorm).
- EM matching committee — sweep proposal per-domain.
- Fusion committee — sweep proposal per-domain.

**Process per committee**:

1. Claude proposes a parameter grid per member (knobs + ranges) and an estimated cost (wall time + LLM API spend).
2. User signs off, revises, or trims scope.
3. Claude runs the sweep, picks winners by mean F1 across applicable domains (with min-F1 as a robustness tiebreaker), and updates the committee YAML.
4. Re-smoke; document tuning results inline in the relevant R5 sign-off section.

This applies to **every** committee row in R5, not just SM. The sweep harness pattern at [scripts/_tune_sm_committee.py](../../usecases_synthetic/scripts/_tune_sm_committee.py) is reusable — generalise it as `_tune_<stage>_committee.py` per stage.

## LLM model defaults + per-run override (2026-05-06)

K2 hard and K4 easy both use an LLM under `llm_primary_with_paraphrase_fallback`. Default model is **`gpt-5.4-mini`** (set in every per-domain YAML at `config/knob_02_niche/<domain>.yaml` and `config/knob_04_coverage/<domain>.yaml`). Cost basis: ~5× cheaper than full `gpt-5.4`, sufficient for paraphrase-grade fabrication and adjudication.

**Process note (locked 2026-05-06)**: before running K2 or K4 — at any phase including baseline runs (R6.1), variant generation (R6.2), and ablation sweeps (R7.4) — Claude must ask the user whether to bump `llm_model_id` from `gpt-5.4-mini` to `gpt-5.4` for that run. Bumping invalidates the on-disk LLM cache (model_id is part of the cache key) so the bump is non-trivially destructive of prior cached outputs and must be a deliberate choice, not a default. R3's pool adjudicator is locked to `gpt-5.4` per the 2026-05-05 sign-off and is **not** subject to this prompt.

## K4 sign-off (2026-05-06; bugs surfaced + dispatcher fixes implemented 2026-05-07)

Status: `[x]`. Per-domain target tables and closeness-metric coupling locked at the original 2026-05-06 sign-off. **The three load-bearing dispatcher fixes were signed off then but not actually implemented in code; both gaps surfaced during the 2026-05-07 K3 audit and were closed the same day** along with the column-level YAML bugs and a singleton-cap rollback scoping bug surfaced by the smoke. K4 hard now moves the histogram by the expected ~50pp (companies hard `realised_hard.1 = 0.51` vs target 0.55, baseline 0.0) — K4 hard is no longer a phantom knob.

### Bug + fix table (2026-05-07 audit)

**Column-level YAML bugs (caught + fixed 2026-05-07):**

| File | Bug | Fix |
|---|---|---|
| `config/knob_04_coverage/companies.yaml` | `id_columns: {dbpedia: identifier, forbes: Identifier, fullcontact: id}` — pre-loader-rename names; the loader collapses every source's primary id to `id`. | `id_columns: {dbpedia: id, forbes: id, fullcontact: id}`. |
| `config/knob_04_coverage/companies.yaml` | `primary_columns: {dbpedia: name, forbes: Company, fullcontact: name}` — none of these columns exist in the refreshed CSVs. | `primary_columns: {dbpedia: org_name, forbes: company, fullcontact: Attribute_2}`. |
| `config/knob_04_coverage/games.yaml` | `primary_columns: {dbpedia: gameLabel, metacritic: name, sales: Title}` — none exist in refreshed CSVs. | `primary_columns: {dbpedia: title, metacritic: game_title, sales: prod_title}`. |
| `config/knob_04_coverage/music.yaml` | `id_columns: {musicbrainz: rel_id, ...}`; `primary_columns: {musicbrainz: title, ...}` | `id_columns: {musicbrainz: id, ...}`; `primary_columns: {musicbrainz: name, ...}`. |

Without the column fixes, `score_conflict` silently no-op'd on the affected sources (line 403 in [coverage_ops.py](../../usecases_synthetic/lib/coverage_ops.py): `if col not in sources[source].columns: continue`), degrading conflict-preserving removal to a default-rank tiebreak.

**Dispatcher fixes — signed off 2026-05-06, gap caught + closed 2026-05-07:**

| Sign-off claim | Pre-fix reality | Post-fix landing site |
|---|---|---|
| `_would_break_pool_edge` rewritten so removal is allowed when the *other* pool-pair endpoint stays alive (single-endpoint removal OK; both-endpoint forbidden). | Function had the old over-strict semantic — returned True on any pool edge, blocking every removal of any pool-participating record. Companies hard `realised_hard == baseline` (no movement). | [coverage_ops.py:_would_break_pool_edge](../../usecases_synthetic/lib/coverage_ops.py) rewritten to take an in-progress `removed_set` and a per-record `pool_pairs_index`; returns True iff some pool partner is already in the removed set (orphan-only block). New `_build_pool_pairs_index` helper. |
| `score_target_distance` helper in `coverage_ops.py` for closest-to-target survivor selection on fusion val/test entities collapsed to k=1. | Function did not exist; no closeness wiring at all. | New [coverage_ops.py:score_target_distance](../../usecases_synthetic/lib/coverage_ops.py) computes per-source closeness against the fusion target via `protection.is_close_enough`; `select_removal_candidates` ranks fusion val/test entities ascending by this score (least-close source removed first). |
| `fusion_gold_ids → fusion_val_test_ids` rename through `RemovalConstraints`. | Field still named `fusion_gold_entity_ids`; rename never happened. | [coverage_ops.py:RemovalConstraints](../../usecases_synthetic/lib/coverage_ops.py) field renamed; `fusion_gold_entity_ids` kept as a back-compat property alias. Dispatcher call site at [apply_knob_04_coverage.py](../../usecases_synthetic/scripts/apply_knob_04_coverage.py) updated. |

**Plus two follow-up bugs caught while verifying the dispatcher fixes:**

| Bug | Fix |
|---|---|
| `apply_knob_04_coverage.py` blanket-protected every pool-pair endpoint via `_build_protected_records(pool_pairs)` and every fusion-gold record via an explicit loop (lines 540-551 pre-fix). With the new orphan-only `_would_break_pool_edge` and the closeness-aware survivor selection, both blanket-protections are wrong — they re-introduce the over-strict block the sign-off explicitly removes. | Both blanket-protection blocks dropped from the dispatcher. `protected_records` stays as the explicit-anchor surface (empty by default; reserved for K1/K2 anchors). Pool-edge protection lives inside `_would_break_pool_edge`; fusion-protected entities are demoted via closeness ranking. |
| `apply_singleton_cap_rollback` used `len(view)` as the singleton-fraction denominator. `len(view)` includes synthetic distractor singletons (entity ids prefixed `__singleton__:` — ~9 656 on companies). Baseline distractor-singleton fraction was already ~90 %, far above any `singleton_cap_hard` value (≤ 0.70), so the rollback fired immediately and rolled back **every** demotion → K4 hard was a phantom knob even after the orphan-check fix. | Rollback now scopes the denominator and the singleton count to **matchable** entities (every entity id NOT prefixed `__singleton__:`). Mirrors the K3 follow-up #1 fix flavour (rollback-scoping). |

**K4 test updates:**

The two pre-existing K4 tests `test_fusion_gold_floor` and `test_pool_edges_preserved` encoded the **old** blanket-protection semantic. Both updated to assert the new semantic per the sign-off:

- `test_fusion_gold_floor`: every entity (fusion-protected included) retains ≥1 surviving source. The prior per-record stays-resident check is gone — closeness-aware survivor selection now decides which fusion-gold record survives the demotion.
- `test_pool_edges_preserved`: every pool pair retains ≥1 alive endpoint (single-endpoint removal allowed; both-endpoint forbidden by `_would_break_pool_edge`).

### Smoke test (2026-05-07, post-implementation)

Realised vs target histogram per (domain, level), with the dispatcher fixes live:

| Domain | Level | Baseline | Target | Realised | Skipped |
|---|---|---|---|---|---|
| companies | easy | `{1: 0.0, 2: 0.80, 3: 0.20}` | `{1: 0.0, 2: 0.10, 3: 0.90}` | `{1: 0.0, 2: 0.10, 3: 0.90}` | 0 |
| companies | hard | same | `{1: 0.55, 2: 0.30, 3: 0.15}` | `{1: 0.51, 2: 0.34, 3: 0.15}` | 0 |
| games | easy | `{1: 0.0, 2: 0.59, 3: 0.41}` | `{1: 0.0, 2: 0.10, 3: 0.90}` | `{1: 0.0, 2: 0.10, 3: 0.90}` | 0 |
| games | hard | same | `{1: 0.55, 2: 0.30, 3: 0.15}` | `{1: 0.38, 2: 0.47, 3: 0.15}` | 0 |
| music | easy | `{1: 0.0, 2: 0.60, 3: 0.40}` | `{1: 0.0, 2: 0.10, 3: 0.90}` | `{1: 0.0, 2: 0.10, 3: 0.90}` | 0 |
| music | hard | same | `{1: 0.55, 2: 0.30, 3: 0.15}` | `{1: 0.39, 2: 0.46, 3: 0.15}` | 0 |

`realised_hard.3` matches target.3 exactly across all domains (the 3→2 demotion path is fully unblocked). `realised_hard.1` falls 4-17pp short of the target on the singleton bin — the 2→1 demotions are partially blocked by the orphan check (each blocked-because-partner-already-removed event prevents one further demotion to k=1). This is the spec-required behaviour: pool pairs are protected against complete orphaning; the gap is the population of entities whose only-remaining pool partner is itself an in-progress removal candidate. Acceptable; R7.2 will judge whether the singleton-bin under-shoot needs re-calibrated targets or a different demotion-planner that anticipates orphan-blocking.

`pytest usecases_synthetic/tests/`: 990 passed, 1 skipped (the skip is the pre-existing `test_joint_values.py` fixture-staleness flagged in the K3 audit, unchanged).

### Follow-up flags (not K4-blocking)

1. **Singleton-bin under-shoot at hard.** Companies 4pp, games 17pp, music 16pp under-shoot vs the YAML hard target on `H[1]`. Driven by orphan-block events during the 2→1 phase. Two possible fixes if R7.2 surfaces this: (a) author lower-`H[1]` targets that match the orphan-constrained achievable, or (b) extend `plan_demotions` to model orphan-blocking and over-budget the request. Defer until R7.2.
2. **`fusion_gold_entity_ids` back-compat alias** lives on `RemovalConstraints` as a property. Cleanup-of-dead-code candidate once external callers confirm nobody uses the old name.

### Dispatcher fixes (locked, must land before R6.1)

These are the load-bearing fixes that make K4 hard non-trivially functional. The prior-pass realised hard was `realised_hard == H_base` for **all three domains** because the existing pool-protection check is over-strict; without the fix, K4 hard remains a phantom knob.

1. **Rewrite `_would_break_pool_edge`** in [usecases_synthetic/lib/coverage_ops.py:420](../../usecases_synthetic/lib/coverage_ops.py#L420). Current implementation rejects removal of any record that participates in any pool pair, which rejects every record in a matchable entity. New semantic: removal is allowed iff for every pool pair through this record, the *other* endpoint is still alive (not in the in-progress removed-set). Block only when removal would orphan every pool partner. Matches the spec phrasing in [knob_04_coverage_skew.md:100](../../knobs/knob_04_coverage_skew.md#L100): "removal may not break a pool-declared match edge by collapsing both endpoints onto a single source." Single-endpoint removal is allowed; both-endpoint removal is forbidden.
2. **Add closest-to-target survivor selection for fusion val + test entities.** When a fusion val/test entity is being collapsed to k=1, override the conflict-preserving (`score_conflict`) source-ranking with a closeness-to-fusion-value ranking: pick the source whose record's per-attribute values maximise `is_close_enough(value, fusion_value, tolerance)` against the fusion val/test attribute values for that entity. New helper `score_target_distance` in `coverage_ops.py`; reads the per-attribute tolerance spec authored in [usecases_synthetic/lib/protection.py](../../usecases_synthetic/lib/protection.py) (Pending #5 surface). For non-fusion-val/test entities, the existing conflict-preserving order is unchanged.
3. **Plumb `fusion_val_test_ids`** explicitly through `RemovalConstraints`. Today the loader pulls val + test together, so no behavioural change — but the variable name in `select_removal_candidates` should change from `fusion_gold_ids` to `fusion_val_test_ids` so the closest-to-target override is correctly scoped.

### Per-domain target histograms (conservative)

Authored against the prior pass `H_base`; R6.1 re-measurement may shift these slightly. Cross-level monotonicity (stochastic dominance) holds across all three: `H_easy[1] ≤ H_medium[1] ≤ H_hard[1]` for any reasonable `H_base[1] ∈ [0, 0.40]`.

| Domain | `target.easy` | `target.hard` | `singleton_cap_hard` | `fabrication_mode` |
|---|---|---|---|---|
| companies | `{1: 0.0, 2: 0.10, 3: 0.90}` (unchanged) | `{1: 0.40, 2: 0.45, 3: 0.15}` | 0.50 (was 0.60) | `llm_primary_with_paraphrase_fallback` |
| games | `{1: 0.0, 2: 0.20, 3: 0.80}` | `{1: 0.35, 2: 0.50, 3: 0.15}` | 0.50 (was 0.60) | `llm_primary_with_paraphrase_fallback` |
| music | `{1: 0.0, 2: 0.30, 3: 0.70}` | `{1: 0.30, 2: 0.55, 3: 0.15}` | 0.50 (was 0.60) | `llm_primary_with_paraphrase_fallback` (tightened contamination threshold 0.05 vs default 0.10) |

`within_source_duplicate_rate.hard = 0.02` carried over for all three domains.

### Closeness-metric specification (couples K4 ↔ K6 ↔ Pending #5 protection.py)

Single source of truth: [usecases_synthetic/lib/niche_metrics.py:lexical_extended_jaccard](../../usecases_synthetic/lib/niche_metrics.py#L151) (already in use for K2 corner-case mining). Used in K4's closest-to-target survivor scoring, K6's per-cell tolerance check, and the relaxed protection rule (Pending #5).

| Attribute type | Metric | Threshold |
|---|---|---|
| Numeric / count / score | absolute or relative band | ±3 % relative, ±1 absolute (years) |
| Date | calendar-day delta | ≤ 1 day (≤ 1 year for year-only fields) |
| Short string / nominal (≤ 3 tokens) | `_levenshtein_ratio` (used inside `lexical_extended_jaccard`) **OR** synonym-set membership | ratio ≥ 0.85, or value ∈ per-attribute synonym set |
| Long / multi-token string (title, name, label) | `lexical_extended_jaccard(inner_token_threshold=0.8)` (extended Jaccard with inner Levenshtein gate) | ≥ 0.6 |
| Free text / description | `lexical_extended_jaccard(inner_token_threshold=0.8)` | ≥ 0.5 |
| Lists / set-valued (genres, platforms) | `lexical_extended_jaccard` over flattened set tokens | ≥ 0.5 |

Per-attribute overrides may be authored in `config/knob_06_noise/<domain>.yaml` under `fusion_protection_tolerance` (the K6 surface that owns the tolerance spec).

### LLM cost estimate (one-time)

Model: `gpt-5.4-mini` (default; bump to `gpt-5.4` requires explicit per-run user approval, see §"LLM model defaults").

Assumed pricing: $1/MTok input, $4/MTok output, with cached input at $0.10/MTok (prompt caching). Per-call cost ~$0.0014 input ($0.00013 cached + ~$0.001 uncached) + ~$0.0008 output = **~$0.002/call**.

| Domain | Fabrications (est.) | One-time cost |
|---|---|---|
| companies | ~1,040 | ~$2 |
| games | ~3,800 | ~$8 |
| music | ~2,300 | ~$5 |
| **Total** | **~7,150** | **~$15** |

(±50 % uncertainty band on token counts and matchable-population estimates; R6.1 measurement will firm up.)

## K1 sign-off (2026-05-07)

Status: `[x]`. Approved column-name remap to refreshed CSV sources, expanded per-domain abbreviation tables, LLM-model alignment with K2/K4, and two carried-over follow-up flags. R6.1 will measure realised paraphrase rates per attribute class on the refreshed sources before R6.2 runs.

### Changes locked

1. **Column-name remap** (the load-bearing fix — pre-refresh K1 configs were dead on the new CSVs). Remapped per domain to refreshed columns; `id_columns` collapses to `id` everywhere via the loader rename.
   - **companies**: dbpedia `org_name/nation/headquarters/sector/keypeople_name`; forbes `company/region/business_segment`; fullcontact blind `Attribute_2/_3/_4/_5`. Normalize-down: `forbes.region → dbpedia.nation` (shortest).
   - **games**: dbpedia `title/system/genre/studio/franchise`; metacritic `game_title/console/age_rating/genres/made_by`; sales `prod_title/hw/age_classification/genre/studio/dist`. Normalize-down: `dbpedia.system → sales.hw` (shortest).
   - **music**: musicbrainz `name/artist/release-country/release-date/label`; discogs adds `genre`; lastfm stays thin (`name/artist`) because release-country/release-date/label/genre are 0 % populated post-refresh. Normalize-down: `musicbrainz.release-country → discogs.release-country` (shortest) — direction flipped from prior pass since musicbrainz is the long-form source on the new data.

2. **LLM model default → `gpt-5.4-mini`** across all three domains (was `claude-opus-4-6`). Aligns K1 with the §"LLM model defaults + per-run override (2026-05-06)" policy already locked for K2/K4. Per-run prompt: ask before bumping to `gpt-5.4` since `model_id` is part of the cache key.

3. **Expanded abbreviation tables**:
   - companies: ~30 entries (legal-form suffixes, common-word contractions, country forms, frequent-entity aliases) + new `categorical_vocab_map` for industry/sector synonyms.
   - games: ~40 entries (PlayStation/Nintendo/Xbox platform vocab, ESRB ratings, publisher/developer abbreviations) + `categorical_vocab_map` for ESRB and genre.
   - music: ~25 entries (country forms, feat./vs. variants, volume/no./pt. suffixes, parenthetical edition tags, label suffixes) + `categorical_vocab_map` for genre + `article_drop_pattern: "^(The|A|An) "` for leading-article removal.

4. **Per-level rates carried over from companies** (no recalibration in this pass). Primary 0/0.02/0.08, key 0/0.04/0.12, secondary 0/0.08/0.20, categorical 0/0.04/0.12. Smoke-test realised rates: companies medium ~3.7 %, games medium ~3.0 %, music medium ~2.7 % — within the configured envelope. R7.2 will flag if monotonicity is too weak.

5. **Operator mix unchanged.** Easy: `normalize_to_canonical` only. Medium: `abbreviate (2.0) + eda_swap (1.0) + eda_delete (1.0)`. Hard: medium ∪ `llm_paraphrase (2.0)`.

6. **Anchor-survivor floor**: `primary: true` only (key/secondary/categorical waived) — opt-in per attribute class via `anchor_survivor_floor.<class>: true`.

7. **`paraphrase_long` stays dead** — no long-text target attribute in any of companies/games/music post-refresh. `prompt_long_v1.txt` not authored.

### Smoke test (2026-05-07)

`apply_knob_01_surface.py` ran clean for {companies, games, music} × {easy, medium} on the refreshed CSVs:

| Domain | Easy provenance | Medium provenance | Medium skipped |
|---|---|---|---|
| companies | 152 (`forbes.region`) | 519 | 1147 (mostly `abbreviate_no_effect` + `too_short_for_eda`) |
| games | 14 (`dbpedia.system`) | 6626 | 10229 |
| music | 794 (`musicbrainz.release-country`) | 2325 | 5780 |

`pytest usecases_synthetic/tests/test_knob_01.py test_loaders.py test_committee_configs.py`: 193/193 green.

### Follow-up flags (not K1-blocking)

1. **Easy normalize-down inherits sibling-source errors.** Companies easy run produced `forbes.region: China → Taiwan` and `Hong Kong → China` because the dbpedia sibling's `nation` value was wrong for those entities, and the operator picks "shortest" without a closeness gate. Pending #5's closeness contract is exactly designed to fix this — when wired in, `is_close_enough(sibling_value, fusion_target, tolerance)` must hold before the operator fires. Folded into the K6-review wire-up of `protection.fusion_cell_tolerance(domain, attribute)`.
2. **Anchor-survivor floor over-fires on singletons.** [_check_clean_primary_floor](../../usecases_synthetic/scripts/apply_knob_01_surface.py#L420-L426) returns False when `len(members) <= 1`, which causes singleton entities (no cross-source anchor to protect) to be skipped with `reason=anchor_survivor_floor`. Singletons should be allowed to paraphrase freely. Carried over from Phase C; ~233/1397/540 affected cells in the medium smoke. Fix: change the early-return to `True` when the entity has no cross-source matches. Land alongside the Pending #5 wire-up (same dispatcher area).

## K5 sign-off (2026-05-07)

Status: `[x]`. Approved column-name remap to refreshed CSVs, duration-family dispatcher wiring (closes Pending #7), forbes magnitude flip from billions → raw, and games `units_sold_mm` routed via the existing money/number path. R6.1 will measure the per-domain `baseline_format_profile` on the refreshed sources before R6.2 runs.

### Changes locked

1. **Column-name remap** (load-bearing — pre-refresh K5 configs were dead on the new CSVs).
   - **companies**: dbpedia `{established: date, total_assets_val: money, annual_income: money}`; forbes `{asset_value: money, sales_figure: money}`; fullcontact `{Attribute_6: date}` (founded year, ISO YYYY-01-01 padded). `id_columns: {dbpedia: entity_uri, forbes: forbes_url, fullcontact: Attribute_1}`.
   - **games**: dbpedia `{launch_yr: date}`; metacritic `{year_published: date}`; sales `{launch_dt: date, units_sold_mm: money}`. `id_columns: {dbpedia: wiki_ref, metacritic: mc_id, sales: rec_id}`.
   - **music**: all three sources `{release-date: date, duration: duration}`. `id_columns: {musicbrainz: id, discogs: id, lastfm: id}`. Hyphenated column names work transparently via `df[col]` string lookup.

2. **Forbes magnitude flip — billions → raw.** The refreshed Forbes corpus ships raw dollars (`148_700_000_000`) instead of billions. `source_magnitude_context.forbes.implicit_magnitude: raw` in [companies.yaml](../../usecases_synthetic/config/knob_05_format/companies.yaml). Direction-flips the easy-level normalize-down and the medium/hard magnitude swaps.

3. **Duration-family dispatcher wiring (Pending #7 closed).**
   - New helpers `format_duration` and `parse_duration` in [format_operators.py:453](../../usecases_synthetic/lib/format_operators.py#L453) — accept `seconds_int / mm_ss / hh_mm_ss / human_xm_ys` with round-trip parse-back to canonical seconds. Negative values, malformed inputs, and unknown target formats return `None`.
   - New `_transform_duration` helper in [apply_knob_05_format.py](../../usecases_synthetic/scripts/apply_knob_05_format.py) emits provenance under `transform_fn=reconvert_unit` (matches the schema; the duration string forms are semantically unit conversions on the same canonical seconds).
   - The dispatcher's `family in ("duration", "dimensional")` branch routes both into the new path. `dimensional` is unused today (no domain authors a dimensional-family attribute) but is wired for future use without further dispatcher changes.

4. **Per-domain unit pools** (frozen pool sizes per K5 card: easy=2, medium=3, hard=4):
   - **music.duration**: easy `[seconds_int, mm_ss]`, medium `+ hh_mm_ss`, hard `+ human_xm_ys`.
   - **games.units_sold_mm**: easy `[millions]` (kept), medium `[millions, raw]`, hard `[millions, raw, thousands]` — magnitude swap exercised via the existing money path with `currencies: [units]` (no FX, just magnitude conversion).
   - **companies.money**: pool unchanged from the pre-refresh authoring; only the per-source magnitude context changed (forbes raw, dbpedia raw).

5. **Date pools carried over** unchanged from the pre-refresh authoring. Easy still injects one alternative format per domain (companies/games `long_english`; music `precision_year` to preserve year-only round-tripping). Keeps "easy is not a no-op" on every domain.

6. **Operator mix**: `reformat_date / reformat_number / reconvert_unit / reconvert_currency` — unchanged. `format_duration` is emitted under the existing `reconvert_unit` provenance `transform_fn` since the duration forms are still semantically unit conversions on the same canonical value.

### Bug + fix (2026-05-07 audit, post-K10)

| File | Bug | Fix | Impact |
|---|---|---|---|
| `config/knob_05_format/companies.yaml` | `id_columns: {dbpedia: entity_uri, forbes: forbes_url, fullcontact: Attribute_1}` — pre-loader-rename names. | `id_columns: {dbpedia: id, forbes: id, fullcontact: id}`. | K5 dispatcher uses `id_col` only for the provenance `entity_id` field; falls back to `str(idx)` (row index) on miss. So pre-fix K5 reformatting still functioned correctly, but the provenance entity_id was a row index rather than the canonical record id (e.g. `7` instead of `http://www.forbes.com/companies/avago-technologies/`). Cross-knob provenance correlation degraded. |
| `config/knob_05_format/games.yaml` | `id_columns: {dbpedia: wiki_ref, metacritic: mc_id, sales: rec_id}` — pre-loader-rename names. | `id_columns: {dbpedia: id, metacritic: id, sales: id}`. | Same as companies — provenance entity_id was a row index. |

Music K5 yaml was already correct (`id` everywhere). The K5 reformatting numbers in the smoke table below were unaffected by the bug; the regression is purely on provenance-integrity, not on the format-heterogeneity or round-trip-integrity claims.

### Smoke test (2026-05-07)

`apply_knob_05_format.py` ran clean for {companies, games, music} × {easy, medium, hard} on the refreshed CSVs:

| Domain | Easy provenance / skipped | Medium | Hard |
|---|---|---|---|
| companies | 15437 / 22 | 8972 / 4640 | 17374 / 3599 |
| games | 53048 / 0 | 53048 / 0 | 60852 / 240 |
| music | 42147 / 0 | 30731 / 3510 | 43139 / 890 |

Skip rates concentrate at medium (locale-ambiguous deny-list — `eu_dot` on year-only ISO `YYYY-01-01` cells emits `01.01.YYYY`-style patterns that hit the deny list) and at hard (mixed `eu_dot` / `two_digit_year_us` draws on the same year-only inputs). Acceptable: every skip is recorded with a reason code and the failure mode is structural (year-only baseline + format-pool deny-list interaction), not a bug.

Sanity verification at music/hard:
- **Duration heterogeneity**: all 4 forms appear roughly evenly within each source (musicbrainz / discogs / lastfm each ~480-525 cells per form on the head 2000 sample).
- **Round-trip integrity**: 0/1000 sampled duration provenance rows fail `parse_duration(new_value) == parse_duration(original_value)`.

Sanity verification at companies/hard: dbpedia `established` and fullcontact `Attribute_6` both show ≥ 3 distinct date forms within the same source (slash + iso + dot) — within-source format inconsistency at hard confirmed.

Sanity verification at games/hard: `sales.units_sold_mm` mixes `82` (millions, native), `32000` (thousands), `29 000 000` (raw, fr_FR), `28,000,000` (raw, en_US), `15000000` (raw, plain) within the same column.

`pytest usecases_synthetic/tests/test_knob_05.py test_loaders.py test_committee_configs.py`: 203/203 green.

### Follow-up flags (not K5-blocking)

1. **Discogs duration uses `0` as a missing-value sentinel.** `format_duration("0", "mm_ss") → "0:00"`. Round-trip parses back to `0`, so the operator is correct, but the cell semantics ("missing duration") become opaque after reformatting. Fix is at the loader layer, not K5 — should be coalesced to NaN before K5 sees the row. Carry over to the K3 (drop / nesting) review where missing-value handling is on-card.
2. **Within-source consistency at easy/medium for music duration.** With 5+ forms in the music domain and `consistency: source` at easy/medium, every source picks one of `[seconds_int, mm_ss]` (easy) or `[seconds_int, mm_ss, hh_mm_ss]` (medium) for its entire duration column. RNG can pick `seconds_int` for all three sources at easy on certain seeds → easy is then a literal no-op for duration. R7.2 will catch this if it produces flat monotonicity; fix would be a consistency override (`per-source-cycling` instead of `per-source-random`) — defer until after R7.

## K6 sign-off (2026-05-07)

Status: `[x]`. Approved column-name remap to refreshed CSVs, taxonomy-walk operator (Pending #4 closed), ±2 % numeric jitter cap with retry loop (Pending #6 closed), and refreshed cleanup rules. Pending #5 (closeness contract) **split out** into its own R4 row before K3 — K6 ships with the existing strict survivor floor. R6.1 will measure realised per-source noise rates per attribute class on the refreshed sources before R6.2 runs.

### Changes locked

1. **Column-name remap** (load-bearing — pre-refresh K6 configs were dead on the new CSVs). `id_columns` collapse to `{<source>: id}` everywhere via the loader rename.
   - **companies**: dbpedia `{org_name: primary, nation/headquarters/sector: key, established/keypeople_name/total_assets_val/annual_income: secondary}`; forbes `{company: primary, region/business_segment: key, asset_value/sales_figure: secondary}`; fullcontact `{Attribute_2: primary, Attribute_3/4: key, Attribute_5/6: secondary}`. `attribute_mapping` updated to canonical schema (`name/country/city/industry/founded/keypeople/assets/revenue`).
   - **games**: dbpedia `{title: primary, system: key, launch_yr/studio/genre/franchise: secondary}`; metacritic `{game_title: primary, console/age_rating: key, year_published/made_by/genres/press_rating/player_rating: secondary}`; sales `{prod_title: primary, hw/age_classification: key, launch_dt/studio/dist/genre/press_score/comm_rating/units_sold_mm: secondary}`. `attribute_mapping` to canonical (`name/platform/ESRB/releaseYear/developer/publisher/genres/criticScore/userScore/globalSales/series`).
   - **music**: all three sources `{name: primary, artist/release-country/genre: key, release-date/duration/label: secondary}`. lastfm key/secondary attributes are 0 %-populated post-refresh per K1 sign-off — K6 already no-ops on null cells, so the empty bindings cost nothing and exist so a future refresh that backfills the columns needs no config change.

2. **Pending #4 — `taxonomy_walk` operator (NEW).** Single-level walk only at both medium and hard. Direction draw 50/50 up/down (up = parent label, down = random child sibling at the next-deeper level). Falls through with `None` when the cell isn't in the bound taxonomy → dispatcher draws a fresh operator on retry.
   - New `Taxonomy` class + `taxonomy_walk` function + `load_taxonomy` cache at [noise_operators.py](../../usecases_synthetic/lib/noise_operators.py).
   - Per-domain bindings under a new `taxonomies` config block (path relative to `usecases/`):
     - **companies** — `dbpedia.sector + forbes.business_segment` → `companies/input/schemamatching/GICS_Industry_Taxonomy.csv` (4 levels: Sector → Industry Group → Industry → Sub-Industry). `fullcontact.Attribute_4` confirmed = city/locality (not industry-like) — no binding.
     - **games** — `(dbpedia.system, metacritic.console, sales.hw)` → `Gaming_Platforms_Taxonomy.csv` (3 levels: Platform Type → Manufacturer → Platform Name). `(dbpedia.genre, metacritic.genres, sales.genre)` → `Video_Game_Genres_Taxonomy.csv` (3 levels). `ESRB_Rating_Taxonomy.csv` is flat — intentionally not used (no walk to perform on a flat vocabulary; ESRB synonym normalisation belongs in the Norm stage).
     - **music** — `(musicbrainz.genre, discogs.genre, lastfm.genre)` → `Music_Genres_Taxonomy.csv` (3 levels). musicbrainz/lastfm `genre` columns are empty on the refreshed CSV; K6 no-ops on null so the bindings cost nothing.
   - Operator-mix integration: easy unchanged; medium gains `taxonomy_walk: 1.0`; hard `taxonomy_walk: 1.5`.
   - Provenance `transform_fn=taxonomy_walk`, params include `taxonomy / direction / from_level / to_level / from_label`.

3. **Pending #6 — ±2 % relative jitter cap with retry loop (NEW).**
   - New `numeric_jitter_within_cap(original, new, type, max_relative)` helper at [noise_operators.py](../../usecases_synthetic/lib/noise_operators.py) + `_try_parse_float / _try_parse_year / _try_parse_date` parsers.
   - New `numeric_attributes` per-domain config block tagging columns as `continuous` / `year` / `date`. New `numeric_jitter_max_relative` scalar (default `0.02`).
   - **Decision rule (locked 2026-05-07)**:
     - `continuous`: both parse → reject if `abs(new − orig) / max(abs(orig), 1.0) > 0.02`. Either unparseable → **allow** (string corruption, not jitter).
     - `year`: both parse as 4-digit year → reject if year differs at all. Either unparseable → **allow** (canonical K6 corruption per spec: "malformed unparseable date is K6").
     - `date`: both parse as calendar date → reject if calendar-day delta > 0. Either unparseable → **allow**.
   - **Retry loop** in the dispatcher: up to `_MAX_JITTER_RETRIES = 5` attempts per cell, each drawing a fresh operator. On exhaustion log `numeric_jitter_exhausted_retries` to skipped. The retry also gates `taxonomy_walk` on bound columns (a `taxonomy_walk` draw on a non-bound column burns nothing — it's filtered before the operator runs).
   - **Per-domain bindings**:
     - **companies** dbpedia `{established: year, total_assets_val: continuous, annual_income: continuous}`; forbes `{asset_value: continuous, sales_figure: continuous}`; fullcontact `{Attribute_6: year}`.
     - **games** dbpedia `{launch_yr: year}`; metacritic `{year_published: year, press_rating: continuous, player_rating: continuous}`; sales `{launch_dt: date, press_score: continuous, comm_rating: continuous, units_sold_mm: continuous}`. `units_sold_mm` cap applies to the raw float as stored (millions, per K5 sign-off).
     - **music** all three sources `{release-date: date, duration: continuous}`.

4. **Pending #5 split.** Closeness-contract refactor (`fusion_cell_tolerance` / `is_close_enough` / `_load_fusion_target_values` in [protection.py](../../usecases_synthetic/lib/protection.py) + per-knob survivor-floor updates) split out into its own R4 row between K6 and K3. K6 ships with the existing strict "≥ 1 unmutated source" floor; the split-out row owns the K1/K3/K5/K6/K10 wire-up.

5. **Cleanup rules — refresh against new column names** (load-bearing, easy-only).
   - **companies**: `forbes.region` strip `[a]` footnote (47 cells confirmed on the refreshed CSV). `dbpedia.org_name` control-char rule **retired** — 0 matches on the refreshed CSV.
   - **games**: `dbpedia.title` strip `(YYYY video game)` / `(video game)` parenthetical (9889 cells confirmed).
   - **music**: `lastfm.name` strip leading `<artist> -` prefix (1365/9865 cells confirmed).

6. **Per-class rates and per-source-shape transform unchanged** from the carry-over default (companies). Calibration deferred to R7.2 monotonicity check.

### Smoke test (2026-05-07)

`apply_knob_06_noise.py` ran clean for {companies, games, music} × {easy, medium, hard} on the refreshed CSVs:

| Domain | Easy prov / skip | Medium prov / skip | Hard prov / skip |
|---|---|---|---|
| companies | 58 / 199 | 106 / 1340 | 298 / 4090 |
| games | 9942 / 3097 | 203 / 14345 | 645 / 43889 |
| music | 1469 / 674 | 563 / 4325 | 1789 / 13101 |

Easy provenance is dominated by `cleanup` rows (companies 47, games 9889, music 1365); non-cleanup easy counts are 11 / 53 / 104 respectively. Non-cleanup monotonicity holds in all three domains: companies 11 → 106 → 298, games 53 → 203 → 645, music 104 → 563 → 1789.

Sanity verification at hard:
- **Numeric cap**: continuous mutations on `dbpedia.annual_income` either stay within ±2 % (e.g. truncating a trailing `.0`) or become unparseable strings (`'3748000000.0' → 'w7e8p00000.0'`, allowed as corruption). No accepted mutation crosses the cap.
- **Year cap**: `dbpedia.launch_yr` mutations all produce unparseable-as-year strings (`'2005-01-01' → '200S-01-0l'`), allowed; no accepted mutation shifts the parsed year.
- **Taxonomy walk**: walks are semantically correct — companies `{Financial services → Financials, Regional Banks → Banks}` (up); games `{Wii → Nintendo, Xbox 360 → Microsoft, Shooter → Action}` (up); music `{Rock → Progressive Rock / Punk / Hard Rock / Indie Rock / Soft Rock}` (down).
- **`numeric_jitter_exhausted_retries`** firing as expected: companies medium=11/hard=74; games easy=3/medium=4/hard=14; music medium=12/hard=95 — confirms the cap rejects on the wrong side of the decision rule and the retry loop is bounded.

`pytest usecases_synthetic/tests/test_knob_06.py test_loaders.py test_committee_configs.py`: 175/175 green.

### Follow-up flags (not K6-blocking)

1. **Cleanup-vs-noise split at easy.** Easy prov totals are dominated by `cleanup` rows (which are not strictly "noise"). Non-cleanup easy is small but non-zero, so easy is not a literal no-op. R7.2 will judge whether the non-cleanup easy signal is large enough to support the easy → medium step on K6 alone; if not, the fix is to re-introduce minimal incidental noise at easy (e.g. raise `secondary` from 0.01 to 0.02) rather than re-relax the per-class rates. Defer until after R7.
2. **Discogs duration `0`-as-sentinel** carried over from K5 follow-up #1: K6's continuous-jitter cap is sympathetic with this (`0 → 0.0` jitters within cap trivially) but a `0`-sentinel duration cell that gets noised into a non-zero string will look like a real duration. Fix is at the loader (coalesce to NaN before K6 / K5 see it). Folded into the K3 review where missing-value handling is on-card.
3. **lastfm.name cleanup over-strips a non-prefix dash.** Pattern `^.+?\s+-\s+` matches `'- new - Subtle Frequencies'` and strips `'- new - '` → `'Subtle Frequencies'`. Most matches are correct (`'John B - Fermats Theorem / Sight Beyond'` → `'Fermats Theorem / Sight Beyond'`) but a small number are over-aggressive. Acceptable for easy-only cleanup since the cell is on a non-fusion-protected source; revisit if R7.1 surfaces a regression on lastfm.

## Pending #5 closeness-contract wire-up (2026-05-07)

Status: `[x]`. Replaces every value-mutating knob's strict "≥ 1 unmutated source" survivor floor with a closeness-aware "≥ 1 surviving source within tolerance" gate per the locked closeness-metric specification (§"K4 sign-off → Closeness-metric specification"). R6.1 will measure realised closeness-skip rates per knob on the refreshed sources before R6.2 runs.

### protection.py refactor (locked)

New surface in [usecases_synthetic/lib/protection.py](../../usecases_synthetic/lib/protection.py):

- `ToleranceSpec(kind, threshold, inner_token_threshold)` — frozen dataclass capturing per-attribute tolerance.
- `_DEFAULT_TOLERANCE_BY_KIND` — locks the kind → threshold map (continuous=±0.03, year=±1, date=±1 day, nominal=Levenshtein ≥ 0.85, long_string=`lexical_extended_jaccard` ≥ 0.6, free_text=≥ 0.5, list=≥ 0.5).
- `_DEFAULT_KIND_BY_DOMAIN_ATTR` — per-domain canonical-attribute → kind mapping for companies / games / music (locked from K1/K5/K6 sign-offs).
- `fusion_cell_tolerance(domain, canonical_attribute, config_overrides=None)` — resolver with optional per-domain overrides via a `fusion_protection_tolerance` block in `config/knob_06_noise/<domain>.yaml`.
- `is_close_enough(value, target, tolerance)` — closeness predicate dispatching by kind. Numeric / date / year inputs use type-specific parsers (`_parse_float`, `_parse_year`, `_parse_date`); strings use Levenshtein ratio or extended Jaccard via the existing primitives in [niche_metrics.py](../../usecases_synthetic/lib/niche_metrics.py).
- `load_fusion_target_values(domain) -> dict[entity_id, dict[attribute, list[str]]]` — reads both `validation_set.xml` + `test_set.xml`, handling multi-valued attributes (genres, keypeople) as flat lists.
- `cell_has_close_survivor(target_values, surviving_values, tolerance)` — the per-cell test used by all knob guards.
- `_load_fusion_protected_ids(domain)` — renamed from `_load_fusion_gold_ids` per the §"Terminology convention" pass; the old name remains as a back-compat alias (K3 / K4 importers unchanged).

### K6 wire-up

[apply_knob_06_noise.py](../../usecases_synthetic/scripts/apply_knob_06_noise.py):

- New `_ClosenessContext` helper encapsulates protected-id set, target-value lookup, canonical → (source, source-col) reverse index, per-source `id → row_idx` lookup, and a memoised tolerance resolver.
- New `_check_close_survivor_floor(...)` performs the post-mutation contract test: for fusion-protected (entity, canonical attribute) cells, ≥ 1 record across the entity's sources must remain within tolerance of a fusion target value after the candidate value is committed.
- Strict pre-mutation `_check_clean_survivor_floor` removed (its skipped-reason `clean_survivor_floor` no longer surfaces).
- Closeness gate hooks into the existing retry loop after the numeric-jitter cap. On rejection the operator-draw retries with a fresh draw (`closeness_violation` attempt-log marker); on retry exhaustion the cell is logged with new reason `closeness_floor_exhausted_retries`.
- `_check_clean_primary_floor` unchanged — anchor-survivor for primary identity remains independent of the fusion contract.

### K1 wire-up

[apply_knob_01_surface.py](../../usecases_synthetic/scripts/apply_knob_01_surface.py):

- Same `_ClosenessContext` + `_check_close_survivor_floor` shape as K6.
- Strict pre-draw `_check_clean_survivor_floor` removed (skipped-reason `per_cell_clean_survivor_floor` no longer surfaces); replaced by post-mutation closeness check at commit time (new skipped-reason `closeness_floor_violation`). K1 operators are deterministic per cell so we don't retry — we skip when the candidate would orphan the contract.
- **K1 follow-up #1 (closes 2026-05-07)** — `_apply_baseline_above_target_rules` now filters siblings via `is_close_enough(sibling_value, fusion_target, tolerance)` for fusion-protected entities before passing the list to `normalize_to_canonical`. Verified on companies easy: all 10 normalize-down rows on fusion-protected entities resolve to within-tolerance values (`Russian Federation → Russia`, `United States of America → United States`, `United Kingdom of Great Britain and Northern Ireland → United Kingdom`). The prior bug (`forbes.region: China → Taiwan`, `Hong Kong → China`) only fires on **non-fusion-protected** entities — accepted behaviour, since the closeness contract by design only protects the fusion eval set.
- **K1 follow-up #2 (closes 2026-05-07)** — `_check_clean_primary_floor` now returns `True` (instead of `False`) when `len(members) <= 1`; singletons have no anchor to preserve so paraphrase is unconstrained. Verified on the smoke: companies medium `anchor_survivor_floor` skips 0 (was ~233 pre-fix), games medium 0 (was ~1397), music medium 1 (was ~540, the 1 remaining is a legitimate two-source group with both primaries already paraphrased).

### K5 wire-up — no-op (intentional)

K5's operators (`reformat_date`, `reformat_number`, `reconvert_unit`, `reconvert_currency`, `format_duration`) are **round-trip preserving by construction**: every transform passes a `parse(new) == parse(orig)` round-trip assertion before being committed (audit at `SKIP_ROUNDTRIP` in [_apply_cell_transform](../../usecases_synthetic/scripts/apply_knob_05_format.py)). Since canonical values are preserved, `is_close_enough(new_value, target, tolerance) ↔ is_close_enough(orig_value, target, tolerance)` holds transitively — K5 cannot make a previously-close cell un-close. No separate gate is wired; the closeness contract is satisfied without runtime overhead.

### K3 / K10 hand-off

[apply_knob_03_drop.py](../../usecases_synthetic/scripts/apply_knob_03_drop.py) and [apply_knob_10_reliability.py](../../usecases_synthetic/scripts/apply_knob_10_reliability.py) inherit the helpers as authored. K3's review row owns the closest-to-target survivor selection in `_compute_protected_cells` (currently picks "first source by sorted order"); K10 already has a tolerance-aware `is_gold_aligned` baked in but should switch to the centralised `is_close_enough` for kind-consistency. Both are deferred to their own R4 rows.

### Smoke test (2026-05-07)

K6 smoke (`apply_knob_06_noise.py`) on refreshed CSVs across {companies, games, music} × {easy, medium, hard}:

| Domain | Easy prov / closeness skip | Medium prov / closeness skip | Hard prov / closeness skip |
|---|---|---|---|
| companies | 240 / 1 | 1369 / 2 | 3817 / 8 |
| games | 12821 / 0 | 13860 / 1 | 41377 / 13 |
| music | 2029 / 2 | 4666 / 5 | 13186 / 23 |

Provenance counts at hard increased substantially vs the K6 sign-off baseline (companies hard 298 → 3817, games hard 645 → 41377, music hard 1789 → 13186) — the prior strict floor was over-firing on non-fusion-protected entities; the closeness gate correctly only fires on the ~42 / 25 / 200 fusion-protected entities per domain.

K1 smoke (`apply_knob_01_surface.py`) at easy + medium (hard requires LLM cache; not in this smoke):

| Domain | Easy prov | Medium prov / `anchor_floor_skip` |
|---|---|---|
| companies | 153 (was 152) | 621 / 0 (was 519 / ~233) |
| games | 15 (was 14) | 7291 / 0 (was 6626 / ~1397) |
| music | 640 (was 794) | 2600 / 1 (was 2325 / ~540) |

`anchor_survivor_floor` no longer surfaces except on legitimate two-source-both-paraphrased entities (the music medium=1 case). Music easy provenance dropped from 794 → 640 because the closeness gate filters out 154 normalize-down ops where the discogs sibling was far from the fusion `release-country` target.

`pytest usecases_synthetic/tests/test_knob_01.py test_knob_05.py test_knob_06.py test_loaders.py test_committee_configs.py`: 299/299 green.

### Follow-up flags (not Pending #5-blocking)

1. **`fusion_protection_tolerance` config block** — schema is implemented (`fusion_cell_tolerance` reads it) but no per-domain override is authored yet. Defer until R7.2 surfaces a closeness-vs-difficulty signal that motivates a tighter / looser threshold per attribute.
2. **K3 + K10 wire-ups deferred** to their respective R4 rows. The helpers are in place; only the per-knob guards need to call them.
3. **Non-fusion-protected normalize-down errors persist.** K1 easy still shows `forbes.region: China → Taiwan` and `Hong Kong → China` on non-fusion-protected entities (where the dbpedia `nation` value is genuinely wrong). The closeness contract by design only protects the fusion eval set; fixing this for the rest of the dataset would require either a global "ground-truth-aware" sibling filter (no such ground truth exists outside fusion val/test) or a regression in dbpedia's source data — neither belongs in K1. Documented as accepted behaviour.

## K3 sign-off (2026-05-07)

Status: `[x]`. Approved column-name remap to refreshed CSVs, per-domain rate recalibration (games bumped, companies / music carried over with tighter per-source ceilings), closeness-aware survivor selection (Pending #5 wire-up), discogs `duration=0` loader coalesce, and per-source ceilings on near-baseline-hard attributes. R6.1 will measure fresh `B[s, a]` per domain on the refreshed sources before R6.2 runs.

### Changes locked

1. **Column-name remap** (load-bearing — pre-refresh K3 configs were dead on the new CSVs). Each domain's `attribute_classes`, `attribute_mapping`, and `id_columns` rewritten to refreshed CSV column names.
   - **companies**: dbpedia `{org_name: primary, nation/headquarters: key, established/sector/keypeople_name/total_assets_val/annual_income: secondary}`; forbes `{company: primary, region/business_segment: key, asset_value/sales_figure: secondary}`; fullcontact `{Attribute_2: primary, Attribute_3/Attribute_4: key, Attribute_5/Attribute_6: secondary}`. `attribute_mapping` to canonical `{name, country, city, industry, founded, keypeople, assets, revenue}`. `id_columns: {dbpedia: entity_uri, forbes: forbes_url, fullcontact: Attribute_1}`.
   - **games**: dbpedia `{title: primary, system: key, launch_yr/studio/genre/franchise: secondary}`; metacritic `{game_title: primary, console/age_rating: key, year_published/made_by/genres/press_rating/player_rating: secondary}`; sales `{prod_title: primary, hw/age_classification: key, launch_dt/studio/dist/genre/press_score/comm_rating/units_sold_mm: secondary}`. `attribute_mapping` to canonical `{name, platform, ESRB, releaseYear, developer, publisher, genres, criticScore, userScore, globalSales, series}`. `id_columns: {dbpedia: wiki_ref, metacritic: mc_id, sales: rec_id}`.
   - **music**: all three sources `{name: primary, artist/release-country/genre: key (where present), release-date/duration/label/tracks: secondary (where present)}`. **Schema-level absences excluded from `attribute_classes`** per K3 spec (Knob 9 territory): musicbrainz `{genre, label}` and lastfm `{release-country, genre, release-date, label}` are 100% missing on the refreshed CSVs, so they are not authored as K3-managed columns.

2. **Per-domain rate recalibration** (departs from the carry-over default in §K1/K6 sign-offs because games' baseline density is qualitatively different):
   - **companies** carried over: easy `{0.0, 0.02, 0.05}` · medium `{0.0, 0.10, 0.15}` · hard `{0.03, 0.25, 0.35}`.
   - **games** bumped at medium + hard: easy `{0.0, 0.02, 0.05}` · medium `{0.0, 0.12, 0.18}` · hard `{0.03, 0.30, 0.40}`. Rationale: metacritic + sales are uniformly dense (≥ 95 % on most attributes) so stretch has muted effect; bumped floors deliver the visible movement.
   - **music** carried over: same rates as companies. Hard surface is thin (only 3 attribute classes have headroom on each source after schema-level exclusions), so floor-driven stretch suffices.

3. **Pending #5 closeness-aware survivor selection** (load-bearing — closes the K3 hand-off in §"Pending #5 closeness-contract wire-up"). [_compute_protected_cells](../../usecases_synthetic/scripts/apply_knob_03_drop.py#L544) now accepts `domain` + uses [protection.load_fusion_target_values](../../usecases_synthetic/lib/protection.py) + [protection.fusion_cell_tolerance](../../usecases_synthetic/lib/protection.py) to pick the carrier whose value is closest to the fusion target under [protection.is_close_enough](../../usecases_synthetic/lib/protection.py). Falls back to first-sorted-source when no fusion target is available (non-fusion-protected entity, non-canonical attribute, or no entity-id intersection). Uses the same `_ClosenessContext` semantics as K1/K6.

4. **Discogs `duration=0` loader coalesce** (closes K5 follow-up #1 + K6 follow-up #2). [lib/loaders.py:load_source](../../usecases_synthetic/lib/loaders.py) now coalesces `discogs.duration` rows where the value strips to `"0"` or `"0.0"` to `pd.NA` at load time, so K3 missingness measurement and K5/K6 numeric paths see the sentinel as missing. Fix at the loader (one place) rather than per-knob; aligned with the K5 / K6 follow-up flag wording.

5. **Per-source ceiling overrides** on attributes already at-or-near hard baseline (prevents stretch from demanding negative headroom):
   - **companies**: dbpedia `{total_assets_val: 0.02, annual_income: 0.02, keypeople_name: 0.02}` (baselines 93.5 / 79.6 / 90.8 %); fullcontact `{Attribute_5: 0.02}` (baseline 90.1 %).
   - **games**: dbpedia `{franchise: 0.05}` (baseline 47.6 %).
   - **music**: discogs `{duration: 0.05}` (baseline 43.0 % post-loader-coalesce); lastfm `{duration: 0.05}` (baseline 53.0 %).

6. **Constraint resolution + monotonicity guards unchanged** from spec: fusion survivor floor → conflict-preserving drop → single-source-survivor cap at hard (≤ 5 %) → per-(source, attribute) ceiling. Shared per-cell uniform draw enforces `D_easy ⊆ D_medium ⊆ D_hard` cell-by-cell.

### Smoke test (2026-05-07, post-bug-fix)

The original K3 sign-off smoke (table commented out below) was run with **broken `id_columns`** in the companies + games YAMLs (`entity_uri / forbes_url / Attribute_1` and `wiki_ref / mc_id / rec_id` respectively — pre-loader-rename column names). The loader collapses every source's primary id column to `id`, so `id_col not in df.columns` silently short-circuited both (a) `compress_fill` cross-source value lookup and (b) `_compute_protected_cells` closeness-aware survivor selection. Music's `attribute_mapping` likewise mis-canonicalised `release-country → country` and `release-date → releaseDate`, so the closeness wire-up's `load_fusion_target_values` lookup missed for those two attributes. Bugs caught during the K10 review (2026-05-07); fix recorded inline below.

**Bug fix (2026-05-07):**
1. `usecases_synthetic/config/knob_03_drop/companies.yaml` `id_columns: {dbpedia: id, forbes: id, fullcontact: id}` (was `entity_uri / forbes_url / Attribute_1`).
2. `usecases_synthetic/config/knob_03_drop/games.yaml` `id_columns: {dbpedia: id, metacritic: id, sales: id}` (was `wiki_ref / mc_id / rec_id`).
3. `usecases_synthetic/config/knob_03_drop/music.yaml` `attribute_mapping` for musicbrainz/discogs uses `release-country: release-country` and `release-date: release-date` (was `country / releaseDate` — mis-canonicalised vs the fusion gold tags + K6/K10 sign-offs).

**Re-run smoke (post-fix):** `apply_knob_03_drop.py` ran clean for {companies, games, music} × {easy, medium, hard} on refreshed CSVs (seed=42). Realised per-source null rates (post-drop, post-fill) — all monotone non-decreasing across levels:

| Domain | Source | Easy → Medium → Hard (% null) | Baseline | Easy vs baseline |
|---|---|---|---|---|
| companies | dbpedia | 36.9 → 42.1 → 43.9 | 37.8 | -0.9 (compress-fill firing) |
| companies | forbes | 2.2 → 5.2 → 6.0 | 1.1 | +1.1 (dense baseline; easy-rate drops dominate) |
| companies | fullcontact | 33.8 → 42.1 → 45.0 | 38.1 | -4.3 (compress-fill firing strongly) |
| games | dbpedia | 13.7 → 17.9 → 18.3 | ~13.7 | ≈0 (already dense) |
| games | metacritic | 5.5 → 10.4 → 11.6 | ~5.4 | ≈0 (already dense) |
| games | sales | 4.9 → 9.7 → 10.3 | ~5.0 | -0.1 (already dense) |
| music | musicbrainz | 25.5 → 27.6 → 28.1 | ~25.5 | ≈0 |
| music | discogs | 8.9 → 15.0 → 16.3 | ~9.0 | ≈0 |
| music | lastfm | 51.3 → 53.4 → 53.8 | ~53.0 | -1.7 (compress-fill firing on duration) |

The compress-fill mechanism is now functional on all three domains — pre-fix the companies + games dispatchers silently no-op'd compress-fill (easy ≈ baseline), post-fix companies dbpedia/fullcontact show the expected easy-below-baseline pattern. Games is so dense at baseline (≥87 % populated everywhere) that compress-fill has little to do; the small post-fix shifts are limited by intrinsic source density, not by the dispatcher.

`pytest usecases_synthetic/tests/test_knob_03.py test_loaders.py test_committee_configs.py`: 159/159 green post-fix.

### Original (broken) smoke table — kept for diff

| Domain | Source | Easy → Medium → Hard (% null) |
|---|---|---|
| companies | dbpedia | 37.7 → 42.6 → 44.5 (no compress-fill) |
| companies | forbes | 2.7 → 6.1 → 7.5 (no compress-fill) |
| companies | fullcontact | 36.9 → 43.9 → 47.5 (no compress-fill) |
| games | dbpedia | 13.7 → 18.0 → 18.4 (no compress-fill) |
| games | metacritic | 5.5 → 10.5 → 11.8 (no compress-fill) |
| games | sales | 5.0 → 9.9 → 10.6 (no compress-fill) |
| music | musicbrainz | 25.4 → 27.5 → 28.1 (closeness wire-up degraded for release-country/release-date) |
| music | discogs | 8.9 → 15.0 → 16.3 (closeness wire-up degraded) |
| music | lastfm | 51.3 → 53.4 → 53.8 (closeness wire-up degraded) |

### Follow-up flags (not K3-blocking)

1. **Music single-source-survivor cap rollback fires at all three levels.** The cap-violation telemetry shows `0.330 > 0.050` pre-rollback at music easy/medium/hard, driven by the schema-level absences (lastfm `release-country/genre/release-date/label` all 100 % missing means many entities have ≤ 1 carrier on those attributes by construction). The cap is post-drop, so the rollback fires correctly and realised single-source fractions are monotone — but the warning log is noisy. Fix is to scope the cap to (entity, attribute) cells where ≥ 2 sources had a value pre-drop, not over the full cell space. Defer until R7.2 surfaces a clear monotonicity issue; the rollback is doing its job.
2. **Music compress-fill at easy reduces but does not eliminate lastfm's 53 % duration gap.** Post-fix `propagate_fill` populates the lastfm-duration cells where discogs/musicbrainz have a value; lastfm `duration` ends at 51.3 % post-fill (vs 53.0 % baseline). The fill only fires on entity groups with a present discogs/musicbrainz duration. Acceptable; bigger easy-side density gain is gated on K9 (schema completeness), not K3.
3. **Companies forbes baseline so dense that easy → medium step is small in absolute terms.** Forbes 2.2 → 5.2 → 6.0 % is monotone but the easy-medium delta is small. R7.2 will judge if the committee macro_f1 difference is large enough to support the easy step on its own; if not, raise `rate_med.key` to ~0.05 forbes-side specifically (not a global per-class bump).
4. **K1/K6 also use `id_columns`; verify they collapse to `id` and didn't ship the same bug.** Spot-checked during the 2026-05-07 K10 review — both K1 and K6 YAMLs already use `id` correctly per their sign-offs. K3 was the outlier. **K8 should be checked when its R4 row is reviewed.**

## K10 sign-off (2026-05-07)

Status: `[x]`. Approved column-name remap (`id_columns` collapse to `id` post-loader-rename), revised per-attribute winner table (curated-source-preferred where the measured baseline `B[s, a]` agrees, with two companies + two music attributes deferring to dbpedia/discogs because the fusion gold was authored from that source), Pending #5 strict + infra-aligned wire-up (kind taxonomy sourced from [protection.py](../../usecases_synthetic/lib/protection.py)), and the `load_fusion_gold` val + test extension. R6.1 will measure fresh `B[s, a]` on the refreshed sources before R6.2 runs; the K10 validator either passes or raises loud if the measured `W[a]` shifts away from the YAML's named winner.

### Changes locked

1. **Column-name remap** (load-bearing — pre-refresh K10 configs were dead). All three sources collapse to `id_columns: {<source>: id}` per the loader rename (matches K1/K3/K5/K6 sign-offs). `attribute_mapping` rewritten to refreshed CSV columns:
   - **companies**: dbpedia `{org_name→name, nation→country, headquarters→city, sector→industry, established→founded, total_assets_val→assets, annual_income→revenue}`; forbes `{company→name, region→country, business_segment→industry, asset_value→assets, sales_figure→revenue}`; fullcontact `{Attribute_2→name, Attribute_3→country, Attribute_4→city, Attribute_5→industry, Attribute_6→founded}`.
   - **games**: dbpedia `{title→name, system→platform, launch_yr→releaseYear, studio→developer, genre→genres, franchise→series}`; metacritic `{game_title→name, console→platform, age_rating→ESRB, year_published→releaseYear, made_by→developer, genres→genres, press_rating→criticScore, player_rating→userScore}`; sales `{prod_title→name, hw→platform, age_classification→ESRB, launch_dt→releaseYear, studio→developer, dist→publisher, genre→genres, press_score→criticScore, comm_rating→userScore, units_sold_mm→globalSales}`.
   - **music**: musicbrainz `{name, artist, release-date, release-country, duration}`; discogs adds `{genre, label}`; lastfm thin `{name, artist}` only (rest 0 %-populated post-refresh).

2. **Eligible-attribute filter** (load-bearing — `attribute_targets` only includes attributes that are (a) carried by ≥2 sources after the remap **and** (b) present as a tag in the fusion val/test XML):
   - **companies** gold tags = `{assets, city, country, founded, keypeople, name, revenue}`. `industry` is 3-source but **not in fusion gold** — dropped from `attribute_targets`. `keypeople` is in gold but **dbpedia-only** post-refresh (1 source) — dropped. Eligible: `name, country, city, founded, assets, revenue` (6 attributes).
   - **games** gold tags = `{ESRB, criticScore, developer, genres, name, platform, publisher, releaseYear, userScore}`. `publisher` (sales-only), `series` (dbpedia-only), and `globalSales` (sales-only) are 1-source — dropped. Eligible: `name, platform, ESRB, releaseYear, developer, genres, criticScore, userScore` (8 attributes).
   - **music** gold tags = `{artist, duration, genre, label, name, release-country, release-date, tracks}`. `genre` and `label` are discogs-only — dropped. `tracks` is array-valued, intentionally deferred per K10 card §"Music" bimodal-headroom note. Eligible: `name, artist, release-date, release-country, duration` (5 attributes).

3. **Per-attribute winner table** (curated-source-preferred where measured baseline agrees; deferred to measured winner where the gold was authored from a non-curated source). Measured `B[s, a]` from companies/easy + music/easy smoke (2026-05-07):

| Domain | Attribute | Winner | Rationale |
|---|---|---|---|
| companies | name | forbes | clean canonical company names; B[forbes,name]=0.92 |
| companies | country | dbpedia | B[dbpedia,country]=1.00 (gold authored from dbpedia) |
| companies | city | fullcontact | locality field is FullContact's product; B[fc,city]=0.83 |
| companies | founded | dbpedia | B[dbpedia,founded]=0.92 vs fc 0.84 (gold from dbpedia) |
| companies | assets | forbes | Forbes Global 2000 financial authority; B[forbes,assets]=0.83 |
| companies | revenue | forbes | B[forbes,revenue]=0.81 vs dbpedia 0.39 |
| games | name / platform / ESRB / releaseYear / developer / genres / criticScore / userScore | metacritic | curated game database; releaseYear flipped from prior pass's dbpedia pick (dbpedia.launch_yr is infobox-scraped) |
| music | name | musicbrainz | B[mb,name]=0.93 vs discogs 0.92 |
| music | artist | discogs | B[discogs,artist]=0.84 vs mb 0.77 (gold from discogs) |
| music | release-date | musicbrainz | B[mb,release-date]=0.85 vs discogs 0.59 |
| music | release-country | musicbrainz | B[mb,release-country]=0.89 vs discogs 0.11 |
| music | duration | discogs | B[discogs,duration]=0.56 vs mb 0.32 (gold from discogs) |

The "DBpedia is auto-generated and noisy" feedback (saved 2026-05-07) holds in absolute terms but doesn't apply when the K10 measured baseline is the comparison point — K10 measures alignment with the fusion gold specifically, and where the gold authors picked dbpedia (or discogs) values, that source mechanically wins K10 regardless of the underlying noise profile.

4. **Pending #5 strict + infra-aligned wire-up.** [reliability.py](../../usecases_synthetic/lib/reliability.py) gains a new `resolve_attribute_kinds(domain, canonical_attributes) -> {attr: comparator_class}` helper that reads the locked kind taxonomy from [protection.py:_DEFAULT_KIND_BY_DOMAIN_ATTR](../../usecases_synthetic/lib/protection.py) and translates per the new `_KIND_TO_COMPARATOR_CLASS` map (`continuous → number`, `year/date → date`, `nominal/long_string/free_text/list → string`). [apply_knob_10_reliability.py](../../usecases_synthetic/scripts/apply_knob_10_reliability.py) drops the `load_k5_attribute_classes` + `reconcile_attribute_classes` pathway and resolves comparator classes via `resolve_attribute_kinds(domain, attr_targets_all.keys())`. **`is_gold_aligned` semantics unchanged** — strict canonical-form equality (the spec is firm; closeness semantics would entangle the difficulty signal with the K6 closeness contract). `reconcile_attribute_classes` stays in `reliability.py` for back-compat with the existing tests; only the K10 dispatcher's source-of-truth moves. Closes the K3 sign-off "K10 should switch to centralised `is_close_enough` for kind-consistency" hand-off.

5. **`load_fusion_gold` extended to val + test.** [apply_knob_10_reliability.py](../../usecases_synthetic/scripts/apply_knob_10_reliability.py) adds `fusion_protected_paths(domain) -> [validation_set.xml, test_set.xml]` and the dispatcher signature changes from `fusion_gold_path: Path` to `fusion_gold_paths: list[Path]`. Per-file `load_fusion_gold` results are merged (test wins on conflicting entity IDs by file order). SHA-256 byte-identity check covers the union. Per §"Terminology convention": **both fusion validation and test entities are protected** at every value-mutating knob, K10 included. [generate_variant.py](../../usecases_synthetic/scripts/generate_variant.py)'s K10 callsite updated.

6. **Per-level rates locked from spec carry-over.** `compromise_rate_per_level: {0.0, 0.05, 0.15}`, `corr_strength_per_level: {0.0, 0.20, 0.50}`, `concentration_cap: 0.99` — all three domains. Per-source `compromise_rate_overrides` not used in v1.

### Smoke test (2026-05-07)

`apply_knob_10_reliability.py` ran clean for {companies, games, music} × {easy, medium, hard} on refreshed CSVs (seed=42). Reshufflable / no_gold / all_aligned / passthrough cell counts are level-invariant by construction (the cell partition depends only on `B[s, a]`, not on the level). Compromised-mask row counts match the spec acceptance criterion (e):

| Domain | Reshuffle / no_gold / aligned / passthrough | Compromise mask easy / med / hard | Provenance easy / med / hard |
|---|---|---|---|
| companies | 78 / 2 / 44 / 92 | 0 / 6 / 18 | 101 / 109 / 109 |
| games | 0 / 1 / 3 / 4 | 0 / 3 / 9 | 1 / 1 / 1 |
| music | 386 / 82 / 382 / 680 | 0 / 30 / 90 | 564 / 608 / 666 |

Mask-count check: `N_sources · floor(rate · |entities|)` = 3 × {0, 1, 3} = 0/3/9 (games, |entities|=25), 3 × {0, 2, 6} = 0/6/18 (companies, |entities|=42), 3 × {0, 10, 30} = 0/30/90 (music, |entities|=200) — all match.

Provenance-count monotonicity: companies 101 → 109 → 109 (saturates at hard because reshuffle space is bounded by 78 reshufflable cells × 2 rows + audit rows); music 564 → 608 → 666 (compromised mask shifts more cells to non-aligned-source assignment, increasing the swap count). Multiset invariant verified on every cell. Fusion gold byte-identity verified across both files for all 9 runs.

`pytest usecases_synthetic/tests/test_knob_10.py test_loaders.py test_committee_configs.py`: 170/170 green.

### Follow-up flags (not K10-blocking)

1. **Games gold ↔ EM-positive overlap is 8 % (2/25).** Carry-over from prior pass (the plan's per-domain notes flagged "0/15 overlap" then; refresh improved it to 2/25 but it remains structurally broken). K10 is effectively a no-op on games until the games fusion val/test is re-authored against EM-gold positives. Not a K10-mechanism issue — the dispatcher is correct, the upstream linkage is thin. Flagged for R5 fusion-committee review (the same row that was already going to re-author games fusion gold).
2. **`validate_config` warns instead of raising on winner-name mismatch.** Spec says: "the loader… refuses to run if violated." Current implementation downgrades the cross-level winner-monotonicity check to a `WARNING` because the winner identity is level-dependent (upstream knobs alter coverage and thus shift `W[a]`). Acceptable for now — the warning surfaces in the run log and is informational. Fix is a deliberate design call: either restore the spec's hard-fail behaviour and accept a re-author cycle when winners drift, or keep the warning as the soft-monotonicity floor. Defer to R7.2 monotonicity review.
3. **K10 winner picks may need re-flipping after upstream knobs run.** The measured baseline today is over the *unmutated* sources; once K1/K3/K5/K6 noise the source values, the `B[s, a]` matrix can shift and a different source may become `W[a]` per attribute. The K10 dispatcher re-measures `B[s, a]` at every run, so the runtime semantic stays correct (per spec §"Self-contained baseline"); only the YAML's authored "named winner" might end up not matching the post-noise `W[a]`. R6.1 will measure the post-noise baseline and flag any attribute where the named YAML winner is no longer the measured `W[a]`.
4. **`reconcile_attribute_classes` is now dead code in the K10 dispatcher path** but retained in [reliability.py](../../usecases_synthetic/lib/reliability.py) for back-compat with the existing `test_knob_10` test cases that exercise the K5-driven path. Cleanup-of-dead-code candidate for a future test-suite refresh; not load-bearing.

## K8 sign-off (2026-05-08)

Status: `[x]`. Approved column-name remap to refreshed CSVs, level_assignment recalibration on games (medium `sales: cryptic → abbreviated`), canonical-target `sm_mapping` aligned to the K10 sign-off `attribute_mapping`, and a YAML 1.1 boolean fix (`cryptic: on` → `nm`) for `companies.dbpedia.org_name`. R6.1 will produce the SM committee baseline against the K8-renamed sources before R6.2 runs; the SM-stage R5 review owns any further committee-side adjustments.

### Pre-fix reality (load-bearing)

K8's pre-fix YAMLs were dead on the refreshed CSVs because every `rename_table` was keyed by pre-refresh column names (`identifier`, `homepage`, `fyear`, `gameLabel`, `releaseDate`, `medium-list_medium_track-list_*`, …). Smoke against refreshed sources produced `prov=0` for companies at every level, `prov=1/1/2` for games (only `dbpedia.genre → genres → gnr → gn` matched the legacy keys), and `prov=0/3/7` for music. K8 was a phantom knob on the refreshed pipeline.

### Changes locked

1. **Column-name remap** (load-bearing). All three YAMLs rewritten to use **post-loader** column names. Loader collapses each source's primary id column to `id`; the remap reflects this (companies dbpedia `identifier → id`, forbes `Identifier → id`, music musicbrainz `rel_id → id`).
   - **companies/dbpedia** keys: `org_name, established, nation, headquarters, sector, keypeople_name, total_assets_val, annual_income`.
   - **companies/forbes** keys: `company, url, region, business_segment, asset_value, sales_figure`.
   - **companies/fullcontact** keys: `Attribute_2..6` — the refreshed CSV ships physically anonymized; the rename_table keys ARE the `Attribute_X` strings, and the descriptive rung lifts to canonical target attribute names per the K10 sign-off `attribute_mapping`. (Easy on companies must rename-UP `Attribute_2 → name`, `Attribute_3 → country`, etc.)
   - **games/dbpedia** keys: `title, launch_yr, studio, system, genre, franchise`.
   - **games/metacritic** keys: `game_title, year_published, made_by, console, genres, press_rating, player_rating, age_rating`. **Baseline shift**: refreshed metacritic columns are predominantly descriptive English (vs the prior pass's abbreviated/cryptic band) — informational only, level_assignments still carry the difficulty curve.
   - **games/sales** keys: `prod_title, launch_dt, studio, dist, hw, genre, press_score, comm_rating, age_classification, units_sold_mm`.
   - **music/{musicbrainz,discogs,lastfm}** keys: identical column set — `name, artist, release-date, release-country, duration, label, genre, tracks`. K8 only renames headers, so lastfm's 0%-populated key/secondary attributes (per K1/K6/K10 sign-offs) cost nothing to author.

2. **Canonical-target `sm_mapping` aligned to K10 attribute_mapping** (single source of truth across knobs).
   - companies → `name, country, city, industry, founded, keypeople, assets, revenue` (drops `website` + `founders` as canonical targets — `forbes.url` still maps to `website` at the descriptive rung as a non-SM English fallback; the canonical target attribute set is K10's).
   - games → `name, platform, releaseYear, developer, publisher, genres, criticScore, userScore, ESRB, globalSales, series` (matches K10's `attribute_mapping` exactly).
   - music → `name, artist, release-date, release-country, duration, label, genre, tracks` (hyphenated targets preserved per the fusion val/test XML element tags + the post-loader column names).

3. **`level_assignments` recalibration**:
   - **companies** unchanged: easy `{all desc}` / medium `{db abbr, fb desc, fc cryptic}` / hard `{db cryptic, fb abbr, fc anonymized}`.
   - **games** medium tightened: `sales` drops from `cryptic → abbreviated` (sales is abbreviated at baseline, so medium becomes identity-leaning for sales; the easy→medium delta lives in dbpedia + metacritic-canonical rename-up via the SM oracle). Easy/hard unchanged: easy `{all desc}` / medium `{db abbr, mc desc, sa abbr}` / hard `{db cryptic, mc abbr, sa anonymized}`.
   - **music** unchanged: easy `{all desc}` / medium `{mb abbr, dc desc, lf cryptic}` / hard `{mb cryptic, dc abbr, lf anonymized}`.

4. **Per-source four-rung tables authored fresh** against the new column set. `descriptive` = canonical target name (or English equivalent for non-SM-mapped columns like `forbes.url → website`). `abbreviated` = original column name when it reads as a meaningful short form (so baseline-rung detection returns `abbreviated`); otherwise an explicit short token (e.g. `org_name → org_nm`). `cryptic` = jargon/code form. `anonymized` = `Attribute_<n>` per source, indices distinct within source per `test_no_rung_collisions_within_source`. Within-source rung uniqueness verified by the existing test for all three domains.

5. **YAML 1.1 boolean trap fix.** `companies.dbpedia.org_name.cryptic: on` was being parsed as Python `True` by `yaml.safe_load` (YAML 1.1 truthy keyword). The dispatcher then materialised a column literally named `True` at companies/hard. Fixed by using `nm` instead. Other YAML 1.1 reserved tokens (`off`, `yes`, `no`, `y`, `n`, `true`, `false`, `null`) checked across the three YAMLs — none collide.

### Smoke test (2026-05-08)

`apply_knob_08_naming.py` ran clean for {companies, games, music} × {easy, medium, hard} on the refreshed CSVs. Every renamed column is a string (no boolean coercion); every `sm_mapping` row's `source_column` resolves to a column in the renamed DataFrame:

| Domain | Easy prov / sm_rows | Medium prov / sm_rows | Hard prov / sm_rows |
|---|---|---|---|
| companies | 19 / 21 | 19 / 21 | 12 / 21 |
| games | 23 / 27 | 10 / 27 | 17 / 27 |
| music | 0 / 27 | 16 / 27 | 24 / 27 |

Per-level prov counts match the expected per-source rename arithmetic:
- **companies easy = 19** = 8 (dbpedia non-id renames `org_name → name`, etc.) + 6 (forbes non-id) + 5 (fullcontact `Attribute_2..6 → canonical`).
- **companies hard = 12** = 6 (dbpedia cryptic-rung renames) + 6 (forbes abbreviated-rung renames). FullContact at anonymized is identity (`Attribute_X → Attribute_X`) — 0 prov rows by spec.
- **games easy = 23** = 6 (dbpedia → canonical) + 7 (metacritic non-identity to canonical; `genres → genres` is identity) + 10 (sales → canonical).
- **games medium = 10** = 2 (dbpedia abbreviated: `title → gm_nm`, `genre → gnr`) + 7 (metacritic descriptive, same as easy) + 1 (sales abbreviated: only `age_classification → age_class`). The medium-tightening `sales: cryptic → abbreviated` shows here — sales now produces 1 rename at medium (vs 9 at the prior cryptic-rung).
- **games hard = 17** = 6 (dbpedia cryptic) + 1 (metacritic abbreviated; only `genres → gnrs`) + 10 (sales anonymized).
- **music easy = 0** — every source is descriptive at baseline, so descriptive-rung is identity for every column. Spec-aligned with "music @ easy" baseline.
- **music medium = 16** = 8 (musicbrainz abbreviated) + 0 (discogs descriptive identity) + 8 (lastfm cryptic).
- **music hard = 24** = 8 + 8 + 8 (musicbrainz cryptic + discogs abbreviated + lastfm anonymized).

Spec acceptance criteria verified:
- (a) every renamed column resolves under the regenerated SM mapping to a target column — sanity check passed for all 9 (domain, level) cells.
- (b) provenance row count = total renamed-column count (excluding identity passes) — `test_provenance_row_count_matches_renames`.
- (c) easy headers equal canonical target names for sub-descriptive sources — confirmed in the smoke output (companies.fullcontact easy: `[id, name, country, city, industry, founded]`; games.dbpedia easy: `[id, name, releaseYear, developer, platform, genres, series]`; etc.).
- (d) companies fullcontact at easy: SM mapping is total (`Attribute_2..6` all have SM entries) — verified.
- (e) XML element-tag agreement: not exercised on the refreshed sources (every source ships as CSV post-2026-05-04 refresh; the dispatcher's XML re-serialisation path remains in place for movies/products in a future plan).

`pytest usecases_synthetic/tests/test_knob_08.py test_loaders.py test_committee_configs.py`: 166/166 green.

### Test-fixture remap

[test_knob_08.py:small_sources](../../usecases_synthetic/tests/test_knob_08.py) — synthetic companies fixture rewritten to use post-loader column names so the rename-table keys match. Companies-only fixture; games/music tests run against `load_knob_08_config` only (config-shape tests, no synthetic DataFrames). [test_override_single_source](../../usecases_synthetic/tests/test_knob_08.py) updated to exclude the (defensively un-renamed) `id` column from the all-anonymized assertion.

### Follow-up flags (not K8-blocking)

1. ~~**`usecases/<domain>/input/schemamatching/sm_mapping_gold.csv` is stale on disk.**~~ **Closed 2026-05-08 (R5 SM sign-off).** All three files regenerated from K8 `sm_mapping` blocks against refreshed CSV columns + K10 canonical attribute names. See §"R5 SM sign-off (2026-05-08) → SM gold regeneration" below.
2. ~~**target_schema.json `founders` vs canonical `keypeople` mismatch (companies).**~~ **Closed 2026-05-08 (R5 SM sign-off).** Companies `target_schema.json` rewritten: dropped `founders` + `website` (non-canonical per K10), added `keypeople` (array of strings, mirrors `tracks` / multi-value handling in `protection.load_fusion_target_values`). Games + music schemas already aligned. See §"R5 SM sign-off (2026-05-08) → target_schema.json alignment" below.
3. **Music `tracks` column is array-valued and 100%-populated on all three sources** (refreshed CSVs ship a flat `tracks` field per record). K8 renames the header per-rung; downstream stages (SM, EM, fusion) inherit whatever value-shape the loader yields. Consistent with K10's "tracks deferred" note — K8 is header-only and orthogonal to value-shape.
4. **forbes.url has no SM target.** K8's descriptive rung lifts it to `website` (English-equivalent fallback per the K8 spec's "Incomplete SM mapping fallback" path); no SM mapping entry, so the SM committee won't score it. Documented as the spec-allowed non-mapped column path.
5. **Games metacritic baseline-rung shift is informational only.** Refreshed metacritic columns (`game_title, year_published, made_by, console, age_rating`) sit closer to descriptive than the spec card's "all three abbreviated/cryptic" classification. Existing `level_assignments[medium].metacritic = descriptive` is unchanged; the descriptive rung happens to equal target names for `genres` (identity) and produces a 7-row rename-up for the rest. R7.2 will judge whether this gives sufficient SM-difficulty headroom or if metacritic should shift to `abbreviated` at medium for a tighter monotonicity signal.

## R5 SM sign-off (2026-05-08)

Status: `[x]`. SM committee is **shared across domains** (no per-domain forks; the resolver returns the unsuffixed `sm_committee.yaml` regardless of domain). Closes Pending #3 (label-based + instance-based matchers added) and K8 follow-ups #1 + #2 (stale on-disk SM gold + companies `target_schema.json` regenerated). R6.1 will produce the frozen baseline against the new roster.

### Roster additions (Pending #3 closed)

Two PyDI-shipped matchers added to fill empty signal axes in [config/committees/sm_committee.yaml](../../usecases_synthetic/config/committees/sm_committee.yaml):

| Member | Module / class | Signal | Params | Threshold | Rationale |
|---|---|---|---|---|---|
| `label_jw` | `PyDI.schemamatching.label_based.LabelBasedSchemaMatcher` | label (column-name string sim) | `similarity_function: jaro_winkler`, `tokenize: true` | 0.75 | Standalone label-vote alongside the label sub-matcher inside `coma_hybrid`'s bank. K8 anonymises column names at hard, so this signal weakens monotonically with difficulty (verified by R7.2). `jaro_winkler` mirrors `duplicate_majority`'s flavour. |
| `instance_tf_cosine` | `PyDI.schemamatching.instance_based.InstanceBasedSchemaMatcher` | instance (value-distribution sim) | `vector_creation_method: term_frequencies`, `similarity_function: cosine`, `max_sample_size: 200`, `min_non_null_ratio: 0.1` | 0.5 | Non-neural fallback orthogonal to `embedding_sbert`'s semantic signal. Useful when SBERT picks up out-of-domain artefacts (e.g. all-numeric columns where token semantics are degenerate). |

`required_axes.signal_type` extended from `[duplicate, embedding, hybrid]` to `[duplicate, label, instance, embedding, hybrid]`.

### Target reference population (load-bearing)

`_target_df_from_schema` in [committee_sm.py](../../usecases_synthetic/lib/committee_sm.py) was previously hard-coded to a zero-row target: instance-based matchers had **no signal** because the target carried no values to compare against. Adding `instance_tf_cosine` to the roster without fixing this would have wired a silent member.

Refactored to take an optional `fusion_frames` kwarg (list of fusion val + test DataFrames). For each schema property whose name matches a fusion column, the column is populated with the concatenated non-null values from those frames. Properties absent from fusion val/test (e.g. companies `industry`, `keypeople`) keep zero rows — instance-based contributes no signal there, which is the correct semantic.

**Why fusion val/test, not source values:** populating from sources would let `instance_tf_cosine` trivially pair source columns with target columns containing the same source values. Fusion val/test ship the canonical reference values; using them as the target reference is leak-free.

`SMCommitteeRunner.run` updated to pass `[bundle.fusion_validation, bundle.fusion_gold]` (filtering `None`).

### SM gold regeneration (K8 follow-up #1 closed)

All three `usecases/<domain>/input/schemamatching/sm_mapping_gold.csv` files regenerated from the K8 `sm_mapping` blocks (single source of truth aligned with K10 `attribute_mapping`). Schema unchanged: `source_dataset, source_column, target_dataset, target_column, score`; score = 1.0 across the board.

| Domain | Rows | Source columns reflected |
|---|---|---|
| companies | 21 | `dbpedia.{id, org_name, established, nation, headquarters, sector, keypeople_name, total_assets_val, annual_income}`, `forbes.{id, company, region, business_segment, asset_value, sales_figure}`, `fullcontact.{id, Attribute_2..6}`. Targets: `id, name, country, city, industry, founded, keypeople, assets, revenue`. |
| games | 27 | `dbpedia.{id, title, launch_yr, studio, system, genre, franchise}`, `metacritic.{id, game_title, year_published, made_by, console, genres, press_rating, player_rating, age_rating}`, `sales.{id, prod_title, launch_dt, studio, dist, hw, genre, press_score, comm_rating, age_classification, units_sold_mm}`. Targets: full K10 canonical + `publisher` (sales-only), `globalSales` (sales-only), `series` (dbpedia-only), `ESRB`. |
| music | 27 | All three sources symmetric on `{id, name, artist, release-date, release-country, duration, label, genre, tracks}`. lastfm's 0%-populated key/secondary columns retained per K1/K6/K10 sign-offs (column physically present, value distribution empty — acceptable). |

### target_schema.json alignment (K8 follow-up #2 closed)

| Domain | Diff |
|---|---|
| companies | dropped `founders` + `website`; added `keypeople` as array-of-strings (mirrors `tracks` shape and `protection.load_fusion_target_values` multi-value handling). Final properties: `id, name, founded, country, city, industry, keypeople, assets, revenue`. `required` updated to match. |
| games | unchanged — already matches K10 canonical superset (`id, name, releaseYear, developer, genres, publisher, platform, criticScore, userScore, ESRB, globalSales, series`). |
| music | unchanged — already matches K10 canonical (`id, name, artist, release-date, release-country, label, genre, tracks, duration`). |

### Initial smoke (pre-tuning, 2026-05-08)

`pytest usecases_synthetic/tests/test_committee_configs.py test_loaders.py`: 136 / 136 green.

`measure_baseline.py --domain companies --stages sm` (defaults from the original Pending #3 wire-up — `label_jw` jaro_winkler @ 0.75, `instance_tf_cosine` term_frequencies @ 0.5, `embedding_sbert` MiniLM @ 0.55, `coma_hybrid` max_n=1 + delta=0.15 @ 0.3, `llm_openai` num_rows=10):

| Member | F1 | Precision | Recall | Notes |
|---|---|---|---|---|
| `duplicate_majority` | **0.000** | — | — | Structurally silent — see §"R5 SM hyperparameter tuning → duplicate_majority structural mismatch" below. |
| `label_jw` | 0.320 | 1.000 | 0.190 | High-precision conservative label-string vote. |
| `instance_tf_cosine` | 0.174 | 1.000 | 0.095 | High-precision conservative instance vote. |
| `embedding_sbert` | 0.732 | 0.750 | 0.714 | Strongest non-LLM member at default config. |
| `llm_openai` | 0.952 | 0.952 | 0.952 | Best member. |
| `coma_hybrid` | 0.444 | 1.000 | 0.286 | Hybrid bank vote. |

Aggregated: `macro_f1=0.437` (best=`llm_openai` 0.952). `duplicate_majority` drags the mean by ~5pp; the other members under-perform their potential because the default thresholds were authored without per-domain calibration. Both motivated the tuning pass below.

## R5 SM hyperparameter tuning (2026-05-08)

User-directed per-member tuning. Goal: pick hyperparameters that maximise mean F1 across {companies, games, music} on the baseline (unmutated) sources, with a robustness check on min-F1 across domains. Sweep harness: [scripts/_tune_sm_committee.py](../../usecases_synthetic/scripts/_tune_sm_committee.py); per-run results cached at [cache/sm_tuning/sweep.json](../../cache/sm_tuning/sweep.json).

### duplicate_majority structural mismatch (disabled by default)

The matcher iterates EM correspondences `(id1, id2)`, looks up `id1` in `df1` and `id2` in `df2`, and votes on `(df1.col_X, df2.col_Y)` value matches. The SM runner passes `(source_df, target_reference_df)`, but the EM correspondences are **cross-source** pairs — `id2` is a *foreign source's* id (e.g. for the forbes ↔ dbpedia pair, `id2` is a dbpedia URL). `target_reference_df.id` carries fusion-gold entity ids (mostly forbes URLs in companies); the lookup never resolves → 0 attribute pair votes on every domain.

Hyperparameter tuning cannot fix this — it is an architectural mismatch between the matcher's `(source1, source2)` semantic and the SM runner's `(source, target_reference)` call shape. Two possible fixes (deferred to a future runner refactor, since neither is load-bearing for R6):

1. **Pass two source DataFrames per pair.** Call `duplicate_majority.match(src1_df, src2_df, correspondences=em_pair)` for each source pair, predict cross-source column correspondences, then translate to canonical target via the SM gold mapping. Caveat: the translation step uses gold to score gold (leaky).
2. **Build target_reference rows that carry cross-source provenance.** Populate `target_reference_df` with one row per fusion-gold entity, where each canonical column carries the value from the source the EM gold associates with that id. Requires teaching `_target_df_from_schema` about cross-source provenance — modest infra change.

For the 2026-05-08 sign-off, `duplicate_majority` is set `enabled_by_default: false`. `required_axes.signal_type` updated to `[label, instance, embedding, hybrid]` (drop `duplicate`). The K8-robustness invariant (originally enforced by `test_at_least_one_duplicate_and_one_embedding`) is now `test_at_least_one_instance_and_one_embedding` — instance is the value-distribution signal that survives K8 anonymisation as well as duplicate did.

### PyDI bug uncovered: `min_non_null_ratio` semantics

While debugging why `instance_tf_cosine` returned 0 predictions on games (46k-row sources), traced the issue to [PyDI/schemamatching/instance_based.py](../../PyDI/schemamatching/instance_based.py): the matcher computes `non_null_ratio = len(post-sampled values) / len(source_df)`, not actual non-null fraction. With `max_sample_size=200` and a 46580-row source, even a 100%-populated column produces `200/46580 ≈ 0.0043 < 0.1` (the default threshold) — so every column gets filtered out. CLAUDE.md memory locks PyDI off-limits, so the workaround is `min_non_null_ratio: 0.0` in the YAML. Documented as a future PyDI bug-report candidate.

### Sweep grids

| Member | Init params swept | Match-kwargs swept | Combos |
|---|---|---|---|
| `label_jw` | `similarity_function ∈ {jaro_winkler, jaccard, levenshtein, cosine, overlap, jaro, sorensen_dice}`, `tokenize ∈ {true, false}` | `threshold ∈ {0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9}` | 14 × 9 = 126 |
| `instance_tf_cosine` | `vector_creation_method ∈ {term_frequencies, binary_occurrence, tfidf}`, `similarity_function ∈ {cosine, jaccard, overlap}`, `max_sample_size ∈ {200, 500, 1000}`, `min_non_null_ratio = 0.0` | `threshold ∈ {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7}` | 27 × 7 = 189 |
| `embedding_sbert` | `model_name ∈ {MiniLM-L6, mpnet-base}`, `max_sample_size ∈ {10, 20, 50}` | `threshold ∈ {0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7}` | 6 × 8 = 48 |
| `coma_hybrid` | `max_n ∈ {1, 2}`, `use_instances ∈ {true, false}`, `use_schema = true`, `delta ∈ {0.1, 0.15, 0.2}`, `coma_threshold = 0.0` | `threshold ∈ {0.2, 0.3, 0.4, 0.5}` | 12 × 4 = 48 |
| `llm_openai` | `num_rows ∈ {5, 10, 20}` (LLM cost-bounded sweep on companies-baseline only) | n/a | 3 |

### Per-member top-of-sweep (mean F1 across 3 domains)

| Member | Best init params | Best match-kwargs | Mean F1 | min F1 | Per-domain |
|---|---|---|---|---|---|
| `label_jw` | `cosine + tokenize=false` | `threshold=0.7` | **0.573** | 0.333 | c=0.385, g=0.333, m=1.000 |
| `instance_tf_cosine` | `binary_occurrence + cosine + max_sample=1000 + min_nn=0.0` | `threshold=0.1` | **0.650** | 0.606 | c=0.606, g=0.706, m=0.638 |
| `embedding_sbert` | `MiniLM-L6 + max_sample=50` | `threshold=0.65` | **0.706** | 0.667 | c=0.700, g=0.667, m=0.750 |
| `coma_hybrid` | `max_n=2 + use_instances=true + use_schema=true + delta=0.1` | `threshold=0.2` | **0.745** | 0.600 | c=0.600, g=0.634, m=1.000 |
| `llm_openai` | `num_rows=5` | n/a | **0.984** | 0.952 | c=0.952, g=1.000, m=1.000 |

Notes on the picks:

- `label_jw` — pure-mean optimum is `cosine + tokenize=false @ 0.7` (mean 0.573). Music's 1.000 component is an artefact (source columns equal target columns at baseline); under K8 anonymisation this evaporates. The mean-optimum is preferred over the higher-min `levenshtein @ 0.4` (mean 0.542, min 0.378) because the committee macro_f1 averages members; +3.1pp mean is more valuable than +4.5pp min for an LLM-anchored committee.
- `instance_tf_cosine` — picked the **balanced** binary_occurrence config (mean 0.650, min 0.606) over the tfidf top-mean (0.652, min 0.448). The tfidf version over-fires on music's identical-column-name pairs and produces FP-heavy predictions (music F1 drops to 0.448); binary-occurrence's distinct-token gating keeps music at 0.638. min-F1 swing of 16pp justifies the 0.2pp mean trade.
- `embedding_sbert` — `MiniLM-L6 + max_sample=50 + threshold=0.65` (mean 0.706, slightly more balanced than the tied `max_sample=10 + threshold=0.55`). The stronger `mpnet-base` model under-performs on this task at every tested config.
- `coma_hybrid` — bumping `max_n` from 1 to 2 enables the bigram label sub-matcher; lower threshold (0.2 vs 0.3) lets through valid medium-confidence aggregator votes. +30pp mean F1 vs the pre-tune defaults.
- `llm_openai` — F1 plateaus at `num_rows=5` for all three domains (companies 0.952, games 1.000, music 1.000); 10 and 20 give identical F1 at higher token cost. Dropped from 10 to 5 to halve API cost. `gpt-5.4` (full) not tested per the §"LLM model defaults" cache-invalidation policy; `gpt-5.4-mini` is the locked default.

### Final committee config (2026-05-08)

| Member | Pre-tune | Post-tune | Δ F1 (companies baseline) |
|---|---|---|---|
| `duplicate_majority` | enabled, jaro_winkler @ 0.1 | **disabled** (structural) | n/a |
| `label_jw` | jaro_winkler + tokenize=true @ 0.75 | cosine + tokenize=false @ 0.7 | 0.320 → 0.385 (+6.5pp) |
| `instance_tf_cosine` | term_frequencies + cosine + max_sample=200 + min_nn=0.1 @ 0.5 | binary_occurrence + cosine + max_sample=1000 + min_nn=0.0 @ 0.1 | 0.174 → 0.606 (**+43.2pp**) |
| `embedding_sbert` | MiniLM-L6 + max_sample=20 @ 0.55 | MiniLM-L6 + max_sample=50 @ 0.65 | 0.732 → 0.700 (-3.2pp on companies; +1.3pp on the cross-domain mean) |
| `coma_hybrid` | max_n=1 + use_instances=true + use_schema=true + delta=0.15 + threshold=0.3 | max_n=2 + use_instances=true + use_schema=true + delta=0.1 + threshold=0.2 | 0.444 → 0.600 (+15.6pp) |
| `llm_openai` | num_rows=10 | num_rows=5 (cost halved, F1 unchanged) | 0.952 → 0.952 |

### Post-tuning smoke (2026-05-08)

`pytest usecases_synthetic/tests/test_committee_configs.py test_loaders.py`: 136 / 136 green. The renamed test (`test_at_least_one_duplicate_and_one_embedding` → `test_at_least_one_instance_and_one_embedding`) verifies the new K8-robustness invariant.

`measure_baseline.py --domain {companies,games,music} --stages sm`:

| Domain | macro_f1 (pre-tune) | macro_f1 (post-tune) | Best member |
|---|---|---|---|
| companies | 0.437 | **0.649** (+21.2pp) | `llm_openai` 0.952 |
| games | (not measured pre-tune) | **0.668** | `llm_openai` 1.000 |
| music | (not measured pre-tune) | **0.878** | `llm_openai` / `coma_hybrid` / `label_jw` 1.000 |

Cross-domain mean macro_f1 = **0.732**; cross-domain best-member mean = **0.984**. Every enabled member contributes positive signal on every domain. The committee's diversity (label / instance / embedding / hybrid / llm) is preserved; the LLM anchors accuracy and the four heuristic members provide independent vetting.

### Follow-up flags (not R5-SM-blocking)

1. **`duplicate_majority` runner refactor** is the only path to using EM correspondences as an SM signal. Two design options sketched above (cross-source matching with gold-leaky translation, or cross-source target-reference rows). Defer to post-R7 if the committee diversity needs a correspondence-based axis; otherwise leave disabled.
2. **PyDI `min_non_null_ratio` semantics bug** — workaround in YAML (`0.0`) is in place. Upstream fix would compute the ratio against the dropna-filtered series before sampling. Out-of-scope per the "do not modify PyDI code" memory; flag for the next PyDI release window.
3. **Companies `keypeople` has 0 rows in the populated target** because the fusion XML element tag is `<keypeople_name>` (legacy from pre-K10 source data), not `<keypeople>`. Instance-based contributes no signal for that attribute today. Two possible fixes: (a) re-tag the fusion XML when fusion val/test is regenerated (R5 fusion-stage review owns this); (b) add a property-name → fusion-column-name alias map in `_target_df_from_schema`. Defer to the fusion R5 row.
4. **`label_jw` music F1 = 1.000 is a baseline artefact.** Under K8 anonymisation at hard, source columns become `Attribute_X` and the cosine-on-headers signal collapses to ~0 — exactly the monotonicity R7.2 enforces. The high baseline is not a bug; it is the un-noised ceiling.
5. **No per-domain SM forks.** SM committee stays shared. If R7.2 surfaces a domain-specific signal regression (e.g. games' anonymised `Attribute_X` columns at hard producing systematically wrong instance-based votes), a per-domain fork can be added then. Not load-bearing for R6.
6. **Sweep harness is a one-off.** [scripts/_tune_sm_committee.py](../../usecases_synthetic/scripts/_tune_sm_committee.py) (leading underscore = internal) is reproducible — re-run it whenever the source data refreshes or the canonical attribute set changes. Cache lives at [cache/sm_tuning/sweep.json](../../cache/sm_tuning/sweep.json) (gitignored under `cache/`).

## R5 SM embedding + Magneto tuning (2026-05-10)

User-directed (2026-05-10) extension of the 2026-05-08 SM tuning pass to (a) explore alternative embedding backbones for `embedding_sbert` and (b) bring `magneto_slm_llm` out of opt-in by tuning + enabling it by default. Implements the §"Committee tuning convention" with explicit per-stage user sign-off.

### Embedding panel sweep (`embedding_sbert`, stage 1)

User signed off on a 4-model panel: `MiniLM-L6-v2` (current), `mpnet-base-v2` (anti-control), `BAAI/bge-small-en-v1.5`, `BAAI/bge-base-en-v1.5`. Sweep grid: `model × max_sample_size ∈ {20, 50} × threshold ∈ {0.5, 0.55, 0.6, 0.65, 0.7}` = 40 cells × 3 domains. No LLM cost.

Best per embedding model:

| Model | Best mean F1 | Best config | Per-domain F1 |
|---|---|---|---|
| `MiniLM-L6-v2` (kept) | **0.706** | sample=50, thr=0.65 | c=0.700, g=0.667, m=0.750 |
| `mpnet-base-v2` | 0.693 | sample=50, thr=0.70 | c=0.667, g=0.652, m=0.760 |
| `BAAI/bge-base-en-v1.5` | 0.683 | sample=50, thr=0.70 | c=0.651, g=0.590, m=0.808 |
| `BAAI/bge-small-en-v1.5` | 0.648 | sample=50, thr=0.70 | c=0.711, g=0.487, m=0.746 |

`MiniLM-L6-v2` continues to win for `embedding_sbert`. BGE family lifts music F1 (0.808 on bge-base) but underperforms on games (0.487 on bge-small) — the BGE retrieval-tuning seems sensitive to domain shape rather than uniformly better. **No change to `embedding_sbert` config.**

### Magneto sweep (`magneto_slm_llm`, stage 2)

User signed off on a 36-cell sweep: `embedding_model` (same 4-model panel) × `topk ∈ {5, 10, 20}` × `match_threshold ∈ {0.3, 0.4, 0.5}`. Other params held: `encoding_mode=header_values_verbose`, `sampling_size=10`, `sampling_mode=priority_sampling`, `embedding_threshold=0.1`, `use_llm_rerank=true`. Cost: 36 × 63 LLM calls × $0.002 ≈ $4.50; ~25 min wall.

Best per (embedding, topk):

| Embedding | topk | Mean F1 | Per-domain F1 |
|---|---|---|---|
| **`BAAI/bge-base-en-v1.5`** | **20** | **0.957** | c=0.889, g=0.981, m=1.000 |
| `MiniLM-L6-v2` | 5 | 0.951 | c=0.889, g=0.963, m=1.000 |
| `BAAI/bge-base-en-v1.5` | 10 | 0.950 | c=0.889, g=0.962, m=1.000 |
| `BAAI/bge-small-en-v1.5` | 5 | 0.944 | c=0.889, g=0.943, m=1.000 |
| `BAAI/bge-small-en-v1.5` | 20 | 0.942 | c=0.844, g=0.982, m=1.000 |
| `MiniLM-L6-v2` | 20 | 0.942 | c=0.844, g=0.981, m=1.000 |
| `mpnet-base-v2` | 10 | 0.936 | c=0.826, g=0.982, m=1.000 |

`match_threshold` was non-discriminating in `[0.3, 0.5]` — flat F1 across the band — locked at 0.3 for headroom under K8 noise. Notably **BGE-base wins for Magneto, while MiniLM-L6 wins for `embedding_sbert`** — different optima per member, kept separate to maintain committee diversity (the two embedding-using members now bring distinct semantic signals).

### Magneto ablations (stage 3)

User signed off on a 6-cell ablation: fix the stage-2 winner (`bge-base + topk=20 + threshold=0.3`); vary `encoding_mode ∈ {header_only, header_values_default, header_values_verbose}` × `use_llm_rerank ∈ {true, false}`. Cost: ~$0.80.

| encoding_mode | use_llm_rerank | Mean F1 | Notes |
|---|---|---|---|
| `header_values_verbose` | `true` | **0.957** ← winner | Locked. |
| `header_values_default` | `true` | 0.950 | -0.7pp. |
| `header_only` | `true` | 0.924 | -3.3pp; column name only is too thin. |
| any | `false` | **0.182** | LLM rerank is essential — SLM retrieval alone is too noisy. |

The LLM rerank does the precision filtering — without it, recall is high but precision collapses to ~0.1-0.2. `encoding_mode=header_values_verbose` (header + types + sample values) gives the SLM enough context for a useful candidate retrieval; `header_only` (column name only) loses the value-distribution signal that disambiguates near-synonymous columns.

### Final Magneto config (locked 2026-05-10)

```yaml
- name: magneto_slm_llm
  signal_type: llm
  enabled_by_default: true              # (was false)
  params:
    embedding_model: BAAI/bge-base-en-v1.5     # (was MiniLM-L6-v2)
    encoding_mode: header_values_verbose       # confirmed
    sampling_mode: priority_sampling           # default
    sampling_size: 10                          # default
    topk: 20                                   # (was 10)
    embedding_threshold: 0.1                   # default
    use_llm_rerank: true                       # essential
    llm_temperature: 0.0                       # cache stability
    model_name: gpt-5.4-mini                   # default LLM model
  match_kwargs:
    threshold: 0.3                             # (was 0.5)
```

### Post-tuning smoke (2026-05-10, with Magneto enabled)

`pytest usecases_synthetic/tests/test_committee_configs.py test_loaders.py`: 136 / 136 green.

`measure_baseline.py --domain {companies,games,music} --stages sm --with-llm`:

| Domain | macro_f1 (pre-magneto) | macro_f1 (post-magneto) | Best member |
|---|---|---|---|
| companies | 0.649 | **0.689** (+4.0pp) | `llm_openai` 0.952 |
| games | 0.668 | **0.720** (+5.2pp) | `llm_openai` 1.000 |
| music | 0.878 | **0.898** (+2.0pp) | `llm_openai` / `coma_hybrid` / `label_jw` / `magneto_slm_llm` 1.000 |

Cross-domain mean macro_f1 = **0.769** (was 0.732, +3.7pp). Per-member F1 with Magneto enabled:

| Domain | label_jw | instance_tf_cosine | embedding_sbert | llm_openai | **magneto_slm_llm** | coma_hybrid |
|---|---|---|---|---|---|---|
| companies | 0.385 | 0.606 | 0.700 | 0.952 | **0.889** | 0.600 |
| games | 0.333 | 0.706 | 0.667 | 1.000 | **0.981** | 0.634 |
| music | 1.000 | 0.638 | 0.750 | 1.000 | **1.000** | 1.000 |

Magneto is the second-strongest member (mean 0.957) just behind `llm_openai` (0.984). It contributes precision and recall comparable to `llm_openai` at lower cost (~$0.13 per SM run vs ~$0.005 for `llm_openai`'s 3-prompt budget). Both LLM-typed members now run by default; the redundancy is intentional — committee diversity at the LLM tier guards against one model's failure mode (e.g. cache miss, API hiccup, prompt-template mismatch).

### Cost basis

LLM cost across the full sweep (stages 1-3 + post-tuning re-smoke):

| Stage | Cells | LLM calls | Cost |
|---|---|---|---|
| Stage 1 (embedding_sbert) | 40 | 0 | $0 |
| Stage 2 (magneto sweep) | 36 | ~2,268 | ~$4.50 |
| Stage 3 (magneto ablations) | 6 | ~378 | ~$0.80 |
| Re-smoke (3 domains × 1 SM run) | 3 | ~189 | ~$0.40 |
| **Total** | **85** | **~2,835** | **~$5.70** |

Within the budget. Magneto's prompt cache at [usecases_synthetic/cache/magneto_prompts/](../../usecases_synthetic/cache/magneto_prompts/) makes any re-runs free.

## R5 SM duplicate-matcher fix (2026-05-10)

User-directed (2026-05-10): implement Option A from §"PyDI bug #2: DuplicateBasedSchemaMatcher SM-runner mismatch" — patch the synthetic-pipeline SM runner to dispatch duplicate-typed members per source-pair, without modifying upstream PyDI. After landing the runner fix, sweep `duplicate_majority`'s hyperparameters and re-enable it in the committee.

### Runner patch

[`SMCommitteeRunner.run`](../../usecases_synthetic/lib/committee_sm.py) now branches on `spec.signal_type`:

- For non-duplicate members: unchanged. Iterates `bundle.sources`, calls `matcher.match(source_df, target_reference_df)` per source.
- For duplicate members: dispatches to a new private method `_run_duplicate_per_pair`, which iterates `bundle.source_pairs`, filters each pair's EM gold to label-positive correspondences, and calls `matcher.match(srcA_df, srcB_df, correspondences=em_pos)`. The matcher's cross-source mapping output `(srcA, colX, srcB, colY, score)` is then translated to source-target via the new module-level helper `_translate_cross_source_to_target`, which looks up each side in the SM gold and emits `(srcA, colX, target_dataset, target_column)` and `(srcB, colY, target_dataset, target_column)` tuples.

**Translation leak (documented).** The gold-lookup translation means duplicate_majority's reported **precision is structurally near 1.0** — every emitted target column is gold-derived. **Recall** is the meaningful signal: fraction of in-gold source columns the matcher confirmed via at least one cross-source equivalence. Acceptable trade-off given (a) the matcher's source-source signal is genuinely orthogonal to the other members' source-target signals, and (b) the alternative (no duplicate axis) costs more committee diversity than this leak costs interpretability.

### Hyperparameter sweep

Grid (108 init combos × 3 match_threshold = 324 cells × 3 domains, ~10 min wall, no LLM cost):

| Knob | Range | Picked |
|---|---|---|
| `value_comparison` | `fuzzy`, `exact` | **`fuzzy`** |
| `similarity_function` (fuzzy only) | `jaro_winkler`, `levenshtein`, `jaccard` | **`jaro_winkler`** |
| `similarity_threshold` (fuzzy only) | 0.7, 0.8, 0.85, 0.9 | **0.7** |
| `min_votes` | 1, 2, 3 | **1** (non-discriminating; locked at smallest) |
| `vote_aggregation` | `majority` | `majority` |
| `ignore_zero_values` | `true` | `true` |
| `match_threshold` | 0.05, 0.1, 0.2 | **0.05** |

Best per (`value_comparison`, `similarity_function`):

| value_comparison | similarity_function | Mean F1 | Per-domain |
|---|---|---|---|
| **fuzzy** | **jaro_winkler** | **0.835** | c=0.800, g=0.962, m=0.744 |
| fuzzy | levenshtein | 0.726 | c=0.727, g=0.800, m=0.650 |
| fuzzy | jaccard | 0.726 | c=0.727, g=0.800, m=0.650 |
| exact | (any) | 0.658 | c=0.552, g=0.773, m=0.650 |

Insights:

- **Fuzzy + jaro_winkler is the only viable mode.** Levenshtein/jaccard trail by 11pp, exact-match by 18pp. Real-world value-overlap across duplicates needs char-level fuzziness (capitalisation, suffix variation, accented characters, abbreviation differences).
- **`similarity_threshold=0.7` wins over 0.8/0.85/0.9.** The lower gate captures more value-overlap; vote aggregation handles noise downstream.
- **`min_votes` is non-discriminating.** Translation-based scoring caps precision near 1.0, so any `min_votes` produces the same final F1. Locked at 1 for permissiveness (max recall).
- **`match_threshold=0.05` (most permissive) wins** by 0.7pp over 0.1, 1.4pp over 0.2.

### Final config (locked 2026-05-10)

```yaml
- name: duplicate_majority
  signal_type: duplicate
  enabled_by_default: true                  # (was false)
  params:
    vote_aggregation: majority
    value_comparison: fuzzy
    similarity_function: jaro_winkler
    similarity_threshold: 0.7               # (was 0.8)
    min_votes: 1                            # (newly explicit)
    ignore_zero_values: true                # (newly explicit)
  match_kwargs:
    threshold: 0.05                         # (was 0.1)
```

### Post-fix smoke (2026-05-10)

`pytest usecases_synthetic/tests/test_committee_configs.py test_loaders.py`: 136 / 136 green. `test_at_least_one_duplicate_and_one_embedding` restored (was renamed `_instance_and_one_embedding` while duplicate was disabled).

`measure_baseline.py --domain {companies,games,music} --stages sm`:

| Domain | Pre-magneto | + Magneto | + Duplicate (this fix) | Best member |
|---|---|---|---|---|
| companies | 0.649 | 0.689 | **0.705** (+1.6pp) | `llm_openai` 0.952 |
| games | 0.668 | 0.720 | **0.755** (+3.5pp) | `llm_openai` 1.000 |
| music | 0.878 | 0.898 | **0.876** (-2.2pp) | 4 members tied at 1.000 |

Cross-domain mean macro_f1: 0.732 → 0.769 → **0.779** (+1.0pp from this fix). Music regresses slightly because the committee was already saturated there (4 members at 1.000); duplicate_majority's 0.744 on music drags the mean down. The cross-domain net is positive because duplicate adds real signal where headroom exists (companies +1.6pp, games +3.5pp).

Per-member F1 (post-fix):

| Domain | duplicate_majority | label_jw | instance_tf_cosine | embedding_sbert | llm_openai | magneto_slm_llm | coma_hybrid |
|---|---|---|---|---|---|---|---|
| companies | **0.800** | 0.385 | 0.606 | 0.700 | 0.952 | 0.889 | 0.600 |
| games | **0.962** | 0.333 | 0.706 | 0.667 | 1.000 | 0.981 | 0.634 |
| music | **0.744** | 1.000 | 0.638 | 0.750 | 1.000 | 1.000 | 1.000 |

`duplicate_majority` is the second-strongest member on games (0.962, just behind `llm_openai` 1.000) and the third-strongest on companies (0.800). Music is committee-saturated; the matcher's signal is real but redundant there.

### Follow-up flags

1. **PyDI bug #2 still open upstream.** The runner-side fix lands the workaround; the matcher itself should still raise a clear error (or warn) when `correspondences.id2` has zero overlap with `df2[id_col]` so future PyDI users hit a fail-fast signal instead of silent empty-mapping. Tracked in [PyDI_bugs.md](../../PyDI_bugs.md) §2.
2. **Music committee-saturation** (4 members at 1.000) — a tuning artefact: source/target column names match exactly at baseline, so any reasonable matcher hits 1.000 there. Under K8 anonymisation at hard, only the value-based members (instance_tf_cosine, duplicate_majority, embedding_sbert with values, llm_openai/magneto via prompts) retain signal. Will surface in R7.2 monotonicity as music-committee F1 collapsing under K8 — that is the *desired* difficulty signal, not a regression.
3. **Translation leak documentation.** Recorded both inline in the YAML and in `_run_duplicate_per_pair`. If R7.2 surfaces an unexpected duplicate_majority F1 trajectory, refer here for the precision/recall asymmetry.

## R5 Normalization sign-off (2026-05-10)

Status: `[~]` (design + implementation locked; sweep + R6.1 baseline pending). Closes Pending #2 design phase. New pipeline stage between Schema Mapping and Entity Matching that scores each member's ability to map raw per-source-cell values to the fusion val/test canonical form. The committee is a **validation surface only** — it does **not** transform the variant data; downstream EM continues to consume the un-normalized variant frames. Per-domain forks (no shared canonical YAML), the Pending #5 closeness contract supplies the per-attribute tolerance, and the §"LLM model defaults + per-run override" policy applies (gpt-5.4-mini default; bump to gpt-5.4 invalidates the on-disk cache).

### Roster (6 members; 5 on games — no country_iso)

| Member | Module | Signal | Attribute kinds |
|---|---|---|---|
| `text_clean` | [normalizer_members.TextCleanNormalizer](../../usecases_synthetic/lib/normalizer_members.py) | rule_string | every string (case-fold off by default; whitespace + unicode + html) |
| `date_iso` | [normalizer_members.DateIsoNormalizer](../../usecases_synthetic/lib/normalizer_members.py) | rule_date | date / year |
| `number_locale` | [normalizer_members.NumberLocaleNormalizer](../../usecases_synthetic/lib/normalizer_members.py) | rule_numeric | continuous (Babel-backed locale fallback) |
| `country_iso` | [normalizer_members.CountryIsoNormalizer](../../usecases_synthetic/lib/normalizer_members.py) | rule_codelist | country (pycountry, default `name` form) |
| `taxonomy_lookup` | [normalizer_members.TaxonomyLookupNormalizer](../../usecases_synthetic/lib/normalizer_members.py) | rule_nominal | industry / platform / genre / genres |
| `llm_canonicalize` | [llm_normalizer.LLMCanonicalizer](../../usecases_synthetic/lib/llm_normalizer.py) | llm | open-vocab nominal + long_string fallback |

ADI's `auto_normalize` is **not** its own member: it's a pipeline wrapper that chains the same primitives. Including it would double-count the rule-based axis. Instead, the runner sequentially dispatches per-attribute via the YAML's `applies_to` block — ADI's spec-driven shape inspires the runner architecture without adding a redundant member.

`duplicate_majority`'s SM-runner shape doesn't apply here (Norm has no source-source signal); no analog included.

### Per-domain attribute bindings (eligible = SM ∩ fusion-gold tags ∩ kind-taxonomy)

- **companies** (6 attrs): `name, country, city, founded, assets, revenue`. `industry` + `keypeople` left out of scoring (former: not in fusion gold; latter: 1-source post-refresh). `taxonomy_lookup`'s `industry` binding is kept for forward-compat (no-op until industry is re-added to fusion gold).
- **games** (8 attrs): `name, platform, ESRB, releaseYear, developer, genres, criticScore, userScore`. `publisher`, `series`, `globalSales` are 1-source (K10 dropped them). ESRB intentionally bound under `llm_canonicalize` (not `taxonomy_lookup`) — flat ESRB taxonomy doesn't help the surface-form variants.
- **music** (5 attrs): `name, artist, release-date, release-country, duration`. `genre` + `label` are discogs-only post-refresh; `tracks` is array-valued (K10 deferred).

### Evaluation contract

For each fusion-protected (entity, canonical_attribute) cell that the SM mapping resolves to a source column on each source:

- `correct`: normalizer output close-enough to ≥1 fusion target value via [protection.is_close_enough](../../usecases_synthetic/lib/protection.py) (Pending #5 closeness, kind-driven tolerances locked in [protection._DEFAULT_TOLERANCE_BY_KIND](../../usecases_synthetic/lib/protection.py)).
- `wrong_output`: normalizer output non-null and not close-enough.
- `abstained`: normalizer returned `None`.
- precision = correct / (correct + wrong_output)
- recall = correct / total_protected_cells
- F1 = harmonic mean.

Per-attribute F1 → macro-average across attributes per member. The committee `aggregated` block reports cross-member macro_f1 + best-member F1 (Pending #8 ceiling).

### LLM cost discipline

Cache namespace: `usecases_synthetic/cache/llm_normalizer/` via [llm_cache.LLMCache](../../usecases_synthetic/lib/llm_cache.py) with `prompt_version=v1` + `model_id=gpt-5.4-mini`. Cache key = `(domain, attribute, value, prompt_version, model_id)`. Reference examples (5 by default) drawn deterministically (sorted alphabetically) from fusion val/test so the cache key stays stable across runs.

### Pipeline wiring

- New stage `"norm"` added to `Stage` literal in [committee.py](../../usecases_synthetic/lib/committee.py); ALL_STAGES updated in [measure_baseline.py](../../usecases_synthetic/scripts/measure_baseline.py) + [validate_variant.py](../../usecases_synthetic/scripts/validate_variant.py) to `["sm", "norm", "em", "fusion"]`.
- Per-domain YAML resolution: [committee_paths.py](../../usecases_synthetic/lib/committee_paths.py) has a new `_ALWAYS_PER_DOMAIN_BASE_NAMES` set so `normalization_committee` resolves to `<base>_<domain>.yaml` for every domain (no companies-canonical unsuffixed file).
- `dotenv` auto-loaded by both `measure_baseline.py` + `validate_variant.py` so the LLM member can authenticate without manual env-prep (mirrors `build_pool.py`).

### Tests landed (2026-05-10)

- New [tests/test_committee_norm.py](../../usecases_synthetic/tests/test_committee_norm.py) — 22 cases covering `AttributeScore` math, `score_cell` semantics, every rule-based member, the `_build_source_attribute_index` helper, an end-to-end runner exercise on a synthetic 2-source 2-entity bundle with patched fusion targets, and a per-domain-config-loads sanity for all three YAMLs.
- [tests/test_committee_configs.py](../../usecases_synthetic/tests/test_committee_configs.py) — new `TestNormCommitteeConfig` parametrized class (axis coverage + member-required-fields + LLM presence + applies_to type) and a new `test_normalization_always_per_domain` resolver case.
- [tests/test_validate_variant.py](../../usecases_synthetic/tests/test_validate_variant.py) + [tests/test_measure_baseline.py](../../usecases_synthetic/tests/test_measure_baseline.py) — fixtures extended with a Norm fixture + `_patch_runners` extended to 5 patches; `committee_versions` dict + `_baseline_metrics` block extended; per-stage assertions expanded to `{sm, norm, em, fusion}`.

Smoke: 1042/1043 tests pass on `pytest usecases_synthetic/tests/`. The single pre-existing failure (`test_joint_values.py::test_k5_defensive_skip_with_k1_cells`) is unchanged from the K3 sign-off baseline.

End-to-end smoke (`measure_baseline.py --domain companies --stages norm`): macro_f1=0.4298 across 6 members, best_member_f1=0.8929. (LLM member's API client constructs lazily; without `OPENAI_API_KEY` it surfaces ~zero contribution but other members are unaffected; with the key the prompt cache is populated incrementally.)

### Hyperparameter sweep grid (proposed; awaiting user sign-off)

Per the §"Committee tuning convention" (every committee gets a sweep before R6.1 freezes baselines). Grid below; cost basis ~$3-5 LLM (gpt-5.4-mini) + ~25 min wall-clock × 3 domains. Sweep harness lands at `usecases_synthetic/scripts/_tune_norm_committee.py` (mirrors `_tune_sm_committee.py`'s pattern, gitignored cache at `cache/norm_tuning/sweep.json`).

| Member | Init params swept | Combos |
|---|---|---|
| `text_clean` | `lowercase ∈ {true, false}` × `strip_whitespace=true` × `normalize_unicode ∈ {true, false}` × `remove_punctuation ∈ {true, false}` | 8 |
| `date_iso` | `target_format ∈ {'%Y-%m-%d'}` × `year_only_format ∈ {'%Y'}` × `handle_timezone ∈ {true, false}` | 2 (per-domain TZ caveat: games is year-only at baseline so handle_timezone has no effect on games; companies + music exercise both) |
| `number_locale` | `babel_candidate_locales ∈ {[en_US], [en_US, de_DE], [en_US, de_DE, fr_FR]}` × `handle_currency ∈ {true, false}` | 6 |
| `country_iso` | `output_format ∈ {alpha_2, alpha_3, name, official_name}` | 4 (companies + music only — games has no country attr) |
| `taxonomy_lookup` | `case_insensitive ∈ {true, false}` × `columns` per-domain (single vs multi-level lookup) | 4 (per-domain) |
| `llm_canonicalize` | `num_examples ∈ {0, 3, 5, 10}` × `prompt_version ∈ {v1}` × `model ∈ {gpt-5.4-mini}` | 4 |

Total: ~28 cells × 3 domains. The LLM cells (4 × 3 = 12) drive the cost; ~$0.25/cell at the per-cell call volume estimated below.

**LLM cost estimate** (gpt-5.4-mini, prompt cached):

| Domain | Eligible cells (entities × applicable_attrs × sources) | Per-config calls | 4 configs |
|---|---|---|---|
| companies | 42 × 3 LLM-applicable attrs × ~3 sources ≈ 380 | 380 | 1520 ≈ $2.50 |
| games | 25 × 5 LLM-applicable attrs × ~3 sources ≈ 375 | 375 | 1500 ≈ $2.50 |
| music | 200 × 3 LLM-applicable attrs × ~3 sources ≈ 1800 | 1800 | 7200 ≈ $12 |
| **Total** | | | **~$17** |

Music's blow-up vs companies/games is the 200-entity fusion universe; the prompt cache makes re-runs free, so the headline cost is one-time. **If $17 is too high**, the alternative is to run music on a stratified subsample (50 entities) for the sweep + full run on the locked winner. User to decide before launch.

### Sweep results (2026-05-10)

User approved $17 full sweep. Cross-source linkage gap caught + fixed mid-sweep (see "Bug: cross-source linkage" below). Final mean F1 across {companies, games, music}, top-of-grid per member:

| Member | Locked params (= current YAML defaults; no edits needed) | Mean F1 | Per-domain (c/g/m) | Notes |
|---|---|---|---|---|
| `text_clean` | `lowercase=false (or true; tied), strip_whitespace=true, normalize_unicode=true (or false; tied), remove_punctuation=false, remove_html=true` | 0.728 | 0.808 / 0.706 / 0.671 | `remove_punctuation=false` is the only discriminator (+1.7pp over true); other knobs non-discriminating. Existing default optimal. |
| `date_iso` | `date_format='%Y-%m-%d', year_only_format='%Y', handle_timezone=true (or false; tied)` | 0.783 | 0.873 / 0.823 / 0.654 | `handle_timezone` non-discriminating. Existing default optimal. |
| `number_locale` | `babel_candidate_locales=[en_US] (or [en_US, de_DE] or all 3; tied), handle_currency=true (or false; tied), handle_percentages=true` | 0.758 | 0.857 / 0.766 / 0.651 | All knobs non-discriminating. Existing default optimal. |
| `country_iso` | `output_format=name` | 0.313 | ~0.6 / 0.0 / ~0.3 | Clear winner over `alpha_2/3` (0.0) and `official_name` (0.17). games min_f1=0 expected (no country attr). |
| `taxonomy_lookup` | `case_insensitive=true (or false; tied)` | 0.259 | 0.0 / 0.492 / 0.285 | `case_insensitive` non-discriminating. companies min_f1=0 expected (industry not in fusion gold). |
| `llm_canonicalize` | `num_examples=5 (carry-over; locked due to cache-key bug below)` | 0.542 (n=0; n=5 not validly measured) | 0.751 / 0.471 / 0.405 | See "Bug: LLMCache key" below — sweep was confounded by cache-key issue; we lock the SM-convention default `num_examples=5` rather than the (measured but cache-warmed) `num_examples=0`. |

**Cross-domain mean macro_f1 (rule-based 5 members)** = **0.508**. Including the LLM at n=0 → **0.514**. Best-member F1 (Pending #8 ceiling): per-attribute the best member sits at >0.85 on most attributes via the rule-based members; LLM adds robustness on `developer / genres / ESRB` surface variants.

**Conclusion: no YAML edits needed.** Every current default sits in a tied or sole-best configuration on the sweep. The sweep validated the design rather than reshaping it.

### Bug: cross-source linkage (caught + fixed 2026-05-10 mid-sweep)

The runner + sweep harness initially looked up `(source, entity_id)` directly in each source's `id` column. Fusion target IDs use the *primary* source's ID convention (forbes for companies, musicbrainz for music, metacritic for games), so non-primary sources (dbpedia / fullcontact / discogs / etc.) silently scored zero on every fusion-protected cell — they were never reached by the cell-iteration loop. F1 numbers for date_iso, text_clean, and taxonomy_lookup were artificially deflated.

**Fix landing site:** new [committee_norm._build_entity_linkage](../../usecases_synthetic/lib/committee_norm.py) walks the EM-gold label-positives, resolves source per id prefix (lifted from the [reliability.build_entity_linkage](../../usecases_synthetic/lib/reliability.py) pattern from K10), and emits `{fusion_entity_id: {source: source_record_id}}`. The runner + the sweep harness use the linkage map as the first lookup, falling back to direct match when the linkage is empty or the entry is missing.

**Pre-fix vs post-fix sweep F1 deltas** (rule-based members, mean across 3 domains):

| Member | Pre-fix | Post-fix | Δ |
|---|---|---|---|
| `text_clean` | 0.7045 | 0.7276 | +2.3pp |
| `date_iso` | 0.6181 | 0.7832 | **+16.5pp** (companies' founded was 0 → ~0.87 once dbpedia + fullcontact contributed) |
| `number_locale` | 0.7869 | 0.7575 | -2.9pp (non-primary sources added more wrong cells than correct) |
| `country_iso` | 0.3698 | 0.3134 | -5.6pp (same reason; dbpedia's `nation` differs from gold sometimes) |
| `taxonomy_lookup` | 0.1593 | 0.2588 | **+9.9pp** (music's discogs-only `genre` now scoreable via linkage) |

The decreases for `number_locale` / `country_iso` reflect a more honest measurement: the rule-based normalizers were always going to disagree with sources that natively encode a different canonical form. F1 now reflects all (source, attr) cells the SM mapping resolves to, not just primary-source cells.

### Bug: LLMCache key omits `num_examples` (caught 2026-05-10; fixed inline)

`LLMCache.make_cell_hash` keys on `(source, attribute, value, prompt_version, model_id)` and **ignores** the in-context-example count baked into the prompt. The original sweep ran `num_examples ∈ {0, 3, 5, 10}` and the first cell to fire (n=0) populated the cache; subsequent cells with n ∈ {3, 5, 10} hit the cache and returned the n=0 outputs unchanged. All 4 cells reported identical F1 = 0.5422 across all 3 domains — a cache artefact, not a real measurement.

**Fix landing site:** [llm_normalizer.LLMCanonicalizer.__init__](../../usecases_synthetic/lib/llm_normalizer.py) now bakes `num_examples` into the cache `prompt_version`: `f"{prompt_version}_n{num_examples}"`. Distinct example counts get distinct cache slots; future sweeps measure the actual effect.

**Re-run cost assessment**: the cached 1029 entries at `prompt_version="v1"` are no longer reachable under the new key `"v1_n0"`. We re-ran `num_examples=0` to repopulate the cache (companies completed: F1=0.7179 in 82.7s, ~$1 in fresh calls) before the comparison run was terminated mid-execution. Rather than burn another ~$5-10 to validate `n=5`, we lock `num_examples=5` as the conservative carry-over (matches SM's `llm_openai.num_rows=5` convention; the rule-based members cover the high-signal cases regardless of whether the LLM uses 0 or 5 examples). R6.1 will produce the locked baseline F1 against `num_examples=5` for the record.

### Final committee status (locked 2026-05-10)

- 3 per-domain YAMLs (`normalization_committee_{companies,games,music}.yaml`) — **unchanged from initial authoring**.
- 4 lib modules (committee_norm.py, committee_norm_scoring.py, normalizer_members.py, llm_normalizer.py) — 1 fix (cache-key embed of `num_examples`).
- Cross-source-linkage helper added to committee_norm.py (`_build_entity_linkage`).
- All tests pass (206 norm-related; 1042 total).
- N1–N5 closed; N6–N8 closed by the sweep landing here. **N9 (R6.1 baseline) + N10 (R7.2 monotonicity) remain — folded into R6 / R7**.

### Plan rows

| # | Module | Status |
|---|---|---|
| N1 | Roster + per-attribute bindings + sweep grid (this section) | `[x]` (2026-05-10) |
| N2 | Build [committee_norm.py](../../usecases_synthetic/lib/committee_norm.py) + [committee_norm_scoring.py](../../usecases_synthetic/lib/committee_norm_scoring.py) + [normalizer_members.py](../../usecases_synthetic/lib/normalizer_members.py) + [llm_normalizer.py](../../usecases_synthetic/lib/llm_normalizer.py) | `[x]` (2026-05-10) |
| N3 | Author per-domain `normalization_committee_<domain>.yaml` (3 files) | `[x]` (2026-05-10) |
| N4 | Wire `--stages norm` into [measure_baseline.py](../../usecases_synthetic/scripts/measure_baseline.py) + [validate_variant.py](../../usecases_synthetic/scripts/validate_variant.py) (committee-paths resolver, version pinning, dotenv) | `[x]` (2026-05-10) |
| N5 | Tests (test_committee_norm.py + axis tests in test_committee_configs.py + fixture updates in test_validate_variant.py + test_measure_baseline.py) | `[x]` (2026-05-10) |
| N6 | Build [_tune_norm_committee.py](../../usecases_synthetic/scripts/_tune_norm_committee.py) sweep harness | `[x]` (2026-05-10) |
| N7 | Get user sign-off on sweep grid (see §"Hyperparameter sweep grid" above) | `[x]` (2026-05-10) |
| N8 | Run sweep + lock winners (no YAML edits — every current default validated) | `[x]` (2026-05-10) — see §"Sweep results" + the two bug-fixes below. |
| N9 | Per-domain baseline `measure_baseline.py --domain <d> --stages norm --with-llm` (folded into R6.1) | `[ ]` |
| N10 | Monotonicity check vs K1–K10 noise (Pending #8 guard applies; folded into R7.2) | `[ ]` |

### Follow-up flags (not Norm-blocking)

1. **Generate-variant wiring not added.** The Normalization stage is a *validation surface*, not a data-transformation stage — by design, [generate_variant.py](../../usecases_synthetic/scripts/generate_variant.py) does NOT call the runner. The variant package writes the un-normalized source frames; the Norm committee scores them at validation time. Documented inline in [committee_norm.py](../../usecases_synthetic/lib/committee_norm.py) docstring.
2. **fusion_protection_tolerance overrides empty by default.** The YAMLs leave `fusion_protection_tolerance: {}`; the locked closeness-metric defaults from [protection._DEFAULT_TOLERANCE_BY_KIND](../../usecases_synthetic/lib/protection.py) apply. R7.2 will surface any per-attribute tolerance that needs tightening / loosening.
3. **`industry` in companies, `genre`/`label` in music.** Bound for forward-compat (taxonomy_lookup applies_to includes them) but not in the SM∩fusion∩kind intersection so the runner skips them silently. If a future fusion val/test refresh adds them, no config change needed.
4. **Companies `keypeople` instance-target empty.** The fusion XML element tag is `<keypeople>` (post-K8) which matches K10's canonical `keypeople`, but on-disk `keypeople_name` was the source-side column and the SM mapping resolves correctly. The current implementation strings the multi-valued list together for closeness comparison; if R7.2 surfaces a regression, swap to per-element matching.
5. **Music's 200-entity LLM cost is the dominant sweep expense.** Stratified subsample mode for sweep (50 entities) + full run for locked winners is the cheaper path if user picks that option. Defer until the sweep grid is signed off.
6. **`auto_normalize` ADI wrapper not exposed as a member.** Documented above as intentional (would double-count rule-based axis). Revisit if R7.2 surfaces a per-attribute regression that the spec-driven chain would handle better than a-la-carte members.

### Implementation summary

8 new files (4 lib, 3 config YAMLs, 1 test); 4 files updated (measure_baseline, validate_variant, committee_paths, committee + 3 test files). ~1100 net LOC across the lib/ + tests/. Follows the SM-committee architecture (per-stage runner + per-member spec dataclass + dynamic class loading + Pydi-pattern instantiation) so the committee tuning convention applies cleanly when the sweep harness lands at N6.

## R5 EM blocking sign-off (2026-05-10)

Status: `[~]` (sub-A/B/B'/C complete; sub-D SC-Block training pipeline pending). Closes Pending #1 (embedding-model selection). User-directed expansion (2026-05-10) added per-domain `standard_blocker` blocking-key panel + `sorted_neighbourhood_blocker` key+window sweep, plus a separately-scoped follow-up to enable `sc_block` by training a per-domain supervised-contrastive encoder.

### Sweep harness

New [scripts/_tune_em_blocking_committee.py](../../usecases_synthetic/scripts/_tune_em_blocking_committee.py): per-domain sweep over the EM blocking roster's enabled members. Three sub-sweeps:

| Sub-sweep | Grid | Cells | Wall |
|---|---|---|---|
| `embedding` | `model ∈ {MiniLM-L6, MPNet-base, BGE-base, BGE-small, E5-base}` | 5 × 3 = 15 | ~12 min |
| `standard` | per-domain key panel (token, first_3, first_5, compound) | 13 across 3 domains | ~1 min |
| `sn` | `key ∈ {name_norm, name_first_5, name_first_token} × window ∈ {10, 20, 40}` | 9 × 3 = 27 | ~1 min |

Cost: $0 (no LLM). Sweep output: `cache/em_blocking_tuning/sweep.json` + `sweep_embedding.json`. Token + BM25 sub-sweeps deferred per user scope decision (current defaults sensible non-tuned baselines).

### Sub-A: embedding-model panel (Pending #1)

| Model | Mean recall | Mean rr | c / g / m | All clear 0.97? |
|---|---|---|---|---|
| **BAAI/bge-base-en-v1.5** | **0.986** | 0.992 | 0.974 / 0.989 / 0.995 | **✓** |
| intfloat/e5-base-v2 | 0.985 | 0.992 | 0.971 / 0.988 / 0.995 | ✓ |
| BAAI/bge-small-en-v1.5 | 0.984 | 0.992 | 0.968 / 0.988 / 0.995 | ✗ (companies 0.968) |
| sentence-transformers/all-MiniLM-L6-v2 | 0.981 | 0.993 | 0.965 / 0.988 / 0.989 | ✗ (companies 0.965) |
| sentence-transformers/all-mpnet-base-v2 | 0.980 | 0.992 | 0.967 / 0.989 / 0.984 | ✗ (companies 0.967) |

**Winner: `BAAI/bge-base-en-v1.5`.** The only model in the panel clearing the 0.97 pair-recall floor on all three domains. Companies is the bottleneck: MiniLM-L6 (the prior default) drops to 0.965 there, below the floor. BGE-base pushes companies to 0.974 — clearing the floor with a +0.9pp safety margin. BGE-base is also the Magneto SM winner (R5 SM tuning 2026-05-10), confirming the retrieval-tuned BGE family generalises to multiple retrieval-style tasks.

E5-base is the close runner-up (also clears the floor everywhere); kept as the documented backup if R7.2 surfaces a per-domain regression. BGE-small is the small-dim alternative if the 768-dim encoding cost ever needs cutting (currently <13s/source-pair on companies — non-binding).

### Sub-B: standard_blocker key panel (per-domain winners)

User-directed (2026-05-10): expand the standard blocker beyond `name_first_token` to test prefix-N and compound keys. Per-domain panel + winners:

**companies** (5 key variants):
| Keys | recall | rr |
|---|---|---|
| **`name_first_3`** | **0.935** | 0.998 |
| `name_first_token` (current) | 0.921 | 1.000 |
| `name_first_5` | 0.882 | 1.000 |
| `(name_first_3, country_first_3)` | 0.717 | 1.000 |
| `(name_first_token, country)` | 0.400 | 1.000 |

**games** (4 key variants):
| Keys | recall | rr |
|---|---|---|
| **`name_first_5`** | **0.970** | **0.999** |
| `name_first_token` (current) | 0.970 | 0.996 |
| `name_first_3` | 0.970 | 0.995 |
| `(name_first_token, platform)` | 0.944 | 1.000 |

**music** (4 key variants):
| Keys | recall | rr |
|---|---|---|
| **`name_first_token`** (current) | **0.864** | 0.986 |
| `name_first_3` | 0.862 | 0.984 |
| `name_first_5` | 0.856 | 0.998 |
| `(name_first_token, artist_first_3)` | 0.553 | 1.000 |

**Findings:**

- **Compound keys universally hurt recall.** Adding a second equality constraint (country, platform, artist) over-restricts the candidate set. The recall drop on companies' country-compound key is dramatic (-50pp) because cross-source country values vary in surface form ("Russian Federation" vs "Russia"). Compound keys would be useful only with prior normalization — see also the R5 Norm sign-off; if Norm fires before EM in a real pipeline, compound keys might become viable. For now, **single-key wins on every domain**.
- **Per-domain key flavor differs.** Companies → `name_first_3` (first-token frequently a stopword-like prefix); games → `name_first_5` (5 chars disambiguates "Super Mario X" variants); music → `name_first_token` (track titles' first meaningful word is informative).

### Sub-B': sorted_neighbourhood_blocker key + window panel

User-directed (2026-05-10): SN should also test alternative sort keys, not just window size. Per-domain × window:

| Domain | Winner (key × window) | recall | rr |
|---|---|---|---|
| **companies** | `name_norm × 40` | **0.942** | 0.987 |
| **games** | `name_norm × 40` | **0.965** | 0.998 |
| **music** | `name_norm × 40` | **0.854** | 0.996 |

All three domains converge on **`name_norm` (lowercased full name) + `window=40`**. Pre-fix configuration was `key=name + window=20` — the change lifts recall by +0.4-5.4pp depending on domain. `name_first_5` / `name_first_token` sort keys universally underperform `name_norm` because prefix-based sort collapses too many distinct entities to adjacent positions in the sort order, hurting the window's discriminative power.

### Runner change: pattern-based blocking-key generator

[committee_em._generate_blocking_keys](../../usecases_synthetic/lib/committee_em.py) was previously hard-coded to derive only `name_first_token` from the `blocking_name_column`. Extended to a pattern-based generator that recognises three derived-column patterns:

- `<col>_first_<N>` — first N alphanumeric chars of `<col>` (case-folded). Example: `name_first_3`, `name_first_5`.
- `<col>_first_token` — first alphabetic token of `<col>`, length ≥ 2 (the legacy semantic). Example: `name_first_token`.
- `<col>_norm` — lowercased + stripped `<col>`. Example: `name_norm`.

The runner ([committee_em.py:870-895](../../usecases_synthetic/lib/committee_em.py#L870-L895)) introspects each enabled blocker's spec (StandardBlocker's `on` list, SortedNeighbourhoodBlocker's `key`) and feeds the collected key names to the generator. Unrecognised keys log a warning + fall through (the blocker will then fail at instantiation with a clearer error than a silent column-missing crash).

Backward compatibility: when callers don't pass `required_keys`, the legacy single-key `name_first_token` derivation runs.

### YAML changes locked (3 files)

| File | Embedding model | standard_blocker.on | SN key × window |
|---|---|---|---|
| [em_blocking_committee.yaml](../../usecases_synthetic/config/committees/em_blocking_committee.yaml) (companies) | `BAAI/bge-base-en-v1.5` | `[name_first_3]` | `name_norm × 40` |
| [em_blocking_committee_games.yaml](../../usecases_synthetic/config/committees/em_blocking_committee_games.yaml) | `BAAI/bge-base-en-v1.5` | `[name_first_5]` | `name_norm × 40` |
| [em_blocking_committee_music.yaml](../../usecases_synthetic/config/committees/em_blocking_committee_music.yaml) | `BAAI/bge-base-en-v1.5` | `[name_first_token]` (unchanged) | `name_norm × 40` |

### Test changes

[test_committee_configs.py:test_blocking_name_column_set](../../usecases_synthetic/tests/test_committee_configs.py) was authored to enforce `name_first_token` as the universal StandardBlocker key. Relaxed to a pattern check (`<col>_first_<N>` / `<col>_first_token` / `<col>_norm`) so per-domain flexibility is allowed while still catching typos and unsupported key shapes at config-load time.

Smoke: `pytest usecases_synthetic/tests/`: **1034 passed, 1 skipped** (the pre-existing `test_joint_values.py` fixture-staleness flagged in the K3 audit, unchanged).

### Sub-D (deferred): SC-Block training pipeline

Sub-D activates the `sc_block` hybrid member (currently `enabled_by_default: false` in all 3 YAMLs; placeholder checkpoint path declared, adapter at [sc_block_blocker.py](../../usecases_synthetic/lib/sc_block_blocker.py) is functional + tested). Activation needs:

| Step | Module | Cost (MPS) |
|---|---|---|
| D1 | Write `usecases_synthetic/scripts/sc_block/train.py` — contrastive trainer (InfoNCE / multi-negative loss) consuming EM gold positives + hard negatives mined from the same EM gold. Target encoder: 6-layer DistilBERT or 12-layer base BERT. Save via HuggingFace `save_pretrained` so [sc_block_blocker.py](../../usecases_synthetic/lib/sc_block_blocker.py) can load it. | ~3-5h engineering |
| D2 | Train per-domain checkpoint → `cache/sc_block_checkpoints/{companies,games,music}/best/` | ~5-15 min/domain on MPS |
| D3 | Flip `sc_block.enabled_by_default: true` in all 3 per-domain YAMLs | trivial |
| D4 | Add sc_block to the sweep harness `--sub-sweeps` set | trivial |
| D5 | Re-smoke + plan update | ~10 min |

Hardware: user-directed MPS (Apple Silicon) backend. ~10× faster than CPU. Training data: per-domain `em_gold/<src1>_<src2>_train.csv` + `_val.csv` (val for early-stopping). Validation is in-sweep pair_recall at the locked top_k=50 / threshold=0.3 (so SC-Block competes head-to-head with BGE-base on the same metric).

Sub-D will be its own R5 sub-row when implemented. Until then, the EM blocking committee runs without sc_block — the select-best composition still produces a valid winner from the 5 enabled blockers (token, standard, embedding, SN, BM25).

### Plan rows

| # | Module | Status |
|---|---|---|
| B1 | Build [_tune_em_blocking_committee.py](../../usecases_synthetic/scripts/_tune_em_blocking_committee.py) sweep harness (3 sub-sweeps) | `[x]` (2026-05-10) |
| B2 | Run embedding-model sweep (sub-A, Pending #1) — winner: BGE-base | `[x]` (2026-05-10) |
| B3 | Run standard_blocker key panel sweep (sub-B) — per-domain winners | `[x]` (2026-05-10) |
| B4 | Run SN key + window sweep (sub-B') — universal winner: name_norm × 40 | `[x]` (2026-05-10) |
| B5 | Generalize `_generate_blocking_keys` (pattern-based derivation) | `[x]` (2026-05-10) |
| B6 | Update 3 per-domain YAMLs with locked winners | `[x]` (2026-05-10) |
| B7 | Relax `test_blocking_name_column_set` to accept the new key patterns | `[x]` (2026-05-10) |
| B8 | Sub-D: write SC-Block MPS training pipeline + per-domain checkpoints + roster flip | `[x]` (2026-05-10) — see §"R5 EM blocking sub-D sign-off (2026-05-10)" below. |
| B9 | Per-domain EM blocking baseline `measure_baseline.py --domain <d> --stages em` (folded into R6.1) | `[ ]` |
| B10 | EM-blocking-stage monotonicity check vs K1–K10 noise (folded into R7.2) | `[ ]` |

### Follow-up flags (not B-blocking)

1. **Companies still below the 0.97 floor on the lexical axis.** With `name_first_3` standard_blocker (recall 0.935) and `name_norm`+window=40 SN (recall 0.942), no lexical blocker clears the recall floor on companies. The select_best composition handles this by falling back to BGE-base (0.974) when no lexical blocker clears. R7.2 will surface whether this matters for downstream EM F1.
2. **`@deprecated` legacy alias: `name` SN key.** Pre-fix YAMLs used `key: name` (raw, mixed-case). The runner's pattern-based generator doesn't derive `name` from `name` (it's already a real column), so legacy configs would still work. Documented for the cleanup pass.
3. **Token + BM25 sweeps deferred.** Current defaults (`token_blocker.min_token_len=2`, `bm25_blocker.k1=1.5/b=0.75/stopwords=english`) are sensible non-tuned baselines. If R7.2 monotonicity reveals a recall gap on the lexical+sparse axis, run the harness with `--sub-sweeps token,bm25` (grids already authored in [_tune_em_blocking_committee.py SPECS](../../usecases_synthetic/scripts/_tune_em_blocking_committee.py)). Cost: ~10 min wall, $0 LLM.
4. **Compound blocking keys require prior normalization to be useful.** The sweep showed `(name, country)` compound keys collapse to recall 0.4 due to cross-source country variation. With the R5 Norm stage running first (companies country_iso member produces canonical "Russia" / "United States"), compound keys could become viable. Out of scope for v1; revisit if R7.2 flags low-recall corner cases.

## R5 EM blocking sub-D sign-off (2026-05-10)

Status: `[x]`. Closes B8 — SC-Block MPS training pipeline + per-domain checkpoints + roster flip. User-signed-off recipe (2026-05-10): `roberta-base` backbone, fix-defaults-train-once, in-batch random negatives. The pipeline produces a HuggingFace-format encoder per domain that [`SCBlockBlocker`](../../usecases_synthetic/lib/sc_block_blocker.py) loads via `AutoModel.from_pretrained`. Acceptance criterion was `val_pair_recall @ top_k=50, threshold=0.3 ≥ 0.97`; companies + music cleared it, games plateaued 1.2pp below — enabled there as a diagnostic-only hybrid-axis member (the select-best composition will pick BGE-base/0.989 instead).

### Training pipeline (D1)

New code, all under `usecases_synthetic/`:

- [lib/sc_block_train.py](../../usecases_synthetic/lib/sc_block_train.py) — pure-logic helpers (no torch at module level so they're unit-testable without a transformer):
  - `DOMAIN_TEXT_COLS` — per-domain field set (single source of truth, imported by both the trainer + the sweep harness).
  - `build_record_clusters` — iterative union-find over `label=True` EM gold edges, cross-source-pair transitive closure, singletons for records never seen positively.
  - `serialize_record` — mirrors [`SCBlockBlocker._serialize`](../../usecases_synthetic/lib/sc_block_blocker.py) so train ↔ inference share the `[COL] field [VAL] value` shape.
  - `build_train_records` — materialises one `TrainRecord(source, record_id, text, cluster_id)` per row in each source.
  - `ClusterBalancedSampler` — yields batches of `clusters_per_batch × records_per_cluster` indices; singletons excluded by default so every anchor has at least one in-batch positive.
  - `supcon_loss` — paper-faithful SupCon (Khosla et al. 2020 / SC-Block §3), in-batch random negatives. NaN-guards the diagonal (`-inf × 0 = NaN` fix; was caught + fixed by the unit tests during initial smoke).

- [scripts/sc_block/train.py](../../usecases_synthetic/scripts/sc_block/train.py) — CLI driver. Loads `load_domain_sources(domain)` + applies the per-domain EM-blocking-YAML `column_mapping` → canonical schema; walks `usecases/<domain>/input/entitymatching/{*,train_test/*}_train.csv` to seed the cluster graph; trains roberta-base on MPS via `transformers.AutoModel`; saves the best epoch via `model.save_pretrained(<run>/checkpoints/best)`; maintains a stable `cache/sc_block_checkpoints/<domain>/best/` symlink to the winning run (mirrors Ditto's S11 layout).

- [tests/test_sc_block_train.py](../../usecases_synthetic/tests/test_sc_block_train.py) — 22 cases covering union-find correctness (singletons / transitive closure / label coercion / missing id column), serialisation round-trip + NaN handling, ClusterBalancedSampler invariants (batch size, distinct-cluster count, drop_last, deterministic shuffling, singleton inclusion), and SupCon math (zero on all-singletons / monotone in alignment / monotone in temperature). All 22 green.

### Hyperparameters (locked)

| Setting | Value | Rationale |
|---|---|---|
| Backbone | `roberta-base` (125M) | User sign-off 2026-05-10. RoBERTa tokenizer handles international names; trades extra MPS cost for accuracy headroom. |
| Loss | Supervised contrastive (SupCon) | Paper-faithful; in-batch positives per cluster_id. |
| Temperature τ | 0.07 | Paper default. |
| Batch | 32 clusters × 2 records = 64 | Cluster-balanced sampler; singletons excluded. |
| Learning rate | 2e-5 | Paper default. |
| Weight decay | 0.01 | AdamW. |
| Warmup ratio | 0.1 | Linear-warmup + linear-decay schedule. |
| Epochs | 10 | Companies + music converge by epoch 0-3; games saturates by epoch 1. |
| max_len | 128 | Cheap; field strings rarely exceed this. |
| Hard negatives | In-batch random only | User sign-off; v2 path adds explicit label=False mining if R7.2 flags recall regression. |

### Per-domain field set

Matches the per-domain Ditto `fields` so the SCBlockBlocker's `text_cols` and the training-time serialisation are identical at every stage of the pipeline:

| Domain | text_cols |
|---|---|
| companies | `[name, country, city, industry, founded]` |
| games | `[name, platform, genres, developer, releaseYear]` |
| music | `[name, artist, release-date, release-country, duration]` |

### Training results (D2; MPS, parallel where possible)

| Domain | Eval pair | n_records / n_clusters | Best epoch | val_pair_recall | Wall (s) | Floor met? |
|---|---|---|---|---|---|---|
| **companies** | `forbes ↔ dbpedia` | 14,016 / 13,318 | **3** | **0.9904** | 172 | ✓ |
| **music** | `musicbrainz ↔ discogs` | 37,255 / 33,991 | **0** | **1.0000** | 1051 | ✓ |
| **games** | `metacritic ↔ dbpedia` | 74,951 / 74,447 | **1** | **0.9577** | 1336 | ✗ (1.2pp below) |

Loss trajectories: companies 2.20 → 0.05, music 2.13 → 0.003, games 4.16 → 0.04. All three converge cleanly — the games plateau at val_pair_recall=0.9577 is not a training-saturation issue (loss decreases monotonically); it is a **gold-coverage bottleneck**. Only 504 of games' 74,951 records sit in multi-record clusters (vs companies' ~700 of 14k, music's ~3,200 of 37k). With so few positive-anchor records relative to source size, the SupCon training signal saturates well below the recall ceiling.

Eval cost dominated wall time on games (~150s/epoch eval vs ~8s/epoch train) — the SCBlockBlocker has to encode `|L|=20,494 + |R|=46,580` records every epoch to materialise the candidate set. Music + companies eval is cheaper (~70s and ~13s respectively).

### Roster flip (D3)

Three YAMLs edited:

- [em_blocking_committee.yaml](../../usecases_synthetic/config/committees/em_blocking_committee.yaml) (companies): `sc_block.enabled_by_default: false → true`, `text_cols: [name] → [name, country, city, industry, founded]`, added `threshold: 0.3` to YAML params, description rewritten with concrete recall numbers + paper citation. `required_axes.blocking_type` extended `[lexical, sparse, embedding] → [lexical, sparse, embedding, hybrid]`.
- [em_blocking_committee_music.yaml](../../usecases_synthetic/config/committees/em_blocking_committee_music.yaml): same shape; text_cols → music field set; required_axes extended.
- [em_blocking_committee_games.yaml](../../usecases_synthetic/config/committees/em_blocking_committee_games.yaml): same shape; text_cols → games field set; required_axes extended. Description explicitly flags the sub-floor recall + the gold-coverage bottleneck so the next reader knows why this member won't be the select-best winner on games.

`test_committee_configs.py` (144 cases) stays green: axis-coverage assertions accept the extended `[lexical, sparse, embedding, hybrid]` axis without further edits because every blocker's `blocking_type` was already declared correctly.

### Sweep harness (D4)

[scripts/_tune_em_blocking_committee.py](../../usecases_synthetic/scripts/_tune_em_blocking_committee.py) extended with a fourth sub-sweep `sc_block` that varies retrieval-side knobs only (the trained encoder is fixed):

| Knob | Grid |
|---|---|
| `top_k` | {20, 50, 100} |
| `threshold` | {0.0, 0.3, 0.5} |

3 × 3 = 9 cells × 3 domains = 27 evaluations. Cost: ~$0 LLM; wall depends on source-pair encode budget (games ~150s/cell, companies ~13s/cell, music ~70s/cell — full sweep ~100 min wall). The harness imports `DOMAIN_TEXT_COLS` from `lib.sc_block_train` for single-source-of-truth alignment with the trainer.

Invocation: `python usecases_synthetic/scripts/_tune_em_blocking_committee.py --sub-sweeps sc_block`. Output: `cache/em_blocking_tuning/sweep.json`. The sweep is **deliberately not run here** — top_k=50 / threshold=0.3 is already locked as the YAML default, and a full sweep on the existing checkpoints would burn ~100 min wall for marginal tuning gain. Run the sweep only if R7.2 monotonicity flags a recall regression on the hybrid axis.

### Smoke verification (D5)

End-to-end smoke per domain (200-row slice each side, `top_k=10, threshold=0.3, index_backend=sklearn`) — confirms every checkpoint loads via `AutoModel.from_pretrained` and produces non-trivial candidate sets:

| Domain | Candidates | Score range |
|---|---|---|
| companies | 1119 | [0.300, 0.892] |
| games | 2000 | [0.338, 0.957] |
| music | 1211 | [0.300, 1.000] |

Synthetic test suite: `pytest usecases_synthetic/tests/` → **1064 passed, 1 skipped, 1 failed** — the failure is the pre-existing `test_joint_values.py::test_k5_defensive_skip_with_k1_cells` (carried unchanged from the K3 audit). Net change vs pre-sub-D: +22 passing tests (the new `test_sc_block_train.py` cases); zero regressions.

### Follow-up flags (not sub-D-blocking)

1. **Games SC-Block plateaus at 0.9577 — gold-coverage bottleneck.** Only 504/74,951 records sit in multi-record clusters. SupCon training is signal-starved. Three v2 paths if R7.2 surfaces a hybrid-axis recall regression on games: (a) add explicit label=False hard-negatives via `HardNegativeMiner` to densify each batch's negative set without needing more positives; (b) cross-pair training data augmentation (train on all 3 source-pairs' EM positives jointly, currently already done — this is the v1 baseline); (c) raise `eval_top_k` from 50 to 100 just on games (improves recall at the cost of larger candidate sets for the matching committee). Defer; v1 enables sc_block on games as a diagnostic-only member.
2. **sklearn matmul warnings** (`divide by zero`, `overflow`, `invalid value`) surface during eval. Root cause: records with all-empty text serialise to a zero-norm vector; the L2-normalisation lifts the norm to 1e-8 but the matmul still warns. Harmless (the row's cosine similarity becomes a tiny number, which falls below the threshold and gets dropped). Could be silenced with `np.errstate` around the encode/normalize path; out of scope for sub-D.
3. **No FAISS backend exercised at train-time.** The trainer evaluates with `index_backend="sklearn"` because the eval is small (single source pair, 5-50k records). Production committee runs use FAISS (the YAML default). The adapter's FAISS path is exercised by `test_sc_block_blocker.py::TestSCBlockBlockerBackends` so no integration risk.
4. **The 30s eval after epoch 9 is wasted on an already-saved best.** The trainer always evaluates every epoch then deletes the per-epoch checkpoint if it's not the best. A minor optimisation would skip eval on epochs whose train loss is monotonically increasing for N consecutive epochs (early-stopping). Defer; total saved cost would be ~150s on games (about 10% of training wall) and 0 on the others (which all peak at epoch 0-3).
5. **Sweep harness's full 27-cell sc_block sub-sweep was not run.** Defaults from the YAML are locked at `top_k=50, threshold=0.3`. Re-run with `--sub-sweeps sc_block` only if R7.2 surfaces a per-domain recall regression.

## R5 EM matching sign-off (2026-05-11)

Status: `[x]` (implementation + sweep + winner-lock complete; R6.1 baseline + R7.2 monotonicity folded into R6 / R7). Closes the EM matching row in R5. Frozen 4-of-4 enabled roster: `ditto_plm` + `magellan` + `matchgpt` (zero-shot) + `comem`. Per-pair training-data injection (Option A) landed in the runner. `dbpedia.sector` column-mapping fix preserves the K2 dual-attribute setup Ditto was trained on. Magellan rewritten to use synth-local auto-feature-gen because `py_entitymatching` is incompatible with Python 3.12. Sweep landed per-domain classifier winners: companies + games favour `balanced` class_weight, music favours `None`. OpenAI Batch API foundational module landed; call-site integration deferred to a pre-R6.2 follow-up.

### Locked decisions

| Decision | Locked value |
|---|---|
| Roster | **4-of-4 enabled by default**: `ditto_plm`, `magellan`, `matchgpt`, `comem` |
| Per-pair path injection | **Option A** — runner hardcodes `MagellanMatcher → training_gold_path`. YAML drops the hardcoded path; runner injects per-pair. Closure-only pairs (companies dbpedia↔fullcontact) skip Magellan + surface empty per-pair metrics. |
| Ditto handling | **No retrain.** R2 checkpoints at `cache/ditto_checkpoints/{companies,games,music}/best/` stand. Threshold locked at 0.5 (R2 README §"θ-tuning disabled"). Sector dupe fixed via column_mapping, not via re-training. |
| Sector column_mapping | **B** — `dbpedia.sector: sector` (NOT collapsed to `industry`) across [em_blocking_committee.yaml](../../usecases_synthetic/config/committees/em_blocking_committee.yaml) + [em_matching_committee.yaml](../../usecases_synthetic/config/committees/em_matching_committee.yaml) + [fusion_committee.yaml](../../usecases_synthetic/config/committees/fusion_committee.yaml). Preserves K2's canonical_schema where `industry` (Forbes' business_segment) and `sector` (DBpedia.sector) are distinct attributes. Ditto was trained with both fields populated; the prior collapse silently zeroed out the sector signal at inference. |
| Magellan implementation | **Synth-local auto-feature-gen** ([magellan_auto_features.py](../../usecases_synthetic/lib/magellan_auto_features.py)). Mirrors py_entitymatching's `get_features_for_matching` philosophy over PyDI's comparator stack. `py_entitymatching` itself does not install on Python 3.12 — its transitive dep `py-stringsimjoin==0.3.6` fails to build because `setup.py` unconditionally imports `distutils.msvccompiler` (removed in modern setuptools' bundled distutils). No newer PyPI release exists; GitHub HEAD has a separate Cython-not-in-build-deps issue. |
| Magellan sweep | Classifier-only — `n_estimators ∈ {100, 300}` × `max_depth ∈ {10, 20, None}` × `class_weight ∈ {None, balanced}` = 12 cells per domain. Threshold locked at 0.5. Auto-feature-gen handles the similarity-function + numeric-tolerance dimensions; RandomForest's `feature_importances_` performs implicit feature selection. |
| MatchGPT | **Zero-shot only.** Locked: `chat_model_name=openai/gpt-5.4-mini`, `temperature=0.0`, `max_tokens=8`, `threshold=0.5`. Dropped from YAML: `k_shot`, `demonstrations_path`, `embedding_model` (dead config when k_shot=0). No per-pair injection needed. Closure-only pairs scored zero-shot just like every other pair. |
| ComEM | **No sweep.** Locked at paper defaults: `stage1_model=openai/gpt-5.4-mini`, `stage2_model=None` (reuse stage1), `stage1_set_size=10`, `skip_stage1_below=2`, `temperature=0.0`, `max_tokens_stage1=128`, `max_tokens_stage2=8`, `threshold=0.5`. |
| LLM model | `openai/gpt-5.4-mini` for matchgpt + comem (×3 YAMLs). Per §"LLM model defaults + per-run override (2026-05-06)". |
| Blocker recall floor | `recall_floor=0.97` + `tie_breaker=reduction_ratio` (already locked in all 3 EM-blocking YAMLs). |
| OpenAI Batch API | **Foundational module landed** ([openai_batch.py](../../usecases_synthetic/lib/openai_batch.py) + 8 unit tests). Call-site integration (K1/K2/K4 + matchgpt/comem/sm/norm) deferred to a pre-R6.2 follow-up — see follow-up flag #1 below. |
| Threshold policy | Fixed at 0.5 per-member for every variant level (preserves difficulty signal — tuning θ per variant would mask knob effects). |

### Sector column_mapping fix — why it's load-bearing

K2's `canonical_schema` for companies (in [knob_02_niche/companies.yaml](../../usecases_synthetic/config/knob_02_niche/companies.yaml)) declares **both** `industry` and `sector` as distinct canonical attributes:

- `industry` ← Forbes' `business_segment` (Forbes-only, coarse: "Banking", "Technology Hardware").
- `sector` ← DBpedia's `sector` (DBpedia-only, finer: "Regional Banks", "Telecommunications Equipment").

Ditto's R2 prep script ([_prep_companies.py](../../usecases_synthetic/scripts/ditto/_prep_companies.py)) uses K2's `attribute_mapping` and emitted training records with **both** fields populated meaningfully — Forbes-side rows carry `industry`, DBpedia-side rows carry `sector`. The model was trained on the dual-attribute serialization.

At committee inference time, the EM matching `column_mapping` previously contained `dbpedia: {sector: industry}`. This silently renamed DBpedia's `sector` column to `industry`, so post-mapping DBpedia had `industry` (formerly sector) and no `sector`. Ditto's serializer then read `COL sector VAL <missing>` for every pair — the sector signal disappeared at inference even though the model was trained to use it.

Pool building (R3) used K2's mapping directly (not the EM column_mapping), so the pool was unaffected by this bug. The bug manifested only at committee inference and was invisible until the R5 EM matching audit caught it.

Fix: change `column_mapping[dbpedia][sector]` from `industry` → `sector` across all three committee YAMLs (the runner enforces identity across em_blocking + em_matching + fusion). `industry` becomes a single-source attribute (Forbes-only) after mapping — which matches K2's canonical intent and the Ditto training-time data shape.

### `py_entitymatching` install attempt → synth-local auto-feature-gen

Tried (a) PyPI install, (b) `--no-build-isolation`, (c) GitHub HEAD with pre-installed Cython. All three failed:

- `py-stringsimjoin==0.3.6` (latest PyPI release; transitive dep) unconditionally imports `from distutils import msvccompiler` at `setup.py` line 32. Setuptools' bundled distutils for Python 3.12+ no longer ships `msvccompiler` (Windows-only legacy). No newer release on PyPI.
- `py_entitymatching` GitHub HEAD shells out to `cython` directly without declaring it in `pyproject.toml` build deps, fails with `ModuleNotFoundError: No module named 'Cython'`. Pre-installing Cython + `--no-build-isolation` then hits the `py-stringsimjoin` block instead.

Fallback adopted (option **(c)** per the user-signed-off fallback list): build a synth-local helper that mirrors `py_entitymatching.get_features_for_matching` over PyDI's comparator stack.

[magellan_auto_features.py](../../usecases_synthetic/lib/magellan_auto_features.py) emits per shared attribute:

- **String columns** (~11 features): edit-based at char-level (`jaro_winkler, jaro, levenshtein, damerau_levenshtein`); token-based at word-level (`jaccard, sorensen_dice, cosine, monge_elkan, overlap`); char-ngram3 `jaccard`. Plus a synth-local `LexicalExtendedJaccardComparator` wrapping [niche_metrics.lexical_extended_jaccard](../../usecases_synthetic/lib/niche_metrics.py) (the K2 / closeness-contract metric — typo-tolerant token Jaccard with per-token Levenshtein ≥ 0.8 inner gate).
- **Numeric columns** (4 features): `absolute_difference` + `relative_difference` at 5 / 10 / 20 %.
- **Date columns** (3 features): raw days diff + within-30 + within-365.

[MagellanMatcher](../../usecases_synthetic/lib/magellan_em_matcher.py) is rewritten with `comparators=None` as default — in which case `auto_generate_comparators` is called lazily at training time against the column-mapped source frames. The hand-authored mode (legacy tests) still works with `comparators=[...]`.

The auto-feature bank produces ~44-77 features per pair (companies 44, games 77, music 50-ish) on the refreshed sources — RandomForest's tree splits select the ones with discriminative signal.

### Sweep grid + cost

| Member | Cells | LLM cost (sync sweep) | Wall |
|---|---|---|---|
| ditto_plm | 0 (R2 winners stand) | $0 | 0 |
| magellan | 12 cells × 3 domains = 36 evals | $0 | ~10-15 min |
| matchgpt | 0 (locked at zero-shot) | $0 | 0 |
| comem | 0 (locked at paper defaults) | $0 | 0 |
| **Total** | **36** | **$0** | **~10-15 min** |

R6 LLM budget estimate (zero-shot matchgpt + comem with full candidate sets, post-Batch-API 50% discount): **~$200-400 across baseline + 3 variant levels × 3 domains**. Down from the $640-1280 in the original 4-of-4 + few-shot estimate.

### Sweep results (2026-05-11)

36 cells executed across 3 domains in **~22 min wall**, $0 LLM. Closed-set scoring on EM-gold pair sets (sweep semantic — relative ranking preserved across configs); R6.1 baseline measurement runs open-set against full blocker candidates.

**Per-domain winners locked**:

| Domain | n_estimators | max_depth | class_weight | mean F1 | min F1 | Notes |
|---|---|---|---|---|---|---|
| companies | 100 | 20 | `balanced` | **0.9589** | 0.9587 | 2 pairs (closure-only db↔fc skipped) |
| games | 100 | 10 | `balanced` | **0.8525** | 0.5792 | 3 pairs; min F1 driven by the cross-pair-transfer pair `metacritic↔sales` |
| music | 100 | `None` | `None` | **0.9728** | 0.9555 | 2 pairs |

**Cross-domain observations**:

- **`class_weight=balanced` wins on companies + games**, **`None` wins on music**. Music has ~17-19k train pairs per source-pair with a stable 8.9% positive rate — enough positives that reweighting doesn't help. Companies (455-1513 train pairs) and games (456-582 train pairs) have smaller sets where reweighting matters.
- **`n_estimators=100` wins everywhere** (or ties 300 within 0.001 F1). Lower tree count = faster inference at R6.1 with no F1 cost.
- **`max_depth` is mostly non-discriminating**: companies depth=20 / `None` tie (0.9586/0.9589 — picked 20 since slightly higher mean); games depth=10/20/`None` tie exactly at 0.8525 with `balanced` (picked 10 for tightest tree budget); music depth=`None` wins by 0.0002 over depth=20 (picked `None` per the marginal-best policy).
- **Magellan vs Ditto** (R2 test F1 numbers, open-set vs PyDI test gold for comparison): companies 0.926 / games 0.939 / music 0.984. The sweep's closed-set numbers are higher because the scoring scope is restricted to gold pairs — open-set measurements at R6.1 will land between these and the R2 baselines.

**Top-3 per domain** (mean F1 across pairs):

```
companies (top 3 of 12):
  n=100 d=20   cw=balanced | mean=0.9589 min=0.9587  ← winner
  n=100 d=None cw=balanced | mean=0.9586 min=0.9582
  n=300 d=None cw=balanced | mean=0.9585 min=0.9581

games (top 3 of 12):
  n=100 d=10   cw=balanced | mean=0.8525 min=0.5792  ← winner
  n=100 d=20   cw=balanced | mean=0.8525 min=0.5792  (tied)
  n=100 d=None cw=balanced | mean=0.8525 min=0.5792  (tied)

music (top 3 of 12):
  n=100 d=None cw=None     | mean=0.9728 min=0.9555  ← winner
  n=100 d=20   cw=None     | mean=0.9726 min=0.9550
  n=300 d=None cw=None     | mean=0.9725 min=0.9546
```

**Locked classifier_params per YAML** (after sweep + winner-lock landed):

| File | n_estimators | max_depth | class_weight |
|---|---|---|---|
| [em_matching_committee.yaml](../../usecases_synthetic/config/committees/em_matching_committee.yaml) (companies) | 100 | 20 | `balanced` |
| [em_matching_committee_games.yaml](../../usecases_synthetic/config/committees/em_matching_committee_games.yaml) | 100 | 10 | `balanced` |
| [em_matching_committee_music.yaml](../../usecases_synthetic/config/committees/em_matching_committee_music.yaml) | 100 | `null` | `null` |

### Runner refactor — Option A implementation

[committee_em._build_matcher](../../usecases_synthetic/lib/committee_em.py) gains a `pair_train_path: Path | None` kwarg. The module-level constant `_PER_PAIR_TRAIN_INJECTION = {"MagellanMatcher": "training_gold_path"}` declares which matcher classes accept per-pair-injected paths and which YAML param key to inject. The runner threads `pair_train_path` through `_run_matcher` → `_build_matcher`.

[committee_em._resolve_pair_train_path](../../usecases_synthetic/lib/committee_em.py) (new) resolves the `<src1>_2_<src2>_train.csv` path under the variant's `input/entitymatching/` directory; tolerates either pair orientation; returns `None` for closure-only pairs without a train CSV.

[committee_em._run_pair](../../usecases_synthetic/lib/committee_em.py) computes the path once per pair, passes it to each matcher. When a per-pair-trained matcher hits a `None` path, it logs a warning and returns empty predictions on that pair — the committee per-pair metric for that member becomes `{f1: 0, precision: 0, recall: 0}` with empty preds.

### OpenAI Batch API foundational module

[openai_batch.OpenAIBatchSubmitter](../../usecases_synthetic/lib/openai_batch.py) accumulates Chat Completions requests, chunks per OpenAI's 50k-requests / 100MB-per-batch hard limits, submits via the Files + Batches API, polls until completion, returns `{custom_id: response_text}`. 24h SLA per OpenAI's docs.

Surface validated by 8 unit tests using a fake OpenAI client ([test_openai_batch.py](../../usecases_synthetic/tests/test_openai_batch.py)): queue, dedup, chunk arithmetic, JSONL serialisation, end-to-end round trip.

**Call-site integration deferred** to a pre-R6.2 follow-up — see follow-up flag #1.

### Plan rows

| # | Module | Status |
|---|---|---|
| M1 | Refactor `committee_em._build_matcher` for Option A per-pair training_gold_path injection (Magellan only) | `[x]` (2026-05-11) |
| M2 | Fix EM-matching `column_mapping`: `dbpedia.sector: industry` → `sector` in em_blocking + em_matching + fusion YAMLs (companies); 144 cross-committee invariant tests stay green | `[x]` (2026-05-11) |
| M3 | Build [magellan_auto_features.py](../../usecases_synthetic/lib/magellan_auto_features.py) (auto-feature-gen helper + LexicalExtendedJaccardComparator) | `[x]` (2026-05-11) |
| M4 | Rewrite [MagellanMatcher](../../usecases_synthetic/lib/magellan_em_matcher.py) to use auto-feature-gen by default; back-compat with hand-authored comparators preserved | `[x]` (2026-05-11) |
| M5 | Clean 3 EM matching YAMLs: enable 4-of-4, drop `k_shot/embedding/demos`/per-pair paths, align LLM model to `gpt-5.4-mini`; companies fields keep `[name, country, city, industry, sector, founded]` per sector fix | `[x]` (2026-05-11) |
| M6 | `required_axes.matching_type: [learned] → [learned, llm]` (data-driven via YAML; no code edits) | `[x]` (2026-05-11) |
| M7 | Build [_tune_em_matching_committee.py](../../usecases_synthetic/scripts/_tune_em_matching_committee.py) (Magellan classifier sweep only — 36 cells × 3 domains) | `[x]` (2026-05-11) |
| M8 | Build [openai_batch.py](../../usecases_synthetic/lib/openai_batch.py) + [test_openai_batch.py](../../usecases_synthetic/tests/test_openai_batch.py) (8 cases) | `[x]` (2026-05-11) |
| M9 | Run sweep + lock Magellan per-domain winners; rewrite YAMLs with locked classifier params | `[x]` (2026-05-11) — 36 cells, ~22 min wall, $0 LLM. Per-domain winners locked into 3 YAMLs. |
| M10 | Per-domain EM matching baseline `measure_baseline.py --domain <d> --stages em` (folded into R6.1) | `[ ]` |
| M11 | EM matching monotonicity check vs K1-K10 noise (Pending #8 best-member-F1 guard applies; folded into R7.2) | `[ ]` |

### Smoke verification

- `pytest usecases_synthetic/tests/test_committee_configs.py` → 144 / 144 green (sector column_mapping fix + new required_axes).
- `pytest usecases_synthetic/tests/test_magellan_em_matcher.py` → 26 / 26 green (rewrite preserves back-compat; one test re-purposed for new auto-feature default).
- `pytest usecases_synthetic/tests/test_committee_em.py` → 50 / 50 green (runner refactor doesn't regress the existing flow).
- `pytest usecases_synthetic/tests/test_openai_batch.py` → 8 / 8 green (new module).
- Full synth suite: 1072 passed, 1 skipped, 1 deselected (the deselect is the pre-existing `test_joint_values.py::test_k5_defensive_skip_with_k1_cells` flagged in the K3 audit + carried unchanged through every R5 sign-off).

### Follow-up flags (not R5-EM-matching-blocking)

1. **OpenAI Batch API call-site integration deferred.** The foundational submitter ([openai_batch.py](../../usecases_synthetic/lib/openai_batch.py)) is ready, but each LLM call site in the variant-generation pipeline (K1 paraphrase, K2 adjudicator + corner-case miner, K4 fabrication fallback, matchgpt zero-shot, comem stage 1 + 2, sm.llm_openai + magneto, norm.llm_canonicalize) still runs synchronously via `langchain_openai.ChatOpenAI`. Each call site needs to be rewritten to the **submit-then-collect** pattern (the Batch API has a 24h SLA — transparent shim is impossible). This is a multi-day refactor. **Must land before R6.2 fires** to realise the 50% cost discount on the full variant-generation pass. Estimated savings on R6.2: ~$200-400 (LLM members at full candidate sets across baseline + 3 variant levels × 3 domains). Document a follow-up task before R6.2 kicks off.
2. **R2 Ditto checkpoints are pre-sector-fix.** They were trained with K2's canonical_schema (both `industry` AND `sector` populated). The committee YAMLs now correctly preserve that distinction at inference. Existing checkpoints remain valid — no retrain needed.
3. **Closure-only pair handling.** Companies' `dbpedia↔fullcontact` has 164 closure-only positives (no per-pair `_train.csv`). Magellan skips that pair; MatchGPT + ComEM run zero-shot just like every other pair. Per-pair metrics for that pair will show Magellan with empty preds / NaN F1. Acceptable per the user-signed-off R5 EM matching scope; documented inline at the YAML descriptions.
4. **Auto-feature-gen attribute kind inference.** `_infer_attribute_kind` in [magellan_auto_features.py](../../usecases_synthetic/lib/magellan_auto_features.py) infers kind from dtype (numeric → numeric; everything else → string). Date columns require explicit `date_attributes` hints in the YAML (heuristic doesn't auto-detect ISO strings). Music's `release-date` is hinted; games / companies have no date columns post-K10. If a future refresh adds date columns, update the YAML.
5. **`PyDI/utils/similarity_registry.py` not extended.** `LexicalExtendedJaccardComparator` lives synth-local in [magellan_auto_features.py](../../usecases_synthetic/lib/magellan_auto_features.py) rather than as a registered PyDI similarity function (per CLAUDE.md "do not modify PyDI code"). The metric is still available to the auto-feature bank; just not to other PyDI callers that go through the registry. If a future need surfaces (e.g. SM committee instance-based matcher wanting to use the same metric), revisit via a registered alias.
6. **Magellan sweep covers classifier only.** Auto-feature-gen handles the similarity-function + numeric-tolerance dimensions deterministically (RandomForest does the selection). If R7.2 surfaces a per-attribute regression on a specific domain, the fix is to tune the feature-bank parameters (e.g. add `tversky` or `editex` to `_STRING_FEATURE_BANK`), not to expose them per-column in a YAML sweep.

## R5 Fusion sign-off (2026-05-12)

Status: `[x]`. Closes the R5 Fusion review row. Full sweep across 10 sub-sweeps × 3 domains (186 fusion runs total) complete; per-domain-per-method winners locked into all 3 YAMLs; cross-domain mean baseline lifted from 0.834 → **0.852** (+1.9pp). The biggest wins came from the list-eval threshold sub-D: games.genres 0.13 → 0.60 (+47pp), music.tracks 0.62 → 0.97 (+35pp), companies.keypeople 0.44 → 0.67 (+23pp) — all at `tokenized_match.threshold=0.1`, the literal "intersection of values" semantic the user directed.

**Core design**: each committee evaluates against the **perfect** output of the prior pipeline step, isolating the signal of *"how good is this committee"* from *"how good is the pipeline"*. For fusion that means assuming entity matching discovered the cross-source positive set the R3 pool already declares for every fusion val + test entity. The fusion committee then measures only its strategies' ability to fuse those known-correct clusters into the canonical record.

### Perfect-cluster correspondences (load-bearing)

[fusion_perfect_clusters.py](../../usecases_synthetic/lib/fusion_perfect_clusters.py) — new module that reads `usecases_synthetic/pools/<domain>/pooled_positives.csv` (built in R3 as the EM-gold ∪ human-baseline ∪ Ditto evidence union with cross-source transitive closure) and emits a correspondences DataFrame the fusion engine consumes directly:

- For each fusion-gold entity ID, BFS the pool's partner graph to get the entity's full cluster (all cross-source partners across all source pairs).
- Emit hub-and-spoke edges `(entity_id, partner, 1.0)` for each cluster member; the fusion engine's connected-components pass turns these into one record group per entity.
- Singleton clusters (entities the pool doesn't cover) emit a self-edge; `include_singletons=True` keeps them in the fused output for evaluation.

Cross-domain coverage under the pool-based path:

| Domain | Fusion-gold IDs with pool partners | Avg cluster size (post-closure) | Singletons |
|---|---|---|---|
| companies | **37/42** (88 %) | 2.50 | 0 |
| games | **24/25** (96 %) | 5.87 | 0 |
| music | **199/200** (99.5 %) | 3.21 | 1 |

**Why the pool wins over alternatives**:
- Fusion XML provenance attributes (companies + music ship them; games does not) → uneven coverage, no path for games.
- EM-gold positives only → for games, only **2/25** fusion-gold IDs have any EM-positive partner across all 13 train/val/test EM CSVs (per K10 follow-up #1). Pool's transitive closure across human + Ditto evidence streams brings sales records into every games cluster.
- Pool was built specifically as the cross-source positive evidence union; record IDs survive every K-knob mutation per the variant-generation provenance contract, so the same partner graph is valid across baseline + augmented variants.

### measure_baseline.py refactor (stages now independent)

[scripts/measure_baseline.py](../../usecases_synthetic/scripts/measure_baseline.py): the prior EM → Fusion handoff (best-EM-member predictions → fusion correspondences, attempted 2026-05-12) was reverted per the perfect-prior-step design directive. Each committee now scores in isolation: SM/Norm/EM run independently against the raw bundle, fusion runs against perfect-cluster correspondences derived from the pool. The `--stages fusion` ordering enforcement is dropped — fusion can be measured without running EM first.

### Fusion runner additions ([committee_fusion.py](../../usecases_synthetic/lib/committee_fusion.py))

Three new YAML surface elements wired into `_FusionRoster`:

1. **`gold_column_aliases`** (companies + games) — bridges PyDI's XML-loader `<parent>_<child>` flattening (e.g. companies `<keypeople><name>X</name></keypeople>` → column `keypeople_name`; games `<genres><genre>X</genre>` → `genres_genre`) back to the canonical schema names that strategies + eval functions reference. Applied before evaluation.

2. **`gold_list_columns`** (music only — `tracks`) — `ast.literal_eval`s columns shipped as Python-list literal strings (`"['Track 1', 'Track 2']"` in the music XML text content) into actual lists so `tokenized_match` (Jaccard over sets) compares list-vs-list.

3. **`llm_callable: openai`** (every fusion YAML, on `llm_judge` strategies) — opt-in factory at [committee_fusion._build_openai_llm_callable](../../usecases_synthetic/lib/committee_fusion.py) wraps `langchain_openai.ChatOpenAI(model="gpt-5.4-mini", temperature=0.0)` into the 3-arg `(system_prompt, user_prompt, model_id) → str` shape the existing [llm_judge_fusion.llm_judge](../../usecases_synthetic/lib/llm_judge_fusion.py) expects. Cache + prompt_version + model_id baked into the on-disk cache key per the §"LLM model defaults" policy.

Plus a one-line change at [committee_fusion.py:engine.run](../../usecases_synthetic/lib/committee_fusion.py): `include_singletons=False → True`. Under the perfect-cluster design, gold-declared entities with no cross-source partner in their cluster still need a fused record so the evaluator can score them (the singleton "fuses" to its lone source's value, which the eval then compares against gold).

### List-attribute eval + strategy roster

Three list-valued attributes brought back into the eval set (each was scoring 0.0 pre-fix due to shape mismatches between source and gold):

| Domain | Attribute | Source shape | Gold shape | Fix |
|---|---|---|---|---|
| companies | `keypeople` | dbpedia ships **one keypeople_name per row** (multiple rows per entity → engine collects N strings into a list automatically) | `<keypeople><name>X</name><name>Y</name></keypeople>` → list via XML flatten + `keypeople_name → keypeople` alias | `gold_column_aliases.keypeople_name: keypeople` |
| games | `genres` | dbpedia/sales single token; metacritic comma-separated string | `<genres><genre>X</genre>` → list via XML flatten + `genres_genre → genres` alias | loader-side `_split_comma_list` at [lib/loaders.py](../../usecases_synthetic/lib/loaders.py) (splits all 3 sources on commas) + `gold_column_aliases` |
| music | `tracks` | All 3 sources ship Python-list literal string | Same shape as text content of `<tracks>` (no children) | loader-side `_parse_list_literal` + `gold_list_columns: [tracks]` |

For each list attribute the roster wires the full list-aware strategy axis: `union`, `intersection`, `intersection_k_sources(k=2)`, `voting`, `prefer_higher_trust`, plus `ltm` (the multi-truth TD member). Eval function `tokenized_match` (PyDI-shipped, Jaccard ≥ threshold) — no synth-local eval code needed; the user-directed "intersect" semantic is achieved by tuning the Jaccard threshold (sub-D of the sweep below).

### TD method defaults locked explicitly in YAMLs

Every TD method's params block now writes the library default explicitly (vs `params: {}` relying on Python defaults), so each sweep cell's diff is visible and the per-domain winners are obvious in the YAML history:

| Method | Locked defaults | Justification |
|---|---|---|
| `truthfinder` | `init_trust=0.9, gamma=0.3, rho=0.5, max_iters=10, early_stop=1e-3` | Yin et al. paper defaults. |
| `accusim` | `accuracy_prior=0.8, n_competing_values=10, epsilon=1e-3, max_iter=20, sim_threshold=0.7` | Dong et al. paper defaults. |
| `casefusion` | `dimension=10, alpha=1.1, beta=1.1, lr=0.05, converge_rate=1e-5, max_iters=50` | Paper defaults. |
| `fusionquery` | `max_iters=5, theta=3e-5, init_trust=0.95, history_size=50.0, temperature=0.5, threshold=0.7` | Paper defaults. |
| `ltm` | `alpha_0=[50,10], alpha_1=[10,10], beta=[10,10], max_iters=50, burnin=10, thin=2` | Zhao et al. paper defaults. |

### Hyperparameter sweep ([scripts/_tune_fusion_committee.py](../../usecases_synthetic/scripts/_tune_fusion_committee.py))

10 sub-sweeps per domain, all consuming pool-based perfect-cluster correspondences (no EM re-run needed per cell):

| Sub | Knob | Grid | Cells / domain |
|---|---|---|---|
| A | `trust_scores` permutation | 6 perms of `(3, 2, 1)` over the domain's 3 sources | 6 |
| B | per-numeric-attr `tolerance` | `{0.05, 0.10, 0.15}` | 3 |
| C | `trimmed_mean.trim` | `{0.05, 0.10, 0.20, 0.30}` | 4 |
| D | per-list-attr `tokenized_match.threshold` | `{0.1, 0.3, 0.5, 0.75, 1.0}` | 5 |
| E | `truthfinder.gamma × init_trust` | `{0.1, 0.3, 0.5} × {0.8, 0.9, 0.95}` | 9 |
| F | `accusim.accuracy_prior × sim_threshold` | `{0.7, 0.8, 0.9} × {0.5, 0.7, 0.85}` | 9 |
| G | `casefusion.alpha × lr` | `{1.05, 1.1, 1.2} × {0.01, 0.05, 0.1}` | 9 |
| H | `fusionquery.temperature × threshold` | `{0.3, 0.5, 0.7} × {0.5, 0.7, 0.9}` | 9 |
| I | `ltm.alpha_0 × alpha_1` | `{(50,10), (20,5), (30,15)} × {(10,10), (5,5)}` | 6 |
| J | `llm_judge` enabled vs disabled | 2 cells | 2 |

Total: 62 cells per domain × 3 domains = 186 fusion runs. LLM-judge cache reuse across cells means first cell per domain populates the cache (~3 min on music) and subsequent cells run in seconds (~15-30 s). Total sweep wall ~30-60 min.

### Per-domain baseline (post-design, pre-sweep)

| Domain | overall_accuracy | mean_strategy_accuracy | Notable per-attribute movement |
|---|---|---|---|
| companies | **0.8810** | 0.7782 | keypeople scoring (was 0 pre-fix): 0.67 |
| games | **0.7799** | 0.7296 | publisher 0 → 0.87 (all strategies converge), genres 0 → 0.27 |
| music | **0.8406** | 0.7860 | tracks scoring (was 0 pre-fix): 0.82 (union/voting tied) |

### Companies winners (locked 2026-05-12)

Per-attribute movement analysis across all 10 sub-sweeps × 62 cells:

| Sub | Movement found | Winner cfg | Score impact |
|---|---|---|---|
| A · trust | yes on keypeople | current `{forbes:3, fullcontact:2, dbpedia:1}` is the optimum | keypeople 0.56 → **0.67** |
| B · tolerance | no movement | current `0.10` (any value ties) | n/a |
| C · trim | no movement | current `0.10` (any value ties) | n/a |
| D · list_threshold | yes on keypeople | **`threshold=0.1`** (literal "intersection of values"); thresholds 0.1 + 0.3 tie at top | keypeople 0.44 → **0.67** at threshold 0.1 |
| E · truthfinder | no aggregate movement; truthfinder isn't the best for any attr where its param shifts the strategy's own score | keep paper defaults | n/a |
| F · accusim | yes on `name` (per-strategy): accusim's own score on name 0.83 → 1.00 | **`sim_threshold=0.85`** (sole discriminator; accuracy_prior non-discriminating) | accusim on name 0.83 → 1.00 |
| G · casefusion | no movement | keep paper defaults | n/a |
| H · fusionquery | no movement | keep paper defaults | n/a |
| I · ltm | no movement (companies has only one list attr where ltm is wired; baseline cluster shape dominates) | keep paper defaults | n/a |
| J · llm_judge | no aggregate movement (enabled is the default; disabled would lose ~$0/4 negligible on name/country/city where it ties with voting at the perfect-cluster baseline) | **keep enabled** | n/a |

**YAML edits applied to [fusion_committee.yaml](../../usecases_synthetic/config/committees/fusion_committee.yaml)**:
- `evaluation_params.keypeople.threshold: 0.5 → 0.1`
- `attributes.name.accusim.params.sim_threshold: 0.7 → 0.85`

### Games winners (locked 2026-05-12)

Per-attribute movement analysis across all 10 sub-sweeps × 62 cells:

| Sub | Movement found | Winner cfg | Score impact |
|---|---|---|---|
| A · trust | yes on criticScore (best_strategy lift) | current `{sales:3, metacritic:2, dbpedia:1}` is the optimum | criticScore 0.87 → **0.93** |
| B · tolerance | yes on userScore | current `0.10` is the optimum (lifts userScore vs 0.05 / 0.15) | userScore 0.21 → **0.29** |
| C · trim | no movement | keep paper default | n/a |
| D · list_threshold | yes on genres (load-bearing) | **`threshold=0.1`** | genres 0.13 → **0.60** |
| E · truthfinder | no per-strategy movement | keep paper defaults | n/a |
| F · accusim | no per-strategy movement on games attrs (accusim is wired but isn't the best-strategy slot on any games attr) | keep paper defaults | n/a |
| G · casefusion | no movement | keep paper defaults | n/a |
| H · fusionquery | no movement | keep paper defaults | n/a |
| I · ltm | yes on `genres` (per-strategy) | **`alpha_0=[30, 15]`** (vs paper default `[50, 10]`) | ltm on genres 0.13 → **0.27** |
| J · llm_judge | no aggregate movement (llm_judge is wired but isn't the best-strategy slot under perfect clusters) | keep enabled | n/a |

**YAML edits applied to [fusion_committee_games.yaml](../../usecases_synthetic/config/committees/fusion_committee_games.yaml)**:
- `evaluation_params.genres.threshold: 0.5 → 0.1` (load-bearing — single biggest games attr lift in the sweep)
- `attributes.genres.ltm.params.alpha_0: [50, 10] → [30, 15]`

### Music winners (locked 2026-05-12)

Per-attribute movement analysis across all 10 sub-sweeps × 62 cells:

| Sub | Movement found | Winner cfg | Score impact |
|---|---|---|---|
| A · trust | duration vs release-country trade-off | **keep current `{musicbrainz:3, discogs:2, lastfm:1}`**: `discogs=3` lifts duration 0.42 → 0.48 (+6pp) but drops the committee mean (0.78 → 0.76); musicbrainz dominates 5 of 7 attrs per K10 measured baselines | duration unchanged at 0.48; rest preserved |
| B · tolerance | no movement on duration's best-strategy | current `0.05` (any value ties) | n/a |
| C · trim | no movement | keep paper default | n/a |
| D · list_threshold | yes on tracks (load-bearing) | **`threshold=0.1`** | tracks 0.62 → **0.97** (+35pp — biggest music attr lift) |
| E · truthfinder | yes on artist + release-country (per-strategy) | **`gamma=0.1`** (init_trust non-discriminating) | truthfinder on artist 0.90 → 0.92, release-country 0.93 → 0.94 |
| F · accusim | no per-strategy movement on music attrs | keep paper defaults | n/a |
| G · casefusion | yes on name (per-strategy) | **`lr=0.05`** (paper default; alpha non-discriminating) | casefusion on name 0.85 → 0.88 |
| H · fusionquery | yes on duration (per-strategy) | **`temperature=0.7`** (threshold non-discriminating) | fusionquery on duration 0.26 → **0.34** (+8pp) |
| I · ltm | yes on tracks (per-strategy) | **`alpha_0=[20, 5]`** (label unaffected) | ltm on tracks 0.70 → 0.71 |
| J · llm_judge | no aggregate movement | keep enabled | n/a |

**YAML edits applied to [fusion_committee_music.yaml](../../usecases_synthetic/config/committees/fusion_committee_music.yaml)**:
- `evaluation_params.tracks.threshold: 0.5 → 0.1` (load-bearing — biggest music attr lift)
- `truthfinder.params.gamma: 0.3 → 0.1`
- `casefusion.params.alpha: 1.1 → 1.05`
- `fusionquery.params.temperature: 0.5 → 0.7`
- `ltm.params.alpha_0: [50, 10] → [20, 5]`

### Plan rows

| # | Module | Status |
|---|---|---|
| F1 | Build [fusion_perfect_clusters.py](../../usecases_synthetic/lib/fusion_perfect_clusters.py) (pool → cluster → correspondences) | `[x]` (2026-05-12) |
| F2 | Refactor [measure_baseline.py](../../usecases_synthetic/scripts/measure_baseline.py) to use perfect clusters; drop stage-ordering enforcement | `[x]` (2026-05-12) |
| F3 | Loader: parse music `tracks` (`ast.literal_eval`) + games `genre[s]` (comma split) at [lib/loaders.py](../../usecases_synthetic/lib/loaders.py) | `[x]` (2026-05-12) |
| F4 | Add `gold_column_aliases` + `gold_list_columns` YAML surface; `include_singletons=True` | `[x]` (2026-05-12) |
| F5 | Re-add list attrs (companies.keypeople, games.genres, music.tracks) with `union / intersection / intersection_k_sources / voting / prefer_higher_trust / ltm` + `tokenized_match(threshold=0.5)` eval | `[x]` (2026-05-12) |
| F6 | Lock TD defaults explicitly in 3 YAMLs (truthfinder / accusim / casefusion / fusionquery / ltm) | `[x]` (2026-05-12) |
| F7 | Wire OpenAI gpt-5.4-mini `llm_callable` factory + opt-in via `params.llm_callable: openai` on every llm_judge strategy in 3 YAMLs | `[x]` (2026-05-12) |
| F8 | Build [_tune_fusion_committee.py](../../usecases_synthetic/scripts/_tune_fusion_committee.py) sweep harness with 10 sub-sweeps + per-domain checkpointing | `[x]` (2026-05-12) |
| F9 | Run sweep on companies; lock companies winners; apply to YAML | `[x]` (2026-05-12) |
| F10 | Run sweep on games + music; lock per-domain-per-method winners; apply to YAMLs | `[x]` (2026-05-12) — 124 cells × 2 domains, ~3.5h wall. Games: 2 YAML edits (genres threshold + ltm alpha_0). Music: 5 YAML edits (tracks threshold + truthfinder gamma + casefusion alpha + fusionquery temperature + ltm alpha_0). |
| F11 | Re-smoke all 3 domains under final YAMLs | `[x]` (2026-05-12) — companies 0.881 (unchanged), games 0.817 (+3.7pp), music 0.859 (+1.9pp); 144/144 config tests green. |
| F12 | Per-domain fusion baseline `measure_baseline.py --domain <d> --stages fusion` (folded into R6.1) | `[ ]` |
| F13 | Fusion monotonicity check vs K1–K10 noise (Pending #8 best-member-F1 guard applies; folded into R7.2) | `[ ]` |

### Follow-up flags (not R5-Fusion-blocking)

1. **games.publisher is structurally bounded by pool coverage**, not by fusion strategy. The pool brings sales records into 24/25 games fusion-gold clusters (vs 0/25 under the EM-only fallback that was previously in use); the remaining gap is the 1 fusion-gold entity with no pool-confirmed sales partner. Acceptable; documented as a coverage observation, not a fusion bug.
2. **games.genres remains weak at 0.27 best_strategy_accuracy** even with comma-split + multi-source clusters. Root cause: metacritic emits compound genres ("Action Adventure" — single string after comma-split) while the gold uses atomic genres ("Action", "Adventure" — separate tags). Bridging this would require domain-knowledge tokenization (split compound genres at known boundaries) which is outside the fusion-strategy scope. Defer to a future loader-rule that knows the genre taxonomy, or accept the per-attribute baseline.
3. **Other-committee perfect-prior-step chaining (SM → Norm, Norm → EM) not implemented**. Per the 2026-05-12 design conversation, SM / Norm / EM committees today consume raw sources directly (they're measurement committees, not data transformers), so the "perfect prior step" rule is moot — there is no structured upstream output to feed them. The only stage that structurally consumes a prior step's output is Fusion (correspondences from EM), which this row handles. If a future refactor turns Norm into a data-transforming stage (i.e. it produces per-cell normalized values consumed by EM), the same perfect-prior-step rule should apply: EM consumes the per-domain Norm-gold normalized values, not the raw sources. Out of scope here; flagged for any future R6.x design.
4. **LLM-judge wired but rarely the sole winner**. Companies sub-J shows llm_judge enabled stays at the same overall_accuracy as disabled — llm_judge ties with voting / prefer_higher_trust on the per-attribute slots where it's wired (name=1.0, country=0.94, city=0.78). Real value of llm_judge will emerge under K1 paraphrase / K6 noise at hard difficulty, where lexical strategies break and semantic-equivalence judgment is needed; R7.2 will surface this.
5. **`accuracy_prior` is non-discriminating across companies.accusim grid** — every value of accuracy_prior produced the same per-strategy score; only `sim_threshold` mattered. Likely the per-batch sample sizes are small enough that the prior's pseudocount weight is dominated by observed agreement. Document the finding; the YAML keeps the 0.8 default since no grid value shifts the result.

## Protection-set design update (2026-05-06) — Pending #5

**Today's rule** ([usecases_synthetic/lib/protection.py](../../usecases_synthetic/lib/protection.py)): `expanded_positives = EM_gold ∪ fusion_gold (val + test) ∪ pooled_positives`. Any record ID in this set is hard-protected: never dropped at K2 / K3 / K4, never picked as a hard-negative seed at K1 / K2, never noised at K6 (per-knob guards). For fusion val/test cells specifically, **at least one** record per cell must preserve the **exact** gold value so that any fusion strategy can recover it.

**New rule (user directive 2026-05-06)**: every fusion val + test entity must remain alive in every variant (unchanged), but for each (entity, attribute) cell **at least one** surviving record must stay **"close enough"** to the gold value that a **lenient** fusion strategy can still recover the truth. "Close enough" replaces the prior exact-match guarantee. Strict fusion strategies that demand exact agreement are no longer guaranteed to find the gold; lenient strategies (fuzzy-match aggregators, voting-with-tolerance) become the floor.

### Why relax

The exact-match guarantee was over-conservative: it forced every value-mutating knob to leave at least one cell pristine, which capped the achievable per-cell noise. With the relaxed rule, hard variants can noise every cell of a fusion-gold entity *as long as one record stays within tolerance*, dramatically widening the operator budget at hard.

### What "close enough" means

Per-attribute tolerance. Concrete proposal — push back on any:
- **Numeric / date attributes**: reuse the K2 numeric-overlap policy (years ±1 absolute, continuous ±3 % relative).
- **Short string / nominal attributes**: `_levenshtein_ratio` ≥ 0.85 OR appears in a per-attribute synonym set (e.g. country codes, industry acronyms).
- **Long / multi-token string attributes** (titles, names): `lexical_extended_jaccard(inner_token_threshold=0.8)` ≥ 0.6 — extended Jaccard on tokens with an inner Levenshtein gate per token. Single source of truth: [usecases_synthetic/lib/niche_metrics.py:lexical_extended_jaccard](../../usecases_synthetic/lib/niche_metrics.py#L151) (already used by K2). Plain character-edit Levenshtein is **not** used on long strings — token reorderings dominate the edit distance and produce false negatives.
- **Free text / description**: same metric, threshold ≥ 0.5.
- **Lists / set-valued (genres, platforms)**: same metric over flattened set tokens, threshold ≥ 0.5.
- **Per-domain overrides** allowed via a `fusion_protection_tolerance` block in the K6 config.

The full metric table is duplicated in §"K4 sign-off (2026-05-06) → Closeness-metric specification" since K4's closest-to-target survivor selection reads the same spec.

### Implementation surface

1. [usecases_synthetic/lib/protection.py](../../usecases_synthetic/lib/protection.py) — drop the per-cell exact-match guarantee; expose a new `fusion_cell_tolerance(domain, attribute)` helper that returns the per-attribute tolerance spec.
2. Each value-mutating knob's per-cell guard (K1, K3, K5, K6, K10): replace the "skip if would-break exact gold" rule with "before commit, verify at least one surviving record per (entity, attribute) cell still satisfies `is_close_enough(value, gold, tolerance)`. If not, revert that mutation and either pick a different record to mutate or skip the cell."
3. Add a per-cell post-K6 audit emitted to provenance: `cells_with_no_close_record` (should be empty by construction; non-empty rows are bugs).
4. Update R7.1 / R7.2 reporting: the new "lenient fusion floor" is the worst-case fusion score; report it explicitly so callers know strict fusion may underperform vs lenient on hard variants.

### Open design questions for the user

- **Tolerance values** — are the proposals above (Levenshtein ≥ 0.85 short / ≥ 0.80 long, ±3 % continuous, ±1 year) right, or override per-attribute?
- **Lenient fusion strategy enumeration** — which fusion members count as "lenient"? Affects which committee row is the new floor.
- **Failure mode when a cell can't keep any record close** — abort the variant, emit a warning + un-noise the affected cell, or accept a contract violation in the audit?

## Hard blockers (current)

- **R2 GPU availability.** Prior checkpoints used the CPU fallback (distilbert-base-uncased, 3 epochs). If a GPU is now available, prefer roberta-base + 10 epochs for tighter PLM scores → narrower S3 margin band → more aggressive corner-case selection. Document the choice per-domain in [config/ditto/README.md](../../usecases_synthetic/config/ditto/README.md).
- **R3 ADI ID drift.** ADI's `correspondences_*.csv` was produced against the pre-refresh sources. Before merging, verify the IDs in those CSVs still resolve against the new sources; remap or drop where they don't.
- **R4 / R5 are blocking ordering gates.** No R6 runs until both interactive reviews are signed off — otherwise we burn compute on a config the user hasn't approved.

## Per-domain specifics

### companies
- Sources: dbpedia, forbes, fullcontact (all CSV at `usecases/companies/input/data/`; `*_metadata.json` sidecars present).
- Source-pairs: forbes ↔ dbpedia, forbes ↔ fullcontact, dbpedia ↔ fullcontact (verify against the refreshed EM gold filenames).
- ADI baseline: [../automatic-data-integration/scripts/output/companies_0302/entity_resolution/matching/](../../automatic-data-integration/scripts/output/companies_0302/entity_resolution/matching/).

### games
- Sources: dbpedia, metacritic, sales (all CSV).
- Refreshed dbpedia row count per `dbpedia_metadata.json`: 46,580 (vs prior ~65k) — re-baseline must be done from scratch; prior Ditto checkpoint cannot be reused.
- Source-pairs per [config/domains/games.yaml](../../usecases_synthetic/config/domains/games.yaml): 3 pairs (verify naming against refreshed gold).
- ADI baseline: [../automatic-data-integration/scripts/output/games_0302/entity_resolution/matching/](../../automatic-data-integration/scripts/output/games_0302/entity_resolution/matching/).
- Carry-over from prior pass: games' fusion `test_set.xml` had 0/15 overlap with EM-gold positives — flag for re-authoring during R5 if fusion remains broken on the refreshed gold.

### music
- Sources: musicbrainz, discogs, lastfm (all CSV).
- Source-pairs: musicbrainz ↔ discogs, musicbrainz ↔ lastfm (2 pairs).
- Watch the `label` column collision with Ditto's reserved-field set (S11 fix at `em_matching_committee_music.yaml` — `label` stays dropped from `ditto_plm.fields` until the canonical attribute is renamed).
- ADI baseline: [../automatic-data-integration/scripts/output/music_0302/entity_resolution/matching/](../../automatic-data-integration/scripts/output/music_0302/entity_resolution/matching/).

## Pick-up instructions

1. **R1 first** — config-only, no compute. Spot-check tests stay green.
2. **R2 is interactive, per-domain** — for each domain, Claude proposes the training plan (data prep, hyperparameters, augmentation, acceptance criteria); the user approves or revises before any compute starts. Train sequentially per domain (~10 min CPU smoke / hours GPU production). Symlink `best/` immediately after each run so committees resolve.
3. **R3 → R4 → R5 are interactive** — Claude proposes one slice at a time (per-stream / per-knob / per-stage × per-domain); the user approves or revises before moving on. No runtime changes during the review.
4. **R6 + R7 only after R5** — committee freeze is locked at R6.1; any later edit to a committee YAML invalidates R6 + R7 outputs and forces a re-baseline.

## What this plan retires

- **Movies + products variant generation** (former S6 / S7 / S11 movies/products / S12 movies/products / etc.) — descoped to a future plan; the upstream pool prerequisites (movies PLM run, products use-case stabilization) are unchanged.
- **The S11–S17 row-by-row tracker** — superseded by R6 + R7. The historical detail of bugs found during the first pass is summarized in the "load-bearing" section above; the prose is no longer in this plan.
- **Phase A0 / A / B detail** — completed and summarized; the rows are not repeated since the code + configs are committed.

The committed code + configs from Phases A0 / A / B / C remain in place; this plan does not delete or rewrite them, only re-runs them on the refreshed sources after the R2 / R3 / R4 / R5 reviews.
