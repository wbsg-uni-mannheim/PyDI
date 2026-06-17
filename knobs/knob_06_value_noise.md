# Knob 6 — Value-noise injection rate

**Status:** LOCKED. **Scenario:** S1 + S2 (fully controllable).

## Definition

Typos, OCR confusions, truncations, whitespace/punctuation corruption, case corruption — values a human would recognize as errors. **Source rows only; the fusion gold file is untouched by construction** (Knob 6 never mutates the gold artifact).

**Boundary vs Knob 1:** kept separate. Knob 1 = legitimate variant (paraphrase); Knob 6 = error. Case corruption stays in Knob 6 even though it looks paraphrase-adjacent — framing is "the source's data-entry pipeline is buggy," not "the source chose a different style."

## Dimensions controlled

- Noise & Corruption (Norm)
- Corner-Case Difficulty (EM)
- Conflict Rate (Fusion)

## Sub-parameters (per attribute class — mirrors Knob 3)

- `noise_rate_primary` — canonical label (title/name). Tiny everywhere; non-zero only at hard.
- `noise_rate_key` — other blocking-relevant attributes.
- `noise_rate_secondary` — everything else.

## Noise types in scope (all in)

Char-level edits (insert/delete/substitute/transpose), OCR confusions (`O`↔`0`, `l`↔`1`, `rn`↔`m`, `cl`↔`d`), truncation, whitespace/punctuation corruption, case corruption.

## Per-source shape via measured baseline

Same approach as Knob 3: measure baseline per-source noise rate per attribute class as a per-domain vector (cheap heuristics — non-ASCII garbage, unbalanced punctuation, edit-distance-1 collisions to a frequent token), then apply a shape transform per level.

## Easy / Medium / Hard

| Level | `noise_rate_primary` | `noise_rate_key` | `noise_rate_secondary` | Per-source transform | Target state |
|---|---|---|---|---|---|
| **Easy** | 0% | ~0% (incidental) | low | **Compress** (propagate clean values from cleanest source) | Sources look clean. Occasional whitespace/case glitch on long secondary fields. |
| **Medium** | 0% | low | moderate | **Identity** | Recognizable but sparse noise. Lexical EM comparators degrade noticeably; fuzzy/learned ones absorb most of it. |
| **Hard** | small but non-zero | moderate | heavy | **Stretch** | Pervasive noise. Even primary labels occasionally corrupted. Lexical blockers lose recall; rule-based EM degrades sharply. |

## Composition

- **Knob 1:** orthogonal — paraphrase = legitimate, noise = error. No overlap by construction.
- **Knob 5:** boundary — malformed unparseable date is Knob 6; different valid format is Knob 5.
- **Knob 3:** orthogonal in effect, **ordered Knob 6 → Knob 3**. Noise applied first, then drops happen on noised data. A noised cell may end up dropped (Knob 3 is not noise-aware); we **accept the dilution**. Provenance records the full noise→drop chain.
- **Knob 7:** distinct — Knob 6 produces *wrong* values, Knob 7 produces *insufficient* values. A typo that accidentally creates a homonym counts as Knob 6.
- **Knob 4 (locked I4):** K4 runs upstream of the joint `1/5/6/7` phase. K4-fabricated cells (`k4_fabricated=True` in the joint provenance index) are **fair game** for Knob 6 noise — fabricated rows are new data, not gold carriers, and there is no semantic reason to exempt them. Knob 6 reads the joint index but explicitly ignores the `k4_fabricated=True` flag during its collision check (all other earlier rows are hard-skipped).

## Fusion safety — reject on collapse, gold untouched

The fusion gold file is never touched. On committee collapse, **roll back** the offending Knob 6 mutations until a monotone drop reappears. No gold extension. Noisy values are errors and must not be promoted into the gold contract.

### Monotonicity guards

1. **Per-entity clean-primary floor (locked).** For every fusion-gold entity, **≥1 source must retain a noise-free primary value**. Hard constraint — guarantees blocking always has a clean anchor.
2. **Per-cell clean-survivor floor.** For every fusion-gold (entity, attribute), ≥1 source retains a noise-free value.
3. **Soft global primary cap at hard.** Upper bound on the fraction of fusion-gold entities with *any* noised primary across sources (default ~30–40%, **pinned during the baseline-measurement pass** — see the `measure_baseline_profile.py` cross-cutting follow-up in [plan_algorithmselection.md](../plan_algorithmselection.md); the pass measures per-domain primary-noise headroom and writes the binding scalar into the per-domain YAML).
4. **Committee check.** SM near-flat; EM and fusion show monotone drops.

## Committee expectations

- **SM:** flat.
- **Blocking:** monotone drop, sharp for lexical/n-gram blockers, mild for embedding blockers.
- **EM:** monotone drop, sharp for rule-based comparators, mild for learned/embedding matchers.
- **Fusion:** monotone drop given the survivor floors. Per-source noise shape rewards provenance-aware fusers.
- **Normalization stage:** secondary target after Knob 5.

## Per-domain notes

- **Companies:** Forbes hand-curated and clean, FullContact mild, DBpedia mixed (~0.3% encoding noise on names; ~32 bad rows). **Forbes has its own baseline noise on country (`[a]` footnotes)** — should be cleaned at easy.
- **Games:** all three sources clean at baseline. Per-source noise shape barely exists.
- **Music:** all three sources reasonably clean at baseline.

## Provenance

`transform_fn ∈ {typo_substitute, ocr_confuse, truncate, whitespace_corrupt, case_corrupt, cleanup, rollback_for_committee}`, `transform_params={positions, chars, ...}`. (`cleanup` and `rollback_for_committee` are first-class members of the enum — see §Algorithm selection.) Rollback emits mirror `transform_fn=rollback_for_committee`. Cells dropped downstream by Knob 3 retain their Knob 6 record.

## Algorithm selection

**Chosen approach.** Tier B — deterministic in-house dispatcher over a fixed taxonomy of cell-level corruption operators lifted from the FEBRL / Christen-Vatsalan literature. No LLM, no ML. The dispatcher is a pure pandas/regex/stdlib function that, for each (source, attribute class) pair, draws per-cell whether to corrupt and then draws which operator to apply; both draws use a seeded `numpy.random.Generator`. The literature contribution lives in the *taxonomy* (which operators exist, what their parameters are, what edit behaviour they model) — our wrapper is plumbing. The operator catalogue is the intersection of the Knob 6 "Noise types in scope" list above with the Christen-Vatsalan corruption taxonomy:

| `transform_fn` | Source method (paper.md) | Notes |
|---|---|---|
| `typo_substitute` | *Character Edit Corruption* (insert / delete / substitute / transpose) with optional QWERTY keyboard-adjacency bias from *Keyboard Proximity Corruption* | Up to `max_edits_per_cell` edits; positions uniform unless attribute class flags prefix/suffix bias. |
| `ocr_confuse` | *OCR Error Simulation* with the default char + char-pair lookup table (`O→0, l→1, I→1, m↔rn, cl↔d, vv→w`) | Longest-match preferred at overlaps. |
| `truncate` | *Character Edit Corruption* — deletion-only at tail (end-weighted `position_distribution`) | Truncation length drawn from `[1, max_truncate_chars]`. |
| `whitespace_corrupt` | FEBRL 11-type set: `space_insert` / `space_delete` | Also covers punctuation collapse (`,` / `.` / `-` deletion) as a minor extension. |
| `case_corrupt` | Not in Christen-Vatsalan core; modelled as `substitute` over ASCII letters restricted to `c.swapcase()` | Framed as "source's data-entry pipeline is buggy" per the Knob 6 Definition section. |
| `cleanup` | Bidirectional inverse of the above operators, applied only at *easy* | Reverts baseline noise (e.g., Forbes `[a]` footnotes) on the cleanest source. Logged to provenance with `original_value` = noisy baseline, `new_value` = cleaned. Out-of-scope at medium/hard. |

Phonetic errors, value-swap, field-swap and missing-value injection from Christen-Vatsalan are **rejected** (out of scope for Knob 6 — value-swap / field-swap overlap with Knob 7 semantics, missing-value belongs to Knob 3, phonetic only fits name attributes and adds a per-domain rule table for small marginal signal).

**Mapping to easy/medium/hard.** Monotonicity is enforced by strictly non-decreasing per-class noise rates and strictly non-shrinking operator sets across the three levels. A single `level` parameter selects a frozen `(noise_rate_primary, noise_rate_key, noise_rate_secondary, operator_mix, per_source_shape_transform)` tuple from a per-domain YAML; the per-source transform (`compress` / `identity` / `stretch`) is applied on top of the measured baseline vector exactly as specified in the "Per-source shape via measured baseline" section.

| Level | Primary rate | Key rate | Secondary rate | Operator set | Per-source transform |
|---|---|---|---|---|---|
| **Easy** | 0% | ~0% | low (≤1%) | `{whitespace_corrupt, case_corrupt, cleanup}`. `cleanup` is exclusive to easy and reverts baseline noise on the cleanest source (see Companies note below). | **Compress** — clean values from the quietest baseline source propagate to noisier sources for primary/key attributes. |
| **Medium** | 0% | ~1–2% | ~3–5% | Easy set ∪ `{typo_substitute (substitute / transpose edits only), ocr_confuse (single-char lookups only)}`. Max 1 edit per cell. | **Identity** — baseline per-source noise shape preserved. |
| **Hard** | small non-zero (≤1%, hard-capped by the clean-primary floor) | ~5–8% | ~10–15% | Full catalogue with all edit sub-types, char-pair OCR confusions, truncation, and keyboard-adjacency bias enabled. Up to `max_edits_per_cell = 3`. | **Stretch** — per-attribute rates of the already-noisier sources scaled up; cleanest source left close to identity to preserve the per-entity clean-survivor floor. |

Independent togglability: Knob 6 reads only the cell values passed to it plus the fusion gold index (for the monotonicity guards) and writes only to source-row cells. It composes **after Knobs 1 and 5, before Knob 7** in the `1/5/6/7` joint phase (canonical order per [README.md](README.md#canonical-knob-application-order)). **Cell-collision contract (locked I1, symmetric with [knob_01_surface_augmentation.md:82](knob_01_surface_augmentation.md#L82)):** before mutating a cell, the dispatcher reads the joint provenance index `output/provenance/knob_0{1,4,5,6,7}_*.csv` and **unconditionally skips** the cell if any earlier row exists for the same `(entity_id, source, attribute)` — matching Knob 1's unconditional-skip language verbatim. The sole exception is K4-fabricated cells (rows with `k4_fabricated=True` in `transform_params`): those are **fair game** for noise injection per the locked C2 contract (resolves I4 — fabricated rows are new data, not gold carriers, and the difficulty signal compounds correctly; see [knob_04_coverage_skew.md](knob_04_coverage_skew.md#joint-cell-collision-index-integration)). Skipped cells are logged to `output/provenance/knob_06_skipped.csv` with `reason=cell_collision_with_{1,5,7}`. Knob 3 runs after Knob 6, so some noised cells get dropped; accepted per Composition and recorded as a noise→drop chain in provenance.

**Literature citations.**
- **Christen & Vatsalan, "A Flexible Data Generation and Corruption Tool"** ([../literature-search-generation/christen_vatsalan_data_gen_corruption/paper.md](../literature-search-generation/christen_vatsalan_data_gen_corruption/paper.md)) — anchor paper. Methods used: *Character Edit Corruption*, *Keyboard Proximity Corruption*, *OCR Error Simulation*. Provides the algorithm-level specification, the parameter ranges (`corruption_probability ∈ [0.05, 0.20]`, `max_edits ∈ [1, 3]`), and the default OCR lookup table we copy verbatim.
- **FEBRL** ([../literature-search-generation/febrl/paper.md](../literature-search-generation/febrl/paper.md)) — *11-Type Corruption Operator Set*. Cited as the originating taxonomy (misspelling, char_insert/delete/sub/transpose, space_insert/delete, etc.) and for the `corruption_probability` semantics used in the per-class rate table above. Our dispatcher implements a strict subset of FEBRL's 11 operators, so citing FEBRL insulates the knob from "why not X?" reviewer challenges.
- **DAPO** ([../literature-search-generation/dapo_large_scale_data_pollution/paper.md](../literature-search-generation/dapo_large_scale_data_pollution/paper.md)) — precedent for combining a parameterised corruption profile with other pollution knobs in a single controlled benchmark run. Cited as the "this composition pattern is not novel" reference.
- No LLM paper cited. The `curated_llm_tabular_augmentation` line is explicitly **not** invoked here — see Rejected Alternatives.

**Determinism & provenance.**
- RNG: a single `numpy.random.default_rng(seed)` per `(domain, variant, knob=6)` tuple; the seed is written into the variant's `config/difficulty.yaml`. Re-runs under the same config are bit-identical. Operator and position draws are derived from the same generator so that independent level changes do not resample unrelated cells.
- Per-domain config file: `usecases_synthetic/config/knob_06_noise/<domain>.yaml`. Keys: per-attribute-class base rates (the `noise_rate_primary/key/secondary` values from the table above, authored once then checked in), per-source baseline noise vector (measured during the Step 5 baseline pass per the cross_cutting.md bootstrap order — **input**, not authored — written by `usecases_synthetic/scripts/measure_baseline_profile.py` with output schema `{(source, attribute): baseline_rate}`), `operator_mix`, `max_edits_per_cell` cap per level, `soft_global_primary_cap_hard` scalar, and `cleanup_rules` (per-(source, attribute) bidirectional reversal patterns consumed by the easy-only `cleanup` operator — e.g. Forbes `country` `[a]` footnote regex).
- Operator lookup tables (QWERTY adjacency map, OCR char + char-pair table) live as static YAML at `usecases_synthetic/config/knob_06_noise/_tables/{qwerty.yaml,ocr.yaml}` — shared across domains.
- Provenance written per corrupted cell to `output/provenance/knob_06_noise.csv` inside the variant directory using the cross_cutting.md schema:
  ```
  (entity_id, source, attribute, original_value, new_value,
   transform_fn ∈ {typo_substitute, ocr_confuse, truncate,
                   whitespace_corrupt, case_corrupt, cleanup,
                   rollback_for_committee},
   transform_params, knob=6, level)
  ```
  `transform_params` is a JSON-encoded string in the CSV column (keys: `positions: [...], chars: [...], edit_type: ..., operator_seq: [...]`), since the cross_cutting.md provenance schema is a flat row.
- **Two distinct "undo" mechanisms, kept separate.** (a) *Skip* on guard violation — apply-time path. Before mutating a cell the dispatcher checks (i) per-entity clean-primary floor, (ii) per-(entity, attribute) clean-survivor floor, (iii) soft global primary cap at hard; if mutating would violate a floor, the dispatcher **never writes** the cell and logs an audit row to `output/provenance/knob_06_skipped.csv`. No provenance row is emitted. (b) *Rollback* on committee collapse — post-apply path, triggered only by the Fusion-safety rejection loop. An already-emitted mutation is reverted and a mirror row is written with `transform_fn=rollback_for_committee`, `new_value == original_value`. Readers can reconstruct the effective state by streaming the provenance log and cancelling each rollback against its matching original row.
- Cells subsequently dropped by Knob 3 keep their Knob 6 record; Knob 3's provenance row references the Knob 6 row via `(entity_id, source, attribute)`.
- Caching: the full output is a file artifact on disk (the noised source datasets plus the provenance CSV). No per-cell in-memory cache. Because the RNG is seeded and the operator tables are static, regeneration is reproducible without caching intermediate cells.
- Committee surface: the Norm / EM / Fusion committees (per the Committee-expectations section of this card and the cross_cutting.md committee mechanism) see the noised source files exactly as written. No operator leakage into the committee evaluation harness.

**Domain-specific adjustments.**
- **Companies** (Forbes clean + FullContact mild + DBpedia mixed, per the Per-domain-notes section): Forbes `country` carries baseline `[a]` footnote noise that must be **cleaned at easy** via the `cleanup` transform_fn (operator-specific reversal of the footnote pattern, authored in the per-domain YAML under a `cleanup_rules` key). DBpedia's ~0.3% encoding noise on names is included in the measured baseline vector, so medium/hard only *extends* it rather than replaces it. The clean-primary floor is most binding on Companies at hard because DBpedia is the primary-noise carrier.
- **Games, Music** (all sources reasonably clean at baseline): per-source noise shape barely exists, so the `compress` transform at easy is near-identity and `stretch` at hard simply scales the authored per-class rates uniformly. These two domains are effectively the "algorithmic clean" reference and should produce the cleanest monotone committee drop.
- **Movies, products**: deferred alongside the Step 6 prototype. The dispatcher no-ops on domains without a `config/knob_06_noise/<domain>.yaml` (warns in the log). No code change when those domains come online — only a new YAML.

**Rejected alternatives.**
- **LLM-based noise generation** (e.g., prompting an LLM to "add realistic typos"). Rejected: the task is mechanical character-level perturbation with a well-validated deterministic taxonomy. LLM use would sacrifice determinism, introduce contamination risk, inflate validation cost (mandatory committee + human spot-check per plan_algorithmselection.md decision framework), and deliver no expected quality gain over FEBRL operators. **LLM not used because the deterministic alternative is sufficient.**
- **BART-error generation** ([../literature-search-generation/bart_error_generation/paper.md](../literature-search-generation/bart_error_generation/paper.md)) — rejected under the plan_algorithmselection.md framework rule against heavyweight ML methods (violates determinism, validation cost, dependency weight simultaneously).
- **Phonetic Error Simulation / Value Swap / Field Swap / Missing Value** from Christen-Vatsalan — rejected as out of Knob 6 scope per the Definition and Composition sections (phonetic is name-only with small marginal signal; value/field swap overlap Knob 7; missing belongs to Knob 3).
- **GeCo / Gecko** ([../literature-search-generation/geco/paper.md](../literature-search-generation/geco/paper.md), [../literature-search-generation/gecko_personal_data_gen/paper.md](../literature-search-generation/gecko_personal_data_gen/paper.md)) — these are complete record-generation frameworks. We only need the corruption component, and FEBRL / Christen-Vatsalan already provide that at a finer granularity. Using GeCo/Gecko wholesale would drag in person-data frequency tables that are irrelevant to the PyDI domains.

**Implementation handoff.** Everything Step 6 needs to implement this knob without re-reading surrounding cards:

- **Target script:** `usecases_synthetic/scripts/apply_knob_06_noise.py` (new, convention matches `apply_knob_08_naming.py`). Standalone runnable from repo root.
- **Function shape (illustrative):**
  ```python
  def apply_knob_06(
      domain: str,
      level: Literal["easy", "medium", "hard"],
      sources: dict[str, pd.DataFrame],
      fusion_gold: pd.DataFrame,                   # for monotonicity guards + clean-survivor floors
      attribute_classes: dict[str, dict[str, Literal["primary", "key", "secondary"]]],
                                                   # {source_name: {column: class}}
      baseline_noise_vector: dict[str, dict[str, float]],
                                                   # measured per (source, attribute) — from the baseline pass
      config_path: Path,                           # usecases_synthetic/config/knob_06_noise/<domain>.yaml
      output_dir: Path,
      seed: int,
  ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
      """Returns (noised_sources, provenance_df, skipped_df)."""
  ```
- **Inputs the script reads:**
  - Source DataFrames (via `PyDI.io.load_*`, preserving `df.attrs["dataset_name"]`).
  - The fusion gold artifact from `usecases/<domain>/input/fusion/` — needed only for the clean-primary and per-cell clean-survivor floors; **never mutated**.
  - Per-domain config at `usecases_synthetic/config/knob_06_noise/<domain>.yaml`.
  - Shared operator tables at `usecases_synthetic/config/knob_06_noise/_tables/{qwerty.yaml,ocr.yaml}`.
  - Baseline noise vector (written by the Step 5 baseline pass; path recorded in the variant's `config/difficulty.yaml`).
- **Outputs the script writes** (under the variant directory):
  - Noised source files in `input/data/` (same format as input — XML/JSON/CSV).
  - Provenance log at `output/provenance/knob_06_noise.csv`.
  - Skipped-cell audit at `output/provenance/knob_06_skipped.csv` (rows where a floor guard blocked mutation).
- **Pipeline integration:** Knob 6 sits inside the `Knobs 1/5/6/7` joint phase of the canonical S1 order from [README.md](README.md#canonical-knob-application-order). It runs *after* Knob 1 and Knob 5 have recorded their per-cell transforms (dispatcher checks the joint provenance index before mutating so that Knob 1 paraphrase + Knob 6 noise on the same cell is bounded) and *before* Knob 3's cell-drop pass.
- **Dependencies:** stdlib + `pandas` + `numpy` + `pyyaml`. No PyDI extension points (Knob 6 prepares input, it doesn't subclass any base matcher).
- **Authoring task before first run:** populate `usecases_synthetic/config/knob_06_noise/{companies,games,music}.yaml` with per-class rates and operator-mix entries using the Easy/Medium/Hard table above as the source of truth; **copy the QWERTY adjacency table from the FEBRL paper** ([../literature-search-generation/febrl/paper.md:224](../literature-search-generation/febrl/paper.md#L224)) into `_tables/qwerty.yaml`, and copy the OCR character + char-pair lookup table from the Christen-Vatsalan paper ([../literature-search-generation/christen_vatsalan_data_gen_corruption/paper.md:316-319](../literature-search-generation/christen_vatsalan_data_gen_corruption/paper.md#L316-L319)) into `_tables/ocr.yaml`. The baseline noise vector is **measured**, not authored.
- **Smoke test:** for each domain with a config, run the script at all three levels and assert (a) the per-entity clean-primary floor holds for every gold entity, (b) the per-(entity, attribute) clean-survivor floor holds, (c) total corrupted cell count at hard > medium > easy per attribute class, (d) the provenance row count equals the number of cell-value mutations, (e) the fusion gold file on disk is byte-identical before and after the run.
