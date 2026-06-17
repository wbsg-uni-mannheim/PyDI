# Knob 3 — Per-source attribute drop rate

**Status:** LOCKED. **Scenario:** S1 + S2 (fully controllable).

## Definition

Probability that a given attribute is missing in a given source record. **Source-level (row-level) missingness only.** Schema-level omission is Knob 9 (S2).

## Dimensions controlled

- Blocking Key Completeness (Block)
- Record Completeness (EM)
- Source Density (Fusion)

## Sub-parameters (per attribute class)

- `drop_rate_primary` — canonical label (title / name). Tiny everywhere; non-zero only at hard.
- `drop_rate_key` — other blocking-relevant attributes (year, artist, developer, country, …).
- `drop_rate_secondary` — everything else.

**Per-source shape via measured baseline.** For each domain, measure baseline per-source missingness per attribute class as a vector. Each level applies a **transformation of that vector** (no source labels — `compress` / `identity` / `stretch`).

## Easy / Medium / Hard

| Level | `drop_rate_primary` | `drop_rate_key` | `drop_rate_secondary` | Per-source transform |
|---|---|---|---|---|
| **Easy** | 0% | low (base) | low (base) | **Compress** spread toward min-missingness source via cross-source value propagation. Parameter `compression_factor ∈ [0,1]`. |
| **Medium** | ~0% | moderate (base) | moderate (base) | **Identity** — baseline preserved. |
| **Hard** | small but non-zero | heavy (base) | heavy (base) | **Stretch** spread: higher-missingness sources amplified, lowest near baseline. Parameter `stretch_factor ≥ 1`. |

## Composition

- **Knobs 1/5/6/7:** ordered Knobs 1/5/6/7 → Knob 3. Drops happen on perturbed data; a noised cell may end up dropped. Dilution accepted. See [knob_06_value_noise.md](knob_06_value_noise.md).
- **Knob 4:** different granularity. Knob 4 controls *which entities* a source covers; Knob 3 controls *which attributes* are missing within covered records.
- **Knob 10:** orthogonal — Knob 10 shuffles which source carries the gold per cell; Knob 3 shapes how often each source has any value at all.
- **Pipeline order:** runs **after** value perturbations and **before** Knob 10.

## Fusion safety

For every fusion-gold (entity, attribute), **at least one source must retain the near-gold value** (where "near-gold" = value accepted by the lenient fusion evaluator's per-attribute canonical-form equality — defined by [knob_10_source_reliability.md §Algorithm selection](knob_10_source_reliability.md) family→comparator routing, with the Knob 5 family map as the schema-side input; so Knob 1/5/6/7 perturbations of the gold still count when they preserve canonical form). Hard constraint on the dropper.

### Monotonicity guards

1. **Conflict-preserving drop.** When ≥2 sources disagree on an attribute (after Knobs 1/5/6/7), preserve at least two conflicting values. Drops come from the redundant-agreement pool first.
2. **Committee check.** Fusion committee must show monotone drop easy→medium→hard per attribute. Inversion → conflict-preservation kicks harder, or drop rate softens locally.
3. **Single-source-survivor cap at hard.** Explicit upper bound on the fraction of (entity, attribute) cells that collapse to a single surviving source. Default `0.05` (locked in §Algorithm selection / `single_source_survivor_cap_hard`); per-domain overridable in YAML.

## Committee expectations

- **Blocking:** candidate recall drops monotonically as `drop_rate_key` rises (sharper for single-key blockers).
- **EM:** F1 drops monotonically; learned matchers with missing-value handling degrade less than rule-based comparators.
- **Fusion:** accuracy drops monotonically *given* the conflict-preserving constraint and survivor cap.

## Per-domain notes

- **Companies:** **asymmetric — already at hard on DBpedia financials** (assets 6.5%, revenue 20.4%). Stretch transform at hard has *negative headroom* there. **Caps must be enforced per (attribute, source)**, not per attribute class globally.
- **Games:** almost uniformly dense (≥87% on every attribute except DBpedia franchise at 52%). Less per-source spread to leverage.
- **Music:** **bimodal headroom.** LastFM scalar gaps are schema-level (Knob 9 territory). Real Knob 3 surface is on the array attributes: LastFM `tracks_*` at 46.6%, Discogs `tracks_track_duration` at 57% — at-or-near hard already on arrays.

## Calibration artifact

Per domain, a measured baseline missingness vector per (attribute, source). Stored alongside the difficulty config; the level transformations are applied on top.

## Provenance

- **Drop:** `transform_fn=drop`, `transform_params={reason: "rate" | "conflict_preserve_skip"}`.
- **Propagation fill (easy):** `transform_fn=propagate_fill`, params include source entity/attribute the value was copied from.

## Algorithm selection

**Chosen approach.** Tier A — deterministic in-house masking over a per-(source, attribute) drop-rate matrix, implemented in pure pandas + `numpy.random.default_rng`. No literature method ported, no LLM. The dispatcher (a) re-measures the per-(source, attribute) baseline missingness vector from the current source DataFrames at the start of every run (never cached across runs — source data can still change between generator invocations, see "Determinism & provenance" below), (b) transforms that vector into a *target* missingness matrix `T_level[s, a]` via one of three named transforms keyed by the level (`compress` / `identity` / `stretch`), (c) draws **one uniform `u[s, a, e] ∼ U(0, 1)` per cell, reused across all three levels**, so that "drop iff `u < T_level[s, a]`" yields monotone nested drop sets `D_easy ⊆ D_medium ⊆ D_hard` (this is the load-bearing mechanism behind the ablation-hardness "independent togglability" requirement — level changes never reshuffle unrelated cells), (d) honours three hard constraints — *fusion survivor floor*, *conflict-preserving drop*, *single-source-survivor cap* — by excluding protected cells from `D_level` *before* applying it, and only then (e) mutates the DataFrames. Easy additionally runs a `propagate_fill` pre-pass that copies values from the lowest-missingness source into higher-missingness sources on the same entity. Crucially, **at easy the compress fill is followed by the same threshold drop step at the small `rate_easy.*` floor** — compress reduces missingness first, then a tiny floor-rate drop is applied on top; the floors are never a no-op, they just sit far below the filled level. All randomness is a single seeded RNG; the output is fully reproducible given (source snapshot, seed). The function is pure pandas / `numpy` / stdlib; no new dependencies.

**Mapping to easy/medium/hard.** Monotonicity is enforced by level-indexed transform selection plus monotone attribute-class rate floors. Let `B[s, a]` be the measured baseline missingness rate for source `s` and attribute `a` and let `cls(a) ∈ {primary, key, secondary}` be the attribute class authored per domain.

| Level | `drop_rate_primary` floor | `drop_rate_key` floor | `drop_rate_secondary` floor | Per-source transform on `B[s, a]` | Extra |
|---|---|---|---|---|---|
| **Easy** | 0 | `rate_easy.key` | `rate_easy.secondary` | `compress(B, compression_factor)` — `T[s, a] = B_min[a] + (1 - compression_factor) · (B[s, a] - B_min[a])`, clipped to floors; `compression_factor ∈ [0, 1]` (default 0.7). Realised by `propagate_fill` from `argmin_s B[s, a]` onto higher-missingness sources on the same entity. | Primary attributes always at 0 drop. |
| **Medium** | ~0 | `rate_med.key` | `rate_med.secondary` | `identity(B) = B`, floors applied. | Baseline preserved. |
| **Hard** | `rate_hard.primary` (small, non-zero) | `rate_hard.key` | `rate_hard.secondary` | `stretch(B, stretch_factor)` — `T[s, a] = B_min[a] + stretch_factor · (B[s, a] - B_min[a])`, capped at 1 and at the per-(attribute, source) ceiling. `stretch_factor ≥ 1` (default 1.5). | Single-source-survivor cap enforced (default ≤ 5% of cells). |

Concrete defaults for the floor rates and the `compression_factor` / `stretch_factor` values live in the per-domain YAML (`rates_per_level`) and are calibrated by the committee loop at Step 8, not pinned on this card. Monotonicity guarantee: for any (source, attribute) the realized drop rate is **non-decreasing** from easy → medium → hard (equalities occur on the min-missingness source and on cells where `B[s, a] ≈ B_min[a]`, where compress ≈ identity); the fusion-safety constraints can only *reduce* realized drops within a level, never push them above the next tier. The guarantee is delivered concretely by the shared per-cell uniform draw described in *Chosen approach* above: `D_easy ⊆ D_medium ⊆ D_hard` holds cell-by-cell because `T_easy ≤ T_med ≤ T_hard` pointwise and the uniform `u[s, a, e]` is shared across levels. Independent togglability: Knob 3 reads the mutated source DataFrames produced by Knobs 1/5/6/7 plus the fusion gold index and writes only to source cells in the already-present columns (never removes columns — that is Knob 9's territory). It composes at position 5 of the canonical S1 order (`Knob 2 → Knob 4 → Knobs 1/5/6/7 → Knob 3 → Knob 10 → Knob 8`, per [README.md](README.md#canonical-knob-application-order)).

**Constraint resolution order** (applied *before* any cell is actually dropped, so the realized drop matrix is the fixed point of):

1. **Fusion survivor floor.** For every (entity, attribute) present in the fusion gold, mark **at least one** source as a *protected carrier* whose cell is never dropped by Knob 3 (matches Knob 4's vocabulary; additional carriers may be protected by Knobs 1/10's anchor-survivor and reshuffle-floor rules — Knob 3 unions all such protections). Selection: the source whose current post-Knobs-1/5/6/7 value is closest to the gold value under the lenient fusion evaluator (ties broken by lowest source index). Protection is per (entity, attribute), not per entity — different attributes of the same entity may be protected on different sources.
2. **Conflict-preserving drop.** Compute the post-perturbation conflict graph: for each (entity, attribute) with ≥ 2 disagreeing source values under canonical-form equality, mark at least two disagreeing sources as *conflict carriers*. Drops come from the redundant-agreement pool first; only after redundant agreeing copies are exhausted may a conflict carrier be considered, and even then only if ≥ 2 disagreeing values remain. **At easy this constraint is effectively a no-op**: the `compress` fill phase produces agreeing copies and the residual floor-rate drop is small, so redundant-agreement supply always covers the tiny drop budget. The constraint is exercised at medium and (especially) hard.
3. **Single-source-survivor cap at hard.** After (1) and (2), count (entity, attribute) cells where only one source still carries a value. If the fraction exceeds the per-domain cap (default 5%, overridable in YAML), roll back the most recent drops in reverse draw order until the cap holds. The cap is logged; persistent rollbacks emit a calibration warning (likely means `stretch_factor` is too aggressive for the domain).
4. **Per-(source, attribute) ceiling** (Companies asymmetry — see Per-domain notes below). Realized drop rate never exceeds `B[s, a]` by more than a per-domain-YAML per-cell delta; this prevents "stretch at hard" from demanding negative headroom on attributes already near 100% missing (DBpedia revenue 20.4%, Discogs `tracks_track_duration` 57%).

Collapse of the committee signal triggers the standard cross-cutting fix path: per [cross_cutting.md](cross_cutting.md#per-knob-fix-strategy-defaults), Knob 3 is **not** listed in the per-knob fix-strategy table, so it inherits the default "soften augmentation locally" action — the offending `(attribute, source)` rate is reduced by a fixed step until the committee monotonicity check passes. The fusion gold artifact is never mutated.

**Literature citations.** Tier A — no literature method is ported wholesale. Two light cites for defensibility:

- **DAPO** ([../literature-search-generation/dapo_large_scale_data_pollution/paper.md](../literature-search-generation/dapo_large_scale_data_pollution/paper.md)) — *Per-Source Corruption Profiles* (paper.md:75–76, 156–164, 323) and the *Missing value injection* operator inside the pollution taxonomy. Cited as the direct precedent for per-source, per-attribute missingness rates as a first-class benchmark-generation knob. Our measured baseline vector + (compress / identity / stretch) transform is a strict extension of DAPO's per-source profile pattern, specialized so that baseline shape is preserved and monotonicity across levels is guaranteed.
- **Christen / Vatsalan data-corruption taxonomy** ([../literature-search-generation/febrl_christen_vatsalan_corruption/paper.md](../literature-search-generation/febrl_christen_vatsalan_corruption/paper.md)) — *Missing values* as one of the taxonomy's named corruption classes, distinguished from typo / OCR / phonetic noise. Cited for the taxonomy boundary: Knob 3 owns "value is absent", Knob 6 owns "value is present but corrupted".

**No literature method beyond these cites.** Justification: the knob behavior we need (controlled, monotone, conflict-preserving, fusion-safe per-cell masking with a measured baseline floor) is a benchmark-engineering operation with no research content beyond the rate taxonomy; a deterministic pandas implementation is strictly simpler, cheaper to validate, and more reproducible than any ported library. The DAPO + Christen/Vatsalan cites cover the taxonomy anchor; the rest is plumbing.

**Determinism & provenance.**
- **Baseline measurement is refreshed every run**: the dispatcher calls an in-script `measure_baseline_missingness(sources)` function over the post-Knobs-1/5/6/7-and-Knob-4 DataFrames at the *start* of every Knob 3 invocation and writes the observed per-(source, attribute) null-rate vector to `output/baselines/knob_03_baseline_missingness.csv` inside the variant directory, timestamped and seeded. **Never cached across runs** — any source-data change (new rows, re-scraped sources, upstream pipeline edits) is automatically reflected on the next run. The measurement is cheap (one pass over source DataFrames, `isna().mean()` per column group) and is always computed fresh inside `apply_knob_03_attribute_drop.py`.
- **Reconciliation with [cross_cutting.md §Bootstrap order](cross_cutting.md#bootstrap-order-committees--baseline--calibration) (locked C5 from the Step 5 cross-knob review).** There are two distinct baseline artifacts in the synthetic-generation pipeline; Knob 3 uses one of them, not both:
  1. **Static per-domain profile baseline** — a one-shot measurement made during the algorithm-selection phase, used by the *committees* for calibration and target-rate authoring. This is what cross_cutting.md §Bootstrap order is referring to when it says "baselines are measured during the algorithm-selection phase, not earlier." It is checked in under `usecases_synthetic/baselines/<domain>/profile.json` and is read by Knob 3 *only* at YAML authoring time (to set the `rates_per_level` numbers and the `per_source_attribute_overrides` values).
  2. **Live per-run null-rate vector** — measured fresh from the *post-Knobs-1/5/6/7 + Knob 4* source frames at the start of every Knob 3 invocation. This is what the Knob 3 dispatcher actually consults at apply time, because upstream knobs in the same variant may have legitimately added or removed rows / cells. It is **not** cached and **not** related to the static profile baseline.

  Implementer guidance: read the static profile when authoring the per-domain YAML; the dispatcher itself only ever reads the live per-run vector. Mixing the two would either silently break monotonicity (using a stale static baseline against fresh post-K4 data) or break independent togglability (binding K3's behavior to a calibration artifact that K3 was designed not to depend on). The cost of a stale baseline dwarfs the compute saving — same rationale as the existing rejected-alternatives note at L134 below.
- **Cross-knob composition (locked I6 from the Step 5 cross-knob review).** Knob 3's interactions with the value-perturbation phase and Knob 10:
  - **K4 runs upstream** of Knob 3 in the canonical S1 order, so the live baseline vector reflects the post-K4 source presence — entities that K4 removed are absent from the input, and entities K4 fabricated are present.
  - **Knob 3's drops reduce Knob 10's reshufflable-cell population** (Knob 10 acknowledges this symmetrically at [knob_10_source_reliability.md:93](knob_10_source_reliability.md#L93)). A cell that K3 drops is no longer a candidate for K10 reshuffling.
  - **Knob 3 may drop K1-paraphrased cells.** The fusion-survivor floor's "closest-to-gold" carrier selection typically coincides with K1's anchor-survivor row because the K1 anchor is un-paraphrased and therefore canonical-form-equal to the gold; K3 protects that row by construction without needing a separate K1 hook.
- RNG: a single `numpy.random.default_rng(seed)` per `(domain, variant, knob=3)` tuple; seed recorded in the variant's `config/difficulty.yaml`. Draw order is the canonical `sources × attributes × entity_ids` iteration so level-independent cells do not reshuffle between level changes.
- Per-domain config file: `usecases_synthetic/config/knob_03_attribute_drop/<domain>.yaml`. Keys:
  - `attribute_classes`: `{source_name: {column: "primary" | "key" | "secondary"}}`. Authored once, checked in. Primary = canonical label, key = blocking-relevant, secondary = everything else.
  - `rates_per_level`: `{easy|medium|hard: {primary, key, secondary}}` — per-attribute-class floor rates.
  - `transform_per_level`: `{easy: "compress", medium: "identity", hard: "stretch"}` (fixed; recorded in YAML for audit).
  - `compression_factor`, `stretch_factor`: scalar hyperparameters per level (defaults 0.7, 1.5).
  - `single_source_survivor_cap_hard`: scalar (default 0.05).
  - `per_cell_ceiling_delta`: scalar (default 0.10) — global default for the max amount by which realized rate may exceed baseline on a single (source, attribute).
  - `per_source_attribute_overrides`: optional **tighter overrides** of `per_cell_ceiling_delta` on specific (source, attribute) pairs where the baseline is already at-or-near hard. Schema: `{source_name: {column: ceiling_delta}}`. Used for Companies DBpedia financials and Music LastFM / Discogs track arrays — see per-domain notes.
- Provenance written per affected cell to `output/provenance/knob_03_attribute_drop.csv` inside the variant directory, following the cross_cutting.md flat-row schema:
  ```
  (entity_id, source, attribute, original_value, new_value,
   transform_fn ∈ {drop, propagate_fill},
   transform_params, knob=3, level)
  ```
  `transform_params` JSON keys by `transform_fn`:
  - `drop`: `{reason ∈ {"floor_rate", "stretch"}, baseline_rate, target_rate}` — `floor_rate` for easy/medium drops driven by `rates_per_level`, `stretch` for hard-tier amplification above baseline.
  - `drop` skipped by constraint: emitted to a separate `output/provenance/knob_03_skipped.csv` with `{reason ∈ {"fusion_survivor_floor", "conflict_preserve", "single_source_cap", "per_cell_ceiling"}}`.
  - `propagate_fill` (easy only): `{source_from, source_to, entity_id, value_copied}`.
- The fusion gold file is **read-only**; byte-identical before and after every run.
- Caching: output is a file artifact on disk (mutated source datasets + provenance CSVs + baseline CSV). Seeded RNG + fresh baseline measurement guarantee reproducible regeneration against the *current* source snapshot.
- Committee surface: Blocking, EM, and Fusion committees (per the Committee-expectations section above and the cross_cutting.md committee mechanism) see the dropped source files exactly as written. The spread between missing-value-aware and missing-value-naive committee members is a secondary difficulty signal.

**Domain-specific adjustments.**
- **Companies** — baseline is **already at hard** on DBpedia financials (assets 6.5%, revenue 20.4%). The stretch transform at hard has negative headroom there, so `per_source_attribute_overrides` tightens `per_cell_ceiling_delta` from the default 0.10 down to 0.02 on `{dbpedia: {total_assets_val: 0.02, revenue: 0.02}}`. The overall target matrix for Companies hard is still monotone over medium elsewhere; stretch amplification is absorbed by the lower-baseline attributes. Easy on Companies uses `compress` as specified — value propagation from the best-covered source (typically Forbes for financials) into DBpedia + Country of Origin into the opposing source.
- **Games** — almost uniformly dense (≥ 87% on every attribute except DBpedia franchise at 52%). Less per-source spread to leverage, so `stretch_factor` has muted effect; hard difficulty comes primarily from the `rate_hard.*` floors, not from the stretch transform. DBpedia franchise is the one attribute where stretch produces visible movement.
- **Music** — bimodal headroom. LastFM scalar gaps are **schema-level** (Knob 9 territory, not Knob 3). The authored `attribute_classes` for Music therefore excludes LastFM-scalar-absent attributes (they are not columns in LastFM's frame at all). Knob 3's real Music surface is on the **array attributes**: `lastfm.tracks_*` at 46.6% baseline and `discogs.tracks_track_duration` at 57% are in the config with per-(source, attribute) ceilings to prevent hard stretch from demanding negative headroom. MusicBrainz-side scalars (which are denser) carry the stretch budget.
- **Movies, products**: deferred alongside the Step 6 prototype. The dispatcher no-ops on domains without a `config/knob_03_attribute_drop/<domain>.yaml` (warns in the log). No code change required when those domains come online — only a new YAML.

**Rejected alternatives.**
- **LLM-based "realistic missingness simulation"** (e.g., prompting an LLM to decide which cells a given source would plausibly omit). Rejected: the task is a parametric rate dial with a measured baseline; it has a strong pandas baseline, and LLM use would sacrifice determinism, explode validation cost, and deliver zero expected quality gain. **LLM not used because the deterministic alternative is sufficient.**
- **Heavyweight ML missing-value generators** (GReaT / CTGAN / TabDDPM conditional masking, diffusion-based tabular completion). Rejected under the plan_algorithmselection.md framework rule against heavyweight ML methods — they violate determinism, validation cost, and dependency weight simultaneously, and they optimize for realistic joint distributions, not for monotone controlled rate dials with conflict preservation and fusion-safety constraints.
- **MAR / MNAR statistical missingness models** (e.g., logistic-regression-driven MAR from the imputation literature). Rejected: these models *learn* a plausible missingness mechanism from the data, which would silently absorb the very difficulty signal we are trying to control. We explicitly want the missingness pattern to be transparently parameterized by (baseline vector, level transform), not inferred.
- **Caching the baseline vector across runs** for speed. Rejected: source data can still change between runs, so the baseline is re-measured from the current source DataFrames at the start of every Knob 3 invocation. Measurement is cheap (one `isna().mean()` pass per source) — the cost of a stale baseline (silently breaking monotonicity and per-cell ceilings) dwarfs the compute saving.
- **Column removal at hard** (i.e., dropping entire columns on a source when `drop_rate_key` gets large). Rejected: column-level omission is Knob 9 (S2 only) per the Definition section; mixing it into Knob 3 would fragment provenance and break the Knob 3 / Knob 9 boundary.

**Implementation handoff.** Everything Step 6 needs to implement this knob without re-reading surrounding cards:

- **Target script:** `usecases_synthetic/scripts/apply_knob_03_attribute_drop.py` (new, convention matches `apply_knob_05_format_unit.py`, `apply_knob_06_noise.py`, `apply_knob_08_naming.py`). Standalone runnable from repo root.
- **Function shape (illustrative):**
  ```python
  def apply_knob_03(
      domain: str,
      level: Literal["easy", "medium", "hard"],
      sources: dict[str, pd.DataFrame],           # post-Knobs-1/5/6/7
      fusion_gold: pd.DataFrame,                  # read-only; survivor-floor lookup
      config_path: Path,                          # usecases_synthetic/config/knob_03_attribute_drop/<domain>.yaml
      output_dir: Path,
      seed: int,
  ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
      """Returns (dropped_sources, provenance_df, skipped_df, baseline_df).

      baseline_df is measured fresh from `sources` at the start of every call
      and written to output/baselines/knob_03_baseline_missingness.csv.
      """
  ```
- **Inputs the script reads:**
  - Source DataFrames (via `PyDI.io.load_*`, preserving `df.attrs["dataset_name"]`), already mutated by Knobs 1/5/6/7 upstream in the canonical S1 order.
  - The fusion gold artifact from `usecases/<domain>/input/fusion/` — read-only, used for the survivor floor; **never mutated** (byte-identical before and after).
  - Per-domain config at `usecases_synthetic/config/knob_03_attribute_drop/<domain>.yaml`.
- **Outputs the script writes** (under the variant directory):
  - Mutated source files in `input/data/` (same format as input — XML/JSON/CSV).
  - Freshly measured baseline at `output/baselines/knob_03_baseline_missingness.csv` (written every run; not cached across runs).
  - Provenance log at `output/provenance/knob_03_attribute_drop.csv`.
  - Skipped-cell audit at `output/provenance/knob_03_skipped.csv` (cells spared by fusion-survivor / conflict-preserve / single-source-cap / per-cell-ceiling).
- **Pipeline integration:** Knob 3 sits at position 5 of the canonical S1 order from [README.md](README.md#canonical-knob-application-order), running after `Knobs 1/5/6/7` and before `Knob 10 → Knob 8`. It sees the perturbed values from the earlier joint phase; Knob 10 then reshuffles which source carries the (possibly dropped) gold cell, and Knob 8 finally renames headers.
- **Dependencies:** stdlib + `pandas` + `numpy` + `pyyaml`. No new runtime dependencies. No PyDI extension points.
- **Authoring task before first run:** populate `usecases_synthetic/config/knob_03_attribute_drop/{companies,games,music}.yaml` with `attribute_classes`, `rates_per_level`, `compression_factor`, `stretch_factor`, `single_source_survivor_cap_hard`, `per_cell_ceiling_delta`, and any `per_source_attribute_overrides` implied by the Per-domain notes above. Baseline missingness is **measured**, not authored.
- **Smoke test:** for each domain with a config, run the script at all three levels and assert (a) the fusion gold file on disk is byte-identical before and after the run, (b) for every (entity, attribute) in the fusion gold, at least one source retains a non-null value, (c) realized per-(source, attribute) drop rates are monotone non-decreasing easy → medium → hard up to constraint-induced rollbacks, (d) primary-attribute drop rate is 0 at easy and medium, (e) at hard, the fraction of single-source-survivor cells does not exceed `single_source_survivor_cap_hard`, (f) the provenance row count equals the number of cell-value mutations, (g) the baseline CSV exists and was written during the current run (mtime ≥ start-of-run), (h) no column is ever removed (columns are Knob 9's territory), (i) re-running with the same seed on the same source snapshot produces bit-identical outputs.
