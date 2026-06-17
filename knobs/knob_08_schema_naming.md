# Knob 8 — Schema naming divergence

**Status:** LOCKED. **Scenario:** S1 + S2 (fully controllable, bidirectional).

## Definition

Distance of source attribute *names* from the target schema. Header-level only (Knob 9 owns column presence/distractors in S2). Affects the SM stage almost exclusively.

## Dimensions controlled

- Naming Heterogeneity (SM)

## Sub-parameter (single)

- `naming_distance` per source — categorical level on a small ladder:
  - **descriptive** — target-like English (`release_date`, `country`)
  - **abbreviated** — short but still meaningful (`rel_dt`, `cntry`)
  - **cryptic** — domain jargon / internal codes (`rdate`, `iso2`, `c_code`)
  - **anonymized** — opaque (`Attribute_3`, `col_07`) — companies-FullContact is the only naturally anonymized source in the corpus (see *Per-domain notes* below for the cross-domain spread story and how it shapes Companies authoring)

## Per-source shape via measured baseline

Same pattern as Knobs 3 / 6: classify each source's current naming level (manual, small N), then transform per level. Bidirectional in S1: easy may require *renaming-up* anonymized columns to descriptive; hard may require *renaming-down* descriptive columns.

## Easy / Medium / Hard

| Level | Target state | Per-source transform |
|---|---|---|
| **Easy** | All sources at *descriptive*. Headers near-trivially alignable to the target by string similarity / dictionary lookup. SM stage exists but is one-shot. | Rename-up: anonymized/cryptic columns get descriptive names propagated from the target schema, using the existing SM mappings as the renaming oracle. |
| **Medium** | Mixed: at least one source *descriptive*, at least one *abbreviated* or *cryptic*. SM needs token-level normalization plus light semantic matching. | Identity for mixed-baseline domains; light shifts otherwise. |
| **Hard** | At least one source *anonymized*; remaining sources *cryptic* or *abbreviated*. No source *descriptive*. SM must rely on **value-based / instance-based matching** because headers carry minimal signal. | Rename-down: descriptive columns rewritten to cryptic/anonymized variants per source via a deterministic per-domain renaming table. |

## Determinism

Anonymization at hard is **deterministic, seeded per domain**. The renaming table (`release_date → Attribute_3`, etc.) is checked into the difficulty config. If overfitting becomes a concern later, a randomized variant can be added behind the same knob.

## Composition

- **All value knobs (1, 5, 6, 7, 10):** orthogonal — Knob 8 touches headers only.
- **Knob 3 / 4:** orthogonal (presence vs naming).
- **Knob 9 (S2 only):** Knob 9 adds/removes columns; Knob 8 renames the survivors. Compose **Knob 9 first, then Knob 8**. **Distractor-renaming merge contract (locked C4 from the Step 5 cross-knob review, see [plan_algorithmselection.md:111](../plan_algorithmselection.md#L111)):** in S2, Knob 9 adds *distractor* columns (`distractor_unrelated`, `distractor_type_matched`, `ambiguous_candidate_for:<target>`) whose catalog entries each carry a `rename_rungs: {descriptive, abbreviated, cryptic, anonymized}` block. Knob 9's dispatcher must merge those `rename_rungs` entries into Knob 8's per-domain rename table at the *active level's rung* **before** Knob 8 runs. If this merge is skipped, the result is anonymized genuine columns sitting next to plain-English distractor headers — a label-leakage failure mode that destroys the SM-isolation story for the variant (an SM matcher would trivially pick the descriptive distractor over the anonymized genuine column on header similarity alone). The merge is Knob 9's responsibility (Knob 9 owns the catalog and the dispatch order); Knob 8 simply consumes the augmented rename table. See [knob_09_schema_completeness.md:198](knob_09_schema_completeness.md#L198) for the symmetric specification on the Knob 9 side.
- **Pipeline order:** **last** in the canonical order (header-only, orthogonal, cache-locality with SM-stage tests). **Knob 8 is orthogonal / does not alter earlier invariants** — it never reads or writes cell values, row presence, fusion gold, or expanded_positives, so every load-bearing constraint established by Knobs 1–7 / 9 / 10 (anchor-survivor floor, fusion-gold survivor floor, joint cell-collision index, reshufflable-cell predicate) survives Knob 8 untouched.

## Test-set treatment

SM mappings (locked I3, mirrors [cross_cutting.md:23](cross_cutting.md)): **In S1**, the SM mapping is unchanged unless Knob 8 is actively perturbing headers at a non-identity level for a given source; in that case the mapping is regenerated for the perturbed sources only (underlying column→target correspondence preserved, only LHS strings change). **In S2**, the SM mapping is regenerated unconditionally alongside Knob 9 (which adds/removes columns before Knob 8 renames the survivors). Fusion / EM gold untouched in both scenarios.

## Fusion safety

N/A — Knob 8 doesn't touch values. Fusion committee flat across levels.

## Committee expectations

- **SM:** **primary target.** Monotone drop expected. String-similarity matchers collapse fast on cryptic/anonymized; instance-based / embedding matchers degrade more gracefully. The spread between matcher types *is* the difficulty signal.
- **Blocking / EM / Fusion / Normalization:** flat.

## Per-domain notes — three domains span the entire scale at baseline

- **Companies:** **at hard** at baseline. Forbes ≈ abbreviated, DBpedia ≈ abbreviated/cryptic, FullContact = anonymized (`Attribute_1..6`). Easy requires renaming-up FullContact *and* DBpedia.
- **Games:** **at medium**. All three sources sit in the same abbreviated/cryptic band — no descriptive source, no anonymized source. Easy requires renaming-up *all three*; hard requires renaming-down at least one. Symmetric headroom.
- **Music:** **at easy**. All three music sources are descriptive at baseline (Step 5 corrected the earlier "MusicBrainz is anonymized" assumption — MusicBrainz uses descriptive English XML element tags `title, artist, date, country, status, quality, packaging`; only `rel_id` is mildly abbreviated). Hard requires active rename-down across all three.

**This is the cleanest cross-domain spread of any knob in the corpus** — companies / games / music span the full Knob 8 scale at baseline.

## Provenance

Column-level (not row-level). One record per renamed column:

```
(source, original_header, new_header,
 transform_fn ∈ {rename_descriptive, rename_abbreviated, rename_cryptic,
                 rename_anonymize, rename_up_from_mapping},
 transform_params={baseline_rung, target_rung, oracle},
 knob=8, level)
```

`baseline_rung` and `target_rung` are first-class top-level columns in the CSV (alongside `source`, `original_header`, `new_header`, `transform_fn`, `knob`, `level`); they are also echoed inside `transform_params` for self-containment when a row is consumed in isolation. The 5-value `transform_fn` enum here is the single source of truth — it supersedes the 4-value enum that previously appeared in this header block. Plus the regenerated SM mapping file.

## Algorithm selection

**Chosen approach.** Tier A — fully deterministic. The "algorithm" is a small set of hand-authored per-domain renaming tables (one entry per `(source, original_header)` at each level on the `descriptive → abbreviated → cryptic → anonymized` ladder), each domain in its own YAML file under `usecases_synthetic/config/knob_08_renaming/<domain>.yaml` so files are pulled in only when that domain runs. At generation time a thin pandas/Python dispatcher walks each source's columns and rewrites headers via the table for the active level; for *easy* on sub-descriptive sources, it walks the existing SM mapping in reverse and propagates the target-schema column name back onto the source. No similarity computation, no clustering, no LLM call. The literature contribution lives in *which* renames we author (taxonomy below), not in *how* we apply them — the dispatcher is plumbing.

**Mapping to easy/medium/hard.** A single `level` parameter selects which column of the per-domain renaming table is read for each source. The table columns are authored as a strict ladder, so monotonicity is by construction.

| Level | Per-source target rung | Generator action |
|---|---|---|
| **Easy** | All sources at *descriptive* | For sub-descriptive sources, look up `target_col := SM_mapping[source_col]` and rename `source_col → target_col`. Sources already at *descriptive*: identity. |
| **Medium** | Mixed: ≥1 *descriptive*, ≥1 *abbreviated*/*cryptic*, no *anonymized*. Per-domain assignment of which source lands where is authored in the `level_assignments` section of the per-domain YAML (see schema below). | Identity for sources already in band; otherwise lookup-based rename, **bidirectional** — sources baselined above the medium target rung get *renamed-down* (descriptive → abbreviated/cryptic via the per-domain table), and sources baselined below get *renamed-up* (cryptic/anonymized → descriptive via the SM-mapping oracle). The Companies note at L117's "rename-up Forbes from abbreviated" path is a medium-rung instance of this bidirectional move. |
| **Hard** | ≥1 *anonymized*, remainder *cryptic*/*abbreviated*, no *descriptive* | Rename-down: replace each surviving column with the entry from the *cryptic* / *anonymized* column of the per-domain table. *Anonymized* uses `Attribute_<n>` where `n` is keyed by `(domain, source, original_header)` so re-runs are bit-identical. |

Independent togglability: the rename code path takes only `level` and an optional per-source override map; it never reads any other knob's output and writes only to column headers, so it composes orthogonally with all value knobs.

**Literature citations.**
- **XBenchMatch** ([../literature-search-generation/xbenchmatch_schema_matching/paper.md](../literature-search-generation/xbenchmatch_schema_matching/paper.md)) — *Schema Heterogeneity Classification and Injection*. Provides the element-level "Naming Conflicts" taxonomy (`naming_operations ∈ {synonym, acronym, abbreviation, tokenization, language}`) that justifies the four-rung ladder and constrains what each rung is allowed to contain. The `descriptive → abbreviated → cryptic → anonymized` progression is the XBenchMatch taxonomy ordered by string-matcher signal loss.
- **Valentine** ([../literature-search-generation/valentine_schema_matching/paper.md](../literature-search-generation/valentine_schema_matching/paper.md)) — *Fabricated Benchmark Generation via Table Transformations*, specifically the `column_rename` primitive. Establishes that deterministic dictionary-based renaming with construction-time ground truth is the standard primitive for schema-matching benchmark generation. Our SM-mapping regeneration step is the same construction.
- **DAPO** ([../literature-search-generation/dapo_large_scale_data_pollution/paper.md](../literature-search-generation/dapo_large_scale_data_pollution/paper.md)) — *Schema Inhomogeneity Injection*. Cited as precedent for combining naming-convention changes with the rest of a controlled pollution profile, since Knob 8 is one rung of a multi-knob ladder rather than a standalone benchmark.

**Determinism & provenance.**
- Renaming tables live as static YAML at `usecases_synthetic/config/knob_08_renaming/<domain>.yaml` — one file per domain. Each file has two top-level sections:
  - `rename_table`: `{source_name: {original_col: {descriptive: str, abbreviated: str, cryptic: str, anonymized: str}}}`. Every `(source, original_col)` pair authors all four rung variants.
  - `level_assignments`: `{easy: {source_name: rung}, medium: {source_name: rung}, hard: {source_name: rung}}` — which rung each source lands on at each level. This is the *assignment* map; the `rename_table` is the *lookup* map. The dispatcher reads `level_assignments[level][source]` to decide which column of the `rename_table` to consult for that source.
  - Both sections authored once per domain and version-controlled.
- **Incomplete SM mapping fallback (easy).** At easy, the dispatcher's first choice is `target_col := SM_mapping[source_col]`. For columns with no SM entry (distractors, source-specific attributes), the dispatcher falls back to `rename_table[source][col][descriptive]`. If that key is also missing inside an otherwise-present per-domain YAML, the dispatcher **raises** — silent identity passes at easy are a bug, not a feature. This raise applies *only* when keys are missing inside a present YAML; the entirely-missing-domain-YAML case (movies / products before they are authored) is handled by the marker-file fallback in §Implementation handoff and is **not** an error.
- The variant's `config/difficulty.yaml` (per [../plan.md](../plan.md#scenario-1-augmented-use-cases)) records only the active `knob_8.level` and any per-source override; the bulky table stays in the per-domain file so the difficulty config remains scannable.
- No RNG required at apply time. The only quasi-random step is the *anonymized* index generator, which is pure: `n = stable_hash(domain, source, original_header) mod table_size`. Collisions are detected at authoring time and resolved by picking the next free index in the table.
- Provenance records (column-scoped) are written to `output/provenance/knob_08_naming.csv` inside the variant directory:
  ```
  (source, original_header, new_header,
   transform_fn ∈ {rename_descriptive, rename_abbreviated, rename_cryptic,
                   rename_anonymize, rename_up_from_mapping},
   transform_params, knob=8, level)
  ```
  `transform_params` is a JSON-encoded string column capturing `{baseline_rung: str, target_rung: str, oracle: "sm_mapping" | "rename_table" | "stable_hash"}` so the debugger can reconstruct *where* the new header came from without re-running the dispatcher. For `rename_up_from_mapping`, `oracle = "sm_mapping"` and the baseline rung is recorded; for `rename_anonymize`, the `stable_hash` index is included.
  Plus the regenerated SM mapping artifact at `input/schemamatching/<source>_to_target.csv` inside the variant directory so the SM stage of the pipeline runs unchanged. The mapping is regenerated by composing `original SM mapping ∘ inverse(rename table)` — pure function, no learning step.
- Caching: trivial — renaming tables and the regenerated mapping are file artifacts on disk; no per-row cache.
- Committee surface: the SM committee sees only renamed headers and the regenerated mapping; underlying values are untouched, so the committee delta is fully attributable to Knob 8 (as the "Committee expectations" section above already demands).

**Domain-specific adjustments.** Same dispatcher across domains; only the renaming-table contents differ. Baselines from the per-domain notes section above:
- **Companies** (baseline at *hard*): Easy requires renaming-up Forbes (abbreviated → descriptive), DBpedia (abbreviated/cryptic → descriptive), and FullContact (`Attribute_1..6` → descriptive) via the existing SM mappings as oracle. Medium partially renames-up DBpedia/FullContact while keeping Forbes near-abbreviated. Hard is identity.
- **Games** (baseline at *medium*, symmetric headroom): Easy renames-up all three via SM mappings. Medium is identity. Hard renames-down at least one source to *anonymized* (`Attribute_<n>`); the others go to *cryptic*. The choice of which source to anonymize is fixed in the difficulty config to keep cross-run symmetry.
- **Music** (baseline at *easy*): Easy is identity. Medium renames-down one source to *abbreviated*, leaves the other two *descriptive*. Hard renames-down all three; at least one to *anonymized*. MusicBrainz's `rel_id` (already mildly abbreviated) is reused as a *cryptic* seed.
- **Movies, products**: renaming tables to be authored alongside Step 6 prototyping using the same dispatcher. No code change.

**Rejected alternatives.**
- **Magneto LLM-based column-name generator** ([../literature-search-generation/magneto_schema_matching/paper.md](../literature-search-generation/magneto_schema_matching/paper.md)) — generates 5–20 LLM-authored variants per column. Rejected: a hand-authored table per domain is small (~30 columns × 4 rungs × 3 sources ≈ a few hundred entries), one-time work, fully deterministic, and avoids paraphrase drift across runs. The benchmark needs *one* canonical rename per `(source, level)`, not a population of paraphrases. **LLM not used because the deterministic alternative is sufficient.**
- **iBench parameterized schema generator** ([../literature-search-generation/ibench/paper.md](../literature-search-generation/ibench/paper.md)) — full schema-mapping primitives (copy/projection/join/union). Overkill: PyDI is 1:1 schema matching and Knob 8 only touches headers. iBench's primitives target structural transformations explicitly out-of-scope per [README.md](README.md#knob-index).
- **Embedding-based renaming via sampled headers** — would re-introduce a similarity computation that the per-domain tables make unnecessary, and would force us to defend a particular embedding model as the authoring oracle. Determinism win is lost for no gain.
- **Heavyweight ML methods (CTGAN, GReaT, BART-error, diffusion)** — schema naming has no generative-model angle; rejected by the framework rule in [../plan_algorithmselection.md](../plan_algorithmselection.md#decision-framework-deterministic-in-house-vs-literature-method-vs-llm).

**Implementation handoff.** Everything Step 6 needs to implement this knob without re-reading the surrounding cards:

- **Target script:** `usecases_synthetic/scripts/apply_knob_08_naming.py` (placeholder already listed in [`../usecases_synthetic/PIPELINE.md`](../usecases_synthetic/PIPELINE.md#phase-2--scenario-1-augmented-use-cases) Phase 2). Standalone runnable from repo root, follows the convention notes at the bottom of `PIPELINE.md`.
- **Function shape (illustrative, not prescriptive):**
  ```python
  def apply_knob_08(
      domain: str,
      level: Literal["easy", "medium", "hard"],
      sources: dict[str, pd.DataFrame],          # source_name -> dataframe
      sm_mapping: dict[str, dict[str, str]],     # source_name -> {source_col -> target_col}
      renaming_table_path: Path,                 # usecases_synthetic/config/knob_08_renaming/<domain>.yaml
      output_dir: Path,                          # variant directory root
      per_source_override: dict[str, str] | None = None,
  ) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, str]], pd.DataFrame]:
      """Returns (renamed_sources, regenerated_sm_mapping, provenance_df)."""
  ```
- **Inputs the script reads:**
  - The original source DataFrames (loaded via `PyDI.io.load_*`, `df.attrs["dataset_name"]` preserved).
  - The original SM mapping CSVs from `usecases/<domain>/input/schemamatching/`.
  - The per-domain renaming YAML at `usecases_synthetic/config/knob_08_renaming/<domain>.yaml`.
  - The active `level` and any `per_source_override` from the variant's `config/difficulty.yaml`.
- **Outputs the script writes** (under the variant directory `usecases/<domain>-augmented/<level>/`):
  - Renamed source files in `input/data/` (same format as the original — XML/JSON/CSV — only headers / element tags changed).
  - Regenerated SM mapping CSVs in `input/schemamatching/<source>_to_target.csv`.
  - Provenance log at `output/provenance/knob_08_naming.csv` with the column schema given above.
- **Pipeline integration:** Knob 8 runs **last** in the canonical S1 order (see [README.md](README.md#canonical-knob-application-order)). It reads whatever the previous knobs produced and only mutates column headers + the SM mapping artifact, so it never conflicts with value perturbations or row drops. In S2, Knob 9 runs first and fixes the column set; Knob 8 then renames the survivors. Both knobs share the same provenance directory but write to disjoint files.
- **Dependencies:** stdlib + `pandas` + `pyyaml`. No PyDI extension points needed (no new `BaseSchemaMatcher` subclass — Knob 8 prepares input *for* schema matching, it doesn't *do* schema matching).
- **Authoring task before first run:** populate `usecases_synthetic/config/knob_08_renaming/{companies,games,music}.yaml` using the per-domain notes section above as the source of truth for which baseline rung each source sits at. **In S2, the rename table is augmented at runtime by Knob 9's distractor-catalog `rename_rungs` blocks** (see the C4 distractor-renaming merge contract under §Composition); Knob 8 reads the merged table, not the on-disk YAML alone, so the YAML only needs to author the genuine columns. Movies and products tables can be authored later — if a domain has no renaming file, the dispatcher logs a `WARNING`, writes a marker file at `output/provenance/knob_08_MISSING_RENAMING_TABLE.txt` inside the variant directory, and runs identity on headers. The marker file makes the no-op visible to anyone auditing the pipeline output.
- **XML sources (MusicBrainz, movies):** DataFrames loaded via `PyDI.io.load_xml` preserve element-tag → column-name identity, but re-serialising requires the XML writer to honour the renamed columns as the new element tag names. The dispatcher re-serialises XML sources by (a) loading the original XML tree once, (b) rewriting element tag names in place via the same `rename_table` lookup used for the DataFrame columns, (c) writing the tree back. The DataFrame returned to the caller has the renamed columns; the on-disk XML has the renamed tags. Both must agree — asserted in the smoke test below.
- **Smoke test:** for each domain with a renaming YAML, run the script at all three levels and assert:
  - (a) every column in every renamed source resolves under the regenerated SM mapping to a target column,
  - (b) the provenance row count equals the total renamed-column count (excluding identity passes),
  - (c) the *easy* output's headers equal the target schema column names for sources whose baseline is below descriptive,
  - (d) for Companies at easy, FullContact's SM mapping is total (`Attribute_1..6` all have SM entries) — because anonymized baselines force total SM coverage for the rename-up path to succeed,
  - (e) for XML sources, the on-disk element tag names match the returned DataFrame column names after renaming.
