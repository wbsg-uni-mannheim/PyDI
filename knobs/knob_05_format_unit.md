# Knob 5 — Format / unit diversity

**Status:** LOCKED. **Scenario:** S1 + S2 (fully controllable).

## Definition

Number of distinct *structured/parseable* formats and units per attribute. Covers dates, numbers, currencies, locale conventions. Distinct from Knob 1 (free-text paraphrase) and Knob 6 (errors): every value here is machine-parseable and semantically exact, just expressed in a different canonical form. Categorical case/spelling variants belong to Knob 1.

## Dimensions controlled

- Format Heterogeneity (Norm) — primary
- Unit & Scale Diversity (Norm)
- residual Representation Heterogeneity (Block)

## Sub-parameters

- `formats_per_attribute` — distinct formats coexisting across sources for a format-bearing attribute (dates, numerics, currencies).
- `unit_diversity` — for measurable quantities (revenue, sales, durations), how many units coexist (USD/EUR; thousands/millions; minutes/hh:mm:ss).
- `locale_mix` — single locale vs mixed (`1,234.56` vs `1.234,56`; `MM/DD/YYYY` vs `DD.MM.YYYY` vs ISO).

## Easy / Medium / Hard

| Level | Target state | Generator action |
|---|---|---|
| **Easy** | 1–2 formats per format-bearing attribute, **consistent within a source**. Single unit per measurable attribute, single locale. Normalization is a one-liner per source — but **not** a no-op. | Per-source format assignment from a tiny pool (≤2). Often requires light reformatting away from baseline. |
| **Medium** | 2–3 formats per attribute, consistent within a source. 1–2 units coexist for measurable quantities (unambiguous conversion factor). Locale split along source lines. | Per-source format assignment from a small pool. |
| **Hard** | 3+ formats per attribute, **inconsistent within a source** (a source mixes formats row-to-row). Multiple units coexist. Locale conventions mixed within a source. Edge cases: 2-digit years, K/M/B suffixes vs raw integers, currency symbols vs ISO codes. **No deliberately ambiguous values** (e.g. no `01/02/03`-style locale-ambiguous dates — that bleeds into Knob 7). | Per-row format/unit randomization from an expanded pool. |

**Easy is not a no-op.** The normalization stage always has work to do.

## Unit conversion at hard — Option (a) with real rates

Conversions are baked in using **real published exchange rates / unit factors as of 2026-03-15**. The generator stores the rate table as a per-domain artifact alongside the difficulty config. Conversions are deterministic; provenance records `from_unit`, `to_unit`, `rate`, `rate_date`. Sources do not carry explicit unit tags — the normalizer must infer the unit from column name / context, as in real-world data.

## Composition

- **Knob 1 (paraphrase):** orthogonal — Knob 1 touches free-text fields, Knob 5 touches structured/parseable fields.
- **Knob 6 (noise):** boundary — a malformed date that no parser accepts is Knob 6; a different valid format is Knob 5. If a Knob 5 transform produces something a parser would reject, it belongs in Knob 6.
- **Knob 3 / 4:** orthogonal (presence vs format).

## Fusion safety

Trivial — canonical-form comparison absorbs format/unit differences. Lenient fusion already handles this.

## Committee expectations

- **SM:** unaffected.
- **Blocking:** monotone drop for blockers that key on raw string form; near-zero impact for blockers that normalize first.
- **EM:** monotone drop for non-normalizing comparators; minimal for type-aware comparators.
- **Fusion:** monotone drop for naive strategies; minimal for canonicalizing strategies. The *spread* between naive and canonicalizing committee members is itself a difficulty signal.
- **Normalization stage:** **primary target.** Knob 5 is the main lever for stressing the normalization stage of the PyDI pipeline.

## Per-domain notes

- **Companies:** **at hard already on financials** — DBpedia `total_assets_val` mixes magnitudes inside a single column (`8.0`, `240560000000.0`, …). Easy on financials requires active normalize-down inside one source. Dates uniform (ISO).
- **Games:** **below easy** — entirely uniform across sources. Score scales agree, date formats agree, sales unit single. **Cleanest demonstration of "easy is not a no-op"** — every level requires active heterogeneity injection.
- **Music:** **at hard already** for dates (Discogs has 4 coexisting date format families inside one column; MusicBrainz has 3 precision levels inside one column) and durations (MusicBrainz integer ms vs Discogs `mm:ss`). **Music is the natural showcase domain for Knob 5.**

## Provenance

`transform_fn ∈ {reformat_date, reformat_number, reconvert_unit, relocale}`, `transform_params={from_format, to_format, rate?, rate_date?}`. (`reformat_number` was missing from this top-level enum and is the authoritative addition — see §Algorithm selection for the per-`transform_fn` parameter schema.)

## Algorithm selection

**Chosen approach.** Tier B — deterministic in-house dispatcher over a fixed taxonomy of structured-format and unit operators, parameterised by per-domain format pools and a real-world unit/FX rate table. No LLM, no ML. The dispatcher is a pure pandas/stdlib (`datetime`, `babel`/`locale`, `decimal`) function that classifies each target attribute into a *format family* (date, number, currency/money, duration, dimensional quantity) and then draws, per (source, attribute) at easy/medium and per (row, attribute) at hard, one canonical format + unit from a level-specific pool. Every emitted value is verified against the canonical semantic value of the input to guarantee the Knob 5 vs Knob 6 boundary: for lossless transforms (`reformat_date`, `reformat_number`, `relocale`) the emitted value must parse back to the exact same canonical value; for lossy transforms (`reconvert_unit` with FX or unit-factor conversion) the emitted value must parse back and, after reapplying the logged `rate` / `factor`, equal the canonical value within floating-point tolerance. The round-trip verification reads `from_unit` and `rate` from its own just-emitted provenance row (the dispatcher writes the provenance row first, then calls the verifier on it), not from inferred column context — this avoids any ambiguity when a column carries mixed units within a single source at hard. On failure the dispatcher rejects the draw and retries; a persistent failure logs a skipped-cell audit and falls back to identity. The literature contribution lives in the *taxonomy of format families and the catalogue of transformation classes* — our wrapper is plumbing. The operator catalogue:

| `transform_fn` | Format family | Operators / pool | Notes |
|---|---|---|---|
| `reformat_date` | date | ISO (`YYYY-MM-DD`), US slash (`MM/DD/YYYY`), EU dot (`DD.MM.YYYY`), EU slash (`DD/MM/YYYY`), long-form English (`Month D, YYYY`), compact (`YYYYMMDD`), 2-digit year variants, precision downgrades (`YYYY-MM`, `YYYY`) | 2-digit year + precision downgrade are **hard-only**. Output always parseable by `dateutil.parser.parse` with the format hint stored in provenance. Explicitly **not** emitted: locale-ambiguous patterns like `01/02/03` (those are Knob 7 territory; the dispatcher has a deny-list). |
| `reformat_number` | number | Decimal/thousands separator pools: `1234567.89`, `1,234,567.89`, `1.234.567,89`, `1 234 567,89`; precision variants; scientific notation (hard-only); K/M/B suffixes (hard-only, with magnitude-preserving semantics) | Drawn from a locale pool keyed by `locale_mix`. |
| `reconvert_unit` | currency / money | Per-domain currency pool (`USD`, `EUR`, `GBP`, `JPY`, …); magnitude scales (`1`, `×10^3`, `×10^6`, `×10^9`); symbol vs ISO code (`$`, `€` vs `USD`, `EUR`); symbol position (prefix vs suffix). Conversions use the per-domain FX rate table (`rate_date = 2026-03-15`) | **Sources carry no explicit unit tag** — the normalizer infers from column name / context, per the "Unit conversion at hard" section above. Symbols vs ISO codes are hard-only. |
| `reconvert_unit` | duration | `seconds` (int), `minutes` (float), `HH:MM:SS`, `MM:SS`, `Xm Ys` human-readable | Per-domain pool; Music showcases this. |
| `reconvert_unit` | dimensional (weight, length, file-size, bytes) | SI ↔ imperial ↔ domain idiomatic (`kg`↔`lbs`, `km`↔`mi`, `MB`↔`MiB`) from a unit-factor table | Used by Companies (revenue magnitude) and Products (package weights, dimensions). |
| `relocale` | locale-aware composite | Applies a locale tag (`en_US`, `en_GB`, `de_DE`, `fr_FR`, `ja_JP`) that fixes date + number + currency conventions jointly for the source (easy/medium) or per-row (hard) | At easy/medium: one locale per source. At hard: locale-mix enabled via per-row draw. |

Phonetic / textual case variants, free-text reformatting, and any operator whose output is *not* losslessly parseable back to the canonical value are **rejected** — those belong to Knob 1 (paraphrase) or Knob 6 (noise). Deliberately ambiguous values belong to Knob 7.

**Mapping to easy/medium/hard.** Monotonicity is enforced by strictly non-shrinking format pools and strictly non-decreasing per-cell draw granularity (source → row). A single `level` parameter selects a frozen tuple `(pool_size_per_family, within_source_consistency, operator_set, locale_mix, unit_pool)` from a per-domain YAML.

| Level | `formats_per_attribute` | Within-source consistency | Unit diversity | Locale | Operators enabled |
|---|---|---|---|---|---|
| **Easy** | 1–2 | **Consistent within source** (one draw per source) | 1 unit per measurable attribute | Single locale across all sources | `reformat_date` (ISO + one alternative), `reformat_number` (single locale), `relocale` (identity). **Not a no-op:** for domains where baseline is already heterogeneous (Music dates, Companies financials) the *easy* pass **normalizes down** — a single canonical format is chosen and all values are rewritten to it, logged as `reformat_*` with `direction=normalize_down`. |
| **Medium** | 2–3 | Consistent within source | 1–2 units with unambiguous conversion factor | Locale split along source lines (one locale per source) | Easy set ∪ full `reformat_number` locale pool ∪ `reconvert_unit` with 2 units and ISO codes only. |
| **Hard** | 3+ | **Inconsistent within source** (per-row draw) | Multiple units coexist | Locale mixed within source (per-row draw) | Full catalogue: 2-digit years, precision downgrades, scientific / K/M/B suffixes, symbol-vs-ISO currency, symbol position, full unit pool. Deny-list for locale-ambiguous date patterns remains active. |

Independent togglability: Knob 5 reads only the cell values passed to it plus the fusion gold index (not mutated — see below) and writes only to source-row cells in format-bearing columns. It composes inside the `Knobs 1/5/6/7` joint phase of the canonical S1 order per [README.md](README.md#canonical-knob-application-order); it runs **before** Knob 6 so that noise operators see the reformatted values. The boundary invariant is one-way: Knob 5 must emit values that are parseable at the moment of emission (guaranteed by the verification step above); Knob 6 is then free to noise those cells and legitimately make them unparseable — that is precisely Knob 6's job and is the normal composition, not a violation. The Knob 5 → Knob 6 ordering ensures provenance correctly attributes "format changed" to Knob 5 and "then corrupted" to Knob 6 on the same cell. Knob 1 paraphrase targets free-text fields only, so collisions are structurally impossible on format-bearing columns; the dispatcher still checks the joint provenance index and skips if a cell has already been touched by Knob 1 on a non-free-text column (defensive).

**Fusion-safety handling.** Per [cross_cutting.md](cross_cutting.md#per-knob-fix-strategy-defaults), Knob 5 is *trivial* — canonicalizing comparison absorbs format/unit differences, so no fix-on-collapse loop is needed. The fusion gold file is **never mutated**; the dispatcher reads it only to verify that every emitted reformatted value round-trips to the same canonical value as the gold entry under canonical-form comparison. Failures log to the skipped-cell audit and fall back to identity.

**Literature citations.**
- **DAPO** ([../literature-search-generation/dapo_large_scale_data_pollution/paper.md](../literature-search-generation/dapo_large_scale_data_pollution/paper.md)) — *Format Variation* inside *Schema Inhomogeneity Injection* (paper.md:174, 286, 315) and *Per-Source Corruption Profiles* (paper.md:77–78, 324). Cited as the direct precedent for per-source format pools (date formats, number formats, decimal separators, currency symbols) coexisting across sources in a controlled benchmark. Our per-source (easy/medium) and per-row (hard) draw patterns are a strict extension of DAPO's per-source profile.
- **Valentine — Fabricated Benchmark Generation via Table Transformations** ([../literature-search-generation/valentine_schema_matching/paper.md](../literature-search-generation/valentine_schema_matching/paper.md)) — *Value corruption* sub-operation, specifically the "format changes (date format, number format)" branch (paper.md:121, 171). Cited as the benchmark-construction precedent: format changes are a named, citeable transformation class with known ground truth.
- **XBenchMatch — Schema Heterogeneity Taxonomy** ([../literature-search-generation/xbenchmatch_schema_matching/paper.md](../literature-search-generation/xbenchmatch_schema_matching/paper.md)) — *Granularity operations* (precision_change, split, merge — paper.md:185) and the *Different precision* / *Different measurement units* heterogeneity classes (paper.md:105, 108). Cited as the taxonomy source for our format-family classification (precision downgrades, unit diversity as first-class heterogeneity classes).
- No LLM paper cited. Format and unit rewriting is mechanical and fully deterministic; LLM use would sacrifice determinism with no expected quality gain. See Rejected Alternatives.

**Determinism & provenance.**
- RNG: a single `numpy.random.default_rng(seed)` per `(domain, variant, knob=5)` tuple; the seed is written into the variant's `config/difficulty.yaml`. Re-runs under the same config are bit-identical. Pool draws at easy/medium are made once per (source, attribute); at hard they are made per (row, attribute) from the same generator in a fixed iteration order (`sources × attributes × rows`) so that independent level changes do not resample unrelated cells.
- Per-domain config file: `usecases_synthetic/config/knob_05_format_unit/<domain>.yaml`. Keys:
  - `attribute_classes`: `{source_name: {column: format_family}}` with `format_family ∈ {date, number, money, duration, dimensional}`. Authored once, checked in. **Shared with Knob 10**: Knob 10's dispatcher reads this block as its single source of truth for canonical-form comparator routing and owns the per-source-nesting collapse + family→comparator routing. See [knob_10_source_reliability.md §Reconciliation with Knob 5's `attribute_classes` taxonomy](knob_10_source_reliability.md). Authoring rule: if multiple sources declare the same attribute with different `format_family`, Knob 10 will warn and use the majority — keep families consistent across sources for any given attribute.
  - `format_pools_per_level`: `{easy|medium|hard: {format_family: [format_id, ...]}}`. Authored from the Easy/Medium/Hard table above. **Pinned pool sizes (frozen scalars per family per level, enforced by the YAML loader):** easy = exactly 2, medium = exactly 3, hard = exactly 4 (where `4 = 3 + at least one hard-only operator: 2-digit year, precision downgrade, K/M/B suffix, scientific notation, or symbol-vs-ISO currency). The `1–2 / 2–3 / 3+` ranges in the level table are *display* ranges; the per-domain YAML must commit to the frozen scalar.

  - **Baseline format profile schema** (consumed via the `baseline_format_profile` input). The Step 5 baseline pass writes a JSON document at `usecases_synthetic/baselines/<domain>/format_profile.json` with shape:
    ```json
    {
      "<source_name>": {
        "<column>": {
          "format_family": "date" | "number" | "money" | "duration" | "dimensional",
          "observed_formats": ["<format_id>", ...],            // ranked descending by frequency
          "dominant_format_share": 0.0,                        // float in [0,1] — share of the top format
          "normalize_down_required": false                     // true iff dominant_format_share < normalize_down_threshold
        }
      }
    }
    ```
    `normalize_down_threshold` is a per-domain scalar (default 0.85) — if no single format covers ≥85% of cells, easy must take the normalize-down branch. Forward reference: written by `usecases_synthetic/scripts/measure_baseline_profile.py` (Step 5 baseline measurement pass — same script Knob 6 forward-references for its baseline noise rates).
  - `locale_pool_per_level` and `within_source_consistency`: `{easy: "source", medium: "source", hard: "row"}`.
  - `unit_pool_per_level`: `{easy|medium|hard: {attribute: [unit, ...]}}`.
  - `baseline_format_profile`: **measured** (not authored) — per (source, attribute) the dominant format(s) observed in the raw data, written by the Step 5 baseline measurement pass (see the `measure_baseline_profile.py` cross-cutting follow-up in [plan_algorithmselection.md](../plan_algorithmselection.md)). Consumed by the dispatcher to decide when *easy* must normalize-down rather than pass through (Music dates, Companies financials).
- Static operator tables shared across domains at `usecases_synthetic/config/knob_05_format_unit/_tables/`:
  - `date_formats.yaml` — format-id → `strftime` pattern + parser hint + `{hard_only: bool, locale_ambiguous_deny: bool}` flags. The locale-ambiguous deny-list (`%d/%m/%y`-style patterns with days ≤ 12) is enforced here, not inlined in code.
  - `number_locales.yaml` — locale-id → `(decimal_sep, thousands_sep, grouping)` triple.
  - `fx_rates.yaml` — the real-world FX rate table as of `rate_date: 2026-03-15`, per the Unit-conversion-at-hard section of this card. Currency codes × base currency. **`base_currency: USD` pinned at the top of the file.** Immutable once checked in; re-runs never refresh rates.
  - `unit_factors.yaml` — dimensional unit factors (weight, length, file-size, durations).
- Provenance written per reformatted cell to `output/provenance/knob_05_format_unit.csv` inside the variant directory, following the cross_cutting.md flat-row schema:
  ```
  (entity_id, source, attribute, original_value, new_value,
   transform_fn ∈ {reformat_date, reformat_number, reconvert_unit, relocale},
   transform_params, knob=5, level)
  ```
  `transform_params` is a JSON-encoded string. Keys by transform_fn:
  - `reformat_date`: `{from_format, to_format, direction ∈ {up, identity, normalize_down}}`
  - `reformat_number`: `{from_locale, to_locale, precision}`
  - `reconvert_unit`: `{from_unit, to_unit, rate, rate_date, magnitude_scale}`
  - `relocale`: `{from_locale, to_locale}`
- Skipped-cell audit at `output/provenance/knob_05_skipped.csv` for the round-trip-parse fallback path (includes the attempted draw and the parser error). **Reason codes:** `roundtrip_parse_fail`, `cell_collision_with_1`, `cell_collision_with_4` (K4-fabricated cell — see [knob_04_coverage_skew.md §Joint cell-collision index integration](knob_04_coverage_skew.md#joint-cell-collision-index-integration-resolves-c2-from-the-step-5-cross-knob-review)), `cell_collision_with_7`, `denylist_locale_ambiguous`.
- Caching: the full output is a file artifact on disk (reformatted source datasets + provenance CSV + skipped CSV). No in-memory cache — seeded RNG + static tables give reproducible regeneration.
- Committee surface: the Norm / Blocking / EM / Fusion committees (per the Committee-expectations section above and the cross_cutting.md committee mechanism) see the reformatted source files exactly as written. The *spread* between normalizing and non-normalizing committee members is the primary difficulty signal for this knob.

**Domain-specific adjustments.**
- **Music** — natural showcase domain and **at hard already** for dates (Discogs: 4 coexisting date format families within one column; MusicBrainz: 3 precision levels) and durations (MusicBrainz integer ms vs Discogs `mm:ss`). Easy and medium on Music dates/durations are **normalize-down** operations: the dispatcher rewrites all source values to a single canonical format, logged with `direction=normalize_down`. **Hard is literal passthrough on Music dates/durations** (baseline already meets or exceeds the target state) — the dispatcher emits `transform_fn=reformat_date` rows with `direction=identity` for audit, but the values are unchanged. Operator extension (K/M/B suffixes on sales, symbol-vs-ISO on currency) applies on the *other* attribute classes. The "easy is not a no-op" invariant is *not* a claim about hard — it applies only to easy and medium, where Music dates/durations are visibly normalized down.
- **Companies** — **at hard already on financials** (DBpedia `total_assets_val` mixes magnitudes `8.0`, `240560000000.0`, … inside one column). Easy on financials requires active normalize-down to a single magnitude (typically USD billions), logged with `direction=normalize_down`. Dates are uniform (ISO) across sources, so date pools are additive at all levels. FX rate table is the binding artifact here — rate drift between authoring and run time is prevented by the immutable `rate_date: 2026-03-15` pin.
- **Games** — **below easy** at baseline (entirely uniform). Cleanest demonstration of "easy is not a no-op" — every level requires active heterogeneity injection. The dispatcher never takes the normalize-down branch for games; easy draws from a minimal 2-format pool, medium and hard extend upward.
- **Movies, products**: deferred alongside the Step 6 prototype. The dispatcher no-ops on domains without a `config/knob_05_format_unit/<domain>.yaml` (warns in the log). No code change required when those domains come online — only a new YAML (and for products, a unit-factor extension for package dimensions).

**Rejected alternatives.**
- **LLM-based format rewriting** (e.g., prompting an LLM to "rewrite this date in a different format"). Rejected: the task is mechanical, fully deterministic, and has strong `strftime` / `babel` / `Decimal` baselines. LLM use would sacrifice determinism, introduce contamination risk, inflate validation cost (mandatory committee + human spot-check per plan_algorithmselection.md decision framework), and deliver zero expected quality gain. **LLM not used because the deterministic alternative is sufficient.**
- **Heavyweight ML format-transfer methods** (BART / GReaT / CTGAN / TabDDPM). Rejected under the plan_algorithmselection.md framework rule against heavyweight ML methods (violates determinism, validation cost, dependency weight simultaneously). Format rewriting is orthogonal to what these models optimise.
- **Precision-downgrade as a separate knob.** Considered — XBenchMatch treats precision change as a first-class granularity operation. Rejected here because precision downgrade is semantically still a format change (a date truncated to year is still a parseable date in a coarser format family) and splitting it off would fragment provenance. Kept as a hard-only operator inside `reformat_date`.
- **Split/merge column operations** (XBenchMatch granularity class). Rejected from Knob 5 because they change the schema shape, which is Knob 8's territory. Cross-referenced only.
- **Valentine fabricated-benchmark "value corruption" branch** used as-is. Rejected as a whole because Valentine bundles format changes with typos and missing values, which would cross the Knob 5 / Knob 6 / Knob 3 boundary we have carefully drawn. We cite the format-change sub-operation only.

**Implementation handoff.** Everything Step 6 needs to implement this knob without re-reading surrounding cards:

- **Target script:** `usecases_synthetic/scripts/apply_knob_05_format_unit.py` (new, convention matches `apply_knob_06_noise.py` and `apply_knob_08_naming.py`). Standalone runnable from repo root.
- **Function shape (illustrative):**
  ```python
  def apply_knob_05(
      domain: str,
      level: Literal["easy", "medium", "hard"],
      sources: dict[str, pd.DataFrame],
      fusion_gold: pd.DataFrame,                   # read-only; round-trip verification
      attribute_classes: dict[str, dict[str, Literal["date", "number", "money", "duration", "dimensional"]]],
      baseline_format_profile: dict[str, dict[str, list[str]]],
                                                   # measured per (source, attribute) by baseline pass
      config_path: Path,                           # usecases_synthetic/config/knob_05_format_unit/<domain>.yaml
      tables_dir: Path,                            # usecases_synthetic/config/knob_05_format_unit/_tables/
      output_dir: Path,
      seed: int,
  ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
      """Returns (reformatted_sources, provenance_df, skipped_df)."""
  ```
- **Inputs the script reads:**
  - Source DataFrames (via `PyDI.io.load_*`, preserving `df.attrs["dataset_name"]`).
  - The fusion gold artifact from `usecases/<domain>/input/fusion/` — read-only, used for round-trip verification; **never mutated** (byte-identical before and after).
  - Per-domain config at `usecases_synthetic/config/knob_05_format_unit/<domain>.yaml`.
  - Shared operator tables under `usecases_synthetic/config/knob_05_format_unit/_tables/{date_formats.yaml, number_locales.yaml, fx_rates.yaml, unit_factors.yaml}`.
  - Baseline format profile (written by the baseline measurement pass; path recorded in the variant's `config/difficulty.yaml`).
- **Outputs the script writes** (under the variant directory):
  - Reformatted source files in `input/data/` (same format as input — XML/JSON/CSV).
  - Provenance log at `output/provenance/knob_05_format_unit.csv`.
  - Skipped-cell audit at `output/provenance/knob_05_skipped.csv` (round-trip failures).
- **Pipeline integration:** Knob 5 sits inside the `Knobs 1/5/6/7` joint phase of the canonical S1 order from [README.md](README.md#canonical-knob-application-order). It runs **before** Knob 6 so that noise operators compose over reformatted cells. Downstream Knob 3 (cell drops) runs after the joint phase and may drop reformatted cells; accepted and recorded as a reformat→drop chain via provenance linkage on `(entity_id, source, attribute)`.
- **Dependencies:** stdlib (`datetime`, `decimal`) + `pandas` + `numpy` + `pyyaml` + `python-dateutil`. Also requires `babel` for locale-aware number formatting — **not currently in [pyproject.toml](../pyproject.toml)**; Step 6 must add it as a runtime dependency of the synthetic-generation tooling (not of PyDI itself). No PyDI extension points.
- **Authoring task before first run:** populate `usecases_synthetic/config/knob_05_format_unit/{companies,games,music}.yaml` with `attribute_classes`, `format_pools_per_level`, `locale_pool_per_level`, and `unit_pool_per_level` using the Easy/Medium/Hard table above as the source of truth; pin the FX rate table in `_tables/fx_rates.yaml` from a single published source on `2026-03-15`. The baseline format profile is **measured**, not authored.
- **Smoke test:** for each domain with a config, run the script at all three levels and assert (a) every emitted cell round-trips to the same canonical value as the original under canonical-form comparison (date: same `datetime.date` or `datetime` truncated to the emitted precision; number: same `Decimal` within tolerance; money: same `Decimal` after applying the logged rate; duration: same `timedelta`), (b) the fusion gold file on disk is byte-identical before and after the run, (c) at hard, within-source format diversity > 1 for at least one format-bearing column per source, (d) at easy/medium, within-source format diversity == 1 for every format-bearing column per source, (e) the provenance row count equals the number of cell-value mutations, (f) no emitted date value matches a pattern on the locale-ambiguous deny-list.
