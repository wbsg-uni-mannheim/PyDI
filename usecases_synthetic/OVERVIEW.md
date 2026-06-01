# Synthetic Use Case Pipeline — Overview

A collaborator-facing tour of how `usecases_synthetic/` turns an original
PyDI use case (companies, games, music, products) into three difficulty-graded
synthetic variants — `easy`, `medium`, `hard` — and validates each level
against a frozen committee of matchers / normalizers / fusion strategies.

For implementation status, history of decisions, and outstanding plans see
[plan_revision.md](../plans/plan_revision.md), [plan_s1_final.md](../plans/archive/plan_s1_final.md),
and [PIPELINE.md](PIPELINE.md) (runbook). This file is meant to give a new
reader a self-contained mental model of the pipeline.

---

## End-to-end flow

```
 ┌──────────────────┐    ┌─────────────────────┐    ┌──────────────────────────┐
 │ Original use     │    │ Phase 0             │    │ Phase 1                  │
 │ case data        │───▶│ Pool construction   │───▶│ Baseline measurement     │
 │ (4 domains)      │    │ (assemble a set of  │    │ (5 committees on         │
 │                  │    │  known-match pairs) │    │  unperturbed data)       │
 └──────────────────┘    └─────────────────────┘    └────────────┬─────────────┘
                                                                 │
                                       ┌─────────────────────────┴────────────────────────┐
                                       ▼                         ▼                        ▼
                          ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
                          │ Phase 2 / easy       │  │ Phase 2 / medium     │  │ Phase 2 / hard       │
                          │ generate_variant.py  │  │ generate_variant.py  │  │ generate_variant.py  │
                          │ (knob stack below)   │  │ (knob stack below)   │  │ (knob stack below)   │
                          └──────────┬───────────┘  └──────────┬───────────┘  └──────────┬───────────┘
                                     ▼                         ▼                         ▼
                          ┌──────────────────────────────────────────────────────────────────────────┐
                          │ Phase 3 — Validation per level                                           │
                          │   validate_variant.py  →  SM / Norm / EM-block / EM-match / Fusion       │
                          │   analyze_monotonicity.py  →  cross-level: macro_f1, best-member ceiling │
                          │                               + per-knob realised intensity audits       │
                          └────────────────────────────────────┬─────────────────────────────────────┘
                                                               ▼
                                                  validation/<domain>/final_report.md
```

---

## Phase 0 — Pool construction

### Why we need a pool

Every PyDI use case ships with a hand-authored **entity-matching gold
standard** — a curated CSV of true cross-source match pairs used to measure
matcher performance. Those golds are typically *incomplete*: a real-world
EM gold covers only a small fraction of the true match set (often single-
to low-double-digit recall against the actual match population), because
exhaustive cross-source labelling at scale is infeasible.

Incompleteness is fine when you're scoring matchers against the gold — you
just need a representative sample. But it becomes a problem when you start
*perturbing* the data to create harder variants (Phase 2 below). If the
perturbation step that thins out crowded entity neighbourhoods only knew
about gold pairs, it could happily delete a pair that's genuinely a match
in the real world but happens to be missing from the gold sample. The
resulting "harder" variant would then have *fewer matchable entities*, not
*harder-to-match entities* — the difficulty signal you'd measure later
would partially be noise from accidentally destroyed matches rather than
the dial setting you set.

The pool fills that gap. It augments the gold's protection coverage with
**cross-validated evidence from matcher pipelines that have previously run
against these same domains**, treating "two independent matchers both
agreed this is a pair" as strong evidence that the pair is real, even when
the hand-authored gold doesn't mention it.

### How the pool is built

[scripts/build_pool.py](scripts/build_pool.py) merges the outputs of two
existing matcher pipelines that we've previously run on each domain:

- a **PLM matcher pipeline** — pre-trained-language-model-based; the
  strong, learned matcher. Treated as **trusted base**.
- a **rule-based matcher pipeline** — hand-weighted string-comparator
  combinations; materially weaker than the PLM matcher. Treated as
  **corroboration only**.

Combination rules:

- A pair the PLM pipeline produced lands in the pool with
  `pool_agreement = 1`.
- A pair produced by both pipelines is upgraded to `pool_agreement = 2`.
- A pair produced **only** by the rule-based pipeline is **dropped** —
  the rule-based matcher's solo claims aren't trusted given its known
  weakness.

The rule-based pipeline ships a pre-computed clustering, but we ignore
that and rebuild components from raw edges on both sides: extract the
pairwise edges from each pipeline, union them, then run
`networkx.connected_components` on the merged edge set. That gives uniform
cluster semantics across both sources and lets a single cluster-size
filter operate on the regenerated components.

An **egregious-cluster filter** then drops the worst transitive-chain
artefacts (the most common pool pathology — A↔B, B↔C, C↔D edges that are
each individually plausible but together form a 100-entity blob that
clearly isn't one entity). Any cluster strictly larger than a per-domain
cap is removed; the cap is data-driven and equals `max(P99, floor)`,
where:

- **P99** is the 99th percentile of observed cluster sizes in that
  domain's pool: sort all clusters by size and P99 is the size where 99%
  of clusters are at or below it, only the top 1% are bigger. This
  adapts to each domain's natural cluster distribution — companies has
  tightly-clustered pairs (P99=7), games has a longer tail (P99=12).
- **Floor** is the structural lower bound `3 × n_sources` (3 sources for
  companies/games/music, 4 for products). It exists so that domains
  with tightly-clustered natural distributions don't get aggressive cuts
  applied to plausibly-real clusters. The reasoning: with N sources, a
  legitimate cross-source duplicate cluster could plausibly contain up
  to ~3 rows per source (one matched row plus slack for in-source
  duplicates), so any cap below `3N` would risk dropping real entities.

The cap in practice: companies P99=7 < floor=9 → cap=9 (floor wins);
games P99=12 > floor=9 → cap=12 (P99 wins); music P99=9 = floor=9 →
cap=9 (tied). On companies this filter dropped a 117-entity "Chinese
companies" cluster that had grown via transitive-link noise.

Products uses a separate pool builder,
[build_pool_products.py](scripts/build_pool_products.py), that derives
clusters directly from each record's `cluster_id` field rather than from
the PLM / rule-based pipelines — products ships its own cluster
identifiers per row, so the agreement-from-two-matchers construction
isn't needed.

### What the pool is used for

The pool is fed forward into Phase 2 as a **protected set** that the
perturbation steps consult before touching any entity. Concretely, the
step that controls how many similar-but-distinct entities cluster
together (K2 — explained below in the knob stack) refuses to remove any
entity that appears in the pool, even when its niche dial points at a
sparser target. The corresponding step for source coverage (K4) and the
cell-drop step (K3) apply analogous protection rules. The net effect is
that **the difficulty signal we measure in Phase 3 reflects the dials we
set, not noise from accidentally destroying known matches**.

### Outputs

Per domain under [pools/](pools/):
- `pooled_positives.csv` — one row per pair, columns
  `id1, id2, source_1, source_2, pool_agreement`.
- `pool_stats.json` — source counts, overlap breakdown, egregious-cluster
  filter telemetry.

Current pool sizes (v2, 2026-04-18):

| Domain    | Pool size | Both-source agreement | Egregious cap applied | Largest dropped component |
|-----------|----------:|----------------------:|----------------------:|--------------------------:|
| companies | 2225      | 490 (22%)             | 9 (P99=7, floor=9)    | 13                        |
| games     | 13795     | 4659 (34%)            | 12 (P99=12, floor=9)  | 31                        |
| music     | 7355      | 3569 (49%)            | 9 (P99=9, floor=9)    | 55                        |

---

## Phase 1 — Baseline measurement

[scripts/measure_baseline.py](scripts/measure_baseline.py) runs the five
committees against each domain's *original* data and writes
`baselines/<domain>/baseline_metrics.json` + `baseline_report.md`. Committee
YAML SHAs are recorded in the metrics file; validate_variant refuses to run
later if those SHAs drift, so a frozen baseline is paired with a frozen
committee for the entire variant lifetime.

The committees (rosters from
[config/committees/](config/committees/), all members
`enabled_by_default: true` unless noted):

- **SM** — schema matching, 7 members spanning four signal families:
  `duplicate_majority` (duplicate), `label_jw` (label string similarity),
  `instance_tf_cosine` (instance tf-cosine), `embedding_sbert` (SBERT
  embedding), `llm_openai`, `magneto_slm_llm`, `coma_hybrid` (COMA-style
  hybrid).
- **Norm** — normalization, per-domain roster of 6 members (see
  `normalization_committee_<domain>.yaml`): `text_clean`, `date_iso`,
  `number_locale`, `country_iso`, `taxonomy_lookup`, `llm_canonicalize`.
- **EM-block** — entity-matching blocking, 6 members:
  `token_blocker`, `standard_blocker`, `embedding_blocker`,
  `sorted_neighbourhood_blocker`, `bm25_blocker`, `sc_block` (the
  Sudowoodo-style contrastive encoder, requires a per-domain trained
  checkpoint).
- **EM-match** — entity-matching matching, 4 members + a pool diagnostic:
  `ditto_plm` (DITTO PLM), `magellan` (Magellan-style comparator stack),
  `llm_matcher` (zero-shot LLM), `comem` (CompactER / contrastive
  embedding matcher).
- **Fusion** — per-attribute strategy sets rather than a flat roster.
  Each attribute is classified by type and routed to an appropriate
  family of resolvers:
  - String / primary-label attributes: `voting`, `longest_string`,
    `most_complete`, `prefer_higher_trust`, plus three truth-discovery
    methods (`accusim`, `fusionquery`, `casefusion`) and an LLM
    adjudicator (`llm_judge`).
  - Numeric attributes (`assets`, `revenue`, `vram_gb`, `storage_gb`):
    `median`, `maximum`, robust aggregators (`trimmed_mean`,
    `huber_m_estimator`), `prefer_higher_trust`, plus `fusionquery`.
  - Date / categorical attributes use the appropriate subset of the
    above (e.g. `year_only_match` for `founded`).

See [config/committees/](config/committees/) and
[plans/validation/module_01_committee_spec.md](../plans/validation/module_01_committee_spec.md)
for member parameters and tuning history.

---

## Phase 2 — Variant generation (S1 augmented use cases)

[scripts/generate_variant.py](scripts/generate_variant.py) is the master
orchestrator. For a given `(domain, level)` it imports each `apply_knob_*`
pure entry point and applies them in canonical S1 order. Two knobs share an
LLM cache (K1, K2) and the dispatcher operates in `strict_cache` mode at
`hard` by default for reproducibility from committed caches — automatically
relaxed when the domain declares a `knob_config_alias` (e.g. `companies-small`
reuses `companies` configs, so the shared cache may need to populate on first
use).

### Knob stack

```
sources (post-Phase-0 protection set in scope)
   │
   ▼  ┌─────────────────────────────────────────────────────────────────────────────┐
      │ K2 niche density            (first — sets the entity population)            │
      └─────────────────────────────────────────────────────────────────────────────┘
   │
   ▼  ┌─────────────────────────────────────────────────────────────────────────────┐
      │ K4 coverage skew            (which sources have which entities)             │
      └─────────────────────────────────────────────────────────────────────────────┘
   │
   ▼  ┌─────────────────────────────────────────────────────────────────────────────┐
      │ K1 + K5 + K6  JOINT cell pass via apply_values_joint.py                     │
      │   K1 surface  →  K5 format  →  K6 noise                                     │
      │   Shared CollisionIndex: K5 skips K1+K4 cells; K6 skips K1+K5 but does     │
      │   touch K4-fabricated cells.                                                │
      └─────────────────────────────────────────────────────────────────────────────┘
   │
   ▼  ┌─────────────────────────────────────────────────────────────────────────────┐
      │ K3 per-source attribute drop  (column-level NaN masking per row)            │
      └─────────────────────────────────────────────────────────────────────────────┘
   │
   ▼  ┌─────────────────────────────────────────────────────────────────────────────┐
      │ K10 source reliability        (which source carries the gold-aligned cell)  │
      └─────────────────────────────────────────────────────────────────────────────┘
   │
   ▼  ┌─────────────────────────────────────────────────────────────────────────────┐
      │ K8 schema naming              (header renames only)                         │
      └─────────────────────────────────────────────────────────────────────────────┘
   │
   ▼
usecases/<domain>-augmented/<level>/
   ├── input/{data, schemamatching, entitymatching, fusion}
   ├── baselines/knob_NN_realised.csv     (per-knob intensity audit)
   ├── provenance_all.csv                  (every modified cell, every level)
   └── config/difficulty.yaml              (resolved per-knob params)
```

### What each knob actually does

#### K2 — Entity niche density   [knob_02_niche_density.md](../knobs/knob_02_niche_density.md) · [apply_knob_02_niche.py](scripts/apply_knob_02_niche.py)

Controls how many similar-but-distinct entities cluster in the same semantic
neighborhood (franchises, discographies, sequels). The dial we actually set
is `corner_case_ratio` — the fraction of EM pairs that are near-twin
pairs — which acts as a measurable proxy for niche density: dense
neighborhoods produce more near-twin EM pairs, sparse neighborhoods
produce fewer. Two shared sub-systems on one multi-metric substrate
(lexical + embedding + attribute-overlap + label-collision): a
**consensus-biased RRF scorer** drives removal at easy/medium, a
**recall-biased per-metric union** mines corner-case pairs for EM test
regeneration and the hard-negative budget.

##### How "density" is computed

Each entity gets a single density score via four-metric Reciprocal Rank
Fusion (RRF) plus a label-collision boost. The metrics live in
[lib/niche_metrics.py](lib/niche_metrics.py); the fusion lives in
[lib/niche_scorer.py](lib/niche_scorer.py).

1. **Four ranking metrics**, each producing a top-K list of similar
   entities per entity (`metric_top_k = 30` per the music config):
   - `lexical_extended_jaccard` — generalised Jaccard over tokenised
     primary labels, with Levenshtein-ratio inner-token matching so
     K6-injected typos don't erase near-twins.
   - `tfidf` — sklearn `TfidfVectorizer` document-term matrix → cosine
     similarity.
   - `embedding` — sentence-transformers `all-MiniLM-L6-v2` embeddings
     (cached on disk per domain) → cosine similarity.
   - `attribute_overlap` — weighted Jaccard over the categorical-
     attribute bag, with per-domain column weights.
2. **RRF fusion.** For every neighbour `n` that any metric ranked in
   entity `e`'s top-K, the fused score is
   `Σ_m  1 / (k₀ + rank_m(n | e))` with `k₀ = 60` (standard RRF
   damping). An *agreement count* tracks how many of the four metrics
   ranked `n` in `e`'s top-K.
3. **Consensus filter** (the "consensus-biased" bit). A neighbour with
   agreement count below `c_min = 2` contributes zero to density —
   one metric flagging `n` isn't enough; at least two have to agree.
4. **Label-collision boost.** Entities whose primary label normalises
   byte-identical (lowercase + accent-fold + strip punctuation + drop
   bracketed suffixes) to ≥1 other entity get a fixed `+5.0` boost.
   This deterministically pulls franchises with exact-match titles
   (`John Williams`, `Crash`) into the same density bucket even when
   the ranking metrics individually disagree.

The result is a per-entity density score: **high = tight cluster of
similar entities; low = uniquely positioned in the population**.

##### How the algorithm decides drop vs interpolate

The dispatcher first measures the *current* corner-case ratio in the
baseline data (using the same four metrics through a separate recall-
biased miner — `t_match` / `t_nonmatch` thresholds per metric, union
across metrics), then compares to the level's target with a ±0.02
tolerance band:

- **`baseline > target + 0.02`** → drop entities ranked by descending
  density (removing the crowded ones thins their cluster).
- **`baseline < target - 0.02`** → **paired LLM interpolation**: for
  each step, generate a near-twin entity seeded from a dense cluster,
  then remove one low-density entity to keep the per-source row count
  close to the original (size-invariant by construction). Capped at
  `max_interp_fraction × n_entities` (`0.60` for music).
- **`|baseline − target| ≤ 0.02`** → no-op.

*Current caveat:* at easy when baseline already exceeds the easy target
— common, since music's natural corner-case ratio is ~0.24, above
easy's 0.20 target — the density-drop operator drifts the ratio
*further* from target rather than toward it (removing dense entities
shrinks the denominator faster than the numerator). K2 therefore
currently no-ops in that regime and reports the baseline as the
realised ratio. See [plan_revision.md §R-1](../plans/plan_revision.md)
G1 + C1 for the planned fix (a genuine "drop corner-touching entities"
operator).

##### Per-level targets

- **Easy** (`corner_case_ratio ≈ 0.20`): niche-aware removal — drop
  non-protected entities from crowded clusters until they thin out.
- **Medium** (`≈ 0.50`): add or remove toward target with no LLM calls.
- **Hard** (`≈ 0.60–0.80`): cached LLM interpolation (default
  `gpt-5.4-mini`) generates near-twin entities seeded from dense
  clusters, with `placement_split` 60/40 between multi-source placement
  (becomes hard positives) and single-source placement (pure hard
  negatives).

**Side-effect:** regenerates the EM test split per variant so the corner-case
ratio is tracked end-to-end. **Audit:** `knob_02_realised.csv` (baseline /
target / final ratio + per-guardrail rejection counters).

#### K4 — Per-entity source coverage skew   [knob_04_coverage_skew.md](../knobs/knob_04_coverage_skew.md) · [apply_knob_04_coverage.py](scripts/apply_knob_04_coverage.py)

Shifts the per-entity source coverage histogram (group size distribution: how
many sources cover each entity). Operates on whole rows, not cells.

- **Easy** (uniform): LLM-fabricates a source-specific representation (style /
  schema / formatting of the missing source) for entities not present in that
  source. Fabricated rows must be consistent with the fusion gold.
- **Medium**: identity — baseline preserved.
- **Hard** (long-tail): removes entity rows to create singletons, gated by a
  fusion-gold floor and a conflict-preserving removal rule.
  `within_source_duplicate_rate` also adds same-entity duplicate rows via K1's
  paraphrase pipeline at hard.

| Music dial — `target_coverage_histogram` (share of entities covered by N sources) | easy | medium | hard |
|---|---:|---:|---:|
| Covered by 1 source (singleton)   |  0% | (identity) | 55% |
| Covered by 2 sources              | 10% | (identity) | 30% |
| Covered by 3 sources (all)        | 90% | (identity) | 15% |
| `within_source_duplicate_rate`    |  0% |    0%      |  2% |

Easy aims for near-uniform coverage so voting strategies always see ≥2
values per attribute; hard creates a long tail so the majority of
entities are visible in only one source and fusion must fall back on
trust signals.

**Audit:** `knob_04_realized_vs_target.csv`.

#### K1 — Surface augmentation   [knob_01_surface_augmentation.md](../knobs/knob_01_surface_augmentation.md) · [apply_knob_01_surface.py](scripts/apply_knob_01_surface.py)

Plausible free-text rewriting of cell values (paraphrase, abbreviate,
reorder). Both forms are correct — the boundary against K6 is "legitimate
variant" vs "error." Tier-C hybrid generator:

- **Easy**: deterministic `normalize-to-canonical` via
  `baseline_above_target_rules` (used when natural baseline is already above
  the target paraphrase rate — pulls back toward a per-entity canonical form).
- **Medium**: deterministic table-driven abbreviation + EDA `random_swap` /
  `random_delete`.
- **Hard**: medium operators ∪ cached LLM paraphrase (`gpt-5.4-mini`) with
  contamination guardrails + committee validation.

Per-attribute-class rates control what fraction of cells in each class
gets touched. A rate of 0.08 on `primary` means ~8% of primary-attribute
cells (e.g. song titles, album names) are paraphrased.

| Music dial — fraction of cells touched | easy | medium | hard |
|---|---:|---:|---:|
| `paraphrase_rate_primary` (e.g. `name`)              |  0% |  2% |  8% |
| `paraphrase_rate_key` (e.g. `artist`, `release-country`) |  0% |  4% | 12% |
| `paraphrase_rate_secondary` (e.g. `label`, `duration`)   |  0% |  8% | 20% |
| `paraphrase_rate_categorical` (e.g. `genre`)         |  0% |  4% | 12% |

The `paraphrase_long` rate from the K1 spec is reserved for future
domains with long-text attributes (movie plots, product marketing copy).

#### K5 — Format / unit diversity   [knob_05_format_unit.md](../knobs/knob_05_format_unit.md) · [apply_knob_05_format.py](scripts/apply_knob_05_format.py)

Structured-format and unit rewriting — every value stays machine-parseable
and semantically exact, just in a different form. Classifies each attribute
by format family (date / number / currency / locale), draws a format
assignment, applies the operator, and verifies a round-trip parse.

- **Easy**: per-(source, attribute) draw from a pool of ≤2 formats; one
  format per source, consistent within a source.
- **Medium**: per-(source, attribute) draw from 2–3 formats; still consistent
  within a source.
- **Hard**: per-(row, attribute) draw from 3+ formats — a single source mixes
  formats row-to-row. Multiple coexisting units (USD/EUR; thousands/millions;
  minutes/hh:mm:ss). Excludes deliberately ambiguous values (those belong to
  K7).

| Music dial — `format_pools_per_level` | easy | medium | hard |
|---|---|---|---|
| `release-date` formats | ISO, `YYYY` year-only | + EU `DD.MM.YYYY` | + per-row mix of 4 formats |
| `duration` formats     | `seconds_int`, `mm:ss` | + `hh:mm:ss` | + `human_xm_ys` (e.g. `17m 35s`), per-row |
| Pool size per attr     | 2 | 3 | 4 |

#### K6 — Value noise   [knob_06_value_noise.md](../knobs/knob_06_value_noise.md) · [apply_knob_06_noise.py](scripts/apply_knob_06_noise.py)

Cell-level corruption using the FEBRL / Christen-Vatsalan operator suite —
char-level edits, OCR confusions (`O↔0`, `l↔1`, `rn↔m`, `cl↔d`), truncations,
whitespace/punctuation corruption, case corruption. These are *errors*, not
variants. Per-attribute-class rates (`noise_rate_primary / _key /
_secondary`). Source rows only — the fusion-gold artifact is never touched.
Easy / medium / hard scale the rates per attribute class; hard also
introduces noise into the primary label, which is zero at easy/medium.

| Music dial — fraction of cells corrupted | easy | medium | hard |
|---|---:|---:|---:|
| `noise_rate_primary`   |  0% | 0% |  1% |
| `noise_rate_key`       |  0% | 2% |  6% |
| `noise_rate_secondary` |  1% | 4% | 12% |

The operator mix also widens with level: easy uses only whitespace +
case corruption, medium adds typos / OCR confusions / taxonomy walks,
hard adds truncation + character transposition.

#### K3 — Per-source attribute drop   [knob_03_attribute_drop.md](../knobs/knob_03_attribute_drop.md) · [apply_knob_03_drop.py](scripts/apply_knob_03_drop.py)

Cell masking (NaN injection) at the (row, column) level. Per-attribute-class
rates (`drop_rate_primary / _key / _secondary`).

- **Cross-level nesting** `D_easy ⊆ D_medium ⊆ D_hard` is enforced
  structurally via shared per-cell uniform draws: all three masks are
  computed in one pass and easy/medium are shrunk against hard so a cell
  dropped at easy is also dropped at medium/hard.
- **Per-source shape:** transforms a *measured-baseline* missingness vector
  — easy `compress` (propagate values cross-source toward the
  min-missingness source), medium `identity`, hard `stretch`.
- **Constraints** applied at each level: fusion floor (cells used by fusion
  gold are protected), conflict preservation, single-source survivor cap.

| Music dial — fraction of cells dropped | easy | medium | hard |
|---|---:|---:|---:|
| `drop_rate_primary`   | 0% |  0% |  3% |
| `drop_rate_key`       | 2% | 10% | 25% |
| `drop_rate_secondary` | 5% | 15% | 35% |

**Audit:** `knob_03_baseline_missingness.csv`.

#### K10 — Source reliability differentiation   [knob_10_source_reliability.md](../knobs/knob_10_source_reliability.md) · [apply_knob_10_reliability.py](scripts/apply_knob_10_reliability.py)

Pure permutation reshuffle. After K1/K5/K6 have produced multiple variants
of the same (entity, attribute) cell, K10 picks **which source** carries the
gold-aligned variant. No new perturbation; no gold mutation (a SHA sentinel
verifies the fusion gold file is byte-identical before / after).

- **`per_attribute_concentration`** — at easy, the per-attribute winner gets
  the gold variant on a high share of cells (trust strategies trivially
  work); at hard the share drops (diffuse trust per attribute).
- **`error_correlation`** — at hard, perturbations burst along the (source,
  entity) axis: if source X mis-handled entity Y on attribute A, elevated
  probability of mis-handling on B too. Models "this source confused Y with a
  near-twin and got everything wrong."
- The per-attribute winner is **named** in the YAML and re-derived from the
  freshly-measured baseline `B[s, a]`; loader fails loud if the named winner
  no longer matches the measured winner (no silent gold-carrier drift across
  runs).

| Music dial — share of `name` cells carrying gold variant (winner: musicbrainz) | easy | medium | hard |
|---|---:|---:|---:|
| musicbrainz (winner) | 85% | 65% | 40% |
| discogs              | 10% | 20% | 30% |
| lastfm               |  5% | 15% | 30% |

Per-attribute targets are authored per source × level (same shape for
`artist`, `release-date`, etc., each with its own measured winner).
Concentration shrinks toward equal-share at hard so trust-weighted
fusion strategies can no longer rely on a single winner.

`error_correlation` (burstiness along the source × entity axis):
`easy=0.0 / medium=0.20 / hard=0.50`.

**Audit:** `knob_10_realised.csv` (rate-based: `swap_rate = swap_cells /
reshufflable_count`, invariant to K3 pool shrinkage).

#### K8 — Schema naming divergence   [knob_08_schema_naming.md](../knobs/knob_08_schema_naming.md) · [apply_knob_08_naming.py](scripts/apply_knob_08_naming.py)

Column headers only — no cell values touched. Per-source rename on a 4-level
ladder via a per-domain `rename_table`:

| Level rung   | Example (`release_date`)        |
|--------------|---------------------------------|
| `descriptive`| `release_date`                  |
| `abbreviated`| `rel_dt`                        |
| `cryptic`    | `rdate`, `c_code`               |
| `anonymized` | `Attribute_3`, `col_07`         |

Bidirectional in S1: easy may rename *up* (anonymized → descriptive) when
baseline starts low; hard renames *down*. Regenerates
`input/schemamatching/sm_mapping.csv` so the SM stage sees the new headers.
Affects almost exclusively the SM stage.

| Music dial — `level_assignments` (rung per source) | easy | medium | hard |
|---|---|---|---|
| musicbrainz | descriptive | abbreviated | cryptic     |
| discogs     | descriptive | descriptive | abbreviated |
| lastfm      | descriptive | cryptic     | anonymized  |

All three sources rename in lockstep at easy (no SM challenge); at hard
the rungs disperse so a single SM matcher faces three different naming
distances from the target schema at once.

**Audit:** `knob_08_naming_intensity` (rung-weighted: descriptive=0 /
abbreviated=1 / cryptic=2 / anonymized=3).

### Knobs not active in v1

- **K7 — Value ambiguity / collision rate** ([knob_07_value_ambiguity.md](../knobs/knob_07_value_ambiguity.md)).
  Design locked, but **not built in v1**. K7 was specced with three
  sub-parameters, and they fail the v1 cost / value test for different
  reasons:
  - `referential_ambiguity_rate` — the active sub-parameter (e.g.
    `"Republic of Korea"` → `"Korea"`, `"John A. Smith Jr."` → `"John
    Smith"`). **Evaluator-compatible**: by design the K7 doc bounds its
    reach to "value pairs the lenient fusion evaluator already tolerates,"
    which is the same `lexical_extended_jaccard` close-enough mechanism the
    rest of the fusion stage uses. The blocker here is **substrate**:
    companies has the strongest case (`country`, partial `founders`
    shortenings) but `founders` only covers ~10% of rows → effective cell
    budget ≈ 190; games' developer/publisher names are usually full studio
    names; music's natural ambiguity is in cross-entity homonyms (e.g.
    `John Williams`), which were moved to K2. Building K7's three new
    mechanisms — a per-attribute ambiguity map curated from gold values, a
    pre-injection lenient-evaluator probe, and per-cell rollback on
    committee collapse (no other knob needs that) — against ≤200 cells in
    the strongest domain didn't pencil out.
  - `multi_sense_conflict_rate` + `polysemy_rate_categorical` — **parked**.
    These two would need fusion gold extended to *accepted sets* (e.g.
    `genre = {Rock, Alternative}` where any member counts), which the
    close-enough evaluator can't model (`Rock` and `Alternative` aren't
    lexically close — the song genuinely is both). Re-opening these is
    blocked on the accepted-sets discussion.
  - The cross-entity label-collision slice of K7's original scope was
    rehomed into K2 as a fourth niche-metric signal — that part is live,
    just not under the K7 label.

  Re-open trigger is single: when the accepted-sets discussion lands
  favorably, K7 grows from one thin sub-parameter to three and the same
  ambiguity-map / probing / rollback machinery amortizes across substrate
  orders of magnitude larger. The dimensions K7 was meant to cover (Value
  Ambiguity in Norm; Conflict Rate and Conflict Subtlety in Fusion) are
  accepted as **under-stressed in v1**.
- **K9 — Schema completeness / distractors** ([knob_09_schema_completeness.md](../knobs/knob_09_schema_completeness.md)).
  Locked but **S2 only** — S1 inherits the original use case's column set
  as-is, so K9 has nothing to inject against. Activates with Scenario 2
  (fully synthetic seed) which is queued behind the S1 prototype.

---

## Training, validation, and test sets in the variants

Each domain's original use case ships hand-authored ground-truth files for
the three downstream tasks: a schema-matching gold mapping, train / val /
test splits for entity matching per source pair, and fusion validation /
test XMLs. The variant pipeline either copies these through verbatim,
augments them with regenerated companions, or rewrites them to track
header changes.

| Stage | In `usecases/<domain>/input/` (originals) | In `usecases/<domain>-augmented/<level>/input/` (variants) | What changed and why |
|---|---|---|---|
| **Schema matching** | `sm_mapping_gold.csv`, `target_schema.json`, per-domain auxiliaries | `sm_mapping.csv`, `target_schema.json` | K8 (naming) renames source column headers, so the gold mapping is rewritten to reference the new headers (otherwise the SM matcher would predict against columns that don't appear in the gold). `target_schema.json` is copied unchanged. The filename loses the `_gold` suffix per project naming convention. |
| **Entity matching** | Per source pair: `<a>_2_<b>.csv`, `_all.csv`, `_train.csv`, `_val.csv`, `_test.csv`, `_train_small.csv` | All originals copied verbatim **plus** `<a>_2_<b>_{train,val,test}_regenerated.csv` | K2 removes entities at easy / medium and interpolates near-twin entities at hard, so the original splits would (a) reference IDs that no longer exist and (b) miss the new K2-injected corner cases. Regenerated splits are the closed-set version — every pair references only entities still present in the variant. The originals are kept for legacy / open-set comparability. |
| **Fusion** | `validation_set.xml`, `test_set.xml` (plus legacy `*_final.xml` variants) | `validation_set.xml`, `test_set.xml` — **byte-identical** to the originals | Fusion gold is **never mutated** by any knob. K10 verifies this with a SHA sentinel before and after the reshuffle; K3 and K4 carry an explicit fusion floor that protects cells / rows referenced by the fusion gold from being dropped. Variants therefore measure fusion against the same gold as the baseline, isolating the difficulty signal to source data quality. |
| **Normalization** | Per-domain rule + lookup files referenced by `normalization_committee_<domain>.yaml` (e.g. `Music_Genres_Taxonomy.csv`) | No new files — the committee reads rules and lookups from the original locations. | Normalization has no train / val / test split: each member is rule-driven or parameterless, and the implicit ground truth is the canonical form for each value. The committee just runs against the perturbed source data. |

### EM split regeneration mechanics

[lib/corner_case_miner.py:regenerate_em_splits](lib/corner_case_miner.py#L624)
runs per `(source_pair, split)`:

1. **Carry over surviving originals.** Every original `(id1, id2, label)`
   whose IDs both still exist in the post-K2 / post-K4 source frames is
   kept verbatim. Pairs whose IDs were K2-removed are dropped. K2 is the
   only knob that can invalidate an original pair — no other knob
   renames, merges, or splits entity clusters, so labels never flip for
   surviving IDs.
2. **Backfill to the original `(size, positive_ratio)`.** Each split
   inherits the size and positive ratio of the corresponding original
   split. A corner-case budget — `round(size × target_corner_case_ratio)`
   from K2's config — is split half-and-half between corner positives and
   corner negatives:
   - **Corner positives** = K2-interpolated cross-source near-twin pairs.
   - **Easy positives** = `cluster_positives ∪ pool_positives` minus
     interpolated (the Phase-0 pool finally acting as a source of truth
     here, not just as a protection set).
   - **Corner negatives** = hard-negative-gated cross-cluster pairs
     (close enough to confuse a blocker but in different gold clusters).
   - **Easy negatives** = the remaining cross-cluster pairs.
3. **Enforce disjointness across train / val / test.** A consumed-pairs
   tracker prevents the same record pair from appearing in two splits.
   If a backfill pool runs dry — common at hard, where K2 has stripped a
   lot of entities — the split undersizes rather than overlaps; a warning
   is logged if the realised positive ratio drifts more than 2pp from
   target.
4. **Re-run post-K4.** K4 may demote additional rows at hard, so the
   regen runs a second time with a refreshed `ids_present` filter
   against the post-K4 sources. Same pools, fresh survivors filter —
   this closes a "K2 emitted regen IDs, then K4 deleted some" orphan-pair
   hole.

### Why two sets of EM files

Both files are kept so the same variant can be scored two ways:

- **Closed-set (regenerated)** — every pair references entities still
  present in the variant. This is the **primary** difficulty surface:
  matchers are measured on a test consistent with the post-variant
  population, and the corner-case ratio tracks K2's dial.
- **Open-set (original)** — every pair references the originals' entity
  IDs, some of which no longer exist. Comparable across baseline and
  variants for legacy purposes, but unfairly penalises matchers for
  missing entities that simply don't exist anymore. Not used as the
  primary metric.

---

## Phase 3 — Validation per level

```
usecases/<domain>-augmented/<level>/input/
                    │
                    ▼
   ┌────────┬────────────┬─────────────┬──────────────┬─────────┐
   ▼        ▼            ▼             ▼              ▼
  SM      Norm       EM-blocking    EM-matching     Fusion       (committees, frozen YAML SHAs)
                    │
                    ▼
   validation/<domain>/<level>/metrics.json
                    │
                    ▼
  analyze_monotonicity.py   →   monotonicity_report.csv
                                ├── committee macro_f1     (baseline → easy → medium → hard)
                                ├── best-member ceiling    (P8)
                                ├── ceiling_responsiveness (per-knob Pearson on best-member F1)
                                └── per-knob intensity verdict (K5 distinct families, K8 naming rank,
                                                                K10 realised swap rate)
                    │
                    ▼
   validation/<domain>/final_report.md
```

[scripts/validate_variant.py](scripts/validate_variant.py) runs each
committee against a packaged variant and writes a `metrics.json` whose every
leaf carries `_baseline` and `_delta` twins (so a reader can see both
absolute performance and the drop versus baseline at a glance). It refuses to
start if the committee YAML SHAs diverge from those recorded in
`baseline_metrics.json` — drift would silently invalidate every delta.

[scripts/analyze_monotonicity.py](scripts/analyze_monotonicity.py) consumes
the per-level metrics across `baseline → easy → medium → hard` and decides
whether each stage's committee macro_f1 and best-member F1 (the "P8 ceiling"
— the strongest individual matcher's F1, which is what a downstream user
would actually deploy) drop monotonically with difficulty. It also writes
per-knob audit verdicts driven by the realised CSVs above. The final
`final_report.md` per domain rolls those up into a human-readable verdict.

A second analyzer, [analyze_ablation.py](scripts/analyze_ablation.py),
consumes per-knob single-knob-hard ablation metrics
([run_ablation_validation.py](scripts/run_ablation_validation.py)) to attribute
the cross-level drop to each individual knob — that's how we tell whether a
knob is actually pulling its weight or being absorbed by another.

---

## Domain status snapshot

(authoritative source: [plans/plan_revision.md](../plans/plan_revision.md))

| Domain          | Phase 0 pool | Phase 1 baseline | Phase 2 variants            | Phase 3 validation             |
|-----------------|:---:|:---:|---|---|
| music-FULL      | ✓ | ✓ | ✓ | ✓  (R7.3 PASS, R-1 fixes pending) |
| games-FULL      | ✓ | ✓ | ✓ | ✓  (R7.3 PASS, R-1 fixes pending) |
| products-FULL   | ✓ | ✓ | ✓ (legacy 4-col schema)  | stale — R1 schema redesign queued |
| companies-FULL  | ✓ | ✓ | — | blocked on larger fusion gold (R0) |
| `*-small` siblings | ✓ | ✓ | ✓ | ✓ |

"R-1 fixes pending" refers to the diagnosed-but-not-yet-landed quality
improvements in [plan_revision.md §R-1](../plans/plan_revision.md): K2 ratio
not moving, easy occasionally easier than baseline, K5/K8 raw-count proxies
non-monotone on some domains, K10 audit replaced with a rate-based metric.
All current FULL variants still satisfy the headline cross-level
monotonicity criterion but the contract is being tightened before companies-FULL
and the products redesign cascade through.

---

## Reading order for collaborators

1. This file — pipeline mental model.
2. [PIPELINE.md](PIPELINE.md) — runbook with commands per phase.
3. [knobs/README.md](../knobs/README.md) — canonical knob order rationale +
   cross-cutting rules.
4. Individual [knobs/knob_NN_*.md](../knobs/) — full per-knob spec.
5. [plans/plan_revision.md](../plans/plan_revision.md) — current state +
   queued improvements.
