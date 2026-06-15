# Knob 10 — Source reliability differentiation

**Status:** LOCKED. **Scenario:** S1 + S2 (fully controllable; baked into source generation in S2).

## Definition

Spread of per-source accuracy across the corpus. **S1 mechanism: reshuffling** — for each (entity, attribute) cell where Knobs 1/5/6/7 produced multiple variants, Knob 10 chooses which source carries the gold-aligned variant. Pure permutation; never mutates the gold artifact.

**Difficulty axis: per-attribute trust concentration** — easy = clear winner per attribute (trust strategies trivially work); hard = diffuse trust per attribute *and* errors burst across an entity's attributes within a single source (per-attribute voting also fails).

## Dimensions controlled

- Trust Ambiguity (Fusion)
- Conflict Rate (Fusion) — indirect

## Sub-parameters

- `per_attribute_concentration` — for each attribute, the share of cells where the *most-reliable source for that attribute* carries the gold-aligned variant. High at easy, low at hard.
- `error_correlation` — burstiness along the **(source, entity)** axis. When a source is perturbed at `(entity Y, attribute A)`, the elevated probability of also being perturbed at `(entity Y, attribute B)`. Models "this source confused entity Y with a near-twin and got everything about Y wrong." 0 at easy, moderate-to-high at hard.

`trust_ordering_stability` (whether the same source is the per-attribute winner across all attributes) is **not** a knob axis — left to chance / domain baseline.

**Per-attribute winner is re-derived every run** from the freshly-measured `B[s, a]` matrix (see §*Self-contained baseline* in §Algorithm selection); the YAML's `T[a, ·]` addresses sources **by name**, not by rank, so a baseline-drift event (e.g., FullContact overtakes DBpedia on `country` after an upstream re-run) does not silently redirect the gold-carrier — the monotonicity validator either still passes (the named winner in YAML matches the new measured `W[a]`) or fails loud (the named winner no longer matches and the loader raises before any cell is touched).

**Per-(source, attribute-cluster)** correlation ("source X is systematically bad at financial fields") is parked as a possible future sub-parameter.

## Per-source shape via measured baseline

Same pattern as Knobs 3/6/8: measure a baseline per-source × per-attribute agreement-with-gold matrix (one fusion-committee pass on the original under lenient evaluation). Each level transforms that matrix.

**Bootstrap order: committee design → baseline measurement → Knob 10 calibration.** The committee is defined later (algorithm-selection phase), so baseline measurement is deferred to then.

## Easy / Medium / Hard

| Level | `per_attribute_concentration` | `error_correlation` | Target state | Generator action |
|---|---|---|---|---|
| **Easy** | High (~90% of cells go to the per-attribute winner) | 0 (i.i.d.) | For each attribute, one source is the clear winner ~always. A "favor source X for attribute A" strategy trivially recovers the gold. Trust-based fusion is the obvious fallback even when value knobs are aggressive. | Per attribute, identify the baseline-best source from the measured matrix and assign the gold-aligned variant to that source on ~all cells. |
| **Medium** | Moderate (~70% on the winner; e.g. 70/20/10) | Low | Per attribute, one source is most-reliable but not absolute. Trust-aware strategies pull ahead of voting; voting still recovers most cells. | Per attribute, assign gold to the baseline winner on ~70% of cells; distribute the remaining 30% among the others. Light burstiness on the perturbed cells. |
| **Hard** | Low / near-uniform (~40/35/25) | Moderate-to-high | Per attribute, **no source is consistently reliable**. Source-level trust strategies fail. **Per-(source, entity) error bursts** sabotage per-attribute voting: when source X is wrong on `(Y, revenue)`, it is *also* likely wrong on `(Y, employee_count)` and `(Y, founding_year)`. With elevated probability *two* sources end up simultaneously compromised on the same entity, which flips voting majorities. | Per attribute, distribute the gold-carrier role nearly uniformly across surviving sources. Then sample a per-(source, entity) "compromised" mask: for each source, pick a fraction of entities and elevate the probability of perturbed variants across *all* of those entities' attributes from that source. |

### Why correlated errors specifically degrade voting

Independent perturbations let majority voting recover most cells. Per-(source, entity) burstiness violates the i.i.d. assumption: the pathological case is when *two* sources end up compromised on the same entity. At that point voting *flips the wrong way* across multiple of Y's attributes, and the only recovery is **entity-level** reliability reasoning. This is what makes hard genuinely hard: cell-local strategies (voting, per-attribute trust) collapse together; recovery requires reasoning over the (source, entity) provenance grid.

## Composition

- **Knobs 1/5/6/7:** Knob 10 runs **after** these and operates on whatever variants they produced. **If all four are at easy → no variants → Knob 10 silently degenerates to a no-op.** Accepted: trust signal only matters when sources actually disagree.
- **Knob 3:** orthogonal — Knob 10 acts only on surviving non-null cells.
- **Knob 4:** orthogonal — Knob 10 reshuffles within whatever sources survive Knob 4.
- **Knob 8:** orthogonal — header only.
- **Pipeline order:** Knob 2 → Knob 4 → Knobs 1/5/6/7 → Knob 3 → **Knob 10** → Knob 8 (S1).

## Test-set treatment

- **Fusion:** gold values and entity membership untouched by construction.
- **EM / SM:** unaffected.

## Fusion safety

Trivial — gold is invariant under permutation. **No collapse risk, no rollback machinery.** The committee will see different fusion-strategy spreads across levels, and that spread is the difficulty signal.

## Committee expectations

- **SM / Blocking / EM:** flat.
- **Fusion:** primary target.
  - **Easy:** per-attribute "favor source X" strategies and voting both win.
  - **Medium:** per-source-weighted strategies pull ahead of plain voting.
  - **Hard:** per-source-trust strategies fail (no per-attribute concentration to exploit) *and* per-attribute voting fails (correlated bursts flip majorities). Recovery requires **entity-level provenance reasoning** or LLM arbitration.
- Knob 10's diagnostic spread is over the type of fusion strategy (cell-local vs entity-aware), same flavor as Knob 9's SM matcher-type spread.

## Scenario 2 treatment

The S1 mechanism is post-hoc reshuffling. In **S2**, reliability priors are baked directly into source generation: the synthetic source generator is given a per-attribute trust profile (and at hard, a per-(source, entity) compromised mask) as a generation parameter, and produces values that already realize that profile. Sub-parameter semantics, level definitions, and committee expectations carry over unchanged.

## Per-domain notes

All three domains naturally sit at the **easy** end of this knob; the generator does most of the work moving them toward hard.

- **Companies:** Forbes likely the clear winner on financials, DBpedia on encyclopedic/founding facts, FullContact on contact-side attributes. Per-attribute concentration naturally **high**.
- **Games:** Metacritic clear on scores, Sales clear on globalSales/publisher, DBpedia on franchise/series. Per-attribute concentrated.
- **Music:** MusicBrainz canonical on identifiers/dates, Discogs on label/genre, LastFM on popularity (which is out-of-target). Per-attribute concentrated on the overlapping attributes.

## Provenance

Per-cell record of the assignment: `(entity_id, attribute, gold_source, perturbed_sources, knob=10, level)`.

**Plus at hard:** per-(source, entity) compromised-mask records — `(source, entity_id, compromised=True, knob=10, level)` — so entity-aware fusion strategies have a recoverable signal in evaluation logs and we can debug whether the burst mechanism actually fired.

## Algorithm selection

**Chosen approach.** Tier A — deterministic in-house pure pandas + `numpy.random.default_rng` source-label permutation, driven by a per-attribute target gold-source distribution and a per-(source, entity) compromised mask. **No literature method ported, no LLM.** The mechanism is *reshuffling*, not corruption: for each fusion-gold (entity, attribute) cell where Knobs 1/5/6/7 produced ≥2 sources whose values are not all gold-aligned, Knob 10 chooses which source carries the gold-aligned variant and permutes the source labels on the cell's value multiset accordingly. The total multiset of values at the cell is invariant under the permutation; only the `source → value` mapping changes. As a direct corollary the fusion gold artifact is **byte-identical before and after every run** (it is read-only) and the per-source value-count per cell is conserved (pure permutation invariant). The dispatcher (a) **measures fresh** at the start of every run a per-(source, attribute) baseline gold-alignment matrix `B[s, a]` from the post-Knobs-1/5/6/7-and-3 source DataFrames using **direct canonical-form equality against the fusion gold value** (no committee call required at apply time — see *Self-contained baseline* below), (b) identifies the per-attribute baseline winner `W[a] = argmax_s B[s, a]`, (c) loads the level-keyed target distribution `T[a, s]` and the level-keyed compromise rates / correlation strength from per-domain YAML, (d) at hard, draws a per-(source, entity) compromised mask `C[s, e]`, (e) for every reshufflable cell, samples the gold-source assignment from `T[a, ·]` re-weighted by `C[·, e]` and conditioned on the actual set of sources present at the cell, then (f) permutes the source labels on the cell's value list to realize the assignment. All randomness is one seeded `numpy.random.default_rng(seed)` per `(domain, variant, knob=10)` tuple. The function is pure pandas / `numpy` / `pyyaml` / stdlib; no new dependencies, no PyDI extension points.

**Self-contained baseline (resolves the bootstrap-order deferral).** The locked card §*Per-source shape via measured baseline* says baseline measurement requires "one fusion-committee pass on the original under lenient evaluation," and §*Bootstrap order* defers Knob 10 calibration until after the committee is designed. Resolution: the *information* the committee would provide is a binary `(s, a, e) → {gold-aligned, perturbed}` label per cell, which is exactly what canonical-form equality against the gold value gives directly — no committee plumbing needed. The committee is still required *post-hoc* to validate the spread of fusion-strategy outputs across the easy/medium/hard variants (per *Committee expectations* below), but it is **not** required during Knob 10 application. The Knob 10 dispatcher therefore implements `is_gold_aligned(value, gold_value, attribute_class) -> bool` as a pure function over the same canonical-form comparator the Knob 5 dispatcher already uses for round-trip verification (date → `datetime.date`, number → `Decimal` within tolerance, money → `Decimal` after FX, duration → `timedelta`, dimensional → `Decimal` after unit normalization, string → casefold + collapse-whitespace + strip-punct).

**Reconciliation with Knob 5's `attribute_classes` taxonomy.** Knob 5 authors `attribute_classes` as `{source_name: {column: format_family}}` with `format_family ∈ {date, number, money, duration, dimensional}` (see [knob_05_format_unit.md:99](knob_05_format_unit.md#L99)). Knob 10 needs a flat per-attribute `{attribute → comparator_class}` map. The reconciliation is owned by Knob 10 and runs at config load:

1. **Family → comparator routing** (one-to-one): `date → date`, `number → number`, `money → money`, `duration → duration` (parse to `timedelta`), `dimensional → dimensional` (parse to `Decimal` after unit normalization via Knob 5's `unit_factors.yaml`). Attributes absent from Knob 5's map default to `string` (casefold + collapse-ws + strip-punct).
2. **Per-source nesting collapse**: for each attribute `a`, take the majority `format_family` across sources that declare it. Tiebreak by canonical (sorted) source order. If sources disagree (e.g. one declares `duration`, another `number` for the same attribute), emit a `WARNING` to the run log naming the attribute and the conflicting (source, family) pairs, and use the majority/tiebreak winner. The disagreement is a Knob 5 authoring bug; Knob 10 does not crash on it but logs loudly so it surfaces in CI.
3. The resulting flat `{attribute → comparator_class}` map is what `is_gold_aligned` consults. This guarantees that a Music `duration` attribute is compared as a parsed `timedelta` (so `180000` ms equals `03:00`) and a Companies `dimensional` revenue is compared as a unit-normalized `Decimal`, instead of falling through to string casefold which would break the canonical-form equality guarantee. This makes Knob 10's algorithm selection self-contained — implementer reads this card + Knob 5's canonical-form comparator and has everything needed; the committee (Step 6/7 deliverable) is not on the critical path for this knob.

**Cross-knob invariants (locked I5 from the Step 5 cross-knob review).**
- **Knob 1's anchor-survivor floor is load-bearing for Knob 10.** [knob_01_surface_augmentation.md:84](knob_01_surface_augmentation.md#L84) guarantees that for every fusion-gold entity ≥1 source retains a non-paraphrased primary value. This floor is what makes the reshufflable-cell predicate's `|S_aligned| ≥ 1` condition hold in practice on primary attributes — without the K1 floor, hard K1 could in principle perturb every source at a primary cell, leaving Knob 10 with only the `no_gold_to_route` audit path. Flagged here for symmetry with the K1 card; Knob 10 does not enforce the floor itself (K1 owns it) but its monotonicity story depends on it.
- **Knob 1 hard gold-extend mutates the gold before Knob 10 runs.** [knob_01_surface_augmentation.md:86](knob_01_surface_augmentation.md#L86) allows K1's fix-on-collapse path to extend the fusion-gold accepted set. Knob 10's SHA-256 byte-identity assertion therefore covers only its own window (pre-K10-run vs post-K10-run), not the full-pipeline window — any gold-extend rows written by K1 land before Knob 10 starts, and the SHA Knob 10 records at `output/baselines/knob_10_gold_hash.txt` is the post-K1 hash. The assertion is still a hard contract (Knob 10 never mutates gold), just scoped to the Knob 10 window.

**Reshufflable-cell predicate.** A fusion-gold cell `(e, a)` is *reshufflable* iff (i) ≥ 2 sources have a value at `(e, a)` after Knobs 3/4 (i.e., the cell survived attribute drop and entity removal), (ii) at least one of those sources is gold-aligned, and (iii) at least one is *not* gold-aligned. Cells that fail any condition are passthroughs and emit no provenance. Concretely:

| Cell state | Knob 10 action | Provenance |
|---|---|---|
| 0 sources have a value | passthrough | none |
| 1 source has a value | passthrough | none |
| ≥ 2 sources, all gold-aligned | passthrough | none |
| ≥ 2 sources, none gold-aligned | passthrough | one row with `transform_fn=no_gold_to_route` for audit |
| ≥ 2 sources, ≥ 1 gold-aligned, ≥ 1 perturbed | reshuffle | one row per cell |

**Mapping to easy/medium/hard.** Monotonicity is enforced by level-indexed selection of three independently togglable scalars: (i) the per-attribute target distribution `T[a, ·]` (concentration on the winner), (ii) the per-source compromise rate `compromise_rate[s]` (fraction of entities flagged compromised on source `s`), and (iii) the correlation strength `corr_strength` (how strongly the compromised mask down-weights a source in the per-cell sampling step). The level table:

| Level | `T[a, W[a]]` (winner share) | `T[a, second]` | `T[a, rest]` | `compromise_rate` (per source) | `corr_strength` |
|---|---|---|---|---|---|
| **Easy** | ~0.90 | ~0.07 | residual (~0.03) | 0.0 | 0.0 |
| **Medium** | ~0.70 | ~0.20 | ~0.10 | ~0.05 | ~0.20 |
| **Hard** | ~0.40 | ~0.35 | ~0.25 | ~0.15 | ~0.50 |

`T[a, ·]` is **mandatory per (domain, attribute) in YAML** with the table above as a reference shape — there is no cross-domain default because the per-attribute winner identity differs (Forbes financials vs. Metacritic scores vs. MusicBrainz IDs) and authoring `T` per attribute is the only way to express the natural concentration. The YAML loader validates monotonicity on load (`T_easy[a, W[a]] ≥ T_med[a, W[a]] ≥ T_hard[a, W[a]]` per attribute, plus `compromise_rate.easy ≤ med ≤ hard` and `corr_strength.easy ≤ med ≤ hard`) and refuses to run if violated. **`W[a]` is the *measured* baseline winner**, resolved at run start from the freshly-measured `B[s, a]` matrix (§*Self-contained baseline*) — not authored. The YAML addresses sources by name in `T[a, ·]` (not by rank), so the monotonicity check looks up `W[a]` from the measured matrix and then indexes the YAML by the corresponding source name. If the measured winner is not present in the YAML's `T[a, ·]` for that attribute, the loader raises immediately.

**Why two correlation knobs and not one.** `compromise_rate` controls *how many* entity-source pairs are bursty; `corr_strength` controls *how aggressive* the burst is (i.e., how much the compromised mask shifts the per-cell sampling weights). Splitting them lets the calibration loop trade off "many mild bursts" vs "few severe bursts" without re-authoring `T[a, ·]`. The default ratio (`compromise_rate.hard = 0.15`, `corr_strength.hard = 0.5`) yields P(any single source compromised at entity `e`) = 1 − (1 − 0.15)³ ≈ 0.39 across N=3 sources (independent per-source draws), and P(≥ 2 sources compromised at the same entity `e`) = 1 − (0.85)³ − 3·0.15·(0.85)² ≈ 0.0608 ≈ 0.07 — so roughly 7% of fusion-gold entities sit in the pathological zone where two sources are simultaneously bursty and per-attribute voting flips the wrong way. This is the load-bearing rate behind the *hard* committee expectation that "per-source-trust strategies fail and per-attribute voting fails too" (§*Committee expectations* above).

**Sampling algorithm (deterministic two-stage).**

1. **Compromised-mask stage** (hard only; medium uses a smaller mask, easy is empty). For each source `s`, sample `floor(compromise_rate[level] · |entities|)` entities uniformly without replacement using the seeded RNG. Store as a `{source → set(entity_id)}` map. Independent across sources by construction. Written to `output/provenance/knob_10_compromised_mask.csv` with columns `(source, entity_id, knob=10, level)` per the locked Provenance section.
2. **Per-cell sampling stage.** For every reshufflable cell `(e, a)`, let `S_present = {sources with a value at (e, a)}` and `S_aligned = {s ∈ S_present : is_gold_aligned(value[s], gold[e, a], attribute_class[a])}`. Compute the per-cell weight vector
   ```
   w[s] = T[a, s] · (1 - corr_strength · 1[s ∈ C[·, e]])      for s ∈ S_present
   ```
   then re-normalize over `S_present` and sample one source `s_gold` from the resulting categorical distribution. The chosen `s_gold` becomes the new gold-carrier at this cell.
   - **Swap-target selection.** Define the swap target deterministically: if `s_gold ∈ S_aligned`, no permutation is needed (emit a `transform_fn=identity` provenance row). Otherwise, swap with `s_swap`, defined as the lowest-indexed source in `S_aligned` under canonical source order (canonical = sorted source name from the per-domain registry). This rule is total: by the reshufflable-cell predicate `|S_aligned| ≥ 1`, so `s_swap` always exists, and when `|S_aligned| ≥ 2` the tiebreaker is unambiguous and stable across runs.
   - **Permutation.** Swap the values at `s_gold` and `s_swap` only — a 2-cycle on the cell's value list. All other sources in `S_present` keep their existing values. The multiset of values at the cell is preserved by construction (2-cycle invariant).

Independent togglability: Knob 10 reads only the post-Knobs-3/1/5/6/7 source DataFrames plus the fusion gold (read-only). It writes only to source-row cells in already-present columns; it never adds or removes rows (Knob 4's territory), never adds or removes columns (Knobs 8 / 9 territory), never drops cells (Knob 3's territory), and mutates cell values via permutation only — the multiset of values at each cell is preserved across the run, so corruption-sense value mutation (Knobs 1/5/6/7 territory) is structurally impossible here. It composes at position 6 of the canonical S1 order (`Knob 2 → Knob 4 → Knobs 1/5/6/7 → Knob 3 → Knob 10 → Knob 8`, per [README.md](README.md#canonical-knob-application-order)). **At all-easy on Knobs 1/5/6/7, the reshufflable-cell set is empty by construction and Knob 10 is a no-op** — accepted per the Composition section of this card.

**Constraint resolution order** (Knob 10 has very few constraints — pure permutation is intrinsically safe):

1. **Pure-permutation invariant.** The multiset of values at every (entity, attribute) cell is preserved. The dispatcher asserts this in-process after each cell is reshuffled (`Counter(values_before) == Counter(values_after)`).
2. **Read-only fusion gold.** The fusion gold file is opened read-only; the dispatcher hashes it before and after the run and asserts byte-identity.
3. **Per-attribute concentration cap** (defensive). If `T[a, W[a]] > 0.99` (extreme winner concentration), the dispatcher caps it at 0.99 to keep the sampling distribution from degenerating to a deterministic assignment. Logged with a warning.
4. **No fix-on-collapse loop.** Per [cross_cutting.md §Per-knob fix-strategy defaults](cross_cutting.md#per-knob-fix-strategy-defaults): "*No gold change needed; gold is reshuffled across sources, not perturbed.*" Knob 10 has no rollback machinery. If the committee shows non-monotone fusion deltas, the fix is to soften the level-transform parameters in YAML (lower `corr_strength`, lower `compromise_rate`, or shift `T[a, ·]` toward identity) and re-run — same flavor as Knob 6's *soften augmentation locally* default but without rollback rows in provenance, because nothing was ever "wrong" in the value sense.

**Literature citations.** Tier A — no literature method is ported wholesale. Three light cites for defensibility:

- **DAPO** ([../literature-search-generation/dapo_large_scale_data_pollution/paper.md](../literature-search-generation/dapo_large_scale_data_pollution/paper.md)) — *Per-Source Corruption Profiles* (paper.md:77–78, 234) and *Multi-Source Dataset Generation* (paper.md:201–219). Cited as the canonical precedent for treating per-source quality differentiation as a first-class benchmark-generation knob ("varying quality across sources is more realistic than uniform corruption"). Knob 10 generalizes DAPO's per-source corruption-rate vector into a per-attribute trust concentration distribution, but the underlying claim — that fusion benchmarks must expose differentiated source quality to be meaningful — is DAPO's.
- **PseudoPeople** ([../literature-search-generation/pseudopeople_census_er/paper.md](../literature-search-generation/pseudopeople_census_er/paper.md)) — *Correlated Noise Injection via Life Events* (paper.md:282–306, 384–393). Cited as the precedent for **per-entity correlated errors across attributes** (the locked card's `error_correlation` axis): "real-world data inconsistencies are often correlated, not independent." PseudoPeople realizes correlation through life-event semantics (marriage → name change cascades across records); Knob 10 abstracts this into a per-(source, entity) compromised mask that elevates the perturbation probability across all of an entity's attributes from one source. The mechanism is the same — joint perturbation per entity rather than i.i.d. per cell — even though the semantic dressing is removed.
- **Christen / Vatsalan data-corruption taxonomy** ([../literature-search-generation/christen_vatsalan_data_gen_corruption/paper.md](../literature-search-generation/christen_vatsalan_data_gen_corruption/paper.md)) — boundary anchor. Christen / Vatsalan §*Limitations* (paper.md:557) explicitly notes "All corruption functions are applied independently per field per record. The framework does not model correlated errors." Knob 10's `error_correlation` sub-parameter is positioned exactly in the gap Christen / Vatsalan flagged; cited to show we are filling a known gap in the standard corruption-taxonomy literature, not inventing a novel axis.

**No literature method beyond these cites.** Justification: the knob behavior we need (deterministic per-attribute target distribution + correlated burst mask + pure permutation of source labels with gold invariance) is a benchmark-engineering operation with no research content beyond the per-source-profile + correlated-error taxonomy anchors above. Truth-discovery methods from the data-fusion literature (TruthFinder, AccuPR, LCA, Latent Truth Models, Bayesian source-trust models) are explicitly **rejected** as a generation primitive — they are *evaluation* methods that fusion strategies are supposed to either succeed or fail at; baking them into the generator would entangle the difficulty signal with the evaluation surface. **LLM not used because the deterministic alternative is sufficient** (and pure permutation has no sub-task that an LLM could plausibly improve on).

**Determinism & provenance.**
- **Baseline gold-alignment matrix is refreshed every run** (same never-cache reasoning as Knobs 3/4/6): the dispatcher calls an in-script `measure_gold_alignment(sources, fusion_gold)` function over the post-Knobs-1/5/6/7-and-3 DataFrames at the start of every Knob 10 invocation and writes the observed per-(source, attribute) gold-alignment-rate matrix to `output/baselines/knob_10_baseline_alignment.csv` inside the variant directory. The `measured_at` timestamp column is recorded inside the row body for human debugging but is **excluded from the bit-identity check** in smoke test (g) — the determinism assertion compares the CSV with the `measured_at` column dropped (or equivalently, only checks the `(source, attribute, baseline_alignment_rate, n_cells, n_aligned)` columns). The measurement is one canonical-form pass per source (`{value: canonicalize(value); is_gold = canonicalize(gold[entity_id, attr])}`) — cheap, no network, no committee.
- RNG: a parent `numpy.random.SeedSequence(seed)` per `(domain, variant, knob=10)` tuple; seed recorded in the variant's `config/difficulty.yaml`. The parent is split via `parent.spawn(2)` into two independent child seed sequences — one drives the compromised-mask stage, the other drives the per-cell sampling stage. Each is wrapped in `numpy.random.default_rng(child)`. This guarantees that changing `T[a, ·]` in the YAML does not perturb the compromised mask, and changing `compromise_rate` / `corr_strength` does not perturb the per-cell sampler — preserving independent togglability between the two sub-parameters. Draw order within each stage is the canonical `entity_ids × attributes` iteration (sorted by ID, then attribute name) so that level-independent cells do not reshuffle between level changes for unrelated reasons.
- Per-domain config file: `usecases_synthetic/config/knob_10_reliability/<domain>.yaml`. Keys:
  - `attribute_targets`: `{attribute: {easy: {source: float}, medium: {source: float}, hard: {source: float}}}`. **Mandatory per attribute** — the per-attribute winner identity is encoded by which source has the highest share at each level. Sums must be 1.0 ± 1e-6 per (attribute, level). Authored once, checked in. Reference shape in the table above.
  - `compromise_rate_per_level`: `{easy: 0.0, medium: 0.05, hard: 0.15}` — fraction of entities to flag compromised on each source, applied uniformly across sources. Per-source overrides allowed via `compromise_rate_overrides: {source: {level: rate}}` for asymmetric domains.
  - `corr_strength_per_level`: `{easy: 0.0, medium: 0.20, hard: 0.50}` — multiplicative down-weight applied to a source's `T[a, s]` at cells where it sits in the compromised mask for the entity.
  - `concentration_cap`: scalar (default 0.99) — defensive cap on `T[a, W[a]]`.
- Provenance written per affected cell to `output/provenance/knob_10_reliability.csv` inside the variant directory, following the [cross_cutting.md §Per-value provenance](cross_cutting.md#per-value-provenance-mandatory) flat-row schema:
  ```
  (entity_id, source, attribute, original_value, new_value,
   transform_fn ∈ {reassign_gold_carrier, identity, no_gold_to_route},
   transform_params, knob=10, level)
  ```
  One row is emitted per source touched by the permutation at a given cell — typically two rows per reshuffled cell (the old gold-carrier loses the gold value, the new gold-carrier gains it). `transform_params` JSON keys:
  - `reassign_gold_carrier`: `{gold_source_before, gold_source_after, perturbed_sources, weight_vector, sampled_from_compromised_mask: bool}`.
  - `identity`: `{gold_source: source, sampled: true}` — the sampler chose the same gold-carrier as before; emitted only when the cell *was* reshufflable (not for trivial passthroughs).
  - `no_gold_to_route`: `{perturbed_sources, reason: "no_aligned_value"}` — emitted for the audit-only case where Knobs 1/5/6/7 perturbed every source at this cell.
- Compromised-mask records (hard only, locked-card requirement) at `output/provenance/knob_10_compromised_mask.csv`:
  ```
  (source, entity_id, compromised=True, knob=10, level)
  ```
  One row per (source, entity) pair sampled into the mask. Used by the committee evaluation logs and for debugging whether the burst mechanism actually fired the way the YAML asked for.
- Baseline gold-alignment matrix at `output/baselines/knob_10_baseline_alignment.csv`:
  ```
  (source, attribute, baseline_alignment_rate, n_cells, n_aligned, knob=10, measured_at)
  ```
- The fusion gold file is **read-only**; the dispatcher reads it directly from disk (independently of the in-memory `fusion_gold` DataFrame parameter, which is used only for the alignment-matrix lookup), computes its SHA-256 before the run, repeats the read after the run, and asserts byte-identity. Hash recorded in `output/baselines/knob_10_gold_hash.txt`.
- Caching: output is a file artifact on disk (mutated source datasets + provenance CSVs + baseline CSV + compromised-mask CSV). Seeded RNG + fresh baseline measurement guarantee reproducible regeneration against the *current* source snapshot.
- Committee surface: the Fusion committee (per the *Committee expectations* section above and [cross_cutting.md §Committee composition (fusion, draft)](cross_cutting.md#committee-composition-fusion-draft)) sees the reshuffled source files exactly as written. The diagnostic spread is over the *type* of fusion strategy: cell-local strategies (`most_frequent`, `voting`, per-attribute trust) collapse together at hard, while entity-aware strategies (per-source trust + per-(source, entity) provenance reasoning, or LLM arbitration) survive — same flavor as Knob 9's SM matcher-type spread, as the locked card already notes. SM, Blocking, and EM committees should be flat (Knob 10 is fusion-only).

**Domain-specific adjustments.** All three domains naturally sit at the **easy** end of this knob per the locked card's *Per-domain notes* — per-attribute trust concentration is high at baseline because each source has natural dominance in its niche. The generator therefore does most of the work moving toward hard, and easy is close to identity (or actively concentrates already-concentrated cells further toward the winner via the ~0.90 share). Per-domain `attribute_targets` authoring guidance:

- **Companies.** Forbes is the per-attribute winner on financial fields (`revenue`, `total_assets_val`, `net_income`, `market_value`), DBpedia on encyclopedic / founding facts (`founded`, `industry`, `founder`, `headquarters_location`), FullContact on contact-side attributes (`employees_count`, `country`, `website`). The easy `attribute_targets` shape places ≥0.85 on the per-attribute winner; medium drops to ~0.65; hard to ~0.40 with `corr_strength.hard = 0.50`. The locked card flags Companies hard as the most consequential — Forbes is the only "clean" source overall, so compromising it on a fraction of entities meaningfully shifts the fusion landscape.
- **Games.** Metacritic is the per-attribute winner on score / rating attributes, Sales on `globalSales` / `publisher` / `platform`, DBpedia on `franchise` / `series` / `developer`. The narrow Sales schema means many fusion-gold attributes only have meaningful values from 1–2 sources — Knob 10's reshufflable-cell set is therefore smaller on Games than on Companies, and the smoke test should expect a lower per-cell mutation count. The committee monotonicity check still applies but the magnitude of the deltas will be more compressed.
- **Music.** MusicBrainz is canonical on identifiers, dates, country, language; Discogs on label, genre, format, packaging; LastFM on popularity (which is an out-of-target attribute and excluded from `attribute_targets`). The bimodal-headroom note from Knob 3 carries over: scalar-attribute reshuffling concentrates on MusicBrainz / Discogs scalar overlap; the array-attribute reshuffling (`tracks_*`) is intrinsically harder to canonicalize and is excluded from `attribute_targets` in v1 (deferred until the committee composition for array attributes is locked).
- **Movies, products.** Deferred alongside the Step 6 prototype. The dispatcher no-ops on domains without a `config/knob_10_reliability/<domain>.yaml` (warns in the log). No code change required when those domains come online — only a new YAML.

**Rejected alternatives.**

- **LLM-based reliability simulation** (e.g., prompting an LLM to "decide which source would plausibly carry the right value at this cell"). Rejected: the task is a parametric source-permutation dial with deterministic targets; it has a strong pandas baseline and LLM use would sacrifice determinism, explode validation cost, and deliver zero expected quality gain. There is no creative sub-task — the values at the cell already exist; Knob 10 only chooses which source gets which one. **LLM not used because the deterministic alternative is sufficient.**
- **Truth-discovery models as generation primitives** (TruthFinder, AccuPR, LCA, Latent Truth Models, Bayesian source-trust models). Rejected: these are *evaluation* methods that fusion strategies are supposed to either succeed or fail at on the generated benchmark. Baking them into the generator would entangle the difficulty signal with the evaluation surface — the benchmark would silently rubber-stamp the very methods it claims to evaluate. The cross-cutting principle is "the benchmark must not rubber-stamp its own pooling systems" ([cross_cutting.md §Protection set semantics](cross_cutting.md#protection-set-semantics-not-replacement-gold)); this is the same principle applied to the fusion stage.
- **Mutating the gold to match a "noisier" majority** (e.g., extending the gold to an accepted set when the reshuffling looks too aggressive). Rejected: pure permutation cannot make the gold unrecoverable, so there is no failure mode that mutating the gold would resolve. The fusion-gold artifact is invariant under Knob 10 by construction.
- **Per-(source, attribute-cluster) systematic-bias modeling** ("source X is systematically bad at financial fields" beyond a single attribute). Parked as a future sub-parameter per the locked card's *Sub-parameters* section. Not implemented in v1 because the per-attribute target distribution `T[a, ·]` already produces meaningful per-attribute reliability patterns; cluster-level bias would require an additional per-domain attribute-clustering authoring step with marginal expected gain over the per-attribute targets.
- **Heavyweight ML correlated-error generators** (deep generative models for joint per-entity perturbation, autoregressive entity-level corruption transformers). Rejected under the [plan_algorithmselection.md](../plan_algorithmselection.md) framework rule against heavyweight ML methods — they violate determinism, validation cost, and dependency weight simultaneously, and they optimize for realistic joint distributions, not for monotone controlled trust dials with per-attribute concentration and compromised-mask burst rates.
- **Caching the baseline gold-alignment matrix across runs** for speed. Rejected per the same Knob 3 / 4 / 6 reasoning: source data can change between runs (Knobs 1/5/6/7 may have been re-run upstream), the measurement is cheap (one canonical-form pass per source), and a stale baseline would silently break the per-attribute winner identification.
- **Replacing the compromised-mask with a per-cell i.i.d. flip** (i.e., dropping the burst mechanism entirely and just sampling each cell independently from `T[a, ·]`). Rejected because i.i.d. cell flips do not violate the voting assumption — majority voting recovers most cells under independent perturbation. The locked card's *Why correlated errors specifically degrade voting* section is the explicit rationale; the burst mechanism is what makes hard genuinely hard.

**Implementation handoff.** Everything Step 6 needs to implement this knob without re-reading surrounding cards:

- **Target script:** `usecases_synthetic/scripts/apply_knob_10_reliability.py` (new, convention matches `apply_knob_03_attribute_drop.py`, `apply_knob_04_coverage_skew.py`, `apply_knob_06_noise.py`, `apply_knob_08_naming.py`). Standalone runnable from repo root.
- **Function shape (illustrative):**
  ```python
  def apply_knob_10(
      domain: str,
      level: Literal["easy", "medium", "hard"],
      sources: dict[str, pd.DataFrame],           # post-Knobs-1/5/6/7-and-3
      fusion_gold: pd.DataFrame,                  # read-only; alignment-matrix lookup only
      fusion_gold_path: Path,                     # on-disk file used for SHA-256 byte-identity check
      config_path: Path,                          # usecases_synthetic/config/knob_10_reliability/<domain>.yaml
      knob_05_config_path: Path,                  # for the shared `attribute_classes` block
      output_dir: Path,
      seed: int,
      canonicalize: Callable[[Any, str], Any] | None = None,
                                                   # shared with Knob 5; default = in-script casefold/decimal/date
  ) -> tuple[
      dict[str, pd.DataFrame],                    # mutated_sources
      pd.DataFrame,                               # provenance_df
      pd.DataFrame,                               # compromised_mask_df  (empty at easy; ~5% mask at medium; ~15% mask at hard)
      pd.DataFrame,                               # baseline_alignment_df
  ]:
      """Reshuffle gold-carrier source labels per fusion-gold cell.

      baseline_alignment_df is measured fresh from `sources` and `fusion_gold` at the
      start of every call and written to output/baselines/knob_10_baseline_alignment.csv.
      The `fusion_gold` DataFrame is used only for the in-memory alignment lookup; the
      on-disk file at `fusion_gold_path` is read directly (independently of the DataFrame),
      hashed before and after the run, and byte-identity is asserted on the second hash.
      """
  ```
- **Inputs the script reads:**
  - Source DataFrames (via `PyDI.io.load_*`, preserving `df.attrs["dataset_name"]`), already mutated by Knobs 1/5/6/7 and Knob 3 upstream in the canonical S1 order.
  - The fusion gold artifact from `usecases/<domain>/input/fusion/` — read-only. Loaded once into a DataFrame for the alignment-matrix lookup, *and* read again as raw bytes from `fusion_gold_path` for the SHA-256 byte-identity check. **Never mutated.**
  - Per-domain Knob 10 config at `usecases_synthetic/config/knob_10_reliability/<domain>.yaml` (`attribute_targets`, `compromise_rate_per_level`, `corr_strength_per_level`, optional `compromise_rate_overrides`, `concentration_cap`).
  - Per-domain Knob 5 config at `usecases_synthetic/config/knob_05_format_unit/<domain>.yaml` — the `attribute_classes` block (`{source: {column: format_family}}` with `format_family ∈ {date, number, money, duration, dimensional}`). Knob 10 collapses the per-source nesting into a flat `{attribute → comparator_class}` map at config load (majority across sources, tiebroken by canonical source order, warn on disagreement) and routes `duration` → `timedelta` parser, `dimensional` → `Decimal` after unit normalization via Knob 5's `_tables/unit_factors.yaml`. Attributes absent from this map default to `string` (casefold + collapse-ws + strip-punct). See *Reconciliation with Knob 5's `attribute_classes` taxonomy* in §Algorithm selection. Knob 10 does not depend on any other Knob 5 config key apart from `unit_factors.yaml` for dimensional unit normalization.
  - The shared canonical-form comparator from Knob 5's dispatcher (or an in-script default if Knob 5 has not been wired up yet — the comparator logic is small enough to inline).
- **Outputs the script writes** (under the variant directory):
  - Mutated source files in `input/data/` (same format as input — XML/JSON/CSV). Only cell *values* are permuted across sources; row counts, column sets, and headers are unchanged.
  - Freshly measured baseline gold-alignment matrix at `output/baselines/knob_10_baseline_alignment.csv` (every run; not cached).
  - Provenance log at `output/provenance/knob_10_reliability.csv`.
  - Compromised-mask log at `output/provenance/knob_10_compromised_mask.csv` (hard primarily; medium emits a small mask; easy emits an empty file with header).
  - Gold-hash sentinel at `output/baselines/knob_10_gold_hash.txt` (SHA-256 of the fusion gold file before the run; checked again after).
- **Pipeline integration:** Knob 10 sits at position 6 of the canonical S1 order from [README.md](README.md#canonical-knob-application-order), running after `Knob 3 (attribute drop)` and before `Knob 8 (header rename)`. It sees Knob 3's dropped cells as missing values and only operates on cells where ≥ 2 sources still carry a value. Knob 8 is header-only and orthogonal, so it sees the reshuffled values with the original headers and renames them last.
- **Dependencies:** stdlib + `pandas` + `numpy` + `pyyaml`. No new runtime dependencies. No PyDI extension points.
- **Authoring task before first run:** populate `usecases_synthetic/config/knob_10_reliability/{companies,games,music}.yaml` with `attribute_targets` (per-attribute, per-level distributions over sources, sums = 1.0), `compromise_rate_per_level`, `corr_strength_per_level`, optional `compromise_rate_overrides`, and `concentration_cap`. The baseline gold-alignment matrix is **measured**, not authored.
- **Smoke test (Phase 2, no committee dependency):** for each domain with a config, run the script at all three levels and assert
  - **(a)** the fusion gold file on disk is byte-identical (SHA-256 match) before and after the run;
  - **(b)** the multiset of values at every (entity, attribute) cell is preserved by the permutation (`Counter(values_before) == Counter(values_after)` per cell — pure-permutation invariant);
  - **(c)** per-source total cell count is unchanged (corollary of (b));
  - **(d)** realized per-attribute concentration on the winner is monotone non-increasing easy → medium → hard, modulo a per-attribute tolerance of ±0.05 driven by sampling noise;
  - **(e)** the compromised-mask CSV row count: at easy, **0 rows**; at medium, `N_sources · floor(0.05 · |entities|)` rows; at hard, `N_sources · floor(0.15 · |entities|)` rows — one row per (source, entity) pair sampled into the mask, summed across sources, modulo per-source overrides from `compromise_rate_overrides`;
  - **(f)** the provenance row count equals `2 · N_swap + 1 · N_identity + 1 · N_no_gold_to_route`, where `N_swap` is the number of reshufflable cells where `s_gold ≠ s_swap` (two rows per cell — the source losing the gold-aligned value and the source gaining it), `N_identity` is reshufflable cells where the sampler chose a `s_gold` that was already in `S_aligned` (one row per cell), and `N_no_gold_to_route` is the audit-only case from row 4 of the *Reshufflable-cell predicate* table (one row per cell);
  - **(g)** re-running with the same seed on the same source snapshot produces bit-identical outputs (mutated sources, provenance CSV, compromised-mask CSV, baseline CSV);
  - **(h)** at all-easy on Knobs 1/5/6/7 (i.e., the upstream value knobs produced no variants), the reshufflable-cell set is empty by construction, the dispatcher emits a header-only `output/provenance/knob_10_reliability.csv` (zero data rows), logs a single `INFO` line stating the no-op condition, and exits cleanly with status 0.
- **Phase 3 acceptance criteria (require the Step 6/7 committees and live in the cross-cutting committee verification harness, not in Knob 10's standalone smoke test):**
  - **(i)** the SM, Blocking, and EM committee deltas relative to baseline are within noise — Knob 10 should not move them (sanity check that Knob 10 is fusion-only as advertised);
  - **(j)** the Fusion committee shows a monotone spread between cell-local strategies (`most_frequent`, `voting`, per-attribute trust) and entity-aware strategies (per-source trust + per-(source, entity) provenance reasoning, or LLM arbitration) that widens easy → medium → hard, matching the *Committee expectations* table on this card.
