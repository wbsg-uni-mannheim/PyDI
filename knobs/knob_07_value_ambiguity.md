# Knob 7 — Value ambiguity / collision rate

**Status:** DEFERRED (design LOCKED, not built v1). **Scenario:** S1 + S2.

> **Implementer note — SKIP THIS CARD IN v1.** Per [plan_algorithmselection.md](../plan_algorithmselection.md) row 7 and the *v1 implementation status* section below, **zero implementation work is required for Knob 7 in v1**. Everything below this note is reference design for the eventual re-open. Step 6 prototyping must not consume effort here.

## Definition

**Within-entity referential ambiguity only.** Cross-entity label collisions ("different things look the same") are explicitly *not* in Knob 7 — they were moved to Knob 2 as a label-collision distractor-selection signal. Knob 7 *would* stay aligned with its declared dimensions (Value Ambiguity (Norm), Conflict Rate (Fusion), Conflict Subtlety (Fusion)) **once built**; in v1 those dimensions are accepted as under-stressed.

## v1 implementation status — not built

All three current domains have thin Knob 7 substrate (companies strongest, but founders coverage ~10%; games and music near zero). Combined with the cost of the per-attribute ambiguity map, lenient-evaluator probing, and rollback machinery — all for one sub-parameter — **the v1 prototype skips Knob 7**. The dimensions Knob 7 was meant to cover are accepted as **under-stressed in v1**.

**Re-open trigger (single canonical condition).** Knob 7 is re-opened iff the parked accepted-sets discussion lands favorably and brings `multi_sense_conflict_rate` + `polysemy_rate_categorical` back online — at which point Knob 7 grows from one thin sub-parameter to three and the machinery cost becomes proportionate. The earlier mention of "movies / products long-text substrate" as an alternate trigger is folded in here as a *secondary* observation: if movies or products land first, their richer substrate strengthens the case but does not by itself re-open the knob — the accepted-sets decision is the binding gate. This matches [plan_algorithmselection.md](../plan_algorithmselection.md) row 7.

## Sub-parameters

### Active

- `referential_ambiguity_rate` — fraction of (entity, attribute) cells where at least one source is mutated to carry a referentially underspecified value (e.g. `"Republic of Korea"` → `"Korea"`; `"John A. Smith Jr."` → `"John Smith"`). Individually defensible but referentially underspecified relative to the gold.

### Parked (revisit after accepted-sets discussion)

- `multi_sense_conflict_rate` — dropped: would require extending the fusion gold to accepted sets (e.g. `{Rock, Alternative}`), incompatible with the current fusion evaluator.
- `polysemy_rate_categorical` — dropped for the same reason.

## Easy / Medium / Hard *(aspirational — not built in v1)*

| Level | `referential_ambiguity_rate` | Target state | Generator action |
|---|---|---|---|
| **Easy** | 0% | Every value is referentially specific. Disambiguators present where natural language allows them. | Normalize-down: where a source carries a known-ambiguous form, replace with the more specific form propagated from a sibling source or the gold. |
| **Medium** | low | A meaningful tail of cells carry shortened/ambiguous forms; most values remain specific. | Identity / light injection. Most domain baselines likely sit here. |
| **Hard** | moderate | Pervasive shortened forms across key and secondary attributes. Naive fusion strategies that pick the modal value frequently land on the underspecified form. | Per-cell rewrite from specific → ambiguous form, drawing from a per-attribute ambiguity map. |

## Generator mechanism — per-attribute ambiguity map *(aspirational — not built in v1)*

Per-domain artifact mapping specific forms to their ambiguous shortened forms, constructed from actual gold values (not a generic gazetteer). Example for companies:

```yaml
country:
  "Republic of Korea": "Korea"
  "Democratic Republic of the Congo": "Congo"
founders:
  "John A. Smith Jr.": "John Smith"
  "Maria Garcia-Lopez": "Maria Garcia"
```

Small, manually curated per domain during the prototype phase; LLM-assisted later. Cells not covered by the map are not eligible for Knob 7 mutation.

## Composition *(aspirational — not built in v1)*

- **Knob 1 (paraphrase):** orthogonal — Knob 1 makes the same referent look different at the surface; Knob 7 makes the value referentially underspecified.
- **Knob 5 (formats):** orthogonal — Knob 5 reformats a fully-specified value; Knob 7 *removes information* from it.
- **Knob 6 (noise):** distinct — noise produces *wrong* values; Knob 7 produces *insufficient* values.
- **Knob 2:** the new home for cross-entity label collisions.
- **Per-source treatment:** Knob 7 is *not* per-source-shaped (unlike Knobs 3/6). Cell-level injection across the corpus.

## Fusion safety *(aspirational — not built in v1)*

**Bounded by lenient fusion (intended).** Knob 7's reach is limited to value pairs the lenient fusion evaluator already tolerates. On committee collapse: **rollback** the offending injections (no gold mutation). Effective range = "what lenient fusion already accepts."

### Monotonicity guards

1. **Per-cell clean-survivor floor** (mirrors Knobs 3 and 6).
2. **Committee check.** Sharpest drop on naive most-frequent / random-pick fusion strategies.
3. **Rollback on collapse.**

## Committee expectations *(aspirational — not built in v1)*

- **SM / Blocking:** flat / mostly flat.
- **EM:** small monotone drop.
- **Fusion:** primary target. `longest_string` strategy specifically resists this knob — that resistance is the intended discrimination signal.

## Per-domain notes *(aspirational — not built in v1)*

- **Companies:** **non-zero baseline** (DBpedia `England` vs `United Kingdom` co-occur for UK entities). Easy must *normalize-down* — direction flip from Step 2's working assumption. Founders coverage ~10% → effective cell budget ~190 cells.
- **Games:** at easy. Developer/publisher names usually full studio names. Effective surface even thinner than companies.
- **Music:** at easy for the *narrow* sub-parameter. Cross-entity homonyms (`John Williams`, `Crash`) belong to Knob 2 label-collision, not here.

## Provenance *(aspirational — not built in v1)*

`transform_fn=referential_ambiguate`, `transform_params={ambiguity_map_entry, original_form, ambiguous_form}`. Rollback emits mirror `transform_fn=rollback_for_committee`.

## Algorithm selection

**Deferred — not built in v1.** Per the *v1 implementation status* section above and [cross_cutting.md §Per-knob fix-strategy defaults](cross_cutting.md#per-knob-fix-strategy-defaults) (row 7: *"deferred — Knob 7 specced but not built in v1"*), no algorithm selection is performed for Knob 7 at Step 5. The dimensions Knob 7 was meant to cover (Value Ambiguity, Conflict Rate, Conflict Subtlety) are accepted as under-stressed in v1. When Knob 7 is re-opened alongside the accepted-sets discussion — or earlier if movies / products bring a long-text attribute with richer ambiguity substrate — algorithm selection should follow the same tier framework as the other knobs: a deterministic per-attribute ambiguity map (Tier A, authored from gold values as the card already specifies) for the single active `referential_ambiguity_rate` sub-parameter, with no LLM involvement. The cross-entity label-collision signal has already moved to Knob 2 ([knob_02_niche_density.md](knob_02_niche_density.md#metric-set), `label_collision` metric) so that slice of the original Knob 7 scope is live.
