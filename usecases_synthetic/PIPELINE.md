# usecases_synthetic — Pipeline manifest

Ordered list of scripts that together reproduce the synthetic use cases. Doubles as an execution runbook and as context-reset memory — anyone (including a fresh Claude session) should be able to read this file and know what's been built, what runs when, what depends on what, and what's still TODO.

**Keep this file current.** When adding or removing a script, update the corresponding entry here in the same commit. When a script changes its inputs/outputs, update the entry. When finishing a TODO entry, mark it done and fill in the actual paths.

Status legend: `[done]` — exists and verified · `[wip]` — partially written · `[todo]` — planned, not started

For the design rationale behind each stage, see [../plan.md](../plan.md) and [../knobs/](../knobs/).

---

## Phase 0 — Pool construction

### 1. `scripts/build_pool.py` `[done]`

Merges two existing matcher pipelines into a "pooled positives" set per domain, used downstream as a protection set during Knob 2 augmentation (never as replacement gold — see [../knobs/cross_cutting.md](../knobs/cross_cutting.md#gold-standard-incompleteness-and-pooling)).

**Inputs**
- `automatic-data-integration/scripts/output/<domain>_0302/entity_resolution/matching/correspondences_*.csv` — PLM pipeline, per-pair files (avoid `correspondences_all.csv`; it drops pair directions for companies)
- `usecases_new/output/<domain>/cluster_analysis/detailed_cluster_info.json` — rule-based pipeline; only the raw pairwise edges inside the `correspondences` array of each cluster are consumed (the pre-computed cluster structure is ignored and components are rebuilt from scratch)

**Outputs**
- `pools/<domain>/pooled_positives.csv` — columns `id1, id2, source_1, source_2, pool_agreement`
- `pools/<domain>/pool_stats.json` — source counts, overlap breakdown, egregious-cluster filter telemetry

**Covered domains:** companies, games, music.
**Not covered:** movies (no PLM pool).
**Products** uses a separate pool builder ([scripts/build_pool_products.py](scripts/build_pool_products.py), tracked in [../plans/plan_s1_products.md](../plans/plan_s1_products.md)) that derives the pool directly from each record's `cluster_id` rather than from PLM/rule-based pipelines.

**Key decisions baked in**
- **PLM is the trusted base; rule-based is corroboration only.** The rule-based matcher (hand-weighted `StringComparator` linear combos) is materially weaker than the learned PLM matcher, so rule-based-only pairs are dropped. Every pool row is a PLM pair; rule-based participation only upgrades `pool_agreement` from 1 to 2.
- **Components are rebuilt from raw edges on both sides.** The rule-based cluster JSON's pre-computed clustering is ignored; we extract the raw pairwise edges, union with PLM edges, and run `nx.connected_components` ourselves. This gives uniform cluster semantics across both sources and lets us apply a single cluster-size filter on the regenerated components.
- **Egregious-cluster filter is data-driven, not hand-set.** Cap = `max(ceil(P99 of observed component sizes), 3 * n_sources)`. The P99 adapts to each domain's distribution; the `3 * n_sources` structural floor reflects the upper bound on plausible cross-source duplicate clusters (n_sources members plus slack for in-source duplicates). Only the most egregious transitive-chain artifacts (e.g. the companies 117-entity "Chinese companies" cluster) get cut.
- **Coverage gap on the rule-based side:** `detailed_cluster_info.json` only contains correspondences for one directed source pair per domain (forbes↔dbpedia for companies, metacritic↔dbpedia for games/music). Under the corroboration-only policy this is tolerable — partial coverage just means fewer PLM pairs get `pool_agreement=2`; no PLM pair is lost.
- Same-source pairs are dropped during normalization.
- Pair IDs are ordered by source label so directional duplicates dedupe.

**Run**
```bash
source pydi-dev/bin/activate
python usecases_synthetic/scripts/build_pool.py --all          # all three domains
python usecases_synthetic/scripts/build_pool.py --domain games # one domain
```

**Current pool sizes (v2, 2026-04-18)**
| Domain | Pool size | Both-source agreement | Egregious cap applied | Largest dropped component |
|---|---|---|---|---|
| companies | 2225 | 490 (22%) | 9 (P99=7, floor=9) | 13 |
| games | 13795 | 4659 (34%) | 12 (P99=12, floor=9) | 31 |
| music | 7355 | 3569 (49%) | 9 (P99=9, floor=9) | 55 |

---

## Phase 0.5 — Domain downsampling (optional)

### `scripts/downsample_domain.py` `[done]`

Produces a reduced clone of an existing domain under a new name (e.g. `companies` → `companies-small`) so that expensive multi-knob ablation matrices (Phase 3) can be exercised in minutes instead of hours. The new domain reuses the source domain's per-knob YAMLs via `knob_config_alias` in the target domain YAML, so no knob configs are duplicated. Per-knob values can still be tuned on the small domain without forking the alias via a `knob_config_overrides: {knob_NN_<name>: {key: value, ...}}` block in the target domain YAML — values deep-merge onto the aliased knob config. See [lib/domain_config.py](lib/domain_config.py) `load_knob_config` and `_resolve_knob_config_overrides`. Example: `companies-small` overrides K3's `single_source_survivor_cap_hard` to 0.15 (default 0.25 was too loose after downsampling).

**Protection policy**
- Every ID referenced by any EM gold CSV (all splits, both columns) is preserved.
- Every ID referenced by the fusion gold XMLs is preserved — both the top-level `<id>` text and each `provenance=`-split source ID.
- Additional non-gold rows are sampled deterministically up to `gold_multiplier * |protected|` per source (subject to `--min-rows` floor and optional `--max-rows` cap).

**Inputs / outputs**
- Reads: `usecases/<source_domain>/input/{data,entitymatching,fusion,schemamatching}/`
- Writes: `usecases/<target_domain>/input/...` (downsampled data files, verbatim gold) and `usecases_synthetic/config/domains/<target_domain>.yaml` with `knob_config_alias: <source_domain>`

**Run**
```bash
source pydi-dev/bin/activate
python usecases_synthetic/scripts/downsample_domain.py --source-domain companies --target-domain companies-small --gold-multiplier 1.5 --seed 0
```

**Current small domains**
| Target | Source | dbpedia | forbes | fullcontact |
|---|---|---|---|---|
| companies-small | companies | 10092 → 1083 | 2000 → 2000 | 1931 → 1931 |

Forbes and fullcontact barely shrink because both serve as anchors in EM gold pairs that cover ~75% of their rows. Dbpedia is the main win (~9× reduction).

---

## Phase 1 — Baseline measurement `[done]`

**Goal:** establish per-domain / per-stage committee baselines on the *original* data, before any augmentation. This is the reference point the committee-validated augmentation loop subtracts from to detect "difficulty deltas" (see [../knobs/cross_cutting.md](../knobs/cross_cutting.md#committee-validated-augmentation), §Bootstrap order).

**Committees implemented:** SM (5 members), EM (4 members + pool diagnostic), Fusion (per-attribute-type routing). See [config/committees/](config/committees/) and [module_01_committee_spec.md](../plans/validation/module_01_committee_spec.md).

**Scripts**
- `scripts/measure_baseline.py [done]` — runs SM/EM/fusion committees on each domain's original data, writes `baselines/<domain>/baseline_metrics.json` + `baseline_report.md`.

**Covered domains:** `companies` and `companies-small` (baselines at [baselines/companies/](baselines/companies/), [baselines/companies-small/](baselines/companies-small/)).
**Deferred:** games, music, movies, products (to plan.md Step 9 Scale).

---

## Phase 2 — Scenario 1 (augmented use cases) `[done]`

**Goal:** apply the knobs to the original use cases in the canonical order defined in [../knobs/README.md](../knobs/README.md#canonical-knob-application-order) to produce `easy`/`medium`/`hard` variants.

**Canonical order (S1):**
`Knob 2 (niche density) → Knob 4 (coverage skew) → Knobs 1/5/6 (value perturbations, jointly per cell) → Knob 3 (attribute drop) → Knob 10 (reliability reshuffle) → Knob 8 (header rename)`

**Per-knob scripts — all implemented.** Each exposes a pure `apply_knob_XX(...)` entry point plus a CLI for standalone runs; the master orchestrator (`generate_variant.py`) calls the pure entry points in canonical order.

- `scripts/apply_knob_02_niche.py [done]` — consumes pooled positives as protection floor; removes or interpolates entities toward the target `corner_case_ratio`. First knob applied.
- `scripts/apply_knob_04_coverage.py [done]` — per-entity source coverage skew, takes Knob 2's placements as fixed.
- `scripts/apply_values_joint.py [done]` — Knobs 1, 5, 6 applied jointly per cell with collision-index coordination to preserve conflict-preservation constraints.
- `scripts/apply_knob_03_drop.py [done]` — per-source attribute drop, runs after value perturbations so drops happen on perturbed data. Cross-level nesting (`D_easy ⊆ D_medium ⊆ D_hard`) is enforced structurally: all three drop masks are computed in one pass, constraints (fusion floor, conflict preserve, single-source cap) are applied at each level, then `easy`/`medium` are shrunk against `hard` via `_enforce_nesting`. Propagate-fill runs after nesting so filled cells are never re-dropped.
- `scripts/apply_knob_10_reliability.py [done]` — source reliability reshuffle via fusion-gold realignment, no raw gold change.
- `scripts/apply_knob_08_naming.py [done]` — header-only schema naming divergence, orthogonal, runs last.

**Orchestrator + packaging**
- `scripts/generate_variant.py [done]` — master CLI. Runs all eight knobs in canonical S1 order for a `(domain, level)`, consolidates provenance, writes `difficulty.yaml`, and (when run with `--level all`) produces a cross-level monotonicity audit covering K3 drop nesting plus per-knob intensity checks for K2/K4/K5/K6/K8/K10. K4 audit skips levels whose `target_coverage_histogram` is `null` (identity by design, not a failure). K8 audit uses summed `rapidfuzz.distance.Levenshtein` over provenance rows (column: `knob_08_naming_edit_distance`) rather than row counts, which were confounded by upstream K3 drops. K1/K2 LLM calls are served from the content-hash caches under `cache/` (`knob_01_paraphrases/`, `knob_02_interpolations/`, `knob_02_non_corner/`, `llm_cache/`), which are committed to git so runs reproduce exactly without re-calling the API. The orchestrator does **not** enable strict-cache: `generate_variant()` defaults both `strict_cache_k1`/`strict_cache_k2` to `False` (never fail on a miss) and exposes no strict-cache CLI flag. On a cache miss it degrades gracefully — deterministic operators / blender when no `OPENAI_API_KEY` is set, or a live (hence non-deterministic) OpenAI call that repopulates the cache when a key is present. (Only the standalone `apply_values_joint.py` CLI still forces strict-cache at `hard` level — `apply_values_joint.py:487` — which would hard-error on a miss; the orchestrator never takes that path.)
- `scripts/package_variant.py [done]` — assembles the per-level variant directory under `usecases/<domain>-augmented/<level>/` with `input/{data,schemamatching,entitymatching,fusion}`, `baselines/`, `provenance_all.csv`, and `config/difficulty.yaml` matching [../plan.md](../plan.md#scenario-1-augmented-use-cases).

**Run**
```bash
source pydi-dev/bin/activate
python usecases_synthetic/scripts/generate_variant.py --domain companies --level easy
python usecases_synthetic/scripts/generate_variant.py --domain companies --level all   # runs easy+medium+hard and writes monotonicity_audit.csv
```

**Prototype target:** companies first (per [../../plans/plan_s1_implementation.md](../../plans/plan_s1_implementation.md)). Games + music are tracked in [../../plans/plan_s1_final.md](../../plans/plan_s1_final.md) (currently in S.6+); products is tracked separately in [../../plans/plan_s1_products.md](../../plans/plan_s1_products.md) (currently in P0); movies remains descoped.

---

## Phase 3 — Committee validation `[done]` (companies-small; `companies` deferred)

**Status:** end-to-end run completed on `companies-small` on 2026-04-16. Verdict: **qualified pass** — 7/8 knobs show correct direction on primary stage in ablation; SM under K8 is a clean monotone signal. One P0 infrastructure bug (EM/Fusion committee runners break on K8 header renames at medium/hard) plus four P1 S1-knob findings documented in [validation/companies-small/final_report.md](validation/companies-small/final_report.md).

**Full `companies` validation deferred** pending P0 fix.

**Goal:** re-run committees on the augmented variants and compare to Phase 1 baselines. Controlled monotone drops = real difficulty signal. Collapse = back off via the fix-strategy table in [../knobs/cross_cutting.md](../knobs/cross_cutting.md#per-knob-fix-strategy-defaults).

- `scripts/validate_variant.py [done]` — runs SM/EM/Fusion committees against a packaged variant (one `(domain, level)` at a time), loads the baseline, and persists per-level metrics with baseline+delta twins on every leaf, plus a per-pair EM CSV, a per-attribute fusion CSV, and a `level_report.md` rollup. Refuses to run if committee YAML hashes diverge from `baseline_metrics.json`'s recorded versions (belt-and-braces drift guard). Uses the baseline's `fusion_input_member` as-is — no per-variant re-selection. Measurement-only: monotonicity and collapse judgement live in M8 (`[todo]`).
- `scripts/analyze_monotonicity.py [done]` (M8) — cross-level monotonicity + collapse detection. Consumes the per-level `metrics.json` files produced above plus the baseline and writes `validation/<domain>/monotonicity_report.md` + `.csv`.
> **M9 ablation DROPPED (user, 2026-06-05) — not being run.** The two scripts below remain in the repo but are out of scope for the current effort.
- `scripts/run_ablation_validation.py [code-present; DROPPED]` (M9) — per-knob ablation runner. For each active knob (K1/K2/K3/K4/K5/K6/K8/K10), generates a single-knob-hard variant and runs `validate_variant` against it. Writes `validation/<domain>/ablation/knob_<id>/metrics.json`. Expensive: 8 knobs × ~20-40 min each.
- `scripts/analyze_ablation.py [code-present; DROPPED]` (M9) — per-knob ablation analyzer. Consumes baseline, full-hard, and per-knob ablation metrics; writes `validation/<domain>/ablation/ablation_report.md` + `.csv` with per-signal deltas and four interaction flags (cross-stage leakage, primary under/over-signal, direction mismatch).

**Run**
```bash
source pydi-dev/bin/activate
python usecases_synthetic/scripts/validate_variant.py --domain companies --level easy
python usecases_synthetic/scripts/validate_variant.py --domain companies --level medium
python usecases_synthetic/scripts/validate_variant.py --domain companies --level hard
python usecases_synthetic/scripts/validate_variant.py --domain companies --level baseline  # sanity: all deltas should be 0
```

**Outputs**
- `validation/<domain>/<level>/metrics.json` — full per-stage / per-member / per-pair / per-attribute metrics with `_baseline` + `_delta` twins
- `validation/<domain>/<level>/level_report.md` — human-readable stage tables with delta columns
- `validation/<domain>/<level>/em_per_pair.csv` — one row per (member, pair)
- `validation/<domain>/<level>/fusion_per_attribute.csv` — one row per fusion attribute

---

## Phase 4 — Scenario 2 (fully synthetic) `[DROPPED — user, 2026-06-05]`

**DROPPED (user, 2026-06-05) — not being implemented/run.** (Was: Scenario 2, fully synthetic.)

Adds Knob 9 (schema completeness / distractors) at the front of the canonical order. Otherwise the pipeline shape is the same as Phase 2, but seeded from LLM entity interpolation rather than existing records. Out of scope until Scenario 1 prototype lands.

- `scripts/generate_synthetic_seed.py [todo]`
- `scripts/apply_knob_09_schema.py [todo]`
- (remaining knobs reuse Phase 2 scripts against the synthetic seed)

---

## Notes on convention

- All scripts import PyDI the same way tests do — rely on `pydi-dev` virtual env and absolute imports from `PyDI`.
- All scripts should be runnable standalone from the repo root (`python usecases_synthetic/scripts/<name>.py`) — no `cd` required, use `Path(__file__).resolve().parents[N]` to locate the repo root.
- All scripts must preserve `DataFrame.attrs` provenance when transforming frames (CLAUDE.md requirement).
- No emojis in console output (CLAUDE.md requirement).
- NumPy-style docstrings (CLAUDE.md requirement).
