# S1 Variant Generation — products tracker

Compact tracker for bringing the **products** domain up to the same stage as music/games/companies in [plan_s1_final.md](plan_s1_final.md). Runs in parallel with the music/games R6+R7 work — products does not share any cache, baseline, or committee SHA with the other three, so the two tracks are independent.

## Conventions

Inherited verbatim from [plan_s1_final.md](plan_s1_final.md) §Conventions: `pydi-dev/`, no emojis, NumPy-style docstrings, mypy strict, preserve `DataFrame.attrs`, no edits to `PyDI/`, train/val/test naming, K2/K4 default `gpt-5.4-mini` (no bump — user directive 2026-05-15), committee tuning convention. New conventions specific to products:

- **`usecases/products/` is OFF-LIMITS for the synthetic pipeline.** The upstream products notebook workflow lives under `usecases/products/` and is never touched. All synthetic-side data + variants live under `usecases_synthetic/usecases/products/`, opt-in via the `data_root: usecases_synthetic/usecases` field in [config/domains/products.yaml](../../usecases_synthetic/config/domains/products.yaml). Path resolution is centralized in [domain_config.data_root_for_domain](../../usecases_synthetic/lib/domain_config.py) — returns the override Path when set, else `None` (callers fall back to the module-level `USECASES_DIR`, which test fixtures can monkeypatch).
- **Source IDs are source-prefixed strings on the synthetic side ONLY.** The upstream `usecases/products/input/data/products_<n>.json` keeps its native bare-int `id` field. The synthetic-side copy at `usecases_synthetic/usecases/products/input/data/products_<n>.json` carries `id = "products_<n>_<original_int>"`. The rewrite is performed by [scripts/_rewrite_products_ids.py](../../usecases_synthetic/scripts/_rewrite_products_ids.py), which reads upstream and writes synthetic-side — idempotent, never mutates `usecases/products/`.
- **EM gold layout differs on the synthetic side.** Upstream: `usecases/products/input/entity_matching_gt/prod<N>_to_prod<M>_<split>.csv` with headers + `0`/`1` labels. Synthetic-side: `usecases_synthetic/usecases/products/input/entitymatching/products_<N>_2_products_<M>_<split>.csv` headerless with `false`/`true` labels — matching the music/games/companies convention so `read_em_gold_csv` works unchanged.
- **Pool built from `cluster_id`**, not from PLM/rule-based pipelines. Products has no ADI/`automatic-data-integration` outputs and no rule-based correspondences. Cross-source membership in the same `cluster_id` is the pool's positive signal; `pool_agreement` is the count of distinct sources sharing the cluster (max 4). [scripts/build_pool_products.py](../../usecases_synthetic/scripts/build_pool_products.py) — separate from the general `build_pool.py`. Egregious-cluster cap reuses the same `max(ceil(P99 of component sizes), 3*n_sources)` rule.
- **Fusion gold is XML on the synthetic side.** Upstream ships `fusion_validation_set.csv` / `fusion_test_set.csv` as flat CSVs (one row per matched cluster). The synthetic-side variant lives at `usecases_synthetic/usecases/products/input/fusion/validation_set.xml` and `test_set.xml`, authored from the upstream CSVs by [scripts/_author_products_fusion_xml.py](../../usecases_synthetic/scripts/_author_products_fusion_xml.py). The XML format matches the music/games/companies element-tag aggregator convention (record element = `<product>`).

## Prior state (load-bearing summary)

Products was previously **descoped** from plan_s1_final.md (see §"What this plan retires" line: "Movies + products variant generation → descoped to a future plan; pool prerequisites unchanged"). All skeleton YAMLs already exist but were never validated against running code:

- [config/domains/products.yaml](../../usecases_synthetic/config/domains/products.yaml) — skeleton, references `products_<n>.json` and `cluster_id` linkage.
- [config/knob_01_surface/products.yaml](../../usecases_synthetic/config/knob_01_surface/products.yaml) … [config/knob_10_reliability/products.yaml](../../usecases_synthetic/config/knob_10_reliability/products.yaml) — all eight per-knob skeletons.
- [config/committees/em_blocking_committee_products.yaml](../../usecases_synthetic/config/committees/em_blocking_committee_products.yaml), `em_matching_committee_products.yaml`, `fusion_committee_products.yaml` — present.
- **Missing:** `normalization_committee_products.yaml`, products entry in [protection._DEFAULT_KIND_BY_DOMAIN_ATTR](../../usecases_synthetic/lib/protection.py), Ditto + sc_block checkpoints, SM gold mapping CSV, canonical-format fusion XML, products pool, products entry in `build_pool.py` (intentionally — see P0.6 — separate script).

## Source profile

| Source | Records | Format | Native `id` | Notes |
|---|---:|---|---|---|
| products_1 | 812 | JSON | int (`12198483` etc.) | Anchor for EM gold pairs. |
| products_2 | 812 | JSON | int | |
| products_3 | 762 | JSON | int | |
| products_4 | 626 | JSON | int | |
| **Total** | **3012** | | | 812 clusters; every cluster spans 2-4 sources, **no singletons**. |

Cluster span distribution: 626 clusters span all 4 sources, 136 span 3, 50 span 2. EM gold pairs with `label=1` always share `cluster_id` (verified on `prod1_to_prod2_train` head-50). EM gold pairs (anchor = products_1, 3 pairs):

| Pair | All | Train | Val | Test |
|---|---:|---:|---:|---:|
| products_1 ↔ products_2 | 3376 | 1831 | 175 | 180 |
| products_1 ↔ products_3 | 2979 | 1599 | 153 | 163 |
| products_1 ↔ products_4 | 2444 | 1326 | 140 | 121 |

Canonical attributes per [config/domains/products.yaml](../../usecases_synthetic/config/domains/products.yaml):
- `title` (primary, long_string)
- `brand` (key, nominal)
- `description` (secondary, free_text)
- `price` (secondary, continuous)
- `priceCurrency` (secondary, nominal)

Schema is shared verbatim across all four sources — no column renames needed in any committee YAML (`column_mapping: {}` everywhere). The target schema also declares optional category-specific fields (`product_type`, `chipset_name`, `vram_gb`, …) that are absent from most rows in the four product files; these stay out of scope of the synthetic pipeline for now (knobs target `title / brand / description / price / priceCurrency` only).

## Phase P0 — Data layout + infrastructure prerequisites

Required before S.1 can run. Each item lands in its own commit (or commit group) per the human-owns-git convention.

| # | Item | Status |
|---|---|---|
| P0.0 | Add `data_root` override mechanism + `data_root_for_domain()` helper in [domain_config.py](../../usecases_synthetic/lib/domain_config.py); update path-resolution call sites in [loaders.py](../../usecases_synthetic/lib/loaders.py), [protection.py](../../usecases_synthetic/lib/protection.py), [variant_loader.py](../../usecases_synthetic/lib/variant_loader.py), [package_variant.py](../../usecases_synthetic/scripts/package_variant.py), [apply_knob_04_coverage.py](../../usecases_synthetic/scripts/apply_knob_04_coverage.py), [downsample_domain.py](../../usecases_synthetic/scripts/downsample_domain.py). Returns `None` for non-overriding domains so the existing `USECASES_DIR` monkeypatch fixtures keep working. | `[x]` (2026-05-15) |
| P0.1 | Author EM gold under the synthetic-side data_root at `usecases_synthetic/usecases/products/input/entitymatching/products_<N>_2_products_<M>_<split>.csv`. Headerless, lowercase boolean labels — matching the music/games/companies convention so `read_em_gold_csv` works unchanged. Upstream `usecases/products/input/entity_matching_gt/` is read-only. Helper: [scripts/_rewrite_products_ids.py](../../usecases_synthetic/scripts/_rewrite_products_ids.py). | `[x]` (2026-05-15) |
| P0.2 | Author source JSON under the synthetic-side data_root at `usecases_synthetic/usecases/products/input/data/products_<n>.json` with `id = "products_<n>_<original_int>"`. Upstream files keep their native bare-int `id`. Same helper as P0.1. | `[x]` (2026-05-15) — 3012 ids written. |
| P0.3 | EM gold ID rewrite: every `id1`/`id2` in the synthetic-side EM gold uses the `products_<n>_<int>` prefix matching P0.2. Same helper. Verified end-to-end: `protection._load_em_gold_ids("products")` returns 3012 unique IDs (= source row total). | `[x]` (2026-05-15) |
| P0.4 | Authored synthetic-side [`usecases_synthetic/usecases/products/input/schemamatching/sm_mapping_gold.csv`](../../usecases_synthetic/usecases/products/input/schemamatching/sm_mapping_gold.csv) — 24 rows (4 sources × 6 columns including id). The original target schema is also mirrored from upstream (`scripts/_rewrite_products_ids.py` Phase 3). | `[x]` (2026-05-15) |
| P0.5 | Convert upstream fusion CSVs → canonical XML at the synthetic side. Helper: [scripts/_author_products_fusion_xml.py](../../usecases_synthetic/scripts/_author_products_fusion_xml.py). Provenance policy: `<id_left>+<id_right>` union per cell (no per-attribute attribution available in the CSV). `<id>` element = `id_left` (products_1 anchor). XML loads cleanly via `load_xml(...)`: shape (100, 11) per file. Upstream `usecases/products/input/fusion/` is read-only. | `[x]` (2026-05-15) — both validation_set.xml + test_set.xml written under `usecases_synthetic/usecases/products/input/fusion/`. |
| P0.6 | Built products pool from `cluster_id` via [scripts/build_pool_products.py](../../usecases_synthetic/scripts/build_pool_products.py). Pool size: 4214 rows across 6 source-pair combinations (812 / 762 / 626 / 762 / 626 / 626). Egregious cap = 12 (P99=4, floor=3×4=12 sources); 0 clusters dropped. Stats at [pools/products/pool_stats.json](../../usecases_synthetic/pools/products/pool_stats.json). The pool loader honors the `data_root` override transparently via `load_domain_sources`. | `[x]` (2026-05-15) |
| P0.7 | Added `"products"` entry to [protection._DEFAULT_KIND_BY_DOMAIN_ATTR](../../usecases_synthetic/lib/protection.py) (`title=long_string, brand=nominal, description=free_text, price=continuous, priceCurrency=nominal`). Verified `kind_map_for_domain("products")` returns the new map. | `[x]` (2026-05-15) |
| P0.8 | Authored [config/committees/normalization_committee_products.yaml](../../usecases_synthetic/config/committees/normalization_committee_products.yaml). Members: text_clean (title, brand, description, priceCurrency), number_locale (price), llm_canonicalize (title, brand, priceCurrency). Intentionally omitted: date_iso, country_iso, taxonomy_lookup. | `[x]` (2026-05-15) |
| P0.9 | [config/domains/products.yaml](../../usecases_synthetic/config/domains/products.yaml) updated: `data_root: usecases_synthetic/usecases` override added; `fusion_files: validation_set.xml/test_set.xml`. End-to-end smoke verified: `cfg.em_dir()` → `usecases_synthetic/usecases/products/input/entitymatching/`, all 4 sources load with prefixed IDs, EM gold IDs total 3012. | `[x]` (2026-05-15) |
| P0.10 | Trained per-domain Ditto checkpoint for products → `usecases_synthetic/cache/ditto_checkpoints/products/best`. Data prep helper [scripts/ditto/_prep_products.py](../../usecases_synthetic/scripts/ditto/_prep_products.py) mixed all 3 source pairs into 4751 train / 468 val / 464 test pairs (synthetic-side EM gold, no upstream reads). Run: `roberta-base --lr 1e-5 --batch-size 32 --max-len 256 --max-field-len 350 --epochs 50 --early-stopping-patience 5 --da del --no-fp16 --fixed-threshold 0.5 --seed 42`. **Test F1 = 0.959** (P=0.987, R=0.932), best_epoch=3, best_val_f1=0.961. Symlink: `usecases_synthetic/cache/ditto_checkpoints/products/best` → `sweep_lr_1e-5_unweighted/run_20260515_152452/checkpoints/best`. DittoMatcher adapter smoke-test passed (2/9 cross-product candidates matched above θ=0.5, both real cluster_id positives). | `[x]` (2026-05-15) — well above F1≥0.85 acceptance. |
| P0.11 | (optional, deferred) Train sc_block checkpoint → `cache/sc_block_checkpoints/products/best`. `enabled_by_default: false` in `em_blocking_committee_products.yaml`. | `[ ]` (deferred) |
| P0.12 | Per-knob YAML calibration pass: `pool_quality: deferred` → `live` in `knob_02_niche/products.yaml`; skeleton/WIP commentary removed from `knob_01_surface/products.yaml`, `em_blocking_committee_products.yaml`, `em_matching_committee_products.yaml`. Per-level rate calibration deferred to S.5a. | `[x]` (2026-05-15, structural pass) |

## Phase S — `products-small` sanity ladder

Mirrors [plan_s1_final.md](plan_s1_final.md) §S exactly. Each step blocks the next.

| # | Module | Status |
|---|---|---|
| S.1 | Delete stale `usecases/products-small/` + `usecases/products-small-augmented/` if they exist. (First run on a fresh products: skip.) | `[ ]` |
| S.2 | `downsample_domain.py --source-domain products --target-domain products-small --gold-multiplier 1.5 --min-rows 50 --seed 0`. Products is small (3012 rows) so the downsample will retain ~all of it — fine, the goal of `*-small` is fast iteration, and products is already fast. Resulting YAML carries `knob_config_alias: products`. | `[ ]` |
| S.3 | `measure_baseline.py --domain products-small --with-llm`. Total runtime ~6 min on M5 Max. Headline: **SM macro_f1=0.814**, **Norm macro_f1=0.259** (best-member 0.394), **EM blocking macro_pair_recall=0.798** (best bm25 recall=0.977, rr=0.931), **EM matching macro_f1=0.600** (best ditto_plm f1=0.702), **Fusion overall_accuracy=0.776**. Wired fixes during this step: (1) `committee_paths._COMMITTEE_DOMAIN_ALIASES` now resolves `products-small` → `products`; (2) `em_matching_committee_products.yaml` `demonstrations_path` + `training_gold_path` repointed to synthetic-side; (3) `matchgpt_em_matcher._load_demonstrations` now sniffs header presence (mirrors the magellan helper) so it accepts the headerless EM gold convention used across all 4 domains. | `[x]` (2026-05-15) — [baselines/products-small/baseline_report.md](../../usecases_synthetic/baselines/products-small/baseline_report.md). |
| S.4 | `generate_variant.py --domain products-small --level all`. K2/K4 default `gpt-5.4-mini`; bump per the conventions if user approves. | `[ ]` |
| S.5 | Spot-check the 3 variant directories. **Results (2026-05-15):** (a) data files present + valid; (b) schemamatching/entitymatching/fusion dirs populated (SM=2, EM=21 including regen splits, fusion=2 XML); (c) provenance=17 + baselines=6 CSVs per level; (d) `config/difficulty.yaml` uses `level: easy/medium/hard` (matches convention); (e) regen EM gold (`*_regenerated.csv`) resolves 0/11376 unresolved IDs across easy+medium and 0/7844 across hard. **Hard variant fusion gold** has 168/200 unresolved entity IDs + 840/2000 unresolved provenance — exposes a cross-domain K4 fusion-protection bug (see F-P1 below). | `[x]` (2026-05-15, with F-P1 caveat) |
| S.5a | EM regen contract per pair × split. **Easy + medium GREEN**: every pair × split has size_drift=+0.0%, pos_ratio_drift=+0.0pp, all surviving pairs preserved (~1829/1829 for the larger train splits; the 2-row gap is intra-split deduplication), cross-split disjoint. **Hard yellow (F-P1 carryover)**: size_drift -27% to -58% and pos_ratio drift -15pp to -40pp on all 9 hard splits. Root cause is the K4 fusion-protection bug (F-P1) — K4 hard removed 627/812 products_1 anchor rows, so K2's regen-positive pool collapses. Cross-split disjointness still OK on hard. Will recover once F-P1 lands. | `[x]` (2026-05-15) — easy+medium green, hard yellow on F-P1. |
| S.6 | n/a — only one `*-small` domain in this plan. | `[ — ]` |
| S.6a | Per-committee member-level audit. **0 silent zeros** across all 5 committees. Per-member means: **SM** (7 members, 24 attrs each): duplicate_majority=1.000, llm_openai=1.000, coma_hybrid=1.000, label_jw=0.833, embedding_sbert=0.792, magneto_slm_llm=0.667, instance_tf_cosine=0.458 — 30 cell-level zeros are real attribute-class mismatches (e.g. instance_tf_cosine doesn't match on numeric columns), not silent failures. **Norm** (3 members): text_clean=0.395, number_locale=0.311, llm_canonicalize=0.070 — llm_canonicalize is low (LLM responses for product titles/brands don't align with the fusion-gold spelling under the long_string Levenshtein closeness gate); follow-up item but not a silent zero. **EM blocking** (5 members): all members non-zero, best=bm25 (0.977 pair_recall). **EM matching** (3 members): all non-zero, best=ditto_plm (0.702). **Fusion** (per-attr per-strategy): 18 strategies × 5 attrs, 0 zeros. Best-per-attr: title=0.84 (longest_string), brand=0.79 (most_complete), description=0.69 (longest_string), price=0.55 (median / prefer_higher_trust), priceCurrency=1.000 (prefer_higher_trust). sc_block correctly deferred (P0.11), reports as disabled. | `[x]` (2026-05-15) |
| S.6b | **Committee-F1 monotonicity** (validate_variant.py × 3 levels) followed by `analyze_monotonicity.py --domain products-small` (P8 best-member-F1 check). | `[x]` (2026-05-15). All 3 validates ran in 10:29 / 11:41 / 8:31 (post-split-runner-refactor in plan_s1_final.md). **Committee macro** (baseline → easy → medium → hard): SM 0.81 → 0.81 → 0.77 → 0.72; Norm 0.26 → 0.30 → 0.19 → 0.16; EM-block 0.80 → 0.80 → 0.80 → 0.17 (crash at hard); EM-match 0.60 → 0.61 → 0.60 → 0.60; Fusion 0.78 → 0.76 → 0.67 → 0.64. **P8 best-member ceiling** (winner per level → hard-vs-baseline Δ): SM=duplicate_majority 1.00→0.98 (-0.02); Norm=text_clean 0.39→0.31 (-0.09, non-monotonic via easy bump); EM-block=token_blocker 0.997→0.169 (**-0.83**, hard crash); EM-match=ditto_plm / llm_matcher 0.70→0.72 (+0.02, ceiling rises slightly); Fusion=title_longest_string 0.67→0.64 (-0.03, non-monotone). Hard EM-block crash mirrors music + companies (K1/K3 corrupting blocking keys). EM-matching ceiling is robust (ditto_plm holds). Fusion ceiling moves only -0.03 because thin val/test + F-P1 carryover. 0 collapses. Reports at [validation/products-small/](../../usecases_synthetic/validation/products-small/). |

## Phase R — Full-domain products

Only after S green. Mirrors [plan_s1_final.md](plan_s1_final.md) §R6 + §R7.

| # | Module | Status |
|---|---|---|
| R6.1 | `measure_baseline.py --domain products`. Records committee SHAs into `baselines/products/baseline_metrics.json`. Drift-guard reference for R7.1. | `[ ]` |
| R6.2 | `generate_variant.py --domain products --level all`. Variants land at `usecases/products-augmented/{easy,medium,hard}/`. | `[ ]` |
| R7.1 | `validate_variant.py --domain products --level <l>` for the 3 levels. SHA-pinned freeze guard catches drift. | `[ ]` |
| R7.2 | `analyze_monotonicity.py --domain products`. Must include best-member-F1 monotonicity (P8 from plan_s1_final.md). | `[ ]` |
| R7.3 | `validation/products/final_report.md` per template. | `[ ]` |
| R7.4 | (optional) ablation sweep on unclear-signal knobs. | `[ ]` |

## Per-domain specifics

- **4 sources is the most of any domain** (companies/games/music have 3, products has 4). Source-pair count stays at 3 (anchor = products_1) — products_2/3/4 are not paired against each other in the EM gold today. Open: should the plan add the missing pairs (`products_2 ↔ 3`, `2 ↔ 4`, `3 ↔ 4`)? Decision deferred to after S.5a — the existing 3-pair gold gives full cluster coverage via transitivity through products_1.
- **Total record count is small** (3012). K2 corner-ratio targets calibrated for ~50k-row sources may behave differently; expect S.4 to surface "under-target after dispatcher" warnings analogous to music's K2 hard drift (F4 in plan_s1_final.md). Initial `interp_pair_factor` in [config/knob_02_niche/products.yaml](../../usecases_synthetic/config/knob_02_niche/products.yaml) defaults to 0.5; tune at S.5a if K2 hard under-shoots.
- **Cluster_id is dense gold.** Unlike companies (incomplete EM gold) or music (no `cluster_id` linkage), every products row has cluster membership. This means the S.5a hard-level structural pos_ratio drift seen on music + companies may NOT appear on products, because the pool can backfill regen positives from cluster_id without re-introducing K2-removed entities. Verify empirically at S.5a.
- **Currency normalization gap.** No `CurrencyIsoNormalizer` exists in [normalizer_members.py](../../usecases_synthetic/lib/normalizer_members.py). P0.8 ships `priceCurrency` via `text_clean + llm_canonicalize` only. If S.6a flags `priceCurrency` closeness as too low, add a `CurrencyIsoNormalizer` member as a follow-up (not a blocker).
- **No taxonomy.** Products has no GICS/genre-style controlled vocabulary, so `TaxonomyLookupNormalizer` is intentionally omitted from `normalization_committee_products.yaml`. Brand could in principle be normalized against a brand taxonomy but none is shipped — out of scope.
- **Ditto field selection.** [config/committees/em_matching_committee_products.yaml](../../usecases_synthetic/config/committees/em_matching_committee_products.yaml) declares `fields: [title, brand, description, price]`. `priceCurrency` is intentionally excluded (3-letter currency codes carry no entity-discrimination signal). `description` is the longest field by far — Ditto's `max_field_len` cap (350 in K2 hard config) will truncate it; that's expected and matches music/games conventions.

## Cross-domain findings (need triage in plan_s1_final.md)

| # | Issue | Site |
|---|---|---|
| F-P1 | **K4 fusion-protection comparison is broken across all 4 domains.** [coverage_ops.py:707-708](../../usecases_synthetic/lib/coverage_ops.py#L707-L708) compares `entity_id` (which is the canonical-frame key `k02_ent_NNNNNN` generated by [apply_knob_02_niche.py:484](../../usecases_synthetic/scripts/apply_knob_02_niche.py#L484)) against `constraints.fusion_val_test_ids` (which is a set of source-record IDs read from the fusion XML's `<id>` text, e.g. `products_1_12198483` / `mbrainz_6891`). These two namespaces never overlap, so the `is_fusion_protected` branch is never taken — K4 silently treats fusion-protected entities as unprotected and applies the conflict-preserving sort path. The bug doesn't manifest on music/games/companies today because K4 hard demotions on those domains rarely target the primary source where fusion `<id>`s live. On products it surfaces immediately: K4 hard removed 627/812 (77%) of `products_1` rows, leaving 168/200 fusion test entities + 840/2000 provenance ids without backing source rows. Fix candidates: (a) translate fusion `<id>` text → canonical `entity_id` via `id_to_canonical` at the call site, or (b) rebuild `fusion_val_test_ids` as a set of source-record `(source, record_id)` tuples and compare against `members[entity_id]`. Probably should land in plan_s1_final.md as an F-row before R7. Affects products S.5(e) hard but does not block S.5a / S.6a since regen EM uses post-K4 IDs. |

## Hard blockers + open decisions

- **P0.5 fusion XML re-authoring is the highest-risk P0 item** because the current CSV layout encodes both source IDs (`id_left`, `id_right`) and the fused values in flat columns; the XML format expects one fused entity with provenance attribution per cell. Decide at P0.5 authoring time how to split mixed-source attribute values across provenance tags (e.g. when a fusion-gold row's `title` could come from either left or right — the CSV doesn't say). Likely policy: tag with the union of `id_left+id_right` for now and tighten later if monotonicity surfaces issues.
- **P0.10 Ditto training is GPU-bound and slow.** ~hours on GPU, ~10+ min CPU smoke. Schedule after the rest of P0 lands so the training data (post-P0.3 rewrite) is final before the run.
- **R7.2 best-member-F1 check (P8 from plan_s1_final.md)** must land in `analyze_monotonicity.py` before R7 verdict is final on products — same constraint as music/games/companies.
- **Movies remains descoped.** This plan is products-only. Movies still has no PLM pool and stays out of scope per plan_s1_scale.md.

## Central reporting

Products' XLSX live alongside music / games / companies at [usecases_synthetic/statistics/products.xlsx](../../usecases_synthetic/statistics/products.xlsx). Three sheets (sizes / committee_summary / per_member). Regenerate via `python usecases_synthetic/scripts/build_statistics.py --domain products` after any new baseline / variant / validate run. See [statistics/README.md](../../usecases_synthetic/statistics/README.md) and [scripts/build_statistics.py](../../usecases_synthetic/scripts/build_statistics.py).

## Pick-up instructions

1. Land P0.1 + P0.2 + P0.3 together (one logical commit) — they're tightly coupled (ID rewrite cascade).
2. Land P0.4 + P0.7 + P0.9 (small file edits + new SM gold + protection.py + domain YAML).
3. Land P0.5 (fusion XML re-authoring) and verify the loader can ingest the new XML via a manual `load_xml(...)` smoke test.
4. Land P0.6 (pool builder) and inspect `pool_stats.json` for plausibility.
5. Land P0.8 (normalization committee YAML).
6. Land P0.12 (per-knob YAML calibration pass).
7. **Pause** for user sign-off on the training plan, then launch P0.10 (Ditto training).
8. Once Ditto best/ symlink exists: run S.2 → S.3 → S.4 → S.5 → S.5a → S.6a per the ladder above.
9. After S green, propose R6.1 / R6.2 / R7 alongside (or after) the music/games full-domain pass.

## What this plan retires

- The "products variant generation descoped" entry in [plan_s1_final.md](plan_s1_final.md) §"What this plan retires". Once products has its own tracker (this file), the line in plan_s1_final.md is replaced with a pointer here.
