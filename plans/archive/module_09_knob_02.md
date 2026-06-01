# Module 9: Knob 2 — Entity Niche Density

## Purpose

Controls how many similar-but-distinct entities cluster in the same semantic neighborhood. The most architecturally complex knob: multi-metric niche scoring, corner-case pair mining, EM test set regeneration, and LLM entity interpolation at hard. Runs **first** in the canonical order.

**Split across 2-3 sessions:** Session 1 = metrics + RRF scorer. Session 2 = removal + corner-case miner + EM test regeneration. Session 3 = LLM interpolation + placement.

## Spec References

- **Knob card:** [knobs/knob_02_niche_density.md](../../knobs/knob_02_niche_density.md) — full specification including niche metric set (5 metrics), RRF density scorer, corner-case pair miner (recall-biased), LLM interpolation for near-twin entities, placement strategies, per-domain baselines, embedding cache, protection-aware entity selection
- **Cross-cutting provenance:** [knobs/cross_cutting.md](../../knobs/cross_cutting.md) § "Per-value provenance" — entity-scoped provenance for K2 operations: `transform_fn ∈ {remove_entity, interpolate_entity, place_entity}`
- **Protection set:** [knobs/cross_cutting.md](../../knobs/cross_cutting.md) § "Gold standard incompleteness" — entities in `expanded_positives` never dropped, never used as parent seeds for single-source distractor interpolation
- **LLM hygiene:** [knobs/cross_cutting.md](../../knobs/cross_cutting.md) — pinned model, temperature=0, cached + committed, committee gating, contamination spot-check (8-gram overlap, first-token memorization)
- **EM test-set treatment:** [knobs/cross_cutting.md](../../knobs/cross_cutting.md) § "Test-set treatment" — EM test set regenerated per variant to track corner-case ratio, stratified from full post-K2 candidate set. Reported F1 still against original human-annotated gold
- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` — pre-computed, cached at `usecases_synthetic/cache/knob_02_embeddings/<domain>.npy`

## Key Mechanism (from knob card § "Algorithm selection")

### Niche Metrics (5 signals)

1. **Lexical-niche:** Extended Jaccard with typo-robustness on tokenized attribute values
2. **Embedding-niche:** Cosine similarity on `all-MiniLM-L6-v2` embeddings of concatenated record text
3. **Attribute-overlap:** Weighted overlap of categorical/set attributes (genre, platform, industry)
4. **Label-collision:** Entities sharing identical or near-identical primary labels
5. **TF-IDF cosine:** Sparse TF-IDF on concatenated text fields

### RRF Density Scorer

Consensus-biased: `rrf_density(entity) = sum(1/(k0 + rank_m(entity)))` over metrics m where entity appears in top-N. Requires ≥`c_min=2` metric agreement. Label-collision adds `boost=5.0` to density.

### Per-Level Behavior

- **Easy (~20% corner cases):** Niche-aware **removal** — drop non-protected entities from crowded clusters. Target: reduce corner-case ratio
- **Medium (~50% corner cases):** Removal-only, add/remove toward target ratio
- **Hard (~60-80% corner cases):** Heavy LLM interpolation — generate near-twin entities from existing ones. Placement: mixed strategy (some cross-source matched near-twins → hard positives; some single-source → pure hard negatives)

### Corner-Case Pair Miner

Recall-biased: any metric flagging a pair as "hard" → classified as corner case. Used to:
1. Measure current corner-case ratio
2. Regenerate EM test set stratified by corner-case status

### LLM Entity Interpolation (hard only)

1. Select 2+ parent entities from same niche
2. LLM generates a fictional "blend" entity combining attributes from parents
3. Verify: entity does not match any known real entity (contamination spot-check)
4. Place across sources: cross-source placement creates hard positive pairs; single-source creates hard negatives
5. Cache interpolation results: never re-query on rerun

## Per-Domain Baselines (from knob card)

| Domain | Baseline | Notes |
|---|---|---|
| Companies | ~medium on matched space | Large unlabeled population → effective hard headroom large. Pooled-set floor most impactful |
| Games | Below medium | Strongest franchise substrate (sequels/remasters). Natural niche signal from DBpedia |
| Music | Lowest density | Strongest label-collision substrate (covers, remasters). Most upward headroom |

## Files to Create

| File | Responsibility |
|---|---|
| `usecases_synthetic/scripts/apply_knob_02_niche.py` | CLI: `--domain`, `--level`. Computes metrics, scores entities, removes/interpolates, regenerates EM test set, writes provenance |
| `usecases_synthetic/lib/niche_metrics.py` | `lexical_extended_jaccard(a, b, threshold)`, `compute_tfidf_matrix(corpus)`, `compute_embedding_matrix(corpus, model_id, cache_path)`, `attribute_overlap(a, b, weights)`, `label_collision_index(corpus)` |
| `usecases_synthetic/lib/niche_scorer.py` | `rrf_density(entity, metric_topk_lists, k0, c_min, boost)`, `rank_entities_by_density(all_entities, metrics)`, `select_for_removal(ranked, target_ratio, protection_set, rng)` |
| `usecases_synthetic/lib/corner_case_miner.py` | `mine_corner_cases(pairs, metrics, thresholds) -> set[tuple]`, `measure_corner_case_ratio(pairs, corner_cases)`, `regenerate_em_test_set(all_pairs, corner_cases, target_ratio, rng)` |
| `usecases_synthetic/lib/entity_interpolation.py` | `select_parent_entities(niche_cluster, rng)`, `interpolate_entity(parents, domain_schema, llm_cache, rng)`, `place_entity(entity, sources, placement_strategy, rng)`, `contamination_check(entity, reference_corpus)` |
| `usecases_synthetic/cache/knob_02_embeddings/` | Pre-computed embeddings per domain (`.npy` files) |
| `usecases_synthetic/config/knob_02_niche/companies.yaml` | `corner_case_ratio` per level, `placement_split`, metric weights, per-domain attribute weights, embedding model ID |
| `usecases_synthetic/tests/test_knob_02.py` | Each metric independently, RRF fusion, density scoring with/without label-collision boost, protection set respected, corner-case miner finds known hard pairs, interpolated entity schema validity, embedding cache hit/miss |

## Acceptance Criteria

1. RRF density is monotone in number of agreeing metrics
2. No protected entity removed at any level
3. At easy, corner-case ratio measured ≤ target from YAML
4. At hard, interpolated entities have valid values for all required schema fields
5. Embeddings cached on disk and loaded on rerun (no recomputation)
6. EM test set regenerated per variant with corner-case stratification
7. Contamination spot-check passes for all interpolated entities
8. `pytest usecases_synthetic/tests/test_knob_02.py -v` passes

## Dependencies

Module 0 (domain config, provenance, RNG, protection set, loaders). Module 6 (`llm_cache.py`). `sentence-transformers` package for embeddings.
