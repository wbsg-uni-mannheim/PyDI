# Cross-cutting policies

Policies that apply to every knob. Per-knob cards reference this file rather than restating these.

## Profile model — Option B (absolute target bands)

Each domain ships four artifacts: `baseline` (reference, untouched) plus `easy` / `medium` / `hard` augmented variants. Each variant is generated to hit an *absolute* target difficulty band for the knob, which may require *adding* heterogeneity (baseline below target) or *reducing* it (baseline above target — e.g. normalize-down for easy). Comparability across domains is a **soft goal**: same intent per level and monotone easy→medium→hard shape required; absolute committee numbers need not align across domains.

## Per-value provenance (mandatory)

For every augmentation, emit a row-level provenance record with at minimum:

```
(entity_id, source, attribute, original_value, new_value, transform_fn, transform_params, knob, level)
```

Knobs that operate on entities (Knob 2) or columns (Knobs 8, 9) emit entity-scoped or column-scoped records instead. Per-knob cards specify the exact `transform_fn` values they use.

## Test-set treatment

- **Fusion test set:** entity membership and gold values frozen. Only mutate when unavoidable, and only via the committee-driven gold replace/extend rules below.
- **EM test set:** regenerated per variant as per-source-pair per-split files (`<src1>_2_<src2>_{train,val,test}_regenerated.csv`). Each split mirrors the original EM gold's row count and positive:negative ratio; the corner-case ratio (driven by Knob 2) grows monotonically easy → medium → hard within each split. The regenerated **val** split is the **primary** EM measurement surface (drives monotonicity + ablation verdicts); the regenerated **test** split is an internal sanity check (confirms val and test move together); **train** is emitted for downstream public benchmark consumers. Pool priority `test > val > train` protects the cleanest sample for the sanity surface; when the positive pool cannot supply all three splits, sizes scale down *proportionally* so the positive:negative ratio is preserved exactly in every split. The negative pool is expanded per authored source-pair via a random top-up pass so **negatives are never the scaling bottleneck** — only positives can be. See [Protection set semantics](#protection-set-semantics-not-replacement-gold) for the primary/secondary/diagnostic split; the original human-annotated gold is retained as a secondary cross-paper metric.
- **SM mappings:** unchanged in S1 unless Knob 8 is actively perturbing headers. Regenerated per variant in S2 (Knobs 8 and 9).

## Committee-validated augmentation

Knobs that touch values (1, 5, 6, 7, 10) risk pushing the dataset outside the regime where the gold is recoverable. Resolution: wrap augmentation in a measurement loop using **committees of standard PyDI methods**, separately at the fusion, SM, and EM stages.

1. **Establish baseline.** Run the committee on the *original* dataset under lenient evaluation. Record per-attribute / per-stage baseline accuracy / F1.
2. **Apply augmentation** with the chosen knob settings.
3. **Re-measure.** Run the same committee on the augmented dataset.
4. **Compare.** Expected outcome: a controlled monotone *drop* (the difficulty signal). Unexpected: collapse to ~random.
5. **Fix on collapse**, in preference order:
   - **Soften source augmentation locally** for the offending entities/attributes (preserves the original gold).
   - **Update the gold** (replace with closest augmented variant, or extend to an accepted set) only if softening can't preserve the difficulty signal.

The committee mechanism is the **safety net** (prevents impossible tasks), the **calibration tool** (turns qualitative scales into measured deltas), and the **validation evidence** required by [../plan.md](../plan.md) §Validation Methodology.

### Per-knob fix-strategy defaults

| Knob | Fix on collapse |
|---|---|
| 1 (paraphrase) | Replace-or-extend gold to accepted set |
| 5 (formats) | Trivial — canonical-form comparison absorbs format/unit differences |
| 6 (noise) | **Reject** the augmentation; gold artifact never touched (typos must not be promoted into the gold contract) |
| 7 (ambiguity) | Extend gold to accepted sets where homonyms/collisions occur (deferred — Knob 7 specced but not built in v1) |
| 10 (reliability) | No gold change needed; gold is reshuffled across sources, not perturbed |

### Committee composition (fusion, draft)

3–5 strategies per attribute type; reflects what someone would actually use, not just what PyDI ships:

- **Strings (titles, descriptions):** `longest_string`, `shortest_string`, `most_frequent`, optional LLM-arbitrate.
- **Numerics:** `median`, `mean`, `most_frequent`.
- **Categoricals:** `most_frequent`, `voting`.
- **Lists (founders, platforms, genres):** `union`, `intersection`, `voting`.
- **Dates:** `earliest`, `latest`, `most_frequent` after canonicalization.

SM and EM committee compositions are deferred to the algorithm-selection phase.

## Bootstrap order (committees → baseline → calibration)

The committee designs must exist before the per-source baseline measurement (which Knob 10 in particular consumes). **Committee design → baseline measurement → per-knob calibration.** Per-domain baselines used by Knobs 3, 6, 8, 10 are measured during the algorithm-selection phase, not earlier.

## Gold standard incompleteness and pooling

The per-domain EM gold is a **sampled subset** of the true match set, not a complete enumeration. Records outside the labeled pairs are not confirmed non-matches — they are *unlabeled*. This matters for every knob that samples from the "non-matched" population (Knob 2 entity selection; Knob 2 + Knob 1 hard-negative mining) and for the committee loop above, which scores committee output against a test set that undercounts real positives.

### Pooled positives set

For each domain we build a **pooled positives set** by merging declared matches from two existing matcher pipelines with diverse inductive biases. No new matching runs are performed — the pool can be refined later by adding further matcher configs. The pool is serialized as `usecases_synthetic/pools/<domain>/pooled_positives.csv` (columns `id1`, `id2`, `source_1`, `source_2`, `pool_agreement`).

**Primary (trusted) source — PLM-based pipeline:** per-pair files under `automatic-data-integration/scripts/output/<domain>_0302/entity_resolution/matching/correspondences_<a>_<b>.csv`. Covers companies, games, music. **Every pool row is a PLM pair.**

**Corroboration source — rule-based human-baseline pipeline:** raw pairwise edges inside the `correspondences` arrays of `usecases_new/output/<domain>/cluster_analysis/detailed_cluster_info.json`. The pre-computed cluster structure is ignored; we extract only the pairs. Rule-based participation *upgrades* a PLM pair's `pool_agreement` from 1 to 2 — it does not add new pairs. Rule-based-only pairs (declared by the rule engine but not by PLM) are dropped because the hand-weighted `StringComparator` linear combinations are materially weaker than the learned PLM matcher and their unique positives are low-confidence.

**`pool_agreement` semantics under this policy:**
- `1` — PLM-only (rule-based did not corroborate, or did not cover this directed pair).
- `2` — both PLM and rule-based declared the pair.

**Coverage gaps:**
- **Movies** has no PLM source yet — pool construction deferred until the movies PLM run is available.
- **Products** has neither source yet (WIP use case). Pool construction deferred until products stabilizes.
- The rule-based cluster JSON only contains correspondences for one directed source pair per domain (forbes↔dbpedia for companies, metacritic↔dbpedia for games/music). Under the corroboration-only policy this is acceptable — partial coverage just means fewer PLM pairs get `agreement=2`; no PLM pair is lost.

**Egregious-cluster filter (applied to the merged graph, not per-source).** After the corroboration-only policy is applied, we run `nx.connected_components` on the surviving edges and drop components whose size exceeds `max(ceil(P99 of observed component sizes), 3 * n_sources)`. The P99 adapts to each domain's distribution; the `3 * n_sources` structural floor reflects the upper bound on plausible cross-source duplicate clusters (n_sources members plus slack for in-source duplicates). This removes only the most egregious transitive-chain artifacts (e.g. the companies 117-entity "Chinese companies" cluster) — anything near the typical cluster size is preserved, so Knob 2 hard-negative headroom is only constrained by the genuine protection contract. Per-domain cap, dropped-component count, and size histogram are reported in `pool_stats.json`.

```
expanded_positives = test_gold ∪ train_gold ∪ val_gold ∪ pooled_positives
```

Pool construction is a prerequisite for the committee loop and for Knob 2 calibration; it runs during the algorithm-selection phase alongside committee design and per-domain baseline measurement. Construction is cheap — read existing files, expand clusters, union, deduplicate.

### Protection set semantics (not replacement gold)

The pool is used as a **"probably-positive, do not perturb" protection set**, never as a replacement gold. **Pool-as-replacement-gold remains forbidden** — using the pool's declared positives as the scoring oracle would rubber-stamp the pooling systems (every pool member would score ≈1.0 against its own declaration on its own inductive bias).

Concretely, the pool constrains the *generator*, not the evaluator:

1. **Knob 2 entity selection:** entities appearing in any `expanded_positives` pair are protected — never dropped (easy removal), never used as parent seeds for single-source distractor interpolation (hard level). This extends the fusion-gold floor into a pooled-positives floor.
2. **Hard-negative mining (Knobs 1, 2):** synthetic hard negatives are drawn only from pairs *outside* `expanded_positives`, with an additional score-based safety margin around the pool systems' similarity thresholds (pair must sit below every pool system's decision boundary by at least margin δ, δ TBD at algorithm selection).
3. **Committee loop diagnostic signal:** compute a "committee-vs-pool agreement" rate as a diagnostic. If primary F1 collapses but pool agreement stays high, the collapse is probably hidden-positive noise rather than real difficulty — soften the augmentation. If both drop together, the difficulty signal is real. Pool agreement is never a reported number.
4. **Knob 6 (noise) is unchanged** — typos never get promoted into any gold contract, pooled or otherwise.

#### Reportable EM F1 — primary vs. secondary

Under S1 the Knob 2 regenerated **validation** split is the **primary EM measurement surface** (authoritative variant F1); the regenerated **test** split is an internal sanity check; the original human-annotated gold is reported as a **secondary cross-paper-comparability metric**:

- **Primary — regenerated validation gold.** Each variant emits `input/entitymatching/<src1>_2_<src2>_val_regenerated.csv` per authored source-pair. Positives are cross-source record pairs within a surviving canonical entity; negatives are cross-cluster pairs mapped to an authored source-pair and gated by a score-margin hard-negative policy (see S3). Ground truth is **known by construction** — the rubber-stamp argument that bars pool-as-gold does not apply, because the pairs are not drawn from any matcher's own decision boundary. Primary F1 is `score_em_correspondences_closed_set(preds, regenerated_val_gold)`: predictions outside the val universe are **out of scope** and do not count as FPs. This closed-set scoping prevents precision collapse when the matcher scans a larger pair space than the benchmark covers.
- **Sanity — regenerated test gold.** Each variant also emits `<src1>_2_<src2>_test_regenerated.csv` — a disjoint sample from the same construction process. Never looked at during knob-intensity tuning. Scored with the same closed-set semantics. Monotonicity + ablation verdicts read the val surface; test should move in the same direction. Val/test divergence is a red flag for overfitting knob calibration to one sample.
- **Downstream — regenerated train gold.** `<src1>_2_<src2>_train_regenerated.csv` matches the original train file's row count + positive ratio so downstream public benchmark consumers can train matchers on the variant and report numbers.
- **Secondary — original human gold.** The per-pair `<src1>_2_<src2>_all.csv` / `_test.csv` files are still scored against predictions (`f1_vs_test_gold`) so we can compare against prior work where the regenerated test set does not exist. This is a reporting metric only — monotonicity checks and ablation verdicts use the primary regenerated F1.
- **Diagnostic — pool agreement.** `pool_precision` / `pool_recall` remain unchanged and inform the collapse-vs-hidden-positive check above.

Committee composition is frozen across variants (same blockers, matchers, and thresholds), so cross-variant deltas on the regenerated F1 isolate the knob's difficulty signal.

**Contamination mitigation (paper defense).** Using val to verify our difficulty signal and also ship it as part of the public benchmark would be doubly problematic: (a) we tune knobs against committee F1 on val (implicit overfitting), and (b) the hard-negative gate uses a Ditto PLM to filter negatives, so Ditto's F1 on val is a ceiling estimate. Our design avoids both: the public benchmark's **authoritative metric for downstream users remains `f1_vs_test_gold` on the original human-annotated test split**, which neither our committee nor the gate curated. Val/test-regenerated are instruments for *our* internal difficulty claim, not for ranking downstream matchers.

**Open-set vs. closed-set scoring — interpreting the test_gold ↔ regen gap.** `f1_vs_test_gold` (open-set) and `f1_vs_regenerated_*` (closed-set) often diverge by an order of magnitude on large domains and that gap is *expected*, not a quality signal. Open-set scoring counts every committee positive prediction outside the gold's positive set as a FP, so a matcher that blocks tens of thousands of candidates on a 100k-row source and predicts hundreds of positives ends up with ~0.05 precision against a 1.7k-pair test_gold even when most of those predictions are correct — we just have no ground truth for them. Closed-set scoring restricts predictions to the gold's *judged* pair universe (positives + explicit negatives), so the same matcher's precision reflects only judged decisions. Verified empirically on games (2026-05-01): all 1757 test_gold pairs survive K2 row removal at every level (100% pair survival), so the 0.07 vs 0.79 gap (test_gold vs regen_test) cannot be attributed to entity removal — it is purely the open-set FP penalty. **Implication for cross-variant interpretation:** treat the open-set absolute number as a benchmarking surface (compare-to-prior-work only); use the closed-set `regen_val` (or `regen_test` when val is missing — see below) as the difficulty surface.

**Train/val/test triplet caveat.** When a domain's original EM gold ships only a `_test.csv` (no `_val.csv` / `_train.csv`), K2's regenerated splits inherit the same shape: only `<src1>_2_<src2>_test_regenerated.csv` is emitted, and `f1_vs_regenerated_val` is `NaN`. In that case the *sanity* surface (`f1_vs_regenerated_test`) becomes the de facto primary signal. Verified on games (test-only EM gold, val NaN, sanity surface 0.59–0.79 across levels with clean monotone descent). When the val surface IS populated (e.g. companies, music), it is preferred as the primary; sanity should track it within ±0.015 (per S4c companies-small smoke test).

**Aggregate vs. best-member reporting.** Committee macro_f1 is the unweighted mean across all enabled members of a stage. That number can mask one strong individual or one disabled-by-failure member dragging the mean down (e.g. games SM at easy: macro_f1 0.59 = mean of {coma_hybrid 1.00, embedding_sbert 0.36, llm_openai 1.00, duplicate_majority 0.00}; the best member is at 1.00 and `duplicate_majority` is structurally zero). When narrating per-stage signals in plan rows / progress reports, **always also report the best-member F1 alongside the committee macro_f1**, e.g. `SM macro_f1=0.59 (best=coma_hybrid 1.00)`. The best-member ceiling is what an end user could obtain by picking the strongest single matcher; the committee mean is what the difficulty validation guarantees against averaging. Both numbers are needed to interpret a delta: a -0.10 macro drop with a -0.20 best-member drop is real; a -0.10 macro drop with a flat best-member is one weak member regressing.

### Residual limitations

- Pooling gives a **recall lower bound**, not completeness. Matches that the PLM misses remain hidden. The corroboration-only policy intentionally trades recall (dropping rule-based-only pairs) for precision; additional stronger matchers (e.g. LLM-based) can be added later to tighten the lower bound without loosening the policy.
- The cost of the protection set is **reduced augmentation headroom** — the generator has fewer entities/pairs to work with. For domains with small labeled gold and large record counts (companies, games), this may materially shrink Knob 2's downward range; per-domain headroom estimates must be re-measured against the pooled set, not the raw EM gold.
- Movies has no PLM source yet; pool construction is deferred.
