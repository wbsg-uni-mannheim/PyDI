# Blocking Committee — Scored Shortlist (C2.4a)

> **USER DECISION (2026-04-21) — roster + composition + metrics frozen.**
>
> **Frozen roster — 6 members:**
> 1. `token_blocker` *(incumbent, lexical)*
> 2. `standard_blocker` *(incumbent, lexical — equality on `name_first_token`)*
> 3. `embedding_blocker` *(incumbent, embedding — backbone **upgraded to `sentence-transformers/all-mpnet-base-v2`**; GPU-always)*
> 4. `sorted_neighbourhood_blocker` *(new, lexical — wired from existing PyDI module)*
> 5. `bm25_blocker` *(new, sparse — Okapi BM25 over serialized records, **`bm25s` backend** for large-dataset throughput, **dedicated tokenizer** with stopword removal + stemming, **not** the shared `preprocess_text` pipeline)*
> 6. `sc_block` *(new, hybrid — Mannheim MIT; supervised contrastive + FAISS; chosen over Sudowoodo for +1 signal-diversity; per-domain checkpoint trained on companies-small gold)*
>
> **Composition strategy — sequential (select-best-blocker-then-match):** the blocking committee runs first over all 6 members, each scored on **recall vs. gold** and **reduction ratio**. The best-performing blocker (primary criterion: recall ≥ 0.97; tie-broken by reduction ratio) is selected, and **its candidate set is passed unchanged to every member of the matching committee**. This is neither pure Cartesian (6 × 5 = 30 combos) nor the old pinned scheme (5 tuples) — it is a two-phase pipeline with a winner-take-all handoff between committees. Runner and YAML both need refactoring to support this shape.
>
> **Blocking recall bar — 0.97.** A blocker is only eligible for selection if it reaches ≥ 97% recall vs. gold on the target source pair. Members that fail this bar are reported (for diagnostic purposes) but cannot become the winning blocker passed to the matching committee.
>
> **Blocking metrics — pair recall + reduction ratio.** [committee_em_scoring.py](../../usecases_synthetic/lib/committee_em_scoring.py) extended to report per-blocker recall (against gold) and reduction ratio (`candidates / |left| × |right|`) pre-matching. Downstream F1 retained as secondary, computed after the matching-committee pass.
>
> **Dropped/superseded choices:**
> - Sudowoodo encoder — dropped (SC-Block preferred).
> - `EmbeddingBlocker` MiniLM-L6 default — superseded by mpnet-base-v2 backbone upgrade.
> - `rank_bm25` — superseded by `bm25s` (large-dataset performance).
> - Shared `preprocess_text` for BM25 — superseded by dedicated tokenizer.
> - Cartesian + pinned composition — superseded by sequential select-best handoff.
>
> Proceed to C2.4b with this roster. The scored shortlist below is preserved as the evidence trail.

---

> **Context.** The [EM shortlist §User Decision (2026-04-21, updated 2026-04-22)](em_shortlist.md#em-committee--scored-shortlist-c23) split the combined EM committee into a **blocking committee** (candidate-set generation) and a **matching committee** (pair classification). The matching-side roster is **frozen** at 4 members (`ditto_plm`, `llm_matcher`, `magellan`, `comem`) — Unicorn was in the 2026-04-21 freeze but was dropped on 2026-04-22. This document is the evidence trail for picking the blocking-side roster.
>
> **This document is advisory, not dispositive.** Following the C1.5 / C2 precedent, the user reviews the scored shortlist and then directs the final roster. The rubric from [plan_committee_finalization.md §Step (iii)](../../plans/plan_committee_finalization.md#step-iii--candidate-shortlist--scoring-rubric) is used as a default heuristic, adapted to blocker-specific signals (see "Rubric adaptation" below).

## Blocking committee — scope

- **Purpose.** Produce a candidate-pair set with high recall vs. the gold at reasonable reduction ratio. Downstream: the **matching committee** classifies those pairs.
- **Scoring differs from matching.** Pair-level F1 is not the right primary signal — a blocker's job is to preserve true-positive pairs while cutting the N×M pair-space; primary metrics are **pair recall** = `|gold ∩ candidates| / |gold|` (fraction of gold pairs retained anywhere in the candidate set; sometimes called *pair completeness* in the ER literature) and **reduction ratio** = `1 − |candidates| / (|L| × |R|)` (fraction of the full pair-space pruned). Both apply uniformly to every blocker regardless of whether it uses a top-k cap internally; `k` is a configuration parameter of `EmbeddingBlocker` / `SC-Block` / `BM25`, not a metric axis. For comparability with downstream matching F1, we still report pair-F1 *under a fixed downstream matcher* (proposed: `ditto_plm`), but the committee-selection rubric below ranks blockers on blocking-specific axes.
- **Roster size.** Target 3–5 members. Blocking diversity matters (lexical + embedding + hybrid), but each additional blocker multiplies the matching committee's work via Cartesian composition (see "Composition strategy" question below).

## Rubric adaptation for blockers

The 5-axis rubric from the plan stays the same, but axis definitions for blocking-specific scoring:

| Axis | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| **Signal diversity** | redundant (same block key + cut) | partial overlap | distinct block signal | distinct + currently absent axis |
| **SOTA alignment** | outdated | baseline | competitive recall/RR | SOTA on a recent blocking benchmark |
| **Integration cost** | rewrite PyDI internals | new adapter | existing adapter | drop-in (already in `PyDI.entitymatching.blocking`) |
| **Determinism / reproducibility** | non-deterministic | flaky, workarounds | seeded, stable | fully deterministic |
| **Runtime budget fit** | GPU-required for inference | GPU-preferred | CPU-tractable with indexes | CPU-tractable, no index build |

Committee-slot cutoff: total ≥ 10, no axis scoring 0. Same threshold as the plan.

## Candidate pool

Pulled from [em_portfolio.md](em_portfolio.md) (incumbents) and [em_external.md](em_external.md) (external search). Only blocker-side candidates are scored here; matcher-only candidates belong in the separate matching shortlist (already frozen).

| # | Candidate | Module / source | Block signal | Score breakdown | **Total** | Verdict |
|---|---|---|---|---|---|---|
| 1 | `TokenBlocker` *(incumbent)* | [`PyDI.entitymatching.blocking.token_blocking`](../../PyDI/entitymatching/blocking/token_blocking.py) | token-shingle inverted index | 2 · 1 · 3 · 3 · 3 | **12** | **Incumbent (keep)** |
| 2 | `StandardBlocker` *(incumbent)* | [`PyDI.entitymatching.blocking.standard`](../../PyDI/entitymatching/blocking/standard.py) | equality on blocking key (e.g. `name_first_token`) | 2 · 1 · 3 · 3 · 3 | **12** | **Incumbent (keep)** |
| 3 | `EmbeddingBlocker` *(incumbent — SBERT-based)* | [`PyDI.entitymatching.blocking.embedding`](../../PyDI/entitymatching/blocking/embedding.py) | sentence-BERT dense ANN top-k (default `all-MiniLM-L6-v2`) | 2 · 2 · 3 · 2 · 2 | **11** | **Incumbent (keep)** — see Q6 re: SBERT backbone upgrade |
| 4 | `SortedNeighbourhoodBlocker` *(already in PyDI, never wired)* | [`PyDI.entitymatching.blocking.sorted_neighbourhood`](../../PyDI/entitymatching/blocking/sorted_neighbourhood.py) | sliding window over a sort key | 3 · 1 · 3 · 3 · 3 | **13** | **Shortlist (add)** — free signal-diversity win |
| 5 | **BM25** (SC-Block top-performing baseline) | `rank_bm25` pip pkg, Apache-2.0; new `BM25Blocker` adapter | sparse lexical scoring (Okapi BM25) over serialized records | 3 · 2 · 3 · 3 · 3 | **14** | **Shortlist (add)** — strongest sparse baseline per the SC-Block paper; CPU-only, deterministic |
| 6 | **SC-Block** (Mannheim, ESWC 2024) | [`literature-search-generation/scblock_supervised_contrastive_blocking`](../../literature-search-generation/scblock_supervised_contrastive_blocking/); MIT, PyTorch + FAISS | supervised-contrastive dense + FAISS/HNSW ANN | 3 · 3 · 2 · 2 · 2 | **12** | **Shortlist (add — conditional)** — fills the `hybrid` / supervised-contrastive axis |
| 7 | **Sudowoodo encoder** (Megagon, ICDE 2023) | [`literature-search-generation/sudowoodo_contrastive_em`](../../literature-search-generation/sudowoodo_contrastive_em/); Apache-2.0 | self-supervised contrastive dense + FAISS ANN | 2 · 3 · 2 · 2 · 2 | **11** | **Defer** — signal-overlap with SC-Block; pick one contrastive encoder |
| 8 | DeepBlocker (VLDB 2021) | [anhaidgroup/deepblocker](https://github.com/anhaidgroup/deepblocker); **no LICENSE** | autoencoder / USI / Tuple-BERT embedding | 1 · 1 · 2 · 2 · 2 | **8** | **Drop** — redundant with `EmbeddingBlocker`; no LICENSE |
| 9 | pyJedAI LSH / Sorted-Neighbourhood | [AI-team-UoA/pyJedAI](https://github.com/AI-team-UoA/pyJedAI); Apache-2.0 | MinHash-LSH, Q-gram-LSH, Sorted-Neighbourhood variants | 2 · 1 · 1 · 3 · 3 | **10** | **Defer** — LSH signal is new, but toolkit drags a heavy dep tree (networkx + faiss-cpu + torch); our own `SortedNeighbourhoodBlocker` covers the SN variant |
| 10 | UCL-Blocker (ASoC 2026) | paper-only, ScienceDirect paywalled | unsupervised contrastive + dynamic temperature | — | **0 (no code)** | **Drop** — paper-only, per plan's "no code → 0" rule |

## Per-candidate rationale

### Included (incumbents + shortlist)

- **`TokenBlocker` (keep).** Lexical inverted-index baseline. Runtime 3 (pure Python, no index build), determinism 3. Not SOTA (axis = 1) but that is the point — cheap, reproducible, runs without any external model.
- **`StandardBlocker` (keep).** Equality-on-blocking-key variant that mirrors the reference companies workflow's `name_first_token` recipe. Distinct from `TokenBlocker` in that it's *exact-equality* on a derived key rather than token overlap — different failure mode under K1/K6 perturbations.
- **`EmbeddingBlocker` (keep).** Already SBERT-based — the `model` parameter defaults to `sentence-transformers/all-MiniLM-L6-v2`, which *is* a sentence-BERT encoder (SBERT distilled, 6-layer, ~22M params, 384-d). Backbone is configurable at committee-YAML time: upgrading to a stronger SBERT variant (e.g. `all-mpnet-base-v2` — 110M params, 768-d, consistently the top-ranked general-purpose SBERT on MTEB) is a one-line YAML change. Axis = 2 with MiniLM-L6; bumps to 3 with mpnet-base-v2 at ~3× inference cost. Covers the `embedding` blocking axis. Runtime 2 (sentence-transformers inference on CPU is workable for companies-small, slower on full `companies`).
- **`SortedNeighbourhoodBlocker` (add).** Already lives in [PyDI/entitymatching/blocking/sorted_neighbourhood.py](../../PyDI/entitymatching/blocking/sorted_neighbourhood.py) but has never been wired into the committee YAML. Signal diversity 3 — sliding-window over a sort key is a distinct lexical signal from both `TokenBlocker` (inverted index) and `StandardBlocker` (equality). Integration 3 (drop-in; uses the same `BaseBlocker` interface). Runtime 3 (O(N log N) sort, no model). The missing-but-cheap axis — expanding the lexical blocker footprint at zero integration cost is a no-brainer.
- **BM25 (add).** Sparse lexical scorer (Okapi BM25) over serialized records — every record is treated as a bag of tokens, query records retrieve top-k matches by BM25 score. The [SC-Block paper](../../literature-search-generation/scblock_supervised_contrastive_blocking/paper.md) reports BM25 as the **strongest non-learned baseline** on its blocking benchmark, often within 1–2 pair recall points of SC-Block itself on several subsets and occasionally *exceeding* the unsupervised dense blockers (DeepBlocker variants). Signal diversity 3 — sparse term-weighting is structurally distinct from both token-set blocking (`TokenBlocker`'s inverted index is unweighted) and dense embeddings (no term-frequency / length-normalization component). SOTA alignment 2 (classical method, competitive on recent benchmarks, not technically SOTA). Integration 3 (`rank_bm25` pip package, Apache-2.0, pure-Python; adapter is <100 LOC over `BaseBlocker`). Determinism 3, runtime 3 (CPU-only, no index build beyond the TF-IDF pass). Total **14 — highest-scoring candidate**. Fills the sparse-scoring gap between the rule-based lexical members (equality / token-set) and the dense-embedding members.
- **SC-Block (add — conditional on user approval).** Fills the *supervised-contrastive + FAISS* axis that no current blocker covers. Score 3 on signal diversity (contrastive learning under label supervision is structurally distinct from both dense unsupervised and lexical), 3 on SOTA (ESWC 2024, Mannheim-preferred, ~50% smaller candidate sets than `EmbeddingBlocker` at equal recall in the published results). Integration 2 (MIT-licensed, PyTorch + FAISS, needs a new adapter under `usecases_synthetic/lib/`). Runtime 2 (inference is CPU-tractable once the encoder is trained; pretraining wants a GPU one-off). **Condition:** requires labeled blocking pairs for the supervised loss — companies-small gold provides these, but SC-Block must be retrained per-domain. If the per-domain training cost is unacceptable, swap to Sudowoodo (self-supervised, no-labels) at the cost of one signal-diversity point.

### Deferred

- **Sudowoodo encoder (Defer).** Self-supervised contrastive encoder that also powers the matching committee's `sudowoodo` member (wait — Sudowoodo is *not* in the frozen matching roster; the matching committee is `ditto / llm / magellan / comem`). So Sudowoodo-as-blocker would be a standalone blocking-only addition here. Signal class overlaps SC-Block (both contrastive + FAISS), so if SC-Block is chosen, Sudowoodo is redundant. If the user prefers label-light over label-supervised, swap: drop SC-Block, add Sudowoodo. Score 11 — held back by redundancy with SC-Block, not individual weakness.
- **pyJedAI LSH / sorted-neighbourhood (Defer).** MinHash-LSH and Q-gram-LSH are genuinely new signal axes (probabilistic set-similarity blocking) but pyJedAI is a heavyweight toolkit with an opinionated pipeline object model; extracting just the LSH blocker would be a non-trivial wrapper. Our own `SortedNeighbourhoodBlocker` covers the SN side. Revisit if the final roster lacks an LSH member and downstream analysis shows set-similarity blocking would plug a recall gap that contrastive + dense embedding don't.

### Dropped

- **DeepBlocker (Drop).** Redundant with `EmbeddingBlocker` (both produce dense ANN embeddings; DeepBlocker's USI/Tuple-BERT variants score similarly to MiniLM on recent benchmarks). No LICENSE file. Total 8 — below cutoff.
- **UCL-Blocker (Drop).** Paper-only as of 2026-04-20, no public code. Plan rule: "no code → 0 on Integration cost". Revisit if/when a repo surfaces.

## Proposed roster (6 members, conditional)

1. **`token_blocker`** *(kept, lexical — `TokenBlocker`; token-set inverted index)*
2. **`standard_blocker`** *(kept, lexical — `StandardBlocker` on `name_first_token`; equality on derived key)*
3. **`embedding_blocker`** *(kept, embedding — `EmbeddingBlocker` / SBERT; default `all-MiniLM-L6-v2`, optional upgrade to `all-mpnet-base-v2` — see Q6)*
4. **`sorted_neighbourhood_blocker`** *(new, lexical — wired from already-available PyDI module; sliding window)*
5. **`bm25_blocker`** *(new, sparse — Okapi BM25 over serialized records; strongest non-learned baseline per SC-Block paper)*
6. **`sc_block`** *(new, hybrid — supervised-contrastive + FAISS; Mannheim, MIT — requires new adapter + per-domain training on companies-small gold)*

### Axis coverage audit

- `blocking_type` — `lexical` (3 members: 1, 2, 4), `sparse` (1 member: 5 — new axis added), `embedding` (1 member: 3), `hybrid` (1 member: 6). Four blocking axes covered. The new `sparse` axis distinguishes BM25's term-weighted scoring from the unweighted token-set membership test used by `token_blocker`.
- Deterministic fallback — members 1, 2, 4, 5 are fully deterministic (BM25 is pure arithmetic over token statistics). `EmbeddingBlocker` is seeded-deterministic; SC-Block inherits the same seeding surface as Sudowoodo / Ditto. ≥4 deterministic-fallback members preserves `--with-llm=false` reproducibility (strictly better than the old EM committee's guarantee).

### Risks flagged for the user

- **SC-Block per-domain training.** Needs labeled matching pairs to compute the SupCon loss. Companies-small ships gold, so the training is feasible, but **every new domain in S5 will require a new SC-Block checkpoint**. If that ongoing cost is unpalatable, swap to Sudowoodo (self-supervised, no per-domain training).
- **Blocking-matching Cartesian explosion.** 6 blockers × 5 matchers = 30 (blocker, matcher) combinations per source pair. With 3 source pairs in companies + macro-F1 aggregation, that's 90 per-member evaluations. Under the default `greedy` 1:1 clustering this is fine for companies-small but scales poorly on S5's full `companies` variant. See Composition strategy question below.
- **BM25 preprocessing sensitivity.** BM25 recall is highly sensitive to tokenization and stopword handling. If the `preprocess_text` pass used for other lexical members (punctuation-strip + lowercase) is applied without stopword removal, BM25's IDF term gets diluted by high-frequency noise words. Recommendation: give the BM25 adapter its own tokenization config (minimum: stopword removal + optional stemming) rather than sharing the committee-wide `preprocess_text` hook.

## Open questions for the user (hooks for C2.4b)

1. **SC-Block vs Sudowoodo for the contrastive/hybrid slot.** SC-Block (supervised, +1 signal-diversity, per-domain training cost) vs Sudowoodo (self-supervised, no training cost, one signal-diversity point lower). Both are Apache-2.0 / MIT. Pick one; dropping both leaves the `hybrid` axis empty and the proposed roster at 5 members.
2. **Composition strategy: Cartesian or pinned?** Two options:
    - **Cartesian** — every blocker feeds every matcher (6 × 5 = 30 combos per pair). Rich diagnostics; quadratic cost.
    - **Pinned** — each matcher declares a preferred blocker (e.g. `ditto_plm` + `EmbeddingBlocker`, `magellan` + `TokenBlocker`, etc.). 5 (blocker, matcher) tuples per pair — same cost as today. Loses the ablation signal but matches the existing runner.
    The current [committee_em.py](../../usecases_synthetic/lib/committee_em.py) follows the **pinned** pattern (each YAML member is a (blocker, matcher) tuple). A true Cartesian split would need a runner refactor.
3. **Retire `EmbeddingBlocker` once SC-Block lands?** SC-Block's encoder subsumes MiniLM on both pair recall and reduction ratio per published results. Keeping both gives a weak member (`EmbeddingBlocker`) alongside a stronger one on the same axis. Drop #3, or keep it as the deterministic-seeded fallback for `--with-llm=false` runs?
4. **Wire `SortedNeighbourhoodBlocker` even without the hybrid slot?** This is an easy yes (it's free — already in PyDI) but worth confirming since it lifts the roster from 3 to 4 lexical-heavy members.
5. **Blocking-specific metrics in the runner.** Current runner reports F1 / precision / recall against gold, computed downstream of the matcher. Should the blocking committee also report **pair recall** and **reduction ratio** pre-matching? If yes, C2.4b needs to extend [committee_em_scoring.py](../../usecases_synthetic/lib/committee_em_scoring.py) with those metrics. Recommendation: yes — otherwise the blocking-committee rollup has no way to show why a blocker was picked.
6. **SBERT backbone for `EmbeddingBlocker`.** Default today is `sentence-transformers/all-MiniLM-L6-v2` (2020; 22M params, 384-d — cheap and SBERT-trained). The follow-on `all-mpnet-base-v2` (110M params, 768-d) is the community-preferred general-purpose SBERT in 2026 and typically delivers 3–5 pair recall points at ~3× inference cost. Upgrade the default, keep the current value, or leave the committee YAML pinning an explicit backbone per environment (CPU-only runs keep MiniLM; GPU/MPS runs use mpnet-base-v2)?
7. **BM25 tokenization.** Share the committee-wide `preprocess_text` pipeline (punctuation strip + lowercase) vs. give BM25 its own tokenizer with stopword removal + optional stemming. Sharing is simpler; a dedicated pipeline is materially better for recall — BM25's IDF term depends on the stopword policy. Recommendation: dedicated pipeline.
8. **BM25 implementation.** `rank_bm25` (pure-Python, Apache-2.0, maintained) vs. `bm25s` (2024; 500× faster via sparse-matrix tricks, MIT). Both have the same BM25 semantics. `bm25s` is meaningfully faster on full `companies` but adds a dependency; `rank_bm25` is sufficient for companies-small and already widely pinned.

## Required-axes update (for C2.4b, if roster accepted)

The blocking committee's YAML (proposed location: `usecases_synthetic/config/committees/em_blocking_committee.yaml`) would replace the old combined `em_committee.yaml`'s blocking-side `required_axes` with:

```yaml
required_axes:
  blocking_type: [lexical, sparse, embedding, hybrid]  # was [lexical, embedding]
```

The matching committee's YAML (separate file) drops `blocking_type` from `required_axes` entirely since blocker-choice is composed at runtime.

## Deliverable

Six-member blocking committee covering `lexical` (3), `sparse` (1), `embedding` (1), `hybrid` (1). Three additions vs. today: `SortedNeighbourhoodBlocker` (drop-in from existing PyDI code), BM25 (strongest non-learned baseline per the SC-Block paper; new adapter over `rank_bm25`), and SC-Block (new adapter + per-domain training). One conditional swap surfaced (SC-Block ↔ Sudowoodo). Eight open questions flagged for the user, covering the contrastive-method choice, composition strategy, the `EmbeddingBlocker` retirement question, the SortedNeighbourhoodBlocker wire-up, blocking-specific metrics, the SBERT backbone upgrade, BM25 tokenization, and the BM25 implementation library. Proceed to C2.4b (YAML + adapter + runner changes) once the user picks.
