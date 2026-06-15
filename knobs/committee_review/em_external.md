# EM Committee — External Literature Search (C2.2)

Targeted external search executed against the query set in [plan_committee_finalization.md §C2](../../plans/plan_committee_finalization.md#c2--entity-matching-committee). Ranking here is by fit-to-committee (which of the six C2.1 gaps it plugs), not citation count.

## Queries executed

| # | Query | Purpose | Top takeaway |
|---|---|---|---|
| 1 | `"entity matching" benchmark 2024..2026` | Current SOTA landscape | Recent survey [Heterogeneity in EM (arXiv 2508.08076)](https://arxiv.org/pdf/2508.08076) confirms HierGAT + Ditto as graph-attention / PLM anchors; generalization across schema drift is the open problem. |
| 2 | `"entity resolution" large language model 2024..2026` | LLM-era matchers | MatchGPT (Mannheim) + ComEM are the two runnable prompt-engineering families. RAG for EM is still a paper-level idea (TableRAG targets QA, not EM). |
| 3 | `"Ditto" OR "HierGAT" OR "Sudowoodo" OR "Jellyfish" entity matching` | Anchor expansion | HierGAT has public code at CGCL-codes/HierGAT. Sudowoodo has public code at megagonlabs/sudowoodo. Jellyfish weights on HuggingFace. |
| 4 | `"zero-shot entity matching" 2024..2026 small language model` | SLM zero-shot alternatives to AnyMatch | Unicorn (SIGMOD '23) is the main structured competitor (DeBERTa-base, multi-task MoE). TableLlama + Jellyfish are larger (7B–13B) and already cached. |
| 5 | `"contrastive learning" entity resolution 2023..2026` | Gap 2 (self-supervised matchers) | Sudowoodo, SC-Block, Peeters & Bizer's "Supervised Contrastive Learning for Product Matching", UCL-Blocker (2026, ScienceDirect, access-restricted). |
| 6 | `"Jellyfish" OR "AnyMatch" OR "LEMONADE" OR "TailorMatch" matching` | Anchor expansion | All four already portfolio-anchored; no new runnable siblings surfaced. |
| 7 | `"blocking" entity matching transformer 2023..2026 FAISS HNSW sparse dense hybrid` | Gap 3 (hybrid blockers) | DeepBlocker (VLDB '21, megagonlabs), SC-Block (ESWC '24, Mannheim), JedAI / pyJedAI toolkits. SparkER was not surfaced with a maintained 2024+ release. |
| 8 | Mannheim WBSG repos | Preferred internal methods | SC-Block, MatchGPT, contrastive-product-matching, jointbert, ALMSER-GEN all active. |
| 9 | Cite-this-paper from Ditto + AnyMatch + Sudowoodo | Forward-citation graph | Confirmed the above; surfaced GraLMatch (EDBT '25) as a graph-based post-matching cleanup, and CSGAT (Nature Sci Reports '25, no code) as a newer graph-attention extension of HierGAT. |

All 9 queries completed without hitting rate limits. Paywalled hits (VLDB / SIGMOD / IEEE proceedings) all had preprint mirrors on arXiv or author pages, so no dead-end was blocking. UCL-Blocker (ScienceDirect) and CSGAT (Nature Sci Reports, open access but paper-only) are the only two that stay at the abstract level — logged as one-liners below.

## Candidates (ranked by gap-fit × code-maturity)

| # | Method / paper | Venue + year | Method class | Code + license | Key idea (one line) | Gap(s) filled |
|---|---|---|---|---|---|---|
| 1 | **Sudowoodo** — Wang, Li, Wang, *Sudowoodo: Contrastive Self-supervised Learning for Multi-purpose Data Integration and Preparation* ([arXiv](https://arxiv.org/abs/2207.04122)) | ICDE 2023 | Contrastive self-supervised PLM for blocking + matching | ✅ [megagonlabs/sudowoodo](https://github.com/megagonlabs/sudowoodo) — Apache-2.0, PyTorch | SimCLR-style contrastive pretraining over unlabeled serialized entities; shared embedder serves both the blocker (nearest-neighbour) and the matcher (fine-tuned or few-shot). | Gap 2 (contrastive), Gap 3 (embedding blocker), Gap 5 (label-light alternative to AnyMatch) |
| 2 | **SC-Block** — Brinkmann, Shraga, Bizer, *SC-Block: Supervised Contrastive Blocking within Entity Resolution Pipelines* ([arXiv](https://arxiv.org/abs/2303.03132)) | ESWC 2024 | Supervised contrastive + ANN blocker | ✅ [wbsg-uni-mannheim/SC-Block](https://github.com/wbsg-uni-mannheim/sc-block) — MIT, PyTorch + FAISS | SupCon loss over entity-pair labels produces embeddings where same-entity records cluster; FAISS/HNSW nearest-neighbour gives a candidate set ~50% smaller than vanilla embedding blockers without F1 loss. | Gap 3 (dense+sparse hybrid via FAISS), Gap 2 (contrastive), Mannheim-preferred |
| 3 | **HierGAT** — Yao, Gu, Liu, Liu, Jin, *Entity Resolution with Hierarchical Graph Attention Networks* ([ACM](https://dl.acm.org/doi/10.1145/3514221.3517872)) | SIGMOD 2022 | Hierarchical graph-attention matcher | ✅ [CGCL-codes/HierGAT](https://github.com/CGCL-codes/HierGAT) — no-licence-file (paper-code release) | Three-level graph (token → attribute → entity) with GAT layers; attention over graph edges identifies discriminative tokens and attributes. Reports +8.7 F1 over Ditto on ER-Magellan. | Gap 1 (graph/attention-pooling, entirely absent) |
| 4 | **Unicorn** — Tu, Fan, Tang, Wang, Lin, Jin, Guo, *Unicorn: A Unified Multi-tasking Model for Supporting Matching Tasks in Data Integration* ([arXiv mirror](https://nantang.github.io/research/pubs/unicorn.pdf)) | SIGMOD 2023 | DeBERTa-base + Mixture-of-Experts multi-task matcher | ✅ [ruc-datalab/Unicorn](https://github.com/ruc-datalab/Unicorn) — Apache-2.0, PyTorch + HuggingFace | Single encoder + MoE layer + binary head trained jointly on 7 matching tasks across 20 datasets; supports zero-shot EM on unseen domains. | Gap 5 (SLM zero-shot alternative to AnyMatch), partial Gap 2 (MoE sharing) |
| 5 | **MatchGPT** — Peeters, Steiner, Bizer, *Entity Matching using Large Language Models* ([arXiv](https://arxiv.org/abs/2310.11244)) | arXiv 2023 (updated v4 2024) | LLM zero/few-shot prompt engineering for EM | ✅ [wbsg-uni-mannheim/MatchGPT](https://github.com/wbsg-uni-mannheim/MatchGPT) — no-licence-file (code + extensive prompt library) | Systematic exploration of hosted + open-source LLMs for EM with structured prompt templates, in-context examples retrieved from a labeled pool, and an "explain errors" postprocessing step. | Gap 4 (runnable retrieval-augmented LLM stack), Mannheim-preferred |
| 6 | **Splink** — Linacre et al., *Splink: A Python package for probabilistic record linkage at scale* ([docs](https://moj-analytical-services.github.io/splink/)) | MoJ open-source library, 2020–2026 active | Probabilistic Fellegi-Sunter linker with EM parameter estimation | ✅ [moj-analytical-services/splink](https://github.com/moj-analytical-services/splink) — MIT, pure-Python (DuckDB / Spark / SQLite backends) | Fellegi-Sunter model with expectation-maximization for m/u probabilities; comparators are explicitly NaN-aware (missing values downweighted via learned u-probability), fully CPU-tractable, deterministic with a fixed seed. | Gap 6 (CPU + NaN-tolerant non-PLM matcher) |
| 7 | GraLMatch — De Meer, Zhou et al., *Matching Groups of Entities with Graphs and Language Models* ([arXiv](https://arxiv.org/abs/2406.15015)) | EDBT 2025 | Graph post-correction of FP pairs | ✅ [FernandoDeMeer/GraLMatch](https://github.com/FernandoDeMeer/GraLMatch) — no-licence-file | Runs a base Ditto matcher, builds a pairwise-prediction graph, and detects/removes false positives via graph transitivity constraints. | Partial Gap 1 (graph signal, different from HierGAT — post-hoc, not encoder-level) |
| 8 | DeepBlocker — Thirumuruganathan, Li, Tang et al., *Deep Learning for Blocking in Entity Matching: A Design Space Exploration* ([PDF](https://www.vldb.org/pvldb/vol14/p2459-thirumuruganathan.pdf)) | VLDB 2021 | Unsupervised deep embedding blocker | ✅ [anhaidgroup/deepblocker](https://github.com/anhaidgroup/deepblocker) — no-licence-file | Design space of 8 deep blockers (autoencoder, Tuple-BERT, USI, etc.) producing ANN-ready embeddings for candidate set generation. Subsumed by Sudowoodo / SC-Block on the contrastive axis but still a solid label-free baseline. | Gap 3 (alternative embedding blocker) — redundant with existing `EmbeddingBlocker` |
| 9 | pyJedAI — Papadakis, Nikoletos, Skoutas et al., *JedAI3: beyond batch, blocking-based Entity Resolution* ([PDF](https://cgi.di.uoa.gr/~koubarak/publications/2020/JedAIrising.pdf)) | EDBT 2020, Python port 2023 | End-to-end ER toolkit (blocking + matching + clustering) | ✅ [AI-team-UoA/pyJedAI](https://github.com/AI-team-UoA/pyJedAI) — Apache-2.0, pypi `pyjedai` | Java JedAI toolkit ported to Python; wraps token-blocking, sorted-neighbourhood, LSH, pyTFIDF matchers, and Markov-clustering. Strong as a *blocking-library* but pulls in a heavy dependency tree (networkx, sentence-transformers, faiss-cpu, torch). | Gap 3 (LSH + sorted-neighbourhood blockers) — alternative to integrating one-off blocker papers |
| 10 | DeepMatcher — Mudgal, Li, Rekatsinas et al. (2018) | SIGMOD 2018 | BiRNN attribute-level attention matcher | ✅ [anhaidgroup/deepmatcher](https://github.com/anhaidgroup/deepmatcher) — BSD, PyTorch | Classical pre-Ditto DL matcher with attribute-level bidirectional-RNN attention. Predates HierGAT and Ditto; both beat it on published benchmarks. | Gap 1 (outdated) — listed only because it is the direct predecessor of HierGAT |
| 11 | UCL-Blocker — *Unsupervised contrastive learning with multi-granularity dynamic fusion for entity blocking* ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1568494626002826)) | Applied Soft Computing 2026 | Unsupervised contrastive blocker | paywalled, no code repo surfaced | Builds on SC-Block with dynamic-temperature contrastive loss + hard-negative focusing. | Gap 3, but no public code — **paper-only, pending release** |
| 12 | CSGAT — *Contextual semantics graph attention network model for entity resolution* ([Nature Sci Reports](https://www.nature.com/articles/s41598-025-11932-9)) | Nature Sci Reports 2025 | Graph-attention matcher (HierGAT successor) | paper-only, no code surfaced | Token-level + attribute-level contextual GAT with semantic fusion; claims +3–5 F1 over HierGAT on WDC + Magellan. | Gap 1, but no code — **paper-only** |
| 13 | ComEM — *Match, Compare, or Select? An Investigation of LLMs for Entity Matching* ([ACL Anthology](https://aclanthology.org/2025.coling-main.8.pdf)) | COLING 2025 | LLM prompt-strategy framework | ✅ [tshu-w/ComEM](https://github.com/tshu-w/ComEM) — Apache-2.0 | Three LLM prompt strategies (Match / Compare / Select). Already covered by the `comem_match_compare_select` portfolio card — **not a new candidate**, flagged here because the repo is now public (was paper-only in C2.1). |
| 14 | contrastive-product-matching — Peeters & Bizer, *Supervised Contrastive Learning for Product Matching* | WWW 2022 | SupCon loss + BERT matcher | ✅ [wbsg-uni-mannheim/contrastive-product-matching](https://github.com/wbsg-uni-mannheim/contrastive-product-matching) — no-licence-file | Predecessor of SC-Block's matching-side counterpart. Product-domain-only. Subsumed by Sudowoodo / SC-Block for committee purposes. | Gap 2 (redundant) |
| 15 | Dedupe — dedupeio, `dedupe` (pip) | open-source library, active | Active-learning Fellegi-Sunter matcher | ✅ [dedupeio/dedupe](https://github.com/dedupeio/dedupe) — MIT | Active-learning probabilistic matcher; requires interactive labeling loop (bad fit for a batch committee but listed for completeness). Splink is the better batch-mode sibling. | Gap 6 — superseded by Splink |

## New paper cards added in C2.2

Shortlist-worthy (rows 1–6 above) — full implementation-ready cards under `literature-search-generation/`:

| Slug | Gap filled |
|---|---|
| [`sudowoodo_contrastive_em`](../../literature-search-generation/sudowoodo_contrastive_em/) | Gaps 2, 3, 5 |
| [`scblock_supervised_contrastive_blocking`](../../literature-search-generation/scblock_supervised_contrastive_blocking/) | Gaps 2, 3 |
| [`hiergat_graph_attention_em`](../../literature-search-generation/hiergat_graph_attention_em/) | Gap 1 |
| [`unicorn_unified_matching`](../../literature-search-generation/unicorn_unified_matching/) | Gap 5 |
| [`matchgpt_llm_em`](../../literature-search-generation/matchgpt_llm_em/) | Gap 4 |
| [`splink_probabilistic_linkage`](../../literature-search-generation/splink_probabilistic_linkage/) | Gap 6 |

Rows 7–15 are one-line entries; no new card warranted:
- **GraLMatch** (row 7) — niche group-matching post-processor; not a roster-sized member.
- **DeepBlocker** (row 8) — signal-redundant with existing `EmbeddingBlocker`.
- **pyJedAI** (row 9) — toolkit, not a single method; pick individual blockers off it if needed.
- **DeepMatcher** (row 10) — historical only.
- **UCL-Blocker** (row 11) and **CSGAT** (row 12) — no code release verified.
- **ComEM** (row 13) — already in portfolio as `comem_match_compare_select`.
- **contrastive-product-matching** (row 14) — subsumed by Sudowoodo + SC-Block.
- **Dedupe** (row 15) — requires interactive labeling.

## Summary

The external search closes all six C2.1 gaps with public code:

- **Gap 1 (graph / attention-pooling)** — HierGAT (primary), GraLMatch (optional).
- **Gap 2 (contrastive / self-supervised)** — Sudowoodo (multi-purpose) + SC-Block (blocking-only variant).
- **Gap 3 (dense+sparse hybrid blockers)** — SC-Block is the clean choice; DeepBlocker and pyJedAI are weaker alternatives.
- **Gap 4 (runnable retrieval-augmented LLM stack)** — MatchGPT ships a curated prompt library and retrieval helpers; no standalone "MatchRAG" system exists.
- **Gap 5 (SLM zero-shot alternative to AnyMatch)** — Unicorn (DeBERTa-base, multi-task).
- **Gap 6 (CPU-tractable NaN-tolerant matcher)** — Splink is the mature choice; Dedupe is a label-hungry fallback.

Three of the six (SC-Block, MatchGPT, contrastive-product-matching) are Mannheim WBSG releases — preferred per the plan's author-preference rule. Code-availability check: all six shortlist candidates have maintained public repos as of 2026-04-20; licensing ranges from MIT/Apache (Splink, Unicorn, Sudowoodo, SC-Block) to no-licence-file (HierGAT, MatchGPT — acceptable for research-internal use but flag before redistribution).

Scoring and final roster selection are deferred to C2.3.
