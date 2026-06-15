# SM Committee — Portfolio Inventory (C1.1)

Committee in scope: schema matching. Current roster (see [../../usecases_synthetic/config/committees/sm_committee.yaml](../../usecases_synthetic/config/committees/sm_committee.yaml)):

| # | Member | Signal | Module |
|---|---|---|---|
| 1 | `label_jaccard` | label (string sim over column names, tokenized) | `PyDI.schemamatching.label_based.LabelBasedSchemaMatcher` |
| 2 | `label_jaro_winkler` | label (character-level, untokenized) | same |
| 3 | `instance_tfidf_cosine` | instance (value-distribution cosine over TF-IDF) | `PyDI.schemamatching.instance_based.InstanceBasedSchemaMatcher` |
| 4 | `duplicate_majority` | duplicate (value co-agreement on known EM correspondences) | `PyDI.schemamatching.duplicate_based.DuplicateBasedSchemaMatcher` |
| 5 | `llm_openai` | LLM (zero-shot prompting over column metadata + sample rows) | `PyDI.schemamatching.llm_based.LLMBasedSchemaMatcher` |

Axis covered today: `label`, `instance`, `duplicate`, `llm`. Missing: `embedding` (contextual/vectorised column representation), `structural` (cross-column consistency, graph propagation), `hybrid-learned` (an SM-specific SLM or PLM fine-tuned on labelled pairs).

## Portfolio anchors (existing `literature-search-generation/` entries)

Only the papers whose method class is *directly reusable* as a committee member. Benchmark-only entries (Valentine, XBenchMatch, Alaska, iBench) are pointed at separately as sources of candidates — not members.

| Anchor | Method class | Maturity | Axis it would fill | Dependency cost | Note |
|---|---|---|---|---|---|
| [magneto_schema_matching](../../literature-search-generation/magneto_schema_matching/) | SLM retrieval + LLM rerank hybrid | research code ([VIDA-NYU/magneto-matcher](https://github.com/VIDA-NYU/magneto-matcher), Python) | `hybrid` / `embedding+llm` | new adapter; pulls in `sentence-transformers` (already in our core deps) + an LLM call budget | SOTA on Valentine + GDC benchmarks (VLDB '25). Our current `llm_openai` is already LLM-based but *not* retrieval-augmented. |
| [valentine_schema_matching](../../literature-search-generation/valentine_schema_matching/) | Benchmark + reference matchers (COMA wrapper, Cupid, Similarity Flooding, DistributionBased, Jaccard) | production ([delftdata/valentine](https://github.com/delftdata/valentine), pip-installable, MIT) | packaged access to COMA / Cupid / SF | `valentine>=0.4.1` pulls a JRE for the COMA wrapper; pure-Python for the other four | Not a new algorithm on its own — relevant as a *delivery channel* for classic matchers if we shortlist any. |
| [xbenchmatch_schema_matching](../../literature-search-generation/xbenchmatch_schema_matching/) | Benchmark + survey of COMA++, SF, YAM | paper-only (no code) | — | n/a | Provides comparison numbers (COMA++ ≈ 0.85 F1 on naming, degrades hard on combined transforms). No code release. |
| [jellyfish_llm_data_preprocessing](../../literature-search-generation/jellyfish_llm_data_preprocessing/) | Fine-tuned LLM for preprocessing (incl. SM) | research code | `llm` (second LLM path) | needs GPU + model weights | Currently slot 5 (`llm_openai`) covers the LLM axis with zero-shot. Jellyfish would be a fine-tuned alternative. |

## Not usable as committee members (but relevant for search seeds)

- [ibench](../../literature-search-generation/ibench/) — generates *matching scenarios*, not matchers.
- [alaska_benchmark](../../literature-search-generation/alaska_benchmark/) — benchmark.
- [xbenchmatch_schema_matching](../../literature-search-generation/xbenchmatch_schema_matching/) — benchmark + survey; numbers only.

## Gaps to investigate in C1.2

The following coverage axes are not represented by any portfolio anchor:

1. **Embedding-based label+instance matcher.** Sentence-transformer over column name + sample values. Modern baseline; Valentine's `EmbDI` family and Magneto's retrieval phase both use this shape.
2. **Learned column-pair classifier** (SMAT, Unicorn, SMAT-style Siamese on GloVe/BERT). Current roster has no supervised SM member.
3. **Hybrid aggregator (COMA-style multi-matcher ensemble).** The committee itself is already doing this aggregation via consensus → architectural redundancy risk if we add COMA as a single member. Evaluated explicitly in C1.3.
4. **Structural / graph-propagation** (Similarity Flooding, SMoG, Cupid's tree-similarity). PyDI sources are flat-tabular today; K3 (attribute_nesting) produces path-like column names but no true tree. Low expected payoff.
5. **LLM-retrieval hybrids other than Magneto** (LLMatch, SCHEMORA, ConStruM, ReMatch — all 2024–2025).

C1.2 runs the external-search queries listed in [plan_committee_finalization.md](../../plans/plan_committee_finalization.md#c1--schema-matching-committee) and scores the candidates in C1.3.
