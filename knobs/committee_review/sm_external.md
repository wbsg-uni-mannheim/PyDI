# SM Committee — External Literature Search (C1.2)

Targeted external search executed against the query set in [plan_committee_finalization.md §C1](../../plans/plan_committee_finalization.md#c1--schema-matching-committee). Ranking here is by fit-to-committee, not citation count.

## Candidates (ranked)

| # | Method / system | Venue | Code | Method class | Signal axis it would fill | Notes |
|---|---|---|---|---|---|---|
| 1 | **Magneto** (Liu et al.) | VLDB '25 | ✅ [VIDA-NYU/magneto-matcher](https://github.com/VIDA-NYU/magneto-matcher) (Python, MIT) | SLM-retrieval + LLM-rerank hybrid | `hybrid` (embedding + LLM) | SOTA on Valentine + new GDC benchmark. Paper card already exists at [literature-search-generation/magneto_schema_matching/](../../literature-search-generation/magneto_schema_matching/). |
| 2 | **COMA 3.0 CE** (Do & Rahm; Massmann et al.) | VLDB '02, CIKM '11 | ✅ [Valentine Coma wrapper](https://github.com/delftdata/valentine) wraps COMA-CE; requires JRE | Hybrid multi-matcher ensemble | `hybrid-ensemble` | Classic; see detailed evaluation in C1.3 exclusion rationale. New paper card at [literature-search-generation/coma_schema_matching/](../../literature-search-generation/coma_schema_matching/). |
| 3 | **Similarity Flooding** (Melnik, Garcia-Molina, Rahm) | ICDE '02 | ✅ Valentine pure-Python | Graph propagation | `structural` | Works on schema graphs; payoff on our flat-tabular sources is low. |
| 4 | **Cupid** (Madhavan, Bernstein, Rahm) | VLDB '01 | ✅ Valentine pure-Python | Tree-structural + linguistic | `structural` | Designed for hierarchical schemas; expects XML-like trees. Not a fit for CSV-style sources. |
| 5 | **DistributionBased** (Koutras et al.) | ICDE '21 | ✅ Valentine pure-Python | EMD over value distributions | `instance` (distinct from TF-IDF cosine) | Already covered by our `instance_tfidf_cosine`; EMD is a different similarity but same signal class. |
| 6 | **Unicorn** (Tu, Fan et al.) | SIGMOD '23 | ✅ [ruc-datalab/Unicorn](https://github.com/ruc-datalab/Unicorn) | Fine-tuned DeBERTa multi-task | `learned` | Needs gold training pairs per domain; overlaps with Ditto's role in EM. For SM we would need to carve training pairs from the gold mapping — feasible but expensive; deferred. |
| 7 | **SMAT** (Zhang et al.) | 2021 | research code | Siamese over GloVe/BERT | `learned` | Predates Unicorn + Magneto; both beat it. No reason to prefer over Unicorn. |
| 8 | **LLMatch** (Wang, Li et al.) | arxiv 2507.10897 / Springer 2025 | ✅ [knowledge-fusion/LLMatch](https://github.com/knowledge-fusion/LLMatch) (Python, pipenv; **MongoDB required**; OpenAI API) | LLM 3-stage framework (schema prep → table selection → rollup/drilldown column align) | `llm` | Re-verified 2026-04-20: code IS public (correction from earlier pass). Rejection now on integration cost (MongoDB hard dep, no PyPI) + redundancy with `llm_openai` and the incoming `magneto_slm_llm` (C1.5). |
| 9 | **SCHEMORA** (Gungor et al.) | arxiv 2507.14376, 2025 | ⚠️ paper cites [ermangungor/schemora](https://github.com/ermangungor/schemora) but the URL returns 404 and the user's public-repo list is empty as of 2026-04-20 | LLM + FAISS + BM25 retrieval with metadata enrichment | `llm` | Code release announced in paper but not actually reachable. Also: FAISS + BM25 + OpenAI embeddings stack is real integration work. Defer until repo appears. |
| 10 | **ConStruM** (Chen, Zhang, Jagadish) | arxiv 2601.20482, Jan 2026 | ❌ no code repo found (unrelated GH org `ConStrum` is Minecraft tooling; author's project page doesn't list it) | Add-on layer that augments an upstream matcher's LLM prompt with structured evidence (context tree + similarity hypergraph) | `llm + ensemble-aug` | Design is explicitly an "add-on" to an upstream matcher — redundant with our committee's aggregation layer. Rejection holds even when code lands. |
| 11 | **SMoG** (Schema Matching on Graph) | arxiv 2511.20285, Nov 2025 | ❌ no code found | Iterative 1-hop SPARQL exploration over knowledge graphs | `structural (KG)` | **Domain mismatch**: requires a SPARQL endpoint / knowledge graph; our sources are flat CSV / Parquet. Not applicable regardless of code availability. |
| 12 | **SBERT baseline** (sentence-transformers over column name + sample values, cosine) | n/a | ✅ `sentence-transformers` (already in PyDI core deps) | Embedding over label + instance | `embedding` | Standard modern baseline — no paper, just a well-trodden pattern (also how Magneto does retrieval). Fills the `embedding` gap at near-zero integration cost. |

## External queries executed

| Query | Purpose | Top takeaway |
|---|---|---|
| `"schema matching" "large language model" 2024..2026` | LLM-era methods | Magneto + LLMatch + SCHEMORA + ConStruM are the named works |
| `"Valentine" OR "Unicorn" OR "Magneto" schema matching` | Identify SOTA contenders with code | Magneto is the one with mature Python + active maintenance |
| `schema matching transformer pretrained 2024..2026` | Supervised PLM methods | Unicorn (SIGMOD '23) is the mature Python release; SMAT is older |
| `wbsg-uni-mannheim schema matching 2024 github` | Mannheim group's work | PyDI itself + TailorMatch (EM-focused) are the active repos; no standalone SM method |
| `COMA 3.0 schema matcher hybrid aggregation Do Rahm python` | COMA availability | Valentine Python package wraps Java COMA-CE; integration needs JRE |

## New paper cards added in C1.2

- [`literature-search-generation/coma_schema_matching/paper.md`](../../literature-search-generation/coma_schema_matching/) — Do & Rahm COMA family (2002 VLDB + 2005 COMA++ + 2011 3.0 evolution) with explicit evaluation for our committee.

No other new paper cards: Magneto already has a full card (see portfolio); the remaining candidates are either no-code arxiv preprints or documented by the Valentine suite rather than a new primary paper.

## Summary

The search surfaces one strong and one contested candidate:

- **Magneto** is the only external method that is both SOTA-2025 and has a maintained Python implementation. Gets a slot in the shortlist.
- **COMA** is the canonical "hybrid matcher" and the user asked explicitly for an evaluation — executed in [sm_shortlist.md](sm_shortlist.md).
- **SBERT baseline** (no named paper) fills the `embedding` axis at effectively zero integration cost since `sentence-transformers` is already a runtime dependency.

All other external candidates are excluded on integration-cost, axis-redundancy, training-data, or domain-fit grounds — rationale in [sm_shortlist.md](sm_shortlist.md).

## Re-verification pass (2026-04-20)

A deeper re-check of the four 2024–2026 preprints — at the user's request — updated the code-availability column above. Summary:

- **LLMatch**: public repo exists (was missed first pass). Still rejected, but now on **MongoDB dependency + signal redundancy** with `llm_openai` / incoming Magneto, not "no code."
- **SCHEMORA**: paper claims a release at `github.com/ermangungor/schemora` but the URL 404s and the author's public-repo list is empty. Treat as unavailable until that changes.
- **ConStruM**: no code repo confirmed. Note the architectural point: ConStruM is designed as an **add-on to an upstream matcher** — the same role our committee's aggregation layer already fills — so it would be architecturally redundant even with code.
- **SMoG**: requires a SPARQL / knowledge-graph endpoint; PyDI sources are flat tabular. Rejection ground is domain fit, not code availability.

Net effect on the roster: none. LLMatch is the only one whose rejection ground was factually wrong; the revised grounds (integration cost + redundancy) independently yield the same verdict.
