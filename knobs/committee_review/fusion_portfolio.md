# Fusion Committee — Portfolio Inventory (C3.1)

Committee in scope: data fusion. Unlike the SM and EM committees, the fusion committee is **per-attribute-type**, not a flat roster — each attribute gets its own 2-4 strategy slate. Current roster (see [../../usecases_synthetic/config/committees/fusion_committee.yaml](../../usecases_synthetic/config/committees/fusion_committee.yaml)):

| Attribute | Class | Strategies (all `PyDI.fusion.conflict_resolution.*`) |
|---|---|---|
| `name` | primary (string) | `voting`, `longest_string`, `most_complete`, `prefer_higher_trust` |
| `assets` | secondary (numeric) | `median`, `maximum`, `prefer_higher_trust` |
| `revenue` | secondary (numeric) | `median`, `maximum`, `prefer_higher_trust` |
| `keypeople` | secondary (list) | `union`, `voting`, `prefer_higher_trust` |
| `founded` | secondary (date) | `voting`, `earliest`, `prefer_higher_trust` |
| `country` | key (categorical) | `voting`, `favour_forbes` (via `favour_sources`), `prefer_higher_trust` |
| `city` | key (string) | `voting`, `shortest_string`, `prefer_higher_trust` |
| `industry` | key (categorical) | `voting`, `most_complete`, `prefer_higher_trust` |

The per-attribute-type strategy split matches the draft guidance in [../cross_cutting.md §Committee composition (fusion, draft)](../cross_cutting.md): strings → `longest_string` / `shortest_string` / `most_complete` / `voting`; numerics → `median` / `maximum`; categoricals → `voting`; lists → `union` / `voting`; dates → `earliest`.

Axes covered today (per `required_axes` in the YAML): `cell_local`, `trust_weighted`. Missing per C3's required-axis block: `truth_discovery`, `llm_adjudicated`, and (informally) `entity_aware` (cross-cell consistency, e.g. `country=USA` constraining `city`).

## Portfolio anchors (existing `literature-search-generation/` entries)

**Finding: zero directly-reusable entries.** A grep over `literature-search-generation/` for `data fusion` / `truth discovery` / `conflict resolution` / `TruthFinder` / `latent truth` / `Xin.*Dong` returns no paper whose primary method class is fusion. The only hits are incidental mentions inside benchmark / schema-matching papers (e.g. [ibench](../../literature-search-generation/ibench/) exposes a `target_reuse` knob that "tests data fusion capabilities" but provides no fusion algorithm; [embench_pp_benchmark](../../literature-search-generation/embench_pp_benchmark/) mentions a "configurable conflict resolution strategy" during entity merge but ships none). The two survey anchors the plan points at are included below as **marginal** — they touch fusion in passing but do not yield an implementable committee member.

| Anchor | Method class | Maturity | Axis it would fill | Dependency cost | Note |
|---|---|---|---|---|---|
| [christophides_er_survey](../../literature-search-generation/christophides_er_survey/) | End-to-end ER survey (ACM Comput. Surv. 2020) | paper-only (survey) | — | n/a | Taxonomy of the ER pipeline (blocking / matching / clustering); "data fusion" appears only as a downstream stage mentioned once. No drop-in method. Useful as a citation seed for C3.2's backward reference graph. |
| [heterogeneity_em_survey](../../literature-search-generation/heterogeneity_em_survey/) | EM heterogeneity survey (DKE 2025) | paper-only (survey) | — | n/a | Taxonomy of representation vs. semantic heterogeneity for EM; does not discuss fusion strategies. Same caveat: seed only. |

No `literature-search-generation/` entry is directly usable as a committee member. **C3.2 must carry all the algorithmic weight.**

## Not usable as committee members

Everything else under [../../literature-search-generation/](../../literature-search-generation/) — ~75 papers on tabular generation, corruption, schema matching, EM, contamination, benchmarking — sits in adjacent pipeline stages and does not contribute a fusion strategy. No exhaustive enumeration; the absence is the finding. Notable mentions that *sound* fusion-adjacent but are not: [ibench](../../literature-search-generation/ibench/) (generates mappings that *stress* fusion), [embench_pp_benchmark](../../literature-search-generation/embench_pp_benchmark/) (merge-with-conflict during entity-evolution simulation), [ground_truth_weakly_supervised_em](../../literature-search-generation/ground_truth_weakly_supervised_em/) (transitivity-enforcing labeling model — relevant to EM ground truth, not multi-source value fusion).

## Gaps to investigate in C3.2

C3.2 must source every candidate externally. Per [../../plans/plan_committee_finalization.md §C3](../../plans/plan_committee_finalization.md#c3--data-fusion-committee) the coverage gaps are:

1. **Truth-discovery / probabilistic fusion.** TruthFinder (Yin/Han/Yu 2008), LTM (Zhao et al. 2012), Accu / AccuNoDep / AccuSim (Dong et al. VLDB 2009–2012), latent-truth models, CATD/CRH families. Score sources by accuracy and jointly infer the fused value — a learned analogue of `prefer_higher_trust`. SOTA for multi-source conflict; fills the `truth_discovery` axis.
2. **LLM-driven conflict resolution (`llm_adjudicated` axis).** Recent (2024–2026) papers on using LLMs as judges over conflicting cells, especially where semantic equivalence matters ("NYC" / "New York" / "New York City"). Priority: surface a concrete per-cell LLM arbitration method with a published prompt template and cost profile. Mannheim WBSG repos and VLDB/SIGMOD 2024–2026 are the likely source.
3. **Entity-aware fusion.** Current committee is strictly cell-local + trust-weighted; entity-aware means using cross-cell consistency constraints (e.g., `country=USA` constrains `city` to US cities). Relevant for the K10 signal and for joint entity-and-value resolution papers.
4. **Robust numeric aggregators.** Median/maximum are in; trimmed mean, Huber M-estimator, winsorised mean, distribution-aware aggregators for skewed attributes (assets, revenue). Low integration cost (pure numpy/scipy), high diversity payoff vs. the existing `median` + `maximum` pair.
5. **Source-trust learning from gold.** Semi-supervised trust estimation to replace the hardcoded `trust_scores: {forbes: 3, fullcontact: 2, dbpedia: 1}` block in the YAML. Classical TruthFinder/Accu jointly learn trust — overlap with gap 1; called out separately here because the user may want a pure trust-learner even without value fusion.
6. **Holistic / joint entity-and-value resolution.** Papers that fuse value conflicts and entity clustering simultaneously (2022–2026). Would fill a sixth axis not in the plan's required list but plausibly useful for the `companies` use case where `industry` / `country` drift with entity identity.

**External search queries from the plan** (verbatim): `"data fusion" "conflict resolution" survey 2020..2026`; `"truth discovery" data integration 2023..2026`; `"TruthFinder" OR "LTM" OR "Accu" OR "latent truth" data fusion`; `"large language model" data fusion OR conflict resolution 2024..2026`; `"multi-source data integration" trust 2023..2026`; `"holistic data fusion" OR "joint entity and value resolution" 2022..2026`; `Dong Srivastava data fusion` (Xin Luna Dong's group is the anchor author); `robust aggregation numeric conflict data integration`.

C3.2 runs those queries against Google Scholar + VLDB/SIGMOD/EDBT/WWW 2023–2026 and the Mannheim WBSG GitHub org, then scores candidates in C3.3.
