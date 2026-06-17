# Fusion Committee — Scored Shortlist (C3.3)

> **USER DECISION (2026-04-21) — supersedes the per-attribute roster proposed below.**
>
> **Fusion committee — fixed to 7 members** (expanded from 5 on 2026-04-21 after diversity review flagged overlap between FusionQuery / TruthFinder / AccuSim; LTM and CASE added as distinct paradigms, both free from the same Apache-2.0 vendor repo):
> 1. **`fusionquery`** — vendor [JunHao-Zhu/FusionQuery](https://github.com/JunHao-Zhu/FusionQuery) (Apache-2.0, last commit 2025-03-11). Strip `sentence-transformers` / FAISS deps (matching-phase only); keep the numpy EM-style fusion core.
> 2. **`truthfinder`** — port from the same FusionQuery repo's [`fusion/baseline.py`](https://github.com/JunHao-Zhu/FusionQuery/blob/main/fusion/baseline.py) (Apache-2.0 `class TruthFinder`, ~85 LoC). Heuristic iterative mutual reinforcement (Yin et al. 2007). Free side effect of vendoring FusionQuery.
> 3. **`ltm`** — port from the same repo's [`fusion/baseline.py`](https://github.com/JunHao-Zhu/FusionQuery/blob/main/fusion/baseline.py) (Apache-2.0 `class LTMFusion`, ~90 LoC). Bayesian graphical model via Gibbs sampling (Zhao et al. 2012). **Multi-truth native** — handles attributes where multiple values are simultaneously correct (lists like `keypeople`). Free side effect of the vendor.
> 4. **`casefusion`** — port from the same repo's [`fusion/baseline.py`](https://github.com/JunHao-Zhu/FusionQuery/blob/main/fusion/baseline.py) (Apache-2.0 `class CASEFusion`, ~120 LoC). **Unsupervised graph-embedding TD** via SGD on a source-candidate graph + source-source similarity graph. Random-init embeddings (no labels, no pretrained model) — paradigm distinct from iterative-mutual-reinforcement / Bayesian / EM / alternating-minimization families. Free side effect of the vendor. Seeded `np.random.seed(42)` for determinism.
> 5. **`accusim`** — paper-reimplementation from [Dong et al. VLDB 2009](../../literature-search-generation/accu_source_dependence/) (~80 LoC). Reference implementations at [MengtingWan/KDEm](https://github.com/MengtingWan/KDEm) + [daqcri/DAFNA-EA](https://github.com/daqcri/DAFNA-EA) serve as correctness oracles but aren't vendored (no LICENSE in either). Adds Bayesian source-accuracy + value-similarity voting to complement TruthFinder.
> 6. **`llm_as_judge`** — new in-repo adapter. Prompted LLM arbitration on conflicts that TD + cell-local can't resolve (semantic equivalence: "NYC" / "New York City"). Cached, budget-capped.
> 7. **`robust_aggregators`** — new adapter built on `scipy.stats` (trimmed mean, Huber M-estimator, median-of-means). Covers numerical primary attributes (`assets`, `revenue`); substitutes for KDEm's numerical-TD axis with deterministic + MIT-compatible code.
>
> **All 5 TD methods (FusionQuery, TruthFinder, LTM, CASE, AccuSim) are unsupervised** — no labels, no training corpus, no pretrained checkpoints. CASE uses SGD as its optimizer but its loss is purely structural (SC-loss + SS-loss over the source-claim graph); runs per-fusion-call with random init.
>
> **Vendoring strategy:** one vendor operation brings in FusionQuery + TruthFinder + LTM + CASE under `usecases_synthetic/third_party/fusionquery/` (Apache-2.0). Strip `sentence-transformers` / FAISS (matching-phase deps only); keep pure numpy/scipy core. AccuSim is paper-reimplemented; robust_aggregators and llm_as_judge are in-house.
>
> **Dropped vs C3.3 scored shortlist:** CRH (13), CATD (12), KDEm (11). Rationale:
> - **KDEm dropped** — author repo has no LICENSE *and* the author explicitly disowns the code ("significantly out-of-date, consult the paper or other algorithms"). No other implementation exists. Its numerical-TD role is absorbed by `robust_aggregators` + FusionQuery's heterogeneous-type handling.
> - **CRH / CATD dropped** — once LTM + CASE + FusionQuery + AccuSim + TruthFinder are in, the optimization-framework niche CRH/CATD would fill is no longer differentiating. CRH confirmed unsupervised (alternating minimization; no labels required) — dropped on redundancy not on paradigm concerns.
>
> **Paradigm coverage (5 distinct TD paradigms + 2 non-TD):**
> - Iterative mutual reinforcement → TruthFinder, AccuSim
> - EM-style joint estimation → FusionQuery
> - Bayesian graphical model + Gibbs → LTM
> - Unsupervised graph-embedding via SGD → CASE
> - LLM arbitration → LLM-as-Judge
> - Robust statistics (non-TD) → robust_aggregators
>
> **Coverage check:**
> - `strategy_type` axes populated: `truth_discovery` (fusionquery, truthfinder, ltm, casefusion, accusim), `llm_adjudicated` (llm_as_judge), `cell_local` (robust_aggregators + existing voting/median/longest_string incumbents).
> - Per-attribute-type coverage: primary strings (TruthFinder, AccuSim, FusionQuery, CASE, LLM-as-Judge) / numerical (robust_aggregators + FusionQuery) / categoricals (TruthFinder, AccuSim, FusionQuery, CASE) / lists + multi-truth (LTM) / keys (LLM-as-Judge, AccuSim value-sim).
> - Deterministic fallback per attribute: incumbents (`voting`, `median`, `longest_string`, etc.) remain as deterministic fallbacks alongside the new TD methods. Of the 5 TD methods, CASE and LTM are seed-dependent; the other three are deterministic once initial conditions are fixed.
>
> **Relaxed-licensing stance (user-directed, 2026-04-21):** for the fusion committee the user accepts reference code "available in some form" — includes paper-to-code reimplementations and unlicensed author-repo references. Not a blanket repo-wide policy; applies to this roster.
>
> Proceed to C3.4 with this fixed roster. The scored shortlist below is preserved as the evidence trail that informed the final choice.

---

Scoring follows the rubric in [plan_committee_finalization.md §Step (iii)](../../plans/plan_committee_finalization.md#step-iii--candidate-shortlist--scoring-rubric). Five axes, 0–3 each; committee-slot cutoff is total ≥ 10 with no axis scoring 0. Candidate pool = portfolio ([fusion_portfolio.md](fusion_portfolio.md)) ∪ external ([fusion_external.md](fusion_external.md)). Zero directly-reusable portfolio anchors exist — all shortlisted candidates come from the external search that produced the nine new `paper.md` cards in [C3.2](fusion_external.md#new-paper-cards-added-in-c32).

Unlike C1/C2, the fusion committee is **per-attribute-type** — each attribute gets its own 2-4 strategy list, not a flat roster. The scoring below ranks the new candidates; the §Proposed per-attribute-type roster section then maps them into per-attribute slots alongside the existing incumbents. **C3.4 is user-directed**: this file is advisory evidence, the user picks the roster, the implementer wires it. Two special cases flagged up front: (a) **LLM-as-judge** has no external runnable code and requires a from-scratch PyDI adapter — a distinct user decision between accepting implementation cost versus deferring Gap 2; (b) the six truth-discovery candidates are partial substitutes for each other and the user needs to pick the 2-3 shipped, not all of them.

## Scoring matrix

| # | Candidate | Method class | Strategy type | Attribute applicability | Signal diversity | SOTA alignment | Integration cost | Determinism | Runtime fit | **Total** | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | [TruthFinder](../../literature-search-generation/truthfinder_fusion/) | iterative mutual-reinforcement TD (heuristic) | truth_discovery | strings, categoricals (native); lists via per-value reduction | 3 | 1 | 2 | 3 | 3 | **12** | **Shortlist** |
| 2 | [Accu / AccuSim](../../literature-search-generation/accu_source_dependence/) | Bayesian point-estimate TD + copying detection | truth_discovery | strings, categoricals; AccuSim adds value-similarity | 2 | 2 | 2 | 3 | 3 | **12** | **Shortlist** |
| 3 | [LTM](../../literature-search-generation/ltm_latent_truth_model/) | Bayesian graphical model + Gibbs sampling | truth_discovery | categoricals, **lists (multi-truth native)** | 3 | 2 | 2 | 2 | 1 | **10** | **Shortlist (conditional — user)** — multi-truth natively covers `keypeople`; Gibbs-sampling runtime at 100 iterations × 2 k rows is the risk |
| 4 | [CRH](../../literature-search-generation/crh_conflict_resolution/) | optimisation-framework TD, type-specific losses | truth_discovery | strings, numerics, categoricals (heterogeneous by design) | 3 | 2 | 2 | 3 | 3 | **13** | **Shortlist** |
| 5 | [CATD](../../literature-search-generation/catd_confidence_aware_td/) | CRH + chi-squared confidence-interval weights | truth_discovery | strings, numerics, categoricals (same as CRH) | 2 | 2 | 2 | 3 | 3 | **12** | **Shortlist (alt. to CRH)** — partial substitute for CRH |
| 6 | [KDEm](../../literature-search-generation/kdem_kernel_density_fusion/) | kernel-density source-weighted fusion | truth_discovery | **numerics (primary)**; multi-modal distributions | 3 | 2 | 2 | 2 | 2 | **11** | **Shortlist** |
| 7 | [FusionQuery](../../literature-search-generation/fusionquery_ondemand/) | EM-style on-demand TD (Apache-2.0, 2024) | truth_discovery | all types (heterogeneous) | 3 | 3 | 2 | 3 | 3 | **14** | **Shortlist** |
| 8 | [Robust aggregators](../../literature-search-generation/robust_aggregators_fusion/) | scipy/statsmodels trimmed mean / Huber / med-of-means | cell_local | **numerics only** | 3 | 1 | 3 | 3 | 3 | **13** | **Shortlist** |
| 9 | LLM-as-judge (in-repo adapter) | prompted LLM arbitration (no external code) | llm_adjudicated | strings, categoricals (semantic equivalence) | 3 | 2 | 1 | 2 | 1 | **9** | **Defer / user-directed** — falls below the 10 cutoff on runtime + integration; the user may still opt in if Gap 2 coverage is worth the implementation cost |
| — | [IEEE TBD 2025 Survey](../../literature-search-generation/survey_truth_discovery_ieee2025/) | survey | n/a | n/a | — | — | — | — | — | — | **Not scored** — reference only |

### Axis-score notes

All nine candidates have at least one MIT- or Apache-2.0-licensed Python reference implementation (see [fusion_external.md §Candidates](fusion_external.md#candidates-ranked-by-gap-fit--code-maturity)), so none of them scores 0 on Integration cost on the licence axis alone. The scoring below isolates each axis against the specific reference impl surfaced in C3.2.

- **TruthFinder** scores 1 on SOTA alignment: every 2023+ survey cites it as the canonical reference baseline, but it is superseded by the optimisation and Bayesian variants on every recent benchmark. Signal-diversity 3 because no incumbent covers the source-accuracy-weighted-voting axis. Runtime 3: pure-numpy loop, 10-50 iterations, O(sources × cells). Determinism 3: deterministic once the initial `tau` is fixed.
- **Accu / AccuSim** — base Accu scores 2 on SOTA (principled Bayesian update is competitive with CRH/CATD); signal-diversity 2 because its base Accu variant overlaps TruthFinder (AccuCopy's copying detection would be 3, but our 3-source companies setup has no known copying). Runtime 3, determinism 3 (closed-form posterior update).
- **LTM** — signal-diversity 3 for multi-truth support (critical for `keypeople`). Determinism 2: Gibbs sampler is **seeded but not fully deterministic under different NumPy versions**; use a pinned `np.random.default_rng(seed=42)`. Runtime 1: 100 Gibbs iterations × 50 burn-in on ~2 k rows × ~8 attributes × 3 sources ≈ 2.4 M sweeps, in python ~10-60 s per attribute — tolerable but not cheap. The flag in the verdict reflects this.
- **CRH** — signal-diversity 3: heterogeneous-data optimisation framework is a distinct signal class from voting-flavoured TruthFinder/Accu; SOTA 2 (still referenced by the 2025 IEEE TBD survey as one of the top-4 baselines on mixed-type benchmarks). Fully deterministic alternating optimisation — determinism 3, runtime 3.
- **CATD** — scored separately because it and CRH are partial substitutes. Signal-diversity 2 (same optimisation framework as CRH, only the source-weight update differs). SOTA 2 (long-tail-specific; our companies setup has only 3 sources so the long-tail benefit is muted). The user has to pick one of CRH/CATD — shipping both is score-redundant.
- **KDEm** — signal-diversity 3: kernel density is a non-parametric, multi-modal-aware aggregator unlike any incumbent. SOTA 2 (canonical continuous-data TD method in the 2025 survey). Determinism 2: deterministic *given fixed bandwidth*, but Silverman's-rule default bandwidth is data-dependent — log the bandwidth for reproducibility. Runtime 2: ~30 iterations of weighted-KDE updates per attribute; scipy KDE is O(sources²) per object which is fine at our scale but not for a very wide source set.
- **FusionQuery** — the best-scoring candidate. SOTA 3 (PVLDB 2024, 10-50× faster than Gibbs, includes TruthFinder/Accu/CRH/LTM as internal baselines). Signal-diversity 3 (on-demand EM is the only method that scales to per-query fusion). Apache-2.0 licence + ~500-line numpy reference implementation means Integration 2 (new adapter, but small). Determinism 3 (closed-form EM, no sampling). Runtime 3 on our scale.
- **Robust aggregators** — total 13 despite SOTA 1: scipy/statsmodels ship these as drop-in (Integration 3), fully deterministic, zero dependencies beyond what we have. Signal-diversity 3 for numerics specifically because `median` and `maximum` are the only existing numeric members — trimmed mean / Huber / median-of-means provide distinct robustness properties (trim handles symmetric outliers; Huber handles heavy tails; med-of-means is heavy-tail-optimal). The low SOTA score reflects the 1964 vintage; doesn't matter for our purpose.
- **LLM-as-judge** — signal-diversity 3 (only route to semantic equivalence reasoning: "NYC" / "New York" / "New York City" collapse); SOTA 2 (active 2024-2026 research area, no winner yet). **Integration cost 1** — no external code; need to write `usecases_synthetic/lib/llm_judge_fusion.py` matching the `MatchGPT` / `ComEM` prompt pattern, with structured-output JSON schema and a file-backed prompt cache. **Determinism 2** (temperature=0 + pinned model + hashed cache gets us seeded/stable, but LLM drift across model revisions is a known flaky pattern). **Runtime 1** — per-cell LLM call, cost-bounded by `cross_cutting.md §LLM hygiene`. Total 9 — below the 10 cutoff; shortlist status depends on whether the user accepts the implementation cost in this cycle (see Open questions).

## Per-candidate rationale

### Included

- **TruthFinder.** Fills Gap 1 (canonical TD) at minimal integration cost. The KDEm port (MIT) has a 100-line `TruthFinder.py` that can be wrapped as a cell-level adapter. Score 12 clears the cutoff. Included as a **string/categorical** TD member; it is the most pedagogically familiar choice and every fusion survey cites it as baseline, so the reviewer cost of explaining why it is present is near zero.
- **Accu.** Same MIT port. Bayesian posterior update is a distinct signal from TruthFinder's heuristic logistic, and AccuSim's value-similarity extension is exactly what we want for name-like attributes where "Apple Inc." and "Apple Inc" should reinforce each other. Score 12. Choose one of TruthFinder/Accu for the first shipped TD member; shipping both is score-redundant (see Open questions).
- **CRH.** Score 13, the highest of the classical family. The heterogeneous-data framework matters because one fusion strategy can serve strings (0-1 loss → weighted majority), numerics (L1 loss → weighted median), and categoricals (same weighted majority) without swapping strategies per attribute. This is the **single most versatile** TD member; if the user picks only one TD method, CRH is the default recommendation.
- **CATD.** Alternative to CRH with chi-squared-CI source weights. Score 12. Only edges out CRH when the source-claim counts are highly unequal (long-tail). Our companies setup has 3 balanced sources — CATD's benefit is muted. Flagged as **alternative**: ship CRH *or* CATD, not both.
- **LTM.** Score 10, just at the cutoff; determinism 2 and runtime 1 are the risks. Included because it is **the only shortlisted method with native multi-truth support**, which directly matches the `keypeople` attribute class (a company has multiple key people; voting + union are the current strategies, neither of which jointly estimates source reliability on multi-valued cells). If the user accepts the Gibbs runtime (see Open questions), LTM is the `keypeople` TD member; otherwise fall back to CRH with a post-hoc per-source union.
- **KDEm.** Score 11, the canonical numeric / multi-modal TD method. Fills the numeric truth-discovery slot for `assets` / `revenue` where robust aggregators (below) cover the non-learned path and KDEm covers the source-reliability-aware path. Dependencies already satisfied (scipy). MIT-licensed reference implementation in the KDEm repo.
- **FusionQuery.** Score 14, highest of the pool. Apache-2.0, Python/numpy, 2024 — the **only modern (post-2022) candidate with a permissive licence and a runnable implementation**. Fills Gap 1 + Gap 3 (on-demand + cross-cell) simultaneously and internally benchmarks against TruthFinder / Accu / CRH / LTM, meaning it can be swapped in for any of them without losing baseline coverage. Strongest argument for including FusionQuery is coverage: if shipped, the committee has a defensible "we use a 2024 SOTA method" answer to any reviewer question.
- **Robust aggregators.** Score 13. Not a truth-discovery method — purely cell-local, closed-form numeric aggregators via scipy / statsmodels. Fills the numeric robustness gap the plan explicitly calls out (`assets` / `revenue` should gain "trimmed mean or similar"). Zero integration cost; deterministic; should be shipped regardless of whether a numeric TD method (KDEm / CRH) is included. The plan's §C3 target specifically names this as required, so this is the **non-negotiable** numeric addition.

### Excluded or deferred

- **CATD deferred as alternative to CRH.** Not strictly excluded — its score clears the cutoff — but shipping CRH and CATD together is score-redundant (both SIGMOD/PVLDB 2014 optimisation-framework, same KDEm port, same loss-function catalog). User picks one. Default recommendation: CRH for broader heterogeneous coverage; CATD if the user later finds the companies setup long-tail-dominated.
- **LLM-as-judge (in-repo adapter).** Score 9 — below the 10 cutoff on Integration (1) and Runtime (1). Not excluded on technical grounds; excluded on **implementation budget** grounds. The axis it fills (semantic equivalence / `llm_adjudicated`) is genuinely absent from the committee today and from every alternative candidate. **Deferral rationale:** C3.4 is scoped to wiring existing components into `fusion_committee.yaml`; writing a new LLM adapter is a C3.4+ task. The user may override and build it now — see Open questions. The design sketch, if implemented, would mirror [`magneto_sm_matcher.py`](../../usecases_synthetic/lib/magneto_sm_matcher.py)'s two-phase cache-then-prompt structure: for each conflicting cell, emit a structured-output JSON request listing the (source, value) tuples + surrounding entity context, ask the LLM to return the canonical value + a confidence, and gate with temperature=0 + `sha256(prompt_version|model|prompt_text)` cache. Budget cap consumed from `cross_cutting.md §LLM hygiene`.
- **IEEE TBD 2025 Survey.** Not scored — reference only per the plan. The survey card is kept in `literature-search-generation/` as the taxonomy citation for the rollup document in C3.4.

## Proposed per-attribute-type roster

Each attribute class gets a 2-4-strategy list mixing **incumbents** (already shipped in `fusion_committee.yaml`, kept unless explicitly replaced) with **new members** from the shortlist. The plan's [§C3 required axis coverage](../../plans/plan_committee_finalization.md#c3--data-fusion-committee) mandates: `cell_local` + `trust_weighted` (present) + `truth_discovery` (new) and/or `llm_adjudicated` (new). At least one truth-discovery strategy on primary-class attributes; at least one robust aggregator on numerics; deterministic fallback per attribute preserved.

### Strings (primary: `name`; key: `city`)

| Attribute | Incumbents (keep) | New member(s) | Axes covered |
|---|---|---|---|
| `name` (primary) | `voting`, `longest_string`, `most_complete`, `prefer_higher_trust` | + **CRH** (or TruthFinder / Accu as alternatives) | cell_local, trust_weighted, **truth_discovery** |
| `city` (key) | `voting`, `shortest_string`, `prefer_higher_trust` | + **CRH** (string loss) or optionally LLM-as-judge for "NYC"↔"New York" | cell_local, trust_weighted, **truth_discovery** (+ optional llm_adjudicated) |

### Numerics (secondary: `assets`, `revenue`)

| Attribute | Incumbents (keep) | New member(s) | Axes covered |
|---|---|---|---|
| `assets` (secondary) | `median`, `maximum`, `prefer_higher_trust` | + **trimmed_mean** (robust aggregator, scipy) + **KDEm** (truth_discovery, numeric) | cell_local, trust_weighted, **truth_discovery** |
| `revenue` (secondary) | `median`, `maximum`, `prefer_higher_trust` | + **trimmed_mean** + **KDEm** | cell_local, trust_weighted, **truth_discovery** |

Trimmed mean is the plan's named default for robust numeric; KDEm is the source-reliability-aware numeric TD. Both are low-cost additions and complementary (trimmed mean is closed-form; KDEm is data-adaptive).

### Categoricals (key: `country`, `industry`)

| Attribute | Incumbents (keep) | New member(s) | Axes covered |
|---|---|---|---|
| `country` (key) | `voting`, `favour_forbes`, `prefer_higher_trust` | + **CRH** (categorical 0-1 loss) | cell_local, trust_weighted, **truth_discovery** |
| `industry` (key) | `voting`, `most_complete`, `prefer_higher_trust` | + **CRH** | cell_local, trust_weighted, **truth_discovery** |

### Lists (secondary: `keypeople`)

| Attribute | Incumbents (keep) | New member(s) | Axes covered |
|---|---|---|---|
| `keypeople` (secondary) | `union`, `voting`, `prefer_higher_trust` | + **LTM** (multi-truth native) OR fall back to per-value CRH if LTM runtime is rejected | cell_local, trust_weighted, **truth_discovery** |

`keypeople` is the only attribute where LTM's multi-truth Gibbs is strictly the right model; if the user drops LTM for runtime reasons, CRH applied per candidate person (with 0-1 loss) is the acceptable fallback that still fills the TD axis.

### Dates (secondary: `founded`)

| Attribute | Incumbents (keep) | New member(s) | Axes covered |
|---|---|---|---|
| `founded` (secondary) | `voting`, `earliest`, `prefer_higher_trust` | none (optional: **FusionQuery**'s EM can consume `founded` if shipped committee-wide) | cell_local, trust_weighted |

Dates are the lowest-priority attribute class — the plan does not require TD on dates. Skip unless FusionQuery ships committee-wide (in which case `founded` can inherit it for free).

### Cross-attribute option — FusionQuery

If the user selects **FusionQuery** as the single modern TD method, it can replace the per-attribute CRH/KDEm/LTM picks with a single on-demand EM pass across all attributes simultaneously. This is the **lowest-maintenance option**: one adapter, one member per attribute, Apache-2.0 licence, 2024 SOTA. Trade-off: loses multi-truth-native semantics for `keypeople` and multi-modal KDE semantics for numerics. The plan's required-axis coverage is still met (one `truth_discovery` strategy per primary attribute).

### Deterministic-fallback audit

Each attribute's first two strategies must be deterministic (no LLM, no sampling). The proposed roster preserves this:

- `name`: `voting` + `longest_string` (deterministic) → CRH (deterministic alternating optimisation) — OK.
- `assets` / `revenue`: `median` + `maximum` (deterministic) → `trimmed_mean` (deterministic) + KDEm (deterministic given fixed bandwidth — pin Silverman+seed) — OK.
- `keypeople`: `union` + `voting` (deterministic) → LTM is **seeded but not strictly deterministic** (flag). Fallback (CRH per-value) is fully deterministic.
- `country` / `industry`: `voting` + `favour_*`/`most_complete` (deterministic) → CRH (deterministic) — OK.
- `city`: `voting` + `shortest_string` (deterministic) → CRH (deterministic) — OK.
- `founded`: `voting` + `earliest` (deterministic) — OK.

### Plan-constraint audit

Checking the proposed roster against the plan's [§C3 required-axis block](../../plans/plan_committee_finalization.md#c3--data-fusion-committee):

- **At least one truth-discovery strategy on primary-class attributes.** `name` is the only primary-class attribute; CRH is proposed there. PASS.
- **At least one robust aggregator on numerics.** Trimmed mean proposed on both `assets` and `revenue` (+ KDEm as the TD numeric). PASS.
- **Axis coverage adds `truth_discovery`.** CRH / KDEm / LTM all tagged `strategy_type: truth_discovery`. PASS.
- **Axis coverage adds `llm_adjudicated`.** Only satisfied if the user opts into building LLM-as-judge. PENDING user decision (Open question 3).
- **Deterministic fallback per attribute preserved.** Audited above — PASS (LTM flagged as seeded-but-not-strict; CRH fallback available).

### Risks flagged

- **GPL-3.0 boundary.** The `truthdiscovery` PyPI package ([joesingo/truthdiscovery](https://github.com/joesingo/truthdiscovery)) is GPL-3.0. Vendoring the MIT-licensed [KDEm](https://github.com/MengtingWan/KDEm) port instead keeps PyDI permissive. All TD methods in the shortlist (TruthFinder, Accu, LTM, CRH, CATD, KDEm) have MIT reference implementations inside the KDEm bundle — vendor from there.
- **LTM Gibbs wall-clock.** 100 iterations × 50 burn-in × ~2 k rows × ~3 sources per attribute ≈ tens of seconds per attribute in Python. Acceptable for the `companies` committee runner but worth measuring before committing. If LTM is dropped, fall back to CRH-per-value for `keypeople`.
- **LLM-as-judge budget.** If the user opts into the in-repo adapter, per-cell LLM calls × affected attributes ("name", "city", possibly "industry") × ~2 k rows = thousands of prompts per committee run. Pin temperature=0, cache by `sha256(prompt_version|model_id|prompt_text)`, and budget-gate via `--with-llm` + `cross_cutting.md §LLM hygiene` cost cap.
- **FusionQuery adapter as cross-attribute member.** If shipped committee-wide, all attributes share the same EM run; a bug in the adapter affects every attribute simultaneously. Mitigate with a per-attribute smoke test in C3.4.

## Open questions for the user

The following decisions cannot be made by the rubric alone — they are C3.4 inputs:

1. **Which 2-3 TD methods ship?** The six TD candidates (TruthFinder, Accu, LTM, CRH, CATD, FusionQuery) are partial substitutes. Default recommendation based on scores: **CRH + KDEm + FusionQuery** — one classical heterogeneous baseline, one numeric specialist, one modern SOTA. Alternative minimal: **CRH only** (covers strings / numerics / categoricals via loss-function switch, lowest adapter count). Alternative max: **CRH + KDEm + LTM + FusionQuery** (adds multi-truth for `keypeople` at the cost of a Gibbs-sampler runtime risk).
2. **KDEm MIT port vs `truthdiscovery` GPL-3.0 PyPI — which source?** Recommendation: vendor from [MengtingWan/KDEm](https://github.com/MengtingWan/KDEm) (MIT) under `PyDI/fusion/conflict_resolution/td/` with an `ORIGIN.md` like the Magneto vendoring pattern. Accept dependency-add of numpy+scipy only (already in pyproject). GPL-3.0 PyPI path would require a licence-boundary review and is not recommended.
3. **Build LLM-as-judge this cycle or defer to Phase C?** Score 9 (below cutoff) on integration + runtime axes. If built now, estimated effort: one adapter file (~200 LOC) modelled on [`magneto_sm_matcher.py`](../../usecases_synthetic/lib/magneto_sm_matcher.py) + a `usecases_synthetic/cache/llm_judge_fusion/` prompt cache. If deferred, Gap 2 remains open and the committee's `llm_adjudicated` axis stays empty. Recommendation: **defer unless S2 evaluation shows semantic-equivalence failures that LLM-judge would demonstrably fix**.
4. **LTM Gibbs runtime acceptable?** 100 × 50 × 2 k × 3 ≈ 30 M sweeps per attribute ≈ 10-60 s in pure numpy. If unacceptable, drop LTM and cover `keypeople` via CRH-per-value (deterministic, ~100 ms). Recommendation: **measure before deciding**; run a one-off benchmark on `companies-small` during C3.4 smoke test.
5. **FusionQuery as single-shipped-modern-member vs. classical-only roster?** Including FusionQuery buys the "post-2022 SOTA" story at the cost of one more adapter. Excluding it keeps the committee lean but means every shipped member is pre-2016. Recommendation: **include** — Apache-2.0, single-file adapter, internally benchmarks against every other TD method, 2024 vintage matches the committee-level SOTA bar set by Magneto (C1.5) and Ditto (EM).

Once the user resolves these five questions, C3.4 can wire the selected members into `fusion_committee.yaml` + add adapters under `PyDI/fusion/conflict_resolution/td/` (promoted to core per the plan, since fusion is not synthetic-specific) + rerun `test_committee_configs.py` + run the per-attribute smoke test on `companies-small`.
