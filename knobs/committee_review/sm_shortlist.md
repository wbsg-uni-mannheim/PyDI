# SM Committee — Scored Shortlist (C1.3)

Scoring follows the rubric in [plan_committee_finalization.md §Step (iii)](../../plans/plan_committee_finalization.md#step-iii--candidate-shortlist--scoring-rubric). Five axes, 0–3 each; committee-slot cutoff is total ≥ 10 with no axis scoring 0.

Candidate pool = portfolio ([sm_portfolio.md](sm_portfolio.md)) ∪ external ([sm_external.md](sm_external.md)).

> **Post-C1.6 update (user-directed, 2026-04-22).** The two incumbent label-based members (`label_jaccard`, `label_jaro_winkler`) were removed from the roster. Rationale: on anonymised headers (K8) they collapse outright, and on non-anonymised headers they are redundant with `embedding_sbert` (which subsumes the column-name signal via SBERT encoding of the header). Rubric scores below are kept for historical continuity; the verdict column is updated to **Removed**. See §User Decision at the bottom of this file.

> **Second post-C1.6 update (user-directed, 2026-04-22).** The incumbent `instance_tfidf_cosine` member was also removed. Rationale: `embedding_sbert` encodes column names *and sample values* via SBERT; the cosine similarity over those embeddings subsumes TF-IDF-over-values as a value-distribution signal (dense semantic-rich representation vs. sparse lexical). Keeping both double-counts the instance-value axis. See §User Decision — 2026-04-22 (instance-tfidf removal) at the bottom of this file.

## Scoring matrix

| # | Candidate | Signal diversity | SOTA alignment | Integration cost | Determinism | Runtime fit | **Total** | Verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | `label_jaccard` (incumbent) | 2 | 1 | 3 | 3 | 3 | **12** | **Removed (2026-04-22, user-directed)** |
| 2 | `label_jaro_winkler` (incumbent) | 2 | 1 | 3 | 3 | 3 | **12** | **Removed (2026-04-22, user-directed)** |
| 3 | `instance_tfidf_cosine` (incumbent) | 3 | 2 | 3 | 3 | 3 | **14** | **Removed (2026-04-22, user-directed — subsumed by `embedding_sbert`)** |
| 4 | `duplicate_majority` (incumbent) | 3 | 2 | 3 | 3 | 3 | **14** | **Keep** |
| 5 | `llm_openai` (incumbent) | 3 | 2 | 3 | 2 | 2 | **12** | **Keep** |
| 6 | **Embedding SM (SBERT over column name + sample values, cosine)** | 3 | 2 | 3 | 3 | 3 | **14** | **Add** |
| 7 | Magneto (SLM-retrieval + LLM-rerank) | 2 | 3 | 1 | 2 | 1 | **9** | **Add (user-directed, 2026-04-20)** — implementation pending; tracked as C1.5 in [plan_committee_finalization.md](../../plans/plan_committee_finalization.md). Wire as `enabled_by_default: false`, opt-in via `--with-llm` alongside `llm_openai` to bound runtime/budget cost. |
| 8 | COMA 3.0 CE (via Valentine wrapper) | 1 | 1 | 1 | 3 | 2 | **8** | **Added** (user-directed override 2026-04-21, C1.6) — original C1.3 rationale below kept for the record; the Integration-cost score no longer applies because C1.6 uses ``ComaPy`` (pure Python; no JRE). See §User Decision — 2026-04-21 (COMA inclusion) at the bottom of this file. |
| 9 | Cupid (via Valentine) | 1 | 1 | 2 | 3 | 3 | **10** | Reject (tree-structural; our schemas are flat) |
| 10 | Similarity Flooding (via Valentine) | 2 | 1 | 2 | 3 | 3 | **11** | Defer (graph propagation has near-zero payoff on flat tabular; revisit when K3 nesting is re-enabled) |
| 11 | DistributionBased / EMD (via Valentine) | 1 | 1 | 2 | 3 | 2 | **9** | Reject (same signal class as TF-IDF cosine; EMD does not win on our fixture widths) |
| 12 | Unicorn (DeBERTa multi-task) | 2 | 2 | 1 | 2 | 1 | **8** | Reject (needs per-domain gold training pairs; we have no SM gold to train on) |
| 13 | SMAT (Siamese on GloVe/BERT) | 1 | 1 | 1 | 2 | 2 | **7** | Reject (superseded by Unicorn and by Magneto) |
| 14a | LLMatch | 2 | 2 | 1 | 2 | 1 | **8** | Reject — code IS public ([knowledge-fusion/LLMatch](https://github.com/knowledge-fusion/LLMatch); re-verified 2026-04-20), but MongoDB hard-dep + signal redundancy with `llm_openai` + Magneto drops it below the cutoff |
| 14b | SCHEMORA | 2 | 2 | 0 | 2 | 1 | **7** | Reject — paper cites a repo that 404s (re-verified 2026-04-20); defer until release is actually reachable |
| 14c | ConStruM | 1 | 2 | 0 | n/a | n/a | **< 10** | Reject — no code; design is an add-on layer over an upstream matcher, architecturally redundant with our committee's aggregation |
| 14d | SMoG | 1 | 2 | n/a | n/a | n/a | **n/a** | Reject — domain mismatch: requires a SPARQL / knowledge-graph endpoint; our sources are flat CSV/Parquet |

## Proposed roster (C1.3 — superseded by two post-C1.6 removals; see §User Decision blocks)

1. `label_jaccard` *(kept at C1.3 — removed 2026-04-22)*
2. `label_jaro_winkler` *(kept at C1.3 — removed 2026-04-22)*
3. `instance_tfidf_cosine` *(kept at C1.3 — removed 2026-04-22)*
4. `duplicate_majority` *(kept)*
5. **`embedding_sbert` *(new)*** — SentenceTransformer over column-name + sample values, cosine similarity. Signal class `embedding`. Fills the only axis that was completely missing and does so with a deterministic, CPU-tractable matcher whose weights are pinned.
6. `llm_openai` *(kept)*

C1.3 axis coverage: `label` (2), `instance` (1), `duplicate` (1), `embedding` (1, new), `llm` (1). Deterministic fallback (runs without LLM) = 5 of 6 members.

## Inclusion rationale (per new member)

### `embedding_sbert`

- **Why include:** This is the only gap-filler that scores 3 on signal diversity (new axis, no overlap) *and* is near-zero integration cost — `sentence-transformers` is already a core PyDI runtime dependency (see `pyproject.toml:32`). A deterministic wrapper with a pinned model (`all-MiniLM-L6-v2` or `paraphrase-MiniLM-L6-v2`, seed=42) is a thin adapter. This is the same retrieval backbone Magneto uses internally, so we pick up most of Magneto's retrieval-phase signal at ~1/10th the runtime cost.
- **Why in `usecases_synthetic/lib/`, not `PyDI/schemamatching/`:** follows the Ditto-matcher precedent in Phase A0 — synthetic-local for now, with a docstring noting the promotion path if the pattern generalises. Keeps this C1 change off PyDI core's public surface until it is battle-tested.

## Exclusion rationale (COMA specifically — user-requested evaluation)

COMA was explicitly flagged by the user as a candidate to evaluate. Here is the long-form argument for rejection so the decision is not relitigated.

**1. Architectural redundancy.** COMA's headline contribution is the *hybrid* design: run many individual matchers, aggregate their similarity cube, select survivors with a consensus rule. Our [sm_committee.yaml](../../usecases_synthetic/config/committees/sm_committee.yaml) does precisely this. Adding COMA as a single committee member means stacking one consensus layer on top of another, which is worse than either (a) adding COMA's underlying individual matchers directly or (b) leaving the committee's consensus as the single aggregation layer. Option (a) is dominated by our existing matchers (the `label_*` members already provide n-gram + Jaro-Winkler; the `instance_*` member provides value-set overlap; SBERT will provide the embedding signal COMA lacks).

**2. Individual-matcher overlap.** The COMA matchers that could theoretically add something new are the structural ones: `NamePath`, `Leaves`, `Parents`, `Children`, `Siblings`. These require a tree-shaped schema. PyDI sources are flat CSV / Parquet / XML-flattened-to-CSV. Knob K3 (`attribute_nesting`) at its hard level produces path-like names (`address.city`) but still flat tables — no cross-column hierarchy for a graph-propagation matcher to exploit. So the structural matchers would either return trivial results or need a custom adapter that re-synthesises a tree from the column-name syntax, which is new implementation work for unclear payoff.

**3. Integration cost.** The realistic integration path is via the Valentine Python package, which shells out to COMA 3.0 CE's Java CLI. That introduces a Java Runtime Environment as a hard dependency of `pydi-dev/` — currently all our matchers run in pure Python. The alternative — a pure-Python re-implementation of COMA's aggregator + key individual matchers — is multi-day work with no clear reliability win over the existing `sm_committee` wiring.

**4. Performance ceiling.** Per the [Valentine ICDE '21 study](../../literature-search-generation/valentine_schema_matching/paper.md#key-findings) and the Magneto VLDB '25 retrospective, COMA++ is clearly behind the 2023+ learned methods (Unicorn, Magneto) on the Valentine benchmarks. On the pre-LLM subset it is competitive with Similarity Flooding; neither is best-in-class today.

**5. Score summary:** signal diversity 1 (mostly redundant), SOTA alignment 1 (pre-2010 method, beaten by Magneto), integration cost 1 (JRE pull-in), determinism 3 (seeded, stable), runtime fit 2 (JVM startup + CLI roundtrip per match). Total 8 — below the 10 cutoff.

**6. Acknowledgement.** COMA is historically important, and its hybrid-aggregation idea is exactly the intellectual parent of our committee design. The right tribute to COMA in this project is that `sm_committee` already *implements* COMA's central principle at a different granularity. Including a wrapped COMA instance on top would be symbolic rather than useful.

## Exclusion rationale (other candidates, one paragraph each)

- **Cupid.** Tree-structural plus linguistic; our sources are flat. Same structural-matcher mismatch as COMA. Signal diversity 1, no gap to fill.
- **Similarity Flooding.** Graph-propagation matcher for schema graphs. Our schemas are flat tables. Payoff estimated at near zero without a true structural heterogeneity knob. **Defer** rather than reject — if a future nested-schema knob lands (K9 schema completeness for S2), SF becomes a natural candidate.
- **DistributionBased (EMD).** Same signal class as our `instance_tfidf_cosine` (value distributions). EMD vs. cosine is a comparator choice, not a new axis.
- **Unicorn.** Fine-tuned DeBERTa multi-task. Would need per-domain SM gold training pairs; we do not have SM gold separate from the known mapping (the mapping IS the gold, with only ~15–20 pairs per pair of sources — insufficient for fine-tuning). Ditto in EM has the same property but there we have thousands of EM gold pairs; SM is two-order-of-magnitudes smaller.
- **SMAT.** Superseded by Unicorn on shared benchmarks; no reason to prefer.
- **Magneto.** Originally deferred on cost grounds (retrieval-then-rerank runs an LLM call per candidate column, O(source_pairs × columns) vs. `llm_openai`'s O(source_pairs)). **Decision reversed 2026-04-20 at user request** — the SOTA signal is worth the cost given Scenario-2 targets. Implementation plan: wire as `enabled_by_default: false` with `signal_type: llm` so the `--with-llm` toggle enables it alongside `llm_openai`; the runtime-heavy variant sweeps run without it, the final validation runs include it. Tracked as C1.5 in [plan_committee_finalization.md](../../plans/plan_committee_finalization.md).
- **LLMatch.** *(Re-verified 2026-04-20: code IS public at [knowledge-fusion/LLMatch](https://github.com/knowledge-fusion/LLMatch) — earlier "no code" claim was wrong.)* Three-stage LLM framework (schema prep → table selection → rollup/drilldown column align). Rejection grounds on the current evidence: (a) hard MongoDB dependency for persistence and no PyPI package, so integration is clone-and-configure rather than an adapter import; (b) signal redundancy — it sits on the same `llm` axis as `llm_openai` and the incoming `magneto_slm_llm` (C1.5), so adding a third LLM-class member buys little diversity at triple the per-pair LLM cost. Rubric score 8/15 (signal diversity 2, SOTA 2, integration cost 1, determinism 2, runtime 1).
- **SCHEMORA.** Paper (arxiv 2507.14376) announces release at `github.com/ermangungor/schemora`, but as of 2026-04-20 the URL 404s and the author's GitHub has no public repos. Treat as effectively unavailable. If/when the repo appears, its stack (FAISS + BM25 + OpenAI embeddings) is still real integration work and the signal axis overlaps `llm_openai` + Magneto — so re-evaluation would likely land below the cutoff anyway.
- **ConStruM.** arxiv 2601.20482 (Jan 2026); no code repo found (`github.com/ConStrum` is unrelated Minecraft tooling; first-author's project page does not list it). Even granting code, ConStruM is explicitly designed as an **add-on that augments an upstream matcher's LLM prompt** — structurally the same role our committee's aggregation layer plays. Redundant with the committee itself.
- **SMoG.** arxiv 2511.20285 (Nov 2025). Uses **iterative 1-hop SPARQL exploration over knowledge graphs**. PyDI's sources are flat CSV/Parquet — there is no SPARQL endpoint to query. Domain mismatch, not a code-availability issue.

## Required-axes update

The C1.3 roster change expanded `required_axes.signal_type` from `[label, instance, duplicate]` to `[label, instance, duplicate, embedding]`. Post-C1.6 label removal (2026-04-22) drops `label`. The second post-C1.6 removal (2026-04-22, `instance_tfidf_cosine`) drops `instance`. Final enforced set is `[duplicate, embedding]`. `llm` remains optional per the `--with-llm` flag; `hybrid` will be enforced once C1.6 (`coma_hybrid`) lands.

## Deliverable

Final roster after all user-directed revisions: 5 declared members (`duplicate_majority`, `embedding_sbert`, `llm_openai`, `magneto_slm_llm` opt-in, `coma_hybrid` enabled by default once C1.6 lands). Exclusion rationales committed (COMA long-form in C1.3, overridden in C1.6; plus one paragraph each for Cupid / SF / EMD / Unicorn / SMAT / the no-code 2024–2025 preprints). Magneto: deferred in C1.3, added in C1.5. Label members: kept in C1.3, removed 2026-04-22 (C1.7). `instance_tfidf_cosine`: kept in C1.3, removed 2026-04-22 (C1.8) after `embedding_sbert` (C1.4) absorbed the value-distribution signal.

## User Decision — 2026-04-22: remove `label_jaccard` and `label_jaro_winkler`

**Decision.** Drop both label-based members from `sm_committee.yaml`. Drop `label` from `required_axes.signal_type`.

**Rationale.**

1. **K8 (anonymised headers) collapses the label signal outright.** Our scenario explicitly includes a K8 level that renames columns to opaque tokens. Both Jaccard and Jaro–Winkler on renamed headers produce near-zero scores across the board and add no discriminative signal — worse, they drag down aggregate confidence when the committee averages.
2. **On non-anonymised headers, `embedding_sbert` subsumes the label signal.** SBERT over the column name already captures lexical similarity (same-token overlaps project near each other) *plus* the semantic generalisation that pure string similarity misses (e.g. `ticker` ↔ `symbol`). Keeping Jaccard/Jaro–Winkler alongside is double-counting a weaker projection of the same signal.
3. **The committee-diversity argument is weak once `embedding` + `instance` + `duplicate` are all present.** Signal-diversity 2 (the C1.3 score) assumed the label axis was otherwise uncovered; after C1.4 added `embedding_sbert`, that assumption no longer holds. The label members' effective diversity contribution is closer to 0–1.
4. **Impact on K8 robustness test.** The `test_at_least_one_label_and_one_instance` sanity check is replaced by `test_at_least_one_instance_and_one_embedding` — both are non-header-dependent anchors, which is the property the original test was approximating.

**Implementation.** [sm_committee.yaml](../../usecases_synthetic/config/committees/sm_committee.yaml) members list trimmed from 7 to 5 (+2 still gated: Magneto opt-in, COMA pending C1.6). `required_axes.signal_type` reduced to `[instance, duplicate, embedding]`. [test_committee_configs.py](../../usecases_synthetic/tests/test_committee_configs.py) test renamed and updated. 21/21 committee-config tests pass (1 pre-existing EM skip unrelated).

## User Decision — 2026-04-22: remove `instance_tfidf_cosine`

**Decision.** Drop the `instance_tfidf_cosine` member from `sm_committee.yaml`. Drop `instance` from `required_axes.signal_type`.

**Rationale.**

1. **`embedding_sbert` subsumes the value-distribution signal.** SBERT encodes column name *and* sample values (`max_sample_size: 20` per the YAML) into a single dense vector. Cosine similarity over those embeddings captures the same "which column has values that look similar?" signal TF-IDF cosine was providing — but with dense semantic representation instead of sparse lexical. Anywhere TF-IDF finds a match via surface-token overlap, SBERT finds the same match plus semantic variants (e.g. `"United States"` ↔ `"USA"`, which TF-IDF cosine misses).
2. **Scoring-matrix score 14 is not preservative when the axis is double-counted.** The 2026-04-22 re-evaluation treats `signal_type.instance` as an axis already filled by `embedding_sbert`'s value-encoding half. Under that framing, `instance_tfidf_cosine`'s signal-diversity score drops from 3 to 1 — recomputed total would be ~11, still above the cutoff but no longer dominant, and the committee benefits more from one strong embedding member than from two value-distribution members that largely agree.
3. **Deterministic fallback preserved.** After removal, the deterministic members (runs without `--with-llm`) are `duplicate_majority` + `embedding_sbert`. Both are seeded, CPU-tractable, reproducible. The `--with-llm=false` path still has ≥2 anchors.
4. **Impact on the K8 test.** `test_at_least_one_instance_and_one_embedding` is renamed to `test_at_least_one_duplicate_and_one_embedding`. Duplicate-based matching is header-independent (runs on known correspondences between value sets), so K8 anonymised-header robustness is still anchored.

**Implementation.** [sm_committee.yaml](../../usecases_synthetic/config/committees/sm_committee.yaml) members list trimmed from 5 to 4 wired (duplicate + embedding + llm + magneto-opt-in; COMA still pending C1.6). `required_axes.signal_type` reduced from `[instance, duplicate, embedding]` to `[duplicate, embedding]`. [test_committee_configs.py](../../usecases_synthetic/tests/test_committee_configs.py) test renamed. `PyDI.schemamatching.instance_based.InstanceBasedSchemaMatcher` kept in place — roster decision, not a deprecation. 21/21 committee-config tests pass (1 pre-existing EM skip unrelated).

## User Decision — 2026-04-21: add `coma_hybrid` (C1.6)

**Decision.** Add COMA 3.0 CE as the 4th enabled-by-default SM committee member, wired via Valentine's pure-Python `ComaPy` implementation. Override of the C1.3 reject (score 8/15, below cutoff).

**Rationale.** The user judged that historical importance + hybrid-aggregation reference signal justify inclusion on grounds outside the rubric.

**Deviation from the plan as written.** [plan_committee_finalization.md §C1.6](../../plans/plan_committee_finalization.md#c1--schema-matching-committee) prescribes the Java CLI path (`valentine.algorithms.Coma`) with a JRE dependency. Valentine deprecated the Java path in favour of `ComaPy` (a pure-Python reimplementation of the same algorithm; JRE-free; will be renamed to `Coma` in valentine v1.0.0). C1.6 uses `ComaPy`:

1. **No JRE dependency** — the biggest integration-cost line item in the C1.3 rejection is eliminated. `pydi-dev/` remains an all-Python environment.
2. **No subprocess / JVM warm-start overhead** — the plan's "keep Valentine process alive across SM calls" budget optimisation is unnecessary; `ComaPy` invocations are in-process and cheap (<1s each at companies-small scale).
3. **Same algorithm** — `ComaPy` is a faithful port of the `COMA_OPT` / `COMA_OPT_INST` strategies, so the committee slot's *semantic* role (hybrid label+instance+structural ensemble) is preserved.
4. **Upstream recommendation** — Valentine's own `Coma` class now emits a `DeprecationWarning` pointing at `ComaPy`.

**Rubric re-score under the ComaPy path.**

| Axis | C1.3 score | ComaPy re-score | Note |
|---|---|---|---|
| Signal diversity | 1 | 2 | Hybrid-ensemble is a distinct axis the committee does not otherwise fill. |
| SOTA alignment | 1 | 1 | Unchanged — still pre-LLM era. |
| Integration cost | 1 | 3 | Drop-in: `pip install valentine` + adapter. No JRE, no subprocess, no caching layer. |
| Determinism | 3 | 3 | Unchanged. |
| Runtime fit | 2 | 3 | CPU-tractable (<1s per source-target on companies-small). |
| **Total** | **8** | **12** | Above the 10-cutoff under the ComaPy re-score. |

Inclusion would therefore be rubric-justified even without the user override; the override short-circuits re-relitigating.

**Preserved C1.3 critique (architectural redundancy).** The original argument — that COMA stacks a consensus layer on top of our committee's consensus — remains valid on paper. In practice, COMA's aggregator operates on *different* sub-signals (label tokens, n-grams, value distributions, structural paths), so the committee sees it as one heterogeneous signal rather than a re-aggregation of its own members. The scoring-matrix re-evaluation treats the two as sufficiently decorrelated at the committee level.

**Implementation.**

1. **Adapter.** [usecases_synthetic/lib/coma_sm_matcher.py](../../usecases_synthetic/lib/coma_sm_matcher.py) — `ComaSchemaMatcher(BaseSchemaMatcher)` wraps `valentine.algorithms.ComaPy` via `valentine_match`, projects on `get_schema_columns` (strips PyDI id columns), returns the standard `(source_dataset, source_column, target_dataset, target_column, score, notes)` mapping shape.
2. **Extras.** New `coma` group in [pyproject.toml](../../pyproject.toml) pulling `valentine>=0.2.0,<1.0.0` only. Separate from `magneto` so a default install does not drag in the LLM-path deps (litellm, fuzzywuzzy, json-repair). `valentine` is also in `magneto` — both extras install the same package with no version conflict.
3. **YAML.** Added `coma_hybrid` to [sm_committee.yaml](../../usecases_synthetic/config/committees/sm_committee.yaml) with `signal_type: hybrid`, `enabled_by_default: true`, `match_kwargs.threshold: 0.3`. `required_axes.signal_type` extended to `[duplicate, embedding, hybrid]`.
4. **Tests.** New [`usecases_synthetic/tests/test_coma_sm_matcher.py`](../../usecases_synthetic/tests/test_coma_sm_matcher.py) — 18 functional tests covering shape + typing, determinism (repeat calls + fresh instances), scoring sanity (identical column names above 0.4 floor), NaN tolerance (source + target), edge cases (empty source / empty target / both empty / zero-row target / PyDI id column stripped), API contract (preprocess kwarg accepted + ignored, notes column tagged). [`test_committee_configs.py`](../../usecases_synthetic/tests/test_committee_configs.py) gained `test_has_hybrid_member`.
5. **Smoke test.** Ran the adapter against the companies-small forbes source (2000 rows) × the companies target schema (10 columns). With a zero-row target (matches the SM committee runner's default): 3 correct matches at precision=1.00 recall=0.50 F1=0.67 in 0.03s. With the target populated by 50 sample values drawn through the gold mapping: 5 correct matches at precision=1.00 recall=0.83 F1=0.91 in 0.53s — only `Company→name` (semantic-only) is missed. Runtime budget well within per-variant limits.

**Test output.** `pytest usecases_synthetic/tests/test_committee_configs.py usecases_synthetic/tests/test_coma_sm_matcher.py -v` → 40/40 pass (1 pre-existing EM PLM skip unrelated).

**Required-axes update.** `signal_type` enforced set becomes `[duplicate, embedding, hybrid]`. `llm` remains opt-in via the `--with-llm` flag. Deterministic fallback = `duplicate_majority` + `embedding_sbert` + `coma_hybrid` (three anchors now, up from two).
