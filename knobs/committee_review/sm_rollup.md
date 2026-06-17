# SM Committee — Rollup (C1.1–C1.6 complete; two 2026-04-22 removals)

Finalised roster for the schema-matching committee. Supersedes the roster frozen at M5.

## Final roster (4 wired + 1 opt-in = 5 declared)

| # | Name | Signal | Module | Status |
|---|---|---|---|---|
| ~~1~~ | ~~`label_jaccard`~~ | ~~label~~ | ~~`PyDI.schemamatching.label_based.LabelBasedSchemaMatcher`~~ | **Removed (C1.7, 2026-04-22, user-directed)** |
| ~~2~~ | ~~`label_jaro_winkler`~~ | ~~label~~ | ~~same~~ | **Removed (C1.7, 2026-04-22, user-directed)** |
| ~~3~~ | ~~`instance_tfidf_cosine`~~ | ~~instance~~ | ~~`PyDI.schemamatching.instance_based.InstanceBasedSchemaMatcher`~~ | **Removed (C1.8, 2026-04-22, user-directed — subsumed by `embedding_sbert`)** |
| 4 | `duplicate_majority` | duplicate | `PyDI.schemamatching.duplicate_based.DuplicateBasedSchemaMatcher` | Kept |
| 5 | **`embedding_sbert`** | embedding | `usecases_synthetic.lib.embedding_sm_matcher.EmbeddingBasedSchemaMatcher` | **Added (C1.4)** |
| 6 | `llm_openai` | llm | `PyDI.schemamatching.llm_based.LLMBasedSchemaMatcher` | Kept |
| 7 | **`magneto_slm_llm`** | llm (opt-in, `enabled_by_default: false`) | `usecases_synthetic.lib.magneto_sm_matcher.MagnetoSchemaMatcher` | **Added (C1.5, 2026-04-20)** |
| 8 | **`coma_hybrid`** | hybrid (`enabled_by_default: true`) | `usecases_synthetic.lib.coma_sm_matcher.ComaSchemaMatcher` | **Added (C1.6, 2026-04-22)** |

`required_axes.signal_type` is `[duplicate, embedding, hybrid]` after C1.6. The LLM axis remains opt-in via the `--with-llm` flag.

## How I got here

1. **[sm_portfolio.md](sm_portfolio.md) — C1.1.** Inventoried the four schema-matching anchors in `literature-search-generation/` (Magneto, Valentine, XBenchMatch, Jellyfish). Identified the missing axes: `embedding`, `learned-SM`, `hybrid-ensemble`, `structural`.
2. **[sm_external.md](sm_external.md) — C1.2.** Ran the six external-search queries from [plan_committee_finalization.md §C1](../../plans/plan_committee_finalization.md#c1--schema-matching-committee). Added a COMA paper card at [literature-search-generation/coma_schema_matching/paper.md](../../literature-search-generation/coma_schema_matching/paper.md). Ranked 12 candidates by fit-to-committee.
3. **[sm_shortlist.md](sm_shortlist.md) — C1.3.** Scored the candidate pool against the 5-axis rubric. Only one candidate (`embedding_sbert`) cleared the cutoff + added distinct signal. Magneto deferred on runtime grounds; COMA rejected on architectural-redundancy and JRE-dependency grounds.
4. **C1.4 — implementation.**
   - New adapter: [usecases_synthetic/lib/embedding_sm_matcher.py](../../usecases_synthetic/lib/embedding_sm_matcher.py) (`EmbeddingBasedSchemaMatcher`). Pinned model `sentence-transformers/all-MiniLM-L6-v2`, deterministic sampling (seed=42), mypy-strict clean.
   - YAML: [sm_committee.yaml](../../usecases_synthetic/config/committees/sm_committee.yaml) now lists the 6 members, updated `required_axes`, and a header comment pointing at this rollup.
   - Tests: [`usecases_synthetic/tests/test_committee_configs.py::TestSMCommitteeConfig`](../../usecases_synthetic/tests/test_committee_configs.py) — 5/5 pass (loads, required fields, classes importable, axis coverage, label+instance sanity).
   - Smoke test: instantiated every non-LLM member and ran each against a 20-row `companies-small/forbes` fixture with an empty companies target. All 5 members returned a valid mapping frame without exception; `embedding_sbert` produced 1 match above the 0.55 threshold.

5. **Label-based removal** (C1.7, 2026-04-22, user-directed). Dropped `label_jaccard` and `label_jaro_winkler`. Rationale: they collapse on K8 anonymised headers, and on non-anonymised headers they are redundant with `embedding_sbert` (which captures header-string similarity *plus* semantic generalisation). Kept the adapter classes in `PyDI/schemamatching/label_based.py` — this is a roster decision, not a deprecation. YAML `required_axes.signal_type` reduced from `[label, instance, duplicate, embedding]` to `[instance, duplicate, embedding]`. The `test_at_least_one_label_and_one_instance` sanity check in [test_committee_configs.py](../../usecases_synthetic/tests/test_committee_configs.py) renamed to `test_at_least_one_instance_and_one_embedding` — same intent (K8-robust anchor present), updated for the new roster. 21/21 committee-config tests pass. Decision block in [sm_shortlist.md §User Decision — 2026-04-22 (label removal)](sm_shortlist.md#user-decision--2026-04-22-remove-label_jaccard-and-label_jaro_winkler).

6. **Instance-tfidf removal** (C1.8, 2026-04-22, user-directed). Dropped `instance_tfidf_cosine`. Rationale: `embedding_sbert`'s SBERT encoding of column name + up-to-20 sample values subsumes TF-IDF-over-values as a value-distribution signal (dense semantic vs. sparse lexical) — anywhere TF-IDF found a match via surface-token overlap, SBERT finds it plus semantic variants TF-IDF would miss. Keeping both double-counts the instance-value axis. Kept `PyDI.schemamatching.instance_based.InstanceBasedSchemaMatcher` in place — roster decision, not a deprecation. YAML `required_axes.signal_type` reduced from `[instance, duplicate, embedding]` to `[duplicate, embedding]`. The K8 sanity test renamed again: `test_at_least_one_instance_and_one_embedding` → `test_at_least_one_duplicate_and_one_embedding`. 21/21 committee-config tests pass. Decision block in [sm_shortlist.md §User Decision — 2026-04-22 (instance-tfidf removal)](sm_shortlist.md#user-decision--2026-04-22-remove-instance_tfidf_cosine).

7. **C1.5 — Magneto integration** (2026-04-20, user-directed override of the C1.3 defer).
   - **Vendoring.** Cloned [VIDA-NYU/magneto-matcher](https://github.com/VIDA-NYU/magneto-matcher) (commit `6620623`, main, Apache-2.0). Copied `algorithms/magneto/magneto/` → [usecases_synthetic/third_party/magneto_matcher/magneto/](../../usecases_synthetic/third_party/magneto_matcher/) (17 `.py` files + `LICENSE` + `ORIGIN.md` + `README.md`). Sole edits: absolute `from magneto.xxx` imports rewritten to relative `from .xxx` (12 files touched — see [README.md](../../usecases_synthetic/third_party/magneto_matcher/README.md) for the list).
   - **Extras.** Added a `magneto` optional-dependency group to [pyproject.toml](../../pyproject.toml) covering `fuzzywuzzy`, `mmh3`, `valentine`, `litellm`, `json-repair` — the five upstream deps that aren't already in PyDI core. Installed into `pydi-dev/` with `uv pip install -e ".[magneto]"`.
   - **Adapter.** New [usecases_synthetic/lib/magneto_sm_matcher.py](../../usecases_synthetic/lib/magneto_sm_matcher.py) — `MagnetoSchemaMatcher(BaseSchemaMatcher)`. Two-phase pipeline (SLM retrieval + optional LLM rerank). `temperature=0`. File-backed prompt cache under `usecases_synthetic/cache/magneto_prompts/` keyed by `sha256(prompt_version | model_id | prompt_text)`. Accepts the committee runner's injected `chat_model` kwarg and derives its litellm model id from it so the YAML uses the same `model_name` contract as `llm_openai`.
   - **YAML.** Added `magneto_slm_llm` member to [sm_committee.yaml](../../usecases_synthetic/config/committees/sm_committee.yaml) with `signal_type: llm`, `enabled_by_default: false`. Opt-in via `--with-llm` — default runs still skip it. LLM-cost comment in the YAML documents the ~11× inflation vs. `llm_openai` (O(columns) vs O(source_pairs)).
   - **Tests.** `test_committee_configs.py` — 22/22 (21 pass, 1 pre-existing EM PLM skip unrelated to C1.5).
   - **Smoke tests.**
     - *SLM-only path.* Instantiated `MagnetoSchemaMatcher(use_llm_rerank=False)` on 20 `companies-small/forbes` rows × 6 canonical target columns. Returned 54 mapping rows above the 0.3 threshold; top-3 matches `Country→country`, `Sector→sector`, `Industry→industry` at score 1.0 (bp_reranker assignment pins). No exceptions.
     - *LLM-rerank path.* Patched `litellm.completion` with a stub response. Confirmed 11 cache files created (one per source column), payload schema `{model_id, prompt, prompt_version, response}`. No live API call required for the wiring check.

## COMA-specific decision summary

The user explicitly asked for a COMA evaluation. Outcome: **reject**, score 8/15 (below 10-cutoff).

The full rationale is in [sm_shortlist.md §Exclusion rationale (COMA specifically)](sm_shortlist.md#exclusion-rationale-coma-specifically--user-requested-evaluation) — five grounds, summarised:

1. **Architectural redundancy:** COMA's hybrid-aggregator design is what `sm_committee` already is at a different granularity — adding COMA means stacking committees.
2. **Individual-matcher overlap:** COMA's structural matchers (`NamePath`, `Leaves`, `Parents/Children/Siblings`) assume tree schemas; our sources are flat. Its label-based matchers are dominated by our existing Jaccard + Jaro-Winkler + SBERT combination.
3. **Integration cost:** Valentine wraps COMA but shells out to a Java CLI — introduces a JRE dependency for `pydi-dev/`.
4. **Performance ceiling:** COMA++ is pre-LLM era; clearly beaten by Magneto (VLDB '25) and Unicorn (SIGMOD '23) on Valentine benchmarks.
5. **Aggregation-weight learning needs gold:** the weighted-aggregator variant needs training pairs per domain; our SM gold has ~20 pairs per source pair.

A paper card documenting the method and this evaluation is committed at [literature-search-generation/coma_schema_matching/](../../literature-search-generation/coma_schema_matching/) so the decision is not relitigated.

## Downstream impact on [plan_s1_scale.md](../../plans/plan_s1_scale.md)

- **S5 re-run checklist:** the SM roster change invalidates any SM baseline numbers previously recorded against the 5-member roster. Document the roster bump in the S5 re-run log; expect the new `embedding_sbert` slot to shift precision/recall per SM stage — this is expected and should be reported.
- **C4 cross-committee consistency review:** the column-name assumptions inside `embedding_sbert` do *not* couple to the EM committee's `column_mapping`, so no coupling change is needed. Trust-score semantics unchanged.
- **C1 unblocks:** C2 / C3 committee finalisation. C1's contribution is complete; S4b (companies-small sanity check) can proceed once C2 and C3 are also frozen.

## C1.6 — COMA integration (2026-04-22, user-directed override of the C1.3 reject)

**Context.** The C1.3 scoring had COMA at 8/15, below the 10-cutoff, on three main grounds: architectural redundancy with the committee's own aggregator, JRE dependency, and pre-LLM performance ceiling. The user explicitly re-requested inclusion on historical-importance + hybrid-ensemble-reference grounds. See [sm_shortlist.md §User Decision — 2026-04-21 (COMA inclusion)](sm_shortlist.md#user-decision--2026-04-21-add-coma_hybrid-c16).

**Deviation from the plan.** `plan_committee_finalization.md` C1.6 prescribes Valentine's Java CLI (`valentine.algorithms.Coma`). Valentine deprecated that path in favour of `ComaPy` — a pure-Python reimplementation of the same `COMA_OPT` / `COMA_OPT_INST` strategy. C1.6 uses `ComaPy`:

- No JRE dependency in `pydi-dev/` (the biggest integration-cost line item from the C1.3 rejection disappears).
- No JVM warm-start or subprocess caching needed — in-process Python, <1s per source-target invocation.
- Same algorithm, so the committee slot's semantic role (hybrid label+instance+structural ensemble) is preserved.

Under the ComaPy re-score the rubric total rises from 8 → 12 (signal-diversity 1→2, integration-cost 1→3, runtime-fit 2→3), so inclusion is also rubric-justified; the user override short-circuits re-relitigating.

**Implementation.**

- **Adapter.** [usecases_synthetic/lib/coma_sm_matcher.py](../../usecases_synthetic/lib/coma_sm_matcher.py) — `ComaSchemaMatcher(BaseSchemaMatcher)` wraps `valentine.algorithms.ComaPy` via `valentine_match`, projects on `get_schema_columns` (strips PyDI id columns), returns the standard `(source_dataset, source_column, target_dataset, target_column, score, notes)` mapping frame. Lazy-imports `valentine` so PyDI core stays free of the dependency when `coma_hybrid` is not enabled.
- **Extras.** New `coma` group in [pyproject.toml](../../pyproject.toml) pulling only `valentine>=0.2.0,<1.0.0` — kept separate from the `magneto` extras so a default install does not drag in the LLM deps (`litellm`, `fuzzywuzzy`, `json-repair`).
- **YAML.** Added `coma_hybrid` member to [sm_committee.yaml](../../usecases_synthetic/config/committees/sm_committee.yaml) with `signal_type: hybrid`, `enabled_by_default: true`, `match_kwargs.threshold: 0.3`, params `max_n=1`, `use_instances=true`, `use_schema=true`, `delta=0.15`. `required_axes.signal_type` extended to `[duplicate, embedding, hybrid]`.
- **Tests.** New [`usecases_synthetic/tests/test_coma_sm_matcher.py`](../../usecases_synthetic/tests/test_coma_sm_matcher.py) — 18 functional tests covering shape + typing, determinism (repeat calls + fresh instances), scoring sanity (identical column names above a 0.4 floor), NaN tolerance (source + target), edge cases (empty / zero-row / PyDI id column stripped), API contract (`preprocess` kwarg accepted + ignored). [`test_committee_configs.py`](../../usecases_synthetic/tests/test_committee_configs.py) gained `test_has_hybrid_member`. 40/40 pass (`pytest usecases_synthetic/tests/test_committee_configs.py usecases_synthetic/tests/test_coma_sm_matcher.py -v`).
- **Smoke test on companies-small forbes.** With a zero-row target: precision 1.00 / recall 0.50 / F1 0.67, 3 correct matches in 0.03s (misses `Company→name`, `Identifier→id`, `Sales→revenue` — all semantic-only pairs without instance signal). With 50 sample values injected into the target: precision 1.00 / recall 0.83 / F1 0.91 in 0.53s — only `Company→name` (SBERT-territory semantic pair) stays missed. Budget well within per-variant limits.

## Exit criteria checklist

- [x] [sm_committee.yaml](../../usecases_synthetic/config/committees/sm_committee.yaml) updated — 4 wired members after C1.6 (duplicate + embedding + llm + coma-hybrid) + Magneto opt-in; axis coverage is `[duplicate, embedding, hybrid]` (+ opt-in `llm`).
- [x] `test_committee_configs.py` — 6/6 SM tests pass, 22/23 suite passes (+1 unrelated EM PLM skip). Plus 18/18 new ComaSchemaMatcher tests.
- [x] Rollup document citing portfolio + external sources for every member (inclusion) and every notable exclusion (COMA long-form + six one-paragraph rejections in the shortlist; plus the 2026-04-22 label/instance-tfidf removal decision blocks).
- [x] Smoke test — all non-LLM members instantiate and return a non-empty mapping frame on the companies-small slice; COMA ends at F1=0.91 with values / F1=0.67 without; Magneto SLM-only returns 54 rows.
- [x] Adapter written: [embedding_sm_matcher.py](../../usecases_synthetic/lib/embedding_sm_matcher.py) (C1.4) + [magneto_sm_matcher.py](../../usecases_synthetic/lib/magneto_sm_matcher.py) (C1.5) + [coma_sm_matcher.py](../../usecases_synthetic/lib/coma_sm_matcher.py) (C1.6).

C1.1–C1.6 complete. Post-C1.5 label removal applied 2026-04-22. C2 and C3 remain the outstanding gates for S4a (plan_s1_scale.md); they can run in parallel with a now-closed C1.
