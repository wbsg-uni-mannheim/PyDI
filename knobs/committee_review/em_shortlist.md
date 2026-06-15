# EM Committee — Scored Shortlist (C2.3)

> **USER DECISION (2026-04-21, updated 2026-04-22) — supersedes the proposed roster below.**
>
> The EM committee is being **split into two separate committees**: a **blocking committee** and a **matching committee**. This is an architectural change, not just a roster pick.
>
> **Matching committee — fixed to 4 members (user-directed, 2026-04-22):**
> 1. `ditto_plm` *(incumbent)*
> 2. `llm_matcher` *(incumbent; prompts upgraded to MatchGPT-style templates — ports the MatchGPT prompt design into our own adapter, sidestepping the no-LICENSE risk)*
> 3. **Magellan** *(new — classical ML on comparator vectors; BSD-licensed py_entitymatching)*
> 4. **ComEM** *(new — LLM prompt-strategy: match/compare/select ensemble over llm_matcher's base)*
>
> **Reversal note (2026-04-22):** Unicorn was in the 2026-04-21 freeze as a 5th member but was **dropped on 2026-04-22**. Rationale: per-domain DeBERTa fine-tune + GPU requirement is not worth a single matching slot that overlaps Ditto's supervised-PLM learned axis (signal diversity 2, not 3; rubric total 8, already below the 10-cutoff — see row 11 in the scoring matrix below). With Unicorn out, the learned axis is anchored by Ditto alone, the LLM axis by `llm_matcher` (MatchGPT-style prompts) + ComEM, and the classical axis by Magellan.
>
> **Blocking committee — roster TBD.** The three rule members currently in the combined committee (`token_rule`, `embedding_rule`, `standard_rule`) move to the blocking committee. Additional blocking members (SC-Block, Sudowoodo's FAISS encoder, hybrid blockers) are open for user direction in a follow-up decision.
>
> **Implications for C2.4 implementation:**
> - Split [em_committee.yaml](../../usecases_synthetic/config/committees/em_committee.yaml) into `em_blocking_committee.yaml` + `em_matching_committee.yaml` (or `blocking_committee.yaml` + add a new top-level matching-only section); the committee-loader + `test_committee_configs.py` schema will need to learn the two-committee shape.
> - New adapters required under `usecases_synthetic/lib/`: `magellan_em_matcher.py`, `comem_em_matcher.py`. Upgrade `llm_matcher`'s prompt templates to MatchGPT-style (no new adapter, prompt file update). (`unicorn_em_matcher.py` is **not** needed — Unicorn was dropped 2026-04-22.)
> - Sudowoodo, Splink, and Unicorn from the scored shortlist below are **not** included in this freeze.
> - HierGAT and MatchGPT (standalone) are **not** included (no-LICENSE risk; MatchGPT prompt design is absorbed into `llm_matcher`).
> - Rule members remain deployed but under the blocking committee's configuration surface.
>
> Proceed to C2.4 with this fixed roster; the scored shortlist below is preserved as the evidence trail that informed the decision.

---

Scoring follows the rubric in [plan_committee_finalization.md §Step (iii)](../../plans/plan_committee_finalization.md#step-iii--candidate-shortlist--scoring-rubric). Five axes, 0–3 each; committee-slot cutoff is total ≥ 10 with no axis scoring 0. The EM target per plan §C2 is a **5–7 member committee with diverse blocking + matching signals** and explicit expected addition of *"one contrastive/self-supervised learned matcher (Sudowoodo or similar) or a retrieval-augmented LLM matcher (LEMONADE-style) to complement Ditto's supervised PLM path."*

Candidate pool = portfolio ([em_portfolio.md](em_portfolio.md)) ∪ external ([em_external.md](em_external.md)). **C2.4 is user-directed** — this document is advisory. The rubric is a heuristic; the user may override cutoffs as they did in C1.5 when they added Magneto despite the C1.3 defer. **Proposed roster at a glance (7 members):** keep the 3 rule/blocker incumbents + `ditto_plm` + `llm_matcher` (promote to on-by-default), add **Sudowoodo** (contrastive learned matcher + hybrid blocker) and **Splink** (CPU + NaN-tolerant probabilistic matcher). This fills Gaps 2, 3, 5, 6 with public code; Gap 1 (HierGAT / graph-attention) and Gap 4 (MatchGPT) are flagged for user decision due to no-LICENSE-file risk.

## Scoring matrix

| # | Candidate | Method class | Blocking type | Matching type | Signal diversity | SOTA alignment | Integration cost | Determinism | Runtime fit | **Total** | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `token_rule` *(incumbent)* | rule, lexical | lexical | rule | 2 | 1 | 3 | 3 | 3 | **12** | **Incumbent (keep)** |
| 2 | `embedding_rule` *(incumbent)* | rule over embedding blocker | embedding | rule | 2 | 1 | 3 | 3 | 3 | **12** | **Incumbent (keep)** |
| 3 | `standard_rule` *(incumbent)* | rule, lexical | lexical | rule | 2 | 1 | 3 | 3 | 3 | **12** | **Incumbent (keep)** |
| 4 | `ditto_plm` *(incumbent)* | fine-tuned PLM | embedding | learned | 3 | 3 | 3 | 2 | 2 | **13** | **Incumbent (keep)** |
| 5 | `llm_matcher` *(incumbent, opt-in)* | zero-shot LLM prompt | embedding | llm | 3 | 2 | 3 | 2 | 2 | **12** | **Incumbent (promote to on-by-default)** |
| 6 | [Sudowoodo](../../literature-search-generation/sudowoodo_contrastive_em/) | contrastive self-supervised PLM | embedding (hybrid-capable via FAISS) | learned | 3 | 3 | 2 | 2 | 2 | **12** | **Shortlist (add)** |
| 7 | [Splink](../../literature-search-generation/splink_probabilistic_linkage/) | Fellegi-Sunter probabilistic, EM-learned m/u | lexical (rule-driven) | learned (probabilistic) | 3 | 2 | 2 | 3 | 3 | **13** | **Shortlist (add)** |
| 8 | [SC-Block](../../literature-search-generation/scblock_supervised_contrastive_blocking/) | supervised contrastive blocker only | embedding (ANN) | n/a — blocker only | 2 | 3 | 2 | 2 | 2 | **11** | **Defer** (signal subsumed by Sudowoodo which also blocks; revisit if blocker-specific eval is needed) |
| 9 | [HierGAT](../../literature-search-generation/hiergat_graph_attention_em/) | hierarchical graph-attention matcher | n/a (requires upstream blocker) | learned | 3 | 2 | 1 | 2 | 1 | **9** | **Defer** (no-LICENSE risk + GPU-required + Integration=1; flag for user) |
| 10 | [MatchGPT](../../literature-search-generation/matchgpt_llm_em/) | retrieval-augmented LLM prompts | embedding | llm | 2 | 3 | 2 | 2 | 2 | **11** | **Defer** (no-LICENSE risk + signal overlap with `llm_matcher`; flag for user) |
| 11 | [Unicorn](../../literature-search-generation/unicorn_unified_matching/) | DeBERTa MoE multi-task matcher | n/a | learned | 2 | 2 | 1 | 2 | 1 | **8** | **Defer** (same reasoning as C1.3 Unicorn exclusion: needs per-domain labeled fine-tune; GPU-required) |
| 12 | [AnyMatch](../../literature-search-generation/anymatch_zero_shot_em/) | instruction-tuned GPT-2 SLM | n/a | learned | 2 | 2 | 2 | 2 | 3 | **11** | **Defer** (signal class overlaps Sudowoodo's label-light learned slot; one suffices) |
| 13 | [TailorMatch](../../literature-search-generation/finetuning_llm_em/) | fine-tuned generative LLM (Llama 8B/70B, GPT-4o) | n/a | llm (fine-tuned) | 2 | 3 | 1 | 2 | 1 | **9** | **Defer** (needs a one-off fine-tune on companies-small gold + GPU; Mannheim-preferred but high integration cost) |
| 14 | [Jellyfish](../../literature-search-generation/jellyfish_llm_data_preprocessing/) | instruction-tuned Llama 2/3 7B-13B | n/a | llm (fine-tuned) | 1 | 2 | 1 | 2 | 1 | **7** | **Drop** (overlaps TailorMatch; 7B/13B weights + GPU) |
| 15 | [ComEM](../../literature-search-generation/comem_match_compare_select/) | LLM prompt-strategy (match/compare/select) | n/a | llm | 1 | 2 | 2 | 2 | 2 | **9** | **Drop** (pure prompt variant of `llm_matcher`; same axis, no diversity win) |
| 16 | [Magellan / py_entitymatching](../../literature-search-generation/magellan_em_system/) | classical ML on comparator vectors | lexical | learned (classical) | 2 | 1 | 2 | 3 | 3 | **11** | **Defer** (solid CPU fallback, but Splink dominates on NaN tolerance at similar integration cost) |

## Per-candidate rationale

### Included — Shortlist (added) and Incumbent (kept)

- **Incumbents 1–3 (`token_rule` / `embedding_rule` / `standard_rule`).** Baseline rule members covering the two `blocking_type` axes (`lexical`, `embedding`). Determinism 3, runtime 3, integration drop-in. Not SOTA (axis = 1) but that is the *point* of the rule slot: cheap, reproducible, `--with-llm=false` safe. No reason to drop.
- **`ditto_plm` (incumbent).** The supervised PLM anchor for the `learned` axis and the only existing `missing_value_tolerant` member. Scores 3 on signal diversity and SOTA; runtime 2 (MPS-preferred), determinism 2 (seed-pinned but PLM fine-tuning can have residual non-determinism). Total 13. Anchors the roster.
- **`llm_matcher` (incumbent, promote).** Zero-shot LLM prompt path. Currently opt-in with `enabled_by_default: false`. Proposing to keep opt-in but also satisfy the `matching_type: llm` axis requirement the plan wants. Runtime 2 (API budget-bound); determinism 2 (temperature=0 stable but provider drift possible). Total 12.
- **Sudowoodo (add).** Scores 3 on signal diversity: contrastive self-supervised learning is a structurally distinct failure mode from Ditto's supervised pair-classifier, and the same encoder drops into a FAISS/HNSW blocker (so it simultaneously contributes a `hybrid`-capable blocking signal). SOTA 3 (ICDE 2023, still the contrastive-EM reference). Integration 2 (Apache-2.0, PyTorch, Megagon Labs) — a new adapter but similar shape to our `DittoMatcher`. Determinism 2, runtime 2 (MPS/GPU preferred for pretraining but inference is CPU-tractable). Total 12. **This is the candidate the plan explicitly names.**
- **Splink (add).** Scores 3 on signal diversity: Fellegi-Sunter probabilistic linkage is a distinct signal class (likelihood-based, not PLM / not rule-pattern), and most importantly it ships *built-in* NaN-aware comparisons — NULL attribute values contribute 0 to the match weight rather than penalising the pair. Fills [em_portfolio.md Gap 6](em_portfolio.md#gaps-to-investigate-in-c22) which was explicitly flagged as the axis where the committee collapses to Ditto-only. SOTA 2 (classical method, actively maintained), integration 2 (MIT-licensed pip package, pure-Python with DuckDB backend), determinism 3 (seeded EM is fully reproducible), runtime 3 (CPU-tractable, no GPU at all). Total 13. Non-PLM, non-LLM fallback that keeps the `missing_value_tolerant` axis satisfied when Ditto is disabled.

### Deferred

- **SC-Block (Defer).** Signal diversity only 2 — Sudowoodo's encoder already produces blocking embeddings, so SC-Block and Sudowoodo occupy the same blocking slot with the supervision-regime being the only difference (SC-Block supervised, Sudowoodo self-supervised). Adding both would double-count the contrastive signal. SOTA is strong (ESWC 2024, Mannheim-preferred) and the clean 50%-smaller-candidate-set result is attractive, so this is a genuine *Defer* rather than Drop: if blocker-quality measurement shows SC-Block meaningfully outperforms Sudowoodo's blocker, swap them. Total 11; held back by redundancy, not merit.
- **HierGAT (Defer).** Fills Gap 1 (graph-attention) which no other candidate covers. But: **no LICENSE file** in [the public repo](https://github.com/CGCL-codes/HierGAT) — the [paper card](../../literature-search-generation/hiergat_graph_attention_em/paper.md) flags this as a redistribution risk. Integration cost 1 (hierarchical graph builder + GAT stack, no PyPI). Runtime 1 (GPU-required per the paper's published setup). Total 9 — misses the cutoff. Signal axis is attractive enough that the user may override, so this is Defer not Drop. **Flag for user: accept the no-LICENSE risk, request upstream to add one, or wait for a graph-attention alternative with a clear licence?**
- **MatchGPT (Defer).** Fills Gap 4 (retrieval-augmented LLM stack) and is Mannheim-preferred. But: **no LICENSE file** in [the repo](https://github.com/wbsg-uni-mannheim/MatchGPT) — same redistribution risk as HierGAT. Signal diversity only 2 because the axis (`llm`) is already covered by the incumbent `llm_matcher`; MatchGPT's retrieval-augmented prompting is a stronger variant of the same axis, not a new class. Total 11. **Flag for user: accept the no-LICENSE risk and add it as an LLM-axis upgrade, or stay with `llm_matcher` plus optional MatchGPT-style prompt templates ported into our own adapter?**
- **Unicorn (Defer).** Same reasoning the C1.3 SM shortlist used for the SM side: DeBERTa multi-task model that requires a per-domain fine-tune on labeled matching pairs to reach its advertised numbers. We have EM gold in companies-small but the multi-task advantage only pays off with training across several tasks simultaneously, which is more plumbing than a single committee slot warrants. Also GPU-required. Signal overlap with Ditto (both are supervised PLM matchers) on the learned axis — diversity 2, not 3. Total 8.
- **AnyMatch (Defer).** SLM zero-shot (GPT-2 124M, instruction-tuned on 9 EM datasets). The plan's expected-addition line names Sudowoodo *or* a retrieval-augmented LLM — AnyMatch is neither but overlaps the "label-light learned" niche Sudowoodo occupies. One label-light learned member suffices; keeping AnyMatch on the bench for possible swap-in if Sudowoodo's contrastive pretraining cost on companies data is too high. Total 11.
- **TailorMatch (Defer).** [Paper card](../../literature-search-generation/finetuning_llm_em/paper.md): fine-tuned generative LLM with structured-explanation augmentation. Mannheim-preferred and SOTA 3 on WDC-80-corner, but integration cost 1 (per-dataset LoRA fine-tune on Llama-8B, GPU-required) and runtime 1 (GPU + OpenAI budget for the explanation-generation step). Total 9. Revisit post-S5 when Phase C fine-tuning is the main focus.
- **Magellan (Defer).** Classical ML (DT/RF/SVM) on our existing comparator vectors, BSD-licensed, CPU-only, deterministic — a solid non-PLM learned member. But Splink dominates it on the only axis Magellan adds over rule members: NaN tolerance (Magellan's sklearn classifiers choke on NaN without imputation). Splink also beats Magellan on the portfolio gap it was meant to fill (Gap 6). Total 11 — held back by redundancy with Splink, not by individual weakness. Swap-candidate if Splink's EM parameter learning proves unstable on our companies-small gold size.

### Dropped

- **Jellyfish (Drop).** Instruction-tuned Llama 2/3 7B–13B for EM + SM + error detection + imputation. Signal diversity 1 (same axis as `llm_matcher`, TailorMatch, MatchGPT — fine-tuned LLM). Integration 1 (download 7B/13B weights, HuggingFace `transformers` stack, GPU-required). Runtime 1 (7B weights do not fit our MPS budget comfortably). Total 7. Superseded by TailorMatch on the same axis.
- **ComEM (Drop).** Pure prompt-strategy variant of the existing `llm_matcher`. Signal diversity 1 — same axis, same model class, just a different prompt template. Total 9. No diversity win.

## Proposed roster (7 members)

1. `token_rule` *(kept, rule / lexical)*
2. `embedding_rule` *(kept, rule / embedding)*
3. `standard_rule` *(kept, rule / lexical)*
4. `ditto_plm` *(kept, learned / embedding, `missing_value_tolerant=true`)*
5. `llm_matcher` *(kept; **promote to on-by-default** to satisfy the plan's `matching_type=llm` constraint)*
6. **`sudowoodo_contrastive`** *(new, learned / embedding, label-light; paper card [sudowoodo_contrastive_em](../../literature-search-generation/sudowoodo_contrastive_em/))*
7. **`splink_probabilistic`** *(new, learned-probabilistic / lexical, `missing_value_tolerant=true`; paper card [splink_probabilistic_linkage](../../literature-search-generation/splink_probabilistic_linkage/))*

### Axis-coverage audit against plan §C2 required axes

- `blocking_type`: `lexical` (3 members: 1, 3, 7), `embedding` (4 members: 2, 4, 5, 6), **`hybrid` still absent** — Sudowoodo's FAISS backing is de-facto hybrid-capable but we are wiring it as `embedding` in the YAML; `hybrid` promotion is a future C2.5 if SC-Block lands.
- `matching_type`: `rule` (3 members: 1, 2, 3), `learned` (3 members: 4, 6, 7), `llm` (1 member: 5) — all three axes present; `llm` is now on-by-default so the axis isn't conditional on `--with-llm`.
- `missing_value_tolerant`: `ditto_plm` + `splink_probabilistic` — two independent NaN-tolerant members, so the axis no longer collapses if Ditto is disabled. **Plan constraint satisfied.**
- Deterministic fallback: members 1, 2, 3, 7 run without any LLM or PLM — 4 of 7 members are deterministic + reproducible, preserving `--with-llm=false` reproducibility.

### Risks flagged for the user

- **Licensing (HierGAT, MatchGPT).** Both have no LICENSE file. Not shortlisted above on that ground, but the user can override. If overridden, vendor them under `usecases_synthetic/third_party/` with an `ORIGIN.md` documenting the risk (same pattern as Magneto in C1.5).
- **GPU requirement for Sudowoodo pretraining.** The contrastive pretraining step wants a GPU; inference is CPU-tractable. Recommendation: pretrain once (offline, either Megagon's released checkpoint or our own companies-pretrain), then ship the encoder as a pinned artifact the way we ship Ditto.
- **LLM budget for promoting `llm_matcher` to on-by-default.** Currently opt-in to keep LLM spend bounded. Promoting it costs O(candidate_pairs) LLM calls per run. Rough estimate for companies-small (20 × 20 blocked down to ~50 pairs): ≈ 50 calls/run × 15 runs/week ≈ 750 calls/week, well under the shared [cross_cutting.md §LLM hygiene](../../plans/cross_cutting.md) budget. Scales linearly with the corpus — S4b full `companies` would be ~3-5× this, still fine; S5 full-scale runs need an explicit budget check.

## Open questions for the user (hooks for C2.4)

1. **HierGAT under the no-LICENSE risk?** Either (a) vendor under `usecases_synthetic/third_party/hiergat/` with an `ORIGIN.md` risk note (same pattern as Magneto in C1.5 but Magneto is Apache-2.0 so the precedent is weaker), (b) reach out to CGCL-codes maintainers to request a LICENSE, or (c) skip the graph-attention signal until a cleanly-licensed alternative surfaces (CSGAT is paper-only as of 2026-04-20; no obvious replacement).
2. **MatchGPT under the no-LICENSE risk?** Same tri-option as HierGAT. Alternatively: port MatchGPT's prompt-template *designs* into our own `LLMBasedMatcher` adapter under `usecases_synthetic/lib/` — ideas aren't copyrighted; prompt text is short enough to re-author. That route sidesteps the licence question entirely.
3. **Should `llm_matcher` be promoted to on-by-default** or stay opt-in (per the SM committee's `--with-llm` convention)? The plan's `matching_type: llm` axis wants an always-on LLM member; the SM committee treats LLMs as opt-in. Pick one convention for consistency across the two committees.
4. **Splink over Magellan for the CPU / NaN-tolerant slot?** The rubric puts Splink slightly ahead (13 vs 11), but Magellan has a much smaller learning surface (scikit-learn classifiers) and reuses our existing comparators. If the user prefers "add the simpler thing first", Magellan is the safer debut and Splink can land in C2.5.

## Required-axes update (for C2.4)

The current [em_committee.yaml](../../usecases_synthetic/config/committees/em_committee.yaml) `required_axes` block enforces `blocking_type ∈ {lexical, embedding}`, `matching_type ∈ {rule, learned}`, `missing_value_tolerant ∈ {true}` (singleton). After the proposed roster change:
- `blocking_type` unchanged (`hybrid` postponed until SC-Block or equivalent lands).
- `matching_type` gains `llm` (assuming Q3 resolves in favour of on-by-default).
- `missing_value_tolerant=true` now has two satisfying members (`ditto_plm`, `splink_probabilistic`) — the "collapses if Ditto is disabled" failure mode in [em_portfolio.md §axes covered today](em_portfolio.md#portfolio-anchors-existing-literature-search-generation-entries) is closed.

C2.4 updates the YAML plus the [test_committee_configs.py](../../usecases_synthetic/tests/test_committee_configs.py) fixture to cover the two new members, adds `SudowoodoMatcher` + `SplinkMatcher` adapters under `usecases_synthetic/lib/` (matching the `DittoMatcher` adapter pattern from Phase A0), and runs the smoke test on a 20-row `companies-small` slice.

## Deliverable

Seven-member EM committee with two new members (Sudowoodo + Splink) filling the contrastive / self-supervised learned slot and the CPU + NaN-tolerant probabilistic slot respectively. Four deferrals (SC-Block redundant with Sudowoodo; HierGAT + MatchGPT on no-LICENSE risk; Unicorn, AnyMatch, TailorMatch, Magellan on cost / redundancy) and two drops (Jellyfish, ComEM on axis overlap) documented. Four open questions surfaced for the user, covering the two no-LICENSE candidates and the two convention choices (`llm_matcher` opt-in vs on-by-default, Splink vs Magellan). Proceed to C2.4 once the user picks the final roster.

**Superseded by User Decision block at the top of this document.** The final user-directed matching-committee roster is 4 members (`ditto_plm`, `llm_matcher`, `magellan`, `comem`) — Unicorn was in the 2026-04-21 freeze but was dropped on 2026-04-22. The scored shortlist above is preserved as evidence.
