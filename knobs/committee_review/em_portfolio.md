# EM Committee — Portfolio Inventory (C2.1)

Committee in scope: entity matching. Current roster (see [../../usecases_synthetic/config/committees/em_committee.yaml](../../usecases_synthetic/config/committees/em_committee.yaml)):

| # | Member | Signal | Module |
|---|---|---|---|
| 1 | `token_rule` | `TokenBlocker` (lexical) + `RuleBasedMatcher` over `name`/`country` comparators | `PyDI.entitymatching.blocking.token_blocking` + `PyDI.entitymatching.rule_based` |
| 2 | `embedding_rule` | `EmbeddingBlocker` (MiniLM-L6) + `RuleBasedMatcher` (same comparators as #1) | `PyDI.entitymatching.blocking.embedding` + `PyDI.entitymatching.rule_based` |
| 3 | `standard_rule` | `StandardBlocker` on `name_first_token` + `RuleBasedMatcher` | `PyDI.entitymatching.blocking.standard` + `PyDI.entitymatching.rule_based` |
| 4 | `ditto_plm` | `EmbeddingBlocker` + `DittoMatcher` (fine-tuned PLM, NaN-tolerant) | `usecases_synthetic.lib.ditto_matcher.DittoMatcher` |
| 5 | `llm_matcher` (opt-in) | `EmbeddingBlocker` + `LLMBasedMatcher` (zero-shot) | `PyDI.entitymatching.llm_based.LLMBasedMatcher` |

Axes covered today (per `required_axes` in the YAML): `blocking_type ∈ {lexical, embedding}`, `matching_type ∈ {rule, learned}` (plus `llm` opt-in), `missing_value_tolerant ∈ {true}` via Ditto. Missing from the enforced axes: a second `missing_value_tolerant` member (if Ditto is disabled the axis collapses), a `hybrid` dense+sparse blocker, a non-fine-tuned `llm` path promoted to always-on, and signal diversity inside `matching_type=learned` (only Ditto).

## Portfolio anchors (existing `literature-search-generation/` entries)

Only the papers whose method class is *directly reusable* as a committee member (blocker or matcher). Benchmark-only entries are listed in the next section.

| Anchor | Method class | Maturity | Axis it would fill | Dependency cost | Note |
|---|---|---|---|---|---|
| [ditto](../../literature-search-generation/ditto/) | Fine-tuned PLM on serialized pairs + Ditto-DA operators | production ([megagonlabs/ditto](https://github.com/megagonlabs/ditto), Apache-2.0) | `learned` / `missing_value_tolerant` | already wired via `usecases_synthetic/lib/ditto_matcher.py`; GPU/MPS preferred | Current slot 4. Anchor for the "fine-tuned supervised PLM" signal. Any new learned member must be *distinct* from Ditto's BERT-on-serialized-text signal. |
| [anymatch_zero_shot_em](../../literature-search-generation/anymatch_zero_shot_em/) | SLM (GPT-2 / T5-small) instruction-tuned on multi-dataset EM pairs; zero-shot inference | research code ([Jantory/anymatch](https://github.com/Jantory/anymatch), Python) | `learned, zero-shot` (no in-domain labels needed) | new adapter; fine-tune once on released pair mixture, then CPU inference on an SLM | Distinct from Ditto: SLM runs on CPU, zero-shot across domains, no companies-specific training. Good "missing-value-tolerant, no-gold" slot. |
| [lemonade_llm_guided_em](../../literature-search-generation/lemonade_llm_guided_em/) | LLM-guided soft/strong augmentation + LLM→SLM knowledge distillation for weakly-supervised EM | paper-only (no code release tracked in `code_repos`) | `learned, weakly-supervised` + an LLM-distilled SLM path | full re-implementation (augmentation pipeline + alignment layers) — ≥ 1 day | Attractive axis (weak supervision) but zero code makes it high-risk per the plan's "no-code→Integration=0" rule. Flag for C2.3 scoring. |
| [jellyfish_llm_data_preprocessing](../../literature-search-generation/jellyfish_llm_data_preprocessing/) | Instruction-tuned LLM (Llama 2/3, Mistral) covering EM + SM + imputation as a single preprocessing model | research code / HuggingFace model weights (no single GitHub repo in frontmatter; instruction-tuning dataset on HF) | `llm, fine-tuned` (a second LLM path beside `llm_matcher`) | needs GPU + downloaded 7B/13B weights; prompt adapter similar to `LLMBasedMatcher` | Would replace/augment current opt-in `llm_matcher` with a fine-tuned alternative. Covers both EM and SM → shared LLM budget. |
| [finetuning_llm_em](../../literature-search-generation/finetuning_llm_em/) | Fine-tuned LLM EM with structured-explanation augmentation + training-set filtration (TailorMatch, Mannheim WBSG) | research code ([wbsg-uni-mannheim/TailorMatch](https://github.com/wbsg-uni-mannheim/TailorMatch), Python) | `learned, llm-finetuned` (GPT-style instead of BERT-style) | adapter + one-off fine-tune on companies-small pairs; GPU required | Mannheim-local method (preferred per plan). Distinct signal from Ditto (causal LM vs encoder). Overlaps with Jellyfish on "fine-tuned LLM" axis — pick one. |
| [mixer_latent_interpolation_er](../../literature-search-generation/mixer_latent_interpolation_er/) | VAE-encoder + latent-space interpolation (EMix) for ER training-data augmentation | paper-only (no code in `code_repos`) | data-augmentation pre-step (*not a member on its own*) | re-implement VAE + EMix — ≥ 2 days | Augmentation technique, not a matcher. Would feed Ditto's training — belongs in Phase A0, not C2. Exclude from roster. |
| [distiller_knowledge_distill_er](../../literature-search-generation/distiller_knowledge_distill_er/) | Framework for distilling LLM teacher → SLM student for ER (informativeness sampling + multi-teacher) | paper-only (EDBT 2026, no code tracked) | `learned, distilled-llm` (efficient CPU student) | meta-framework — needs picking a teacher + student + running distillation; ≥ 2 days | Same role as LEMONADE (distill LLM into deployable SLM). Pick at most one to avoid redundancy. No code = Integration=0 risk. |
| [comem_match_compare_select](../../literature-search-generation/comem_match_compare_select/) | LLM prompt strategies for EM: binary matching vs pairwise comparing vs set-based selecting | paper-only (COLING 2025, no code tracked) | `llm, prompt-strategy-variant` | adapter over existing `LLMBasedMatcher` to swap prompt template | Cheapest LLM variant to add — it's just a prompt change. Gives a second `llm` path (set-based) with *different* cost profile (O(k) vs O(k^2)). |
| [magellan_em_system](../../literature-search-generation/magellan_em_system/) (discovered while walking INDEX — not in C2 anchor list but directly reusable) | Classical ML-on-hand-crafted-features: DT/RF/SVM/LR/NB over comparator vectors | production ([anhaidgroup/py_entitymatching](https://github.com/anhaidgroup/py_entitymatching), pip, BSD) | `learned, classical-ml` (distinct from Ditto's PLM signal) | adapter around `py_entitymatching`; reuses our existing comparators | Gives a *second* `learned` member that runs CPU-only, is deterministic, and doesn't need gold-pair fine-tuning of a PLM. Mitigates the "Ditto disabled → axis collapses" risk. |
| [data_aug_er_comparative](../../literature-search-generation/data_aug_er_comparative/) | Empirical study of EDA / back-translation / MixDA augmentations for ER | research code ([amazon-science/data-augmentation-for-entity-resolution](https://github.com/amazon-science/data-augmentation-for-entity-resolution)) | augmentation reference (*not a member*) | n/a for committee | Feeds Ditto/Magellan training-data choices. Not a roster candidate. |

## Not usable as committee members (but relevant for search seeds)

- [alaska_benchmark](../../literature-search-generation/alaska_benchmark/) — heterogeneous-web-products benchmark, no matcher.
- [bridging_gap_em_benchmark](../../literature-search-generation/bridging_gap_em_benchmark/) — open-entity benchmark reconstruction; evaluation only.
- [critical_reevaluation_er_benchmarks](../../literature-search-generation/critical_reevaluation_er_benchmarks/) — linearity/complexity re-evaluation of existing benchmarks.
- [embench](../../literature-search-generation/embench/) / [embench_pp_benchmark](../../literature-search-generation/embench_pp_benchmark/) — benchmark generators, not matchers.
- [machamp_entity_matching_benchmark](../../literature-search-generation/machamp_entity_matching_benchmark/) — cross-format EM benchmark; defines tasks, ships no matcher.
- [wdc_products_benchmark](../../literature-search-generation/wdc_products_benchmark/) / [leipzig_er_benchmarks](../../literature-search-generation/leipzig_er_benchmarks/) — benchmark datasets.
- [profiling_em_benchmarks](../../literature-search-generation/profiling_em_benchmarks/) — benchmark profiling toolkit (CompERBench).
- [sigmod_er_contests](../../literature-search-generation/sigmod_er_contests/) — contest retrospective.
- [pretrained_embeddings_er](../../literature-search-generation/pretrained_embeddings_er/) — 17-dataset×12-model experimental analysis; useful for picking a blocker backbone but not a member on its own.
- [cross_dataset_em_edbt2025](../../literature-search-generation/cross_dataset_em_edbt2025/) — cross-dataset evaluation protocol over existing matchers.
- [ground_truth_weakly_supervised_em](../../literature-search-generation/ground_truth_weakly_supervised_em/) — Snorkel-style label-model for weak supervision; ground-truth generator, not an EM member.
- [nl_explanations_em](../../literature-search-generation/nl_explanations_em/) — explanation-generation side-channel; evaluation/interpretability helper.
- [goby_enterprise_benchmark](../../literature-search-generation/goby_enterprise_benchmark/) — enterprise benchmark.
- [frost_benchmarking_platform](../../literature-search-generation/frost_benchmarking_platform/) — result-exploration platform.
- [almost_all_entity_resolution](../../literature-search-generation/almost_all_entity_resolution/) / [christophides_er_survey](../../literature-search-generation/christophides_er_survey/) / [heterogeneity_em_survey](../../literature-search-generation/heterogeneity_em_survey/) — surveys.

## Gaps to investigate in C2.2

The following coverage axes (from the C2 "Coverage gaps to investigate" block in [plans/plan_committee_finalization.md](../../plans/plan_committee_finalization.md#c2--entity-matching-committee)) are not covered by any portfolio anchor with a production code release:

1. **Graph / attention-pooling matchers** (HierGAT, MCAN, DeepMatcher-Hybrid, GNEM). Attribute-level attention over token graphs is structurally distinct from Ditto's flat-serialization signal and from Magellan's hand-crafted features. No portfolio anchor covers this.
2. **Contrastive / self-supervised learned matchers** (Sudowoodo, BarlowMatch, DeepBlocker contrastive variants). Would give a second `learned` member that doesn't need gold labels — complements Ditto and AnyMatch.
3. **Dense+sparse hybrid blockers** (SparkER, JedAI, BLOCKER-style hybrid, FAISS/HNSW-accelerated embedding blocker using the `hnsw` extra already in `pyproject.toml`). Current blockers cover `lexical` and dense-`embedding` but no hybrid.
4. **Production LLM-matching stacks with prompt retrieval** (LEMONADE, TailorMatch retrieval-augmented prompting, MatchGPT). Partial coverage via the portfolio anchors above but none of them ship runnable code; C2.2 must find a retrieval-augmented LLM matcher we can actually drop in.
5. **Small-model zero-shot alternatives to AnyMatch** (e.g. Flan-T5-base tuned on EM, Unicorn's EM head, TabularRAG-style few-shot retrieval). Sanity-check that AnyMatch is still the best SLM candidate in 2025/2026.
6. **CPU-tractable NaN-tolerant rule matchers** so `missing_value_tolerant` isn't 100% Ditto-dependent — a NaN-wildcard comparator or a robust imputation-before-matching step would suffice.

C2.2 runs the external-search queries listed in [plan_committee_finalization.md §C2](../../plans/plan_committee_finalization.md#c2--entity-matching-committee) (HierGAT, Sudowoodo, Jellyfish, AnyMatch, LEMONADE, TailorMatch, Mannheim WBSG repos) and scores the candidates in C2.3.
