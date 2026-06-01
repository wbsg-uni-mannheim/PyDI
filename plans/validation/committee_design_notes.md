# Committee Design Notes

Narrative justification for the committee rosters in `usecases_synthetic/config/committees/`. Each section explains why each member is in the roster, which knob cards predict it will differentiate, and which alternatives were rejected.

## Schema Matching

The SM committee has four enabled-by-default members spanning two signal types (label and instance/duplicate), plus one opt-in LLM member.

**Label-based members (label_jaccard, label_jaro_winkler).** Two string-similarity matchers using different similarity functions. Jaccard (tokenized) captures multi-word header overlaps ("company_name" vs "name_of_company"); Jaro-Winkler (untokenized) captures prefix-similarity for short headers ("city" vs "cityName"). Both are expected to collapse under K8 hard (anonymised/cryptic headers), which is the primary SM difficulty signal (K8 Committee expectations: "string-similarity matchers collapse fast on cryptic/anonymised"). Having two label-based members with different similarity semantics lets us distinguish partial collapse (one survives) from full collapse (both fail), which informs whether K8 hard is over-tuned.

**Instance-based member (instance_tfidf_cosine).** Uses TF-IDF value distributions to match columns regardless of header text. This is the K8-robust anchor: K8 Committee expectations predict "instance-based / embedding matchers degrade more gracefully" under header perturbation. The spread between label-based and instance-based F1 *is* the K8 difficulty signal. A low threshold (0.1) is used because instance-based matching on heterogeneous source schemas (dbpedia JSON vs forbes TSV vs fullcontact XML) produces lower raw similarity scores than label matching.

**Duplicate-based member (duplicate_majority).** Uses known EM correspondences to vote on column alignment. This provides a third signal type orthogonal to both label and instance matching. It is K8-robust (does not read headers) but depends on EM gold quality, so it serves as a cross-stage diagnostic: if duplicate-based SM accuracy drops on a variant, the cause is EM gold degradation, not SM difficulty.

**Rejected: embedding-based SM.** PyDI does not ship a dedicated embedding-based schema matcher. Instance-based with TF-IDF cosine partially fills this niche. Adding a custom embedding SM was considered but rejected as out of scope for M1 — the existing four members already span the required axes.

**LLM member (llm_openai).** Disabled by default to avoid cost and non-determinism in every validation run. K8 and K9 Committee expectations call out LLM-matcher spread, but this is an opt-in diagnostic, not a core committee member. Available via `--with-llm` flag on `measure_baseline.py` and `validate_variant.py`.

## Entity Matching

The EM committee has four enabled-by-default members spanning two blocking types (lexical, embedding), two matching types (rule-based, learned), and includes a missing-value-tolerant member. Plus one opt-in LLM member.

**Standard-blocker + rule-based (standard_rule).** Directly replicates the existing `test_workflow_companies.py` pipeline — StandardBlocker on `name_first_token` with Jaccard comparators on name/country/industry. This is the known-working baseline and the member most sensitive to K1 (paraphrase destroys token overlap), K6 (typos degrade Jaccard), and K3 (missing attributes break comparators). Its inclusion ensures the committee baseline matches the existing companies workflow, making baseline measurements directly comparable to the assert values in the test file (F1 ~0.786 accuracy on forbes-dbpedia).

**Token-blocker + rule-based (token_rule).** Broader recall than StandardBlocker (blocks on all name tokens, not just the first), with the same Jaccard comparators. Sensitive to the same knobs as standard_rule but with different recall/precision trade-off. The spread between standard_rule and token_rule on K6 hard tells us whether the difficulty signal is in blocking recall (standard_rule drops more) or matching precision (both drop equally).

**Embedding-blocker + rule-based (embedding_rule).** Semantic blocking with Jaro-Winkler comparators. K1 Committee expectations: "monotone drop, sharper for lexical blockers than embedding matchers" — this member is the K1-robust anchor. K6: "monotone drop, sharp for lexical/n-gram blockers, mild for embedding blockers" — same role. The blocking-stage spread between token_rule and embedding_rule is the K1/K6 blocking difficulty signal.

**Embedding-blocker + PLM matcher (embedding_plm).** The learned member. K2 expectations: "monotone F1 drop, sharper for similarity-threshold matchers than learned matchers" — PLM is the K2-robust anchor. K3 expectations: "learned matchers with missing-value handling degrade less than rule-based comparators" — PLM handles NaN natively (missing_value_tolerant: true). This member is essential for detecting K3's difficulty signal. Already used in pool construction (plan_algorithmselection.md Tier A), so embeddings are cached and the cost is low.

**Rejected: ML-based matcher (Magellan-style random forest).** Would add another learned member but requires feature engineering and training, with unclear marginal benefit over PLM. The PLM matcher already covers the "learned, missing-value tolerant" axis. Deferred to v2 if the committee spread on K2/K3 is insufficient.

**LLM matcher (llm_matcher).** Same opt-in reasoning as SM. Available via `--with-llm`.

**Metric aggregation.** Macro-F1 across source pairs (forbes-dbpedia, forbes-fullcontact) is the primary metric. Per-pair F1 is reported as secondary. Companies has only 2 labeled source pairs (3 sources, but dbpedia-fullcontact has no gold), so macro-averaging gives each pair equal weight. This was chosen over micro-averaging because the pair sizes differ significantly (~200 vs ~100 gold pairs), and micro would be dominated by the larger pair.

**Threshold policy.** Fixed per-member at baseline measurement time. The same threshold is used on all variants. Moving the threshold per variant would hide the difficulty signal — if a member achieves the same F1 on hard by lowering its threshold, we've measured threshold sensitivity, not task difficulty.

## Fusion

The fusion committee evaluates 2-4 strategies per attribute across 8 attributes, spanning cell-local and trust-weighted approaches.

**Cell-local strategies (voting, longest_string, shortest_string, most_complete, median, maximum, union, earliest).** These decide per-cell without considering source provenance. K10 expectations: "Easy: per-attribute 'favor source X' strategies and voting both win" — cell-local strategies should perform well on easy. K1 expectations: "needs gold extend-to-accepted-set on hard for long string attributes" — the spread between longest_string and voting on hard-paraphrased name fields is K1's fusion difficulty signal.

**Trust-weighted strategies (prefer_higher_trust, favour_sources).** These use per-source trust scores (forbes=3, fullcontact=2, dbpedia=1), matching the existing workflow. K10 expectations: "Medium: per-source-weighted strategies pull ahead of plain voting" — trust-weighted members should outperform cell-local on medium. K10 hard: "per-source-trust strategies fail (no per-attribute concentration to exploit) *and* per-attribute voting fails (correlated bursts flip majorities)" — both strategy types should degrade, with trust-weighted degrading *less* on medium but converging on hard. This spread is K10's primary diagnostic.

**Per-attribute strategy assignment.** Follows the draft in cross_cutting.md with refinements:
- **name** (primary): voting + longest_string + most_complete + prefer_higher_trust. Four strategies because name is the richest signal for K1/K10 spread.
- **assets, revenue** (secondary, numeric): median + maximum + prefer_higher_trust. Median is robust to outliers; maximum is K6-sensitive (noise inflates values).
- **keypeople** (secondary, list): union + voting + prefer_higher_trust. Union captures completeness; voting captures consensus.
- **founded** (secondary, date): voting + earliest + prefer_higher_trust.
- **country, city** (key): voting + shortest_string/favour_sources + prefer_higher_trust.
- **industry** (key): voting + most_complete + prefer_higher_trust.

**Rejected: entity-level provenance reasoning / LLM arbitration.** K10 hard predicts that recovery requires "entity-level provenance reasoning or LLM arbitration." This is acknowledged as the theoretical ceiling but is out of roster for v1 — too expensive for every validation run and not implementable with existing PyDI fusion primitives. If K10 hard shows both cell-local and trust-weighted collapsing to random, the triage report (M10) will recommend this as future work rather than a fix-on-collapse within the current validation cycle.

**Evaluation functions.** Per-attribute evaluation uses the same functions as the existing workflow: `tokenized_match` for strings, `numeric_tolerance_match` (tolerance=0.1) for assets/revenue, `year_only_match` for founded, `set_equality_match` for keypeople. This ensures fusion accuracy numbers are directly comparable to the existing test assertions.
