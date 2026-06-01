# plan_e2e_metrics.md

End-to-end pipeline-wide evaluation metrics for PyDI. Compare the
fused output of a full integration pipeline against a reference
("silver-standard") dataset to characterize overall pipeline quality
in a single sweep, complementing the existing per-stage evaluations
in [PyDI/schemamatching/](../PyDI/schemamatching/),
[PyDI/entitymatching/evaluation.py](../PyDI/entitymatching/evaluation.py),
and [PyDI/fusion/evaluation.py](../PyDI/fusion/evaluation.py).

**Status:** §6 / §6.5 closed; **T1 + T4 landed 2026-05-27**; **v2
metric rework landed 2026-05-28** after a realistic-scenario audit
surfaced enough blind spots to warrant restructuring. See "v2
rework" subsection below for what changed. The text below
through T1 is original; v2 rework summary appended at end.

New
[PyDI/evaluation/](../PyDI/evaluation/) subpackage with
`silver_standard.py` (SilverStandard dataclass + `load_workflow_silver`
+ `load_synthetic_silver`), `distributional.py` (Wasserstein-1 / JS /
TV helpers, schema-diff, cluster-size summary, type-routed per-column
metrics, universal `column_drift`), `clustering.py` (BCubed P/R/F1,
ARI, NMI, pairwise P/R/F1, greedy cluster alignment via
`build_record_groups_from_correspondences`), `attribute_quality.py`
(§3.6 + §3.7.1–§3.7.7 metrics), `composite.py` (tier-weighted
composite per §6 Q6 — drops conflict-only contribution when no
conflicts were evaluable so perfect pipelines on no-disagreement data
don't get penalised), `panel.py` (orchestrator emitting `panel.json`
/ `panel.csv` / `schema_diff.json` / `column_metrics.csv` /
`cluster_alignment.csv` / `composite_score.json`). 42 tests in
[tests/evaluation_test/](../tests/evaluation_test/) cover the
fixtures from §7 T4: identity panel, over-merge BCubed drop,
missingness via `nan_rate_delta`, schema-diff with pipe-only column,
list-attribute set-F1 regression, plus loader round-trips for both
synthetic CSV and workflow XML silvers.

**Open follow-ups from T1.**
- **Synthetic per-cell provenance.** The current synthetic silver CSV
  emitted by
  [usecases_synthetic/lib/fusion_silver_standard.py](../usecases_synthetic/lib/fusion_silver_standard.py)
  carries only the cluster-level `source_ids` member list, **not**
  per-cell winning-source provenance — contrary to the plan's earlier
  claim at §3.7.2 ("emitted by fusion_silver_standard for synthetic").
  `load_synthetic_silver` therefore returns `cell_provenance=None` and
  §3.7.2 source-attribution + §3.7.7 synthesis-rate are skipped with
  a panel warning on synthetic-silver runs. Extending the silver
  builder to surface per-attribute winning-source is a separate
  workstream — not blocking the rest of T1.
- T2 (use-case YAML schema additions), T3 (already covered by
  `load_workflow_silver`), T5 (tutorial notebook), T6 (wire the panel
  into use-case notebooks) all pending.

---

## v2 metric rework (2026-05-28)

The audit documented in
[docs/tutorial/e2e_evaluation/e2e_evaluation_metrics.md](../docs/tutorial/e2e_evaluation/e2e_evaluation_metrics.md#cross-cutting-audit--what-the-panel-can-and-cant-see)
("Cross-cutting audit") surfaced concrete blind spots in the v1
panel. Three structural problems:

1. **Tier 2 distributional metrics are blind to histogram-preserving
   record swaps.** A pipeline that emits the right overall column
   distribution while assigning the wrong values to every record
   scores `column_drift = 0` / `js_divergence = 0` — proven on
   music silver with US ↔ DE country swaps (Tier 2 = 0, Tier 4
   accuracy = 0.77) and duration column rotations (Tier 2 = 0,
   Tier 4 accuracy = 0.0005).
2. **Cluster-size metrics are blind to assignment errors.** Full
   shuffle preserving cluster sizes gives size W1 = 0 / size JS = 0
   while BCubed F1 = 0.35, ARI = 0. The "cluster shape" signal the
   v1 panel had was the wrong kind of shape.
3. **NMI, ARI, pairwise F1 mostly duplicate or mislead.** NMI
   stays at 0.86 on full-shuffle clusterings (BCubed F1 = 0.35);
   ARI is hard to interpret (0 = "as good as random" reads
   confusingly); pairwise inherits big-cluster dominance.

**Changes that landed in v2** (all in [PyDI/evaluation/](../PyDI/evaluation/),
[tests/evaluation_test/](../tests/evaluation_test/), and
[docs/tutorial/e2e_evaluation/e2e_evaluation_metrics.md](../docs/tutorial/e2e_evaluation/e2e_evaluation_metrics.md)):

**Tier renames** in `panel.json` (no API change beyond keys):

- `tier_1_coarse` → `entity_coverage`
- `tier_2_distributional` → `column_shape`
- `tier_3_clustering` → `cluster_correctness`
- `tier_4_fused_attribute_quality` → `value_correctness`

**Removed from default panel output** (kept as utility functions):

- ARI / NMI / pairwise P/R/F1 (clustering.py utilities; not in panel).
- Cluster-size Wasserstein-1 + JS divergence (distributional.py
  utility; not in panel — `cluster_size_summary` still callable).
- Universal `column_drift` mean and per-column rows
  (distributional.py utility; not in panel — `column_drift` /
  `column_drift_panel` still callable).
- Categorical `tv_distance` (JS kept).
- Cluster-purity aggregates `cluster_purity_pipe` /
  `inverse_purity_silver` (redundant with `mean_jaccard`).
- Tier 1 `schema_fidelity` block (kept only in `schema_diff.json`
  audit artifact).

**Added in v2**:

- **`source_composition.py`** new module (
  [PyDI/evaluation/source_composition.py](../PyDI/evaluation/source_composition.py)
  ) — same-source collision rate (overall + per source), source-
  mix distribution JS divergence, per-source coverage rate.
  Surfaced under `cluster_correctness.source_composition`.
  Highest-value single addition: catches EM over-merge that BCubed
  averages over (on the smoke-test degraded music pipeline, BCubed
  F1 = 0.955 yet same-source collision rate = 0.263 — a real EM
  bug invisible to BCubed alone).
- **Aligned per-cluster size signals** in `cluster_alignment` —
  `size_match_rate`, `mean_size_delta`, `max_size_overshoot`.
  Surfaced under `cluster_correctness.alignment`. Replaces the
  removed aggregate size W1/JS with the per-aligned-cluster
  equivalent (strictly better — uses the actual alignment instead
  of comparing two distributions blindly).
- **Normalization fingerprint** for text attributes — each text
  attribute's `per_attribute` entry now carries
  `accuracy_similarity_gap` and `mismatch_fingerprint`
  (`normalization_difference_suspected` / `real_value_errors` /
  `mixed`). Surfaces accuracy-vs-similarity_mean drift
  automatically (previously a manual eyeballing step).
- **`cluster_attribute_correctness.csv`** new artifact — per-aligned-
  cluster Tier 4 drill-down with one boolean column per evaluable
  attribute, plus `n_attributes_correct` and `fully_correct`. The
  Tier 4 analog of `cluster_alignment.csv` for Tier 3.
- **Pattern-based diagnostic warnings** in `panel.warnings` — six
  patterns (ID mismatch, histogram-preserving errors, hidden EM
  over-merge, source coverage regression, normalization differences,
  provenance availability gates). On the v2 smoke-test degraded
  pipeline, all four applicable warnings fire correctly.

**Composite recipe changes**:

- Weights shifted from `0.10 / 0.25 / 0.35 / 0.30` to
  `0.10 / 0.20 / 0.40 / 0.30` (column_shape down 0.05,
  cluster_correctness up 0.05).
- New per-tier recipes use the new signals — see
  `composite.tier_subscore_recipe` in `composite_score.json`.

**Simplifications**:

- `list_attribute_set_metrics` outputs only `set_f1` /
  `set_jaccard` / `count` (precision and recall dropped; redundant).

**Tests**: 49/49 evaluation tests pass after rework (was 47 pre-
rework). Smoke-test on real music silver (4 280 clusters) executes
in ~3 seconds with all 4 applicable diagnostic warnings firing
correctly on a controlled-defect degraded pipeline.

**Open follow-ups (carry-forward from v1, unchanged by v2)**:

- Synthetic per-cell provenance — the synthetic silver CSV doesn't
  carry per-cell winning-source provenance, so source attribution
  + synthesis rate are skipped with a panel warning on synthetic
  silvers. Extending the silver builder is a separate workstream.
- T2 / T5 / T6 — see original plan.

---

## v2.1 KGpipe-paper adoption (2026-05-28)

After reading Hofer & Rahm 2025 ("KGpipe", arXiv:2511.18364), three
additions adapted from their KG-pipeline benchmark to PyDI's
tabular world. All additive — no removals, no renames.

### Added: `column_shape.validity_per_column`

New module
[PyDI/evaluation/constraint_validity.py](../PyDI/evaluation/constraint_validity.py).
Per-column rate of cells that (a) parse to the declared
`column_types` tag and (b) satisfy declared constraints (range /
enum / regex / format / length / size). Computed on both pipe and
silver; only negative deltas (pipeline regressions) penalise the
composite. New optional `column_constraints` kwarg to
`compute_e2e_panel`. New diagnostic warning when any column drops
≥ 5pp.

Concept: paper's semantic-tier `O-LT/F` literal-format check,
adapted to tabular columns (`pd.to_numeric`/`pd.to_datetime` +
per-column constraints) instead of RDF datatype validation.

Catches a class of bugs none of the v2 metrics see: cells that
parse but violate declared range/enum constraints (e.g.
`release-year = 9999` in a column constrained to `[1900, 2030]`).
Smoke-test on music silver with 5% out-of-range `duration`
injection: validity rate cleanly drops from 1.00 to 0.95 on pipe,
silver unchanged, diagnostic warning fires.

### Added: `semantic_value_similarity` callable (Tier 4, opt-in)

New optional kwargs `semantic_value_similarity` (a `Callable[[str,
str], float]`) and `semantic_value_threshold` (default 0.85). When
provided, mismatched **text or categorical** cells get a per-cell
similarity score; `per_attribute[*]` gains `semantic_accuracy`,
`semantic_similarity_mean`, and `semantic_vs_strict_gap`. The
normalization fingerprint upgrades to
`normalization_difference_confirmed` (vs `suspected`) when the gap
> 0.10. New warning calls out confirmed cases.

Concept: paper's `~R'_KG` reference-KG overlap with embedding-based
literal similarity. Caller supplies whatever similarity function
they like (sentence-transformers, OpenAI embeddings, dictionary
lookup, ...) — no PyDI dependency on a specific embedding model.

Smoke-test: pipe renames `release-country` to ISO codes (USA/UK/DE),
caller-supplied similarity declares them equivalent to silver's
full names. Result: accuracy=0.78, semantic_accuracy=1.00,
gap=0.22, fingerprint flips to `normalization_difference_confirmed`,
warning fires.

This is the **only normalization-robust value-comparison surface**
the panel offers when per-cell provenance (source attribution,
§3.7.2) isn't available — directly addresses the "biggest single
weakness" the v2 audit documented.

### Added: `resource_usage` block (orthogonal, opt-in)

New optional kwargs `pipeline_duration_seconds`,
`pipeline_peak_memory_mb`, `pipeline_api_cost`,
`pipeline_api_cost_currency`. When any are provided, the panel
emits a `resource_usage` block; otherwise omitted. **No composite
contribution by default** — cross-pipeline normalisation needs a
reference the panel can't infer.

Concept: paper's resource tier (`Q-D`, `Q-M`, `Q-C`). We made it
opt-in because the panel can't measure pipeline runtime/memory
itself (by the time the panel runs, the pipeline is done — the
caller wraps execution with timing).

Smoke-test: caller passes `duration_seconds=3215.4`,
`peak_memory_mb=8192.0`, `api_cost=1.85`. Block appears verbatim
in `panel.json["resource_usage"]`.

### Deliberately not adopted

- **Graph density, disjoint types, relation direction, untyped
  entities** — KG-specific.
- **Fuzzy entity overlap (`~R_SE` label-based)** — PyDI tabular
  workflows have stable source IDs; less applicable.
- **EM P/R / OM P/R as panel metrics** — already covered by
  per-stage `PyDI.entitymatching.evaluation`.
- **Statistical-tier raw counts (fact count, type count, relation
  count)** — most are already in Tier 1 row counts and Tier 2
  `cardinality_delta`.

### Tests

71/71 evaluation tests pass (22 new). Smoke-test on the real music
silver (4 280 clusters) with three concurrent realistic defects —
range-violating durations, ISO-code country renames, simulated
resource report — produces correct values for all three new
metrics and fires the new diagnostic warnings cleanly.

**Documentation convention.** Every metric in this plan and in the
eventual implementation **must** be accompanied by a plain-language
explanation framed in terms of the pipeline vs silver-standard
comparison — what the metric *answers* about the pipeline, with a
small worked example. The audience is a PyDI user reading the panel
output, not a statistics expert. Both the docstrings and the panel
report should follow this convention; if a metric can't be explained
simply in that frame, it doesn't belong on the panel.

---

## 1. Goal & scope

After a pipeline (schema matching → normalization → entity matching →
fusion) produces a fused DataFrame `D_pipe`, we want a small suite of
metrics that summarize *how close* `D_pipe` is to a reference
dataset `D_silver`.

**Silver-standard source — resolved.** Only one silver source:
[usecases_synthetic/lib/fusion_silver_standard.py](../usecases_synthetic/lib/fusion_silver_standard.py)'s
output, which IS the human-baseline silver (the per-domain fusion
stack extracted from each `usecases/<domain>/<domain>_workflow.ipynb`
applied to the pooled clusters). There is no separate "pool-based
vs human-baseline" duality — both names refer to the same artifact.

The runner consumes:
- `D_pipe`: fused DataFrame from the pipeline.
- `correspondences_pipe`: the correspondence DataFrame (post-clusterer
  output: `id1, id2, score, notes`) — same artifact the fusion engine
  already takes. Cluster membership is reconstructed internally via
  `PyDI.fusion.engine.build_record_groups_from_correspondences`,
  matching what fusion does.
- `silver`: the silver-standard artifact (fused DataFrame + its own
  membership; see §5).
- `column_types`: per-use-case dict mapping column name to type tag
  (`categorical | numerical | text | datetime | list | identifier`).
  Required input; no automatic detection.

**Non-goal.** This plan does not replace the per-stage evaluations.
Those answer "is schema matching / EM / fusion correct?"; this plan
answers "did the whole pipeline reproduce the silver dataset?".

---

## 2. Critique of the three proposed metrics

### 2.1 Row count comparison

**Proposal.** Compare `len(D_pipe)` to `len(D_silver)`.

**Assessment.** Useful as a coarse sanity check — picks up
catastrophic over- or under-merging. Limitations:
- Single scalar; symmetric over-merge / under-merge are
  distinguishable only with extra decomposition.
- Doesn't distinguish "right number of clusters, wrong assignments"
  from "right assignments".

**Recommended refinement.** Report all of:
- `n_pipe`, `n_silver`, absolute diff, relative diff
  (`(n_pipe − n_silver) / n_silver`).
- **Entity-overlap decomposition** when record-level alignment is
  available (see §3.1): `n_shared`, `n_silver_only`, `n_pipe_only`.
  This separates "missing entities" from "spurious entities" and is
  more diagnostic than a single delta.

### 2.2 Cluster size distribution comparison

**Proposal.** KL divergence on the histogram of cluster sizes.

**Assessment.** The right *idea* (cluster size shape is informative
— over-merging shrinks the singleton mass and grows the right tail),
but **KL is the wrong default**:
- **Zero-support sensitivity.** Cluster sizes are unbounded
  positive integers; if `D_pipe` produces a size that `D_silver`
  never produces (or vice versa), `KL(P‖Q)` is `∞` unless smoothed,
  and smoothed-KL becomes hyperparameter-sensitive.
- **Asymmetry.** `KL(P‖Q) ≠ KL(Q‖P)`. We have no principled reason
  to pick one direction.
- **Ignores ordering.** Cluster sizes are ordinal: a histogram where
  the only mass shifts from size-2 to size-3 is "closer" than one
  where mass shifts from size-2 to size-100. KL treats both as
  unrelated bins.

**Recommended refinement.** Primary metric:
- **Wasserstein-1 (Earth Mover's) distance** between the empirical
  size distributions. Symmetric, bounded by the size range,
  interpretable as "average number of records you'd have to move to
  convert one distribution into the other". Available via
  `scipy.stats.wasserstein_distance`.

Secondary metrics (cheap, report all):
- **Jensen-Shannon divergence** as a smoothed symmetric alternative
  to KL — bounded in `[0, log 2]`, no infinity issue.
- **Summary statistics:** singleton rate (% of clusters of size 1),
  max cluster size, mean cluster size, gini of cluster sizes.
  Singletons in particular are extremely diagnostic: over-merging
  collapses singletons, under-merging inflates them.
- Smoothed-KL is offered for completeness but not as the headline.

### 2.3 Per-column value distribution comparison

**Proposal.** KL divergence between value distributions per column.

**Assessment.** Same KL issues as §2.2, *plus*: column data types
matter. The same metric on `genre` (categorical), `release_year`
(numerical), `name` (text/high-cardinality) is semantically
incoherent.

**Recommended refinement — type-aware metric routing.** The
`column_types` config (per §6 Q5) supplies the tag per column; the
panel dispatches on that tag, no detection.

| Column kind | `column_types` tag | Primary metric | Secondary |
|---|---|---|---|
| Categorical (low-card, ≤ ~50 uniques) | `categorical` | JS divergence on freq histogram | total variation distance (TV) |
| Numerical | `numerical` | Wasserstein-1 | KS statistic |
| Text / high-cardinality string | `text` | Wasserstein-1 on **string-length distribution** + JS on **token (or char-n-gram) frequency distribution** | top-k overlap |
| Boolean | `categorical` (with 2 values) | TV on `[True, False]` mass | — |
| Identifier-like | `identifier` | **skip** (record metric instead) | — |
| Datetime | `datetime` | Wasserstein-1 on epoch seconds | — |
| List-valued | `list` | **skip** at column-distribution level — handled cluster-by-cluster in §3.7.3 | — |

**Configuring multi-value delimited strings** (e.g. music's
`genre = "Electronic|Funk / Soul"`). Either pre-split the values
into a list and tag the column `list` (so §3.7.3 set metrics
apply), or leave it as a single string and tag `categorical` (so
the whole delimited string becomes a single histogram bucket).
First is correct when the delimiter is semantically a list
separator; second is acceptable for tiny vocabularies. Document
the choice in the use-case YAML.

**Always also report**, regardless of type:
- **Missingness rate** per column on both sides (Δ = pipe NaN rate −
  silver NaN rate). A normalization step that corrupts cells to NaN
  is invisible to a distribution-of-non-null-values metric but is
  a critical signal.
- **Cardinality** (`n_unique`) per column. Catches degenerate
  collapse (everything becomes one value) and over-fragmentation
  (e.g. case-sensitivity differences).
- **Schema fidelity** at the top level: columns present in
  silver-only, pipe-only, dtype mismatches. If the schemas don't
  align, the per-column metrics are moot — report this before
  anything else.

**Universal column-drift metric — `column_drift`.** The type-routed
metrics above are diagnostic per column but can't be averaged across
columns into one number (different scales, different semantics). For
cross-pipeline sweep comparisons and a quick at-a-glance read we also
emit a *universal* per-column metric on a uniform representation.

- **Metric:** **Jensen-Shannon divergence** (base-2 log so values are
  bounded in `[0, 1]`, symmetric) computed on value-frequency
  histograms built by (i) string-casting every value and (ii) binning
  numerical columns to a fixed number of bins (default 50).
- **Output node:** `column_drift` in the panel — one entry per
  column plus a `mean` key holding the unweighted mean across
  columns. Per-column values also surface as `metric=column_drift`
  rows in `column_metrics.csv` next to the type-routed metric for
  that column.
- **Why JS, not TVD.** JS is log-based and more sensitive to small
  probability differences than the L1-style TVD, which matches what
  the headline is for (catching subtle drift). The §2.3 type-routed
  metric for *categorical* columns is already JS, so the universal
  and the categorical-type-routed values coincide cleanly for that
  column type; for text/numerical the universal is JS on the
  string-cast+binned representation, distinct from the type-routed
  metric.

**Tradeoffs the universal metric hides** — flag these in the panel
docs so readers don't misuse it:
- Numerical columns lose ordinality after binning + string-cast: a
  pipeline that shifts all years by 1 looks the same as one that
  shifts by 1000.
- High-cardinality text columns (names, titles) max out the metric
  on tiny case/whitespace differences and give little signal.
- Missingness and cardinality drift are not reflected — those stay
  on the always-also-report list above.

The grave mistake to avoid: reading only the `column_drift.mean`
and ignoring the per-column values and the type-routed metrics.
Always emit all three layers; document the mean as "use for ranking
pipelines, not for diagnosing them".

---

## 3. Alignment-based clustering metrics

The three proposed metrics are all **aggregate distributional**.
Two pipelines can produce identical row counts and identical column
distributions while assigning records to completely wrong clusters.
We need a co-equal pillar that exploits the record-level alignment.

**Precondition.** Cluster membership for both sides is built from
the correspondences DataFrame passed to the runner (per §6 Q3) plus
the silver's membership table (§5). Both are required inputs.

**Canonical clustering metric set.** All of the following land on
the panel by default. Plain-language framings (per the documentation
convention at the top of this plan) accompany each metric in
docstrings and in the report.

### 3.1 BCubed precision / recall / F1 — primary

**What it answers.** For each source record, "did the pipeline lump
me with the right other records?". Computed per record, then
averaged.

For a record `r`:
- **BCubed precision(r)** = of the records `r` was lumped with by
  the pipeline, what fraction actually belong with `r` per silver?
- **BCubed recall(r)** = of the records `r` should have been lumped
  with per silver, what fraction did the pipeline actually lump `r`
  with?
- **BCubed F1(r)** = harmonic mean.

Final scores = mean over all records. Each record gets one vote
regardless of its cluster's size, which makes BCubed robust to
cluster-size skew — the property that makes it preferable to
pairwise P/R/F1 as the headline clustering metric.

### 3.2 Adjusted Rand Index (ARI)

**What it answers.** Across all pairs of records, what fraction does
the pipeline agree with silver on (whether "same cluster" or
"different cluster"), corrected for what a random clustering would
score on average.

Random clustering scores ~0; perfect agreement scores 1; worse-
than-random is negative. One chance-corrected scalar, useful for
ranking pipelines against each other. Implementation:
`sklearn.metrics.adjusted_rand_score`.

### 3.3 Normalized Mutual Information (NMI)

**What it answers.** If you tell me which silver cluster a record is
in, how much does that reduce my uncertainty about which pipeline
cluster it is in (and vice versa)?

Bounded `[0, 1]`. 0 = the two clusterings are statistically
independent (silver tells you nothing about pipeline); 1 = they
agree up to a relabeling of cluster IDs. Symmetric. Implementation:
`sklearn.metrics.normalized_mutual_info_score`.

### 3.4 Pairwise precision / recall / F1 — secondary

**What it answers.** Of all `n × (n−1) / 2` pairs of records, the
silver and pipeline each label every pair as "same cluster" or
"different cluster". Treat one as labels, one as predictions; report
standard P/R/F1.

Classical and free (we already compute the pairs for BCubed), but
**big clusters dominate** — a cluster of size 100 contributes 4950
pairs; 100 singletons contribute 0. Reported for completeness, but
BCubed is the load-bearing clustering F1.

### 3.5 Cluster alignment table + mean Jaccard

**What it answers.** Not a scalar — a *triage artifact*. For each
silver cluster, find the pipeline cluster it overlaps with most
(greedy maximum-overlap; Hungarian on Jaccard is an alternative
when exact 1-1 alignment matters and the cluster count is small
enough). Emit a CSV row per silver cluster:

| silver_cluster_id | best_pipe_cluster_id | overlap_count | silver_size | pipe_size | jaccard |
|---|---|---|---|---|---|

Plus aggregates:
- **Mean Jaccard** of aligned pairs (headline scalar from this
  surface).
- **Matched-cluster rate.** % of silver clusters with an aligned
  pipeline cluster whose Jaccard exceeds a threshold (default 0.5).
- **Cluster purity** (per pipeline cluster): largest silver-cluster
  share / pipeline-cluster size, averaged.
- **Inverse purity** (per silver cluster): symmetric variant.

The table is the most useful artifact when the user needs to point
at *which* clusters went wrong; scalars summarize, the table
explains.

### 3.6 Fused-attribute quality (on aligned clusters)

**What it answers.** Once silver and pipeline clusters have been
aligned, for each aligned pair of clusters and each attribute, does
the pipeline's fused value match silver's fused value?

Per attribute:
- Categorical / text: exact-match accuracy + normalized
  Levenshtein similarity.
- Numerical: MAE, MedAE, % within tolerance.
- Datetime: absolute time delta in days.

Aggregated micro- and macro-averaged across aligned clusters. This
is essentially `PyDI/fusion/evaluation.py` semantics applied
**post-clustering against silver**, which is a different scope than
its current usage (compare to a pre-aligned fusion gold set). The
shared utility code in `PyDI/fusion/evaluation.py` should be
factored so this can call it without duplicating logic.

---

### 3.7 Fusion-specific quality metrics

§3.6 covers row-level fused-value correctness; §2.3 covers aggregate
column-shape correctness. The metrics in this subsection sit between
them — each isolates a fusion-quality signal that the other two
either dilute or miss entirely. All require cluster alignment (§3.5).
§3.7.2 and §3.7.7 additionally require per-cell source attribution:
silver-side is always available (parsed from XML or
fusion_silver_standard output, §5); pipeline-side is optional and
gated on §6.5 Q8.

#### 3.7.1 Conflict-only fused-attribute accuracy

**What it answers.** Restricted to clusters × attributes where the
input source records actually disagreed, did the pipeline pick the
same fused value as silver?

§3.6's overall accuracy is dominated by trivial cells where all
sources agreed — every reasonable fusion policy gets those right.
Conflict-only accuracy isolates the cells where fusion actually had
to decide between competing values, which is the real test of
fusion strategy. Two pipelines can score identically on §3.6 yet
differ sharply here.

Computed by walking the pre-fusion clustered values, flagging each
(cluster, attribute) cell as "conflict" if ≥ 2 distinct non-null
values appeared across sources, then averaging §3.6 metrics over
conflict cells only.

#### 3.7.2 Source-attribution distribution per attribute

**What it answers.** Across clusters, which source "won" each fused
cell, and does the pipeline's distribution of winners match
silver's?

For each attribute, build the histogram `P(winning_source | attribute)`
on both sides. Compare with JS divergence (or TV). Diagnoses
fusion-*policy* drift independently of output-value correctness:
when sources happen to agree, every policy looks right on §3.6, but
this metric catches "pipeline never picks Forbes for `industry`" or
"pipeline always prefers DBpedia regardless of trust" bugs that
value comparison alone cannot.

**Composite provenance** (per §6.5 Q9): when a fused cell carries
multiple source IDs (e.g. silver's `provenance="A+B"` marking union
or synthesis), mass is split equally across the listed sources —
`A+B` contributes 0.5 to each bucket. Synthesis-vs-pick differences
between silver and pipeline are captured separately via §3.7.7.

**Availability** (per §6.5 Q8): requires `cell_provenance_pipe` as
a runner argument. Skipped silently with a panel warning if the
pipeline doesn't pass it; silver-side provenance is always
available (parsed from XML for non-synthetic, emitted by
fusion_silver_standard for synthetic).

#### 3.7.3 Multi-truth / list-valued attribute agreement

**What it answers.** For attributes whose value is a *set* (genres,
platforms, authors, languages, industry tags), do the pipeline's
sets match silver's sets per aligned cluster?

Per aligned cluster × set-valued attribute:
- **Set precision** = |silver ∩ pipe| / |pipe|
- **Set recall** = |silver ∩ pipe| / |silver|
- **Set F1**, **Jaccard**

Averaged across clusters per attribute, then aggregated.

Critical for any domain with list-typed attributes, which is most
PyDI use cases. The §2.3 per-column distribution metrics handle
these badly: flattening a list column to a bag-of-tokens loses the
per-cluster set structure, so a pipeline that swaps every cluster's
set with another cluster's set scores identically to silver under
§2.3 but fails badly here.

**Note on ordered lists.** Some list-valued attributes carry order
information (music's `tracks` is a track listing). V1 treats every
`list` column as an unordered set, which is right for genres /
languages / tags and acceptable for tracks (most consumers don't
care about order). An ordered-list metric (e.g. Kendall tau on
matched element ranks) is a future refinement, not part of v1.

#### 3.7.4 Per-attribute density / coverage

**What it answers.** Per attribute, what fraction of fused rows
have a non-null value, and how does that compare to silver?

Catches: pipeline never fills attribute X that silver fills 80% of
the time (fusion is dropping a column's signal); pipeline forces a
value when silver correctly leaves it null because sources
disagreed irrecoverably (fusion is fabricating).

Partly overlaps with §2.3's per-column NaN-rate Δ, but framed at
the row × attribute fusion level rather than the column
distribution level. Surface both Δ values in the panel; they tell
the same story in different units.

#### 3.7.5 Conflict rate (diagnostic context, not quality)

**What it answers.** What fraction of clusters had at least one
attribute where input sources disagreed? Pipeline vs silver.

Not a quality metric on its own — it's the denominator for
interpreting §3.7.1. A pipeline reporting 99% fused-attribute
accuracy on a dataset with 2% conflict rate is mostly telling you
about clustering quality; the same number on a 60% conflict rate
dataset is a genuine fusion result. Always report conflict rate
alongside conflict-only accuracy so the reader can calibrate.

#### 3.7.6 Per-cluster fully-correct rate

**What it answers.** Of all aligned clusters, in what fraction were
*all* attribute values simultaneously correct (per §3.6)?

§3.6's macro-averaged accuracy can mask "right values on average,
never all right at once" failure modes. Downstream consumers that
treat a fused entity as a single record (recommender features,
lookups, joins) care more about fully-correct entities than about
partial correctness. Cheap to compute once §3.6 is in place: per
aligned cluster, AND across attribute-level correctness flags;
report the mean of the resulting boolean.

#### 3.7.7 Synthesis rate per attribute

**What it answers.** Per attribute, how often does each side
produce a fused value drawn from *multiple* sources (composite
provenance) rather than picking one? And does the pipeline's
synthesis rate match silver's?

For each attribute, compute the fraction of cells whose
`source_ids` list has length > 1, on both sides. Report:
- `synthesis_rate_silver[attr]`
- `synthesis_rate_pipe[attr]`
- `synthesis_rate_delta[attr]` = pipe − silver

Catches "pipeline always picks one source where silver
synthesizes" (Δ negative) and "pipeline over-combines"
(Δ positive) failure modes — both of which §3.7.2's
mass-split JS divergence can dilute when the underlying
per-source distributions look similar.

Same availability gate as §3.7.2 — silver always has it; pipeline
needs to pass `cell_provenance_pipe`.

---

## 4. Proposed metric panel (final shape)

Each end-to-end run emits a single JSON + CSV pair:

```
output/<usecase>/<run_id>/e2e_metrics/
    panel.json          # one nested dict with every metric below
    panel.csv           # flat (metric_name, value) for spreadsheets
    schema_diff.json    # column-level schema diff
    column_metrics.csv  # one row per column × metric
    cluster_alignment.csv  # written when correspondences are passed
    composite_score.json # weighted tier-average headline + per-tier
                        #   subscores + the exact weights used
```

### 4.1 Tier 1 — coarse summary
- Row counts: `n_pipe`, `n_silver`, abs/rel diff
- Entity overlap (if alignable): `n_shared`, `n_silver_only`,
  `n_pipe_only`
- Schema fidelity: column overlap, dtype mismatches

### 4.2 Tier 2 — distributional
- Cluster size: Wasserstein-1 (primary), JS-divergence,
  singleton-rate Δ, max-size Δ, mean-size Δ, gini-Δ
- `column_drift`: per-column JS divergence on string-cast +
  binned-numerical histograms, plus a `mean` key (§2.3 universal
  metric)
- Per column type-routed metric (§2.3 table), NaN-rate Δ,
  cardinality Δ — surfaced in `column_metrics.csv`

### 4.3 Tier 3 — clustering
- BCubed P/R/F1 (primary)
- ARI
- NMI
- Pairwise P/R/F1 (secondary)
- Cluster alignment table → `cluster_alignment.csv` + mean Jaccard
  + matched-cluster rate + cluster purity + inverse purity

### 4.4 Tier 4 — fused-attribute quality
- §3.6 per-attribute accuracy / MAE / sim, micro- and macro-averaged
- §3.7.1 conflict-only fused-attribute accuracy
- §3.7.2 source-attribution JS divergence per attribute (skipped
  with a warning when `cell_provenance_pipe` is not passed; see
  §6.5 Q8)
- §3.7.3 set precision / recall / F1 / Jaccard per list-valued
  attribute, averaged across clusters
- §3.7.4 per-attribute density Δ (pipe coverage − silver coverage)
- §3.7.5 conflict rate (pipe + silver, reported as context)
- §3.7.6 per-cluster fully-correct rate
- §3.7.7 synthesis rate per attribute (silver / pipe / Δ — same
  gate as §3.7.2)

---

## 5. Silver-standard data contract

There are two silver sources sharing one shape — the runner doesn't
care which it gets:

- **Synthetic use cases.**
  [usecases_synthetic/lib/fusion_silver_standard.py](../usecases_synthetic/lib/fusion_silver_standard.py)'s
  output (the per-domain workflow-notebook fusion stack applied to
  the pooled clusters). ~4,280 clusters for music; comparable orders
  for games / companies.
- **Non-synthetic use cases.** The hand-authored fusion gold sets at
  `usecases/<domain>/input/fusion/{validation,test}_set.xml`. Each
  `<release>` (or domain equivalent) is a cluster; per-attribute
  XML elements carry a `provenance` attribute naming the source
  record(s) the value came from. **~100 + 100 clusters** per
  domain — much smaller than the synthetic silver, which makes the
  distributional and clustering tiers noisier on non-synthetic
  runs. Panel docs should note "use synthetic silver where
  available".

The runner takes a small bundle, not a Protocol-based adapter:

```python
# PyDI/evaluation/silver_standard.py  (new module)

@dataclass(frozen=True)
class SilverStandard:
    """Reference dataset for end-to-end pipeline evaluation.

    Attributes
    ----------
    fused : pd.DataFrame
        Per-cluster fused values. Columns: ``cluster_id`` plus the
        fused attribute set.
    membership : pd.DataFrame
        Long-form ``(record_id, source, cluster_id)`` used for the
        alignment-based clustering metrics in §3.1–§3.5.
    cell_provenance : pd.DataFrame | None
        Long-form ``(cluster_id, attribute, source_ids)`` where
        ``source_ids`` is a list (composite provenance like
        ``["discogs_4601", "mbrainz_974"]`` for union/synthesis).
        Used by §3.7.2 source-attribution metrics with mass split
        equally across listed sources (§6.5 Q9), and by §3.7.7
        synthesis rate. ``None`` when the silver doesn't carry
        per-cell provenance (rare — both `load_synthetic_silver`
        and `load_workflow_silver` produce it).
    """
    fused: pd.DataFrame
    membership: pd.DataFrame
    cell_provenance: pd.DataFrame | None
```

**Loaders.** Two concrete loaders in
`PyDI/evaluation/silver_standard.py`:
- `load_synthetic_silver(domain)` — reads the silver CSV/JSON
  artifacts that `fusion_silver_standard` writes, plus the pool
  file that produced them.
- `load_workflow_silver(usecase_dir)` — parses
  `input/fusion/validation_set.xml` + `input/fusion/test_set.xml`,
  yielding the fused frame from element values, the membership
  from XML cluster structure (one `<release>` = one cluster), and
  cell_provenance from the per-attribute `provenance="..."` tags
  (composite tags like `"A+B"` get split per §6.5 Q11).

Both return a `SilverStandard`; neither requires a pool to be
pre-built outside of what already exists in the repo.

---

## 6. Resolved design decisions

All seven questions are resolved. Decisions captured here for the
audit trail; the rest of the plan reflects them already.

1. **(Q1 → resolved) Silver source.**
   [fusion_silver_standard.py](../usecases_synthetic/lib/fusion_silver_standard.py)
   IS the human-baseline silver — per its docstring, it applies each
   domain's workflow-notebook fusion stack to the pooled clusters.
   The plan's earlier "human-baseline vs pool-based" duality is
   dropped; there is one silver source. Non-synthetic use cases get
   the panel once pool/cluster artifacts have been produced for them
   (separate workstream, §5).

2. **(Q2 → resolved) Code location.** New `PyDI/evaluation/`
   subpackage (modules listed in §7 T1).

3. **(Q3 → resolved) Membership API.** The runner takes the
   correspondences DataFrame as an argument and rebuilds cluster
   membership internally via
   `PyDI.fusion.engine.build_record_groups_from_correspondences`,
   matching how fusion already derives groups. `df.attrs` does not
   carry cluster membership in PyDI today (only pipeline metadata
   like `fusion_strategy` / `num_groups` per
   [PyDI/fusion/engine.py:593-596](../PyDI/fusion/engine.py#L593-L596));
   there is no `_source_ids` column on fused outputs. The
   correspondence file is the authoritative source.

4. **(Q4 → resolved) Schema mismatch.** Report schema-diff (silver-
   only / pipe-only columns, dtype mismatches) and skip per-column
   metrics on mismatched columns. No silent fuzzy/LLM alignment.

5. **(Q5 → resolved) Column types.** No automatic detection.
   Per-use-case `column_types` dict is a required runner input —
   maps column name to `categorical | numerical | text | datetime |
   list | identifier`. Lives in the use-case YAML config. The
   type-aware routing in §2.3 dispatches off this map directly.

6. **(Q6 → resolved) Composite score.** Yes — emit a single
   weighted average across tiers as `composite_score` in
   `panel.json` / `panel.csv`, plus a separate `composite_score.json`
   with the per-tier subscores and exact weights. Default weights
   (tunable in the use-case YAML):
   - Tier 1 (coarse): 0.10
   - Tier 2 (distributional): 0.25
   - Tier 3 (clustering): 0.35
   - Tier 4 (fused-attribute quality): 0.30
   Per-tier subscore aggregation is the mean of normalized
   `[0, 1]` metric values within the tier (distances inverted to
   similarities; F1s passed through). Documented alongside the
   panel as "use for ranking, not for diagnosing" — same caveat as
   the §2.3 universal headline.

7. **(Q7 → resolved) Tolerance.** Global default **4% relative
   tolerance** for `% within tolerance` on numerical attribute
   quality, with per-column override in the use-case YAML for
   columns where relative tolerance doesn't fit (`release_year`
   wants `±1 absolute`, etc.).

---

## 6.5 Resolved design decisions, round 2 (2026-05-27)

Surfaced during a plan-consistency + music-pipeline review. All
three concerned per-cell provenance for §3.7.2.

8. **(Q8 → resolved) Pipeline-side per-cell provenance.** Optional
   runner argument `cell_provenance_pipe`; if absent, §3.7.2
   source-attribution is skipped and the panel emits a named
   warning. Other Tier 4 metrics still run. Rationale: not all
   pipelines opt into
   [PyDI/fusion/provenance.py](../PyDI/fusion/provenance.py); the
   panel shouldn't gate on it.

9. **(Q9 → resolved) Composite provenance counting.** Split mass
   equally across listed sources — `provenance="A+B"` contributes
   0.5 to A's bucket and 0.5 to B's in the source-attribution
   histogram. Standard per-source space; no combinatorial bucket
   explosion. The multi-source signal is captured separately via
   Q10.

10. **(Q10 → resolved) `synthesis_rate_per_attribute` diagnostic.**
    Yes — add to Tier 4. Per attribute, fraction of cells with
    composite (multi-source) provenance, silver vs pipeline.
    Cheap to compute from the same data Q9 already needs.
    Surfaces "pipeline always picks one source where silver
    synthesizes" failures that the source-attribution JS can
    dilute. Lands as §3.7.7 below.

---

## 7. Implementation todos

§6 + §6.5 are closed; the following lands as a series of small PRs.

**Touchpoints in existing PyDI.** This is primarily a pure
extension — a new `PyDI/evaluation/` subpackage. Existing modules
are imported read-only, not rewritten:
- `PyDI.fusion.engine.build_record_groups_from_correspondences` —
  membership reconstruction.
- `PyDI.utils.cluster_stats` — cluster-size histogram helpers.

Two soft caveats:
- **Possible small extract from `PyDI/fusion/evaluation.py`.** §3.6
  reuses fused-attribute-quality primitives (accuracy / MAE /
  similarity). If those functions are reusable as-is via plain
  import, T1 adds nothing to `fusion/evaluation.py`. If their
  current API only fits the pre-aligned fusion-gold use case, T1
  may extract a helper or two. Decide during T1's
  `attribute_quality.py` work.
- **Pipeline-side opt-in for §3.7.2 + §3.7.7.** These metrics need
  per-cell provenance from the pipeline side, which means the
  pipeline must opt into [PyDI/fusion/provenance.py](../PyDI/fusion/provenance.py).
  The provenance module itself is unchanged — but the use-case
  notebook (or whatever drives the pipeline) needs to wire it in
  if those two metrics should actually compute. Per §6.5 Q8 the
  metrics skip gracefully with a panel warning when this isn't
  done, so it's an extension of *how the pipeline is called*, not
  a breaking change to PyDI core.

No breaking changes to existing PyDI APIs.

- **T1.** New subpackage `PyDI/evaluation/` with:
  - `silver_standard.py` — `SilverStandard` dataclass + both
    loaders from §5 (`load_synthetic_silver`,
    `load_workflow_silver`).
  - `distributional.py` — Wasserstein-1 / JS / TV helpers,
    schema-diff, cluster-size summary (reuses
    [PyDI/utils/cluster_stats.py](../PyDI/utils/cluster_stats.py)
    rather than reimplementing the size-histogram compute),
    per-column type-routed metrics (dispatch on the `column_types`
    input from §1).
  - `clustering.py` — `build_record_groups_from_correspondences`
    consumer; BCubed P/R/F1, ARI/NMI, pairwise P/R/F1, greedy
    cluster alignment table.
  - `attribute_quality.py` — §3.6 + §3.7 metrics. Reuses
    [PyDI/fusion/evaluation.py](../PyDI/fusion/evaluation.py)
    primitives where possible. Numerical tolerance default 4% per
    §6 Q7; per-column override read from the use-case YAML. §3.7.2
    + §3.7.7 are skipped with a panel warning when the optional
    `cell_provenance_pipe` argument is not passed (per §6.5 Q8).
    Composite-provenance counting splits mass equally across listed
    sources (per §6.5 Q9).
  - `composite.py` — tier-weighted composite per §6 Q6; weights
    overridable via YAML.
  - `panel.py` — orchestrates Tier 1–4, writes `panel.json` /
    `panel.csv` / `schema_diff.json` / `column_metrics.csv` /
    `cluster_alignment.csv` / `composite_score.json`.
- **T2.** Use-case YAML schema additions:
  - `column_types`: required mapping for the panel to dispatch
    type-routed metrics.
  - `tolerance_overrides`: optional per-column numerical tolerance
    overrides.
  - `composite_weights`: optional override of tier weights.
- **T3.** `load_workflow_silver` XML→SilverStandard loader (per §5).
  Parses `input/fusion/{validation,test}_set.xml` for each
  non-synthetic use case. Unlocks the panel on non-synthetic
  pipelines without requiring any pool to be built — both silvers
  already exist in the repo. Pool construction for non-synthetic
  use cases (to get a larger silver than ~200 clusters) remains a
  separate follow-up workstream.
- **T4.** Tests under [tests/](../tests/):
  - Synthetic toy fixtures (two records, two clusters, perfect
    pipeline) → identity panel (all 0 distances, F1 = 1, composite
    = 1).
  - Mis-clustered toy → BCubed and ARI drop, cluster-size W1
    nonzero, composite drops in the expected tier.
  - Missingness regression → per-column NaN-rate Δ surfaces it.
  - Schema-diff regression → schema gate fires, per-column metrics
    skipped on mismatched columns.
  - List-valued attribute regression → §3.7.3 set F1 surfaces it.
- **T5.** Tutorial notebook(s) under [docs/tutorial/](../docs/tutorial/)
  demonstrating the panel. **LANDED 2026-05-27** as two companion
  notebooks under
  [docs/tutorial/e2e_evaluation/](../docs/tutorial/e2e_evaluation/):
    - **`e2e_evaluation_tutorial.ipynb`** — end-to-end on one use case.
      21 cells (10 markdown + 11 code): loads the synthetic music
      silver (4 280 clusters), configures `column_types`, identity
      case (composite = 1.0), degraded case with three controlled
      defects (50-cluster over-merge, 20% `name` corruption,
      `tracks` list shrinkage) → composite 0.9549 with tier_4 = 0.873
      as the biggest drop, per-attribute breakdown fingering `name`
      (accuracy = 0.79) and `tracks` (set_recall = 0.84), and the
      six panel artifacts written via `E2EPanel.write`.
    - **`e2e_evaluation_metric_deep_dive.ipynb`** — companion
      tier-by-tier deep dive. 41 cells (21 markdown + 20 code).
      Every metric explained in plain language with a small toy
      example (4-record / 2-cluster fixtures) so each demo can be
      traced by hand. Covers Tier 1 (row diff, entity overlap,
      schema diff), Tier 2 (cluster-size W1/JS/singleton-delta,
      universal `column_drift`, type-routed metrics for each of
      categorical / numerical / text / datetime, NaN-rate +
      cardinality deltas), Tier 3 (BCubed P/R/F1, ARI, NMI,
      pairwise, cluster-alignment table), Tier 4 (fused-attribute
      quality, list set F1, density delta, fully-correct rate,
      conflict-only accuracy + conflict rate context, source
      attribution + synthesis-rate availability gate), composite
      score, and a one-page "what do I read for X?" cheat sheet.
  Both notebooks are `nbconvert --execute`-clean.
- **T6.** Wire the panel into the existing use-case notebooks
  ([usecases/](../usecases/)) as a final cell so every pipeline run
  emits a panel by default. **Benched per user 2026-05-27.**

Sequencing: T1 first with the distributional tier self-contained
(needs `column_types` config but not the silver). T2 lands in
parallel with the first concrete use case. T3 is a follow-up
workstream. T4 grows alongside each tier.

---

## 8. Out of scope

- Per-stage evaluation (already exists; see `evaluation.py` in each
  submodule).
- Drift-over-time / regression-vs-prior-run dashboards. The panel
  per run is a building block for that, but the dashboard isn't
  part of this plan.
- LLM-judge-based "is the output reasonable" scoring. Useful, but
  orthogonal to silver-standard comparison.
- Causal attribution of pipeline-quality drops to specific stages.
  Hard, separately scoped.

---

## Appendix A — Example panel output

Hypothetical run of the music pipeline against the synthetic music
silver standard (4 280 silver clusters). Numbers are illustrative,
chosen to demonstrate a pipeline that performs reasonably but
over-merges slightly and has fusion-conflict trouble on `artist`
and `tracks`. Attributes mirror the real music schema
(`musicbrainz` / `discogs` / `lastfm` source CSVs):
`name`, `artist`, `release-date`, `release-country`, `duration`,
`label`, `genre`, `tracks`.

### A.1 `panel.json`

```json
{
  "usecase": "music",
  "run_id": "2026-05-26_15h32m",
  "silver_source": "fusion_silver_standard/music",
  "headline": {
    "bcubed_f1": 0.872,
    "composite_score": 0.851
  },
  "tier_1_coarse": {
    "n_pipe": 4150,
    "n_silver": 4280,
    "row_count_abs_diff": -130,
    "row_count_rel_diff": -0.0304,
    "entity_overlap": {
      "n_shared": 3980,
      "n_silver_only": 300,
      "n_pipe_only": 170
    },
    "schema_fidelity": {
      "n_columns_shared": 9,
      "n_columns_silver_only": 0,
      "n_columns_pipe_only": 1,
      "n_dtype_mismatches": 0
    }
  },
  "tier_2_distributional": {
    "cluster_size": {
      "wasserstein_1": 0.42,
      "js_divergence": 0.081,
      "singleton_rate_pipe": 0.51,
      "singleton_rate_silver": 0.56,
      "singleton_rate_delta": -0.05,
      "max_size_pipe": 14,
      "max_size_silver": 11,
      "mean_size_pipe": 2.18,
      "mean_size_silver": 2.04,
      "gini_pipe": 0.41,
      "gini_silver": 0.38
    },
    "column_drift": {
      "name": 0.041,
      "artist": 0.063,
      "release-date": 0.054,
      "release-country": 0.031,
      "duration": 0.069,
      "label": 0.058,
      "genre": 0.077,
      "mean": 0.056
    }
  },
  "tier_3_clustering": {
    "bcubed": {"precision": 0.911, "recall": 0.836, "f1": 0.872},
    "ari": 0.814,
    "nmi": 0.893,
    "pairwise": {"precision": 0.864, "recall": 0.715, "f1": 0.782},
    "alignment": {
      "mean_jaccard": 0.783,
      "matched_cluster_rate_at_0.5": 0.918,
      "cluster_purity_pipe": 0.901,
      "inverse_purity_silver": 0.872
    }
  },
  "tier_4_fused_attribute_quality": {
    "macro_accuracy": 0.847,
    "micro_accuracy": 0.881,
    "conflict_only_accuracy": 0.712,
    "conflict_rate_pipe": 0.34,
    "conflict_rate_silver": 0.31,
    "fully_correct_cluster_rate": 0.621,
    "source_attribution_js_per_attribute": {
      "artist": 0.121,
      "release-date": 0.041,
      "duration": 0.033,
      "label": 0.058,
      "genre": 0.072,
      "tracks": 0.094,
      "release-country": 0.028
    },
    "synthesis_rate_per_attribute": {
      "artist":          {"silver": 0.10, "pipe": 0.08, "delta": -0.02},
      "release-date":    {"silver": 0.05, "pipe": 0.03, "delta": -0.02},
      "duration":        {"silver": 0.02, "pipe": 0.01, "delta": -0.01},
      "label":           {"silver": 0.05, "pipe": 0.04, "delta": -0.01},
      "genre":           {"silver": 0.04, "pipe": 0.02, "delta": -0.02},
      "tracks":          {"silver": 0.62, "pipe": 0.32, "delta": -0.30},
      "release-country": {"silver": 0.02, "pipe": 0.01, "delta": -0.01}
    }
  },
  "warnings": [
    "Column 'preview_url' present in D_pipe but not in silver - skipping per-column metrics for it (Q4 policy).",
    "Pipeline singleton rate is 5pp below silver - possible over-merging."
  ]
}
```

### A.2 `panel.csv`

```csv
metric_name,value
n_pipe,4150
n_silver,4280
row_count_rel_diff,-0.0304
n_shared,3980
n_silver_only,300
n_pipe_only,170
cluster_size_wasserstein_1,0.42
cluster_size_js_divergence,0.081
singleton_rate_delta,-0.05
column_drift_mean,0.056
bcubed_precision,0.911
bcubed_recall,0.836
bcubed_f1,0.872
ari,0.814
nmi,0.893
pairwise_f1,0.782
mean_jaccard,0.783
matched_cluster_rate_at_0.5,0.918
macro_attribute_accuracy,0.847
conflict_only_accuracy,0.712
conflict_rate_pipe,0.34
conflict_rate_silver,0.31
fully_correct_cluster_rate,0.621
composite_score,0.851
```

### A.3 `schema_diff.json`

```json
{
  "columns_shared": ["id", "name", "artist", "release-date",
                     "release-country", "duration", "label", "genre",
                     "tracks"],
  "columns_silver_only": [],
  "columns_pipe_only": ["preview_url"],
  "dtype_mismatches": [],
  "column_types_used": {
    "id": "identifier",
    "name": "text",
    "artist": "text",
    "release-date": "datetime",
    "release-country": "categorical",
    "duration": "numerical",
    "label": "text",
    "genre": "text",
    "tracks": "list"
  },
  "skipped_columns_for_per_column_metrics": ["id", "preview_url", "tracks"]
}
```

`tracks` is `list`-typed so its per-column distribution is skipped
(§2.3); cluster-level set F1 / Jaccard from §3.7.3 covers it
instead. `id` is `identifier`-typed and always skipped.

### A.4 `column_metrics.csv`

```csv
column,type,metric,value,silver_value,pipe_value
name,text,column_drift,0.041,,
name,text,length_wasserstein_1,1.4,,
name,text,token_js_divergence,0.038,,
name,text,nan_rate_delta,0.002,0.001,0.003
name,text,cardinality_delta,-12,4218,4206
artist,text,column_drift,0.063,,
artist,text,length_wasserstein_1,0.9,,
artist,text,token_js_divergence,0.057,,
artist,text,nan_rate_delta,0.000,0.000,0.000
release-date,datetime,column_drift,0.054,,
release-date,datetime,wasserstein_1_days,18.4,,
release-date,datetime,nan_rate_delta,0.005,0.012,0.017
release-country,categorical,column_drift,0.031,,
release-country,categorical,js_divergence,0.028,,
release-country,categorical,tv_distance,0.041,,
release-country,categorical,cardinality_delta,0,52,52
duration,numerical,column_drift,0.069,,
duration,numerical,wasserstein_1,1.51,,
duration,numerical,mae,3.4,,
duration,numerical,pct_within_4pct_tolerance,0.881,,
duration,numerical,nan_rate_delta,0.003,0.020,0.023
label,text,column_drift,0.058,,
label,text,length_wasserstein_1,0.8,,
label,text,nan_rate_delta,0.011,0.075,0.086
genre,text,column_drift,0.077,,
genre,text,length_wasserstein_1,0.6,,
genre,text,token_js_divergence,0.063,,
genre,text,nan_rate_delta,0.018,0.082,0.100
tracks,list,set_f1,0.781,,
tracks,list,set_precision,0.823,,
tracks,list,set_recall,0.744,,
tracks,list,mean_jaccard,0.692,,
```

Notes:
- `column_drift` is the universal JS-on-string-cast metric; the
  other rows are the type-routed metrics from §2.3. For
  `release-country` (categorical) `column_drift` ≈ `js_divergence`
  because both compute JS on the same value histogram, modulo
  binning treatment.
- `tracks` is `list`-typed and gets only cluster-level set metrics
  (no `column_drift` row, no per-column distribution metric — see
  §2.3 table).
- `id` (identifier) is omitted entirely.

### A.5 `cluster_alignment.csv`

```csv
silver_cluster_id,best_pipe_cluster_id,overlap_count,silver_size,pipe_size,jaccard
mb_artist_001,pipe_c_0042,5,5,5,1.000
mb_artist_002,pipe_c_0087,3,3,3,1.000
mb_artist_003,pipe_c_0091,2,3,2,0.667
mb_artist_004,pipe_c_0091,1,2,2,0.333
mb_artist_005,pipe_c_0103,4,4,5,0.800
mb_artist_006,pipe_c_0124,2,2,4,0.500
mb_artist_007,,0,2,,0.000
mb_artist_008,pipe_c_0156,3,3,3,1.000
mb_artist_009,pipe_c_0156,2,2,3,0.500
mb_artist_010,pipe_c_0203,1,1,1,1.000
```

Two failure patterns visible:
- Rows 3+4 (silver 003 and 004 both aligning to pipe cluster 0091)
  and rows 8+9 (silver 008 and 009 both aligning to pipe 0156) show
  the pipeline **over-merging** distinct silver clusters.
- Row 7 (silver 007 with no aligned pipe counterpart) shows the
  pipeline **missing** a silver cluster entirely.

### A.6 `composite_score.json`

```json
{
  "composite_score": 0.851,
  "weights": {
    "tier_1_coarse": 0.10,
    "tier_2_distributional": 0.25,
    "tier_3_clustering": 0.35,
    "tier_4_fused_attribute_quality": 0.30
  },
  "tier_subscores": {
    "tier_1_coarse": 0.949,
    "tier_2_distributional": 0.862,
    "tier_3_clustering": 0.872,
    "tier_4_fused_attribute_quality": 0.781
  },
  "tier_subscore_recipe": {
    "tier_1_coarse": "mean(1 - |row_count_rel_diff|, 1 - n_silver_only/n_silver, 1 - n_pipe_only/n_pipe, schema_match_fraction)",
    "tier_2_distributional": "mean(1 - cluster_size_W1_normalized, 1 - column_drift.mean)",
    "tier_3_clustering": "bcubed_f1",
    "tier_4_fused_attribute_quality": "mean(macro_attribute_accuracy, conflict_only_accuracy, fully_correct_cluster_rate, 1 - mean(source_attribution_js))"
  },
  "caveat": "Composite is a ranking number, not a diagnostic. Inspect tier_subscores and per-column metrics to understand failures."
}
```

### A.7 Reading this example

Quick interpretation:
1. **Tier 1 (0.95).** Pipeline produced 3% fewer rows than silver,
   schemas align, one pipe-only column (`preview_url`) — minor.
2. **Tier 2 (0.86).** Cluster sizes are slightly different shape
   (W1 ≈ 0.42 records); singleton rate down 5pp (over-merge hint).
   `column_drift.mean = 0.056` is pulled up by `genre` (0.077) and
   `duration` (0.069) and down by `release-country` (0.031) and
   `name` (0.041).
3. **Tier 3 (0.87).** BCubed-F1 0.87 with precision 0.91 > recall
   0.84 confirms the over-merge. ARI 0.81 / NMI 0.89 agree. The
   `cluster_alignment.csv` table points at *which* clusters
   over-merged.
4. **Tier 4 (0.78).** Macro accuracy 0.85, but conflict-only
   accuracy 0.71 — when sources disagreed, fusion picked wrong
   nearly 1 in 3 times. `artist` source-attribution JS 0.121 means
   the pipeline systematically prefers a different source than
   silver for `artist`. `tracks` set F1 0.78 is the weakest
   attribute (list-typed; cluster-level set metric from §3.7.3).
   The `synthesis_rate` for `tracks` shows the pipeline only
   produces composite (multi-source) values 32% of the time vs
   silver's 62% — the pipeline picks a single source's track
   listing where silver unions them. That single signal (Δ −0.30
   on tracks) explains the set-F1 gap better than the
   source-attribution JS does alone. Fully-correct-cluster rate
   0.62 means 38% of clusters have at least one wrong attribute.
5. **Composite 0.85** ranks the pipeline for a sweep; the panel
   makes clear fusion is the weakest link, specifically on conflict
   resolution and the `artist` source-attribution policy.
