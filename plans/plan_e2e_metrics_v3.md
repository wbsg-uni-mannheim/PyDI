# plan_e2e_metrics_v3.md — 2D panel restructure (Coverage / Consistency / Correctness × RF / SR / GR)

Plan for migrating the end-to-end panel from its single-axis v2.1
tier model (`entity_coverage` / `column_shape` /
`cluster_correctness` / `value_correctness` + opt-in
`resource_usage`) to the **two-dimensional** Quality × Reference
categorization documented in
[docs/tutorial/e2e_evaluation/metrics.md](../docs/tutorial/e2e_evaluation/metrics.md).

The v3 categorization came out of a colleague brainstorm. It's a
**clean break** from v2.1's tier shape — no backwards-compatibility
shims (consistent with how v2 and v2.1 landed).

**Status**: **phases A–D landed 2026-05-28**. 88 tests passing
(17 net new for v3 paths + gold + RF + task_step_metrics +
auto-built cell_provenance). Markdown rewritten to the v3 shape.
Real-data smoke-test on music silver + workflow-XML gold confirms
all RF / SR / GR sub-blocks populate correctly and all four
diagnostic warnings fire on a degraded pipeline. See "Completion
notes" at the end for what landed and what's deferred.

---

## 1. The new categorization

### 1.1 Quality dimensions

| Dimension | Question | Lives under |
|---|---|---|
| **Coverage** | Did we produce the right set of *things* (entities, facts, source-attributions)? | `coverage` |
| **Consistency** | Does the output respect declared formats and constraints? | `consistency` (placeholder, see §4) |
| **Correctness** | Do the values match a reference? | `correctness` |
| **Resources** | What did the pipeline cost? | `resource_usage` (unchanged from v2.1) |
| **Task/Step** | Per-stage quality (blocking, schema matching, EM) | `task_step` (placeholder, see §4) |

### 1.2 Reference levels

| Level | Code | Source | Notes |
|---|---|---|---|
| Reference-free | **RF** | None | Output-only metrics (density, row count, source distribution). Always computable. |
| Silver reference | **SR** | `load_synthetic_silver` (synthetic) or `load_workflow_silver` (XML) | The "human-baseline notebook output applied to pooled clusters" silver. ~4 280 clusters on music. |
| Gold reference | **GR** | `input/fusion/{validation,test}_set.xml` parsed via `load_workflow_silver` | The hand-curated fusion test set. ~100 + 100 clusters per domain. |

**Important**: SR and GR are both `SilverStandard` bundles under the
hood — `load_workflow_silver` already produces the gold one. The
runner needs to accept *both* and compute SR-applicable metrics
against each. The naming convention (silver vs gold) is purely
semantic — the data structures are identical.

### 1.3 Open question: cluster correctness — EM test set or fusion test set?

The brainstorm itself flags this: *"GR: BCubed P, R, F1 …
against entity matching test set? Or fusion test set?"*

Tradeoff:

- **EM test set** (`<src1>_2_<src2>_test.csv`) gives per-pair gold
  labels. To get cluster memberships you'd do transitive closure
  on the positive pairs (same logic as
  `build_record_groups_from_correspondences`). Pro: it's *the*
  gold artifact for entity matching. Con: only covers labelled
  pairs, not full clusters.
- **Fusion test set** (`input/fusion/test_set.xml`) gives full
  cluster memberships directly via the `<release>` structure and
  `provenance="..."` attributes. Pro: drop-in fit for BCubed.
  Con: it's the *fusion* gold, designed for value-comparison, not
  for cluster-quality evaluation.

**Recommendation**: use the fusion test set for GR cluster
correctness — it's the cleaner fit and `load_workflow_silver`
already produces the right shape. Run a brief sanity check that
the fusion-test cluster memberships match the EM gold's positive
pairs before committing; if they don't, the question becomes more
serious. Document this decision.

---

## 2. Target panel.json shape

Concrete layout the orchestrator will produce. Compared to the v2.1
shape, **every metric block now sits under a (quality, reference)
key path**.

```json
{
  "usecase": "music",
  "run_id": "...",
  "silver_source": "...",
  "gold_source": "usecases/music/input/fusion/test_set.xml",  // optional
  "headline": {
    "bcubed_f1_SR": 0.95,
    "bcubed_f1_GR": 0.91,                   // optional
    "macro_accuracy_SR": 0.86,
    "macro_accuracy_GR": 0.84,
    "composite_score": 0.88
  },

  "coverage": {
    "entity": {
      "RF": {                                // always computed
        "n_rows_output": 4150,
        "n_rows_largest_input": 4750,
        "row_gain_vs_largest_input": -0.126
      },
      "SR": {                                // present iff silver provided
        "n_recovered": 3480, "n_partial": 800, "n_lost": 0,
        "n_fabricated": 0, "recovery_rate": 0.8131
      },
      "GR": {                                // present iff gold provided
        "n_recovered": 78, "n_partial": 17, "n_lost": 5,
        "n_fabricated": 0, "recovery_rate": 0.78
      }
    },
    "fact": {
      "RF": {
        "density_output": 0.62,             // fraction of non-null cells
        "density_largest_input": 0.51,
        "density_gain": +0.11
      },
      "SR": {
        "per_column_drift_normalized": {...},
        "overall_drift": 0.034,
        "density_delta_per_attribute": {...}
      },
      "GR": {                                // same metrics vs gold
        "per_column_drift_normalized": {...},
        "overall_drift": 0.041,
        "density_delta_per_attribute": {...}
      }
    },
    "source_based": {
      "RF": {
        "winning_source_distribution_per_attribute": {...}  // no comparison; just the histogram
      },
      "SR": {
        "same_source_collision_rate": {...},
        "source_mix_distribution_js": 0.034,
        "per_source_coverage_rate": {...}
      },
      "GR": {                                // same metrics vs gold
        ...
      }
    }
  },

  "consistency": {                           // placeholder for now
    "_placeholder": true,
    "_design_owner": "Aaron",
    "_provisional_signals": {
      // Pulled here from v2.1's column_shape.validity_per_column —
      // exact fit for the consistency dimension. The placeholder
      // status applies to the *broader* consistency design (ontology
      // disjointness etc. on the KG side); the validity rate is
      // already implemented and usable.
      "validity_per_column": {
        "SR": {...},
        "GR": {...}
      }
    }
  },

  "correctness": {
    "cluster": {
      "SR": {
        "bcubed": {...},
        "alignment": {"mean_jaccard": ..., "size_match_rate": ...,
                      "mean_size_delta": ..., "max_size_overshoot": ...}
      },
      "GR": {                                // same metrics vs gold
        "bcubed": {...},
        "alignment": {...}
      }
    },
    "fact": {
      "SR": {
        "per_attribute": {attr: {accuracy, similarity_mean,
                                 semantic_accuracy, mismatch_fingerprint,
                                 mae, medae, ...}},
        "macro_accuracy": 0.86,
        "micro_accuracy": 0.88,
        "conflict_only_accuracy": 0.71,
        "conflict_rate_delta": 0.03,
        "fully_correct_cluster_rate": 0.62,
        "list_attribute_set_metrics": {...}
      },
      "GR": {                                // same vs gold
        ...
      }
    }
  },

  "resource_usage": {                        // unchanged from v2.1, opt-in
    "duration_seconds": ..., "peak_memory_mb": ..., "api_cost": ...
  },

  "task_step": {                             // placeholder
    "_placeholder": true,
    "_design_intent": "Integrate per-stage metrics from PyDI.schemamatching.evaluation, PyDI.entitymatching.evaluation, PyDI.fusion.evaluation. Callers provide these via optional task_step_metrics kwarg; the orchestrator just packs them under the right shape."
  },

  "aggregated": {                            // placeholder
    "_placeholder": true,
    "_design_intent": "Two-dimensional weighting (Quality × Reference). Revisit after seeing first metric results across multiple pipelines."
  },

  "warnings": [...]
}
```

**Block-by-block presence rules:**

- `coverage.*.RF`, `consistency.*.RF`, etc. → always present (no
  reference needed).
- `coverage.*.SR`, `consistency.*.SR`, etc. → present iff a silver
  is passed.
- `coverage.*.GR`, `consistency.*.GR`, etc. → present iff a gold is
  passed.
- `correctness.*` blocks omit the RF entry (the brainstorm marks
  these as "Not possible" without ground truth).
- `resource_usage` → unchanged from v2.1: opt-in via kwargs, omitted
  entirely when no kwargs provided.
- `task_step` → present only when caller supplies per-stage metrics
  via a new `task_step_metrics` kwarg.

---

## 3. Mapping current → new (exhaustive)

Where every v2.1 metric lands in v3. Items marked **NEW** are
metrics the brainstorm calls for that the panel doesn't currently
emit; items marked **GR-NEW** are existing SR metrics that need a
gold-reference twin.

### Coverage

| v3 path | v2.1 path | Status |
|---|---|---|
| `coverage.entity.RF.n_rows_output` | `entity_coverage.n_pipe` | move + rename |
| `coverage.entity.RF.n_rows_largest_input` | — | **NEW** (panel needs sources_pipe sizes; trivial) |
| `coverage.entity.RF.row_gain_vs_largest_input` | — | **NEW** derived |
| `coverage.entity.SR.n_recovered/...` | `entity_coverage.entity_overlap` | move (rename keys to match shape) |
| `coverage.entity.GR.n_recovered/...` | — | **GR-NEW** — same code path, different reference |
| `coverage.fact.RF.density_output` | — | **NEW** (mean non-null rate across attributes) |
| `coverage.fact.RF.density_gain` | — | **NEW** derived |
| `coverage.fact.SR.per_column_drift_normalized` | `column_shape.per_column_drift_normalized` | move |
| `coverage.fact.SR.overall_drift` | — | **NEW** (mean across columns; trivial derivation) |
| `coverage.fact.SR.density_delta_per_attribute` | `value_correctness.density_delta_per_attribute` | move from Tier 4 to Tier-Coverage-SR |
| `coverage.fact.GR.*` | — | **GR-NEW** mirror of SR |
| `coverage.source_based.RF.winning_source_distribution_per_attribute` | — | **NEW** (needs `cell_provenance_pipe`; can also be derived from `sources_pipe` + `pipe_membership` heuristically when provenance absent) |
| `coverage.source_based.SR.same_source_collision_rate/...` | `cluster_correctness.source_composition` | move (and rename to match category) |
| `coverage.source_based.GR.*` | — | **GR-NEW** mirror |

### Consistency

| v3 path | v2.1 path | Status |
|---|---|---|
| `consistency._placeholder` | — | **NEW** (placeholder block) |
| `consistency._provisional_signals.validity_per_column.SR/GR` | `column_shape.validity_per_column` | move (rename + carry across two references) |

### Correctness

| v3 path | v2.1 path | Status |
|---|---|---|
| `correctness.cluster.SR.bcubed` | `cluster_correctness.bcubed` | move |
| `correctness.cluster.SR.alignment.{mean_jaccard, size_match_rate, mean_size_delta, max_size_overshoot}` | `cluster_correctness.alignment` | move |
| `correctness.cluster.GR.*` | — | **GR-NEW** |
| `correctness.fact.SR.per_attribute` | `value_correctness.per_attribute` | move |
| `correctness.fact.SR.macro_accuracy` etc. | `value_correctness.macro_accuracy` etc. | move |
| `correctness.fact.SR.list_attribute_set_metrics` | `value_correctness.list_attribute_set_metrics` | move |
| `correctness.fact.SR.conflict_*` | `value_correctness.conflict_*` | move |
| `correctness.fact.SR.fully_correct_cluster_rate` | `value_correctness.fully_correct_cluster_rate` | move |
| `correctness.fact.GR.*` | — | **GR-NEW** |

### Resources

| v3 path | v2.1 path | Status |
|---|---|---|
| `resource_usage.*` | `resource_usage.*` | unchanged |

### Task/Step + Aggregated

Both placeholders.

### Removed in v3 (already removed or moved elsewhere)

- `entity_coverage.row_count_abs_diff` / `row_count_rel_diff` —
  redundant with `coverage.entity.RF.row_gain_vs_largest_input` and
  the entity-overlap counts. Drop.
- `value_correctness.source_attribution_js_per_attribute` /
  `synthesis_rate_per_attribute` — these are source-attribution
  metrics; in v3 they live under
  `coverage.source_based.{SR,GR}.source_attribution_js` and
  `coverage.source_based.{SR,GR}.synthesis_rate`. Same provenance
  availability gate, same skip warning.

---

## 4. Placeholders

The brainstorm explicitly asks for two placeholders. The plan
treats them as **typed empty blocks** so consumers can detect their
absence without crashing.

### 4.1 `consistency` — first-class with one signal, more pending

Per §9 decision 4: `validity_per_column` is first-class, not
provisional. The broader consistency design (ontology-style
disjointness, cross-attribute constraints) is still pending Aaron's
input — the panel surfaces a text marker for that.

Concrete shape:

```json
"consistency": {
  "SR": {
    "validity_per_column": {
      "release-date": {"validity_rate_silver": 1.0,
                       "validity_rate_pipe": 0.97,
                       "delta": -0.03,
                       "parse_failures_pipe": 0,
                       "constraint_failures_pipe": 128,
                       ...},
      ...
    }
  },
  "GR": {
    "validity_per_column": {... same shape ...}
  },
  "_design_extensions_pending": "Aaron — disjoint-class checks, cross-attribute consistency, etc."
}
```

### 4.2 `task_step`

Status: per-stage metrics already exist in PyDI but aren't in the
panel.

Provisional shape:

```json
"task_step": {
  "_placeholder": true,
  "_design_intent": "Integrate per-stage metrics from PyDI.schemamatching.evaluation, PyDI.entitymatching.evaluation, PyDI.fusion.evaluation. Caller passes them in via optional task_step_metrics kwarg; orchestrator packs them under {schema_matching, blocking, entity_matching, fusion} keys."
}
```

When wired, the caller supplies a dict the orchestrator passes
through verbatim. No new metric computation lives in the panel —
this is just a packing layer.

### 4.3 `aggregated`

Status: "Todo: Think about how to aggregate coverage, consistency
and correctness metrics."

Provisional shape:

```json
"aggregated": {
  "_placeholder": true,
  "_design_intent": "Two-dimensional weighting (Quality × Reference). E.g. composite = w_cov · cov_score + w_cons · cons_score + w_corr · corr_score, where each X_score is itself a weighted average over RF/SR/GR. Revisit after collecting first metric results across multiple pipelines."
}
```

The v2.1 composite score (single number weighted across the four
tiers) **stays in the panel** under `headline.composite_score` so
existing consumers keep working. The new aggregated structure
lives alongside but is a placeholder until the 2D weighting design
is agreed.

---

## 5. API changes

### `compute_e2e_panel` signature changes

```python
def compute_e2e_panel(
    *,
    pipe_fused: pd.DataFrame,
    correspondences_pipe: pd.DataFrame,
    sources_pipe: Sequence[pd.DataFrame],
    silver: Optional[SilverStandard] = None,         # was required; now optional
    gold: Optional[SilverStandard] = None,           # NEW — second reference
    column_types: Mapping[str, str],
    pipe_id_column: str = "_fusion_group_id",
    silver_id_column: str = "cluster_id",
    gold_id_column: str = "cluster_id",              # NEW
    pipe_membership: Optional[pd.DataFrame] = None,
    pipe_source_id_column: Optional[str] = None,
    silver_source_records: Optional[Mapping[str, Mapping[str, Any]]] = None,
    gold_source_records: Optional[Mapping[str, Mapping[str, Any]]] = None,    # NEW
    cell_provenance_pipe: Optional[pd.DataFrame] = None,
    numerical_tolerance: float = 0.04,
    numerical_tolerance_overrides: Optional[Mapping[str, float]] = None,
    column_constraints: Optional[Mapping[str, Mapping[str, Any]]] = None,
    semantic_value_similarity: Optional[Callable[[str, str], float]] = None,
    semantic_value_threshold: float = 0.85,
    pipeline_duration_seconds: Optional[float] = None,
    pipeline_peak_memory_mb: Optional[float] = None,
    pipeline_api_cost: Optional[float] = None,
    pipeline_api_cost_currency: str = "EUR",
    task_step_metrics: Optional[Mapping[str, Any]] = None,    # NEW
    composite_weights: Optional[Mapping[str, float]] = None,
    source_prefix_map: Optional[Mapping[str, str]] = None,
    usecase: str = "",
    run_id: str = "",
    silver_source_label: str = "",
    gold_source_label: str = "",                              # NEW
) -> E2EPanel:
```

Breaking changes:

- `silver` becomes optional (it's still expected for almost every
  use case, but a pipeline can now be evaluated reference-free
  only).
- New `gold` parameter accepting a second `SilverStandard` bundle.
- Output keys change wholesale (see §2).

### `E2EPanel` dataclass

Adds:

- `cluster_alignment_table_gold: Optional[pd.DataFrame]`
- `cluster_attribute_correctness_gold: Optional[pd.DataFrame]`

`write()` emits two CSVs per artifact when gold is supplied:
`cluster_alignment_silver.csv` + `cluster_alignment_gold.csv`, and
same for `cluster_attribute_correctness`. Single-reference runs
write only the silver CSV under its old name to keep CSV consumers
simple.

---

## 6. Implementation phases

Sequencing matters because some phases unlock others.

### Phase A — Structural restructure (no metric changes)

Goal: emit existing v2.1 metrics under the v3 2D shape. No new
metrics, no new references. Just regrouping.

Touchpoints:

- [PyDI/evaluation/panel.py](../PyDI/evaluation/panel.py) — rewrite
  the `panel = {...}` dict construction; reorganise the existing
  per-tier compute blocks into per-(quality, reference) compute
  helpers.
- [PyDI/evaluation/composite.py](../PyDI/evaluation/composite.py) —
  update subscore lookups (the keys they read from change).
- [tests/evaluation_test/test_panel.py](../tests/evaluation_test/test_panel.py)
  — rewrite all assertions against the new shape.

Acceptance: existing 71 tests pass against the new shape; panel
outputs the same *values* but under new key paths.

### Phase B — Add gold-reference (GR) support

Goal: accept a `gold` argument and run every SR-applicable metric
twice.

Touchpoints:

- Add `gold` / `gold_id_column` / `gold_source_records` /
  `gold_source_label` kwargs to `compute_e2e_panel`.
- Refactor compute blocks (`_build_coverage_entity_sr`,
  `_build_correctness_cluster_sr`, etc.) into reference-agnostic
  helpers that take a `SilverStandard` and return the metric block.
  Call each helper twice when gold is provided.
- The expensive shared computations (`cluster_alignment`,
  `_align_clusters`, source-record indexing) are silver-vs-gold
  independent, so the gold path runs cheaper than silver did the
  first time only when the two memberships overlap; otherwise it's
  ~2× the silver runtime.
- Decide the cluster correctness GR question (§1.3) — recommended:
  use fusion test set via `load_workflow_silver`.

Acceptance:

- A new test class `TestPanelGoldReference` covers identity (silver
  ≡ gold) and small-gold (silver = pool, gold = first-100 subset)
  cases.
- Panel runs against the real music silver + workflow gold produce
  reasonable numbers on both columns.

### Phase C — Reference-free (RF) metrics

Goal: add the metrics that need no reference.

Touchpoints:

- New module `PyDI/evaluation/reference_free.py` with:
  - `n_rows_output`, `n_rows_largest_input`, `row_gain_vs_largest_input`
  - `density_output`, `density_largest_input`, `density_gain`
  - `winning_source_distribution_per_attribute` (from `cell_provenance_pipe`
    when available; from `sources_pipe` + `pipe_membership` heuristic
    fallback otherwise)
- Wire into `coverage.entity.RF`, `coverage.fact.RF`,
  `coverage.source_based.RF`.

Acceptance:

- Three new test classes covering each RF metric on toy data.
- Smoke-test on music silver produces sensible numbers.

### Phase D — Placeholders + reorganisation around them

Goal: emit `consistency` and `aggregated` and `task_step` blocks as
placeholders.

Touchpoints:

- Add `_placeholder` block builders.
- Move `validity_per_column` under
  `consistency._provisional_signals` (it's a v2.1 metric that fits
  there).
- Add `task_step_metrics` pass-through kwarg.

Acceptance:

- Panel JSON carries the three blocks with `_placeholder: true` and
  documented `_design_intent`.
- Tests confirm placeholders don't break composite computation.

### Phase E — Per-stage task/step integration (optional, deferred)

Goal: caller can pass per-stage metric dicts (from
`PyDI.{schemamatching,entitymatching,fusion}.evaluation`) and the
panel surfaces them under `task_step`.

Touchpoints:

- `task_step_metrics: Mapping[str, Any]` kwarg → verbatim packing.
- No metric computation; just a structured output position.

This is deferable — placeholder lands in Phase D; actual wiring
when needed.

### Phase F — Aggregation design + implementation (deferred)

Goal: define the 2D weighting and emit an aggregated score.

Open work item; revisit after seeing first metric results across a
handful of real pipelines.

---

## 7. Markdown doc + ancillary updates

After Phase A lands:

- Rewrite [docs/tutorial/e2e_evaluation/e2e_evaluation_metrics.md](../docs/tutorial/e2e_evaluation/e2e_evaluation_metrics.md)
  to the 2D structure. Keep the v2 / v2.1 changelog sections; add
  a v3 section explaining the restructure.
- Refresh the executable notebook
  [e2e_evaluation_metric_deep_dive.ipynb](../docs/tutorial/e2e_evaluation/e2e_evaluation_metric_deep_dive.ipynb)
  — currently it references v2.1 tier names. Easiest path: rewrite
  it from `metrics.md` (the user's brainstorm doc) once the shape
  stabilises.
- Update the cheat-sheet table.

After Phase B:

- Add a "Reading SR vs GR" section explaining the difference
  between the two reference columns and when each one matters.

---

## 8. Testing strategy

- **Phase A**: 71 v2.1 tests get rewritten to v3 paths. Net delta:
  same coverage, different key paths.
- **Phase B**: ~10 new tests for the gold path. Key scenarios:
  - identity (silver ≡ gold, all metrics match)
  - gold-only (no silver; only GR + RF in output)
  - silver-only (no gold; only SR + RF in output)
  - gold-cluster-mismatch (caller-passed `gold_id_column` doesn't
    align with `pipe_membership`)
- **Phase C**: 3 toy-fixture tests per RF metric, plus a smoke-test
  on real music data verifying RF numbers match the underlying
  source frames.
- **Phase D**: 3 tests verifying placeholder blocks are present
  with the right `_placeholder: true` flag and don't crash
  composite.
- **End-to-end smoke**: real music silver + real music fusion gold
  → panel produces non-trivial numbers in all RF/SR/GR columns
  for every quality dimension. Diagnostic warnings still fire
  correctly.

---

## 9. Decisions (locked-in 2026-05-28)

1. **Cluster correctness GR**: use the **fusion test set** via
   `load_workflow_silver`. ✓ decided.
2. **`coverage.source_based.RF` provenance source**: emit-when-
   available. The PyDI fusion engine already writes per-attribute
   provenance into every fused row's `_fusion_metadata` column —
   no opt-in flag exists or is needed. Implementation: new helper
   `build_cell_provenance_from_fused(pipe_fused)` that reshapes
   the existing `_fusion_metadata` into the long-form
   `(cluster_id, attribute, source_ids)` shape the panel already
   consumes. Pipelines that don't use the fusion engine still have
   the metric skipped with a clear panel warning. ✓ decided.
3. **v2.1 composite**: stays as `headline.composite_score`, a
   stable ranking number until the v3 aggregated 2D weighting
   lands. ✓ decided.
4. **`validity_per_column` placement**: surface directly under
   `consistency.SR.validity_per_column` and
   `consistency.GR.validity_per_column` — not under a
   `_provisional_signals` wrapper. Reasoning: it *is* a consistency
   signal by design (format/constraint adherence), it's already
   implemented and useful, and burying it in a provisional subdict
   makes consumers do extra work. When Aaron's broader consistency
   design lands, additional fields just join it under the same
   shape. ✓ decided.
5. **Small-N GR confidence**: trust — emit raw numbers, no
   bootstrap, no minimum-N gate. ✓ decided.
6. **`gold_source_records` parsing**: yes, parse source XMLs to
   build the gold-side source records index. Enables conflict-only
   accuracy on the GR reference. ✓ decided.

### Implication of #2 — `consistency` no longer placeholder-only

With validity_per_column promoted to first-class
`consistency.{SR,GR}` and the broader ontology-style consistency
design still pending, the `consistency` block in v3 carries:

- `consistency.SR.validity_per_column`     (implemented)
- `consistency.GR.validity_per_column`     (implemented)
- `consistency._design_extensions_pending` (text-only marker
  describing what Aaron's broader design will add — disjoint-
  class checks, cross-attribute consistency, etc.)

The `_placeholder: true` flag is therefore not needed for
`consistency` — only for `task_step` and `aggregated`.

---

## 10. Sequencing recommendation

Land in the order A → B → C → D → (E + F when ready).

- A is mostly mechanical — the existing logic is sound, only the
  output keys change. Probably ~1 PR.
- B is the substantive add — gold-reference computation, decisions
  on cluster-correctness gold source. ~1 PR with careful tests.
- C is small — three new RF metrics, mostly trivial.
- D is small — placeholders + reorganisation around them.
- E + F are deferable until the brainstorm settles further.

Phases A through D close out the "implement what the brainstorm
specifies" scope.

---

## 11. Completion notes (2026-05-28)

Phases A–D have landed. Summary of what's in the codebase:

### Code changes

- **[PyDI/evaluation/panel.py](../PyDI/evaluation/panel.py)** —
  rewritten to emit the v3 panel shape. New helpers:
  - `_compute_reference_free(...)` — RF block builder
    (n_rows_output, row_gain_vs_largest_input, density_output,
    density_gain, winning_source_distribution_per_attribute).
  - `_compute_against_reference(reference=..., ...)` — runs every
    reference-dependent metric against a single SilverStandard.
    Called once for silver and (when provided) once for gold.
  - `_winning_source_distribution_per_attribute(...)` — derives
    the RF source-distribution from `cell_provenance_pipe`.
- **[PyDI/evaluation/composite.py](../PyDI/evaluation/composite.py)**
  — composite_score reads from v3 paths. Subscore names
  (`entity_coverage`, `column_shape`, `cluster_correctness`,
  `value_correctness`) preserved for backwards-compatible
  `composite_score.json` consumers.
- **[PyDI/evaluation/cell_provenance.py](../PyDI/evaluation/cell_provenance.py)**
  — new module with `build_cell_provenance_from_fused(pipe_fused)`
  reshaping `_fusion_metadata` into the long-form
  `(cluster_id, attribute, source_ids)` DataFrame.
- **[PyDI/evaluation/__init__.py](../PyDI/evaluation/__init__.py)**
  — module docstring rewritten for v3; exports
  `build_cell_provenance_from_fused`.

### API changes (in `compute_e2e_panel`)

New kwargs:

- `gold: Optional[SilverStandard]` — second reference for GR metrics.
- `gold_id_column: str` — cluster id column in `gold.fused`.
- `gold_source_records` — optional record-id → record-values index
  for GR conflict-only accuracy. Falls back to pipe sources (which
  is correct for fusion-XML golds — they reference pipe records).
- `gold_source_label: str` — informational.
- `task_step_metrics: Optional[Mapping[str, Any]]` — verbatim pack
  into `panel.task_step`.

### Output shape changes

- v2.1 keys `entity_coverage` / `column_shape` /
  `cluster_correctness` / `value_correctness` are **removed** from
  `panel.json`.
- New keys: `coverage`, `consistency`, `correctness`, `task_step`,
  `aggregated` (placeholder), `gold_source` (when gold provided).
- `headline` carries `bcubed_f1_SR` / `bcubed_f1_GR` /
  `macro_accuracy_SR` / `macro_accuracy_GR` / `composite_score`,
  plus legacy `bcubed_f1` (= `bcubed_f1_SR`).
- New CSVs when gold is provided:
  `cluster_alignment_gold.csv`,
  `cluster_attribute_correctness_gold.csv`.

### Tests

- **88 tests pass** in [tests/evaluation_test/](../tests/evaluation_test/),
  up from 83 (5 new gold-reference tests, 4 new RF tests, 1 new
  task_step_metrics test). All v2.1 tests rewritten to v3 paths.

### Decision §9.6 update

The decision to "parse source XMLs for `gold_source_records`"
turned out to be moot: pipe and gold share the same source
datasets (`sources_pipe`), and the existing fallback
`reference_source_records or pipe_source_records` is correct for
both. No new XML parser needed.

### Deferred work

- **Phase F — Aggregated 2D weighting**: `aggregated` block is a
  `_placeholder: true`. The v2.1-style 1D composite stays in
  `headline.composite_score` as the stable ranking metric. Revisit
  when there's first-pipeline data to calibrate against.
- **Broader consistency design** (Aaron) — `consistency` carries
  only `validity_per_column` for now. `_design_extensions_pending`
  marker is in the panel.

---

## 12. v3.1 — Optional references / RF-only mode (2026-05-28)

The original v3 implementation required `silver` as an argument.
Per user request: callers should be able to evaluate a pipeline
without any reference and still get the RF (reference-free)
signals. **Landed 2026-05-28**:

### API change

`compute_e2e_panel` signature: `silver` and `correspondences_pipe`
became `Optional`. The only truly required args are now:

* `pipe_fused` (the pipeline output)
* `sources_pipe` (for RF row-gain / density-largest-input)
* `column_types` (for RF validity + which columns to evaluate)

Everything else is optional. `silver`, `gold`,
`correspondences_pipe`, `pipe_membership`, all the constraint and
semantic / resource kwargs are gated on what's relevant for the
chosen mode.

### Three modes

| Mode | Caller passes | Panel emits |
|---|---|---|
| **RF-only** | no silver, no gold | `coverage.*.RF` + `consistency.RF` only. No `correctness` (no signal without a reference). No `composite_score`. ~10× faster than SR mode (~0.3 s vs ~5 s on music silver) because skips membership reconstruction + type-routed per-column metrics. Writes only `panel.json` + `panel.csv`. |
| **SR** | `silver=…` | Above + `coverage.*.SR` + `consistency.SR` + `correctness.*.SR` + `composite_score`. Writes the v3 SR-side CSV/JSON artifacts. |
| **SR + GR** | `silver=…` + `gold=…` | Above + GR sub-blocks + gold-side CSV artifacts. |

### Implementation details

- `pipe_membership_built` is built only when at least one of
  `silver` or `gold` is provided. In RF-only mode this and the
  type-routed `column_metrics_df` are entirely skipped.
- `_compute_reference_free` now also computes a
  `consistency.RF.validity_per_column` block (pipe-only validity
  rate per column — no delta version, since there's nothing to
  delta against).
- `composite_score` is conditional on `silver_blocks is not None`.
  Without SR signals the recipe has nothing to weight; the panel
  reports `headline.composite_score` only in SR-bearing modes.
- `E2EPanel` fields `schema_diff`, `column_metrics`,
  `cluster_alignment_table`, `cluster_attribute_correctness`, and
  `composite` are now all `Optional[...] = None`. `write()` skips
  artifacts whose value is `None` or empty.
- A new diagnostic warning fires in RF-only mode to suggest
  passing `silver=` / `gold=` for the fuller picture.

### Tests

92/92 evaluation tests pass (4 new RF-only-mode tests in
`TestPanelRFOnlyMode`: RF blocks present, SR/GR absent, composite
omitted, artifacts skipped, warning fires; plus pipe-only
validity check).

### Smoke-test on real music silver (RF-only)

| metric | value |
|---|---:|
| panel runtime | 0.33 s (vs 5.4 s in SR+GR mode) |
| `coverage.entity.RF.n_rows_output` | 4 570 |
| `coverage.entity.RF.row_gain_vs_largest_input` | −0.006 |
| `coverage.fact.RF.density_output` | 0.940 |
| `consistency.RF.validity_per_column['duration'].validity_rate_pipe` | 1.000 |
| `correctness` | `{}` (empty) |
| `headline.composite_score` | not present |
| warnings | `"No reference supplied — only RF metrics are emitted. Pass silver= …"` |
