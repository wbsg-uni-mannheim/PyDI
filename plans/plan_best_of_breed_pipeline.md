# plan_best_of_breed_pipeline.md

A new "best-of-breed" sequential pipeline framework that competes the
existing committee members at every pipeline stage, selects the winner
by validation-set score, threads its output to the next stage, and
finally emits an end-to-end metric panel comparable to the human
baseline notebooks at [usecases/](../usecases/).

**Status:** scaffolding + sweep loader landed; per-stage chained-sweep
harness in progress (2026-05-28). Products is the testbed; the full
4-domain sweep is gated behind an explicit user checkpoint per the
carry-over from [plan_s1_final.md gating
memory](../usecases_synthetic/PIPELINE.md) — finish products end-to-end
+ user review **before** running music / games / companies.

**Critical design correction (2026-05-28):** the existing
`usecases_synthetic/cache/<stage>_tuning/sweep.json` files were
produced by the committee tuning scripts in
[usecases_synthetic/scripts/_tune_*.py](../usecases_synthetic/scripts/);
those sweeps run each stage on **"perfect" baseline inputs** (every
stage tuned independently against gold). For the best-of-breed
pipeline, **each stage's sweep must run on the previous stage's
winner's output** — the chained semantics is what makes "best of
breed" meaningful as a pipeline choice rather than a per-stage
choice.

The pipeline framework therefore:
- Reuses the hyperparameter grids (the `SPECS` dicts in each
  `_tune_*_committee.py` script).
- Writes its own sweep results + model checkpoints under
  [pipelines/<domain>/](../pipelines/) — separate from
  [usecases_synthetic/cache/](../usecases_synthetic/cache/) and
  [usecases_synthetic/cache/ditto_checkpoints/](../usecases_synthetic/cache/ditto_checkpoints/).
- Re-trains learned matchers (Ditto, sc_block, Magellan classifier)
  from scratch with the pipeline-specific stage k-1 winner as input.

Companion plans:
- Metric panel definition: [plan_e2e_metrics.md](plan_e2e_metrics.md)
- Variant-quality program / committee infra source-of-truth:
  [plan_revision.md](plan_revision.md)

---

## 1. Goal & scope

Produce, per use-case domain, a fused dataset that comes from a
deterministic recipe: at each pipeline stage, every committee member
runs on the validation set, the highest-F1 member is selected, its
output flows into the next stage, and its test-set score is recorded
alongside. After fusion, the fully fused output is run through the
end-to-end metric panel (`PyDI/evaluation/panel.compute_e2e_panel`)
against a silver standard. The same panel is also computed against the
fused output of the matching human-baseline notebook in
[usecases/<domain>/](../usecases/) so both pipelines can be compared
on identical metric surfaces.

**Stages competed**, in order, on **products** (original data, not
synthetic variants):

1. **Schema matching (SM)** — pick the SM member with best F1 against
   `sm_mapping_gold.csv`. Translated frames flow forward.
2. **Normalization (Norm)** — pick the norm member with best
   macro-F1 over fusion-protected (entity, attribute) cells.
3. **EM blocking** — pick the blocker with highest reduction-ratio
   among those clearing the val pair-recall floor.
4. **EM matching** — on the winning blocker's candidate set, pick the
   matcher with best val F1 against the per-pair EM val gold.
5. **Post-clustering refinement** — pick the best of
   `{baseline (none), greedy, mbm}` on val F1 (mirrors the human
   notebook's Step 6).
6. **Data fusion** — pick the fusion-committee member with best
   macro-accuracy on the fusion validation set.
7. **Final e2e panel** — run `compute_e2e_panel` on the fused output
   against the products silver. Same panel also run against the
   human-baseline notebook's fused output. Headline + per-tier diff
   reported.

At every stage we report:
- **val F1 / accuracy** per member (the selection surface).
- **test F1 / accuracy** for every member (so the user can see whether
  val selection generalised; we do **not** select on test).
- **winner name + winning scores** explicitly named.

**Non-goal.** Joint optimisation across stages. A locally-optimal
choice at stage *k* may be globally suboptimal; this is a known limit
of greedy chained selection and is what makes the framework cheap.
The plan is to surface this as a deliberate caveat, not to fix it.

**Non-goal.** New member types. The pipeline only competes what the
existing
[usecases_synthetic/config/committees/](../usecases_synthetic/config/committees/)
rosters already define. The committee YAMLs are the single source of
truth for what gets competed.

---

## 1.5 Chained-sweep semantics

For each stage k in the pipeline:

1. **Load upstream state** — the frames / candidates / correspondences
   produced by stage k-1's winner. For k=1 (SM), the upstream state is
   the raw `VariantBundle`.
2. **For each member m in stage k's roster:**
   a. For each HP combo h in member m's grid (from the existing
      `SPECS` dict):
      - Instantiate m with HP h
      - Run m on the upstream state
      - Evaluate against stage k's val gold
      - Cache the (member, HP) → val score row
   b. Pick member m's val-best HP h*; record its predictions +
      test score.
3. **Pick cross-member winner** for stage k: argmax_m val_score(m, h*_m).
4. **Persist stage k state**: write the winner's output to
   `pipelines/<domain>/state/post_<stage>/...` so stage k+1 can
   read it.
5. **Persist sweep artefacts**:
   - `pipelines/<domain>/sweeps/<stage>/sweep.json` — every (member,
     HP) → val + test scores.
   - `pipelines/<domain>/sweeps/<stage>/winners.json` — per-member
     val-best HP + cross-member winner.
   - Stage-specific model artefacts (e.g. retrained Ditto
     checkpoints) under
     `pipelines/<domain>/checkpoints/<stage>/<member>/<hp_hash>/`.
     The winner-HP checkpoint also linked to
     `pipelines/<domain>/checkpoints/<stage>/<member>/winner/`.

**No model reuse from `usecases_synthetic/`.** Learned matchers
(Ditto, sc_block, Magellan classifier) are retrained from scratch
under the chained pipeline state. The existing
`usecases_synthetic/cache/ditto_checkpoints/products/best/` is
**not** read by the best-of-breed pipeline. Same for any sc_block
checkpoints.

**HP grids reused, results not.** The cartesian grids defined in
`_tune_sm_committee.py SPECS`, `_tune_norm_committee.py SPECS`,
`_tune_em_blocking_committee.py` sub-sweep params,
`_tune_em_matching_committee.py` magellan grid, and
`_tune_fusion_committee.py` sub-sweeps are imported and reused
verbatim. Sweep *outputs* (which configs scored highest) go to
`pipelines/<domain>/sweeps/`, never to
`usecases_synthetic/cache/`.

---

## 2. Architecture

### 2.1 Code location

New top-level [pipelines/](../pipelines/) directory at the repo root,
parallel to `usecases/` and `usecases_synthetic/`:

```
pipelines/
    README.md                       # framework overview + cheat sheet
    __init__.py
    lib/
        __init__.py
        pipeline.py                 # BestOfBreedPipeline orchestrator
        bundle.py                   # OriginalDataBundle adapter
        stage_runners.py            # per-stage wrappers (SM/Norm/EM/Fusion)
        report.py                   # per-stage + panel report writers
    configs/
        products.yaml               # per-domain pipeline config
        # music.yaml / games.yaml / companies.yaml later
    scripts/
        run_best_of_breed.py        # main CLI
        compare_to_human_baseline.py  # panel-on-notebook-output
    products/
        # outputs: per-stage selection json, fused csv, e2e panel/, ...
        # gitignored except for a tiny summary.md committed for the record
    tests/
        test_pipeline_smoke.py
        test_stage_chaining.py
        test_products_e2e.py
```

The `lib/` modules are thin orchestration over the existing committee
runners in
[usecases_synthetic/lib/committee_*.py](../usecases_synthetic/lib/) —
**no fork** of committee logic; we import.

### 2.2 Reuse vs new code

**Reuse as-is** (read-only imports):

- `usecases_synthetic.lib.committee_sm.SMCommitteeRunner`
- `usecases_synthetic.lib.committee_norm.NormCommitteeRunner`
- `usecases_synthetic.lib.committee_em.EMBlockingCommitteeRunner`
- `usecases_synthetic.lib.committee_em.EMMatchingCommitteeRunner`
- `usecases_synthetic.lib.committee_fusion.FusionCommitteeRunner`
- `usecases_synthetic.lib.committee.CommitteeResult` / `MemberResult`
- `usecases_synthetic.lib.variant_loader.load_variant(domain,
  level="baseline")` — already targets the original
  `usecases/<domain>/` tree.
- `usecases_synthetic.lib.committee_paths.resolve_committee_path` —
  resolves the per-domain or canonical committee YAML.
- `PyDI.evaluation.panel.compute_e2e_panel` and its
  `load_workflow_silver` / new `load_products_silver` loaders.

**New code** (in `pipelines/lib/`):

- `BestOfBreedPipeline` — orchestrates the 7 stages, owns the chaining
  logic (each winner's predictions flow into the next stage's input).
- `OriginalDataBundle` — thin wrapper around `VariantBundle` that
  exposes the same shape the committee runners expect, while also
  carrying the per-stage *mutable* current state (the chained
  intermediate frames). This is the load-bearing abstraction: the
  baseline committees were designed to score independently against
  gold; here we need to thread one stage's output into the next.
- `StageSelection` dataclass — per-stage record: members ran, val
  scores, test scores, winner name, winning val score, winning test
  score, runtime, artifacts written.
- Report writers — produce
  `<run>/stage_<n>_<stage>_selection.json`,
  `<run>/per_stage_summary.csv`,
  `<run>/fused.csv`, `<run>/e2e_panel/` (the panel's six artefacts),
  `<run>/comparison_to_baseline.md`.

### 2.3 The orchestration loop

Pseudocode for the chained loop:

```python
bundle = load_variant("products", level="baseline")
state = OriginalDataBundle.from_variant(bundle, pipeline_config)

# Stage 1: SM
sm_result = SMCommitteeRunner(yaml).run(state.to_committee_input("sm"))
winner = pick_best_f1(sm_result)            # val surface
state.apply_sm(winner)                        # translate frames now use winner's mapping

# Stage 2: Norm
norm_result = NormCommitteeRunner(yaml).run(state.to_committee_input("norm"))
winner = pick_best_macro_f1(norm_result)
state.apply_norm(winner)

# Stage 3: EM blocking
# Stage 4: EM matching
# ... etc

# Stage 6: Fusion → produces the fully fused DataFrame
# Stage 7: panel
panel = compute_e2e_panel(
    pipe_fused=state.fused,
    correspondences_pipe=state.correspondences,
    sources_pipe=list(state.sources.values()),
    silver=load_products_silver(...),
    column_types=pipeline_config["column_types"],
    ...
)
panel.write(out_dir / "e2e_panel")
```

The catch: the existing committee runners are not "transformers". They
each *score* their own members against gold and return per-member
metrics, but they don't have an `.apply(winner)` method that
re-runs just the winner and exposes its predictions for downstream
consumption. That's because the runners already store
`predictions: Any` per `MemberResult` (`committee.py:53`). The
orchestrator pulls `result.per_member[winner].predictions` and uses
that directly — no rerun needed. The new code is glue that adapts
each stage's `predictions` shape to the next stage's expected input.

### 2.4 Selection surface per stage

| Stage | Selection surface | Test surface |
|---|---|---|
| SM | F1 vs `sm_mapping_gold.csv` (per-source, then mean) | — (no held-out test SM gold; we report val only and flag this) |
| Norm | macro-F1 over fusion-protected cells from `fusion_validation_set.csv` | macro-F1 over cells from `fusion_test_set.csv` |
| EM blocking | reduction-ratio among blockers with val pair-recall ≥ floor (0.97; YAML) | pair-recall + reduction-ratio on test gold |
| EM matching | F1 on per-pair `*_val.csv` from `output/entity_matching_final_ground_truth/per_file_splits/` | F1 on per-pair `*_test.csv` |
| Refinement (none / greedy / mbm) | F1 on val pairs | F1 on test pairs |
| Fusion | macro-accuracy on `fusion_validation_set.csv` | macro-accuracy on `fusion_test_set.csv` |
| e2e panel | — (descriptive only) | — |

"Macro F1" / "macro accuracy" means per-attribute mean, per the
existing committee scoring contracts.

### 2.5 Where the products inputs come from

Per the resolved Q3 (Norm input = data_cleaned_final):

- Source DataFrames come from
  `usecases/products/input/data_cleaned_final/dataset_{1..4}_normalized.json`.
  Same files the notebook loads ([usecases/products/products_workflow_minimal.ipynb](../usecases/products/products_workflow_minimal.ipynb)).
- SM target schema:
  `usecases/products/input/schemamatching/products_target_schema.json`.
- EM val / test splits per pair:
  `usecases/products/output/entity_matching_final_ground_truth/per_file_splits/prod1_to_prod{2,3,4}_{train,val,test}.csv`.
- Fusion val / test:
  `usecases/products/input/fusion/fusion_{validation,test}_set.csv`.

`load_variant` already covers most of this for synthetic variants;
the orchestrator's `OriginalDataBundle.from_variant` may need a small
"products-original" path patch where the file naming differs from the
synthetic-side convention (notably `data_cleaned_final/` rather than
`data/`). To be confirmed during T1; if the patch is non-trivial we
write a thin `load_original_products_bundle` constructor in
`pipelines/lib/bundle.py` instead of touching the synthetic loader.

---

## 3. New prerequisites

Three small artifacts have to land before the products pipeline can
run end-to-end. Each is a one-off; tracked as T2, T3, T4 below.

### 3.1 Products SM gold (T2)

`usecases/products/input/schemamatching/sm_mapping_gold.csv` does not
exist today. Hand-author it: one row per `(source, source_column,
target_column)` triple. Roughly 20–30 rows (4 sources × ~6 columns
canonical mapping). Schema must match the existing
`usecases/companies/input/schemamatching/sm_mapping_gold.csv` shape so
the SM committee runner can consume it without changes. Author via
inspection of the four `dataset_{1..4}_normalized.json` files +
`products_target_schema.json`.

### 3.2 Products silver loader (T3)

New `load_products_silver(usecase_dir, *, split="combined")` in
[PyDI/evaluation/silver_standard.py](../PyDI/evaluation/silver_standard.py).
Reads the CSV pair `fusion_{validation,test}_set.csv`, builds a
`SilverStandard` with:

- `fused`: one row per cluster (id from `id_left` or `cluster_id`;
  prefer `cluster_id` — it's present and stable). Columns are the
  fused-attribute columns the CSV carries (`title`, `brand`,
  `description`, `price`, `priceCurrency`, plus the
  data_cleaned_final additions when present).
- `membership`: long-form `(record_id, source, cluster_id)`. The CSV
  pairs are 2-record clusters by construction (`id_left` /
  `id_right`); the loader expands them into the long-form contract.
- `cell_provenance`: `None`. The CSV doesn't carry per-cell
  provenance, so source-attribution + synthesis-rate metrics will
  be skipped with a panel warning (the existing §6.5 Q8 path).

Test: round-trip the loader on a fixture file, assert the membership
table reflects the CSV pair structure, assert `cell_provenance=None`
falls through without crashing in the panel.

### 3.3 Products committee config sanity-check (T4)

The four products committee YAMLs at
[usecases_synthetic/config/committees/](../usecases_synthetic/config/committees/)
(`*_products.yaml`) were authored for the synthetic-pipeline inputs.
The pipeline runner needs them to also work against original products
data. The two material gaps to verify:

- `em_matching_committee_products.yaml` Ditto checkpoint path
  (`usecases_synthetic/cache/ditto_checkpoints/products/best`)
  exists. If not, document `ditto_plm` member is auto-disabled for
  the original-data run and the committee falls through to the
  remaining 3 matching members (`llm_matcher`, `comem`, `magellan`).
  Skipping a member that lacks a checkpoint is a fine degradation;
  the framework reports it in the per-stage JSON.
- `fusion_committee_products.yaml` evaluation function attribute
  coverage matches whatever schema the products silver CSV actually
  carries. T4 verifies this by loading the CSV + the YAML and
  diffing attribute sets.

T4 is read-only verification; produces a one-line status per
committee under `pipelines/products/prerequisites_check.md`. No
committee YAMLs are edited.

---

## 4. Comparison to the human baseline

End goal of the framework: a single side-by-side report between the
best-of-breed pipeline output and the human notebook's fused output.

The human-baseline fused output is what
[usecases/products/products_workflow_minimal.ipynb](../usecases/products/products_workflow_minimal.ipynb)
produces in its final fusion cell — `fused` DataFrame with 768 rows
(per the cell output) plus the correspondences `all_correspondences`
DataFrame with 2032 pairs.

Two reasonable comparison contracts:

(a) **Re-execute the notebook programmatically** via
    `jupyter nbconvert --execute` to get a fresh fused output to
    compare against. Predictable but slow + requires the LLM API to
    be reachable.

(b) **Cache the notebook's fused output once** and load it for
    comparison. Faster, but the cache can drift if the notebook is
    edited.

The plan picks **(b) with a freshness sanity check** — at
comparison time, hash the notebook file + its last-run timestamp;
if either changed since the cache was written, warn loudly and
prompt the user. Cache lives at
`pipelines/products/baselines/notebook_fused.parquet` +
`notebook_correspondences.parquet` +
`notebook_run_meta.json` (notebook hash + git SHA + run timestamp).

`compare_to_human_baseline.py` then:

1. Loads the cached notebook outputs.
2. Computes the e2e panel against the products silver for **both**
   the cached notebook output and the best-of-breed output.
3. Emits `pipelines/products/comparison.md` with two panel summaries
   side by side + a per-tier delta table + a short interpretation
   guide derived from `plan_e2e_metrics.md` §A.7.

The point is **not** to declare a winner — it's to make the
comparison legible.

---

## 5. Implementation todos

Sequencing is strict: T1 + T2 + T3 are blockers for T5 (the actual
pipeline run); T4 can land in parallel. T6 (panel-on-notebook +
comparison report) needs T5 done. T7 (the multi-domain sweep) is
**gated behind explicit user review** of the products run.

- **T1.** `pipelines/` scaffolding.
  - `pipelines/lib/pipeline.py`: `BestOfBreedPipeline` class +
    `StageSelection` dataclass. Stage chaining loop per §2.3.
  - `pipelines/lib/bundle.py`: `OriginalDataBundle.from_variant` +
    mutable state mutator methods (`apply_sm`, `apply_norm`,
    `apply_blocker`, `apply_matcher`, `apply_refinement`,
    `apply_fusion`).
  - `pipelines/lib/stage_runners.py`: per-stage adapters that take
    the bundle, instantiate the right committee runner, run, return
    a `StageSelection`.
  - `pipelines/lib/report.py`: writers for the per-stage JSON,
    per-stage summary CSV, and the run-level `summary.md`.
  - `pipelines/configs/products.yaml`: pipeline-level config (paths,
    `column_types`, refinement options to compete, pipeline-output
    layout).
  - `pipelines/scripts/run_best_of_breed.py`: CLI entry point
    (`--domain products --out pipelines/products/<run_id>/`).

- **T2.** Hand-author
  `usecases/products/input/schemamatching/sm_mapping_gold.csv` (per
  §3.1). Validate it loads through the same path
  `committee_sm.py` already uses for the other domains. Tracked in
  `pipelines/products/prerequisites_check.md`.

- **T3.** New `load_products_silver` in
  [PyDI/evaluation/silver_standard.py](../PyDI/evaluation/silver_standard.py)
  per §3.2. Test under
  [tests/evaluation_test/](../tests/evaluation_test/) (round-trip
  loader on the existing fusion CSV).

- **T4.** Prerequisites sanity-check (per §3.3). Output:
  `pipelines/products/prerequisites_check.md` with a one-line
  pass/fail/skip per committee member.

- **T5.** First products run, end-to-end. Run
  `pipelines/scripts/run_best_of_breed.py --domain products --out
  pipelines/products/<run_id>/`. Expected artifacts:
  - `stage_1_sm_selection.json`
  - `stage_2_norm_selection.json`
  - `stage_3_em_blocking_selection.json`
  - `stage_4_em_matching_selection.json`
  - `stage_5_refinement_selection.json`
  - `stage_6_fusion_selection.json`
  - `per_stage_summary.csv` (one row per stage: winner, val, test)
  - `fused.csv` + `correspondences.csv`
  - `e2e_panel/` (the 6 panel artifacts from `panel.py`)
  - `summary.md` — winner per stage + headline metrics

  Run **after** T1+T2+T3 land; T4 runs alongside.

- **T6.** Comparison to human baseline (per §4).
  - `pipelines/scripts/compare_to_human_baseline.py`. First run:
    cache the notebook fused output to
    `pipelines/products/baselines/`.
  - Compute e2e panel against the products silver for cached
    notebook output.
  - Emit `pipelines/products/comparison.md` (the two panels side by
    side + per-tier delta).

- **T7.** **GATED.** Extend to music / games / companies. Requires
  user sign-off on the products comparison report. Per-domain steps:
  - YAML at `pipelines/configs/<domain>.yaml`.
  - Re-use existing `load_workflow_silver` for music / games /
    companies (no new loader needed — XML silvers exist).
  - Run + comparison report under
    `pipelines/<domain>/<run_id>/`.

  The gating policy is **explicit**: when T6 finishes for products,
  the runner pauses and notifies the user; we do not start T7 until
  the user confirms. Carries forward the same gate pattern recorded
  in [memory](../usecases_synthetic/PIPELINE.md) for the synthetic
  full-domain sweep.

- **T8.** Tests (incremental, alongside T1–T5).
  - **T8a.** Pipeline smoke: tiny 2-source / 4-record / 2-cluster
    fixture; assert every stage produces a winner and the fused
    output has the expected shape.
  - **T8b.** Stage chaining: assert each stage's output type
    matches the next stage's expected input type (catch the
    `predictions` shape mismatches early).
  - **T8c.** Products E2E: nightly-ish test that runs the full
    products pipeline on a downsampled fixture (one source pair,
    ~50 records) and asserts the per-stage JSON + e2e_panel both
    land.
  - **T8d.** Loader round-trip: see T3.

---

## 6. Per-stage selection algorithm (precise)

For stages where the committee runner returns a
`per_member: dict[str, MemberResult]`, selection is:

```python
def pick_winner(result: CommitteeResult, metric_key: str = "f1") -> str:
    candidates = [
        (name, m.metrics.get(metric_key, float("-inf")))
        for name, m in result.per_member.items()
        if m.metrics.get(metric_key) is not None
    ]
    if not candidates:
        raise PipelineError("No member produced a metric for selection")
    # Sort by metric DESC, then by name ASC for deterministic ties.
    candidates.sort(key=lambda x: (-x[1], x[0]))
    return candidates[0][0]
```

Per-stage `metric_key`:

- SM: `"f1"` (mean over per-source mapping F1)
- Norm: `"macro_f1"` (cross-attribute mean per member)
- EM blocking: handled by `EMBlockingCommitteeRunner`'s
  `composition` block — runner's own winner under `per_pair.winner`,
  not a generic `pick_winner` call.
- EM matching: `"f1"` against per-pair val gold (the `val` split in
  the regen-split contract; see C11 in [plan_revision.md](plan_revision.md)).
- Refinement: `"f1"` against the same val pairs after the refiner runs.
- Fusion: `"macro_accuracy"` against `fusion_validation_set.csv`.

Test-surface scores are computed for **every** member (not just the
winner), so the JSON output lets the user see whether val-best ==
test-best per stage.

---

## 7. Configuration

`pipelines/configs/products.yaml` shape (minimal first pass):

```yaml
domain: products
silver_loader: products_csv          # "products_csv" | "workflow_xml"
silver_dir: usecases/products/input/fusion

sources:
  - products_1
  - products_2
  - products_3
  - products_4

source_pairs:
  - [products_1, products_2]
  - [products_1, products_3]
  - [products_1, products_4]

# Stage-by-stage opt-out. Default: all stages compete.
stages:
  sm: { enabled: true }
  norm: { enabled: true }
  em_blocking: { enabled: true }
  em_matching: { enabled: true }
  refinement: { enabled: true, methods: [baseline, greedy, mbm] }
  fusion: { enabled: true }

# Required for the panel.
column_types:
  id: identifier
  title: text
  brand: categorical
  description: text
  price: numerical
  priceCurrency: categorical
  product_type: categorical
  model: text
  model_number: text
  chipset_name: text
  vram_gb: numerical
  storage_gb: numerical
  # ... per the products_target_schema columns we evaluate against

# Numerical tolerance for the panel.
panel_tolerance:
  default: 0.04
  overrides:
    price: 0.10
    vram_gb: 0.10
    storage_gb: 0.10
```

LLM-backed members are opt-in via the existing committee runners'
`with_llm` flag. The pipeline CLI exposes `--with-llm` and passes it
through to every stage runner.

---

## 8. Open questions & known gaps

1. **No SM test gold for products.** The plan reports SM val F1
   only. Authoring a held-out SM test split would be a small follow-up;
   not blocking.
2. **Locally-optimal ≠ globally-optimal.** Greedy per-stage selection
   may pick an SM that translates well but produces a column layout
   the downstream EM matcher struggles with. Acknowledged as a known
   limitation in §1; surfaced in `summary.md` as a caveat.
3. **Cache staleness for the human-baseline comparison.** T6 includes
   a hash-based sanity check, but the check fires only when the
   notebook file changes — if the notebook is re-executed in place
   without any source edit, the cache can drift silently. We accept
   this risk for v1.
4. **Norm members may no-op on data_cleaned_final.** Starting from
   already-normalised inputs means the Norm committee may have very
   little signal to differentiate members. If the Norm winner's val
   macro-F1 is ≥ 0.99 across all members, we log a warning and report
   "Norm members tied — selection vacuous" in the per-stage JSON.
   This is informational, not a fault.
5. **Ditto checkpoint** for products may need retraining against the
   `data_cleaned_final` field set if T4 reveals a schema drift
   between the synthetic-side checkpoint and the original data. If
   so: skip Ditto for v1 (committee runner already handles a missing
   checkpoint by dropping the member), and add Ditto retraining as
   a follow-up.

---

## 9. Reporting contract (T8 final report)

When T6 lands, produce a one-page summary back to the user covering:

- Per-stage winner table (stage → winner → val score → test score).
- The composite `panel.json` headline + composite_score for the
  best-of-breed pipeline.
- The same headline + composite_score for the cached human-baseline
  output.
- A per-tier delta table (Tier 1–4) with a one-line interpretation
  per tier.
- Any panel warnings that fired (e.g. "source-attribution skipped:
  cell_provenance_pipe not passed").
- The list of stages where Norm / etc. were "vacuous" (per §8.4).
- The list of LLM-backed members that ran (if `--with-llm` was on).

This satisfies the memory note about "Implementation must be
rigorously tested + end with a final report".

---

## 10. Out of scope

- Beam-search / joint-optimisation across stages (see §1, §8).
- New committee members. The pipeline competes only what the existing
  committee rosters define.
- The synthetic variants — this framework runs on **original** data
  only. Running the same framework against the synthetic variants
  (for difficulty-aware selection) is a future workstream.
- A dashboard / UI for browsing the panel — file outputs only.
