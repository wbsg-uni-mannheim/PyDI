# pipelines/products — STATUS (2026-05-28, end-of-day v2)

Best-of-breed pipeline framework + products run status.

---

## Landed

### Framework (all under [pipelines/](.))

- **Orchestrator** ([../lib/pipeline.py](../lib/pipeline.py)) with `--mode {sweep,replay}` dispatch
  and per-stage sweep-result threading. Filters are **rewrite-checkpoint-path**, no
  longer silent-disable (per the 2026-05-28 "no silent dropping" directive):
  - `--ditto-checkpoint-override <path>` rewrites `ditto_plm.checkpoint_path` to the
    pipeline-isolated location. When omitted and the YAML default sits under
    `usecases_synthetic/cache/`, the member is LOUDLY disabled with retrain
    instructions in the error log.
  - `--sc-block-checkpoint-override <path>` — same for `sc_block`.
  - `--no-llm` disables LLM members across SM/EM/Fusion. Default is to include
    them (no silent drop of YAML-enabled members).
  - `--fusion-members <allowlist>` restricts the C12 roster to a subset (debug aid).
- **All four chained-sweep harnesses implemented** in
  [../lib/sweep_harness.py](../lib/sweep_harness.py) — no more stubs:
  - `sweep_sm`: iterates `_tune_sm_committee.SPECS`.
  - `sweep_norm`: iterates `_tune_norm_committee.SPECS` (text_clean,
    date_iso, number_locale, country_iso, taxonomy_lookup, llm_canonicalize) +
    scores against fusion-protected cells via the existing `_score_member`.
  - `sweep_em_blocking`: calls `_run_embedding_sweep`, `_run_standard_sweep`,
    `_run_sn_sweep` from `_tune_em_blocking_committee` + a pipeline-isolated
    sc_block sub-sweep that takes `sc_block_checkpoint_override`.
  - `sweep_em_matching`: calls `_run_magellan_sweep` for Magellan's 12-cell
    classifier grid; ditto_plm picks up the pipeline-isolated checkpoint via the
    matching-YAML rewrite.
  - `sweep_refinement`: 3-method sweep.
  - `sweep_fusion`: iterates `_tune_fusion_committee` sub-sweeps (trust,
    tolerance, trim, list_threshold, truthfinder, accusim, casefusion,
    fusionquery, ltm, llm_judge) via `_score_run` end-to-end.
- **Bundle + state container**, stage runners, report writers, notebook
  baseline extractor, comparison harness — unchanged from earlier.
- **21/21 tests passing** including the rewritten orchestration tests
  (`test_orchestration.py`) and chaining tests.

### Pipeline-isolated checkpoints

- **sc_block** retrained 2026-05-28 23:22 →
  `pipelines/products/checkpoints/em_blocking/sc_block/best/`
  (val_pair_recall = 1.0, 8 epochs, 161s wall).
- **Ditto** retraining in progress 2026-05-28 23:26 →
  `pipelines/products/checkpoints/em_matching/ditto/runs/run_20260528_232623/`
  (currently epoch 7/50, early_stopping_patience=10).
- Neither checkpoint reuses `usecases_synthetic/cache/`.

### Stage winners observed earlier (run_v5 partial; killed at fusion C12 hang)

| Stage | Winner | Val | Test | Note |
|---|---|---|---|---|
| SM | `duplicate_majority` | 1.0000 | 1.0000 | trivial; products shares schema |
| Norm | `passthrough` | 0.4788 | 0.4788 | vacuous tie flagged |
| EM blocking | per-pair via composition | — | — | sc_block was disabled earlier; now competes via override |
| EM matching | `magellan` | 0.6867 | 0.7924 | ditto+llm_matcher+comem were dropped earlier; now compete via override + with_llm=True |
| Refinement | `baseline` | 0.7343 | 0.7924 | 2153 correspondences |
| Fusion | — | — | — | C12 hung at member 2 |

These numbers will change in the next run because the previous run silently dropped
sc_block, ditto_plm, llm_matcher, comem, matchgpt, llm_only. The next run includes
all of them (per "no silent dropping").

---

## Open: C12 fusion hang

C12 fusion runner hangs reproducibly at the **second** member's
`DataFusionEngine.run` call, at 0% CPU with two active TCP connections (Cloudflare /
AWS). First member (`pydi_per_attribute_optimal`) completes in 0.34s.

py-spy is installed (`uv pip install py-spy` → 0.4.2). Next step is to launch the
products pipeline once Ditto retraining completes, wait for fusion to hang, and run
`py-spy dump --pid <pid>` to get the blocked call stack.

---

## How to resume

```bash
# 1. Wait for Ditto retrain to finish (epoch 7/50; ETA ~45 min):
tail -f pipelines/products/checkpoints/em_matching/ditto_train.log

# 2. Full-committee run with both pipeline-isolated checkpoints:
source pydi-dev/bin/activate
python pipelines/scripts/run_best_of_breed.py \
    --config pipelines/configs/products.yaml \
    --out pipelines/products/run_v7/ \
    --mode replay \
    --ditto-checkpoint-override pipelines/products/checkpoints/em_matching/ditto/runs/LATEST_RUN/best \
    --sc-block-checkpoint-override pipelines/products/checkpoints/em_blocking/sc_block/best

# 3. If fusion hangs, py-spy the stuck process:
ps aux | grep run_best_of_breed
py-spy dump --pid <PID>

# 4. Once fusion completes and a panel lands:
python pipelines/scripts/extract_notebook_baseline.py \
    --notebook usecases/products/products_workflow_minimal.ipynb \
    --cache-dir pipelines/products/baselines
python pipelines/scripts/compare_to_human_baseline.py \
    --domain products --pipeline-run pipelines/products/run_v7/
python pipelines/scripts/write_final_report.py \
    --run-dir pipelines/products/run_v7/ --domain products

# 5. Once replay works, opt into sweep mode (much longer; iterates HP grids):
python pipelines/scripts/run_best_of_breed.py \
    --config pipelines/configs/products.yaml \
    --out pipelines/products/run_sweep_v1/ \
    --mode sweep \
    --ditto-checkpoint-override pipelines/products/checkpoints/em_matching/ditto/runs/LATEST_RUN/best \
    --sc-block-checkpoint-override pipelines/products/checkpoints/em_blocking/sc_block/best
```

## Caveats

- The pipeline competes everything the committee YAMLs declare. With LLM enabled
  and the full C12 fusion roster, the LLM cost is non-trivial (~$10-50/run for
  products on gpt-5.4-mini). Use `--no-llm` to drop only the LLM members
  *explicitly* (not silently).
- The C12 hang must be diagnosed before fusion sweeps will complete in finite
  time. Each fusion sub-sweep cell invokes `_score_run` which constructs a fresh
  `FusionCommitteeRunner` and calls it end-to-end.
- The Norm sweep includes `llm_canonicalize` only when `--no-llm` is not passed.
  Same for fusion's `llm_judge`.
