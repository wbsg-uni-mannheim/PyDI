# pipelines/

Best-of-breed sequential data-integration pipeline framework.

At each pipeline stage (SM → Norm → EM blocking → EM matching →
Refinement → Fusion) every committee member is **swept over its
hyperparameter grid against the upstream stage's winner output**, the
val-best HP is locked per member, and the cross-member val-best wins
the stage. The winner's output flows to the next stage; the loser
outputs are discarded.

This is **separate** from the synthetic-pipeline committee tuning
under [usecases_synthetic/](../usecases_synthetic/). That tuning runs
on per-stage "perfect" inputs (each stage independently against its
own gold). The best-of-breed framework reuses the committee
hyperparameter grids but writes its own sweep results + model
checkpoints under `pipelines/<domain>/`, and re-trains learned
matchers (Ditto, Magellan classifier) from scratch on the
chained-pipeline state.

See [plans/plan_best_of_breed_pipeline.md](../plans/plan_best_of_breed_pipeline.md)
for the design rationale and [plans/plan_e2e_metrics.md](../plans/plan_e2e_metrics.md)
for the metric panel definition.

## Layout

```
pipelines/
├── README.md                  # this file
├── configs/
│   ├── products.yaml          # per-domain pipeline config
│   └── ...
├── lib/
│   ├── pipeline.py            # BestOfBreedPipeline orchestrator
│   ├── bundle.py              # PipelineState + bundle loader
│   ├── stage_runners.py       # per-stage runners (YAML-default mode)
│   ├── sweep_harness.py       # per-stage chained-sweep runners
│   ├── sweep.py               # legacy: read usecases_synthetic caches
│   └── report.py              # artifact writers
├── scripts/
│   ├── run_best_of_breed.py            # main entrypoint
│   └── compare_to_human_baseline.py    # panel vs notebook side-by-side
├── tests/
│   └── ...
└── <domain>/                  # outputs (gitignored except summaries)
    ├── sweeps/
    │   ├── sm/
    │   │   ├── sweep.json     # every (member, HP) → val + test
    │   │   └── winners.json   # per-member val-best HP + winner
    │   ├── norm/
    │   ├── em_blocking/
    │   ├── em_matching/
    │   ├── refinement/
    │   └── fusion/
    ├── checkpoints/
    │   ├── em_matching/
    │   │   └── ditto/
    │   │       ├── <hp_hash>/   # per-HP-combo Ditto checkpoint
    │   │       └── winner/      # symlinks/copies of val-best checkpoint
    │   └── ...
    ├── state/
    │   ├── post_sm/sources/        # post-translation source frames
    │   ├── post_norm/sources/      # post-normalization frames
    │   ├── post_em_blocking/candidates_<pair>.csv
    │   ├── post_em_matching/predictions_<pair>.csv
    │   ├── post_refinement/correspondences.csv
    │   └── post_fusion/fused.csv
    ├── e2e_panel/                  # final metric panel artifacts
    ├── per_stage_summary.csv
    ├── comparison.md               # vs human-baseline notebook
    └── summary.md
```

## Running

```
# Full chained sweep + pipeline run (long; one-time per domain)
python pipelines/scripts/run_best_of_breed.py \
    --config pipelines/configs/products.yaml \
    --mode sweep \
    --out pipelines/products/

# Replay using cached sweep winners (fast, deterministic)
python pipelines/scripts/run_best_of_breed.py \
    --config pipelines/configs/products.yaml \
    --mode replay \
    --out pipelines/products/replay_<id>/
```

## Conventions

- **No model reuse** from `usecases_synthetic/cache/`. Ditto / sc_block
  / Magellan classifier are trained from scratch with the
  pipeline-specific stage k-1 winner output, checkpointed under
  `pipelines/<domain>/checkpoints/`.
- **HP grids reused, results not.** Sweep grids come from the
  existing `_tune_<stage>_committee.py SPECS` dicts (imported
  directly). Sweep outputs go to `pipelines/<domain>/sweeps/`.
- **Outputs gitignored except summaries.** Each `pipelines/<domain>/`
  has `summary.md` + `per_stage_summary.csv` + `comparison.md`
  committed; the raw sweep + state + checkpoint trees are large and
  per-run, not checked in.
