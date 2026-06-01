# Module 5: Phase 1 — Baseline Measurement

## Purpose

Run SM/EM/Fusion committees on the *original* companies data (untouched `usecases/companies/`) and persist the per-stage, per-member, per-attribute metrics as the reference point for all subsequent variant validation. This is PIPELINE.md **Phase 1** (currently `[todo]`).

Per [cross_cutting.md § Bootstrap order](../../knobs/cross_cutting.md#bootstrap-order-committees--baseline--calibration): **committee design → baseline measurement → per-knob calibration**. Committee design is M1. Per-knob calibration is M7/M8.

## Spec References

- **Bootstrap order:** [../../knobs/cross_cutting.md § Bootstrap order](../../knobs/cross_cutting.md#bootstrap-order-committees--baseline--calibration)
- **PIPELINE.md Phase 1:** [../../usecases_synthetic/PIPELINE.md](../../usecases_synthetic/PIPELINE.md#phase-1--baseline-measurement-todo) — planned script `scripts/measure_baseline.py [todo]`
- **Committee runners:** [module_02_sm_committee.md](module_02_sm_committee.md), [module_03_em_committee.md](module_03_em_committee.md), [module_04_fusion_committee.md](module_04_fusion_committee.md)
- **Metric storage format:** [module_00_infrastructure.md](module_00_infrastructure.md) § `baseline_loader.py`
- **Baseline path convention:** `usecases_synthetic/baselines/<domain>/baseline_metrics.json`

## Files to Create

### Script

| File | Responsibility |
|---|---|
| `usecases_synthetic/scripts/measure_baseline.py` | CLI: `--domain companies [--with-llm] [--out-dir ...]`. Loads the baseline `VariantBundle` (original usecase dir), runs all three committees, persists `baseline_metrics.json` and `baseline_report.md` under `usecases_synthetic/baselines/<domain>/` |

### Tests

| File | What it tests |
|---|---|
| `test_measure_baseline.py` | Monkey-patches committee runners to return tiny fixture results; verifies `baseline_metrics.json` is written with all three stages, is re-readable by `baseline_loader`, and the markdown report contains per-stage tables |

## CLI

```bash
pydi-dev/bin/python usecases_synthetic/scripts/measure_baseline.py --domain companies
```

Options:
- `--domain <name>` (required) — currently only `companies` is supported
- `--with-llm` — includes LLM committee members (cost: multiple API calls)
- `--out-dir <path>` — overrides default `usecases_synthetic/baselines/<domain>/`
- `--stages sm,em,fusion` — run only a subset (default: all)
- `--fusion-input-member <name>` — override auto-selection of EM member fed into fusion (default: highest-F1 EM member on baseline)

## baseline_metrics.json shape

```json
{
  "domain": "companies",
  "generated_at": "2026-04-11T...",
  "with_llm": false,
  "committee_versions": {
    "sm": "sm_committee.yaml@<sha>",
    "em": "em_committee.yaml@<sha>",
    "fusion": "fusion_committee.yaml@<sha>"
  },
  "sm": {
    "per_member": {
      "label_based_jaro": {"precision": ..., "recall": ..., "f1": ...},
      "instance_based_overlap": {...}
    },
    "aggregated": {"macro_f1": ..., "min_f1": ..., "max_f1": ...},
    "per_attribute": {
      "name": {"label_based_jaro": 1.0, "instance_based_overlap": 0.9},
      ...
    }
  },
  "em": {
    "per_member": {
      "standard+rule": {"f1": 0.78, "precision": ..., "recall": ...,
                         "pool_precision": 0.92, "pool_recall": 0.35},
      "embedding+rule": {...}
    },
    "aggregated": {"macro_f1": ..., ...},
    "per_partition": {
      "forbes_dbpedia": {"standard+rule": 0.78, ...},
      "forbes_fullcontact": {...},
      "dbpedia_fullcontact": null
    },
    "fusion_input_member": "standard+rule"
  },
  "fusion": {
    "per_member": {
      "name|voting": {"accuracy": 0.82},
      "name|longest_string": {"accuracy": 0.75},
      ...
    },
    "aggregated": {"overall_accuracy": 0.658, "overall_spread": 0.21},
    "per_attribute": {
      "name": {"best_accuracy": 0.82, "mean_accuracy": 0.78,
               "spread": 0.07, "strategies": ["voting", "longest_string", "most_frequent"]},
      ...
    }
  }
}
```

`baseline_loader.py` (M0) loads this into a `BaselineMetrics` dataclass that M7/M8 consume.

## baseline_report.md

Human-readable rollup:
- Per-stage tables (one row per committee member, columns = metrics)
- Per-attribute tables (SM: one row per source column; Fusion: one row per attribute with per-strategy numbers)
- Flag members that did not run (e.g. `dbpedia↔fullcontact` missing gold)
- Surface the `fusion_input_member` choice explicitly

No plots. Markdown tables only.

## Acceptance Criteria

1. `pydi-dev/bin/python usecases_synthetic/scripts/measure_baseline.py --domain companies` runs end-to-end on the real companies data without error (may be slow — embedding blockers + multi-member runs).
2. `usecases_synthetic/baselines/companies/baseline_metrics.json` exists, matches the schema above, and is loadable via `baseline_loader.load_baseline("companies")`.
3. `baseline_report.md` renders per-stage tables with all committee members.
4. The fusion `overall_accuracy` on the baseline is consistent with the known companies workflow (≈0.658 for `voting` strategy on `name` per [test_workflow_companies.py:170](../../tests/companies_test/test_workflow_companies.py#L170)) — sanity check that we wired up the same data.
5. The EM `fusion_input_member` choice is the member with the highest macro F1 on baseline and is recorded in the json.
6. `pydi-dev/bin/pytest usecases_synthetic/tests/test_measure_baseline.py -v` passes.

## Dependencies

M0, M1, M2, M3, M4. Cannot run until all three committee runners exist.

## Notes

- This script is run ONCE per domain (re-run only if committee composition changes). Its output is *committed to the repo* so validation runs are reproducible without re-measuring.
- Baseline measurement is not cheap — on companies, embedding blockers over ~3 sources × ~1K-5K entities each takes minutes. Budget one session.
- When re-running after committee edits, the script must produce bit-identical numbers for unchanged members (determinism test). If not, investigate before accepting new baselines.
- The `commitee_versions` field lets us detect stale baselines downstream. M7 refuses to run if the committee YAML hash doesn't match.
