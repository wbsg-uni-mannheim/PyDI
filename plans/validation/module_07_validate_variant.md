# Module 7: Phase 3 — Per-Variant Validation Runner

**Status:** `[done]` — 2026-04-14. `usecases_synthetic/scripts/validate_variant.py` is implemented and `usecases_synthetic/tests/test_validate_variant.py` passes 24/24. EM committee runner extended with `retain_predictions_for` so the fusion stage consumes predictions from the baseline-recorded `fusion_input_member` without re-running the full EM roster.



## Purpose

Run all three committee runners against a packaged variant (produced by M6), compare each stage's metrics to the baseline (from M5), and persist the per-level metrics under `usecases_synthetic/validation/<domain>/<level>/metrics.json`. This is PIPELINE.md **Phase 3** (currently `[todo]`): `scripts/validate_variant.py [todo]`.

M7 is single-level — it processes one `(domain, level)` at a time. Cross-level monotonicity analysis is M8. Ablation is M9.

## Spec References

- **PIPELINE.md Phase 3:** [../../usecases_synthetic/PIPELINE.md](../../usecases_synthetic/PIPELINE.md#phase-3--committee-validation-todo)
- **Committee validation protocol:** [../../knobs/cross_cutting.md § Committee-validated augmentation](../../knobs/cross_cutting.md#committee-validated-augmentation) — steps 1-4 (baseline → apply → re-measure → compare)
- **Pool diagnostic:** [../../knobs/cross_cutting.md § Protection set semantics](../../knobs/cross_cutting.md#protection-set-semantics-not-replacement-gold) — point 3, committee-vs-pool agreement as diagnostic signal
- **Committee runners:** M2, M3, M4
- **Baseline loader:** M0 `baseline_loader.py`
- **M5 baseline metrics:** `usecases_synthetic/baselines/companies/baseline_metrics.json`

## Files to Create

### Script

| File | Responsibility |
|---|---|
| `usecases_synthetic/scripts/validate_variant.py` | CLI: `--domain companies --level {easy,medium,hard} [--with-llm]`. Loads the packaged variant + baseline, runs SM/EM/Fusion committees on the variant, computes deltas, persists metrics + report. Refuses to run if committee YAML hash differs from baseline's `committee_versions` field |

### Tests

| File | What it tests |
|---|---|
| `test_validate_variant.py` | Monkey-patches committee runners to return known deltas; verifies `metrics.json` is written with deltas computed correctly; asserts refusal on committee-version mismatch; verifies the `level_report.md` renders per-stage delta tables |

## CLI

```bash
pydi-dev/bin/python usecases_synthetic/scripts/validate_variant.py --domain companies --level easy
pydi-dev/bin/python usecases_synthetic/scripts/validate_variant.py --domain companies --level medium
pydi-dev/bin/python usecases_synthetic/scripts/validate_variant.py --domain companies --level hard
```

Options:
- `--domain <name>` (required)
- `--level {easy,medium,hard,baseline}` (required). `baseline` re-runs the committees against the original usecase — used for sanity checks
- `--with-llm` — must match the flag used for baseline measurement, else refuse
- `--stages sm,em,fusion` — run subset
- `--fusion-input-member <name>` — override; default taken from baseline metrics json

## Output layout

```
usecases_synthetic/validation/companies/<level>/
  metrics.json               # full per-stage committee result + deltas
  level_report.md            # human-readable, per-stage tables with delta columns
  em_per_pair.csv            # one row per (source_pair, member) with F1 + pool-agree
  fusion_per_attribute.csv   # one row per (attribute, strategy) with accuracy + delta
```

## metrics.json shape

```json
{
  "domain": "companies",
  "level": "hard",
  "generated_at": "2026-04-11T...",
  "baseline_source": "usecases_synthetic/baselines/companies/baseline_metrics.json",
  "with_llm": false,
  "committee_versions": {"sm": "...@sha", "em": "...@sha", "fusion": "...@sha"},
  "sm": {
    "per_member": {"label_based_jaro": {"f1": 0.42, "f1_baseline": 0.95, "f1_delta": -0.53}, ...},
    "aggregated": {"macro_f1": 0.58, "macro_f1_baseline": 0.93, "macro_f1_delta": -0.35},
    "per_attribute": {...}
  },
  "em": {
    "per_member": {"standard+rule": {"f1": 0.71, "f1_baseline": 0.78, "f1_delta": -0.07,
                                      "pool_precision": 0.85, "pool_recall": 0.32,
                                      "pool_precision_baseline": 0.92, ...}},
    "aggregated": {...},
    "per_partition": {"forbes_dbpedia": {...}, ...},
    "fusion_input_member": "standard+rule"
  },
  "fusion": {
    "per_member": {"name|voting": {"accuracy": 0.62, "accuracy_baseline": 0.82, "accuracy_delta": -0.20}, ...},
    "aggregated": {"overall_accuracy": 0.55, "overall_accuracy_delta": -0.10},
    "per_attribute": {...}
  }
}
```

Every measured number gets a `_baseline` and `_delta` twin via `metrics.delta()` from M0.

## Behavior

1. **Load baseline.** Refuse if `committee_versions` in the baseline don't match the current on-disk YAMLs (hashed at load time). This prevents silent drift.
2. **Load variant.** Via `variant_loader.load_variant(domain, level)`. Validate the bundle (all three committees' required inputs present).
3. **Run committees.** SM, then EM, then Fusion (Fusion needs EM correspondences from the `fusion_input_member` recorded in baseline). Pass `correspondences` from the matching member in the EM result, NOT re-run the whole roster.
4. **Compute deltas.** For every leaf metric in the committee result, compute the twin `_baseline` and `_delta`. Keep the original baseline value visible in the json so reports don't have to re-load the baseline file.
5. **Write metrics.json** and the three CSVs (`em_per_pair`, `fusion_per_attribute`) plus `level_report.md`.
6. **Do NOT decide whether the variant is "good" or "collapsed"** — that's M8. M7 is measurement only.

## `level_report.md`

Markdown tables:
- Stage summary table: one row per stage, columns = macro metric + baseline + delta
- SM per-member table: member × (F1, baseline, delta)
- EM per-member table: member × (macro F1, baseline, delta, pool_precision, pool_recall)
- EM per-pair table: pair × member × F1
- Fusion per-attribute table: attribute × (best_accuracy, baseline, delta, spread, baseline_spread, spread_delta)

## Acceptance Criteria

1. `validate_variant.py --domain companies --level easy` runs end-to-end against the M6-produced variant and writes a complete `metrics.json` + `level_report.md`.
2. All baseline numbers in `metrics.json` match the `baseline_metrics.json` values exactly.
3. Running on `--level baseline` produces zero deltas (sanity check: committee on baseline vs stored baseline → identical → deltas are all 0).
4. Committee-version mismatch is detected and the script exits non-zero with a clear error.
5. `em.fusion_input_member` matches the baseline's choice (not re-selected per variant).
6. `pool_precision` and `pool_recall` diagnostics present on every EM member.
7. `pydi-dev/bin/pytest usecases_synthetic/tests/test_validate_variant.py -v` passes.

## Dependencies

M0, M2, M3, M4, M5 (needs baseline metrics), M6 (needs packaged variants). The longest dependency chain in the plan — gate sequentially.

## Notes

- M7 is deliberately dumb: it measures and persists. All judgments about whether the numbers are "good" or "bad" or "monotone" live in M8 to keep measurement and interpretation decoupled.
- The pool-agreement diagnostic is stored with every EM member's metrics but is never used to adjust the reported F1. It's only read by M8 when disambiguating collapses.
- The `--stages` flag allows re-running just one stage after a baseline update. Useful during iteration.
- Committee-version pinning via YAML hash is belt-and-braces: it guards against someone editing `em_committee.yaml` without re-running M5.
