# Module 6: Generate Real Companies Variants

## Purpose

Run the existing S1 orchestrator (`usecases_synthetic/scripts/generate_variant.py`, built in [plans/module_10_orchestrator.md](../module_10_orchestrator.md)) on the companies domain to produce real `easy`, `medium`, and `hard` variant directories under `usecases/companies-augmented/`. These variants are the inputs M7 validates.

Despite being "just run a command", this module is budgeted as its own session because the S1 orchestrator has never been executed end-to-end with real inputs — all per-module tests in `usecases_synthetic/tests/` stub the knob runners ([test_generate_variant.py:11-16](../../usecases_synthetic/tests/test_generate_variant.py#L11-L16)). M6 is when orchestrator bugs surface, and catching them here — while S1 implementation is still fresh — is explicitly what plan.md Step 7 calls for.

## Spec References

- **Orchestrator:** [../module_10_orchestrator.md](../module_10_orchestrator.md) — CLI contract, variant layout
- **Orchestrator script:** [../../usecases_synthetic/scripts/generate_variant.py](../../usecases_synthetic/scripts/generate_variant.py)
- **Packager script:** [../../usecases_synthetic/scripts/package_variant.py](../../usecases_synthetic/scripts/package_variant.py)
- **PIPELINE.md Phase 2:** [../../usecases_synthetic/PIPELINE.md](../../usecases_synthetic/PIPELINE.md#phase-2--scenario-1-augmented-use-cases-done) — run commands, current status `[done]` at implementation-level
- **Domain config:** [../../usecases_synthetic/config/domains/companies.yaml](../../usecases_synthetic/config/domains/companies.yaml)
- **Plan.md Step 7 rationale:** [../../plan.md](../../plan.md) line 119 — *"Run on the S1 variants first so any knob calibration issues are caught while the S1 code is fresh"*
- **LLM cache policy:** [../../knobs/cross_cutting.md](../../knobs/cross_cutting.md) — strict-cache mode at hard level for K1/K2 to guarantee reproducibility

## Files to Create

**None** in the strict sense. M6 is an *execution* module. What it produces on disk:

```
usecases/companies-augmented/
  easy/    input/{data,schemamatching,entitymatching,fusion}
           output/{provenance,baselines}
           config/difficulty.yaml
  medium/  (same shape)
  hard/    (same shape)
```

Plus, under `usecases_synthetic/output/companies/`:
```
easy/, medium/, hard/       # per-knob intermediate artifacts
monotonicity_report.csv     # cross-level audit from --level all
```

## Procedure

1. **Sanity check the orchestrator on `--level easy` first** so failures are cheap:
   ```bash
   source pydi-dev/bin/activate
   pydi-dev/bin/python usecases_synthetic/scripts/generate_variant.py --domain companies --level easy
   ```
   Expected wall clock: minutes (LLM calls hit the committed cache for K1/K2).

2. **Inspect the easy variant directory** against [module_10_orchestrator.md § Variant Directory Layout](../module_10_orchestrator.md). Check:
   - All four `input/` subdirectories populated
   - `config/difficulty.yaml` present with all active knobs + seeds
   - `output/provenance/` contains per-knob CSVs (at least 8 files for K1, K2, K3, K4, K5, K6, K8, K10)
   - `DataFrame.attrs["dataset_name"]` preserved in the emitted CSVs (verify by loading one source into pandas)

3. **Run `--level medium` then `--level hard`** separately (not `--level all` yet — we want the three variants before the cross-level check in case one fails):
   ```bash
   pydi-dev/bin/python usecases_synthetic/scripts/generate_variant.py --domain companies --level medium
   pydi-dev/bin/python usecases_synthetic/scripts/generate_variant.py --domain companies --level hard
   ```

4. **Run `--level all`** to regenerate all three alongside the monotonicity audit:
   ```bash
   pydi-dev/bin/python usecases_synthetic/scripts/generate_variant.py --domain companies --level all
   ```
   This emits `usecases_synthetic/output/companies/monotonicity_report.csv` per [module_10_orchestrator.md § Cross-Level Monotonicity Checks](../module_10_orchestrator.md#cross-level-monotonicity-checks).

5. **Read the monotonicity report** and note any failing checks. These are *structural* monotonicity (e.g. K3 drop nesting, K8 naming distance) — distinct from the *semantic* committee-based monotonicity M8 will do. Any structural failure here is a bug in the S1 knob implementation that must be fixed before continuing — it goes back to `plans/module_<knob>.md`.

6. **Save the `--level all` console log** to `plans/validation/companies_variant_generation.log` for reference. It's a one-time artifact that documents the actual first real run.

## Expected failure modes and responses

Since the orchestrator has not been exercised end-to-end:

| Failure | Response |
|---|---|
| Orchestrator crashes mid-pipeline (exception) | Debug in-place. File the root cause against the relevant `plans/module_*.md`. Do NOT hack around it in M6 |
| Knob crash on companies-specific edge case | Same as above. Report to user and fix in the originating knob module |
| Structural monotonicity fails (e.g. K3 drop nesting broken) | Investigate via the per-knob `output/provenance/` CSVs. Likely indicates a shared-seed bug in the knob runner. Fix at origin |
| `difficulty.yaml` missing a knob | Verify against orchestrator's knob list. Fix in `generate_variant.py` |
| DataFrame attrs lost somewhere in the pipeline | Trace via provenance — likely a knob forgot to preserve attrs. Fix at origin per CLAUDE.md rule |
| LLM cache miss (hard level) | Strict-cache mode should raise. Unblock by running with `--allow-cache-miss` only if user approves — otherwise investigate why the cache is incomplete |
| Monotonicity report gaps (e.g. K2 corner-case ratio flat) | Report to user. Likely a config issue (difficulty.yaml thresholds not distinct enough) or a bug in the ratio calculation |
| Run succeeds but variants look identical to baseline | Check that knob scripts actually mutated the frames — verify by diffing source files against original. If same, deeper bug in orchestrator state passing |

**Explicit non-goal:** M6 does not implement fixes for S1 knob bugs. It surfaces them, documents them in a triage section of `companies_variant_generation.log`, and blocks until resolved (by the user or by spawning a return to `plans/module_*.md`).

## Acceptance Criteria

1. `usecases/companies-augmented/{easy,medium,hard}/` exist with the layout defined in [module_10_orchestrator.md § Variant Directory Layout](../module_10_orchestrator.md#variant-directory-layout).
2. Each variant's `config/difficulty.yaml` contains all 8 active knobs with explicit parameters and seeds.
3. Each variant's `output/provenance/` contains per-knob CSVs and a consolidated `provenance_all.csv`.
4. `usecases_synthetic/output/companies/monotonicity_report.csv` exists and all 7 structural checks from [module_10_orchestrator.md § Cross-Level Monotonicity Checks](../module_10_orchestrator.md#cross-level-monotonicity-checks) report monotone results. Non-monotone entries are investigated and either fixed or documented as known S1 limitations.
5. `companies_variant_generation.log` captured under `plans/validation/` with stdout + any failure notes.
6. Each variant's sources are loadable by `variant_loader.load_variant("companies", "<level>")` (M0) — smoke check at end of M6.
7. The `variant_loader` smoke check passes for all three levels.

## Dependencies

M0 (variant_loader for the smoke check). Independent of M1/M2/M3/M4 — M6 can run in parallel with committee-runner development.

## Notes

- This module is the "first production run" of the S1 pipeline. Anything surprising is informative. Pay attention to console output, not just exit codes.
- Budget 1 session, but be prepared to hand back to S1 modules if structural bugs surface. That's not scope creep — that's the whole point of plan.md Step 7 insisting S1 variants come first.
- If multiple S1 knob bugs surface, coordinate with the user on whether to batch-fix them or address sequentially. The validation plan stalls until M6 is green.
- Keep the console log narrow — don't commit the generated variants themselves (`usecases/companies-augmented/` should be gitignored per existing conventions; check before first run). Only `baselines/`, `validation/`, and `plans/validation/` are committed artifacts from this plan.
