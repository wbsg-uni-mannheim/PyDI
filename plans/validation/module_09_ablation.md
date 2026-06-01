# Module 9: Per-Knob Ablation Validation

## Purpose

Generate "each knob alone at hard" variants — one variant per knob where all other knobs are identity (easy or off) and only the target knob is set to `hard` — and run the committee validation against each. This isolates each knob's contribution to the overall difficulty signal and confirms independent togglability as required by [knobs/ablations.md](../../knobs/ablations.md).

Ablation is what turns "the hard variant is hard" into "the hard variant is hard *because of knob X specifically*". Without it, interactions between knobs can mask or fake individual knob signals.

## Spec References

- **Ablation spec:** [../../knobs/ablations.md](../../knobs/ablations.md) — independent togglability is a hard requirement
- **Ablation granularity:** one variant per knob, set to the knob's hard level, with all other knobs at their identity / easy setting (per-knob card's "easy = identity" rule where applicable)
- **S1 orchestrator:** [../../usecases_synthetic/scripts/generate_variant.py](../../usecases_synthetic/scripts/generate_variant.py) — must support `--only-knob <id>` or equivalent. If it doesn't, this capability is part of M9
- **Committee runners + validator:** M2-M7

## Files to Create

### Possible orchestrator patch

If `generate_variant.py` does not already support per-knob ablation variants (check before starting M9), add:

| File | Change |
|---|---|
| `usecases_synthetic/scripts/generate_variant.py` | Add `--only-knob <id>` flag: sets the named knob to `hard` level, all other knobs to `identity` (or their minimum). Emits the variant under `usecases/<domain>-augmented/ablation_knob_<id>/` |

Before editing, check whether the existing `--level` plumbing admits a composite "ablation_knob_XX" level. If the orchestrator already has a per-knob override, reuse it.

### Scripts

| File | Responsibility |
|---|---|
| `usecases_synthetic/scripts/run_ablation_validation.py` | CLI: `--domain companies [--knobs 1,2,3,...]`. For each requested knob: (1) generates the ablation variant via `generate_variant.py --only-knob`; (2) runs `validate_variant.py` against it; (3) writes metrics under `usecases_synthetic/validation/<domain>/ablation/knob_<id>/metrics.json` |
| `usecases_synthetic/scripts/analyze_ablation.py` | Consumes ablation metrics + baseline; produces `ablation_report.md` with per-knob effect sizes and interaction flags |

### Tests

| File | What it tests |
|---|---|
| `test_ablation.py` | Smoke test that ablation_knob_08 variant produces an SM signal (K8 is SM-only) and near-zero EM/Fusion deltas; ablation_knob_10 variant produces a Fusion signal and near-zero SM/EM deltas. This is testable on the real companies data if M6 is green, else on fixtures |

## Scope

For companies, M9 produces:

- 8 ablation variants: `knob_01`, `knob_02`, `knob_03`, `knob_04`, `knob_05`, `knob_06`, `knob_08`, `knob_10` (K7 deferred, K9 S2-only).
- 8 corresponding `metrics.json` files.
- One aggregated `ablation_report.md`.

Budget-wise, each ablation variant involves: one orchestrator run (minutes) + one `validate_variant.py` run (minutes for SM/Fusion, ~tens of minutes for EM with embeddings). 8 × ~20-40 min is the realistic cost. Plan for this — it's the most expensive single module in the validation plan.

Pool and LLM caches should be hot by this point (M6 populated K1/K2 caches, M7 populated embedding caches). Re-use is critical.

## What the ablation report reveals

For each knob K, we expect per the knob card:

```
ablation_K_at_hard:
  primary_stage: <signal>  # matches knob card's primary target
  other_stages: <~flat>    # other stages should show near-zero delta
```

Violations of this expectation are interaction flags:
1. **Cross-stage leakage** — e.g. if K8 (SM-only per its card) shows a big Fusion delta in ablation, something is coupling SM→Fusion more than expected. Usually a bug in how the variant is packaged (e.g. renamed headers breaking fusion comparators).
2. **Primary-stage under-signal** — e.g. K8 ablation produces only a small SM delta compared to the full-hard variant's SM delta. Means K8 is *dominated* in the full-hard variant by another knob also hitting SM. Usually fine, but worth logging.
3. **Primary-stage over-signal** — ablation delta exceeds full-hard delta. Means knobs are *cancelling each other* in the full variant. Worth investigating — may indicate a scheduling bug in the canonical knob order.

The report surfaces all three categories. M10 triages.

## Acceptance Criteria

1. `generate_variant.py --only-knob <id>` works for all 8 active knobs (add the flag if missing).
2. `run_ablation_validation.py --domain companies` generates all 8 ablation variants and runs validation against each. Caches are reused where possible.
3. `usecases_synthetic/validation/companies/ablation/knob_<id>/metrics.json` exists for each of the 8 knobs.
4. `ablation_report.md` aggregates per-knob deltas with (primary_stage, delta, direction_match, interaction_flag) columns.
5. Per-knob sanity: K8 ablation → SM delta dominant; K10 ablation → Fusion delta dominant; K1/K3/K6 ablations → EM delta dominant; K5 ablation → EM + Fusion deltas both present; K4 ablation → Fusion delta dominant; K2 ablation → EM delta dominant.
6. `pydi-dev/bin/pytest usecases_synthetic/tests/test_ablation.py -v` passes.

## Dependencies

M0, M2, M3, M4, M5 (baseline), M6 (orchestrator working end-to-end), M7 (validator). M8 is NOT a dependency — ablation is orthogonal to cross-level monotonicity and they can in principle run in parallel after M7.

## Notes

- This is the most expensive module. Consider running it in chunks — `--knobs 8` first (fastest, SM-only), then `--knobs 10,4` (Fusion-focused), then the EM-heavy ones (`--knobs 1,2,3,5,6`).
- If `generate_variant.py` lacks `--only-knob` support, adding it is structurally simple but touches the orchestrator; coordinate with the user before editing, since it's a change to code that was just declared done in S1.
- The ablation variants are *disposable* — they're generated once, validated, and then their directories can be removed if disk is tight. Only the `metrics.json` files under `validation/ablation/knob_<id>/` need to persist.
- If an ablation variant reveals a cross-stage leak that points to a bug in one of the S1 knob modules, file it as a finding for M10. Do not fix in M9.
- Determinism across ablation runs is still required — the same seed used in a `--only-knob` run should produce the same variant on every invocation.
