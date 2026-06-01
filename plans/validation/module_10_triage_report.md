# Module 10: Triage + Final Report

## Purpose

Consume all the artifacts produced by M5-M9 and write the single consolidated `final_report.md` that answers plan.md Step 7: **do the S1 knob settings produce measurable, monotone performance differences in PyDI's pipeline?** For each of the four goals stated in [plan_s1_validation.md § Goal](plan_s1_validation.md#goal-precise), report evidence (pass / qualified-pass / fail) with links to the backing metrics.

M10 also produces a prioritized **triage list** of fix items: what needs to change (in knob configs, knob implementations, committee composition, or this plan itself) before proceeding to plan.md Step 8 (Scenario 2 prototype). This is the bridge document from Step 7 to Step 8.

## Spec References

- **Plan.md Step 7:** [../../plan.md](../../plan.md) line 119
- **Fix-strategy defaults:** [../../knobs/cross_cutting.md § Per-knob fix-strategy defaults](../../knobs/cross_cutting.md#per-knob-fix-strategy-defaults)
- **Validation methodology:** [../../plan.md § Validation Methodology](../../plan.md#validation-methodology) — distribution comparison, downstream pipeline performance, human spot-checks
- **Previous phases:** M5 (baseline), M6 (variant generation log), M7 (per-level metrics), M8 (monotonicity report), M9 (ablation report)

## Files to Create

### Report (`usecases_synthetic/validation/companies/`)

| File | Contents |
|---|---|
| `final_report.md` | Single consolidated report. Sections below |

### Optional companion

| File | Contents |
|---|---|
| `triage_list.md` | Machine-readable-ish triage items with severity, owner (implementer / user), target module, and a one-line repro |

### No test file

M10 is a reporting module; tests are not meaningful. The correctness check is peer review of `final_report.md` by the user.

## `final_report.md` structure

```markdown
# S1 Difficulty Validation — Final Report (companies)

## Verdict

One of: **pass** / **qualified pass** / **fail**. One-paragraph justification.

## Goals (from plan_s1_validation.md)

| Goal | Status | Evidence |
|---|---|---|
| 1. Baseline exists | pass | baselines/companies/baseline_metrics.json |
| 2. Signals are real | ... | monotonicity_report.md § Signal table |
| 3. No silent collapses | ... | monotonicity_report.md § Collapse table |
| 4. Per-knob attribution | ... | ablation_report.md |

## Per-knob summary

One subsection per knob (K1-K10 except K7, K9). Each subsection:

### Knob N (<name>)

- **Card prediction:** quote the *Committee expectations* section in one line
- **Full-variant signal:** easy/medium/hard delta on primary stage, with link to monotonicity_report.csv
- **Ablation signal:** delta when run alone, with link to ablation/knob_N/metrics.json
- **Interaction flags:** from M9 — cross-stage leakage, under-signal, over-signal
- **Status:** pass / qualified / fail
- **Follow-up:** one of [none / adjust difficulty.yaml / fix knob code / adjust committee / revise card prediction]

## Collapses found

Table from M8's collapse_report.csv with triage classification and suggested action per cross_cutting.md's fix-strategy defaults.

## Known gaps and deferrals

- Movies / products validation deferred (no variants generated, no pool)
- Games / music validation deferred to Step 9 (domain YAMLs not populated)
- K7 value ambiguity not validated (not built in v1)
- K9 schema completeness not validated (S2 only)
- LLM-member committee runs not exercised unless --with-llm was used in baseline + all per-level runs

## Triage list (ordered by priority)

1. [S1 knob fix] ... → fix in plans/module_<knob>.md
2. [Difficulty config] ... → adjust usecases_synthetic/config/knob_<id>/companies.yaml
3. [Committee composition] ... → adjust usecases_synthetic/config/committees/*.yaml and re-run M5
4. [Plan.md Step 8 prerequisite] ... → resolve before starting Scenario 2 prototype

## Recommendation on plan.md Step 8

- **GO** / **HOLD** / **PARTIAL GO** with a one-paragraph rationale.
- If HOLD: explicit list of fix items that must clear before Step 8 starts.
- If PARTIAL GO: which aspects of Step 8 can proceed (e.g. Scenario 2 infrastructure) vs which should wait (e.g. K-specific tuning).
```

## Triage classification

Every finding gets one of four labels. M10 is responsible for applying them consistently:

| Label | Meaning | Action routing |
|---|---|---|
| **P0 — blocker** | Collapse that breaks the benchmark; wrong direction signal; cross-stage leakage | Fix before Step 8 |
| **P1 — important** | Correct direction but out-of-range magnitude; ablation mismatch with card prediction | Fix before Step 8 unless user opts to document as known limitation |
| **P2 — nice-to-have** | Qualitative-only signal works but card prediction was vague; committee member variance high | Defer to Step 9 (Scale) or later |
| **P3 — noted** | Known deferrals (movies, products, K7, K9, games/music) | No action; track in PIPELINE.md |

## Acceptance Criteria

1. `usecases_synthetic/validation/companies/final_report.md` exists.
2. All four goals from `plan_s1_validation.md` have status + evidence links.
3. Each active knob (8 total) has a per-knob subsection with card prediction + full-variant signal + ablation signal + status.
4. The triage list is non-empty (even on a clean pass — "no P0/P1 items, P3 deferrals documented" is a valid entry).
5. The recommendation on plan.md Step 8 is explicit: GO / HOLD / PARTIAL GO with rationale.
6. `PIPELINE.md` is updated to mark Phase 1 and Phase 3 `[done]` with links to the baseline and validation directories.
7. `plan.md` Next Steps checklist is updated: Step 7 is marked `[done]` with a backlink to `final_report.md`.

## Dependencies

M5, M6, M7, M8, M9. M10 is strictly last.

## Notes

- M10 is the one place in this plan where Claude synthesizes across all artifacts. Keep the final report opinionated — a laundry list of numbers isn't useful to the user. The goal is a clear recommendation on whether to proceed to Step 8.
- If any module's artifacts are missing (e.g. M9 ablation skipped due to time), the final report must call this out in the "Known gaps and deferrals" section. Do not silently omit.
- The triage list is the main deliverable for downstream work. Prioritize ruthlessly — if everything is P0, nothing is P0.
- Update memory files (`.claude/projects/.../memory/`) if this validation surfaces durable feedback or project-level facts that belong there. This is the only module where memory writes are expected as part of the plan.
