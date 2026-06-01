# Module 8: Cross-Level Monotonicity + Collapse Detection

**Status:** `[done]` — 2026-04-14. `usecases_synthetic/lib/monotonicity.py`, `usecases_synthetic/scripts/analyze_monotonicity.py`, and `usecases_synthetic/config/knob_expected_signals.yaml` are implemented; `usecases_synthetic/tests/test_monotonicity.py` passes 26/26 (`pydi-dev/bin/pytest usecases_synthetic/tests/test_monotonicity.py -v`). mypy --strict clean on the new files (only the pre-existing repo-wide `yaml` stubs warning remains). CLI is wired up: `pydi-dev/bin/python usecases_synthetic/scripts/analyze_monotonicity.py --domain companies` runs to completion once M7 has produced `validation/companies/{easy,medium,hard}/metrics.json` (the script errors with a clear "Metrics file not found" when those are absent, as it does today).

Covers 8 active knobs (K1, K2, K3, K4, K5, K6, K8, K10) across sm/em/fusion stages; every signal is direction-only (`qualitative_only: true`) because no knob card pins numeric ranges — this is itself a finding M10 must revisit. Authoring notes live inline at the top of the YAML.

## Purpose

Consume the three per-level `metrics.json` files from M7 (easy, medium, hard) plus the baseline from M5, and answer:

1. **Is each knob's predicted signal monotone** across baseline → easy → medium → hard, in the direction and stage predicted by its *Committee expectations* section?
2. **Has any attribute or stage collapsed** (F1 below a threshold, or F1 drop too steep) such that the difficulty signal is no longer recoverable?
3. **When a test-gold F1 collapses, does the pool-agreement diagnostic corroborate or contradict it?** Per [cross_cutting.md § Protection set semantics](../../knobs/cross_cutting.md#protection-set-semantics-not-replacement-gold), divergence between the two indicates hidden-positive noise rather than real difficulty.

Output is a single `monotonicity_report.md` + `monotonicity_report.csv` under `usecases_synthetic/validation/<domain>/` that M10 uses as the primary evidence for the final triage.

M8 does NOT fix collapses. It surfaces them.

## Spec References

- **Monotonicity requirement:** [../../knobs/cross_cutting.md § Profile model](../../knobs/cross_cutting.md#profile-model--option-b-absolute-target-bands) — "monotone easy→medium→hard shape required"
- **Fix-on-collapse preference order:** [../../knobs/cross_cutting.md § Committee-validated augmentation](../../knobs/cross_cutting.md#committee-validated-augmentation) step 5, and the per-knob table under § Per-knob fix-strategy defaults
- **Pool-vs-test-gold disambiguation:** [../../knobs/cross_cutting.md § Protection set semantics](../../knobs/cross_cutting.md#protection-set-semantics-not-replacement-gold) point 3
- **Per-knob expected signals:** each knob card's § "Committee expectations" — authoritative source for "what should go down and how much"
- **Structural monotonicity (already done):** the orchestrator's `monotonicity_report.csv` from M6 covers structural axis (K3 drop nesting, K8 naming distance, etc.) but NOT semantic committee-based monotonicity. M8 is the committee-based counterpart

## Files to Create

### Library (`usecases_synthetic/lib/`)

| File | Responsibility |
|---|---|
| `monotonicity.py` | Pure functions: `check_monotone(values: list[float], direction: Literal["down","up"]) -> bool`; `knob_expected_signals() -> dict[knob_id, list[SignalExpectation]]` — returns a machine-readable version of every *Committee expectations* section, hand-curated; `match_signals(baseline, levels, expectations) -> list[SignalCheck]`; `detect_collapses(levels, threshold) -> list[Collapse]` |

### Scripts

| File | Responsibility |
|---|---|
| `usecases_synthetic/scripts/analyze_monotonicity.py` | CLI: `--domain companies`. Reads `baselines/companies/baseline_metrics.json` and `validation/companies/{easy,medium,hard}/metrics.json`, runs the signal checks + collapse detection, writes reports |

### Data file

| File | Contents |
|---|---|
| `usecases_synthetic/config/knob_expected_signals.yaml` | Machine-readable per-knob signal expectations, hand-curated from the knob cards. One entry per (knob, stage, metric) with `direction: up|down|flat`, `target_delta_range: [min, max]`, `source: knob_XX_card.md section` |

### Tests

| File | What it tests |
|---|---|
| `test_monotonicity.py` | `check_monotone` edge cases; `match_signals` correctly flags non-monotone runs; `detect_collapses` triggers on test-gold-collapse + pool-agreement-stable pattern |

## `knob_expected_signals.yaml` shape

Hand-authored from the knob cards. Example entries for companies:

```yaml
knob_01_surface:
  - stage: em
    metric: macro_f1
    direction: down
    target_delta_range: [-0.30, -0.05]
    source: knob_01_surface_augmentation.md § Committee expectations
    notes: "sharper for lexical blockers than embedding matchers"
  - stage: em
    metric: "lexical_blocker_member_f1_minus_embedding_blocker_member_f1"
    direction: up  # spread widens
    target_delta_range: [0.05, 0.40]
    notes: "spread between blocker types IS the Knob 1 signal"

knob_08_naming:
  - stage: sm
    metric: macro_f1
    direction: down
    target_delta_range: [-0.50, -0.10]
    source: knob_08_schema_naming.md § Committee expectations
  - stage: sm
    metric: "label_based_member_f1_minus_instance_based_member_f1"
    direction: down  # instance-based should be flatter
    notes: "spread widening — label-based collapses faster than instance-based"

knob_10_reliability:
  - stage: fusion
    metric: overall_accuracy
    direction: down
    target_delta_range: [-0.25, -0.05]
  - stage: fusion
    metric: "voting_strategy_accuracy_minus_per_source_trust_strategy_accuracy"
    direction: up  # spread widens on hard
    notes: "knob 10 medium level — per-source-weighted pulls ahead of voting"
  # ... more entries
```

Authoring this file is a substantial portion of M8's effort. One knob card at a time. If a card's Committee expectations are vague ("monotone drop expected"), record it as `direction: down, target_delta_range: null` and flag it as "qualitative only" — M8 will still check direction but not magnitude.

## Behavior

```python
for knob, expectations in knob_expected_signals.items():
    for exp in expectations:
        levels = [baseline, easy, medium, hard]
        values = [get_metric(L, exp.stage, exp.metric) for L in levels]
        is_monotone = check_monotone(values, exp.direction)
        within_range = (
            exp.target_delta_range is None
            or exp.target_delta_range[0] <= (values[-1] - values[0]) <= exp.target_delta_range[1]
        )
        record SignalCheck(knob, stage, metric, values, is_monotone, within_range)

for level in [easy, medium, hard]:
    for stage in [sm, em, fusion]:
        for member in level.stage.per_member:
            if member.f1 < 0.15 or (baseline.f1 - member.f1) > 0.5:
                if abs(member.pool_agreement - baseline.pool_agreement) < 0.1:
                    # test gold collapsed but pool is stable — hidden-positive noise
                    record Collapse(..., classification="hidden_positive_noise")
                else:
                    # both moved — real collapse
                    record Collapse(..., classification="real_collapse")
```

## `monotonicity_report.md` sections

1. **Executive summary** — one line per knob: "K8 SM signal ✓ monotone, in range / K2 EM signal ✗ non-monotone" (use ASCII check/cross, not emoji per CLAUDE.md).
2. **Per-knob signal table** — columns: knob, stage, metric, baseline→easy→medium→hard values, is_monotone, within_range, source card link.
3. **Collapse table** — rows: level, stage, member, baseline_f1, measured_f1, delta, pool_classification, recommended action (from the per-knob fix-strategy defaults table).
4. **Open questions** — any SignalCheck with `target_delta_range=null` that is direction-correct but magnitude-unclear. These become discussion items for M10.

No automatic fixing. No modifications to knob configs. Just measurement + interpretation.

## Acceptance Criteria

1. `analyze_monotonicity.py --domain companies` runs without error given M5+M7 outputs.
2. `knob_expected_signals.yaml` covers all 8 active knobs (K1, K2, K3, K4, K5, K6, K8, K10); each knob has at least one primary-stage entry with direction specified.
3. `monotonicity_report.md` renders the executive summary, signal table, and collapse table.
4. Pool-agreement-vs-test-gold disambiguation is exercised: the test uses a fixture where one stage has a synthetic collapse pattern and verifies it is classified as `hidden_positive_noise`.
5. `monotonicity_report.csv` is machine-readable by M10 for triage synthesis.
6. `pydi-dev/bin/pytest usecases_synthetic/tests/test_monotonicity.py -v` passes.

## Dependencies

M0, M5, M7. M8 assumes M7 has already been run for all three levels.

## Notes

- `knob_expected_signals.yaml` is the *first* place where the knob cards' qualitative predictions become machine-checkable. Authoring it carefully is the main intellectual cost of M8; everything else is routing and reporting.
- If the knob cards' predictions turn out to be under-specified (many "qualitative only" entries), that's itself a finding for M10 — the algorithm selection phase didn't pin them tightly enough to validate empirically.
- M8 deliberately avoids making fix decisions. Even when a collapse is classified as `hidden_positive_noise`, M8 only *recommends* softening via the fix-strategy table — actual softening is out of scope (it would mean re-running the knob with different params and regenerating variants, which loops back to M6).
- Threshold for collapse detection (F1 < 0.15 or delta > 0.5) is stored in the script config and can be tuned. Default matches M0 `collapse_flag`.
