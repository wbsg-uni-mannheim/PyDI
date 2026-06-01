# Module 4: Fusion Committee Runner

## Purpose

Concrete `FusionCommitteeRunner` that runs the per-attribute fusion strategies declared in `fusion_committee.yaml` against a `VariantBundle`, produces fused records, and evaluates them against the domain's fusion gold standard. Reports per-attribute per-strategy accuracy, per-attribute majority-wins accuracy, and overall accuracy.

Fusion is the stage where K10 (source reliability), K4 (coverage skew), and K1 (surface augmentation on long strings) produce their primary signals, so the runner must expose per-attribute metrics cleanly enough to attribute any collapse to a single knob.

## Spec References

- **Fusion committee draft:** [cross_cutting.md § Committee composition (fusion, draft)](../../knobs/cross_cutting.md#committee-composition-fusion-draft) — the starting point, extended by M1
- **Target stage knobs:**
  - [knob_10_source_reliability.md](../../knobs/knob_10_source_reliability.md) § "Committee expectations" — primary target; per-source-trust vs voting vs entity-level provenance spread
  - [knob_04_coverage_skew.md](../../knobs/knob_04_coverage_skew.md) § "Committee expectations" — primary Fusion target; voting vs trust-weighted
  - [knob_01_surface_augmentation.md](../../knobs/knob_01_surface_augmentation.md) § "Committee expectations" — long-string gold extend-to-accepted-set
  - [knob_05_format_unit.md](../../knobs/knob_05_format_unit.md) § "Committee expectations" — naive vs canonicalizing spread
  - [knob_06_value_noise.md](../../knobs/knob_06_value_noise.md) § "Committee expectations" — per-source noise shape rewards provenance-aware fusers
- **PyDI Fusion module:** [../../PyDI/fusion/](../../PyDI/fusion/) — `DataFusionStrategy`, `DataFusionEngine`, `DataFusionEvaluator`, and conflict resolution functions (`longest_string`, `voting`, `most_frequent`, `prefer_higher_trust`, `median`, `mean`, `union`, `intersection`, etc.)
- **Fusion gold format:** `usecases/companies/input/fusion/test_set.xml` and `validation_set.xml` — XML gold, loaded by `PyDI.io` helpers; see [test_workflow_companies.py:155-169](../../tests/companies_test/test_workflow_companies.py#L155-L169) for the evaluation pattern
- **M0 infrastructure:** [module_00_infrastructure.md](module_00_infrastructure.md)
- **M1 roster:** [module_01_committee_spec.md](module_01_committee_spec.md) — `fusion_committee.yaml`
- **EM input to fusion:** uses the best-performing EM committee member's correspondences (M3) as the input for fusion. The choice is frozen at baseline time and reused on variants

## Files to Create

### Library (`usecases_synthetic/lib/`)

| File | Responsibility |
|---|---|
| `committee_fusion.py` | `FusionCommitteeRunner(CommitteeRunner)`. For each attribute, for each strategy list variant, build a `DataFusionStrategy`, run `DataFusionEngine.run()`, evaluate with `DataFusionEvaluator.evaluate()`, and record per-attribute accuracy. Also computes committee-aggregated per-attribute "best strategy wins" accuracy |
| `committee_fusion_scoring.py` | `score_fusion(fused_df, gold_df, per_attribute_evaluators)` — wraps `DataFusionEvaluator` to return a flat metric dict shaped for `CommitteeResult.per_attribute` |

### Tests (`usecases_synthetic/tests/`)

| File | What it tests |
|---|---|
| `test_committee_fusion.py` | Runs a 2-strategy fixture roster on a 3-source fixture with known gold; per-attribute accuracy computed correctly; smoke check that `voting` strategy on identical sources gives accuracy 1.0; `longest_string` on 3 sources with distinct-length values picks the longest |

## Interfaces

```python
# usecases_synthetic/lib/committee_fusion.py
class FusionCommitteeRunner(CommitteeRunner):
    stage: Literal["fusion"] = "fusion"

    def __init__(self, roster_path: Path):
        """
        Load fusion_committee.yaml:
            {
              attributes: {
                name: [
                  {strategy: voting, evaluation: tokenized_match},
                  {strategy: longest_string, evaluation: tokenized_match},
                  {strategy: most_frequent, evaluation: tokenized_match},
                ],
                assets: [...],
                ...
              }
            }
        """

    def run(self, bundle: VariantBundle,
            correspondences: pd.DataFrame) -> CommitteeResult:
        """
        For each attribute a:
            for each strategy s in roster[a]:
                strategy_obj = build_strategy([(a, s, other_attrs_default)])
                fused = DataFusionEngine(strategy_obj).run(
                    datasets=list(bundle.sources.values()),
                    correspondences=correspondences,
                    id_column="id",
                    include_singletons=False,
                )
                metric = score_fusion(fused, bundle.fusion_gold,
                                      {a: roster[a][i]["evaluation"]})
                record member metric

        Aggregated:
            per_attribute[a]["best_strategy_accuracy"] = max over strategies
            per_attribute[a]["mean_strategy_accuracy"] = mean over strategies
            per_attribute[a]["spread"] = max - min  (used by K10 detection)
            aggregated["overall_accuracy"] = mean over attributes of best strategy
            aggregated["overall_mean_accuracy"] = mean over attributes of mean strategy

        Returns CommitteeResult(
            stage="fusion",
            per_member={(attribute, strategy): MemberResult},
            aggregated={overall_accuracy, overall_mean_accuracy, overall_spread},
            per_attribute={attribute: {best_accuracy, mean_accuracy, spread, strategies}},
            ...)
        """
```

## Correspondence input handling

Fusion needs EM correspondences as input. Decision:
- **Baseline:** uses the EM committee's best-F1 member's correspondences on the *original* data. Choice is frozen by M5's baseline run and recorded in `baselines/companies/baseline_metrics.json` under `em.fusion_input_member`.
- **Variants:** uses the **same** member class on the variant, not necessarily the variant's best member. This prevents the fusion signal from being polluted by EM re-ranking across levels.
- Rationale: we're measuring fusion difficulty under the *same matching pipeline*, not fusion+EM jointly. Fixing the EM input controls for that.

Implementation: `validate_variant.py` (M7) runs the EM runner first, passes the `fusion_input_member`'s correspondences into `FusionCommitteeRunner.run()`.

## Per-attribute strategy lists (companies, starting from cross_cutting.md draft)

M1 finalises these; M4 implements whatever M1 decided. The starting point (subject to M1 review):

| Attribute | Type | Strategies |
|---|---|---|
| `name` | string | `voting`, `longest_string`, `most_frequent` |
| `assets` | numeric | `median`, `mean`, `most_frequent`, `prefer_higher_trust` |
| `revenue` | numeric | `median`, `mean`, `most_frequent`, `prefer_higher_trust` |
| `keypeople_name` | list | `union`, `intersection`, `voting` |
| `founded` | date | `earliest`, `latest`, `most_frequent` (after year canonicalization) |
| `country` | categorical | `voting`, `most_frequent` |
| `city` | string | `voting`, `shortest_string`, `most_frequent` |
| `industry` | categorical | `voting`, `most_frequent` |

Evaluation functions (per [test_workflow_companies.py:152-159](../../tests/companies_test/test_workflow_companies.py#L152-L159)):
- `name`, `country`, `city`, `industry`: `tokenized_match`
- `assets`, `revenue`: `numeric_tolerance_match` with `tolerance=0.1`
- `keypeople_name`: `set_equality_match`
- `founded`: `year_only_match`

## Acceptance Criteria

1. `FusionCommitteeRunner` runs a fixture roster on a 3-source fixture, producing per-attribute accuracy numbers.
2. On the companies baseline end-to-end (through `measure_baseline.py`), overall accuracy is in the same ballpark as [test_workflow_companies.py:170](../../tests/companies_test/test_workflow_companies.py#L170) (≈0.658) — if the `voting` strategy for `name` is the committee choice — confirming we haven't wired fusion wrong.
3. `per_attribute` cleanly exposes per-strategy accuracy so M8 can attribute collapse to K10 (per-source-trust vs voting spread is the K10 signal).
4. `per_attribute[a]["spread"]` is computed as `max - min` across strategies, which is the metric M8 uses to detect K10's predicted expected monotone widening.
5. `pydi-dev/bin/pytest usecases_synthetic/tests/test_committee_fusion.py -v` passes.

## Dependencies

M0 (infrastructure), M1 (fusion roster). M4 depends on correspondences produced by M3 but does not *import* M3 — correspondences are passed in as a `pd.DataFrame` argument.

## Notes

- `DataFusionEngine` sets `forbes.attrs["trust_score"] = 3` etc. in the reference workflow; M4 must mirror this for trust-aware strategies. The trust scores come from `fusion_committee.yaml` (`per_source_trust` section) or the domain config — pick one and document. Recommendation: put trust scores in `fusion_committee.yaml` so the committee definition is self-contained.
- For variants where K10 has reshuffled reliability, the trust scores themselves should NOT change — the committee is measuring how well each strategy adapts to reshuffled reality with fixed trust assignments. This is the K10 signal.
- Per-attribute fusion strategies are evaluated **independently** (one strategy object per (attribute, strategy) pair). This is N×M runs where N=attributes, M=max strategies per attribute. For companies N=8, M≈4 → ~32 runs. Cheap.
- Do NOT cluster correspondences inside M4 — the clustering decision lives in M3 (`clustering="mbm"` by default).
