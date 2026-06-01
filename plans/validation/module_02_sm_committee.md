# Module 2: SM Committee Runner

## Purpose

Concrete `SMCommitteeRunner` that consumes the M1 `sm_committee.yaml` roster, runs each schema matcher against the sources in a `VariantBundle`, and returns a stage-typed `CommitteeResult`. Reports per-matcher mapping F1 against the target schema (SM ground truth), per-member rollup, and a committee-aggregated "majority mapping" view.

SM is the simplest stage to validate — no blocking, no fusion, and Knob 8 is the only knob that targets it. Starting here validates the `CommitteeRunner` contract from M0 before the more complex EM/Fusion runners lean on it.

## Spec References

- **Target stage knob:** [../../knobs/knob_08_schema_naming.md](../../knobs/knob_08_schema_naming.md) § "Committee expectations" — primary SM target; string-similarity collapses fast on cryptic/anonymized, instance-based degrades gracefully
- **Secondary stage knob:** [../../knobs/knob_09_schema_completeness.md](../../knobs/knob_09_schema_completeness.md) § "Committee expectations" — S2 only, but the SM roster must be able to detect its predicted spread (forward-compatibility)
- **PyDI SM module:** [../../PyDI/schemamatching/](../../PyDI/schemamatching/) — `BaseSchemaMatcher.match(dfs, **kwargs)`, `evaluation.py` for scoring
- **SM ground truth:** `usecases/companies/input/schemamatching/target_schema.json` (original) + `usecases/companies-augmented/<level>/input/schemamatching/sm_mapping.csv` (per-variant, written by K8)
- **K8 expected mapping output:** [../../plans/module_01_knob_08.md](../module_01_knob_08.md) — what `sm_mapping.csv` looks like
- **M0 infrastructure:** [module_00_infrastructure.md](module_00_infrastructure.md) § "Interfaces" — `CommitteeRunner`, `VariantBundle`
- **M1 roster:** [module_01_committee_spec.md](module_01_committee_spec.md) — `sm_committee.yaml` schema

## Files to Create

### Library (`usecases_synthetic/lib/`)

| File | Responsibility |
|---|---|
| `committee_sm.py` | `SMCommitteeRunner(CommitteeRunner)`. Instantiates each matcher in the roster, calls `.match(sources, target_schema=...)`, collects each matcher's output as a correspondence DataFrame (`source_column, target_column, score`), scores each against the ground-truth mapping, returns `CommitteeResult` with stage=`"sm"` |
| `committee_sm_scoring.py` (optional, else inline) | `score_sm_mapping(predicted: pd.DataFrame, gold: pd.DataFrame) -> dict[str, float]`: precision/recall/F1 against the set of `(source, column, target_column)` triples in the gold. Also computes per-source-column scores (which attributes were mapped correctly by each matcher) |

### Scripts

None in M2 — SM committee is invoked via `measure_baseline.py` (M5) and `validate_variant.py` (M7).

### Tests (`usecases_synthetic/tests/`)

| File | What it tests |
|---|---|
| `test_committee_sm.py` | `SMCommitteeRunner` instantiation from fixture roster; run on a tiny 2-source fixture produces a `CommitteeResult` with per-member F1 values; label-based matcher scores 1.0 on identical headers, <1.0 on renamed headers (smoke check of the K8 signal direction) |

## Interfaces

```python
# usecases_synthetic/lib/committee_sm.py
class SMCommitteeRunner(CommitteeRunner):
    stage: Literal["sm"] = "sm"

    def __init__(self, roster_path: Path, with_llm: bool = False):
        """Load roster from YAML, filter out LLM members unless with_llm."""

    def run(self, bundle: VariantBundle) -> CommitteeResult:
        """
        For each matcher in the enabled roster:
            predicted = matcher.match(list(bundle.sources.values()),
                                      target_schema=bundle.target_schema)
            member_metrics = score_sm_mapping(predicted, gold_mapping)
        Aggregated metrics:
            - macro F1 across members (mean)
            - per-matcher F1
            - per-source per-column "was mapped correctly by >=1 member" rate
        Returns CommitteeResult(
            stage="sm",
            per_member={name: MemberResult(...)},
            aggregated={"macro_f1": ..., "majority_f1": ..., "min_f1": ..., "max_f1": ...},
            per_attribute={column: {matcher_name: f1, ...}},
            ...)
        """
```

Gold mapping for SM:
- **Baseline** (`level="baseline"`): `usecases/companies/input/schemamatching/target_schema.json` + the SM mappings implied by existing `tests/companies_test/test_workflow_companies.py` comparator column choices. In practice, the hand-authored column name alignment across dbpedia/forbes/fullcontact serves as the gold.
- **Variant** (`level in {easy, medium, hard}`): `usecases/companies-augmented/<level>/input/schemamatching/sm_mapping.csv` as produced by K8 — K8 renames headers but writes the correct source→target mapping alongside. That file IS the gold for the variant.

## Acceptance Criteria

1. `SMCommitteeRunner` loads fixture roster (2 matchers: one label-based, one instance-based) and runs against a 2-source fixture.
2. `CommitteeResult.per_member` contains one entry per enabled matcher with `precision`, `recall`, `f1` populated.
3. `per_attribute` correctly reports per-source-column F1.
4. Smoke check: running the label-based matcher on a source pair with **identical** headers yields F1 ≈ 1.0; running it on a source pair where headers are renamed to `Attribute_1..N` yields F1 < baseline (K8 signal direction).
5. Baseline (identity) and hard-variant loaders both work through the same runner.
6. `pydi-dev/bin/pytest usecases_synthetic/tests/test_committee_sm.py -v` passes.

## Dependencies

M0 (infrastructure), M1 (`sm_committee.yaml`). Must not depend on M3/M4.

## Notes

- SM committee F1 is against the *mapping* (pairs of source column → target column), not against any downstream stage. Do not short-circuit by comparing to a hand-labeled target schema only — the matcher might produce the right target schema but wrong per-source column mappings.
- For the baseline case, the "gold mapping" is implicit in the existing comparator setup in the companies workflow. M2 should extract that explicit mapping into `usecases/companies/input/schemamatching/sm_mapping_gold.csv` — a one-time addition that future validation runs can reuse. If this creates churn in the existing SM files, coordinate with the user first.
- Do not call PyDI's existing `SchemaMatchingEvaluator` blindly — verify first it produces the metric shape we need for `per_attribute`. If not, implement `score_sm_mapping` directly in a small helper; the evaluator's output format is not a hard dependency here.
