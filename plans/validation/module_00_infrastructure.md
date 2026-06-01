# Module 0: Validation Infrastructure

**Status:** done (2026-04-11). 37 tests passing. Full `usecases_synthetic/tests/` suite: 367 passed.

## Delivered files

| File | Notes |
|---|---|
| [../../usecases_synthetic/lib/validation_metrics.py](../../usecases_synthetic/lib/validation_metrics.py) | `f1`, `precision_recall_f1`, `macro_f1`, `per_attribute_accuracy`, `delta`, `collapse_flag`. Named `validation_metrics` (not `metrics`) so it does not shadow any other metric module under `lib/` |
| [../../usecases_synthetic/lib/committee.py](../../usecases_synthetic/lib/committee.py) | `CommitteeRunner` ABC, `CommitteeResult`, `MemberResult`. `as_dict()` drops predictions and emits a JSON-serialisable snapshot for M5/M7 |
| [../../usecases_synthetic/lib/variant_loader.py](../../usecases_synthetic/lib/variant_loader.py) | `VariantBundle`, `load_variant`. `level == "baseline"` loads the original use case via the registered per-source formats; other levels load CSV-serialised sources from `usecases/<domain>-augmented/<level>/`. Pooled positives auto-attached when `pools/<domain>/pooled_positives.csv` exists |
| [../../usecases_synthetic/lib/baseline_loader.py](../../usecases_synthetic/lib/baseline_loader.py) | `BaselineMetrics` + `load_baseline`. Format round-trips with `validation_report.write_metrics_json` |
| [../../usecases_synthetic/lib/validation_report.py](../../usecases_synthetic/lib/validation_report.py) | `write_metrics_json` (numpy/Path/set fallback via `_json_default`) and `write_report_md` with per-stage aggregated + per-attribute tables; optional delta column + collapse flag when a baseline is supplied |
| [../../usecases_synthetic/tests/test_validation_metrics.py](../../usecases_synthetic/tests/test_validation_metrics.py) | 21 tests — F1 edge cases, set-based P/R/F1, macro_f1 partitions, delta, collapse flag, per-attribute accuracy |
| [../../usecases_synthetic/tests/test_variant_loader.py](../../usecases_synthetic/tests/test_variant_loader.py) | Fixture augmented variant on tmp_path + real baseline load against the companies use case dir |
| [../../usecases_synthetic/tests/test_validation_report.py](../../usecases_synthetic/tests/test_validation_report.py) | JSON round-trip (incl. via `baseline_loader`), markdown tables, delta column rendering |
| [../../usecases_synthetic/tests/test_committee_base.py](../../usecases_synthetic/tests/test_committee_base.py) | ABC plumbing smoke tests: roster name extraction, fallback naming, `as_dict()` drops predictions |

## Purpose

Shared library consumed by every validation module. No committee logic — only metric helpers, the runner framework (callable roster → metrics), loaders for packaged variants, and the report-writing utilities. Also sets up the validation test infrastructure under `usecases_synthetic/tests/`.

## Spec References

- **Metric conventions:** [../../knobs/cross_cutting.md](../../knobs/cross_cutting.md) § "Committee-validated augmentation" — monotone drop comparison, per-attribute / per-stage
- **Provenance-preserving IO:** [../../CLAUDE.md](../../CLAUDE.md) — `DataFrame.attrs["dataset_name"]` must survive
- **Existing loader patterns:** [../../usecases_synthetic/lib/loaders.py](../../usecases_synthetic/lib/loaders.py) — reference shape
- **Variant layout:** [../../plans/module_10_orchestrator.md](../module_10_orchestrator.md) § "Variant Directory Layout" — what `validate_variant.py` will load
- **Correspondence schema:** [../../CLAUDE.md](../../CLAUDE.md) — `id1, id2, score` columns

## Files to Create

### Library (`usecases_synthetic/lib/`)

| File | Responsibility |
|---|---|
| `committee.py` | `CommitteeRunner` abstract base: roster list, `run(inputs) -> CommitteeResult`. `CommitteeResult` dataclass: `per_member: dict[str, MemberResult]`, `aggregated: dict[str, float]`, `per_attribute: dict[str, dict[str, float]]`, `runtime_s: float`. No stage-specific logic — that's M2/M3/M4 |
| `metrics.py` | Stage-agnostic helpers: `f1(tp, fp, fn)`, `precision_recall_f1(pred: set, gold: set)`, `macro_f1(per_partition)`, `per_attribute_accuracy(fused, gold, columns)`, `delta(baseline, measured)`, `collapse_flag(measured, baseline, threshold)`. Collapse threshold: default F1 drop > 0.5 OR F1 < 0.15 |
| `variant_loader.py` | `load_variant(domain, level) -> VariantBundle`: reads `usecases/<domain>-augmented/<level>/input/{data,schemamatching,entitymatching,fusion}`, returns a dataclass with `sources: dict[str, pd.DataFrame]`, `sm_mapping: pd.DataFrame`, `em_gold: dict[pair, pd.DataFrame]`, `fusion_gold: pd.DataFrame`, `target_schema: dict`. Must set `DataFrame.attrs["dataset_name"]` to match original use case convention |
| `baseline_loader.py` | `load_baseline(domain) -> BaselineMetrics`: reads `usecases_synthetic/baselines/<domain>/baseline_metrics.json`. Matching schema written by M5. Used by M7/M8 |
| `report.py` | `write_metrics_json(path, metrics)`, `write_report_md(path, metrics, baseline=None)`: renders a human-readable markdown rollup with per-stage and per-attribute tables. Keep narrow — no plotting, no HTML |

### Scripts (no new CLI scripts in M0 — those come in M5/M7/M9)

### Tests (`usecases_synthetic/tests/`)

| File | What it tests |
|---|---|
| `test_validation_metrics.py` | `precision_recall_f1` edge cases (empty, full overlap, no overlap); `delta` and `collapse_flag`; `macro_f1` with empty partitions |
| `test_variant_loader.py` | Loading a fixture variant dir produces a bundle with all sources set; `attrs["dataset_name"]` populated; raises on missing fusion gold |
| `test_report.py` | `write_metrics_json` round-trip; `write_report_md` produces valid markdown tables |

## Interfaces

```python
# usecases_synthetic/lib/committee.py
@dataclass
class MemberResult:
    name: str
    predictions: Any  # stage-specific (correspondences, fused df, mapping df)
    metrics: dict[str, float]  # flat metric dict (f1, precision, recall, ...)

@dataclass
class CommitteeResult:
    stage: Literal["sm", "em", "fusion"]
    per_member: dict[str, MemberResult]
    aggregated: dict[str, float]        # majority / mean across members
    per_attribute: dict[str, dict[str, float]]  # attribute -> metric -> value
    per_partition: dict[str, dict[str, float]]  # e.g. source-pair -> metric -> value
    runtime_s: float

class CommitteeRunner(ABC):
    def __init__(self, roster: list[Any], config: dict): ...
    @abstractmethod
    def run(self, bundle: VariantBundle) -> CommitteeResult: ...
```

```python
# usecases_synthetic/lib/variant_loader.py
@dataclass
class VariantBundle:
    domain: str
    level: str  # "baseline" | "easy" | "medium" | "hard"
    sources: dict[str, pd.DataFrame]
    sm_mapping: pd.DataFrame | None         # None for baseline
    target_schema: dict
    em_gold: dict[tuple[str, str], pd.DataFrame]  # (src1, src2) -> test gold
    em_splits: dict[tuple[str, str], dict[str, pd.DataFrame]]  # train/val/test
    fusion_gold: pd.DataFrame
    pooled_positives: pd.DataFrame | None   # None when pool absent (movies, products)
```

## Acceptance Criteria

1. `CommitteeRunner` ABC imports cleanly; concrete subclasses in M2/M3/M4 can be instantiated from a YAML roster.
2. `load_variant("companies", "baseline")` successfully reads the *original* `usecases/companies/` directory (baseline == untouched original).
3. `load_variant("companies", "easy")` reads a packaged variant once M6 produces it. Until then, an M0-level test with a fixture dir under `tests/fixtures/companies-variant-easy/` verifies the loader.
4. `precision_recall_f1` returns (1.0, 1.0, 1.0) on identical sets, (0, 0, 0) on disjoint.
5. `collapse_flag` returns `True` iff F1 drop > 0.5 from baseline OR absolute F1 < 0.15.
6. `write_metrics_json` + `write_report_md` round-trip: json can be re-read into the same dict; markdown renders a per-stage table plus a per-attribute table.
7. `pydi-dev/bin/pytest usecases_synthetic/tests/test_validation_metrics.py test_variant_loader.py test_report.py -v` passes.

## Dependencies

None. Foundation module.

## Notes

- `VariantBundle.level == "baseline"` means the ORIGINAL `usecases/<domain>/` directory — the runner framework treats baseline as "a variant that happens to be the identity". This lets M5 reuse the same committee runners as M7.
- Do NOT put stage-specific logic here. Resist the temptation to merge M0 with M2/M3/M4.
- Metrics are stored as flat dicts so `delta()` is a pure dict subtraction with no schema coupling.
