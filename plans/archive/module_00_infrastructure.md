# Module 0: Core Infrastructure

## Purpose

Foundational library consumed by every knob. Contains no knob logic — only shared utilities for provenance tracking, deterministic RNG, domain config loading, protection sets, cell-collision detection, and data loading. Also sets up the test infrastructure.

## Spec References

- **Provenance schema:** [knobs/cross_cutting.md](../../knobs/cross_cutting.md) § "Per-value provenance (mandatory)" — defines the row-level schema `(entity_id, source, attribute, original_value, new_value, transform_fn, transform_params, knob, level)` and the entity/column-scoped variant
- **Determinism contract:** [knobs/cross_cutting.md](../../knobs/cross_cutting.md) § "Determinism Requirements" — single `numpy.random.default_rng(seed)` per `(domain, variant, knob)` tuple, seed in `difficulty.yaml`, re-runs bit-identical
- **Protection set semantics:** [knobs/cross_cutting.md](../../knobs/cross_cutting.md) § "Gold standard incompleteness and pooling" — `expanded_positives = EM_gold ∪ fusion_gold ∪ pooled_positives`, protection rules for K2/K6
- **Cell-collision coordination:** [knobs/cross_cutting.md](../../knobs/cross_cutting.md) § "Cell-collision coordination" — joint provenance index, skip rules (K1: unconditional skip on prior row; K6: skip except k4_fabricated; K5/K7: defensive skip)
- **Profile model:** [knobs/cross_cutting.md](../../knobs/cross_cutting.md) § "Profile model — absolute target bands" — four artifacts per domain (baseline + easy/medium/hard), monotone requirement
- **Canonical application order:** [knobs/README.md](../../knobs/README.md) § "Canonical knob application order" — S1: K2→K4→K1/5/6/7→K3→K10→K8
- **Existing script patterns:** [usecases_synthetic/scripts/build_pool.py](../../usecases_synthetic/scripts/build_pool.py) — argparse CLI, `Path(__file__).resolve().parents[N]`, print progress, `DomainConfig` dataclass
- **Existing test fixtures:** [tests/conftest.py](../../tests/conftest.py) — `repo_root`, `input_dir`, `get_input_data`, `get_correspondences`, `get_fusion_test_set` patterns
- **Pool artifacts:** [usecases_synthetic/pools/](../../usecases_synthetic/pools/) — `pooled_positives.csv` (columns: `id1, id2, source_1, source_2, pool_agreement`)
- **EM gold standards:** `usecases/<domain>/input/entitymatching/<s1>_2_<s2>_{train,val,test,all}.csv` (columns: `id1, id2, label`)

## Files to Create

### Library (`usecases_synthetic/lib/`)

| File | Responsibility |
|---|---|
| `__init__.py` | Package marker |
| `provenance.py` | `ProvenanceLog` class: append rows, flush to CSV, read existing provenance, merge provenance files. Row schema matches cross_cutting.md spec |
| `rng.py` | `make_rng(domain: str, variant: str, knob: int, master_seed: int = 42) -> numpy.random.Generator` via `SeedSequence.spawn()`. Sub-knob delegation helper |
| `domain_config.py` | `DomainConfig` dataclass (source names, file paths, attribute classes, format families). YAML loader with monotonicity validation across easy/medium/hard. `REPO_ROOT` / path resolution mirroring `build_pool.py` |
| `protection.py` | `build_expanded_positives(domain) -> set[str]`: loads EM gold from `usecases/<domain>/input/entitymatching/` (union of train+val+test), fusion gold entity IDs from `usecases/<domain>/input/fusion/test_set.xml`, pooled positives from `usecases_synthetic/pools/<domain>/pooled_positives.csv`. Exposes `is_protected(entity_id, expanded_positives) -> bool` |
| `loaders.py` | Thin wrappers around `PyDI.io.load_xml`, `load_csv`, `load_json` that preserve `DataFrame.attrs["dataset_name"]`. `load_domain_sources(domain) -> dict[str, DataFrame]` returns all sources keyed by name |
| `collision_index.py` | `CollisionIndex` class: reads provenance CSVs from `output/provenance/`, tracks touched `(entity_id, source, attribute)` triples, `is_touched(entity_id, source, attribute) -> bool`, `is_k4_fabricated(entity_id, source, attribute) -> bool` for K6's exception |

### Tests (`usecases_synthetic/tests/`)

| File | What it tests |
|---|---|
| `__init__.py` | Package marker |
| `conftest.py` | Fixtures: `companies_sources()` → 3 small synthetic DataFrames mimicking companies schema (~20 rows each); `rng()` → seeded generator; `tmp_output_dir(tmp_path)` → temp dir for provenance; `mock_protection_set()` → small expanded_positives with known IDs; `repo_root()` → repository root path |
| `test_provenance.py` | ProvenanceLog append/flush/read round-trip; CSV schema matches spec; 1000-row performance |
| `test_rng.py` | Determinism: same inputs → same first 100 draws; different (domain, variant, knob) → divergent sequences |
| `test_protection.py` | `expanded_positives` contains pooled positives + EM gold IDs; `is_protected` returns True for known protected IDs |
| `test_domain_config.py` | YAML loading succeeds for valid config; rejects non-monotone configs; validates attribute class assignments |

## Acceptance Criteria

1. `ProvenanceLog` appends 1000 rows and flushes to CSV in <1s; re-read matches original
2. `make_rng` with identical inputs produces identical first 100 draws; different tuples diverge
3. `expanded_positives` for companies loads and contains pooled positives (2803 pairs → entity IDs) plus EM gold entity IDs
4. YAML loader rejects a config where `easy` target > `hard` target (monotonicity violation)
5. `CollisionIndex` correctly reports a cell as touched when its provenance row exists
6. All tests pass: `pytest usecases_synthetic/tests/test_provenance.py test_rng.py test_protection.py test_domain_config.py -v`

## Dependencies

None — this is the foundation module.
