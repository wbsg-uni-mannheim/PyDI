# Module 7: Joint Value Perturbation Orchestrator

## Purpose

Thin orchestration layer that applies Knobs 1, 5, and 6 **jointly** in fixed order `K1 → K5 → K6` with shared cell-collision avoidance. Not a new knob — just the coordinator ensuring the three value-perturbation knobs compose correctly without double-touching cells.

## Spec References

- **Canonical order:** [knobs/README.md](../../knobs/README.md) § "Canonical knob application order" — Knobs 1/5/6/7 are the "value perturbation group", applied jointly before K3 and K10
- **Cell-collision coordination:** [knobs/cross_cutting.md](../../knobs/cross_cutting.md) § "Cell-collision coordination" — joint provenance index at `output/provenance/knob_0{1,4,5,6}_*.csv`; K1 unconditional skip on prior; K5 defensive skip; K6 skip except `k4_fabricated`
- **K4 fabricated cells:** [knobs/knob_04_coverage_skew.md](../../knobs/knob_04_coverage_skew.md) § "Joint cell-collision" — K4-fabricated cells have `k4_fabricated=True` in provenance; skipped by K1 and K5, but fair game for K6
- **Per-knob modules:** [module_04_knob_06.md](module_04_knob_06.md), [module_05_knob_05.md](module_05_knob_05.md), [module_06_knob_01.md](module_06_knob_01.md)

## Key Mechanism

1. Load post-K4 DataFrames (output of Knob 4 or baseline if K4 is medium/identity)
2. Initialize `CollisionIndex` from K4 provenance (if any K4 fabrication rows exist)
3. Call K1 dispatcher → writes `output/provenance/knob_01_*.csv`
4. Reload collision index (now includes K1 rows)
5. Call K5 dispatcher → writes `output/provenance/knob_05_*.csv`
6. Reload collision index (now includes K1 + K5 rows)
7. Call K6 dispatcher → writes `output/provenance/knob_06_*.csv`
8. Final collision audit: verify no unexpected overlaps

**Collision rules summary:**
| Knob | Skips cells touched by | Exception |
|---|---|---|
| K1 | K4 | None (unconditional skip) |
| K5 | K1, K4 | None (defensive skip) |
| K6 | K1, K5 | K4-fabricated cells NOT skipped |

## Files to Create

| File | Responsibility |
|---|---|
| `usecases_synthetic/scripts/apply_values_joint.py` | CLI: `--domain`, `--level`. Loads post-K4 data, runs K1→K5→K6 sequentially, manages collision index lifecycle, runs collision audit |
| `usecases_synthetic/tests/test_joint_values.py` | No cell double-touched by K1+K5; K4-fabricated cells only in K6 provenance (not K1/K5); dispatch order enforced; end-to-end on small 10-entity dataset |

## Acceptance Criteria

1. No `(entity_id, source, attribute)` triple appears in more than one of K1/K5 provenance outputs
2. K4-fabricated cells appear in K6 provenance but NOT in K1/K5 provenance
3. Total provenance rows = K1 + K5 + K6 (no gaps, no overlaps except K6 on K4 cells)
4. Order enforcement: K1 provenance timestamps < K5 < K6 (or sequence numbers)
5. `pytest usecases_synthetic/tests/test_joint_values.py -v` passes

## Dependencies

Modules 4 (K6), 5 (K5), 6 (K1) — the three value knobs must be implemented. Module 0 (collision index).
