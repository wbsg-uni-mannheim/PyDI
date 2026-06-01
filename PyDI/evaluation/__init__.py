"""
End-to-end pipeline evaluation for PyDI.

This subpackage compares the fused output of a full integration pipeline
against a reference ("silver-standard") dataset. It complements the
per-stage evaluations under ``schemamatching/``, ``entitymatching/``,
and ``fusion/``: those answer "is each stage correct?"; this subpackage
answers "did the whole pipeline reproduce the silver dataset?".

The panel is organised by **Quality dimension × Reference level**:

* ``coverage``     — Did we produce the right entities, facts, and
                     source-attributions? Has RF (reference-free), SR
                     (silver), and (when gold is supplied) GR keys.
* ``consistency``  — Does the output respect declared formats and
                     constraints? Validity_per_column is the first
                     first-class signal.
* ``correctness``  — Do the values and clusters match the reference?
                     SR (and optional GR) only — no RF version.
* ``resource_usage`` — Optional pipeline cost block.
* ``task_step`` / ``aggregated`` — Placeholders (see plan v3).

See ``docs/tutorial/e2e_evaluation/e2e_evaluation_metrics.md`` for
the human-readable reference and
``plans/plan_e2e_metrics_v3.md`` for the design rationale.
"""

from __future__ import annotations

from .cell_provenance import build_cell_provenance_from_fused
from .panel import E2EPanel, compute_e2e_panel
from .silver_standard import (
    SilverStandard,
    load_synthetic_silver,
    load_workflow_silver,
)

__all__ = [
    "SilverStandard",
    "load_synthetic_silver",
    "load_workflow_silver",
    "compute_e2e_panel",
    "E2EPanel",
    "build_cell_provenance_from_fused",
]
