"""
End-to-end metric panel orchestrator.

:func:`compute_e2e_panel` is the single entry point for the panel. It
takes the pipeline's fused output + correspondences + source datasets,
plus a silver-standard bundle, and produces a panel structured by
**Quality dimension × Reference level** (see
``docs/tutorial/e2e_evaluation/metrics.md``):

* **`coverage`** — did we produce the right entities, facts, and
  source-attributions? Subdimensions: ``entity``, ``fact``,
  ``source_based``.
* **`consistency`** — does the output respect declared formats and
  constraints?
* **`correctness`** — do the values and clusters match the reference?
  Subdimensions: ``cluster``, ``fact``.

Each block carries ``RF`` (always), ``SR`` (when ``silver`` is
passed), and ``GR`` (when ``gold`` is passed) sub-blocks.

Artifacts written under ``<output_dir>/``:

* ``panel.json`` — nested panel (canonical machine-readable surface).
* ``panel.csv`` — flat ``(metric_name, value)`` for spreadsheets.
* ``panel_glossary.json`` — companion file documenting every metric
  in panel.json. Static; copied from
  ``PyDI/evaluation/panel_metrics_glossary.json``.
* ``schema_diff.json`` — column overlap + dtype mismatches (audit).
* ``column_metrics.csv`` — one row per column × type-routed metric.
* ``cluster_alignment.csv`` — per-silver-cluster alignment
  (correctness.cluster triage).
* ``cluster_attribute_correctness.csv`` — per-aligned-cluster
  correctness.fact drill-down.
* ``composite_score.json`` — composite headline + per-subscore
  values + the exact weights used.

The orchestrator emits pattern-based diagnostic warnings (e.g.
"per_column_drift low but macro accuracy low — likely
histogram-preserving record errors") so the reader doesn't have to
learn cross-metric reading recipes by heart.

The full design rationale lives in
``plans/plan_e2e_metrics.md``.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .cell_provenance import build_cell_provenance_from_fused
from .attribute_quality import (
    conflict_metrics,
    fused_attribute_quality,
    list_attribute_set_metrics,
    per_attribute_density_delta,
    per_cluster_fully_correct_rate,
    source_attribution_metrics,
)
from .clustering import (
    bcubed_scores,
    cluster_alignment,
    membership_from_correspondences,
)
from .composite import composite_score
from .constraint_validity import (
    column_validity_rate,
    compare_column_validity,
    mean_validity_delta,
)
from .distributional import (
    compute_type_routed_metrics,
    schema_diff,
)
from .schema_consistency import (
    SchemaInput,
    evaluate_schema_consistency,
    write_metric_report,
)
from .silver_standard import SilverStandard
from .source_composition import source_composition_summary

logger = logging.getLogger(__name__)


@dataclass
class E2EPanel:
    """In-memory representation of one panel run.

    The dataclass keeps the structured panel together with the CSV
    surfaces so callers can either read fields directly or persist via
    :meth:`write`.
    """

    panel: Dict[str, Any]
    schema_diff: Optional[Dict[str, Any]] = None
    column_metrics: Optional[pd.DataFrame] = None
    cluster_alignment_table: Optional[pd.DataFrame] = None
    cluster_attribute_correctness: Optional[pd.DataFrame] = None
    composite: Optional[Dict[str, Any]] = None
    warnings: List[str] = field(default_factory=list)
    cluster_alignment_table_gold: Optional[pd.DataFrame] = None
    cluster_attribute_correctness_gold: Optional[pd.DataFrame] = None

    def write(self, output_dir: Union[str, Path]) -> Path:
        """Persist all artifacts under *output_dir*.

        Artifacts present in each mode:

        * **RF-only** (no silver, no gold): ``panel.json`` +
          ``panel.csv`` + ``composite_score.json`` (RF-only composite).
        * **Silver supplied**: adds ``schema_diff.json``,
          ``column_metrics.csv``, ``cluster_alignment.csv``,
          ``cluster_attribute_correctness.csv``; ``composite_score.json``
          now carries both RF and SR composites.
        * **Silver + gold supplied**: adds ``cluster_alignment_gold.csv``
          and ``cluster_attribute_correctness_gold.csv``;
          ``composite_score.json`` additionally carries the GR composite.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(output_dir / "panel.json", self.panel)
        _write_panel_csv(output_dir / "panel.csv", self.panel)
        _copy_glossary(output_dir / "panel_glossary.json")
        if self.schema_diff is not None:
            _write_json(output_dir / "schema_diff.json", self.schema_diff)
        if self.column_metrics is not None and not self.column_metrics.empty:
            self.column_metrics.to_csv(output_dir / "column_metrics.csv", index=False)
        if (
            self.cluster_alignment_table is not None
            and not self.cluster_alignment_table.empty
        ):
            self.cluster_alignment_table.to_csv(
                output_dir / "cluster_alignment.csv", index=False
            )
        if (
            self.cluster_attribute_correctness is not None
            and not self.cluster_attribute_correctness.empty
        ):
            self.cluster_attribute_correctness.to_csv(
                output_dir / "cluster_attribute_correctness.csv", index=False
            )
        if (
            self.cluster_alignment_table_gold is not None
            and not self.cluster_alignment_table_gold.empty
        ):
            self.cluster_alignment_table_gold.to_csv(
                output_dir / "cluster_alignment_gold.csv", index=False
            )
        if (
            self.cluster_attribute_correctness_gold is not None
            and not self.cluster_attribute_correctness_gold.empty
        ):
            self.cluster_attribute_correctness_gold.to_csv(
                output_dir / "cluster_attribute_correctness_gold.csv", index=False
            )
        if self.composite is not None:
            _write_json(output_dir / "composite_score.json", self.composite)
        return output_dir

    def write_per_metric(
        self,
        output_dir: Union[str, Path],
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Path]:
        """Emit one JSON file per quality dimension in the shared envelope.

        Each file uses the same ``{metric, metadata, result}`` envelope as
        ``consistency.json`` (via :func:`write_metric_report`). The files are
        slices of the same :attr:`panel` aggregate written by :meth:`write`,
        so they cannot drift from ``panel.json``.

        Files written (when the block is present and non-empty):
        ``coverage.json``, ``consistency.json``, ``correctness.json``,
        ``headline.json``, and — when resource fields were supplied —
        ``resource_usage.json``.

        ``consistency.json`` carries the flat schema-consistency result
        (``consistency_score`` + ``per_column``) when the panel was computed
        with a ``target_schema`` — matching the standalone consistency report
        — otherwise it carries the whole consistency block. The reference
        (SR/GR) consistency deltas remain available under ``panel.json``.

        Parameters
        ----------
        output_dir : str or Path
            Destination directory (created if absent).
        metadata : mapping, optional
            Extra provenance merged into every file's ``metadata`` block on
            top of the panel-level ``usecase`` / ``run_id`` / ``silver_source``
            / ``gold_source`` fields.

        Returns
        -------
        dict
            ``{metric_name: written_path}``.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        base_metadata: Dict[str, Any] = {
            "usecase": self.panel.get("usecase", ""),
            "run_id": self.panel.get("run_id", ""),
            "silver_source": self.panel.get("silver_source", ""),
        }
        if self.panel.get("gold_source"):
            base_metadata["gold_source"] = self.panel["gold_source"]
        if metadata:
            base_metadata.update(dict(metadata))

        written: Dict[str, Path] = {}
        for metric in ("coverage", "consistency", "correctness", "headline"):
            block = self.panel.get(metric)
            if not block:
                continue
            result: Mapping[str, Any] = block
            if metric == "consistency":
                rf = block.get("RF")
                if isinstance(rf, Mapping) and "consistency_score" in rf:
                    # Flat schema-consistency result — same shape as the
                    # standalone consistency.json.
                    result = rf
            written[metric] = write_metric_report(
                metric,
                result,
                output_dir / f"{metric}.json",
                metadata=base_metadata,
            )

        resource_usage = self.panel.get("resource_usage")
        if resource_usage:
            written["resource_usage"] = write_metric_report(
                "resource_usage",
                resource_usage,
                output_dir / "resource_usage.json",
                metadata=base_metadata,
            )
        return written


def _schema_property_names(target_schema: SchemaInput) -> set[str]:
    """Names of the target schema's declared properties (the canonical
    attributes). Used to restrict the scored column set to the schema."""
    if isinstance(target_schema, Mapping):
        schema = target_schema
    else:
        with open(target_schema, "r", encoding="utf-8") as f:
            schema = json.load(f)
    return set((schema.get("properties") or {}).keys())


def compute_e2e_panel(
    *,
    pipe_fused: pd.DataFrame,
    sources_pipe: Sequence[pd.DataFrame],
    column_types: Mapping[str, str],
    correspondences_pipe: Optional[pd.DataFrame] = None,
    silver: Optional[SilverStandard] = None,
    gold: Optional[SilverStandard] = None,
    pipe_id_column: str = "_fusion_group_id",
    silver_id_column: str = "cluster_id",
    gold_id_column: str = "cluster_id",
    pipe_membership: Optional[pd.DataFrame] = None,
    pipe_source_id_column: Optional[str] = None,
    silver_source_records: Optional[Mapping[str, Mapping[str, Any]]] = None,
    gold_source_records: Optional[Mapping[str, Mapping[str, Any]]] = None,
    cell_provenance_pipe: Optional[pd.DataFrame] = None,
    numerical_tolerance: float = 0.04,
    numerical_tolerance_overrides: Optional[Mapping[str, float]] = None,
    column_constraints: Optional[Mapping[str, Mapping[str, Any]]] = None,
    target_schema: Optional[SchemaInput] = None,
    taxonomy_base_path: Optional[Union[str, Path]] = None,
    semantic_value_similarity: Optional[Callable[[str, str], float]] = None,
    semantic_value_threshold: float = 0.85,
    pipeline_duration_seconds: Optional[float] = None,
    pipeline_peak_memory_mb: Optional[float] = None,
    pipeline_api_cost: Optional[float] = None,
    pipeline_api_cost_currency: str = "EUR",
    pipeline_api_tokens: Optional[Mapping[str, int]] = None,
    pipeline_api_notes: Optional[str] = None,
    task_step_metrics: Optional[Mapping[str, Any]] = None,
    composite_weights: Optional[Mapping[str, Any]] = None,
    source_prefix_map: Optional[Mapping[str, str]] = None,
    usecase: str = "",
    run_id: str = "",
    silver_source_label: str = "",
    gold_source_label: str = "",
) -> E2EPanel:
    """Compute the end-to-end metric panel.

    Operates in one of three modes, selected by which references are
    supplied:

    * **RF-only** (no ``silver``, no ``gold``): emits only the
      reference-free coverage and consistency blocks plus
      ``resource_usage`` / ``task_step``. No clustering or
      value-correctness signal is computed (those require a
      reference); the RF composite is still emitted (structural
      only). Fast — skips membership reconstruction and per-column
      type-routed metrics.
    * **SR** (``silver`` supplied): adds the silver-reference
      sub-blocks under ``coverage``, ``consistency``, and ``correctness``,
      plus ``composite_score``, ``column_metrics.csv``,
      ``cluster_alignment.csv``, ``cluster_attribute_correctness.csv``,
      ``schema_diff.json``.
    * **SR + GR** (``silver`` *and* ``gold`` supplied): adds the
      gold-reference sub-blocks (``coverage.*.GR``,
      ``consistency.GR``, ``correctness.*.GR``) and the two
      ``*_gold.csv`` artifacts.

    Parameters
    ----------
    pipe_fused : DataFrame
        Pipeline's fused output. Must contain ``pipe_id_column``
        when any reference is supplied.
    sources_pipe : sequence of DataFrame
        Pipeline source datasets. Each must carry ``dataset_name`` in
        ``df.attrs`` (PyDI convention). Required — used for RF
        metrics (largest input size, density of largest input).
    column_types : mapping
        Required: maps every column to be evaluated → one of
        ``categorical | numerical | text | datetime | list | identifier``.
    correspondences_pipe : DataFrame, optional
        Post-clusterer correspondences (``id1``, ``id2``, ``score``).
        Used to reconstruct cluster membership when a reference is
        supplied and ``pipe_membership`` is None. Ignored entirely in
        RF-only mode.
    silver : SilverStandard, optional
        Silver reference. When ``None``, no SR sub-blocks are emitted
        and no ``composite_score`` is computed.
    gold : SilverStandard, optional
        Gold reference (typically ``load_workflow_silver(usecase)``).
        When supplied, every SR-applicable metric is also computed
        against gold and emitted under GR keys.
    pipe_id_column, silver_id_column : str
        Cluster id columns on either side. Default for ``pipe_fused``
        is ``"_fusion_group_id"`` (PyDI fusion engine convention).
    pipe_membership : DataFrame, optional
        Pre-built pipeline membership ``(record_id, source,
        cluster_id)``. Built from correspondences when ``None``.
    pipe_source_id_column : str, optional
        ID column on the pipeline source DataFrames.
    silver_source_records : mapping, optional
        ``record_id -> {attribute: value}`` snapshot of the source
        records the silver was built from. When ``None`` the
        conflict-only metric uses ``sources_pipe`` for both sides.
    cell_provenance_pipe : DataFrame, optional
        Pipeline-side per-cell provenance. When ``None``,
        source-attribution + synthesis-rate metrics are skipped with
        a panel warning.
    numerical_tolerance : float, default 0.04
        Global relative tolerance for ``% within tolerance``.
    numerical_tolerance_overrides : mapping, optional
        Per-column overrides.
    pipeline_api_tokens : mapping, optional
        Total LLM token usage attributable to the FINAL pipeline
        configuration only — losing-candidate tokens from selection
        should NOT be included. Caller decides what counts. Expected
        keys (any subset, all coerced to ``int``): ``input_tokens``,
        ``output_tokens``, ``total_tokens``, ``n_calls``. Surfaces under
        ``resource_usage.api_tokens``; omitted when ``None`` or empty.
    pipeline_api_notes : str, optional
        Free-form audit string surfaced under ``resource_usage.api_notes``
        — typically lists which pipeline steps' tokens were counted and
        which were skipped (and why). Omitted when ``None``.
    composite_weights : mapping, optional
        Override of the per-level default weights. Accepts two shapes
        (auto-detected): a flat ``{subscore: weight}`` dict applied to
        SR + GR (ignored for RF), or a per-level nested dict
        ``{'RF': {...}, 'SR': {...}, 'GR': {...}}`` where each level's
        entry takes precedence for that level. See
        :func:`composite.composite_score` for the full contract.
    source_prefix_map : mapping, optional
        Source-id prefix → source name. Passed to source-attribution
        bucketing.
    target_schema : mapping or path, optional
        Canonical target schema (JSON-Schema object or path to a
        ``target_schema.json``). When supplied, the **RF consistency**
        block is computed by the schema-aware engine
        (:func:`evaluate_schema_consistency`) — it validates each filled
        cell against the schema's native constraints plus the
        ``x-pydi-*`` extensions and reports a cell-weighted
        ``consistency_score`` + per-column breakdown. It is required for
        the consistency dimension: at RF the pipeline output is scored, and
        at SR/GR both the output and the reference are scored so the block
        carries ``consistency_score_pipe``, ``consistency_score_reference``,
        and their ``delta``. When omitted, the consistency blocks are empty
        (the type-only ``validity_per_column`` fallback has been removed).
    taxonomy_base_path : str or Path, optional
        Root for resolving relative ``x-pydi-taxonomy`` CSV paths in
        ``target_schema``. Defaults to the schema directory.
    usecase, run_id, silver_source_label : str
        Informational fields written to ``panel.json``.

    Returns
    -------
    E2EPanel
        Structured panel; call :meth:`E2EPanel.write` to persist.
    """
    warnings: List[str] = []
    sources_pipe = list(sources_pipe)

    # --- Build pipe_membership only if at least one reference is supplied ---
    pipe_membership_built: Optional[pd.DataFrame] = None
    if silver is not None or gold is not None:
        if pipe_membership is not None:
            pipe_membership_built = pipe_membership
        elif correspondences_pipe is not None:
            pipe_membership_built = membership_from_correspondences(
                sources_pipe,
                correspondences_pipe,
                id_column=pipe_source_id_column,
            )
        else:
            pipe_membership_built = pd.DataFrame(
                columns=["record_id", "source", "cluster_id"]
            )

        if pipe_id_column in pipe_fused.columns and not pipe_membership_built.empty:
            pipe_fused_ids = set(pipe_fused[pipe_id_column].astype(str))
            membership_cluster_ids = set(
                pipe_membership_built["cluster_id"].astype(str)
            )
            if membership_cluster_ids and not (pipe_fused_ids & membership_cluster_ids):
                warnings.append(
                    f"Pipe fused cluster ids (column '{pipe_id_column}') do not "
                    f"overlap with membership cluster ids — correctness.fact will not "
                    f"align rows. If pipe_membership was rebuilt from "
                    f"correspondences, the membership uses fusion-engine "
                    f"'group_*' ids; pass pipe_id_column='_fusion_group_id' or "
                    f"supply a pipe_membership whose cluster_ids match pipe_fused."
                )

    # --- Auto-derive cell_provenance_pipe from fusion-engine metadata
    # when caller didn't supply it but pipe_fused looks like it came
    # from PyDI's DataFusionEngine (has _fusion_metadata column). ---
    if cell_provenance_pipe is None and "_fusion_metadata" in pipe_fused.columns:
        cell_provenance_pipe = build_cell_provenance_from_fused(
            pipe_fused, pipe_id_column=pipe_id_column
        )
        if cell_provenance_pipe.empty:
            cell_provenance_pipe = None

    # --- Restrict the scored column set to the target schema ---
    # Only attributes declared in the target schema are relevant for the
    # value / distribution / density metrics (RF density + gain, value drift,
    # value-density delta, value correctness, validity) at every reference
    # level. Identifier-typed columns (id, cluster_id, doi, url, ...) are
    # dropped, and columns present in column_types but absent from the schema
    # (e.g. source-only extras) are NOT scored. Cluster- and row-level metrics
    # (recovery, BCubed, Jaccard, row gain, fusion ratio) are unaffected, and
    # the schema-aware consistency score already follows this rule.
    if target_schema is not None:
        _schema_props = _schema_property_names(target_schema)
        if _schema_props:
            column_types = {
                c: t
                for c, t in column_types.items()
                if c in _schema_props and t != "identifier"
            }

    # --- Per-reference schema diff → independent skipped-column sets ---
    # Reference-free metrics never skip columns (there is no reference to
    # mismatch against); each reference block uses ONLY its own schema diff, so
    # silver and gold are scored independently within a single call (passing
    # one no longer perturbs RF or the other reference's numbers).
    schema_diff_result: Optional[Dict[str, Any]] = None
    column_metrics_df: Optional[pd.DataFrame] = None
    silver_columns_skipped: set[str] = set()
    gold_columns_skipped: set[str] = set()

    def _columns_skipped_against(reference_fused: pd.DataFrame) -> set[str]:
        d = schema_diff(pipe_fused, reference_fused)
        # Only genuinely absent columns are skipped. dtype mismatches are NOT
        # skipped: the per-column metrics route by the declared column_types and
        # coerce (pd.to_numeric / pd.to_datetime), so a pandas dtype difference
        # — e.g. a numeric attribute parsed from gold XML as strings vs a typed
        # pipe column — does not prevent the comparison.
        return set(d["columns_pipe_only"]) | set(d["columns_silver_only"])

    # Source records power conflict detection and are available whenever
    # sources are supplied, independent of which reference(s) are present.
    pipe_source_records: Dict[str, Mapping[str, Any]] = _index_source_records(
        sources_pipe, pipe_source_id_column or "id"
    )

    if silver is not None:
        schema_diff_result = schema_diff(pipe_fused, silver.fused)
        silver_columns_skipped = (
            set(schema_diff_result["columns_pipe_only"])
            | set(schema_diff_result["columns_silver_only"])
        )
        # column_metrics.csv stays silver-only (per plan §3)
        column_metrics_rows = compute_type_routed_metrics(
            pipe_fused,
            silver.fused,
            column_types,
            skipped_columns=silver_columns_skipped,
        )
        column_metrics_df = pd.DataFrame(column_metrics_rows)
    if gold is not None:
        gold_columns_skipped = _columns_skipped_against(gold.fused)

    # --- Reference-free (RF) metrics — always computed, reference-free ---
    rf_blocks = _compute_reference_free(
        pipe_fused=pipe_fused,
        sources_pipe=sources_pipe,
        column_types=column_types,
        columns_skipped=set(),
        cell_provenance_pipe=cell_provenance_pipe,
        source_prefix_map=source_prefix_map,
        column_constraints=column_constraints,
        target_schema=target_schema,
        taxonomy_base_path=taxonomy_base_path,
    )

    # --- Silver reference (SR), if provided ---
    silver_blocks: Optional[Dict[str, Any]] = None
    if silver is not None and pipe_membership_built is not None:
        silver_blocks = _compute_against_reference(
            pipe_fused=pipe_fused,
            pipe_membership=pipe_membership_built,
            reference=silver,
            column_types=column_types,
            column_constraints=column_constraints,
            columns_skipped=silver_columns_skipped,
            pipe_id_column=pipe_id_column,
            reference_id_column=silver_id_column,
            numerical_tolerance=numerical_tolerance,
            numerical_tolerance_overrides=numerical_tolerance_overrides,
            semantic_value_similarity=semantic_value_similarity,
            semantic_value_threshold=semantic_value_threshold,
            pipe_source_records=pipe_source_records,
            reference_source_records=silver_source_records,
            cell_provenance_pipe=cell_provenance_pipe,
            source_prefix_map=source_prefix_map,
            target_schema=target_schema,
            taxonomy_base_path=taxonomy_base_path,
            warnings_sink=warnings,
            reference_label_for_warnings="silver",
        )

    # --- Gold reference (GR), if provided ---
    gold_blocks: Optional[Dict[str, Any]] = None
    if gold is not None and pipe_membership_built is not None:
        gold_blocks = _compute_against_reference(
            pipe_fused=pipe_fused,
            pipe_membership=pipe_membership_built,
            reference=gold,
            column_types=column_types,
            column_constraints=column_constraints,
            columns_skipped=gold_columns_skipped,
            pipe_id_column=pipe_id_column,
            reference_id_column=gold_id_column,
            numerical_tolerance=numerical_tolerance,
            numerical_tolerance_overrides=numerical_tolerance_overrides,
            semantic_value_similarity=semantic_value_similarity,
            semantic_value_threshold=semantic_value_threshold,
            pipe_source_records=pipe_source_records,
            reference_source_records=gold_source_records,
            cell_provenance_pipe=cell_provenance_pipe,
            source_prefix_map=source_prefix_map,
            target_schema=target_schema,
            taxonomy_base_path=taxonomy_base_path,
            warnings_sink=warnings,
            reference_label_for_warnings="gold",
        )

    if silver is None and gold is None:
        warnings.append(
            "No reference supplied — only RF (reference-free) metrics are "
            "emitted. Pass silver= and/or gold= (SilverStandard bundles) to "
            "get SR / GR sub-blocks under coverage / consistency / "
            "correctness."
        )

    # --- Pack v3 panel — SR / GR sub-blocks added only when computed ---
    coverage: Dict[str, Any] = {
        "entity": {"RF": rf_blocks["coverage_entity"]},
        "fact": {"RF": rf_blocks["coverage_fact"]},
        "source_based": {"RF": rf_blocks["coverage_source_based"]},
    }
    consistency: Dict[str, Any] = {
        "RF": rf_blocks["consistency"],
        "_design_extensions_pending": (
            "Broader consistency design (ontology-style disjointness checks, "
            "cross-attribute constraints) is still pending. The target-schema "
            "consistency_score (native + x-pydi constraints) is the first "
            "first-class consistency signal."
        ),
    }
    correctness: Dict[str, Any] = {}

    if silver_blocks is not None:
        coverage["entity"]["SR"] = silver_blocks["coverage_entity"]
        coverage["fact"]["SR"] = silver_blocks["coverage_fact"]
        coverage["source_based"]["SR"] = silver_blocks["coverage_source_based"]
        consistency["SR"] = silver_blocks["consistency"]
        correctness["cluster"] = {"SR": silver_blocks["correctness_cluster"]}
        correctness["fact"] = {"SR": silver_blocks["correctness_fact"]}

    if gold_blocks is not None:
        coverage["entity"]["GR"] = gold_blocks["coverage_entity"]
        coverage["fact"]["GR"] = gold_blocks["coverage_fact"]
        coverage["source_based"]["GR"] = gold_blocks["coverage_source_based"]
        consistency["GR"] = gold_blocks["consistency"]
        correctness.setdefault("cluster", {})["GR"] = gold_blocks["correctness_cluster"]
        correctness.setdefault("fact", {})["GR"] = gold_blocks["correctness_fact"]

    # --- Composite (RF always computed; SR/GR added when present) ---
    levels: List[str] = ["RF"]
    if silver_blocks is not None:
        levels.append("SR")
    if gold_blocks is not None:
        levels.append("GR")
    composite: Dict[str, Any] = dict(
        composite_score(
            coverage=coverage,
            consistency=consistency,
            correctness=correctness,
            weights=composite_weights,
            levels=tuple(levels),
        )
    )

    # Headline mirrors the rest of the panel: nested by reference level.
    # Each level carries composite_score; SR/GR additionally carry
    # bcubed_f1 + macro_accuracy. RF has no correctness signal so it
    # only carries the (structural-only) composite_score.
    headline: Dict[str, Any] = {}
    if "RF" in composite:
        headline["RF"] = {"composite_score": composite["RF"]["composite_score"]}
    if silver_blocks is not None:
        headline["SR"] = {
            "bcubed_f1": silver_blocks["correctness_cluster"]["bcubed"]["f1"],
            "macro_accuracy": silver_blocks["correctness_fact"]["macro_accuracy"],
        }
        if "SR" in composite:
            headline["SR"]["composite_score"] = composite["SR"]["composite_score"]
    if gold_blocks is not None:
        headline["GR"] = {
            "bcubed_f1": gold_blocks["correctness_cluster"]["bcubed"]["f1"],
            "macro_accuracy": gold_blocks["correctness_fact"]["macro_accuracy"],
        }
        if "GR" in composite:
            headline["GR"]["composite_score"] = composite["GR"]["composite_score"]

    warnings.extend(
        _diagnostic_warnings(
            coverage=coverage,
            consistency=consistency,
            correctness=correctness,
        )
    )

    resource_usage = _build_resource_usage(
        pipeline_duration_seconds=pipeline_duration_seconds,
        pipeline_peak_memory_mb=pipeline_peak_memory_mb,
        pipeline_api_cost=pipeline_api_cost,
        pipeline_api_cost_currency=pipeline_api_cost_currency,
        pipeline_api_tokens=pipeline_api_tokens,
        pipeline_api_notes=pipeline_api_notes,
    )

    panel: Dict[str, Any] = {
        "usecase": usecase,
        "run_id": run_id,
        "silver_source": silver_source_label,
        "headline": headline,
        "coverage": coverage,
        "consistency": consistency,
        "correctness": correctness,
        "task_step": (
            dict(task_step_metrics)
            if task_step_metrics is not None
            else {
                "_placeholder": True,
                "_design_intent": (
                    "Integrate per-stage metrics from PyDI.{schemamatching,"
                    "entitymatching,fusion}.evaluation. Caller passes them in "
                    "via the optional task_step_metrics kwarg; orchestrator "
                    "packs them verbatim into this block."
                ),
            }
        ),
        "aggregated": {
            "_placeholder": True,
            "_design_intent": (
                "Two-dimensional weighting (Quality × Reference). Revisit "
                "after collecting first metric results across multiple "
                "pipelines. headline.<level>.composite_score carries a "
                "per-reference-level single-axis composite in the meantime."
            ),
        },
        "warnings": warnings,
    }
    if resource_usage is not None:
        panel["resource_usage"] = resource_usage

    schema_payload: Optional[Dict[str, Any]] = None
    if schema_diff_result is not None:
        schema_payload = {
            **_schema_payload(schema_diff_result, column_types),
            "skipped_columns_for_per_column_metrics": sorted(silver_columns_skipped),
        }

    if gold_source_label:
        panel["gold_source"] = gold_source_label

    return E2EPanel(
        panel=panel,
        schema_diff=schema_payload,
        column_metrics=column_metrics_df,
        cluster_alignment_table=(
            silver_blocks["alignment_table"] if silver_blocks is not None else None
        ),
        cluster_attribute_correctness=(
            silver_blocks["cluster_attribute_correctness"]
            if silver_blocks is not None
            else None
        ),
        cluster_alignment_table_gold=(
            gold_blocks["alignment_table"] if gold_blocks is not None else None
        ),
        cluster_attribute_correctness_gold=(
            gold_blocks["cluster_attribute_correctness"]
            if gold_blocks is not None
            else None
        ),
        composite=composite,
        warnings=warnings,
    )


def write_usecase_metrics(
    output_dir: Union[str, Path],
    *,
    pipe_fused: pd.DataFrame,
    sources_pipe: Sequence[pd.DataFrame],
    column_types: Mapping[str, str],
    target_schema: SchemaInput,
    taxonomy_base_path: Optional[Union[str, Path]] = None,
    extra_metadata: Optional[Mapping[str, Any]] = None,
    **panel_kwargs: Any,
) -> E2EPanel:
    """Compute the panel for one use case and persist it under *output_dir*.

    Convenience wrapper used by the per-use-case workflow notebooks: it
    runs :func:`compute_e2e_panel` (schema-aware consistency, via
    ``target_schema``) and writes, under ``<output_dir>`` (typically
    ``usecases/<domain>/output/metrics/``):

    * ``panel.json`` (+ ``panel.csv`` / ``panel_glossary.json`` /
      ``composite_score.json`` and, when a reference is supplied, the
      ``schema_diff.json`` / ``*.csv`` side-artifacts) — the big aggregate.
    * one enveloped ``{metric, metadata, result}`` file per quality
      dimension (``coverage.json``, ``consistency.json``,
      ``correctness.json``, ``headline.json``, and ``resource_usage.json``
      when resource fields were passed) via
      :meth:`E2EPanel.write_per_metric`.

    Pass ``silver=`` (and optionally ``gold=``) through ``panel_kwargs`` to
    get the SR/GR sub-blocks under coverage/correctness/consistency; with no
    reference the panel is reference-free (``consistency.json`` and the RF
    coverage blocks are still emitted).

    Parameters
    ----------
    output_dir : str or Path
        Destination directory (created if absent).
    pipe_fused, sources_pipe, column_types, target_schema, taxonomy_base_path
        Forwarded to :func:`compute_e2e_panel`. ``target_schema`` makes the
        consistency dimension schema-aware (the same engine that backs the
        standalone ``consistency.json``).
    extra_metadata : mapping, optional
        Extra provenance merged into every per-metric file's ``metadata``.
    **panel_kwargs
        Any other :func:`compute_e2e_panel` keyword (``silver``, ``gold``,
        ``correspondences_pipe``, ``pipe_id_column``, ``usecase``,
        ``run_id``, resource fields, ...).

    Returns
    -------
    E2EPanel
        The computed panel (already persisted).
    """
    panel = compute_e2e_panel(
        pipe_fused=pipe_fused,
        sources_pipe=sources_pipe,
        column_types=column_types,
        target_schema=target_schema,
        taxonomy_base_path=taxonomy_base_path,
        **panel_kwargs,
    )
    output_dir = Path(output_dir)
    panel.write(output_dir)
    panel.write_per_metric(output_dir, metadata=extra_metadata)
    return panel


# ---------------------------------------------------------------------------
# Per-reference compute block — runs every reference-dependent metric
# against a single SilverStandard. Called once for silver and (when
# provided) once for gold.
# ---------------------------------------------------------------------------


def _compute_reference_free(
    *,
    pipe_fused: pd.DataFrame,
    sources_pipe: Sequence[pd.DataFrame],
    column_types: Mapping[str, str],
    columns_skipped: set[str],
    cell_provenance_pipe: Optional[pd.DataFrame],
    source_prefix_map: Optional[Mapping[str, str]],
    column_constraints: Optional[Mapping[str, Mapping[str, Any]]] = None,
    target_schema: Optional[SchemaInput] = None,
    taxonomy_base_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Reference-free metrics — computed on the output and the sources only.

    Returns dicts for the RF sub-blocks under each quality dimension:

    * ``coverage_entity``       — ``n_rows_output``, ``n_rows_largest_input``,
                                  ``row_gain_vs_largest_input``.
    * ``coverage_fact``         — ``density_output``,
                                  ``density_largest_input``, ``density_gain``.
    * ``coverage_source_based`` — ``winning_source_distribution_per_attribute``
                                  (per-attribute histogram of which source
                                  contributed the winning value; requires
                                  ``cell_provenance_pipe``).
    * ``consistency``           — pipe-only validity rate per column
                                  (does each non-null cell parse to its
                                  declared type and satisfy its declared
                                  constraints?).
    """
    n_rows_output = len(pipe_fused)

    sizes = [len(s) for s in sources_pipe]
    n_rows_largest_input = max(sizes) if sizes else 0
    row_gain = (
        (n_rows_output - n_rows_largest_input) / n_rows_largest_input
        if n_rows_largest_input
        else 0.0
    )
    coverage_entity = {
        "n_rows_output": n_rows_output,
        "n_rows_largest_input": n_rows_largest_input,
        "row_gain_vs_largest_input": row_gain,
    }

    evaluable_columns = [
        c
        for c, t in column_types.items()
        if t not in {"identifier"}
        and c in pipe_fused.columns
        and c not in columns_skipped
    ]
    if evaluable_columns and n_rows_output:
        density_output = float(
            np.mean(
                [1.0 - float(pipe_fused[c].isna().mean()) for c in evaluable_columns]
            )
        )
    else:
        density_output = 0.0

    if sources_pipe and evaluable_columns:
        largest = max(sources_pipe, key=len)
        cols_in_largest = [c for c in evaluable_columns if c in largest.columns]
        if cols_in_largest and len(largest):
            density_largest_input = float(
                np.mean(
                    [1.0 - float(largest[c].isna().mean()) for c in cols_in_largest]
                )
            )
        else:
            density_largest_input = 0.0
    else:
        density_largest_input = 0.0

    coverage_fact = {
        "density_output": density_output,
        "density_largest_input": density_largest_input,
        "density_gain": density_output - density_largest_input,
    }

    coverage_source_based: Dict[str, Any] = {}
    if cell_provenance_pipe is not None and not cell_provenance_pipe.empty:
        winning = _winning_source_distribution_per_attribute(
            cell_provenance_pipe, source_prefix_map
        )
        if winning:
            coverage_source_based["winning_source_distribution_per_attribute"] = winning

    # Reference-free consistency is ALWAYS schema-aware: every filled cell is
    # validated against the target schema's native constraints + x-pydi-*
    # extensions (the same engine that backs the per-usecase consistency.json).
    # The former type-only ``validity_per_column`` fallback has been removed —
    # without a target schema there is no reference-free consistency score.
    if target_schema is not None:
        consistency = evaluate_schema_consistency(
            pipe_fused,
            target_schema,
            taxonomy_base_path=taxonomy_base_path,
            exclude_identifier_columns=True,
        )
    else:
        consistency = {}

    return {
        "coverage_entity": coverage_entity,
        "coverage_fact": coverage_fact,
        "coverage_source_based": coverage_source_based,
        "consistency": consistency,
    }


def _winning_source_distribution_per_attribute(
    cell_provenance: pd.DataFrame,
    source_prefix_map: Optional[Mapping[str, str]],
) -> Dict[str, Dict[str, float]]:
    """Per attribute, fraction of cells whose winning provenance bucket is X.

    Composite-provenance cells (multiple source IDs) split mass equally
    across the listed sources — same convention as the SR-side source
    attribution metric.
    """
    out: Dict[str, Dict[str, float]] = {}

    def _bucket(record_id: str) -> str:
        if source_prefix_map:
            for prefix, source in source_prefix_map.items():
                if record_id.startswith(prefix):
                    return source
        return record_id

    if cell_provenance.empty:
        return out

    for attribute, group in cell_provenance.groupby("attribute"):
        counter: Dict[str, float] = {}
        total = 0.0
        for source_ids in group["source_ids"]:
            if isinstance(source_ids, str):
                ids = [s.strip() for s in source_ids.split("+") if s.strip()]
            elif isinstance(source_ids, (list, tuple)):
                ids = [str(s) for s in source_ids]
            else:
                continue
            if not ids:
                continue
            weight = 1.0 / len(ids)
            for rid in ids:
                bucket = _bucket(rid)
                counter[bucket] = counter.get(bucket, 0.0) + weight
            total += 1.0
        if total > 0:
            out[str(attribute)] = {k: v / total for k, v in counter.items()}
    return out


def _compute_against_reference(
    *,
    pipe_fused: pd.DataFrame,
    pipe_membership: pd.DataFrame,
    reference: SilverStandard,
    column_types: Mapping[str, str],
    column_constraints: Optional[Mapping[str, Mapping[str, Any]]],
    columns_skipped: set[str],
    pipe_id_column: str,
    reference_id_column: str,
    numerical_tolerance: float,
    numerical_tolerance_overrides: Optional[Mapping[str, float]],
    semantic_value_similarity: Optional[Callable[[str, str], float]],
    semantic_value_threshold: float,
    pipe_source_records: Mapping[str, Mapping[str, Any]],
    reference_source_records: Optional[Mapping[str, Mapping[str, Any]]],
    cell_provenance_pipe: Optional[pd.DataFrame],
    source_prefix_map: Optional[Mapping[str, str]],
    target_schema: Optional[SchemaInput],
    taxonomy_base_path: Optional[Union[str, Path]],
    warnings_sink: List[str],
    reference_label_for_warnings: str,
) -> Dict[str, Any]:
    """Compute every reference-dependent metric block against one SilverStandard.

    Returns a dict with the per-reference content of each v3 sub-block:

    * ``coverage_entity``       — row counts + entity overlap counts
    * ``coverage_fact``         — per-column drift + density delta
    * ``coverage_source_based`` — source composition (+ source attribution
                                  / synthesis rate when provenance available)
    * ``consistency``           — schema-aware consistency_score for pipe and
                                  reference + their delta (per column too)
    * ``correctness_cluster``   — bcubed + alignment scalars
    * ``correctness_fact``      — per_attribute (with fingerprint) + macro/
                                  micro + conflict + fully-correct + list F1
    * ``alignment_table``       — DataFrame (for CSV)
    * ``cluster_attribute_correctness`` — DataFrame (for CSV)

    Provenance-availability warnings get appended to ``warnings_sink``
    with the ``reference_label_for_warnings`` prefix so a user reading
    the warnings can tell which reference is missing what.
    """
    # Cluster alignment against this reference
    alignment = cluster_alignment(pipe_membership, reference.membership)
    alignment_table = alignment["table"]

    # Per-column drift (type-routed)
    column_metrics_rows = compute_type_routed_metrics(
        pipe_fused,
        reference.fused,
        column_types,
        skipped_columns=columns_skipped,
    )
    column_metrics_df_local = pd.DataFrame(column_metrics_rows)
    per_column_drift_normalized = _compute_per_column_drift_normalized(
        column_metrics_df_local, pipe_fused, reference.fused, column_types
    )

    # Schema-aware consistency (pipe vs reference): both sides are scored
    # against the target schema's native + x-pydi constraints, so the
    # reference levels report a schema-driven validity delta. The type-only
    # compare_column_validity fallback has been removed.
    if target_schema is not None:
        pipe_consistency = evaluate_schema_consistency(
            pipe_fused,
            target_schema,
            taxonomy_base_path=taxonomy_base_path,
            exclude_identifier_columns=True,
        )
        reference_consistency = evaluate_schema_consistency(
            reference.fused,
            target_schema,
            taxonomy_base_path=taxonomy_base_path,
            exclude_identifier_columns=True,
        )
        cs_pipe = pipe_consistency.get("consistency_score")
        cs_ref = reference_consistency.get("consistency_score")
        pipe_pc = pipe_consistency.get("per_column") or {}
        ref_pc = reference_consistency.get("per_column") or {}
        per_column_consistency: Dict[str, Any] = {}
        for col in sorted(set(pipe_pc) | set(ref_pc)):
            p = (pipe_pc.get(col) or {}).get("consistency_score")
            r = (ref_pc.get(col) or {}).get("consistency_score")
            per_column_consistency[col] = {
                "consistency_score_pipe": p,
                "consistency_score_reference": r,
                "delta": (p - r) if (p is not None and r is not None) else None,
            }
        consistency_block = {
            "consistency_score_pipe": cs_pipe,
            "consistency_score_reference": cs_ref,
            "delta": (
                (cs_pipe - cs_ref)
                if (cs_pipe is not None and cs_ref is not None)
                else None
            ),
            "per_column": per_column_consistency,
        }
    else:
        consistency_block = {}

    # Cluster-level metrics
    bcubed = bcubed_scores(pipe_membership, reference.membership)
    source_composition = source_composition_summary(
        pipe_membership, reference.membership
    )

    # Fact-level metrics
    reference_source_records_resolved = (
        dict(reference_source_records)
        if reference_source_records is not None
        else pipe_source_records
    )
    fused_quality = fused_attribute_quality(
        pipe_fused,
        reference.fused,
        alignment_table,
        column_types,
        pipe_id_column=pipe_id_column,
        silver_id_column=reference_id_column,
        numerical_tolerance=numerical_tolerance,
        numerical_tolerance_overrides=numerical_tolerance_overrides,
        semantic_value_similarity=semantic_value_similarity,
        semantic_value_threshold=semantic_value_threshold,
    )
    conflict = conflict_metrics(
        pipe_fused,
        reference.fused,
        alignment_table,
        column_types,
        pipe_id_column=pipe_id_column,
        silver_id_column=reference_id_column,
        pipe_membership=pipe_membership,
        silver_membership=reference.membership,
        pipe_source_records=pipe_source_records,
        silver_source_records=reference_source_records_resolved,
        numerical_tolerance=numerical_tolerance,
        numerical_tolerance_overrides=numerical_tolerance_overrides,
    )
    list_metrics_raw = list_attribute_set_metrics(
        pipe_fused,
        reference.fused,
        alignment_table,
        column_types,
        pipe_id_column=pipe_id_column,
        silver_id_column=reference_id_column,
    )
    list_metrics_simplified = _simplify_list_metrics(list_metrics_raw)
    density_delta = per_attribute_density_delta(
        pipe_fused, reference.fused, column_types
    )
    fully_correct_rate = per_cluster_fully_correct_rate(
        fused_quality["per_cluster_correctness"],
        fused_quality["evaluable_attributes"],
    )
    per_attribute_enriched = _enrich_with_normalization_fingerprint(
        fused_quality["per_attribute"], column_types
    )

    # Source attribution + synthesis rate (provenance-gated)
    source_attribution_block: Dict[str, Any] = {}
    if cell_provenance_pipe is not None and reference.cell_provenance is not None:
        attribution = source_attribution_metrics(
            reference.cell_provenance,
            cell_provenance_pipe,
            alignment_table,
            source_inference=source_prefix_map,
        )
        source_attribution_block = {
            "source_attribution_js_per_attribute": attribution[
                "source_attribution_js_per_attribute"
            ],
            "synthesis_rate_per_attribute": attribution["synthesis_rate_per_attribute"],
        }
    else:
        reason_parts = []
        if reference.cell_provenance is None:
            reason_parts.append(
                f"{reference_label_for_warnings}.cell_provenance is None"
            )
        if cell_provenance_pipe is None:
            reason_parts.append("cell_provenance_pipe was not passed")
        warnings_sink.append(
            f"Source-attribution and synthesis-rate metrics skipped against "
            f"{reference_label_for_warnings} (" + ", ".join(reason_parts) + ")."
        )

    # Entity-overlap counts (derived from alignment table)
    n_pipe = len(pipe_fused)
    n_ref = len(reference.fused)
    abs_diff = n_pipe - n_ref
    rel_diff = abs_diff / n_ref if n_ref else 0.0

    if not alignment_table.empty:
        jaccards = alignment_table["jaccard"].astype(float)
        n_recovered = int((jaccards >= 1.0 - 1e-9).sum())
        n_partial = int(((jaccards > 0) & (jaccards < 1.0 - 1e-9)).sum())
        n_lost = int((jaccards <= 1e-9).sum())
        matched_pipe_ids = set(
            alignment_table["best_pipe_cluster_id"].dropna().astype(str)
        )
        all_pipe_cluster_ids = (
            set(pipe_membership["cluster_id"].astype(str))
            if not pipe_membership.empty
            else set()
        )
        n_fabricated = len(all_pipe_cluster_ids - matched_pipe_ids)
    else:
        n_recovered = 0
        n_partial = 0
        n_lost = n_ref
        n_fabricated = (
            pipe_membership["cluster_id"].nunique() if not pipe_membership.empty else 0
        )
    recovery_rate = n_recovered / n_ref if n_ref else 0.0

    overall_drift = (
        float(
            np.mean([v for v in per_column_drift_normalized.values() if v is not None])
        )
        if per_column_drift_normalized
        else 0.0
    )

    coverage_entity = {
        "n_reference": n_ref,
        "row_count_abs_diff": abs_diff,
        "row_count_rel_diff": rel_diff,
        "n_recovered": n_recovered,
        "n_partial": n_partial,
        "n_lost": n_lost,
        "n_fabricated": n_fabricated,
        "recovery_rate": recovery_rate,
    }
    coverage_fact = {
        "per_column_drift_normalized": per_column_drift_normalized,
        "overall_drift": overall_drift,
        "density_delta_per_attribute": density_delta,
    }
    coverage_source_based = {
        "same_source_collision_rate": source_composition["same_source_collision_rate"],
        "source_mix_distribution_js": source_composition["source_mix_distribution_js"],
        "per_source_coverage_rate": source_composition["per_source_coverage_rate"],
        "source_mix_distribution_reference": source_composition[
            "source_mix_distribution_reference"
        ],
        "source_mix_distribution_pipe": source_composition[
            "source_mix_distribution_pipe"
        ],
    }
    coverage_source_based.update(source_attribution_block)

    correctness_cluster = {
        "bcubed": bcubed,
        "alignment": {
            "mean_jaccard": alignment["mean_jaccard"],
            "matched_cluster_rate_at_threshold": alignment[
                "matched_cluster_rate_at_threshold"
            ],
            "matched_threshold": alignment["matched_threshold"],
            "size_match_rate": alignment["size_match_rate"],
            "mean_size_delta": alignment["mean_size_delta"],
            "max_size_overshoot": alignment["max_size_overshoot"],
        },
    }
    correctness_fact = {
        "per_attribute": per_attribute_enriched,
        "macro_accuracy": fused_quality["macro_accuracy"],
        "micro_accuracy": fused_quality["micro_accuracy"],
        "conflict_only_accuracy": conflict["macro_accuracy"],
        "conflict_only_micro_accuracy": conflict["micro_accuracy"],
        "conflict_only_per_attribute": conflict["per_attribute"],
        "conflict_rate_pipe": conflict["conflict_rate_pipe"],
        "conflict_rate_reference": conflict["conflict_rate_silver"],
        "conflict_rate_delta": (
            conflict["conflict_rate_pipe"] - conflict["conflict_rate_silver"]
        ),
        "fully_correct_cluster_rate": fully_correct_rate,
        "list_attribute_set_metrics": list_metrics_simplified,
    }

    cluster_attr_correctness_df = _build_cluster_attribute_correctness(
        fused_quality["per_cluster_correctness"],
        fused_quality["evaluable_attributes"],
    )

    return {
        "coverage_entity": coverage_entity,
        "coverage_fact": coverage_fact,
        "coverage_source_based": coverage_source_based,
        "consistency": consistency_block,
        "correctness_cluster": correctness_cluster,
        "correctness_fact": correctness_fact,
        "alignment_table": alignment_table,
        "cluster_attribute_correctness": cluster_attr_correctness_df,
    }


# ---------------------------------------------------------------------------
# Per-column normalized drift used by composite
# ---------------------------------------------------------------------------


def _compute_per_column_drift_normalized(
    column_metrics: pd.DataFrame,
    pipe_fused: pd.DataFrame,
    silver_fused: pd.DataFrame,
    column_types: Mapping[str, str],
) -> Dict[str, float]:
    """Per-column normalized drift in [0, 1] for composite scoring.

    Reads the per-column type-routed metrics already in
    ``column_metrics`` and normalizes each to a comparable
    ``[0, 1]`` "drift" value:

    * ``categorical`` — `js_divergence` (already 0-1).
    * ``numerical`` — `wasserstein_1` divided by silver's range,
      clipped to [0, 1].
    * ``text`` — `token_js_divergence` (already 0-1).
    * ``datetime`` — `wasserstein_1_days / 365`, clipped to [0, 1].

    Columns of type ``list`` or ``identifier`` are excluded
    (cluster-level set metrics in correctness.fact handle ``list``).
    """
    if column_metrics.empty:
        return {}

    drift: Dict[str, float] = {}
    for column in column_metrics["column"].unique():
        col_type = column_types.get(column)
        if col_type is None or col_type in {"identifier", "list"}:
            continue
        rows = column_metrics[column_metrics["column"] == column]
        value: Optional[float] = None
        if col_type == "categorical":
            cell = rows[rows["metric"] == "js_divergence"]
            if not cell.empty:
                value = float(cell.iloc[0]["value"])
        elif col_type == "numerical":
            cell = rows[rows["metric"] == "wasserstein_1"]
            if not cell.empty and column in silver_fused.columns:
                silver_vals = pd.to_numeric(
                    silver_fused[column], errors="coerce"
                ).dropna()
                if not silver_vals.empty:
                    span = float(silver_vals.max() - silver_vals.min())
                    if span > 0:
                        value = float(cell.iloc[0]["value"]) / span
        elif col_type == "text":
            cell = rows[rows["metric"] == "token_js_divergence"]
            if not cell.empty:
                value = float(cell.iloc[0]["value"])
        elif col_type == "datetime":
            cell = rows[rows["metric"] == "wasserstein_1_days"]
            if not cell.empty:
                value = float(cell.iloc[0]["value"]) / 365.0
        if value is None:
            continue
        drift[str(column)] = float(max(0.0, min(1.0, value)))
    return drift


# ---------------------------------------------------------------------------
# correctness.fact enrichments (normalization fingerprint)
# ---------------------------------------------------------------------------


def _enrich_with_normalization_fingerprint(
    per_attribute: Dict[str, Dict[str, Any]],
    column_types: Mapping[str, str],
) -> Dict[str, Dict[str, Any]]:
    """Tag each text attribute with a normalization-vs-real-error fingerprint.

    For text attributes only. Two signal sources:

    1. **Lexical** (always available): ``similarity_mean − accuracy``
       gap. Gap > 0.10 → close-to-right values → normalization
       difference suspected. Gap < 0.05 → real value errors.
    2. **Semantic** (only when the caller supplies
       ``semantic_value_similarity``): ``semantic_accuracy − accuracy``
       gap. Larger and more reliable than the Levenshtein-based gap —
       catches "USA" vs "United States" type differences the lexical
       gap can't see.

    The semantic gap supersedes the lexical one when both are
    available; the fingerprint then says "normalization differences
    confirmed by semantic similarity" rather than "suspected".
    """
    out: Dict[str, Dict[str, Any]] = {}
    for attr, metrics in per_attribute.items():
        new_metrics = dict(metrics)
        col_type = column_types.get(attr)
        acc = metrics.get("accuracy")
        sem_acc = metrics.get("semantic_accuracy")
        sim = metrics.get("similarity_mean")

        # Semantic gap (when caller supplied semantic_value_similarity) —
        # applies to both text and categorical columns.
        if (
            col_type in {"text", "categorical"}
            and acc is not None
            and sem_acc is not None
        ):
            gap = sem_acc - acc
            if gap > 0.10:
                fingerprint = "normalization_difference_confirmed"
            elif gap < 0.05:
                fingerprint = "real_value_errors"
            else:
                fingerprint = "mixed"
            new_metrics["semantic_vs_strict_gap"] = float(gap)
            new_metrics["mismatch_fingerprint"] = fingerprint
        # Lexical gap (text only — Levenshtein doesn't help on
        # categorical short codes).
        elif col_type == "text" and acc is not None and sim is not None:
            gap = sim - acc
            if gap > 0.10:
                fingerprint = "normalization_difference_suspected"
            elif gap < 0.05:
                fingerprint = "real_value_errors"
            else:
                fingerprint = "mixed"
            new_metrics["accuracy_similarity_gap"] = float(gap)
            new_metrics["mismatch_fingerprint"] = fingerprint
        out[attr] = new_metrics
    return out


def _simplify_list_metrics(
    raw: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """Drop set_precision / set_recall — keep only set_f1, set_jaccard, count."""
    out: Dict[str, Dict[str, float]] = {}
    for attr, metrics in raw.items():
        keep = {}
        for key in ("set_f1", "set_jaccard", "count"):
            if key in metrics:
                keep[key] = metrics[key]
        out[attr] = keep
    return out


def _build_cluster_attribute_correctness(
    per_cluster_correctness: pd.DataFrame,
    evaluable_attributes: Sequence[str],
) -> pd.DataFrame:
    """Build the per-aligned-cluster correctness.fact drill-down CSV.

    Includes ``silver_cluster_id``, ``pipe_cluster_id``, the boolean
    correctness flag per evaluable attribute, plus
    ``n_attributes_correct`` and ``fully_correct``.
    """
    if per_cluster_correctness.empty:
        return pd.DataFrame(
            columns=[
                "silver_cluster_id",
                "pipe_cluster_id",
                *evaluable_attributes,
                "n_attributes_correct",
                "fully_correct",
            ]
        )
    df = per_cluster_correctness.copy()
    attr_cols = [a for a in evaluable_attributes if a in df.columns]

    def _count_correct(row: pd.Series) -> int:
        return int(sum(1 for a in attr_cols if row.get(a) is True))

    def _fully(row: pd.Series) -> bool:
        flags = [row.get(a) for a in attr_cols]
        flags = [f for f in flags if f is not None and not _is_missing(f)]
        return bool(flags) and all(bool(f) for f in flags)

    df["n_attributes_correct"] = df.apply(_count_correct, axis=1)
    df["fully_correct"] = df.apply(_fully, axis=1)
    return df


def _is_missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _build_resource_usage(
    *,
    pipeline_duration_seconds: Optional[float],
    pipeline_peak_memory_mb: Optional[float],
    pipeline_api_cost: Optional[float],
    pipeline_api_cost_currency: str,
    pipeline_api_tokens: Optional[Mapping[str, Any]] = None,
    pipeline_api_notes: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Pack the caller-supplied pipeline runtime/memory/cost numbers.

    Returns ``None`` when no field was provided — the panel omits
    the ``resource_usage`` block entirely in that case so it doesn't
    clutter pipelines that don't care about cost.

    The panel does not measure these itself — by the time the panel
    runs, the pipeline is done. Callers wrap their pipeline execution
    with timing/memory tracking and pass the numbers in.

    ``pipeline_api_tokens`` is emitted under ``api_tokens`` when the
    mapping is non-empty and contains at least one of ``input_tokens``,
    ``output_tokens``, ``total_tokens``, or ``n_calls``. All recognised
    fields are coerced to ``int``. ``pipeline_api_notes`` (free-form
    audit string) is emitted under ``api_notes`` when supplied.
    """
    fields: Dict[str, Any] = {}
    if pipeline_duration_seconds is not None:
        fields["duration_seconds"] = float(pipeline_duration_seconds)
    if pipeline_peak_memory_mb is not None:
        fields["peak_memory_mb"] = float(pipeline_peak_memory_mb)
    if pipeline_api_cost is not None:
        fields["api_cost"] = float(pipeline_api_cost)
        fields["api_cost_currency"] = pipeline_api_cost_currency
    if pipeline_api_tokens:
        recognised = ("input_tokens", "output_tokens", "total_tokens", "n_calls")
        token_fields = {
            key: int(pipeline_api_tokens[key])
            for key in recognised
            if key in pipeline_api_tokens and pipeline_api_tokens[key] is not None
        }
        if token_fields:
            fields["api_tokens"] = token_fields
    if pipeline_api_notes is not None:
        fields["api_notes"] = str(pipeline_api_notes)
    return fields or None


# ---------------------------------------------------------------------------
# Pattern-based diagnostic warnings
# ---------------------------------------------------------------------------


def _diagnostic_warnings(
    *,
    coverage: Mapping[str, Any],
    consistency: Mapping[str, Any],
    correctness: Mapping[str, Any],
) -> List[str]:
    """Surface combinations of metric values that signal known failure modes.

    Reads from the v3 panel shape (``coverage.*``, ``consistency.*``,
    ``correctness.*``). Each warning is a pattern that would require
    cross-metric reading to spot manually; the panel calls them out
    instead.
    """
    warnings: List[str] = []

    # Histogram-preserving record errors: fact_coverage drift low + fact_correctness macro acc low
    fact_sr = (coverage.get("fact") or {}).get("SR") or {}
    drifts = (fact_sr.get("per_column_drift_normalized") or {}).values()
    max_drift = max(drifts) if drifts else 0.0
    fact_correctness = (correctness.get("fact") or {}).get("SR") or {}
    macro_acc = float(fact_correctness.get("macro_accuracy") or 0.0)
    if max_drift < 0.05 and macro_acc < 0.9 and macro_acc > 0.0:
        warnings.append(
            f"coverage.fact.SR.per_column_drift is low (max {max_drift:.3f}) "
            f"but correctness.fact.SR.macro_accuracy is {macro_acc:.3f}. "
            "This is the fingerprint of histogram-preserving record swaps "
            "— distributional metrics see the right column shape, but "
            "individual records have wrong values. Read "
            "cluster_attribute_correctness.csv or "
            "correctness.fact.SR.per_attribute."
        )

    # EM over-merge hidden by BCubed: high F1 + non-trivial same-source collisions
    cluster_sr = (correctness.get("cluster") or {}).get("SR") or {}
    bcubed_f1 = float((cluster_sr.get("bcubed") or {}).get("f1") or 0.0)
    sb_sr = (coverage.get("source_based") or {}).get("SR") or {}
    coll = (sb_sr.get("same_source_collision_rate") or {}).get("pipe")
    if coll is not None and coll > 0.05 and bcubed_f1 > 0.85:
        warnings.append(
            f"correctness.cluster.SR.bcubed.f1 = {bcubed_f1:.3f} looks "
            f"healthy but coverage.source_based.SR.same_source_collision_rate"
            f".pipe is {coll:.3f}. The pipeline is lumping multiple records "
            "from the same source into one cluster — likely EM false "
            "positives that BCubed averages over. Read "
            "coverage.source_based.SR.same_source_collision_rate.by_source "
            "for the offending sources."
        )

    # Source coverage regression
    cov_rate = sb_sr.get("per_source_coverage_rate") or {}
    big_coverage_drops = [
        (source, info.get("delta", 0.0))
        for source, info in cov_rate.items()
        if info.get("delta", 0.0) < -0.15
    ]
    if big_coverage_drops:
        worst = min(big_coverage_drops, key=lambda kv: kv[1])
        warnings.append(
            f"Source coverage dropped by more than 15pp for "
            f"{len(big_coverage_drops)} source(s); worst: {worst[0]} "
            f"delta={worst[1]:+.3f}. Likely cause: blocking-recall or "
            "matcher-recall regression on cross-source pairs for that "
            "source."
        )

    # Normalization-difference fingerprints (text + categorical)
    per_attribute = fact_correctness.get("per_attribute") or {}
    confirmed_normalization = [
        attr
        for attr, m in per_attribute.items()
        if m.get("mismatch_fingerprint") == "normalization_difference_confirmed"
    ]
    suspected_normalization = [
        attr
        for attr, m in per_attribute.items()
        if m.get("mismatch_fingerprint") == "normalization_difference_suspected"
    ]
    if confirmed_normalization:
        warnings.append(
            f"Normalization differences *confirmed* (semantic_accuracy ≫ "
            f"accuracy) on attributes: {sorted(confirmed_normalization)}. "
            "Share the NormalizationSpec between silver builder and pipeline."
        )
    if suspected_normalization:
        warnings.append(
            f"Possible normalization differences (accuracy ≪ similarity_mean) "
            f"on text attributes: {sorted(suspected_normalization)}. Share "
            "the NormalizationSpec between silver builder and pipeline to "
            "rule out, or pass a semantic_value_similarity callable to "
            "confirm."
        )

    # Schema-consistency regression — reads consistency.SR.per_column deltas
    cons_sr = consistency.get("SR") or {}
    per_column = cons_sr.get("per_column") or {}
    validity_regressions = [
        (col, info["delta"])
        for col, info in per_column.items()
        if info.get("delta") is not None and info["delta"] <= -0.05
    ]
    if validity_regressions:
        worst = min(validity_regressions, key=lambda kv: kv[1])
        warnings.append(
            f"Schema consistency dropped by ≥ 5pp vs silver on "
            f"{len(validity_regressions)} column(s); worst: '{worst[0]}' "
            f"delta={worst[1]:+.3f}. The pipeline emits cells that violate the "
            "target schema's declared type or constraints (out-of-range "
            "values, enum/taxonomy violations). Read "
            "consistency.SR.per_column.<column> for the per-side scores."
        )

    return warnings


# ---------------------------------------------------------------------------
# Schema diff payload + source-record indexing
# ---------------------------------------------------------------------------


def _schema_payload(
    schema_diff_result: Mapping[str, Any], column_types: Mapping[str, str]
) -> Dict[str, Any]:
    return {
        "columns_shared": list(schema_diff_result.get("columns_shared", [])),
        "columns_silver_only": list(schema_diff_result.get("columns_silver_only", [])),
        "columns_pipe_only": list(schema_diff_result.get("columns_pipe_only", [])),
        "dtype_mismatches": list(schema_diff_result.get("dtype_mismatches", [])),
        "column_types_used": dict(column_types),
    }


def _index_source_records(
    sources: Sequence[pd.DataFrame], id_column: str
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for df in sources:
        column = id_column if id_column in df.columns else None
        if column is None:
            for candidate in ("_id", "id"):
                if candidate in df.columns:
                    column = candidate
                    break
        if column is None:
            logger.debug(
                "Source DataFrame lacks id/_id column; skipping in source-records index"
            )
            continue
        for _, row in df.iterrows():
            record_id = row.get(column)
            if record_id is None or pd.isna(record_id):
                continue
            out[str(record_id)] = row.to_dict()
    return out


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


_GLOSSARY_FILENAME = "panel_metrics_glossary.json"


def _copy_glossary(destination: Path) -> None:
    """Copy the canonical metric glossary next to ``panel.json``.

    The glossary ships alongside the ``PyDI.evaluation`` package as
    ``panel_metrics_glossary.json``. A missing source file is logged
    and otherwise ignored — emitting the panel should never fail
    because the static companion file is unavailable.
    """
    source = Path(__file__).parent / _GLOSSARY_FILENAME
    if not source.exists():
        logger.warning(
            "Panel glossary file %s not found; skipping panel_glossary.json", source
        )
        return
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    except OSError as exc:
        logger.warning("Failed to copy panel glossary to %s: %s", destination, exc)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.tolist()
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, frozenset):
        return sorted(value)
    return str(value)


def _write_panel_csv(path: Path, panel: Mapping[str, Any]) -> None:
    rows: List[Dict[str, Any]] = []

    def _emit(prefix: str, value: Any) -> None:
        if isinstance(value, Mapping):
            for k, v in value.items():
                _emit(f"{prefix}.{k}" if prefix else str(k), v)
        elif isinstance(value, (list, tuple)):
            return
        else:
            rows.append({"metric_name": prefix, "value": value})

    for dimension in ("coverage", "consistency", "correctness"):
        _emit(dimension, panel.get(dimension, {}))
    if "resource_usage" in panel:
        _emit("resource_usage", panel["resource_usage"])

    headline_block = panel.get("headline", {}) or {}
    for ref_level in ("RF", "SR", "GR"):
        for metric_name, value in (headline_block.get(ref_level) or {}).items():
            rows.append(
                {"metric_name": f"headline.{ref_level}.{metric_name}", "value": value}
            )

    pd.DataFrame(rows, columns=["metric_name", "value"]).to_csv(path, index=False)
