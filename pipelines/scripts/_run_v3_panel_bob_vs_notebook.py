"""Run the v3 e2e panel: BOB pipeline output as pipe, notebook output as silver.

Standalone script — emits panel artifacts under
``pipelines/products/run_v7d/v3_panel_vs_notebook/`` and prints a
human-readable summary to stdout.
"""

from __future__ import annotations

import ast
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml

from PyDI.evaluation import compute_e2e_panel, SilverStandard
from PyDI.evaluation.clustering import membership_from_correspondences

REPO = Path(__file__).resolve().parents[2]
PIPELINES = REPO / "pipelines"
PRODUCTS_OUT = PIPELINES / "products" / "run_v7d"
NOTEBOOK_BASELINE = PIPELINES / "products" / "baselines"
SOURCES_DIR = REPO / "usecases" / "products" / "input" / "data_cleaned_final"
OUT_DIR = PRODUCTS_OUT / "v3_panel_vs_notebook"

# Stage-winner names that invoke a remote LLM API at pipeline runtime.
# Conservative: ditto_plm + embedding_blocker are LOCAL transformer/SBERT
# models and do NOT count. Update this set if/when a new LLM-API stage
# winner is added.
LLM_API_WINNERS = {
    "llm_openai",
    "magneto_slm_llm",
    "label_llm",
    "matcher_llm",
}

# Path of the per-stage llm_usage_summary.json files. Stages listed here
# track LLM usage; stages absent here never invoked an LLM API.
STAGE_LLM_USAGE_PATHS = {
    "sm": REPO
    / "usecases"
    / "products"
    / "output"
    / "schemamatching"
    / "llm_usage_summary.json",
}

# Always-on LLM components that run regardless of stage winners (e.g.
# information extraction is required upstream of schema matching). These
# tokens are counted unconditionally.
ALWAYS_ON_LLM_USAGE = {
    "ie": REPO
    / "usecases"
    / "products"
    / "output"
    / "informationextraction"
    / "llm_usage_summary.json",
}


def _load_llm_usage(path: Path) -> Dict[str, int]:
    """Read a PyDI llm_usage_summary.json into a normalised totals dict.

    PyDI writes ``total_calls`` / ``total_input_tokens`` /
    ``total_output_tokens`` / ``total_tokens``. The panel's
    ``pipeline_api_tokens`` expects ``n_calls`` / ``input_tokens`` /
    ``output_tokens`` / ``total_tokens``. This helper bridges the two.
    """
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {
        "n_calls": int(raw.get("total_calls", 0) or 0),
        "input_tokens": int(raw.get("total_input_tokens", 0) or 0),
        "output_tokens": int(raw.get("total_output_tokens", 0) or 0),
        "total_tokens": int(raw.get("total_tokens", 0) or 0),
    }


def _collect_api_tokens(
    task_step_metrics: Dict[str, Any],
) -> tuple[Dict[str, int] | None, str | None]:
    """Sum LLM token usage attributable to the FINAL pipeline only.

    Rule: a stage's tokens count when (a) the stage is in
    ``ALWAYS_ON_LLM_USAGE`` (always counted) or (b) the stage's winner
    is in ``LLM_API_WINNERS``. Stages whose winner is non-LLM are
    skipped — selection-time LLM calls for losing candidates are
    selection overhead, not pipeline cost.

    Returns ``(totals, notes)`` where ``totals`` is the dict to pass as
    ``pipeline_api_tokens`` (``None`` when no tokens were counted) and
    ``notes`` is a human-readable audit string listing counted +
    skipped stages with their token counts.
    """
    counted_lines: List[str] = []
    skipped_lines: List[str] = []
    totals = {
        "n_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }

    # Always-on LLM components — count unconditionally if the file exists.
    for label, path in ALWAYS_ON_LLM_USAGE.items():
        if not path.exists():
            continue
        usage = _load_llm_usage(path)
        for k in totals:
            totals[k] += usage[k]
        counted_lines.append(
            f"{label}=always-on ({usage['total_tokens']} tokens, "
            f"{usage['n_calls']} calls)"
        )

    # Stage-conditional LLM usage — count only if the stage winner uses
    # an LLM API.
    for stage_name, usage_path in STAGE_LLM_USAGE_PATHS.items():
        if not usage_path.exists():
            continue
        stage_payload = task_step_metrics.get(stage_name) or {}
        winner = str(stage_payload.get("winner", ""))
        usage = _load_llm_usage(usage_path)
        if winner in LLM_API_WINNERS:
            for k in totals:
                totals[k] += usage[k]
            counted_lines.append(
                f"{stage_name}=winner={winner} ({usage['total_tokens']} tokens, "
                f"{usage['n_calls']} calls)"
            )
        else:
            skipped_lines.append(
                f"{stage_name}=winner={winner or '<unknown>'} non-LLM "
                f"({usage['total_tokens']} tokens, {usage['n_calls']} calls)"
            )

    if totals["total_tokens"] == 0:
        return None, None

    notes_parts = []
    if counted_lines:
        notes_parts.append("counted: " + "; ".join(counted_lines))
    if skipped_lines:
        notes_parts.append("skipped: " + "; ".join(skipped_lines))
    notes = " | ".join(notes_parts) if notes_parts else None
    return totals, notes


def _parse_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str) and value.startswith("["):
        try:
            parsed = ast.literal_eval(value)
            return [str(v) for v in parsed] if isinstance(parsed, list) else []
        except (ValueError, SyntaxError):
            return []
    return []


def build_gold_from_fusion_csv(csv_path: Path) -> SilverStandard:
    """Build a SilverStandard from ``usecases/products/input/fusion/fusion_test_set.csv``.

    The CSV stores one row per gold cluster in a pair-row schema:
    ``id_left, source_left, id_right, source_right, cluster_id, <attribute columns>``.
    Source names are short (``p1``..``p4``); BOB uses ``products_1``..``products_4``,
    and the qualified record IDs are ``products_<N>_<bare_id>``. This rewrites
    both so the gold reference aligns with the pipeline output.
    """
    df = pd.read_csv(csv_path)
    if "filled" in df.columns:
        df = df[df["filled"] == "y"].copy()

    source_map = {
        "p1": "products_1",
        "p2": "products_2",
        "p3": "products_3",
        "p4": "products_4",
    }

    drop_cols = {
        "id_left",
        "id_right",
        "source_left",
        "source_right",
        "filled",
        "gt_source_url",
        "gt_source_2",
        "title_left_raw",
        "title_right_raw",
        "desc_left_raw",
        "desc_right_raw",
        "sampling_type",
        "url",
    }
    attribute_cols = [c for c in df.columns if c not in drop_cols and c != "cluster_id"]
    fused = df[["cluster_id"] + attribute_cols].copy()
    fused["cluster_id"] = fused["cluster_id"].astype(str)

    left = df[["cluster_id", "id_left", "source_left"]].rename(
        columns={"id_left": "record_id", "source_left": "source"}
    )
    right = df[["cluster_id", "id_right", "source_right"]].rename(
        columns={"id_right": "record_id", "source_right": "source"}
    )
    membership = pd.concat([left, right], ignore_index=True)
    membership["source"] = (
        membership["source"].map(source_map).fillna(membership["source"])
    )
    membership["record_id"] = (
        membership["source"].astype(str) + "_" + membership["record_id"].astype(str)
    )
    membership["cluster_id"] = membership["cluster_id"].astype(str)
    membership = membership[["record_id", "source", "cluster_id"]]

    return SilverStandard(fused=fused, membership=membership, cell_provenance=None)


def build_silver_from_notebook(nb_fused: pd.DataFrame) -> SilverStandard:
    """Turn the notebook's fused output into a SilverStandard.

    Notebook IDs are bare integers; BOB uses the ``products_<N>_<id>``
    convention. We use ``_fusion_source_datasets`` to rebuild the
    qualified IDs so they line up with BOB's record IDs.
    """
    membership_rows: List[Dict[str, str]] = []
    fused_rows: List[Dict[str, Any]] = []
    for _, row in nb_fused.iterrows():
        cluster_id = str(row["cluster_id"])
        bare_ids = _parse_list(row["_fusion_sources"])
        datasets = _parse_list(row["_fusion_source_datasets"])
        if len(bare_ids) != len(datasets):
            # Some rows have mismatched lists; skip those.
            continue
        # Build qualified record IDs to match BOB's pipe membership.
        for bare_id, dataset_name in zip(bare_ids, datasets):
            membership_rows.append(
                {
                    "record_id": f"{dataset_name}_{bare_id}",
                    "source": dataset_name,
                    "cluster_id": cluster_id,
                }
            )
        # Pass the fused row through; the panel only needs the
        # attribute values + cluster_id.
        fused_rows.append(row.to_dict())

    fused = pd.DataFrame(fused_rows)
    membership = pd.DataFrame(
        membership_rows, columns=["record_id", "source", "cluster_id"]
    ).drop_duplicates(ignore_index=True)
    return SilverStandard(fused=fused, membership=membership, cell_provenance=None)


def main() -> None:
    print("loading inputs…")
    bob_fused = pd.read_csv(PRODUCTS_OUT / "fused.csv", low_memory=False)
    bob_corr = pd.read_csv(PRODUCTS_OUT / "correspondences.csv", low_memory=False)
    nb_fused = pd.read_csv(NOTEBOOK_BASELINE / "notebook_fused.csv", low_memory=False)
    print(
        f"  BOB fused: {bob_fused.shape}  notebook fused: {nb_fused.shape}  "
        f"BOB correspondences: {bob_corr.shape}"
    )

    # Load source DataFrames (already normalized) and attach the
    # dataset_name attribute the panel expects.
    sources: List[pd.DataFrame] = []
    for i in range(1, 5):
        path = SOURCES_DIR / f"dataset_{i}_normalized.json"
        with path.open("r", encoding="utf-8") as f:
            records = json.load(f)
        df = pd.DataFrame(records)
        df.attrs["dataset_name"] = f"products_{i}"
        # Source records carry bare IDs; build the qualified ID for
        # alignment with BOB's pipe membership.
        df["_id"] = f"products_{i}_" + df["id"].astype(str)
        sources.append(df)
        print(f"  dataset_{i}: {len(df)} records")

    silver = build_silver_from_notebook(nb_fused)
    print(
        f"  silver: {len(silver.fused)} fused rows, "
        f"{len(silver.membership)} membership rows"
    )

    # Also load the fusion test set as gold reference (GR). Uses the
    # full-schema CSV at usecases/products/input/fusion/fusion_test_set.csv
    # (not the trimmed synthetic XML, which only carries 5 attributes).
    gold_path = (
        REPO / "usecases" / "products" / "input" / "fusion" / "fusion_test_set.csv"
    )
    gold = build_gold_from_fusion_csv(gold_path)
    print(
        f"  gold: {len(gold.fused)} fused rows, "
        f"{len(gold.membership)} membership rows"
    )

    # Pipe membership: rebuilt from BOB correspondences using the
    # qualified-IDs that already match BOB's _fusion_sources.
    pipe_membership = membership_from_correspondences(
        sources, bob_corr, id_column="_id"
    )
    # Translate to BOB's cluster_id convention: BOB writes cluster_id as
    # an integer in fused.csv; the membership_from_correspondences helper
    # assigns "group_<n>" ids. The two don't overlap, so we have to align
    # them — easiest path: walk BOB's fused rows, build a map from each
    # record's qualified id back to BOB's cluster_id, then re-label the
    # membership.
    record_to_bob_cluster: Dict[str, str] = {}
    for _, row in bob_fused.iterrows():
        bob_cluster_id = str(row["cluster_id"])
        for bare in _parse_list(row["_fusion_sources"]):
            record_to_bob_cluster[bare] = bob_cluster_id
    pipe_membership["cluster_id"] = pipe_membership["record_id"].map(
        lambda r: record_to_bob_cluster.get(r, "<unknown>")
    )
    pipe_membership = pipe_membership[pipe_membership["cluster_id"] != "<unknown>"]
    print(
        f"  pipe membership: {len(pipe_membership)} rows, "
        f"{pipe_membership['cluster_id'].nunique()} clusters"
    )

    # Load products column_types config from YAML
    config = yaml.safe_load((PIPELINES / "configs" / "products.yaml").read_text())
    column_types = config["column_types"]
    # Restrict to columns actually present in both fused frames so the
    # type-routed metrics don't choke on missing columns.
    column_types = {
        k: v
        for k, v in column_types.items()
        if k in bob_fused.columns and k in silver.fused.columns
    }
    panel_tol = config.get("panel_tolerance", {})
    composite_weights = config.get("composite_weights")
    source_prefix_map = config.get("source_prefix_map", {})
    print(f"  column_types: {len(column_types)} columns evaluable")

    # Per-stage metrics + total runtime from BOB's selection artifacts.
    task_step_metrics: Dict[str, Any] = {}
    total_runtime_s: float = 0.0
    stage_peak_memory_mbs: List[float] = []
    any_peak_memory_present = False
    for stage_path in sorted(PRODUCTS_OUT.glob("stage_*_selection.json")):
        with stage_path.open("r", encoding="utf-8") as f:
            stage_payload = json.load(f)
        stage_name = stage_payload.get("stage", stage_path.stem)
        task_step_metrics[stage_name] = stage_payload
        total_runtime_s += float(stage_payload.get("runtime_s", 0.0))
        if "peak_memory_mb" in stage_payload:
            any_peak_memory_present = True
            try:
                stage_peak_memory_mbs.append(
                    float(stage_payload.get("peak_memory_mb") or 0.0)
                )
            except (TypeError, ValueError):
                pass
    pipeline_peak_memory_mb: float = (
        max(stage_peak_memory_mbs) if stage_peak_memory_mbs else 0.0
    )
    print(
        f"  task_step stages: {list(task_step_metrics)}; "
        f"pipeline runtime ≈ {total_runtime_s:.1f}s, "
        f"peak_memory_mb ≈ {pipeline_peak_memory_mb:.1f}"
    )
    if not any_peak_memory_present:
        print("  peak_memory_mb unavailable — re-run the pipeline to capture it")

    api_tokens, api_notes = _collect_api_tokens(task_step_metrics)
    if api_tokens is not None:
        print(
            f"  api_tokens: total={api_tokens['total_tokens']} "
            f"n_calls={api_tokens['n_calls']} "
            f"(input={api_tokens['input_tokens']}, "
            f"output={api_tokens['output_tokens']})"
        )
        if api_notes:
            print(f"  api_notes: {api_notes}")
    else:
        print("  api_tokens: none attributable to final pipeline")

    panel_kwargs: Dict[str, Any] = {}
    if api_tokens is not None:
        panel_kwargs["pipeline_api_tokens"] = api_tokens
    if api_notes is not None:
        panel_kwargs["pipeline_api_notes"] = api_notes
    if pipeline_peak_memory_mb > 0:
        panel_kwargs["pipeline_peak_memory_mb"] = pipeline_peak_memory_mb

    print("\nrunning compute_e2e_panel…")
    t0 = time.time()
    result = compute_e2e_panel(
        pipe_fused=bob_fused,
        sources_pipe=sources,
        column_types=column_types,
        correspondences_pipe=bob_corr,
        pipe_membership=pipe_membership,
        silver=silver,
        gold=gold,
        pipe_id_column="cluster_id",
        silver_id_column="cluster_id",
        gold_id_column="cluster_id",
        pipe_source_id_column="_id",
        numerical_tolerance=panel_tol.get("default", 0.04),
        numerical_tolerance_overrides=panel_tol.get("overrides", {}),
        composite_weights=composite_weights,
        source_prefix_map=source_prefix_map,
        usecase="products",
        run_id="run_v7d_vs_notebook_v3",
        silver_source_label="pipelines/products/baselines/notebook_fused.csv",
        gold_source_label="usecases/products/input/fusion/fusion_test_set.csv",
        task_step_metrics=task_step_metrics or None,
        pipeline_duration_seconds=total_runtime_s or None,
        **panel_kwargs,
    )
    elapsed = time.time() - t0
    print(f"panel computed in {elapsed:.2f}s\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result.write(OUT_DIR)
    print(f"artifacts → {OUT_DIR.relative_to(REPO)}\n")

    # --- Print a human-readable summary ---
    panel = result.panel
    print("=" * 70)
    print("PANEL SUMMARY  (BOB pipeline = pipe; notebook output = silver)")
    print("=" * 70)
    print()
    print("HEADLINE")
    headline = panel["headline"]
    for ref_level in ("RF", "SR", "GR"):
        block = headline.get(ref_level) or {}
        for metric_name, value in block.items():
            label = f"{ref_level}.{metric_name}"
            if isinstance(value, float):
                print(f"  {label:<25s} {value:>8.4f}")
            else:
                print(f"  {label:<25s} {value}")

    cov_entity = panel["coverage"]["entity"]
    print("\nCOVERAGE — ENTITY (RF + SR)")
    print(
        f"  RF:  n_rows_output={cov_entity['RF']['n_rows_output']}, "
        f"n_rows_largest_input={cov_entity['RF']['n_rows_largest_input']}, "
        f"row_gain={cov_entity['RF']['row_gain_vs_largest_input']:+.4f}"
    )
    if "SR" in cov_entity:
        sr = cov_entity["SR"]
        print(
            f"  SR:  n_reference={sr['n_reference']}, "
            f"recovery_rate={sr['recovery_rate']:.4f}, "
            f"n_recovered={sr['n_recovered']}, n_partial={sr['n_partial']}, "
            f"n_lost={sr['n_lost']}, n_fabricated={sr['n_fabricated']}"
        )

    cov_fact = panel["coverage"]["fact"]
    print("\nCOVERAGE — FACT")
    print(
        f"  RF:  density_output={cov_fact['RF']['density_output']:.4f}, "
        f"density_largest_input={cov_fact['RF']['density_largest_input']:.4f}, "
        f"density_gain={cov_fact['RF']['density_gain']:+.4f}"
    )
    if "SR" in cov_fact:
        sr = cov_fact["SR"]
        print(f"  SR:  overall_drift={sr['overall_drift']:.4f}")
        print(f"  SR:  per-column drift (top 5):")
        sorted_drift = sorted(
            sr["per_column_drift_normalized"].items(), key=lambda x: -x[1]
        )
        for col, drift in sorted_drift[:5]:
            print(f"         {col:<30s} {drift:>7.4f}")

    cov_sb = panel["coverage"]["source_based"]
    print("\nCOVERAGE — SOURCE-BASED")
    if "winning_source_distribution_per_attribute" in cov_sb.get("RF", {}):
        print(
            f"  RF:  winning_source_distribution available "
            f"({len(cov_sb['RF']['winning_source_distribution_per_attribute'])} attrs)"
        )
    else:
        print("  RF:  (no cell_provenance — distribution unavailable)")
    if "SR" in cov_sb:
        sr = cov_sb["SR"]
        coll = sr.get("same_source_collision_rate", {})
        print(
            f"  SR:  same_source_collision_rate  "
            f"silver={coll.get('silver', 0):.4f}, pipe={coll.get('pipe', 0):.4f}, "
            f"delta={coll.get('delta', 0):+.4f}"
        )
        print(
            f"  SR:  source_mix_distribution_js = {sr.get('source_mix_distribution_js', 0):.4f}"
        )
        cov = sr.get("per_source_coverage_rate", {})
        for source, info in cov.items():
            print(
                f"         {source:<15s} reference={info['reference']:.3f} "
                f"pipe={info['pipe']:.3f} delta={info['delta']:+.4f}"
            )

    cons = panel.get("consistency", {})
    if "SR" in cons:
        sr = cons["SR"]
        print(f"\nCONSISTENCY — SR")
        print(f"  mean_validity_delta = {sr.get('mean_validity_delta', 0):+.4f}")
        # Show top regressions
        validity = sr.get("validity_per_column", {})
        regressions = sorted(
            [
                (c, info["delta"])
                for c, info in validity.items()
                if info.get("delta", 0) < 0
            ],
            key=lambda x: x[1],
        )[:5]
        if regressions:
            print(f"  top validity regressions:")
            for c, d in regressions:
                v = validity[c]
                print(
                    f"         {c:<30s} reference={v['validity_rate_reference']:.3f} "
                    f"pipe={v['validity_rate_pipe']:.3f} delta={d:+.4f}"
                )

    cl_sr = panel.get("correctness", {}).get("cluster", {}).get("SR")
    if cl_sr:
        bc = cl_sr.get("bcubed", {})
        align = cl_sr.get("alignment", {})
        print(f"\nCORRECTNESS — CLUSTER (SR)")
        print(
            f"  bcubed P/R/F1 = {bc.get('precision', 0):.4f} / "
            f"{bc.get('recall', 0):.4f} / {bc.get('f1', 0):.4f}"
        )
        print(
            f"  mean_jaccard={align.get('mean_jaccard', 0):.4f}, "
            f"size_match_rate={align.get('size_match_rate', 0):.4f}, "
            f"mean_size_delta={align.get('mean_size_delta', 0):+.4f}"
        )

    fact_sr = panel.get("correctness", {}).get("fact", {}).get("SR")
    if fact_sr:
        print(f"\nCORRECTNESS — FACT (SR)")
        print(f"  macro_accuracy = {fact_sr['macro_accuracy']:.4f}")
        print(f"  micro_accuracy = {fact_sr['micro_accuracy']:.4f}")
        print(
            f"  fully_correct_cluster_rate = {fact_sr['fully_correct_cluster_rate']:.4f}"
        )
        print(
            f"  conflict_rate_pipe={fact_sr['conflict_rate_pipe']:.4f}, "
            f"silver={fact_sr['conflict_rate_reference']:.4f}, "
            f"conflict_only_accuracy={fact_sr['conflict_only_accuracy']:.4f}"
        )
        print(f"  per-attribute accuracy (sorted ascending — worst first):")
        per_attr = fact_sr.get("per_attribute", {})
        sorted_attrs = sorted(per_attr.items(), key=lambda x: x[1].get("accuracy", 1.0))
        for attr, m in sorted_attrs[:8]:
            sim = m.get("similarity_mean", "—")
            sim_s = f"{sim:.3f}" if isinstance(sim, (int, float)) else "—"
            fp = m.get("mismatch_fingerprint", "—")
            print(
                f"         {attr:<25s} acc={m['accuracy']:.4f} "
                f"sim_mean={sim_s} count={m['count']} fp={fp}"
            )

    print(f"\nWARNINGS ({len(result.warnings)}):")
    for w in result.warnings:
        print(f"  - {w[:230]}")

    if result.composite is not None:
        print(f"\nCOMPOSITE  (per reference level)")
        for level, payload in result.composite.items():
            if level == "caveat":
                continue
            print(f"  [{level}]")
            for k, v in payload["subscores"].items():
                w = payload["weights"][k]
                print(f"    {k:<28s} {v:.4f}  (weight={w})")
            print(f"    {'composite_score':<28s} {payload['composite_score']:.4f}")


if __name__ == "__main__":
    main()
