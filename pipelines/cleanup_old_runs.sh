#!/usr/bin/env bash
# Remove old pipeline runs and stale v2-schema panel artifacts.
#
# Keeps:
#   - pipelines/products/run_v7d/         (most recent run)
#   - pipelines/products/run_v7d.log
#   - pipelines/products/baselines/       (silver reference — used by the panel)
#   - pipelines/products/checkpoints/     (965MB; em_matching + em_blocking)
#   - pipelines/products/STATUS.md
#   - pipelines/products/baselines.log
#
# Inside run_v7d/ keeps: fused.csv, correspondences.csv,
# per_stage_summary.csv, stage_*_selection.json, effective_committees/,
# v3_panel_vs_notebook/
#
# Run from anywhere; uses absolute paths.
set -euo pipefail

ROOT="/Users/ralph/Dev/PyDI/pipelines/products"

echo "=== Removing old run directories ==="
for d in run_v3 run_v4 run_v5 run_v6 run_v7 run_v7b run_v7c run_v7d_repanel run_replay_v2 run_replay_skipditto; do
    if [[ -d "$ROOT/$d" ]]; then
        echo "  rm -rf $ROOT/$d  ($(du -sh "$ROOT/$d" | cut -f1))"
        rm -rf "$ROOT/$d"
    fi
done

echo ""
echo "=== Removing old run logs ==="
for f in run_v3.log run_v4.log run_v5.log run_v6.log run_v7.log run_v7b.log run_v7c.log run_replay_v2.log run_replay_skipditto.log run_t5.log; do
    if [[ -f "$ROOT/$f" ]]; then
        echo "  rm $ROOT/$f"
        rm -f "$ROOT/$f"
    fi
done

echo ""
echo "=== Cleaning stale artifacts inside run_v7d/ ==="
RUN="$ROOT/run_v7d"
for d in e2e_panel vs_notebook_panel; do
    if [[ -d "$RUN/$d" ]]; then
        echo "  rm -rf $RUN/$d  ($(du -sh "$RUN/$d" | cut -f1))  [stale v2 panel]"
        rm -rf "$RUN/$d"
    fi
done
for f in comparison.md summary.md; do
    if [[ -f "$RUN/$f" ]]; then
        echo "  rm $RUN/$f  [stale generated analysis]"
        rm -f "$RUN/$f"
    fi
done

echo ""
echo "=== Done. Remaining pipelines/products/ contents: ==="
ls -la "$ROOT"
echo ""
echo "=== Remaining run_v7d/ contents: ==="
ls -la "$RUN"
