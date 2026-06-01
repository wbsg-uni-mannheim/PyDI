"""Smoke tests for the best-of-breed pipeline scaffolding.

These tests verify the *plumbing* (config loads, bundle loads, stage
runner imports, dataclass shapes) without running the heavy committee
members. The end-to-end products run is exercised separately by
``run_best_of_breed.py`` in T5.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_config_loads() -> None:
    from pipelines.lib.pipeline import PipelineConfig

    cfg = PipelineConfig.from_yaml(
        REPO_ROOT / "pipelines" / "configs" / "products.yaml"
    )
    assert cfg.domain == "products"
    assert "sm" in cfg.stages
    assert "fusion" in cfg.stages
    assert len(cfg.column_types) > 5
    assert "title" in cfg.column_types
    # Composite weights, if present, sum close to 1.
    if cfg.composite_weights:
        total = sum(cfg.composite_weights.values())
        assert abs(total - 1.0) < 1e-6, f"weights don't sum to 1: {total}"


def test_bundle_loads() -> None:
    from pipelines.lib.bundle import load_pipeline_bundle

    bundle = load_pipeline_bundle("products")
    assert set(bundle.sources.keys()) == {
        "products_1",
        "products_2",
        "products_3",
        "products_4",
    }
    assert bundle.sm_mapping is not None and not bundle.sm_mapping.empty
    assert bundle.fusion_gold is not None and not bundle.fusion_gold.empty
    assert bundle.fusion_validation is not None and not bundle.fusion_validation.empty
    assert len(bundle.em_gold) == 3
    for pair, splits in bundle.em_splits.items():
        # Each pair must have val + test for best-of-breed selection.
        assert "val" in splits, f"missing val for pair {pair}"
        assert "test" in splits, f"missing test for pair {pair}"


def test_stage_selection_shape() -> None:
    from pipelines.lib.stage_runners import StageSelection

    sel = StageSelection(
        stage="dummy",
        winner="member_a",
        val_score=0.9,
        test_score=0.85,
        per_member_val={"member_a": 0.9, "member_b": 0.7},
        per_member_test={"member_a": 0.85, "member_b": 0.65},
    )
    d = sel.as_dict()
    assert d["winner"] == "member_a"
    assert d["val_score"] == 0.9
    assert d["test_score"] == 0.85
    assert set(d["per_member_val"]) == {"member_a", "member_b"}


def test_pick_winner_tie_break_by_name() -> None:
    from pipelines.lib.stage_runners import _pick_winner

    # Ascending name wins on tie.
    assert _pick_winner({"zebra": 0.9, "apple": 0.9}) == "apple"
    # Higher score wins.
    assert _pick_winner({"apple": 0.7, "zebra": 0.9}) == "zebra"
    # Empty input.
    assert _pick_winner({}) == ""


def test_pipeline_constructs() -> None:
    """Construct the pipeline without running it."""
    from pipelines.lib.pipeline import BestOfBreedPipeline, PipelineConfig

    cfg = PipelineConfig.from_yaml(
        REPO_ROOT / "pipelines" / "configs" / "products.yaml"
    )
    pipe = BestOfBreedPipeline(
        cfg,
        committee_dir=REPO_ROOT / "usecases_synthetic" / "config" / "committees",
        with_llm=False,
    )
    assert pipe.config.domain == "products"


def test_report_writer_with_minimal_state(tmp_path: Path) -> None:
    """Report writer handles a partial run without crashing."""
    from pipelines.lib.bundle import PipelineState, load_pipeline_bundle
    from pipelines.lib.pipeline import PipelineConfig, PipelineRunResult
    from pipelines.lib.report import write_run_artifacts
    from pipelines.lib.stage_runners import StageSelection

    cfg = PipelineConfig.from_yaml(
        REPO_ROOT / "pipelines" / "configs" / "products.yaml"
    )
    bundle = load_pipeline_bundle("products")
    state = PipelineState(bundle=bundle)
    state.fused = pd.DataFrame({"_id": ["c1"], "title": ["Demo"]})
    state.correspondences = pd.DataFrame({"id1": [], "id2": [], "score": []})

    result = PipelineRunResult(
        state=state,
        stage_selections=[
            StageSelection(
                stage="sm",
                winner="label_jw",
                val_score=0.91,
                test_score=0.91,
                per_member_val={"label_jw": 0.91, "instance_tf_cosine": 0.78},
                per_member_test={"label_jw": 0.91, "instance_tf_cosine": 0.78},
                metric_key="f1",
                runtime_s=2.4,
            ),
        ],
        total_runtime_s=3.1,
    )

    write_run_artifacts(result, out_dir=tmp_path, config=cfg)

    # Verify expected files exist.
    assert (tmp_path / "stage_1_sm_selection.json").exists()
    assert (tmp_path / "per_stage_summary.csv").exists()
    assert (tmp_path / "fused.csv").exists()
    assert (tmp_path / "summary.md").exists()

    # Summary CSV has the row we wrote.
    summary = pd.read_csv(tmp_path / "per_stage_summary.csv")
    assert len(summary) == 1
    assert summary.iloc[0]["winner"] == "label_jw"


@pytest.mark.parametrize("non_baseline", ["easy", "medium", "hard"])
def test_load_pipeline_bundle_rejects_non_baseline(non_baseline: str) -> None:
    """The best-of-breed loader only accepts baseline."""
    from pipelines.lib.bundle import load_pipeline_bundle

    with pytest.raises(ValueError, match="baseline data only"):
        load_pipeline_bundle("products", level=non_baseline)
