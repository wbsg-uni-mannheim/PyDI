"""Stage-chaining tests.

These run the lightweight stages (SM, refinement on synthetic
correspondences) against the products baseline bundle and assert
that state mutates between stages as designed. Stages that need an
OpenAI key + LLM members (full SM, EM matchers, fusion LLM members)
are skipped when ``OPENAI_API_KEY`` is unset.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMITTEE_DIR = REPO_ROOT / "usecases_synthetic" / "config" / "committees"

_HAS_LLM_KEY = bool(os.environ.get("OPENAI_API_KEY"))


@pytest.mark.skipif(not _HAS_LLM_KEY, reason="needs OPENAI_API_KEY")
def test_sm_stage_mutates_state() -> None:
    """SM stage runs end-to-end + state.sm_mapping_df + state.sm_winner set."""
    from pipelines.lib.bundle import PipelineState, load_pipeline_bundle
    from pipelines.lib.stage_runners import run_sm
    from usecases_synthetic.lib.committee_paths import resolve_committee_path

    bundle = load_pipeline_bundle("products")
    state = PipelineState(bundle=bundle)
    sm_yaml = resolve_committee_path(
        "sm_committee", "products", committee_dir=COMMITTEE_DIR
    )

    sel = run_sm(state, sm_yaml=sm_yaml, with_llm=False)
    assert sel.winner != "", "SM stage produced no winner"
    assert sel.stage == "sm"
    assert state.sm_winner == sel.winner
    assert state.sm_mapping_df is not None
    assert not state.sm_mapping_df.empty
    # The mapping should at least match the gold size (24 rows for products).
    assert len(state.sm_mapping_df) >= len(bundle.sm_mapping)


def test_refinement_chains_off_synthetic_matcher_predictions() -> None:
    """Refinement stage processes per-pair predictions and emits correspondences."""
    from pipelines.lib.bundle import PipelineState, load_pipeline_bundle
    from pipelines.lib.stage_runners import run_refinement

    bundle = load_pipeline_bundle("products")
    state = PipelineState(bundle=bundle)

    # Fabricate matcher predictions: use the EM val pairs themselves
    # (positives only) so the "predictions" are perfect. Greedy/MBM
    # refinement should keep them all.
    state.matcher_predictions = {}
    for pair, splits in bundle.em_splits.items():
        if "val" not in splits:
            continue
        gold = splits["val"]
        pos = gold[
            gold["label"].astype(str).str.lower().isin({"true", "1", "1.0", "t"})
        ]
        if pos.empty:
            continue
        pair_key = f"{pair[0]}_{pair[1]}"
        # Predictions need id1, id2, score columns.
        preds = pd.DataFrame(
            {
                "id1": pos["id1"].astype(str).values,
                "id2": pos["id2"].astype(str).values,
                "score": [1.0] * len(pos),
            }
        )
        state.matcher_predictions[pair_key] = preds

    assert state.matcher_predictions, "no matcher predictions seeded"

    sel = run_refinement(state, methods=["baseline", "greedy", "mbm"])
    assert sel.stage == "refinement"
    assert sel.winner in {"baseline", "greedy", "mbm"}
    assert state.refinement_winner == sel.winner
    assert state.correspondences is not None
    assert not state.correspondences.empty
    # Winning val F1 should be > 0 since predictions are perfect positives.
    assert sel.val_score > 0.0, f"val_score should be > 0, got {sel.val_score}"


def test_em_gold_swap_restores_correctly() -> None:
    """The _swap/_restore helper does what it says."""
    from pipelines.lib.bundle import PipelineState, load_pipeline_bundle
    from pipelines.lib.stage_runners import _restore_em_gold, _swap_em_gold

    bundle = load_pipeline_bundle("products")
    state = PipelineState(bundle=bundle)
    original_gold = state.bundle.em_gold

    prev = _swap_em_gold(state, split="val")
    # Val split exists for products → swap should have happened.
    assert prev is not None
    assert prev is original_gold
    assert state.bundle.em_gold is not original_gold

    _restore_em_gold(state, prev)
    assert state.bundle.em_gold is original_gold


def test_pipeline_constructs_with_per_stage_llm_flags() -> None:
    """Per-stage LLM toggles are stored on the constructor (default
    SM on, EM/Fusion off per the 2026-06-01 directive); the legacy
    ``with_llm`` kwarg still forces all three to the same value for
    back-compat."""
    from pipelines.lib.pipeline import BestOfBreedPipeline, PipelineConfig

    cfg = PipelineConfig.from_yaml(
        REPO_ROOT / "pipelines" / "configs" / "products.yaml"
    )
    # Defaults: SM on, EM off, Fusion off.
    pipe_default = BestOfBreedPipeline(cfg, committee_dir=COMMITTEE_DIR)
    assert pipe_default.with_llm_sm is True
    assert pipe_default.with_llm_em is False
    assert pipe_default.with_llm_fusion is False

    # Explicit per-stage overrides.
    pipe_em_on = BestOfBreedPipeline(
        cfg, committee_dir=COMMITTEE_DIR, with_llm_em=True, with_llm_fusion=True
    )
    assert pipe_em_on.with_llm_sm is True
    assert pipe_em_on.with_llm_em is True
    assert pipe_em_on.with_llm_fusion is True

    # Legacy with_llm kwarg forces all three.
    pipe_legacy_off = BestOfBreedPipeline(
        cfg, committee_dir=COMMITTEE_DIR, with_llm=False
    )
    assert pipe_legacy_off.with_llm_sm is False
    assert pipe_legacy_off.with_llm_em is False
    assert pipe_legacy_off.with_llm_fusion is False
    pipe_legacy_on = BestOfBreedPipeline(
        cfg, committee_dir=COMMITTEE_DIR, with_llm=True
    )
    assert pipe_legacy_on.with_llm_sm is True
    assert pipe_legacy_on.with_llm_em is True
    assert pipe_legacy_on.with_llm_fusion is True
