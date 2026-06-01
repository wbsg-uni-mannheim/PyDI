"""Tests for the v2 orchestration: mode dispatch, ditto/sc_block filters,
sweep harness bridge.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMITTEE_DIR = REPO_ROOT / "usecases_synthetic" / "config" / "committees"


def _make_pipeline(tmp_path: Path, **kwargs):
    from pipelines.lib.pipeline import BestOfBreedPipeline, PipelineConfig

    cfg = PipelineConfig.from_yaml(
        REPO_ROOT / "pipelines" / "configs" / "products.yaml"
    )
    kwargs.setdefault("with_llm", False)
    return BestOfBreedPipeline(
        cfg,
        committee_dir=COMMITTEE_DIR,
        out_dir=tmp_path,
        **kwargs,
    )


def test_mode_validation() -> None:
    from pipelines.lib.pipeline import BestOfBreedPipeline, PipelineConfig

    cfg = PipelineConfig.from_yaml(
        REPO_ROOT / "pipelines" / "configs" / "products.yaml"
    )
    with pytest.raises(ValueError, match="Unknown mode"):
        BestOfBreedPipeline(cfg, committee_dir=COMMITTEE_DIR, mode="invalid_mode")


def test_matching_yaml_filter_drops_ditto_without_override(tmp_path: Path) -> None:
    """Without --ditto-checkpoint-override, ditto_plm is LOUDLY disabled
    when the YAML default lives under usecases_synthetic/cache/.
    """
    pipe = _make_pipeline(tmp_path)  # no ditto_checkpoint_override
    src_yaml = COMMITTEE_DIR / "em_matching_committee_products.yaml"
    effective = pipe._maybe_filter_matching_yaml(src_yaml)
    data = yaml.safe_load(effective.read_text())
    by_name = {m["name"]: m for m in data["members"]}
    assert by_name["ditto_plm"]["enabled_by_default"] is False
    # Other members untouched.
    assert by_name["llm_matcher"].get("enabled_by_default", True) is True
    assert by_name["magellan"].get("enabled_by_default", True) is True
    # Original yaml unchanged.
    src_data = yaml.safe_load(src_yaml.read_text())
    src_by_name = {m["name"]: m for m in src_data["members"]}
    assert src_by_name["ditto_plm"].get("enabled_by_default", True) is True


def test_matching_yaml_filter_ditto_checkpoint_override_keeps_member(
    tmp_path: Path,
) -> None:
    """With --ditto-checkpoint-override, ditto_plm stays enabled and the
    checkpoint path is rewritten to the pipeline-isolated location.
    """
    fake_ckpt = tmp_path / "my_pipeline_ditto"
    fake_ckpt.mkdir()
    pipe = _make_pipeline(tmp_path, ditto_checkpoint_override=fake_ckpt)
    src_yaml = COMMITTEE_DIR / "em_matching_committee_products.yaml"
    effective = pipe._maybe_filter_matching_yaml(src_yaml)
    data = yaml.safe_load(effective.read_text())
    by_name = {m["name"]: m for m in data["members"]}
    assert by_name["ditto_plm"].get("enabled_by_default", True) is True
    assert by_name["ditto_plm"]["matcher"]["params"]["checkpoint_path"] == str(
        fake_ckpt
    )


def test_fusion_yaml_filter_removes_llm_only(tmp_path: Path) -> None:
    """with_llm=False removes llm_only from the C12 roster entirely.

    The C12 ``FusionCommitteeRunner`` ignores ``enabled_by_default`` on
    individual members, so the only way to drop a member is to remove
    it from the ``members:`` list. (Regression guard for the v3-v7c
    bug where ``llm_only`` ran silently despite the filter.)
    """
    pipe = _make_pipeline(tmp_path, with_llm=False)
    src_yaml = COMMITTEE_DIR / "fusion_committee_products.yaml"
    effective = pipe._maybe_filter_fusion_yaml(src_yaml)
    data = yaml.safe_load(effective.read_text())
    by_name = {m["name"]: m for m in data["members"]}
    # llm_only must not appear in the rewritten roster at all.
    assert "llm_only" not in by_name, f"llm_only still in roster: {list(by_name)}"
    # Non-LLM members survive.
    for non_llm in {"voting_only", "prefer_higher_trust_only"}:
        assert non_llm in by_name, f"{non_llm} unexpectedly removed"
    # Original yaml unchanged.
    src_data = yaml.safe_load(src_yaml.read_text())
    assert "llm_only" in {m["name"] for m in src_data["members"]}


def test_fusion_yaml_filter_with_llm_is_passthrough(tmp_path: Path) -> None:
    pipe = _make_pipeline(tmp_path, with_llm=True)
    src_yaml = COMMITTEE_DIR / "fusion_committee_products.yaml"
    assert pipe._maybe_filter_fusion_yaml(src_yaml) == src_yaml


def test_blocking_yaml_filter_drops_sc_block_without_override(
    tmp_path: Path,
) -> None:
    """Without --sc-block-checkpoint-override, sc_block is LOUDLY
    disabled when the YAML default lives under usecases_synthetic/cache/.
    """
    pipe = _make_pipeline(tmp_path)
    src_yaml = COMMITTEE_DIR / "em_blocking_committee_products.yaml"
    effective = pipe._maybe_filter_blocking_yaml(src_yaml)
    data = yaml.safe_load(effective.read_text())
    by_name = {m["name"]: m for m in data["members"]}
    if "sc_block" in by_name:  # sc_block is in the products roster
        assert by_name["sc_block"]["enabled_by_default"] is False


def test_blocking_yaml_filter_sc_block_checkpoint_override_keeps_member(
    tmp_path: Path,
) -> None:
    """With --sc-block-checkpoint-override, sc_block stays enabled and
    the checkpoint path is rewritten to the pipeline-isolated location.
    """
    fake_ckpt = tmp_path / "my_pipeline_scblock"
    fake_ckpt.mkdir()
    pipe = _make_pipeline(tmp_path, sc_block_checkpoint_override=fake_ckpt)
    src_yaml = COMMITTEE_DIR / "em_blocking_committee_products.yaml"
    effective = pipe._maybe_filter_blocking_yaml(src_yaml)
    data = yaml.safe_load(effective.read_text())
    by_name = {m["name"]: m for m in data["members"]}
    if "sc_block" in by_name:
        assert by_name["sc_block"].get("enabled_by_default", True) is True
        assert by_name["sc_block"]["blocker"]["params"]["checkpoint_path"] == str(
            fake_ckpt
        )
    # Other blockers untouched.
    for name in {"token_blocker", "standard_blocker", "embedding_blocker"}:
        if name in by_name:
            assert by_name[name].get("enabled_by_default", True) is True


def test_selection_from_sweep_bridge() -> None:
    """The bridge converts a StageSweepResult into a usable StageSelection."""
    from pipelines.lib.pipeline import _selection_from_sweep
    from pipelines.lib.sweep_harness import HpCell, StageSweepResult

    sweep = StageSweepResult(
        stage="sm",
        cells=[
            HpCell(
                member="member_a",
                hp={"init.x": 1},
                hp_id="abc123",
                val_score=0.9,
                test_score=0.85,
                metric_key="f1",
                runtime_s=1.0,
            ),
            HpCell(
                member="member_b",
                hp={"init.x": 2},
                hp_id="def456",
                val_score=0.7,
                test_score=0.65,
                metric_key="f1",
                runtime_s=1.0,
            ),
        ],
        per_member_winner={},
        cross_member_winner="member_a",
        cross_member_winner_val_score=0.9,
        cross_member_winner_test_score=0.85,
        metric_key="f1",
        runtime_s=2.0,
    )
    # Manually populate per_member_winner before calling bridge.
    sweep.per_member_winner = {
        "member_a": sweep.cells[0],
        "member_b": sweep.cells[1],
    }
    sel = _selection_from_sweep(sweep)
    assert sel.stage == "sm"
    assert sel.winner == "member_a"
    assert sel.val_score == 0.9
    assert sel.test_score == 0.85
    assert sel.per_member_val["member_b"] == 0.7
    assert sel.notes["from_sweep"] is True


def test_sweep_out_dir_resolution(tmp_path: Path) -> None:
    pipe = _make_pipeline(tmp_path)
    assert pipe._sweep_out_dir("sm") == tmp_path / "sweeps" / "sm"
    # When out_dir is None, returns None.
    from pipelines.lib.pipeline import BestOfBreedPipeline, PipelineConfig

    cfg = PipelineConfig.from_yaml(
        REPO_ROOT / "pipelines" / "configs" / "products.yaml"
    )
    pipe_no_out = BestOfBreedPipeline(cfg, committee_dir=COMMITTEE_DIR)
    assert pipe_no_out._sweep_out_dir("sm") is None
