"""Tests for the em_blocking ``StageSelection`` builder.

The pipeline now reports ``pair_completeness`` (recall) as the primary
metric for the blocking stage, with ``reduction_ratio`` retained as a
secondary side metric. These tests pin that contract on the pure
helper that assembles the ``StageSelection``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pipelines.lib.stage_runners import _build_em_blocking_selection


@dataclass
class _FakeMemberResult:
    """Minimal stand-in for :class:`usecases_synthetic.lib.committee.MemberResult`.

    Only the fields read by ``_build_em_blocking_selection`` are
    populated.
    """

    name: str
    metrics: dict[str, float]
    notes: dict[str, Any] = field(default_factory=dict)


def _make_per_blocker() -> dict[str, _FakeMemberResult]:
    """Three blockers with hand-picked metrics + per-pair selections.

    - ``standard``: selected on both pairs, mid recall, high RR.
    - ``token``  : selected on neither pair, highest recall, low RR.
    - ``sorted`` : selected on no pairs, low recall, low RR.

    The per-pair tally picks ``standard`` as the stage-level winner;
    the per-member maps still surface recall + RR for the others.
    """
    return {
        "standard": _FakeMemberResult(
            name="standard",
            metrics={"pair_completeness": 0.985, "reduction_ratio": 0.9309},
            notes={
                "per_pair": {
                    "p1_p2": {"selected": True},
                    "p1_p3": {"selected": True},
                    "p1_p4": {"selected": False},
                }
            },
        ),
        "token": _FakeMemberResult(
            name="token",
            metrics={"pair_completeness": 0.997, "reduction_ratio": 0.5102},
            notes={"per_pair": {"p1_p2": {"selected": False}}},
        ),
        "sorted": _FakeMemberResult(
            name="sorted",
            metrics={"pair_completeness": 0.840, "reduction_ratio": 0.8800},
            notes={
                "per_pair": {
                    "p1_p4": {"selected": True},
                }
            },
        ),
    }


def test_em_blocking_selection_primary_metric_is_recall() -> None:
    """Primary metric_key + val_score / test_score reflect recall."""
    per_blocker = _make_per_blocker()
    sel, per_pair = _build_em_blocking_selection(
        per_blocker=per_blocker,
        runtime_s=1.23,
        val_split_available=True,
        test_split_available=False,
    )

    # Primary metric is recall (pair_completeness), tiebreak by RR.
    assert "pair_completeness" in sel.metric_key
    assert "reduction_ratio" in sel.metric_key

    # 'standard' is selected on 2 pairs vs 'sorted' on 1, so it wins
    # the per-pair tally even though 'token' has higher recall.
    assert sel.winner == "standard"
    assert sel.val_score == 0.985
    # No separate test pass; test mirrors val.
    assert sel.test_score == 0.985

    # per_pair winner map matches the 'selected' notes above.
    assert per_pair == {"p1_p2": "standard", "p1_p3": "standard", "p1_p4": "sorted"}


def test_em_blocking_selection_per_member_maps_are_recall() -> None:
    """per_member_val / per_member_test carry the recall values."""
    per_blocker = _make_per_blocker()
    sel, _ = _build_em_blocking_selection(
        per_blocker=per_blocker,
        runtime_s=0.0,
        val_split_available=True,
        test_split_available=False,
    )
    assert sel.per_member_val == {
        "standard": 0.985,
        "token": 0.997,
        "sorted": 0.840,
    }
    # Test mirrors val.
    assert sel.per_member_test == sel.per_member_val


def test_em_blocking_selection_secondary_rr_in_notes() -> None:
    """reduction_ratio is exposed as a secondary side metric in notes."""
    per_blocker = _make_per_blocker()
    sel, _ = _build_em_blocking_selection(
        per_blocker=per_blocker,
        runtime_s=0.0,
        val_split_available=True,
        test_split_available=False,
    )
    rr_val = sel.notes["per_member_reduction_ratio_val"]
    rr_test = sel.notes["per_member_reduction_ratio_test"]
    # Winner's RR is preserved under the side-metric channel.
    assert rr_val["standard"] == 0.9309
    assert rr_test["standard"] == 0.9309
    # All blockers represented.
    assert set(rr_val.keys()) == {"standard", "token", "sorted"}
    assert rr_val == rr_test


def test_em_blocking_selection_recall_floor_recorded() -> None:
    """The 0.97 recall floor is surfaced in notes."""
    per_blocker = _make_per_blocker()
    sel, _ = _build_em_blocking_selection(
        per_blocker=per_blocker,
        runtime_s=0.0,
        val_split_available=True,
        test_split_available=False,
    )
    assert sel.notes["recall_floor"] == 0.97


def test_em_blocking_selection_strategy_string_mentions_recall_floor() -> None:
    """selection_strategy string mentions both metrics explicitly."""
    per_blocker = _make_per_blocker()
    sel, _ = _build_em_blocking_selection(
        per_blocker=per_blocker,
        runtime_s=0.0,
        val_split_available=True,
        test_split_available=False,
    )
    strategy = sel.notes["selection_strategy"]
    assert "pair_completeness" in strategy
    assert "reduction_ratio" in strategy


def test_em_blocking_selection_fallback_uses_recall_when_no_per_pair() -> None:
    """When no blocker has any selected per-pair entry, fall back to
    the recall ranking — the highest-recall blocker wins."""
    per_blocker = {
        "standard": _FakeMemberResult(
            name="standard",
            metrics={"pair_completeness": 0.985, "reduction_ratio": 0.9309},
            notes={"per_pair": {}},
        ),
        "token": _FakeMemberResult(
            name="token",
            metrics={"pair_completeness": 0.997, "reduction_ratio": 0.5102},
            notes={"per_pair": {}},
        ),
    }
    sel, per_pair = _build_em_blocking_selection(
        per_blocker=per_blocker,
        runtime_s=0.0,
        val_split_available=False,
        test_split_available=False,
    )
    assert per_pair == {}
    # 'token' has the higher recall; the fallback ranks by recall.
    assert sel.winner == "token"
    assert sel.val_score == 0.997


# ---------------------------------------------------------------------------
# EM test-gold surface selection (2026-06-05 directive): variants score on the
# variant-aligned corner-filled test; base scores on the standard test split.
# ---------------------------------------------------------------------------
import pandas as pd  # noqa: E402

from pipelines.lib.stage_runners import _em_test_gold_for  # noqa: E402


@dataclass
class _FakeBundle:
    level: str
    em_splits: dict
    em_gold_regenerated: dict


def _gold(tag: str) -> pd.DataFrame:
    return pd.DataFrame({"id1": [f"{tag}_a"], "id2": [f"{tag}_b"], "label": ["true"]})


def test_base_uses_standard_test_split() -> None:
    pair = ("p1", "p2")
    b = _FakeBundle(
        level="baseline",
        em_splits={pair: {"test": _gold("std")}},
        em_gold_regenerated={},
    )
    out = _em_test_gold_for(b, pair)
    assert out is not None and out.iloc[0]["id1"] == "std_a"


def test_variant_uses_corner_filled_test() -> None:
    pair = ("p1", "p2")
    b = _FakeBundle(
        level="hard",
        em_splits={pair: {"test": _gold("std")}},
        em_gold_regenerated={pair: {"test": {"corner_filled": _gold("corner")}}},
    )
    out = _em_test_gold_for(b, pair)
    assert out is not None and out.iloc[0]["id1"] == "corner_a"


def test_variant_falls_back_to_standard_when_no_corner_filled() -> None:
    pair = ("p1", "p2")
    b = _FakeBundle(
        level="easy",
        em_splits={pair: {"test": _gold("std")}},
        em_gold_regenerated={pair: {"test": {}}},  # no corner_filled
    )
    out = _em_test_gold_for(b, pair)
    assert out is not None and out.iloc[0]["id1"] == "std_a"
