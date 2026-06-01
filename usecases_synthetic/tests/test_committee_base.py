"""Smoke tests for ``usecases_synthetic.lib.committee``.

The concrete stage runners land in M2/M3/M4; this file only exercises
the ABC plumbing so the interface is pinned.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from usecases_synthetic.lib.committee import (
    CommitteeResult,
    CommitteeRunner,
    MemberResult,
)
from usecases_synthetic.lib.variant_loader import VariantBundle


class _DummyMember:
    def __init__(self, name: str) -> None:
        self.name = name


class _DummyRunner(CommitteeRunner):
    stage = "sm"

    def run(self, bundle: VariantBundle) -> CommitteeResult:
        members = {
            member.name: MemberResult(
                name=member.name,
                predictions=None,
                metrics={"f1": 0.5},
            )
            for member in self.roster
        }
        return CommitteeResult(
            stage=self.stage,
            domain=bundle.domain,
            level=bundle.level,
            per_member=members,
            aggregated={"f1": 0.5},
            per_attribute={"name": {"f1": 0.5}},
            per_partition={"part1": {"f1": 0.5}},
            roster=list(members),
            runtime_s=0.1,
        )


def _make_bundle() -> VariantBundle:
    return VariantBundle(
        domain="companies",
        level="baseline",
        sources={},
        target_schema={},
        sm_mapping=None,
        em_gold={},
        em_splits={},
        fusion_gold=pd.DataFrame(),
        fusion_validation=None,
        pooled_positives=None,
        variant_root=__import__("pathlib").Path("/tmp"),
    )


def test_runner_roster_names_from_member_attribute() -> None:
    runner = _DummyRunner(
        roster=[_DummyMember("alpha"), _DummyMember("beta")],
        config={"threshold": 0.8},
    )
    assert runner.roster_names == ["alpha", "beta"]
    assert runner.config == {"threshold": 0.8}


def test_runner_roster_names_fallback() -> None:
    class _Nameless:
        pass

    runner = _DummyRunner(roster=[_Nameless(), _Nameless()])
    assert runner.roster_names == ["_Nameless_0", "_Nameless_1"]


def test_committee_result_as_dict_drops_predictions() -> None:
    runner = _DummyRunner(
        roster=[_DummyMember("alpha"), _DummyMember("beta")],
    )
    result = runner.run(_make_bundle())
    assert isinstance(result, CommitteeResult)

    payload: dict[str, Any] = result.as_dict()
    assert payload["stage"] == "sm"
    assert payload["domain"] == "companies"
    assert payload["level"] == "baseline"
    assert payload["aggregated"] == {"f1": 0.5}
    assert payload["per_attribute"] == {"name": {"f1": 0.5}}
    assert payload["per_partition"] == {"part1": {"f1": 0.5}}
    assert set(payload["per_member"]) == {"alpha", "beta"}
    # Predictions must not leak into the serialisable snapshot.
    for member_payload in payload["per_member"].values():
        assert "predictions" not in member_payload
        assert member_payload["metrics"] == {"f1": 0.5}
