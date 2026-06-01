"""Tests for ``build_buckets`` after the 2026-05-26 policy tightening.

Per plan_revision_step4g_findings.md follow-up: the LLM adjudicator
now arbitrates **every** bucket-C disagreement. The legacy
``score >= theta + delta`` auto-include and ``score < theta - delta``
auto-drop paths were removed because Ditto's per-domain precision on
raw data is too low to overrule the human-baseline matcher on
confidence alone.
"""

from __future__ import annotations

from typing import Callable

import pytest

from usecases_synthetic.lib.pool_builder import (
    BucketResult,
    build_buckets,
)


def _adjudicator_yes(_pair: tuple[str, str]) -> bool:
    """Stub LLM adjudicator that always says yes."""
    return True


def _adjudicator_no(_pair: tuple[str, str]) -> bool:
    """Stub LLM adjudicator that always says no."""
    return False


def _counting_adjudicator() -> (
    tuple[Callable[[tuple[str, str]], bool], list[tuple[str, str]]]
):
    """Stub LLM adjudicator that always says yes and logs every call."""
    calls: list[tuple[str, str]] = []

    def adjudicate(pair: tuple[str, str]) -> bool:
        calls.append(pair)
        return True

    return adjudicate, calls


class TestBucketABRouting:
    """Bucket A (gold) and bucket B (agreement) routing — unchanged by policy."""

    def test_gold_positive_lands_in_bucket_a(self) -> None:
        result = build_buckets(
            gold_positives={("a1", "b1")},
            human_pairs=set(),
            ditto_scores={},
            delta=0.1,
            adjudicator=_adjudicator_no,
        )

        assert result.bucket_a == 1
        assert result.bucket_b == 0
        assert result.bucket_c_total == 0
        assert len(result.pool_df) == 1
        assert result.pool_df.iloc[0]["decision_path"] == "gold"
        assert result.pool_df.iloc[0]["score"] == 1.0

    def test_human_and_ditto_agree_lands_in_bucket_b(self) -> None:
        result = build_buckets(
            gold_positives=set(),
            human_pairs={("a1", "b1")},
            ditto_scores={("a1", "b1"): 0.8},
            delta=0.1,
            adjudicator=_adjudicator_no,
        )

        assert result.bucket_b == 1
        assert result.bucket_c_total == 0
        assert result.pool_df.iloc[0]["decision_path"] == "agreement"


class TestBucketCAlwaysGoesThroughLLM:
    """Every bucket-C disagreement is adjudicated by the LLM, regardless of Ditto score."""

    def test_ditto_only_far_above_threshold_still_calls_adjudicator(self) -> None:
        """Pre-policy this auto-included (plm_check_confident_pos); post-policy it must call LLM."""
        adjudicate, calls = _counting_adjudicator()

        result = build_buckets(
            gold_positives=set(),
            human_pairs=set(),
            ditto_scores={("a1", "b1"): 0.95},  # well above 0.5 + 0.1
            delta=0.1,
            adjudicator=adjudicate,
        )

        assert calls == [
            ("a1", "b1")
        ], "adjudicator must be called for Ditto-only positive"
        assert result.bucket_c_total == 1
        assert result.bucket_c_kept_llm == 1
        assert (
            result.bucket_c_kept_confident == 0
        ), "legacy auto-include must be retired"
        assert result.pool_df.iloc[0]["decision_path"] == "plm_check_llm_yes"

    def test_human_only_with_low_ditto_still_calls_adjudicator(self) -> None:
        """Pre-policy this auto-dropped (plm_check_confident_neg); post-policy it must call LLM."""
        adjudicate, calls = _counting_adjudicator()

        result = build_buckets(
            gold_positives=set(),
            human_pairs={("a1", "b1")},
            ditto_scores={("a1", "b1"): 0.05},  # well below 0.5 - 0.1
            delta=0.1,
            adjudicator=adjudicate,
        )

        # Adjudicator says yes → kept via LLM path (not auto-dropped).
        assert calls == [
            ("a1", "b1")
        ], "adjudicator must be called for human-only positive"
        assert result.bucket_c_total == 1
        assert result.bucket_c_kept_llm == 1
        assert (
            result.bucket_c_dropped_confident == 0
        ), "legacy auto-drop must be retired"
        assert result.pool_df.iloc[0]["decision_path"] == "plm_check_llm_yes"

    def test_llm_says_no_drops_pair_via_llm_path_not_confident(self) -> None:
        result = build_buckets(
            gold_positives=set(),
            human_pairs={("a1", "b1")},  # human says yes
            ditto_scores={("a1", "b1"): 0.05},  # Ditto says no, confidently
            delta=0.1,
            adjudicator=_adjudicator_no,
        )

        # Pair was dropped — but via the LLM gate, not the score shortcut.
        assert result.bucket_c_total == 1
        assert result.bucket_c_dropped_llm == 1
        assert result.bucket_c_dropped_confident == 0
        assert result.bucket_c_kept_llm == 0
        assert result.bucket_c_kept_confident == 0
        assert len(result.pool_df) == 0

    def test_legacy_confident_counters_always_zero(self) -> None:
        """Synthetic stress: every disagreement type, verify _confident
        counters stay zero regardless of Ditto score.

        - (a1, b1): human=yes, Ditto=0.99 → agreement (bucket B, no LLM)
        - (a2, b2): human=yes, Ditto=0.01 → was confident_neg → now LLM
        - (a3, b3): human=no, Ditto=0.55 → Ditto-only marginal → LLM
        - (a4, b4): human=no, Ditto=0.95 → was confident_pos → now LLM
        """
        ditto_scores = {
            ("a1", "b1"): 0.99,
            ("a2", "b2"): 0.01,
            ("a3", "b3"): 0.55,
            ("a4", "b4"): 0.95,
        }
        adjudicate, calls = _counting_adjudicator()

        result = build_buckets(
            gold_positives=set(),
            human_pairs={("a1", "b1"), ("a2", "b2")},
            ditto_scores=ditto_scores,
            delta=0.1,
            adjudicator=adjudicate,
        )

        assert result.bucket_b == 1, "(a1, b1) is full agreement"
        assert result.bucket_c_total == 3, "three disagreements must hit bucket C"
        assert len(calls) == 3, "every disagreement must call the LLM"
        assert result.bucket_c_kept_confident == 0
        assert result.bucket_c_dropped_confident == 0
        assert result.bucket_c_kept_llm + result.bucket_c_dropped_llm == 3


class TestBucketCAdjudicatorNoneVerdict:
    """Adjudicator returning None (cache miss with no api_client) drops the pair."""

    def test_adjudicator_returns_none_drops_pair(self) -> None:
        def _none_adj(_pair: tuple[str, str]) -> bool | None:
            return None

        result = build_buckets(
            gold_positives=set(),
            human_pairs={("a1", "b1")},
            ditto_scores={("a1", "b1"): 0.05},
            delta=0.1,
            adjudicator=_none_adj,
        )

        # None verdict treated as conservative drop.
        assert result.bucket_c_total == 1
        assert result.bucket_c_dropped_llm == 1
        assert len(result.pool_df) == 0


class TestDeltaIsTelemetryOnly:
    """``delta`` is preserved on the result but no longer used for routing."""

    def test_delta_recorded_on_result(self) -> None:
        result = build_buckets(
            gold_positives=set(),
            human_pairs=set(),
            ditto_scores={},
            delta=0.17,
            adjudicator=_adjudicator_no,
        )
        assert result.delta_used == 0.17

    @pytest.mark.parametrize("delta", [0.0, 0.05, 0.1, 0.2, 0.5])
    def test_delta_value_does_not_change_routing(self, delta: float) -> None:
        """Same inputs across a delta sweep must produce identical bucket
        outcomes — proves delta is no longer load-bearing."""
        adjudicate, calls = _counting_adjudicator()
        result = build_buckets(
            gold_positives=set(),
            human_pairs={("a1", "b1")},
            ditto_scores={("a1", "b1"): 0.99},  # human + Ditto agree
            delta=delta,
            adjudicator=adjudicate,
        )

        # All deltas → same bucket-B classification, no LLM call.
        assert result.bucket_b == 1
        assert result.bucket_c_total == 0
        assert len(calls) == 0


class TestEmptyInputs:
    def test_no_pairs_returns_empty_result(self) -> None:
        result = build_buckets(
            gold_positives=set(),
            human_pairs=set(),
            ditto_scores={},
            delta=0.1,
            adjudicator=_adjudicator_no,
        )
        assert isinstance(result, BucketResult)
        assert len(result.pool_df) == 0
        assert result.bucket_a == result.bucket_b == result.bucket_c_total == 0
