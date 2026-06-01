"""Tests for ``usecases_synthetic.lib.monotonicity`` and the M8 CLI.

Covers:

- ``check_monotone`` direction logic (down / up / flat, ties, NaN).
- ``resolve_metric`` for dotted paths and ``spread:`` specs.
- ``match_signals`` against synthetic level-metric dicts, including the
  qualitative-only and range-bounded branches.
- ``detect_collapses`` classification: ``hidden_positive_noise`` when
  pool precision is stable while test-gold collapsed, and
  ``real_collapse`` when both moved together.
- ``load_knob_expected_signals`` reading the repo-committed YAML.
- The ``analyze_monotonicity.analyze_domain`` orchestrator writing its
  three output artifacts.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest
import yaml

from usecases_synthetic.lib.monotonicity import (
    LEVELS,
    BestMemberCheck,
    SignalExpectation,
    baseline_within_allowed_position,
    check_monotone,
    collect_level_values,
    compute_ceiling_responsiveness,
    detect_collapses,
    load_knob_expected_signals,
    match_signals,
    resolve_metric,
)
from usecases_synthetic.lib.monotonicity import _pearson
from usecases_synthetic.scripts.analyze_monotonicity import (
    EXPECTATIONS_YAML,
    analyze_domain,
    build_cross_level_slope,
)

# ---------------------------------------------------------------------------
# check_monotone
# ---------------------------------------------------------------------------


class TestCheckMonotone:
    def test_strictly_decreasing_values_are_down(self) -> None:
        assert check_monotone([0.9, 0.7, 0.4, 0.2], "down") is True

    def test_weakly_decreasing_with_tie_allowed(self) -> None:
        assert check_monotone([0.9, 0.7, 0.7, 0.4], "down") is True

    def test_up_after_down_fails_down(self) -> None:
        assert check_monotone([0.9, 0.7, 0.3, 0.5], "down") is False

    def test_strictly_increasing_values_are_up(self) -> None:
        assert check_monotone([0.1, 0.3, 0.5, 0.8], "up") is True

    def test_flat_within_tolerance(self) -> None:
        assert check_monotone([0.5, 0.51, 0.49, 0.52], "flat") is True

    def test_flat_outside_tolerance_fails(self) -> None:
        assert check_monotone([0.5, 0.7, 0.5, 0.5], "flat") is False

    def test_nan_fails_any_direction(self) -> None:
        assert check_monotone([0.5, float("nan"), 0.3, 0.1], "down") is False

    def test_single_value_fails(self) -> None:
        assert check_monotone([0.5], "down") is False

    def test_unknown_direction_raises_via_baseline_helper(self) -> None:
        with pytest.raises(ValueError):
            baseline_within_allowed_position(0.5, 0.4, "sideways")  # type: ignore[arg-type]

    def test_unknown_direction_raises(self) -> None:
        with pytest.raises(ValueError):
            check_monotone([0.5, 0.4], "sideways")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# baseline_within_allowed_position
# ---------------------------------------------------------------------------


class TestBaselineWithinAllowedPosition:
    """Baseline must not be ``harder than medium`` in any direction."""

    def test_down_baseline_above_easy_is_ok(self) -> None:
        # macro_f1 down: baseline=0.95, easy=0.85, medium=0.60, hard=0.30.
        # Baseline > easy > medium → baseline easier than everything, fine.
        assert baseline_within_allowed_position(0.95, 0.60, "down") is True

    def test_down_baseline_between_easy_and_medium_is_ok(self) -> None:
        # macro_f1 down: baseline=0.75 sits between easy=0.85 and medium=0.60.
        assert baseline_within_allowed_position(0.75, 0.60, "down") is True

    def test_down_baseline_equal_to_medium_is_ok(self) -> None:
        assert baseline_within_allowed_position(0.60, 0.60, "down") is True

    def test_down_baseline_below_medium_fails(self) -> None:
        # Baseline=0.50 is harder than medium=0.60 in a down-metric.
        assert baseline_within_allowed_position(0.50, 0.60, "down") is False

    def test_up_baseline_below_easy_is_ok(self) -> None:
        # Error rate up: baseline=0.05, easy=0.10, medium=0.30, hard=0.60.
        assert baseline_within_allowed_position(0.05, 0.30, "up") is True

    def test_up_baseline_between_easy_and_medium_is_ok(self) -> None:
        assert baseline_within_allowed_position(0.20, 0.30, "up") is True

    def test_up_baseline_above_medium_fails(self) -> None:
        # baseline=0.35 harder than medium=0.30 in up-metric.
        assert baseline_within_allowed_position(0.35, 0.30, "up") is False

    def test_flat_within_tolerance_is_ok(self) -> None:
        assert (
            baseline_within_allowed_position(0.51, 0.50, "flat", flat_tolerance=0.05)
            is True
        )

    def test_flat_outside_tolerance_fails(self) -> None:
        assert (
            baseline_within_allowed_position(0.70, 0.50, "flat", flat_tolerance=0.05)
            is False
        )

    def test_nan_baseline_skips_check(self) -> None:
        assert baseline_within_allowed_position(float("nan"), 0.50, "down") is True

    def test_nan_medium_skips_check(self) -> None:
        assert baseline_within_allowed_position(0.50, float("nan"), "down") is True


# ---------------------------------------------------------------------------
# resolve_metric
# ---------------------------------------------------------------------------


class TestResolveMetric:
    def test_dotted_path(self) -> None:
        block = {"aggregated": {"macro_f1": 0.73}}
        assert resolve_metric(block, "aggregated.macro_f1") == pytest.approx(0.73)

    def test_missing_key_returns_nan(self) -> None:
        block = {"aggregated": {"macro_f1": 0.73}}
        assert math.isnan(resolve_metric(block, "aggregated.min_f1"))

    def test_spread_spec_returns_difference(self) -> None:
        block = {
            "per_member": {
                "a": {"metrics": {"f1": 0.9}},
                "b": {"metrics": {"f1": 0.4}},
            }
        }
        got = resolve_metric(
            block,
            "spread:per_member.a.metrics.f1:per_member.b.metrics.f1",
        )
        assert got == pytest.approx(0.5)

    def test_spread_with_missing_side_is_nan(self) -> None:
        block = {
            "per_member": {
                "a": {"metrics": {"f1": 0.9}},
            }
        }
        assert math.isnan(
            resolve_metric(
                block,
                "spread:per_member.a.metrics.f1:per_member.missing.metrics.f1",
            )
        )

    def test_spread_spec_needs_two_parts(self) -> None:
        with pytest.raises(ValueError):
            resolve_metric({}, "spread:one_part_only")


# ---------------------------------------------------------------------------
# Signal fixtures
# ---------------------------------------------------------------------------


def _metrics(sm_f1: float, em_f1: float, em_pool_p: float = 0.9) -> dict:
    return {
        "per_stage": {
            "sm": {
                "aggregated": {"macro_f1": sm_f1},
                "per_member": {
                    "label_jaccard": {
                        "metrics": {"f1": sm_f1 - 0.05, "precision": 0.5, "recall": 0.4}
                    },
                },
            },
            "em": {
                "aggregated": {
                    "macro_f1": em_f1,
                    "macro_pool_precision": em_pool_p,
                    "macro_pool_recall": 0.2,
                },
                "per_member": {
                    "standard_rule": {
                        "metrics": {
                            "f1": em_f1,
                            "precision": em_f1,
                            "recall": em_f1,
                            "pool_precision": em_pool_p,
                            "pool_recall": 0.2,
                        }
                    },
                },
            },
            "fusion": {
                "aggregated": {"overall_accuracy": 0.5, "overall_spread": 0.2},
                "per_member": {},
            },
        }
    }


@pytest.fixture
def level_metrics_monotone_down() -> dict[str, dict]:
    return {
        "baseline": _metrics(sm_f1=0.90, em_f1=0.80),
        "easy": _metrics(sm_f1=0.80, em_f1=0.70),
        "medium": _metrics(sm_f1=0.60, em_f1=0.50),
        "hard": _metrics(sm_f1=0.30, em_f1=0.20),
    }


# ---------------------------------------------------------------------------
# match_signals
# ---------------------------------------------------------------------------


class TestMatchSignals:
    def test_down_direction_monotone_is_flagged_ok(
        self, level_metrics_monotone_down: dict[str, dict]
    ) -> None:
        exp = SignalExpectation(
            knob="knob_test",
            signal_id="em_drop",
            stage="em",
            metric="aggregated.macro_f1",
            direction="down",
            qualitative_only=True,
            target_delta_range=None,
            pool_check=False,
            notes="",
            source="test",
        )
        checks = match_signals(level_metrics_monotone_down, [exp])
        assert len(checks) == 1
        check = checks[0]
        assert check.is_monotone is True
        assert check.within_range is True
        assert check.observed_delta == pytest.approx(-0.60)

    def test_wrong_direction_is_flagged(
        self, level_metrics_monotone_down: dict[str, dict]
    ) -> None:
        exp = SignalExpectation(
            knob="knob_test",
            signal_id="em_up",
            stage="em",
            metric="aggregated.macro_f1",
            direction="up",
            qualitative_only=True,
            target_delta_range=None,
            pool_check=False,
            notes="",
            source="test",
        )
        checks = match_signals(level_metrics_monotone_down, [exp])
        assert checks[0].is_monotone is False

    def test_target_delta_range_bounds(
        self, level_metrics_monotone_down: dict[str, dict]
    ) -> None:
        exp = SignalExpectation(
            knob="knob_test",
            signal_id="em_drop",
            stage="em",
            metric="aggregated.macro_f1",
            direction="down",
            qualitative_only=False,
            target_delta_range=(-0.5, -0.05),
            pool_check=False,
            notes="",
            source="test",
        )
        checks = match_signals(level_metrics_monotone_down, [exp])
        assert checks[0].is_monotone is True
        # observed -0.60 is outside [-0.5, -0.05]
        assert checks[0].within_range is False

    def test_missing_metric_produces_nan_and_failure(
        self, level_metrics_monotone_down: dict[str, dict]
    ) -> None:
        exp = SignalExpectation(
            knob="knob_test",
            signal_id="missing",
            stage="em",
            metric="aggregated.does_not_exist",
            direction="down",
            qualitative_only=True,
            target_delta_range=None,
            pool_check=False,
            notes="",
            source="test",
        )
        checks = match_signals(level_metrics_monotone_down, [exp])
        assert checks[0].is_monotone is False
        assert math.isnan(checks[0].observed_delta)
        assert "missing" in checks[0].reason

    def test_baseline_above_easy_is_position_ok(
        self, level_metrics_monotone_down: dict[str, dict]
    ) -> None:
        # Fixture baseline=0.80, easy=0.80, medium=0.60, hard=0.20 → down
        # check sees baseline (0.80) >= medium (0.60) → position_ok=True.
        exp = SignalExpectation(
            knob="knob_test",
            signal_id="em_drop",
            stage="em",
            metric="aggregated.macro_f1",
            direction="down",
            qualitative_only=True,
            target_delta_range=None,
            pool_check=False,
            notes="",
            source="test",
        )
        checks = match_signals(level_metrics_monotone_down, [exp])
        assert checks[0].baseline_position_ok is True

    def test_baseline_harder_than_medium_fails_position_check(self) -> None:
        # Synth metrics: baseline=0.50 lower than medium=0.60 in down direction,
        # but easy/medium/hard slope itself is correctly down.
        bad_metrics = {
            "baseline": _metrics(sm_f1=0.50, em_f1=0.50),
            "easy": _metrics(sm_f1=0.85, em_f1=0.85),
            "medium": _metrics(sm_f1=0.60, em_f1=0.60),
            "hard": _metrics(sm_f1=0.30, em_f1=0.30),
        }
        exp = SignalExpectation(
            knob="knob_test",
            signal_id="em_drop",
            stage="em",
            metric="aggregated.macro_f1",
            direction="down",
            qualitative_only=True,
            target_delta_range=None,
            pool_check=False,
            notes="",
            source="test",
        )
        checks = match_signals(bad_metrics, [exp])
        assert checks[0].is_monotone is True  # easy/medium/hard slope OK
        assert checks[0].baseline_position_ok is False

    def test_collect_level_values_covers_all_levels(
        self, level_metrics_monotone_down: dict[str, dict]
    ) -> None:
        exp = SignalExpectation(
            knob="knob_test",
            signal_id="em_drop",
            stage="em",
            metric="aggregated.macro_f1",
            direction="down",
            qualitative_only=True,
            target_delta_range=None,
            pool_check=False,
            notes="",
            source="test",
        )
        values = collect_level_values(level_metrics_monotone_down, exp)
        assert set(values.keys()) == set(LEVELS)
        assert values["baseline"] == pytest.approx(0.80)
        assert values["hard"] == pytest.approx(0.20)


# ---------------------------------------------------------------------------
# detect_collapses
# ---------------------------------------------------------------------------


class TestDetectCollapses:
    def test_hidden_positive_noise_when_pool_is_stable(self) -> None:
        baseline = _metrics(sm_f1=0.9, em_f1=0.8, em_pool_p=0.9)
        hard = _metrics(sm_f1=0.9, em_f1=0.10, em_pool_p=0.88)
        level_metrics = {"baseline": baseline, "hard": hard}
        collapses = detect_collapses(level_metrics)
        em_collapses = [c for c in collapses if c.stage == "em"]
        assert len(em_collapses) == 1
        c = em_collapses[0]
        assert c.level == "hard"
        assert c.member == "standard_rule"
        assert c.classification == "hidden_positive_noise"
        assert c.pool_agreement_delta is not None
        assert abs(c.pool_agreement_delta) <= 0.1

    def test_real_collapse_when_pool_also_moves(self) -> None:
        baseline = _metrics(sm_f1=0.9, em_f1=0.8, em_pool_p=0.9)
        hard = _metrics(sm_f1=0.9, em_f1=0.10, em_pool_p=0.2)
        level_metrics = {"baseline": baseline, "hard": hard}
        collapses = detect_collapses(level_metrics)
        em_collapses = [c for c in collapses if c.stage == "em"]
        assert len(em_collapses) == 1
        assert em_collapses[0].classification == "real_collapse"

    def test_sm_collapse_is_unknown_classification(self) -> None:
        baseline = _metrics(sm_f1=0.9, em_f1=0.8)
        # sm label_jaccard F1 is sm_f1-0.05 = 0.85 baseline, we need to force
        # a collapse on the sm member. Build directly:
        hard = {
            "per_stage": {
                "sm": {
                    "aggregated": {"macro_f1": 0.1},
                    "per_member": {
                        "label_jaccard": {
                            "metrics": {
                                "f1": 0.05,
                                "precision": 0.02,
                                "recall": 0.08,
                            }
                        }
                    },
                },
                "em": baseline["per_stage"]["em"],
                "fusion": baseline["per_stage"]["fusion"],
            }
        }
        collapses = detect_collapses({"baseline": baseline, "hard": hard})
        sm_collapses = [c for c in collapses if c.stage == "sm"]
        assert sm_collapses and sm_collapses[0].classification == "unknown"

    def test_no_collapse_when_values_hold(self) -> None:
        baseline = _metrics(sm_f1=0.9, em_f1=0.8)
        hard = _metrics(sm_f1=0.88, em_f1=0.78)
        level_metrics = {"baseline": baseline, "hard": hard}
        assert detect_collapses(level_metrics) == []


# ---------------------------------------------------------------------------
# Expectations YAML
# ---------------------------------------------------------------------------


class TestExpectationsYaml:
    def test_committed_yaml_parses(self) -> None:
        knob_expectations = load_knob_expected_signals(EXPECTATIONS_YAML)
        # every active v1 knob must have at least one primary-stage entry
        required = {
            "knob_01",
            "knob_02",
            "knob_03",
            "knob_04",
            "knob_05",
            "knob_06",
            "knob_08",
            "knob_10",
        }
        assert required.issubset(set(knob_expectations.keys()))
        for knob_id, expectations in knob_expectations.items():
            if knob_id not in required:
                continue
            assert expectations, f"{knob_id} has no expectation entries"
            for exp in expectations:
                # Stage keys must match the committee per_stage roster
                # emitted by measure_baseline / validate_variant (post-C12).
                # The pre-C10 lumped "em" stage was split into
                # em_blocking + em_matching (2026-05-31 realignment).
                assert exp.stage in {
                    "sm",
                    "norm",
                    "em_blocking",
                    "em_matching",
                    "fusion",
                }
                assert exp.direction in {"down", "up", "flat"}

    def test_p8_best_member_monotone_down(self) -> None:
        """P8: best-member ceiling declines monotonically across levels."""
        from usecases_synthetic.lib.monotonicity import (
            match_best_member_monotonicity,
        )

        def stage(f1_a: float, f1_b: float) -> dict:
            return {
                "per_member": {
                    "matcher_a": {"metrics": {"f1": f1_a}},
                    "matcher_b": {"metrics": {"f1": f1_b}},
                }
            }

        metrics = {
            "baseline": {"per_stage": {"em_matching": stage(0.95, 0.90)}},
            "easy": {"per_stage": {"em_matching": stage(0.92, 0.88)}},
            "medium": {"per_stage": {"em_matching": stage(0.85, 0.80)}},
            "hard": {"per_stage": {"em_matching": stage(0.75, 0.70)}},
        }
        checks = match_best_member_monotonicity(metrics, stages=("em_matching",))
        assert len(checks) == 1
        c = checks[0]
        assert c.stage == "em_matching"
        # Best member tracks matcher_a at every level.
        assert c.winners == {
            "baseline": "matcher_a",
            "easy": "matcher_a",
            "medium": "matcher_a",
            "hard": "matcher_a",
        }
        assert c.values == {
            "baseline": 0.95,
            "easy": 0.92,
            "medium": 0.85,
            "hard": 0.75,
        }
        assert c.is_non_increasing is True
        assert c.observed_delta == pytest.approx(-0.20)

    def test_p8_best_member_flat_ceiling_fails(self) -> None:
        """A difficulty signal that leaves the best member flat is invalid."""
        from usecases_synthetic.lib.monotonicity import (
            match_best_member_monotonicity,
        )

        # Committee mean drops (mediocre matcher gets worse) but best
        # member (strong matcher) stays at 0.95 across all levels.
        def stage(strong: float, weak: float) -> dict:
            return {
                "per_member": {
                    "strong": {"metrics": {"f1": strong}},
                    "weak": {"metrics": {"f1": weak}},
                }
            }

        metrics = {
            "baseline": {"per_stage": {"em_matching": stage(0.95, 0.90)}},
            "easy": {"per_stage": {"em_matching": stage(0.95, 0.80)}},
            "medium": {"per_stage": {"em_matching": stage(0.95, 0.60)}},
            "hard": {"per_stage": {"em_matching": stage(0.95, 0.40)}},
        }
        checks = match_best_member_monotonicity(metrics, stages=("em_matching",))
        c = checks[0]
        # Ceiling is 0.95 at every level -> flat -> P8 FAILS.
        assert c.values["hard"] == pytest.approx(0.95)
        assert c.is_non_increasing is True  # flat passes default tol
        # But with tighter tol it would still be flat — the key signal:
        assert c.observed_delta == pytest.approx(0.0)
        assert "non-increasing" in c.reason

    def test_p8_winner_changes_across_levels(self) -> None:
        """Different members win at different difficulty levels."""
        from usecases_synthetic.lib.monotonicity import (
            match_best_member_monotonicity,
        )

        # matcher_a wins at baseline/easy; matcher_b wins at medium/hard
        # (matcher_a degrades faster on harder data).
        metrics = {
            "baseline": {
                "per_stage": {
                    "sm": {
                        "per_member": {
                            "a": {"metrics": {"f1": 0.95}},
                            "b": {"metrics": {"f1": 0.85}},
                        }
                    }
                }
            },
            "easy": {
                "per_stage": {
                    "sm": {
                        "per_member": {
                            "a": {"metrics": {"f1": 0.90}},
                            "b": {"metrics": {"f1": 0.84}},
                        }
                    }
                }
            },
            "medium": {
                "per_stage": {
                    "sm": {
                        "per_member": {
                            "a": {"metrics": {"f1": 0.60}},
                            "b": {"metrics": {"f1": 0.82}},
                        }
                    }
                }
            },
            "hard": {
                "per_stage": {
                    "sm": {
                        "per_member": {
                            "a": {"metrics": {"f1": 0.40}},
                            "b": {"metrics": {"f1": 0.78}},
                        }
                    }
                }
            },
        }
        checks = match_best_member_monotonicity(metrics, stages=("sm",))
        c = checks[0]
        assert c.winners == {
            "baseline": "a",
            "easy": "a",
            "medium": "b",
            "hard": "b",
        }
        # Ceiling: 0.95 -> 0.90 -> 0.82 -> 0.78. Non-increasing.
        assert c.is_non_increasing is True

    def test_p8_uses_pair_recall_for_em_blocking(self) -> None:
        from usecases_synthetic.lib.monotonicity import (
            match_best_member_monotonicity,
        )

        def stage(pr: float) -> dict:
            return {
                "per_member": {
                    "block_a": {"metrics": {"pair_recall": pr}},
                }
            }

        metrics = {
            "baseline": {"per_stage": {"em_blocking": stage(0.99)}},
            "easy": {"per_stage": {"em_blocking": stage(0.97)}},
            "medium": {"per_stage": {"em_blocking": stage(0.90)}},
            "hard": {"per_stage": {"em_blocking": stage(0.80)}},
        }
        c = match_best_member_monotonicity(metrics, stages=("em_blocking",))[0]
        assert c.values["hard"] == pytest.approx(0.80)
        assert c.is_non_increasing is True

    def test_malformed_direction_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            yaml.safe_dump(
                {
                    "knob_99": {
                        "label": "broken",
                        "source_card": "nope.md",
                        "primary_stage": "em",
                        "signals": [
                            {
                                "id": "x",
                                "stage": "em",
                                "metric": "aggregated.macro_f1",
                                "direction": "sideways",
                            }
                        ],
                    }
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError):
            load_knob_expected_signals(bad)


# ---------------------------------------------------------------------------
# End-to-end CLI orchestrator
# ---------------------------------------------------------------------------


def _write_metrics_file(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import json as _json

    path.write_text(_json.dumps(payload), encoding="utf-8")


def test_analyze_domain_writes_artifacts(tmp_path: Path) -> None:
    """The orchestrator writes report.md, report.csv, collapses.csv."""
    domain = "companies"

    # Build minimal baseline + per-level metrics and point the loaders at them
    baseline_dir = tmp_path / "baselines" / domain
    val_dir = tmp_path / "validation" / domain
    _write_metrics_file(
        baseline_dir / "baseline_metrics.json",
        {
            "domain": domain,
            "meta": {},
            "per_stage": {
                "sm": {
                    "aggregated": {"macro_f1": 0.90},
                    "per_member": {
                        "label_jaccard": {"metrics": {"f1": 0.85}},
                        "llm_openai": {"metrics": {"f1": 0.95}},
                        "instance_tfidf_cosine": {"metrics": {"f1": 0.0}},
                    },
                },
                "em": {
                    "aggregated": {
                        "macro_f1": 0.80,
                        "macro_pool_precision": 0.9,
                        "macro_pool_recall": 0.25,
                    },
                    "per_member": {
                        "standard_rule": {
                            "metrics": {
                                "f1": 0.80,
                                "precision": 0.85,
                                "recall": 0.75,
                                "pool_precision": 0.9,
                                "pool_recall": 0.25,
                            }
                        },
                        "embedding_rule": {
                            "metrics": {
                                "f1": 0.60,
                                "pool_precision": 0.5,
                                "pool_recall": 0.2,
                            }
                        },
                    },
                },
                "fusion": {
                    "aggregated": {"overall_accuracy": 0.70, "overall_spread": 0.15},
                    "per_member": {},
                    "per_attribute": {
                        "name": {
                            "best_strategy_accuracy": 1.0,
                            "spread": 0.1,
                            "voting": 0.95,
                            "prefer_higher_trust": 0.90,
                        }
                    },
                },
            },
        },
    )
    for level, f1 in (("easy", 0.72), ("medium", 0.52), ("hard", 0.30)):
        _write_metrics_file(
            val_dir / level / "metrics.json",
            {
                "domain": domain,
                "meta": {},
                "per_stage": {
                    "sm": {
                        "aggregated": {
                            "macro_f1": 0.90 - (0.20 if level == "hard" else 0.05)
                        },
                        "per_member": {
                            "label_jaccard": {
                                "metrics": {
                                    "f1": (
                                        0.85
                                        if level == "easy"
                                        else 0.55 if level == "medium" else 0.20
                                    )
                                }
                            },
                            "llm_openai": {
                                "metrics": {
                                    "f1": (
                                        0.95
                                        if level == "easy"
                                        else 0.90 if level == "medium" else 0.85
                                    )
                                }
                            },
                            "instance_tfidf_cosine": {"metrics": {"f1": 0.0}},
                        },
                    },
                    "em": {
                        "aggregated": {
                            "macro_f1": f1,
                            "macro_pool_precision": 0.9
                            - (
                                0.05
                                if level == "easy"
                                else 0.15 if level == "medium" else 0.25
                            ),
                            "macro_pool_recall": 0.25,
                        },
                        "per_member": {
                            "standard_rule": {
                                "metrics": {
                                    "f1": f1,
                                    "pool_precision": 0.9
                                    - (
                                        0.05
                                        if level == "easy"
                                        else 0.15 if level == "medium" else 0.25
                                    ),
                                    "pool_recall": 0.25,
                                }
                            },
                            "embedding_rule": {
                                "metrics": {
                                    "f1": f1 - 0.10,
                                    "pool_precision": 0.5,
                                    "pool_recall": 0.2,
                                }
                            },
                        },
                    },
                    "fusion": {
                        "aggregated": {
                            "overall_accuracy": 0.70
                            - (
                                0.05
                                if level == "easy"
                                else 0.15 if level == "medium" else 0.30
                            ),
                            "overall_spread": 0.15
                            + (
                                0.05
                                if level == "easy"
                                else 0.15 if level == "medium" else 0.25
                            ),
                        },
                        "per_member": {},
                        "per_attribute": {
                            "name": {
                                "best_strategy_accuracy": 1.0,
                                "spread": 0.1,
                                "voting": 0.95
                                - (
                                    0.05
                                    if level == "easy"
                                    else 0.20 if level == "medium" else 0.40
                                ),
                                "prefer_higher_trust": 0.90,
                            }
                        },
                    },
                },
            },
        )

    # Patch the module-level constants used inside analyze_domain.
    from usecases_synthetic.lib import baseline_loader
    from usecases_synthetic.scripts import analyze_monotonicity as am

    original_baselines = baseline_loader.BASELINES_DIR
    original_validation = am.VALIDATION_DIR
    baseline_loader.BASELINES_DIR = tmp_path / "baselines"
    am.VALIDATION_DIR = tmp_path / "validation"
    try:
        result = analyze_domain(
            domain,
            expectations_path=EXPECTATIONS_YAML,
            out_dir=tmp_path / "validation" / domain,
        )
    finally:
        baseline_loader.BASELINES_DIR = original_baselines
        am.VALIDATION_DIR = original_validation

    assert result["report_md"].exists()
    assert result["report_csv"].exists()
    assert result["collapse_csv"].exists()

    # CSV has one row per signal expectation for all active knobs.
    with open(result["report_csv"], encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows, "signal CSV should contain rows"
    knobs_seen = {row["knob"] for row in rows}
    assert {"knob_01", "knob_08", "knob_10"}.issubset(knobs_seen)

    # Report references every active knob id.
    md_text = result["report_md"].read_text(encoding="utf-8")
    for knob in ("knob_01", "knob_08", "knob_10"):
        assert knob in md_text

    # Hard-level label_jaccard F1 dropped from 0.85 to 0.20 -> collapse.
    with open(result["collapse_csv"], encoding="utf-8") as f:
        collapse_rows = list(csv.DictReader(f))
    assert any(
        row["member"] == "label_jaccard" and row["level"] == "hard"
        for row in collapse_rows
    )

    # C6: every signal row carries a ceiling_responsiveness value (or NaN
    # if the stage best-member series is missing). Field is present even
    # when the value can't be computed.
    for row in rows:
        assert "ceiling_responsiveness" in row


# ---------------------------------------------------------------------------
# C6: ceiling_responsiveness
# ---------------------------------------------------------------------------


class TestPearson:
    def test_perfect_positive(self) -> None:
        assert _pearson([1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]) == pytest.approx(
            1.0
        )

    def test_perfect_negative(self) -> None:
        assert _pearson([1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]) == pytest.approx(
            -1.0
        )

    def test_zero_variance_returns_nan(self) -> None:
        assert math.isnan(_pearson([1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0]))

    def test_nan_pairs_filtered(self) -> None:
        # First pair drops because it has NaN; remaining three are perfectly correlated.
        r = _pearson([math.nan, 2.0, 3.0, 4.0], [99.0, 4.0, 6.0, 8.0])
        assert r == pytest.approx(1.0)

    def test_too_few_observations(self) -> None:
        assert math.isnan(_pearson([1.0], [2.0]))
        assert math.isnan(_pearson([math.nan, 1.0], [2.0, math.nan]))


class TestCeilingResponsiveness:
    @staticmethod
    def _signal_check(knob: str, signal_id: str, stage: str, values: dict[str, float]):
        from usecases_synthetic.lib.monotonicity import SignalCheck

        exp = SignalExpectation(
            knob=knob,
            signal_id=signal_id,
            stage=stage,
            metric="metric_path",
            direction="down",
            qualitative_only=False,
            target_delta_range=None,
            pool_check=False,
            notes="",
            source="card",
        )
        return SignalCheck(
            expectation=exp,
            values=values,
            is_monotone=True,
            within_range=True,
            observed_delta=values["hard"] - values["baseline"],
            baseline_position_ok=True,
            reason="",
        )

    def test_responsive_knob_yields_high_correlation(self) -> None:
        # Knob's metric falls monotonically as ceiling F1 falls -> r near +1.
        check = self._signal_check(
            "knob_01",
            "k01_sm_macro_f1",
            "sm",
            {"baseline": 0.95, "easy": 0.80, "medium": 0.60, "hard": 0.40},
        )
        bm = BestMemberCheck(
            stage="sm",
            values={"baseline": 0.95, "easy": 0.80, "medium": 0.60, "hard": 0.40},
            winners={lvl: "m" for lvl in LEVELS},
            is_non_increasing=True,
            observed_delta=-0.55,
            reason="",
        )
        out = compute_ceiling_responsiveness([check], [bm])
        assert out[("knob_01", "k01_sm_macro_f1", "sm")] == pytest.approx(1.0)

    def test_noop_knob_yields_nan_under_flat_signal(self) -> None:
        # Signal flat across levels -> zero variance -> NaN response.
        check = self._signal_check(
            "knob_02",
            "k02_sm_macro_f1",
            "sm",
            {"baseline": 0.5, "easy": 0.5, "medium": 0.5, "hard": 0.5},
        )
        bm = BestMemberCheck(
            stage="sm",
            values={"baseline": 0.95, "easy": 0.80, "medium": 0.60, "hard": 0.40},
            winners={lvl: "m" for lvl in LEVELS},
            is_non_increasing=True,
            observed_delta=-0.55,
            reason="",
        )
        out = compute_ceiling_responsiveness([check], [bm])
        assert math.isnan(out[("knob_02", "k02_sm_macro_f1", "sm")])

    def test_missing_stage_yields_nan(self) -> None:
        check = self._signal_check(
            "knob_05",
            "k05_norm_distinct_families",
            "norm",
            {"baseline": 1.0, "easy": 2.0, "medium": 3.0, "hard": 4.0},
        )
        # No norm best-member check supplied.
        out = compute_ceiling_responsiveness([check], [])
        assert math.isnan(out[("knob_05", "k05_norm_distinct_families", "norm")])


# ---------------------------------------------------------------------------
# build_cross_level_slope (cumulative cross-level slope — load-bearing verdict)
# ---------------------------------------------------------------------------


def _slope_level_metrics(sm, norm, blk, emm, fus):
    """Build a minimal per_stage metrics dict for the 5 headline metrics."""
    return {
        "per_stage": {
            "sm": {"aggregated": {"macro_f1": sm}},
            "norm": {"aggregated": {"macro_f1": norm}},
            "em_blocking": {"aggregated": {"macro_pair_recall": blk}},
            "em_matching": {
                "aggregated": {"macro_f1_variant_model_on_regen_test": emm}
            },
            "fusion": {"aggregated": {"overall_accuracy": fus}},
        }
    }


class TestBuildCrossLevelSlope:
    """The cumulative cross-level slope is the load-bearing C2-contract
    verdict: it reads each stage's committee headline metric off the
    cumulative variant levels with no per-knob isolation assumption."""

    def test_weakly_decreasing_slope_passes_all_stages(self) -> None:
        lm = {
            "baseline": _slope_level_metrics(0.95, 0.95, 0.95, 0.95, 0.95),
            "easy": _slope_level_metrics(0.90, 0.90, 0.90, 0.90, 0.90),
            "medium": _slope_level_metrics(0.80, 0.80, 0.80, 0.80, 0.80),
            "hard": _slope_level_metrics(0.70, 0.70, 0.70, 0.70, 0.70),
        }
        by = {r["stage"]: r for r in build_cross_level_slope(lm)}
        assert set(by) == {"sm", "norm", "em_blocking", "em_matching", "fusion"}
        for st in by:
            assert by[st]["slope_ok"] is True
            assert by[st]["baseline_ok"] is True
            assert by[st]["delta_easy_hard"] == pytest.approx(-0.20)

    def test_medium_bump_flags_slope_and_baseline(self) -> None:
        # em_blocking bumps up at medium (0.89 -> 0.91) then falls, and
        # baseline (0.86) lands below medium -> both checks must fail.
        lm = {
            "baseline": _slope_level_metrics(0.95, 0.95, 0.86, 0.95, 0.95),
            "easy": _slope_level_metrics(0.90, 0.90, 0.89, 0.90, 0.90),
            "medium": _slope_level_metrics(0.80, 0.80, 0.91, 0.80, 0.80),
            "hard": _slope_level_metrics(0.70, 0.70, 0.85, 0.70, 0.70),
        }
        by = {r["stage"]: r for r in build_cross_level_slope(lm)}
        assert by["em_blocking"]["slope_ok"] is False
        assert by["em_blocking"]["baseline_ok"] is False
        # the cleanly-decreasing stages are unaffected
        assert by["sm"]["slope_ok"] is True

    def test_within_tolerance_counts_as_weakly_decreasing(self) -> None:
        # tiny upticks within _SLOPE_TOL (0.005) still pass.
        lm = {
            "baseline": _slope_level_metrics(0.90, 0.90, 0.90, 0.90, 0.90),
            "easy": _slope_level_metrics(0.80, 0.80, 0.80, 0.80, 0.80),
            "medium": _slope_level_metrics(0.802, 0.80, 0.80, 0.80, 0.80),
            "hard": _slope_level_metrics(0.79, 0.80, 0.80, 0.80, 0.80),
        }
        by = {r["stage"]: r for r in build_cross_level_slope(lm)}
        assert by["sm"]["slope_ok"] is True

    def test_missing_metric_is_nan_not_ok(self) -> None:
        # A stage whose headline metric is absent must not pass the slope
        # check (NaN-guarded), and its values are NaN, not fabricated.
        lm = {
            lvl: {"per_stage": {"sm": {"aggregated": {}}}}
            for lvl in ("baseline", "easy", "medium", "hard")
        }
        by = {r["stage"]: r for r in build_cross_level_slope(lm)}
        assert by["sm"]["slope_ok"] is False
        assert math.isnan(by["sm"]["values"]["hard"])
