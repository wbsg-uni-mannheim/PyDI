"""Tests for the EM committee runner.

Exercises ``EMCommitteeRunner`` instantiation from a fixture roster,
scoring against synthetic gold, pool-agreement diagnostics, and graceful
handling of missing source pairs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from usecases_synthetic.lib.committee import CommitteeResult, MemberResult
from usecases_synthetic.lib.committee_em import (
    EMCommitteeRunner,
    EMMatchingCommitteeRunner,
    _CompositionConfig,
    _resolve_column_mapping,
    _resolve_variant_checkpoint_path,
    _resolve_variant_train_path,
    _select_best_blocker,
)
from usecases_synthetic.lib.committee_em_scoring import (
    blocking_pair_recall,
    pool_agreement,
    reduction_ratio,
    score_em_correspondences_closed_set,
)
from usecases_synthetic.lib.variant_loader import VariantBundle

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_fixture_roster(
    tmp_path: Path,
    *,
    threshold: float = 0.3,
) -> tuple[Path, Path]:
    """Write a minimal split EM roster (blocking + matching YAMLs) for tests.

    Returns
    -------
    tuple of (Path, Path)
        ``(blocking_roster_path, matching_roster_path)``.

    Blocking roster: two members (``token_blocker``, ``standard_blocker``)
    covering distinct selection paths so the winner-selection logic has
    something to choose between. Composition defaults (``select_best`` /
    ``recall_floor=0.97`` / ``tie_breaker=reduction_ratio``) apply.

    Matching roster: two rule-based matchers (``rule_jaccard_a``,
    ``rule_jaccard_b``) with identical logic at different thresholds so
    per-member aggregation has more than one number to fold together.
    Both use the StringComparator on ``name`` with the Jaccard similarity
    function at a low threshold so the tiny synthetic fixture produces
    matches.
    """
    blocking_members: list[dict[str, Any]] = [
        {
            "name": "token_blocker",
            "description": "Token-set inverted-index blocker (lexical).",
            "blocker": {
                "class": "TokenBlocker",
                "module": "PyDI.entitymatching.blocking.token_blocking",
                "params": {"column": "name", "min_token_len": 2},
            },
            "blocking_type": "lexical",
            "enabled_by_default": True,
        },
        {
            "name": "standard_blocker",
            "description": "Equality on derived name_first_token (lexical).",
            "blocker": {
                "class": "StandardBlocker",
                "module": "PyDI.entitymatching.blocking.standard",
                "params": {"on": ["name_first_token"]},
            },
            "blocking_type": "lexical",
            "enabled_by_default": True,
        },
    ]

    blocking_roster: dict[str, Any] = {
        "seed": 42,
        "preprocess_text": "normalize_text",
        "members": blocking_members,
        "column_mapping": {},
        "composition": {
            "strategy": "select_best",
            "recall_floor": 0.97,
            "tie_breaker": "reduction_ratio",
        },
        "required_axes": {
            "blocking_type": ["lexical"],
        },
    }

    rule_comparators = [
        {
            "class": "StringComparator",
            "module": "PyDI.entitymatching.comparators",
            "params": {
                "column": "name",
                "similarity_function": "jaccard",
            },
        }
    ]

    matching_members: list[dict[str, Any]] = [
        {
            "name": "rule_jaccard_a",
            "description": "Rule-based matcher (Jaccard on name, low threshold).",
            "matcher": {
                "class": "RuleBasedMatcher",
                "module": "PyDI.entitymatching.rule_based",
                "params": {},
            },
            "comparators": rule_comparators,
            "weights": [1.0],
            "threshold": threshold,
            "matching_type": "rule",
            "missing_value_tolerant": False,
            "enabled_by_default": True,
        },
        {
            "name": "rule_jaccard_b",
            "description": "Rule-based matcher (Jaccard on name, same threshold).",
            "matcher": {
                "class": "RuleBasedMatcher",
                "module": "PyDI.entitymatching.rule_based",
                "params": {},
            },
            "comparators": rule_comparators,
            "weights": [1.0],
            "threshold": threshold,
            "matching_type": "rule",
            "missing_value_tolerant": False,
            "enabled_by_default": True,
        },
    ]

    matching_roster: dict[str, Any] = {
        "seed": 42,
        "preprocess_text": "normalize_text",
        "members": matching_members,
        "column_mapping": {},
        "required_axes": {
            "matching_type": ["rule"],
        },
    }

    blocking_path = tmp_path / "em_blocking_committee.yaml"
    matching_path = tmp_path / "em_matching_committee.yaml"
    with open(blocking_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(blocking_roster, f)
    with open(matching_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(matching_roster, f)
    return blocking_path, matching_path


def _make_two_source_data(
    n: int = 20,
) -> tuple[
    dict[str, pd.DataFrame],
    dict[tuple[str, str], pd.DataFrame],
    pd.DataFrame | None,
]:
    """Create 2 sources with overlapping entities and a test gold.

    Returns ``(sources, em_gold, pooled_positives)``.

    The sources share entities by name prefix:
    - source_a has ``"alpha corp", "beta inc", ...``
    - source_b has ``"alpha corporation", "beta incorporated", ...``

    Gold: the first ``n//2`` entities in each source are true matches.
    """
    rng = np.random.default_rng(42)

    names_a = [f"company_{i}" for i in range(n)]
    names_b = [f"company_{i}" for i in range(n)]

    src_a = pd.DataFrame(
        {
            "id": [f"a_{i}" for i in range(n)],
            "name": names_a,
            "country": rng.choice(["US", "DE", "JP"], size=n).tolist(),
        }
    )
    src_a.attrs["dataset_name"] = "source_a"

    src_b = pd.DataFrame(
        {
            "id": [f"b_{i}" for i in range(n)],
            "name": names_b,
            "country": rng.choice(["US", "DE", "JP"], size=n).tolist(),
        }
    )
    src_b.attrs["dataset_name"] = "source_b"

    sources = {"source_a": src_a, "source_b": src_b}

    # Gold: first n//2 are true matches, rest are false.
    n_match = n // 2
    rows = []
    for i in range(n_match):
        rows.append((f"a_{i}", f"b_{i}", "true"))
    for i in range(n_match, n):
        rows.append((f"a_{i}", f"b_{i}", "false"))
    gold = pd.DataFrame(rows, columns=["id1", "id2", "label"])

    em_gold: dict[tuple[str, str], pd.DataFrame] = {
        ("source_a", "source_b"): gold,
    }

    # Pool: subset of the true matches.
    pool_rows = [
        (f"a_{i}", f"b_{i}", "source_a", "source_b", 1) for i in range(n_match // 2)
    ]
    pooled = pd.DataFrame(
        pool_rows,
        columns=["id1", "id2", "source_1", "source_2", "pool_agreement"],
    )

    return sources, em_gold, pooled


def _make_bundle(
    sources: dict[str, pd.DataFrame],
    em_gold: dict[tuple[str, str], pd.DataFrame],
    pooled_positives: pd.DataFrame | None = None,
    *,
    level: str = "baseline",
    em_gold_regenerated: (
        dict[tuple[str, str], dict[str, dict[str, pd.DataFrame]]] | None
    ) = None,
) -> VariantBundle:
    """Build a minimal VariantBundle for EM testing."""
    return VariantBundle(
        domain="companies",
        level=level,
        sources=sources,
        target_schema={"title": "Company", "properties": {"id": {}, "name": {}}},
        sm_mapping=None,
        em_gold=em_gold,
        em_splits={},
        em_gold_regenerated=em_gold_regenerated or {},
        fusion_gold=pd.DataFrame(),
        fusion_validation=None,
        pooled_positives=pooled_positives,
        variant_root=Path("/tmp/em_test"),
    )


# ---------------------------------------------------------------------------
# Scoring unit tests
# ---------------------------------------------------------------------------


class TestScoreEMCorrespondencesClosedSet:
    """Tests for ``score_em_correspondences_closed_set``.

    The closed-set scorer restricts predictions to the gold's judged
    pair set (positives + negatives) before computing P/R/F1. Pairs
    outside the gold's universe are out of scope: they do not count as
    FPs. That is the correct semantics for a closed-set benchmark
    (e.g. the Knob-2 regenerated validation split).
    """

    def test_perfect_on_closed_set(self) -> None:
        """Predicting every gold positive with no FP gives F1 = 1.0."""
        gold = pd.DataFrame(
            [("a", "x", "true"), ("b", "y", "false")],
            columns=["id1", "id2", "label"],
        )
        pred = pd.DataFrame(
            [("a", "x", 0.9)],
            columns=["id1", "id2", "score"],
        )
        metrics = score_em_correspondences_closed_set(pred, gold)
        assert metrics["f1"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["pred_scoped"] == 1.0

    def test_out_of_scope_predictions_ignored(self) -> None:
        """Predicted pairs outside the gold universe are NOT counted as FP.

        This is the contamination fix that motivated S4c: without
        scoping, the matcher's predictions on the full dataset dominate
        the FP count and crush precision even when every in-scope
        prediction is correct.
        """
        gold = pd.DataFrame(
            [("a", "x", "true"), ("b", "y", "false")],
            columns=["id1", "id2", "label"],
        )
        pred = pd.DataFrame(
            [
                ("a", "x", 0.9),  # gold positive -> TP
                ("c", "z", 0.8),  # outside gold universe -> ignored
                ("d", "w", 0.7),  # outside gold universe -> ignored
            ],
            columns=["id1", "id2", "score"],
        )
        metrics = score_em_correspondences_closed_set(pred, gold)
        # Only the in-scope prediction contributes. Out-of-scope ones
        # are dropped before P/R/F1 is computed.
        assert metrics["tp"] == 1.0
        assert metrics["fp"] == 0.0
        assert metrics["fn"] == 0.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0
        assert metrics["pred_scoped"] == 1.0

    def test_false_positive_on_regen_negative(self) -> None:
        """Predicting a match on a gold *negative* counts as FP."""
        gold = pd.DataFrame(
            [("a", "x", "true"), ("b", "y", "false")],
            columns=["id1", "id2", "label"],
        )
        pred = pd.DataFrame(
            [
                ("a", "x", 0.9),  # gold positive -> TP
                ("b", "y", 0.6),  # gold negative -> FP (deceptive niche)
            ],
            columns=["id1", "id2", "score"],
        )
        metrics = score_em_correspondences_closed_set(pred, gold)
        assert metrics["tp"] == 1.0
        assert metrics["fp"] == 1.0
        assert metrics["fn"] == 0.0
        assert metrics["precision"] == 0.5
        assert metrics["recall"] == 1.0

    def test_missed_regen_positive(self) -> None:
        """A gold positive the matcher did not predict is a FN."""
        gold = pd.DataFrame(
            [
                ("a", "x", "true"),
                ("b", "y", "true"),
                ("c", "z", "false"),
            ],
            columns=["id1", "id2", "label"],
        )
        pred = pd.DataFrame(
            [("a", "x", 0.9)],
            columns=["id1", "id2", "score"],
        )
        metrics = score_em_correspondences_closed_set(pred, gold)
        assert metrics["tp"] == 1.0
        assert metrics["fp"] == 0.0
        assert metrics["fn"] == 1.0
        assert metrics["recall"] == 0.5

    def test_empty_prediction(self) -> None:
        """Empty predictions yield P=0, R=0, F1=0 (regen has positives)."""
        gold = pd.DataFrame(
            [("a", "x", "true")],
            columns=["id1", "id2", "label"],
        )
        pred = pd.DataFrame(columns=["id1", "id2", "score"])
        metrics = score_em_correspondences_closed_set(pred, gold)
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["f1"] == 0.0
        assert metrics["pred_scoped"] == 0.0

    def test_reversed_orientation(self) -> None:
        """``(a, b)`` matches ``(b, a)`` — orientation is ignored."""
        gold = pd.DataFrame(
            [("a", "x", "true")],
            columns=["id1", "id2", "label"],
        )
        pred = pd.DataFrame(
            [("x", "a", 0.9)],
            columns=["id1", "id2", "score"],
        )
        metrics = score_em_correspondences_closed_set(pred, gold)
        assert metrics["tp"] == 1.0
        assert metrics["f1"] == 1.0


class TestPoolAgreement:
    """Tests for ``pool_agreement``."""

    def test_full_overlap(self) -> None:
        """All predictions appear in the pool."""
        pred = pd.DataFrame(
            [("a", "x"), ("b", "y")],
            columns=["id1", "id2"],
        )
        pool = pd.DataFrame(
            [("a", "x", "s1", "s2", 1), ("b", "y", "s1", "s2", 1)],
            columns=["id1", "id2", "source_1", "source_2", "pool_agreement"],
        )
        metrics = pool_agreement(pred, pool)
        assert metrics["pool_precision"] == 1.0
        assert metrics["pool_recall"] == 1.0

    def test_no_pool(self) -> None:
        """When pool is None, all metrics are 0."""
        pred = pd.DataFrame(
            [("a", "x")],
            columns=["id1", "id2"],
        )
        metrics = pool_agreement(pred, None)
        assert metrics["pool_precision"] == 0.0
        assert metrics["pool_recall"] == 0.0

    def test_partial_overlap(self) -> None:
        """Half of predictions in pool, half of pool covered."""
        pred = pd.DataFrame(
            [("a", "x"), ("c", "z")],
            columns=["id1", "id2"],
        )
        pool = pd.DataFrame(
            [("a", "x", "s1", "s2", 1), ("b", "y", "s1", "s2", 1)],
            columns=["id1", "id2", "source_1", "source_2", "pool_agreement"],
        )
        metrics = pool_agreement(pred, pool)
        assert metrics["pool_precision"] == 0.5
        assert metrics["pool_recall"] == 0.5
        assert metrics["pool_overlap"] == 1.0


class TestBlockingPairRecall:
    """Tests for ``blocking_pair_recall``."""

    def test_perfect_recall(self) -> None:
        """Every positive gold pair present in candidates → recall = 1.0."""
        gold = pd.DataFrame(
            [("a", "x", "true"), ("b", "y", "true"), ("c", "z", "false")],
            columns=["id1", "id2", "label"],
        )
        candidates = pd.DataFrame(
            [("a", "x"), ("b", "y"), ("a", "y")],
            columns=["id1", "id2"],
        )
        m = blocking_pair_recall(candidates, gold)
        assert m["pair_recall"] == 1.0
        assert m["gold_positives"] == 2.0
        assert m["covered"] == 2.0
        assert m["missed"] == 0.0

    def test_partial_recall(self) -> None:
        """One of two gold positives retained → recall = 0.5."""
        gold = pd.DataFrame(
            [("a", "x", "true"), ("b", "y", "true")],
            columns=["id1", "id2", "label"],
        )
        candidates = pd.DataFrame(
            [("a", "x"), ("c", "z")],
            columns=["id1", "id2"],
        )
        m = blocking_pair_recall(candidates, gold)
        assert m["pair_recall"] == 0.5
        assert m["covered"] == 1.0
        assert m["missed"] == 1.0

    def test_reversed_pair_matches(self) -> None:
        """Candidate (b, a) satisfies gold (a, b) — unordered comparison."""
        gold = pd.DataFrame(
            [("a", "b", "true")],
            columns=["id1", "id2", "label"],
        )
        candidates = pd.DataFrame(
            [("b", "a")],
            columns=["id1", "id2"],
        )
        m = blocking_pair_recall(candidates, gold)
        assert m["pair_recall"] == 1.0

    def test_empty_candidates(self) -> None:
        """No candidates → recall = 0.0 but gold count stays correct."""
        gold = pd.DataFrame(
            [("a", "x", "true")],
            columns=["id1", "id2", "label"],
        )
        candidates = pd.DataFrame(columns=["id1", "id2"])
        m = blocking_pair_recall(candidates, gold)
        assert m["pair_recall"] == 0.0
        assert m["gold_positives"] == 1.0
        assert m["missed"] == 1.0

    def test_no_positive_gold(self) -> None:
        """No gold positives → all-zero result (no ZeroDivisionError)."""
        gold = pd.DataFrame(
            [("a", "x", "false")],
            columns=["id1", "id2", "label"],
        )
        candidates = pd.DataFrame(
            [("a", "x")],
            columns=["id1", "id2"],
        )
        m = blocking_pair_recall(candidates, gold)
        assert m["pair_recall"] == 0.0
        assert m["gold_positives"] == 0.0

    def test_false_gold_rows_do_not_count(self) -> None:
        """Only ``label == "true"`` rows are treated as positives."""
        gold = pd.DataFrame(
            [
                ("a", "x", "true"),
                ("b", "y", "false"),
                ("c", "z", "true"),
            ],
            columns=["id1", "id2", "label"],
        )
        candidates = pd.DataFrame(
            [("a", "x"), ("c", "z")],
            columns=["id1", "id2"],
        )
        m = blocking_pair_recall(candidates, gold)
        assert m["gold_positives"] == 2.0
        assert m["pair_recall"] == 1.0


class TestReductionRatio:
    """Tests for ``reduction_ratio``."""

    def test_half_cut(self) -> None:
        """Candidate set covers half the full pair-space → RR = 0.5."""
        candidates = pd.DataFrame(
            [("a", "x"), ("a", "y")],
            columns=["id1", "id2"],
        )
        m = reduction_ratio(candidates, n_left=2, n_right=2)
        assert m["reduction_ratio"] == 0.5
        assert m["candidate_count"] == 2.0
        assert m["full_space"] == 4.0

    def test_no_pruning(self) -> None:
        """Candidates equal the full pair-space → RR = 0."""
        candidates = pd.DataFrame(
            [("a", "x"), ("a", "y"), ("b", "x"), ("b", "y")],
            columns=["id1", "id2"],
        )
        m = reduction_ratio(candidates, n_left=2, n_right=2)
        assert m["reduction_ratio"] == 0.0

    def test_empty_candidates(self) -> None:
        """Empty candidate set → RR = 1.0 (all pairs pruned)."""
        candidates = pd.DataFrame(columns=["id1", "id2"])
        m = reduction_ratio(candidates, n_left=10, n_right=10)
        assert m["reduction_ratio"] == 1.0
        assert m["candidate_count"] == 0.0

    def test_duplicate_candidates_deduped(self) -> None:
        """Duplicate pairs collapse before computing the ratio."""
        candidates = pd.DataFrame(
            [("a", "x"), ("a", "x"), ("x", "a")],
            columns=["id1", "id2"],
        )
        m = reduction_ratio(candidates, n_left=2, n_right=2)
        assert m["candidate_count"] == 1.0
        assert m["reduction_ratio"] == 0.75

    def test_negative_size_raises(self) -> None:
        """Negative source sizes are rejected at the boundary."""
        candidates = pd.DataFrame(columns=["id1", "id2"])
        with pytest.raises(ValueError, match="non-negative"):
            reduction_ratio(candidates, n_left=-1, n_right=2)

    def test_zero_pair_space_raises(self) -> None:
        """An empty source makes RR undefined — raise rather than divide."""
        candidates = pd.DataFrame(columns=["id1", "id2"])
        with pytest.raises(ValueError, match="undefined"):
            reduction_ratio(candidates, n_left=0, n_right=5)


# ---------------------------------------------------------------------------
# EMCommitteeRunner tests
# ---------------------------------------------------------------------------


class TestEMCommitteeRunner:
    """Tests for the full EM committee runner."""

    def test_instantiation(self, tmp_path: Path) -> None:
        """Runner loads the split rosters and parses members."""
        blocking_path, matching_path = _write_fixture_roster(tmp_path)
        runner = EMCommitteeRunner(blocking_path, matching_path)

        assert runner.roster_names == ["rule_jaccard_a", "rule_jaccard_b"]
        assert runner.blocking_roster_names == ["token_blocker", "standard_blocker"]
        assert len(runner.roster) == 2

    def test_run_produces_committee_result(self, tmp_path: Path) -> None:
        """Basic end-to-end run produces a valid CommitteeResult."""
        blocking_path, matching_path = _write_fixture_roster(tmp_path)
        runner = EMCommitteeRunner(blocking_path, matching_path)

        sources, em_gold, pooled = _make_two_source_data()
        bundle = _make_bundle(sources, em_gold, pooled)

        result = runner.run(bundle)

        assert isinstance(result, CommitteeResult)
        assert result.stage == "em"
        assert result.domain == "companies"
        assert result.level == "baseline"
        assert set(result.per_member) == {"rule_jaccard_a", "rule_jaccard_b"}
        assert result.roster == ["rule_jaccard_a", "rule_jaccard_b"]
        assert set(result.per_blocker) == {"token_blocker", "standard_blocker"}

        # Aggregated metrics present.
        assert "macro_f1" in result.aggregated
        assert "min_f1" in result.aggregated
        assert "max_f1" in result.aggregated
        assert "macro_pool_precision" in result.aggregated
        assert "macro_pool_recall" in result.aggregated

    def test_per_partition_has_source_pair(self, tmp_path: Path) -> None:
        """per_partition should have an entry for the source pair."""
        blocking_path, matching_path = _write_fixture_roster(tmp_path)
        runner = EMCommitteeRunner(blocking_path, matching_path)

        sources, em_gold, pooled = _make_two_source_data()
        bundle = _make_bundle(sources, em_gold, pooled)

        result = runner.run(bundle)

        assert "source_a_source_b" in result.per_partition
        pair_metrics = result.per_partition["source_a_source_b"]
        assert "macro_f1" in pair_metrics
        assert "n_members" in pair_metrics
        assert pair_metrics["n_members"] == 2.0
        # Winner reporting populated by Phase 1.
        assert "winner_pair_recall" in pair_metrics
        assert "winner_reduction_ratio" in pair_metrics

    def test_per_member_has_pair_detail(self, tmp_path: Path) -> None:
        """Each matcher's notes should contain per_pair detail."""
        blocking_path, matching_path = _write_fixture_roster(tmp_path)
        runner = EMCommitteeRunner(blocking_path, matching_path)

        sources, em_gold, pooled = _make_two_source_data()
        bundle = _make_bundle(sources, em_gold, pooled)

        result = runner.run(bundle)

        for member_name, member_result in result.per_member.items():
            assert member_result.notes["role"] == "matcher"
            assert "per_pair" in member_result.notes
            per_pair = member_result.notes["per_pair"]
            assert "source_a_source_b" in per_pair
            pair_m = per_pair["source_a_source_b"]
            assert "f1" in pair_m
            assert "pool_precision" in pair_m

    def test_per_blocker_has_pair_detail(self, tmp_path: Path) -> None:
        """Each blocker's notes carry per-pair recall + reduction_ratio."""
        blocking_path, matching_path = _write_fixture_roster(tmp_path)
        runner = EMCommitteeRunner(blocking_path, matching_path)

        sources, em_gold, pooled = _make_two_source_data()
        bundle = _make_bundle(sources, em_gold, pooled)

        result = runner.run(bundle)

        for blocker_name, blocker_result in result.per_blocker.items():
            assert blocker_result.notes["role"] == "blocker"
            assert "per_pair" in blocker_result.notes
            per_pair = blocker_result.notes["per_pair"]
            assert "source_a_source_b" in per_pair
            pair_m = per_pair["source_a_source_b"]
            assert "pair_recall" in pair_m
            assert "reduction_ratio" in pair_m
            assert "selected" in pair_m
            assert pair_m["selected"] in {0.0, 1.0}
        # Exactly one blocker is selected per pair.
        selection_rates = sum(
            m.metrics["selection_rate"] for m in result.per_blocker.values()
        )
        assert selection_rates == pytest.approx(1.0)

    def test_pool_agreement_populated(self, tmp_path: Path) -> None:
        """Pool agreement metrics should be non-negative."""
        blocking_path, matching_path = _write_fixture_roster(tmp_path)
        runner = EMCommitteeRunner(blocking_path, matching_path)

        sources, em_gold, pooled = _make_two_source_data()
        bundle = _make_bundle(sources, em_gold, pooled)

        result = runner.run(bundle)

        for member_result in result.per_member.values():
            assert member_result.metrics["pool_precision"] >= 0.0
            assert member_result.metrics["pool_recall"] >= 0.0

    def test_no_clustering(self, tmp_path: Path) -> None:
        """Runner works with clustering disabled."""
        blocking_path, matching_path = _write_fixture_roster(tmp_path)
        runner = EMCommitteeRunner(blocking_path, matching_path, clustering="none")

        sources, em_gold, pooled = _make_two_source_data()
        bundle = _make_bundle(sources, em_gold, pooled)

        result = runner.run(bundle)
        assert isinstance(result, CommitteeResult)

    def test_missing_gold_raises(self, tmp_path: Path) -> None:
        """Runner raises ValueError when no EM gold is available."""
        blocking_path, matching_path = _write_fixture_roster(tmp_path)
        runner = EMCommitteeRunner(blocking_path, matching_path)

        bundle = _make_bundle(
            sources={"source_a": pd.DataFrame(), "source_b": pd.DataFrame()},
            em_gold={},
        )

        with pytest.raises(ValueError, match="No EM gold"):
            runner.run(bundle)

    def test_no_pool_still_works(self, tmp_path: Path) -> None:
        """Runner runs fine when pooled_positives is None."""
        blocking_path, matching_path = _write_fixture_roster(tmp_path)
        runner = EMCommitteeRunner(blocking_path, matching_path)

        sources, em_gold, _ = _make_two_source_data()
        bundle = _make_bundle(sources, em_gold, None)

        result = runner.run(bundle)
        assert isinstance(result, CommitteeResult)
        # Pool metrics should be zero.
        for member_result in result.per_member.values():
            assert member_result.metrics["pool_precision"] == 0.0
            assert member_result.metrics["pool_recall"] == 0.0

    def test_result_as_dict_serializable(self, tmp_path: Path) -> None:
        """CommitteeResult.as_dict() produces a JSON-serialisable dict."""
        import json

        blocking_path, matching_path = _write_fixture_roster(tmp_path)
        runner = EMCommitteeRunner(blocking_path, matching_path)

        sources, em_gold, pooled = _make_two_source_data()
        bundle = _make_bundle(sources, em_gold, pooled)

        result = runner.run(bundle)
        payload = result.as_dict()
        json.dumps(payload)
        assert "per_blocker" in payload

    def test_determinism(self, tmp_path: Path) -> None:
        """Two sequential runs produce identical F1 numbers."""
        blocking_path, matching_path = _write_fixture_roster(tmp_path)

        sources, em_gold, pooled = _make_two_source_data()
        bundle = _make_bundle(sources, em_gold, pooled)

        runner1 = EMCommitteeRunner(blocking_path, matching_path)
        result1 = runner1.run(bundle)

        runner2 = EMCommitteeRunner(blocking_path, matching_path)
        result2 = runner2.run(bundle)

        for member_name in result1.per_member:
            f1_a = result1.per_member[member_name].metrics["f1"]
            f1_b = result2.per_member[member_name].metrics["f1"]
            assert f1_a == pytest.approx(f1_b), (
                f"Non-deterministic F1 for {member_name}: " f"{f1_a} != {f1_b}"
            )
        # Winner selection is also deterministic.
        assert result1.per_partition["source_a_source_b"][
            "winner_reduction_ratio"
        ] == pytest.approx(
            result2.per_partition["source_a_source_b"]["winner_reduction_ratio"]
        )

    def test_llm_filtered_by_default(self, tmp_path: Path) -> None:
        """LLM matchers are excluded when with_llm=False."""
        blocking_roster = {
            "seed": 42,
            "members": [
                {
                    "name": "token_blocker",
                    "blocker": {
                        "class": "TokenBlocker",
                        "module": "PyDI.entitymatching.blocking.token_blocking",
                        "params": {"column": "name", "min_token_len": 2},
                    },
                    "blocking_type": "lexical",
                    "enabled_by_default": True,
                }
            ],
        }
        matching_roster = {
            "seed": 42,
            "members": [
                {
                    "name": "rule_jaccard",
                    "matcher": {
                        "class": "RuleBasedMatcher",
                        "module": "PyDI.entitymatching.rule_based",
                        "params": {},
                    },
                    "comparators": [
                        {
                            "class": "StringComparator",
                            "module": "PyDI.entitymatching.comparators",
                            "params": {
                                "column": "name",
                                "similarity_function": "jaccard",
                            },
                        }
                    ],
                    "weights": [1.0],
                    "threshold": 0.3,
                    "matching_type": "rule",
                    "enabled_by_default": True,
                },
                {
                    "name": "llm_matcher",
                    "matcher": {
                        "class": "LLMBasedMatcher",
                        "module": "PyDI.entitymatching.llm_based",
                        "params": {},
                    },
                    "comparators": [],
                    "weights": [],
                    "threshold": 0.5,
                    "matching_type": "llm",
                    "enabled_by_default": False,
                },
            ],
        }

        blocking_path = tmp_path / "em_blocking_with_llm.yaml"
        matching_path = tmp_path / "em_matching_with_llm.yaml"
        with open(blocking_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(blocking_roster, f)
        with open(matching_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(matching_roster, f)

        runner = EMCommitteeRunner(blocking_path, matching_path, with_llm=False)
        assert runner.roster_names == ["rule_jaccard"]


class TestSelectBestBlocker:
    """Functional tests for the Phase 1 blocker-selection policy."""

    def _cfg(
        self,
        recall_floor: float = 0.97,
        tie_breaker: str = "reduction_ratio",
    ) -> _CompositionConfig:
        return _CompositionConfig(
            strategy="select_best",
            recall_floor=recall_floor,
            tie_breaker=tie_breaker,
        )

    def test_picks_highest_reduction_ratio_among_survivors(self) -> None:
        """Among blockers clearing the recall floor, highest RR wins."""
        metrics = {
            "token_blocker": {"pair_recall": 1.00, "reduction_ratio": 0.80},
            "standard_blocker": {"pair_recall": 1.00, "reduction_ratio": 0.60},
            "embedding_blocker": {"pair_recall": 1.00, "reduction_ratio": 0.95},
        }
        winner, cleared = _select_best_blocker(metrics, self._cfg())
        assert winner == "embedding_blocker"
        assert cleared is True

    def test_filters_below_recall_floor(self) -> None:
        """Blockers below the floor are excluded even if their RR is higher."""
        metrics = {
            "greedy_pruner": {"pair_recall": 0.80, "reduction_ratio": 0.99},
            "safe_blocker": {"pair_recall": 0.98, "reduction_ratio": 0.50},
        }
        winner, cleared = _select_best_blocker(metrics, self._cfg())
        assert winner == "safe_blocker"
        assert cleared is True

    def test_alphabetical_tie_break_on_equal_rr(self) -> None:
        """Deterministic tie-break: lexicographically earliest name wins."""
        metrics = {
            "zzz_blocker": {"pair_recall": 1.0, "reduction_ratio": 0.90},
            "aaa_blocker": {"pair_recall": 1.0, "reduction_ratio": 0.90},
            "mmm_blocker": {"pair_recall": 1.0, "reduction_ratio": 0.90},
        }
        winner, cleared = _select_best_blocker(metrics, self._cfg())
        assert winner == "aaa_blocker"
        assert cleared is True

    def test_fallback_when_no_blocker_clears_floor(self) -> None:
        """All below floor → pick highest-recall blocker + signal shortfall."""
        metrics = {
            "poor_a": {"pair_recall": 0.70, "reduction_ratio": 0.99},
            "poor_b": {"pair_recall": 0.80, "reduction_ratio": 0.50},
            "poor_c": {"pair_recall": 0.75, "reduction_ratio": 0.95},
        }
        winner, cleared = _select_best_blocker(metrics, self._cfg())
        assert winner == "poor_b"
        assert cleared is False

    def test_fallback_tie_alphabetical_on_equal_recall(self) -> None:
        """Fallback ties on pair_recall resolved alphabetically too."""
        metrics = {
            "zzz": {"pair_recall": 0.90, "reduction_ratio": 0.10},
            "aaa": {"pair_recall": 0.90, "reduction_ratio": 0.99},
        }
        winner, cleared = _select_best_blocker(metrics, self._cfg())
        assert winner == "aaa"
        assert cleared is False

    def test_empty_metrics_raises(self) -> None:
        """Empty blocker_metrics is a programmer error — raise rather than guess."""
        with pytest.raises(ValueError, match="empty blocker_metrics"):
            _select_best_blocker({}, self._cfg())


class TestEMCommitteeRunnerValidation:
    """Split-roster invariant checks at instantiation time."""

    def _write_rosters(
        self,
        tmp_path: Path,
        *,
        blocking_column_mapping: dict | None = None,
        matching_column_mapping: dict | None = None,
        matching_seed: int = 42,
    ) -> tuple[Path, Path]:
        blocking: dict[str, Any] = {
            "seed": 42,
            "members": [
                {
                    "name": "token_blocker",
                    "blocker": {
                        "class": "TokenBlocker",
                        "module": "PyDI.entitymatching.blocking.token_blocking",
                        "params": {"column": "name", "min_token_len": 2},
                    },
                    "blocking_type": "lexical",
                    "enabled_by_default": True,
                }
            ],
        }
        if blocking_column_mapping is not None:
            blocking["column_mapping"] = blocking_column_mapping

        matching: dict[str, Any] = {
            "seed": matching_seed,
            "members": [
                {
                    "name": "rule_jaccard",
                    "matcher": {
                        "class": "RuleBasedMatcher",
                        "module": "PyDI.entitymatching.rule_based",
                        "params": {},
                    },
                    "comparators": [
                        {
                            "class": "StringComparator",
                            "module": "PyDI.entitymatching.comparators",
                            "params": {
                                "column": "name",
                                "similarity_function": "jaccard",
                            },
                        }
                    ],
                    "weights": [1.0],
                    "threshold": 0.3,
                    "matching_type": "rule",
                    "enabled_by_default": True,
                }
            ],
        }
        if matching_column_mapping is not None:
            matching["column_mapping"] = matching_column_mapping

        blocking_path = tmp_path / "blocking.yaml"
        matching_path = tmp_path / "matching.yaml"
        with open(blocking_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(blocking, f)
        with open(matching_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(matching, f)
        return blocking_path, matching_path

    def test_divergent_column_mapping_raises(self, tmp_path: Path) -> None:
        """Non-matching column_mapping between the two rosters is fatal."""
        blocking_path, matching_path = self._write_rosters(
            tmp_path,
            blocking_column_mapping={"src_a": {"old": "new"}},
            matching_column_mapping={"src_a": {"old": "different"}},
        )
        with pytest.raises(ValueError, match="column_mapping"):
            EMCommitteeRunner(blocking_path, matching_path)

    def test_mismatched_seeds_raise(self, tmp_path: Path) -> None:
        """Rosters with different seeds would break variant determinism."""
        blocking_path, matching_path = self._write_rosters(tmp_path, matching_seed=7)
        with pytest.raises(ValueError, match="different seeds"):
            EMCommitteeRunner(blocking_path, matching_path)

    def test_empty_blocking_roster_raises(self, tmp_path: Path) -> None:
        """Zero enabled blockers ⇒ nothing for Phase 2 to consume."""
        blocking = {
            "seed": 42,
            "members": [
                {
                    "name": "disabled_blocker",
                    "blocker": {
                        "class": "TokenBlocker",
                        "module": "PyDI.entitymatching.blocking.token_blocking",
                        "params": {"column": "name"},
                    },
                    "blocking_type": "lexical",
                    "enabled_by_default": False,
                }
            ],
        }
        matching = {
            "seed": 42,
            "members": [
                {
                    "name": "rule_jaccard",
                    "matcher": {
                        "class": "RuleBasedMatcher",
                        "module": "PyDI.entitymatching.rule_based",
                        "params": {},
                    },
                    "comparators": [],
                    "weights": [],
                    "threshold": 0.3,
                    "matching_type": "rule",
                    "enabled_by_default": True,
                }
            ],
        }
        blocking_path = tmp_path / "blocking_empty.yaml"
        matching_path = tmp_path / "matching.yaml"
        with open(blocking_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(blocking, f)
        with open(matching_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(matching, f)

        with pytest.raises(ValueError, match="No enabled blockers"):
            EMCommitteeRunner(blocking_path, matching_path)

    def test_empty_matching_roster_raises(self, tmp_path: Path) -> None:
        """Zero enabled matchers ⇒ blocker selection has no consumer."""
        blocking = {
            "seed": 42,
            "members": [
                {
                    "name": "token_blocker",
                    "blocker": {
                        "class": "TokenBlocker",
                        "module": "PyDI.entitymatching.blocking.token_blocking",
                        "params": {"column": "name"},
                    },
                    "blocking_type": "lexical",
                    "enabled_by_default": True,
                }
            ],
        }
        matching = {
            "seed": 42,
            "members": [
                {
                    "name": "disabled_matcher",
                    "matcher": {
                        "class": "RuleBasedMatcher",
                        "module": "PyDI.entitymatching.rule_based",
                        "params": {},
                    },
                    "comparators": [],
                    "weights": [],
                    "threshold": 0.3,
                    "matching_type": "rule",
                    "enabled_by_default": False,
                }
            ],
        }
        blocking_path = tmp_path / "blocking.yaml"
        matching_path = tmp_path / "matching_empty.yaml"
        with open(blocking_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(blocking, f)
        with open(matching_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(matching, f)

        with pytest.raises(ValueError, match="No enabled matchers"):
            EMCommitteeRunner(blocking_path, matching_path)


class TestEMScoreFallbackChain:
    """Tests for the C10 ``regen_test → baseline_test → pool`` fallback
    chain in ``EMCommitteeRunner._score_predictions``.

    Each (pair, split) carries two versions per plan_revision.md C11:
    ``corner_filled`` (Set 2 — the load-bearing monotonicity surface)
    and ``baseline_pruned`` (Set 1 — the per-level reference). The
    headline ``f1`` falls back ``regen_test → baseline_test → pool``;
    open-set scoring against the original human gold is retired.
    """

    def test_headline_f1_uses_regen_test_when_present(self, tmp_path: Path) -> None:
        """When corner_filled test is present, headline f1 should
        equal f1_regen_test, not the baseline-pruned reference and not
        the pool fallback."""
        blocking_path, matching_path = _write_fixture_roster(tmp_path)
        runner = EMCommitteeRunner(blocking_path, matching_path)

        sources, em_gold, pooled = _make_two_source_data()
        gold_df = em_gold[("source_a", "source_b")]
        em_gold_regenerated = {
            ("source_a", "source_b"): {
                "test": {
                    "corner_filled": gold_df.copy(),
                    "baseline_pruned": gold_df.copy(),
                },
            },
        }
        bundle = _make_bundle(
            sources, em_gold, pooled, em_gold_regenerated=em_gold_regenerated
        )

        result = runner.run(bundle)

        for member_name, mr in result.per_member.items():
            per_pair = mr.notes["per_pair"]["source_a_source_b"]
            assert "f1_regen_test" in per_pair
            assert "f1_baseline_test" in per_pair
            assert "f1_vs_pool" in per_pair
            assert "f1_vs_test_gold" not in per_pair
            assert "f1_vs_regenerated_val" not in per_pair
            assert "f1_vs_regenerated_test" not in per_pair
            rt = per_pair["f1_regen_test"]
            f1 = per_pair["f1"]
            if not (isinstance(rt, float) and rt != rt):
                assert abs(f1 - rt) < 1e-9, (
                    f"{member_name}: headline f1={f1} should equal "
                    f"f1_regen_test={rt}"
                )

    def test_headline_f1_falls_back_to_baseline_test_when_corner_missing(
        self, tmp_path: Path
    ) -> None:
        """When only baseline_pruned is present, headline f1 falls
        back to f1_baseline_test (not pool)."""
        blocking_path, matching_path = _write_fixture_roster(tmp_path)
        runner = EMCommitteeRunner(blocking_path, matching_path)

        sources, em_gold, pooled = _make_two_source_data()
        gold_df = em_gold[("source_a", "source_b")]
        em_gold_regenerated = {
            ("source_a", "source_b"): {
                "test": {"baseline_pruned": gold_df.copy()},
            },
        }
        bundle = _make_bundle(
            sources, em_gold, pooled, em_gold_regenerated=em_gold_regenerated
        )

        result = runner.run(bundle)

        for member_name, mr in result.per_member.items():
            per_pair = mr.notes["per_pair"]["source_a_source_b"]
            bt = per_pair["f1_baseline_test"]
            f1 = per_pair["f1"]
            if not (isinstance(bt, float) and bt != bt):
                assert abs(f1 - bt) < 1e-9, (
                    f"{member_name}: headline f1={f1} should equal "
                    f"f1_baseline_test={bt}"
                )

    def test_headline_f1_falls_back_to_pool_when_neither_regen_present(
        self, tmp_path: Path
    ) -> None:
        """With no regen splits at all (the baseline shape), headline
        f1 should fall back to f1_vs_pool — preserving prior behaviour
        for domains that have not yet been augmented."""
        blocking_path, matching_path = _write_fixture_roster(tmp_path)
        runner = EMCommitteeRunner(blocking_path, matching_path)

        sources, em_gold, pooled = _make_two_source_data()
        bundle = _make_bundle(sources, em_gold, pooled, em_gold_regenerated={})

        result = runner.run(bundle)

        for member_name, mr in result.per_member.items():
            per_pair = mr.notes["per_pair"]["source_a_source_b"]
            pool = per_pair["f1_vs_pool"]
            f1 = per_pair["f1"]
            assert abs(f1 - pool) < 1e-9, (
                f"{member_name}: with no regen splits headline f1={f1} "
                f"should equal f1_vs_pool={pool}"
            )

    def test_macro_keys_replace_retired_surfaces(self, tmp_path: Path) -> None:
        """Aggregated macro keys carry the C10 names; retired
        ``macro_f1_vs_test_gold`` / ``macro_f1_vs_regenerated_*``
        keys are gone."""
        blocking_path, matching_path = _write_fixture_roster(tmp_path)
        runner = EMCommitteeRunner(blocking_path, matching_path)

        sources, em_gold, pooled = _make_two_source_data()
        gold_df = em_gold[("source_a", "source_b")]
        em_gold_regenerated = {
            ("source_a", "source_b"): {
                "test": {
                    "corner_filled": gold_df.copy(),
                    "baseline_pruned": gold_df.copy(),
                },
            },
        }
        bundle = _make_bundle(
            sources, em_gold, pooled, em_gold_regenerated=em_gold_regenerated
        )

        result = runner.run(bundle)

        assert "macro_f1_regen_test" in result.aggregated
        assert "macro_f1_baseline_test" in result.aggregated
        assert "macro_f1_vs_test_gold" not in result.aggregated
        assert "macro_f1_vs_regenerated_val" not in result.aggregated
        assert "macro_f1_vs_regenerated_test" not in result.aggregated


class TestResolveVariantCheckpointPath:
    """``_resolve_variant_checkpoint_path`` — R7b dual-model variant
    checkpoint lookup.

    Contract: at baseline level, returns the baseline path with
    ``is_variant_distinct=False`` regardless of whether a variant
    sibling exists. At variant levels, looks for
    ``<baseline_parent>/variant_<level>/best``; returns that path with
    ``is_variant_distinct=True`` when present, otherwise falls back to
    baseline with ``False``. ``None`` input → ``(None, False)`` (no
    checkpoint matcher).
    """

    def test_none_checkpoint_returns_none(self) -> None:
        path, distinct = _resolve_variant_checkpoint_path(None, "easy")
        assert path is None
        assert distinct is False

    def test_baseline_level_returns_baseline_path(self, tmp_path: Path) -> None:
        baseline = tmp_path / "domain" / "best"
        baseline.mkdir(parents=True)
        path, distinct = _resolve_variant_checkpoint_path(baseline, "baseline")
        assert path == baseline
        assert distinct is False

    def test_variant_level_without_sibling_falls_back(self, tmp_path: Path) -> None:
        baseline = tmp_path / "domain" / "best"
        baseline.mkdir(parents=True)
        # No variant_easy/ directory created.
        path, distinct = _resolve_variant_checkpoint_path(baseline, "easy")
        assert path == baseline
        assert distinct is False

    def test_variant_level_with_sibling_returns_variant(self, tmp_path: Path) -> None:
        baseline = tmp_path / "domain" / "best"
        baseline.mkdir(parents=True)
        variant = tmp_path / "domain" / "variant_easy" / "best"
        variant.mkdir(parents=True)
        path, distinct = _resolve_variant_checkpoint_path(baseline, "easy")
        assert path == variant
        assert distinct is True

    @pytest.mark.parametrize("level", ["easy", "medium", "hard"])
    def test_each_variant_level_has_its_own_subdir(
        self, tmp_path: Path, level: str
    ) -> None:
        baseline = tmp_path / "domain" / "best"
        baseline.mkdir(parents=True)
        variant = tmp_path / "domain" / f"variant_{level}" / "best"
        variant.mkdir(parents=True)
        path, distinct = _resolve_variant_checkpoint_path(baseline, level)
        assert path == variant
        assert distinct is True


class TestResolveVariantTrainPath:
    """``_resolve_variant_train_path`` — R7b Magellan-style retraining
    on the regenerated variant train CSV.

    Contract: at baseline level, returns the un-versioned
    ``<pair>_train.csv`` lookup (today's behaviour) with
    ``is_variant_distinct=False``. At variant levels, prefers
    ``<pair>_train_corner_filled.csv`` (C11 regen). Falls back to
    un-versioned baseline train if regen absent (legacy pre-C11
    variants, or closure-only pairs).
    """

    def _make_bundle_with_em_dir(self, tmp_path: Path, level: str) -> VariantBundle:
        em_dir = tmp_path / "input" / "entitymatching"
        em_dir.mkdir(parents=True)
        return VariantBundle(
            domain="games",
            level=level,
            sources={},
            target_schema={"title": "Game"},
            sm_mapping=None,
            em_gold={},
            em_splits={},
            em_gold_regenerated={},
            fusion_gold=pd.DataFrame(),
            fusion_validation=None,
            pooled_positives=None,
            variant_root=tmp_path,
        )

    def test_baseline_level_returns_unversioned_train(self, tmp_path: Path) -> None:
        bundle = self._make_bundle_with_em_dir(tmp_path, "baseline")
        em_dir = tmp_path / "input" / "entitymatching"
        (em_dir / "dbpedia_2_sales_train.csv").write_text("id1,id2,label\n")
        path, distinct = _resolve_variant_train_path(bundle, ("dbpedia", "sales"))
        assert path is not None
        assert path.name == "dbpedia_2_sales_train.csv"
        assert distinct is False

    def test_variant_with_corner_filled_returns_corner_filled(
        self, tmp_path: Path
    ) -> None:
        bundle = self._make_bundle_with_em_dir(tmp_path, "easy")
        em_dir = tmp_path / "input" / "entitymatching"
        (em_dir / "dbpedia_2_sales_train.csv").write_text("id1,id2,label\n")
        (em_dir / "dbpedia_2_sales_train_corner_filled.csv").write_text(
            "id1,id2,label\n"
        )
        path, distinct = _resolve_variant_train_path(bundle, ("dbpedia", "sales"))
        assert path is not None
        assert path.name == "dbpedia_2_sales_train_corner_filled.csv"
        assert distinct is True

    def test_variant_without_corner_filled_falls_back_to_baseline_train(
        self, tmp_path: Path
    ) -> None:
        bundle = self._make_bundle_with_em_dir(tmp_path, "easy")
        em_dir = tmp_path / "input" / "entitymatching"
        (em_dir / "dbpedia_2_sales_train.csv").write_text("id1,id2,label\n")
        # No corner_filled present.
        path, distinct = _resolve_variant_train_path(bundle, ("dbpedia", "sales"))
        assert path is not None
        assert path.name == "dbpedia_2_sales_train.csv"
        assert distinct is False

    def test_variant_corner_filled_in_reverse_direction(self, tmp_path: Path) -> None:
        """Reverse-direction corner_filled (e.g. games metacritic_dbpedia
        pair where the train CSV is metacritic_2_dbpedia_*) is also
        recognised, matching variant_loader's direction-tolerance."""
        bundle = self._make_bundle_with_em_dir(tmp_path, "easy")
        em_dir = tmp_path / "input" / "entitymatching"
        (em_dir / "metacritic_2_dbpedia_train_corner_filled.csv").write_text(
            "id1,id2,label\n"
        )
        path, distinct = _resolve_variant_train_path(bundle, ("dbpedia", "metacritic"))
        assert path is not None
        assert path.name == "metacritic_2_dbpedia_train_corner_filled.csv"
        assert distinct is True


# ---------------------------------------------------------------------------
# R10-F: end-to-end dual-test gold wiring smoke test
# ---------------------------------------------------------------------------


class TestR10FDualTestGoldWiring:
    """Smoke test: the EM matching runner scores f1_baseline_test against the
    ``baseline_pruned`` test gold and f1_regen_test against ``corner_filled``.

    Constructs a variant bundle whose two test-gold versions have a
    *deliberately different* label distribution so the two surfaces cannot
    accidentally agree, then runs the real ``EMMatchingCommitteeRunner`` and
    asserts the two aggregated surfaces are distinct and map to the right
    gold. Guards against a regression of the R10-F glob bug (which made every
    surface fall back to the same baseline gold).
    """

    @staticmethod
    def _regen_bundle() -> VariantBundle:
        sources, em_gold, pooled = _make_two_source_data(n=20)
        pair = ("source_a", "source_b")
        # baseline_pruned test: all-positive diagonal -> matcher (Jaccard 1.0
        # on identical names) predicts every pair correctly -> F1 = 1.0.
        baseline_pruned = pd.DataFrame(
            [("a_0", "b_0", "true"), ("a_1", "b_1", "true"), ("a_2", "b_2", "true")],
            columns=["id1", "id2", "label"],
        )
        # corner_filled test: same positives plus deceptive negatives the
        # all-match matcher predicts -> false positives -> F1 < 1.0.
        corner_filled = pd.DataFrame(
            [
                ("a_0", "b_0", "true"),
                ("a_1", "b_1", "true"),
                ("a_2", "b_2", "true"),
                ("a_3", "b_3", "false"),
                ("a_4", "b_4", "false"),
            ],
            columns=["id1", "id2", "label"],
        )
        em_gold_regenerated = {
            pair: {
                "test": {
                    "baseline_pruned": baseline_pruned,
                    "corner_filled": corner_filled,
                },
                "val": {"corner_filled": corner_filled},
            }
        }
        return _make_bundle(
            sources,
            em_gold,
            pooled,
            level="medium",
            em_gold_regenerated=em_gold_regenerated,
        )

    def test_baseline_and_regen_surfaces_are_distinct(self, tmp_path: Path) -> None:
        _, matching_path = _write_fixture_roster(tmp_path)
        runner = EMMatchingCommitteeRunner(matching_path)
        result = runner.run(self._regen_bundle())

        macro_bl = result.aggregated["macro_f1_baseline_test"]
        macro_rg = result.aggregated["macro_f1_regen_test"]

        # Both surfaces must be real (not NaN/degenerate)...
        assert not _is_nan_value(macro_bl)
        assert not _is_nan_value(macro_rg)
        # ...and DISTINCT — the central R10-F guarantee.
        assert macro_bl != macro_rg
        # baseline_pruned is all-positive (F1=1.0); corner_filled adds FPs
        # (F1<1.0), so the baseline surface must score strictly higher. If the
        # wiring were swapped, this ordering would flip.
        assert macro_bl > macro_rg
        assert macro_bl == pytest.approx(1.0)
        assert macro_rg < 1.0

    def test_per_pair_dual_test_keys_map_to_correct_gold(self, tmp_path: Path) -> None:
        _, matching_path = _write_fixture_roster(tmp_path)
        runner = EMMatchingCommitteeRunner(matching_path)
        result = runner.run(self._regen_bundle())

        # Inspect any member's per-pair detail for the source pair.
        member = next(iter(result.per_member.values()))
        per_pair = member.notes["per_pair"]
        pair_key = next(iter(per_pair))
        pm = per_pair[pair_key]

        # baseline_model_on_baseline_test uses baseline_pruned (all TP) = 1.0;
        # baseline_model_on_regen_test uses corner_filled (has FP) < 1.0.
        assert pm["f1_baseline_model_on_baseline_test"] == pytest.approx(1.0)
        assert pm["f1_baseline_model_on_regen_test"] < 1.0
        assert (
            pm["f1_baseline_model_on_baseline_test"]
            != pm["f1_baseline_model_on_regen_test"]
        )


def _is_nan_value(x: float) -> bool:
    return x != x


class TestResolveColumnMappingIdGuard:
    """`_resolve_column_mapping` rejects any mapping that touches the 'id'
    join key (hardcoded id_column='id' for every blocker/matcher)."""

    def test_normal_mapping_passes(self) -> None:
        m = {"dbpedia": {"org_name": "name", "nation": "country"}}
        assert _resolve_column_mapping(m, m) == m

    def test_empty_and_none_pass(self) -> None:
        assert _resolve_column_mapping(None, None) == {}
        assert _resolve_column_mapping({"s": {}}, None) == {"s": {}}

    def test_rename_id_away_raises(self) -> None:
        # {id: entity_id} would delete the join column.
        with pytest.raises(ValueError, match="rename the 'id' join key away"):
            _resolve_column_mapping({"s": {"id": "entity_id"}}, None)

    def test_rename_onto_id_raises(self) -> None:
        # {rel_id: id} would silently overwrite the real id values.
        with pytest.raises(ValueError, match="must not rename .* to 'id'"):
            _resolve_column_mapping(None, {"s": {"rel_id": "id"}})

    def test_identity_id_entry_is_allowed(self) -> None:
        # An explicit {id: id} is a no-op rename, not a corruption.
        assert _resolve_column_mapping({"s": {"id": "id"}}, {"s": {"id": "id"}}) == {
            "s": {"id": "id"}
        }
