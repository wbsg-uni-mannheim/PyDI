"""Per-stage runners for the best-of-breed pipeline.

Each runner takes the current :class:`PipelineState`, runs the
appropriate committee, picks a winner by validation-set score, threads
the winner's output into the state, and returns a
:class:`StageSelection` summarising what happened.

The committee runners are imported read-only from
:mod:`usecases_synthetic.lib`; no committee logic is reimplemented
here.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from PyDI.entitymatching.post_clustering.greedy_one_to_one import (
    GreedyOneToOneMatchingAlgorithm,
)
from PyDI.entitymatching.post_clustering.maximum_bipartite_matching import (
    MaximumBipartiteMatching,
)
from usecases_synthetic.lib.committee_em import (
    EMCommitteeRunner,
    score_em_correspondences_closed_set,
)
from usecases_synthetic.lib.committee_fusion import FusionCommitteeRunner
from usecases_synthetic.lib.committee_norm import NormCommitteeRunner
from usecases_synthetic.lib.committee_sm import SMCommitteeRunner

from ._resource_tracking import PeakRSSTracker
from .bundle import PipelineState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage result dataclass
# ---------------------------------------------------------------------------


@dataclass
class StageSelection:
    """Per-stage selection record.

    Parameters
    ----------
    stage : str
        Stage identifier (``"sm"``, ``"norm"``, ``"em_blocking"``,
        ``"em_matching"``, ``"refinement"``, ``"fusion"``).
    winner : str
        Winning member name. Empty string if no member produced a
        scoreable result.
    val_score : float
        Winning member's validation-set selection metric.
    test_score : float
        Winning member's held-out test-set metric. ``float('nan')``
        when there is no test surface for this stage.
    per_member_val : dict[str, float]
        Validation-set score per member.
    per_member_test : dict[str, float]
        Test-set score per member.
    metric_key : str
        Which metric key from the committee result drove selection.
    runtime_s : float
        Wall-clock runtime of the stage.
    notes : dict
        Stage-specific notes (e.g. blocker selection per pair,
        vacuous-flag, members that crashed).
    peak_memory_mb : float
        Peak resident set size observed while the stage executed, in
        MB. ``0.0`` when the tracker was a no-op (psutil missing) or
        the stage was skipped entirely.
    """

    stage: str
    winner: str
    val_score: float
    test_score: float
    per_member_val: dict[str, float] = field(default_factory=dict)
    per_member_test: dict[str, float] = field(default_factory=dict)
    metric_key: str = "f1"
    runtime_s: float = 0.0
    notes: dict[str, Any] = field(default_factory=dict)
    peak_memory_mb: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        """Plain-dict view for JSON serialisation."""
        return {
            "stage": self.stage,
            "winner": self.winner,
            "metric_key": self.metric_key,
            "val_score": self.val_score,
            "test_score": self.test_score,
            "per_member_val": self.per_member_val,
            "per_member_test": self.per_member_test,
            "runtime_s": self.runtime_s,
            "notes": self.notes,
            "peak_memory_mb": self.peak_memory_mb,
        }


# ---------------------------------------------------------------------------
# Selection helper
# ---------------------------------------------------------------------------


def _pick_winner(
    per_member: dict[str, float],
    *,
    higher_is_better: bool = True,
) -> str:
    """Pick the winning member.

    Deterministic tie-break: by member name (ascending).

    Parameters
    ----------
    per_member : dict[str, float]
        ``{member_name: score}``.
    higher_is_better : bool, default True
        ``True`` for F1 / accuracy; ``False`` for distance metrics.

    Returns
    -------
    str
        Winning member name. Empty string if no candidates.
    """
    if not per_member:
        return ""
    candidates = [(name, score) for name, score in per_member.items()]
    sign = -1.0 if higher_is_better else 1.0
    candidates.sort(key=lambda x: (sign * x[1], x[0]))
    return candidates[0][0]


# ---------------------------------------------------------------------------
# Stage 1: SM
# ---------------------------------------------------------------------------


def run_sm(
    state: PipelineState,
    *,
    sm_yaml: Path,
    with_llm: bool = False,
) -> StageSelection:
    """Run SM committee on the bundle and pick the highest-F1 member.

    The SM committee scores every member against
    ``bundle.sm_mapping`` (``sm_mapping_gold.csv``). There is no
    held-out SM test split for products baseline so ``test_score`` is
    set equal to ``val_score`` and the JSON output flags this.
    """
    t0 = time.monotonic()
    with PeakRSSTracker() as _rss:
        runner = SMCommitteeRunner(sm_yaml, with_llm=with_llm)
        result = runner.run(state.bundle)

        per_member_val = {
            name: float(m.metrics.get("f1", 0.0))
            for name, m in result.per_member.items()
        }
        # SM has no test surface separate from val; mirror for transparency.
        per_member_test = dict(per_member_val)

        winner = _pick_winner(per_member_val)
        val_score = per_member_val.get(winner, 0.0)
        test_score = per_member_test.get(winner, 0.0)

        if winner:
            state.sm_winner = winner
            state.sm_mapping_df = result.per_member[winner].predictions
            logger.info("SM winner: %s (f1=%.4f)", winner, val_score)
        else:
            logger.warning("SM: no member produced a scoreable result")

    return StageSelection(
        stage="sm",
        winner=winner,
        val_score=val_score,
        test_score=test_score,
        per_member_val=per_member_val,
        per_member_test=per_member_test,
        metric_key="f1",
        runtime_s=time.monotonic() - t0,
        notes={
            "test_eq_val": True,
            "test_eq_val_reason": "no held-out SM test split for baseline",
        },
        peak_memory_mb=_rss.peak_mb,
    )


# ---------------------------------------------------------------------------
# Stage 2: Norm
# ---------------------------------------------------------------------------


def run_norm(
    state: PipelineState,
    *,
    norm_yaml: Path,
    vacuous_epsilon: float = 0.005,
    apply_winner: bool = False,
    scoring_surface: str = "xml_targets",
) -> StageSelection:
    """Run Norm committee; pick the highest macro-F1 member.

    Per the plan §8.4, ``apply_winner=False`` means we score members
    on fusion-protection cells but do **not** transform downstream
    frames. If the spread across members is smaller than
    ``vacuous_epsilon``, the selection is flagged vacuous in the
    notes block.

    ``scoring_surface`` selects how each member's normalized output is
    scored. ``"xml_targets"`` (default) compares to per-entity fusion
    XML target values — the historical synthetic-side surface.
    ``"schema_constraints"`` checks that the output satisfies the
    JSON-Schema + ``x-pydi-consistency`` constraints declared in the
    canonical target_schema.json — recommended for canonical runs.
    """
    t0 = time.monotonic()
    with PeakRSSTracker() as _rss:
        runner = NormCommitteeRunner(norm_yaml, scoring_surface=scoring_surface)
        result = runner.run(state.bundle)

        per_member_val = {
            name: float(m.metrics.get("macro_f1", 0.0))
            for name, m in result.per_member.items()
        }
        per_member_test = dict(per_member_val)
        winner = _pick_winner(per_member_val)
        val_score = per_member_val.get(winner, 0.0)
        test_score = per_member_test.get(winner, 0.0)

        spread = (
            max(per_member_val.values()) - min(per_member_val.values())
            if per_member_val
            else 0.0
        )
        vacuous = spread < vacuous_epsilon

        if winner and apply_winner:
            logger.warning(
                "Norm apply_winner=True is not implemented in v1; downstream "
                "frames pass through untouched."
            )

        state.norm_winner = winner
        if winner:
            logger.info(
                "Norm winner: %s (macro_f1=%.4f, spread=%.4f, vacuous=%s)",
                winner,
                val_score,
                spread,
                vacuous,
            )

    return StageSelection(
        stage="norm",
        winner=winner,
        val_score=val_score,
        test_score=test_score,
        per_member_val=per_member_val,
        per_member_test=per_member_test,
        metric_key="macro_f1",
        runtime_s=time.monotonic() - t0,
        notes={
            "applied_to_downstream": False,
            "spread": spread,
            "vacuous": vacuous,
            "vacuous_epsilon": vacuous_epsilon,
        },
        peak_memory_mb=_rss.peak_mb,
    )


# ---------------------------------------------------------------------------
# Stage 3 + 4: EM (blocking + matching, joint via EMCommitteeRunner)
# ---------------------------------------------------------------------------


def _swap_em_gold(
    state: PipelineState, *, split: str
) -> dict[tuple[str, str], pd.DataFrame] | None:
    """Temporarily swap ``bundle.em_gold`` to point at the given split.

    The committee runners read ``bundle.em_gold`` as their headline
    test surface. For best-of-breed we want to drive selection from
    the val split and report test separately. We mutate the bundle's
    ``em_gold`` in place, save the previous value, and the caller
    restores it after running.

    Returns
    -------
    dict or None
        The previous ``em_gold`` mapping (for restoration). ``None``
        when no val/test split files exist (caller should fall through).
    """
    out: dict[tuple[str, str], pd.DataFrame] = {}
    for pair, splits in state.bundle.em_splits.items():
        if split in splits:
            out[pair] = splits[split]
    if not out:
        return None
    previous = state.bundle.em_gold
    state.bundle.em_gold = out
    return previous


def _restore_em_gold(
    state: PipelineState,
    previous: dict[tuple[str, str], pd.DataFrame] | None,
) -> None:
    """Restore ``bundle.em_gold`` after a temporary swap."""
    if previous is not None:
        state.bundle.em_gold = previous


def _build_em_blocking_selection(
    *,
    per_blocker: dict[str, Any],
    runtime_s: float,
    val_split_available: bool,
    test_split_available: bool,
    recall_floor: float = 0.97,
) -> tuple[StageSelection, dict[str, str]]:
    """Build the em_blocking ``StageSelection`` from a committee result.

    Primary metric is ``pair_completeness`` (recall — fraction of true
    gold pairs the blocker preserves). ``reduction_ratio`` is exposed
    as a secondary side metric under ``notes.per_member_reduction_ratio_*``.
    The per-pair selection logic in the committee still uses RR as a
    tiebreaker once the recall floor is cleared.

    Parameters
    ----------
    per_blocker : dict[str, MemberResult]
        ``val_result.per_blocker`` from the EM committee. Each entry's
        ``metrics`` dict must carry ``pair_completeness`` and
        ``reduction_ratio``; ``notes.per_pair`` carries per-pair
        ``selected`` flags.
    runtime_s : float
        Wall-clock seconds to attribute to this stage.
    val_split_available, test_split_available : bool
        Forwarded into the ``notes`` for traceability.
    recall_floor : float, default 0.97
        Surfaced under ``notes.recall_floor``; the floor itself is
        enforced inside the committee composition, not here.

    Returns
    -------
    (StageSelection, dict[str, str])
        The stage selection record and the ``{pair_key: blocker_name}``
        per-pair winner map (also stamped on ``PipelineState``).
    """
    blocker_per_pair: dict[str, str] = {}
    for blocker_name, m in per_blocker.items():
        per_pair = m.notes.get("per_pair", {})
        for pair_key, pair_metrics in per_pair.items():
            if pair_metrics.get("selected"):
                blocker_per_pair[pair_key] = blocker_name

    blocker_per_member_val_recall = {
        name: float(m.metrics.get("pair_completeness", 0.0))
        for name, m in per_blocker.items()
    }
    blocker_per_member_val_rr = {
        name: float(m.metrics.get("reduction_ratio", 0.0))
        for name, m in per_blocker.items()
    }
    # No separate test pass — blocker test scores mirror val (blockers
    # are evaluated on the same candidate-set surface for both val/test;
    # the test_split_available flag in notes captures the source).
    blocker_per_member_test_recall = dict(blocker_per_member_val_recall)
    blocker_per_member_test_rr = dict(blocker_per_member_val_rr)

    # Blocker "winner" is per-pair; the stage-level winner is the
    # one selected most often. Determined from blocker_per_pair tally.
    if blocker_per_pair:
        from collections import Counter

        counter = Counter(blocker_per_pair.values())
        blocker_winner = counter.most_common(1)[0][0]
    else:
        blocker_winner = _pick_winner(blocker_per_member_val_recall)

    logger.info(
        "EM blocker winner: %s (val_recall=%.4f, val_rr=%.4f)",
        blocker_winner,
        blocker_per_member_val_recall.get(blocker_winner, 0.0),
        blocker_per_member_val_rr.get(blocker_winner, 0.0),
    )

    selection = StageSelection(
        stage="em_blocking",
        winner=blocker_winner,
        val_score=blocker_per_member_val_recall.get(blocker_winner, 0.0),
        test_score=blocker_per_member_test_recall.get(blocker_winner, 0.0),
        per_member_val=blocker_per_member_val_recall,
        per_member_test=blocker_per_member_test_recall,
        metric_key="pair_completeness (>=0.97 floor; reduction_ratio tiebreak)",
        runtime_s=runtime_s,
        notes={
            "per_pair_winner": blocker_per_pair,
            "selection_strategy": (
                "composition (per-pair: pair_completeness >= 0.97, "
                "reduction_ratio tiebreak)"
            ),
            "val_split_available": val_split_available,
            "test_split_available": test_split_available,
            "recall_floor": recall_floor,
            "per_member_reduction_ratio_val": blocker_per_member_val_rr,
            "per_member_reduction_ratio_test": blocker_per_member_test_rr,
        },
    )
    return selection, blocker_per_pair


def run_em(
    state: PipelineState,
    *,
    blocking_yaml: Path,
    matching_yaml: Path,
    with_llm: bool = False,
    clustering: str = "greedy",
) -> tuple[StageSelection, StageSelection]:
    """Run EM blocking + matching committees on val, then on test.

    Returns two ``StageSelection`` records: one for ``em_blocking``
    and one for ``em_matching``. The blocking winner is picked
    per-pair by the existing composition logic in
    ``em_blocking_committee.yaml`` (``select_best`` w/ recall floor);
    we expose the per-pair winners under ``notes["per_pair_winner"]``.

    The matching winner is picked by val ``f1``; we also report each
    matcher's test ``f1`` for transparency. The winner's per-pair
    correspondences are stored on ``state.matcher_predictions``.
    """
    t0 = time.monotonic()

    _rss_ctx = PeakRSSTracker()
    _rss_ctx.__enter__()
    # First pass: val. Drives selection. Predictions retained for all
    # matchers so the winner's per-pair predictions are accessible.
    val_runner = EMCommitteeRunner(
        blocking_roster_path=blocking_yaml,
        matching_roster_path=matching_yaml,
        with_llm=with_llm,
        clustering=clustering,
        retain_predictions_for={
            spec.name for spec in _matcher_names_from_yaml(matching_yaml, with_llm)
        },
    )

    # 2026-05-28 bottleneck fix: only run the committee ONCE (against
    # val gold), then derive test scores by direct closed-set scoring
    # of the val-time predictions against per-pair test gold. The
    # second EMCommitteeRunner pass was doubling EM runtime (each
    # committee pass already scores against val_gold_corner +
    # test_gold_corner + test_gold_baseline internally), and the
    # synchronous post-training scoring loop was the wall-clock
    # bottleneck.
    prev_gold = _swap_em_gold(state, split="val")
    val_split_available = prev_gold is not None
    if not val_split_available:
        logger.warning(
            "No val EM split available for any pair; scoring against "
            "default em_gold (_all.csv or _test.csv) for selection."
        )
    try:
        val_result = val_runner.run(state.bundle)
    finally:
        _restore_em_gold(state, prev_gold)

    # Compute test scores by direct closed-set scoring of each
    # matcher's val-time per-pair predictions against the per-pair
    # test gold. Same scoring contract the committee uses internally,
    # without paying for blocker re-runs + auto-feature regeneration.
    test_per_member_f1: dict[str, float] = {}
    test_split_available = False
    for name, member in val_result.per_member.items():
        preds = member.predictions
        if not isinstance(preds, dict):
            test_per_member_f1[name] = float("nan")
            continue
        pair_f1s: list[float] = []
        for pair, splits in state.bundle.em_splits.items():
            if "test" not in splits:
                continue
            test_split_available = True
            pair_key = f"{pair[0]}_{pair[1]}"
            pair_preds = preds.get(pair_key)
            if pair_preds is None or pair_preds.empty:
                pair_f1s.append(0.0)
                continue
            metrics = score_em_correspondences_closed_set(pair_preds, splits["test"])
            pair_f1s.append(float(metrics.get("f1", 0.0)))
        test_per_member_f1[name] = (
            sum(pair_f1s) / len(pair_f1s) if pair_f1s else float("nan")
        )

    # --- Blocker selection summary (already chosen by the committee
    # composition; here we expose it for the per-stage record). ---
    blocking_selection, blocker_per_pair = _build_em_blocking_selection(
        per_blocker=val_result.per_blocker,
        runtime_s=time.monotonic() - t0,
        val_split_available=val_split_available,
        test_split_available=test_split_available,
    )
    state.blocker_winner_per_pair = blocker_per_pair

    # --- Matcher selection ---
    matcher_per_member_val = {
        name: float(m.metrics.get("f1", 0.0))
        for name, m in val_result.per_member.items()
    }
    # test_per_member_f1 was computed above by direct closed-set scoring
    # of val-time per-pair predictions against per-pair test gold.
    matcher_per_member_test = {
        name: float(v) if v == v else 0.0  # NaN-safe
        for name, v in test_per_member_f1.items()
    }
    matcher_winner = _pick_winner(matcher_per_member_val)

    if matcher_winner:
        predictions = val_result.per_member[matcher_winner].predictions
        if isinstance(predictions, dict):
            state.matcher_predictions = predictions
        else:
            logger.warning(
                "Matcher %s did not retain per-pair predictions; "
                "downstream stages will have no correspondences",
                matcher_winner,
            )
        state.matcher_winner = matcher_winner
        logger.info(
            "EM matcher winner: %s (val_f1=%.4f, test_f1=%.4f)",
            matcher_winner,
            matcher_per_member_val[matcher_winner],
            matcher_per_member_test.get(matcher_winner, float("nan")),
        )

    matching_selection = StageSelection(
        stage="em_matching",
        winner=matcher_winner,
        val_score=matcher_per_member_val.get(matcher_winner, 0.0),
        test_score=matcher_per_member_test.get(matcher_winner, 0.0),
        per_member_val=matcher_per_member_val,
        per_member_test=matcher_per_member_test,
        metric_key="f1",
        runtime_s=time.monotonic() - t0,
        notes={
            "val_split_available": val_split_available,
            "test_split_available": test_split_available,
            "clustering_applied_within_em": clustering,
        },
    )

    _rss_ctx.__exit__(None, None, None)
    # The EM committee runs blocking + matching back-to-back inside
    # the same tracked window. Apply the shared peak to both records.
    em_peak_mb = _rss_ctx.peak_mb
    blocking_selection.peak_memory_mb = em_peak_mb
    matching_selection.peak_memory_mb = em_peak_mb

    return blocking_selection, matching_selection


def _matcher_names_from_yaml(yaml_path: Path, with_llm: bool) -> list[Any]:
    """Read matcher names from the YAML for ``retain_predictions_for``.

    Returns a list of objects with a ``name`` attribute (matches the
    shape EMCommitteeRunner expects via ``set(spec.name for ...)``).
    """
    import yaml

    raw = yaml.safe_load(yaml_path.read_text())
    out = []

    class _Holder:
        def __init__(self, n: str) -> None:
            self.name = n

    for entry in raw.get("members", []):
        if not entry.get("enabled_by_default", True):
            continue
        if entry.get("matching_type") == "llm" and not with_llm:
            continue
        out.append(_Holder(entry["name"]))
    return out


# ---------------------------------------------------------------------------
# Stage 5: Post-clustering refinement
# ---------------------------------------------------------------------------


def run_refinement(
    state: PipelineState,
    *,
    methods: list[str],
) -> StageSelection:
    """Compete refinement methods on the winning matcher's per-pair output.

    Methods supported: ``baseline`` (no refinement), ``greedy``,
    ``mbm``. Selection is by val F1 against the per-pair val gold;
    test F1 against per-pair test gold is reported alongside.
    """
    t0 = time.monotonic()
    _rss_ctx = PeakRSSTracker()
    _rss_ctx.__enter__()

    if not state.matcher_predictions:
        logger.warning("Refinement stage: no matcher predictions in state; skipping.")
        _rss_ctx.__exit__(None, None, None)
        return StageSelection(
            stage="refinement",
            winner="",
            val_score=0.0,
            test_score=0.0,
            metric_key="f1",
            runtime_s=time.monotonic() - t0,
            notes={"skipped_reason": "no matcher predictions"},
            peak_memory_mb=_rss_ctx.peak_mb,
        )

    refiners: dict[str, Any] = {}
    if "baseline" in methods:
        refiners["baseline"] = None
    if "greedy" in methods:
        refiners["greedy"] = GreedyOneToOneMatchingAlgorithm()
    if "mbm" in methods:
        refiners["mbm"] = MaximumBipartiteMatching()
    if not refiners:
        raise ValueError(f"No refinement methods enabled (got {methods!r})")

    # Gather val + test gold per pair.
    val_gold_per_pair: dict[str, pd.DataFrame] = {}
    test_gold_per_pair: dict[str, pd.DataFrame] = {}
    for pair, splits in state.bundle.em_splits.items():
        pair_key = f"{pair[0]}_{pair[1]}"
        if "val" in splits:
            val_gold_per_pair[pair_key] = splits["val"]
        if "test" in splits:
            test_gold_per_pair[pair_key] = splits["test"]

    per_member_val: dict[str, float] = {}
    per_member_test: dict[str, float] = {}
    refined_outputs: dict[str, dict[str, pd.DataFrame]] = {}

    for method_name, refiner in refiners.items():
        per_pair_refined: dict[str, pd.DataFrame] = {}
        val_f1s: list[float] = []
        test_f1s: list[float] = []
        for pair_key, preds in state.matcher_predictions.items():
            if refiner is None:
                refined = preds
            else:
                # Post-clusterer expects an id1/id2/score frame; emit the
                # same shape back.
                refined = refiner.cluster(preds)
            per_pair_refined[pair_key] = refined

            if pair_key in val_gold_per_pair:
                val_metrics = score_em_correspondences_closed_set(
                    refined, val_gold_per_pair[pair_key]
                )
                val_f1s.append(float(val_metrics.get("f1", 0.0)))
            if pair_key in test_gold_per_pair:
                test_metrics = score_em_correspondences_closed_set(
                    refined, test_gold_per_pair[pair_key]
                )
                test_f1s.append(float(test_metrics.get("f1", 0.0)))

        refined_outputs[method_name] = per_pair_refined
        per_member_val[method_name] = sum(val_f1s) / len(val_f1s) if val_f1s else 0.0
        per_member_test[method_name] = (
            sum(test_f1s) / len(test_f1s) if test_f1s else 0.0
        )

    winner = _pick_winner(per_member_val)
    val_score = per_member_val.get(winner, 0.0)
    test_score = per_member_test.get(winner, 0.0)

    if winner:
        state.refinement_winner = winner
        # Concatenate per-pair refined outputs into a single
        # correspondences frame for fusion. Cast ids back to native
        # types in case post-clustering upcast them.
        all_dfs: list[pd.DataFrame] = []
        for pair_key, refined in refined_outputs[winner].items():
            if refined is None or refined.empty:
                continue
            all_dfs.append(refined)
        if all_dfs:
            corr = pd.concat(all_dfs, ignore_index=True)
        else:
            corr = pd.DataFrame(columns=["id1", "id2", "score"])
        state.correspondences = corr
        logger.info(
            "Refinement winner: %s (val_f1=%.4f, test_f1=%.4f, n_correspondences=%d)",
            winner,
            val_score,
            test_score,
            len(state.correspondences),
        )

    _rss_ctx.__exit__(None, None, None)
    return StageSelection(
        stage="refinement",
        winner=winner,
        val_score=val_score,
        test_score=test_score,
        per_member_val=per_member_val,
        per_member_test=per_member_test,
        metric_key="f1",
        runtime_s=time.monotonic() - t0,
        notes={
            "methods_competed": list(refiners.keys()),
            "n_correspondences": (
                len(state.correspondences) if state.correspondences is not None else 0
            ),
        },
        peak_memory_mb=_rss_ctx.peak_mb,
    )


# ---------------------------------------------------------------------------
# Stage 6: Fusion
# ---------------------------------------------------------------------------


def run_fusion(
    state: PipelineState,
    *,
    fusion_yaml: Path,
) -> StageSelection:
    """Run Fusion committee with the winning correspondences as input.

    Passes ``state.correspondences`` (the refined EM output) directly
    to :meth:`FusionCommitteeRunner.run` as the ``correspondences``
    argument so the runner clusters records from the EM winner's
    output rather than from gold-derived perfect clusters.

    The fusion runner only computes a test-side ``macro_accuracy``
    against ``bundle.fusion_gold``. We compute val ``macro_accuracy``
    here by re-scoring each member's predictions against
    ``bundle.fusion_validation`` via
    :func:`committee_fusion_scoring.score_fusion`. Selection is val.
    """
    t0 = time.monotonic()
    _rss_ctx = PeakRSSTracker()
    _rss_ctx.__enter__()

    if state.correspondences is None or state.correspondences.empty:
        logger.warning("Fusion stage: no correspondences in state; skipping.")
        _rss_ctx.__exit__(None, None, None)
        return StageSelection(
            stage="fusion",
            winner="",
            val_score=0.0,
            test_score=0.0,
            metric_key="macro_accuracy",
            runtime_s=time.monotonic() - t0,
            notes={"skipped_reason": "no correspondences"},
            peak_memory_mb=_rss_ctx.peak_mb,
        )

    runner = FusionCommitteeRunner(fusion_yaml)
    result = runner.run(state.bundle, correspondences=state.correspondences)

    # The runner's per_member metrics are scored against bundle.fusion_gold
    # (= test_set.xml). We score val ourselves below.
    from usecases_synthetic.lib.committee_fusion_scoring import score_fusion

    # Resolve the roster's eval_specs/eval_params + id columns.
    roster = runner._roster  # private but stable across both legacy and C12 runners
    eval_specs = roster.eval_specs
    eval_params = roster.eval_params
    fused_id_column = roster.fused_id_column
    gold_id_column = roster.gold_id_column

    val_gold_df = state.bundle.fusion_validation
    val_available = val_gold_df is not None and not val_gold_df.empty

    per_member_val: dict[str, float] = {}
    per_member_test: dict[str, float] = {}
    per_member_predictions: dict[str, pd.DataFrame] = {}
    for name, m in result.per_member.items():
        test_metrics = m.metrics or {}
        per_member_test[name] = float(test_metrics.get("macro_accuracy", 0.0))
        if isinstance(m.predictions, pd.DataFrame) and not m.predictions.empty:
            per_member_predictions[name] = m.predictions
            if val_available:
                try:
                    val_metrics = score_fusion(
                        fused_df=m.predictions,
                        gold_df=val_gold_df,
                        eval_specs=eval_specs,
                        eval_params=eval_params,
                        fused_id_column=fused_id_column,
                        gold_id_column=gold_id_column,
                    )
                    per_member_val[name] = float(val_metrics.get("macro_accuracy", 0.0))
                except Exception:
                    logger.exception(
                        "score_fusion failed on val for member %s; falling "
                        "back to test_score as val_score",
                        name,
                    )
                    per_member_val[name] = per_member_test[name]
            else:
                per_member_val[name] = per_member_test[name]
        else:
            per_member_val[name] = 0.0

    winner = _pick_winner(per_member_val)
    val_score = per_member_val.get(winner, 0.0)
    test_score = per_member_test.get(winner, 0.0)

    if winner and winner in per_member_predictions:
        state.fusion_winner = winner
        state.fused = per_member_predictions[winner]
        logger.info(
            "Fusion winner: %s (val_macro_acc=%.4f, test_macro_acc=%.4f, " "n_rows=%d)",
            winner,
            val_score,
            test_score,
            len(state.fused),
        )

    _rss_ctx.__exit__(None, None, None)
    return StageSelection(
        stage="fusion",
        winner=winner,
        val_score=val_score,
        test_score=test_score,
        per_member_val=per_member_val,
        per_member_test=per_member_test,
        metric_key="macro_accuracy",
        runtime_s=time.monotonic() - t0,
        notes={
            "val_surface_available": val_available,
            "n_fused_rows": (len(state.fused) if state.fused is not None else 0),
            "n_correspondences_used": len(state.correspondences),
        },
        peak_memory_mb=_rss_ctx.peak_mb,
    )


__all__ = [
    "StageSelection",
    "run_sm",
    "run_norm",
    "run_em",
    "run_refinement",
    "run_fusion",
]
