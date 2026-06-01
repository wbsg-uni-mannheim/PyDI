"""Pool-builder primitives for plan_s1_scale.md R3.

Builds the per-domain "likely positive" protection pool by combining

1. **Human EM gold positives** — all rows with label True across the
   train/val/test splits in ``usecases/<domain>/input/entitymatching/``.
2. **Human baseline pipeline correspondences** — every pair the matcher
   in the domain's Jupyter workflow notebook emitted, persisted at
   ``usecases/<domain>/output/debug_results_entity_matching/matching_detailed_results.csv``.
3. **Ditto PLM predictions** — softmax probabilities from the per-domain
   R2 checkpoint (``cache/ditto_checkpoints/<domain>/best``) over the
   union of (blocker output, human-baseline pairs, EM gold pairs).

The resulting buckets feed the pool with three certainty levels:

- **A (gold)**: kept unconditionally.
- **B (agreement)**: kept iff human-baseline AND Ditto agree (Ditto
  score >= theta).
- **C (singleton)**: exactly one of the two methods declared positive.
  Adjudicated by the existing PLM-based gate (Ditto score with LLM
  adjudicator on the margin band ``[theta - delta, theta + delta]``).

The blocker that feeds the Ditto candidate set is selected per source
pair under the same policy used by the EM blocking committee
(``recall_floor=0.97``, ``tie_breaker=reduction_ratio``).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from PyDI.entitymatching.blocking.embedding import EmbeddingBlocker
from PyDI.entitymatching.blocking.sorted_neighbourhood import (
    SortedNeighbourhoodBlocker,
)
from PyDI.entitymatching.blocking.standard import StandardBlocker
from PyDI.entitymatching.blocking.token_blocking import TokenBlocker

from .bm25_blocker import BM25Blocker
from .ditto_matcher import DittoMatcher
from .llm_cache import LLMCache, LLMCacheMiss
from .loaders import read_em_gold_csv

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DITTO_THRESHOLD: float = 0.5
RECALL_FLOOR: float = 0.97
RECALL_FALLBACK_MARGIN: float = 0.05
# Hard cap on candidate-set size when picking a winner. Even if a
# blocker clears RECALL_FLOOR, we won't pick it if its candidate set
# is larger than this — Ditto inference at 400 rows/sec on MPS makes
# multi-million candidate sets impractical for the pool's purpose.
# When the cap demotes the floor-clearing winner, the fallback picker
# (recall within RECALL_FALLBACK_MARGIN of max, max RR) takes over.
MAX_CANDIDATES_PREF: int = 1_000_000
DELTA_FLOOR: float = 0.05
DELTA_CAP: float = 0.20
DELTA_PERCENTILE: float = 90.0
POOL_PROMPT_VERSION: str = "r3_pool_v1"

POOL_ADJUDICATOR_PROMPT: str = (
    "You are deduplicating two records from different data sources. "
    "Decide whether they describe the same real-world entity.\n\n"
    "Record A: {record_a}\n"
    "Record B: {record_b}\n\n"
    "Answer with a single token: yes or no."
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


Pair = tuple[str, str]


@dataclass
class BlockerCandidate:
    """One entry in the per-source-pair blocker sweep."""

    name: str
    blocker_factory: Callable[[pd.DataFrame, pd.DataFrame], Any]


@dataclass
class BlockerResult:
    """Outcome of running one blocker for one source pair."""

    name: str
    candidates: pd.DataFrame  # columns: id1, id2
    pair_recall: float
    reduction_ratio: float
    n_candidates: int
    n_total_possible: int
    n_gold_positives_seen: int
    n_gold_positives_total: int


@dataclass
class BlockerSweepResult:
    """Aggregate sweep outcome for one source pair."""

    winner: str
    cleared_floor: bool
    per_blocker: dict[str, dict[str, Any]]
    candidates: pd.DataFrame  # winner's candidate set


@dataclass
class BucketResult:
    """Per-source-pair bucket assembly outcome.

    Bucket-C policy: **every disagreement goes through the LLM
    adjudicator** (plan_revision_step4g_findings.md follow-up,
    2026-05-26). The legacy ``score >= theta + delta`` auto-include
    and ``score < theta - delta`` auto-drop paths were removed because
    Ditto's per-domain precision (e.g. games dbpedia_sales: 0.43) is
    not high enough to overrule the human-baseline matcher on
    confidence alone.

    The ``_confident`` counters are kept on the dataclass for
    backwards compatibility with downstream readers but they will
    always be zero post-policy-change. ``delta_used`` remains as a
    telemetry record of the would-have-been margin band.
    """

    pool_df: pd.DataFrame  # final kept positives
    bucket_a: int  # gold positives
    bucket_b: int  # agreement positives
    bucket_c_total: int  # all disagreements that hit the LLM adjudicator
    bucket_c_kept_confident: int  # legacy; always 0 under current policy
    bucket_c_kept_llm: int  # LLM said yes on a disagreement
    bucket_c_dropped_confident: int  # legacy; always 0 under current policy
    bucket_c_dropped_llm: int  # LLM said no on a disagreement
    delta_used: float


# ---------------------------------------------------------------------------
# Source-pair canonicalisation
# ---------------------------------------------------------------------------


def canonical_pair(a: str, b: str) -> Pair:
    """Return ``(id1, id2)`` sorted lexicographically.

    Pool semantics treat ``(a, b)`` and ``(b, a)`` as the same pair.
    """
    return (a, b) if a <= b else (b, a)


def apply_column_mapping(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Rename ``df``'s columns per ``mapping``; preserve ``df.attrs``.

    Columns absent from the mapping are kept under their original name.
    """
    if not mapping:
        out = df.copy()
        out.attrs = df.attrs.copy()
        return out
    keep = {old: new for old, new in mapping.items() if old in df.columns}
    out = df.rename(columns=keep)
    out.attrs = df.attrs.copy()
    return out


# ---------------------------------------------------------------------------
# Human EM gold loading
# ---------------------------------------------------------------------------


def _label_to_bool(value: str) -> bool | None:
    """Interpret an EM gold label cell as a bool, or ``None`` if unparseable."""
    s = str(value).strip().lower()
    if s in {"true", "1", "yes", "t"}:
        return True
    if s in {"false", "0", "no", "f"}:
        return False
    return None


@dataclass
class GoldSplit:
    """One source-pair's combined gold splits."""

    positives: set[Pair]
    all_pairs: set[Pair]
    val_pairs: pd.DataFrame  # for delta estimation; cols: id1, id2, label


def load_em_gold_for_pair(
    em_dir: Path, files: Sequence[str], val_files: Sequence[str] | None = None
) -> GoldSplit:
    """Load and combine EM gold CSVs for one source pair.

    Parameters
    ----------
    em_dir : Path
        Directory holding the EM gold CSVs.
    files : sequence of str
        Filenames whose union forms the pair's full gold set.
    val_files : sequence of str or None
        Subset of ``files`` (or separate filenames) that should be
        treated as the validation split for delta estimation. When
        ``None``, falls back to the first file in ``files``.

    Returns
    -------
    GoldSplit
        Pair-canonicalised positive and full pair sets, plus the
        validation DataFrame (``id1``, ``id2``, ``label``).
    """
    positives: set[Pair] = set()
    all_pairs: set[Pair] = set()
    for fname in files:
        path = em_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"Missing EM gold file: {path}")
        df = read_em_gold_csv(path)
        for row in df.itertuples(index=False):
            label = _label_to_bool(row.label)
            if label is None:
                continue
            pair = canonical_pair(str(row.id1), str(row.id2))
            all_pairs.add(pair)
            if label:
                positives.add(pair)

    val_source = val_files if val_files else [files[0]]
    val_frames: list[pd.DataFrame] = []
    for fname in val_source:
        path = em_dir / fname
        if not path.exists():
            continue
        df = read_em_gold_csv(path)
        df["__label_bool"] = df["label"].map(_label_to_bool)
        df = df[df["__label_bool"].notna()][["id1", "id2", "__label_bool"]]
        df = df.rename(columns={"__label_bool": "label"})
        val_frames.append(df)
    val_df = (
        pd.concat(val_frames, ignore_index=True)
        if val_frames
        else pd.DataFrame(columns=["id1", "id2", "label"])
    )
    return GoldSplit(positives=positives, all_pairs=all_pairs, val_pairs=val_df)


# ---------------------------------------------------------------------------
# Human baseline correspondences
# ---------------------------------------------------------------------------


def load_human_baseline_pairs_from_files(
    correspondence_csvs: Iterable[Path],
    source_id_to_source: Callable[[str], str],
) -> dict[frozenset[str], set[Pair]]:
    """Load per-source-pair correspondence CSVs into a pair-keyed dict.

    Each CSV is the output of one ``RuleBasedMatcher.match`` invocation
    (notebook-emitted via the auto-inserted "persist per-source-pair
    correspondences" cell — see ``usecases/<d>/<d>_workflow.ipynb``).
    Files have ``id1, id2, score, notes`` columns; we only need
    ``id1`` / ``id2``. Pairs are canonicalised and grouped by their
    unordered source-pair key. Same-source pairs and pairs whose
    endpoints don't resolve to a known source are dropped.

    Parameters
    ----------
    correspondence_csvs : iterable of Path
        Paths to per-source-pair correspondence CSVs.
    source_id_to_source : callable
        Maps an entity id string to its source label.

    Returns
    -------
    dict mapping ``frozenset({source_a, source_b}) -> set[Pair]``.
    """
    out: dict[frozenset[str], set[Pair]] = {}
    for path in correspondence_csvs:
        if not path.exists():
            logger.warning("missing correspondence file: %s", path)
            continue
        df = pd.read_csv(path)
        if "id1" not in df.columns or "id2" not in df.columns:
            logger.warning(
                "%s missing id1/id2 columns; got %s — skipping",
                path,
                list(df.columns),
            )
            continue
        for row in df.itertuples(index=False):
            id1 = str(row.id1)
            id2 = str(row.id2)
            s1 = source_id_to_source(id1)
            s2 = source_id_to_source(id2)
            if s1 == s2 or s1 == "unknown" or s2 == "unknown":
                continue
            key = frozenset({s1, s2})
            out.setdefault(key, set()).add(canonical_pair(id1, id2))
    return out


# ---------------------------------------------------------------------------
# Blocker sweep
# ---------------------------------------------------------------------------


def make_blocker_candidates(
    name_column: str = "name",
    *,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    embedding_top_k: int = 10,
    embedding_threshold: float = 0.3,
    bm25_top_k: int = 10,
    sn_window: int = 20,
    output_dir: str = "/tmp/pool_builder_blocker_debug",
) -> list[BlockerCandidate]:
    """Roster of blockers tried per source pair.

    Mirrors the enabled members of
    ``config/committees/em_blocking_committee*.yaml`` (token, standard,
    embedding, sorted_neighbourhood, bm25). The BlockerCandidate's
    factory is invoked with ``(left_df, right_df)`` and is responsible
    for any per-blocker preprocessing (e.g. derived blocking-key
    column).
    """

    def _add_first_token(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["name_first_token"] = (
            out[name_column]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.split()
            .str[0]
            .fillna("")
        )
        return out

    return [
        BlockerCandidate(
            "token_blocker",
            lambda l, r: TokenBlocker(
                l,
                r,
                column=name_column,
                id_column="id",
                min_token_len=2,
                output_dir=output_dir,
            ),
        ),
        BlockerCandidate(
            "standard_blocker",
            lambda l, r: StandardBlocker(
                _add_first_token(l),
                _add_first_token(r),
                on=["name_first_token"],
                id_column="id",
                output_dir=output_dir,
            ),
        ),
        BlockerCandidate(
            "embedding_blocker",
            lambda l, r: EmbeddingBlocker(
                l,
                r,
                text_cols=[name_column],
                id_column="id",
                model=embedding_model,
                top_k=embedding_top_k,
                threshold=embedding_threshold,
                output_dir=output_dir,
            ),
        ),
        BlockerCandidate(
            "sorted_neighbourhood_blocker",
            lambda l, r: SortedNeighbourhoodBlocker(
                l,
                r,
                key=name_column,
                id_column="id",
                window=sn_window,
                output_dir=output_dir,
            ),
        ),
        BlockerCandidate(
            "bm25_blocker",
            lambda l, r: BM25Blocker(
                l,
                r,
                text_cols=[name_column],
                id_column="id",
                top_k=bm25_top_k,
            ),
        ),
    ]


def evaluate_blocker(
    blocker: Any,
    *,
    gold_positives: set[Pair],
    n_left: int,
    n_right: int,
) -> tuple[pd.DataFrame, BlockerResult]:
    """Materialise a blocker and compute its recall + reduction ratio.

    Parameters
    ----------
    blocker : BaseBlocker
        Instantiated blocker.
    gold_positives : set of canonicalised pairs
        Reference positives for recall.
    n_left, n_right : int
        Source row counts (Cartesian denominator for RR).

    Returns
    -------
    tuple
        ``(candidates_df, result)``. ``candidates_df`` has columns
        ``id1``, ``id2``.
    """
    candidates = blocker.materialize()
    if candidates.empty:
        candidates = pd.DataFrame(columns=["id1", "id2"])
    else:
        candidates = candidates[["id1", "id2"]].astype(str).drop_duplicates()
    seen: set[Pair] = {
        canonical_pair(r.id1, r.id2) for r in candidates.itertuples(index=False)
    }
    n_total_possible = max(1, n_left * n_right)
    n_candidates = len(seen)
    n_gold_seen = sum(1 for p in gold_positives if p in seen)
    n_gold_total = len(gold_positives)
    pair_recall = n_gold_seen / n_gold_total if n_gold_total else 0.0
    reduction_ratio = 1.0 - n_candidates / n_total_possible
    return candidates, BlockerResult(
        name=getattr(blocker, "__class__", type(blocker)).__name__,
        candidates=candidates,
        pair_recall=pair_recall,
        reduction_ratio=reduction_ratio,
        n_candidates=n_candidates,
        n_total_possible=n_total_possible,
        n_gold_positives_seen=n_gold_seen,
        n_gold_positives_total=n_gold_total,
    )


def sweep_blockers(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    gold_positives: set[Pair],
    blocker_candidates: list[BlockerCandidate],
) -> BlockerSweepResult:
    """Try every blocker; pick winner per recall_floor + tie_breaker.

    Composition strategy mirrors
    ``em_blocking_committee*.yaml``: ``select_best`` with
    ``recall_floor=0.97`` and ``tie_breaker=reduction_ratio``. When no
    blocker clears the floor, the highest-recall blocker is selected
    (RR-tie-broken) and ``cleared_floor=False`` is reported so the
    caller can warn / record.
    """
    per_blocker: dict[str, dict[str, Any]] = {}
    candidate_frames: dict[str, pd.DataFrame] = {}
    n_left = len(df_left)
    n_right = len(df_right)

    for cand in blocker_candidates:
        try:
            blocker = cand.blocker_factory(df_left, df_right)
            cands, result = evaluate_blocker(
                blocker,
                gold_positives=gold_positives,
                n_left=n_left,
                n_right=n_right,
            )
        except Exception as exc:
            logger.warning("blocker %s failed: %s", cand.name, exc)
            per_blocker[cand.name] = {"error": str(exc)}
            continue
        candidate_frames[cand.name] = cands
        per_blocker[cand.name] = {
            "pair_recall": result.pair_recall,
            "reduction_ratio": result.reduction_ratio,
            "n_candidates": result.n_candidates,
            "n_gold_positives_seen": result.n_gold_positives_seen,
            "n_gold_positives_total": result.n_gold_positives_total,
        }
        logger.info(
            "  %s: recall=%.4f rr=%.6f cands=%d (gold %d/%d)",
            cand.name,
            result.pair_recall,
            result.reduction_ratio,
            result.n_candidates,
            result.n_gold_positives_seen,
            result.n_gold_positives_total,
        )

    if not candidate_frames:
        raise RuntimeError("All blockers failed; cannot select a winner")

    # Floor-clearing AND under the hard candidate-count cap. The cap
    # protects against picking a blocker like ``token_blocker`` whose
    # candidate set runs into the millions even when its recall is
    # >= RECALL_FLOOR — Ditto inference cost makes that impractical.
    qualifying = [
        n
        for n, m in per_blocker.items()
        if "error" not in m
        and m["pair_recall"] >= RECALL_FLOOR
        and m["n_candidates"] <= MAX_CANDIDATES_PREF
    ]
    if qualifying:
        winner = max(qualifying, key=lambda n: per_blocker[n]["reduction_ratio"])
        cleared = True
    else:
        # Fallback: prefer the smallest candidate set among blockers
        # within RECALL_FALLBACK_MARGIN of the floor (default 5pp).
        # Picking purely on max-recall here gravitates to whatever
        # blocker emits the most candidates (e.g. token_blocker), which
        # explodes downstream Ditto inference cost. Capping at
        # near-floor recall with min-RR-loss is a better trade-off for
        # the pool's protection-set use case.
        valid = [n for n, m in per_blocker.items() if "error" not in m]
        max_recall = max(per_blocker[n]["pair_recall"] for n in valid)
        margin_threshold = max(0.0, max_recall - RECALL_FALLBACK_MARGIN)
        contenders = [
            n
            for n in valid
            if per_blocker[n]["pair_recall"] >= margin_threshold
            and per_blocker[n]["n_candidates"] <= MAX_CANDIDATES_PREF
        ]
        if not contenders:
            # No within-margin blocker fits under the cap. Demote to
            # the highest-recall blocker that DOES fit under the cap,
            # even if its recall drops several pp below the in-margin
            # set — Ditto inference cost on millions of pairs is too
            # high to justify a few extra recall points. Recovery on
            # the missed pairs comes from the human-baseline + Ditto
            # transitive closure across the domain's other source
            # pairs.
            under_cap = [
                n
                for n in valid
                if per_blocker[n]["n_candidates"] <= MAX_CANDIDATES_PREF
            ]
            if under_cap:
                winner = max(
                    under_cap,
                    key=lambda n: (
                        per_blocker[n]["pair_recall"],
                        per_blocker[n]["reduction_ratio"],
                    ),
                )
                logger.warning(
                    "  no blocker within %.2fpp of max-recall %.4f fits "
                    "under cap=%d candidates; demoted to '%s' (recall=%.4f, "
                    "cands=%d) — pool relies on cross-pair transitive closure "
                    "for recovery",
                    RECALL_FALLBACK_MARGIN,
                    max_recall,
                    MAX_CANDIDATES_PREF,
                    winner,
                    per_blocker[winner]["pair_recall"],
                    per_blocker[winner]["n_candidates"],
                )
            else:
                # Every blocker exceeds the cap. Take the smallest
                # candidate set as a last resort.
                winner = min(valid, key=lambda n: per_blocker[n]["n_candidates"])
                logger.warning(
                    "  every blocker exceeds cap=%d; fell back to "
                    "smallest-candidate '%s' (recall=%.4f, cands=%d)",
                    MAX_CANDIDATES_PREF,
                    winner,
                    per_blocker[winner]["pair_recall"],
                    per_blocker[winner]["n_candidates"],
                )
        else:
            winner = max(contenders, key=lambda n: per_blocker[n]["reduction_ratio"])
            logger.warning(
                "  no blocker cleared recall_floor=%.2f; "
                "fell back to '%s' (recall=%.4f, rr=%.6f) — best RR among "
                "blockers within %.2fpp of max-recall %.4f under cap=%d",
                RECALL_FLOOR,
                winner,
                per_blocker[winner]["pair_recall"],
                per_blocker[winner]["reduction_ratio"],
                RECALL_FALLBACK_MARGIN,
                max_recall,
                MAX_CANDIDATES_PREF,
            )
        cleared = False

    return BlockerSweepResult(
        winner=winner,
        cleared_floor=cleared,
        per_blocker=per_blocker,
        candidates=candidate_frames[winner],
    )


# ---------------------------------------------------------------------------
# Candidate-set assembly
# ---------------------------------------------------------------------------


def assemble_candidate_set(
    blocker_pairs: pd.DataFrame,
    human_pairs: set[Pair],
    gold_pairs: set[Pair],
) -> pd.DataFrame:
    """Combine blocker + human-baseline + gold pairs into one set.

    Returns a deterministically-ordered DataFrame with columns ``id1``,
    ``id2``. Order matters for the DittoMatcher inference cache key.
    """
    pairs: set[Pair] = set()
    if not blocker_pairs.empty:
        for row in blocker_pairs.itertuples(index=False):
            pairs.add(canonical_pair(str(row.id1), str(row.id2)))
    pairs.update(human_pairs)
    pairs.update(gold_pairs)

    if not pairs:
        return pd.DataFrame(columns=["id1", "id2"])
    sorted_pairs = sorted(pairs)
    return pd.DataFrame(sorted_pairs, columns=["id1", "id2"])


# ---------------------------------------------------------------------------
# Ditto inference
# ---------------------------------------------------------------------------


def score_with_ditto(
    matcher: DittoMatcher,
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """Score every pair with Ditto; return id1/id2/score.

    Calls ``DittoMatcher.match`` with ``threshold=0.0`` so every
    candidate's softmax probability is returned, regardless of
    decision boundary. The matcher's per-batch CSV cache makes this
    resumable.
    """
    if candidates.empty:
        return pd.DataFrame(columns=["id1", "id2", "score"])
    result = matcher.match(
        df_left=df_left,
        df_right=df_right,
        candidates=candidates,
        id_column="id",
        threshold=0.0,
    )
    keep = result[["id1", "id2", "score"]].copy()
    keep["id1"] = keep["id1"].astype(str)
    keep["id2"] = keep["id2"].astype(str)
    return keep


# ---------------------------------------------------------------------------
# Delta estimation
# ---------------------------------------------------------------------------


def estimate_delta_from_predictions(
    predictions_csv: Path,
    *,
    threshold: float = DITTO_THRESHOLD,
    percentile: float = DELTA_PERCENTILE,
    floor: float = DELTA_FLOOR,
    cap: float = DELTA_CAP,
) -> tuple[float, dict[str, Any]]:
    """Estimate delta from R2 evaluation predictions.

    R2's ``evaluate.py`` writes ``predictions.csv`` with columns
    ``idx``, ``pair_id``, ``gold``, ``pred``, ``prob``. We compute
    ``|prob - threshold|`` over rows where ``pred != gold`` and take
    the configured percentile, clipped to ``[floor, cap]``. The
    rationale: this band captures the bulk of the checkpoint's
    decision-boundary errors, which is exactly where the LLM
    adjudicator is most likely to add value.

    Returns
    -------
    tuple
        ``(delta, telemetry_dict)``.
    """
    df = pd.read_csv(predictions_csv)
    needed = {"gold", "pred", "prob"}
    if not needed.issubset(df.columns):
        raise ValueError(
            f"{predictions_csv} missing columns: needed {needed}, got {set(df.columns)}"
        )
    miscls = df[df["pred"] != df["gold"]]
    n_miscls = int(len(miscls))
    n_total = int(len(df))
    if n_miscls == 0:
        delta = floor
        pct = float("nan")
    else:
        margins = (miscls["prob"] - threshold).abs().to_numpy()
        pct = float(np.percentile(margins, percentile))
        delta = max(floor, min(cap, pct))
    return delta, {
        "n_predictions": n_total,
        "n_misclassified": n_miscls,
        "raw_percentile": pct,
        "delta_used": delta,
        "percentile": percentile,
        "floor": floor,
        "cap": cap,
    }


# ---------------------------------------------------------------------------
# Pool LLM adjudicator
# ---------------------------------------------------------------------------


def serialise_record_for_pool(
    rid: str,
    sources: dict[str, pd.DataFrame],
    fields: Sequence[str],
    source_lookup: Callable[[str], str],
) -> str:
    """Serialise a record as ``col=value | col=value`` for the LLM prompt.

    Each source DataFrame must have an ``id`` column and the field
    columns referenced in ``fields`` (already mapped to canonical
    names).
    """
    src = source_lookup(rid)
    df = sources.get(src)
    if df is None:
        raise KeyError(f"unknown source for id {rid!r}")
    sub = df[df["id"] == rid]
    if sub.empty:
        raise KeyError(f"id {rid!r} not found in source {src!r}")
    row = sub.iloc[0]
    parts: list[str] = []
    for col in fields:
        if col not in row.index:
            continue
        val = row[col]
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue
        parts.append(f"{col}={val}")
    return " | ".join(parts) if parts else f"id={rid}"


def build_pool_llm_adjudicator(
    *,
    domain: str,
    sources: dict[str, pd.DataFrame],
    fields: Sequence[str],
    source_lookup: Callable[[str], str],
    llm_cache: LLMCache,
    api_client: Callable[[str], str] | None,
    strict_cache: bool = False,
    prompt_template: str = POOL_ADJUDICATOR_PROMPT,
) -> Callable[[Pair], bool | None]:
    """Return a callable adjudicator for bucket-C singletons.

    The cache namespace is ``r3_pool_adjudicator::<domain>`` so entries
    do not collide with the K2 hard-negative gate's cache. ``model_id``
    is baked into the cache key by ``LLMCache``, so swapping models
    invalidates the prior namespace's entries.

    Returns ``True`` / ``False`` for a successful verdict, or ``None``
    when the call is unavailable (cache miss with ``api_client=None``
    and ``strict_cache=False``) — callers should treat ``None`` as
    "drop conservatively" to avoid leaking unjudged singletons into
    the pool.
    """

    def adjudicate(pair: Pair) -> bool | None:
        rid_a, rid_b = pair
        try:
            text_a = serialise_record_for_pool(rid_a, sources, fields, source_lookup)
            text_b = serialise_record_for_pool(rid_b, sources, fields, source_lookup)
        except KeyError as exc:
            logger.warning("pool adjudicator: %s; dropping pair", exc)
            return None

        prompt = prompt_template.format(record_a=text_a, record_b=text_b)
        cache_value = f"{rid_a}|{rid_b}|{text_a}||{text_b}"

        def _call() -> dict[str, Any]:
            if api_client is None:
                raise RuntimeError("pool adjudicator cache miss but api_client=None")
            reply = api_client(prompt).strip().lower()
            return {
                "prompt": prompt,
                "reply": reply,
                "says_match": reply.startswith("y"),
            }

        try:
            payload = llm_cache.call_or_cache(
                source=f"r3_pool_adjudicator::{domain}",
                attribute="pair",
                value=cache_value,
                api_fn=_call,
                strict=strict_cache,
            )
        except LLMCacheMiss:
            logger.warning(
                "pool adjudicator strict-cache miss for %s|%s — dropping",
                rid_a,
                rid_b,
            )
            return None
        except RuntimeError as exc:
            if "api_client=None" in str(exc):
                logger.warning(
                    "pool adjudicator cache miss + no api_client for %s|%s — dropping",
                    rid_a,
                    rid_b,
                )
                return None
            raise

        result = payload.get("result") if "result" in payload else payload
        if isinstance(result, dict) and "says_match" in result:
            return bool(result["says_match"])
        if isinstance(payload, dict) and "says_match" in payload:
            return bool(payload["says_match"])
        return None

    return adjudicate


# ---------------------------------------------------------------------------
# Bucket assembly
# ---------------------------------------------------------------------------


def build_buckets(
    *,
    gold_positives: set[Pair],
    human_pairs: set[Pair],
    ditto_scores: dict[Pair, float],
    delta: float,
    adjudicator: Callable[[Pair], bool | None],
    threshold: float = DITTO_THRESHOLD,
) -> BucketResult:
    """Assemble bucket A/B/C and run the LLM check on every bucket-C pair.

    Pair classification:

    - Bucket A: gold positive — kept unconditionally (score = 1.0).
    - Bucket B: in human-baseline AND Ditto score >= threshold —
      kept (Ditto + human agree).
    - Bucket C: any disagreement (exactly one of {human-baseline,
      Ditto>=threshold} says positive) — **every** pair goes through
      ``adjudicator`` (LLM). No more score-based shortcuts.

    Pairs absent from ``ditto_scores`` are assigned score ``0.0`` (the
    Ditto candidate set should be a superset of (human ∪ gold) so this
    only fires for pathological inputs).

    Policy change (2026-05-26, plan_revision_step4g_findings.md
    follow-up): the legacy ``score >= theta + delta`` auto-include and
    ``score < theta - delta`` auto-drop paths were removed. Ditto's
    per-domain precision on raw data is too low to overrule the
    human-baseline matcher on score alone (games dbpedia_sales: 0.43
    at threshold 0.5). The LLM adjudicator now arbitrates **every**
    disagreement. The ``delta`` parameter is no longer consulted by
    the routing logic but is preserved on the result for telemetry.
    """
    rows: list[dict[str, Any]] = []
    bucket_a = bucket_b = 0
    bucket_c_total = 0
    c_kept_llm = c_dropped_llm = 0

    pairs_to_consider: set[Pair] = (
        set(gold_positives)
        | set(human_pairs)
        | {p for p, s in ditto_scores.items() if s >= threshold}
    )

    for pair in sorted(pairs_to_consider):
        score = float(ditto_scores.get(pair, 0.0))
        in_gold = pair in gold_positives
        in_human = pair in human_pairs
        in_ditto = score >= threshold

        if in_gold:
            rows.append(
                {
                    "id1": pair[0],
                    "id2": pair[1],
                    "score": 1.0,
                    "in_gold": True,
                    "in_human": in_human,
                    "in_ditto": in_ditto,
                    "decision_path": "gold",
                }
            )
            bucket_a += 1
            continue

        if in_human and in_ditto:
            rows.append(
                {
                    "id1": pair[0],
                    "id2": pair[1],
                    "score": score,
                    "in_gold": False,
                    "in_human": True,
                    "in_ditto": True,
                    "decision_path": "agreement",
                }
            )
            bucket_b += 1
            continue

        # Disagreement: human says yes XOR Ditto says yes. LLM decides.
        bucket_c_total += 1
        verdict = adjudicator(pair)
        if verdict is True:
            rows.append(
                {
                    "id1": pair[0],
                    "id2": pair[1],
                    "score": score,
                    "in_gold": False,
                    "in_human": in_human,
                    "in_ditto": in_ditto,
                    "decision_path": "plm_check_llm_yes",
                }
            )
            c_kept_llm += 1
        else:
            c_dropped_llm += 1

    pool_df = pd.DataFrame(
        rows,
        columns=[
            "id1",
            "id2",
            "score",
            "in_gold",
            "in_human",
            "in_ditto",
            "decision_path",
        ],
    )
    return BucketResult(
        pool_df=pool_df,
        bucket_a=bucket_a,
        bucket_b=bucket_b,
        bucket_c_total=bucket_c_total,
        bucket_c_kept_confident=0,
        bucket_c_kept_llm=c_kept_llm,
        bucket_c_dropped_confident=0,
        bucket_c_dropped_llm=c_dropped_llm,
        delta_used=delta,
    )


# ---------------------------------------------------------------------------
# Cross-pair transitive closure (per evidence stream)
# ---------------------------------------------------------------------------


def transitive_closure_across_pairs(
    pairs_per_source_pair: dict[frozenset[str], set[Pair]],
    *,
    source_lookup: Callable[[str], str],
) -> dict[frozenset[str], set[Pair]]:
    """Expand each source-pair's edge set to the connected-components closure.

    Builds one undirected graph from the union of all per-source-pair
    edges, computes connected components, then for every component
    emits all pairwise edges between nodes whose source labels differ
    — grouping the resulting edges by their unordered source pair.
    Edges already present per-pair are kept; new edges are added.

    The intent (per ``plans/plan_s1_scale.md`` R3) is to take an
    evidence stream that only directly covers some source-pairs (e.g.
    notebook ran the matcher only for two of three pairs) and infer
    the missing-pair correspondences by transitivity through the
    pivot source. Run separately on the human-baseline stream and on
    the Ditto-positive stream so the two streams stay independent
    inputs to the bucket-B agreement test.

    Parameters
    ----------
    pairs_per_source_pair : dict
        Mapping ``frozenset({src_a, src_b}) -> set[Pair]``. Each Pair
        is a canonical (id1, id2) tuple. Empty values are tolerated.
    source_lookup : callable
        Maps an entity id to its source label.

    Returns
    -------
    dict
        Same shape as the input, with the closure expansion applied.
        Source-pair keys absent from the input but present after
        closure (e.g. ``frozenset({"metacritic", "sales"})`` when only
        ``{dbpedia, metacritic}`` and ``{dbpedia, sales}`` had direct
        edges) are added.
    """
    try:
        import networkx as nx  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("networkx not installed; transitive closure is a no-op")
        return {k: set(v) for k, v in pairs_per_source_pair.items()}

    g = nx.Graph()
    for edges in pairs_per_source_pair.values():
        for id1, id2 in edges:
            g.add_edge(id1, id2)

    expanded: dict[frozenset[str], set[Pair]] = {
        k: set(v) for k, v in pairs_per_source_pair.items()
    }

    for component in nx.connected_components(g):
        if len(component) < 2:
            continue
        nodes = sorted(component)
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                a, b = nodes[i], nodes[j]
                src_a = source_lookup(a)
                src_b = source_lookup(b)
                if src_a == src_b or src_a == "unknown" or src_b == "unknown":
                    continue
                key = frozenset({src_a, src_b})
                expanded.setdefault(key, set()).add(canonical_pair(a, b))
    return expanded


def closure_added_counts(
    pre: dict[frozenset[str], set[Pair]],
    post: dict[frozenset[str], set[Pair]],
) -> dict[str, int]:
    """Per-source-pair count of edges added by transitive closure."""
    out: dict[str, int] = {}
    for key in post:
        added = len(post[key]) - len(pre.get(key, set()))
        out["__".join(sorted(key))] = max(0, added)
    return out


# ---------------------------------------------------------------------------
# Cluster size telemetry (informational only; no filtering)
# ---------------------------------------------------------------------------


def cluster_size_distribution(pool_df: pd.DataFrame) -> dict[str, Any]:
    """Compute connected-component size distribution for telemetry.

    Pool is pair-level; connected components are an audit signal only
    (no egregious-cluster filter is applied per the R3 design — see
    plans/plan_s1_scale.md R3 review).
    """
    try:
        import networkx as nx  # type: ignore[import-untyped]
    except ImportError:
        return {"available": False}
    if pool_df.empty:
        return {"available": True, "n_components": 0, "top_sizes": []}
    g = nx.Graph()
    for row in pool_df[["id1", "id2"]].itertuples(index=False):
        g.add_edge(row.id1, row.id2)
    sizes = sorted((len(c) for c in nx.connected_components(g)), reverse=True)
    return {
        "available": True,
        "n_components": len(sizes),
        "max_size": sizes[0] if sizes else 0,
        "top_sizes": sizes[:10],
        "p95_size": int(np.percentile(sizes, 95)) if sizes else 0,
        "p99_size": int(np.percentile(sizes, 99)) if sizes else 0,
    }


__all__ = [
    "DITTO_THRESHOLD",
    "RECALL_FLOOR",
    "DELTA_FLOOR",
    "DELTA_CAP",
    "DELTA_PERCENTILE",
    "POOL_PROMPT_VERSION",
    "POOL_ADJUDICATOR_PROMPT",
    "Pair",
    "BlockerCandidate",
    "BlockerResult",
    "BlockerSweepResult",
    "BucketResult",
    "GoldSplit",
    "canonical_pair",
    "apply_column_mapping",
    "load_em_gold_for_pair",
    "load_human_baseline_pairs_from_files",
    "make_blocker_candidates",
    "evaluate_blocker",
    "sweep_blockers",
    "assemble_candidate_set",
    "score_with_ditto",
    "estimate_delta_from_predictions",
    "build_pool_llm_adjudicator",
    "serialise_record_for_pool",
    "build_buckets",
    "cluster_size_distribution",
    "transitive_closure_across_pairs",
    "closure_added_counts",
]
