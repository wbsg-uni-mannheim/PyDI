"""Entity-matching committee runner (split-roster, two-phase).

Under the C2.4b architecture, the EM committee is split into two
rosters: a blocking committee (``em_blocking_committee.yaml``) whose
members emit candidate-pair sets, and a matching committee
(``em_matching_committee.yaml``) whose members classify those pairs.
The runner executes them sequentially per source pair:

1. **Phase 1 (blocker selection).** Every enabled blocker emits its
   candidate set; the runner scores each on pair recall and reduction
   ratio. The winner is the blocker with the highest reduction ratio
   among those clearing the composition ``recall_floor`` (0.97 by
   default). Ties are broken alphabetically on blocker name for
   determinism. If no blocker clears the floor, the runner falls back
   to the highest-recall blocker and logs a warning — the pipeline
   stays functional and the shortfall is surfaced in ``per_blocker``.

2. **Phase 2 (matching on winner's candidates).** Every enabled
   matching-committee member runs against the winner's candidate set,
   producing per-pair F1 / precision / recall against the regenerated
   (primary) / pool / test gold surfaces. Clustering (greedy / mbm /
   none) is applied post-matcher as before.

The result carries matching-committee members under ``per_member``
(so ``aggregated`` macro F1 is untouched by blocker pair-recall
numbers) and the blocking-committee members under the new
``per_blocker`` field. Each blocker's ``notes.per_pair`` records the
winner flag so downstream analysis can see which blocker was selected
for each source pair.
"""

from __future__ import annotations

import importlib
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml

from PyDI.entitymatching.base import BaseComparator, BaseMatcher
from PyDI.entitymatching.blocking.base import BaseBlocker
from PyDI.entitymatching.post_clustering.greedy_one_to_one import (
    GreedyOneToOneMatchingAlgorithm,
)
from PyDI.entitymatching.post_clustering.maximum_bipartite_matching import (
    MaximumBipartiteMatching,
)

from .column_mapping import apply_column_mapping
from .loaders import (
    _source_filename_tokens,
    em_gold_candidates,
    read_em_gold_pair,
)
from .committee import CommitteeResult, CommitteeRunner, MemberResult
from .committee_em_scoring import (
    blocking_pair_recall,
    pool_agreement,
    reduction_ratio,
    score_em_correspondences_closed_set,
    score_em_vs_pool,
)
from .variant_loader import VariantBundle

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parsed roster member types
# ---------------------------------------------------------------------------


@dataclass
class _EMBlockingRosterMember:
    """Parsed representation of a single blocking-roster entry."""

    name: str
    description: str
    blocker_spec: dict[str, Any]
    blocking_type: str
    enabled_by_default: bool


@dataclass
class _EMMatchingRosterMember:
    """Parsed representation of a single matching-roster entry."""

    name: str
    description: str
    matcher_spec: dict[str, Any]
    comparator_specs: list[dict[str, Any]]
    weights: list[float]
    threshold: float
    matching_type: str
    missing_value_tolerant: bool
    enabled_by_default: bool


@dataclass
class _CompositionConfig:
    """Blocking-committee composition rules."""

    strategy: str
    recall_floor: float
    tie_breaker: str
    # Upper bound on the winning blocker's candidate count. Guards against the
    # recall-floor fallback selecting an UNCAPPED blocker (e.g. token_blocker)
    # whose candidate set is 10-100x the top_k-capped blockers', which makes
    # downstream matching grind for hours. 0 disables the cap.
    max_candidates: int = 5_000_000


def _parse_blocking_roster(
    raw_members: list[dict[str, Any]],
) -> list[_EMBlockingRosterMember]:
    """Parse and filter the blocking roster YAML.

    Only ``enabled_by_default`` members are kept — disabled entries
    (placeholders for pending adapters like ``sc_block``) are dropped
    so they do not participate in blocker selection.

    Parameters
    ----------
    raw_members : list of dict
        ``members`` list from the blocking-roster YAML.

    Returns
    -------
    list of _EMBlockingRosterMember
        Enabled blocking members in YAML declaration order.
    """
    out: list[_EMBlockingRosterMember] = []
    for entry in raw_members:
        enabled = entry.get("enabled_by_default", True)
        if not enabled:
            continue
        out.append(
            _EMBlockingRosterMember(
                name=entry["name"],
                description=entry.get("description", ""),
                blocker_spec=dict(entry.get("blocker", {})),
                blocking_type=entry.get("blocking_type", "unknown"),
                enabled_by_default=enabled,
            )
        )
    return out


def _parse_matching_roster(
    raw_members: list[dict[str, Any]],
    *,
    with_llm: bool = False,
) -> list[_EMMatchingRosterMember]:
    """Parse and filter the matching roster YAML.

    Parameters
    ----------
    raw_members : list of dict
        ``members`` list from the matching-roster YAML.
    with_llm : bool
        When ``False`` (default), members with ``matching_type == "llm"``
        are excluded. Set to ``True`` to opt LLM matchers in.

    Returns
    -------
    list of _EMMatchingRosterMember
        Enabled matching members in YAML declaration order.
    """
    out: list[_EMMatchingRosterMember] = []
    for entry in raw_members:
        enabled = entry.get("enabled_by_default", True)
        is_llm = entry.get("matching_type") == "llm"

        if is_llm and not with_llm:
            continue
        if not enabled and not (is_llm and with_llm):
            continue

        out.append(
            _EMMatchingRosterMember(
                name=entry["name"],
                description=entry.get("description", ""),
                matcher_spec=dict(entry.get("matcher", {})),
                comparator_specs=list(entry.get("comparators", [])),
                weights=list(entry.get("weights", [])),
                threshold=float(entry.get("threshold", 0.5)),
                matching_type=entry.get("matching_type", "unknown"),
                missing_value_tolerant=bool(entry.get("missing_value_tolerant", False)),
                enabled_by_default=enabled,
            )
        )
    return out


def _parse_composition(raw: dict[str, Any] | None) -> _CompositionConfig:
    """Parse the ``composition`` block from the blocking roster.

    Falls back to the user-frozen defaults (``select_best`` /
    ``recall_floor=0.97`` / ``tie_breaker=reduction_ratio``) when the
    block is omitted — this keeps ad-hoc fixture rosters usable in tests
    without duplicating the composition config in every fixture.

    Parameters
    ----------
    raw : dict or None
        The ``composition`` block from the blocking roster YAML, or
        ``None`` if absent.

    Returns
    -------
    _CompositionConfig
        Parsed composition rules.
    """
    raw = raw or {}
    return _CompositionConfig(
        strategy=str(raw.get("strategy", "select_best")),
        recall_floor=float(raw.get("recall_floor", 0.97)),
        tie_breaker=str(raw.get("tie_breaker", "reduction_ratio")),
        max_candidates=int(raw.get("max_candidates", 5_000_000)),
    )


# ---------------------------------------------------------------------------
# Dynamic instantiation helpers
# ---------------------------------------------------------------------------


def _import_class(module_path: str, cls_name: str) -> type:
    """Import a class from a dotted module path.

    Parameters
    ----------
    module_path : str
        Dotted module path (e.g. ``"PyDI.entitymatching.blocking.token_blocking"``).
    cls_name : str
        Class name within the module.

    Returns
    -------
    type
        The class object.
    """
    mod = importlib.import_module(module_path)
    return getattr(mod, cls_name)


def _build_comparators(
    specs: list[dict[str, Any]],
    preprocess_fn: Any | None,
) -> list[BaseComparator]:
    """Instantiate comparators from YAML specs.

    Parameters
    ----------
    specs : list of dict
        Comparator specifications.
    preprocess_fn : callable or None
        Shared preprocess function (e.g. ``normalize_text``).

    Returns
    -------
    list of BaseComparator
        Instantiated comparators.
    """
    comparators: list[BaseComparator] = []
    for spec in specs:
        cls = _import_class(spec["module"], spec["class"])
        params = dict(spec.get("params", {}))
        if preprocess_fn is not None and "preprocess" not in params:
            params["preprocess"] = preprocess_fn
        comparators.append(cls(**params))
    return comparators


def _build_blocker(
    spec: dict[str, Any],
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    id_column: str,
    *,
    checkpoint_override: Path | None = None,
) -> BaseBlocker:
    """Instantiate a blocker from a YAML spec.

    Parameters
    ----------
    spec : dict
        Blocker specification with ``class``, ``module``, ``params``.
    df_left, df_right : DataFrame
        Source DataFrames for this pair.
    id_column : str
        ID column name.
    checkpoint_override : Path or None
        Substitutes ``params.checkpoint_path`` when present (R7b dual-
        model: builds the variant-trained sc_block instance alongside
        the baseline-trained one). No-op for blockers without a
        ``checkpoint_path`` param.

    Returns
    -------
    BaseBlocker
        Instantiated blocker ready to materialise.
    """
    cls = _import_class(spec["module"], spec["class"])
    params = dict(spec.get("params", {}))
    if checkpoint_override is not None and "checkpoint_path" in params:
        params["checkpoint_path"] = str(checkpoint_override)
    return cls(df_left, df_right, id_column=id_column, **params)


# Matcher classes that require per-source-pair training/demonstration data.
# The runner injects the path at instantiation time inside the per-pair loop
# (`_run_pair`); the YAML must NOT declare these keys in `params` because
# the static path would silently train every pair against the same CSV.
# Closure-only source pairs (no `<src1>_2_<src2>_train.csv` on disk) skip
# the affected matcher and surface "skipped: no_train_data" in per-pair
# metrics.
_PER_PAIR_TRAIN_INJECTION: dict[str, str] = {
    "MagellanMatcher": "training_gold_path",
}


def _build_matcher(
    spec: dict[str, Any],
    *,
    pair_train_path: Path | None = None,
    checkpoint_override: Path | None = None,
) -> BaseMatcher:
    """Instantiate a matcher from a YAML spec.

    For matcher classes listed in :data:`_PER_PAIR_TRAIN_INJECTION`, the
    per-pair training path is injected into ``params`` at instantiation
    time. Other matcher classes ignore ``pair_train_path``.

    For matcher classes with a ``checkpoint_path`` parameter (Ditto,
    sc_block, etc.), an optional ``checkpoint_override`` lets the
    runner substitute the variant-specific checkpoint at instantiation
    time without re-authoring the YAML. R7b dual-model evaluation uses
    this to build a "variant model" instance alongside the baseline.

    Parameters
    ----------
    spec : dict
        Matcher YAML spec with keys ``module``, ``class``, ``params``.
    pair_train_path : Path or None
        Per-source-pair training CSV path resolved by the runner.
    checkpoint_override : Path or None
        Substitutes ``params.checkpoint_path`` when present (and the
        spec actually carries such a param). No-op for matchers without
        a checkpoint_path field (e.g. zero-shot LLM matchers).

    Returns
    -------
    BaseMatcher
        Ready-to-call matcher instance.
    """
    cls = _import_class(spec["module"], spec["class"])
    params = dict(spec.get("params", {}))
    inject_key = _PER_PAIR_TRAIN_INJECTION.get(str(spec.get("class", "")))
    if inject_key is not None and pair_train_path is not None:
        params[inject_key] = str(pair_train_path)
    if checkpoint_override is not None and "checkpoint_path" in params:
        params["checkpoint_path"] = str(checkpoint_override)
    return cls(**params)


def _resolve_pair_train_path(
    bundle: VariantBundle,
    pair: tuple[str, str],
) -> Path | None:
    """Resolve the per-source-pair ``_train.csv`` path for *pair*.

    The runner uses this path to inject per-pair training data into
    matchers that require it (currently MagellanMatcher). Returns
    ``None`` when no train file exists for either pair orientation
    (e.g. closure-only pairs without authored EM gold).

    Parameters
    ----------
    bundle : VariantBundle
        The variant bundle (carries ``variant_root``).
    pair : tuple of str
        ``(src1, src2)`` source pair.

    Returns
    -------
    Path or None
        Path to the existing ``_train.csv`` or ``None`` when neither
        orientation has a file on disk.
    """
    em_dir = bundle.variant_root / "input" / "entitymatching"
    src1, src2 = pair
    # Non-versioned canonical names FIRST so every domain that ships
    # ``<src1>_2_<src2>_train.csv`` resolves exactly as before. The 2026
    # papers domain ships NO non-versioned train file under that name; it
    # uses condensed, ``_2_``-less naming (``dblp_crossref_train.csv``) and
    # for variants only the K2 split versions exist, so fall back to
    # ``_train_corner_filled`` (matched-distribution training set) then
    # ``_train_baseline_pruned``. Each name is also tried under the
    # condensed papers source tokens (``open_alex`` -> ``openalex``) via
    # ``_source_filename_tokens``, which is a no-op for every other domain.
    suffixes = ("train", "train_corner_filled", "train_baseline_pruned")
    seen: set[Path] = set()
    for suffix in suffixes:
        for t1 in _source_filename_tokens(src1):
            for t2 in _source_filename_tokens(src2):
                for cand in (
                    em_dir / f"{t1}_2_{t2}_{suffix}.csv",
                    em_dir / f"{t2}_2_{t1}_{suffix}.csv",
                    em_dir / f"{t1}_{t2}_{suffix}.csv",
                    em_dir / f"{t2}_{t1}_{suffix}.csv",
                ):
                    if cand in seen:
                        continue
                    seen.add(cand)
                    if cand.exists():
                        return cand
    return None


def _resolve_variant_checkpoint_path(
    baseline_checkpoint: object, level: str
) -> tuple[Path | None, bool]:
    """Return ``(resolved_checkpoint, is_variant_distinct)`` for a matcher checkpoint.

    R7b dual-model evaluation: at variant levels (easy / medium / hard),
    look for a variant-specific checkpoint at
    ``<baseline_parent>/variant_<level>/best``. If it exists, return that
    path + ``is_variant_distinct=True``. Otherwise fall back to the
    baseline checkpoint + ``is_variant_distinct=False`` — the runner
    treats this as "variant model is identical to baseline model" and
    skips the duplicate inference pass.

    Parameters
    ----------
    baseline_checkpoint : str or Path or None
        The matcher YAML's ``params.checkpoint_path`` (or ``None`` for
        matchers without a checkpoint, e.g. zero-shot LLM matchers).
    level : str
        Bundle level (``"baseline"`` / ``"easy"`` / ``"medium"`` /
        ``"hard"``).

    Returns
    -------
    tuple
        ``(Path | None, bool)`` — resolved checkpoint path and whether
        the variant checkpoint is distinct from the baseline one.
    """
    if baseline_checkpoint is None:
        return None, False
    baseline_path = Path(str(baseline_checkpoint))
    if level == "baseline":
        return baseline_path, False
    variant_path = baseline_path.parent / f"variant_{level}" / "best"
    if variant_path.exists():
        return variant_path, True
    return baseline_path, False


def _stratified_holdout_val(
    gold: pd.DataFrame, *, val_fraction: float = 0.2, seed: int = 42
) -> pd.DataFrame:
    """Hold out a stratified val slice from a train gold frame.

    Used for EM evaluation when a pair ships no ``*_val`` split (e.g. games,
    train-only by design). The slice matches the trainer's own held-out val
    (``ditto/_prep_games._split_train_val_stratified``: seed=42,
    val_fraction=0.2, stratified on ``label``), so the reported EM val F1 is a
    true held-out number rather than a skipped pair.
    """
    from sklearn.model_selection import train_test_split

    if gold is None or gold.empty or "label" not in gold.columns:
        return gold
    strat = gold["label"] if gold["label"].nunique() > 1 else None
    try:
        _, val_df = train_test_split(
            gold, test_size=val_fraction, random_state=seed, stratify=strat
        )
    except ValueError:
        _, val_df = train_test_split(gold, test_size=val_fraction, random_state=seed)
    return val_df.reset_index(drop=True)


def _load_labelled_split_from_bundle(
    bundle: VariantBundle,
    pair: tuple[str, str],
    split: str,
    version: str = "corner_filled",
) -> pd.DataFrame | None:
    """Return the labelled ``(id1, id2, label)`` gold for one split.

    Resolution: K2 regenerated split for the requested ``version`` if
    present in ``bundle.em_gold_regenerated[pair][split][version]``
    (variants under plan_revision.md C11), else the raw
    ``<src1>_2_<src2>_<split>.csv`` from the variant's entitymatching
    directory (baseline). Either orientation of the filename is
    accepted — when loading from the reverse direction, ``id1`` and
    ``id2`` are swapped so the returned frame matches the declared
    ``(src1, src2)`` pair direction downstream consumers expect.

    Shared between EM matching + EM blocking runners (R7b).

    Parameters
    ----------
    bundle : VariantBundle
        The variant bundle (carries ``variant_root`` + regen dict).
    pair : tuple of str
        ``(src1, src2)`` source pair.
    split : str
        ``"train"`` / ``"val"`` / ``"test"`` / ``"all"``.
    version : str, default ``"corner_filled"``
        One of ``"baseline_pruned"`` or ``"corner_filled"`` (C11).

    Returns
    -------
    DataFrame or None
        Frame with ``id1``, ``id2``, ``label`` columns; ``None`` if
        neither the regen entry nor any direction's file exists.
    """
    regen_pair = bundle.em_gold_regenerated.get(pair, {})
    regen_split = regen_pair.get(split, {})
    if version in regen_split:
        return regen_split[version]
    em_dir = bundle.variant_root / "input" / "entitymatching"
    # Candidate resolution (canonical ``<src1>_2_<src2>`` first, then the
    # condensed papers naming) shared with ``variant_loader._load_em_gold``.
    # ``read_em_gold_pair`` swaps id1<->id2 for reverse-direction files so
    # id1 always belongs to the declared pair's src1 (the matcher / blocker
    # input expects this; 2026-05-27 regression on games metacritic_dbpedia
    # where direction-tolerance without the swap silently produced F1=0 and
    # crashed magellan).
    match = next(
        (
            (path, swap)
            for path, swap in em_gold_candidates(em_dir, pair, split)
            if path.exists()
        ),
        None,
    )
    if match is not None:
        return read_em_gold_pair(*match)
    # Fallback: no ``*_val`` gold for this pair (e.g. games, train-only by
    # design). Derive the val surface from a stratified hold-out of the SAME
    # train version — the exact slice the trainer early-stops on — so EM val
    # scoring has a true held-out gold instead of skipping the pair.
    if split == "val":
        train_gold = _load_labelled_split_from_bundle(
            bundle, pair, "train", version=version
        )
        if train_gold is not None and not train_gold.empty:
            return _stratified_holdout_val(train_gold)
    return None


def _resolve_variant_train_path(
    bundle: VariantBundle,
    pair: tuple[str, str],
) -> tuple[Path | None, bool]:
    """Return ``(variant_train_path, is_variant_distinct)`` for Magellan-style retraining.

    R7b dual-model: at variant levels, prefer the K2-regenerated
    ``<pair>_train_corner_filled.csv`` (the matched-distribution training
    set under C11). Falls back to the un-versioned baseline
    ``<pair>_train.csv`` if the regen file is absent (legacy pre-C11
    variants, or closure-only pairs without K2 regen output).

    Parameters
    ----------
    bundle : VariantBundle
        The variant bundle (carries ``variant_root`` and ``level``).
    pair : tuple of str
        ``(src1, src2)`` source pair.

    Returns
    -------
    tuple
        ``(Path | None, bool)`` — resolved train CSV path (or ``None``
        if neither variant nor baseline path exists) and whether the
        variant train is distinct from baseline.
    """
    if bundle.level == "baseline":
        return _resolve_pair_train_path(bundle, pair), False
    em_dir = bundle.variant_root / "input" / "entitymatching"
    src1, src2 = pair
    for candidate in (
        em_dir / f"{src1}_2_{src2}_train_corner_filled.csv",
        em_dir / f"{src2}_2_{src1}_train_corner_filled.csv",
    ):
        if candidate.exists():
            return candidate, True
    return _resolve_pair_train_path(bundle, pair), False


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------


def _normalize_text(s: str) -> str:
    """Strip punctuation and lowercase.

    Mirrors the ``normalize_text`` function used in the reference
    companies workflow (``test_workflow_companies.py``).

    Parameters
    ----------
    s : str
        Input string.

    Returns
    -------
    str
        Normalised string.
    """
    if s is None:
        return ""
    return re.sub(r"[^\w\s]|_", "", s).lower()


_PREPROCESS_REGISTRY: dict[str, Any] = {
    "normalize_text": _normalize_text,
}


_KEY_FIRST_N_RE = re.compile(r"^(.+)_first_(\d+)$")
_KEY_FIRST_TOKEN_RE = re.compile(r"^(.+)_first_token$")
_KEY_NORM_RE = re.compile(r"^(.+)_norm$")


def _alnum_lower(value: Any) -> str:
    """Lowercase + strip non-alphanumeric. Used by the derived-key generators."""
    if value is None:
        return ""
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _first_token(name: Any) -> str:
    """First alphabetic token (length >= 2), lowercase."""
    if not isinstance(name, str):
        name = str(name) if name is not None else ""
    tokens = re.split(r"[^a-z]", name.lower())
    first = [t for t in tokens if len(t) > 1]
    return first[0] if first else name.lower()


def _derive_blocking_key(
    df: pd.DataFrame,
    key_name: str,
    *,
    blocking_name_column: str = "name",
) -> bool:
    """Derive a single blocking-key column by name pattern.

    Recognised patterns (the source column is the prefix before the
    pattern suffix; e.g. ``name_first_3`` derives from ``name``):

    - ``<col>_first_token`` — first alphabetic token of *col*, length
      ≥ 2 (legacy companies-workflow convention).
    - ``<col>_first_<N>`` — first N alphanumeric chars of *col*
      (case-folded, non-alphanumeric stripped).
    - ``<col>_norm`` — lowercased + whitespace-stripped *col*.

    ``blocking_name_column`` is used as the base column when the
    pattern's prefix is the literal ``"name"`` (this preserves
    backward-compat with the per-domain ``blocking_name_column``
    config for domains where the canonical primary column isn't
    literally ``name`` — e.g. movies/products use ``title``).

    Returns True if the column was derived, False if no pattern
    matched.
    """
    if key_name in df.columns:
        return True

    # name_first_token — also handle the legacy "name" alias via blocking_name_column.
    m = _KEY_FIRST_TOKEN_RE.match(key_name)
    if m:
        base_col = m.group(1)
        if base_col == "name" and blocking_name_column != "name":
            base_col = blocking_name_column
        if base_col in df.columns:
            df[key_name] = df[base_col].apply(_first_token)
            return True
        return False

    m = _KEY_FIRST_N_RE.match(key_name)
    if m:
        base_col = m.group(1)
        n = int(m.group(2))
        if base_col == "name" and blocking_name_column != "name":
            base_col = blocking_name_column
        if base_col in df.columns:
            df[key_name] = df[base_col].apply(lambda v: _alnum_lower(v)[:n])
            return True
        return False

    m = _KEY_NORM_RE.match(key_name)
    if m:
        base_col = m.group(1)
        if base_col == "name" and blocking_name_column != "name":
            base_col = blocking_name_column
        if base_col in df.columns:
            df[key_name] = df[base_col].apply(
                lambda v: "" if v is None else str(v).strip().lower()
            )
            return True
        return False

    return False


def _generate_blocking_keys(
    df: pd.DataFrame,
    column: str = "name",
    *,
    required_keys: list[str] | None = None,
) -> pd.DataFrame:
    """Derive blocking-key columns on *df* in place.

    Backward-compat: when *required_keys* is ``None`` (legacy callers
    only — the new EM blocking committee passes the keys it needs),
    derives ``name_first_token`` for the configured *column* so the
    pre-2026-05-10 standard_blocker spec keeps working.

    When *required_keys* is supplied (per the 2026-05-10 R5 EM
    blocking sign-off), each key is generated via
    :func:`_derive_blocking_key`. Patterns supported: ``_first_token``,
    ``_first_<N>``, ``_norm``. Compound keys are not yet supported in
    the runner — the standard_blocker `on` list takes the AND of
    multiple keys directly.
    """
    if required_keys is None:
        # Legacy path: just derive name_first_token from the configured column.
        if column in df.columns:
            df["name_first_token"] = df[column].apply(_first_token)
        return df

    for key in required_keys:
        ok = _derive_blocking_key(df, key, blocking_name_column=column)
        if not ok:
            logger.warning(
                "Could not derive blocking key %r (base column missing or "
                "pattern not recognised). Available columns: %s",
                key,
                list(df.columns),
            )
    return df


# ---------------------------------------------------------------------------
# Determinism helpers
# ---------------------------------------------------------------------------


def _set_deterministic(seed: int) -> None:
    """Set seeds for numpy and torch determinism.

    Parameters
    ----------
    seed : int
        Random seed.
    """
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Blocker selection
# ---------------------------------------------------------------------------


def _select_best_blocker(
    blocker_metrics: dict[str, dict[str, float]],
    composition: _CompositionConfig,
) -> tuple[str, bool]:
    """Pick the winning blocker for a single source pair.

    Blockers whose ``candidate_count`` exceeds ``composition.max_candidates``
    are excluded FIRST (an uncapped blocker can both explode the candidate set
    AND clear the recall floor, so the cap must gate the survivor path too).
    Among the remaining (within-cap) blockers clearing
    ``composition.recall_floor`` on pair recall, the one with the highest
    reduction ratio wins; ties break alphabetically. If none clear the floor,
    the highest-recall within-cap blocker is chosen and
    ``recall_floor_cleared=False`` is returned so callers can surface the
    shortfall. If EVERY blocker exceeds the cap, the smallest candidate set
    wins (still ``False``).

    Parameters
    ----------
    blocker_metrics : dict
        ``{blocker_name: {"pair_recall": ..., "reduction_ratio": ...}}``
        for every blocker that ran on this pair.
    composition : _CompositionConfig
        Parsed composition block from the blocking roster.

    Returns
    -------
    tuple of (str, bool)
        ``(winner_name, recall_floor_cleared)``. ``winner_name`` is a
        valid key in ``blocker_metrics``. ``recall_floor_cleared`` is
        ``True`` when at least one blocker met the floor.
    """
    if not blocker_metrics:
        raise ValueError("_select_best_blocker called with empty blocker_metrics")

    # HARD candidate cap, applied BEFORE the floor/RR logic. An UNCAPPED blocker
    # (e.g. token_blocker) can emit 10-100x the top_k-capped blockers' pairs and
    # — because it is exhaustive — can even *clear the recall floor*, winning via
    # the survivor path and making downstream matching grind for hours. Excluding
    # over-cap blockers up front means neither the survivor nor the fallback path
    # can select one. Only if EVERY blocker exceeds the cap do we keep them all
    # (the caller logs the shortfall and _run_pair warns on the winner's size).
    cap = composition.max_candidates
    pool = {
        name: m
        for name, m in blocker_metrics.items()
        if cap <= 0 or m.get("candidate_count", 0.0) <= cap
    }
    if not pool:
        # Everything exploded — take the smallest candidate set to bound cost.
        smallest = sorted(
            blocker_metrics.items(),
            key=lambda item: (item[1].get("candidate_count", 0.0), item[0]),
        )
        return smallest[0][0], False

    survivors = [
        (name, metrics)
        for name, metrics in pool.items()
        if metrics.get("pair_recall", 0.0) >= composition.recall_floor
    ]
    if survivors:
        survivors.sort(key=lambda item: (-item[1]["reduction_ratio"], item[0]))
        return survivors[0][0], True

    # No within-cap blocker cleared the floor → highest-recall within-cap blocker.
    fallback = sorted(
        pool.items(),
        key=lambda item: (-item[1].get("pair_recall", 0.0), item[0]),
    )
    return fallback[0][0], False


# ---------------------------------------------------------------------------
# EM Committee Runner
# ---------------------------------------------------------------------------


class EMCommitteeRunner(CommitteeRunner):
    """Entity-matching committee runner with split blocking + matching rosters.

    Loads the blocking and matching roster YAMLs, runs the two phases
    sequentially per source pair (select-best blocker → matching on
    winner's candidates), scores predictions against the regenerated /
    pool / test gold surfaces, and computes the pool-agreement diagnostic
    required by ``knobs/cross_cutting.md``.

    Parameters
    ----------
    blocking_roster_path : Path
        Path to ``em_blocking_committee.yaml``.
    matching_roster_path : Path
        Path to ``em_matching_committee.yaml``. May share the
        ``column_mapping`` block with the blocking YAML; if both files
        declare it and they disagree, a ``ValueError`` is raised at
        instantiation so the split invariant is enforced early.
    with_llm : bool
        Enable LLM-based matching members. Default ``False``.
    clustering : {"none", "greedy", "mbm"}
        Post-matching clustering strategy. ``"greedy"`` applies
        :class:`GreedyOneToOneMatchingAlgorithm`; ``"mbm"`` applies
        :class:`MaximumBipartiteMatching`. Default ``"greedy"``.
    retain_predictions_for : set of str, optional
        Matching-member names whose per-pair predictions should be kept
        on the :class:`CommitteeResult`. Downstream fusion reads them
        via ``validate_variant.py``'s ``_extract_fusion_correspondences``.
    """

    stage: Literal["em"] = "em"

    def __init__(
        self,
        blocking_roster_path: Path,
        matching_roster_path: Path,
        *,
        with_llm: bool = False,
        clustering: Literal["none", "greedy", "mbm"] = "greedy",
        retain_predictions_for: set[str] | None = None,
    ) -> None:
        blocking_raw = _load_roster_yaml(blocking_roster_path)
        matching_raw = _load_roster_yaml(matching_roster_path)

        self._blocking_specs = _parse_blocking_roster(blocking_raw["members"])
        self._matching_specs = _parse_matching_roster(
            matching_raw["members"], with_llm=with_llm
        )
        if not self._blocking_specs:
            raise ValueError(
                f"No enabled blockers in {blocking_roster_path}; "
                "refusing to run — the matching phase needs at least one "
                "candidate set to classify."
            )
        if not self._matching_specs:
            raise ValueError(
                f"No enabled matchers in {matching_roster_path} "
                f"(with_llm={with_llm}); refusing to run — the blocker "
                "selection has no consumers."
            )

        blocking_seed = int(blocking_raw.get("seed", 42))
        matching_seed = int(matching_raw.get("seed", blocking_seed))
        if blocking_seed != matching_seed:
            raise ValueError(
                "Blocking and matching rosters declare different seeds "
                f"({blocking_seed} != {matching_seed}); keep them in sync "
                "so variant-level determinism is preserved."
            )
        self._seed = blocking_seed

        self._clustering = clustering
        self._composition = _parse_composition(blocking_raw.get("composition"))
        self._column_mapping: dict[str, dict[str, str]] = _resolve_column_mapping(
            blocking_raw.get("column_mapping"),
            matching_raw.get("column_mapping"),
        )
        self._retain_predictions_for: set[str] = (
            set(retain_predictions_for) if retain_predictions_for else set()
        )

        preprocess_name = blocking_raw.get("preprocess_text") or matching_raw.get(
            "preprocess_text"
        )
        self._preprocess_fn = (
            _PREPROCESS_REGISTRY.get(preprocess_name) if preprocess_name else None
        )

        # Source column the standard blocker keys on. ``name`` for
        # companies/games/music; ``title`` for movies/products. The
        # output column the StandardBlocker spec references is always
        # ``name_first_token`` so the blocker spec stays uniform across
        # domains. Per S10 of plans/plan_s1_scale.md.
        self._blocking_name_column: str = str(
            blocking_raw.get("blocking_name_column")
            or matching_raw.get("blocking_name_column")
            or "name"
        )

        super().__init__(
            roster=list(self._matching_specs),
            config={
                "seed": self._seed,
                "with_llm": with_llm,
                "clustering": clustering,
                "blocking_roster_path": str(blocking_roster_path),
                "matching_roster_path": str(matching_roster_path),
                "recall_floor": self._composition.recall_floor,
                "tie_breaker": self._composition.tie_breaker,
            },
        )

    @property
    def roster_names(self) -> list[str]:
        """Return matching-member names in declaration order.

        The matching roster drives ``aggregated`` and ``per_member``;
        the blocking roster is reported separately under
        :attr:`CommitteeResult.per_blocker`. Call
        :attr:`blocking_roster_names` for the blocker names.
        """
        return [spec.name for spec in self._matching_specs]

    @property
    def blocking_roster_names(self) -> list[str]:
        """Return enabled blocking-member names in declaration order."""
        return [spec.name for spec in self._blocking_specs]

    # -----------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------

    def run(self, bundle: VariantBundle) -> CommitteeResult:
        """Run blocker selection then matching on every source pair.

        Parameters
        ----------
        bundle : VariantBundle
            Loaded variant (baseline or augmented).

        Returns
        -------
        CommitteeResult
            Per-member metrics for matchers under ``per_member``,
            per-blocker metrics under ``per_blocker``, pair-level
            aggregates under ``per_partition``.
        """
        if not bundle.em_gold:
            raise ValueError(
                f"No EM gold for {bundle.domain}/{bundle.level}. "
                "Ensure test CSVs exist in the entitymatching directory."
            )

        _set_deterministic(self._seed)

        effective_column_mapping = bundle.resolve_column_mapping(self._column_mapping)

        matcher_pair_metrics: dict[str, dict[str, dict[str, float]]] = {
            spec.name: {} for spec in self._matching_specs
        }
        matcher_pair_predictions: dict[str, dict[str, pd.DataFrame]] = {
            spec.name: {} for spec in self._matching_specs
        }
        matcher_runtime: dict[str, float] = {
            spec.name: 0.0 for spec in self._matching_specs
        }
        blocker_pair_metrics: dict[str, dict[str, dict[str, float]]] = {
            spec.name: {} for spec in self._blocking_specs
        }
        blocker_runtime: dict[str, float] = {
            spec.name: 0.0 for spec in self._blocking_specs
        }

        per_partition: dict[str, dict[str, Any]] = {}

        t0_total = time.monotonic()

        for pair, gold_df in bundle.em_gold.items():
            src1, src2 = pair
            pair_key = f"{src1}_{src2}"

            try:
                pair_state = self._run_pair(
                    bundle, pair, gold_df, effective_column_mapping
                )
            except Exception:
                logger.exception(
                    "Fatal error running committee on pair %s; recording "
                    "empty metrics for every member.",
                    pair_key,
                )
                pair_state = None

            if pair_state is None:
                for b_spec in self._blocking_specs:
                    blocker_pair_metrics[b_spec.name][pair_key] = {
                        "pair_recall": 0.0,
                        "reduction_ratio": 0.0,
                        "candidate_count": 0.0,
                        "full_space": 0.0,
                        "runtime_s": 0.0,
                        "selected": 0.0,
                        "recall_floor_cleared": 0.0,
                    }
                for m_spec in self._matching_specs:
                    matcher_pair_metrics[m_spec.name][
                        pair_key
                    ] = _empty_matcher_metrics()
                    if m_spec.name in self._retain_predictions_for:
                        matcher_pair_predictions[m_spec.name][pair_key] = pd.DataFrame(
                            columns=["id1", "id2", "score"]
                        )
                per_partition[pair_key] = {"error": 1.0}
                continue

            winner_name = pair_state["winner_name"]
            recall_cleared = pair_state["recall_floor_cleared"]

            for b_name, b_metrics in pair_state["blocker_metrics"].items():
                blocker_pair_metrics[b_name][pair_key] = {
                    **b_metrics,
                    "selected": 1.0 if b_name == winner_name else 0.0,
                    "recall_floor_cleared": 1.0 if recall_cleared else 0.0,
                }
                blocker_runtime[b_name] += float(b_metrics.get("runtime_s", 0.0))

            for m_name, (m_metrics, m_preds, m_runtime) in pair_state[
                "matcher_outputs"
            ].items():
                matcher_pair_metrics[m_name][pair_key] = m_metrics
                matcher_runtime[m_name] += m_runtime
                if m_name in self._retain_predictions_for:
                    matcher_pair_predictions[m_name][pair_key] = m_preds

            per_partition[pair_key] = {
                "winner_blocker": winner_name,
                "winner_pair_recall": pair_state["blocker_metrics"][winner_name][
                    "pair_recall"
                ],
                "winner_reduction_ratio": pair_state["blocker_metrics"][winner_name][
                    "reduction_ratio"
                ],
                "recall_floor_cleared": 1.0 if recall_cleared else 0.0,
            }

        total_runtime = time.monotonic() - t0_total

        per_member = self._build_per_member_results(
            matcher_pair_metrics, matcher_pair_predictions, matcher_runtime
        )
        per_blocker = self._build_per_blocker_results(
            blocker_pair_metrics, blocker_runtime
        )

        aggregated = _compute_aggregated(per_member)

        for pair_key in per_partition:
            f1s = [
                matcher_pair_metrics[m_name][pair_key].get("f1", 0.0)
                for m_name in matcher_pair_metrics
                if pair_key in matcher_pair_metrics[m_name]
                and not _is_nan(matcher_pair_metrics[m_name][pair_key].get("f1"))
            ]
            n = len(f1s)
            per_partition[pair_key].update(
                {
                    "macro_f1": sum(f1s) / n if n else float("nan"),
                    "min_f1": min(f1s) if n else float("nan"),
                    "max_f1": max(f1s) if n else float("nan"),
                    "n_members": float(n),
                }
            )

        return CommitteeResult(
            stage="em",
            domain=bundle.domain,
            level=bundle.level,
            per_member=per_member,
            per_blocker=per_blocker,
            aggregated=aggregated,
            per_attribute={},
            per_partition={k: _coerce_partition(v) for k, v in per_partition.items()},
            runtime_s=total_runtime,
            roster=self.roster_names,
        )

    # -----------------------------------------------------------------
    # Per-pair execution
    # -----------------------------------------------------------------

    def _run_pair(
        self,
        bundle: VariantBundle,
        pair: tuple[str, str],
        gold_df: pd.DataFrame,
        column_mapping: dict[str, dict[str, str]],
    ) -> dict[str, Any]:
        """Run Phase 1 (blocker selection) and Phase 2 (matching) for one pair.

        Parameters
        ----------
        bundle : VariantBundle
            Full variant bundle.
        pair : tuple of str
            ``(src1, src2)`` source pair.
        gold_df : DataFrame
            Test gold for this pair.
        column_mapping : dict
            Effective per-source column rename map (already resolved
            through any K8 renames).

        Returns
        -------
        dict
            ``{"winner_name", "recall_floor_cleared", "blocker_metrics",
            "matcher_outputs"}`` where ``matcher_outputs`` is
            ``{matcher_name: (metrics_dict, predictions_df, runtime_s)}``.
        """
        src1, src2 = pair
        df_left = bundle.sources[src1].copy()
        df_right = bundle.sources[src2].copy()
        id_column = "id"

        if column_mapping:
            left_map = column_mapping.get(src1, {})
            if left_map:
                df_left = apply_column_mapping(df_left, left_map)
            right_map = column_mapping.get(src2, {})
            if right_map:
                df_right = apply_column_mapping(df_right, right_map)

        # Collect every blocking-key column requested by enabled blockers.
        # StandardBlocker's ``on`` list and SortedNeighbourhoodBlocker's
        # ``key`` field both name derived columns; the pattern-based
        # generator in :func:`_generate_blocking_keys` resolves
        # ``<col>_first_token`` / ``<col>_first_<N>`` / ``<col>_norm``
        # against the configured ``blocking_name_column``.
        required_keys: set[str] = set()
        for spec in self._blocking_specs:
            cls = spec.blocker_spec.get("class")
            params = spec.blocker_spec.get("params", {}) or {}
            if cls == "StandardBlocker":
                on_keys = params.get("on", []) or []
                if isinstance(on_keys, str):
                    on_keys = [on_keys]
                required_keys.update(str(k) for k in on_keys)
            elif cls == "SortedNeighbourhoodBlocker":
                key = params.get("key")
                if key:
                    required_keys.add(str(key))
        if required_keys:
            _generate_blocking_keys(
                df_left,
                column=self._blocking_name_column,
                required_keys=sorted(required_keys),
            )
            _generate_blocking_keys(
                df_right,
                column=self._blocking_name_column,
                required_keys=sorted(required_keys),
            )

        n_left = len(df_left)
        n_right = len(df_right)

        blocker_metrics: dict[str, dict[str, float]] = {}
        blocker_candidates: dict[str, pd.DataFrame] = {}

        for b_spec in self._blocking_specs:
            t0 = time.monotonic()
            try:
                blocker = _build_blocker(
                    b_spec.blocker_spec, df_left, df_right, id_column
                )
                candidates = blocker.materialize()
            except Exception:
                logger.exception(
                    "Blocker %s failed on pair %s-%s", b_spec.name, src1, src2
                )
                candidates = pd.DataFrame(columns=["id1", "id2"])
            elapsed = time.monotonic() - t0

            recall = blocking_pair_recall(candidates, gold_df)
            if n_left > 0 and n_right > 0:
                rr = reduction_ratio(candidates, n_left, n_right)
            else:
                rr = {
                    "reduction_ratio": 0.0,
                    "candidate_count": float(len(candidates)),
                    "full_space": 0.0,
                }

            blocker_metrics[b_spec.name] = {
                "pair_recall": recall["pair_recall"],
                "gold_positives": recall["gold_positives"],
                "covered": recall["covered"],
                "missed": recall["missed"],
                "reduction_ratio": rr["reduction_ratio"],
                "candidate_count": rr["candidate_count"],
                "full_space": rr["full_space"],
                "runtime_s": elapsed,
            }
            blocker_candidates[b_spec.name] = candidates

        winner_name, recall_cleared = _select_best_blocker(
            blocker_metrics, self._composition
        )
        if not recall_cleared:
            logger.warning(
                "No blocker cleared recall_floor=%.2f for pair %s-%s; "
                "falling back to highest-recall blocker '%s' (recall=%.3f).",
                self._composition.recall_floor,
                src1,
                src2,
                winner_name,
                blocker_metrics[winner_name]["pair_recall"],
            )

        winner_candidates = blocker_candidates[winner_name]

        # Loud guardrail: even after count-aware selection, if the winner's
        # candidate set is enormous (e.g. every blocker exceeded the cap on a
        # very large / heavily-perturbed pair), downstream Ditto/Magellan
        # matching will grind for hours. Surface it instead of silently
        # stalling so the run is diagnosable.
        cap = self._composition.max_candidates
        n_winner_cands = len(winner_candidates)
        if cap > 0 and n_winner_cands > cap:
            logger.warning(
                "EM blocking pair %s-%s: winner '%s' emitted %d candidate pairs "
                "(> max_candidates=%d); downstream matching may be very slow. "
                "Consider a smaller top_k, a tighter recall_floor, or raising "
                "the cap intentionally.",
                src1,
                src2,
                winner_name,
                n_winner_cands,
                cap,
            )

        # R10-F: use the variant-aware resolver so that at variant levels a
        # trainable matcher trains on the K2-regenerated
        # ``<pair>_train_corner_filled.csv`` (matched distribution) rather
        # than the baseline train. At baseline level this returns the
        # baseline train path unchanged, so this is a no-op there. (The
        # split-out EMMatchingCommitteeRunner already used the variant
        # resolver; this aligns the bundled runner.)
        pair_train_path, _ = _resolve_variant_train_path(bundle, pair)

        matcher_outputs: dict[str, tuple[dict[str, float], pd.DataFrame, float]] = {}
        for m_spec in self._matching_specs:
            t0 = time.monotonic()
            try:
                preds = self._run_matcher(
                    m_spec,
                    df_left,
                    df_right,
                    winner_candidates,
                    pair_train_path=pair_train_path,
                )
            except Exception:
                logger.exception(
                    "Matcher %s failed on pair %s-%s", m_spec.name, src1, src2
                )
                preds = pd.DataFrame(columns=["id1", "id2", "score"])
            elapsed = time.monotonic() - t0

            metrics = self._score_predictions(preds, bundle, pair, gold_df)
            matcher_outputs[m_spec.name] = (metrics, preds, elapsed)

        return {
            "winner_name": winner_name,
            "recall_floor_cleared": recall_cleared,
            "blocker_metrics": blocker_metrics,
            "matcher_outputs": matcher_outputs,
        }

    def _run_matcher(
        self,
        spec: _EMMatchingRosterMember,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        candidates: pd.DataFrame,
        *,
        pair_train_path: Path | None = None,
    ) -> pd.DataFrame:
        """Run a single matching member against the winning candidate set.

        Parameters
        ----------
        spec : _EMMatchingRosterMember
            Parsed matcher roster entry.
        df_left, df_right : DataFrame
            Source frames (already column-mapped).
        candidates : DataFrame
            Winning blocker's candidate pairs.
        pair_train_path : Path or None
            Per-source-pair training CSV path. Injected into matcher
            params at instantiation when the class needs per-pair train
            data (see :data:`_PER_PAIR_TRAIN_INJECTION`). When the
            matcher class needs per-pair train data and the path is
            missing or does not exist, the matcher is skipped on this
            pair and an empty correspondence frame is returned.

        Returns
        -------
        DataFrame
            Correspondence predictions with ``id1``, ``id2``, ``score``.
            Empty when the candidate set is empty, the matcher drops
            everything, or per-pair training data is unavailable for a
            matcher that requires it.
        """
        if candidates.empty:
            return pd.DataFrame(columns=["id1", "id2", "score"])

        matcher_class = str(spec.matcher_spec.get("class", ""))
        if matcher_class in _PER_PAIR_TRAIN_INJECTION:
            if pair_train_path is None or not pair_train_path.exists():
                logger.warning(
                    "Skipping %s on pair without per-pair train data (%s)",
                    spec.name,
                    pair_train_path,
                )
                return pd.DataFrame(columns=["id1", "id2", "score"])

        comparators = _build_comparators(spec.comparator_specs, self._preprocess_fn)
        matcher = _build_matcher(spec.matcher_spec, pair_train_path=pair_train_path)

        match_kwargs: dict[str, Any] = {
            "id_column": "id",
            "threshold": spec.threshold,
        }
        if comparators:
            match_kwargs["comparators"] = comparators
        if spec.weights:
            match_kwargs["weights"] = spec.weights

        preds = matcher.match(df_left, df_right, candidates, **match_kwargs)

        if self._clustering == "greedy" and not preds.empty:
            preds = GreedyOneToOneMatchingAlgorithm(threshold=0.0).cluster(preds)
        elif self._clustering == "mbm" and not preds.empty:
            preds = MaximumBipartiteMatching().cluster(preds)

        return preds

    def _score_predictions(
        self,
        preds: pd.DataFrame,
        bundle: VariantBundle,
        pair: tuple[str, str],
        gold_df: pd.DataFrame,
    ) -> dict[str, float]:
        """Assemble the merged matcher metric dict for a single pair.

        Two closed-set EM surfaces are surfaced (plan_revision.md C10):

        1. **``f1_baseline_test``** — closed-set on
           ``em_test_baseline_pruned.csv`` (Set 1 per C11). The
           original baseline test gold, pruned of any pair whose id
           dropped post-K2/K3/K4. Smaller and corner-biased as K2
           intensity rises; a per-level reference value, **not** the
           monotonicity verdict surface.
        2. **``f1_regen_test``** — closed-set on
           ``em_test_corner_filled.csv`` (Set 2 per C11). The
           baseline-pruned split backfilled with corner-mined pairs
           to the original size. Headline F1 and the load-bearing
           surface for EM committee macro_f1 monotonicity.

        Pool-as-gold F1 (``f1_vs_pool``) and pool-agreement
        diagnostics are retained unchanged for M8 collapse-vs-hidden-
        positive analysis. The primary ``f1`` field falls back
        through the chain ``regen_test → baseline_test → pool``.
        ``pool`` is reached only for baselines (no K2 regen at all)
        or when both regen test versions are missing.

        Open-set ``f1_vs_test_gold`` is retired (C10); on large
        domains predicting many more pairs than the small human-gold
        positive list, every prediction outside that list counted as
        FP and the number was uninterpretable.

        The corner-filled val split is read separately and used as a
        val/test agreement sanity check — logged at debug level on
        divergence but not surfaced in the public per-member metrics.

        Parameters
        ----------
        preds : DataFrame
            Predicted correspondences.
        bundle : VariantBundle
            Source of regenerated gold, pool and original gold.
        pair : tuple of str
            Source pair being scored.
        gold_df : DataFrame
            Original human-annotated test gold for this pair
            (from ``bundle.em_gold``). Unused for scoring under C10
            but kept in the signature so callers can still pass it.

        Returns
        -------
        dict[str, float]
            Flat metric dict.
        """
        del gold_df  # retired open-set surface (C10).

        # plan_revision.md C11 schema: bundle.em_gold_regenerated is
        # {pair: {split: {version: DataFrame}}}.
        regen_splits = bundle.em_gold_regenerated.get(pair, {})
        regen_test_versions = regen_splits.get("test", {})
        regen_val_versions = regen_splits.get("val", {})
        regen_test_corner = regen_test_versions.get("corner_filled")
        regen_test_baseline = regen_test_versions.get("baseline_pruned")
        regen_val_corner = regen_val_versions.get("corner_filled")

        pool_f1 = score_em_vs_pool(preds, bundle.pooled_positives, pair)
        pool_diag = pool_agreement(preds, bundle.pooled_positives)

        def _closed_or_nan(
            gold: pd.DataFrame | None,
        ) -> dict[str, float]:
            """Score closed-set against *gold* or emit NaNs when absent."""
            if gold is None or gold.empty:
                return {
                    "f1": float("nan"),
                    "precision": float("nan"),
                    "recall": float("nan"),
                    "tp": float("nan"),
                    "fp": float("nan"),
                    "fn": float("nan"),
                    "pred_scoped": float("nan"),
                }
            return score_em_correspondences_closed_set(preds, gold)

        regen_test_metrics = _closed_or_nan(regen_test_corner)
        baseline_test_metrics = _closed_or_nan(regen_test_baseline)
        # Internal val/test agreement sanity check (C10): logged on
        # divergence, not surfaced in the public metric dict.
        regen_val_metrics = _closed_or_nan(regen_val_corner)
        if (
            regen_val_corner is not None
            and not regen_val_corner.empty
            and regen_test_corner is not None
            and not regen_test_corner.empty
            and not _is_nan(regen_val_metrics["f1"])
            and not _is_nan(regen_test_metrics["f1"])
            and abs(regen_val_metrics["f1"] - regen_test_metrics["f1"]) > 0.15
        ):
            logger.debug(
                "EM regen val/test divergence > 0.15 on pair %s: "
                "val_f1=%.3f test_f1=%.3f",
                pair,
                regen_val_metrics["f1"],
                regen_test_metrics["f1"],
            )

        # Headline fallback chain (C10): regen_test → baseline_test
        # → pool. Pool is the last-resort surface for baselines
        # (no K2 regen at all) or domains whose regen builder
        # produced nothing.
        if regen_test_corner is not None and not regen_test_corner.empty:
            primary = {
                "f1": regen_test_metrics["f1"],
                "precision": regen_test_metrics["precision"],
                "recall": regen_test_metrics["recall"],
                "tp": regen_test_metrics["tp"],
                "fp": regen_test_metrics["fp"],
                "fn": regen_test_metrics["fn"],
            }
        elif regen_test_baseline is not None and not regen_test_baseline.empty:
            primary = {
                "f1": baseline_test_metrics["f1"],
                "precision": baseline_test_metrics["precision"],
                "recall": baseline_test_metrics["recall"],
                "tp": baseline_test_metrics["tp"],
                "fp": baseline_test_metrics["fp"],
                "fn": baseline_test_metrics["fn"],
            }
        else:
            primary = pool_f1

        return {
            **primary,
            "f1_baseline_test": baseline_test_metrics["f1"],
            "precision_baseline_test": baseline_test_metrics["precision"],
            "recall_baseline_test": baseline_test_metrics["recall"],
            "pred_scoped_baseline_test": baseline_test_metrics["pred_scoped"],
            "f1_regen_test": regen_test_metrics["f1"],
            "precision_regen_test": regen_test_metrics["precision"],
            "recall_regen_test": regen_test_metrics["recall"],
            "pred_scoped_regen_test": regen_test_metrics["pred_scoped"],
            "f1_vs_pool": pool_f1["f1"],
            "precision_vs_pool": pool_f1["precision"],
            "recall_vs_pool": pool_f1["recall"],
            **pool_diag,
        }

    # -----------------------------------------------------------------
    # Result assembly
    # -----------------------------------------------------------------

    def _build_per_member_results(
        self,
        matcher_pair_metrics: dict[str, dict[str, dict[str, float]]],
        matcher_pair_predictions: dict[str, dict[str, pd.DataFrame]],
        matcher_runtime: dict[str, float],
    ) -> dict[str, MemberResult]:
        """Fold per-pair matcher metrics into :class:`MemberResult` entries."""
        per_member: dict[str, MemberResult] = {}
        for spec in self._matching_specs:
            pair_metrics = matcher_pair_metrics[spec.name]
            member_metrics = _aggregate_member_pairs(pair_metrics)
            retain = spec.name in self._retain_predictions_for
            per_member[spec.name] = MemberResult(
                name=spec.name,
                predictions=(matcher_pair_predictions[spec.name] if retain else None),
                metrics=member_metrics,
                runtime_s=matcher_runtime[spec.name],
                notes={
                    "role": "matcher",
                    "matching_type": spec.matching_type,
                    "missing_value_tolerant": spec.missing_value_tolerant,
                    "per_pair": pair_metrics,
                },
            )
        return per_member

    def _build_per_blocker_results(
        self,
        blocker_pair_metrics: dict[str, dict[str, dict[str, float]]],
        blocker_runtime: dict[str, float],
    ) -> dict[str, MemberResult]:
        """Fold per-pair blocker metrics into :class:`MemberResult` entries."""
        per_blocker: dict[str, MemberResult] = {}
        for spec in self._blocking_specs:
            pair_metrics = blocker_pair_metrics[spec.name]
            member_metrics = _aggregate_blocker_pairs(pair_metrics)
            per_blocker[spec.name] = MemberResult(
                name=spec.name,
                predictions=None,
                metrics=member_metrics,
                runtime_s=blocker_runtime[spec.name],
                notes={
                    "role": "blocker",
                    "blocking_type": spec.blocking_type,
                    "per_pair": pair_metrics,
                },
            )
        return per_blocker


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


_MATCHER_AVG_KEYS = [
    "f1",
    "precision",
    "recall",
    "f1_val",
    "precision_val",
    "recall_val",
    "pool_precision",
    "pool_recall",
    "f1_baseline_test",
    "precision_baseline_test",
    "recall_baseline_test",
    "pred_scoped_baseline_test",
    "f1_regen_test",
    "precision_regen_test",
    "recall_regen_test",
    "pred_scoped_regen_test",
    "f1_vs_pool",
    "precision_vs_pool",
    "recall_vs_pool",
]


_BLOCKER_AVG_KEYS = [
    "pair_recall",
    "pair_recall_val",
    "reduction_ratio",
    "candidate_count",
    "gold_positives",
    "covered",
    "missed",
    # R7b dual-model dual-test cells — stored per-pair (committee_em.py
    # ~2168) but previously omitted here, so the aggregated
    # macro_pair_recall_*_model_on_*_test fields rolled up as 0.0. Include
    # them so the frozen-baseline blocking surface
    # (macro_pair_recall_baseline_model_on_regen_test) is populated,
    # symmetric with the matching headline. For the 5 non-trainable blockers
    # the four cells alias (variant==baseline); only sc_block carries a
    # genuine variant-vs-baseline distinction.
    "pair_recall_baseline_model_on_baseline_test",
    "pair_recall_baseline_model_on_regen_test",
    "pair_recall_variant_model_on_baseline_test",
    "pair_recall_variant_model_on_regen_test",
]


def _empty_matcher_metrics() -> dict[str, float]:
    """Zeros for every matcher metric key — used for error-path rows."""
    base = {k: 0.0 for k in _MATCHER_AVG_KEYS}
    base.update(
        {
            "tp": 0.0,
            "fp": 0.0,
            "fn": 0.0,
            "pool_overlap": 0.0,
        }
    )
    return base


def _aggregate_member_pairs(
    pair_metrics: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Macro-average matcher metrics across source pairs.

    NaN sentinels (emitted when the regenerated gold is absent) are
    dropped before averaging so a baseline variant without regen gold
    does not poison the average for augmented variants.

    Parameters
    ----------
    pair_metrics : dict
        ``{pair_key: {metric: value}}``.

    Returns
    -------
    dict[str, float]
        Member-level aggregated metrics.
    """
    if not pair_metrics:
        return {k: 0.0 for k in _MATCHER_AVG_KEYS}

    result: dict[str, float] = {}
    for key in _MATCHER_AVG_KEYS:
        values = [
            float(pm.get(key, 0.0))
            for pm in pair_metrics.values()
            if not _is_nan(pm.get(key))
        ]
        result[key] = float("nan") if not values else sum(values) / len(values)
    return result


def _aggregate_blocker_pairs(
    pair_metrics: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Macro-average blocker metrics + selection-rate across source pairs.

    Parameters
    ----------
    pair_metrics : dict
        ``{pair_key: {"pair_recall": ..., "reduction_ratio": ...,
        "selected": 0|1, ...}}``.

    Returns
    -------
    dict[str, float]
        Macro averages plus ``selection_rate`` (fraction of pairs where
        this blocker was chosen as the winner) and
        ``recall_floor_clear_rate`` (fraction of pairs where *some*
        blocker cleared the floor).
    """
    if not pair_metrics:
        return {k: 0.0 for k in _BLOCKER_AVG_KEYS} | {
            "selection_rate": 0.0,
            "recall_floor_clear_rate": 0.0,
        }

    result: dict[str, float] = {}
    for key in _BLOCKER_AVG_KEYS:
        values = [float(pm.get(key, 0.0)) for pm in pair_metrics.values()]
        result[key] = sum(values) / len(values) if values else 0.0

    selected = [float(pm.get("selected", 0.0)) for pm in pair_metrics.values()]
    cleared = [
        float(pm.get("recall_floor_cleared", 0.0)) for pm in pair_metrics.values()
    ]
    result["selection_rate"] = sum(selected) / len(selected) if selected else 0.0
    result["recall_floor_clear_rate"] = sum(cleared) / len(cleared) if cleared else 0.0
    return result


def _is_nan(value: Any) -> bool:
    """Return ``True`` if *value* is ``float('nan')``; never for other types."""
    try:
        return bool(np.isnan(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def _compute_aggregated(
    per_member: dict[str, MemberResult],
) -> dict[str, float]:
    """Compute committee-level aggregated metrics from matcher members."""
    if not per_member:
        return {
            "macro_f1": 0.0,
            "min_f1": 0.0,
            "max_f1": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_pool_precision": 0.0,
            "macro_pool_recall": 0.0,
        }

    def _macro(key: str) -> float:
        values = [
            float(m.metrics.get(key, 0.0))
            for m in per_member.values()
            if not _is_nan(m.metrics.get(key))
        ]
        return sum(values) / len(values) if values else float("nan")

    # NaN-skip the headline macros + min/max so a single skipped member
    # (e.g. magellan with no per-pair train CSV) cannot poison the whole
    # committee aggregate to NaN. Mirrors the per-key ``_macro`` helper.
    f1s = [
        float(m.metrics.get("f1", 0.0))
        for m in per_member.values()
        if not _is_nan(m.metrics.get("f1"))
    ]

    return {
        "macro_f1": _macro("f1"),
        "min_f1": min(f1s) if f1s else float("nan"),
        "max_f1": max(f1s) if f1s else float("nan"),
        "macro_precision": _macro("precision"),
        "macro_recall": _macro("recall"),
        "macro_pool_precision": _macro("pool_precision"),
        "macro_pool_recall": _macro("pool_recall"),
        "macro_f1_baseline_test": _macro("f1_baseline_test"),
        "macro_f1_regen_test": _macro("f1_regen_test"),
        "macro_f1_vs_pool": _macro("f1_vs_pool"),
    }


def _coerce_partition(raw: dict[str, Any]) -> dict[str, float]:
    """Coerce a per-partition entry to the ``dict[str, float]`` contract.

    ``winner_blocker`` is the only non-float field carried today; it is
    rendered through ``hash()`` would be lossy, so it is dropped from
    the flat dict and exposed separately via the blocker notes. Float
    fields pass through.
    """
    out: dict[str, float] = {}
    for key, value in raw.items():
        if isinstance(value, (int, float)):
            out[key] = float(value)
    return out


def _resolve_column_mapping(
    blocking: dict[str, dict[str, str]] | None,
    matching: dict[str, dict[str, str]] | None,
) -> dict[str, dict[str, str]]:
    """Reconcile column_mapping declared in both rosters.

    The two YAMLs must either (a) both omit the block, (b) have exactly
    one declare it, or (c) agree exactly when both declare it. Anything
    else raises at instantiation time so the split invariant is
    enforced early rather than silently divergent.

    Parameters
    ----------
    blocking, matching : dict or None
        ``column_mapping`` block from each roster.

    Returns
    -------
    dict[str, dict[str, str]]
        Merged column mapping.

    Raises
    ------
    ValueError
        If the blocking / matching blocks disagree, or if any mapping
        entry would rename the ``id`` join key (see
        :func:`_validate_no_id_rename`).
    """
    if blocking is None and matching is None:
        resolved: dict[str, dict[str, str]] = {}
    elif blocking is None:
        resolved = dict(matching or {})
    elif matching is None:
        resolved = dict(blocking)
    elif blocking != matching:
        raise ValueError(
            "column_mapping blocks in the blocking and matching roster "
            "YAMLs disagree; keep them in sync. Blocking file keys: "
            f"{sorted((blocking or {}).keys())}, matching file keys: "
            f"{sorted((matching or {}).keys())}."
        )
    else:
        resolved = dict(blocking)
    _validate_no_id_rename(resolved)
    return resolved


def _validate_no_id_rename(mapping: dict[str, dict[str, str]]) -> None:
    """Reject any column_mapping entry that touches the ``id`` join key.

    Every blocker / matcher joins source rows to gold pairs on a hardcoded
    ``id_column = "id"``. A mapping entry that renames ``id`` away
    (``{id: other}``) makes the join column disappear (a loud
    ``"must have 'id' column"`` downstream), and one that renames *onto*
    ``id`` (``{other: id}``) silently *drops* the real id column and
    overwrites it with another column's values
    (see :func:`column_mapping.apply_column_mapping`), corrupting the join
    with no error. Reject both at committee-load time with a clear message
    rather than failing confusingly (or silently) at run time.
    """
    for source, source_map in mapping.items():
        for src_col, tgt_col in source_map.items():
            if src_col == tgt_col:
                continue  # identity rename is a harmless no-op
            if src_col == "id":
                raise ValueError(
                    f"column_mapping for {source!r} must not rename the 'id' "
                    f"join key away ({src_col!r} -> {tgt_col!r}); 'id' is the "
                    "hardcoded blocker/matcher join column and must be preserved."
                )
            if tgt_col == "id":
                raise ValueError(
                    f"column_mapping for {source!r} must not rename {src_col!r} "
                    "to 'id'; that would drop the real 'id' join key and "
                    "silently overwrite it with another column's values."
                )


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------


def _load_roster_yaml(path: Path) -> dict[str, Any]:
    """Load and return the roster YAML."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Perfect-prior-step committees (2026-05-13)
#
# Split out from :class:`EMCommitteeRunner` per the user directive to
# measure each pipeline step under the assumption of a perfect prior
# step. The bundled runner conflated blocking + matching into one F1;
# the split runners isolate each signal:
#
# - :class:`EMBlockingCommitteeRunner` — recall + reduction_ratio on the
#   full source DataFrames, scored against the test gold positives.
# - :class:`EMMatchingCommitteeRunner` — pure pair-classification on the
#   labelled splits. Trains on ``_train.csv`` (via the existing
#   ``pair_train_path`` injection), scores on ``_val.csv`` (primary) and
#   ``_test.csv`` (secondary) under the closed-set semantic so the
#   labelled negatives provide precision.
#
# The bundled :class:`EMCommitteeRunner` stays for
# ``scripts/validate_variant.py`` (which validates the realistic end-to-
# end pipeline against a variant) and its unit tests.
# ---------------------------------------------------------------------------


class EMBlockingCommitteeRunner(CommitteeRunner):
    """Blocking-only committee runner under the perfect-prior-step design.

    Runs every blocker on the full source DataFrames and emits per-pair
    + per-blocker ``pair_recall`` + ``reduction_ratio``. No matchers are
    instantiated; no winner is selected. Difficulty signal: pair_recall
    should stay above the composition ``recall_floor`` (typically 0.97)
    from easy → hard variants, while reduction_ratio is allowed to
    degrade.

    Parameters
    ----------
    blocking_roster_path : Path
        Path to ``em_blocking_committee.yaml``.
    """

    stage: Literal["em_blocking"] = "em_blocking"  # type: ignore[assignment]

    def __init__(self, blocking_roster_path: Path) -> None:
        blocking_raw = _load_roster_yaml(blocking_roster_path)
        self._blocking_specs = _parse_blocking_roster(blocking_raw["members"])
        if not self._blocking_specs:
            raise ValueError(
                f"No enabled blockers in {blocking_roster_path}; " "refusing to run."
            )
        self._seed = int(blocking_raw.get("seed", 42))
        self._composition = _parse_composition(blocking_raw.get("composition"))
        self._column_mapping: dict[str, dict[str, str]] = _resolve_column_mapping(
            blocking_raw.get("column_mapping"),
            None,
        )
        preprocess_name = blocking_raw.get("preprocess_text")
        self._preprocess_fn = (
            _PREPROCESS_REGISTRY.get(preprocess_name) if preprocess_name else None
        )
        self._blocking_name_column: str = str(
            blocking_raw.get("blocking_name_column") or "name"
        )

        super().__init__(
            roster=list(self._blocking_specs),
            config={
                "seed": self._seed,
                "blocking_roster_path": str(blocking_roster_path),
                "recall_floor": self._composition.recall_floor,
                "tie_breaker": self._composition.tie_breaker,
            },
        )

    @property
    def roster_names(self) -> list[str]:
        return [s.name for s in self._blocking_specs]

    def run(self, bundle: VariantBundle) -> CommitteeResult:
        """Evaluate every blocker on every source pair."""
        if not bundle.em_gold:
            raise ValueError(
                f"No EM gold for {bundle.domain}/{bundle.level}. "
                "Need a test gold CSV to compute pair_recall."
            )

        _set_deterministic(self._seed)
        effective_column_mapping = bundle.resolve_column_mapping(self._column_mapping)

        blocker_pair_metrics: dict[str, dict[str, dict[str, float]]] = {
            spec.name: {} for spec in self._blocking_specs
        }
        blocker_runtime: dict[str, float] = {
            spec.name: 0.0 for spec in self._blocking_specs
        }
        per_partition: dict[str, dict[str, Any]] = {}

        t0_total = time.monotonic()

        for pair, gold_df in bundle.em_gold.items():
            src1, src2 = pair
            pair_key = f"{src1}_{src2}"

            df_left = bundle.sources[src1].copy()
            df_right = bundle.sources[src2].copy()
            if effective_column_mapping:
                left_map = effective_column_mapping.get(src1, {})
                if left_map:
                    df_left = apply_column_mapping(df_left, left_map)
                right_map = effective_column_mapping.get(src2, {})
                if right_map:
                    df_right = apply_column_mapping(df_right, right_map)

            required_keys: set[str] = set()
            for spec in self._blocking_specs:
                cls = spec.blocker_spec.get("class")
                params = spec.blocker_spec.get("params", {}) or {}
                if cls == "StandardBlocker":
                    on_keys = params.get("on", []) or []
                    if isinstance(on_keys, str):
                        on_keys = [on_keys]
                    required_keys.update(str(k) for k in on_keys)
                elif cls == "SortedNeighbourhoodBlocker":
                    key = params.get("key")
                    if key:
                        required_keys.add(str(key))
            if required_keys:
                _generate_blocking_keys(
                    df_left,
                    column=self._blocking_name_column,
                    required_keys=sorted(required_keys),
                )
                _generate_blocking_keys(
                    df_right,
                    column=self._blocking_name_column,
                    required_keys=sorted(required_keys),
                )

            n_left = len(df_left)
            n_right = len(df_right)

            # R7b dual-model: load both gold surfaces (baseline_pruned
            # for "original test minus dropped" + corner_filled for
            # K2-mined regen). At baseline level both fall back to the
            # same PyDI test gold. Use ``is None`` checks instead of
            # ``or`` — the loader returns a DataFrame or None, and
            # ``df or fallback`` raises ``ValueError: The truth value
            # of a DataFrame is ambiguous`` on a non-empty frame.
            _test_baseline = _load_labelled_split_from_bundle(
                bundle, pair, "test", version="baseline_pruned"
            )
            test_gold_baseline = gold_df if _test_baseline is None else _test_baseline
            _test_corner = _load_labelled_split_from_bundle(
                bundle, pair, "test", version="corner_filled"
            )
            test_gold_corner = gold_df if _test_corner is None else _test_corner
            # VAL surface (user spec: score val AND test). Shared loader, so
            # games (no shipped val) gets the stratified train hold-out.
            val_gold_corner = _load_labelled_split_from_bundle(
                bundle, pair, "val", version="corner_filled"
            )

            # R10-F: surface the otherwise-silent fallback to baseline gold.
            # At variant levels the regenerated splits MUST be present (the
            # package_variant glob copies *_{baseline_pruned,corner_filled}.csv);
            # if they are missing here the dual-test surfaces collapse onto
            # the baseline gold and become indistinguishable. Warn loudly so
            # a future regression in the glob/copy step is not invisible.
            if bundle.level != "baseline" and (
                _test_baseline is None or _test_corner is None
            ):
                logger.warning(
                    "EM blocking pair %s-%s at level %s: regenerated test gold "
                    "missing (baseline_pruned=%s, corner_filled=%s); falling "
                    "back to baseline gold -- dual-test surfaces will be "
                    "indistinguishable. Check package_variant copied the "
                    "*_{baseline_pruned,corner_filled}.csv files (R10-F).",
                    pair[0],
                    pair[1],
                    bundle.level,
                    _test_baseline is not None,
                    _test_corner is not None,
                )

            pair_metrics_by_blocker: dict[str, dict[str, float]] = {}
            for b_spec in self._blocking_specs:
                t0 = time.monotonic()
                # R7b: decide whether to build a separate variant
                # blocker instance. Trainable blockers (sc_block) with a
                # variant checkpoint at <cache>/variant_<level>/best
                # get a second instance; everything else aliases to
                # baseline (non-trainable blockers have no model axis).
                baseline_ckpt = b_spec.blocker_spec.get("params", {}).get(
                    "checkpoint_path"
                )
                variant_ckpt, variant_ckpt_distinct = _resolve_variant_checkpoint_path(
                    baseline_ckpt, bundle.level
                )

                try:
                    baseline_blocker = _build_blocker(
                        b_spec.blocker_spec, df_left, df_right, "id"
                    )
                    baseline_candidates = baseline_blocker.materialize()
                except Exception:
                    logger.exception(
                        "Blocker %s (baseline model) failed on pair %s-%s",
                        b_spec.name,
                        src1,
                        src2,
                    )
                    baseline_candidates = pd.DataFrame(columns=["id1", "id2"])

                if variant_ckpt_distinct:
                    try:
                        variant_blocker = _build_blocker(
                            b_spec.blocker_spec,
                            df_left,
                            df_right,
                            "id",
                            checkpoint_override=variant_ckpt,
                        )
                        variant_candidates = variant_blocker.materialize()
                    except Exception:
                        logger.exception(
                            "Blocker %s (variant model) failed on pair %s-%s",
                            b_spec.name,
                            src1,
                            src2,
                        )
                        variant_candidates = baseline_candidates
                        variant_ckpt_distinct = False
                else:
                    variant_candidates = baseline_candidates

                elapsed = time.monotonic() - t0

                def _recall_rr(cands: pd.DataFrame, gold: pd.DataFrame) -> dict:
                    r = blocking_pair_recall(cands, gold)
                    rr_ = (
                        reduction_ratio(cands, n_left, n_right)
                        if n_left > 0 and n_right > 0
                        else {
                            "reduction_ratio": 0.0,
                            "candidate_count": float(len(cands)),
                            "full_space": 0.0,
                        }
                    )
                    return {
                        "pair_recall": r["pair_recall"],
                        "gold_positives": r["gold_positives"],
                        "covered": r["covered"],
                        "missed": r["missed"],
                        "reduction_ratio": rr_["reduction_ratio"],
                        "candidate_count": rr_["candidate_count"],
                        "full_space": rr_["full_space"],
                    }

                # R7b 4-cell metric matrix. The load-bearing surface for
                # blocking monotonicity is
                # ``pair_recall_variant_model_on_regen_test`` —
                # symmetric with the matching headline (R7b). Non-
                # trainable blockers + missing variant checkpoint cause
                # variant_model_* to equal baseline_model_* by aliasing.
                m_bb = _recall_rr(baseline_candidates, test_gold_baseline)
                m_br = _recall_rr(baseline_candidates, test_gold_corner)
                if variant_ckpt_distinct:
                    m_vb = _recall_rr(variant_candidates, test_gold_baseline)
                    m_vr = _recall_rr(variant_candidates, test_gold_corner)
                else:
                    m_vb = m_bb
                    m_vr = m_br

                # VAL pair-recall: score the headline candidate set (variant
                # if distinct, else baseline) against the val gold. No extra
                # blocking pass — the candidates are already materialized.
                head_candidates = (
                    variant_candidates if variant_ckpt_distinct else baseline_candidates
                )
                m_val = (
                    _recall_rr(head_candidates, val_gold_corner)
                    if val_gold_corner is not None and not val_gold_corner.empty
                    else m_vr
                )

                # Headline pair_recall + reduction_ratio (used by
                # composition / winner selection — kept on the variant-
                # on-corner_filled surface as the R7b load-bearing
                # number; at baseline level all 4 collapse).
                pair_metrics_by_blocker[b_spec.name] = {
                    "pair_recall": m_vr["pair_recall"],
                    "pair_recall_val": m_val["pair_recall"],
                    "gold_positives": m_vr["gold_positives"],
                    "covered": m_vr["covered"],
                    "missed": m_vr["missed"],
                    "reduction_ratio": m_vr["reduction_ratio"],
                    "candidate_count": m_vr["candidate_count"],
                    "full_space": m_vr["full_space"],
                    "runtime_s": elapsed,
                    # R7b dual-model dual-test cells.
                    "pair_recall_baseline_model_on_baseline_test": m_bb["pair_recall"],
                    "pair_recall_baseline_model_on_regen_test": m_br["pair_recall"],
                    "pair_recall_variant_model_on_baseline_test": m_vb["pair_recall"],
                    "pair_recall_variant_model_on_regen_test": m_vr["pair_recall"],
                    "reduction_ratio_baseline_model": m_bb["reduction_ratio"],
                    "reduction_ratio_variant_model": m_vr["reduction_ratio"],
                    "candidate_count_baseline_model": m_bb["candidate_count"],
                    "candidate_count_variant_model": m_vr["candidate_count"],
                    "variant_model_distinct": 1.0 if variant_ckpt_distinct else 0.0,
                }
                blocker_runtime[b_spec.name] += elapsed

            winner_name, recall_cleared = _select_best_blocker(
                pair_metrics_by_blocker, self._composition
            )

            for b_name, b_metrics in pair_metrics_by_blocker.items():
                blocker_pair_metrics[b_name][pair_key] = {
                    **b_metrics,
                    "selected": 1.0 if b_name == winner_name else 0.0,
                    "recall_floor_cleared": 1.0 if recall_cleared else 0.0,
                }

            per_partition[pair_key] = {
                "winner_blocker_pair_recall": pair_metrics_by_blocker[winner_name][
                    "pair_recall"
                ],
                "winner_blocker_reduction_ratio": pair_metrics_by_blocker[winner_name][
                    "reduction_ratio"
                ],
                "recall_floor_cleared": 1.0 if recall_cleared else 0.0,
            }

        total_runtime = time.monotonic() - t0_total

        per_blocker: dict[str, MemberResult] = {}
        for spec in self._blocking_specs:
            pair_metrics = blocker_pair_metrics[spec.name]
            member_metrics = _aggregate_blocker_pairs(pair_metrics)
            per_blocker[spec.name] = MemberResult(
                name=spec.name,
                predictions=None,
                metrics=member_metrics,
                runtime_s=blocker_runtime[spec.name],
                notes={
                    "role": "blocker",
                    "blocking_type": spec.blocking_type,
                    "per_pair": pair_metrics,
                },
            )

        # Aggregated: macro across blockers of macro-across-pairs metrics.
        recalls = [m.metrics.get("pair_recall", 0.0) for m in per_blocker.values()]
        rrs = [m.metrics.get("reduction_ratio", 0.0) for m in per_blocker.values()]
        n = len(per_blocker) or 1
        survivors = [
            (m.name, m.metrics["reduction_ratio"])
            for m in per_blocker.values()
            if m.metrics.get("pair_recall", 0.0) >= self._composition.recall_floor
        ]
        if survivors:
            best_name = max(survivors, key=lambda kv: kv[1])[0]
        else:
            best_name = max(
                per_blocker.values(), key=lambda m: m.metrics.get("pair_recall", 0.0)
            ).name
        best_member = per_blocker[best_name]

        def _macro_blocker(key: str) -> float:
            values = [
                float(m.metrics.get(key, 0.0))
                for m in per_blocker.values()
                if not _is_nan(m.metrics.get(key))
            ]
            return sum(values) / len(values) if values else float("nan")

        aggregated = {
            "macro_pair_recall": sum(recalls) / n if recalls else 0.0,
            "min_pair_recall": min(recalls) if recalls else 0.0,
            "max_pair_recall": max(recalls) if recalls else 0.0,
            "macro_reduction_ratio": sum(rrs) / n if rrs else 0.0,
            # R7b dual-model dual-test aggregates. The load-bearing
            # monotonicity key is
            # ``macro_pair_recall_variant_model_on_regen_test`` —
            # symmetric with EM matching's variant_model_on_regen_test
            # (R7b: variant-trained model isolates intrinsic difficulty
            # from distribution-shift). At baseline level or when no
            # variant checkpoint exists, all 4 cells collapse to the
            # baseline-model value.
            "macro_pair_recall_baseline_model_on_baseline_test": _macro_blocker(
                "pair_recall_baseline_model_on_baseline_test"
            ),
            "macro_pair_recall_baseline_model_on_regen_test": _macro_blocker(
                "pair_recall_baseline_model_on_regen_test"
            ),
            "macro_pair_recall_variant_model_on_baseline_test": _macro_blocker(
                "pair_recall_variant_model_on_baseline_test"
            ),
            "macro_pair_recall_variant_model_on_regen_test": _macro_blocker(
                "pair_recall_variant_model_on_regen_test"
            ),
            "best_member_name": best_name,
            "best_member_pair_recall": best_member.metrics.get("pair_recall", 0.0),
            "best_member_reduction_ratio": best_member.metrics.get(
                "reduction_ratio", 0.0
            ),
            "recall_floor": self._composition.recall_floor,
        }

        return CommitteeResult(
            stage="em_blocking",  # type: ignore[arg-type]
            domain=bundle.domain,
            level=bundle.level,
            per_member=per_blocker,
            per_blocker=per_blocker,
            aggregated=aggregated,
            per_attribute={},
            per_partition={k: _coerce_partition(v) for k, v in per_partition.items()},
            runtime_s=total_runtime,
            roster=self.roster_names,
        )


class EMMatchingCommitteeRunner(CommitteeRunner):
    """Matching-only committee runner under the perfect-prior-step design.

    Skips blocking entirely. For each source pair, feeds the labelled
    ``_val.csv`` and ``_test.csv`` pairs (positives + negatives) to each
    matcher as candidates and scores predictions under the closed-set
    semantic (:func:`score_em_correspondences_closed_set`) — predictions
    outside the labelled universe are out of scope, not FPs, so
    precision is meaningful even though the matcher only sees a few
    hundred pairs.

    For ML matchers, ``pair_train_path`` is still injected at matcher
    construction, so they fit on ``_train.csv``.

    Primary headline metric: macro ``f1_regen_test`` (closed-set on
    ``em_test_corner_filled.csv``). Secondary reference:
    macro ``f1_baseline_test`` (closed-set on
    ``em_test_baseline_pruned.csv``). The corner-filled val split
    is used internally as a val/test agreement sanity check but not
    surfaced in per-pair metrics (plan_revision.md C10).

    Parameters
    ----------
    matching_roster_path : Path
        Path to ``em_matching_committee.yaml``.
    with_llm : bool
        Enable LLM-based matching members.
    retain_predictions_for : set of str, optional
        Member names whose per-pair predictions should be retained on
        :class:`MemberResult`.
    """

    stage: Literal["em_matching"] = "em_matching"  # type: ignore[assignment]

    def __init__(
        self,
        matching_roster_path: Path,
        *,
        with_llm: bool = False,
        retain_predictions_for: set[str] | None = None,
    ) -> None:
        matching_raw = _load_roster_yaml(matching_roster_path)
        self._matching_specs = _parse_matching_roster(
            matching_raw["members"], with_llm=with_llm
        )
        if not self._matching_specs:
            raise ValueError(
                f"No enabled matchers in {matching_roster_path} "
                f"(with_llm={with_llm}); refusing to run."
            )
        self._seed = int(matching_raw.get("seed", 42))
        self._column_mapping: dict[str, dict[str, str]] = _resolve_column_mapping(
            None,
            matching_raw.get("column_mapping"),
        )
        preprocess_name = matching_raw.get("preprocess_text")
        self._preprocess_fn = (
            _PREPROCESS_REGISTRY.get(preprocess_name) if preprocess_name else None
        )
        self._retain_predictions_for: set[str] = (
            set(retain_predictions_for) if retain_predictions_for else set()
        )

        super().__init__(
            roster=list(self._matching_specs),
            config={
                "seed": self._seed,
                "with_llm": with_llm,
                "matching_roster_path": str(matching_roster_path),
            },
        )

    @property
    def roster_names(self) -> list[str]:
        return [s.name for s in self._matching_specs]

    def _load_labelled_split(
        self,
        bundle: VariantBundle,
        pair: tuple[str, str],
        split: str,
        version: str = "corner_filled",
    ) -> pd.DataFrame | None:
        """Backwards-compatible wrapper around :func:`_load_labelled_split_from_bundle`.

        Kept as a method for test continuity; new callers should use
        the module-level function so EMBlockingCommitteeRunner shares
        the same loader.
        """
        return _load_labelled_split_from_bundle(bundle, pair, split, version=version)

    def run(self, bundle: VariantBundle) -> CommitteeResult:
        """Train + score every matcher on every source pair."""
        if not bundle.em_gold:
            raise ValueError(
                f"No EM gold for {bundle.domain}/{bundle.level}. "
                "Need test gold CSVs to compute matcher F1."
            )

        _set_deterministic(self._seed)
        effective_column_mapping = bundle.resolve_column_mapping(self._column_mapping)

        matcher_pair_metrics: dict[str, dict[str, dict[str, float]]] = {
            spec.name: {} for spec in self._matching_specs
        }
        matcher_pair_predictions: dict[str, dict[str, pd.DataFrame]] = {
            spec.name: {} for spec in self._matching_specs
        }
        matcher_runtime: dict[str, float] = {
            spec.name: 0.0 for spec in self._matching_specs
        }
        per_partition: dict[str, dict[str, Any]] = {}

        t0_total = time.monotonic()

        for pair in bundle.em_gold:
            src1, src2 = pair
            pair_key = f"{src1}_{src2}"

            df_left = bundle.sources[src1].copy()
            df_right = bundle.sources[src2].copy()
            if effective_column_mapping:
                left_map = effective_column_mapping.get(src1, {})
                if left_map:
                    df_left = apply_column_mapping(df_left, left_map)
                right_map = effective_column_mapping.get(src2, {})
                if right_map:
                    df_right = apply_column_mapping(df_right, right_map)

            # plan_revision.md C10/C11: load both test versions per
            # pair. ``corner_filled`` (Set 2) is the headline surface;
            # ``baseline_pruned`` (Set 1) is a per-level reference.
            # val uses the corner_filled version only — it's an
            # internal val/test agreement sanity check for learned
            # matchers, not surfaced in the public per-pair dict.
            val_gold_corner = self._load_labelled_split(
                bundle, pair, "val", version="corner_filled"
            )
            test_gold_corner = self._load_labelled_split(
                bundle, pair, "test", version="corner_filled"
            )
            test_gold_baseline = self._load_labelled_split(
                bundle, pair, "test", version="baseline_pruned"
            )
            # R10-F: at variant levels the regenerated test splits MUST be
            # present; their absence collapses f1_*_on_regen_test and
            # f1_*_on_baseline_test onto the same surface. Warn so a missing
            # corner_filled/baseline_pruned file is not silently masked.
            if bundle.level != "baseline" and (
                test_gold_corner is None or test_gold_baseline is None
            ):
                logger.warning(
                    "EM matching pair %s-%s at level %s: regenerated test gold "
                    "missing (corner_filled=%s, baseline_pruned=%s); regen-test "
                    "surfaces will be NaN/degenerate. Check package_variant "
                    "copied the *_{baseline_pruned,corner_filled}.csv files "
                    "(R10-F).",
                    pair[0],
                    pair[1],
                    bundle.level,
                    test_gold_corner is not None,
                    test_gold_baseline is not None,
                )
            # R7b dual-model: resolve baseline + variant artifacts per pair.
            baseline_train_path = _resolve_pair_train_path(bundle, pair)
            variant_train_path, variant_train_distinct = _resolve_variant_train_path(
                bundle, pair
            )

            for m_spec in self._matching_specs:
                t0 = time.monotonic()
                regen_val_metrics = _empty_closed_set_metrics()
                # 4 metric blocks per (pair, member) — R7b dual-model dual-test.
                # Keys: m_<train>_<test> where train ∈ {b=baseline, v=variant},
                # test ∈ {b=baseline_pruned, r=corner_filled aka regen_test}.
                m_bb = _empty_closed_set_metrics()
                m_br = _empty_closed_set_metrics()
                m_vb = _empty_closed_set_metrics()
                m_vr = _empty_closed_set_metrics()
                # Surfaced VAL headline (headline model on the val corner_filled
                # gold). None until computed; None -> val == test (zero-shot).
                m_val: dict[str, float] | None = None
                val_preds = pd.DataFrame(columns=["id1", "id2", "score"])
                baseline_test_preds = pd.DataFrame(columns=["id1", "id2", "score"])
                variant_test_preds = pd.DataFrame(columns=["id1", "id2", "score"])

                # Zero-shot matchers (matching_type='llm') have no
                # hyperparameters tunable on val and no train-data
                # dependency, so val scoring yields the same F1 as test
                # under the closed-set semantic — pure duplicate work.
                # Score them on test only. Learned matchers (Ditto with
                # early stopping, Magellan classifier sweep) DO benefit
                # from val as an overfit-vs-test sanity surface, so they
                # score both. Per the user directive 2026-05-13.
                score_val = (
                    m_spec.matching_type == "learned"
                    and val_gold_corner is not None
                    and not val_gold_corner.empty
                )

                # R7b: decide whether to build a separate variant_model
                # instance, or alias to baseline_model. A variant model
                # is "distinct" iff either (a) a variant checkpoint
                # exists at <cache>/variant_<level>/best, or (b) the
                # variant train CSV (<pair>_train_corner_filled.csv) is
                # present and different from the baseline train. At
                # baseline level neither holds, so variant_distinct =
                # False and the runner skips the duplicate inference
                # pass.
                baseline_ckpt = m_spec.matcher_spec.get("params", {}).get(
                    "checkpoint_path"
                )
                variant_ckpt, variant_ckpt_distinct = _resolve_variant_checkpoint_path(
                    baseline_ckpt, bundle.level
                )
                variant_distinct = variant_ckpt_distinct or variant_train_distinct

                try:
                    matcher_class = str(m_spec.matcher_spec.get("class", ""))
                    if matcher_class in _PER_PAIR_TRAIN_INJECTION and (
                        baseline_train_path is None or not baseline_train_path.exists()
                    ):
                        logger.warning(
                            "Skipping %s on pair %s-%s without per-pair train data",
                            m_spec.name,
                            src1,
                            src2,
                        )
                    else:
                        comparators = _build_comparators(
                            m_spec.comparator_specs, self._preprocess_fn
                        )
                        baseline_matcher = _build_matcher(
                            m_spec.matcher_spec,
                            pair_train_path=baseline_train_path,
                        )
                        if variant_distinct:
                            # Variant matcher gets the regen-train (if
                            # distinct) and/or the variant checkpoint
                            # (if cached). For Magellan: a separate fit
                            # on <pair>_train_corner_filled.csv. For
                            # Ditto/sc_block: loads
                            # variant_<level>/best. The Magellan-only
                            # train-data path requires the file to
                            # exist; skip if absent for that matcher.
                            if matcher_class in _PER_PAIR_TRAIN_INJECTION and (
                                variant_train_path is None
                                or not variant_train_path.exists()
                            ):
                                variant_matcher = baseline_matcher
                                variant_distinct = False
                            else:
                                variant_matcher = _build_matcher(
                                    m_spec.matcher_spec,
                                    pair_train_path=variant_train_path,
                                    checkpoint_override=variant_ckpt,
                                )
                        else:
                            # Aliasing: variant_model identical to
                            # baseline_model. m_v* will equal m_b* by
                            # construction; we still emit all 4 keys
                            # so consumers don't branch on level.
                            variant_matcher = baseline_matcher
                        match_kwargs: dict[str, Any] = {
                            "id_column": "id",
                            "threshold": m_spec.threshold,
                        }
                        if comparators:
                            match_kwargs["comparators"] = comparators
                        if m_spec.weights:
                            match_kwargs["weights"] = m_spec.weights

                        if score_val:
                            # Val scoring is baseline-model-only (it's
                            # an overfit-vs-test sanity log for learned
                            # matchers; not surfaced publicly).
                            val_preds = baseline_matcher.match(
                                df_left,
                                df_right,
                                val_gold_corner[["id1", "id2"]].copy(),
                                **match_kwargs,
                            )
                            regen_val_metrics = score_em_correspondences_closed_set(
                                val_preds, val_gold_corner
                            )

                        # Test inference: run each model once on the
                        # corner_filled candidate set (Set 2 ⊇ Set 1),
                        # then score against both gold surfaces. The
                        # closed-set scorer scopes predictions to each
                        # gold universe internally — single inference
                        # pass per model.
                        if test_gold_corner is not None and not test_gold_corner.empty:
                            cand_test = test_gold_corner[["id1", "id2"]].copy()
                            baseline_test_preds = baseline_matcher.match(
                                df_left, df_right, cand_test, **match_kwargs
                            )
                            m_br = score_em_correspondences_closed_set(
                                baseline_test_preds, test_gold_corner
                            )
                            if (
                                test_gold_baseline is not None
                                and not test_gold_baseline.empty
                            ):
                                m_bb = score_em_correspondences_closed_set(
                                    baseline_test_preds, test_gold_baseline
                                )
                            if variant_distinct:
                                variant_test_preds = variant_matcher.match(
                                    df_left, df_right, cand_test, **match_kwargs
                                )
                                m_vr = score_em_correspondences_closed_set(
                                    variant_test_preds, test_gold_corner
                                )
                                if (
                                    test_gold_baseline is not None
                                    and not test_gold_baseline.empty
                                ):
                                    m_vb = score_em_correspondences_closed_set(
                                        variant_test_preds, test_gold_baseline
                                    )
                            else:
                                # Variant model aliases baseline — copy.
                                variant_test_preds = baseline_test_preds
                                m_vr = m_br
                                m_vb = m_bb
                        elif (
                            test_gold_baseline is not None
                            and not test_gold_baseline.empty
                        ):
                            # Corner-filled missing but baseline-pruned
                            # present — score against baseline-pruned
                            # directly. regen surfaces (m_br, m_vr)
                            # stay empty in this path.
                            cand_test = test_gold_baseline[["id1", "id2"]].copy()
                            baseline_test_preds = baseline_matcher.match(
                                df_left, df_right, cand_test, **match_kwargs
                            )
                            m_bb = score_em_correspondences_closed_set(
                                baseline_test_preds, test_gold_baseline
                            )
                            if variant_distinct:
                                variant_test_preds = variant_matcher.match(
                                    df_left, df_right, cand_test, **match_kwargs
                                )
                                m_vb = score_em_correspondences_closed_set(
                                    variant_test_preds, test_gold_baseline
                                )
                            else:
                                variant_test_preds = baseline_test_preds
                                m_vb = m_bb

                        # EM VALIDATION surface (user spec: score val AND test).
                        # Run the headline model (variant if distinct, else
                        # baseline) on the val corner_filled gold. Learned
                        # matchers benefit from a real val pass; LLM zero-shot
                        # matchers have no val/test distinction (no tunable
                        # hyperparameters) so their val == test and is filled
                        # from `primary` below (m_val stays None) — avoids
                        # doubling LLM cost. games (no shipped val) is covered
                        # because _load_labelled_split derives a stratified
                        # hold-out from train.
                        if (
                            m_spec.matching_type == "learned"
                            and val_gold_corner is not None
                            and not val_gold_corner.empty
                        ):
                            val_model = (
                                variant_matcher if variant_distinct else baseline_matcher
                            )
                            val_head_preds = val_model.match(
                                df_left,
                                df_right,
                                val_gold_corner[["id1", "id2"]].copy(),
                                **match_kwargs,
                            )
                            m_val = score_em_correspondences_closed_set(
                                val_head_preds, val_gold_corner
                            )
                except Exception:
                    logger.exception(
                        "Matcher %s failed on pair %s-%s",
                        m_spec.name,
                        src1,
                        src2,
                    )

                # Internal val/test agreement sanity check (C10):
                # log on divergence, do not surface in metrics dict.
                # Compares baseline-model val vs baseline-model
                # regen_test (m_br) — both same training distribution.
                if (
                    not _is_nan(regen_val_metrics["f1"])
                    and not _is_nan(m_br["f1"])
                    and abs(regen_val_metrics["f1"] - m_br["f1"]) > 0.15
                ):
                    logger.debug(
                        "EM matching regen val/test divergence > 0.15 on "
                        "pair %s member %s: val_f1=%.3f test_f1=%.3f",
                        pair,
                        m_spec.name,
                        regen_val_metrics["f1"],
                        m_br["f1"],
                    )

                elapsed = time.monotonic() - t0
                # Headline F1 (R7b): the load-bearing surface for
                # monotonicity is f1_variant_model_on_regen_test (m_vr).
                # Fallback chain when artifacts missing:
                #   m_vr (variant + regen) → m_vb (variant + baseline) →
                #   m_br (baseline + regen) → m_bb (baseline + baseline).
                # At baseline-level pipelines, variant aliases baseline
                # so this collapses to m_bb (single F1).
                if (
                    not _is_nan(m_vr.get("f1", float("nan")))
                    and m_vr.get("pred_scoped", 0) > 0
                ):
                    primary = m_vr
                elif (
                    not _is_nan(m_vb.get("f1", float("nan")))
                    and m_vb.get("pred_scoped", 0) > 0
                ):
                    primary = m_vb
                elif (
                    not _is_nan(m_br.get("f1", float("nan")))
                    and m_br.get("pred_scoped", 0) > 0
                ):
                    primary = m_br
                elif (
                    not _is_nan(m_bb.get("f1", float("nan")))
                    and m_bb.get("pred_scoped", 0) > 0
                ):
                    primary = m_bb
                else:
                    primary = _empty_closed_set_metrics()

                # VAL headline: real val metrics for learned matchers; for
                # zero-shot LLM matchers (m_val is None) val == test == primary.
                val_head = m_val if m_val is not None else primary
                matcher_pair_metrics[m_spec.name][pair_key] = {
                    "f1": primary["f1"],
                    "precision": primary["precision"],
                    "recall": primary["recall"],
                    "f1_val": val_head["f1"],
                    "precision_val": val_head["precision"],
                    "recall_val": val_head["recall"],
                    "tp": primary.get("tp", 0.0),
                    "fp": primary.get("fp", 0.0),
                    "fn": primary.get("fn", 0.0),
                    # R7b dual-model dual-test surfaces (4 per pair).
                    "f1_baseline_model_on_baseline_test": m_bb["f1"],
                    "precision_baseline_model_on_baseline_test": m_bb["precision"],
                    "recall_baseline_model_on_baseline_test": m_bb["recall"],
                    "pred_scoped_baseline_model_on_baseline_test": m_bb.get(
                        "pred_scoped", 0.0
                    ),
                    "f1_baseline_model_on_regen_test": m_br["f1"],
                    "precision_baseline_model_on_regen_test": m_br["precision"],
                    "recall_baseline_model_on_regen_test": m_br["recall"],
                    "pred_scoped_baseline_model_on_regen_test": m_br.get(
                        "pred_scoped", 0.0
                    ),
                    "f1_variant_model_on_baseline_test": m_vb["f1"],
                    "precision_variant_model_on_baseline_test": m_vb["precision"],
                    "recall_variant_model_on_baseline_test": m_vb["recall"],
                    "pred_scoped_variant_model_on_baseline_test": m_vb.get(
                        "pred_scoped", 0.0
                    ),
                    "f1_variant_model_on_regen_test": m_vr["f1"],
                    "precision_variant_model_on_regen_test": m_vr["precision"],
                    "recall_variant_model_on_regen_test": m_vr["recall"],
                    "pred_scoped_variant_model_on_regen_test": m_vr.get(
                        "pred_scoped", 0.0
                    ),
                    "variant_model_distinct": float(1.0 if variant_distinct else 0.0),
                    # Legacy aliases (pre-R7b): kept for backwards
                    # compatibility with downstream consumers + tests.
                    # Both refer to the baseline-trained model on each
                    # gold surface — that was always their meaning;
                    # R7b just makes the model-axis explicit.
                    "f1_baseline_test": m_bb["f1"],
                    "precision_baseline_test": m_bb["precision"],
                    "recall_baseline_test": m_bb["recall"],
                    "pred_scoped_baseline_test": m_bb.get("pred_scoped", 0.0),
                    "f1_regen_test": m_br["f1"],
                    "precision_regen_test": m_br["precision"],
                    "recall_regen_test": m_br["recall"],
                    "pred_scoped_regen_test": m_br.get("pred_scoped", 0.0),
                }
                matcher_runtime[m_spec.name] += elapsed
                if m_spec.name in self._retain_predictions_for:
                    # Retain variant-model test predictions when
                    # available (the new headline surface); else
                    # baseline-model test preds; else val preds.
                    matcher_pair_predictions[m_spec.name][pair_key] = (
                        variant_test_preds
                        if variant_distinct and not variant_test_preds.empty
                        else (
                            baseline_test_preds
                            if not baseline_test_preds.empty
                            else val_preds
                        )
                    )

            f1s = [
                matcher_pair_metrics[m.name][pair_key].get("f1", 0.0)
                for m in self._matching_specs
                if not _is_nan(matcher_pair_metrics[m.name][pair_key].get("f1"))
            ]
            n = len(f1s)
            per_partition[pair_key] = {
                "macro_f1": sum(f1s) / n if n else float("nan"),
                "min_f1": min(f1s) if n else float("nan"),
                "max_f1": max(f1s) if n else float("nan"),
                "n_members": float(n),
            }

        total_runtime = time.monotonic() - t0_total

        per_member: dict[str, MemberResult] = {}
        avg_keys = [
            "f1",
            "precision",
            "recall",
            "f1_val",
            "precision_val",
            "recall_val",
            # R7b dual-model dual-test (4 cells).
            "f1_baseline_model_on_baseline_test",
            "precision_baseline_model_on_baseline_test",
            "recall_baseline_model_on_baseline_test",
            "pred_scoped_baseline_model_on_baseline_test",
            "f1_baseline_model_on_regen_test",
            "precision_baseline_model_on_regen_test",
            "recall_baseline_model_on_regen_test",
            "pred_scoped_baseline_model_on_regen_test",
            "f1_variant_model_on_baseline_test",
            "precision_variant_model_on_baseline_test",
            "recall_variant_model_on_baseline_test",
            "pred_scoped_variant_model_on_baseline_test",
            "f1_variant_model_on_regen_test",
            "precision_variant_model_on_regen_test",
            "recall_variant_model_on_regen_test",
            "pred_scoped_variant_model_on_regen_test",
            # Pre-R7b legacy aliases (kept for backward compat).
            "f1_baseline_test",
            "precision_baseline_test",
            "recall_baseline_test",
            "pred_scoped_baseline_test",
            "f1_regen_test",
            "precision_regen_test",
            "recall_regen_test",
            "pred_scoped_regen_test",
        ]
        for spec in self._matching_specs:
            pair_metrics = matcher_pair_metrics[spec.name]
            member_metrics: dict[str, float] = {}
            for key in avg_keys:
                values = [
                    float(pm.get(key, 0.0))
                    for pm in pair_metrics.values()
                    if not _is_nan(pm.get(key))
                ]
                member_metrics[key] = (
                    float("nan") if not values else sum(values) / len(values)
                )
            retain = spec.name in self._retain_predictions_for
            per_member[spec.name] = MemberResult(
                name=spec.name,
                predictions=(matcher_pair_predictions[spec.name] if retain else None),
                metrics=member_metrics,
                runtime_s=matcher_runtime[spec.name],
                notes={
                    "role": "matcher",
                    "matching_type": spec.matching_type,
                    "missing_value_tolerant": spec.missing_value_tolerant,
                    "per_pair": pair_metrics,
                },
            )

        def _macro(key: str) -> float:
            values = [
                float(m.metrics.get(key, 0.0))
                for m in per_member.values()
                if not _is_nan(m.metrics.get(key))
            ]
            return sum(values) / len(values) if values else float("nan")

        # NaN-skip the headline f1 macros + min/max + best-member selection so
        # a single skipped member (e.g. magellan with no per-pair train CSV)
        # never poisons the committee aggregate or hides the real best member.
        f1s = [
            float(m.metrics.get("f1", 0.0))
            for m in per_member.values()
            if not _is_nan(m.metrics.get("f1"))
        ]

        best_name = max(
            per_member.values(),
            key=lambda m: (
                m.metrics.get("f1", 0.0)
                if not _is_nan(m.metrics.get("f1"))
                else float("-inf")
            ),
        ).name
        aggregated = {
            "macro_f1": _macro("f1"),
            "min_f1": min(f1s) if f1s else float("nan"),
            "max_f1": max(f1s) if f1s else float("nan"),
            "macro_precision": _macro("precision"),
            "macro_recall": _macro("recall"),
            # R7b dual-model dual-test aggregates. The load-bearing
            # monotonicity key is ``macro_f1_variant_model_on_regen_test``
            # (per plan_revision.md R7b — variant-trained model isolates
            # intrinsic difficulty from transfer-learning gap).
            "macro_f1_baseline_model_on_baseline_test": _macro(
                "f1_baseline_model_on_baseline_test"
            ),
            "macro_f1_baseline_model_on_regen_test": _macro(
                "f1_baseline_model_on_regen_test"
            ),
            "macro_f1_variant_model_on_baseline_test": _macro(
                "f1_variant_model_on_baseline_test"
            ),
            "macro_f1_variant_model_on_regen_test": _macro(
                "f1_variant_model_on_regen_test"
            ),
            # Legacy aliases (pre-R7b).
            "macro_f1_baseline_test": _macro("f1_baseline_test"),
            "macro_f1_regen_test": _macro("f1_regen_test"),
            "best_member_name": best_name,
            "best_member_f1": per_member[best_name].metrics.get("f1", 0.0),
        }

        return CommitteeResult(
            stage="em_matching",  # type: ignore[arg-type]
            domain=bundle.domain,
            level=bundle.level,
            per_member=per_member,
            per_blocker={},
            aggregated=aggregated,
            per_attribute={},
            per_partition={k: _coerce_partition(v) for k, v in per_partition.items()},
            runtime_s=total_runtime,
            roster=self.roster_names,
        )


def _empty_closed_set_metrics() -> dict[str, float]:
    """Zero / NaN sentinel for the closed-set metric dict."""
    return {
        "f1": float("nan"),
        "precision": float("nan"),
        "recall": float("nan"),
        "tp": 0.0,
        "fp": 0.0,
        "fn": 0.0,
        "pred_scoped": 0.0,
    }
