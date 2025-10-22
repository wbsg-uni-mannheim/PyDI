"""
Interactive entity matching workflows for labeling support.

This module provides orchestration utilities for combining PyDI's blocking
and matching components into an interactive sampling loop.  It is intended to
power lightweight UIs (for example Streamlit dashboards) that need to surface
balanced batches of candidate pairs for human review while keeping recall
high through blocker unions and epsilon exploration.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..base import BaseComparator
from ..blocking.base import BaseBlocker
from ..blocking.embedding import EmbeddingBlocker
from ..blocking.sorted_neighbourhood import SortedNeighbourhoodBlocker
from ..blocking.standard import StandardBlocker
from ..blocking.token_blocking import TokenBlocker
from ..comparators import DateComparator, NumericComparator, StringComparator
from ..rule_based import RuleBasedMatcher


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class MatchingSessionResult:
    """Result bundle returned by :func:`run_interactive_matching`.

    Attributes
    ----------
    candidate_pairs:
        Unique candidate pairs with blocker provenance and metadata.
    scored_pairs:
        Candidate pairs enriched with matcher scores, heuristics, and bins.
    feature_matrix:
        Pivot table of comparator similarities per pair.
    sampled_batch:
        DataFrame containing the curated batch according to the requested mix.
    metrics:
        Dictionary with aggregate diagnostics (bin counts, reduction ratio, etc.).
    config:
        Echo of the configuration that produced this result (useful for UI state).
    """

    candidate_pairs: pd.DataFrame
    scored_pairs: pd.DataFrame
    feature_matrix: pd.DataFrame
    sampled_batch: pd.DataFrame
    metrics: Dict[str, Any]
    config: Dict[str, Any]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _normalize_weights(weights: Sequence[float]) -> List[float]:
    """Normalize comparator weights while avoiding division by zero."""
    weights = [float(max(0.0, w)) for w in weights]
    total = sum(weights)
    if total == 0:
        # Default to uniform weights if everything is zero
        return [1.0 / len(weights)] * len(weights) if weights else []
    return [w / total for w in weights]


def _safe_row_mean(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    """Compute a mean across columns, ignoring missing columns and NaNs."""
    cols = [col for col in columns if col in frame.columns]
    if not cols:
        return pd.Series([np.nan] * len(frame), index=frame.index)
    return frame[cols].mean(axis=1, skipna=True)


def _preview_value_to_text(value: Any, separator: str = " | ") -> Optional[str]:
    """Normalize preview values for readable string rendering."""
    if value is None:
        return None
    if isinstance(value, pd.Series):
        return _preview_value_to_text(value.dropna().tolist(), separator)
    if isinstance(value, np.ndarray):
        return _preview_value_to_text(value.tolist(), separator)
    if isinstance(value, (list, tuple, set)):
        parts = [str(item).strip() for item in value if item is not None and str(item).strip()]
        return separator.join(parts) if parts else None
    if isinstance(value, dict):
        parts = [f"{k}: {v}" for k, v in value.items() if v is not None and str(v).strip()]
        return separator.join(parts) if parts else None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        # Non-numeric iterables raise TypeError; treat them via str below
        pass
    text = str(value).strip()
    return text or None


def _instantiate_blocker(
    blocker_cfg: Dict[str, Any],
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    id_column: str,
) -> Tuple[str, BaseBlocker]:
    """Instantiate a blocker from configuration."""
    blocker_type = blocker_cfg.get("type")
    name = blocker_cfg.get("name") or blocker_type
    params = blocker_cfg.get("params", {})

    if blocker_type == "standard":
        on = params.get("on") or []
        if not on:
            raise ValueError("Standard blocker configuration requires 'on' columns")
        blocker = StandardBlocker(
            df_left,
            df_right,
            on=on,
            id_column=id_column,
            batch_size=params.get("batch_size", 100_000),
            output_dir=params.get("output_dir", "output"),
            preprocess=params.get("preprocess"),
        )
        return name or "StandardBlocker", blocker

    if blocker_type == "sorted_neighbourhood":
        key = params.get("key")
        window = params.get("window", 3)
        if not key:
            raise ValueError("Sorted neighbourhood blocker requires a 'key'")
        blocker = SortedNeighbourhoodBlocker(
            df_left,
            df_right,
            key=key,
            id_column=id_column,
            window=window,
            batch_size=params.get("batch_size", 100_000),
            output_dir=params.get("output_dir", "output"),
            preprocess=params.get("preprocess"),
        )
        return name or "SortedNeighbourhoodBlocker", blocker

    if blocker_type == "token":
        column = params.get("column")
        if not column:
            raise ValueError("Token blocker requires a 'column'")
        blocker = TokenBlocker(
            df_left,
            df_right,
            column=column,
            id_column=id_column,
            tokenizer=params.get("tokenizer"),
            batch_size=params.get("batch_size", 100_000),
            min_token_len=params.get("min_token_len", 2),
            output_dir=params.get("output_dir", "output"),
            ngram_size=params.get("ngram_size"),
            ngram_type=params.get("ngram_type"),
            preprocess=params.get("preprocess"),
        )
        return name or "TokenBlocker", blocker

    if blocker_type == "embedding":
        text_cols = params.get("text_cols") or []
        if not text_cols:
            raise ValueError("Embedding blocker requires 'text_cols'")
        blocker = EmbeddingBlocker(
            df_left,
            df_right,
            text_cols=text_cols,
            id_column=id_column,
            model=params.get(
                "model", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            index_backend=params.get("index_backend", "sklearn"),
            metric=params.get("metric", "cosine"),
            top_k=params.get("top_k", 50),
            threshold=params.get("threshold", 0.3),
            normalize=params.get("normalize", True),
            batch_size=params.get("batch_size", 25_000),
            query_batch_size=params.get("query_batch_size", 2048),
            embedder=params.get("embedder"),
            left_embeddings=params.get("left_embeddings"),
            right_embeddings=params.get("right_embeddings"),
            device=params.get("device"),
            output_dir=params.get("output_dir", "output"),
            preprocess=params.get("preprocess"),
        )
        return name or "EmbeddingBlocker", blocker

    raise ValueError(f"Unsupported blocker type: {blocker_type}")


def _union_blocker_outputs(blockers: Iterable[Tuple[str, BaseBlocker]]) -> pd.DataFrame:
    """Union candidate pairs emitted by the supplied blockers."""
    batches: List[pd.DataFrame] = []
    for blocker_name, blocker in blockers:
        for batch in blocker:
            if batch.empty:
                continue
            df_batch = batch.copy()
            if "id1" not in df_batch.columns or "id2" not in df_batch.columns:
                raise ValueError(f"Blocker '{blocker_name}' emitted invalid batch")
            df_batch = df_batch[["id1", "id2"] + [c for c in df_batch.columns if c not in {"id1", "id2"}]]
            df_batch["source_blocker"] = blocker_name
            batches.append(df_batch)

    if not batches:
        return pd.DataFrame(columns=["id1", "id2", "source_blockers", "source_count", "block_keys", "raw_hits", "epsilon_flag"])

    combined = pd.concat(batches, ignore_index=True)

    # Normalise provenance information
    def _aggregate_block_keys(series: pd.Series) -> Tuple[str, ...]:
        valid = [str(val) for val in series if isinstance(val, str) and val]
        return tuple(sorted(set(valid)))

    aggregated = (
        combined.groupby(["id1", "id2"], as_index=False)
        .agg(
            source_blockers=("source_blocker", lambda s: tuple(sorted(set(s)))),
            block_keys=("block_key", _aggregate_block_keys) if "block_key" in combined.columns else ("source_blocker", lambda _: tuple()),
            raw_hits=("source_blocker", "count"),
        )
    )
    aggregated["source_count"] = aggregated["source_blockers"].apply(len)
    aggregated["epsilon_flag"] = False
    return aggregated


def _inject_epsilon_random(
    candidates: pd.DataFrame,
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    id_column: str,
    epsilon: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Add epsilon-random exploration pairs to the candidate set."""
    if epsilon <= 0:
        return candidates

    total_possible = len(df_left) * len(df_right)
    if total_possible == 0:
        return candidates

    target_pairs = max(1, int(round(total_possible * epsilon)))
    left_ids = df_left[id_column].tolist()
    right_ids = df_right[id_column].tolist()

    existing_pairs = set(zip(candidates["id1"], candidates["id2"])) if not candidates.empty else set()
    new_rows: List[Dict[str, Any]] = []

    while len(new_rows) < target_pairs and left_ids and right_ids:
        id1 = left_ids[rng.integers(len(left_ids))]
        id2 = right_ids[rng.integers(len(right_ids))]
        pair = (id1, id2)
        if pair in existing_pairs:
            # Append epsilon flag to the existing row
            mask = (candidates["id1"] == id1) & (candidates["id2"] == id2)
            if mask.any():
                idx = candidates.index[mask][0]
                blockers = set(candidates.at[idx, "source_blockers"])
                blockers.add("EpsilonRandom")
                candidates.at[idx, "source_blockers"] = tuple(sorted(blockers))
                candidates.at[idx, "source_count"] = len(blockers)
                candidates.at[idx, "epsilon_flag"] = True
            continue
        existing_pairs.add(pair)
        new_rows.append(
            {
                "id1": id1,
                "id2": id2,
                "source_blockers": ("EpsilonRandom",),
                "source_count": 1,
                "block_keys": tuple(),
                "raw_hits": 1,
                "epsilon_flag": True,
            }
        )

    if not new_rows:
        return candidates

    epsilon_df = pd.DataFrame(new_rows)
    if candidates.empty:
        return epsilon_df

    combined = pd.concat([candidates, epsilon_df], ignore_index=True)
    return combined


def _combine_text_values(row: pd.Series, text_cols: Sequence[str]) -> str:
    """Concatenate text values from multiple columns into a single string."""
    parts: List[str] = []
    for col in text_cols:
        value = row.get(col, "")
        if pd.isna(value) or value is None:
            continue
        parts.append(str(value))
    return " ".join(parts)


def _compute_embedding_similarity(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    id_column: str,
    pairs: pd.DataFrame,
    text_cols: Sequence[str],
    model_name: str,
    device: Optional[str],
) -> np.ndarray:
    """Compute cosine similarities for the supplied pairs using sentence embeddings."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "sentence-transformers is required for embedding similarity. "
            "Install it using 'pip install sentence-transformers'."
        ) from exc

    if pairs.empty:
        return np.asarray([], dtype=float)

    left_lookup = df_left.set_index(id_column, drop=False)
    right_lookup = df_right.set_index(id_column, drop=False)

    left_ids = pairs["id1"].unique().tolist()
    right_ids = pairs["id2"].unique().tolist()

    left_subset = left_lookup.loc[left_ids]
    right_subset = right_lookup.loc[right_ids]

    if isinstance(left_subset, pd.Series):
        left_subset = left_subset.to_frame().T
    if isinstance(right_subset, pd.Series):
        right_subset = right_subset.to_frame().T

    left_texts = left_subset.apply(lambda row: _combine_text_values(row, text_cols), axis=1).tolist()
    right_texts = right_subset.apply(lambda row: _combine_text_values(row, text_cols), axis=1).tolist()

    model = SentenceTransformer(model_name, device=device)
    left_embeddings = model.encode(left_texts, normalize_embeddings=True, show_progress_bar=False)
    right_embeddings = model.encode(right_texts, normalize_embeddings=True, show_progress_bar=False)

    left_id_to_pos = {idx: pos for pos, idx in enumerate(left_ids)}
    right_id_to_pos = {idx: pos for pos, idx in enumerate(right_ids)}

    similarities = np.empty(len(pairs), dtype=float)
    for i, (id1, id2) in enumerate(zip(pairs["id1"], pairs["id2"])):
        left_vec = left_embeddings[left_id_to_pos[id1]]
        right_vec = right_embeddings[right_id_to_pos[id2]]
        similarities[i] = float(np.dot(left_vec, right_vec))

    return similarities


def _build_feature_matrix(debug_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot debug comparator output into a wide feature matrix."""
    if debug_df.empty:
        return pd.DataFrame(columns=["id1", "id2"])

    feature_matrix = (
        debug_df.pivot_table(
            index=["id1", "id2"],
            columns="comparator_name",
            values="postprocessed_similarity",
            aggfunc="mean",
        )
        .reset_index()
    )
    # Flatten the MultiIndex columns if present
    feature_matrix.columns = [
        col if isinstance(col, str) else col[1] for col in feature_matrix.columns.to_flat_index()
    ]
    return feature_matrix


def _prepare_preview_lookup(
    df: pd.DataFrame,
    id_column: str,
    preview_columns: Sequence[str],
) -> Dict[Any, str]:
    """Create a mapping from identifier to a compact preview string."""
    cols = [col for col in preview_columns if col in df.columns]
    if not cols:
        return {row[id_column]: str(row[id_column]) for _, row in df[[id_column]].iterrows()}

    lookup: Dict[Any, str] = {}
    for _, row in df[[id_column] + cols].iterrows():
        parts = []
        for col in cols:
            val = row[col]
            text = _preview_value_to_text(val)
            if text is None:
                continue
            parts.append(f"{col}: {text}")
        lookup[row[id_column]] = " | ".join(parts) if parts else str(row[id_column])
    return lookup


def _assign_bins(
    scored_pairs: pd.DataFrame,
    tau_low: float,
    tau_high: float,
) -> pd.Series:
    """Assign high/low/corner bins based on matchness thresholds."""
    bins = pd.Series(["CORNER"] * len(scored_pairs), index=scored_pairs.index, dtype=object)
    bins.loc[scored_pairs["matchness"] >= tau_high] = "EASY_POS"
    bins.loc[scored_pairs["matchness"] <= tau_low] = "EASY_NEG"
    return bins


def _sample_batch(
    scored_pairs: pd.DataFrame,
    batch_size: int,
    mix: Tuple[float, float, float],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Sample a batch according to the requested mix and heuristics."""
    if scored_pairs.empty or batch_size <= 0:
        return scored_pairs.head(0)

    mix = tuple(max(0.0, m) for m in mix)
    mix_total = sum(mix)
    if mix_total == 0:
        mix = (1.0, 0.0, 0.0)
        mix_total = 1.0
    mix = tuple(m / mix_total for m in mix)

    counts = [math.floor(batch_size * m) for m in mix]
    remainder = batch_size - sum(counts)
    order = np.argsort(mix)[::-1]
    for idx in order[:remainder]:
        counts[idx] += 1
    target = dict(zip(["EASY_POS", "EASY_NEG", "CORNER"], counts))

    samples: List[pd.DataFrame] = []
    shortages: Dict[str, int] = {}

    for bin_name in ["EASY_POS", "EASY_NEG"]:
        subset = scored_pairs[scored_pairs["bin"] == bin_name]
        n = target[bin_name]
        if len(subset) == 0 or n == 0:
            if n > 0:
                shortages[bin_name] = n
            continue
        n = min(n, len(subset))
        samples.append(subset.sample(n=n, random_state=rng.integers(0, 1_000_000)))

    # Corner handling with disagreement emphasis
    n_corner = target["CORNER"]
    if n_corner > 0:
        corner_pool = scored_pairs[scored_pairs["bin"] == "CORNER"]
        disagreements = corner_pool[corner_pool["is_disagreement"]]
        others = corner_pool[~corner_pool["is_disagreement"]]

        desired_disagreement = min(len(disagreements), math.ceil(n_corner * 0.4))
        if desired_disagreement > 0:
            samples.append(
                disagreements.sample(
                    n=desired_disagreement,
                    random_state=rng.integers(0, 1_000_000),
                )
            )
        remaining = n_corner - desired_disagreement

        if remaining > 0 and len(others) > 0:
            weights = others["rarity_score"].fillna(0.0)
            if weights.sum() == 0:
                weights = None
            samples.append(
                others.sample(
                    n=min(remaining, len(others)),
                    weights=weights,
                    random_state=rng.integers(0, 1_000_000),
                )
            )

        if remaining > len(others):
            shortages["CORNER"] = remaining - len(others)

    combined = pd.concat(samples, ignore_index=True) if samples else scored_pairs.head(0)

    # Redistribute shortages opportunistically
    for bin_name, shortage in shortages.items():
        if shortage <= 0:
            continue
        available = scored_pairs[~scored_pairs.index.isin(combined.index)]
        if available.empty:
            break
        extra = available.sample(
            n=min(shortage, len(available)),
            random_state=rng.integers(0, 1_000_000),
        )
        combined = pd.concat([combined, extra], ignore_index=True)

    # Drop duplicate pairs while preserving order
    if not combined.empty:
        combined = combined.drop_duplicates(subset=["id1", "id2"])

    return combined.head(batch_size)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_interactive_matching(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    *,
    id_column: str,
    blocking_options: Sequence[Dict[str, Any]],
    comparator_options: Sequence[Dict[str, Any]],
    matcher_threshold: float = 0.0,
    tau_low: float = 0.25,
    tau_high: float = 0.75,
    epsilon_random: float = 0.0001,
    batch_size: int = 50,
    mix: Tuple[float, float, float] = (0.35, 0.35, 0.30),
    random_seed: Optional[int] = None,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    embedding_device: Optional[str] = None,
    preview_columns: Optional[Sequence[str]] = None,
) -> MatchingSessionResult:
    """Run an interactive matching session with the configured components.

    Parameters
    ----------
    df_left, df_right:
        Source datasets containing the `id_column`.
    id_column:
        Name of the identifier column shared across both datasets.
    blocking_options:
        Iterable of blocker configuration dictionaries. Each dictionary should
        contain a ``type`` key (``"standard"``, ``"sorted_neighbourhood"``,
        ``"token"``, or ``"embedding"``) and optional ``params``.
    comparator_options:
        Iterable of comparator configuration dictionaries with keys:
        ``type`` (``"string"``, ``"numeric"``, ``"date"``), ``column``,
        ``weight`` (float), and optional parameter overrides passed to the
        comparator constructor.
    matcher_threshold:
        Minimum score threshold for the rule-based matcher. Set to 0.0 to keep
        all scored pairs and apply thresholds downstream.
    tau_low, tau_high:
        Thresholds for easy non-matches and easy matches respectively when
        computing bins.
    epsilon_random:
        Fraction of the Cartesian product to explore randomly for recall guard.
        A value like 0.0001 will sample ~0.01% of pairs.
    batch_size:
        Number of pairs to return in the curated batch.
    mix:
        Tuple describing the desired fraction of (easy+, easy-, corner) pairs.
    random_seed:
        Optional random seed to make sampling reproducible.
    embedding_model, embedding_device:
        Parameters passed to the sentence-transformer model when embedding
        similarities need to be computed.
    preview_columns:
        Optional iterable of columns to include in preview strings for UI display.

    Returns
    -------
    MatchingSessionResult
        Dataclass encapsulating the candidates, scored pairs, sample batch, and metrics.
    """
    if id_column not in df_left.columns or id_column not in df_right.columns:
        raise ValueError(f"ID column '{id_column}' must exist in both datasets")

    rng = np.random.default_rng(random_seed)

    # ------------------------------------------------------------------
    # Instantiate blockers and generate candidate pairs
    # ------------------------------------------------------------------
    blocker_instances: List[Tuple[str, BaseBlocker]] = []
    embedding_text_cols: List[str] = []

    for cfg in blocking_options:
        blocker_name, blocker = _instantiate_blocker(cfg, df_left, df_right, id_column)
        blocker_instances.append((blocker_name, blocker))
        if cfg.get("type") == "embedding":
            embedding_text_cols = cfg.get("params", {}).get("text_cols", []) or embedding_text_cols

    candidates = _union_blocker_outputs(blocker_instances)
    candidates = _inject_epsilon_random(
        candidates,
        df_left,
        df_right,
        id_column,
        epsilon_random,
        rng,
    )

    if candidates.empty:
        empty_df = pd.DataFrame(columns=["id1", "id2", "score", "matchness", "bin"])
        metrics = {
            "total_candidates": 0,
            "total_possible_pairs": len(df_left) * len(df_right),
            "reduction_ratio": 1.0,
            "bin_counts": {"EASY_POS": 0, "EASY_NEG": 0, "CORNER": 0},
        }
        return MatchingSessionResult(
            candidate_pairs=candidates,
            scored_pairs=empty_df,
            feature_matrix=pd.DataFrame(),
            sampled_batch=empty_df,
            metrics=metrics,
            config={
                "id_column": id_column,
                "blocking_options": blocking_options,
                "comparator_options": comparator_options,
                "matcher_threshold": matcher_threshold,
                "tau_low": tau_low,
                "tau_high": tau_high,
                "epsilon_random": epsilon_random,
                "batch_size": batch_size,
                "mix": mix,
                "random_seed": random_seed,
            },
        )

    # Degrees and rarity prior to scoring
    left_degree = candidates["id1"].value_counts()
    right_degree = candidates["id2"].value_counts()
    candidates["left_degree"] = candidates["id1"].map(left_degree).astype(int)
    candidates["right_degree"] = candidates["id2"].map(right_degree).astype(int)
    candidates["rarity_score"] = 1.0 / (candidates["left_degree"] + candidates["right_degree"]).replace(0, np.nan)

    # ------------------------------------------------------------------
    # Instantiate comparators and run rule-based matcher
    # ------------------------------------------------------------------
    comparators: List[BaseComparator] = []
    weights: List[float] = []
    comparator_meta: List[Dict[str, Any]] = []

    for cfg in comparator_options:
        comp_type = cfg.get("type")
        column = cfg.get("column")
        weight = cfg.get("weight", 1.0)
        params = cfg.get("params", {})

        if comp_type == "string":
            comparator = StringComparator(
                column=column,
                similarity_function=params.get("similarity_function", "jaro_winkler"),
                tokenization=params.get("tokenization"),
                preprocess=params.get("preprocess"),
                list_strategy=params.get("list_strategy"),
            )
            debug_name = (
                f"StringComparator({column}, {comparator.similarity_function}"
                f"{', list_strategy=' + str(comparator.list_strategy) if comparator.list_strategy else ''})"
            )
            comparator_meta.append({"name": debug_name, "category": "string", "column": column})
        elif comp_type == "numeric":
            comparator = NumericComparator(
                column=column,
                method=params.get("method", "relative_difference"),
                max_difference=params.get("max_difference"),
                list_strategy=params.get("list_strategy"),
            )
            debug_name = (
                f"NumericComparator({column}, {comparator.method}"
                f"{', list_strategy=' + str(comparator.list_strategy) if comparator.list_strategy else ''})"
            )
            comparator_meta.append({"name": debug_name, "category": "numeric", "column": column})
        elif comp_type == "date":
            comparator = DateComparator(
                column=column,
                max_days_difference=params.get("max_days_difference"),
                list_strategy=params.get("list_strategy"),
            )
            debug_name = (
                f"DateComparator({column}"
                f"{', list_strategy=' + str(comparator.list_strategy) if comparator.list_strategy else ''})"
            )
            comparator_meta.append({"name": debug_name, "category": "date", "column": column})
        else:
            raise ValueError(f"Unsupported comparator type: {comp_type}")

        comparators.append(comparator)
        weights.append(weight)

    if not comparators:
        raise ValueError("At least one comparator must be configured")

    norm_weights = _normalize_weights(weights)
    matcher = RuleBasedMatcher()
    candidate_pairs_only = candidates[["id1", "id2"]]

    correspondences, debug_df = matcher.match(
        df_left=df_left,
        df_right=df_right,
        candidates=candidate_pairs_only,
        id_column=id_column,
        comparators=comparators,
        weights=norm_weights,
        threshold=matcher_threshold,
        debug=True,
    )

    feature_matrix = _build_feature_matrix(debug_df)

    scored_pairs = candidates.merge(correspondences[["id1", "id2", "score"]], on=["id1", "id2"], how="left")
    scored_pairs["score"] = scored_pairs["score"].fillna(0.0)
    scored_pairs = scored_pairs.merge(feature_matrix, on=["id1", "id2"], how="left")

    # ------------------------------------------------------------------
    # Derived heuristics & matchness
    # ------------------------------------------------------------------
    string_cols = [meta["name"] for meta in comparator_meta if meta["category"] == "string"]
    numeric_cols = [meta["name"] for meta in comparator_meta if meta["category"] == "numeric"]
    date_cols = [meta["name"] for meta in comparator_meta if meta["category"] == "date"]

    scored_pairs["string_similarity_mean"] = _safe_row_mean(scored_pairs, string_cols)
    scored_pairs["numeric_similarity_mean"] = _safe_row_mean(scored_pairs, numeric_cols)
    scored_pairs["date_similarity_mean"] = _safe_row_mean(scored_pairs, date_cols)

    KEY_BLOCKERS = {"StandardBlocker", "TokenBlocker"}
    def _key_overlap_score(sources: Tuple[str, ...]) -> float:
        sources_set = set(sources)
        if not sources_set:
            return 0.0
        if sources_set & KEY_BLOCKERS:
            return 1.0
        if "SortedNeighbourhoodBlocker" in sources_set:
            return 0.5
        return 0.0

    scored_pairs["key_overlap_score"] = scored_pairs["source_blockers"].apply(_key_overlap_score)

    if embedding_text_cols:
        scored_pairs["embedding_similarity"] = _compute_embedding_similarity(
            df_left,
            df_right,
            id_column,
            scored_pairs[["id1", "id2"]],
            embedding_text_cols,
            model_name=embedding_model,
            device=embedding_device,
        )
    else:
        scored_pairs["embedding_similarity"] = np.nan

    def _compute_matchness(row: pd.Series) -> float:
        components = []
        if not math.isnan(row.get("embedding_similarity", np.nan)):
            components.append((0.55, float(row["embedding_similarity"])))
        if not math.isnan(row.get("key_overlap_score", np.nan)):
            components.append((0.25, float(row["key_overlap_score"])))
        string_component = row.get("score", np.nan)
        if not math.isnan(string_component):
            components.append((0.20, float(string_component)))

        if not components:
            return float(row.get("score", 0.0))

        weight_sum = sum(weight for weight, value in components if not math.isnan(value))
        if weight_sum == 0:
            return float(row.get("score", 0.0))
        matchness = sum(weight * value for weight, value in components if not math.isnan(value))
        return matchness / weight_sum

    scored_pairs["matchness"] = scored_pairs.apply(_compute_matchness, axis=1)

    # Uncertainty (entropy) derived from matchness as pseudo probability
    prob = scored_pairs["matchness"].clip(lower=1e-6, upper=1 - 1e-6)
    scored_pairs["uncertainty"] = -(prob * np.log(prob) + (1 - prob) * np.log(1 - prob))

    feature_cols = [col for col in scored_pairs.columns if col.startswith(("StringComparator", "NumericComparator", "DateComparator"))]
    if feature_cols:
        scored_pairs["similarity_span"] = scored_pairs[feature_cols].max(axis=1, skipna=True) - scored_pairs[feature_cols].min(axis=1, skipna=True)
    else:
        scored_pairs["similarity_span"] = 0.0

    scored_pairs["is_disagreement"] = (
        scored_pairs["similarity_span"].fillna(0.0) >= 0.4
    ) | (scored_pairs["source_count"] > 1) | scored_pairs["epsilon_flag"]

    scored_pairs["bin"] = _assign_bins(scored_pairs, tau_low, tau_high)
    scored_pairs["bin_reason"] = np.where(
        scored_pairs["bin"] == "CORNER",
        np.where(
            scored_pairs["is_disagreement"],
            "disagreement",
            np.where(
                scored_pairs["uncertainty"] >= np.quantile(scored_pairs["uncertainty"], 0.7),
                "uncertainty",
                "threshold"
            ),
        ),
        scored_pairs["bin"],
    )

    # Previews for UI rendering
    columns_for_preview = list(preview_columns) if preview_columns else [meta["column"] for meta in comparator_meta][:3]
    left_preview_lookup = _prepare_preview_lookup(df_left, id_column, columns_for_preview)
    right_preview_lookup = _prepare_preview_lookup(df_right, id_column, columns_for_preview)
    scored_pairs["left_preview"] = scored_pairs["id1"].map(left_preview_lookup).fillna(scored_pairs["id1"].astype(str))
    scored_pairs["right_preview"] = scored_pairs["id2"].map(right_preview_lookup).fillna(scored_pairs["id2"].astype(str))
    scored_pairs["source_blockers_display"] = scored_pairs["source_blockers"].apply(lambda src: ", ".join(src))

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    batch_df = _sample_batch(scored_pairs, batch_size, mix, rng)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    total_candidates = len(scored_pairs)
    total_possible = len(df_left) * len(df_right)
    reduction_ratio = 1.0 if total_possible == 0 else 1 - (total_candidates / total_possible)
    bin_counts = scored_pairs["bin"].value_counts().to_dict()

    metrics: Dict[str, Any] = {
        "total_candidates": total_candidates,
        "total_possible_pairs": total_possible,
        "reduction_ratio": reduction_ratio,
        "bin_counts": {k: bin_counts.get(k, 0) for k in ["EASY_POS", "EASY_NEG", "CORNER"]},
        "epsilon_pairs": int(scored_pairs["epsilon_flag"].sum()),
        "mix": {
            "requested": {"easy_pos": mix[0], "easy_neg": mix[1], "corner": mix[2]},
            "batch_counts": batch_df["bin"].value_counts().to_dict(),
        },
        "thresholds": {"tau_low": tau_low, "tau_high": tau_high},
    }

    return MatchingSessionResult(
        candidate_pairs=candidates,
        scored_pairs=scored_pairs,
        feature_matrix=feature_matrix,
        sampled_batch=batch_df,
        metrics=metrics,
        config={
            "id_column": id_column,
            "blocking_options": blocking_options,
            "comparator_options": comparator_options,
            "matcher_threshold": matcher_threshold,
            "tau_low": tau_low,
            "tau_high": tau_high,
            "epsilon_random": epsilon_random,
            "batch_size": batch_size,
            "mix": mix,
            "random_seed": random_seed,
            "embedding_model": embedding_model,
            "embedding_device": embedding_device,
        },
    )


__all__ = ["MatchingSessionResult", "run_interactive_matching"]
