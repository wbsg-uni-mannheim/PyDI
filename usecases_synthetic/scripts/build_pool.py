#!/usr/bin/env python3
"""Build the per-domain likely-positive pool for plan_s1_scale.md R3.

The pool is a "probably-positive, do not perturb" protection set used
during knob-driven augmentation. It combines two evidence streams:

1. **Human EM gold** (``usecases/<d>/input/entitymatching/*.csv``) and
   the human-baseline pipeline correspondences
   (``usecases/<d>/output/debug_results_entity_matching/matching_detailed_results.csv``).
2. **Ditto PLM predictions** at ``theta=0.5`` from the R2 checkpoint at
   ``cache/ditto_checkpoints/<d>/best``, computed over a candidate set
   produced by the per-source-pair blocker sweep
   (``recall_floor=0.97``, tie-breaker = reduction ratio, mirroring
   ``config/committees/em_blocking_committee*.yaml``).

Bucket policy:

- A. Gold positives — kept unconditionally.
- B. Pairs in (human-baseline AND Ditto>=theta) — kept (agreement).
- C. Singletons (only one method declared positive) — gated by the
  PLM-based check: Ditto score >= theta+delta keeps; below theta-delta
  drops; in the margin band an LLM (``gpt-5.4``, temperature=0)
  adjudicates.

``delta`` is estimated per-domain from the R2 evaluation predictions
(``predictions.csv`` next to ``metrics.json`` in the chosen run): the
90th percentile of ``|prob - theta|`` on Ditto misclassifications,
clipped to ``[0.05, 0.20]``.

Output
------
``usecases_synthetic/pools/<domain>/pooled_positives.csv``
    Columns: ``id1``, ``id2``, ``source_1``, ``source_2``, ``score``,
    ``in_gold``, ``in_human``, ``in_ditto``, ``decision_path``.

``usecases_synthetic/pools/<domain>/pool_stats.json``
    Per-source-pair blocker sweep, candidate counts, bucket
    breakdowns, delta estimation telemetry, cluster-size distribution.

Resumability
------------
Each per-source-pair stage caches intermediate artefacts at
``usecases_synthetic/cache/pool_builder/<domain>/<src1>_<src2>/``:

- ``candidates.parquet`` — full candidate set
- ``ditto_scores.parquet`` — Ditto scores

Re-running the script reuses these files when present (delete them to
force a refresh). The ``DittoMatcher`` itself flushes per-batch scores
to ``cache/ditto_inference/`` after every batch, so a SIGINT or
lid-close mid-inference resumes from the last completed batch.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Load OPENAI_API_KEY (and any other secrets) from .env so the
# bucket-C adjudicator's ChatOpenAI client can authenticate.
try:
    from dotenv import load_dotenv  # noqa: E402

    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

from usecases_synthetic.lib.ditto_matcher import DittoMatcher  # noqa: E402
from usecases_synthetic.lib.llm_cache import LLMCache  # noqa: E402
from usecases_synthetic.lib.loaders import load_domain_sources  # noqa: E402
from usecases_synthetic.lib.pool_builder import (  # noqa: E402
    DITTO_THRESHOLD,
    POOL_PROMPT_VERSION,
    BlockerSweepResult,
    BucketResult,
    Pair,
    apply_column_mapping,
    assemble_candidate_set,
    build_buckets,
    build_pool_llm_adjudicator,
    canonical_pair,
    closure_added_counts,
    cluster_size_distribution,
    estimate_delta_from_predictions,
    load_em_gold_for_pair,
    load_human_baseline_pairs_from_files,
    make_blocker_candidates,
    score_with_ditto,
    sweep_blockers,
    transitive_closure_across_pairs,
)

logger = logging.getLogger("build_pool")


# ---------------------------------------------------------------------------
# Domain configuration
# ---------------------------------------------------------------------------


@dataclass
class SourceMapping:
    """Per-source column rename map (raw -> canonical) for a domain."""

    name_column: str  # raw column name carrying the entity label
    column_mapping: dict[str, str] = field(default_factory=dict)


@dataclass
class PairSpec:
    """Per-source-pair gold-file specification."""

    sources: tuple[str, str]
    gold_files: list[str]
    val_files: list[str] = field(default_factory=list)


@dataclass
class DomainSpec:
    """All per-domain wiring needed to build the pool."""

    name: str
    source_prefix_rules: list[tuple[str, str]]  # (substring, source_label)
    column_mappings: dict[str, SourceMapping]
    pairs: list[PairSpec]
    ditto_fields: list[str]
    ditto_checkpoint: Path
    em_gold_dir: Path
    correspondence_dir: Path


# Source-id -> source-label rules. First match wins (mirrors the prior
# build_pool.py and the K2 source_lookup convention).
COMPANIES_PREFIXES = [
    ("http://dbpedia.org/", "dbpedia"),
    ("http://www.forbes.com/", "forbes"),
    ("fullcontact_", "fullcontact"),
]
GAMES_PREFIXES = [
    ("dbpedia_", "dbpedia"),
    ("metacritic_", "metacritic"),
    ("sales_", "sales"),
]
MUSIC_PREFIXES = [
    ("mbrainz_", "musicbrainz"),
    ("discogs_", "discogs"),
    ("lastFM_", "lastfm"),
    ("lastfm_", "lastfm"),
]


def _r2_run_dir(domain: str) -> Path:
    """Resolve the R2 checkpoint's run directory.

    ``best/`` is a symlink to ``run_<ts>/checkpoints/best/``; the
    sibling ``metrics.json`` and ``predictions.csv`` live two levels up
    from ``best/``.
    """
    best = REPO_ROOT / "cache" / "ditto_checkpoints" / domain / "best"
    if not best.exists():
        raise FileNotFoundError(f"Missing Ditto best/ symlink: {best}")
    real = best.resolve()  # .../run_<ts>/checkpoints/best
    return real.parent.parent  # .../run_<ts>


COMPANIES_SPEC = DomainSpec(
    name="companies",
    source_prefix_rules=COMPANIES_PREFIXES,
    column_mappings={
        "dbpedia": SourceMapping(
            name_column="name",
            column_mapping={
                "org_name": "name",
                "established": "founded",
                "nation": "country",
                "headquarters": "city",
                "sector": "industry",
            },
        ),
        "forbes": SourceMapping(
            name_column="name",
            column_mapping={
                "company": "name",
                "region": "country",
                "business_segment": "industry",
            },
        ),
        "fullcontact": SourceMapping(
            name_column="name",
            column_mapping={
                "Attribute_2": "name",
                "Attribute_3": "country",
                "Attribute_4": "city",
                "Attribute_6": "founded",
            },
        ),
    },
    pairs=[
        PairSpec(
            sources=("forbes", "dbpedia"),
            gold_files=[
                "forbes_2_dbpedia_train.csv",
                "forbes_2_dbpedia_val.csv",
                "forbes_2_dbpedia_test.csv",
            ],
            val_files=["forbes_2_dbpedia_val.csv"],
        ),
        PairSpec(
            sources=("forbes", "fullcontact"),
            gold_files=[
                "forbes_2_fullcontact_train.csv",
                "forbes_2_fullcontact_val.csv",
                "forbes_2_fullcontact_test.csv",
            ],
            val_files=["forbes_2_fullcontact_val.csv"],
        ),
    ],
    ditto_fields=["name", "country", "city", "industry", "sector", "founded"],
    ditto_checkpoint=REPO_ROOT / "cache" / "ditto_checkpoints" / "companies" / "best",
    em_gold_dir=REPO_ROOT / "usecases" / "companies" / "input" / "entitymatching",
    correspondence_dir=REPO_ROOT / "usecases" / "companies" / "output",
)


GAMES_SPEC = DomainSpec(
    name="games",
    source_prefix_rules=GAMES_PREFIXES,
    column_mappings={
        "dbpedia": SourceMapping(
            name_column="name",
            column_mapping={
                "title": "name",
                "launch_yr": "releaseYear",
                "studio": "developer",
                "system": "platform",
                "genre": "genres",
                "franchise": "series",
            },
        ),
        "metacritic": SourceMapping(
            name_column="name",
            column_mapping={
                "game_title": "name",
                "year_published": "releaseYear",
                "made_by": "developer",
                "console": "platform",
                "press_rating": "criticScore",
                "player_rating": "userScore",
                "age_rating": "ESRB",
            },
        ),
        "sales": SourceMapping(
            name_column="name",
            column_mapping={
                "prod_title": "name",
                "launch_dt": "releaseYear",
                "studio": "developer",
                "dist": "publisher",
                "hw": "platform",
                "genre": "genres",
                "press_score": "criticScore",
                "comm_rating": "userScore",
                "age_classification": "ESRB",
                "units_sold_mm": "globalSales",
            },
        ),
    },
    # Games gold files come in mixed orientation (e.g.
    # metacritic_2_dbpedia_train.csv plus dbpedia_2_metacritic_test.csv);
    # all are deduplicated post-canonical_pair.
    pairs=[
        PairSpec(
            sources=("dbpedia", "metacritic"),
            gold_files=[
                "metacritic_2_dbpedia_train.csv",
                "dbpedia_2_metacritic_test.csv",
                "metacritic_2_dbpedia_test.csv",
            ],
            val_files=[],  # no _val split shipped; falls back to first file
        ),
        PairSpec(
            sources=("dbpedia", "sales"),
            gold_files=[
                "dbpedia_2_sales_train.csv",
                "dbpedia_2_sales_test.csv",
            ],
            val_files=[],
        ),
        PairSpec(
            sources=("metacritic", "sales"),
            gold_files=[
                "metacritic_2_sales_train.csv",
                "metacritic_2_sales_test.csv",
            ],
            val_files=[],
        ),
    ],
    ditto_fields=[
        "name",
        "platform",
        "genres",
        "developer",
        "publisher",
        "releaseYear",
        "criticScore",
        "userScore",
        "ESRB",
    ],
    ditto_checkpoint=REPO_ROOT / "cache" / "ditto_checkpoints" / "games" / "best",
    em_gold_dir=REPO_ROOT / "usecases" / "games" / "input" / "entitymatching",
    correspondence_dir=REPO_ROOT / "usecases" / "games" / "output",
)


MUSIC_SPEC = DomainSpec(
    name="music",
    source_prefix_rules=MUSIC_PREFIXES,
    column_mappings={
        "musicbrainz": SourceMapping(name_column="name"),
        "discogs": SourceMapping(name_column="name"),
        "lastfm": SourceMapping(name_column="name"),
    },
    pairs=[
        PairSpec(
            sources=("musicbrainz", "discogs"),
            gold_files=[
                "musicbrainz_2_discogs_train.csv",
                "musicbrainz_2_discogs_val.csv",
                "musicbrainz_2_discogs_test.csv",
            ],
            val_files=["musicbrainz_2_discogs_val.csv"],
        ),
        PairSpec(
            sources=("musicbrainz", "lastfm"),
            gold_files=[
                "musicbrainz_2_lastfm_train.csv",
                "musicbrainz_2_lastfm_val.csv",
                "musicbrainz_2_lastfm_test.csv",
            ],
            val_files=["musicbrainz_2_lastfm_val.csv"],
        ),
    ],
    # ``label`` stays out of Ditto fields (collides with binary
    # classification target — see plans/plan_s1_scale.md per-domain
    # music caveats).
    ditto_fields=[
        "name",
        "artist",
        "release-date",
        "release-country",
        "duration",
        "genre",
        # 2026-05-31: added to match the committee Ditto's effective scope
        # (committee_ditto_fields("music") == these 7; the 8th committee
        # field `label` is a reserved Ditto WDC key, dropped at
        # serialization, and is discogs-only/asymmetric anyway). `tracks`
        # is 100/100/47% covered and highly discriminative for releases,
        # so the pool reuses the committee Ditto on a matching surface.
        "tracks",
    ],
    ditto_checkpoint=REPO_ROOT / "cache" / "ditto_checkpoints" / "music" / "best",
    em_gold_dir=REPO_ROOT / "usecases" / "music" / "input" / "entitymatching",
    correspondence_dir=REPO_ROOT / "usecases" / "music" / "output",
)


SPECS: dict[str, DomainSpec] = {
    "companies": COMPANIES_SPEC,
    "games": GAMES_SPEC,
    "music": MUSIC_SPEC,
}


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------


def build_openai_client(model_name: str) -> Callable[[str], str]:
    """Build a thin OpenAI chat caller bound to ``model_name``.

    Uses ``langchain_openai.ChatOpenAI`` (already a transitive dep via
    other synthetic modules). Temperature is pinned to 0 for cache
    determinism.
    """
    from langchain_openai import ChatOpenAI

    chat = ChatOpenAI(model=model_name, temperature=0)

    def _call(prompt: str) -> str:
        result = chat.invoke(prompt)
        if hasattr(result, "content"):
            content = result.content
            if isinstance(content, list):
                return "".join(
                    str(p) if not isinstance(p, dict) else str(p.get("text", ""))
                    for p in content
                )
            return str(content)
        return str(result)

    return _call


# ---------------------------------------------------------------------------
# Per-source-pair builder
# ---------------------------------------------------------------------------


def _source_lookup_factory(rules: list[tuple[str, str]]) -> Callable[[str], str]:
    def lookup(rid: str) -> str:
        for substr, label in rules:
            if rid.startswith(substr) or substr in rid:
                return label
        return "unknown"

    return lookup


def _materialise_pair_inputs(
    spec: DomainSpec,
    pair: PairSpec,
    sources: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, set[Pair]]:
    """Return (df_left, df_right, gold_positives) ready for the sweep."""
    src_a, src_b = pair.sources
    if src_a not in sources or src_b not in sources:
        raise KeyError(f"Source missing for pair {pair.sources}")
    df_left = sources[src_a]
    df_right = sources[src_b]
    gold = load_em_gold_for_pair(spec.em_gold_dir, pair.gold_files, pair.val_files)
    return df_left, df_right, gold.positives


def _persist_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except (ImportError, ValueError):
        df.to_csv(path.with_suffix(".csv"), index=False)


def _load_parquet_or_csv(path: Path) -> pd.DataFrame | None:
    if path.exists():
        try:
            return pd.read_parquet(path)
        except (ImportError, ValueError):
            pass
    csv_path = path.with_suffix(".csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


@dataclass
class PairSignals:
    """Phase-1 output for one source-pair: blocker + Ditto inference."""

    src_a: str
    src_b: str
    pair_key: str
    pair_dir: Path
    df_left: pd.DataFrame
    df_right: pd.DataFrame
    left_ids: set[str]
    right_ids: set[str]
    gold_positives: set[Pair]
    candidates: pd.DataFrame
    ditto_scores: dict[Pair, float]
    sweep_telemetry: dict[str, Any]


def compute_pair_signals(
    spec: DomainSpec,
    pair: PairSpec,
    sources: dict[str, pd.DataFrame],
    human_pairs: set[Pair],
    matcher: DittoMatcher,
    *,
    cache_dir: Path,
    refresh: bool = False,
) -> PairSignals:
    """Phase 1 of the pool build: blocker sweep + Ditto inference.

    Doesn't apply transitive closure or build buckets; just produces
    the per-pair Ditto-score dict needed downstream. Cached
    intermediates (candidates, ditto_scores) are reused unless
    ``refresh=True``.
    """
    src_a, src_b = pair.sources
    pair_key = f"{src_a}_{src_b}"
    pair_dir = cache_dir / pair_key
    pair_dir.mkdir(parents=True, exist_ok=True)

    df_left_raw, df_right_raw, gold_positives = _materialise_pair_inputs(
        spec, pair, sources
    )
    df_left = apply_column_mapping(
        df_left_raw, spec.column_mappings[src_a].column_mapping
    )
    df_right = apply_column_mapping(
        df_right_raw, spec.column_mappings[src_b].column_mapping
    )

    name_col_left = spec.column_mappings[src_a].name_column
    if name_col_left not in df_left.columns:
        raise KeyError(
            f"name column {name_col_left!r} missing from {src_a} after mapping; "
            f"have {list(df_left.columns)[:8]}"
        )

    candidates_path = pair_dir / "candidates.parquet"
    sweep_path = pair_dir / "sweep.json"
    cached_candidates = _load_parquet_or_csv(candidates_path) if not refresh else None
    sweep_telemetry: dict[str, Any]

    if cached_candidates is not None and sweep_path.exists():
        with open(sweep_path) as f:
            sweep_telemetry = json.load(f)
        candidates = cached_candidates
        logger.info(
            "[%s/%s] reusing cached candidates (%d) from %s",
            spec.name,
            pair_key,
            len(candidates),
            candidates_path,
        )
    else:
        logger.info("[%s/%s] running blocker sweep ...", spec.name, pair_key)
        blocker_candidates = make_blocker_candidates(
            name_column=spec.column_mappings[src_a].name_column,
            output_dir=str(pair_dir / "blocker_debug"),
        )
        sweep: BlockerSweepResult = sweep_blockers(
            df_left, df_right, gold_positives, blocker_candidates
        )
        sweep_telemetry = {
            "winner": sweep.winner,
            "cleared_floor": sweep.cleared_floor,
            "per_blocker": sweep.per_blocker,
        }
        with open(sweep_path, "w") as f:
            json.dump(sweep_telemetry, f, indent=2, sort_keys=True)
        candidates = assemble_candidate_set(
            sweep.candidates, human_pairs, gold_positives
        )
        _persist_parquet(candidates, candidates_path)
        logger.info(
            "[%s/%s] winner=%s cleared_floor=%s; assembled %d candidates",
            spec.name,
            pair_key,
            sweep.winner,
            sweep.cleared_floor,
            len(candidates),
        )

    ditto_path = pair_dir / "ditto_scores.parquet"
    cached_scores = _load_parquet_or_csv(ditto_path) if not refresh else None
    if cached_scores is not None:
        scores_df = cached_scores
        logger.info(
            "[%s/%s] reusing cached Ditto scores (%d) from %s",
            spec.name,
            pair_key,
            len(scores_df),
            ditto_path,
        )
    else:
        logger.info(
            "[%s/%s] scoring %d candidates with Ditto ...",
            spec.name,
            pair_key,
            len(candidates),
        )
        # Re-orient candidates to the (df_left, df_right) source order.
        # `assemble_candidate_set` returns canonical-sorted (id1 <= id2)
        # pairs for set semantics, but DittoMatcher needs id1 to live
        # in df_left's index and id2 in df_right's. The blocker output
        # was already in (left, right) order, but human-baseline + gold
        # pairs may carry either orientation.
        left_ids = set(df_left["id"].astype(str))
        right_ids = set(df_right["id"].astype(str))
        oriented_rows: list[tuple[str, str]] = []
        skipped = 0
        for row in candidates.itertuples(index=False):
            id1, id2 = str(row.id1), str(row.id2)
            if id1 in left_ids and id2 in right_ids:
                oriented_rows.append((id1, id2))
            elif id2 in left_ids and id1 in right_ids:
                oriented_rows.append((id2, id1))
            else:
                skipped += 1
        if skipped:
            logger.warning(
                "[%s/%s] %d candidate pairs dropped (id not in source frames)",
                spec.name,
                pair_key,
                skipped,
            )
        oriented = pd.DataFrame(oriented_rows, columns=["id1", "id2"])
        scores_df = score_with_ditto(matcher, df_left, df_right, oriented)
        _persist_parquet(scores_df, ditto_path)

    ditto_scores: dict[Pair, float] = {}
    for row in scores_df.itertuples(index=False):
        ditto_scores[canonical_pair(str(row.id1), str(row.id2))] = float(row.score)

    return PairSignals(
        src_a=src_a,
        src_b=src_b,
        pair_key=pair_key,
        pair_dir=pair_dir,
        df_left=df_left,
        df_right=df_right,
        left_ids=set(df_left["id"].astype(str)),
        right_ids=set(df_right["id"].astype(str)),
        gold_positives=gold_positives,
        candidates=candidates,
        ditto_scores=ditto_scores,
        sweep_telemetry=sweep_telemetry,
    )


def score_extra_pairs_with_ditto(
    matcher: DittoMatcher,
    signals: PairSignals,
    new_pairs: set[Pair],
) -> dict[Pair, float]:
    """Score Ditto on transitively-implied pairs not yet in the cache.

    The transitive closure can introduce cross-source pairs that were
    not present in any blocker output, so they have no entry in the
    per-pair ``signals.ditto_scores`` dict. The bucket logic needs a
    score for the margin-band check; without one, those pairs would
    default to ``score=0.0`` and be dropped as confident-negative even
    though they came out of the high-confidence closure.

    Returns a dict mapping each newly-scored canonical pair to its
    Ditto softmax probability.
    """
    if not new_pairs:
        return {}
    oriented_rows: list[tuple[str, str]] = []
    for id1, id2 in new_pairs:
        if id1 in signals.left_ids and id2 in signals.right_ids:
            oriented_rows.append((id1, id2))
        elif id2 in signals.left_ids and id1 in signals.right_ids:
            oriented_rows.append((id2, id1))
    if not oriented_rows:
        return {}
    oriented = pd.DataFrame(oriented_rows, columns=["id1", "id2"])
    scores_df = score_with_ditto(matcher, signals.df_left, signals.df_right, oriented)
    out: dict[Pair, float] = {}
    for row in scores_df.itertuples(index=False):
        out[canonical_pair(str(row.id1), str(row.id2))] = float(row.score)
    return out


def assemble_pair_buckets(
    signals: PairSignals,
    *,
    expanded_human: set[Pair],
    expanded_ditto_positive: set[Pair],
    delta: float,
    adjudicator: Callable[[Pair], bool | None],
) -> tuple[BucketResult, dict[str, Any]]:
    """Phase-3 bucket assembly using closure-expanded human + Ditto streams.

    ``expanded_human`` and ``expanded_ditto_positive`` are the per-
    source-pair entries from the cross-pair transitive closure. The
    Ditto-side closure may include pairs without a direct
    ``signals.ditto_scores`` entry; the caller must have rescored those
    via :func:`score_extra_pairs_with_ditto` first.

    Pairs that are in ``expanded_ditto_positive`` but lack any direct
    ditto_scores entry (rescore failed, e.g. id resolved to a different
    pair's source frame) get a synthetic score of 1.0 — they came from
    closure of the high-confidence stream, so this errs on the side of
    inclusion.
    """
    ditto_scores_for_buckets = dict(signals.ditto_scores)
    for pair in expanded_ditto_positive:
        if pair not in ditto_scores_for_buckets:
            # Closure-implied but unscored: treat as confident positive.
            ditto_scores_for_buckets[pair] = 1.0
    n_ditto_pos = sum(
        1 for s in ditto_scores_for_buckets.values() if s >= DITTO_THRESHOLD
    )

    bucket_result = build_buckets(
        gold_positives=signals.gold_positives,
        human_pairs=expanded_human,
        ditto_scores=ditto_scores_for_buckets,
        delta=delta,
        adjudicator=adjudicator,
        threshold=DITTO_THRESHOLD,
    )
    bucket_result.pool_df.insert(2, "source_1", signals.src_a)
    bucket_result.pool_df.insert(3, "source_2", signals.src_b)

    telemetry = {
        "sources": [signals.src_a, signals.src_b],
        "n_left": int(len(signals.df_left)),
        "n_right": int(len(signals.df_right)),
        "n_gold_positives": int(len(signals.gold_positives)),
        "n_human_pairs_expanded": int(len(expanded_human)),
        "n_candidates": int(len(signals.candidates)),
        "n_ditto_positive_at_theta": int(n_ditto_pos),
        "blocker_sweep": signals.sweep_telemetry,
        "buckets": {
            "a_gold": bucket_result.bucket_a,
            "b_agreement": bucket_result.bucket_b,
            "c_total": bucket_result.bucket_c_total,
            "c_kept_confident": bucket_result.bucket_c_kept_confident,
            "c_kept_llm": bucket_result.bucket_c_kept_llm,
            "c_dropped_confident": bucket_result.bucket_c_dropped_confident,
            "c_dropped_llm": bucket_result.bucket_c_dropped_llm,
        },
        "delta_used": bucket_result.delta_used,
        "pool_size": int(len(bucket_result.pool_df)),
    }
    return bucket_result, telemetry


# ---------------------------------------------------------------------------
# Per-domain orchestration
# ---------------------------------------------------------------------------


def build_domain(
    spec: DomainSpec,
    *,
    llm_model: str,
    no_llm: bool,
    refresh: bool,
) -> None:
    """Build the pool for one domain."""
    logger.info("[%s] loading sources ...", spec.name)
    sources = load_domain_sources(spec.name)
    logger.info(
        "[%s] sources: %s",
        spec.name,
        {n: len(df) for n, df in sources.items()},
    )

    canonical_sources: dict[str, pd.DataFrame] = {}
    for src_name, df in sources.items():
        mapping = spec.column_mappings.get(src_name)
        if mapping is None:
            canonical_sources[src_name] = df
        else:
            canonical_sources[src_name] = apply_column_mapping(
                df, mapping.column_mapping
            )

    source_lookup = _source_lookup_factory(spec.source_prefix_rules)
    logger.info("[%s] loading human-baseline correspondences ...", spec.name)
    correspondence_files = sorted(spec.correspondence_dir.glob("correspondences_*.csv"))
    if not correspondence_files:
        logger.warning(
            "[%s] no correspondences_*.csv in %s; human-baseline stream will be empty",
            spec.name,
            spec.correspondence_dir,
        )
    human_by_pair = load_human_baseline_pairs_from_files(
        correspondence_files, source_lookup
    )
    for k, v in sorted(human_by_pair.items(), key=lambda kv: sorted(kv[0])):
        logger.info("[%s]   human-baseline %s: %d pairs", spec.name, sorted(k), len(v))

    run_dir = _r2_run_dir(spec.name)
    predictions_csv = run_dir / "predictions.csv"
    if not predictions_csv.exists():
        raise FileNotFoundError(
            f"R2 predictions.csv not found next to checkpoint: {predictions_csv}"
        )
    delta, delta_telemetry = estimate_delta_from_predictions(predictions_csv)
    logger.info(
        "[%s] delta=%.4f (raw_p%g=%.4f, %d misclassified of %d)",
        spec.name,
        delta,
        delta_telemetry["percentile"],
        delta_telemetry["raw_percentile"],
        delta_telemetry["n_misclassified"],
        delta_telemetry["n_predictions"],
    )

    cache_dir = REPO_ROOT / "usecases_synthetic" / "cache" / "pool_builder" / spec.name
    cache_dir.mkdir(parents=True, exist_ok=True)

    matcher = DittoMatcher(
        checkpoint_path=spec.ditto_checkpoint,
        fields=spec.ditto_fields,
        batch_size=128,
    )

    llm_cache_dir = (
        REPO_ROOT / "usecases_synthetic" / "cache" / "llm_cache" / "r3_pool" / spec.name
    )
    llm_cache_dir.mkdir(parents=True, exist_ok=True)
    llm_cache = LLMCache(
        cache_dir=llm_cache_dir,
        prompt_version=POOL_PROMPT_VERSION,
        model_id=llm_model,
    )
    api_client: Callable[[str], str] | None
    if no_llm:
        api_client = None
        logger.info(
            "[%s] --no-llm: bucket-C margin band will drop conservatively", spec.name
        )
    else:
        try:
            api_client = build_openai_client(llm_model)
        except Exception as exc:
            logger.error(
                "[%s] failed to build LLM client (%s); falling back to no-LLM",
                spec.name,
                exc,
            )
            api_client = None

    adjudicator = build_pool_llm_adjudicator(
        domain=spec.name,
        sources=canonical_sources,
        fields=spec.ditto_fields,
        source_lookup=source_lookup,
        llm_cache=llm_cache,
        api_client=api_client,
        strict_cache=False,
    )

    # ----- Phase 1: per-pair signals (blocker sweep + Ditto inference) -----
    signals_by_pair: dict[frozenset[str], PairSignals] = {}
    spec_by_pair_key: dict[frozenset[str], PairSpec] = {
        frozenset(p.sources): p for p in spec.pairs
    }
    for pair in spec.pairs:
        pair_key = f"{pair.sources[0]}_{pair.sources[1]}"
        human_for_pair = human_by_pair.get(frozenset(pair.sources), set())
        signals = compute_pair_signals(
            spec=spec,
            pair=pair,
            sources=canonical_sources,
            human_pairs=human_for_pair,
            matcher=matcher,
            cache_dir=cache_dir,
            refresh=refresh,
        )
        signals_by_pair[frozenset(pair.sources)] = signals
        logger.info(
            "[%s/%s] phase1 done: %d candidates, %d Ditto-positive at theta=%.2f",
            spec.name,
            pair_key,
            len(signals.candidates),
            sum(1 for s in signals.ditto_scores.values() if s >= DITTO_THRESHOLD),
            DITTO_THRESHOLD,
        )

    # ----- Phase 2: cross-pair transitive closure (per evidence stream) -----
    human_per_pair: dict[frozenset[str], set[Pair]] = {
        k: set(v) for k, v in human_by_pair.items()
    }
    ditto_pos_per_pair: dict[frozenset[str], set[Pair]] = {
        frozenset((s.src_a, s.src_b)): {
            p for p, sc in s.ditto_scores.items() if sc >= DITTO_THRESHOLD
        }
        for s in signals_by_pair.values()
    }

    expanded_human = transitive_closure_across_pairs(
        human_per_pair, source_lookup=source_lookup
    )
    expanded_ditto = transitive_closure_across_pairs(
        ditto_pos_per_pair, source_lookup=source_lookup
    )

    closure_human_added = closure_added_counts(human_per_pair, expanded_human)
    closure_ditto_added = closure_added_counts(ditto_pos_per_pair, expanded_ditto)
    logger.info(
        "[%s] transitive closure: human added per-pair %s; ditto added per-pair %s",
        spec.name,
        closure_human_added,
        closure_ditto_added,
    )

    # ----- Phase 2b: synthesise PairSignals for closure-only source pairs -----
    # The closure can surface pairs whose source-pair was never
    # explicitly configured in spec.pairs (e.g. companies has only
    # forbes-dbpedia and forbes-fullcontact pairs in its spec; the
    # closure surfaces dbpedia-fullcontact via the forbes pivot). For
    # each such closure-only source-pair, fabricate a PairSignals so
    # the bucket assembler can include the implied pairs in the pool.
    closure_only_keys = (set(expanded_human.keys()) | set(expanded_ditto.keys())) - set(
        signals_by_pair.keys()
    )
    for key in sorted(closure_only_keys, key=lambda k: sorted(k)):
        src_a, src_b = sorted(key)
        if src_a not in canonical_sources or src_b not in canonical_sources:
            logger.warning(
                "[%s] closure-only source-pair %s skipped: source missing",
                spec.name,
                sorted(key),
            )
            continue
        synth_pair_key = f"{src_a}_{src_b}"
        synth_pair_dir = cache_dir / synth_pair_key
        synth_pair_dir.mkdir(parents=True, exist_ok=True)
        df_left = canonical_sources[src_a]
        df_right = canonical_sources[src_b]
        synth_signals = PairSignals(
            src_a=src_a,
            src_b=src_b,
            pair_key=synth_pair_key,
            pair_dir=synth_pair_dir,
            df_left=df_left,
            df_right=df_right,
            left_ids=set(df_left["id"].astype(str)),
            right_ids=set(df_right["id"].astype(str)),
            gold_positives=set(),
            candidates=pd.DataFrame(columns=["id1", "id2"]),
            ditto_scores={},
            sweep_telemetry={"closure_only": True},
        )
        signals_by_pair[key] = synth_signals
        logger.info(
            "[%s/%s] closure-only pair: %d human + %d ditto pairs from closure",
            spec.name,
            synth_pair_key,
            len(expanded_human.get(key, set())),
            len(expanded_ditto.get(key, set())),
        )

    # ----- Phase 3: score Ditto on transitively-implied pairs that lack scores -----
    extra_ditto_calls = 0
    for key, signals in signals_by_pair.items():
        # Pairs that are in either expanded set but not yet in this
        # source-pair's ditto_scores. These are the closure-implied
        # pairs that need a Ditto verdict for the margin-band check.
        in_expanded = expanded_human.get(key, set()) | expanded_ditto.get(key, set())
        missing = in_expanded - set(signals.ditto_scores.keys())
        if not missing:
            continue
        logger.info(
            "[%s/%s] scoring %d closure-implied pairs with Ditto ...",
            spec.name,
            signals.pair_key,
            len(missing),
        )
        extra = score_extra_pairs_with_ditto(matcher, signals, missing)
        signals.ditto_scores.update(extra)
        extra_ditto_calls += len(extra)
        # Persist the augmented Ditto score table for this source-pair
        # so subsequent --no-refresh runs see the closure scores too.
        ditto_path = signals.pair_dir / "ditto_scores.parquet"
        merged_df = pd.DataFrame(
            [(p[0], p[1], s) for p, s in signals.ditto_scores.items()],
            columns=["id1", "id2", "score"],
        )
        _persist_parquet(merged_df, ditto_path)

    # ----- Phase 4: per-pair bucket assembly with expanded streams -----
    # Iterate signals_by_pair (configured + closure-only synthetic
    # pairs) rather than spec.pairs alone, so closure-implied source
    # pairs are not silently dropped.
    per_pair_pools: list[pd.DataFrame] = []
    per_pair_telemetry: dict[str, dict[str, Any]] = {}
    for key in sorted(signals_by_pair.keys(), key=lambda k: sorted(k)):
        signals = signals_by_pair[key]
        sources_sorted = sorted(key)
        bucket, telemetry = assemble_pair_buckets(
            signals,
            expanded_human=expanded_human.get(key, set()),
            expanded_ditto_positive=expanded_ditto.get(key, set()),
            delta=delta,
            adjudicator=adjudicator,
        )
        # Per-pair telemetry includes the closure deltas (what came
        # from the cross-pair expansion vs the directly-observed set).
        telemetry["closure"] = {
            "human_added": closure_human_added.get("__".join(sources_sorted), 0),
            "ditto_added": closure_ditto_added.get("__".join(sources_sorted), 0),
            "n_human_pre_closure": len(human_per_pair.get(key, set())),
            "n_ditto_pos_pre_closure": len(ditto_pos_per_pair.get(key, set())),
            "closure_only_pair": key not in spec_by_pair_key,
        }
        per_pair_pools.append(bucket.pool_df)
        per_pair_telemetry[signals.pair_key] = telemetry
        logger.info(
            "[%s/%s] pool_size=%d (A=%d B=%d C+conf=%d C+llm=%d / "
            "dropped C-conf=%d C-llm=%d) closure_added: human=%d ditto=%d",
            spec.name,
            signals.pair_key,
            telemetry["pool_size"],
            telemetry["buckets"]["a_gold"],
            telemetry["buckets"]["b_agreement"],
            telemetry["buckets"]["c_kept_confident"],
            telemetry["buckets"]["c_kept_llm"],
            telemetry["buckets"]["c_dropped_confident"],
            telemetry["buckets"]["c_dropped_llm"],
            telemetry["closure"]["human_added"],
            telemetry["closure"]["ditto_added"],
        )

    pool_df = (
        pd.concat(per_pair_pools, ignore_index=True)
        if per_pair_pools
        else pd.DataFrame(
            columns=[
                "id1",
                "id2",
                "source_1",
                "source_2",
                "score",
                "in_gold",
                "in_human",
                "in_ditto",
                "decision_path",
            ]
        )
    )
    pool_df = pool_df.sort_values(
        ["source_1", "source_2", "id1", "id2"], kind="mergesort"
    ).reset_index(drop=True)

    out_dir = REPO_ROOT / "usecases_synthetic" / "pools" / spec.name
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "pooled_positives.csv"
    json_path = out_dir / "pool_stats.json"
    pool_df.to_csv(csv_path, index=False)

    cluster_stats = cluster_size_distribution(pool_df)

    stats = {
        "domain": spec.name,
        "ditto_checkpoint": str(spec.ditto_checkpoint),
        "ditto_threshold": DITTO_THRESHOLD,
        "delta_estimation": delta_telemetry,
        "llm_model_id": llm_model,
        "llm_used": api_client is not None,
        "pool_size": int(len(pool_df)),
        "per_pair": per_pair_telemetry,
        "transitive_closure": {
            "human_added_per_pair": closure_human_added,
            "ditto_added_per_pair": closure_ditto_added,
            "extra_ditto_calls": int(extra_ditto_calls),
        },
        "cluster_size_distribution": cluster_stats,
    }
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2, sort_keys=True)
    logger.info(
        "[%s] wrote pool_size=%d to %s",
        spec.name,
        len(pool_df),
        csv_path.relative_to(REPO_ROOT),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--domain",
        choices=list(SPECS),
        help="Build the pool for a single domain.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Build pools for all configured domains (companies, games, music).",
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-5.4",
        help="OpenAI model id used for bucket-C adjudication "
        "(default: gpt-5.4; matches the gpt-5.4-mini convention "
        "already used in committee_sm.py).",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip live LLM calls; pairs in the margin band are dropped "
        "conservatively. Use for cache-only / dry runs.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore cached blocker outputs and Ditto scores; rebuild from scratch.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level (default: INFO).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.domain and not args.all:
        parser.error("Specify --domain <name> or --all")

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    targets = list(SPECS) if args.all else [args.domain]
    for d in targets:
        spec = SPECS[d]
        build_domain(
            spec,
            llm_model=args.llm_model,
            no_llm=args.no_llm,
            refresh=args.refresh,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
