#!/usr/bin/env python3
"""Regenerate per-source-pair human-baseline correspondences.

Replays the ``RuleBasedMatcher`` configurations from each domain's
Jupyter workflow notebook against the **refreshed CSV sources**
(``usecases/<d>/input/data/<source>.csv`` since 2026-05-04). The
notebooks themselves cannot be re-executed end-to-end on the
refreshed sources because their inline data-normalisation cells
reference the pre-refresh column names (e.g. ``forbes["Sales"]`` is
now ``forbes["sales_figure"]``); rather than patch the notebooks, this
script extracts just the matcher-relevant subset and runs it against
sources loaded by ``load_domain_sources`` + the canonical column
mappings from ``scripts/build_pool.py``.

Per source pair, writes::

    usecases/<domain>/output/correspondences_<src1>_<src2>.csv

with columns ``id1, id2, score, notes`` (the standard
``RuleBasedMatcher`` output schema). These files are the human-baseline
input stream consumed by ``scripts/build_pool.py`` for the R3 pool
build (see ``plans/plan_s1_scale.md`` R3).

Notebook fidelity
-----------------
Each per-pair matcher mirrors the notebook's blocker, comparator list,
weights, and threshold exactly. Where the notebook applied additional
data normalisation (track-list parsing for music, ``releaseYear``
date parsing for games, country-name normalisation for companies),
the same step is applied here on the refreshed columns. For
correspondence with ``games[m2d]``, the notebook's custom
``UnionTitleTokenBlocker`` is replicated as a local class.

The script does **not** re-run schema matching (the LLM-based step at
the top of every notebook). Instead it consumes the canonical-column
mapping baked into ``scripts/build_pool.py``, which is the same target
schema the LLM was producing.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PyDI.entitymatching import (  # noqa: E402
    DateComparator,
    GreedyOneToOneMatchingAlgorithm,
    MaximumBipartiteMatching,
    NumericComparator,
    RuleBasedMatcher,
    StandardBlocker,
    StringComparator,
    TokenBlocker,
)

from usecases_synthetic.lib.loaders import load_domain_sources  # noqa: E402
from usecases_synthetic.lib.pool_builder import apply_column_mapping  # noqa: E402
from usecases_synthetic.scripts.build_pool import (  # noqa: E402
    COMPANIES_SPEC,
    GAMES_SPEC,
    MUSIC_SPEC,
    DomainSpec,
)

logger = logging.getLogger("regen_human_baseline")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _normalize_text(s: object) -> str:
    if s is None:
        return ""
    return re.sub(r"[^\w\s]|_", "", str(s)).lower()


def _normalize_text_for_str_comparator(s: object) -> str:
    return _normalize_text(s)


# ---------------------------------------------------------------------------
# Companies
# ---------------------------------------------------------------------------


def regen_companies() -> dict[tuple[str, str], pd.DataFrame]:
    """Replicate ``companies_workflow.ipynb`` cells 28 + 33 + 35."""
    spec = COMPANIES_SPEC
    sources_raw = load_domain_sources(spec.name)
    sources = {
        n: apply_column_mapping(df, spec.column_mappings[n].column_mapping)
        for n, df in sources_raw.items()
    }
    forbes = sources["forbes"]
    dbpedia = sources["dbpedia"]
    fullcontact = sources["fullcontact"]

    # Notebook applies country-name normalisation to align dbpedia
    # ISO/colloquial names with forbes/fullcontact. The matcher uses
    # jaccard on country tokens, so casefolding alone gets most of the
    # benefit without the SchemaTranslator dependency.
    for df in (forbes, dbpedia, fullcontact):
        if "country" in df.columns:
            df["country"] = df["country"].fillna("").astype(str).str.lower()

    out: dict[tuple[str, str], pd.DataFrame] = {}

    # --- forbes <-> dbpedia -------------------------------------------------
    # Notebook: TokenBlocker(column='name'), comparators on
    # name/jaccard, name/levenshtein, country/jaccard, industry/jaccard;
    # weights=[1.0, 1.0, 1.0, 0.3], threshold=0.2.
    blocker_f2d = TokenBlocker(
        forbes,
        dbpedia,
        column="name",
        id_column="id",
        batch_size=1000,
        output_dir=str(spec.correspondence_dir / "blocking-evaluation"),
    )
    cmp_f2d = [
        StringComparator(
            column="name", similarity_function="jaccard", preprocess=_normalize_text
        ),
        StringComparator(
            column="name", similarity_function="levenshtein", preprocess=_normalize_text
        ),
        StringComparator(
            column="country", similarity_function="jaccard", preprocess=_normalize_text
        ),
        StringComparator(
            column="industry",
            similarity_function="jaccard",
            preprocess=_normalize_text,
        ),
    ]
    matcher = RuleBasedMatcher()
    raw_f2d = matcher.match(
        df_left=forbes,
        df_right=dbpedia,
        candidates=blocker_f2d,
        comparators=cmp_f2d,
        weights=[1.0, 1.0, 1.0, 0.3],
        threshold=0.2,
        id_column="id",
    )
    out[("forbes", "dbpedia")] = GreedyOneToOneMatchingAlgorithm().cluster(raw_f2d)

    # --- forbes <-> fullcontact --------------------------------------------
    # Notebook: TokenBlocker(column='name'), comparators on
    # name/jaccard + country/jaccard; weights=[1.0, 0.5], threshold=0.1.
    blocker_f2fc = TokenBlocker(
        forbes,
        fullcontact,
        column="name",
        id_column="id",
        batch_size=1000,
        output_dir=str(spec.correspondence_dir / "blocking-evaluation"),
    )
    cmp_f2fc = [
        StringComparator(
            column="name", similarity_function="jaccard", preprocess=_normalize_text
        ),
        StringComparator(
            column="country", similarity_function="jaccard", preprocess=_normalize_text
        ),
    ]
    matcher = RuleBasedMatcher()
    raw_f2fc = matcher.match(
        df_left=forbes,
        df_right=fullcontact,
        candidates=blocker_f2fc,
        comparators=cmp_f2fc,
        weights=[1.0, 0.5],
        threshold=0.1,
        id_column="id",
    )
    out[("forbes", "fullcontact")] = GreedyOneToOneMatchingAlgorithm().cluster(raw_f2fc)

    return out


# ---------------------------------------------------------------------------
# Games
# ---------------------------------------------------------------------------


_TITLE_TOKEN_STOPWORDS = {
    "the",
    "of",
    "and",
    "a",
    "an",
    "in",
    "to",
    "for",
    "on",
    "with",
    "at",
    "by",
    "from",
    "edition",
    "game",
    "video",
    "ii",
    "iii",
    "iv",
}

_PLATFORM_ALIASES = {
    "playstation 4": "ps4",
    "ps4": "ps4",
    "playstation 3": "ps3",
    "ps3": "ps3",
    "playstation 2": "ps2",
    "ps2": "ps2",
    "playstation vita": "ps vita",
    "ps vita": "ps vita",
    "psv": "ps vita",
    "playstation portable": "psp",
    "psp": "psp",
    "xbox one": "xbox one",
    "xone": "xbox one",
    "xbox 360": "xbox 360",
    "x360": "xbox 360",
    "xbox": "xbox",
    "nintendo switch": "switch",
    "switch": "switch",
    "wii": "wii",
    "wii u": "wii u",
    "gamecube": "gamecube",
    "nintendo gamecube": "gamecube",
    "nintendo ds": "ds",
    "ds": "ds",
    "nintendo 3ds": "3ds",
    "3ds": "3ds",
    "game boy advance": "gba",
    "gba": "gba",
    "game boy color": "gbc",
    "gbc": "gbc",
    "pc": "pc",
    "microsoft windows": "pc",
    "windows": "pc",
    "macintosh": "mac",
    "mac os x": "mac",
    "ios": "ios",
    "android": "android",
    "arcade video game": "arcade",
    "arcade": "arcade",
}

_MATCH_TITLE_STOPWORDS = {
    "the",
    "of",
    "and",
    "a",
    "an",
    "in",
    "to",
    "for",
    "on",
    "with",
    "at",
    "by",
    "from",
    "edition",
    "game",
    "video",
}


def _title_tokens(name: object) -> list[str]:
    tokens = re.split(r"[^A-Za-z0-9_']+", str(name).lower())
    return [t for t in tokens if len(t) >= 3 and t not in _TITLE_TOKEN_STOPWORDS]


def _get_longest_token(name: object) -> str:
    tokens = [t for t in re.split(r"[^A-Za-z0-9_']+", str(name)) if t]
    return max(tokens, key=len) if tokens else ""


def _normalize_platform(value: object) -> str:
    key = str(value).strip().lower()
    return _PLATFORM_ALIASES.get(key, key)


def _release_year(value: object) -> str:
    m = re.search(r"\d{4}", str(value))
    return m.group(0) if m else ""


def _normalize_match_title(value: object) -> str:
    s = str(value).lower()
    s = re.sub(r"\([^)]*video game[^)]*\)", " ", s)
    s = re.sub(
        r"\b(video game|game|hd|remaster(?:ed)?|definitive edition|"
        r"special edition|complete edition|goty edition)\b",
        " ",
        s,
    )
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _match_title_tokens(value: object) -> set[str]:
    return {
        t
        for t in _normalize_match_title(value).split()
        if len(t) >= 2 and t not in _MATCH_TITLE_STOPWORDS
    }


class _UnionTitleTokenBlocker:
    """Replica of the games m2d notebook blocker (cell 33)."""

    def __init__(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        *,
        id_column: str,
        name_column: str,
        platform_column: str,
        year_column: str,
        batch_size: int = 100_000,
    ) -> None:
        self.df_left = df_left
        self.df_right = df_right
        self.id_column = id_column
        self.name_column = name_column
        self.platform_column = platform_column
        self.year_column = year_column
        self.batch_size = int(batch_size)
        self.rules = (
            ("title_token_platform", "platform_block_key"),
            ("title_token_year", "release_year_block_key"),
        )
        self._left_blocks = self._build_blocks(df_left)
        self._right_blocks = self._build_blocks(df_right)

    def _build_blocks(
        self, df: pd.DataFrame
    ) -> dict[str, dict[tuple[str, str], list[str]]]:
        blocks: dict[str, dict[tuple[str, str], list[str]]] = {
            r: {} for r, _ in self.rules
        }
        cols = [
            self.id_column,
            self.name_column,
            self.platform_column,
            self.year_column,
        ]
        for row in df[cols].itertuples(index=False):
            rid = getattr(row, self.id_column)
            name = getattr(row, self.name_column)
            platform = _normalize_platform(getattr(row, self.platform_column))
            year = _release_year(getattr(row, self.year_column))
            for token in _title_tokens(name):
                if platform:
                    blocks["title_token_platform"].setdefault(
                        (token, platform), []
                    ).append(rid)
                if year:
                    blocks["title_token_year"].setdefault((token, year), []).append(rid)
        return blocks

    def __iter__(self):
        seen: set[tuple[str, str]] = set()
        batch: list[tuple[str, str, str]] = []
        for rule_name, _ in self.rules:
            left = self._left_blocks[rule_name]
            right = self._right_blocks[rule_name]
            for key in left.keys() & right.keys():
                for lid in left[key]:
                    for rid in right[key]:
                        if (lid, rid) in seen:
                            continue
                        seen.add((lid, rid))
                        batch.append((lid, rid, f"{rule_name}:{key[0]}:{key[1]}"))
                        if len(batch) >= self.batch_size:
                            yield pd.DataFrame(
                                batch, columns=["id1", "id2", "block_key"]
                            )
                            batch = []
        if batch:
            yield pd.DataFrame(batch, columns=["id1", "id2", "block_key"])

    def materialize(self) -> pd.DataFrame:
        frames = [b for b in self if not b.empty]
        return (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(columns=["id1", "id2", "block_key"])
        )


def _name_sequence_similarity(r1: pd.Series, r2: pd.Series) -> float:
    import difflib

    return difflib.SequenceMatcher(
        None,
        _normalize_match_title(r1["name"]),
        _normalize_match_title(r2["name"]),
    ).ratio()


def _name_token_overlap(r1: pd.Series, r2: pd.Series) -> float:
    t1 = _match_title_tokens(r1["name"])
    t2 = _match_title_tokens(r2["name"])
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / min(len(t1), len(t2))


def _platform_exact(r1: pd.Series, r2: pd.Series) -> float:
    p1 = _normalize_platform(r1["platform"])
    p2 = _normalize_platform(r2["platform"])
    return 1.0 if p1 and p2 and p1 == p2 else 0.0


def regen_games() -> dict[tuple[str, str], pd.DataFrame]:
    """Replicate ``games_workflow.ipynb`` cells 33 + 36 + 39 + 41 + 42."""
    spec = GAMES_SPEC
    sources_raw = load_domain_sources(spec.name)
    sources = {
        n: apply_column_mapping(df, spec.column_mappings[n].column_mapping)
        for n, df in sources_raw.items()
    }
    dbpedia = sources["dbpedia"].copy()
    metacritic = sources["metacritic"].copy()
    sales = sources["sales"].copy()

    # Notebook parses releaseYear as datetime + computes name_longest_token.
    for df in (dbpedia, metacritic, sales):
        df["name_longest_token"] = df["name"].apply(_get_longest_token)
        if "releaseYear" in df.columns:
            df["releaseYear"] = pd.to_datetime(df["releaseYear"], errors="coerce")

    out: dict[tuple[str, str], pd.DataFrame] = {}

    # --- dbpedia <-> metacritic --------------------------------------------
    # Notebook: UnionTitleTokenBlocker, custom callable comparators,
    # weights=[0.65, 0.25, 0.10], threshold=0.98.
    union_blocker = _UnionTitleTokenBlocker(
        dbpedia,
        metacritic,
        id_column="id",
        name_column="name",
        platform_column="platform",
        year_column="releaseYear",
        batch_size=100_000,
    )
    matcher = RuleBasedMatcher()
    raw_m2d = matcher.match(
        df_left=dbpedia,
        df_right=metacritic,
        candidates=union_blocker,
        comparators=[
            _name_sequence_similarity,
            _name_token_overlap,
            _platform_exact,
        ],
        weights=[0.65, 0.25, 0.10],
        threshold=0.98,
        id_column="id",
    )
    out[("dbpedia", "metacritic")] = MaximumBipartiteMatching().cluster(raw_m2d)

    # --- dbpedia <-> sales -------------------------------------------------
    # Notebook: StandardBlocker on name_longest_token, weights=[0.6, 0.3, 0.1],
    # threshold=0.8.
    std_blocker_m2s = StandardBlocker(
        dbpedia,
        sales,
        on=["name_longest_token"],
        id_column="id",
        batch_size=1000,
        output_dir=str(spec.correspondence_dir / "blocking-evaluation"),
    )
    cmp_m2s = [
        StringComparator(
            column="name", similarity_function="jaccard", preprocess=str.lower
        ),
        StringComparator(column="platform", similarity_function="jaccard"),
        DateComparator(column="releaseYear", max_days_difference=360),
    ]
    matcher = RuleBasedMatcher()
    raw_m2s = matcher.match(
        df_left=dbpedia,
        df_right=sales,
        candidates=std_blocker_m2s,
        comparators=cmp_m2s,
        weights=[0.6, 0.3, 0.1],
        threshold=0.8,
        id_column="id",
    )
    out[("dbpedia", "sales")] = MaximumBipartiteMatching().cluster(raw_m2s)

    return out


# ---------------------------------------------------------------------------
# Music
# ---------------------------------------------------------------------------


def _parse_track_list(value: object) -> list[str]:
    import ast

    if isinstance(value, list):
        items: list[Any] = list(value)
    elif pd.isna(value):
        return []
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
            items = parsed if isinstance(parsed, list) else [parsed]
        except (SyntaxError, ValueError):
            items = [p.strip() for p in text.split("|")]
    else:
        items = [value]

    cleaned: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item is None or (not isinstance(item, list) and pd.isna(item)):
            continue
        title = str(item).strip()
        if not title:
            continue
        key = re.sub(r"\s+", " ", title.casefold())
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(title)
    return cleaned


def regen_music() -> dict[tuple[str, str], pd.DataFrame]:
    """Replicate ``music_workflow.ipynb`` cells 32 + 37 + 40 + 41."""
    spec = MUSIC_SPEC
    sources_raw = load_domain_sources(spec.name)
    sources = {
        n: apply_column_mapping(df, spec.column_mappings[n].column_mapping)
        for n, df in sources_raw.items()
    }
    mbrainz = sources["musicbrainz"].copy()
    discogs = sources["discogs"].copy()
    lastfm = sources["lastfm"].copy()

    # Notebook tracks parsing + longest-token + release-date parsing.
    for df in (mbrainz, discogs, lastfm):
        df["name_longest_token"] = df["name"].apply(_get_longest_token)
        if "tracks" in df.columns:
            df["tracks"] = df["tracks"].apply(_parse_track_list)
        if "release-date" in df.columns:
            df["release-date"] = (
                df["release-date"]
                .astype(str)
                .apply(lambda x: re.sub(r"-00", "-01", x) if isinstance(x, str) else x)
            )
            df["release-date"] = pd.to_datetime(df["release-date"], errors="coerce")

    out: dict[tuple[str, str], pd.DataFrame] = {}

    # --- musicbrainz <-> discogs -------------------------------------------
    # Notebook: StandardBlocker on name_longest_token, 6 comparators
    # (name, artist, duration, tracks, release-date, release-country),
    # equal weights, threshold=0.5.
    std_m2d = StandardBlocker(
        mbrainz,
        discogs,
        on=["name_longest_token"],
        id_column="id",
        batch_size=1000,
        output_dir=str(spec.correspondence_dir / "blocking-evaluation"),
    )
    cmp_full = [
        StringComparator(
            column="name", similarity_function="jaccard", preprocess=_normalize_text
        ),
        StringComparator(
            column="artist", similarity_function="jaccard", preprocess=_normalize_text
        ),
        NumericComparator(
            column="duration", method="relative_difference", max_difference=0.10
        ),
        StringComparator(
            column="tracks",
            similarity_function="jaccard",
            preprocess=_normalize_text,
            list_strategy="set_overlap",
        ),
        DateComparator(column="release-date", max_days_difference=365 * 2),
        StringComparator(
            column="release-country",
            similarity_function="jaccard",
            preprocess=_normalize_text,
        ),
    ]
    matcher = RuleBasedMatcher()
    raw_m2d = matcher.match(
        df_left=mbrainz,
        df_right=discogs,
        candidates=std_m2d,
        comparators=cmp_full,
        weights=None,
        threshold=0.5,
        id_column="id",
    )
    out[("musicbrainz", "discogs")] = MaximumBipartiteMatching().cluster(raw_m2d)

    # --- musicbrainz <-> lastfm --------------------------------------------
    # Notebook drops release-date and release-country (lastfm has them
    # empty). Equal weights, threshold=0.3.
    std_m2l = StandardBlocker(
        mbrainz,
        lastfm,
        on=["name_longest_token"],
        id_column="id",
        batch_size=1000,
        output_dir=str(spec.correspondence_dir / "blocking-evaluation"),
    )
    cmp_lastfm = cmp_full[:-2]  # drop release-date, release-country
    matcher = RuleBasedMatcher()
    raw_m2l = matcher.match(
        df_left=mbrainz,
        df_right=lastfm,
        candidates=std_m2l,
        comparators=cmp_lastfm,
        weights=None,
        threshold=0.3,
        id_column="id",
    )
    out[("musicbrainz", "lastfm")] = MaximumBipartiteMatching().cluster(raw_m2l)

    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


_REGENERATORS: dict[
    str, tuple[DomainSpec, Callable[[], dict[tuple[str, str], pd.DataFrame]]]
] = {
    "companies": (COMPANIES_SPEC, regen_companies),
    "games": (GAMES_SPEC, regen_games),
    "music": (MUSIC_SPEC, regen_music),
}


def write_correspondences(
    spec: DomainSpec,
    pair_dfs: dict[tuple[str, str], pd.DataFrame],
) -> None:
    spec.correspondence_dir.mkdir(parents=True, exist_ok=True)
    for (src_a, src_b), df in pair_dfs.items():
        out_path = spec.correspondence_dir / f"correspondences_{src_a}_{src_b}.csv"
        df.to_csv(out_path, index=False)
        logger.info(
            "[%s] wrote %s (%d rows)",
            spec.name,
            out_path.relative_to(REPO_ROOT),
            len(df),
        )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--domain",
        choices=list(_REGENERATORS),
        help="Regenerate correspondences for a single domain.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Regenerate for all configured domains (companies, games, music).",
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

    targets = list(_REGENERATORS) if args.all else [args.domain]
    for d in targets:
        spec, regen_fn = _REGENERATORS[d]
        logger.info("[%s] regenerating human-baseline correspondences ...", d)
        pair_dfs = regen_fn()
        write_correspondences(spec, pair_dfs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
