"""Per-domain value normalization specs.

Shared module for the value-level transformations the human-baseline
notebook applies before matching / blocking. Two call sites consume
these:

1. ``scripts/regen_human_baseline.py`` — already had inline copies of
   these helpers; this module is the canonical home and ``regen_human_baseline``
   should be migrated to import from here. (No behaviour change there;
   the local definitions are byte-equivalent.)
2. ``scripts/ditto/prepare_em_training_data.py`` — uses the per-domain
   normaliser to canonicalise dataframe values **before** Ditto's COL/VAL
   serialisation, so the Ditto training data and inference inputs share
   the same value distribution as the human-baseline.

Per ``plans/plan_revision_step4g_findings.md`` §2 + the 2026-05-26
follow-up discussion: the committee Ditto on games (dbpedia_sales)
over-predicts at precision 0.43 / recall 0.94 — hypothesised to be at
least partly driven by raw-platform mismatch (e.g. ``"PS3"`` vs
``"Playstation 3"``). The R2 Ditto checkpoints were trained on raw
values, so an A/B retrain on normalised data is the empirical test.

Scope today: **games only**. Companies + music + products will be added
when the games A/B result lands. Each domain's normalisation map is
specific to its notebook; cross-domain re-use of the helper is the
intent, not a uniform spec.
"""

from __future__ import annotations

import re
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Games
# ---------------------------------------------------------------------------

GAMES_PLATFORM_ALIASES: dict[str, str] = {
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


def normalize_games_platform(value: Any) -> str:
    """Canonicalise a games platform value per the notebook's alias map.

    Empty / NaN passes through as ``""`` so downstream COL/VAL
    serialisation skips the field cleanly. Unknown platforms pass
    through lowercased and stripped (best-effort canonicalisation when
    the alias map doesn't cover a value).
    """
    if value is None:
        return ""
    if isinstance(value, float):
        # NaN check without importing pandas — NaN != NaN
        if value != value:
            return ""
    key = str(value).strip().lower()
    if not key:
        return ""
    return GAMES_PLATFORM_ALIASES.get(key, key)


def normalize_games_title(value: Any) -> str:
    """Canonicalise a games title per the notebook's title-cleanup regex.

    Mirrors ``regen_human_baseline._normalize_match_title``: lowercases,
    strips ``"(... video game ...)"`` parens, strips edition / remaster
    suffixes, collapses non-alphanumerics to single spaces.
    """
    if value is None:
        return ""
    if isinstance(value, float):
        if value != value:
            return ""
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


# ---------------------------------------------------------------------------
# Per-domain dispatcher
# ---------------------------------------------------------------------------

# Keys match the canonical schema field names (post-attribute_mapping in
# the knob 02 YAML), not the raw source-column names.
_DOMAIN_NORMALIZERS: dict[str, dict[str, Callable[[Any], Any]]] = {
    "games": {
        "platform": normalize_games_platform,
        "name": normalize_games_title,
    },
}


def get_value_normalizer(domain: str) -> dict[str, Callable[[Any], Any]] | None:
    """Return the per-field value-normaliser map for ``domain``, or ``None``.

    The returned mapping is keyed by **canonical field name** (post
    column-mapping). ``None`` means "no normalisation configured" —
    callers should treat that as a no-op rather than an error.
    """
    return _DOMAIN_NORMALIZERS.get(domain)
