"""Protection set construction + closeness contract for the synthetic generation pipeline.

Implements ``knobs/cross_cutting.md`` §"Gold standard incompleteness and
pooling":

    expanded_positives = test_gold ∪ train_gold ∪ val_gold ∪ pooled_positives

The protection set constrains the generator (never the evaluator).
Protected entity IDs must not be dropped (K2 easy), used as distractor
seeds (K2 hard), or mined as hard negatives (K1/K2).

Closeness contract (Pending #5, 2026-05-06)
-------------------------------------------
For every fusion val + test (entity, attribute) cell, ≥1 surviving
record across all sources must remain "close enough" to the fusion
target value so a lenient fusion strategy can recover the truth.
"Close enough" replaces the prior exact-match guarantee. The thresholds
are locked in §"Closeness-metric specification" of plan_s1_scale.md and
implemented below via :func:`is_close_enough` + :class:`ToleranceSpec`.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from .domain_config import (
    POOLS_DIR,
    USECASES_DIR,
    data_root_for_domain,
    load_domain_config,
)
from .loaders import read_em_gold_csv
from .niche_metrics import _levenshtein_ratio, lexical_extended_jaccard

# ---------------------------------------------------------------------------
# Entity-ID protection sets
# ---------------------------------------------------------------------------


def _load_em_gold_ids(domain: str) -> set[str]:
    """Load all entity IDs from EM gold CSVs (train + val + test)."""
    root = data_root_for_domain(domain) or USECASES_DIR
    em_dir = root / domain / "input" / "entitymatching"
    ids: set[str] = set()

    for split in ("_train.csv", "_val.csv", "_test.csv"):
        for csv_path in sorted(em_dir.glob(f"*{split}")):
            df = read_em_gold_csv(csv_path)
            ids.update(df["id1"].astype(str))
            ids.update(df["id2"].astype(str))

    return ids


def _load_fusion_protected_ids(domain: str) -> set[str]:
    """Load entity IDs from the fusion validation + test set XMLs.

    Reads both fusion files declared by the domain config's
    ``fusion_files`` block (defaults: ``validation_set.xml`` and
    ``test_set.xml``) so that any entity referenced by either fusion
    split is protected.
    """
    cfg = load_domain_config(domain)
    ids: set[str] = set()

    for path in cfg.fusion_paths():
        if not path.exists():
            continue
        tree = ET.parse(path)
        root = tree.getroot()
        for id_elem in root.iter("id"):
            if id_elem.text:
                ids.add(id_elem.text.strip())

    return ids


# Back-compat alias (older callers in K3/K4 still import this name).
_load_fusion_gold_ids = _load_fusion_protected_ids


def _load_pooled_positive_ids(domain: str) -> set[str]:
    """Load entity IDs from the pooled positives CSV."""
    pool_path = POOLS_DIR / domain / "pooled_positives.csv"
    ids: set[str] = set()

    if not pool_path.exists():
        return ids

    df = pd.read_csv(pool_path)
    ids.update(df["id1"].astype(str))
    ids.update(df["id2"].astype(str))

    return ids


def build_expanded_positives(domain: str) -> set[str]:
    """Build the expanded positives protection set for a domain.

    Computes::

        expanded_positives = EM_gold_ids ∪ fusion_protected_ids ∪ pooled_positive_ids
    """
    em_ids = _load_em_gold_ids(domain)
    fusion_ids = _load_fusion_protected_ids(domain)
    pool_ids = _load_pooled_positive_ids(domain)
    return em_ids | fusion_ids | pool_ids


def build_drop_corner_protection_set(
    domain: str, protection_source: str = "gold"
) -> set[str]:
    """Protection set for K2's drop-corner-touching operator (step 4i).

    The drop-corner operator removes existing canonical entities to
    reduce the corner-pair ratio. It must respect a narrower protection
    set than ``build_expanded_positives``: the fusion val/test gold is
    the only universally-required protection (it is the fixed
    cross-level evaluation surface for fusion accuracy). EM gold
    positives are NOT protected here — step 4c / C11's
    ``regenerate_em_splits`` already handles the case where EM gold
    members get dropped (pruning Set 1 and corner-mining Set 2 from the
    surviving pool). Without this narrowing, ``pool_quality: live``
    domains (products) have EM gold coextensive with the full record
    set, so every drop candidate is "protected" and the operator
    becomes a noop.

    Parameters
    ----------
    domain : str
        Domain name (e.g. ``"products"``).
    protection_source : str
        ``"gold"`` (default): fusion val/test gold only. EM gold and
        pool members may be dropped. ``"silver"``: also includes every
        pool member, matching the C9 silver-standard "every cluster
        member is fusion-recoverable, therefore protected" semantics.
        On pool-live domains under ``"silver"`` the operator becomes a
        noop (every entity protected); the caller should expect that.

    Returns
    -------
    set of str
        Protected record IDs. Caller compares canonical entity members
        against this set.
    """
    fusion_ids = _load_fusion_protected_ids(domain)
    if protection_source == "silver":
        pool_ids = _load_pooled_positive_ids(domain)
        return fusion_ids | pool_ids
    return fusion_ids


def is_protected(entity_id: str, expanded_positives: set[str]) -> bool:
    """Check whether an entity ID is in the protection set."""
    return entity_id in expanded_positives


# ---------------------------------------------------------------------------
# Closeness contract — tolerance specification + per-cell test
# ---------------------------------------------------------------------------


_VALID_KINDS = (
    "continuous",
    "year",
    "date",
    "nominal",
    "long_string",
    "free_text",
    "list",
)


@dataclass(frozen=True)
class ToleranceSpec:
    """Per-attribute tolerance specification for the closeness contract.

    Parameters
    ----------
    kind : str
        One of ``continuous`` (numeric, relative band), ``year``
        (calendar year, ±1 absolute), ``date`` (calendar day delta ≤ N
        days), ``nominal`` (Levenshtein ratio on short strings),
        ``long_string`` (extended Jaccard on tokenised long strings),
        ``free_text`` (extended Jaccard, looser threshold), ``list``
        (extended Jaccard over flattened set tokens).
    threshold : float
        Numeric tolerance (relative for ``continuous``; absolute years
        for ``year``; absolute calendar-days for ``date``; ratio in
        ``[0, 1]`` for string kinds).
    inner_token_threshold : float
        Inner-Levenshtein gate for the extended-Jaccard kinds.
    """

    kind: str
    threshold: float
    inner_token_threshold: float = 0.8


_DEFAULT_TOLERANCE_BY_KIND: dict[str, ToleranceSpec] = {
    "continuous": ToleranceSpec(kind="continuous", threshold=0.03),
    "year": ToleranceSpec(kind="year", threshold=1.0),
    "date": ToleranceSpec(kind="date", threshold=1.0),
    "nominal": ToleranceSpec(kind="nominal", threshold=0.85),
    "long_string": ToleranceSpec(
        kind="long_string", threshold=0.6, inner_token_threshold=0.8
    ),
    "free_text": ToleranceSpec(
        kind="free_text", threshold=0.5, inner_token_threshold=0.8
    ),
    "list": ToleranceSpec(kind="list", threshold=0.5, inner_token_threshold=0.8),
}


# Per-domain canonical-attribute → kind map. Locked from K1/K5/K6 sign-offs
# (2026-05-07) in plan_s1_scale.md. Per-attribute overrides may be authored
# in ``config/knob_06_noise/<domain>.yaml::fusion_protection_tolerance``.
_DEFAULT_KIND_BY_DOMAIN_ATTR: dict[str, dict[str, str]] = {
    "companies": {
        "name": "long_string",
        "country": "nominal",
        "city": "nominal",
        "industry": "nominal",
        "founded": "year",
        "keypeople": "long_string",
        "assets": "continuous",
        "revenue": "continuous",
    },
    "games": {
        "name": "long_string",
        "platform": "nominal",
        "ESRB": "nominal",
        "releaseYear": "year",
        "developer": "nominal",
        "publisher": "nominal",
        "genres": "list",
        "criticScore": "continuous",
        "userScore": "continuous",
        "globalSales": "continuous",
        "series": "nominal",
    },
    "music": {
        "name": "long_string",
        "artist": "long_string",
        "release-country": "nominal",
        "genre": "nominal",
        "release-date": "date",
        "duration": "continuous",
        "label": "nominal",
    },
    "products": {
        "title": "long_string",
        "brand": "nominal",
        "description": "free_text",
        "price": "continuous",
        "priceCurrency": "nominal",
    },
}


def kind_map_for_domain(domain: str) -> dict[str, str]:
    """Return the per-attribute kind map for ``domain``, alias-aware.

    Aliased domains (e.g. ``music-small`` with ``knob_config_alias: music``
    in its domain YAML) inherit the source domain's kind map.
    """
    from .domain_config import _resolve_knob_config_alias

    direct = _DEFAULT_KIND_BY_DOMAIN_ATTR.get(domain)
    if direct:
        return direct
    alias = _resolve_knob_config_alias(domain)
    if alias:
        return _DEFAULT_KIND_BY_DOMAIN_ATTR.get(alias, {})
    return {}


def fusion_cell_tolerance(
    domain: str,
    canonical_attribute: str,
    config_overrides: dict[str, dict[str, float | str]] | None = None,
) -> ToleranceSpec:
    """Resolve the tolerance spec for a (domain, canonical attribute) cell.

    Resolution order:

    1. ``config_overrides[canonical_attribute]`` — per-attribute override
       authored in ``config/knob_06_noise/<domain>.yaml`` under the
       ``fusion_protection_tolerance`` block. Each entry may carry ``kind``
       and/or ``threshold`` and/or ``inner_token_threshold``.
    2. The per-domain default kind from
       :data:`_DEFAULT_KIND_BY_DOMAIN_ATTR`, with default thresholds from
       :data:`_DEFAULT_TOLERANCE_BY_KIND`.
    3. Fallback to ``long_string`` (a safe, moderately strict default).

    Parameters
    ----------
    domain : str
        Domain name (``"companies"``, ``"games"``, ``"music"``).
    canonical_attribute : str
        Canonical attribute name as authored in
        ``config/domains/<domain>.yaml`` (e.g. ``"name"``, ``"country"``).
    config_overrides : dict or None
        Optional per-domain override block.

    Returns
    -------
    ToleranceSpec
    """
    domain_kinds = kind_map_for_domain(domain)
    default_kind = domain_kinds.get(canonical_attribute, "long_string")
    base = _DEFAULT_TOLERANCE_BY_KIND[default_kind]

    if not config_overrides:
        return base

    override = config_overrides.get(canonical_attribute)
    if not override:
        return base

    kind = str(override.get("kind", base.kind))
    if kind not in _VALID_KINDS:
        raise ValueError(
            f"Unknown tolerance kind {kind!r} for "
            f"{domain}.{canonical_attribute}; valid: {_VALID_KINDS}"
        )

    threshold = override.get("threshold", _DEFAULT_TOLERANCE_BY_KIND[kind].threshold)
    inner = override.get(
        "inner_token_threshold",
        _DEFAULT_TOLERANCE_BY_KIND[kind].inner_token_threshold,
    )
    return ToleranceSpec(
        kind=kind,
        threshold=float(threshold),
        inner_token_threshold=float(inner),
    )


# ---- Parsers used by is_close_enough --------------------------------------

_YEAR_RE = re.compile(r"\b(\d{4})\b")
_LIST_SPLIT_RE = re.compile(r"[,;|]|\s/\s|\s\|\s")


def _parse_float(value: str) -> float | None:
    s = value.strip()
    if not s:
        return None
    # Handle scientific notation, currency punctuation. Drop common
    # thousands separators / spaces / currency markers; keep decimal
    # point and exponent.
    s2 = re.sub(r"[\s,$€£¥₩]", "", s)
    try:
        return float(s2)
    except ValueError:
        return None


def _parse_year(value: str) -> int | None:
    s = value.strip()
    if not s:
        return None
    # Try ISO-style first.
    m = re.match(r"^(\d{4})", s)
    if m:
        try:
            year = int(m.group(1))
            if 1500 <= year <= 2200:
                return year
        except ValueError:
            return None
    # Fallback: look for any 4-digit run.
    m = _YEAR_RE.search(s)
    if m:
        try:
            year = int(m.group(1))
            if 1500 <= year <= 2200:
                return year
        except ValueError:
            return None
    return None


_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y/%m/%d",
    "%d.%m.%Y",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d-%m-%Y",
)


def _parse_date(value: str) -> datetime | None:
    s = value.strip()
    if not s:
        return None
    # Strip trailing timezone offsets like ``+01:00`` / ``Z``.
    s2 = re.sub(r"([+-]\d{2}:?\d{2}|Z)\s*$", "", s)
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s2, fmt)
        except ValueError:
            continue
    return None


def _split_list_tokens(value: str) -> list[str]:
    """Split a list-like cell into tokens.

    Handles ``"['a', 'b']"`` JSON-ish forms, comma / semicolon /
    pipe-separated forms, and falls back to whitespace tokenisation.
    """
    s = value.strip()
    if not s:
        return []
    # JSON-ish list: ``['a', 'b']`` → strip brackets + quotes + commas.
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1]
        parts = [p.strip().strip("'\"") for p in inner.split(",")]
        return [p for p in parts if p]
    parts = _LIST_SPLIT_RE.split(s)
    return [p.strip() for p in parts if p.strip()]


# ---- Closeness predicate --------------------------------------------------


def is_close_enough(
    value: str | float | int | None,
    target: str | float | int | None,
    tolerance: ToleranceSpec,
) -> bool:
    """Test whether *value* is close enough to *target* under *tolerance*.

    Returns False on any unparseable input under numeric / date kinds —
    the closeness contract requires evidence of closeness, not absence
    of evidence to the contrary.

    Parameters
    ----------
    value : str, float, int, or None
        Source-side cell value.
    target : str, float, int, or None
        Fusion target value.
    tolerance : ToleranceSpec
        Resolved tolerance for this attribute.
    """
    if value is None or target is None:
        return False
    if isinstance(value, float) and pd.isna(value):
        return False
    if isinstance(target, float) and pd.isna(target):
        return False

    v = str(value).strip()
    t = str(target).strip()
    if not t:
        # If target has no value, contract is vacuously true.
        return True
    if not v:
        return False

    kind = tolerance.kind

    if kind == "continuous":
        fv = _parse_float(v)
        ft = _parse_float(t)
        if fv is None or ft is None:
            return False
        denom = max(abs(ft), 1e-9)
        return abs(fv - ft) / denom <= tolerance.threshold

    if kind == "year":
        yv = _parse_year(v)
        yt = _parse_year(t)
        if yv is None or yt is None:
            return False
        return abs(yv - yt) <= int(tolerance.threshold)

    if kind == "date":
        dv = _parse_date(v)
        dt = _parse_date(t)
        if dv is None or dt is None:
            # Fall back to year-level if either side parses only as year.
            yv = _parse_year(v)
            yt = _parse_year(t)
            if yv is not None and yt is not None:
                # 1-day tolerance translates loosely to "same year" here.
                return yv == yt
            return False
        return abs((dv - dt).days) <= int(tolerance.threshold)

    if kind == "nominal":
        return _levenshtein_ratio(v.casefold(), t.casefold()) >= tolerance.threshold

    if kind in ("long_string", "free_text"):
        sim = lexical_extended_jaccard(
            v, t, inner_token_threshold=tolerance.inner_token_threshold
        )
        return sim >= tolerance.threshold

    if kind == "list":
        toks_v = _split_list_tokens(v)
        toks_t = _split_list_tokens(t)
        if not toks_v and not toks_t:
            return True
        if not toks_v or not toks_t:
            return False
        # Flatten back to a string so lexical_extended_jaccard can tokenise.
        sim = lexical_extended_jaccard(
            " ".join(toks_v),
            " ".join(toks_t),
            inner_token_threshold=tolerance.inner_token_threshold,
        )
        return sim >= tolerance.threshold

    raise ValueError(f"Unknown tolerance kind: {kind!r}")


# ---------------------------------------------------------------------------
# Fusion target value loader
# ---------------------------------------------------------------------------


def load_fusion_target_values(domain: str) -> dict[str, dict[str, list[str]]]:
    """Load per-(entity, attribute) target values from fusion val + test.

    Reads both fusion files declared by the domain config's
    ``fusion_files`` block (defaults: ``validation_set.xml`` and
    ``test_set.xml``). For multi-valued attributes (e.g. games'
    ``<genres><genre>...</genre></genres>``, companies'
    ``<keypeople><name>...</name></keypeople>``), the inner text values
    are aggregated into a list. For scalar attributes the list contains
    the single text value. Empty / null cells are skipped.

    Parameters
    ----------
    domain : str
        Domain name.

    Returns
    -------
    dict
        ``{entity_id: {attribute: [value, ...]}}``. Test-set wins on
        conflicting entity IDs (val is read first, then test overrides).
    """
    cfg = load_domain_config(domain)
    out: dict[str, dict[str, list[str]]] = {}

    for path in cfg.fusion_paths():
        if not path.exists():
            continue
        tree = ET.parse(path)
        root = tree.getroot()
        for entity_elem in root:
            id_elem = entity_elem.find("id")
            if id_elem is None or not id_elem.text:
                continue
            eid = id_elem.text.strip()
            attrs: dict[str, list[str]] = {}
            for child in entity_elem:
                if child.tag == "id":
                    continue
                # Multi-valued (has child elements).
                inner_values: list[str] = []
                if list(child):
                    for sub in child:
                        if sub.text and sub.text.strip():
                            inner_values.append(sub.text.strip())
                elif child.text and child.text.strip():
                    inner_values.append(child.text.strip())
                if inner_values:
                    attrs[child.tag] = inner_values
            if attrs:
                out[eid] = attrs

    return out


# ---------------------------------------------------------------------------
# Per-cell closeness gate (used by K1/K3/K5/K6/K10 dispatchers)
# ---------------------------------------------------------------------------


def cell_has_close_survivor(
    target_values: list[str],
    surviving_values: Iterable[str | None],
    tolerance: ToleranceSpec,
) -> bool:
    """Return True if any surviving value is close to any target value.

    Used as the per-cell closeness gate: after a candidate mutation, the
    caller passes the post-commit values for all sources mapped to the
    same canonical attribute (the candidate value for the source being
    mutated, the current values for others). This function returns
    True iff ≥1 of those values is within tolerance of ≥1 target value.

    Parameters
    ----------
    target_values : list[str]
        Fusion target value(s) for this (entity, canonical attribute).
        For multi-valued list attributes, callers may pass the joined
        list (recommended) or one entry per gold value (less strict).
    surviving_values : iterable of str or None
        Post-commit values across all sources mapped to the same
        canonical attribute. Empty / None entries are ignored.
    tolerance : ToleranceSpec
        Resolved tolerance for the canonical attribute.

    Returns
    -------
    bool
    """
    if not target_values:
        # No target authored → contract is vacuously true.
        return True
    for sv in surviving_values:
        if sv is None:
            continue
        if isinstance(sv, float) and pd.isna(sv):
            continue
        s = str(sv).strip()
        if not s:
            continue
        for tv in target_values:
            if is_close_enough(s, tv, tolerance):
                return True
    return False
