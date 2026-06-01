"""Per-domain committee YAML path resolution.

The companies committee rosters are checked in under their canonical
unsuffixed filenames in :mod:`config/committees/` (``sm_committee.yaml``,
``em_blocking_committee.yaml``, ``em_matching_committee.yaml``,
``fusion_committee.yaml``).  Per S10 of ``plans/plan_s1_scale.md``, the
EM blocking / EM matching / Fusion rosters are forked per domain so each
domain carries its own ``column_mapping``, ``trust_scores``,
per-attribute strategy lists, and Ditto checkpoint paths.  The SM
committee YAML stays a single shared file because it has no
domain-specific configuration (it consumes ``target_schema.json`` from
the variant bundle directly).

This module provides a single resolver function used by
``measure_baseline.py``, ``validate_variant.py``, and the per-domain
committee config tests so the path conventions are defined in exactly
one place.

Notes
-----
``companies-small`` aliases ``companies`` (the small clone shares the
canonical companies schema), so both resolve to the unsuffixed
companies YAMLs.
"""

from __future__ import annotations

from pathlib import Path

# Files that ship a per-domain fork.  ``sm_committee`` is intentionally
# omitted: SM committee members are domain-agnostic.
_PER_DOMAIN_BASE_NAMES: frozenset[str] = frozenset(
    {
        "em_blocking_committee",
        "em_matching_committee",
        "fusion_committee",
        # Normalization ships per-domain forks per the 2026-05-10
        # R5 Normalization sign-off (Pending #2). Each domain authors
        # its own per-attribute strategy block + taxonomy bindings.
        "normalization_committee",
    }
)

# Downsampled clones reuse the parent domain's per-domain committees
# because they share the source schema (per ``plans/plan_s1_scale.md``
# working notes, resolved 2026-04-29).
_COMMITTEE_DOMAIN_ALIASES: dict[str, str] = {
    "companies-small": "companies",
    "games-small": "games",
    "music-small": "music",
    "products-small": "products",
}


def canonical_committee_domain(domain: str) -> str:
    """Return the committee-domain key for ``domain``.

    Parameters
    ----------
    domain : str
        Domain name (e.g. ``"companies"``, ``"games"``, ``"companies-small"``).

    Returns
    -------
    str
        Canonical committee-domain key. ``companies-small`` resolves to
        ``companies``; everything else passes through unchanged.
    """
    return _COMMITTEE_DOMAIN_ALIASES.get(domain, domain)


_ALWAYS_PER_DOMAIN_BASE_NAMES: frozenset[str] = frozenset(
    {
        # Normalization rosters ship as per-domain files for every
        # domain (no unsuffixed canonical), so the resolver always
        # appends ``_<domain>``. EM/Fusion rosters use companies as the
        # canonical unsuffixed file and add ``_<domain>`` only for
        # non-companies domains.
        "normalization_committee",
    }
)


def resolve_committee_path(base_name: str, domain: str, *, committee_dir: Path) -> Path:
    """Resolve the on-disk committee YAML path for ``(base_name, domain)``.

    Parameters
    ----------
    base_name : str
        Filename stem without the ``.yaml`` extension (e.g.
        ``"em_blocking_committee"``).
    domain : str
        Domain name.
    committee_dir : Path
        Absolute path to ``usecases_synthetic/config/committees``.

    Returns
    -------
    Path
        Resolved YAML path. SM resolves to the unsuffixed file (single
        shared roster). EM blocking / EM matching / fusion resolve to
        the companies unsuffixed file when ``domain == "companies"`` and
        to ``<base_name>_<domain>.yaml`` otherwise. Normalization always
        resolves to the suffixed path (no companies-canonical file).
    """
    canonical = canonical_committee_domain(domain)
    if base_name in _ALWAYS_PER_DOMAIN_BASE_NAMES:
        return committee_dir / f"{base_name}_{canonical}.yaml"
    if base_name in _PER_DOMAIN_BASE_NAMES and canonical != "companies":
        return committee_dir / f"{base_name}_{canonical}.yaml"
    return committee_dir / f"{base_name}.yaml"


def per_domain_base_names() -> frozenset[str]:
    """Return the set of committee filenames that ship per-domain forks.

    Returns
    -------
    frozenset of str
        Filename stems (no extension) for which per-domain YAMLs exist.
    """
    return _PER_DOMAIN_BASE_NAMES
