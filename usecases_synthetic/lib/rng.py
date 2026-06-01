"""Deterministic RNG factory.

Follows ``knobs/cross_cutting.md`` §"Determinism Requirements": a single
``numpy.random.default_rng`` per ``(domain, variant, knob)`` tuple, seeded
via ``SeedSequence.spawn()``, so re-runs are bit-identical.
"""

from __future__ import annotations

import hashlib

import numpy as np


def _string_to_entropy(s: str) -> int:
    """Convert an arbitrary string to a stable integer for seeding.

    Uses SHA-256 truncated to 128 bits so the seed is deterministic and
    independent of Python's hash randomization (``PYTHONHASHSEED``).
    """
    digest = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(digest[:16], "big")


def make_rng(
    domain: str,
    variant: str,
    knob: int,
    master_seed: int = 42,
) -> np.random.Generator:
    """Create a deterministic RNG for a specific (domain, variant, knob) tuple.

    Parameters
    ----------
    domain : str
        Domain name (e.g. ``"companies"``).
    variant : str
        Variant / difficulty level (e.g. ``"easy"``, ``"medium"``, ``"hard"``).
    knob : int
        Knob number (1-10).
    master_seed : int, default 42
        Master seed from ``difficulty.yaml``.

    Returns
    -------
    numpy.random.Generator
        A seeded ``Generator`` instance.
    """
    label = f"{domain}:{variant}:knob{knob}"
    label_entropy = _string_to_entropy(label)
    ss = np.random.SeedSequence(master_seed, spawn_key=(label_entropy,))
    return np.random.default_rng(ss)


def cell_selection_uniform(
    domain: str,
    source: str,
    entity_id: str,
    attribute: str,
    knob: int,
    master_seed: int = 42,
) -> float:
    """Deterministic, level-independent uniform draw in ``[0, 1)`` for a cell.

    R10-A (``plans/plan_revision.md``): the rate-gated value knobs
    (K1 surface augmentation, K6 value noise) select a cell for
    perturbation when its uniform draw is below that level's target rate.
    Deriving the draw from cell identity *without* the difficulty level
    makes the selected set nest across levels (``easy ⊆ medium ⊆ hard``)
    by construction, because the per-level target rates are monotone
    non-decreasing.

    The draw is keyed on cell identity (``domain``, ``source``,
    ``entity_id``, ``attribute``) rather than row position, so nesting
    survives upstream knobs (K2 entity drop, K4 coverage skew) removing
    different rows at different levels: a cell present at two levels gets
    the same uniform at both. ``knob`` is part of the key so different
    knobs (1 vs 6) draw independent uniforms and do not select correlated
    cells.

    Parameters
    ----------
    domain : str
        Domain name (e.g. ``"products"``).
    source : str
        Source table name.
    entity_id : str
        Record identifier within the source.
    attribute : str
        Column name.
    knob : int
        Knob number; distinct knobs draw independent uniforms.
    master_seed : int, default 42
        Master seed from ``difficulty.yaml``.

    Returns
    -------
    float
        A deterministic value in ``[0.0, 1.0)``.
    """
    label = (
        f"{domain}|{source}|{entity_id}|{attribute}"
        f"|knob{knob}|select|seed{master_seed}"
    )
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def spawn_sub_rng(
    rng: np.random.Generator,
    label: str,
) -> np.random.Generator:
    """Spawn a child RNG from an existing generator for sub-knob delegation.

    Useful when a knob internally needs multiple independent streams
    (e.g. one per attribute or one per source).

    Parameters
    ----------
    rng : numpy.random.Generator
        Parent generator (typically from ``make_rng``).
    label : str
        Distinguishing label for the child stream.

    Returns
    -------
    numpy.random.Generator
        A child ``Generator`` seeded deterministically from the parent.
    """
    parent_state = rng.bit_generator.state
    parent_ss = parent_state.get("state", {}).get("s", {}).get("seed_seq")

    # Use the parent's internal state plus the label to derive a new seed
    label_entropy = _string_to_entropy(label)
    # Draw an integer from the parent to advance its state deterministically
    parent_draw = int(rng.integers(0, 2**63))
    child_ss = np.random.SeedSequence(parent_draw, spawn_key=(label_entropy,))
    return np.random.default_rng(child_ss)
