"""LLM non-corner refill for Knob 02 — step 4i drop-corner branch.

Implements the refill side of the new K2 dispatch branch at
``baseline_ratio > target_ratio + tol``: for every entity dropped by the
drop-corner-touching operator, generate a synthetic non-corner entity
that occupies a distinct semantic niche from the remaining canonical
set. The refill is 1-for-1 so the per-source row count stays stable.

The hygiene contract mirrors :mod:`entity_interpolation`:

- Pinned ``non_corner_prompt_version`` (default ``v1``) and ``model_id``.
- ``temperature=0`` at the caller.
- Responses cached at
  ``usecases_synthetic/cache/knob_02_non_corner/<domain>/`` — a separate
  namespace from the near-twin interpolation cache so the two prompts
  don't collide.
- Contamination guard: synthesised primary label must not collide with
  any existing canonical entity's normalised primary label.

Strict cache mode (``strict_cache=True``) raises :class:`LLMCacheMiss`
on a miss instead of invoking the API.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import pandas as pd

from .llm_cache import LLMCache, LLMCacheMiss
from .niche_metrics import normalize_label

logger = logging.getLogger(__name__)


@dataclass
class NonCornerEntity:
    """An LLM-generated non-corner entity (synthetic refill).

    Parameters
    ----------
    entity_id : str
        Synthetic stable identifier (``<domain>_noncorner_<idx>``).
    reference_ids : list of str
        Canonical entity IDs used as the "be dissimilar to" anchor set.
    attributes : dict
        Column → value mapping for the synthetic entity.
    source_placements : list of str
        Ordered source names this entity is injected into.
    cache_path : str
        Relative path of the cached LLM payload.
    contamination_check_status : str
        ``"passed"`` when no normalised-label collision with a real
        entity was detected.
    """

    entity_id: str
    reference_ids: list[str]
    attributes: dict[str, Any]
    source_placements: list[str]
    cache_path: str
    contamination_check_status: str


# ---------------------------------------------------------------------------
# Reference selection
# ---------------------------------------------------------------------------


def select_reference_anchor(
    survivor_indices: Sequence[int],
    densities,
    *,
    k: int,
    rng,
    pool_multiplier: int = 4,
) -> list[int]:
    """Pick *k* low-density entities from the survivors as the anchor.

    Used by the K2 drop-corner refill path: the refilled entity should
    be dissimilar to a representative slice of the *remaining* canonical
    set after drops. The anchor is sampled from the **low-density
    prefix** of the survivor pool — the bottom
    ``min(len(survivors), k * pool_multiplier)`` survivors by density.
    Within that low-density prefix, *rng* draws *k* without replacement,
    so distinct callers (different ``spawn_sub_rng`` seeds) get
    distinct anchor combinations. This diversifies the per-call cache
    key in :func:`refill_non_corner_entity` — pre-2026-05-28 the
    function ignored *rng* and returned the deterministic bottom *k*,
    which collapsed all per-drop refill cache lookups to one entry
    (~99% of refills then rejected as ``contamination_collision``).

    Parameters
    ----------
    survivor_indices : Sequence[int]
        Canonical-frame indices of the entities that remain after the
        planned drops.
    densities : list of EntityDensity
        Per-index density score (lower = more isolated).
    k : int
        Anchor size.
    rng : numpy.random.Generator
        Supplies stochasticity for the within-pool sample.
    pool_multiplier : int, default 4
        The pool is the bottom ``k * pool_multiplier`` survivors by
        density. Larger values trade "stay low-density" for more
        anchor variety (fewer cache collisions). Default 4 → C(20, 5)
        = 15504 distinct anchor combinations at the typical k=5.

    Returns
    -------
    list of int
        ``k`` survivor indices (or all of them when ``k`` exceeds the
        pool size), sorted so the anchor hash is permutation-invariant.
    """
    if not survivor_indices or k <= 0:
        return []
    ordered = sorted(
        survivor_indices,
        key=lambda i: (densities[i].density, i),
    )
    if len(ordered) <= k:
        return list(ordered)
    pool_size = min(len(ordered), max(k, k * pool_multiplier))
    pool = ordered[:pool_size]
    picks = rng.choice(len(pool), size=k, replace=False)
    return sorted(int(pool[i]) for i in picks)


# ---------------------------------------------------------------------------
# Contamination check
# ---------------------------------------------------------------------------


def contamination_check(
    entity: dict[str, Any],
    *,
    primary_column: str,
    reference_labels: set[str],
) -> str:
    """Return ``"passed"`` or a reason string describing the failure.

    Mirrors :func:`entity_interpolation.contamination_check`: refuse a
    synthetic entity whose normalised primary label is empty or collides
    with any existing canonical entity's primary label.
    """
    label = str(entity.get(primary_column, "") or "")
    norm = normalize_label(label)
    if not norm:
        return "empty_primary_label"
    if norm in reference_labels:
        return "collision_with_real_entity"
    return "passed"


# ---------------------------------------------------------------------------
# Cache key derivation
# ---------------------------------------------------------------------------


def reference_anchor_hash(reference_ids: Sequence[str]) -> str:
    """Stable SHA-256 over a sorted tuple of reference entity IDs.

    Used to key the non-corner refill cache. Same algorithm shape as
    :func:`entity_interpolation.parent_pair_hash` so the two caches
    remain comparable; namespacing is via the surrounding ``cache_dir``.
    """
    payload = "|".join(sorted(reference_ids))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Core refill call
# ---------------------------------------------------------------------------


LLMNonCornerFn = Callable[[str, list[dict[str, Any]]], dict[str, Any]]


def refill_non_corner_entity(
    *,
    reference_rows: list[pd.Series],
    primary_column: str,
    schema_columns: list[str],
    domain: str,
    prompt_template: str,
    llm_cache: LLMCache,
    api_client: LLMNonCornerFn | None = None,
    reference_labels: set[str],
    source_placements: list[str],
    entity_id: str,
    strict_cache: bool = False,
    rejection_log: dict[str, int] | None = None,
) -> NonCornerEntity | None:
    """Produce a single synthetic non-corner entity.

    Cache-first: if a payload for the reference anchor exists in
    *llm_cache*, its ``result`` is reused. Otherwise *api_client* is
    invoked on a fresh call.

    Result is gated by schema validity + :func:`contamination_check`.

    Parameters
    ----------
    reference_rows : list of pandas.Series
        K low-density survivor rows defining the "be dissimilar to"
        anchor set.
    primary_column : str
        Column carrying the primary label.
    schema_columns : list of str
        Columns the synthetic entity must populate.
    domain : str
        Domain name (used in ``entity_id``).
    prompt_template : str
        Raw prompt string (non_corner_v1.txt or successor).
    llm_cache : LLMCache
        Shared LLM cache. Caller is responsible for pointing this at
        ``cache/knob_02_non_corner/<domain>/`` (namespace-distinct from
        the interpolation cache).
    api_client : callable or None
        ``(prompt_template, reference_records) -> dict``. Only called
        on cache miss when ``strict_cache`` is False.
    reference_labels : set of str
        Normalised real-entity labels for contamination checks.
    source_placements : list of str
        Target source names for the synthetic entity.
    entity_id : str
        Synthetic stable identifier.
    strict_cache : bool, default False
        When True, a cache miss raises :class:`LLMCacheMiss`.
    rejection_log : dict of str to int or None
        If provided, increment per-reason rejection counters in-place.
        Keys: ``"nondict_result"``, ``"empty_primary_label"``,
        ``"contamination_empty_primary_label"``,
        ``"contamination_collision_with_real_entity"``,
        ``"strict_cache_miss"``.

    Returns
    -------
    NonCornerEntity or None
        ``None`` when the synthetic entity fails a guardrail; caller
        skips and may retry with a different anchor.
    """
    reference_ids = [str(row.name) for row in reference_rows]
    anchor_hash = reference_anchor_hash(reference_ids)

    cache_source = f"{domain}_noncorner"
    cache_attr = f"anchor_{anchor_hash}"
    cache_value = "|".join(reference_ids)

    def _api_call() -> dict[str, Any]:
        if api_client is None:
            raise RuntimeError("api_client required on cache miss")
        reference_records = [
            {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}
            for row in reference_rows
        ]
        return api_client(prompt_template, reference_records)

    try:
        payload = llm_cache.call_or_cache(
            cache_source,
            cache_attr,
            cache_value,
            _api_call if not strict_cache else None,
            strict=strict_cache,
        )
    except LLMCacheMiss:
        if rejection_log is not None:
            rejection_log["strict_cache_miss"] = (
                rejection_log.get("strict_cache_miss", 0) + 1
            )
        raise

    raw_result = payload.get("result") or {}
    if not isinstance(raw_result, dict):
        logger.warning(
            "Non-corner refill result is not a dict for %s: %r",
            entity_id,
            raw_result,
        )
        if rejection_log is not None:
            rejection_log["nondict_result"] = rejection_log.get("nondict_result", 0) + 1
        return None

    attributes: dict[str, Any] = {}
    for col in schema_columns:
        attributes[col] = raw_result.get(col, "")

    primary_value = str(attributes.get(primary_column, "") or "").strip()
    if not primary_value:
        logger.info("Non-corner refill %s rejected: empty primary label", entity_id)
        if rejection_log is not None:
            rejection_log["empty_primary_label"] = (
                rejection_log.get("empty_primary_label", 0) + 1
            )
        return None

    contam_status = contamination_check(
        attributes,
        primary_column=primary_column,
        reference_labels=reference_labels,
    )
    if contam_status != "passed":
        logger.info("Non-corner refill %s rejected: %s", entity_id, contam_status)
        if rejection_log is not None:
            key = f"contamination_{contam_status}"
            rejection_log[key] = rejection_log.get(key, 0) + 1
        return None

    cache_path = f"{cache_source}/{cache_attr}.json"

    return NonCornerEntity(
        entity_id=entity_id,
        reference_ids=reference_ids,
        attributes=attributes,
        source_placements=list(source_placements),
        cache_path=cache_path,
        contamination_check_status=contam_status,
    )


# ---------------------------------------------------------------------------
# Deterministic fallback (used when no real LLM client is wired)
# ---------------------------------------------------------------------------


def default_api_client_from_attributes(
    *,
    schema_columns: list[str],
    primary_column: str,
    rng,
    salt: int = 0,
) -> LLMNonCornerFn:
    """Build a deterministic non-corner refill callable for tests.

    Generates a synthetic entity whose primary label is a SHA-256-derived
    token unlikely to collide with any real-entity label. Other columns
    are filled with anchor-distinct placeholder values. Mirror the
    ``entity_interpolation.default_api_client_from_attributes`` pattern
    so unit tests can exercise the dispatcher without an OpenAI key.
    """

    def _client(
        prompt_template: str, reference_records: list[dict[str, Any]]
    ) -> dict[str, Any]:
        # Anchor on a sorted-stable hash of reference primaries + salt
        # so the same (anchor, salt) yields the same fake entity.
        primaries = sorted(
            str(r.get(primary_column, "") or "") for r in reference_records
        )
        digest = hashlib.sha256(
            ("|".join(primaries) + f"|salt={salt}").encode("utf-8")
        ).hexdigest()[:12]
        fake: dict[str, Any] = {}
        for col in schema_columns:
            if col == primary_column:
                fake[col] = f"noncorner_{digest}"
            else:
                fake[col] = f"nc_{digest}_{col}"
        return fake

    return _client


# ---------------------------------------------------------------------------
# OpenAI-backed non-corner refill client
# ---------------------------------------------------------------------------


def build_openai_non_corner_client(
    *,
    model_id: str,
    temperature: float = 0.0,
    max_tokens: int | None = 2048,
) -> LLMNonCornerFn:
    """Build an OpenAI-backed :data:`LLMNonCornerFn` for K2 non-corner refill.

    Mirrors :func:`entity_interpolation.build_openai_interpolation_client`
    but substitutes the non-corner prompt's placeholders
    (``{reference_records_json}`` + ``{schema_columns_json}``) instead of
    the interpolation prompt's ``{parent_records_json}``. Using the
    interpolation client for non-corner refill is the 2026-05-28 bug: it
    hits ``KeyError`` on ``{reference_records_json}``, returns ``{}``,
    and every refill is rejected as ``empty_primary_label``.

    Parameters
    ----------
    model_id : str
        OpenAI chat model id (e.g. ``"gpt-5.4-mini"``).
    temperature : float, default 0.0
        Pinned to 0 for cache determinism.
    max_tokens : int or None, default 2048
        Per-call token cap. Reasoning models (gpt-5*) require ≥ 1024.

    Returns
    -------
    LLMNonCornerFn
        Callable matching ``Callable[[str, list[dict[str, Any]]],
        dict[str, Any]]`` suitable for the ``api_client`` argument of
        :func:`refill_non_corner_entity`.
    """
    from .llm_client import build_chat_openai

    chat = build_chat_openai(
        model=model_id,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    def _call(template: str, reference_records: list[dict[str, Any]]) -> dict[str, Any]:
        schema_columns: list[str] = []
        seen: set[str] = set()
        for record in reference_records:
            for key in record.keys():
                if key not in seen:
                    seen.add(key)
                    schema_columns.append(key)

        try:
            prompt = template.format(
                reference_records_json=json.dumps(reference_records, default=str),
                schema_columns_json=json.dumps(schema_columns),
            )
        except KeyError as exc:
            logger.warning(
                "K2 non-corner OpenAI client: prompt template missing "
                "placeholder %s; cannot synthesise refill",
                exc,
            )
            return {}

        try:
            response = chat.invoke(prompt)
        except Exception as exc:  # pragma: no cover - network/runtime errors
            logger.warning("K2 non-corner OpenAI client: invoke failed: %s", exc)
            return {}

        content = getattr(response, "content", response)
        if isinstance(content, list):
            content = "".join(
                str(p) if not isinstance(p, dict) else str(p.get("text", ""))
                for p in content
            )
        text = str(content).strip()

        if text.startswith("```"):
            text = text[3:]
            if text.lower().startswith("json"):
                text = text[4:]
            text = text.lstrip("\n")
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.warning(
                "K2 non-corner OpenAI client: JSON decode failed (%s); raw=%r",
                exc,
                text[:200],
            )
            return {}

        if not isinstance(parsed, dict):
            logger.warning(
                "K2 non-corner OpenAI client: response is not a dict: %r",
                parsed,
            )
            return {}
        return parsed

    return _call
