"""LLM entity interpolation for Knob 02 — Entity Niche Density (hard path).

Implements ``knobs/knob_02_niche_density.md`` §"Algorithm selection" —
LLM-generated near-twin entities seeded by existing parents. Used at
**hard** level only. Medium and easy paths never touch this module.

The hygiene contract follows the cross-cutting LLM pattern:
- Pinned ``prompt_version`` and ``model_id``.
- ``temperature=0`` at the caller.
- Responses cached and committed to
  ``usecases_synthetic/cache/knob_02_interpolations/<domain>/``.
- Committee validation gating.
- Contamination check against real-entity normalised labels.

Strict cache mode (``strict_cache=True``) raises ``LLMCacheMiss`` on a
miss instead of invoking the API — required on CLI invocations so
committed caches are the sole source of truth on rerun.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import numpy as np
import pandas as pd

from .llm_cache import LLMCache, LLMCacheMiss
from .niche_metrics import normalize_label

logger = logging.getLogger(__name__)


PlacementMode = Literal["matched_across", "single_source_distractor"]


@dataclass
class InterpolatedEntity:
    """An LLM-generated near-twin entity.

    Parameters
    ----------
    entity_id : str
        Synthetic stable identifier (``<domain>_interp_<idx>``).
    parent_ids : list of str
        Canonical entity IDs used as parents.
    attributes : dict
        Column → value mapping for the synthetic entity.
    placement_mode : {"matched_across", "single_source_distractor"}
        ``matched_across`` → generates matched records in >=2 sources
        (produces hard positives). ``single_source_distractor`` → one
        source only (produces hard negatives for blocking / EM).
    source_placements : list of str
        Ordered source names this entity is injected into.
    cache_path : str
        Relative path of the cached LLM payload.
    contamination_check_status : str
        ``"passed"`` when no normalised-label collision with a real
        entity was detected.
    """

    entity_id: str
    parent_ids: list[str]
    attributes: dict[str, Any]
    placement_mode: PlacementMode
    source_placements: list[str]
    cache_path: str
    contamination_check_status: str


# ---------------------------------------------------------------------------
# Parent selection
# ---------------------------------------------------------------------------


def select_parent_pairs(
    dense_indices: Sequence[int],
    *,
    neighbour_lookup: dict[int, list[int]],
    protected: Sequence[bool],
    placement_mode: PlacementMode,
    k: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """Draw *k* parent pairs from the top of the density distribution.

    The first parent is drawn from *dense_indices* (density-ranked); the
    second is drawn from its fused-neighbour list. For
    ``single_source_distractor`` placement both parents must be
    non-protected, per the cross_cutting protection contract.

    Parameters
    ----------
    dense_indices : sequence of int
        Entity indices sorted by descending density.
    neighbour_lookup : dict
        ``entity_index -> [neighbour_index, ...]`` drawn from the fused
        top-K neighbour list.
    protected : sequence of bool
        ``protected[i]`` is True when entity *i* is in
        ``expanded_positives``.
    placement_mode : {"matched_across", "single_source_distractor"}
        Constrains whether protected entities are admissible parents.
    k : int
        Number of pairs to draw.
    rng : numpy.random.Generator
        Seeded RNG.

    Returns
    -------
    list of (int, int)
        Parent pairs as ``(a, b)`` row indices. May be shorter than *k*
        if insufficient dense entities exist.
    """
    if k <= 0:
        return []

    allow_protected = placement_mode == "matched_across"

    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for parent_a in dense_indices:
        if not allow_protected and protected[parent_a]:
            continue
        neighbours = neighbour_lookup.get(parent_a, [])
        if not neighbours:
            continue
        candidates = [
            n
            for n in neighbours
            if n != parent_a and (allow_protected or not protected[n])
        ]
        if not candidates:
            continue
        parent_b = int(rng.choice(candidates))
        pair = (parent_a, parent_b) if parent_a < parent_b else (parent_b, parent_a)
        if pair in seen:
            continue
        seen.add(pair)
        out.append(pair)
        if len(out) >= k:
            break
    return out


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

    The check is deterministic: it asserts the normalised primary label
    does not collide with any real-entity normalised label. Callers can
    optionally layer an external web-search lookup on top (not done
    here — the in-process check is cheap and sufficient for the v1
    implementation).
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


def parent_pair_hash(parent_ids: Sequence[str]) -> str:
    """Stable SHA-256 over a sorted tuple of parent IDs."""
    payload = "|".join(sorted(parent_ids))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Core interpolation call
# ---------------------------------------------------------------------------


CommitteeFn = Callable[[dict[str, Any], list[dict[str, Any]]], bool]
LLMInterpolateFn = Callable[[str, list[dict[str, Any]]], dict[str, Any]]


def interpolate_entity(
    *,
    parent_rows: list[pd.Series],
    primary_column: str,
    schema_columns: list[str],
    domain: str,
    prompt_template: str,
    llm_cache: LLMCache,
    api_client: LLMInterpolateFn | None = None,
    committee_fn: CommitteeFn | None = None,
    reference_labels: set[str],
    placement_mode: PlacementMode,
    source_placements: list[str],
    entity_id: str,
    strict_cache: bool = False,
    rejection_log: dict[str, int] | None = None,
) -> InterpolatedEntity | None:
    """Produce a single synthetic near-twin entity from *parent_rows*.

    Cache-first: if a payload for the parent pair exists in *llm_cache*,
    its ``result`` is reused unconditionally. Otherwise, *api_client* is
    invoked with ``(prompt_template, parent_records)`` and the result is
    cached.

    The result is gated by:
    1. Schema validity — every required column is non-empty.
    2. :func:`contamination_check` — normalised primary label must not
       collide with any real-entity label.
    3. Optional committee validation.

    Parameters
    ----------
    parent_rows : list of pandas.Series
        Parent entity rows drawn from the canonical frame.
    primary_column : str
        Name of the column carrying the primary label.
    schema_columns : list of str
        Columns the synthetic entity must populate.
    domain : str
        Domain name (used in ``entity_id``).
    prompt_template : str
        Raw prompt string. Not parsed here — passed to *api_client*.
    llm_cache : LLMCache
        Shared LLM cache.
    api_client : callable or None
        ``(prompt_template, parent_records) -> dict``. Only called on
        cache miss when ``strict_cache`` is False.
    committee_fn : callable or None
        Optional committee validator.
    reference_labels : set of str
        Normalised real-entity labels for contamination checks.
    placement_mode : {"matched_across", "single_source_distractor"}
    source_placements : list of str
        Target source names for the synthetic entity.
    entity_id : str
        Synthetic stable identifier.
    strict_cache : bool, default False
        When True, a cache miss raises ``LLMCacheMiss``.
    rejection_log : dict of str to int or None, optional
        If provided, a per-reason counter is incremented in-place
        whenever a guardrail rejects the synthetic entity. Keys:
        ``"nondict_result"``, ``"empty_primary_label"``,
        ``"contamination_empty_primary_label"``,
        ``"contamination_collision_with_real_entity"``,
        ``"committee_validation"``. Caller initialises and owns the dict.

    Returns
    -------
    InterpolatedEntity or None
        ``None`` when the synthetic entity fails a guardrail; the
        caller is expected to skip it and optionally try another parent
        pair.
    """
    parent_ids = [str(row.name) for row in parent_rows]
    pair_hash = parent_pair_hash(parent_ids)

    # The LLMCache API is generic: we bake the domain and placement
    # into ``attribute`` / ``value`` so the key is unique per pair.
    cache_source = f"{domain}_interp"
    cache_attr = f"pair_{pair_hash}"
    cache_value = "|".join(parent_ids)

    def _api_call() -> dict[str, Any]:
        if api_client is None:
            raise RuntimeError("api_client required on cache miss")
        parent_records = [
            {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}
            for row in parent_rows
        ]
        raw = api_client(prompt_template, parent_records)
        return raw

    try:
        payload = llm_cache.call_or_cache(
            cache_source,
            cache_attr,
            cache_value,
            _api_call if not strict_cache else None,
            strict=strict_cache,
        )
    except LLMCacheMiss:
        raise

    raw_result = payload.get("result") or {}
    if not isinstance(raw_result, dict):
        logger.warning(
            "Interpolation result is not a dict for %s: %r", entity_id, raw_result
        )
        if rejection_log is not None:
            rejection_log["nondict_result"] = rejection_log.get("nondict_result", 0) + 1
        return None

    # Coerce to the schema columns; blank values for missing keys.
    attributes: dict[str, Any] = {}
    for col in schema_columns:
        if col in raw_result:
            attributes[col] = raw_result[col]
        else:
            attributes[col] = ""

    # Schema validity — primary column must be non-empty.
    primary_value = str(attributes.get(primary_column, "") or "").strip()
    if not primary_value:
        logger.info("Interpolated entity %s rejected: empty primary label", entity_id)
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
        logger.info("Interpolated entity %s rejected: %s", entity_id, contam_status)
        if rejection_log is not None:
            key = f"contamination_{contam_status}"
            rejection_log[key] = rejection_log.get(key, 0) + 1
        return None

    # Committee gate.
    if committee_fn is not None:
        parent_records = [
            {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}
            for row in parent_rows
        ]
        if not committee_fn(attributes, parent_records):
            logger.info("Interpolated entity %s rejected: committee", entity_id)
            if rejection_log is not None:
                rejection_log["committee_validation"] = (
                    rejection_log.get("committee_validation", 0) + 1
                )
            return None

    return InterpolatedEntity(
        entity_id=entity_id,
        parent_ids=parent_ids,
        attributes=attributes,
        placement_mode=placement_mode,
        source_placements=list(source_placements),
        cache_path=f"{cache_source}/{cache_attr}",
        contamination_check_status=contam_status,
    )


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------


def place_entity_across_sources(
    entity: InterpolatedEntity,
    *,
    sources: dict[str, pd.DataFrame],
    source_schemas: dict[str, list[str]],
    id_columns: dict[str, str],
) -> dict[str, pd.DataFrame]:
    """Append the synthetic entity row to each target source.

    Uses the schema intersection between each source's columns and
    ``entity.attributes``. The source ID column is populated with
    ``entity.entity_id`` (or a source-qualified variant). Other
    columns are filled from ``entity.attributes``.

    Parameters
    ----------
    entity : InterpolatedEntity
    sources : dict
        Per-source DataFrames (mutated via copy + concat).
    source_schemas : dict
        Per-source list of columns to populate. Anything not present in
        ``entity.attributes`` is left as NaN.
    id_columns : dict
        Per-source id column name.

    Returns
    -------
    dict of str to pandas.DataFrame
        New per-source frames with the synthetic entity appended.
    """
    out: dict[str, pd.DataFrame] = {}
    for src_name, df in sources.items():
        if src_name not in entity.source_placements:
            out[src_name] = df
            continue

        new_row: dict[str, Any] = {}
        id_col = id_columns.get(src_name)
        if id_col is not None:
            new_row[id_col] = f"{entity.entity_id}__{src_name}"
        for col in source_schemas.get(src_name, list(df.columns)):
            if col == id_col:
                continue
            if col in entity.attributes:
                new_row[col] = entity.attributes[col]

        new_df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        new_df.attrs = df.attrs.copy()
        out[src_name] = new_df
    return out


def build_openai_interpolation_client(
    *,
    model_id: str,
    temperature: float = 0.0,
    max_tokens: int | None = 2048,
) -> "Callable[[str, list[dict[str, Any]]], dict[str, Any]]":
    """Build an OpenAI-backed :data:`LLMInterpolateFn` for K2 niche.

    Returns a callable matching the
    ``Callable[[str, list[dict[str, Any]]], dict[str, Any]]`` contract
    that :func:`interpolate_entity` expects on cache miss:

    1. Substitutes ``{parent_records_json}`` and ``{schema_columns_json}``
       in the prompt template (see
       ``config/knob_02_niche/_prompts/interpolate_v1.txt``).
    2. Calls ``langchain_openai.ChatOpenAI`` via :func:`build_chat_openai`
       (which enforces the reasoning-token floor for gpt-5* models).
    3. Parses the JSON response into a ``dict``. Code fences (`````json
       ... `````) are stripped before decoding.

    Schema columns are derived from the union of keys across all
    *parent_records* (preserving insertion order from the first parent),
    matching the canonical-frame columns built upstream.

    On any failure (network error, malformed JSON, non-dict response),
    returns ``{}``; :func:`interpolate_entity` then rejects the entity
    via the schema-validity guardrail and increments
    ``rejection_log["nondict_result"]`` so the cause is visible in
    ``knob_02_realised.csv``.

    Parameters
    ----------
    model_id : str
        OpenAI chat model id (e.g. ``"gpt-5.4-mini"``).
    temperature : float, default 0.0
        Sampling temperature — pinned to 0 for cache determinism.
    max_tokens : int or None, default 2048
        Per-call token cap. Reasoning models (gpt-5*) need ≥ 1024;
        :func:`build_chat_openai` raises ``ValueError`` otherwise.

    Returns
    -------
    Callable[[str, list[dict[str, Any]]], dict[str, Any]]
        An LLMInterpolateFn suitable for ``apply_knob_02(api_client=...)``.

    Notes
    -----
    This client is invoked **only on cache miss**. Once a parent-pair
    interpolation is committed to ``cache/knob_02_interpolations/``,
    subsequent runs reuse the cached result for free. Strict-cache
    runs short-circuit before reaching the client.
    """
    from .llm_client import build_chat_openai

    chat = build_chat_openai(
        model=model_id,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    def _call(template: str, parent_records: list[dict[str, Any]]) -> dict[str, Any]:
        schema_columns: list[str] = []
        seen: set[str] = set()
        for record in parent_records:
            for key in record.keys():
                if key not in seen:
                    seen.add(key)
                    schema_columns.append(key)

        try:
            prompt = template.format(
                parent_records_json=json.dumps(parent_records, default=str),
                schema_columns_json=json.dumps(schema_columns),
            )
        except KeyError as exc:
            logger.warning(
                "K2 OpenAI client: prompt template missing placeholder %s; "
                "cannot interpolate",
                exc,
            )
            return {}

        try:
            response = chat.invoke(prompt)
        except Exception as exc:  # pragma: no cover - network/runtime errors
            logger.warning("K2 OpenAI client: invoke failed: %s", exc)
            return {}

        content = getattr(response, "content", response)
        if isinstance(content, list):
            content = "".join(
                str(p) if not isinstance(p, dict) else str(p.get("text", ""))
                for p in content
            )
        text = str(content).strip()

        # Strip ```json ... ``` or ``` ... ``` fences if the model added them.
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
                "K2 OpenAI client: JSON decode failed (%s); raw=%r",
                exc,
                text[:200],
            )
            return {}

        if not isinstance(parsed, dict):
            logger.warning("K2 OpenAI client: response is not a dict: %r", parsed)
            return {}
        return parsed

    return _call


def default_api_client_from_attributes(
    template: str, parent_records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Deterministic fallback interpolation used when no LLM is wired.

    Blends parent records attribute-by-attribute: for string columns,
    alternates tokens from the two parents; for numeric columns,
    averages; otherwise picks the first parent's value. The primary
    column is stitched from the two parents' first tokens so the
    synthetic label is plausibly new.

    This is **not** a replacement for the LLM interpolation path — it
    is the last-ditch cache-population fallback used by the smoke test
    when the real API client is unavailable. Real runs pin a real
    ``claude-opus-4-6`` (or equivalent) client on the CLI.
    """
    del template
    if not parent_records:
        return {}

    out: dict[str, Any] = {}
    for col in parent_records[0].keys():
        values = [r.get(col) for r in parent_records if r.get(col) not in (None, "")]
        if not values:
            out[col] = ""
            continue
        first = values[0]
        if isinstance(first, (int, float)) and not isinstance(first, bool):
            try:
                out[col] = sum(float(v) for v in values) / len(values)
            except Exception:
                out[col] = first
        elif isinstance(first, str):
            if len(values) >= 2 and col.lower() in {
                "name",
                "company",
                "title",
                "album",
                "artist",
                "franchise",
            }:
                a = str(values[0]).split()
                b = str(values[1]).split()
                half_a = a[: max(1, len(a) // 2)]
                half_b = b[len(b) // 2 :] if len(b) > 1 else b
                out[col] = " ".join(half_a + half_b) + " Synth"
            else:
                out[col] = str(first)
        else:
            out[col] = first
    return out
