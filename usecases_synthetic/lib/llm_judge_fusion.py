"""LLM-as-Judge per-cell fusion adapter (prompt v2).

Prompts an LLM to arbitrate per-cell fusion conflicts. The v2 prompt
(C12, 2026-05-25) removes the v1 verbatim-only constraint and lets the
LLM choose its operation per call from a fixed set
(``verbatim_pick``, ``aggregation``, ``union``, ``intersection``,
``normalization``, ``interpolation``). Synthesis is permitted; the
returned value need not appear in the candidate list.

Hygiene
-------
- ``temperature=0`` enforced via the prompt-version + model-id baked into
  the cache key.
- File-backed cache under ``usecases_synthetic/cache/llm_judge_fusion/``
  keyed by ``sha256(prompt_version | model_id | prompt_text)``. Bumping
  the prompt content (and therefore ``_DEFAULT_PROMPT_VERSION``)
  invalidates the existing cache by construction.
- ``llm_callable`` injection hook bypasses any external API call so unit
  tests run fully offline.
- Structured-output JSON contract:
  ``{"value": ..., "operation": "<op>", "confidence": 0..1, "reasoning": "..."}``.
  On parse failure the adapter falls back to majority voting.
- Operation log: when ``op_log_path`` is wired by the runner, every
  non-trivial call appends one row to that CSV with the chosen value,
  operation, confidence, reasoning, and cache_hit flag — used to
  power per-(entity, attribute) audits like "for the duration attribute,
  LLM picked ``interpolation`` 78% of calls".

Per-cell scope
--------------
Unlike the TD adapters this method is *natively* per-cell — the LLM
judges each conflicting cell in isolation. No corpus-wide state is
required.
"""

from __future__ import annotations

import csv
import datetime as _dt
import hashlib
import json
import logging
import threading
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

FusionResult = Tuple[Any, float, Dict[str, Any]]


def is_valid_value(value: Any) -> bool:
    """Return True if ``value`` is fusable (non-None, non-NaN, non-empty list)."""
    if value is None:
        return False
    if isinstance(value, (list, tuple)):
        return len(value) > 0
    try:
        if isinstance(value, np.ndarray):
            return value.size > 0
    except Exception:
        pass
    try:
        return not pd.isna(value)
    except Exception:
        return True


def empty_result(rule: str) -> FusionResult:
    """Standard early-return for cells with no fusable values."""
    return None, 0.0, {"reason": "no_valid_values", "rule": rule}


logger = logging.getLogger(__name__)

_RULE = "llm_judge"
_DEFAULT_PROMPT_VERSION = "v2"
_DEFAULT_MODEL = "openai/gpt-4o-mini"
_DEFAULT_CACHE_DIR = Path("usecases_synthetic/cache/llm_judge_fusion")


VALID_OPERATIONS_V2: frozenset[str] = frozenset(
    {
        "verbatim_pick",
        "aggregation",
        "union",
        "intersection",
        "normalization",
        "interpolation",
    }
)


_SYSTEM_PROMPT_V2 = (
    "You arbitrate data-fusion conflicts. Multiple data sources have "
    "provided candidate values for the same attribute of the same "
    "real-world entity.\n\n"
    "Choose the single canonical value for this attribute. Available "
    "operations:\n"
    "- verbatim_pick: return one of the candidates verbatim.\n"
    "- aggregation: combine numeric candidates statistically "
    "(e.g. median, mean, robust aggregate).\n"
    "- union: union of list-valued candidates.\n"
    "- intersection: intersection of list-valued candidates.\n"
    "- normalization: canonicalize/clean to a single form "
    '(e.g. "NYC" -> "New York City").\n'
    "- interpolation: synthesize a value not present verbatim in any "
    "source when the candidates collectively imply a more precise / "
    "complete answer (e.g. infer 2007-05-12 from two partial dates).\n\n"
    "Synthesis is permitted; the chosen value need not appear in the "
    "candidate list. Tag your output with the operation that best "
    "describes what you did.\n\n"
    "Respond with strict JSON of the form:\n"
    '{"value": <chosen value (string/number/array/etc.)>, '
    '"operation": "<one of: verbatim_pick, aggregation, union, '
    'intersection, normalization, interpolation>", '
    '"confidence": <float in 0..1>, '
    '"reasoning": "<short text, <= 1 sentence>"}\n'
    "No prose. No markdown. No code fences."
)


def _build_prompt(
    attribute: str,
    candidates: List[Dict[str, Any]],
) -> str:
    """Render the user-side prompt body for a v2 fusion-judge call.

    Parameters
    ----------
    attribute
        Attribute name being arbitrated. Used only in the prompt text.
    candidates
        List of ``{"source": str, "value": str}`` dicts in input order.
    """
    lines = [f"Attribute: {attribute}", "Candidates:"]
    for i, c in enumerate(candidates, start=1):
        lines.append(f"  {i}. source={c['source']!r}  value={c['value']!r}")
    lines.append(
        'Return strict JSON {"value": ..., "operation": ..., '
        '"confidence": ..., "reasoning": ...}.'
    )
    return "\n".join(lines)


def _make_cache_key(
    prompt_version: str,
    model_id: str,
    prompt_text: str,
) -> str:
    payload = "|".join([prompt_version, model_id, prompt_text])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_cache(cache_dir: Path, key: str) -> Dict[str, Any] | None:
    path = cache_dir / f"{key}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)  # type: ignore[no-any-return]


def _store_cache(cache_dir: Path, key: str, payload: Dict[str, Any]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)


def _parse_response(raw: str) -> Dict[str, Any] | None:
    """Parse a raw LLM response into a v2 payload.

    Returns
    -------
    dict with keys ``{"value", "operation", "confidence", "reasoning"}``
    on success; ``None`` if the response cannot be parsed or the operation
    tag isn't in :data:`VALID_OPERATIONS_V2`.

    Notes
    -----
    Unlike the v1 parser, this does **not** enforce that ``value`` appear
    verbatim in the candidate list — synthesis is allowed under v2.
    """
    text = (raw or "").strip()
    if text.startswith("```"):
        lines = [l for l in text.splitlines() if not l.startswith("```")]
        text = "\n".join(lines).strip()
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, dict):
        return None
    if "value" not in parsed or "operation" not in parsed:
        return None
    operation = parsed.get("operation")
    if not isinstance(operation, str) or operation not in VALID_OPERATIONS_V2:
        return None
    try:
        confidence_f = float(parsed.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence_f = 0.5
    reasoning_raw = parsed.get("reasoning")
    reasoning = "" if reasoning_raw is None else str(reasoning_raw).strip()
    return {
        "value": parsed["value"],
        "operation": operation,
        "confidence": max(0.0, min(1.0, confidence_f)),
        "reasoning": reasoning,
    }


# ---------------------------------------------------------------------------
# Operation log (CSV append; thread-safe)
# ---------------------------------------------------------------------------

_OP_LOG_HEADER: list[str] = [
    "timestamp",
    "group_id",
    "attribute",
    "num_sources",
    "candidates",
    "chosen_value",
    "operation",
    "confidence",
    "reasoning",
    "cache_hit",
    "model_id",
]

_op_log_lock = threading.Lock()


def _append_op_log(
    path: Path,
    *,
    group_id: Any,
    attribute: str,
    num_sources: int,
    candidates: List[str],
    chosen_value: Any,
    operation: str,
    confidence: float,
    reasoning: str,
    cache_hit: bool,
    model_id: str,
) -> None:
    """Append one row to the per-(member, level) llm_only operation log.

    Writes ``llm_only_operations.csv`` columns matching :data:`_OP_LOG_HEADER`.
    Creates parents and writes the header on first append. Thread-safe via
    a process-wide lock; the engine runs cells sequentially so contention is
    rare but the lock makes the writer safe regardless.
    """
    with _op_log_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(_OP_LOG_HEADER)
            writer.writerow(
                [
                    _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                    "" if group_id is None else str(group_id),
                    attribute,
                    num_sources,
                    json.dumps(candidates, ensure_ascii=False),
                    json.dumps(chosen_value, ensure_ascii=False, default=str),
                    operation,
                    f"{confidence:.4f}",
                    reasoning,
                    int(bool(cache_hit)),
                    model_id,
                ]
            )


def llm_judge(
    values: List[Any],
    *,
    llm_callable: Callable[[str, str, str], str] | None = None,
    prompt_version: str = _DEFAULT_PROMPT_VERSION,
    model_id: str = _DEFAULT_MODEL,
    cache_dir: Path | str = _DEFAULT_CACHE_DIR,
    strict_cache: bool = False,
    fallback_on_parse_fail: bool = True,
    op_log_path: Path | str | None = None,
    **kwargs: Any,
) -> FusionResult:
    """LLM-as-Judge per-cell fusion under prompt v2.

    Parameters
    ----------
    values
        Per-cell candidate values.
    llm_callable
        Function ``(system_prompt, user_prompt, model_id) -> raw_text``.
        When ``None`` and the cache misses, the call falls back to
        majority voting.
    prompt_version
        Prompt-version tag baked into the cache key. Default ``"v2"``.
        Bumping invalidates the on-disk cache by construction.
    model_id
        Model identifier baked into the cache key.
    cache_dir
        Directory holding per-prompt JSON cache files.
    strict_cache
        When ``True`` and the cache misses *and* no ``llm_callable`` is
        wired, raise ``RuntimeError`` rather than silently falling back to
        voting.
    fallback_on_parse_fail
        When ``True``, fall back to majority voting if the LLM response
        cannot be parsed.
    op_log_path
        Optional path to an operation-log CSV. When set, every non-trivial
        call (i.e. one that actually invoked the LLM judge or hit cache,
        not single-source short-circuits) appends a row recording the
        operation tag + chosen value. The runner wires this per (member,
        domain, level) to power downstream stats like "for ``duration``,
        ``llm_only`` picked ``interpolation`` 78% of calls".
    **kwargs
        Engine-supplied cell context (``sources``, ``source_datasets``,
        ``attribute``, optionally ``group_id``).
    """
    valid_pairs: List[Tuple[Any, str]] = []
    sources = kwargs.get("sources") or []
    source_datasets: Dict[str, str] = kwargs.get("source_datasets") or {}
    attribute = str(kwargs.get("attribute", "value"))
    group_id = kwargs.get("group_id")

    for idx, val in enumerate(values):
        if not is_valid_value(val):
            continue
        rid = sources[idx] if idx < len(sources) else f"src_{idx}"
        ds = source_datasets.get(rid, str(rid))
        valid_pairs.append((val, ds))

    if not valid_pairs:
        return empty_result(_RULE)

    if len(valid_pairs) == 1:
        return (
            valid_pairs[0][0],
            1.0,
            {
                "rule": _RULE,
                "num_sources": 1,
                "note": "single_source_short_circuit",
            },
        )

    # Build canonical candidate list (preserve input order, drop duplicates).
    candidate_strings: List[str] = []
    candidate_originals: List[Any] = []
    seen: set[str] = set()
    candidates_for_prompt: List[Dict[str, Any]] = []
    for val, ds in valid_pairs:
        s = str(val)
        if s in seen:
            continue
        seen.add(s)
        candidate_strings.append(s)
        candidate_originals.append(val)
        candidates_for_prompt.append({"source": ds, "value": s})

    cache_path = Path(cache_dir)
    prompt_text = _build_prompt(attribute, candidates_for_prompt)
    key = _make_cache_key(prompt_version, model_id, prompt_text)

    cached = _load_cache(cache_path, key)
    parsed: Dict[str, Any] | None = None
    cache_hit = cached is not None
    if cached is not None:
        parsed = _parse_response(cached.get("raw_response", ""))

    if parsed is None and not cache_hit:
        if llm_callable is None:
            if strict_cache:
                raise RuntimeError(
                    f"llm_judge: strict cache miss with no llm_callable for "
                    f"attribute={attribute!r}, prompt_version={prompt_version!r}, "
                    f"model_id={model_id!r}"
                )
            return _fallback_voting(
                candidate_originals,
                candidate_strings,
                valid_pairs,
                attribute,
                reason="no_llm_callable_no_cache",
            )
        try:
            raw = llm_callable(_SYSTEM_PROMPT_V2, prompt_text, model_id)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("llm_judge LLM call failed: %s", exc)
            return _fallback_voting(
                candidate_originals,
                candidate_strings,
                valid_pairs,
                attribute,
                reason=f"llm_call_error:{type(exc).__name__}",
            )

        parsed = _parse_response(raw)
        _store_cache(
            cache_path,
            key,
            {
                "prompt_version": prompt_version,
                "model_id": model_id,
                "attribute": attribute,
                "candidates": candidates_for_prompt,
                "raw_response": raw,
                "parsed": parsed,
            },
        )

    if parsed is None:
        if not fallback_on_parse_fail:
            return (
                None,
                0.0,
                {
                    "rule": _RULE,
                    "reason": "parse_failed",
                    "cache_hit": cache_hit,
                },
            )
        return _fallback_voting(
            candidate_originals,
            candidate_strings,
            valid_pairs,
            attribute,
            reason="parse_failed",
        )

    raw_value = parsed["value"]
    operation = parsed["operation"]
    confidence = float(parsed["confidence"])
    reasoning = parsed["reasoning"]

    # If the LLM returned a value that matches a candidate verbatim, prefer
    # the original Python value so types (list / number) survive the
    # round-trip; otherwise pass the LLM output through unchanged
    # (synthesis path).
    raw_value_str = str(raw_value)
    if raw_value_str in candidate_strings:
        chosen_idx = candidate_strings.index(raw_value_str)
        chosen_value: Any = candidate_originals[chosen_idx]
        synthesized = False
    else:
        chosen_value = raw_value
        synthesized = True

    if op_log_path is not None:
        try:
            _append_op_log(
                Path(op_log_path),
                group_id=group_id,
                attribute=attribute,
                num_sources=len(valid_pairs),
                candidates=candidate_strings,
                chosen_value=chosen_value,
                operation=operation,
                confidence=confidence,
                reasoning=reasoning,
                cache_hit=cache_hit,
                model_id=model_id,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("llm_judge op-log append failed: %s", exc)

    return (
        chosen_value,
        confidence,
        {
            "rule": _RULE,
            "prompt_version": prompt_version,
            "model_id": model_id,
            "attribute": attribute,
            "num_sources": len(valid_pairs),
            "num_candidates": len(candidate_strings),
            "candidates": candidate_strings,
            "operation": operation,
            "reasoning": reasoning,
            "synthesized": synthesized,
            "cache_hit": cache_hit,
            "cache_key": key,
        },
    )


def _fallback_voting(
    candidate_originals: List[Any],
    candidate_strings: List[str],
    valid_pairs: List[Tuple[Any, str]],
    attribute: str,
    *,
    reason: str,
) -> FusionResult:
    """Robust fallback when the LLM is unavailable / response unparseable.

    Picks the most-claimed candidate; ties are broken by input order.
    """
    counter = Counter(str(v) for v, _ in valid_pairs)
    most_common = counter.most_common()
    winner_str, winner_count = most_common[0]
    chosen_idx = candidate_strings.index(winner_str)
    chosen_value = candidate_originals[chosen_idx]
    confidence = float(winner_count) / float(len(valid_pairs))
    return (
        chosen_value,
        confidence,
        {
            "rule": _RULE,
            "attribute": attribute,
            "fallback": "voting",
            "reason": reason,
            "vote_distribution": dict(counter),
            "num_sources": len(valid_pairs),
        },
    )


__all__ = ["llm_judge", "VALID_OPERATIONS_V2"]
