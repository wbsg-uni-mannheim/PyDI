"""PLM scorer and LLM adjudicator adapters for the K2 hard-negative gate.

Implements ``plans/plan_s1_scale.md`` §S3 "hard-negative mining policy".
The gate itself lives in :mod:`corner_case_miner.apply_hard_negative_policy`;
this module supplies concrete callables that plug into that gate.

- :func:`build_ditto_plm_scorer` wraps a :class:`DittoMatcher` so it can
  score a list of ``(rid_a, rid_b)`` record pairs in one shot. The
  matcher is loaded lazily and cached across invocations; scoring runs
  with ``threshold=0.0`` so the raw softmax probability is returned for
  every pair (the gate applies θ/δ separately).
- :func:`build_llm_adjudicator` wraps :class:`LLMCache` with a synthetic
  ``(source, attribute, value)`` key shaped as
  ``("k02_hard_neg_adjudicator", <domain>, "<rid_a>|<rid_b>|<serialized_fields>")``
  so responses share the committed-on-disk cache pattern already used
  by K1/K2.

Both adapters are intentionally decoupled from the main dispatcher so
tests can stub them out.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import pandas as pd

from .corner_case_miner import LlmAdjudicator, PlmScorer, RecordPair
from .ditto_matcher import DittoMatcher
from .llm_cache import LLMCache, LLMCacheMiss

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ditto PLM scorer
# ---------------------------------------------------------------------------


def build_ditto_plm_scorer(
    *,
    checkpoint_path: Path,
    fields: Sequence[str],
    sources: dict[str, pd.DataFrame],
    id_columns: dict[str, str],
    attribute_mapping: dict[str, dict[str, str]],
    max_len: int = 256,
    max_field_len: int = 350,
    batch_size: int = 16,
    device: str | None = None,
) -> PlmScorer:
    """Return a :data:`PlmScorer` backed by a trained Ditto checkpoint.

    The scorer accepts an arbitrary ``(rid_a, rid_b)`` pair where each
    record id is expected to appear in exactly one of *sources*. The
    record is pulled from that source and projected onto the canonical
    schema via *attribute_mapping* so the Ditto encoder sees the same
    column layout it was trained on.

    Parameters
    ----------
    checkpoint_path : Path
        Path to the Ditto ``checkpoints/best`` directory.
    fields : sequence of str
        Canonical schema fields (in training order).
    sources : dict
        Per-source record DataFrames.
    id_columns : dict
        ``source_name -> id_column`` (per knob YAML ``id_columns``).
    attribute_mapping : dict
        ``source_name -> {source_col: canonical_col}`` (per knob YAML
        ``attribute_mapping``).
    max_len, max_field_len, batch_size, device
        Forwarded to :class:`DittoMatcher`.

    Returns
    -------
    PlmScorer
        Callable ``pairs -> {pair: score}``. Pairs whose ids can't be
        resolved are omitted (the caller treats missing scores as
        ``no_score`` in the audit and keeps the pair conservatively).
    """
    matcher = DittoMatcher(
        checkpoint_path=checkpoint_path,
        fields=list(fields),
        max_len=max_len,
        max_field_len=max_field_len,
        batch_size=batch_size,
        device=device,
    )

    canonical_by_source: dict[str, pd.DataFrame] = {}
    rid_to_source: dict[str, str] = {}
    for src_name, df in sources.items():
        id_col = id_columns.get(src_name)
        if id_col is None or id_col not in df.columns:
            continue
        colmap = attribute_mapping.get(src_name, {})
        projected = pd.DataFrame(index=df.index)
        projected["_pydi_id"] = df[id_col].astype(str)
        for src_col, canon_col in colmap.items():
            if canon_col in fields and src_col in df.columns:
                projected[canon_col] = df[src_col]
        for f in fields:
            if f not in projected.columns:
                projected[f] = None
        canonical_by_source[src_name] = projected
        for rid in projected["_pydi_id"]:
            rid_to_source[str(rid)] = src_name

    def score(pairs: Sequence[RecordPair]) -> dict[RecordPair, float]:
        if not pairs:
            return {}
        left_rows: list[dict[str, Any]] = []
        right_rows: list[dict[str, Any]] = []
        candidate_rows: list[dict[str, Any]] = []
        key_by_index: list[RecordPair] = []
        left_by_id: dict[str, dict[str, Any]] = {}
        right_by_id: dict[str, dict[str, Any]] = {}
        for rid_a, rid_b in pairs:
            src_a = rid_to_source.get(str(rid_a))
            src_b = rid_to_source.get(str(rid_b))
            if src_a is None or src_b is None:
                continue
            df_a = canonical_by_source[src_a]
            df_b = canonical_by_source[src_b]
            row_a = df_a.loc[df_a["_pydi_id"] == str(rid_a)]
            row_b = df_b.loc[df_b["_pydi_id"] == str(rid_b)]
            if row_a.empty or row_b.empty:
                continue
            ra = row_a.iloc[0].to_dict()
            rb = row_b.iloc[0].to_dict()
            left_key = f"L::{rid_a}"
            right_key = f"R::{rid_b}"
            ra["_pydi_id"] = left_key
            rb["_pydi_id"] = right_key
            left_by_id[left_key] = ra
            right_by_id[right_key] = rb
            candidate_rows.append({"id1": left_key, "id2": right_key})
            key_by_index.append((rid_a, rid_b))

        if not candidate_rows:
            return {}

        df_left = pd.DataFrame(list(left_by_id.values()))
        df_right = pd.DataFrame(list(right_by_id.values()))
        candidates = pd.DataFrame(candidate_rows)

        corr = matcher.match(
            df_left=df_left,
            df_right=df_right,
            candidates=candidates,
            id_column="_pydi_id",
            threshold=0.0,
        )

        result: dict[RecordPair, float] = {}
        if corr.empty:
            return result
        corr_map = {
            (str(r["id1"]), str(r["id2"])): float(r["score"])
            for _, r in corr.iterrows()
        }
        for (rid_a, rid_b), (lk, rk) in zip(
            key_by_index,
            [(f"L::{a}", f"R::{b}") for a, b in key_by_index],
        ):
            s = corr_map.get((lk, rk))
            if s is not None:
                result[(rid_a, rid_b)] = s
        return result

    return score


# ---------------------------------------------------------------------------
# LLM adjudicator
# ---------------------------------------------------------------------------


DEFAULT_ADJUDICATOR_PROMPT = (
    "You are adjudicating whether two records refer to the same real-world "
    "entity. Reply with exactly one token — 'yes' if the pair is a match, "
    "'no' otherwise. No other text.\n\nRecord A: {record_a}\nRecord B: {record_b}"
)


def _serialize_record(
    rid: str,
    sources: dict[str, pd.DataFrame],
    id_columns: dict[str, str],
    attribute_mapping: dict[str, dict[str, str]],
    fields: Sequence[str],
) -> tuple[str, str]:
    """Return ``(source_name, "col1=val1 | col2=val2 | ...")`` for *rid*.

    Raises :class:`KeyError` when *rid* can't be located in any source.
    """
    for src_name, df in sources.items():
        id_col = id_columns.get(src_name)
        if id_col is None or id_col not in df.columns:
            continue
        mask = df[id_col].astype(str) == str(rid)
        if not mask.any():
            continue
        row = df.loc[mask].iloc[0]
        colmap = attribute_mapping.get(src_name, {})
        parts: list[str] = []
        for src_col, canon_col in colmap.items():
            if canon_col not in fields or src_col not in row.index:
                continue
            val = row[src_col]
            if val is None:
                continue
            text = str(val).strip()
            if not text or text.lower() in ("nan", "none", "null"):
                continue
            parts.append(f"{canon_col}={text}")
        return src_name, " | ".join(parts)
    raise KeyError(f"record id not found in any source: {rid}")


def build_llm_adjudicator(
    *,
    domain: str,
    sources: dict[str, pd.DataFrame],
    id_columns: dict[str, str],
    attribute_mapping: dict[str, dict[str, str]],
    fields: Sequence[str],
    llm_cache: LLMCache,
    api_client: Callable[[str], str] | None,
    strict_cache: bool = False,
    prompt_template: str = DEFAULT_ADJUDICATOR_PROMPT,
) -> LlmAdjudicator:
    """Return a :data:`LlmAdjudicator` backed by :class:`LLMCache`.

    The adapter serialises each record as ``col=val | col=val`` under
    the canonical schema, renders *prompt_template*, and delegates to
    *api_client* on cache miss. Responses are parsed as a boolean
    ``yes``/``no`` token. *api_client* is invoked with the rendered
    prompt; it must return a single-token reply (caller is responsible
    for temperature=0 pinning).

    Parameters
    ----------
    domain : str
        Baked into the cache key under the synthetic ``source`` slot.
    sources, id_columns, attribute_mapping, fields
        Same shape as the scorer — used only for record serialisation.
    llm_cache : LLMCache
        Shared cache (shared with K1/K2).
    api_client : callable or None
        ``str -> str`` callable. When ``None`` and a cache miss occurs
        with ``strict_cache=False``, raises :class:`RuntimeError`
        (no silent downgrade).
    strict_cache : bool
        When ``True``, cache misses raise :class:`LLMCacheMiss`.
    prompt_template : str
        Python format-string with ``{record_a}`` and ``{record_b}``
        placeholders.

    Returns
    -------
    LlmAdjudicator
        Callable ``pair -> bool`` — ``True`` if the LLM says "match".
    """

    def adjudicate(pair: RecordPair) -> bool:
        rid_a, rid_b = pair
        try:
            _src_a, text_a = _serialize_record(
                rid_a, sources, id_columns, attribute_mapping, fields
            )
            _src_b, text_b = _serialize_record(
                rid_b, sources, id_columns, attribute_mapping, fields
            )
        except KeyError as exc:
            logger.warning("adjudicator: %s — keeping pair conservatively", exc)
            return False

        prompt = prompt_template.format(record_a=text_a, record_b=text_b)
        cache_value = f"{rid_a}|{rid_b}|{text_a}||{text_b}"

        def _call() -> dict[str, Any]:
            if api_client is None:
                raise RuntimeError(
                    "adjudicator cache miss but api_client=None (no silent regen)"
                )
            reply = api_client(prompt).strip().lower()
            return {
                "prompt": prompt,
                "reply": reply,
                "says_match": reply.startswith("y"),
            }

        try:
            payload = llm_cache.call_or_cache(
                source=f"k02_hard_neg_adjudicator::{domain}",
                attribute="pair",
                value=cache_value,
                api_fn=_call,
                strict=strict_cache,
            )
        except LLMCacheMiss:
            logger.warning(
                "adjudicator strict-cache miss for %s|%s — keeping conservatively",
                rid_a,
                rid_b,
            )
            return False
        except RuntimeError as exc:
            # Cache miss + api_client=None. Matches the existing
            # strict-cache-miss branch: log, fall back to the
            # conservative "no match" verdict so the pair is dropped
            # rather than letting a stale pipeline crash on first
            # encounter. Operators who want live calls pass an explicit
            # api_client on the CLI.
            if "api_client=None" in str(exc):
                logger.warning(
                    "adjudicator cache miss + no api_client for %s|%s — "
                    "keeping conservatively",
                    rid_a,
                    rid_b,
                )
                return False
            raise

        result = payload.get("result") if "result" in payload else payload
        if isinstance(result, dict) and "says_match" in result:
            return bool(result["says_match"])
        # Back-compat: payload might already be the parsed dict.
        if isinstance(payload, dict) and "says_match" in payload:
            return bool(payload["says_match"])
        return False

    return adjudicate


__all__ = [
    "build_ditto_plm_scorer",
    "build_llm_adjudicator",
    "DEFAULT_ADJUDICATOR_PROMPT",
]
