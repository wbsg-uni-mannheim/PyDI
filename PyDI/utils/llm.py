"""Utilities for unified LLM call logging.

This module provides a small, provider-agnostic logger that captures:
- prompts/messages sent to the model
- raw responses
- timing and basic request params
- token usage metadata when available (best-effort across providers)

It is designed to work in both blocking/synchronous call paths and to be
reusable across modules. It does not depend on LangChain directly; it only
expects a "chat model" object and a "response" object that may expose
metadata in a few common places.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional


logger = logging.getLogger(__name__)


def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def _serialize_messages(messages: List[Any]) -> List[Dict[str, Any]]:
    """Convert provider/LC messages to a simple serializable structure.

    Tries to read common attributes (type/role, content) and falls back to str().
    """
    serialized: List[Dict[str, Any]] = []
    for msg in messages or []:
        msg_type = _safe_getattr(msg, "type", None) or _safe_getattr(
            msg, "role", None) or type(msg).__name__
        content = _safe_getattr(msg, "content", None)
        if content is None:
            try:
                content = str(msg)
            except Exception:
                content = "<unserializable>"
        serialized.append({"type": str(msg_type), "content": content})
    return serialized


def _extract_usage_metadata(response: Any) -> Optional[Dict[str, Any]]:
    """Best-effort extraction of token usage from diverse response objects.

    Looks in common locations used by LangChain/OpenAI/Anthropic wrappers.
    """
    # LangChain OpenAI new adapters often expose `usage_metadata`
    usage = _safe_getattr(response, "usage_metadata", None)
    if isinstance(usage, dict):
        return usage

    # Some responses have a nested `response_metadata` with `token_usage` or `usage`
    meta = _safe_getattr(response, "response_metadata", None)
    if isinstance(meta, dict):
        token_usage = meta.get("token_usage") or meta.get("usage")
        if isinstance(token_usage, dict):
            return token_usage

    # OpenAI python client v1 sometimes exposes `usage`
    usage2 = _safe_getattr(response, "usage", None)
    if isinstance(usage2, dict):
        return usage2

    return None


def _get_model_name(chat_model: Any) -> str:
    return (
        _safe_getattr(chat_model, "model_name", None)
        or _safe_getattr(chat_model, "model", None)
        or _safe_getattr(chat_model, "name", None)
        or "unknown"
    )


@dataclass
class LLMCallRecord:
    """Structured record for a single LLM call."""

    timestamp: str
    row_index: int
    attempt: int
    provider_class: str
    model: str
    duration_ms: float
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    usage: Optional[Dict[str, Any]] = None
    request_messages: List[Dict[str, Any]] = field(default_factory=list)
    response_text_preview: Optional[str] = None


class LLMCallLogger:
    """Aggregator for unified, provider-agnostic LLM call logging.

    Usage:
        logger = LLMCallLogger()
        logger.record_call(...)
        logger.flush(write_artifact)  # optional, writes JSON files when desired

    - Always emits a structured info log for each call
    - Optionally writes aggregated artifacts via the provided writer
    """

    def __init__(self) -> None:
        self._records: List[LLMCallRecord] = []

    def record_call(
        self,
        *,
        chat_model: Any,
        messages: List[Any],
        response: Any,
        row_index: int,
        attempt: int,
        duration_ms: float,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> None:
        """Record a single blocking call's details and emit a structured log."""
        model_name = _get_model_name(chat_model)
        provider_class = type(chat_model).__name__
        usage = _extract_usage_metadata(response)

        # Build safe response preview (avoid huge payloads)
        try:
            response_text = _safe_getattr(response, "content", None)
            if response_text is None:
                response_text = str(response)
            preview = str(response_text)[:2000]
        except Exception:
            preview = "<unserializable>"

        rec = LLMCallRecord(
            timestamp=datetime.utcnow().isoformat() + "Z",
            row_index=row_index,
            attempt=attempt,
            provider_class=provider_class,
            model=model_name,
            duration_ms=float(duration_ms),
            temperature=temperature,
            max_tokens=max_tokens,
            usage=usage,
            request_messages=_serialize_messages(messages),
            response_text_preview=preview,
        )

        self._records.append(rec)

        # Emit to Python logging immediately so it shows up even without artifacts
        try:
            logger.info("LLM call: %s", json.dumps(rec.__dict__))
        except Exception:
            # Fallback if json serialization fails for any reason
            logger.info("LLM call (non-JSON serializable)")

    def flush(self, write_artifact: Callable[[str, Any], Any]) -> None:
        """Write aggregated artifacts (JSON list and usage summary).

        The provided writer should accept (filename, data) and persist it.
        """
        if not self._records:
            return

        # Write all calls as JSON array
        data = [r.__dict__ for r in self._records]
        write_artifact("llm_calls.json", data)

        # Create a small usage summary
        total_calls = len(self._records)
        total_input_tokens = 0
        total_output_tokens = 0
        total_tokens = 0
        for r in self._records:
            if not r.usage:
                continue
            # Try multiple common keys
            total_input_tokens += (
                r.usage.get("input_tokens")
                or r.usage.get("prompt_tokens")
                or 0
            )
            total_output_tokens += (
                r.usage.get("output_tokens")
                or r.usage.get("completion_tokens")
                or 0
            )
            total_tokens += (
                r.usage.get("total_tokens")
                or (r.usage.get("input_tokens") or r.usage.get("prompt_tokens") or 0)
                + (r.usage.get("output_tokens")
                   or r.usage.get("completion_tokens") or 0)
            )

        summary = {
            "total_calls": total_calls,
            "total_input_tokens": int(total_input_tokens),
            "total_output_tokens": int(total_output_tokens),
            "total_tokens": int(total_tokens),
        }
        write_artifact("llm_usage_summary.json", summary)
