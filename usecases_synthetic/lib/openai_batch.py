"""OpenAI Batch API wrapper for cost-bounded variant-generation LLM calls.

The synthetic pipeline issues large numbers of LLM calls during R6.2
variant generation (K2 corner-case adjudicator, K4 fabrication, K1
paraphrase, matchgpt + comem EM matchers, sm.llm_openai + magneto,
norm.llm_canonicalize). At full-candidate scale across 3 domains × 4
levels, the synchronous bill is $640-1280; OpenAI's Batch API offers
**50% cost discount** with a 24-hour completion SLA, which is acceptable
for batch-style variant generation but not for interactive sweeps.

Design
------

The submitter collects request payloads into an in-memory queue.
``flush()`` uploads the queue as a JSONL file via the OpenAI Files API,
creates a batch job, polls until completion, and returns a
``{request_id: response_text}`` dict. Caller code constructs stable
``request_id`` strings (recommended: a hash of the prompt content so
re-runs are idempotent) and looks up the response text after the
batch settles.

Integration with :class:`llm_cache.LLMCache`:
the typical flow is (a) cache-check per request, (b) for cache misses
``add()`` to the batch, (c) once the batch settles, persist each
response into the cache. The cache layer therefore keeps its
read-through semantics; only writes get batched.

Limits (per OpenAI Batch API docs as of 2026):
- 50,000 requests per batch
- 100 MB total JSONL size

The submitter chunks automatically when either limit would be exceeded.

Usage
-----

>>> from openai import OpenAI
>>> from usecases_synthetic.lib.openai_batch import OpenAIBatchSubmitter
>>>
>>> client = OpenAI()
>>> sub = OpenAIBatchSubmitter(client=client, model="gpt-5.4-mini")
>>> sub.add("req-1", messages=[{"role": "user", "content": "hi"}])
>>> sub.add("req-2", messages=[{"role": "user", "content": "hello"}])
>>> results = sub.flush(poll_interval=60)  # dict[str, str]

When integrated with the variant-generation pipeline, each call site
wraps its existing per-cell prompt into ``add()`` and consumes results
after the pipeline-wide ``flush()``.
"""

from __future__ import annotations

import io
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# OpenAI Batch API hard limits (2026 docs).
_MAX_REQUESTS_PER_BATCH = 50_000
_MAX_BYTES_PER_BATCH = 100 * 1024 * 1024  # 100 MB


@dataclass
class _PendingRequest:
    """One queued request payload for the Batch API JSONL."""

    custom_id: str
    body: dict[str, Any]
    serialized: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        if not self.serialized:
            obj = {
                "custom_id": self.custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": self.body,
            }
            self.serialized = json.dumps(obj, ensure_ascii=False) + "\n"


class OpenAIBatchSubmitter:
    """Accumulate Chat Completions requests + submit via OpenAI's Batch API.

    Parameters
    ----------
    client : openai.OpenAI
        OpenAI client instance (typically constructed by the caller
        with ``OPENAI_API_KEY`` from env).
    model : str
        Model id (e.g. ``"gpt-5.4-mini"``). Per :data:`_DEFAULT_MODEL`
        in ``LLMCache``-using modules.
    temperature : float, default ``0.0``
        Sampling temperature applied to every request. Pinned to 0.0
        for cache stability.
    max_tokens : int or None, optional
        Per-request max-output-tokens. ``None`` = use server default.
    completion_window : str, default ``"24h"``
        Batch completion SLA passed to the OpenAI API.
    """

    def __init__(
        self,
        client: Any,
        model: str,
        *,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        completion_window: str = "24h",
    ) -> None:
        self._client = client
        self._model = model
        self._temperature = float(temperature)
        self._max_tokens = max_tokens
        self._completion_window = completion_window
        self._pending: list[_PendingRequest] = []
        self._seen_ids: set[str] = set()

    @property
    def model(self) -> str:
        return self._model

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def add(
        self,
        custom_id: str,
        messages: list[dict[str, str]],
        *,
        max_tokens: Optional[int] = None,
    ) -> None:
        """Queue a single Chat Completions request.

        Parameters
        ----------
        custom_id : str
            Stable identifier the caller will use to look up the
            response. Must be unique within a single batch.
        messages : list of dict
            Chat-format message list (``role`` + ``content`` per item).
        max_tokens : int, optional
            Per-request override. Falls back to the submitter's
            ``max_tokens``.
        """
        if custom_id in self._seen_ids:
            raise ValueError(f"Duplicate custom_id in batch: {custom_id!r}")
        body: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
        }
        eff_tokens = max_tokens if max_tokens is not None else self._max_tokens
        if eff_tokens is not None:
            body["max_tokens"] = int(eff_tokens)
        req = _PendingRequest(custom_id=custom_id, body=body)
        self._pending.append(req)
        self._seen_ids.add(custom_id)

    def _chunk_by_limits(self) -> list[list[_PendingRequest]]:
        """Split pending requests into batch-API-compliant chunks."""
        chunks: list[list[_PendingRequest]] = []
        current: list[_PendingRequest] = []
        current_bytes = 0
        for req in self._pending:
            size = len(req.serialized.encode("utf-8"))
            if (
                len(current) >= _MAX_REQUESTS_PER_BATCH
                or current_bytes + size > _MAX_BYTES_PER_BATCH
            ):
                chunks.append(current)
                current = []
                current_bytes = 0
            current.append(req)
            current_bytes += size
        if current:
            chunks.append(current)
        return chunks

    def _submit_chunk(self, chunk: list[_PendingRequest]) -> str:
        """Upload a single chunk + create a batch job. Returns the batch id."""
        jsonl_bytes = "".join(r.serialized for r in chunk).encode("utf-8")
        upload = self._client.files.create(
            file=("batch.jsonl", io.BytesIO(jsonl_bytes), "application/jsonl"),
            purpose="batch",
        )
        batch = self._client.batches.create(
            input_file_id=upload.id,
            endpoint="/v1/chat/completions",
            completion_window=self._completion_window,
        )
        logger.info(
            "Submitted batch %s with %d requests (%d bytes)",
            batch.id,
            len(chunk),
            len(jsonl_bytes),
        )
        return batch.id

    def _await_batch(
        self,
        batch_id: str,
        poll_interval: int = 60,
        max_wait_s: int = 24 * 3600,
    ) -> dict[str, str]:
        """Poll for batch completion + return ``{custom_id: response_text}``.

        ``response_text`` is the first message choice's content string.
        Failed individual requests appear with an empty string value.
        """
        deadline = time.monotonic() + max_wait_s
        while True:
            batch = self._client.batches.retrieve(batch_id)
            status = batch.status
            if status in {"completed", "failed", "expired", "cancelled"}:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Batch {batch_id} not done after {max_wait_s}s "
                    f"(last status={status})"
                )
            logger.info(
                "Batch %s status=%s; sleeping %ds", batch_id, status, poll_interval
            )
            time.sleep(poll_interval)
        if status != "completed":
            raise RuntimeError(
                f"Batch {batch_id} ended with status {status}: "
                f"{getattr(batch, 'errors', None)}"
            )
        # Download output file.
        out_file_id = batch.output_file_id
        if not out_file_id:
            return {}
        content = self._client.files.content(out_file_id).read()
        if isinstance(content, bytes):
            content = content.decode("utf-8")
        results: dict[str, str] = {}
        for line in content.splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            custom_id = row.get("custom_id")
            response = row.get("response", {}) or {}
            body = response.get("body", {}) or {}
            choices = body.get("choices", []) or []
            if not choices:
                results[custom_id] = ""
                continue
            msg = choices[0].get("message", {}) or {}
            results[custom_id] = str(msg.get("content", "") or "")
        return results

    def flush(
        self,
        *,
        poll_interval: int = 60,
        max_wait_s: int = 24 * 3600,
    ) -> dict[str, str]:
        """Submit pending requests + block until all chunks complete.

        Returns
        -------
        dict
            ``{custom_id: response_text}`` for every successfully
            completed request. Requests still in flight at the
            ``max_wait_s`` deadline raise :class:`TimeoutError`.
        """
        if not self._pending:
            return {}
        chunks = self._chunk_by_limits()
        logger.info(
            "Flushing %d requests in %d chunk(s) to model %s",
            len(self._pending),
            len(chunks),
            self._model,
        )
        all_results: dict[str, str] = {}
        for chunk in chunks:
            batch_id = self._submit_chunk(chunk)
            chunk_results = self._await_batch(
                batch_id, poll_interval=poll_interval, max_wait_s=max_wait_s
            )
            all_results.update(chunk_results)
        self._pending.clear()
        self._seen_ids.clear()
        return all_results


__all__ = ["OpenAIBatchSubmitter"]
