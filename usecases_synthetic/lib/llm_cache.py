"""File-based LLM output cache shared by Knobs 1, 2, and 4.

Implements ``knobs/cross_cutting.md`` §"LLM hygiene" and
``knobs/knob_01_surface_augmentation.md`` §"Determinism & provenance":

- Cache key = ``sha256(source|attribute|value|prompt_version|model_id)``.
- Per-cell JSON files are **committed to the repo** — the cache is the
  sole source of truth on rerun. A cache miss on an unchanged variant is
  a hard error when strict mode is on, not a silent regeneration trigger.
- ``temperature=0`` and pinned ``model_id`` are honoured by the caller;
  the cache itself only stores/retrieves results.

Parameters
----------
cache_dir : Path
    Directory containing the per-cell JSON cache files. Created on first
    write.
prompt_version : str
    Pinned prompt version (e.g. ``"v1"``). Baked into the cache key so
    prompt edits invalidate the cache.
model_id : str
    Pinned model identifier (e.g. ``"claude-opus-4-6"`` or
    ``"gpt-4o-2024-08-06"``). Baked into the cache key.
"""

from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable


class LLMCacheMiss(RuntimeError):
    """Raised when a strict-mode cache lookup fails to find a cached value."""


class LLMCache:
    """File-based JSON cache for LLM outputs.

    One file per ``(source, attribute, value)`` cell under *cache_dir*,
    named ``<cell_hash>.json``. Cell hashes also embed ``prompt_version``
    and ``model_id`` so version bumps invalidate old entries without
    overwriting them.

    Examples
    --------
    >>> cache = LLMCache(Path("cache"), prompt_version="v1", model_id="gpt-4o")
    >>> h = cache.make_cell_hash("forbes", "Company", "Apple Inc.")
    >>> cache.get(h) is None
    True
    >>> cache.put(h, {"paraphrase": "Apple Incorporated"})
    >>> cache.get(h)["paraphrase"]
    'Apple Incorporated'
    """

    def __init__(
        self,
        cache_dir: Path,
        prompt_version: str,
        model_id: str,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.prompt_version = prompt_version
        self.model_id = model_id
        self._memory: dict[str, dict[str, Any]] = {}
        # Thread-safety for concurrent prewarm (flush_concurrent).
        self._lock = threading.Lock()
        # Collect mode: when active, ``call_or_cache`` records cache-miss
        # api_fns instead of calling them, then ``flush_concurrent`` runs
        # them through a thread pool. ``None`` means collect mode is off.
        self._collect: list[tuple[str, str, str, str, Callable[[], Any]]] | None = None
        self._collect_seen: set[str] | None = None

    # ---- Key derivation ----------------------------------------------------

    def make_cell_hash(
        self,
        source: str,
        attribute: str,
        value: str,
    ) -> str:
        """Derive the stable cache key for a cell.

        Parameters
        ----------
        source : str
            Source dataset name (e.g. ``"forbes"``).
        attribute : str
            Column/attribute name.
        value : str
            Original cell value (pre-paraphrase).

        Returns
        -------
        str
            Hex-encoded SHA-256 of ``source|attribute|value|prompt_version|model_id``.
        """
        payload = "|".join(
            [
                source,
                attribute,
                value,
                self.prompt_version,
                self.model_id,
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # ---- Read / write ------------------------------------------------------

    def _path(self, cell_hash: str) -> Path:
        return self.cache_dir / f"{cell_hash}.json"

    def get(self, cell_hash: str) -> dict[str, Any] | None:
        """Look up a cached entry by cell hash.

        Returns
        -------
        dict or None
            The cached payload, or ``None`` if no entry exists.
        """
        if cell_hash in self._memory:
            return self._memory[cell_hash]

        path = self._path(cell_hash)
        if not path.exists():
            return None

        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        self._memory[cell_hash] = payload
        return payload

    def put(self, cell_hash: str, payload: dict[str, Any]) -> None:
        """Write an entry to the cache (both on disk and in memory).

        Parameters
        ----------
        cell_hash : str
            Cache key.
        payload : dict
            JSON-serialisable payload. Must include at least a
            ``result`` key; callers typically also include
            ``source``, ``attribute``, ``original_value``,
            ``prompt_version``, ``model_id`` for audit.
        """
        with self._lock:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            path = self._path(cell_hash)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
            self._memory[cell_hash] = payload

    # ---- Orchestration -----------------------------------------------------

    def call_or_cache(
        self,
        source: str,
        attribute: str,
        value: str,
        api_fn: Callable[[], Any] | None,
        *,
        strict: bool = False,
    ) -> dict[str, Any]:
        """Return cached entry or invoke *api_fn* and cache the result.

        If *strict* is True and the cache misses, raise ``LLMCacheMiss``
        without invoking *api_fn*. This enforces the "committed cache is
        the sole source of truth on rerun" contract from the K1 card.

        Parameters
        ----------
        source, attribute, value : str
            Cell identifiers (hashed into the cache key).
        api_fn : Callable[[], Any] or None
            Function that calls the LLM and returns a JSON-serialisable
            result. Ignored when the cache hits. Must be provided when
            *strict* is False and the cache misses.
        strict : bool, default False
            When True, raise on cache miss instead of calling *api_fn*.

        Returns
        -------
        dict
            The cached or freshly generated payload, always with at least
            ``{"source", "attribute", "original_value", "prompt_version",
            "model_id", "result"}``.
        """
        cell_hash = self.make_cell_hash(source, attribute, value)
        cached = self.get(cell_hash)
        if cached is not None:
            return cached

        # Collect mode (concurrent prewarm): record the cache-miss api_fn for
        # later concurrent execution and return a ``result=None`` sentinel so
        # the caller degrades gracefully (the K1 paraphrase caller treats a
        # null result as "no paraphrase" -> deterministic fallback). A second
        # real pass over the (now warm) cache produces the genuine result.
        if self._collect is not None and not strict and api_fn is not None:
            if self._collect_seen is not None and cell_hash not in self._collect_seen:
                self._collect_seen.add(cell_hash)
                self._collect.append((cell_hash, source, attribute, value, api_fn))
            return {
                "source": source,
                "attribute": attribute,
                "original_value": value,
                "prompt_version": self.prompt_version,
                "model_id": self.model_id,
                "result": None,
            }

        if strict:
            raise LLMCacheMiss(
                f"Strict-mode cache miss for ({source!r}, {attribute!r}, "
                f"{value[:40]!r}, prompt={self.prompt_version!r}, "
                f"model={self.model_id!r}) — rerunning without regenerating."
            )

        if api_fn is None:
            raise ValueError("api_fn is required when strict=False and cache misses")

        result = api_fn()
        payload: dict[str, Any] = {
            "source": source,
            "attribute": attribute,
            "original_value": value,
            "prompt_version": self.prompt_version,
            "model_id": self.model_id,
            "result": result,
        }
        self.put(cell_hash, payload)
        return payload

    # ---- Concurrent prewarm ------------------------------------------------

    def begin_collect(self) -> None:
        """Enter collect mode: cache-miss ``call_or_cache`` calls are recorded
        (returning a ``result=None`` sentinel) instead of invoking the LLM."""
        self._collect = []
        self._collect_seen = set()

    def end_collect(self) -> None:
        """Leave collect mode (discards any un-flushed recorded requests)."""
        self._collect = None
        self._collect_seen = None

    def flush_concurrent(
        self,
        max_workers: int = 30,
        result_ok: Callable[[Any], bool] | None = None,
    ) -> tuple[int, int]:
        """Execute all recorded collect-mode api_fns through a thread pool.

        Each request is de-duplicated by cache key (done at record time), so
        every prompt is issued exactly once; results are persisted via the
        thread-safe :meth:`put`. Determinism is preserved because the model is
        pinned at ``temperature=0`` — concurrent execution yields the same
        cache entries a sequential run would, only the write order differs.

        Parameters
        ----------
        max_workers : int, default 30
            Thread-pool width.
        result_ok : Callable[[Any], bool] or None, default None
            Optional predicate over the raw ``api_fn`` result. When it
            returns ``False`` the result is treated as a transient failure
            and is **not** cached, so the warm-cache assumption never gets
            poisoned by e.g. a rate-limit-induced empty response — the real
            (sequential) pass simply re-issues that one cell at its slower
            rate, where 429s do not recur. ``None`` caches every result.

        Returns
        -------
        tuple[int, int]
            ``(filled, skipped)`` — entries written to cache, and results
            rejected by *result_ok* (left uncached for the real pass).
            Safe to call with an empty queue (returns ``(0, 0)``).
        """
        pending = list(self._collect or [])
        if not pending:
            return 0, 0

        skipped_lock = threading.Lock()
        skipped = 0

        def _run(item: tuple[str, str, str, str, Callable[[], Any]]) -> None:
            nonlocal skipped
            cell_hash, source, attribute, value, api_fn = item
            if self.get(cell_hash) is not None:
                return
            result = api_fn()
            if result_ok is not None and not result_ok(result):
                with skipped_lock:
                    skipped += 1
                return
            self.put(
                cell_hash,
                {
                    "source": source,
                    "attribute": attribute,
                    "original_value": value,
                    "prompt_version": self.prompt_version,
                    "model_id": self.model_id,
                    "result": result,
                },
            )

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            list(pool.map(_run, pending))

        if self._collect is not None:
            self._collect = []
        if self._collect_seen is not None:
            self._collect_seen = set()
        return len(pending) - skipped, skipped

    # ---- Diagnostics -------------------------------------------------------

    def exists(self, cell_hash: str) -> bool:
        """Return True if a cache entry (in-memory or on-disk) exists."""
        if cell_hash in self._memory:
            return True
        return self._path(cell_hash).exists()

    def clear_memory(self) -> None:
        """Drop the in-memory layer (on-disk entries are untouched)."""
        self._memory.clear()
