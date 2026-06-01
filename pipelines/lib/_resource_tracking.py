"""Per-stage peak resident-set-size (RSS) tracking utilities.

This module provides a light-weight context manager
(:class:`PeakRSSTracker`) plus a process-lifetime peak query
(:func:`process_lifetime_peak_mb`) for instrumenting the
best-of-breed pipeline with memory usage telemetry.

The tracker spawns a background thread that polls
``psutil.Process.memory_info().rss`` at roughly 100 ms intervals and
records the maximum observed value. Both the in-window peak and the
process-lifetime peak are reported in megabytes (MB =
:math:`1024^2` bytes) for human readability.

Notes
-----
- When ``psutil`` is not installed, the tracker degrades gracefully:
  it becomes a no-op returning ``0.0`` and logs a one-time warning.
- The process-lifetime peak relies on :mod:`resource`'s
  ``RUSAGE_SELF`` interface. The ``ru_maxrss`` field is reported in
  *bytes* on macOS and *kilobytes* on Linux; this helper normalises
  both to MB.
"""

from __future__ import annotations

import logging
import resource
import sys
import threading
import time
from types import TracebackType
from typing import Optional, Type

logger = logging.getLogger(__name__)


try:  # pragma: no cover - import-time branch
    import psutil

    _PSUTIL_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when psutil missing
    psutil = None  # type: ignore[assignment]
    _PSUTIL_AVAILABLE = False


_PSUTIL_WARNED = False


def _warn_psutil_missing_once() -> None:
    """Emit a single warning when psutil is unavailable."""
    global _PSUTIL_WARNED
    if not _PSUTIL_WARNED:
        logger.warning(
            "psutil not available; PeakRSSTracker will return 0.0. "
            "Install psutil to enable per-stage peak RSS tracking."
        )
        _PSUTIL_WARNED = True


class PeakRSSTracker:
    """Context manager that tracks peak resident set size (RSS).

    Spawns a daemon thread which polls the current process's RSS at
    a fixed sampling interval and keeps the maximum value seen.
    Exiting the context joins the sampler and freezes the
    :attr:`peak_mb` value.

    Parameters
    ----------
    sampling_interval_s : float, default 0.1
        How often the background thread samples RSS, in seconds.

    Attributes
    ----------
    peak_mb : float
        Peak RSS observed within the ``with`` block, in MB.
        Returns ``0.0`` if ``psutil`` is unavailable.

    Examples
    --------
    >>> with PeakRSSTracker() as tracker:
    ...     # do work
    ...     pass
    >>> peak = tracker.peak_mb  # MB
    """

    def __init__(self, sampling_interval_s: float = 0.1) -> None:
        self.sampling_interval_s = sampling_interval_s
        self._peak_bytes: int = 0
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._process: Optional["psutil.Process"] = None

    @property
    def peak_mb(self) -> float:
        """Peak RSS observed (MB). Returns ``0.0`` when psutil is missing."""
        return float(self._peak_bytes) / (1024.0 * 1024.0)

    def __enter__(self) -> "PeakRSSTracker":
        if not _PSUTIL_AVAILABLE:
            _warn_psutil_missing_once()
            return self
        self._process = psutil.Process()
        # Seed with an initial reading so the peak is non-zero even
        # for very short workloads.
        try:
            self._peak_bytes = int(self._process.memory_info().rss)
        except Exception:  # pragma: no cover - defensive
            self._peak_bytes = 0
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._poll_loop, name="PeakRSSTracker", daemon=True
        )
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        if not _PSUTIL_AVAILABLE:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        # Final sample to capture any allocation that happened after
        # the last poll tick.
        if self._process is not None:
            try:
                final = int(self._process.memory_info().rss)
                if final > self._peak_bytes:
                    self._peak_bytes = final
            except Exception:  # pragma: no cover - defensive
                pass

    def _poll_loop(self) -> None:
        """Background sampler. Stops when ``_stop_event`` is set."""
        assert self._process is not None
        while not self._stop_event.is_set():
            try:
                rss = int(self._process.memory_info().rss)
                if rss > self._peak_bytes:
                    self._peak_bytes = rss
            except Exception:  # pragma: no cover - defensive
                # Process may have terminated or psutil may have
                # raised; bail out quietly.
                break
            self._stop_event.wait(self.sampling_interval_s)


def process_lifetime_peak_mb() -> float:
    """Return the process's lifetime peak RSS in MB.

    Uses :func:`resource.getrusage` with ``RUSAGE_SELF`` to query
    ``ru_maxrss``. The reporting unit differs across platforms:

    - **macOS / Darwin**: bytes
    - **Linux**: kilobytes (1000 bytes by Linux convention, but the
      kernel reports KiB → treated as 1024 here, which is the common
      practical interpretation)

    Returns
    -------
    float
        Lifetime peak RSS in MB. Returns ``0.0`` if the value is
        unavailable or cannot be normalised.
    """
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
    except Exception:  # pragma: no cover - defensive
        return 0.0
    max_rss = float(usage.ru_maxrss)
    if max_rss <= 0:
        return 0.0
    if sys.platform == "darwin":
        # bytes -> MB
        return max_rss / (1024.0 * 1024.0)
    # Linux + others: ru_maxrss is in KB.
    return max_rss / 1024.0


__all__ = [
    "PeakRSSTracker",
    "process_lifetime_peak_mb",
    "_PSUTIL_AVAILABLE",
]
