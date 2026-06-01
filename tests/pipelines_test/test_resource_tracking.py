"""Tests for :mod:`pipelines.lib._resource_tracking`."""

from __future__ import annotations

import time

import numpy as np

from pipelines.lib._resource_tracking import (
    _PSUTIL_AVAILABLE,
    PeakRSSTracker,
    process_lifetime_peak_mb,
)


def test_peak_rss_tracker_reports_positive_peak_after_allocation() -> None:
    """Allocating a sizeable numpy array bumps the observed peak RSS."""
    with PeakRSSTracker(sampling_interval_s=0.02) as tracker:
        # Allocate ~80 MB of random floats to make the peak visible.
        big = np.random.rand(10_000_000)
        # Keep a reference around so the allocator can't release it
        # before the sampler ticks.
        assert big.size == 10_000_000
        time.sleep(0.1)
    if _PSUTIL_AVAILABLE:
        assert tracker.peak_mb > 0.0
    else:
        # No-op fallback path: peak stays at 0.0.
        assert tracker.peak_mb == 0.0


def test_peak_rss_tracker_peak_mb_is_float() -> None:
    """``peak_mb`` is always a float, even for an empty context."""
    with PeakRSSTracker() as tracker:
        pass
    assert isinstance(tracker.peak_mb, float)
    assert tracker.peak_mb >= 0.0


def test_process_lifetime_peak_mb_returns_positive_float() -> None:
    """The lifetime peak query returns a finite positive float."""
    value = process_lifetime_peak_mb()
    assert isinstance(value, float)
    # Test process itself has already allocated some memory.
    assert value > 0.0
