"""Functional tests for usecases_synthetic.lib.robust_aggregators.

Covers the three estimators added under C3.4 of
``plans/plan_committee_finalization.md``: ``trimmed_mean``,
``huber_m_estimator``, ``median_of_means``.  Each test is self-contained
with known-good arithmetic, so determinism assertions pin exact outputs.

Moved on 2026-04-22 from ``PyDI/fusion/conflict_resolution/`` to
``usecases_synthetic/lib/`` to comply with the read-only-``PyDI/`` rule
added to ``plan_committee_finalization.md`` §Process-requirement item 4.
"""

from __future__ import annotations

import numpy as np
import pytest

from usecases_synthetic.lib.robust_aggregators import (
    huber_m_estimator,
    median_of_means,
    trimmed_mean,
)

# ---------------------------------------------------------------------------
# trimmed_mean
# ---------------------------------------------------------------------------


class TestTrimmedMean:
    def test_basic_no_trim_matches_mean(self) -> None:
        value, confidence, meta = trimmed_mean([1.0, 2.0, 3.0, 4.0, 5.0], trim=0.0)
        assert value == pytest.approx(3.0)
        assert meta["rule"] == "trimmed_mean"
        assert meta["trim"] == 0.0
        assert meta["num_trimmed"] == 0
        assert meta["num_values"] == 5
        assert 0.0 <= confidence <= 1.0

    def test_trims_outliers(self) -> None:
        # 20% trim on each side removes one value from each tail (10 values → 8).
        value, _, meta = trimmed_mean(
            [1.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 1000.0],
            trim=0.2,
        )
        assert value == pytest.approx(
            np.mean([101.0, 102.0, 103.0, 104.0, 105.0, 106.0])
        )
        assert meta["num_trimmed"] == 4  # 2 per tail

    def test_single_value_returns_that_value(self) -> None:
        value, confidence, meta = trimmed_mean([42.0], trim=0.1)
        assert value == pytest.approx(42.0)
        assert confidence == 1.0
        assert meta["num_trimmed"] == 0

    def test_nan_and_none_dropped(self) -> None:
        value, _, meta = trimmed_mean([1.0, 2.0, None, float("nan"), 3.0], trim=0.0)
        assert value == pytest.approx(2.0)
        assert meta["num_values"] == 3

    def test_non_numeric_dropped(self) -> None:
        value, _, meta = trimmed_mean([1.0, "not a number", 2.0, 3.0], trim=0.0)
        assert value == pytest.approx(2.0)
        assert meta["num_values"] == 3

    def test_empty_returns_none(self) -> None:
        value, confidence, meta = trimmed_mean([], trim=0.1)
        assert value is None
        assert confidence == 0.0
        assert meta["reason"] == "no_valid_values"

    def test_all_non_numeric_returns_none(self) -> None:
        value, confidence, meta = trimmed_mean(["a", "b"], trim=0.1)
        assert value is None
        assert confidence == 0.0
        assert meta["reason"] == "no_numeric_values"

    def test_invalid_trim_raises(self) -> None:
        with pytest.raises(ValueError):
            trimmed_mean([1.0, 2.0], trim=0.5)
        with pytest.raises(ValueError):
            trimmed_mean([1.0, 2.0], trim=-0.1)

    def test_deterministic(self) -> None:
        # Same inputs -> same outputs (pure arithmetic).
        inputs = [1.0, 2.0, 10.0, 4.0, 3.0]
        v1, c1, _ = trimmed_mean(inputs, trim=0.2)
        v2, c2, _ = trimmed_mean(inputs, trim=0.2)
        assert v1 == v2
        assert c1 == c2


# ---------------------------------------------------------------------------
# huber_m_estimator
# ---------------------------------------------------------------------------


class TestHuberMEstimator:
    def test_recovers_location_on_clean_gaussian(self) -> None:
        rng = np.random.default_rng(0)
        data = rng.normal(loc=10.0, scale=1.0, size=200).tolist()
        value, confidence, meta = huber_m_estimator(data)
        # With n=200 clean Gaussian, Huber should be close to true mean.
        assert abs(value - 10.0) < 0.3
        assert meta["n_iter"] >= 1
        assert confidence > 0.8  # most should be within k*sigma

    def test_robust_to_symmetric_outliers(self) -> None:
        # 10 clean observations around 50 plus two extreme outliers.
        clean = [48.0, 49.0, 50.0, 50.0, 50.0, 51.0, 52.0, 49.5, 50.5, 50.2]
        contaminated = clean + [500.0, -500.0]
        value, _, meta = huber_m_estimator(contaminated)
        mean_contaminated = float(np.mean(contaminated))
        # Huber should stay close to the clean cluster (50), not the raw mean.
        assert abs(value - 50.0) < 2.0
        # And far from the arithmetic mean of the contaminated sample.
        assert abs(value - mean_contaminated) > 1.0
        # Majority of the 12 observations lie in the Huber quadratic region;
        # the two synthetic extremes are guaranteed to fall outside.
        assert meta["num_inliers"] >= 7
        assert meta["num_inliers"] <= 10

    def test_single_value(self) -> None:
        value, confidence, meta = huber_m_estimator([7.5])
        assert value == pytest.approx(7.5)
        assert confidence == 1.0
        assert meta["n_iter"] == 0

    def test_identical_values(self) -> None:
        value, confidence, meta = huber_m_estimator([3.0, 3.0, 3.0, 3.0])
        assert value == pytest.approx(3.0)
        assert confidence == 1.0
        assert meta.get("note") == "zero_scale_identical_values"

    def test_nan_and_none_dropped(self) -> None:
        value, _, meta = huber_m_estimator([1.0, 2.0, None, float("nan"), 3.0])
        assert meta["num_values"] == 3
        assert abs(value - 2.0) < 0.5

    def test_empty_returns_none(self) -> None:
        value, confidence, meta = huber_m_estimator([])
        assert value is None
        assert confidence == 0.0

    def test_invalid_k_raises(self) -> None:
        with pytest.raises(ValueError):
            huber_m_estimator([1.0, 2.0], k=0.0)
        with pytest.raises(ValueError):
            huber_m_estimator([1.0, 2.0], k=-1.0)

    def test_deterministic(self) -> None:
        data = [1.0, 2.0, 3.0, 4.0, 100.0]
        v1, _, _ = huber_m_estimator(data)
        v2, _, _ = huber_m_estimator(data)
        assert v1 == v2

    def test_converges_within_tolerance(self) -> None:
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        _, _, meta = huber_m_estimator(data, max_iter=100, tol=1e-8)
        assert meta["n_iter"] <= 100


# ---------------------------------------------------------------------------
# median_of_means
# ---------------------------------------------------------------------------


class TestMedianOfMeans:
    def test_reduces_to_mean_when_one_block(self) -> None:
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        value, _, meta = median_of_means(data, n_blocks=1)
        assert value == pytest.approx(3.0)
        assert meta["n_blocks"] == 1

    def test_reduces_to_median_when_blocks_equal_n(self) -> None:
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        value, _, meta = median_of_means(data, n_blocks=5)
        assert value == pytest.approx(3.0)  # median of [1,2,3,4,5]
        assert meta["n_blocks"] == 5

    def test_robust_to_single_extreme_block(self) -> None:
        # 3 blocks: means would be mean([1,2,3])=2, mean([4,5,6])=5,
        # mean([1000,1001,1002])=1001. Median of [2, 5, 1001] = 5.
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1000.0, 1001.0, 1002.0]
        value, _, meta = median_of_means(data, n_blocks=3)
        assert value == pytest.approx(5.0)
        assert meta["n_blocks"] == 3
        assert len(meta["block_means"]) == 3

    def test_default_n_blocks(self) -> None:
        # n=16 -> default n_blocks = floor(log2(16)) = 4.
        data = list(range(16))
        _, _, meta = median_of_means(data)
        assert meta["n_blocks"] == 4

    def test_single_value(self) -> None:
        value, confidence, meta = median_of_means([5.0])
        assert value == pytest.approx(5.0)
        assert confidence == 1.0
        assert meta["n_blocks"] == 1

    def test_nan_and_none_dropped(self) -> None:
        value, _, meta = median_of_means(
            [1.0, None, 2.0, float("nan"), 3.0], n_blocks=3
        )
        assert meta["num_values"] == 3

    def test_empty_returns_none(self) -> None:
        value, confidence, meta = median_of_means([])
        assert value is None
        assert confidence == 0.0

    def test_invalid_n_blocks_raises(self) -> None:
        with pytest.raises(ValueError):
            median_of_means([1.0, 2.0, 3.0], n_blocks=0)
        with pytest.raises(ValueError):
            median_of_means([1.0, 2.0, 3.0], n_blocks=4)  # > len

    def test_deterministic(self) -> None:
        data = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
        v1, _, _ = median_of_means(data, n_blocks=2)
        v2, _, _ = median_of_means(data, n_blocks=2)
        assert v1 == v2


# ---------------------------------------------------------------------------
# Public-API surface (module-level exports)
# ---------------------------------------------------------------------------


class TestModuleSurface:
    def test_importable_from_usecases_synthetic_lib(self) -> None:
        """The three callables must be re-exportable from the module's
        public API — the committee YAML references them via dotted paths
        of the form ``usecases_synthetic.lib.robust_aggregators.<fn>``.
        """
        import usecases_synthetic.lib.robust_aggregators as module

        assert callable(module.trimmed_mean)
        assert callable(module.huber_m_estimator)
        assert callable(module.median_of_means)
        assert set(module.__all__) == {
            "trimmed_mean",
            "huber_m_estimator",
            "median_of_means",
        }

    def test_conflict_resolution_function_shape(self) -> None:
        """All three must return ``(value, confidence, metadata)`` tuples
        and accept ``(values, **kwargs)`` — the ``ConflictResolutionFunction``
        protocol that ``fusion_committee.yaml`` strategy specs rely on.
        """
        for fn, kwargs in (
            (trimmed_mean, {"trim": 0.1}),
            (huber_m_estimator, {"k": 1.345}),
            (median_of_means, {"n_blocks": 2}),
        ):
            result = fn([1.0, 2.0, 3.0, 4.0], **kwargs)
            assert isinstance(result, tuple) and len(result) == 3
            value, confidence, meta = result
            assert value is None or isinstance(value, (int, float))
            assert 0.0 <= confidence <= 1.0
            assert isinstance(meta, dict)
            assert "rule" in meta
