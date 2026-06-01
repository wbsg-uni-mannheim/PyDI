"""Unit tests for ``usecases_synthetic.lib.ditto_matcher.DittoMatcher``.

Focus is the per-batch inference cache (resume-safe scoring): a SIGINT
or lid-close mid-EM otherwise loses every Ditto forward pass scored so
far on a domain-scale variant. The model itself is mocked via
``_ensure_loaded`` / ``_score_batch`` overrides so these tests stay
checkpoint-free.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import pandas as pd
import pytest

from usecases_synthetic.lib.ditto_matcher import DittoMatcher


def _make_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    left = pd.DataFrame(
        {
            "id": ["L1", "L2", "L3"],
            "name": ["alpha", "beta", "gamma"],
        }
    )
    right = pd.DataFrame(
        {
            "id": ["R1", "R2", "R3"],
            "name": ["alpha", "beta inc", "delta"],
        }
    )
    return left, right


def _make_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id1": ["L1", "L1", "L2", "L2", "L3"],
            "id2": ["R1", "R2", "R1", "R2", "R3"],
        }
    )


class _DeterministicMatcher(DittoMatcher):
    """DittoMatcher subclass that bypasses model load and scores
    deterministically based on the (id1, id2) pair so cache behaviour
    can be verified without a real checkpoint."""

    def __init__(self, *args, score_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._score_fn = score_fn or (lambda l, r: 0.7)
        self.batch_calls: list[list[tuple[str, str]]] = []

    def _ensure_loaded(self) -> None:  # type: ignore[override]
        # Skip checkpoint load entirely.
        if self._model is None:
            self._model = object()
            self._tokenizer = object()

    def _score_batch(  # type: ignore[override]
        self, texts: list[tuple[str, str]]
    ) -> list[float]:
        self.batch_calls.append(list(texts))
        return [self._score_fn(l, r) for l, r in texts]

    def _pair_text(  # type: ignore[override]
        self,
        left_row: pd.Series,
        right_row: pd.Series,
    ) -> tuple[str, str]:
        # Use ids as the deterministic pair-text so _score_fn can key off them.
        return str(left_row["id"]), str(right_row["id"])


def _checkpoint_dir(tmp_path: Path) -> Path:
    """Create a fake checkpoint directory so the existence + mtime
    branch in _cache_key has something to read."""
    ck = tmp_path / "ckpt"
    ck.mkdir()
    (ck / "model.pt").write_bytes(b"\x00")
    return ck


# ---------------------------------------------------------------------------
# Cache hashing
# ---------------------------------------------------------------------------


class TestCacheKey:
    def test_same_inputs_same_key(self, tmp_path: Path) -> None:
        ck = _checkpoint_dir(tmp_path)
        m = _DeterministicMatcher(ck, fields=["name"])
        cands = _make_candidates()
        left, right = _make_sources()
        assert m._cache_key(cands, left, right, "id") == m._cache_key(
            cands, left, right, "id"
        )

    def test_pair_order_independent(self, tmp_path: Path) -> None:
        """Cache key sorts pairs internally so input order doesn't matter."""
        ck = _checkpoint_dir(tmp_path)
        m = _DeterministicMatcher(ck, fields=["name"])
        cands1 = _make_candidates()
        cands2 = cands1.iloc[::-1].reset_index(drop=True)
        left, right = _make_sources()
        assert m._cache_key(cands1, left, right, "id") == m._cache_key(
            cands2, left, right, "id"
        )

    def test_different_pairs_different_key(self, tmp_path: Path) -> None:
        ck = _checkpoint_dir(tmp_path)
        m = _DeterministicMatcher(ck, fields=["name"])
        cands1 = _make_candidates()
        cands2 = cands1.head(3).copy()
        left, right = _make_sources()
        assert m._cache_key(cands1, left, right, "id") != m._cache_key(
            cands2, left, right, "id"
        )

    def test_field_change_invalidates(self, tmp_path: Path) -> None:
        ck = _checkpoint_dir(tmp_path)
        m1 = _DeterministicMatcher(ck, fields=["name"])
        m2 = _DeterministicMatcher(ck, fields=["name", "country"])
        cands = _make_candidates()
        left, right = _make_sources()
        assert m1._cache_key(cands, left, right, "id") != m2._cache_key(
            cands, left, right, "id"
        )

    def test_max_len_change_invalidates(self, tmp_path: Path) -> None:
        ck = _checkpoint_dir(tmp_path)
        m1 = _DeterministicMatcher(ck, fields=["name"], max_len=128)
        m2 = _DeterministicMatcher(ck, fields=["name"], max_len=256)
        cands = _make_candidates()
        left, right = _make_sources()
        assert m1._cache_key(cands, left, right, "id") != m2._cache_key(
            cands, left, right, "id"
        )

    def test_checkpoint_mtime_invalidates(self, tmp_path: Path) -> None:
        """Touching the checkpoint dir mtime changes the cache key (so a
        retrain on the same path doesn't reuse stale scores)."""
        ck = _checkpoint_dir(tmp_path)
        m = _DeterministicMatcher(ck, fields=["name"])
        cands = _make_candidates()
        left, right = _make_sources()
        before = m._cache_key(cands, left, right, "id")
        # Force mtime forward by ≥1s and re-stat.
        time.sleep(1.05)
        ck.touch()
        # Drop any cached stat — Path doesn't cache by default.
        after = m._cache_key(cands, left, right, "id")
        assert before != after

    def test_value_change_invalidates(self, tmp_path: Path) -> None:
        """R10-F: same pairs + perturbed source values => different key.

        This is the load-bearing fix — pre-R10-F the same (id1, id2) pair
        scored against perturbed records across variant levels reused the
        first level's cached score (flat ditto_plm curve).
        """
        ck = _checkpoint_dir(tmp_path)
        m = _DeterministicMatcher(ck, fields=["name"])
        cands = _make_candidates()
        left, right = _make_sources()
        before = m._cache_key(cands, left, right, "id")
        # Perturb a value behind one candidate pair; ids are unchanged.
        right_perturbed = right.copy()
        right_perturbed.loc[right_perturbed["id"] == "R1", "name"] = "alpha CORRUPTED"
        after = m._cache_key(cands, left, right_perturbed, "id")
        assert before != after

    def test_identical_values_same_key_fresh_frames(self, tmp_path: Path) -> None:
        """Two independently built frames with identical content => same key."""
        ck = _checkpoint_dir(tmp_path)
        m = _DeterministicMatcher(ck, fields=["name"])
        cands = _make_candidates()
        left1, right1 = _make_sources()
        left2, right2 = _make_sources()
        assert m._cache_key(cands, left1, right1, "id") == m._cache_key(
            cands, left2, right2, "id"
        )

    def test_value_change_outside_candidate_pairs_still_invalidates(
        self, tmp_path: Path
    ) -> None:
        """A value change on a record referenced by a pair changes the key;
        changing an unreferenced record does not (defensive: confirms the
        key is driven by the pairs' records, not the whole frame blindly)."""
        ck = _checkpoint_dir(tmp_path)
        m = _DeterministicMatcher(ck, fields=["name"])
        # Candidates only reference R1/R2 on the right.
        cands = pd.DataFrame({"id1": ["L1", "L2"], "id2": ["R1", "R2"]})
        left, right = _make_sources()
        baseline = m._cache_key(cands, left, right, "id")
        # R3 is not referenced by any pair -> key unchanged.
        right_r3 = right.copy()
        right_r3.loc[right_r3["id"] == "R3", "name"] = "delta CHANGED"
        assert m._cache_key(cands, left, right_r3, "id") == baseline
        # R2 IS referenced -> key changes.
        right_r2 = right.copy()
        right_r2.loc[right_r2["id"] == "R2", "name"] = "beta CHANGED"
        assert m._cache_key(cands, left, right_r2, "id") != baseline


# ---------------------------------------------------------------------------
# Resume behaviour
# ---------------------------------------------------------------------------


class TestInferenceCache:
    def test_cache_dir_false_disables_caching(self, tmp_path: Path) -> None:
        """``cache_dir=False`` opts out of caching entirely."""
        ck = _checkpoint_dir(tmp_path)
        m = _DeterministicMatcher(ck, fields=["name"], cache_dir=False)
        assert m.cache_dir is None
        left, right = _make_sources()
        cands = _make_candidates()
        out = m.match(left, right, cands, id_column="id", threshold=0.5)
        assert len(out) == 5  # all pairs above 0.5
        # Nothing written.
        if (tmp_path / "ckpt").exists():
            files = [p for p in (tmp_path / "ckpt").glob("ditto_inference_*.csv")]
            assert files == []

    def test_cache_dir_default_resolves_to_synthetic_path(self, tmp_path: Path) -> None:
        """``cache_dir=None`` (the default) resolves to the synthetic-local path."""
        ck = _checkpoint_dir(tmp_path)
        m = _DeterministicMatcher(ck, fields=["name"])
        assert m.cache_dir is not None
        assert m.cache_dir.name == "ditto_inference"
        # Walks up to usecases_synthetic/cache/ditto_inference.
        assert m.cache_dir.parent.name == "cache"
        assert m.cache_dir.parent.parent.name == "usecases_synthetic"

    def test_first_run_writes_cache(self, tmp_path: Path) -> None:
        ck = _checkpoint_dir(tmp_path)
        cache_dir = tmp_path / "infcache"
        m = _DeterministicMatcher(
            ck, fields=["name"], batch_size=2, cache_dir=cache_dir
        )
        left, right = _make_sources()
        cands = _make_candidates()
        m.match(left, right, cands, id_column="id", threshold=0.5)

        files = list(cache_dir.glob("ditto_inference_*.csv"))
        assert len(files) == 1
        df = pd.read_csv(files[0])
        # All 5 pairs cached.
        assert len(df) == 5
        assert set(df.columns) == {"id1", "id2", "score"}

    def test_resume_skips_already_scored_pairs(self, tmp_path: Path) -> None:
        """Second match() with cache present scores zero new pairs."""
        ck = _checkpoint_dir(tmp_path)
        cache_dir = tmp_path / "infcache"
        m1 = _DeterministicMatcher(
            ck, fields=["name"], batch_size=2, cache_dir=cache_dir
        )
        left, right = _make_sources()
        cands = _make_candidates()
        m1.match(left, right, cands, id_column="id", threshold=0.5)

        # Fresh matcher, same cache dir → should reuse cache entirely.
        m2 = _DeterministicMatcher(
            ck, fields=["name"], batch_size=2, cache_dir=cache_dir
        )
        out = m2.match(left, right, cands, id_column="id", threshold=0.5)

        assert m2.batch_calls == []  # zero new inference batches
        assert len(out) == 5  # same output

    def test_partial_cache_resumes_remaining_pairs(self, tmp_path: Path) -> None:
        """Cache with only some pairs → only the missing ones get scored."""
        ck = _checkpoint_dir(tmp_path)
        cache_dir = tmp_path / "infcache"

        # First call: a subset of candidates (3 pairs).
        m1 = _DeterministicMatcher(
            ck, fields=["name"], batch_size=2, cache_dir=cache_dir
        )
        left, right = _make_sources()
        partial = _make_candidates().head(3)
        m1.match(left, right, partial, id_column="id", threshold=0.5)
        # Cache for the partial candidate set lives at one hash.
        assert len(list(cache_dir.glob("ditto_inference_*.csv"))) == 1

        # Second call: full 5-pair set → new cache file (different hash),
        # forcing all 5 to score from scratch.
        m2 = _DeterministicMatcher(
            ck, fields=["name"], batch_size=2, cache_dir=cache_dir
        )
        full = _make_candidates()
        m2.match(left, right, full, id_column="id", threshold=0.5)
        assert sum(len(c) for c in m2.batch_calls) == 5

        # Third call on the SAME full set: should resume from m2's cache.
        m3 = _DeterministicMatcher(
            ck, fields=["name"], batch_size=2, cache_dir=cache_dir
        )
        m3.match(left, right, full, id_column="id", threshold=0.5)
        assert m3.batch_calls == []

    def test_threshold_change_does_not_invalidate_cache(self, tmp_path: Path) -> None:
        """Threshold filter is applied at return time, so cache survives
        a threshold change between runs."""
        ck = _checkpoint_dir(tmp_path)
        cache_dir = tmp_path / "infcache"

        # Score everything at 0.7 (deterministic).
        m1 = _DeterministicMatcher(
            ck,
            fields=["name"],
            batch_size=2,
            cache_dir=cache_dir,
            score_fn=lambda l, r: 0.7,
        )
        left, right = _make_sources()
        cands = _make_candidates()
        out_loose = m1.match(left, right, cands, id_column="id", threshold=0.5)
        assert len(out_loose) == 5  # 0.7 >= 0.5

        # Re-run with a stricter threshold; no new inference, output filtered.
        m2 = _DeterministicMatcher(
            ck,
            fields=["name"],
            batch_size=2,
            cache_dir=cache_dir,
            score_fn=lambda l, r: 0.7,
        )
        out_strict = m2.match(left, right, cands, id_column="id", threshold=0.8)
        assert m2.batch_calls == []
        assert len(out_strict) == 0  # 0.7 < 0.8

    def test_truncated_cache_line_tolerated(self, tmp_path: Path) -> None:
        """A SIGKILL mid-write can truncate the last cache row; the
        loader skips it and the next run re-scores the partial pair."""
        ck = _checkpoint_dir(tmp_path)
        cache_dir = tmp_path / "infcache"
        cache_dir.mkdir()

        # Write a cache file with 1 valid row and a truncated trailing one.
        m_for_key = _DeterministicMatcher(ck, fields=["name"], cache_dir=cache_dir)
        cands = _make_candidates()
        left, right = _make_sources()
        path = m_for_key._cache_path_for(cands, left, right, "id")
        path.write_text(
            "id1,id2,score\n"
            "L1,R1,0.91\n"
            "L1,R2,0.7\n"
            "L2,R1\n",  # truncated last row — only 2 fields
            encoding="utf-8",
        )

        m = _DeterministicMatcher(
            ck, fields=["name"], batch_size=2, cache_dir=cache_dir
        )
        out = m.match(left, right, cands, id_column="id", threshold=0.5)

        # 2 cached + 3 newly scored = 5 total; only 1 batch worth of
        # work avoided (the truncated row gets re-scored).
        scored = sum(len(c) for c in m.batch_calls)
        assert scored == 3  # L2-R1 (truncated) + L2-R2 + L3-R3

    def test_invalid_cache_header_triggers_full_rescore(self, tmp_path: Path) -> None:
        """A cache file with the wrong header (e.g. older schema) is
        ignored and treated as empty so the run continues correctly."""
        ck = _checkpoint_dir(tmp_path)
        cache_dir = tmp_path / "infcache"
        cache_dir.mkdir()

        m_for_key = _DeterministicMatcher(ck, fields=["name"], cache_dir=cache_dir)
        cands = _make_candidates()
        left, right = _make_sources()
        path = m_for_key._cache_path_for(cands, left, right, "id")
        path.write_text("a,b,c\nL1,R1,0.9\n", encoding="utf-8")

        m = _DeterministicMatcher(
            ck, fields=["name"], batch_size=2, cache_dir=cache_dir
        )
        m.match(left, right, cands, id_column="id", threshold=0.5)
        # All 5 pairs were re-scored.
        assert sum(len(c) for c in m.batch_calls) == 5
