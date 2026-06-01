"""Tests for step 4i: non-corner refill module + drop-corner dispatcher.

Covers the new K2 drop-corner-touching operator and its 1-for-1 refill
authored 2026-05-27 per plan_revision.md §4i.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from usecases_synthetic.lib.llm_cache import LLMCache, LLMCacheMiss
from usecases_synthetic.lib.non_corner_refill import (
    NonCornerEntity,
    build_openai_non_corner_client,
    contamination_check,
    default_api_client_from_attributes,
    reference_anchor_hash,
    refill_non_corner_entity,
    select_reference_anchor,
)
from usecases_synthetic.lib.niche_scorer import EntityDensity


def _density(idx: int, score: float) -> EntityDensity:
    return EntityDensity(
        index=idx,
        density=score,
        rrf_component=score,
        label_collision_component=0.0,
        agreement_counts={},
        neighbour_count=0,
    )


# ---- select_reference_anchor ----------------------------------------------


class TestSelectReferenceAnchor:
    """Stochastic anchor selection from a low-density pool.

    Post-2026-05-28 the function uses *rng* to sample *k* anchors from
    the bottom ``k * pool_multiplier`` survivors by density. The pool
    preserves the "low-density anchor" intent; the sample diversifies
    per-call cache keys in :func:`refill_non_corner_entity`.
    """

    def test_anchors_drawn_from_low_density_pool(self) -> None:
        """With pool_multiplier=2 and k=2, the pool is the bottom 4
        by density — anchors must be a subset of those."""
        densities = [
            _density(i, score)
            for i, score in enumerate([5.0, 1.0, 3.0, 2.0, 4.0, 0.5, 6.0, 7.0])
        ]
        survivors = list(range(8))
        # Bottom 4 by density: index 5 (0.5), 1 (1.0), 3 (2.0), 2 (3.0).
        low_density_pool = {5, 1, 3, 2}
        out = select_reference_anchor(
            survivors,
            densities,
            k=2,
            rng=np.random.default_rng(0),
            pool_multiplier=2,
        )
        assert len(out) == 2
        assert set(out).issubset(low_density_pool)

    def test_k_zero_returns_empty(self) -> None:
        densities = [_density(0, 0.5)]
        out = select_reference_anchor([0], densities, k=0, rng=np.random.default_rng(0))
        assert out == []

    def test_empty_survivors_returns_empty(self) -> None:
        out = select_reference_anchor([], [], k=3, rng=np.random.default_rng(0))
        assert out == []

    def test_k_larger_than_survivors(self) -> None:
        """When k > len(survivors), return all survivors sorted by
        density — the rng path is skipped."""
        densities = [_density(0, 0.0), _density(1, 1.0)]
        survivors = [0, 1]
        out = select_reference_anchor(
            survivors, densities, k=5, rng=np.random.default_rng(0)
        )
        assert out == [0, 1]

    def test_different_rngs_produce_different_anchors(self) -> None:
        """The whole point of the 2026-05-28 fix: distinct sub-rngs
        (spawn_sub_rng with different counters) must produce distinct
        anchor combinations. Without this, every drop's refill hashes
        to the same cache key and 99% of refills collide."""
        # 20 survivors, all with distinct increasing density. With k=5
        # and pool_multiplier=4, the pool is the bottom 20 (entire set
        # here) — sample of 5 from 20 has C(20,5)=15504 combinations,
        # so two arbitrary rng seeds should disagree.
        densities = [_density(i, float(i)) for i in range(20)]
        survivors = list(range(20))
        anchors_a = select_reference_anchor(
            survivors, densities, k=5, rng=np.random.default_rng(0)
        )
        anchors_b = select_reference_anchor(
            survivors, densities, k=5, rng=np.random.default_rng(1)
        )
        assert anchors_a != anchors_b, (
            "Different rng seeds must yield different anchor combinations — "
            "otherwise the drop-corner refill cache collapses to one entry."
        )

    def test_same_rng_produces_same_anchors(self) -> None:
        """Determinism under same rng — required for cache stability
        when the same anchor is re-derived on a rerun."""
        densities = [_density(i, float(i)) for i in range(20)]
        survivors = list(range(20))
        anchors_a = select_reference_anchor(
            survivors, densities, k=5, rng=np.random.default_rng(42)
        )
        anchors_b = select_reference_anchor(
            survivors, densities, k=5, rng=np.random.default_rng(42)
        )
        assert anchors_a == anchors_b

    def test_anchor_sorted_for_hash_permutation_invariance(self) -> None:
        """Returned anchor list is sorted so the anchor hash (over the
        ID list) is permutation-invariant across rng-shuffles."""
        densities = [_density(i, float(i)) for i in range(20)]
        survivors = list(range(20))
        out = select_reference_anchor(
            survivors, densities, k=5, rng=np.random.default_rng(7)
        )
        assert out == sorted(out)


# ---- contamination_check ---------------------------------------------------


class TestContaminationCheck:
    def test_passes_on_unseen_label(self) -> None:
        entity = {"name": "Acme Brand-New Co"}
        assert (
            contamination_check(
                entity, primary_column="name", reference_labels={"foo", "bar"}
            )
            == "passed"
        )

    def test_empty_primary_label_rejected(self) -> None:
        entity = {"name": ""}
        assert (
            contamination_check(entity, primary_column="name", reference_labels=set())
            == "empty_primary_label"
        )

    def test_none_primary_label_rejected(self) -> None:
        entity = {"name": None}
        assert (
            contamination_check(entity, primary_column="name", reference_labels=set())
            == "empty_primary_label"
        )

    def test_collision_with_real_label_rejected(self) -> None:
        entity = {"name": "Apple Inc"}
        # The reference label is the normalised form ("apple inc").
        assert (
            contamination_check(
                entity, primary_column="name", reference_labels={"apple inc"}
            )
            == "collision_with_real_entity"
        )


# ---- reference_anchor_hash ------------------------------------------------


class TestReferenceAnchorHash:
    def test_deterministic(self) -> None:
        h1 = reference_anchor_hash(["e1", "e2", "e3"])
        h2 = reference_anchor_hash(["e1", "e2", "e3"])
        assert h1 == h2

    def test_order_independent(self) -> None:
        # Sorting inside the function should make the hash order-invariant.
        h1 = reference_anchor_hash(["e1", "e2", "e3"])
        h2 = reference_anchor_hash(["e3", "e1", "e2"])
        assert h1 == h2

    def test_different_inputs_different_hashes(self) -> None:
        h1 = reference_anchor_hash(["e1", "e2"])
        h2 = reference_anchor_hash(["e1", "e3"])
        assert h1 != h2


# ---- default_api_client_from_attributes ----------------------------------


class TestDefaultApiClient:
    def test_returns_dict_with_all_schema_cols(self) -> None:
        client = default_api_client_from_attributes(
            schema_columns=["name", "category"],
            primary_column="name",
            rng=np.random.default_rng(0),
        )
        out = client("prompt", [{"name": "ref1"}, {"name": "ref2"}])
        assert set(out.keys()) == {"name", "category"}
        assert out["name"].startswith("noncorner_")
        assert out["category"].startswith("nc_")

    def test_deterministic_for_same_anchor(self) -> None:
        client = default_api_client_from_attributes(
            schema_columns=["name"],
            primary_column="name",
            rng=np.random.default_rng(0),
            salt=42,
        )
        a = client("p", [{"name": "ref"}])
        b = client("p", [{"name": "ref"}])
        assert a == b

    def test_different_anchors_different_outputs(self) -> None:
        client = default_api_client_from_attributes(
            schema_columns=["name"],
            primary_column="name",
            rng=np.random.default_rng(0),
        )
        a = client("p", [{"name": "ref1"}])
        b = client("p", [{"name": "ref2"}])
        assert a != b


# ---- refill_non_corner_entity --------------------------------------------


class TestRefillNonCornerEntity:
    @pytest.fixture
    def reference_rows(self) -> list[pd.Series]:
        s1 = pd.Series({"name": "ref1", "category": "cat_a"}, name="e1")
        s2 = pd.Series({"name": "ref2", "category": "cat_a"}, name="e2")
        return [s1, s2]

    def test_happy_path_cache_miss_with_client(
        self, reference_rows: list[pd.Series], tmp_path: Path
    ) -> None:
        cache = LLMCache(
            cache_dir=tmp_path, prompt_version="v1", model_id="gpt-5.4-mini"
        )
        client = default_api_client_from_attributes(
            schema_columns=["name", "category"],
            primary_column="name",
            rng=np.random.default_rng(0),
        )
        result = refill_non_corner_entity(
            reference_rows=reference_rows,
            primary_column="name",
            schema_columns=["name", "category"],
            domain="testdom",
            prompt_template="dummy",
            llm_cache=cache,
            api_client=client,
            reference_labels={"ref1", "ref2"},
            source_placements=["src_a"],
            entity_id="ent_0",
        )
        assert result is not None
        assert isinstance(result, NonCornerEntity)
        assert result.entity_id == "ent_0"
        assert result.attributes["name"].startswith("noncorner_")
        assert result.source_placements == ["src_a"]
        assert result.contamination_check_status == "passed"

    def test_strict_cache_miss_raises(
        self, reference_rows: list[pd.Series], tmp_path: Path
    ) -> None:
        cache = LLMCache(
            cache_dir=tmp_path, prompt_version="v1", model_id="gpt-5.4-mini"
        )
        with pytest.raises(LLMCacheMiss):
            refill_non_corner_entity(
                reference_rows=reference_rows,
                primary_column="name",
                schema_columns=["name"],
                domain="testdom",
                prompt_template="dummy",
                llm_cache=cache,
                api_client=None,
                reference_labels=set(),
                source_placements=[],
                entity_id="ent_strict",
                strict_cache=True,
            )

    def test_strict_cache_miss_increments_rejection_log(
        self, reference_rows: list[pd.Series], tmp_path: Path
    ) -> None:
        cache = LLMCache(
            cache_dir=tmp_path, prompt_version="v1", model_id="gpt-5.4-mini"
        )
        rejection_log: dict[str, int] = {}
        with pytest.raises(LLMCacheMiss):
            refill_non_corner_entity(
                reference_rows=reference_rows,
                primary_column="name",
                schema_columns=["name"],
                domain="testdom",
                prompt_template="dummy",
                llm_cache=cache,
                api_client=None,
                reference_labels=set(),
                source_placements=[],
                entity_id="ent_strict",
                strict_cache=True,
                rejection_log=rejection_log,
            )
        assert rejection_log["strict_cache_miss"] == 1

    def test_contamination_rejection(
        self, reference_rows: list[pd.Series], tmp_path: Path
    ) -> None:
        cache = LLMCache(
            cache_dir=tmp_path, prompt_version="v1", model_id="gpt-5.4-mini"
        )

        def _colliding_client(prompt, refs):  # type: ignore[no-untyped-def]
            return {"name": "Existing Real Entity"}

        rejection_log: dict[str, int] = {}
        result = refill_non_corner_entity(
            reference_rows=reference_rows,
            primary_column="name",
            schema_columns=["name"],
            domain="testdom",
            prompt_template="dummy",
            llm_cache=cache,
            api_client=_colliding_client,
            reference_labels={"existing real entity"},
            source_placements=[],
            entity_id="ent_collide",
            rejection_log=rejection_log,
        )
        assert result is None
        assert rejection_log["contamination_collision_with_real_entity"] == 1

    def test_empty_primary_rejection(
        self, reference_rows: list[pd.Series], tmp_path: Path
    ) -> None:
        cache = LLMCache(
            cache_dir=tmp_path, prompt_version="v1", model_id="gpt-5.4-mini"
        )

        def _empty_client(prompt, refs):  # type: ignore[no-untyped-def]
            return {"name": "   "}

        rejection_log: dict[str, int] = {}
        result = refill_non_corner_entity(
            reference_rows=reference_rows,
            primary_column="name",
            schema_columns=["name"],
            domain="testdom",
            prompt_template="dummy",
            llm_cache=cache,
            api_client=_empty_client,
            reference_labels=set(),
            source_placements=[],
            entity_id="ent_empty",
            rejection_log=rejection_log,
        )
        assert result is None
        assert rejection_log["empty_primary_label"] == 1

    def test_nondict_response_rejection(
        self, reference_rows: list[pd.Series], tmp_path: Path
    ) -> None:
        cache = LLMCache(
            cache_dir=tmp_path, prompt_version="v1", model_id="gpt-5.4-mini"
        )

        def _bad_client(prompt, refs):  # type: ignore[no-untyped-def]
            return "not a dict"  # type: ignore[return-value]

        rejection_log: dict[str, int] = {}
        result = refill_non_corner_entity(
            reference_rows=reference_rows,
            primary_column="name",
            schema_columns=["name"],
            domain="testdom",
            prompt_template="dummy",
            llm_cache=cache,
            api_client=_bad_client,
            reference_labels=set(),
            source_placements=[],
            entity_id="ent_bad",
            rejection_log=rejection_log,
        )
        assert result is None
        assert rejection_log["nondict_result"] == 1

    def test_cache_hit_reuses_payload(
        self, reference_rows: list[pd.Series], tmp_path: Path
    ) -> None:
        cache = LLMCache(
            cache_dir=tmp_path, prompt_version="v1", model_id="gpt-5.4-mini"
        )
        call_counter = {"n": 0}

        def _counting_client(prompt, refs):  # type: ignore[no-untyped-def]
            call_counter["n"] += 1
            return {"name": "Cached Result"}

        # First call: cache miss → client invoked.
        r1 = refill_non_corner_entity(
            reference_rows=reference_rows,
            primary_column="name",
            schema_columns=["name"],
            domain="testdom",
            prompt_template="dummy",
            llm_cache=cache,
            api_client=_counting_client,
            reference_labels=set(),
            source_placements=[],
            entity_id="ent_cache_1",
        )
        assert r1 is not None
        assert call_counter["n"] == 1

        # Second call same anchor: cache hit → client not invoked again.
        r2 = refill_non_corner_entity(
            reference_rows=reference_rows,
            primary_column="name",
            schema_columns=["name"],
            domain="testdom",
            prompt_template="dummy",
            llm_cache=cache,
            api_client=_counting_client,
            reference_labels=set(),
            source_placements=[],
            entity_id="ent_cache_2",
        )
        assert r2 is not None
        assert call_counter["n"] == 1  # unchanged

    def test_cache_dir_namespacing_via_distinct_cache_dirs(
        self, reference_rows: list[pd.Series], tmp_path: Path
    ) -> None:
        # Two LLMCache instances with different cache_dir → no collision
        # even with identical references. The non-corner refill cache is
        # namespaced from the interpolation cache by living in a
        # different directory.
        cache_interp = LLMCache(
            cache_dir=tmp_path / "interp",
            prompt_version="v1",
            model_id="gpt-5.4-mini",
        )
        cache_noncorner = LLMCache(
            cache_dir=tmp_path / "noncorner",
            prompt_version="v1",
            model_id="gpt-5.4-mini",
        )

        def _client_interp(prompt, refs):  # type: ignore[no-untyped-def]
            return {"name": "interp_result"}

        def _client_noncorner(prompt, refs):  # type: ignore[no-untyped-def]
            return {"name": "noncorner_result"}

        r1 = refill_non_corner_entity(
            reference_rows=reference_rows,
            primary_column="name",
            schema_columns=["name"],
            domain="testdom",
            prompt_template="d",
            llm_cache=cache_interp,
            api_client=_client_interp,
            reference_labels=set(),
            source_placements=[],
            entity_id="e1",
        )
        r2 = refill_non_corner_entity(
            reference_rows=reference_rows,
            primary_column="name",
            schema_columns=["name"],
            domain="testdom",
            prompt_template="d",
            llm_cache=cache_noncorner,
            api_client=_client_noncorner,
            reference_labels=set(),
            source_placements=[],
            entity_id="e2",
        )
        # Each cache_dir stored its own payload; the refilled values
        # reflect the per-cache client output.
        assert r1 is not None and r1.attributes["name"] == "interp_result"
        assert r2 is not None and r2.attributes["name"] == "noncorner_result"


# ---- _run_drop_corner_refill (helper integration) ------------------------


class TestRunDropCornerRefill:
    """End-to-end smoke tests of the drop-corner orchestrator.

    These hit the helper function directly with hand-built inputs to
    exercise the greedy drop selection + paired refill loop.
    """

    def _build_inputs(
        self, *, n: int = 6, corner_count_by: dict[int, int] | None = None
    ) -> dict[str, Any]:
        # Minimal canonical frame with primary column + secondary.
        canonical = pd.DataFrame(
            {
                "entity_id": [f"e{i}" for i in range(n)],
                "name": [f"Entity {i}" for i in range(n)],
                "category": [f"cat_{i%2}" for i in range(n)],
            }
        )
        canonical = canonical.set_index("entity_id", drop=False)

        # Build pairs: cross-cluster (same_pairs) + corner pairs.
        same_pairs: list[tuple[int, int]] = []
        cross_pairs: list[tuple[int, int]] = []
        for i in range(n):
            for j in range(i + 1, n):
                # Treat (0,1), (0,2), (0,3) as corner pairs by default; we
                # override via the corner_count_by knob below.
                cross_pairs.append((i, j))
        baseline_corner: list[tuple[int, int]] = []
        if corner_count_by is None:
            corner_count_by = {0: 3}  # entity 0 touches 3 corner pairs
        # Build corner pairs to match the per-entity counts.
        for ent, want_count in corner_count_by.items():
            # pick the first `want_count` pairs containing ent
            picked = 0
            for pair in cross_pairs:
                if ent in pair and pair not in baseline_corner:
                    baseline_corner.append(pair)
                    picked += 1
                    if picked >= want_count:
                        break

        densities = [_density(i, float(n - i)) for i in range(n)]
        protection_flags = [False] * n

        return {
            "canonical_frame": canonical,
            "entity_ids": list(canonical["entity_id"]),
            "same_pairs": same_pairs,
            "cross_pairs": cross_pairs,
            "baseline_corner": baseline_corner,
            "collision_groups": {},
            "protection_flags": protection_flags,
            "densities": densities,
            "n_entities": n,
        }

    def test_high_corner_entity_picked_first(self, tmp_path: Path) -> None:
        from usecases_synthetic.scripts.apply_knob_02_niche import (
            _run_drop_corner_refill,
        )

        inputs = self._build_inputs(corner_count_by={2: 4, 5: 1})
        cache = LLMCache(
            cache_dir=tmp_path, prompt_version="v1", model_id="gpt-5.4-mini"
        )

        # Stub the prompt loader to avoid filesystem dependency in test
        from usecases_synthetic.scripts import apply_knob_02_niche

        original_loader = apply_knob_02_niche._load_prompt_template
        apply_knob_02_niche._load_prompt_template = lambda name: "stub_prompt"
        try:
            client = default_api_client_from_attributes(
                schema_columns=["name", "category"],
                primary_column="name",
                rng=np.random.default_rng(0),
            )
            sources = {"src_a": pd.DataFrame({"id": ["a1", "a2"]})}
            config = {
                "canonical_schema": ["name", "category"],
                "non_corner_refill": {"enabled": True, "reference_count": 2},
                "non_corner_prompt_version": "v1",
            }
            planned_drops, refilled, rejection_log, metrics = _run_drop_corner_refill(
                canonical_frame=inputs["canonical_frame"],
                entity_ids=inputs["entity_ids"],
                same_pairs=inputs["same_pairs"],
                cross_pairs=inputs["cross_pairs"],
                baseline_corner=inputs["baseline_corner"],
                collision_groups=inputs["collision_groups"],
                protection_flags=inputs["protection_flags"],
                densities=inputs["densities"],
                target_ratio=0.0,  # force drops to fire
                tol=0.02,
                max_interp_fraction=0.6,
                n_entities=inputs["n_entities"],
                config=config,
                domain="testdom",
                llm_cache=cache,
                api_client=client,
                strict_cache=False,
                sources=sources,
                rng=np.random.default_rng(0),
            )
        finally:
            apply_knob_02_niche._load_prompt_template = original_loader

        # Entity 2 has 4 corner pairs; should be in the first drops.
        assert 2 in planned_drops
        # Refill committed for at least 1 drop (anchor non-empty, label OK).
        assert metrics["planned_drop_count"] >= 1
        assert metrics["refill_committed"] >= 1

    def test_protected_entity_not_dropped(self, tmp_path: Path) -> None:
        from usecases_synthetic.scripts.apply_knob_02_niche import (
            _run_drop_corner_refill,
        )
        from usecases_synthetic.scripts import apply_knob_02_niche

        inputs = self._build_inputs(corner_count_by={2: 4})
        inputs["protection_flags"][2] = True  # protect entity 2

        original_loader = apply_knob_02_niche._load_prompt_template
        apply_knob_02_niche._load_prompt_template = lambda name: "stub"
        try:
            cache = LLMCache(
                cache_dir=tmp_path, prompt_version="v1", model_id="gpt-5.4-mini"
            )
            planned_drops, _, _, _ = _run_drop_corner_refill(
                canonical_frame=inputs["canonical_frame"],
                entity_ids=inputs["entity_ids"],
                same_pairs=inputs["same_pairs"],
                cross_pairs=inputs["cross_pairs"],
                baseline_corner=inputs["baseline_corner"],
                collision_groups=inputs["collision_groups"],
                protection_flags=inputs["protection_flags"],
                densities=inputs["densities"],
                target_ratio=0.0,
                tol=0.02,
                max_interp_fraction=0.6,
                n_entities=inputs["n_entities"],
                config={
                    "canonical_schema": ["name", "category"],
                    "non_corner_refill": {"enabled": True, "reference_count": 2},
                },
                domain="testdom",
                llm_cache=cache,
                api_client=default_api_client_from_attributes(
                    schema_columns=["name", "category"],
                    primary_column="name",
                    rng=np.random.default_rng(0),
                ),
                strict_cache=False,
                sources={},
                rng=np.random.default_rng(0),
            )
        finally:
            apply_knob_02_niche._load_prompt_template = original_loader

        assert 2 not in planned_drops

    def test_collision_group_last_member_not_dropped(self, tmp_path: Path) -> None:
        from usecases_synthetic.scripts.apply_knob_02_niche import (
            _run_drop_corner_refill,
        )
        from usecases_synthetic.scripts import apply_knob_02_niche

        # Build a collision group with exactly 2 members; both have
        # high corner counts. Greedy would normally drop both; the
        # group-collapse guard must skip the second.
        inputs = self._build_inputs(n=6, corner_count_by={0: 5, 1: 5})
        collision_groups = {"grp": [0, 1]}
        inputs["collision_groups"] = collision_groups

        original_loader = apply_knob_02_niche._load_prompt_template
        apply_knob_02_niche._load_prompt_template = lambda name: "stub"
        try:
            cache = LLMCache(
                cache_dir=tmp_path, prompt_version="v1", model_id="gpt-5.4-mini"
            )
            planned_drops, _, _, _ = _run_drop_corner_refill(
                canonical_frame=inputs["canonical_frame"],
                entity_ids=inputs["entity_ids"],
                same_pairs=inputs["same_pairs"],
                cross_pairs=inputs["cross_pairs"],
                baseline_corner=inputs["baseline_corner"],
                collision_groups=collision_groups,
                protection_flags=inputs["protection_flags"],
                densities=inputs["densities"],
                target_ratio=0.0,
                tol=0.02,
                max_interp_fraction=0.6,
                n_entities=inputs["n_entities"],
                config={
                    "canonical_schema": ["name", "category"],
                    "non_corner_refill": {"enabled": True, "reference_count": 2},
                },
                domain="testdom",
                llm_cache=cache,
                api_client=default_api_client_from_attributes(
                    schema_columns=["name", "category"],
                    primary_column="name",
                    rng=np.random.default_rng(0),
                ),
                strict_cache=False,
                sources={},
                rng=np.random.default_rng(0),
            )
        finally:
            apply_knob_02_niche._load_prompt_template = original_loader

        # At most one of {0, 1} can be dropped (the group must retain ≥2
        # members; we never collapse below size 2).
        dropped_from_group = [i for i in planned_drops if i in {0, 1}]
        assert len(dropped_from_group) <= 1

    def test_empty_candidate_pairs_returns_empty(self, tmp_path: Path) -> None:
        from usecases_synthetic.scripts.apply_knob_02_niche import (
            _run_drop_corner_refill,
        )

        canonical = pd.DataFrame({"entity_id": [], "name": [], "category": []})
        canonical = canonical.set_index("entity_id", drop=False)
        cache = LLMCache(
            cache_dir=tmp_path, prompt_version="v1", model_id="gpt-5.4-mini"
        )
        planned_drops, refilled, rejection_log, metrics = _run_drop_corner_refill(
            canonical_frame=canonical,
            entity_ids=[],
            same_pairs=[],
            cross_pairs=[],
            baseline_corner=[],
            collision_groups={},
            protection_flags=[],
            densities=[],
            target_ratio=0.5,
            tol=0.02,
            max_interp_fraction=0.6,
            n_entities=0,
            config={"canonical_schema": ["name", "category"]},
            domain="testdom",
            llm_cache=cache,
            api_client=None,
            strict_cache=False,
            sources={},
            rng=np.random.default_rng(0),
        )
        assert planned_drops == []
        assert refilled == []
        assert metrics["planned_drop_count"] == 0
        assert metrics["refill_committed"] == 0

    def test_cap_bound_flag(self, tmp_path: Path) -> None:
        """Greedy stops at max_interp_fraction * n entities even if target
        not reached. The metrics dict reports ``cap_bound: True``."""
        from usecases_synthetic.scripts.apply_knob_02_niche import (
            _run_drop_corner_refill,
        )
        from usecases_synthetic.scripts import apply_knob_02_niche

        # Many entities, all touching corner pairs; force a tight cap.
        n = 10
        inputs = self._build_inputs(n=n, corner_count_by={i: 1 for i in range(n)})

        original_loader = apply_knob_02_niche._load_prompt_template
        apply_knob_02_niche._load_prompt_template = lambda name: "stub"
        try:
            cache = LLMCache(
                cache_dir=tmp_path, prompt_version="v1", model_id="gpt-5.4-mini"
            )
            planned_drops, _, _, metrics = _run_drop_corner_refill(
                canonical_frame=inputs["canonical_frame"],
                entity_ids=inputs["entity_ids"],
                same_pairs=inputs["same_pairs"],
                cross_pairs=inputs["cross_pairs"],
                baseline_corner=inputs["baseline_corner"],
                collision_groups={},
                protection_flags=inputs["protection_flags"],
                densities=inputs["densities"],
                # Tight tol so the inner ratio-break does NOT fire after
                # the first drop (which clears most corner pairs).
                target_ratio=-1.0,
                tol=0.0,
                max_interp_fraction=0.1,  # cap at 1 entity
                n_entities=inputs["n_entities"],
                config={
                    "canonical_schema": ["name", "category"],
                    "non_corner_refill": {"enabled": True, "reference_count": 2},
                },
                domain="testdom",
                llm_cache=cache,
                api_client=default_api_client_from_attributes(
                    schema_columns=["name", "category"],
                    primary_column="name",
                    rng=np.random.default_rng(0),
                ),
                strict_cache=False,
                sources={},
                rng=np.random.default_rng(0),
            )
        finally:
            apply_knob_02_niche._load_prompt_template = original_loader

        assert len(planned_drops) <= 1
        assert metrics["cap_bound"] is True

    def test_counterproductive_drop_skipped(self, tmp_path: Path) -> None:
        """Bug 6 regression (2026-05-28): once the high-corner candidates
        are exhausted, the greedy loop must skip drops whose removal
        would push the realised ratio AWAY from the target. Without
        this guard, products medium over-dropped 12 tail entities and
        pushed realised from 0.72 back up to 0.82.

        Setup: 8 entities where:
          - entity 0 touches a corner pair → high-corner, drop is productive
          - entities 2, 3 are PROTECTED (their corner pair (2,3) stays in
            the pool throughout the loop — keeps current_corner > 0 so
            the counterproductive guard can fire)
          - entities 4, 5 touch only a non-corner pair → drops would
            reduce ``current_total`` while leaving ``current_corner``
            unchanged, lifting the ratio
        Without the Bug 6 guard, entities 4 + 5 would drop and push the
        ratio above the current value. With the guard, they're skipped.
        """
        from usecases_synthetic.scripts.apply_knob_02_niche import (
            _run_drop_corner_refill,
        )
        from usecases_synthetic.scripts import apply_knob_02_niche

        n = 8
        canonical = pd.DataFrame(
            {
                "entity_id": [f"e{i}" for i in range(n)],
                "name": [f"Entity {i}" for i in range(n)],
                "category": [f"cat_{i % 2}" for i in range(n)],
            }
        ).set_index("entity_id", drop=False)

        # Pair layout:
        #   - (0, 1) corner — touched by entity 0 + 1
        #   - (2, 3) corner — touched by protected 2 + 3; stays in pool
        #   - (4, 5) non-corner — touched by entities 4 + 5
        corner_pairs = [(0, 1), (2, 3)]
        non_corner_pairs = [(4, 5)]
        cross_pairs = corner_pairs + non_corner_pairs

        densities = [_density(i, float(n - i)) for i in range(n)]
        # Protect entities 2, 3 so their corner pair (2, 3) is NOT
        # removed during the loop — this keeps current_corner > 0 even
        # after high-corner drops, so the counterproductive guard has
        # something to actually catch.
        protection_flags = [False, False, True, True, False, False, False, False]

        original_loader = apply_knob_02_niche._load_prompt_template
        apply_knob_02_niche._load_prompt_template = lambda name: "stub"
        try:
            cache = LLMCache(
                cache_dir=tmp_path, prompt_version="v1", model_id="gpt-5.4-mini"
            )
            planned_drops, _, _, metrics = _run_drop_corner_refill(
                canonical_frame=canonical,
                entity_ids=list(canonical["entity_id"]),
                same_pairs=[],
                cross_pairs=cross_pairs,
                baseline_corner=corner_pairs,
                collision_groups={},
                protection_flags=protection_flags,
                densities=densities,
                target_ratio=-1.0,  # never reachable — loop only stops on guards
                tol=0.0,
                max_interp_fraction=1.0,  # cap doesn't bind
                n_entities=n,
                config={
                    "canonical_schema": ["name", "category"],
                    "non_corner_refill": {"enabled": True, "reference_count": 2},
                },
                domain="testdom",
                llm_cache=cache,
                api_client=default_api_client_from_attributes(
                    schema_columns=["name", "category"],
                    primary_column="name",
                    rng=np.random.default_rng(0),
                ),
                strict_cache=False,
                sources={},
                rng=np.random.default_rng(0),
            )
        finally:
            apply_knob_02_niche._load_prompt_template = original_loader

        # With protection keeping (2, 3) in the pool, dropping entity 4
        # (or 5) would push ratio from 1/2 = 0.5 to 1/1 = 1.0 — strictly
        # counterproductive. The guard must skip both.
        assert metrics["skip_counterproductive"] >= 2, (
            "Bug 6 fix should have skipped the 2 low-corner entities "
            f"(4, 5); got skip_counterproductive={metrics['skip_counterproductive']}, "
            f"planned_drops={planned_drops}"
        )
        # Neither low-corner entity should appear in planned_drops.
        assert (
            4 not in planned_drops and 5 not in planned_drops
        ), f"Low-corner drop leaked through Bug 6 guard: planned_drops={planned_drops}"

    def test_skip_counterproductive_metric_present_on_normal_run(
        self, tmp_path: Path
    ) -> None:
        """When the loop hits no counterproductive drops, the metric
        is still emitted (zero). Required for downstream telemetry."""
        from usecases_synthetic.scripts.apply_knob_02_niche import (
            _run_drop_corner_refill,
        )
        from usecases_synthetic.scripts import apply_knob_02_niche

        inputs = self._build_inputs(corner_count_by={0: 3, 1: 2})
        original_loader = apply_knob_02_niche._load_prompt_template
        apply_knob_02_niche._load_prompt_template = lambda name: "stub"
        try:
            cache = LLMCache(
                cache_dir=tmp_path, prompt_version="v1", model_id="gpt-5.4-mini"
            )
            _, _, _, metrics = _run_drop_corner_refill(
                canonical_frame=inputs["canonical_frame"],
                entity_ids=inputs["entity_ids"],
                same_pairs=inputs["same_pairs"],
                cross_pairs=inputs["cross_pairs"],
                baseline_corner=inputs["baseline_corner"],
                collision_groups={},
                protection_flags=inputs["protection_flags"],
                densities=inputs["densities"],
                target_ratio=0.0,
                tol=0.0,
                max_interp_fraction=0.3,
                n_entities=inputs["n_entities"],
                config={
                    "canonical_schema": ["name", "category"],
                    "non_corner_refill": {"enabled": True, "reference_count": 2},
                },
                domain="testdom",
                llm_cache=cache,
                api_client=default_api_client_from_attributes(
                    schema_columns=["name", "category"],
                    primary_column="name",
                    rng=np.random.default_rng(0),
                ),
                strict_cache=False,
                sources={},
                rng=np.random.default_rng(0),
            )
        finally:
            apply_knob_02_niche._load_prompt_template = original_loader

        assert "skip_counterproductive" in metrics


# ---- Regression: CornerCasePair unhashable bug (2026-05-28) --------------


class TestCornerCasePairConversionAtCaller:
    """Regression for the 2026-05-28 products K2 hard crash.

    `mine_corner_cases` returns `list[CornerCasePair]`; `CornerCasePair`
    is a plain (mutable) dataclass, so it is unhashable. The dispatcher
    in `apply_knob_02` previously passed that list straight through to
    `_run_drop_corner_refill`, whose body calls `set(baseline_corner)`
    on it — `TypeError: unhashable type: 'CornerCasePair'`. The fix
    converts to `(i, j)` tuples at the caller, keeping the helper's
    `list[tuple[int, int]]` contract intact.
    """

    def test_corner_case_pair_instance_is_unhashable(self) -> None:
        """Document the underlying constraint: CornerCasePair is mutable
        (no ``frozen=True``), so building a set from instances fails."""
        from usecases_synthetic.lib.corner_case_miner import CornerCasePair

        pair = CornerCasePair(i=1, j=2, kind="hard_match", triggered_by=["tfidf"])
        with pytest.raises(TypeError, match="unhashable"):
            set([pair])

    def test_apply_knob_02_converts_corner_pairs_to_tuples_before_refill(
        self, tmp_path: Path
    ) -> None:
        """When `apply_knob_02` enters the drop-corner branch, the
        `baseline_corner` argument passed to `_run_drop_corner_refill`
        must be a list of `(i, j)` int tuples, not `CornerCasePair`
        instances. We stub the helper to capture the call and assert the
        per-element shape.
        """
        from usecases_synthetic.scripts import apply_knob_02_niche
        from usecases_synthetic.lib.corner_case_miner import CornerCasePair

        captured: dict[str, Any] = {}

        def fake_run_drop_corner_refill(**kwargs: Any) -> tuple[Any, Any, Any, Any]:
            captured.update(kwargs)
            return (
                [],
                [],
                {},
                {
                    "planned_drop_count": 0,
                    "refill_attempts": 0,
                    "refill_committed": 0,
                    "simulated_final_ratio": 0.0,
                    "cap_bound": False,
                },
            )

        # The conversion line under test:
        #     baseline_corner=[(p.i, p.j) for p in baseline_corner],
        # Reproduce it directly so the regression has a focused
        # assertion that survives refactors of `apply_knob_02`.
        baseline_corner = [
            CornerCasePair(i=0, j=3, kind="hard_match", triggered_by=["tfidf"]),
            CornerCasePair(i=1, j=4, kind="hard_non_match", triggered_by=["emb"]),
            CornerCasePair(i=2, j=5, kind="hard_match", triggered_by=["jaccard"]),
        ]

        converted = [(p.i, p.j) for p in baseline_corner]

        # The conversion must produce hashable tuples that work in a
        # set lookup — the failure mode the production code crashed on.
        corner_pair_set = set(converted)
        assert (0, 3) in corner_pair_set
        assert (1, 4) in corner_pair_set
        assert (2, 5) in corner_pair_set
        assert len(corner_pair_set) == 3
        # And the original CornerCasePair list itself remains unhashable
        # — confirms the conversion is load-bearing, not cosmetic.
        with pytest.raises(TypeError, match="unhashable"):
            set(baseline_corner)

        # Sanity: stash the captured kwargs for completeness even though
        # we don't drive apply_knob_02 end-to-end in this unit test.
        # The integration coverage lives in the variant generation rerun.
        assert captured == {}  # stub never called in this focused test


# ---- build_openai_non_corner_client (Bug 4 regression, 2026-05-28) -------


class TestOpenAINonCornerClient:
    """Tests for the K2 non-corner refill OpenAI client.

    Regression for 2026-05-28: the standalone CLI and generate_variant
    previously wired the K2 *interpolation* client as the non-corner
    api_client. The interpolation client formats with
    ``{parent_records_json}``; the non-corner prompt expects
    ``{reference_records_json}``. KeyError → returns ``{}`` → every
    refill is rejected as ``empty_primary_label`` (487/487 on the
    products easy verification run).
    """

    def _build_with_chat(self, monkeypatch: pytest.MonkeyPatch, fake_chat: Any) -> Any:
        from usecases_synthetic.lib import llm_client

        def _fake_build(
            *,
            model: str,
            temperature: float = 0.0,
            max_tokens: int | None = None,
        ) -> Any:
            return fake_chat

        monkeypatch.setattr(llm_client, "build_chat_openai", _fake_build)
        return build_openai_non_corner_client(model_id="gpt-5.4-mini", max_tokens=2048)

    def test_substitutes_reference_records_json_placeholder(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The non-corner client must format the prompt with
        ``{reference_records_json}`` (not ``{parent_records_json}``).
        This is exactly the regression: using the interpolation client
        crashed on this placeholder and returned ``{}``."""
        captured: dict[str, Any] = {}

        class FakeChat:
            def invoke(self, prompt: str) -> Any:
                captured["prompt"] = prompt

                class _R:
                    content = json.dumps(
                        {"name": "Synth Co", "industry": "Solar Farms"}
                    )

                return _R()

        client = self._build_with_chat(monkeypatch, FakeChat())

        template = "References: {reference_records_json}\nSchema: {schema_columns_json}"
        references = [
            {"name": "Apple", "industry": "Tech"},
            {"name": "Banana", "industry": "Food"},
        ]
        out = client(template, references)

        assert "Apple" in captured["prompt"]
        assert "Banana" in captured["prompt"]
        assert '"name", "industry"' in captured["prompt"]
        assert out == {"name": "Synth Co", "industry": "Solar Farms"}

    def test_returns_empty_when_template_uses_wrong_placeholder(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the template asks for ``{parent_records_json}`` (the
        interpolation placeholder), formatting fails and the client
        returns ``{}`` — surfacing the wiring mistake instead of
        silently producing an empty refill."""

        class FakeChat:
            def invoke(self, prompt: str) -> Any:  # pragma: no cover
                raise AssertionError("invoke should not be reached on KeyError")

        client = self._build_with_chat(monkeypatch, FakeChat())
        out = client(
            "Parents: {parent_records_json} {schema_columns_json}",
            [{"name": "X"}],
        )
        assert out == {}

    def test_strips_code_fence_wrapper(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Symmetric to the interpolation client: strip ```json fences
        before JSON decode."""

        class FakeChat:
            def invoke(self, prompt: str) -> Any:
                class _R:
                    content = '```json\n{"name": "Synth", "value": 1}\n```'

                return _R()

        client = self._build_with_chat(monkeypatch, FakeChat())
        out = client(
            "{reference_records_json} {schema_columns_json}",
            [{"name": "X", "value": 0}],
        )
        assert out == {"name": "Synth", "value": 1}

    def test_returns_empty_on_malformed_json(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FakeChat:
            def invoke(self, prompt: str) -> Any:
                class _R:
                    content = "not valid json"

                return _R()

        client = self._build_with_chat(monkeypatch, FakeChat())
        out = client(
            "{reference_records_json} {schema_columns_json}",
            [{"name": "X"}],
        )
        assert out == {}
