"""Tests for Knob 04 — Per-entity Source Coverage Skew.

Acceptance criteria (post-2026-05-07 K4 sign-off Pending #5 wire-up):

1. Hard: no entity drops to zero sources (fusion-gold floor at the
   *entity* level — individual fusion-gold records may be removed when
   the entity is collapsed via :func:`score_target_distance`, but every
   protected entity retains ≥1 surviving source).
2. Hard: pool pairs are protected against orphaning — for every pool
   pair, at least one endpoint stays resident in its source. Single-
   endpoint removal is allowed (the new orphan-only semantic in
   :func:`_would_break_pool_edge`).
3. Hard: singleton fraction over *matchable* entities ≤ cap from YAML
   (the rollback excludes synthetic distractor singletons).
4. Easy: fabricated rows have ``k4_fabricated=True`` in provenance.
5. Easy: stochastic dominance
   ``sum(H_easy[k] for k<=j) <= sum(H_medium[k] for k<=j) <= sum(H_hard[k] for k<=j)``.
6. Medium: identity (zero rows added or removed).
7. ``pytest usecases_synthetic/tests/test_knob_04.py -v`` passes.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import pytest

from usecases_synthetic.lib.coverage_ops import (
    EntityView,
    build_entity_view,
    measure_coverage_histogram,
    plan_demotions,
    plan_promotions,
    validate_target_histogram,
)
from usecases_synthetic.scripts.apply_knob_04_coverage import (
    EntityLinkage,
    apply_knob_04,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def k4_config() -> dict[str, Any]:
    """Minimal K4 config for a 3-source domain."""
    return {
        "domain": "test",
        "source_count": 3,
        "id_columns": {
            "source_a": "id",
            "source_b": "id",
            "source_c": "id",
        },
        "primary_columns": {
            "source_a": "name",
            "source_b": "name",
            "source_c": "name",
        },
        "target_coverage_histogram": {
            "easy": {1: 0.05, 2: 0.15, 3: 0.80},
            "medium": None,
            "hard": {1: 0.50, 2: 0.30, 3: 0.20},
        },
        "within_source_duplicate_rate": {
            "easy": 0.0,
            "medium": 0.0,
            "hard": 0.02,
        },
        "singleton_cap_hard": 0.70,
        "delta_softening_step": 0.05,
        "fabrication_mode": "paraphrase_only",
        "llm_prompt_version": "v1",
        "llm_model_id": "test",
        "llm_temperature": 0.0,
    }


@pytest.fixture
def three_source_data() -> dict[str, pd.DataFrame]:
    """Three source DataFrames with a known coverage distribution.

    - 30 entities total.
    - 10 entities present in all 3 sources (coverage=3).
    - 10 entities present in 2 sources (source_a + source_b) (coverage=2).
    - 10 entities present in 1 source (source_c only) (coverage=1).

    Each source records its own record IDs with source-specific prefixes
    so the tests can reason about pool pairs easily.
    """
    rng = np.random.default_rng(7)

    # 10 triples (coverage=3): entities 0..9
    # 10 pairs (coverage=2): entities 10..19 (source_a + source_b)
    # 10 singletons (coverage=1): entities 20..29 (source_c only)

    rows_a: list[dict[str, Any]] = []
    rows_b: list[dict[str, Any]] = []
    rows_c: list[dict[str, Any]] = []

    for i in range(10):
        rows_a.append(
            {
                "id": f"a_{i}",
                "name": f"Entity {i}",
                "country": "US",
            }
        )
        rows_b.append(
            {
                "id": f"b_{i}",
                "name": f"Entity {i}",
                "country": "US",
            }
        )
        rows_c.append(
            {
                "id": f"c_{i}",
                "name": f"Entity {i}",
                "country": "US",
            }
        )

    for i in range(10, 20):
        rows_a.append(
            {
                "id": f"a_{i}",
                "name": f"Entity {i}",
                "country": "DE",
            }
        )
        rows_b.append(
            {
                "id": f"b_{i}",
                "name": f"Entity {i}",
                "country": "DE",
            }
        )

    for i in range(20, 30):
        rows_c.append(
            {
                "id": f"c_{i}",
                "name": f"Entity {i}",
                "country": "JP",
            }
        )

    df_a = pd.DataFrame(rows_a)
    df_a.attrs["dataset_name"] = "source_a"
    df_b = pd.DataFrame(rows_b)
    df_b.attrs["dataset_name"] = "source_b"
    df_c = pd.DataFrame(rows_c)
    df_c.attrs["dataset_name"] = "source_c"

    return {"source_a": df_a, "source_b": df_b, "source_c": df_c}


@pytest.fixture
def three_source_linkage() -> EntityLinkage:
    """Build the linkage matching :func:`three_source_data`.

    Groups:
    - group_0 .. group_9   — triples (a, b, c)
    - group_10 .. group_19 — pairs  (a, b)
    - group_20 .. group_29 — matchable singletons in source_c

    The singleton groups (20..29) are declared explicitly so they
    count as legitimate coverage=1 entities rather than distractor
    singletons (which are excluded from the coverage histogram by
    default).
    """
    groups: dict[str, list[tuple[str, str]]] = {}
    index: dict[str, str] = {}
    for i in range(10):
        gid = f"group_{i}"
        members = [
            ("source_a", f"a_{i}"),
            ("source_b", f"b_{i}"),
            ("source_c", f"c_{i}"),
        ]
        groups[gid] = members
        for src, rid in members:
            index[rid] = gid
    for i in range(10, 20):
        gid = f"group_{i}"
        members = [
            ("source_a", f"a_{i}"),
            ("source_b", f"b_{i}"),
        ]
        groups[gid] = members
        for src, rid in members:
            index[rid] = gid
    for i in range(20, 30):
        gid = f"group_{i}"
        members = [("source_c", f"c_{i}")]
        groups[gid] = members
        for src, rid in members:
            index[rid] = gid
    return EntityLinkage(groups=groups, index=index)


@pytest.fixture
def fusion_gold_ids() -> set[str]:
    """Fusion-gold anchors the first 5 triples."""
    ids: set[str] = set()
    for i in range(5):
        ids.add(f"a_{i}")
        ids.add(f"b_{i}")
        ids.add(f"c_{i}")
    return ids


@pytest.fixture
def pool_pairs() -> list[tuple[tuple[str, str], tuple[str, str]]]:
    """Pool-protected pairs: the first 3 triples on the (a,b) edge."""
    return [(("source_a", f"a_{i}"), ("source_b", f"b_{i}")) for i in range(3)]


# ---------------------------------------------------------------------------
# Coverage-op unit tests
# ---------------------------------------------------------------------------


class TestEntityView:
    def test_view_counts_all_entities(
        self,
        three_source_data: dict[str, pd.DataFrame],
        three_source_linkage: EntityLinkage,
    ) -> None:
        view = build_entity_view(
            three_source_linkage.groups,
            three_source_data,
            id_columns={"source_a": "id", "source_b": "id", "source_c": "id"},
            source_count=3,
        )
        assert len(view) == 30  # 10 triples + 10 pairs + 10 singletons

    def test_coverage_values(
        self,
        three_source_data: dict[str, pd.DataFrame],
        three_source_linkage: EntityLinkage,
    ) -> None:
        view = build_entity_view(
            three_source_linkage.groups,
            three_source_data,
            id_columns={"source_a": "id", "source_b": "id", "source_c": "id"},
            source_count=3,
        )
        for i in range(10):
            assert view.coverage(f"group_{i}") == 3
        for i in range(10, 20):
            assert view.coverage(f"group_{i}") == 2


class TestHistogram:
    def test_measure_baseline(
        self,
        three_source_data: dict[str, pd.DataFrame],
        three_source_linkage: EntityLinkage,
    ) -> None:
        view = build_entity_view(
            three_source_linkage.groups,
            three_source_data,
            id_columns={"source_a": "id", "source_b": "id", "source_c": "id"},
            source_count=3,
        )
        hist = measure_coverage_histogram(view)
        # 10/30 in each bin.
        assert hist[1] == pytest.approx(10 / 30)
        assert hist[2] == pytest.approx(10 / 30)
        assert hist[3] == pytest.approx(10 / 30)
        assert sum(hist.values()) == pytest.approx(1.0)

    def test_validate_target_histogram(self) -> None:
        validate_target_histogram({1: 0.5, 2: 0.3, 3: 0.2}, source_count=3)
        with pytest.raises(ValueError):
            validate_target_histogram({1: 0.5, 2: 0.3}, source_count=3)
        with pytest.raises(ValueError):
            validate_target_histogram({1: 0.4, 2: 0.3, 3: 0.2}, source_count=3)


class TestPlanDemotions:
    def test_single_bin_shift(self) -> None:
        base = {1: 0.33, 2: 0.33, 3: 0.34}
        target = {1: 0.6, 2: 0.2, 3: 0.2}
        demotions = plan_demotions(base, target, total_entities=30, source_count=3)
        # Move ~4 entities from bin 3 -> bin 2, and ~4 more from bin 2 -> bin 1
        assert demotions[3] >= 1
        assert demotions[2] >= demotions[3]

    def test_identity_no_demotions(self) -> None:
        base = {1: 0.3, 2: 0.4, 3: 0.3}
        target = dict(base)
        demotions = plan_demotions(base, target, total_entities=100, source_count=3)
        for v in demotions.values():
            assert v == 0


class TestPlanPromotions:
    def test_single_bin_shift(self) -> None:
        base = {1: 0.6, 2: 0.2, 3: 0.2}
        target = {1: 0.2, 2: 0.2, 3: 0.6}
        promotions = plan_promotions(base, target, total_entities=30, source_count=3)
        assert promotions[1] >= 1
        assert promotions[2] >= 1


# ---------------------------------------------------------------------------
# Acceptance Criterion 6: Medium is identity
# ---------------------------------------------------------------------------


class TestMediumIdentity:
    def test_medium_no_mutation(
        self,
        three_source_data: dict[str, pd.DataFrame],
        three_source_linkage: EntityLinkage,
        fusion_gold_ids: set[str],
        pool_pairs: list[tuple[tuple[str, str], tuple[str, str]]],
        k4_config: dict[str, Any],
    ) -> None:
        srcs = {k: v.copy() for k, v in three_source_data.items()}
        for k, v in srcs.items():
            srcs[k].attrs = v.attrs.copy()

        out_srcs, prov, skipped, hists = apply_knob_04(
            domain="test",
            level="medium",
            sources=srcs,
            config=k4_config,
            linkage=three_source_linkage,
            fusion_gold_ids=fusion_gold_ids,
            pool_pairs=pool_pairs,
            seed=42,
        )
        assert len(prov) == 0
        for name, df in out_srcs.items():
            orig = three_source_data[name]
            assert len(df) == len(orig)
            assert df.equals(orig)


# ---------------------------------------------------------------------------
# Acceptance Criteria 1, 2, 3: Hard path constraints
# ---------------------------------------------------------------------------


class TestHardRemoval:
    def _run(
        self,
        three_source_data: dict[str, pd.DataFrame],
        three_source_linkage: EntityLinkage,
        fusion_gold_ids: set[str],
        pool_pairs: list[tuple[tuple[str, str], tuple[str, str]]],
        k4_config: dict[str, Any],
    ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
        srcs = {k: v.copy() for k, v in three_source_data.items()}
        for k, v in srcs.items():
            srcs[k].attrs = v.attrs.copy()
        out_srcs, prov, skipped, hists = apply_knob_04(
            domain="test",
            level="hard",
            sources=srcs,
            config=k4_config,
            linkage=three_source_linkage,
            fusion_gold_ids=fusion_gold_ids,
            pool_pairs=pool_pairs,
            seed=42,
        )
        return out_srcs, prov, hists

    def test_fusion_gold_floor(
        self,
        three_source_data: dict[str, pd.DataFrame],
        three_source_linkage: EntityLinkage,
        fusion_gold_ids: set[str],
        pool_pairs: list[tuple[tuple[str, str], tuple[str, str]]],
        k4_config: dict[str, Any],
    ) -> None:
        """Criterion 1 (post-2026-05-07): every entity (fusion val/test
        included) retains ≥1 surviving source after K4 hard. Individual
        fusion-gold records may be removed — closeness ranking via
        :func:`score_target_distance` chooses which source survives the
        demotion — but the entity-level floor holds.
        """
        out_srcs, _, _ = self._run(
            three_source_data,
            three_source_linkage,
            fusion_gold_ids,
            pool_pairs,
            k4_config,
        )
        from usecases_synthetic.lib.coverage_ops import build_entity_view

        view = build_entity_view(
            three_source_linkage.groups,
            out_srcs,
            id_columns={"source_a": "id", "source_b": "id", "source_c": "id"},
            source_count=3,
        )
        # No entity (matchable, including fusion val/test) drops to zero
        # sources.
        for entity_id in three_source_linkage.groups:
            assert (
                view.coverage(entity_id) >= 1
            ), f"Entity {entity_id} dropped to zero sources"
        # Every entity that contains any fusion-gold record must still
        # have ≥1 surviving source. (Per K4 sign-off Pending #5 wire-up,
        # individual fusion-gold *records* are no longer blanket-protected;
        # the entity-level floor + closeness-aware survivor selection
        # together guarantee the fusion-protected universe stays evaluable.)
        for entity_id, members in three_source_linkage.groups.items():
            entity_has_gold = any(rid in fusion_gold_ids for _, rid in members)
            if not entity_has_gold:
                continue
            assert (
                view.coverage(entity_id) >= 1
            ), f"Fusion-protected entity {entity_id} dropped to zero sources"

    def test_pool_edges_preserved(
        self,
        three_source_data: dict[str, pd.DataFrame],
        three_source_linkage: EntityLinkage,
        fusion_gold_ids: set[str],
        pool_pairs: list[tuple[tuple[str, str], tuple[str, str]]],
        k4_config: dict[str, Any],
    ) -> None:
        """Criterion 2 (post-2026-05-07): every pool pair retains ≥1
        alive endpoint after K4 hard. Single-endpoint removal is allowed
        (per the orphan-only semantic in :func:`_would_break_pool_edge`);
        only both-endpoint removal is forbidden.
        """
        out_srcs, _, _ = self._run(
            three_source_data,
            three_source_linkage,
            fusion_gold_ids,
            pool_pairs,
            k4_config,
        )
        for (ls, lid), (rs, rid) in pool_pairs:
            assert ls != rs, "pool pair must originate in distinct sources"
            left_alive = lid in set(out_srcs[ls]["id"].astype(str))
            right_alive = rid in set(out_srcs[rs]["id"].astype(str))
            assert left_alive or right_alive, (
                f"Pool pair (({ls}:{lid}), ({rs}:{rid})) orphaned — both "
                f"endpoints removed. The orphan-check in "
                f"_would_break_pool_edge must keep ≥1 endpoint alive."
            )

    def test_singleton_cap_respected(
        self,
        three_source_data: dict[str, pd.DataFrame],
        three_source_linkage: EntityLinkage,
        fusion_gold_ids: set[str],
        pool_pairs: list[tuple[tuple[str, str], tuple[str, str]]],
        k4_config: dict[str, Any],
    ) -> None:
        """Criterion 3: singleton fraction <= cap at hard."""
        k4_config["singleton_cap_hard"] = 0.40
        # Ensure the target nominally asks for more singletons than the cap
        # so the cap actually binds.
        k4_config["target_coverage_histogram"]["hard"] = {
            1: 0.70,
            2: 0.20,
            3: 0.10,
        }
        out_srcs, _, hists = self._run(
            three_source_data,
            three_source_linkage,
            fusion_gold_ids,
            pool_pairs,
            k4_config,
        )
        realised = hists[hists["label"] == "realised_hard"]
        h1 = float(realised[realised["coverage"] == 1]["fraction"].iloc[0])
        assert h1 <= k4_config["singleton_cap_hard"] + 1e-6, (
            f"Singleton fraction {h1:.3f} exceeds cap "
            f"{k4_config['singleton_cap_hard']:.3f}"
        )


# ---------------------------------------------------------------------------
# Acceptance Criteria 4, 5: Easy path
# ---------------------------------------------------------------------------


def _stub_paraphrase_factory(target_source: str):  # noqa: D401 - test helper
    def _fn(
        col: str, value: str, rng: np.random.Generator
    ) -> tuple[str, dict[str, Any]]:
        # Trivial paraphrase: uppercase the value so output differs from sibling.
        return value.upper(), {"transform_fn": "stub_upper"}

    return _fn


class TestEasyFabrication:
    def _run(
        self,
        three_source_data: dict[str, pd.DataFrame],
        three_source_linkage: EntityLinkage,
        fusion_gold_ids: set[str],
        pool_pairs: list[tuple[tuple[str, str], tuple[str, str]]],
        k4_config: dict[str, Any],
    ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
        srcs = {k: v.copy() for k, v in three_source_data.items()}
        for k, v in srcs.items():
            srcs[k].attrs = v.attrs.copy()
        out_srcs, prov, skipped, hists = apply_knob_04(
            domain="test",
            level="easy",
            sources=srcs,
            config=k4_config,
            linkage=three_source_linkage,
            fusion_gold_ids=fusion_gold_ids,
            pool_pairs=pool_pairs,
            seed=42,
            k1_config={
                "attribute_classes": {
                    "source_a": {"name": "primary", "country": "key"},
                    "source_b": {"name": "primary", "country": "key"},
                    "source_c": {"name": "primary", "country": "key"},
                },
            },
            paraphrase_fn_factory=_stub_paraphrase_factory,
        )
        return out_srcs, prov, hists

    def test_fabricated_provenance_flag(
        self,
        three_source_data: dict[str, pd.DataFrame],
        three_source_linkage: EntityLinkage,
        fusion_gold_ids: set[str],
        pool_pairs: list[tuple[tuple[str, str], tuple[str, str]]],
        k4_config: dict[str, Any],
    ) -> None:
        """Criterion 4: fabricated rows have ``k4_fabricated=True``."""
        _, prov, _ = self._run(
            three_source_data,
            three_source_linkage,
            fusion_gold_ids,
            pool_pairs,
            k4_config,
        )
        assert len(prov) > 0, "Expected fabrications for easy level"
        fab_rows = prov[prov["transform_fn"] == "propagate_and_paraphrase"]
        assert len(fab_rows) > 0
        for _, row in fab_rows.iterrows():
            params = json.loads(row["transform_params"])
            assert params.get("k4_fabricated") is True

    def test_fabrication_increases_coverage(
        self,
        three_source_data: dict[str, pd.DataFrame],
        three_source_linkage: EntityLinkage,
        fusion_gold_ids: set[str],
        pool_pairs: list[tuple[tuple[str, str], tuple[str, str]]],
        k4_config: dict[str, Any],
    ) -> None:
        """Easy realised histogram's singleton fraction is <= baseline's."""
        _, _, hists = self._run(
            three_source_data,
            three_source_linkage,
            fusion_gold_ids,
            pool_pairs,
            k4_config,
        )
        base = hists[hists["label"] == "baseline"]
        easy = hists[hists["label"] == "realised_easy"]
        base_h1 = float(base[base["coverage"] == 1]["fraction"].iloc[0])
        easy_h1 = float(easy[easy["coverage"] == 1]["fraction"].iloc[0])
        assert easy_h1 <= base_h1 + 1e-6


# ---------------------------------------------------------------------------
# Acceptance Criterion 5: Cross-level stochastic dominance
# ---------------------------------------------------------------------------


class TestStochasticDominance:
    def test_cdf_dominance(
        self,
        three_source_data: dict[str, pd.DataFrame],
        three_source_linkage: EntityLinkage,
        fusion_gold_ids: set[str],
        pool_pairs: list[tuple[tuple[str, str], tuple[str, str]]],
        k4_config: dict[str, Any],
    ) -> None:
        """Realised histograms satisfy
        ``CDF_easy[j] <= CDF_medium[j] <= CDF_hard[j]`` for all j."""

        def _realised(level: str) -> dict[int, float]:
            srcs = {k: v.copy() for k, v in three_source_data.items()}
            for k, v in srcs.items():
                srcs[k].attrs = v.attrs.copy()
            _, _, _, hists = apply_knob_04(
                domain="test",
                level=level,
                sources=srcs,
                config=k4_config,
                linkage=three_source_linkage,
                fusion_gold_ids=fusion_gold_ids,
                pool_pairs=pool_pairs,
                seed=42,
                k1_config={
                    "attribute_classes": {
                        "source_a": {"name": "primary", "country": "key"},
                        "source_b": {"name": "primary", "country": "key"},
                        "source_c": {"name": "primary", "country": "key"},
                    },
                },
                paraphrase_fn_factory=_stub_paraphrase_factory,
            )
            rr = hists[hists["label"] == f"realised_{level}"]
            return {int(r["coverage"]): float(r["fraction"]) for _, r in rr.iterrows()}

        h_easy = _realised("easy")
        h_med = _realised("medium")
        h_hard = _realised("hard")

        cdf_easy = 0.0
        cdf_med = 0.0
        cdf_hard = 0.0
        for k in (1, 2, 3):
            cdf_easy += h_easy.get(k, 0.0)
            cdf_med += h_med.get(k, 0.0)
            cdf_hard += h_hard.get(k, 0.0)
            assert (
                cdf_easy <= cdf_med + 1e-6
            ), f"CDF easy > medium at k={k}: {cdf_easy:.3f} > {cdf_med:.3f}"
            assert (
                cdf_med <= cdf_hard + 1e-6
            ), f"CDF medium > hard at k={k}: {cdf_med:.3f} > {cdf_hard:.3f}"
