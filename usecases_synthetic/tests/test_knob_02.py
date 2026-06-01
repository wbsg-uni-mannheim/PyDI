"""Tests for Knob 02 — Entity Niche Density.

Acceptance criteria (from plans/module_09_knob_02.md):

1. RRF density is monotone in number of agreeing metrics.
2. No protected entity removed at any level.
3. At easy, corner-case ratio measured ≤ target from YAML.
4. At hard, interpolated entities have valid values for all required
   schema fields.
5. Embeddings cached on disk and loaded on rerun (no recomputation).
6. EM test set regenerated per variant with corner-case stratification.
7. Contamination spot-check passes for all interpolated entities.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from usecases_synthetic.lib.corner_case_miner import (
    CornerCasePair,
    MetricThresholds,
    classify_pair,
    measure_corner_case_ratio,
    mine_corner_cases,
    regenerate_em_test_set,
)
from usecases_synthetic.lib.entity_interpolation import (
    InterpolatedEntity,
    contamination_check,
    default_api_client_from_attributes,
    interpolate_entity,
    parent_pair_hash,
    select_parent_pairs,
)
from usecases_synthetic.lib.llm_cache import LLMCache
from usecases_synthetic.lib.niche_metrics import (
    attribute_overlap,
    attribute_overlap_neighbours,
    build_label_list,
    build_text_corpus,
    compute_embedding_matrix,
    compute_tfidf_matrix,
    label_collision_index,
    lexical_extended_jaccard,
    lexical_extended_jaccard_neighbours,
    normalize_label,
    tfidf_neighbours,
)
from usecases_synthetic.lib.niche_scorer import (
    EntityDensity,
    MetricNeighbourhoods,
    compute_rrf_density,
    rank_entities_by_density,
    reciprocal_rank_fusion,
    select_for_removal,
)

# ---- Metric unit tests ----------------------------------------------------


class TestNicheMetrics:
    def test_normalize_label_strips_bracketed_and_punct(self) -> None:
        assert normalize_label("Apple Inc. (Company)") == "apple inc"
        assert normalize_label("  Microsoft,  Corp.  ") == "microsoft corp"
        assert normalize_label(None) == ""  # type: ignore[arg-type]

    def test_extended_jaccard_identical_and_disjoint(self) -> None:
        assert lexical_extended_jaccard("Apple Inc", "Apple Inc") == 1.0
        # Completely disjoint tokens
        assert lexical_extended_jaccard("Apple", "Microsoft") == 0.0

    def test_extended_jaccard_typo_robustness(self) -> None:
        # Single-character typo should still match above 0.8 threshold
        sim = lexical_extended_jaccard("Microsoft", "Microsft")
        assert sim == 1.0  # Levenshtein ratio high enough

    def test_extended_jaccard_inner_token_threshold_gate(self) -> None:
        # With threshold 1.0 typos no longer match.
        sim = lexical_extended_jaccard(
            "Microsoft", "Microsft", inner_token_threshold=1.0
        )
        assert sim == 0.0

    def test_label_collision_index(self) -> None:
        labels = [
            "John Williams",
            "John Williams (composer)",
            "John Williams",
            "Hans Zimmer",
        ]
        groups = label_collision_index(labels)
        assert "john williams" in groups
        assert sorted(groups["john williams"]) == [0, 1, 2]
        assert "hans zimmer" not in groups  # singleton excluded

    def test_attribute_overlap_weighted(self) -> None:
        bag_a = {"industry": "tech", "country": "us"}
        bag_b = {"industry": "tech", "country": "de"}
        weights = {"industry": 0.5, "country": 0.5}
        sim = attribute_overlap(bag_a, bag_b, weights)
        assert abs(sim - 0.5) < 1e-9  # 0.5 / 1.0

    def test_extended_jaccard_neighbours_top_k(self) -> None:
        labels = ["apple", "apple inc", "microsoft", "microsoft corp"]
        nbrs = lexical_extended_jaccard_neighbours(labels, top_k=1)
        assert len(nbrs) == 4
        # Apple's best neighbour should be "apple inc"
        assert nbrs[0][0][0] == 1
        # Microsoft's best should be "microsoft corp"
        assert nbrs[2][0][0] == 3

    def test_extended_jaccard_neighbours_prefix_blocking_drops_disjoint_pairs(
        self,
    ) -> None:
        # Two clusters with no shared 3-char token prefix: blocker keeps them
        # apart and avoids the O(n^2) all-pairs scan.
        labels = [
            "apple",
            "apple inc",
            "microsoft",
            "microsoft corp",
            "zynga",
            "zynga ltd",
        ]
        nbrs = lexical_extended_jaccard_neighbours(labels, top_k=5)
        assert len(nbrs) == 6
        # Each label only finds neighbours within its own prefix bucket.
        nbr_idxs = [{j for j, _ in row} for row in nbrs]
        assert nbr_idxs[0] <= {1}
        assert nbr_idxs[2] <= {3}
        assert nbr_idxs[4] <= {5}

    def test_extended_jaccard_neighbours_oversized_buckets_dropped(self) -> None:
        # 3000 single-token labels sharing the same "alp" prefix, plus a
        # niche cluster of 2 with a unique prefix. With max_block_size=2000
        # the 3000-element "alp" bucket is dropped, but the niche pair
        # survives. Single-token labels avoid per-i numeric prefix buckets.
        labels = [f"alphagame{i:04d}xx" for i in range(3000)] + ["zynga", "zynga ltd"]
        nbrs = lexical_extended_jaccard_neighbours(labels, top_k=5, max_block_size=2000)
        # The only prefix bucket each "alphagame*" label produces is "alp"
        # (size 3000, dropped) → all 3000 rows have empty neighbour lists.
        assert all(row == [] for row in nbrs[:3000])
        # Niche pair "zyn" prefix bucket stays.
        assert nbrs[3000][0][0] == 3001
        assert nbrs[3001][0][0] == 3000

    def test_extended_jaccard_neighbours_typo_recovered_when_prefix_shared(
        self,
    ) -> None:
        # "Microsoft" / "Microsft" share the "mic" prefix → blocker keeps them
        # candidates and the inner-Levenshtein scorer catches the typo.
        labels = ["microsoft", "microsft", "apple"]
        nbrs = lexical_extended_jaccard_neighbours(labels, top_k=2)
        assert nbrs[0][0][0] == 1  # microsoft → microsft
        # Apple has no candidates from blocking (different prefix).
        assert nbrs[2] == []

    def test_tfidf_neighbours(self) -> None:
        corpus = [
            "apple inc tech",
            "apple corp tech",
            "microsoft software",
            "microsoft inc",
        ]
        m = compute_tfidf_matrix(corpus)
        nbrs = tfidf_neighbours(m, top_k=1)
        assert len(nbrs) == 4
        # Apple docs should be closest to each other
        assert nbrs[0][0][0] == 1
        assert nbrs[1][0][0] == 0

    def test_attribute_overlap_neighbours_inverted_index_blocking(self) -> None:
        # Two clusters: {0,1,2} share industry=tech, {3,4} share industry=auto;
        # no (col,val) pair crosses the cluster boundary so blocker keeps them
        # apart and avoids the all-pairs scan.
        bags = [
            {"industry": "tech", "country": "us"},
            {"industry": "tech", "country": "us"},
            {"industry": "tech", "country": "de"},
            {"industry": "auto", "country": "jp"},
            {"industry": "auto", "country": "de"},
        ]
        weights = {"industry": 0.7, "country": 0.3}
        nbrs = attribute_overlap_neighbours(bags, weights=weights, top_k=5)
        # Tech cluster only finds tech entities as neighbours.
        assert {j for j, _ in nbrs[0]} == {1, 2}
        # Auto cluster shares country=de with index 2 (tech), so 2 → 4 via
        # country bucket and 4 → 2 the same way.
        assert 2 in {j for j, _ in nbrs[4]}
        assert 4 in {j for j, _ in nbrs[2]}

    def test_attribute_overlap_neighbours_oversized_buckets_dropped(self) -> None:
        # 3000 bags share industry=tech (oversized → dropped at threshold
        # 2000); a niche pair has industry=niche and is preserved.
        bags = [{"industry": "tech"} for _ in range(3000)] + [
            {"industry": "niche"},
            {"industry": "niche"},
        ]
        weights = {"industry": 1.0}
        nbrs = attribute_overlap_neighbours(
            bags, weights=weights, top_k=5, max_block_size=2000
        )
        assert all(row == [] for row in nbrs[:3000])
        assert nbrs[3000][0][0] == 3001
        assert nbrs[3001][0][0] == 3000

    def test_attribute_overlap_neighbours_zero_weights_returns_empty(self) -> None:
        bags = [{"industry": "tech"}, {"industry": "tech"}]
        nbrs = attribute_overlap_neighbours(bags, weights={"industry": 0.0}, top_k=1)
        assert nbrs == [[], []]


# ---- Embedding cache tests ------------------------------------------------


class TestEmbeddingCache:
    def test_embedding_cache_roundtrip(self, tmp_path: Path) -> None:
        corpus = ["apple inc", "apple company", "microsoft", "google"]
        cache = tmp_path / "test_domain.npy"
        m1 = compute_embedding_matrix(
            corpus,
            model_id="sentence-transformers/all-MiniLM-L6-v2",
            cache_path=cache,
            concat_order=["name"],
        )
        assert cache.exists()
        meta = cache.with_suffix(".meta.json")
        assert meta.exists()
        # Second call should hit the cache (identity must match)
        m2 = compute_embedding_matrix(
            corpus,
            model_id="sentence-transformers/all-MiniLM-L6-v2",
            cache_path=cache,
            concat_order=["name"],
        )
        assert m1.shape == m2.shape
        assert np.allclose(m1, m2)

    def test_embedding_cache_invalidated_on_content_change(
        self, tmp_path: Path
    ) -> None:
        cache = tmp_path / "test_domain.npy"
        m1 = compute_embedding_matrix(
            ["apple", "banana"],
            model_id="sentence-transformers/all-MiniLM-L6-v2",
            cache_path=cache,
            concat_order=["name"],
        )
        # Different corpus => new cache write
        m2 = compute_embedding_matrix(
            ["cherry", "date"],
            model_id="sentence-transformers/all-MiniLM-L6-v2",
            cache_path=cache,
            concat_order=["name"],
        )
        with open(cache.with_suffix(".meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["model_id"] == "sentence-transformers/all-MiniLM-L6-v2"
        # m2 is the second corpus.
        assert m2.shape[0] == 2
        assert not np.allclose(m1, m2)


# ---- RRF scorer tests -----------------------------------------------------


class TestRrfScorer:
    def test_rrf_monotone_in_agreement_count(self) -> None:
        # Entity 0 has 3 metrics agreeing on neighbour 1.
        # Entity 2 has 2 metrics agreeing on neighbour 3.
        # Entity 4 has 1 metric (only m1).
        m1 = MetricNeighbourhoods(
            "a",
            [
                [(1, 0.9)],
                [(0, 0.9)],  # entities 0,1 agree on each other
                [(3, 0.9)],
                [(2, 0.9)],  # entities 2,3
                [(5, 0.9)],
                [(4, 0.9)],  # entities 4,5 (only here)
            ],
        )
        m2 = MetricNeighbourhoods(
            "b",
            [
                [(1, 0.9)],
                [(0, 0.9)],
                [(3, 0.9)],
                [(2, 0.9)],
                [],
                [],
            ],
        )
        m3 = MetricNeighbourhoods(
            "c",
            [
                [(1, 0.9)],
                [(0, 0.9)],
                [],
                [],
                [],
                [],
            ],
        )
        d = compute_rrf_density([m1, m2, m3], n_entities=6, k0=60, c_min=2)
        # 3 metrics agree => highest density
        assert d[0].density > d[2].density
        # 1 metric agreement => filtered by c_min=2 => zero density
        assert d[4].density == 0.0
        assert d[5].density == 0.0

    def test_label_collision_boost(self) -> None:
        # Entities with zero RRF neighbours receive only the boost.
        m = MetricNeighbourhoods("a", [[], [], []])
        collision = {"foo": [0, 1]}
        d = compute_rrf_density(
            [m],
            n_entities=3,
            c_min=1,
            label_collision_groups=collision,
            boost_label_collision=5.0,
        )
        assert d[0].density == 5.0
        assert d[1].density == 5.0
        assert d[2].density == 0.0

    def test_rank_entities_by_density_descending(self) -> None:
        densities = [
            EntityDensity(
                index=0, density=1.0, rrf_component=1.0, label_collision_component=0.0
            ),
            EntityDensity(
                index=1, density=5.0, rrf_component=5.0, label_collision_component=0.0
            ),
            EntityDensity(
                index=2, density=3.0, rrf_component=3.0, label_collision_component=0.0
            ),
        ]
        rng = np.random.default_rng(42)
        order = rank_entities_by_density(densities, rng)
        assert order == [1, 2, 0]

    def test_select_for_removal_respects_protection(self) -> None:
        ranked = [0, 1, 2, 3, 4]
        flags = [True, False, True, False, False]  # 0 and 2 are protected
        queue = select_for_removal(
            ranked, protection_flags=flags, removal_fraction_cap=1.0
        )
        # Protected indices MUST NOT appear in the queue.
        assert 0 not in queue
        assert 2 not in queue
        assert set(queue) == {1, 3, 4}

    def test_select_for_removal_fraction_cap(self) -> None:
        ranked = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        flags = [False] * 10
        queue = select_for_removal(
            ranked, protection_flags=flags, removal_fraction_cap=0.3
        )
        assert len(queue) == 3

    def test_rrf_fusion_k0_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            reciprocal_rank_fusion([], k0=0, n_entities=0)


# ---- Corner case miner tests ----------------------------------------------


class TestCornerCaseMiner:
    def test_classify_hard_match_by_low_similarity(self) -> None:
        thr = MetricThresholds(
            t_match={"ext_jaccard": 0.5}, t_nonmatch={"ext_jaccard": 0.5}
        )
        pair = classify_pair(
            0,
            1,
            sims={"ext_jaccard": 0.2},
            same_cluster=True,
            thresholds=thr,
            labels=["Apple", "Pear"],
        )
        assert pair is not None
        assert pair.kind == "hard_match"
        assert "ext_jaccard" in pair.triggered_by

    def test_classify_hard_non_match_by_high_similarity(self) -> None:
        thr = MetricThresholds(
            t_match={"ext_jaccard": 0.3}, t_nonmatch={"ext_jaccard": 0.5}
        )
        pair = classify_pair(
            0,
            1,
            sims={"ext_jaccard": 0.9},
            same_cluster=False,
            thresholds=thr,
            labels=["Apple", "Aple"],
        )
        assert pair is not None
        assert pair.kind == "hard_non_match"

    def test_label_collision_crosses_cluster_is_hard_non_match(self) -> None:
        thr = MetricThresholds(t_match={}, t_nonmatch={})
        pair = classify_pair(
            0,
            1,
            sims={},
            same_cluster=False,
            thresholds=thr,
            labels=["John Williams", "John Williams"],
        )
        assert pair is not None
        assert pair.kind == "hard_non_match"
        assert "label_collision" in pair.triggered_by

    def test_mine_corner_cases_finds_known_hard_pairs(self) -> None:
        labels = [
            "Apple Corp Tech Group",
            "Apple Corp Tech Company",
            "Microsoft",
            "Banana Fruit",
        ]
        thr = MetricThresholds(
            t_match={"ext_jaccard": 0.5},
            t_nonmatch={"ext_jaccard": 0.5},
        )
        cluster_of = [0, 1, 2, 3]  # all different clusters
        pairs = [(0, 1), (0, 2), (2, 3)]
        results = mine_corner_cases(
            candidate_pairs=pairs,
            cluster_of=cluster_of,
            labels=labels,
            tfidf_matrix=None,
            embeddings=None,
            thresholds=thr,
        )
        # (0, 1) is a cross-cluster pair with high ext_jaccard => hard_non_match
        hard_pairs = [(r.i, r.j) for r in results]
        assert (0, 1) in hard_pairs

    def test_regenerate_em_splits_keeps_surviving_originals(self) -> None:
        """When all originals survive K2, the regen mirrors them verbatim.

        With every original id in ``ids_present`` and a generous backfill
        pool, the regenerator should:

        - Carry over every original row into BOTH versions
          (baseline_pruned and corner_filled — Set 1 ⊂ Set 2 per C11).
        - Add no backfill rows under target_ratio=0.0.
        - Respect cross-split disjointness within a source pair within
          each version.
        """
        from usecases_synthetic.lib.corner_case_miner import (
            REGEN_VERSIONS,
            SplitSpec,
            regenerate_em_splits,
        )

        pair_ab = ("a", "b")
        # Original splits — 10 pos + 10 neg per split.
        originals = {
            pair_ab: {
                "train": [(f"a{i}", f"b{i}", True) for i in range(10)]
                + [(f"a{i}", f"b{i + 50}", False) for i in range(10)],
                "val": [(f"a{i}", f"b{i}", True) for i in range(10, 20)]
                + [(f"a{i}", f"b{i + 50}", False) for i in range(10, 20)],
                "test": [(f"a{i}", f"b{i}", True) for i in range(20, 30)]
                + [(f"a{i}", f"b{i + 50}", False) for i in range(20, 30)],
            }
        }
        ids_present = {f"a{i}" for i in range(40)} | {f"b{i}" for i in range(100)}

        # Backfill pools — generous, but we shouldn't need them
        # (target_ratio=0 ⇒ no corner backfill triggered).
        cluster_pos = {pair_ab: [(f"a{i}", f"b{i}") for i in range(30)]}
        pool_pos = {pair_ab: [(f"a{i + 30}", f"b{i + 30}") for i in range(20)]}
        interp_pos = {pair_ab: []}
        negatives = {
            pair_ab: [
                (f"a{i}", f"b{j + 50}") for i in range(30) for j in range(30) if i != j
            ][:100]
        }
        corner_negs = {pair_ab: set()}

        specs = {
            pair_ab: [
                SplitSpec("train", 20, 0.5),
                SplitSpec("val", 20, 0.5),
                SplitSpec("test", 20, 0.5),
            ],
        }

        rng = np.random.default_rng(11)
        rows = regenerate_em_splits(
            original_pairs_by_split=originals,
            ids_present=ids_present,
            pool_positives_by_pair=pool_pos,
            interpolated_positives_by_pair=interp_pos,
            cluster_positives_by_pair=cluster_pos,
            negatives_by_pair=negatives,
            corner_case_negatives_by_pair=corner_negs,
            split_specs_by_pair=specs,
            target_ratio=0.0,
            rng=rng,
        )
        df = pd.DataFrame(rows)

        # Every original row appears verbatim under both versions.
        assert set(df["version"].unique()) == set(REGEN_VERSIONS)
        for version in REGEN_VERSIONS:
            ver_df = df[df["version"] == version]
            for split, pairs in originals[pair_ab].items():
                sub = ver_df[ver_df["split"] == split]
                assert (
                    len(sub) == 20
                ), f"version={version} split={split}: expected 20, got {len(sub)}"
                expected = {(p[0], p[1]) for p in pairs}
                actual = {(r["id1"], r["id2"]) for _, r in sub.iterrows()}
                assert actual == expected

        # Cross-split disjointness within the source pair within each
        # version. (Across versions, the same canonical pair is
        # intentionally re-emitted — that's the Set 1 ⊂ Set 2 invariant.)
        for version in REGEN_VERSIONS:
            pair_df = df[(df["pair_name"] == "a_2_b") & (df["version"] == version)]
            canonical = set()
            for _, row in pair_df.iterrows():
                a, b = row["id1"], row["id2"]
                key = (a, b) if a < b else (b, a)
                assert key not in canonical, f"reused {key} within version={version}"
                canonical.add(key)

    def test_regenerate_em_splits_backfills_dropped_ids(self) -> None:
        """Dropped originals are replaced from the corner-mined pools.

        Half the original positives reference an id absent from
        ``ids_present`` (simulating K2 removal). Under C11 the
        regenerator should:

        - Drop those originals.
        - baseline_pruned: survivors only, no backfill.
        - corner_filled: survivors + corner-mined backfill from
          interpolated_positives / corner_case_negatives. NO easy fills.
        - Preserve disjointness within each version.
        - Maintain Set 1 ⊂ Set 2 (every baseline_pruned pair appears
          in corner_filled).
        """
        from usecases_synthetic.lib.corner_case_miner import (
            VERSION_BASELINE_PRUNED,
            VERSION_CORNER_FILLED,
            SplitSpec,
            regenerate_em_splits,
        )

        pair_ab = ("a", "b")
        originals = {
            pair_ab: {
                "test": [(f"a{i}", f"b{i}", True) for i in range(10)]
                + [(f"a{i}", f"b{i + 100}", False) for i in range(10)],
            }
        }
        # First 5 a-ids are K2-removed.
        ids_present = (
            {f"a{i}" for i in range(5, 200)}
            | {f"b{i}" for i in range(200)}
            | {f"interp{k}__{s}" for k in range(5) for s in ("a", "b")}
        )

        # Backfill candidates use only b-ids in ids_present (b0..b199).
        # cluster_pos / pool_pos retained in signature but no longer
        # consumed under C11 — easy backfill is removed.
        cluster_pos = {pair_ab: [(f"a{i}", f"b{i + 150}") for i in range(20, 30)]}
        pool_pos = {pair_ab: [(f"a{i}", f"b{i + 160}") for i in range(30, 40)]}
        interp_pos = {pair_ab: [(f"interp{k}__a", f"interp{k}__b") for k in range(5)]}
        # Negative pool that actually contains the corner candidates.
        corner_negs_set = {(f"a{i}", f"b{i + 130}") for i in range(40, 45)}
        easy_neg_seed = [
            (f"a{i}", f"b{j + 170}") for i in range(20, 30) for j in range(20)
        ]
        negatives = {pair_ab: sorted(corner_negs_set) + easy_neg_seed}
        corner_negs = {pair_ab: corner_negs_set}

        specs = {pair_ab: [SplitSpec("test", 20, 0.5)]}
        rng = np.random.default_rng(13)
        rows = regenerate_em_splits(
            original_pairs_by_split=originals,
            ids_present=ids_present,
            pool_positives_by_pair=pool_pos,
            interpolated_positives_by_pair=interp_pos,
            cluster_positives_by_pair=cluster_pos,
            negatives_by_pair=negatives,
            corner_case_negatives_by_pair=corner_negs,
            split_specs_by_pair=specs,
            target_ratio=0.4,
            rng=rng,
        )
        df = pd.DataFrame(rows)

        # ---- baseline_pruned: survivors only (5 pos + 5 neg = 10) ----
        bp = df[df["version"] == VERSION_BASELINE_PRUNED]
        assert len(bp) == 10
        bp_pos = {(r["id1"], r["id2"]) for _, r in bp[bp["label"] == "true"].iterrows()}
        bp_neg = {
            (r["id1"], r["id2"]) for _, r in bp[bp["label"] == "false"].iterrows()
        }
        assert bp_pos == {(f"a{i}", f"b{i}") for i in range(5, 10)}
        assert bp_neg == {(f"a{i}", f"b{i + 100}") for i in range(5, 10)}

        # ---- corner_filled: survivors + 100% corner backfill (20 rows)
        cf = df[df["version"] == VERSION_CORNER_FILLED]
        cf_pos = cf[cf["label"] == "true"]
        cf_neg = cf[cf["label"] == "false"]
        # 5 surviving + 5 interp corners = 10 positives.
        assert len(cf_pos) == 10
        # 5 surviving + 5 corner negs = 10 negatives.
        assert len(cf_neg) == 10
        assert len(cf) == 20

        # corner_filled positives consist of survivors + interpolated only
        # (NO easy fills from cluster_pos / pool_pos).
        cf_pos_pairs = {(r["id1"], r["id2"]) for _, r in cf_pos.iterrows()}
        interp_set = {(f"interp{k}__a", f"interp{k}__b") for k in range(5)}
        assert cf_pos_pairs == bp_pos | interp_set, (
            "corner_filled positives must be survivors ∪ interpolated only "
            "(no easy backfill under C11)"
        )

        # corner_filled negatives consist of survivors + corner_negs only.
        cf_neg_pairs = {(r["id1"], r["id2"]) for _, r in cf_neg.iterrows()}
        assert cf_neg_pairs == bp_neg | corner_negs_set, (
            "corner_filled negatives must be survivors ∪ corner_negs only "
            "(no easy backfill under C11)"
        )

        # Set 1 ⊂ Set 2 invariant.
        bp_pairs = {(r["id1"], r["id2"]) for _, r in bp.iterrows()}
        cf_pairs = {(r["id1"], r["id2"]) for _, r in cf.iterrows()}
        assert bp_pairs.issubset(
            cf_pairs
        ), f"baseline_pruned ⊄ corner_filled: extras={bp_pairs - cf_pairs}"

        # No removed id appears anywhere.
        for _, row in df.iterrows():
            assert row["id1"] in ids_present, row["id1"]
            assert row["id2"] in ids_present, row["id2"]

    def test_regenerate_em_splits_does_not_steal_other_splits_gold(
        self,
    ) -> None:
        """Backfill must not consume canonical pairs owned by gold.

        Regression for plan_s1_final.md F10. Pre-fix: if test's backfill
        pool happened to contain a pair that train's gold owns, the
        higher-priority test pass would ``consume`` the canonical pair
        first; train's survival pass then saw it in ``consumed`` and
        silently skipped — relocating the survivor from train to test.
        Fix: backfill pools are filtered to exclude every gold canon up
        front, so survivors never compete with backfill for their own
        pair.
        """
        from usecases_synthetic.lib.corner_case_miner import (
            SplitSpec,
            regenerate_em_splits,
        )

        pair_ab = ("a", "b")
        # Disjoint gold splits at canonical-pair level — train owns
        # (a1, b1), val (a2, b2), test (a3, b3). All positives.
        originals = {
            pair_ab: {
                "train": [(f"a{i}", f"b{i}", True) for i in range(1, 6)],
                "val": [(f"a{i}", f"b{i}", True) for i in range(10, 13)],
                "test": [(f"a{i}", f"b{i}", True) for i in range(20, 22)],
            }
        }
        ids_present = {
            f"a{i}"
            for i in list(range(1, 6)) + list(range(10, 13)) + list(range(20, 22))
        }
        ids_present |= {
            f"b{i}"
            for i in list(range(1, 6)) + list(range(10, 13)) + list(range(20, 22))
        }
        # Pool positives intentionally overlap with train's gold so the
        # pre-fix code would let test/val backfill ``steal`` it.
        overlapping_pool = [("a1", "b1"), ("a3", "b3")]
        specs = {
            pair_ab: [
                SplitSpec("train", 5, 1.0),
                SplitSpec("val", 3, 1.0),
                SplitSpec("test", 2, 1.0),
            ]
        }
        rng = np.random.default_rng(0)
        rows = regenerate_em_splits(
            original_pairs_by_split=originals,
            ids_present=ids_present,
            pool_positives_by_pair={pair_ab: overlapping_pool},
            interpolated_positives_by_pair={pair_ab: []},
            cluster_positives_by_pair={pair_ab: []},
            negatives_by_pair={pair_ab: []},
            corner_case_negatives_by_pair={pair_ab: set()},
            split_specs_by_pair=specs,
            target_ratio=0.0,
            rng=rng,
        )
        # Every gold pair must end up in its OWN split — no relocation.
        rows_by_split: dict[str, set[tuple[str, str]]] = {
            "train": set(),
            "val": set(),
            "test": set(),
        }
        for row in rows:
            key = (str(row["id1"]), str(row["id2"]))
            rows_by_split[str(row["split"])].add(key)
        for i in range(1, 6):
            assert (f"a{i}", f"b{i}") in rows_by_split[
                "train"
            ], f"train survivor a{i} relocated"
        for i in range(10, 13):
            assert (f"a{i}", f"b{i}") in rows_by_split[
                "val"
            ], f"val survivor a{i} relocated"
        for i in range(20, 22):
            assert (f"a{i}", f"b{i}") in rows_by_split[
                "test"
            ], f"test survivor a{i} relocated"

    def test_regenerate_em_splits_undersizes_when_pool_dry(self) -> None:
        """When backfill pools can't cover the shortfall, accept undersize.

        Every original id is dropped and backfill pools are empty. The
        regenerator emits an empty split rather than reusing ids.
        """
        from usecases_synthetic.lib.corner_case_miner import (
            SplitSpec,
            regenerate_em_splits,
        )

        pair_ab = ("a", "b")
        originals = {
            pair_ab: {
                "test": [(f"a{i}", f"b{i}", True) for i in range(5)]
                + [(f"a{i}", f"b{i + 100}", False) for i in range(5)],
            }
        }
        ids_present: set[str] = set()  # everyone removed
        specs = {pair_ab: [SplitSpec("test", 10, 0.5)]}

        rng = np.random.default_rng(17)
        rows = regenerate_em_splits(
            original_pairs_by_split=originals,
            ids_present=ids_present,
            pool_positives_by_pair={pair_ab: []},
            interpolated_positives_by_pair={pair_ab: []},
            cluster_positives_by_pair={pair_ab: []},
            negatives_by_pair={pair_ab: []},
            corner_case_negatives_by_pair={pair_ab: set()},
            split_specs_by_pair=specs,
            target_ratio=0.4,
            rng=rng,
        )
        assert rows == []

    def test_regenerate_em_test_set_hits_target_ratio(self) -> None:
        # Source-record-level pairs (rid strings) — the canonical→record
        # mapping is the caller's responsibility.
        same = [
            ("p0", "q0"),
            ("p1", "q1"),
            ("p2", "q2"),
            ("p3", "q3"),
            ("p4", "q4"),
        ]
        cross = [
            ("p0", "q1"),
            ("p1", "q3"),
            ("p2", "q4"),
            ("p3", "q0"),
            ("p4", "q2"),
        ]
        corner_neg = {("p0", "q1"), ("p1", "q3")}
        rng = np.random.default_rng(1)
        out = regenerate_em_test_set(
            positive_record_pairs=same,
            negative_record_pairs=cross,
            corner_case_negatives=corner_neg,
            target_ratio=0.4,
            target_size=10,
            rng=rng,
        )
        # At most 10 pairs.
        assert len(out) <= 10
        # The output has both labels (positives from same, negatives from cross).
        labels = {t[2] for t in out}
        assert True in labels and False in labels


# ---- C11 invariants — baseline_pruned + corner_filled dual emission ------


class TestRegenSplitVersionsC11:
    """plan_revision.md C11 (2026-05-22) — emit two parallel versions per
    (pair, split): ``baseline_pruned`` (survivors only) and
    ``corner_filled`` (survivors + 100% corner-mined backfill).
    """

    def _build_minimal_inputs(
        self,
        n_originals: int,
        n_dropped: int,
        interp_count: int,
        corner_neg_count: int,
    ) -> dict[str, object]:
        """Build a one-pair input bundle for the version-invariant tests.

        n_originals positives + n_originals negatives in a single
        ``test`` split (positive_ratio=0.5, size=2 * n_originals).
        n_dropped of the positives reference an id absent from
        ids_present (simulating K2 removal). interp_count interpolated
        positives and corner_neg_count corner negatives are available
        for backfill.
        """
        from usecases_synthetic.lib.corner_case_miner import SplitSpec

        pair_ab = ("a", "b")
        originals = {
            pair_ab: {
                "test": [(f"a{i}", f"b{i}", True) for i in range(n_originals)]
                + [(f"a{i}", f"b{i + 100}", False) for i in range(n_originals)],
            }
        }
        # Drop first n_dropped a-ids (affects both pos and neg sides).
        # ids_present covers a-ids through (n_originals + 5 +
        # corner_neg_count) so the corner-neg pool a-ids survive the
        # eligibility filter.
        ids_present = (
            {f"a{i}" for i in range(n_dropped, n_originals + 5 + corner_neg_count + 1)}
            | {f"b{i}" for i in range(0, n_originals + 200)}
            | {f"interp{k}__{s}" for k in range(interp_count) for s in ("a", "b")}
        )
        interp_pos = {
            pair_ab: [(f"interp{k}__a", f"interp{k}__b") for k in range(interp_count)]
        }
        corner_negs_set = {
            (f"a{i}", f"b{i + 130}")
            for i in range(n_originals + 5, n_originals + 5 + corner_neg_count)
        }
        # Generous easy-neg seed (must NOT be consumed under C11).
        easy_neg_seed = [
            (f"a{i}", f"b{j + 170}")
            for i in range(n_dropped, n_originals)
            for j in range(20)
        ]
        negatives = {pair_ab: sorted(corner_negs_set) + easy_neg_seed}
        corner_negs = {pair_ab: corner_negs_set}
        specs = {pair_ab: [SplitSpec("test", 2 * n_originals, 0.5)]}
        return {
            "pair_ab": pair_ab,
            "original_pairs_by_split": originals,
            "ids_present": ids_present,
            "pool_positives_by_pair": {pair_ab: []},
            "interpolated_positives_by_pair": interp_pos,
            "cluster_positives_by_pair": {pair_ab: []},
            "negatives_by_pair": negatives,
            "corner_case_negatives_by_pair": corner_negs,
            "split_specs_by_pair": specs,
            "target_ratio": 0.5,
            "rng": np.random.default_rng(42),
        }

    def test_set1_subset_of_set2(self) -> None:
        """Set 1 (baseline_pruned) ⊂ Set 2 (corner_filled), always."""
        from usecases_synthetic.lib.corner_case_miner import (
            VERSION_BASELINE_PRUNED,
            VERSION_CORNER_FILLED,
            regenerate_em_splits,
        )

        inputs = self._build_minimal_inputs(
            n_originals=10, n_dropped=5, interp_count=10, corner_neg_count=10
        )
        # Drop the `pair_ab` key — it's only used by the test helper.
        inputs.pop("pair_ab")
        rows = regenerate_em_splits(**inputs)  # type: ignore[arg-type]
        df = pd.DataFrame(rows)
        bp_pairs = {
            (r["id1"], r["id2"])
            for _, r in df[df["version"] == VERSION_BASELINE_PRUNED].iterrows()
        }
        cf_pairs = {
            (r["id1"], r["id2"])
            for _, r in df[df["version"] == VERSION_CORNER_FILLED].iterrows()
        }
        assert bp_pairs, "baseline_pruned must be non-empty when survivors exist"
        missing = bp_pairs - cf_pairs
        assert not missing, (
            f"Set 1 ⊄ Set 2 invariant violated; missing in corner_filled: " f"{missing}"
        )

    def test_corner_filled_size_matches_original_when_pool_sufficient(
        self,
    ) -> None:
        """Set 2 hits |original| when both corner pools cover the shortfall."""
        from usecases_synthetic.lib.corner_case_miner import (
            VERSION_CORNER_FILLED,
            regenerate_em_splits,
        )

        # 10 pos + 10 neg originals, drop 5 pos → 5 pos slots to backfill;
        # interp_count=5 corner positives exactly cover the shortfall.
        # Negatives all survive (no a-side drop affects b{i+100} side
        # except for a0-a4 — so 5 neg surviving and 5 corner negs
        # needed).
        inputs = self._build_minimal_inputs(
            n_originals=10, n_dropped=5, interp_count=5, corner_neg_count=5
        )
        inputs.pop("pair_ab")
        rows = regenerate_em_splits(**inputs)  # type: ignore[arg-type]
        df = pd.DataFrame(rows)
        cf = df[df["version"] == VERSION_CORNER_FILLED]
        assert len(cf) == 20, f"expected |corner_filled| == 20, got {len(cf)}"

    def test_corner_filled_undersizes_when_corner_pool_dry(self) -> None:
        """When corner pool can't cover the shortfall, accept undersize.

        Per C11 option (i) — no easy spillover. interp_count=2 means
        only 2 of the 5 positive slots can be filled; the realised
        size for corner_filled positives is 5 survivors + 2 corners
        = 7, not 10.
        """
        from usecases_synthetic.lib.corner_case_miner import (
            VERSION_CORNER_FILLED,
            regenerate_em_splits,
        )

        inputs = self._build_minimal_inputs(
            n_originals=10, n_dropped=5, interp_count=2, corner_neg_count=5
        )
        inputs.pop("pair_ab")
        rows = regenerate_em_splits(**inputs)  # type: ignore[arg-type]
        df = pd.DataFrame(rows)
        cf = df[df["version"] == VERSION_CORNER_FILLED]
        cf_pos = cf[cf["label"] == "true"]
        # 5 surviving + 2 interp picked = 7 (undersize).
        assert (
            len(cf_pos) == 7
        ), f"expected 7 positives (5 surv + 2 corners), got {len(cf_pos)}"

    def test_corner_filled_excludes_easy_backfill(self) -> None:
        """corner_filled never consumes easy-positive or easy-negative pools.

        cluster_positives / pool_positives / non-corner negatives are
        present but must NOT appear in either version under C11.
        """
        from usecases_synthetic.lib.corner_case_miner import (
            SplitSpec,
            VERSION_CORNER_FILLED,
            regenerate_em_splits,
        )

        pair_ab = ("a", "b")
        # 10 originals, drop 5 positives → need 5 pos backfill, 5 neg
        # backfill. interp_count=5 + corner_neg_count=5 exactly cover
        # the corner-backfill demand. Generous EASY pools provided —
        # must NOT be consumed.
        originals = {
            pair_ab: {
                "test": [(f"a{i}", f"b{i}", True) for i in range(10)]
                + [(f"a{i}", f"b{i + 100}", False) for i in range(10)],
            }
        }
        ids_present = (
            {f"a{i}" for i in range(5, 200)}
            | {f"b{i}" for i in range(200)}
            | {f"interp{k}__{s}" for k in range(5) for s in ("a", "b")}
        )
        easy_pos_cluster = [(f"a{i}", f"b{i + 150}") for i in range(20, 30)]
        easy_pos_pool = [(f"a{i}", f"b{i + 160}") for i in range(30, 40)]
        interp_pos = [(f"interp{k}__a", f"interp{k}__b") for k in range(5)]
        corner_negs_set = {(f"a{i}", f"b{i + 130}") for i in range(40, 45)}
        easy_neg_seed = [
            (f"a{i}", f"b{j + 170}") for i in range(20, 30) for j in range(20)
        ]
        rng = np.random.default_rng(7)
        rows = regenerate_em_splits(
            original_pairs_by_split=originals,
            ids_present=ids_present,
            pool_positives_by_pair={pair_ab: easy_pos_pool},
            interpolated_positives_by_pair={pair_ab: interp_pos},
            cluster_positives_by_pair={pair_ab: easy_pos_cluster},
            negatives_by_pair={pair_ab: sorted(corner_negs_set) + easy_neg_seed},
            corner_case_negatives_by_pair={pair_ab: corner_negs_set},
            split_specs_by_pair={pair_ab: [SplitSpec("test", 20, 0.5)]},
            target_ratio=0.5,
            rng=rng,
        )
        df = pd.DataFrame(rows)
        cf = df[df["version"] == VERSION_CORNER_FILLED]
        cf_pairs = {(r["id1"], r["id2"]) for _, r in cf.iterrows()}
        easy_pos_set = set(easy_pos_cluster) | set(easy_pos_pool)
        easy_neg_set = set(easy_neg_seed)
        leaked_easy_pos = cf_pairs & easy_pos_set
        leaked_easy_neg = cf_pairs & easy_neg_set
        assert (
            not leaked_easy_pos
        ), f"corner_filled leaked easy positives: {leaked_easy_pos}"
        assert (
            not leaked_easy_neg
        ), f"corner_filled leaked easy negatives: {leaked_easy_neg}"

    def test_dropped_ids_pruned_from_baseline_pruned(self) -> None:
        """K2/K3/K4-removed ids never appear in baseline_pruned."""
        from usecases_synthetic.lib.corner_case_miner import (
            VERSION_BASELINE_PRUNED,
            regenerate_em_splits,
        )

        inputs = self._build_minimal_inputs(
            n_originals=10, n_dropped=5, interp_count=0, corner_neg_count=0
        )
        ids_present = inputs["ids_present"]
        inputs.pop("pair_ab")
        rows = regenerate_em_splits(**inputs)  # type: ignore[arg-type]
        df = pd.DataFrame(rows)
        bp = df[df["version"] == VERSION_BASELINE_PRUNED]
        for _, row in bp.iterrows():
            assert row["id1"] in ids_present, f"id1={row['id1']} pruned-id leak"
            assert row["id2"] in ids_present, f"id2={row['id2']} pruned-id leak"

    def test_positive_ratio_preserved_on_corner_filled(self) -> None:
        """corner_filled targets the original split's positive_ratio."""
        from usecases_synthetic.lib.corner_case_miner import (
            VERSION_CORNER_FILLED,
            regenerate_em_splits,
        )

        inputs = self._build_minimal_inputs(
            n_originals=10, n_dropped=5, interp_count=5, corner_neg_count=5
        )
        inputs.pop("pair_ab")
        rows = regenerate_em_splits(**inputs)  # type: ignore[arg-type]
        df = pd.DataFrame(rows)
        cf = df[df["version"] == VERSION_CORNER_FILLED]
        n_pos = (cf["label"] == "true").sum()
        n_neg = (cf["label"] == "false").sum()
        # spec.positive_ratio = 0.5 → expect 50/50 split when pools cover.
        assert (
            n_pos == 10 and n_neg == 10
        ), f"positive_ratio drift: pos={n_pos}, neg={n_neg}"


class TestVariantLoaderRegenVersions:
    """plan_revision.md C11 — variant_loader reads both versions per (pair, split)."""

    def test_loads_per_split_per_version(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        """Writer emits 2 versions × N splits; loader picks them up."""
        from usecases_synthetic.lib.variant_loader import _load_em_gold_regenerated

        em_dir = tmp_path / "input" / "entitymatching"
        em_dir.mkdir(parents=True)
        for split in ("train", "val", "test"):
            for version in ("baseline_pruned", "corner_filled"):
                df = pd.DataFrame(
                    {
                        "id1": [f"a{split}_{version}_0"],
                        "id2": [f"b{split}_{version}_0"],
                        "source_1": ["a"],
                        "source_2": ["b"],
                        "label": ["true"],
                    }
                )
                path = em_dir / f"a_2_b_{split}_{version}.csv"
                df.to_csv(path, index=False)

        out = _load_em_gold_regenerated(em_dir, [("a", "b")])
        assert ("a", "b") in out
        per_split = out[("a", "b")]
        assert set(per_split.keys()) == {"train", "val", "test"}
        for split in ("train", "val", "test"):
            per_version = per_split[split]
            assert set(per_version.keys()) == {"baseline_pruned", "corner_filled"}
            for version, frame in per_version.items():
                assert set(frame.columns) == {"id1", "id2", "label"}
                assert len(frame) == 1
                assert frame.iloc[0]["id1"] == f"a{split}_{version}_0"

    def test_partial_versions_loaded(self, tmp_path: pytest.TempPathFactory) -> None:
        """Only corner_filled present (e.g. legacy partial regen)."""
        from usecases_synthetic.lib.variant_loader import _load_em_gold_regenerated

        em_dir = tmp_path / "input" / "entitymatching"
        em_dir.mkdir(parents=True)
        df = pd.DataFrame(
            {
                "id1": ["a1"],
                "id2": ["b1"],
                "source_1": ["a"],
                "source_2": ["b"],
                "label": ["true"],
            }
        )
        df.to_csv(em_dir / "a_2_b_test_corner_filled.csv", index=False)
        out = _load_em_gold_regenerated(em_dir, [("a", "b")])
        per_split = out[("a", "b")]
        # Only ``test`` split present, only ``corner_filled`` version.
        assert set(per_split.keys()) == {"test"}
        assert set(per_split["test"].keys()) == {"corner_filled"}
        frame = per_split["test"]["corner_filled"]
        assert list(frame.columns) == ["id1", "id2", "label"]
        assert len(frame) == 1
        assert frame.iloc[0]["id1"] == "a1"
        # pandas may parse the literal string ``"true"`` as a numpy
        # bool — compare via str() so both forms accept.
        assert str(frame.iloc[0]["label"]).lower() == "true"

    def test_legacy_regenerated_file_ignored(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        """Pre-C11 ``*_regenerated.csv`` files are no longer loaded."""
        from usecases_synthetic.lib.variant_loader import _load_em_gold_regenerated

        em_dir = tmp_path / "input" / "entitymatching"
        em_dir.mkdir(parents=True)
        df = pd.DataFrame(
            {
                "id1": ["a1"],
                "id2": ["b1"],
                "source_1": ["a"],
                "source_2": ["b"],
                "label": ["true"],
            }
        )
        df.to_csv(em_dir / "a_2_b_test_regenerated.csv", index=False)
        out = _load_em_gold_regenerated(em_dir, [("a", "b")])
        assert out == {}, "legacy *_regenerated.csv files should be ignored under C11"


# ---- Source-record pair helpers (S1 fix) ---------------------------------


class TestSourceRecordPairHelpers:
    def test_enumerate_cross_source_positive_pairs_basic(self) -> None:
        from usecases_synthetic.scripts.apply_knob_02_niche import (
            enumerate_cross_source_positive_pairs,
        )

        groups = {
            "e0": [("a", "a_0"), ("b", "b_0")],
            "e1": [("a", "a_1")],  # singleton, no positive emitted
            "e2": [("a", "a_2"), ("b", "b_2"), ("c", "c_2")],
        }
        pairs = enumerate_cross_source_positive_pairs(groups)
        # e0 contributes 1 pair; e2 contributes 3 cross-source pairs.
        assert ("a_0", "b_0") in pairs
        assert ("a_2", "b_2") in pairs
        assert ("a_2", "c_2") in pairs
        assert ("b_2", "c_2") in pairs
        assert len(pairs) == 4

    def test_enumerate_cross_source_positive_pairs_excluded(self) -> None:
        from usecases_synthetic.scripts.apply_knob_02_niche import (
            enumerate_cross_source_positive_pairs,
        )

        groups = {
            "e0": [("a", "a_0"), ("b", "b_0")],
            "e1": [("a", "a_1"), ("b", "b_1")],
        }
        pairs = enumerate_cross_source_positive_pairs(
            groups, excluded_canonical_ids={"e1"}
        )
        assert pairs == [("a_0", "b_0")]

    def test_enumerate_cross_source_positive_pairs_filter(self) -> None:
        from usecases_synthetic.scripts.apply_knob_02_niche import (
            enumerate_cross_source_positive_pairs,
        )

        groups = {
            "e0": [("a", "a_0"), ("b", "b_0"), ("c", "c_0")],
        }
        # Only {a, b} is in the authored EM gold pair list.
        pairs = enumerate_cross_source_positive_pairs(
            groups, source_pair_filter={frozenset({"a", "b"})}
        )
        assert pairs == [("a_0", "b_0")]

    def test_enumerate_cross_source_positive_pairs_skips_same_source(
        self,
    ) -> None:
        from usecases_synthetic.scripts.apply_knob_02_niche import (
            enumerate_cross_source_positive_pairs,
        )

        # Same-source dups within a cluster are NOT positives.
        groups = {"e0": [("a", "a_0"), ("a", "a_1")]}
        assert enumerate_cross_source_positive_pairs(groups) == []

    def test_pick_cross_cluster_record_pair_prefers_filter(self) -> None:
        from usecases_synthetic.scripts.apply_knob_02_niche import (
            pick_cross_cluster_record_pair,
        )

        rng = np.random.default_rng(0)
        groups = {
            "e0": [("a", "a_0"), ("c", "c_0")],
            "e1": [("b", "b_1"), ("c", "c_1")],
        }
        # Only (a, b) is in the filter — c-side options should be skipped.
        pair = pick_cross_cluster_record_pair(
            "e0",
            "e1",
            groups,
            source_pair_filter={frozenset({"a", "b"})},
            rng=rng,
        )
        assert pair == ("a_0", "b_1")

    def test_pick_cross_cluster_record_pair_no_cross_source(self) -> None:
        from usecases_synthetic.scripts.apply_knob_02_niche import (
            pick_cross_cluster_record_pair,
        )

        rng = np.random.default_rng(0)
        groups = {
            "e0": [("a", "a_0")],
            "e1": [("a", "a_1")],
        }
        # Both entities only have a-source records — no valid cross-source
        # negative exists.
        assert (
            pick_cross_cluster_record_pair(
                "e0", "e1", groups, source_pair_filter=None, rng=rng
            )
            is None
        )

    def test_pick_cross_cluster_record_pair_strict_filter(self) -> None:
        """Unauthored source pairs are rejected (not silently downgraded)."""
        from usecases_synthetic.scripts.apply_knob_02_niche import (
            pick_cross_cluster_record_pair,
        )

        rng = np.random.default_rng(0)
        groups = {
            "e0": [("a", "a_0")],
            "e1": [("b", "b_1")],
        }
        # (a, b) is cross-source but not in the filter — return None
        # rather than fall back to an unauthored pair. This keeps every
        # emitted negative on an authored source-pair so the variant
        # loader's per-pair split never drops rows silently.
        assert (
            pick_cross_cluster_record_pair(
                "e0",
                "e1",
                groups,
                source_pair_filter={frozenset({"a", "c"})},
                rng=rng,
            )
            is None
        )


class TestLoadPoolPositivesByPair:
    """Regression: pool CSV id1/id2 are lex-sorted, not source-aligned.

    Before the rid_to_source orientation fix, ``_load_pool_positives_by_pair``
    assumed ``id1`` corresponded to ``source_1`` per-row. The pool writer
    actually canonicalises ``(id1, id2)`` lex-wise via ``canonical_pair``,
    so when ``id2`` lex-sorts smaller than ``id1`` from src1, the loader
    placed the wrong id in src1's slot. Downstream this produced regen
    rows with ``id1`` from src2 — see plan_s1_final.md F6.
    """

    def _write_pool(
        self, tmp_path: Path, rows: list[tuple[str, str, str, str]]
    ) -> Path:
        domain = "tiny_test_dom"
        pool_dir = tmp_path / domain
        pool_dir.mkdir(parents=True)
        pool_path = pool_dir / "pooled_positives.csv"
        with open(pool_path, "w", encoding="utf-8") as f:
            f.write("id1,id2,source_1,source_2,score\n")
            for id1, id2, s1, s2 in rows:
                f.write(f"{id1},{id2},{s1},{s2},0.9\n")
        return pool_dir

    def test_orientation_via_rid_to_source(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from usecases_synthetic.scripts import apply_knob_02_niche as mod

        # Simulate canonical_pair-sorted pool: "discogs_X" lex < "mbrainz_Y"
        # so the writer puts the discogs id in id1 even though source_1
        # column says musicbrainz (the pair's canonical first source).
        rows = [
            ("discogs_1", "mbrainz_1", "musicbrainz", "discogs"),
            ("discogs_2", "mbrainz_2", "musicbrainz", "discogs"),
            ("mbrainz_3", "discogs_3", "musicbrainz", "discogs"),  # already aligned
        ]
        pools_root = self._write_pool(tmp_path, rows)
        # Point POOLS_DIR at the temp tree and disable alias resolution.
        from usecases_synthetic.lib import domain_config

        monkeypatch.setattr(domain_config, "POOLS_DIR", pools_root.parent)
        monkeypatch.setattr(mod, "POOLS_DIR", pools_root.parent, raising=False)
        monkeypatch.setattr(mod, "resolve_cache_domain", lambda d: d)

        rid_to_source = {
            "mbrainz_1": "musicbrainz",
            "mbrainz_2": "musicbrainz",
            "mbrainz_3": "musicbrainz",
            "discogs_1": "discogs",
            "discogs_2": "discogs",
            "discogs_3": "discogs",
        }
        authored = [("musicbrainz", "discogs")]
        out = mod._load_pool_positives_by_pair(
            "tiny_test_dom", authored, rid_to_source=rid_to_source
        )
        # All emitted pairs must have id1 in src1 (musicbrainz) and
        # id2 in src2 (discogs) regardless of CSV row ordering.
        pairs = out[("musicbrainz", "discogs")]
        assert len(pairs) == 3
        for a, b in pairs:
            assert rid_to_source[a] == "musicbrainz", f"id1={a} not in musicbrainz"
            assert rid_to_source[b] == "discogs", f"id2={b} not in discogs"

    def test_orientation_drops_unknown_ids(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from usecases_synthetic.scripts import apply_knob_02_niche as mod

        rows = [
            ("mbrainz_1", "discogs_1", "musicbrainz", "discogs"),
            ("mbrainz_orphan", "discogs_1", "musicbrainz", "discogs"),
        ]
        pools_root = self._write_pool(tmp_path, rows)
        from usecases_synthetic.lib import domain_config

        monkeypatch.setattr(domain_config, "POOLS_DIR", pools_root.parent)
        monkeypatch.setattr(mod, "POOLS_DIR", pools_root.parent, raising=False)
        monkeypatch.setattr(mod, "resolve_cache_domain", lambda d: d)

        rid_to_source = {
            "mbrainz_1": "musicbrainz",
            "discogs_1": "discogs",
        }
        out = mod._load_pool_positives_by_pair(
            "tiny_test_dom",
            [("musicbrainz", "discogs")],
            rid_to_source=rid_to_source,
        )
        # mbrainz_orphan is not in rid_to_source — drop that row.
        assert out[("musicbrainz", "discogs")] == [("mbrainz_1", "discogs_1")]

    def test_legacy_column_mode_still_works(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """rid_to_source=None falls back to column-based orientation.

        Documented quirk: this misorients when id1/id2 are lex-sorted.
        The test pins the legacy behaviour so callers that don't supply
        rid_to_source aren't surprised by a silent change. New callers
        should always supply rid_to_source.
        """
        from usecases_synthetic.scripts import apply_knob_02_niche as mod

        # Row where id1 lex < id2 but source_1 says id1 is in src "a".
        # Without rid_to_source, the loader trusts the column.
        rows = [("a_only", "b_only", "a", "b")]
        pools_root = self._write_pool(tmp_path, rows)
        from usecases_synthetic.lib import domain_config

        monkeypatch.setattr(domain_config, "POOLS_DIR", pools_root.parent)
        monkeypatch.setattr(mod, "POOLS_DIR", pools_root.parent, raising=False)
        monkeypatch.setattr(mod, "resolve_cache_domain", lambda d: d)

        out = mod._load_pool_positives_by_pair("tiny_test_dom", [("a", "b")])
        assert out[("a", "b")] == [("a_only", "b_only")]


# ---- Entity interpolation tests -------------------------------------------


class TestEntityInterpolation:
    def test_parent_pair_hash_stable(self) -> None:
        h1 = parent_pair_hash(["a", "b"])
        h2 = parent_pair_hash(["b", "a"])
        assert h1 == h2

    def test_contamination_check_flags_real_label_collision(self) -> None:
        status = contamination_check(
            {"name": "Apple Inc"},
            primary_column="name",
            reference_labels={"apple inc", "microsoft"},
        )
        assert status == "collision_with_real_entity"

    def test_contamination_check_passes_for_novel_label(self) -> None:
        status = contamination_check(
            {"name": "Foobar Synthetica"},
            primary_column="name",
            reference_labels={"apple inc"},
        )
        assert status == "passed"

    def test_default_api_client_blends_attributes(self) -> None:
        parents = [
            {"name": "Apple Inc", "country": "US", "founded": 1976},
            {"name": "Pear Corp", "country": "US", "founded": 1980},
        ]
        out = default_api_client_from_attributes("", parents)
        assert out["name"]  # non-empty blended name
        assert out["country"] == "US"
        assert isinstance(out["founded"], float)

    def test_interpolate_entity_cache_hit_respects_committed_result(
        self, tmp_path: Path
    ) -> None:
        cache = LLMCache(tmp_path, prompt_version="v1", model_id="test-model")
        parent_a = pd.Series({"name": "Apple Inc", "country": "US"})
        parent_a.name = "ent_a"
        parent_b = pd.Series({"name": "Pear Corp", "country": "US"})
        parent_b.name = "ent_b"

        entity = interpolate_entity(
            parent_rows=[parent_a, parent_b],
            primary_column="name",
            schema_columns=["name", "country"],
            domain="test",
            prompt_template="",
            llm_cache=cache,
            api_client=default_api_client_from_attributes,
            committee_fn=None,
            reference_labels=set(),
            placement_mode="matched_across",
            source_placements=["src1", "src2"],
            entity_id="k02_interp_test_0",
            strict_cache=False,
        )
        assert entity is not None
        assert entity.attributes["name"]
        assert entity.contamination_check_status == "passed"

        # Second call hits the cache (we can force strict_cache=True now).
        entity2 = interpolate_entity(
            parent_rows=[parent_a, parent_b],
            primary_column="name",
            schema_columns=["name", "country"],
            domain="test",
            prompt_template="",
            llm_cache=cache,
            api_client=None,
            committee_fn=None,
            reference_labels=set(),
            placement_mode="matched_across",
            source_placements=["src1", "src2"],
            entity_id="k02_interp_test_1",
            strict_cache=True,
        )
        assert entity2 is not None

    def test_interpolate_entity_rejection_log_counts_each_guardrail(
        self, tmp_path: Path
    ) -> None:
        cache = LLMCache(tmp_path, prompt_version="v1", model_id="test-model")
        parent_a = pd.Series({"name": "Apple Inc", "country": "US"})
        parent_a.name = "ent_a"
        parent_b = pd.Series({"name": "Pear Corp", "country": "US"})
        parent_b.name = "ent_b"

        rejection_log: dict[str, int] = {}

        # 1) contamination collision — reference set contains the blended label
        blended = default_api_client_from_attributes(
            "", [parent_a.to_dict(), parent_b.to_dict()]
        )
        from usecases_synthetic.lib.niche_metrics import normalize_label

        reference_labels = {normalize_label(str(blended["name"]))}
        entity = interpolate_entity(
            parent_rows=[parent_a, parent_b],
            primary_column="name",
            schema_columns=["name", "country"],
            domain="test",
            prompt_template="",
            llm_cache=cache,
            api_client=default_api_client_from_attributes,
            committee_fn=None,
            reference_labels=reference_labels,
            placement_mode="matched_across",
            source_placements=["src1", "src2"],
            entity_id="k02_interp_collision",
            strict_cache=False,
            rejection_log=rejection_log,
        )
        assert entity is None
        assert rejection_log.get("contamination_collision_with_real_entity", 0) == 1

        # 2) committee rejection — separate cache so we don't reuse step 1.
        cache2 = LLMCache(tmp_path / "c2", prompt_version="v1", model_id="test-model")
        entity = interpolate_entity(
            parent_rows=[parent_a, parent_b],
            primary_column="name",
            schema_columns=["name", "country"],
            domain="test",
            prompt_template="",
            llm_cache=cache2,
            api_client=default_api_client_from_attributes,
            committee_fn=lambda attrs, parents: False,
            reference_labels=set(),
            placement_mode="matched_across",
            source_placements=["src1", "src2"],
            entity_id="k02_interp_committee",
            strict_cache=False,
            rejection_log=rejection_log,
        )
        assert entity is None
        assert rejection_log.get("committee_validation", 0) == 1

        # Successful interpolation does NOT touch the log.
        cache3 = LLMCache(tmp_path / "c3", prompt_version="v1", model_id="test-model")
        entity = interpolate_entity(
            parent_rows=[parent_a, parent_b],
            primary_column="name",
            schema_columns=["name", "country"],
            domain="test",
            prompt_template="",
            llm_cache=cache3,
            api_client=default_api_client_from_attributes,
            committee_fn=None,
            reference_labels=set(),
            placement_mode="matched_across",
            source_placements=["src1", "src2"],
            entity_id="k02_interp_ok",
            strict_cache=False,
            rejection_log=rejection_log,
        )
        assert entity is not None
        # Counts unchanged after a successful pass.
        assert sum(rejection_log.values()) == 2

    def test_select_parent_pairs_excludes_protected_for_single_source(
        self,
    ) -> None:
        ranked = [0, 1, 2, 3]
        neighbour_lookup = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}
        protected = [True, False, False, False]
        rng = np.random.default_rng(7)
        # matched_across: protected parents are admissible
        matched = select_parent_pairs(
            ranked,
            neighbour_lookup=neighbour_lookup,
            protected=protected,
            placement_mode="matched_across",
            k=2,
            rng=rng,
        )
        # single_source: protected parents are NOT admissible
        single = select_parent_pairs(
            ranked,
            neighbour_lookup=neighbour_lookup,
            protected=protected,
            placement_mode="single_source_distractor",
            k=2,
            rng=rng,
        )
        assert all(0 not in pair for pair in single)


# ---- End-to-end dispatcher tests ------------------------------------------


def _mini_companies_config() -> dict[str, Any]:
    """In-memory Knob 02 config for a 6-entity toy fixture."""
    return {
        "domain": "companies",
        "id_columns": {"a": "id", "b": "id"},
        "primary_column_canonical": "name",
        "canonical_schema": ["name", "country", "industry"],
        "attribute_mapping": {
            "a": {
                "id": "id",
                "name": "name",
                "country": "country",
                "industry": "industry",
            },
            "b": {
                "id": "id",
                "name": "name",
                "country": "country",
                "industry": "industry",
            },
        },
        "source_priority": ["a", "b"],
        "text_concat_order": ["name", "industry", "country"],
        "metrics": {
            "ext_jaccard": True,
            "tfidf": True,
            "embedding": False,  # skip for speed
            "attribute_overlap": True,
            "label_collision": True,
        },
        "metric_top_k": 5,
        "rrf_k0": 60,
        "c_min": 2,
        "boost_label_collision": 5.0,
        "inner_token_threshold": 0.8,
        "pair_miner_thresholds": {
            "t_match": {"ext_jaccard": 0.5, "tfidf": 0.4},
            "t_nonmatch": {"ext_jaccard": 0.5, "tfidf": 0.4},
        },
        "attribute_overlap_weights": {
            "industry": 0.5,
            "country": 0.5,
        },
        "stopword_list": ["inc", "corp", "ltd", "the"],
        "levels": {
            "easy": {
                "target_corner_case_ratio": 0.2,
                "removal_fraction_cap": 0.5,
                "interpolation_count": 0,
                "placement_split": 0.0,
            },
            "medium": {
                "target_corner_case_ratio": 0.5,
                "removal_fraction_cap": 0.2,
                "interpolation_count": 0,
                "placement_split": 0.0,
            },
            "hard": {
                "target_corner_case_ratio": 0.6,
                "removal_fraction_cap": 0.1,
                "interpolation_count": 3,
                "placement_split": 0.6,
            },
        },
        "max_interp_fraction": 0.6,
        "em_test_regeneration": {
            "target_size": 10,
            "candidate_pool_cap": 200,
        },
        "llm_prompt_version": "v1",
        "llm_model_id": "test-model",
    }


def _mini_sources() -> dict[str, pd.DataFrame]:
    a = pd.DataFrame(
        {
            "id": [f"a_{i}" for i in range(6)],
            "name": [
                "Apple Inc",
                "Apple Corp",
                "Apple Incorporated",
                "Microsoft",
                "Google",
                "Banana Split",
            ],
            "country": ["US", "US", "US", "US", "US", "BR"],
            "industry": ["tech", "tech", "tech", "tech", "tech", "food"],
        }
    )
    a.attrs["dataset_name"] = "a"
    b = pd.DataFrame(
        {
            "id": [f"b_{i}" for i in range(3)],
            "name": ["Apple Incorporation", "Microsoft Corp", "Yahoo"],
            "country": ["US", "US", "US"],
            "industry": ["tech", "tech", "tech"],
        }
    )
    b.attrs["dataset_name"] = "b"
    return {"a": a, "b": b}


class TestApplyKnob02:
    def test_dispatcher_easy_no_protection_removed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from usecases_synthetic.scripts import apply_knob_02_niche as mod

        # Monkeypatch the canonical view builder + protection so we do
        # not touch the real usecases/ directory.
        def _fake_canonical(domain, sources, config):
            rows = []
            egroups: dict[str, list[tuple[str, str]]] = {}
            id_to_canon: dict[str, str] = {}
            for i in range(6):
                cid = f"k02_ent_{i:06d}"
                rows.append(
                    {
                        "entity_id": cid,
                        "name": sources["a"]["name"].iloc[i],
                        "country": sources["a"]["country"].iloc[i],
                        "industry": sources["a"]["industry"].iloc[i],
                    }
                )
                members = [("a", f"a_{i}")]
                if i < 3:
                    members.append(("b", f"b_{i}"))
                egroups[cid] = members
                for _s, rid in members:
                    id_to_canon[rid] = cid
            df = pd.DataFrame(rows).set_index("entity_id", drop=False)
            df.attrs["dataset_name"] = "companies_canonical"
            return df, egroups, id_to_canon

        monkeypatch.setattr(mod, "build_canonical_view", _fake_canonical)

        config = _mini_companies_config()
        sources = _mini_sources()
        protection = {"a_0", "b_0"}  # canonical entity 0 is protected

        new_sources, canonical, regen, prov, scores, _k2_metrics, _regen_pools = (
            mod.apply_knob_02(
                domain="companies",
                level="easy",
                sources=sources,
                config=config,
                expanded_positives=protection,
                embedding_cache_path=tmp_path / "emb.npy",
                seed=123,
                source_pair_filter={frozenset({"a", "b"})},
            )
        )

        # Acceptance: no protected entity is removed.
        removed = prov[prov["transform_fn"] == "remove_entity"]
        assert "k02_ent_000000" not in set(removed["entity_id"])

        # Canonical contains the surviving entities (protection preserved).
        assert "k02_ent_000000" in canonical["entity_id"].tolist()

        # Regenerated EM test set has labels "true"/"false" only. Under
        # C11 (plan_revision.md, 2026-05-22) the regenerator emits two
        # versions: ``baseline_pruned`` (survivors only — empty here
        # because the fallback spec has no original gold) and
        # ``corner_filled`` (survivors + 100% corner-mined backfill, no
        # easy fills). At K2 easy with no interpolation pool, the only
        # backfill source is the corner-negative pool — so positives
        # are legitimately absent. Pre-C11 behaviour relied on easy
        # cluster-positive backfill which has been intentionally removed.
        assert not regen.empty
        assert set(regen["label"].unique()).issubset({"true", "false"})
        assert "version" in regen.columns
        assert set(regen["version"].unique()).issubset(
            {"baseline_pruned", "corner_filled"}
        )
        # IDs must be source-record IDs (a_* / b_*), not canonical IDs.
        flat_ids = set(regen["id1"]).union(set(regen["id2"]))
        assert not any(
            rid.startswith("k02_ent_") for rid in flat_ids
        ), "Regenerated test must emit source-record IDs, not canonical IDs."

        # Niche-score audit written for every entity.
        assert len(scores) == 6
        assert "density" in scores.columns

    def test_drop_corner_protection_always_gold_regardless_of_cli_flag(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Regression for the 2026-05-28 C13-vs-Bug-3 conflict.

        Under plan_revision.md §C13, ``--protection-source`` only
        affects K1/K6 drift protection. K2's existence protection (the
        drop-corner protection set) must STAY gold-only — i.e. fusion
        val/test only — regardless of the CLI flag. Otherwise on
        pool-live domains (products) the K2 dial dies under silver
        mode: the broader protection set covers every pool member, no
        drops happen, no clusters break, the C13 intact-cluster gate
        becomes a noop.

        The contract: ``apply_knob_02(protection_source="silver")``
        must still call ``build_drop_corner_protection_set(domain,
        protection_source="gold")``.
        """
        from usecases_synthetic.scripts import apply_knob_02_niche as mod
        from usecases_synthetic.lib import protection as proto_mod

        # Reuse the same fake canonical view from the easy-no-protection
        # test so we don't need real source data.
        def _fake_canonical(domain, sources, config):
            rows = []
            egroups: dict[str, list[tuple[str, str]]] = {}
            id_to_canon: dict[str, str] = {}
            for i in range(6):
                cid = f"k02_ent_{i:06d}"
                rows.append(
                    {
                        "entity_id": cid,
                        "name": sources["a"]["name"].iloc[i],
                        "country": sources["a"]["country"].iloc[i],
                        "industry": sources["a"]["industry"].iloc[i],
                    }
                )
                members = [("a", f"a_{i}")]
                if i < 3:
                    members.append(("b", f"b_{i}"))
                egroups[cid] = members
                for _s, rid in members:
                    id_to_canon[rid] = cid
            df = pd.DataFrame(rows).set_index("entity_id", drop=False)
            df.attrs["dataset_name"] = "companies_canonical"
            return df, egroups, id_to_canon

        monkeypatch.setattr(mod, "build_canonical_view", _fake_canonical)

        captured_calls: list[dict[str, object]] = []

        def fake_build_drop_corner_protection_set(domain, protection_source="gold"):
            captured_calls.append(
                {"domain": domain, "protection_source": protection_source}
            )
            # Return a small set — the apply_knob_02 internals don't care
            # about content for this test.
            return {"a_0", "b_0"}

        # Patch at the module where apply_knob_02 imported the symbol
        # from (not at the source module).
        monkeypatch.setattr(
            mod,
            "build_drop_corner_protection_set",
            fake_build_drop_corner_protection_set,
        )

        config = _mini_companies_config()
        sources = _mini_sources()

        mod.apply_knob_02(
            domain="companies",
            level="easy",
            sources=sources,
            config=config,
            expanded_positives={"a_0", "b_0"},
            embedding_cache_path=tmp_path / "emb.npy",
            seed=123,
            source_pair_filter={frozenset({"a", "b"})},
            protection_source="silver",  # ← key part: CLI says silver
        )

        # Even though apply_knob_02 received protection_source="silver",
        # the drop-corner builder must have been called with "gold".
        assert captured_calls, "build_drop_corner_protection_set was never called"
        for call in captured_calls:
            assert call["protection_source"] == "gold", (
                f"K2 drop-corner protection must always be 'gold' regardless of "
                f"the CLI flag (C13 design). Got call kwargs: {call!r}"
            )

    def test_dispatcher_hard_interpolation_creates_valid_entities(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from usecases_synthetic.scripts import apply_knob_02_niche as mod

        def _fake_canonical(domain, sources, config):
            rows = []
            egroups: dict[str, list[tuple[str, str]]] = {}
            id_to_canon: dict[str, str] = {}
            for i in range(6):
                cid = f"k02_ent_{i:06d}"
                rows.append(
                    {
                        "entity_id": cid,
                        "name": sources["a"]["name"].iloc[i],
                        "country": sources["a"]["country"].iloc[i],
                        "industry": sources["a"]["industry"].iloc[i],
                    }
                )
                egroups[cid] = [("a", f"a_{i}")]
                id_to_canon[f"a_{i}"] = cid
            df = pd.DataFrame(rows).set_index("entity_id", drop=False)
            return df, egroups, id_to_canon

        monkeypatch.setattr(mod, "build_canonical_view", _fake_canonical)

        config = _mini_companies_config()
        sources = _mini_sources()
        cache = LLMCache(
            tmp_path / "interp_cache",
            prompt_version="v1",
            model_id="test-model",
        )

        new_sources, canonical, regen, prov, scores, _k2_metrics, _regen_pools = (
            mod.apply_knob_02(
                domain="companies",
                level="hard",
                sources=sources,
                config=config,
                expanded_positives=set(),
                llm_cache=cache,
                # Leave api_client as None + strict_cache=False so the
                # default attribute-blender fabricates interpolation payloads.
                api_client=None,
                strict_cache=False,
                embedding_cache_path=tmp_path / "emb.npy",
                seed=7,
            )
        )

        interp_rows = prov[prov["transform_fn"] == "llm_interpolate_entity"]
        if len(interp_rows) > 0:
            # Acceptance: every interpolated entity has a non-empty
            # primary label in the canonical frame.
            canon_interp = canonical[
                canonical["entity_id"].astype(str).str.startswith("k02_interp_")
            ]
            assert (canon_interp["name"].astype(str).str.len() > 0).all()

    def test_dispatcher_deterministic_under_fixed_seed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from usecases_synthetic.scripts import apply_knob_02_niche as mod

        def _fake_canonical(domain, sources, config):
            rows = []
            egroups: dict[str, list[tuple[str, str]]] = {}
            id_to_canon: dict[str, str] = {}
            for i in range(6):
                cid = f"k02_ent_{i:06d}"
                rows.append(
                    {
                        "entity_id": cid,
                        "name": sources["a"]["name"].iloc[i],
                        "country": sources["a"]["country"].iloc[i],
                        "industry": sources["a"]["industry"].iloc[i],
                    }
                )
                egroups[cid] = [("a", f"a_{i}")]
                id_to_canon[f"a_{i}"] = cid
            df = pd.DataFrame(rows).set_index("entity_id", drop=False)
            return df, egroups, id_to_canon

        monkeypatch.setattr(mod, "build_canonical_view", _fake_canonical)

        config = _mini_companies_config()

        def _run():
            return mod.apply_knob_02(
                domain="companies",
                level="easy",
                sources=_mini_sources(),
                config=config,
                expanded_positives=set(),
                embedding_cache_path=tmp_path / "emb.npy",
                seed=999,
            )

        _, canon1, regen1, prov1, _, _, _ = _run()
        _, canon2, regen2, prov2, _, _, _ = _run()
        pd.testing.assert_frame_equal(
            canon1.reset_index(drop=True), canon2.reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(
            prov1.reset_index(drop=True), prov2.reset_index(drop=True)
        )


# ---- Hard-negative score-margin gate --------------------------------------


class TestHardNegativePolicy:
    def test_noop_when_policy_is_none(self) -> None:
        from usecases_synthetic.lib.corner_case_miner import (
            apply_hard_negative_policy,
        )

        pairs = [("a", "b"), ("c", "d")]
        kept, audit = apply_hard_negative_policy(pairs, policy=None)
        assert kept == pairs
        assert len(audit) == 2
        assert all(row.verdict == "no_score" for row in audit)

    def test_noop_when_plm_scorer_is_none(self) -> None:
        from usecases_synthetic.lib.corner_case_miner import (
            HardNegativePolicy,
            apply_hard_negative_policy,
        )

        policy = HardNegativePolicy(
            plm_scorer=None,
            plm_threshold_theta=0.5,
            plm_margin_delta=0.1,
        )
        pairs = [("a", "b")]
        kept, audit = apply_hard_negative_policy(pairs, policy=policy)
        assert kept == pairs
        assert audit[0].verdict == "no_score"
        assert audit[0].theta == 0.5
        assert audit[0].delta == 0.1

    def test_gate_keeps_strong_drops_above_theta(self) -> None:
        from usecases_synthetic.lib.corner_case_miner import (
            HardNegativePolicy,
            apply_hard_negative_policy,
        )

        scores = {
            ("a", "b"): 0.1,  # strong no — keep
            ("c", "d"): 0.9,  # strong yes — drop
            ("e", "f"): 0.42,  # margin band [0.40, 0.50) — needs LLM
        }

        def scorer(pairs):  # type: ignore[no-untyped-def]
            return {p: scores[p] for p in pairs if p in scores}

        # No adjudicator: margin band conservatively dropped.
        policy = HardNegativePolicy(
            plm_scorer=scorer,
            plm_threshold_theta=0.5,
            plm_margin_delta=0.1,
            llm_adjudicator=None,
        )
        kept, audit = apply_hard_negative_policy(
            [("a", "b"), ("c", "d"), ("e", "f")], policy=policy
        )
        assert kept == [("a", "b")]
        verdicts = {(r.rid_a, r.rid_b): r.verdict for r in audit}
        assert verdicts[("a", "b")] == "keep_strong"
        assert verdicts[("c", "d")] == "drop_above_theta"
        assert verdicts[("e", "f")] == "drop_adjudicated"

    def test_adjudicator_rescues_margin_band(self) -> None:
        from usecases_synthetic.lib.corner_case_miner import (
            HardNegativePolicy,
            apply_hard_negative_policy,
        )

        def scorer(pairs):  # type: ignore[no-untyped-def]
            return {("e", "f"): 0.42, ("g", "h"): 0.45}

        # Adjudicator says (e, f) is a match → drop;
        # says (g, h) is NOT a match → keep.
        def adjudicator(pair):  # type: ignore[no-untyped-def]
            return pair == ("e", "f")

        policy = HardNegativePolicy(
            plm_scorer=scorer,
            plm_threshold_theta=0.5,
            plm_margin_delta=0.1,
            llm_adjudicator=adjudicator,
        )
        kept, audit = apply_hard_negative_policy(
            [("e", "f"), ("g", "h")], policy=policy
        )
        assert kept == [("g", "h")]
        verdicts = {(r.rid_a, r.rid_b): (r.verdict, r.llm_says_match) for r in audit}
        assert verdicts[("e", "f")] == ("drop_adjudicated", True)
        assert verdicts[("g", "h")] == ("keep_adjudicated", False)

    def test_missing_score_keeps_pair_conservatively(self) -> None:
        from usecases_synthetic.lib.corner_case_miner import (
            HardNegativePolicy,
            apply_hard_negative_policy,
        )

        def scorer(pairs):  # type: ignore[no-untyped-def]
            return {}  # every pair has no score

        policy = HardNegativePolicy(
            plm_scorer=scorer,
            plm_threshold_theta=0.5,
            plm_margin_delta=0.1,
        )
        kept, audit = apply_hard_negative_policy([("a", "b")], policy=policy)
        assert kept == [("a", "b")]
        assert audit[0].verdict == "no_score"
        assert audit[0].plm_score is None

    def test_boundary_theta_minus_delta_keeps_strong(self) -> None:
        from usecases_synthetic.lib.corner_case_miner import (
            HardNegativePolicy,
            apply_hard_negative_policy,
        )

        # Score exactly at theta - delta = 0.40 is the boundary:
        # score < 0.40 → keep_strong; score == 0.40 falls in margin band.
        def scorer(pairs):  # type: ignore[no-untyped-def]
            return {("a", "b"): 0.399, ("c", "d"): 0.40}

        policy = HardNegativePolicy(
            plm_scorer=scorer,
            plm_threshold_theta=0.5,
            plm_margin_delta=0.1,
        )
        kept, audit = apply_hard_negative_policy(
            [("a", "b"), ("c", "d")], policy=policy
        )
        assert ("a", "b") in kept
        assert ("c", "d") not in kept  # margin band, no adjudicator → drop
        verdicts = {(r.rid_a, r.rid_b): r.verdict for r in audit}
        assert verdicts[("a", "b")] == "keep_strong"
        assert verdicts[("c", "d")] == "drop_adjudicated"

    def test_invalid_theta_or_delta_raises(self) -> None:
        from usecases_synthetic.lib.corner_case_miner import (
            HardNegativePolicy,
            apply_hard_negative_policy,
        )

        def scorer(pairs):  # type: ignore[no-untyped-def]
            return {("a", "b"): 0.5}

        policy_bad_theta = HardNegativePolicy(
            plm_scorer=scorer,
            plm_threshold_theta=1.5,
            plm_margin_delta=0.1,
        )
        with pytest.raises(ValueError):
            apply_hard_negative_policy([("a", "b")], policy=policy_bad_theta)

        policy_bad_delta = HardNegativePolicy(
            plm_scorer=scorer,
            plm_threshold_theta=0.5,
            plm_margin_delta=-0.1,
        )
        with pytest.raises(ValueError):
            apply_hard_negative_policy([("a", "b")], policy=policy_bad_delta)


class TestLlmAdjudicator:
    def test_adjudicator_uses_cache_and_parses_yes_no(self, tmp_path: Path) -> None:
        from usecases_synthetic.lib.hard_negative_plm import build_llm_adjudicator

        sources = {
            "s1": pd.DataFrame(
                [
                    {"id": "a", "name": "Apple Inc."},
                    {"id": "b", "name": "Microsoft"},
                ]
            ),
            "s2": pd.DataFrame(
                [
                    {"id": "x", "name": "Apple"},
                    {"id": "y", "name": "Google LLC"},
                ]
            ),
        }
        id_columns = {"s1": "id", "s2": "id"}
        attribute_mapping = {
            "s1": {"name": "name"},
            "s2": {"name": "name"},
        }

        calls: list[str] = []

        def api(prompt: str) -> str:
            calls.append(prompt)
            # First cell is apple/apple → yes; second is microsoft/google → no.
            if "Apple" in prompt and "Apple" in prompt.split("Record B")[1]:
                return "yes"
            return "no"

        cache = LLMCache(
            cache_dir=tmp_path / "cache",
            prompt_version="v1",
            model_id="test-model",
        )
        adjudicate = build_llm_adjudicator(
            domain="t",
            sources=sources,
            id_columns=id_columns,
            attribute_mapping=attribute_mapping,
            fields=["name"],
            llm_cache=cache,
            api_client=api,
        )

        assert adjudicate(("a", "x")) is True  # apple/apple → match
        assert adjudicate(("b", "y")) is False  # microsoft/google → no

        # Second call to same pair should hit cache (no new api call).
        before = len(calls)
        assert adjudicate(("a", "x")) is True
        assert len(calls) == before

    def test_adjudicator_unknown_record_returns_false(self, tmp_path: Path) -> None:
        from usecases_synthetic.lib.hard_negative_plm import build_llm_adjudicator

        sources = {"s": pd.DataFrame([{"id": "a", "name": "Apple"}])}
        cache = LLMCache(
            cache_dir=tmp_path / "cache",
            prompt_version="v1",
            model_id="test-model",
        )
        adjudicate = build_llm_adjudicator(
            domain="t",
            sources=sources,
            id_columns={"s": "id"},
            attribute_mapping={"s": {"name": "name"}},
            fields=["name"],
            llm_cache=cache,
            api_client=lambda _p: "yes",
        )
        assert adjudicate(("a", "missing_rid")) is False


class TestOpenAIInterpolationClient:
    """Tests for the K2 OpenAI interpolation client (plan_revision C1)."""

    def _build_with_chat(self, monkeypatch: pytest.MonkeyPatch, fake_chat: Any) -> Any:
        from usecases_synthetic.lib import entity_interpolation as ei
        from usecases_synthetic.lib import llm_client

        def _fake_build(
            *, model: str, temperature: float = 0.0, max_tokens: int | None = None
        ) -> Any:
            return fake_chat

        # The constructor imports ``build_chat_openai`` lazily from
        # ``llm_client``, so patching that module's attribute is sufficient.
        monkeypatch.setattr(llm_client, "build_chat_openai", _fake_build)
        return ei.build_openai_interpolation_client(
            model_id="gpt-5.4-mini", max_tokens=2048
        )

    def test_substitutes_placeholders_and_parses_json(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}

        class FakeChat:
            def invoke(self, prompt: str) -> Any:
                captured["prompt"] = prompt

                class _R:
                    content = json.dumps(
                        {"name": "Synth Co", "industry": "Tech", "country": "US"}
                    )

                return _R()

        client = self._build_with_chat(monkeypatch, FakeChat())

        template = "Parents: {parent_records_json}\nSchema: {schema_columns_json}"
        parents = [
            {"name": "Apple", "industry": "Tech", "country": "US"},
            {"name": "Banana", "industry": "Food", "country": "UK"},
        ]
        out = client(template, parents)

        # Schema column order follows insertion order from the first parent.
        assert '"name", "industry", "country"' in captured["prompt"]
        # Parents are JSON-dumped.
        assert "Apple" in captured["prompt"]
        assert "Banana" in captured["prompt"]
        assert out == {"name": "Synth Co", "industry": "Tech", "country": "US"}

    def test_strips_code_fence_wrapper(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeChat:
            def invoke(self, prompt: str) -> Any:
                class _R:
                    content = '```json\n{"name": "Synth", "value": 1}\n```'

                return _R()

        client = self._build_with_chat(monkeypatch, FakeChat())
        out = client(
            "{parent_records_json} {schema_columns_json}",
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
            "{parent_records_json} {schema_columns_json}",
            [{"name": "X"}],
        )
        assert out == {}

    def test_returns_empty_on_non_dict_payload(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FakeChat:
            def invoke(self, prompt: str) -> Any:
                class _R:
                    content = json.dumps(["a", "b"])  # JSON array, not object

                return _R()

        client = self._build_with_chat(monkeypatch, FakeChat())
        out = client(
            "{parent_records_json} {schema_columns_json}",
            [{"name": "X"}],
        )
        assert out == {}

    def test_handles_list_content_blocks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeChat:
            def invoke(self, prompt: str) -> Any:
                class _R:
                    content = [
                        {"text": '{"name": "Synth"'},
                        {"text": ', "value": 7}'},
                    ]

                return _R()

        client = self._build_with_chat(monkeypatch, FakeChat())
        out = client(
            "{parent_records_json} {schema_columns_json}",
            [{"name": "X", "value": 0}],
        )
        assert out == {"name": "Synth", "value": 7}

    def test_returns_empty_when_template_missing_placeholder(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        called = {"n": 0}

        class FakeChat:
            def invoke(self, prompt: str) -> Any:
                called["n"] += 1

                class _R:
                    content = "{}"

                return _R()

        client = self._build_with_chat(monkeypatch, FakeChat())
        # Template references an undefined placeholder.
        out = client("Parents: {undefined}", [{"name": "X"}])
        assert out == {}
        assert called["n"] == 0  # invoke never called
