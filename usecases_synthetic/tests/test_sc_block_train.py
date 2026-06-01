"""Unit tests for SC-Block contrastive-training helpers.

Goes beyond shape-only smoke tests: the union-find clustering, batch
sampler, and SupCon loss math each have explicit assertions on
behaviour rather than just "doesn't crash". No transformer / torch
checkpoint is pulled at test time — the loss test imports torch
lazily and runs on a 4-record synthetic input, and the sampler tests
are pure-pandas / pure-numpy.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from usecases_synthetic.lib.sc_block_train import (
    ClusterBalancedSampler,
    TrainRecord,
    build_record_clusters,
    build_train_records,
    serialize_record,
    supcon_loss,
)

# ---------------------------------------------------------------------------
# build_record_clusters
# ---------------------------------------------------------------------------


def _make_sources(rows_per_source: dict[str, list[str]]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for name, ids in rows_per_source.items():
        out[name] = pd.DataFrame({"id": ids, "name": [f"{name}_{i}" for i in ids]})
    return out


class TestBuildRecordClusters:
    def test_singletons_when_no_positive_pairs(self) -> None:
        sources = _make_sources({"a": ["x1", "x2"], "b": ["y1"]})
        clusters = build_record_clusters({}, sources)
        ids = sorted(clusters.values())
        assert ids == [0, 1, 2]
        assert len(set(clusters.values())) == 3

    def test_positive_pair_collapses_two_records(self) -> None:
        sources = _make_sources({"a": ["x1", "x2"], "b": ["y1"]})
        gold = pd.DataFrame({"id1": ["x1"], "id2": ["y1"], "label": ["true"]})
        clusters = build_record_clusters({("a", "b"): gold}, sources)
        assert clusters[("a", "x1")] == clusters[("b", "y1")]
        assert clusters[("a", "x2")] != clusters[("a", "x1")]

    def test_transitive_closure_across_pairs(self) -> None:
        sources = _make_sources({"a": ["x1"], "b": ["y1"], "c": ["z1"]})
        gold_ab = pd.DataFrame({"id1": ["x1"], "id2": ["y1"], "label": [True]})
        gold_bc = pd.DataFrame({"id1": ["y1"], "id2": ["z1"], "label": [True]})
        clusters = build_record_clusters(
            {("a", "b"): gold_ab, ("b", "c"): gold_bc}, sources
        )
        assert clusters[("a", "x1")] == clusters[("b", "y1")] == clusters[("c", "z1")]

    def test_negative_pairs_do_not_collapse(self) -> None:
        sources = _make_sources({"a": ["x1"], "b": ["y1"]})
        gold = pd.DataFrame({"id1": ["x1"], "id2": ["y1"], "label": ["false"]})
        clusters = build_record_clusters({("a", "b"): gold}, sources)
        assert clusters[("a", "x1")] != clusters[("b", "y1")]

    def test_label_accepts_str_bool_int(self) -> None:
        sources = _make_sources({"a": ["x1", "x2", "x3"], "b": ["y1", "y2", "y3"]})
        gold = pd.DataFrame(
            {
                "id1": ["x1", "x2", "x3"],
                "id2": ["y1", "y2", "y3"],
                "label": ["true", True, 1],
            }
        )
        clusters = build_record_clusters({("a", "b"): gold}, sources)
        for left, right in [("x1", "y1"), ("x2", "y2"), ("x3", "y3")]:
            assert clusters[("a", left)] == clusters[("b", right)]

    def test_missing_id_column_raises(self) -> None:
        sources = {"a": pd.DataFrame({"foo": ["x1"]})}
        with pytest.raises(ValueError, match="missing id column"):
            build_record_clusters({}, sources)


# ---------------------------------------------------------------------------
# serialize_record + build_train_records
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_basic_tag_layout(self) -> None:
        row = pd.Series({"name": "ACME", "country": "US"})
        text = serialize_record(row, ["name", "country"])
        assert text == "[COL] name [VAL] ACME [COL] country [VAL] US"

    def test_nan_renders_as_empty_val(self) -> None:
        row = pd.Series({"name": None, "country": "US"})
        text = serialize_record(row, ["name", "country"])
        assert text.startswith("[COL] name [VAL]")
        assert "[COL] country [VAL] US" in text

    def test_build_train_records_round_trip(self) -> None:
        sources = _make_sources({"a": ["x1", "x2"], "b": ["y1"]})
        gold = pd.DataFrame({"id1": ["x1"], "id2": ["y1"], "label": [True]})
        clusters = build_record_clusters({("a", "b"): gold}, sources)
        recs = build_train_records(sources, clusters, ["name"])
        assert len(recs) == 3
        keys = sorted((r.source, r.record_id) for r in recs)
        assert keys == sorted([("a", "x1"), ("a", "x2"), ("b", "y1")])
        # x1 + y1 share a cluster; x2 is alone.
        cluster_for = {(r.source, r.record_id): r.cluster_id for r in recs}
        assert cluster_for[("a", "x1")] == cluster_for[("b", "y1")]
        assert cluster_for[("a", "x2")] != cluster_for[("a", "x1")]

    def test_missing_text_col_raises(self) -> None:
        sources = _make_sources({"a": ["x1"]})
        clusters = build_record_clusters({}, sources)
        with pytest.raises(ValueError, match="missing text_cols"):
            build_train_records(sources, clusters, ["nonexistent"])


# ---------------------------------------------------------------------------
# ClusterBalancedSampler
# ---------------------------------------------------------------------------


def _make_records_with_clusters(cluster_sizes: dict[int, int]) -> list[TrainRecord]:
    out: list[TrainRecord] = []
    rid = 0
    for cluster_id, size in cluster_sizes.items():
        for _ in range(size):
            out.append(
                TrainRecord(
                    source="a",
                    record_id=str(rid),
                    text=f"r{rid}",
                    cluster_id=cluster_id,
                )
            )
            rid += 1
    return out


class TestClusterBalancedSampler:
    def test_batch_size_property(self) -> None:
        records = _make_records_with_clusters({0: 2, 1: 2, 2: 2, 3: 2})
        sampler = ClusterBalancedSampler(
            records, clusters_per_batch=2, records_per_cluster=2
        )
        assert sampler.batch_size == 4

    def test_singletons_excluded_by_default(self) -> None:
        records = _make_records_with_clusters({0: 1, 1: 2, 2: 2})
        sampler = ClusterBalancedSampler(
            records, clusters_per_batch=2, records_per_cluster=2
        )
        # Only 2 multi-record clusters → 1 batch of 2-clusters.
        batches = list(sampler)
        assert len(batches) == 1
        records_in_batch = [records[i] for i in batches[0]]
        cluster_counts = Counter(r.cluster_id for r in records_in_batch)
        assert all(c == 2 for c in cluster_counts.values())
        # cluster 0 (singleton) must not appear.
        assert 0 not in cluster_counts

    def test_each_batch_has_expected_distinct_clusters(self) -> None:
        records = _make_records_with_clusters({0: 2, 1: 2, 2: 2, 3: 2, 4: 2, 5: 2})
        sampler = ClusterBalancedSampler(
            records, clusters_per_batch=3, records_per_cluster=2, seed=1
        )
        for batch in sampler:
            records_in_batch = [records[i] for i in batch]
            cluster_set = {r.cluster_id for r in records_in_batch}
            assert len(cluster_set) == 3
            counts = Counter(r.cluster_id for r in records_in_batch)
            assert all(c == 2 for c in counts.values())

    def test_drop_last_behaviour(self) -> None:
        # 5 multi-record clusters, batch_clusters=2 → drop_last leaves 2 batches.
        records = _make_records_with_clusters({i: 2 for i in range(5)})
        sampler = ClusterBalancedSampler(
            records, clusters_per_batch=2, records_per_cluster=2, drop_last=True
        )
        assert len(list(sampler)) == 2
        sampler_no_drop = ClusterBalancedSampler(
            records, clusters_per_batch=2, records_per_cluster=2, drop_last=False
        )
        assert len(list(sampler_no_drop)) == 3

    def test_epoch_shuffle_is_deterministic(self) -> None:
        records = _make_records_with_clusters({i: 2 for i in range(6)})
        sampler1 = ClusterBalancedSampler(
            records, clusters_per_batch=2, records_per_cluster=2, seed=7
        )
        sampler2 = ClusterBalancedSampler(
            records, clusters_per_batch=2, records_per_cluster=2, seed=7
        )
        sampler1.set_epoch(0)
        sampler2.set_epoch(0)
        b1 = list(sampler1)
        b2 = list(sampler2)
        assert b1 == b2

    def test_different_epochs_yield_different_orderings(self) -> None:
        records = _make_records_with_clusters({i: 2 for i in range(8)})
        sampler = ClusterBalancedSampler(
            records, clusters_per_batch=2, records_per_cluster=2, seed=42
        )
        sampler.set_epoch(0)
        order_a = [records[i].cluster_id for batch in sampler for i in batch]
        sampler.set_epoch(1)
        order_b = [records[i].cluster_id for batch in sampler for i in batch]
        assert order_a != order_b

    def test_include_singletons_yields_them(self) -> None:
        records = _make_records_with_clusters({0: 1, 1: 1, 2: 2})
        sampler = ClusterBalancedSampler(
            records,
            clusters_per_batch=2,
            records_per_cluster=2,
            include_singletons=True,
            seed=1,
        )
        # 3 eligible clusters with batch_clusters=2, drop_last=True → 1 batch.
        batches = list(sampler)
        assert len(batches) == 1
        records_in_batch = [records[i] for i in batches[0]]
        assert (
            len(records_in_batch) == 4
        )  # 2 clusters x 2 records (singletons over-sampled)


# ---------------------------------------------------------------------------
# supcon_loss
# ---------------------------------------------------------------------------


class TestSupConLoss:
    @pytest.fixture(autouse=True)
    def _torch(self) -> None:
        pytest.importorskip("torch")

    def test_zero_when_all_singletons(self) -> None:
        import torch

        z = torch.nn.functional.normalize(torch.randn(4, 16), dim=-1)
        labels = torch.tensor([0, 1, 2, 3])
        loss = supcon_loss(z, labels, temperature=0.07)
        # No anchor has a positive → loss == 0.
        assert float(loss) == pytest.approx(0.0)

    def test_lower_when_positives_align(self) -> None:
        """A batch where positives are colinear should beat a random batch."""
        import torch

        torch.manual_seed(0)
        # 2 clusters, 2 records each. Make positives colinear.
        a = torch.tensor([1.0, 0.0, 0.0, 0.0])
        b = torch.tensor([0.0, 1.0, 0.0, 0.0])
        z_aligned = torch.stack([a, a, b, b], dim=0)
        z_random = torch.nn.functional.normalize(torch.randn(4, 4), dim=-1)
        labels = torch.tensor([0, 0, 1, 1])
        loss_aligned = float(supcon_loss(z_aligned, labels, temperature=0.07))
        loss_random = float(supcon_loss(z_random, labels, temperature=0.07))
        assert loss_aligned < loss_random

    def test_smaller_temperature_increases_loss_on_misaligned(self) -> None:
        """Tighter temperature amplifies the penalty for diffuse positives."""
        import torch

        torch.manual_seed(1)
        z = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        loss_loose = float(supcon_loss(z, labels, temperature=1.0))
        loss_tight = float(supcon_loss(z, labels, temperature=0.07))
        assert loss_tight > loss_loose

    def test_returns_scalar(self) -> None:
        import torch

        z = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
        labels = torch.tensor([0, 0, 1, 1])
        loss = supcon_loss(z, labels)
        assert loss.ndim == 0

    def test_handles_batch_size_one(self) -> None:
        import torch

        z = torch.tensor([[1.0, 0.0]])
        labels = torch.tensor([0])
        loss = supcon_loss(z, labels)
        assert float(loss) == pytest.approx(0.0)
