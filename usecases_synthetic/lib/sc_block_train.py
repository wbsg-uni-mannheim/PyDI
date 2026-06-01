"""Data-prep + loss helpers for SC-Block contrastive training.

Pure-logic surface — no HuggingFace / torch imports at module level —
so the cluster construction, batch sampling, and SupCon math can be
unit-tested without pulling a transformer at test time. The torch-
dependent loss is imported inside the function body.

The SC-Block recipe (Brinkmann et al., ESWC 2024) trains a transformer
encoder with a supervised contrastive loss: each batch contains
records tagged with a cluster id; for each anchor, positives are the
other in-batch records sharing the cluster id, negatives are the rest.
The construction here mirrors that contract — given EM gold pairs
``(id1, id2, label)`` per source pair and the per-source DataFrames,
:func:`build_record_clusters` runs union-find on the ``label=True``
edges (cross-source-pair, so a record linked through two different
gold splits ends up in one cluster) and emits per-record cluster ids.

:class:`ClusterBalancedSampler` then yields batch indices such that
each batch holds ``records_per_cluster`` records from each of
``clusters_per_batch`` distinct clusters — guaranteeing at least one
in-batch positive per anchor when ``records_per_cluster >= 2``.

:func:`serialize_record` matches the
``[COL] field [VAL] value`` shape used by
:class:`usecases_synthetic.lib.sc_block_blocker.SCBlockBlocker`, so
train-time and inference-time serialization stay aligned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Per-domain field set (matches the per-domain Ditto fields so the SC-Block
# trainer's serialization shape and the SCBlockBlocker's inference-time
# `text_cols` stay aligned). Edit-here-only: the trainer CLI and the
# `_tune_em_blocking_committee.py` sweep harness both import this dict.
# ---------------------------------------------------------------------------


# R10-I (2026-05-29) widened every domain's text_cols to the full
# per-domain schema (the wide-scope EM committee expansion). Each list
# here MUST stay byte-identical to the ``sc_block.text_cols`` in
# ``em_blocking_committee[_<domain>].yaml`` (train ↔ inference [COL]/[VAL]
# serialisation) and equal the ``ditto_plm.fields`` in the matching
# committee per the "ditto fields = sc_block text_cols" invariant — see
# tests/test_em_field_scope_consistency.py. Sources with a heterogeneous
# schema (e.g. companies forbes lacks city/founded) are handled by
# filling the missing text_cols with empty in both the trainer
# (_load_domain_data / _build_variant_data) and SCBlockBlocker (R10-I).
DOMAIN_TEXT_COLS: dict[str, list[str]] = {
    "companies": [
        "name",
        "country",
        "city",
        "industry",
        "sector",
        "founded",
        "keypeople",
        "assets",
        "revenue",
    ],
    "games": [
        "name",
        "releaseYear",
        "developer",
        "platform",
        "genres",
        "series",
        "criticScore",
        "userScore",
        "ESRB",
        "publisher",
        "globalSales",
    ],
    "music": [
        "name",
        "artist",
        "release-date",
        "release-country",
        "duration",
        "label",
        "genre",
        "tracks",
    ],
    "products": [
        "title",
        "brand",
        "description",
        "price",
        "priceCurrency",
        "title_description",
        "product_type",
        "model",
        "model_number",
        "chipset_name",
        "vram_gb",
        "storage_gb",
        "bus_type",
        "interface_type",
        "memory_type",
        "storage_connection_type",
        "form_factor",
        "read_speed_mb_s",
        "write_speed_mb_s",
    ],
}


# ---------------------------------------------------------------------------
# Record cluster construction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainRecord:
    """A single training record.

    Attributes
    ----------
    source : str
        Source name (e.g. ``"dbpedia"``).
    record_id : str
        Within-source primary key (the ``id`` column).
    text : str
        Serialized ``[COL] field [VAL] value`` record string.
    cluster_id : int
        Connected-component id from the union-find graph; records in
        the same component describe the same real-world entity. Records
        absent from every ``label=True`` edge get singleton cluster ids
        (one record per cluster).
    """

    source: str
    record_id: str
    text: str
    cluster_id: int


class _UnionFind:
    """Iterative union-find with path compression + union by rank."""

    def __init__(self) -> None:
        self._parent: dict[Any, Any] = {}
        self._rank: dict[Any, int] = {}

    def find(self, x: Any) -> Any:
        self._parent.setdefault(x, x)
        self._rank.setdefault(x, 0)
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        cur = x
        while self._parent[cur] != root:
            nxt = self._parent[cur]
            self._parent[cur] = root
            cur = nxt
        return root

    def union(self, a: Any, b: Any) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1


def _normalize_label(value: Any) -> bool:
    """Coerce any of ``"true"``/``"True"``/``True``/``1`` to bool ``True``."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def build_record_clusters(
    em_pairs_by_pair: dict[tuple[str, str], pd.DataFrame],
    sources: dict[str, pd.DataFrame],
    *,
    id_column: str = "id",
) -> dict[tuple[str, str], int]:
    """Run union-find over ``label=True`` edges and emit cluster ids.

    Parameters
    ----------
    em_pairs_by_pair : dict
        Maps ``(src1, src2)`` to a DataFrame with at least ``id1``,
        ``id2``, ``label`` columns. ``label`` may be a bool, an int
        ``{0, 1}``, or a string ``{"true", "false", "True", "False"}``.
    sources : dict
        Maps source name to its source DataFrame; used to seed the
        union-find with every record (so records never seen in a
        ``label=True`` edge still end up with their own cluster id).
    id_column : str, default="id"
        Column in each source DataFrame carrying the primary key.

    Returns
    -------
    dict
        ``{(source, record_id): cluster_id}`` with integer cluster ids
        starting at 0. Records reachable from each other via positive
        edges share the same id; records never seen positively get
        singleton ids.
    """
    uf = _UnionFind()
    keyed_records: set[tuple[str, str]] = set()
    for source_name, df in sources.items():
        if id_column not in df.columns:
            raise ValueError(
                f"source {source_name!r} missing id column {id_column!r}; "
                f"has {list(df.columns)}"
            )
        for rid in df[id_column].astype(str):
            keyed_records.add((source_name, rid))
    for key in keyed_records:
        uf.find(key)

    for (src1, src2), gold in em_pairs_by_pair.items():
        if gold is None or gold.empty:
            continue
        for _, row in gold.iterrows():
            if not _normalize_label(row["label"]):
                continue
            a = (src1, str(row["id1"]))
            b = (src2, str(row["id2"]))
            # Seed both keys so foreign-source ids that don't appear in
            # sources still cluster together (we keep the cluster ids
            # but the record will be filtered later when we materialise).
            uf.find(a)
            uf.find(b)
            uf.union(a, b)

    root_to_id: dict[Any, int] = {}
    out: dict[tuple[str, str], int] = {}
    for key in keyed_records:
        root = uf.find(key)
        if root not in root_to_id:
            root_to_id[root] = len(root_to_id)
        out[key] = root_to_id[root]
    return out


def serialize_record(
    row: pd.Series,
    text_cols: Sequence[str],
) -> str:
    """Serialize a record into the ``[COL] field [VAL] value`` shape.

    Mirrors :meth:`SCBlockBlocker._serialize` so train-time and
    inference-time serialization stay aligned. NaN / ``None`` values
    render as an empty ``[VAL]`` chunk so the encoder still sees every
    declared column tag.

    Parameters
    ----------
    row : Series
        Row to serialize.
    text_cols : sequence of str
        Columns to emit, in order.

    Returns
    -------
    str
        Concatenation of ``[COL] field [VAL] value`` chunks, space-joined.
    """
    parts: list[str] = []
    for col in text_cols:
        value = row.get(col) if hasattr(row, "get") else row[col]
        if value is None or (isinstance(value, float) and np.isnan(value)):
            value_str = ""
        else:
            value_str = str(value)
        parts.append(f"[COL] {col} [VAL] {value_str}".strip())
    return " ".join(parts)


def build_train_records(
    sources_mapped: dict[str, pd.DataFrame],
    record_to_cluster: dict[tuple[str, str], int],
    text_cols: Sequence[str],
    *,
    id_column: str = "id",
) -> list[TrainRecord]:
    """Materialize the flat list of training records (one per record id).

    Parameters
    ----------
    sources_mapped : dict
        Source DataFrames with canonical column names already applied.
    record_to_cluster : dict
        Output of :func:`build_record_clusters`.
    text_cols : sequence of str
        Field set to serialize per record (e.g. ``["name", "country"]``).
    id_column : str, default="id"
        Primary key column.

    Returns
    -------
    list of TrainRecord
        One :class:`TrainRecord` per row in each source. Records absent
        from ``record_to_cluster`` are skipped (defensive; should not
        happen when :func:`build_record_clusters` is given the same
        sources).
    """
    records: list[TrainRecord] = []
    for source_name, df in sources_mapped.items():
        missing = [c for c in text_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"source {source_name!r} missing text_cols {missing}; "
                f"has {list(df.columns)}"
            )
        for _, row in df.iterrows():
            rid = str(row[id_column])
            key = (source_name, rid)
            if key not in record_to_cluster:
                continue
            text = serialize_record(row, text_cols)
            records.append(
                TrainRecord(
                    source=source_name,
                    record_id=rid,
                    text=text,
                    cluster_id=record_to_cluster[key],
                )
            )
    return records


# ---------------------------------------------------------------------------
# Cluster-balanced batch sampler
# ---------------------------------------------------------------------------


class ClusterBalancedSampler:
    """Yield batches that hold ``records_per_cluster`` records per cluster.

    Each batch contains exactly ``clusters_per_batch`` distinct clusters
    with ``records_per_cluster`` records each (``batch_size = C * P``).
    Singleton clusters (one record) are filtered out by default — they
    contribute negatives but cannot serve as anchors with in-batch
    positives. When ``include_singletons`` is True they are mixed in to
    increase batch diversity; their records get sampled once.

    Parameters
    ----------
    records : list of TrainRecord
        Pre-materialised training records.
    clusters_per_batch : int
        Number of distinct clusters per batch (must be >= 1).
    records_per_cluster : int, default=2
        Records drawn from each cluster (must be >= 1; >= 2 guarantees
        each anchor has at least one in-batch positive).
    shuffle : bool, default=True
        Shuffle cluster order each epoch.
    drop_last : bool, default=True
        Drop the final partial batch when ``num_clusters % C != 0``.
    seed : int, default=42
        RNG seed.
    include_singletons : bool, default=False
        When True, also yield singleton clusters as anchor batches.

    Attributes
    ----------
    batch_size : int
        ``clusters_per_batch * records_per_cluster``.
    """

    def __init__(
        self,
        records: list[TrainRecord],
        *,
        clusters_per_batch: int,
        records_per_cluster: int = 2,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 42,
        include_singletons: bool = False,
    ) -> None:
        if clusters_per_batch < 1:
            raise ValueError(
                f"clusters_per_batch must be >= 1, got {clusters_per_batch}"
            )
        if records_per_cluster < 1:
            raise ValueError(
                f"records_per_cluster must be >= 1, got {records_per_cluster}"
            )

        self.clusters_per_batch = int(clusters_per_batch)
        self.records_per_cluster = int(records_per_cluster)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.include_singletons = bool(include_singletons)
        self._epoch = 0

        index_by_cluster: dict[int, list[int]] = {}
        for i, rec in enumerate(records):
            index_by_cluster.setdefault(rec.cluster_id, []).append(i)
        if include_singletons:
            self._eligible: list[int] = sorted(index_by_cluster.keys())
        else:
            self._eligible = sorted(
                [cid for cid, idxs in index_by_cluster.items() if len(idxs) >= 2]
            )
        self._index_by_cluster = index_by_cluster

    @property
    def batch_size(self) -> int:
        return self.clusters_per_batch * self.records_per_cluster

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch number so ``__iter__`` shuffles deterministically."""
        self._epoch = int(epoch)

    def __len__(self) -> int:
        if not self._eligible:
            return 0
        n_batches = len(self._eligible) // self.clusters_per_batch
        if not self.drop_last and len(self._eligible) % self.clusters_per_batch:
            n_batches += 1
        return n_batches

    def __iter__(self) -> Iterator[list[int]]:
        if not self._eligible:
            return
        rng = np.random.default_rng(self.seed + self._epoch)
        cluster_order = list(self._eligible)
        if self.shuffle:
            rng.shuffle(cluster_order)
        for start in range(0, len(cluster_order), self.clusters_per_batch):
            chunk = cluster_order[start : start + self.clusters_per_batch]
            if self.drop_last and len(chunk) < self.clusters_per_batch:
                break
            batch: list[int] = []
            for cluster_id in chunk:
                pool = self._index_by_cluster[cluster_id]
                if len(pool) >= self.records_per_cluster:
                    chosen = rng.choice(
                        pool, size=self.records_per_cluster, replace=False
                    ).tolist()
                else:
                    chosen = rng.choice(
                        pool, size=self.records_per_cluster, replace=True
                    ).tolist()
                batch.extend(int(i) for i in chosen)
            yield batch


# ---------------------------------------------------------------------------
# Supervised contrastive loss
# ---------------------------------------------------------------------------


def supcon_loss(
    embeddings: Any,
    cluster_ids: Any,
    *,
    temperature: float = 0.07,
) -> Any:
    """Supervised contrastive loss (Khosla et al. 2020, SC-Block §3).

    Computes::

        L_i = -1/|P(i)| * sum_{p in P(i)}
                log( exp(z_i . z_p / tau) /
                     sum_{a != i} exp(z_i . z_a / tau) )

    where ``P(i)`` is the set of positives for anchor ``i`` (records
    sharing the cluster id) and the denominator runs over all
    non-self records. Anchors with no in-batch positives contribute 0
    to the loss (consistent with the SC-Block reference implementation;
    they still contribute negatives for other anchors via the
    denominator).

    Parameters
    ----------
    embeddings : Tensor of shape (B, D)
        L2-normalized record embeddings.
    cluster_ids : Tensor of shape (B,)
        Per-record cluster ids (int).
    temperature : float, default=0.07
        Loss temperature. SC-Block paper default.

    Returns
    -------
    Tensor
        Scalar loss averaged over anchors with at least one positive.
        Returns ``0.0`` when no anchor has a positive.
    """
    import torch
    import torch.nn.functional as F

    z = embeddings
    labels = cluster_ids
    batch_size = z.shape[0]
    if batch_size < 2:
        return z.new_zeros(())

    sims = torch.matmul(z, z.transpose(0, 1)) / float(temperature)
    eye = torch.eye(batch_size, dtype=torch.bool, device=z.device)
    sims = sims.masked_fill(eye, float("-inf"))

    labels_2d = labels.view(-1, 1)
    positives_mask = (labels_2d == labels_2d.t()) & ~eye

    log_prob = F.log_softmax(sims, dim=1)
    # log_softmax sets the diagonal to -inf (we masked the inputs).
    # Multiplying by the (zero) diagonal of positives_mask would yield
    # NaN under IEEE 754 (-inf * 0). Zero the diagonal explicitly so the
    # mean-over-positives sum stays finite.
    log_prob = log_prob.masked_fill(eye, 0.0)
    pos_counts = positives_mask.sum(dim=1)
    has_pos = pos_counts > 0
    if not bool(has_pos.any()):
        return z.new_zeros(())

    mean_log_prob_pos = (log_prob * positives_mask.float()).sum(dim=1)[
        has_pos
    ] / pos_counts[has_pos].clamp(min=1).float()
    return -mean_log_prob_pos.mean()


__all__ = [
    "DOMAIN_TEXT_COLS",
    "TrainRecord",
    "build_record_clusters",
    "serialize_record",
    "build_train_records",
    "ClusterBalancedSampler",
    "supcon_loss",
]
