"""Build perfect-cluster correspondences for fusion-committee evaluation.

R5 Fusion sign-off (2026-05-12): each committee evaluates against the
**perfect** output of the prior pipeline step, isolating the signal of
"how good is *this* committee" from "how good is the pipeline". For the
fusion committee, "perfect prior step" means: assume entity matching
discovered the cross-source positive set the R3 pool already declares
for the fusion validation + test entities.

Cluster ground truth comes from
``usecases_synthetic/pools/<domain>/pooled_positives.csv`` (built in
R3 as the evidence union of EM gold + human-baseline rule-matcher +
Ditto, with cross-source transitive closure). Per the R5 Fusion sign-
off conversation: the pool already names every cross-source positive
the variant pipeline can reach; record IDs survive every K-knob
mutation so the pool stays the authoritative cluster source across
baseline + augmented variants.

For each fusion-gold entity ID:

1. Look up the entity in the pool's partner graph (symmetric over
   ``id1`` / ``id2`` columns).
2. Build the transitive closure across all source-pair partners so a
   metacritic↔dbpedia + dbpedia↔sales partnership becomes a single
   3-source cluster.
3. Emit hub-and-spoke correspondences (entity_id → each partner)
   that the fusion engine's connected-components grouping turns into
   a single record group.

Entities that appear in the fusion gold but not in the pool are
emitted as singleton self-edges so ``include_singletons=True`` in the
fusion engine keeps them in the fused output for evaluation (the eval
then degrades gracefully to "compare the lone source's value to the
gold").
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .variant_loader import VariantBundle

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
POOL_DIR = REPO_ROOT / "usecases_synthetic" / "pools"


# ---------------------------------------------------------------------------
# Pool loading + partner graph
# ---------------------------------------------------------------------------


def _pool_path(domain: str) -> Path:
    return POOL_DIR / domain / "pooled_positives.csv"


def _partner_graph(domain: str) -> dict[str, set[str]]:
    """Return a symmetric ``{id: {partner, ...}}`` map from the R3 pool."""
    path = _pool_path(domain)
    if not path.exists():
        raise FileNotFoundError(
            f"Pool for domain {domain!r} not found at {path}. "
            "Run scripts/build_pool.py to (re)generate it."
        )
    pool = pd.read_csv(path)
    partners: dict[str, set[str]] = {}
    for a, b in zip(pool["id1"].astype(str), pool["id2"].astype(str), strict=True):
        partners.setdefault(a, set()).add(b)
        partners.setdefault(b, set()).add(a)
    return partners


def _transitive_closure(seed: str, partners: dict[str, set[str]]) -> set[str]:
    """BFS the partner graph starting at ``seed`` to return its full cluster."""
    visited: set[str] = set()
    queue: list[str] = [seed]
    while queue:
        node = queue.pop()
        if node in visited:
            continue
        visited.add(node)
        for nbr in partners.get(node, set()):
            if nbr not in visited:
                queue.append(nbr)
    return visited


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def build_perfect_clusters(domain: str, bundle: VariantBundle) -> dict[str, set[str]]:
    """Return the perfect-cluster map for a domain.

    Maps each fusion-gold entity ID to the set of source-record IDs that
    belong to its cluster (always includes the entity ID itself).
    Clusters are derived from the transitive closure of the R3 pool's
    partner graph.
    """
    if bundle.fusion_gold is None or bundle.fusion_gold.empty:
        return {}
    partners = _partner_graph(domain)
    clusters: dict[str, set[str]] = {}
    for entity_id in bundle.fusion_gold["id"].astype(str):
        eid = entity_id.strip()
        if not eid:
            continue
        cluster = _transitive_closure(eid, partners)
        if not cluster:
            cluster = {eid}
        clusters[eid] = cluster
    return clusters


def build_perfect_clusters_correspondences(
    domain: str, bundle: VariantBundle
) -> pd.DataFrame:
    """Return correspondences DataFrame that yields the perfect clusters.

    Strategy: for each cluster of N members, emit ``N-1`` hub-and-spoke
    edges between member[0] and every other member. Connected components
    of the resulting graph = the perfect clusters. Singleton clusters
    (N=1) emit one self-edge so ``include_singletons=True`` keeps them
    in the fused output.

    Parameters
    ----------
    domain
        Domain name (``companies`` / ``games`` / ``music``).
    bundle
        Variant bundle (baseline or augmented).

    Returns
    -------
    DataFrame
        Columns ``id1``, ``id2``, ``score`` (= 1.0 everywhere).
    """
    clusters = build_perfect_clusters(domain, bundle)
    rows: list[tuple[str, str, float]] = []
    n_singletons = 0
    cluster_sizes: list[int] = []
    for entity_id, members in clusters.items():
        sorted_members = sorted(members)
        cluster_sizes.append(len(sorted_members))
        if len(sorted_members) == 1:
            sole = sorted_members[0]
            rows.append((sole, sole, 1.0))
            n_singletons += 1
            continue
        hub = sorted_members[0]
        for other in sorted_members[1:]:
            rows.append((hub, other, 1.0))
    if not rows:
        return pd.DataFrame(columns=["id1", "id2", "score"])
    df = pd.DataFrame(rows, columns=["id1", "id2", "score"])
    df = df.drop_duplicates(subset=["id1", "id2"], ignore_index=True)
    if cluster_sizes:
        avg_size = sum(cluster_sizes) / len(cluster_sizes)
    else:
        avg_size = 0.0
    logger.info(
        "Perfect-cluster correspondences for %s: %d clusters, %d edges, "
        "%d singletons, avg cluster size %.2f",
        domain,
        len(clusters),
        len(df),
        n_singletons,
        avg_size,
    )
    return df
