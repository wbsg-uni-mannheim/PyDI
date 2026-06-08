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
import re
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
# DOI-keyed fusion gold (papers): map gold DOIs to source-record anchors
# ---------------------------------------------------------------------------

_DOI_PREFIX_RE = re.compile(r"^(https?://)?(dx\.)?doi\.org/", re.IGNORECASE)


def _normalize_doi(value: object) -> str | None:
    """Lowercase + prefix-strip a DOI; ``None`` for empty/NaN. Mirrors
    ``build_pool_papers._normalize_doi`` so the gold DOI and the source DOI
    column normalise identically before matching."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in ("none", "nan"):
        return None
    text = _DOI_PREFIX_RE.sub("", text)
    return text.lower() or None


def _doi_to_record_ids(bundle: VariantBundle) -> dict[str, list[str]]:
    """Return ``{normalized_doi: [source_record_id, ...]}`` across all sources.

    Used for the papers fusion gold, which is keyed by DOI rather than a
    per-record ``id``: a DOI's source records are exactly its fusion cluster
    (the pool is built from DOI-exact matches)."""
    out: dict[str, list[str]] = {}
    for df in bundle.sources.values():
        if "doi" not in df.columns or "id" not in df.columns:
            continue
        for rid, doi in zip(df["id"].astype(str), df["doi"], strict=False):
            nd = _normalize_doi(doi)
            if nd is not None:
                out.setdefault(nd, []).append(rid)
    return out


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
    fg = bundle.fusion_gold
    clusters: dict[str, set[str]] = {}
    if "id" in fg.columns:
        # Standard domains: each fusion-gold entity ID is itself a source-record
        # ID and an anchor node in the pool partner graph.
        for entity_id in fg["id"].astype(str):
            eid = entity_id.strip()
            if not eid:
                continue
            cluster = _transitive_closure(eid, partners)
            if not cluster:
                cluster = {eid}
            clusters[eid] = cluster
    elif "doi" in fg.columns:
        # Papers: the fusion gold is keyed by DOI (no per-record ``id``). The
        # pool is built from DOI-exact matches, so a DOI's source records ARE
        # its cluster. Map each gold DOI to its source records and seed the
        # closure from them; key the cluster by the normalized DOI (the fusion
        # eval joins on DOI).
        doi_to_records = _doi_to_record_ids(bundle)
        for raw_doi in fg["doi"].astype(str):
            nd = _normalize_doi(raw_doi)
            if nd is None:
                continue
            anchors = doi_to_records.get(nd, [])
            if not anchors:
                continue
            cluster: set[str] = set()
            for rid in anchors:
                cluster |= _transitive_closure(rid, partners)
            clusters[nd] = cluster or set(anchors)
    else:
        raise KeyError(
            f"fusion_gold for domain {domain!r} has neither an 'id' nor a 'doi' "
            f"column to key clusters on; columns={list(fg.columns)}"
        )
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
