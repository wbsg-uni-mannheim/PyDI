"""Similarity metrics for Knob 02 — Entity Niche Density.

Implements the five deterministic metrics used by both the niche-density
scorer (consensus-biased) and the corner-case pair miner (recall-biased).
See ``knobs/knob_02_niche_density.md`` §"Metric set" for the full spec.

Metrics
-------
``lexical_extended_jaccard``
    Generalised Jaccard on whitespace-tokenised primary labels with a
    typo-robust inner-token comparator (Levenshtein ratio
    ``>= inner_token_threshold`` counts as a match). Replaces plain
    token Jaccard so Knob 06-injected typos do not silently erase near
    twins.
``compute_tfidf_matrix``
    ``sklearn.feature_extraction.text.TfidfVectorizer`` fit on the full
    corpus; returns the sparse document-term matrix and the vocabulary
    size for downstream cosine similarities.
``compute_embedding_matrix``
    ``sentence-transformers/all-MiniLM-L6-v2`` embeddings with an on-disk
    cache at ``usecases_synthetic/cache/knob_02_embeddings/<domain>.npy``.
    The sidecar ``<domain>.meta.json`` stores model id, concat-order and
    content hash for cache invalidation.
``attribute_overlap``
    Weighted Jaccard over the categorical-attribute bag with per-domain
    column weights. Pure pandas.
``label_collision_index``
    Exact match on ``normalize(name)`` (lowercase, strip punctuation,
    collapse whitespace, drop bracketed suffixes). Groups entities whose
    normalised primary label is byte-identical; contributes a density
    boost in the scorer and a hard deterministic rule in the miner.

All functions are deterministic pure functions of their inputs. RNG is
not used by this module.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    import Levenshtein  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    Levenshtein = None  # type: ignore[assignment]

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
except ImportError:  # pragma: no cover
    TfidfVectorizer = None  # type: ignore[assignment]
    NearestNeighbors = None  # type: ignore[assignment]

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------


_BRACKETED = re.compile(r"\s*[\(\[\{].*?[\)\]\}]\s*")
_NON_ALNUM = re.compile(r"[^\w\s]+", re.UNICODE)
_WHITESPACE = re.compile(r"\s+")


def normalize_label(value: str) -> str:
    """Normalise a primary-label string for label-collision detection.

    Steps: lowercase → NFKD-fold and strip combining marks (so accented
    characters compare equal to their ASCII variants) → drop bracketed
    suffixes → strip punctuation → collapse whitespace. The folded form
    is used only inside the K2 similarity / collision pipeline; raw
    accented values remain in the source frames written to disk.
    Deterministic, stdlib only.

    Parameters
    ----------
    value : str
        Raw primary-label value.

    Returns
    -------
    str
        Normalised label. Empty string on empty or null input.
    """
    if value is None:
        return ""
    text = str(value).lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = _BRACKETED.sub(" ", text)
    text = _NON_ALNUM.sub(" ", text)
    text = _WHITESPACE.sub(" ", text).strip()
    return text


def tokenize(value: str, stopwords: set[str] | None = None) -> list[str]:
    """Lowercase whitespace tokenisation with optional stopword removal."""
    stopwords = stopwords or set()
    tokens = [t for t in normalize_label(value).split() if t]
    return [t for t in tokens if t not in stopwords]


# ---------------------------------------------------------------------------
# Metric 1 — Lexical extended Jaccard
# ---------------------------------------------------------------------------


def _levenshtein_ratio(a: str, b: str) -> float:
    """Return the Levenshtein ratio between *a* and *b* in ``[0, 1]``.

    Uses ``python-Levenshtein`` when available; falls back to a pure
    Python implementation otherwise.
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if Levenshtein is not None:
        return float(Levenshtein.ratio(a, b))
    # Pure Python fallback — O(len(a) * len(b)).
    la, lb = len(a), len(b)
    prev = list(range(lb + 1))
    curr = [0] * (lb + 1)
    for i in range(1, la + 1):
        curr[0] = i
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    dist = prev[lb]
    return 1.0 - dist / max(la, lb)


def lexical_extended_jaccard(
    a: str,
    b: str,
    *,
    inner_token_threshold: float = 0.8,
    stopwords: set[str] | None = None,
) -> float:
    """Extended Jaccard on tokenised labels with a typo-robust inner comparator.

    Two tokens are considered equal when their Levenshtein ratio is at
    or above *inner_token_threshold*. Replaces plain token Jaccard so
    Knob 06-injected typos still resolve near twins. Implementation
    follows the WDC Products inner-token-match pattern (default 0.8).

    Parameters
    ----------
    a, b : str
        Raw primary-label strings.
    inner_token_threshold : float, default 0.8
        Per-token Levenshtein ratio threshold. ``1.0`` recovers exact
        token Jaccard.
    stopwords : set of str or None
        Optional lowercase stopword set applied before tokenisation.

    Returns
    -------
    float
        Generalised Jaccard similarity in ``[0, 1]``.
    """
    ta = tokenize(a, stopwords=stopwords)
    tb = tokenize(b, stopwords=stopwords)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0

    # Greedy matching: each token in the smaller set is matched to the
    # best unmatched token in the larger set.
    small, large = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    used: set[int] = set()
    matches = 0
    for s in small:
        best_ratio = 0.0
        best_idx = -1
        for j, l in enumerate(large):
            if j in used:
                continue
            r = _levenshtein_ratio(s, l)
            if r > best_ratio:
                best_ratio = r
                best_idx = j
        if best_idx >= 0 and best_ratio >= inner_token_threshold:
            used.add(best_idx)
            matches += 1

    union = len(ta) + len(tb) - matches
    if union <= 0:
        return 0.0
    return matches / union


def lexical_extended_jaccard_neighbours(
    labels: Sequence[str],
    *,
    top_k: int,
    inner_token_threshold: float = 0.8,
    stopwords: set[str] | None = None,
    max_block_size: int = 2000,
) -> list[list[tuple[int, float]]]:
    """Compute top-K extended-Jaccard neighbours via token-prefix blocking.

    A naive all-pairs scan is O(n^2) and intractable above ~20K labels
    (games / music / movies are larger). We block by sharing a
    non-stopword token *prefix* (lowercase first 3 chars per token; the
    whole token if shorter than 3 chars). Candidates per label are the
    union of all labels in any of its token-prefix buckets. The exact
    extended-Jaccard scoring is then applied only over the candidate
    set, preserving algorithm semantics within each block.

    Buckets larger than *max_block_size* are dropped: a token prefix
    shared by thousands of labels is non-selective and dominates runtime
    without changing the final top-K (most pairs in such a bucket would
    score below the lower-similarity entries already kept from
    smaller, more discriminative buckets).

    Recall caveat: pairs whose tokens differ in their first ≥3
    characters are missed even if their inner-Levenshtein ratio would
    have crossed *inner_token_threshold*. With τ=0.8 and typical token
    lengths ≥4, matching tokens almost always share their 3-char prefix
    (a single typo at position ≥3 is the dominant case). For shorter
    tokens (≤3 chars) the prefix is the whole token, so they only match
    exact duplicates — acceptable because the inner-Levenshtein matcher
    cannot distinguish small tokens reliably anyway.

    Parameters
    ----------
    labels : sequence of str
        Primary labels indexed by entity row number.
    top_k : int
        Number of neighbours to retain per entity (excludes self).
    inner_token_threshold, stopwords
        Forwarded to :func:`lexical_extended_jaccard`.
    max_block_size : int, default 2000
        Drop token-prefix buckets exceeding this size. Set to ``0`` to
        disable bucket-size filtering (use all blocks).

    Returns
    -------
    list of list of (int, float)
        For each entity, a sorted list of ``(neighbour_index, sim)``
        tuples (descending by similarity, length ``<= top_k``).
    """
    n = len(labels)

    # Tokenise once, reused for blocking + scoring.
    label_tokens: list[list[str]] = [
        tokenize(lbl, stopwords=stopwords) for lbl in labels
    ]

    # Build prefix-3 inverted index per non-stopword token. Each label is
    # added at most once per distinct prefix it produces so common tokens
    # in the same label do not duplicate postings.
    prefix_buckets: dict[str, list[int]] = defaultdict(list)
    for i, toks in enumerate(label_tokens):
        seen: set[str] = set()
        for t in toks:
            p = t[:3] if len(t) >= 3 else t
            if p in seen:
                continue
            prefix_buckets[p].append(i)
            seen.add(p)

    if max_block_size > 0:
        oversized = [
            p for p, idxs in prefix_buckets.items() if len(idxs) > max_block_size
        ]
        for p in oversized:
            del prefix_buckets[p]
        if oversized:
            logger.info(
                "lexical_extended_jaccard_neighbours: dropped %d oversized prefix "
                "buckets (size > %d)",
                len(oversized),
                max_block_size,
            )

    out: list[list[tuple[int, float]]] = []
    for i in range(n):
        candidates: set[int] = set()
        for t in label_tokens[i]:
            p = t[:3] if len(t) >= 3 else t
            bucket = prefix_buckets.get(p)
            if not bucket:
                continue
            candidates.update(bucket)
        candidates.discard(i)

        row: list[tuple[int, float]] = []
        if candidates:
            for j in candidates:
                sim = lexical_extended_jaccard(
                    labels[i],
                    labels[j],
                    inner_token_threshold=inner_token_threshold,
                    stopwords=stopwords,
                )
                if sim > 0.0:
                    row.append((j, sim))
            row.sort(key=lambda t: (-t[1], t[0]))
        out.append(row[:top_k])
    return out


# ---------------------------------------------------------------------------
# Metric 2 — TF-IDF cosine
# ---------------------------------------------------------------------------


def compute_tfidf_matrix(corpus: Sequence[str]) -> Any:
    """Fit a ``TfidfVectorizer`` on *corpus* and return the sparse matrix.

    Parameters
    ----------
    corpus : sequence of str
        Concatenated text block per entity (primary + secondary text).

    Returns
    -------
    scipy.sparse.csr_matrix
        ``(n_docs, vocab_size)`` sparse TF-IDF matrix.
    """
    if TfidfVectorizer is None:  # pragma: no cover
        raise ImportError("scikit-learn is required for compute_tfidf_matrix")
    vec = TfidfVectorizer(lowercase=True, analyzer="word")
    return vec.fit_transform(list(corpus))


def tfidf_neighbours(
    tfidf_matrix: Any,
    *,
    top_k: int,
) -> list[list[tuple[int, float]]]:
    """Compute top-K cosine neighbours from a TF-IDF matrix.

    Uses ``NearestNeighbors`` with cosine metric for exact top-K
    retrieval. Returns ``(index, similarity)`` tuples.
    """
    if NearestNeighbors is None:  # pragma: no cover
        raise ImportError("scikit-learn is required for tfidf_neighbours")
    n = tfidf_matrix.shape[0]
    if n == 0:
        return []
    # Request top_k + 1 so we can drop self.
    k = min(top_k + 1, n)
    nn = NearestNeighbors(n_neighbors=k, metric="cosine")
    nn.fit(tfidf_matrix)
    distances, indices = nn.kneighbors(tfidf_matrix)
    out: list[list[tuple[int, float]]] = []
    for i in range(n):
        row: list[tuple[int, float]] = []
        for dist, idx in zip(distances[i], indices[i]):
            if idx == i:
                continue
            sim = 1.0 - float(dist)
            row.append((int(idx), sim))
        row.sort(key=lambda t: (-t[1], t[0]))
        out.append(row[:top_k])
    return out


# ---------------------------------------------------------------------------
# Metric 3 — Embedding cosine (with on-disk cache)
# ---------------------------------------------------------------------------


@dataclass
class EmbeddingCacheMeta:
    """Sidecar metadata for the embedding cache."""

    model_id: str
    input_column_concat_order: list[str]
    content_hash: str

    def to_json(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "input_column_concat_order": self.input_column_concat_order,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "EmbeddingCacheMeta":
        return cls(
            model_id=str(data["model_id"]),
            input_column_concat_order=list(data["input_column_concat_order"]),
            content_hash=str(data["content_hash"]),
        )


def _content_hash(corpus: Sequence[str]) -> str:
    """Stable SHA-256 over the corpus used for cache invalidation."""
    h = hashlib.sha256()
    for doc in corpus:
        h.update(doc.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def compute_embedding_matrix(
    corpus: Sequence[str],
    *,
    model_id: str,
    cache_path: Path,
    concat_order: list[str] | None = None,
) -> np.ndarray:
    """Compute (or load from cache) the embedding matrix for *corpus*.

    Parameters
    ----------
    corpus : sequence of str
        Concatenated per-entity text blocks in the canonical row order.
    model_id : str
        ``sentence-transformers`` model identifier
        (``sentence-transformers/all-MiniLM-L6-v2`` by default).
    cache_path : Path
        Output ``.npy`` file. Sidecar ``<stem>.meta.json`` stores
        ``{model_id, input_column_concat_order, content_hash}`` for
        invalidation.
    concat_order : list of str or None
        Columns used to build *corpus*. Recorded in the sidecar meta;
        empty list when unknown.

    Returns
    -------
    numpy.ndarray
        ``(n_docs, embed_dim)`` float32 embedding matrix.
    """
    cache_path = Path(cache_path)
    meta_path = cache_path.with_suffix(".meta.json")
    c_hash = _content_hash(corpus)
    expected_meta = EmbeddingCacheMeta(
        model_id=model_id,
        input_column_concat_order=list(concat_order or []),
        content_hash=c_hash,
    )

    if cache_path.exists() and meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            got = EmbeddingCacheMeta.from_json(json.load(f))
        if (
            got.model_id == expected_meta.model_id
            and got.content_hash == expected_meta.content_hash
        ):
            matrix = np.load(cache_path)
            if matrix.shape[0] == len(corpus):
                logger.info(
                    "Embedding cache hit: %s (shape=%s)",
                    cache_path,
                    matrix.shape,
                )
                return matrix
        logger.info("Embedding cache invalidated (meta or content mismatch)")

    if SentenceTransformer is None:  # pragma: no cover
        raise ImportError(
            "sentence-transformers is required for compute_embedding_matrix. "
            "Install via the [embedding] extra."
        )

    logger.info("Encoding %d docs with %s", len(corpus), model_id)
    model = SentenceTransformer(model_id)
    matrix = model.encode(
        list(corpus),
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, matrix)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(expected_meta.to_json(), f, indent=2, sort_keys=True)
    return matrix


def embedding_neighbours(
    embeddings: np.ndarray,
    *,
    top_k: int,
) -> list[list[tuple[int, float]]]:
    """Compute top-K cosine neighbours from an embedding matrix.

    Embeddings should be L2-normalised. Uses exact top-K via
    ``NearestNeighbors`` with cosine metric.
    """
    if NearestNeighbors is None:  # pragma: no cover
        raise ImportError("scikit-learn is required for embedding_neighbours")
    n = embeddings.shape[0]
    if n == 0:
        return []
    k = min(top_k + 1, n)
    nn = NearestNeighbors(n_neighbors=k, metric="cosine")
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)
    out: list[list[tuple[int, float]]] = []
    for i in range(n):
        row: list[tuple[int, float]] = []
        for dist, idx in zip(distances[i], indices[i]):
            if idx == i:
                continue
            sim = 1.0 - float(dist)
            row.append((int(idx), sim))
        row.sort(key=lambda t: (-t[1], t[0]))
        out.append(row[:top_k])
    return out


# ---------------------------------------------------------------------------
# Metric 4 — Weighted attribute overlap
# ---------------------------------------------------------------------------


def _coerce_numeric(value: Any) -> float | None:
    """Best-effort float coercion for numeric attribute matching.

    Accepts ints, floats, numeric strings (with optional whitespace).
    Returns ``None`` on NaN, empty, or non-coercible inputs.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        f = float(value)
        if not np.isfinite(f):
            return None
        return f
    s = str(value).strip()
    if not s or s.lower() in ("nan", "none", "null"):
        return None
    try:
        f = float(s)
    except ValueError:
        return None
    if not np.isfinite(f):
        return None
    return f


def _bag_for_row(
    row: pd.Series,
    columns: Sequence[str],
    *,
    numeric_columns: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build a ``{column: value}`` bag for a single row.

    String columns store the lowercase-stripped string. Columns listed
    in *numeric_columns* are coerced to ``float`` so the caller can
    apply a tolerance band; rows where the coercion fails are omitted
    for that column.
    """
    numeric_set = set(numeric_columns or ())
    bag: dict[str, Any] = {}
    for col in columns:
        if col not in row.index:
            continue
        val = row[col]
        if col in numeric_set:
            num = _coerce_numeric(val)
            if num is None:
                continue
            bag[col] = num
            continue
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        sv = str(val).strip().lower()
        if not sv or sv in ("nan", "none", "null"):
            continue
        bag[col] = sv
    return bag


def _numeric_match(a: float, b: float, spec: Mapping[str, Any] | None) -> bool:
    """Decide whether two numeric values match under *spec*.

    *spec* shape: ``{"kind": "absolute"|"relative", "tolerance": float}``.
    Absolute: ``|a - b| <= tolerance``. Relative: ``|a - b| <= tolerance
    * max(|a|, |b|)``. Falls back to exact equality when *spec* is None
    or malformed.
    """
    if spec is None:
        return a == b
    kind = str(spec.get("kind", "")).lower()
    try:
        tol = float(spec.get("tolerance", 0.0))
    except (TypeError, ValueError):
        return a == b
    if tol < 0:
        return a == b
    diff = abs(a - b)
    if kind == "absolute":
        return diff <= tol
    if kind == "relative":
        scale = max(abs(a), abs(b))
        if scale == 0.0:
            return diff <= tol
        return diff <= tol * scale
    return a == b


def attribute_overlap(
    bag_a: dict[str, Any],
    bag_b: dict[str, Any],
    weights: dict[str, float],
    *,
    numeric_overlap: Mapping[str, Mapping[str, Any]] | None = None,
) -> float:
    """Weighted Jaccard over two ``{column: value}`` bags.

    Parameters
    ----------
    bag_a, bag_b : dict
        Column → value mappings (see :func:`_bag_for_row`). Numeric
        columns carry float values; everything else carries lowercased
        strings.
    weights : dict
        Per-column non-negative weights. Missing columns get weight 0.
    numeric_overlap : mapping, optional
        ``{column: {"kind": "absolute"|"relative", "tolerance": float}}``.
        Columns listed here are matched with a tolerance band; e.g.
        ``{"founded": {"kind": "absolute", "tolerance": 1}}`` accepts a
        ±1-year mismatch as a match. Columns absent from this mapping
        fall back to exact equality.

    Returns
    -------
    float
        Weighted Jaccard similarity in ``[0, 1]``.
    """
    if not weights:
        return 0.0
    numer = 0.0
    denom = 0.0
    all_cols = set(bag_a) | set(bag_b)
    for col in all_cols:
        w = float(weights.get(col, 0.0))
        if w <= 0.0:
            continue
        va = bag_a.get(col)
        vb = bag_b.get(col)
        if va is not None and vb is not None:
            spec = None
            if numeric_overlap is not None:
                raw = numeric_overlap.get(col)
                if isinstance(raw, Mapping):
                    spec = raw
            if spec is not None and isinstance(va, float) and isinstance(vb, float):
                matched = _numeric_match(va, vb, spec)
            else:
                matched = va == vb
            if matched:
                numer += w
            denom += w
        else:
            denom += w
    if denom <= 0.0:
        return 0.0
    return numer / denom


def attribute_overlap_matrix(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str],
    weights: dict[str, float],
    numeric_columns: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Precompute per-row attribute bags for *columns*.

    Returns the list of bags in row order; callers use these to build
    neighbour lists with :func:`attribute_overlap_neighbours`. Columns
    in *numeric_columns* are coerced to floats so a tolerance band can
    be applied during scoring.
    """
    del weights  # Accepted for API symmetry; not used in bag construction.
    bags: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        bags.append(_bag_for_row(row, columns, numeric_columns=numeric_columns))
    return bags


def attribute_overlap_neighbours(
    bags: Sequence[dict[str, Any]],
    *,
    weights: dict[str, float],
    top_k: int,
    max_block_size: int = 2000,
    numeric_overlap: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[list[tuple[int, float]]]:
    """Compute top-K weighted-Jaccard neighbours over attribute bags.

    Two bags can only score above zero if they share at least one
    ``(column, value)`` pair (attribute_overlap's numerator is zero
    otherwise). We exploit this by building an inverted index from
    ``(column, value) -> [entity_idx, ...]`` and treating any
    co-occurrence as a candidate. The exact weighted-Jaccard scoring is
    then applied only over each entity's candidate union, preserving
    algorithm semantics.

    Buckets larger than *max_block_size* are dropped: a single attribute
    value shared by thousands of entities is non-selective and dominates
    runtime without changing the final top-K (other attribute buckets
    + the four other niche metrics still propose the entities that
    would have ranked highest in the dropped bucket).

    Parameters
    ----------
    bags : sequence of dict
        Per-entity ``{column: value}`` mappings.
    weights : dict
        Per-column non-negative weights forwarded to
        :func:`attribute_overlap`.
    top_k : int
        Number of neighbours to retain per entity (excludes self).
    max_block_size : int, default 2000
        Drop ``(column, value)`` buckets exceeding this size. Set to
        ``0`` to disable bucket-size filtering.

    Returns
    -------
    list of list of (int, float)
        For each entity, a sorted list of ``(neighbour_index, sim)``
        tuples (descending by similarity, length ``<= top_k``).
    """
    n = len(bags)

    weighted_columns = {col for col, w in weights.items() if float(w) > 0.0}
    if not weighted_columns:
        return [[] for _ in range(n)]

    # Numeric columns are excluded from the inverted index — float
    # values rarely byte-match, so they would never seed candidates.
    # They still participate in scoring once string columns pull a
    # candidate in, and the tolerance band is applied there.
    numeric_columns = set((numeric_overlap or {}).keys())
    indexable_columns = weighted_columns - numeric_columns

    value_buckets: dict[tuple[str, Any], list[int]] = defaultdict(list)
    for i, bag in enumerate(bags):
        for col, val in bag.items():
            if col not in indexable_columns:
                continue
            value_buckets[(col, val)].append(i)

    if max_block_size > 0:
        oversized = [
            key for key, idxs in value_buckets.items() if len(idxs) > max_block_size
        ]
        for key in oversized:
            del value_buckets[key]
        if oversized:
            logger.info(
                "attribute_overlap_neighbours: dropped %d oversized value buckets "
                "(size > %d)",
                len(oversized),
                max_block_size,
            )

    out: list[list[tuple[int, float]]] = []
    for i in range(n):
        bag = bags[i]
        candidates: set[int] = set()
        for col, val in bag.items():
            if col not in indexable_columns:
                continue
            bucket = value_buckets.get((col, val))
            if not bucket:
                continue
            candidates.update(bucket)
        candidates.discard(i)

        row: list[tuple[int, float]] = []
        if candidates:
            for j in candidates:
                sim = attribute_overlap(
                    bag, bags[j], weights, numeric_overlap=numeric_overlap
                )
                if sim > 0.0:
                    row.append((j, sim))
            row.sort(key=lambda t: (-t[1], t[0]))
        out.append(row[:top_k])
    return out


# ---------------------------------------------------------------------------
# Metric 5 — Label collision index
# ---------------------------------------------------------------------------


def label_collision_index(labels: Sequence[str]) -> dict[str, list[int]]:
    """Group entity indices by their normalised primary label.

    Parameters
    ----------
    labels : sequence of str
        Raw primary labels in canonical row order.

    Returns
    -------
    dict
        Mapping ``normalised_label -> [row_index, ...]`` with at least
        two entries per group. Empty labels are excluded.
    """
    groups: dict[str, list[int]] = {}
    for i, raw in enumerate(labels):
        key = normalize_label(raw)
        if not key:
            continue
        groups.setdefault(key, []).append(i)
    return {k: v for k, v in groups.items() if len(v) >= 2}


# ---------------------------------------------------------------------------
# Concatenation helper for text-heavy metrics
# ---------------------------------------------------------------------------


def build_text_corpus(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> list[str]:
    """Concatenate named columns into a text-block corpus.

    Non-string values are stringified; NaN/None become empty. Columns
    that do not exist in *frame* are silently skipped (the caller is
    responsible for config validation).

    Parameters
    ----------
    frame : pandas.DataFrame
        Entity frame in canonical row order.
    columns : sequence of str
        Columns to concatenate. Order is significant — baked into the
        embedding cache's ``input_column_concat_order`` metadata.

    Returns
    -------
    list of str
        Per-entity text blocks.
    """
    present = [c for c in columns if c in frame.columns]
    corpus: list[str] = []
    for _, row in frame.iterrows():
        parts: list[str] = []
        for col in present:
            val = row[col]
            if val is None:
                continue
            if isinstance(val, float) and np.isnan(val):
                continue
            s = str(val).strip()
            if s and s.lower() not in ("nan", "none", "null"):
                parts.append(s)
        corpus.append(" ".join(parts))
    return corpus


def build_label_list(
    frame: pd.DataFrame,
    primary_column: str,
) -> list[str]:
    """Extract the primary label column as a list of strings.

    Null values become empty strings; the row-count matches
    ``len(frame)``.
    """
    if primary_column not in frame.columns:
        return ["" for _ in range(len(frame))]
    labels: list[str] = []
    for val in frame[primary_column]:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            labels.append("")
        else:
            labels.append(str(val))
    return labels
