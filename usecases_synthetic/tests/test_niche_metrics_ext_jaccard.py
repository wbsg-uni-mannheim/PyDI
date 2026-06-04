"""Equivalence guard for the optimized lexical extended-Jaccard neighbours.

The 2026-06-03 optimization of ``lexical_extended_jaccard_neighbours``
(pre-tokenized scorer + exact-equality short-circuit + length-difference
prune + per-label parallelism) must be *byte-identical* to the original
serial, re-tokenising, unpruned implementation. This module pins that:
it reconstructs the original algorithm as a reference and asserts the
optimized public function returns exactly the same ``(index, sim)``
neighbour lists, on an adversarial synthetic set (designed to exercise
exact matches, fuzzy near-threshold matches, and length-pruned pairs)
and on a real papers-title sample.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from usecases_synthetic.lib.niche_metrics import (  # noqa: E402
    _levenshtein_ratio,
    lexical_extended_jaccard_neighbours,
    tokenize,
)

# ---------------------------------------------------------------------------
# Reference implementation = the pre-optimization algorithm, verbatim.
# ---------------------------------------------------------------------------


def _ref_ejacc(a: str, b: str, thr: float, stopwords) -> float:
    ta = tokenize(a, stopwords=stopwords)
    tb = tokenize(b, stopwords=stopwords)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
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
        if best_idx >= 0 and best_ratio >= thr:
            used.add(best_idx)
            matches += 1
    union = len(ta) + len(tb) - matches
    return 0.0 if union <= 0 else matches / union


def _ref_neighbours(labels, *, top_k, thr=0.8, stopwords=None, max_block_size=2000):
    n = len(labels)
    label_tokens = [tokenize(lbl, stopwords=stopwords) for lbl in labels]
    buckets: dict[str, list[int]] = defaultdict(list)
    for i, toks in enumerate(label_tokens):
        seen: set[str] = set()
        for t in toks:
            p = t[:3] if len(t) >= 3 else t
            if p in seen:
                continue
            buckets[p].append(i)
            seen.add(p)
    if max_block_size > 0:
        for p in [p for p, idx in buckets.items() if len(idx) > max_block_size]:
            del buckets[p]
    out = []
    for i in range(n):
        cands: set[int] = set()
        for t in label_tokens[i]:
            p = t[:3] if len(t) >= 3 else t
            b = buckets.get(p)
            if b:
                cands.update(b)
        cands.discard(i)
        row = []
        for j in cands:
            sim = _ref_ejacc(labels[i], labels[j], thr, stopwords)
            if sim > 0.0:
                row.append((j, sim))
        row.sort(key=lambda t: (-t[1], t[0]))
        out.append(row[:top_k])
    return out


# ---------------------------------------------------------------------------
# Adversarial label set — deterministic, exercises every code path.
# ---------------------------------------------------------------------------


def _adversarial_labels() -> list[str]:
    """Build labels stressing exact / fuzzy / length-pruned token pairs.

    Includes: identical titles, single-char typos (fuzzy match above
    threshold), length variants straddling the prune boundary, shared
    3-char prefixes with divergent suffixes, short tokens, and unrelated
    titles. Deterministic (no RNG)."""
    base = [
        "scalable graph processing on modern gpus",
        "scalable graph processing on modern gpu",  # singular -> length diff
        "scalable graph procesing on modern gpus",  # typo 'procesing'
        "scalable graph processing on modern tpus",  # gpus->tpus (prefix-shared)
        "deep learning for entity resolution",
        "deep learning for entity resolutions",  # plural
        "deep lerning for entity resolution",  # typo 'lerning'
        "a survey of data fusion techniques",
        "the survey of data fusion technique",  # article + singular
        "quantum error correction codes",
        "quantum error correcting codes",  # correction->correcting
        "transactions on database systems",
        "transaction on database system",
        "fast approximate nearest neighbor search",
        "fast approximate nearest neighbour search",  # us/uk spelling
        "abc def",  # short tokens
        "abc deff",  # short typo
        "completely unrelated title about cats",
        "another wholly different paper on weather",
        "x y z",  # very short
    ]
    # Repeat with index-tagged duplicates to grow buckets + force ties.
    labels: list[str] = []
    for rep in range(6):
        for b in base:
            labels.append(b if rep == 0 else f"{b} part {rep}")
    return labels


@pytest.mark.parametrize("top_k", [5, 10, 30])
def test_optimized_matches_reference_adversarial(top_k: int) -> None:
    labels = _adversarial_labels()
    new = lexical_extended_jaccard_neighbours(labels, top_k=top_k)
    ref = _ref_neighbours(labels, top_k=top_k)
    assert new == ref


def test_optimized_matches_reference_threshold_variants() -> None:
    labels = _adversarial_labels()
    for thr in (0.7, 0.8, 0.9, 1.0):
        new = lexical_extended_jaccard_neighbours(
            labels, top_k=10, inner_token_threshold=thr
        )
        ref = _ref_neighbours(labels, top_k=10, thr=thr)
        assert new == ref, f"mismatch at inner_token_threshold={thr}"


def test_optimized_matches_reference_real_papers_sample() -> None:
    """Byte-identical on a real papers-title sample (cross-source near-dups)."""
    try:
        from usecases_synthetic.lib.loaders import load_domain_sources

        sources = load_domain_sources("papers")
    except Exception:  # noqa: BLE001 - data may be unavailable in some checkouts
        pytest.skip("papers sources unavailable")

    titles = sources["dblp"]["title"].astype(str).tolist()
    # Diverse, bounded sample to keep the (slow) reference tractable.
    sample = [t for t in titles[::40] if t and t.strip()][:1500]
    new = lexical_extended_jaccard_neighbours(sample, top_k=15)
    ref = _ref_neighbours(sample, top_k=15)
    assert new == ref
