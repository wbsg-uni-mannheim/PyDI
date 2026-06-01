"""Tune EM blocking committee hyperparameters per member.

One-off sweep harness for the R5 EM-blocking-stage tuning pass
(2026-05-10). Closes Pending #1 (embedding-model brainstorm) + the
user-directed expansions (2026-05-10): standard-blocker blocking-key
panel + sorted-neighbourhood-blocker key/window sweep.

Per-domain results land at ``cache/em_blocking_tuning/sweep.json``.
Per-(member, init-config) score = mean pair_recall + mean
reduction_ratio across all source-pairs in the domain; winner is the
config that clears the per-domain ``recall_floor`` (0.97 default)
with the highest reduction_ratio, ties broken alphabetically on init.

Three sub-sweeps:

- ``embedding`` (Pending #1) — 5-model panel for ``embedding_blocker``.
- ``standard`` — multi-key panel for ``standard_blocker`` (prefix,
  token, value, compound). Per-domain key candidates declared inline.
- ``sn`` — key + window panel for ``sorted_neighbourhood_blocker``.

Token + BM25 sweeps deliberately deferred: their current YAML
defaults (min_token_len=2, k1=1.5/b=0.75/stopwords=english) are
sensible non-tuned baselines. If R7.2 monotonicity reveals a recall
gap, revisit.

Usage::

    python usecases_synthetic/scripts/_tune_em_blocking_committee.py \\
        --sub-sweeps embedding,standard,sn \\
        --domains companies,games,music
"""

from __future__ import annotations

import argparse
import importlib
import itertools
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

from usecases_synthetic.lib.column_mapping import apply_column_mapping  # noqa: E402
from usecases_synthetic.lib.committee_em_scoring import (  # noqa: E402
    blocking_pair_recall,
    reduction_ratio,
)
from usecases_synthetic.lib.variant_loader import load_variant  # noqa: E402

logger = logging.getLogger("tune_em_blocking")


CACHE_DIR = REPO_ROOT / "cache" / "em_blocking_tuning"

RECALL_FLOOR = 0.97


# ---------------------------------------------------------------------------
# Blocking-key derivation
# ---------------------------------------------------------------------------


def _derive_key_column(df: pd.DataFrame, spec: dict[str, Any]) -> str:
    """Generate a derived blocking-key column on *df* in place.

    Spec shapes:

    - ``{type: 'token', column: 'name'}`` → ``<column>_first_token``
      (first non-trivial alphanumeric token; matches
      :func:`committee_em._generate_blocking_keys`).
    - ``{type: 'prefix', column: 'name', n: 3}`` → ``<column>_first_<n>``
      (first N lowercased alphanumeric chars).
    - ``{type: 'value', column: 'country'}`` → ``<column>_norm``
      (lowercased + stripped; for nominal attrs).
    - ``{type: 'compound', parts: [<spec>, <spec>, ...]}`` →
      ``__key_<hash>``, value = parts joined with '|'.

    Returns the derived column name.
    """
    import re as _re

    kind = spec["type"]

    def _alnum_lower(s: Any) -> str:
        if s is None:
            return ""
        s = str(s)
        return _re.sub(r"[^a-z0-9]", "", s.lower())

    def _first_token(s: Any) -> str:
        if s is None:
            return ""
        s = str(s).lower()
        toks = [t for t in _re.split(r"[^a-z0-9]", s) if len(t) > 1]
        return toks[0] if toks else _alnum_lower(s)

    if kind == "token":
        col = spec["column"]
        out_col = f"{col}_first_token"
        if out_col not in df.columns and col in df.columns:
            df[out_col] = df[col].apply(_first_token)
        return out_col

    if kind == "prefix":
        col = spec["column"]
        n = int(spec.get("n", 3))
        out_col = f"{col}_first_{n}"
        if out_col not in df.columns and col in df.columns:
            df[out_col] = df[col].apply(lambda v: _alnum_lower(v)[:n])
        return out_col

    if kind == "value":
        col = spec["column"]
        out_col = f"{col}_norm"
        if out_col not in df.columns and col in df.columns:
            df[out_col] = df[col].apply(
                lambda v: "" if v is None else str(v).strip().lower()
            )
        return out_col

    if kind == "compound":
        # Derive all parts first, then concat.
        part_cols = [_derive_key_column(df, p) for p in spec["parts"]]
        out_col = "__cmp_" + "_".join(part_cols)
        if out_col not in df.columns:
            df[out_col] = df[part_cols].astype(str).agg("|".join, axis=1)
        return out_col

    raise ValueError(f"Unknown key spec type: {kind!r}")


# ---------------------------------------------------------------------------
# Per-domain context
# ---------------------------------------------------------------------------


def _column_mapping_for_domain(
    bundle, em_blocking_committee_path: Path
) -> dict[str, dict[str, str]]:
    """Load the static column_mapping from the per-domain EM blocking YAML."""
    import yaml

    with open(em_blocking_committee_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    static = raw.get("column_mapping", {})
    # Translate through K8 renames if any (the runner does this in
    # production via bundle.resolve_column_mapping; mirror it here).
    return bundle.resolve_column_mapping(static)


def _domain_context(domain: str) -> dict[str, Any]:
    bundle = load_variant(domain, level="baseline")
    committee_dir = REPO_ROOT / "usecases_synthetic" / "config" / "committees"
    suffix = "" if domain == "companies" else f"_{domain}"
    em_yaml = committee_dir / f"em_blocking_committee{suffix}.yaml"
    column_mapping = _column_mapping_for_domain(bundle, em_yaml)
    # Pre-apply column_mapping to each source DF so all blockers see
    # canonical column names.
    sources_mapped: dict[str, pd.DataFrame] = {}
    for src_name, df in bundle.sources.items():
        rename_map = column_mapping.get(src_name, {})
        sources_mapped[src_name] = (
            apply_column_mapping(df, rename_map) if rename_map else df.copy()
        )
    return {
        "domain": domain,
        "bundle": bundle,
        "sources": sources_mapped,
        "em_gold": bundle.em_gold,
        "source_pairs": bundle.source_pairs,
        "column_mapping": column_mapping,
    }


# ---------------------------------------------------------------------------
# Score one blocker config across a domain
# ---------------------------------------------------------------------------


def _import_class(module: str, cls: str):
    mod = importlib.import_module(module)
    return getattr(mod, cls)


def _score_blocker_config(
    ctx: dict[str, Any],
    *,
    blocker_class: tuple[str, str],
    blocker_params: dict[str, Any],
    derive_keys: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run a blocker config across every source pair in *ctx*; aggregate."""
    cls = _import_class(*blocker_class)
    sources = ctx["sources"]
    em_gold = ctx["em_gold"]
    pair_results: list[dict[str, Any]] = []
    t0 = time.monotonic()
    for src1, src2 in ctx["source_pairs"]:
        gold = em_gold.get((src1, src2))
        if gold is None or gold.empty:
            continue
        df_left = sources[src1].copy()
        df_right = sources[src2].copy()
        # Derive any blocking-key columns.
        if derive_keys:
            for spec in derive_keys:
                _derive_key_column(df_left, spec)
                _derive_key_column(df_right, spec)
        try:
            blocker = cls(df_left, df_right, id_column="id", **blocker_params)
            candidates = blocker.materialize()
        except Exception as e:
            logger.warning(
                "Blocker %s failed on %s-%s: %s",
                blocker_class,
                src1,
                src2,
                e,
            )
            candidates = pd.DataFrame(columns=["id1", "id2"])
        rec = blocking_pair_recall(candidates, gold)
        rr = reduction_ratio(candidates, len(df_left), len(df_right))
        pair_results.append(
            {
                "pair": f"{src1}_{src2}",
                "pair_recall": rec["pair_recall"],
                "reduction_ratio": rr["reduction_ratio"],
                "n_candidates": rr["candidate_count"],
                "n_left": len(df_left),
                "n_right": len(df_right),
            }
        )
    elapsed = time.monotonic() - t0
    if not pair_results:
        return {
            "mean_pair_recall": 0.0,
            "mean_reduction_ratio": 0.0,
            "per_pair": [],
            "elapsed_s": round(elapsed, 1),
        }
    mean_recall = sum(p["pair_recall"] for p in pair_results) / len(pair_results)
    mean_rr = sum(p["reduction_ratio"] for p in pair_results) / len(pair_results)
    return {
        "mean_pair_recall": mean_recall,
        "mean_reduction_ratio": mean_rr,
        "per_pair": pair_results,
        "elapsed_s": round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# Sweep grids
# ---------------------------------------------------------------------------


EMBEDDING_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-small-en-v1.5",
    "intfloat/e5-base-v2",
]


# Per-domain primary text column for the embedding blocker. companies /
# games / music share ``name``; products has ``title`` instead. New
# domains should be added here.
EMBEDDING_TEXT_COLS_BY_DOMAIN: dict[str, list[str]] = {
    "companies": ["name"],
    "games": ["name"],
    "music": ["name"],
    "products": ["title"],
}


def _embedding_sweep_grid(domain: str) -> list[dict[str, Any]]:
    text_cols = EMBEDDING_TEXT_COLS_BY_DOMAIN.get(domain, ["name"])
    out: list[dict[str, Any]] = []
    for model in EMBEDDING_MODELS:
        out.append(
            {
                "model": model,
                "text_cols": text_cols,
                "top_k": 50,
                "threshold": 0.3,
            }
        )
    return out


# Per-domain standard_blocker key panel.
STANDARD_KEY_PANEL: dict[str, list[list[dict[str, Any]]]] = {
    "companies": [
        [{"type": "token", "column": "name"}],
        [{"type": "prefix", "column": "name", "n": 3}],
        [{"type": "prefix", "column": "name", "n": 5}],
        [
            {
                "type": "compound",
                "parts": [
                    {"type": "token", "column": "name"},
                    {"type": "value", "column": "country"},
                ],
            }
        ],
        [
            {
                "type": "compound",
                "parts": [
                    {"type": "prefix", "column": "name", "n": 3},
                    {"type": "prefix", "column": "country", "n": 3},
                ],
            }
        ],
    ],
    "games": [
        [{"type": "token", "column": "name"}],
        [{"type": "prefix", "column": "name", "n": 3}],
        [{"type": "prefix", "column": "name", "n": 5}],
        [
            {
                "type": "compound",
                "parts": [
                    {"type": "token", "column": "name"},
                    {"type": "value", "column": "platform"},
                ],
            }
        ],
    ],
    "music": [
        [{"type": "token", "column": "name"}],
        [{"type": "prefix", "column": "name", "n": 3}],
        [{"type": "prefix", "column": "name", "n": 5}],
        [
            {
                "type": "compound",
                "parts": [
                    {"type": "token", "column": "name"},
                    {"type": "prefix", "column": "artist", "n": 3},
                ],
            }
        ],
    ],
    "products": [
        # Title is products' primary high-signal column (parallels
        # ``name`` for the other 3 domains). Product_type (~10
        # distinct categorical values: GPU / CPU / SSD / HDD / ...)
        # is the canonical block-partitioning attribute used by the
        # notebook (cell 27f0c352: ``StandardBlocker(... on=['product_type'])``).
        # Brand is the second most discriminative categorical.
        [{"type": "token", "column": "title"}],
        [{"type": "prefix", "column": "title", "n": 3}],
        [{"type": "prefix", "column": "title", "n": 5}],
        [{"type": "value", "column": "product_type"}],
        [
            {
                "type": "compound",
                "parts": [
                    {"type": "token", "column": "title"},
                    {"type": "value", "column": "product_type"},
                ],
            }
        ],
        [
            {
                "type": "compound",
                "parts": [
                    {"type": "prefix", "column": "title", "n": 5},
                    {"type": "value", "column": "brand"},
                ],
            }
        ],
    ],
}


# Per-domain sorted_neighbourhood_blocker key panel × window.
SN_KEY_PANEL_BY_DOMAIN: dict[str, list[dict[str, Any]]] = {
    "companies": [
        {"type": "value", "column": "name"},
        {"type": "prefix", "column": "name", "n": 5},
        {"type": "token", "column": "name"},
    ],
    "games": [
        {"type": "value", "column": "name"},
        {"type": "prefix", "column": "name", "n": 5},
        {"type": "token", "column": "name"},
    ],
    "music": [
        {"type": "value", "column": "name"},
        {"type": "prefix", "column": "name", "n": 5},
        {"type": "token", "column": "name"},
    ],
    "products": [
        # SNB sliding-window keys for products. Title is the primary
        # signal; product_type partitions categorically.
        {"type": "prefix", "column": "title", "n": 5},
        {"type": "token", "column": "title"},
        {"type": "value", "column": "product_type"},
    ],
}

SN_WINDOWS = [10, 20, 40]


# sc_block sub-sweep (retrieval-side only — the trained checkpoint is fixed).
SCBLOCK_TOP_KS = [20, 50, 100]
SCBLOCK_THRESHOLDS = [0.0, 0.3, 0.5]


# ---------------------------------------------------------------------------
# Per-sub-sweep driver
# ---------------------------------------------------------------------------


def _run_embedding_sweep(contexts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    # Build per-domain grids and dispatch each (domain, model) cell with
    # its own ``text_cols`` (products needs ``title``, not ``name``).
    for model in EMBEDDING_MODELS:
        per_domain: dict[str, dict[str, float]] = {}
        for domain, ctx in contexts.items():
            text_cols = EMBEDDING_TEXT_COLS_BY_DOMAIN.get(domain, ["name"])
            params = {
                "model": model,
                "text_cols": text_cols,
                "top_k": 50,
                "threshold": 0.3,
            }
            res = _score_blocker_config(
                ctx,
                blocker_class=(
                    "PyDI.entitymatching.blocking.embedding",
                    "EmbeddingBlocker",
                ),
                blocker_params=params,
            )
            per_domain[domain] = res
            logger.info(
                "embedding model=%s domain=%s recall=%.3f rr=%.3f (%.1fs)",
                model,
                domain,
                res["mean_pair_recall"],
                res["mean_reduction_ratio"],
                res["elapsed_s"],
            )
        mean_recall = sum(d["mean_pair_recall"] for d in per_domain.values()) / max(
            len(per_domain), 1
        )
        mean_rr = sum(d["mean_reduction_ratio"] for d in per_domain.values()) / max(
            len(per_domain), 1
        )
        rows.append(
            {
                "sweep": "embedding",
                "params": params,
                "mean_recall": mean_recall,
                "mean_rr": mean_rr,
                "per_domain": per_domain,
            }
        )
    return rows


def _run_standard_sweep(contexts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    # Iterate per-domain panel since key candidates are per-domain.
    for domain, ctx in contexts.items():
        for key_spec_list in STANDARD_KEY_PANEL.get(domain, []):
            # Derive columns up-front, then ``on`` is the derived names.
            derive_keys = list(key_spec_list)
            on_cols: list[str] = []
            # Determine target derived column names.
            df_probe = next(iter(ctx["sources"].values())).copy()
            for spec in derive_keys:
                on_cols.append(_derive_key_column(df_probe, spec))
            res = _score_blocker_config(
                ctx,
                blocker_class=(
                    "PyDI.entitymatching.blocking.standard",
                    "StandardBlocker",
                ),
                blocker_params={"on": on_cols},
                derive_keys=derive_keys,
            )
            label = ",".join(on_cols)
            logger.info(
                "standard domain=%s keys=%s recall=%.3f rr=%.3f (%.1fs)",
                domain,
                label,
                res["mean_pair_recall"],
                res["mean_reduction_ratio"],
                res["elapsed_s"],
            )
            rows.append(
                {
                    "sweep": "standard",
                    "domain": domain,
                    "on_cols": on_cols,
                    "key_spec": key_spec_list,
                    "mean_recall": res["mean_pair_recall"],
                    "mean_rr": res["mean_reduction_ratio"],
                    "per_pair": res["per_pair"],
                    "elapsed_s": res["elapsed_s"],
                }
            )
    return rows


def _run_scblock_sweep(contexts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Sweep ``top_k`` x ``threshold`` for the SC-Block hybrid blocker.

    Loads the per-domain checkpoint from
    ``cache/sc_block_checkpoints/<domain>/best/``; the encoder is fixed
    so the grid only varies retrieval-side knobs. Per-domain
    ``text_cols`` are sourced from
    :data:`usecases_synthetic.lib.sc_block_train.DOMAIN_TEXT_COLS` so
    the inference shape matches the SC-Block training shape.
    """
    from usecases_synthetic.lib.sc_block_train import DOMAIN_TEXT_COLS

    rows: list[dict[str, Any]] = []
    cache_root = REPO_ROOT / "usecases_synthetic" / "cache" / "sc_block_checkpoints"
    for domain, ctx in contexts.items():
        ckpt = cache_root / domain / "best"
        if not ckpt.exists():
            logger.warning(
                "sc_block sweep: checkpoint missing at %s; skipping %s",
                ckpt,
                domain,
            )
            continue
        text_cols = DOMAIN_TEXT_COLS.get(domain)
        if text_cols is None:
            logger.warning("sc_block sweep: no text_cols for %s; skipping", domain)
            continue
        for top_k in SCBLOCK_TOP_KS:
            for threshold in SCBLOCK_THRESHOLDS:
                params = {
                    "text_cols": text_cols,
                    "checkpoint_path": str(ckpt),
                    "top_k": top_k,
                    "threshold": threshold,
                    "index_backend": "sklearn",
                }
                res = _score_blocker_config(
                    ctx,
                    blocker_class=(
                        "usecases_synthetic.lib.sc_block_blocker",
                        "SCBlockBlocker",
                    ),
                    blocker_params=params,
                )
                logger.info(
                    "sc_block domain=%s top_k=%d threshold=%.2f recall=%.3f rr=%.3f (%.1fs)",
                    domain,
                    top_k,
                    threshold,
                    res["mean_pair_recall"],
                    res["mean_reduction_ratio"],
                    res["elapsed_s"],
                )
                rows.append(
                    {
                        "sweep": "sc_block",
                        "domain": domain,
                        "top_k": top_k,
                        "threshold": threshold,
                        "mean_recall": res["mean_pair_recall"],
                        "mean_rr": res["mean_reduction_ratio"],
                        "per_pair": res["per_pair"],
                        "elapsed_s": res["elapsed_s"],
                    }
                )
    return rows


def _run_sn_sweep(contexts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for domain, ctx in contexts.items():
        for key_spec in SN_KEY_PANEL_BY_DOMAIN.get(domain, []):
            for window in SN_WINDOWS:
                df_probe = next(iter(ctx["sources"].values())).copy()
                key_col = _derive_key_column(df_probe, key_spec)
                res = _score_blocker_config(
                    ctx,
                    blocker_class=(
                        "PyDI.entitymatching.blocking.sorted_neighbourhood",
                        "SortedNeighbourhoodBlocker",
                    ),
                    blocker_params={"key": key_col, "window": window},
                    derive_keys=[key_spec],
                )
                logger.info(
                    "sn domain=%s key=%s window=%d recall=%.3f rr=%.3f (%.1fs)",
                    domain,
                    key_col,
                    window,
                    res["mean_pair_recall"],
                    res["mean_reduction_ratio"],
                    res["elapsed_s"],
                )
                rows.append(
                    {
                        "sweep": "sn",
                        "domain": domain,
                        "key_col": key_col,
                        "key_spec": key_spec,
                        "window": window,
                        "mean_recall": res["mean_pair_recall"],
                        "mean_rr": res["mean_reduction_ratio"],
                        "per_pair": res["per_pair"],
                        "elapsed_s": res["elapsed_s"],
                    }
                )
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sub-sweeps",
        default="embedding,standard,sn",
        help="Comma-separated subset of {embedding, standard, sn, sc_block}.",
    )
    parser.add_argument(
        "--domains",
        default="companies,games,music",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=CACHE_DIR / "sweep.json",
    )
    args = parser.parse_args()

    sub_sweeps = [s.strip() for s in args.sub_sweeps.split(",") if s.strip()]
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]

    contexts = {d: _domain_context(d) for d in domains}
    for d, ctx in contexts.items():
        logger.info(
            "Domain %s: %d source pairs, source row counts: %s",
            d,
            len(ctx["source_pairs"]),
            {n: len(df) for n, df in ctx["sources"].items()},
        )

    results: dict[str, list[dict[str, Any]]] = {}
    if "embedding" in sub_sweeps:
        logger.info("=== embedding sweep ===")
        results["embedding"] = _run_embedding_sweep(contexts)
    if "standard" in sub_sweeps:
        logger.info("=== standard sweep ===")
        results["standard"] = _run_standard_sweep(contexts)
    if "sn" in sub_sweeps:
        logger.info("=== sn sweep ===")
        results["sn"] = _run_sn_sweep(contexts)
    if "sc_block" in sub_sweeps:
        logger.info("=== sc_block sweep ===")
        results["sc_block"] = _run_scblock_sweep(contexts)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote %s", args.out)

    # Print concise tables.
    if "embedding" in results:
        print("\n=== Embedding model sweep (mean across 3 domains) ===")
        print(f"{'Model':<55} {'recall':>8} {'rr':>8}")
        rows = sorted(
            results["embedding"], key=lambda r: (-r["mean_recall"], -r["mean_rr"])
        )
        for r in rows:
            print(
                f"  {r['params']['model']:<53} {r['mean_recall']:>8.3f} {r['mean_rr']:>8.3f}"
            )
    if "standard" in results:
        print("\n=== Standard blocker keys (per-domain) ===")
        by_domain: dict[str, list[dict[str, Any]]] = {}
        for r in results["standard"]:
            by_domain.setdefault(r["domain"], []).append(r)
        for d in domains:
            print(f"\n  Domain: {d}")
            print(f"  {'Keys':<60} {'recall':>8} {'rr':>8}")
            rows = sorted(
                by_domain.get(d, []),
                key=lambda r: (-r["mean_recall"], -r["mean_rr"]),
            )
            for r in rows:
                print(
                    f"    {','.join(r['on_cols']):<58} {r['mean_recall']:>8.3f} {r['mean_rr']:>8.3f}"
                )
    if "sn" in results:
        print("\n=== Sorted neighbourhood blocker (per-domain) ===")
        by_domain = {}
        for r in results["sn"]:
            by_domain.setdefault(r["domain"], []).append(r)
        for d in domains:
            print(f"\n  Domain: {d}")
            print(f"  {'Key':<30} {'window':>8} {'recall':>8} {'rr':>8}")
            rows = sorted(
                by_domain.get(d, []),
                key=lambda r: (-r["mean_recall"], -r["mean_rr"]),
            )
            for r in rows:
                print(
                    f"    {r['key_col']:<28} {r['window']:>8} "
                    f"{r['mean_recall']:>8.3f} {r['mean_rr']:>8.3f}"
                )
    if "sc_block" in results:
        print("\n=== SC-Block hybrid blocker (per-domain) ===")
        by_domain = {}
        for r in results["sc_block"]:
            by_domain.setdefault(r["domain"], []).append(r)
        for d in domains:
            print(f"\n  Domain: {d}")
            print(f"  {'top_k':>6} {'threshold':>10} {'recall':>8} {'rr':>8}")
            rows = sorted(
                by_domain.get(d, []),
                key=lambda r: (-r["mean_recall"], -r["mean_rr"]),
            )
            for r in rows:
                print(
                    f"  {r['top_k']:>6} {r['threshold']:>10.2f} "
                    f"{r['mean_recall']:>8.3f} {r['mean_rr']:>8.3f}"
                )


if __name__ == "__main__":
    main()
