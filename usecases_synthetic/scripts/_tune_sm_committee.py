"""Tune SM committee hyperparameters per member.

One-off sweep harness for the R5 SM-stage hyperparameter optimisation
(2026-05-08). Loads each domain's baseline bundle, runs each matcher
under a parameter grid against every source, scores the combined
mapping vs ``sm_mapping_gold.csv``, and reports the best param combo
per member by average F1 across companies + games + music.

Skipped from the sweep:

- ``duplicate_majority``: structurally incompatible with the SM
  runner's (source, target_reference) call shape — id2 lookups on
  cross-source EM correspondences fail because target_reference
  doesn't carry source-2 IDs. Fixing this needs a runner refactor,
  not hyperparams.
- ``llm_openai`` / ``magneto_slm_llm``: LLM-cost-prohibitive sweep.
  ``llm_openai`` already at F1 0.952 on companies; verify a small
  ``num_rows`` sweep separately if motivated.

Usage::

    python usecases_synthetic/scripts/_tune_sm_committee.py \\
        --members label_jw,instance_tf_cosine,embedding_sbert,coma_hybrid \\
        --domains companies,games,music

Output is printed and written to ``cache/sm_tuning/sweep.json``.
"""

from __future__ import annotations

import argparse
import importlib
import itertools
import json
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd

from usecases_synthetic.lib.committee_sm import (
    _target_df_from_schema,
    score_sm_mapping,
)
from usecases_synthetic.lib.variant_loader import load_variant

logger = logging.getLogger("tune_sm")


REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "cache" / "sm_tuning"


def _make_target_df(bundle, gold_target_name: str | None) -> pd.DataFrame:
    fusion_frames: list[pd.DataFrame] = []
    if bundle.fusion_validation is not None:
        fusion_frames.append(bundle.fusion_validation)
    if bundle.fusion_gold is not None:
        fusion_frames.append(bundle.fusion_gold)
    return _target_df_from_schema(
        bundle.target_schema,
        bundle.sources,
        target_name=gold_target_name,
        fusion_frames=fusion_frames or None,
    )


def _gold_target_name(gold: pd.DataFrame) -> str | None:
    if "target_dataset" in gold.columns and not gold.empty:
        return str(gold["target_dataset"].iloc[0])
    return None


def _evaluate(
    matcher,
    bundle,
    *,
    target_df: pd.DataFrame,
    match_kwargs: dict[str, Any],
) -> dict[str, float]:
    mappings: list[pd.DataFrame] = []
    for source_name, source_df in bundle.sources.items():
        try:
            mapping = matcher.match(source_df, target_df, **match_kwargs)
            if mapping is not None and not mapping.empty:
                mappings.append(mapping)
        except Exception as e:
            logger.warning("Matcher failed on %s/%s: %s", bundle.domain, source_name, e)
    if not mappings:
        combined = pd.DataFrame(
            columns=[
                "source_dataset",
                "source_column",
                "target_dataset",
                "target_column",
                "score",
            ]
        )
    else:
        combined = pd.concat(mappings, ignore_index=True)
    return score_sm_mapping(combined, bundle.sm_mapping)


def _evaluate_duplicate(
    matcher,
    bundle,
    *,
    match_kwargs: dict[str, Any],
) -> dict[str, float]:
    """Evaluate a duplicate-typed matcher per source-pair.

    Mirrors :meth:`SMCommitteeRunner._run_duplicate_per_pair` semantics
    so the sweep harness scores duplicate-typed members the same way the
    production runner does (Option A in plan_s1_scale.md §"R5 SM
    duplicate-matcher fix").
    """
    from usecases_synthetic.lib.committee_sm import (
        _translate_cross_source_to_target,
    )

    gold = bundle.sm_mapping
    gold_lookup: dict[tuple[str, str], tuple[str, str]] = {}
    for _, row in gold.iterrows():
        gold_lookup[(str(row["source_dataset"]), str(row["source_column"]))] = (
            str(row["target_dataset"]),
            str(row["target_column"]),
        )

    mappings: list[pd.DataFrame] = []
    for src1_name, src2_name in bundle.source_pairs:
        em_pair = bundle.em_gold.get((src1_name, src2_name))
        if em_pair is None or em_pair.empty:
            continue
        if "label" in em_pair.columns:
            truthy = em_pair["label"].astype(str).str.lower()
            em_pos = em_pair[truthy.isin(("true", "1", "yes"))]
        else:
            em_pos = em_pair
        if em_pos.empty:
            continue
        src1_df = bundle.sources.get(src1_name)
        src2_df = bundle.sources.get(src2_name)
        if src1_df is None or src2_df is None:
            continue
        kw = dict(match_kwargs)
        kw["correspondences"] = em_pos
        try:
            cross = matcher.match(src1_df, src2_df, **kw)
        except Exception as e:
            logger.warning(
                "Duplicate matcher failed on %s/%s↔%s: %s",
                bundle.domain,
                src1_name,
                src2_name,
                e,
            )
            continue
        if cross is None or cross.empty:
            continue
        translated = _translate_cross_source_to_target(cross, gold_lookup)
        if not translated.empty:
            mappings.append(translated)

    if not mappings:
        combined = pd.DataFrame(
            columns=[
                "source_dataset",
                "source_column",
                "target_dataset",
                "target_column",
                "score",
            ]
        )
    else:
        combined = pd.concat(mappings, ignore_index=True).drop_duplicates(
            subset=[
                "source_dataset",
                "source_column",
                "target_dataset",
                "target_column",
            ]
        )
    return score_sm_mapping(combined, gold)


def _instantiate(module_path: str, cls_name: str, params: dict[str, Any]):
    mod = importlib.import_module(module_path)
    cls = getattr(mod, cls_name)
    return cls(**params)


# ---------------------------------------------------------------------------
# Per-member sweep specs
# ---------------------------------------------------------------------------
# Each spec: ((module, cls), init_param_grid, match_kwargs_grid).
# init_param_grid keys -> list of values (cartesian-product)
# match_kwargs_grid keys -> list of values (cartesian-product)


SPECS: dict[str, tuple[tuple[str, str], dict[str, list[Any]], dict[str, list[Any]]]] = {
    "duplicate_majority": (
        ("PyDI.schemamatching.duplicate_based", "DuplicateBasedSchemaMatcher"),
        {
            "vote_aggregation": ["majority"],
            "value_comparison": ["fuzzy", "exact"],
            "similarity_function": ["jaro_winkler", "levenshtein", "jaccard"],
            "similarity_threshold": [0.7, 0.8, 0.85, 0.9],
            "min_votes": [1, 2, 3],
            "ignore_zero_values": [True],
        },
        {
            "threshold": [0.05, 0.1, 0.2],
        },
    ),
    "label_jw": (
        ("PyDI.schemamatching.label_based", "LabelBasedSchemaMatcher"),
        {
            "similarity_function": [
                "jaro_winkler",
                "jaccard",
                "levenshtein",
                "cosine",
                "overlap",
                "jaro",
                "sorensen_dice",
            ],
            "tokenize": [True, False],
        },
        {
            "threshold": [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
        },
    ),
    "instance_tf_cosine": (
        ("PyDI.schemamatching.instance_based", "InstanceBasedSchemaMatcher"),
        {
            "vector_creation_method": [
                "term_frequencies",
                "binary_occurrence",
                "tfidf",
            ],
            "similarity_function": ["cosine", "jaccard", "overlap"],
            "max_sample_size": [200, 500, 1000],
            # PyDI computes non_null_ratio as post-sampled / source_size,
            # so any positive threshold filters every column out on large
            # sources (games dbpedia 46k). Force 0.0.
            "min_non_null_ratio": [0.0],
        },
        {
            "threshold": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        },
    ),
    "embedding_sbert": (
        ("usecases_synthetic.lib.embedding_sm_matcher", "EmbeddingBasedSchemaMatcher"),
        {
            "model_name": [
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/all-mpnet-base-v2",
                "BAAI/bge-small-en-v1.5",
                "BAAI/bge-base-en-v1.5",
            ],
            "max_sample_size": [20, 50],
            "random_state": [42],
        },
        {
            "threshold": [0.5, 0.55, 0.6, 0.65, 0.7],
        },
    ),
    "magneto_slm_llm": (
        ("usecases_synthetic.lib.magneto_sm_matcher", "MagnetoSchemaMatcher"),
        {
            "embedding_model": [
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/all-mpnet-base-v2",
                "BAAI/bge-small-en-v1.5",
                "BAAI/bge-base-en-v1.5",
            ],
            "encoding_mode": ["header_values_verbose"],
            "sampling_mode": ["priority_sampling"],
            "sampling_size": [10],
            "topk": [5, 10, 20],
            "embedding_threshold": [0.1],
            "use_llm_rerank": [True],
            "llm_model": ["openai/gpt-5.4-mini"],
            "llm_temperature": [0.0],
        },
        {
            "threshold": [0.3, 0.4, 0.5],
        },
    ),
    "magneto_ablation": (
        # Stage-3 ablation: fix the stage-2 winner (bge-base + topk=20 +
        # threshold=0.3) and vary encoding_mode + use_llm_rerank to measure
        # how much each contributes to Magneto's F1.
        ("usecases_synthetic.lib.magneto_sm_matcher", "MagnetoSchemaMatcher"),
        {
            "embedding_model": ["BAAI/bge-base-en-v1.5"],
            "encoding_mode": [
                "header_only",
                "header_values_default",
                "header_values_verbose",
            ],
            "sampling_mode": ["priority_sampling"],
            "sampling_size": [10],
            "topk": [20],
            "embedding_threshold": [0.1],
            "use_llm_rerank": [True, False],
            "llm_model": ["openai/gpt-5.4-mini"],
            "llm_temperature": [0.0],
        },
        {
            "threshold": [0.3],
        },
    ),
    "coma_hybrid": (
        ("usecases_synthetic.lib.coma_sm_matcher", "ComaSchemaMatcher"),
        {
            "max_n": [1, 2],
            "use_instances": [True, False],
            "use_schema": [True],
            "delta": [0.1, 0.15, 0.2],
            "coma_threshold": [0.0],
        },
        {
            "threshold": [0.2, 0.3, 0.4, 0.5],
        },
    ),
}


def _grid(d: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not d:
        return [{}]
    keys = list(d.keys())
    return [
        dict(zip(keys, combo, strict=True))
        for combo in itertools.product(*[d[k] for k in keys])
    ]


def sweep_member(member: str, bundles: dict[str, Any]) -> list[dict[str, Any]]:
    (mod, cls), init_grid, match_grid = SPECS[member]
    init_combos = _grid(init_grid)
    match_combos = _grid(match_grid)

    results: list[dict[str, Any]] = []
    matcher_cache: dict[tuple, Any] = {}
    target_cache: dict[str, pd.DataFrame] = {}

    n_total = len(init_combos) * len(match_combos)
    logger.info(
        "Sweeping %s: %d init x %d match = %d combos x %d domains",
        member,
        len(init_combos),
        len(match_combos),
        n_total,
        len(bundles),
    )

    for i, init_params in enumerate(init_combos):
        # Cache matcher instances keyed by init params (esp. for SBERT).
        key = tuple(sorted(init_params.items()))
        if key not in matcher_cache:
            matcher_cache[key] = _instantiate(mod, cls, init_params)
        matcher = matcher_cache[key]

        for match_kwargs in match_combos:
            row: dict[str, Any] = {
                "member": member,
                **{f"init.{k}": v for k, v in init_params.items()},
                **{f"match.{k}": v for k, v in match_kwargs.items()},
            }
            f1s = []
            for domain, bundle in bundles.items():
                if domain not in target_cache:
                    target_cache[domain] = _make_target_df(
                        bundle, _gold_target_name(bundle.sm_mapping)
                    )
                t0 = time.monotonic()
                if member == "duplicate_majority":
                    metrics = _evaluate_duplicate(
                        matcher,
                        bundle,
                        match_kwargs=match_kwargs,
                    )
                else:
                    metrics = _evaluate(
                        matcher,
                        bundle,
                        target_df=target_cache[domain],
                        match_kwargs=match_kwargs,
                    )
                elapsed = time.monotonic() - t0
                row[f"{domain}.f1"] = metrics["f1"]
                row[f"{domain}.precision"] = metrics["precision"]
                row[f"{domain}.recall"] = metrics["recall"]
                row[f"{domain}.runtime_s"] = elapsed
                f1s.append(metrics["f1"])
            row["mean_f1"] = sum(f1s) / len(f1s)
            row["min_f1"] = min(f1s)
            results.append(row)
            if (i * len(match_combos) + match_combos.index(match_kwargs) + 1) % 10 == 0:
                logger.info(
                    "  %s combo %d/%d mean_f1=%.3f",
                    member,
                    i * len(match_combos) + match_combos.index(match_kwargs) + 1,
                    n_total,
                    row["mean_f1"],
                )
    return results


def report(member: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    rows_sorted = sorted(rows, key=lambda r: r["mean_f1"], reverse=True)
    print(f"\n=== {member} top-5 by mean_f1 ===")
    keys = [
        k
        for k in rows_sorted[0].keys()
        if k.startswith("init.") or k.startswith("match.")
    ]
    keys = sorted(keys)
    for r in rows_sorted[:5]:
        params = ", ".join(f"{k.split('.', 1)[1]}={r[k]}" for k in keys)
        domain_f1s = ", ".join(
            f"{d}={r[f'{d}.f1']:.3f}"
            for d in sorted(
                {
                    k.split(".", 1)[0]
                    for k in r.keys()
                    if "." in k and k.split(".")[1] == "f1"
                }
            )
        )
        print(f"  mean_f1={r['mean_f1']:.3f} | {params} | per-domain: {domain_f1s}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--members",
        default="label_jw,instance_tf_cosine,embedding_sbert,coma_hybrid",
    )
    ap.add_argument("--domains", default="companies,games,music")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    members = [m.strip() for m in args.members.split(",") if m.strip()]
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]

    logger.info("Loading bundles for: %s", domains)
    bundles = {d: load_variant(d, "baseline") for d in domains}

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, list[dict[str, Any]]] = {}
    for member in members:
        if member not in SPECS:
            logger.warning("Unknown member %s; skipping", member)
            continue
        results = sweep_member(member, bundles)
        out[member] = results
        report(member, results)

    cache_path = CACHE_DIR / "sweep.json"
    with cache_path.open("w") as f:
        json.dump(out, f, indent=2)
    logger.info("Wrote %s", cache_path)


if __name__ == "__main__":
    main()
