#!/usr/bin/env python3
"""Audit where each domain's best-of-breed bundle physically reads its
EM / SM / fusion gold from.

The user directive (2026-06-01) is that the BoB pipeline must evaluate
on train/val/test sets from canonical PyDI ``usecases/<domain>/``, not
from the derived ``usecases_synthetic/usecases/<domain>/`` tree.

This script:
1. Loads the bundle per domain and prints ``variant_root``.
2. Flags domains that read from ``usecases_synthetic/`` as DIVERGENT.
3. For ``products`` specifically (the one domain with a synthetic
   ``data_root`` override), compares the synthetic-translated EM gold
   files row-by-row against canonical ``usecases/products/input/
   entity_matching_gt/`` and reports content equivalence.

Usage
-----
    python pipelines/scripts/audit_data_sources.py
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from pipelines.lib.bundle import load_pipeline_bundle
from pipelines.lib.pipeline import PipelineConfig

DOMAINS = ("products", "music", "games", "companies", "papers")
SYNTHETIC_PREFIX = REPO_ROOT / "usecases_synthetic"
CANONICAL_PREFIX = REPO_ROOT / "usecases"


def _classify_root(root: Path) -> str:
    try:
        root.relative_to(SYNTHETIC_PREFIX)
        return "SYNTHETIC"
    except ValueError:
        pass
    try:
        root.relative_to(CANONICAL_PREFIX)
        return "CANONICAL"
    except ValueError:
        pass
    return "OTHER"


def _audit_domain(domain: str) -> dict:
    config_path = REPO_ROOT / "pipelines" / "configs" / f"{domain}.yaml"
    bundle_source = "synthetic_baseline"
    if config_path.exists():
        cfg = PipelineConfig.from_yaml(config_path)
        bundle_source = cfg.bundle_source
    bundle = load_pipeline_bundle(domain, bundle_source=bundle_source)
    root = Path(bundle.variant_root).resolve()
    classification = _classify_root(root)
    em_pairs = sorted(bundle.em_gold.keys()) if bundle.em_gold else []
    em_first_pair = em_pairs[0] if em_pairs else None
    em_summary = {}
    if em_first_pair and bundle.em_gold:
        df = bundle.em_gold[em_first_pair]
        em_summary = {
            "first_pair": em_first_pair,
            "rows": int(len(df)),
            "columns": list(df.columns),
            "id1_sample": str(df["id1"].iloc[0]) if len(df) else None,
            "id2_sample": str(df["id2"].iloc[0]) if len(df) else None,
            "label_sample": str(df["label"].iloc[0]) if len(df) else None,
        }
    return {
        "domain": domain,
        "variant_root": str(root.relative_to(REPO_ROOT)),
        "classification": classification,
        "n_em_pairs": len(em_pairs),
        "em_pairs": em_pairs,
        "em_summary": em_summary,
        "has_sm_gold": (
            bool(
                getattr(bundle, "sm_mapping", None) is not None
                and not bundle.sm_mapping.empty
            )
            if hasattr(bundle, "sm_mapping")
            else False
        ),
        "n_sources": len(bundle.sources) if bundle.sources else 0,
    }


def _read_em_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _verify_products_canonical_equivalence() -> dict:
    """For each products pair + split, compare the synthetic-translated
    EM file against canonical (header + ID prefix + label format)."""
    pairs = (
        ("products_1", "products_2", "prod1_to_prod2"),
        ("products_1", "products_3", "prod1_to_prod3"),
        ("products_1", "products_4", "prod1_to_prod4"),
    )
    splits = ("train", "val", "test", "all")
    canonical_em = CANONICAL_PREFIX / "products" / "input" / "entity_matching_gt"
    synth_em = SYNTHETIC_PREFIX / "usecases" / "products" / "input" / "entitymatching"
    results = []
    for src1, src2, canon_stem in pairs:
        for split in splits:
            canon_path = canonical_em / f"{canon_stem}_{split}.csv"
            synth_path = synth_em / f"{src1}_2_{src2}_{split}.csv"
            if not canon_path.exists() or not synth_path.exists():
                results.append(
                    {
                        "pair": (src1, src2),
                        "split": split,
                        "status": "MISSING",
                        "canon_exists": canon_path.exists(),
                        "synth_exists": synth_path.exists(),
                    }
                )
                continue
            canon = _read_em_csv(canon_path)
            synth = _read_em_csv(synth_path)
            # Synth has no header (3 columns positional); canonical has
            # id1,id2,label header. Re-read synth without header.
            synth = pd.read_csv(synth_path, header=None, names=["id1", "id2", "label"])
            row_match = len(canon) == len(synth)
            # Translate canonical IDs to synth's prefix scheme.
            canon_translated_id1 = [f"{src1}_{x}" for x in canon["id1"].astype(str)]
            canon_translated_id2 = [f"{src2}_{x}" for x in canon["id2"].astype(str)]
            id1_match = canon_translated_id1 == synth["id1"].astype(str).tolist()
            id2_match = canon_translated_id2 == synth["id2"].astype(str).tolist()
            # Translate canonical labels (0/1) to synth's bool (false/true).
            canon_translated_label = [
                "true" if int(v) == 1 else "false" for v in canon["label"]
            ]
            label_match = (
                canon_translated_label
                == synth["label"].astype(str).str.lower().tolist()
            )
            status = (
                "EQUIVALENT"
                if (row_match and id1_match and id2_match and label_match)
                else "DIVERGENT"
            )
            results.append(
                {
                    "pair": (src1, src2),
                    "split": split,
                    "status": status,
                    "canon_rows": int(len(canon)),
                    "synth_rows": int(len(synth)),
                    "id1_translation_match": id1_match,
                    "id2_translation_match": id2_match,
                    "label_translation_match": label_match,
                }
            )
    return {"per_pair": results}


def main() -> int:
    print("=" * 78)
    print("Best-of-breed bundle data-source audit")
    print(f"REPO_ROOT = {REPO_ROOT}")
    print("=" * 78)
    print()

    rows = []
    for domain in DOMAINS:
        info = _audit_domain(domain)
        rows.append(info)

    # Pretty per-domain print
    for info in rows:
        cls = info["classification"]
        flag = "✓" if cls == "CANONICAL" else ("✗" if cls == "SYNTHETIC" else "?")
        print(f"--- {info['domain']} ---")
        print(f"  variant_root    : {info['variant_root']}")
        print(f"  classification  : {cls}  {flag}")
        print(f"  n_sources       : {info['n_sources']}")
        print(f"  n_em_pairs      : {info['n_em_pairs']}")
        if info["em_summary"]:
            em = info["em_summary"]
            print(f"  first EM pair   : {em['first_pair']}")
            print(f"  EM gold rows    : {em['rows']}")
            print(f"  EM id1 sample   : {em['id1_sample']}")
            print(f"  EM id2 sample   : {em['id2_sample']}")
            print(f"  EM label sample : {em['label_sample']}")
        print(f"  SM gold present : {info['has_sm_gold']}")
        print()

    # Summary
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    canonical = [r for r in rows if r["classification"] == "CANONICAL"]
    synth = [r for r in rows if r["classification"] == "SYNTHETIC"]
    print(
        f"CANONICAL (reads from usecases/<domain>/): "
        f"{', '.join(r['domain'] for r in canonical) or '(none)'}"
    )
    print(
        f"SYNTHETIC (reads from usecases_synthetic/usecases/<domain>/): "
        f"{', '.join(r['domain'] for r in synth) or '(none)'}"
    )
    print()

    # Products deep equivalence check
    if any(
        r["domain"] == "products" and r["classification"] == "SYNTHETIC" for r in rows
    ):
        print("=" * 78)
        print("PRODUCTS deep equivalence: synthetic-translated EM gold vs canonical")
        print("=" * 78)
        report = _verify_products_canonical_equivalence()
        eq_count = sum(1 for r in report["per_pair"] if r["status"] == "EQUIVALENT")
        div_count = sum(1 for r in report["per_pair"] if r["status"] == "DIVERGENT")
        missing = sum(1 for r in report["per_pair"] if r["status"] == "MISSING")
        for r in report["per_pair"]:
            pair = "_2_".join(r["pair"])
            if r["status"] == "EQUIVALENT":
                print(
                    f"  ✓ {pair} {r['split']:>5}: "
                    f"rows={r['canon_rows']} (canonical == synth translation)"
                )
            elif r["status"] == "DIVERGENT":
                print(
                    f"  ✗ {pair} {r['split']:>5}: "
                    f"canon_rows={r.get('canon_rows', '?')} "
                    f"synth_rows={r.get('synth_rows', '?')} "
                    f"id1_match={r.get('id1_translation_match')} "
                    f"id2_match={r.get('id2_translation_match')} "
                    f"label_match={r.get('label_translation_match')}"
                )
            else:
                print(
                    f"  ? {pair} {r['split']:>5}: MISSING "
                    f"canon_exists={r['canon_exists']} "
                    f"synth_exists={r['synth_exists']}"
                )
        print()
        print(f"  {eq_count} equivalent, {div_count} divergent, {missing} missing")
        print()
        if div_count == 0 and missing == 0:
            print(
                "  Conclusion: products' synthetic-side EM gold is an EXACT "
                "translation of canonical"
            )
            print(
                "  (ID prefix + label-format normalization only). No "
                "semantic divergence."
            )

    return 0 if all(r["classification"] == "CANONICAL" for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
