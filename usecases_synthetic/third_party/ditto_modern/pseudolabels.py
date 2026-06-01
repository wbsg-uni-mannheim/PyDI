from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from .data import REQUIRED_PAIR_COLUMNS, load_wdc_json_gz, write_wdc_json_gz


REQUIRED_PRED_COLS = {"pair_id", "pred_precision", "pred_balanced", "pred_recall", "pred_vote"}


def _resolve_output_columns(df: pd.DataFrame) -> list[str]:
    pair_meta = [c for c in REQUIRED_PAIR_COLUMNS if c in df.columns]
    dynamic_fields = [c for c in df.columns if c.endswith("_left") or c.endswith("_right")]
    out: list[str] = []
    for c in pair_meta + dynamic_fields:
        if c not in out:
            out.append(c)
    return out


def agreement(row: pd.Series) -> Tuple[int, int]:
    votes = [int(row["pred_precision"]), int(row["pred_balanced"]), int(row["pred_recall"])]
    yes = sum(votes)
    return max(yes, 3 - yes), yes


def build_pseudolabels(
    source_json_gz: str,
    multi_agent_run_dir: str,
    output_json_gz: str,
    policy: str = "consensus",
    include_majority_with_weight: bool = False,
    majority_weight: float = 0.5,
) -> Dict[str, object]:
    source_df = load_wdc_json_gz(source_json_gz)

    pred_path = Path(multi_agent_run_dir) / "predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {pred_path}")
    pred_df = pd.read_csv(pred_path)

    missing = REQUIRED_PRED_COLS - set(pred_df.columns)
    if missing:
        raise ValueError(f"Predictions file missing required columns: {sorted(missing)}")

    merged = source_df.merge(pred_df, on="pair_id", how="left", validate="one_to_one")
    if merged[["pred_precision", "pred_balanced", "pred_recall", "pred_vote"]].isna().any().any():
        n_missing = int(
            merged[["pred_precision", "pred_balanced", "pred_recall", "pred_vote"]]
            .isna()
            .any(axis=1)
            .sum()
        )
        raise ValueError(f"Missing predictions for {n_missing} source rows (pair_id mismatch)")

    agree_cols = merged.apply(agreement, axis=1, result_type="expand")
    merged["agreement_count"] = agree_cols[0].astype(int)

    if policy not in {"consensus", "majority"}:
        raise ValueError("policy must be one of: consensus, majority")

    if policy == "consensus":
        consensus = merged[merged["agreement_count"] == 3].copy()
        majority_only = merged[merged["agreement_count"] == 2].copy()
        if include_majority_with_weight:
            kept = pd.concat([consensus, majority_only], ignore_index=True)
            kept["sample_weight"] = kept["agreement_count"].map({3: 1.0, 2: float(majority_weight)}).astype(float)
        else:
            kept = consensus
            kept["sample_weight"] = 1.0
    else:
        kept = merged.copy()
        kept["sample_weight"] = 1.0

    kept["label"] = kept["pred_vote"].astype(int)
    out_cols = _resolve_output_columns(kept)
    out_df = kept[out_cols].copy()
    write_wdc_json_gz(out_df, output_json_gz)

    out_path = Path(output_json_gz)
    stem = out_path.name
    if stem.endswith(".json.gz"):
        stem = stem[:-8]

    report_path = out_path.parent / f"{stem}_filter_report.json"
    weights_path = out_path.parent / f"{stem}_sample_weights.csv"

    kept[["pair_id", "sample_weight", "agreement_count", "pred_vote"]].to_csv(weights_path, index=False)

    report = {
        "source_json_gz": str(source_json_gz),
        "multi_agent_run_dir": str(multi_agent_run_dir),
        "output_json_gz": str(output_json_gz),
        "policy": policy,
        "include_majority_with_weight": bool(include_majority_with_weight),
        "majority_weight": float(majority_weight),
        "rows_total": int(len(merged)),
        "rows_kept": int(len(out_df)),
        "rows_dropped": int(len(merged) - len(out_df)),
        "agreement": {
            "consensus_3of3": int((merged["agreement_count"] == 3).sum()),
            "majority_2of3": int((merged["agreement_count"] == 2).sum()),
        },
        "class_balance": {
            "label_1": int((out_df["label"] == 1).sum()),
            "label_0": int((out_df["label"] == 0).sum()),
        },
        "sample_weights_csv": str(weights_path),
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    return report
