#!/usr/bin/env python3
"""D5 helper: build tiny disjoint train/val/test Ditto json.gz files for companies.

Samples balanced slices from ``forbes_2_dbpedia_all.csv`` (32+32 / 8+8 / 8+8
positive/negative pairs) and writes them through
``build_ditto_pair_records_from_gold`` so the D5 smoke test has real
pair-text inputs without polluting the checked-in EM gold directory.

Run from the repo root:

    python usecases_synthetic/scripts/ditto/_prep_tiny_companies.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from usecases_synthetic.lib.domain_config import USECASES_DIR, SYNTHETIC_DIR
from usecases_synthetic.lib.loaders import read_em_gold_csv
from usecases_synthetic.scripts.ditto.prepare_em_training_data import (
    build_ditto_pair_records_from_gold,
    write_json_gz,
)

DOMAIN = "companies"
PAIR = "forbes_2_dbpedia"
SRC1, SRC2 = "forbes", "dbpedia"
GOLD_PATH = USECASES_DIR / DOMAIN / "input" / "entitymatching" / f"{PAIR}_all.csv"
OUTPUT_DIR = SYNTHETIC_DIR / "output" / "ditto" / "trial_companies"

SPLIT_SIZES = {
    "tiny_train": (32, 32),  # (pos, neg)
    "tiny_val": (8, 8),
    "tiny_test": (8, 8),
}
SEED = 42


def main() -> None:
    gold = read_em_gold_csv(GOLD_PATH)
    pos = gold[gold["label"].str.strip().str.upper().isin({"TRUE", "1", "T", "YES"})]
    neg = gold[gold["label"].str.strip().str.upper().isin({"FALSE", "0", "F", "NO"})]
    total_pos = sum(p for p, _ in SPLIT_SIZES.values())
    total_neg = sum(n for _, n in SPLIT_SIZES.values())
    if len(pos) < total_pos or len(neg) < total_neg:
        raise RuntimeError(
            f"not enough gold pairs: have {len(pos)} pos / {len(neg)} neg, "
            f"need {total_pos} / {total_neg}"
        )

    pos_shuffled = pos.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    neg_shuffled = neg.sample(frac=1.0, random_state=SEED + 1).reset_index(drop=True)

    pos_cursor = 0
    neg_cursor = 0
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split_name, (n_pos, n_neg) in SPLIT_SIZES.items():
        slice_pos = pos_shuffled.iloc[pos_cursor : pos_cursor + n_pos]
        slice_neg = neg_shuffled.iloc[neg_cursor : neg_cursor + n_neg]
        pos_cursor += n_pos
        neg_cursor += n_neg
        gold_slice = pd.concat([slice_pos, slice_neg], ignore_index=True)
        gold_slice = gold_slice.sample(frac=1.0, random_state=SEED + 7).reset_index(
            drop=True
        )

        records = build_ditto_pair_records_from_gold(gold_slice, DOMAIN, SRC1, SRC2)
        out_path = OUTPUT_DIR / f"{split_name}.json.gz"
        write_json_gz(records, out_path)
        got_pos = sum(1 for r in records if r["label"] == 1)
        got_neg = len(records) - got_pos
        print(
            f"{split_name}: {len(records)} pairs ({got_pos} pos / {got_neg} neg) -> {out_path}"
        )


if __name__ == "__main__":
    main()
