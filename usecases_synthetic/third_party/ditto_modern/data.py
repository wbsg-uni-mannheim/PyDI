from __future__ import annotations

import gzip
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd
import torch
from torch.utils.data import Dataset

from .augment import Augmenter

# Legacy WDC schema. Kept for backward compatibility with older datasets/tests.
WDC_COLUMNS = [
    "id_left",
    "brand_left",
    "title_left",
    "description_left",
    "price_left",
    "priceCurrency_left",
    "cluster_id_left",
    "id_right",
    "brand_right",
    "title_right",
    "description_right",
    "price_right",
    "priceCurrency_right",
    "cluster_id_right",
    "pair_id",
    "label",
    "is_hard_negative",
]

REQUIRED_PAIR_COLUMNS = ["id_left", "id_right", "pair_id", "label", "is_hard_negative"]
RESERVED_SERIALIZATION_FIELDS = {
    "id",
    "__rid",
    "pair_id",
    "label",
    "is_hard_negative",
    "rid1",
    "rid2",
    "similarity",
}


@dataclass(frozen=True)
class PairExample:
    idx: int
    pair_id: str
    left: str
    right: str
    label: int


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    text = re.sub(r"\s+", " ", text.replace("\n", " ").replace("\t", " ")).strip()
    return text


def _normalize_binary_label(value: object) -> int:
    if value is None:
        raise ValueError("label value is missing")
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        iv = int(value)
        if iv in {0, 1}:
            return iv
        raise ValueError(f"label column must contain 0/1 values, got {value!r}")
    s = str(value).strip().upper()
    if s in {"1", "TRUE", "T", "YES", "Y"}:
        return 1
    if s in {"0", "FALSE", "F", "NO", "N"}:
        return 0
    raise ValueError(f"label column must contain 0/1 values, got {value!r}")


def serialize_entity(
    record: Dict[str, object], side: str, fields: Sequence[str], max_field_len: int
) -> str:
    parts: List[str] = []
    for field in fields:
        key = f"{field}_{side}"
        value = normalize_text(record.get(key, ""))
        if not value:
            continue
        if len(value) > max_field_len:
            value = value[:max_field_len] + "..."
        parts.append(f"COL {field} VAL {value}")
    return " ".join(parts)


def load_wdc_json_gz(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    with gzip.open(p, "rt") as f:
        records = [json.loads(line) for line in f]
    df = pd.DataFrame(records)
    if "label" not in df.columns:
        raise ValueError("Missing required label column")
    if "pair_id" not in df.columns:
        df["pair_id"] = [f"idx-{i}" for i in range(len(df))]
    if "id_left" not in df.columns:
        df["id_left"] = ""
    if "id_right" not in df.columns:
        df["id_right"] = ""
    if "is_hard_negative" not in df.columns:
        df["is_hard_negative"] = 0

    df["label"] = df["label"].map(_normalize_binary_label).astype(int)
    return df


def write_wdc_json_gz(df: pd.DataFrame, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt") as f:
        for _, row in df.iterrows():
            record = {c: row[c] for c in df.columns}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def wdc_to_pair_examples(
    df: pd.DataFrame,
    fields: Sequence[str],
    max_field_len: int,
) -> List[PairExample]:
    bad_fields = [f for f in fields if str(f).strip() in RESERVED_SERIALIZATION_FIELDS]
    if bad_fields:
        raise ValueError(
            f"Reserved metadata fields cannot be used for Ditto serialization: {sorted(set(bad_fields))}"
        )

    missing_fields: List[str] = []
    for field in fields:
        for side in ("left", "right"):
            col = f"{field}_{side}"
            if col not in df.columns:
                missing_fields.append(col)
    if missing_fields:
        raise ValueError(
            f"Missing required field columns for serialization: {sorted(set(missing_fields))}"
        )

    rows: List[PairExample] = []
    for idx, row in enumerate(df.to_dict(orient="records")):
        pair_id = normalize_text(row.get("pair_id", f"idx-{idx}")) or f"idx-{idx}"
        left = serialize_entity(row, "left", fields, max_field_len)
        right = serialize_entity(row, "right", fields, max_field_len)
        rows.append(
            PairExample(
                idx=idx,
                pair_id=pair_id,
                left=left,
                right=right,
                label=int(row["label"]),
            )
        )
    return rows


def examples_to_ditto_lines(examples: Iterable[PairExample]) -> List[str]:
    return [f"{ex.left}\t{ex.right}\t{ex.label}" for ex in examples]


class PairDataset(Dataset):
    """
    Ditto-style pair dataset with optional MixDA augmentation.
    Returns variable-length token-id sequences and uses `pad` as collate_fn.
    """

    def __init__(
        self,
        examples: Sequence[PairExample],
        tokenizer,
        max_len: int,
        weights: Dict[int, float] | None = None,
        da: str | None = None,
    ):
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.weights = weights or {}
        self.da = da
        self.augmenter = Augmenter() if da else None

    def __len__(self) -> int:
        return len(self.examples)

    def _encode_pair(self, left: str, right: str) -> List[int]:
        return self.tokenizer.encode(
            text=left,
            text_pair=right,
            max_length=self.max_len,
            truncation=True,
        )

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        x = self._encode_pair(ex.left, ex.right)
        sample_weight = float(self.weights.get(ex.idx, 1.0))

        if self.augmenter is not None:
            combined = f"{ex.left} [SEP] {ex.right}"
            aug = self.augmenter.augment_sent(combined, op=self.da)
            if " [SEP] " in aug:
                left_aug, right_aug = aug.split(" [SEP] ", 1)
            else:
                # Fallback to original pair if op produced invalid split
                left_aug, right_aug = ex.left, ex.right
            x_aug = self._encode_pair(left_aug, right_aug)
            return x, x_aug, ex.label, ex.idx, sample_weight

        return x, ex.label, ex.idx, sample_weight

    def pad(self, batch):
        """Collate variable-length token id sequences into padded tensors.

        R7 fix (2026-05-27): use the tokenizer's actual ``pad_token_id``
        for padding and compute ``attention_mask = (tok != pad_id)``.
        Pre-R7 the function hard-coded ``0`` as the pad value, which
        for RoBERTa is the CLS token id (real pad_token_id = 1). The
        attention_mask computation ``1 if tok != 0`` then incorrectly
        masked the CLS token's attention to 0, so the model learned
        to classify with CLS attention masked out. Inference under HF
        defaults (which pad correctly with id=1 and give CLS
        attention=1) produced a different attention pattern → score
        distribution shifted. See plan_revision.md R7 + R6-4.

        Bound-instance method (not @staticmethod) so it can read
        ``self.tokenizer.pad_token_id``. Callers pass ``dataset.pad``
        as the DataLoader's ``collate_fn``.
        """
        pad_id = int(self.tokenizer.pad_token_id)

        def _attn(seq: List[int]) -> List[int]:
            # attention_mask = 1 wherever the token is real (not pad).
            return [1 if tok != pad_id else 0 for tok in seq]

        if len(batch[0]) == 5:
            x1, x2, y, idxs, weights = zip(*batch)
            maxlen = max(max(len(x) for x in x1), max(len(x) for x in x2))
            x1 = [xi + [pad_id] * (maxlen - len(xi)) for xi in x1]
            x2 = [xi + [pad_id] * (maxlen - len(xi)) for xi in x2]
            att1 = [_attn(xi) for xi in x1]
            att2 = [_attn(xi) for xi in x2]
            return (
                torch.LongTensor(x1),
                torch.LongTensor(att1),
                torch.LongTensor(x2),
                torch.LongTensor(att2),
                torch.LongTensor(y),
                torch.LongTensor(idxs),
                torch.FloatTensor(weights),
            )

        x, y, idxs, weights = zip(*batch)
        maxlen = max(len(xi) for xi in x)
        x = [xi + [pad_id] * (maxlen - len(xi)) for xi in x]
        att = [_attn(xi) for xi in x]
        return (
            torch.LongTensor(x),
            torch.LongTensor(att),
            torch.LongTensor(y),
            torch.LongTensor(idxs),
            torch.FloatTensor(weights),
        )
