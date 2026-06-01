#!/usr/bin/env python3
"""R2.1 helper (PyDI-data variant): build Ditto json.gz files for music.

Per pair:

* musicbrainz_discogs
    train = PyDI musicbrainz_2_discogs_train.csv (~19k pairs, ~8% pos)
    val   = PyDI musicbrainz_2_discogs_val.csv   (~9k pairs)
    test  = PyDI musicbrainz_2_discogs_test.csv  (1,000 pairs)

* musicbrainz_lastfm
    train = PyDI musicbrainz_2_lastfm_train.csv  (~17k pairs, ~10% pos)
    val   = PyDI musicbrainz_2_lastfm_val.csv    (~8k pairs)
    test  = PyDI musicbrainz_2_lastfm_test.csv   (1,000 pairs)

Music uses **option (b) from the R2.2 redo: pure PyDI throughout** —
PyDI's gold for music is 15× larger than ADI's training pool (~36k vs
2.4k pairs), so the ADI-train approach (used for companies/games) is
not the best lever here. PyDI's natural train/val/test splits are
clean and disjoint.

Side-alignment: all PyDI files use the `(musicbrainz, X)` ordering, so
no id1↔id2 swap is needed — the canonical (src_left, src_right) order
is `(musicbrainz, discogs)` and `(musicbrainz, lastfm)` matching the
PyDI filename convention.

Field projection (R10-I): the **wide committee scope**
(``committee_ditto_fields("music")`` == ``DOMAIN_TEXT_COLS["music"]`` ==
``em_matching_committee_music.yaml`` ``ditto_plm.fields`` — `[name, artist,
release-date, release-country, duration, label, genre, tracks]`),
column-mapped off the base PyDI sources the way the committee runner maps
them at inference (music uses an identity column_mapping). Here `label` is
the music *record label* attribute, distinct from the binary class column
(records carry `label_left`/`label_right` plus the integer `label` class
field — no collision). lastfm is sparse on several attributes; empty
fields are dropped from the serialization by Ditto's serializer.

Leak removal: PyDI test pairs dropped from train and val (cross-source
overlap detected via frozenset key on (id1, id2)). Intra-train
duplicates deduped.

Outputs (all under ``usecases_synthetic/output/ditto/music/``):
    train.json.gz
    val.json.gz
    test.json.gz
    train_sample_weights.csv

Run from the repo root:

    python usecases_synthetic/scripts/ditto/_prep_music.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from usecases_synthetic.lib.domain_config import SYNTHETIC_DIR, USECASES_DIR
from usecases_synthetic.lib.loaders import load_domain_sources, read_em_gold_csv
from usecases_synthetic.scripts.ditto.prepare_em_training_data import (
    build_ditto_pair_records_committee_scope,
    committee_ditto_fields,
    write_committee_fields_sidecar,
    write_json_gz,
)

DOMAIN = "music"
PYDI_EM_GOLD_DIR = USECASES_DIR / DOMAIN / "input" / "entitymatching"
OUTPUT_DIR = SYNTHETIC_DIR / "output" / "ditto" / DOMAIN


@dataclass(frozen=True)
class Pair:
    name: str
    src_left: str
    src_right: str
    pydi_stem: str  # PyDI gold filename stem (e.g. "musicbrainz_2_discogs")


PAIRS: list[Pair] = [
    Pair(
        name="musicbrainz_discogs",
        src_left="musicbrainz",
        src_right="discogs",
        pydi_stem="musicbrainz_2_discogs",
    ),
    Pair(
        name="musicbrainz_lastfm",
        src_left="musicbrainz",
        src_right="lastfm",
        pydi_stem="musicbrainz_2_lastfm",
    ),
]


def _normalize_label_to_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    s = str(value).strip().upper()
    if s in {"1", "TRUE", "T", "YES", "Y"}:
        return 1
    if s in {"0", "FALSE", "F", "NO", "N"}:
        return 0
    raise ValueError(f"unexpected label value: {value!r}")


def _load_pydi_split(stem: str, split: str) -> pd.DataFrame:
    df = read_em_gold_csv(PYDI_EM_GOLD_DIR / f"{stem}_{split}.csv")
    df = df[["id1", "id2", "label"]].copy()
    df["id1"] = df["id1"].astype(str)
    df["id2"] = df["id2"].astype(str)
    df["label"] = df["label"].map(_normalize_label_to_int).astype(int)
    return df


def _pair_keys(df: pd.DataFrame) -> set[frozenset[str]]:
    return set(frozenset((str(a), str(b))) for a, b in zip(df["id1"], df["id2"]))


def _drop_pairs_in(df: pd.DataFrame, leak_keys: set[frozenset[str]]) -> pd.DataFrame:
    if not leak_keys:
        return df
    keep_mask = [
        frozenset((str(a), str(b))) not in leak_keys
        for a, b in zip(df["id1"], df["id2"])
    ]
    return df.loc[keep_mask].reset_index(drop=True)


def _build_records(
    gold: pd.DataFrame, pair: Pair, sources: dict[str, pd.DataFrame]
) -> list[dict]:
    return build_ditto_pair_records_committee_scope(
        gold, DOMAIN, pair.src_left, pair.src_right, sources=sources
    )


def _balanced_class_weights(records: list[dict]) -> dict[int, float]:
    n = len(records)
    n_pos = sum(1 for r in records if r["label"] == 1)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        raise RuntimeError(f"degenerate class balance: pos={n_pos} neg={n_neg}")
    return {1: n / (2.0 * n_pos), 0: n / (2.0 * n_neg)}


def _dedupe_by_pair(records: list[dict], split_label: str) -> list[dict]:
    seen: dict[frozenset[str], dict] = {}
    label_conflicts = 0
    for r in records:
        key = frozenset((str(r["id_left"]), str(r["id_right"])))
        prev = seen.get(key)
        if prev is None:
            seen[key] = r
            continue
        if int(prev["label"]) != int(r["label"]):
            label_conflicts += 1
    deduped = list(seen.values())
    n_drop = len(records) - len(deduped)
    if n_drop:
        msg = f"{split_label}: deduped {n_drop} duplicate pair records"
        if label_conflicts:
            msg += f" ({label_conflicts} conflicting-label collisions — kept first)"
        print(msg)
    return deduped


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # R10-I: wide committee field scope, column-mapped off the base PyDI
    # sources the way the committee runner maps them at inference (music uses
    # an identity column_mapping). Committee baseline Ditto trains on base.
    sources = load_domain_sources(DOMAIN)
    fields = committee_ditto_fields(DOMAIN)
    print(f"Field scope ({len(fields)}): {', '.join(fields)}")

    train_records: list[dict] = []
    val_records: list[dict] = []
    test_records: list[dict] = []

    print("Loading per-pair sets, computing leaks, building records...")
    for pair in PAIRS:
        train_df = _load_pydi_split(pair.pydi_stem, "train")
        val_df = _load_pydi_split(pair.pydi_stem, "val")
        test_df = _load_pydi_split(pair.pydi_stem, "test")
        test_keys = _pair_keys(test_df)

        n_tr = len(train_df)
        n_vl = len(val_df)
        train_df = _drop_pairs_in(train_df, test_keys)
        val_df = _drop_pairs_in(val_df, test_keys)
        train_dropped = n_tr - len(train_df)
        val_dropped = n_vl - len(val_df)

        train_records.extend(_build_records(train_df, pair, sources))
        val_records.extend(_build_records(val_df, pair, sources))
        test_records.extend(_build_records(test_df, pair, sources))

        print(
            f"  {pair.name:<22}  "
            f"train(PyDI) {len(train_df)} (dropped {train_dropped} leak)  "
            f"val(PyDI) {len(val_df)} (dropped {val_dropped} leak)  "
            f"test(PyDI) {len(test_df)}"
        )

    train_records = _dedupe_by_pair(train_records, "train")

    for split, records in [
        ("train", train_records),
        ("val", val_records),
        ("test", test_records),
    ]:
        out_path = OUTPUT_DIR / f"{split}.json.gz"
        write_json_gz(records, out_path)
        n_pos = sum(1 for r in records if r["label"] == 1)
        n_neg = len(records) - n_pos
        pct = (100.0 * n_pos / len(records)) if records else 0.0
        print(
            f"{split}: {len(records)} pairs ({n_pos} pos / {n_neg} neg, "
            f"{pct:.1f}% pos) -> {out_path}"
        )

    weights = _balanced_class_weights(train_records)
    rows = [
        {"pair_id": r["pair_id"], "sample_weight": weights[r["label"]]}
        for r in train_records
    ]
    sw_path = OUTPUT_DIR / "train_sample_weights.csv"
    pd.DataFrame(rows).to_csv(sw_path, index=False)
    print(
        f"sample weights (sklearn-balanced): pos={weights[1]:.4f}  "
        f"neg={weights[0]:.4f} -> {sw_path}"
    )

    fields_path = write_committee_fields_sidecar(OUTPUT_DIR, DOMAIN)
    print(f"field scope -> {fields_path}")
    print(
        "Train the baseline committee Ditto on this wide scope with:\n"
        f"  python usecases_synthetic/scripts/ditto/train.py --domain {DOMAIN} "
        f"--train-json-gz {OUTPUT_DIR / 'train.json.gz'} "
        f"--val-json-gz {OUTPUT_DIR / 'val.json.gz'} --config <recipe.yaml>"
    )


if __name__ == "__main__":
    main()
