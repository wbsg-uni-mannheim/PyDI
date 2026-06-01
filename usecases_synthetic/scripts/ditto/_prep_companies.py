#!/usr/bin/env python3
"""Build Ditto json.gz files for companies.

Two training-data sources (selected via ``--train-source``):

* ``--train-source pydi`` **(default, committee-correct per
  plan_revision.md R6-3)**: train/val/test all come from
  ``usecases/companies/input/entitymatching/<pydi_pair>_<split>.csv``.
  The committee Ditto checkpoint must train on the same gold
  distribution the committee evaluates against — ADI's labeled pool is
  out-of-distribution and is reserved for the pool-builder Ditto only.
  Output: ``usecases_synthetic/output/ditto/companies_pydi/``.

* ``--train-source adi`` **(legacy / pool-builder path)**: train from
  ``automatic-data-integration/scripts/output/companies_0302/entity_resolution/training/training_<adi_pair>_latest.csv``,
  val from ``.../validation/similarity_validation_faiss_<adi_pair>.csv``,
  test from PyDI. This is the R2.2 setup used by the pool builder
  (which lives in ``lib/pool_builder.py`` + ``scripts/build_pool.py``);
  it must NOT be wired to the committee per R6-3.
  Output: ``usecases_synthetic/output/ditto/companies/``.

**Leak removal**: pairs appearing in train ∩ PyDI-test or
val ∩ PyDI-test are dropped from train/val.

**Side-alignment**: every pair is normalised to a canonical
``(src_left, src_right)`` ordering — PyDI gold for ``forbes_2_dbpedia``
has id1=forbes,id2=dbpedia, so the loader swaps to match the declared
``(dbpedia, forbes)`` direction.

Outputs:
    train.json.gz / val.json.gz / test.json.gz / train_sample_weights.csv

Run from the repo root:

    # Committee-correct training (default)
    python usecases_synthetic/scripts/ditto/_prep_companies.py

    # Legacy ADI training (pool builder)
    python usecases_synthetic/scripts/ditto/_prep_companies.py --train-source adi
"""

from __future__ import annotations

import argparse
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
    build_ditto_pair_records_from_gold,
    committee_ditto_fields,
    write_committee_fields_sidecar,
    write_json_gz,
)

DOMAIN = "companies"
ADI_DIR = (
    ROOT
    / "automatic-data-integration"
    / "scripts"
    / "output"
    / "companies_0302"
    / "entity_resolution"
)
PYDI_EM_GOLD_DIR = USECASES_DIR / DOMAIN / "input" / "entitymatching"
OUTPUT_DIR_ADI = SYNTHETIC_DIR / "output" / "ditto" / DOMAIN
OUTPUT_DIR_PYDI = SYNTHETIC_DIR / "output" / "ditto" / f"{DOMAIN}_pydi"


@dataclass(frozen=True)
class Pair:
    name: str  # display name used in pair_id and output dir
    src_left: str  # canonical left source for serialization
    src_right: str  # canonical right source
    adi_pair: str  # ADI's pair-name in file paths (e.g. "dbpedia_forbes")
    pydi_pair: str | None  # PyDI gold pair name (None means no PyDI test)
    pydi_left_is_left: bool  # True iff PyDI gold's id1 already == src_left


PAIRS: list[Pair] = [
    Pair(
        name="dbpedia_forbes",
        src_left="dbpedia",
        src_right="forbes",
        adi_pair="dbpedia_forbes",
        pydi_pair="forbes_2_dbpedia",
        pydi_left_is_left=False,  # PyDI id1=forbes, id2=dbpedia → must swap
    ),
    Pair(
        name="forbes_fullcontact",
        src_left="forbes",
        src_right="fullcontact",
        adi_pair="forbes_fullcontact",
        pydi_pair="forbes_2_fullcontact",
        pydi_left_is_left=True,
    ),
    Pair(
        name="dbpedia_fullcontact",
        src_left="dbpedia",
        src_right="fullcontact",
        adi_pair="dbpedia_fullcontact",
        pydi_pair=None,  # no PyDI gold for this pair
        pydi_left_is_left=False,
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


def _load_adi_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not {"id1", "id2", "label"}.issubset(df.columns):
        raise ValueError(f"{path} missing id1/id2/label columns: {list(df.columns)}")
    df = df[["id1", "id2", "label"]].copy()
    df["id1"] = df["id1"].astype(str)
    df["id2"] = df["id2"].astype(str)
    df["label"] = df["label"].map(_normalize_label_to_int).astype(int)
    return df


def _load_pydi_split(pydi_pair: str, split: str, swap_ids: bool) -> pd.DataFrame:
    """Load a PyDI gold split (train/val/test) for one source pair.

    Returns frame with id1/id2/label, optionally swapping id1<->id2 to
    match the canonical (src_left, src_right) direction. Companies has
    on-disk train + val + test for the two PyDI-gold pairs.
    """
    df = read_em_gold_csv(PYDI_EM_GOLD_DIR / f"{pydi_pair}_{split}.csv")
    df = df[["id1", "id2", "label"]].copy()
    df["id1"] = df["id1"].astype(str)
    df["id2"] = df["id2"].astype(str)
    df["label"] = df["label"].map(_normalize_label_to_int).astype(int)
    if swap_ids:
        df = df.rename(columns={"id1": "id2", "id2": "id1"})[["id1", "id2", "label"]]
    return df


def _load_pydi_test(pydi_pair: str, swap_ids: bool) -> pd.DataFrame:
    """Backwards-compatible wrapper around ``_load_pydi_split(..., 'test')``."""
    return _load_pydi_split(pydi_pair, "test", swap_ids)


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
    gold: pd.DataFrame,
    pair: Pair,
    *,
    use_committee_scope: bool,
    sources: dict[str, pd.DataFrame],
) -> list[dict]:
    """Build pair records for one split.

    ``use_committee_scope`` (the committee-correct ``--train-source pydi``
    path) builds the *wide* committee field scope, column-mapped off the
    base PyDI sources exactly the way the committee runner maps them at
    inference. The legacy ``--train-source adi`` path keeps the narrow
    knob-02 ``canonical_schema`` projection — it feeds only the
    pool-builder Ditto, never the committee, and its frozen checkpoint was
    trained on that narrow surface.
    """
    if use_committee_scope:
        return build_ditto_pair_records_committee_scope(
            gold, DOMAIN, pair.src_left, pair.src_right, sources=sources
        )
    return build_ditto_pair_records_from_gold(
        gold, DOMAIN, pair.src_left, pair.src_right
    )


def _balanced_class_weights(records: list[dict]) -> dict[int, float]:
    n = len(records)
    n_pos = sum(1 for r in records if r["label"] == 1)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        raise RuntimeError(f"degenerate class balance: pos={n_pos} neg={n_neg}")
    return {1: n / (2.0 * n_pos), 0: n / (2.0 * n_neg)}


def _dedupe_by_pair(records: list[dict], split_label: str) -> list[dict]:
    """Drop duplicate (id_left, id_right) pairs, keeping first occurrence.

    Logs any conflicting-label collisions (same pair appearing with both
    labels) — those indicate an upstream gold inconsistency.
    """
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
            msg += (
                f" (with {label_conflicts} conflicting-label collisions — kept first)"
            )
        print(msg)
    return deduped


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare Ditto train/val/test json.gz files for companies. "
            "Default --train-source pydi is committee-correct per "
            "plan_revision.md R6-3 (committee Ditto must train on the "
            "same gold distribution it evaluates against). --train-source "
            "adi is the legacy R2.2 setup reserved for the pool builder."
        )
    )
    parser.add_argument(
        "--train-source",
        choices=["pydi", "adi"],
        default="pydi",
        help=(
            "Where train + val records come from. 'pydi' loads "
            "<pydi_pair>_{train,val}.csv from usecases/companies/input/"
            "entitymatching/ (committee-correct, default). 'adi' loads "
            "from automatic-data-integration/scripts/output/companies_0302/ "
            "(legacy / pool-builder only)."
        ),
    )
    args = parser.parse_args()
    train_source = args.train_source

    output_dir = OUTPUT_DIR_PYDI if train_source == "pydi" else OUTPUT_DIR_ADI
    output_dir.mkdir(parents=True, exist_ok=True)

    # R10-I: the committee (pydi) path builds the wide committee field scope
    # column-mapped off the base sources; the legacy ADI path stays narrow.
    use_committee_scope = train_source == "pydi"
    sources: dict[str, pd.DataFrame] = (
        load_domain_sources(DOMAIN) if use_committee_scope else {}
    )
    if use_committee_scope:
        fields = committee_ditto_fields(DOMAIN)
        print(f"Field scope ({len(fields)}): {', '.join(fields)}")

    train_records: list[dict] = []
    val_records: list[dict] = []
    test_records: list[dict] = []

    src_tag = train_source.upper()
    print(
        f"Loading per-pair sets, computing leaks, building records " f"({src_tag})..."
    )
    for pair in PAIRS:
        # Test: PyDI only (when available).
        if pair.pydi_pair is not None:
            pydi_test = _load_pydi_split(
                pair.pydi_pair, "test", swap_ids=not pair.pydi_left_is_left
            )
            test_keys = _pair_keys(pydi_test)
        else:
            pydi_test = None
            test_keys = set()

        train_df: pd.DataFrame | None = None
        val_df: pd.DataFrame | None = None
        if train_source == "adi":
            train_df = _load_adi_csv(
                ADI_DIR / "training" / f"training_{pair.adi_pair}_latest.csv"
            )
            val_df = _load_adi_csv(
                ADI_DIR
                / "validation"
                / f"similarity_validation_faiss_{pair.adi_pair}.csv"
            )
        elif train_source == "pydi":
            # PyDI train + val only exist for the two pairs with PyDI gold
            # (dbpedia_fullcontact has pydi_pair=None — skipped).
            if pair.pydi_pair is not None:
                train_df = _load_pydi_split(
                    pair.pydi_pair, "train", swap_ids=not pair.pydi_left_is_left
                )
                val_df = _load_pydi_split(
                    pair.pydi_pair, "val", swap_ids=not pair.pydi_left_is_left
                )

        train_dropped = val_dropped = 0
        if train_df is not None:
            n_train_before = len(train_df)
            train_df = _drop_pairs_in(train_df, test_keys)
            train_dropped = n_train_before - len(train_df)
        if val_df is not None:
            n_val_before = len(val_df)
            val_df = _drop_pairs_in(val_df, test_keys)
            val_dropped = n_val_before - len(val_df)

        if train_df is not None:
            train_records.extend(
                _build_records(
                    train_df,
                    pair,
                    use_committee_scope=use_committee_scope,
                    sources=sources,
                )
            )
        if val_df is not None:
            val_records.extend(
                _build_records(
                    val_df,
                    pair,
                    use_committee_scope=use_committee_scope,
                    sources=sources,
                )
            )
        if pydi_test is not None:
            test_records.extend(
                _build_records(
                    pydi_test,
                    pair,
                    use_committee_scope=use_committee_scope,
                    sources=sources,
                )
            )

        train_count = len(train_df) if train_df is not None else 0
        val_count = len(val_df) if val_df is not None else 0
        print(
            f"  {pair.name:<22}  "
            f"train({src_tag}) {train_count} (dropped {train_dropped} leak)  "
            f"val({src_tag}) {val_count} (dropped {val_dropped} leak)  "
            f"test(PyDI) {len(pydi_test) if pydi_test is not None else 0}"
        )

    train_records = _dedupe_by_pair(train_records, "train")

    for split, records in [
        ("train", train_records),
        ("val", val_records),
        ("test", test_records),
    ]:
        out_path = output_dir / f"{split}.json.gz"
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
    sw_path = output_dir / "train_sample_weights.csv"
    pd.DataFrame(rows).to_csv(sw_path, index=False)
    print(
        f"sample weights (sklearn-balanced): pos={weights[1]:.4f}  "
        f"neg={weights[0]:.4f} -> {sw_path}"
    )

    if use_committee_scope:
        fields_path = write_committee_fields_sidecar(output_dir, DOMAIN)
        print(f"field scope -> {fields_path}")
        print(
            "Train the baseline committee Ditto on this wide scope with:\n"
            f"  python usecases_synthetic/scripts/ditto/train.py --domain {DOMAIN} "
            f"--train-json-gz {output_dir / 'train.json.gz'} "
            f"--val-json-gz {output_dir / 'val.json.gz'} --config <recipe.yaml>"
        )


if __name__ == "__main__":
    main()
