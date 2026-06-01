#!/usr/bin/env python3
"""R2.1 helper (ADI-data variant): build Ditto json.gz files for games.

Per pair:

* dbpedia_metacritic
    train = ADI training_dbpedia_metacritic_latest.csv
    val   = ADI similarity_validation_faiss_dbpedia_metacritic.csv
    test  = PyDI top-level dbpedia_2_metacritic_test.csv (337 pairs, clean ids)

* dbpedia_sales
    train = ADI training_dbpedia_sales_latest.csv
    val   = ADI similarity_validation_faiss_dbpedia_sales.csv
    test  = PyDI top-level dbpedia_2_sales_test.csv (402 pairs, clean ids)

* metacritic_sales
    train = (none — option A from R2.2 redo: ADI has no data for this
            pair, and the top-level PyDI train/test files are
            byte-identical so cannot be split safely. Model must learn
            mc↔sales by transfer from the other 2 pairs.)
    val   = (none — same reason)
    test  = PyDI top-level metacritic_2_sales_test.csv (582 pairs, clean ids)

Leak removal: PyDI test pairs dropped from train and val (cross-source
overlap detected via frozenset key on (id1, id2)). Intra-train duplicates
deduped.

Side-alignment: all top-level PyDI test files for the 2 ADI pairs are
already in ADI's (src_left, src_right) order so no id1<->id2 swap is
needed — db↔mc test is dbpedia_2_metacritic_test (id1=db, id2=mc), and
db↔sales test is dbpedia_2_sales_test (id1=db, id2=sales).

Outputs (all under ``usecases_synthetic/output/ditto/games/``):
    train.json.gz
    val.json.gz
    test.json.gz
    train_sample_weights.csv

Run from the repo root:

    python usecases_synthetic/scripts/ditto/_prep_games.py
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

DOMAIN = "games"
ADI_DIR = (
    ROOT
    / "automatic-data-integration"
    / "scripts"
    / "output"
    / "games_0302"
    / "entity_resolution"
)
PYDI_EM_GOLD_DIR = USECASES_DIR / DOMAIN / "input" / "entitymatching"
OUTPUT_DIR_RAW = SYNTHETIC_DIR / "output" / "ditto" / DOMAIN
OUTPUT_DIR_NORMALIZED = SYNTHETIC_DIR / "output" / "ditto" / f"{DOMAIN}_normalized"
OUTPUT_DIR_PYDI_RAW = SYNTHETIC_DIR / "output" / "ditto" / f"{DOMAIN}_pydi_raw"
OUTPUT_DIR_PYDI_NORMALIZED = (
    SYNTHETIC_DIR / "output" / "ditto" / f"{DOMAIN}_pydi_normalized"
)

# 80/20 train/val split for the PyDI-train path (PyDI ships no on-disk
# val for games — train CSV gets split here with a fixed seed so reruns
# are deterministic).
PYDI_VAL_FRACTION = 0.2
PYDI_SPLIT_SEED = 42


@dataclass(frozen=True)
class Pair:
    name: str
    src_left: str
    src_right: str
    # Train source: "adi" (ADI training pool) or "none" (test-only,
    # transfer-learned from the other pairs).
    train_source: str
    adi_pair: str | None
    # Top-level PyDI EM gold filename (without the .csv suffix), e.g.
    # "dbpedia_2_metacritic_test"
    pydi_test_file: str


PAIRS: list[Pair] = [
    Pair(
        name="dbpedia_metacritic",
        src_left="dbpedia",
        src_right="metacritic",
        train_source="adi",
        adi_pair="dbpedia_metacritic",
        pydi_test_file="dbpedia_2_metacritic_test",
    ),
    Pair(
        name="dbpedia_sales",
        src_left="dbpedia",
        src_right="sales",
        train_source="adi",
        adi_pair="dbpedia_sales",
        pydi_test_file="dbpedia_2_sales_test",
    ),
    Pair(
        name="metacritic_sales",
        src_left="metacritic",
        src_right="sales",
        train_source="none",
        adi_pair=None,
        pydi_test_file="metacritic_2_sales_test",
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


def _normalize_pair_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["id1", "id2", "label"]].copy()
    df["id1"] = df["id1"].astype(str)
    df["id2"] = df["id2"].astype(str)
    df["label"] = df["label"].map(_normalize_label_to_int).astype(int)
    return df


def _load_adi_csv(path: Path) -> pd.DataFrame:
    return _normalize_pair_df(pd.read_csv(path))


def _load_pydi_em_gold(stem: str) -> pd.DataFrame:
    return _normalize_pair_df(read_em_gold_csv(PYDI_EM_GOLD_DIR / f"{stem}.csv"))


def _load_pydi_train_for_pair(pair: "Pair") -> pd.DataFrame | None:
    """Load PyDI train CSV for ``pair``, normalizing id1/id2 to canonical direction.

    Tries ``<src_left>_2_<src_right>_train.csv`` first; falls back to the
    reversed-direction ``<src_right>_2_<src_left>_train.csv`` and swaps
    ``id1`` / ``id2`` so the returned frame has ``id1 = src_left id``,
    ``id2 = src_right id`` (matches what ``_canonical_record`` expects).

    Mirrors the direction tolerance in ``variant_loader._load_em_gold``
    (landed 2026-05-26), with the addition of an explicit column swap
    here because Ditto's COL/VAL serialisation is order-sensitive (the
    ``_left`` / ``_right`` sides must match the trained model's
    convention). Closed-set scoring is order-invariant via
    ``tuple(sorted(p))``, so this only matters for the matcher input.

    Returns ``None`` when neither direction's train CSV exists (e.g.
    ``metacritic_sales``).
    """
    fwd = PYDI_EM_GOLD_DIR / f"{pair.src_left}_2_{pair.src_right}_train.csv"
    rev = PYDI_EM_GOLD_DIR / f"{pair.src_right}_2_{pair.src_left}_train.csv"
    if fwd.exists():
        return _normalize_pair_df(read_em_gold_csv(fwd))
    if rev.exists():
        df_rev = _normalize_pair_df(read_em_gold_csv(rev))
        # Swap so id1 = src_left id, id2 = src_right id.
        df = pd.DataFrame(
            {
                "id1": df_rev["id2"].values,
                "id2": df_rev["id1"].values,
                "label": df_rev["label"].values,
            }
        )
        return df
    return None


def _split_train_val_stratified(
    df: pd.DataFrame, *, val_fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """80/20 stratified split keeping the per-class proportion stable.

    Stratifies on ``label`` so train and val see the same positive ratio
    as the source frame. Reset indices so downstream callers don't
    depend on the input's index.
    """
    from sklearn.model_selection import train_test_split

    train_df, val_df = train_test_split(
        df,
        test_size=val_fraction,
        random_state=seed,
        stratify=df["label"],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


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
    normalize: bool = False,
) -> list[dict]:
    """Build pair records for one split.

    ``use_committee_scope`` (the committee-correct ``--train-source pydi``
    path) builds the *wide* committee field scope column-mapped off the base
    PyDI sources, matching the committee runner's inference serialization.
    The legacy ``--train-source adi`` path keeps the narrow knob-02
    ``canonical_schema`` projection — it feeds only the pool-builder Ditto,
    never the committee, and is intentionally left untouched.
    """
    if use_committee_scope:
        return build_ditto_pair_records_committee_scope(
            gold,
            DOMAIN,
            pair.src_left,
            pair.src_right,
            sources=sources,
            normalize=normalize,
        )
    return build_ditto_pair_records_from_gold(
        gold, DOMAIN, pair.src_left, pair.src_right, normalize=normalize
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
    parser = argparse.ArgumentParser(
        description=(
            "Prepare Ditto train/val/test json.gz files for games. "
            "With --train-source pydi, trains on PyDI's gold splits "
            "(in-distribution with the eval test); the legacy ADI path "
            "remains as the default for backwards compatibility. With "
            "--normalize, applies the human-baseline platform alias + "
            "title cleanup before serialisation."
        )
    )
    parser.add_argument(
        "--train-source",
        choices=["pydi", "adi"],
        default="pydi",
        help=(
            "Where train + val records come from. 'pydi' (default, "
            "committee-correct per plan_revision.md R6-3) loads "
            "<pair>_train.csv from usecases/games/input/entitymatching/ "
            "and holds out 20%% as val with seed=42 (PyDI has no "
            "on-disk val for games). 'adi' is the legacy R2 setup that "
            "trains on automatic-data-integration's labeled pool — "
            "kept only for pool-builder use; must NOT be wired to the "
            "committee."
        ),
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help=(
            "Apply per-domain value normalisation "
            "(usecases_synthetic.lib.domain_value_norm) before "
            "serialisation; writes to the *_normalized output directory."
        ),
    )
    args = parser.parse_args()
    normalize = bool(args.normalize)
    train_source = args.train_source

    if train_source == "pydi":
        output_dir = OUTPUT_DIR_PYDI_NORMALIZED if normalize else OUTPUT_DIR_PYDI_RAW
    else:
        output_dir = OUTPUT_DIR_NORMALIZED if normalize else OUTPUT_DIR_RAW
    output_dir.mkdir(parents=True, exist_ok=True)

    # R10-I: the committee (pydi) path builds the wide committee field scope
    # column-mapped off the base PyDI sources; the legacy ADI path (pool
    # builder only) stays on the narrow knob-02 projection.
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

    norm_tag = "normalized" if normalize else "raw"
    src_tag = train_source.upper()
    print(
        f"Loading per-pair sets, computing leaks, building records "
        f"({src_tag}, {norm_tag})..."
    )
    for pair in PAIRS:
        # F11 dropped metacritic_2_sales_test.csv (100% canonical-pair
        # overlap with the test split). The pair is gone from
        # config/domains/games.yaml; skip it here too rather than
        # crash. Both pre-F11 and post-F11 callers stay happy.
        pydi_test_path = PYDI_EM_GOLD_DIR / f"{pair.pydi_test_file}.csv"
        if not pydi_test_path.exists():
            print(f"  {pair.name:<22}  SKIP (missing {pydi_test_path.name})")
            continue
        pydi_test = _load_pydi_em_gold(pair.pydi_test_file)
        test_keys = _pair_keys(pydi_test)

        train_df: pd.DataFrame | None = None
        val_df: pd.DataFrame | None = None
        if train_source == "adi" and pair.train_source == "adi":
            assert pair.adi_pair is not None
            train_df = _load_adi_csv(
                ADI_DIR / "training" / f"training_{pair.adi_pair}_latest.csv"
            )
            val_df = _load_adi_csv(
                ADI_DIR
                / "validation"
                / f"similarity_validation_faiss_{pair.adi_pair}.csv"
            )
        elif train_source == "pydi":
            pydi_train = _load_pydi_train_for_pair(pair)
            if pydi_train is not None:
                # 80/20 train/val split (stratified by label, fixed seed).
                train_df, val_df = _split_train_val_stratified(
                    pydi_train,
                    val_fraction=PYDI_VAL_FRACTION,
                    seed=PYDI_SPLIT_SEED,
                )

        train_dropped = 0
        if train_df is not None:
            n_train_before = len(train_df)
            train_df = _drop_pairs_in(train_df, test_keys)
            train_dropped = n_train_before - len(train_df)

        val_dropped = 0
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
                    normalize=normalize,
                )
            )
        if val_df is not None:
            val_records.extend(
                _build_records(
                    val_df,
                    pair,
                    use_committee_scope=use_committee_scope,
                    sources=sources,
                    normalize=normalize,
                )
            )
        test_records.extend(
            _build_records(
                pydi_test,
                pair,
                use_committee_scope=use_committee_scope,
                sources=sources,
                normalize=normalize,
            )
        )

        train_count = len(train_df) if train_df is not None else 0
        val_count = len(val_df) if val_df is not None else 0
        print(
            f"  {pair.name:<22}  train({src_tag}) {train_count} "
            f"(dropped {train_dropped} leak)  "
            f"val({src_tag}) {val_count} (dropped {val_dropped} leak)  "
            f"test(PyDI) {len(pydi_test)}"
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
