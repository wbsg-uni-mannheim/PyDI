#!/usr/bin/env python3
"""R10-G: retrain the *variant* Ditto checkpoint for one (domain, level).

Builds Ditto WDC training data from the K2-regenerated
``<pair>_train_corner_filled.csv`` splits joined against the *variant*
(perturbed) source records, invokes the existing Ditto trainer
(``scripts/ditto/train.py``), and places the result at the variant
checkpoint path the committee runner reads:
``cache/ditto_checkpoints/<domain>/variant_<level>/best``
(see :func:`committee_em._resolve_variant_checkpoint_path`).

R10-I (2026-05-29): the training data is built on the *wide* committee
field scope (``ditto_plm.fields`` == ``DOMAIN_TEXT_COLS``), column-mapped
exactly the way the committee EM runner maps sources before inference, so
the variant checkpoint trains on the same surface it scores against. The
variant sources carry K8 (schema-naming) renames at medium/hard; those are
handled by translating the committee ``column_mapping`` through K8 via
``VariantBundle.resolve_column_mapping`` (the same call the committee
runner makes), which restores every column to its canonical name.

Phase 1 (R10-G) is code-only: this script + a smoke test. The actual
training runs per-domain in phase 2 of each domain's step-5 cascade,
driven by ``scripts/retrain_variant_cascade.py``.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from usecases_synthetic.lib.domain_config import (  # noqa: E402
    SYNTHETIC_DIR,
    VALID_DOMAINS,
    load_knob_config,
)
from usecases_synthetic.lib.variant_loader import load_variant  # noqa: E402
from usecases_synthetic.scripts.ditto.prepare_em_training_data import (  # noqa: E402
    build_ditto_pair_records_committee_scope,
    committee_column_mapping,
    committee_ditto_fields,
    write_json_gz,
)

logger = logging.getLogger(__name__)

_DITTO_TRAIN_PY = SYNTHETIC_DIR / "scripts" / "ditto" / "train.py"
_DEFAULT_TRAIN_YAML = SYNTHETIC_DIR / "config" / "ditto" / "default_train.yaml"
_VARIANT_LEVELS = ("easy", "medium", "hard")

# Same convention the baseline prep uses when a pair ships no held-out val
# (ditto/_prep_games._split_train_val_stratified): hold out a stratified 20%
# of train as the early-stopping val split, seeded for reproducibility.
_VAL_FRACTION = 0.2
_SPLIT_SEED = 42


def _split_train_val_stratified(
    df: "Any", *, val_fraction: float = _VAL_FRACTION, seed: int = _SPLIT_SEED
):
    """Stratified train/val split on ``label`` (mirrors _prep_games).

    Used when a packaged variant ships ``*_train_corner_filled.csv`` but no
    ``*_val_corner_filled.csv`` (e.g. games): the Ditto trainer needs a held-out
    val split for early stopping, so we carve one from the train gold rather
    than fall back to the baseline checkpoint. Falls back to an unstratified
    split if a class has too few rows to stratify.
    """
    from sklearn.model_selection import train_test_split

    try:
        train_df, val_df = train_test_split(
            df, test_size=val_fraction, random_state=seed, stratify=df["label"]
        )
    except ValueError:
        # too few rows in a class to stratify — split without stratification
        train_df, val_df = train_test_split(
            df, test_size=val_fraction, random_state=seed
        )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def _ditto_variant_dir(domain: str, level: str) -> Path:
    """Return ``cache/ditto_checkpoints/<domain>/variant_<level>``.

    This is ``<baseline_checkpoint_parent>/variant_<level>`` for the
    baseline path declared in ``em_matching_committee_<domain>.yaml``, so
    the symlinked ``best`` inside it is exactly what
    :func:`committee_em._resolve_variant_checkpoint_path` looks up.
    """
    return SYNTHETIC_DIR / "cache" / "ditto_checkpoints" / domain / f"variant_{level}"


def _dedupe_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop duplicate pairs (by unordered ``(id_left, id_right)``)."""
    seen: set[frozenset[str]] = set()
    out: list[dict[str, Any]] = []
    for rec in records:
        key = frozenset((str(rec["id_left"]), str(rec["id_right"])))
        if key in seen:
            continue
        seen.add(key)
        out.append(rec)
    return out


def _invoke_ditto_train(
    train_json: Path,
    val_json: Path,
    run_parent: Path,
    *,
    fields: str,
    batch_size: int,
    max_len: int,
    max_field_len: int,
    config_path: Path,
) -> Path:
    """Run ``scripts/ditto/train.py`` and return the produced ``best`` dir.

    Isolated as a single boundary so the smoke test can monkeypatch it
    (the real call shells out to the trainer, which needs MPS/CUDA).
    """
    run_parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(_DITTO_TRAIN_PY),
        "--train-json-gz",
        str(train_json),
        "--val-json-gz",
        str(val_json),
        "--output-dir",
        str(run_parent),
        "--config",
        str(config_path),
        "--fields",
        fields,
        "--batch-size",
        str(batch_size),
        "--max-len",
        str(max_len),
        "--max-field-len",
        str(max_field_len),
    ]
    logger.info("Invoking Ditto trainer: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    bests = sorted(
        run_parent.glob("run_*/checkpoints/best"),
        key=lambda p: p.stat().st_mtime,
    )
    if not bests:
        raise RuntimeError(f"Ditto trainer produced no checkpoint under {run_parent}")
    return bests[-1]


def _place_checkpoint(variant_dir: Path, produced_best: Path) -> Path:
    """Symlink ``variant_dir/best`` to the produced checkpoint directory."""
    variant_dir.mkdir(parents=True, exist_ok=True)
    link = variant_dir / "best"
    if link.is_symlink():
        link.unlink()
    elif link.exists():
        shutil.rmtree(link)
    link.symlink_to(produced_best.resolve(), target_is_directory=True)
    logger.info("Placed variant Ditto checkpoint: %s -> %s", link, produced_best)
    return link


def retrain_variant_ditto(
    domain: str,
    level: str,
    *,
    root_override: Path | None = None,
    work_dir: Path | None = None,
    out_dir: Path | None = None,
) -> Path:
    """Retrain + place the variant Ditto checkpoint for ``(domain, level)``.

    Returns the variant ``best`` checkpoint path.

    ``out_dir`` overrides where the stable ``best`` symlink is placed. When
    ``None`` (the committee default) it lands at
    ``cache/ditto_checkpoints/<domain>/variant_<level>`` (read by the
    committee runner). The best-of-breed pipeline passes a pipeline-isolated
    location under ``pipelines/<domain>/checkpoints/...`` (no committee-cache
    reuse). Pass the SAME path as ``work_dir`` so the produced ``run_*``
    directories are isolated too.
    """
    if level not in _VARIANT_LEVELS:
        raise ValueError(
            f"level must be one of {_VARIANT_LEVELS}; got {level!r} "
            "(baseline has no variant checkpoint)"
        )

    bundle = load_variant(domain, level, root_override=root_override)
    # knob-02 still supplies the PLM *hyperparameters* (batch size, seq /
    # field length) — the R2 winner recipe. The *field scope* now comes from
    # the wide committee list, not knob-02 canonical_schema.
    knob02 = load_knob_config(2, domain)

    # R10-I: wide committee field scope + the committee's own column_mapping,
    # translated through this variant's K8 renames so a mapped source carries
    # exactly the canonical column names the DittoMatcher reads at inference.
    fields = committee_ditto_fields(domain)
    column_mapping = bundle.resolve_column_mapping(committee_column_mapping(domain))

    train_records: list[dict[str, Any]] = []
    val_records: list[dict[str, Any]] = []
    for src1, src2 in bundle.source_pairs:
        regen = bundle.em_gold_regenerated.get((src1, src2), {})
        train_gold = regen.get("train", {}).get("corner_filled")
        val_gold = regen.get("val", {}).get("corner_filled")
        if train_gold is None or train_gold.empty:
            # No corner_filled train gold for this pair — nothing to learn from.
            continue
        if val_gold is None or val_gold.empty:
            # Packaged variant shipped no *_val_corner_filled for this pair
            # (e.g. games): hold out a stratified 20% of the train gold as the
            # early-stopping val split, same as the baseline prep. Keeps a
            # genuine variant-retrained checkpoint instead of aliasing to the
            # baseline.
            train_gold, val_gold = _split_train_val_stratified(train_gold)
            logger.info(
                "%s/%s %s_2_%s: no val_corner_filled; held out %d/%d rows as "
                "stratified val split (seed=%d)",
                domain,
                level,
                src1,
                src2,
                len(val_gold),
                len(train_gold) + len(val_gold),
                _SPLIT_SEED,
            )
        train_records.extend(
            build_ditto_pair_records_committee_scope(
                train_gold,
                domain,
                src1,
                src2,
                sources=bundle.sources,
                fields=fields,
                column_mapping=column_mapping,
            )
        )
        val_records.extend(
            build_ditto_pair_records_committee_scope(
                val_gold,
                domain,
                src1,
                src2,
                sources=bundle.sources,
                fields=fields,
                column_mapping=column_mapping,
            )
        )

    train_records = _dedupe_records(train_records)
    val_records = _dedupe_records(val_records)
    if not train_records:
        raise RuntimeError(
            f"No corner_filled train records for {domain}/{level}; "
            "did generate_variant + package_variant (R10-F) land the "
            "*_train_corner_filled.csv files?"
        )
    if not val_records:
        raise RuntimeError(
            f"No corner_filled val records for {domain}/{level}; the Ditto "
            "trainer needs a validation split for early stopping."
        )

    work = work_dir or (SYNTHETIC_DIR / "output" / "ditto_variant" / domain / level)
    train_json = work / "train.json.gz"
    val_json = work / "val.json.gz"
    write_json_gz(train_records, train_json)
    write_json_gz(val_records, val_json)
    logger.info(
        "%s/%s: wrote %d train + %d val Ditto records",
        domain,
        level,
        len(train_records),
        len(val_records),
    )

    produced = _invoke_ditto_train(
        train_json,
        val_json,
        work / "runs",
        fields=",".join(fields),
        batch_size=int(knob02.get("plm_batch_size", 16)),
        max_len=int(knob02.get("plm_max_len", 256)),
        max_field_len=int(knob02.get("plm_max_field_len", 350)),
        config_path=_DEFAULT_TRAIN_YAML,
    )
    target_dir = out_dir if out_dir is not None else _ditto_variant_dir(domain, level)
    return _place_checkpoint(target_dir, produced)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrain the variant Ditto checkpoint for one (domain, level)."
    )
    parser.add_argument("--domain", required=True, choices=sorted(VALID_DOMAINS))
    parser.add_argument("--level", required=True, choices=list(_VARIANT_LEVELS))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Pipeline-isolated checkpoint dir for the stable best symlink "
            "(also used as work_dir for run_* isolation). Default: the "
            "committee cache path cache/ditto_checkpoints/<domain>/variant_<level>."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    ckpt = retrain_variant_ditto(
        args.domain,
        args.level,
        work_dir=args.out_dir,
        out_dir=args.out_dir,
    )
    logger.info("Variant Ditto checkpoint ready: %s", ckpt)


if __name__ == "__main__":
    main()
