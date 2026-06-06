#!/usr/bin/env python3
"""R10-G: retrain the *variant* SC-Block checkpoint for one (domain, level).

Loads the K8-resolved *variant* (perturbed) source records and the
K2-regenerated ``<pair>_{train,val}_corner_filled.csv`` splits, then
invokes the existing SC-Block trainer (``scripts/sc_block/train.py``)
with that data injected via its ``data_override`` hook. The trainer
writes (and symlinks) ``best`` under the variant output directory, which
is exactly the path the committee runner reads:
``cache/sc_block_checkpoints/<domain>/variant_<level>/best``
(see :func:`committee_em._resolve_variant_checkpoint_path`).

Phase 1 (R10-G) is code-only: this script + a smoke test. The actual
training runs per-domain in phase 2, driven by
``scripts/retrain_variant_cascade.py``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from usecases_synthetic.lib.column_mapping import apply_column_mapping  # noqa: E402
from usecases_synthetic.lib.domain_config import (  # noqa: E402
    SYNTHETIC_DIR,
    load_knob_config,
)
from usecases_synthetic.lib.sc_block_train import DOMAIN_TEXT_COLS  # noqa: E402
from usecases_synthetic.lib.variant_loader import load_variant  # noqa: E402
from usecases_synthetic.scripts.sc_block.train import (  # noqa: E402
    _DEFAULT_EVAL_PAIRS,
    _load_blocking_column_mapping,
)

logger = logging.getLogger(__name__)

_VARIANT_LEVELS = ("easy", "medium", "hard")

_DataOverride = tuple[
    dict[str, pd.DataFrame],
    dict[tuple[str, str], pd.DataFrame],
    dict[tuple[str, str], dict[str, pd.DataFrame]],
]


def _scblock_variant_dir(domain: str, level: str) -> Path:
    """Return ``cache/sc_block_checkpoints/<domain>/variant_<level>``."""
    return (
        SYNTHETIC_DIR / "cache" / "sc_block_checkpoints" / domain / f"variant_{level}"
    )


def _build_variant_data(domain: str, bundle: Any) -> _DataOverride:
    """Build (sources_mapped, em_train_by_pair, em_splits_by_pair) for SC-Block.

    Mirrors ``sc_block.train._load_domain_data`` but on the variant
    bundle: variant sources mapped to the canonical ``text_cols`` via the
    K8-resolved blocking column mapping, and the per-pair corner_filled
    train/val splits as the contrastive supervision.
    """
    text_cols = DOMAIN_TEXT_COLS[domain]
    resolved = bundle.resolve_column_mapping(_load_blocking_column_mapping(domain))

    sources_mapped: dict[str, pd.DataFrame] = {}
    for src, df in bundle.sources.items():
        mapping = resolved.get(src, {})
        mapped = apply_column_mapping(df, mapping) if mapping else df.copy()
        missing = [c for c in text_cols if c not in mapped.columns]
        if missing:
            logger.warning(
                "source %s missing text_cols %s after column_mapping; "
                "those fields will serialize as empty",
                src,
                missing,
            )
            for col in missing:
                mapped[col] = pd.NA
        sources_mapped[src] = mapped

    em_train_by_pair: dict[tuple[str, str], pd.DataFrame] = {}
    em_splits_by_pair: dict[tuple[str, str], dict[str, pd.DataFrame]] = {}
    for pair in bundle.source_pairs:
        regen = bundle.em_gold_regenerated.get(pair, {})
        train_gold = regen.get("train", {}).get("corner_filled")
        if train_gold is None or train_gold.empty:
            continue
        em_train_by_pair[pair] = train_gold
        splits: dict[str, pd.DataFrame] = {"train": train_gold}
        val_gold = regen.get("val", {}).get("corner_filled")
        if val_gold is not None and not val_gold.empty:
            splits["val"] = val_gold
        em_splits_by_pair[pair] = splits

    return sources_mapped, em_train_by_pair, em_splits_by_pair


def _invoke_scblock_train(
    domain: str,
    eval_pair: tuple[str, str],
    output_dir: Path,
    data_override: _DataOverride,
    eval_top_k: int = 50,
) -> dict[str, Any]:
    """Run the SC-Block trainer with injected variant data.

    Isolated as a single boundary so the smoke test can monkeypatch it
    (the real call trains a RoBERTa encoder, which needs MPS/CUDA).

    ``eval_top_k`` caps the per-query candidate set in the per-epoch val
    recall eval (the dominant cost on large domains like papers); lower it
    to speed training. Defaults to 50 to preserve the trainer default.
    """
    from usecases_synthetic.scripts.sc_block.train import train as sc_train

    return sc_train(
        domain=domain,
        eval_pair=eval_pair,
        output_dir=output_dir,
        data_override=data_override,
        eval_top_k=eval_top_k,
    )


def retrain_variant_sc_block(
    domain: str,
    level: str,
    *,
    eval_pair: tuple[str, str] | None = None,
    root_override: Path | None = None,
    out_dir: Path | None = None,
    eval_top_k: int = 50,
) -> Path:
    """Retrain + place the variant SC-Block checkpoint for ``(domain, level)``.

    Returns the variant ``best`` checkpoint path.

    ``out_dir`` overrides the trainer output directory. When ``None`` (the
    committee default) it lands at
    ``cache/sc_block_checkpoints/<domain>/variant_<level>`` (read by the
    committee runner). The best-of-breed pipeline passes a pipeline-isolated
    location under ``pipelines/<domain>/checkpoints/...`` (no committee-cache
    reuse).
    """
    if level not in _VARIANT_LEVELS:
        raise ValueError(
            f"level must be one of {_VARIANT_LEVELS}; got {level!r} "
            "(baseline has no variant checkpoint)"
        )
    # Touch the knob-02 config so an unknown domain fails fast + symmetric
    # with the Ditto retrain script.
    load_knob_config(2, domain)

    bundle = load_variant(domain, level, root_override=root_override)
    sources_mapped, em_train_by_pair, em_splits_by_pair = _build_variant_data(
        domain, bundle
    )
    if not em_train_by_pair:
        raise RuntimeError(
            f"No corner_filled train splits for {domain}/{level}; did "
            "generate_variant + package_variant (R10-F) land the "
            "*_train_corner_filled.csv files?"
        )

    chosen = eval_pair or _DEFAULT_EVAL_PAIRS[domain]
    if chosen not in em_splits_by_pair:
        reversed_pair = (chosen[1], chosen[0])
        if reversed_pair in em_splits_by_pair:
            chosen = reversed_pair
        else:
            chosen = next(iter(em_splits_by_pair))
            logger.warning("eval_pair %s absent; falling back to %s", eval_pair, chosen)

    output_dir = out_dir if out_dir is not None else _scblock_variant_dir(domain, level)
    _invoke_scblock_train(
        domain,
        chosen,
        output_dir,
        (sources_mapped, em_train_by_pair, em_splits_by_pair),
        eval_top_k=eval_top_k,
    )
    best = output_dir / "best"
    logger.info("Variant SC-Block checkpoint ready: %s", best)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrain the variant SC-Block checkpoint for one (domain, level)."
    )
    parser.add_argument("--domain", required=True, choices=sorted(DOMAIN_TEXT_COLS))
    parser.add_argument("--level", required=True, choices=list(_VARIANT_LEVELS))
    parser.add_argument(
        "--eval-pair",
        default=None,
        help="Comma-separated <src1>,<src2>; defaults to the domain's default.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Pipeline-isolated trainer output dir (the best symlink lands at "
            "<out-dir>/best). Default: the committee cache path "
            "cache/sc_block_checkpoints/<domain>/variant_<level>."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    eval_pair: tuple[str, str] | None = None
    if args.eval_pair:
        parts = tuple(p.strip() for p in args.eval_pair.split(","))
        if len(parts) != 2:
            parser.error("--eval-pair must be '<src1>,<src2>'")
        eval_pair = (parts[0], parts[1])

    ckpt = retrain_variant_sc_block(
        args.domain, args.level, eval_pair=eval_pair, out_dir=args.out_dir
    )
    logger.info("Variant SC-Block checkpoint ready: %s", ckpt)


if __name__ == "__main__":
    main()
