#!/usr/bin/env python3
"""R10-G phase 2 driver: retrain variant EM checkpoints per domain.

For each requested domain, loops ``easy / medium / hard`` and retrains
both variant matchers — Ditto (EM matching) and SC-Block (EM blocking) —
writing the checkpoints the committee runner reads at
``cache/<model>_checkpoints/<domain>/variant_<level>/best``. A per-level
training log is written to
``usecases_synthetic/output/<domain>/<level>/r7c_retrain.log`` for the
cascade audit trail.

Phase 1 (R10-G) ships this driver as code; the per-domain runs execute in
phase 2 of each domain's step-5 cascade (Magellan is covered by the
committee runner's runtime per-pair fit, so it needs no checkpoint here).

Usage::

    python usecases_synthetic/scripts/retrain_variant_cascade.py \\
        --domain products [--domain music ...] [--levels easy medium hard]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from usecases_synthetic.lib.domain_config import SYNTHETIC_DIR  # noqa: E402
from usecases_synthetic.scripts.ditto.retrain_variant import (  # noqa: E402
    retrain_variant_ditto,
)
from usecases_synthetic.scripts.sc_block.retrain_variant import (  # noqa: E402
    retrain_variant_sc_block,
)

logger = logging.getLogger(__name__)

_VARIANT_LEVELS = ("easy", "medium", "hard")


def _level_log_path(domain: str, level: str) -> Path:
    return SYNTHETIC_DIR / "output" / domain / level / "r7c_retrain.log"


def _add_file_handler(path: Path) -> logging.Handler:
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    logging.getLogger().addHandler(handler)
    return handler


def _is_corner_filled_data_gap(exc: Exception) -> bool:
    """True for the known "packaged variant lacks corner_filled splits" error.

    Both retrain_variant_ditto and retrain_variant_sc_block raise a
    ``RuntimeError`` whose message contains ``corner_filled`` when the
    packaged variant shipped no ``*_train_corner_filled.csv`` at all — i.e.
    there is genuinely nothing to train on. The current trigger is **papers**,
    whose variants ship only ``*_test_corner_filled.csv`` (a real
    variant-generation gap — track upstream). NOTE: a *missing val* split
    (games ships ``*_train_corner_filled`` but no ``*_val_corner_filled``) is
    NOT a data gap here — retrain_variant_ditto holds out a stratified val
    split from train in that case, so games does not reach this skip. Any
    other exception is a genuine bug and must NOT be swallowed.
    """
    return isinstance(exc, RuntimeError) and "corner_filled" in str(exc)


def retrain_domain_level(
    domain: str, level: str, *, out_root: Path | None = None, eval_top_k: int = 50
) -> dict[str, Path | None]:
    """Retrain both variant matchers for one ``(domain, level)``.

    ``out_root`` redirects both checkpoints off the committee cache into a
    pipeline-isolated tree (the best-of-breed pipeline's no-committee-reuse
    policy). When set, ditto lands at
    ``<out_root>/em_matching/ditto/variant_<level>/best`` and sc_block at
    ``<out_root>/em_blocking/sc_block/variant_<level>/best`` (ditto's run_*
    work dirs are isolated under the same ditto dir). When ``None`` (the
    committee default) both land under ``cache/<model>_checkpoints/...``.

    Returns ``{"ditto": <path|None>, "sc_block": <path|None>}``. When a
    matcher cannot be retrained because the packaged variant lacks the
    corner_filled EM splits, that matcher is SKIPPED with a loud ``error``
    log (NOT a silent drop) instead of aborting the whole cascade: the
    committee member stays active and ``validate_variant`` falls back to
    the baseline ``/best`` checkpoint for that level
    (``committee_em._resolve_variant_checkpoint_path`` →
    ``variant_ckpt_distinct=False``). Any other exception propagates so
    genuine training bugs still fail the job loudly. A per-level log file
    captures the run for the cascade audit trail.
    """
    ditto_out: Path | None = None
    sc_out: Path | None = None
    if out_root is not None:
        ditto_out = out_root / "em_matching" / "ditto" / f"variant_{level}"
        sc_out = out_root / "em_blocking" / "sc_block" / f"variant_{level}"
    handler = _add_file_handler(_level_log_path(domain, level))
    try:
        logger.info("=== R10-G variant retrain: %s / %s ===", domain, level)
        ditto_ckpt: Path | None = None
        sc_ckpt: Path | None = None
        try:
            ditto_ckpt = retrain_variant_ditto(
                domain, level, work_dir=ditto_out, out_dir=ditto_out
            )
        except RuntimeError as exc:
            if not _is_corner_filled_data_gap(exc):
                raise
            logger.error(
                "SKIP ditto variant %s/%s (missing corner_filled data: %s); "
                "validate_variant will fall back to the baseline ditto "
                "checkpoint for this level.",
                domain,
                level,
                exc,
            )
        try:
            sc_ckpt = retrain_variant_sc_block(
                domain, level, out_dir=sc_out, eval_top_k=eval_top_k
            )
        except RuntimeError as exc:
            if not _is_corner_filled_data_gap(exc):
                raise
            logger.error(
                "SKIP sc_block variant %s/%s (missing corner_filled data: %s); "
                "validate_variant will fall back to the baseline sc_block "
                "checkpoint for this level.",
                domain,
                level,
                exc,
            )
        logger.info(
            "Done %s/%s: ditto=%s sc_block=%s", domain, level, ditto_ckpt, sc_ckpt
        )
        return {"ditto": ditto_ckpt, "sc_block": sc_ckpt}
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()


def run_cascade(
    domains: list[str],
    levels: tuple[str, ...] = _VARIANT_LEVELS,
    *,
    out_root: Path | None = None,
    eval_top_k: int = 50,
) -> dict[tuple[str, str], dict[str, Path | None]]:
    """Retrain variant checkpoints for every ``(domain, level)``.

    ``out_root`` (when set) redirects all checkpoints into the
    pipeline-isolated tree — see :func:`retrain_domain_level`.
    ``eval_top_k`` caps the sc_block per-epoch val-recall eval candidate
    set (the dominant cost on large domains); lower it to speed training.
    """
    results: dict[tuple[str, str], dict[str, Path | None]] = {}
    for domain in domains:
        for level in levels:
            results[(domain, level)] = retrain_domain_level(
                domain, level, out_root=out_root, eval_top_k=eval_top_k
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrain variant Ditto + SC-Block checkpoints per domain."
    )
    parser.add_argument(
        "--domain",
        action="append",
        required=True,
        help="Domain to retrain (repeatable for a cross-domain batch).",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        default=list(_VARIANT_LEVELS),
        choices=list(_VARIANT_LEVELS),
        help="Difficulty levels to retrain (default: easy medium hard).",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help=(
            "Pipeline-isolated checkpoint root (e.g. "
            "pipelines/<domain>/checkpoints). When set, ditto + sc_block "
            "variant checkpoints land under "
            "<out-root>/em_{matching,blocking}/<model>/variant_<level>/best "
            "instead of the committee cache. Use one --domain at a time when "
            "passing a domain-specific --out-root."
        ),
    )
    parser.add_argument(
        "--eval-top-k",
        type=int,
        default=50,
        help=(
            "sc_block per-epoch val-recall eval candidate cap (top-k per "
            "query) — the dominant training cost on large domains. Lower to "
            "~20 to speed papers. Default 50 (trainer default)."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    results: dict[tuple[str, str], dict[str, Any]] = run_cascade(
        args.domain,
        tuple(args.levels),
        out_root=args.out_root,
        eval_top_k=args.eval_top_k,
    )
    for (domain, level), ckpts in results.items():
        logger.info(
            "%s/%s -> ditto=%s sc_block=%s",
            domain,
            level,
            ckpts["ditto"],
            ckpts["sc_block"],
        )
    skipped = [
        f"{domain}/{level}:{model}"
        for (domain, level), ckpts in results.items()
        for model in ("ditto", "sc_block")
        if ckpts.get(model) is None
    ]
    if skipped:
        logger.warning(
            "Variant retrain SKIPPED for %d (domain/level:model) — missing "
            "corner_filled data; validate_variant uses the baseline /best "
            "checkpoint for these (committee member stays active, "
            "variant_model_distinct=0): %s",
            len(skipped),
            ", ".join(skipped),
        )


if __name__ == "__main__":
    main()
