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


def retrain_domain_level(domain: str, level: str) -> dict[str, Path]:
    """Retrain both variant matchers for one ``(domain, level)``.

    Returns ``{"ditto": <path>, "sc_block": <path>}``. A per-level log
    file captures the run for the cascade audit trail.
    """
    handler = _add_file_handler(_level_log_path(domain, level))
    try:
        logger.info("=== R10-G variant retrain: %s / %s ===", domain, level)
        ditto_ckpt = retrain_variant_ditto(domain, level)
        sc_ckpt = retrain_variant_sc_block(domain, level)
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
) -> dict[tuple[str, str], dict[str, Path]]:
    """Retrain variant checkpoints for every ``(domain, level)``."""
    results: dict[tuple[str, str], dict[str, Path]] = {}
    for domain in domains:
        for level in levels:
            results[(domain, level)] = retrain_domain_level(domain, level)
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
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    results: dict[tuple[str, str], dict[str, Any]] = run_cascade(
        args.domain, tuple(args.levels)
    )
    for (domain, level), ckpts in results.items():
        logger.info(
            "%s/%s -> ditto=%s sc_block=%s",
            domain,
            level,
            ckpts["ditto"],
            ckpts["sc_block"],
        )


if __name__ == "__main__":
    main()
