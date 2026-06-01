#!/usr/bin/env python3
"""M9 — per-knob ablation generator + validator.

For each requested active knob, generate a single-knob ablation
variant (target knob at ``hard``, all other knobs at ``easy``) and run
``validate_variant`` against it, writing metrics under
``usecases_synthetic/validation/<domain>/ablation/knob_<id>/``.

This is the runner. Aggregation / per-knob effect-size analysis lives
in ``analyze_ablation.py``.

Usage
-----
::

    python usecases_synthetic/scripts/run_ablation_validation.py \\
        --domain companies

    python usecases_synthetic/scripts/run_ablation_validation.py \\
        --domain companies --knobs 8 --skip-existing

The default set covers the eight active S1 knobs (K1, K2, K3, K4, K5,
K6, K8, K10). K7 is deferred; K9 is S2-only.

Outputs per knob
----------------
- ``usecases/<domain>-augmented/ablation_knob_<id>/`` — packaged
  variant (generated once and reusable across reruns when
  ``--skip-existing`` is passed).
- ``usecases_synthetic/validation/<domain>/ablation/knob_<id>/metrics.json``
- ``usecases_synthetic/validation/<domain>/ablation/knob_<id>/level_report.md``

The metrics.json files are the canonical artefact consumed by
``analyze_ablation.py``.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Ensure repo root is on sys.path.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from usecases_synthetic.lib.committee import Stage
from usecases_synthetic.lib.domain_config import SYNTHETIC_DIR, USECASES_DIR
from usecases_synthetic.scripts.generate_variant import (
    ACTIVE_KNOB_IDS,
    ablation_label,
    build_ablation_knob_levels,
    generate_variant,
)
from usecases_synthetic.scripts.validate_variant import validate_variant

logger = logging.getLogger(__name__)


ABLATION_VALIDATION_ROOT: Path = SYNTHETIC_DIR / "validation"


def ablation_variant_dir(domain: str, knob_id: str) -> Path:
    """Return the packaged-variant directory for an ablation knob."""
    return USECASES_DIR / f"{domain}-augmented" / ablation_label(knob_id)


def ablation_metrics_dir(domain: str, knob_id: str) -> Path:
    """Return the validation metrics directory for an ablation knob."""
    return ABLATION_VALIDATION_ROOT / domain / "ablation" / knob_id


def _normalise_knob_ids(raw_tokens: list[str]) -> list[str]:
    """Normalise a list of user-provided knob tokens to canonical ids.

    Parameters
    ----------
    raw_tokens : list of str
        Each element may be ``"1"``, ``"01"``, or ``"knob_01"``.

    Returns
    -------
    list of str
        Deduplicated canonical ids in canonical order, restricted to
        :data:`ACTIVE_KNOB_IDS`.

    Raises
    ------
    ValueError
        If any token is not an active knob id.
    """
    seen: set[str] = set()
    for raw in raw_tokens:
        token = raw.strip()
        if token.startswith("knob_"):
            kid = token
        else:
            try:
                kid = f"knob_{int(token):02d}"
            except ValueError as exc:
                raise ValueError(f"Invalid knob token {raw!r}") from exc
        if kid not in ACTIVE_KNOB_IDS:
            raise ValueError(
                f"Knob {kid!r} is not an active S1 knob. Active: "
                f"{list(ACTIVE_KNOB_IDS)}"
            )
        seen.add(kid)
    return [kid for kid in ACTIVE_KNOB_IDS if kid in seen]


def run_ablation(
    domain: str,
    knob_id: str,
    *,
    master_seed: int | None = None,
    with_llm: bool = False,
    stages: list[Stage] | None = None,
    fusion_input_member: str | None = None,
    ablation_level: str = "hard",
    identity_level: str = "easy",
    skip_existing_variant: bool = False,
    strict_cache: bool | None = None,
) -> dict[str, Any]:
    """Generate and validate a single ablation variant.

    Parameters
    ----------
    domain : str
        Domain name.
    knob_id : str
        Canonical knob id (e.g. ``"knob_08"``).
    master_seed : int or None
        Master RNG seed; defaults to the domain config seed.
    with_llm : bool
        Forwarded to ``validate_variant``; must match the baseline.
    stages : list of str, optional
        Pipeline stages to validate. Default: all three (``sm``,
        ``em``, ``fusion``).
    fusion_input_member : str, optional
        Override the EM member feeding Fusion. Default: baseline's
        recorded value.
    ablation_level, identity_level : str
        Level of the target knob and non-target knobs, respectively.
    skip_existing_variant : bool
        When ``True``, skip the generation step if a packaged variant
        directory already exists at the canonical ablation path.
    strict_cache : bool or None
        Forwarded to ``generate_variant`` as both ``strict_cache_k1``
        and ``strict_cache_k2``. ``None`` keeps each knob's default
        (strict at hard, non-strict elsewhere). Set to ``False`` on
        aliased domains (e.g. ``companies-small``) whose LLM caches
        have not been warmed yet — K1/K2 will make live LLM calls.

    Returns
    -------
    dict
        ``{"knob_id": ..., "variant_dir": ..., "metrics_dir": ...,
        "runtime_s": ..., "generated": bool}``.
    """
    variant_dir = ablation_variant_dir(domain, knob_id)
    metrics_dir = ablation_metrics_dir(domain, knob_id)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    generated = False
    t0 = time.monotonic()

    # --- Generation -------------------------------------------------------
    if skip_existing_variant and (variant_dir / "input" / "data").exists():
        logger.info("[%s] skip-existing: reusing variant at %s", knob_id, variant_dir)
    else:
        knob_levels = build_ablation_knob_levels(
            knob_id,
            ablation_level=ablation_level,
            identity_level=identity_level,
        )
        logger.info(
            "[%s] generating ablation variant (target=%s, others=%s)",
            knob_id,
            ablation_level,
            identity_level,
        )
        gv_kwargs: dict[str, Any] = {
            "domain": domain,
            "level": ablation_level,
            "master_seed": master_seed,
            "knob_levels": knob_levels,
            "label": ablation_label(knob_id),
        }
        if strict_cache is not None:
            gv_kwargs["strict_cache_k1"] = strict_cache
            gv_kwargs["strict_cache_k2"] = strict_cache
        generate_variant(**gv_kwargs)
        generated = True

    # --- Validation -------------------------------------------------------
    logger.info("[%s] validating variant at %s", knob_id, variant_dir)
    validate_variant(
        domain=domain,
        level=ablation_level,
        stages=stages,
        with_llm=with_llm,
        fusion_input_member=fusion_input_member,
        out_dir=metrics_dir,
        variant_root=variant_dir,
    )

    runtime = time.monotonic() - t0
    logger.info("[%s] ablation complete in %.1fs", knob_id, runtime)
    return {
        "knob_id": knob_id,
        "variant_dir": str(variant_dir),
        "metrics_dir": str(metrics_dir),
        "runtime_s": round(runtime, 2),
        "generated": generated,
    }


def run_ablations(
    domain: str,
    knob_ids: list[str],
    *,
    master_seed: int | None = None,
    with_llm: bool = False,
    stages: list[Stage] | None = None,
    fusion_input_member: str | None = None,
    ablation_level: str = "hard",
    identity_level: str = "easy",
    skip_existing_variant: bool = False,
    continue_on_error: bool = False,
    strict_cache: bool | None = None,
) -> list[dict[str, Any]]:
    """Run ablations for a list of knobs.

    Each entry in the returned list records success/failure per knob.
    On ``continue_on_error=True`` (default off), failures are logged
    and the runner proceeds to the next knob instead of raising.
    """
    results: list[dict[str, Any]] = []
    for kid in knob_ids:
        try:
            result = run_ablation(
                domain,
                kid,
                master_seed=master_seed,
                with_llm=with_llm,
                stages=stages,
                fusion_input_member=fusion_input_member,
                ablation_level=ablation_level,
                identity_level=identity_level,
                skip_existing_variant=skip_existing_variant,
                strict_cache=strict_cache,
            )
            result["status"] = "ok"
            results.append(result)
        except Exception as exc:  # noqa: BLE001 — propagate unless told otherwise
            logger.exception("[%s] ablation failed: %s", kid, exc)
            results.append(
                {
                    "knob_id": kid,
                    "status": "error",
                    "error": repr(exc),
                }
            )
            if not continue_on_error:
                raise
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_stages(raw: str) -> list[Stage]:
    """Parse a comma-separated stages string."""
    valid = {"sm", "em", "fusion"}
    parts = [s.strip() for s in raw.split(",") if s.strip()]
    for s in parts:
        if s not in valid:
            raise ValueError(f"Unknown stage: {s!r}. Valid: {sorted(valid)}")
    return parts  # type: ignore[return-value]


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate per-knob ablation variants and run committee "
            "validation against each. See plans/validation/module_09_ablation.md."
        ),
    )
    parser.add_argument("--domain", required=True, help="Domain name.")
    parser.add_argument(
        "--knobs",
        type=str,
        default=None,
        help=(
            "Comma-separated knob ids to ablate (e.g. '1,8,10'). "
            "Default: all 8 active S1 knobs "
            "(knob_01/02/03/04/05/06/08/10)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Master RNG seed (defaults to domain config master_seed).",
    )
    parser.add_argument(
        "--with-llm",
        action="store_true",
        default=False,
        help="Include LLM committee members. Must match baseline.",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default=None,
        help="Comma-separated validation stages (default: all three).",
    )
    parser.add_argument(
        "--fusion-input-member",
        type=str,
        default=None,
        help="Override EM member feeding Fusion. Default: baseline value.",
    )
    parser.add_argument(
        "--ablation-level",
        type=str,
        default="hard",
        choices=["easy", "medium", "hard"],
        help="Level of the target knob (default: hard).",
    )
    parser.add_argument(
        "--identity-level",
        type=str,
        default="easy",
        choices=["easy", "medium", "hard"],
        help="Level of non-target knobs (default: easy).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=False,
        help=(
            "Skip generation when the packaged variant directory already "
            "exists; only re-run validation."
        ),
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=False,
        help=(
            "Log failures per knob and proceed instead of raising. "
            "Useful for long batches."
        ),
    )
    parser.add_argument(
        "--no-strict-cache",
        action="store_true",
        default=False,
        help=(
            "Allow K1/K2 to make live LLM calls at hard level instead "
            "of requiring a pre-warmed cache. Use on aliased domains "
            "(e.g. companies-small) whose LLM cache is not yet populated."
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.knobs is None:
        knob_ids = list(ACTIVE_KNOB_IDS)
    else:
        knob_ids = _normalise_knob_ids(
            [tok for tok in args.knobs.split(",") if tok.strip()]
        )

    stages: list[Stage] | None = None
    if args.stages is not None:
        stages = _parse_stages(args.stages)

    logger.info(
        "Running ablations for domain=%s over %d knob(s): %s",
        args.domain,
        len(knob_ids),
        knob_ids,
    )

    results = run_ablations(
        domain=args.domain,
        knob_ids=knob_ids,
        master_seed=args.seed,
        with_llm=args.with_llm,
        stages=stages,
        fusion_input_member=args.fusion_input_member,
        ablation_level=args.ablation_level,
        identity_level=args.identity_level,
        skip_existing_variant=args.skip_existing,
        continue_on_error=args.continue_on_error,
        strict_cache=False if args.no_strict_cache else None,
    )

    errors = [r for r in results if r.get("status") == "error"]
    logger.info(
        "Ablation batch complete: %d succeeded, %d failed",
        len(results) - len(errors),
        len(errors),
    )
    for r in results:
        if r.get("status") == "error":
            logger.error("  %s: FAILED (%s)", r["knob_id"], r.get("error", ""))
        else:
            logger.info(
                "  %s: metrics=%s runtime=%.1fs",
                r["knob_id"],
                r["metrics_dir"],
                r.get("runtime_s", 0.0),
            )
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
