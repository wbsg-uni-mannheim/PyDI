"""Pipeline state container for the best-of-breed orchestrator.

Wraps an existing :class:`usecases_synthetic.lib.variant_loader.VariantBundle`
and threads each stage's winner output through the pipeline. The
underlying ``VariantBundle`` carries the immutable inputs (source
DataFrames, SM gold, EM gold splits, fusion XML silvers); the
``PipelineState`` adds the mutable per-stage outputs (winner predictions
+ chained intermediates).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from usecases_synthetic.lib.variant_loader import VariantBundle, load_variant

logger = logging.getLogger(__name__)


@dataclass
class PipelineState:
    """Mutable state threaded through the best-of-breed pipeline.

    Parameters
    ----------
    bundle : VariantBundle
        The underlying baseline bundle. Read-only inputs.
    sm_winner : str
        Name of the SM committee member chosen for this run.
    sm_mapping_df : pandas.DataFrame
        The winning SM mapping (``source_dataset``, ``source_column``,
        ``target_dataset``, ``target_column``, ``score``).
    norm_winner : str
        Name of the Norm committee member chosen (informational only
        in v1 — Norm does not transform downstream frames here).
    blocker_winner_per_pair : dict
        ``{pair_key: blocker_name}`` — picked per source-pair by the
        EM blocking committee composition logic.
    matcher_winner : str
        Name of the EM matcher chosen for this run.
    matcher_predictions : dict
        ``{pair_key: DataFrame[id1, id2, score]}`` — the winning
        matcher's post-clustering predictions per pair.
    refinement_winner : str
        Name of the chosen refinement method (``baseline`` /
        ``greedy`` / ``mbm``).
    correspondences : pandas.DataFrame
        Concatenated post-refinement correspondences across all pairs
        (``id1``, ``id2``, ``score``, ``notes``). Input to fusion.
    fusion_winner : str
        Name of the chosen fusion committee member.
    fused : pandas.DataFrame
        Final fused output from the winning fusion strategy.
    """

    bundle: VariantBundle

    sm_winner: str = ""
    sm_mapping_df: pd.DataFrame | None = None

    norm_winner: str = ""

    blocker_winner_per_pair: dict[str, str] = field(default_factory=dict)
    matcher_winner: str = ""
    matcher_predictions: dict[str, pd.DataFrame] = field(default_factory=dict)

    refinement_winner: str = ""
    correspondences: pd.DataFrame | None = None

    fusion_winner: str = ""
    fused: pd.DataFrame | None = None


def load_pipeline_bundle(
    domain: str,
    *,
    level: str = "baseline",
    bundle_source: str = "synthetic_baseline",
) -> VariantBundle:
    """Load a baseline ``VariantBundle`` for the best-of-breed pipeline.

    Parameters
    ----------
    domain : str
        Domain name (e.g. ``"products"``, ``"music"``).
    level : str, default ``"baseline"``
        Bundle level. The best-of-breed pipeline only consumes
        ``"baseline"`` (original data, not augmented variants).
    bundle_source : str, default ``"synthetic_baseline"``
        Which physical data tree to load from.

        - ``"synthetic_baseline"``: route through
          :func:`usecases_synthetic.lib.variant_loader.load_variant`.
          For domains with a synthetic ``data_root`` override
          (currently only products) this lands on
          ``usecases_synthetic/usecases/<domain>/``; for others it
          lands on canonical ``usecases/<domain>/``.
        - ``"canonical"``: read directly from canonical
          ``usecases/<domain>/`` regardless of any synthetic
          ``data_root`` override. Implemented for products via
          :func:`pipelines.lib.canonical_loader.load_canonical_products_bundle`;
          for music / games / companies this is equivalent to
          ``synthetic_baseline`` because no override is in effect.

    Returns
    -------
    VariantBundle
        Loaded bundle.
    """
    # Synthetic-side variant_loader.VALID_BUNDLE_LEVELS =
    # ['baseline','easy','medium','hard']. The canonical_loader path
    # currently only handles baseline; pass any non-baseline level
    # through to load_variant (which serves the usecases/<domain>-
    # augmented/<level>/ tree).
    if level not in {"baseline", "easy", "medium", "hard"}:
        raise ValueError(
            f"Unknown level {level!r}; expected one of "
            "{baseline, easy, medium, hard}."
        )
    if bundle_source not in {"synthetic_baseline", "canonical"}:
        raise ValueError(
            f"Unknown bundle_source {bundle_source!r}; "
            "expected 'synthetic_baseline' or 'canonical'."
        )

    if level == "baseline" and bundle_source == "canonical" and domain == "products":
        from .canonical_loader import load_canonical_products_bundle

        logger.info("Loading %s canonical baseline bundle (products-specific)", domain)
        return load_canonical_products_bundle()

    if level == "baseline" and domain == "papers":
        # ``papers`` is not registered in usecases_synthetic's
        # VALID_DOMAINS; load_variant would raise. The canonical
        # loader is the only path for now (variants not yet generated).
        from .canonical_loader import load_canonical_papers_bundle

        logger.info("Loading papers canonical bundle (no synthetic-side variant)")
        return load_canonical_papers_bundle()

    if level != "baseline" and domain == "papers":
        # ``usecases/papers-augmented/{easy,medium,hard}/`` IS now on disk
        # (2026-06-04), and ``variant_loader.load_variant`` has explicit
        # papers handling (``variant_loader.py:381``). The packaged fusion
        # gold uses the canonical ``test_set.xml``/``validation_set.xml``
        # filenames but ships JSONL content; ``_load_fusion_file``
        # content-sniffs the first non-whitespace byte to dispatch
        # correctly regardless of extension. Route through the standard
        # variant path.
        logger.info(
            "Loading papers %s variant via load_variant "
            "(canonical_loader supports baseline only)",
            level,
        )
    if level != "baseline" and domain == "products" and bundle_source == "canonical":
        # Variants of products go through the standard load_variant path
        # against usecases/products-augmented/<level>/.
        logger.info(
            "Loading products %s variant via load_variant "
            "(canonical_loader supports baseline only)",
            level,
        )

    logger.info(
        "Loading %s %s bundle via load_variant (bundle_source=%s)",
        domain,
        level,
        bundle_source,
    )
    return load_variant(domain, level=level)


__all__ = ["PipelineState", "load_pipeline_bundle"]
