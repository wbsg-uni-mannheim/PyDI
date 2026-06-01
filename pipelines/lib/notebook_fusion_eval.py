"""Per-domain ``DataFusionEvaluator`` configurations mirroring the
human-baseline notebooks.

Each ``build_notebook_strategy(domain)`` returns a
:class:`PyDI.fusion.DataFusionStrategy` configured with the SAME
evaluation functions and kwargs the corresponding workflow notebook
uses to score its fused output. This lets the comparison harness
report apples-to-apples per-attribute fusion accuracy: both the
best-of-breed fused frame and the notebook fused frame are scored
under the same rules.

Sources
-------
- products: usecases/products/products_workflow_minimal.ipynb cell 43
- music   : usecases/music/music_workflow.ipynb cell 42
- games   : usecases/games/games_workflow.ipynb cell 59
- companies: usecases/companies/companies_workflow.ipynb cell 51
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

from PyDI.fusion import (
    DataFusionEvaluator,
    DataFusionStrategy,
    exact_match,
    intersection,
    numeric_tolerance_match,
    set_equality_match,
    tokenized_match,
    year_only_match,
)


def hardware_strict_spec_match(fused_value: Any, expected_value: Any) -> bool:
    """Strict numeric-token match for hardware spec strings.

    Verbatim from products_workflow_minimal.ipynb cell 35: prevents
    e.g. 'PCIE x8' from matching 'PCIE x16' by requiring identical
    digit tokens before comparing the alphanumeric remainder.
    """
    if pd.isna(fused_value) or pd.isna(expected_value):
        return False
    f = str(fused_value).lower()
    e = str(expected_value).lower()
    if re.findall(r"\d+", f) != re.findall(r"\d+", e):
        return False
    f_clean = re.sub(r"[^a-z0-9]", "", f)
    e_clean = re.sub(r"[^a-z0-9]", "", e)
    return e_clean in f_clean or f_clean in e_clean


@dataclass(frozen=True)
class NotebookFusionSpec:
    """Per-domain fusion-evaluation spec extracted from the notebook."""

    domain: str
    strategy_name: str
    fused_id_column: str
    gold_id_column: str
    rules: tuple[tuple[str, Callable[..., Any], dict[str, Any]], ...]


_PRODUCTS_RULES = (
    ("brand", exact_match, {}),
    ("product_type", exact_match, {}),
    ("vram_gb", numeric_tolerance_match, {"tolerance": 0.15}),
    ("storage_gb", numeric_tolerance_match, {"tolerance": 0.15}),
    ("read_speed_mb_s", numeric_tolerance_match, {"tolerance": 0.15}),
    ("write_speed_mb_s", numeric_tolerance_match, {"tolerance": 0.15}),
    ("width_mm", numeric_tolerance_match, {"tolerance": 0.15}),
    ("length_mm", numeric_tolerance_match, {"tolerance": 0.15}),
    ("height_mm", numeric_tolerance_match, {"tolerance": 0.15}),
    ("weight_g", numeric_tolerance_match, {"tolerance": 0.15}),
    ("chipset_name", hardware_strict_spec_match, {}),
    ("bus_type", hardware_strict_spec_match, {}),
    ("interface_type", hardware_strict_spec_match, {}),
    ("memory_type", hardware_strict_spec_match, {}),
)

_MUSIC_RULES = (
    ("name", tokenized_match, {}),
    ("artist", tokenized_match, {}),
    ("duration", numeric_tolerance_match, {"tolerance": 10}),
    ("release-date", year_only_match, {}),
    ("release-country", tokenized_match, {}),
    ("label", tokenized_match, {}),
    ("tracks", tokenized_match, {}),
)

_GAMES_RULES = (
    ("name", exact_match, {}),
    ("platform", exact_match, {}),
    ("developer", exact_match, {}),
    ("releaseYear", year_only_match, {}),
    ("ESRB", exact_match, {}),
    ("criticScore", numeric_tolerance_match, {"tolerance": 2}),
    ("userScore", numeric_tolerance_match, {"tolerance": 0.2}),
    # Notebook uses `intersection` (a fuser signature) as evaluation
    # function for genres. Preserved verbatim for fidelity.
    ("genres", intersection, {}),
)

# Companies' notebook registers `assets` twice (tokenized_match then
# numeric_tolerance_match tol=0.1). DataFusionStrategy stores eval
# functions in an attribute->callable dict so the later registration
# wins. We mirror that ordering.
_COMPANIES_RULES = (
    ("name", tokenized_match, {}),
    ("assets", tokenized_match, {}),
    ("revenue", numeric_tolerance_match, {"tolerance": 0.1}),
    ("assets", numeric_tolerance_match, {"tolerance": 0.1}),
    ("founders", set_equality_match, {}),
    ("founded", year_only_match, {}),
    ("country", tokenized_match, {}),
    ("city", tokenized_match, {}),
)

_SPECS: dict[str, NotebookFusionSpec] = {
    "products": NotebookFusionSpec(
        domain="products",
        strategy_name="hardware_fusion_strategy",
        fused_id_column="p1_id",
        gold_id_column="id_left",
        rules=_PRODUCTS_RULES,
    ),
    "music": NotebookFusionSpec(
        domain="music",
        strategy_name="music_fusion_strategy",
        fused_id_column="id",
        gold_id_column="id",
        rules=_MUSIC_RULES,
    ),
    "games": NotebookFusionSpec(
        domain="games",
        strategy_name="game_fusion_strategy",
        fused_id_column="metacritic_id",
        gold_id_column="id",
        rules=_GAMES_RULES,
    ),
    "companies": NotebookFusionSpec(
        domain="companies",
        strategy_name="company_fusion_strategy",
        fused_id_column="forbes_id",
        gold_id_column="id",
        rules=_COMPANIES_RULES,
    ),
}


def get_spec(domain: str) -> NotebookFusionSpec:
    """Return the notebook fusion-eval spec for ``domain``."""
    try:
        return _SPECS[domain]
    except KeyError:
        raise ValueError(
            f"No notebook fusion-eval spec registered for domain {domain!r}. "
            f"Supported: {sorted(_SPECS)}"
        ) from None


def build_notebook_strategy(domain: str) -> DataFusionStrategy:
    """Build the per-domain ``DataFusionStrategy`` with evaluation
    functions matching the workflow notebook.

    Only evaluation functions are registered — fusers are intentionally
    NOT added because this strategy is used solely to score an already
    fused frame.
    """
    spec = get_spec(domain)
    strategy = DataFusionStrategy(spec.strategy_name)
    for attr, fn, kwargs in spec.rules:
        strategy.add_evaluation_function(attr, fn, **kwargs)
    return strategy


def evaluate_with_notebook_strategy(
    fused_df: pd.DataFrame,
    *,
    domain: str,
    gold_df: pd.DataFrame,
    fused_id_column: str | None = None,
    gold_id_column: str | None = None,
    debug_file: Any = None,
) -> dict[str, float]:
    """Run :class:`DataFusionEvaluator` against ``gold_df`` using the
    notebook's per-attribute rules.

    Parameters
    ----------
    fused_df : pd.DataFrame
        The fused frame produced by the pipeline (or the notebook).
    domain : str
        One of ``"products"``, ``"music"``, ``"games"``, ``"companies"``.
    gold_df : pd.DataFrame
        Pre-loaded + pre-processed gold (caller is responsible for
        domain-specific prep — e.g. products' ``filled=='y'`` filter,
        companies' ``keypeople_name -> founders`` rename, games'
        ``genres_genre -> genres`` rename and ``releaseYear`` to
        datetime).
    fused_id_column, gold_id_column : str or None
        Override the per-domain default id-column names (defined in
        the spec).
    debug_file : Path or None
        Optional debug-output path passed through to the evaluator.

    Returns
    -------
    dict
        Per-attribute accuracy + overall_accuracy as returned by
        :meth:`DataFusionEvaluator.evaluate`.
    """
    spec = get_spec(domain)
    strategy = build_notebook_strategy(domain)
    evaluator = DataFusionEvaluator(
        strategy,
        debug=debug_file is not None,
        debug_file=debug_file,
        debug_format="json",
    )
    return evaluator.evaluate(
        fused_df=fused_df,
        fused_id_column=fused_id_column or spec.fused_id_column,
        gold_df=gold_df,
        gold_id_column=gold_id_column or spec.gold_id_column,
    )


__all__ = [
    "NotebookFusionSpec",
    "build_notebook_strategy",
    "evaluate_with_notebook_strategy",
    "get_spec",
    "hardware_strict_spec_match",
]
