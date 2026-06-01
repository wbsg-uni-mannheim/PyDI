"""Fusion scoring helpers for committee validation.

Wraps ``DataFusionEvaluator`` to return a flat metric dict shaped for
:attr:`CommitteeResult.per_attribute`.  Each evaluation function is
resolved by name from the ``PyDI.fusion.evaluation`` module and bound
with optional ``functools.partial`` parameters (e.g. tolerance).
"""

from __future__ import annotations

import importlib
import logging
from functools import partial
from typing import Any, Callable

import pandas as pd

from PyDI.fusion.evaluation import DataFusionEvaluator
from PyDI.fusion.strategy import DataFusionStrategy

logger = logging.getLogger(__name__)

# Registry of evaluation function names → callables.
# Populated lazily from ``PyDI.fusion.evaluation``.
_EVAL_FN_CACHE: dict[str, Callable[..., bool]] = {}


def _resolve_eval_fn(name: str) -> Callable[..., bool]:
    """Resolve an evaluation function by name.

    Looks up ``name`` in ``PyDI.fusion.evaluation``.  The result is
    cached for subsequent calls.

    Parameters
    ----------
    name : str
        Function name (e.g. ``"tokenized_match"``).

    Returns
    -------
    Callable
        The evaluation function.

    Raises
    ------
    AttributeError
        If *name* is not found in the evaluation module.
    """
    if name not in _EVAL_FN_CACHE:
        mod = importlib.import_module("PyDI.fusion.evaluation")
        _EVAL_FN_CACHE[name] = getattr(mod, name)
    return _EVAL_FN_CACHE[name]


def score_fusion(
    fused_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    eval_specs: dict[str, str],
    eval_params: dict[str, dict[str, Any]] | None = None,
    *,
    fused_id_column: str = "_id",
    gold_id_column: str = "id",
) -> dict[str, float]:
    """Evaluate a fused DataFrame against a gold standard.

    Builds a throwaway :class:`DataFusionStrategy` with evaluation
    functions registered per attribute, runs
    :meth:`DataFusionEvaluator.evaluate`, and returns the flat metric
    dict.

    Parameters
    ----------
    fused_df : DataFrame
        Fused output from ``DataFusionEngine.run()``.
    gold_df : DataFrame
        Gold-standard fused records (e.g. ``test_set.xml``).
    eval_specs : dict[str, str]
        ``{attribute: evaluation_function_name}`` — maps each attribute
        to its evaluation function (e.g. ``{"name": "tokenized_match"}``).
    eval_params : dict[str, dict[str, Any]] or None
        Optional per-attribute kwargs to bind via ``functools.partial``
        (e.g. ``{"assets": {"tolerance": 0.1}}``).
    fused_id_column : str
        ID column in the fused DataFrame.  Default ``"forbes_id"``
        (matches the companies workflow).
    gold_id_column : str
        ID column in the gold DataFrame.  Default ``"id"``.

    Returns
    -------
    dict[str, float]
        Flat metric dict with keys ``"overall_accuracy"``,
        ``"macro_accuracy"``, and ``"<attribute>_accuracy"`` per
        evaluated attribute.
    """
    eval_params = eval_params or {}

    strategy = DataFusionStrategy(name="committee_eval")

    for attr, fn_name in eval_specs.items():
        fn = _resolve_eval_fn(fn_name)
        attr_params = eval_params.get(attr, {})
        if attr_params:
            strategy.add_evaluation_function(attr, fn, **attr_params)
        else:
            strategy.add_evaluation_function(attr, fn)

    evaluator = DataFusionEvaluator(strategy)
    results = evaluator.evaluate(
        fused_df=fused_df,
        fused_id_column=fused_id_column,
        expected_df=gold_df,
        expected_id_column=gold_id_column,
    )
    return results
