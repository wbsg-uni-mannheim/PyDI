"""
Per-column constraint-validity metrics.

Inspired by the KGpipe paper's "semantic" layer (Hofer & Rahm 2025,
arXiv:2511.18364). The PyDI panel previously had no equivalent —
columns were trusted to be well-formed once their dtypes matched.
This module catches cells that *parse* but **violate declared
constraints**: a date outside an allowed range, a country outside
the allowed enum, a numeric value with a wrong unit.

Two layers of check per non-null cell:

1. **Type validity** — does the value parse to the declared
   `column_types` tag? Always computed.
2. **Constraint validity** — does the parsed value satisfy the
   per-column constraints (range / enum / regex / format /
   min-length / max-length / min-size / max-size)? Only computed
   when a constraint is declared for the column.

Output is a per-column dict carrying ``validity_rate`` (fraction
of non-null cells passing all applicable checks), plus the failure
counts so failures can be traced to type-parse vs constraint
violation.

The metric is computed for both pipeline and silver so the delta
surfaces the difference (a low validity rate that's the same on
both sides usually points at the constraint being too strict, not
at the pipeline being broken).
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-cell checks
# ---------------------------------------------------------------------------


def _is_missing(value: Any) -> bool:
    if isinstance(value, (list, tuple, set)):
        return False  # an empty list is a "missing" value handled separately
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _check_type(value: Any, col_type: str) -> bool:
    """Does *value* parse to *col_type*? Type-validity layer."""
    if col_type == "numerical":
        try:
            float(value)
            return True
        except (TypeError, ValueError):
            return False
    if col_type == "datetime":
        try:
            parsed = pd.to_datetime(value, errors="coerce")
        except Exception:
            return False
        return not pd.isna(parsed)
    if col_type == "list":
        return isinstance(value, (list, tuple))
    # categorical, text, identifier — anything not missing is structurally fine
    return True


def _check_constraints(
    value: Any, col_type: str, constraints: Mapping[str, Any]
) -> bool:
    """Constraint-validity layer. Returns True if value passes all checks."""
    if not constraints:
        return True

    # Format (datetime)
    if col_type == "datetime" and "format" in constraints:
        try:
            datetime.strptime(str(value), str(constraints["format"]))
        except (TypeError, ValueError):
            return False

    # Range (numerical / datetime)
    if "range" in constraints:
        lo, hi = constraints["range"]
        if col_type == "numerical":
            try:
                v = float(value)
            except (TypeError, ValueError):
                return False
            if v < float(lo) or v > float(hi):
                return False
        elif col_type == "datetime":
            try:
                v_dt = pd.to_datetime(value)
                lo_dt = pd.to_datetime(lo)
                hi_dt = pd.to_datetime(hi)
            except (
                TypeError,
                ValueError,
                OverflowError,
                pd.errors.OutOfBoundsDatetime,
            ):
                return False
            if v_dt < lo_dt or v_dt > hi_dt:
                return False

    # Enum (categorical)
    if "enum" in constraints:
        allowed = set(constraints["enum"])
        if str(value) not in allowed and value not in allowed:
            return False

    # Regex (text)
    if "regex" in constraints:
        try:
            if not re.search(str(constraints["regex"]), str(value)):
                return False
        except re.error:
            return False

    # Length bounds (text)
    if "min_length" in constraints or "max_length" in constraints:
        n = len(str(value))
        if "min_length" in constraints and n < int(constraints["min_length"]):
            return False
        if "max_length" in constraints and n > int(constraints["max_length"]):
            return False

    # Size bounds (list)
    if "min_size" in constraints or "max_size" in constraints:
        if not isinstance(value, (list, tuple)):
            return False
        n = len(value)
        if "min_size" in constraints and n < int(constraints["min_size"]):
            return False
        if "max_size" in constraints and n > int(constraints["max_size"]):
            return False

    return True


# ---------------------------------------------------------------------------
# Per-column rollup
# ---------------------------------------------------------------------------


def column_validity_rate(
    series: pd.Series,
    col_type: str,
    constraints: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute per-column validity rate for *series*.

    Identifier-typed columns are not validated (any value is fine).
    Empty series return ``validity_rate = 1.0`` with zero counts.

    Returns
    -------
    dict
        ``{validity_rate, parse_failures, constraint_failures,
        n_evaluated}``. ``n_evaluated`` excludes nulls.
    """
    if col_type == "identifier":
        return {
            "validity_rate": 1.0,
            "parse_failures": 0,
            "constraint_failures": 0,
            "n_evaluated": 0,
        }

    constraints = constraints or {}
    parse_failures = 0
    constraint_failures = 0
    n_evaluated = 0

    for value in series:
        if _is_missing(value):
            continue
        n_evaluated += 1
        if not _check_type(value, col_type):
            parse_failures += 1
            continue
        if constraints and not _check_constraints(value, col_type, constraints):
            constraint_failures += 1

    if n_evaluated == 0:
        return {
            "validity_rate": 1.0,
            "parse_failures": 0,
            "constraint_failures": 0,
            "n_evaluated": 0,
        }

    failures = parse_failures + constraint_failures
    return {
        "validity_rate": (n_evaluated - failures) / n_evaluated,
        "parse_failures": parse_failures,
        "constraint_failures": constraint_failures,
        "n_evaluated": n_evaluated,
    }


def compare_column_validity(
    pipe_df: pd.DataFrame,
    silver_df: pd.DataFrame,
    column_types: Mapping[str, str],
    column_constraints: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Compute validity rate per column on both sides and return the delta.

    Plain-language question: "For each column, what fraction of
    non-null cells parse to the declared type and satisfy the
    declared constraints — and how does that compare to silver?"

    A low pipe validity rate with a high silver validity rate is a
    pipeline regression (the pipeline emitted invalid cells silver
    has clean). A low rate on both sides means the constraint is
    too strict for the dataset, not a pipeline bug.

    Parameters
    ----------
    pipe_df, silver_df : DataFrame
        Pipeline-fused output and silver reference.
    column_types : mapping
        Per-column type tags. Identifier-typed columns are skipped.
    column_constraints : mapping, optional
        Per-column constraint dicts. Keys ``range`` /
        ``enum`` / ``regex`` / ``format`` / ``min_length`` /
        ``max_length`` / ``min_size`` / ``max_size`` are recognised.
        When ``None`` or column missing, only the type-validity
        layer is checked.

    Returns
    -------
    dict
        ``{column: {validity_rate_pipe, validity_rate_reference, delta,
        parse_failures_pipe, parse_failures_reference,
        constraint_failures_pipe, constraint_failures_reference,
        n_evaluated_pipe, n_evaluated_reference}}``.
    """
    column_constraints = column_constraints or {}
    out: Dict[str, Dict[str, Any]] = {}

    shared = [c for c in pipe_df.columns if c in silver_df.columns]

    for column in shared:
        col_type = column_types.get(column)
        if col_type is None or col_type == "identifier":
            continue
        constraints = column_constraints.get(column)
        pipe = column_validity_rate(pipe_df[column], col_type, constraints)
        silver = column_validity_rate(silver_df[column], col_type, constraints)
        out[column] = {
            "validity_rate_pipe": pipe["validity_rate"],
            "validity_rate_reference": silver["validity_rate"],
            "delta": pipe["validity_rate"] - silver["validity_rate"],
            "parse_failures_pipe": pipe["parse_failures"],
            "parse_failures_reference": silver["parse_failures"],
            "constraint_failures_pipe": pipe["constraint_failures"],
            "constraint_failures_reference": silver["constraint_failures"],
            "n_evaluated_pipe": pipe["n_evaluated"],
            "n_evaluated_reference": silver["n_evaluated"],
        }
    return out


def mean_validity_delta(
    per_column: Mapping[str, Mapping[str, Any]],
) -> float:
    """Mean of `validity_rate_pipe - validity_rate_reference` across columns.

    Used by the composite scorer to penalise pipelines that introduce
    constraint violations silver doesn't have. Returns 0.0 on empty
    input.
    """
    deltas = [
        v["delta"]
        for v in per_column.values()
        if "delta" in v and v.get("n_evaluated_pipe", 0) > 0
    ]
    if not deltas:
        return 0.0
    return float(np.mean(deltas))
