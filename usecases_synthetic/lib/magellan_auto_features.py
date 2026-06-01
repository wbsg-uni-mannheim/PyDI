"""Auto-feature generation for Magellan-style classical ML matching.

Mirrors ``py_entitymatching.get_features_for_matching`` in spirit: given
two column-mapped source frames (so the attribute equivalence is implicit
in shared canonical column names), emit a comprehensive feature bank.
RandomForest then performs implicit feature selection via
``feature_importances_``.

``py_entitymatching`` itself is not installable on Python 3.12 (its
transitive dep ``py-stringsimjoin==0.3.6`` imports
``distutils.msvccompiler`` which was removed in modern setuptools'
bundled distutils, and no newer release exists). This synth-local helper
provides equivalent functionality over PyDI's comparator stack plus the
synth-local ``lexical_extended_jaccard`` metric.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import pandas as pd

from PyDI.entitymatching.base import BaseComparator
from PyDI.entitymatching.comparators import (
    DateComparator,
    NumericComparator,
    StringComparator,
)

from .niche_metrics import lexical_extended_jaccard

logger = logging.getLogger(__name__)


_STRING_FEATURE_BANK: list[tuple[str, str | None]] = [
    ("jaro_winkler", None),
    ("jaro", None),
    ("levenshtein", None),
    ("damerau_levenshtein", None),
    ("jaccard", "word"),
    ("sorensen_dice", "word"),
    ("cosine", "word"),
    ("monge_elkan", "word"),
    ("overlap", "word"),
    ("jaccard", "ngram_3"),
]

_NUMERIC_FEATURE_BANK: list[tuple[str, float | None]] = [
    ("absolute_difference", None),
    ("relative_difference", 0.05),
    ("relative_difference", 0.10),
    ("relative_difference", 0.20),
]

_DATE_FEATURE_BANK: list[int | None] = [None, 30, 365]


class LexicalExtendedJaccardComparator(BaseComparator):
    """Wrap :func:`niche_metrics.lexical_extended_jaccard` as a BaseComparator.

    Provides Magellan-flavour typo-tolerant token-Jaccard (with inner
    Levenshtein gate) without modifying PyDI's similarity registry. The
    metric is the closeness-contract single source of truth used elsewhere
    in the synthetic pipeline (K2 / K4 / K6 / R5 Norm).
    """

    def __init__(self, column: str, inner_token_threshold: float = 0.8) -> None:
        super().__init__(f"LexicalExtendedJaccard({column}, t={inner_token_threshold})")
        self.column = column
        self.inner_token_threshold = inner_token_threshold

    def compare(self, record1: pd.Series, record2: pd.Series) -> float:
        try:
            val1 = record1[self.column]
            val2 = record2[self.column]
        except KeyError:
            return 0.0
        if val1 is None or val2 is None:
            return 0.0
        if isinstance(val1, float) and pd.isna(val1):
            return 0.0
        if isinstance(val2, float) and pd.isna(val2):
            return 0.0
        try:
            return lexical_extended_jaccard(
                str(val1),
                str(val2),
                inner_token_threshold=self.inner_token_threshold,
            )
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning("LexicalExtendedJaccardComparator error: %s", exc)
            return 0.0


def _infer_attribute_kind(
    series_left: pd.Series,
    series_right: pd.Series,
    *,
    numeric_hints: frozenset[str],
    date_hints: frozenset[str],
    column_name: str,
) -> str:
    """Heuristic attribute kind: ``'numeric'``, ``'date'``, or ``'string'``."""
    if column_name in date_hints:
        return "date"
    if column_name in numeric_hints:
        return "numeric"
    if pd.api.types.is_numeric_dtype(series_left) and pd.api.types.is_numeric_dtype(
        series_right
    ):
        return "numeric"
    return "string"


def auto_generate_comparators(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    *,
    attributes: Sequence[str] | None = None,
    id_columns: Sequence[str] = ("id",),
    numeric_attributes: Sequence[str] = (),
    date_attributes: Sequence[str] = (),
    preprocess_fn: Any | None = None,
    include_lexical_extended_jaccard: bool = True,
) -> list[BaseComparator]:
    """Generate the full Magellan feature bank for given source frames.

    For each attribute shared between ``df_left`` and ``df_right`` (post
    column_mapping), emit one BaseComparator per slot in the per-kind
    feature bank:

    - **String columns** — ~10 comparators across the
      ``_STRING_FEATURE_BANK`` (edit-based at char level; token-based at
      word level; char-ngram3). Plus ``LexicalExtendedJaccardComparator``
      when ``include_lexical_extended_jaccard=True``.
    - **Numeric columns** — 4 comparators
      (``absolute_difference`` + 3 ``relative_difference`` bands at 5 / 10 / 20 %).
    - **Date columns** — 3 ``DateComparator`` slots
      (raw days diff + within-30 + within-365).

    The RandomForest classifier's tree splits perform implicit feature
    selection — high-importance features get used, low-importance ones
    get ignored. This is the Magellan philosophy ported to PyDI's
    comparator stack.

    Parameters
    ----------
    df_left, df_right : pandas.DataFrame
        Column-mapped source frames. Shared canonical column names define
        the attribute equivalence.
    attributes : sequence of str, optional
        Explicit attribute list. Defaults to the intersection of
        ``df_left.columns`` and ``df_right.columns`` minus ``id_columns``.
    id_columns : sequence of str, default ``("id",)``
        Column names to exclude from the attribute set.
    numeric_attributes : sequence of str, default ``()``
        Columns to treat as numeric (overrides dtype inference). Useful
        for numeric columns that ship as object dtype after CSV load.
    date_attributes : sequence of str, default ``()``
        Columns to treat as dates (heuristic does not auto-detect).
    preprocess_fn : callable, optional
        Shared text-preprocessing function injected into every
        ``StringComparator`` under the ``preprocess`` kwarg.
    include_lexical_extended_jaccard : bool, default ``True``
        Whether to also emit the synth-local LexicalExtendedJaccard
        comparator per string column.

    Returns
    -------
    list of BaseComparator
        Comparators ready to feed to
        :class:`PyDI.entitymatching.feature_extraction.FeatureExtractor`.
    """
    if attributes is None:
        shared = set(df_left.columns) & set(df_right.columns)
        attributes = sorted(shared - set(id_columns))

    numeric_set = frozenset(numeric_attributes)
    date_set = frozenset(date_attributes)

    comparators: list[BaseComparator] = []
    for attr in attributes:
        if attr not in df_left.columns or attr not in df_right.columns:
            logger.debug(
                "auto_generate_comparators: skipping %s (not in both frames)",
                attr,
            )
            continue
        kind = _infer_attribute_kind(
            df_left[attr],
            df_right[attr],
            numeric_hints=numeric_set,
            date_hints=date_set,
            column_name=attr,
        )
        if kind == "numeric":
            for method, max_diff in _NUMERIC_FEATURE_BANK:
                params: dict[str, Any] = {"column": attr, "method": method}
                if max_diff is not None:
                    params["max_difference"] = max_diff
                comparators.append(NumericComparator(**params))
        elif kind == "date":
            for max_days in _DATE_FEATURE_BANK:
                params: dict[str, Any] = {"column": attr}
                if max_days is not None:
                    params["max_days_difference"] = max_days
                comparators.append(DateComparator(**params))
        else:
            for sim_func, tokenization in _STRING_FEATURE_BANK:
                params: dict[str, Any] = {
                    "column": attr,
                    "similarity_function": sim_func,
                }
                if tokenization is not None:
                    params["tokenization"] = tokenization
                if preprocess_fn is not None:
                    params["preprocess"] = preprocess_fn
                comparators.append(StringComparator(**params))
            if include_lexical_extended_jaccard:
                comparators.append(LexicalExtendedJaccardComparator(column=attr))

    if not comparators:
        raise ValueError(
            "auto_generate_comparators produced no comparators — verify that "
            "the source frames share at least one non-id attribute"
        )
    logger.info(
        "auto_generate_comparators emitted %d features over %d attributes",
        len(comparators),
        len(list(attributes)),
    )
    return comparators


__all__ = ["auto_generate_comparators", "LexicalExtendedJaccardComparator"]
