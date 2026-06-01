"""Baseline missingness measurement.

Shared utility consumed by Knobs 3, 6, and 10. Measures per-(source,
attribute) null rates from the current source DataFrames. The baseline
is always computed fresh — never cached across runs.
"""

from __future__ import annotations

import pandas as pd


def measure_missingness(
    sources: dict[str, pd.DataFrame],
    managed_columns: dict[str, list[str]],
) -> dict[str, dict[str, float]]:
    """Measure per-(source, attribute) null fraction for managed columns.

    Parameters
    ----------
    sources : dict[str, DataFrame]
        Source DataFrames keyed by source name.
    managed_columns : dict[str, list[str]]
        Per-source list of columns to measure. Columns not in the source
        DataFrame are silently skipped.

    Returns
    -------
    dict[str, dict[str, float]]
        Nested dict ``{source: {column: null_rate}}``.
    """
    result: dict[str, dict[str, float]] = {}
    for source_name, df in sources.items():
        cols = managed_columns.get(source_name)
        if cols is None:
            continue
        rates: dict[str, float] = {}
        for col in cols:
            if col in df.columns:
                rates[col] = float(df[col].isna().mean())
        result[source_name] = rates
    return result


def baseline_to_dataframe(
    baseline: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """Convert a baseline dict to a flat DataFrame for serialisation.

    Parameters
    ----------
    baseline : dict[str, dict[str, float]]
        Output of :func:`measure_missingness`.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``source``, ``attribute``, ``null_rate``.
    """
    rows: list[dict[str, str | float]] = []
    for source, cols in sorted(baseline.items()):
        for col, rate in sorted(cols.items()):
            rows.append({"source": source, "attribute": col, "null_rate": rate})
    return pd.DataFrame(rows, columns=["source", "attribute", "null_rate"])
