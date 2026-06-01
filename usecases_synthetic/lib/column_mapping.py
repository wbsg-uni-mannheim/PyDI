"""Shared helper for applying ``column_mapping`` rename specs.

The committee YAMLs declare per-source ``column_mapping`` to normalise
heterogeneous source columns onto the canonical schema (e.g. musicbrainz
``rel_id`` → ``id``, ``title`` → ``name``). A plain
``DataFrame.rename(columns=...)`` produces duplicate column labels when
a rename target already exists in the frame (e.g. musicbrainz carries
both an auto-extracted XML ``id`` and a separate ``rel_id`` that the
YAML wants surfaced as ``id``). Subsequent ``df[col]`` accesses then
return a 2-column frame and break downstream code -- the EM committee
runner and the fusion engine both crash on this. This helper drops any
pre-existing target columns first, so the post-rename frame always has
unique column labels.
"""

from __future__ import annotations

import pandas as pd


def apply_column_mapping(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Apply ``rename(columns=mapping)`` while resolving collisions.

    Parameters
    ----------
    df : DataFrame
        Source DataFrame.
    mapping : dict[str, str]
        Per-column rename map (``{src_col: tgt_col}``).

    Returns
    -------
    DataFrame
        New DataFrame with ``mapping`` applied. Columns whose post-
        rename label would collide with an existing column are dropped
        first so the output has unique labels.
    """
    drop_cols: list[str] = []
    for src, tgt in mapping.items():
        if src == tgt:
            continue
        if tgt in df.columns and tgt != src:
            drop_cols.append(tgt)
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df.rename(columns=mapping)
