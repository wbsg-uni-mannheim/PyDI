import os
from typing import List

import pandas as pd
import pytest

from PyDI.io.loaders import load_xml
from PyDI.entitymatching.base import ensure_record_ids
from PyDI.entitymatching.blocking import StandardBlocking


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                        "input", "movies", "entitymatching", "data")


def _load_movies() -> tuple[pd.DataFrame, pd.DataFrame]:
    left = load_xml(os.path.join(
        DATA_DIR, "academy_awards.xml"), name="academy_awards")
    right = load_xml(os.path.join(DATA_DIR, "actors.xml"), name="actors")
    return left, right


def _pick_common_columns(df_left: pd.DataFrame, df_right: pd.DataFrame, prefer: List[str]) -> List[str]:
    common = [c for c in df_left.columns if c in df_right.columns]
    # Exclude id columns
    common = [c for c in common if c not in {
        "_id", "academy_awards_id", "actors_id"}]
    # Prefer known columns if present
    for col in prefer:
        if col in common:
            return [col]
    # Fallback to any non-numeric column
    for col in common:
        if pd.api.types.is_string_dtype(df_left[col]) and pd.api.types.is_string_dtype(df_right[col]):
            return [col]
    # As last resort, return first common
    return common[:1]


@pytest.mark.parametrize("prefer", [["title"], ["name"], ["label"]])
def test_standard_blocking_generates_equal_value_pairs(prefer):
    df_left, df_right = _load_movies()
    # Ensure global ids for consistent indexing
    df_left = ensure_record_ids(df_left)
    df_right = ensure_record_ids(df_right)
    on_cols = _pick_common_columns(df_left, df_right, prefer)
    if not on_cols:
        pytest.skip("No suitable common columns for StandardBlocking")

    blocker = StandardBlocking(df_left, df_right, on=on_cols, batch_size=5000)

    # Collect at most 2 batches to limit runtime
    pairs = []
    for i, batch in enumerate(blocker):
        pairs.append(batch)
        if i >= 1:
            break

    if not pairs:
        pytest.skip(
            "No candidate pairs produced; dataset may lack overlap on chosen columns")

    cands = pd.concat(pairs, ignore_index=True)
    assert set(["id1", "id2"]).issubset(cands.columns)

    # Verify equality on blocking columns for a sample
    left_idx = df_left.set_index("_id")
    right_idx = df_right.set_index("_id")
    sample = cands.head(min(50, len(cands)))
    for _, row in sample.iterrows():
        l = left_idx.loc[row["id1"]]
        r = right_idx.loc[row["id2"]]
        for col in on_cols:
            assert str(l[col]) == str(r[col])
