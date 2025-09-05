import os
import pandas as pd
import pytest

from PyDI.io.loaders import load_xml
from PyDI.entitymatching.blocking import SortedNeighbourhood, NoBlocking


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                        "input", "movies", "entitymatching", "data")


def _load_movies() -> tuple[pd.DataFrame, pd.DataFrame]:
    left = load_xml(os.path.join(
        DATA_DIR, "academy_awards.xml"), name="academy_awards")
    right = load_xml(os.path.join(DATA_DIR, "actors.xml"), name="actors")
    return left, right


def _pick_text_key(df_left: pd.DataFrame, df_right: pd.DataFrame) -> str | None:
    for col in ["title", "name", "label", "actor_name", "director_name"]:
        if col in df_left.columns and col in df_right.columns:
            if pd.api.types.is_string_dtype(df_left[col]) and pd.api.types.is_string_dtype(df_right[col]):
                return col
    # fallback to any common string column
    common = [c for c in df_left.columns if c in df_right.columns]
    for col in common:
        if pd.api.types.is_string_dtype(df_left[col]) and pd.api.types.is_string_dtype(df_right[col]):
            return col
    return None


def test_sorted_neighbourhood_produces_less_pairs_than_cartesian():
    df_left, df_right = _load_movies()
    key = _pick_text_key(df_left, df_right)
    if not key:
        pytest.skip("No suitable common text key for SortedNeighbourhood")

    no_block = NoBlocking(df_left, df_right, batch_size=50_000)
    all_pairs_est = no_block.estimate_pairs() or (len(df_left) * len(df_right))

    sn = SortedNeighbourhood(df_left, df_right, key=key,
                             window=5, batch_size=50_000)
    emitted = 0
    for batch in sn:
        emitted += len(batch)
        if emitted > 100_000:  # early stop; enough evidence
            break

    assert emitted < all_pairs_est
