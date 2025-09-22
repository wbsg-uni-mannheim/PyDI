import pandas as pd

from PyDI.entitymatching.blocking import TokenBlocking


def test_token_blocking_overlap_and_no_duplicates():
    # Minimal synthetic frames to be deterministic
    df_left = pd.DataFrame({
        "_id": ["L_000001", "L_000002"],
        "title": ["The Lord of the Rings", "Star Wars"]
    })
    df_left.attrs["dataset_name"] = "left"

    df_right = pd.DataFrame({
        "_id": ["R_000001", "R_000002"],
        "title": ["Lord of Rings Trilogy", "Star Trek"]
    })
    df_right.attrs["dataset_name"] = "right"

    tb = TokenBlocking(df_left, df_right, column="title", batch_size=10)
    pairs = tb.materialize()

    # Expect at least one pair due to token overlap on 'lord'/'rings'
    assert not pairs.empty

    # No duplicates
    assert len(pairs) == len(pairs.drop_duplicates())
