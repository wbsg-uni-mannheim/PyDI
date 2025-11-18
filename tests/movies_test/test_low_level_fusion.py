import pytest
import pandas as pd
from PyDI.fusion import DataFusionEngine
from PyDI.fusion import (DataFusionStrategy,
    longest_string, shortest_string, most_complete,
    # Numeric functions
    average, median, maximum, minimum, sum_values,
    # Date functions
    most_recent, earliest,
    # List functions
    union, intersection, intersection_k_sources,
    # General functions
    voting, favour_sources, random_value, weighted_voting, prefer_higher_trust,
)

from PyDI.fusion import (
    exact_match,
    tokenized_match,
    year_only_match,
    numeric_tolerance_match,
    set_equality_match,
    boolean_match,
)

@pytest.fixture
def load_test_input(get_test_input):
    actors = get_test_input("movies", "actors")
    academy_awards = get_test_input("movies", "academy_awards")
    golden_globes = get_test_input("movies", "golden_globes")
    return actors, academy_awards, golden_globes

@pytest.fixture
def load_test_correspondences(get_test_correspondences):
    return (
        get_test_correspondences("movies", "academy_awards", "actors"),
        get_test_correspondences("movies", "actors", "golden_globes")
    )

def test_run_fusion(load_test_input, load_test_correspondences, get_test_fusion_goldstandard):
    actors, academy_awards, golden_globes = load_test_input
    academy_awards["academy_awards_id"] = academy_awards["id"]
    academy_awards.attrs["trust_score"] = 3
    actors.attrs["trust_score"] = 2
    golden_globes.attrs["trust_score"] = 1

    aa2a, a2g = load_test_correspondences
    all_correspondences = pd.concat([a2g, aa2a], ignore_index=True)

    strategy = DataFusionStrategy('movie_fusion_strategy')
    strategy.add_attribute_fuser('title', prefer_higher_trust, trust_key="trust_score")
    strategy.add_attribute_fuser('director_name', longest_string)
    strategy.add_attribute_fuser('date', prefer_higher_trust, trust_key="trust_score")
    strategy.add_attribute_fuser('actors_actor_name', union)
    
    engine = DataFusionEngine(strategy)

    fused = engine.run(
        datasets=[academy_awards, actors, golden_globes],
        correspondences=all_correspondences,
        id_column="id",
        include_singletons=False,
    )

    gs_test = get_test_fusion_goldstandard("movies")

    assert len(fused) == len(gs_test)

    fused["id"] = fused["academy_awards_id"]
    merged = pd.merge(
    fused, gs_test, on="id", suffixes=("_fused", "_gs")
    )

    for col in ["title", "director_name", "date", "actors_actor_name"]:
        f = merged[f"{col}_fused"]
        g = merged[f"{col}_gs"]

        def equal(x, y):
            if isinstance(x, list) and isinstance(y, list):
                return sorted(x) == sorted(y)
            if pd.isna(x) and pd.isna(y):
                return True
            return x == y

        matches = [equal(a, b) for a, b in zip(f, g)]
        assert all(matches), f"Mismatch found in column '{col}'"

def test_longest_string_fuser():
    values = ["A", "The Longest Title", None, "Short"]
    result = longest_string(values)
    selected = result[0]
    assert selected == "The Longest Title"

def test_shortest_string_fuser():
    values = ["A", "The Longest Title", None, "Short"]
    result = shortest_string(values)
    selected = result[0]
    assert selected == "A"

def test_union_fuser():
    a = ["Alice", "Bob"]
    b = ["Bob", "Charlie"]
    result = union([a, b])
    assert set(result[0]) == {"Alice", "Bob", "Charlie"}

def test_most_complete_string():
    values = ["A", "The Completest Title", None, "Short"]
    result = most_complete(values)
    selected = result[0]
    assert selected == "The Completest Title"

def test_numeric_fusers():
    vals = [1, 3, 5, "6", None]
    avg, c1, _ = average(vals)
    med, c2, _ = median(vals)
    mx, c3, _ = maximum(vals)
    mn, c4, _ = minimum(vals)
    sm, c5, _ = sum_values(vals)
    assert avg == pytest.approx(3.75)
    assert med == 4.0
    assert mx == 6.0
    assert mn == 1.0
    assert sm == 15.0
    assert all(0.0 <= c <= 1.0 for c in [c1, c2, c3, c4, c5])

def test_date_fusers():
    # most_recent and earliest accept heterogeneous date-like inputs
    vals = ["2020-01-01", "2019-12-31", pd.Timestamp("2021-06-01")]
    mr, c1, _ = most_recent(vals)
    er, c2, _ = earliest(vals)
    assert str(mr) == "2021-06-01 00:00:00"
    assert str(er) == "2019-12-31"
    assert c1 <= 1.0 and c2 <= 1.0

def test_list_intersections():
    vals = [["a", "b", "c"], ["b", "c", "d"], ["b", "x"]]
    inter, ci, _ = intersection(vals)
    assert inter == ["b"]
    interk, ck, meta = intersection_k_sources(vals, k=2)
    assert set(interk) == {"b", "c"}
    assert meta["item_counts"]["b"] == 3

def test_general_voting_rules():
    vals = ["x", "y", "x", None]
    v, cv, _ = voting(vals)
    assert v == "x" and 0.0 <= cv <= 1.0

    wf, cw, _ = weighted_voting(["a", "b", "a", "c"], weights=[1, 2, 1, 5])
    assert wf == "c"

    # favour_sources respects preferred ordering of sources
    values = ["A", "B", "C"]
    sources = ["s1", "s2", "s3"]
    fav, cf, _ = favour_sources(values, source_preferences=["s3", "s1"], sources=sources)
    assert fav == "C"

    # random_value returns one of the inputs and stable with seed
    r1, _, _ = random_value(["u", "v", "w"], seed=42)
    r2, _, _ = random_value(["u", "v", "w"], seed=42)
    assert r1 == r2 and r1 in {"u", "v", "w"}

def test_prefer_higher_trust():
    values = ["LowTrustValue", "HighTrustValue", None]
    sources = ["record_low", "record_high", "record_mid"]
    source_datasets = {
        "record_low": "dataset_low",
        "record_high": "dataset_high",
        "record_mid": "dataset_mid",
    }
    trust_map = {"dataset_low": 1.0, "dataset_high": 3.0, "dataset_mid": 2.0}

    result = prefer_higher_trust(
        values,
        sources=sources,
        source_datasets=source_datasets,
        trust_map=trust_map,
        trust_key="trust",
    )

    selected = result[0]
    confidence = result[1]

    assert selected == "HighTrustValue"
    assert confidence == 1.0

def test_evaluation_matchers():
    # exact
    assert exact_match("A", "A")
    assert not exact_match("A", "B")

    # tokenized_match: strings and lists
    assert tokenized_match("The Lord of the Rings", "Lord Rings The", threshold=0.6)
    assert not tokenized_match("Alpha Beta", "Gamma Delta", threshold=0.9)
    assert tokenized_match(["a", "b"], ["b", "a"], threshold=1.0)

    # year_only_match: diverse inputs
    assert year_only_match("2020-05-01", pd.Timestamp("2020-12-31"))
    assert not year_only_match("2019-01-01", "2020-01-01")

    # numeric_tolerance_match
    assert numeric_tolerance_match(10.0, 10.005, tolerance=0.01)
    assert not numeric_tolerance_match(10.0, 10.2, tolerance=0.01)

    # set_equality_match
    assert set_equality_match([1, 2], [2, 1])
    assert not set_equality_match([1, 2], [1, 2, 3])

    # boolean_match with varied representations
    assert boolean_match(True, "true")
    assert boolean_match("Yes", 1)
    assert boolean_match("no", 0) 
    assert not boolean_match(True, False)

