import pytest
from datetime import datetime
import pandas as pd
from PyDI.entitymatching import (
    StandardBlocker,
    NoBlocker,
    StringComparator,
    DateComparator,
    RuleBasedMatcher,
)

@pytest.fixture
def movie_dataframes():
    df1 = pd.DataFrame([
        {"id": "movie1", "title": "Star Wars IV", "director": "George Lucas", "date": datetime(1977, 5, 25)},
        {"id": "movie2", "title": "Star Wars V", "director": "Irvin Kershner", "date": datetime(1980, 5, 21)},
    ])
    df2 = pd.DataFrame([
        {"id": "movie3", "title": "Star Wars IV", "director": "Irvin Kershner", "date": datetime(1977, 5, 25)},
        {"id": "movie4", "title": "Star Wars IV", "director": "George Lucas", "date": datetime(1977, 5, 25)},
    ])
    return df1, df2

@pytest.fixture
def assert_correspondence_score():
    def _assert(df, id1, id2, expected):
        row = df.query("id1 == @id1 and id2 == @id2")
        assert not row.empty, f"Missing correspondence for ({id1}, {id2})"
        actual = row.iloc[0]["score"]
        assert actual == pytest.approx(expected, rel=1e-6), f"Expected {expected}, got {actual}"
    return _assert

def test_data_import(return_input_data):
    actors = return_input_data("movies", "actors")
    golden_globes = return_input_data("movies", "golden_globes")
    academy_awards = return_input_data("movies", "academy_awards")

    assert not actors.empty
    assert not golden_globes.empty
    assert not academy_awards.empty

def test_run_standard_blocking(return_input_data):
    actors_df = return_input_data("movies", "actors")
    academy_df = return_input_data("movies", "academy_awards")

    actors_df['title_prefix'] = actors_df['title'].astype(str).apply(lambda x: ''.join([word[:2].upper() for word in x.split()[:3]]))
    academy_df['title_prefix'] = academy_df['title'].astype(str).apply(lambda x: ''.join([word[:2].upper() for word in x.split()[:3]]))

    standard_blocker = StandardBlocker(
        actors_df, academy_df,
        on=['title_prefix'],
        batch_size=1000,
        id_column='id'
    )
    
    standard_blocker.materialize()

def test_run_matching(return_input_data):
    actors_df = return_input_data("movies", "actors")
    academy_df = return_input_data("movies", "academy_awards")

    actors_df['title_prefix'] = actors_df['title'].astype(str).apply(lambda x: ''.join([word[:2].upper() for word in x.split()[:3]]))
    academy_df['title_prefix'] = academy_df['title'].astype(str).apply(lambda x: ''.join([word[:2].upper() for word in x.split()[:3]]))

    standard_blocker = StandardBlocker(
        actors_df, academy_df,
        on=['title_prefix'],
        batch_size=1000,
        id_column='id'
    )
    
    comparators = [
        StringComparator('title', 'jaccard', preprocess=str.lower),
        DateComparator('date', max_days_difference=365),
        StringComparator('actors_actor_name', 'jaccard', preprocess=str.lower, list_strategy='concatenate')
    ]

    matcher = RuleBasedMatcher()

    correspondences = matcher.match(
        df_left=actors_df,
        df_right=academy_df, 
        candidates=standard_blocker,
        comparators=comparators,
        weights=[0.7, 0.2, 0.1],
        threshold=0.7,
        id_column='id'
    )

def test_matcher_title_only(movie_dataframes):
    df1, df2 = movie_dataframes
    matcher = RuleBasedMatcher()
    comparator = StringComparator("title", "levenshtein")
    blocker = NoBlocker(df_left=df1, df_right=df2, id_column='id')
    correspondences = matcher.match(df_left=df1, df_right=df2, candidates=blocker, comparators=[comparator], 
                                    threshold=1.0, id_column='id')
    expected_ids = {("movie1", "movie3"), ("movie1", "movie4")}
    actual_ids = set((row["id1"], row["id2"]) for _, row in correspondences.iterrows())
    assert actual_ids == expected_ids

def test_matcher_title_director(movie_dataframes):
    df1, df2 = movie_dataframes
    matcher = RuleBasedMatcher()
    comparator_title = StringComparator("title", "levenshtein")
    comparator_director = StringComparator("director", "levenshtein")
    comparators = [comparator_title, comparator_director]
    blocker = NoBlocker(df_left=df1, df_right=df2, id_column='id')
    correspondences = matcher.match(df_left=df1, df_right=df2, candidates=blocker, comparators=comparators, 
                                    weights=[0.1, 0.9], threshold=1.0, id_column='id')
    expected_ids = {("movie1", "movie4")}
    actual_ids = set((row["id1"], row["id2"]) for _, row in correspondences.iterrows())
    assert actual_ids == expected_ids

def test_matcher_title_director_date(movie_dataframes):
    df1, df2 = movie_dataframes
    matcher = RuleBasedMatcher()
    comparators = [
        StringComparator("title", "levenshtein"),
        StringComparator("director", "levenshtein"),
        DateComparator("date", max_days_difference=365*2),
    ]
    blocker = NoBlocker(df_left=df1, df_right=df2, id_column='id')

    correspondences = matcher.match(
        df_left=df1,
        df_right=df2,
        candidates=blocker,
        comparators=comparators,
        weights=[0.5, 0.25, 0.25],
        threshold=0.75,
        id_column='id',
    )
    expected_ids = {("movie1", "movie3"), ("movie1", "movie4")}
    actual_ids = set((row["id1"], row["id2"]) for _, row in correspondences.iterrows())
    assert actual_ids == expected_ids

def test_matcher_normalization(movie_dataframes, assert_correspondence_score):
    import inspect
    print(inspect.getfile(RuleBasedMatcher))
    df1, df2 = movie_dataframes
    matcher = RuleBasedMatcher()
    comparators = [
        StringComparator("title", "levenshtein"),
        StringComparator("director", "levenshtein"),
        DateComparator("date", max_days_difference=365*2),
    ]
    blocker = NoBlocker(df_left=df1, df_right=df2, id_column='id')

    correspondences = matcher.match(
        df_left=df1,
        df_right=df2,
        candidates=blocker,
        comparators=comparators,
        weights=[2.0, 1.0, 1.0],
        threshold=0.75,
        id_column='id',
    )

    assert_correspondence_score(correspondences, "movie1", "movie3", 0.767857)
    assert_correspondence_score(correspondences, "movie1", "movie4", 1.0)