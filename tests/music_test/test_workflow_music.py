import numpy as np
import torch
import pytest
import pandas as pd
import re

from PyDI.entitymatching import EntityMatchingEvaluator
from PyDI.entitymatching import (
    EmbeddingBlocker,
    StandardBlocker,
    StringComparator,
    DateComparator,
    NumericComparator,
    RuleBasedMatcher,
    MaximumBipartiteMatching,
)
from PyDI.fusion import (
    DataFusionStrategy, 
    DataFusionEngine,
    DataFusionEvaluator, 
    longest_string, 
    shortest_string,
    voting,
    maximum,
    union, 
    prefer_higher_trust, 
    tokenized_match, 
    year_only_match, 
    boolean_match,
    numeric_tolerance_match,
    set_equality_match
)
torch.manual_seed(42)
np.random.seed(42)
torch.use_deterministic_algorithms(True)

def test_run_music(get_input_data, get_correspondences, get_fusion_test_set):
    discogs = get_input_data("music", "discogs")
    mbrainz = get_input_data("music", "musicbrainz")
    lastfm = get_input_data("music", "lastfm")

    def get_longest_token(name):
        tokens = re.split(r"[^A-Za-z0-9_']+", str(name))    
        tokens = [t for t in tokens if t]
        return max(tokens, key=len) if tokens else ''

    mbrainz['name_longest_token'] = mbrainz['name'].apply(get_longest_token)
    discogs['name_longest_token'] = discogs['name'].apply(get_longest_token)
    lastfm['name_longest_token'] = lastfm['name'].apply(get_longest_token)
    
    #### Blocking ####

    standard_blocker_m2d = StandardBlocker(
        mbrainz, discogs,
        on=['name_longest_token'],
        batch_size=1000,
        id_column='id'
    )
    standard_candidates_m2d = standard_blocker_m2d.materialize()

    standard_blocker_m2l = StandardBlocker(
        mbrainz, lastfm,
        on=['name_longest_token'],
        batch_size=1000,
        id_column='id'
    )
    standard_candidates_m2l = standard_blocker_m2l.materialize()

    assert len(standard_candidates_m2l) == 154209, f"Expected 154209 candidates, got {len(standard_candidates_m2l)}"

    #### Matching ####

    def normalize_text(s: str) -> str: 
        if s is None:
            return ""
        return re.sub(r"[^\w\s]|_", "", s).lower()

    comparators = [
        StringComparator(column='name', similarity_function='jaccard', preprocess=normalize_text),
        StringComparator(column='artist', similarity_function='jaccard', preprocess=normalize_text),
        DateComparator(column='release-date', max_days_difference=365 * 2),
        StringComparator(column='release-country', similarity_function='jaccard', preprocess=normalize_text),
        NumericComparator(column='duration', method='relative_difference', max_difference=0.10),
        StringComparator(column='tracks_track_name', similarity_function='jaccard', preprocess=normalize_text, list_strategy="set_overlap")
    ]

    def sum_duration(val):
        if isinstance(val, list):
            return int(np.nansum([int(x) for x in val if str(x).isdigit()]))
        try:
            return int(val)
        except Exception:
            return np.nan

    mbrainz["duration"] = mbrainz["duration"].apply(sum_duration)

    matcher = RuleBasedMatcher()

    correspondences_m2d = matcher.match(
        df_left=mbrainz,
        df_right=discogs, 
        candidates=standard_candidates_m2d,
        comparators=comparators,
        weights=None,
        threshold=0.5,
        id_column='id'
    )
    assert len(correspondences_m2d) == 3886, f"Expected 3886 correspondences, got {len(correspondences_m2d)}"

    correspondences_m2l = matcher.match(
        df_left=mbrainz,
        df_right=lastfm, 
        candidates=standard_candidates_m2l,
        comparators=comparators,
        weights=None,
        threshold=0.5,
        id_column='id',
    )
    assert len(correspondences_m2l) == 1415, f"Expected 1415 correspondences, got {len(correspondences_m2l)}"

    test_gt_m2d = get_correspondences("music", "musicbrainz", "discogs")
    eval_results = EntityMatchingEvaluator.evaluate_matching(
        correspondences_m2d,
        test_gt_m2d,
        out_dir=None
    )
    assert eval_results['accuracy'] == pytest.approx(0.967, abs=0.01), f"Expected accuracy == 0.967 +/- 0.01, got {eval_results['accuracy']}"

    clusterer = MaximumBipartiteMatching()
    correspondences_m2d = clusterer.cluster(correspondences_m2d)
    assert len(correspondences_m2d) == 3051, f"Expected 3051 correspondences after MBM, got {len(correspondences_m2d)}"

    #### Fusion ####

    all_correspondences = pd.concat([correspondences_m2d, correspondences_m2l], ignore_index=True)
    assert len(all_correspondences) == 4466, f"Expected 4466 total correspondences, got {len(all_correspondences)}"

    mbrainz["mbrainz_id"] = mbrainz["id"]

    mbrainz.attrs["trust_score"] = 1
    discogs.attrs["trust_score"] = 2
    lastfm.attrs["trust_score"] = 2

    strategy = DataFusionStrategy('music_fusion_strategy')
    strategy.add_attribute_fuser('name', shortest_string)
    strategy.add_attribute_fuser('artist', longest_string)
    strategy.add_attribute_fuser('release-date', voting)
    strategy.add_attribute_fuser('release-country', longest_string)
    strategy.add_attribute_fuser('duration', maximum)
    strategy.add_attribute_fuser('tracks_track_name', union)
    strategy.add_attribute_fuser('label', longest_string)

    engine = DataFusionEngine(strategy)

    fused = engine.run(
        datasets=[mbrainz, discogs, lastfm],
        correspondences=all_correspondences,
        id_column="id",
        include_singletons=False,
    )
    assert len(fused) == 3662, f"Expected 3662 fused records, got {len(fused)}"

    strategy.add_evaluation_function("name", tokenized_match)
    strategy.add_evaluation_function("artist", tokenized_match)
    strategy.add_evaluation_function("duration", numeric_tolerance_match)
    strategy.add_evaluation_function("release-date", year_only_match)
    strategy.add_evaluation_function("release-country", tokenized_match)
    strategy.add_evaluation_function("label", tokenized_match)
    strategy.add_evaluation_function("tracks_track_name", set_equality_match)

    fusion_test_set = get_fusion_test_set("music")
    evaluator = DataFusionEvaluator(strategy)

    evaluation_results = evaluator.evaluate(
        fused_df=fused,
        fused_id_column='mbrainz_id',
        gold_df=fusion_test_set,
        gold_id_column='id',
    )
    assert evaluation_results['overall_accuracy'] == pytest.approx(0.66, abs=0.01), f"Expected overall accuracy 0.67 +/- 0.01, got {evaluation_results['overall_accuracy']}"

