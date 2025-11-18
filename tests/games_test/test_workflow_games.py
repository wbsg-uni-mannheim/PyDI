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
    StableMatching
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

def test_run_games(get_input_data, get_correspondences, get_fusion_test_set):
    dbpedia = get_input_data("games", "dbpedia")
    metacritic = get_input_data("games", "forbes")
    sales = get_input_data("games", "fullcontact")

    dbpedia['name_prefix'] = dbpedia['name'].astype(str).apply(lambda x: ''.join([word[:2].upper() for word in x.split()[:3]]))
    metacritic['name_prefix'] = metacritic['name'].astype(str).apply(lambda x: ''.join([word[:2].upper() for word in x.split()[:3]]))
    sales['name_prefix'] = sales['name'].astype(str).apply(lambda x: ''.join([word[:2].upper() for word in x.split()[:3]]))
    
    #### Blocking ####

    standard_blocker_m2d = StandardBlocker(
        metacritic, dbpedia,
        on=['name_prefix'],
        batch_size=1000,
        id_column='id'
    )
    standard_candidates_m2d = standard_blocker_m2d.materialize()
    assert len(standard_candidates_m2d) == 154209, f"Expected 154209 candidates, got {len(standard_candidates_m2d)}"

    standard_blocker_m2s = StandardBlocker(
        metacritic, sales,
        on=['name_prefix'],
        batch_size=1000,
        id_column='id'
    )
    standard_candidates_m2s = standard_blocker_m2s.materialize()
    assert len(standard_candidates_m2s) == 154209, f"Expected 154209 candidates, got {len(standard_candidates_m2s)}"

    #### Matching ####

    comparators = [
        StringComparator(column='name', similarity_function='jaccard', preprocess=str.lower),
        StringComparator(column='platform', similarity_function='jaccard', preprocess=str.lower),
        DateComparator(column='releaseYear'),
    ]

    matcher = RuleBasedMatcher()

    correspondences_m2d = matcher.match(
        df_left=metacritic,
        df_right=dbpedia, 
        candidates=standard_blocker_m2d,
        comparators=comparators,
        weights=[0.6, 0.3, 0.1],
        threshold=0.8,
        id_column='id'
    )
    assert len(correspondences_m2d) == 3873, f"Expected 3873 correspondences, got {len(correspondences_m2d)}"

    correspondences_m2s = matcher.match(
        df_left=metacritic,
        df_right=sales, 
        candidates=standard_blocker_m2s,
        comparators=comparators,
        weights=[0.6, 0.3, 0.1],
        threshold=0.8,
        id_column='id'
    )
    assert len(correspondences_m2s) == 144, f"Expected 144 correspondences, got {len(correspondences_m2s)}"

    test_gt_m2d = get_correspondences("games", "metacritic", "dbpedia")
    eval_results = EntityMatchingEvaluator.evaluate_matching(
        correspondences_m2d,
        test_gt_m2d,
        out_dir=None
    )
    assert eval_results['accuracy'] == pytest.approx(0.768, rel=1e-2), f"Expected accuracy == 0.768, got {eval_results['accuracy']}"

    clusterer = StableMatching()
    correspondences_m2s = clusterer.cluster(correspondences_m2s)
    assert len(correspondences_m2s) == 80, f"Expected 80 correspondences after MBM, got {len(correspondences_m2s)}"

    #### Fusion ####

    all_correspondences = pd.concat([correspondences_m2d, correspondences_m2s], ignore_index=True)    
    assert len(all_correspondences) == 230, f"Expected 230 total correspondences, got {len(all_correspondences)}"

    metacritic["metacritic_id"] = metacritic["id"]

    metacritic.attrs["trust_score"] = 3
    sales.attrs["trust_score"] = 2
    dbpedia.attrs["trust_score"] = 1

    strategy = DataFusionStrategy('game_fusion_strategy')
    strategy.add_attribute_fuser('name', voting)
    strategy.add_attribute_fuser('platform', voting)
    strategy.add_attribute_fuser('developer', longest_string)
    strategy.add_attribute_fuser('releaseYear', voting, trust_key="trust_score")
    strategy.add_attribute_fuser('ESRB', prefer_higher_trust, trust_key="trust_score")
    strategy.add_attribute_fuser('criticScore', voting)
    strategy.add_attribute_fuser('userScore', voting)

    engine = DataFusionEngine(strategy)

    fused = engine.run(
        datasets=[metacritic, dbpedia, sales],
        correspondences=all_correspondences,
        id_column="id",
        include_singletons=False,
    )
    assert len(fused) == 135, f"Expected 135 fused records, got {len(fused)}"

    strategy.add_evaluation_function("title", tokenized_match)
    strategy.add_evaluation_function("director_name", tokenized_match)
    strategy.add_evaluation_function("actors_actor_name", tokenized_match)
    strategy.add_evaluation_function("date", year_only_match)
    strategy.add_evaluation_function("oscar", boolean_match)

    fusion_test_set = get_fusion_test_set("games")
    evaluator = DataFusionEvaluator(strategy)

    evaluation_results = evaluator.evaluate(
        fused_df=fused,
        fused_id_column='metacritic_id',
        gold_df=fusion_test_set,
        gold_id_column='id',
    )
    assert evaluation_results['overall_accuracy'] == pytest.approx(0.768, abs=0.01), f"Expected overall accuracy 0.768 +/- 0.01, got {evaluation_results['overall_accuracy']}"

