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

def test_run_companies(get_input_data, get_correspondences, get_fusion_test_set):
    dbpedia = get_input_data("companies", "dbpedia")
    forbes = get_input_data("companies", "forbes")
    fullcontact = get_input_data("companies", "fullcontact")

    def generate_blocking_keys_tokens(company_name: str):
        tokens = re.split(r'[^a-z]', company_name.lower())
        first_token = [token for token in tokens if len(token) > 1]
        if first_token:
            return first_token[0]
        else:
            return company_name

    dbpedia['name_first_token'] = dbpedia['name'].apply(generate_blocking_keys_tokens)
    forbes['name_first_token'] = forbes['name'].apply(generate_blocking_keys_tokens)
    fullcontact['name_first_token'] = fullcontact['name'].apply(generate_blocking_keys_tokens)
    
    #### Blocking ####

    standard_blocker_f2d = StandardBlocker(
        forbes, dbpedia,
        on=['name_first_token'],
        batch_size=1000,
        id_column='id'
    )
    standard_candidates_f2d = standard_blocker_f2d.materialize()

    standard_blocker_f2fc = StandardBlocker(
        forbes, fullcontact,
        on=['name_first_token'],
        batch_size=1000,
        id_column='id'
    )
    standard_candidates_f2fc = standard_blocker_f2fc.materialize()
    assert len(standard_candidates_f2fc) == 1563, f"Expected 1563 candidates, got {len(standard_candidates_f2fc)}"

    #### Matching ####

    def normalize_text(s: str) -> str: 
        if s is None:
            return ""
        return re.sub(r"[^\w\s]|_", "", s).lower()

    comparators = [
        StringComparator(column='name', similarity_function='jaccard', preprocess=normalize_text),
        StringComparator(column='country', similarity_function='jaccard', preprocess=normalize_text),
        StringComparator(column='industry', similarity_function='jaccard', preprocess=normalize_text),
    ]

    matcher = RuleBasedMatcher()

    correspondences_f2d = matcher.match(
        df_left=forbes,
        df_right=dbpedia, 
        candidates=standard_candidates_f2d,
        comparators=comparators,
        weights=[1.0, 0.5, 1.0],
        threshold=0.5,
        id_column='id'
    )
    assert len(correspondences_f2d) == 294, f"Expected 294 correspondences, got {len(correspondences_f2d)}"

    correspondences_f2fc = matcher.match(
        df_left=forbes,
        df_right=fullcontact, 
        candidates=standard_candidates_f2fc,
        comparators=comparators,
        weights=[1.0, 0.5, 1.0],
        threshold=0.5,
        id_column='id',
    )
    assert len(correspondences_f2fc) == 187, f"Expected 187 correspondences, got {len(correspondences_f2fc)}"

    test_gt_f2d = get_correspondences("companies", "forbes", "dbpedia")
    eval_results = EntityMatchingEvaluator.evaluate_matching(
        correspondences_f2d,
        test_gt_f2d,
        out_dir=None
    )
    assert eval_results['accuracy'] == pytest.approx(0.786, abs=0.01), f"Expected accuracy == 0.786 +/- 0.01, got {eval_results['accuracy']}"

    clusterer = MaximumBipartiteMatching()
    correspondences_f2d = clusterer.cluster(correspondences_f2d)
    assert len(correspondences_f2d) == 265, f"Expected 265 correspondences after MBM, got {len(correspondences_f2d)}"

    #### Fusion ####

    all_correspondences = pd.concat([correspondences_f2d, correspondences_f2fc], ignore_index=True)
    assert len(all_correspondences) == 452, f"Expected 452 total correspondences, got {len(all_correspondences)}"

    forbes["forbes_id"] = forbes["id"]

    forbes.attrs["trust_score"] = 3
    dbpedia.attrs["trust_score"] = 1
    fullcontact.attrs["trust_score"] = 2

    strategy = DataFusionStrategy('company_fusion_strategy')
    strategy.add_attribute_fuser('name', voting)
    strategy.add_attribute_fuser('assets', prefer_higher_trust)
    strategy.add_attribute_fuser('revenue', prefer_higher_trust)
    strategy.add_attribute_fuser('keypeople_name', union)
    strategy.add_attribute_fuser('founded', voting)
    strategy.add_attribute_fuser('country', voting)
    strategy.add_attribute_fuser('city', shortest_string)

    engine = DataFusionEngine(strategy)

    fused = engine.run(
        datasets=[forbes, dbpedia, fullcontact],
        correspondences=all_correspondences,
        id_column="id",
        include_singletons=False,
    )
    assert len(fused) == 417, f"Expected 417 fused records, got {len(fused)}"

    strategy.add_evaluation_function("name", tokenized_match)
    strategy.add_evaluation_function("assets", tokenized_match)
    strategy.add_evaluation_function("revenue", numeric_tolerance_match, tolerance=0.1)
    strategy.add_evaluation_function("assets", numeric_tolerance_match, tolerance=0.1)
    strategy.add_evaluation_function("keypeople_name", set_equality_match)
    strategy.add_evaluation_function("founded", year_only_match)
    strategy.add_evaluation_function("country", tokenized_match)
    strategy.add_evaluation_function("city", tokenized_match)

    fusion_test_set = get_fusion_test_set("companies")
    evaluator = DataFusionEvaluator(strategy)

    evaluation_results = evaluator.evaluate(
        fused_df=fused,
        fused_id_column='forbes_id',
        gold_df=fusion_test_set,
        gold_id_column='id',
    )
    assert evaluation_results['overall_accuracy'] == pytest.approx(0.658, abs=0.01), f"Expected overall accuracy 0.684 +/- 0.01, got {evaluation_results['overall_accuracy']}"

