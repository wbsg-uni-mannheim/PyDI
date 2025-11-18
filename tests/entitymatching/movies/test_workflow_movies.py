import pytest
import pandas as pd
from PyDI.entitymatching import EntityMatchingEvaluator
from PyDI.entitymatching import (
    EmbeddingBlocker,
    StandardBlocker,
    StringComparator,
    DateComparator,
    RuleBasedMatcher,
    MaximumBipartiteMatching,
)
from PyDI.fusion import (
    DataFusionStrategy, 
    DataFusionEngine,
    DataFusionEvaluator, 
    longest_string, 
    union, 
    prefer_higher_trust, 
    tokenized_match, 
    year_only_match, 
    boolean_match
)

def test_run_matching(get_input_data, get_correspondences, get_fusion_test_set):
    actors = get_input_data("movies", "actors")
    golden_globes = get_input_data("movies", "golden_globes")
    academy_awards = get_input_data("movies", "academy_awards")

    actors['title_prefix'] = actors['title'].astype(str).apply(lambda x: ''.join([word[:2].upper() for word in x.split()[:3]]))
    academy_awards['title_prefix'] = academy_awards['title'].astype(str).apply(lambda x: ''.join([word[:2].upper() for word in x.split()[:3]]))

    #### Blocking ####

    embedding_blocker_a2g = EmbeddingBlocker(
        actors, golden_globes,
        text_cols=['title'],
        model="sentence-transformers/all-MiniLM-L6-v2",
        index_backend="sklearn",
        top_k=20,
        id_column='id'
    )

    standard_blocker_aa2a = StandardBlocker(
        academy_awards, actors,
        on=['title_prefix'],
        batch_size=1000,
        id_column='id'
    )
    embedding_candidates_a2g = embedding_blocker_a2g.materialize()
    assert len(embedding_candidates_a2g) == 2945, f"Expected 2945 candidates, got {len(embedding_candidates_a2g)}"

    #### Matching ####

    comparators = [
        StringComparator('title', 'jaccard', preprocess=str.lower),
        DateComparator('date', max_days_difference=365),
        StringComparator('actors_actor_name', 'jaccard', preprocess=str.lower, list_strategy='concatenate')
    ]

    matcher = RuleBasedMatcher()

    correspondences_a2g = matcher.match(
        df_left=actors,
        df_right=golden_globes, 
        candidates=embedding_blocker_a2g,
        comparators=comparators,
        weights=[0.7, 0.2, 0.1],
        threshold=0.7,
        id_column='id'
    )
    assert len(correspondences_a2g) == 86, f"Expected 86 correspondences, got {len(correspondences_a2g)}"

    correspondences_aa2a = matcher.match(
        df_left=academy_awards,
        df_right=actors, 
        candidates=standard_blocker_aa2a,
        comparators=comparators,
        weights=[0.7, 0.2, 0.1],
        threshold=0.7,
        id_column='id',
    )
    assert len(correspondences_aa2a) == 144, f"Expected 144 correspondences, got {len(correspondences_aa2a)}"

    test_gt_a2g = get_correspondences("movies", "actors", "golden_globes")
    eval_results = EntityMatchingEvaluator.evaluate_matching(
        correspondences_a2g,
        test_gt_a2g,
        out_dir=None
    )
    assert eval_results['accuracy'] == pytest.approx(0.768, rel=1e-2), f"Expected accuracy == 0.768, got {eval_results['accuracy']}"

    clusterer = MaximumBipartiteMatching()
    mbm_correspondences_a2g = clusterer.cluster(correspondences_a2g)
    assert len(mbm_correspondences_a2g) == 80, f"Expected 80 correspondences after MBM, got {len(mbm_correspondences_a2g)}"

    #### Fusion ####

    all_correspondences = pd.concat([correspondences_a2g, correspondences_aa2a], ignore_index=True)
    assert len(all_correspondences) == 230, f"Expected 230 total correspondences, got {len(all_correspondences)}"

    academy_awards["academy_awards_id"] = academy_awards["id"]

    academy_awards.attrs["trust_score"] = 3
    actors.attrs["trust_score"] = 2
    golden_globes.attrs["trust_score"] = 1


    strategy = DataFusionStrategy('movie_fusion_strategy')
    strategy.add_attribute_fuser('title', longest_string)
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
    assert len(fused) == 135, f"Expected 135 fused records, got {len(fused)}"

    strategy.add_evaluation_function("title", tokenized_match)
    strategy.add_evaluation_function("director_name", tokenized_match)
    strategy.add_evaluation_function("actors_actor_name", tokenized_match)
    strategy.add_evaluation_function("date", year_only_match)
    strategy.add_evaluation_function("oscar", boolean_match)

    fusion_test_set = get_fusion_test_set("movies")
    evaluator = DataFusionEvaluator(strategy)

    evaluation_results = evaluator.evaluate(
        fused_df=fused,
        fused_id_column='academy_awards_id',
        gold_df=fusion_test_set,
        gold_id_column='id',
    )

    assert evaluation_results['overall_accuracy'] == pytest.approx(0.758, rel=1e-2), f"Expected overall accuracy 0.758, got {evaluation_results['overall_accuracy']}"
