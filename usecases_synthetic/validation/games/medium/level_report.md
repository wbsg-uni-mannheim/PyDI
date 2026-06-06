# Validation report - games / medium

_Generated at 2026-06-06T03:11:37.613976+00:00_

- domain: `games`
- level: `medium`
- with_llm: `True`
- committee_versions: sm=`sm_committee.yaml@cb4a6847ac9e`, norm=`normalization_committee_games.yaml@25731d7e9982`, em_blocking=`em_blocking_committee_games.yaml@c07cf762a7d5`, em_matching=`em_matching_committee_games.yaml@ead92fa3a2a8`, fusion=`fusion_committee_games.yaml@fd979439ba3d`

## Stage summary

| stage | metric | measured | baseline | delta |
|---|---|---|---|---|
| sm | macro_f1 | 0.7700 | 0.7478 | 0.0221 |
| norm | macro_f1 | 0.8718 | 0.8971 | -0.0254 |
| em_blocking | macro_pair_recall_variant_model_on_regen_test | 0.9540 | 0.9560 | -0.0020 |
| em_matching | macro_f1_variant_model_on_regen_test | 0.6716 | 0.6093 | 0.0623 |
| fusion | overall_accuracy | 0.6869 | 0.7202 | -0.0333 |

## Stage: sm - per member

| member | f1 | f1_baseline | f1_delta |
|---|---|---|---|
| coma_hybrid | 0.7727 | 0.7442 | 0.0285 |
| duplicate_majority | 0.7442 | 0.8444 | -0.1003 |
| embedding_sbert | 0.7059 | 0.6400 | 0.0659 |
| instance_tf_cosine | 0.6383 | 0.7200 | -0.0817 |
| label_jw | 0.6190 | 0.3429 | 0.2762 |
| llm_openai | 0.9811 | 1.0000 | -0.0189 |
| magneto_slm_llm | 0.9286 | 0.9434 | -0.0148 |

## Stage: norm - per member

| member | macro_f1 | macro_f1_baseline | macro_f1_delta |
|---|---|---|---|
| llm_only | 0.8362 | 0.8547 | -0.0184 |
| passthrough | 0.9442 | 0.9809 | -0.0367 |
| rule_per_attribute_optimal | 0.8349 | 0.8559 | -0.0209 |

## Stage: em_blocking - per member

| member | pair_recall | pair_recall_baseline | pair_recall_delta | reduction_ratio | reduction_ratio_baseline |
|---|---|---|---|---|---|
| bm25_blocker | 0.9476 | 0.9528 | -0.0052 | 0.9964 | 0.9963 |
| embedding_blocker | 0.9429 | 0.9387 | 0.0042 | 0.9964 | 0.9963 |
| sc_block | 0.9667 | 0.9434 | 0.0233 | 0.9964 | 0.9963 |
| sorted_neighbourhood_blocker | 0.9004 | 0.9155 | -0.0151 | 0.9988 | 0.9988 |
| standard_blocker | 0.9667 | 0.9858 | -0.0192 | 0.9989 | 0.9988 |
| token_blocker | 1.0000 | 1.0000 | 0.0000 | 0.9545 | 0.9530 |

## Stage: em_matching - per member

| member | f1 | f1_baseline | f1_delta | f1_baseline_test | f1_baseline_test_baseline | f1_regen_test | f1_regen_test_baseline |
|---|---|---|---|---|---|---|---|
| comem | 0.5818 | 0.6071 | -0.0253 | 0.5835 | 0.6071 | 0.5818 | 0.6071 |
| ditto_plm | 0.7683 | 0.7155 | 0.0528 | 0.6788 | 0.7155 | 0.6770 | 0.7155 |
| llm_matcher | 0.5728 | 0.6054 | -0.0326 | 0.5744 | 0.6054 | 0.5728 | 0.6054 |
| magellan | 0.7634 | 0.5092 | 0.2542 | 0.7554 | 0.5092 | 0.7554 | 0.5092 |

## Stage: em_matching - per pair

| pair | member | f1 | f1_baseline | f1_delta |
|---|---|---|---|---|
| dbpedia_sales | comem | 0.7021 | 0.7143 | -0.0122 |
| dbpedia_sales | ditto_plm | 0.8231 | 0.8122 | 0.0108 |
| dbpedia_sales | llm_matcher | 0.6872 | 0.7143 | -0.0271 |
| dbpedia_sales | magellan | 0.8103 | 0.5934 | 0.2169 |
| metacritic_dbpedia | comem | 0.4615 | 0.5000 | -0.0385 |
| metacritic_dbpedia | ditto_plm | 0.7136 | 0.6188 | 0.0948 |
| metacritic_dbpedia | llm_matcher | 0.4583 | 0.4965 | -0.0381 |
| metacritic_dbpedia | magellan | 0.7164 | 0.4250 | 0.2914 |

## Stage: fusion - per member

| member | overall_accuracy | overall_accuracy_baseline | overall_accuracy_delta |
|---|---|---|---|
| accusim_only | 0.6363 | 0.6485 | -0.0122 |
| casefusion_only | 0.6541 | 0.6986 | -0.0445 |
| fusionquery_only | 0.6763 | 0.7208 | -0.0445 |
| llm_only | 0.5951 | 0.6140 | -0.0189 |
| ltm_only | 0.6585 | 0.6796 | -0.0211 |
| prefer_higher_trust_only | 0.6730 | 0.7186 | -0.0456 |
| pydi_per_attribute_optimal | 0.6518 | 0.6986 | -0.0467 |
| truthfinder_only | 0.6874 | 0.7208 | -0.0334 |
| voting_only | 0.6240 | 0.6440 | -0.0200 |

## Stage: fusion - per attribute

| attribute | best_accuracy | baseline | delta | spread | spread_baseline | spread_delta |
|---|---|---|---|---|---|---|
| ESRB | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| criticScore | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| developer | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| genres | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| name | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| platform | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| publisher | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| releaseYear | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| userScore | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
