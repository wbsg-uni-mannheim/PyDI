# Validation report - games / medium

_Generated at 2026-06-05T04:30:40.234313+00:00_

- domain: `games`
- level: `medium`
- with_llm: `True`
- committee_versions: sm=`sm_committee.yaml@cb4a6847ac9e`, norm=`normalization_committee_games.yaml@25731d7e9982`, em_blocking=`em_blocking_committee_games.yaml@c07cf762a7d5`, em_matching=`em_matching_committee_games.yaml@ead92fa3a2a8`, fusion=`fusion_committee_games.yaml@fd979439ba3d`

## Stage summary

| stage | metric | measured | baseline | delta |
|---|---|---|---|---|
| sm | macro_f1 | 0.7677 | 0.7478 | 0.0198 |
| norm | macro_f1 | 0.8713 | 0.8971 | -0.0258 |
| em_blocking | macro_pair_recall_variant_model_on_regen_test | 0.0000 | 0.0000 | 0.0000 |
| em_matching | macro_f1_variant_model_on_regen_test | 0.6573 | 0.6093 | 0.0480 |
| fusion | overall_accuracy | 0.6902 | 0.7202 | -0.0300 |

## Stage: sm - per member

| member | f1 | f1_baseline | f1_delta |
|---|---|---|---|
| coma_hybrid | 0.7727 | 0.7442 | 0.0285 |
| duplicate_majority | 0.7442 | 0.8444 | -0.1003 |
| embedding_sbert | 0.7059 | 0.6400 | 0.0659 |
| instance_tf_cosine | 0.6383 | 0.7200 | -0.0817 |
| label_jw | 0.6190 | 0.3429 | 0.2762 |
| llm_openai | 0.9811 | 1.0000 | -0.0189 |
| magneto_slm_llm | 0.9123 | 0.9434 | -0.0311 |

## Stage: norm - per member

| member | macro_f1 | macro_f1_baseline | macro_f1_delta |
|---|---|---|---|
| llm_only | 0.8362 | 0.8547 | -0.0184 |
| passthrough | 0.9428 | 0.9809 | -0.0380 |
| rule_per_attribute_optimal | 0.8349 | 0.8559 | -0.0209 |

## Stage: em_blocking - per member

| member | pair_recall | pair_recall_baseline | pair_recall_delta | reduction_ratio | reduction_ratio_baseline |
|---|---|---|---|---|---|
| bm25_blocker | 0.9387 | 0.9528 | -0.0142 | 0.9964 | 0.9963 |
| embedding_blocker | 0.9292 | 0.9387 | -0.0094 | 0.9964 | 0.9963 |
| sc_block | 0.9434 | 0.9434 | 0.0000 | 0.9964 | 0.9963 |
| sorted_neighbourhood_blocker | 0.8875 | 0.9155 | -0.0280 | 0.9988 | 0.9988 |
| standard_blocker | 0.9579 | 0.9858 | -0.0279 | 0.9989 | 0.9988 |
| token_blocker | 0.9858 | 1.0000 | -0.0142 | 0.9546 | 0.9530 |

## Stage: em_matching - per member

| member | f1 | f1_baseline | f1_delta | f1_baseline_test | f1_baseline_test_baseline | f1_regen_test | f1_regen_test_baseline |
|---|---|---|---|---|---|---|---|
| comem | 0.5727 | 0.6071 | -0.0344 | 0.5727 | 0.6071 | 0.5727 | 0.6071 |
| ditto_plm | 0.7300 | 0.7155 | 0.0145 | 0.6708 | 0.7155 | 0.6708 | 0.7155 |
| llm_matcher | 0.5710 | 0.6054 | -0.0344 | 0.5710 | 0.6054 | 0.5710 | 0.6054 |
| magellan | 0.7555 | 0.5092 | 0.2463 | 0.7497 | 0.5092 | 0.7497 | 0.5092 |

## Stage: em_matching - per pair

| pair | member | f1 | f1_baseline | f1_delta |
|---|---|---|---|---|
| dbpedia_sales | comem | 0.6947 | 0.7143 | -0.0195 |
| dbpedia_sales | ditto_plm | 0.7681 | 0.8122 | -0.0441 |
| dbpedia_sales | llm_matcher | 0.6837 | 0.7143 | -0.0306 |
| dbpedia_sales | magellan | 0.8069 | 0.5934 | 0.2135 |
| metacritic_dbpedia | comem | 0.4507 | 0.5000 | -0.0493 |
| metacritic_dbpedia | ditto_plm | 0.6919 | 0.6188 | 0.0732 |
| metacritic_dbpedia | llm_matcher | 0.4583 | 0.4965 | -0.0381 |
| metacritic_dbpedia | magellan | 0.7041 | 0.4250 | 0.2791 |

## Stage: fusion - per member

| member | overall_accuracy | overall_accuracy_baseline | overall_accuracy_delta |
|---|---|---|---|
| accusim_only | 0.6396 | 0.6485 | -0.0089 |
| casefusion_only | 0.6607 | 0.6986 | -0.0378 |
| fusionquery_only | 0.6763 | 0.7208 | -0.0445 |
| llm_only | 0.5996 | 0.6140 | -0.0145 |
| ltm_only | 0.6496 | 0.6796 | -0.0300 |
| prefer_higher_trust_only | 0.6808 | 0.7186 | -0.0378 |
| pydi_per_attribute_optimal | 0.6552 | 0.6986 | -0.0434 |
| truthfinder_only | 0.6908 | 0.7208 | -0.0300 |
| voting_only | 0.6285 | 0.6440 | -0.0156 |

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
