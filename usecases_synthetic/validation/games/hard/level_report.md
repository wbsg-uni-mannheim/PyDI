# Validation report - games / hard

_Generated at 2026-05-16T17:28:44.033701+00:00_

- domain: `games`
- level: `hard`
- with_llm: `True`
- committee_versions: sm=`sm_committee.yaml@cb4a6847ac9e`, norm=`normalization_committee_games.yaml@ec8e48a06ee7`, em_blocking=`em_blocking_committee_games.yaml@e4adebfad54e`, em_matching=`em_matching_committee_games.yaml@d2adcee14e35`, fusion=`fusion_committee_games.yaml@aef79ffb0514`

## Stage summary

| stage | metric | measured | baseline | delta |
|---|---|---|---|---|
| sm | macro_f1 | 0.6779 | 0.7267 | -0.0488 |
| norm | macro_f1 | 0.6595 | 0.6821 | -0.0226 |
| em_blocking | macro_pair_recall | 0.5489 | 0.9986 | -0.4497 |
| em_matching | macro_f1_vs_test | 0.5773 | 0.6799 | -0.1026 |
| fusion | overall_accuracy | 0.6606 | 0.7469 | -0.0863 |

## Stage: sm - per member

| member | f1 | f1_baseline | f1_delta |
|---|---|---|---|
| coma_hybrid | 0.5641 | 0.7556 | -0.1915 |
| duplicate_majority | 0.6829 | 0.6829 | 0.0000 |
| embedding_sbert | 0.6667 | 0.6275 | 0.0392 |
| instance_tf_cosine | 0.6667 | 0.7059 | -0.0392 |
| label_jw | 0.3030 | 0.3333 | -0.0303 |
| llm_openai | 1.0000 | 1.0000 | 0.0000 |
| magneto_slm_llm | 0.8621 | 0.9818 | -0.1197 |

## Stage: norm - per member

| member | macro_f1 | macro_f1_baseline | macro_f1_delta |
|---|---|---|---|
| date_iso | 0.9433 | 0.9750 | -0.0317 |
| llm_canonicalize | 0.5557 | 0.5853 | -0.0296 |
| number_locale | 0.4301 | 0.4385 | -0.0084 |
| taxonomy_lookup | 0.6208 | 0.6276 | -0.0068 |
| text_clean | 0.7476 | 0.7840 | -0.0364 |

## Stage: em_blocking - per member

| member | pair_recall | pair_recall_baseline | pair_recall_delta | reduction_ratio | reduction_ratio_baseline |
|---|---|---|---|---|---|
| bm25_blocker | 0.5603 | 1.0000 | -0.4397 | 0.9937 | 0.9937 |
| embedding_blocker | 0.5603 | 1.0000 | -0.4397 | 0.9937 | 0.9937 |
| sc_block | 0.5431 | 1.0000 | -0.4569 | 0.9937 | 0.9937 |
| sorted_neighbourhood_blocker | 0.5345 | 0.9914 | -0.4569 | 0.9986 | 0.9986 |
| standard_blocker | 0.5345 | 1.0000 | -0.4655 | 0.9979 | 0.9987 |
| token_blocker | 0.5603 | 1.0000 | -0.4397 | 0.9553 | 0.9533 |

## Stage: em_matching - per member

| member | f1 | f1_baseline | f1_delta | f1_vs_val | f1_vs_val_baseline | f1_vs_test | f1_vs_test_baseline |
|---|---|---|---|---|---|---|---|
| comem | 0.5402 | 0.7708 | -0.2306 | nan | nan | 0.5402 | 0.7708 |
| ditto_plm | 0.5541 | 0.5860 | -0.0319 | nan | nan | 0.5541 | 0.5860 |
| llm_matcher | 0.5682 | 0.7604 | -0.1922 | nan | nan | 0.5682 | 0.7604 |
| magellan | 0.6468 | 0.6024 | 0.0444 | nan | nan | 0.6468 | 0.6024 |

## Stage: em - per pair

| pair | member | f1 | f1_baseline | f1_delta |
|---|---|---|---|---|
| dbpedia_sales | comem | 0.5402 | 0.7708 | -0.2306 |
| dbpedia_sales | ditto_plm | 0.5541 | 0.5860 | -0.0319 |
| dbpedia_sales | llm_matcher | 0.5682 | 0.7604 | -0.1922 |
| dbpedia_sales | magellan | 0.6468 | 0.6024 | 0.0444 |

## Stage: fusion - per member

| member | overall_accuracy | overall_accuracy_baseline | overall_accuracy_delta |
|---|---|---|---|
| ESRB_llm_judge | 0.6119 | 0.6541 | -0.0422 |
| ESRB_most_complete | 0.6119 | 0.6541 | -0.0422 |
| ESRB_prefer_higher_trust | 0.6119 | 0.6541 | -0.0422 |
| ESRB_voting | 0.6106 | 0.6541 | -0.0434 |
| criticScore_fusionquery | 0.6081 | 0.6541 | -0.0460 |
| criticScore_huber_m_estimator | 0.6106 | 0.6541 | -0.0434 |
| criticScore_maximum | 0.6106 | 0.6541 | -0.0434 |
| criticScore_median | 0.6106 | 0.6541 | -0.0434 |
| criticScore_median_of_means | 0.6106 | 0.6541 | -0.0434 |
| criticScore_prefer_higher_trust | 0.6081 | 0.6552 | -0.0471 |
| criticScore_trimmed_mean | 0.6106 | 0.6541 | -0.0434 |
| developer_longest_string | 0.6068 | 0.6396 | -0.0328 |
| developer_most_complete | 0.6068 | 0.6396 | -0.0328 |
| developer_prefer_higher_trust | 0.6157 | 0.6541 | -0.0384 |
| developer_voting | 0.6106 | 0.6541 | -0.0434 |
| genres_intersection | 0.5879 | 0.6463 | -0.0584 |
| genres_intersection_k_sources | 0.5929 | 0.6885 | -0.0956 |
| genres_ltm | 0.6233 | 0.7052 | -0.0820 |
| genres_prefer_higher_trust | 0.6485 | 0.7164 | -0.0678 |
| genres_union | 0.6448 | 0.7286 | -0.0838 |
| genres_voting | 0.6106 | 0.6541 | -0.0434 |
| name_accusim | 0.6094 | 0.6563 | -0.0469 |
| name_casefusion | 0.6081 | 0.6518 | -0.0437 |
| name_fusionquery | 0.6106 | 0.6618 | -0.0512 |
| name_llm_judge | 0.6094 | 0.6563 | -0.0469 |
| name_longest_string | 0.6068 | 0.6552 | -0.0483 |
| name_most_complete | 0.6068 | 0.6552 | -0.0483 |
| name_prefer_higher_trust | 0.6081 | 0.6630 | -0.0549 |
| name_voting | 0.6106 | 0.6541 | -0.0434 |
| platform_most_complete | 0.5980 | 0.6429 | -0.0450 |
| platform_prefer_higher_trust | 0.6081 | 0.6541 | -0.0460 |
| platform_truthfinder | 0.6094 | 0.6541 | -0.0447 |
| platform_voting | 0.6106 | 0.6541 | -0.0434 |
| publisher_longest_string | 0.6106 | 0.6541 | -0.0434 |
| publisher_prefer_higher_trust | 0.6106 | 0.6541 | -0.0434 |
| publisher_voting | 0.6106 | 0.6541 | -0.0434 |
| releaseYear_earliest | 0.6169 | 0.6529 | -0.0360 |
| releaseYear_prefer_higher_trust | 0.6157 | 0.6607 | -0.0451 |
| releaseYear_voting | 0.6106 | 0.6541 | -0.0434 |
| userScore_huber_m_estimator | 0.6094 | 0.6529 | -0.0436 |
| userScore_maximum | 0.6106 | 0.6541 | -0.0434 |
| userScore_median | 0.6094 | 0.6529 | -0.0436 |
| userScore_median_of_means | 0.6094 | 0.6518 | -0.0425 |
| userScore_prefer_higher_trust | 0.6106 | 0.6563 | -0.0457 |
| userScore_trimmed_mean | 0.6094 | 0.6518 | -0.0425 |

## Stage: fusion - per attribute

| attribute | best_accuracy | baseline | delta | spread | spread_baseline | spread_delta |
|---|---|---|---|---|---|---|
| ESRB | 0.9545 | 0.9800 | -0.0255 | 0.0114 | 0.0000 | 0.0114 |
| criticScore | 0.2386 | 0.2700 | -0.0314 | 0.0227 | 0.0100 | 0.0127 |
| developer | 0.7841 | 0.8700 | -0.0859 | 0.0795 | 0.1300 | -0.0505 |
| genres | 0.5568 | 0.8400 | -0.2832 | 0.5455 | 0.7400 | -0.1945 |
| name | 0.9545 | 0.9700 | -0.0155 | 0.0341 | 0.1000 | -0.0659 |
| platform | 0.9205 | 0.9700 | -0.0495 | 0.1136 | 0.1000 | 0.0136 |
| publisher | 0.4318 | 0.6600 | -0.2282 | 0.0000 | 0.0000 | 0.0000 |
| releaseYear | 0.8864 | 0.9400 | -0.0536 | 0.0568 | 0.0700 | -0.0132 |
| userScore | 0.2184 | 0.2222 | -0.0038 | 0.0115 | 0.0404 | -0.0289 |
