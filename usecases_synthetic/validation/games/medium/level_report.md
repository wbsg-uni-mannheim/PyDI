# Validation report - games / medium

_Generated at 2026-05-16T17:14:46.682030+00:00_

- domain: `games`
- level: `medium`
- with_llm: `True`
- committee_versions: sm=`sm_committee.yaml@cb4a6847ac9e`, norm=`normalization_committee_games.yaml@ec8e48a06ee7`, em_blocking=`em_blocking_committee_games.yaml@e4adebfad54e`, em_matching=`em_matching_committee_games.yaml@d2adcee14e35`, fusion=`fusion_committee_games.yaml@aef79ffb0514`

## Stage summary

| stage | metric | measured | baseline | delta |
|---|---|---|---|---|
| sm | macro_f1 | 0.7475 | 0.7267 | 0.0208 |
| norm | macro_f1 | 0.6573 | 0.6821 | -0.0248 |
| em_blocking | macro_pair_recall | 0.9971 | 0.9986 | -0.0014 |
| em_matching | macro_f1_vs_test | 0.7176 | 0.6799 | 0.0376 |
| fusion | overall_accuracy | 0.7147 | 0.7469 | -0.0322 |

## Stage: sm - per member

| member | f1 | f1_baseline | f1_delta |
|---|---|---|---|
| coma_hybrid | 0.7727 | 0.7556 | 0.0172 |
| duplicate_majority | 0.5789 | 0.6829 | -0.1040 |
| embedding_sbert | 0.7059 | 0.6275 | 0.0784 |
| instance_tf_cosine | 0.6250 | 0.7059 | -0.0809 |
| label_jw | 0.6190 | 0.3333 | 0.2857 |
| llm_openai | 1.0000 | 1.0000 | 0.0000 |
| magneto_slm_llm | 0.9310 | 0.9818 | -0.0508 |

## Stage: norm - per member

| member | macro_f1 | macro_f1_baseline | macro_f1_delta |
|---|---|---|---|
| date_iso | 0.9378 | 0.9750 | -0.0372 |
| llm_canonicalize | 0.5613 | 0.5853 | -0.0240 |
| number_locale | 0.4257 | 0.4385 | -0.0127 |
| taxonomy_lookup | 0.6085 | 0.6276 | -0.0192 |
| text_clean | 0.7531 | 0.7840 | -0.0309 |

## Stage: em_blocking - per member

| member | pair_recall | pair_recall_baseline | pair_recall_delta | reduction_ratio | reduction_ratio_baseline |
|---|---|---|---|---|---|
| bm25_blocker | 1.0000 | 1.0000 | 0.0000 | 0.9937 | 0.9937 |
| embedding_blocker | 1.0000 | 1.0000 | 0.0000 | 0.9937 | 0.9937 |
| sc_block | 1.0000 | 1.0000 | 0.0000 | 0.9937 | 0.9937 |
| sorted_neighbourhood_blocker | 0.9914 | 0.9914 | 0.0000 | 0.9986 | 0.9986 |
| standard_blocker | 0.9914 | 1.0000 | -0.0086 | 0.9987 | 0.9987 |
| token_blocker | 1.0000 | 1.0000 | 0.0000 | 0.9533 | 0.9533 |

## Stage: em_matching - per member

| member | f1 | f1_baseline | f1_delta | f1_vs_val | f1_vs_val_baseline | f1_vs_test | f1_vs_test_baseline |
|---|---|---|---|---|---|---|---|
| comem | 0.7273 | 0.7708 | -0.0436 | nan | nan | 0.7273 | 0.7708 |
| ditto_plm | 0.5860 | 0.5860 | 0.0000 | nan | nan | 0.5860 | 0.5860 |
| llm_matcher | 0.7488 | 0.7604 | -0.0116 | nan | nan | 0.7488 | 0.7604 |
| magellan | 0.8082 | 0.6024 | 0.2058 | nan | nan | 0.8082 | 0.6024 |

## Stage: em - per pair

| pair | member | f1 | f1_baseline | f1_delta |
|---|---|---|---|---|
| dbpedia_sales | comem | 0.7273 | 0.7708 | -0.0436 |
| dbpedia_sales | ditto_plm | 0.5860 | 0.5860 | 0.0000 |
| dbpedia_sales | llm_matcher | 0.7488 | 0.7604 | -0.0116 |
| dbpedia_sales | magellan | 0.8082 | 0.6024 | 0.2058 |

## Stage: fusion - per member

| member | overall_accuracy | overall_accuracy_baseline | overall_accuracy_delta |
|---|---|---|---|
| ESRB_llm_judge | 0.6463 | 0.6541 | -0.0078 |
| ESRB_most_complete | 0.6463 | 0.6541 | -0.0078 |
| ESRB_prefer_higher_trust | 0.6518 | 0.6541 | -0.0022 |
| ESRB_voting | 0.6474 | 0.6541 | -0.0067 |
| criticScore_fusionquery | 0.6474 | 0.6541 | -0.0067 |
| criticScore_huber_m_estimator | 0.6463 | 0.6541 | -0.0078 |
| criticScore_maximum | 0.6463 | 0.6541 | -0.0078 |
| criticScore_median | 0.6463 | 0.6541 | -0.0078 |
| criticScore_median_of_means | 0.6463 | 0.6541 | -0.0078 |
| criticScore_prefer_higher_trust | 0.6463 | 0.6552 | -0.0089 |
| criticScore_trimmed_mean | 0.6463 | 0.6541 | -0.0078 |
| developer_longest_string | 0.6352 | 0.6396 | -0.0044 |
| developer_most_complete | 0.6352 | 0.6396 | -0.0044 |
| developer_prefer_higher_trust | 0.6452 | 0.6541 | -0.0089 |
| developer_voting | 0.6474 | 0.6541 | -0.0067 |
| genres_intersection | 0.6129 | 0.6463 | -0.0334 |
| genres_intersection_k_sources | 0.6485 | 0.6885 | -0.0400 |
| genres_ltm | 0.6630 | 0.7052 | -0.0423 |
| genres_prefer_higher_trust | 0.6752 | 0.7164 | -0.0412 |
| genres_union | 0.6885 | 0.7286 | -0.0400 |
| genres_voting | 0.6474 | 0.6541 | -0.0067 |
| name_accusim | 0.6496 | 0.6563 | -0.0067 |
| name_casefusion | 0.6452 | 0.6518 | -0.0067 |
| name_fusionquery | 0.6529 | 0.6618 | -0.0089 |
| name_llm_judge | 0.6485 | 0.6563 | -0.0078 |
| name_longest_string | 0.6474 | 0.6552 | -0.0078 |
| name_most_complete | 0.6474 | 0.6552 | -0.0078 |
| name_prefer_higher_trust | 0.6541 | 0.6630 | -0.0089 |
| name_voting | 0.6474 | 0.6541 | -0.0067 |
| platform_most_complete | 0.6352 | 0.6429 | -0.0078 |
| platform_prefer_higher_trust | 0.6474 | 0.6541 | -0.0067 |
| platform_truthfinder | 0.6485 | 0.6541 | -0.0056 |
| platform_voting | 0.6474 | 0.6541 | -0.0067 |
| publisher_longest_string | 0.6474 | 0.6541 | -0.0067 |
| publisher_prefer_higher_trust | 0.6474 | 0.6541 | -0.0067 |
| publisher_voting | 0.6474 | 0.6541 | -0.0067 |
| releaseYear_earliest | 0.6563 | 0.6529 | 0.0033 |
| releaseYear_prefer_higher_trust | 0.6607 | 0.6607 | 0.0000 |
| releaseYear_voting | 0.6474 | 0.6541 | -0.0067 |
| userScore_huber_m_estimator | 0.6440 | 0.6529 | -0.0089 |
| userScore_maximum | 0.6474 | 0.6541 | -0.0067 |
| userScore_median | 0.6440 | 0.6529 | -0.0089 |
| userScore_median_of_means | 0.6440 | 0.6518 | -0.0078 |
| userScore_prefer_higher_trust | 0.6485 | 0.6563 | -0.0078 |
| userScore_trimmed_mean | 0.6440 | 0.6518 | -0.0078 |

## Stage: fusion - per attribute

| attribute | best_accuracy | baseline | delta | spread | spread_baseline | spread_delta |
|---|---|---|---|---|---|---|
| ESRB | 0.9600 | 0.9800 | -0.0200 | 0.0500 | 0.0000 | 0.0500 |
| criticScore | 0.2600 | 0.2700 | -0.0100 | 0.0100 | 0.0100 | 0.0000 |
| developer | 0.8100 | 0.8700 | -0.0600 | 0.1100 | 0.1300 | -0.0200 |
| genres | 0.8100 | 0.8400 | -0.0300 | 0.6800 | 0.7400 | -0.0600 |
| name | 0.9600 | 0.9700 | -0.0100 | 0.0800 | 0.1000 | -0.0200 |
| platform | 0.9500 | 0.9700 | -0.0200 | 0.1200 | 0.1000 | 0.0200 |
| publisher | 0.5600 | 0.6600 | -0.1000 | 0.0000 | 0.0000 | 0.0000 |
| releaseYear | 0.9000 | 0.9400 | -0.0400 | 0.1200 | 0.0700 | 0.0500 |
| userScore | 0.2222 | 0.2222 | 0.0000 | 0.0404 | 0.0404 | 0.0000 |
