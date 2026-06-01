# Validation report - games / easy

_Generated at 2026-05-16T17:01:53.246208+00:00_

- domain: `games`
- level: `easy`
- with_llm: `True`
- committee_versions: sm=`sm_committee.yaml@cb4a6847ac9e`, norm=`normalization_committee_games.yaml@ec8e48a06ee7`, em_blocking=`em_blocking_committee_games.yaml@e4adebfad54e`, em_matching=`em_matching_committee_games.yaml@d2adcee14e35`, fusion=`fusion_committee_games.yaml@aef79ffb0514`

## Stage summary

| stage | metric | measured | baseline | delta |
|---|---|---|---|---|
| sm | macro_f1 | 0.8608 | 0.7267 | 0.1341 |
| norm | macro_f1 | 0.6743 | 0.6821 | -0.0077 |
| em_blocking | macro_pair_recall | 0.9986 | 0.9986 | 0.0000 |
| em_matching | macro_f1_vs_test | 0.7328 | 0.6799 | 0.0529 |
| fusion | overall_accuracy | 0.7425 | 0.7469 | -0.0044 |

## Stage: sm - per member

| member | f1 | f1_baseline | f1_delta |
|---|---|---|---|
| coma_hybrid | 1.0000 | 0.7556 | 0.2444 |
| duplicate_majority | 0.6500 | 0.6829 | -0.0329 |
| embedding_sbert | 0.7778 | 0.6275 | 0.1503 |
| instance_tf_cosine | 0.6667 | 0.7059 | -0.0392 |
| label_jw | 1.0000 | 0.3333 | 0.6667 |
| llm_openai | 1.0000 | 1.0000 | 0.0000 |
| magneto_slm_llm | 0.9310 | 0.9818 | -0.0508 |

## Stage: norm - per member

| member | macro_f1 | macro_f1_baseline | macro_f1_delta |
|---|---|---|---|
| date_iso | 0.9514 | 0.9750 | -0.0236 |
| llm_canonicalize | 0.5818 | 0.5853 | -0.0035 |
| number_locale | 0.4359 | 0.4385 | -0.0026 |
| taxonomy_lookup | 0.6242 | 0.6276 | -0.0034 |
| text_clean | 0.7784 | 0.7840 | -0.0056 |

## Stage: em_blocking - per member

| member | pair_recall | pair_recall_baseline | pair_recall_delta | reduction_ratio | reduction_ratio_baseline |
|---|---|---|---|---|---|
| bm25_blocker | 1.0000 | 1.0000 | 0.0000 | 0.9954 | 0.9937 |
| embedding_blocker | 1.0000 | 1.0000 | 0.0000 | 0.9954 | 0.9937 |
| sc_block | 1.0000 | 1.0000 | 0.0000 | 0.9954 | 0.9937 |
| sorted_neighbourhood_blocker | 0.9914 | 0.9914 | 0.0000 | 0.9990 | 0.9986 |
| standard_blocker | 1.0000 | 1.0000 | 0.0000 | 0.9924 | 0.9987 |
| token_blocker | 1.0000 | 1.0000 | 0.0000 | 0.9618 | 0.9533 |

## Stage: em_matching - per member

| member | f1 | f1_baseline | f1_delta | f1_vs_val | f1_vs_val_baseline | f1_vs_test | f1_vs_test_baseline |
|---|---|---|---|---|---|---|---|
| comem | 0.7380 | 0.7708 | -0.0329 | nan | nan | 0.7380 | 0.7708 |
| ditto_plm | 0.5860 | 0.5860 | 0.0000 | nan | nan | 0.5860 | 0.5860 |
| llm_matcher | 0.7419 | 0.7604 | -0.0185 | nan | nan | 0.7419 | 0.7604 |
| magellan | 0.8654 | 0.6024 | 0.2630 | nan | nan | 0.8654 | 0.6024 |

## Stage: em - per pair

| pair | member | f1 | f1_baseline | f1_delta |
|---|---|---|---|---|
| dbpedia_sales | comem | 0.7380 | 0.7708 | -0.0329 |
| dbpedia_sales | ditto_plm | 0.5860 | 0.5860 | 0.0000 |
| dbpedia_sales | llm_matcher | 0.7419 | 0.7604 | -0.0185 |
| dbpedia_sales | magellan | 0.8654 | 0.6024 | 0.2630 |

## Stage: fusion - per member

| member | overall_accuracy | overall_accuracy_baseline | overall_accuracy_delta |
|---|---|---|---|
| ESRB_llm_judge | 0.6440 | 0.6541 | -0.0100 |
| ESRB_most_complete | 0.6440 | 0.6541 | -0.0100 |
| ESRB_prefer_higher_trust | 0.6440 | 0.6541 | -0.0100 |
| ESRB_voting | 0.6440 | 0.6541 | -0.0100 |
| criticScore_fusionquery | 0.6440 | 0.6541 | -0.0100 |
| criticScore_huber_m_estimator | 0.6440 | 0.6541 | -0.0100 |
| criticScore_maximum | 0.6440 | 0.6541 | -0.0100 |
| criticScore_median | 0.6440 | 0.6541 | -0.0100 |
| criticScore_median_of_means | 0.6440 | 0.6541 | -0.0100 |
| criticScore_prefer_higher_trust | 0.6452 | 0.6552 | -0.0100 |
| criticScore_trimmed_mean | 0.6440 | 0.6541 | -0.0100 |
| developer_longest_string | 0.6318 | 0.6396 | -0.0078 |
| developer_most_complete | 0.6318 | 0.6396 | -0.0078 |
| developer_prefer_higher_trust | 0.6452 | 0.6541 | -0.0089 |
| developer_voting | 0.6440 | 0.6541 | -0.0100 |
| genres_intersection | 0.6374 | 0.6463 | -0.0089 |
| genres_intersection_k_sources | 0.6752 | 0.6885 | -0.0133 |
| genres_ltm | 0.6774 | 0.7052 | -0.0278 |
| genres_prefer_higher_trust | 0.7041 | 0.7164 | -0.0122 |
| genres_union | 0.7175 | 0.7286 | -0.0111 |
| genres_voting | 0.6440 | 0.6541 | -0.0100 |
| name_accusim | 0.6463 | 0.6563 | -0.0100 |
| name_casefusion | 0.6429 | 0.6518 | -0.0089 |
| name_fusionquery | 0.6485 | 0.6618 | -0.0133 |
| name_llm_judge | 0.6463 | 0.6563 | -0.0100 |
| name_longest_string | 0.6474 | 0.6552 | -0.0078 |
| name_most_complete | 0.6474 | 0.6552 | -0.0078 |
| name_prefer_higher_trust | 0.6496 | 0.6630 | -0.0133 |
| name_voting | 0.6440 | 0.6541 | -0.0100 |
| platform_most_complete | 0.6340 | 0.6429 | -0.0089 |
| platform_prefer_higher_trust | 0.6452 | 0.6541 | -0.0089 |
| platform_truthfinder | 0.6463 | 0.6541 | -0.0078 |
| platform_voting | 0.6440 | 0.6541 | -0.0100 |
| publisher_longest_string | 0.6440 | 0.6541 | -0.0100 |
| publisher_prefer_higher_trust | 0.6440 | 0.6541 | -0.0100 |
| publisher_voting | 0.6440 | 0.6541 | -0.0100 |
| releaseYear_earliest | 0.6496 | 0.6529 | -0.0033 |
| releaseYear_prefer_higher_trust | 0.6574 | 0.6607 | -0.0033 |
| releaseYear_voting | 0.6440 | 0.6541 | -0.0100 |
| userScore_huber_m_estimator | 0.6418 | 0.6529 | -0.0111 |
| userScore_maximum | 0.6440 | 0.6541 | -0.0100 |
| userScore_median | 0.6418 | 0.6529 | -0.0111 |
| userScore_median_of_means | 0.6407 | 0.6518 | -0.0111 |
| userScore_prefer_higher_trust | 0.6463 | 0.6563 | -0.0100 |
| userScore_trimmed_mean | 0.6407 | 0.6518 | -0.0111 |

## Stage: fusion - per attribute

| attribute | best_accuracy | baseline | delta | spread | spread_baseline | spread_delta |
|---|---|---|---|---|---|---|
| ESRB | 0.9800 | 0.9800 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| criticScore | 0.2700 | 0.2700 | 0.0000 | 0.0100 | 0.0100 | 0.0000 |
| developer | 0.8600 | 0.8700 | -0.0100 | 0.1200 | 0.1300 | -0.0100 |
| genres | 0.8400 | 0.8400 | 0.0000 | 0.7200 | 0.7400 | -0.0200 |
| name | 0.9700 | 0.9700 | 0.0000 | 0.0600 | 0.1000 | -0.0400 |
| platform | 0.9800 | 0.9700 | 0.0100 | 0.1100 | 0.1000 | 0.0100 |
| publisher | 0.6300 | 0.6600 | -0.0300 | 0.0000 | 0.0000 | 0.0000 |
| releaseYear | 0.9300 | 0.9400 | -0.0100 | 0.1200 | 0.0700 | 0.0500 |
| userScore | 0.2222 | 0.2222 | 0.0000 | 0.0505 | 0.0404 | 0.0101 |
