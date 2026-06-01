# Validation report - companies-small / hard

_Generated at 2026-05-15T17:35:36.212306+00:00_

- domain: `companies-small`
- level: `hard`
- with_llm: `True`
- committee_versions: sm=`sm_committee.yaml@cb4a6847ac9e`, norm=`normalization_committee_companies.yaml@b703da3ec1bf`, em_blocking=`em_blocking_committee.yaml@0ebf2f19e86c`, em_matching=`em_matching_committee.yaml@d3a60c2da539`, fusion=`fusion_committee.yaml@e94eb667ceb5`

## Stage summary

| stage | metric | measured | baseline | delta |
|---|---|---|---|---|
| sm | macro_f1 | 0.5940 | 0.6730 | -0.0790 |
| norm | macro_f1 | 0.4701 | 0.5574 | -0.0873 |
| em_blocking | macro_pair_recall | 0.5112 | 0.9572 | -0.4459 |
| em_matching | macro_f1_vs_test | 0.9031 | 0.8880 | 0.0151 |
| fusion | overall_accuracy | 0.6111 | 0.8810 | -0.2698 |

## Stage: sm - per member

| member | f1 | f1_baseline | f1_delta |
|---|---|---|---|
| coma_hybrid | 0.2500 | 0.5517 | -0.3017 |
| duplicate_majority | 0.8000 | 0.8000 | 0.0000 |
| embedding_sbert | 0.5789 | 0.7000 | -0.1211 |
| instance_tf_cosine | 0.5333 | 0.6471 | -0.1137 |
| label_jw | 0.3200 | 0.3200 | 0.0000 |
| llm_openai | 0.8837 | 0.8837 | 0.0000 |
| magneto_slm_llm | 0.7917 | 0.8085 | -0.0168 |

## Stage: norm - per member

| member | macro_f1 | macro_f1_baseline | macro_f1_delta |
|---|---|---|---|
| country_iso | 0.7792 | 0.8492 | -0.0699 |
| date_iso | 0.8000 | 0.8125 | -0.0125 |
| llm_canonicalize | 0.2019 | 0.2138 | -0.0119 |
| number_locale | 0.3194 | 0.7286 | -0.4091 |
| taxonomy_lookup | 0.0000 | 0.0000 | 0.0000 |
| text_clean | 0.7201 | 0.7405 | -0.0204 |

## Stage: em_blocking - per member

| member | pair_recall | pair_recall_baseline | pair_recall_delta | reduction_ratio | reduction_ratio_baseline |
|---|---|---|---|---|---|
| bm25_blocker | 0.5143 | 0.9551 | -0.4407 | 0.9730 | 0.9640 |
| embedding_blocker | 0.5319 | 0.9805 | -0.4485 | 0.9730 | 0.9640 |
| sc_block | 0.5263 | 0.9735 | -0.4473 | 0.9730 | 0.9640 |
| sorted_neighbourhood_blocker | 0.4949 | 0.9451 | -0.4501 | 0.9802 | 0.9771 |
| standard_blocker | 0.4874 | 0.9350 | -0.4477 | 0.9974 | 0.9979 |
| token_blocker | 0.5125 | 0.9538 | -0.4413 | 0.7787 | 0.9891 |

## Stage: em_matching - per member

| member | f1 | f1_baseline | f1_delta | f1_vs_val | f1_vs_val_baseline | f1_vs_test | f1_vs_test_baseline |
|---|---|---|---|---|---|---|---|
| comem | 0.8818 | 0.9073 | -0.0255 | nan | nan | 0.8818 | 0.9073 |
| ditto_plm | 0.8957 | 0.8466 | 0.0491 | 0.9199 | 0.9058 | 0.8957 | 0.8466 |
| llm_matcher | 0.9140 | 0.9086 | 0.0055 | nan | nan | 0.9140 | 0.9086 |
| magellan | 0.9210 | 0.8897 | 0.0313 | 0.9243 | 0.9101 | 0.9210 | 0.8897 |

## Stage: em - per pair

| pair | member | f1 | f1_baseline | f1_delta |
|---|---|---|---|---|
| forbes_dbpedia | comem | 0.8468 | 0.8750 | -0.0282 |
| forbes_dbpedia | ditto_plm | 0.9206 | 0.8550 | 0.0657 |
| forbes_dbpedia | llm_matcher | 0.8966 | 0.8947 | 0.0018 |
| forbes_dbpedia | magellan | 0.9242 | 0.9104 | 0.0138 |
| forbes_fullcontact | comem | 0.9167 | 0.9395 | -0.0229 |
| forbes_fullcontact | ditto_plm | 0.8707 | 0.8382 | 0.0325 |
| forbes_fullcontact | llm_matcher | 0.9315 | 0.9224 | 0.0091 |
| forbes_fullcontact | magellan | 0.9177 | 0.8689 | 0.0489 |

## Stage: fusion - per member

| member | overall_accuracy | overall_accuracy_baseline | overall_accuracy_delta |
|---|---|---|---|
| assets_fusionquery | 0.5641 | 0.8376 | -0.2735 |
| assets_huber_m_estimator | 0.5556 | 0.7863 | -0.2308 |
| assets_maximum | 0.5726 | 0.8034 | -0.2308 |
| assets_median | 0.5556 | 0.7863 | -0.2308 |
| assets_median_of_means | 0.5556 | 0.7863 | -0.2308 |
| assets_prefer_higher_trust | 0.5556 | 0.7949 | -0.2393 |
| assets_trimmed_mean | 0.5556 | 0.7863 | -0.2308 |
| city_llm_judge | 0.5726 | 0.8462 | -0.2735 |
| city_prefer_higher_trust | 0.5641 | 0.8632 | -0.2991 |
| city_shortest_string | 0.5897 | 0.8803 | -0.2906 |
| city_truthfinder | 0.5641 | 0.8632 | -0.2991 |
| city_voting | 0.5641 | 0.8376 | -0.2735 |
| country_favour_forbes | 0.5470 | 0.8376 | -0.2906 |
| country_llm_judge | 0.5470 | 0.7863 | -0.2393 |
| country_prefer_higher_trust | 0.5556 | 0.7863 | -0.2308 |
| country_truthfinder | 0.5897 | 0.8462 | -0.2564 |
| country_voting | 0.5641 | 0.8376 | -0.2735 |
| founded_earliest | 0.5641 | 0.8376 | -0.2735 |
| founded_prefer_higher_trust | 0.5641 | 0.8376 | -0.2735 |
| founded_voting | 0.5641 | 0.8376 | -0.2735 |
| keypeople_intersection | 0.5556 | 0.8120 | -0.2564 |
| keypeople_intersection_k_sources | 0.5556 | 0.8120 | -0.2564 |
| keypeople_ltm | 0.5641 | 0.8376 | -0.2735 |
| keypeople_prefer_higher_trust | 0.5641 | 0.8462 | -0.2821 |
| keypeople_union | 0.5556 | 0.8462 | -0.2906 |
| keypeople_voting | 0.5641 | 0.8376 | -0.2735 |
| name_accusim | 0.5556 | 0.8376 | -0.2821 |
| name_casefusion | 0.5385 | 0.8120 | -0.2735 |
| name_fusionquery | 0.5470 | 0.8034 | -0.2564 |
| name_llm_judge | 0.5470 | 0.8120 | -0.2650 |
| name_longest_string | 0.5385 | 0.8120 | -0.2735 |
| name_most_complete | 0.5385 | 0.8120 | -0.2735 |
| name_prefer_higher_trust | 0.5385 | 0.8120 | -0.2735 |
| name_voting | 0.5641 | 0.8376 | -0.2735 |
| revenue_fusionquery | 0.5641 | 0.8376 | -0.2735 |
| revenue_huber_m_estimator | 0.5641 | 0.7863 | -0.2222 |
| revenue_maximum | 0.5641 | 0.8291 | -0.2650 |
| revenue_median | 0.5641 | 0.7863 | -0.2222 |
| revenue_median_of_means | 0.5641 | 0.7863 | -0.2222 |
| revenue_prefer_higher_trust | 0.5641 | 0.8120 | -0.2479 |
| revenue_trimmed_mean | 0.5641 | 0.7863 | -0.2222 |

## Stage: fusion - per attribute

| attribute | best_accuracy | baseline | delta | spread | spread_baseline | spread_delta |
|---|---|---|---|---|---|---|
| assets | 0.2222 | 0.9444 | -0.7222 | 0.1111 | 0.3333 | -0.2222 |
| city | 0.7222 | 0.8333 | -0.1111 | 0.1667 | 0.2778 | -0.1111 |
| country | 0.9444 | 1.0000 | -0.0556 | 0.2778 | 0.3889 | -0.1111 |
| founded | 0.7778 | 0.9444 | -0.1667 | 0.0000 | 0.0000 | 0.0000 |
| keypeople | 0.4444 | 0.6667 | -0.2222 | 0.1111 | 0.4444 | -0.3333 |
| name | 0.9444 | 1.0000 | -0.0556 | 0.1667 | 0.2222 | -0.0556 |
| revenue | 0.2222 | 0.7778 | -0.5556 | 0.0000 | 0.3333 | -0.3333 |
