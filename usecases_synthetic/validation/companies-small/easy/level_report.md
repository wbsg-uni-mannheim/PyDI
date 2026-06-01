# Validation report - companies-small / easy

_Generated at 2026-05-15T17:06:43.068999+00:00_

- domain: `companies-small`
- level: `easy`
- with_llm: `True`
- committee_versions: sm=`sm_committee.yaml@cb4a6847ac9e`, norm=`normalization_committee_companies.yaml@b703da3ec1bf`, em_blocking=`em_blocking_committee.yaml@0ebf2f19e86c`, em_matching=`em_matching_committee.yaml@d3a60c2da539`, fusion=`fusion_committee.yaml@e94eb667ceb5`

## Stage summary

| stage | metric | measured | baseline | delta |
|---|---|---|---|---|
| sm | macro_f1 | 0.8303 | 0.6730 | 0.1573 |
| norm | macro_f1 | 0.4633 | 0.5574 | -0.0941 |
| em_blocking | macro_pair_recall | 0.9577 | 0.9572 | 0.0005 |
| em_matching | macro_f1_vs_test | 0.8923 | 0.8880 | 0.0043 |
| fusion | overall_accuracy | 0.6190 | 0.8810 | -0.2619 |

## Stage: sm - per member

| member | f1 | f1_baseline | f1_delta |
|---|---|---|---|
| coma_hybrid | 0.9524 | 0.5517 | 0.4007 |
| duplicate_majority | 0.8000 | 0.8000 | 0.0000 |
| embedding_sbert | 0.7692 | 0.7000 | 0.0692 |
| instance_tf_cosine | 0.6000 | 0.6471 | -0.0471 |
| label_jw | 0.9091 | 0.3200 | 0.5891 |
| llm_openai | 0.9302 | 0.8837 | 0.0465 |
| magneto_slm_llm | 0.8511 | 0.8085 | 0.0426 |

## Stage: norm - per member

| member | macro_f1 | macro_f1_baseline | macro_f1_delta |
|---|---|---|---|
| country_iso | 0.8865 | 0.8492 | 0.0373 |
| date_iso | 0.8544 | 0.8125 | 0.0419 |
| llm_canonicalize | 0.2138 | 0.2138 | 0.0000 |
| number_locale | 0.0373 | 0.7286 | -0.6913 |
| taxonomy_lookup | 0.0000 | 0.0000 | 0.0000 |
| text_clean | 0.7879 | 0.7405 | 0.0474 |

## Stage: em_blocking - per member

| member | pair_recall | pair_recall_baseline | pair_recall_delta | reduction_ratio | reduction_ratio_baseline |
|---|---|---|---|---|---|
| bm25_blocker | 0.9557 | 0.9551 | 0.0007 | 0.9733 | 0.9640 |
| embedding_blocker | 0.9805 | 0.9805 | 0.0000 | 0.9733 | 0.9640 |
| sc_block | 0.9759 | 0.9735 | 0.0024 | 0.9733 | 0.9640 |
| sorted_neighbourhood_blocker | 0.9451 | 0.9451 | 0.0000 | 0.9824 | 0.9771 |
| standard_blocker | 0.9350 | 0.9350 | 0.0000 | 0.9981 | 0.9979 |
| token_blocker | 0.9538 | 0.9538 | 0.0000 | 0.9910 | 0.9891 |

## Stage: em_matching - per member

| member | f1 | f1_baseline | f1_delta | f1_vs_val | f1_vs_val_baseline | f1_vs_test | f1_vs_test_baseline |
|---|---|---|---|---|---|---|---|
| comem | 0.8924 | 0.9073 | -0.0149 | nan | nan | 0.8924 | 0.9073 |
| ditto_plm | 0.8466 | 0.8466 | 0.0000 | 0.9058 | 0.9058 | 0.8466 | 0.8466 |
| llm_matcher | 0.8940 | 0.9086 | -0.0145 | nan | nan | 0.8940 | 0.9086 |
| magellan | 0.9364 | 0.8897 | 0.0467 | 0.9481 | 0.9101 | 0.9364 | 0.8897 |

## Stage: em - per pair

| pair | member | f1 | f1_baseline | f1_delta |
|---|---|---|---|---|
| forbes_dbpedia | comem | 0.8545 | 0.8750 | -0.0205 |
| forbes_dbpedia | ditto_plm | 0.8550 | 0.8550 | 0.0000 |
| forbes_dbpedia | llm_matcher | 0.8468 | 0.8947 | -0.0479 |
| forbes_dbpedia | magellan | 0.9921 | 0.9104 | 0.0817 |
| forbes_fullcontact | comem | 0.9302 | 0.9395 | -0.0093 |
| forbes_fullcontact | ditto_plm | 0.8382 | 0.8382 | 0.0000 |
| forbes_fullcontact | llm_matcher | 0.9412 | 0.9224 | 0.0188 |
| forbes_fullcontact | magellan | 0.8807 | 0.8689 | 0.0118 |

## Stage: fusion - per member

| member | overall_accuracy | overall_accuracy_baseline | overall_accuracy_delta |
|---|---|---|---|
| assets_fusionquery | 0.5299 | 0.8376 | -0.3077 |
| assets_huber_m_estimator | 0.5299 | 0.7863 | -0.2564 |
| assets_maximum | 0.5385 | 0.8034 | -0.2650 |
| assets_median | 0.5299 | 0.7863 | -0.2564 |
| assets_median_of_means | 0.5299 | 0.7863 | -0.2564 |
| assets_prefer_higher_trust | 0.5385 | 0.7949 | -0.2564 |
| assets_trimmed_mean | 0.5299 | 0.7863 | -0.2564 |
| city_llm_judge | 0.5556 | 0.8462 | -0.2906 |
| city_prefer_higher_trust | 0.5897 | 0.8632 | -0.2735 |
| city_shortest_string | 0.5897 | 0.8803 | -0.2906 |
| city_truthfinder | 0.5897 | 0.8632 | -0.2735 |
| city_voting | 0.5299 | 0.8376 | -0.3077 |
| country_favour_forbes | 0.5299 | 0.8376 | -0.3077 |
| country_llm_judge | 0.5299 | 0.7863 | -0.2564 |
| country_prefer_higher_trust | 0.5299 | 0.7863 | -0.2564 |
| country_truthfinder | 0.5385 | 0.8462 | -0.3077 |
| country_voting | 0.5299 | 0.8376 | -0.3077 |
| founded_earliest | 0.5299 | 0.8376 | -0.3077 |
| founded_prefer_higher_trust | 0.5299 | 0.8376 | -0.3077 |
| founded_voting | 0.5299 | 0.8376 | -0.3077 |
| keypeople_intersection | 0.5299 | 0.8120 | -0.2821 |
| keypeople_intersection_k_sources | 0.5299 | 0.8120 | -0.2821 |
| keypeople_ltm | 0.5299 | 0.8376 | -0.3077 |
| keypeople_prefer_higher_trust | 0.5299 | 0.8462 | -0.3162 |
| keypeople_union | 0.5299 | 0.8462 | -0.3162 |
| keypeople_voting | 0.5299 | 0.8376 | -0.3077 |
| name_accusim | 0.5299 | 0.8376 | -0.3077 |
| name_casefusion | 0.5556 | 0.8120 | -0.2564 |
| name_fusionquery | 0.5470 | 0.8034 | -0.2564 |
| name_llm_judge | 0.5299 | 0.8120 | -0.2821 |
| name_longest_string | 0.5299 | 0.8120 | -0.2821 |
| name_most_complete | 0.5299 | 0.8120 | -0.2821 |
| name_prefer_higher_trust | 0.5556 | 0.8120 | -0.2564 |
| name_voting | 0.5299 | 0.8376 | -0.3077 |
| revenue_fusionquery | 0.5299 | 0.8376 | -0.3077 |
| revenue_huber_m_estimator | 0.5299 | 0.7863 | -0.2564 |
| revenue_maximum | 0.5299 | 0.8291 | -0.2991 |
| revenue_median | 0.5299 | 0.7863 | -0.2564 |
| revenue_median_of_means | 0.5299 | 0.7863 | -0.2564 |
| revenue_prefer_higher_trust | 0.5299 | 0.8120 | -0.2821 |
| revenue_trimmed_mean | 0.5299 | 0.7863 | -0.2564 |

## Stage: fusion - per attribute

| attribute | best_accuracy | baseline | delta | spread | spread_baseline | spread_delta |
|---|---|---|---|---|---|---|
| assets | 0.0556 | 0.9444 | -0.8889 | 0.0556 | 0.3333 | -0.2778 |
| city | 0.8333 | 0.8333 | 0.0000 | 0.3889 | 0.2778 | 0.1111 |
| country | 1.0000 | 1.0000 | 0.0000 | 0.0556 | 0.3889 | -0.3333 |
| founded | 0.9444 | 0.9444 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| keypeople | 0.4444 | 0.6667 | -0.2222 | 0.0000 | 0.4444 | -0.4444 |
| name | 1.0000 | 1.0000 | 0.0000 | 0.1667 | 0.2222 | -0.0556 |
| revenue | 0.0556 | 0.7778 | -0.7222 | 0.0000 | 0.3333 | -0.3333 |
