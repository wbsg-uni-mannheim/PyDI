# Validation report - companies-small / medium

_Generated at 2026-05-15T17:21:30.946882+00:00_

- domain: `companies-small`
- level: `medium`
- with_llm: `True`
- committee_versions: sm=`sm_committee.yaml@cb4a6847ac9e`, norm=`normalization_committee_companies.yaml@b703da3ec1bf`, em_blocking=`em_blocking_committee.yaml@0ebf2f19e86c`, em_matching=`em_matching_committee.yaml@d3a60c2da539`, fusion=`fusion_committee.yaml@e94eb667ceb5`

## Stage summary

| stage | metric | measured | baseline | delta |
|---|---|---|---|---|
| sm | macro_f1 | 0.7023 | 0.6730 | 0.0293 |
| norm | macro_f1 | 0.4949 | 0.5574 | -0.0625 |
| em_blocking | macro_pair_recall | 0.9539 | 0.9572 | -0.0033 |
| em_matching | macro_f1_vs_test | 0.8764 | 0.8880 | -0.0117 |
| fusion | overall_accuracy | 0.7857 | 0.8810 | -0.0952 |

## Stage: sm - per member

| member | f1 | f1_baseline | f1_delta |
|---|---|---|---|
| coma_hybrid | 0.6250 | 0.5517 | 0.0733 |
| duplicate_majority | 0.7647 | 0.8000 | -0.0353 |
| embedding_sbert | 0.6286 | 0.7000 | -0.0714 |
| instance_tf_cosine | 0.5625 | 0.6471 | -0.0846 |
| label_jw | 0.6857 | 0.3200 | 0.3657 |
| llm_openai | 0.8837 | 0.8837 | 0.0000 |
| magneto_slm_llm | 0.7660 | 0.8085 | -0.0426 |

## Stage: norm - per member

| member | macro_f1 | macro_f1_baseline | macro_f1_delta |
|---|---|---|---|
| country_iso | 0.7953 | 0.8492 | -0.0538 |
| date_iso | 0.8125 | 0.8125 | 0.0000 |
| llm_canonicalize | 0.2087 | 0.2138 | -0.0051 |
| number_locale | 0.4219 | 0.7286 | -0.3067 |
| taxonomy_lookup | 0.0000 | 0.0000 | 0.0000 |
| text_clean | 0.7310 | 0.7405 | -0.0095 |

## Stage: em_blocking - per member

| member | pair_recall | pair_recall_baseline | pair_recall_delta | reduction_ratio | reduction_ratio_baseline |
|---|---|---|---|---|---|
| bm25_blocker | 0.9538 | 0.9551 | -0.0012 | 0.9695 | 0.9640 |
| embedding_blocker | 0.9805 | 0.9805 | 0.0000 | 0.9695 | 0.9640 |
| sc_block | 0.9735 | 0.9735 | 0.0000 | 0.9695 | 0.9640 |
| sorted_neighbourhood_blocker | 0.9343 | 0.9451 | -0.0107 | 0.9804 | 0.9771 |
| standard_blocker | 0.9275 | 0.9350 | -0.0076 | 0.9976 | 0.9979 |
| token_blocker | 0.9538 | 0.9538 | 0.0000 | 0.9318 | 0.9891 |

## Stage: em_matching - per member

| member | f1 | f1_baseline | f1_delta | f1_vs_val | f1_vs_val_baseline | f1_vs_test | f1_vs_test_baseline |
|---|---|---|---|---|---|---|---|
| comem | 0.8785 | 0.9073 | -0.0288 | nan | nan | 0.8785 | 0.9073 |
| ditto_plm | 0.8466 | 0.8466 | 0.0000 | 0.9058 | 0.9058 | 0.8466 | 0.8466 |
| llm_matcher | 0.8729 | 0.9086 | -0.0357 | nan | nan | 0.8729 | 0.9086 |
| magellan | 0.9075 | 0.8897 | 0.0178 | 0.9196 | 0.9101 | 0.9075 | 0.8897 |

## Stage: em - per pair

| pair | member | f1 | f1_baseline | f1_delta |
|---|---|---|---|---|
| forbes_dbpedia | comem | 0.8224 | 0.8750 | -0.0526 |
| forbes_dbpedia | ditto_plm | 0.8550 | 0.8550 | 0.0000 |
| forbes_dbpedia | llm_matcher | 0.8333 | 0.8947 | -0.0614 |
| forbes_dbpedia | magellan | 0.9242 | 0.9104 | 0.0138 |
| forbes_fullcontact | comem | 0.9346 | 0.9395 | -0.0050 |
| forbes_fullcontact | ditto_plm | 0.8382 | 0.8382 | 0.0000 |
| forbes_fullcontact | llm_matcher | 0.9124 | 0.9224 | -0.0099 |
| forbes_fullcontact | magellan | 0.8907 | 0.8689 | 0.0218 |

## Stage: fusion - per member

| member | overall_accuracy | overall_accuracy_baseline | overall_accuracy_delta |
|---|---|---|---|
| assets_fusionquery | 0.6838 | 0.8376 | -0.1538 |
| assets_huber_m_estimator | 0.6838 | 0.7863 | -0.1026 |
| assets_maximum | 0.7009 | 0.8034 | -0.1026 |
| assets_median | 0.6838 | 0.7863 | -0.1026 |
| assets_median_of_means | 0.6838 | 0.7863 | -0.1026 |
| assets_prefer_higher_trust | 0.7265 | 0.7949 | -0.0684 |
| assets_trimmed_mean | 0.6838 | 0.7863 | -0.1026 |
| city_llm_judge | 0.7009 | 0.8462 | -0.1453 |
| city_prefer_higher_trust | 0.6923 | 0.8632 | -0.1709 |
| city_shortest_string | 0.7179 | 0.8803 | -0.1624 |
| city_truthfinder | 0.6923 | 0.8632 | -0.1709 |
| city_voting | 0.6838 | 0.8376 | -0.1538 |
| country_favour_forbes | 0.6752 | 0.8376 | -0.1624 |
| country_llm_judge | 0.6410 | 0.7863 | -0.1453 |
| country_prefer_higher_trust | 0.6325 | 0.7863 | -0.1538 |
| country_truthfinder | 0.6752 | 0.8462 | -0.1709 |
| country_voting | 0.6838 | 0.8376 | -0.1538 |
| founded_earliest | 0.6838 | 0.8376 | -0.1538 |
| founded_prefer_higher_trust | 0.6838 | 0.8376 | -0.1538 |
| founded_voting | 0.6838 | 0.8376 | -0.1538 |
| keypeople_intersection | 0.6752 | 0.8120 | -0.1368 |
| keypeople_intersection_k_sources | 0.6752 | 0.8120 | -0.1368 |
| keypeople_ltm | 0.6838 | 0.8376 | -0.1538 |
| keypeople_prefer_higher_trust | 0.6838 | 0.8462 | -0.1624 |
| keypeople_union | 0.6752 | 0.8462 | -0.1709 |
| keypeople_voting | 0.6838 | 0.8376 | -0.1538 |
| name_accusim | 0.6838 | 0.8376 | -0.1538 |
| name_casefusion | 0.6752 | 0.8120 | -0.1368 |
| name_fusionquery | 0.6752 | 0.8034 | -0.1282 |
| name_llm_judge | 0.6667 | 0.8120 | -0.1453 |
| name_longest_string | 0.6667 | 0.8120 | -0.1453 |
| name_most_complete | 0.6667 | 0.8120 | -0.1453 |
| name_prefer_higher_trust | 0.6752 | 0.8120 | -0.1368 |
| name_voting | 0.6838 | 0.8376 | -0.1538 |
| revenue_fusionquery | 0.6838 | 0.8376 | -0.1538 |
| revenue_huber_m_estimator | 0.6752 | 0.7863 | -0.1111 |
| revenue_maximum | 0.7265 | 0.8291 | -0.1026 |
| revenue_median | 0.6752 | 0.7863 | -0.1111 |
| revenue_median_of_means | 0.6752 | 0.7863 | -0.1111 |
| revenue_prefer_higher_trust | 0.7179 | 0.8120 | -0.0940 |
| revenue_trimmed_mean | 0.6752 | 0.7863 | -0.1111 |

## Stage: fusion - per attribute

| attribute | best_accuracy | baseline | delta | spread | spread_baseline | spread_delta |
|---|---|---|---|---|---|---|
| assets | 0.9444 | 0.9444 | 0.0000 | 0.2778 | 0.3333 | -0.0556 |
| city | 0.7778 | 0.8333 | -0.0556 | 0.2222 | 0.2778 | -0.0556 |
| country | 1.0000 | 1.0000 | 0.0000 | 0.3333 | 0.3889 | -0.0556 |
| founded | 0.9444 | 0.9444 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| keypeople | 0.5556 | 0.6667 | -0.1111 | 0.1111 | 0.4444 | -0.3333 |
| name | 0.9444 | 1.0000 | -0.0556 | 0.1111 | 0.2222 | -0.1111 |
| revenue | 0.3333 | 0.7778 | -0.4444 | 0.3333 | 0.3333 | -0.0000 |
