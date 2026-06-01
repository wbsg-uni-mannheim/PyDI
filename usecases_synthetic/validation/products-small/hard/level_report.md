# Validation report - products-small / hard

_Generated at 2026-05-15T18:06:55.053619+00:00_

- domain: `products-small`
- level: `hard`
- with_llm: `True`
- committee_versions: sm=`sm_committee.yaml@cb4a6847ac9e`, norm=`normalization_committee_products.yaml@baa6116ae974`, em_blocking=`em_blocking_committee_products.yaml@cf98a2e9359a`, em_matching=`em_matching_committee_products.yaml@4a4c34ff0cd2`, fusion=`fusion_committee_products.yaml@a060f29a6e87`

## Stage summary

| stage | metric | measured | baseline | delta |
|---|---|---|---|---|
| sm | macro_f1 | 0.7215 | 0.8138 | -0.0924 |
| norm | macro_f1 | 0.1736 | 0.2588 | -0.0852 |
| em_blocking | macro_pair_recall | 0.1395 | 0.7979 | -0.6584 |
| em_matching | macro_f1_vs_test | 0.5796 | 0.6004 | -0.0208 |
| fusion | overall_accuracy | 0.7200 | 0.7760 | -0.0560 |

## Stage: sm - per member

| member | f1 | f1_baseline | f1_delta |
|---|---|---|---|
| coma_hybrid | 0.7451 | 0.8571 | -0.1120 |
| duplicate_majority | 0.9787 | 1.0000 | -0.0213 |
| embedding_sbert | 0.8511 | 0.8000 | 0.0511 |
| instance_tf_cosine | 0.5965 | 0.6441 | -0.0476 |
| label_jw | 0.3158 | 0.8000 | -0.4842 |
| llm_openai | 0.8571 | 0.8571 | 0.0000 |
| magneto_slm_llm | 0.7059 | 0.7385 | -0.0326 |

## Stage: norm - per member

| member | macro_f1 | macro_f1_baseline | macro_f1_delta |
|---|---|---|---|
| llm_canonicalize | 0.0538 | 0.0704 | -0.0166 |
| number_locale | 0.1609 | 0.3114 | -0.1506 |
| text_clean | 0.3062 | 0.3945 | -0.0883 |

## Stage: em_blocking - per member

| member | pair_recall | pair_recall_baseline | pair_recall_delta | reduction_ratio | reduction_ratio_baseline |
|---|---|---|---|---|---|
| bm25_blocker | 0.1662 | 0.9773 | -0.8112 | 0.9331 | 0.9310 |
| embedding_blocker | 0.1584 | 0.9384 | -0.7801 | 0.9331 | 0.9310 |
| sorted_neighbourhood_blocker | 0.0927 | 0.4363 | -0.3436 | 0.9578 | 0.9742 |
| standard_blocker | 0.1116 | 0.6407 | -0.5291 | 0.9564 | 0.9519 |
| token_blocker | 0.1686 | 0.9968 | -0.8282 | 0.6519 | 0.6411 |

## Stage: em_matching - per member

| member | f1 | f1_baseline | f1_delta | f1_vs_val | f1_vs_val_baseline | f1_vs_test | f1_vs_test_baseline |
|---|---|---|---|---|---|---|---|
| comem | 0.5939 | 0.4891 | 0.1048 | nan | nan | 0.5939 | 0.4891 |
| ditto_plm | 0.4255 | 0.7019 | -0.2764 | 0.7338 | 0.7385 | 0.4255 | 0.7019 |
| llm_matcher | 0.7195 | 0.6102 | 0.1094 | nan | nan | 0.7195 | 0.6102 |

## Stage: em - per pair

| pair | member | f1 | f1_baseline | f1_delta |
|---|---|---|---|---|
| products_1_products_2 | comem | 0.6197 | 0.5896 | 0.0301 |
| products_1_products_2 | ditto_plm | 0.4444 | 0.6667 | -0.2222 |
| products_1_products_2 | llm_matcher | 0.8736 | 0.8381 | 0.0355 |
| products_1_products_3 | comem | 0.5588 | 0.4354 | 0.1234 |
| products_1_products_3 | ditto_plm | 0.4615 | 0.7143 | -0.2527 |
| products_1_products_3 | llm_matcher | 0.6389 | 0.4967 | 0.1422 |
| products_1_products_4 | comem | 0.6032 | 0.4425 | 0.1607 |
| products_1_products_4 | ditto_plm | 0.3704 | 0.7246 | -0.3543 |
| products_1_products_4 | llm_matcher | 0.6462 | 0.4957 | 0.1504 |

## Stage: fusion - per member

| member | overall_accuracy | overall_accuracy_baseline | overall_accuracy_delta |
|---|---|---|---|
| brand_llm_judge | 0.5000 | 0.5540 | -0.0540 |
| brand_most_complete | 0.5000 | 0.5580 | -0.0580 |
| brand_prefer_higher_trust | 0.5000 | 0.5520 | -0.0520 |
| brand_truthfinder | 0.4800 | 0.5580 | -0.0780 |
| brand_voting | 0.5000 | 0.5540 | -0.0540 |
| description_longest_string | 0.5800 | 0.6240 | -0.0440 |
| description_ltm | 0.4600 | 0.5100 | -0.0500 |
| description_most_complete | 0.5600 | 0.6220 | -0.0620 |
| description_prefer_higher_trust | 0.5000 | 0.5540 | -0.0540 |
| description_voting | 0.5000 | 0.5540 | -0.0540 |
| priceCurrency_most_complete | 0.3800 | 0.4600 | -0.0800 |
| priceCurrency_prefer_higher_trust | 0.5000 | 0.5840 | -0.0840 |
| priceCurrency_voting | 0.5000 | 0.5540 | -0.0540 |
| price_fusionquery | 0.5000 | 0.4940 | 0.0060 |
| price_huber_m_estimator | 0.4600 | 0.4580 | 0.0020 |
| price_maximum | 0.5000 | 0.5320 | -0.0320 |
| price_median | 0.4600 | 0.4600 | 0.0000 |
| price_median_of_means | 0.4600 | 0.4460 | 0.0140 |
| price_prefer_higher_trust | 0.5000 | 0.5520 | -0.0520 |
| price_trimmed_mean | 0.4600 | 0.4460 | 0.0140 |
| title_accusim | 0.5000 | 0.5380 | -0.0380 |
| title_casefusion | 0.5000 | 0.5540 | -0.0540 |
| title_fusionquery | 0.5000 | 0.5600 | -0.0600 |
| title_llm_judge | 0.5000 | 0.5540 | -0.0540 |
| title_longest_string | 0.6400 | 0.6740 | -0.0340 |
| title_most_complete | 0.6200 | 0.6680 | -0.0480 |
| title_prefer_higher_trust | 0.5000 | 0.5540 | -0.0540 |
| title_voting | 0.5000 | 0.5540 | -0.0540 |

## Stage: fusion - per attribute

| attribute | best_accuracy | baseline | delta | spread | spread_baseline | spread_delta |
|---|---|---|---|---|---|---|
| brand | 0.8000 | 0.7900 | 0.0100 | 0.1000 | 0.0300 | 0.0700 |
| description | 0.7000 | 0.7000 | 0.0000 | 0.6000 | 0.5700 | 0.0300 |
| price | 0.3000 | 0.5500 | -0.2500 | 0.2000 | 0.5300 | -0.3300 |
| priceCurrency | 0.9000 | 1.0000 | -0.1000 | 0.6000 | 0.6200 | -0.0200 |
| title | 0.9000 | 0.8400 | 0.0600 | 0.7000 | 0.6800 | 0.0200 |
