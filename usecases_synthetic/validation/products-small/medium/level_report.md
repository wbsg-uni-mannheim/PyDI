# Validation report - products-small / medium

_Generated at 2026-05-15T17:58:23.976294+00:00_

- domain: `products-small`
- level: `medium`
- with_llm: `True`
- committee_versions: sm=`sm_committee.yaml@cb4a6847ac9e`, norm=`normalization_committee_products.yaml@baa6116ae974`, em_blocking=`em_blocking_committee_products.yaml@cf98a2e9359a`, em_matching=`em_matching_committee_products.yaml@4a4c34ff0cd2`, fusion=`fusion_committee_products.yaml@a060f29a6e87`

## Stage summary

| stage | metric | measured | baseline | delta |
|---|---|---|---|---|
| sm | macro_f1 | 0.7738 | 0.8138 | -0.0400 |
| norm | macro_f1 | 0.1949 | 0.2588 | -0.0639 |
| em_blocking | macro_pair_recall | 0.7991 | 0.7979 | 0.0012 |
| em_matching | macro_f1_vs_test | 0.5977 | 0.6004 | -0.0027 |
| fusion | overall_accuracy | 0.6700 | 0.7760 | -0.1060 |

## Stage: sm - per member

| member | f1 | f1_baseline | f1_delta |
|---|---|---|---|
| coma_hybrid | 0.8364 | 0.8571 | -0.0208 |
| duplicate_majority | 1.0000 | 1.0000 | 0.0000 |
| embedding_sbert | 0.8571 | 0.8000 | 0.0571 |
| instance_tf_cosine | 0.5965 | 0.6441 | -0.0476 |
| label_jw | 0.5532 | 0.8000 | -0.2468 |
| llm_openai | 0.8571 | 0.8571 | 0.0000 |
| magneto_slm_llm | 0.7164 | 0.7385 | -0.0220 |

## Stage: norm - per member

| member | macro_f1 | macro_f1_baseline | macro_f1_delta |
|---|---|---|---|
| llm_canonicalize | 0.0680 | 0.0704 | -0.0024 |
| number_locale | 0.1384 | 0.3114 | -0.1730 |
| text_clean | 0.3784 | 0.3945 | -0.0161 |

## Stage: em_blocking - per member

| member | pair_recall | pair_recall_baseline | pair_recall_delta | reduction_ratio | reduction_ratio_baseline |
|---|---|---|---|---|---|
| bm25_blocker | 0.9780 | 0.9773 | 0.0007 | 0.9319 | 0.9310 |
| embedding_blocker | 0.9389 | 0.9384 | 0.0005 | 0.9319 | 0.9310 |
| sorted_neighbourhood_blocker | 0.4381 | 0.4363 | 0.0017 | 0.9746 | 0.9742 |
| standard_blocker | 0.6438 | 0.6407 | 0.0031 | 0.9521 | 0.9519 |
| token_blocker | 0.9968 | 0.9968 | 0.0000 | 0.6412 | 0.6411 |

## Stage: em_matching - per member

| member | f1 | f1_baseline | f1_delta | f1_vs_val | f1_vs_val_baseline | f1_vs_test | f1_vs_test_baseline |
|---|---|---|---|---|---|---|---|
| comem | 0.4560 | 0.4891 | -0.0331 | nan | nan | 0.4560 | 0.4891 |
| ditto_plm | 0.7429 | 0.7019 | 0.0411 | 0.7385 | 0.7385 | 0.7429 | 0.7019 |
| llm_matcher | 0.5941 | 0.6102 | -0.0161 | nan | nan | 0.5941 | 0.6102 |

## Stage: em - per pair

| pair | member | f1 | f1_baseline | f1_delta |
|---|---|---|---|---|
| products_1_products_2 | comem | 0.4459 | 0.5896 | -0.1437 |
| products_1_products_2 | ditto_plm | 0.6667 | 0.6667 | 0.0000 |
| products_1_products_2 | llm_matcher | 0.8173 | 0.8381 | -0.0208 |
| products_1_products_3 | comem | 0.4138 | 0.4354 | -0.0216 |
| products_1_products_3 | ditto_plm | 0.8374 | 0.7143 | 0.1232 |
| products_1_products_3 | llm_matcher | 0.4564 | 0.4967 | -0.0404 |
| products_1_products_4 | comem | 0.5085 | 0.4425 | 0.0660 |
| products_1_products_4 | ditto_plm | 0.7246 | 0.7246 | 0.0000 |
| products_1_products_4 | llm_matcher | 0.5085 | 0.4957 | 0.0127 |

## Stage: fusion - per member

| member | overall_accuracy | overall_accuracy_baseline | overall_accuracy_delta |
|---|---|---|---|
| brand_llm_judge | 0.4740 | 0.5540 | -0.0800 |
| brand_most_complete | 0.4760 | 0.5580 | -0.0820 |
| brand_prefer_higher_trust | 0.4740 | 0.5520 | -0.0780 |
| brand_truthfinder | 0.4700 | 0.5580 | -0.0880 |
| brand_voting | 0.4740 | 0.5540 | -0.0800 |
| description_longest_string | 0.5480 | 0.6240 | -0.0760 |
| description_ltm | 0.4400 | 0.5100 | -0.0700 |
| description_most_complete | 0.5460 | 0.6220 | -0.0760 |
| description_prefer_higher_trust | 0.4740 | 0.5540 | -0.0800 |
| description_voting | 0.4740 | 0.5540 | -0.0800 |
| priceCurrency_most_complete | 0.3800 | 0.4600 | -0.0800 |
| priceCurrency_prefer_higher_trust | 0.5020 | 0.5840 | -0.0820 |
| priceCurrency_voting | 0.4740 | 0.5540 | -0.0800 |
| price_fusionquery | 0.4760 | 0.4940 | -0.0180 |
| price_huber_m_estimator | 0.4600 | 0.4580 | 0.0020 |
| price_maximum | 0.4700 | 0.5320 | -0.0620 |
| price_median | 0.4600 | 0.4600 | 0.0000 |
| price_median_of_means | 0.4580 | 0.4460 | 0.0120 |
| price_prefer_higher_trust | 0.4740 | 0.5520 | -0.0780 |
| price_trimmed_mean | 0.4580 | 0.4460 | 0.0120 |
| title_accusim | 0.4300 | 0.5380 | -0.1080 |
| title_casefusion | 0.4780 | 0.5540 | -0.0760 |
| title_fusionquery | 0.4360 | 0.5600 | -0.1240 |
| title_llm_judge | 0.4740 | 0.5540 | -0.0800 |
| title_longest_string | 0.5640 | 0.6740 | -0.1100 |
| title_most_complete | 0.5580 | 0.6680 | -0.1100 |
| title_prefer_higher_trust | 0.4780 | 0.5540 | -0.0760 |
| title_voting | 0.4740 | 0.5540 | -0.0800 |

## Stage: fusion - per attribute

| attribute | best_accuracy | baseline | delta | spread | spread_baseline | spread_delta |
|---|---|---|---|---|---|---|
| brand | 0.7600 | 0.7900 | -0.0300 | 0.0300 | 0.0300 | 0.0000 |
| description | 0.6700 | 0.7000 | -0.0300 | 0.5400 | 0.5700 | -0.0300 |
| price | 0.1100 | 0.5500 | -0.4400 | 0.0900 | 0.5300 | -0.4400 |
| priceCurrency | 0.9700 | 1.0000 | -0.0300 | 0.6100 | 0.6200 | -0.0100 |
| title | 0.8400 | 0.8400 | 0.0000 | 0.6700 | 0.6800 | -0.0100 |
