# Validation report - products-small / easy

_Generated at 2026-05-15T17:46:43.458135+00:00_

- domain: `products-small`
- level: `easy`
- with_llm: `True`
- committee_versions: sm=`sm_committee.yaml@cb4a6847ac9e`, norm=`normalization_committee_products.yaml@baa6116ae974`, em_blocking=`em_blocking_committee_products.yaml@cf98a2e9359a`, em_matching=`em_matching_committee_products.yaml@4a4c34ff0cd2`, fusion=`fusion_committee_products.yaml@a060f29a6e87`

## Stage summary

| stage | metric | measured | baseline | delta |
|---|---|---|---|---|
| sm | macro_f1 | 0.8143 | 0.8138 | 0.0005 |
| norm | macro_f1 | 0.2986 | 0.2588 | 0.0398 |
| em_blocking | macro_pair_recall | 0.7971 | 0.7979 | -0.0009 |
| em_matching | macro_f1_vs_test | 0.6099 | 0.6004 | 0.0095 |
| fusion | overall_accuracy | 0.7580 | 0.7760 | -0.0180 |

## Stage: sm - per member

| member | f1 | f1_baseline | f1_delta |
|---|---|---|---|
| coma_hybrid | 0.8571 | 0.8571 | 0.0000 |
| duplicate_majority | 1.0000 | 1.0000 | 0.0000 |
| embedding_sbert | 0.8148 | 0.8000 | 0.0148 |
| instance_tf_cosine | 0.6441 | 0.6441 | 0.0000 |
| label_jw | 0.8000 | 0.8000 | 0.0000 |
| llm_openai | 0.8571 | 0.8571 | 0.0000 |
| magneto_slm_llm | 0.7273 | 0.7385 | -0.0112 |

## Stage: norm - per member

| member | macro_f1 | macro_f1_baseline | macro_f1_delta |
|---|---|---|---|
| llm_canonicalize | 0.1082 | 0.0704 | 0.0378 |
| number_locale | 0.3145 | 0.3114 | 0.0031 |
| text_clean | 0.4730 | 0.3945 | 0.0785 |

## Stage: em_blocking - per member

| member | pair_recall | pair_recall_baseline | pair_recall_delta | reduction_ratio | reduction_ratio_baseline |
|---|---|---|---|---|---|
| bm25_blocker | 0.9783 | 0.9773 | 0.0009 | 0.9312 | 0.9310 |
| embedding_blocker | 0.9358 | 0.9384 | -0.0026 | 0.9312 | 0.9310 |
| sorted_neighbourhood_blocker | 0.4342 | 0.4363 | -0.0022 | 0.9742 | 0.9742 |
| standard_blocker | 0.6397 | 0.6407 | -0.0010 | 0.9517 | 0.9519 |
| token_blocker | 0.9973 | 0.9968 | 0.0005 | 0.6389 | 0.6411 |

## Stage: em_matching - per member

| member | f1 | f1_baseline | f1_delta | f1_vs_val | f1_vs_val_baseline | f1_vs_test | f1_vs_test_baseline |
|---|---|---|---|---|---|---|---|
| comem | 0.4826 | 0.4891 | -0.0066 | nan | nan | 0.4826 | 0.4891 |
| ditto_plm | 0.7365 | 0.7019 | 0.0346 | 0.7385 | 0.7385 | 0.7365 | 0.7019 |
| llm_matcher | 0.6107 | 0.6102 | 0.0005 | nan | nan | 0.6107 | 0.6102 |

## Stage: em - per pair

| pair | member | f1 | f1_baseline | f1_delta |
|---|---|---|---|---|
| products_1_products_2 | comem | 0.5562 | 0.5896 | -0.0334 |
| products_1_products_2 | ditto_plm | 0.6667 | 0.6667 | 0.0000 |
| products_1_products_2 | llm_matcher | 0.8269 | 0.8381 | -0.0112 |
| products_1_products_3 | comem | 0.4354 | 0.4354 | 0.0000 |
| products_1_products_3 | ditto_plm | 0.8182 | 0.7143 | 0.1039 |
| products_1_products_3 | llm_matcher | 0.4967 | 0.4967 | 0.0000 |
| products_1_products_4 | comem | 0.4561 | 0.4425 | 0.0137 |
| products_1_products_4 | ditto_plm | 0.7246 | 0.7246 | 0.0000 |
| products_1_products_4 | llm_matcher | 0.5085 | 0.4957 | 0.0127 |

## Stage: fusion - per member

| member | overall_accuracy | overall_accuracy_baseline | overall_accuracy_delta |
|---|---|---|---|
| brand_llm_judge | 0.5740 | 0.5540 | 0.0200 |
| brand_most_complete | 0.5800 | 0.5580 | 0.0220 |
| brand_prefer_higher_trust | 0.5700 | 0.5520 | 0.0180 |
| brand_truthfinder | 0.5780 | 0.5580 | 0.0200 |
| brand_voting | 0.5740 | 0.5540 | 0.0200 |
| description_longest_string | 0.6400 | 0.6240 | 0.0160 |
| description_ltm | 0.5400 | 0.5100 | 0.0300 |
| description_most_complete | 0.6380 | 0.6220 | 0.0160 |
| description_prefer_higher_trust | 0.5740 | 0.5540 | 0.0200 |
| description_voting | 0.5740 | 0.5540 | 0.0200 |
| priceCurrency_most_complete | 0.4880 | 0.4600 | 0.0280 |
| priceCurrency_prefer_higher_trust | 0.6040 | 0.5840 | 0.0200 |
| priceCurrency_voting | 0.5740 | 0.5540 | 0.0200 |
| price_fusionquery | 0.5080 | 0.4940 | 0.0140 |
| price_huber_m_estimator | 0.4940 | 0.4580 | 0.0360 |
| price_maximum | 0.5540 | 0.5320 | 0.0220 |
| price_median | 0.4980 | 0.4600 | 0.0380 |
| price_median_of_means | 0.4820 | 0.4460 | 0.0360 |
| price_prefer_higher_trust | 0.5800 | 0.5520 | 0.0280 |
| price_trimmed_mean | 0.4820 | 0.4460 | 0.0360 |
| title_accusim | 0.5140 | 0.5380 | -0.0240 |
| title_casefusion | 0.5740 | 0.5540 | 0.0200 |
| title_fusionquery | 0.5280 | 0.5600 | -0.0320 |
| title_llm_judge | 0.5740 | 0.5540 | 0.0200 |
| title_longest_string | 0.6500 | 0.6740 | -0.0240 |
| title_most_complete | 0.6440 | 0.6680 | -0.0240 |
| title_prefer_higher_trust | 0.5740 | 0.5540 | 0.0200 |
| title_voting | 0.5740 | 0.5540 | 0.0200 |

## Stage: fusion - per attribute

| attribute | best_accuracy | baseline | delta | spread | spread_baseline | spread_delta |
|---|---|---|---|---|---|---|
| brand | 0.7900 | 0.7900 | 0.0000 | 0.0500 | 0.0300 | 0.0200 |
| description | 0.6900 | 0.7000 | -0.0100 | 0.5000 | 0.5700 | -0.0700 |
| price | 0.5100 | 0.5500 | -0.0400 | 0.4900 | 0.5300 | -0.0400 |
| priceCurrency | 0.9600 | 1.0000 | -0.0400 | 0.5800 | 0.6200 | -0.0400 |
| title | 0.8400 | 0.8400 | 0.0000 | 0.6800 | 0.6800 | 0.0000 |
