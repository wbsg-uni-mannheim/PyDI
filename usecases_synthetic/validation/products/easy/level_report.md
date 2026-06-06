# Validation report - products / easy

_Generated at 2026-06-06T02:20:35.951389+00:00_

- domain: `products`
- level: `easy`
- with_llm: `True`
- committee_versions: sm=`sm_committee.yaml@cb4a6847ac9e`, norm=`normalization_committee_products.yaml@6726f3b09a5f`, em_blocking=`em_blocking_committee_products.yaml@b8c80e7a6cca`, em_matching=`em_matching_committee_products.yaml@bb3a271dc0f4`, fusion=`fusion_committee_products.yaml@9017a21d09da`

## Stage summary

| stage | metric | measured | baseline | delta |
|---|---|---|---|---|
| sm | macro_f1 | 0.7449 | 0.6764 | 0.0685 |
| norm | macro_f1 | 0.6287 | 0.5931 | 0.0356 |
| em_blocking | macro_pair_recall_variant_model_on_regen_test | 0.8463 | 0.8500 | -0.0037 |
| em_matching | macro_f1_variant_model_on_regen_test | 0.8674 | 0.9003 | -0.0329 |
| fusion | overall_accuracy | 0.5412 | 0.6981 | -0.1569 |

## Stage: sm - per member

| member | f1 | f1_baseline | f1_delta |
|---|---|---|---|
| coma_hybrid | 0.9615 | 0.8529 | 0.1086 |
| duplicate_majority | 0.9953 | 0.9091 | 0.0863 |
| embedding_sbert | 0.4615 | 0.4498 | 0.0117 |
| instance_tf_cosine | 0.4185 | 0.4042 | 0.0143 |
| label_jw | 0.7813 | 0.5568 | 0.2244 |
| llm_openai | 0.9811 | 0.9811 | 0.0000 |
| magneto_slm_llm | 0.6149 | 0.5806 | 0.0343 |

## Stage: norm - per member

| member | macro_f1 | macro_f1_baseline | macro_f1_delta |
|---|---|---|---|
| llm_only | 0.6068 | 0.5415 | 0.0653 |
| passthrough | 0.6006 | 0.6188 | -0.0182 |
| rule_per_attribute_optimal | 0.6786 | 0.6188 | 0.0598 |

## Stage: em_blocking - per member

| member | pair_recall | pair_recall_baseline | pair_recall_delta | reduction_ratio | reduction_ratio_baseline |
|---|---|---|---|---|---|
| bm25_blocker | 0.9516 | 0.9267 | 0.0249 | 0.9343 | 0.9310 |
| embedding_blocker | 1.0000 | 1.0000 | 0.0000 | 0.9343 | 0.9310 |
| sc_block | 1.0000 | 0.9800 | 0.0200 | 0.8685 | 0.9490 |
| sorted_neighbourhood_blocker | 0.5378 | 0.5333 | 0.0045 | 0.9494 | 0.9486 |
| standard_blocker | 0.5881 | 0.6600 | -0.0719 | 0.9783 | 0.9519 |
| token_blocker | 1.0000 | 1.0000 | 0.0000 | 0.8121 | 0.6411 |

## Stage: em_matching - per member

| member | f1 | f1_baseline | f1_delta | f1_baseline_test | f1_baseline_test_baseline | f1_regen_test | f1_regen_test_baseline |
|---|---|---|---|---|---|---|---|
| comem | 0.9131 | 0.8931 | 0.0200 | 0.9296 | 0.8931 | 0.9131 | 0.8931 |
| ditto_plm | 0.9433 | 0.9339 | 0.0094 | 0.9592 | 0.9339 | 0.7579 | 0.9339 |
| llm_matcher | 0.8865 | 0.8851 | 0.0014 | 0.9030 | 0.8851 | 0.8865 | 0.8851 |
| magellan | 0.7267 | 0.8893 | -0.1626 | 0.8835 | 0.8893 | 0.4016 | 0.8893 |

## Stage: em_matching - per pair

| pair | member | f1 | f1_baseline | f1_delta |
|---|---|---|---|---|
| products_1_products_2 | comem | 0.9524 | 0.9149 | 0.0375 |
| products_1_products_2 | ditto_plm | 0.9302 | 0.9259 | 0.0043 |
| products_1_products_2 | llm_matcher | 0.9524 | 0.9375 | 0.0149 |
| products_1_products_2 | magellan | 0.8077 | 0.9074 | -0.0997 |
| products_1_products_3 | comem | 0.9744 | 0.8632 | 0.1112 |
| products_1_products_3 | ditto_plm | 0.9268 | 0.9143 | 0.0125 |
| products_1_products_3 | llm_matcher | 0.8947 | 0.8387 | 0.0560 |
| products_1_products_3 | magellan | 0.7059 | 0.9009 | -0.1950 |
| products_1_products_4 | comem | 0.8125 | 0.9011 | -0.0886 |
| products_1_products_4 | ditto_plm | 0.9730 | 0.9615 | 0.0114 |
| products_1_products_4 | llm_matcher | 0.8125 | 0.8791 | -0.0666 |
| products_1_products_4 | magellan | 0.6667 | 0.8596 | -0.1930 |

## Stage: fusion - per member

| member | overall_accuracy | overall_accuracy_baseline | overall_accuracy_delta |
|---|---|---|---|
| accusim_only | 0.4127 | 0.5127 | -0.0999 |
| casefusion_only | 0.4471 | 0.5516 | -0.1045 |
| fusionquery_only | 0.4400 | 0.5672 | -0.1272 |
| llm_only | 0.5114 | 0.5256 | -0.0143 |
| ltm_only | 0.4289 | 0.5399 | -0.1110 |
| prefer_higher_trust_only | 0.4919 | 0.6022 | -0.1103 |
| pydi_per_attribute_optimal | 0.5665 | 0.7015 | -0.1350 |
| truthfinder_only | 0.4296 | 0.5250 | -0.0954 |
| voting_only | 0.4802 | 0.5964 | -0.1162 |

## Stage: fusion - per attribute

| attribute | best_accuracy | baseline | delta | spread | spread_baseline | spread_delta |
|---|---|---|---|---|---|---|
| brand | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| bus_type | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| chipset_name | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| description | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| form_factor | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| interface_type | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| memory_type | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| model | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| model_number | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| price | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| priceCurrency | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| product_type | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| read_speed_mb_s | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| storage_connection_type | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| storage_gb | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| title | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| title_description | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| vram_gb | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| write_speed_mb_s | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
