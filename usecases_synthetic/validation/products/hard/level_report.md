# Validation report - products / hard

_Generated at 2026-06-06T02:33:22.599029+00:00_

- domain: `products`
- level: `hard`
- with_llm: `True`
- committee_versions: sm=`sm_committee.yaml@cb4a6847ac9e`, norm=`normalization_committee_products.yaml@6726f3b09a5f`, em_blocking=`em_blocking_committee_products.yaml@b8c80e7a6cca`, em_matching=`em_matching_committee_products.yaml@bb3a271dc0f4`, fusion=`fusion_committee_products.yaml@9017a21d09da`

## Stage summary

| stage | metric | measured | baseline | delta |
|---|---|---|---|---|
| sm | macro_f1 | 0.6207 | 0.6764 | -0.0557 |
| norm | macro_f1 | 0.5184 | 0.5931 | -0.0747 |
| em_blocking | macro_pair_recall_variant_model_on_regen_test | 0.8662 | 0.8500 | 0.0162 |
| em_matching | macro_f1_variant_model_on_regen_test | 0.8215 | 0.9003 | -0.0788 |
| fusion | overall_accuracy | 0.5138 | 0.6981 | -0.1843 |

## Stage: sm - per member

| member | f1 | f1_baseline | f1_delta |
|---|---|---|---|
| coma_hybrid | 0.7143 | 0.8529 | -0.1387 |
| duplicate_majority | 0.8324 | 0.9091 | -0.0767 |
| embedding_sbert | 0.5018 | 0.4498 | 0.0519 |
| instance_tf_cosine | 0.4532 | 0.4042 | 0.0490 |
| label_jw | 0.2985 | 0.5568 | -0.2583 |
| llm_openai | 0.9811 | 0.9811 | 0.0000 |
| magneto_slm_llm | 0.5633 | 0.5806 | -0.0174 |

## Stage: norm - per member

| member | macro_f1 | macro_f1_baseline | macro_f1_delta |
|---|---|---|---|
| llm_only | 0.4868 | 0.5415 | -0.0547 |
| passthrough | 0.5237 | 0.6188 | -0.0951 |
| rule_per_attribute_optimal | 0.5446 | 0.6188 | -0.0742 |

## Stage: em_blocking - per member

| member | pair_recall | pair_recall_baseline | pair_recall_delta | reduction_ratio | reduction_ratio_baseline |
|---|---|---|---|---|---|
| bm25_blocker | 0.9373 | 0.9267 | 0.0106 | 0.9115 | 0.9310 |
| embedding_blocker | 1.0000 | 1.0000 | 0.0000 | 0.9115 | 0.9310 |
| sc_block | 1.0000 | 0.9800 | 0.0200 | 0.8530 | 0.9490 |
| sorted_neighbourhood_blocker | 0.5901 | 0.5333 | 0.0568 | 0.9350 | 0.9486 |
| standard_blocker | 0.6698 | 0.6600 | 0.0098 | 0.9558 | 0.9519 |
| token_blocker | 1.0000 | 1.0000 | 0.0000 | 0.6623 | 0.6411 |

## Stage: em_matching - per member

| member | f1 | f1_baseline | f1_delta | f1_baseline_test | f1_baseline_test_baseline | f1_regen_test | f1_regen_test_baseline |
|---|---|---|---|---|---|---|---|
| comem | 0.8651 | 0.8931 | -0.0279 | 0.8696 | 0.8931 | 0.8651 | 0.8931 |
| ditto_plm | 0.9190 | 0.9339 | -0.0150 | 0.9340 | 0.9339 | 0.8625 | 0.9339 |
| llm_matcher | 0.8200 | 0.8851 | -0.0651 | 0.8244 | 0.8851 | 0.8200 | 0.8851 |
| magellan | 0.6819 | 0.8893 | -0.2074 | 0.8007 | 0.8893 | 0.5608 | 0.8893 |

## Stage: em_matching - per pair

| pair | member | f1 | f1_baseline | f1_delta |
|---|---|---|---|---|
| products_1_products_2 | comem | 0.9630 | 0.9149 | 0.0481 |
| products_1_products_2 | ditto_plm | 0.9492 | 0.9259 | 0.0232 |
| products_1_products_2 | llm_matcher | 0.8679 | 0.9375 | -0.0696 |
| products_1_products_2 | magellan | 0.6452 | 0.9074 | -0.2622 |
| products_1_products_3 | comem | 0.8254 | 0.8632 | -0.0378 |
| products_1_products_3 | ditto_plm | 0.8824 | 0.9143 | -0.0319 |
| products_1_products_3 | llm_matcher | 0.8065 | 0.8387 | -0.0323 |
| products_1_products_3 | magellan | 0.7500 | 0.9009 | -0.1509 |
| products_1_products_4 | comem | 0.8070 | 0.9011 | -0.0941 |
| products_1_products_4 | ditto_plm | 0.9254 | 0.9615 | -0.0362 |
| products_1_products_4 | llm_matcher | 0.7857 | 0.8791 | -0.0934 |
| products_1_products_4 | magellan | 0.6506 | 0.8596 | -0.2090 |

## Stage: fusion - per member

| member | overall_accuracy | overall_accuracy_baseline | overall_accuracy_delta |
|---|---|---|---|
| accusim_only | 0.4162 | 0.5127 | -0.0965 |
| casefusion_only | 0.4411 | 0.5516 | -0.1105 |
| fusionquery_only | 0.4587 | 0.5672 | -0.1085 |
| llm_only | 0.4595 | 0.5256 | -0.0661 |
| ltm_only | 0.4507 | 0.5399 | -0.0892 |
| prefer_higher_trust_only | 0.4106 | 0.6022 | -0.1916 |
| pydi_per_attribute_optimal | 0.5188 | 0.7015 | -0.1826 |
| truthfinder_only | 0.4483 | 0.5250 | -0.0767 |
| voting_only | 0.4202 | 0.5964 | -0.1762 |

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
