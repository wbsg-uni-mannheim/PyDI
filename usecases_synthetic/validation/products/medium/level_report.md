# Validation report - products / medium

_Generated at 2026-06-05T05:43:33.008628+00:00_

- domain: `products`
- level: `medium`
- with_llm: `True`
- committee_versions: sm=`sm_committee.yaml@cb4a6847ac9e`, norm=`normalization_committee_products.yaml@6726f3b09a5f`, em_blocking=`em_blocking_committee_products.yaml@b8c80e7a6cca`, em_matching=`em_matching_committee_products.yaml@bb3a271dc0f4`, fusion=`fusion_committee_products.yaml@9017a21d09da`

## Stage summary

| stage | metric | measured | baseline | delta |
|---|---|---|---|---|
| sm | macro_f1 | 0.6864 | 0.6764 | 0.0100 |
| norm | macro_f1 | 0.5310 | 0.5931 | -0.0620 |
| em_blocking | macro_pair_recall_variant_model_on_regen_test | 0.0000 | 0.0000 | 0.0000 |
| em_matching | macro_f1_variant_model_on_regen_test | 0.8221 | 0.9003 | -0.0782 |
| fusion | overall_accuracy | 0.6112 | 0.6981 | -0.0869 |

## Stage: sm - per member

| member | f1 | f1_baseline | f1_delta |
|---|---|---|---|
| coma_hybrid | 0.8020 | 0.8529 | -0.0509 |
| duplicate_majority | 0.8511 | 0.9091 | -0.0580 |
| embedding_sbert | 0.5152 | 0.4498 | 0.0653 |
| instance_tf_cosine | 0.4492 | 0.4042 | 0.0450 |
| label_jw | 0.6032 | 0.5568 | 0.0464 |
| llm_openai | 0.9811 | 0.9811 | 0.0000 |
| magneto_slm_llm | 0.6032 | 0.5806 | 0.0225 |

## Stage: norm - per member

| member | macro_f1 | macro_f1_baseline | macro_f1_delta |
|---|---|---|---|
| llm_only | 0.4957 | 0.5415 | -0.0458 |
| passthrough | 0.5366 | 0.6188 | -0.0822 |
| rule_per_attribute_optimal | 0.5608 | 0.6188 | -0.0580 |

## Stage: em_blocking - per member

| member | pair_recall | pair_recall_baseline | pair_recall_delta | reduction_ratio | reduction_ratio_baseline |
|---|---|---|---|---|---|
| bm25_blocker | 0.9412 | 0.9267 | 0.0145 | 0.9343 | 0.9310 |
| embedding_blocker | 1.0000 | 1.0000 | 0.0000 | 0.9343 | 0.9310 |
| sc_block | 1.0000 | 0.9800 | 0.0200 | 0.9430 | 0.9490 |
| sorted_neighbourhood_blocker | 0.6239 | 0.5333 | 0.0906 | 0.9495 | 0.9486 |
| standard_blocker | 0.6858 | 0.6600 | 0.0258 | 0.9775 | 0.9519 |
| token_blocker | 1.0000 | 1.0000 | 0.0000 | 0.7990 | 0.6411 |

## Stage: em_matching - per member

| member | f1 | f1_baseline | f1_delta | f1_baseline_test | f1_baseline_test_baseline | f1_regen_test | f1_regen_test_baseline |
|---|---|---|---|---|---|---|---|
| comem | 0.8861 | 0.8931 | -0.0070 | 0.9010 | 0.8931 | 0.8861 | 0.8931 |
| ditto_plm | 0.9494 | 0.9339 | 0.0155 | 0.9790 | 0.9339 | 0.7279 | 0.9339 |
| llm_matcher | 0.8275 | 0.8851 | -0.0576 | 0.8606 | 0.8851 | 0.8275 | 0.8851 |
| magellan | 0.6255 | 0.8893 | -0.2638 | 0.8578 | 0.8893 | 0.3774 | 0.8893 |

## Stage: em_matching - per pair

| pair | member | f1 | f1_baseline | f1_delta |
|---|---|---|---|---|
| products_1_products_2 | comem | 0.9362 | 0.9149 | 0.0213 |
| products_1_products_2 | ditto_plm | 0.9615 | 0.9259 | 0.0356 |
| products_1_products_2 | llm_matcher | 0.9020 | 0.9375 | -0.0355 |
| products_1_products_2 | magellan | 0.6301 | 0.9074 | -0.2773 |
| products_1_products_3 | comem | 0.8571 | 0.8632 | -0.0060 |
| products_1_products_3 | ditto_plm | 0.9565 | 0.9143 | 0.0422 |
| products_1_products_3 | llm_matcher | 0.7805 | 0.8387 | -0.0582 |
| products_1_products_3 | magellan | 0.6667 | 0.9009 | -0.2342 |
| products_1_products_4 | comem | 0.8649 | 0.9011 | -0.0362 |
| products_1_products_4 | ditto_plm | 0.9302 | 0.9615 | -0.0313 |
| products_1_products_4 | llm_matcher | 0.8000 | 0.8791 | -0.0791 |
| products_1_products_4 | magellan | 0.5797 | 0.8596 | -0.2799 |

## Stage: fusion - per member

| member | overall_accuracy | overall_accuracy_baseline | overall_accuracy_delta |
|---|---|---|---|
| accusim_only | 0.4283 | 0.5127 | -0.0844 |
| casefusion_only | 0.4951 | 0.5516 | -0.0565 |
| fusionquery_only | 0.4997 | 0.5672 | -0.0675 |
| llm_only | 0.4860 | 0.5256 | -0.0396 |
| ltm_only | 0.4867 | 0.5399 | -0.0532 |
| prefer_higher_trust_only | 0.4802 | 0.6022 | -0.1220 |
| pydi_per_attribute_optimal | 0.6009 | 0.7015 | -0.1006 |
| truthfinder_only | 0.4835 | 0.5250 | -0.0415 |
| voting_only | 0.4770 | 0.5964 | -0.1194 |

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
