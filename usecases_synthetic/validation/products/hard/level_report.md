# Validation report - products / hard

_Generated at 2026-05-30T23:26:45.454703+00:00_

- domain: `products`
- level: `hard`
- with_llm: `True`
- committee_versions: sm=`sm_committee.yaml@cb4a6847ac9e`, norm=`normalization_committee_products.yaml@ba3a7ab9d0fd`, em_blocking=`em_blocking_committee_products.yaml@5485fd074f41`, em_matching=`em_matching_committee_products.yaml@457e751943e8`, fusion=`fusion_committee_products.yaml@70c645adfccc`

## Stage summary

| stage | metric | measured | baseline | delta |
|---|---|---|---|---|
| sm | macro_f1 | 0.5088 | 0.6537 | -0.1449 |
| norm | macro_f1 | 0.4007 | 0.4524 | -0.0517 |
| em_blocking | macro_pair_recall_variant_model_on_regen_test | 0.0000 | 0.0000 | 0.0000 |
| em_matching | macro_f1_variant_model_on_regen_test | 0.8959 | 0.9507 | -0.0549 |
| fusion | overall_accuracy | 0.5275 | 0.8220 | -0.2945 |

## Stage: sm - per member

| member | f1 | f1_baseline | f1_delta |
|---|---|---|---|
| coma_hybrid | 0.4564 | 0.8511 | -0.3947 |
| duplicate_majority | 0.9189 | 1.0000 | -0.0811 |
| embedding_sbert | 0.3540 | 0.3465 | 0.0075 |
| instance_tf_cosine | 0.2815 | 0.2800 | 0.0015 |
| label_jw | 0.1961 | 0.6780 | -0.4819 |
| llm_openai | 0.8511 | 0.8511 | 0.0000 |
| magneto_slm_llm | 0.5035 | 0.5694 | -0.0658 |

## Stage: norm - per member

| member | macro_f1 | macro_f1_baseline | macro_f1_delta |
|---|---|---|---|
| llm_only | 0.3568 | 0.3996 | -0.0428 |
| passthrough | 0.4219 | 0.4788 | -0.0569 |
| rule_per_attribute_optimal | 0.4234 | 0.4788 | -0.0554 |

## Stage: em_blocking - per member

| member | pair_recall | pair_recall_baseline | pair_recall_delta | reduction_ratio | reduction_ratio_baseline |
|---|---|---|---|---|---|
| bm25_blocker | 0.9234 | 0.9020 | 0.0215 | 0.9115 | 0.9310 |
| embedding_blocker | 0.9949 | 0.9915 | 0.0034 | 0.9115 | 0.9310 |
| sc_block | 0.9948 | 1.0000 | -0.0052 | 0.9579 | 0.8733 |
| sorted_neighbourhood_blocker | 0.5815 | 0.5794 | 0.0021 | 0.9353 | 0.9486 |
| standard_blocker | 0.6513 | 0.6986 | -0.0473 | 0.9560 | 0.9519 |
| token_blocker | 0.9949 | 0.9971 | -0.0022 | 0.6639 | 0.6411 |

## Stage: em_matching - per member

| member | f1 | f1_baseline | f1_delta | f1_baseline_test | f1_baseline_test_baseline | f1_regen_test | f1_regen_test_baseline |
|---|---|---|---|---|---|---|---|
| comem | 0.9147 | 0.9293 | -0.0146 | 0.9147 | 0.9293 | 0.9147 | 0.9293 |
| ditto_plm | 0.9448 | 0.9786 | -0.0338 | 0.9796 | 0.9786 | 0.9625 | 0.9786 |
| llm_matcher | 0.8947 | 0.9263 | -0.0316 | 0.8947 | 0.9263 | 0.8947 | 0.9263 |
| magellan | 0.8293 | 0.9688 | -0.1395 | 0.9437 | 0.9688 | 0.8791 | 0.9688 |

## Stage: em_matching - per pair

| pair | member | f1 | f1_baseline | f1_delta |
|---|---|---|---|---|
| products_1_products_2 | comem | 0.8696 | 0.9115 | -0.0419 |
| products_1_products_2 | ditto_plm | 0.9516 | 0.9750 | -0.0234 |
| products_1_products_2 | llm_matcher | 0.8596 | 0.9351 | -0.0754 |
| products_1_products_2 | magellan | 0.7568 | 0.9752 | -0.2184 |
| products_1_products_3 | comem | 0.9153 | 0.9302 | -0.0150 |
| products_1_products_3 | ditto_plm | 0.9440 | 0.9782 | -0.0342 |
| products_1_products_3 | llm_matcher | 0.8870 | 0.9302 | -0.0433 |
| products_1_products_3 | magellan | 0.8644 | 0.9427 | -0.0783 |
| products_1_products_4 | comem | 0.9592 | 0.9461 | 0.0131 |
| products_1_products_4 | ditto_plm | 0.9388 | 0.9827 | -0.0439 |
| products_1_products_4 | llm_matcher | 0.9375 | 0.9136 | 0.0239 |
| products_1_products_4 | magellan | 0.8667 | 0.9885 | -0.1218 |

## Stage: fusion - per member

| member | overall_accuracy | overall_accuracy_baseline | overall_accuracy_delta |
|---|---|---|---|
| accusim_only | 0.3675 | 0.4000 | -0.0325 |
| casefusion_only | 0.3825 | 0.4980 | -0.1155 |
| fusionquery_only | 0.4125 | 0.5760 | -0.1635 |
| llm_only | 0.3675 | 0.4460 | -0.0785 |
| ltm_only | 0.4375 | 0.4720 | -0.0345 |
| prefer_higher_trust_only | 0.3800 | 0.6260 | -0.2460 |
| pydi_per_attribute_optimal | 0.5275 | 0.8220 | -0.2945 |
| truthfinder_only | 0.4100 | 0.4300 | -0.0200 |
| voting_only | 0.3950 | 0.5960 | -0.2010 |

## Stage: fusion - per attribute

| attribute | best_accuracy | baseline | delta | spread | spread_baseline | spread_delta |
|---|---|---|---|---|---|---|
| brand | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| description | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| price | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| priceCurrency | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| title | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
