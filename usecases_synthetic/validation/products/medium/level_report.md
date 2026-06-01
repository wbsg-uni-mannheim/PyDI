# Validation report - products / medium

_Generated at 2026-05-30T23:00:43.137356+00:00_

- domain: `products`
- level: `medium`
- with_llm: `True`
- committee_versions: sm=`sm_committee.yaml@cb4a6847ac9e`, norm=`normalization_committee_products.yaml@ba3a7ab9d0fd`, em_blocking=`em_blocking_committee_products.yaml@5485fd074f41`, em_matching=`em_matching_committee_products.yaml@457e751943e8`, fusion=`fusion_committee_products.yaml@70c645adfccc`

## Stage summary

| stage | metric | measured | baseline | delta |
|---|---|---|---|---|
| sm | macro_f1 | 0.5827 | 0.6537 | -0.0711 |
| norm | macro_f1 | 0.4139 | 0.4524 | -0.0385 |
| em_blocking | macro_pair_recall_variant_model_on_regen_test | 0.0000 | 0.0000 | 0.0000 |
| em_matching | macro_f1_variant_model_on_regen_test | 0.9337 | 0.9507 | -0.0170 |
| fusion | overall_accuracy | 0.6620 | 0.8220 | -0.1600 |

## Stage: sm - per member

| member | f1 | f1_baseline | f1_delta |
|---|---|---|---|
| coma_hybrid | 0.6512 | 0.8511 | -0.1999 |
| duplicate_majority | 0.9404 | 1.0000 | -0.0596 |
| embedding_sbert | 0.3486 | 0.3465 | 0.0022 |
| instance_tf_cosine | 0.2920 | 0.2800 | 0.0120 |
| label_jw | 0.4615 | 0.6780 | -0.2164 |
| llm_openai | 0.8511 | 0.8511 | 0.0000 |
| magneto_slm_llm | 0.5338 | 0.5694 | -0.0356 |

## Stage: norm - per member

| member | macro_f1 | macro_f1_baseline | macro_f1_delta |
|---|---|---|---|
| llm_only | 0.3611 | 0.3996 | -0.0384 |
| passthrough | 0.4402 | 0.4788 | -0.0386 |
| rule_per_attribute_optimal | 0.4402 | 0.4788 | -0.0386 |

## Stage: em_blocking - per member

| member | pair_recall | pair_recall_baseline | pair_recall_delta | reduction_ratio | reduction_ratio_baseline |
|---|---|---|---|---|---|
| bm25_blocker | 0.9571 | 0.9020 | 0.0551 | 0.9343 | 0.9310 |
| embedding_blocker | 1.0000 | 0.9915 | 0.0085 | 0.9343 | 0.9310 |
| sc_block | 1.0000 | 1.0000 | 0.0000 | 0.8705 | 0.8733 |
| sorted_neighbourhood_blocker | 0.7347 | 0.5794 | 0.1554 | 0.9495 | 0.9486 |
| standard_blocker | 0.7510 | 0.6986 | 0.0524 | 0.9775 | 0.9519 |
| token_blocker | 1.0000 | 0.9971 | 0.0029 | 0.7997 | 0.6411 |

## Stage: em_matching - per member

| member | f1 | f1_baseline | f1_delta | f1_baseline_test | f1_baseline_test_baseline | f1_regen_test | f1_regen_test_baseline |
|---|---|---|---|---|---|---|---|
| comem | 0.8885 | 0.9293 | -0.0408 | 0.8885 | 0.9293 | 0.8885 | 0.9293 |
| ditto_plm | 0.9909 | 0.9786 | 0.0123 | 0.9959 | 0.9786 | 0.9959 | 0.9786 |
| llm_matcher | 0.8790 | 0.9263 | -0.0473 | 0.8790 | 0.9263 | 0.8790 | 0.9263 |
| magellan | 0.9767 | 0.9688 | 0.0078 | 0.9840 | 0.9688 | 0.9840 | 0.9688 |

## Stage: em_matching - per pair

| pair | member | f1 | f1_baseline | f1_delta |
|---|---|---|---|---|
| products_1_products_2 | comem | 0.9020 | 0.9115 | -0.0095 |
| products_1_products_2 | ditto_plm | 0.9821 | 0.9750 | 0.0071 |
| products_1_products_2 | llm_matcher | 0.8687 | 0.9351 | -0.0664 |
| products_1_products_2 | magellan | 0.9550 | 0.9752 | -0.0203 |
| products_1_products_3 | comem | 0.9184 | 0.9302 | -0.0119 |
| products_1_products_3 | ditto_plm | 0.9905 | 0.9782 | 0.0123 |
| products_1_products_3 | llm_matcher | 0.9072 | 0.9302 | -0.0230 |
| products_1_products_3 | magellan | 1.0000 | 0.9427 | 0.0573 |
| products_1_products_4 | comem | 0.8451 | 0.9461 | -0.1010 |
| products_1_products_4 | ditto_plm | 1.0000 | 0.9827 | 0.0173 |
| products_1_products_4 | llm_matcher | 0.8611 | 0.9136 | -0.0525 |
| products_1_products_4 | magellan | 0.9750 | 0.9885 | -0.0135 |

## Stage: fusion - per member

| member | overall_accuracy | overall_accuracy_baseline | overall_accuracy_delta |
|---|---|---|---|
| accusim_only | 0.3580 | 0.4000 | -0.0420 |
| casefusion_only | 0.4120 | 0.4980 | -0.0860 |
| fusionquery_only | 0.4560 | 0.5760 | -0.1200 |
| llm_only | 0.3680 | 0.4460 | -0.0780 |
| ltm_only | 0.4340 | 0.4720 | -0.0380 |
| prefer_higher_trust_only | 0.4620 | 0.6260 | -0.1640 |
| pydi_per_attribute_optimal | 0.6620 | 0.8220 | -0.1600 |
| truthfinder_only | 0.3880 | 0.4300 | -0.0420 |
| voting_only | 0.4640 | 0.5960 | -0.1320 |

## Stage: fusion - per attribute

| attribute | best_accuracy | baseline | delta | spread | spread_baseline | spread_delta |
|---|---|---|---|---|---|---|
| brand | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| description | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| price | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| priceCurrency | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| title | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
