# Validation report - products / easy

_Generated at 2026-05-30T22:42:32.841896+00:00_

- domain: `products`
- level: `easy`
- with_llm: `True`
- committee_versions: sm=`sm_committee.yaml@cb4a6847ac9e`, norm=`normalization_committee_products.yaml@ba3a7ab9d0fd`, em_blocking=`em_blocking_committee_products.yaml@5485fd074f41`, em_matching=`em_matching_committee_products.yaml@457e751943e8`, fusion=`fusion_committee_products.yaml@70c645adfccc`

## Stage summary

| stage | metric | measured | baseline | delta |
|---|---|---|---|---|
| sm | macro_f1 | 0.6591 | 0.6537 | 0.0054 |
| norm | macro_f1 | 0.4669 | 0.4524 | 0.0145 |
| em_blocking | macro_pair_recall_variant_model_on_regen_test | 0.0000 | 0.0000 | 0.0000 |
| em_matching | macro_f1_variant_model_on_regen_test | 0.9444 | 0.9507 | -0.0063 |
| fusion | overall_accuracy | 0.7020 | 0.8220 | -0.1200 |

## Stage: sm - per member

| member | f1 | f1_baseline | f1_delta |
|---|---|---|---|
| coma_hybrid | 0.8511 | 0.8511 | 0.0000 |
| duplicate_majority | 1.0000 | 1.0000 | 0.0000 |
| embedding_sbert | 0.3619 | 0.3465 | 0.0154 |
| instance_tf_cosine | 0.2838 | 0.2800 | 0.0038 |
| label_jw | 0.6780 | 0.6780 | 0.0000 |
| llm_openai | 0.8511 | 0.8511 | 0.0000 |
| magneto_slm_llm | 0.5882 | 0.5694 | 0.0188 |

## Stage: norm - per member

| member | macro_f1 | macro_f1_baseline | macro_f1_delta |
|---|---|---|---|
| llm_only | 0.4129 | 0.3996 | 0.0133 |
| passthrough | 0.4939 | 0.4788 | 0.0151 |
| rule_per_attribute_optimal | 0.4939 | 0.4788 | 0.0151 |

## Stage: em_blocking - per member

| member | pair_recall | pair_recall_baseline | pair_recall_delta | reduction_ratio | reduction_ratio_baseline |
|---|---|---|---|---|---|
| bm25_blocker | 0.9457 | 0.9020 | 0.0438 | 0.9343 | 0.9310 |
| embedding_blocker | 1.0000 | 0.9915 | 0.0085 | 0.9343 | 0.9310 |
| sc_block | 1.0000 | 1.0000 | 0.0000 | 0.9159 | 0.8733 |
| sorted_neighbourhood_blocker | 0.7159 | 0.5794 | 0.1365 | 0.9493 | 0.9486 |
| standard_blocker | 0.7003 | 0.6986 | 0.0017 | 0.9782 | 0.9519 |
| token_blocker | 1.0000 | 0.9971 | 0.0029 | 0.8108 | 0.6411 |

## Stage: em_matching - per member

| member | f1 | f1_baseline | f1_delta | f1_baseline_test | f1_baseline_test_baseline | f1_regen_test | f1_regen_test_baseline |
|---|---|---|---|---|---|---|---|
| comem | 0.9363 | 0.9293 | 0.0071 | 0.9363 | 0.9293 | 0.9363 | 0.9293 |
| ditto_plm | 0.9235 | 0.9786 | -0.0551 | 0.9895 | 0.9786 | 0.9895 | 0.9786 |
| llm_matcher | 0.9359 | 0.9263 | 0.0096 | 0.9359 | 0.9263 | 0.9359 | 0.9263 |
| magellan | 0.9819 | 0.9688 | 0.0131 | 0.9785 | 0.9688 | 0.9785 | 0.9688 |

## Stage: em_matching - per pair

| pair | member | f1 | f1_baseline | f1_delta |
|---|---|---|---|---|
| products_1_products_2 | comem | 0.9592 | 0.9115 | 0.0477 |
| products_1_products_2 | ditto_plm | 0.8913 | 0.9750 | -0.0837 |
| products_1_products_2 | llm_matcher | 0.9485 | 0.9351 | 0.0134 |
| products_1_products_2 | magellan | 0.9703 | 0.9752 | -0.0049 |
| products_1_products_3 | comem | 0.9070 | 0.9302 | -0.0233 |
| products_1_products_3 | ditto_plm | 0.9070 | 0.9782 | -0.0712 |
| products_1_products_3 | llm_matcher | 0.9318 | 0.9302 | 0.0016 |
| products_1_products_3 | magellan | 0.9892 | 0.9427 | 0.0465 |
| products_1_products_4 | comem | 0.9429 | 0.9461 | -0.0033 |
| products_1_products_4 | ditto_plm | 0.9722 | 0.9827 | -0.0104 |
| products_1_products_4 | llm_matcher | 0.9275 | 0.9136 | 0.0140 |
| products_1_products_4 | magellan | 0.9863 | 0.9885 | -0.0022 |

## Stage: fusion - per member

| member | overall_accuracy | overall_accuracy_baseline | overall_accuracy_delta |
|---|---|---|---|
| accusim_only | 0.4140 | 0.4000 | 0.0140 |
| casefusion_only | 0.4760 | 0.4980 | -0.0220 |
| fusionquery_only | 0.5680 | 0.5760 | -0.0080 |
| llm_only | 0.3820 | 0.4460 | -0.0640 |
| ltm_only | 0.4520 | 0.4720 | -0.0200 |
| prefer_higher_trust_only | 0.5640 | 0.6260 | -0.0620 |
| pydi_per_attribute_optimal | 0.7020 | 0.8220 | -0.1200 |
| truthfinder_only | 0.4300 | 0.4300 | 0.0000 |
| voting_only | 0.5580 | 0.5960 | -0.0380 |

## Stage: fusion - per attribute

| attribute | best_accuracy | baseline | delta | spread | spread_baseline | spread_delta |
|---|---|---|---|---|---|---|
| brand | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| description | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| price | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| priceCurrency | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| title | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
