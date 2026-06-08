# Validation report - papers / easy

_Generated at 2026-06-08T04:21:34.160406+00:00_

- domain: `papers`
- level: `easy`
- with_llm: `True`
- committee_versions: sm=`sm_committee.yaml@cb4a6847ac9e`, norm=`normalization_committee_papers.yaml@a3dd1a8b0e71`, em_blocking=`em_blocking_committee_papers.yaml@4c879705ce30`, em_matching=`em_matching_committee_papers.yaml@3d4e6082ed0c`, fusion=`fusion_committee_papers.yaml@c898dae65e10`

## Stage summary

| stage | metric | measured | baseline | delta |
|---|---|---|---|---|
| sm | macro_f1 | 0.7588 | 0.8340 | -0.0751 |
| norm | macro_f1 | 0.6020 | 0.7608 | -0.1587 |
| em_blocking | macro_pair_recall_variant_model_on_regen_test | 0.9575 | 0.9580 | -0.0005 |
| em_matching | macro_f1_variant_model_on_regen_test | 0.9661 | 0.9664 | -0.0003 |
| fusion | overall_accuracy | 0.5826 | 0.6104 | -0.0278 |

## Stage: sm - per member

| member | f1 | f1_baseline | f1_delta |
|---|---|---|---|
| coma_hybrid | 0.9286 | 0.9714 | -0.0429 |
| duplicate_majority | 0.8608 | 0.9697 | -0.1089 |
| embedding_sbert | 0.6078 | 0.7778 | -0.1699 |
| instance_tf_cosine | 0.3937 | 0.4151 | -0.0214 |
| label_jw | 0.8125 | 0.8608 | -0.0483 |
| llm_openai | 0.9286 | 0.9714 | -0.0429 |
| magneto_slm_llm | 0.7800 | 0.8718 | -0.0918 |

## Stage: norm - per member

| member | macro_f1 | macro_f1_baseline | macro_f1_delta |
|---|---|---|---|
| llm_only | 0.6401 | 0.8161 | -0.1759 |
| passthrough | 0.6466 | 0.8257 | -0.1791 |
| rule_per_attribute_optimal | 0.5194 | 0.6405 | -0.1211 |

## Stage: em_blocking - per member

| member | pair_recall | pair_recall_baseline | pair_recall_delta | reduction_ratio | reduction_ratio_baseline |
|---|---|---|---|---|---|
| bm25_blocker | 0.9579 | 0.9583 | -0.0004 | 0.9997 | 0.9997 |
| embedding_blocker | 0.9018 | 0.9019 | -0.0000 | 0.9997 | 0.9997 |
| sc_block | 0.9933 | 0.9949 | -0.0016 | 0.9997 | 0.9997 |
| sorted_neighbourhood_blocker | 0.9770 | 0.9769 | 0.0001 | 0.9997 | 0.9997 |

## Stage: em_matching - per member

| member | f1 | f1_baseline | f1_delta | f1_baseline_test | f1_baseline_test_baseline | f1_regen_test | f1_regen_test_baseline |
|---|---|---|---|---|---|---|---|
| comem | 0.9349 | 0.9318 | 0.0030 | 0.9349 | 0.9318 | 0.9349 | 0.9318 |
| ditto_plm | 0.9980 | 0.9988 | -0.0008 | 0.9934 | 0.9988 | 0.9925 | 0.9988 |
| llm_matcher | 0.9330 | 0.9354 | -0.0024 | 0.9330 | 0.9354 | 0.9330 | 0.9354 |
| magellan | 0.9986 | 0.9995 | -0.0009 | 0.9994 | 0.9995 | 0.9971 | 0.9995 |

## Stage: em_matching - per pair

| pair | member | f1 | f1_baseline | f1_delta |
|---|---|---|---|---|
| dblp_crossref | comem | 0.9793 | 0.9773 | 0.0020 |
| dblp_crossref | ditto_plm | 0.9976 | 0.9979 | -0.0003 |
| dblp_crossref | llm_matcher | 0.9771 | 0.9810 | -0.0040 |
| dblp_crossref | magellan | 0.9979 | 0.9994 | -0.0015 |
| dblp_open_alex | comem | 0.8905 | 0.8864 | 0.0041 |
| dblp_open_alex | ditto_plm | 0.9985 | 0.9997 | -0.0012 |
| dblp_open_alex | llm_matcher | 0.8890 | 0.8897 | -0.0007 |
| dblp_open_alex | magellan | 0.9994 | 0.9997 | -0.0003 |

## Stage: fusion - per member

| member | overall_accuracy | overall_accuracy_baseline | overall_accuracy_delta |
|---|---|---|---|
| accusim_only | 0.5541 | 0.6240 | -0.0699 |
| casefusion_only | 0.4902 | 0.4833 | 0.0069 |
| fusionquery_only | 0.4902 | 0.4833 | 0.0069 |
| llm_only | 0.5787 | 0.5896 | -0.0108 |
| ltm_only | 0.5207 | 0.5364 | -0.0157 |
| prefer_higher_trust_only | 0.5965 | 0.6181 | -0.0217 |
| pydi_per_attribute_optimal | 0.5148 | 0.5404 | -0.0256 |
| truthfinder_only | 0.5404 | 0.5610 | -0.0207 |
| voting_only | 0.5827 | 0.6102 | -0.0276 |

## Stage: fusion - per attribute

| attribute | best_accuracy | baseline | delta | spread | spread_baseline | spread_delta |
|---|---|---|---|---|---|---|
| authors | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| first_page | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| issue | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| journal | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| keywords | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| last_page | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| publication_year | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| referenced_works_count | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| title | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| type | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| volume | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
