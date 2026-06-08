# Validation report - papers / medium

_Generated at 2026-06-08T05:34:14.518903+00:00_

- domain: `papers`
- level: `medium`
- with_llm: `True`
- committee_versions: sm=`sm_committee.yaml@cb4a6847ac9e`, norm=`normalization_committee_papers.yaml@a3dd1a8b0e71`, em_blocking=`em_blocking_committee_papers.yaml@4c879705ce30`, em_matching=`em_matching_committee_papers.yaml@3d4e6082ed0c`, fusion=`fusion_committee_papers.yaml@c898dae65e10`

## Stage summary

| stage | metric | measured | baseline | delta |
|---|---|---|---|---|
| sm | macro_f1 | 0.5824 | 0.8340 | -0.2516 |
| norm | macro_f1 | 0.5662 | 0.7608 | -0.1945 |
| em_blocking | macro_pair_recall_variant_model_on_regen_test | 0.9539 | 0.9580 | -0.0041 |
| em_matching | macro_f1_variant_model_on_regen_test | 0.9563 | 0.9664 | -0.0101 |
| fusion | overall_accuracy | 0.4766 | 0.6104 | -0.1338 |

## Stage: sm - per member

| member | f1 | f1_baseline | f1_delta |
|---|---|---|---|
| coma_hybrid | 0.3765 | 0.9714 | -0.5950 |
| duplicate_majority | 0.7143 | 0.9697 | -0.2554 |
| embedding_sbert | 0.5176 | 0.7778 | -0.2601 |
| instance_tf_cosine | 0.4576 | 0.4151 | 0.0425 |
| label_jw | 0.5138 | 0.8608 | -0.3470 |
| llm_openai | 0.9048 | 0.9714 | -0.0667 |
| magneto_slm_llm | 0.5920 | 0.8718 | -0.2798 |

## Stage: norm - per member

| member | macro_f1 | macro_f1_baseline | macro_f1_delta |
|---|---|---|---|
| llm_only | 0.5908 | 0.8161 | -0.2252 |
| passthrough | 0.5981 | 0.8257 | -0.2276 |
| rule_per_attribute_optimal | 0.5097 | 0.6405 | -0.1308 |

## Stage: em_blocking - per member

| member | pair_recall | pair_recall_baseline | pair_recall_delta | reduction_ratio | reduction_ratio_baseline |
|---|---|---|---|---|---|
| bm25_blocker | 0.9583 | 0.9583 | 0.0000 | 0.9997 | 0.9997 |
| embedding_blocker | 0.9016 | 0.9019 | -0.0003 | 0.9997 | 0.9997 |
| sc_block | 0.9895 | 0.9949 | -0.0054 | 0.9997 | 0.9997 |
| sorted_neighbourhood_blocker | 0.9661 | 0.9769 | -0.0108 | 0.9997 | 0.9997 |

## Stage: em_matching - per member

| member | f1 | f1_baseline | f1_delta | f1_baseline_test | f1_baseline_test_baseline | f1_regen_test | f1_regen_test_baseline |
|---|---|---|---|---|---|---|---|
| comem | 0.9121 | 0.9318 | -0.0197 | 0.9121 | 0.9318 | 0.9121 | 0.9318 |
| ditto_plm | 0.9982 | 0.9988 | -0.0006 | 0.9868 | 0.9988 | 0.9868 | 0.9988 |
| llm_matcher | 0.9156 | 0.9354 | -0.0197 | 0.9156 | 0.9354 | 0.9156 | 0.9354 |
| magellan | 0.9992 | 0.9995 | -0.0003 | 0.9992 | 0.9995 | 0.9992 | 0.9995 |

## Stage: em_matching - per pair

| pair | member | f1 | f1_baseline | f1_delta |
|---|---|---|---|---|
| dblp_crossref | comem | 0.9542 | 0.9773 | -0.0231 |
| dblp_crossref | ditto_plm | 0.9979 | 0.9979 | -0.0000 |
| dblp_crossref | llm_matcher | 0.9558 | 0.9810 | -0.0252 |
| dblp_crossref | magellan | 0.9994 | 0.9994 | 0.0000 |
| dblp_open_alex | comem | 0.8701 | 0.8864 | -0.0162 |
| dblp_open_alex | ditto_plm | 0.9985 | 0.9997 | -0.0012 |
| dblp_open_alex | llm_matcher | 0.8755 | 0.8897 | -0.0142 |
| dblp_open_alex | magellan | 0.9991 | 0.9997 | -0.0006 |

## Stage: fusion - per member

| member | overall_accuracy | overall_accuracy_baseline | overall_accuracy_delta |
|---|---|---|---|
| accusim_only | 0.4449 | 0.6240 | -0.1791 |
| casefusion_only | 0.3819 | 0.4833 | -0.1014 |
| fusionquery_only | 0.3848 | 0.4833 | -0.0984 |
| llm_only | 0.4833 | 0.5896 | -0.1063 |
| ltm_only | 0.4183 | 0.5364 | -0.1181 |
| prefer_higher_trust_only | 0.4823 | 0.6181 | -0.1358 |
| pydi_per_attribute_optimal | 0.4065 | 0.5404 | -0.1339 |
| truthfinder_only | 0.4360 | 0.5610 | -0.1250 |
| voting_only | 0.4665 | 0.6102 | -0.1437 |

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
