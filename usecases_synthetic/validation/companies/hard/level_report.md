# Validation report - companies / hard

_Generated at 2026-06-05T04:31:14.864929+00:00_

- domain: `companies`
- level: `hard`
- with_llm: `True`
- committee_versions: sm=`sm_committee.yaml@cb4a6847ac9e`, norm=`normalization_committee_companies.yaml@d6f28b520075`, em_blocking=`em_blocking_committee.yaml@042119f4d1bf`, em_matching=`em_matching_committee.yaml@f0bb40e2173f`, fusion=`fusion_committee.yaml@6f2bb9461525`

## Stage summary

| stage | metric | measured | baseline | delta |
|---|---|---|---|---|
| sm | macro_f1 | 0.6459 | 0.7270 | -0.0811 |
| norm | macro_f1 | 0.7252 | 0.8706 | -0.1454 |
| em_blocking | macro_pair_recall_variant_model_on_regen_test | 0.0000 | 0.0000 | 0.0000 |
| em_matching | macro_f1_variant_model_on_regen_test | 0.8737 | 0.8842 | -0.0106 |
| fusion | overall_accuracy | 0.3484 | 0.4577 | -0.1093 |

## Stage: sm - per member

| member | f1 | f1_baseline | f1_delta |
|---|---|---|---|
| coma_hybrid | 0.3200 | 0.6452 | -0.3252 |
| duplicate_majority | 0.8000 | 0.8000 | 0.0000 |
| embedding_sbert | 0.6061 | 0.7179 | -0.1119 |
| instance_tf_cosine | 0.5625 | 0.6286 | -0.0661 |
| label_jw | 0.3200 | 0.3846 | -0.0646 |
| llm_openai | 1.0000 | 1.0000 | 0.0000 |
| magneto_slm_llm | 0.9130 | 0.9130 | 0.0000 |

## Stage: norm - per member

| member | macro_f1 | macro_f1_baseline | macro_f1_delta |
|---|---|---|---|
| llm_only | 0.7519 | 0.8340 | -0.0821 |
| passthrough | 0.6899 | 0.9388 | -0.2490 |
| rule_per_attribute_optimal | 0.7340 | 0.8389 | -0.1050 |

## Stage: em_blocking - per member

| member | pair_recall | pair_recall_baseline | pair_recall_delta | reduction_ratio | reduction_ratio_baseline |
|---|---|---|---|---|---|
| bm25_blocker | 0.9876 | 1.0000 | -0.0124 | 0.9850 | 0.9846 |
| embedding_blocker | 1.0000 | 1.0000 | 0.0000 | 0.9850 | 0.9846 |
| sc_block | 0.9921 | 1.0000 | -0.0079 | 0.9701 | 0.9691 |
| sorted_neighbourhood_blocker | 0.9744 | 0.9735 | 0.0009 | 0.9872 | 0.9867 |
| standard_blocker | 0.9699 | 0.9735 | -0.0035 | 0.9977 | 0.9985 |
| token_blocker | 0.9788 | 0.9788 | 0.0000 | 0.9932 | 0.9930 |

## Stage: em_matching - per member

| member | f1 | f1_baseline | f1_delta | f1_baseline_test | f1_baseline_test_baseline | f1_regen_test | f1_regen_test_baseline |
|---|---|---|---|---|---|---|---|
| comem | 0.8601 | 0.8950 | -0.0350 | 0.8814 | 0.8950 | 0.8601 | 0.8950 |
| ditto_plm | 0.8784 | 0.8699 | 0.0084 | 0.8601 | 0.8699 | 0.9221 | 0.8699 |
| llm_matcher | 0.8517 | 0.8848 | -0.0331 | 0.8652 | 0.8848 | 0.8517 | 0.8848 |
| magellan | 0.9045 | 0.8872 | 0.0173 | 0.8650 | 0.8872 | 0.9133 | 0.8872 |

## Stage: em_matching - per pair

| pair | member | f1 | f1_baseline | f1_delta |
|---|---|---|---|---|
| forbes_dbpedia | comem | 0.8000 | 0.8649 | -0.0649 |
| forbes_dbpedia | ditto_plm | 0.8571 | 0.8750 | -0.0179 |
| forbes_dbpedia | llm_matcher | 0.8113 | 0.8673 | -0.0559 |
| forbes_dbpedia | magellan | 0.9037 | 0.9104 | -0.0067 |
| forbes_fullcontact | comem | 0.9202 | 0.9252 | -0.0050 |
| forbes_fullcontact | ditto_plm | 0.8996 | 0.8649 | 0.0347 |
| forbes_fullcontact | llm_matcher | 0.8920 | 0.9023 | -0.0103 |
| forbes_fullcontact | magellan | 0.9053 | 0.8640 | 0.0413 |

## Stage: fusion - per member

| member | overall_accuracy | overall_accuracy_baseline | overall_accuracy_delta |
|---|---|---|---|
| accusim_only | 0.3092 | 0.4197 | -0.1104 |
| casefusion_only | 0.3076 | 0.4009 | -0.0933 |
| fusionquery_only | 0.3277 | 0.3560 | -0.0283 |
| llm_only | 0.3395 | 0.3994 | -0.0599 |
| ltm_only | 0.3513 | 0.4009 | -0.0496 |
| prefer_higher_trust_only | 0.3496 | 0.4616 | -0.1121 |
| pydi_per_attribute_optimal | 0.3210 | 0.4501 | -0.1291 |
| truthfinder_only | 0.3193 | 0.3589 | -0.0396 |
| voting_only | 0.3193 | 0.4153 | -0.0960 |

## Stage: fusion - per attribute

| attribute | best_accuracy | baseline | delta | spread | spread_baseline | spread_delta |
|---|---|---|---|---|---|---|
| assets | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| city | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| country | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| founded | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| keypeople | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| name | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| revenue | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
