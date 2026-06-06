# Validation report - companies / medium

_Generated at 2026-06-06T02:13:12.256072+00:00_

- domain: `companies`
- level: `medium`
- with_llm: `True`
- committee_versions: sm=`sm_committee.yaml@cb4a6847ac9e`, norm=`normalization_committee_companies.yaml@d6f28b520075`, em_blocking=`em_blocking_committee.yaml@042119f4d1bf`, em_matching=`em_matching_committee.yaml@f0bb40e2173f`, fusion=`fusion_committee.yaml@6f2bb9461525`

## Stage summary

| stage | metric | measured | baseline | delta |
|---|---|---|---|---|
| sm | macro_f1 | 0.7702 | 0.7270 | 0.0431 |
| norm | macro_f1 | 0.7185 | 0.8706 | -0.1521 |
| em_blocking | macro_pair_recall_variant_model_on_regen_test | 0.9822 | 0.9876 | -0.0054 |
| em_matching | macro_f1_variant_model_on_regen_test | 0.8721 | 0.8842 | -0.0121 |
| fusion | overall_accuracy | 0.4130 | 0.4577 | -0.0447 |

## Stage: sm - per member

| member | f1 | f1_baseline | f1_delta |
|---|---|---|---|
| coma_hybrid | 0.6875 | 0.6452 | 0.0423 |
| duplicate_majority | 0.8000 | 0.8000 | 0.0000 |
| embedding_sbert | 0.7222 | 0.7179 | 0.0043 |
| instance_tf_cosine | 0.5625 | 0.6286 | -0.0661 |
| label_jw | 0.7059 | 0.3846 | 0.3213 |
| llm_openai | 1.0000 | 1.0000 | 0.0000 |
| magneto_slm_llm | 0.9130 | 0.9130 | 0.0000 |

## Stage: norm - per member

| member | macro_f1 | macro_f1_baseline | macro_f1_delta |
|---|---|---|---|
| llm_only | 0.7353 | 0.8340 | -0.0987 |
| passthrough | 0.7152 | 0.9388 | -0.2236 |
| rule_per_attribute_optimal | 0.7050 | 0.8389 | -0.1339 |

## Stage: em_blocking - per member

| member | pair_recall | pair_recall_baseline | pair_recall_delta | reduction_ratio | reduction_ratio_baseline |
|---|---|---|---|---|---|
| bm25_blocker | 0.9921 | 1.0000 | -0.0079 | 0.9853 | 0.9846 |
| embedding_blocker | 1.0000 | 1.0000 | 0.0000 | 0.9853 | 0.9846 |
| sc_block | 1.0000 | 1.0000 | 0.0000 | 0.9707 | 0.9691 |
| sorted_neighbourhood_blocker | 0.9611 | 0.9735 | -0.0124 | 0.9875 | 0.9867 |
| standard_blocker | 0.9611 | 0.9735 | -0.0124 | 0.9983 | 0.9985 |
| token_blocker | 0.9788 | 0.9788 | 0.0000 | 0.9931 | 0.9930 |

## Stage: em_matching - per member

| member | f1 | f1_baseline | f1_delta | f1_baseline_test | f1_baseline_test_baseline | f1_regen_test | f1_regen_test_baseline |
|---|---|---|---|---|---|---|---|
| comem | 0.8711 | 0.8950 | -0.0240 | 0.8711 | 0.8950 | 0.8711 | 0.8950 |
| ditto_plm | 0.8725 | 0.8699 | 0.0026 | 0.8716 | 0.8699 | 0.8716 | 0.8699 |
| llm_matcher | 0.8640 | 0.8848 | -0.0208 | 0.8640 | 0.8848 | 0.8640 | 0.8848 |
| magellan | 0.8808 | 0.8872 | -0.0064 | 0.8823 | 0.8872 | 0.8823 | 0.8872 |

## Stage: em_matching - per pair

| pair | member | f1 | f1_baseline | f1_delta |
|---|---|---|---|---|
| forbes_dbpedia | comem | 0.8440 | 0.8649 | -0.0208 |
| forbes_dbpedia | ditto_plm | 0.8824 | 0.8750 | 0.0074 |
| forbes_dbpedia | llm_matcher | 0.8113 | 0.8673 | -0.0559 |
| forbes_dbpedia | magellan | 0.9023 | 0.9104 | -0.0082 |
| forbes_fullcontact | comem | 0.8981 | 0.9252 | -0.0271 |
| forbes_fullcontact | ditto_plm | 0.8627 | 0.8649 | -0.0021 |
| forbes_fullcontact | llm_matcher | 0.9167 | 0.9023 | 0.0143 |
| forbes_fullcontact | magellan | 0.8594 | 0.8640 | -0.0046 |

## Stage: fusion - per member

| member | overall_accuracy | overall_accuracy_baseline | overall_accuracy_delta |
|---|---|---|---|
| accusim_only | 0.3936 | 0.4197 | -0.0260 |
| casefusion_only | 0.3603 | 0.4009 | -0.0405 |
| fusionquery_only | 0.3690 | 0.3560 | 0.0130 |
| llm_only | 0.3821 | 0.3994 | -0.0174 |
| ltm_only | 0.4124 | 0.4009 | 0.0116 |
| prefer_higher_trust_only | 0.3965 | 0.4616 | -0.0651 |
| pydi_per_attribute_optimal | 0.4168 | 0.4501 | -0.0333 |
| truthfinder_only | 0.3661 | 0.3589 | 0.0072 |
| voting_only | 0.3719 | 0.4153 | -0.0434 |

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
