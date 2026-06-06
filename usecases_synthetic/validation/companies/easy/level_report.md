# Validation report - companies / easy

_Generated at 2026-06-06T02:10:03.915414+00:00_

- domain: `companies`
- level: `easy`
- with_llm: `True`
- committee_versions: sm=`sm_committee.yaml@cb4a6847ac9e`, norm=`normalization_committee_companies.yaml@d6f28b520075`, em_blocking=`em_blocking_committee.yaml@042119f4d1bf`, em_matching=`em_matching_committee.yaml@f0bb40e2173f`, fusion=`fusion_committee.yaml@6f2bb9461525`

## Stage summary

| stage | metric | measured | baseline | delta |
|---|---|---|---|---|
| sm | macro_f1 | 0.8620 | 0.7270 | 0.1349 |
| norm | macro_f1 | 0.6089 | 0.8706 | -0.2617 |
| em_blocking | macro_pair_recall_variant_model_on_regen_test | 0.9829 | 0.9876 | -0.0047 |
| em_matching | macro_f1_variant_model_on_regen_test | 0.9066 | 0.8842 | 0.0224 |
| fusion | overall_accuracy | 0.4003 | 0.4577 | -0.0574 |

## Stage: sm - per member

| member | f1 | f1_baseline | f1_delta |
|---|---|---|---|
| coma_hybrid | 1.0000 | 0.6452 | 0.3548 |
| duplicate_majority | 0.8000 | 0.8000 | 0.0000 |
| embedding_sbert | 0.7778 | 0.7179 | 0.0598 |
| instance_tf_cosine | 0.5625 | 0.6286 | -0.0661 |
| label_jw | 1.0000 | 0.3846 | 0.6154 |
| llm_openai | 1.0000 | 1.0000 | 0.0000 |
| magneto_slm_llm | 0.8936 | 0.9130 | -0.0194 |

## Stage: norm - per member

| member | macro_f1 | macro_f1_baseline | macro_f1_delta |
|---|---|---|---|
| llm_only | 0.5806 | 0.8340 | -0.2534 |
| passthrough | 0.6619 | 0.9388 | -0.2770 |
| rule_per_attribute_optimal | 0.5841 | 0.8389 | -0.2548 |

## Stage: em_blocking - per member

| member | pair_recall | pair_recall_baseline | pair_recall_delta | reduction_ratio | reduction_ratio_baseline |
|---|---|---|---|---|---|
| bm25_blocker | 0.9921 | 1.0000 | -0.0079 | 0.9861 | 0.9846 |
| embedding_blocker | 1.0000 | 1.0000 | 0.0000 | 0.9861 | 0.9846 |
| sc_block | 0.9956 | 1.0000 | -0.0044 | 0.9722 | 0.9691 |
| sorted_neighbourhood_blocker | 0.9655 | 0.9735 | -0.0079 | 0.9879 | 0.9867 |
| standard_blocker | 0.9655 | 0.9735 | -0.0079 | 0.9976 | 0.9985 |
| token_blocker | 0.9788 | 0.9788 | 0.0000 | 0.9924 | 0.9930 |

## Stage: em_matching - per member

| member | f1 | f1_baseline | f1_delta | f1_baseline_test | f1_baseline_test_baseline | f1_regen_test | f1_regen_test_baseline |
|---|---|---|---|---|---|---|---|
| comem | 0.8972 | 0.8950 | 0.0022 | 0.8972 | 0.8950 | 0.8972 | 0.8950 |
| ditto_plm | 0.9092 | 0.8699 | 0.0393 | 0.8747 | 0.8699 | 0.8747 | 0.8699 |
| llm_matcher | 0.8911 | 0.8848 | 0.0064 | 0.8911 | 0.8848 | 0.8911 | 0.8848 |
| magellan | 0.9288 | 0.8872 | 0.0416 | 0.9344 | 0.8872 | 0.9344 | 0.8872 |

## Stage: em_matching - per pair

| pair | member | f1 | f1_baseline | f1_delta |
|---|---|---|---|---|
| forbes_dbpedia | comem | 0.8649 | 0.8649 | 0.0000 |
| forbes_dbpedia | ditto_plm | 0.9197 | 0.8750 | 0.0447 |
| forbes_dbpedia | llm_matcher | 0.8649 | 0.8673 | -0.0024 |
| forbes_dbpedia | magellan | 0.9767 | 0.9104 | 0.0663 |
| forbes_fullcontact | comem | 0.9296 | 0.9252 | 0.0043 |
| forbes_fullcontact | ditto_plm | 0.8988 | 0.8649 | 0.0339 |
| forbes_fullcontact | llm_matcher | 0.9174 | 0.9023 | 0.0151 |
| forbes_fullcontact | magellan | 0.8810 | 0.8640 | 0.0170 |

## Stage: fusion - per member

| member | overall_accuracy | overall_accuracy_baseline | overall_accuracy_delta |
|---|---|---|---|
| accusim_only | 0.3763 | 0.4197 | -0.0434 |
| casefusion_only | 0.3372 | 0.4009 | -0.0637 |
| fusionquery_only | 0.3314 | 0.3560 | -0.0246 |
| llm_only | 0.3546 | 0.3994 | -0.0449 |
| ltm_only | 0.3589 | 0.4009 | -0.0420 |
| prefer_higher_trust_only | 0.3965 | 0.4616 | -0.0651 |
| pydi_per_attribute_optimal | 0.4038 | 0.4501 | -0.0463 |
| truthfinder_only | 0.3372 | 0.3589 | -0.0217 |
| voting_only | 0.3560 | 0.4153 | -0.0593 |

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
