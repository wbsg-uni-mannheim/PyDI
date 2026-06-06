# Baseline report - companies

_Generated at 2026-06-06T02:07:00.369430+00:00_

## Stage: sm

### SM - aggregated
| metric | value |
|---|---|
| best_member_f1 | 1.0000 |
| best_member_name | llm_openai |
| macro_f1 | 0.7270 |
| macro_precision | 0.9148 |
| macro_recall | 0.6531 |
| max_f1 | 1.0000 |
| min_f1 | 0.3846 |

### SM - per attribute
| attribute | any_correct | coma_hybrid | duplicate_majority | embedding_sbert | instance_tf_cosine | label_jw | llm_openai | magneto_slm_llm |
|---|---|---|---|---|---|---|---|---|
| dbpedia.annual_income | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| dbpedia.established | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| dbpedia.headquarters | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| dbpedia.id | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| dbpedia.keypeople_name | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| dbpedia.nation | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| dbpedia.org_name | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| dbpedia.sector | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| dbpedia.total_assets_val | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| forbes.asset_value | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| forbes.business_segment | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| forbes.company | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| forbes.id | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| forbes.region | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| forbes.sales_figure | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| fullcontact.Attribute_2 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| fullcontact.Attribute_3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| fullcontact.Attribute_4 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| fullcontact.Attribute_5 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| fullcontact.Attribute_6 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| fullcontact.id | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |

## Stage: norm

### NORM - aggregated
| metric | value |
|---|---|
| best_member_f1 | 0.9388 |
| best_member_name_f1 | 1.0000 |
| macro_f1 | 0.8706 |
| macro_precision | 0.9048 |
| macro_recall | 0.8458 |
| max_f1 | 0.9388 |
| min_f1 | 0.8340 |

### NORM - per attribute
| attribute | any_correct | best_member_f1 | llm_only | passthrough | rule_per_attribute_optimal |
|---|---|---|---|---|---|
| assets | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| city | 1.0000 | 0.9041 | 0.9041 | 0.9041 | 0.9041 |
| country | 1.0000 | 0.9685 | 0.9685 | 0.9685 | 0.9685 |
| founded | 1.0000 | 0.6992 | 0.0000 | 0.6992 | 0.0000 |
| id | 1.0000 | 1.0000 | 0.9685 | 1.0000 | 1.0000 |
| name | 1.0000 | 1.0000 | 0.9972 | 1.0000 | 1.0000 |
| revenue | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Stage: em_blocking

### EM_BLOCKING - aggregated
| metric | value |
|---|---|
| best_member_name | standard_blocker |
| best_member_pair_recall | 0.9735 |
| best_member_reduction_ratio | 0.9985 |
| macro_pair_recall | 0.9876 |
| macro_pair_recall_baseline_model_on_baseline_test | 0.9876 |
| macro_pair_recall_baseline_model_on_regen_test | 0.9876 |
| macro_pair_recall_variant_model_on_baseline_test | 0.9876 |
| macro_pair_recall_variant_model_on_regen_test | 0.9876 |
| macro_reduction_ratio | 0.9861 |
| max_pair_recall | 1.0000 |
| min_pair_recall | 0.9735 |
| recall_floor | 0.9700 |

## Stage: em_matching

### EM_MATCHING - aggregated
| metric | value |
|---|---|
| best_member_f1 | 0.8950 |
| best_member_name | comem |
| macro_f1 | 0.8842 |
| macro_f1_baseline_model_on_baseline_test | 0.8842 |
| macro_f1_baseline_model_on_regen_test | 0.8842 |
| macro_f1_baseline_test | 0.8842 |
| macro_f1_regen_test | 0.8842 |
| macro_f1_variant_model_on_baseline_test | 0.8842 |
| macro_f1_variant_model_on_regen_test | 0.8842 |
| macro_precision | 0.8879 |
| macro_recall | 0.8987 |
| max_f1 | 0.8950 |
| min_f1 | 0.8699 |

## Stage: fusion

### FUSION - aggregated
| metric | value |
|---|---|
| best_member_macro_accuracy | 0.4577 |
| macro_accuracy | 0.4037 |
| max_accuracy | 0.4577 |
| min_accuracy | 0.3534 |
| overall_accuracy | 0.4577 |

### FUSION - per attribute
| attribute | accusim_only | best_member_accuracy | casefusion_only | fusionquery_only | llm_only | ltm_only | mean_member_accuracy | prefer_higher_trust_only | pydi_per_attribute_optimal | truthfinder_only | voting_only |
|---|---|---|---|---|---|---|---|---|---|---|---|
| assets | 0.1900 | 0.1900 | 0.1500 | 0.1500 | 0.1400 | 0.1500 | 0.1578 | 0.1500 | 0.1500 | 0.1500 | 0.1900 |
| city | 0.4100 | 0.5000 | 0.4100 | 0.4500 | 0.4800 | 0.4600 | 0.4478 | 0.4700 | 0.5000 | 0.4400 | 0.4100 |
| country | 0.5800 | 0.8400 | 0.6800 | 0.6000 | 0.6600 | 0.6200 | 0.6544 | 0.8400 | 0.6900 | 0.5300 | 0.6900 |
| founded | 0.5700 | 0.5900 | 0.5900 | 0.5400 | 0.5800 | 0.5500 | 0.5722 | 0.5500 | 0.5900 | 0.5900 | 0.5900 |
| keypeople | 0.1429 | 0.1538 | 0.1538 | 0.1538 | 0.1429 | 0.1429 | 0.1490 | 0.1538 | 0.1538 | 0.1538 | 0.1429 |
| name | 0.8500 | 0.8600 | 0.6200 | 0.4000 | 0.6300 | 0.6800 | 0.6711 | 0.8600 | 0.8600 | 0.4500 | 0.6900 |
| revenue | 0.1700 | 0.1800 | 0.1800 | 0.1800 | 0.1400 | 0.1800 | 0.1733 | 0.1800 | 0.1800 | 0.1800 | 0.1700 |
