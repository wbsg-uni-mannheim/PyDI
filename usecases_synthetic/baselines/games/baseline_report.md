# Baseline report - games

_Generated at 2026-06-05T03:12:23.203585+00:00_

## Stage: sm

### SM - aggregated
| metric | value |
|---|---|
| best_member_f1 | 1.0000 |
| best_member_name | llm_openai |
| macro_f1 | 0.7478 |
| macro_precision | 0.8501 |
| macro_recall | 0.6923 |
| max_f1 | 1.0000 |
| min_f1 | 0.3429 |

### SM - per attribute
| attribute | any_correct | coma_hybrid | duplicate_majority | embedding_sbert | instance_tf_cosine | label_jw | llm_openai | magneto_slm_llm |
|---|---|---|---|---|---|---|---|---|
| dbpedia.franchise | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| dbpedia.genre | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| dbpedia.id | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| dbpedia.launch_yr | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| dbpedia.studio | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| dbpedia.system | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| dbpedia.title | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| metacritic.age_rating | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| metacritic.console | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| metacritic.game_title | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| metacritic.genres | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| metacritic.id | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| metacritic.made_by | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| metacritic.player_rating | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| metacritic.press_rating | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| metacritic.year_published | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| sales.age_classification | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| sales.comm_rating | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| sales.dist | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| sales.genre | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| sales.hw | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| sales.id | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| sales.launch_dt | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| sales.press_score | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| sales.prod_title | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| sales.studio | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |

## Stage: norm

### NORM - aggregated
| metric | value |
|---|---|
| best_member_f1 | 0.9809 |
| best_member_name_f1 | 1.0000 |
| macro_f1 | 0.8971 |
| macro_precision | 0.9167 |
| macro_recall | 0.8821 |
| max_f1 | 0.9809 |
| min_f1 | 0.8547 |

### NORM - per attribute
| attribute | any_correct | best_member_f1 | llm_only | passthrough | rule_per_attribute_optimal |
|---|---|---|---|---|---|
| criticScore | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| developer | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| id | 1.0000 | 1.0000 | 0.9903 | 1.0000 | 1.0000 |
| name | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| platform | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| releaseYear | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| series | 1.0000 | 0.8571 | 0.8571 | 0.8571 | 0.8571 |
| userScore | 1.0000 | 0.9899 | 0.9899 | 0.9899 | 0.9899 |

## Stage: em_blocking

### EM_BLOCKING - aggregated
| metric | value |
|---|---|
| best_member_name | standard_blocker |
| best_member_pair_recall | 0.9858 |
| best_member_reduction_ratio | 0.9988 |
| macro_pair_recall | 0.9560 |
| macro_pair_recall_baseline_model_on_baseline_test | 0.0000 |
| macro_pair_recall_baseline_model_on_regen_test | 0.0000 |
| macro_pair_recall_variant_model_on_baseline_test | 0.0000 |
| macro_pair_recall_variant_model_on_regen_test | 0.0000 |
| macro_reduction_ratio | 0.9899 |
| max_pair_recall | 1.0000 |
| min_pair_recall | 0.9155 |
| recall_floor | 0.9700 |

## Stage: em_matching

### EM_MATCHING - aggregated
| metric | value |
|---|---|
| best_member_f1 | 0.7155 |
| best_member_name | ditto_plm |
| macro_f1 | 0.6093 |
| macro_f1_baseline_model_on_baseline_test | 0.6093 |
| macro_f1_baseline_model_on_regen_test | 0.6093 |
| macro_f1_baseline_test | 0.6093 |
| macro_f1_regen_test | 0.6093 |
| macro_f1_variant_model_on_baseline_test | 0.6093 |
| macro_f1_variant_model_on_regen_test | 0.6093 |
| macro_precision | 0.8668 |
| macro_recall | 0.4883 |
| max_f1 | 0.7155 |
| min_f1 | 0.5092 |

## Stage: fusion

### FUSION - aggregated
| metric | value |
|---|---|
| best_member_macro_accuracy | 0.7202 |
| macro_accuracy | 0.6821 |
| max_accuracy | 0.7202 |
| min_accuracy | 0.6136 |
| overall_accuracy | 0.7202 |

### FUSION - per attribute
| attribute | accusim_only | best_member_accuracy | casefusion_only | fusionquery_only | llm_only | ltm_only | mean_member_accuracy | prefer_higher_trust_only | pydi_per_attribute_optimal | truthfinder_only | voting_only |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ESRB | 0.9800 | 0.9800 | 0.9800 | 0.9800 | 0.9800 | 0.9800 | 0.9800 | 0.9800 | 0.9800 | 0.9800 | 0.9800 |
| criticScore | 0.2800 | 0.2800 | 0.2800 | 0.2800 | 0.2800 | 0.2800 | 0.2800 | 0.2800 | 0.2800 | 0.2800 | 0.2800 |
| developer | 0.8800 | 0.8800 | 0.7300 | 0.8800 | 0.7700 | 0.8500 | 0.8289 | 0.8700 | 0.7400 | 0.8700 | 0.8700 |
| genres | 0.1700 | 0.8200 | 0.8200 | 0.8200 | 0.7500 | 0.5600 | 0.6311 | 0.7400 | 0.8200 | 0.8200 | 0.1800 |
| name | 0.9100 | 0.9700 | 0.8900 | 0.9600 | 0.9600 | 0.9200 | 0.9300 | 0.9700 | 0.9000 | 0.9600 | 0.9000 |
| platform | 0.9600 | 0.9800 | 0.9700 | 0.9600 | 0.6500 | 0.9800 | 0.9311 | 0.9700 | 0.9600 | 0.9700 | 0.9600 |
| publisher | 0.5100 | 0.5100 | 0.5100 | 0.5100 | 0.5100 | 0.5100 | 0.5100 | 0.5100 | 0.5100 | 0.5100 | 0.5100 |
| releaseYear | 0.9200 | 0.9300 | 0.9000 | 0.8900 | 0.4200 | 0.8300 | 0.8400 | 0.9300 | 0.8900 | 0.8900 | 0.8900 |
| userScore | 0.2222 | 0.2222 | 0.2020 | 0.2020 | 0.2020 | 0.2020 | 0.2076 | 0.2121 | 0.2020 | 0.2020 | 0.2222 |
