# Baseline report - companies-small

_Generated at 2026-05-15T10:31:58.130216+00:00_

## Stage: sm

### SM - aggregated
| metric | value |
|---|---|
| macro_f1 | 0.6730 |
| macro_precision | 0.8825 |
| macro_recall | 0.6054 |
| max_f1 | 0.8837 |
| min_f1 | 0.3200 |

### SM - per attribute
| attribute | any_correct | coma_hybrid | duplicate_majority | embedding_sbert | instance_tf_cosine | label_jw | llm_openai | magneto_slm_llm |
|---|---|---|---|---|---|---|---|---|
| dbpedia.annual_income | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| dbpedia.established | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| dbpedia.headquarters | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| dbpedia.id | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| dbpedia.keypeople_name | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| dbpedia.nation | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| dbpedia.org_name | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| dbpedia.sector | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| dbpedia.total_assets_val | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| forbes.asset_value | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| forbes.business_segment | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| forbes.company | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| forbes.id | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| forbes.region | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| forbes.sales_figure | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| fullcontact.Attribute_2 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| fullcontact.Attribute_3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| fullcontact.Attribute_4 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| fullcontact.Attribute_5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| fullcontact.Attribute_6 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| fullcontact.id | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |

## Stage: fusion

### FUSION - aggregated
| metric | value |
|---|---|
| overall_accuracy | 0.8810 |
| overall_mean_accuracy | 0.7205 |
| overall_spread | 0.2857 |

### FUSION - per attribute
| attribute | accusim | best_strategy_accuracy | casefusion | earliest | favour_forbes | fusionquery | huber_m_estimator | intersection | intersection_k_sources | llm_judge | longest_string | ltm | maximum | mean_strategy_accuracy | median | median_of_means | most_complete | prefer_higher_trust | shortest_string | spread | trimmed_mean | truthfinder | union | voting |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| assets |  | 0.9444 |  |  |  | 0.9444 | 0.6111 |  |  |  |  |  | 0.7222 | 0.6825 | 0.6111 | 0.6111 |  | 0.6667 |  | 0.3333 | 0.6111 |  |  |  |
| city |  | 0.8333 |  |  |  |  |  |  |  | 0.6111 |  |  |  | 0.6889 |  |  |  | 0.7222 | 0.8333 | 0.2778 |  | 0.7222 |  | 0.5556 |
| country |  | 1.0000 |  |  | 0.9444 |  |  |  |  | 0.6111 |  |  |  | 0.8222 |  |  |  | 0.6111 |  | 0.3889 |  | 1.0000 |  | 0.9444 |
| founded |  | 0.9444 |  | 0.9444 |  |  |  |  |  |  |  |  |  | 0.9444 |  |  |  | 0.9444 |  | 0.0000 |  |  |  | 0.9444 |
| keypeople |  | 0.6667 |  |  |  |  |  | 0.2222 | 0.2222 |  |  | 0.5556 |  | 0.4815 |  |  |  | 0.6667 |  | 0.4444 |  |  | 0.6667 | 0.5556 |
| name | 1.0000 | 1.0000 | 0.8333 |  |  | 0.7778 |  |  |  | 0.8333 | 0.8333 |  |  | 0.8681 |  |  | 0.8333 | 0.8333 |  | 0.2222 |  |  |  | 1.0000 |
| revenue |  | 0.7778 |  |  |  | 0.7778 | 0.4444 |  |  |  |  |  | 0.7222 | 0.5556 | 0.4444 | 0.4444 |  | 0.6111 |  | 0.3333 | 0.4444 |  |  |  |
