# Baseline report - products-small

_Generated at 2026-05-15T14:34:49.673226+00:00_

## Stage: sm

### SM - aggregated
| metric | value |
|---|---|
| macro_f1 | 0.8138 |
| macro_precision | 0.7149 |
| macro_recall | 0.9583 |
| max_f1 | 1.0000 |
| min_f1 | 0.6441 |

### SM - per attribute
| attribute | any_correct | coma_hybrid | duplicate_majority | embedding_sbert | instance_tf_cosine | label_jw | llm_openai | magneto_slm_llm |
|---|---|---|---|---|---|---|---|---|
| products_1.brand | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| products_1.description | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| products_1.id | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| products_1.price | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| products_1.priceCurrency | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| products_1.title | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |
| products_2.brand | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| products_2.description | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| products_2.id | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| products_2.price | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| products_2.priceCurrency | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| products_2.title | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |
| products_3.brand | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| products_3.description | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| products_3.id | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| products_3.price | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| products_3.priceCurrency | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| products_3.title | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |
| products_4.brand | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| products_4.description | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| products_4.id | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| products_4.price | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| products_4.priceCurrency | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| products_4.title | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |

## Stage: fusion

### FUSION - aggregated
| metric | value |
|---|---|
| overall_accuracy | 0.7760 |
| overall_mean_accuracy | 0.5107 |
| overall_spread | 0.4860 |

### FUSION - per attribute
| attribute | accusim | best_strategy_accuracy | casefusion | fusionquery | huber_m_estimator | llm_judge | longest_string | ltm | maximum | mean_strategy_accuracy | median | median_of_means | most_complete | prefer_higher_trust | spread | trimmed_mean | truthfinder | voting |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| brand |  | 0.7900 |  |  |  | 0.7700 |  |  |  | 0.7760 |  |  | 0.7900 | 0.7600 | 0.0300 |  | 0.7900 | 0.7700 |
| description |  | 0.7000 |  |  |  |  | 0.7000 | 0.1300 |  | 0.4440 |  |  | 0.6900 | 0.3500 | 0.5700 |  |  | 0.3500 |
| price |  | 0.5500 |  | 0.2600 | 0.0800 |  |  |  | 0.4500 | 0.2100 | 0.0900 | 0.0200 |  | 0.5500 | 0.5300 | 0.0200 |  |  |
| priceCurrency |  | 1.0000 |  |  |  |  |  |  |  | 0.7433 |  |  | 0.3800 | 1.0000 | 0.6200 |  |  | 0.8500 |
| title | 0.1600 | 0.8400 | 0.2400 | 0.2700 |  | 0.2400 | 0.8400 |  |  | 0.3800 |  |  | 0.8100 | 0.2400 | 0.6800 |  |  | 0.2400 |
