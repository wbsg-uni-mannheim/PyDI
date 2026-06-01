# Validation report - companies / hard

_Generated at 2026-04-28T20:03:55.311897+00:00_

- domain: `companies`
- level: `hard`
- with_llm: `False`
- committee_versions: sm=`sm_committee.yaml@5022dec6c8d2`, em=`em_blocking_committee.yaml@44fd6bfd276d+em_matching_committee.yaml@47093abf698a`, fusion=`fusion_committee.yaml@0a388fd41a72`

## Stage summary

| stage | metric | measured | baseline | delta |
|---|---|---|---|---|
| sm | macro_f1 | 0.1944 | 0.3756 | -0.1811 |
| em | macro_f1 | 0.8393 | 0.7203 | 0.1190 |
| fusion | overall_accuracy | 0.3542 | 0.4792 | -0.1250 |

## Stage: sm - per member

| member | f1 | f1_baseline | f1_delta |
|---|---|---|---|
| coma_hybrid | 0.0800 | 0.2400 | -0.1600 |
| duplicate_majority | 0.0000 | 0.0000 | 0.0000 |
| embedding_sbert | 0.0000 | 0.3077 | -0.3077 |
| llm_openai | 0.6977 | 0.9545 | -0.2569 |

## Stage: em - per member

| member | f1 | f1_baseline | f1_delta | pool_precision | pool_precision_baseline | pool_recall | pool_recall_baseline |
|---|---|---|---|---|---|---|---|
| ditto_plm | 0.8393 | 0.7203 | 0.1190 | 0.7291 | 0.7825 | 0.2013 | 0.2106 |

## Stage: em - per pair

| pair | member | f1 | f1_baseline | f1_delta |
|---|---|---|---|---|
| forbes_dbpedia | ditto_plm | 0.8817 | 0.6464 | 0.2353 |
| forbes_fullcontact | ditto_plm | 0.7969 | 0.7942 | 0.0027 |

## Stage: fusion - per member

| member | overall_accuracy | overall_accuracy_baseline | overall_accuracy_delta |
|---|---|---|---|
| assets_fusionquery | 0.3981 | 0.5926 | -0.1944 |
| assets_huber_m_estimator | 0.3981 | 0.5833 | -0.1852 |
| assets_maximum | 0.3981 | 0.5926 | -0.1944 |
| assets_median | 0.3981 | 0.5833 | -0.1852 |
| assets_median_of_means | 0.3981 | 0.5833 | -0.1852 |
| assets_prefer_higher_trust | 0.3981 | 0.5833 | -0.1852 |
| assets_trimmed_mean | 0.3981 | 0.5833 | -0.1852 |
| city_llm_judge | 0.3981 | 0.5926 | -0.1944 |
| city_prefer_higher_trust | 0.4352 | 0.6204 | -0.1852 |
| city_shortest_string | 0.4352 | 0.6389 | -0.2037 |
| city_truthfinder | 0.4259 | 0.0000 | 0.4259 |
| city_voting | 0.3981 | 0.5926 | -0.1944 |
| country_favour_forbes | 0.3981 | 0.5926 | -0.1944 |
| country_llm_judge | 0.3981 | 0.5926 | -0.1944 |
| country_prefer_higher_trust | 0.3704 | 0.4352 | -0.0648 |
| country_truthfinder | 0.4352 | 0.5556 | -0.1204 |
| country_voting | 0.3981 | 0.5926 | -0.1944 |
| founded_earliest | 0.3981 | 0.5926 | -0.1944 |
| founded_prefer_higher_trust | 0.3519 | 0.5185 | -0.1667 |
| founded_voting | 0.3981 | 0.5926 | -0.1944 |
| industry_accusim | 0.3981 | 0.5926 | -0.1944 |
| industry_llm_judge | 0.3981 | 0.5926 | -0.1944 |
| industry_most_complete | 0.3981 | 0.5926 | -0.1944 |
| industry_prefer_higher_trust | 0.3981 | 0.5926 | -0.1944 |
| industry_voting | 0.3981 | 0.5926 | -0.1944 |
| keypeople_ltm | 0.3981 | 0.5926 | -0.1944 |
| keypeople_prefer_higher_trust | 0.3981 | 0.5926 | -0.1944 |
| keypeople_union | 0.3981 | 0.5926 | -0.1944 |
| keypeople_voting | 0.3981 | 0.5926 | -0.1944 |
| name_accusim | 0.3981 | 0.5926 | -0.1944 |
| name_casefusion | 0.3796 | 0.5741 | -0.1944 |
| name_fusionquery | 0.3981 | 0.5648 | -0.1667 |
| name_llm_judge | 0.3981 | 0.5926 | -0.1944 |
| name_longest_string | 0.3796 | 0.5648 | -0.1852 |
| name_most_complete | 0.3796 | 0.5648 | -0.1852 |
| name_prefer_higher_trust | 0.3981 | 0.5648 | -0.1667 |
| name_voting | 0.3981 | 0.5926 | -0.1944 |
| revenue_fusionquery | 0.3981 | 0.5926 | -0.1944 |
| revenue_huber_m_estimator | 0.3981 | 0.5833 | -0.1852 |
| revenue_maximum | 0.3981 | 0.5926 | -0.1944 |
| revenue_median | 0.3981 | 0.5833 | -0.1852 |
| revenue_median_of_means | 0.3981 | 0.5833 | -0.1852 |
| revenue_prefer_higher_trust | 0.3981 | 0.5833 | -0.1852 |
| revenue_trimmed_mean | 0.3981 | 0.5833 | -0.1852 |

## Stage: fusion - per attribute

| attribute | best_accuracy | baseline | delta | spread | spread_baseline | spread_delta |
|---|---|---|---|---|---|---|
| assets | 0.0000 | 0.0556 | -0.0556 | 0.0000 | 0.0556 | -0.0556 |
| city | 0.6111 | 0.8333 | -0.2222 | 0.2222 | 0.8333 | -0.6111 |
| country | 0.7222 | 0.9444 | -0.2222 | 0.3889 | 0.9444 | -0.5556 |
| founded | 0.6667 | 0.8889 | -0.2222 | 0.2778 | 0.4444 | -0.1667 |
| industry | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| keypeople | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| name | 0.7778 | 1.0000 | -0.2222 | 0.1111 | 0.1667 | -0.0556 |
| revenue | 0.0556 | 0.1111 | -0.0556 | 0.0000 | 0.0556 | -0.0556 |
