# Validation report - companies / medium

_Generated at 2026-04-28T19:55:25.464600+00:00_

- domain: `companies`
- level: `medium`
- with_llm: `False`
- committee_versions: sm=`sm_committee.yaml@5022dec6c8d2`, em=`em_blocking_committee.yaml@44fd6bfd276d+em_matching_committee.yaml@47093abf698a`, fusion=`fusion_committee.yaml@0a388fd41a72`

## Stage summary

| stage | metric | measured | baseline | delta |
|---|---|---|---|---|
| sm | macro_f1 | 0.2430 | 0.3756 | -0.1326 |
| em | macro_f1 | 0.8246 | 0.7203 | 0.1044 |
| fusion | overall_accuracy | 0.3333 | 0.4792 | -0.1458 |

## Stage: sm - per member

| member | f1 | f1_baseline | f1_delta |
|---|---|---|---|
| coma_hybrid | 0.1538 | 0.2400 | -0.0862 |
| duplicate_majority | 0.0000 | 0.0000 | 0.0000 |
| embedding_sbert | 0.0909 | 0.3077 | -0.2168 |
| llm_openai | 0.7273 | 0.9545 | -0.2273 |

## Stage: em - per member

| member | f1 | f1_baseline | f1_delta | pool_precision | pool_precision_baseline | pool_recall | pool_recall_baseline |
|---|---|---|---|---|---|---|---|
| ditto_plm | 0.8246 | 0.7203 | 0.1044 | 0.7618 | 0.7825 | 0.2054 | 0.2106 |

## Stage: em - per pair

| pair | member | f1 | f1_baseline | f1_delta |
|---|---|---|---|---|
| forbes_dbpedia | ditto_plm | 0.8249 | 0.6464 | 0.1785 |
| forbes_fullcontact | ditto_plm | 0.8244 | 0.7942 | 0.0303 |

## Stage: fusion - per member

| member | overall_accuracy | overall_accuracy_baseline | overall_accuracy_delta |
|---|---|---|---|
| assets_fusionquery | 0.3796 | 0.5926 | -0.2130 |
| assets_huber_m_estimator | 0.3796 | 0.5833 | -0.2037 |
| assets_maximum | 0.3796 | 0.5926 | -0.2130 |
| assets_median | 0.3796 | 0.5833 | -0.2037 |
| assets_median_of_means | 0.3796 | 0.5833 | -0.2037 |
| assets_prefer_higher_trust | 0.3796 | 0.5833 | -0.2037 |
| assets_trimmed_mean | 0.3796 | 0.5833 | -0.2037 |
| city_llm_judge | 0.3796 | 0.5926 | -0.2130 |
| city_prefer_higher_trust | 0.4074 | 0.6204 | -0.2130 |
| city_shortest_string | 0.4167 | 0.6389 | -0.2222 |
| city_truthfinder | 0.3889 | 0.0000 | 0.3889 |
| city_voting | 0.3796 | 0.5926 | -0.2130 |
| country_favour_forbes | 0.3704 | 0.5926 | -0.2222 |
| country_llm_judge | 0.3796 | 0.5926 | -0.2130 |
| country_prefer_higher_trust | 0.2963 | 0.4352 | -0.1389 |
| country_truthfinder | 0.3796 | 0.5556 | -0.1759 |
| country_voting | 0.3796 | 0.5926 | -0.2130 |
| founded_earliest | 0.3796 | 0.5926 | -0.2130 |
| founded_prefer_higher_trust | 0.3426 | 0.5185 | -0.1759 |
| founded_voting | 0.3796 | 0.5926 | -0.2130 |
| industry_accusim | 0.3796 | 0.5926 | -0.2130 |
| industry_llm_judge | 0.3796 | 0.5926 | -0.2130 |
| industry_most_complete | 0.3796 | 0.5926 | -0.2130 |
| industry_prefer_higher_trust | 0.3796 | 0.5926 | -0.2130 |
| industry_voting | 0.3796 | 0.5926 | -0.2130 |
| keypeople_ltm | 0.3796 | 0.5926 | -0.2130 |
| keypeople_prefer_higher_trust | 0.3796 | 0.5926 | -0.2130 |
| keypeople_union | 0.3796 | 0.5926 | -0.2130 |
| keypeople_voting | 0.3796 | 0.5926 | -0.2130 |
| name_accusim | 0.3796 | 0.5926 | -0.2130 |
| name_casefusion | 0.3426 | 0.5741 | -0.2315 |
| name_fusionquery | 0.4074 | 0.5648 | -0.1574 |
| name_llm_judge | 0.3796 | 0.5926 | -0.2130 |
| name_longest_string | 0.3611 | 0.5648 | -0.2037 |
| name_most_complete | 0.3611 | 0.5648 | -0.2037 |
| name_prefer_higher_trust | 0.4074 | 0.5648 | -0.1574 |
| name_voting | 0.3796 | 0.5926 | -0.2130 |
| revenue_fusionquery | 0.3796 | 0.5926 | -0.2130 |
| revenue_huber_m_estimator | 0.3796 | 0.5833 | -0.2037 |
| revenue_maximum | 0.3796 | 0.5926 | -0.2130 |
| revenue_median | 0.3796 | 0.5833 | -0.2037 |
| revenue_median_of_means | 0.3796 | 0.5833 | -0.2037 |
| revenue_prefer_higher_trust | 0.3704 | 0.5833 | -0.2130 |
| revenue_trimmed_mean | 0.3796 | 0.5833 | -0.2037 |

## Stage: fusion - per attribute

| attribute | best_accuracy | baseline | delta | spread | spread_baseline | spread_delta |
|---|---|---|---|---|---|---|
| assets | 0.0000 | 0.0556 | -0.0556 | 0.0000 | 0.0556 | -0.0556 |
| city | 0.5556 | 0.8333 | -0.2778 | 0.2222 | 0.8333 | -0.6111 |
| country | 0.6667 | 0.9444 | -0.2778 | 0.5000 | 0.9444 | -0.4444 |
| founded | 0.5000 | 0.8889 | -0.3889 | 0.2222 | 0.4444 | -0.2222 |
| industry | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| keypeople | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| name | 0.8889 | 1.0000 | -0.1111 | 0.3889 | 0.1667 | 0.2222 |
| revenue | 0.0556 | 0.1111 | -0.0556 | 0.0556 | 0.0556 | 0.0000 |
