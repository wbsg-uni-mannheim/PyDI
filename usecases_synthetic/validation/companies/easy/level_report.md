# Validation report - companies / easy

_Generated at 2026-05-12T21:14:34.697310+00:00_

- domain: `companies`
- level: `easy`
- with_llm: `False`
- committee_versions: fusion=`fusion_committee.yaml@d569a2b1af6f`

## Stage summary

| stage | metric | measured | baseline | delta |
|---|---|---|---|---|
| fusion | overall_accuracy | 0.0000 | 0.8810 | -0.8810 |

## Stage: fusion - per member

| member | overall_accuracy | overall_accuracy_baseline | overall_accuracy_delta |
|---|---|---|---|
| assets_fusionquery | 0.0000 | 0.8376 | -0.8376 |
| assets_huber_m_estimator | 0.0000 | 0.7863 | -0.7863 |
| assets_maximum | 0.0000 | 0.8034 | -0.8034 |
| assets_median | 0.0000 | 0.7863 | -0.7863 |
| assets_median_of_means | 0.0000 | 0.7863 | -0.7863 |
| assets_prefer_higher_trust | 0.0000 | 0.7949 | -0.7949 |
| assets_trimmed_mean | 0.0000 | 0.7863 | -0.7863 |
| city_llm_judge | 0.0000 | 0.8462 | -0.8462 |
| city_prefer_higher_trust | 0.0000 | 0.8632 | -0.8632 |
| city_shortest_string | 0.0000 | 0.8803 | -0.8803 |
| city_truthfinder | 0.0000 | 0.8632 | -0.8632 |
| city_voting | 0.0000 | 0.8376 | -0.8376 |
| country_favour_forbes | 0.0000 | 0.8376 | -0.8376 |
| country_llm_judge | 0.0000 | 0.7863 | -0.7863 |
| country_prefer_higher_trust | 0.0000 | 0.7863 | -0.7863 |
| country_truthfinder | 0.0000 | 0.8462 | -0.8462 |
| country_voting | 0.0000 | 0.8376 | -0.8376 |
| founded_earliest | 0.0000 | 0.8376 | -0.8376 |
| founded_prefer_higher_trust | 0.0000 | 0.8376 | -0.8376 |
| founded_voting | 0.0000 | 0.8376 | -0.8376 |
| keypeople_intersection | 0.0000 | 0.8120 | -0.8120 |
| keypeople_intersection_k_sources | 0.0000 | 0.8120 | -0.8120 |
| keypeople_ltm | 0.0000 | 0.8376 | -0.8376 |
| keypeople_prefer_higher_trust | 0.0000 | 0.8462 | -0.8462 |
| keypeople_union | 0.0000 | 0.8462 | -0.8462 |
| keypeople_voting | 0.0000 | 0.8376 | -0.8376 |
| name_accusim | 0.0000 | 0.8376 | -0.8376 |
| name_casefusion | 0.0000 | 0.8120 | -0.8120 |
| name_fusionquery | 0.0000 | 0.8034 | -0.8034 |
| name_llm_judge | 0.0000 | 0.8120 | -0.8120 |
| name_longest_string | 0.0000 | 0.8120 | -0.8120 |
| name_most_complete | 0.0000 | 0.8120 | -0.8120 |
| name_prefer_higher_trust | 0.0000 | 0.8120 | -0.8120 |
| name_voting | 0.0000 | 0.8376 | -0.8376 |
| revenue_fusionquery | 0.0000 | 0.8376 | -0.8376 |
| revenue_huber_m_estimator | 0.0000 | 0.7863 | -0.7863 |
| revenue_maximum | 0.0000 | 0.8291 | -0.8291 |
| revenue_median | 0.0000 | 0.7863 | -0.7863 |
| revenue_median_of_means | 0.0000 | 0.7863 | -0.7863 |
| revenue_prefer_higher_trust | 0.0000 | 0.8120 | -0.8120 |
| revenue_trimmed_mean | 0.0000 | 0.7863 | -0.7863 |

## Stage: fusion - per attribute

| attribute | best_accuracy | baseline | delta | spread | spread_baseline | spread_delta |
|---|---|---|---|---|---|---|
| assets | 0.0000 | 0.9444 | -0.9444 | 0.0000 | 0.3333 | -0.3333 |
| city | 0.0000 | 0.8333 | -0.8333 | 0.0000 | 0.2778 | -0.2778 |
| country | 0.0000 | 1.0000 | -1.0000 | 0.0000 | 0.3889 | -0.3889 |
| founded | 0.0000 | 0.9444 | -0.9444 | 0.0000 | 0.0000 | 0.0000 |
| keypeople | 0.0000 | 0.6667 | -0.6667 | 0.0000 | 0.4444 | -0.4444 |
| name | 0.0000 | 1.0000 | -1.0000 | 0.0000 | 0.2222 | -0.2222 |
| revenue | 0.0000 | 0.7778 | -0.7778 | 0.0000 | 0.3333 | -0.3333 |
