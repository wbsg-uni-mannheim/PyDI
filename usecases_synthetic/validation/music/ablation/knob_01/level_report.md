# Validation report - music / hard

_Generated at 2026-05-03T08:28:24.195664+00:00_

- domain: `music`
- level: `hard`
- with_llm: `False`
- committee_versions: sm=`sm_committee.yaml@5022dec6c8d2`, em=`em_blocking_committee_music.yaml@189cb163d2e7+em_matching_committee_music.yaml@f3010dcb53aa`, fusion=`fusion_committee_music.yaml@49086db928a4`

## Stage summary

| stage | metric | measured | baseline | delta |
|---|---|---|---|---|
| sm | macro_f1 | 0.4704 | 0.3620 | 0.1084 |
| em | macro_f1 | 0.9639 | 0.8508 | 0.1131 |
| fusion | overall_accuracy | 0.5776 | 0.5501 | 0.0276 |

## Stage: sm - per member

| member | f1 | f1_baseline | f1_delta |
|---|---|---|---|
| coma_hybrid | 0.8372 | 0.3448 | 0.4924 |
| duplicate_majority | 0.0000 | 0.0000 | 0.0000 |
| embedding_sbert | 0.3243 | 0.2353 | 0.0890 |
| llm_openai | 0.7200 | 0.8679 | -0.1479 |

## Stage: em - per member

| member | f1 | f1_baseline | f1_delta | pool_precision | pool_precision_baseline | pool_recall | pool_recall_baseline |
|---|---|---|---|---|---|---|---|
| ditto_plm | 0.9639 | 0.8508 | 0.1131 | 0.7056 | 0.9146 | 0.3913 | 0.3945 |

## Stage: em - per pair

| pair | member | f1 | f1_baseline | f1_delta |
|---|---|---|---|---|
| musicbrainz_discogs | ditto_plm | 0.9626 | 0.7966 | 0.1660 |
| musicbrainz_lastfm | ditto_plm | 0.9652 | 0.9050 | 0.0602 |

## Stage: fusion - per member

| member | overall_accuracy | overall_accuracy_baseline | overall_accuracy_delta |
|---|---|---|---|
| artist_llm_judge | 0.5430 | 0.5497 | -0.0066 |
| artist_longest_string | 0.5563 | 0.5497 | 0.0066 |
| artist_prefer_higher_trust | 0.5430 | 0.5497 | -0.0066 |
| artist_truthfinder | 0.5497 | 0.5497 | 0.0000 |
| artist_voting | 0.5430 | 0.5497 | -0.0066 |
| duration_fusionquery | 0.5430 | 0.5497 | -0.0066 |
| duration_huber_m_estimator | 0.5430 | 0.5497 | -0.0066 |
| duration_maximum | 0.5430 | 0.5497 | -0.0066 |
| duration_median | 0.5430 | 0.5497 | -0.0066 |
| duration_median_of_means | 0.5430 | 0.5497 | -0.0066 |
| duration_prefer_higher_trust | 0.5430 | 0.5497 | -0.0066 |
| duration_trimmed_mean | 0.5430 | 0.5497 | -0.0066 |
| genre_accusim | 0.5430 | 0.5497 | -0.0066 |
| genre_most_complete | 0.5430 | 0.5497 | -0.0066 |
| genre_prefer_higher_trust | 0.5430 | 0.5497 | -0.0066 |
| genre_voting | 0.5430 | 0.5497 | -0.0066 |
| label_longest_string | 0.5430 | 0.5497 | -0.0066 |
| label_ltm | 0.5430 | 0.5497 | -0.0066 |
| label_prefer_higher_trust | 0.5430 | 0.5497 | -0.0066 |
| label_voting | 0.5430 | 0.5497 | -0.0066 |
| name_accusim | 0.5497 | 0.5497 | 0.0000 |
| name_casefusion | 0.5430 | 0.5497 | -0.0066 |
| name_fusionquery | 0.5430 | 0.5497 | -0.0066 |
| name_llm_judge | 0.5430 | 0.5497 | -0.0066 |
| name_longest_string | 0.5364 | 0.5364 | 0.0000 |
| name_most_complete | 0.5364 | 0.5364 | 0.0000 |
| name_prefer_higher_trust | 0.5430 | 0.5497 | -0.0066 |
| name_voting | 0.5430 | 0.5497 | -0.0066 |
| release-country_llm_judge | 0.5430 | 0.5497 | -0.0066 |
| release-country_most_complete | 0.5497 | 0.5828 | -0.0331 |
| release-country_prefer_higher_trust | 0.5430 | 0.5497 | -0.0066 |
| release-country_truthfinder | 0.5364 | 0.5497 | -0.0132 |
| release-country_voting | 0.5430 | 0.5497 | -0.0066 |
| release-date_earliest | 0.5430 | 0.5497 | -0.0066 |
| release-date_prefer_higher_trust | 0.5430 | 0.5497 | -0.0066 |
| release-date_voting | 0.5430 | 0.5497 | -0.0066 |

## Stage: fusion - per attribute

| attribute | best_accuracy | baseline | delta | spread | spread_baseline | spread_delta |
|---|---|---|---|---|---|---|
| artist | 0.9565 | 0.9565 | 0.0000 | 0.0870 | 0.0000 | 0.0870 |
| duration | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| genre | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| label | 1.0000 | 0.9375 | 0.0625 | 0.0000 | 0.0000 | 0.0000 |
| name | 0.8696 | 0.8261 | 0.0435 | 0.0870 | 0.0870 | -0.0000 |
| release-country | 0.2609 | 0.2174 | 0.0435 | 0.0870 | 0.2174 | -0.1304 |
| release-date | 0.9565 | 0.9130 | 0.0435 | 0.0000 | 0.0000 | 0.0000 |
