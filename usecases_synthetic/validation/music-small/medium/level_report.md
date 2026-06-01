# Validation report - music-small / medium

_Generated at 2026-05-15T16:02:12.961278+00:00_

- domain: `music-small`
- level: `medium`
- with_llm: `True`
- committee_versions: sm=`sm_committee.yaml@cb4a6847ac9e`, norm=`normalization_committee_music.yaml@c06387b2e5ef`, em_blocking=`em_blocking_committee_music.yaml@1eed2384fe7e`, em_matching=`em_matching_committee_music.yaml@fef75de2e979`, fusion=`fusion_committee_music.yaml@4a03339ecdef`

## Stage summary

| stage | metric | measured | baseline | delta |
|---|---|---|---|---|
| sm | macro_f1 | 0.7514 | 0.8742 | -0.1228 |
| norm | macro_f1 | 0.3393 | 0.4368 | -0.0975 |
| em_blocking | macro_pair_recall | 0.9468 | 0.9511 | -0.0043 |
| em_matching | macro_f1_vs_test | 0.8937 | 0.7499 | 0.1438 |
| fusion | overall_accuracy | 0.7906 | 0.8594 | -0.0687 |

## Stage: sm - per member

| member | f1 | f1_baseline | f1_delta |
|---|---|---|---|
| coma_hybrid | 0.7143 | 1.0000 | -0.2857 |
| duplicate_majority | 0.6500 | 0.7442 | -0.0942 |
| embedding_sbert | 0.7451 | 0.7500 | -0.0049 |
| instance_tf_cosine | 0.5778 | 0.6250 | -0.0472 |
| label_jw | 0.6977 | 1.0000 | -0.3023 |
| llm_openai | 1.0000 | 1.0000 | 0.0000 |
| magneto_slm_llm | 0.8750 | 1.0000 | -0.1250 |

## Stage: norm - per member

| member | macro_f1 | macro_f1_baseline | macro_f1_delta |
|---|---|---|---|
| country_iso | 0.1307 | 0.1340 | -0.0033 |
| date_iso | 0.3957 | 0.6541 | -0.2583 |
| llm_canonicalize | 0.1295 | 0.1295 | 0.0000 |
| number_locale | 0.3418 | 0.6506 | -0.3088 |
| taxonomy_lookup | 0.2811 | 0.2846 | -0.0035 |
| text_clean | 0.7571 | 0.7683 | -0.0112 |

## Stage: em_blocking - per member

| member | pair_recall | pair_recall_baseline | pair_recall_delta | reduction_ratio | reduction_ratio_baseline |
|---|---|---|---|---|---|
| bm25_blocker | 0.9946 | 0.9951 | -0.0005 | 0.9962 | 0.9962 |
| embedding_blocker | 0.9920 | 0.9952 | -0.0032 | 0.9962 | 0.9962 |
| sc_block | 0.9966 | 0.9993 | -0.0027 | 0.9962 | 0.9962 |
| sorted_neighbourhood_blocker | 0.8435 | 0.8540 | -0.0104 | 0.9958 | 0.9957 |
| standard_blocker | 0.8552 | 0.8639 | -0.0087 | 0.9855 | 0.9858 |
| token_blocker | 0.9990 | 0.9991 | -0.0002 | 0.9044 | 0.9064 |

## Stage: em_matching - per member

| member | f1 | f1_baseline | f1_delta | f1_vs_val | f1_vs_val_baseline | f1_vs_test | f1_vs_test_baseline |
|---|---|---|---|---|---|---|---|
| comem | 0.8190 | 0.5363 | 0.2826 | nan | nan | 0.8190 | 0.5363 |
| ditto_plm | 0.9756 | 0.9756 | 0.0000 | 0.8667 | 0.8667 | 0.9756 | 0.9756 |
| llm_matcher | 0.8388 | 0.5396 | 0.2992 | nan | nan | 0.8388 | 0.5396 |
| magellan | 0.9415 | 0.9480 | -0.0065 | 0.9311 | 0.9482 | 0.9415 | 0.9480 |

## Stage: em - per pair

| pair | member | f1 | f1_baseline | f1_delta |
|---|---|---|---|---|
| musicbrainz_discogs | comem | 0.8833 | 0.7091 | 0.1743 |
| musicbrainz_discogs | ditto_plm | 0.9780 | 0.9780 | 0.0000 |
| musicbrainz_discogs | llm_matcher | 0.9179 | 0.7115 | 0.2063 |
| musicbrainz_discogs | magellan | 0.9073 | 0.9156 | -0.0083 |
| musicbrainz_lastfm | comem | 0.7546 | 0.3636 | 0.3909 |
| musicbrainz_lastfm | ditto_plm | 0.9731 | 0.9731 | 0.0000 |
| musicbrainz_lastfm | llm_matcher | 0.7596 | 0.3676 | 0.3920 |
| musicbrainz_lastfm | magellan | 0.9756 | 0.9803 | -0.0047 |

## Stage: fusion - per member

| member | overall_accuracy | overall_accuracy_baseline | overall_accuracy_delta |
|---|---|---|---|
| artist_llm_judge | 0.6455 | 0.8169 | -0.1714 |
| artist_longest_string | 0.6351 | 0.8091 | -0.1740 |
| artist_prefer_higher_trust | 0.6416 | 0.8208 | -0.1792 |
| artist_truthfinder | 0.6403 | 0.8195 | -0.1792 |
| artist_voting | 0.6429 | 0.8208 | -0.1779 |
| duration_fusionquery | 0.6364 | 0.8260 | -0.1896 |
| duration_huber_m_estimator | 0.6558 | 0.8026 | -0.1468 |
| duration_maximum | 0.6558 | 0.8364 | -0.1805 |
| duration_median | 0.6558 | 0.8169 | -0.1610 |
| duration_median_of_means | 0.6558 | 0.8026 | -0.1468 |
| duration_prefer_higher_trust | 0.6455 | 0.8195 | -0.1740 |
| duration_trimmed_mean | 0.6558 | 0.8026 | -0.1468 |
| genre_accusim | 0.6416 | 0.8195 | -0.1779 |
| genre_most_complete | 0.6403 | 0.8182 | -0.1779 |
| genre_prefer_higher_trust | 0.6429 | 0.8208 | -0.1779 |
| genre_voting | 0.6429 | 0.8208 | -0.1779 |
| label_longest_string | 0.6468 | 0.8234 | -0.1766 |
| label_ltm | 0.6416 | 0.8195 | -0.1779 |
| label_prefer_higher_trust | 0.6429 | 0.8208 | -0.1779 |
| label_voting | 0.6429 | 0.8208 | -0.1779 |
| name_accusim | 0.6468 | 0.8221 | -0.1753 |
| name_casefusion | 0.6442 | 0.8195 | -0.1753 |
| name_fusionquery | 0.6286 | 0.8039 | -0.1753 |
| name_llm_judge | 0.6377 | 0.8117 | -0.1740 |
| name_longest_string | 0.6221 | 0.8000 | -0.1779 |
| name_most_complete | 0.6221 | 0.8000 | -0.1779 |
| name_prefer_higher_trust | 0.6442 | 0.8195 | -0.1753 |
| name_voting | 0.6429 | 0.8208 | -0.1779 |
| release-country_llm_judge | 0.6766 | 0.8299 | -0.1532 |
| release-country_most_complete | 0.6779 | 0.8273 | -0.1494 |
| release-country_prefer_higher_trust | 0.6506 | 0.8325 | -0.1818 |
| release-country_truthfinder | 0.6766 | 0.8286 | -0.1519 |
| release-country_voting | 0.6429 | 0.8208 | -0.1779 |
| release-date_earliest | 0.6390 | 0.8169 | -0.1779 |
| release-date_prefer_higher_trust | 0.6429 | 0.8195 | -0.1766 |
| release-date_voting | 0.6429 | 0.8208 | -0.1779 |
| tracks_intersection | 0.6351 | 0.8000 | -0.1649 |
| tracks_intersection_k_sources | 0.6506 | 0.8182 | -0.1675 |
| tracks_ltm | 0.6896 | 0.8143 | -0.1247 |
| tracks_prefer_higher_trust | 0.6429 | 0.8208 | -0.1779 |
| tracks_union | 0.7299 | 0.8273 | -0.0974 |
| tracks_voting | 0.6429 | 0.8208 | -0.1779 |

## Stage: fusion - per attribute

| attribute | best_accuracy | baseline | delta | spread | spread_baseline | spread_delta |
|---|---|---|---|---|---|---|
| artist | 0.9000 | 0.9300 | -0.0300 | 0.0800 | 0.0900 | -0.0100 |
| duration | 0.2700 | 0.4200 | -0.1500 | 0.1500 | 0.2600 | -0.1100 |
| genre | 0.8571 | 0.8831 | -0.0260 | 0.0260 | 0.0260 | -0.0000 |
| label | 0.8280 | 0.8817 | -0.0538 | 0.0430 | 0.0323 | 0.0108 |
| name | 0.9200 | 0.9000 | 0.0200 | 0.1900 | 0.1700 | 0.0200 |
| release-country | 0.8500 | 0.9700 | -0.1200 | 0.2700 | 0.0900 | 0.1800 |
| release-date | 0.9200 | 0.9200 | 0.0000 | 0.0300 | 0.0300 | 0.0000 |
| tracks | 0.7800 | 0.9700 | -0.1900 | 0.7300 | 0.2100 | 0.5200 |
