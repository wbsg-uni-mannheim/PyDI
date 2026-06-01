# Validation report - music / medium

_Generated at 2026-05-16T10:21:06.112144+00:00_

- domain: `music`
- level: `medium`
- with_llm: `True`
- committee_versions: sm=`sm_committee.yaml@cb4a6847ac9e`, norm=`normalization_committee_music.yaml@c06387b2e5ef`, em_blocking=`em_blocking_committee_music.yaml@1eed2384fe7e`, em_matching=`em_matching_committee_music.yaml@fef75de2e979`, fusion=`fusion_committee_music.yaml@4a03339ecdef`

## Stage summary

| stage | metric | measured | baseline | delta |
|---|---|---|---|---|
| sm | macro_f1 | 0.7604 | 0.8761 | -0.1157 |
| norm | macro_f1 | 0.3613 | 0.4337 | -0.0724 |
| em_blocking | macro_pair_recall | 0.9481 | 0.9511 | -0.0030 |
| em_matching | macro_f1_vs_test | 0.7957 | 0.7499 | 0.0458 |
| fusion | overall_accuracy | 0.8072 | 0.8594 | -0.0522 |

## Stage: sm - per member

| member | f1 | f1_baseline | f1_delta |
|---|---|---|---|
| coma_hybrid | 0.7143 | 1.0000 | -0.2857 |
| duplicate_majority | 0.7143 | 0.7442 | -0.0299 |
| embedding_sbert | 0.7308 | 0.7500 | -0.0192 |
| instance_tf_cosine | 0.5909 | 0.6383 | -0.0474 |
| label_jw | 0.6977 | 1.0000 | -0.3023 |
| llm_openai | 1.0000 | 1.0000 | 0.0000 |
| magneto_slm_llm | 0.8750 | 1.0000 | -0.1250 |

## Stage: norm - per member

| member | macro_f1 | macro_f1_baseline | macro_f1_delta |
|---|---|---|---|
| country_iso | 0.1304 | 0.1340 | -0.0036 |
| date_iso | 0.6556 | 0.6541 | 0.0015 |
| llm_canonicalize | 0.1088 | 0.1108 | -0.0020 |
| number_locale | 0.2338 | 0.6506 | -0.4168 |
| taxonomy_lookup | 0.2811 | 0.2846 | -0.0035 |
| text_clean | 0.7584 | 0.7683 | -0.0098 |

## Stage: em_blocking - per member

| member | pair_recall | pair_recall_baseline | pair_recall_delta | reduction_ratio | reduction_ratio_baseline |
|---|---|---|---|---|---|
| bm25_blocker | 0.9944 | 0.9951 | -0.0007 | 0.9964 | 0.9964 |
| embedding_blocker | 0.9938 | 0.9952 | -0.0014 | 0.9964 | 0.9964 |
| sc_block | 0.9968 | 0.9993 | -0.0025 | 0.9964 | 0.9964 |
| sorted_neighbourhood_blocker | 0.8469 | 0.8540 | -0.0071 | 0.9959 | 0.9959 |
| standard_blocker | 0.8575 | 0.8639 | -0.0064 | 0.9858 | 0.9858 |
| token_blocker | 0.9991 | 0.9991 | 0.0000 | 0.9064 | 0.9063 |

## Stage: em_matching - per member

| member | f1 | f1_baseline | f1_delta | f1_vs_val | f1_vs_val_baseline | f1_vs_test | f1_vs_test_baseline |
|---|---|---|---|---|---|---|---|
| comem | 0.6195 | 0.5363 | 0.0832 | nan | nan | 0.6195 | 0.5363 |
| ditto_plm | 0.9756 | 0.9756 | 0.0000 | 0.8667 | 0.8667 | 0.9756 | 0.9756 |
| llm_matcher | 0.6410 | 0.5396 | 0.1014 | nan | nan | 0.6410 | 0.5396 |
| magellan | 0.9466 | 0.9480 | -0.0013 | 0.9383 | 0.9482 | 0.9466 | 0.9480 |

## Stage: em - per pair

| pair | member | f1 | f1_baseline | f1_delta |
|---|---|---|---|---|
| musicbrainz_discogs | comem | 0.8428 | 0.7091 | 0.1338 |
| musicbrainz_discogs | ditto_plm | 0.9780 | 0.9780 | 0.0000 |
| musicbrainz_discogs | llm_matcher | 0.8483 | 0.7115 | 0.1367 |
| musicbrainz_discogs | magellan | 0.9176 | 0.9156 | 0.0020 |
| musicbrainz_lastfm | comem | 0.3962 | 0.3636 | 0.0326 |
| musicbrainz_lastfm | ditto_plm | 0.9731 | 0.9731 | 0.0000 |
| musicbrainz_lastfm | llm_matcher | 0.4338 | 0.3676 | 0.0661 |
| musicbrainz_lastfm | magellan | 0.9757 | 0.9803 | -0.0046 |

## Stage: fusion - per member

| member | overall_accuracy | overall_accuracy_baseline | overall_accuracy_delta |
|---|---|---|---|
| artist_llm_judge | 0.6610 | 0.8169 | -0.1558 |
| artist_longest_string | 0.6494 | 0.8091 | -0.1597 |
| artist_prefer_higher_trust | 0.6571 | 0.8208 | -0.1636 |
| artist_truthfinder | 0.6545 | 0.8195 | -0.1649 |
| artist_voting | 0.6558 | 0.8208 | -0.1649 |
| duration_fusionquery | 0.6468 | 0.8260 | -0.1792 |
| duration_huber_m_estimator | 0.6766 | 0.8026 | -0.1260 |
| duration_maximum | 0.6792 | 0.8364 | -0.1571 |
| duration_median | 0.6766 | 0.8169 | -0.1403 |
| duration_median_of_means | 0.6766 | 0.8026 | -0.1260 |
| duration_prefer_higher_trust | 0.6558 | 0.8195 | -0.1636 |
| duration_trimmed_mean | 0.6766 | 0.8026 | -0.1260 |
| genre_accusim | 0.6545 | 0.8195 | -0.1649 |
| genre_most_complete | 0.6532 | 0.8182 | -0.1649 |
| genre_prefer_higher_trust | 0.6558 | 0.8208 | -0.1649 |
| genre_voting | 0.6558 | 0.8208 | -0.1649 |
| label_longest_string | 0.6571 | 0.8234 | -0.1662 |
| label_ltm | 0.6532 | 0.8195 | -0.1662 |
| label_prefer_higher_trust | 0.6558 | 0.8208 | -0.1649 |
| label_voting | 0.6558 | 0.8208 | -0.1649 |
| name_accusim | 0.6597 | 0.8221 | -0.1623 |
| name_casefusion | 0.6519 | 0.8195 | -0.1675 |
| name_fusionquery | 0.6390 | 0.8039 | -0.1649 |
| name_llm_judge | 0.6455 | 0.8117 | -0.1662 |
| name_longest_string | 0.6338 | 0.8000 | -0.1662 |
| name_most_complete | 0.6338 | 0.8000 | -0.1662 |
| name_prefer_higher_trust | 0.6519 | 0.8195 | -0.1675 |
| name_voting | 0.6558 | 0.8208 | -0.1649 |
| release-country_llm_judge | 0.6831 | 0.8299 | -0.1468 |
| release-country_most_complete | 0.6844 | 0.8273 | -0.1429 |
| release-country_prefer_higher_trust | 0.6623 | 0.8325 | -0.1701 |
| release-country_truthfinder | 0.6844 | 0.8286 | -0.1442 |
| release-country_voting | 0.6558 | 0.8208 | -0.1649 |
| release-date_earliest | 0.6532 | 0.8169 | -0.1636 |
| release-date_prefer_higher_trust | 0.6558 | 0.8195 | -0.1636 |
| release-date_voting | 0.6558 | 0.8208 | -0.1649 |
| tracks_intersection | 0.6494 | 0.8000 | -0.1506 |
| tracks_intersection_k_sources | 0.6649 | 0.8182 | -0.1532 |
| tracks_ltm | 0.6922 | 0.8143 | -0.1221 |
| tracks_prefer_higher_trust | 0.6558 | 0.8208 | -0.1649 |
| tracks_union | 0.7429 | 0.8273 | -0.0844 |
| tracks_voting | 0.6558 | 0.8208 | -0.1649 |

## Stage: fusion - per attribute

| attribute | best_accuracy | baseline | delta | spread | spread_baseline | spread_delta |
|---|---|---|---|---|---|---|
| artist | 0.9100 | 0.9300 | -0.0200 | 0.0900 | 0.0900 | 0.0000 |
| duration | 0.3600 | 0.4200 | -0.0600 | 0.2500 | 0.2600 | -0.0100 |
| genre | 0.8831 | 0.8831 | 0.0000 | 0.0260 | 0.0260 | 0.0000 |
| label | 0.7742 | 0.8817 | -0.1075 | 0.0323 | 0.0323 | 0.0000 |
| name | 0.9300 | 0.9000 | 0.0300 | 0.2000 | 0.1700 | 0.0300 |
| release-country | 0.9100 | 0.9700 | -0.0600 | 0.2200 | 0.0900 | 0.1300 |
| release-date | 0.9100 | 0.9200 | -0.0100 | 0.0200 | 0.0300 | -0.0100 |
| tracks | 0.7800 | 0.9700 | -0.1900 | 0.7200 | 0.2100 | 0.5100 |
