# Validation report - music-small / easy

_Generated at 2026-05-15T15:13:46.004393+00:00_

- domain: `music-small`
- level: `easy`
- with_llm: `True`
- committee_versions: sm=`sm_committee.yaml@cb4a6847ac9e`, norm=`normalization_committee_music.yaml@c06387b2e5ef`, em_blocking=`em_blocking_committee_music.yaml@1eed2384fe7e`, em_matching=`em_matching_committee_music.yaml@fef75de2e979`, fusion=`fusion_committee_music.yaml@4a03339ecdef`

## Stage summary

| stage | metric | measured | baseline | delta |
|---|---|---|---|---|
| sm | macro_f1 | 0.9022 | 0.8742 | 0.0280 |
| norm | macro_f1 | 0.3908 | 0.4368 | -0.0460 |
| em_blocking | macro_pair_recall | 0.9701 | 0.9511 | 0.0190 |
| em_matching | macro_f1_vs_test | 0.8750 | 0.7499 | 0.1252 |
| fusion | overall_accuracy | 0.8531 | 0.8594 | -0.0062 |

## Stage: sm - per member

| member | f1 | f1_baseline | f1_delta |
|---|---|---|---|
| coma_hybrid | 1.0000 | 1.0000 | 0.0000 |
| duplicate_majority | 0.7442 | 0.7442 | 0.0000 |
| embedding_sbert | 0.8364 | 0.7500 | 0.0864 |
| instance_tf_cosine | 0.7347 | 0.6250 | 0.1097 |
| label_jw | 1.0000 | 1.0000 | 0.0000 |
| llm_openai | 1.0000 | 1.0000 | 0.0000 |
| magneto_slm_llm | 1.0000 | 1.0000 | 0.0000 |

## Stage: norm - per member

| member | macro_f1 | macro_f1_baseline | macro_f1_delta |
|---|---|---|---|
| country_iso | 0.1385 | 0.1340 | 0.0045 |
| date_iso | 0.3991 | 0.6541 | -0.2550 |
| llm_canonicalize | 0.1295 | 0.1295 | 0.0000 |
| number_locale | 0.6189 | 0.6506 | -0.0317 |
| taxonomy_lookup | 0.2846 | 0.2846 | 0.0000 |
| text_clean | 0.7744 | 0.7683 | 0.0061 |

## Stage: em_blocking - per member

| member | pair_recall | pair_recall_baseline | pair_recall_delta | reduction_ratio | reduction_ratio_baseline |
|---|---|---|---|---|---|
| bm25_blocker | 0.9956 | 0.9951 | 0.0005 | 0.9965 | 0.9962 |
| embedding_blocker | 0.9960 | 0.9952 | 0.0009 | 0.9965 | 0.9962 |
| sc_block | 0.9990 | 0.9993 | -0.0003 | 0.9965 | 0.9962 |
| sorted_neighbourhood_blocker | 0.9106 | 0.8540 | 0.0566 | 0.9959 | 0.9957 |
| standard_blocker | 0.9205 | 0.8639 | 0.0566 | 0.9847 | 0.9858 |
| token_blocker | 0.9990 | 0.9991 | -0.0002 | 0.9075 | 0.9064 |

## Stage: em_matching - per member

| member | f1 | f1_baseline | f1_delta | f1_vs_val | f1_vs_val_baseline | f1_vs_test | f1_vs_test_baseline |
|---|---|---|---|---|---|---|---|
| comem | 0.7838 | 0.5363 | 0.2474 | nan | nan | 0.7838 | 0.5363 |
| ditto_plm | 0.9756 | 0.9756 | 0.0000 | 0.8667 | 0.8667 | 0.9756 | 0.9756 |
| llm_matcher | 0.7982 | 0.5396 | 0.2586 | nan | nan | 0.7982 | 0.5396 |
| magellan | 0.9425 | 0.9480 | -0.0054 | 0.9598 | 0.9482 | 0.9425 | 0.9480 |

## Stage: em - per pair

| pair | member | f1 | f1_baseline | f1_delta |
|---|---|---|---|---|
| musicbrainz_discogs | comem | 0.7661 | 0.7091 | 0.0571 |
| musicbrainz_discogs | ditto_plm | 0.9780 | 0.9780 | 0.0000 |
| musicbrainz_discogs | llm_matcher | 0.7979 | 0.7115 | 0.0863 |
| musicbrainz_discogs | magellan | 0.9016 | 0.9156 | -0.0139 |
| musicbrainz_lastfm | comem | 0.8014 | 0.3636 | 0.4378 |
| musicbrainz_lastfm | ditto_plm | 0.9731 | 0.9731 | 0.0000 |
| musicbrainz_lastfm | llm_matcher | 0.7986 | 0.3676 | 0.4309 |
| musicbrainz_lastfm | magellan | 0.9834 | 0.9803 | 0.0031 |

## Stage: fusion - per member

| member | overall_accuracy | overall_accuracy_baseline | overall_accuracy_delta |
|---|---|---|---|
| artist_llm_judge | 0.8169 | 0.8169 | 0.0000 |
| artist_longest_string | 0.8091 | 0.8091 | 0.0000 |
| artist_prefer_higher_trust | 0.8169 | 0.8208 | -0.0039 |
| artist_truthfinder | 0.8169 | 0.8195 | -0.0026 |
| artist_voting | 0.8182 | 0.8208 | -0.0026 |
| duration_fusionquery | 0.8169 | 0.8260 | -0.0091 |
| duration_huber_m_estimator | 0.8052 | 0.8026 | 0.0026 |
| duration_maximum | 0.8247 | 0.8364 | -0.0117 |
| duration_median | 0.8117 | 0.8169 | -0.0052 |
| duration_median_of_means | 0.8052 | 0.8026 | 0.0026 |
| duration_prefer_higher_trust | 0.8143 | 0.8195 | -0.0052 |
| duration_trimmed_mean | 0.8052 | 0.8026 | 0.0026 |
| genre_accusim | 0.8169 | 0.8195 | -0.0026 |
| genre_most_complete | 0.8156 | 0.8182 | -0.0026 |
| genre_prefer_higher_trust | 0.8182 | 0.8208 | -0.0026 |
| genre_voting | 0.8182 | 0.8208 | -0.0026 |
| label_longest_string | 0.8208 | 0.8234 | -0.0026 |
| label_ltm | 0.8169 | 0.8195 | -0.0026 |
| label_prefer_higher_trust | 0.8182 | 0.8208 | -0.0026 |
| label_voting | 0.8182 | 0.8208 | -0.0026 |
| name_accusim | 0.8182 | 0.8221 | -0.0039 |
| name_casefusion | 0.8156 | 0.8195 | -0.0039 |
| name_fusionquery | 0.8013 | 0.8039 | -0.0026 |
| name_llm_judge | 0.8091 | 0.8117 | -0.0026 |
| name_longest_string | 0.8000 | 0.8000 | 0.0000 |
| name_most_complete | 0.8000 | 0.8000 | 0.0000 |
| name_prefer_higher_trust | 0.8156 | 0.8195 | -0.0039 |
| name_voting | 0.8182 | 0.8208 | -0.0026 |
| release-country_llm_judge | 0.8351 | 0.8299 | 0.0052 |
| release-country_most_complete | 0.8364 | 0.8273 | 0.0091 |
| release-country_prefer_higher_trust | 0.8260 | 0.8325 | -0.0065 |
| release-country_truthfinder | 0.8338 | 0.8286 | 0.0052 |
| release-country_voting | 0.8182 | 0.8208 | -0.0026 |
| release-date_earliest | 0.8130 | 0.8169 | -0.0039 |
| release-date_prefer_higher_trust | 0.8182 | 0.8195 | -0.0013 |
| release-date_voting | 0.8182 | 0.8208 | -0.0026 |
| tracks_intersection | 0.7987 | 0.8000 | -0.0013 |
| tracks_intersection_k_sources | 0.8169 | 0.8182 | -0.0013 |
| tracks_ltm | 0.8182 | 0.8143 | 0.0039 |
| tracks_prefer_higher_trust | 0.8182 | 0.8208 | -0.0026 |
| tracks_union | 0.8247 | 0.8273 | -0.0026 |
| tracks_voting | 0.8182 | 0.8208 | -0.0026 |

## Stage: fusion - per attribute

| attribute | best_accuracy | baseline | delta | spread | spread_baseline | spread_delta |
|---|---|---|---|---|---|---|
| artist | 0.9100 | 0.9300 | -0.0200 | 0.0700 | 0.0900 | -0.0200 |
| duration | 0.4000 | 0.4200 | -0.0200 | 0.1500 | 0.2600 | -0.1100 |
| genre | 0.8831 | 0.8831 | 0.0000 | 0.0260 | 0.0260 | 0.0000 |
| label | 0.8817 | 0.8817 | 0.0000 | 0.0323 | 0.0323 | 0.0000 |
| name | 0.9300 | 0.9000 | 0.0300 | 0.1400 | 0.1700 | -0.0300 |
| release-country | 0.9300 | 0.9700 | -0.0400 | 0.1400 | 0.0900 | 0.0500 |
| release-date | 0.9300 | 0.9200 | 0.0100 | 0.0400 | 0.0300 | 0.0100 |
| tracks | 0.9600 | 0.9700 | -0.0100 | 0.2000 | 0.2100 | -0.0100 |
