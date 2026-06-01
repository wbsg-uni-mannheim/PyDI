# Validation report - music-small / hard

_Generated at 2026-05-15T16:52:21.209003+00:00_

- domain: `music-small`
- level: `hard`
- with_llm: `True`
- committee_versions: sm=`sm_committee.yaml@cb4a6847ac9e`, norm=`normalization_committee_music.yaml@c06387b2e5ef`, em_blocking=`em_blocking_committee_music.yaml@1eed2384fe7e`, em_matching=`em_matching_committee_music.yaml@fef75de2e979`, fusion=`fusion_committee_music.yaml@4a03339ecdef`

## Stage summary

| stage | metric | measured | baseline | delta |
|---|---|---|---|---|
| sm | macro_f1 | 0.6496 | 0.8742 | -0.2246 |
| norm | macro_f1 | 0.3271 | 0.4368 | -0.1098 |
| em_blocking | macro_pair_recall | 0.5425 | 0.9511 | -0.4086 |
| em_matching | macro_f1_vs_test | 0.8348 | 0.7499 | 0.0850 |
| fusion | overall_accuracy | 0.5413 | 0.8594 | -0.3180 |

## Stage: sm - per member

| member | f1 | f1_baseline | f1_delta |
|---|---|---|---|
| coma_hybrid | 0.4118 | 1.0000 | -0.5882 |
| duplicate_majority | 0.7442 | 0.7442 | 0.0000 |
| embedding_sbert | 0.7347 | 0.7500 | -0.0153 |
| instance_tf_cosine | 0.5217 | 0.6250 | -0.1033 |
| label_jw | 0.3636 | 1.0000 | -0.6364 |
| llm_openai | 0.9200 | 1.0000 | -0.0800 |
| magneto_slm_llm | 0.8511 | 1.0000 | -0.1489 |

## Stage: norm - per member

| member | macro_f1 | macro_f1_baseline | macro_f1_delta |
|---|---|---|---|
| country_iso | 0.1438 | 0.1340 | 0.0098 |
| date_iso | 0.5785 | 0.6541 | -0.0756 |
| llm_canonicalize | 0.1237 | 0.1295 | -0.0058 |
| number_locale | 0.1475 | 0.6506 | -0.5031 |
| taxonomy_lookup | 0.1995 | 0.2846 | -0.0851 |
| text_clean | 0.7694 | 0.7683 | 0.0011 |

## Stage: em_blocking - per member

| member | pair_recall | pair_recall_baseline | pair_recall_delta | reduction_ratio | reduction_ratio_baseline |
|---|---|---|---|---|---|
| bm25_blocker | 0.5740 | 0.9951 | -0.4211 | 0.9959 | 0.9962 |
| embedding_blocker | 0.5755 | 0.9952 | -0.4197 | 0.9959 | 0.9962 |
| sc_block | 0.5923 | 0.9993 | -0.4070 | 0.9959 | 0.9962 |
| sorted_neighbourhood_blocker | 0.4640 | 0.8540 | -0.3899 | 0.9958 | 0.9957 |
| standard_blocker | 0.4700 | 0.8639 | -0.3939 | 0.9857 | 0.9858 |
| token_blocker | 0.5790 | 0.9991 | -0.4201 | 0.9035 | 0.9064 |

## Stage: em_matching - per member

| member | f1 | f1_baseline | f1_delta | f1_vs_val | f1_vs_val_baseline | f1_vs_test | f1_vs_test_baseline |
|---|---|---|---|---|---|---|---|
| comem | 0.7220 | 0.5363 | 0.1857 | nan | nan | 0.7220 | 0.5363 |
| ditto_plm | 0.9357 | 0.9756 | -0.0399 | 0.8203 | 0.8667 | 0.9357 | 0.9756 |
| llm_matcher | 0.7392 | 0.5396 | 0.1996 | nan | nan | 0.7392 | 0.5396 |
| magellan | 0.9424 | 0.9480 | -0.0056 | 0.9293 | 0.9482 | 0.9424 | 0.9480 |

## Stage: em - per pair

| pair | member | f1 | f1_baseline | f1_delta |
|---|---|---|---|---|
| musicbrainz_discogs | comem | 0.8057 | 0.7091 | 0.0966 |
| musicbrainz_discogs | ditto_plm | 0.9542 | 0.9780 | -0.0238 |
| musicbrainz_discogs | llm_matcher | 0.8156 | 0.7115 | 0.1041 |
| musicbrainz_discogs | magellan | 0.9123 | 0.9156 | -0.0032 |
| musicbrainz_lastfm | comem | 0.6384 | 0.3636 | 0.2747 |
| musicbrainz_lastfm | ditto_plm | 0.9172 | 0.9731 | -0.0560 |
| musicbrainz_lastfm | llm_matcher | 0.6627 | 0.3676 | 0.2951 |
| musicbrainz_lastfm | magellan | 0.9725 | 0.9803 | -0.0079 |

## Stage: fusion - per member

| member | overall_accuracy | overall_accuracy_baseline | overall_accuracy_delta |
|---|---|---|---|
| artist_llm_judge | 0.5104 | 0.8169 | -0.3065 |
| artist_longest_string | 0.5013 | 0.8091 | -0.3078 |
| artist_prefer_higher_trust | 0.5065 | 0.8208 | -0.3143 |
| artist_truthfinder | 0.5078 | 0.8195 | -0.3117 |
| artist_voting | 0.5065 | 0.8208 | -0.3143 |
| duration_fusionquery | 0.5078 | 0.8260 | -0.3182 |
| duration_huber_m_estimator | 0.5130 | 0.8026 | -0.2896 |
| duration_maximum | 0.5143 | 0.8364 | -0.3221 |
| duration_median | 0.5130 | 0.8169 | -0.3039 |
| duration_median_of_means | 0.5130 | 0.8026 | -0.2896 |
| duration_prefer_higher_trust | 0.5065 | 0.8195 | -0.3130 |
| duration_trimmed_mean | 0.5130 | 0.8026 | -0.2896 |
| genre_accusim | 0.5078 | 0.8195 | -0.3117 |
| genre_most_complete | 0.5065 | 0.8182 | -0.3117 |
| genre_prefer_higher_trust | 0.5065 | 0.8208 | -0.3143 |
| genre_voting | 0.5065 | 0.8208 | -0.3143 |
| label_longest_string | 0.5078 | 0.8234 | -0.3156 |
| label_ltm | 0.5078 | 0.8195 | -0.3117 |
| label_prefer_higher_trust | 0.5065 | 0.8208 | -0.3143 |
| label_voting | 0.5065 | 0.8208 | -0.3143 |
| name_accusim | 0.5078 | 0.8221 | -0.3143 |
| name_casefusion | 0.5026 | 0.8195 | -0.3169 |
| name_fusionquery | 0.5065 | 0.8039 | -0.2974 |
| name_llm_judge | 0.5013 | 0.8117 | -0.3104 |
| name_longest_string | 0.4974 | 0.8000 | -0.3026 |
| name_most_complete | 0.4974 | 0.8000 | -0.3026 |
| name_prefer_higher_trust | 0.5026 | 0.8195 | -0.3169 |
| name_voting | 0.5065 | 0.8208 | -0.3143 |
| release-country_llm_judge | 0.5286 | 0.8299 | -0.3013 |
| release-country_most_complete | 0.5299 | 0.8273 | -0.2974 |
| release-country_prefer_higher_trust | 0.5078 | 0.8325 | -0.3247 |
| release-country_truthfinder | 0.5273 | 0.8286 | -0.3013 |
| release-country_voting | 0.5065 | 0.8208 | -0.3143 |
| release-date_earliest | 0.5078 | 0.8169 | -0.3091 |
| release-date_prefer_higher_trust | 0.5039 | 0.8195 | -0.3156 |
| release-date_voting | 0.5065 | 0.8208 | -0.3143 |
| tracks_intersection | 0.5052 | 0.8000 | -0.2948 |
| tracks_intersection_k_sources | 0.5052 | 0.8182 | -0.3130 |
| tracks_ltm | 0.5078 | 0.8143 | -0.3065 |
| tracks_prefer_higher_trust | 0.5065 | 0.8208 | -0.3143 |
| tracks_union | 0.5052 | 0.8273 | -0.3221 |
| tracks_voting | 0.5065 | 0.8208 | -0.3143 |

## Stage: fusion - per attribute

| attribute | best_accuracy | baseline | delta | spread | spread_baseline | spread_delta |
|---|---|---|---|---|---|---|
| artist | 0.9100 | 0.9300 | -0.0200 | 0.0700 | 0.0900 | -0.0200 |
| duration | 0.1200 | 0.4200 | -0.3000 | 0.0600 | 0.2600 | -0.2000 |
| genre | 0.3766 | 0.8831 | -0.5065 | 0.0130 | 0.0260 | -0.0130 |
| label | 0.3441 | 0.8817 | -0.5376 | 0.0108 | 0.0323 | -0.0215 |
| name | 0.8300 | 0.9000 | -0.0700 | 0.0800 | 0.1700 | -0.0900 |
| release-country | 0.8300 | 0.9700 | -0.1400 | 0.1800 | 0.0900 | 0.0900 |
| release-date | 0.8900 | 0.9200 | -0.0300 | 0.0300 | 0.0300 | 0.0000 |
| tracks | 0.0300 | 0.9700 | -0.9400 | 0.0200 | 0.2100 | -0.1900 |
