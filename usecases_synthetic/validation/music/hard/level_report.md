# Validation report - music / hard

_Generated at 2026-05-16T10:57:06.542315+00:00_

- domain: `music`
- level: `hard`
- with_llm: `True`
- committee_versions: sm=`sm_committee.yaml@cb4a6847ac9e`, norm=`normalization_committee_music.yaml@c06387b2e5ef`, em_blocking=`em_blocking_committee_music.yaml@1eed2384fe7e`, em_matching=`em_matching_committee_music.yaml@fef75de2e979`, fusion=`fusion_committee_music.yaml@4a03339ecdef`

## Stage summary

| stage | metric | measured | baseline | delta |
|---|---|---|---|---|
| sm | macro_f1 | 0.6699 | 0.8761 | -0.2062 |
| norm | macro_f1 | 0.3175 | 0.4337 | -0.1162 |
| em_blocking | macro_pair_recall | 0.5432 | 0.9511 | -0.4079 |
| em_matching | macro_f1_vs_test | 0.6801 | 0.7499 | -0.0697 |
| fusion | overall_accuracy | 0.5738 | 0.8594 | -0.2856 |

## Stage: sm - per member

| member | f1 | f1_baseline | f1_delta |
|---|---|---|---|
| coma_hybrid | 0.4571 | 1.0000 | -0.5429 |
| duplicate_majority | 0.7442 | 0.7442 | 0.0000 |
| embedding_sbert | 0.7755 | 0.7500 | 0.0255 |
| instance_tf_cosine | 0.5778 | 0.6383 | -0.0605 |
| label_jw | 0.3636 | 1.0000 | -0.6364 |
| llm_openai | 0.9200 | 1.0000 | -0.0800 |
| magneto_slm_llm | 0.8511 | 1.0000 | -0.1489 |

## Stage: norm - per member

| member | macro_f1 | macro_f1_baseline | macro_f1_delta |
|---|---|---|---|
| country_iso | 0.1538 | 0.1340 | 0.0199 |
| date_iso | 0.5547 | 0.6541 | -0.0994 |
| llm_canonicalize | 0.0981 | 0.1108 | -0.0127 |
| number_locale | 0.1466 | 0.6506 | -0.5040 |
| taxonomy_lookup | 0.1758 | 0.2846 | -0.1087 |
| text_clean | 0.7761 | 0.7683 | 0.0079 |

## Stage: em_blocking - per member

| member | pair_recall | pair_recall_baseline | pair_recall_delta | reduction_ratio | reduction_ratio_baseline |
|---|---|---|---|---|---|
| bm25_blocker | 0.5756 | 0.9951 | -0.4195 | 0.9961 | 0.9964 |
| embedding_blocker | 0.5777 | 0.9952 | -0.4175 | 0.9961 | 0.9964 |
| sc_block | 0.5902 | 0.9993 | -0.4091 | 0.9961 | 0.9964 |
| sorted_neighbourhood_blocker | 0.4633 | 0.8540 | -0.3907 | 0.9958 | 0.9959 |
| standard_blocker | 0.4709 | 0.8639 | -0.3930 | 0.9861 | 0.9858 |
| token_blocker | 0.5812 | 0.9991 | -0.4179 | 0.9097 | 0.9063 |

## Stage: em_matching - per member

| member | f1 | f1_baseline | f1_delta | f1_vs_val | f1_vs_val_baseline | f1_vs_test | f1_vs_test_baseline |
|---|---|---|---|---|---|---|---|
| comem | 0.5191 | 0.5363 | -0.0173 | nan | nan | 0.5191 | 0.5363 |
| ditto_plm | 0.8745 | 0.9756 | -0.1011 | 0.7306 | 0.8667 | 0.8745 | 0.9756 |
| llm_matcher | 0.5320 | 0.5396 | -0.0076 | nan | nan | 0.5320 | 0.5396 |
| magellan | 0.7950 | 0.9480 | -0.1530 | 0.8313 | 0.9482 | 0.7950 | 0.9480 |

## Stage: em - per pair

| pair | member | f1 | f1_baseline | f1_delta |
|---|---|---|---|---|
| musicbrainz_discogs | comem | 0.5616 | 0.7091 | -0.1475 |
| musicbrainz_discogs | ditto_plm | 0.9006 | 0.9780 | -0.0774 |
| musicbrainz_discogs | llm_matcher | 0.5907 | 0.7115 | -0.1208 |
| musicbrainz_discogs | magellan | 0.7118 | 0.9156 | -0.2038 |
| musicbrainz_lastfm | comem | 0.4766 | 0.3636 | 0.1130 |
| musicbrainz_lastfm | ditto_plm | 0.8483 | 0.9731 | -0.1248 |
| musicbrainz_lastfm | llm_matcher | 0.4732 | 0.3676 | 0.1056 |
| musicbrainz_lastfm | magellan | 0.8781 | 0.9803 | -0.1022 |

## Stage: fusion - per member

| member | overall_accuracy | overall_accuracy_baseline | overall_accuracy_delta |
|---|---|---|---|
| artist_llm_judge | 0.5506 | 0.8169 | -0.2662 |
| artist_longest_string | 0.5390 | 0.8091 | -0.2701 |
| artist_prefer_higher_trust | 0.5455 | 0.8208 | -0.2753 |
| artist_truthfinder | 0.5442 | 0.8195 | -0.2753 |
| artist_voting | 0.5442 | 0.8208 | -0.2766 |
| duration_fusionquery | 0.5442 | 0.8260 | -0.2818 |
| duration_huber_m_estimator | 0.5455 | 0.8026 | -0.2571 |
| duration_maximum | 0.5481 | 0.8364 | -0.2883 |
| duration_median | 0.5455 | 0.8169 | -0.2714 |
| duration_median_of_means | 0.5455 | 0.8026 | -0.2571 |
| duration_prefer_higher_trust | 0.5442 | 0.8195 | -0.2753 |
| duration_trimmed_mean | 0.5455 | 0.8026 | -0.2571 |
| genre_accusim | 0.5442 | 0.8195 | -0.2753 |
| genre_most_complete | 0.5442 | 0.8182 | -0.2740 |
| genre_prefer_higher_trust | 0.5442 | 0.8208 | -0.2766 |
| genre_voting | 0.5442 | 0.8208 | -0.2766 |
| label_longest_string | 0.5429 | 0.8234 | -0.2805 |
| label_ltm | 0.5403 | 0.8195 | -0.2792 |
| label_prefer_higher_trust | 0.5442 | 0.8208 | -0.2766 |
| label_voting | 0.5442 | 0.8208 | -0.2766 |
| name_accusim | 0.5442 | 0.8221 | -0.2779 |
| name_casefusion | 0.5416 | 0.8195 | -0.2779 |
| name_fusionquery | 0.5403 | 0.8039 | -0.2636 |
| name_llm_judge | 0.5403 | 0.8117 | -0.2714 |
| name_longest_string | 0.5273 | 0.8000 | -0.2727 |
| name_most_complete | 0.5273 | 0.8000 | -0.2727 |
| name_prefer_higher_trust | 0.5416 | 0.8195 | -0.2779 |
| name_voting | 0.5442 | 0.8208 | -0.2766 |
| release-country_llm_judge | 0.5688 | 0.8299 | -0.2610 |
| release-country_most_complete | 0.5675 | 0.8273 | -0.2597 |
| release-country_prefer_higher_trust | 0.5481 | 0.8325 | -0.2844 |
| release-country_truthfinder | 0.5623 | 0.8286 | -0.2662 |
| release-country_voting | 0.5442 | 0.8208 | -0.2766 |
| release-date_earliest | 0.5416 | 0.8169 | -0.2753 |
| release-date_prefer_higher_trust | 0.5416 | 0.8195 | -0.2779 |
| release-date_voting | 0.5442 | 0.8208 | -0.2766 |
| tracks_intersection | 0.5442 | 0.8000 | -0.2558 |
| tracks_intersection_k_sources | 0.5442 | 0.8182 | -0.2740 |
| tracks_ltm | 0.5442 | 0.8143 | -0.2701 |
| tracks_prefer_higher_trust | 0.5442 | 0.8208 | -0.2766 |
| tracks_union | 0.5442 | 0.8273 | -0.2831 |
| tracks_voting | 0.5442 | 0.8208 | -0.2766 |

## Stage: fusion - per attribute

| attribute | best_accuracy | baseline | delta | spread | spread_baseline | spread_delta |
|---|---|---|---|---|---|---|
| artist | 0.9200 | 0.9300 | -0.0100 | 0.0900 | 0.0900 | 0.0000 |
| duration | 0.1300 | 0.4200 | -0.2900 | 0.0300 | 0.2600 | -0.2300 |
| genre | 0.4286 | 0.8831 | -0.4545 | 0.0000 | 0.0260 | -0.0260 |
| label | 0.4516 | 0.8817 | -0.4301 | 0.0323 | 0.0323 | 0.0000 |
| name | 0.8400 | 0.9000 | -0.0600 | 0.1300 | 0.1700 | -0.0400 |
| release-country | 0.9000 | 0.9700 | -0.0700 | 0.1900 | 0.0900 | 0.1000 |
| release-date | 0.9000 | 0.9200 | -0.0200 | 0.0200 | 0.0300 | -0.0100 |
| tracks | 0.0200 | 0.9700 | -0.9500 | 0.0000 | 0.2100 | -0.2100 |
