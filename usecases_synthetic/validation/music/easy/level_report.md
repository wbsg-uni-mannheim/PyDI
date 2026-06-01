# Validation report - music / easy

_Generated at 2026-05-16T09:44:00.259486+00:00_

- domain: `music`
- level: `easy`
- with_llm: `True`
- committee_versions: sm=`sm_committee.yaml@cb4a6847ac9e`, norm=`normalization_committee_music.yaml@c06387b2e5ef`, em_blocking=`em_blocking_committee_music.yaml@1eed2384fe7e`, em_matching=`em_matching_committee_music.yaml@fef75de2e979`, fusion=`fusion_committee_music.yaml@4a03339ecdef`

## Stage summary

| stage | metric | measured | baseline | delta |
|---|---|---|---|---|
| sm | macro_f1 | 0.8925 | 0.8761 | 0.0165 |
| norm | macro_f1 | 0.3936 | 0.4337 | -0.0401 |
| em_blocking | macro_pair_recall | 0.9703 | 0.9511 | 0.0192 |
| em_matching | macro_f1_vs_test | 0.8978 | 0.7499 | 0.1479 |
| fusion | overall_accuracy | 0.8394 | 0.8594 | -0.0200 |

## Stage: sm - per member

| member | f1 | f1_baseline | f1_delta |
|---|---|---|---|
| coma_hybrid | 1.0000 | 1.0000 | 0.0000 |
| duplicate_majority | 0.7442 | 0.7442 | 0.0000 |
| embedding_sbert | 0.8235 | 0.7500 | 0.0735 |
| instance_tf_cosine | 0.6800 | 0.6383 | 0.0417 |
| label_jw | 1.0000 | 1.0000 | 0.0000 |
| llm_openai | 1.0000 | 1.0000 | 0.0000 |
| magneto_slm_llm | 1.0000 | 1.0000 | 0.0000 |

## Stage: norm - per member

| member | macro_f1 | macro_f1_baseline | macro_f1_delta |
|---|---|---|---|
| country_iso | 0.1383 | 0.1340 | 0.0044 |
| date_iso | 0.6557 | 0.6541 | 0.0016 |
| llm_canonicalize | 0.1097 | 0.1108 | -0.0011 |
| number_locale | 0.3991 | 0.6506 | -0.2514 |
| taxonomy_lookup | 0.2846 | 0.2846 | 0.0000 |
| text_clean | 0.7742 | 0.7683 | 0.0060 |

## Stage: em_blocking - per member

| member | pair_recall | pair_recall_baseline | pair_recall_delta | reduction_ratio | reduction_ratio_baseline |
|---|---|---|---|---|---|
| bm25_blocker | 0.9962 | 0.9951 | 0.0011 | 0.9967 | 0.9964 |
| embedding_blocker | 0.9962 | 0.9952 | 0.0010 | 0.9967 | 0.9964 |
| sc_block | 0.9993 | 0.9993 | 0.0000 | 0.9967 | 0.9964 |
| sorted_neighbourhood_blocker | 0.9105 | 0.8540 | 0.0566 | 0.9961 | 0.9959 |
| standard_blocker | 0.9205 | 0.8639 | 0.0566 | 0.9848 | 0.9858 |
| token_blocker | 0.9990 | 0.9991 | -0.0002 | 0.9075 | 0.9063 |

## Stage: em_matching - per member

| member | f1 | f1_baseline | f1_delta | f1_vs_val | f1_vs_val_baseline | f1_vs_test | f1_vs_test_baseline |
|---|---|---|---|---|---|---|---|
| comem | 0.8172 | 0.5363 | 0.2808 | nan | nan | 0.8172 | 0.5363 |
| ditto_plm | 0.9756 | 0.9756 | 0.0000 | 0.8667 | 0.8667 | 0.9756 | 0.9756 |
| llm_matcher | 0.8471 | 0.5396 | 0.3075 | nan | nan | 0.8471 | 0.5396 |
| magellan | 0.9513 | 0.9480 | 0.0034 | 0.9590 | 0.9482 | 0.9513 | 0.9480 |

## Stage: em - per pair

| pair | member | f1 | f1_baseline | f1_delta |
|---|---|---|---|---|
| musicbrainz_discogs | comem | 0.8394 | 0.7091 | 0.1303 |
| musicbrainz_discogs | ditto_plm | 0.9780 | 0.9780 | 0.0000 |
| musicbrainz_discogs | llm_matcher | 0.8758 | 0.7115 | 0.1643 |
| musicbrainz_discogs | magellan | 0.9208 | 0.9156 | 0.0053 |
| musicbrainz_lastfm | comem | 0.7950 | 0.3636 | 0.4313 |
| musicbrainz_lastfm | ditto_plm | 0.9731 | 0.9731 | 0.0000 |
| musicbrainz_lastfm | llm_matcher | 0.8183 | 0.3676 | 0.4507 |
| musicbrainz_lastfm | magellan | 0.9818 | 0.9803 | 0.0015 |

## Stage: fusion - per member

| member | overall_accuracy | overall_accuracy_baseline | overall_accuracy_delta |
|---|---|---|---|
| artist_llm_judge | 0.8000 | 0.8169 | -0.0169 |
| artist_longest_string | 0.7922 | 0.8091 | -0.0169 |
| artist_prefer_higher_trust | 0.7987 | 0.8208 | -0.0221 |
| artist_truthfinder | 0.7987 | 0.8195 | -0.0208 |
| artist_voting | 0.8000 | 0.8208 | -0.0208 |
| duration_fusionquery | 0.7922 | 0.8260 | -0.0338 |
| duration_huber_m_estimator | 0.8169 | 0.8026 | 0.0143 |
| duration_maximum | 0.8169 | 0.8364 | -0.0195 |
| duration_median | 0.8169 | 0.8169 | 0.0000 |
| duration_median_of_means | 0.8169 | 0.8026 | 0.0143 |
| duration_prefer_higher_trust | 0.8026 | 0.8195 | -0.0169 |
| duration_trimmed_mean | 0.8169 | 0.8026 | 0.0143 |
| genre_accusim | 0.7987 | 0.8195 | -0.0208 |
| genre_most_complete | 0.7974 | 0.8182 | -0.0208 |
| genre_prefer_higher_trust | 0.8000 | 0.8208 | -0.0208 |
| genre_voting | 0.8000 | 0.8208 | -0.0208 |
| label_longest_string | 0.8026 | 0.8234 | -0.0208 |
| label_ltm | 0.7987 | 0.8195 | -0.0208 |
| label_prefer_higher_trust | 0.8000 | 0.8208 | -0.0208 |
| label_voting | 0.8000 | 0.8208 | -0.0208 |
| name_accusim | 0.8000 | 0.8221 | -0.0221 |
| name_casefusion | 0.7987 | 0.8195 | -0.0208 |
| name_fusionquery | 0.7844 | 0.8039 | -0.0195 |
| name_llm_judge | 0.7909 | 0.8117 | -0.0208 |
| name_longest_string | 0.7818 | 0.8000 | -0.0182 |
| name_most_complete | 0.7818 | 0.8000 | -0.0182 |
| name_prefer_higher_trust | 0.7987 | 0.8195 | -0.0208 |
| name_voting | 0.8000 | 0.8208 | -0.0208 |
| release-country_llm_judge | 0.8104 | 0.8299 | -0.0195 |
| release-country_most_complete | 0.8104 | 0.8273 | -0.0169 |
| release-country_prefer_higher_trust | 0.8078 | 0.8325 | -0.0247 |
| release-country_truthfinder | 0.8104 | 0.8286 | -0.0182 |
| release-country_voting | 0.8000 | 0.8208 | -0.0208 |
| release-date_earliest | 0.7948 | 0.8169 | -0.0221 |
| release-date_prefer_higher_trust | 0.7974 | 0.8195 | -0.0221 |
| release-date_voting | 0.8000 | 0.8208 | -0.0208 |
| tracks_intersection | 0.7818 | 0.8000 | -0.0182 |
| tracks_intersection_k_sources | 0.7987 | 0.8182 | -0.0195 |
| tracks_ltm | 0.7987 | 0.8143 | -0.0156 |
| tracks_prefer_higher_trust | 0.8000 | 0.8208 | -0.0208 |
| tracks_union | 0.8078 | 0.8273 | -0.0195 |
| tracks_voting | 0.8000 | 0.8208 | -0.0208 |

## Stage: fusion - per attribute

| attribute | best_accuracy | baseline | delta | spread | spread_baseline | spread_delta |
|---|---|---|---|---|---|---|
| artist | 0.9000 | 0.9300 | -0.0300 | 0.0600 | 0.0900 | -0.0300 |
| duration | 0.2900 | 0.4200 | -0.1300 | 0.1900 | 0.2600 | -0.0700 |
| genre | 0.8831 | 0.8831 | 0.0000 | 0.0260 | 0.0260 | 0.0000 |
| label | 0.8817 | 0.8817 | 0.0000 | 0.0323 | 0.0323 | 0.0000 |
| name | 0.9300 | 0.9000 | 0.0300 | 0.1400 | 0.1700 | -0.0300 |
| release-country | 0.9400 | 0.9700 | -0.0300 | 0.0800 | 0.0900 | -0.0100 |
| release-date | 0.9300 | 0.9200 | 0.0100 | 0.0400 | 0.0300 | 0.0100 |
| tracks | 0.9600 | 0.9700 | -0.0100 | 0.2000 | 0.2100 | -0.0100 |
